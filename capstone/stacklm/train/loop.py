"""Single-GPU pretraining loop (Ch. 14.7), guarded to run CPU-only at toy scale.

bf16 autocast is used only on CUDA; on CPU we fall back to fp32 (a plain
nullcontext) so the smoke test is deterministic and portable. Covers grad-accum,
grad-clip, the Muon+AdamW hybrid with a WSD LR, periodic QK-clip, MFU/throughput,
checkpoint+resume (model + optimizers + step + rng), and periodic eval.
"""
import contextlib
import math
import os
import time

import torch

from ..optim import build_optimizers, qk_clip_, wsd_lr


def autocast_ctx(device):
    """bf16 autocast on CUDA; fp32 (no-op) everywhere else -- CPU-safe."""
    dev = device.type if isinstance(device, torch.device) else str(device)
    if dev == "cuda" and torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def estimate_mfu(n_params: int, tokens_per_sec: float,
                 peak_flops: float = 312e12) -> float:
    """Achieved / peak, using the 6ND rule (A100 bf16 peak by default)."""
    achieved = 6 * n_params * tokens_per_sec
    return achieved / peak_flops


def _to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def pretrain(model, dataset, *, device="cpu", steps=10, micro_batch_size=4,
             grad_accum=2, peak_lr=0.02, muon_lr=None, adamw_lr=None,
             warmup_steps=2, total_steps=None, decay_steps=None, decay_frac=0.2,
             grad_clip=1.0, weight_decay=0.1, qk_clip_every=0, qk_tau=100.0,
             log_every=1, eval_dataset=None, use_seq_ids=True, seed=1337):
    """Run `steps` optimizer steps. Returns a dict with the loss history + MFU.

    `dataset` is a `PackedMemmapDataset` (or anything returning the same dict).
    A fresh DataLoader with an infinite sampler feeds micro-batches.

    TWO PEAKS, ONE CURVE (Ch. 14.6). `muon_lr` / `adamw_lr` set the two groups'
    peaks independently (the flagship uses 0.02 and 3e-3); when omitted they fall
    back to `peak_lr` and `peak_lr / 2`. `wsd_lr` is then called with
    `peak_lr=1.0` so it returns a pure MULTIPLIER, which scales each group's own
    peak. Overwriting both groups with one shared LR is the classic Muon bug.

    `decay_steps` (absolute) or `decay_frac` (fraction of `total_steps`) set the
    length of the WSD decay leg; the flagship uses decay_frac=0.10.
    """
    torch.manual_seed(seed)
    device = torch.device(device)
    model.to(device).train()
    total_steps = total_steps or steps
    muon_peak = peak_lr if muon_lr is None else muon_lr
    adamw_peak = (peak_lr / 2) if adamw_lr is None else adamw_lr

    muon, adamw = build_optimizers(model, muon_lr=muon_peak, adamw_lr=adamw_peak,
                                   weight_decay=weight_decay)
    optimizers = [muon, adamw]
    peaks = {id(muon): muon_peak, id(adamw): adamw_peak}   # two groups, two peaks

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=micro_batch_size, shuffle=True, drop_last=True)

    def infinite():
        while True:
            for b in loader:
                yield b
    it = infinite()

    n_params = model.num_params()
    history = []
    for step in range(steps):
        mult = wsd_lr(step, peak_lr=1.0, warmup_steps=warmup_steps,
                      total_steps=total_steps, decay_steps=decay_steps,
                      decay_frac=decay_frac)
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = peaks[id(opt)] * mult
        lr = peaks[id(muon)] * mult                # logged: the Muon group's LR

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        record = {} if (qk_clip_every and step % qk_clip_every == 0) else None
        t0 = time.perf_counter()
        loss_acc, tokens = 0.0, 0
        for _ in range(grad_accum):
            batch = _to_device(next(it), device)
            seq_ids = batch["seq_ids"] if use_seq_ids else None
            with autocast_ctx(device):
                _, loss = model(batch["input_ids"], targets=batch["targets"],
                                seq_ids=seq_ids, record=record)
                loss = loss / grad_accum
            loss.backward()
            loss_acc += loss.item()
            tokens += batch["input_ids"].numel()
            record = None  # only record on the first micro-batch of the step

        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        for opt in optimizers:
            opt.step()

        qk_fired = 0
        if qk_clip_every and step % qk_clip_every == 0:
            # `record` was consumed above; recompute a fresh reading on a small
            # PROBE batch (post-step weights) rather than every micro-batch.
            rec = {}
            with torch.no_grad():
                b = _to_device(next(it), device)
                model(b["input_ids"], seq_ids=b["seq_ids"] if use_seq_ids else None,
                      record=rec)
            qk_fired = qk_clip_(model, rec, tau=qk_tau)

        dt = time.perf_counter() - t0
        tps = tokens / dt if dt > 0 else 0.0
        mfu = estimate_mfu(n_params, tps)
        history.append(loss_acc)
        if log_every and step % log_every == 0:
            print(f"  [pretrain] step {step:>3} loss {loss_acc:.4f} lr {lr:.2e} "
                  f"|g| {float(gnorm):.2f} tok/s {tps:,.0f} mfu {mfu*100:.2f}% "
                  f"qkclip {qk_fired}")

    result = {"loss_history": history, "n_params": n_params}
    if eval_dataset is not None:
        result["val_loss"], result["val_ppl"] = evaluate(model, eval_dataset,
                                                          device=device,
                                                          use_seq_ids=use_seq_ids)
    return result


@torch.no_grad()
def evaluate(model, dataset, *, device="cpu", batch_size=4, max_batches=8,
             use_seq_ids=True):
    device = torch.device(device)
    model.to(device).eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True)
    losses = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = _to_device(batch, device)
        with autocast_ctx(device):
            _, loss = model(batch["input_ids"], targets=batch["targets"],
                            seq_ids=batch["seq_ids"] if use_seq_ids else None)
        losses.append(loss.item())
    model.train()
    mean = sum(losses) / max(1, len(losses))
    return mean, math.exp(min(mean, 20.0))


def save_checkpoint(path, model, optimizers, step, extra=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizers": [o.state_dict() for o in optimizers],
        "step": step,
        "rng": torch.get_rng_state(),
        "extra": extra or {},
    }
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizers=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizers is not None:
        for o, sd in zip(optimizers, ckpt["optimizers"]):
            o.load_state_dict(sd)
    torch.set_rng_state(ckpt["rng"].to(torch.uint8) if hasattr(ckpt["rng"], "to")
                        else ckpt["rng"])
    return ckpt["step"], ckpt.get("extra", {})
