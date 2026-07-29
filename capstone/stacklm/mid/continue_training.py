"""Mid-training (Ch. 14.8): the phase between raw pretraining and post-training.

Three moves, one continuous WSD decay: (1) the decay leg runs on a higher-quality
mix (annealing); (2) long-context extension continues training at a longer
`seq_len` with NTK-aware RoPE base rescaling; (3) capability injection concentrates
math/code at the LR floor. `extend_context` mutates the model's RoPE cache in place;
`run_mid_training` walks the three sub-phases WITHOUT resetting the LR schedule at a
boundary (that is the whole point -- see Ch. 14.8).
"""
from dataclasses import dataclass, field

import torch

from ..model.rope import ntk_rescaled_base
from ..optim import build_optimizers
from ..optim.schedule import wsd_decay_multiplier
from ..train.loop import autocast_ctx


def _unwrap(model):
    """`torch.compile` wraps the module; reach the real one for cfg/buffer surgery."""
    return model._orig_mod if hasattr(model, "_orig_mod") else model


@torch.no_grad()
def extend_context(model, new_seq_len: int, device="cpu") -> float:
    """Rescale RoPE base for a longer context and rebuild the cache. Returns the
    new theta. NoPE layers are unaffected (they never consult the cache)."""
    m = _unwrap(model)
    cfg = m.cfg
    new_base = ntk_rescaled_base(cfg.rope_theta, cfg.head_dim,
                                 old_len=cfg.max_seq_len, new_len=new_seq_len)
    m.rebuild_rope(new_seq_len, new_base, device=device)
    return new_base


def mid_train(model, dataset, *, device="cpu", steps=10, micro_batch_size=4,
              grad_accum=2, peak_lr=3e-3, muon_lr=None, adamw_lr=None,
              grad_clip=1.0, extend_to=None, use_seq_ids=True, log_every=1,
              seed=1234):
    """Continued-training entry: one continuous WSD decay across the mid-train
    budget, optionally extending context once at the start.

    `muon_lr` / `adamw_lr` set the two groups' peaks independently (Ch. 14.6 fixes
    6e-3 and 3e-3 -- a 2:1 ratio, not an order of magnitude: Muon's RMS-matched
    update already lives on Adam's scale). When omitted they fall back to
    `peak_lr` and `peak_lr / 2`.
    """
    torch.manual_seed(seed)
    device = torch.device(device)
    model.to(device).train()

    if extend_to is not None:
        new_base = extend_context(model, extend_to, device=device)
        print(f"  [mid] RoPE base rescaled -> {new_base:.0f}, seq_len -> {extend_to}")

    muon_peak = peak_lr if muon_lr is None else muon_lr
    adamw_peak = (peak_lr / 2) if adamw_lr is None else adamw_lr
    muon, adamw = build_optimizers(model, muon_lr=muon_peak, adamw_lr=adamw_peak)
    optimizers = [muon, adamw]
    peaks = {id(muon): muon_peak, id(adamw): adamw_peak}

    loader = torch.utils.data.DataLoader(dataset, batch_size=micro_batch_size,
                                         shuffle=True, drop_last=True)

    def infinite():
        while True:
            for b in loader:
                yield b
    it = infinite()

    history = []
    for step in range(steps):
        mult = wsd_decay_multiplier(step, steps)        # continuous decay, 1 -> 0
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = peaks[id(opt)] * mult
            opt.zero_grad(set_to_none=True)

        loss_acc = 0.0
        for _ in range(grad_accum):
            batch = {k: v.to(device) for k, v in next(it).items()}
            with autocast_ctx(device):
                # position_ids MUST be threaded: without them the model falls back
                # to a contiguous arange, so per-document position resets vanish.
                _, loss = model(batch["input_ids"], targets=batch["targets"],
                                position_ids=batch.get("position_ids"),
                                seq_ids=batch["seq_ids"] if use_seq_ids else None)
                loss = loss / grad_accum
            loss.backward()
            loss_acc += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        for opt in optimizers:
            opt.step()
        history.append(loss_acc)
        if log_every and step % log_every == 0:
            print(f"  [mid] step {step:>3} loss {loss_acc:.4f} "
                  f"lr_muon {peaks[id(muon)] * mult:.2e}")
    return {"loss_history": history}


# --------------------------------------------------------------------------- #
# The three-sub-phase driver: broad -> long -> narrow, one continuous decay.   #
# --------------------------------------------------------------------------- #

@dataclass
class SubPhase:
    """One mid-training sub-phase. `steps` overrides the token-budget arithmetic
    (used by CI, which runs a handful of steps instead of ~1e9 tokens)."""
    name: str
    tokens: int
    seq_len: int
    mix: dict = field(default_factory=dict)
    extend_now: bool = False
    steps: int = 0                 # 0 => derive from `tokens`


def steps_for(sub: SubPhase, global_batch_tokens: int) -> int:
    return sub.steps or (sub.tokens // global_batch_tokens)


def run_mid_training(model, phases, loader_fn, *, device="cpu",
                     global_batch_tokens=524_288, micro_batch_tokens=65_536,
                     muon_lr=6e-3, adamw_lr=3e-3, grad_clip=1.0,
                     optimizers=None, use_seq_ids=True, log_every=50, seed=1234,
                     checkpoint_fn=None):
    """Walk `phases` back to back under ONE WSD decay leg.

    `loader_fn(sub, micro_batch_size)` returns an *infinite* iterator of batch
    dicts (`input_ids`, `targets`, `position_ids`, `seq_ids`) at `sub.seq_len`
    -- all four are consumed by the model; dropping `position_ids` silently
    reverts RoPE to contiguous positions. In the real run
    that is a weighted mixture over the sub-phase's sources (Ch. 14.8); in CI it
    is a toy `PackedMemmapDataset`. `checkpoint_fn(name, model, optimizers, step)`
    is called at every sub-phase boundary.

    Pass `optimizers=[muon, adamw]` to CONTINUE the stable phase's optimizer state
    (Muon momentum, AdamW moments) restored from `ckpt_stable.pt`; leave it None
    and fresh optimizers are built at the given peaks.
    """
    torch.manual_seed(seed)
    device = torch.device(device)
    model.to(device).train()

    if optimizers is None:
        muon, adamw = build_optimizers(model, muon_lr=muon_lr, adamw_lr=adamw_lr)
        optimizers = [muon, adamw]
    else:
        muon, adamw = optimizers[0], optimizers[-1]
    peaks = {id(muon): muon_lr, id(adamw): adamw_lr}

    total_decay_steps = sum(steps_for(p, global_batch_tokens) for p in phases)
    mid_step, history = 0, []

    for sub in phases:
        if sub.extend_now:
            new_base = extend_context(model, sub.seq_len, device=device)
            print(f"  [{sub.name}] RoPE base -> {new_base:.0f}, seq_len -> {sub.seq_len}")

        micro_bs = max(1, micro_batch_tokens // sub.seq_len)
        accum = max(1, (global_batch_tokens // sub.seq_len) // micro_bs)
        it = loader_fn(sub, micro_bs)

        for _ in range(steps_for(sub, global_batch_tokens)):
            mult = wsd_decay_multiplier(mid_step, total_decay_steps)
            for opt in optimizers:
                for g in opt.param_groups:
                    g["lr"] = peaks[id(opt)] * mult
                opt.zero_grad(set_to_none=True)

            loss_acc = 0.0
            for _micro in range(accum):
                batch = {k: v.to(device) for k, v in next(it).items()}
                with autocast_ctx(device):
                    _, loss = model(batch["input_ids"], targets=batch["targets"],
                                    position_ids=batch.get("position_ids"),
                                    seq_ids=batch["seq_ids"] if use_seq_ids else None)
                    loss = loss / accum
                loss.backward()
                loss_acc += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            for opt in optimizers:
                opt.step()
            history.append(loss_acc)
            if log_every and mid_step % log_every == 0:
                print(f"  [{sub.name}] mid_step {mid_step:>5} loss {loss_acc:.4f} "
                      f"lr_muon {peaks[id(muon)] * mult:.2e} seq {sub.seq_len}")
            mid_step += 1

        if checkpoint_fn is not None:
            checkpoint_fn(sub.name, model, optimizers, mid_step)

    return {"loss_history": history, "mid_steps": mid_step,
            "total_decay_steps": total_decay_steps}
