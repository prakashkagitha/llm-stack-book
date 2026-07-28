"""Mid-training (Ch. 14.8): the phase between raw pretraining and post-training.

Two moves: (1) the WSD *decay* phase runs on a higher-quality mix (annealing);
(2) long-context extension continues training at a longer `seq_len` with NTK-aware
RoPE base rescaling. `extend_context` mutates the model's RoPE cache in place.
"""
import math

import torch

from ..model.rope import ntk_rescaled_base
from ..optim import build_optimizers
from ..optim.schedule import wsd_decay_multiplier
from ..train.loop import autocast_ctx


@torch.no_grad()
def extend_context(model, new_seq_len: int, device="cpu") -> float:
    """Rescale RoPE base for a longer context and rebuild the cache. Returns the
    new theta. NoPE layers are unaffected (they never consult the cache)."""
    cfg = model.cfg
    new_base = ntk_rescaled_base(cfg.rope_theta, cfg.head_dim,
                                 old_len=cfg.max_seq_len, new_len=new_seq_len)
    model.rebuild_rope(new_seq_len, new_base, device=device)
    return new_base


def mid_train(model, dataset, *, device="cpu", steps=10, micro_batch_size=4,
              grad_accum=2, peak_lr=3e-3, grad_clip=1.0, extend_to=None,
              use_seq_ids=True, log_every=1, seed=1234):
    """Continued-training entry: one continuous WSD decay across the mid-train
    budget, optionally extending context once at the start."""
    torch.manual_seed(seed)
    device = torch.device(device)
    model.to(device).train()

    if extend_to is not None:
        new_base = extend_context(model, extend_to, device=device)
        print(f"  [mid] RoPE base rescaled -> {new_base:.0f}, seq_len -> {extend_to}")

    muon, adamw = build_optimizers(model, muon_lr=peak_lr, adamw_lr=peak_lr / 2)
    optimizers = [muon, adamw]

    loader = torch.utils.data.DataLoader(dataset, batch_size=micro_batch_size,
                                         shuffle=True, drop_last=True)

    def infinite():
        while True:
            for b in loader:
                yield b
    it = infinite()

    history = []
    for step in range(steps):
        lr = peak_lr * wsd_decay_multiplier(step, steps)   # continuous decay
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)

        loss_acc = 0.0
        for _ in range(grad_accum):
            batch = {k: v.to(device) for k, v in next(it).items()}
            with autocast_ctx(device):
                _, loss = model(batch["input_ids"], targets=batch["targets"],
                                seq_ids=batch["seq_ids"] if use_seq_ids else None)
                loss = loss / grad_accum
            loss.backward()
            loss_acc += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        for opt in optimizers:
            opt.step()
        history.append(loss_acc)
        if log_every and step % log_every == 0:
            print(f"  [mid] step {step:>3} loss {loss_acc:.4f} lr {lr:.2e}")
    return {"loss_history": history}
