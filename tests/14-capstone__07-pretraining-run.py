"""
Runs the CPU-runnable Python code blocks from:
    content/14-capstone/07-pretraining-run.md

This chapter wires together components owned by EARLIER capstone chapters
(Stack100M the model, PackedMemmapDataset the loader, wsd_lr/qk_clip_/
build_optimizers the optimizer) that are only specified here as *contracts*
("you will not find their bodies here" -- the chapter's own words). Those
bodies are therefore GLUE in this file: small, honestly-labelled stand-ins
that satisfy the documented contract, defined right before the first tested
block that needs them, and clearly marked as such. Every block actually
TESTED below is copied verbatim from the chapter and then exercised with
tiny fixtures.

Tested blocks (chapter numbering):
    #5  (line ~387)  -- Stack100M.forward, the loss_chunk==0 (unchunked) branch
    #6  (line ~404)  -- fused_ce_z_loss / _chunk_ce (the chunked loss head)
    #7  (line ~446)  -- the two-line branch that selects the chunked path
    #8  (line ~554)  -- all_params_of / attach_base_lrs / set_lr (WSD x 2 optimizers)
    #9  (line ~610)  -- maybe_qk_clip (MuonClip on a schedule, fixed probe batch)
    #10 (line ~651)  -- the "inside the main loop, once per optimizer step" snippet
    #11 (line ~682)  -- FlexAttention Option A: make_doc_block_mask
    #13 (line ~734)  -- CheckpointedBlock / enable_activation_checkpointing
    #15 (line ~916)  -- ResumableShuffleSampler / make_collate / build_loader
    #16 (line ~979)  -- the resume snippet (samples_per_step, start_sample)
    #17 (line ~1017) -- the non-finite-gradient skip guard
    #18 (line ~1104) -- flops_per_token / utilization (MFU/HFU, the 6ND+attention rule)

Skipped blocks (per the assignment's heuristic classification):
    #0  non-python (the ASCII data-flow diagram)
    #1  fragment    (StackConfig/Stack100M/PackedMemmapDataset/optim/tokenizer
                      *contracts* -- dataclass/class bodies of `...`, not runnable)
    #2  needs-gpu   (TrainConfig: `device: str = "cuda"` trips the GPU heuristic;
                      the dataclass itself is inert, but nothing here calls it)
    #3  needs-gpu   (`torch.autocast(device_type="cuda", ...)` module-level ctx)
    #4  needs-gpu   (`accumulate`: hard-codes `torch.autocast("cuda", ...)`.
                      A CPU-safe GLUE stand-in is defined below so block #10,
                      which CALLS accumulate, can still execute -- see the
                      comment at its definition.)
    #12 needs-gpu   (varlen FlashAttention: `from flash_attn import ...`, needs GPU)
    #14 needs-gpu   (checkpoint.py: `torch.cuda.get_rng_state_all`, `map_location="cuda"`)
    #19 needs-gpu   (`torch.profiler` with `ProfilerActivity.CUDA`)
    #20 needs-gpu   (`evaluate`/`sample_text`/`log_metrics`: `cfg.device="cuda"`,
                      `torch.autocast("cuda", ...)`)
    #21 needs-gpu   (the full `train.py` `main()`: `.to(cfg.device="cuda")`, etc.)
    #22,24 needs-gpu (RTX4090/T4 config + DDP/FSDP scale-out notes, later in the
                      chapter -- GPU launch scripts)
    #23,25 fragment  (later config/CLI fragments, not standalone)

No network access and no optional third-party imports: only torch and the
Python standard library, both in the guaranteed CI list.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset, Sampler, default_collate


# =====================================================================
# Shared GLUE: a minimal TrainConfig/StackConfig stand-in.
# Block #2 (the real TrainConfig dataclass) is SKIPPED as needs-gpu, so this
# is NOT that block -- it is just enough of its *shape* (cfg.model.n_layers,
# cfg.warmup_steps, ...) for the tested blocks below to have something to
# read attributes off of.
# =====================================================================


@dataclass
class ModelCfg:
    n_layers: int = 4
    max_seq_len: int = 8
    d_model: int = 16


@dataclass
class Cfg:
    model: ModelCfg = field(default_factory=ModelCfg)
    shard_dir: str = ""
    micro_batch_size: int = 2
    grad_accum_steps: int = 3
    num_workers: int = 0
    seed: int = 1337
    warmup_steps: int = 5
    total_steps: int = 40
    decay_steps: int = 10
    grad_clip: float = 1.0
    qk_clip_tau: float = 30.0
    qk_clip_every: int = 4
    qk_probe_seqs: int = 2
    max_consecutive_skips: int = 3
    device: str = "cpu"


# =====================================================================
# Block #5 (chapter line ~387: Stack100M.forward, the loss_chunk == 0 branch)
# Copied verbatim; wrapped in a tiny function (`return loss` added) so it can
# actually be CALLED, since in the book it is an inline fragment of a larger
# method body, not a standalone unit.
# =====================================================================


def _unchunked_loss_branch(self, x, targets):
    logits = self._cap(self.lm_head(x))                            # (B, T, V) bf16   4.3 GB
    lf = logits.float()                                            # fp32 copy        8.6 GB
    ce = F.cross_entropy(lf.view(-1, lf.size(-1)),                 # + log_softmax    8.6 GB
                         targets.reshape(-1), ignore_index=-100)
    logz = torch.logsumexp(lf, dim=-1)                             # z-loss reuses lf, but
    loss = ce + self.cfg.z_loss_coef * (logz ** 2).mean()          # a naive version would copy again
    return loss


class _FakeModelHead:
    """GLUE: the minimal `self` that block #5/#7's fragments need -- an
    `lm_head`, a `_cap` (soft-cap, off by default), and `cfg.{z_loss_coef,
    logit_soft_cap,loss_chunk}`. Stands in for the parts of Ch. 14.4's
    Stack100M this chapter does not re-show."""

    def __init__(self, d_model, vocab_size, z_loss_coef, soft_cap=0.0, loss_chunk=0):
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.cfg = SimpleNamespace(z_loss_coef=z_loss_coef, logit_soft_cap=soft_cap,
                                   loss_chunk=loss_chunk)

    def _cap(self, logits):
        cap = self.cfg.logit_soft_cap
        return cap * torch.tanh(logits / cap) if cap > 0 else logits


torch.manual_seed(0)
_B, _T, _D, _V = 2, 5, 16, 32
_fake5 = _FakeModelHead(_D, _V, z_loss_coef=1e-4)
_x5 = torch.randn(_B, _T, _D)
_targets5 = torch.randint(0, _V, (_B, _T))
_loss5 = _unchunked_loss_branch(_fake5, _x5, _targets5)
assert _loss5.dim() == 0 and torch.isfinite(_loss5), "unchunked branch must return a finite scalar"
assert _loss5.item() > 0, "cross-entropy + non-negative z-loss must be positive"

print("[block #5 OK] unchunked loss_chunk==0 branch runs and returns a finite scalar loss.\n")


# =====================================================================
# Block #6 (chapter line ~404: stacklm/model/loss.py -- the chunked fused loss)
# Copied verbatim.
# =====================================================================


def _chunk_ce(h, w, t, cap: float):
    """One chunk -> (sum CE, sum lse^2, n_valid). Logits live only inside this
    function, so `checkpoint` frees them after the forward and rebuilds them one
    chunk at a time during backward."""
    logits = F.linear(h, w).float()                  # (C, V) fp32 -- transient
    if cap > 0:                                      # SAME soft cap as the inference path
        logits = cap * torch.tanh(logits / cap)
    lse = torch.logsumexp(logits, dim=-1)            # (C,) -- reused by BOTH losses
    valid = (t != -100)
    tgt = t.clamp_min(0).unsqueeze(-1)               # gather needs an in-range index
    ce = lse - logits.gather(-1, tgt).squeeze(-1)    # CE = logsumexp - logit[target]
    return (ce * valid).sum(), (lse.pow(2) * valid).sum(), valid.sum()


def fused_ce_z_loss(hidden, weight, targets, z_coef: float,
                    chunk: int = 8192, soft_cap: float = 0.0):
    """Returns (cross_entropy, z_loss), each a mean over VALID positions.
    Numerically equal to the unchunked path; peak logit memory is chunk*V and no
    longer grows with batch size. Gradients w.r.t. both `hidden` and the (tied)
    `weight` accumulate correctly because autograd sums every checkpointed call."""
    h = hidden.reshape(-1, hidden.shape[-1])         # (B*T, d)
    t = targets.reshape(-1)                          # (B*T,)
    ce = h.new_zeros((), dtype=torch.float32)
    z = h.new_zeros((), dtype=torch.float32)
    n = torch.zeros((), dtype=torch.long, device=h.device)
    for i in range(0, h.shape[0], chunk):
        a, b, c = checkpoint(_chunk_ce, h[i:i + chunk], weight, t[i:i + chunk],
                             soft_cap, use_reentrant=False)   # non-reentrant: compile-safe
        ce, z, n = ce + a, z + b, n + c
    n = n.clamp_min(1)
    return ce / n, z_coef * (z / n)


# --- exercise block #6: no ignored positions -> the docstring's "numerically
#     equal to the unchunked path" claim, checked against block #5's output.
_ce6, _z6 = fused_ce_z_loss(_x5, _fake5.lm_head.weight, _targets5,
                            z_coef=_fake5.cfg.z_loss_coef, chunk=3, soft_cap=0.0)
_loss6 = _ce6 + _z6
assert torch.allclose(_loss6, _loss5, atol=1e-4), (_loss6.item(), _loss5.item())

# Chunk size >= B*T (a single chunk) must reproduce the SAME numbers, since
# chunking only changes how many times the loop body runs, not the math.
_ce6b, _z6b = fused_ce_z_loss(_x5, _fake5.lm_head.weight, _targets5,
                              z_coef=_fake5.cfg.z_loss_coef, chunk=_B * _T, soft_cap=0.0)
assert torch.allclose(_ce6b + _z6b, _loss5, atol=1e-5)

# With -100 targets present, the chunked CE must match F.cross_entropy's own
# ignore_index handling (the mean-over-valid-positions claim).
_targets6 = _targets5.clone()
_targets6[0, 0] = -100
_targets6[1, 2] = -100
_ce_expected = F.cross_entropy(_fake5.lm_head(_x5).float().reshape(-1, _V),
                               _targets6.reshape(-1), ignore_index=-100)
_ce6c, _z6c = fused_ce_z_loss(_x5, _fake5.lm_head.weight, _targets6,
                              z_coef=_fake5.cfg.z_loss_coef, chunk=3, soft_cap=0.0)
assert torch.allclose(_ce6c, _ce_expected, atol=1e-4), (_ce6c.item(), _ce_expected.item())
assert torch.isfinite(_z6c) and _z6c.item() >= 0

print("[block #6 OK] fused_ce_z_loss matches the unchunked path and honours ignore_index=-100.\n")


# =====================================================================
# Block #7 (chapter line ~446: the two-line branch that selects the chunked path)
# Copied verbatim; wrapped in a function so it can be CALLED.
# =====================================================================


def _chunked_loss_branch(self, x, targets):
    if targets is not None and self.cfg.loss_chunk > 0:
        ce, zl = fused_ce_z_loss(x, self.lm_head.weight, targets, self.cfg.z_loss_coef,
                                 chunk=self.cfg.loss_chunk, soft_cap=self.cfg.logit_soft_cap)
        return None, ce + zl        # training path never materializes the logit tensor


_fake7 = _FakeModelHead(_D, _V, z_loss_coef=1e-4, loss_chunk=3)
_fake7.lm_head.load_state_dict(_fake5.lm_head.state_dict())  # same weights as block #5
_logits7, _loss7 = _chunked_loss_branch(_fake7, _x5, _targets5)
assert _logits7 is None, "the chunked training path must never materialize the logit tensor"
assert torch.allclose(_loss7, _loss5, atol=1e-4), (_loss7.item(), _loss5.item())
# loss_chunk == 0 must skip the branch entirely (returns None implicitly).
_fake7_off = _FakeModelHead(_D, _V, z_loss_coef=1e-4, loss_chunk=0)
assert _chunked_loss_branch(_fake7_off, _x5, _targets5) is None

print("[block #7 OK] the loss_chunk>0 branch matches the unchunked loss and hides the logits.\n")


# =====================================================================
# GLUE: a minimal wsd_lr, standing in for Ch. 14.6's frozen schedule (owned
# by an earlier chapter, not shown here). Matches the contract this chapter
# states: warmup ramps linearly to 1, stable holds at 1, decay (absolute
# `decay_steps`) ramps toward `final_frac`.
# =====================================================================


def wsd_lr(step: int, *, peak_lr: float, warmup_steps: int, total_steps: int,
           decay_steps: int | None = None, decay_frac: float = 0.2,
           final_frac: float = 0.0) -> float:
    if decay_steps is None:
        decay_steps = int(decay_frac * total_steps)
    decay_start = total_steps - decay_steps
    if step < warmup_steps:
        mult = (step + 1) / warmup_steps
    elif step < decay_start:
        mult = 1.0
    else:
        frac = min((step - decay_start) / max(1, decay_steps), 1.0)
        mult = 1.0 - (1.0 - final_frac) * frac
    return peak_lr * mult


# =====================================================================
# Block #8 (chapter line ~554: all_params_of / attach_base_lrs / set_lr)
# Copied verbatim (the `from stacklm.optim import wsd_lr` import is dropped;
# the GLUE `wsd_lr` above stands in for it, per the module docstring).
# =====================================================================


def all_params_of(optimizers):
    """Flat list of every trainable tensor across both optimizers -- the set the
    GLOBAL grad-norm clip must cover."""
    return [p for opt in optimizers for g in opt.param_groups for p in g["params"]]


def attach_base_lrs(optimizers):
    """Record each group's peak LR once, at build time: Muon 6e-3, AdamW 3e-3
    (Ch. 14.6). Only the *shape* of the WSD curve is shared, not the value."""
    for opt in optimizers:
        for g in opt.param_groups:
            g.setdefault("base_lr", g["lr"])


def set_lr(optimizers, step, cfg):
    """Apply this step's WSD multiplier to both optimizers, preserving the 2:1 ratio."""
    mult = wsd_lr(step, peak_lr=1.0, warmup_steps=cfg.warmup_steps,
                  total_steps=cfg.total_steps, decay_steps=cfg.decay_steps)
    for opt in optimizers:
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * mult
    return mult


_p1_8 = nn.Parameter(torch.randn(4, 4))
_p2_8 = nn.Parameter(torch.randn(4))
_muon8 = torch.optim.SGD([_p1_8], lr=6e-3)
_adamw8 = torch.optim.SGD([_p2_8], lr=3e-3)
_optimizers8 = [_muon8, _adamw8]
attach_base_lrs(_optimizers8)
_params8 = all_params_of(_optimizers8)
assert _params8 == [_p1_8, _p2_8]

_cfg8 = Cfg(warmup_steps=5, total_steps=40, decay_steps=10)
_m0 = set_lr(_optimizers8, 0, _cfg8)
_m4 = set_lr(_optimizers8, 4, _cfg8)
_m20 = set_lr(_optimizers8, 20, _cfg8)     # deep in the stable phase
assert 0 < _m0 < _m4 <= 1.0, "warmup must ramp the multiplier up monotonically"
assert math.isclose(_m20, 1.0), "stable phase must hold the multiplier at exactly 1.0"
assert math.isclose(_muon8.param_groups[0]["lr"], 6e-3 * _m20)
assert math.isclose(_adamw8.param_groups[0]["lr"], 3e-3 * _m20)
# The 2:1 ratio between the two peaks is preserved at every step.
for _s in (0, 4, 20, 35):
    _m = set_lr(_optimizers8, _s, _cfg8)
    _ratio = _muon8.param_groups[0]["lr"] / _adamw8.param_groups[0]["lr"]
    assert math.isclose(_ratio, 2.0, rel_tol=1e-9), _ratio

print("[block #8 OK] set_lr ramps warmup, holds the plateau, and preserves the 6e-3:3e-3 ratio.\n")


# =====================================================================
# GLUE: a minimal Stack100M-shaped model, standing in for Ch. 14.4's real
# transformer (out of scope for this chapter, which only consumes its
# forward()/`.layers[i].q_norm_gain` surface). Used by blocks #9, #10, #17.
# =====================================================================


class _TinyLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.mix = nn.Linear(d_model, d_model)
        self.q_norm_gain = nn.Parameter(torch.ones(n_heads))
        self.k_norm_gain = nn.Parameter(torch.ones(n_heads))

    def forward(self, x):
        return x + torch.tanh(self.mix(x))


class _TinyStackModel(nn.Module):
    """GLUE: only the surface this chapter's loop code touches -- forward(idx,
    targets=, position_ids=, seq_ids=, record=, logits_to_keep=) ->
    (logits|None, loss|None), plus `.layers[i].q_norm_gain/k_norm_gain`, the
    parameters MuonClip (`qk_clip_`) rescales under QK-norm."""

    def __init__(self, vocab_size, d_model, n_layers, n_heads):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(_TinyLayer(d_model, n_heads) for _ in range(n_layers))
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, targets=None, position_ids=None, seq_ids=None,
                record=None, logits_to_keep=0, **kw):
        x = self.tok_emb(idx)
        if position_ids is not None:
            x = x + 0.01 * position_ids.unsqueeze(-1).float()
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if record is not None:
                # A per-head "max attention logit" proxy driven by the layer's
                # own q/k gains -- exactly what qk_clip_ is meant to rescale.
                record[i] = layer.q_norm_gain.detach() * layer.k_norm_gain.detach() * 50.0
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=-100)
        return logits, loss


def qk_clip_(model, max_logits, tau: float = 30.0):
    """GLUE stand-in for Ch. 14.6's qk_clip_: under QK-norm the parameter that
    controls logit scale is the q_norm/k_norm gain (RMSNorm makes rescaling
    W_Q/W_K a provable no-op -- see the chapter text), so that is what a
    firing layer gets rescaled on. Returns the number of layers that fired."""
    fired = 0
    for i, per_head_max in max_logits.items():
        over = per_head_max > tau
        if over.any():
            fired += 1
            scale = torch.where(over, tau / per_head_max.clamp_min(1e-6),
                                torch.ones_like(per_head_max))
            layer = model.layers[i]
            layer.q_norm_gain.data.mul_(scale.sqrt())
            layer.k_norm_gain.data.mul_(scale.sqrt())
    return fired


# =====================================================================
# Block #9 (chapter line ~610: maybe_qk_clip)
# Copied verbatim (the `from stacklm.optim import qk_clip_` import is
# dropped; the GLUE `qk_clip_` above stands in for it).
# =====================================================================


@torch.no_grad()
def maybe_qk_clip(raw_model, probe, cfg, step):
    """MuonClip on a schedule. Returns the number of layers that fired, or None
    when this step is not a measurement step."""
    if not cfg.qk_clip_every or step % cfg.qk_clip_every:
        return None
    record: "dict[int, torch.Tensor]" = {}          # a DICT, keyed by layer index
    raw_model(probe["input_ids"], position_ids=probe["position_ids"],
              seq_ids=probe["seq_ids"], record=record,
              logits_to_keep=1)     # record is filled in the blocks; skip the lm_head
    return qk_clip_(raw_model, record, tau=cfg.qk_clip_tau)


torch.manual_seed(1)
_tiny9 = _TinyStackModel(vocab_size=16, d_model=8, n_layers=3, n_heads=2)
with torch.no_grad():
    _tiny9.layers[1].q_norm_gain.fill_(2.0)   # will exceed tau=30 after the *50 proxy factor
_probe9 = {
    "input_ids": torch.randint(0, 16, (2, 4)),
    "position_ids": torch.arange(4).unsqueeze(0).repeat(2, 1),
    "seq_ids": torch.zeros(2, 4, dtype=torch.long),
}
_cfg9 = Cfg(qk_clip_every=5, qk_clip_tau=30.0)
assert maybe_qk_clip(_tiny9, _probe9, _cfg9, step=1) is None, "non-measurement steps must return None"
_before9 = _tiny9.layers[1].q_norm_gain.clone()
_fired9 = maybe_qk_clip(_tiny9, _probe9, _cfg9, step=5)
assert isinstance(_fired9, int) and _fired9 >= 1, _fired9
assert not torch.equal(_tiny9.layers[1].q_norm_gain, _before9), \
    "qk_clip_ must actually rescale a firing layer's gain"

print(f"[block #9 OK] maybe_qk_clip gates on the schedule and rescales {_fired9} firing layer(s).\n")


# =====================================================================
# GLUE for block #10: block #4's real `accumulate` hard-codes
# `torch.autocast("cuda", ...)` and is SKIPPED as needs-gpu. This CPU-safe
# stand-in keeps the exact same contract -- grad_accum_steps micro-batches,
# `loss / grad_accum_steps` BEFORE backward, mean loss returned as a device
# tensor plus `data_wait` -- so block #10, which CALLS accumulate, can run.
# =====================================================================


def accumulate(model, optimizers, train_iter, cfg):
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)
    loss_sum = torch.zeros((), device=cfg.device)
    data_wait = 0.0
    for _ in range(cfg.grad_accum_steps):
        t_fetch = time.perf_counter()
        batch = next(train_iter)
        data_wait += time.perf_counter() - t_fetch
        logits, loss = model(batch["input_ids"], targets=batch["targets"],
                             position_ids=batch["position_ids"], seq_ids=batch["seq_ids"])
        loss = loss / cfg.grad_accum_steps          # scale BEFORE backward
        loss.backward()
        loss_sum += loss.detach()
    return loss_sum, data_wait


def _toy_batches(vocab_size, batch_size, seq_len, seed=0):
    g = torch.Generator().manual_seed(seed)
    while True:
        ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g)
        yield {"input_ids": ids[:, :-1], "targets": ids[:, 1:],
               "position_ids": torch.arange(seq_len - 1).unsqueeze(0).repeat(batch_size, 1),
               "seq_ids": torch.zeros(batch_size, seq_len - 1, dtype=torch.long)}


# =====================================================================
# Block #10 (chapter line ~651: "inside the main loop, once per optimizer step")
# Copied verbatim, with `model`/`optimizers`/`params`/`train_iter`/`raw_model`/
# `probe`/`cfg`/`step` bound beforehand so the pasted lines run unmodified.
# =====================================================================


torch.manual_seed(2)
model = _TinyStackModel(vocab_size=16, d_model=8, n_layers=3, n_heads=2)
raw_model = model
muon_params10 = [layer.mix.weight for layer in model.layers]
adamw_params10 = ([model.tok_emb.weight, model.lm_head.weight]
                  + [p for layer in model.layers
                     for p in (layer.mix.bias, layer.q_norm_gain, layer.k_norm_gain)])
optimizers = [torch.optim.SGD(muon_params10, lr=6e-3), torch.optim.SGD(adamw_params10, lr=3e-3)]
attach_base_lrs(optimizers)
params = all_params_of(optimizers)
train_iter = _toy_batches(vocab_size=16, batch_size=2, seq_len=5, seed=3)
probe = {
    "input_ids": torch.randint(0, 16, (2, 4)),
    "position_ids": torch.arange(4).unsqueeze(0).repeat(2, 1),
    "seq_ids": torch.zeros(2, 4, dtype=torch.long),
}
cfg = Cfg(grad_accum_steps=3, grad_clip=1.0, qk_clip_every=2, qk_clip_tau=30.0,
         device="cpu", warmup_steps=5, total_steps=40, decay_steps=10)
step = 2

# --- inside the main loop, once per optimizer step ---
lr_mult              = set_lr(optimizers, step, cfg)
loss_sum, data_wait  = accumulate(model, optimizers, train_iter, cfg)
grad_norm            = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)   # PRE-clip norm
for opt in optimizers:
    opt.step()                                       # Muon (Newton-Schulz), then fused AdamW
qk_fired = maybe_qk_clip(raw_model, probe, cfg, step)   # MuonClip: AFTER the step, on a schedule

assert 0 < lr_mult <= 1.0
assert loss_sum.dim() == 0 and torch.isfinite(loss_sum)
assert torch.isfinite(grad_norm) and grad_norm.item() >= 0
assert step % cfg.qk_clip_every == 0 and isinstance(qk_fired, int), \
    "step=2 with qk_clip_every=2 must be a measurement step"

print(f"[block #10 OK] one full optimizer step: lr_mult={lr_mult:.3f} "
      f"loss={loss_sum.item():.3f} |g|={grad_norm.item():.3f} qk_fired={qk_fired}.\n")


# =====================================================================
# Block #11 (chapter line ~682: FlexAttention Option A -- make_doc_block_mask)
# Copied verbatim.
# =====================================================================

from torch.nn.attention.flex_attention import create_block_mask, flex_attention

def make_doc_block_mask(seq_ids):
    """`seq_ids[b, i]` is the document index of token i (Ch. 14.2) -- exactly the
    signal the mask needs, no cumsum required."""
    def doc_causal(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (seq_ids[b, q_idx] == seq_ids[b, kv_idx])

    B, T = seq_ids.shape
    # BlockMask stores which 128x128 blocks are entirely masked, so the kernel
    # *skips* them: packing many short documents gets cheaper, not more expensive.
    return create_block_mask(doc_causal, B=B, H=None, Q_LEN=T, KV_LEN=T,
                             device=seq_ids.device)

# in the attention module:  out = flex_attention(q, k, v, block_mask=bm, enable_gqa=True)


torch.manual_seed(3)
_B11, _T11, _H11, _Dh11 = 1, 8, 2, 4
_seq_ids11 = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])   # two 4-token documents
_bm11 = make_doc_block_mask(_seq_ids11)
_q11 = torch.randn(_B11, _H11, _T11, _Dh11)
_k11 = torch.randn(_B11, _H11, _T11, _Dh11)
_v11a = torch.randn(_B11, _H11, _T11, _Dh11)
_out11a = flex_attention(_q11, _k11, _v11a, block_mask=_bm11)
_v11b = _v11a.clone()
_v11b[:, :, 4:, :] = torch.randn(_B11, _H11, 4, _Dh11)   # perturb ONLY doc-1's values
_out11b = flex_attention(_q11, _k11, _v11b, block_mask=_bm11)
assert torch.allclose(_out11a[:, :, :4, :], _out11b[:, :, :4, :], atol=1e-5), \
    "doc-0's output must NOT depend on doc-1's values -- the whole point of the mask"
assert not torch.allclose(_out11a[:, :, 4:, :], _out11b[:, :, 4:, :]), \
    "doc-1's own output SHOULD depend on doc-1's own values"

print("[block #11 OK] make_doc_block_mask blocks cross-document attention (verified by leakage test).\n")


# =====================================================================
# Block #13 (chapter line ~734: CheckpointedBlock / enable_activation_checkpointing)
# Copied verbatim.
# =====================================================================


class CheckpointedBlock(nn.Module):
    """Wraps one Stack100M transformer block so its *internal* activations are
    recomputed in the backward pass instead of held for the whole forward pass."""
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x, cos, sin, **kw):
        if self.training:
            # use_reentrant=False: the modern, autograd-graph-friendly variant.
            # Required for torch.compile compatibility and for correct autocast
            # state during recompute.
            return checkpoint(self.block, x, cos, sin, use_reentrant=False, **kw)
        return self.block(x, cos, sin, **kw)


def enable_activation_checkpointing(model) -> None:
    """In-place: wrap all 30 blocks. Keeps the 30 block-input activations
    (~2.0 GB at micro_batch 32) and recomputes everything inside each block
    (~11-17 GB), i.e. a ~3-8x cut on block activations for ~33% more FLOPs --
    one extra forward pass per block, and forward is ~1/3 of fwd+bwd.
    Does NOT touch the loss head, which sits outside the blocks."""
    model.blocks = nn.ModuleList(CheckpointedBlock(b) for b in model.blocks)


class _ToyBlock(nn.Module):
    """GLUE: a tiny stand-in for one Stack100M transformer block, with the
    same (x, cos, sin) forward signature CheckpointedBlock wraps."""
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x, cos, sin, **kw):
        return x + torch.tanh(self.lin(x)) * cos.mean() * sin.mean()


class _ToyBlockyModel(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.blocks = nn.ModuleList(_ToyBlock(d) for _ in range(n))


torch.manual_seed(4)
_m13 = _ToyBlockyModel(d=6, n=3)
enable_activation_checkpointing(_m13)
assert all(isinstance(b, CheckpointedBlock) for b in _m13.blocks)

_x13 = torch.randn(2, 6, requires_grad=True)
_cos13, _sin13 = torch.ones(6), torch.ones(6)
_m13.train()
_out13 = _x13
for _b in _m13.blocks:
    _out13 = _b(_out13, _cos13, _sin13)
_out13.sum().backward()
assert _x13.grad is not None and torch.isfinite(_x13.grad).all(), \
    "checkpointed backward must still produce finite gradients"

_m13.eval()
with torch.no_grad():
    _out13_eval = _x13.detach()
    for _b in _m13.blocks:
        _out13_eval = _b(_out13_eval, _cos13, _sin13)
assert _out13_eval.shape == _x13.shape

print("[block #13 OK] CheckpointedBlock recomputes correctly (finite grads through checkpoint).\n")


# =====================================================================
# GLUE for blocks #15/#16: a minimal PackedMemmapDataset (Ch. 14.2, a
# different chapter, not tested here) -- map-style, __getitem__ -> 4
# LongTensors of length seq_len-1. The last item is padded so the collate's
# pad->-100 masking has something to actually mask.
# =====================================================================


class PackedMemmapDataset(Dataset):
    def __init__(self, shard_dir: str, bos_id: int | None = None,
                n: int = 9, seq_len: int = 6, pad_id: int = 99):
        self.n, self.seq_len, self.bos_id, self.pad_id = n, seq_len, bos_id, pad_id

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        if idx == self.n - 1:            # final window: padded
            ids = torch.full((self.seq_len,), self.pad_id, dtype=torch.long)
            ids[0] = 1
        else:
            g = torch.Generator().manual_seed(idx)
            ids = torch.randint(1, 50, (self.seq_len,), generator=g)
        pos = torch.arange(self.seq_len)
        seq = torch.zeros(self.seq_len, dtype=torch.long)
        return {"input_ids": ids[:-1], "position_ids": pos[:-1],
                "seq_ids": seq[:-1], "targets": ids[1:]}


# =====================================================================
# Block #15 (chapter line ~916: ResumableShuffleSampler / make_collate / build_loader)
# Copied verbatim (the `from stacklm.data.dataset import PackedMemmapDataset`
# import is dropped -- the GLUE class above, defined earlier in this file,
# has that exact name).
# =====================================================================


class ResumableShuffleSampler(Sampler):
    """Infinite, deterministic index stream over a map-style dataset.

    Epoch `e` uses permutation `randperm(n, seed + e)`, so the global sample
    offset `start` uniquely determines the entire remaining order. Resume is
    therefore just arithmetic -- no iterator state to serialize."""
    def __init__(self, n: int, seed: int, start: int = 0):
        self.n, self.seed, self.start = n, seed, start

    def __iter__(self):
        pos = self.start
        while True:
            epoch, offset = divmod(pos, self.n)
            g = torch.Generator().manual_seed(self.seed + epoch)
            perm = torch.randperm(self.n, generator=g).tolist()
            yield from perm[offset:]
            pos = (epoch + 1) * self.n
    # Deliberately no __len__: this stream is infinite, and len(dataloader)
    # should fail loudly rather than lie.


def make_collate(pad_id: int):
    """Ch. 14.2's packer pads the final window of each shard with `pad_id`; the
    loss must ignore those positions. `pad_id` is passed in from the tokenizer
    object -- NEVER hard-coded, because Ch. 14.3 puts the nine specials in the
    final nine ids (pad_id = 32761) and a wrong constant would mask out a common
    BPE merge token instead, silently deleting supervision."""
    def collate(samples):
        batch = default_collate(samples)          # keeps all four keys, incl. seq_ids
        batch["targets"] = batch["targets"].masked_fill(batch["targets"] == pad_id, -100)
        return batch
    return collate


def build_loader(shard_dir, cfg, tok, *, start_sample=0, shuffle=True):
    # bos_id also lives in the shard dir's manifest.json; passing it explicitly
    # makes the loader work even against a shard set whose manifest went missing.
    ds = PackedMemmapDataset(shard_dir, bos_id=tok.bos_id)
    sampler = (ResumableShuffleSampler(len(ds), cfg.seed, start_sample)
               if shuffle else None)
    return DataLoader(
        ds,
        batch_size=cfg.micro_batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=cfg.num_workers,     # safe: map-style datasets are index-sharded
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=4 if cfg.num_workers > 0 else None,
        collate_fn=make_collate(tok.pad_id),
    )


class _Tok:
    bos_id = 1
    pad_id = 99


_cfg15 = Cfg(micro_batch_size=3, num_workers=0, seed=7)
_loader15 = build_loader("unused/shard/dir", _cfg15, _Tok())
_batch15 = next(iter(_loader15))
assert set(_batch15.keys()) == {"input_ids", "position_ids", "seq_ids", "targets"}
assert _batch15["input_ids"].shape[0] == _cfg15.micro_batch_size

# Determinism: same seed + same start must reproduce the exact index order,
# and resuming from an offset must land exactly where a from-scratch stream
# would be at that offset -- the whole point of the resume-by-arithmetic design.
_it_a = iter(ResumableShuffleSampler(9, seed=7, start=0))
_seq_a = [next(_it_a) for _ in range(9)]
_it_b = iter(ResumableShuffleSampler(9, seed=7, start=0))
_seq_b = [next(_it_b) for _ in range(9)]
assert _seq_a == _seq_b
_it_c = iter(ResumableShuffleSampler(9, seed=7, start=3))
_seq_c = [next(_it_c) for _ in range(6)]
assert _seq_c == _seq_a[3:9], (_seq_c, _seq_a[3:9])

# The collate function must rewrite pad_id targets to -100, and only those.
_ds15 = PackedMemmapDataset("x", bos_id=1)
_collate15 = make_collate(_ds15.pad_id)
_batch_pad = _collate15([_ds15[len(_ds15) - 1]])
assert (_batch_pad["targets"] == -100).any(), "pad tokens must be masked to -100"
_batch_clean = _collate15([_ds15[0]])
assert not (_batch_clean["targets"] == -100).any(), "a non-padded window must be untouched"

print("[block #15 OK] resumable sampler is deterministic/resumable; collate masks pad_id to -100.\n")


# =====================================================================
# Block #16 (chapter line ~979: the resume snippet)
# Copied verbatim, with `cfg`/`tok`/`step` bound beforehand.
# =====================================================================


cfg = Cfg(micro_batch_size=3, grad_accum_steps=4, num_workers=0, seed=7, shard_dir="dummy")
tok = _Tok()
step = 2

samples_per_step = cfg.micro_batch_size * cfg.grad_accum_steps      # 256
train_loader = build_loader(cfg.shard_dir, cfg, tok,
                            start_sample=step * samples_per_step)

assert samples_per_step == cfg.micro_batch_size * cfg.grad_accum_steps == 12
assert train_loader.sampler.start == step * samples_per_step == 24, train_loader.sampler.start
_batch16 = next(iter(train_loader))
assert _batch16["input_ids"].shape[0] == cfg.micro_batch_size

print(f"[block #16 OK] samples_per_step={samples_per_step}, resume start_sample={step * samples_per_step}.\n")


# =====================================================================
# Block #17 (chapter line ~1017: the non-finite-gradient skip guard)
# Copied verbatim, wrapped in a function so it can be CALLED repeatedly with
# different (step, consecutive_skips, skipped_total) state -- the book's
# fragment mutates module-level loop variables in place, which a standalone
# test file cannot do faithfully without reintroducing them as parameters.
# =====================================================================


def _block17_step(params, optimizers, cfg, step, consecutive_skips, skipped_total):
    grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)

    if torch.isfinite(grad_norm):
        for opt in optimizers:
            opt.step()
        consecutive_skips = 0
    else:
        # Discard this step entirely: the parameters and optimizer state are untouched.
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        consecutive_skips += 1
        skipped_total += 1
        print(f"step {step}: non-finite grad norm ({grad_norm}), skipping "
              f"({consecutive_skips} in a row, {skipped_total} total)")
        if consecutive_skips >= cfg.max_consecutive_skips:
            raise RuntimeError(
                f"{consecutive_skips} consecutive non-finite steps -- the model state is "
                f"probably already corrupt. Roll back to an earlier step_*.pt.")
    return grad_norm, consecutive_skips, skipped_total


_p17 = nn.Parameter(torch.randn(3, 3))
_opt17 = torch.optim.SGD([_p17], lr=0.1)
_cfg17 = Cfg(grad_clip=1.0, max_consecutive_skips=3)

# A finite grad_norm must step the optimizer (parameter changes) and reset
# consecutive_skips.
_before17 = _p17.detach().clone()
_p17.grad = torch.randn(3, 3)
_gn, _cs, _st = _block17_step([_p17], [_opt17], _cfg17, step=0,
                              consecutive_skips=0, skipped_total=0)
assert torch.isfinite(_gn) and _cs == 0 and _st == 0
assert not torch.equal(_p17.detach(), _before17), "a finite grad_norm must step the optimizer"

# A RUN of non-finite grads must skip every time (parameter frozen) and abort
# with RuntimeError exactly on the `max_consecutive_skips`-th consecutive one.
_frozen17 = _p17.detach().clone()
_cs = 0
_raised = False
for _s in range(1, _cfg17.max_consecutive_skips + 1):
    # zero_grad(set_to_none=True) inside the guard clears .grad after every
    # skip, exactly like the real loop's next micro-batch would repopulate
    # it; re-supply a fresh bad gradient each step to simulate "still poisoned".
    _p17.grad = torch.full((3, 3), float("nan"))
    try:
        _gn, _cs, _ = _block17_step([_p17], [_opt17], _cfg17, step=_s,
                                    consecutive_skips=_cs, skipped_total=0)
        assert torch.equal(_p17.detach(), _frozen17), "a skipped step must leave params untouched"
    except RuntimeError:
        _raised = True
        assert _s == _cfg17.max_consecutive_skips, \
            f"must raise on exactly the {_cfg17.max_consecutive_skips}th consecutive skip, got {_s}"
        break
assert _raised, "max_consecutive_skips consecutive non-finite grad norms must raise RuntimeError"

print("[block #17 OK] finite grads step the optimizer; a run of NaNs skips then aborts correctly.\n")


# =====================================================================
# Block #18 (chapter line ~1104: flops_per_token / utilization -- MFU/HFU)
# Copied verbatim.
# =====================================================================


A100_BF16_PEAK    = 312e12   # A100 80GB SXM, bf16 dense (no structured sparsity)
RTX4090_BF16_PEAK = 165e12   # RTX 4090, bf16/fp16 tensor core with fp32 accumulate, dense
T4_FP16_PEAK      =  65e12   # Turing T4 -- fp16 only; no bf16 tensor cores


def flops_per_token(n_params, n_layers, seq_len, d_model, causal=True):
    """6ND plus the attention score/value matmuls the 6ND rule omits."""
    attn = 6 * n_layers * seq_len * d_model
    return 6 * n_params + (attn if causal else 2 * attn)


def utilization(n_params, tokens_per_sec, cfg, peak_flops=A100_BF16_PEAK,
                recompute_factor=1.0):
    """Returns (mfu, hfu). `recompute_factor` is 1.0 with no activation
    checkpointing and ~4/3 with full-block checkpointing."""
    m = cfg.model
    f = flops_per_token(n_params, m.n_layers, m.max_seq_len, m.d_model)
    mfu = f * tokens_per_sec / peak_flops
    return mfu, mfu * recompute_factor


# --- exercise block #18: the chapter's own worked example, verbatim numbers.
_cfg18 = Cfg(model=ModelCfg(n_layers=30, max_seq_len=2048, d_model=512))
_n_params18 = 101.4e6
_tokens_per_sec18 = 227_951

_fpt18 = flops_per_token(_n_params18, 30, 2048, 512)
assert math.isclose(_fpt18 / 1e6, 797.1, rel_tol=0.005), _fpt18 / 1e6   # chapter: "797.1 MFLOP/token"

_mfu18, _hfu18 = utilization(_n_params18, _tokens_per_sec18, _cfg18, peak_flops=A100_BF16_PEAK)
assert math.isclose(_mfu18, 0.582, rel_tol=0.02), _mfu18                # chapter: "MFU ~ 58.2%"
assert _mfu18 == _hfu18, "recompute_factor defaults to 1.0 (no checkpointing): MFU must equal HFU"

_mfu18_ckpt, _hfu18_ckpt = utilization(_n_params18, _tokens_per_sec18, _cfg18,
                                       peak_flops=A100_BF16_PEAK, recompute_factor=4 / 3)
assert math.isclose(_hfu18_ckpt, _mfu18 * 4 / 3, rel_tol=1e-9)          # HFU ~ 1.33x MFU with checkpointing
assert math.isclose(_mfu18_ckpt, _mfu18, rel_tol=1e-9), "checkpointing must NOT change MFU"

print(f"[block #18 OK] flops/token={_fpt18/1e6:.1f}M, MFU={_mfu18*100:.1f}%, "
      f"HFU(ckpt)={_hfu18_ckpt*100:.1f}%.\n")


print("=== All tested blocks (#5,#6,#7,#8,#9,#10,#11,#13,#15,#16,#17,#18) "
      "executed and verified successfully. ===")
