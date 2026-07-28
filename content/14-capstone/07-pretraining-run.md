# 14.7 The Pretraining Run: A Complete Single-GPU Training Loop

Every chapter so far has produced one *piece* of Stack-100M: the data shards (Ch. 14.2), the
tokenizer (Ch. 14.3), the model definition (Ch. 14.4), a fitted scaling law that justified the
final size and token budget (Ch. 14.5), and the optimizer plus learning-rate schedule (Ch. 14.6).
This chapter wires all of it into the one artifact that actually spends the GPU-hours: `train.py`.

By the end of this chapter you will have a complete, resumable, single-GPU training loop that
takes `StackLM`, `PackedDataset`, the Muon+AdamW optimizer, and the WSD schedule, and turns
them into a ~15–25 GPU-hour job on one A100 that ends with a real ~100M-parameter language model.
We will measure *how well* the GPU is being used (model FLOPs utilization, not just
`nvidia-smi`'s misleading utilization percentage), make the run crash-safe with proper
checkpoint/resume, and give configs for the three compute tiers the capstone supports: the
flagship A100, a consumer RTX 4090, and a free Colab T4. We close with a scale-out note — DDP,
then FSDP — for readers who want to go faster than one GPU, while being explicit that **none of
that is required** to train Stack-100M; a single A100 is sufficient for the whole plan.

## From Components to a Running Job

Training loops fail in boring, expensive ways: a crash three hours before the end with no
checkpoint, a silent NaN nobody notices until the next morning, a batch-size bug that quietly
halves the effective learning rate. The job of this chapter is to make each of those failure
modes structurally impossible, not just "usually fine."

Here is the data flow we are wiring together — the shape every subsection below fills in:

```text
 PackedDataset (Ch 14.2)          StackLM (Ch 14.4)
 .bin shards, doc-aware masks  →  forward pass (bf16 autocast)
        │                                │
        │ (x, y, doc_mask)               ▼
        │                          logits, loss = CE(logits, y) + z-loss
        │                                │
        │                                ▼ backward (bf16 autocast region)
        │                          accumulate grads over micro-batches
        │                                │
        │                                ▼
        │                          clip_grad_norm_(params, 1.0)
        │                                │
        │                                ▼
        │                  Muon+AdamW.step()  ← lr from wsd_lr(step)  (Ch 14.6)
        │                                │
        │                                ▼
        │                    every N steps: eval on held-out shard,
        │                    sample generation, checkpoint (model+opt+step+rng)
        ▼                                │
  throughput timer  ───────────────────► MFU = 6·N·(tokens/s) / peak_FLOPs
```

Everything downstream of the model and data already exists by the time this chapter starts; our
job is the loop, not the components. We assume the following interfaces from earlier chapters —
you will not find their bodies here, only the contracts this chapter's code relies on:

```python
# stacklm/model.py   (built in Ch. 14.4 — architecture, cited components)
class StackLMConfig:
    vocab_size: int = 32768
    d_model: int = 512
    n_layers: int = 30
    n_heads: int = 8
    n_kv_heads: int = 2       # GQA, 4:1 ratio
    head_dim: int = 64
    intermediate: int = 1408  # SwiGLU
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    tie_embeddings: bool = True

class StackLM(torch.nn.Module):
    def forward(self, input_ids: "LongTensor[B, T]",
                doc_mask: "BoolTensor[B, T] | None" = None) -> "FloatTensor[B, T, V]": ...

# stacklm/data.py   (built in Ch. 14.2 — streaming, dedup, packing, doc-aware masking)
class PackedDataset(torch.utils.data.IterableDataset):
    def __iter__(self) -> "Iterator[tuple[LongTensor, LongTensor, BoolTensor]]": ...
    def state_dict(self) -> dict: ...          # shard index + in-shard offset, for exact resume
    def load_state_dict(self, state: dict) -> None: ...

# stacklm/optim.py   (built in Ch. 14.6 — Muon+AdamW hybrid, WSD schedule, MuonClip/QK-clip)
def build_optimizer(model: "StackLM", weight_decay: float, betas: tuple[float, float]) -> "torch.optim.Optimizer": ...
def wsd_lr(step: int, peak_lr: float, warmup_steps: int, decay_steps: int, total_steps: int) -> float: ...

# stacklm/tokenizer.py   (built in Ch. 14.3 — byte-level BPE, vocab_size=32768)
class Tokenizer:
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
```

### The `TrainConfig`

One dataclass fixes every run-mechanics number so the script is fully reproducible from a single
object (which we also checkpoint, so a resumed run cannot silently drift from its original
configuration):

```python
# stacklm/train_config.py
from dataclasses import dataclass, field

@dataclass
class TrainConfig:
    # --- model: fixed by capstone/PLAN.md §1, do not change per-run ---
    vocab_size: int = 32768
    d_model: int = 512
    n_layers: int = 30
    n_heads: int = 8
    n_kv_heads: int = 2
    head_dim: int = 64
    intermediate: int = 1408
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    tie_embeddings: bool = True

    # --- data ---
    train_shards: str = "data/train/*.bin"
    val_shards: str = "data/val/*.bin"
    seq_len: int = 2048
    micro_batch_size: int = 32     # sequences per forward/backward pass
    grad_accum_steps: int = 8      # 32 * 2048 * 8 = 524,288 tokens/optimizer-step ≈ 0.5M

    # --- optimizer & schedule: fixed by Ch. 14.6 ---
    peak_lr: float = 6e-3          # Muon group's peak LR (AdamW group uses a lower internal LR)
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    z_loss_coef: float = 1e-4      # PaLM-style logsumexp penalty (Chowdhery et al., 2022)
    warmup_steps: int = 500
    decay_steps: int = 6000        # final WSD "decay" phase, annealed on the mid-training mix
    total_steps: int = 38_147      # ceil(20e9 tokens / 524,288 tokens/step)

    # --- run mechanics ---
    device: str = "cuda"
    autocast_dtype: str = "bfloat16"
    activation_checkpointing: bool = False
    compile_model: bool = True
    eval_every: int = 500
    eval_iters: int = 50
    sample_every: int = 2000
    log_every: int = 10
    ckpt_every: int = 1000
    ckpt_dir: str = "checkpoints/stack-100m"
    seed: int = 1337
```

Two numbers deserve a note. First, `total_steps = 38_147` comes directly from the capstone's
~20B-token budget (Ch. 14.5's fitted scaling law extrapolated to this size, deliberately
over-trained past the ~2B-token Chinchilla-optimal point for [Stack-100M](../14-capstone/05-mini-scaling-laws.html)):
$20\times10^9 \text{ tokens} / 524{,}288 \text{ tokens/step} \approx 38{,}147\text{ steps}$.
Second, `micro_batch_size * seq_len * grad_accum_steps = 32 \times 2048 \times 8 = 524{,}288`,
which lands almost exactly on the PLAN's target effective batch of ≈0.5M tokens — this is the
number every later section's arithmetic will use.

## Precision, Batching, and the Effective Batch Size

### bf16 autocast, no loss scaler

We train in **bf16** (`bfloat16`), not fp16. bf16 keeps fp32's 8-bit exponent (same dynamic
range) and trims the mantissa to 7 bits, so unlike fp16 (5-bit exponent, prone to overflow on
large activations or gradients) it never needs [`GradScaler`](../01-foundations/04-numerics-precision.html)-style
dynamic loss scaling — the classic source of "loss is `nan`, script has silently rescaled to zero
and stalled." We keep the model's master parameters in **fp32** and use `torch.autocast` only
around the forward pass and loss computation; PyTorch's autograd then produces gradients that are
accumulated in the parameters' native fp32 dtype, which matters at these small per-step learning
rates where a pure-bf16 accumulator would round tiny updates away. (Some frontier recipes skip
the fp32 master copy entirely and train fully in bf16 for the memory savings; at 101M params that
saving is negligible; see [Mixed Precision, bf16 & FP8 Training](../03-pretraining/08-mixed-precision-fp8.html)
for the full trade-off.)

```python
import torch

autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

# One micro-batch, forward + loss, entirely inside the autocast region:
with autocast_ctx:
    logits = model(x, doc_mask=doc_mask)                      # (B, T, V) computed in bf16
    ce = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)).float(),              # upcast logits for a stable loss
        y.view(-1),
        ignore_index=-1,                                        # padding / cross-doc positions
    )
    # PaLM-style z-loss: penalize log-sum-exp of the logits growing unboundedly,
    # a cheap stabilizer against the logit blow-ups discussed in
    # Training Stability, Loss Spikes & Debugging Large Runs.
    z_loss = (torch.logsumexp(logits.float(), dim=-1) ** 2).mean()
    loss = ce + cfg.z_loss_coef * z_loss
```

### Gradient accumulation: micro-batch × accum = effective batch

{{fig:grad-accum-effective-batch}}

An 80GB A100 could fit a much larger micro-batch than 32 sequences of length 2048 for a 101M
model, but the *effective* batch size — the number of tokens averaged into one optimizer step —
is what the schedule and optimizer in Ch. 14.6 were tuned around (≈0.5M tokens, in the same
family as recent small-model recipes). We reach that effective batch by accumulating gradients
over `grad_accum_steps` micro-batches before every `optimizer.step()`:

```python
def train_step(model, optimizer, micro_batches, cfg):
    """One optimizer step = grad_accum_steps micro-batches, gradients averaged."""
    optimizer.zero_grad(set_to_none=True)   # zero ONCE per optimizer step, not per micro-batch
    total_loss = 0.0
    for x, y, doc_mask in micro_batches:    # len(micro_batches) == cfg.grad_accum_steps
        x, y, doc_mask = x.to(cfg.device, non_blocking=True), \
                          y.to(cfg.device, non_blocking=True), \
                          doc_mask.to(cfg.device, non_blocking=True)
        with autocast_ctx:
            logits = model(x, doc_mask=doc_mask)
            ce = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)).float(), y.view(-1), ignore_index=-1)
            z_loss = (torch.logsumexp(logits.float(), dim=-1) ** 2).mean()
            loss = (ce + cfg.z_loss_coef * z_loss) / cfg.grad_accum_steps   # scale BEFORE backward
        loss.backward()
        total_loss += loss.item()
    return total_loss   # already averaged over the effective batch
```

Note the `/ cfg.grad_accum_steps` scaling *inside* the loop, before `.backward()` — because
gradients simply add across accumulated backward passes, dividing the loss (and hence its
gradient) by the accumulation count is what makes the accumulated gradient equal the *mean*
gradient over the full 524,288-token batch, matching what a single giant batch would have
produced.

!!! warning "Common pitfall: zeroing gradients inside the accumulation loop"

    A very easy bug: calling `optimizer.zero_grad()` on every micro-batch instead of once per
    optimizer step. That silently turns "8 micro-batches accumulated into one 524k-token step"
    into "8 independent, tiny 65k-token steps at 1/8th the intended effective batch size" —
    the loss curve still goes down, so it is easy to miss, but the run no longer matches the WSD
    schedule or the Muon hyperparameters tuned in Ch. 14.6, and throughput/MFU numbers become
    meaningless because the accounting no longer matches reality. Zero once, accumulate `N`
    times, then step — always assert `len(micro_batches) == cfg.grad_accum_steps` if you refactor
    this loop.

## Gradient Clipping and the WSD Schedule in the Loop

After the accumulated backward pass, we clip the **global** gradient norm across all parameters
(not per-tensor) to `grad_clip = 1.0`, matching Ch. 14.6:

$$
\hat g = g \cdot \min\!\left(1,\; \frac{c}{\lVert g \rVert_2 + \epsilon}\right), \qquad c = 1.0
$$

and we set the learning rate for *this* step before calling `optimizer.step()`, reading it from
the Warmup-Stable-Decay schedule built in [the optimizer chapter](../14-capstone/06-optimizer-and-schedule.html)
(cross-linking [Learning Rate Schedules, Warmup, Batch Size & Hyperparameters](../03-pretraining/10-lr-schedules-hparams.html)
for the general theory): a short linear warmup, a long constant "stable" plateau, then a short
decay at the very end — the decay phase is exactly where [mid-training](../14-capstone/08-mid-training.html)
anneals onto a higher-quality data mix.

### One clip, two optimizers

`build_optimizer` (Ch. 14.6) does not hand every parameter to the same update rule: 2-D hidden
weight matrices (attention Q/K/V/O projections, the SwiGLU up/gate/down projections) go to Muon's
Newton-Schulz-orthogonalized momentum update, while the tied embedding table and every 1-D tensor
(RMSNorm and QK-norm scale vectors) go to AdamW — Muon's orthogonalization is only defined for
matrices, and 1-D parameters have no meaningful spectral structure to orthogonalize. We surface
that split here only to make explicit which tensors the `lr_scale` multiplier in `set_lr` actually
reaches, since it is easy to assume all parameters share one learning-rate curve:

```python
def param_groups_for_stacklm(model):
    """2D hidden weight matrices -> Muon; everything else (tied embedding,
    RMSNorm/QK-norm scales) -> AdamW. Built once inside build_optimizer (Ch. 14.6);
    reproduced here only to show which parameters `lr_scale` partitions."""
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if p.ndim == 2 and "embed" not in name and "lm_head" not in name:
            muon_params.append(p)       # attention Q/K/V/O, SwiGLU up/gate/down
        else:
            adamw_params.append(p)      # tied embedding, RMSNorm/QK-norm scales
    return muon_params, adamw_params
```

Crucially, `clip_grad_norm_` below is still computed **once, jointly, over every parameter** —
Muon-bound and AdamW-bound alike — because the point of global-norm clipping is to catch a
*model-wide* gradient explosion (e.g., a bad batch, a numerical instability in one deep layer
propagating backward) regardless of which optimizer will consume which slice of it; clipping each
group separately would let one group's blow-up hide behind the other's normal-sized gradients.

```python
from stacklm.optim import wsd_lr

def set_lr(optimizer, step, cfg):
    lr = wsd_lr(step, peak_lr=cfg.peak_lr, warmup_steps=cfg.warmup_steps,
                decay_steps=cfg.decay_steps, total_steps=cfg.total_steps)
    for group in optimizer.param_groups:
        # Scale each group relative to its own base LR (Muon group vs. AdamW group run at
        # different absolute learning rates; only the *shape* of the WSD curve is shared).
        group["lr"] = lr * group.get("lr_scale", 1.0)
    return lr

# --- inside the main loop, once per optimizer step ---
lr = set_lr(optimizer, step, cfg)
loss = train_step(model, optimizer, next(micro_batch_iter), cfg)
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
optimizer.step()   # Muon (Newton-Schulz orthogonalized momentum) + AdamW, MuonClip applied inside
```

`clip_grad_norm_` returns the *pre-clip* norm, which is worth logging: a sudden multi-sigma jump
in `grad_norm` — clipped away or not — is often the earliest warning of an instability, well
before it shows up in the loss (see [Training Stability, Loss Spikes & Debugging Large Runs](../03-pretraining/11-training-stability.html)).
Ch. 14.6's **MuonClip / QK-clip** operates *inside* `optimizer.step()` on the attention Q/K
projection weights specifically, rescaling them when the Muon update would push attention logits
into an unstable range — it is complementary to, not a replacement for, the global grad-norm clip
here.

## Memory: Optional Activation Checkpointing

{{fig:training-memory-budget}}

At 101M parameters with `micro_batch_size=32`, `seq_len=2048`, Stack-100M's activation memory
comfortably fits in an A100's 80GB without checkpointing — the worked example below does the
arithmetic. But the same code needs to run on a 24GB RTX 4090 and a 16GB Colab T4, and
you may want a larger micro-batch on the A100 for higher throughput. **Activation checkpointing**
(Chen et al., 2016) trades recomputation for memory: instead of storing every transformer block's
activations for the backward pass, we store only block boundaries and *recompute* each block's
forward pass during backward. This is the same lever used at much larger scale — see
[Memory-Efficient Training: Checkpointing, Offloading & LoRA Math](../04-kernels-efficiency/10-memory-efficient-training.html)
for the full memory/compute trade-off, and Korthikanti et al. (2022) for how large-model training
minimizes *how much* to recompute rather than an all-or-nothing choice.

!!! example "Worked example: does the A100 actually need checkpointing?"

    Activations, not weights, dominate training memory at this parameter count. With
    [FlashAttention](../04-kernels-efficiency/03-flash-attention-2-3.html)-backed SDPA (no
    materialized `T×T` score matrix), the tensors we must keep for the backward pass are
    dominated by the per-block residual stream and the SwiGLU intermediate. For one micro-batch of
    `B×T = 32×2048 = 65{,}536` tokens in bf16 (2 bytes each):

    - residual-stream input saved at each of the 30 blocks: $65{,}536 \times 512 \times 2\text{ B} \times 30 \approx 2.0\text{ GB}$;
    - SwiGLU intermediate ($d_\text{ff}=1408$) per block: $65{,}536 \times 1408 \times 2\text{ B} \times 30 \approx 5.5\text{ GB}$;
    - attention Q/K/V/O projections, QK-norm, and MLP up/gate activations add a few GB more.

    That puts activations **on the order of 10–15 GB** — comfortably inside 80 GB alongside the
    fp32 master weights (~0.4 GB), the bf16 compute copies, the gradients (~0.4 GB in bf16), and
    ~0.9 GB of optimizer state. So the flagship A100 needs no checkpointing. On a 24 GB 4090 the
    *same* 32-sequence micro-batch would not leave room, which is exactly why that tier drops to 8
    sequences and turns checkpointing on — collapsing the `30×` residual-stream term to a single
    block's worth of live activations.

```python
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class CheckpointedBlock(nn.Module):
    """Wraps one StackLM transformer block so its activations are recomputed
    in the backward pass instead of held in memory for the whole forward pass."""
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x, doc_mask=None):
        if self.training:
            # use_reentrant=False: the modern, autograd-graph-friendly checkpoint variant;
            # required for compatibility with torch.compile and with QK-norm's internal state.
            return checkpoint(self.block, x, doc_mask, use_reentrant=False)
        return self.block(x, doc_mask)


def enable_activation_checkpointing(model: "StackLM") -> None:
    """In-place: wrap every one of the 30 blocks. Cuts activation memory by roughly a
    factor of n_layers (only block-boundary activations survive to the backward pass)
    at the cost of ~30% more FLOPs — one extra forward pass per checkpointed block."""
    model.blocks = nn.ModuleList(CheckpointedBlock(b) for b in model.blocks)

if cfg.activation_checkpointing:
    enable_activation_checkpointing(model)
```

For the A100 flagship tier we leave this **off** by default — the extra ~30% compute directly
costs GPU-hours we would rather spend on tokens — and reserve it for the 24GB/16GB tiers where it
is the difference between fitting in memory and not.

## Checkpoint and Resume: Model, Optimizer, Step, and RNG State

A 15–25 GPU-hour job on a rented A100 *will* occasionally be interrupted — a spot instance
reclaim, a driver hiccup, you closing your laptop. The checkpoint must capture everything needed
to resume as if the interruption never happened: model weights, optimizer state (Muon's momentum
buffers and AdamW's `m`/`v`), the step counter (so the WSD schedule picks up exactly where it left
off), the data loader's position (so we do not silently re-train on the same tokens or skip a
shard), and every RNG state that affects what happens next. This generalizes the pattern in
[Checkpointing, Fault Tolerance & Long-Running Jobs](../03-pretraining/12-checkpointing-fault-tolerance.html)
to our single-GPU case.

```python
import os, random
import numpy as np
import torch
from dataclasses import asdict

def _unwrap(model):
    """torch.compile wraps the module and renames its state_dict keys under
    `_orig_mod.*`; always save/load the *uncompiled* module's state dict so
    checkpoints remain loadable with or without compilation enabled."""
    return model._orig_mod if hasattr(model, "_orig_mod") else model

def save_checkpoint(path, model, optimizer, step, tokens_seen, train_loader, cfg):
    ckpt = {
        "model": _unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "data_state": train_loader.dataset.state_dict(),   # shard index + in-shard offset (Ch 14.2)
        "config": asdict(cfg),
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, path)     # atomic rename on POSIX: never leaves a half-written file

def load_checkpoint(path, model, optimizer, train_loader, device="cuda"):
    ckpt = torch.load(path, map_location=device)
    _unwrap(model).load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    train_loader.dataset.load_state_dict(ckpt["data_state"])
    torch.set_rng_state(ckpt["rng_state"]["torch"])
    torch.cuda.set_rng_state_all(ckpt["rng_state"]["cuda"])
    np.random.set_state(ckpt["rng_state"]["numpy"])
    random.setstate(ckpt["rng_state"]["python"])
    return ckpt["step"], ckpt["tokens_seen"]
```

Two details matter enough to call out explicitly. First, the **atomic write**: `torch.save`
directly to `path` risks a truncated, unloadable checkpoint if the process dies mid-write (very
plausible on a preemptible instance); writing to a `.tmp` file and `os.replace`-ing it is atomic
on POSIX filesystems, so the on-disk checkpoint is always either the previous complete one or the
new complete one, never a half-written mixture. Second, we save `train_loader.dataset.state_dict()`
— not just the RNG seed — because `PackedDataset` (Ch. 14.2) tracks an exact cursor (which shard,
what byte offset) into the packed `.bin` files; without it, a resumed run would restart the data
stream from the beginning of the epoch, quietly re-weighting the data mix.

### How big is a checkpoint, and how many do you keep?

At Stack-100M's scale this is a non-issue, but it is worth doing the arithmetic once so "how many
checkpoints can I afford to keep" stops being a guess. The fp32 model state dict is
$101.4\times10^6 \times 4\text{ bytes} \approx 406\text{ MB}$. Muon stores a single momentum
buffer per 2-D parameter it owns (roughly the 84.6M block-weight parameters from the split above),
$\approx 84.6\times10^6 \times 4\text{ bytes} \approx 338\text{ MB}$; AdamW stores two buffers
($m$, $v$) for the remaining ~16.8M embedding/norm parameters, $\approx 2 \times 16.8\times10^6
\times 4\text{ bytes} \approx 134\text{ MB}$. Total optimizer state is on the order of 470MB, and a
full checkpoint (model + optimizer + small metadata) lands **on the order of 0.9GB**. Keeping the
last 5 checkpoints plus `final.pt` costs under 6GB — trivial next to even the 24GB tier's VRAM,
let alone disk — so there is no real reason not to keep several rolling checkpoints rather than
overwriting a single `latest.pt` in place, in case the *most recent* one turns out to be corrupted
or was written during a loss spike you would rather roll back past.

!!! tip "Practitioner tip"

    Before trusting a checkpoint on a rented GPU you are paying for by the hour, dry-run the
    resume path locally: train for 20 steps, save, kill the process, resume, and diff the loss at
    step 21 against a fresh run that never stopped. On a correctly wired loop these should match
    to floating-point precision (bf16 arithmetic on the same op sequence is deterministic given
    the same input and RNG state). If they diverge, something in the checkpoint is incomplete —
    almost always the data cursor or the optimizer state for the Muon parameter group.

## Measuring Throughput and Model FLOPs Utilization

{{fig:mfu-vs-gpu-util}}

`nvidia-smi`'s "GPU-Util" percentage tells you the GPU was doing *something* during a sampling
window — it is 100% just as happily whether you're running at 90% of peak matmul throughput or
15%. The metric that actually tells you whether you are getting your money's worth is **Model
FLOPs Utilization (MFU)**: achieved FLOPs/s divided by the accelerator's peak FLOPs/s, popularized
as a training-efficiency metric by Chowdhery et al. (PaLM, 2022). We get achieved FLOPs/s from the
**6ND** rule used throughout [the scaling-laws chapter](../03-pretraining/04-scaling-laws.html)
and [Ch. 14.5](../14-capstone/05-mini-scaling-laws.html): one forward-plus-backward pass costs
approximately $6$ FLOPs per parameter per token.

```python
A100_BF16_PEAK_FLOPS = 312e12   # A100 80GB SXM, bf16 tensor-core peak, dense (no structured sparsity)

def estimate_mfu(n_params: int, tokens_per_sec: float, peak_flops: float = A100_BF16_PEAK_FLOPS) -> float:
    """6ND rule: forward+backward costs ~6 FLOPs/parameter/token (Kaplan et al. 2020)."""
    achieved_flops = 6 * n_params * tokens_per_sec
    return achieved_flops / peak_flops

# --- inside the main loop ---
torch.cuda.synchronize()
t0 = time.perf_counter()
loss = train_step(model, optimizer, next(micro_batch_iter), cfg)  # cfg.grad_accum_steps micro-batches
torch.cuda.synchronize()
dt = time.perf_counter() - t0

tokens_this_step = cfg.micro_batch_size * cfg.seq_len * cfg.grad_accum_steps
tokens_per_sec = tokens_this_step / dt
mfu = estimate_mfu(n_params=101_400_000, tokens_per_sec=tokens_per_sec)
```

`torch.cuda.synchronize()` before and after the timed region matters: CUDA kernel launches are
asynchronous, so an un-synchronized wall clock mostly measures how fast the CPU can enqueue work,
not how fast the GPU executes it.

!!! example "Worked example: from step time to GPU-hours"

    Suppose we measure a step time of `dt = 2.3s` for one full optimizer step (8 accumulated
    micro-batches of 32×2048 tokens each, plus the clip and Muon+AdamW update).

    **Tokens/sec:** $524{,}288 \text{ tokens} / 2.3\text{s} \approx 227{,}951 \text{ tokens/s}$.

    **Achieved FLOPs/s** via 6ND with $N \approx 101.4\times10^6$ parameters:
    $$
    6 \times 101.4\times10^6 \times 227{,}951 \approx 1.387\times10^{14} \text{ FLOPs/s} = 138.7 \text{ TFLOP/s}
    $$

    **MFU:** $138.7 / 312 \approx 44.5\%$ — a solid, realistic figure for an un-fused PyTorch
    training loop on a dense, GQA-attention 100M model at this batch size; well-tuned setups
    with `torch.compile`, fused optimizer kernels, and [FlashAttention-backed SDPA](../04-kernels-efficiency/03-flash-attention-2-3.html)
    can push higher.

    **Projected wall-clock for the full ~20B-token run:**
    $$
    20\times10^9 \text{ tokens} / 227{,}951\text{ tokens/s} \approx 87{,}742\text{s} \approx 24.4 \text{ GPU-hours}
    $$

    That sits right at the upper end of the capstone plan's ~15–25 GPU-hour envelope (and, at
    roughly USD 1.50/GPU-hour, about USD 37 — in the same ballpark as the plan's ~\$40–\$100
    figure). Reaching the *lower* end of that range in practice means pushing MFU toward
    55–65% via `torch.compile`, larger micro-batches, and fused kernels — all "on the order of,"
    never a number to take as a guaranteed benchmark.

We recompute MFU on the same schedule as our logging cadence and treat a sustained MFU *drop*
(not just a low absolute value) as a signal worth investigating — a data-loading stall, a memory
fragmentation event triggering more frequent allocator calls, or thermal throttling on a shared
cloud instance can all show up as GPU-Util staying near 100% while MFU quietly falls.

## Evaluation, Sampling, and Logging During the Run

Every `cfg.eval_every` steps we measure held-out loss on a val shard the model has never trained
on (the split created in [the data-pipeline chapter](../14-capstone/02-data-pipeline.html)), and
every `cfg.sample_every` steps we generate a short free-running sample so a human can sanity-check
qualitative progress that a loss number alone can miss (repetition loops, garbled tokenization,
degenerate outputs).

```python
@torch.no_grad()
def evaluate(model, val_iter, cfg, eval_iters=50):
    model.eval()
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        x, y, doc_mask = next(val_iter)
        x, y, doc_mask = x.to(cfg.device), y.to(cfg.device), doc_mask.to(cfg.device)
        with autocast_ctx:
            logits = model(x, doc_mask=doc_mask)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)).float(), y.view(-1), ignore_index=-1)
        losses[i] = loss.item()
    model.train()
    mean_loss = losses.mean().item()
    return mean_loss, math.exp(mean_loss)   # (nats/token, perplexity)


@torch.no_grad()
def generate_sample(model, tokenizer, prompt: str, cfg, max_new_tokens=64,
                     temperature=0.8, top_p=0.95):
    """Naive autoregressive sampling, no KV cache — fine for a periodic sanity check
    during training; production serving uses the KV-cache machinery in
    The Anatomy of LLM Inference: Prefill, Decode & The KV Cache."""
    model.eval()
    ids = torch.tensor([tokenizer.encode(prompt)], device=cfg.device)
    for _ in range(max_new_tokens):
        with autocast_ctx:
            logits = model(ids[:, -cfg.max_seq_len:])[:, -1, :].float()
        logits = logits / max(temperature, 1e-5)
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cutoff = (sorted_probs.cumsum(-1) > top_p).float().argmax(-1).item()
        keep = sorted_idx[:, :cutoff + 1]
        mask = torch.zeros_like(probs).scatter_(-1, keep, 1.0)
        next_id = torch.multinomial(probs * mask, num_samples=1)
        ids = torch.cat([ids, next_id], dim=-1)
    model.train()
    return tokenizer.decode(ids[0].tolist())


def log_metrics(log_path, step, loss, lr, grad_norm, tokens_per_sec, mfu, val_loss=None):
    """Append-only JSONL log — trivial to load into a dataframe for the loss-curve plots
    the capstone's retrospective chapter (14.12) draws from."""
    import json, time
    record = dict(step=step, loss=loss, lr=lr, grad_norm=grad_norm,
                   tokens_per_sec=tokens_per_sec, mfu=mfu, val_loss=val_loss, ts=time.time())
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    if step % cfg.log_every == 0:
        vl = f"  val {val_loss:.3f}" if val_loss is not None else ""
        print(f"step {step:>6}/{cfg.total_steps}  loss {loss:.3f}{vl}  "
              f"lr {lr:.2e}  |g| {grad_norm:.2f}  tok/s {tokens_per_sec:,.0f}  mfu {mfu*100:.1f}%")
```

### Expected loss curve magnitudes

At initialization, cross-entropy over a fresh 32,768-token vocabulary starts at essentially
$\ln(32{,}768) \approx 10.4$ nats/token — the model is guessing uniformly. Warmup and the first
few hundred steps drop this quickly, since the easiest signal (token frequency statistics) is
learned almost immediately; by the end of warmup, loss is typically already down to something
on the order of low-to-mid single digits of nats/token. Through the long **stable** phase of the
WSD schedule, loss decreases slowly and fairly smoothly — this is the bulk of the ~38,000 steps,
and the log-loss-vs-log-tokens curve should look close to a straight line, the empirical signature
the scaling law in Ch. 14.5 was fit from. During the final **decay** phase, as the LR anneals to
near zero *and* [mid-training](../14-capstone/08-mid-training.html) anneals onto the
higher-quality mix, there is typically a further, visible drop — WSD's whole appeal is that this
decay-phase improvement is disproportionate to its short duration. On the Stack-100M data mix, a
finished run should land with a **final train loss on the order of 2.8–3.2 nats/token** (as stated
in `capstone/PLAN.md`) — corresponding to a perplexity on the order of $e^{2.8}\!\approx\!16$ to
$e^{3.2}\!\approx\!25$. Treat both the intermediate numbers and this final range as illustrative
magnitudes to sanity-check your run against, never as a benchmark to match exactly — the precise
value depends on your data mix, dedup quality, and tokenizer.

## The Complete Training Script

Putting every piece above together into one runnable entry point:

```python
# stacklm/train.py
"""Stack-100M pretraining: single-GPU (A100) flagship path.
Usage:  python -m stacklm.train --config configs/a100.yaml
"""
import os, math, time, itertools
import torch
from stacklm.model import StackLM, StackLMConfig
from stacklm.data import PackedDataset
from stacklm.optim import build_optimizer, wsd_lr
from stacklm.tokenizer import Tokenizer
from stacklm.train_config import TrainConfig
from stacklm.checkpoint import save_checkpoint, load_checkpoint  # functions defined above
from stacklm.train_utils import (evaluate, generate_sample, log_metrics,
                                  enable_activation_checkpointing, estimate_mfu)


def build_loader(shards_glob, cfg, split):
    ds = PackedDataset(shards_glob, seq_len=cfg.seq_len, seed=cfg.seed, split=split)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.micro_batch_size,
                                      num_workers=2, pin_memory=True, drop_last=True)
    return dl


def main(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.benchmark = True   # fine: fixed seq_len every step

    model_cfg = StackLMConfig(vocab_size=cfg.vocab_size, d_model=cfg.d_model,
                               n_layers=cfg.n_layers, n_heads=cfg.n_heads,
                               n_kv_heads=cfg.n_kv_heads, head_dim=cfg.head_dim,
                               intermediate=cfg.intermediate, max_seq_len=cfg.max_seq_len,
                               rope_theta=cfg.rope_theta, tie_embeddings=cfg.tie_embeddings)
    model = StackLM(model_cfg).to(cfg.device)  # fp32 master weights; bf16 only inside autocast
    n_params = sum(p.numel() for p in model.parameters())

    if cfg.activation_checkpointing:
        enable_activation_checkpointing(model)
    if cfg.compile_model:
        model = torch.compile(model)   # see Kernel Fusion, torch.compile, CUDA Graphs & Compilers

    optimizer = build_optimizer(model, weight_decay=cfg.weight_decay, betas=cfg.betas)

    train_loader = build_loader(cfg.train_shards, cfg, split="train")
    val_loader = build_loader(cfg.val_shards, cfg, split="val")
    train_iter = iter(train_loader)
    val_iter = itertools.cycle(val_loader)
    tokenizer = Tokenizer.load("tokenizer/stack100m.bpe")

    step, tokens_seen = 0, 0
    resume_path = f"{cfg.ckpt_dir}/latest.pt"
    if os.path.exists(resume_path):
        step, tokens_seen = load_checkpoint(resume_path, model, optimizer, train_loader, cfg.device)
        print(f"resumed from step {step} ({tokens_seen:,} tokens seen)")

    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    while step < cfg.total_steps:
        lr = wsd_lr(step, cfg.peak_lr, cfg.warmup_steps, cfg.decay_steps, cfg.total_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr * group.get("lr_scale", 1.0)

        micro_batches = [next(train_iter) for _ in range(cfg.grad_accum_steps)]

        torch.cuda.synchronize(); t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for x, y, doc_mask in micro_batches:
            x, y, doc_mask = x.to(cfg.device, non_blocking=True), \
                              y.to(cfg.device, non_blocking=True), \
                              doc_mask.to(cfg.device, non_blocking=True)
            with autocast_ctx:
                logits = model(x, doc_mask=doc_mask)
                ce = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)).float(), y.view(-1), ignore_index=-1)
                z_loss = (torch.logsumexp(logits.float(), dim=-1) ** 2).mean()
                loss = (ce + cfg.z_loss_coef * z_loss) / cfg.grad_accum_steps
            loss.backward()
            total_loss += loss.item()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        torch.cuda.synchronize(); dt = time.perf_counter() - t0

        tokens_this_step = cfg.micro_batch_size * cfg.seq_len * cfg.grad_accum_steps
        tokens_seen += tokens_this_step
        tokens_per_sec = tokens_this_step / dt
        mfu = estimate_mfu(n_params, tokens_per_sec)

        val_loss = None
        if step % cfg.eval_every == 0:
            val_loss, val_ppl = evaluate(model, val_iter, cfg, cfg.eval_iters)
        if step % cfg.sample_every == 0:
            sample = generate_sample(model, tokenizer, "The history of", cfg)
            print(f"  sample @ step {step}: {sample[:200]!r}")
        log_metrics(f"{cfg.ckpt_dir}/log.jsonl", step, total_loss, lr,
                     grad_norm.item(), tokens_per_sec, mfu, val_loss)
        if step % cfg.ckpt_every == 0 and step > 0:
            save_checkpoint(resume_path, model, optimizer, step, tokens_seen, train_loader, cfg)

        step += 1

    save_checkpoint(f"{cfg.ckpt_dir}/final.pt", model, optimizer, step, tokens_seen, train_loader, cfg)
    print(f"done: {step} steps, {tokens_seen:,} tokens, final loss {total_loss:.3f}")


if __name__ == "__main__":
    main(TrainConfig())
```

This is the whole flagship loop — nothing here needs multiple GPUs, a job scheduler, or a
distributed launcher. That single-GPU sufficiency is itself a point worth taking seriously: a
101M-parameter dense model with a 524k-token effective batch fits, with room to spare, on one
80GB accelerator.

## Scaling Out: DDP, FSDP, and Beyond One GPU

Stack-100M does not *need* more than one GPU — it is a deliberate choice to keep the flagship
path reproducible on hardware anyone can rent by the hour. But if you have access to the
capstone's optional 8×H100 box (for a faster wall clock on the same model, or as a stepping stone
toward the 1B-parameter scale-up discussed in [the retrospective chapter](../14-capstone/12-retrospective-and-scaleup.html)),
the upgrade path is exactly the one covered in
[Distributed Training I: Data Parallelism, DDP, ZeRO & FSDP](../03-pretraining/05-distributed-data-parallel.html):

**DDP first.** Since the entire model, optimizer state, and activations already fit on one GPU,
the natural next step is pure data parallelism: replicate the model on every GPU, run independent
micro-batches, and all-reduce gradients before each optimizer step.

```python
import contextlib
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
model = DDP(model.to(local_rank), device_ids=[local_rank])

# Skip the gradient all-reduce on every micro-batch except the last one in the
# accumulation window — DDP would otherwise synchronize grad_accum_steps times
# per optimizer step for no benefit.
for i, (x, y, doc_mask) in enumerate(micro_batches):
    ctx = model.no_sync() if i < len(micro_batches) - 1 else contextlib.nullcontext()
    with ctx, autocast_ctx:
        ...
        loss.backward()
```

launched with `torchrun --nproc_per_node=8 -m stacklm.train`. Eight GPUs at near-linear scaling
turns a ~24-GPU-hour single-A100 run into roughly ~3 wall-clock hours on 8 — the *cost* in
GPU-hours is unchanged, only the wall clock improves, which is the entire point of DDP: it does
not reduce total compute or per-GPU memory pressure.

**FSDP when the model no longer fits.** [Distributed Training I](../03-pretraining/05-distributed-data-parallel.html)
covers Fully Sharded Data Parallel in depth; the short version is that FSDP shards parameters,
gradients, *and* optimizer state across GPUs (à la ZeRO, Rajbhandari et al., 2020) instead of
replicating them, trading extra communication for a memory footprint that shrinks with GPU count.
At 101M parameters this is unnecessary — the whole point of Stack-100M's size is that it fits
comfortably on one accelerator — but it becomes the relevant tool the moment you scale toward the
1B-parameter regime the retrospective chapter discusses, where optimizer state (Muon's momentum
buffers plus AdamW's `m`/`v` for the embedding and norm parameters) no longer fits alongside
activations on a single GPU.

**Multi-node, one paragraph.** Beyond one 8-GPU box, the same DDP/FSDP code launches across
machines via `torchrun --nnodes=N --node_rank=R --rdzv_endpoint=...`, with NCCL collectives now
crossing the network instead of NVLink — see [Parallel Computing & Collective Communication](../01-foundations/09-parallel-collectives.html)
for the interconnect trade-offs (InfiniBand/EFA vs. Ethernet) and
[Megatron-LM, DeepSpeed & Parallelism in Practice](../03-pretraining/07-megatron-deepspeed.html)
for the production frameworks that combine data, tensor, and pipeline parallelism at that scale.
None of it is necessary to train Stack-100M; it is the path if you decide to chase a much bigger
model afterward.

## Compute Tiers: A100, RTX 4090, and Free Colab

The same `train.py` and `TrainConfig` serve all three compute tiers documented in the capstone
plan — only the numbers change.

| Tier | GPU | VRAM | precision | micro-batch × seq | grad accum | eff. batch (tokens) | act. ckpt | est. wall-clock | est. cost |
|---|---|---|---|---|---|---|---|---|---|
| **Flagship** | 1×A100 80GB SXM | 80GB | bf16 autocast, no scaler | 32 × 2048 | 8 | ≈524k | off (optional) | ~15–25 GPU-hr | ~USD 40–100 |
| **Consumer** | RTX 4090 | 24GB | bf16 autocast, no scaler | 8 × 2048 | 32 | ≈524k | on (recommended) | ~2–4× the A100 wall-clock | ~USD 0 (owned; electricity only) |
| **Free on-ramp** | Colab T4 | 16GB | fp16 + `GradScaler` | scaled-down model, seq 1024 | tune to fit | reduced target; on-ramp only, not the full run | on | hours, a subset run | USD 0 |

Two of these rows need unpacking beyond the table.

### RTX 4090: same recipe, smaller micro-batch

A 4090's 24GB has to hold fp32 master weights (~101.4M × 4 bytes ≈ 406MB), Muon+AdamW optimizer
state (~1.2× the parameter count in fp32 buffers — one Muon momentum buffer for the 2-D weights
plus two AdamW moments for the embedding/norm parameters, ≈0.5GB total, the same ~470MB from the
checkpoint arithmetic above), and activations — the term that dominates and scales with the
micro-batch. With `activation_checkpointing=True` and a micro-batch of 8 sequences instead of 32,
the same 524,288-token effective batch is reached via 32 accumulation steps instead of 8 — more forward
passes per optimizer step, hence the PLAN's stated ~2–4× wall-clock versus the A100, on top of the
4090's somewhat lower dense bf16 tensor-core throughput. Nothing else in `train.py` changes; only
`TrainConfig.micro_batch_size`, `grad_accum_steps`, and `activation_checkpointing` move.

!!! note "Colab's T4 has no bf16 tensor cores"

    The free-tier Colab T4 is a Turing-generation GPU (compute capability 7.5); bf16 hardware
    acceleration first arrived with Ampere (A100, RTX 30-series and up, compute capability ≥8.0).
    On a T4 the training loop must switch to classic mixed precision — `torch.autocast(dtype=torch.float16)`
    together with `torch.cuda.amp.GradScaler` to guard against fp16's narrower exponent range —
    rather than the scaler-free bf16 path used everywhere else in this chapter. See
    [Numerical Computing, Floating Point & Precision](../01-foundations/04-numerics-precision.html)
    for exactly what the extra 3 exponent bits in bf16 buy you.

    The Colab config is also explicitly a **scaled-down on-ramp**, not the full Stack-100M
    recipe — a smaller model (fewer layers, narrower `d_model`) and a token budget an order of
    magnitude below 20B, chosen so the run finishes in a session that a free Colab instance will
    actually let you keep. It exists so a reader with zero dollars can *see the loop work end to
    end* before committing real money to the flagship A100 run.

!!! interview "Interview Corner"

    **Q:** You're pretraining a ~100M-parameter dense transformer on an A100. `nvidia-smi` shows
    85% GPU utilization the whole time, but when you compute MFU from your measured tokens/sec you
    get only 18%. What does that gap tell you, and what would you check first?

    **A:** GPU-Util just means *some* kernel was running on the SM array during the sampling
    window — it says nothing about whether that kernel was a dense, well-fused matmul running near
    peak throughput or a small, memory-bound, or poorly-batched op leaving most of the tensor
    cores idle. An 85%-util / 18%-MFU run is the classic signature of the GPU spending most of its
    "busy" time on cheap, unfused, or small operations rather than the big matmuls the 6ND FLOP
    count assumes. First checks, roughly in order of likely payoff at this model size: (1) is
    `torch.compile` actually on and fusing the elementwise ops around attention/SwiGLU, rather than
    launching each op as a separate small kernel; (2) is the micro-batch too small — a 100M model
    with `micro_batch_size=4` leaves the SMs under-fed even at bf16, since the matmuls themselves
    become memory-bound rather than compute-bound at small batch × seq_len; (3) is the data loader
    keeping the GPU fed — CPU-side tokenization/packing stalls show up as high `GPU-Util` between
    bursts of idle waiting, not as low util; (4) is attention using a fused SDPA/FlashAttention
    kernel rather than an unfused einsum-based implementation. MFU is the metric that would have
    caught this; util alone would not.

## Key Takeaways

!!! key "Key Takeaways"

    - A single A100 is sufficient for the entire Stack-100M pretraining run (~15–25 GPU-hours,
      ~USD 40–100 illustrative); distributed training (DDP, then FSDP) is an optional speed-up,
      never a requirement, at this model size.
    - **bf16 autocast needs no loss scaler**, unlike fp16 — bf16 shares fp32's 8-bit exponent
      range, so there is no overflow risk to guard against with dynamic loss scaling.
    - **Effective batch size = micro_batch_size × seq_len × grad_accum_steps.** Zero gradients
      once per optimizer step, not once per micro-batch, and scale the loss by
      `1/grad_accum_steps` before `.backward()` so accumulated gradients equal the true batch mean.
    - Clip the **global** gradient norm (not per-tensor) before `optimizer.step()`; a spike in the
      pre-clip norm is often the earliest warning of an instability, ahead of the loss itself.
    - Checkpoints must save **model + optimizer + step + data-loader cursor + every RNG state**,
      written atomically (temp file + rename), to guarantee bit-consistent resumption after an
      interruption.
    - **MFU = 6·N·(tokens/s) ÷ peak FLOPs/s** is the throughput metric that matters; GPU-Util in
      `nvidia-smi` can read near 100% while MFU — and your money — sits far below what the
      hardware can deliver.
    - Activation checkpointing trades ~30% more compute for a large activation-memory cut; skip it
      on an 80GB A100 at these batch sizes, but treat it as necessary on 24GB/16GB tiers.
    - Expect final train loss on the order of **2.8–3.2 nats/token** on the Stack-100M mix — never
      treat this, or any illustrative number in this chapter, as a benchmark to hit exactly.

## Further reading

- Kaplan et al., *Scaling Laws for Neural Language Models* (2020) — the 6ND FLOPs-per-token rule used throughout this chapter's MFU calculation.
- Hoffmann et al., *Training Compute-Optimal Large Language Models* ("Chinchilla," 2022) — the compute-optimal frontier Stack-100M deliberately over-trains past.
- Micikevicius et al., *Mixed Precision Training* (2017) — the loss-scaling machinery bf16 lets us skip.
- Chen et al., *Training Deep Nets with Sublinear Memory Cost* (2016) — the original activation/gradient checkpointing idea.
- Korthikanti et al., *Reducing Activation Recomputation in Large Transformer Models* (2022) — selective, not all-or-nothing, checkpointing at scale.
- Chowdhery et al., *PaLM: Scaling Language Modeling with Pathways* (2022) — popularized Model FLOPs Utilization as a training-efficiency metric; source of the z-loss coefficient used here.
- Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* (2020) — the sharding idea FSDP implements.
- Karpathy, *nanoGPT* and *llm.c* (open-source GPT-2 reproduction projects) — the direct lineage this capstone's single-GPU, cost-conscious training loop updates for 2025–2026.

## Exercises

**1.** The chapter trains in bf16 autocast and states this needs no `GradScaler`, while the
free-tier Colab T4 path must switch to `torch.autocast(dtype=torch.float16)` *with* a
`GradScaler`. Explain (a) what problem `GradScaler` solves for fp16, and (b) why bf16 makes it
unnecessary. What does bf16 give up in exchange, and why does the chapter argue that trade-off is
acceptable here?

??? note "Solution"
    **(a) What `GradScaler` solves for fp16.** fp16 has only a 5-bit exponent, giving it a narrow
    dynamic range. Small gradients underflow to zero and large activations/gradients overflow to
    `inf`, either of which corrupts the update. `GradScaler` multiplies the loss by a large scale
    factor before `.backward()` so that small gradients are pushed up into fp16's representable
    range, then unscales the gradients before the optimizer step, dynamically backing the scale
    factor off whenever an `inf`/`nan` is detected. It is exactly the machinery the chapter
    describes as "the classic source of `loss is nan`, script has silently rescaled to zero and
    stalled."

    **(b) Why bf16 removes the need.** bf16 keeps fp32's full **8-bit exponent** (same dynamic
    range as fp32) and trims only the mantissa to 7 bits. Because the exponent range is identical
    to fp32, there is no realistic overflow/underflow risk on activations or gradients, so there is
    nothing for dynamic loss scaling to guard against — you can call `.backward()` and
    `optimizer.step()` directly.

    **The trade-off.** bf16 gives up mantissa precision: 7 explicit mantissa bits vs fp16's 10, so
    each individual value is represented more coarsely. The chapter accepts this because (i) the
    model's *master* parameters are kept in fp32 and gradients accumulate in fp32, so tiny per-step
    updates at small learning rates are not rounded away; autocast uses bf16 only inside the
    forward/loss region; and (ii) the cross-entropy loss itself is computed on logits explicitly
    upcast with `.float()`, protecting the numerically sensitive step. Range matters more than the
    last few mantissa bits for training stability, which is precisely what bf16 optimizes for.

**2.** In `train_step`, the per-micro-batch loss is divided by `cfg.grad_accum_steps` *before*
`.backward()`. Suppose a teammate deletes that `/ cfg.grad_accum_steps` (leaving everything else,
including the single `zero_grad` per optimizer step, correct). With the chapter's config
(`grad_accum_steps = 8`), what happens to the gradient that reaches `optimizer.step()`, and what is
the practical effect on the run? Contrast this with the *other* pitfall the chapter warns about —
calling `zero_grad()` inside the accumulation loop.

??? note "Solution"
    **Effect of dropping `/ grad_accum_steps`.** Gradients from successive `.backward()` calls are
    *summed* into `.grad` (that is why we zero only once per optimizer step). With the division in
    place, each micro-batch contributes $\frac{1}{8}$ of its gradient, so the accumulated gradient
    equals the **mean** gradient over the full 524,288-token batch — exactly what a single giant
    batch would produce. Delete the division and the accumulated gradient is the **sum** of 8
    micro-batch gradients, i.e. $8\times$ too large. Since $\nabla$ is $8\times$ larger, the update
    is effectively at $8\times$ the intended learning rate.

    **Practical effect.** The run no longer matches the Muon/WSD hyperparameters tuned in Ch. 14.6.
    Global grad-norm clipping to $c=1.0$ partly masks it — the clip rescales the inflated norm back
    toward 1 — but the *direction* is preserved and the effective step size is distorted whenever
    the norm is below the clip threshold, so early/late-training behavior (small gradients) is most
    affected. Expect instability or a mis-scaled loss curve.

    **Contrast with the `zero_grad`-inside-the-loop pitfall.** That bug goes the *opposite*
    direction: zeroing on every micro-batch throws away the previous 7 gradients, so
    `optimizer.step()` sees only the *last* micro-batch — one independent 65,536-token step at
    $\frac{1}{8}$ the intended effective batch, not the 524,288-token step you designed. Both bugs
    leave the loss curve going down (so both are easy to miss), but one inflates the effective LR by
    $8\times$ while the other shrinks the effective batch by $8\times$; the chapter's rule "zero
    once, accumulate $N$, then step, and scale the loss by $1/N$ before backward" is what makes the
    accumulated gradient equal the true batch mean.

**3.** You are configuring a scaled-down on-ramp tier with a smaller model. You choose
`micro_batch_size = 4` and `seq_len = 1024`, and you want a **reduced** effective batch of exactly
262,144 tokens per optimizer step with a total token budget of $2\times10^9$ tokens. Compute (a)
the `grad_accum_steps` you must set, and (b) `total_steps`. Show the arithmetic.

??? note "Solution"
    **(a) `grad_accum_steps`.** Effective batch
    $= \texttt{micro\_batch\_size} \times \texttt{seq\_len} \times \texttt{grad\_accum\_steps}$, so
    $$
    \texttt{grad\_accum\_steps} = \frac{262{,}144}{4 \times 1024}
    = \frac{262{,}144}{4{,}096} = 64.
    $$
    Check: $4 \times 1024 \times 64 = 262{,}144$ tokens/step — exactly the target.

    **(b) `total_steps`.** As in the chapter, take the ceiling of budget over tokens-per-step:
    $$
    \texttt{total\_steps} = \left\lceil \frac{2\times10^9}{262{,}144} \right\rceil
    = \lceil 7629.39\ldots \rceil = 7630.
    $$
    So `grad_accum_steps = 64` and `total_steps = 7630`. (For comparison, the flagship values are
    the same computation with a 524,288-token batch and a 20B-token budget, giving 38,147 steps.)

**4.** During a flagship A100 run you measure a full-optimizer-step time of `dt = 1.8s` (the 8
accumulated micro-batches plus clip and Muon+AdamW update). Using the chapter's constants
($N \approx 101.4\times10^6$ parameters, effective batch 524,288 tokens, A100 bf16 peak
$312\text{ TFLOP/s}$, the **6ND** rule), compute (a) tokens/sec, (b) achieved FLOP/s and MFU, and
(c) the projected wall-clock GPU-hours for the full $20\times10^9$-token run. Is this a plausible
figure for a well-optimized loop?

??? note "Solution"
    **(a) Tokens/sec.**
    $$
    \frac{524{,}288 \text{ tokens}}{1.8\text{ s}} \approx 291{,}271 \text{ tokens/s}.
    $$

    **(b) Achieved FLOP/s and MFU.** By 6ND, forward+backward costs $\approx 6N$ FLOPs per token:
    $$
    6 \times 101.4\times10^6 \times 291{,}271 \approx 1.772\times10^{14}\text{ FLOP/s}
    = 177.2\text{ TFLOP/s}.
    $$
    $$
    \text{MFU} = \frac{177.2}{312} \approx 0.568 = 56.8\%.
    $$

    **(c) Projected GPU-hours.**
    $$
    \frac{20\times10^9}{291{,}271 \text{ tokens/s}} \approx 68{,}665\text{ s}
    \approx 19.1 \text{ GPU-hours}.
    $$

    **Plausibility.** Yes: ~57% MFU sits near the top of the chapter's "well-tuned setups can push
    higher" range (it cites 55–65% as the target with `torch.compile`, fused optimizer kernels, and
    FlashAttention-backed SDPA), and 19.1 GPU-hours lands at the *lower* end of the ~15–25 GPU-hour
    envelope — consistent with a faster step time (1.8s vs. the un-optimized worked example's 2.3s /
    44.5% MFU / 24.4 GPU-hours).

**5.** The chapter argues for keeping several rolling checkpoints rather than overwriting a single
`latest.pt`, "in case the most recent one turns out to be corrupted or was written during a loss
spike you would rather roll back past." Implement `save_rolling_checkpoint(...)` that writes a
step-stamped checkpoint (reusing the chapter's `save_checkpoint`), then prunes so that only the
`keep_last` most recent step checkpoints survive. Preserve the chapter's atomicity guarantee and do
not let the prune delete a `final.pt`.

??? note "Solution"
    Reuse `save_checkpoint` (which already writes to a `.tmp` file and `os.replace`s it atomically),
    add a step-stamped name so checkpoints sort chronologically, then delete all but the newest
    `keep_last`. Globbing only `step_*.pt` naturally excludes `final.pt` from the prune.

    ```python
    import os, glob

    def save_rolling_checkpoint(model, optimizer, step, tokens_seen,
                                train_loader, cfg, keep_last=5):
        # Zero-padded step so lexical sort == chronological sort.
        path = f"{cfg.ckpt_dir}/step_{step:07d}.pt"
        save_checkpoint(path, model, optimizer, step, tokens_seen, train_loader, cfg)

        # Prune: keep only the newest `keep_last` step checkpoints. The glob pattern
        # matches only step_*.pt, so final.pt (and latest.pt) are never candidates.
        ckpts = sorted(glob.glob(f"{cfg.ckpt_dir}/step_*.pt"))
        for old in ckpts[:-keep_last]:
            os.remove(old)
        return path
    ```

    Notes on correctness: (1) atomicity is inherited unchanged from `save_checkpoint`'s
    temp-file-plus-`os.replace`, so an interrupted write never leaves a corrupt `step_*.pt` on disk;
    (2) `ckpts[:-keep_last]` is empty while fewer than `keep_last` checkpoints exist, so nothing is
    deleted early in the run; (3) the seven-digit zero pad (`{step:07d}`) makes lexical sort agree
    with numeric order up to 9,999,999 steps — well past the ~38,147 this run needs. With
    `keep_last=5` plus a separate `final.pt`, total checkpoint disk is ~6 GB at the chapter's
    ~0.9 GB per checkpoint.

**6.** Scale-out (implementation). The DDP snippet in the chapter wraps only the *last* micro-batch
in the accumulation window in a synchronizing context and the earlier ones in `model.no_sync()`.
Rewrite the flagship accumulation loop from `train_step` into a `ddp_train_step` that (i) skips the
gradient all-reduce on every micro-batch except the last, and (ii) still produces the correct
**mean** gradient over the effective batch. Then explain in one or two sentences why the
`no_sync()` optimization changes wall-clock but not the resulting gradient.

??? note "Solution"
    DDP triggers a gradient all-reduce as each `.backward()` completes. During accumulation we only
    want *one* all-reduce per optimizer step — after the final micro-batch — so we wrap the earlier
    backward passes in `model.no_sync()`, which suppresses the collective. The loss is still divided
    by `grad_accum_steps` before every `.backward()` so the local accumulated gradient is the mean;
    the single all-reduce on the last step then averages those per-rank means across ranks.

    ```python
    import contextlib

    def ddp_train_step(model, optimizer, micro_batches, cfg):
        """One optimizer step under DDP: all-reduce gradients exactly once,
        after the final accumulated micro-batch, and still average correctly."""
        assert len(micro_batches) == cfg.grad_accum_steps
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        last = len(micro_batches) - 1
        for i, (x, y, doc_mask) in enumerate(micro_batches):
            x, y, doc_mask = x.to(cfg.device, non_blocking=True), \
                             y.to(cfg.device, non_blocking=True), \
                             doc_mask.to(cfg.device, non_blocking=True)
            # Suppress the DDP all-reduce on every micro-batch except the last.
            sync_ctx = contextlib.nullcontext() if i == last else model.no_sync()
            with sync_ctx:
                with autocast_ctx:
                    logits = model(x, doc_mask=doc_mask)
                    ce = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)).float(),
                        y.view(-1), ignore_index=-1)
                    z_loss = (torch.logsumexp(logits.float(), dim=-1) ** 2).mean()
                    loss = (ce + cfg.z_loss_coef * z_loss) / cfg.grad_accum_steps
                loss.backward()
            total_loss += loss.item()
        return total_loss
    ```

    **Why it changes wall-clock but not the gradient.** Suppressing the all-reduce on the first
    `grad_accum_steps - 1` backward passes only defers *communication*; the local `.grad` buffers
    still accumulate every micro-batch's contribution locally. Because gradients add linearly and
    all-reduce (sum/mean) is linear, reducing once over the summed local gradients is
    mathematically identical to reducing after each micro-batch — you just do it once instead of
    `grad_accum_steps` times, saving `grad_accum_steps - 1` collectives per optimizer step (the
    chapter's stated motivation: DDP would "otherwise synchronize grad_accum_steps times per
    optimizer step for no benefit").
