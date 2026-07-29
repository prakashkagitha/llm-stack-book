# 14.7 The Pretraining Run: A Complete Single-GPU Training Loop

Every chapter so far has produced one *piece* of Stack-100M: the packed data shards (Ch. 14.2), the
tokenizer (Ch. 14.3), the model definition (Ch. 14.4), a fitted scaling law that justified the
final size and token budget (Ch. 14.5), and the Muon+AdamW optimizer plus the WSD schedule
(Ch. 14.6). This chapter wires all of it into the one artifact that actually spends the GPU-hours:
`train.py`.

By the end of this chapter you will have a complete, resumable, single-GPU training loop that takes
`Stack100M`, `PackedMemmapDataset`, the Muon+AdamW pair, and the WSD schedule, and turns them into
a ~20 GPU-hour job on one A100 that ends with `ckpt_stable.pt` — a real ~100M-parameter language
model, deliberately left *undecayed* so that [mid-training](../14-capstone/08-mid-training.html) can
spend the decay phase on premium data. We will do the memory accounting honestly (including the
tensor that actually dominates it, which is not the one most people count), measure *how well* the
GPU is being used (model FLOPs utilization, not `nvidia-smi`'s misleading utilization percentage),
make the run crash-safe against both preemption and NaNs, and give configs for the three compute
tiers the capstone supports: the flagship A100, a consumer RTX 4090, and a free Colab T4. We close
with a scale-out note — DDP, then FSDP — for readers who want to go faster than one GPU, while being
explicit that **none of that is required** to train Stack-100M; a single A100 is sufficient for the
whole plan.

## From Components to a Running Job

Training loops fail in boring, expensive ways: a crash three hours before the end with no
checkpoint, a silent NaN nobody notices until the next morning, a batch-size bug that quietly
halves the effective learning rate, an out-of-memory error at step 12,000 because the *one* tensor
you never budgeted for grew with the batch, a forgotten keyword argument that lets the model attend
across document boundaries for 17 billion tokens. The job of this chapter is to make each of those
failure modes structurally impossible, not just "usually fine."

Here is the data flow we are wiring together — the shape every subsection below fills in:

```text
 PackedMemmapDataset (Ch 14.2)               Stack100M (Ch 14.4)
 uint16 .bin shards  ───────────────────►  30 pre-norm blocks (bf16 autocast)
        │                                            │  seq_ids -> no cross-doc mask
        │ {input_ids, position_ids,                  │  position_ids -> RoPE index
        │  seq_ids, targets}                         ▼  final RMSNorm -> hidden (B,T,512)
        │                                    fused_ce_z_loss (loss_chunk=8192):
        │                                    loss = CE + z_coef·logsumexp²  (fp32, chunked)
        │                                            │
        │                                            ▼ backward (autocast region)
        │                                    accumulate grads over 8 micro-batches
        │                                            │
        │                                            ▼
        │                          clip_grad_norm_(all params, 1.0) ──► finite?
        │                                            │            │no
        │                                            │yes         └─► skip step, log, continue
        │                                            ▼
        │                    muon.step(); adamw.step()  ← lr = wsd_lr(step) (Ch 14.6)
        │                    every 200 steps: probe forward -> qk_clip_(model, record, τ=30)
        │                                            │
        │                                            ▼
        │                         every N steps: eval on held-out shard,
        │                         sample generation, checkpoint (model+opts+step+rng)
        ▼                                            │
  step timer + data_wait  ─────────────────────────► MFU, HFU, tokens/s
```

Everything downstream of the model and data already exists by the time this chapter starts; our job
is the loop, not the components. The interfaces below are the *canonical* ones fixed by the earlier
capstone chapters — you will not find their bodies here, only the contracts this chapter's code
relies on. Read them carefully: three of the four bugs that make a loop *silently* wrong rather than
loudly broken are contract violations at exactly these boundaries.

```python
# stacklm/config.py   (Ch. 14.4 / PLAN.md §1 — the frozen architecture)
@dataclass
class StackConfig:
    vocab_size: int = 32768       # byte-level BPE (Ch. 14.3)
    d_model: int = 512            # narrow: the "thin" in deep-and-thin
    n_layers: int = 30            # deep
    n_heads: int = 8
    n_kv_heads: int = 2           # GQA, 4:1
    head_dim: int = 64
    intermediate: int = 1408      # SwiGLU
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    qk_norm: bool = True          # RMSNorm on Q,K -- decides which QK-clip path runs
    nope_every: int = 4           # every 4th layer skips RoPE (SmolLM3-style)
    norm_eps: float = 1e-5
    z_loss_coef: float = 1e-4     # PaLM-style logsumexp penalty
    logit_soft_cap: float = 0.0   # optional Gemma-2 tanh cap; 0 = off
    loss_chunk: int = 0           # >0 = chunked fused lm_head+CE. This chapter sets 8192.

# stacklm/model/transformer.py   (Ch. 14.4 — RMSNorm, RoPE+NoPE, GQA, QK-norm, SwiGLU, tied emb)
class Stack100M(torch.nn.Module):
    cfg: StackConfig
    lm_head: torch.nn.Linear      # weight tied to tok_emb.weight
    def forward(self,
                idx:          "LongTensor[B, T]",
                targets:      "LongTensor[B, T] | None" = None,
                position_ids: "LongTensor[B, T] | None" = None,   # RoPE index, per token
                seq_ids:      "LongTensor[B, T] | None" = None,   # document id, per token
                kv_cache=None, start_pos: int = 0, logits_to_keep: int = 0,
                record: "dict[int, FloatTensor[n_heads]] | None" = None,
                ) -> "tuple[FloatTensor[B, T, V] | None, FloatTensor[] | None]": ...
    def generate(self, idx, max_new_tokens=64, temperature=0.8, top_p=0.95,
                 top_k=0, eos_id=None, use_cache=True) -> "LongTensor[B, T']": ...
    def num_params(self, non_embedding: bool = False) -> int: ...

# stacklm/data/dataset.py   (Ch. 14.2 — streaming, dedup, packing, document-aware masking)
class PackedMemmapDataset(torch.utils.data.Dataset):
    """Map-style. __getitem__(i) -> FOUR LongTensors of length seq_len - 1:
    input_ids, position_ids, seq_ids, targets.  `position_ids` restarts at 0
    inside each document; `seq_ids` is that document's index in the window. Both
    are derived on the fly from `input_ids == bos_id`, where `bos_id` comes from
    the shard directory's manifest.json (or is passed explicitly)."""
    def __init__(self, shard_dir: str, bos_id: int | None = None): ...

# stacklm/optim/   (Ch. 14.6 — Muon+AdamW hybrid, WSD schedule, MuonClip/QK-clip)
def build_optimizers(model, muon_lr=6e-3, adamw_lr=3e-3,
                     weight_decay=0.1, betas=(0.9, 0.95)) -> "tuple[Muon, torch.optim.AdamW]": ...
def wsd_lr(step: int, *, peak_lr: float, warmup_steps: int, total_steps: int,
           decay_steps: int | None = None, decay_frac: float = 0.2,
           final_frac: float = 0.0) -> float: ...
def qk_clip_(model, max_logits: "dict[int, FloatTensor[n_heads]]",
             tau: float = 30.0) -> int: ...          # returns #layers that fired

# stacklm/tokenizer/bpe.py   (Ch. 14.3 — byte-level BPE, vocab_size = 32768)
class StackTokenizer:
    bos_id: int   # 32759   the nine specials ALWAYS occupy the final nine ids,
    eos_id: int   # 32760   so pad_id is 32761, not some small number
    pad_id: int   # 32761
    @classmethod
    def load(cls, path: str) -> "StackTokenizer": ...
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
```

!!! warning "Common pitfall: the three keyword arguments that fail silently"

    Every one of these produces a run that trains, logs a falling loss, and is wrong.

    1. **Dropping `seq_ids`.** `Stack100M._build_mask` builds the no-cross-document mask from
       `seq_ids`, *not* from `position_ids`. Pass `seq_ids=None` and the model takes its
       plain-causal fast path, so every packed window lets token 1900 of document C attend to
       document A — for the entire 17-billion-token run. PLAN.md §2 mandates document isolation
       precisely because this leaks statistics across unrelated texts. It is invisible in the loss
       curve (it *lowers* training loss slightly, by leaking context), so assert it at startup.
    2. **Dropping `position_ids`.** Then RoPE indexes `arange(T)` and every document after the
       first in a window is told it starts at position 700, not 0.
    3. **Guessing `pad_id`.** Ch. 14.2's packer pads only the final window of each shard, with
       `tokenizer.pad_id = 32761`. If you hard-code a small "reserved special" id instead, the
       collate function rewrites a *common BPE merge token* to `-100` and deletes a slice of your
       supervision signal, while masking no actual pad. Derive `pad_id` from the tokenizer object;
       never type the integer.

!!! note "Aside: the one amendment this chapter makes to earlier code"

    Almost everything this loop needs already exists. Ch. 14.4 ships `fused_ce_z_loss` behind
    `StackConfig.loss_chunk`, threads `position_ids`/`seq_ids`/`record` through `forward`, and
    provides a KV-cached `generate`; Ch. 14.6 ships `build_optimizers`, `wsd_lr`, and `qk_clip_`.
    The single genuine addition is at the *data* boundary: **padding must map to `-100`.** The
    dataset hands back raw token ids, so the collate function rewrites `pad_id` targets to `-100`,
    the `ignore_index` that `fused_ce_z_loss` already honours. That is amendment enough; everything
    else in this chapter is composition.

    One bookkeeping note: `PackedMemmapDataset` shifts a stored 2048-token window by one to make
    `(input_ids, targets)`, so each sample supervises 2047 positions, not 2048. Every token count
    below rounds that to 2048; the 0.05% difference does not move any conclusion. If you want the
    accounting exact, widen the packer's window to 2049 tokens.

### The `TrainConfig`

One dataclass fixes every run-mechanics number so the script is fully reproducible from a single
object (which we also checkpoint, so a resumed run cannot silently drift from its original
configuration). Note what it does *not* contain: the architecture. `StackConfig`'s defaults already
*are* PLAN.md §1's frozen numbers, so we embed the object rather than restating its fields — one
source of truth, one fewer surface for two chapters to disagree on.

```python
# stacklm/train_config.py
from dataclasses import dataclass, field
from stacklm.config import StackConfig


@dataclass
class TrainConfig:
    # --- architecture: PLAN.md §1, frozen. Only `loss_chunk` differs from the
    #     model chapter's default (which is 0, so its tests can inspect logits).
    model: StackConfig = field(default_factory=lambda: StackConfig(loss_chunk=8192))

    # --- data: `data/packed/{train,val}` are the two subdirectories Ch. 14.2's
    #     build_corpus(out_dir="data/packed", ...) writes, each with a manifest.json
    shard_dir: str = "data/packed/train"
    val_shard_dir: str = "data/packed/val"
    tokenizer_path: str = "tokenizer/stack100m-32768.json"
    micro_batch_size: int = 32     # sequences per forward/backward pass
    grad_accum_steps: int = 8      # 32 * 2048 * 8 = 524,288 tokens/optimizer-step ≈ 0.5M
    num_workers: int = 4

    # --- optimizer & schedule: Ch. 14.6's frozen table, verbatim ---
    muon_peak_lr: float = 6e-3     # 2-D hidden matrices (RMS-matched Newton-Schulz update)
    adamw_peak_lr: float = 3e-3    # = muon/2: tied embedding + 1-D norm/QK-norm gains
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    qk_clip_tau: float = 30.0      # MuonClip threshold; 30 because QK-norm is ON
    qk_clip_every: int = 200       # logit drift is slow; measuring is not free
    qk_probe_seqs: int = 4         # sequences in the fixed QK-clip probe batch
    warmup_steps: int = 500
    total_steps: int = 38_147      # ceil(20e9 / 524,288) — the FULL budget, shapes the curve
    decay_steps: int = 6_000       # the WSD decay leg = Ch. 14.8's mid-training window
    stop_at_step: int = 32_147     # THIS chapter stops here: 16.9B tokens, LR still at plateau

    # --- run mechanics ---
    device: str = "cuda"
    activation_checkpointing: bool = False
    compile_model: bool = True
    eval_every: int = 500
    eval_iters: int = 50
    sample_every: int = 2000
    log_every: int = 10
    ckpt_every: int = 1000
    keep_last_ckpts: int = 5
    max_consecutive_skips: int = 20   # abort if the grad norm is non-finite this many times
    ckpt_dir: str = "checkpoints/stack-100m"
    seed: int = 1337

    @property
    def seq_len(self) -> int:
        return self.model.max_seq_len          # 2048 (see the 2047 note above)
```

Four things deserve a note.

**`grad_accum_steps`.** `micro_batch_size × seq_len × grad_accum_steps = 32 × 2048 × 8 = 524{,}288`,
which lands exactly on Ch. 14.6's target effective batch of ≈0.5M tokens. This is the number every
later section's arithmetic uses.

**The three step counts.** `total_steps = 38{,}147` comes from the capstone's ~20B-token budget
($20\times10^9 / 524{,}288$), which Ch. 14.5's fitted scaling law chose by deliberately
over-training past the ~2B-token Chinchilla-optimal point for
[Stack-100M](../14-capstone/05-mini-scaling-laws.html). It defines the *shape* of the WSD curve:
500 warmup / 31,647 stable / 6,000 decay, exactly Ch. 14.6's frozen split. **`stop_at_step =
32{,}147` is not a typo:** it is $38{,}147 - 6{,}000$. WSD's decay leg is where
[mid-training](../14-capstone/08-mid-training.html) anneals the model onto a higher-quality data
mix, so this chapter runs only warmup + stable — 16.85B tokens — and hands over a checkpoint whose
learning rate is still at its plateau, leaving 3.15B tokens of decay for Ch. 14.8. Saving a
*pre-decay* checkpoint is a deliberate design decision, not an accident: resuming from a
fully-decayed checkpoint would force an LR re-warm and cost you loss you then have to claw back
(Ibrahim et al., 2024; see Ch. 14.8).

**Two peak learning rates, one curve.** Ch. 14.6 routes 2-D hidden matrices to Muon and the tied
embedding plus every 1-D gain to AdamW. The two peaks are `6e-3` and `3e-3` — a **2:1 ratio, not an
order of magnitude**. That is the whole payoff of Muon's RMS-matching scale $0.2\sqrt{\max(m,n)}$:
it puts the orthogonalized update in the same decade as AdamW's, so you tune one number and derive
the other. (The factor of two is not RMS-related; it is insurance on the row-sparse embedding
gradient.) What the two groups *share* is the shape of the WSD schedule, and the loop below
preserves the ratio by storing each group's base LR once and multiplying by a single scalar.

**`loss_chunk = 8192`.** Ch. 14.4 ships the chunked fused loss head but leaves it off
(`loss_chunk = 0`) so that chapter's tests can inspect the logit tensor. Pretraining turns it on.
The next two sections explain why that single integer is the difference between a micro-batch of 8
and a micro-batch of 32.

## Precision, Batching, and the Effective Batch Size

### bf16 autocast, no loss scaler

We train in **bf16** (`bfloat16`), not fp16. bf16 keeps fp32's 8-bit exponent (same dynamic range)
and trims the mantissa to 7 bits, so unlike fp16 (5-bit exponent, prone to overflow on large
activations or gradients) it never needs
[`GradScaler`](../01-foundations/04-numerics-precision.html)-style dynamic loss scaling — the classic
source of "loss is `nan`, script has silently rescaled to zero and stalled." We keep the model's
master parameters in **fp32** and use `torch.autocast` only around the forward pass and loss
computation; PyTorch's autograd then produces gradients in the parameters' native fp32 dtype, which
matters at these small per-step learning rates where a pure-bf16 accumulator would round tiny
updates away. (Some frontier recipes skip the fp32 master copy entirely and train fully in bf16 for
the memory savings; at 101M params that saving is negligible. FP8 training — `torchao.float8`,
NVIDIA Transformer Engine — is a real 2026 option on Hopper/Blackwell and a non-option on the
hardware tiers this capstone targets; see
[Mixed Precision, bf16 & FP8 Training](../03-pretraining/08-mixed-precision-fp8.html).)

Two one-line settings belong next to the autocast context and are easy to forget:

```python
import torch

torch.backends.cudnn.benchmark = True          # fixed shapes every step -> algo cache is a win
torch.set_float32_matmul_precision("high")     # TF32 tensor cores for the fp32 ops that remain
                                               # (RMSNorm reductions, the fp32 optimizer math)

autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
```

### Gradient accumulation: micro-batch × accum = effective batch

{{fig:grad-accum-effective-batch}}

An 80GB A100 could fit a much larger micro-batch than 32 sequences of length 2048 for a 101M model,
but the *effective* batch size — the number of tokens averaged into one optimizer step — is what
the schedule and optimizer in Ch. 14.6 were tuned around (≈0.5M tokens, near the critical batch size
for a model this size). We reach that effective batch by accumulating gradients over
`grad_accum_steps` micro-batches before every optimizer step. This is the only place in the chapter
where the forward/backward pass is written out, and `train.py` below calls it rather than inlining
a second copy:

```python
# stacklm/train_utils.py
import time
import torch

BATCH_KEYS = ("input_ids", "targets", "position_ids", "seq_ids")


def accumulate(model, optimizers, train_iter, cfg):
    """One optimizer step's worth of forward/backward: cfg.grad_accum_steps
    micro-batches, gradients averaged.

    Returns (mean loss as a *device* tensor, seconds blocked on the loader). The
    micro-batch fetch is timed here, inside the step, on purpose -- see the MFU
    section: a loop that prefetches before starting the clock reports pure compute
    time and hides loader stalls completely."""
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)     # zero ONCE per optimizer step, not per micro-batch
    loss_sum = torch.zeros((), device=cfg.device)
    data_wait = 0.0

    for _ in range(cfg.grad_accum_steps):
        t_fetch = time.perf_counter()
        batch = next(train_iter)            # blocks here if the loader is behind
        data_wait += time.perf_counter() - t_fetch
        x, y, pos, seq = (batch[k].to(cfg.device, non_blocking=True) for k in BATCH_KEYS)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            # seq_ids is what makes the mask block-diagonal. Omit it and you train
            # with cross-document attention and never find out.
            _, loss = model(x, targets=y, position_ids=pos, seq_ids=seq)
            loss = loss / cfg.grad_accum_steps          # scale BEFORE backward
        loss.backward()
        loss_sum += loss.detach()           # stays on-device; no host sync per micro-batch

    return loss_sum, data_wait              # loss already averaged over the effective batch
```

Note the `/ cfg.grad_accum_steps` scaling *inside* the loop, before `.backward()` — because
gradients simply add across accumulated backward passes, dividing the loss (and hence its gradient)
by the accumulation count is what makes the accumulated gradient equal the *mean* gradient over the
full 524,288-token batch, matching what a single giant batch would have produced.

Note also what is *not* there: a `loss.item()` per micro-batch. `.item()` copies a scalar from
device to host, which forces a full CUDA synchronization and stalls the launch queue eight times
per optimizer step. Accumulating a detached device tensor and reading it once per step costs
nothing and is one of the cheapest throughput wins in the whole loop.

!!! warning "Common pitfall: zeroing gradients inside the accumulation loop"

    A very easy bug: calling `zero_grad()` on every micro-batch instead of once per optimizer step.
    That silently turns "8 micro-batches accumulated into one 524k-token step" into "8 independent,
    tiny 65k-token steps at 1/8th the intended effective batch size" — the loss curve still goes
    down, so it is easy to miss, but the run no longer matches the WSD schedule or the Muon
    hyperparameters tuned in Ch. 14.6, and throughput/MFU numbers become meaningless because the
    accounting no longer matches reality. Zero once, accumulate `N` times, then step. This is not a
    hypothetical: Unsloth's widely-reproduced 2024 write-up documents the mirror-image bug (a
    missing `1/N`) shipping in production trainers with a measurable loss-curve gap.

## The Loss Head Is the Memory Budget

Ask a practitioner which tensor dominates memory when training a 100M model and you will usually
hear "the activations." For Stack-100M that answer is wrong, and it is wrong in a way that is
specific to the **deep-and-thin** shape PLAN.md §1 mandates: `d_model = 512` with `vocab_size =
32768` means the output projection is 64× wider than the residual stream, so the logits tensor
dwarfs everything the transformer trunk holds. This is why `StackConfig.loss_chunk` exists, and this
section is the arithmetic that justifies turning it on.

Do the arithmetic on one micro-batch of $B \times T = 32 \times 2048 = 65{,}536$ tokens with
$V = 32{,}768$, for the *unchunked* path (`loss_chunk = 0`):

| Tensor | dtype | bytes | size |
|---|---|---|---|
| logits `(B·T, V)` | bf16 | 2 | $65{,}536 \times 32{,}768 \times 2 = 4.29$ GB |
| fp32 upcast of the logits for cross-entropy | fp32 | 4 | 8.59 GB |
| `log_softmax` output, saved for backward | fp32 | 4 | 8.59 GB |
| a *second* fp32 copy if the z-loss recomputes `logsumexp` | fp32 | 4 | 8.59 GB |
| **loss head, peak** | | | **≈ 30 GB** |

Compare that to the entire 30-block trunk, which we cost out at 15–25 GB below. The loss head is
the largest allocation in the job by a comfortable margin, and it scales with $B \cdot T \cdot V$ —
so every attempt to raise the micro-batch for throughput runs into it first. Worse, activation
checkpointing does **nothing** about it: the head sits outside the blocks.

Ch. 14.4's unchunked branch is written the standard way, and is therefore as bad as it can be:

```python
# Stack100M.forward, the loss_chunk == 0 branch -- correct, and memory-hungry at B*T = 65,536.
logits = self._cap(self.lm_head(x))                            # (B, T, V) bf16   4.3 GB
lf = logits.float()                                            # fp32 copy        8.6 GB
ce = F.cross_entropy(lf.view(-1, lf.size(-1)),                 # + log_softmax    8.6 GB
                     targets.reshape(-1), ignore_index=-100)
logz = torch.logsumexp(lf, dim=-1)                             # z-loss reuses lf, but
loss = ce + self.cfg.z_loss_coef * (logz ** 2).mean()          # a naive version would copy again
```

### Chunked linear cross-entropy: never materialize `(B·T, V)`

The fix is the standard 2026 one: fuse the `lm_head` matmul and the loss into a single operation
that processes tokens in chunks, so only `chunk × V` logits exist at any moment, and recompute
those logits during backward rather than storing them. This is the body of the `fused_ce_z_loss`
Ch. 14.4 ships:

```python
# stacklm/model/loss.py   (Ch. 14.4; enabled by StackConfig.loss_chunk > 0)
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


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
```

which `Stack100M.forward` selects with two lines:

```python
if targets is not None and self.cfg.loss_chunk > 0:
    ce, zl = fused_ce_z_loss(x, self.lm_head.weight, targets, self.cfg.z_loss_coef,
                             chunk=self.cfg.loss_chunk, soft_cap=self.cfg.logit_soft_cap)
    return None, ce + zl        # training path never materializes the logit tensor
```

Three things to understand about why this works.

**Memory.** With `loss_chunk = 8192` and $V = 32{,}768$, one chunk's fp32 logits are
$8192 \times 32768 \times 4 = 1.07$ GB. During the forward pass `checkpoint` discards them
immediately; during backward exactly one chunk is rebuilt at a time, so peak head memory is roughly
the logits plus their gradient — about **2.1 GB**, a ~14× reduction, and *independent of
`micro_batch_size`*, which is the property that makes larger micro-batches possible at all. Halving
to `loss_chunk = 4096` halves that again (≈1.1 GB) at the cost of twice as many kernel launches;
8192 is Ch. 14.4's shipped default and the value we use.

**Compute.** The price is recomputing the `lm_head` matmul in backward:
$2 \cdot B \cdot T \cdot d \cdot V = 2 \times 65{,}536 \times 512 \times 32{,}768 \approx 2.20$
TFLOP per micro-batch, against a step cost of
$6ND = 6 \times 101.4\times10^6 \times 65{,}536 \approx 39.9$ TFLOP. That is **≈5.5% extra FLOPs**
— cheap for a 14–30× memory cut, and cheaper still with a fused kernel that avoids the recompute
entirely (below). The percentage does not depend on `chunk`.

**Numerics.** The z-loss reuses the *same* `logsumexp` the cross-entropy already needs. That is not
just a memory win: it guarantees the two terms see identical logits. Note one deliberate behavioural
difference from a naive implementation — both terms are averaged over *valid* (non-ignored)
positions rather than over all positions, which is the more defensible normalization anyway.

!!! tip "Practitioner tip: use the fused kernel in production"

    You now understand exactly what the library does, which is the point of writing it out. In a
    real run, reach for a fused Triton implementation that skips the recompute and the fp32
    materialization altogether:

    - **[Liger-Kernel](https://github.com/linkedin/Liger-Kernel)** (LinkedIn, 2024) —
      `LigerFusedLinearCrossEntropy` fuses the `lm_head` matmul with cross-entropy and computes the
      gradient in-place chunk by chunk. Recent versions expose an `lse_square_scale` argument that
      folds the z-loss into the same kernel; check your version's signature before wiring it up.
    - **[cut-cross-entropy](https://github.com/apple/ml-cross-entropy)** (Apple; Wijmans et al.,
      *Cut Your Losses in Large-Vocabulary Language Models*, 2024) — computes the loss without ever
      writing a global logit matrix, using a flash-attention-style online reduction over the vocab.
    - **torchtune** ships a chunked-output cross-entropy for the same reason, and
      **torchtitan**/**nanotron** chunk over the sequence dimension in their pretraining loops.

    All of them exist because of the arithmetic in the table above: at large vocabularies the loss
    head, not attention, is what caps your batch size.

!!! example "Worked example: the whole memory budget, one micro-batch"

    $B \times T = 32 \times 2048 = 65{,}536$ tokens, bf16 activations, 30 blocks, with
    [FlashAttention](../04-kernels-efficiency/03-flash-attention-2-3.html)-backed SDPA (no
    materialized $T \times T$ score matrix).

    **Weights, gradients, optimizer state** (these do *not* scale with batch):
    fp32 master weights $101.4\times10^6 \times 4 \approx 0.41$ GB; fp32 gradients another
    0.41 GB; Muon's single momentum buffer over the ~84.5M 2-D block parameters $\approx 0.34$ GB;
    AdamW's $m$ and $v$ over the remaining ~16.8M embedding/norm parameters $\approx 0.13$ GB.
    **Total ≈ 1.3 GB** — genuinely negligible, which is the whole point of a 100M model.

    **Block activations saved for backward** (eager mode): the residual-stream input at each of the
    30 blocks is $65{,}536 \times 512 \times 2\text{ B} \times 30 \approx 2.0$ GB; the SwiGLU
    sublayer saves its 1408-wide intermediates (gate pre-activation, up projection, and their
    product are all needed for the backward of `down(silu(gate)·up)`), so budget
    $2\text{–}3 \times 65{,}536 \times 1408 \times 2\text{ B} \times 30 \approx 11\text{–}17$ GB;
    attention Q/K/V/O projections, QK-norm, and RoPE outputs add a few GB more. Call the trunk
    **15–25 GB**; `torch.compile`'s fusion recomputes some elementwise intermediates and typically
    lands nearer the low end.

    **Loss head:** ≈30 GB at `loss_chunk = 0`, ≈2.1 GB at `loss_chunk = 8192`.

    **Verdict.** Unchunked, peak is roughly $1.3 + 20 + 30 \approx 50$ GB. It *fits* on an 80 GB
    A100 — so "the flagship tier needs no activation checkpointing" stays true — but with far less
    headroom than an activations-only estimate suggests, and it is why raising `micro_batch_size` to
    64 OOMs. Chunked, peak is roughly $1.3 + 20 + 2 \approx 23$ GB and a micro-batch of 64 becomes
    comfortable. On the 24 GB 4090 tier the difference is decisive rather than convenient: at
    `micro_batch_size = 8` the unchunked head alone costs
    $16{,}384 \times 32{,}768 \times 14\text{ B} \approx 7.5$ GB — a third of the card.

    Always confirm on your own hardware rather than trusting the table:
    `torch.cuda.max_memory_allocated() / 2**30` after a few steps, and
    `torch.cuda.reset_peak_memory_stats()` to re-arm it.

## Clipping, the WSD Schedule, and QK-Clip

This section wires in the three per-step control signals built in
[the optimizer chapter](../14-capstone/06-optimizer-and-schedule.html) — the global gradient clip,
the Warmup-Stable-Decay learning rate (see
[Learning Rate Schedules, Warmup, Batch Size & Hyperparameters](../03-pretraining/10-lr-schedules-hparams.html)
for the general theory), and MuonClip.

After the accumulated backward pass, we clip the **global** gradient norm across all parameters
(not per-tensor) to `grad_clip = 1.0`, matching Ch. 14.6:

$$
\hat g = g \cdot \min\!\left(1,\; \frac{c}{\lVert g \rVert_2 + \epsilon}\right), \qquad c = 1.0
$$

### One clip, two optimizers

`build_optimizers` (Ch. 14.6) does not hand every parameter to the same update rule: 2-D hidden
weight matrices (attention Q/K/V/O projections, the SwiGLU up/gate/down projections) go to Muon's
Newton-Schulz-orthogonalized momentum update, while the tied embedding table and every 1-D tensor
(RMSNorm and QK-norm gains) go to AdamW — Muon's orthogonalization is only defined for matrices,
and 1-D parameters have no meaningful spectral structure to orthogonalize. Because that returns
*two* optimizer objects, the loop has to be explicit about three things: which parameters the clip
covers, how one schedule drives two different base learning rates, and the order of `step()` calls.

```python
from stacklm.optim import wsd_lr


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
```

Calling `wsd_lr` with `peak_lr=1.0` turns it into a pure multiplier in $[0, 1]$ — warmup ramps it
linearly from $1/500$ to 1 over the first 500 steps, the stable phase holds it at 1, and the decay
leg would bring it to 0. That single scalar scales Muon's `6e-3` and AdamW's `3e-3` identically. Two
consequences worth stating: passing `decay_steps=6_000` explicitly (rather than a `decay_frac`) is
what makes the decay leg *absolute*, so it stays 6,000 steps even if you re-budget `total_steps`;
and because this chapter stops at 32,147 — the first decay step — `mult` is exactly 1.0 for every
step from 500 to 32,146. This run never enters the decay branch at all. Ch. 14.8 does.

Crucially, `clip_grad_norm_` is computed **once, jointly, over every parameter** — Muon-bound and
AdamW-bound alike — because the point of global-norm clipping is to catch a *model-wide* gradient
explosion (a bad batch, a numerical instability in one deep layer propagating backward) regardless
of which optimizer will consume which slice of it; clipping each group separately would let one
group's blow-up hide behind the other's normal-sized gradients. `clip_grad_norm_` returns the
*pre-clip* norm, which is worth logging: a sudden multi-sigma jump in `grad_norm` — clipped away or
not — is often the earliest warning of an instability, well before it shows up in the loss (see
[Training Stability, Loss Spikes & Debugging Large Runs](../03-pretraining/11-training-stability.html)).

### MuonClip on a probe batch, every 200 steps

Ch. 14.6's `qk_clip_` reads a **`dict[int, Tensor(n_heads)]`** — layer index to that layer's per-head
maximum pre-softmax attention logit — and rescales whatever parameter actually controls the logit
scale for any layer that exceeded $\tau$. Under Stack-100M's QK-norm that parameter is the learned
`q_norm`/`k_norm` gain, not $W_Q/W_K$ (RMSNorm is scale-invariant, so rescaling the projections is
provably a no-op); Ch. 14.6 derives this, and it is why $\tau = 30$ rather than Kimi K2's 100.

Three implementation decisions follow, and all three differ from the naive "record every forward,
clip every step" reading:

```python
from stacklm.optim import qk_clip_


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
```

**Why a probe forward rather than the training forward.** `qk_clip_` must see the weights the
optimizers just produced, so the reading has to come from a *post-step* forward. Harvesting it from
the eight micro-batches of the step that just ended would give you pre-step weights, and — because
`Attention.forward` *assigns* into `record` rather than max-reducing — only the last micro-batch's
value at that. One extra forward pass on a small batch is both correct and cheap.

**Why a fixed probe batch.** We draw the probe once, from the validation loader, and reuse it. That
keeps the training stream untouched (so the "data position = `step × samples_per_step`" arithmetic
in the resume section stays exact) and makes the recorded $S_{\max}$ series comparable step to step
— a rising trend on *fixed* inputs is unambiguously the model drifting, not the data changing.

**Why every 200 steps.** Attention-logit drift is slow: the gains move by one decayed learning rate
per step. Measuring 38,147 times to catch a phenomenon that evolves over thousands of steps is pure
overhead, so Ch. 14.6 gates it behind `qk_clip_every = 200` (~160 measurements across this run). By
default `Attention` records the **Cauchy–Schwarz bound**
$\max_i\lVert q_i\rVert \cdot \max_j\lVert k_j\rVert / \sqrt{d_h}$ rather than the exact max, which
is $O(BHTd_h)$ and keeps the fused SDPA path; setting `attn.record_exact = True` gives the exact
per-head max at the cost of a $(B,H,T,T)$ tensor and eager attention.

Two more small things. Run the probe on `raw_model`, not the `torch.compile`d wrapper: writing into
a Python dict from inside a compiled forward is a graph break at best and a silently-baked-out
no-op at worst. And log the return value — `qk_clip_` reports how many layers fired, and a trigger
rate that *rises* late in training is the cleanest early signal that the learning rate is too high.

```python
# --- inside the main loop, once per optimizer step ---
lr_mult              = set_lr(optimizers, step, cfg)
loss_sum, data_wait  = accumulate(model, optimizers, train_iter, cfg)
grad_norm            = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)   # PRE-clip norm
for opt in optimizers:
    opt.step()                                       # Muon (Newton-Schulz), then fused AdamW
qk_fired = maybe_qk_clip(raw_model, probe, cfg, step)   # MuonClip: AFTER the step, on a schedule
```

Do not confuse MuonClip with the global grad-norm clip: they act on different objects (weights vs.
gradients) at different times, and they are complementary, not redundant.

## Memory: Activations, Attention Masks, and Checkpointing

{{fig:training-memory-budget}}

With the loss head chunked, activations are back to being the batch-scaling term. Two levers control
them: how much of the block you recompute, and whether your attention mask lets you use a fused
kernel at all.

### The document mask decides which attention kernel you get

`Stack100M._build_mask` turns `seq_ids` into an exact $(B, 1, T, T)$ boolean mask that blocks
cross-document attention. That mask is correct, and it is a throughput trap: passing an explicit
dense `attn_mask` to `torch.nn.functional.scaled_dot_product_attention` **disqualifies the
FlashAttention backend**, which supports only `is_causal=True` or no mask. SDPA silently falls back
to the memory-efficient or math backend, materializes more, and your MFU drops by a large factor
with no error message. Ch. 14.4 ships the dense mask because it is correct, dependency-free, and
composes with the KV cache; here is how you buy the FLOPs back.

```python
# Option A -- FlexAttention (PyTorch >= 2.5): a compiled, block-sparse mask.
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
```

```python
# Option B -- varlen FlashAttention: unpad into one long sequence + cumulative lengths.
from flash_attn import flash_attn_varlen_func    # Dao-AILab/flash-attention
# q, k, v: (total_tokens, n_heads, head_dim); cu_seqlens: int32 offsets of each document
out = flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                             max_seqlen_q=max_len, max_seqlen_k=max_len, causal=True)
```

FlexAttention is the lower-friction choice inside a PyTorch-native loop: `create_block_mask` should
itself be `torch.compile`d and its result cached per micro-batch (it costs a few hundred
microseconds, nothing against a 2-second step), and block-sparse skipping actually *recovers* most
of the FLOPs that document masking removes. The varlen path is what Megatron-LM-style stacks use and
is fastest if you are already unpadding. Either way, the rule to internalize is: **an explicit
boolean mask is the slow path**; express your mask as structure the kernel understands. See
[FlashAttention 2 & 3](../04-kernels-efficiency/03-flash-attention-2-3.html) for why, and Ch. 14.4
for the full `score_mod`/`enable_gqa` treatment.

### Activation checkpointing, correctly described

**Activation checkpointing** (Chen et al., 2016) trades recomputation for memory: instead of
storing every intermediate a block computes, we store only the block's *input* and recompute the
block's forward pass during backward. See
[Memory-Efficient Training: Checkpointing, Offloading & LoRA Math](../04-kernels-efficiency/10-memory-efficient-training.html)
for the full trade-off and the offloading levers we do not need at this scale.

It is worth being precise about what that saves, because the usual one-liner ("cuts activation
memory by a factor of `n_layers`") is wrong. Full-block checkpointing **keeps** the 30
block-boundary activations — the 2.0 GB residual-stream term above is exactly the set of tensors it
preserves — and **discards** everything computed *inside* each block, i.e. the 11–17 GB of SwiGLU
intermediates plus the attention temporaries, recomputing one block's worth at a time during
backward. So the block-activation total goes from roughly 15–25 GB to roughly 2.0 GB plus one
block's working set (a few hundred MB): a **3–8× cut**, not a 30× one.

```python
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

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
```

Two refinements matter at 2026 practice level. First, PyTorch ships this as a library function —
`torch.distributed.algorithms._checkpoint.checkpoint_wrapper.apply_activation_checkpointing` — which
you should prefer to a hand-rolled wrapper once you move to FSDP, because it composes with sharding.
Second, all-or-nothing checkpointing is not frontier practice: **selective** recompute (Korthikanti
et al., 2022) recomputes only the cheap-to-recompute, expensive-to-store sublayers — here, the
SwiGLU intermediates — while keeping the results of the expensive matmuls. PyTorch exposes this as
*selective activation checkpointing* via `torch.utils.checkpoint`'s
`create_selective_checkpoint_contexts` and a per-op `CheckpointPolicy`, and torchtitan uses it as its
default for exactly this reason. For Stack-100M the naive version is fine; for anything larger, the
selective version is close to free.

For the A100 flagship tier we leave checkpointing **off** by default — the extra ~33% compute
directly costs GPU-hours we would rather spend on tokens — and reserve it for the 24GB/16GB tiers,
where, note, `loss_chunk` buys more memory than checkpointing does.

## Crash-Safety: Checkpoints, Resume, and NaN Guards

A ~20 GPU-hour job on a rented A100 *will* occasionally be interrupted — a spot-instance reclaim, a
driver hiccup, you closing your laptop — and it may occasionally poison itself with a NaN. Both
need to be survivable.

### What the checkpoint must contain

Model weights, both optimizers' state (Muon's momentum buffers and AdamW's `m`/`v`), the step
counter (so the WSD schedule picks up exactly where it left off), the position in the data stream
(so we do not silently re-train on the same tokens), and every RNG state that affects what happens
next. This generalizes the pattern in
[Checkpointing, Fault Tolerance & Long-Running Jobs](../03-pretraining/12-checkpointing-fault-tolerance.html)
to our single-GPU case.

```python
# stacklm/checkpoint.py
import os, glob, random
import numpy as np
import torch
from dataclasses import asdict, is_dataclass


def _unwrap(model):
    """torch.compile wraps the module and renames its state_dict keys under
    `_orig_mod.*`; always save/load the *uncompiled* module's state dict so
    checkpoints remain loadable with or without compilation enabled."""
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def _rng_snapshot() -> dict:
    """Every RNG state, expressed ONLY as tensors and plain Python containers.

    This is deliberate: since PyTorch 2.6, `torch.load` defaults to
    weights_only=True, whose restricted unpickler rejects arbitrary objects --
    including the numpy ndarray that `np.random.get_state()` hides inside a
    tuple. Storing plain types keeps the safe loader working."""
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return {
        "torch":  torch.get_rng_state(),                                  # ByteTensor
        "cuda":   torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "np_keys": torch.from_numpy(keys.astype(np.int64)),               # 624 uint32 -> tensor
        "np_meta": [str(name), int(pos), int(has_gauss), float(cached)],
        "python": list(random.getstate()[1]),                             # 625 ints
    }


def _restore_rng(r: dict) -> None:
    torch.set_rng_state(r["torch"].cpu())
    if torch.cuda.is_available() and r["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in r["cuda"]])
    name, pos, has_gauss, cached = r["np_meta"]
    np.random.set_state((name, r["np_keys"].cpu().numpy().astype(np.uint32),
                         pos, has_gauss, cached))
    random.setstate((3, tuple(r["python"]), None))


def save_checkpoint(path, model, optimizers, *, step, tokens_seen=0,
                    config=None, **extra):
    """Atomic, resumable checkpoint. `optimizers` is the [muon, adamw] list from
    Ch. 14.6; `extra` carries anything a later stage needs (Ch. 14.8 passes
    `data_seed=`)."""
    if is_dataclass(config):
        config = asdict(config)          # nested StackConfig flattens too
    ckpt = {
        "model": _unwrap(model).state_dict(),
        "optimizers": [opt.state_dict() for opt in optimizers],
        "step": step,
        "tokens_seen": tokens_seen,
        "config": config,
        "rng": _rng_snapshot(),
        **extra,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)     # atomic rename on POSIX: never a half-written file


def load_checkpoint(path, model, optimizers, map_location="cuda") -> dict:
    """Returns the full checkpoint dict (Ch. 14.8's `mid_train` reads ckpt['step'])."""
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    _unwrap(model).load_state_dict(ckpt["model"])
    for opt, sd in zip(optimizers, ckpt["optimizers"]):
        opt.load_state_dict(sd)
    _restore_rng(ckpt["rng"])
    return ckpt


def save_rolling(model, optimizers, cfg, *, step, tokens_seen, keep_last=5):
    """Step-stamped checkpoint + prune. Also refreshes `latest.pt` so the resume
    path is a single well-known filename. WRITE first, PRUNE last."""
    path = f"{cfg.ckpt_dir}/step_{step:07d}.pt"
    save_checkpoint(path, model, optimizers, step=step,
                    tokens_seen=tokens_seen, config=cfg)
    save_checkpoint(f"{cfg.ckpt_dir}/latest.pt", model, optimizers, step=step,
                    tokens_seen=tokens_seen, config=cfg)
    for old in sorted(glob.glob(f"{cfg.ckpt_dir}/step_*.pt"))[:-keep_last]:
        os.remove(old)
    return path
```

Three details matter enough to call out.

**The atomic write.** `torch.save` directly to `path` risks a truncated, unloadable checkpoint if
the process dies mid-write (very plausible on a preemptible instance). Writing to a `.tmp` file and
`os.replace`-ing it is atomic on POSIX filesystems, so the on-disk checkpoint is always either the
previous complete one or the new complete one, never a half-written mixture.

**`weights_only=True` is not optional in 2026.** PyTorch 2.6 flipped the `torch.load` default to
`weights_only=True`, and the restricted unpickler it uses rejects the numpy ndarray buried inside
`np.random.get_state()`'s tuple. A checkpoint written the naive way raises `UnpicklingError` on your
first resume — that is, mid-run, on a GPU you are paying for by the hour. The snapshot above stores
only tensors and plain containers, so the safe loader works unmodified. If you must load a legacy
checkpoint, pass `weights_only=False` *and* understand that you are executing arbitrary pickled code
from that file; only ever do it for checkpoints you produced yourself. (The alternative is
`torch.serialization.add_safe_globals`, which allowlists specific types.)

**Beyond one GPU, use DCP.** `torch.distributed.checkpoint` (DCP) saves a *sharded* checkpoint in
parallel across ranks and supports asynchronous saves that overlap with compute — which matters when
a checkpoint is 200 GB rather than 1 GB. torchtitan uses it by default. At Stack-100M's size
`torch.save` is entirely adequate; know that DCP is where you go next.

The fp32 model state dict is $101.4\times10^6 \times 4\text{ B} \approx 406$ MB; Muon's single
momentum buffer over the ~84.5M 2-D block parameters is $\approx 338$ MB; AdamW's $m$ and $v$ over
the remaining ~16.8M parameters are $\approx 134$ MB. A full checkpoint lands **on the order of
0.9 GB**, so the last 5 plus `ckpt_stable.pt` cost under 6 GB — trivial. There is no reason to
overwrite a single `latest.pt` in place, and one strong reason not to: if the most recent checkpoint
was written *during* a loss spike, or after a NaN got into the weights, an in-place scheme has
destroyed the only good state you had.

### Resuming the data stream without a stateful loader

Here is where a lot of otherwise-correct loops quietly break. Ch. 14.2's `PackedMemmapDataset` is a
**map-style** dataset — `__len__` plus `__getitem__` — so the order in which samples are consumed is
decided by the sampler, not by the dataset. That gives us an exact, prefetch-proof resume: make the
order a deterministic function of a seed, and derive the position from `step`.

```python
# stacklm/train_data.py
import torch
from torch.utils.data import DataLoader, Sampler, default_collate
from stacklm.data.dataset import PackedMemmapDataset


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
```

and on resume:

```python
samples_per_step = cfg.micro_batch_size * cfg.grad_accum_steps      # 256
train_loader = build_loader(cfg.shard_dir, cfg, tok,
                            start_sample=step * samples_per_step)
```

!!! warning "Common pitfall: reading the data cursor out of the loader"

    It is tempting to ask the loader where it is and save *that*. Do not, for two reasons.

    **Prefetch skew.** With `num_workers > 0` and `prefetch_factor = 4`, the sampler in the main
    process has already handed out up to `num_workers × prefetch_factor` batches that the training
    loop has not consumed. A cursor read from the sampler therefore over-reports, and every resume
    silently skips a few thousand samples. Deriving the position from `step` is exact by
    construction — which is also why the QK-clip probe above draws from the *val* loader: an extra
    `next(train_iter)` every 200 steps would put the two out of sync by ~160 micro-batches.

    **Iterable datasets duplicate.** If you replace the map-style dataset with a streaming
    `IterableDataset` — a natural instinct for a shard-based corpus — each of the `num_workers`
    worker processes runs the *entire* `__iter__` unless the dataset shards on
    `torch.utils.data.get_worker_info()`. Every token is then seen `num_workers` times, and the
    dataset object you call `state_dict()` on in the main process is a copy that was never
    iterated, so its cursor is frozen at zero forever. Both bugs are invisible in the loss curve.
    If you do need a streaming loader, use
    **[`torchdata.stateful_dataloader.StatefulDataLoader`](https://github.com/pytorch/data)**, whose
    `state_dict()`/`load_state_dict()` round-trips per-worker iterator state correctly; it is what
    torchtitan uses for resumable pretraining.

### Non-finite gradients: skip the step, do not poison the run

A single bad micro-batch can produce an `inf` or `nan` gradient. If nothing checks for it,
`clip_grad_norm_` returns `nan`, `optimizer.step()` writes `nan` into every parameter *and* every
Muon momentum buffer, and the next scheduled checkpoint overwrites your good state with the poisoned
one. The whole run is then unrecoverable and the loss log looks fine right up to the point where it
becomes `nan`.

The guard is a handful of lines:

```python
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
```

An isolated skip is normal and harmless; a *run* of them means the damage is already in the weights,
and the correct response is the standard loss-spike recipe from
[Training Stability, Loss Spikes & Debugging Large Runs](../03-pretraining/11-training-stability.html):
stop, roll back to a step checkpoint from before the spike (which is why we keep five), skip the
data window that produced it by advancing `start_sample` past it, and resume. That recipe only works
if rolling checkpoints exist — which is exactly why `save_rolling` never overwrites its own history.

Note that `if torch.isfinite(grad_norm)` reads a CUDA scalar on the host and therefore synchronizes.
That costs one sync per optimizer step, which the timing code below performs anyway. It is not a
reason to skip the check.

!!! tip "Practitioner tip: rehearse the resume before you need it"

    Before trusting a checkpoint on a rented GPU you are paying for by the hour, dry-run the resume
    path locally: train for 20 steps, save, kill the process, resume, and compare the loss at step
    21 against a fresh run that never stopped. On a correctly wired loop these should match to
    within tight floating-point tolerance — the same op sequence on the same inputs with the same
    RNG state is deterministic, though exact bit-equality also requires
    `torch.use_deterministic_algorithms(True)` and `cudnn.benchmark = False`, which cost throughput
    and are worth turning on only for this test. If they diverge materially, something in the
    checkpoint is incomplete — almost always the data position or Muon's momentum buffers.

## Measuring Throughput: MFU, HFU, and Where 6ND Breaks

{{fig:mfu-vs-gpu-util}}

`nvidia-smi`'s "GPU-Util" percentage tells you the GPU was doing *something* during a sampling
window — it reads 100% just as happily whether you are running at 90% of peak matmul throughput or
15%. The metric that tells you whether you are getting your money's worth is **Model FLOPs
Utilization (MFU)**: the FLOPs the *model math* requires, divided by the accelerator's peak
FLOPs/s, popularized as a training-efficiency metric by Chowdhery et al. (PaLM, 2022).

Two things about how the loop measures it. First, `torch.cuda.synchronize()` before and after the
timed region is mandatory: CUDA kernel launches are asynchronous, so an un-synchronized wall clock
mostly measures how fast the CPU can enqueue work, not how fast the GPU executes it. Second — and
this is the part that is easy to get wrong — the micro-batch fetches must be **inside** the timed
region, which is why `accumulate` above times them and returns `data_wait`. A loop that prefetches
all eight micro-batches before starting the clock reports pure compute time and hides loader stalls
completely, making the MFU number useless for exactly the diagnosis MFU is best at. Logging
`data_wait` alongside `dt` turns "MFU is low" into "MFU is low *and* 40% of the step is
`next(train_iter)`", which is an actionable statement. When `data_wait` is non-trivial, raise
`num_workers` and `prefetch_factor`, or move packing work offline into the shard build.

### The 6ND rule, and where it stops being enough

The standard estimate is **6ND**, used throughout
[the scaling-laws chapter](../03-pretraining/04-scaling-laws.html) and
[Ch. 14.5](../14-capstone/05-mini-scaling-laws.html): one forward-plus-backward pass costs
approximately 6 FLOPs per parameter per token (Kaplan et al., 2020) — 2 for the forward matmuls, 4
for the backward's two matmuls per weight. It is a good approximation when the model's FLOPs are
dominated by weight matmuls. Attention's score and value-aggregation matmuls, though, involve no
weights at all: their cost scales with sequence length, and 6ND misses them entirely.

For a decoder-only transformer the fuller model-FLOPs-per-token count is

$$
F_{\text{token}} \;\approx\; \underbrace{6N}_{\text{weight matmuls}} \;+\;
\underbrace{6 \cdot n_{\text{layers}} \cdot T \cdot d_{\text{model}}}_{QK^\top \text{ and } PV,\ \text{causal}}
$$

where the second term is the PaLM paper's $12 \cdot L \cdot H \cdot Q \cdot T$ (with
$H \cdot Q = d_{\text{model}}$) halved, because a causal kernel computes only the lower triangle.
The ratio of the two terms is roughly $T / (6 \cdot d_{\text{model}})$ — negligible for a wide,
short-context model, and *not* negligible for Stack-100M, which PLAN.md §1 deliberately makes
**deep and thin**.

```python
# stacklm/train_utils.py
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
```

!!! example "Worked example: from step time to GPU-hours, both FLOP conventions"

    Suppose we measure `dt = 2.3 s` for one full optimizer step (8 accumulated micro-batches of
    32×2048 tokens, plus the clip and both optimizer steps), on a `torch.compile`d loop with
    FlashAttention-backed SDPA.

    **Tokens/sec:** $524{,}288 / 2.3 \approx 227{,}951$ tokens/s.

    **FLOPs per token, 6ND only:** $6 \times 101.4\times10^6 \approx 608.4$ MFLOP/token.
    Achieved: $608.4\times10^6 \times 227{,}951 \approx 1.387\times10^{14} = 138.7$ TFLOP/s.
    **MFU $= 138.7 / 312 \approx 44.5\%$** on an A100 80GB SXM (bf16 dense peak, no structured
    sparsity).

    **The attention correction:** $6 \times 30 \times 2048 \times 512 \approx 188.7$ MFLOP/token —
    a **31%** addition that 6ND silently drops. Total $\approx 797.1$ MFLOP/token, achieved
    $\approx 181.7$ TFLOP/s, **MFU $\approx 58.2\%$**.

    **Projected wall-clock.** Utilization changes; wall-clock does not, because it comes from
    tokens/s. This chapter's 32,147 steps are
    $32{,}147 \times 524{,}288 \approx 16.85\times10^9$ tokens:
    $$
    \frac{16.85\times10^9}{227{,}951} \approx 73{,}940\text{ s} \approx 20.5 \text{ GPU-hours},
    $$
    plus roughly 3.8 more GPU-hours for Ch. 14.8's 6,000 decay steps ($\approx 3.15\times10^9$
    tokens): **≈24.4 GPU-hours for the full 20B-token budget**. At roughly USD 1.50/GPU-hour that is
    about USD 37, inside the plan's ~USD 25–50 figure and mid-band of its 22–29 GPU-hour
    envelope. Reaching the *lower* end means pushing tokens/s up via a larger micro-batch (which
    `loss_chunk` now permits), `torch.compile`, and fused kernels — all "on the order of," never a
    guaranteed benchmark.

    **The lesson for reporting.** 44.5% and 58.2% describe the same run. Always state which FLOP
    convention you used; comparing your 6ND-only number against someone else's
    attention-inclusive number will make your loop look 30% worse than it is.

### MFU vs. HFU

There is a second distinction the same literature draws, and it is exactly why we care about
activation checkpointing (Korthikanti et al., 2022):

- **MFU** counts only the FLOPs the model *mathematically requires*. Recomputation is invisible to
  it: checkpointing does not change the math, so it can only lower MFU (by lowering tokens/s).
- **HFU** (hardware FLOPs utilization) counts every FLOP the hardware actually executed, including
  recomputed forward passes.

On the A100 tier, where checkpointing is off, MFU = HFU. On the 4090 and T4 tiers with full-block
checkpointing, each block's forward runs twice, so hardware FLOPs are ~4/3 of model FLOPs and
$\text{HFU} \approx 1.33 \times \text{MFU}$. A run showing 40% MFU and 53% HFU is not two numbers in
conflict: it is telling you the silicon is well-fed and one third of its work is being thrown away
to buy memory. Report MFU when you are asking "how efficiently am I turning dollars into a model,"
and HFU when you are asking "how efficiently am I driving the tensor cores."

### When the numbers disagree with you, profile

MFU tells you *that* something is wrong; it does not tell you *what*. The PyTorch profiler does, and
costs a handful of steps:

```python
from torch.profiler import profile, schedule, ProfilerActivity, tensorboard_trace_handler

prof = profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=5, warmup=3, active=5, repeat=1),   # skip compile + warmup steps
    on_trace_ready=tensorboard_trace_handler(f"{cfg.ckpt_dir}/trace"),
    record_shapes=True, profile_memory=True, with_stack=False,
)
# ... prof.start() before the loop, prof.step() at the end of each iteration, prof.stop() after.
```

Open the trace in TensorBoard or `chrome://tracing` and look for three signatures: long gaps on the
CUDA stream (you are launch-bound or loader-bound), a wall of tiny elementwise kernels
(`torch.compile` is not fusing — check for graph breaks with `TORCH_LOGS=graph_breaks`, and see
[Kernel Fusion, torch.compile, CUDA Graphs & Compilers](../04-kernels-efficiency/09-compilers-fusion.html)
for what the compiler is and is not able to do here), or an attention kernel named something other
than the flash variant (your document mask fell off the fast path, see above). NVIDIA Nsight Systems
(`nsys profile`) gives the same picture with more hardware counters.

## Evaluation, Sampling, and Logging During the Run

Every `cfg.eval_every` steps we measure held-out loss on the val shards the model has never trained
on (the document-hash split created in
[the data-pipeline chapter](../14-capstone/02-data-pipeline.html)), and every `cfg.sample_every`
steps we generate a short free-running sample so a human can sanity-check qualitative progress that
a loss number alone can miss (repetition loops, garbled tokenization, degenerate outputs).

```python
import contextlib, math
import torch


@contextlib.contextmanager
def z_loss_off(raw_model):
    """Report *pure* cross-entropy at eval time. The z-loss is a training
    regularizer, not part of the quantity we compare across steps or against
    other models -- folding it into a reported val loss makes your perplexity
    incomparable to everyone else's. `fused_ce_z_loss` multiplies the z-term by
    `z_coef`, so zeroing the coefficient removes it exactly, on both the chunked
    and the unchunked path."""
    old = raw_model.cfg.z_loss_coef
    raw_model.cfg.z_loss_coef = 0.0
    try:
        yield
    finally:
        raw_model.cfg.z_loss_coef = old


@torch.no_grad()
def evaluate(raw_model, val_loader, cfg):
    """Held-out loss on a FIXED prefix of the val stream.

    Building the iterator fresh from a non-shuffled loader means every call sees
    the *same* eval_iters batches, so the val curve moves only because the model
    moved. (Never use itertools.cycle for this: it caches every batch it has
    yielded, so cycling a shard-backed loader slowly consumes all your RAM.)

    We evaluate through `raw_model`, not the compiled wrapper: toggling
    `cfg.z_loss_coef` is a Python-float change that would fail a Dynamo guard and
    trigger a recompile on every eval. Fifty eager batches every 500 steps is
    well under 1% of the run."""
    raw_model.eval()
    total, n = 0.0, 0
    with z_loss_off(raw_model):
        for i, batch in enumerate(val_loader):
            if i >= cfg.eval_iters:
                break
            x, y, pos, seq = (batch[k].to(cfg.device, non_blocking=True)
                              for k in ("input_ids", "targets", "position_ids", "seq_ids"))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = raw_model(x, targets=y, position_ids=pos, seq_ids=seq)
            total += loss.item(); n += 1
    raw_model.train()
    mean = total / max(n, 1)
    return mean, math.exp(mean)      # (nats/token, perplexity)


@torch.no_grad()
def sample_text(raw_model, tok, prompt, cfg, max_new_tokens=64):
    """Ch. 14.4's `generate` already does KV-cached, O(T) decoding with top-p
    sampling -- there is no reason to hand-roll a second sampler here. Prefix the
    prompt with <|bos|> so the model sees the same document-start marker the
    packer wrote into every training window."""
    ids = torch.tensor([[tok.bos_id, *tok.encode(prompt)]], device=cfg.device)
    out = raw_model.generate(ids, max_new_tokens=max_new_tokens,
                             temperature=0.8, top_p=0.95, eos_id=tok.eos_id)
    return tok.decode(out[0].tolist())


def log_metrics(cfg, log_path, **record):
    """Append-only JSONL -- trivial to load into a dataframe for the loss-curve
    plots the retrospective chapter (14.12) draws from. Swap or supplement with
    an experiment tracker (`wandb.log(record)`, HuggingFace `trackio`, or
    `mlflow`) when you want live curves; the JSONL is the artifact that survives
    the tracker going away."""
    import json, time
    record["ts"] = time.time()
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    if record["step"] % cfg.log_every == 0:
        vl = f"  val {record['val_loss']:.3f}" if record.get("val_loss") is not None else ""
        qk = f"  qk {record['qk_fired']}" if record.get("qk_fired") is not None else ""
        print(f"step {record['step']:>6}/{cfg.stop_at_step}  loss {record['loss']:.3f}{vl}  "
              f"lr×{record['lr_mult']:.3f}  |g| {record['grad_norm']:.2f}{qk}  "
              f"tok/s {record['tokens_per_sec']:,.0f}  mfu {record['mfu']*100:.1f}%  "
              f"data_wait {record['data_wait']*1e3:.0f}ms  mem {record['peak_gb']:.1f}GB")
```

### Expected loss curve magnitudes

At initialization, cross-entropy over a fresh 32,768-token vocabulary starts at essentially
$\ln(32{,}768) \approx 10.4$ nats/token — the model is guessing uniformly. Warmup and the first few
hundred steps drop this quickly, since the easiest signal (token frequency statistics) is learned
almost immediately; by the end of the 500-step warmup, loss is typically already down to something
on the order of low-to-mid single digits of nats/token. Through the long **stable** phase, loss
decreases slowly and fairly smoothly — this is the bulk of the 32,147 steps this chapter runs, and
the log-loss-vs-log-tokens curve should look close to a straight line, the empirical signature the
scaling law in Ch. 14.5 was fit from. This chapter therefore *ends* on a plateau, with train loss in
the ballpark of the low 3s nats/token and no dramatic final drop — that drop belongs to the decay
phase, which [mid-training](../14-capstone/08-mid-training.html) runs on the higher-quality mix, and
WSD's whole appeal is that the decay-phase improvement is disproportionate to its short duration.
After Ch. 14.8 completes the decay, a finished run should land with a **final train loss on the
order of 2.8–3.2 nats/token** (as stated in `capstone/PLAN.md`) — a perplexity on the order of
$e^{2.8}\!\approx\!16$ to $e^{3.2}\!\approx\!25$. Treat both the intermediate numbers and this final
range as illustrative magnitudes to sanity-check against, never as a benchmark to match exactly; the
precise value depends on your data mix, dedup quality, and tokenizer.

## The Complete Training Script

Putting every piece above together into one runnable entry point. Note that it *calls* the helpers
defined earlier rather than re-inlining them — there is exactly one copy of the forward/backward
body in this chapter, and it is `accumulate`.

```python
# stacklm/train.py
"""Stack-100M pretraining: single-GPU (A100) flagship path, warmup + stable.
Ends at cfg.stop_at_step with ckpt_stable.pt -- the LR is still on its plateau,
because Ch. 14.8 spends the WSD decay leg on the mid-training mix.

Usage:  python -m stacklm.train --config configs/a100.yaml
"""
import os, time
import torch

from stacklm.model import Stack100M
from stacklm.optim import build_optimizers
from stacklm.tokenizer import StackTokenizer
from stacklm.train_config import TrainConfig
from stacklm.train_data import build_loader
from stacklm.checkpoint import load_checkpoint, save_checkpoint, save_rolling
from stacklm.train_utils import (accumulate, evaluate, sample_text, log_metrics,
                                 enable_activation_checkpointing, maybe_qk_clip,
                                 utilization, all_params_of, attach_base_lrs,
                                 set_lr, A100_BF16_PEAK, BATCH_KEYS)


def main(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.benchmark = True            # fixed shapes every step
    torch.set_float32_matmul_precision("high")       # TF32 for the remaining fp32 math

    tok = StackTokenizer.load(cfg.tokenizer_path)
    assert tok.pad_id == 32761, "Ch. 14.3 puts the nine specials in the FINAL nine ids"

    # ---- model (fp32 masters; bf16 only inside autocast) ---------------------
    model = Stack100M(cfg.model).to(cfg.device)
    n_params = model.num_params()
    if cfg.activation_checkpointing:
        enable_activation_checkpointing(model)

    # ---- optimizers (built BEFORE loading, so their state can be restored) ---
    optimizers = list(build_optimizers(model, muon_lr=cfg.muon_peak_lr,
                                       adamw_lr=cfg.adamw_peak_lr,
                                       weight_decay=cfg.weight_decay, betas=cfg.betas))
    attach_base_lrs(optimizers)
    params = all_params_of(optimizers)

    # ---- resume ------------------------------------------------------------
    step, tokens_seen, skipped_total = 0, 0, 0
    resume_path = f"{cfg.ckpt_dir}/latest.pt"
    if os.path.exists(resume_path):
        ckpt = load_checkpoint(resume_path, model, optimizers, map_location=cfg.device)
        step, tokens_seen = ckpt["step"], ckpt["tokens_seen"]
        print(f"resumed from step {step} ({tokens_seen:,} tokens seen)")

    # Compile AFTER loading: `_unwrap` keeps checkpoints portable either way, but
    # loading into the uncompiled module avoids `_orig_mod.` key surprises.
    raw_model = model
    if cfg.compile_model:
        model = torch.compile(model)   # TorchDynamo + TorchInductor; see Ch. 4.9

    # ---- data --------------------------------------------------------------
    samples_per_step = cfg.micro_batch_size * cfg.grad_accum_steps      # 256
    train_loader = build_loader(cfg.shard_dir, cfg, tok,
                                start_sample=step * samples_per_step)
    val_loader = build_loader(cfg.val_shard_dir, cfg, tok, shuffle=False)
    train_iter = iter(train_loader)

    # The single most valuable assertion in the file: without seq_ids the model
    # takes its plain-causal fast path and attends ACROSS documents for the whole
    # run, which lowers the loss slightly and is otherwise invisible.
    assert set(BATCH_KEYS) <= set(train_loader.dataset[0]), \
        "dataset must supply seq_ids (Ch. 14.2) or document masking is silently off"

    # Fixed probe batch for MuonClip: drawn from val, so the training stream --
    # and therefore the step -> start_sample resume arithmetic -- is untouched.
    probe = {k: v[:cfg.qk_probe_seqs].to(cfg.device)
             for k, v in next(iter(val_loader)).items()}

    recompute_factor = 4 / 3 if cfg.activation_checkpointing else 1.0
    consecutive_skips = 0

    # ---- the loop ----------------------------------------------------------
    while step < cfg.stop_at_step:
        lr_mult = set_lr(optimizers, step, cfg)

        torch.cuda.synchronize(); t0 = time.perf_counter()
        loss_sum, data_wait = accumulate(model, optimizers, train_iter, cfg)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)

        if torch.isfinite(grad_norm):
            for opt in optimizers:
                opt.step()
            consecutive_skips = 0
        else:
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            consecutive_skips += 1; skipped_total += 1
            print(f"step {step}: non-finite grad norm, skipping "
                  f"({consecutive_skips} in a row)")
            if consecutive_skips >= cfg.max_consecutive_skips:
                raise RuntimeError("too many consecutive non-finite steps; "
                                   "roll back to an earlier step_*.pt")

        qk_fired = maybe_qk_clip(raw_model, probe, cfg, step)   # MuonClip, post-step
        torch.cuda.synchronize(); dt = time.perf_counter() - t0

        # ---- metrics --------------------------------------------------------
        tokens_this_step = samples_per_step * cfg.seq_len          # 524,288
        tokens_seen += tokens_this_step
        tokens_per_sec = tokens_this_step / dt
        mfu, hfu = utilization(n_params, tokens_per_sec, cfg,
                               peak_flops=A100_BF16_PEAK,
                               recompute_factor=recompute_factor)

        val_loss = None
        if step % cfg.eval_every == 0:
            val_loss, val_ppl = evaluate(raw_model, val_loader, cfg)
        if step % cfg.sample_every == 0:
            print(f"  sample @ {step}: "
                  f"{sample_text(raw_model, tok, 'The history of', cfg)[:200]!r}")

        log_metrics(cfg, f"{cfg.ckpt_dir}/log.jsonl",
                    step=step, loss=loss_sum.item(), lr_mult=lr_mult,
                    grad_norm=grad_norm.item(), tokens_seen=tokens_seen,
                    tokens_per_sec=tokens_per_sec, dt=dt, data_wait=data_wait,
                    mfu=mfu, hfu=hfu, val_loss=val_loss, qk_fired=qk_fired,
                    skipped=skipped_total,
                    peak_gb=torch.cuda.max_memory_allocated() / 2**30)

        if step % cfg.ckpt_every == 0 and step > 0:
            save_rolling(model, optimizers, cfg, step=step, tokens_seen=tokens_seen,
                         keep_last=cfg.keep_last_ckpts)
        step += 1

    # ---- hand off to mid-training ------------------------------------------
    save_checkpoint(f"{cfg.ckpt_dir}/ckpt_stable.pt", model, optimizers,
                    step=step, tokens_seen=tokens_seen, config=cfg,
                    data_seed=cfg.seed)
    print(f"stable phase done: {step} steps, {tokens_seen:,} tokens, "
          f"{skipped_total} skipped. LR still at plateau -> ckpt_stable.pt "
          f"(Ch. 14.8 runs the 6,000-step WSD decay leg from here).")


if __name__ == "__main__":
    main(TrainConfig())
```

This is the whole flagship loop — nothing here needs multiple GPUs, a job scheduler, or a
distributed launcher. That single-GPU sufficiency is itself a point worth taking seriously: a
101M-parameter dense model with a 524k-token effective batch fits, with room to spare, on one 80GB
accelerator.

### What the libraries would have done for you

We hand-rolled every piece because the point of this book is that nothing stays a black box. In a
production run you would reach for the following, and now you know exactly what each is doing and
why it exists:

| Hand-rolled here | The library that owns this job |
|---|---|
| accumulation, clipping, AMP, device placement | HuggingFace `accelerate`, `transformers.Trainer`, PyTorch Lightning |
| resumable data ordering | `torchdata.stateful_dataloader.StatefulDataLoader` (pytorch/data) |
| chunked linear cross-entropy | Liger-Kernel `LigerFusedLinearCrossEntropy`; Apple `cut-cross-entropy`; torchtune chunked-output loss |
| WSD schedule | `transformers.get_wsd_schedule`; torchtitan's warmup/stable/decay phase config |
| document-aware attention masks | `torch.nn.attention.flex_attention`; `flash_attn_varlen_func` (Dao-AILab) |
| activation checkpointing | `apply_activation_checkpointing`; selective AC via `create_selective_checkpoint_contexts` |
| checkpoint + resume at scale | `torch.distributed.checkpoint` (DCP), sharded and async |
| throughput, MFU, tracing | `torch.profiler`, NVIDIA Nsight Systems, `wandb` / `trackio` / `mlflow` |
| fp8 / low-precision training | `torchao.float8`, NVIDIA Transformer Engine (Hopper/Blackwell only) |
| the entire loop, multi-GPU | torchtitan, Megatron-LM, DeepSpeed, nanotron, levanter, litgpt |

The honest summary: at 100M parameters on one GPU, a hand-written loop is *better* than a framework
— it is 300 lines you can read in full, and every number in it is one you derived. Past a few
billion parameters and a few dozen GPUs the calculus flips, and the frameworks in that last row
exist because the failure modes stop being ones you can hold in your head.

## Scaling Out: DDP, FSDP, and Beyond One GPU

Stack-100M does not *need* more than one GPU — it is a deliberate choice to keep the flagship path
reproducible on hardware anyone can rent by the hour. But if you have access to the capstone's
optional 8×H100 box (for a faster wall clock on the same model, or as a stepping stone toward the
1B-parameter scale-up discussed in
[the retrospective chapter](../14-capstone/12-retrospective-and-scaleup.html)), the upgrade path is
exactly the one covered in
[Distributed Training I: Data Parallelism, DDP, ZeRO & FSDP](../03-pretraining/05-distributed-data-parallel.html).

**DDP first.** Since the entire model, optimizer state, and activations already fit on one GPU, the
natural next step is pure data parallelism: replicate the model on every GPU, run independent
micro-batches, and all-reduce gradients before each optimizer step.

```python
import contextlib, os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend="nccl")
rank, world = dist.get_rank(), dist.get_world_size()
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
model = DDP(model.to(local_rank), device_ids=[local_rank])

# Each rank must read a DISJOINT slice of the stream: offset the sampler by rank
# and stride by world size, or keep grad_accum_steps * world constant so the
# effective batch stays at 524,288 tokens.
start = step * cfg.micro_batch_size * cfg.grad_accum_steps * world + rank

# Skip the gradient all-reduce on every micro-batch except the last one in the
# accumulation window -- DDP would otherwise synchronize grad_accum_steps times
# per optimizer step for no benefit.
for i in range(cfg.grad_accum_steps):
    ctx = model.no_sync() if i < cfg.grad_accum_steps - 1 else contextlib.nullcontext()
    with ctx:
        ...
        loss.backward()
```

launched with `torchrun --nproc_per_node=8 -m stacklm.train`. Eight GPUs at near-linear scaling
turns a ~24-GPU-hour single-A100 run into roughly ~3 wall-clock hours on 8 — the *cost* in GPU-hours
is unchanged, only the wall clock improves, which is the entire point of DDP: it does not reduce
total compute or per-GPU memory pressure. Three things must change in the surrounding code, and all
three are easy to forget: the data stream must be sharded by rank (above), only rank 0 should write
checkpoints and logs, and `qk_clip_` must run on every rank with the *same* probe batch (it is an
in-place parameter edit, so ranks that skip it immediately diverge from ranks that do not).

**FSDP when the model no longer fits.**
[Distributed Training I](../03-pretraining/05-distributed-data-parallel.html) covers Fully Sharded
Data Parallel in depth; the short version is that FSDP shards parameters, gradients, *and* optimizer
state across GPUs (à la ZeRO, Rajbhandari et al., 2020) instead of replicating them, trading extra
communication for a memory footprint that shrinks with GPU count. PyTorch's current API is FSDP2
(`fully_shard`), which uses `DTensor` and composes cleanly with `torch.compile` and DCP. At 101M
parameters this is unnecessary — the whole point of Stack-100M's size is that it fits comfortably on
one accelerator — but it becomes the relevant tool the moment you scale toward the 1B-parameter
regime, where optimizer state no longer fits alongside activations on a single GPU.

**Multi-node, one paragraph.** Beyond one 8-GPU box, the same DDP/FSDP code launches across machines
via `torchrun --nnodes=N --node_rank=R --rdzv_endpoint=...`, with NCCL collectives now crossing the
network instead of NVLink — see
[Parallel Computing & Collective Communication](../01-foundations/09-parallel-collectives.html) for
the interconnect trade-offs (InfiniBand/EFA vs. Ethernet) and
[Megatron-LM, DeepSpeed & Parallelism in Practice](../03-pretraining/07-megatron-deepspeed.html) for
the production frameworks that combine data, tensor, and pipeline parallelism at that scale. None of
it is necessary to train Stack-100M; it is the path if you decide to chase a much bigger model
afterward.

## Compute Tiers: A100, RTX 4090, and Free Colab

The same `train.py` and `TrainConfig` serve all three compute tiers documented in the capstone plan
— only the numbers change.

| Tier | GPU | VRAM | bf16 dense peak | precision | micro-batch × seq | grad accum | eff. batch | act. ckpt | est. wall-clock | est. cost |
|---|---|---|---|---|---|---|---|---|---|---|
| **Flagship** | 1×A100 80GB SXM | 80GB | 312 TFLOP/s | bf16 autocast, no scaler | 32 × 2048 | 8 | ≈524k | off | ~22–29 GPU-hr | ~USD 25–50 |
| **Consumer** | RTX 4090 | 24GB | ~165 TFLOP/s | bf16 autocast, no scaler | 8 × 2048 | 32 | ≈524k | optional | ~2–4× the A100 wall-clock | ~USD 0 (owned; electricity) |
| **Free on-ramp** | Colab T4 | 16GB | none (fp16: ~65 TFLOP/s) | fp16 + `GradScaler` | scaled-down model, seq 1024 | tune to fit | reduced target | on | hours, a subset run | USD 0 |

Peak figures are vendor dense specifications with no structured sparsity; treat them as the
denominator of your MFU calculation, not as achievable throughput.

### RTX 4090: same recipe, smaller micro-batch

A 4090's 24GB has to hold fp32 master weights (~406 MB), gradients (~406 MB), Muon+AdamW state
(~470 MB), activations, and the loss head. With `loss_chunk = 8192`, `micro_batch_size = 8` costs
roughly 4 GB of block activations plus ~2 GB for the head plus ~1.3 GB of persistent state —
comfortably inside 24 GB *without* activation checkpointing, which is why the table marks it
optional rather than required. That is a direct consequence of the chunked head: at `loss_chunk = 0`
the head alone was 7.5 GB at this micro-batch and checkpointing was mandatory. If you have headroom,
raise `micro_batch_size` to 16 and drop `grad_accum_steps` to 16 — same 524,288-token effective
batch, fewer kernel launches, better MFU. If you OOM, the safe fallback is 8 × 32 with checkpointing
on, or `loss_chunk = 4096`. The ~2–4× wall-clock versus the A100 comes from the 4090's lower dense
bf16 tensor-core throughput and the extra accumulation steps; nothing else in `train.py` changes.

!!! note "Colab's T4 has no bf16 tensor cores"

    The free-tier Colab T4 is a Turing-generation GPU (compute capability 7.5); bf16 hardware
    acceleration first arrived with Ampere (A100, RTX 30-series and up, compute capability ≥8.0).
    On a T4 the training loop must switch to classic mixed precision —
    `torch.autocast(dtype=torch.float16)` together with `torch.amp.GradScaler("cuda")` to guard
    against fp16's narrower exponent range — rather than the scaler-free bf16 path used everywhere
    else in this chapter. Note that this interacts with the NaN guard: with a `GradScaler` you must
    call `scaler.unscale_(opt)` before `clip_grad_norm_`, and the scaler *already* skips steps whose
    gradients are non-finite, so the guard becomes a logging point rather than a control decision.
    See [Numerical Computing, Floating Point & Precision](../01-foundations/04-numerics-precision.html)
    for exactly what the extra 3 exponent bits in bf16 buy you.

    The Colab config is also explicitly a **scaled-down on-ramp**, not the full Stack-100M recipe — a
    smaller model (fewer layers, narrower `d_model`) and a token budget an order of magnitude below
    20B, chosen so the run finishes in a session that a free Colab instance will actually let you
    keep. It exists so a reader with zero dollars can *see the loop work end to end* before
    committing real money to the flagship A100 run.

!!! interview "Interview Corner"

    **Q1:** You're pretraining a ~100M-parameter dense transformer on an A100. `nvidia-smi` shows
    85% GPU utilization the whole time, but when you compute MFU from your measured tokens/sec you
    get only 18%. What does that gap tell you, and what would you check first?

    **A1:** GPU-Util just means *some* kernel was resident on the SM array during the sampling
    window — it says nothing about whether that kernel was a dense, well-fused matmul running near
    peak or a small memory-bound op leaving the tensor cores idle. An 85%-util / 18%-MFU run is the
    classic signature of the GPU spending its "busy" time on cheap, unfused, or small operations
    rather than the big matmuls the FLOP count assumes. First checks, roughly in order of payoff at
    this size: (1) is `torch.compile` actually on and fusing — check `TORCH_LOGS=graph_breaks`,
    because one graph break inside the block loop turns 30 fused blocks into hundreds of tiny
    kernels; (2) is the attention path actually hitting FlashAttention — passing an explicit dense
    boolean document mask to SDPA silently drops you to the math backend, which is the single most
    common cause of this exact symptom in a document-packed pretraining loop, and the fix is
    FlexAttention's `BlockMask` or varlen `cu_seqlens` rather than removing the mask; (3) is the
    micro-batch too small to saturate the SMs; (4) is the loader starving the GPU — log `data_wait`
    next to `dt`, because a stall shows up as high util between bursts of idle waiting, not as low
    util. Confirm with `torch.profiler` rather than guessing. Two framing points worth stating
    unprompted: MFU is convention-dependent (6ND-only vs. including the attention score/value
    matmuls, a ~31% difference for a deep-thin model at seq_len 2048), and MFU is not HFU — with
    activation checkpointing on, a third of the hardware's work is recompute that MFU deliberately
    does not credit.

    **Q2:** Your 100M model trains fine at micro-batch 8 but OOMs at micro-batch 32, even though a
    back-of-envelope activation calculation says it should fit in 80 GB. Where did the memory go?

    **A2:** Almost certainly the loss head. Activation estimates usually count the residual stream
    and MLP intermediates and forget the logits, which are `B·T·V` — and at `V = 32768` with a
    `d_model` of only 512 the vocab projection is 64× wider than the residual stream. At
    `B·T = 65,536` the bf16 logits are 4.3 GB, and a naive implementation adds an fp32 upcast for
    cross-entropy, the fp32 `log_softmax` output saved for backward, and often a *second* fp32 copy
    if the z-loss recomputes `logsumexp` separately — about 30 GB total, which dwarfs the trunk.
    Activation checkpointing does not help because the head is outside the blocks. The fix is a
    chunked or fused linear-cross-entropy that never materializes the full logit tensor:
    Liger-Kernel's `LigerFusedLinearCrossEntropy`, Apple's cut-cross-entropy, or the
    chunk-plus-`torch.utils.checkpoint` loop in `fused_ce_z_loss`, which costs about 5% extra FLOPs
    and makes peak head memory a function of `chunk`, not of batch size.

## Key Takeaways

!!! key "Key Takeaways"

    - A single A100 is sufficient for the entire Stack-100M pretraining run (~22–29 GPU-hours,
      ~USD 25–50 illustrative); DDP and FSDP are optional speed-ups, never requirements, at this
      model size.
    - **At large vocabularies the loss head, not attention, caps your batch size.** At
      `B·T = 65,536` and `V = 32,768` the logits plus their fp32 copies are ~30 GB — more than the
      whole 30-block trunk — and activation checkpointing does nothing about it. `loss_chunk > 0`
      makes peak head memory a function of the chunk, not the batch: ~2.1 GB at 8192, for ~5% more
      FLOPs. Liger-Kernel and cut-cross-entropy are the fused production versions.
    - **Thread `seq_ids` and `position_ids` through every forward call.** Without `seq_ids` the model
      falls back to a plain causal mask and attends across document boundaries for the whole run —
      a bug that *lowers* training loss and is invisible in the curve. Assert it at startup, and
      derive `pad_id` from the tokenizer object rather than typing a constant.
    - **Effective batch = micro_batch × seq_len × grad_accum**, and **bf16 autocast needs no loss
      scaler** (bf16 shares fp32's 8-bit exponent, so there is no overflow to guard against). Zero
      gradients once per optimizer step, scale the loss by `1/grad_accum` before `.backward()`, and
      never call `.item()` inside the accumulation loop.
    - **An explicit boolean attention mask is the slow path.** Express document isolation as
      structure the kernel understands — FlexAttention's `BlockMask` or varlen FlashAttention with
      `cu_seqlens` — or SDPA silently falls off the fused kernel and your MFU collapses.
    - Clip the **global** gradient norm across *both* optimizers before stepping, then **check it is
      finite**: skip the step if it is not, abort after a run of skips, and keep rolling
      step-stamped checkpoints so a poisoned `latest.pt` is never the only state you have. Run
      **MuonClip after the step, on a fixed probe batch, every 200 steps** (τ = 30 under QK-norm) —
      not on every training forward, where it would read pre-step weights.
    - Checkpoints must save **model + both optimizers + step + data position + RNG**, written
      atomically, with RNG stored as tensors and plain containers so PyTorch ≥2.6's default
      `weights_only=True` load still works. Derive the data position from `step`, not from the
      loader, whose prefetch runs ahead of the loop.
    - **MFU = model FLOPs ÷ peak FLOPs/s** is the metric that matters, and `nvidia-smi` GPU-Util can
      read near 100% while MFU sits far below it. State your FLOP convention: 6ND alone understates
      Stack-100M by ~31% because it omits attention's score/value matmuls. MFU ≠ HFU: with
      full-block checkpointing HFU ≈ 1.33 × MFU. Activation checkpointing keeps the
      *block-boundary* activations and recomputes what is *inside* each block — a 3–8× cut for ~33%
      more FLOPs, not an `n_layers`-fold cut.
    - This chapter deliberately **stops at step 32,147, before the decay**: `ckpt_stable.pt` is
      handed to mid-training with the LR still at plateau, so WSD's high-value 6,000-step decay leg
      is spent on premium data. Expect final train loss on the order of **2.8–3.2 nats/token** only
      *after* Ch. 14.8 — never treat that, or any illustrative number here, as a benchmark to hit.

!!! sota "State of the Art & Resources (2026)"
    Single-GPU training loops like this one now borrow directly from the open speedrunning
    community — Muon-based optimizers, `torch.compile`, fused loss heads, and PyTorch-native
    sharding have all moved from research curiosities to defaults since 2024.

    **Foundational work**

    - [Narayanan et al., *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM* (2021)](https://arxiv.org/abs/2104.04473) — established reporting achieved-vs-peak FLOPs/s as the standard way to measure training efficiency, the methodology MFU generalizes to a single GPU.
    - [Korthikanti et al., *Reducing Activation Recomputation in Large Transformer Models* (2022)](https://arxiv.org/abs/2205.05198) — the source of the MFU/HFU distinction and of selective, rather than all-or-nothing, activation recompute.

    **Recent advances (2023–2026)**

    - [Jordan, *Muon: An optimizer for hidden layers in neural networks* (2024)](https://kellerjordan.github.io/posts/muon/) — the Newton-Schulz-orthogonalized momentum optimizer this loop applies to 2-D hidden weights.
    - [Wijmans et al., *Cut Your Losses in Large-Vocabulary Language Models* (2024)](https://github.com/apple/ml-cross-entropy) — computes cross-entropy without ever materializing the global logit matrix; the memory problem `fused_ce_z_loss` solves by hand.
    - [Hsu et al., *Liger Kernel: Efficient Triton Kernels for LLM Training* (2024)](https://github.com/linkedin/Liger-Kernel) — production Triton kernels including the fused linear cross-entropy used across the open fine-tuning stack.
    - [PyTorch team, *FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention* (2024)](https://pytorch.org/blog/flexattention/) — compiled, block-sparse attention masks; the modern way to express document-aware masking without leaving the fast kernel.
    - [Unsloth, *Bugs in LLM Training – Gradient Accumulation Fix* (2024)](https://unsloth.ai/blog/gradient) — a real, widely-reproduced instance of exactly the accumulation-scaling pitfall this chapter warns about, with measured loss-curve impact.
    - [PyTorch team, *PyTorch 2: Faster ML Through Dynamic Bytecode Transformation and Graph Compilation* (ASPLOS 2024)](https://pytorch.org/blog/pytorch-pytorch-2-paper-tutorial/) — the TorchDynamo/TorchInductor stack behind `torch.compile`.

    **Open-source & tools**

    - [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — the minimal single-GPU GPT training reference this capstone's loop is a direct descendant of.
    - [karpathy/llm.c](https://github.com/karpathy/llm.c) — GPT-2/GPT-3 training in pure C/CUDA, a from-scratch counterpoint to the PyTorch loop here.
    - [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) — a community speedrunning record chain (Muon, FlashAttention, custom kernels) tracking how fast a small GPT can be trained end to end.
    - [pytorch/torchtitan](https://github.com/pytorch/torchtitan) — PyTorch-native pretraining platform (FSDP2, `torch.compile`, DCP, `StatefulDataLoader`, selective AC) implementing essentially every technique in this chapter at cluster scale.
    - [pytorch/data](https://github.com/pytorch/data) — `StatefulDataLoader`, the correct answer to resumable streaming data loading.

    **Go deeper**

    - [Getting Started with Fully Sharded Data Parallel (FSDP2)](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) — official PyTorch tutorial for the sharding path this chapter's FSDP discussion references.
    - [PyTorch Distributed Checkpoint (DCP)](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html) — sharded, asynchronous checkpointing for when `torch.save` stops being adequate.
    - [Methods and tools for efficient training on a single GPU](https://huggingface.co/docs/transformers/perf_train_gpu_one) — HuggingFace's practitioner guide to the same batching/precision/memory levers this chapter walks through by hand.

## Further reading

- Kaplan et al., *Scaling Laws for Neural Language Models* (2020) — the 6ND FLOPs-per-token rule this chapter's MFU calculation starts from, and corrects.
- Hoffmann et al., *Training Compute-Optimal Large Language Models* ("Chinchilla," 2022) — the compute-optimal frontier Stack-100M deliberately over-trains past.
- Micikevicius et al., *Mixed Precision Training* (2017) — the loss-scaling machinery bf16 lets us skip.
- Chen et al., *Training Deep Nets with Sublinear Memory Cost* (2016) — the original activation/gradient checkpointing idea.
- Korthikanti et al., *Reducing Activation Recomputation in Large Transformer Models* (2022) — selective, not all-or-nothing, checkpointing; MFU vs. HFU.
- Chowdhery et al., *PaLM: Scaling Language Modeling with Pathways* (2022) — popularized MFU as a training-efficiency metric; source of both the z-loss and the attention-inclusive FLOP formula.
- Wijmans et al., *Cut Your Losses in Large-Vocabulary Language Models* (2024) — why the loss head, not attention, is the memory bottleneck at small `d_model` and large `V`.
- Hu et al., *MiniCPM* (2024) — the Warmup-Stable-Decay schedule whose decay leg this chapter deliberately leaves unspent.
- Ibrahim et al., *Simple and Scalable Strategies to Continually Pre-train Large Language Models* (2024) — why resuming from a *pre-decay* checkpoint beats re-warming a decayed one.
- Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* (2020) — the sharding idea FSDP implements.
- Karpathy, *nanoGPT* and *llm.c* — the direct lineage this capstone's single-GPU, cost-conscious training loop updates for 2025–2026.

## Exercises

**1.** The chapter trains in bf16 autocast and states this needs no `GradScaler`, while the
free-tier Colab T4 path must switch to `torch.autocast(dtype=torch.float16)` *with* a `GradScaler`.
Explain (a) what problem `GradScaler` solves for fp16, and (b) why bf16 makes it unnecessary. What
does bf16 give up in exchange, and why does the chapter argue that trade-off is acceptable here?

??? note "Solution"
    **(a) What `GradScaler` solves for fp16.** fp16 has only a 5-bit exponent, giving it a narrow
    dynamic range. Small gradients underflow to zero and large activations/gradients overflow to
    `inf`, either of which corrupts the update. `GradScaler` multiplies the loss by a large scale
    factor before `.backward()` so that small gradients are pushed up into fp16's representable
    range, then unscales the gradients before the optimizer step, dynamically backing the scale
    factor off whenever an `inf`/`nan` is detected.

    **(b) Why bf16 removes the need.** bf16 keeps fp32's full **8-bit exponent** (same dynamic range
    as fp32) and trims only the mantissa to 7 bits. Because the exponent range is identical to fp32,
    there is no realistic overflow/underflow risk on activations or gradients, so there is nothing
    for dynamic loss scaling to guard against.

    **The trade-off.** bf16 gives up mantissa precision: 7 explicit mantissa bits vs fp16's 10. The
    chapter accepts this because (i) the model's *master* parameters are kept in fp32 and gradients
    accumulate in fp32, so tiny per-step updates at small learning rates are not rounded away; and
    (ii) the loss itself is computed in fp32 — inside `fused_ce_z_loss`, `F.linear(h, w).float()`
    upcasts each chunk before the `logsumexp`, protecting the numerically sensitive step. Range
    matters more than the last few mantissa bits for training stability, which is precisely what
    bf16 optimizes for.

    **One extra wrinkle on the T4 path.** With a `GradScaler` in the loop, `clip_grad_norm_` must be
    preceded by `scaler.unscale_(opt)` — otherwise you clip the *scaled* gradient and the effective
    clip threshold becomes the scale factor times what you asked for, which drifts every time the
    scaler adjusts.

**2.** In `accumulate`, the per-micro-batch loss is divided by `cfg.grad_accum_steps` *before*
`.backward()`. Suppose a teammate deletes that division (leaving everything else, including the
single `zero_grad` per optimizer step, correct). With the chapter's config (`grad_accum_steps = 8`),
what happens to the gradient that reaches `optimizer.step()`, and what is the practical effect on
the run? Contrast this with the *other* pitfall the chapter warns about — calling `zero_grad()`
inside the accumulation loop.

??? note "Solution"
    **Effect of dropping `/ grad_accum_steps`.** Gradients from successive `.backward()` calls are
    *summed* into `.grad`. With the division in place the accumulated gradient is the **mean** over
    the full 524,288-token batch — exactly what a single giant batch would produce. Delete it and
    the accumulated gradient is the **sum** of 8 micro-batch gradients, i.e. $8\times$ too large, so
    the update is effectively at $8\times$ the intended learning rate.

    **Practical effect.** The run no longer matches the Muon/WSD hyperparameters tuned in Ch. 14.6.
    Global grad-norm clipping to $c=1.0$ partly masks it — the clip rescales the inflated norm back
    toward 1 — but the effective step size is distorted whenever the norm is *below* the clip
    threshold, so early and late training (small gradients) are most affected. This is precisely the
    bug Unsloth documented shipping in production trainers.

    **Contrast.** Zeroing inside the loop goes the *opposite* direction: it throws away the previous
    7 gradients, so `optimizer.step()` sees only the last micro-batch — one 65,536-token step at
    $\frac{1}{8}$ the intended effective batch. Both leave the loss curve going down, but one
    inflates the effective LR by $8\times$ and the other shrinks the effective batch by $8\times$.

**3.** You are configuring a scaled-down on-ramp tier with a smaller model. You choose
`micro_batch_size = 4` and `seq_len = 1024`, and you want a **reduced** effective batch of exactly
262,144 tokens per optimizer step with a total token budget of $2\times10^9$ tokens. Compute (a) the
`grad_accum_steps` you must set, (b) `total_steps`, and (c) — keeping the flagship's *decay
fraction* $6{,}000/38{,}147$ — the `decay_steps` and the `stop_at_step` at which this tier's
pretraining run should hand off to mid-training.

??? note "Solution"
    **(a) `grad_accum_steps`.**
    $$
    \texttt{grad\_accum\_steps} = \frac{262{,}144}{4 \times 1024}
    = \frac{262{,}144}{4{,}096} = 64.
    $$

    **(b) `total_steps`.**
    $$
    \texttt{total\_steps} = \left\lceil \frac{2\times10^9}{262{,}144} \right\rceil
    = \lceil 7629.39\ldots \rceil = 7630.
    $$

    **(c) Decay and hand-off.** The flagship decay fraction is
    $6{,}000/38{,}147 \approx 0.15729$, so
    $$
    \texttt{decay\_steps} = \operatorname{round}(0.15729 \times 7630) = 1{,}200,
    \qquad \texttt{stop\_at\_step} = 7630 - 1200 = 6{,}430.
    $$
    Pretraining therefore covers $6{,}430 \times 262{,}144 \approx 1.69\times10^9$ tokens and
    mid-training spends the remaining 1,200 steps ($\approx 0.31\times10^9$ tokens) on the annealed
    mix. Pass `decay_steps=1_200` to `wsd_lr` directly rather than a fraction — the absolute
    argument wins, and it keeps the leg fixed if you later re-budget `total_steps`.

**4.** During a flagship A100 run you measure a full-optimizer-step time of `dt = 1.8 s`. Using the
chapter's constants ($N \approx 101.4\times10^6$, effective batch 524,288 tokens, A100 bf16 dense
peak $312\text{ TFLOP/s}$), compute (a) tokens/sec, (b) MFU under the 6ND convention, (c) MFU
including the causal attention term, and (d) the projected wall-clock GPU-hours for the full
$20\times10^9$-token budget. Which of these four numbers changes if you switch on activation
checkpointing?

??? note "Solution"
    **(a) Tokens/sec.** $524{,}288 / 1.8 \approx 291{,}271$ tokens/s.

    **(b) MFU, 6ND only.** $6N \approx 608.4$ MFLOP/token, so
    $$
    608.4\times10^6 \times 291{,}271 \approx 177.2\text{ TFLOP/s},
    \qquad \text{MFU} = \frac{177.2}{312} \approx 56.8\%.
    $$

    **(c) MFU including attention.** The causal score+value term is
    $6 \cdot L \cdot T \cdot d_{\text{model}} = 6 \times 30 \times 2048 \times 512 \approx 188.7$
    MFLOP/token, giving $797.1$ MFLOP/token total:
    $$
    797.1\times10^6 \times 291{,}271 \approx 232.2\text{ TFLOP/s},
    \qquad \text{MFU} \approx 74.4\%.
    $$
    That is high enough to be a warning sign rather than a boast: double-check the measured `dt`
    (is the data fetch inside the timer? did you synchronize?) before reporting it.

    **(d) Projected GPU-hours.**
    $$
    \frac{20\times10^9}{291{,}271} \approx 68{,}665\text{ s} \approx 19.1 \text{ GPU-hours},
    $$
    of which this chapter's 32,147 steps are
    $16.85\times10^9 / 291{,}271 \approx 16.1$ GPU-hours and Ch. 14.8's decay leg the remaining
    ~3.0 — just under the 22–29 GPU-hour envelope, consistent with a faster step time than
    the worked example's 2.3 s.

    **What activation checkpointing changes.** It cannot change (b) or (c) *as definitions*, because
    MFU counts only model FLOPs and recompute is not model FLOPs. What it changes is `dt`: the extra
    forward pass makes each step ~33% slower, so tokens/sec falls, and MFU and GPU-hours both get
    worse proportionally. The number that stays roughly flat is **HFU**, which credits the recompute
    — approximately $1.33 \times$ the new MFU, i.e. roughly the old MFU. That is the whole reason
    both metrics exist.

**5.** The chapter argues for keeping several rolling checkpoints rather than overwriting a single
`latest.pt`, and pairs that with a non-finite-gradient guard. Suppose a run hits a genuine loss
spike: `grad_norm` jumps from ~0.4 to ~90 at step 21,400, the guard does *not* fire (90 is finite),
and by step 21,600 the loss has climbed from 3.2 to 6.1 and is not recovering. Write the recovery
procedure using only the primitives defined in this chapter, and say why each step is needed.

??? note "Solution"
    The guard only catches `inf`/`nan`; a large-but-finite gradient sails through, which is exactly
    the loss-spike regime described in
    [Training Stability, Loss Spikes & Debugging Large Runs](../03-pretraining/11-training-stability.html).
    Recovery:

    1. **Stop the run.** Every additional step applies updates derived from a diverged state and is
       also advancing `latest.pt` toward overwriting your good history.
    2. **Pick a checkpoint from before the spike.** With `ckpt_every = 1000` and `keep_last = 5`,
       `step_0021000.pt` is on disk and predates the step-21,400 spike; the five-deep window is what
       makes this possible at all. Confirm from `log.jsonl` that `grad_norm` was in its normal band
       at that step.
    3. **Skip the offending data window.** The data order is a deterministic function of
       `(seed, start_sample)`, so the batches around step 21,400 will be replayed identically on
       resume and the spike will very likely recur. Restart from step 21,000 but build the loader
       with `start_sample = 21_600 * samples_per_step`, or change `cfg.seed`, which reshuffles
       entirely. The former is surgical; the latter also invalidates exact reproducibility from the
       original seed.
    4. **Optionally lower the ceiling.** If spikes recur at different data offsets, the problem is
       the recipe, not the data: tighten `grad_clip` (1.0 → 0.5), lower `qk_clip_tau` (30 → 20) or
       `qk_clip_every` (200 → 50) so MuonClip engages earlier and more often, or check `qk_fired` in
       the log — a trigger rate that was climbing before the spike is the signature of an
       attention-logit blow-up, and Ch. 14.6's answer is to halve the peak LR.
    5. **Add a soft guard for next time.** Track a running median of `grad_norm` and skip steps
       whose pre-clip norm exceeds, say, 10× it, logging every skip. This catches finite-but-absurd
       gradients that `torch.isfinite` cannot:

       ```python
       # alongside the isfinite check
       gn = grad_norm.item()
       recent.append(gn)                       # collections.deque(maxlen=200)
       med = statistics.median(recent)
       if len(recent) == recent.maxlen and gn > 10 * med:
           for opt in optimizers:
               opt.zero_grad(set_to_none=True)  # skip: outlier batch
           spike_skips += 1
           continue
       ```

**6.** Audit `save_rolling` above. (a) Why must the prune run *after* the write, not before?
(b) Why does the glob pattern make `ckpt_stable.pt` and `final.pt` safe without any special-casing,
and what breaks if you widen it to `*.pt`? (c) Why is the step stamp zero-padded to seven digits?
(d) The function is called with `keep_last=cfg.keep_last_ckpts`. What happens if someone sets that
to `0`, and how would you make the function refuse rather than misbehave?

??? note "Solution"
    **(a) Write-then-prune.** If you pruned first there would be a window — the several seconds
    `torch.save` takes on a ~0.9 GB checkpoint — in which you hold `keep_last - 1` old checkpoints
    and zero new ones. A crash inside that window (the preemption case this whole scheme exists for)
    leaves you one checkpoint poorer than designed, and repeating it every `ckpt_every` steps walks
    the window down to nothing. Write-then-prune makes "at least `keep_last` complete checkpoints
    exist on disk" true at every instant, and `os.replace` makes each of those checkpoints complete.

    **(b) The glob.** `step_*.pt` matches only the rolling series, so `latest.pt`, `ckpt_stable.pt`,
    and `final.pt` are never candidates for deletion — the naming convention *is* the protection.
    Widen it to `*.pt` and the very first prune deletes the hand-off checkpoint that Ch. 14.8's
    `mid_train` is waiting for, plus the `latest.pt` your own resume path reads.

    **(c) The pad.** `sorted()` on filenames is lexical, and lexical order agrees with numeric order
    only when every stamp has the same width. `step_9999.pt` sorts *after* `step_10000.pt` unpadded;
    seven digits keeps them aligned past 9,999,999 steps, comfortably beyond this run's 32,147.

    **(d) `keep_last = 0`.** `lst[:-0]` is `lst[:0]`, i.e. the empty list — so nothing is pruned and
    checkpoints accumulate forever, the exact opposite of what was asked. (It is the same reason
    `lst[:-keep_last]` is correctly empty while fewer than `keep_last` checkpoints exist.) Guard it:
    `assert keep_last >= 1, "keep_last must be >= 1"`. With `keep_last = 5` plus `ckpt_stable.pt`,
    total checkpoint disk is ~5.4 GB at ~0.9 GB each — cheap insurance against a poisoned
    `latest.pt`.

**7.** Memory accounting (arithmetic). For Stack-100M ($V = 32{,}768$, $d_{\text{model}} = 512$,
$n_{\text{layers}} = 30$) compute, for a micro-batch of $B = 32$ sequences of $T = 2048$: (a) the
peak bytes held by the unchunked loss head (bf16 logits + fp32 upcast + fp32 `log_softmax` output +
a second fp32 copy for a separately-computed z-loss); (b) the peak logit bytes with
`loss_chunk = 8192`; (c) the extra FLOPs the chunked version costs, as a percentage of the step's
$6ND$; and (d) the largest `micro_batch_size` whose unchunked loss head alone would still fit in a
24 GB RTX 4090.

??? note "Solution"
    **(a) Unchunked head.** $B \cdot T = 65{,}536$ tokens. Per token the head holds
    $V \times (2 + 4 + 4 + 4) = 32{,}768 \times 14 = 458{,}752$ bytes:
    $$
    65{,}536 \times 458{,}752 \approx 3.01\times10^{10}\text{ B} \approx 30.1\text{ GB}.
    $$

    **(b) Chunked head.** Only one chunk's logits are live. In backward, that chunk's fp32 logits
    and their gradient coexist:
    $$
    8192 \times 32{,}768 \times 4\text{ B} = 1.07\times10^9 \approx 1.07\text{ GB (each)},
    $$
    so roughly **2.1 GB** peak — a ~14× reduction, and *independent of `micro_batch_size`*, which is
    the property that makes larger micro-batches possible at all. `loss_chunk = 4096` halves it
    again to ~1.1 GB.

    **(c) Extra FLOPs.** The recomputed `lm_head` forward matmul is
    $2 \cdot B \cdot T \cdot d \cdot V = 2 \times 65{,}536 \times 512 \times 32{,}768
    \approx 2.20\times10^{12}$ FLOP. The step's model FLOPs are
    $6ND = 6 \times 101.4\times10^6 \times 65{,}536 \approx 3.99\times10^{13}$ FLOP. Ratio
    $\approx 5.5\%$ — and note it does not depend on `chunk`, because every token's logits are
    recomputed exactly once either way.

    **(d) 4090 ceiling for the unchunked head.** Ignoring everything else (weights, optimizer state,
    and block activations, which together need several more GB), the head alone fills 24 GB at
    $$
    B \cdot T = \frac{24 \times 2^{30}}{458{,}752} \approx 56{,}174 \text{ tokens}
    \;\Rightarrow\; B \approx 27.
    $$
    Since the rest of the job needs roughly 5–6 GB, the realistic unchunked ceiling is around
    $B \approx 8\text{–}10$ — which is exactly why the 4090 tier used `micro_batch_size = 8` and
    mandatory activation checkpointing before `loss_chunk` existed.

**8.** Scale-out (implementation). The DDP snippet wraps only the *last* micro-batch in the
accumulation window in a synchronizing context and the earlier ones in `model.no_sync()`. Rewrite
`accumulate` into a `ddp_train_step` that (i) skips the gradient all-reduce on every micro-batch
except the last, and (ii) still produces the correct **mean** gradient over the effective batch.
Then explain why `no_sync()` changes wall-clock but not the resulting gradient, and name the two
*other* things that must change in the surrounding script when you go from 1 GPU to 8.

??? note "Solution"
    DDP triggers a gradient all-reduce as each `.backward()` completes. During accumulation we only
    want *one* all-reduce per optimizer step — after the final micro-batch — so we wrap the earlier
    backward passes in `model.no_sync()`, which suppresses the collective. The loss is still divided
    by `grad_accum_steps` before every `.backward()` so the local accumulated gradient is the mean;
    the single all-reduce on the last step then averages those per-rank means across ranks.

    ```python
    import contextlib
    import torch
    from stacklm.train_utils import BATCH_KEYS

    def ddp_train_step(model, optimizers, train_iter, cfg):
        """One optimizer step under DDP: all-reduce gradients exactly once,
        after the final accumulated micro-batch, and still average correctly."""
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=cfg.device)
        last = cfg.grad_accum_steps - 1
        for i in range(cfg.grad_accum_steps):
            batch = next(train_iter)
            x, y, pos, seq = (batch[k].to(cfg.device, non_blocking=True)
                              for k in BATCH_KEYS)
            # Suppress the DDP all-reduce on every micro-batch except the last.
            sync_ctx = contextlib.nullcontext() if i == last else model.no_sync()
            with sync_ctx:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = model(x, targets=y, position_ids=pos, seq_ids=seq)
                    loss = loss / cfg.grad_accum_steps
                loss.backward()
            loss_sum += loss.detach()
        return loss_sum
    ```

    **Why it changes wall-clock but not the gradient.** Suppressing the all-reduce on the first
    `grad_accum_steps - 1` backward passes only defers *communication*; the local `.grad` buffers
    still accumulate every micro-batch's contribution. Because gradients add linearly and all-reduce
    (sum/mean) is linear, reducing once over the summed local gradients is mathematically identical
    to reducing after each micro-batch — you just do it once instead of `grad_accum_steps` times.

    **The two other required changes.** (1) **Shard the data stream by rank.** Every rank must read
    disjoint samples, or you train on each token `world_size` times and the "effective batch" is a
    fiction; offset `ResumableShuffleSampler`'s `start` by `rank` and stride by `world_size`, and
    either divide `grad_accum_steps` by `world_size` or accept an 8× larger effective batch (which
    would then need its own LR retune). (2) **Rank-0-only I/O.** Only rank 0 should write
    checkpoints, append to `log.jsonl`, and print — otherwise eight processes race on the same
    `.tmp` file and `os.replace` no longer guarantees a coherent checkpoint. Add a `dist.barrier()`
    after saving. (A third, easy to miss: `qk_clip_` mutates parameters in place, so it must run on
    *every* rank with the same probe batch, or the replicas silently diverge.)

**9.** A colleague copies your loop into a new project but calls the model as
`model(x, targets=y, position_ids=pos)` — dropping `seq_ids`. The run completes; training loss is
about 0.05 nats *lower* than yours at the same step, and held-out perplexity on their own val split
also looks slightly better. (a) What is actually happening inside `Stack100M.forward`? (b) Why does
the loss go *down* rather than up? (c) Why is their val number not a valid rebuttal? (d) Write the
cheapest test that would have caught this in CI.

??? note "Solution"
    **(a)** With `seq_ids=None`, `T == kv_len` and `start_pos == 0`, so `_build_mask` returns `None`
    — the plain-causal fast path. SDPA runs with `is_causal=True` and every token attends to every
    earlier token *in the packed window*, across document boundaries. `position_ids` does not save
    them: it only indexes the RoPE tables, it does not build the mask.

    **(b)** Cross-document context is *extra information*, and a language model will happily use it.
    Tokens near the end of a window get to condition on hundreds of tokens of unrelated text, which
    on average makes the next token slightly easier to predict (shared topic drift, repeated
    boilerplate, and — after dedup imperfections — occasionally near-duplicate content). So the
    metric improves while the model learns a conditioning distribution that will never occur at
    inference time, where prompts do not come pre-pended with a random unrelated document.

    **(c)** Their val loader packs documents the same way, so the val split has the same leak. The
    metric is measured under the same broken conditioning, which makes the comparison
    self-consistent and meaningless. A held-out number only certifies what it measures; here it
    measures a train/serve mismatch rather than detecting it.

    **(d)** Assert the mask matters, on toy shapes, in a couple of milliseconds — no training
    required:

    ```python
    def test_document_mask_is_actually_applied():
        cfg = toy_config(); m = Stack100M(cfg).eval()
        x = torch.randint(0, cfg.vocab_size, (1, 16))
        seq = torch.tensor([[0]*8 + [1]*8])            # two documents in one window
        with torch.no_grad():
            masked, _ = m(x, seq_ids=seq)
            unmasked, _ = m(x)
            # Perturbing document 0 must NOT change document 1's outputs...
            x2 = x.clone(); x2[0, :8] = torch.randint(0, cfg.vocab_size, (8,))
            masked2, _ = m(x2, seq_ids=seq)
        assert torch.allclose(masked[:, 8:], masked2[:, 8:], atol=1e-5)   # isolated
        assert not torch.allclose(masked, unmasked, atol=1e-5)            # mask did something
    ```

    The first assertion is the real specification — *document 1's logits are independent of
    document 0's tokens* — and it fails loudly the moment `seq_ids` stops being threaded through.
    The second guards against a mask that is silently all-True.
