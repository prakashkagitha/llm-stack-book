# %% [markdown]
# # Fitting a Bigger Model: Activation Checkpointing + 8-bit Optimizer
#
# > **Hardware:** 1x H100 80GB. Runtime: a few minutes. Not executed in the
# > book — run it to get your own numbers.
#
# You'll build a small decoder-only transformer, measure exactly where its
# training-step memory goes (parameters, gradients, optimizer state,
# activations) with `torch.cuda.max_memory_allocated`, and then apply three
# independent levers — bf16 mixed precision, activation/gradient
# checkpointing, and an 8-bit AdamW — watching the peak-memory number drop
# after each one.
#
# See [the chapter](https://prakashkagitha.github.io/llm-stack-book/04-kernels-efficiency/10-memory-efficient-training.html)
# for the full explanation.

# %%
# bitsandbytes ships the 8-bit optimizers (Adam8bit, block-wise int8
# quantization of the Adam moment buffers). torch is preinstalled on the
# GPU image. We do NOT need flash-attn separately: torch's own
# `F.scaled_dot_product_attention` already dispatches to a fused
# flash-attention kernel on H100 for bf16 inputs.
%pip install -q bitsandbytes

import gc
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

assert torch.cuda.is_available(), "This notebook targets a CUDA GPU (1x H100 80GB)."
device = torch.device("cuda")
torch.manual_seed(0)

# We compute in bf16 for the "mixed precision" configs below (H100 tensor
# cores are fastest in bf16/fp8; bf16 has the same exponent range as fp32,
# so unlike fp16 it needs no loss-scaling).
BF16 = torch.bfloat16

print(f"device: {torch.cuda.get_device_name(0)}")
print(f"total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# %% [markdown]
# ## The model: a small GPT-style decoder
#
# Per the chapter's static-memory accounting, naive fp32 Adam training costs
# **16 bytes/parameter**: 4 bytes for the fp32 weight, 4 for the fp32
# gradient, and 8 for Adam's own `m_t`/`v_t` moment buffers (4 bytes each).
# On top of that sits **activation memory**, which scales with
# `batch_size x seq_len x n_layers` rather than with parameter count — for
# long sequences or large batches it can dwarf the static budget.
#
# We use `F.scaled_dot_product_attention` with `is_causal=True` for
# attention: on H100 with bf16 inputs it dispatches to a fused
# flash-attention kernel, so the full `(T, T)` attention-score matrix is
# never materialized — matching how real training stacks are built today.

# %%
@dataclass
class GPTConfig:
    vocab_size: int = 32000
    d_model: int = 1024
    n_layers: int = 12
    n_heads: int = 16
    ffn_mult: int = 4
    seq_len: int = 2048


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, H, T, head_dim)
        # Fused flash-attention kernel on H100 for bf16; never materializes
        # the (T, T) score matrix.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = cfg.ffn_mult * cfg.d_model
        self.fc1 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    """One pre-norm transformer block. This is the unit we checkpoint."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying
        self.use_checkpoint = False  # toggled per experiment below

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                # use_reentrant=False is the current recommended mode: it
                # does not save/restore RNG state globally and composes
                # better with torch.compile.
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)


cfg = GPTConfig()
n_params = sum(p.numel() for p in GPT(cfg).parameters())
print(f"model config: {cfg}")
print(f"parameter count: {n_params / 1e6:.1f}M")

# %% [markdown]
# ## Measurement methodology
#
# GPU kernels are launched asynchronously, so timing must go through
# `torch.cuda.Event` + `torch.cuda.synchronize()`, never `time.time()`
# around unsynchronized code. For memory we use
# `torch.cuda.reset_peak_memory_stats()` + `torch.cuda.max_memory_allocated()`,
# which tracks the single highest allocator watermark since the last reset
# — exactly the number that determines whether a step fits in 80 GB.
#
# For each configuration we:
# 1. build a fresh model + optimizer (dtype and checkpoint flag vary),
# 2. run a few **warmup** steps — this both lets cuBLAS/cuDNN pick and
#    cache kernels/workspaces, and lazily allocates Adam's `m_t`/`v_t`
#    buffers (PyTorch optimizers allocate state tensors on the *first*
#    `.step()`, not at construction time),
# 3. do one extra forward+backward (no `.step()`) to read off params/grad/
#    optimizer-state byte counts with grads actually populated,
# 4. reset peak stats and time+measure a few more full steps in steady
#    state — that peak is the number that matters in practice.

# %%
def gb(nbytes: float) -> float:
    return nbytes / (1024**3)


def param_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def grad_bytes(model: nn.Module) -> int:
    return sum(p.grad.numel() * p.grad.element_size() for p in model.parameters() if p.grad is not None)


def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    """Generic byte-counter over whatever tensors an optimizer's state holds.

    Works unmodified for torch.optim.AdamW (fp32 exp_avg/exp_avg_sq) and for
    bitsandbytes' Adam8bit (int8 state1/state2 + small per-block quantization
    maps/scales) since it just sums every tensor's numel * element_size.
    """
    total = 0
    for state in optimizer.state.values():
        for v in state.values():
            if torch.is_tensor(v):
                total += v.numel() * v.element_size()
    return total


def make_batch(cfg: GPTConfig, batch_size: int):
    idx = torch.randint(0, cfg.vocab_size, (batch_size, cfg.seq_len), device=device)
    targets = torch.randint(0, cfg.vocab_size, (batch_size, cfg.seq_len), device=device)
    return idx, targets


def timed_step(model, optimizer, idx, targets):
    """One full forward + backward + optimizer step, timed with CUDA events."""
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    logits = model(idx)
    # Loss in fp32 regardless of model dtype -- standard practice for
    # numerically stable cross-entropy under bf16/fp16 compute.
    loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    end_evt.record()
    torch.cuda.synchronize()
    return start_evt.elapsed_time(end_evt), loss.item()


def run_config(name, cfg, idx, targets, dtype, use_checkpoint, optimizer_factory,
                n_warmup=3, n_measure=5):
    """Build a fresh model+optimizer for one configuration and report its
    memory breakdown + step time. Returns a dict of results."""
    torch.manual_seed(0)
    model = GPT(cfg).to(device=device, dtype=dtype)
    model.use_checkpoint = use_checkpoint
    model.train()
    optimizer = optimizer_factory(model.parameters())

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    for _ in range(n_warmup):
        timed_step(model, optimizer, idx, targets)

    # One more forward+backward (no step) purely to read off a clean
    # component breakdown with grads populated.
    optimizer.zero_grad(set_to_none=True)
    logits = model(idx)
    loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))
    loss.backward()
    torch.cuda.synchronize()
    p_bytes = param_bytes(model)
    g_bytes = grad_bytes(model)
    o_bytes = optimizer_state_bytes(optimizer)  # populated by warmup .step() calls
    optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times_ms = []
    for _ in range(n_measure):
        t, _ = timed_step(model, optimizer, idx, targets)
        times_ms.append(t)
    peak_bytes = torch.cuda.max_memory_allocated()

    result = dict(
        name=name,
        param_gb=gb(p_bytes),
        grad_gb=gb(g_bytes),
        optstate_gb=gb(o_bytes),
        peak_gb=gb(peak_bytes),
        step_ms=sum(times_ms) / len(times_ms),
    )
    print(f"[{name}] params={result['param_gb']:.3f} GB  grads={result['grad_gb']:.3f} GB  "
          f"opt_state={result['optstate_gb']:.3f} GB  peak(step)={result['peak_gb']:.3f} GB  "
          f"avg_step={result['step_ms']:.1f} ms")

    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


results = []
BATCH_SIZE = 8
idx, targets = make_batch(cfg, BATCH_SIZE)
print(f"batch: {BATCH_SIZE} x {cfg.seq_len} tokens")

# %% [markdown]
# ## Lever 0: naive fp32 baseline
#
# Everything — weights, gradients, Adam's `m_t`/`v_t` — lives in fp32. This
# is exactly the chapter's **16 bytes/parameter** static-memory rule:
# 4 (weight) + 4 (grad) + 4 + 4 (Adam moments) = 16 bytes/param, on top of
# whatever the forward pass's activations cost. Expect the largest peak-
# memory number of all four configs below.

# %%
r0 = run_config(
    name="fp32 baseline",
    cfg=cfg, idx=idx, targets=targets,
    dtype=torch.float32,
    use_checkpoint=False,
    optimizer_factory=lambda params: torch.optim.AdamW(params, lr=3e-4),
)
results.append(r0)

# %% [markdown]
# ## Lever 1: + bf16 mixed precision
#
# Casting the whole model to bf16 roughly halves *every* per-element byte
# count: weights 4→2 bytes, grads 4→2 bytes, and (since `torch.optim.AdamW`
# allocates its state tensors in the parameter's own dtype) Adam's moments
# 4→2 bytes each too. Activations shrink by the same 2x factor. You should
# see roughly a **~2x drop** in every category and in the step's peak
# memory versus Lever 0 — and, thanks to H100 tensor cores, a noticeably
# *faster* step despite doing the "same" math.
#
# (Production pretraining recipes more often keep an fp32 *master* copy of
# the weights for optimizer updates and only compute the forward/backward
# in bf16 via `torch.autocast` — that trades some of this memory saving
# back for numerical stability. We use pure bf16 end-to-end here to keep
# the demo simple; see the chapter for the master-weights variant.)

# %%
r1 = run_config(
    name="bf16",
    cfg=cfg, idx=idx, targets=targets,
    dtype=BF16,
    use_checkpoint=False,
    optimizer_factory=lambda params: torch.optim.AdamW(params, lr=3e-4),
)
results.append(r1)
print(f"peak memory vs fp32 baseline: {r1['peak_gb'] / r0['peak_gb']:.2f}x")

# %% [markdown]
# ## Lever 2: + activation/gradient checkpointing
#
# `GPT.forward` already wraps each `Block` in `torch.utils.checkpoint.checkpoint`
# when `model.use_checkpoint=True`: instead of saving every block's
# activations for backward, PyTorch discards them after the forward pass and
# **recomputes** that block's forward during backward. Memory for stored
# activations drops from `O(n_layers)` toward `O(sqrt(n_layers))` (or `O(1)`
# with full per-block checkpointing, at the cost of one extra forward pass).
# This targets *only* the activation term — params/grads/optimizer state are
# unchanged from Lever 1.
#
# Expect the peak-memory number to drop again (on the order of a large
# fraction of the activation term, more pronounced at larger batch/sequence
# length than the modest batch=8 used here), paired with a modest increase
# in average step time — full per-block checkpointing recomputes roughly one
# extra forward pass, so a step-time increase on the order of tens of percent
# is the classic recompute-for-memory tradeoff.

# %%
r2 = run_config(
    name="bf16 + checkpoint",
    cfg=cfg, idx=idx, targets=targets,
    dtype=BF16,
    use_checkpoint=True,
    optimizer_factory=lambda params: torch.optim.AdamW(params, lr=3e-4),
)
results.append(r2)
print(f"peak memory vs bf16 (no checkpoint): {r2['peak_gb'] / r1['peak_gb']:.2f}x")
print(f"step time vs bf16 (no checkpoint):   {r2['step_ms'] / r1['step_ms']:.2f}x")

# %% [markdown]
# ## Lever 3: + 8-bit AdamW (bitsandbytes)
#
# `bitsandbytes.optim.Adam8bit` stores the `m_t`/`v_t` moment buffers as
# int8 with small per-block scale factors (Dettmers et al., *8-bit
# Optimizers via Block-wise Quantization*) instead of fp32 or bf16 — roughly
# `2P` bytes of optimizer state instead of `8P` for fp32 AdamW or `4P` for
# the bf16 AdamW of Lever 2. So relative to the fp32 baseline that term
# shrinks by about 4x, and relative to Lever 2's bf16 AdamW (the ratio
# printed below) by about 2x. This lever touches only the optimizer-state
# term, orthogonally to the bf16 and checkpointing levers already applied.
# `optimizer_state_bytes` below counts the actual int8 state +
# quantization-map/scale tensors bitsandbytes allocates, so the printed
# number reflects real overhead, not just the int8 payload.

# %%
try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    bnb = None
    HAS_BNB = False
    print("bitsandbytes is not importable -- run `%pip install bitsandbytes` "
          "and restart the kernel to try Lever 3.")

if HAS_BNB:
    r3 = run_config(
        name="bf16 + checkpoint + Adam8bit",
        cfg=cfg, idx=idx, targets=targets,
        dtype=BF16,
        use_checkpoint=True,
        optimizer_factory=lambda params: bnb.optim.Adam8bit(params, lr=3e-4),
    )
    results.append(r3)
    print(f"optimizer-state bytes vs bf16+checkpoint (fp32 AdamW moments): "
          f"{r3['optstate_gb'] / r2['optstate_gb']:.2f}x")

# %% [markdown]
# ## Summary across all four configurations
#
# One combined table, computed from the `results` list populated above —
# every number here comes from this run, not a canned constant.

# %%
header = f"{'config':<32}{'params(GB)':>12}{'grads(GB)':>12}{'opt(GB)':>10}{'peak(GB)':>11}{'step(ms)':>10}"
print(header)
print("-" * len(header))
for r in results:
    print(f"{r['name']:<32}{r['param_gb']:>12.3f}{r['grad_gb']:>12.3f}"
          f"{r['optstate_gb']:>10.3f}{r['peak_gb']:>11.3f}{r['step_ms']:>10.1f}")

# %% [markdown]
# ## What you should see
#
# - Peak training-step memory should decrease **monotonically** across the
#   four configs, roughly: fp32 baseline → ~half at bf16 → a further cut
#   from checkpointing → a smaller additional cut to the optimizer-state
#   column from Adam8bit. At this model's modest scale (~150-200M
#   parameters, batch 8 x 2048 tokens) the absolute numbers will be a small
#   number of GB — the point is the *shape* of the curve, which is what
#   matters when you scale this same recipe up to a 7B+ model that no
#   longer fits in 80 GB at all under Lever 0.
# - Expect the checkpointing step to cost modestly more wall-clock time per
#   step (on the order of tens of percent) than the equivalent
#   non-checkpointed bf16 run — you're trading recompute for memory, not
#   getting it for free.
# - Expect the Adam8bit optimizer-state column to shrink versus the bf16
#   `AdamW` run — by roughly 2x against Lever 2's bf16 moments (about 4x
#   against the fp32 baseline's fp32 moments) — while `param_gb`/`grad_gb`
#   stay essentially unchanged (Adam8bit only touches optimizer state, not
#   the model or its gradients).
#
# **Key takeaways**
# 1. Peak step memory = static (params + grads + optimizer state) +
#    activations — each of the three levers here attacks exactly one of
#    those terms, so they compose (you can and typically should stack all
#    three).
# 2. Checkpointing is a pure memory-for-compute trade: it never reduces
#    static memory, only the activations you'd otherwise have to hold live.
# 3. bf16 is close to a free lunch on H100 — it shrinks nearly every tensor
#    by 2x while also using faster tensor-core math paths.
# 4. 8-bit optimizer state is likewise close to free — it targets the
#    optimizer-state term only and is orthogonal to precision and
#    checkpointing choices.
#
# **Next step:** once these levers together still don't get a large model's
# static budget under 80 GB, the next tool is sharding that budget across
# GPUs — see [Distributed Training I: Data Parallelism, DDP, ZeRO & FSDP](https://prakashkagitha.github.io/llm-stack-book/03-pretraining/05-distributed-data-parallel.html),
# or shrink the *trainable* parameter count itself with LoRA/QLoRA, covered
# in [the source chapter](https://prakashkagitha.github.io/llm-stack-book/04-kernels-efficiency/10-memory-efficient-training.html)'s
# PEFT section.
