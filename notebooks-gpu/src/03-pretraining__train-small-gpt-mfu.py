# %% [markdown]
# # Train a Small GPT on One H100: Loss Curve, Throughput & MFU
#
# > **Hardware:** 1x H100 80GB. Runtime: a few minutes. Not executed in the book — run it to get your own numbers.
#
# You will train a real ~124M-parameter GPT end to end in bf16 on a single GPU, then measure the two numbers that tell you whether your training loop is actually using the hardware: **tokens/s** and **MFU** (model FLOPs utilization).
#
# See [the chapter](https://prakashkagitha.github.io/llm-stack-book/03-pretraining/17-end-to-end-pretrain-recipe.html) for the full explanation.

# %%
%pip install -q numpy matplotlib

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataclasses import dataclass

# torch is preinstalled on GPU images; numpy/matplotlib are the only extra deps.
# No flash-attn / transformers / bitsandbytes needed: PyTorch's built-in
# F.scaled_dot_product_attention already dispatches to a fused, Flash-Attention-
# style kernel on H100 for bf16 inputs -- see the kernels & efficiency chapters.

assert torch.cuda.is_available(), "This notebook targets 1x H100 80GB; no CUDA device found."
device = "cuda"
device_name = torch.cuda.get_device_name(0)
assert torch.cuda.is_bf16_supported(), f"{device_name} does not report bf16 support."

# bf16 is the standing default for pretraining (wide dynamic range, no loss-scaler
# needed unlike fp16, native Tensor Core support) -- see Mixed Precision, bf16 &
# FP8 Training (chapter 3.8) for why bf16 beats fp16 here.
compute_dtype = torch.bfloat16

torch.manual_seed(1337)
np.random.seed(1337)

# TF32 for any residual fp32 matmuls (norms, etc.); harmless alongside bf16 autocast.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print(f"Device: {device_name}")
print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %% [markdown]
# ## The model: the same GPT from chapter 2.7
#
# We reuse the exact `GPT` class built and explained in [Building a GPT From Scratch](https://prakashkagitha.github.io/llm-stack-book/02-transformer/07-build-gpt-from-scratch.html): pre-norm blocks, fused QKV attention via `F.scaled_dot_product_attention` (a Flash-Attention-style kernel), a 4x-expansion GELU MLP, weight-tied embeddings, and the GPT-2 scaled-residual init. Nothing here is new architecture — this notebook's job is to *run* it on real hardware and *measure* it.
#
# We pick `n_layer=12, n_head=12, n_embd=768, block_size=1024, vocab_size=50304` — the standard "single-GPU 124M" (GPT-2-small) configuration (50304 = GPT-2's 50257-token vocab padded up to a multiple of 64, which keeps every matmul dimension friendly to Tensor Cores).

# %%
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304   # GPT-2 vocab (50257) padded to a multiple of 64
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False


class LayerNorm(nn.Module):
    """LayerNorm with an optional bias (torch.nn.LayerNorm always has one)."""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)   # each (B, T, C)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)    # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        # Fused kernel: causal mask + scaling + softmax + dropout in one call.
        # On H100 with bf16 inputs this dispatches to a Flash-Attention-style
        # backend automatically -- no separate flash-attn install needed.
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # pre-norm attention, residual add
        x = x + self.mlp(self.ln_2(x))    # pre-norm MLP,       residual add
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight   # weight tying

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                # GPT-2's scaled residual init: keeps the residual stream's
                # variance from growing linearly with depth.
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / (2 * config.n_layer) ** 0.5)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"sequence length {T} > block_size {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :])   # inference: only need the last position
            loss = None
        return logits, loss

# %% [markdown]
# ## Instantiate on the GPU, count parameters, and (optionally) compile
#
# We move the model to the H100 with fp32 master weights — bf16 arrives later via `torch.autocast`, not by casting the parameters themselves (the standard "bf16 autocast, fp32 master weights" recipe). We also try `torch.compile`: on H100 it typically fuses elementwise ops and can meaningfully raise MFU, but the first few steps pay a one-time compilation/autotune cost — which is exactly why the measurement loop below discards a handful of warmup steps before averaging.
#
# Expect roughly 120-130M parameters printed below — the well-known GPT-2-small count (~124M), since we deliberately reproduce that single-GPU configuration.

# %%
config = GPTConfig(block_size=1024, vocab_size=50304, n_layer=12, n_head=12, n_embd=768, dropout=0.0, bias=False)
model = GPT(config).to(device)   # fp32 master weights; bf16 arrives via autocast below

n_params = sum(p.numel() for p in model.parameters())   # lm_head is tied to wte, so no double count
print(f"n_layer={config.n_layer} n_head={config.n_head} n_embd={config.n_embd} "
      f"block_size={config.block_size} vocab_size={config.vocab_size}")
print(f"Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

try:
    compiled_model = torch.compile(model)
    print("torch.compile: enabled")
except Exception as e:
    compiled_model = model
    print(f"torch.compile unavailable, running eager: {e}")

# %% [markdown]
# ## A tiny synthetic token stream (offline, no download)
#
# For a self-contained demo we generate tokens from a simple order-1 Markov chain over a reduced 256-token "active" alphabet, embedded in the full `vocab_size=50304` (unused ids simply never appear, same as a real corpus's long tail of rare tokens). Unlike i.i.d. random tokens — whose per-token entropy is exactly `ln(vocab_size)` and *never drops*, however long you train — a Markov chain has learnable short-range structure that a single attention layer can pick up almost immediately, so you get a genuine, fast-moving loss curve without needing a real dataset or network access.
#
# Swap this for a real tokenized corpus (see [Pretraining Data](https://prakashkagitha.github.io/llm-stack-book/03-pretraining/01-pretraining-data.html) and its uint16 `.bin` shard convention) to see the slower, harder-won loss curve of real language — and note the very different scaling regime that governs it in [Scaling Laws](https://prakashkagitha.github.io/llm-stack-book/03-pretraining/04-scaling-laws.html).
#
# We also build the standard weight-decay-split AdamW and the linear-warmup/cosine-decay schedule used throughout the book.

# %%
active_vocab = 256          # the chain only ever emits these ids (< vocab_size)
concentration = 0.15        # Dirichlet alpha: small -> peaked/learnable transition rows

rng = np.random.default_rng(0)
transition = rng.dirichlet(alpha=np.full(active_vocab, concentration), size=active_vocab)  # (V, V), rows sum to 1
cum_transition = np.cumsum(transition, axis=1)


def sample_markov_corpus(n_tokens, seed=0):
    """Simulate the order-1 chain via inverse-CDF sampling. Pure CPU/numpy,
    runs once before any GPU timing -- a few seconds for a few hundred
    thousand tokens."""
    draw_rng = np.random.default_rng(seed)
    u = draw_rng.random(n_tokens)
    toks = np.empty(n_tokens, dtype=np.int64)
    toks[0] = 0
    for t in range(1, n_tokens):
        toks[t] = np.searchsorted(cum_transition[toks[t - 1]], u[t])
    return toks.astype(np.uint16)


corpus = sample_markov_corpus(300_000)
n_train = int(0.9 * len(corpus))
train_data, val_data = corpus[:n_train], corpus[n_train:]

# Analytic entropy floor: the best achievable expected loss once the model has
# fully learned the transition matrix (compare against ln(vocab_size) at step 0).
row_entropy = -(transition * np.log(transition + 1e-12)).sum(axis=1)
entropy_floor = float(row_entropy.mean())
print(f"Synthetic corpus: {len(corpus):,} tokens, active_vocab={active_vocab}")
print(f"ln(vocab_size)      = {math.log(config.vocab_size):.2f} nats  (expected loss at step 0)")
print(f"chain entropy floor = {entropy_floor:.2f} nats  (best achievable once learned)")


def get_batch(split, block_size, batch_size, device):
    """Sample batch_size random contiguous windows; y is x shifted by one --
    the whole autoregressive labelling."""
    data = train_data if split == "train" else val_data
    ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + 1 + block_size] for i in ix]).astype(np.int64)
    return (torch.from_numpy(x).to(device, non_blocking=True),
            torch.from_numpy(y).to(device, non_blocking=True))


def configure_optimizer(model, weight_decay, lr, betas, device_type):
    # 2D+ params (matmul weights, embeddings) get weight decay; 1D params
    # (biases, norm gains) don't -- shrinking those toward zero hurts.
    params = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay = [p for p in params.values() if p.dim() >= 2]
    no_decay = [p for p in params.values() if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=(device_type == "cuda"))


def lr_at(step, warmup, total, base_lr, min_lr):
    """Linear warmup into cosine decay -- the schedule used throughout the book."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    r = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * r))

# %% [markdown]
# ## Measuring throughput and MFU correctly
#
# GPU kernels launch asynchronously, so `time.time()` around a training step mostly measures how fast the CPU can *enqueue* work, not how long the GPU spent computing — it can under- or over-count by a large factor depending on queue depth. The correct tool is `torch.cuda.Event(enable_timing=True)`: two events bracket the step, and `torch.cuda.synchronize()` blocks until the GPU has actually reached the closing event before we read the elapsed time. We also discard the first several steps from every average — `torch.compile`'s autotuning and cuDNN's kernel benchmarking make early steps unrepresentative.
#
# One more subtlety the loop below handles: we sample each step's micro-batches (CPU numpy + host-to-device copy) *before* recording `start_evt`, so the timed region contains only the forward/backward/step GPU work and the throughput/MFU numbers reflect compute, not dataloading.
#
# **MFU** (model FLOPs utilization) is the fraction of the GPU's peak arithmetic throughput your training loop actually achieves. With the standard 6N-FLOPs-per-token rule for a forward+backward pass (derived in [The Roofline Model & Performance Engineering](https://prakashkagitha.github.io/llm-stack-book/04-kernels-efficiency/01-roofline-performance.html)):
#
# ```
# MFU = 6 * N_params * tokens_per_sec / peak_FLOPs_per_sec
# ```
#
# We use the H100 SXM's bf16 dense (no sparsity) peak of roughly 990 TFLOP/s as the denominator. Expect very roughly **35-50% MFU** from this notebook's simple loop (`torch.compile` + SDPA's fused attention, no further kernel tuning) — healthy for a single GPU; production-grade runs with heavier kernel fusion and larger batches can push higher. Treat that band as an order-of-magnitude expectation, not a guarantee: your number depends on batch size, `torch.compile` warmup, and driver/kernel versions.

# %%
H100_BF16_PEAK_TFLOPS = 990.0   # bf16 dense (no sparsity) peak, per H100 SXM spec


def mfu(num_params, tokens_per_sec, peak_tflops=H100_BF16_PEAK_TFLOPS, num_gpus=1):
    """6N-FLOPs-per-token rule: forward+backward costs ~6 FLOPs per parameter
    per token (see the roofline chapter for the derivation)."""
    flops_per_token = 6 * num_params
    achieved_flops_per_sec = flops_per_token * tokens_per_sec
    peak_flops_per_sec = num_gpus * peak_tflops * 1e12
    return achieved_flops_per_sec / peak_flops_per_sec


# Sanity check the arithmetic on an illustrative example: a 124M model at
# ~170,000 tok/s on a single A100 (bf16 dense peak ~312 TFLOP/s) works out to
# ~40% MFU. This just confirms the formula wiring -- it is not a measured result.
_sanity = mfu(num_params=124e6, tokens_per_sec=170_000, peak_tflops=312.0)
print(f"Sanity check (illustrative A100 example): MFU = {_sanity * 100:.1f}%")

# %% [markdown]
# ## The training loop: grad accumulation, bf16 autocast, clipping, and the two measurements
#
# Each optimizer step runs `grad_accum` micro-batches under `torch.autocast(dtype=torch.bfloat16)`, backpropagating a `1/grad_accum`-scaled loss into the same gradient buffers before a single clipped `AdamW.step()` — the standard accumulation pattern (under DDP each micro-step would be wrapped in `model.no_sync()`; irrelevant here since we're on one GPU). We keep `n_steps` small (a few dozen) so this cell finishes in a couple of minutes, not because that's how many steps a real pretraining run takes — a real run is thousands to tens of thousands of steps.
#
# What to expect: loss should start near `ln(vocab_size) ~= 10.8` nats and fall quickly over the first ~10-20 steps toward the synthetic chain's entropy floor printed above, since an order-1 Markov chain is about the easiest possible structure for a single attention layer to learn. Tokens/s and MFU should each settle into a fairly narrow band once the warmup steps are excluded — if MFU keeps climbing or falling steadily rather than settling, you likely need more warmup steps, or are still inside `torch.compile`'s autotuning window.

# %%
# --- Training / measurement hyperparameters ---------------------------------
local_bsz      = 8     # micro-batch size per forward/backward pass
grad_accum     = 4     # micro-steps per optimizer step
n_warmup_steps = 5     # excluded from throughput/MFU averages (compile/cuDNN warmup)
n_steps        = 40    # total optimizer steps -- a demo, not a real run (see above)
lr_warmup      = 10
base_lr        = 6e-4  # a standard 124M peak LR (GPT-2-small / nanoGPT scale)
min_lr         = base_lr * 0.1
grad_clip      = 1.0
weight_decay   = 0.1
betas          = (0.9, 0.95)

opt = configure_optimizer(model, weight_decay, base_lr, betas, device_type="cuda")
tokens_per_step = local_bsz * config.block_size * grad_accum
print(f"tokens/step = {local_bsz} x {config.block_size} x {grad_accum} = {tokens_per_step:,}")

loss_history, tokens_per_sec_history, mfu_history, step_ms_history = [], [], [], []

torch.cuda.reset_peak_memory_stats()
compiled_model.train()
for step in range(n_steps):
    for g in opt.param_groups:
        g["lr"] = lr_at(step, lr_warmup, n_steps, base_lr, min_lr)

    # Sample this step's micro-batches OUTSIDE the timed region so CPU numpy
    # work and the host-to-device copies don't inflate the measured GPU time.
    batches = [get_batch("train", config.block_size, local_bsz, device)
               for _ in range(grad_accum)]

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    opt.zero_grad(set_to_none=True)
    step_loss_t = torch.zeros((), device=device)

    start_evt.record()
    for x, y in batches:                      # gradient accumulation: G micro-steps per opt step
        with torch.autocast(device_type="cuda", dtype=compute_dtype):
            logits, loss = compiled_model(x, y)
            loss = loss / grad_accum          # mean over G micro-batches
        loss.backward()
        step_loss_t += loss.detach()          # stays on-device -- no host sync inside the timed region
    grad_norm_t = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)   # global-norm clip, post-accum, pre-step
    opt.step()
    end_evt.record()

    torch.cuda.synchronize()                  # the one sync point: now it's safe to read timings + scalars
    step_ms = start_evt.elapsed_time(end_evt)  # milliseconds, GPU wall-clock for this optimizer step
    step_loss = step_loss_t.item()
    grad_norm = grad_norm_t.item()

    step_tokens_per_sec = tokens_per_step / (step_ms / 1000.0)
    step_mfu = mfu(n_params, step_tokens_per_sec)

    loss_history.append(step_loss)
    step_ms_history.append(step_ms)
    tokens_per_sec_history.append(step_tokens_per_sec)
    mfu_history.append(step_mfu)

    if step % 5 == 0 or step == n_steps - 1:
        print(f"step {step:3d} | loss {step_loss:6.3f} | grad_norm {grad_norm:5.2f} | "
              f"{step_ms:7.1f} ms/step | {step_tokens_per_sec:10,.0f} tok/s | MFU {step_mfu * 100:5.1f}%")

steady = slice(n_warmup_steps, None)
avg_tokens_per_sec = float(np.mean(tokens_per_sec_history[steady]))
avg_mfu = float(np.mean(mfu_history[steady]))
peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

print(f"\nSteady-state average (steps {n_warmup_steps}-{n_steps - 1}, "
      f"first {n_warmup_steps} warmup steps excluded):")
print(f"  avg tokens/s : {avg_tokens_per_sec:,.0f}")
print(f"  avg MFU      : {avg_mfu * 100:.1f}%")
print(f"  peak memory  : {peak_mem_gb:.1f} GB allocated (of 80 GB total)")

# %% [markdown]
# ## Plotting the loss curve, throughput, and MFU
#
# Three views of the same run: the loss curve (with the two reference lines computed above), tokens/s per step, and MFU per step. The first few (warmup) steps are shaded to make clear they're excluded from the steady-state averages printed above.

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
steps_x = np.arange(len(loss_history))
warmup_end = n_warmup_steps

# --- Loss ---
ax = axes[0]
ax.axvspan(-0.5, warmup_end - 0.5, color="0.9", zorder=0)   # shade excluded warmup steps
ax.plot(steps_x, loss_history, color="#2a78d6", linewidth=2)
ax.axhline(math.log(config.vocab_size), color="0.5", linestyle="--", linewidth=1)
ax.text(steps_x[-1], math.log(config.vocab_size), "  ln(vocab_size)",
        va="bottom", ha="right", color="0.4", fontsize=8)
ax.axhline(entropy_floor, color="0.5", linestyle=":", linewidth=1)
ax.text(steps_x[-1], entropy_floor, "  chain entropy floor",
        va="top", ha="right", color="0.4", fontsize=8)
ax.set_xlabel("step"); ax.set_ylabel("train loss (nats)"); ax.set_title("Loss")
ax.grid(alpha=0.3)

# --- Tokens/s ---
ax = axes[1]
ax.axvspan(-0.5, warmup_end - 0.5, color="0.9", zorder=0)
ax.plot(steps_x, tokens_per_sec_history, color="#eb6834", linewidth=2)
ax.set_xlabel("step"); ax.set_ylabel("tokens / s"); ax.set_title("Throughput")
ax.grid(alpha=0.3)

# --- MFU ---
ax = axes[2]
ax.axvspan(-0.5, warmup_end - 0.5, color="0.9", zorder=0)
ax.plot(steps_x, [m * 100 for m in mfu_history], color="#1baf7a", linewidth=2)
ax.axhspan(35, 50, color="#1baf7a", alpha=0.08, zorder=0)   # the healthy band discussed above
ax.set_xlabel("step"); ax.set_ylabel("MFU (%)"); ax.set_title("Model FLOPs Utilization")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("loss_throughput_mfu.png", dpi=150)
plt.show()

# %% [markdown]
# ## What you should see
#
# - **Loss**: starts near `ln(vocab_size) ~= 10.8` nats and drops quickly toward the synthetic chain's entropy floor (printed above, typically a few nats lower) within the first few dozen steps — this synthetic task is deliberately trivial (order-1 structure) so the drop is visible in a couple of minutes; a real corpus grinds far more slowly and never fully plateaus at this scale (see the chapter's discussion of the "healthy loss curve" shape).
# - **Tokens/s**: this model at this context length on one H100 should land on the order of a few hundred thousand tokens/s — roughly what you'd expect by scaling a representative single-A100 figure (order 10^5 tok/s at ~40% MFU on ~312 TFLOP/s bf16) up by the H100's roughly 3x higher bf16 peak (~990 vs. ~312 TFLOP/s) at a comparable MFU. Treat this as an order-of-magnitude planning number, not a guarantee; your actual figure depends on batch size, `torch.compile` warmup, and driver/kernel versions.
# - **MFU**: very roughly **35-50%** is a healthy band for a single GPU with `torch.compile` and SDPA's fused attention and no further kernel tuning. Well below ~30% with an otherwise-idle-looking GPU points at a dataloader, batch-size, or kernel-selection problem rather than a model problem.
# - **Memory**: at `local_bsz=8, ctx=1024` a 124M model should use a small fraction of the H100's 80GB — plenty of headroom to raise `local_bsz` or `ctx` before you need FSDP or activation checkpointing (see [Memory-Efficient Training](https://prakashkagitha.github.io/llm-stack-book/04-kernels-efficiency/10-memory-efficient-training.html)).
#
# **Key takeaways:**
#
# 1. Never time GPU work with `time.time()` — asynchronous kernel launches make it lie; bracket with `torch.cuda.Event` and `synchronize()` instead.
# 2. MFU turns "tokens/s" into a hardware-relative number you can compare across GPUs and model sizes — and it is *linear* in the parameter count you plug in, so a wrong `N` silently scales your reported MFU by the same factor.
# 3. Discard warmup steps before averaging throughput, and keep dataloading out of the timed region — `torch.compile` autotuning and cuDNN benchmarking make the first several steps unrepresentative of steady state.
# 4. This model size and MFU band is the "is my training loop healthy" checkpoint before you scale up — the next question, "how big a model and how many tokens should I actually train," is the subject of [Scaling Laws](https://prakashkagitha.github.io/llm-stack-book/03-pretraining/04-scaling-laws.html).
