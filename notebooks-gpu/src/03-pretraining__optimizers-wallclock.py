# %% [markdown]
# # AdamW vs Lion vs Muon: Wall-Clock & Memory on GPU
#
# > **Hardware:** 1x H100 80GB. Runtime: a few minutes. Not executed in the book — run it to get your own numbers.
#
# You will train the same small GPT-style model with three optimizers — AdamW, Lion (sign-based momentum), and a Muon+AdamW hybrid (Newton-Schulz orthogonalized momentum for 2-D weights) — and measure their optimizer-state memory, per-step wall-clock, and loss curves on a real GPU.
#
# See [the chapter](https://prakashkagitha.github.io/llm-stack-book/03-pretraining/09-optimizers.html) for the full explanation.

# %%
# matplotlib is used only for the final loss-curve plots; torch is preinstalled.
%pip install -q matplotlib

import gc
import math
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
import matplotlib.pyplot as plt

assert torch.cuda.is_available(), "This notebook targets a CUDA GPU (1x H100). Run it on a GPU instance."
device = torch.device("cuda")

# TF32 matmuls are a free throughput win on Ampere/Hopper tensor cores.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.manual_seed(0)
print("Device:", torch.cuda.get_device_name(0))
print("bf16 supported:", torch.cuda.is_bf16_supported())

# %% [markdown]
# ## A small GPT to optimize
#
# We need a model with both 2-D weight matrices (attention/MLP projections — these are what
# Muon orthogonalizes) and 1-D / embedding parameters (LayerNorm gains, token & position
# embeddings, the LM head — these stay on AdamW per the chapter's hybrid recipe). A ~10-20M
# parameter GPT trained on a synthetic token stream is enough to see real, distinct
# optimizer behavior in a few minutes on one H100.
#
# Linear layers use `bias=False` (common in modern LLMs) so the only 1-D parameters are the
# LayerNorm weight/bias pairs, plus the embedding tables and LM head (which are 2-D but are
# *excluded* from Muon by convention — see the chapter's "why the embeddings get AdamW" note).

# %%
class CausalSelfAttention(nn.Module):
    """Standard multi-head causal self-attention via SDPA (real, fused GPU kernel)."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)   # 2-D: Muon-eligible
        self.proj = nn.Linear(d_model, d_model, bias=False)      # 2-D: Muon-eligible

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # scaled_dot_product_attention dispatches to the flash/mem-efficient kernel on H100.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, d_model, mult=4):
        super().__init__()
        self.fc1 = nn.Linear(d_model, mult * d_model, bias=False)  # 2-D: Muon-eligible
        self.fc2 = nn.Linear(mult * d_model, d_model, bias=False)  # 2-D: Muon-eligible
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)   # 1-D weight+bias: stays on AdamW
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)   # 1-D weight+bias: stays on AdamW
        self.mlp = MLP(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, max_seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)   # 2-D but excluded from Muon
        self.pos_emb = nn.Embedding(max_seq_len, d_model)  # 2-D but excluded from Muon
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)  # 2-D but excluded from Muon

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.lm_head(x)


# Model / data hyperparameters — small enough to train in a few minutes on one H100.
VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEADS, SEQ_LEN, BATCH_SIZE = 256, 384, 4, 6, 128, 32


def make_model():
    """Fresh model with a FIXED init so every optimizer starts from identical weights."""
    torch.manual_seed(1234)
    return TinyGPT(VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEADS, SEQ_LEN).to(device)


_probe = make_model()
n_params = sum(p.numel() for p in _probe.parameters())
n_2d = sum(p.numel() for n, p in _probe.named_parameters()
           if p.ndim == 2 and not any(s in n for s in ("tok_emb", "pos_emb", "lm_head")))
print(f"Total params: {n_params:,}  (2-D Muon-eligible: {n_2d:,}, "
      f"{100*n_2d/n_params:.1f}% of all params)")
del _probe

# %% [markdown]
# ## Three optimizers, three memory footprints
#
# **AdamW** (Kingma & Ba, [*Adam*, 2015](https://arxiv.org/abs/1412.6980); Loshchilov & Hutter,
# [*Decoupled Weight Decay Regularization*, 2019](https://arxiv.org/abs/1711.05101)) stores two
# extra tensors per parameter (`exp_avg`, `exp_avg_sq`). Since our params are fp32 here (no
# separate bf16-weights/fp32-master split, unlike a production bf16-mixed-precision trainer),
# that is roughly **8 bytes/param** of optimizer state.
#
# **Lion** (Chen et al., [*Symbolic Discovery of Optimization Algorithms*, 2023](https://arxiv.org/abs/2302.06675))
# keeps one momentum buffer and takes a sign-based update — roughly **4 bytes/param**.
#
# **Muon** (Jordan et al., 2024) keeps one momentum buffer per 2-D weight,
# orthogonalized via a Newton-Schulz iteration — roughly 4 bytes/param on the matrices, with a
# small auxiliary AdamW (roughly 8 bytes/param) for the embeddings/LM head/norms. Below we
# implement Lion and Muon from scratch, matching the chapter's reference code, as real
# `torch.optim.Optimizer` subclasses.

# %%
@torch.no_grad()
def newton_schulz5(G, steps=5, eps=1e-7):
    """Approximate UV^T (the orthogonalization of G's SVD) via Newton-Schulz iteration.
    Matmul-only (runs at full GPU throughput in bf16); coefficients from Jordan et al.
    tuned so ~5 iterations push all singular values toward 1.
    """
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = False
    if X.size(0) > X.size(1):          # iterate on the smaller dimension for less compute
        X = X.T
        transposed = True
    X = X / (X.norm() + eps)           # spectral pre-normalization
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(Optimizer):
    """Muon (Jordan et al., 2024): momentum orthogonalized via Newton-Schulz.
    Valid ONLY for 2-D weight matrices. One momentum buffer per parameter (4 bytes/param).
    """
    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, mu = group["lr"], group["momentum"]
            ns_steps, wd = group["ns_steps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                assert p.ndim == 2, "Muon only supports 2-D weight matrices"
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buf"] = torch.zeros_like(p)   # ONE buffer, not two
                buf = state["momentum_buf"]
                buf.mul_(mu).add_(g)                              # heavy-ball momentum
                update = newton_schulz5(buf, steps=ns_steps).to(p.dtype)  # orthogonalize
                if wd != 0:
                    p.mul_(1.0 - lr * wd)                         # decoupled decay
                # sqrt(max(rows,cols)) scale makes the per-element update RMS ~= lr,
                # independent of matrix shape (chapter section on Muon's update scaling).
                scale = max(p.shape) ** 0.5
                p.add_(update, alpha=-lr * scale)
        return loss


class Lion(Optimizer):
    """Lion (Chen et al., 2023): sign of an interpolated momentum. ONE buffer/param."""
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, (b1, b2) = group["lr"], group["betas"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                m = state["exp_avg"]
                if wd != 0:
                    p.mul_(1.0 - lr * wd)                          # decoupled decay
                c = m.mul(b1).add(g, alpha=1.0 - b1)               # temp, NOT stored
                p.add_(c.sign(), alpha=-lr)                        # every coord moves +/-lr
                m.mul_(b2).add_(g, alpha=1.0 - b2)                 # buffer uses beta2 & raw g
        return loss


def split_params_for_muon(model):
    """2-D hidden weight matrices -> Muon; everything else (norms, embeddings, LM head) -> AdamW."""
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if p.ndim == 2 and not any(s in name for s in ("tok_emb", "pos_emb", "lm_head")):
            muon_params.append(p)
        else:
            adamw_params.append(p)
    return muon_params, adamw_params


def build_config(name):
    """Fresh model + the optimizer(s) for one of 'adamw' / 'lion' / 'muon'."""
    model = make_model()
    if name == "adamw":
        optimizers = [torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95),
                                         weight_decay=0.1, fused=True)]
    elif name == "lion":
        # Lion needs a smaller LR and larger decay than AdamW (chapter's practitioner tip).
        optimizers = [Lion(model.parameters(), lr=3e-5, betas=(0.9, 0.99), weight_decay=1.0)]
    elif name == "muon":
        muon_params, adamw_params = split_params_for_muon(model)
        optimizers = [
            Muon(muon_params, lr=0.02, momentum=0.95, ns_steps=5, weight_decay=0.0),
            torch.optim.AdamW(adamw_params, lr=3e-4, betas=(0.9, 0.95),
                               weight_decay=0.1, fused=True),
        ]
    else:
        raise ValueError(name)
    return model, optimizers


CONFIGS = ["adamw", "lion", "muon"]

# %% [markdown]
# ## Measuring optimizer-state memory
#
# Optimizer state tensors (`exp_avg`, `exp_avg_sq`, Muon's momentum buffer, ...) are allocated
# **lazily**, on the first `.step()` call. So the recipe is: run one forward+backward to
# populate `.grad`, snapshot allocated memory, call `.step()` once, snapshot again. We report
# three numbers: the `memory_allocated` delta across the step (a direct measure of the state
# tensors), `torch.cuda.max_memory_allocated()` over the whole probe (includes activations —
# an upper bound), and an exact analytic sum of every state tensor's bytes (the ground truth,
# independent of the CUDA allocator's behavior).

# %%
def measure_optimizer_memory(name, x, y):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model, optimizers = build_config(name)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    loss.backward()
    torch.cuda.synchronize()
    mem_before_step = torch.cuda.memory_allocated()

    for opt in optimizers:
        opt.step()          # <-- optimizer state tensors are allocated here, lazily
    torch.cuda.synchronize()
    mem_after_step = torch.cuda.memory_allocated()
    peak_mb = torch.cuda.max_memory_allocated() / 1e6

    # Analytic ground truth: sum bytes of every tensor actually stored in optimizer state.
    state_bytes = sum(
        v.numel() * v.element_size()
        for opt in optimizers for st in opt.state.values() for v in st.values()
        if torch.is_tensor(v)
    )
    n_p = sum(p.numel() for p in model.parameters())

    result = dict(
        name=name,
        delta_mb=(mem_after_step - mem_before_step) / 1e6,
        peak_mb=peak_mb,
        bytes_per_param=state_bytes / n_p,
    )
    del model, optimizers
    gc.collect()
    torch.cuda.empty_cache()
    return result


x_probe = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
y_probe = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)

print(f"{'optimizer':<8} {'state delta (MB)':>18} {'peak alloc (MB)':>16} {'bytes/param':>12}")
for cfg in CONFIGS:
    r = measure_optimizer_memory(cfg, x_probe, y_probe)
    print(f"{r['name']:<8} {r['delta_mb']:>18.2f} {r['peak_mb']:>16.2f} {r['bytes_per_param']:>12.2f}")

# %% [markdown]
# ## Measuring per-step wall-clock
#
# Muon's Newton-Schulz iteration adds real matmul compute to the optimizer step itself, on top
# of what AdamW/Lion do. We isolate that cost by timing the `.step()` call alone (after
# `.backward()`, gradients already populated) with `torch.cuda.Event`, separately from the
# full forward+backward+step time. Warmup iterations are excluded (kernel selection / cuDNN
# autotuning / lazy CUDA context work happens on the first few calls) and every timed region
# is followed by `torch.cuda.synchronize()` before reading the elapsed time.

# %%
def benchmark_step_time(name, x, y, n_warmup=10, n_iters=30):
    model, optimizers = build_config(name)

    def run_once():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        for opt in optimizers:
            opt.step()

    for _ in range(n_warmup):     # warmup: not timed
        run_once()
    torch.cuda.synchronize()

    step_ms, full_ms = [], []
    for _ in range(n_iters):
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        ev_full_start = torch.cuda.Event(enable_timing=True)
        ev_step_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)

        ev_full_start.record()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        ev_step_start.record()
        for opt in optimizers:
            opt.step()
        ev_end.record()
        torch.cuda.synchronize()

        step_ms.append(ev_step_start.elapsed_time(ev_end))       # optimizer .step() only
        full_ms.append(ev_full_start.elapsed_time(ev_end))       # fwd + bwd + step

    del model, optimizers
    gc.collect()
    torch.cuda.empty_cache()
    return dict(name=name, step_ms=statistics.median(step_ms), full_ms=statistics.median(full_ms))


print(f"{'optimizer':<8} {'median step() ms':>18} {'median full-step ms':>20}")
for cfg in CONFIGS:
    r = benchmark_step_time(cfg, x_probe, y_probe)
    print(f"{r['name']:<8} {r['step_ms']:>18.3f} {r['full_ms']:>20.3f}")

# %% [markdown]
# ## Loss vs. step and loss vs. wall-clock
#
# Now a real (if tiny) training run for each optimizer. We use a synthetic periodic token
# stream — a random fixed-length "motif" repeated with injected noise — so the model has
# actual learnable structure to fit (pure i.i.d. random tokens would keep the loss pinned near
# `log(vocab_size)` forever, which would tell us nothing about optimizer *dynamics*). Each
# training step is timed with `torch.cuda.Event` and accumulated into a cumulative wall-clock
# axis, so we can plot loss against both step count and actual GPU time.

# %%
def make_synthetic_corpus(vocab_size=VOCAB_SIZE, period=64, length=200_000,
                           noise_frac=0.15, seed=0):
    """A repeating random 'motif' of `period` tokens with `noise_frac` random tokens mixed
    in. Learnable (the model can predict most positions from the periodic pattern) but not
    trivially memorizable in one pass, giving a real, non-degenerate loss curve."""
    g = torch.Generator().manual_seed(seed)
    motif = torch.randint(0, vocab_size, (period,), generator=g)
    reps = length // period + 1
    tokens = motif.repeat(reps)[:length].clone()
    noise_mask = torch.rand(length, generator=g) < noise_frac
    noise_vals = torch.randint(0, vocab_size, (length,), generator=g)
    tokens[noise_mask] = noise_vals[noise_mask]
    return tokens


CORPUS = make_synthetic_corpus().to(device)


def get_batch(batch_size, seq_len):
    """Random contiguous crops of the corpus -> (x, y) next-token-prediction pairs."""
    max_start = CORPUS.numel() - seq_len - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([CORPUS[s:s + seq_len] for s in starts])
    y = torch.stack([CORPUS[s + 1:s + seq_len + 1] for s in starts])
    return x, y


print(f"Synthetic corpus: {CORPUS.numel():,} tokens, vocab={VOCAB_SIZE}, "
      f"log(vocab)={math.log(VOCAB_SIZE):.3f} nats (uniform-guess loss)")

# %%
def train_run(name, n_steps=200):
    model, optimizers = build_config(name)
    losses, cum_ms = [], []
    total_ms = 0.0
    for _ in range(n_steps):
        x, y = get_batch(BATCH_SIZE, SEQ_LEN)
        x, y = x.to(device), y.to(device)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        ev_start.record()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        for opt in optimizers:
            opt.step()
        ev_end.record()
        torch.cuda.synchronize()

        total_ms += ev_start.elapsed_time(ev_end)
        losses.append(loss.item())
        cum_ms.append(total_ms)

    del model, optimizers
    gc.collect()
    torch.cuda.empty_cache()
    return dict(name=name, losses=losses, cum_ms=cum_ms)


N_STEPS = 200
runs = {cfg: train_run(cfg, n_steps=N_STEPS) for cfg in CONFIGS}
for cfg in CONFIGS:
    r = runs[cfg]
    print(f"{cfg:<8} final loss (last 10 avg): {sum(r['losses'][-10:]) / 10:.3f}  "
          f"total GPU time: {r['cum_ms'][-1] / 1000:.2f} s")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for cfg in CONFIGS:
    r = runs[cfg]
    axes[0].plot(range(1, N_STEPS + 1), r["losses"], label=cfg)
    axes[1].plot([ms / 1000 for ms in r["cum_ms"]], r["losses"], label=cfg)

axes[0].axhline(math.log(VOCAB_SIZE), color="gray", ls="--", lw=1, label="uniform guess")
axes[0].set_xlabel("step"); axes[0].set_ylabel("train loss (nats)")
axes[0].set_title("Loss vs. step"); axes[0].legend()

axes[1].axhline(math.log(VOCAB_SIZE), color="gray", ls="--", lw=1)
axes[1].set_xlabel("cumulative GPU time (s)"); axes[1].set_ylabel("train loss (nats)")
axes[1].set_title("Loss vs. wall-clock"); axes[1].legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## What you should see
#
# - **Memory:** AdamW's optimizer state should come out at roughly 8 bytes/param (fp32
#   `exp_avg` + `exp_avg_sq`), Lion at roughly 4 bytes/param (one buffer), and the Muon+AdamW
#   hybrid somewhere in between and closer to 4 — most parameters here are 2-D matrix weights
#   that get Muon's single momentum buffer, with only the embeddings/LM head/norms paying the
#   full AdamW 8-byte tax. In a production bf16-mixed-precision trainer, add ~4 bytes/param
#   more for the fp32 master-weight copy on top of whatever each optimizer already needs (see
#   the chapter's memory-tax table).
# - **Per-step wall-clock:** the isolated `.step()` timing should show Muon noticeably
#   slower than AdamW/Lion — the Newton-Schulz iteration is a handful of extra matmuls per
#   2-D weight (roughly a few times to perhaps an order of magnitude slower on the optimizer
#   step alone, depending on layer shapes and `ns_steps`). But the *full* step (forward +
#   backward + optimizer) should look much closer across the three, on the order of a modest
#   overhead at most, since attention/MLP forward-backward compute dominates total step time
#   for a model this size.
# - **Loss curves:** all three should reduce the loss well below `log(vocab_size)` on the
#   periodic synthetic task, showing the model is learning the motif. Do not expect one
#   optimizer to obviously "win" here — each was given only a roughly-sane, not carefully
#   swept, learning rate, and the chapter is explicit that Lion needs a much smaller LR (and
#   larger weight decay) than AdamW, while Muon's LR is scaled to make its update RMS shape-
#   independent. Fair comparisons need per-optimizer hyperparameter sweeps, not shared defaults.
#
# **Key takeaways**
# - Optimizer-state memory is a real, measurable systems cost, not just a formula: Lion and
#   Muon roughly halve it versus AdamW by keeping one state buffer per parameter instead of
#   two, which is exactly why they matter at the tens-to-hundreds-of-billions-of-parameters scale.
# - Muon trades some extra optimizer-step compute (the Newton-Schulz matmuls) for a
#   spectrally-uniform, half-memory update on 2-D weights; whether that trade is worth it
#   depends on how large your matrices are relative to the rest of the forward/backward cost.
# - The hybrid recipe (Muon for hidden 2-D weights, AdamW for everything else) is not
#   optional plumbing — embeddings and norms genuinely behave better under AdamW, so a real
#   Muon run always needs two live optimizer instances, as built here.
# - Next step: see the capstone notebook (single-H100 Stack-100M pretraining run) for these
#   same optimizers wired into a full training loop with a real WSD schedule and MFU logging.
