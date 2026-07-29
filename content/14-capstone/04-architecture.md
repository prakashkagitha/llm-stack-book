# 14.4 The Stack-100M Architecture: SOTA Components, Cited and Assembled

Every design decision in a modern language model is a small bet, made under a budget, against a specific failure mode. A shallow-and-wide model wastes depth. An un-normalized query-key product blows up your attention logits at high learning rate. A 50k-token vocabulary eats a quarter of a 100M-parameter model before you have written a single transformer block. A `(B, T, V)` logit tensor cast to fp32 eats more memory than every activation in the network combined. This chapter is where we stop surveying the frontier and *commit*: we assemble the exact `Stack-100M` architecture — the one every other capstone chapter trains, aligns, quantizes, and serves — from named, cited, 2024–2026 state-of-the-art components, and we write clean from-scratch PyTorch for each.

The canonical configuration is fixed in the capstone spec (`capstone/PLAN.md`, §1) and reproduced here. Our job is not to invent numbers; it is to *justify every one of them from first principles*, implement it, and reproduce the parameter count to the last norm vector. This chapter builds directly on the survey in [Modern Architecture Improvements & Design Choices](../02-transformer/10-modern-arch-improvements.html); on the block anatomy in [The Transformer Block: Norms, Residuals, MLPs & Activations](../02-transformer/06-transformer-block.html); on GQA/MLA in [Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html); and on rotary embeddings in [Positional Encodings: Sinusoidal, Learned, RoPE & ALiBi](../02-transformer/05-positional-encoding.html). Where those chapters derive a mechanism in full, we cite them and move on; here we make choices and wire them together into one runnable `stacklm.model` module — one that trains with document-aware packing (Ch. 14.2), decodes with a KV cache (Ch. 14.7, 14.10, 14.11), and can be handed to the HuggingFace / vLLM / llama.cpp ecosystem (Ch. 14.11) without a rewrite.

---

## Design philosophy: why deep-and-thin at 100M

Before a single line of code, one macro-decision governs the whole design: the *aspect ratio* of the network — how depth (`n_layers`) trades against width (`d_model`) at a fixed parameter budget.

The parameter count of the transformer body is dominated, per layer, by four attention projections ($\approx 4 d^2$ when heads are square) and the SwiGLU MLP ($3 d \cdot d_{\text{ff}}$, with $d_{\text{ff}} \approx 2.75 d$). So per-layer params scale as $\approx d^2 (4 + 3 \cdot 2.75) \approx 12.25\, d^2$, and the total body is $\approx 12.25\, d^2 \cdot L$. For a *fixed* budget $P$, you can spend it on a large $d$ with few layers, or a modest $d$ with many layers — the product $d^2 L$ is what is pinned. Doubling $d$ at fixed $P$ costs you three-quarters of your layers; doubling $L$ costs you a factor of $\sqrt 2$ off your width. The question is which of those two currencies buys more capability per unit.

At large scale (billions of parameters) they trade roughly evenly and width tends to win on hardware efficiency (wide matmuls are more FLOP-efficient and expose more parallelism). But **at small scale the evidence flips decisively toward depth.** Liu et al., *MobileLLM* (2024), ran this exact ablation for sub-billion models and found that, at fixed parameters, *deeper-and-thinner* models are consistently more accurate — a 30-ish-layer 125M model beats a shallow 12-layer one of the same size on commonsense reasoning. The intuition: capability at small scale is bottlenecked by the number of *sequential nonlinear transformations* the residual stream can undergo (the model's "reasoning depth"), not by the dimensionality of each transformation. A wide-but-shallow model can store a lot per token but can only compose a few steps of computation before emitting; a deep-but-narrow model composes many. Recent small models agree in practice: Qwen3-0.6B, the GLM small models, and SmolLM all lean deep-and-thin.

So we fix an aggressive aspect ratio: **`d_model = 512`, `n_layers = 30`** — 30 sequential blocks of a narrow 512-wide stream. The conventional shape metric is *width-per-layer*, $d_{\text{model}}/n_{\text{layers}}$: for Stack-100M that is $512/30 \approx 17$, versus $\approx 64$ for GPT-2-small ($768/12$). Stack-100M is nearly **4× thinner per layer** than a classic GPT-2 of comparable size — a deliberately extreme point on the deep-thin axis, chosen because the sub-billion evidence rewards it. We do pay for the depth: 30 layers is 30 sequential kernel launches per token in the decode path, a latency cost we accept because (a) at 100M the matrices are tiny and launch overhead dominates anyway, and (b) `torch.compile` / CUDA-graph capture (see [Kernel Fusion, torch.compile, CUDA Graphs & Compilers](../04-kernels-efficiency/09-compilers-fusion.html)) folds the launches into a single replayable graph. Depth is where a small model's quality lives.

{{fig:deep-thin-budget-trade}}

!!! note "Aside: the other small-model lever is vocabulary"

    A second small-model insight, fixed in Ch. 14.3, interacts with this one. With tied embeddings a vocabulary of $V$ tokens costs $V \times d$ parameters. At $V=50{,}257$ (GPT-2) and $d=512$ that is 25.7M parameters — on the order of **a quarter of the whole model** spent on the lookup table. We choose $V = 32768$, costing 16.78M. Vocabulary size is a first-class architectural knob at 100M in a way it simply is not at 100B. The two levers compound: deep-thin keeps the body cheap, a lean vocab keeps the embedding cheap, and the freed budget buys layers. The catch — as the memory section below shows — is that a big vocab is *also* the dominant activation cost during training, which is a second, independent reason to keep it lean.

---

## The canonical config and exact parameter accounting

Here is the whole architecture as a frozen dataclass. This is the object every capstone chapter imports as `stacklm.model.StackConfig`; the numbers are copied verbatim from `PLAN.md` §1 and must not drift.

```python
# stacklm/model.py
from dataclasses import dataclass

@dataclass
class StackConfig:
    """The canonical Stack-100M configuration (PLAN.md §1). FROZEN."""
    vocab_size: int   = 32768   # byte-level BPE we train ourselves (Ch. 14.3)
    d_model: int      = 512     # narrow: the deep-thin bet (MobileLLM, 2024)
    n_layers: int     = 30      # deep
    n_heads: int      = 8       # query heads
    n_kv_heads: int   = 2       # GQA: 2 KV heads, 4 query heads share each (Ainslie, 2023)
    head_dim: int     = 64      # 8 * 64 = 512 = d_model
    intermediate: int = 1408    # SwiGLU inner dim = 2.75 * d_model, a multiple of 64
    max_seq_len: int  = 2048    # pretrain length; extended to 8192 in mid-training (14.8)
    rope_theta: float = 10000.0 # RoPE base frequency (Su et al., 2021)
    nope_every: int   = 4       # every 4th layer is NoPE (SmolLM3, 2025); 0 disables
    norm_eps: float   = 1e-5    # RMSNorm epsilon
    qk_norm: bool     = True    # RMSNorm on Q and K before attention (stability)
    tie_embeddings: bool = True # input embedding == output projection (Press & Wolf, 2017)
    z_loss_coef: float = 1e-4   # penalty on logsumexp(logits)^2 (stability)
    logit_soft_cap: float = 0.0 # 0 disables; Gemma-2 uses 30.0 on final logits
    attn_soft_cap: float  = 0.0 # 0 disables; Gemma-2 uses 50.0 on attention logits
    loss_chunk: int   = 0       # >0 = chunked fused lm_head+CE (see "Memory" below)

    def head_groups(self) -> int:
        """How many query heads share one KV head (= 4)."""
        assert self.n_heads % self.n_kv_heads == 0
        return self.n_heads // self.n_kv_heads

    def uses_rope(self, layer_idx: int) -> bool:
        """RoPE on every layer except every `nope_every`-th (SmolLM3 interleave)."""
        return self.nope_every <= 0 or ((layer_idx + 1) % self.nope_every) != 0
```

Now the arithmetic every reader must be able to reproduce by hand. We count parameters **exactly** — including every norm vector, because "approximately" is how config drift starts.

**Tied token embedding.** One matrix of shape $V \times d = 32768 \times 512$:

$$
32768 \times 512 = 16{,}777{,}216 \approx 16.78\text{M}.
$$

Because we *tie* the embedding to the output projection (Press & Wolf, *Using the Output Embedding to Tie Word Vectors*, 2017), this matrix is counted **once**, not twice. Untied, we would pay another 16.78M — about a sixth of the model — for a table that empirically performs no better at this scale.

**Per transformer block.** Attention has four projections. With $d = 512$, $n_h = 8$ query heads, $n_{kv} = 2$ KV heads, and $d_h = 64$:

$$
\begin{aligned}
W_Q &: d \times n_h d_h = 512 \times 512 = 262{,}144,\\
W_K &: d \times n_{kv} d_h = 512 \times 128 = 65{,}536,\\
W_V &: d \times n_{kv} d_h = 512 \times 128 = 65{,}536,\\
W_O &: n_h d_h \times d = 512 \times 512 = 262{,}144.
\end{aligned}
$$

Attention subtotal: $262144 + 65536 + 65536 + 262144 = 655{,}360$. Note how GQA shrinks $W_K, W_V$ by 4×: full multi-head attention (8 KV heads) would make them $512 \times 512$ each, adding $\approx 0.39$M *per layer* — $\approx 11.8$M over 30 layers, more than four extra blocks' worth, for quality gains that are negligible at this width.

The SwiGLU MLP has three matrices (gate, up, down), each $d \times d_{\text{ff}}$ up to transpose, with $d_{\text{ff}} = 1408$:

$$
3 \times 512 \times 1408 = 2{,}162{,}688.
$$

Norms: two RMSNorm weight vectors of length $d$ ($2\times512$) for the pre-attention and pre-MLP norms, plus (with QK-norm on) two more of length $d_h$ ($2\times64$) — **1152 parameters per block**, which we carry rather than round away. Each block is therefore

$$
655{,}360 + 2{,}162{,}688 + 1{,}152 = 2{,}819{,}200.
$$

**Full model.** Thirty blocks, one final RMSNorm ($512$ params), and the tied embedding:

$$
30 \times 2{,}819{,}200 = 84{,}576{,}000, \qquad
84{,}576{,}000 + 512 = 84{,}576{,}512 \ \ (\text{the non-embedding body}),
$$

$$
16{,}777{,}216 + 84{,}576{,}512 = 101{,}353{,}728.
$$

$$
\boxed{\;101{,}353{,}728 \text{ parameters} \;=\; \approx 101.4\text{M}\;}
$$

matching the spec's "$\approx 101.4$M" exactly. Roughly 83.4% of the parameters live in the 30-layer body and 16.6% in the embedding — exactly the balance the deep-thin + lean-vocab choices were designed to produce. This is a *deterministic* integer, not an estimate: the CI test at the end of this chapter asserts it.

{{fig:param-budget-allocation}}

!!! example "Worked example: where do the FLOPs and the KV cache go?"

    Two magnitudes a practitioner should be able to estimate on the spot.

    **Training FLOPs (the 6ND rule).** A forward+backward step costs on the order of $6 N D$ FLOPs for $N$ non-embedding parameters over $D$ tokens (see [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html)). With $N \approx 84.58$M (body params) and the capstone budget $D \approx 20$B tokens:
    $$
    6 \times 84.58\times10^6 \times 20\times10^9 \approx 1.0 \times 10^{19}\ \text{FLOPs}.
    $$
    On a single A100 at, say, 40% of its ~312 bf16 TFLOP/s ($\approx 1.25\times10^{14}$ eff. FLOP/s), that is $\approx 8\times10^4$ s $\approx 22$ GPU-hours — squarely in the "\$40–\$100, 15–25 GPU-hr" flagship envelope from PLAN §0. The arithmetic *predicts the budget*: this is not a guess, it is $6ND$ divided by realized throughput.

    **KV cache per token.** GQA stores $K$ *and* $V$ for $n_{kv}=2$ heads of $d_h=64$ across $L=30$ layers, in bf16 (2 bytes):
    $$
    \underbrace{2}_{K,V} \times \underbrace{2}_{\text{KV heads}} \times 64 \times 30 \times 2\ \text{bytes} = 15{,}360\ \text{bytes} = 15\ \text{KiB/token}.
    $$
    A full 2048-token context is $2048 \times 15{,}360 = 31{,}457{,}280$ bytes $= 30$ MiB; the 8192-token mid-trained context is $120$ MiB. Under full multi-head attention (8 KV heads) those become $60$ KiB/token, $120$ MiB and $480$ MiB — the 4× GQA win, made concrete. This is the number the serving chapter (14.11) and [PagedAttention & KV-Cache Memory Management](../04-kernels-efficiency/06-paged-attention-kv.html) care about, and the `KVCache.nbytes()` method below prints it for you.

---

## The residual stream: pre-norm, RMSNorm, QK-norm, and logit hygiene

### Pre-norm blocks and RMSNorm

`Stack-100M` uses **pre-normalization**: each sublayer normalizes its input *before* the transformation and adds the result back to a clean residual, $x \leftarrow x + \text{Sublayer}(\text{Norm}(x))$. The residual stream itself is never normalized in place, so gradients flow through the identity path unimpeded — the property that makes deep (30-layer) stacks trainable without the warmup gymnastics post-norm demands. This is standard in every modern LLM and derived fully in [The Transformer Block](../02-transformer/06-transformer-block.html); we adopt it without ablation.

The normalizer is **RMSNorm** (Zhang & Sennrich, *Root Mean Square Layer Normalization*, 2019), which drops LayerNorm's mean-subtraction and bias, keeping only the learned scale:

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \odot \gamma.
$$

Cheaper (one reduction, no centering) and, empirically, no worse. Note we compute the norm in float32 even under bf16 autocast — the sum of squares is exactly where low precision bites (a stream with RMS a few units wide has squared terms that lose mantissa bits in bf16; see [Numerical Computing, Floating Point & Precision](../01-foundations/04-numerics-precision.html)).

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # gamma; init to 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()                                       # accumulate in fp32
        rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        # scale in fp32, then cast the whole product back to the input dtype
        return (self.weight * (xf * rms)).to(dtype)
```

!!! tip "Practitioner tip: you do not have to write this kernel yourself"

    Our `RMSNorm` is three fused-able ops that PyTorch will launch separately. `torch.compile` will fuse them; `torch.nn.functional.rms_norm` (PyTorch ≥ 2.4) gives you a single native op; and `apex.normalization.FusedRMSNorm` or [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel)'s `LigerRMSNorm` give you hand-written Triton kernels with a fused backward. We keep the explicit version in the book because you should be able to read it, and swap in the fused one in Ch. 14.7 where throughput matters. The Triton mechanics are in [Writing GPU Kernels with Triton](../04-kernels-efficiency/04-triton-kernels.html).

### QK-norm: the high-learning-rate stabilizer

The single most common way a small model at an aggressive learning rate diverges is an **attention-logit blowup**: the dot product $q \cdot k$ grows without bound, softmax saturates to a near one-hot distribution, the gradient into $W_Q, W_K$ collapses, and the loss spikes irrecoverably (the failure mode dissected in [Training Stability, Loss Spikes & Debugging Large Runs](../03-pretraining/11-training-stability.html)). **QK-norm** — applying RMSNorm to the query and key vectors *per head* before the dot product — bounds the geometry of that product. It normalizes $q$ and $k$ to a controlled scale so that $q \cdot k$ cannot run away regardless of how large the raw projections drift.

This trick (query-key normalization, Henry et al., *Query-Key Normalization for Transformers*, 2020, revived by many 2024–2025 open models: OLMo 2, Qwen3, SmolLM3, and the Muon-trained Kimi K2 which pairs it with QK-clip) is what lets us train `Stack-100M` with Muon at a learning rate that would otherwise diverge — the optimizer story is Ch. 14.6, and it leans on [Optimizers: SGD, Adam, Adafactor, Lion, Muon & Shampoo](../03-pretraining/09-optimizers.html). We apply the norm over the `head_dim` axis, before RoPE.

!!! warning "Ablate to learn: turn QK-norm off and watch it die"

    A genuinely instructive ablation. Train the config at the target LR (e.g. peak $\approx 3\text{e-}3$ for the Muon path) with `qk_norm=False`. On the order of a few hundred to a few thousand steps in, you will typically see the attention logits' max magnitude climb past ~50, softmax entropy collapse, and the loss NaN out. Re-enable QK-norm, same LR, and the run is stable. The lesson is causal: QK-norm is not decoration, it is what *buys* the high learning rate that makes the run cheap. Its cost is 2 tiny RMSNorm vectors per layer (128 params) — free. The `record=` hook on `Attention.forward` below harvests exactly the per-head max logit you need to watch this happen (and is the same hook Ch. 14.6's QK-clip reads).

!!! example "Worked example: how QK-norm bounds the attention logits"

    Make the mechanism quantitative. Without QK-norm, the pre-softmax logit for a query-key pair is $q\cdot k = \sum_{i=1}^{d_h} q_i k_i$. If $q$ and $k$ have per-component standard deviation $\sigma$ (independent, zero mean), each product $q_i k_i$ has variance $\sigma^4$, so the dot product over $d_h=64$ dimensions has standard deviation $\approx \sigma^2\sqrt{d_h} = 8\sigma^2$, and after the $1/\sqrt{d_h}=1/8$ scaling, $\approx \sigma^2$. During training $\sigma$ can drift upward as $W_Q, W_K$ grow; if $\sigma$ reaches, say, 6, the scaled logit std is $\approx 36$, and the *max* logit over a 2048-token context (a few std out) can exceed 100. A softmax with a gap of ~100 between the top logit and the rest is numerically a hard argmax: its gradient with respect to the losing logits is $\approx e^{-100}\approx 0$ — the attention pattern is frozen, learning stalls, and one bad step NaNs the run.

    **With QK-norm**, each $q$ and $k$ is RMS-normalized before the dot product, so $\|q\|,\|k\|\approx\sqrt{d_h}=8$ regardless of how large the raw projections grow, and by Cauchy–Schwarz $|q\cdot k|\le \|q\|\,\|k\| = 64$; after $1/\sqrt{d_h}$ scaling the logit is bounded by $\pm 8$ by construction, with a *learned* temperature reintroduced through the RMSNorm $\gamma$. The blow-up is structurally impossible. This is why QK-norm, not merely a lower learning rate, is the right fix: it removes the failure mode instead of tiptoeing around it.

{{fig:qk-norm-logit-bound}}

### z-loss and logit soft-cap

Two more cheap stabilizers guard the *output* logits. The **z-loss** (introduced in the PaLM / T5X training recipes) adds a small penalty on the log-partition function of the softmax:

$$
\mathcal{L}_z = \lambda_z \,\big(\operatorname{logsumexp}(\text{logits})\big)^2,
$$

which gently pulls $\log \sum_j e^{z_j}$ toward zero, keeping logits from drifting to large absolute values and keeping the softmax well-conditioned in bf16. We use $\lambda_z = 10^{-4}$. Note that the cross-entropy loss *already computes* $\operatorname{logsumexp}$ internally — so a good implementation reuses it rather than paying for a second pass over the $(B,T,V)$ tensor. Ours does (see `fused_ce_z_loss` below).

Optionally, a **logit soft-cap** (Gemma-2 style) squashes logits through a scaled tanh, $z \leftarrow c \cdot \tanh(z / c)$, hard-bounding them to $(-c, c)$; Gemma-2 reports $c=30$ on final logits and $c=50$ on attention logits. We leave both off by default because z-loss usually suffices and soft-capping the *attention* logits forces you off the FlashAttention fast path — but we expose them, and, critically, we apply the final cap on **both** the training and the inference path. Capping at train time and not at sample time is a silent train/serve mismatch that changes the effective temperature of every generated token; the code below computes logits once and caps once so the bug is unrepresentable. The pretraining-objective chapter, [The Pretraining Objective & Loss](../03-pretraining/03-pretraining-objective.html), treats z-loss in full.

---

## Positional information: RoPE, position ids, and NoPE on every 4th layer

### RoPE

`Stack-100M` encodes position with **Rotary Position Embeddings** (Su et al., *RoFormer*, 2021): rather than adding a position vector, RoPE *rotates* each 2-dimensional slice of the query and key by an angle proportional to the absolute position, so that the attention dot product depends only on the *relative* offset $m - n$. With base $\theta = 10000$ and per-pair frequencies $\omega_i = \theta^{-2i/d_h}$, the query at position $m$ is rotated by $m\omega_i$ in the $i$-th plane. The full derivation lives in [Positional Encodings](../02-transformer/05-positional-encoding.html); here we implement it and, crucially, keep `rope_theta` as a config knob because mid-training (Ch. 14.8) *rescales* it to extend context from 2048 to 8192 (see [Long-Context Pretraining & Context Extension](../03-pretraining/13-long-context-pretraining.html)).

One detail matters more than it looks: we never hard-code positions to `arange(T)`. The cos/sin tables are **indexed by an explicit `position_ids` tensor**. That single indirection is what makes three later chapters work — document packing needs positions to *reset* at every document boundary (Ch. 14.2), incremental decoding needs position $t$ for a single-token step (Ch. 14.11), and long-context extension needs positions beyond the training range (Ch. 14.8).

```python
def build_rope_cache(head_dim: int, max_seq: int, theta: float,
                     device=None, dtype=torch.float32):
    """Precompute cos/sin tables of shape (max_seq, head_dim)."""
    # frequencies for each PAIR of dims: theta^(-2i/head_dim), i=0..head_dim/2-1
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq, device=device).float()          # positions
    freqs = torch.outer(t, inv_freq)                          # (max_seq, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)                   # (max_seq, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos, sin):
    """q, k: (B, n_heads, T, head_dim). cos/sin: (T, head_dim) for the shared-position
    case, or (B, T, head_dim) when positions differ per batch row (packed documents)."""
    if cos.dim() == 2:
        cos, sin = cos[None, None, :, :], sin[None, None, :, :]   # (1,1,T,d_h)
    else:
        cos, sin = cos[:, None, :, :], sin[:, None, :, :]         # (B,1,T,d_h)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot

def ntk_rescaled_base(base: float, head_dim: int, old_len: int, new_len: int) -> float:
    """NTK-aware RoPE base rescaling used in Ch. 14.8: theta' = theta * s^(d/(d-2))."""
    s = new_len / old_len
    return base * (s ** (head_dim / (head_dim - 2)))
```

This is the "GPT-NeoX / HuggingFace" RoPE layout, where the two halves of the head vector form the rotation pairs (index $i$ pairs with $i + d_h/2$); `emb = cat(freqs, freqs)` and `rotate_half` are matched to that layout, so the code is self-consistent. (The other common layout — the interleaved even/odd pairing of the original RoFormer code — is mathematically equivalent up to a fixed permutation of the head dimension, but weights are **not** interchangeable between the two, which is a classic source of "my converted checkpoint outputs garbage.") Applied to $q$ of shape `(B, 8, T, 64)` and $k$ of shape `(B, 2, T, 64)`, the tables broadcast across the batch and head axes identically — RoPE is applied to the *narrow* GQA key tensor **before** the KV heads are expanded, which is both correct (rotation is per-position, head-count-agnostic) and cheap.

### NoPE on every 4th layer

Here is a genuinely modern choice. We do **not** apply RoPE on every layer. Following **SmolLM3** (HuggingFace, 2025) and the analysis of Kazemnejad et al. (*The Impact of Positional Encoding on Length Generalization in Transformers*, "NoPE", NeurIPS 2023), **every 4th layer uses no positional encoding at all** — "NoPE."

The reasoning: a decoder-only model with a causal mask can *infer* absolute position from the mask alone (the number of tokens a position can attend to is itself a positional signal), so it does not strictly need an injected encoding on every layer. Layers that are freed from RoPE are not tied to the specific rotation frequencies seen during training, and empirically the mixture generalizes to *longer* sequences than training — precisely the property we want for the 2048→8192 context extension in mid-training. Keeping RoPE on the majority of layers preserves the strong local-position inductive bias that makes short-context modeling sample-efficient; interleaving NoPE layers buys length robustness. SmolLM3 reports this hybrid as an ingredient of its long-context behavior.

Concretely, with `nope_every = 4`, layer index $\ell$ (0-based) uses NoPE iff $(\ell + 1) \bmod 4 = 0$ — layers 3, 7, 11, …, 27, i.e. **7 of the 30 layers**; the other 23 keep RoPE. Setting `nope_every = 0` disables the interleave entirely, which — as the ecosystem section explains — is the switch that makes the checkpoint loadable by stock `transformers`/vLLM without custom code.

!!! note "Aside: NoPE is not free lunch, it is a mixture"

    A pure-NoPE model tends to underperform on *short* context — the causal-mask position signal is weak and diffuse compared to RoPE's sharp relative encoding. That is why we interleave rather than remove. The design point is: RoPE for local precision on most layers, a minority of NoPE layers for length extrapolation. The 1-in-4 ratio is SmolLM3's; treat it as a tuned constant, not a law.

{{fig:rope-nope-layer-stack}}

---

## Attention: grouped-query attention, masking, and the KV cache

Now we assemble the attention module: **GQA with 2 KV heads** (Ainslie et al., *GQA*, 2023), **QK-norm** on the per-head queries and keys, RoPE (or NoPE) applied conditionally, an **explicit attention mask** so document packing and incremental decoding both work, an optional **KV cache**, and PyTorch's fused scaled-dot-product attention (which dispatches to a FlashAttention kernel when available — see [FlashAttention I](../04-kernels-efficiency/02-flash-attention-1.html)).

GQA is the middle ground between full multi-head attention (one KV head per query head — maximal quality, maximal KV cache) and multi-query attention (one KV head total — minimal cache, some quality loss). With 8 query heads sharing 2 KV heads (a 4:1 group), we cut the KV cache 4× versus MHA while retaining nearly all the quality; the mechanism and quality tradeoff are dissected in [Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html).

{{fig:gqa-head-sharing-kv-cache}}

### The KV cache

Autoregressive decoding recomputes nothing it has already computed: after prefilling a prompt of $T_0$ tokens, each new token needs only its *own* $q$ dotted against the $K, V$ of all previous positions. Cache them and a decode step costs $O(1)$ layers of work instead of $O(T)$. Without a cache, generating $n$ tokens from a 30-layer model is $O(n \cdot T^2)$ — which is why the naive `if targets is None: logits = lm_head(x[:, -1:])` pattern found in most tutorials is a performance trap: it saves the `lm_head` matmul and *nothing else*.

```python
class KVCache:
    """Pre-allocated K/V store, one slab per layer: (n_layers, B, n_kv, max_seq, d_h).

    Pre-allocating (rather than torch.cat-ing each step) avoids a realloc + copy of the
    whole cache on every token -- the single biggest cost in a naive decode loop.
    Production servers go one step further and page this memory: see vLLM's
    PagedAttention (Ch. 4.6 / 7.3), which keeps fixed-size blocks in a global pool so
    many sequences of different lengths share one allocation without fragmentation."""

    def __init__(self, cfg, batch_size: int, max_seq: int, device, dtype=torch.bfloat16):
        shape = (cfg.n_layers, batch_size, cfg.n_kv_heads, max_seq, cfg.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.max_seq = max_seq

    def update(self, layer_idx: int, k, v, start_pos: int):
        """Write this step's K/V at [start_pos : start_pos+T]; return the full prefix."""
        T = k.shape[2]
        assert start_pos + T <= self.max_seq, "KV cache overflow -- grow max_seq"
        self.k[layer_idx, :, :, start_pos:start_pos + T] = k.to(self.k.dtype)
        self.v[layer_idx, :, :, start_pos:start_pos + T] = v.to(self.v.dtype)
        return (self.k[layer_idx, :, :, :start_pos + T],
                self.v[layer_idx, :, :, :start_pos + T])

    def nbytes(self) -> int:
        return (self.k.numel() + self.v.numel()) * self.k.element_size()
```

For `StackConfig()`, `KVCache(cfg, batch_size=1, max_seq=2048, dtype=torch.bfloat16).nbytes()` prints `31457280` — the 30 MiB we computed by hand above. Run it; matching a hand estimate to a measured integer is the habit this whole chapter is trying to build.

!!! warning "Common pitfall: SDPA's `is_causal=True` is TOP-LEFT aligned"

    This is the bug that silently ruins most first KV-cache retrofits. When you call `F.scaled_dot_product_attention(q, k, v, is_causal=True)` with `q_len == kv_len`, you get the lower-triangular mask you expect. But on a decode step `q_len = 1` and `kv_len = N`, and PyTorch aligns the causal mask to the **top-left** of the $1 \times N$ score matrix — so query 0 may attend only to key 0, and your model reads position 0 of the prompt and nothing else. Output is fluent-looking garbage and the loss curve never tells you, because it only happens at generation time.

    Two correct fixes: (1) build the mask from *positions*, `q_pos[:, None] >= kv_pos[None, :]`, which is what our `_build_mask` does and which is unambiguous by construction; or (2) use `torch.nn.attention.bias.causal_lower_right(q_len, kv_len)` and pass it as `attn_mask=`, which asks PyTorch for the bottom-right-aligned causal bias and still hits a fused kernel. Never pass `is_causal=True` with a rectangular score matrix.

### The attention module

```python
class Attention(nn.Module):
    def __init__(self, cfg: StackConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.groups = cfg.head_groups()           # query heads per KV head (=4)
        self.use_rope = cfg.uses_rope(layer_idx)  # NoPE on every 4th layer (SmolLM3)

        # projections; no biases (modern default)
        self.wq = nn.Linear(cfg.d_model, self.n_heads    * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, cfg.d_model, bias=False)

        # QK-norm: RMSNorm over head_dim, applied to q and k per head
        self.q_norm = RMSNorm(self.head_dim, cfg.norm_eps) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, cfg.norm_eps) if cfg.qk_norm else nn.Identity()
        self.attn_soft_cap = cfg.attn_soft_cap

    def forward(self, x, cos, sin, attn_mask=None, kv_cache=None, start_pos=0, record=None):
        """attn_mask: bool (B|1, 1, T, kv_len), True = attend. None => plain causal
        (valid ONLY when T == kv_len). kv_cache/start_pos drive incremental decode."""
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # QK-norm BEFORE RoPE: bound the geometry of q·k (stability at high LR)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # conditional positional encoding: RoPE on most layers, NoPE on every 4th
        if self.use_rope:
            q, k = apply_rope(q, k, cos, sin)

        # cache the NARROW (n_kv-head), already-rotated K/V, then read the whole prefix
        if kv_cache is not None:
            k, v = kv_cache.update(self.layer_idx, k, v, start_pos)

        # GQA: expand the 2 KV heads to 8 by repeating each group `groups` times.
        # (PyTorch >= 2.5 can do this inside the kernel with enable_gqa=True -- see note.)
        k = k.repeat_interleave(self.groups, dim=1)   # (B, n_heads, kv_len, d_h)
        v = v.repeat_interleave(self.groups, dim=1)
        scale = 1.0 / math.sqrt(self.head_dim)

        # optional instrumentation: per-head max attention logit, for QK-clip (Ch. 14.6)
        if record is not None:
            with torch.no_grad():
                a = (q.float() @ k.float().transpose(-2, -1)) * scale
                if attn_mask is not None:
                    a = a.masked_fill(~attn_mask, float("-inf"))
                record[self.layer_idx] = a.amax(dim=(0, 2, 3))     # (n_heads,)

        if self.attn_soft_cap > 0:
            # Gemma-2 style: manual attention so we can tanh-cap the logits.
            # NOTE: this leaves the fused FlashAttention path -- O(T^2) memory. Use
            # flex_attention's score_mod (below) if you want capping AND a fused kernel.
            a = (q @ k.transpose(-2, -1)) * scale
            c = self.attn_soft_cap
            a = c * torch.tanh(a / c)
            if attn_mask is None:
                causal = torch.ones(T, k.shape[2], dtype=torch.bool, device=x.device).tril()
                a = a.masked_fill(~causal, float("-inf"))
            else:
                a = a.masked_fill(~attn_mask, float("-inf"))
            out = a.softmax(dim=-1).to(v.dtype) @ v
        elif attn_mask is None:
            # fast path: fused SDPA (FlashAttention kernel when available), square causal
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, n_heads*d_h)
        return self.wo(out)
```

Four implementation notes worth internalizing.

**QK-norm is applied before RoPE**, not after: we normalize the *content* geometry of $q$ and $k$, then inject position. RoPE is norm-preserving (a rotation leaves $\|q\|$ unchanged), so the choice of order does not change the *magnitude* bound — but it does change how the learned per-dimension scale $\gamma$ interacts with the rotation. Normalizing first keeps $\gamma$ acting in the unrotated content frame, which is the standard order (OLMo 2, Qwen3, SmolLM3).

**We cache the narrow, post-RoPE K/V.** Caching *before* rotation would force a re-rotation on every read; caching *after* head expansion would quadruple the cache and throw away the entire GQA win. Two KV heads in, two KV heads cached.

**`repeat_interleave` is pedagogy, not production.** A real kernel never materializes the expansion: FlashAttention's GQA path and vLLM's PagedAttention read the shared KV head directly. Since PyTorch 2.5 you can get that behavior from SDPA itself — `F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)` accepts `k, v` with `n_kv_heads` and does the broadcast inside the kernel. It is numerically identical (verify it: the max absolute difference is exactly 0) and strictly cheaper. We keep the explicit expansion in the book because it makes the 4:1 sharing visible, and flip it on in Ch. 14.7.

**The default scale is explicit.** SDPA's default is $1/\sqrt{d_h}$; the manual soft-cap path reproduces the same scale so the two paths are numerically comparable when `attn_soft_cap=0`.

### Document-aware masking: making packing correct

PLAN §2 packs many short documents into every 2048-token training sequence, and requires that **no token attends across a document boundary** and that **position ids reset** at each boundary. Both are model-side responsibilities, and both are cheap once positions and masks are explicit. Ch. 14.2's dataset already emits exactly the two tensors we need per row: `position_ids` (0,1,2,… restarting at each document) and `seq_ids` (0,0,0,1,1,2,2,… the per-token document index).

The mask is then the conjunction of "causal" and "same document":

```python
    def _build_mask(self, seq_ids, T, kv_len, start_pos, device):
        """Bool mask (B|1, 1, T, kv_len); True = attend. None = plain causal fast path."""
        if seq_ids is None and kv_len == T and start_pos == 0:
            return None                                  # let SDPA use is_causal=True
        q_pos = torch.arange(start_pos, start_pos + T, device=device)
        kv_pos = torch.arange(kv_len, device=device)
        m = (q_pos[:, None] >= kv_pos[None, :])[None, None]          # (1,1,T,kv_len)
        if seq_ids is not None:                                       # no cross-document
            same = seq_ids[:, -T:, None] == seq_ids[:, None, :kv_len]  # (B,T,kv_len)
            m = m & same[:, None]
        return m
```

This is correct and, for a 100M model at $T=2048$, perfectly affordable: the mask is $2048^2$ bools per batch row, and SDPA's memory-efficient backend consumes it without materializing scores. But it does force you off the FlashAttention fast path, and at 8192 tokens (mid-training) a dense $8192^2$ mask per row is 64 MB of bools — the wrong shape of solution.

**The 2026 answer is block-sparse masking.** PyTorch's `torch.nn.attention.flex_attention` compiles a *predicate* over `(b, h, q_idx, kv_idx)` into a FlashAttention-class kernel that skips entire blocks where the predicate is false everywhere. A packed sequence of documents is exactly block-sparse, so the savings are real, not cosmetic:

```python
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

def document_causal_block_mask(seq_ids, device):
    """Block-sparse 'causal AND same-document' mask (PyTorch >= 2.5).
    seq_ids: (B, T) int tensor of per-token document indices."""
    B, T = seq_ids.shape

    def mask_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (seq_ids[b, q_idx] == seq_ids[b, kv_idx])

    # H=None broadcasts the same mask across heads
    return create_block_mask(mask_mod, B=B, H=None, Q_LEN=T, KV_LEN=T, device=device)

def softcap_score_mod(cap: float):
    """Gemma-2 attention soft-cap WITHOUT leaving the fused kernel."""
    def score_mod(score, b, h, q_idx, kv_idx):
        return cap * torch.tanh(score / cap)
    return score_mod

# usage (compile it -- uncompiled flex_attention materializes the score matrix):
flex = torch.compile(flex_attention)
out = flex(q, k, v, block_mask=block_mask,
           score_mod=softcap_score_mod(50.0), enable_gqa=True)   # k,v have n_kv heads
```

Two things to notice. First, `flex_attention` takes `enable_gqa=True`, so again no `repeat_interleave`. Second, `score_mod` is the mechanism that lets you keep **both** attention soft-capping and a fused kernel — the one combination the manual path above cannot give you. Verify the block mask against the dense reference once (they agree to fp32 round-off), then never look back.

The alternative, and the one the frontier training stacks actually use, is **variable-length packing**: instead of a mask, hand the kernel a `cu_seqlens` array of cumulative document lengths and let it treat the packed buffer as a ragged batch. That is `flash_attn_varlen_func` from [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention), and it is what Megatron-LM, TRL's packing path, and most production pretraining loops call. It is strictly the fastest option; we teach the mask because it composes with everything else in this chapter and needs no extra dependency.

---

## The MLP: SwiGLU

The feed-forward sublayer is **SwiGLU** (Shazeer, *GLU Variants Improve Transformer*, 2020): a gated linear unit whose gate is the SiLU (swish) activation. Where a vanilla FFN is $W_2\,\sigma(W_1 x)$, SwiGLU is

$$
\text{SwiGLU}(x) = W_{\text{down}}\big(\text{SiLU}(W_{\text{gate}}\, x) \odot (W_{\text{up}}\, x)\big),\qquad \text{SiLU}(z) = z\,\sigma(z).
$$

Three matrices instead of two. To keep the parameter count comparable to a $4d$ vanilla FFN, GLU-family FFNs use a smaller inner dimension: the common $\tfrac{8}{3}d \approx 2.67d$ heuristic, which we round up to $d_{\text{ff}} = 1408 = 2.75 \times 512$, a multiple of 64 for good tensor-core tiling. The gate lets the network modulate each hidden unit multiplicatively, which consistently improves quality per parameter — it is now universal in Llama, Qwen, DeepSeek, and Mistral.

```python
class SwiGLU(nn.Module):
    def __init__(self, cfg: StackConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.intermediate, bias=False)
        self.up   = nn.Linear(cfg.d_model, cfg.intermediate, bias=False)
        self.down = nn.Linear(cfg.intermediate, cfg.d_model, bias=False)

    def forward(self, x):
        # SiLU gate * linear up, then project down
        return self.down(F.silu(self.gate(x)) * self.up(x))
```

The SwiGLU forward saves four `(B, T, 1408)` intermediates for backward (gate, up, silu(gate), and their product). Liger-Kernel's `LigerSwiGLUMLP` fuses them into one Triton kernel that recomputes rather than stores — a straight ~2×activation-memory win on the MLP with no math change. Worth switching on in Ch. 14.7 once the from-scratch version is understood.

---

## Assembling the block, the loss, and the full model

A `Block` is pre-norm attention plus pre-norm SwiGLU, each wrapped in a residual add. The full model is an embedding, 30 blocks, a final norm, and a tied output projection. Here is the data flow:

```text
  tokens ─► Embedding (32768×512, tied) ─► x  (B,T,512)
     position_ids ─► rope_cos/sin gather ─► (B,T,64)
     seq_ids ──────► _build_mask ─────────► (B,1,T,kv_len) bool
                                            │
        ┌───────────────────────────────────┤  ×30 blocks
        │   x ──► RMSNorm ──► Attention ──►(+)      Attention:
        │   │                              │          Q:512→512  K,V:512→128 (GQA 2 KV)
        │   └──────────────residual────────┘          QK-norm ► RoPE* ► [KV cache] ► SDPA ► O
        │   x ──► RMSNorm ──► SwiGLU ────►(+)      SwiGLU: gate,up:512→1408; down:1408→512
        │   │                              │       * layer (ℓ+1)%4==0 ► NoPE (no RoPE)
        └──────────────residual────────────┘
                                            │
                          x ──► RMSNorm ──► lm_head (tied, 512→32768) ──► soft-cap ──► logits
                                            └► CE + z-loss (fp32; chunked if loss_chunk>0)
```

```python
class Block(nn.Module):
    def __init__(self, cfg: StackConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg, layer_idx)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, attn_mask=None, kv_cache=None, start_pos=0, record=None):
        x = x + self.attn(self.attn_norm(x), cos, sin, attn_mask=attn_mask,
                          kv_cache=kv_cache, start_pos=start_pos, record=record)  # pre-norm
        x = x + self.mlp(self.mlp_norm(x))                                        # pre-norm
        return x
```

### The loss: cross-entropy, z-loss, and why the logits tensor is the memory bomb

The output head is where a *small* model with a *large* vocabulary springs a nasty surprise. At a micro-batch of $16 \times 2048 = 32{,}768$ tokens, the logits tensor is $32768 \times 32768 \approx 1.07\times10^9$ elements. In bf16 that is 2.1 GB; `logits.float()` makes it 4.3 GB; `F.cross_entropy` internally saves a `log_softmax` of the same shape, another 4.3 GB; a separate `torch.logsumexp` pass for the z-loss adds more. **The logit tensor alone can exceed every other activation in a 101M-parameter model combined.** This is a direct, unavoidable consequence of the small-model/large-vocab regime we chose on purpose.

The fix is to never materialize the whole thing: chunk over tokens, fuse `lm_head + log_softmax + gather` inside each chunk, and *recompute* the chunk in the backward pass instead of storing it. Note also that CE and z-loss share the same $\operatorname{logsumexp}$, so we compute it once.

```python
from torch.utils.checkpoint import checkpoint

def _chunk_ce(h, w, t, cap: float):
    """One chunk: hidden (n, d) @ w.T -> logits (n, V) -> (sum CE, sum lse^2, n_valid).
    Everything here is recomputed in backward, so the (n, V) logits never persist."""
    logits = F.linear(h, w).float()                     # fp32 for stability
    if cap > 0:
        logits = cap * torch.tanh(logits / cap)         # SAME cap as inference
    lse = torch.logsumexp(logits, dim=-1)               # (n,) -- reused by BOTH losses
    valid = (t != -100)
    tgt = t.clamp_min(0).unsqueeze(-1)
    ce = lse - logits.gather(-1, tgt).squeeze(-1)       # CE = logsumexp - logit[target]
    return (ce * valid).sum(), (lse.pow(2) * valid).sum(), valid.sum()

def fused_ce_z_loss(hidden, weight, targets, z_coef: float,
                    chunk: int = 8192, soft_cap: float = 0.0):
    """Memory-lean lm_head + cross-entropy + z-loss. Peak logit memory is
    (chunk x vocab) instead of (B*T x vocab): at chunk=8192, V=32768 that is
    1.07 GB fp32 rather than 4.3 GB -- and it does not grow with batch size."""
    h = hidden.reshape(-1, hidden.shape[-1])
    t = targets.reshape(-1)
    ce = h.new_zeros((), dtype=torch.float32)
    z  = h.new_zeros((), dtype=torch.float32)
    n  = torch.zeros((), dtype=torch.long, device=h.device)
    for i in range(0, h.shape[0], chunk):
        a, b, c = checkpoint(_chunk_ce, h[i:i + chunk], weight, t[i:i + chunk],
                             soft_cap, use_reentrant=False)
        ce, z, n = ce + a, z + b, n + c
    n = n.clamp_min(1)
    return ce / n, z_coef * (z / n)
```

The `checkpoint` call is load-bearing: a plain Python loop over chunks saves *nothing*, because autograd retains every chunk's logits for the backward pass. Checkpointing frees them at the end of the forward and recomputes each chunk during backward, so peak logit memory is one chunk. The cost is one extra `lm_head` matmul — a few percent of step time.

!!! tip "Practitioner tip: the library versions of this are better than ours"

    Two production kernels do the same thing without the recompute overhead, by fusing the linear + cross-entropy into a single Triton kernel that accumulates gradients on the fly and **never writes a logit tensor at all**:

    - [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) — `LigerFusedLinearCrossEntropy`, a drop-in for the `lm_head + F.cross_entropy` pair (it also has fused RMSNorm, SwiGLU and RoPE kernels, and a monkey-patch entry point for HF models).
    - [apple/ml-cross-entropy](https://github.com/apple/ml-cross-entropy) — Cut Cross-Entropy (Wijmans et al., *Cut Your Losses in Large-Vocabulary Language Models*, 2024), `linear_cross_entropy(hidden, classifier, targets)`, which computes the loss in on-chip memory and reports reducing the loss-computation memory of large-vocab models to a small fraction of the naive path.

    Use ours to understand the mechanism; use theirs in the real run. Both are more valuable to a 100M model with a 32k vocab than to a 70B model with the same vocab — the smaller the trunk, the more the logit tensor dominates.

### The full model

```python
class Stack100M(nn.Module):
    def __init__(self, cfg: StackConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight   # TIED (Press & Wolf, 2017)

        # RoPE cos/sin cache (buffers; rebuilt when mid-training extends the context)
        cos, sin = build_rope_cache(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # scale residual-projection inits by 1/sqrt(2*n_layers) (GPT-2 trick)
        scale = (2 * cfg.n_layers) ** -0.5
        for blk in self.blocks:
            nn.init.normal_(blk.attn.wo.weight, mean=0.0, std=0.02 * scale)
            nn.init.normal_(blk.mlp.down.weight, mean=0.0, std=0.02 * scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    def rebuild_rope(self, max_seq_len: int, rope_theta: float, device=None):
        """Ch. 14.8 calls this to extend 2048 -> 8192 with an NTK-rescaled base."""
        cos, sin = build_rope_cache(self.cfg.head_dim, max_seq_len, rope_theta, device=device)
        self.rope_cos, self.rope_sin = cos, sin
        self.cfg.max_seq_len, self.cfg.rope_theta = max_seq_len, rope_theta

    def _build_mask(self, seq_ids, T, kv_len, start_pos, device):
        if seq_ids is None and kv_len == T and start_pos == 0:
            return None
        q_pos = torch.arange(start_pos, start_pos + T, device=device)
        kv_pos = torch.arange(kv_len, device=device)
        m = (q_pos[:, None] >= kv_pos[None, :])[None, None]
        if seq_ids is not None:
            same = seq_ids[:, -T:, None] == seq_ids[:, None, :kv_len]
            m = m & same[:, None]
        return m

    def _cap(self, logits):
        c = self.cfg.logit_soft_cap
        return c * torch.tanh(logits / c) if c > 0 else logits

    def forward(self, idx, targets=None, position_ids=None, seq_ids=None,
                kv_cache=None, start_pos=0, logits_to_keep=0, record=None):
        """idx: (B, T) tokens. position_ids/seq_ids: (B, T) from the Ch. 14.2 packer.
        kv_cache + start_pos: incremental decode. logits_to_keep: compute logits for
        only the last k positions (HF's `logits_to_keep` convention)."""
        B, T = idx.shape
        dev = idx.device
        if position_ids is None:   # contiguous positions starting at start_pos
            position_ids = torch.arange(start_pos, start_pos + T, device=dev).expand(B, T)
        cos = self.rope_cos.to(dev)[position_ids]    # (B, T, head_dim)
        sin = self.rope_sin.to(dev)[position_ids]

        kv_len = start_pos + T if kv_cache is not None else T
        attn_mask = self._build_mask(seq_ids, T, kv_len, start_pos, dev)

        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x, cos, sin, attn_mask=attn_mask, kv_cache=kv_cache,
                    start_pos=start_pos, record=record)
        x = self.final_norm(x)
        if logits_to_keep:
            x = x[:, -logits_to_keep:, :]            # skip the lm_head on dead positions

        if targets is not None and self.cfg.loss_chunk > 0:
            ce, zl = fused_ce_z_loss(x, self.lm_head.weight, targets,
                                     self.cfg.z_loss_coef, chunk=self.cfg.loss_chunk,
                                     soft_cap=self.cfg.logit_soft_cap)
            return None, ce + zl                     # logits deliberately not materialized

        logits = self._cap(self.lm_head(x))          # cap on BOTH train and inference paths
        loss = None
        if targets is not None:
            lf = logits.float()                      # fp32 for stability
            ce = F.cross_entropy(lf.view(-1, lf.shape[-1]),
                                 targets.reshape(-1), ignore_index=-100)
            logz = torch.logsumexp(lf, dim=-1)       # (B, T)
            loss = ce + self.cfg.z_loss_coef * (logz ** 2).mean()
        return logits, loss

    @torch.no_grad()
    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and self.cfg.tie_embeddings:
            n -= self.tok_emb.weight.numel()   # tied head shares the same tensor
        return n
```

A few points that make this *correct* and not just plausible. The residual-projection initializations (`wo`, `down`) are scaled by $1/\sqrt{2L}$ — the GPT-2 trick — so that the variance added into the residual stream does not grow with depth. Each layer contributes two residual writes (attention, MLP), so after $L$ layers the stream has accumulated $2L$ writes; scaling each output projection's init std by $1/\sqrt{2L}$ keeps the accumulated variance $O(1)$ instead of $O(L)$. With 30 layers this matters, and skipping it is a common cause of early-training instability in deep-thin models. Because `lm_head.weight` is *the same tensor* as `tok_emb.weight`, `self.apply(self._init_weights)` initializing it twice is harmless — both draw from the same $\mathcal N(0, 0.02^2)$, and the tie is a shared reference, not a copy, so the two stay identical through training. And because the head is tied, `num_params(non_embedding=True)` subtracts the shared tensor exactly once to recover the 84,576,512 body count used in the $6ND$ estimate.

### Generation: one `generate()` the whole capstone imports

Ch. 14.7 samples during training, Ch. 14.9 samples for DPO and GRPO rollouts, Ch. 14.10 runs a multi-step ReAct loop, and Ch. 14.11 measures decode latency on a laptop. All four import *this*:

```python
def sample_next(logits, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0):
    """(B, V) logits -> (B, 1) sampled ids. temperature<=0 => greedy.
    Sampling theory (and why top-p beats top-k) is in Ch. 7.9."""
    logits = logits.float()
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k > 0:
        kth = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = logits.softmax(dim=-1)
    if top_p < 1.0:
        sp, si = probs.sort(dim=-1, descending=True)
        keep = (sp.cumsum(-1) - sp) < top_p        # keep the smallest nucleus with mass >= p
        sp = sp * keep
        sp = sp / sp.sum(-1, keepdim=True)
        return si.gather(-1, torch.multinomial(sp, num_samples=1))
    return torch.multinomial(probs, num_samples=1)
```

```python
class Stack100M(nn.Module):      # ... continued: this is a method of the class above

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int = 64, temperature: float = 0.8,
                 top_p: float = 0.95, top_k: int = 0, eos_id=None, use_cache: bool = True):
        """Prefill once, then one cached step per token. O(T) instead of O(T^2)."""
        self.eval()
        B, T0 = idx.shape
        total = T0 + max_new_tokens
        assert total <= self.cfg.max_seq_len, "extend max_seq_len or rebuild_rope() first"
        p = next(self.parameters())
        cache = KVCache(self.cfg, B, total, p.device, p.dtype) if use_cache else None

        logits, _ = self.forward(idx, kv_cache=cache, start_pos=0, logits_to_keep=1)
        out = idx
        done = torch.zeros(B, 1, dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            nxt = sample_next(logits[:, -1, :], temperature, top_p, top_k)
            if eos_id is not None:
                nxt = torch.where(done, torch.full_like(nxt, eos_id), nxt)  # freeze finished
                done = done | (nxt == eos_id)
            out = torch.cat([out, nxt], dim=1)
            if eos_id is not None and bool(done.all()):
                break
            if cache is None:                                # reference path (slow, O(T^2))
                logits, _ = self.forward(out[:, -self.cfg.max_seq_len:], logits_to_keep=1)
            else:                                            # cached path: ONE new token
                logits, _ = self.forward(nxt, kv_cache=cache,
                                         start_pos=out.shape[1] - 1, logits_to_keep=1)
        return out
```

The `use_cache=False` branch exists for one reason: it is the oracle. Greedy generation with and without the cache must produce **identical** token ids; if they diverge you have a cache, mask, or position bug. That two-line test is in the CI block below, and it is the single highest-value test in this chapter.

Everything past this is a serving concern, not a modelling one: continuous batching, paged KV blocks, prefix reuse, CUDA-graph capture of the decode step. Those live in [The Anatomy of LLM Inference](../07-inference-serving/01-anatomy-inference.html), [PagedAttention & KV-Cache Memory Management](../04-kernels-efficiency/06-paged-attention-kv.html), and [vLLM: Architecture, PagedAttention & Internals](../07-inference-serving/03-vllm-internals.html). Our `generate()` is the correct, readable baseline that those systems optimize.

### Sanity check and CI assertions

```python
# stacklm/tests/test_model.py -- these run in CI on CPU in a few seconds.
if __name__ == "__main__":
    cfg = StackConfig()
    m = Stack100M(cfg)

    # 1. exact parameter accounting -- deterministic integers, no drift allowed
    print(f"total params      : {m.num_params():,}")        # 101,353,728
    print(f"non-embedding body: {m.num_params(True):,}")    #  84,576,512
    assert m.num_params() == 101_353_728
    assert m.num_params(True) == 84_576_512

    # 2. KV cache footprint at the pretrain context (bf16, batch 1)
    kv = KVCache(cfg, batch_size=1, max_seq=cfg.max_seq_len, device="cpu",
                 dtype=torch.bfloat16)
    print(f"KV cache @2048    : {kv.nbytes():,} bytes")     # 31,457,280  (30 MiB)
    assert kv.nbytes() == 31_457_280

    # 3. NoPE interleave: 7 of 30 layers, at indices 3,7,...,27
    nope = [i for i in range(cfg.n_layers) if not cfg.uses_rope(i)]
    assert nope == [3, 7, 11, 15, 19, 23, 27] and len(nope) == 7

    # --- behavioural tests on a tiny config (seconds on CPU) ---
    tiny = StackConfig(vocab_size=97, d_model=64, n_layers=4, n_heads=4, n_kv_heads=2,
                       head_dim=16, intermediate=128, max_seq_len=64)
    net = Stack100M(tiny).eval()
    idx = torch.randint(0, 97, (2, 10))

    # 4. cached incremental decode == full recompute (catches the top-left mask bug)
    full, _ = net(idx)
    cache = KVCache(tiny, 2, 20, "cpu", torch.float32)
    net(idx[:, :6], kv_cache=cache, start_pos=0)
    inc = torch.stack([net(idx[:, t:t + 1], kv_cache=cache, start_pos=t)[0][:, -1]
                       for t in range(6, 10)], dim=1)
    assert torch.allclose(full[:, 6:], inc, atol=1e-5)

    # 5. document masking: packed docs == the same docs run separately
    a, b = torch.randint(0, 97, (1, 5)), torch.randint(0, 97, (1, 7))
    joint, _ = net(torch.cat([a, b], 1),
                   position_ids=torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6]]),
                   seq_ids=torch.tensor([[0] * 5 + [1] * 7]))
    assert torch.allclose(joint[:, :5], net(a)[0], atol=1e-5)
    assert torch.allclose(joint[:, 5:], net(b)[0], atol=1e-5)

    # 6. chunked fused loss == the reference loss (value and gradients)
    tg = torch.randint(0, 97, (2, 10)); tg[0, 3] = -100
    ref = Stack100M(tiny); ref.zero_grad(); ref(idx, targets=tg)[1].backward()
    g_ref = ref.tok_emb.weight.grad.clone()
    ref.cfg.loss_chunk = 4
    ref.zero_grad(); ref(idx, targets=tg)[1].backward()
    assert torch.allclose(g_ref, ref.tok_emb.weight.grad, atol=1e-4)

    # 7. greedy generation is cache-invariant (the highest-value test here)
    torch.manual_seed(0); o1 = net.generate(idx[:, :4], 6, temperature=0.0)
    torch.manual_seed(0); o2 = net.generate(idx[:, :4], 6, temperature=0.0, use_cache=False)
    assert torch.equal(o1, o2)
    print("all model invariants OK")
```

---

## Budgeting the run: memory, micro-batch, and what actually fits

$6ND$ tells you the *compute*; it says nothing about whether your step fits in the card. Sizing the micro-batch is a first-principles exercise every CS336-grade practitioner should be able to do on a napkin, and for Stack-100M it has a surprising answer: **the model is nearly free and the logits are not.**

Start with the persistent state. Ch. 14.6 uses the standard hybrid: **Muon** on the 2D hidden matrices, **AdamW** on embeddings, norms, and 1D params. That partition is exactly our parameter accounting:

| Group | Params | Optimizer | States | fp32 bytes |
|---|---|---|---|---|
| 2D body matrices (30 × attn+MLP) | 84,541,440 | Muon | 1 momentum buffer | 338 MB |
| Embedding + all norms | 16,812,288 | AdamW | m, v | 134 MB |
| Master weights (fp32) | 101,353,728 | — | — | 405 MB |
| Gradients (fp32) | 101,353,728 | — | — | 405 MB |
| **Persistent total** | | | | **≈ 1.28 GB** |

That is it. A 101M-parameter model's entire trainable state is **under 1.3 GB** — 1.6% of an A100-80GB. Everything else in your memory profile is activations.

Per token per layer, the pre-norm block saves on the order of $11{,}000$ elements for backward (the two norm inputs and outputs, $q,k,v$ pre- and post-QK-norm and post-RoPE, the expanded $k,v$, the attention output, and SwiGLU's four `(·,1408)` intermediates). In bf16 that is **≈ 22 KB/token/layer**, so **≈ 0.66 MB/token** across 30 layers. Then the logits: $V \times 4$ bytes in fp32 for `logits.float()`, plus another $V\times4$ that `cross_entropy` saves — call it **≈ 0.26 MB/token** on the naive path, or ~0 on the chunked/fused path.

| Tier | micro-batch × seq | tokens/micro-batch | activations | logits (naive / fused) | + persistent | grad accum → 0.5M tok |
|---|---|---|---|---|---|---|
| A100 80GB | 16 × 2048 | 32,768 | ≈ 21.6 GB | ≈ 10.7 GB / ≈ 1.1 GB | ≈ 23.9 GB fused | 16 → 524,288 |
| RTX 4090 24GB | 8 × 2048 | 16,384 | ≈ 10.8 GB | ≈ 5.4 GB / ≈ 1.1 GB | ≈ 13.2 GB fused | 32 → 524,288 |
| Colab T4 16GB | 2 × 2048 | 4,096 | ≈ 2.7 GB | ≈ 1.3 GB / ≈ 1.1 GB | ≈ 5.1 GB fused | 128 → 524,288 |

Three lessons fall out of this table, and they are the reason to build it:

1. **Fused/chunked cross-entropy is not a micro-optimization at this scale.** On the 4090 tier it is the difference between an 18.5 GB step and a 13.2 GB step — one OOM and one comfortable run. The naive path spends more memory on a single `(B,T,32768)` tensor than on the entire 30-layer body.
2. **Activation checkpointing is the other big lever.** Wrapping each `Block` in `torch.utils.checkpoint.checkpoint` stores only the 30 block *inputs* (30 × 512 × 2 B = 30 KB/token, ≈ 1.0 GB at the A100 micro-batch) and recomputes the rest, for roughly +30% step time. That turns 21.6 GB of activations into under 2 GB — enough to quadruple the micro-batch. See [Memory-Efficient Training: Checkpointing, Offloading & LoRA Math](../04-kernels-efficiency/10-memory-efficient-training.html).
3. **The 0.5M-token batch from PLAN §5 is reached by gradient accumulation, not by a bigger card.** All three tiers train the *same* effective batch; they differ only in how many micro-steps it takes. That is what makes the recipe portable.

These are analytic estimates ("on the order of") — real `torch.cuda.max_memory_allocated()` will differ by allocator fragmentation, cuBLAS workspaces, and whether `torch.compile` fused something away. Measure, then compare to the napkin. When they disagree by 2×, you have learned something.

!!! interview "Interview Corner"

    **Q:** At 100M parameters you chose GQA with 2 KV heads, tied embeddings, and a 32k vocab. Walk me through *why each of those specific numbers* rather than the "obvious" MHA / untied / 50k defaults, and what you'd lose if you flipped them.

    **A:** All three are driven by the fact that at 100M the budget — both parameters *and* memory — is dominated by a few big tensors, so cheap-but-good choices compound. (1) **GQA 2 KV heads**: full MHA would add ~0.39M params/layer (~11.8M, over four blocks' worth) and, more importantly, 4× the KV cache — 15 KiB/token → 60 KiB/token, i.e. 30 MiB → 120 MiB at the 2048 context and 120 MiB → 480 MiB at the 8192 mid-trained context — for a quality gain that is negligible at this width. 2 KV heads keeps the model within a hair of MHA quality while cutting cache 4×, which is what makes it cheap to *serve*. (2) **Tied embeddings**: input embedding and output projection are both $V\times d$; tying saves 16.8M params — about a sixth of the model — and at small scale the shared representation actually helps (Press & Wolf). (3) **32k vocab**: a 50k vocab costs 25.7M tied params vs 16.8M; the ~9M saved buys roughly three more transformer blocks, which at small scale (deep-thin, MobileLLM) is a better use of the budget than finer-grained tokenization. And there is a second-order reason interviewers rarely expect: the vocab also sets the size of the training logit tensor, which at a 32k micro-batch is *larger than every other activation combined* — so a lean vocab buys memory headroom as well as parameters. Flip any of them and you bloat the serving footprint (MHA), waste a sixth of the model (untying), or trade depth *and* memory for vocab granularity (50k) — all bad trades *specifically because 100M is small*. At 100B these trades look very different: the embedding is a rounding error, MHA's cache is amortized over far more compute, and a bigger vocab pays off on multilingual coverage.

---

## From `stacklm` to the ecosystem: making the checkpoint loadable

A from-scratch module is a teaching artifact until somebody else's runtime can load it. Ch. 14.11 promises int8/int4 post-training quantization and a measured-latency CPU run, and those paths go through HuggingFace `transformers`, `vLLM`, and `llama.cpp`/GGUF. None of them can load a `torch.save` of `Stack100M` — so here is the bridge, and the honest statement of what it costs.

**The good news: Stack-100M is architecturally a Qwen3.** Grouped-query attention, RMSNorm, SwiGLU, no biases, tied embeddings, and — the distinguishing detail — **QK-norm as an RMSNorm over `head_dim` applied before RoPE** is exactly the `Qwen3ForCausalLM` recipe, and `Qwen3Config` exposes `head_dim` independently of `hidden_size / num_attention_heads`, which our 512/8/64 shape needs. So with one config flag flipped, our checkpoint becomes a first-class citizen of `transformers`, `vLLM`, `SGLang`, `llama.cpp`, and every quantization toolchain that speaks those.

**The one incompatibility is NoPE.** No stock architecture supports "skip RoPE on every 4th layer." You therefore choose, deliberately:

- **Export path A — maximum compatibility.** Train with `nope_every = 0` (pure RoPE). Export as `Qwen3ForCausalLM` with a pure key-rename script. You lose the length-extrapolation ingredient of Ch. 14.8 and gain the entire ecosystem for free.
- **Export path B — keep NoPE.** Ship a `trust_remote_code` model: a `configuration_stacklm.py` / `modeling_stacklm.py` pair registered with `AutoConfig.register` / `AutoModelForCausalLM.register`, which `transformers` loads directly and which vLLM can pick up via `ModelRegistry.register_model("StackLMForCausalLM", StackLMForCausalLM)` in an out-of-tree plugin. `llama.cpp` is the hard one: GGUF conversion needs a `Model` subclass in `convert_hf_to_gguf.py` *and* a matching architecture implemented in C++, so a genuinely new architecture is not a weekend port. For laptop inference with NoPE kept, Ch. 14.11 uses our own round-to-nearest int8/int4 + the `generate()` above on CPU, which is the point of having written it.

The rename map for path A is short enough to print in full:

```python
# stacklm/serve/export_hf.py  --  Stack-100M -> Qwen3ForCausalLM (requires nope_every=0)
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

def to_qwen3(model, cfg):
    assert cfg.nope_every <= 0, "stock Qwen3 has no NoPE layers; use trust_remote_code"
    assert cfg.qk_norm, "Qwen3 expects q_norm/k_norm over head_dim"
    hf_cfg = Qwen3Config(
        vocab_size=cfg.vocab_size, hidden_size=cfg.d_model,
        num_hidden_layers=cfg.n_layers, num_attention_heads=cfg.n_heads,
        num_key_value_heads=cfg.n_kv_heads, head_dim=cfg.head_dim,
        intermediate_size=cfg.intermediate, max_position_embeddings=cfg.max_seq_len,
        rope_theta=cfg.rope_theta, rms_norm_eps=cfg.norm_eps,
        tie_word_embeddings=cfg.tie_embeddings, attention_bias=False,
    )
    sd, out = model.state_dict(), {}
    out["model.embed_tokens.weight"] = sd["tok_emb.weight"]
    out["model.norm.weight"] = sd["final_norm.weight"]
    for i in range(cfg.n_layers):
        s, d = f"blocks.{i}.", f"model.layers.{i}."
        out[d + "input_layernorm.weight"]          = sd[s + "attn_norm.weight"]
        out[d + "post_attention_layernorm.weight"] = sd[s + "mlp_norm.weight"]
        out[d + "self_attn.q_proj.weight"] = sd[s + "attn.wq.weight"]
        out[d + "self_attn.k_proj.weight"] = sd[s + "attn.wk.weight"]
        out[d + "self_attn.v_proj.weight"] = sd[s + "attn.wv.weight"]
        out[d + "self_attn.o_proj.weight"] = sd[s + "attn.wo.weight"]
        out[d + "self_attn.q_norm.weight"] = sd[s + "attn.q_norm.weight"]
        out[d + "self_attn.k_norm.weight"] = sd[s + "attn.k_norm.weight"]
        out[d + "mlp.gate_proj.weight"] = sd[s + "mlp.gate.weight"]
        out[d + "mlp.up_proj.weight"]   = sd[s + "mlp.up.weight"]
        out[d + "mlp.down_proj.weight"] = sd[s + "mlp.down.weight"]
    hf = Qwen3ForCausalLM(hf_cfg)
    hf.load_state_dict(out, strict=False)   # lm_head is tied to embed_tokens
    hf.tie_weights()
    return hf
```

!!! warning "Common pitfall: verify the port numerically, not structurally"

    `load_state_dict` succeeding proves the *shapes* match, not the *semantics*. The two conventions that silently differ between implementations are the **RoPE pair layout** (halved vs interleaved — ours is halved/NeoX, matching HF) and **whether QK-norm is applied before or after RoPE** (ours is before, matching Qwen3). Both produce a model that loads, runs, and generates fluent nonsense. Always assert `torch.allclose(ours(idx).logits_or_logits, hf(idx).logits, atol=1e-3)` on a fixed random input before you trust an export — and do it *before* you quantize, or you will debug two bugs at once.

Once the checkpoint is a `Qwen3ForCausalLM`, the rest of the stack is free: `AutoGPTQ`/`GPTQModel` or `autoawq` for weight-only int4 (Ch. 4.7), `bitsandbytes` for int8 (Ch. 4.8), `vllm serve` for an OpenAI-compatible endpoint (Ch. 7.3), `convert_hf_to_gguf.py` + `llama-cli` for the laptop run (Ch. 14.11). That single config flag — `nope_every` — is the difference between "a research artifact" and "a model anyone can run."

---

## Efficiency variants: options you can bolt on, cited and implemented

The default `Stack-100M` above is what the rest of the capstone trains. But part of the pedagogical point is to show *where the frontier is going* and give runnable code for the three most important efficiency moves — each an option, each cited, none the default path.

### MLA — Multi-head Latent Attention (DeepSeek-V2)

GQA shrinks the KV cache by *sharing* KV heads. **MLA** (DeepSeek-V2, 2024) shrinks it by *compressing*: it projects the KV information down to a small **latent** vector $c^{KV}$ of dimension $d_c \ll n_h d_h$, caches only that latent, and up-projects to per-head K and V on the fly. To keep RoPE compatible — a rotation does not commute with a low-rank projection, so a compressed key cannot carry position — MLA uses a **decoupled** design: a small separate RoPE-carrying key channel of dimension $d_r$ is concatenated to the up-projected content key, and the matching query gets its own $d_r$-dimensional rotary slice. The cache therefore holds $d_c + d_r$ numbers per token per layer instead of $2\,n_{kv}\,d_h$.

Here is the complete module — decoupled RoPE included, because a positionless MLA is not MLA, it is a bug:

```python
class MLAttention(nn.Module):
    """Multi-head Latent Attention (DeepSeek-V2, 2024). Caches a d_c latent plus a
    single shared d_rope key that carries position. Owns its own RoPE tables because
    the decoupled channel has width d_rope, not head_dim."""

    def __init__(self, cfg: StackConfig, layer_idx: int = 0, d_c: int = 128, d_rope: int = 32):
        super().__init__()
        self.H, self.dh, self.d_c, self.dr = cfg.n_heads, cfg.head_dim, d_c, d_rope
        self.layer_idx = layer_idx
        self.use_rope = cfg.uses_rope(layer_idx)
        d = cfg.d_model
        self.wq   = nn.Linear(d, self.H * self.dh, bias=False)   # content queries
        self.wqr  = nn.Linear(d, self.H * self.dr, bias=False)   # decoupled rotary queries
        self.w_dkv = nn.Linear(d, d_c, bias=False)               # down-project  <-- CACHE c_kv
        self.kv_norm = RMSNorm(d_c, cfg.norm_eps)
        self.w_uk = nn.Linear(d_c, self.H * self.dh, bias=False) # up-project keys  (NOT cached)
        self.w_uv = nn.Linear(d_c, self.H * self.dh, bias=False) # up-project values(NOT cached)
        self.w_kr = nn.Linear(d, self.dr, bias=False)            # ONE shared rotary key <-- CACHE
        self.wo   = nn.Linear(self.H * self.dh, d, bias=False)
        cos, sin = build_rope_cache(d_rope, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("cos_r", cos, persistent=False)
        self.register_buffer("sin_r", sin, persistent=False)

    def forward(self, x, cos=None, sin=None, attn_mask=None, kv_cache=None,
                start_pos=0, record=None):
        B, T, _ = x.shape
        H, dh, dr = self.H, self.dh, self.dr
        c_kv = self.kv_norm(self.w_dkv(x))                          # (B,T,d_c)   CACHE THIS
        k_r = self.w_kr(x).view(B, T, 1, dr).transpose(1, 2)        # (B,1,T,dr)  AND THIS
        q_c = self.wq(x).view(B, T, H, dh).transpose(1, 2)
        q_r = self.wqr(x).view(B, T, H, dr).transpose(1, 2)
        if self.use_rope:
            pos = torch.arange(start_pos, start_pos + T, device=x.device)
            cr, sr = self.cos_r[pos], self.sin_r[pos]
            q_r, k_r = apply_rope(q_r, k_r, cr, sr)                 # rotate BOTH sides
        # (a real cache would append (c_kv, k_r) here and slice the prefix back out)
        S = c_kv.shape[1]
        k_c = self.w_uk(c_kv).view(B, S, H, dh).transpose(1, 2)
        v   = self.w_uv(c_kv).view(B, S, H, dh).transpose(1, 2)
        q = torch.cat([q_c, q_r], dim=-1)                           # (B,H,T,dh+dr)
        k = torch.cat([k_c, k_r.expand(B, H, S, dr)], dim=-1)       # (B,H,S,dh+dr)
        out = F.scaled_dot_product_attention(                       # NOTE: V is narrower
            q, k, v, attn_mask=attn_mask, is_causal=(attn_mask is None),
            scale=(dh + dr) ** -0.5)
        return self.wo(out.transpose(1, 2).contiguous().view(B, T, -1))
```

!!! note "Aside: the absorption trick is why MLA is *fast*, not just small"

    Compressing the cache would be pointless if you had to up-project it on every decode step — that is a $d_c \times n_h d_h$ matmul per token per layer. You do not, because the up-projection can be **absorbed into the query**. For head $h$ with up-projection block $W_{UK}^{(h)} \in \mathbb{R}^{d_h \times d_c}$, the content logit is
    $$
    q_h^\top \big(W_{UK}^{(h)} c\big) \;=\; \big(W_{UK}^{(h)\top} q_h\big)^\top c,
    $$
    so you pre-multiply the *query* into the $d_c$-dimensional latent space once and dot it directly against the cached $c^{KV}$. Per-head keys are never materialized at decode. The same absorption folds $W_{UV}$ into $W_O$. (Verify the identity numerically in five lines — it is exact to fp32 round-off.) Without absorption, MLA is a smaller cache and a slower decoder; with it, it is both smaller and faster, which is why DeepSeek could serve it.

At 100M the arithmetic is honest and unflattering to MLA: with $d_c=128, d_r=32$ the cache is $(128+32)\times 2 \times 30 = 9{,}600$ bytes/token versus GQA's $15{,}360$ — a 1.6× win on a cache that was already only 30 MiB. DeepSeek's win is dramatic because their MHA baseline is 128 heads of dim 128; ours is 2 heads of dim 64. So MLA is here to be *understood and measured*, not because Stack-100M needs it. For a model you intend to serve at long context or high concurrency, it is the upgrade.

### MTP — Multi-Token Prediction (DeepSeek-V3)

Standard training predicts token $t{+}1$ from position $t$. **Multi-Token Prediction** (DeepSeek-V3, 2024; Gloeckle et al., *Better & Faster Large Language Models via Multi-token Prediction*, 2024) adds an auxiliary module that also predicts token $t{+}2$, giving a **denser training signal** and, as a bonus, a natural draft head for self-speculative decoding at inference (see [Speculative Decoding](../07-inference-serving/06-speculative-decoding.html)).

The detail that most reimplementations get wrong: DeepSeek-V3's MTP module is *not* just an extra block on the trunk state. It concatenates the trunk hidden state $h_i$ with the **embedding of the already-known next token** $t_{i+1}$, projects the pair back to $d$, and only then runs a transformer block — so the module predicts $t_{i+2}$ *given* $t_{i+1}$, keeping the full causal chain intact. That conditioning is the whole point; stacking modules $k=1,2,\dots$ sequentially (each consuming the previous module's output) is how you extend to further horizons.

```python
class MTPHead(nn.Module):
    """DeepSeek-V3-style MTP module (depth k=1: predict t+2 given the trunk state at t
    and the embedding of t+1). Shares the tied embedding AND the tied output head."""
    def __init__(self, cfg: StackConfig, tok_emb: nn.Embedding, lm_head: nn.Linear):
        super().__init__()
        self.h_norm = RMSNorm(cfg.d_model, cfg.norm_eps)   # RMSNorm(h_i)
        self.e_norm = RMSNorm(cfg.d_model, cfg.norm_eps)   # RMSNorm(Emb(t_{i+1}))
        self.proj = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False)   # M_k
        self.block = Block(cfg, layer_idx=0)               # TRM_k (pre-norms internally)
        self.tok_emb, self.lm_head = tok_emb, lm_head      # SHARED, not copied

    def forward(self, h, next_ids, cos, sin, targets_plus2=None, attn_mask=None):
        e = self.tok_emb(next_ids)                                    # (B,T,d)
        z = self.proj(torch.cat([self.h_norm(h), self.e_norm(e)], dim=-1))
        h2 = self.block(z, cos, sin, attn_mask=attn_mask)
        logits = self.lm_head(h2)
        loss = None
        if targets_plus2 is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]),
                                   targets_plus2.reshape(-1), ignore_index=-100)
        return logits, loss

# training step:  total = main_loss + lambda_mtp * mtp_loss
# DeepSeek-V3 reports lambda = 0.3 for the bulk of training, lowered to 0.1 at the end.
```

Its own parameters — one block (2,819,200) + the $2d \times d$ projection (524,288) + two norms (1,024) — come to **3,344,512**, a 3.3% training-time overhead that you discard at inference unless you keep it for self-speculation. In the capstone's ablation menu (Ch. 14.5/14.7) this is a clean, cheap experiment: does a denser signal buy faster early loss descent at 100M?

### A Liquid LFM2-style gated short-conv hybrid block

Attention is not the only sequence mixer. **Liquid AI's LFM2** (2025) interleaves attention with cheap **gated short-range convolution** blocks — a depthwise causal conv (kernel ~3) wrapped in multiplicative input/output gates — which capture local structure at a fraction of attention's cost and carry no KV cache at all. This is a concrete instance of the broader hybrid-architecture trend covered in [Beyond Attention: SSMs, Mamba, RWKV & Linear Attention](../02-transformer/11-ssm-and-alternatives.html). Replacing some of `Stack-100M`'s attention blocks with conv-mixer blocks trades a little long-range capacity for real decode-time speed.

```python
class GatedShortConv(nn.Module):
    """LFM2-style (Liquid AI, 2025) double-gated causal short convolution mixer.
    No KV cache; O(T) local mixing. A cheap alternative to an attention block.
    (For decode, the 'cache' is the last k-1 inputs -- a 2-element ring buffer.)"""
    def __init__(self, cfg: StackConfig, kernel: int = 3):
        super().__init__()
        d = cfg.d_model
        self.k = kernel
        self.in_b = nn.Linear(d, d, bias=False)   # input gate B
        self.in_c = nn.Linear(d, d, bias=False)   # output gate C
        self.in_x = nn.Linear(d, d, bias=False)   # value path
        # depthwise causal conv over time (groups=d => per-channel filter)
        self.conv = nn.Conv1d(d, d, kernel, groups=d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, *args, **kwargs):        # ignores cos/sin/mask: no positions needed
        z = self.in_b(x) * self.in_x(x)           # input gate * value
        z = z.transpose(1, 2)                     # (B, d, T) for conv1d
        z = F.pad(z, (self.k - 1, 0))             # left-pad => causal
        z = self.conv(z).transpose(1, 2)          # (B, T, d)
        return self.out(self.in_c(x) * z)         # output gate
```

The design point (LFM2's own recipe): use *mostly* conv blocks with a *few* attention blocks for long-range routing. At 100M this is a research option, not the default — our narrow auto-research agent (Ch. 14.10) benefits more from attention's exact retrieval than from conv's speed — but the code is here so a reader can build the hybrid and measure the tradeoff.

!!! tip "Practitioner tip: keep variants behind the config, never fork the model"

    Every variant above is a *drop-in module* with the same `(x, cos, sin, attn_mask=, kv_cache=, start_pos=) -> x` contract as the default `Attention` (`GatedShortConv` absorbs the extras with `*args, **kwargs`). The right engineering pattern is a factory: `Block` reads `cfg.mixer in {"gqa","mla","conv"}` and constructs the right mixer, so you can ablate architectures by changing one config field and re-running the *same* training loop from Ch. 14.7. Forking `model.py` per variant is how capstones rot. One model, config-selected mixers — Exercise 6 has you build it.

---

## Key Takeaways

!!! key "Key Takeaways"

    - **Deep-and-thin is the small-model bet**: at fixed 100M params, 30 layers × 512 width beats shallow-wide (MobileLLM, 2024); depth = sequential reasoning steps, which is what small models are bottlenecked on. The width-per-layer ratio $d/L \approx 17$ is ~4× thinner than GPT-2-small's ~64.
    - **The exact config is frozen** (PLAN §1): `vocab 32768, d_model 512, n_layers 30, 8 heads / 2 KV heads, head_dim 64, SwiGLU 1408, tied embeddings` → **exactly 101,353,728 params** (≈101.4M), of which **84,576,512** is the non-embedding body and 16,777,216 the tied embedding. Reproduce that integer by hand, then assert it in CI.
    - **GQA (2 KV heads)** cuts the KV cache 4× — 15 KiB/token, 30 MiB at the 2048 context (120 MiB at 8192) versus MHA's 60 KiB/token — at negligible quality cost; **tied embeddings** save ~16% of the model; **a 32k vocab** frees ~9M params (≈3 blocks) versus 50k *and* shrinks the training logit tensor.
    - **Stability is engineered, not hoped for**: pre-norm + RMSNorm (fp32), **QK-norm** to bound attention logits within $\pm 8$ by Cauchy–Schwarz (what buys the high Muon learning rate), **z-loss** on the log-partition, optional Gemma-2 soft-caps applied on *both* train and inference paths, and $1/\sqrt{2L}$ residual-init scaling for the deep stack.
    - **RoPE + NoPE-every-4th-layer** (SmolLM3, 2025; Kazemnejad et al., 2023) with **explicit `position_ids`** — the indirection that makes document packing, incremental decode, and the 2048→8192 context extension all work from one code path.
    - **Correct masking is a model responsibility.** Document-aware packing needs a causal-AND-same-document mask and per-document position resets; incremental decode needs a bottom-right-aligned mask, because SDPA's `is_causal=True` is *top-left* aligned and silently reads only position 0. `flex_attention`'s `create_block_mask` (block-sparse, fused, and `score_mod`-capable) and `flash_attn_varlen_func` + `cu_seqlens` are the two production answers.
    - **At 100M the logits are the memory bomb, not the model.** The whole trainable state is ≈1.3 GB; a $(16, 2048, 32768)$ fp32 logit tensor is ≈4.3 GB by itself. Chunked/fused linear-cross-entropy (Liger-Kernel, Cut Cross-Entropy) plus activation checkpointing is what makes the recipe fit a 4090 as well as an A100.
    - **Plan the ecosystem exit before you train.** Stack-100M with `nope_every=0` is architecturally a **Qwen3**, so a pure key-rename makes it loadable by `transformers`/vLLM/`llama.cpp` and every quantizer; keep NoPE and you owe a `trust_remote_code` pair (and a `ModelRegistry.register_model` for vLLM). Verify any port numerically — shapes matching is not semantics matching.
    - **Efficiency variants are options, not the default**: MLA (DeepSeek-V2) compresses KV into a latent — and the *absorption* of $W_{UK}$ into the query is why it is fast, not just small; MTP (DeepSeek-V3) conditions on the next token's embedding before predicting $t{+}2$; an LFM2-style gated short-conv block trades long-range capacity for KV-free decode speed.

!!! sota "State of the Art & Resources (2026)"
    Every mechanism wired into `Stack-100M` — GQA, RMSNorm, QK-norm, RoPE/NoPE, SwiGLU, MLA, MTP — is a named, shipped choice in today's frontier open models; the list below links the primary sources plus the repos and write-ups worth reading alongside the code in this chapter.

    **Foundational work**

    - [Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021)](https://arxiv.org/abs/2104.09864) — the RoPE mechanism `build_rope_cache`/`apply_rope` implement.
    - [Zhang & Sennrich, *Root Mean Square Layer Normalization* (2019)](https://arxiv.org/abs/1910.07467) — RMSNorm, used for every norm in the model including QK-norm.
    - [Shazeer, *GLU Variants Improve Transformer* (2020)](https://arxiv.org/abs/2002.05202) — SwiGLU, the MLP sublayer.

    **Recent advances (2023–2026)**

    - [Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (2023)](https://arxiv.org/abs/2305.13245) — the 8-query/2-KV-head attention design.
    - [Liu et al., *MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases* (2024)](https://arxiv.org/abs/2402.14905) — the deep-and-thin evidence behind `d_model=512, n_layers=30`.
    - [DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model* (2024)](https://arxiv.org/abs/2405.04434) — introduces MLA, implemented here as the optional `MLAttention` module.
    - [DeepSeek-AI, *DeepSeek-V3 Technical Report* (2024)](https://arxiv.org/abs/2412.19437) — ships Multi-Token Prediction at scale, the basis for `MTPHead`.
    - [Gemma Team, *Gemma 2: Improving Open Language Models at a Practical Size* (2024)](https://arxiv.org/abs/2408.00118) — the logit/attention soft-capping this chapter exposes as `logit_soft_cap`/`attn_soft_cap`.
    - [Kimi Team, *Kimi K2: Open Agentic Intelligence* (2025)](https://arxiv.org/abs/2507.20534) — MuonClip's QK-clip, the trillion-parameter cousin of this chapter's QK-norm stability argument.
    - [Wijmans et al., *Cut Your Losses in Large-Vocabulary Language Models* (2024)](https://arxiv.org/abs/2411.09009) — the fused linear-cross-entropy that removes this chapter's dominant activation cost.

    **Open-source & tools**

    - [huggingface/smollm](https://github.com/huggingface/smollm) — training code and recipe for SmolLM3, the model that popularized the every-4th-layer NoPE interleave this chapter adopts.
    - [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) — the fused, GQA-aware attention kernel PyTorch's `scaled_dot_product_attention` dispatches to, and the home of `flash_attn_varlen_func` + `cu_seqlens` for packed sequences.
    - [PyTorch FlexAttention](https://pytorch.org/blog/flexattention/) — `create_block_mask` / `score_mod`: document-causal block-sparse masking and soft-capping inside a fused kernel.
    - [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) — Triton kernels for fused RMSNorm, SwiGLU, RoPE and `LigerFusedLinearCrossEntropy`.
    - [apple/ml-cross-entropy](https://github.com/apple/ml-cross-entropy) — Cut Cross-Entropy, the loss that never materializes a logit tensor.
    - [vllm-project/vllm](https://github.com/vllm-project/vllm) — PagedAttention and `ModelRegistry.register_model` for serving a custom architecture.

    **Go deeper**

    - [SmolLM3: smol, multilingual, long-context reasoner](https://huggingface.co/blog/smollm3) — HuggingFace's own writeup of the GQA + NoPE ablations behind the design choices this chapter cites.

## Further reading

- Liu et al., *MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases*, 2024 — the deep-and-thin evidence.
- Zhang & Sennrich, *Root Mean Square Layer Normalization*, NeurIPS 2019 — RMSNorm.
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021 — RoPE.
- Kazemnejad et al., *The Impact of Positional Encoding on Length Generalization in Transformers* ("NoPE"), NeurIPS 2023; and the SmolLM3 model report, HuggingFace, 2025 — RoPE/NoPE interleaving.
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, 2023 — grouped-query attention.
- Henry et al., *Query-Key Normalization for Transformers*, 2020 — QK-norm; revived by OLMo 2 (2024), Qwen3 (2025) and Kimi K2 (Moonshot, 2025).
- Shazeer, *GLU Variants Improve Transformer*, 2020 — SwiGLU.
- Press & Wolf, *Using the Output Embedding to Tie Word Vectors*, 2017 — tied embeddings.
- Dao et al., *FlashAttention* (2022) and *FlashAttention-2* (2023) — the fused attention kernels SDPA dispatches to, and the varlen/`cu_seqlens` packing API.
- Wijmans et al., *Cut Your Losses in Large-Vocabulary Language Models*, 2024 — fused linear-cross-entropy for large-vocab / small-model regimes.
- DeepSeek-AI, *DeepSeek-V2* (2024, MLA) and *DeepSeek-V3* (2024, MTP); Gloeckle et al., *Better & Faster Large Language Models via Multi-token Prediction*, 2024.
- Liquid AI, *LFM2* technical report, 2025 — gated short-convolution hybrid blocks.
- Gemma Team, *Gemma 2*, 2024 — logit soft-capping (final 30.0, attention 50.0) and small-model design choices.

---

## Exercises

**1.** The chapter fixes `d_model = 512`, `n_layers = 30`, giving a width-per-layer of $d/L \approx 17$, versus $\approx 64$ for GPT-2-small ($768/12$). Suppose a colleague proposes instead a *shallow-and-wide* 100M model with `d_model = 1024`, `n_layers ~= 8`. State its width-per-layer, and explain — using the chapter's argument — why this is expected to be *worse* at 100M even though it has the same parameter budget. What single quantity does depth buy that width does not?

??? note "Solution"
    Width-per-layer for the proposal is $d/L = 1024/8 = 128$ — about 7.5x wider per layer than Stack-100M's $\approx 17$, and even wider than GPT-2-small's $\approx 64$. It sits at the extreme *shallow-wide* end of the aspect-ratio axis.

    The chapter's argument (Liu et al., *MobileLLM*, 2024) is that **at sub-billion scale the evidence flips decisively toward depth**: at fixed parameters, deeper-and-thinner models are consistently more accurate. The mechanism is that capability at small scale is bottlenecked by the number of *sequential nonlinear transformations* the residual stream can undergo — the model's "reasoning depth" — not by the dimensionality of each transformation. A wide-but-shallow model can store a lot per token but can only compose a few steps of computation before emitting; a deep-but-narrow model composes many. So the shallow-wide 8-layer model can only compose $\approx 8$ steps of computation, while Stack-100M composes $\approx 30$, and at fixed budget the latter wins on tasks like commonsense reasoning.

    What depth buys is **sequential composition** (reasoning steps); width only buys per-step representational capacity, which the chapter argues is not the binding constraint at 100M. (At billions of parameters the two trade roughly evenly and width starts to win on hardware efficiency — but that regime is not ours.)

**2.** With `nope_every = 4`, the rule is that layer index $\ell$ (0-based) uses NoPE iff $(\ell + 1) \bmod 4 = 0$. (a) List the NoPE layers and count them; how many of the 30 layers keep RoPE? (b) Why can a decoder-only model afford to drop the positional encoding on some layers at all, and what property does the NoPE mixture buy that pure-RoPE does not? (c) Why not go all the way to a *pure*-NoPE model? (d) The chapter says `nope_every = 0` is the switch that "makes the checkpoint loadable by stock `transformers`/vLLM." Why?

??? note "Solution"
    (a) NoPE layers are those with $(\ell+1)\bmod 4 = 0$, i.e. $\ell = 3, 7, 11, 15, 19, 23, 27$ — **7 layers**. The remaining $30 - 7 = 23$ layers keep RoPE.

    (b) A decoder-only model with a **causal mask** can *infer* absolute position from the mask alone: the number of tokens a position is allowed to attend to is itself a positional signal. So the model does not strictly need an injected encoding on every layer. Layers freed from RoPE are not tied to the specific rotation frequencies seen during training, and empirically the RoPE/NoPE mixture **generalizes to longer sequences than seen in training** — exactly the length-robustness the chapter wants for the 2048 -> 8192 context extension in mid-training (Ch. 14.8).

    (c) A pure-NoPE model tends to underperform on *short* context: the causal-mask position signal is weak and diffuse compared to RoPE's sharp relative encoding. So the design interleaves rather than removes — RoPE on the majority of layers for local positional precision and sample-efficient short-context modeling, a minority (1-in-4) of NoPE layers for length extrapolation. The 1-in-4 ratio is SmolLM3's tuned constant, not a law.

    (d) Because no stock architecture in `transformers` expresses "skip RoPE on every 4th layer." Every other Stack-100M component — GQA, RMSNorm, SwiGLU, no biases, tied embeddings, and QK-norm as an RMSNorm over `head_dim` applied *before* RoPE — is exactly the `Qwen3ForCausalLM` recipe, and `Qwen3Config` even exposes `head_dim` independently of `hidden_size/num_attention_heads`, which our 512/8/64 shape needs. With `nope_every=0` the export is a pure key rename (`blocks.i.attn.wq` -> `model.layers.i.self_attn.q_proj`, etc.) and the checkpoint is a first-class citizen of `transformers`, vLLM, SGLang, `llama.cpp`/GGUF, and the GPTQ/AWQ quantizers. Keep NoPE and you owe a `trust_remote_code` `configuration_stacklm.py`/`modeling_stacklm.py` pair plus a vLLM `ModelRegistry.register_model` — and GGUF conversion needs new C++, which is a real project, not a weekend.

**3.** Reproduce the parameter arithmetic by hand for two variants.
(a) Count the parameters of **one** Stack-100M transformer block (attention + SwiGLU + norms), using $d = 512$, $n_h = 8$, $n_{kv} = 2$, $d_h = 64$, $d_{\text{ff}} = 1408$, then assemble the full model total.
(b) Now suppose the block used **full multi-head attention** ($n_{kv} = 8$) instead of GQA. How many extra parameters per layer, and how many over all 30 layers? Relate the total to "blocks' worth" of budget.

??? note "Solution"
    (a) **Attention** has four projections. With $d = 512$, $n_h d_h = 512$, $n_{kv} d_h = 128$:

    $$
    \begin{aligned}
    W_Q &: 512 \times 512 = 262{,}144,\\
    W_K &: 512 \times 128 = 65{,}536,\\
    W_V &: 512 \times 128 = 65{,}536,\\
    W_O &: 512 \times 512 = 262{,}144.
    \end{aligned}
    $$

    Attention subtotal $= 655{,}360$.

    **SwiGLU** has three matrices (gate, up, down), each $512 \times 1408$: $3 \times 512 \times 1408 = 2{,}162{,}688$.

    **Norms**: two RMSNorm weight vectors of length $d=512$ (pre-attn, pre-MLP) plus two QK-norm vectors of length $d_h=64$: $2\times512 + 2\times64 = 1152$.

    Block total $= 655{,}360 + 2{,}162{,}688 + 1{,}152 = \mathbf{2{,}819{,}200}$.

    Full model: $30 \times 2{,}819{,}200 = 84{,}576{,}000$; plus the final RMSNorm ($512$) gives the non-embedding body $\mathbf{84{,}576{,}512}$; plus the tied embedding $32768\times512 = 16{,}777{,}216$ gives

    $$
    16{,}777{,}216 + 84{,}576{,}512 = \mathbf{101{,}353{,}728}\ \ (\approx 101.4\text{M}).
    $$

    Nothing is rounded away — `Stack100M.num_params()` returns this integer exactly, and the chapter's CI asserts it.

    (b) Under full MHA, $n_{kv} = 8$, so $W_K$ and $W_V$ each become $512 \times (8 \times 64) = 512 \times 512 = 262{,}144$ instead of $65{,}536$. Extra per matrix $= 196{,}608$; for the two matrices, $393{,}216 \approx 0.39\text{M}$ per layer. Over 30 layers:

    $$
    30 \times 393{,}216 = 11{,}796{,}480 \approx 11.8\text{M}.
    $$

    That is $11.8\text{M} / 2.82\text{M} \approx 4.2$ **blocks' worth** of parameters spent for a quality gain the chapter calls negligible at this width. GQA shrinks $W_K, W_V$ by 4x (128 vs 512 columns) precisely to reclaim that budget — and cuts the KV cache 4x as a bonus.

**4.** The chapter derives a GQA KV cache of $15{,}360$ bytes/token and says mid-training (Ch. 14.8) extends the context from 2048 to 8192. (a) Re-derive the per-token figure from the config, then compute the full KV footprint of a single 8192-token sequence under GQA. (b) What would the same sequence cost under full multi-head attention (8 KV heads)? (c) With `d_c = 128, d_rope = 32`, what does MLA cache per token per layer, and what is the ratio to GQA? One sentence: why does the chapter still say MLA's *memory* win "is not the reason to reach for it here"?

??? note "Solution"
    (a) Per token you store **both** K and V, for $n_{kv}=2$ heads of $d_h=64$, in **every** one of $L=30$ layers, at 2 bytes each (bf16):

    $$
    2 \times 2 \times 64 \times 30 \times 2 = 15{,}360\ \text{bytes} = 15\ \text{KiB/token}.
    $$

    A full 8192-token context is $8192 \times 15{,}360 = 125{,}829{,}120$ bytes $= 120$ MiB. (At the 2048 pretrain context it is 30 MiB — `KVCache.nbytes()` prints exactly $31{,}457{,}280$.)

    (b) MHA uses 8 KV heads instead of 2 — a 4x larger cache per token (60 KiB/token). The same 8192-token sequence costs $4 \times 120 = \mathbf{480}$ **MiB**, and the 2048 context costs 120 MiB. That 4x is what makes the model cheap to *serve* at long context and high concurrency.

    (c) MLA caches the latent plus the shared decoupled rotary key: $d_c + d_r = 128 + 32 = 160$ values per token per layer, versus GQA's $2 \times n_{kv} \times d_h = 2\times2\times64 = 256$ — a ratio of $160/256 = 0.625$, i.e. only a **1.6x** win (9,600 vs 15,360 bytes/token). DeepSeek's headline MLA win comes from replacing a *128-head, 128-dim* MHA baseline; against an already-lean 2-KV-head GQA at $d_h=64$ there is very little left to compress, so MLA belongs in this chapter as a mechanism to understand and measure — and for its *decode-speed* absorption trick — not as a memory necessity at 100M.

**5.** QK-norm bounds the attention logits "by construction." Take $d_h = 64$ and assume the RMSNorm scale $\gamma \approx 1$. (a) After RMS-normalizing a query vector $q$ over its 64 dimensions, what is $\|q\|$ (approximately)? (b) Use the Cauchy-Schwarz inequality to bound the *scaled* pre-softmax logit $\tfrac{1}{\sqrt{d_h}}\,q\cdot k$. (c) Contrast with the un-normalized case where the per-component std $\sigma$ drifts up to $6$: what is the scaled-logit std then, and why does that NaN the run? (d) Why is QK-norm applied *before* RoPE, and why does the ordering not change the magnitude bound?

??? note "Solution"
    (a) RMSNorm divides $q$ by $\sqrt{\tfrac{1}{d_h}\sum_i q_i^2 + \epsilon}$, which forces the mean square of the output to $\approx 1$ (with $\gamma \approx 1$). Then $\sum_i q_i^2 \approx d_h$, so

    $$
    \|q\| = \sqrt{\textstyle\sum_i q_i^2} \approx \sqrt{d_h} = \sqrt{64} = 8,
    $$

    and likewise $\|k\| \approx 8$.

    (b) By Cauchy-Schwarz, $|q\cdot k| \le \|q\|\,\|k\| = 8 \times 8 = 64$. After the $1/\sqrt{d_h} = 1/8$ scaling,

    $$
    \Big|\tfrac{1}{8}\, q\cdot k\Big| \le \tfrac{64}{8} = 8.
    $$

    The scaled logit is bounded to $\pm 8$ by construction, regardless of how large the raw projections $W_Q, W_K$ grow. A learned temperature can still be reintroduced through the RMSNorm $\gamma$.

    (c) Without QK-norm, each product $q_i k_i$ has variance $\sigma^4$, so $q\cdot k$ over 64 dims has std $\approx \sigma^2\sqrt{d_h} = 8\sigma^2$; after $1/8$ scaling the std is $\approx \sigma^2$. At $\sigma = 6$ that is $\approx 36$, and the *max* logit over a long context (a few std out) can exceed 100. A softmax with a $\sim$100 gap between the top logit and the rest is numerically a hard argmax: its gradient w.r.t. the losing logits is $\approx e^{-100} \approx 0$, so the attention pattern freezes, learning stalls, and one bad step NaNs the run. QK-norm removes this failure mode structurally rather than tiptoeing around it with a lower LR — which is what buys the high Muon learning rate that makes the run cheap. (The `record=` hook in `Attention.forward` logs exactly this per-head max so you can watch it; Ch. 14.6's QK-clip reads the same signal.)

    (d) QK-norm is applied first so the learned per-dimension scale $\gamma$ acts in the *unrotated content frame* (the standard OLMo 2 / Qwen3 / SmolLM3 order): we normalize the content geometry of $q, k$, then inject position. The ordering does **not** change the magnitude bound because RoPE is a rotation and rotations are norm-preserving ($\|q\|$ is unchanged by RoPE); it only changes how $\gamma$ interacts with the rotation, not the Cauchy-Schwarz bound of $\pm 8$. (It *does* matter for checkpoint portability: Qwen3 uses this same order, so an exported checkpoint is numerically faithful. Swap the order and the weights load but the model is wrong.)

**6.** Implement the "one model, config-selected mixers" pattern from the chapter's practitioner tip. Add a `mixer` field to `StackConfig` and rewrite `Block` so it constructs `Attention` (GQA), `MLAttention`, or `GatedShortConv` from `cfg.mixer` and calls it with the *same* signature the training loop already uses. The rest of the training loop (Ch. 14.7) must run unchanged. Then extend it to a true LFM2-style hybrid (mostly conv, attention every 4th layer).

??? note "Solution"
    The key design move is to make every mixer accept the **same** call signature — `(x, cos, sin, attn_mask=, kv_cache=, start_pos=, record=)` — rather than having `Block` special-case each one. `MLAttention` already owns its own $d_{\text{rope}}$ RoPE tables (it must: the decoupled channel has width `d_rope`, not `head_dim`), so it can simply ignore the `cos/sin` it is handed; `GatedShortConv` swallows the extras with `*args, **kwargs`. Then the factory is three lines and `Block.forward` never branches:

    ```python
    from dataclasses import dataclass

    @dataclass
    class StackConfig:
        # ... all existing fields unchanged ...
        mixer: str = "gqa"          # {"gqa", "mla", "conv"} -- selects the sequence mixer
        attn_every: int = 0         # >0: LFM2-style hybrid, attention every k-th layer

    def make_mixer(cfg: StackConfig, layer_idx: int) -> nn.Module:
        kind = cfg.mixer
        # LFM2-style hybrid: conv everywhere EXCEPT every attn_every-th layer
        if cfg.attn_every > 0:
            kind = "gqa" if ((layer_idx + 1) % cfg.attn_every == 0) else "conv"
        if kind == "gqa":
            return Attention(cfg, layer_idx)
        if kind == "mla":
            return MLAttention(cfg, layer_idx)      # owns its own d_rope RoPE tables
        if kind == "conv":
            return GatedShortConv(cfg)              # absorbs cos/sin/mask via *args
        raise ValueError(f"unknown mixer: {kind!r}")

    class Block(nn.Module):
        def __init__(self, cfg: StackConfig, layer_idx: int):
            super().__init__()
            self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
            self.attn = make_mixer(cfg, layer_idx)   # name kept: checkpoints stay loadable
            self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
            self.mlp = SwiGLU(cfg)

        def forward(self, x, cos, sin, attn_mask=None, kv_cache=None, start_pos=0, record=None):
            x = x + self.attn(self.attn_norm(x), cos, sin, attn_mask=attn_mask,
                              kv_cache=kv_cache, start_pos=start_pos, record=record)
            x = x + self.mlp(self.mlp_norm(x))
            return x
    ```

    Points that keep the rest of the capstone unchanged:

    - `Block.forward` keeps the exact signature `Stack100M` already calls, so `Stack100M` needs **no edits**; ablating architectures is a one-field change (`StackConfig(mixer="mla")`, `StackConfig(attn_every=4)`) re-running the *same* Ch. 14.7 loop and Ch. 14.5 scaling ladder.
    - Keeping the attribute name `self.attn` (rather than `self.mixer`) means state-dict keys do not shift, so a checkpoint trained with one mixer still loads its norms and MLP into another config — useful for architecture-transfer ablations.
    - **Two honest caveats the factory must not hide.** (i) `MLAttention` as written appends nothing to a `KVCache` — it would need its own `(c_kv, k_r)` cache, since the whole point is that it caches a *different tensor* than GQA does; until you add that, `mixer="mla"` is a training-only path and `generate()` should assert against it. (ii) `GatedShortConv` has no positional encoding *and* no long-range mixing, so a pure-conv stack is not a language model in any useful sense — LFM2's recipe is a hybrid, which is what `attn_every` gives you. Wire either one in without knowing this and you get a mysteriously bad loss curve and no idea why.

**7.** You retrofit a KV cache into `Attention` and, on the decode path, keep calling `F.scaled_dot_product_attention(q, k, v, is_causal=True)` with `q_len = 1` and `kv_len = N`. (a) What does PyTorch actually compute, and why? (b) Why does training loss look fine while generations are fluent nonsense? (c) Give two correct fixes. (d) Write the single test that would have caught this.

??? note "Solution"
    (a) PyTorch's `is_causal=True` builds the causal mask **top-left aligned** on the $q_{\text{len}} \times kv_{\text{len}}$ score matrix. With $q_{\text{len}}=1$ that is a single row whose only allowed column is index 0, so the decode step attends **exclusively to the first cached position** and ignores everything else, at every layer. (Verify it in three lines: the output equals `v[:, :, :1, :]` exactly.)

    (b) Training never exercises the rectangular case — during training $q_{\text{len}} = kv_{\text{len}} = T$, where top-left and bottom-right alignment coincide and the mask is the ordinary lower triangle. So the loss curve, the eval perplexity, and every gradient are correct. The bug lives *only* in the incremental-decode path, which is why it survives to production and shows up as "the model was fine yesterday, why does it repeat itself?"

    (c) (i) Build the mask from **positions** rather than from indices: `q_pos = arange(start_pos, start_pos+T)`, `kv_pos = arange(kv_len)`, mask `= q_pos[:, None] >= kv_pos[None, :]`. This is what `Stack100M._build_mask` does and it is correct for prefill, chunked prefill, and single-token decode alike. (ii) Pass `attn_mask=torch.nn.attention.bias.causal_lower_right(q_len, kv_len)`, which asks PyTorch for the bottom-right-aligned causal bias and still dispatches to a fused kernel. (A third, cheap-but-narrow option: when $q_{\text{len}}=1$ exactly, no mask is needed at all, since the single query legitimately attends to every cached key — but that breaks the moment you do chunked prefill.)

    (d) **Greedy generation must be cache-invariant.** One assertion:

    ```python
    torch.manual_seed(0); a = model.generate(prompt, 16, temperature=0.0, use_cache=True)
    torch.manual_seed(0); b = model.generate(prompt, 16, temperature=0.0, use_cache=False)
    assert torch.equal(a, b)
    ```

    The `use_cache=False` path recomputes the full prefix each step, so it is slow but unambiguously correct; any cache, mask, or `position_ids` bug makes the two diverge, usually within the first two or three tokens. Its stronger sibling — `full[:, k:] == incremental` on raw logits, as in the chapter's CI block — localizes the failure to a layer instead of a token.

**8.** Estimate the peak training memory for one micro-batch of $8 \times 2048$ tokens on an RTX 4090 (24 GB), using the chapter's per-token figures: ≈22 KB/token/layer of bf16 activations over 30 layers, and a naive fp32 loss path costing ≈0.26 MB/token of logits. (a) Break down the total. (b) Which single term dominates, and by how much does chunked fused cross-entropy reduce it? (c) You now want to double the micro-batch without OOM. Name the one change that gets you there and state its cost.

??? note "Solution"
    (a) $8 \times 2048 = 16{,}384$ tokens.

    - **Persistent state**: fp32 master weights $101{,}353{,}728 \times 4 \approx 0.41$ GB, fp32 gradients $\approx 0.41$ GB, Muon momentum on the 84,541,440 2D body matrices $\approx 0.34$ GB, AdamW $m,v$ on the 16,812,288 embedding+norm params $\approx 0.13$ GB. Total $\approx 1.28$ GB — and note it does **not** depend on batch size.
    - **Activations**: $16{,}384 \times 22\ \text{KB} \times 30 \approx 10.8$ GB.
    - **Logits (naive)**: $16{,}384 \times 0.26\ \text{MB} \approx 4.3$ GB — that is the `logits.float()` copy plus `cross_entropy`'s saved `log_softmax`, i.e. $16384 \times 32768 \times 4\ \text{B} = 2.15$ GB twice over.

    Total $\approx 16.4$ GB, which fits in 24 GB but leaves little room for fragmentation, cuBLAS workspaces, or a second eval batch.

    (b) **Activations dominate** at 10.8 GB, but the logits are the *surprising* term: 4.3 GB for a single tensor in a 101M-parameter model whose entire trainable state is 1.28 GB. Chunked fused CE (`loss_chunk=8192`) caps peak logit memory at one chunk, $8192 \times 32768 \times 4 \approx 1.07$ GB, and — crucially — that number no longer grows with batch size. Total drops to $\approx 13.2$ GB. Liger-Kernel's `LigerFusedLinearCrossEntropy` or Cut Cross-Entropy push it lower still by never writing a logit tensor at all.

    (c) **Activation checkpointing** on each `Block`. Instead of ~22 KB/token/layer you store only the block *input*, $512 \times 2\ \text{B} = 1$ KB/token/layer, i.e. 30 KB/token total ($\approx 0.5$ GB at this micro-batch) plus one block's worth of recompute scratch. Activations fall from 10.8 GB to under 1 GB, easily funding a 2–4x larger micro-batch. The cost is roughly **+30% step time** (one extra forward per block during backward), and the right way to spend it is to *keep the effective batch fixed* — halve the gradient-accumulation count as you double the micro-batch, so the 0.5M-token step from PLAN §5 is unchanged. See [Memory-Efficient Training: Checkpointing, Offloading & LoRA Math](../04-kernels-efficiency/10-memory-efficient-training.html).
