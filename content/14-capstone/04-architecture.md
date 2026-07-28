# 14.4 The Stack-100M Architecture: SOTA Components, Cited and Assembled

Every design decision in a modern language model is a small bet, made under a budget, against a specific failure mode. A shallow-and-wide model wastes depth. An un-normalized query-key product blows up your attention logits at high learning rate. A 50k-token vocabulary eats a quarter of a 100M-parameter model before you have written a single transformer block. This chapter is where we stop surveying the frontier and *commit*: we assemble the exact `Stack-100M` architecture — the one every other capstone chapter trains, aligns, quantizes, and serves — from named, cited, 2024–2026 state-of-the-art components, and we write clean from-scratch PyTorch for each.

The canonical configuration is fixed in the capstone spec (`capstone/PLAN.md`, §1) and reproduced here. Our job is not to invent numbers; it is to *justify every one of them from first principles*, implement it, and reproduce the parameter count to the megabyte. This chapter builds directly on the survey in [Modern Architecture Improvements & Design Choices](../02-transformer/10-modern-arch-improvements.html); on the block anatomy in [The Transformer Block: Norms, Residuals, MLPs & Activations](../02-transformer/06-transformer-block.html); on GQA/MLA in [Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html); and on rotary embeddings in [Positional Encodings: Sinusoidal, Learned, RoPE & ALiBi](../02-transformer/05-positional-encoding.html). Where those chapters derive a mechanism in full, we cite them and move on; here we make choices and wire them together into one runnable `stacklm.model` module.

---

## Design philosophy: why deep-and-thin at 100M

Before a single line of code, one macro-decision governs the whole design: the *aspect ratio* of the network — how depth (`n_layers`) trades against width (`d_model`) at a fixed parameter budget.

The parameter count of the transformer body is dominated, per layer, by four attention projections ($\approx 4 d^2$ when heads are square) and the SwiGLU MLP ($3 d \cdot d_{\text{ff}}$, with $d_{\text{ff}} \approx 2.75 d$). So per-layer params scale as $\approx d^2 (4 + 3 \cdot 2.75) \approx 12.25\, d^2$, and the total body is $\approx 12.25\, d^2 \cdot L$. For a *fixed* budget $P$, you can spend it on a large $d$ with few layers, or a modest $d$ with many layers — the product $d^2 L$ is what is pinned. Doubling $d$ at fixed $P$ costs you three-quarters of your layers; doubling $L$ costs you a factor of $\sqrt 2$ off your width. The question is which of those two currencies buys more capability per unit.

At large scale (billions of parameters) they trade roughly evenly and width tends to win on hardware efficiency (wide matmuls are more FLOP-efficient and expose more parallelism). But **at small scale the evidence flips decisively toward depth.** Liu et al., *MobileLLM* (2024), ran this exact ablation for sub-billion models and found that, at fixed parameters, *deeper-and-thinner* models are consistently more accurate — a 30-ish-layer 125M model beats a shallow 12-layer one of the same size on commonsense reasoning. The intuition: capability at small scale is bottlenecked by the number of *sequential nonlinear transformations* the residual stream can undergo (the model's "reasoning depth"), not by the dimensionality of each transformation. A wide-but-shallow model can store a lot per token but can only compose a few steps of computation before emitting; a deep-but-narrow model composes many. Recent small models agree in practice: Qwen3-0.6B, the GLM small models, and SmolLM all lean deep-and-thin.

So we fix an aggressive aspect ratio: **`d_model = 512`, `n_layers = 30`** — 30 sequential blocks of a narrow 512-wide stream. The conventional shape metric is *width-per-layer*, $d_{\text{model}}/n_{\text{layers}}$: for Stack-100M that is $512/30 \approx 17$, versus $\approx 64$ for GPT-2-small ($768/12$). Stack-100M is nearly **4× thinner per layer** than a classic GPT-2 of comparable size — a deliberately extreme point on the deep-thin axis, chosen because the sub-billion evidence rewards it. We do pay for the depth: 30 layers is 30 sequential kernel launches per token in the decode path, a latency cost we accept because (a) at 100M the matrices are tiny and launch overhead dominates anyway, and (b) `torch.compile` / CUDA-graph capture (see [Kernel Fusion, torch.compile, CUDA Graphs & Compilers](../04-kernels-efficiency/09-compilers-fusion.html)) folds the launches into a single replayable graph. Depth is where a small model's quality lives.

!!! note "Aside: the other small-model lever is vocabulary"

    A second small-model insight, fixed in Ch. 14.3, interacts with this one. With tied embeddings a vocabulary of $V$ tokens costs $V \times d$ parameters. At $V=50{,}257$ (GPT-2) and $d=512$ that is 25.7M parameters — on the order of **a quarter of the whole model** spent on the lookup table. We choose $V = 32768$, costing 16.78M. Vocabulary size is a first-class architectural knob at 100M in a way it simply is not at 100B. The two levers compound: deep-thin keeps the body cheap, a lean vocab keeps the embedding cheap, and the freed budget buys layers.

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
    nope_every: int   = 4       # every 4th layer is NoPE (SmolLM3, 2025)
    norm_eps: float   = 1e-5    # RMSNorm epsilon
    qk_norm: bool     = True    # RMSNorm on Q and K before attention (stability)
    tie_embeddings: bool = True # input embedding == output projection (Press & Wolf, 2017)
    z_loss_coef: float = 1e-4   # penalty on logsumexp(logits)^2 (stability)
    logit_soft_cap: float = 0.0 # 0 disables; Gemma-2 uses 30.0 on final logits
    attn_soft_cap: float  = 0.0 # 0 disables; Gemma-2 uses 50.0 on attention logits

    def head_groups(self) -> int:
        # how many query heads share one KV head
        assert self.n_heads % self.n_kv_heads == 0
        return self.n_heads // self.n_kv_heads
```

Now the arithmetic every reader must be able to reproduce by hand. We count parameters exactly.

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

Attention subtotal: $262144 + 65536 + 65536 + 262144 = 655{,}360 \approx 0.655\text{M}$. Note how GQA shrinks $W_K, W_V$ by 4×: full multi-head attention (8 KV heads) would make them $512 \times 512$ each, adding $\approx 0.39$M *per layer* — $\approx 12$M over 30 layers, more than two extra layers' worth, for quality gains that are negligible at this width.

The SwiGLU MLP has three matrices (gate, up, down), each $d \times d_{\text{ff}}$ up to transpose, with $d_{\text{ff}} = 1408$:

$$
3 \times 512 \times 1408 = 2{,}162{,}688 \approx 2.163\text{M}.
$$

Norms are negligible: two RMSNorm weight vectors of length $d$ ($2\times512$) for the pre-attention and pre-MLP norms, plus (with QK-norm on) two more of length $d_h$ ($2\times64$), totaling 1152 parameters per block. Rounding them into the noise, each block is

$$
655{,}360 + 2{,}162{,}688 + 1152 \approx 2{,}819{,}200 \approx 2.82\text{M}.
$$

**Full model.** Thirty blocks plus one final RMSNorm ($512$ params), plus the tied embedding. Carrying the norms exactly ($30 \times 1152 = 34{,}560$ for QK+block norms, already folded into the block total) the body is $30 \times 2{,}818{,}048 + 34{,}560 = 84{,}575{,}\text{-ish}$; rounding the per-block norms into the noise as the spec does gives the headline arithmetic:

$$
16{,}777{,}216 + 30 \times 2{,}818{,}048 + 512 = 16{,}777{,}216 + 84{,}541{,}440 + 512 = 101{,}319{,}168.
$$

$$
\boxed{\;\approx 101.3\text{M parameters}\;}
$$

matching the spec's "$\approx 101$M." Roughly 83% of the parameters live in the 30-layer body and 17% in the embedding — exactly the balance the deep-thin + lean-vocab choices were designed to produce. (The `num_params()` sanity check at the end of the chapter reports the exact integer including every norm vector; the hand arithmetic above rounds the norms into the noise, which is why they differ by a few thousand.)

!!! example "Worked example: where do the FLOPs and the KV cache go?"

    Two magnitudes a practitioner should be able to estimate on the spot.

    **Training FLOPs (the 6ND rule).** A forward+backward step costs on the order of $6 N D$ FLOPs for $N$ non-embedding parameters over $D$ tokens (see [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html)). With $N \approx 84.5$M (body params) and the capstone budget $D \approx 20$B tokens:
    $$
    6 \times 84.5\times10^6 \times 20\times10^9 \approx 1.0 \times 10^{19}\ \text{FLOPs}.
    $$
    On a single A100 at, say, 40% of its ~312 bf16 TFLOP/s ($\approx 1.25\times10^{14}$ eff. FLOP/s), that is $\approx 8\times10^4$ s $\approx 22$ GPU-hours — squarely in the "\$40–\$100, 15–25 GPU-hr" flagship envelope from PLAN §0. The arithmetic *predicts the budget*: this is not a guess, it is $6ND$ divided by realized throughput.

    **KV cache per token.** GQA stores $K,V$ for $n_{kv}=2$ heads of $d_h=64$ across $L=30$ layers, in bf16 (2 bytes), for both K and V:
    $$
    2\ (\text{K,V}) \times 2\ (\text{KV heads}) \times 64 \times 30 \times 2\ \text{bytes} = 30{,}720\ \text{bytes} \approx 30\ \text{KB/token}.
    $$
    A full 2048-token context is $\approx 60$ MB. Under full multi-head attention (8 KV heads) it would be $\approx 240$ MB — the 4× GQA win, made concrete. This is the number the serving chapter (14.11) and [PagedAttention & KV-Cache Memory Management](../04-kernels-efficiency/06-paged-attention-kv.html) care about.

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

### QK-norm: the high-learning-rate stabilizer

The single most common way a small model at an aggressive learning rate diverges is an **attention-logit blowup**: the dot product $q \cdot k$ grows without bound, softmax saturates to a near one-hot distribution, the gradient into $W_Q, W_K$ collapses, and the loss spikes irrecoverably (the failure mode dissected in [Training Stability, Loss Spikes & Debugging Large Runs](../03-pretraining/11-training-stability.html)). **QK-norm** — applying RMSNorm to the query and key vectors *per head* before the dot product — bounds the geometry of that product. It normalizes $q$ and $k$ to a controlled scale so that $q \cdot k$ cannot run away regardless of how large the raw projections drift.

This trick (query-key normalization, Henry et al., *Query-Key Normalization for Transformers*, 2020, revived by many 2024–2025 open models: OLMo 2, SmolLM3, and the Muon-trained Kimi K2 which pairs it with QK-clip) is what lets us train `Stack-100M` with Muon at a learning rate that would otherwise diverge — the optimizer story is Ch. 14.6, and it leans on [Optimizers: SGD, Adam, Adafactor, Lion, Muon & Shampoo](../03-pretraining/09-optimizers.html). We apply the norm over the `head_dim` axis, before RoPE.

!!! warning "Ablate to learn: turn QK-norm off and watch it die"

    A genuinely instructive ablation. Train the config at the target LR (e.g. peak $\approx 3\text{e-}3$ for the Muon path) with `qk_norm=False`. On the order of a few hundred to a few thousand steps in, you will typically see the attention logits' max magnitude climb past ~50, softmax entropy collapse, and the loss NaN out. Re-enable QK-norm, same LR, and the run is stable. The lesson is causal: QK-norm is not decoration, it is what *buys* the high learning rate that makes the run cheap. Its cost is 2 tiny RMSNorm vectors per layer (128 params) — free.

!!! example "Worked example: how QK-norm bounds the attention logits"

    Make the mechanism quantitative. Without QK-norm, the pre-softmax logit for a query-key pair is $q\cdot k = \sum_{i=1}^{d_h} q_i k_i$. If $q$ and $k$ have per-component standard deviation $\sigma$ (independent, zero mean), each product $q_i k_i$ has variance $\sigma^4$, so the dot product over $d_h=64$ dimensions has standard deviation $\approx \sigma^2\sqrt{d_h} = 8\sigma^2$, and after the $1/\sqrt{d_h}=1/8$ scaling, $\approx \sigma^2$. During training $\sigma$ can drift upward as $W_Q, W_K$ grow; if $\sigma$ reaches, say, 6, the scaled logit std is $\approx 36$, and the *max* logit over a 2048-token context (a few std out) can exceed 100. A softmax with a gap of ~100 between the top logit and the rest is numerically a hard argmax: its gradient with respect to the losing logits is $\approx e^{-100}\approx 0$ — the attention pattern is frozen, learning stalls, and one bad step NaNs the run.

    **With QK-norm**, each $q$ and $k$ is RMS-normalized before the dot product, so $\|q\|,\|k\|\approx\sqrt{d_h}=8$ regardless of how large the raw projections grow, and by Cauchy–Schwarz $|q\cdot k|\le \|q\|\,\|k\| = 64$; after $1/\sqrt{d_h}$ scaling the logit is bounded by $\pm 8$ by construction, with a *learned* temperature reintroduced through the RMSNorm $\gamma$. The blow-up is structurally impossible. This is why QK-norm, not merely a lower learning rate, is the right fix: it removes the failure mode instead of tiptoeing around it.

### z-loss and logit soft-cap

Two more cheap stabilizers guard the *output* logits. The **z-loss** (introduced in the PaLM / T5X training recipes) adds a small penalty on the log-partition function of the softmax:

$$
\mathcal{L}_z = \lambda_z \,\big(\operatorname{logsumexp}(\text{logits})\big)^2,
$$

which gently pulls $\log \sum_j e^{z_j}$ toward zero, keeping logits from drifting to large absolute values and keeping the softmax well-conditioned in bf16. We use $\lambda_z = 10^{-4}$. Optionally, a **logit soft-cap** (Gemma-2 style) squashes logits through a scaled tanh, $z \leftarrow c \cdot \tanh(z / c)$, hard-bounding them to $(-c, c)$; we leave it off by default (`logit_soft_cap=0`) because z-loss usually suffices and soft-cap complicates KV-cache-friendly and FlashAttention kernels, but we expose it. The pretraining-objective chapter, [The Pretraining Objective & Loss](../03-pretraining/03-pretraining-objective.html), treats z-loss in full.

---

## Positional information: RoPE, and NoPE on every 4th layer

### RoPE

`Stack-100M` encodes position with **Rotary Position Embeddings** (Su et al., *RoFormer*, 2021): rather than adding a position vector, RoPE *rotates* each 2-dimensional slice of the query and key by an angle proportional to the absolute position, so that the attention dot product depends only on the *relative* offset $m - n$. With base $\theta = 10000$ and per-pair frequencies $\omega_i = \theta^{-2i/d_h}$, the query at position $m$ is rotated by $m\omega_i$ in the $i$-th plane. The full derivation lives in [Positional Encodings](../02-transformer/05-positional-encoding.html); here we implement it and, crucially, keep `rope_theta` as a config knob because mid-training (Ch. 14.8) *rescales* it to extend context from 2048 to 8192 (see [Long-Context Pretraining & Context Extension](../03-pretraining/13-long-context-pretraining.html)).

```python
def build_rope_cache(head_dim: int, max_seq: int, theta: float, device=None):
    """Precompute cos/sin tables of shape (max_seq, head_dim)."""
    # frequencies for each PAIR of dims: theta^(-2i/head_dim), i=0..head_dim/2-1
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq, device=device).float()          # positions
    freqs = torch.outer(t, inv_freq)                          # (max_seq, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)                   # (max_seq, head_dim)
    return emb.cos(), emb.sin()

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos, sin):
    # q, k: (B, n_heads, T, head_dim); cos, sin: (T, head_dim) -> broadcast over B, heads
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot
```

This is the "GPT-NeoX / HuggingFace" RoPE layout, where the two halves of the head vector form the rotation pairs (index $i$ pairs with $i + d_h/2$); `emb = cat(freqs, freqs)` and `rotate_half` are matched to that layout, so the code is self-consistent. Applied to $q$ of shape `(B, 8, T, 64)` and $k$ of shape `(B, 2, T, 64)`, the `(T, 64)` tables broadcast across the batch and head axes identically — RoPE is applied to the *narrow* GQA key tensor **before** the KV heads are expanded, which is both correct (rotation is per-position, head-count-agnostic) and cheap.

### NoPE on every 4th layer

Here is a genuinely modern choice. We do **not** apply RoPE on every layer. Following **SmolLM3** (HuggingFace, 2025) and the analysis of Kazemnejad et al. (*The Impact of Positional Encoding on Length Generalization in Transformers*, "NoPE", NeurIPS 2023), **every 4th layer uses no positional encoding at all** — "NoPE."

The reasoning: a decoder-only model with a causal mask can *infer* absolute position from the mask alone (the number of tokens a position can attend to is itself a positional signal), so it does not strictly need an injected encoding on every layer. Layers that are freed from RoPE are not tied to the specific rotation frequencies seen during training, and empirically the mixture generalizes to *longer* sequences than training — precisely the property we want for the 2048→8192 context extension in mid-training. Keeping RoPE on the majority of layers preserves the strong local-position inductive bias that makes short-context modeling sample-efficient; interleaving NoPE layers buys length robustness. SmolLM3 reports this hybrid as an ingredient of its long-context behavior.

Concretely, with `nope_every = 4`, layer index $\ell$ (0-based) uses NoPE iff $(\ell + 1) \bmod 4 = 0$ — layers 3, 7, 11, …, 27, i.e. **7 of the 30 layers**; the other 23 keep RoPE.

!!! note "Aside: NoPE is not free lunch, it is a mixture"

    A pure-NoPE model tends to underperform on *short* context — the causal-mask position signal is weak and diffuse compared to RoPE's sharp relative encoding. That is why we interleave rather than remove. The design point is: RoPE for local precision on most layers, a minority of NoPE layers for length extrapolation. The 1-in-4 ratio is SmolLM3's; treat it as a tuned constant, not a law.

---

## Attention: grouped-query attention with QK-norm

Now we assemble the attention module: **GQA with 2 KV heads** (Ainslie et al., *GQA*, 2023), **QK-norm** on the per-head queries and keys, RoPE (or NoPE) applied conditionally, and a causal mask via PyTorch's fused scaled-dot-product attention (which dispatches to a FlashAttention kernel when available — see [FlashAttention I](../04-kernels-efficiency/02-flash-attention-1.html)).

GQA is the middle ground between full multi-head attention (one KV head per query head — maximal quality, maximal KV cache) and multi-query attention (one KV head total — minimal cache, some quality loss). With 8 query heads sharing 2 KV heads (a 4:1 group), we cut the KV cache 4× versus MHA while retaining nearly all the quality; the mechanism and quality tradeoff are dissected in [Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html).

```python
class Attention(nn.Module):
    def __init__(self, cfg: StackConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.n_heads = cfg.n_heads
        self.n_kv = cfg.n_kv_heads
        self.d_h = cfg.head_dim
        self.groups = cfg.head_groups()          # query heads per KV head (=4)
        # this layer is NoPE if it is every-4th (SmolLM3): (idx+1) % nope_every == 0
        self.use_rope = ((layer_idx + 1) % cfg.nope_every) != 0

        # projections; no biases (modern default)
        self.wq = nn.Linear(cfg.d_model, self.n_heads * self.d_h, bias=False)
        self.wk = nn.Linear(cfg.d_model, self.n_kv    * self.d_h, bias=False)
        self.wv = nn.Linear(cfg.d_model, self.n_kv    * self.d_h, bias=False)
        self.wo = nn.Linear(self.n_heads * self.d_h, cfg.d_model, bias=False)

        # QK-norm: RMSNorm over head_dim, applied to q and k per head
        self.q_norm = RMSNorm(self.d_h, cfg.norm_eps) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.d_h, cfg.norm_eps) if cfg.qk_norm else nn.Identity()
        self.attn_soft_cap = cfg.attn_soft_cap

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        # project and reshape to (B, heads, T, head_dim)
        q = self.wq(x).view(B, T, self.n_heads, self.d_h).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv,    self.d_h).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv,    self.d_h).transpose(1, 2)

        # QK-norm BEFORE RoPE: bound the geometry of q·k (stability at high LR)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # conditional positional encoding: RoPE on most layers, NoPE on every 4th
        if self.use_rope:
            q, k = apply_rope(q, k, cos, sin)

        # GQA: expand the 2 KV heads to 8 by repeating each group `groups` times
        k = k.repeat_interleave(self.groups, dim=1)   # (B, n_heads, T, d_h)
        v = v.repeat_interleave(self.groups, dim=1)

        if self.attn_soft_cap and self.attn_soft_cap > 0:
            # Gemma-2 style: manual attention so we can tanh-cap the logits
            scale = 1.0 / (self.d_h ** 0.5)
            att = (q @ k.transpose(-2, -1)) * scale
            cap = self.attn_soft_cap
            att = cap * torch.tanh(att / cap)
            mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), 1)
            att = (att + mask).softmax(dim=-1)
            out = att @ v
        else:
            # fast path: fused SDPA (FlashAttention kernel when available), causal mask
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, n_heads*d_h)
        return self.wo(out)
```

Three implementation notes worth internalizing. First, **QK-norm is applied before RoPE**, not after: we normalize the *content* geometry of $q$ and $k$, then inject position. RoPE is norm-preserving (a rotation leaves $\|q\|$ unchanged), so the choice of order does not change the *magnitude* bound — but it does change how the learned per-dimension scale $\gamma$ interacts with the rotation. Normalizing first keeps $\gamma$ acting in the unrotated content frame, which is the standard order (OLMo 2, SmolLM3). Second, we expand KV heads with `repeat_interleave` for pedagogical clarity — a production kernel (FlashAttention's GQA path, vLLM) never materializes the expansion, it reads the shared KV head directly, which is the entire point of the memory saving; in training with `is_causal=True` SDPA the expansion is cheap and the kernel still fuses. Third, the fast path uses SDPA's default scale $1/\sqrt{d_h}$; the manual soft-cap path reproduces that same scale explicitly so the two paths are numerically comparable when `attn_soft_cap=0`.

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

---

## Assembling the block and the full model

A `Block` is pre-norm attention plus pre-norm SwiGLU, each wrapped in a residual add. The full model is an embedding, 30 blocks, a final norm, and a tied output projection, with the forward pass computing cross-entropy plus z-loss. Here is the data flow through one block and the whole stack:

```text
  tokens ─► Embedding (32768×512, tied) ─► x  (B,T,512)
                                            │
        ┌───────────────────────────────────┤  ×30 blocks
        │   x ──► RMSNorm ──► Attention ──►(+)      Attention:
        │   │                              │          Q:512→512  K,V:512→128 (GQA 2 KV)
        │   └──────────────residual────────┘          QK-norm ► RoPE* ► SDPA(causal) ► O
        │   x ──► RMSNorm ──► SwiGLU ────►(+)      SwiGLU: gate,up:512→1408; down:1408→512
        │   │                              │       * layer (ℓ+1)%4==0 ► NoPE (no RoPE)
        └──────────────residual────────────┘
                                            │
                          x ──► RMSNorm ──► lm_head (tied, 512→32768) ──► logits
                                            └► CE loss + z-loss  (both in fp32)
```

```python
class Block(nn.Module):
    def __init__(self, cfg: StackConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg, layer_idx)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)   # pre-norm attention residual
        x = x + self.mlp(self.mlp_norm(x))               # pre-norm MLP residual
        return x


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

        # RoPE cos/sin cache (registered as buffers; rebuilt if seq_len grows in mid-training)
        cos, sin = build_rope_cache(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # scale residual-projection inits by 1/sqrt(2*n_layers) (GPT-2 trick) for deep stability
        scale = (2 * cfg.n_layers) ** -0.5
        for blk in self.blocks:
            torch.nn.init.normal_(blk.attn.wo.weight, mean=0.0, std=0.02 * scale)
            torch.nn.init.normal_(blk.mlp.down.weight, mean=0.0, std=0.02 * scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        cos = self.rope_cos[:T].to(idx.device)
        sin = self.rope_sin[:T].to(idx.device)
        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.final_norm(x)

        if targets is None:
            # inference: only compute logits for the last position (decode)
            logits = self.lm_head(x[:, -1:, :])
            return logits, None

        logits = self.lm_head(x)                      # (B, T, vocab)
        if self.cfg.logit_soft_cap > 0:               # optional Gemma-2 final cap
            c = self.cfg.logit_soft_cap
            logits = c * torch.tanh(logits / c)

        # cross-entropy in fp32 for stability
        ce = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
        )
        # z-loss: penalize large logsumexp (keeps softmax well-conditioned)
        logz = torch.logsumexp(logits.float(), dim=-1)      # (B, T)
        z_loss = self.cfg.z_loss_coef * (logz ** 2).mean()
        return logits, ce + z_loss

    @torch.no_grad()
    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and self.cfg.tie_embeddings:
            n -= self.tok_emb.weight.numel()   # tied head shares the same tensor
        return n
```

A few points that make this *correct* and not just plausible. The residual-projection initializations (`wo`, `down`) are scaled by $1/\sqrt{2L}$ — the GPT-2 trick — so that the variance added into the residual stream does not grow with depth. Each layer contributes two residual writes (attention, MLP), so after $L$ layers the stream has accumulated $2L$ writes; scaling each output projection's init std by $1/\sqrt{2L}$ keeps the accumulated variance $O(1)$ instead of $O(L)$. With 30 layers this matters, and skipping it is a common cause of early-training instability in deep-thin models. Note also that because `lm_head.weight` is *the same tensor* as `tok_emb.weight`, `self.apply(self._init_weights)` initializing it twice (once as an `Embedding`, once as a `Linear`) is harmless — both draw from the same $\mathcal N(0, 0.02^2)$, and the tie is a shared reference, not a copy, so the two stay identical through training. Cross-entropy and z-loss are computed in fp32 even under bf16 autocast. And because the head is tied, `num_params(non_embedding=True)` correctly subtracts the shared tensor once to recover the ~84.5M body count used in the 6ND estimate.

```python
# sanity check: reproduce the parameter count
if __name__ == "__main__":
    m = Stack100M(StackConfig())
    print(f"total params      : {m.num_params():,}")            # ~101,319,168
    print(f"non-embedding body: {m.num_params(True):,}")        # ~84,541,952
```

!!! interview "Interview Corner"

    **Q:** At 100M parameters you chose GQA with 2 KV heads, tied embeddings, and a 32k vocab. Walk me through *why each of those specific numbers* rather than the "obvious" MHA / untied / 50k defaults, and what you'd lose if you flipped them.

    **A:** All three are driven by the fact that at 100M the parameter budget is dominated by a few big tensors, so cheap-but-good choices compound. (1) **GQA 2 KV heads**: full MHA would add ~0.39M params/layer and, more importantly, 4× the KV cache (~30 KB/token → ~120 KB/token) for a quality gain that is negligible at this width; 2 KV heads keeps the model within a hair of MHA quality while cutting cache 4×, which is what makes it cheap to *serve*. (2) **Tied embeddings**: the input embedding and output projection are both $V\times d$; tying saves 16.8M params — about a sixth of the model — and at small scale the shared representation actually helps, not hurts (Press & Wolf). Untying would spend a sixth of the model on a redundant table. (3) **32k vocab**: a 50k vocab costs 25.7M tied params vs 16.8M; the ~9M saved buys roughly three more transformer layers, which at small scale (deep-thin, MobileLLM) is a better use of the budget than finer-grained tokenization. Flip any of them and you either bloat the serving footprint (MHA), waste a sixth of the model (untying), or trade depth for vocab granularity (50k) — all bad trades *specifically because 100M is small*. At 100B these trades look very different: the embedding is a rounding error, MHA's cache is amortized over far more compute, and a bigger vocab pays off on multilingual coverage.

---

## Efficiency variants: options you can bolt on, cited and implemented

The default `Stack-100M` above is what the rest of the capstone trains. But part of the pedagogical point is to show *where the frontier is going* and give runnable code for the three most important efficiency moves — each an option, each cited, none the default path.

### MLA — Multi-head Latent Attention (DeepSeek-V2)

GQA shrinks the KV cache by *sharing* KV heads. **MLA** (DeepSeek-V2, 2024) shrinks it by *compressing*: it projects the KV information down to a small **latent** vector $c^{KV}$ of dimension $d_c \ll n_h d_h$, caches only that latent, and up-projects to per-head K and V on the fly. The KV cache stores $d_c$ numbers per token instead of $2 n_{kv} d_h$. To keep RoPE compatible (rotations do not commute with a low-rank projection, so a naively compressed key cannot carry position), MLA uses a **decoupled** design: a small separate RoPE-carrying key dimension is concatenated to the up-projected content key. DeepSeek reports MLA matching or beating MHA quality at a fraction of the KV footprint; the mechanism is dissected in [Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html).

```python
class MLAttention(nn.Module):
    """Multi-head Latent Attention (DeepSeek-V2, 2024), compact form.
    Caches only a d_c latent + a small decoupled RoPE key, instead of full KV."""
    def __init__(self, cfg: StackConfig, d_c: int = 128, d_rope: int = 32):
        super().__init__()
        self.n_heads, self.d_h = cfg.n_heads, cfg.head_dim
        self.d_c, self.d_rope = d_c, d_rope          # latent dim, decoupled RoPE dim
        d = cfg.d_model
        # queries: full per-head (optionally also low-rank; kept simple here)
        self.wq = nn.Linear(d, self.n_heads * self.d_h, bias=False)
        # KV down-projection to the cached latent  (this is what the cache stores)
        self.w_dkv = nn.Linear(d, d_c, bias=False)
        # up-projections from latent to content K and V (NOT cached; recomputed)
        self.w_uk = nn.Linear(d_c, self.n_heads * self.d_h, bias=False)
        self.w_uv = nn.Linear(d_c, self.n_heads * self.d_h, bias=False)
        # decoupled RoPE key: a small shared key that carries position (cached, tiny)
        self.w_kr = nn.Linear(d, self.d_rope, bias=False)
        self.wo = nn.Linear(self.n_heads * self.d_h, d, bias=False)

    def forward(self, x, cos_r, sin_r):
        B, T, _ = x.shape
        c_kv = self.w_dkv(x)                                   # (B,T,d_c) <-- cache THIS
        k = self.w_uk(c_kv).view(B, T, self.n_heads, self.d_h).transpose(1, 2)
        v = self.w_uv(c_kv).view(B, T, self.n_heads, self.d_h).transpose(1, 2)
        q = self.wq(x).view(B, T, self.n_heads, self.d_h).transpose(1, 2)
        # decoupled RoPE branch: rotate a small shared key + a slice of each query
        k_r = self.w_kr(x).view(B, T, 1, self.d_rope).transpose(1, 2)  # (B,1,T,d_rope)
        # (a faithful impl rotates k_r and a matching query slice, then concatenates
        #  along head_dim so position rides a separate un-compressed channel;
        #  omitted for brevity — see Ch. 2.4). Here: content attention only.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(out.transpose(1, 2).contiguous().view(B, T, -1))
```

At 100M the KV cache is already tiny (~60 MB at full context), so MLA's *memory* win is not the reason to reach for it here — GQA is plenty. The reason to teach it is that MLA is the KV strategy of the strongest current open models, and swapping `Attention` for `MLAttention` in `Block` is a one-line change if you want to study it. For a model you intend to serve at long context or high concurrency, MLA is the upgrade.

### MTP — Multi-Token Prediction (DeepSeek-V3)

Standard training predicts token $t{+}1$ from position $t$. **Multi-Token Prediction** (DeepSeek-V3, 2024; Gloeckle et al., *Better & Faster Large Language Models via Multi-token Prediction*, 2024) adds an auxiliary head that also predicts token $t{+}2$ (or further), giving a **denser training signal** — every position now contributes two prediction losses — which speeds convergence and, as a bonus, yields a natural draft head for self-speculative decoding at inference (see [Speculative Decoding](../07-inference-serving/06-speculative-decoding.html)). The auxiliary head is a small transformer module that reuses the main trunk's hidden state; its loss is added with a small weight and it is *discarded at inference* unless used for speculation.

```python
class MTPHead(nn.Module):
    """Auxiliary next-2 token head (DeepSeek-V3 / Gloeckle et al., 2024).
    A one-block predictor on top of the trunk hidden state, tied to lm_head."""
    def __init__(self, cfg: StackConfig, lm_head: nn.Linear):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.block = Block(cfg, layer_idx=0)     # one extra transformer block
        self.lm_head = lm_head                   # SHARE the tied output projection

    def forward(self, h, cos, sin, targets_plus2):
        # h: trunk hidden state (B,T,d). Predict token t+2.
        h2 = self.block(self.norm(h), cos, sin)
        logits2 = self.lm_head(h2)
        return F.cross_entropy(
            logits2.float().view(-1, logits2.size(-1)),
            targets_plus2.view(-1), ignore_index=-100,
        )

# in the training step: total = main_loss + lambda_mtp * mtp_loss   (e.g. lambda_mtp=0.3)
```

Used in the capstone's ablation menu (Ch. 14.5/14.7): turning MTP on with $\lambda \approx 0.3$ typically buys faster early loss descent for a modest compute overhead, and you keep the head around only if you later want self-speculation.

### A Liquid LFM2-style gated short-conv hybrid block

Attention is not the only sequence mixer. **Liquid AI's LFM2** (2025) interleaves attention with cheap **gated short-range convolution** blocks — a depthwise causal conv (kernel ~3) wrapped in multiplicative input/output gates — which capture local structure at a fraction of attention's cost and carry no KV cache at all. This is a concrete instance of the broader hybrid-architecture trend covered in [Beyond Attention: SSMs, Mamba, RWKV & Linear Attention](../02-transformer/11-ssm-and-alternatives.html). Replacing some of `Stack-100M`'s attention blocks with conv-mixer blocks trades a little long-range capacity for real decode-time speed.

```python
class GatedShortConv(nn.Module):
    """LFM2-style (Liquid AI, 2025) double-gated causal short convolution mixer.
    No KV cache; O(T) local mixing. A cheap alternative to an attention block."""
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

    def forward(self, x):
        B, T, d = x.shape
        z = self.in_b(x) * self.in_x(x)                 # input gate * value
        z = z.transpose(1, 2)                           # (B, d, T) for conv1d
        z = F.pad(z, (self.k - 1, 0))                   # left-pad => causal
        z = self.conv(z).transpose(1, 2)                # (B, T, d)
        return self.out(self.in_c(x) * z)               # output gate
```

The design point (LFM2's own recipe): use *mostly* conv blocks with a *few* attention blocks for long-range routing. At 100M this is a research option, not the default — our narrow auto-research agent (Ch. 14.10) benefits more from attention's exact retrieval than from conv's speed — but the code is here so a reader can build the hybrid and measure the tradeoff.

!!! tip "Practitioner tip: keep variants behind the config, never fork the model"

    Every variant above is a *drop-in module* with the same `(x, cos, sin) -> x` (or `x -> x`) contract as the default `Attention`/`Block`. The right engineering pattern is a factory: `Block` reads `cfg.mixer in {"gqa","mla","conv"}` and constructs the right mixer, so you can ablate architectures by changing one config field and re-running the *same* training loop from Ch. 14.7. Forking `model.py` per variant is how capstones rot. One model, config-selected mixers.

---

## Key Takeaways

!!! key "Key Takeaways"

    - **Deep-and-thin is the small-model bet**: at fixed 100M params, 30 layers × 512 width beats shallow-wide (MobileLLM, 2024); depth = sequential reasoning steps, which is what small models are bottlenecked on. The width-per-layer ratio $d/L \approx 17$ is ~4× thinner than GPT-2-small's ~64.
    - **The exact config is frozen** (PLAN §1): `vocab 32768, d_model 512, n_layers 30, 8 heads / 2 KV heads, head_dim 64, SwiGLU 1408, tied embeddings` → **≈101.3M params**, of which ~84.5M is the trainable body and ~16.8M the tied embedding — reproduce this arithmetic by hand.
    - **GQA (2 KV heads)** cuts the KV cache 4× (~30 KB/token vs ~120 KB) at negligible quality cost; **tied embeddings** save ~16% of the model; **a 32k vocab** frees ~9M params (≈3 layers) versus 50k — small-scale budget choices that compound.
    - **Stability is engineered, not hoped for**: pre-norm + RMSNorm (fp32), **QK-norm** to bound attention logits (what buys the high Muon learning rate), **z-loss** on the log-partition, optional Gemma-2 soft-caps, and $1/\sqrt{2L}$ residual-init scaling for the deep stack.
    - **RoPE + NoPE-every-4th-layer** (SmolLM3, 2025; Kazemnejad et al., 2023): RoPE for local positional precision on 23 layers, 7 NoPE layers for length extrapolation — the ingredient behind the 2048→8192 context extension in mid-training.
    - **Efficiency variants are options, not the default**: MLA (DeepSeek-V2) compresses the KV cache to a latent; MTP (DeepSeek-V3 / Gloeckle et al.) adds a next-2-token head for denser signal + free draft head; an LFM2-style gated short-conv block trades long-range capacity for KV-free decode speed.
    - **One model, config-selected mixers**: keep every variant behind the same module contract so the Ch. 14.7 training loop and Ch. 14.5 scaling ladder run unchanged — coherence across the capstone is the whole point.

## Further reading

- Liu et al., *MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases*, 2024 — the deep-and-thin evidence.
- Zhang & Sennrich, *Root Mean Square Layer Normalization*, NeurIPS 2019 — RMSNorm.
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021 — RoPE.
- Kazemnejad et al., *The Impact of Positional Encoding on Length Generalization in Transformers* ("NoPE"), NeurIPS 2023; and the SmolLM3 model report, HuggingFace, 2025 — RoPE/NoPE interleaving.
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, 2023 — grouped-query attention.
- Henry et al., *Query-Key Normalization for Transformers*, 2020 — QK-norm; revived by OLMo 2 (2024) and Kimi K2 (Moonshot, 2025).
- Shazeer, *GLU Variants Improve Transformer*, 2020 — SwiGLU.
- Press & Wolf, *Using the Output Embedding to Tie Word Vectors*, 2017 — tied embeddings.
- DeepSeek-AI, *DeepSeek-V2* (2024, MLA) and *DeepSeek-V3* (2024, MTP); Gloeckle et al., *Better & Faster Large Language Models via Multi-token Prediction*, 2024.
- Liquid AI, *LFM2* technical report, 2025 — gated short-convolution hybrid blocks.
- Gemma Team, *Gemma 2*, 2024 — logit soft-capping (final 30.0, attention 50.0) and small-model design choices.
