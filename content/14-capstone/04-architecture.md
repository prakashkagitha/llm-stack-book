# 14.4 The Stack-100M Architecture: SOTA Components, Cited and Assembled

Every design decision in a modern language model is a small bet, made under a budget, against a specific failure mode. A shallow-and-wide model wastes depth. An un-normalized query-key product blows up your attention logits at high learning rate. A 50k-token vocabulary eats a quarter of a 100M-parameter model before you have written a single transformer block. And a `(B, T, V)` logit tensor cast to fp32 outweighs the model's *entire* trainable state by nearly an order of magnitude. This chapter is where we stop surveying the frontier and *commit*: we assemble the exact `Stack-100M` architecture — the one every other capstone chapter trains, aligns, quantizes, and serves — from named, cited, 2024–2026 state-of-the-art components, and we write clean from-scratch PyTorch for each.

The canonical configuration is fixed in the capstone spec (`capstone/PLAN.md`, §1) and reproduced here. Our job is not to invent numbers; it is to *justify every one of them from first principles*, implement it, and reproduce the parameter count to the last norm vector. This chapter builds directly on the survey in [Modern Architecture Improvements & Design Choices](../02-transformer/10-modern-arch-improvements.html); on the block anatomy in [The Transformer Block: Norms, Residuals, MLPs & Activations](../02-transformer/06-transformer-block.html); on GQA/MLA in [Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html); and on rotary embeddings in [Positional Encodings: Sinusoidal, Learned, RoPE & ALiBi](../02-transformer/05-positional-encoding.html). Where those chapters derive a mechanism in full, we cite them and move on; here we make choices and wire them together into one runnable `stacklm.model` module — one that trains with document-aware packing (Ch. 14.2), decodes with a KV cache (Ch. 14.7, 14.10, 14.11), and — with a single config flag flipped — can be handed to the HuggingFace / vLLM / llama.cpp ecosystem (Ch. 14.11) without a rewrite. That flag, and exactly what it costs, is spelled out in the ecosystem section.

---

## Design philosophy: why deep-and-thin at 100M

Before a single line of code, one macro-decision governs the whole design: the *aspect ratio* of the network — how depth (`n_layers`) trades against width (`d_model`) at a fixed parameter budget.

The parameter count of the transformer body is dominated, per layer, by four attention projections and the SwiGLU MLP ($3 d \cdot d_{\text{ff}}$, with $d_{\text{ff}} \approx 2.75 d$). Write per-layer params as $\kappa d^2$: for full multi-head attention $\kappa \approx 4 + 3(2.75) = 12.25$; grouped-query attention with 2 of 8 KV heads shrinks the attention share from $4d^2$ to $2.5d^2$, giving $\kappa = 10.75$ for *our* config. Either way the total body is $\kappa d^2 L$, so for a *fixed* budget $P$ the product $d^2 L$ is what is pinned. Doubling $d$ at fixed $P$ costs you three-quarters of your layers; doubling $L$ costs you a factor of $\sqrt 2$ off your width. The question is which of those two currencies buys more capability per unit.

At large scale (billions of parameters) they trade roughly evenly and width tends to win on hardware efficiency (wide matmuls are more FLOP-efficient and expose more parallelism). But **at small scale the evidence flips decisively toward depth.** Liu et al., *MobileLLM* (2024), ran this exact ablation for sub-billion models and found that, at fixed parameters, *deeper-and-thinner* models are consistently more accurate — a 30-ish-layer 125M model beats a shallow 12-layer one of the same size on commonsense reasoning. The intuition: capability at small scale is bottlenecked by the number of *sequential nonlinear transformations* the residual stream can undergo (the model's "reasoning depth"), not by the dimensionality of each transformation. A wide-but-shallow model can store a lot per token but can only compose a few steps of computation before emitting; a deep-but-narrow model composes many. Recent small models agree in practice: Qwen3-0.6B, the GLM small models, and SmolLM all lean deep-and-thin.

So we fix an aggressive aspect ratio: **`d_model = 512`, `n_layers = 30`** — 30 sequential blocks of a narrow 512-wide stream. The conventional shape metric is *width-per-layer*, $d_{\text{model}}/n_{\text{layers}}$: for Stack-100M that is $512/30 \approx 17$, versus $\approx 64$ for GPT-2-small ($768/12$). Stack-100M is nearly **4× thinner per layer** than a classic GPT-2 of comparable size — a deliberately extreme point on the deep-thin axis, chosen because the sub-billion evidence rewards it. We do pay for the depth: 30 layers is 30 sequential kernel launches per token in the decode path, a latency cost we accept because (a) at 100M the matrices are tiny and launch overhead dominates anyway, and (b) `torch.compile` / CUDA-graph capture (see [Kernel Fusion, torch.compile, CUDA Graphs & Compilers](../04-kernels-efficiency/09-compilers-fusion.html)) folds the launches into a single replayable graph. Depth is where a small model's quality lives.

{{fig:deep-thin-budget-trade}}

!!! note "Aside: the other small-model lever is vocabulary"

    A second small-model insight, fixed in Ch. 14.3, interacts with this one. With tied embeddings a vocabulary of $V$ tokens costs $V \times d$ parameters. At $V=50{,}257$ (GPT-2) and $d=512$ that is 25.7M parameters — on the order of **a quarter of the whole model** spent on the lookup table. We choose $V = 32768$, costing 16.78M. Vocabulary size is a first-class architectural knob at 100M in a way it simply is not at 100B. The two levers compound: deep-thin keeps the body cheap, a lean vocab keeps the embedding cheap, and the freed budget buys layers. The catch — as the memory section below shows — is that $V$ also sets the size of the training logit tensor, which is the largest single tensor in the step once activation checkpointing is on. A lean vocab buys memory headroom as well as parameters.

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
    loss_chunk: int   = 0       # >0 = chunked fused lm_head+CE (see "Budgeting the run")

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

(Cross-check against the aspect-ratio algebra: the 2D matrices alone are $655{,}360 + 2{,}162{,}688 = 2{,}818{,}048 = 10.75 \times 512^2$ — exactly the $\kappa = 10.75$ predicted above for GQA with a 2.75 MLP ratio.)

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

!!! example "Worked example: the honest FLOP budget, and the KV cache"

    Two magnitudes a practitioner should be able to estimate on the spot — and one place where the folklore estimate is wrong by a third.

    **Training FLOPs.** The famous rule is $6ND$ FLOPs for $N$ non-embedding parameters over $D$ tokens (see [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html)). But $6ND$ counts only the *body's parameter matmuls*. Two terms it omits are individually negligible at 7B and jointly enormous at our shape:

    $$
    \frac{\text{FLOPs}}{\text{token}} \;\approx\; \underbrace{6 N_{\text{body}}}_{\text{parameter matmuls}} \;+\; \underbrace{12\,L\,d\,\bar T}_{\text{attention scores + values}} \;+\; \underbrace{6\,V d}_{\text{tied lm\_head}},
    \qquad \bar T = \tfrac{T}{2},
    $$

    where $\bar T$ is the average number of keys a query attends to under a causal mask, so the middle term is $6 L d T$. Plugging in $N_{\text{body}} = 84{,}576{,}512$, $L=30$, $d=512$, $T=2048$, $V=32768$:

    $$
    \begin{aligned}
    6N_{\text{body}} &= 5.07\times10^{8} \quad(\text{100\%, the reference}),\\
    6\,L\,d\,T &= 6 \cdot 30 \cdot 512 \cdot 2048 = 1.89\times10^{8} \quad(+37\%),\\
    6\,V d &= 6 \cdot 32768 \cdot 512 = 1.01\times10^{8} \quad(+20\%),\\
    \text{total} &= 7.97\times10^{8}\ \text{FLOP/token}.
    \end{aligned}
    $$

    The attention-to-parameter ratio is $T/(\kappa d) = 2048/(10.75\cdot 512) = 0.37$ — deep, thin, and long-context is precisely the regime where the $O(T^2)$ term matters. (At $T=8192$ in mid-training it becomes $1.49$: attention *outweighs* every weight matrix in the model.) Over $D = 20$B tokens the run is $1.59\times10^{19}$ FLOPs, not the $1.01\times10^{19}$ that bare $6ND$ predicts.

    On a single A100 (312 bf16 TFLOP/s peak), wall-clock $= 14.2 / \text{MFU}$ hours:

    | realized MFU | 25% | 40% | 50% | 57% |
    |---|---|---|---|---|
    | A100-hours for 20B tokens | 57 | 36 | 28 | 25 |

    PLAN §0's canonical **≈22–29 GPU-hr** stable-phase band is the well-utilized end of this table: it needs $\approx$50–58% model-FLOPs utilization, which at 100M requires `torch.compile`, fused cross-entropy and a FlashAttention kernel all switched on (Ch. 14.7 measures what you actually get, and lands at 58.2% attention-inclusive). At \$1–2/GPU-hr the dollar envelope is **\$25–\$50**. This formula is also the denominator of Ch. 14.7's MFU meter — use the three-term version there, or your MFU will read 37% high.

    **KV cache per token.** GQA stores $K$ *and* $V$ for $n_{kv}=2$ heads of $d_h=64$ across $L=30$ layers, in bf16 (2 bytes):
    $$
    \underbrace{2}_{K,V} \times \underbrace{2}_{\text{KV heads}} \times 64 \times 30 \times 2\ \text{bytes} = 15{,}360\ \text{bytes} = 15\ \text{KiB/token}.
    $$
    A full 2048-token context is $2048 \times 15{,}360 = 31{,}457{,}280$ bytes $= 30$ MiB; the 8192-token mid-trained context is $120$ MiB. Under full multi-head attention (8 KV heads) those become $60$ KiB/token, $120$ MiB and $480$ MiB — the 4× GQA win, made concrete. These are the figures the serving chapter (14.11) and [PagedAttention & KV-Cache Memory Management](../04-kernels-efficiency/06-paged-attention-kv.html) care about, and `KVCache.nbytes()` below prints the first one for you.

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

    Our `RMSNorm` is three fuse-able ops that PyTorch will launch separately. `torch.compile` will fuse them; `torch.nn.functional.rms_norm` (PyTorch ≥ 2.4) gives you a single native op; and `apex.normalization.FusedRMSNorm` or [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel)'s `LigerRMSNorm` give you hand-written Triton kernels with a fused backward. We keep the explicit version in the book because you should be able to read it, and swap in the fused one in Ch. 14.7 where throughput matters. The Triton mechanics are in [Writing GPU Kernels with Triton](../04-kernels-efficiency/04-triton-kernels.html).

### QK-norm: the high-learning-rate stabilizer

The single most common way a small model at an aggressive learning rate diverges is an **attention-logit blowup**: the dot product $q \cdot k$ grows without bound, softmax saturates to a near one-hot distribution, the gradient into $W_Q, W_K$ collapses, and the loss spikes irrecoverably (the failure mode dissected in [Training Stability, Loss Spikes & Debugging Large Runs](../03-pretraining/11-training-stability.html)). **QK-norm** — applying RMSNorm to the query and key vectors *per head* before the dot product — bounds the geometry of that product.

Two distinct lineages share the name, and it is worth keeping them apart. Henry et al., *Query-Key Normalization for Transformers* (2020), $\ell_2$-normalize $q$ and $k$ and reintroduce a single learned scalar temperature. The variant that today's open models ship — and the one we implement — is a *per-head LayerNorm/RMSNorm* over the `head_dim` axis with a learned vector $\gamma$, introduced for stability at scale by Dehghani et al. (*Scaling Vision Transformers to 22 Billion Parameters*, 2023) and studied systematically by Wortsman et al. (*Small-scale proxies for large-scale Transformer training instabilities*, 2023). It ships in **OLMo 2**, **Qwen3**, **Gemma 3**, and **Chameleon**. It is what lets us train `Stack-100M` with Muon at a learning rate that would otherwise diverge — the optimizer story is Ch. 14.6, and it leans on [Optimizers: SGD, Adam, Adafactor, Lion, Muon & Shampoo](../03-pretraining/09-optimizers.html). We apply the norm over the `head_dim` axis, before RoPE.

!!! warning "Ablate to learn: turn QK-norm off and watch it die"

    A genuinely instructive ablation. Train the config at the target LR (e.g. peak $\approx 3\text{e-}3$ for the Muon path) with `qk_norm=False`. On the order of a few hundred to a few thousand steps in, you will typically see the attention logits' max magnitude climb past ~50, softmax entropy collapse, and the loss NaN out. Re-enable QK-norm, same LR, and the run is stable. The lesson is causal: QK-norm is not decoration, it is what *buys* the high learning rate that makes the run cheap. Its cost is 2 tiny RMSNorm vectors per layer (128 params) — free. The `record=` hook on `Attention.forward` below harvests the per-head max logit you need to watch this happen.

!!! example "Worked example: how far QK-norm actually bounds the logits"

    Make the mechanism quantitative — and note where the bound stops being free. Without QK-norm, the pre-softmax logit is $q\cdot k = \sum_{i=1}^{d_h} q_i k_i$. If $q$ and $k$ have per-component standard deviation $\sigma$ (independent, zero mean), each product $q_i k_i$ has variance $\sigma^4$, so the dot product over $d_h=64$ dimensions has standard deviation $\approx \sigma^2\sqrt{d_h} = 8\sigma^2$, and after the $1/\sqrt{d_h}=1/8$ scaling, $\approx \sigma^2$. During training $\sigma$ can drift upward as $W_Q, W_K$ grow; if $\sigma$ reaches 6, the scaled logit std is $\approx 36$, and the *max* logit over a 2048-token context (a few std out) can exceed 100. A softmax with a gap of ~100 between the top logit and the rest is numerically a hard argmax: its gradient with respect to the losing logits is $\approx e^{-100}\approx 0$ — the attention pattern is frozen, learning stalls, and one bad step NaNs the run.

    **With QK-norm**, RMSNorm forces the *unscaled* part of each vector to unit mean-square, so $\|q\| \le \sqrt{d_h}\,\|\gamma_q\|_\infty$ and likewise for $k$. By Cauchy–Schwarz, after the $1/\sqrt{d_h}$ scaling,

    $$
    \Big|\tfrac{1}{\sqrt{d_h}}\, q\cdot k\Big| \;\le\; \sqrt{d_h}\,\|\gamma_q\|_\infty \|\gamma_k\|_\infty \;=\; 8\,\|\gamma_q\|_\infty\|\gamma_k\|_\infty .
    $$

    At initialization $\gamma \equiv 1$ and the logit is bounded by $\pm 8$. The crucial caveat: $\gamma$ is *learned and unconstrained*, so the bound is structural but not fixed — it grows with the learned scales. Growth is slow and, unlike the raw projections, directly monitorable: log $\max|\gamma_q|, \max|\gamma_k|$ per layer alongside the `record=` max-logit hook. This residual freedom is exactly why Kimi K2 needed **QK-clip on top of** a normalization scheme (MuonClip rescales $W_Q, W_K$ post-hoc when the observed max logit exceeds a threshold, rather than normalizing) — the mechanism Ch. 14.6 implements. QK-norm removes the *runaway*; QK-clip caps what is left.

{{fig:qk-norm-logit-bound}}

### z-loss and logit soft-cap

Two more cheap stabilizers guard the *output* logits. The **z-loss** (introduced in the PaLM / T5X training recipes) adds a small penalty on the log-partition function of the softmax:

$$
\mathcal{L}_z = \lambda_z \,\big(\operatorname{logsumexp}(\text{logits})\big)^2,
$$

which gently pulls $\log \sum_j e^{z_j}$ toward zero, keeping logits from drifting to large absolute values and keeping the softmax well-conditioned in bf16. We use $\lambda_z = 10^{-4}$. Two implementation obligations follow. First, cross-entropy *already computes* $\operatorname{logsumexp}$ internally, so a good implementation reuses it rather than paying for a second pass over the $(B,T,V)$ tensor. Second — and this is the one people get wrong — **z-loss must use the same normalizer as the CE term**: average over *valid* (non-`ignore_index`) positions only. Averaging z-loss over all $B\times T$ positions while CE averages over unmasked ones makes the two terms silently disagree, which is invisible at $\lambda_z = 10^{-4}$ during pretraining and very visible in the assistant-only-masked SFT regime of Ch. 14.9, where most positions are masked. Both code paths below mask.

Optionally, a **logit soft-cap** (Gemma-2 style) squashes logits through a scaled tanh, $z \leftarrow c \cdot \tanh(z / c)$, hard-bounding them to $(-c, c)$; Gemma-2 reports $c=30$ on final logits and $c=50$ on attention logits. We leave both off by default because z-loss usually suffices and soft-capping the *attention* logits forces you off the FlashAttention fast path — but we expose them, and, critically, we apply the final cap on **both** the training and the inference path. Capping at train time and not at sample time is a silent train/serve mismatch that changes the effective temperature of every generated token; the code below computes logits once and caps once so the bug is unrepresentable. The pretraining-objective chapter, [The Pretraining Objective & Loss](../03-pretraining/03-pretraining-objective.html), treats z-loss in full.

---

## Positional information: RoPE, position ids, and NoPE on every 4th layer

### RoPE

`Stack-100M` encodes position with **Rotary Position Embeddings** (Su et al., *RoFormer*, 2021): rather than adding a position vector, RoPE *rotates* each 2-dimensional slice of the query and key by an angle proportional to the absolute position, so that the attention dot product depends only on the *relative* offset $m - n$. With base $\theta = 10000$ and per-pair frequencies $\omega_i = \theta^{-2i/d_h}$, the query at position $m$ is rotated by $m\omega_i$ in the $i$-th plane. The full derivation lives in [Positional Encodings](../02-transformer/05-positional-encoding.html); here we implement it and keep `rope_theta` as a config knob because mid-training (Ch. 14.8) *rescales* it to extend context from 2048 to 8192 (see [Long-Context Pretraining & Context Extension](../03-pretraining/13-long-context-pretraining.html)).

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

    This is the bug that silently ruins most first KV-cache retrofits. When you call `F.scaled_dot_product_attention(q, k, v, is_causal=True)` with `q_len == kv_len`, you get the lower-triangular mask you expect. But on a decode step `q_len = 1` and `kv_len = N`, and PyTorch aligns the causal mask to the **top-left** of the $1 \times N$ score matrix — so query 0 may attend only to key 0, and your model reads position 0 of the prompt and nothing else. Output is fluent-looking garbage and the loss curve never tells you, because training only ever exercises the square case where top-left and bottom-right alignment coincide.

    Two correct fixes: (1) build the mask from *positions*, `q_pos[:, None] >= kv_pos[None, :]`, which is what `build_doc_causal_mask` below does and which is unambiguous for prefill, chunked prefill and single-token decode alike; or (2) use `torch.nn.attention.bias.causal_lower_right(q_len, kv_len)` and pass it as `attn_mask=`, which asks PyTorch for the bottom-right-aligned causal bias and still hits a fused kernel. Never pass `is_causal=True` with a rectangular score matrix.

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

    def _record_max_logit(self, q, k, scale, attn_mask, record, tile: int = 256):
        """DIAGNOSTIC ONLY -- per-head max attention logit, for QK-clip (Ch. 14.6).
        Tiles over the query axis so peak memory is (B, H, tile, kv_len) fp32 rather
        than the full score matrix. Even so: run this on a SMALL PROBE BATCH every
        N steps, never on the training micro-batch (see the warning below)."""
        with torch.no_grad():
            kf = k.float()
            m = torch.full((self.n_heads,), float("-inf"), device=q.device)
            for i in range(0, q.shape[2], tile):
                a = (q[:, :, i:i + tile].float() @ kf.transpose(-2, -1)) * scale
                if attn_mask is not None:
                    a = a.masked_fill(~attn_mask[..., i:i + tile, :], float("-inf"))
                m = torch.maximum(m, a.amax(dim=(0, 2, 3)))
            record[self.layer_idx] = m                          # (n_heads,)

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

        if record is not None:
            self._record_max_logit(q, k, scale, attn_mask, record)

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

!!! warning "Common pitfall: instrumentation that costs more than the model"

    `record=` materializes attention scores, which is exactly what FlashAttention exists to avoid. Even tiled at 256 queries, the probe holds $(B, H, 256, T)$ fp32 — 268 MB at the A100 micro-batch ($B{=}16$, $H{=}8$, $T{=}2048$), *per layer*. Never leave it on for the training micro-batch. The correct pattern is a **probe batch**: every $N$ steps, run one forward with $B{=}1, T{=}512$ under `no_grad` with `record={}`, log the per-head maxima, throw it away. If even that is too much, Ch. 14.6's QK-clip can trigger on the cheap proxy $\max|q|\cdot\max|k|\cdot\sqrt{d_h}/\sqrt{d_h}$ per head — an upper bound computed from the $q,k$ tensors alone, with no score matrix at all.

Three more implementation notes worth internalizing.

**QK-norm is applied before RoPE**, not after: we normalize the *content* geometry of $q$ and $k$, then inject position. RoPE is norm-preserving (a rotation leaves $\|q\|$ unchanged), so the order does not change the magnitude bound — but it does change how the learned per-dimension scale $\gamma$ interacts with the rotation. Normalizing first keeps $\gamma$ acting in the unrotated content frame, which is the order OLMo 2 and Qwen3 use, and therefore the order that makes our checkpoint numerically faithful when exported as a Qwen3 below.

**We cache the narrow, post-RoPE K/V.** Caching *before* rotation would force a re-rotation on every read; caching *after* head expansion would quadruple the cache and throw away the entire GQA win. Two KV heads in, two KV heads cached.

**`repeat_interleave` is pedagogy, not production.** A real kernel never materializes the expansion: FlashAttention's GQA path and vLLM's PagedAttention read the shared KV head directly. Since PyTorch 2.5 you can get that behavior from SDPA itself — `F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)` accepts `k, v` with `n_kv_heads` and does the broadcast inside the kernel. It is numerically identical (verify it: the max absolute difference is exactly 0) and strictly cheaper. We keep the explicit expansion in the book because it makes the 4:1 sharing visible, and flip it on in Ch. 14.7.

### Document-aware masking: making packing correct

PLAN §2 packs many short documents into every 2048-token training sequence, and requires that **no token attends across a document boundary** and that **position ids reset** at each boundary. Both are model-side responsibilities, and both are cheap once positions and masks are explicit. Ch. 14.2's dataset already emits exactly the two tensors we need per row: `position_ids` (0,1,2,… restarting at each document) and `seq_ids` (0,0,0,1,1,2,2,… the per-token document index).

The mask is then the conjunction of "causal" and "same document". This is the single function the whole model uses — prefill, packed training, and decode:

```python
def build_doc_causal_mask(seq_ids, T, kv_len, start_pos, device):
    """Bool mask (B|1, 1, T, kv_len); True = attend. None = plain causal fast path.
    Positions, not indices: correct for square prefill, rectangular decode, and
    chunked prefill alike -- which is what makes the is_causal footgun unreachable."""
    if seq_ids is None and kv_len == T and start_pos == 0:
        return None                                  # let SDPA use is_causal=True
    q_pos = torch.arange(start_pos, start_pos + T, device=device)
    kv_pos = torch.arange(kv_len, device=device)
    m = (q_pos[:, None] >= kv_pos[None, :])[None, None]           # (1,1,T,kv_len)
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

The SwiGLU forward saves four `(B, T, 1408)` intermediates for backward (gate, up, silu(gate), and their product). Liger-Kernel's `LigerSwiGLUMLP` fuses them into one Triton kernel that recomputes rather than stores — a straight ~2×-activation-memory win on the MLP with no math change. Worth switching on in Ch. 14.7 once the from-scratch version is understood.

---

## Assembling the block, the loss, and the full model

A `Block` is pre-norm attention plus pre-norm SwiGLU, each wrapped in a residual add. The full model is an embedding, 30 blocks, a final norm, and a tied output projection. Here is the data flow:

```text
  tokens ─► Embedding (32768×512, tied) ─► x  (B,T,512)
     position_ids ─► rope_cos/sin gather ─► (B,T,64)
     seq_ids ──────► build_doc_causal_mask► (B,1,T,kv_len) bool
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

### The loss: cross-entropy, z-loss, and the logit tensor

The output head is where a *small* model with a *large* vocabulary springs a nasty surprise. At a micro-batch of $16 \times 2048 = 32{,}768$ tokens, the logits tensor is $32768 \times 32768 \approx 1.07\times10^9$ elements. In bf16 that is 2.1 GB; `logits.float()` makes it 4.3 GB; `F.cross_entropy` internally saves a `log_softmax` of the same shape, another 4.3 GB — **10.7 GB for one tensor in a model whose entire trainable state is 1.28 GB.** It is not larger than every other activation put together (uncheckpointed activations are about twice as big), but it *is* the largest single tensor in the step, it is 8× the whole model, and the moment you turn on activation checkpointing — which you should, and which the next section quantifies — it becomes the dominant term outright. This is a direct, unavoidable consequence of the small-model/large-vocab regime we chose on purpose.

The fix is to never materialize the whole thing: chunk over tokens, fuse `lm_head + log_softmax + gather` inside each chunk, and *recompute* the chunk in the backward pass instead of storing it. CE and z-loss share the same $\operatorname{logsumexp}$, so we compute it once — and both are normalized by the same valid-token count.

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
    (min(chunk, n_tokens) x vocab) instead of (B*T x vocab), and does NOT grow
    with batch size. CE and z-loss share one logsumexp and one normalizer."""
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
        attn_mask = build_doc_causal_mask(seq_ids, T, kv_len, start_pos, dev)

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
            tg = targets.reshape(-1)
            valid = (tg != -100)
            ce = F.cross_entropy(lf.view(-1, lf.shape[-1]), tg, ignore_index=-100)
            logz = torch.logsumexp(lf, dim=-1).reshape(-1)                 # (B*T,)
            # z-loss uses the SAME normalizer as CE -- valid tokens only (Ch. 14.9 masks most)
            zl = (logz.pow(2) * valid).sum() / valid.sum().clamp_min(1)
            loss = ce + self.cfg.z_loss_coef * zl
        return logits, loss

    @torch.no_grad()
    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and self.cfg.tie_embeddings:
            n -= self.tok_emb.weight.numel()   # tied head shares the same tensor
        return n
```

A few points that make this *correct* and not just plausible. The residual-projection initializations (`wo`, `down`) are scaled by $1/\sqrt{2L}$ — the GPT-2 trick — so that the variance added into the residual stream does not grow with depth. Each layer contributes two residual writes (attention, MLP), so after $L$ layers the stream has accumulated $2L$ writes; scaling each output projection's init std by $1/\sqrt{2L}$ keeps the accumulated variance $O(1)$ instead of $O(L)$. With 30 layers this matters, and skipping it is a common cause of early-training instability in deep-thin models. Because `lm_head.weight` is *the same tensor* as `tok_emb.weight`, `self.apply(self._init_weights)` initializing it twice is harmless — both draw from the same $\mathcal N(0, 0.02^2)$, and the tie is a shared reference, not a copy, so the two stay identical through training. And because the head is tied, `num_params(non_embedding=True)` subtracts the shared tensor exactly once to recover the 84,576,512 body count used in the FLOP estimate.

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

    # 4. the FLOP model Ch. 14.7's MFU meter divides by (three terms, not one)
    def flops_per_token(c, N_body, T):
        return 6 * N_body + 6 * c.n_layers * c.d_model * T + 6 * c.vocab_size * c.d_model
    assert flops_per_token(cfg, 84_576_512, 2048) == 796_866_048

    # --- behavioural tests on a tiny config (seconds on CPU) ---
    tiny = StackConfig(vocab_size=97, d_model=64, n_layers=4, n_heads=4, n_kv_heads=2,
                       head_dim=16, intermediate=128, max_seq_len=64)
    net = Stack100M(tiny).eval()
    idx = torch.randint(0, 97, (2, 10))

    # 5. cached incremental decode == full recompute (catches the top-left mask bug)
    full, _ = net(idx)
    cache = KVCache(tiny, 2, 20, "cpu", torch.float32)
    net(idx[:, :6], kv_cache=cache, start_pos=0)
    inc = torch.stack([net(idx[:, t:t + 1], kv_cache=cache, start_pos=t)[0][:, -1]
                       for t in range(6, 10)], dim=1)
    assert torch.allclose(full[:, 6:], inc, atol=1e-5)

    # 6. document masking: packed docs == the same docs run separately
    a, b = torch.randint(0, 97, (1, 5)), torch.randint(0, 97, (1, 7))
    joint, _ = net(torch.cat([a, b], 1),
                   position_ids=torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6]]),
                   seq_ids=torch.tensor([[0] * 5 + [1] * 7]))
    assert torch.allclose(joint[:, :5], net(a)[0], atol=1e-5)
    assert torch.allclose(joint[:, 5:], net(b)[0], atol=1e-5)

    # 7. chunked fused loss == the reference loss, VALUE and gradients, with a masked
    #    target and an exaggerated z_coef -- the setting that catches a normalizer
    #    mismatch between the two paths (see the z-loss discussion above).
    tg = torch.randint(0, 97, (2, 10)); tg[0, 3] = -100
    ref = Stack100M(tiny); ref.cfg.z_loss_coef = 0.5
    ref.zero_grad(); l0 = ref(idx, targets=tg)[1]; l0.backward()
    g_ref = ref.tok_emb.weight.grad.clone()
    ref.cfg.loss_chunk = 4
    ref.zero_grad(); l1 = ref(idx, targets=tg)[1]; l1.backward()
    assert torch.allclose(l0, l1, atol=1e-5)
    assert torch.allclose(g_ref, ref.tok_emb.weight.grad, atol=1e-4)

    # 8. greedy generation is cache-invariant (the highest-value test here)
    torch.manual_seed(0); o1 = net.generate(idx[:, :4], 6, temperature=0.0)
    torch.manual_seed(0); o2 = net.generate(idx[:, :4], 6, temperature=0.0, use_cache=False)
    assert torch.equal(o1, o2)
    print("all model invariants OK")
```

---

## Budgeting the run: memory, micro-batch, and what actually fits

The FLOP model tells you the *compute*; it says nothing about whether your step fits in the card. Sizing the micro-batch is a first-principles exercise every CS336-grade practitioner should be able to do on a napkin, and for Stack-100M it has a surprising answer: **the model is nearly free and the logits are not.**

Start with the persistent state. Ch. 14.6 uses the standard hybrid: **Muon** on the 2D hidden matrices, **AdamW** on embeddings, norms, and 1D params. That partition is exactly our parameter accounting:

| Group | Params | Optimizer | States | fp32 bytes |
|---|---|---|---|---|
| 2D body matrices (30 × attn+MLP) | 84,541,440 | Muon | 1 momentum buffer | 338 MB |
| Embedding + all norms | 16,812,288 | AdamW | m, v | 134 MB |
| Master weights (fp32) | 101,353,728 | — | — | 405 MB |
| Gradients (fp32) | 101,353,728 | — | — | 405 MB |
| **Persistent total** | | | | **≈ 1.28 GB** |

That is it. A 101M-parameter model's entire trainable state is **under 1.3 GB** — 1.6% of an A100-80GB. Everything else in your memory profile is activations, and there are exactly three numbers to know, all per *token*:

- **Block activations, stored.** Per layer the pre-norm block saves on the order of 11,000–12,000 elements for backward (the two norm inputs and outputs; $q,k,v$ pre- and post-QK-norm and post-RoPE; the expanded $k,v$; the attention output; and SwiGLU's four `(·,1408)` intermediates). In bf16 that is $\approx 22$ KiB per token per layer, so **≈ 0.68 MB/token** across 30 layers.
- **Block activations, checkpointed.** Wrap each `Block` in `torch.utils.checkpoint.checkpoint` and you store only the 30 block *inputs*: $30 \times 512 \times 2\ \text{B} = $ **30 KiB/token**, a 22× reduction, plus one block's worth of recompute scratch live at a time ($\approx 22$ KiB/token).
- **Logits.** The naive path holds $V \times (2 + 4 + 4) = 10$ bytes per token — bf16 logits, the `.float()` copy, and `cross_entropy`'s saved `log_softmax` — i.e. **0.328 MB/token**. The chunked path holds $\min(\text{chunk}, n_{\text{tokens}}) \times V \times 4$ bytes *in total*, independent of batch size.

Now the tiers, in decimal GB:

| Tier | micro-batch × seq | tokens | activations (stored / ckpt+scratch) | logits (naive / chunked) | persistent | recommended peak |
|---|---|---|---|---|---|---|
| A100 80GB | 16 × 2048 | 32,768 | 22.1 / 1.0 + 0.7 | 10.7 / 1.07 | 1.28 | **≈ 4.1 GB** |
| RTX 4090 24GB | 8 × 2048 | 16,384 | 11.1 / 0.50 + 0.37 | 5.4 / 1.07 | 1.28 | **≈ 3.2 GB** |
| Colab T4 16GB | 2 × 2048 | 4,096 | 2.8 / 0.13 + 0.09 | 1.34 / 0.54 | 1.28 | **≈ 2.0 GB** |

("Recommended peak" = checkpointing on, `loss_chunk=8192`. Note the T4 row's chunked figure is 0.54 GB, not 1.07 — with only 4,096 tokens in the micro-batch, `chunk=8192` never binds.)

Four lessons fall out, and they are the reason to build the table:

1. **Chunked/fused cross-entropy is not a micro-optimization at this scale.** On the 4090 the naive step is $11.1 + 5.4 + 1.28 = 17.8$ GB of 24 — it *fits*, with no room for allocator fragmentation, cuBLAS workspaces, or a concurrent eval batch. Chunking drops it to 13.4 GB and, more importantly, **decouples logit memory from batch size**: doubling the micro-batch adds nothing to that term.
2. **Activation checkpointing is the bigger lever, and it changes which term dominates.** With checkpointing on, the A100 step is 1.0 GB of stored activations against 10.7 GB of naive logits — the logit tensor is now roughly 80% of the step and 8× the entire trainable state. This is the configuration you actually train in, and it is why the two levers belong together: checkpointing alone leaves you logit-bound, chunking alone leaves you activation-bound.
3. **With both on, Stack-100M is not memory-bound at all.** An A100 step at 4.1 GB is using 5% of the card. The correct response is to *raise* the micro-batch until throughput stops improving (checkpointing costs roughly +30% step time, which a larger micro-batch partly buys back), then set gradient accumulation to hit PLAN §5's 0.5M-token effective batch. See [Memory-Efficient Training: Checkpointing, Offloading & LoRA Math](../04-kernels-efficiency/10-memory-efficient-training.html).
4. **The 0.5M-token batch is reached by gradient accumulation, not by a bigger card.** All three tiers train the *same* effective batch; they differ only in how many micro-steps it takes. That is what makes the recipe portable.

These are analytic estimates — real `torch.cuda.max_memory_allocated()` will differ by allocator fragmentation, cuBLAS workspaces, and whether `torch.compile` fused something away. Measure, then compare to the napkin. When they disagree by 2×, you have learned something.

!!! interview "Interview Corner"

    **Q:** At 100M parameters you chose GQA with 2 KV heads, tied embeddings, and a 32k vocab. Walk me through *why each of those specific numbers* rather than the "obvious" MHA / untied / 50k defaults, and what you'd lose if you flipped them.

    **A:** All three are driven by the fact that at 100M the budget — both parameters *and* memory — is dominated by a few big tensors, so cheap-but-good choices compound. (1) **GQA with 2 KV heads**: full MHA would cost ~11.8M extra parameters (over four blocks' worth) and, more importantly, 4× the KV cache — the 15 KiB/token figure derived earlier becomes 60 — for a quality gain that is negligible at $d_h=64$. That 4× is what makes the model cheap to *serve* at long context and high concurrency, which is the whole economic argument for a small model. (2) **Tied embeddings**: input embedding and output projection are both $V\times d$; tying saves 16.8M params — about a sixth of the model — and at small scale the shared representation is not a quality regression (Press & Wolf). (3) **32k vocab**: a 50k vocab costs 25.7M tied params vs 16.8M; the ~9M saved buys roughly three more transformer blocks, and at small scale (deep-thin, MobileLLM) depth is the better use of the budget. The second-order reason interviewers rarely expect: $V$ also sets the size of the training logit tensor, which under activation checkpointing is the *largest single tensor in the step* — 10.7 GB at a 32k-token micro-batch, 8× the model's entire trainable state — so a lean vocab buys memory headroom as well as parameters. Flip any of them and you bloat the serving footprint (MHA), waste a sixth of the model (untying), or trade depth *and* memory for tokenization granularity (50k) — all bad trades *specifically because 100M is small*. At 100B these trades invert: the embedding is a rounding error, MHA's cache is amortized over far more compute per token, and a bigger vocab pays for itself on multilingual coverage.

---

## From `stacklm` to the ecosystem: making the checkpoint loadable

A from-scratch module is a teaching artifact until somebody else's runtime can load it. Ch. 14.11 promises int8/int4 post-training quantization and a measured-latency CPU run, and those paths go through HuggingFace `transformers`, `vLLM`, and `llama.cpp`/GGUF. None of them can load a `torch.save` of `Stack100M` — so here is the bridge, and the honest statement of what it costs.

**The good news: Stack-100M is architecturally a Qwen3.** Grouped-query attention, RMSNorm, SwiGLU, no biases, tied embeddings, and — the distinguishing detail — **QK-norm as an RMSNorm over `head_dim` applied before RoPE** is exactly the `Qwen3ForCausalLM` recipe, and `Qwen3Config` exposes `head_dim` independently of `hidden_size / num_attention_heads`, which our 512/8/64 shape needs.

**The one incompatibility is NoPE**, and it is the chapter's default. No stock architecture supports "skip RoPE on every 4th layer," so you choose deliberately, *before* you train:

- **Export path A — maximum compatibility.** Train with `nope_every = 0` (pure RoPE). Export as `Qwen3ForCausalLM` with a pure key-rename script. You lose the length-extrapolation ingredient of Ch. 14.8 and gain the entire ecosystem — `transformers`, vLLM, SGLang, `llama.cpp`, GPTQ/AWQ/bitsandbytes — for free.
- **Export path B — keep NoPE (the book's default).** Ship a `trust_remote_code` model: a `configuration_stacklm.py` / `modeling_stacklm.py` pair registered with `AutoConfig.register` / `AutoModelForCausalLM.register`, which `transformers` loads directly and which vLLM can pick up via `ModelRegistry.register_model("StackLMForCausalLM", StackLMForCausalLM)` in an out-of-tree plugin. `llama.cpp` is the hard one: GGUF conversion needs a `Model` subclass in `convert_hf_to_gguf.py` *and* a matching architecture implemented in C++, so a genuinely new architecture is not a weekend port. For the laptop run with NoPE kept, Ch. 14.11 uses our own round-to-nearest int8/int4 on top of the `generate()` above running on CPU — which is precisely why we wrote it from scratch.

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

    `load_state_dict` succeeding proves the *shapes* match, not the *semantics*. The two conventions that silently differ between implementations are the **RoPE pair layout** (halved vs interleaved — ours is halved/NeoX, matching HF) and **whether QK-norm is applied before or after RoPE** (ours is before, matching Qwen3). Both produce a model that loads, runs, and generates fluent nonsense. Always assert `torch.allclose(ours(idx)[0], hf(idx).logits, atol=1e-3)` on a fixed random input before you trust an export — and do it *before* you quantize, or you will debug two bugs at once.

Once the checkpoint is a `Qwen3ForCausalLM`, the rest of the stack is free: `AutoGPTQ`/`GPTQModel` or `autoawq` for weight-only int4 (Ch. 4.7), `bitsandbytes` for int8 (Ch. 4.8), `vllm serve` for an OpenAI-compatible endpoint (Ch. 7.3), `convert_hf_to_gguf.py` + `llama-cli` for the laptop run (Ch. 14.11). That single config flag — `nope_every` — is the difference between "a research artifact" and "a model anyone can run," and it is a decision you make *before* the 20B-token run, not after.

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
        # An MLA KV cache stores (c_kv, k_r) -- a DIFFERENT pair of tensors than KVCache
        # above -- so `mixer="mla"` needs its own cache class; Exercise 5 builds it.
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
    so you pre-multiply the *query* into the $d_c$-dimensional latent space once and dot it directly against the cached $c^{KV}$. Per-head keys are never materialized at decode. The same absorption folds $W_{UV}$ into $W_O$. Exercise 6 has you verify the identity numerically. Without absorption, MLA is a smaller cache and a slower decoder; with it, it is both smaller and faster, which is why DeepSeek could serve it.

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

    Every variant above is a *drop-in module* with the same `(x, cos, sin, attn_mask=, kv_cache=, start_pos=) -> x` contract as the default `Attention` (`GatedShortConv` absorbs the extras with `*args, **kwargs`). The right engineering pattern is a factory: `Block` reads `cfg.mixer in {"gqa","mla","conv"}` and constructs the right mixer, so you can ablate architectures by changing one config field and re-running the *same* training loop from Ch. 14.7. Forking `model.py` per variant is how capstones rot. One model, config-selected mixers — Exercise 5 has you build it.

---

## Key Takeaways

!!! key "Key Takeaways"

    - **Deep-and-thin is the small-model bet**: at fixed 100M params, 30 layers × 512 width beats shallow-wide (MobileLLM, 2024); depth = sequential reasoning steps, which is what small models are bottlenecked on. The width-per-layer ratio $d/L \approx 17$ is ~4× thinner than GPT-2-small's ~64.
    - **The exact config is frozen** (PLAN §1): `vocab 32768, d_model 512, n_layers 30, 8 heads / 2 KV heads, head_dim 64, SwiGLU 1408, tied embeddings` → **exactly 101,353,728 params**, of which **84,576,512** is the non-embedding body and 16,777,216 the tied embedding. Reproduce that integer by hand, then assert it in CI. Each of the three cheap choices pays twice: **GQA (2 KV heads)** saves ~11.8M params *and* cuts the KV cache 4× (15 KiB/token, 30 MiB at 2048); **tied embeddings** save a sixth of the model; **a 32k vocab** frees ~9M params (≈3 blocks) versus 50k *and* shrinks the training logit tensor.
    - **$6ND$ under-counts by 57% at this shape.** Use FLOPs/token $= 6N_{\text{body}} + 6LdT + 6Vd = 7.97\times10^8$: attention adds +37% (the ratio is $T/(\kappa d) = 0.37$, and 1.49 at the 8192 mid-training context), the tied `lm_head` adds +20%. That is $1.59\times10^{19}$ FLOPs for 20B tokens — 25–45 A100-hours depending on realized MFU, and the correct denominator for Ch. 14.7's MFU meter.
    - **Stability is engineered, not hoped for**: pre-norm + RMSNorm (fp32); **QK-norm** bounds the scaled attention logit by $8\|\gamma_q\|_\infty\|\gamma_k\|_\infty$ — $\pm 8$ at init, but $\gamma$ is learned, which is exactly why Ch. 14.6 adds QK-clip on top; **z-loss** on the log-partition, normalized over *valid* tokens like CE; optional Gemma-2 soft-caps applied on *both* train and inference paths; $1/\sqrt{2L}$ residual-init scaling for the deep stack.
    - **RoPE + NoPE-every-4th-layer** (SmolLM3, 2025; Kazemnejad et al., 2023) with **explicit `position_ids`** — the indirection that makes document packing, incremental decode, and the 2048→8192 context extension all work from one code path.
    - **Correct masking is a model responsibility.** Document-aware packing needs a causal-AND-same-document mask and per-document position resets; incremental decode needs a bottom-right-aligned mask, because SDPA's `is_causal=True` is *top-left* aligned and silently reads only position 0. `flex_attention`'s `create_block_mask` (block-sparse, fused, `score_mod`-capable) and `flash_attn_varlen_func` + `cu_seqlens` are the two production answers.
    - **At 100M the logits are the memory bomb, not the model.** The whole trainable state is ≈1.28 GB; the naive loss path holds 10 bytes/token/vocab-entry = 10.7 GB at a 32k-token micro-batch — 8× the model, and 82% of the step once activation checkpointing is on. Chunked/fused linear-cross-entropy (Liger-Kernel, Cut Cross-Entropy) **plus** checkpointing is what puts an A100 step at ~4 GB and makes the recipe fit a 4090 and a T4.
    - **Plan the ecosystem exit before you train.** Stack-100M with `nope_every=0` is architecturally a **Qwen3**, so a pure key-rename makes it loadable by `transformers`/vLLM/`llama.cpp` and every quantizer; keep NoPE (the default) and you owe a `trust_remote_code` pair, a vLLM `ModelRegistry.register_model`, and C++ for GGUF. Verify any port numerically — shapes matching is not semantics matching.
    - **Efficiency variants are options, not the default**: MLA (DeepSeek-V2) compresses KV into a latent — and the *absorption* of $W_{UK}$ into the query is why it is fast, not just small; MTP (DeepSeek-V3) conditions on the next token's embedding before predicting $t{+}2$; an LFM2-style gated short-conv block trades long-range capacity for KV-free decode speed.

!!! sota "State of the Art & Resources (2026)"
    Every mechanism wired into `Stack-100M` — GQA, RMSNorm, QK-norm, RoPE/NoPE, SwiGLU, MLA, MTP — is a named, shipped choice in today's frontier open models; the list below links the primary sources plus the repos worth reading alongside the code in this chapter.

    **Foundational work**

    - [Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021)](https://arxiv.org/abs/2104.09864) — the RoPE mechanism `build_rope_cache`/`apply_rope` implement.
    - [Zhang & Sennrich, *Root Mean Square Layer Normalization* (2019)](https://arxiv.org/abs/1910.07467) — RMSNorm, used for every norm in the model including QK-norm.
    - [Shazeer, *GLU Variants Improve Transformer* (2020)](https://arxiv.org/abs/2002.05202) — SwiGLU, the MLP sublayer.
    - [Press & Wolf, *Using the Output Embedding to Improve Language Models* (2017)](https://arxiv.org/abs/1608.05859) — tied input/output embeddings, worth 16.8M params here.

    **Stability and the QK-norm lineage**

    - Henry et al., *Query-Key Normalization for Transformers* (2020) — the original: $\ell_2$-normalize $q,k$ with a learned scalar temperature.
    - [Dehghani et al., *Scaling Vision Transformers to 22 Billion Parameters* (2023)](https://arxiv.org/abs/2302.05442) — the per-head LayerNorm-on-Q/K variant that today's LLMs actually ship.
    - [Wortsman et al., *Small-scale proxies for large-scale Transformer training instabilities* (2023)](https://arxiv.org/abs/2309.14322) — systematic study of attention-logit growth and qk-norm as the fix; the empirical backing for the ablation box above.
    - [Kimi Team, *Kimi K2: Open Agentic Intelligence* (2025)](https://arxiv.org/abs/2507.20534) — MuonClip/QK-clip: the *alternative* stabilizer that rescales $W_Q, W_K$ post-hoc instead of normalizing. Ch. 14.6 implements it on top of QK-norm.
    - OLMo 2 (AI2, 2024), Qwen3 (Alibaba, 2025), Gemma 3 (Google DeepMind, 2025) and Chameleon (Meta, 2024) — open models that ship per-head QK-norm; Qwen3 is the export target here.

    **Recent advances (2023–2026)**

    - [Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (2023)](https://arxiv.org/abs/2305.13245) — the 8-query/2-KV-head attention design.
    - [Liu et al., *MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases* (2024)](https://arxiv.org/abs/2402.14905) — the deep-and-thin evidence behind `d_model=512, n_layers=30`.
    - [Kazemnejad et al., *The Impact of Positional Encoding on Length Generalization in Transformers* (2023)](https://arxiv.org/abs/2305.19466) — "NoPE", the analysis behind the every-4th-layer interleave.
    - [DeepSeek-AI, *DeepSeek-V2* (2024)](https://arxiv.org/abs/2405.04434) — introduces MLA, implemented here as the optional `MLAttention` module.
    - [DeepSeek-AI, *DeepSeek-V3 Technical Report* (2024)](https://arxiv.org/abs/2412.19437) — ships Multi-Token Prediction at scale, the basis for `MTPHead`; see also [Gloeckle et al. (2024)](https://arxiv.org/abs/2404.19737).
    - [Gemma Team, *Gemma 2* (2024)](https://arxiv.org/abs/2408.00118) — the logit/attention soft-capping this chapter exposes as `logit_soft_cap`/`attn_soft_cap`.
    - [Wijmans et al., *Cut Your Losses in Large-Vocabulary Language Models* (2024)](https://arxiv.org/abs/2411.09009) — the fused linear-cross-entropy that removes this chapter's dominant activation cost.
    - Liquid AI, *LFM2* (2025) — the gated short-convolution + attention hybrid behind `GatedShortConv`.

    **Open-source & tools**

    - [huggingface/smollm](https://github.com/huggingface/smollm) — training code and recipe for SmolLM3, the model that popularized the every-4th-layer NoPE interleave.
    - [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) — the fused, GQA-aware kernel PyTorch's `scaled_dot_product_attention` dispatches to (Dao et al., *FlashAttention* 2022 and *FlashAttention-2* 2023), and the home of `flash_attn_varlen_func` + `cu_seqlens` for packed sequences.
    - [PyTorch FlexAttention](https://pytorch.org/blog/flexattention/) — `create_block_mask` / `score_mod`: document-causal block-sparse masking and soft-capping inside a fused kernel.
    - [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) — Triton kernels for fused RMSNorm, SwiGLU, RoPE and `LigerFusedLinearCrossEntropy`.
    - [apple/ml-cross-entropy](https://github.com/apple/ml-cross-entropy) — Cut Cross-Entropy, the loss that never materializes a logit tensor.
    - [vllm-project/vllm](https://github.com/vllm-project/vllm) — PagedAttention and `ModelRegistry.register_model` for serving a custom architecture.
    - [SmolLM3: smol, multilingual, long-context reasoner](https://huggingface.co/blog/smollm3) — HuggingFace's own writeup of the GQA + NoPE design choices this chapter cites.

## Further reading

If you read only five things after this chapter, read them in this order:

1. **Liu et al., *MobileLLM* (2024)** — the single paper that justifies the whole aspect-ratio decision; read the depth-vs-width ablations first.
2. **The SmolLM3 model report (HuggingFace, 2025)** — the closest published relative of Stack-100M, and the source of the NoPE interleave and intra-document masking choices.
3. **Wortsman et al., *Small-scale proxies for large-scale Transformer training instabilities* (2023)** — how to *see* an attention-logit blowup before it kills a run, at a scale you can afford to reproduce.
4. **Wijmans et al., *Cut Your Losses in Large-Vocabulary Language Models* (2024)** — the definitive treatment of the memory bomb this chapter spends a whole section on.
5. **DeepSeek-V2 (2024), §2 on MLA** — read it *with the absorption identity in hand*; it is the clearest worked example in the literature of co-designing an architecture with its inference kernel.

---

## Exercises

**1.** With `nope_every = 4`, layer index $\ell$ (0-based) uses NoPE iff $(\ell + 1) \bmod 4 = 0$. (a) List the NoPE layers and count them; how many of the 30 keep RoPE? (b) Why can a decoder-only model afford to drop the positional encoding on *some* layers at all, and what does the mixture buy that pure RoPE does not? (c) Why not go all the way to pure NoPE? (d) Why is `nope_every = 0` the switch that makes the checkpoint loadable by stock `transformers`/vLLM, and what exactly do you owe if you keep it at 4?

??? note "Solution"
    (a) NoPE layers are $\ell = 3, 7, 11, 15, 19, 23, 27$ — **7 layers**; the remaining **23** keep RoPE.

    (b) A causal mask is itself a positional signal: the number of tokens a position may attend to encodes its absolute index, so the model can infer position without an injected encoding. Layers freed from RoPE are not tied to the rotation frequencies seen during training, and the mixture empirically generalizes to sequences **longer than training** — the length robustness the 2048→8192 extension in Ch. 14.8 needs.

    (c) Pure NoPE underperforms on *short* context: the mask signal is diffuse compared to RoPE's sharp relative encoding. Hence the interleave — RoPE on the majority for local precision and sample efficiency, a 1-in-4 minority of NoPE layers for extrapolation. The ratio is SmolLM3's tuned constant, not a law.

    (d) No stock `transformers` architecture expresses "skip RoPE on every 4th layer." Everything else about Stack-100M — GQA, RMSNorm, SwiGLU, no biases, tied embeddings, and per-head QK-norm applied *before* RoPE — is precisely the `Qwen3ForCausalLM` recipe, and `Qwen3Config` exposes `head_dim` independently of `hidden_size / num_attention_heads`, which our 512/8/64 shape requires. With `nope_every = 0` the export is a pure key rename and the checkpoint is a first-class citizen of `transformers`, vLLM, SGLang, GGUF and the GPTQ/AWQ quantizers. Keep NoPE and you owe: a `trust_remote_code` `configuration_stacklm.py`/`modeling_stacklm.py` pair, a vLLM out-of-tree plugin calling `ModelRegistry.register_model`, and — for `llama.cpp` — both a `convert_hf_to_gguf.py` `Model` subclass *and* a new C++ architecture. Decide before the 20B-token run, not after.

**2.** Suppose the block used **full multi-head attention** ($n_{kv} = 8$) instead of GQA. (a) How many extra parameters per layer, and over all 30 layers? Express the total in "blocks' worth" of budget. (b) Recompute the per-token KV cache and the footprint of one 8192-token sequence under both schemes. (c) What is the new value of $\kappa$ (per-layer params $/\,d^2$), and how does that change the attention-to-parameter FLOP ratio $T/(\kappa d)$ at $T = 2048$?

??? note "Solution"
    (a) Under MHA, $W_K$ and $W_V$ each become $512 \times 512 = 262{,}144$ instead of $65{,}536$: extra $196{,}608$ per matrix, $393{,}216 \approx 0.39$M per layer, and $30 \times 393{,}216 = 11{,}796{,}480 \approx 11.8$M over the stack. That is $11.8/2.82 \approx 4.2$ **blocks' worth** of parameters for a quality gain that is negligible at $d_h = 64$.

    (b) GQA: $2 \times 2 \times 64 \times 30 \times 2\ \text{B} = 15{,}360$ B/token, so $8192 \times 15{,}360 = 125{,}829{,}120$ B $= 120$ MiB. MHA has 4× the KV heads: 60 KiB/token and **480 MiB** for the same sequence (and 120 MiB rather than 30 MiB at the 2048 pretrain context). That 4× is the serving argument for GQA.

    (c) MHA attention costs $4d^2$ per layer versus GQA's $2.5d^2$, so $\kappa$ rises from $10.75$ to $12.25$. The attention-FLOP ratio $T/(\kappa d)$ falls from $2048/5504 = 0.372$ to $2048/6272 = 0.327$ — the *relative* attention cost drops slightly, because you added parameter FLOPs without adding score FLOPs. Note the direction: MHA is not cheaper, it just shifts more of the same total into dense matmul, which is why the memory argument (4× cache) rather than the FLOP argument decides this.

**3.** Derive the corrected training-FLOP budget and the MFU it implies. (a) Write the three-term per-token formula and evaluate each term for Stack-100M at $T = 2048$. (b) What total FLOPs does the 20B-token run cost, and how does that compare to bare $6ND$? (c) At 40% MFU on an A100 (312 bf16 TFLOP/s peak) how many GPU-hours is that, and what MFU would you need to land inside PLAN §0's 22–29 GPU-hr band? (d) Recompute the attention term at the 8192-token mid-training context and say what it implies for Ch. 14.8's step time.

??? note "Solution"
    (a) FLOPs/token $\approx 6N_{\text{body}} + 12 L d \bar T + 6Vd$ with $\bar T = T/2$, i.e. $6N_{\text{body}} + 6LdT + 6Vd$. With $N_{\text{body}} = 84{,}576{,}512$, $L=30$, $d=512$, $T=2048$, $V=32768$:

    - parameter matmuls: $6 \times 84{,}576{,}512 = 507{,}459{,}072$;
    - attention (scores + values, fwd+bwd): $6 \times 30 \times 512 \times 2048 = 188{,}743{,}680$ (**+37%**);
    - tied `lm_head`: $6 \times 32768 \times 512 = 100{,}663{,}296$ (**+20%**);
    - total $= 796{,}866{,}048 \approx 7.97\times10^{8}$ FLOP/token.

    (b) $7.97\times10^8 \times 2\times10^{10} = 1.59\times10^{19}$ FLOPs, versus $6ND = 1.01\times10^{19}$ — bare $6ND$ under-counts by **57%**.

    (c) A100 peak $\times$ 3600 s $= 1.123\times10^{18}$ FLOP/hour, so hours $= 14.2/\text{MFU}$. At 40% that is **35.5 GPU-hours** (not the 22.6 that $6ND$ alone predicts). To land inside PLAN §0's 22–29 GPU-hour band you need MFU between $14.2/29 \approx 49\%$ and $14.2/22 \approx 65\%$; the measured loop's 58% attention-inclusive utilization gives 24.4 h, comfortably inside. At \$1–2/GPU-hr the \$25–\$50 envelope holds, and Ch. 14.7 measures which end you land on.

    (d) At $T = 8192$ the attention term quadruples to $7.55\times10^8$ FLOP/token, more than the parameter term: total $\approx 1.36\times10^9$, a **1.7×** per-token cost increase versus the 2048 phase. Mid-training is therefore much more expensive per token than pretraining — which is exactly why Ch. 14.8 anneals on a *short* high-quality budget rather than re-running 20B tokens at 8192, and why FlashAttention (or `flex_attention` with a document block mask) stops being optional there.

**4.** Estimate peak training memory for one micro-batch of $8 \times 2048$ tokens on an RTX 4090 (24 GB). (a) Break down the naive path using the chapter's per-token figures. (b) Turn on `loss_chunk = 8192` *and* per-`Block` activation checkpointing; recompute, and say which term dominates in each of the four combinations (naive/chunked × stored/checkpointed). (c) With both levers on you are at ~3 GB of 24. What is the right next move, and what constraint from PLAN §5 must it respect?

??? note "Solution"
    (a) $8 \times 2048 = 16{,}384$ tokens.

    - **Persistent** (batch-independent): fp32 master weights $101{,}353{,}728 \times 4 \approx 0.41$ GB, fp32 grads $\approx 0.41$ GB, Muon momentum on 84,541,440 2D params $\approx 0.34$ GB, AdamW $m,v$ on 16,812,288 params $\approx 0.13$ GB ⇒ **1.28 GB**.
    - **Activations, stored**: $16{,}384 \times 0.68$ MB $\approx$ **11.1 GB**.
    - **Logits, naive**: $16{,}384 \times 32768 \times 10\ \text{B} \approx$ **5.4 GB** (bf16 logits + `.float()` copy + `cross_entropy`'s saved `log_softmax`).

    Total $\approx$ **17.8 GB** — it fits in 24 GB, but with nothing left for fragmentation, cuBLAS workspaces or a concurrent eval batch.

    (b) Chunked logits: $8192 \times 32768 \times 4\ \text{B} \approx 1.07$ GB, independent of batch. Checkpointed activations: $16{,}384 \times 30$ KiB $\approx 0.50$ GB stored, plus one block of recompute scratch $\approx 0.37$ GB.

    | | stored activations | checkpointed |
    |---|---|---|
    | naive logits | 17.8 GB — *activations* dominate (11.1) | 7.5 GB — **logits** dominate (5.4) |
    | chunked logits | 13.4 GB — *activations* dominate (11.1) | **3.2 GB** — persistent state dominates (1.28) |

    The instructive cell is the top-right: checkpointing alone leaves you logit-bound. That is why the two levers belong together, and why the chapter's headline claim is about the *checkpointed* configuration.

    (c) Raise the micro-batch — you are using 13% of the card, and a bigger micro-batch partly pays back checkpointing's ~+30% step time by improving matmul occupancy. The constraint is PLAN §5's **0.5M-token effective batch**: every doubling of the micro-batch must halve the gradient-accumulation count, so the optimizer sees the identical batch on all three tiers. Increase until tokens/s stops improving, then stop.

**5.** Implement the "one model, config-selected mixers" pattern, and then close the gap the chapter leaves open. (a) Add a `mixer` field to `StackConfig` and rewrite `Block` so it constructs `Attention`, `MLAttention`, or `GatedShortConv` from config, with the training loop unchanged; extend it to an LFM2-style hybrid (mostly conv, attention every $k$-th layer). (b) `MLAttention` currently cannot decode incrementally. Write the cache class it needs and state precisely why `KVCache` cannot be reused.

??? note "Solution"
    (a) The key move is that every mixer accepts the **same** call signature — `(x, cos, sin, attn_mask=, kv_cache=, start_pos=, record=)` — so `Block.forward` never branches:

    ```python
    @dataclass
    class StackConfig:
        # ... all existing fields unchanged ...
        mixer: str = "gqa"          # {"gqa", "mla", "conv"}
        attn_every: int = 0         # >0: LFM2-style hybrid, attention every k-th layer

    def make_mixer(cfg: StackConfig, layer_idx: int) -> nn.Module:
        kind = cfg.mixer
        if cfg.attn_every > 0:      # conv everywhere EXCEPT every attn_every-th layer
            kind = "gqa" if ((layer_idx + 1) % cfg.attn_every == 0) else "conv"
        if kind == "gqa":  return Attention(cfg, layer_idx)
        if kind == "mla":  return MLAttention(cfg, layer_idx)   # owns its own d_rope RoPE
        if kind == "conv": return GatedShortConv(cfg)           # absorbs cos/sin via *args
        raise ValueError(f"unknown mixer: {kind!r}")

    class Block(nn.Module):
        def __init__(self, cfg: StackConfig, layer_idx: int):
            super().__init__()
            self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
            self.attn = make_mixer(cfg, layer_idx)   # name kept: state-dict keys stay stable
            self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
            self.mlp = SwiGLU(cfg)
    ```

    `Stack100M` needs **no edits**, so ablating architectures is a one-field change re-running the same Ch. 14.7 loop and Ch. 14.5 ladder. Keeping the attribute name `self.attn` means norms and MLPs still load across mixer variants. One honest caveat the factory must not hide: `GatedShortConv` has no positional encoding *and* no long-range mixing, so a pure-conv stack is not a usable language model — LFM2's recipe is a hybrid, which is what `attn_every` gives you.

    (b) `KVCache` stores `(n_layers, B, n_kv, max_seq, head_dim)` for K and V — the wrong tensors entirely. MLA caches a $d_c$-dimensional **latent** shared by all heads plus **one** $d_r$-dimensional rotary key, and reconstructs per-head K and V from them; there is no per-head K or V to store. Hence:

    ```python
    class LatentKVCache:
        """MLA cache: (n_layers, B, max_seq, d_c) latents + (n_layers, B, max_seq, d_r) keys.
        Per token per layer: (d_c + d_r) * 2 bytes = 320 B, vs GQA's 512 B."""
        def __init__(self, n_layers, B, max_seq, d_c, d_r, device, dtype=torch.bfloat16):
            self.c = torch.zeros(n_layers, B, max_seq, d_c, device=device, dtype=dtype)
            self.kr = torch.zeros(n_layers, B, max_seq, d_r, device=device, dtype=dtype)
            self.max_seq = max_seq

        def update(self, layer_idx, c_kv, k_r, start_pos):
            T = c_kv.shape[1]
            assert start_pos + T <= self.max_seq
            self.c[layer_idx, :, start_pos:start_pos + T] = c_kv.to(self.c.dtype)
            self.kr[layer_idx, :, start_pos:start_pos + T] = k_r.to(self.kr.dtype)
            return (self.c[layer_idx, :, :start_pos + T],
                    self.kr[layer_idx, :, :start_pos + T])
    ```

    In `MLAttention.forward`, call `c_kv, k_r_flat = kv_cache.update(self.layer_idx, c_kv, k_r.squeeze(1), start_pos)` immediately after the RoPE step (cache the *rotated* $k_r$, exactly as GQA caches rotated K), then reshape `k_r_flat` back to `(B,1,S,dr)` before the concatenation. Until you add this, `generate()` should assert `cfg.mixer != "mla"` rather than silently producing a wrong result.

**6.** Verify the MLA absorption identity numerically, then use it to explain the decode-cost claim. (a) For one head, show in code that $q_h^\top (W_{UK}^{(h)} c) = (W_{UK}^{(h)\top} q_h)^\top c$ to fp32 round-off. (b) Count the per-token per-layer decode FLOPs for the content-logit path with and without absorption, at $d_c = 128$, $H = 8$, $d_h = 64$, cache length $S$. (c) The chapter says MLA's *memory* win is only 1.6× here. Given (b), what is the argument for MLA at 100M — and against?

??? note "Solution"
    (a) The identity is just associativity of the matrix product; the point of writing it is that the *cached* tensor changes side.

    ```python
    torch.manual_seed(0)
    H, dh, dc, S = 8, 64, 128, 17
    W_uk = torch.randn(H, dh, dc)          # per-head up-projection block
    q    = torch.randn(H, dh)              # one query, per head
    c    = torch.randn(S, dc)              # cached latents

    naive    = torch.einsum('hd,hdc,sc->hs', q, W_uk, c)   # up-project keys, then dot
    absorbed = torch.einsum('hd,hdc->hc', q, W_uk) @ c.T   # absorb into the query, then dot
    assert torch.allclose(naive, absorbed, atol=1e-4)
    ```

    (b) Without absorption you must up-project the whole cache each step ($2 S d_c H d_h$ FLOPs) and then dot per head ($2 S H d_h$): at $S=2048$ that is $2.68\times10^8 + 2.1\times10^6 \approx 2.70\times10^8$, and it grows linearly in $S$ with a large constant. With absorption you project the *query* into latent space once ($2 H d_h d_c = 1.31\times10^5$, independent of $S$) and dot it against the cached latents ($2 S H d_c = 4.19\times10^6$): $\approx 4.3\times10^6$ total, a **~60× reduction** in the content-logit path. The structural point matters more than the constant: absorption removes the $O(S\, d_c H d_h)$ up-projection entirely, which is what would otherwise have made MLA *slower* than GQA at long context.

    (c) **For:** absorption makes decode read a 160-value-per-token cache instead of a 256-value one *and* replaces per-head key materialization with a single latent dot — the cost model that lets DeepSeek serve enormous contexts at high concurrency. **Against, at 100M:** our GQA baseline is already only 2 KV heads of dim 64, so the compression ratio is 1.6× on a 30 MiB cache; MLA adds five projection matrices, a second RoPE table, a bespoke cache class, and takes you off every stock kernel and serving runtime. It belongs in this chapter as a mechanism to understand and measure, and in your model only if you intend to serve long context at scale.

**7.** You retrofit a KV cache into `Attention` and, on the decode path, keep calling `F.scaled_dot_product_attention(q, k, v, is_causal=True)` with `q_len = 1`, `kv_len = N`. Write the single test that would have caught this, explain why it is a *sufficient* oracle, and say what its stronger sibling adds.

??? note "Solution"
    **Greedy generation must be cache-invariant.** One assertion:

    ```python
    torch.manual_seed(0); a = model.generate(prompt, 16, temperature=0.0, use_cache=True)
    torch.manual_seed(0); b = model.generate(prompt, 16, temperature=0.0, use_cache=False)
    assert torch.equal(a, b)
    ```

    It is sufficient because the `use_cache=False` path recomputes the full prefix every step and therefore only ever exercises the square $q_{\text{len}} = kv_{\text{len}}$ case — where top-left and bottom-right causal alignment coincide and the mask is the ordinary lower triangle. That path is slow but unambiguously correct, so it is an independent oracle rather than a self-consistency check. Any cache, mask, or `position_ids` bug makes the two diverge, usually within two or three tokens. Greedy decoding (`temperature=0.0`) is essential: with sampling, small logit differences would be masked by RNG or would diverge for benign reasons.

    Its stronger sibling — test 5 in the CI block, `full[:, k:] == incremental` on raw *logits* — adds **localization**. Token-level equality tells you *that* something is wrong; logit-level equality lets you bisect by layer (run the comparison on the hidden state after each block) and typically points at the exact sublayer within a minute. Keep both: the generation test is the one that must never be deleted, the logit test is the one you reach for when it fails.

    Worth internalizing *why* this bug survives to production: training never exercises the rectangular case, so loss curves, eval perplexity and gradients are all correct. The failure lives exclusively in the incremental-decode path, which is also the only path your users see.
