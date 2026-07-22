# 2.6 The Transformer Block: Norms, Residuals, MLPs & Activations

Every modern large language model is, at its heart, a stack of identical *transformer blocks*. Whether you are reading the weights of GPT-4, Llama 3, Gemma 2, or Mistral, the same four-element recipe repeats dozens or hundreds of times: a normalization step, a self-attention sublayer, another normalization step, and a feed-forward network (FFN) sublayer — all wired together through residual connections. Getting this wiring right is not a detail. It is the reason transformers train stably at scale when many predecessor architectures did not.

This chapter dissects every component of the transformer block from first principles. We start with the residual stream — the conceptual backbone — then cover the two normalization variants (LayerNorm and RMSNorm), the critical pre-norm versus post-norm distinction, the FFN/MLP sublayer, and modern activation functions (ReLU, GELU, SwiGLU, GeGLU). We close with dropout, the complete block wiring diagram, a heavily commented implementation, and worked numerical examples. If you have already read [The Attention Mechanism From Scratch](../02-transformer/03-attention-from-scratch.html) and [Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html), this chapter completes the picture of how a single layer is assembled. [Building a GPT From Scratch (nanoGPT-style)](../02-transformer/07-build-gpt-from-scratch.html) then stacks these blocks into a full model.

---

## The Residual Stream

Before any formula, the most important mental model: a transformer block does not transform its input — it *adds a correction* to it.

{{fig:transformer-block}}

Formally, if the input to a block is $\mathbf{x} \in \mathbb{R}^{T \times d}$ (a sequence of $T$ vectors of dimension $d$), and the sublayer function is $F$, the output is:

$$
\mathbf{x}' = \mathbf{x} + F(\mathbf{x})
$$

This is a **residual connection** (skip connection), introduced in ResNets (He et al., *Deep Residual Learning for Image Recognition*, 2015) and adopted wholesale by Vaswani et al. in the original transformer. The residual pattern appears twice per block: once around attention, once around the FFN.

### Why residuals matter

Without residuals, a 96-layer network requires every layer to cooperate perfectly to pass signal from input to output. A single near-zero weight matrix suffocates the gradient and the layer "dies." With residuals, the *identity path* is always open; gradients flow backwards through it unimpeded, and each sublayer only needs to learn a small corrective delta. This is why deep transformers converge while deep vanilla MLPs of the same depth do not.

A useful way to think about this: the model maintains a *residual stream* — a single vector of dimension $d$ that flows from the embedding layer through every block to the output projection. Each attention sublayer and each FFN sublayer *reads from* the residual stream and *writes a small update back to it*. Mechanistic interpretability research (Elhage et al., *A Mathematical Framework for Transformer Circuits*, 2021) formalizes this view and shows that the residual stream is the primary communication channel between layers.

### Gradient flow through residuals

For a depth-$L$ network, the gradient of the loss $\mathcal{L}$ with respect to the input $\mathbf{x}_0$ expands as:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}_0} = \prod_{l=1}^{L} \left(I + \frac{\partial F_l}{\partial \mathbf{x}_{l-1}}\right) \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L}
$$

Because each factor contains the identity matrix $I$, even if the Jacobians $\frac{\partial F_l}{\partial \mathbf{x}_{l-1}}$ are small (near zero at initialization), the product never vanishes. Contrast this with a plain chain $\prod_l \frac{\partial F_l}{\partial \mathbf{x}_{l-1}}$, which suffers exponential vanishing or explosion.

---

## Layer Normalization

The second pillar of block stability is normalization. We normalize the activations to prevent the mean and variance of the residual stream from drifting arbitrarily large, which would cause saturated activations, exploding attention logits, and loss spikes.

### LayerNorm

**Layer Normalization** (Ba et al., *Layer Normalization*, 2016) normalizes across the *feature* dimension for each token independently:

$$
\text{LayerNorm}(\mathbf{x}) = \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \boldsymbol{\gamma} + \boldsymbol{\beta}
$$

where, for a single token vector $\mathbf{x} \in \mathbb{R}^d$:

$$
\mu = \frac{1}{d}\sum_{i=1}^{d} x_i, \qquad \sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2
$$

The learnable parameters $\boldsymbol{\gamma}, \boldsymbol{\beta} \in \mathbb{R}^d$ (called *scale* and *shift* or *weight* and *bias*) allow the network to undo the normalization if that turns out to be optimal. Each token is normalized independently of other tokens in the sequence, so LayerNorm is invariant to batch size and to sequence position — critical properties for autoregressive models.

$\epsilon$ (typically $10^{-5}$ or $10^{-6}$) prevents division by zero and is also beneficial for numerical stability in low-precision regimes (see [Numerical Computing, Floating Point & Precision](../01-foundations/04-numerics-precision.html)).

### RMSNorm

**Root Mean Square Layer Normalization** (Zhang & Sennrich, *Root Mean Square Layer Normalization*, 2019) drops the mean-centering step entirely:

$$
\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})} \odot \boldsymbol{\gamma}, \qquad \text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}
$$

There is no $\boldsymbol{\beta}$ shift term. Llama, Llama 2, Llama 3, Mistral, Gemma, and most modern open-weight models use RMSNorm in place of LayerNorm.

**Why RMSNorm?** Two reasons:

1. *Speed.* RMSNorm requires one pass over the vector (to compute the squared sum) instead of two (one for the mean, one for the variance around the mean). In wall-clock terms the savings are modest but meaningful at the million-token-per-second throughputs of large training runs.
2. *Hypothesis: the shift is redundant.* If the residual stream already has a near-zero mean (which empirically it often does), the centering step wastes compute without helping. The scale parameter $\boldsymbol{\gamma}$ retains the expressive power to rescale each feature.

The two norms behave identically when $\mu \approx 0$, which is the common case during training.

!!! example "Worked example: LayerNorm vs RMSNorm on a small vector"

    Let $\mathbf{x} = [1.0,\ 3.0,\ -1.0,\ 5.0]$, $d = 4$, $\epsilon = 0$, $\boldsymbol{\gamma} = \mathbf{1}$, $\boldsymbol{\beta} = \mathbf{0}$.

    **LayerNorm:**
    $\mu = (1 + 3 - 1 + 5) / 4 = 2.0$

    $\sigma^2 = [(1-2)^2 + (3-2)^2 + (-1-2)^2 + (5-2)^2] / 4 = (1 + 1 + 9 + 9) / 4 = 5.0$

    $\text{std} = \sqrt{5} \approx 2.236$

    Output: $[-0.447,\ 0.447,\ -1.342,\ 1.342]$  — zero mean, unit variance.

    **RMSNorm:**
    $\text{RMS}(\mathbf{x}) = \sqrt{(1 + 9 + 1 + 25) / 4} = \sqrt{9} = 3.0$

    Output: $[0.333,\ 1.0,\ -0.333,\ 1.667]$  — rescaled but *not* zero-mean.

    Note the difference: RMSNorm preserves the offset structure of the vector, only rescaling its overall magnitude.

{{fig:tblock-layernorm-vs-rmsnorm}}

---

## Pre-Norm vs Post-Norm

The *position* of the normalization within the residual block is as important as the choice of norm. There are two canonical layouts:

{{fig:tblock-pre-vs-post-norm}}

In post-norm, normalization is applied *after* the residual addition: $\mathbf{x}' = \text{Norm}(\mathbf{x} + F(\mathbf{x}))$. In pre-norm, normalization is applied *before* the sublayer: $\mathbf{x}' = \mathbf{x} + F(\text{Norm}(\mathbf{x}))$.

### Why pre-norm dominates modern training

The original transformer used post-norm with learning rate warmup because post-norm is unstable at initialization. Here is the mechanical reason:

At initialization $F(\mathbf{x}) \approx \mathbf{0}$, so $\mathbf{x} + F(\mathbf{x}) \approx \mathbf{x}$. The variance of the residual sum is dominated by the variance of the skip path. In post-norm, the normalization then divides by this variance, which is fine. But the *gradients* of the loss with respect to the pre-norm input scale as $1/\text{std}$, and std can vary wildly early in training when the sublayer outputs are small. This creates extremely large gradients through the normalization, requiring careful warmup to survive.

In pre-norm, the normalization acts on $\mathbf{x}$ *before* $F$, stabilizing the input to $F$ regardless of what $F$ outputs. Crucially, the residual skip path ($\mathbf{x}$ itself, unnormalized) always contributes a unit-variance gradient path backward. This means pre-norm transformers train stably even without warmup and tolerate much larger learning rates (Xiong et al., *On Layer Normalization in the Transformer Architecture*, 2020).

A subtlety worth knowing: because the last block's output is not normalized before the final linear projection in pre-norm, most modern architectures add a *final LayerNorm/RMSNorm* after the last transformer block. GPT-2 (`ln_f`), Llama (`norm`), and others all do this.

!!! interview "Interview Corner"

    **Q:** Why do modern LLMs like Llama use pre-norm with RMSNorm instead of post-norm with LayerNorm as in the original transformer paper?

    **A:** Two independent improvements were combined. Pre-norm (normalizing before the sublayer rather than after) places the normalization on the input rather than on the residual sum, which stabilizes gradients at initialization and removes the need for careful learning rate warmup. The identity skip path in pre-norm guarantees a clean gradient highway of magnitude 1, whereas post-norm gradients scale as $1/\text{std}$ of the residual sum, which can be large and noisy early in training. RMSNorm replaces LayerNorm for efficiency: it drops the mean-centering step (one fewer pass over the vector), uses no bias parameter, and achieves near-identical training loss in practice. Together, these two changes make training faster and more robust without any measurable quality loss.

---

## The Feed-Forward Network (FFN) Sublayer

The FFN (also called the MLP sublayer) provides the "storage" and nonlinear processing complement to the "routing" performed by attention. Each token's residual stream vector is processed *independently* — there is no cross-token interaction in the FFN, making it embarrassingly parallelizable along the sequence dimension.

### Standard two-layer FFN

The classic FFN is a two-layer MLP with an inner dimension $d_\text{ff}$:

$$
\text{FFN}(\mathbf{x}) = W_2 \cdot \phi(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2
$$

where $W_1 \in \mathbb{R}^{d_\text{ff} \times d}$, $W_2 \in \mathbb{R}^{d \times d_\text{ff}}$, and $\phi$ is a pointwise activation function. The original transformer used $d_\text{ff} = 4d$, and this $4\times$ ratio remains the most common choice. For GPT-3 with $d = 12{,}288$, the FFN expansion is $d_\text{ff} = 49{,}152$.

The FFN accounts for roughly two-thirds of the total parameter count of a decoder-only transformer (attention contributes roughly one-third), making it the single biggest parameter block. Research on FFNs as key-value memories (Geva et al., *Transformer Feed-Forward Layers Are Key-Value Memories*, 2021) suggests each FFN neuron stores a pattern-to-value association: the first matrix $W_1$ identifies patterns and the second matrix $W_2$ retrieves associated information.

### Parameter count worked example

For a Llama 2 7B block with $d = 4{,}096$ and the gated FFN described below with $d_\text{ff} = 11{,}008$:

- $W_\text{gate}$: $4{,}096 \times 11{,}008 = 45.1\text{M}$ parameters
- $W_\text{up}$: $4{,}096 \times 11{,}008 = 45.1\text{M}$ parameters
- $W_\text{down}$: $11{,}008 \times 4{,}096 = 45.1\text{M}$ parameters
- Total per FFN: $\approx 135\text{M}$

Multiplied across 32 blocks: $\approx 4.3\text{B}$ parameters — about 62% of the model's 7B total.

{{fig:tblock-parameter-budget}}

---

## Activation Functions: From ReLU to SwiGLU

### ReLU and its successors

The original transformer used ReLU: $\phi(x) = \max(0, x)$. ReLU is fast, sparse, and interpretable, but it produces dead neurons (units that output exactly zero for all inputs after a bad gradient update), which can reduce effective capacity.

**GELU** (Gaussian Error Linear Unit; Hendrycks & Gimpel, 2016) smooths the hard zero threshold:

$$
\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right]
$$

where $\Phi$ is the cumulative distribution function of the standard normal. GELU is approximately $x\sigma(1.702x)$ and can be efficiently approximated as $0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$. BERT, GPT-2, and GPT-3 all used GELU. Its smooth, stochastic-looking gate (the unit "decides" probabilistically whether to propagate the input) empirically outperforms ReLU on most language tasks.

### Gated Linear Units and SwiGLU

A class of *gated* activations uses an elementwise product to implement a soft gate:

$$
\text{GLU}(x, W, V, b, c) = \sigma(xW + b) \odot (xV + c)
$$

where $\sigma$ is the sigmoid. Dauphin et al. (*Language Modeling with Gated Convolutional Networks*, 2017) introduced GLUs for convolutions; the idea transfers cleanly to transformers.

**SwiGLU** (Shazeer, 2020) replaces sigmoid with Swish ($\text{Swish}(x) = x \cdot \sigma(x) = x / (1 + e^{-x})$):

$$
\text{SwiGLU}(x, W, V) = \text{Swish}(xW) \odot (xV)
$$

The full SwiGLU FFN sublayer thus has *three* weight matrices instead of two:

$$
\text{FFN}_\text{SwiGLU}(\mathbf{x}) = W_\text{down}\big(\text{Swish}(W_\text{gate}\mathbf{x}) \odot (W_\text{up}\mathbf{x})\big)
$$

To keep parameter count and FLOPs comparable to the standard $4d$ expansion, the inner dimension is reduced to $\frac{2}{3} \times 4d \approx \frac{8d}{3}$, then rounded up to a hardware-friendly multiple. Llama 2 7B rounds $\frac{8 \times 4096}{3} = 10{,}922.7$ up to a multiple of 256 (its `multiple_of` hyperparameter), giving $d_\text{ff} = 11{,}008$. Rounding the same $8d/3$ target up to a multiple of 64 instead — as the reference implementation later in this chapter does — yields $d_\text{ff} = 10{,}944$; it is the identical recipe with a different alignment constant.

**GeGLU** is the same idea with GELU instead of Swish: $\text{GeGLU}(x, W, V) = \text{GELU}(xW) \odot (xV)$. Gemma 2 uses GeGLU.

### Why gated activations work

The intuition: the gate $\text{Swish}(W_\text{gate}\mathbf{x})$ can *suppress* entire features (output near zero) when the input pattern is not relevant, while the value path $W_\text{up}\mathbf{x}$ determines *what* to write when the gate is open. This is conceptually similar to the forget/input gates of an LSTM, but computed in a single feedforward pass without recurrence. Empirically, SwiGLU and GeGLU consistently outperform GELU and ReLU at the same parameter count on language modeling benchmarks.

{{fig:tblock-activation-curves}}

---

## Dropout in the Transformer Block

Dropout (Srivastava et al., *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*, 2014) is applied at two points in the classic transformer block:

1. After the attention weights (before the weighted sum over values) — *attention dropout*.
2. After each sublayer's output, before the residual addition — *residual dropout*.

During pretraining of large models on large datasets, dropout is often set to 0.0 — the models are underfit, not overfit, and dropout hurts loss. GPT-3 used $p = 0.1$; Llama and subsequent models use $p = 0.0$ during pretraining and may introduce small dropout during fine-tuning.

If you train on small datasets or fine-tune with very few samples, residual dropout of 0.05–0.1 remains a useful regularizer. See [PEFT I: LoRA, QLoRA, DoRA & The Adapter Family](../05-posttraining-alignment/03-peft-lora-qlora.html) for fine-tuning configurations.

---

## The Complete Transformer Block: Wiring Diagram

Here is the full pre-norm transformer block with SwiGLU and RMSNorm, as used in the Llama family:

{{fig:tblock-complete-wiring}}

Note that in some implementations the attention output also passes through a projection dropout before the residual add. The final output has the same shape as the input, enabling stacking.

---

## Implementation: A Complete Transformer Block in PyTorch

```python
"""
transformer_block.py — A complete, heavily-commented transformer block
implementing the modern pre-norm + RMSNorm + SwiGLU + RoPE-ready design
used in the Llama / Mistral family of models.

Requires: torch >= 2.0
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# RMSNorm
# ─────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Normalizes by RMS(x) rather than by std(x − mean(x)).
    No bias term — only a learnable scale γ.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Learnable per-feature scale, initialized to 1
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, dim]  (or any shape ending in dim)
        # Compute RMS along last dimension
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


# ─────────────────────────────────────────────────────────────────────────────
# SwiGLU Feed-Forward Network
# ─────────────────────────────────────────────────────────────────────────────

class SwiGLUFFN(nn.Module):
    """
    Feed-forward sublayer using the SwiGLU gated activation (Shazeer, 2020).

    FFN(x) = W_down( Swish(W_gate x) ⊙ W_up x )

    The inner dimension is set to 8/3 * dim by convention so that the total
    FLOP count matches a standard 4× FFN with a plain activation.
    """

    def __init__(self, dim: int, hidden_dim: Optional[int] = None,
                 bias: bool = False, dropout: float = 0.0):
        super().__init__()
        if hidden_dim is None:
            # 8/3 * dim, rounded to nearest multiple of 64
            hidden_dim = int(8 * dim / 3)
            hidden_dim = 64 * ((hidden_dim + 63) // 64)

        # gate branch: produces the soft gate via Swish
        self.w_gate = nn.Linear(dim, hidden_dim, bias=bias)
        # up projection: produces the values
        self.w_up   = nn.Linear(dim, hidden_dim, bias=bias)
        # down projection: projects back to model dimension
        self.w_down = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Swish(gate) ⊙ up — the two-branch gated activation
        gate = F.silu(self.w_gate(x))   # silu = Swish = x * sigmoid(x)
        up   = self.w_up(x)
        fused = gate * up                # elementwise gating
        return self.dropout(self.w_down(fused))


# ─────────────────────────────────────────────────────────────────────────────
# Minimal Multi-Head Self-Attention (for completeness; full version in ch 2.4)
# ─────────────────────────────────────────────────────────────────────────────

class MinimalMHA(nn.Module):
    """
    Causal multi-head self-attention. Minimal implementation for block wiring.
    For MQA, GQA, RoPE, FlashAttention, see chapter 2.4 and 2.5.
    """

    def __init__(self, dim: int, n_heads: int, bias: bool = False,
                 attn_dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv   = nn.Linear(dim, 3 * dim, bias=bias)
        self.proj  = nn.Linear(dim, dim,     bias=bias)
        self.attn_drop = nn.Dropout(attn_dropout)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        # Compute Q, K, V in one shot then split
        q, k, v = self.qkv(x).chunk(3, dim=-1)     # each: [B, T, C]

        # Reshape to [B, n_heads, T, head_dim]
        def reshape(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = map(reshape, (q, k, v))

        # Scaled dot-product attention with optional causal mask
        # torch.nn.functional.scaled_dot_product_attention uses FlashAttention
        # when available (torch >= 2.0 with CUDA).
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=(mask is None),   # if no explicit mask, use causal
        )  # [B, n_heads, T, head_dim]

        # Merge heads and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(attn_out)


# ─────────────────────────────────────────────────────────────────────────────
# The Transformer Block
# ─────────────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    One pre-norm transformer block as used in Llama / Mistral:

        x_attn = x     + Attention(RMSNorm(x))
        x_out  = x_attn + FFN(RMSNorm(x_attn))

    Parameters
    ----------
    dim         : model dimension (d)
    n_heads     : number of attention heads
    ffn_hidden  : inner FFN dimension (defaults to ⌊8d/3⌋ rounded to 64)
    bias        : whether to include bias in linear layers
    dropout     : residual dropout probability (0.0 for large-scale pretraining)
    norm_eps    : epsilon for RMSNorm
    """

    def __init__(self, dim: int, n_heads: int,
                 ffn_hidden: Optional[int] = None,
                 bias: bool = False,
                 dropout: float = 0.0,
                 norm_eps: float = 1e-6):
        super().__init__()
        # Normalization: applied BEFORE each sublayer (pre-norm)
        self.norm_attn = RMSNorm(dim, eps=norm_eps)
        self.norm_ffn  = RMSNorm(dim, eps=norm_eps)

        # Sublayers
        self.attn = MinimalMHA(dim, n_heads, bias=bias)
        self.ffn  = SwiGLUFFN(dim, ffn_hidden, bias=bias, dropout=dropout)

        # Residual dropout (applied after each sublayer, before addition)
        self.res_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # ── Attention sublayer (pre-norm residual) ─────────────────────────
        # Normalize first, pass through attention, add back to residual stream
        x = x + self.res_drop(self.attn(self.norm_attn(x), mask))

        # ── FFN sublayer (pre-norm residual) ───────────────────────────────
        # Normalize first, pass through FFN, add back to residual stream
        x = x + self.res_drop(self.ffn(self.norm_ffn(x)))
        # Note: self.ffn already applies internal dropout before returning

        return x   # shape unchanged: [batch, T, dim]


# ─────────────────────────────────────────────────────────────────────────────
# Stack of blocks (GPT-style)
# ─────────────────────────────────────────────────────────────────────────────

class TransformerStack(nn.Module):
    """
    N stacked transformer blocks with a final RMSNorm.
    This is the 'trunk' of a decoder-only LLM.
    """

    def __init__(self, dim: int, n_heads: int, n_layers: int,
                 ffn_hidden: Optional[int] = None,
                 bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, ffn_hidden, bias, dropout)
            for _ in range(n_layers)
        ])
        # Final norm before the lm_head projection
        self.norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check and parameter count
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    # Configuration roughly matching Llama 2 7B single-block sizes
    dim, n_heads, n_layers = 4096, 32, 32

    # Single block
    block = TransformerBlock(dim=dim, n_heads=n_heads)
    block_params = sum(p.numel() for p in block.parameters())
    print(f"Single block parameters: {block_params / 1e6:.1f}M")

    # Micro model for shape check
    model = TransformerStack(dim=512, n_heads=8, n_layers=4)
    x = torch.randn(2, 128, 512)   # batch=2, seq=128, dim=512
    out = model(x)
    print(f"Input shape:  {x.shape}")   # [2, 128, 512]
    print(f"Output shape: {out.shape}") # [2, 128, 512]

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Small model parameters: {total_params / 1e6:.2f}M")
```

Running the sanity-check section above with the Llama 2 7B single-block dimensions prints approximately **201.6M** parameters per block: attention $4d^2 = 67.1\text{M}$ (fused QKV $3d^2$ plus the output projection $d^2$) plus the SwiGLU FFN $3 \cdot d \cdot d_\text{ff} = 3 \cdot 4096 \cdot 10{,}944 = 134.5\text{M}$ (this chapter's code rounds $8d/3$ up to a multiple of 64, giving $d_\text{ff}=10{,}944$), plus two RMSNorms ($2 \cdot 4096$, negligible). To sanity-check against the full model, multiply $201.6\text{M} \times 32 \approx 6.45\text{B}$ and add the two $32{,}000 \times 4096 \approx 131\text{M}$ embedding matrices (input token embedding plus the untied LM head), giving $\approx 6.71\text{B}$ — close to Llama 2 7B's true 6.74B. The small residual gap is the FFN alignment constant: the real Llama 2 uses `multiple_of=256`, which rounds $8d/3$ up to $d_\text{ff}=11{,}008$ (not 10,944), raising each block to $\approx 202.4\text{M}$ and the 32-block trunk to $\approx 6.48\text{B}$; adding the $\approx 0.26\text{B}$ of embeddings recovers the reported 6.74B exactly. Note that the naive '$7\text{B}/32 \approx 219\text{M}$ per layer' estimate is misleading precisely because the nominal 7B includes $\approx 0.26\text{B}$ of embedding parameters that live outside the transformer blocks.

---

## Numerical Stability & Precision Considerations

Understanding the block's numerical behavior is essential for training at scale.

### Pre-norm keeps the norm bounded

At layer $l$, the residual stream has (empirically) roughly unit variance after the final norm. Because we normalize *before* the sublayer, the sublayer always sees a well-conditioned input. The output of the sublayer is added back to the (un-normalized) residual stream, whose variance grows slowly as $\mathcal{O}(\sqrt{l})$ in theory (as a sum of independent random variables). In practice with careful initialization (weight std $\propto 1/\sqrt{d}$ or with the "scaled init" used in GPT-2), growth is much slower.

### Initialization scaling for deep stacks

Radford et al. (*Language Models are Unsupervised Multitask Learners* / GPT-2, 2019) and Shoeybi et al. (*Megatron-LM*, 2019) note that naive Xavier/Kaiming initialization for a deep stack can produce variance blowup in the residual stream. GPT-2 addresses this by scaling the residual output projections by $1/\sqrt{N}$ (with $N$ the number of residual layers), and Megatron-LM uses the equivalent $1/\sqrt{2L}$ factor (two residual additions per block over $L$ blocks). The fix: scale the output projections of each sublayer by $1/\sqrt{2L}$ (where $L$ is the total number of layers), so that the $L$ residual contributions have unit total variance. This is the `init_scale` trick in many industrial implementations.

### bfloat16 and overflow in the FFN

The inner FFN activations after the gating product can have large magnitudes (on the order of 10–100 at the start of training). In float16 this overflows to `inf`; in bfloat16 the larger dynamic range handles it. This is one reason most modern LLM pretraining uses bfloat16. See [Numerical Computing, Floating Point & Precision](../01-foundations/04-numerics-precision.html) and [Mixed Precision, bf16 & FP8 Training](../03-pretraining/08-mixed-precision-fp8.html).

!!! example "Worked example: Residual stream variance growth"

    Suppose each sublayer output has norm approximately $\|\delta_l\| \approx c$ for some small constant $c$. After $L$ blocks the residual stream norm is:

    $$\|\mathbf{x}_L\| \approx \|\mathbf{x}_0\| + L \cdot c$$

    For $L = 32$, $\|\mathbf{x}_0\| \approx \sqrt{d} = \sqrt{4096} = 64$ (a unit-normal $d$-dim vector), and $c \approx 2$ (a rough estimate from empirical norms early in training), we get $\|\mathbf{x}_{32}\| \approx 64 + 64 = 128$. Pre-norm normalizes this back to $\approx 1$ before each sublayer input, so each sublayer sees a clean signal despite the stream growing. Post-norm would normalize *after* the addition, applying different normalization constants at each block, which has been observed to interact poorly with gradient flow.

---

## The Block's Role in Mechanistic Interpretability

Understanding the block helps you reason about what goes wrong (and right) during training. A few practitioner observations:

**Attention is communication; FFN is computation.** Attention moves information between token positions. The FFN processes each token independently, applying nonlinear transformations that empirically recall factual associations. This division of labor is why sparse MoE architectures (see [Mixture-of-Experts (MoE) Architectures](../02-transformer/09-mixture-of-experts.html)) replace only the FFN with a mixture of expert networks — the attention mechanism is shared.

**The residual stream accumulates structure.** Early layers tend to refine token-level features; later layers build task-level representations. This has been exploited in layer-selective fine-tuning (LoRA applied only to certain layers) and in early-exit inference (stopping at layer $k$ rather than $L$).

**Gradient checkpointing interacts with block boundaries.** Because each block is a self-contained module, gradient checkpointing can recompute activations at block granularity — recompute one block's activations during the backward pass rather than storing them. This is the standard memory-efficiency technique described in [Memory-Efficient Training: Checkpointing, Offloading & LoRA Math](../04-kernels-efficiency/10-memory-efficient-training.html).

---

## Architecture Variants and Modern Improvements

The four-component pre-norm block described above is a stable baseline. Modern architectures iterate on it in several ways.

### Parallel attention and FFN

Google's PaLM (Chowdhery et al., 2022) and GPT-NeoX (Black et al., 2022) run the attention and FFN sublayers *in parallel* rather than sequentially:

$$
\mathbf{x}' = \mathbf{x} + \text{Attn}(\text{Norm}(\mathbf{x})) + \text{FFN}(\text{Norm}(\mathbf{x}))
$$

This saves one residual addition and one normalization call, and allows fusing the attention QKV projection with the FFN $W_\text{gate}$/$W_\text{up}$ projections into a single large matrix multiply — a throughput win on hardware with slow memory bandwidth relative to compute. The gradient flow is slightly different but empirically yields similar quality.

### DeepNorm

DeepNorm (Wang et al., 2022) scales the residual before adding:

$$
\mathbf{x}' = \text{Norm}(\alpha \mathbf{x} + F(\mathbf{x}))
$$

with $\alpha > 1$, combined with a scaled initialization. The authors prove that this keeps the expected update to the model bounded, enabling stable training of post-norm transformers at depths up to 1000 layers.

### Sandwich norm and dual residuals

Some architectures apply a second normalization after the sublayer function and before the addition (both pre- and post-norm). This "sandwich norm" trades compute for additional stability and was found beneficial in certain ultra-deep configurations.

For more on these and other architectural choices, see [Modern Architecture Improvements & Design Choices](../02-transformer/10-modern-arch-improvements.html).

---

## Key Takeaways

!!! key "Key Takeaways"

    - The transformer block has a simple four-element structure: Norm → Attention → Norm → FFN, all wired with residual connections. The residual stream is the block's backbone and the primary gradient highway.
    - Pre-norm (normalizing before the sublayer) is preferred over post-norm because it stabilizes gradients at initialization without warmup: the skip path always contributes a unit-variance gradient path backward.
    - RMSNorm is preferred over LayerNorm in modern models: it drops mean-centering (one fewer pass over the activations), removes the bias term, and achieves near-identical empirical quality while being marginally faster.
    - SwiGLU (and GeGLU) outperform plain ReLU and GELU on language tasks by adding a multiplicative gate that suppresses irrelevant features while passing relevant ones through. They require three weight matrices instead of two and conventionally use $d_\text{ff} = \frac{8d}{3}$ to match FLOPs.
    - The FFN accounts for roughly two-thirds of transformer parameters (at the standard $4\times$ or $\frac{8}{3}\times$ expansion ratio) and processes each token independently — making it the primary site of factual storage.
    - Dropout is typically 0.0 during large-scale pretraining (data is more abundant than model capacity), but 0.05–0.1 is useful for fine-tuning on small datasets.
    - Deep stacks benefit from careful output-projection scaling ($1/\sqrt{2L}$) to prevent residual stream variance blowup. Using bfloat16 (rather than float16) avoids FFN activation overflow.
    - The final RMSNorm after the last block is essential in pre-norm architectures: without it, the last block's output is un-normalized before the language-model head projection.

---

!!! sota "State of the Art & Resources (2026)"
    The pre-norm + RMSNorm + SwiGLU transformer block is the settled standard for large-scale LLM training as of 2026, with Llama 3, Gemma 2, Mistral, and most frontier models converging on this design. Active research has shifted toward stability at extreme depth (1000+ layers), understanding what FFN neurons actually store, and architectural variants such as parallel attention-FFN blocks and sparse MoE substitutions for the FFN.

    **Foundational work**

    - [Ba et al., *Layer Normalization* (2016)](https://arxiv.org/abs/1607.06450) — introduced per-token feature-dimension normalization; the baseline every modern norm is compared against.
    - [Zhang & Sennrich, *Root Mean Square Layer Normalization* (2019)](https://arxiv.org/abs/1910.07467) — dropped mean-centering to get RMSNorm; adopted by Llama, Mistral, Gemma, and most post-2022 models.
    - [Shazeer, *GLU Variants Improve Transformer* (2020)](https://arxiv.org/abs/2002.05202) — introduced SwiGLU and GeGLU; the paper behind Llama's three-matrix gated FFN design.
    - [Xiong et al., *On Layer Normalization in the Transformer Architecture* (2020)](https://arxiv.org/abs/2002.04745) — proved theoretically why pre-norm stabilizes gradients at initialization and removes the need for warmup.

    **Recent advances (2023–2026)**

    - [Grattafiori et al., *The Llama 3 Herd of Models* (2024)](https://arxiv.org/abs/2407.21783) — canonical modern reference for pre-norm + RMSNorm + SwiGLU + GQA at scale (8B–405B parameters).
    - [Gemma Team, *Gemma 2: Improving Open Language Models at a Practical Size* (2024)](https://arxiv.org/abs/2408.00118) — GeGLU with interleaved local-global attention and post+pre dual-norm; competitive with models 2–3× larger.
    - [Wang et al., *DeepNet: Scaling Transformers to 1,000 Layers* (2022)](https://arxiv.org/abs/2203.00555) — DeepNorm residual scaling with a theoretically bounded update rule; shows post-norm can be stable at extreme depth with the right init.

    **Mechanistic understanding**

    - [Geva et al., *Transformer Feed-Forward Layers Are Key-Value Memories* (2021)](https://arxiv.org/abs/2012.14913) — reframes each FFN neuron as a key-value memory storing pattern-to-vocabulary associations; foundational for interpretability of the FFN sublayer.
    - [Elhage et al., *A Mathematical Framework for Transformer Circuits* (2021)](https://transformer-circuits.pub/2021/framework/index.html) — formalizes the residual stream as a shared communication channel read and written by every sublayer; essential vocabulary for reasoning about block internals.

    **Open-source & tools**

    - [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — ~300-line readable PyTorch GPT implementation; the clearest reference for transformer block wiring in code.
    - [EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox) — production-grade multi-GPU training library (Megatron + DeepSpeed); supports RMSNorm, RoPE, flash attention, and MoE out of the box.

## Further Reading

- Vaswani et al., *Attention Is All You Need* (NeurIPS 2017) — original post-norm transformer.
- Ba et al., *Layer Normalization* (arXiv 2016) — the canonical LayerNorm paper.
- Zhang & Sennrich, *Root Mean Square Layer Normalization* (NeurIPS 2019) — RMSNorm.
- Xiong et al., *On Layer Normalization in the Transformer Architecture* (ICML 2020) — theoretical analysis of pre-norm stability.
- Hendrycks & Gimpel, *Gaussian Error Linear Units (GELUs)* (arXiv 2016) — GELU activation.
- Dauphin et al., *Language Modeling with Gated Convolutional Networks* (ICML 2017) — GLU activations.
- Shazeer, *GLU Variants Improve Transformer* (arXiv 2020) — SwiGLU and GeGLU; the paper underpinning Llama's FFN design.
- He et al., *Deep Residual Learning for Image Recognition* (CVPR 2016) — origin of residual connections.
- Geva et al., *Transformer Feed-Forward Layers Are Key-Value Memories* (EMNLP 2021) — mechanistic interpretation of the FFN.
- Elhage et al., *A Mathematical Framework for Transformer Circuits* (Anthropic, 2021) — the residual stream framing of transformer computation.
- Touvron et al., *Llama 2* (Meta AI, 2023) — practical reference for the pre-norm + RMSNorm + SwiGLU design.

---

## Exercises

**1.** (Conceptual) A colleague builds a 96-layer decoder-only transformer but, to "simplify" the block, removes both residual connections so that each block computes $\mathbf{x}' = \text{FFN}(\text{Norm}(\text{Attn}(\text{Norm}(\mathbf{x}))))$ with no skip paths. Training loss immediately plateaus and never improves. Using the gradient-flow argument from this chapter, explain why. What single term in the backward-pass product is responsible for the identity "gradient highway," and why does removing residuals reintroduce the vanishing-gradient problem?

??? note "Solution"
    With residuals, the gradient of the loss with respect to the block input expands as

    $$
    \frac{\partial \mathcal{L}}{\partial \mathbf{x}_0} = \prod_{l=1}^{L} \left(I + \frac{\partial F_l}{\partial \mathbf{x}_{l-1}}\right) \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L}.
    $$

    The load-bearing term is the identity matrix $I$ inside each factor. Even when every sublayer Jacobian $\frac{\partial F_l}{\partial \mathbf{x}_{l-1}}$ is near zero (which is exactly the case at initialization, where $F(\mathbf{x}) \approx \mathbf{0}$), each factor is still approximately $I$, so their product stays close to $I$ and gradient signal reaches $\mathbf{x}_0$ undiminished.

    Removing the residuals deletes the $I$ from every factor, so the product collapses to the plain chain

    $$
    \prod_{l=1}^{L} \frac{\partial F_l}{\partial \mathbf{x}_{l-1}}.
    $$

    This is a product of 96 Jacobians whose singular values are generically not exactly 1. If they are typically below 1 the product shrinks exponentially (vanishing gradients); if above 1 it grows exponentially (exploding gradients). At 96 layers the vanishing case wins with overwhelming probability at initialization: the early layers receive essentially zero gradient, cannot learn, and the loss plateaus. This is precisely the failure mode that residual connections were introduced to cure, and why deep vanilla stacks of the same depth do not converge while pre-norm residual transformers do.

**2.** (Quantitative) Let $\mathbf{x} = [2.0,\ -2.0,\ 4.0,\ 0.0]$ with $d = 4$, $\epsilon = 0$, $\boldsymbol{\gamma} = \mathbf{1}$, $\boldsymbol{\beta} = \mathbf{0}$. Compute by hand (a) the LayerNorm output and (b) the RMSNorm output. Then state, in one sentence, the structural difference you observe between the two outputs.

??? note "Solution"
    **(a) LayerNorm.**

    $\mu = (2 - 2 + 4 + 0)/4 = 1.0$

    $\sigma^2 = [(2-1)^2 + (-2-1)^2 + (4-1)^2 + (0-1)^2]/4 = [1 + 9 + 9 + 1]/4 = 20/4 = 5.0$

    $\text{std} = \sqrt{5} \approx 2.236$

    Output $= (\mathbf{x} - \mu)/\text{std}$:

    - $(2-1)/2.236 = 0.447$
    - $(-2-1)/2.236 = -1.342$
    - $(4-1)/2.236 = 1.342$
    - $(0-1)/2.236 = -0.447$

    LayerNorm output $= [0.447,\ -1.342,\ 1.342,\ -0.447]$ — mean 0, unit variance.

    **(b) RMSNorm.**

    $\text{RMS}(\mathbf{x}) = \sqrt{(2^2 + (-2)^2 + 4^2 + 0^2)/4} = \sqrt{(4 + 4 + 16 + 0)/4} = \sqrt{24/4} = \sqrt{6} \approx 2.449$

    Output $= \mathbf{x}/\text{RMS}$:

    - $2/2.449 = 0.816$
    - $-2/2.449 = -0.816$
    - $4/2.449 = 1.633$
    - $0/2.449 = 0.0$

    RMSNorm output $= [0.816,\ -0.816,\ 1.633,\ 0.0]$.

    **Structural difference:** LayerNorm re-centers the vector to zero mean before scaling, so the sign/offset pattern is shifted; RMSNorm only rescales magnitude and leaves the offset structure intact — note RMSNorm keeps the third-component-largest, the fourth exactly zero, and does not force the mean to zero (its output mean is $0.408$, not $0$).

**3.** (Conceptual) Pre-norm architectures leave the last block's output un-normalized on the residual stream. (a) Why do GPT-2, Llama, and friends therefore add a *final* norm (`ln_f` / `norm`) after the last block, and what specifically would degrade without it? (b) The original post-norm transformer required learning-rate *warmup* to train; pre-norm does not. Give the one-line mechanical reason, referencing how gradients scale in each layout.

??? note "Solution"
    **(a) Final norm.** In pre-norm, each block computes $\mathbf{x}' = \mathbf{x} + F(\text{Norm}(\mathbf{x}))$, so the normalization only ever touches the *input* to a sublayer — it never normalizes the value that leaves the last block. As the chapter's variance discussion notes, the residual stream grows across depth (roughly $\mathcal{O}(\sqrt{l})$, and up to a norm of $\approx 128$ in the worked $L=32$ example), so the final block emits a vector of large and layer-count-dependent magnitude. Feeding that directly into the language-model head would send poorly-conditioned, large-magnitude logits into the softmax, hurting stability and calibration. Inserting a final RMSNorm/LayerNorm rescales the stream back to a well-conditioned unit scale before the LM head projection, which is why every pre-norm model adds one.

    **(b) Warmup.** In post-norm the normalization sits on top of the residual sum, so the gradient with respect to the pre-norm input scales as $1/\text{std}$ of that sum; early in training $F(\mathbf{x}) \approx \mathbf{0}$ makes the sum's variance small and volatile, producing huge, noisy gradients that only slow warmup can survive. In pre-norm the unnormalized skip path always contributes a clean unit-variance gradient path backward regardless of what $F$ outputs, so gradients stay bounded from step one and no warmup is needed.

**4.** (Quantitative) You are sizing a SwiGLU FFN for a model with $d = 2048$, using this chapter's reference recipe: set the inner dimension to $\lfloor 8d/3 \rfloor$ rounded *up* to the nearest multiple of 64, and use no bias. (a) Compute $d_\text{ff}$. (b) Compute the total FFN parameter count. (c) Compare it against a standard two-matrix FFN with the classic $d_\text{ff} = 4d$ expansion, and comment on why the $8/3$ factor is chosen.

??? note "Solution"
    **(a) Inner dimension.**

    $8d/3 = 8 \times 2048 / 3 = 16384/3 = 5461.33\ldots$, so $\lfloor 8d/3 \rfloor = 5461$.

    Round up to a multiple of 64: $64 \times \lceil 5461/64 \rceil = 64 \times \lceil 85.33 \rceil = 64 \times 86 = 5504$.

    So $d_\text{ff} = 5504$.

    **(b) SwiGLU parameter count.** SwiGLU has three matrices ($W_\text{gate}, W_\text{up}: d \to d_\text{ff}$ and $W_\text{down}: d_\text{ff} \to d$), each with $d \cdot d_\text{ff}$ entries and no bias:

    $$
    3 \cdot d \cdot d_\text{ff} = 3 \times 2048 \times 5504 = 3 \times 11{,}272{,}192 = 33{,}816{,}576 \approx 33.8\text{M}.
    $$

    **(c) Comparison with the classic $4d$ FFN.** A standard two-matrix FFN with $d_\text{ff} = 4d = 8192$ has

    $$
    2 \cdot d \cdot d_\text{ff} = 2 \times 2048 \times 8192 = 33{,}554{,}432 \approx 33.6\text{M}.
    $$

    The two are essentially equal ($33.8\text{M}$ vs $33.6\text{M}$). That is exactly the point of the $8/3$ factor: SwiGLU spends a third matrix on the gate, so to hold parameters and FLOPs roughly fixed against the classic $4\times$ MLP you shrink the inner width by $2/3$, giving $\tfrac{2}{3}\cdot 4d = 8d/3$. This lets SwiGLU's better quality-per-parameter be measured fairly, without simply throwing more parameters at the FFN.

**5.** (Implementation) Gemma 2 uses **GeGLU** instead of SwiGLU — the same gated FFN but with GELU as the gating nonlinearity instead of Swish. Starting from the chapter's `SwiGLUFFN`, implement a `GeGLUFFN` module in the same style (same $8d/3$-rounded-to-64 default width, no-bias linears, internal dropout). What is the *only* line that must change, and why does the parameter count stay identical?

??? note "Solution"
    GeGLU is $W_\text{down}(\text{GELU}(W_\text{gate}\mathbf{x}) \odot (W_\text{up}\mathbf{x}))$. Structurally it is identical to SwiGLU; only the pointwise gate function changes from `F.silu` (Swish) to `F.gelu`. Because both are pointwise activations applied to the gate branch, they add *no* parameters, so the three-matrix parameter count is unchanged.

    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from typing import Optional


    class GeGLUFFN(nn.Module):
        """
        Feed-forward sublayer using the GeGLU gated activation (Gemma 2).

        FFN(x) = W_down( GELU(W_gate x) ⊙ W_up x )

        Identical to SwiGLUFFN except the gate uses GELU instead of Swish.
        """

        def __init__(self, dim: int, hidden_dim: Optional[int] = None,
                     bias: bool = False, dropout: float = 0.0):
            super().__init__()
            if hidden_dim is None:
                # 8/3 * dim, rounded up to nearest multiple of 64
                hidden_dim = int(8 * dim / 3)
                hidden_dim = 64 * ((hidden_dim + 63) // 64)

            self.w_gate = nn.Linear(dim, hidden_dim, bias=bias)
            self.w_up   = nn.Linear(dim, hidden_dim, bias=bias)
            self.w_down = nn.Linear(hidden_dim, dim, bias=bias)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            gate = F.gelu(self.w_gate(x))   # ONLY change: F.silu -> F.gelu
            up   = self.w_up(x)
            fused = gate * up
            return self.dropout(self.w_down(fused))
    ```

    The single load-bearing change is `F.silu(...)` becoming `F.gelu(...)` on the gate branch. Everything else — matrix shapes, the elementwise product, the down-projection, dropout — is untouched, so the parameter count is exactly the same $3 \cdot d \cdot d_\text{ff}$ as SwiGLU.

**6.** (Implementation, harder) PaLM and GPT-NeoX use the **parallel** block layout from this chapter,

$$
\mathbf{x}' = \mathbf{x} + \text{Attn}(\text{Norm}(\mathbf{x})) + \text{FFN}(\text{Norm}(\mathbf{x})),
$$

where attention and FFN read the *same* normalized input. Using the chapter's `RMSNorm`, `MinimalMHA`, and `SwiGLUFFN`, implement a `ParallelTransformerBlock` with the same constructor signature as `TransformerBlock`. Compared with the sequential block, how many normalization calls does it make per forward pass, and what hardware optimization does the shared normalized input enable?

??? note "Solution"
    The key structural change from the sequential `TransformerBlock` is that both sublayers consume one *shared* normalized tensor `h = self.norm(x)`, and their outputs are both added back to the *same* original residual `x` (rather than the FFN reading the post-attention stream).

    ```python
    import torch
    import torch.nn as nn
    from typing import Optional


    class ParallelTransformerBlock(nn.Module):
        """
        Parallel attention + FFN block (PaLM / GPT-NeoX):

            h = Norm(x)
            x' = x + Attn(h) + FFN(h)

        Both sublayers read the SAME normalized input h, and their outputs
        are summed into the residual stream in one shot.
        """

        def __init__(self, dim: int, n_heads: int,
                     ffn_hidden: Optional[int] = None,
                     bias: bool = False,
                     dropout: float = 0.0,
                     norm_eps: float = 1e-6):
            super().__init__()
            # Single shared normalization instead of one per sublayer
            self.norm = RMSNorm(dim, eps=norm_eps)
            self.attn = MinimalMHA(dim, n_heads, bias=bias)
            self.ffn  = SwiGLUFFN(dim, ffn_hidden, bias=bias, dropout=dropout)
            self.res_drop = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor,
                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            h = self.norm(x)                      # ONE norm call, shared
            attn_out = self.res_drop(self.attn(h, mask))
            ffn_out  = self.res_drop(self.ffn(h))
            return x + attn_out + ffn_out         # single residual update
    ```

    **Norm calls:** the sequential block calls `RMSNorm` **twice** per forward pass (once before attention, once before the FFN); the parallel block calls it **once**, saving a normalization and one residual addition per block.

    **Hardware optimization:** because attention and the FFN now consume the identical input `h`, their input projections — attention's QKV projection and the FFN's $W_\text{gate}$/$W_\text{up}$ projections — can be fused into a single large matrix multiply on `h`. As the chapter notes, this is a throughput win on hardware whose memory bandwidth is slow relative to compute, since one big GEMM launches more efficiently than several smaller ones. The gradient flow differs slightly from the sequential layout but empirically yields similar quality.
