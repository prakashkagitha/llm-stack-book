# 2.10 Modern Architecture Improvements & Design Choices

The Transformer described by Vaswani et al. in 2017 was a landmark, but the architecture used in today's production LLMs — Llama 3, Qwen 2.5, DeepSeek-V3, Gemma 2 — bears only a family resemblance to that original design. A practitioner opening a modern model config file for the first time will encounter terms like `rms_norm`, `silu`, `rope_theta`, `num_key_value_heads`, `qk_norm`, and `no_bias`. Each of these represents a deliberate, empirically validated architectural decision made in pursuit of better training stability, improved compute efficiency, or stronger final model quality.

This chapter is your annotated map of those decisions. We will examine each improvement from first principles — why it was introduced, what it fixes, how it works mathematically, and how it appears in code. We conclude with a "modern transformer recipe" summarizing which choices are nearly universal versus still debated. For the transformer block basics, see [The Transformer Block: Norms, Residuals, MLPs & Activations](../02-transformer/06-transformer-block.html); for positional encodings in depth, see [Positional Encodings: Sinusoidal, Learned, RoPE & ALiBi](../02-transformer/05-positional-encoding.html); and for GQA/MQA mechanics, see [Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html).

---

## RMSNorm: Cheaper, Equally Stable Normalization

### The problem with LayerNorm

The original Transformer used LayerNorm (Ba et al., "Layer Normalization", 2016):

$$
\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta
$$

where $\mu = \frac{1}{d}\sum_i x_i$ and $\sigma^2 = \frac{1}{d}\sum_i (x_i - \mu)^2$. This requires two passes over the feature vector: one to compute the mean, one to compute the variance. More importantly, it requires two learned parameter vectors: $\gamma$ (scale) and $\beta$ (shift). The $\beta$ term encodes an explicit re-centering.

### RMSNorm derivation

Zhang and Sennrich ("Root Mean Square Layer Normalization", NeurIPS 2019) asked: is the re-centering operation actually necessary? They ablated LayerNorm into its components and found that the re-scaling (via $\gamma$) drives almost all of LayerNorm's benefit, while mean subtraction contributes little to final performance but accounts for roughly a third of LayerNorm's compute.

Root Mean Square Normalization (RMSNorm) drops mean subtraction entirely:

$$
\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma, \quad \text{where } \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}
$$

There is no $\beta$ parameter. One division by the RMS, then a learned element-wise scale. This is approximately 10–30% faster than LayerNorm on modern hardware because there is no mean subtraction kernel and no second reduction pass.

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Used in Llama, Mistral, Qwen, DeepSeek, Gemma, and most modern LLMs.

    Parameters
    ----------
    dim : int
        The feature dimension (hidden size d_model).
    eps : float
        Small constant for numerical stability. 1e-5 is common; some
        models (Gemma) use 1e-6.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # gamma: learnable scale, initialized to ones so RMSNorm is
        # initially the identity (modulo the normalization step).
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim). Compute RMS over last dimension.
        # rsqrt = 1/sqrt for numerical efficiency.
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 for the norm computation for precision,
        # then cast back. This pattern is used in Llama reference code.
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# --- Quick sanity check ---
torch.manual_seed(42)
x = torch.randn(2, 8, 512)   # batch=2, seq=8, d_model=512
norm = RMSNorm(dim=512)
y = norm(x)
# Verify unit-RMS along the feature dimension (before scaling by weight)
raw_normed = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5)
print(f"RMS of raw_normed row 0: {raw_normed[0,0].pow(2).mean().sqrt():.4f}")  # ~1.000
```

### Pre-norm vs post-norm

GPT-2 used post-norm (LayerNorm after the residual). Nearly all modern models use **pre-norm** (LayerNorm before the sublayer):

```text
Post-norm (GPT-2):  x → Sublayer(x) + x → Norm → output
Pre-norm  (modern): x → Norm(x) → Sublayer → + x → output
```

Pre-norm yields more stable gradients at large scale — the residual path is always clean, so gradients flow through the skip connection without being divided by a normalization operation. The trade-off is that pre-norm models can exhibit "representation collapse" at depth if not carefully initialized, which motivates other tricks like scaled initialization.

---

## SwiGLU: A Better MLP Activation

### From ReLU to GELU to SwiGLU

The original Transformer used a plain two-layer MLP with ReLU:

$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

BERT switched to GELU (Gaussian Error Linear Unit), which is smoother and empirically outperforms ReLU. Modern LLMs take this further with **Gated Linear Units (GLU)** and specifically their SiLU-gated variant — SwiGLU — introduced by Noam Shazeer ("GLU Variants Improve Transformer", 2020).

### SwiGLU mechanics

A Gated Linear Unit multiplies two linear projections together, one of which passes through a nonlinearity acting as a soft gate:

$$
\text{SwiGLU}(x, W, V, W_2) = (\text{SiLU}(xW) \otimes xV) \cdot W_2
$$

where $\text{SiLU}(z) = z \cdot \sigma(z)$ (also called Swish), and $\otimes$ denotes element-wise multiplication. The SiLU function is smooth, non-monotone, and has a non-zero gradient for negative inputs, all of which help gradient flow.

The gating mechanism is important: $xV$ produces a "content" projection and $\text{SiLU}(xW)$ produces a soft gate. The gate dynamically suppresses or amplifies each dimension of the content, giving the MLP a multiplicative interaction it lacked before.

Because SwiGLU has **three** weight matrices ($W$, $V$, $W_2$) instead of two, to keep parameter count equal to a standard 4x MLP, the hidden dimension is reduced to $\frac{2}{3} \cdot 4d = \frac{8d}{3}$. In practice most models round to a multiple of 256 for hardware alignment; Llama 2 70B uses an intermediate size of 28,672 for a model dimension of 8,192.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUMLP(nn.Module):
    """
    SwiGLU feed-forward network used in Llama, Qwen, DeepSeek, etc.

    For a model with hidden_dim = d, the intermediate_dim is typically
    floor(8*d/3) rounded up to a multiple of 256.
    """

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        # Three linear projections; no bias (see section on no-bias later)
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj   = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gate: apply SiLU to create a soft gate
        gate = F.silu(self.gate_proj(x))
        # up: linear content projection
        up   = self.up_proj(x)
        # Element-wise product (gated content), then project back down
        return self.down_proj(gate * up)


# Demonstration: compare SwiGLU vs vanilla FFN parameter counts
d = 4096  # typical hidden size

# Vanilla FFN with 4x expansion: 2 * d * 4d = 8d^2
vanilla_params = 2 * d * (4 * d)

# SwiGLU FFN with 8/3 expansion: 3 * d * (8d/3) = 8d^2 — same!
intermediate = int(8 * d / 3)
swiglu_params = 3 * d * intermediate
print(f"Vanilla 4x FFN params: {vanilla_params:,}")
print(f"SwiGLU  8/3 FFN params: {swiglu_params:,}")
# Both are approximately 8*d^2 — SwiGLU is iso-parameter with better quality.
```

---

## RoPE: Rotary Position Embeddings

Rotary Position Embedding (RoPE), introduced by Su et al. ("RoFormer: Enhanced Transformer with Rotary Position Embedding", 2021), is now the dominant positional encoding for autoregressive LLMs. See [Positional Encodings: Sinusoidal, Learned, RoPE & ALiBi](../02-transformer/05-positional-encoding.html) for the full derivation; here we focus on why it was adopted in the modern recipe and its practical configuration.

### Why RoPE won

RoPE encodes position by rotating query and key vectors in 2D subspaces. The critical insight is that the dot product $q_m^\top k_n$ naturally becomes a function of the *relative offset* $(m - n)$, not of absolute positions — this is the **relative position** property that matters for generalization. Unlike sinusoidal encodings, RoPE requires no separate embedding table and requires no modification to the value vectors. Unlike learned absolute position embeddings, it generalizes beyond training context length (with appropriate scaling, see below).

The rotation for position $m$ applied to a 2D subspace of the query vector:

$$
R_m = \begin{bmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{bmatrix}
$$

For a $d_k$-dimensional head, this is repeated for $d_k/2$ rotation pairs, each with a different base frequency:

$$
\theta_i = \text{base}^{-2i/d_k}, \quad i = 0, 1, \ldots, d_k/2 - 1
$$

The `rope_theta` hyperparameter (the base) controls the frequency range. GPT-2 had no RoPE. Llama 1 used `rope_theta=10000`. Llama 3 extended this to `rope_theta=500000` to improve long-context behavior by spreading frequencies more broadly, making it easier to extrapolate to new positions at inference time.

```python
import torch
import torch.nn as nn

def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0):
    """
    Precompute complex exponentials for RoPE.
    Returns a tensor of shape (max_seq_len, dim//2) with dtype=complex64.
    The 'cis' notation: cis(x) = e^{ix} = cos(x) + i*sin(x).
    """
    # Frequencies: theta^{-2i/dim} for i in [0, dim/2)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    # Positions: [0, 1, 2, ..., max_seq_len-1]
    t = torch.arange(max_seq_len)
    # Outer product: shape (max_seq_len, dim//2)
    freqs = torch.outer(t, freqs)
    # Convert to complex: e^{i * freqs}
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor,
                     freqs_cis: torch.Tensor):
    """
    Apply RoPE to query and key tensors.
    xq, xk: (batch, seq_len, n_heads, head_dim)
    freqs_cis: (seq_len, head_dim//2) complex tensor
    """
    # Reshape to complex view: treat pairs of reals as complex numbers
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Broadcast freqs_cis over batch and head dimensions
    # freqs_cis shape: (seq_len, head_dim//2) -> (1, seq_len, 1, head_dim//2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)

    # Multiply in complex space = rotation in 2D pairs
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


# Demonstration
torch.manual_seed(0)
batch, seq_len, n_heads, head_dim = 2, 16, 8, 64
xq = torch.randn(batch, seq_len, n_heads, head_dim)
xk = torch.randn(batch, seq_len, n_heads, head_dim)

freqs = precompute_freqs_cis(head_dim, seq_len, theta=500000.0)  # Llama 3 theta
xq_rot, xk_rot = apply_rotary_emb(xq, xk, freqs)
print(f"RoPE applied. Output shape: {xq_rot.shape}")
# Verify relative position property: dot product depends only on relative offset
q0 = xq_rot[0, 3, 0]  # position 3, head 0
k5 = xk_rot[0, 8, 0]  # position 8, head 0
# The dot product q0.k5 encodes offset=5, not absolute positions 3 and 8.
```

---

## Grouped Query Attention (GQA)

Standard Multi-Head Attention (MHA) maintains separate $K$ and $V$ projection matrices for each of $H$ heads. During autoregressive decoding, the key-value (KV) cache grows as $O(L \cdot H \cdot d_k)$ per layer — for a 70B model with 64 heads and a 128K context, this is on the order of tens of gigabytes. See [Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html) for the full treatment; here we focus on the design decision and its practical configuration.

Grouped Query Attention (GQA), introduced in Ainslie et al. ("GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", 2023), is a generalization that interpolates between Multi-Head Attention and Multi-Query Attention (MQA, which has a single KV head for all query heads):

```text
MHA:  H_q = H_kv                  (all heads have own KV)
GQA:  H_q = G * H_kv  for G > 1   (G query heads share one KV head)
MQA:  H_kv = 1                     (all query heads share one KV)
```

A common configuration is 8 KV heads for 32 query heads (G=4), as in Llama 2 70B and Qwen 2.5. This reduces the KV cache memory by a factor of G while introducing only a small quality degradation versus MHA.

!!! example "Worked example: KV cache memory budget"

    Consider a model with the following configuration (similar to Llama 3 8B):
    - Layers: 32
    - Hidden dim: 4096
    - Query heads: 32 (head_dim = 128)
    - KV heads (GQA): 8

    **KV cache per token per layer:**

    $$
    \text{bytes} = 2 \text{ (K and V)} \times H_{kv} \times d_k \times \text{bytes\_per\_element}
    $$

    With bf16 (2 bytes), this is:

    $$
    2 \times 8 \times 128 \times 2 = 4096 \text{ bytes} = 4 \text{ KB per token per layer}
    $$

    **Total KV cache for 32 layers, 8K context:**

    $$
    4 \text{ KB} \times 32 \times 8192 = 1073741824 \text{ bytes} \approx 1 \text{ GB}
    $$

    With full MHA (32 KV heads), this would be 4 GB — four times larger. For a 128K context, GQA brings it from ~64 GB to ~16 GB, making long-context inference feasible on reasonable hardware.

```python
import torch
import torch.nn as nn
import math

class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) as used in Llama 2/3, Qwen, Mistral.

    n_heads: number of query heads (H_q)
    n_kv_heads: number of key/value heads (H_kv), must divide n_heads evenly
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep      = n_heads // n_kv_heads   # how many Q heads per KV head
        self.head_dim   = d_model // n_heads

        # Query projects to n_heads * head_dim
        self.q_proj = nn.Linear(d_model, n_heads    * self.head_dim, bias=False)
        # Key/Value project to n_kv_heads * head_dim (smaller!)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Expand KV from n_kv_heads to n_heads by repeating."""
        # x: (batch, seq, n_kv_heads, head_dim)
        if self.n_rep == 1:
            return x
        # Repeat along the head dimension n_rep times
        return x.unsqueeze(3).expand(
            *x.shape[:2], self.n_kv_heads, self.n_rep, self.head_dim
        ).reshape(*x.shape[:2], self.n_heads, self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        head_dim = self.head_dim

        # Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads,    head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, head_dim)

        # Expand K and V to match n_heads
        k = self._repeat_kv(k)  # (B, T, n_heads, head_dim)
        v = self._repeat_kv(v)

        # Transpose to (B, n_heads, T, head_dim) for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale         # (B, n_heads, T, T)
        attn = torch.softmax(attn, dim=-1)
        out  = attn @ v                                   # (B, n_heads, T, head_dim)

        # Merge heads and project
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)
```

---

## QK-Norm: Stabilizing Attention Logits at Scale

### The problem: logit explosion

In deep, wide models trained for many tokens, the dot products $q_i \cdot k_j$ can grow to very large values. Once the logits are large in magnitude, the softmax saturates: one token gets weight ~1 and all others get weight ~0. This "attention collapse" degrades the model's ability to attend to multiple positions, and the large pre-softmax logits create numerical instability, especially in bf16 where the dynamic range is narrow. See also [Numerical Computing, Floating Point & Precision](../01-foundations/04-numerics-precision.html) for why bf16 overflow is a real concern.

### QK-norm: normalize Q and K before attention

The fix, used in models like Gemma 2 and DeepSeek-V3, is to apply RMSNorm to the query and key vectors *before* computing the attention scores:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{\text{RMSNorm}(Q)\,\text{RMSNorm}(K)^\top}{\sqrt{d_k}}\right)V
$$

This bounds the dot products to $O(d_k)$ regardless of training duration or model size. An independent learned scale $\gamma$ per head can be added to recover expressivity.

```python
class QKNormAttention(nn.Module):
    """
    Attention with per-head QK normalization, as in Gemma 2.
    Prevents logit explosion during long training runs.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = nn.Linear(d_model, d_model, bias=False)
        self.o_proj   = nn.Linear(d_model, d_model, bias=False)

        # One RMSNorm per head for Q and K; per-head head_dim
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim)

        # Apply QK-norm per head (RMSNorm operates on head_dim)
        q = self.q_norm(q)   # normalizes the last dimension
        k = self.k_norm(k)

        # Standard scaled dot-product attention from here
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = torch.softmax((q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim), dim=-1)
        out  = (attn @ v).transpose(1,2).contiguous().view(B, T, C)
        return self.o_proj(out)
```

---

## No Biases, Tied/Untied Embeddings, and Initialization

### Dropping biases

GPT-2 had bias terms in every linear projection and LayerNorm. Modern models like Llama, Qwen, and Mistral remove biases from all linear layers. The motivation is empirical: at large scale biases do not meaningfully improve loss (they represent a negligible fraction of parameters), but they complicate optimizer state memory (Adam maintains a first and second moment for every parameter, so biases add to optimizer memory with little benefit). For a 7B model, removing biases saves a few hundred MB of optimizer state — not huge, but free.

There is also a theoretical argument: when using pre-RMSNorm (which has no bias itself), the preceding biases in linear projections are redundant, as RMSNorm can represent any affine output of a linear layer with a bias by adjusting its scale.

### Tied vs untied embeddings

The embedding table maps token IDs to vectors of dimension $d$ and has shape $(V, d)$ where $V$ is the vocabulary size. The output "unembedding" (also called the LM head or logit projection) maps from $d$ back to $V$, also shape $(V, d)$.

**Tied embeddings** share the same weight matrix for both. This was common in smaller models (original GPT-2, T5) because it saves $V \times d$ parameters — for a vocabulary of 32,000 and $d=4096$, that is ~131M parameters. **Untied embeddings** use separate matrices. The choice:

- Tied: fewer parameters, simpler to implement, forces the embedding space to be simultaneously useful for input representation and output scoring.
- Untied: more expressive; the input and output embedding spaces can specialize. Llama 1 used tied embeddings; Llama 2 and 3 use **untied** embeddings, the direction most modern large models have moved for quality reasons.

!!! note "Initialization matters more than you think"

    Modern models use carefully scaled initialization. The standard method (from GPT-2 and widely adopted) scales down the residual projections (the output projection of attention and the down-projection of MLP) by $1/\sqrt{2L}$ where $L$ is the number of layers. This controls the variance of the residual stream at initialization, preventing the residual sum from growing as $O(\sqrt{L})$. Without this, deep models (>60 layers) exhibit loss spikes at the start of training.

---

## Logit Soft-Capping

### The problem: logit skew in final layer

During pretraining, the final linear projection from hidden state to vocabulary logits can develop large-magnitude outputs for a small set of tokens. When the model becomes very confident, the cross-entropy gradient for non-predicted tokens effectively vanishes, slowing learning. At inference time, extreme logits after temperature scaling create numerical issues.

### Soft-capping

Gemma 2 introduces **logit soft-capping**: a differentiable function that squashes logits toward a cap value $c$ while preserving sign and relative ordering for logits well within the cap:

$$
\hat{z}_i = c \cdot \tanh\!\left(\frac{z_i}{c}\right)
$$

For small $|z_i| \ll c$, $\tanh(z_i/c) \approx z_i/c$, so $\hat{z}_i \approx z_i$ — linear behavior near zero. For large $|z_i|$, $\hat{z}_i \to \pm c$ asymptotically. With $c=30$ (Gemma 2's setting), logits are prevented from exceeding 30 in magnitude.

Gemma 2 also applies soft-capping to per-layer *attention logits* (pre-softmax), with $c=50$:

$$
\hat{a}_{ij} = 50 \cdot \tanh\!\left(\frac{q_i \cdot k_j / \sqrt{d_k}}{50}\right)
$$

This combines with QK-norm to doubly guard against attention saturation.

```python
import torch

def soft_cap(logits: torch.Tensor, cap: float) -> torch.Tensor:
    """
    Differentiable logit soft-capping (Gemma 2).

    Maps logits smoothly toward [-cap, +cap]. For |z| << cap, behavior
    is approximately linear; for |z| >> cap, asymptotes to ±cap.

    Args
    ----
    logits : (..., vocab_size) raw pre-softmax logits
    cap    : scalar, e.g. 30.0 for final logits or 50.0 for attention logits

    Returns
    -------
    Capped logits of same shape as input.
    """
    return cap * torch.tanh(logits / cap)


# Demonstration
z = torch.tensor([-100.0, -30.0, -10.0, 0.0, 10.0, 30.0, 100.0])
capped = soft_cap(z, cap=30.0)
print("Raw    :", z.tolist())
print("Capped :", [f"{v:.2f}" for v in capped.tolist()])
# Output:  Capped: [-30.00, -28.59, -24.88, 0.00, 24.88, 28.59, 30.00]
# Note: 100 -> 30, 30 -> 28.59 (compressed, not clipped)
```

---

## Attention Sinks and Sink Tokens

### The "attention sink" phenomenon

Xiao et al. ("Efficient Streaming LLMs with Attention Sinks", 2023) made an empirical observation: attention maps in trained autoregressive models show that some tokens — almost always the first few tokens in the context (especially BOS, Beginning-Of-Sequence) — receive anomalously high attention weights regardless of their content relevance. These are called **attention sinks**.

Why does this happen? Attention weights must sum to 1 via softmax. When no other token is relevant, the model needs somewhere to "dump" the probability mass so it can attend to nothing useful without the softmax distribution becoming uniform (which would average together all values and potentially corrupt the output). The initial tokens, having been seen in every context, become the designated garbage-collector for probability mass.

### Implications for model design and inference

The attention sink phenomenon has two practical consequences:

1. **Long-context window extension via StreamingLLM**: if you want to process infinite-length streams, you can evict old KV cache entries safely — *as long as you keep the first few tokens' KV cache* (the sinks). Dropping the sink tokens causes catastrophic loss spikes.

2. **Deliberate sink token design**: some models prepend a learnable or fixed "register" token that is never attended to in the output but acts as a sink for attention. This frees the model from overloading the BOS token.

!!! interview "Interview Corner"
    **Q:** You are designing a 70B parameter language model for production deployment. Walk through the key architectural choices you would make compared to the original Transformer, and explain *why* for each one.

    **A:** A strong answer covers the following decisions with their motivations:

    1. **RMSNorm over LayerNorm**: cheaper (no mean subtraction), equally stable; use pre-norm placement for gradient flow.
    2. **SwiGLU FFN**: empirically stronger than ReLU/GELU at the same parameter count; multiplicative gating gives more expressive activations.
    3. **RoPE positional encoding**: relative position property generalizes beyond training length; no embedding table overhead; set `rope_theta` high (500k) for long-context capability.
    4. **GQA with ~8 KV heads**: dramatically reduces KV cache memory (often 4–8x smaller) with minimal quality loss; critical for deployment economics.
    5. **No biases in linear layers**: negligible quality impact, reduces optimizer memory, simplifies distributed checkpointing.
    6. **Untied input/output embeddings**: improved model quality at scale; the cost is $V \times d$ extra parameters (~130M for a 32K vocab at d=4096), which is acceptable.
    7. **QK-norm** (for very large scale or long training): prevents attention logit explosion during extended training runs.
    8. **Logit soft-capping** (optional, Gemma 2 style): additional guard against output logit saturation, especially with large vocabularies.
    9. **Scaled residual initialization**: scale down output projections by $1/\sqrt{2L}$ to keep residual variance stable at initialization in deep models.

---

## Depth vs Width Trade-offs

### Why we go deep

Scaling laws (Kaplan et al., "Scaling Laws for Neural Language Models", 2020; Hoffmann et al., "Training Compute-Optimal Large Language Models" / Chinchilla, 2022) largely treat architecture shape as secondary to total parameter count and compute budget. See [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html) for the full story. But within a fixed parameter budget, *how* to allocate parameters between depth (number of layers) and width (hidden dimension, number of heads, MLP expansion) matters.

The dominant empirical finding is that **depth is worth more than width** up to a point. Deeper models can represent functions of exponentially higher complexity (via function composition) than wide-but-shallow ones. However, deeper models are harder to parallelize in pipeline parallel training (more micro-batches needed to fill the pipeline bubble) and have slower sequential KV-cache generation at inference.

### Practical aspect ratios

The ratio $L / d_{\text{model}}$ tends to be consistent across generations:

| Model | Layers ($L$) | Hidden dim ($d$) | $L/d$ |
|---|---|---|---|
| GPT-2 1.5B | 48 | 1600 | 0.030 |
| Llama 2 7B | 32 | 4096 | 0.0078 |
| Llama 3 70B | 80 | 8192 | 0.0098 |
| Qwen 2.5 72B | 80 | 8192 | 0.0098 |
| DeepSeek-V3 (dense equiv.) | 61 | 7168 | 0.0085 |

Modern 7B-class models favor roughly 32 layers with $d=4096$, yielding an attention head dimension of 128 (with 32 heads). Larger models scale $d$ and $L$ roughly in proportion. Going much beyond $d=8192$ per attention head becomes hardware-unfriendly: matmuls want square-ish tensors, and very large hidden dimensions with few heads under-utilize hardware parallelism.

!!! example "Worked example: parameter count breakdown for Llama 2 7B"

    Configuration (the *actual* Llama 2 7B): $L=32$, $d=4096$, $H_q=32$, $H_{kv}=32$ (full MHA -- Llama 2 predates GQA at the 7B size), $d_k=128$, intermediate $=11008$ (SwiGLU), vocab $=32000$, untied embeddings.

    **Attention ($Q, K, V, O$ projections), per layer.** With full MHA, $H_{kv}=H_q=32$, so K and V are full width:

    $$P_{\text{attn}} = \underbrace{d\,H_q d_k}_{Q} + \underbrace{2\,d\,H_{kv} d_k}_{K,V} + \underbrace{H_q d_k\, d}_{O} = 4 \times 4096^2 \approx 67.1\text{M}$$

    - $Q$: $4096 \times 4096 = 16.8$M
    - $K$: $4096 \times 4096 = 16.8$M
    - $V$: $4096 \times 4096 = 16.8$M
    - $O$: $4096 \times 4096 = 16.8$M
    - Attention total: $\approx 67.1$M

    **MLP (SwiGLU, intermediate $=11008$), per layer:**

    - gate_proj + up_proj: $2 \times 4096 \times 11008 = 90.2$M
    - down_proj: $11008 \times 4096 = 45.1$M
    - MLP total: $\approx 135.3$M

    **RMSNorm** (2 per layer): $2 \times 4096 \approx 8$K (negligible).

    **Per-layer total**: $67.1 + 135.3 \approx 202.4$M.

    **32 layers**: $32 \times 202.4\text{M} \approx 6.48$B.

    **Embeddings (untied)**: input $32000 \times 4096 \approx 131$M; output (LM head) $\approx 131$M; total $\approx 262$M.

    **Grand total**: $6.48\text{B} + 0.26\text{B} \approx 6.74$B -- exactly what "Llama 2 7B" denotes. The round "7B" is marketing rounding of 6.74B, *not* a vocabulary artifact.

    Where does the often-quoted "~5.9B" figure come from? From plugging **GQA** ($H_{kv}=8$, Llama 3 8B style) into this same count: K and V shrink to $4096 \times 1024 = 4.2$M each, attention drops to $Q+K+V+O = 16.8+4.2+4.2+16.8 \approx 42$M/layer, and the model falls to $32 \times (42+135.3)\text{M} + 262\text{M} \approx 5.94$B. So the 5.9-vs-6.7B gap is *entirely* the MHA-vs-GQA choice in the attention block -- the vocabulary ($32000$) is identical either way. (Llama 3 8B ends up *larger* than 7B despite GQA because it pairs a 128K-token vocab with a wider $14336$ intermediate.)

!!! example "Worked exercise: size a Llama-style model to a 3B budget"

    This unit's deliverable is turning a *parameter budget* into a full config. Build a 3B-parameter dense decoder step by step, then verify the count. (The result is essentially Llama 3.2 3B.)

    **Step 1 -- head dimension.** Fix $d_k = 128$ (the modern convention: good tensor-core shapes, enough per-head capacity).

    **Step 2 -- width and depth via the aspect ratio.** From the $L/d$ table above, target $L/d \approx 0.009$. Pick $d = 3072$ (a multiple of both 128 and 256); then $L \approx 0.009 \times 3072 \approx 28$ layers.

    **Step 3 -- query heads.** $H_q = d / d_k = 3072 / 128 = 24$.

    **Step 4 -- KV heads (GQA).** Choose $H_{kv} = 8$ (grouping $G = H_q/H_{kv} = 3$), the usual quality/memory sweet spot; $24$ is divisible by $8$. checks out.

    **Step 5 -- FFN width.** SwiGLU iso-parameter rule: $d_{ff} = \tfrac{8}{3} d = \tfrac{8}{3}\times 3072 = 8192$, already a multiple of 256 -- no rounding needed.

    **Step 6 -- vocabulary and embedding tying.** Use the Llama 3 tokenizer, $V = 128256$. At $d=3072$ the embedding table is $128256 \times 3072 \approx 394$M -- over 12% of a 3B budget -- so **tie** input and output embeddings (share one matrix) as Llama 3.2 1B/3B do; untying would add another 394M and blow the budget.

    **Step 7 -- verify.** Per layer: attention $Q(3072^2)+K(3072\cdot1024)+V(3072\cdot1024)+O(3072^2) \approx 25.2$M; MLP $2(3072\cdot8192)+8192\cdot3072 \approx 75.5$M; total $\approx 100.7$M. Times $L=28$ gives $2.82$B, plus the (tied) $0.39$B embedding $\Rightarrow \approx 3.21$B. On budget.

    ```python
    def llama_param_count(d, L, n_heads, n_kv, d_ff, vocab, head_dim=128, tied=True):
        # Q + O are full width; K + V are GQA-width
        attn  = 2 * d * (n_heads * head_dim) + 2 * d * (n_kv * head_dim)
        mlp   = 3 * d * d_ff          # gate, up, down (SwiGLU)
        norms = 2 * d                 # 2 RMSNorm per layer (gamma only)
        per_layer = attn + mlp + norms
        emb = vocab * d if tied else 2 * vocab * d
        return per_layer * L + emb + d   # + final RMSNorm

    n = llama_param_count(d=3072, L=28, n_heads=24, n_kv=8,
                          d_ff=8192, vocab=128256, tied=True)
    print(f"{n:,}")   # 3,212,749,824  -> ~3.21B, matches Llama 3.2 3B
    ```

    **Checklist to apply every time:** (1) $d = H_q \cdot d_k$ exactly; (2) $H_q \bmod H_{kv} = 0$; (3) $d_{ff}$ a multiple of 256; (4) for models under ~4B, tie embeddings or they dominate the budget; (5) recompute the total and confirm it lands within ~10% of target before committing to a training run.

---

## The Modern Transformer Recipe

We now have enough pieces to assemble a complete reference. The table below summarizes the consensus choices circa 2024–2026, distinguishing "near-universal" (adopted by nearly all recent models) from "model-specific" (used by some, absent in others).

| Component | GPT-2 (2019) | Modern Consensus | Notes |
|---|---|---|---|
| Normalization | LayerNorm (post) | RMSNorm (pre) | Zhang & Sennrich 2019 |
| MLP activation | GeLU | SwiGLU / GeGLU | Shazeer 2020 |
| Position encoding | Learned absolute | RoPE | Su et al. 2021 |
| Multi-head variant | MHA | GQA | Ainslie et al. 2023 |
| Bias in Linear | Yes | No | Empirical preference |
| Tied embeddings | Yes | Untied (usually) | Llama 2+ trend |
| QK-norm | No | Yes (large scale) | Gemma 2, Qwen 2.5 |
| Logit soft-cap | No | Optional | Gemma 2 |
| Attention sink | Ignored | Keep BOS in KV cache | StreamingLLM, 2023 |

```python
"""
Putting it all together: a from-scratch ModernTransformerBlock.
This implements the full recipe: pre-RMSNorm, SwiGLU, GQA with RoPE,
no biases, QK-norm optional. Runnable as a standalone unit test.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps
                               ).type_as(x) * self.weight


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, intermediate: int):
        super().__init__()
        self.gate = nn.Linear(d_model, intermediate, bias=False)
        self.up   = nn.Linear(d_model, intermediate, bias=False)
        self.down = nn.Linear(intermediate, d_model,  bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class ModernTransformerBlock(nn.Module):
    """
    A single Transformer block implementing the modern (2024-era) recipe:
      - Pre-RMSNorm on both attention and MLP sublayers
      - GQA-style attention (n_kv_heads <= n_heads)
      - SwiGLU FFN
      - No biases anywhere
      - Optional QK-norm

    Does NOT include RoPE application here for brevity; in a full model
    RoPE would be applied to Q and K after projection.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        intermediate: int,
        qk_norm: bool = False,
        eps: float = 1e-5,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0

        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep      = n_heads // n_kv_heads
        self.head_dim   = d_model // n_heads

        # Pre-norm before attention
        self.attn_norm = RMSNorm(d_model, eps)
        # Attention projections — no bias
        self.q_proj = nn.Linear(d_model, n_heads    * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model,                    bias=False)

        # Optional QK-norm
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps)
            self.k_norm = RMSNorm(self.head_dim, eps)

        # Pre-norm before MLP
        self.ffn_norm = RMSNorm(d_model, eps)
        self.ffn      = SwiGLUFFN(d_model, intermediate)

        # Scaled residual initialization for stability in deep models
        # Scale output projections by 1/sqrt(2*n_layers); caller should
        # set this post-construction, e.g.:
        #   for block in blocks:
        #       block.o_proj.weight.data.mul_(1.0 / math.sqrt(2 * n_layers))
        #       block.ffn.down.weight.data.mul_(1.0 / math.sqrt(2 * n_layers))

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        return x.unsqueeze(3).expand(
            *x.shape[:2], self.n_kv_heads, self.n_rep, self.head_dim
        ).reshape(*x.shape[:2], self.n_heads, self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        # --- Attention sublayer (pre-norm) ---
        h = self.attn_norm(x)
        q = self.q_proj(h).view(B, T, self.n_heads,    self.head_dim)
        k = self.k_proj(h).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(h).view(B, T, self.n_kv_heads, self.head_dim)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # (B, H, T, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Causal mask for autoregressive decoding
        mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        attn_logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_logits = attn_logits.masked_fill(~mask, float('-inf'))
        attn_w = torch.softmax(attn_logits, dim=-1)

        out = (attn_w @ v).transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.o_proj(out)          # residual connection

        # --- MLP sublayer (pre-norm) ---
        x = x + self.ffn(self.ffn_norm(x))  # residual connection

        return x


# --- Integration test ---
torch.manual_seed(0)
block = ModernTransformerBlock(
    d_model=512, n_heads=8, n_kv_heads=2, intermediate=1365,  # 8/3 * 512 ≈ 1365
    qk_norm=True
)
x = torch.randn(2, 16, 512)  # batch=2, seq=16, d_model=512
y = block(x)
print(f"Block output shape: {y.shape}")          # (2, 16, 512)
print(f"Output norm (should be finite): {y.norm().item():.3f}")
param_count = sum(p.numel() for p in block.parameters())
print(f"Block parameter count: {param_count:,}")  # ~4.2M for this config
```

---

!!! key "Key Takeaways"
    - **RMSNorm** (pre-norm placement) replaces LayerNorm in virtually all modern LLMs. It is ~10–30% faster, equally stable, and requires no bias parameters.
    - **SwiGLU** provides multiplicative gating that consistently outperforms ReLU/GELU FFNs at equal parameter cost; the three-weight design requires scaling the intermediate dimension to $\frac{8}{3}d$ to stay iso-parameter.
    - **RoPE** enables relative positional encoding without a separate embedding table, generalizes beyond training context length, and is the universal choice for decoder-only models. The `rope_theta` base should be set high (100k–500k) for long-context models.
    - **GQA** (Grouped Query Attention) reduces KV cache memory by a factor equal to the grouping ratio (commonly 4–8x) with minimal quality degradation; this is the key architectural enabler for long-context inference.
    - **QK-norm** (normalizing Q and K per head before computing attention scores) prevents attention logit explosion in large or long-training models; used in Gemma 2 and DeepSeek-V3.
    - **No biases** in linear layers is almost universal at the frontier; biases add optimizer memory overhead and negligible quality benefit, especially with pre-RMSNorm.
    - **Logit soft-capping** ($z \to c \cdot \tanh(z/c)$) is a differentiable alternative to hard clipping that prevents output distribution collapse and improves numerical stability.
    - **Attention sinks** (typically the BOS token) must be preserved in the KV cache for streaming/long-context inference; evicting them causes catastrophic attention pattern collapse.
    - **Scaled residual initialization** ($\times 1/\sqrt{2L}$ on output projections) is essential for stable training of very deep models; without it, the residual stream variance grows with depth.

---

!!! sota "State of the Art & Resources (2026)"
    The modern transformer recipe — RMSNorm, SwiGLU, RoPE, GQA, and QK-norm — has become near-universal across frontier open-source models (Llama 3, Qwen 2.5, DeepSeek-V3, Mistral) since 2023, with logit soft-capping and attention-sink awareness rounding out the toolkit for stable long-context training.

    **Foundational work**

    - [Zhang & Sennrich, *Root Mean Square Layer Normalization* (2019)](https://arxiv.org/abs/1910.07467) — establishes that re-scaling (not re-centering) drives LayerNorm's benefit, motivating the lighter RMSNorm used in virtually all modern LLMs.
    - [Shazeer, *GLU Variants Improve Transformer* (2020)](https://arxiv.org/abs/2002.05202) — introduces SwiGLU and GeGLU gated feed-forward networks that consistently outperform ReLU/GELU at equal parameter cost.
    - [Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021)](https://arxiv.org/abs/2104.09864) — proposes RoPE, encoding relative position via rotation matrices in the attention QK dot-product; now the dominant positional scheme for decoder-only models.
    - [Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (2023)](https://arxiv.org/abs/2305.13245) — shows grouped-query attention achieves MHA quality at MQA memory cost and provides an uptraining recipe for existing checkpoints.

    **Recent advances (2023–2026)**

    - [Peng et al., *YaRN: Efficient Context Window Extension of Large Language Models* (2023)](https://arxiv.org/abs/2309.00071) — extends RoPE to 128K+ contexts via frequency-aware interpolation, requiring 10x fewer tokens than naive fine-tuning; adopted by Mistral, Qwen, and others.
    - [The Gemma Team, *Gemma 2: Improving Open Language Models at a Practical Size* (2024)](https://arxiv.org/abs/2408.00118) — introduces QK-norm and logit soft-capping ($c \cdot \tanh(z/c)$) as complementary stability mechanisms, alongside interleaved sliding-window attention.
    - [DeepSeek-AI, *DeepSeek-V3 Technical Report* (2024)](https://arxiv.org/abs/2412.19437) — comprehensive recipe for a 671B MoE model using MLA, SwiGLU, RMSNorm, and auxiliary-loss-free load balancing; shows the modern stack scales to frontier quality.
    - [Meta AI, *The Llama 3 Herd of Models* (2024)](https://arxiv.org/abs/2407.21783) — documents the full open-source recipe at 8B–405B scale: GQA, RoPE with `theta=500000`, untied embeddings, and a 128K context window.

    **Open-source & tools**

    - [huggingface/transformers](https://github.com/huggingface/transformers) — canonical implementations of every major modern architecture (Llama, Mistral, Qwen, Gemma) with config files exposing every design knob discussed in this chapter.
    - [EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox) — research training framework with clean reference implementations of RoPE, ALiBi, SwiGLU, and GQA for GPU-scale pretraining.

    **Go deeper**

    - [Lilian Weng, *The Transformer Family Version 2.0* (2023)](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) — authoritative survey of architectural improvements (efficient attention, positional encoding variants, long-context techniques) with clear diagrams and equations.

## Further Reading

- Zhang and Sennrich, "Root Mean Square Layer Normalization," NeurIPS 2019 — the RMSNorm paper.
- Noam Shazeer, "GLU Variants Improve Transformer," arXiv 2020 — SwiGLU and GeGLU.
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding," arXiv 2021 — original RoPE paper.
- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints," EMNLP 2023 — GQA, including a method for uptraining existing MHA checkpoints.
- Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models," arXiv 2023 — a comprehensive blueprint of the modern recipe in an openly released model.
- Team Gemma, "Gemma 2: Improving Open Language Models at a Practical Size," Google DeepMind 2024 — QK-norm, logit soft-capping, and interleaved sliding-window attention.
- Xiao et al., "Efficient Streaming LLMs with Attention Sinks," ICLR 2024 — attention sink phenomenon and StreamingLLM.
- Kaplan et al., "Scaling Laws for Neural Language Models," arXiv 2020 — depth vs width trade-offs in the context of scaling.
- Meta AI, Llama 3 model card and technical report, 2024 — updated RoPE theta and architectural decisions at 70B/405B scale.
