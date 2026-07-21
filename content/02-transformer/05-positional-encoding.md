# 2.5 Positional Encodings: Sinusoidal, Learned, RoPE & ALiBi

Take the attention equation from the previous chapter and stare at it for a moment:

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

Now permute the rows of the input — swap token 2 with token 17, reverse the whole sequence, shuffle it like a deck of cards. What happens to the output? **The output permutes in exactly the same way and nothing else changes.** Attention is *permutation-equivariant*: it treats its input as a **set**, not a **sequence**. The dot product $q_i \cdot k_j$ depends on *what* token $i$ and token $j$ are, never on *where* they sit. To attention, "dog bites man" and "man bites dog" are the same bag of three vectors. The MLPs that follow are applied independently per position and do not mix anything across the sequence either. So a bare Transformer is, astonishingly, **blind to word order**.

That is a disaster for language, code, and basically every sequence we care about. The cure is **positional encoding**: we inject information about each token's position so the model can tell *where* every token is and, more importantly, *how far apart* any two tokens are. This chapter is the complete tour of how that is done — from the original sinusoidal scheme, through learned tables, to **Rotary Position Embedding (RoPE)** and **ALiBi**, the two designs that dominate modern LLMs — and then the practical art of stretching a model trained on 4K tokens to run at 128K via **position interpolation, NTK-aware scaling, and YaRN**. We build RoPE entirely from scratch, twice, and verify its defining mathematical property numerically.

This chapter assumes the [attention mechanism](../02-transformer/03-attention-from-scratch.html) and the [embedding pipeline](../02-transformer/02-embeddings-input.html). It pairs tightly with [Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html) (RoPE is applied per head) and feeds directly into [Long-Context Pretraining & Context Extension](../03-pretraining/13-long-context-pretraining.html).

## Why Attention Needs Position, and the Two Ways to Add It

Let us nail down the permutation-equivariance claim, because it is the entire reason this chapter exists. Let $P$ be a permutation matrix that reorders the rows of the input $X \in \mathbb{R}^{n \times d}$. Self-attention computes $Q = XW_Q$, $K = XW_K$, $V = XW_V$, then $\operatorname{softmax}(QK^\top/\sqrt{d_k})V$. Permute the input to $PX$ and every projection permutes: $Q \to PQ$, $K \to PK$, $V \to PV$. The score matrix becomes $PQ(PK)^\top = PQK^\top P^\top$, the softmax (applied row-wise) commutes with the row/column permutation, and the final output is $P \cdot \operatorname{Attention}(Q,K,V)$. In words: **shuffle the input, and the output is the same vectors shuffled the same way.** There is nowhere in this computation for absolute or relative order to enter. We must add it ourselves.

There are two fundamentally different places to inject position, and the entire chapter is organized around this fork:

```text
  APPROACH A: add to the representation        APPROACH B: add to the attention score
  --------------------------------------       --------------------------------------
  x_i  ->  x_i + p_i   (before attention)       S_ij = (q_i · k_j)/√d_k  +  bias(i, j)
  sinusoidal, learned absolute                  relative bias, ALiBi
  RoPE rotates q,k (a hybrid: see §RoPE)         T5 relative position bias
```

**Approach A — modify the inputs.** Give every position $i$ a vector $p_i$ and fold it into the token representation, classically by addition: $\tilde{x}_i = x_i + p_i$. The model then sees a representation that *encodes* position, and attention can learn to use it. Sinusoidal and learned absolute encodings live here.

**Approach B — modify the attention scores.** Leave the token representations alone and instead add a position-dependent term directly to the pre-softmax scores $S_{ij}$. Because attention's only cross-token interaction is that score, this is the most surgical place to inject *relative* position $(i - j)$. ALiBi and T5's relative bias live here.

**RoPE is the brilliant hybrid.** It modifies $q_i$ and $k_j$ (Approach A's locus — the inputs to the score) but does so with a position-dependent *rotation* engineered so that the resulting dot product $q_i \cdot k_j$ depends only on the relative offset $i - j$ (Approach B's goal). It gets the implementation simplicity of touching $Q$ and $K$ with the semantics of a relative scheme. We will see exactly how.

Three properties separate good schemes from bad ones, and we will grade every method against them:

1. **Relative awareness.** Language cares about *distance* ("the adjective modifies the next noun"), not absolute index. A scheme that makes the score depend on $i-j$ rather than on $i$ and $j$ separately is structurally aligned with language.
2. **Extrapolation / length generalization.** If we train on sequences of length 4096 and run on 32768, does the model fall apart? Schemes with a learned table per position *cannot* extrapolate at all (there is no row for position 5000). Smooth functional schemes can, to varying degrees.
3. **Efficiency and KV-cache friendliness.** The encoding must be cheap and must not break the [KV cache](../07-inference-serving/01-anatomy-inference.html): the key for token $j$ must be the same whether the sequence currently has length 100 or 100000, so we can compute it once and reuse it forever.

## Sinusoidal and Learned Absolute Encodings

### The original sinusoidal encoding

The 2017 Transformer of Vaswani et al. used a fixed, parameter-free encoding. For position $\text{pos} \in \{0, 1, \dots\}$ and embedding dimension index $i \in \{0, \dots, d-1\}$, define

$$
PE_{\text{pos},\,2k} = \sin\!\left(\frac{\text{pos}}{10000^{\,2k/d}}\right), \qquad
PE_{\text{pos},\,2k+1} = \cos\!\left(\frac{\text{pos}}{10000^{\,2k/d}}\right).
$$

Each *pair* of dimensions $(2k, 2k+1)$ is a sine/cosine at its own frequency. The frequencies form a geometric progression: dimension pair $k=0$ oscillates fastest (wavelength $2\pi$), and the wavelengths grow geometrically up to $2\pi \cdot 10000$ for the last pair. The intuition is a **binary-clock for continuous space**: just as the bits of an integer toggle at frequencies $1, 2, 4, 8, \dots$, the dimensions of $PE_{\text{pos}}$ oscillate at a spectrum of frequencies, so the full vector encodes the position with high resolution (fast dims) and long range (slow dims) simultaneously.

The clever part is *why sinusoids specifically*. For any fixed offset $\Delta$, $PE_{\text{pos}+\Delta}$ is a **linear function** of $PE_{\text{pos}}$ — a rotation, in fact. Using the angle-addition formulas,

$$
\begin{bmatrix} \sin(\omega(\text{pos}+\Delta)) \\ \cos(\omega(\text{pos}+\Delta)) \end{bmatrix}
=
\begin{bmatrix} \cos(\omega\Delta) & \sin(\omega\Delta) \\ -\sin(\omega\Delta) & \cos(\omega\Delta) \end{bmatrix}
\begin{bmatrix} \sin(\omega\,\text{pos}) \\ \cos(\omega\,\text{pos}) \end{bmatrix}.
$$

The shift-by-$\Delta$ operator is a $2\times 2$ rotation that **does not depend on $\text{pos}$**, only on $\Delta$. So a linear attention projection can in principle learn to detect *relative* offsets even though the encoding is absolute. (Hold onto this rotation matrix — RoPE is exactly this idea, moved from the input embedding into $q$ and $k$.)

```python
import numpy as np

def sinusoidal_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """Classic Vaswani et al. (2017) sinusoidal positional encoding.

    Returns an (seq_len, d_model) matrix PE where PE[pos] is added to the
    token embedding at position `pos`. Parameter-free and deterministic.
    """
    pe = np.zeros((seq_len, d_model), dtype=np.float64)
    position = np.arange(seq_len)[:, None]                 # (seq_len, 1)
    # 10000^(2k/d) for k = 0,1,...,d/2-1 -> the per-pair wavelengths.
    div_term = base ** (np.arange(0, d_model, 2) / d_model)  # (d_model/2,)
    pe[:, 0::2] = np.sin(position / div_term)              # even dims: sin
    pe[:, 1::2] = np.cos(position / div_term)              # odd  dims: cos
    return pe

PE = sinusoidal_encoding(seq_len=6, d_model=8)
print(PE.round(3))
# Add to embeddings: x_tilde = token_embedding + PE[:seq_len]
```

**Grading sinusoidal.** Relative awareness: *implicit and weak* — the structure is there (the rotation property) but the model must learn to exploit it, and empirically it does so only partially. Extrapolation: *poor* — the sinusoids are defined for any position so there is no crash, but trained attention heads simply have not seen the high-position patterns and generalize badly past the training length. Efficiency: *excellent* — zero parameters, precompute once. It was a fine first design; the field has largely moved past it.

### Learned absolute encodings

GPT, GPT-2, BERT, and most early large models replaced the fixed sinusoids with a **learned lookup table**: a trainable matrix $W_{\text{pos}} \in \mathbb{R}^{L_{\max} \times d}$ where row $i$ is the embedding for position $i$, learned by gradient descent exactly like a token embedding. You pick a maximum context length $L_{\max}$ (e.g., 1024 for GPT-2, 512 for BERT), allocate that many rows, and add $W_{\text{pos}}[i]$ to the token embedding at position $i$.

```python
import torch
import torch.nn as nn

class LearnedAbsolutePositionalEmbedding(nn.Module):
    """GPT-2 style learned position embeddings: a trainable (L_max, d) table."""
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)  # one learned vector per position
        self.max_len = max_len

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        # token_embeddings: (batch, seq_len, d_model)
        seq_len = token_embeddings.size(1)
        assert seq_len <= self.max_len, (
            f"seq_len={seq_len} exceeds max_len={self.max_len}: "
            "learned absolute encodings CANNOT extrapolate — there is no row for this position."
        )
        positions = torch.arange(seq_len, device=token_embeddings.device)
        return token_embeddings + self.pos_emb(positions)  # broadcast over batch
```

Learned tables are flexible — the model carves out whatever positional geometry minimizes loss — and they were the workhorse for years. But they have a fatal flaw for the long-context era, made explicit by the `assert` above: **they cannot extrapolate even one token past $L_{\max}$.** There is simply no parameter for position 1025 in a model with 1024 rows. Worse, they tend to overfit absolute index and underperform on relative reasoning. This is *the* reason the field abandoned them.

!!! warning "Common pitfall: the off-by-one and the hard length ceiling"
    With learned absolute encodings, the maximum sequence length is a *hard architectural ceiling baked into the parameter shapes*, not a soft preference. Feeding a longer sequence indexes out of bounds and crashes (or, worse, silently wraps if you forget the assert). You cannot fine-tune your way past it without adding rows and retraining those rows from scratch. This brittleness — plus the lack of relative structure — is why nearly every model since roughly 2021 (Llama, GPT-NeoX, PaLM, Mistral, Qwen, DeepSeek, Gemma) uses RoPE or ALiBi instead.

## Relative Position and the Road to RoPE

The insight that reorganized the field: *what attention actually needs is the relative offset $i - j$, not the absolute indices.* A relative scheme makes the score between query $i$ and key $j$ a function of $(i-j)$, which is automatically translation-invariant ("two tokens apart" means the same thing at the start and end of a document) and naturally extrapolates (offset $-3$ is the same operation everywhere).

The first influential realization, from Shaw et al. (2018) and refined in Transformer-XL (Dai et al., 2019), added learned vectors indexed by the *clipped relative distance* into the keys and values. T5 (Raffel et al., 2019) simplified this to a pure **scalar bias on the score** (Approach B):

$$
S_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}} + b_{\,\text{bucket}(i-j)},
$$

where $\text{bucket}(\cdot)$ maps relative distances into a modest number of learned buckets (logarithmically spaced, so nearby offsets get fine resolution and far ones get coarse, shared buckets), and $b$ is a learned scalar per bucket *per head*. T5's scheme is genuinely relative, parameter-light, and extrapolates moderately (far distances share a bucket). Its weaknesses: the bias is added to *every* layer's scores (a small but real cost), and the bucketing is a discretization with hand-tuned hyperparameters.

This sets the stage. We want a relative scheme that is (a) continuous, not bucketed; (b) needs no extra learned parameters; (c) is cheap; and (d) plays perfectly with the KV cache. **RoPE is that scheme.**

## RoPE: Rotary Position Embedding From Scratch

RoPE (Su et al., *RoFormer*, 2021) is the dominant positional encoding in modern LLMs, and it deserves a careful derivation because the idea is genuinely elegant. The goal is stated as a constraint: find functions $f_q(x, m)$ and $f_k(x, n)$ that encode position into the query at position $m$ and the key at position $n$ such that their inner product depends *only on the relative position $m - n$* and the original vectors:

$$
\langle f_q(x_m, m),\; f_k(x_n, n) \rangle = g(x_m, x_n, m - n).
$$

{{fig:rope-rotation}}

### The 2-D derivation

Start in two dimensions ($d=2$). Treat a 2-vector $x = (x_0, x_1)$ as a complex number $x_0 + i x_1$. Define the encoding as **multiplication by a unit complex number whose angle is proportional to the position**:

$$
f(x, m) = x \cdot e^{i m \theta} \quad\text{(complex)} \;\;\Longleftrightarrow\;\;
\begin{bmatrix} x_0' \\ x_1' \end{bmatrix} =
\underbrace{\begin{bmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{bmatrix}}_{R(m\theta)}
\begin{bmatrix} x_0 \\ x_1 \end{bmatrix}.
$$

This is just **rotating the 2-D vector by an angle $m\theta$ that grows linearly with position.** Now compute the inner product of a rotated query (position $m$) with a rotated key (position $n$). In complex form, the real inner product of two complex numbers $a$ and $b$ is $\operatorname{Re}(a \bar{b})$, so

$$
\langle f(q, m), f(k, n) \rangle
= \operatorname{Re}\!\big( (q e^{i m\theta})\,\overline{(k e^{i n\theta})} \big)
= \operatorname{Re}\!\big( q \bar{k}\, e^{i(m - n)\theta} \big).
$$

The position dependence collapsed to a single factor $e^{i(m-n)\theta}$ — **it depends only on $m - n$.** That is the defining property, achieved exactly, with no learned parameters and no approximation. Equivalently, in matrix form, $R(m\theta)^\top R(n\theta) = R((n-m)\theta)$ because rotation matrices satisfy $R(\alpha)^\top R(\beta) = R(\beta - \alpha)$. The two absolute rotations combine into one relative rotation.

### Lifting to $d$ dimensions

A head dimension $d$ is much larger than 2, so RoPE **partitions the $d$-dimensional vector into $d/2$ consecutive pairs and rotates each pair by its own frequency.** Pair $k$ (dimensions $2k, 2k+1$) gets angular frequency

$$
\theta_k = \text{base}^{-2k/d}, \qquad k = 0, 1, \dots, d/2 - 1,
$$

with $\text{base} = 10000$ by default — the *same* geometric frequency ladder as the sinusoidal encoding, which is no coincidence. Low-index pairs rotate fast (encode fine, local position); high-index pairs rotate slowly (encode coarse, long-range position). The full RoPE transform is a block-diagonal rotation matrix:

$$
R_{\Theta}(m) = \begin{bmatrix}
R(m\theta_0) & & \\
& \ddots & \\
& & R(m\theta_{d/2 - 1})
\end{bmatrix} \in \mathbb{R}^{d \times d},
$$

and because each $2\times 2$ block individually satisfies the relative property, the whole thing does: $R_\Theta(m)^\top R_\Theta(n) = R_\Theta(n - m)$. The score becomes

$$
S_{mn} = \big(R_\Theta(m)\,q_m\big)^\top \big(R_\Theta(n)\,k_n\big) = q_m^\top R_\Theta(n - m)\, k_n,
$$

a function of $q_m$, $k_n$, and the **relative** offset $n - m$ alone. RoPE is applied to $q$ and $k$ **after the QKV projection and before the dot product, independently in every head and every layer**; the value vectors $V$ are *not* rotated (only the *similarity* needs position; the payload does not).

### Implementation: the rotate-half trick

We never materialize the $d \times d$ matrix $R_\Theta(m)$ — it is $99.6\%$ zeros. Instead we precompute per-position $\cos$ and $\sin$ vectors and apply the rotation elementwise. There are two equivalent conventions for *which* dimensions form a pair:

- **Interleaved** (original RoPE paper): pairs are adjacent dims $(0,1), (2,3), \dots$.
- **`rotate_half`** (GPT-NeoX / Llama / HuggingFace): pairs are dim $k$ with dim $k + d/2$, i.e., the first half is paired with the second half. This permits a vectorized `rotate_half` that is friendlier to hardware.

Both produce mathematically equivalent models *as long as you are consistent* (you can permute one into the other). The Llama-style `rotate_half` is by far the most common in practice, so we implement that, with the interleaved version as a from-scratch reference.

```python
import torch

def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0,
                     device=None, dtype=torch.float32):
    """Precompute cos/sin tables for RoPE (Llama / rotate_half convention).

    Returns cos, sin of shape (seq_len, head_dim). The first half of the
    frequency ladder is *duplicated* into the second half so that the same
    cos/sin vector multiplies x and rotate_half(x) elementwise.
    """
    assert head_dim % 2 == 0, "RoPE needs an even head dimension."
    # inv_freq[k] = base^(-2k/d) for k = 0..d/2-1  (the angular frequencies)
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    pos = torch.arange(seq_len, device=device).float()           # (seq_len,)
    freqs = torch.outer(pos, inv_freq)                           # (seq_len, d/2): angle = pos * theta_k
    emb = torch.cat([freqs, freqs], dim=-1)                      # (seq_len, d): duplicate the half
    return emb.cos().to(dtype), emb.sin().to(dtype)

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Map (x1, x2) -> (-x2, x1) where x1,x2 are the two halves of the last dim."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x of shape (batch, n_heads, seq_len, head_dim).

    cos, sin: (seq_len, head_dim). We unsqueeze to broadcast over batch & heads.
    The elementwise formula  x * cos + rotate_half(x) * sin  is exactly the
    per-pair 2x2 rotation, written without ever forming the dense matrix.
    """
    cos = cos[None, None, :, :]   # (1, 1, seq_len, head_dim)
    sin = sin[None, None, :, :]
    return x * cos + rotate_half(x) * sin

# --- Wire it into a single attention head ---
torch.manual_seed(0)
B, H, T, D = 2, 4, 16, 64          # batch, heads, seq_len, head_dim
q = torch.randn(B, H, T, D)
k = torch.randn(B, H, T, D)
v = torch.randn(B, H, T, D)

cos, sin = build_rope_cache(T, D)
q_rot = apply_rope(q, cos, sin)   # rotate queries by their positions
k_rot = apply_rope(k, cos, sin)   # rotate keys   by their positions
# Now do ordinary scaled-dot-product attention with q_rot, k_rot, and *unrotated* v:
scores = (q_rot @ k_rot.transpose(-2, -1)) / D ** 0.5
# ... add causal mask, softmax, multiply by v (see Chapter 2.3) ...
print("q_rot shape:", q_rot.shape)   # torch.Size([2, 4, 16, 64])
```

Let us verify the defining property directly — that two *absolutely* rotated vectors yield a dot product depending only on the *relative* offset.

```python
import torch

def rope_single(x, pos, base=10000.0):
    """Apply RoPE (rotate_half convention) to a single vector x at position `pos`."""
    d = x.shape[-1]
    half = d // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half).float() / half))
    ang = pos * inv_freq                       # (d/2,)
    emb = torch.cat([ang, ang])                # (d,)
    cos, sin = emb.cos(), emb.sin()
    x1, x2 = x[:half], x[half:]
    rot = torch.cat([-x2, x1])
    return x * cos + rot * sin

torch.manual_seed(7)
d = 64
q = torch.randn(d); k = torch.randn(d)
m, n = 25, 10                                  # query at 25, key at 10  -> offset 15
lhs = rope_single(q, m) @ rope_single(k, n)    # absolute positions 25 and 10
rhs = rope_single(q, m - n) @ rope_single(k, 0)  # relative: offset 15 and 0
print(float(lhs), float(rhs))                  # the two numbers match
assert torch.allclose(lhs, rhs, atol=1e-4), "RoPE relative property failed!"
print("RoPE relative-position property verified.")
```

This runs and prints two essentially identical numbers: the score from rotating $q$ at position 25 and $k$ at position 10 equals the score from rotating $q$ at offset 15 and $k$ at offset 0. **The absolute positions cancel; only the offset $15 = 25 - 10$ survives.**

!!! example "Worked example: how fast does each RoPE dimension spin?"
    Take head dimension $d = 128$, base $= 10000$, so there are $64$ frequency pairs with $\theta_k = 10000^{-2k/128} = 10000^{-k/64}$.

    - **Fastest pair, $k=0$:** $\theta_0 = 1$ radian/token. Its wavelength is $2\pi \approx 6.28$ tokens — it completes a full rotation every $\approx 6$ tokens. This pair encodes very local position.
    - **Slowest pair, $k=63$:** $\theta_{63} = 10000^{-63/64} \approx 10000^{-0.984} \approx 1.16 \times 10^{-4}$ radian/token. Its wavelength is $2\pi / \theta_{63} \approx 5.4 \times 10^{4} \approx 54{,}000$ tokens — it barely turns across the whole context. This pair encodes coarse, long-range position.

    Now the key extrapolation observation: at the *original* training length 4096, the slow pairs never complete even a fraction of a turn (4096 / 54000 ≈ 0.076 of a cycle), so the model has only ever seen them in a narrow angular range. Push the context to 100,000 tokens and those slow dimensions suddenly rotate into angles the model has *never observed during training* — which is precisely why naive RoPE degrades past the training length, and exactly what NTK/YaRN scaling (next section) repairs by adjusting the frequency ladder.

{{fig:rope-frequency-ladder}}

!!! note "Aside: why rotate Q and K but not V"
    Position should modulate *who attends to whom* (the similarity), not *what content is retrieved* (the payload). The score $q_m^\top R_\Theta(n-m) k_n$ is where relative position belongs. The value $v_n$ is the information token $n$ hands back once it has been selected; rotating it would entangle position into the retrieved content and break the clean "values are a convex blend of payloads" picture from [the attention chapter](../02-transformer/03-attention-from-scratch.html). So RoPE touches $Q$ and $K$ only.

**Grading RoPE.** Relative awareness: *exact, by construction*. Extrapolation: *good and tunable* (the basis for the scaling tricks below). Efficiency: *excellent* — no parameters, a cheap elementwise op, and it is **fully KV-cache compatible** because the rotation applied to key $n$ depends only on $n$, never on the current sequence length, so cached keys never need recomputation. This combination is why RoPE won.

## ALiBi and NoPE: Biasing the Score, or Nothing at All

### ALiBi: a linear distance penalty

**ALiBi** (Attention with Linear Biases; Press, Smith, Lewis, 2021) takes the radical Approach-B route: add *no* positional information to the embeddings or to $q$/$k$ at all, and instead subtract a penalty from each attention score that grows **linearly with the distance** between query and key:

$$
S_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}} \;-\; m_h \cdot |i - j| \quad (\text{causal: only } j \le i).
$$

Here $|i - j|$ is the relative distance and $m_h$ is a fixed, *non-learned* slope assigned per head. The slopes form a geometric sequence; for a model with $H$ heads they are typically $m_h = 2^{-8h/H}$ for $h = 1, \dots, H$ — so some heads have a steep penalty (sharply local, attend only to very recent tokens) and others have a shallow penalty (nearly global). The bias is added to the score matrix and otherwise the attention is ordinary.

```python
import torch

def alibi_slopes(n_heads: int) -> torch.Tensor:
    """Geometric slopes m_h = 2^(-8h/H), the standard ALiBi schedule (power-of-two heads)."""
    start = 2 ** (-8 / n_heads)             # ratio so that head 1 gets 2^(-8/H), head H gets 2^-8
    return torch.tensor([start ** (i + 1) for i in range(n_heads)])

def alibi_bias(seq_len: int, n_heads: int) -> torch.Tensor:
    """Per-head additive bias of shape (n_heads, seq_len, seq_len): -m_h * |i - j|."""
    slopes = alibi_slopes(n_heads)                                  # (n_heads,)
    i = torch.arange(seq_len)[:, None]                              # query index (rows)
    j = torch.arange(seq_len)[None, :]                              # key   index (cols)
    distance = (j - i).clamp(max=0).abs().float()                  # causal: |i-j| for j<=i, else 0
    causal = (j <= i)                                              # boolean causal mask
    bias = -slopes[:, None, None] * distance[None]                # (n_heads, seq_len, seq_len)
    bias = bias.masked_fill(~causal[None], float("-inf"))         # forbid attending to the future
    return bias

bias = alibi_bias(seq_len=8, n_heads=4)
# Usage: scores = q @ k.transpose(-2,-1) / sqrt(d); scores = scores + bias; softmax(scores) @ v
print(bias.shape)   # torch.Size([4, 8, 8])
print(alibi_slopes(8).round(decimals=4))
```

ALiBi's headline virtue is **length extrapolation**: because the penalty is a smooth linear function of distance defined for *any* distance, a model trained at length 1024 runs at length 4096+ with little degradation — the bias at distance 3000 is just a bigger version of the bias at distance 800, nothing the model has not effectively seen the shape of. It is also dead simple and cheap. Its limitation is that the strict monotonic decay with distance is an *inductive bias toward recency* — fine for left-to-right language modeling, but a poor fit for tasks needing strong long-range retrieval, where a far-away token must sometimes dominate. ALiBi powered BLOOM and MPT; RoPE ultimately became more popular for frontier models partly because of this retrieval ceiling, though ALiBi remains an excellent, robust choice.

### NoPE: maybe you need nothing

A genuinely surprising 2023 result (Kazemnejad et al., *The Impact of Positional Encoding on Length Generalization*): a **decoder-only** Transformer with a causal mask and *no positional encoding at all* (NoPE) can still learn position — and sometimes generalizes to longer lengths *better* than explicit schemes. How? The causal mask itself breaks permutation symmetry: token $i$ can attend to $\{0, \dots, i\}$ but token $j > i$ can attend to a strictly larger set. The model can count how many tokens are visible (e.g., via attention patterns that effectively measure the size of the attended set), recovering absolute position implicitly. NoPE is mostly a research curiosity and a clarifying conceptual point — it proves position info is *latent in causality* — rather than a production default, but it is a favorite interview probe and a reminder that the causal mask is itself doing positional work.

!!! interview "Interview Corner"
    **Q:** Modern LLMs overwhelmingly use RoPE. Explain *why* RoPE beat learned absolute and sinusoidal encodings, and name a concrete situation where you would still reach for ALiBi instead.

    **A:** RoPE wins on the three axes that matter. (1) **It is exactly relative**: rotating $q$ at position $m$ and $k$ at position $n$ makes their dot product a function of $m-n$ only — $q_m^\top R_\Theta(n-m)k_n$ — which matches how language uses position (distance, not absolute index) and is translation-invariant. (2) **It extrapolates and is *tunable*** — you can stretch the frequency ladder post-hoc with NTK/YaRN/position-interpolation to go from 4K to 128K with light or no fine-tuning, something a learned table (hard ceiling at $L_{\max}$, no row for new positions) simply cannot do. (3) **It is cheap and KV-cache-friendly**: zero parameters, an elementwise rotate-half, and the rotation on key $n$ depends only on $n$, so cached keys are never invalidated as the sequence grows. Sinusoidal-add encodings have the right rotational structure but bury it in the residual stream where the model only weakly recovers it; learned absolute encodings can't extrapolate at all. **When would I pick ALiBi?** When robust, train-short-test-long extrapolation with minimal fuss is the priority and the workload is recency-dominated left-to-right generation (e.g., streaming) rather than precise long-range retrieval — ALiBi's linear-distance penalty extrapolates gracefully out of the box with no scaling machinery. The flip side, and a good thing to volunteer: ALiBi's monotonic recency bias hurts tasks that need a distant token to dominate (needle-in-a-haystack retrieval), which is one reason frontier long-context models lean RoPE-plus-YaRN.

## Context-Length Extension: Stretching RoPE to 128K

You have a model pretrained with RoPE at 4096 tokens. You want it to handle 32K or 128K. Retraining from scratch at long context is enormously expensive (attention is $\mathcal{O}(n^2)$; see the [long-context chapter](../03-pretraining/13-long-context-pretraining.html)). The RoPE-scaling family lets you extend context with *little or no* additional training, by manipulating the frequency ladder. This is one of the most practically important — and interview-hot — topics in the chapter.

The core problem, restated from the worked example: at inference position $p > L_{\text{train}}$, the slow RoPE dimensions rotate into angles never seen in training. There are three families of fixes, best understood through the metaphor of an instrument with many strings tuned to different frequencies.

{{fig:rope-scaling-methods}}

### Position Interpolation (PI)

**Position Interpolation** (Chen et al., 2023) is the simplest idea: if the model was trained to understand positions $0$ to $L$, *don't let positions exceed $L$* — instead **squeeze** the longer sequence into the trained range. To extend to length $L' = s \cdot L$ (scale factor $s = L'/L$), replace each position $m$ with $m / s$ before computing rotations:

$$
\theta_k^{\text{PI}}(m) = \frac{m}{s}\,\theta_k.
$$

Geometrically, you compress all positions so position 8000 in a 2× extension is *treated like* position 4000 — an angle the model has seen. Every dimension's rotation now stays within the trained angular range. The cost: you have reduced the *resolution* of position (adjacent tokens are now only $\theta_k/s$ apart in angle, so the model must distinguish finer differences). PI works remarkably well but typically needs a short fine-tune (a few hundred steps) to recover the lost fine-grained resolution.

### NTK-aware scaling

PI's flaw is that it scales *every* frequency uniformly, including the fast (high-frequency, local) dimensions that were *not* the problem — squeezing them needlessly destroys local resolution. **NTK-aware scaling** (named for Neural Tangent Kernel intuitions; originally a community/blog contribution, "bloc97") fixes this by scaling **non-uniformly**: change the RoPE *base* instead of the positions, which stretches slow dimensions a lot and fast dimensions barely at all.

$$
\text{base}' = \text{base} \cdot s^{\,d / (d - 2)} \quad\Longrightarrow\quad \theta_k' = (\text{base}')^{-2k/d}.
$$

The high-frequency pairs (small $k$) are nearly unchanged (local precision preserved); the low-frequency pairs (large $k$) are stretched the most (so they don't run off into unseen angles). The headline practical fact: **NTK-aware scaling often extends context with *zero* fine-tuning** — you can sometimes 2–4× a model's context by changing one number, the base, at inference time. This is why "rope_theta" appears as a config knob in every modern serving stack.

```python
import torch

def ntk_aware_inv_freq(head_dim: int, scale: float, base: float = 10000.0):
    """NTK-aware RoPE: stretch the base by scale^(d/(d-2)), then recompute frequencies.

    `scale` = target_len / train_len. High-freq dims barely change;
    low-freq dims stretch the most, keeping their angles in the trained range.
    """
    base_prime = base * (scale ** (head_dim / (head_dim - 2)))
    half = head_dim // 2
    return 1.0 / (base_prime ** (torch.arange(0, half).float() / half))

# Extend a d=128 model's context 4x with ZERO retraining, just by changing the base:
orig = 1.0 / (10000.0 ** (torch.arange(0, 64).float() / 64))
ntk  = ntk_aware_inv_freq(head_dim=128, scale=4.0)
print("fast dim ratio (k=0): ", float(ntk[0] / orig[0]))    # ~1.0  (local: unchanged)
print("slow dim ratio (k=63):", float(ntk[63] / orig[63]))  # <1.0  (long-range: stretched)
```

### YaRN: the production-grade combination

**YaRN** (Yet another RoPE extensioN; Peng et al., 2023) is the method most long-context production models actually ship. It refines NTK with two extra ideas:

1. **NTK-by-parts (per-dimension ramp).** Instead of one global rule, classify each frequency dimension by its wavelength relative to the training context. Dimensions whose wavelength is *short* compared to $L$ (they complete many full rotations within the trained window) are left **untouched** — they already generalize. Dimensions whose wavelength is *long* (they never complete a rotation in $L$) get **interpolated** like PI. A smooth ramp blends the two regimes for in-between dimensions. This is strictly better than treating all dimensions identically.
2. **Attention temperature scaling.** As you stretch context, the softmax distribution over many more keys tends to flatten and entropy rises. YaRN multiplies the attention logits by a temperature factor $\frac{1}{t}$ defined by $\sqrt{1/t} = 0.1 \ln(s) + 1$ (a function of the scale $s$), so the logits are effectively scaled by $\big(0.1 \ln s + 1\big)^2$ — this *sharpens* the flattened distribution back and recovers effective attention. Crucially it costs nothing at runtime: implementations fold the per-vector factor $m_{\text{scale}} = \sqrt{1/t} = 0.1 \ln(s) + 1$ into the RoPE $\cos/\sin$ tables and apply it to *both* $q$ and $k$, so the square $m_{\text{scale}}^2 = 1/t$ arises automatically from the $q\cdot k$ dot product. (Watch this factor-of-two-in-the-exponent when reading the paper: the paper's $\sqrt{1/t}$ is the per-tensor scale, not the logit scale.)

The empirical result is that YaRN extends context further than PI or plain NTK *and* with less fine-tuning — it is the basis of many "32K → 128K" and "128K → 1M" model releases. The lineage is worth memorizing: **PI** (uniform squeeze, needs fine-tune) → **NTK-aware** (non-uniform via base, often training-free) → **YaRN** (per-dimension ramp + logit temperature, the production default). Linear/dynamic-NTK variants in HuggingFace (`rope_scaling={"type": "yarn"/"linear"/"dynamic", ...}`) implement these; the `dynamic` variant recomputes the scale on the fly so short sequences stay un-degraded.

```python
import torch

def rope_scaling_summary():
    """The decision the lineage encodes, in code-as-table form."""
    return {
        "linear / PI":  "pos -> pos/s; uniform squeeze; simplest; usually needs a short fine-tune",
        "ntk-aware":    "base -> base * s^(d/(d-2)); non-uniform; OFTEN training-free for 2-4x",
        "yarn":         "per-dim ramp (NTK-by-parts) + logit temperature; best quality, longest reach",
        "dynamic-ntk":  "recompute scale per actual seq_len; no penalty on short inputs",
    }
for k, v in rope_scaling_summary().items():
    print(f"{k:14s}: {v}")
```

!!! tip "Practitioner tip: always evaluate the extended model on long-range *retrieval*, not just perplexity"
    A context-extended model can have a deceptively low perplexity at 128K — perplexity is dominated by the easy local tokens — while completely failing to *use* information from the far end of the context. Always validate with a targeted long-range probe such as needle-in-a-haystack (plant a fact at a random depth, ask for it) or RULER-style synthetic tasks. It is common to see PI/NTK pass perplexity but flunk retrieval at the new length, which then guides whether you need YaRN, more fine-tuning data at length, or a larger scale factor. See [Long-Context Pretraining & Context Extension](../03-pretraining/13-long-context-pretraining.html) for the full evaluation playbook.

!!! warning "Common pitfall: re-applying RoPE scaling on top of a model already trained long"
    RoPE scaling configs are *not* idempotent and *not* always composable. If a model was already pretrained or fine-tuned with, say, a YaRN factor of 8, layering another linear scale of 4 at serving time does not give you 32× — it double-counts the frequency stretch and usually *destroys* quality. Check the model's published `rope_theta` and `rope_scaling` in its config before adding your own. When in doubt, set the scale relative to the *original pretraining* length, not the current advertised context.

## Putting It Together: Choosing and Reasoning About Position

We have traversed the whole landscape. Step back and the design space is a small, comprehensible table.

| Scheme | Where injected | Relative? | Extrapolation | Params | KV-cache-safe | Used by |
|---|---|---|---|---|---|---|
| Sinusoidal | input ($x + p$) | implicit/weak | poor | 0 | yes | orig. Transformer |
| Learned absolute | input ($x + p$) | no | **none** (hard ceiling) | $L_{\max}\!\cdot d$ | yes | GPT-2, BERT |
| T5 relative bias | score $S_{ij}$ | yes | moderate | few/head | yes | T5 |
| **RoPE** | rotate $q,k$ | **exact** | good + **tunable** | 0 | yes | Llama, Qwen, GPT-NeoX, DeepSeek, Mistral, Gemma |
| **ALiBi** | score $S_{ij}$ | yes | **excellent** | 0 | yes | BLOOM, MPT |
| NoPE | nothing | implicit (causal) | surprisingly good | 0 | yes | research |

The story of the field is a march from Approach A's absolute encodings (sinusoidal, learned) toward relative ones, with the two survivors being RoPE (rotate $q,k$ — exact relative, tunable extrapolation, the default for almost all frontier models) and ALiBi (a linear score penalty — supremely simple and robust extrapolation, recency-biased). The context-extension toolkit — PI, NTK-aware, YaRN — exists *because* RoPE's frequency ladder is a continuous, manipulable object, turning "train short, serve long" from impossible (learned tables) into a config flag.

From here, RoPE'd attention slots into [a full Transformer block](../02-transformer/06-transformer-block.html) and a [from-scratch GPT](../02-transformer/07-build-gpt-from-scratch.html); its interaction with [GQA/MLA](../02-transformer/04-mha-gqa-mla.html) matters because RoPE is applied per-head and MLA must carefully preserve the rotation through its low-rank compression; and the long-context economics it enables drive [inference](../07-inference-serving/01-anatomy-inference.html) and [scaling-law](../03-pretraining/04-scaling-laws.html) decisions throughout the stack.

!!! key "Key Takeaways"
    - **Bare attention is permutation-equivariant** — it sees a set, not a sequence. Without an explicit positional scheme, a Transformer cannot tell "dog bites man" from "man bites dog." Position must be injected.
    - **Two injection sites:** add to the representation (sinusoidal, learned absolute — Approach A) or add a bias to the attention score (T5 relative, ALiBi — Approach B). RoPE is the hybrid: it rotates $q,k$ (Approach A's locus) so the dot product depends on relative offset (Approach B's goal).
    - **Learned absolute encodings have a hard length ceiling** ($L_{\max}$ baked into parameter shapes) and cannot extrapolate by even one token — the central reason the field abandoned them.
    - **RoPE rotates each $q,k$ pair by an angle proportional to position**, so $q_m^\top R_\Theta(n-m)k_n$ depends only on the relative offset $n-m$. It has zero parameters, is a cheap rotate-half elementwise op, and is fully KV-cache compatible (key $n$'s rotation depends only on $n$). It is the modern default.
    - **ALiBi** adds a per-head linear distance penalty $-m_h|i-j|$ to the score, with no embedding-side position at all. It extrapolates beautifully out of the box but encodes a recency bias that can hurt long-range retrieval.
    - **NoPE** shows the causal mask alone leaks enough information for a decoder-only model to recover position — a clarifying result, not a production default.
    - **Context extension manipulates RoPE's frequency ladder:** Position Interpolation (uniform squeeze, needs fine-tune) → NTK-aware (stretch the base non-uniformly, often training-free for 2–4×) → YaRN (per-dimension ramp + attention-logit temperature, the production-grade method behind most long-context releases).
    - **Always validate extended context with long-range *retrieval* probes** (needle-in-a-haystack), not perplexity, which is dominated by easy local tokens. And never blindly stack a second RoPE-scaling factor on an already-scaled model.

!!! sota "State of the Art & Resources (2026)"
    RoPE is now the default positional encoding in virtually every frontier LLM (Llama, Mistral, Qwen, DeepSeek, Gemma). The active frontier is context extension: LongRoPE and YaRN variants push trained-at-4K models to 1M+ tokens, and NoPE's surprising results continue to sharpen our theoretical understanding of what position encodings actually do.

    **Foundational work**

    - [Vaswani et al., *Attention Is All You Need* (2017)](https://arxiv.org/abs/1706.03762) — introduces sinusoidal positional encodings and the original Transformer; the fixed-frequency rotation property motivates RoPE.
    - [Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021)](https://arxiv.org/abs/2104.09864) — the RoPE paper; derives the complex-multiplication rotation and proves the exact relative-position property.
    - [Press, Smith & Lewis, *Train Short, Test Long: Attention with Linear Biases* (2021)](https://arxiv.org/abs/2108.12409) — introduces ALiBi; demonstrates robust out-of-the-box length extrapolation via a simple linear distance penalty on attention scores.

    **Recent advances (2023–2026)**

    - [Chen et al., *Extending Context Window via Positional Interpolation* (2023)](https://arxiv.org/abs/2306.15595) — Position Interpolation: linearly squeeze positions into the trained range to extend RoPE context with ~1000 fine-tuning steps.
    - [Peng et al., *YaRN: Efficient Context Window Extension* (2023)](https://arxiv.org/abs/2309.00071) — per-dimension NTK-by-parts ramp plus attention temperature scaling; the production-grade method behind most 32K→128K model releases.
    - [Kazemnejad et al., *The Impact of Positional Encoding on Length Generalization* (NeurIPS 2023)](https://arxiv.org/abs/2305.19466) — shows decoder-only Transformers with no explicit positional encoding (NoPE) can outperform RoPE and ALiBi on out-of-distribution lengths.
    - [Ding et al., *LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens* (2024)](https://arxiv.org/abs/2402.13753) — non-uniform per-dimension rescaling with evolutionary search; used in Microsoft Phi-3 to reach 2M-token context.

    **Open-source & tools**

    - [microsoft/LongRoPE](https://github.com/microsoft/LongRoPE) — official LongRoPE implementation and evolution-search code; integrated into the Phi-3 model series.
    - [HuggingFace Transformers: Utilities for Rotary Embedding](https://huggingface.co/docs/transformers/main/en/internal/rope_utils) — canonical reference for `rope_scaling` config (linear, dynamic, yarn, longrope, llama3 variants) used across the HF ecosystem.

    **Go deeper**

    - [EleutherAI Blog: *Rotary Embeddings — A Relative Revolution*](https://blog.eleuther.ai/rotary-embeddings/) — the accessible explainer that popularized RoPE in the English-language ML community; includes PyTorch and JAX implementations.

## Further reading

- Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin — *Attention Is All You Need* (2017). Introduces the sinusoidal positional encoding and the rotation-by-fixed-offset property.
- Su, Lu, Pan, Murtadha, Wen, Liu — *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021). The RoPE paper; the 2-D complex-multiplication derivation and the $d$-dimensional block-rotation construction.
- Press, Smith, Lewis — *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation* (2021). The ALiBi method and its extrapolation story.
- Raffel, Shazeer, Roberts, et al. — *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (T5, 2019). The bucketed relative-position scalar bias on attention scores.
- Shaw, Uszkoreit, Vaswani — *Self-Attention with Relative Position Representations* (2018), and Dai et al. — *Transformer-XL* (2019). The origins of relative position representations.
- Chen, Wong, Chen, Tian — *Extending Context Window of Large Language Models via Positional Interpolation* (2023). Position Interpolation.
- Peng, Quesnelle, Fan, Shippole — *YaRN: Efficient Context Window Extension of Large Language Models* (2023). NTK-by-parts plus attention temperature; the dominant production extension method.
- Kazemnejad, Padhi, Natesan Ramamurthy, Das, Reddy — *The Impact of Positional Encoding on Length Generalization in Transformers* (2023). The NoPE result for decoder-only models.
