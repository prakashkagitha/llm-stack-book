"""
Runs the CPU-runnable Python blocks from
content/02-transformer/05-positional-encoding.md, concatenated in order so
that later blocks can rely on names defined by earlier ones (as they do in
the chapter itself). Each block is copied verbatim from the chapter; the
only additions are clearly marked "GLUE" (e.g. instantiating a class the
chapter only defines, or a light assertion checking a printed claim).

Blocks covered (all 7 heuristically CPU-runnable blocks in the chapter):
  #0 (line ~59)  - sinusoidal_encoding (Vaswani et al. sinusoidal PE)
  #1 (line ~87)  - LearnedAbsolutePositionalEmbedding (GPT-2 style learned table)
  #2 (line ~194) - build_rope_cache / rotate_half / apply_rope (RoPE, rotate_half convention)
  #3 (line ~249) - rope_single + numerical verification of RoPE's relative-position property
  #4 (line ~304) - alibi_slopes / alibi_bias
  #5 (line ~368) - ntk_aware_inv_freq (NTK-aware RoPE base scaling)
  #6 (line ~397) - rope_scaling_summary (lineage table as code)

No blocks were skipped. No network/API calls anywhere in this chapter.
"""

import torch

print("=" * 70)
print("Block #0 (line ~59): sinusoidal_encoding")
print("=" * 70)

# --- verbatim from the chapter ---
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
# --- end verbatim ---

assert PE.shape == (6, 8)
# pos=0 -> sin(0)=0 on even dims, cos(0)=1 on odd dims
assert np.allclose(PE[0, 0::2], 0.0)
assert np.allclose(PE[0, 1::2], 1.0)
# every row of a sin/cos pair-encoding has unit norm per pair
pair_norms = np.sqrt(PE[:, 0::2] ** 2 + PE[:, 1::2] ** 2)
assert np.allclose(pair_norms, 1.0, atol=1e-8)


print()
print("=" * 70)
print("Block #1 (line ~87): LearnedAbsolutePositionalEmbedding")
print("=" * 70)

# --- verbatim from the chapter ---
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
# --- end verbatim ---

# GLUE: instantiate and actually call forward() on a tiny fixture, plus
# exercise the documented hard-ceiling assert with a too-long sequence.
torch.manual_seed(0)
max_len, d_model = 16, 8
learned_pe = LearnedAbsolutePositionalEmbedding(max_len=max_len, d_model=d_model)
tok_emb = torch.randn(2, 10, d_model)          # batch=2, seq_len=10 <= max_len
out = learned_pe(tok_emb)
print("learned PE output shape:", out.shape)   # torch.Size([2, 10, 8])
assert out.shape == (2, 10, d_model)
assert not torch.allclose(out, tok_emb)        # positional info was actually added

# The book explicitly documents this as a hard ceiling that crashes past max_len.
too_long = torch.randn(1, max_len + 1, d_model)
try:
    learned_pe(too_long)
    raise AssertionError("expected AssertionError for seq_len > max_len")
except AssertionError as e:
    assert "exceeds max_len" in str(e)
    print("confirmed: learned absolute PE raises past its hard length ceiling, as documented.")


print()
print("=" * 70)
print("Block #2 (line ~194): build_rope_cache / rotate_half / apply_rope")
print("=" * 70)

# --- verbatim from the chapter ---
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
# --- end verbatim ---

assert q_rot.shape == (B, H, T, D)
assert scores.shape == (B, H, T, T)
# RoPE is a rotation: it must preserve per-vector norm.
assert torch.allclose(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-4)


print()
print("=" * 70)
print("Block #3 (line ~249): rope_single + relative-position property")
print("=" * 70)

# --- verbatim from the chapter ---
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
# --- end verbatim ---


print()
print("=" * 70)
print("Block #4 (line ~304): alibi_slopes / alibi_bias")
print("=" * 70)

# --- verbatim from the chapter ---
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
# --- end verbatim ---

assert bias.shape == (4, 8, 8)
assert torch.isinf(bias[0, 0, 1])                 # future position masked to -inf
assert bias[0, 3, 3].item() == 0.0                 # zero distance -> zero bias on the diagonal


print()
print("=" * 70)
print("Block #5 (line ~368): ntk_aware_inv_freq")
print("=" * 70)

# --- verbatim from the chapter ---
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
# --- end verbatim ---

assert abs(float(ntk[0] / orig[0]) - 1.0) < 0.05
assert float(ntk[63] / orig[63]) < 1.0


print()
print("=" * 70)
print("Block #6 (line ~397): rope_scaling_summary")
print("=" * 70)

# --- verbatim from the chapter ---
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
# --- end verbatim ---

summary = rope_scaling_summary()
assert set(summary.keys()) == {"linear / PI", "ntk-aware", "yarn", "dynamic-ntk"}


print()
print("=" * 70)
print("ALL BLOCKS PASSED")
print("=" * 70)
