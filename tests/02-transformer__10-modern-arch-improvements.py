"""
Executable test for content/02-transformer/10-modern-arch-improvements.md

Assembles the chapter's CPU-runnable Python blocks, in order, into one
module and executes them. Later blocks reuse names (e.g. RMSNorm) defined
by earlier blocks, exactly as the chapter itself does (block #9 even
redefines RMSNorm/SwiGLU standalone -- that's intentional in the book,
since it's presented as a self-contained "putting it all together" unit).

Blocks tested (from the chapter):
  - block #0 (line ~33)  : RMSNorm
  - block #2 (line ~121) : SwiGLUMLP
  - block #3 (line ~188) : RoPE (precompute_freqs_cis / apply_rotary_emb)
  - block #5 (line ~290) : GroupedQueryAttention (GQA)
  - block #7 (line ~465) : soft_cap (Gemma 2 logit soft-capping)
  - block #9 (line ~642) : ModernTransformerBlock (full recipe)

Blocks skipped:
  - block #1: non-python (text diagram of pre-norm vs post-norm)
  - block #4: non-python (text diagram of MHA/GQA/MQA head counts)
  - block #6: fragment (QKNormAttention depends on RMSNorm/math from
    earlier blocks/imports in the chapter's surrounding prose; its logic
    is already fully exercised via QK-norm inside block #9's
    ModernTransformerBlock, which is instantiated with qk_norm=True below)
  - block #8: fragment (llama_param_count, a pure arithmetic helper inside
    a worked-example callout, not one of the 6 designated blocks)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Block #0 (line ~33): RMSNorm
# ============================================================

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
assert abs(raw_normed[0, 0].pow(2).mean().sqrt().item() - 1.0) < 1e-3
assert y.shape == x.shape


# ============================================================
# Block #2 (line ~121): SwiGLUMLP
# ============================================================

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

# Actually exercise the module (small dims for CPU speed), since the
# chapter only prints param counts here -- we call the module too.
small_mlp = SwiGLUMLP(hidden_dim=32, intermediate_dim=int(8 * 32 / 3))
mlp_out = small_mlp(torch.randn(3, 5, 32))
assert mlp_out.shape == (3, 5, 32)


# ============================================================
# Block #3 (line ~188): RoPE
# ============================================================

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
assert xq_rot.shape == xq.shape
assert xk_rot.shape == xk.shape
assert torch.isfinite(xq_rot).all()
# Verify relative position property: dot product depends only on relative offset
q0 = xq_rot[0, 3, 0]  # position 3, head 0
k5 = xk_rot[0, 8, 0]  # position 8, head 0
# The dot product q0.k5 encodes offset=5, not absolute positions 3 and 8.
offset5_dot = torch.dot(q0, k5).item()
print(f"q0.k5 dot product (offset=5): {offset5_dot:.4f}")
assert math.isfinite(offset5_dot)


# ============================================================
# Block #5 (line ~290): GroupedQueryAttention (GQA)
# ============================================================

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


# --- Exercise GQA (the chapter defines this class but does not call it
#     in-place; we instantiate and run it here on a tiny toy input) ---
torch.manual_seed(1)
gqa = GroupedQueryAttention(d_model=64, n_heads=8, n_kv_heads=2)
gqa_x = torch.randn(2, 10, 64)
gqa_out = gqa(gqa_x)
print(f"GQA output shape: {gqa_out.shape}")
assert gqa_out.shape == gqa_x.shape
assert torch.isfinite(gqa_out).all()


# ============================================================
# Block #7 (line ~465): soft_cap (Gemma 2 logit soft-capping)
# ============================================================

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
# Output:  Capped: [-29.92, -22.85, -9.65, 0.00, 9.65, 22.85, 29.92]
# Note: 100 -> 29.92, 30 -> 22.85 (compressed, not clipped)
# (The chapter's original comment stated [-30.00, -28.59, -24.88, ...],
#  which does not match tanh(z/cap)*cap for these inputs -- fixed to the
#  actual computed values; see the book-bug note below.)
assert capped.abs().max().item() <= 30.0 + 1e-4
expected = [-29.92, -22.85, -9.65, 0.00, 9.65, 22.85, 29.92]
for got, exp in zip(capped.tolist(), expected):
    assert abs(got - exp) < 0.02, f"soft_cap mismatch: {got} vs {exp}"


# ============================================================
# Block #9 (line ~642): ModernTransformerBlock (full recipe)
# ============================================================
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

assert y.shape == (2, 16, 512)
assert torch.isfinite(y).all()
assert param_count > 0

# Also exercise qk_norm=False path (untested branch otherwise)
block_no_qknorm = ModernTransformerBlock(
    d_model=32, n_heads=4, n_kv_heads=1, intermediate=int(8 * 32 / 3),
    qk_norm=False
)
y2 = block_no_qknorm(torch.randn(1, 6, 32))
assert y2.shape == (1, 6, 32)
assert torch.isfinite(y2).all()


print("\nAll blocks executed successfully.")
