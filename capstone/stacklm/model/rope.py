"""RoPE (Su et al., 2021) with a NoPE option (Kazemnejad et al., 2023; SmolLM3).

NoPE is applied per-layer (every `nope_every`-th layer skips RoPE) -- the
decision is made in `Attention.__init__`; this module just supplies the cache and
the rotation. `ntk_rescaled_base` implements the NTK-aware base rescaling used for
long-context extension in mid-training (Ch. 14.8).
"""
import torch


def build_rope_cache(head_dim: int, max_seq: int, theta: float,
                     device=None, dtype=torch.float32):
    """Precompute cos/sin tables of shape (max_seq, head_dim)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq, device=device).float()          # positions
    freqs = torch.outer(t, inv_freq)                          # (max_seq, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)                   # (max_seq, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    """q, k: (B, n_heads, T, head_dim). cos/sin: (T, head_dim) when every batch row
    shares positions, or (B, T, head_dim) when they do not -- which is the packed-
    document case, where position ids reset at each document boundary (Ch. 14.2/14.4)."""
    if cos.dim() == 2:
        cos, sin = cos[None, None, :, :], sin[None, None, :, :]   # (1,1,T,d_h)
    else:
        cos, sin = cos[:, None, :, :], sin[:, None, :, :]         # (B,1,T,d_h)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


def ntk_rescaled_base(base: float, head_dim: int, old_len: int, new_len: int) -> float:
    """NTK-aware RoPE base rescaling: theta' = theta * s^(d/(d-2))."""
    s = new_len / old_len
    return base * (s ** (head_dim / (head_dim - 2)))
