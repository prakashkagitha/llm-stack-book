"""Multi-head Latent Attention (DeepSeek-V2, 2024) -- the "if you want DeepSeek's
trick" upgrade taught in Ch. 14.4. KV is compressed into a low-rank latent `c_kv`, a
further KV-cache win beyond GQA. Kept importable but OFF the default path.

The decoupled RoPE branch is implemented in full, and it is not optional: a rotation
does not commute with a low-rank projection, so a compressed key cannot carry
position. MLA therefore concatenates a small, *uncompressed*, rotary key channel of
width `d_rope` to the up-projected content key (and a matching per-head query slice).
An MLA without it is a positionless attention -- a silent, mysterious-loss-curve bug.

Not yet wired to `KVCache`: MLA caches a DIFFERENT tensor (c_kv and k_r) than GQA
does, so it needs its own cache type. Training-only for now; `generate()` asserts
against it rather than pretending.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import StackConfig
from .rmsnorm import RMSNorm
from .rope import apply_rope, build_rope_cache


class MLAttention(nn.Module):
    def __init__(self, cfg: StackConfig, layer_idx: int = 0, d_c: int = 128, d_rope: int = 32):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.d_c = d_c            # KV compression latent dim   <-- the cached tensor
        self.d_rope = d_rope      # decoupled RoPE dim          <-- also cached (tiny)
        self.use_rope = cfg.uses_rope(layer_idx)
        d = cfg.d_model
        H, dh, dr = self.n_heads, self.head_dim, self.d_rope

        self.wq = nn.Linear(d, H * dh, bias=False)          # content queries
        self.wqr = nn.Linear(d, H * dr, bias=False)         # decoupled rotary queries
        self.w_dkv = nn.Linear(d, d_c, bias=False)          # down-project to the latent
        self.kv_norm = RMSNorm(d_c, cfg.norm_eps)
        self.w_uk = nn.Linear(d_c, H * dh, bias=False)      # up-project keys   (not cached)
        self.w_uv = nn.Linear(d_c, H * dh, bias=False)      # up-project values (not cached)
        self.w_kr = nn.Linear(d, dr, bias=False)            # ONE shared rotary key
        self.wo = nn.Linear(H * dh, d, bias=False)

        # Own RoPE tables: the decoupled channel has width d_rope, NOT head_dim.
        cos, sin = build_rope_cache(dr, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("cos_r", cos, persistent=False)
        self.register_buffer("sin_r", sin, persistent=False)

    def forward(self, x, cos=None, sin=None, attn_mask=None, kv_cache=None,
                start_pos=0, record=None):
        B, T, _ = x.shape
        H, dh, dr = self.n_heads, self.head_dim, self.d_rope
        c_kv = self.kv_norm(self.w_dkv(x))                          # (B,T,d_c)  CACHE THIS
        k_r = self.w_kr(x).view(B, T, 1, dr).transpose(1, 2)        # (B,1,T,dr) AND THIS
        q_c = self.wq(x).view(B, T, H, dh).transpose(1, 2)
        q_r = self.wqr(x).view(B, T, H, dr).transpose(1, 2)
        if self.use_rope:
            pos = torch.arange(start_pos, start_pos + T, device=x.device)
            cr, sr = self.cos_r.to(x.device)[pos], self.sin_r.to(x.device)[pos]
            q_r, k_r = apply_rope(q_r, k_r, cr, sr)                 # rotate BOTH sides
        S = c_kv.shape[1]
        k_c = self.w_uk(c_kv).view(B, S, H, dh).transpose(1, 2)
        v = self.w_uv(c_kv).view(B, S, H, dh).transpose(1, 2)
        q = torch.cat([q_c, q_r], dim=-1)                           # (B,H,T,dh+dr)
        k = torch.cat([k_c, k_r.expand(B, H, S, dr)], dim=-1)       # (B,H,S,dh+dr)
        out = F.scaled_dot_product_attention(                       # V is narrower than Q,K
            q, k, v, attn_mask=attn_mask, is_causal=(attn_mask is None),
            scale=(dh + dr) ** -0.5)
        return self.wo(out.transpose(1, 2).contiguous().view(B, T, -1))

    def absorbed_query(self, q_c, head: int):
        """The absorption trick -- why MLA is FAST at decode, not merely small.

        For head h, q_h^T (W_uk^(h) c) == (W_uk^(h)^T q_h)^T c, so you push the query
        into the d_c latent space ONCE and dot it straight against the cached c_kv;
        per-head keys are never materialized. (W_uv folds into W_o the same way.)
        """
        dh = self.head_dim
        w_h = self.w_uk.weight[head * dh:(head + 1) * dh]     # (dh, d_c)
        return q_c @ w_h                                      # (..., d_c)
