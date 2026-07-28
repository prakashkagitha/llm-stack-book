"""Multi-head Latent Attention (DeepSeek-V2, 2024) -- the "if you want DeepSeek's
trick" upgrade taught in Ch. 14.4. KV is compressed into a low-rank latent, a
further KV-cache win beyond GQA. Kept importable but OFF the default path.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import StackConfig
from .rmsnorm import RMSNorm
from .rope import apply_rope


class MLAttention(nn.Module):
    def __init__(self, cfg: StackConfig, layer_idx: int = 0, d_c: int = 128, d_rope: int = 32):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.d_c = d_c            # KV compression latent dim
        self.d_rope = d_rope      # per-head decoupled RoPE dim
        self.use_rope = ((layer_idx + 1) % cfg.nope_every) != 0

        self.wq = nn.Linear(cfg.d_model, self.n_heads * self.head_dim, bias=False)
        self.w_dkv = nn.Linear(cfg.d_model, d_c, bias=False)               # down-project to latent
        self.w_uk = nn.Linear(d_c, self.n_heads * self.head_dim, bias=False)   # up-project keys
        self.w_uv = nn.Linear(d_c, self.n_heads * self.head_dim, bias=False)   # up-project values
        self.w_kr = nn.Linear(cfg.d_model, self.n_heads * d_rope, bias=False)  # decoupled RoPE keys
        self.wo = nn.Linear(self.n_heads * self.head_dim, cfg.d_model, bias=False)

    def forward(self, x, cos=None, sin=None, attn_mask=None, record=None):
        B, T, _ = x.shape
        H, dh = self.n_heads, self.head_dim
        q = self.wq(x).view(B, T, H, dh).transpose(1, 2)
        c_kv = self.w_dkv(x)                                     # (B, T, d_c)
        k = self.w_uk(c_kv).view(B, T, H, dh).transpose(1, 2)
        v = self.w_uv(c_kv).view(B, T, H, dh).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=(attn_mask is None),
                                             attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)
