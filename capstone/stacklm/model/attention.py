"""Grouped-query attention (Ainslie et al., 2023) with QK-norm and per-layer NoPE.

Attribute names (`n_heads`, `n_kv_heads`, `head_dim`, `groups`) are the ones the
MuonClip / QK-clip routine (`stacklm.optim.qk_clip`) reaches into, so keep them
stable. `record` is an optional dict the training loop passes in to harvest the
per-head max attention logit needed by QK-clip.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import StackConfig
from .rmsnorm import RMSNorm
from .rope import apply_rope


class Attention(nn.Module):
    def __init__(self, cfg: StackConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.groups = cfg.head_groups()               # query heads per KV head (=4)
        # NoPE on every `nope_every`-th layer (layers 3,7,11,... with nope_every=4)
        self.use_rope = ((layer_idx + 1) % cfg.nope_every) != 0

        self.wq = nn.Linear(cfg.d_model, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, cfg.d_model, bias=False)

        self.q_norm = RMSNorm(self.head_dim, cfg.norm_eps) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, cfg.norm_eps) if cfg.qk_norm else nn.Identity()
        self.attn_soft_cap = cfg.attn_soft_cap

    def forward(self, x, cos, sin, attn_mask=None, record=None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # QK-norm BEFORE RoPE (training stability at high LR)
        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.use_rope:
            q, k = apply_rope(q, k, cos, sin)

        # GQA: expand the 2 KV heads up to 8 query heads.
        k = k.repeat_interleave(self.groups, dim=1)   # (B, n_heads, T, head_dim)
        v = v.repeat_interleave(self.groups, dim=1)

        scale = 1.0 / (self.head_dim ** 0.5)

        # Optional: record per-head max attention logit for MuonClip/QK-clip.
        if record is not None:
            with torch.no_grad():
                att = (q.float() @ k.float().transpose(-2, -1)) * scale  # (B,H,T,T)
                causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
                att = att.masked_fill(~causal, float("-inf"))
                record[self.layer_idx] = att.amax(dim=(0, 2, 3))         # (n_heads,)

        if self.attn_soft_cap and self.attn_soft_cap > 0:
            att = (q @ k.transpose(-2, -1)) * scale
            cap = self.attn_soft_cap
            att = cap * torch.tanh(att / cap)
            if attn_mask is not None:
                att = att.masked_fill(~attn_mask, float("-inf"))
            else:
                causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
                att = att.masked_fill(~causal, float("-inf"))
            att = att.softmax(dim=-1)
            out = att @ v
        elif attn_mask is not None:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, n_heads*head_dim)
        return self.wo(out)
