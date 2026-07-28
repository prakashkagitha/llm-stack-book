"""A pre-norm decoder block: RMSNorm -> attention -> residual, RMSNorm -> SwiGLU -> residual."""
import torch.nn as nn

from ..config import StackConfig
from .rmsnorm import RMSNorm
from .attention import Attention
from .swiglu import SwiGLU


class Block(nn.Module):
    def __init__(self, cfg: StackConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg, layer_idx)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, attn_mask=None, record=None):
        x = x + self.attn(self.attn_norm(x), cos, sin, attn_mask=attn_mask, record=record)
        x = x + self.mlp(self.mlp_norm(x))
        return x
