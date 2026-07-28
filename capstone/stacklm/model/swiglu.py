"""SwiGLU gated MLP (Shazeer, 2020). Three matrices: gate, up, down."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import StackConfig


class SwiGLU(nn.Module):
    def __init__(self, cfg: StackConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.intermediate, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.intermediate, bias=False)
        self.down = nn.Linear(cfg.intermediate, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))
