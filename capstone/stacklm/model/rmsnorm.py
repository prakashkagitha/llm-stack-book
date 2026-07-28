"""RMSNorm (Zhang & Sennrich, 2019) -- pre-norm residual blocks use it, and it
doubles as QK-norm on the per-head query/key vectors."""
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # gamma; init to 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()                                        # accumulate in fp32
        rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * (xf * rms)).to(dtype)
