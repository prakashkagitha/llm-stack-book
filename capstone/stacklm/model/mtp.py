"""Multi-Token Prediction head (DeepSeek-V3, 2024; Gloeckle et al., 2024): an
auxiliary head predicting the next-2 token for a denser training signal. Kept
importable but OFF the default path.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import StackConfig
from .rmsnorm import RMSNorm
from .block import Block


class MTPHead(nn.Module):
    def __init__(self, cfg: StackConfig, lm_head: nn.Linear):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.block = Block(cfg, layer_idx=0)
        self.lm_head = lm_head           # shared with the main model

    def forward(self, h, cos, sin, targets_plus2=None):
        h = self.block(self.norm(h), cos, sin)
        logits = self.lm_head(h)
        loss = None
        if targets_plus2 is not None:
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                targets_plus2.view(-1), ignore_index=-100,
            )
        return logits, loss
