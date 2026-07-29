"""Multi-Token Prediction (DeepSeek-V3, 2024; Gloeckle et al., 2024) -- an auxiliary
module that predicts token t+2 for a denser training signal, and doubles as a draft
head for self-speculative decoding. Kept importable but OFF the default path.

The defining feature, which generic "extra depth head" implementations miss: the
module concatenates the trunk hidden state h_i with the EMBEDDING of the already-known
next token t_{i+1}, projects the pair back to d_model, and only then runs a transformer
block. That conditioning is what keeps the causal chain intact; stacking modules
k = 1, 2, ... sequentially (each consuming the previous one's output) extends the
prediction horizon.

Own params: one block (2,819,200) + the 2d x d projection (524,288) + two norms
(1,024) = 3,344,512 -- a 3.3% training-time overhead, discarded at inference.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import StackConfig
from .rmsnorm import RMSNorm
from .block import Block


class MTPHead(nn.Module):
    def __init__(self, cfg: StackConfig, tok_emb: nn.Embedding, lm_head: nn.Linear):
        super().__init__()
        self.h_norm = RMSNorm(cfg.d_model, cfg.norm_eps)     # RMSNorm(h_i)
        self.e_norm = RMSNorm(cfg.d_model, cfg.norm_eps)     # RMSNorm(Emb(t_{i+1}))
        self.proj = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False)   # M_k
        self.block = Block(cfg, layer_idx=0)                 # TRM_k (pre-norms internally)
        self.tok_emb = tok_emb            # SHARED with the trunk, not copied
        self.lm_head = lm_head            # SHARED with the trunk, not copied

    def forward(self, h, next_ids, cos, sin, targets_plus2=None, attn_mask=None):
        """h: (B,T,d) trunk hidden state at position i.
        next_ids: (B,T) token t_{i+1}.  targets_plus2: (B,T) token t_{i+2}."""
        e = self.tok_emb(next_ids)
        z = self.proj(torch.cat([self.h_norm(h), self.e_norm(e)], dim=-1))
        h2 = self.block(z, cos, sin, attn_mask=attn_mask)
        logits = self.lm_head(h2)
        loss = None
        if targets_plus2 is not None:
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                targets_plus2.reshape(-1), ignore_index=-100,
            )
        return logits, loss

# training step: total = main_loss + lambda_mtp * mtp_loss
# DeepSeek-V3 reports lambda = 0.3 for the bulk of training, lowered to 0.1 at the end.
