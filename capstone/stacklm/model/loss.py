"""Memory-lean lm_head + cross-entropy + z-loss (Ch. 14.4).

At 100M params with a 32k vocab the LOGIT tensor is the dominant activation: a
(16, 2048, 32768) fp32 copy is ~4.3 GB, more than every other activation in the
network combined. We chunk over tokens and recompute each chunk in the backward
pass, so peak logit memory is (chunk x vocab) and does not grow with batch size.
CE and z-loss share the same logsumexp -- we compute it once.

The production versions of this idea are `liger_kernel`'s LigerFusedLinearCrossEntropy
and Apple's Cut Cross-Entropy (`cut_cross_entropy.linear_cross_entropy`), which fuse
the whole thing into a Triton kernel and never write a logit tensor at all.
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _chunk_ce(h, w, t, cap: float):
    """One chunk -> (sum CE, sum lse^2, n_valid). Recomputed in backward."""
    logits = F.linear(h, w).float()                  # fp32 for stability
    if cap > 0:
        logits = cap * torch.tanh(logits / cap)      # SAME cap as the inference path
    lse = torch.logsumexp(logits, dim=-1)            # reused by BOTH losses
    valid = (t != -100)
    tgt = t.clamp_min(0).unsqueeze(-1)
    ce = lse - logits.gather(-1, tgt).squeeze(-1)    # CE = logsumexp - logit[target]
    return (ce * valid).sum(), (lse.pow(2) * valid).sum(), valid.sum()


def fused_ce_z_loss(hidden, weight, targets, z_coef: float,
                    chunk: int = 8192, soft_cap: float = 0.0):
    """Returns (cross_entropy, z_loss). Numerically equal to the unchunked path."""
    h = hidden.reshape(-1, hidden.shape[-1])
    t = targets.reshape(-1)
    ce = h.new_zeros((), dtype=torch.float32)
    z = h.new_zeros((), dtype=torch.float32)
    n = torch.zeros((), dtype=torch.long, device=h.device)
    for i in range(0, h.shape[0], chunk):
        a, b, c = checkpoint(_chunk_ce, h[i:i + chunk], weight, t[i:i + chunk],
                             soft_cap, use_reentrant=False)
        ce, z, n = ce + a, z + b, n + c
    n = n.clamp_min(1)
    return ce / n, z_coef * (z / n)
