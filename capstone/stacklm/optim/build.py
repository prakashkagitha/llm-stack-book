"""Hybrid optimizer construction (Ch. 14.6): Muon for the 2D hidden matrices,
AdamW for embeddings, norms, and 1D params. `build_optimizers` returns the hybrid
pair; `build_optimizer` returns a single AdamW-on-everything optimizer, which is
what the post-training stages (SFT/DPO/GRPO) prefer.
"""
import torch

from .muon import Muon


def _fused_ok() -> bool:
    return torch.cuda.is_available()


def build_optimizers(model, muon_lr=0.02, adamw_lr=3e-3,
                     weight_decay=0.1, betas=(0.9, 0.95)):
    """Return (muon, adamw). Muon gets the hidden 2D matrices; AdamW gets the
    tied embedding, RMSNorm gains, and any 1D params."""
    muon_params, adamw_params = [], []
    embed_ids = {id(model.tok_emb.weight)}
    for _name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in embed_ids:
            adamw_params.append(p)                 # embedding -> AdamW
        elif p.ndim == 2:
            muon_params.append(p)                  # hidden matrix -> Muon
        else:
            adamw_params.append(p)                 # 1D norms/biases -> AdamW

    muon = Muon(muon_params, lr=muon_lr, momentum=0.95, nesterov=True,
                weight_decay=weight_decay, ns_steps=5)
    decay, no_decay = [], []
    for p in adamw_params:
        (decay if p.ndim >= 2 else no_decay).append(p)
    adamw = torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=adamw_lr, betas=betas, eps=1e-8, fused=_fused_ok())
    return muon, adamw


def build_optimizer(model, lr=2e-5, weight_decay=0.0, betas=(0.9, 0.95)):
    """Plain AdamW over all params -- the standard choice for post-training."""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=betas, eps=1e-8, fused=_fused_ok())
