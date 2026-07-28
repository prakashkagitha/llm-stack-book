"""Stack-100M: the deep-and-thin decoder-only transformer (PLAN.md sec. 1).

`forward` always returns `(logits, loss)`; loss is None when no targets are
given. Document-aware masking is supported via `seq_ids` (a per-token segment id);
when omitted, attention is plain causal. `rebuild_rope` lets mid-training swap in
a rescaled RoPE cache for long-context extension (Ch. 14.8).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import StackConfig
from .rmsnorm import RMSNorm
from .block import Block
from .rope import build_rope_cache


class Stack100M(nn.Module):
    def __init__(self, cfg: StackConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight   # TIED (Press & Wolf, 2017)

        cos, sin = build_rope_cache(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scale residual-projection inits by 1/sqrt(2*n_layers) (GPT-2 trick).
        scale = (2 * cfg.n_layers) ** -0.5
        for blk in self.blocks:
            nn.init.normal_(blk.attn.wo.weight, mean=0.0, std=0.02 * scale)
            nn.init.normal_(blk.mlp.down.weight, mean=0.0, std=0.02 * scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    def rebuild_rope(self, max_seq_len: int, rope_theta: float, device=None):
        cos, sin = build_rope_cache(self.cfg.head_dim, max_seq_len, rope_theta, device=device)
        self.rope_cos, self.rope_sin = cos, sin
        self.cfg.max_seq_len = max_seq_len
        self.cfg.rope_theta = rope_theta

    def _attn_mask(self, seq_ids, T, device):
        if seq_ids is None:
            return None
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
        same = seq_ids[:, :, None] == seq_ids[:, None, :]      # (B, T, T)
        return (causal[None] & same).unsqueeze(1)              # (B, 1, T, T)

    def forward(self, idx, targets=None, seq_ids=None, record=None):
        B, T = idx.shape
        cos = self.rope_cos[:T].to(idx.device)
        sin = self.rope_sin[:T].to(idx.device)
        attn_mask = self._attn_mask(seq_ids, T, idx.device)

        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x, cos, sin, attn_mask=attn_mask, record=record)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        if self.cfg.logit_soft_cap > 0:
            c = self.cfg.logit_soft_cap
            logits = c * torch.tanh(logits / c)

        loss = None
        if targets is not None:
            ce = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
            logz = torch.logsumexp(logits.float(), dim=-1)     # (B, T)
            z_loss = self.cfg.z_loss_coef * (logz ** 2).mean()
            loss = ce + z_loss
        return logits, loss

    @torch.no_grad()
    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and self.cfg.tie_embeddings:
            n -= self.tok_emb.weight.numel()
        return n


# Back-compat alias: several chapters refer to the model as `StackLM`.
StackLM = Stack100M
