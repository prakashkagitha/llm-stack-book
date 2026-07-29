"""Stack-100M: the deep-and-thin decoder-only transformer (PLAN.md sec. 1).

`forward` always returns `(logits, loss)`; loss is None when no targets are given.

Three things are threaded explicitly rather than assumed, because three later
chapters need them (Ch. 14.4 explains each):
  * `position_ids` -- RoPE tables are INDEXED, never hard-coded to arange(T), so
    packed documents can reset positions and decode steps can be positioned.
  * `seq_ids`      -- per-token document id; gives document-aware attention masking
    (no cross-document attention) for the packed 2048-token sequences of Ch. 14.2.
  * `kv_cache` + `start_pos` -- incremental decoding, so `generate` is O(T) not O(T^2).

`rebuild_rope` lets mid-training swap in a rescaled RoPE cache for long-context
extension (Ch. 14.8).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import StackConfig
from .rmsnorm import RMSNorm
from .block import Block
from .rope import build_rope_cache
from .kv_cache import KVCache
from .loss import fused_ce_z_loss
from .sampling import sample_next


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

    def _build_mask(self, seq_ids, T, kv_len, start_pos, device):
        """Bool mask (B|1, 1, T, kv_len); True = attend. None = plain-causal fast path.

        Built from POSITIONS, not indices -- which is what makes it correct for a
        rectangular (decode) score matrix. `F.scaled_dot_product_attention(...,
        is_causal=True)` aligns its mask TOP-LEFT when q_len != kv_len, so a naive
        KV-cache retrofit silently attends to position 0 only.
        """
        if seq_ids is None and kv_len == T and start_pos == 0:
            return None
        q_pos = torch.arange(start_pos, start_pos + T, device=device)
        kv_pos = torch.arange(kv_len, device=device)
        m = (q_pos[:, None] >= kv_pos[None, :])[None, None]           # (1,1,T,kv_len)
        if seq_ids is not None:                                        # no cross-document
            same = seq_ids[:, -T:, None] == seq_ids[:, None, :kv_len]  # (B,T,kv_len)
            m = m & same[:, None]
        return m

    def _cap(self, logits):
        """Gemma-2 final soft-cap, applied on the training AND inference paths -- capping
        at train time only is a silent train/serve temperature mismatch."""
        c = self.cfg.logit_soft_cap
        return c * torch.tanh(logits / c) if c > 0 else logits

    def forward(self, idx, targets=None, position_ids=None, seq_ids=None,
                kv_cache=None, start_pos=0, logits_to_keep=0, record=None):
        B, T = idx.shape
        dev = idx.device
        if position_ids is None:
            position_ids = torch.arange(start_pos, start_pos + T, device=dev).expand(B, T)
        cos = self.rope_cos.to(dev)[position_ids]        # (B, T, head_dim)
        sin = self.rope_sin.to(dev)[position_ids]

        kv_len = start_pos + T if kv_cache is not None else T
        attn_mask = self._build_mask(seq_ids, T, kv_len, start_pos, dev)

        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x, cos, sin, attn_mask=attn_mask, kv_cache=kv_cache,
                    start_pos=start_pos, record=record)
        x = self.final_norm(x)
        if logits_to_keep:
            x = x[:, -logits_to_keep:, :]                # skip lm_head on dead positions

        if targets is not None and self.cfg.loss_chunk > 0:
            ce, zl = fused_ce_z_loss(x, self.lm_head.weight, targets,
                                     self.cfg.z_loss_coef, chunk=self.cfg.loss_chunk,
                                     soft_cap=self.cfg.logit_soft_cap)
            return None, ce + zl        # logits deliberately never materialized

        logits = self._cap(self.lm_head(x))

        loss = None
        if targets is not None:
            lf = logits.float()                          # fp32 for stability
            ce = F.cross_entropy(lf.view(-1, lf.size(-1)),
                                 targets.reshape(-1), ignore_index=-100)
            logz = torch.logsumexp(lf, dim=-1)           # (B, T)
            loss = ce + self.cfg.z_loss_coef * (logz ** 2).mean()
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int = 64, temperature: float = 0.8,
                 top_p: float = 0.95, top_k: int = 0, eos_id=None,
                 use_cache: bool = True):
        """Prefill once, then one cached step per token: O(T) instead of O(T^2).

        `use_cache=False` recomputes the full prefix every step -- slow, but the
        oracle: greedy generation MUST be cache-invariant, and that assertion is the
        single highest-value test of the KV-cache/mask/position plumbing.
        """
        was_training = self.training
        self.eval()
        B, T0 = idx.shape
        total = T0 + max_new_tokens
        assert total <= self.cfg.max_seq_len, "extend max_seq_len or rebuild_rope() first"
        p = next(self.parameters())
        cache = KVCache(self.cfg, B, total, p.device, p.dtype) if use_cache else None

        logits, _ = self.forward(idx, kv_cache=cache, start_pos=0, logits_to_keep=1)
        out = idx
        done = torch.zeros(B, 1, dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            nxt = sample_next(logits[:, -1, :], temperature, top_p, top_k)
            if eos_id is not None:
                nxt = torch.where(done, torch.full_like(nxt, eos_id), nxt)
                done = done | (nxt == eos_id)
            out = torch.cat([out, nxt], dim=1)
            if eos_id is not None and bool(done.all()):
                break
            if cache is None:
                logits, _ = self.forward(out[:, -self.cfg.max_seq_len:], logits_to_keep=1)
            else:
                logits, _ = self.forward(nxt, kv_cache=cache,
                                         start_pos=out.shape[1] - 1, logits_to_keep=1)
        if was_training:
            self.train()
        return out

    @torch.no_grad()
    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and self.cfg.tie_embeddings:
            n -= self.tok_emb.weight.numel()
        return n


# Back-compat alias: several chapters refer to the model as `StackLM`.
StackLM = Stack100M
