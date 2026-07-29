"""Canonical Stack-100M configuration (see capstone/PLAN.md sec. 1).

Every number here is FROZEN by the spec. The whole capstone package derives its
shapes from `StackConfig`. `count_params` reproduces the ~101M param arithmetic
that Ch. 14.4 asks the reader to be able to do by hand.
"""
from dataclasses import dataclass


@dataclass
class StackConfig:
    # --- core shape (fixed for the whole capstone, PLAN.md sec. 1) ---
    vocab_size: int = 32768          # byte-level BPE we train ourselves (Ch. 14.3)
    d_model: int = 512               # narrow -- the "thin" in deep-and-thin
    n_layers: int = 30               # deep
    n_heads: int = 8                 # query heads
    n_kv_heads: int = 2              # GQA: 4 query heads share each KV head
    head_dim: int = 64               # n_heads * head_dim == d_model (8*64=512)
    intermediate: int = 1408         # SwiGLU hidden size, ~2.75 * d_model
    max_seq_len: int = 2048          # pretrain context; -> 8192 in mid-training (Ch. 14.8)
    rope_theta: float = 10000.0      # RoPE base; rescaled for long-context (Ch. 14.8)

    # --- stability / small-model tricks (Ch. 14.4) ---
    tie_embeddings: bool = True      # input embed == output projection (Press & Wolf, 2017)
    qk_norm: bool = True             # RMSNorm on Q, K before the attention dot product
    nope_every: int = 4              # every 4th layer skips RoPE entirely (SmolLM3-style)
    norm_eps: float = 1e-5
    z_loss_coef: float = 1e-4        # penalty on logsumexp(logits) for softmax stability
    logit_soft_cap: float = 0.0      # Gemma-2-style tanh soft-cap; 0.0 = off
    loss_chunk: int = 0              # >0 = chunked fused lm_head+CE (Ch. 14.4)
    attn_soft_cap: float = 0.0       # optional attention-logit soft-cap; 0.0 = off

    # --- optional efficiency variants, OFF by default (Ch. 14.4 "DeepSeek's trick") ---
    use_mla: bool = False            # Multi-head Latent Attention (DeepSeek-V2) instead of GQA
    mtp_heads: int = 0               # Multi-Token Prediction aux heads (DeepSeek-V3); 0 = off

    def head_groups(self) -> int:
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"
        return self.n_heads // self.n_kv_heads

    def uses_rope(self, layer_idx: int) -> bool:
        """RoPE on every layer except every `nope_every`-th (SmolLM3). 0 disables
        the interleave entirely, which is what makes the checkpoint exportable as a
        stock Qwen3 architecture (Ch. 14.4 "ecosystem" section)."""
        return self.nope_every <= 0 or ((layer_idx + 1) % self.nope_every) != 0


def count_params(cfg: StackConfig) -> dict:
    """Analytic parameter accounting -- matches `Stack100M.num_params()` exactly.

    Reproduces the Ch. 14.4 arithmetic: tied embedding counted once, per-block
    attention (Q/K/V/O with GQA-shrunk K,V), SwiGLU MLP, and the norm gains.
    """
    # This accounting is exact ONLY for the default (GQA, no-MTP, bias-free) path.
    # MLA replaces Q/K/V with down/up latent projections and MTP adds a whole extra
    # block + head -- both change the count, so refuse to report a wrong number.
    if cfg.use_mla:
        raise NotImplementedError(
            "count_params() covers the GQA path only; MLA re-shapes the attention "
            "projections. Use Stack100M(cfg).num_params() (Ch. 14.4) for MLA."
        )
    if cfg.mtp_heads:
        raise NotImplementedError(
            "count_params() covers mtp_heads=0 only; each MTP head adds an extra "
            "transformer block. Use Stack100M(cfg).num_params() (Ch. 14.4)."
        )

    embed = cfg.vocab_size * cfg.d_model

    q_width = cfg.n_heads * cfg.head_dim      # = d_model by construction (8*64=512)
    kv_width = cfg.n_kv_heads * cfg.head_dim  # = 128 (GQA shrinks this vs. q_width)
    q_proj = cfg.d_model * q_width
    k_proj = cfg.d_model * kv_width
    v_proj = cfg.d_model * kv_width
    o_proj = q_width * cfg.d_model
    attn_per_block = q_proj + k_proj + v_proj + o_proj

    mlp_per_block = 3 * cfg.d_model * cfg.intermediate

    rmsnorm_per_block = 2 * cfg.d_model                    # attn_norm + mlp_norm gains
    qk_norm_per_block = (2 * cfg.head_dim) if cfg.qk_norm else 0
    final_norm = cfg.d_model

    per_block = attn_per_block + mlp_per_block + rmsnorm_per_block + qk_norm_per_block
    all_blocks = per_block * cfg.n_layers

    lm_head = 0 if cfg.tie_embeddings else cfg.vocab_size * cfg.d_model
    total = embed + all_blocks + final_norm + lm_head

    return {
        "embedding (tied)": embed,
        "attn_per_block": attn_per_block,
        "mlp_per_block": mlp_per_block,
        "norms_per_block": rmsnorm_per_block + qk_norm_per_block,
        "per_block_total": per_block,
        "all_blocks (x n_layers)": all_blocks,
        "final_norm": final_norm,
        "lm_head (untied)": lm_head,
        "total": total,
    }


def toy_config() -> StackConfig:
    """Tiny CONFIG for the CPU smoke test -- exercises every code path (GQA 4:2,
    QK-norm, NoPE-every-4) at a scale that trains in seconds."""
    cfg = StackConfig(
        vocab_size=256,       # raw-byte-ish; the toy tokenizer trains a small vocab
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,          # still exercise the GQA code path (2:4 ratio, not 1:1)
        head_dim=16,           # 4 * 16 == 64
        intermediate=64,
        max_seq_len=64,
        qk_norm=True,
        # nope_every=2, NOT 4: with only 2 layers, `(layer_idx+1) % 4 != 0` is true
        # for BOTH layers, so a nope_every=4 toy would never execute the NoPE branch.
        # At 2 it does (layer 1 skips RoPE), so CI really covers both code paths.
        nope_every=2,
    )
    assert cfg.n_heads * cfg.head_dim == cfg.d_model
    return cfg


if __name__ == "__main__":
    cfg = StackConfig()
    for name, n in count_params(cfg).items():
        print(f"{name:26s} {n:>12,}  ({n / 1e6:7.3f}M)")
