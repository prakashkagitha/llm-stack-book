"""
Runs the CPU-runnable Python code blocks from:
    content/14-capstone/01-overview-and-landscape.md

Blocks are copied faithfully from the chapter (verbatim logic) and concatenated
in document order. Each block is then actually exercised with tiny fixtures, so
every tested block EXECUTES rather than merely defining names.

Tested blocks:
    #0 (line ~191, `capstone/stacklm/config.py` in full) -- StackConfig,
       count_params, toy_config. Exercised against the chapter's own headline
       numbers (embed=16,777,216, per_block_total=2,819,200, all_blocks=
       84,576,000, total=101,353,728 ~ 101.35M) AND against toy_config(), which
       must satisfy the same accounting identity at toy scale.
    #2 (line ~586, `smoke_test.py` fragment) -- `tok = StackTokenizer(); tok.train(...);
       cfg = dataclasses.replace(toy_config(), vocab_size=tok.vocab_size, max_seq_len=96)`.
       StackTokenizer itself is Ch. 14.3's BPE tokenizer, not defined in THIS chapter's
       code blocks (it is only referenced by name here), so a tiny honest stub matching
       the interface the prose describes -- `.train(text, vocab_size, special_tokens)`
       building "256 bytes + 9 specials + merges" and exposing `.vocab_size` -- stands
       in for it. The block's own point (deriving the model config's vocab_size from
       the *trained* tokenizer via `dataclasses.replace`, never hard-coding it) is what
       actually executes and is asserted.

Skipped blocks:
    #1 (line ~481, fenced ```text```) -- SKIP(non-python): a `tree`-style directory
       listing, not code.
    #3 (line ~731, indented fragment inside a "Solution" admonition) -- SKIP(fragment):
       "# inside count_params(), replacing the `if cfg.mtp_heads: raise` guard" -- a
       diff-like snippet meant to be read in place, not a standalone unit; it reuses
       `per_block`, `embed`, `all_blocks`, `final_norm`, `lm_head` as free variables
       from inside `count_params()`'s own body, not as an importable block.

No network access and no optional third-party imports are exercised: only the
standard library (dataclasses) is used, exactly as the chapter's own code does.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass


# =====================================================================
# Block #0 (chapter: capstone/stacklm/config.py, in full)
# =====================================================================


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


# --- exercise block #0 ------------------------------------------------------
_cfg = StackConfig()
_counts = count_params(_cfg)

# The chapter's own headline numbers, verbatim (prose right after the code block).
assert _counts["embedding (tied)"] == 16_777_216, _counts["embedding (tied)"]
assert _counts["attn_per_block"] == 655_360, _counts["attn_per_block"]
assert _counts["mlp_per_block"] == 2_162_688, _counts["mlp_per_block"]
assert _counts["per_block_total"] == 2_819_200, _counts["per_block_total"]
assert _counts["all_blocks (x n_layers)"] == 84_576_000, _counts["all_blocks (x n_layers)"]
assert _counts["lm_head (untied)"] == 0, "tied embeddings must not add a separate lm_head"
assert _counts["total"] == 101_353_728, _counts["total"]
assert round(_counts["total"] / 1e6, 2) == 101.35

# The worked-example ratios quoted in the "50k vocabulary" callout.
assert abs(_counts["embedding (tied)"] / _counts["total"] - 0.166) < 0.001

# GQA really does shrink K/V relative to a naive 4*d_model^2 count.
_q_naive = 4 * _cfg.d_model * _cfg.d_model
assert _counts["attn_per_block"] < _q_naive, "GQA must be cheaper than full MHA"

# use_mla / mtp_heads are explicitly unsupported by this analytic function --
# the chapter is explicit that calling it on those configs must fail loudly,
# not silently report a wrong number.
try:
    count_params(dataclasses.replace(_cfg, use_mla=True))
    raise AssertionError("count_params must refuse an MLA config")
except NotImplementedError:
    pass
try:
    count_params(dataclasses.replace(_cfg, mtp_heads=1))
    raise AssertionError("count_params must refuse an mtp_heads>0 config")
except NotImplementedError:
    pass

# toy_config() must satisfy the identical accounting identity at toy scale, and
# must actually exercise both the RoPE and the NoPE branch of uses_rope (the
# chapter's own "branch-complete" claim about nope_every=2 with n_layers=2).
_toy = toy_config()
_toy_counts = count_params(_toy)
assert _toy_counts["embedding (tied)"] == _toy.vocab_size * _toy.d_model == 16_384
assert _toy_counts["total"] == (
    _toy_counts["embedding (tied)"] + _toy_counts["all_blocks (x n_layers)"]
    + _toy_counts["final_norm"] + _toy_counts["lm_head (untied)"]
)
assert _toy.head_groups() == 2, "4 query heads : 2 kv heads == 2 heads per group"
assert _toy.uses_rope(0) is True, "layer 0 (nope_every=2) must take the RoPE branch"
assert _toy.uses_rope(1) is False, "layer 1 (nope_every=2) must take the NoPE branch"

print("\n[block #0 OK] StackConfig / count_params / toy_config match the chapter's "
      f"101.35M headline number ({_counts['total']:,} params) and the toy config "
      "exercises both the RoPE and NoPE branches.\n")


# =====================================================================
# Block #2 (chapter: `smoke_test.py` fragment, line ~586)
#
# StackTokenizer is Ch. 14.3's own BPE tokenizer and is only *referenced* by
# name in this chapter -- its implementation lives in a different chapter's
# code, not this one. This is a tiny, honest, offline stand-in that matches
# the interface the surrounding prose describes ("256 bytes + 9 specials +
# merges" -> vocab_size=384), so that the block's OWN logic under test here
# -- deriving the model config's vocab_size from the trained tokenizer via
# `dataclasses.replace`, rather than hard-coding it -- really executes.
# =====================================================================


class StackTokenizer:
    """Minimal offline stand-in for Ch. 14.3's byte-level BPE tokenizer."""

    def __init__(self):
        self.vocab_size = 0

    def train(self, text: str, vocab_size: int, special_tokens) -> None:
        # 256 raw byte tokens + the requested special tokens is the tokenizer's
        # floor; any remaining budget goes to learned merges. This stand-in
        # doesn't actually learn BPE merges, but it reproduces the one thing
        # this block's assertions depend on: the trained vocab_size.
        floor = 256 + len(special_tokens)
        if vocab_size < floor:
            raise ValueError(f"vocab_size={vocab_size} too small for {floor} "
                              "reserved bytes + specials")
        self.vocab_size = vocab_size


SPECIAL_TOKENS = [
    "<|bos|>", "<|eos|>", "<|pad|>",
    "<|user|>", "<|assistant|>", "<|system|>", "<|end|>",
    "<|tool_call|>", "<|tool_result|>",
]
assert len(SPECIAL_TOKENS) == 9, "Ch. 14.3 reserves exactly nine special tokens"

text = "photosynthesis converts sunlight into chemical energy. " * 40

# --- the chapter's block, verbatim ------------------------------------------
tok = StackTokenizer()
tok.train(text, vocab_size=384, special_tokens=SPECIAL_TOKENS)   # 256 bytes + 9 specials + merges
cfg = dataclasses.replace(toy_config(), vocab_size=tok.vocab_size, max_seq_len=96)
# -----------------------------------------------------------------------------

assert tok.vocab_size == 384
assert cfg.vocab_size == 384, "cfg must be DERIVED from the trained tokenizer, not hard-coded"
assert cfg.max_seq_len == 96
# Every other field of toy_config() must survive the replace() untouched.
assert cfg.d_model == _toy.d_model and cfg.n_layers == _toy.n_layers
assert cfg.n_heads == _toy.n_heads and cfg.n_kv_heads == _toy.n_kv_heads
assert cfg.nope_every == _toy.nope_every and cfg.qk_norm == _toy.qk_norm
# The derived config must still satisfy count_params()'s accounting identity.
_cfg2_counts = count_params(cfg)
assert _cfg2_counts["embedding (tied)"] == cfg.vocab_size * cfg.d_model

print(f"[block #2 OK] tokenizer trained to vocab_size={tok.vocab_size}, "
      f"cfg derived via dataclasses.replace (vocab_size={cfg.vocab_size}, "
      f"max_seq_len={cfg.max_seq_len}).\n")


# =====================================================================
# SKIP notes (not executed -- see module docstring for rationale)
# =====================================================================
# block #1 (```text``` fence, line ~481) -- SKIP(non-python): a directory
#   tree listing, not code.
# block #3 (indented fragment inside "Solution" admonition, line ~731) --
#   SKIP(fragment): a diff-style snippet meant to be read in place inside
#   count_params()'s own body (`per_block`, `embed`, `all_blocks`, etc. are
#   free variables from that enclosing scope, not a standalone unit).

print("=== All tested blocks (#0, #2) executed and verified successfully. ===")
