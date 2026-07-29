"""Full training-FLOP accounting for the scaling ladder (Ch. 14.5).

6ND is a good rule at 7B and a bad one at 100M. The real per-token training cost
of a decoder-only transformer has three terms (every "x3" is fwd+bwd ~ 3x fwd):

    blocks    = 6 * N_nonembed
    attention = 6 * n_layers * seq_len * d_model     (causal masking halves it)
    head      = 6 * d_model * vocab_size             (input embedding is a gather)

Note the identity 6*N_nonembed + 6*d*V == 6*N_total for a TIED embedding, so this
is the same cost model Ch. 14.1 writes as (6*N_total + 6*L*s*d) * D.
"""
from .ladder import LadderConfig  # noqa: F401  (re-exported for convenience)


def flops_per_token(cfg) -> dict:
    """Blocks + causal attention + tied head, in FLOPs per training token."""
    blocks = 6.0 * cfg.nonembed_params()
    attn = 6.0 * cfg.n_layers * cfg.seq_len * cfg.d_model
    head = 6.0 * cfg.d_model * cfg.vocab_size
    return dict(blocks=blocks, attn=attn, head=head, total=blocks + attn + head)


def training_flops(cfg, n_tokens: float) -> float:
    """Total training FLOPs -- the number you budget and schedule against."""
    return flops_per_token(cfg)["total"] * n_tokens


def training_flops_6nd(cfg, n_tokens: float) -> float:
    """The 6ND approximation, kept ONLY for contrast and for reproducing
    published numbers that were quoted that way."""
    return 6.0 * cfg.nonembed_params() * n_tokens


def gpu_hours(flops: float, peak_flops_per_s: float = 312e12,
              mfu: float = 0.35) -> float:
    """Wall-clock on ONE accelerator. 312 TFLOP/s ~ A100 bf16 dense peak. MFU is
    measured against the FULL model FLOPs above -- the only honest denominator
    (Ch. 14.1 derives the 0.30-0.45 band; Ch. 14.7 measures it on the real run)."""
    return flops / (peak_flops_per_s * mfu) / 3600.0
