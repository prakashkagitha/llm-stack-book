"""Design the ladder sweep: an IsoFLOP backbone plus off-diagonal points, with a
per-run batch size and schedule (Ch. 14.5).

Two constraints size every batch: at least MIN_STEPS optimizer steps (so the WSD
warmup/stable/decay phases mean something) and at or below the critical batch
size (Zhang et al., ICLR 2025). Evaluated at the flagship's 20B tokens the rule
returns 2**19 = 524,288 tokens -- exactly the batch Ch. 14.6 freezes.
"""
import math

from .ladder import BY_NAME
from .flops import flops_per_token, training_flops

# (true_compute_budget_C, rung). Every run in a slice costs exactly C under the
# FULL FLOP model -- so the slices really are iso-FLOP.
ISO = [(1e16, "S1"), (1e16, "S2"),
       (3e16, "S1"), (3e16, "S2"), (3e16, "S3"),
       (9e16, "S1"), (9e16, "S2"), (9e16, "S3"),
       (2.7e17, "S2"), (2.7e17, "S3"), (2.7e17, "S4")]

# Off-diagonal fixed-model points: (rung, tokens_per_param). These break the
# degeneracy of the iso-compute direction so alpha and beta separate.
EXTRA = [("S1", 12), ("S2", 12), ("S3", 12), ("S4", 12), ("S1", 400), ("S2", 150)]

MIN_STEPS = 2000


def critical_batch_tokens(D: float) -> float:
    """Empirical critical-batch-size fit (Zhang et al., 2025): B* grows ~ D^0.47
    and depends only weakly on N. Illustrative constants -- re-fit on your own
    runs; the point is that B* is a function of D, not a constant."""
    return 22.91 * D ** 0.47


def batch_tokens(D: float) -> int:
    """Tokens per optimizer step: small enough for >= MIN_STEPS steps AND at or
    below the critical batch size, rounded DOWN to a power of two, clipped to
    [2**14, 2**19]."""
    b = min(D / MIN_STEPS, critical_batch_tokens(D))
    b = 2 ** int(math.floor(math.log2(b)))
    return int(min(max(b, 2 ** 14), 2 ** 19))


def build_runs():
    """Return the 17 sweep runs, each a dict the harness consumes."""
    runs = []
    for C, name in ISO:
        c = BY_NAME[name]
        D = C / flops_per_token(c)["total"]     # tokens s.t. TRUE cost == C
        runs.append(dict(cfg=c, N=c.nonembed_params(), D=D, C=C,
                         tpp=D / c.nonembed_params(), kind="iso"))
    for name, tpp in EXTRA:
        c = BY_NAME[name]
        D = float(tpp * c.nonembed_params())
        runs.append(dict(cfg=c, N=c.nonembed_params(), D=D,
                         C=training_flops(c, D), tpp=float(tpp), kind="fixed"))
    for r in runs:                              # size the batch, then the schedule
        r["batch_tokens"] = batch_tokens(r["D"])
        r["steps"] = int(r["D"] / r["batch_tokens"])
        r["warmup_steps"] = min(2000, max(50, round(0.05 * r["steps"])))
    return runs


# --- muP-style per-rung learning rates (Ch. 14.5 / Ch. 14.6) ----------------
D_BASE = 512          # muP base width = the target's d_model
MUON_LR = 6e-3        # measured on the S4 rung in Ch. 14.6; width-invariant
ADAMW_LR = MUON_LR / 2   # 3e-3 at the base width


def rung_lrs(cfg) -> dict:
    """Muon peak is width-invariant (the 0.2*sqrt(max(m,n)) RMS match divides the
    shape dependence out). The tied embedding/head follows muP's READOUT rule,
    eta ~ 1/d_model; norms and 1D params keep a width-independent LR."""
    return dict(muon=MUON_LR,
                adamw_head=ADAMW_LR * D_BASE / cfg.d_model,
                adamw_norms=ADAMW_LR)
