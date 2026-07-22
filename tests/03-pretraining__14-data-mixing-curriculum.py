"""
Runnable test for content/03-pretraining/14-data-mixing-curriculum.md

Block #0 (line ~177, 74 lines): a self-contained numpy DoReMi-style Group-DRO
reweighting toy. It is CPU-safe, pure numpy, no network/GPU dependencies.
Reproduced verbatim from the chapter below.

Block #1 (a `text` fenced block showing expected console output) is not
Python and is skipped — it is only illustrative sample output, not code to
execute.
"""

import numpy as np

# =============================================================================
# Block #0 (verbatim from the chapter, line ~177)
# =============================================================================

rng = np.random.default_rng(0)

# -----------------------------------------------------------------------------
# 1. Define k domains. Each has:
#    - floor:  irreducible loss (entropy of that domain's text), in nats/token
#    - rate:   power-law exponent for how fast loss falls with tokens seen
#    - scale:  loss = floor + scale * (tokens_seen + 1)^(-rate)
#    Domains differ in BOTH difficulty (floor) and headroom/speed (scale,rate).
# -----------------------------------------------------------------------------
domains = ["web", "code", "math", "books", "multi"]
floor = np.array([1.70, 1.10, 1.55, 1.80, 2.30])   # math & multi are "hard"
scale = np.array([2.0, 3.5, 4.0, 1.8, 2.2])        # code & math have big headroom
rate  = np.array([0.32, 0.28, 0.22, 0.30, 0.18])   # math & multi learn slowly
k = len(domains)

def domain_loss(tokens_seen):
    """Per-domain held-out loss given cumulative tokens seen in that domain."""
    return floor + scale * np.power(tokens_seen + 1.0, -rate)

# -----------------------------------------------------------------------------
# 2. Reference model: train once on the NATURAL mixture (proportional to pool
#    sizes), then read off its per-domain loss. This defines the excess-loss
#    baseline. We simulate "training" simply by accumulating tokens per domain.
# -----------------------------------------------------------------------------
pool = np.array([3000., 250., 30., 80., 400.])     # unique tokens (in B), illustrative
w_nat = pool / pool.sum()                          # natural mixture
REF_TOKENS = 5.0e4                                 # arbitrary proxy-scale token units

ref_tokens_per_domain = w_nat * REF_TOKENS
ref_loss = domain_loss(ref_tokens_per_domain)
print("natural mixture  :", np.round(w_nat, 3))
print("reference loss   :", np.round(ref_loss, 3))

# -----------------------------------------------------------------------------
# 3. Proxy model with online Group-DRO (DoReMi step 2).
#    - w:        current sampling weights over domains (the adversary's play)
#    - seen:     cumulative tokens per domain (the proxy's "knowledge")
#    Each step we draw a batch split by w, accumulate tokens, recompute the
#    proxy's per-domain loss, form CLAMPED excess loss vs the reference, and
#    apply an exponentiated-gradient (multiplicative-weights) update to w.
# -----------------------------------------------------------------------------
STEPS        = 4000
BATCH_TOKENS = 10.0          # tokens added per step (proxy-scale units)
ETA          = 1.0           # weight learning rate (exp-gradient step size)
SMOOTH_C     = 0.05          # uniform smoothing -> every domain keeps a floor weight

w    = w_nat.copy()          # start from the natural mixture
seen = np.zeros(k)           # proxy has seen nothing yet
w_history = []

for t in range(STEPS):
    # (a) Draw this step's batch according to current weights and "train":
    #     allocate BATCH_TOKENS across domains in proportion to w.
    seen += w * BATCH_TOKENS

    # (b) Proxy's current per-domain loss and CLAMPED excess loss vs reference.
    proxy_loss = domain_loss(seen)
    excess = np.maximum(proxy_loss - ref_loss, 0.0)   # cannot beat ref "for free"

    # (c) Exponentiated-gradient ascent on weights toward high-excess domains.
    w = w * np.exp(ETA * excess)
    w = w / w.sum()                                   # back onto the simplex
    w = (1.0 - SMOOTH_C) * w + SMOOTH_C / k           # uniform smoothing (floor)

    w_history.append(w.copy())

w_bar = np.mean(w_history, axis=0)   # DoReMi outputs the AVERAGED weights
print("\nfinal-step weights:", np.round(w, 3))
print("AVERAGED weights  :", np.round(w_bar, 3))
print("vs natural        :", np.round(w_nat, 3))
print("upweight factor   :", np.round(w_bar / w_nat, 2))

# =============================================================================
# Sanity checks (glue, not part of the book's block) confirming the block
# actually executed and produced results consistent with the qualitative
# claims made in the surrounding prose.
# =============================================================================

assert w_history and len(w_history) == STEPS
assert np.isclose(w.sum(), 1.0)
assert np.isclose(w_bar.sum(), 1.0)
assert np.all(w >= 0) and np.all(w_bar >= 0)

# The chapter claims web collapses from ~80% natural share to a much smaller
# averaged share, while code/math/books get upweighted relative to natural.
assert w_bar[domains.index("web")] < w_nat[domains.index("web")]
for d in ["code", "math", "books"]:
    i = domains.index(d)
    assert w_bar[i] > w_nat[i], f"{d} expected to be upweighted vs natural mixture"

print("\nAll sanity checks passed.")

# =============================================================================
# SKIP(non-python): block #1 is a fenced ```text``` block showing illustrative
# sample console output for the run above (not executable Python code).
# =============================================================================
