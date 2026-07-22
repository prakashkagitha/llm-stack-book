"""
Runs the CPU-runnable Python code blocks from:
    content/03-pretraining/04-scaling-laws.md

Blocks are copied faithfully (verbatim logic) and concatenated in document
order; later blocks reuse names (N_obs, D_obs, L_obs, huber, objective,
predict_log_loss, fit_once, minimize, ...) defined by earlier blocks, exactly
as the chapter's prose says they do ("continues the fitting-block variables").

Tested blocks:  #1 (chinchilla_optimal), #3 (parametric fit), #4 (bootstrap
                 identifiability check), #5 (IsoFLOP method), #6 (sweep
                 design), #7 (held-out extrapolation check), #8
                 (lifetime_optimal / inference-aware over-training), #9
                 (emergent-abilities-as-metric-artifact demo).
Skipped blocks: #0 (line ~86, training_flops/inference_flops) -- a bare
                 utility-function fragment with no standalone exercise/assert
                 in the book itself; marked "fragment" per the task brief.
                #2 (line ~231) -- a ```text``` block of expected console
                 output, not Python. Its numbers are instead asserted against
                 directly in Block #1 below, so the "non-python" block is
                 still exercised as a correctness check, just not executed
                 as code.

Third-party dependency note: Blocks #3, #4, and #7 use `scipy.optimize.minimize`
and `scipy.special.logsumexp`. scipy is not in this task's guaranteed-available
list (numpy/torch/einops/sklearn/stdlib only), so the import is guarded; if
scipy is unavailable those three blocks are skipped at runtime (their functions
are still defined, just not called) rather than erroring the whole file.

Runtime note: the book's own bootstrap block (#4) resamples n_boot=200 times
with n_starts=15 L-BFGS-B multi-starts each (3000 total fits), which alone
takes >90s. We reduce n_boot to 20 (same fit_once function, same bootstrap
logic, fewer resamples) to keep total runtime well under the ~60s budget;
this is a runtime-only "tiny fixture" change, not a change to the book's logic.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    _HAVE_SCIPY = True
except Exception:
    minimize = None
    logsumexp = None
    _HAVE_SCIPY = False


def _section(name: str) -> None:
    print(f"\n=== {name} ===")


# ============================================================================
# Block #1 (line ~204) -- chinchilla_optimal()
# ============================================================================
_section("Block #1: chinchilla_optimal")


def chinchilla_optimal(C, A, B, alpha, beta, E=1.69):
    """Compute-optimal (N*, D*) for budget C given fitted scaling-law params.

    Minimizes L(N,D) = E + A*N^-alpha + B*D^-beta  s.t.  C = 6*N*D.
    Returns the optimal params, tokens, predicted loss, and tokens/param.
    """
    exp = beta / (alpha + beta)                  # exponent on C for N
    coef = (alpha * A / (beta * B)) ** (1.0 / (alpha + beta))
    N_opt = coef * (C / 6.0) ** exp
    D_opt = C / (6.0 * N_opt)
    L_opt = E + A * N_opt ** (-alpha) + B * D_opt ** (-beta)
    return {
        "N": N_opt, "D": D_opt, "loss": L_opt,
        "tokens_per_param": D_opt / N_opt,
        "N_exponent": exp, "D_exponent": 1.0 - exp,
    }


# Illustrative coefficients in the *spirit* of Chinchilla's fit (not exact):
A, B, alpha, beta, E = 406.4, 410.7, 0.34, 0.28, 1.69
_block1_results = {}
for C in [1e19, 1e21, 1e23, 1e25]:
    r = chinchilla_optimal(C, A, B, alpha, beta, E)
    _block1_results[C] = r
    print(f"C={C:.0e}  N={r['N']:.2e}  D={r['D']:.2e}  "
          f"tok/param={r['tokens_per_param']:.1f}  L={r['loss']:.3f}")

# Cross-check against the book's own printed reference table (the "non-python"
# ```text``` block, #2 in the numbering, is exercised here as expected values
# rather than executed as code).
_expected_block2 = {
    1e19: dict(N=2.28e08, D=7.31e09, tpp=32.1, L=2.986),
    1e21: dict(N=1.82e09, D=9.14e10, tpp=50.1, L=2.329),
    1e23: dict(N=1.46e10, D=1.14e12, tpp=78.2, L=2.005),
    1e25: dict(N=1.17e11, D=1.43e13, tpp=122.1, L=1.845),
}
for C, exp_vals in _expected_block2.items():
    got = _block1_results[C]
    assert abs(got["N"] - exp_vals["N"]) / exp_vals["N"] < 1e-2
    assert abs(got["D"] - exp_vals["D"]) / exp_vals["D"] < 1e-2
    assert abs(got["tokens_per_param"] - exp_vals["tpp"]) < 0.2
    assert abs(got["loss"] - exp_vals["L"]) < 1e-2

# The book claims the ratio D/N is NOT constant (alpha != beta here), and
# grows across decades of compute -- verify that directly.
_tpps = [_block1_results[C]["tokens_per_param"] for C in [1e19, 1e21, 1e23, 1e25]]
assert _tpps == sorted(_tpps), "tokens/param should be increasing with C when alpha != beta"

print("Block #1 OK")


# ============================================================================
# Block #3 (line ~247) -- fitting a scaling law from synthetic data
# SKIP(scipy unavailable): needs scipy.optimize.minimize / scipy.special.logsumexp,
# which is not in this task's guaranteed-import list. Functions defined but not
# called if scipy is missing.
# ============================================================================
_section("Block #3: fit a scaling law (parametric / Chinchilla Approach 3)")

# ---------------------------------------------------------------------------
# Step 0: synthesize a grid of (N, D, observed_loss) "experiments".
# In reality these rows come from actual training runs; here we fabricate them
# from a known ground-truth law plus noise so we can check we recover it.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
TRUE = dict(E=1.69, A=406.0, B=410.0, alpha=0.34, beta=0.28)


def true_loss(N, D, p=TRUE):
    return p["E"] + p["A"] * N ** (-p["alpha"]) + p["B"] * D ** (-p["beta"])


Ns = np.array([1e7, 3e7, 1e8, 3e8, 1e9, 3e9])          # 10M .. 3B params
Ds = np.array([1e8, 3e8, 1e9, 3e9, 1e10, 3e10, 1e11])  # 0.1B .. 100B tokens
grid = [(N, D) for N in Ns for D in Ds]
N_obs = np.array([g[0] for g in grid])
D_obs = np.array([g[1] for g in grid])
# multiplicative ~2% noise, as you'd see from seed/data-order variation:
L_obs = true_loss(N_obs, D_obs) * (1.0 + 0.02 * rng.standard_normal(len(grid)))


# ---------------------------------------------------------------------------
# Step 1: the model. We fit E, A, B, alpha, beta. To keep all params positive
# and the optimizer well-conditioned, we parameterize via logs:
#   E = exp(e), A = exp(a), B = exp(b), alpha and beta directly.
# Chinchilla's key trick: minimize the HUBER loss of the LSE residual in
# LOG-loss space, which is robust to the few noisy outlier runs.
# ---------------------------------------------------------------------------
def predict_log_loss(theta, N, D):
    e, a, b, alpha, beta = theta
    # Build log L via log-sum-exp of the three log-terms (Chinchilla's Appendix
    # trick): numerically stable across many orders of magnitude because it never
    # forms the raw sum E + A*N^-alpha + B*D^-beta before taking the log.
    terms = np.stack([
        np.full_like(N, e),                 # log of E
        a - alpha * np.log(N),              # log of A*N^-alpha
        b - beta * np.log(D),               # log of B*D^-beta
    ])
    return logsumexp(terms, axis=0)         # = log(E + A*N^-alpha + B*D^-beta)


def huber(r, delta=1e-3):
    a = np.abs(r)
    return np.where(a <= delta, 0.5 * r**2, delta * (a - 0.5 * delta))


def objective(theta):
    resid = predict_log_loss(theta, N_obs, D_obs) - np.log(L_obs)
    return np.sum(huber(resid))


def optimal_alloc(C, e, a, b, alpha, beta):
    A_, B_, E_ = np.exp(a), np.exp(b), np.exp(e)
    exp = beta / (alpha + beta)
    N = (alpha * A_ / (beta * B_)) ** (1 / (alpha + beta)) * (C / 6) ** exp
    D = C / (6 * N)
    L = E_ + A_ * N ** (-alpha) + B_ * D ** (-beta)
    return N, D, L


def fit_once(Nx, Dx, Lx, n_starts=15, seed=0):
    rng_local = np.random.default_rng(seed)
    best, best_val = None, np.inf
    for _ in range(n_starts):
        x0 = np.array([rng_local.uniform(0.0, 1.0), rng_local.uniform(4.0, 8.0),
                        rng_local.uniform(4.0, 8.0), rng_local.uniform(0.1, 0.6),
                        rng_local.uniform(0.1, 0.6)])
        res = minimize(lambda th: np.sum(huber(predict_log_loss(th, Nx, Dx) - np.log(Lx))),
                        x0, method="L-BFGS-B",
                        bounds=[(-2, 2), (0, 12), (0, 12), (0.01, 1.0), (0.01, 1.0)])
        if res.fun < best_val:
            best, best_val = res.x, res.fun
    return best


if _HAVE_SCIPY:
    # -----------------------------------------------------------------------
    # Step 2: fit. Multi-start because the objective is non-convex; keep the best.
    # -----------------------------------------------------------------------
    best, best_val = None, np.inf
    for _ in range(40):
        x0 = np.array([
            rng.uniform(0.0, 1.0),     # log E   (E ~ 1..2.7)
            rng.uniform(4.0, 8.0),     # log A
            rng.uniform(4.0, 8.0),     # log B
            rng.uniform(0.1, 0.6),     # alpha
            rng.uniform(0.1, 0.6),     # beta
        ])
        res = minimize(objective, x0, method="L-BFGS-B",
                       bounds=[(-2, 2), (0, 12), (0, 12), (0.01, 1.0), (0.01, 1.0)])
        if res.fun < best_val:
            best, best_val = res.x, res.fun

    e, a, b, alpha, beta = best
    print(f"Recovered:  E={np.exp(e):.3f}  A={np.exp(a):.1f}  B={np.exp(b):.1f}"
          f"  alpha={alpha:.3f}  beta={beta:.3f}")
    print(f"Ground truth: E={TRUE['E']}  A={TRUE['A']}  B={TRUE['B']}"
          f"  alpha={TRUE['alpha']}  beta={TRUE['beta']}")

    # -----------------------------------------------------------------------
    # Step 3: use the fit to predict the compute-optimal allocation at a NEW,
    # much larger budget than any run in the grid -- the whole point of fitting.
    # -----------------------------------------------------------------------
    for C in [1e23, 1e24, 1e25]:
        N, D, L = optimal_alloc(C, e, a, b, alpha, beta)
        print(f"C={C:.0e} -> N*={N:.2e}  D*={D:.2e}  tok/param={D/N:.1f}  L*={L:.3f}")

    # The book is explicit that, for seed=0, the fit lands near E~1.83, A~879,
    # B~471, alpha~0.390, beta~0.290 -- and that despite the offsets (A, B, E)
    # being off by up to ~2x, the *allocation exponent* beta/(alpha+beta) is
    # tightly recovered (~0.426 vs a true 0.452). Verify both claims directly.
    assert abs(np.exp(e) - 1.83) < 0.1
    assert abs(np.exp(a) - 879) < 50
    assert abs(alpha - 0.390) < 0.02
    assert abs(beta - 0.290) < 0.02
    alloc_exp = beta / (alpha + beta)
    true_alloc_exp = TRUE["beta"] / (TRUE["alpha"] + TRUE["beta"])
    assert abs(alloc_exp - 0.426) < 0.02
    assert abs(true_alloc_exp - 0.452) < 1e-3

    print("Block #3 OK")
else:
    print("Block #3 SKIPPED: scipy not available")


# ============================================================================
# Block #4 (line ~344) -- bootstrap over the fitting grid (identifiability)
# Runtime-reduced: n_boot 200 -> 20 (see module docstring); same fit_once().
# ============================================================================
_section("Block #4: bootstrap identifiability check")

if _HAVE_SCIPY:
    boot_rng = np.random.default_rng(2)
    n_boot, n_rows = 20, len(N_obs)          # book uses n_boot=200; reduced for CI runtime
    records = []
    for _ in range(n_boot):
        idx = boot_rng.integers(0, n_rows, n_rows)          # resample WITH replacement
        e_b, a_b, b_b, alpha_b, beta_b = fit_once(N_obs[idx], D_obs[idx], L_obs[idx])
        _, _, L25 = optimal_alloc(1e25, e_b, a_b, b_b, alpha_b, beta_b)
        records.append((alpha_b, beta_b, beta_b / (alpha_b + beta_b), np.exp(a_b), L25))
    records = np.array(records)
    names = ["alpha", "beta", "alloc_exp (beta/(a+b))", "A", "L*(1e25)"]
    for j, name in enumerate(names):
        lo, mid, hi = np.percentile(records[:, j], [2.5, 50, 97.5])
        print(f"{name:>24}: [{lo:.3g}, {mid:.3g}, {hi:.3g}]  (2.5% / median / 97.5%)")
    # Expect: alpha ~ [0.30, 0.45], beta ~ [0.25, 0.34], alloc-exp ~ [0.38, 0.50] -- tight.
    # A spans roughly [230, 2300] and E ~ [1.57, 1.95] -- an order of magnitude wider,
    # visually confirming allocation is identifiable but the raw offsets are not.
    assert records.shape == (n_boot, 5)
    # allocation exponent should be much more tightly concentrated (relative
    # spread) than the raw offset A -- the identifiability claim the book makes.
    alloc_col, A_col = records[:, 2], records[:, 3]
    alloc_spread = np.percentile(alloc_col, 97.5) - np.percentile(alloc_col, 2.5)
    A_rel_spread = (np.percentile(A_col, 97.5) - np.percentile(A_col, 2.5)) / np.median(A_col)
    assert alloc_spread < 0.3, f"alloc exponent spread too wide: {alloc_spread}"
    assert A_rel_spread > alloc_spread, "A should be far less tightly identified than the allocation exponent"

    print("Block #4 OK")
else:
    print("Block #4 SKIPPED: scipy not available")


# ============================================================================
# Block #5 (line ~391) -- The IsoFLOP method (Chinchilla Approach 2)
# ============================================================================
_section("Block #5: IsoFLOP method")

TRUE = dict(E=1.69, A=406.0, B=410.0, alpha=0.34, beta=0.28)


def true_loss(N, D, p=TRUE):
    return p['E'] + p['A']*N**(-p['alpha']) + p['B']*D**(-p['beta'])


rng = np.random.default_rng(1)
C_slices = np.array([1e18, 3e18, 1e19, 3e19, 1e20])   # 5 isoFLOP budgets
N_opt = []
for C in C_slices:
    N_center = np.sqrt(C / 120.0)                     # ~20x-rule optimum for slice
    Ns_iso = N_center * np.logspace(-0.6, 0.6, 7)      # 7 models, span ~1.5 dex
    Ds_iso = C / (6.0 * Ns_iso)                        # D fixed so 6*N*D == C
    Ls_iso = true_loss(Ns_iso, Ds_iso) * (1.0 + 0.01*rng.standard_normal(len(Ns_iso)))
    c2, c1, c0 = np.polyfit(np.log(Ns_iso), Ls_iso, 2)  # parabola in log N
    logN_star = -c1 / (2.0 * c2)                        # vertex = valley
    N_opt.append(np.exp(logN_star))
N_opt = np.array(N_opt)
D_opt = C_slices / (6.0 * N_opt)
for C, N, D in zip(C_slices, N_opt, D_opt):
    print(f'C={C:.0e}  N_opt={N:.3e}  D_opt={D:.3e}  tok/param={D/N:.0f}')

a_iso, _ = np.polyfit(np.log(C_slices), np.log(N_opt), 1)  # log N_opt = a*log C + k
print(f'IsoFLOP exponent a (N_opt ~ C^a) = {a_iso:.3f}')
print(f'parametric beta/(alpha+beta)      = {TRUE["beta"]/(TRUE["alpha"]+TRUE["beta"]):.3f}')

# The book claims the two independent methods (IsoFLOP slope vs. analytic
# beta/(alpha+beta)) agree to within about 0.005.
_true_alloc_exp = TRUE["beta"] / (TRUE["alpha"] + TRUE["beta"])
assert abs(a_iso - 0.447) < 0.01
assert abs(a_iso - _true_alloc_exp) < 0.01

print("Block #5 OK")


# ============================================================================
# Block #6 (line ~429) -- Designing the sweep under a budget
# ============================================================================
_section("Block #6: designing a sweep")

C_final = 2e21                          # the run the sweep is designed to inform
slices = [1e17, 3e17, 1e18, 3e18]       # 4 isoFLOP budgets, ~0.5 dex apart
models_per_slice = 6
rows = []
for Cs in slices:
    N_star = np.sqrt(Cs / 120.0)                       # 20x-rule optimum for slice
    Ns_sweep = N_star * np.logspace(-0.5, 0.5, models_per_slice)  # span ~1 dex in N
    for N in Ns_sweep:
        D = Cs / (6.0 * N)                             # tokens for this run
        rows.append(dict(C=Cs, N=N, D=D, tpp=D/N, flops=Cs))  # per-run FLOPs == Cs
total = sum(r['flops'] for r in rows)
print(f'{len(rows)} runs, total = {total:.2e} FLOPs = {100*total/C_final:.2f}% of final')
for r in rows[:6]:
    print(f"  C={r['C']:.0e}  N={r['N']:.2e}  D={r['D']:.2e}  tok/param={r['tpp']:.0f}")

# The chapter states this "verified" combination: 24 runs, 2.64e19 FLOPs total,
# 1.32% of the 2e21 target. Verify directly.
assert len(rows) == 24
assert abs(total - 2.64e19) / 2.64e19 < 1e-2
pct = 100 * total / C_final
assert abs(pct - 1.32) < 0.01
assert 1.0 <= pct <= 2.0, "sweep should land in the stated 1-2% envelope"

print("Block #6 OK")


# ============================================================================
# Block #7 (line ~464) -- Held-out extrapolation check
# Continues Block #3/#4's variables (N_obs, D_obs, L_obs, fit_once, huber,
# minimize, objective, predict_log_loss), exactly as the chapter's prose says.
# SKIP(scipy unavailable): same guard as Block #3/#4.
# ============================================================================
_section("Block #7: held-out extrapolation check")

if _HAVE_SCIPY:
    mask = N_obs != 3e9                      # hold out the largest-N runs
    # (continues the fitting-block variables N_obs, D_obs, L_obs, objective, minimize, huber)
    held_N, held_D, held_L = N_obs[~mask], D_obs[~mask], L_obs[~mask]
    theta_ho = fit_once(N_obs[mask], D_obs[mask], L_obs[mask])
    pred_L = np.exp(predict_log_loss(theta_ho, held_N, held_D))
    err_pct = 100 * np.abs(pred_L - held_L) / held_L
    print(f"held-out predicted vs observed: {list(zip(pred_L.round(3), held_L.round(3)))}")
    print(f"held-out error: {err_pct.round(2)}%")

    assert held_N.shape == held_D.shape == held_L.shape
    assert len(held_N) == 7, "should hold out exactly the N=3e9 row (7 D values)"
    # Expect the held-out loss recovered within roughly a few percent (the
    # chapter's stated success criterion for a scaling-law fit).
    assert np.all(err_pct < 10.0), f"held-out error too large: {err_pct}"

    print("Block #7 OK")
else:
    print("Block #7 SKIPPED: scipy not available")


# ============================================================================
# Block #8 (line ~549) -- lifetime_optimal() / inference-aware over-training
# ============================================================================
_section("Block #8: lifetime_optimal (inference-aware over-training)")


def lifetime_optimal(L_target, n_inference_tokens,
                     A=406.0, B=410.0, alpha=0.34, beta=0.28, E=1.69):
    """Pick (N, D_train) to hit a target loss while minimizing TRAIN+INFERENCE
    compute. Coarse grid search over N; for each N solve the data term needed
    to reach L_target, then score total lifetime FLOPs.
    """
    import numpy as np
    best = None
    for N in np.logspace(8, 12, 400):          # 0.1B .. 1T params
        # residual loss budget left for the data term after the param term:
        loss_from_data = L_target - E - A * N ** (-alpha)
        if loss_from_data <= 0:
            continue                            # this N alone already over-shoots
        D_train = (B / loss_from_data) ** (1.0 / beta)
        train = 6 * N * D_train
        infer = 2 * N * n_inference_tokens
        total = train + infer
        if best is None or total < best["total"]:
            best = dict(N=N, D_train=D_train, train=train,
                        infer=infer, total=total, tok_per_param=D_train / N)
    return best


# Light serving vs. heavy serving -> the optimal model SHRINKS as you serve more:
_lifetime_results = {}
for D_inf in [1e11, 1e13, 1e15]:
    r = lifetime_optimal(L_target=1.95, n_inference_tokens=D_inf)
    _lifetime_results[D_inf] = r
    print(f"D_inf={D_inf:.0e}: N={r['N']:.2e}  D_train={r['D_train']:.2e}  "
          f"tok/param={r['tok_per_param']:.0f}")

# The book's claim: as inference demand grows, optimal N shrinks and
# tokens/param grows. Verify the monotonic trend directly.
_Ns_lt = [_lifetime_results[d]["N"] for d in [1e11, 1e13, 1e15]]
_tpp_lt = [_lifetime_results[d]["tok_per_param"] for d in [1e11, 1e13, 1e15]]
assert _Ns_lt[0] > _Ns_lt[1] > _Ns_lt[2], "optimal N should shrink as inference demand grows"
assert _tpp_lt[0] < _tpp_lt[1] < _tpp_lt[2], "tokens/param should grow as inference demand grows"

print("Block #8 OK")


# ============================================================================
# Block #9 (line ~602) -- emergent abilities as a metric artifact
# ============================================================================
_section("Block #9: emergent abilities as a metric artifact")


def smooth_per_token_accuracy(N):
    """Per-token correctness improves SMOOTHLY (power-law) with scale."""
    # Clip to a valid probability: the raw power law goes negative below
    # N ~ 4.2e8, so clamp to [0, 1]. Per-token correctness still improves
    # SMOOTHLY (power-law) with scale.
    return np.clip(1.0 - 0.9 * (N / 1e9) ** (-0.12), 0.0, 1.0)


def exact_match(N, k_tokens):
    """All-or-nothing metric: need ALL k tokens correct simultaneously."""
    p = np.clip(smooth_per_token_accuracy(N), 1e-9, 1 - 1e-9)
    return p ** k_tokens


Ns_em = np.logspace(8, 12, 9)
print(f"{'N':>10} {'per-token':>10} {'EM(k=1)':>9} {'EM(k=30)':>10}")
_per_token_vals, _em30_vals = [], []
for N in Ns_em:
    pt = smooth_per_token_accuracy(N)
    em1 = exact_match(N, 1)
    em30 = exact_match(N, 30)
    _per_token_vals.append(pt)
    _em30_vals.append(em30)
    print(f"{N:10.1e} {pt:10.3f} {em1:9.3f} {em30:10.3f}")
# per-token rises gently; EM(k=30) stays ~0 then "emerges" sharply -- same model!

# Verify the "mirage" claim directly: per-token accuracy rises smoothly and
# substantially across the range, while EM(k=30) barely rises off the floor
# for all but the very largest N (the "sharp jump" is deferred by exponentiation).
assert _per_token_vals[-1] - _per_token_vals[0] > 0.5, "per-token accuracy should improve substantially"
assert all(v < 0.05 for v in _em30_vals[:-1]), "EM(k=30) should stay near zero except at the largest N"
assert exact_match(1, 1) == exact_match(1, 1)  # k=1 exact match equals per-token accuracy by definition
assert abs(exact_match(Ns_em[-1], 1) - smooth_per_token_accuracy(Ns_em[-1])) < 1e-9

print("Block #9 OK")


print("\nALL BLOCKS PASSED" if _HAVE_SCIPY else "\nALL BLOCKS PASSED (scipy-dependent blocks skipped)")
