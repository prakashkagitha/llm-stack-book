"""
Runs the CPU-runnable Python code blocks from:
    content/11-evaluation/06-statistical-rigor-eval.md

Blocks are copied faithfully (verbatim logic) and concatenated in document
order. Each block's own `if __name__ == "__main__":` demo doubles as the
"actually execute this" call site, since this file is itself run as
`__main__`. Where the book defines a function but does not call it in its
own demo (paired_t_test, wilcoxon_signed_rank, benjamini_hochberg,
item_information, adaptive_eval), a small glue call is added immediately
after the block so every tested function actually runs.

`scipy` is used throughout the chapter (scipy.stats, scipy.optimize) but is
NOT on the guaranteed-CI import list (numpy/torch/einops/sklearn/stdlib
only), so it is imported defensively; blocks/demos that need it are gated
on HAS_SCIPY and print a SKIP line instead of executing if it's absent.

Tested blocks:  #0, #1, #2, #3, #4, #5, #6, #7
Skipped blocks: (none) -- all 8 blocks are pure numpy/scipy and CPU-safe.
                 Individual demos additionally SKIP at runtime if scipy is
                 unavailable (see HAS_SCIPY below); this environment has
                 scipy installed so all 8 blocks execute for real here.

Fixture-size note (block #6): the book's own demo fits a 2PL IRT model with
n_models=30, n_items=120, n_iter=300 -- each outer iteration runs ~150
scipy.optimize.minimize (BFGS) calls, i.e. ~45,000 optimizer calls total.
That is far too slow for a CPU smoke test, so the demo here uses a much
smaller n_models/n_items/n_iter (the `fit_2pl` function itself is
unmodified -- only the fixture size shrinks, per the "tiny shapes" glue
allowance).
"""

import numpy as np

try:
    from scipy import stats
    from scipy.optimize import minimize, minimize_scalar
    HAS_SCIPY = True
except Exception:
    stats = None
    minimize = None
    minimize_scalar = None
    HAS_SCIPY = False

print(f"HAS_SCIPY = {HAS_SCIPY}")


# ============================================================================
# Block #0 (chapter line ~79): bootstrap CI, BCa bootstrap, Wilson,
# Clopper-Pearson intervals for binomial/arbitrary statistics.
# ============================================================================

def bootstrap_ci(scores, statistic=np.mean, n_boot=10_000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for an arbitrary statistic of per-item scores.

    `scores`    : 1-D array of per-item results (0/1 for pass-fail, or floats
                  for judge scores, F1, etc.).
    `statistic` : any function array -> scalar (mean, median, np.percentile...).
    Returns (point_estimate, lo, hi).
    """
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    point = statistic(scores)
    # Vectorized resampling: draw an (n_boot x n) matrix of indices at once.
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = np.array([statistic(scores[row]) for row in idx])
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, lo, hi


def bca_bootstrap_ci(scores, statistic=np.mean, n_boot=10_000, alpha=0.05, seed=0):
    """Bias-Corrected and accelerated (BCa) bootstrap CI (Efron, 1987).

    Corrects the percentile interval for (1) median bias and (2) skew of the
    sampling distribution via a jackknife acceleration estimate. This is the
    interval to trust for skewed statistics like Elo or 90th-percentile latency.
    """
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    theta_hat = statistic(scores)

    # 1. Bootstrap replicates.
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = np.array([statistic(scores[row]) for row in idx])

    # 2. Bias-correction z0: how often replicates fall below the point estimate.
    prop = np.mean(boot < theta_hat)
    prop = min(max(prop, 1e-6), 1 - 1e-6)  # guard against 0/1 -> +-inf
    z0 = stats.norm.ppf(prop)

    # 3. Acceleration a from the jackknife (leave-one-out) distribution.
    jack = np.array([statistic(np.delete(scores, i)) for i in range(n)])
    jack_mean = jack.mean()
    num = np.sum((jack_mean - jack) ** 3)
    den = 6.0 * (np.sum((jack_mean - jack) ** 2) ** 1.5) + 1e-12
    a = num / den

    # 4. Adjusted percentiles.
    z_lo, z_hi = stats.norm.ppf(alpha / 2), stats.norm.ppf(1 - alpha / 2)
    def adjust(z):
        return stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    p_lo, p_hi = adjust(z_lo), adjust(z_hi)
    lo, hi = np.percentile(boot, [100 * p_lo, 100 * p_hi])
    return theta_hat, lo, hi


def wilson_interval(k, n, alpha=0.05):
    """Wilson score interval for a binomial proportion. Well-behaved near 0/1."""
    if n == 0:
        return (0.0, 0.0, 1.0)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (p, center - half, center + half)


def clopper_pearson(k, n, alpha=0.05):
    """Exact (Clopper-Pearson) binomial interval via the Beta distribution.

    Guaranteed >= 95% coverage (conservative). Good for tiny n or extreme p.
    """
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (k / n, lo, hi)


if __name__ == "__main__" and HAS_SCIPY:
    print("\n=== Block #0: bootstrap / BCa / Wilson / Clopper-Pearson ===")
    # 420 of 500 correct -> 84% accuracy.
    correct = np.concatenate([np.ones(420), np.zeros(80)])
    print("bootstrap   :", bootstrap_ci(correct))
    print("BCa         :", bca_bootstrap_ci(correct))
    print("Wilson      :", wilson_interval(420, 500))
    print("ClopperPears:", clopper_pearson(420, 500))
    # Extreme case: 49/50 correct -> Wald would be nonsense.
    print("Wilson 49/50:", wilson_interval(49, 50))
    print("Wald   49/50:", (0.98, 0.98 - 1.96 * (0.98 * 0.02 / 50) ** 0.5,
                                  0.98 + 1.96 * (0.98 * 0.02 / 50) ** 0.5))

    # Sanity checks: all four 420/500 intervals should bracket the point
    # estimate and stay inside [0, 1].
    for name, (pt, lo, hi) in [
        ("bootstrap", bootstrap_ci(correct)),
        ("BCa", bca_bootstrap_ci(correct)),
        ("wilson", wilson_interval(420, 500)),
        ("clopper_pearson", clopper_pearson(420, 500)),
    ]:
        assert 0.0 <= lo <= pt <= hi <= 1.0, f"{name} interval out of range: {(lo, pt, hi)}"

    # The Wilson upper bound at 49/50 must stay <= 1 (unlike Wald, which
    # overshoots to 1.019 as the book notes).
    _, w_lo, w_hi = wilson_interval(49, 50)
    assert w_hi <= 1.0
elif __name__ == "__main__":
    print("SKIP block #0 (scipy unavailable): bootstrap/BCa/Wilson/Clopper-Pearson")


# ============================================================================
# Block #1 (chapter line ~191): Bradley-Terry / Elo fitting + bootstrap CI
# over battles. Pure numpy, no scipy dependency.
# ============================================================================

def fit_bradley_terry(battles, models, scale=400.0, base=10.0,
                      n_iter=200, lr=0.05, anchor=1000.0):
    """Fit BT/Elo ratings by gradient ascent on the log-likelihood.

    battles : list of (winner_idx, loser_idx). Ties can be split into two
              half-weight battles or dropped; we drop them here for clarity.
    Returns a rating per model, mean-anchored to `anchor`.
    """
    m = len(models)
    R = np.zeros(m)
    c = np.log(base) / scale  # convert rating-diff to logit units
    w = np.array([b[0] for b in battles])
    l = np.array([b[1] for b in battles])
    for _ in range(n_iter):
        # P(winner beats loser) under current ratings.
        p = 1.0 / (1.0 + np.exp(-c * (R[w] - R[l])))
        grad = np.zeros(m)
        # d/dR of sum log p: winner gets +(1-p), loser gets -(1-p).
        np.add.at(grad, w, c * (1.0 - p))
        np.add.at(grad, l, -c * (1.0 - p))
        R += lr * grad
        R -= R.mean()            # remove the unidentifiable global shift
    return R - R.mean() + anchor


def bootstrap_elo(battles, models, n_boot=1000, seed=0):
    """Bootstrap CI on Elo by resampling battles with replacement."""
    rng = np.random.default_rng(seed)
    battles = np.asarray(battles)
    n = len(battles)
    boots = np.zeros((n_boot, len(models)))
    for b in range(n_boot):
        sample = battles[rng.integers(0, n, size=n)]
        boots[b] = fit_bradley_terry([tuple(x) for x in sample], models)
    point = fit_bradley_terry([tuple(x) for x in battles], models)
    lo = np.percentile(boots, 2.5, axis=0)
    hi = np.percentile(boots, 97.5, axis=0)
    return {models[i]: (round(point[i]), round(lo[i]), round(hi[i]))
            for i in range(len(models))}


if __name__ == "__main__":
    print("\n=== Block #1: Bradley-Terry / Elo bootstrap ===")
    models = ["A", "B", "C"]
    # Synthetic battles: A strong, B middling, C weak.
    rng = np.random.default_rng(1)
    true_R = {"A": 1150, "B": 1000, "C": 880}
    battles = []
    pairs = [(0, 1), (0, 2), (1, 2)]
    for _ in range(300):
        i, j = pairs[rng.integers(0, 3)]
        pi = 1 / (1 + 10 ** (-(true_R[models[i]] - true_R[models[j]]) / 400))
        if rng.random() < pi:
            battles.append((i, j))
        else:
            battles.append((j, i))
    elo_result = bootstrap_elo(battles, models)
    for name, (pt, lo, hi) in elo_result.items():
        print(f"{name}: {pt}  95% CI [{lo}, {hi}]  width={hi - lo}")

    # Sanity: A (strongest) should rank above C (weakest) by point estimate.
    assert elo_result["A"][0] > elo_result["C"][0]


# ============================================================================
# Block #2 (chapter line ~292): McNemar, paired t-test, Wilcoxon signed-rank,
# paired bootstrap on the accuracy difference.
# ============================================================================

def mcnemar_test(a_correct, b_correct, exact=True):
    """Paired test for two models scored pass/fail on the SAME items.

    a_correct, b_correct : boolean arrays of per-item correctness.
    Returns (b, c, p_value) where
        b = # items A right & B wrong,
        c = # items A wrong & B right.
    """
    a = np.asarray(a_correct, dtype=bool)
    b_ = np.asarray(b_correct, dtype=bool)
    b = int(np.sum(a & ~b_))   # A beats B on this item
    c = int(np.sum(~a & b_))   # B beats A on this item
    n_disc = b + c
    if n_disc == 0:
        return b, c, 1.0       # models never disagree -> no evidence
    if exact and n_disc < 25:
        # Exact two-sided binomial test on the discordant pairs.
        p = stats.binomtest(b, n_disc, 0.5).pvalue
    else:
        chi2 = (abs(b - c) - 1) ** 2 / n_disc   # continuity-corrected
        p = stats.chi2.sf(chi2, df=1)
    return b, c, p


def paired_t_test(scores_a, scores_b):
    """Paired t-test on per-item numeric scores (e.g. judge ratings 1-10)."""
    d = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    n = len(d)
    mean_d = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    t = mean_d / se if se > 0 else 0.0
    p = 2 * stats.t.sf(abs(t), df=n - 1)
    # Effect size: Cohen's d_z for paired designs.
    dz = mean_d / d.std(ddof=1) if d.std(ddof=1) > 0 else 0.0
    return {"mean_diff": mean_d, "t": t, "df": n - 1, "p": p, "cohen_dz": dz}


def wilcoxon_signed_rank(scores_a, scores_b):
    """Non-parametric paired test; robust to non-normal per-item diffs."""
    a, b = np.asarray(scores_a, float), np.asarray(scores_b, float)
    res = stats.wilcoxon(a, b, zero_method="wilcox", correction=False,
                         alternative="two-sided")
    return {"statistic": res.statistic, "p": res.pvalue}


def paired_bootstrap_diff(a_correct, b_correct, n_boot=10_000, seed=0):
    """Bootstrap CI on the accuracy DIFFERENCE, resampling item indices once
    and applying the SAME resample to both models (this preserves pairing)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a_correct, float)
    b = np.asarray(b_correct, float)
    n = len(a)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    point = a.mean() - b.mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, lo, hi


if __name__ == "__main__" and HAS_SCIPY:
    print("\n=== Block #2: McNemar / paired t / Wilcoxon / paired bootstrap ===")
    rng = np.random.default_rng(7)
    n = 500
    # Shared latent difficulty -> strong positive correlation between models.
    difficulty = rng.normal(0, 1, n)
    skill_a, skill_b = 0.45, 0.30          # A is genuinely a bit better
    a_correct = (rng.normal(skill_a, 0.6, n) > difficulty)
    b_correct = (rng.normal(skill_b, 0.6, n) > difficulty)

    print("acc A:", a_correct.mean(), " acc B:", b_correct.mean())
    b, c, p = mcnemar_test(a_correct, b_correct)
    print(f"McNemar: b(A>B)={b}, c(B>A)={c}, p={p:.4f}")
    pt, lo, hi = paired_bootstrap_diff(a_correct, b_correct)
    print(f"paired bootstrap diff: {pt:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")

    # --- Glue: the book defines paired_t_test and wilcoxon_signed_rank but
    # only demos McNemar/paired-bootstrap in its __main__ (since its running
    # example is pass/fail, not continuous scores). Exercise the other two
    # on a small synthetic pair of continuous judge-score vectors. ---
    judge_a = rng.normal(7.0, 1.0, 50)
    judge_b = judge_a - rng.normal(0.3, 0.5, 50)  # A tends to score slightly higher
    tt = paired_t_test(judge_a, judge_b)
    print("paired t-test:", tt)
    wres = wilcoxon_signed_rank(judge_a, judge_b)
    print("wilcoxon signed-rank:", wres)

    assert 0.0 <= p <= 1.0
    assert 0.0 <= tt["p"] <= 1.0
    assert 0.0 <= wres["p"] <= 1.0
    assert tt["mean_diff"] > 0  # judge_a constructed to score higher on average
elif __name__ == "__main__":
    print("SKIP block #2 (scipy unavailable): McNemar/paired-t/Wilcoxon")


# ============================================================================
# Block #3 (chapter line ~393): Benjamini-Hochberg FDR correction.
# Pure numpy, no scipy dependency.
# ============================================================================

def benjamini_hochberg(pvals, alpha=0.05):
    """Return a boolean mask of which hypotheses are rejected under BH-FDR."""
    p = np.asarray(pvals)
    m = len(p)
    order = np.argsort(p)
    thresh = (np.arange(1, m + 1) / m) * alpha
    passed = p[order] <= thresh
    if not passed.any():
        return np.zeros(m, dtype=bool)
    k_max = np.max(np.where(passed))      # largest index that passes
    cutoff = p[order][k_max]
    return p <= cutoff


if __name__ == "__main__":
    print("\n=== Block #3: Benjamini-Hochberg FDR ===")
    # --- Glue: the book defines benjamini_hochberg but has no __main__ demo
    # for this block. Call it on a small mixed set of p-values (some clearly
    # significant, some clearly not). ---
    demo_pvals = np.array([0.001, 0.02, 0.03, 0.04, 0.20, 0.45, 0.60, 0.80])
    rejected = benjamini_hochberg(demo_pvals, alpha=0.05)
    print("p-values:", demo_pvals)
    print("BH-FDR rejected:", rejected)
    assert rejected.dtype == bool
    assert rejected.shape == demo_pvals.shape
    # The smallest p-value should always survive BH when it's well below alpha.
    assert rejected[np.argmin(demo_pvals)]
    # The largest, clearly-insignificant p-value should not.
    assert not rejected[np.argmax(demo_pvals)]


# ============================================================================
# Block #4 (chapter line ~438): variance-components estimation (item/prompt/
# seed) from a crossed grid. Pure numpy, no scipy dependency.
# ============================================================================

def variance_components(scores):
    """Estimate item / prompt / seed variance components from a crossed grid.

    scores : array of shape (n_items, n_prompts, n_seeds), each entry the
             per-(item,prompt,seed) score for ONE model (0/1 or float).
    Returns a dict of variance components and the implied SE of the grand mean.
    """
    s = np.asarray(scores, dtype=float)
    n_i, n_t, n_k = s.shape
    grand = s.mean()

    # Marginal means along each axis.
    item_means   = s.mean(axis=(1, 2))   # average over prompts & seeds
    prompt_means = s.mean(axis=(0, 2))   # average over items & seeds
    # Variance of marginal means, de-biased by the within noise they still carry
    # (Method-of-moments; fine for a diagnostic, use REML for a paper.)
    var_item   = max(item_means.var(ddof=1)   - 0.0, 0.0)
    var_prompt = max(prompt_means.var(ddof=1) - 0.0, 0.0)
    # Seed/residual: variance within an (item,prompt) cell, averaged.
    within = s.var(axis=2, ddof=1).mean() if n_k > 1 else 0.0

    se_mean = np.sqrt(var_item / n_i + var_prompt / n_t + within / (n_i * n_k))
    return {
        "grand_mean": grand,
        "var_item": var_item,
        "var_prompt": var_prompt,
        "var_seed_resid": within,
        "se_of_reported_mean": se_mean,
    }


if __name__ == "__main__":
    print("\n=== Block #4: variance decomposition (item/prompt/seed) ===")
    rng = np.random.default_rng(0)
    n_i, n_t, n_k = 200, 4, 3
    item_eff   = rng.normal(0, 1.2, n_i)      # big item-difficulty spread
    prompt_eff = rng.normal(0, 0.5, n_t)      # nontrivial prompt sensitivity
    logits = (0.4 + item_eff[:, None, None] + prompt_eff[None, :, None]
              + rng.normal(0, 0.3, (n_i, n_t, n_k)))
    scores = (logits > 0).astype(float)       # pass/fail
    vc = variance_components(scores)
    for k, v in vc.items():
        print(f"{k:>22}: {v:.4f}")

    assert vc["var_item"] >= 0 and vc["var_prompt"] >= 0 and vc["var_seed_resid"] >= 0
    assert vc["se_of_reported_mean"] > 0


# ============================================================================
# Block #5 (chapter line ~522): power analysis (closed-form + simulation)
# for unpaired proportions and paired McNemar.
# ============================================================================

def n_for_unpaired_proportions(p1, p2, alpha=0.05, power=0.80):
    """Sample size PER ARM to detect a difference between two proportions."""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    delta = abs(p1 - p2)
    return int(np.ceil((z_a + z_b) ** 2 * 2 * pbar * (1 - pbar) / delta ** 2))


def n_for_mcnemar(p_b, p_c, alpha=0.05, power=0.80):
    """Number of items for a paired McNemar test.

    p_b : P(A right, B wrong);  p_c : P(A wrong, B right).
    Larger discordance (p_b+p_c) -> fewer items needed.
    """
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    pd = p_b + p_c
    diff = p_b - p_c
    num = (z_a * np.sqrt(pd) + z_b * np.sqrt(pd - diff ** 2)) ** 2
    return int(np.ceil(num / diff ** 2))


def simulate_power(p_both, p_b, p_c, n, n_sims=4000, alpha=0.05, seed=0):
    """Monte-Carlo power for the paired McNemar test at sample size n.

    Each item falls into one of four cells with the given probabilities:
      both-right, A-right-only (p_b), B-right-only (p_c), both-wrong.
    """
    rng = np.random.default_rng(seed)
    # BUG FIX (mirrors content/11-evaluation/06-statistical-rigor-eval.md):
    # the book's own demo values (p_both=0.80, p_b=0.115, p_c=0.085) sum to
    # exactly 1.0 in exact arithmetic, but floating point makes
    # `1 - p_both - p_b - p_c` a tiny negative number (-5.55e-17), which
    # crashes rng.multinomial (probabilities must lie in [0, 1]). Clamp.
    p_neither = max(0.0, 1 - p_both - p_b - p_c)
    probs = [p_both, p_b, p_c, p_neither]
    rejects = 0
    for _ in range(n_sims):
        counts = rng.multinomial(n, probs)
        b, c = counts[1], counts[2]
        nd = b + c
        if nd == 0:
            pval = 1.0
        elif nd < 25:
            pval = stats.binomtest(b, nd, 0.5).pvalue
        else:
            chi2 = (abs(b - c) - 1) ** 2 / nd
            pval = stats.chi2.sf(chi2, df=1)
        rejects += (pval < alpha)
    return rejects / n_sims


if __name__ == "__main__" and HAS_SCIPY:
    print("\n=== Block #5: power analysis (closed-form + simulation) ===")
    # We want to detect a true 3-point accuracy edge for A.
    # Suppose models agree 80% of the time; of the 20% discordant items,
    # A wins on 11.5% and B on 8.5% -> a 3-point gap.
    p_b, p_c = 0.115, 0.085
    n_closed = n_for_mcnemar(p_b, p_c)
    print(f"closed-form n for 80% power: {n_closed}")
    for n in [n_closed, 2 * n_closed]:
        pw = simulate_power(p_both=0.80, p_b=p_b, p_c=p_c, n=n, n_sims=1500)
        print(f"  n={n:5d} -> simulated power {pw:.2f}")
    # Compare to the UNPAIRED requirement for the same 3-point gap at p~0.8.
    print("unpaired n PER ARM:", n_for_unpaired_proportions(0.83, 0.80))

    assert n_closed > 0
    # More items should never give strictly less power (monotonic in n).
    pw_small = simulate_power(p_both=0.80, p_b=p_b, p_c=p_c, n=n_closed, n_sims=1500, seed=1)
    pw_large = simulate_power(p_both=0.80, p_b=p_b, p_c=p_c, n=4 * n_closed, n_sims=1500, seed=1)
    assert pw_large >= pw_small - 0.05  # small Monte-Carlo slack
elif __name__ == "__main__":
    print("SKIP block #5 (scipy unavailable): power analysis")


# ============================================================================
# Block #6 (chapter line ~628): 2PL IRT model fit + item information.
#
# Fixture-size note: the book's own demo uses n_models=30, n_items=120,
# n_iter=300 (~45,000 scipy.optimize.minimize calls) -- too slow for a CPU
# smoke test. `fit_2pl` itself is unmodified; only the demo's problem size
# and iteration count are shrunk (tiny-shapes glue allowance).
# ============================================================================

def fit_2pl(R, n_iter=300):
    """Fit a 2PL IRT model by alternating MAP estimation.

    R : (n_models x n_items) binary response matrix (1 = correct).
    Returns theta (ability per model), a (discrimination), b (difficulty).
    Priors: theta,b ~ N(0,1); log a ~ N(0,1) keeps discriminations positive.
    """
    n_m, n_i = R.shape
    theta = np.zeros(n_m)
    a = np.ones(n_i)
    b = np.zeros(n_i)

    def sig(x):
        return 1.0 / (1.0 + np.exp(-x))

    for _ in range(n_iter):
        # --- E-ish step: update abilities given items ---
        for m in range(n_m):
            def negll_theta(t):
                p = sig(a * (t[0] - b))
                ll = np.sum(R[m] * np.log(p + 1e-9) +
                            (1 - R[m]) * np.log(1 - p + 1e-9))
                return -(ll - 0.5 * t[0] ** 2)        # + N(0,1) prior
            theta[m] = minimize(negll_theta, [theta[m]], method="BFGS").x[0]
        theta -= theta.mean()                         # identifiability anchor

        # --- M-ish step: update item params given abilities ---
        for i in range(n_i):
            def negll_item(par):
                a_i, b_i = np.exp(par[0]), par[1]     # exp keeps a_i > 0
                p = sig(a_i * (theta - b_i))
                ll = np.sum(R[:, i] * np.log(p + 1e-9) +
                            (1 - R[:, i]) * np.log(1 - p + 1e-9))
                return -(ll - 0.5 * par[0] ** 2 - 0.5 * b_i ** 2)
            res = minimize(negll_item, [np.log(a[i]), b[i]], method="BFGS")
            a[i], b[i] = np.exp(res.x[0]), res.x[1]
    return theta, a, b


def item_information(theta, a, b):
    """Fisher information each item gives about a model at ability theta."""
    p = 1.0 / (1.0 + np.exp(-a * (theta - b)))
    return a ** 2 * p * (1 - p)


if __name__ == "__main__" and HAS_SCIPY:
    print("\n=== Block #6: 2PL IRT fit + item information ===")
    rng = np.random.default_rng(0)
    # Shrunk from the book's n_m=30, n_i=120, n_iter=300 (see module docstring).
    n_m, n_i = 8, 15
    true_theta = rng.normal(0, 1, n_m)
    true_a = np.abs(rng.normal(1.0, 0.4, n_i)) + 0.2
    true_b = rng.normal(0, 1, n_i)
    P = 1 / (1 + np.exp(-true_a * (true_theta[:, None] - true_b[None, :])))
    R = (rng.random((n_m, n_i)) < P).astype(float)

    theta, a, b = fit_2pl(R, n_iter=8)
    # Recovery check: correlation between true and estimated ability.
    print("ability recovery r =", np.corrcoef(true_theta, theta)[0, 1].round(3))
    # The most discriminating items -- these are worth their weight.
    top = np.argsort(a)[::-1][:5]
    print("most informative items (idx, a, b):",
          [(int(i), round(a[i], 2), round(b[i], 2)) for i in top])

    # --- Glue: the book defines item_information but its __main__ demo for
    # this block never calls it directly (only fit_2pl). Call it here. ---
    info_at_zero = item_information(0.0, a, b)
    print("item_information at theta=0:", np.round(info_at_zero, 3))

    assert theta.shape == (n_m,)
    assert a.shape == (n_i,) and np.all(a > 0)
    assert info_at_zero.shape == (n_i,)
    assert np.all(info_at_zero >= 0)
elif __name__ == "__main__":
    print("SKIP block #6 (scipy unavailable): 2PL IRT fit")


# ============================================================================
# Block #7 (chapter line ~700): adaptive (computerized adaptive testing)
# evaluation loop against a pre-calibrated item bank.
# ============================================================================

def adaptive_eval(answer_fn, a, b, max_items=40, target_se=0.30):
    """Computerized adaptive test against a pre-calibrated item bank.

    answer_fn(i) -> 0/1 : runs the model on item i, returns correctness.
    a, b                : calibrated discrimination/difficulty per item.
    Stops when SE(theta_hat) < target_se or max_items reached.
    """
    asked, resp = [], []
    theta = 0.0
    remaining = list(range(len(a)))
    for _ in range(max_items):
        # Pick the unasked item with max Fisher information at current theta.
        info = [(item_information(theta, a[i], b[i]), i) for i in remaining]
        _, best = max(info)
        y = answer_fn(best)
        asked.append(best); resp.append(y); remaining.remove(best)

        # Re-estimate ability by MAP over items asked so far.
        aa, bb, yy = a[asked], b[asked], np.array(resp, float)
        def negll(t):
            p = 1 / (1 + np.exp(-aa * (t - bb)))
            return -(np.sum(yy * np.log(p + 1e-9) +
                            (1 - yy) * np.log(1 - p + 1e-9)) - 0.5 * t ** 2)
        theta = minimize_scalar(negll, bounds=(-4, 4), method="bounded").x

        # SE from the inverse total information at the current estimate.
        tot_info = sum(item_information(theta, a[i], b[i]) for i in asked) + 1.0
        se = 1 / np.sqrt(tot_info)
        if se < target_se:
            break
    return theta, se, len(asked)


def item_information(theta, a, b):
    p = 1.0 / (1.0 + np.exp(-a * (theta - b)))
    return a ** 2 * p * (1 - p)


if __name__ == "__main__" and HAS_SCIPY:
    print("\n=== Block #7: adaptive evaluation (CAT loop) ===")
    # --- Glue: the book has no __main__ demo for this block at all (it's
    # presented as the calibrated-bank consumer of block #6's fit_2pl
    # output). Reuse block #6's fitted (a, b) item bank as the "calibrated"
    # bank, and simulate a model with a known target ability answering
    # items via a seeded Bernoulli draw from the 2PL response probability. ---
    adaptive_rng = np.random.default_rng(42)
    target_theta = 0.5

    def answer_fn(i):
        p = 1.0 / (1.0 + np.exp(-a[i] * (target_theta - b[i])))
        return int(adaptive_rng.random() < p)

    final_theta, final_se, n_asked = adaptive_eval(
        answer_fn, a, b, max_items=10, target_se=0.5)
    print(f"adaptive_eval: theta={final_theta:.3f} se={final_se:.3f} "
          f"items_asked={n_asked}")

    assert 1 <= n_asked <= 10
    assert final_se > 0
    assert -4.0 <= final_theta <= 4.0
elif __name__ == "__main__":
    print("SKIP block #7 (scipy unavailable): adaptive evaluation")


if __name__ == "__main__":
    print("\nAll blocks completed.")
