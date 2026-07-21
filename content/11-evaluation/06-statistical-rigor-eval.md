# 11.6 Statistical Rigor in Evaluation: Confidence Intervals & Significance

A leaderboard says model A scores 84.2 % and model B scores 83.6 % on a 500-question benchmark. The press release announces "a new state of the art." A careful engineer asks a different question: *is 0.6 percentage points even distinguishable from noise?* With 500 questions, the standard error on each number is around 1.6 points, so the gap is well inside the uncertainty. The "improvement" might be a coin flip.

This chapter is about treating evaluation as an experiment, not a scoreboard. Every benchmark number is a *random variable*: it depends on which items you sampled, which random seeds you used, which prompts and decoding parameters you chose, and — when a model grades a model — which judge you trusted. A score reported without an interval is a measurement reported without error bars, and in any other branch of empirical science that would be unpublishable.

We build the toolkit a scientist would demand: bootstrap confidence intervals on accuracy and Elo, paired significance tests (paired $t$, McNemar, Wilcoxon) for the head-to-head A-vs-B comparison that practitioners actually care about, a variance decomposition that tells you *where* your noise lives (items? seeds? judges?), and a power analysis to size a test set *before* you spend money running it. We close with item-response theory (IRT) and adaptive evaluation — how to extract more signal per item — and the single most important leaderboard-reading skill: knowing that **overlapping confidence intervals mean "no demonstrated gap."**

This chapter is the statistical spine of Part XI. It assumes you have met the benchmark landscape in [The Evaluation Problem & Benchmark Landscape](../11-evaluation/01-eval-landscape.html), the judge paradigm in [LLM-as-a-Judge & Automated Evaluation](../11-evaluation/02-llm-as-judge.html), and harness mechanics in [Building Eval Harnesses](../11-evaluation/03-eval-harnesses.html). The probability foundations live in [Probability, Statistics & Information Theory](../01-foundations/02-probability-information.html). Online A/B testing — the production cousin of everything here — is in [Online Evaluation: A/B Testing, Canaries & Guardrail Metrics](../12-production-mlops/07-online-eval-ab-testing.html).

---

## Why Every Eval Number Needs an Error Bar

### Accuracy is a sample statistic

When you evaluate a model on a benchmark, you score it on $n$ items drawn from some universe of possible items. The accuracy you report, $\hat p$, is an *estimate* of the true accuracy $p$ — the rate the model would achieve on the entire (infinite, hypothetical) population the benchmark is meant to represent. Because $\hat p$ is computed from a finite sample, it has sampling variance.

If each item is scored pass/fail (a Bernoulli trial) and items are independent, then the number correct is $\text{Binomial}(n, p)$, and the standard error of the proportion is

$$
\text{SE}(\hat p) = \sqrt{\frac{p(1-p)}{n}} \approx \sqrt{\frac{\hat p(1-\hat p)}{n}}.
$$

The familiar normal-approximation ("Wald") 95 % confidence interval is $\hat p \pm 1.96\,\text{SE}$. Plug in numbers: at $\hat p = 0.84$, $n = 500$,

$$
\text{SE} = \sqrt{\frac{0.84 \times 0.16}{500}} = \sqrt{0.0002688} \approx 0.0164,
$$

so the 95 % CI is roughly $0.84 \pm 0.032$, i.e. $[0.808, 0.872]$. That is a band more than six points wide. A rival at 0.836 sits squarely inside it. **There is no measurable difference.** The width shrinks only as $1/\sqrt n$: to halve the interval you must quadruple the test set.

### The width-of-CI table you should memorize

The half-width $1.96\sqrt{p(1-p)/n}$ near $p=0.5$ (the worst case, where variance is maximal) gives a handy rule of thumb: the 95 % margin of error is about $1/\sqrt n$. Some magnitudes worth carrying in your head:

| $n$ (items) | Margin of error at $p\approx0.5$ | Margin at $p=0.9$ |
|-------------|-----------------------------------|--------------------|
| 100         | $\pm 9.8$ pts                      | $\pm 5.9$ pts       |
| 500         | $\pm 4.4$ pts                      | $\pm 2.6$ pts       |
| 1,000       | $\pm 3.1$ pts                      | $\pm 1.9$ pts       |
| 5,000       | $\pm 1.4$ pts                      | $\pm 0.8$ pts       |
| 10,000      | $\pm 1.0$ pts                      | $\pm 0.6$ pts       |

So MMLU (around 14,000 items) can resolve differences of roughly a point; a bespoke 200-item eval cannot reliably resolve anything smaller than 7 points. This single table kills most "our model beats theirs by 0.4 points" claims.

!!! warning "The Wald interval lies at the extremes"

    The normal-approximation interval is badly behaved when $\hat p$ is near 0 or 1, or when $n$ is small: it can produce limits below 0 or above 1, and its true coverage can dip well under 95 %. If your model scores 49/50 ($\hat p = 0.98$), do **not** report $0.98 \pm 1.96\sqrt{0.98\times0.02/50} = 0.98 \pm 0.039$. Use a **Wilson score interval** or **Clopper–Pearson exact interval** instead (both shown below), or bootstrap. For pass/fail metrics the Wilson interval is the pragmatic default.

### Two kinds of variance: items vs. generation

There are actually *two* independent sources of randomness, and conflating them is a classic mistake:

1. **Item sampling variance.** You happened to draw *these* questions. A different draw of 500 questions gives a different score. This is the binomial variance above, and it is what a confidence interval over items captures.
2. **Generation/seed variance.** For a fixed question, a model with temperature $>0$ produces different outputs on different runs (different seeds), and may pass on one run and fail on the next. Greedy decoding ($T=0$) removes this *in principle*, but in practice non-determinism from GPU kernel reductions, batching, and floating-point non-associativity means even "greedy" runs can differ — see [Numerical Computing, Floating Point & Precision](../01-foundations/04-numerics-precision.html) and the sampling discussion in [Sampling Strategies & Decoding Algorithms](../07-inference-serving/09-sampling-decoding.html).

If you report a single greedy run, you have *no estimate at all* of generation variance — you are pretending it is zero. The honest protocol is to run each item $k$ times (e.g. $k=3$ or $k=5$) at your production temperature and report mean accuracy with an interval that accounts for both axes. We formalize this with a variance decomposition later in the chapter.

---

## Confidence Intervals That Don't Lie: Bootstrap & Beyond

### The bootstrap in one paragraph

The bootstrap (Efron, 1979) is the Swiss-army knife of eval statistics because it makes almost no distributional assumptions and works for *any* statistic — accuracy, F1, Elo, mean judge score, BLEU, a 90th-percentile latency. The idea: your sample of $n$ scored items *is* your best estimate of the population. To learn how much your statistic would wobble across hypothetical re-samples of the population, you re-sample **with replacement** from your own data, recompute the statistic, and repeat $B$ times (say $B=10{,}000$). The spread of those $B$ recomputed values approximates the sampling distribution of your statistic. The 2.5th and 97.5th percentiles give a 95 % **percentile bootstrap interval**.

$$
\hat\theta^{*(b)} = s\!\left(\{x_{i}^{*(b)}\}_{i=1}^{n}\right), \quad x^{*(b)} \sim \text{resample}(x_1,\dots,x_n),
\qquad \text{CI}_{95\%} = \left[\hat\theta^{*}_{(2.5\%)},\ \hat\theta^{*}_{(97.5\%)}\right].
$$

{{fig:statrig-bootstrap-resample}}

### Bootstrap CI, Wilson, and Clopper–Pearson — from scratch

Here is a complete, dependency-light implementation you can drop into an eval harness. It computes a percentile bootstrap CI, a BCa (bias-corrected and accelerated) bootstrap CI — which corrects for skew and is the one to prefer — and the two analytic intervals for binary outcomes.

```python
import numpy as np
from scipy import stats  # only used for the inverse-normal and beta quantiles


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


if __name__ == "__main__":
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
```

Running this, the four methods agree closely for 420/500 (all give roughly $[0.807, 0.870]$), which is reassuring — when $n$ is large and $p$ is mid-range, everything converges. The 49/50 case is where they diverge: Wilson gives roughly $[0.894, 0.997]$ while the naive Wald interval gives $[0.941, 1.019]$ — an upper limit *above 1.0*, which is nonsense. That is the bug Wilson exists to fix.

!!! tip "Cluster-bootstrap when items aren't independent"

    Many benchmarks violate item independence: a coding benchmark may have 5 variations of the same underlying problem, an agentic eval may have multiple turns from one task, or a long-document QA set may ask 10 questions about the same passage. Resampling individual items then *underestimates* variance because correlated items don't carry independent information. The fix is the **cluster (block) bootstrap**: resample whole *groups* (problems, documents, tasks) with replacement, keeping each group intact. The same code works — just resample group indices instead of item indices, then flatten. Ignoring clustering is one of the most common ways eval CIs come out too narrow.

### Bootstrapping an Elo / Bradley–Terry rating

Pairwise-comparison leaderboards (LMArena-style human votes, or pairwise LLM-judge tournaments) summarize models with an **Elo** or, equivalently, a **Bradley–Terry** rating. The Bradley–Terry model says the probability that model $i$ beats model $j$ is a logistic function of their rating difference:

$$
P(i \succ j) = \frac{1}{1 + 10^{-(R_i - R_j)/400}} = \sigma\!\left(\frac{\ln 10}{400}(R_i - R_j)\right).
$$

Ratings $R_i$ are fit by maximum likelihood over the observed battle outcomes. The crucial point for this chapter: **a single Elo number is also a point estimate with sampling error**, and that error is often large — tens of points — when a model has played few battles. You estimate it by *bootstrapping over battles*: resample the table of (model A, model B, winner) rows with replacement, refit the ratings, and repeat. The spread of refit ratings is the CI.

```python
import numpy as np


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
    for name, (pt, lo, hi) in bootstrap_elo(battles, models).items():
        print(f"{name}: {pt}  95% CI [{lo}, {hi}]  width={hi - lo}")
```

With only 300 battles split across three pairings, the per-model Elo CIs span roughly $\pm 40$–$60$ points. That is why real leaderboards report rating *intervals* and explicitly group models into tiers: if A's interval and B's interval overlap, the leaderboard rank between them is not statistically meaningful, no matter what the ordering of the point estimates says.

---

## The Question You Actually Care About: Is A Better Than B?

A confidence interval on a single model is necessary but rarely the real question. The real question is comparative: **does my new checkpoint beat the baseline?** And here a subtle but decisive trick is available: **pairing**. You should evaluate both models on the *same* items, then test the *per-item difference*. This cancels the enormous item-to-item difficulty variance and gives a vastly more powerful test than comparing two independent CIs.

### Why pairing crushes the variance

Suppose item difficulty varies wildly — some questions are trivial (both models pass), some are impossible (both fail), and only a minority are *discriminating* (one passes, one fails). For independent (unpaired) comparison, the variance of $\hat p_A - \hat p_B$ is $\text{Var}(\hat p_A) + \text{Var}(\hat p_B)$. For the paired difference $d_i = \text{score}_A(i) - \text{score}_B(i)$, the variance is

$$
\text{Var}(\bar d) = \frac{\text{Var}(d_i)}{n} = \frac{\sigma_A^2 + \sigma_B^2 - 2\,\text{Cov}(A,B)}{n}.
$$

Because two competent models tend to pass and fail the *same* easy/hard items, $\text{Cov}(A,B)$ is large and positive, so the paired variance is much smaller. In practice pairing can be equivalent to a 4–10× larger unpaired test set. **Always pair when you can.**

### Three paired tests and when to use each

| Test | Data type | Null hypothesis | Use when |
|------|-----------|------------------|----------|
| **McNemar's test** | binary pass/fail | discordant pairs equally likely either way | accuracy / pass-rate on the same items |
| **Paired $t$-test** | continuous (judge scores, F1, log-prob) | mean difference $= 0$ | per-item numeric scores, roughly symmetric diffs |
| **Wilcoxon signed-rank** | continuous, non-normal | median difference $= 0$ | skewed/heavy-tailed per-item diffs |

**McNemar's test** is the right tool for the most common case — both models scored pass/fail on the same items. It looks *only* at the discordant pairs: items where exactly one model was right. Let $b$ = count where A right, B wrong, and $c$ = count where A wrong, B right. Concordant pairs (both right, both wrong) carry no information about which is better and are *discarded*. Under the null "the two models are equally good," each discordant item is a coin flip, so $b \sim \text{Binomial}(b+c, 0.5)$. The test statistic is

$$
\chi^2 = \frac{(|b - c| - 1)^2}{b + c} \sim \chi^2_1 \quad\text{(with continuity correction)},
$$

and for small $b+c$ you should use the **exact binomial** test instead of the chi-square approximation.

{{fig:statrig-pairing-discordant-cells}}

### Paired tests — from scratch

```python
import numpy as np
from scipy import stats


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


if __name__ == "__main__":
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
```

!!! example "Worked example: pairing rescues a borderline result"

    A team compares fine-tune **A** against baseline **B** on a 500-item eval. Headline accuracies: A = 82.6 %, B = 79.0 % — a 3.6-point gap.

    **Naive unpaired check.** Each model's Wald SE is about $\sqrt{0.8\times0.2/500}\approx 0.0179$. The SE of the *difference* (treating them as independent) is $\sqrt{0.0179^2 + 0.0179^2}\approx 0.0253$, so the unpaired 95 % CI on the gap is $0.036 \pm 0.050 = [-0.014,\ 0.086]$. **It crosses zero** — by this analysis you cannot claim A is better.

    **Paired analysis on the same 500 items.** Cross-tabulating: 372 items both got right, 38 both got wrong, $b = 51$ items where only A was right, $c = 33$ where only B was right. There are only $b+c = 84$ discordant items. McNemar's chi-square is

    $$
    \chi^2 = \frac{(|51-33|-1)^2}{84} = \frac{17^2}{84} = \frac{289}{84} \approx 3.44,
    $$

    giving $p \approx 0.064$. The paired bootstrap CI on the accuracy difference is roughly $[+0.004,\ +0.069]$ — it *barely* excludes zero. Pairing shrank the interval from $\pm 0.050$ to about $\pm 0.033$ because the two models agreed on 410 of 500 items, and those agreements carried no comparative signal but did inflate the unpaired variance. The honest verdict: **suggestive ($p\approx 0.06$) but not conclusive at $\alpha=0.05$** — you would want more items or more discriminating items before shipping a "beats baseline" claim. This is exactly the kind of nuance a single accuracy number hides.

### Multiple comparisons: the leaderboard trap

If you test 20 candidate checkpoints against a baseline at $\alpha = 0.05$, you expect *one false positive on average even if none is truly better* — because $0.05 \times 20 = 1$. Leaderboards with hundreds of models are multiple-comparison machines: some model will look "significantly best" by chance. Two standard corrections:

- **Bonferroni:** test each comparison at $\alpha/m$ for $m$ comparisons. Simple, conservative, controls the family-wise error rate (probability of *any* false positive).
- **Benjamini–Hochberg (FDR):** sort the $m$ p-values, find the largest $k$ with $p_{(k)} \le \frac{k}{m}\alpha$, reject all below it. Controls the *false discovery rate* (expected fraction of false positives among rejections) — more powerful and usually the right choice when you are screening many candidates.

```python
import numpy as np

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
```

---

## Variance Decomposition: Where Does Your Noise Live?

A benchmark score is buffeted by several noise sources at once. Knowing which one dominates tells you *what to fix*: more items, more seeds, prompt averaging, or a better judge. The framework is the **variance components** model from classical experimental design (and, in psychometrics, **generalizability theory**).

### A linear variance model

Model the score of model $m$ on item $i$, under prompt template $t$, seed $s$, judged by judge $j$ as

$$
y_{mitsj} = \mu + \alpha_m + \beta_i + \gamma_t + \delta_s + \zeta_j + \varepsilon_{mitsj},
$$

where each term is a zero-mean random effect with its own variance: $\sigma^2_{\text{item}}$ (item difficulty), $\sigma^2_{\text{prompt}}$ (prompt-template sensitivity), $\sigma^2_{\text{seed}}$ (generation noise), $\sigma^2_{\text{judge}}$ (judge disagreement), and residual $\sigma^2_{\varepsilon}$. The variance of your *reported mean* over $n_i$ items, $n_t$ prompts, $n_s$ seeds, and $n_j$ judges is approximately

$$
\text{Var}(\bar y) \approx \frac{\sigma^2_{\text{item}}}{n_i} + \frac{\sigma^2_{\text{prompt}}}{n_t} + \frac{\sigma^2_{\text{seed}}}{n_i n_s} + \frac{\sigma^2_{\text{judge}}}{n_j} + \frac{\sigma^2_{\varepsilon}}{n_i n_s n_j}.
$$

The practical lesson is in the denominators. If $\sigma^2_{\text{prompt}}$ is large and you used **one** prompt template, that term is divided by $n_t = 1$ — no amount of extra items reduces it. This is why "we changed the system prompt and the score moved 4 points" is so common: prompt variance is frequently the *dominant* term, and it is invisible to an item-bootstrap CI. The robustness chapter [Red-Teaming, Safety & Robustness Evaluation](../11-evaluation/05-redteaming-safety-eval.html) treats prompt sensitivity as a first-class failure mode.

### Estimating the components

A clean way to estimate components is a crossed design: run the eval over a grid of $n_i$ items $\times$ $n_t$ prompts $\times$ $n_s$ seeds and fit the variances by ANOVA or REML. Here is a compact estimator for the two axes practitioners most often neglect — prompt and seed — using nested means.

```python
import numpy as np


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
    rng = np.random.default_rng(0)
    n_i, n_t, n_k = 200, 4, 3
    item_eff   = rng.normal(0, 1.2, n_i)      # big item-difficulty spread
    prompt_eff = rng.normal(0, 0.5, n_t)      # nontrivial prompt sensitivity
    logits = (0.4 + item_eff[:, None, None] + prompt_eff[None, :, None]
              + rng.normal(0, 0.3, (n_i, n_t, n_k)))
    scores = (logits > 0).astype(float)       # pass/fail
    for k, v in variance_components(scores).items():
        print(f"{k:>22}: {v:.4f}")
```

The output makes the dependence concrete: with strong item effects and four prompts, the prompt term may rival the item term in the SE budget even though each individual prompt contributes only modestly. If you had reported a single prompt, your CI would have been a fiction.

!!! tip "Report on at least 3 prompts and 3 seeds"

    A cheap, defensible protocol: evaluate every model on $\ge 3$ paraphrased prompt templates and $\ge 3$ seeds, report the mean and a CI computed by bootstrapping over the (item, prompt, seed) tuples (use a cluster bootstrap that resamples items as blocks). This folds prompt and seed variance into the interval automatically and immunizes you against the "we picked the lucky prompt" critique. The cost is a 9× compute increase — often worth it for a headline claim, skippable for a quick dev-loop check.

---

## Power Analysis: Sizing the Test Set *Before* You Run It

Confidence intervals are retrospective — they describe the experiment you already ran. **Power analysis** is prospective: given the smallest effect you would care about, how many items do you need so that, *if the effect is real*, your test will actually detect it? Running an underpowered eval is the worst of both worlds: you spend the compute and still cannot conclude anything.

### The four interlocking quantities

Power analysis ties together four numbers; fix any three and the fourth is determined:

- **$\alpha$** — false-positive rate (typically 0.05). $z_{1-\alpha/2} = 1.96$ for a two-sided test.
- **Power $1-\beta$** — probability of detecting a true effect (typically 0.80; sometimes 0.90). $z_{1-\beta} = 0.84$ for 80 %.
- **Effect size $\Delta$** — the minimum difference worth detecting (the *minimum detectable effect*, MDE).
- **$n$** — the number of items.

For comparing two proportions on **paired** data, the McNemar-based sample size depends on the discordant-pair rate. A convenient and widely-used approximation for the number of items needed, given expected discordant proportion $p_d = p_b + p_c$ and effect $p_b - p_c$, is

$$
n \approx \frac{\left(z_{1-\alpha/2}\sqrt{p_d} + z_{1-\beta}\sqrt{p_d - (p_b - p_c)^2}\right)^2}{(p_b - p_c)^2}.
$$

For the simpler **unpaired** two-proportion case (or a back-of-envelope), the classic formula with pooled proportion $\bar p$ is

$$
n_{\text{per arm}} \approx \frac{\left(z_{1-\alpha/2} + z_{1-\beta}\right)^2 \cdot 2\,\bar p (1-\bar p)}{\Delta^2}.
$$

### Power analysis — from scratch, including simulation

Closed forms rely on normal approximations; for paired binary outcomes with small discordant counts, a **simulation-based power analysis** is more trustworthy and barely more code. You assume a data-generating process, simulate the experiment thousands of times, and count how often the test rejects.

```python
import numpy as np
from scipy import stats


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
    p_neither = 1 - p_both - p_b - p_c
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


if __name__ == "__main__":
    # We want to detect a true 3-point accuracy edge for A.
    # Suppose models agree 80% of the time; of the 20% discordant items,
    # A wins on 11.5% and B on 8.5% -> a 3-point gap.
    p_b, p_c = 0.115, 0.085
    n_closed = n_for_mcnemar(p_b, p_c)
    print(f"closed-form n for 80% power: {n_closed}")
    for n in [n_closed, 2 * n_closed]:
        pw = simulate_power(p_both=0.80, p_b=p_b, p_c=p_c, n=n)
        print(f"  n={n:5d} -> simulated power {pw:.2f}")
    # Compare to the UNPAIRED requirement for the same 3-point gap at p~0.8.
    print("unpaired n PER ARM:", n_for_unpaired_proportions(0.83, 0.80))
```

!!! example "How big a test set to catch a 1-point gap?"

    You want to reliably detect a **1-point** accuracy difference (say 80.0 % vs 81.0 %) at $\alpha=0.05$, power 0.80. Using the unpaired formula with $\bar p = 0.805$ and $\Delta=0.01$:

    $$
    n \approx \frac{(1.96 + 0.84)^2 \cdot 2 \times 0.805 \times 0.195}{0.01^2}
    = \frac{7.84 \times 0.3140}{0.0001} \approx 24{,}600 \text{ per arm}.
    $$

    Roughly **25,000 items per model** — which is why no 500-item benchmark can adjudicate a one-point claim, and why frontier leaderboards still cannot cleanly separate the top few models. Now switch to a *paired* design: if the models agree on 85 % of items, the effective discordant sample is far richer, and `n_for_mcnemar` returns on the order of a **few thousand** items for the same 1-point effect — an order-of-magnitude saving from pairing alone. The takeaway: choose your MDE honestly, then either accept that small gaps need huge test sets, or pair aggressively, or stop reporting differences you cannot resolve.

---

## Squeezing More Signal Per Item: IRT & Adaptive Evaluation

Power analysis says "buy more items." But items are not equally informative — a question every model gets right (or every model gets wrong) tells you *nothing* about relative ability. **Item Response Theory** (IRT), the measurement framework behind standardized tests like the SAT and GRE, formalizes this and lets you (a) build a better ability estimate from the same items and (b) *select* the most informative items to ask, cutting eval cost dramatically.

### The 2-parameter logistic model

In the **2PL** IRT model, the probability that a test-taker (here, a model) with latent ability $\theta$ answers item $i$ correctly is

$$
P(\text{correct} \mid \theta, a_i, b_i) = \sigma\big(a_i(\theta - b_i)\big) = \frac{1}{1 + e^{-a_i(\theta - b_i)}},
$$

where $b_i$ is the item's **difficulty** (the ability at which a model has a 50 % chance) and $a_i$ is its **discrimination** (how sharply the pass probability rises with ability). High-$a_i$ items are the ones that cleanly separate strong from weak models; low-$a_i$ items (everyone guesses, or a typo makes the answer ambiguous) are noise. Note the structural kinship with Bradley–Terry — both are logistic latent-trait models — and with the logistic regression in [Machine Learning Fundamentals](../01-foundations/05-ml-fundamentals.html).

The **Fisher information** an item contributes about ability $\theta$ is

$$
I_i(\theta) = a_i^2\, P_i(\theta)\big(1 - P_i(\theta)\big),
$$

maximized when $P_i(\theta) = 0.5$, i.e. when the item's difficulty matches the model's ability. This is the engine of **adaptive testing**: ask each model items near its own ability frontier, where each answer is maximally informative, and you reach a target precision with a fraction of the items.

### Fitting a tiny IRT model and computing item information

```python
import numpy as np
from scipy.optimize import minimize


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


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_m, n_i = 30, 120
    true_theta = rng.normal(0, 1, n_m)
    true_a = np.abs(rng.normal(1.0, 0.4, n_i)) + 0.2
    true_b = rng.normal(0, 1, n_i)
    P = 1 / (1 + np.exp(-true_a * (true_theta[:, None] - true_b[None, :])))
    R = (rng.random((n_m, n_i)) < P).astype(float)

    theta, a, b = fit_2pl(R)
    # Recovery check: correlation between true and estimated ability.
    print("ability recovery r =", np.corrcoef(true_theta, theta)[0, 1].round(3))
    # The 10 most discriminating items -- these are worth their weight.
    top = np.argsort(a)[::-1][:10]
    print("most informative items (idx, a, b):",
          [(int(i), round(a[i], 2), round(b[i], 2)) for i in top])
```

### Adaptive evaluation: same precision, a fraction of the items

Once items are calibrated (their $a_i, b_i$ are known from a reference panel of models), evaluating a *new* model adaptively is cheap: ask an item near the current ability estimate, update the estimate, repeat until the standard error on $\hat\theta$ drops below a threshold. Because each item is chosen to maximize information at the current $\hat\theta$, this **computerized adaptive testing** (CAT) loop typically reaches the precision of a fixed 100-item test in 20–40 items. This is the statistical core behind "efficient benchmarking" work (e.g. tinyBenchmarks, Anchor Points, and IRT-based leaderboard analyses): you do not need all 14,000 MMLU items if you have calibrated the item bank — a carefully chosen few hundred recover the full-benchmark ranking within its own confidence band.

```python
import numpy as np
from scipy.optimize import minimize_scalar


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
```

!!! warning "Adaptive evaluation needs a calibrated, uncontaminated bank"

    IRT-based shortcuts are only valid if the item parameters were estimated on a representative panel of models and the items are *not* in the model's training data. A contaminated item (memorized from training) looks artificially easy and corrupts the calibration — see the contamination discussion in [The Evaluation Problem & Benchmark Landscape](../11-evaluation/01-eval-landscape.html). Re-calibrate when the model population shifts (e.g., a new generation of reasoning models), because an item that was discriminating for last year's models may be trivially solved by this year's.

---

## Reading Leaderboards Like a Statistician

Bring it together with the skill this chapter exists to instill. When you look at a leaderboard:

1. **Find the error bars.** If a leaderboard reports no intervals, mentally attach $\pm 1/\sqrt n$ to every number ($n$ = items). LMArena-style boards *do* publish rating CIs and group models into tiers — use them.
2. **Overlapping CIs ⇒ no demonstrated gap.** If model A's 95 % interval overlaps model B's, you cannot claim A ranks above B from this data, regardless of point-estimate order. (Strictly, *non*-overlapping intervals imply a significant difference, but *overlapping* intervals do **not** imply non-significance — for a rigorous yes/no you still want the paired test on the difference. Overlap is a quick screen, the paired test is the verdict.)
3. **Ask how many evaluations.** Two models near the top of an Elo board may have played thousands of battles each (tight CIs) while a new entrant has played 200 (a $\pm 50$-point interval). Rank is meaningless until the new entrant has enough battles.
4. **Suspect the prompt.** A 3-point swing between two papers' numbers for the *same* model is almost always prompt/harness variance, not a real capability change. Demand the prompt template and the harness version.
5. **Check for multiple comparisons.** "Best on 9 of 12 benchmarks" with 12 noisy benchmarks is roughly what chance produces. Look for a pre-registered headline metric, not a victory lap across cherry-picked subsets.

{{fig:statrig-ci-overlap-verdict}}

!!! interview "Interview Corner"

    **Q:** Your team's new model scores 71.4 % on a 1,000-item internal benchmark; the previous model scored 70.1 %. Leadership wants to ship the new one as "better." Walk me through how you'd decide whether that 1.3-point gain is real, and what you'd report.

    **A:** First, recognize both numbers are estimates with sampling error. The single-model 95 % margin at $n=1{,}000$, $p\approx0.7$ is about $1.96\sqrt{0.7\times0.3/1000}\approx 2.8$ points — so the two *independent* intervals overlap heavily and a naive comparison is inconclusive. But I wouldn't compare independent intervals; I'd **pair**. Both models were run on the same 1,000 items, so I'd build the per-item correctness vectors and run **McNemar's test** on the discordant pairs, plus a **paired bootstrap CI on the accuracy difference** (resample item indices once, apply to both models). If the paired CI excludes zero and McNemar's $p < 0.05$, the gain is statistically credible.

    I'd also check three things that frequently overturn such a result: (1) **prompt and seed variance** — I'd re-run both models on $\ge 3$ prompt templates and $\ge 3$ seeds, because a 1.3-point gap can vanish under a different system prompt; (2) **clustering** — if the benchmark has grouped items (e.g. multiple questions per document), I'd use a cluster bootstrap so the CI isn't falsely narrow; (3) **contamination/regressions** — even if the average improved, I'd inspect the items the *new* model newly fails (the McNemar $c$ cell), since a net gain can hide a meaningful regression on an important slice. What I'd report: the paired difference with its 95 % CI, the McNemar p-value, the per-prompt spread, and an explicit minimum-detectable-effect statement ("at this $n$ we can resolve gaps down to ~X points"). If the CI is, say, $[+0.1, +2.5]$ points, I'd say "likely a small real improvement, but within the range where prompt choice matters — ship behind an online A/B test ([Online Evaluation: A/B Testing, Canaries & Guardrail Metrics](../12-production-mlops/07-online-eval-ab-testing.html)) rather than declaring a decisive win."

!!! note "Frequentist CIs vs. Bayesian credible intervals"

    Everything above is frequentist: a 95 % CI means "95 % of intervals built this way would contain the true value." A **Bayesian credible interval** instead says "given a prior and the data, there's a 95 % posterior probability the value lies in this range" — often what people *think* a CI means. For a binomial accuracy with a uniform $\text{Beta}(1,1)$ prior, the posterior is $\text{Beta}(k+1, n-k+1)$, and its 2.5/97.5 percentiles give a credible interval that closely matches the Wilson interval for moderate $n$. Bayesian framing shines for leaderboards: you can report $P(\text{model A} > \text{model B})$ directly from the posterior over abilities — a more decision-relevant statement than a p-value. The mechanics live in [Probability, Statistics & Information Theory](../01-foundations/02-probability-information.html).

---

## Key Takeaways

!!! key "Key Takeaways"

    - **Every eval number is a random variable.** Report a confidence interval, never a bare point estimate. The 95 % margin near $p=0.5$ is roughly $1/\sqrt n$ — so a 500-item benchmark cannot resolve gaps smaller than about 4 points, and a 1-point claim needs tens of thousands of items.
    - **Use the right interval.** Wilson or Clopper–Pearson for binomial accuracy (the Wald interval breaks near 0/1); **bootstrap** (prefer BCa) for any non-trivial statistic — Elo, F1, judge means, percentiles. Use a **cluster bootstrap** when items are grouped, or your CIs will be too narrow.
    - **Pair whenever possible.** Evaluate A and B on the *same* items and test the per-item difference (McNemar for pass/fail, paired $t$ or Wilcoxon for scores). Pairing cancels item-difficulty variance and can be worth a 4–10× larger unpaired test set.
    - **Know where your noise lives.** Decompose variance across items, prompts, seeds, and judges. Prompt-template variance is often dominant and is *invisible* to an item-only CI — run $\ge 3$ prompts and $\ge 3$ seeds before any headline claim.
    - **Size the test set before running it.** Power analysis turns "what's the smallest gap I care about?" into "how many items do I need?" Simulate the paired test when the closed-form normal approximation is shaky.
    - **Correct for multiple comparisons.** Testing many checkpoints/benchmarks manufactures false positives; use Benjamini–Hochberg (FDR) or Bonferroni, and prefer a pre-registered headline metric.
    - **IRT buys efficiency.** Items differ in discrimination and information; a calibrated item bank plus adaptive selection reaches fixed-test precision in a fraction of the items — provided the bank is uncontaminated and re-calibrated as models evolve.
    - **Overlapping CIs mean "no demonstrated gap."** On a leaderboard, rank order without separation is noise. Demand error bars, battle counts, and the prompt/harness version before believing any "new state of the art."

!!! sota "State of the Art & Resources (2026)"
    Statistical rigor in LLM evaluation has moved from academic concern to mainstream practice: leaderboards now publish rating confidence intervals, ICML 2025 accepted a spotlight position paper on CLT failures at small sample sizes, and IRT-based methods that cut evaluation cost 10–100× have been published and packaged. The field consensus is that bare point estimates are no longer acceptable for any published claim.

    **Foundational work**

    - [Brown, Cai & DasGupta, *Interval Estimation for a Binomial Proportion* (2001)](https://projecteuclid.org/journals/statistical-science/volume-16/issue-2/Interval-Estimation-for-a-Binomial-Proportion/10.1214/ss/1009213286.full) — the canonical paper proving Wald intervals are defective near 0/1 and recommending Wilson/Jeffreys alternatives.
    - [Dror et al., *The Hitchhiker's Guide to Testing Statistical Significance in NLP* (ACL 2018)](https://aclanthology.org/P18-1128/) — practical protocol for choosing McNemar, Wilcoxon, or permutation tests across NLP eval setups.
    - [Chiang et al., *Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference* (2024)](https://arxiv.org/abs/2403.04132) — the pairwise human-preference leaderboard that popularized Bradley–Terry ratings with bootstrap confidence intervals and model tier grouping.

    **Recent advances (2023–2026)**

    - [Polo et al., *tinyBenchmarks: Evaluating LLMs with Fewer Examples* (ICML 2024)](https://arxiv.org/abs/2402.14992) — IRT-based method that estimates full-benchmark performance from ~100 curated items; shows MMLU rankings recoverable from 100 vs 14 000 items.
    - [Bowyer, Aitchison & Ivanova, *Don't Use the CLT in LLM Evals With Fewer Than a Few Hundred Datapoints* (ICML 2025 spotlight)](https://arxiv.org/abs/2503.01747) — demonstrates CLT-based CIs dramatically underestimate uncertainty on specialized small benchmarks; proposes frequentist and Bayesian alternatives.
    - [Wu, Nair & Candès, *Efficient Evaluation of LLM Performance with Statistical Guarantees* (2025)](https://arxiv.org/abs/2601.20251) — Factorized Active Querying (FAQ) achieves up to 5× effective sample-size gains over baselines while preserving valid frequentist coverage.
    - [Ameli et al., *A Statistical Framework for Ranking LLM-Based Chatbots* (ICLR 2025)](https://arxiv.org/abs/2412.18407) — extends Bradley–Terry with a factored tie model and covariance structure for better-calibrated leaderboard intervals.

    **Open-source & tools**

    - [felipemaiapolo/tinyBenchmarks](https://github.com/felipemaiapolo/tinyBenchmarks) — Python package for IRT/p-IRT/gp-IRT estimation on MMLU, GSM8K, and Open LLM Leaderboard subsets.
    - [Kaleidophon/deep-significance](https://github.com/Kaleidophon/deep-significance) — library implementing Almost Stochastic Order, bootstrap, and permutation tests for comparing deep-learning models, with power-analysis utilities.
    - [rtmdrr/testSignificanceNLP](https://github.com/rtmdrr/testSignificanceNLP) — companion code for Dror et al. 2018; runs Shapiro–Wilk, t-test, Wilcoxon, and McNemar from the command line on any score files.

    **Go deeper**

    - [Cameron R. Wolfe, *Applying Statistics to LLM Evaluations* (2024)](https://cameronrwolfe.substack.com/p/stats-llm-evals) — practitioner-focused deep-dive covering CLT, clustered SEs, paired comparisons, and power analysis with worked examples.

## Further reading

- Bradley, R. A. & Terry, M. E., *Rank Analysis of Incomplete Block Designs* (1952) — the pairwise-comparison model underlying Elo and modern LLM arenas.
- Efron, B. & Tibshirani, R. J., *An Introduction to the Bootstrap* (1993) — the definitive treatment of bootstrap CIs, including BCa.
- Brown, Cai & DasGupta, *Interval Estimation for a Binomial Proportion* (Statistical Science, 2001) — why Wald is bad and Wilson/Agresti–Coull are good.
- Dietterich, T. G., *Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms* (1998) — paired tests, McNemar, and the perils of naive comparison in ML.
- Benjamini, Y. & Hochberg, Y., *Controlling the False Discovery Rate* (1995) — the standard multiple-comparison correction for screening many candidates.
- Lord, F. M., *Applications of Item Response Theory to Practical Testing Problems* (1980) — the foundational IRT and adaptive-testing reference.
- Polo et al., *tinyBenchmarks: Evaluating LLMs with Fewer Examples*, and Vivek et al., *Anchor Points* — IRT-style efficient LLM evaluation.
- Miller, E., *Adding Error Bars to Evals* — a practitioner-facing call to report confidence intervals on LLM benchmarks.
- Chiang et al., *Chatbot Arena* — the human-preference leaderboard whose Bradley–Terry ratings and confidence intervals popularized statistical rigor in LLM ranking.
