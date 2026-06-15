# 12.7 Online Evaluation: A/B Testing, Canaries & Guardrail Metrics

Offline benchmarks are necessary but not sufficient. A model that tops your held-out evaluation set may still lose you users, inflate costs, or silently regress on edge cases that only emerge at production scale. The gap between an offline MMLU score and real user satisfaction is not a failure of offline evaluation — it is a structural property: users are not IID draws from a benchmark corpus, and the distribution of inputs, intents, and behaviors that matter commercially is only observable in production.

This chapter closes the loop. We cover how to design statistically sound A/B and interleaving experiments for LLM features, which guardrail metrics to instrument and how to prevent them from being gamed, how to make decisions faster with CUPED variance reduction and sequential testing, how to roll out safely via shadow deployments and canaries, and how to keep live judges scoring sampled traffic so you stay honest about quality even after launch. We also examine the systematic biases — novelty effects, feedback loops, position bias — that corrupt online signals if left unaddressed.

Related chapters that provide useful context: [The Evaluation Problem & Benchmark Landscape](../11-evaluation/01-eval-landscape.html), [LLM-as-a-Judge & Automated Evaluation](../11-evaluation/02-llm-as-judge.html), [Statistical Rigor in Evaluation: Confidence Intervals & Significance](../11-evaluation/06-statistical-rigor-eval.html), [Data Flywheels & Continuous Improvement](../12-production-mlops/05-data-flywheel.html), and [Observability, Logging & LLMOps](../12-production-mlops/02-observability-llmops.html).

---

## The Offline–Online Gap and Why It Exists

Before designing experiments, we need to understand why offline metrics so often disagree with online outcomes.

**Distribution shift.** A benchmark is a frozen snapshot. Real users evolve: they learn to phrase queries that exploit a new model's strengths, or they hit topics that weren't in your eval set. A model fine-tuned on last quarter's support tickets may degrade subtly on the new product surface launched this quarter.

**Task proxies are imperfect.** ROUGE-L measures n-gram overlap; users care about whether the answer resolved their problem. Perplexity on a held-out set predicts coherence but not helpfulness. Even carefully designed human evals may not reflect actual user intent distributions.

**Unobservable label noise.** Offline evals often rely on human annotations, which have inter-annotator disagreement, recency bias, and annotator fatigue. The "ground truth" is fuzzier than it appears.

**Survivor bias in retrieval.** In RAG systems, offline retrieval evals are run on queries that have known relevant documents. Live traffic contains queries for which no document exists — a silent failure mode invisible in offline settings.

The practical implication: offline improvements should be treated as *evidence* for online experiments, not as proof. The question is never "did our offline eval improve?" but "does our offline eval predict online outcomes?"

A useful diagnostic is the **offline–online correlation coefficient**: track, across many launches, whether a 1% offline improvement predicts a positive online signal. If this correlation is weak (below ~0.5), your offline eval is not a reliable proxy and you are flying blind between offline and live.

---

## Guardrail Metrics: What to Measure and Why

Before running experiments, you must decide what you are measuring. For LLM products, metrics fall into three tiers.

### Primary metrics (goal metrics)

These are the business outcomes you ultimately care about:

- **Resolution rate**: the fraction of user sessions in which the user's problem was solved without escalation, re-query, or abandonment. For a customer support bot, this is the north star.
- **Task success rate**: for agent or coding assistant products, automated verification that the task was completed (tests pass, form submitted, booking confirmed).
- **Session engagement**: downstream engagement signals (clicks, purchases, return visits) in contexts where helpfulness precedes a downstream action.

### Guardrail metrics (do-no-harm metrics)

These must not regress below a defined threshold, even if the primary metric improves. Violating a guardrail halts a rollout regardless of primary metric performance.

| Guardrail metric | What it catches |
|---|---|
| Deflection rate (support context) | Model is telling users to "contact a human" too aggressively |
| Hallucination rate (sampled + judged) | Model generating factually incorrect content at elevated rate |
| Toxicity / safety policy violations | Guardrail model flags on sampled responses |
| Latency P99 (time-to-first-token) | Slower model degrading user experience |
| Cost per session | More expensive model offsetting quality gains |
| Regeneration rate | Users clicking "regenerate" — a soft dislike signal |
| Thumb-down rate | Explicit negative user feedback |

The regeneration rate deserves special attention: it is a behavioral signal that does not require users to explicitly rate anything, making it much higher-coverage than explicit thumbs-up/down while still directionally reliable.

### Sensitivity metrics (canary signals)

These are leading indicators with lower latency than the primary metric. Because resolution rate may take days or weeks of data to reach significance (a user may not return to signal resolution), you need sensitive proxies that respond within hours:

- **Thumbs-up/thumbs-down ratio**: available immediately after each session.
- **Session depth**: number of follow-up turns (more turns often indicates confusion or failure).
- **Copy rate**: users copying model output — a strong positive behavioral signal.
- **Share rate** (where applicable): users sharing the response externally.

!!! warning "Common pitfall: metric gaming"
    Once an engineer knows a metric is being measured, the system — and the feature — may be inadvertently optimized against it. A model trained to maximize thumbs-up rate may learn to be sycophantic (tell users what they want to hear) rather than accurate. Regeneration rate can drop if you simply hide the regenerate button. Resolution rate can be gamed by filtering out unresolved sessions from the denominator. The defense: maintain a diverse metric portfolio; if primary and guardrail metrics diverge in unexpected directions, investigate before concluding success.

---

## Designing A/B Experiments for LLM Features

### Randomization unit

For LLM features, randomization is almost always at the **user-session level** (randomly assign each user to a variant, then serve that variant consistently throughout the experiment). Randomizing at the request level within a user session violates the stable unit treatment value assumption (SUTVA): a model swap mid-conversation changes the distribution of subsequent turns, introducing contamination.

For products without user identity (anonymous API access), randomize on a stable session token or a hashed combination of IP + user agent. The key property is that the same user sees the same variant across multiple requests in a session.

### Control and treatment specification

An A/B experiment compares a **control** (the current production model) against one or more **treatments** (candidate models or features). Common structures:


{{fig:online-eval-ab-traffic-split}}


For a new product with no established baseline, use an **A/A test** first: split traffic into two identical control arms and verify that your metrics show no significant difference. An A/A test failing is a red flag for systematic logging errors, selection bias, or a broken randomization layer.

### Sample size and statistical power

For a two-sided test at significance level $\alpha$ and power $1 - \beta$, the required number of users per arm is:

$$
n = \frac{(z_{\alpha/2} + z_\beta)^2 \cdot 2\sigma^2}{\delta^2}
$$

where $\delta$ is the minimum detectable effect (MDE) you care about, $\sigma^2$ is the within-arm variance of the metric, and $z_{\alpha/2}$, $z_\beta$ are the corresponding normal quantiles ($z_{0.025} \approx 1.96$, $z_{0.2} \approx 0.84$ for 80% power).

!!! example "Sample size worked example"
    Suppose your thumbs-up rate is currently 0.40 with standard deviation 0.49 (Bernoulli), and you want to detect a 2-percentage-point improvement (MDE = 0.02) with 80% power at $\alpha = 0.05$.

    $$
    n = \frac{(1.96 + 0.84)^2 \cdot 2 \times 0.49^2}{0.02^2}
    = \frac{7.84 \times 0.4802}{0.0004}
    = \frac{3.765}{0.0004} \approx 9{,}413 \text{ users per arm}
    $$

    At 100,000 active daily users split 50/50, you reach this in under 4 hours. At 1,000 active daily users, it takes roughly 19 days. This illustrates why low-traffic products need either a higher MDE (coarser test) or variance reduction techniques (see CUPED below).

### Running the test

```python
import hashlib
import math
from dataclasses import dataclass
from typing import Literal

# -------------------------------------------------------------------------
# Deterministic user assignment: same user → same variant every call.
# We use a SHA-256 hash of (experiment_id + user_id) for uniform assignment.
# -------------------------------------------------------------------------

@dataclass
class Experiment:
    experiment_id: str
    traffic_fraction: float = 1.0   # fraction of total traffic to enroll
    treatment_fraction: float = 0.5  # of enrolled users, fraction to treatment

def assign_variant(
    user_id: str,
    experiment: Experiment,
) -> Literal["control", "treatment", "holdout"]:
    """
    Returns the arm assignment for a given user in a given experiment.
    'holdout' means the user is not enrolled (outside traffic_fraction).
    """
    # Hash to [0, 1) using experiment_id as salt so different experiments
    # produce independent assignments for the same user.
    digest = hashlib.sha256(
        f"{experiment.experiment_id}:{user_id}".encode()
    ).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF  # uniform [0, 1)

    if bucket >= experiment.traffic_fraction:
        return "holdout"

    # Re-hash to assign within enrolled users (avoids correlation between
    # enrollment and treatment assignment).
    digest2 = hashlib.sha256(
        f"{experiment.experiment_id}:assign:{user_id}".encode()
    ).hexdigest()
    bucket2 = int(digest2[:8], 16) / 0xFFFFFFFF

    return "treatment" if bucket2 < experiment.treatment_fraction else "control"


# -------------------------------------------------------------------------
# Simple two-proportion z-test for a Bernoulli metric (e.g. thumbs-up rate).
# -------------------------------------------------------------------------

from scipy import stats
import numpy as np

def two_proportion_z_test(
    n_control: int,
    k_control: int,   # successes in control
    n_treatment: int,
    k_treatment: int,
) -> dict:
    """
    Returns p-value, confidence interval, and relative lift.
    Uses the pooled proportion for the null hypothesis.
    """
    p_c = k_control / n_control
    p_t = k_treatment / n_treatment
    p_pool = (k_control + k_treatment) / (n_control + n_treatment)

    se = math.sqrt(p_pool * (1 - p_pool) * (1/n_control + 1/n_treatment))
    z = (p_t - p_c) / se if se > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    # 95% CI on absolute difference using unpooled SE
    se_unpooled = math.sqrt(
        p_c * (1 - p_c) / n_control + p_t * (1 - p_t) / n_treatment
    )
    diff = p_t - p_c
    ci_lo = diff - 1.96 * se_unpooled
    ci_hi = diff + 1.96 * se_unpooled

    return {
        "p_control": p_c,
        "p_treatment": p_t,
        "absolute_lift": diff,
        "relative_lift": diff / p_c if p_c > 0 else float("nan"),
        "z_statistic": z,
        "p_value": p_value,
        "ci_95": (ci_lo, ci_hi),
        "significant": p_value < 0.05,
    }

# Example usage:
result = two_proportion_z_test(
    n_control=10_000, k_control=4_000,
    n_treatment=10_000, k_treatment=4_200,
)
# Expected: ~+5% relative lift, p ≈ 0.001 → significant
print(result)
```

---

## Interleaving: A Faster Alternative to A/B for Preference Signals

Traditional A/B tests require large samples to detect small effects because the between-user variance dominates. **Interleaving** (also called interleaved comparison) sidesteps this by showing outputs from both models *within the same user session*, then measuring which model's outputs users prefer.

### How interleaving works for LLM outputs

In search/recommendation, interleaving mixes ranked lists. For LLM chat products, one variant is: present two completions side-by-side (a "compare" UI) and record which the user acts on. A more subtle variant records which completion a user copies, continues the conversation from, or clicks "insert" on in a coding assistant.

For document editing or summarization, you can show two alternative completions and ask the user to select or edit one. The fraction of users who prefer treatment over control — the **win rate** — is the primary signal.

The statistical efficiency gain is substantial. Because each user sees both models, within-user variance is eliminated. Empirically, interleaving experiments have been reported to require on the order of 100x fewer user-sessions to detect the same effect size as a parallel A/B test for ranking systems (Radlinski & Craswell, "Optimized Interleaving for Online Retrieval Evaluation," WSDM 2013). The gain for LLM completions is product-dependent but typically a factor of 10–30x.

```python
from collections import defaultdict
from typing import NamedTuple

class InterleavingSession(NamedTuple):
    user_id: str
    control_response: str
    treatment_response: str
    # Which response did the user take a positive action on?
    # 'control', 'treatment', or 'none'
    preferred: str

def compute_interleaving_win_rate(
    sessions: list[InterleavingSession],
) -> dict:
    """
    Compute treatment win rate and a two-sided binomial test.
    Only sessions with a preference (not 'none') are counted.
    """
    decisive = [s for s in sessions if s.preferred != "none"]
    n = len(decisive)
    if n == 0:
        return {"win_rate": float("nan"), "n_decisive": 0}

    wins_treatment = sum(1 for s in decisive if s.preferred == "treatment")
    win_rate = wins_treatment / n

    # Under H0: win_rate = 0.5; use binomial test
    p_value = stats.binom_test(wins_treatment, n, p=0.5, alternative="two-sided")

    return {
        "win_rate": win_rate,
        "n_decisive": n,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }
```

Interleaving is best suited for *preference* signals (which response is better?) rather than *outcome* signals (did the user's problem get resolved?). For resolution rate and similar task-completion metrics, you still need an A/B test since both models cannot solve the same problem simultaneously in a meaningful way.

---

## CUPED and Sequential Testing for Low-Traffic Products

Many LLM products do not have the luxury of millions of daily users. A specialized enterprise copilot may have a few thousand active users. The standard A/B test becomes impractically slow — waiting weeks for significance means slow iteration velocity.

### CUPED: Controlled-experiment Using Pre-Experiment Data

CUPED (Deng et al., "Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data," WSDM 2013) exploits pre-experiment covariate information to reduce variance in the metric estimator.

The idea: compute the user's metric value in the period *before* the experiment. This pre-period value is strongly correlated with the in-experiment value (users who frequently gave thumbs-up before the experiment are likely to do so again). By subtracting the projection of the metric onto this covariate, we obtain a lower-variance estimator of the treatment effect.

The CUPED-adjusted estimator for user $i$ in arm $a$ is:

$$
\tilde{Y}_i = Y_i - \theta (X_i - \bar{X})
$$

where $Y_i$ is the in-experiment metric, $X_i$ is the pre-experiment covariate, $\bar{X}$ is the covariate mean (pooled across arms), and the optimal coefficient is:

$$
\theta^* = \frac{\text{Cov}(Y, X)}{\text{Var}(X)}
$$

The variance of the adjusted estimator is:

$$
\text{Var}(\tilde{Y}) = \text{Var}(Y)(1 - \rho^2)
$$

where $\rho$ is the Pearson correlation between $Y$ and $X$. If $\rho = 0.7$ (typical for behavioral metrics), variance drops by $1 - 0.49 = 51\%$, and required sample size drops by half.

```python
import numpy as np
from scipy import stats

def cuped_estimate(
    y_control: np.ndarray,
    y_treatment: np.ndarray,
    x_control: np.ndarray,    # pre-experiment covariate, control arm
    x_treatment: np.ndarray,  # pre-experiment covariate, treatment arm
) -> dict:
    """
    Compute CUPED-adjusted treatment effect and t-test p-value.

    y_*: in-experiment metric values per user
    x_*: pre-experiment metric values for the same users
    """
    # Pool covariate mean and compute theta using pooled data
    x_all = np.concatenate([x_control, x_treatment])
    y_all = np.concatenate([y_control, y_treatment])
    x_bar = x_all.mean()

    theta = np.cov(y_all, x_all, ddof=1)[0, 1] / np.var(x_all, ddof=1)

    # Adjust each user's metric
    y_control_adj = y_control - theta * (x_control - x_bar)
    y_treatment_adj = y_treatment - theta * (x_treatment - x_bar)

    # Two-sample t-test on adjusted values
    t_stat, p_value = stats.ttest_ind(y_treatment_adj, y_control_adj)

    delta = y_treatment_adj.mean() - y_control_adj.mean()
    se = np.sqrt(
        np.var(y_treatment_adj, ddof=1) / len(y_treatment_adj)
        + np.var(y_control_adj, ddof=1) / len(y_control_adj)
    )

    # Variance reduction achieved
    var_unadjusted = np.var(np.concatenate([y_control, y_treatment]), ddof=1)
    var_adjusted = np.var(np.concatenate([y_control_adj, y_treatment_adj]), ddof=1)
    rho_sq = 1 - var_adjusted / var_unadjusted

    return {
        "delta": delta,
        "p_value": p_value,
        "ci_95": (delta - 1.96 * se, delta + 1.96 * se),
        "theta": theta,
        "variance_reduction_fraction": rho_sq,
        "significant": p_value < 0.05,
    }

# Simulate: 500 users per arm, thumbs-up rate 0.40 control / 0.42 treatment
rng = np.random.default_rng(42)
n = 500
x_c = rng.binomial(1, 0.40, n).astype(float)  # pre-exp covariate
x_t = rng.binomial(1, 0.40, n).astype(float)
# In-experiment: add treatment effect + correlation with pre-exp
y_c = np.clip(x_c * 0.7 + rng.binomial(1, 0.12, n), 0, 1)
y_t = np.clip(x_t * 0.7 + rng.binomial(1, 0.14, n), 0, 1)

result = cuped_estimate(y_c, y_t, x_c, x_t)
print(f"Delta: {result['delta']:.4f}, p={result['p_value']:.4f}, "
      f"variance reduction: {result['variance_reduction_fraction']:.1%}")
```

### Sequential testing (always-valid p-values)

The temptation with slow experiments is to peek repeatedly at the p-value and stop early if $p < 0.05$. This inflates the false positive rate dramatically: peeking daily for 14 days at $\alpha = 0.05$ yields an actual false positive rate of roughly 20–25%.

**Sequential testing** (also called always-valid inference) provides a test statistic that can be evaluated at any time without type-I error inflation. The mSPRT (mixture Sequential Probability Ratio Test, Johari et al., 2017) and e-values framework guarantee that at any stopping time $\tau$:

$$
P\!\left(\text{reject } H_0 \text{ at any time } t \leq \tau \mid H_0\right) \leq \alpha
$$

For Bernoulli outcomes, a practical implementation uses the Robbins confidence sequence — an always-valid confidence interval that shrinks as data accumulates. Many experimentation platforms (Statsig, Optimizely) now implement this by default.

```python
def always_valid_ci(
    n: int,
    k: int,       # successes so far
    alpha: float = 0.05,
) -> tuple[float, float]:
    """
    Approximate always-valid (anytime) confidence interval for a
    Bernoulli proportion using the Howard et al. (2021) normal-mixture bound.

    This CI is valid at any sample size n ≥ 1 without correction for peeking.
    """
    p_hat = k / n
    # Confidence sequence width parameter (from Howard et al., 2021, eq. 1)
    # rho: mixing parameter, typical value 0.5 for balanced experiments
    rho = 0.5
    log_term = math.log(2 * math.log(n + 1) / alpha)
    width = math.sqrt(
        (p_hat * (1 - p_hat) / n) * (log_term + rho * math.log(log_term + 1))
    )
    return (max(0.0, p_hat - width), min(1.0, p_hat + width))

# Compare width at n=100 vs n=1000 for p≈0.4
for n, k in [(100, 40), (1000, 400), (5000, 2000)]:
    lo, hi = always_valid_ci(n, k)
    print(f"n={n:5d}: {lo:.3f} – {hi:.3f}  (width {hi-lo:.3f})")
```

---

## Shadow Deployments and Canary Rollouts

Statistical significance tells you the *effect* is real; it does not tell you the system is *safe* to deploy at scale. Canary and shadow deployments are the operational complement to experiment design.

### Shadow mode

In shadow mode, you run the candidate model on all production traffic but *discard its responses* — users never see them. Shadow mode lets you:

1. **Profile latency and cost** at realistic load without user impact. You discover that the new model has 40% higher P99 latency under batch pressure before any user is affected.
2. **Run offline judges on shadow outputs.** Sample shadow responses at, say, 1% and run your LLM-as-judge pipeline (see [LLM-as-a-Judge & Automated Evaluation](../11-evaluation/02-llm-as-judge.html)) to get a quality distribution before traffic exposure.
3. **Test infrastructure integration** — does the new model endpoint return the expected JSON schema? Are there new failure modes (timeout patterns, empty responses on edge inputs)?


{{fig:online-eval-shadow-deployment}}


### Canary rollout

A canary routes a small, controllable fraction of real traffic to the new model, with automatic rollback triggers tied to guardrail metrics.

```yaml
# Example: canary rollout configuration (Kubernetes + Argo Rollouts style)
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: llm-api-server
spec:
  strategy:
    canary:
      steps:
        - setWeight: 1       # 1% canary
          pause: {duration: 30m}
        - analysis:
            templates:
              - templateName: guardrail-check
        - setWeight: 10      # 10% canary
          pause: {duration: 2h}
        - analysis:
            templates:
              - templateName: guardrail-check
        - setWeight: 50      # 50% — effective A/B
          pause: {duration: 24h}
        - setWeight: 100     # full rollout
      # Automatic rollback if any guardrail fires
      autoPromotionEnabled: false
```

```python
# Guardrail check logic that would back the AnalysisTemplate above
import statistics

def should_rollback(
    canary_metrics: dict,
    baseline_metrics: dict,
    thresholds: dict,
) -> tuple[bool, str]:
    """
    Returns (rollback, reason) based on guardrail metric comparisons.
    Thresholds define maximum *relative* degradation allowed.

    Example thresholds:
      {
        "thumb_down_rate": 0.20,     # allow up to 20% increase
        "latency_p99_ms": 0.30,      # allow up to 30% increase
        "cost_per_session_usd": 0.15,
        "safety_violation_rate": 0.0, # zero tolerance
      }
    """
    for metric, max_relative_increase in thresholds.items():
        baseline_val = baseline_metrics.get(metric)
        canary_val = canary_metrics.get(metric)
        if baseline_val is None or canary_val is None:
            continue
        if baseline_val == 0:
            if canary_val > 0:
                return True, f"{metric}: baseline=0, canary={canary_val} (zero tolerance)"
            continue

        relative_change = (canary_val - baseline_val) / baseline_val
        if relative_change > max_relative_increase:
            return True, (
                f"{metric}: baseline={baseline_val:.4f}, canary={canary_val:.4f}, "
                f"relative increase={relative_change:.1%} > threshold={max_relative_increase:.1%}"
            )

    return False, "all guardrails passed"


# Example call
rollback, reason = should_rollback(
    canary_metrics={
        "thumb_down_rate": 0.062,
        "latency_p99_ms": 1850,
        "cost_per_session_usd": 0.041,
        "safety_violation_rate": 0.0,
    },
    baseline_metrics={
        "thumb_down_rate": 0.055,
        "latency_p99_ms": 1420,
        "cost_per_session_usd": 0.038,
        "safety_violation_rate": 0.0,
    },
    thresholds={
        "thumb_down_rate": 0.20,
        "latency_p99_ms": 0.30,
        "cost_per_session_usd": 0.15,
        "safety_violation_rate": 0.0,
    },
)
print(f"Rollback: {rollback}, reason: {reason}")
# Rollback: False, reason: all guardrails passed
# (P99 latency increased 30.3% — just barely passes; in practice, tighten to 0.25)
```

A well-run canary pipeline can catch regressions within minutes. The 1% initial stage exists specifically for catastrophic failures (model returns empty strings, crashes with certain inputs). The 10% stage provides the first statistically meaningful read on behavioral metrics. The 50% stage is where you run the full hypothesis test.

---

## Live Judges: Scoring Sampled Production Traffic

Even with good behavioral metrics, they are *proxies*. The only way to know whether output quality has changed is to evaluate outputs directly. Live judges do this continuously.

### The pipeline


{{fig:online-eval-live-judge-pipeline}}


The LLM judge rates each sampled response on a rubric (helpfulness, accuracy, safety, format). Human raters review a smaller fraction (on the order of 0.05–0.1% of traffic) to calibrate judge accuracy and detect drift in judge behavior. See [LLM-as-a-Judge & Automated Evaluation](../11-evaluation/02-llm-as-judge.html) for the judge design.

### Stratified sampling

Uniform sampling misses tail behaviors. A stratified sampler over-represents:

- **New users** (first-session queries are often unusual)
- **Long sessions** (users who are stuck)
- **Queries with low confidence** (model's own log-probability is low)
- **Queries triggering guardrails** (safety classifier near-misses)
- **Queries in underrepresented categories** (rare intents)

```python
import random
from typing import Callable

def stratified_sampler(
    request: dict,
    base_rate: float = 0.005,  # 0.5% baseline
    boost_rules: list[tuple[Callable[[dict], bool], float]] | None = None,
) -> bool:
    """
    Returns True if this request should be sampled for judging.

    boost_rules: list of (predicate, multiplier) pairs. The highest
    applicable multiplier is used (not additive, to avoid double-counting).
    """
    if boost_rules is None:
        boost_rules = []

    effective_rate = base_rate
    for predicate, multiplier in boost_rules:
        if predicate(request):
            effective_rate = max(effective_rate, base_rate * multiplier)

    return random.random() < effective_rate

# Example configuration
boost_rules = [
    (lambda r: r.get("session_turn_count", 0) == 1, 5.0),    # first turn
    (lambda r: r.get("session_length", 0) > 10, 4.0),         # long session
    (lambda r: r.get("model_log_prob", 0.0) < -2.5, 8.0),     # low-confidence
    (lambda r: r.get("safety_score", 0.0) > 0.4, 20.0),       # near-miss safety
    (lambda r: r.get("user_is_new", False), 3.0),              # new user
]

# Simulate over 1M requests
sample_count = sum(
    1 for _ in range(1_000_000)
    if stratified_sampler(
        {"session_turn_count": random.randint(1, 15),
         "model_log_prob": random.gauss(-1.0, 1.5)},
        boost_rules=boost_rules,
    )
)
print(f"Estimated sample rate: {sample_count / 1_000_000:.2%}")
```

### Monitoring quality metrics over time

Rather than a single pass/fail, maintain rolling statistics so regressions are visible as trends:

```python
from collections import deque
import numpy as np

class RollingQualityMonitor:
    """
    Maintains a sliding window of judge scores and raises an alert
    when the mean score drops below a configurable threshold.
    """

    def __init__(self, window_size: int = 1000, alert_threshold: float = 0.05):
        self.scores = deque(maxlen=window_size)
        self.alert_threshold = alert_threshold  # max allowed drop from baseline
        self.baseline_mean: float | None = None

    def set_baseline(self, scores: list[float]) -> None:
        """Call once with initial production scores to establish baseline."""
        self.baseline_mean = float(np.mean(scores))

    def add_score(self, score: float) -> dict:
        """Add a new judge score; returns alert status."""
        self.scores.append(score)
        current_mean = float(np.mean(self.scores))
        alert = False
        reason = None

        if self.baseline_mean is not None and len(self.scores) >= 50:
            drop = (self.baseline_mean - current_mean) / self.baseline_mean
            if drop > self.alert_threshold:
                alert = True
                reason = (
                    f"Mean quality dropped {drop:.1%} below baseline "
                    f"(current={current_mean:.3f}, baseline={self.baseline_mean:.3f})"
                )

        return {
            "current_mean": current_mean,
            "n_samples": len(self.scores),
            "alert": alert,
            "reason": reason,
        }
```

---

## Biases That Corrupt Online Signals

Online experiments are not automatically reliable. Several systematic biases can lead to wrong conclusions.

### Novelty effect

Users often engage more positively with any change, simply because it is new. A new UI for the LLM chat box may generate elevated thumbs-up rates for the first few days regardless of underlying quality. The novelty effect typically decays over one to two weeks.

**Defense:** Run experiments for at least two weeks, and segment by user tenure (days since first use). If new users show higher lift than returning users, novelty is likely the confounder. Report the effect separately for users who have been in the experiment for more than one week.

### Feedback-loop bias

In many LLM applications, model outputs become inputs to future sessions: documents users compose with an AI, code they commit, answers they cite. Over time, the model trained on data from production begins to see its own outputs in training data, creating a feedback loop. Metrics may improve because the world has shifted to match the model, not because the model has improved.

**Defense:** Maintain a **holdout cohort** — a randomly selected 1–5% of users who are never exposed to any treatment variant. Compare the holdout's metric trajectory over months to detect secular trends in the overall user base.

### Position and presentation bias

If your experiment changes how responses are presented (length, formatting, bullet points vs. prose), behavioral metrics like copy rate and click-through can change for presentation reasons unrelated to quality.

**Defense:** Separate model quality experiments from UI experiments. Run them sequentially, not simultaneously, unless your experiment platform supports factorial designs and you have sufficient traffic for interaction terms.

### Selection bias from opt-in feedback

Thumb ratings are provided by a self-selected subset of users — typically those with strong opinions (very positive or very negative). The non-response majority may have different preferences. A treatment that increases the thumbs-up *count* may simply be eliciting more feedback, not improving quality.

**Defense:** Track the feedback *rate* (what fraction of sessions result in any rating) alongside the *conditional* thumbs-up rate. A rising feedback rate with stable conditional thumbs-up rate signals more engagement, not more satisfaction.

### Peeking and multiple testing

See the sequential testing discussion above. Additionally, if you are running 10 simultaneous A/B experiments (common in fast-moving teams), the probability that at least one shows a false positive at $\alpha = 0.05$ is $1 - 0.95^{10} \approx 40\%$. Apply Bonferroni correction or control the False Discovery Rate (Benjamini-Hochberg) when reporting across many simultaneous tests.

---

## Connecting the Pieces: The Evaluation Lifecycle

A mature evaluation stack is not a collection of ad-hoc tests — it is a pipeline that continuously validates every change.


{{fig:online-eval-lifecycle-stages}}


The lifecycle makes explicit that offline evaluation, shadow testing, canaries, A/B experiments, and live judging are not alternatives — they are sequential filters. Each stage catches different failure modes, and bypassing any stage (usually in the name of speed) shifts the risk onto downstream stages or onto users.

---

!!! interview "Interview Corner"
    **Q:** You're a PM at a company running an A/B test for a new LLM feature. After 3 days, the treatment arm shows a statistically significant 8% lift in thumbs-up rate (p = 0.02). Your manager wants to ship immediately. What do you tell them?

    **A:** Three days is almost certainly too short for three reasons. First, there is a novelty effect: users engage more positively with any change for the first week; the real signal may be much smaller or even negative once novelty decays. Second, three days may not capture weekly usage patterns — if thumbs-up rates differ between weekdays and weekends, a three-day window is biased by which days are included. Third, we need to check guardrail metrics: did latency, safety violation rate, or cost change? An 8% lift in thumbs-up is worthless if safety violations increased 50%. The recommendation is to run the experiment for at least two weeks, check guardrails, and segment the effect by user tenure in the experiment. Additionally, if we care about resolution rate (not just thumbs-up, which is a proxy), we may need even longer to observe enough resolved sessions for significance.

---

!!! key "Key Takeaways"
    - Offline metrics are proxies for online outcomes, not substitutes. Track the offline-online correlation across launches to know how much to trust your evaluation suite.
    - Guardrail metrics (safety, latency, cost, regeneration rate) should be defined *before* the experiment and treated as hard blockers, not advisory.
    - Randomize at the user-session level, not the request level, to satisfy SUTVA and avoid contamination within sessions.
    - CUPED can halve required sample sizes by exploiting pre-experiment correlations; sequential testing (always-valid p-values) allows anytime peeking without inflating false positive rates.
    - Interleaving surfaces preference signals with 10–30x fewer user-sessions than parallel A/B; use it for quick directional reads on completion quality.
    - Shadow deployments let you profile latency, cost, and judge scores at production scale before any user sees the new model.
    - Canary rollouts with automated guardrail-based rollback catch infrastructure and behavioral regressions at 1% traffic before they become incidents.
    - Novelty effects, feedback-loop bias, and selection bias in opt-in feedback can all produce misleading online signals; hold-out cohorts and segmentation by user tenure are the primary defenses.
    - Live LLM judges scoring sampled production traffic provide continuous quality monitoring even after full rollout, decoupling quality assurance from the experiment lifecycle.

---

!!! sota "State of the Art & Resources (2026)"
    Online evaluation of LLM systems is a mature engineering discipline: production teams routinely combine sequential testing (always-valid p-values), CUPED variance reduction, interleaving preference signals, and canary rollouts with automated guardrail-based rollback. Open-source experimentation platforms and LLM-observability frameworks have made these techniques accessible to teams of any size.

    **Foundational work**

    - [Deng et al., *Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data* (WSDM 2013)](https://dl.acm.org/doi/10.1145/2433396.2433413) — the original CUPED paper; still the canonical reference for variance reduction in A/B tests.
    - [Radlinski & Craswell, *Optimized Interleaving for Online Retrieval Evaluation* (WSDM 2013)](https://dl.acm.org/doi/10.1145/2433396.2433429) — formalises interleaving as an optimisation problem; underpins the 10–100× efficiency gains over parallel A/B for preference signals.
    - [Benjamini & Hochberg, *Controlling the False Discovery Rate* (JRSS-B 1995)](https://academic.oup.com/jrsssb/article/57/1/289/7035855) — the FDR correction used when running many simultaneous experiments.

    **Recent advances (2023–2026)**

    - [Johari et al., *Always Valid Inference: Bringing Sequential Analysis to A/B Testing* (2015/2019)](https://arxiv.org/abs/1512.04922) — derives always-valid p-values from mSPRT, enabling anytime peeking without type-I error inflation; deployed by Optimizely and Statsig.
    - [Howard, Ramdas et al., *Time-uniform, nonparametric, nonasymptotic confidence sequences* (Annals of Statistics 2021)](https://arxiv.org/abs/1810.08240) — the theoretical backbone for Robbins confidence sequences used in sequential A/B testing.
    - [Johari, Koomen, Pekelis & Walsh, *Peeking at A/B Tests: Why It Matters, and What to Do About It* (KDD 2017)](https://dl.acm.org/doi/10.1145/3097983.3097992) — practical treatment of the peeking problem with mSPRT; the industry reference for always-valid experimentation.

    **Open-source & tools**

    - [argoproj/argo-rollouts](https://github.com/argoproj/argo-rollouts) — Kubernetes progressive delivery controller; canary and blue-green rollouts with metric-based automatic promotion/rollback as shown in the chapter.
    - [evidentlyai/evidently](https://github.com/evidentlyai/evidently) — open-source ML and LLM observability framework with 100+ metrics for continuous quality monitoring of production traffic.
    - [growthbook/growthbook](https://github.com/growthbook/growthbook) — open-source A/B testing and feature-flag platform with built-in CUPED, Bayesian, and sequential statistics; used by several major LLM companies.

    **Go deeper**

    - [Kohavi, Tang & Xu, *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing* (Cambridge UP, 2020)](https://www.cambridge.org/core/books/trustworthy-online-controlled-experiments/D97B26382EB0EB2DC2019A7A7B518F59) — the definitive practitioner book on running experiments at scale, written by leaders from Microsoft, Google, and LinkedIn.
    - [Statsig, *Beyond Prompts: A Data-Driven Approach to LLM Optimization* (2024)](https://www.statsig.com/blog/llm-optimization-online-experimentation) — end-to-end walkthrough of applying online A/B experimentation to prompt, model, and parameter tuning for LLM products.

## Further Reading

- Deng, A., Xu, Y., Kohavi, R., Walker, T. — "Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data" (CUPED), *WSDM 2013*.
- Johari, R., Koomen, P., Pekelis, L., Walsh, D. — "Peeking at A/B Tests: Why It Matters, and What to Do About It" (mSPRT), *KDD 2017*.
- Howard, S. R., Ramdas, A., McAuliffe, J., Sekhon, J. — "Time-uniform, nonparametric, nonasymptotic confidence sequences," *Annals of Statistics 2021*.
- Radlinski, F., Craswell, N. — "Optimized Interleaving for Online Retrieval Evaluation," *WSDM 2013*.
- Kohavi, R., Tang, D., Xu, Y. — *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing*, Cambridge University Press, 2020.
- Benjamini, Y., Hochberg, Y. — "Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing," *Journal of the Royal Statistical Society B, 1995*.
- Gu, Y. et al. — "A Survey of LLM Evaluation" (covers live evaluation methodology), arXiv, 2024.
