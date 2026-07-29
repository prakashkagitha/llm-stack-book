"""
Runnability test for content/12-production-mlops/07-online-eval-ab-testing.md

Chapter has 9 heuristically CPU-runnable Python blocks:
  - block #0: assign_variant() + two_proportion_z_test()
  - block #1: srm_check()
  - block #2: ratio_metric_delta_method()
  - block #3: InterleavingSession / compute_interleaving_win_rate()
  - block #4: cuped_estimate()
  - block #5: cs_halfwidth() / tune_rho() / always_valid_ci()
  - block #7: should_rollback()
  - block #8: stratified_sampler()
  - block #9: RollingQualityMonitor

Skipped:
  - Argo Rollouts + Istio VirtualService YAML, not Python. # SKIP(non-python)

scipy is NOT in the guaranteed-available import list for this test harness
(only numpy, torch-cpu, einops, scikit-learn, and stdlib are guaranteed), so
`scipy.stats` is imported defensively at module scope. Blocks #0, #1, #2 use
it faithfully (matching the book's code) when available, and are skipped
individually if it is not.

Real bugs found & fixed in the book's source (mirrored here):
  1. The interleaving block used `scipy.stats.binom_test`, which was
     deprecated in SciPy 1.7 and removed entirely in SciPy 1.12+. Fixed to
     use `scipy.stats.binomtest(...).pvalue`.
  2. The `should_rollback` worked example used
     canary_metrics["latency_p99_ms"]=1850 vs baseline=1420, a relative
     increase of 30.28% against a 0.30 (30%) threshold with a strict `>`
     comparison — so the *actual* output is `Rollback: True`, contradicting
     the book's claimed `Rollback: False, reason: all guardrails passed`.
     Fixed the example's canary latency to 1845 (a 29.9% increase), which
     genuinely passes and matches the "just barely passes" narrative.
"""

import hashlib
import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Callable, Literal, NamedTuple

import numpy as np

try:
    from scipy import stats
except Exception:
    stats = None


# =========================================================================
# Block #0 (line ~112): deterministic user assignment + two-proportion z-test
# =========================================================================

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


if stats is not None:
    # Example usage:
    result = two_proportion_z_test(
        n_control=10_000, k_control=4_000,
        n_treatment=10_000, k_treatment=4_200,
    )
    # Expected: ~+5% relative lift, p ≈ 0.001 → significant
    print(result)
    assert bool(result["significant"]) is True
    assert abs(result["relative_lift"] - 0.05) < 1e-9
else:
    print("SKIP(no scipy): block #0 two_proportion_z_test call skipped")

# Sanity-check assign_variant is stable and produces all arms across users.
_exp = Experiment(experiment_id="exp-1", traffic_fraction=1.0, treatment_fraction=0.5)
_assignments = {assign_variant(f"user-{i}", _exp) for i in range(200)}
assert _assignments <= {"control", "treatment", "holdout"}
assert assign_variant("user-42", _exp) == assign_variant("user-42", _exp)  # deterministic
print(f"assign_variant arms observed over 200 users: {_assignments}")


# =========================================================================
# Block #1 (line ~226): interleaving win rate
# =========================================================================

class InterleavingSession(NamedTuple):
    user_id: str
    control_response: str
    treatment_response: str
    # Which response did the user take a positive action on?
    # 'control', 'treatment', or 'none'
    preferred: str


def compute_interleaving_win_rate(
    sessions: "list[InterleavingSession]",
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
    # (scipy.stats.binom_test was deprecated in SciPy 1.7 and removed in 1.12+;
    # use the modern binomtest API, which returns a result object.)
    p_value = stats.binomtest(wins_treatment, n, p=0.5, alternative="two-sided").pvalue

    return {
        "win_rate": win_rate,
        "n_decisive": n,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }


if stats is not None:
    _rng = random.Random(7)
    _sessions = [
        InterleavingSession(
            user_id=f"u{i}",
            control_response="c",
            treatment_response="t",
            preferred=_rng.choices(["control", "treatment", "none"], weights=[35, 55, 10])[0],
        )
        for i in range(300)
    ]
    interleaving_result = compute_interleaving_win_rate(_sessions)
    print(interleaving_result)
    assert 0.0 <= interleaving_result["win_rate"] <= 1.0
    assert interleaving_result["n_decisive"] <= 300
else:
    print("SKIP(no scipy): block #1 compute_interleaving_win_rate call skipped")


# =========================================================================
# Block #2 (line ~300): CUPED variance reduction
# =========================================================================

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


if stats is not None:
    # Simulate: 500 users per arm, thumbs-up rate 0.40 control / 0.42 treatment
    rng = np.random.default_rng(42)
    n = 500
    x_c = rng.binomial(1, 0.40, n).astype(float)  # pre-exp covariate
    x_t = rng.binomial(1, 0.40, n).astype(float)
    # In-experiment: add treatment effect + correlation with pre-exp
    y_c = np.clip(x_c * 0.7 + rng.binomial(1, 0.12, n), 0, 1)
    y_t = np.clip(x_t * 0.7 + rng.binomial(1, 0.14, n), 0, 1)

    cuped_result = cuped_estimate(y_c, y_t, x_c, x_t)
    print(f"Delta: {cuped_result['delta']:.4f}, p={cuped_result['p_value']:.4f}, "
          f"variance reduction: {cuped_result['variance_reduction_fraction']:.1%}")
    assert 0.0 <= cuped_result["variance_reduction_fraction"] <= 1.0
else:
    print("SKIP(no scipy): block #2 cuped_estimate call skipped")

# =========================================================================
# Block #1: sample ratio mismatch (SRM) chi-square check
# Block #2: delta-method SE for a ratio-of-sums metric
# =========================================================================

if stats is not None:

    def srm_check(
        n_control: int,
        n_treatment: int,
        expected_treatment_fraction: float = 0.5,
        alarm_p: float = 0.001,
    ) -> dict:
        """Chi-square goodness-of-fit test on the observed traffic split."""
        total = n_control + n_treatment
        expected = [total * (1 - expected_treatment_fraction),
                    total * expected_treatment_fraction]
        observed = [n_control, n_treatment]
        chi2 = sum((o - e) ** 2 / e for o, e in zip(observed, expected))
        p_value = float(stats.chi2.sf(chi2, df=1))
        return {"observed_treatment_fraction": n_treatment / total,
                "chi2": chi2, "p_value": p_value, "srm": p_value < alarm_p}

    _big = srm_check(100_000, 98_000)
    _small = srm_check(1_000, 980)
    print(_big)
    print(_small)
    # Same 1% imbalance: an alarm at 100k sessions, pure noise at 1k.
    assert _big["srm"] is True and _big["p_value"] < 1e-4
    assert _small["srm"] is False and _small["p_value"] > 0.1
    # A perfectly balanced split must never fire.
    assert srm_check(50_000, 50_000)["p_value"] == 1.0
else:
    print("SKIP(no scipy): block #1 srm_check skipped")


def ratio_metric_delta_method(y: np.ndarray, d: np.ndarray) -> dict:
    """
    SE of a ratio-of-sums metric when the randomisation unit is the user:
    y[i] = user i's numerator events, d[i] = user i's denominator events.
    """
    k = len(y)
    d_bar = d.mean()
    m = y.sum() / d.sum()
    var = (np.var(y, ddof=1)
           - 2 * m * np.cov(y, d, ddof=1)[0, 1]
           + m ** 2 * np.var(d, ddof=1)) / (k * d_bar ** 2)
    return {"metric": m, "se": float(np.sqrt(var)), "n_users": k}


_rng = np.random.default_rng(0)
_k_users = 2_000
_d = _rng.poisson(8, _k_users) + 1
_p_user = _rng.beta(4, 6, _k_users)
_y = _rng.binomial(_d, _p_user)

_clustered = ratio_metric_delta_method(_y, _d)
_m = _clustered["metric"]
_naive_se = np.sqrt(_m * (1 - _m) / _d.sum())
_ratio = _clustered["se"] / _naive_se
print(f"metric={_m:.4f}  clustered SE={_clustered['se']:.4f}  "
      f"naive SE={_naive_se:.4f}  ratio={_ratio:.2f}x")
# The book claims metric=0.3960, clustered SE=0.0049, naive SE=0.0036, 1.34x.
assert abs(_m - 0.3960) < 5e-4
assert abs(_clustered["se"] - 0.0049) < 5e-4
assert abs(_naive_se - 0.0036) < 5e-4
assert abs(_ratio - 1.34) < 0.02
# Clustering can only ever widen the interval relative to the i.i.d. fiction.
assert _ratio > 1.0
# The book also claims the factor grows with events per user: ~2x at 30/user.
_d30 = _rng.poisson(30, _k_users) + 1
_y30 = _rng.binomial(_d30, _rng.beta(4, 6, _k_users))
_c30 = ratio_metric_delta_method(_y30, _d30)
_naive30 = np.sqrt(_c30["metric"] * (1 - _c30["metric"]) / _d30.sum())
assert _c30["se"] / _naive30 > _ratio, "clustering penalty must grow with d/user"


# =========================================================================
# Block #5: always-valid (anytime) confidence sequence
# =========================================================================

def cs_halfwidth(n: int, rho: float, alpha: float = 0.05,
                 sigma: float = 0.5) -> float:
    """Half-width of the normal-mixture confidence sequence at sample size n."""
    return sigma * math.sqrt(
        2.0 * (n * rho ** 2 + 1.0) / (n ** 2 * rho ** 2)
        * math.log(math.sqrt(n * rho ** 2 + 1.0) / alpha)
    )


def tune_rho(n_target: int, alpha: float = 0.05) -> float:
    """Choose where the confidence sequence is tightest (ternary search)."""
    lo, hi = 1e-5, 10.0
    for _ in range(200):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if cs_halfwidth(n_target, m1, alpha) < cs_halfwidth(n_target, m2, alpha):
            hi = m2
        else:
            lo = m1
    return 0.5 * (lo + hi)


def always_valid_ci(
    n: int,
    k: int,
    alpha: float = 0.05,
    n_target: int = 10_000,
) -> "tuple[float, float]":
    """Anytime-valid CI for a Bernoulli proportion; valid at every n >= 1."""
    p_hat = k / n
    w = cs_halfwidth(n, tune_rho(n_target, alpha), alpha)
    return (max(0.0, p_hat - w), min(1.0, p_hat + w))


_widths = []
for _n, _k in [(100, 40), (1_000, 400), (5_000, 2_000), (20_000, 8_000)]:
    _lo, _hi = always_valid_ci(_n, _k, n_target=10_000)
    _p = _k / _n
    _fixed = 2 * 1.96 * math.sqrt(_p * (1 - _p) / _n)
    print(f"n={_n:6d}: anytime {_lo:.3f}-{_hi:.3f} (width {_hi-_lo:.3f})"
          f"   fixed-n width {_fixed:.3f}")
    assert 0.0 <= _lo <= _hi <= 1.0
    # An anytime-valid interval is never narrower than the fixed-n interval.
    assert (_hi - _lo) > _fixed
    _widths.append(_hi - _lo)
# The confidence sequence must shrink as more data accumulates.
assert _widths[0] > _widths[1] > _widths[2] > _widths[3]
# The book quotes these two rows verbatim.
assert abs(_widths[1] - 0.121) < 1e-3
assert abs(_widths[3] - 0.022) < 1e-3
# tune_rho really does minimise the boundary at its target.
_r = tune_rho(10_000)
assert cs_halfwidth(10_000, _r) < cs_halfwidth(10_000, _r * 2)
assert cs_halfwidth(10_000, _r) < cs_halfwidth(10_000, _r / 2)

# Block #4 (line ~428): YAML Argo Rollouts canary config, not Python.
# SKIP(non-python): nothing to execute.


# =========================================================================
# Block #5 (line ~455): guardrail rollback check
# =========================================================================

import statistics  # noqa: F401  (imported by the book's block; unused by the logic)


def should_rollback(
    canary_metrics: dict,
    baseline_metrics: dict,
    thresholds: dict,
) -> "tuple[bool, str]":
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
        "latency_p99_ms": 1845,
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
# (P99 latency increased 29.9% — just barely passes; in practice, tighten to 0.25)
assert rollback is False
assert reason == "all guardrails passed"

# Also exercise the rollback=True branch, since the book's example only
# hits the passing path.
rollback2, reason2 = should_rollback(
    canary_metrics={"safety_violation_rate": 0.01},
    baseline_metrics={"safety_violation_rate": 0.0},
    thresholds={"safety_violation_rate": 0.0},
)
assert rollback2 is True
print(f"Rollback (zero-tolerance breach): {rollback2}, reason: {reason2}")


# =========================================================================
# Block #6 (line ~548): stratified sampler for live judging
# =========================================================================

def stratified_sampler(
    request: dict,
    base_rate: float = 0.005,  # 0.5% baseline
    boost_rules: "list[tuple[Callable[[dict], bool], float]] | None" = None,
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
assert sample_count > 0


# =========================================================================
# Block #7 (line ~598): rolling quality monitor
# =========================================================================

class RollingQualityMonitor:
    """
    Maintains a sliding window of judge scores and raises an alert
    when the mean score drops below a configurable threshold.
    """

    def __init__(self, window_size: int = 1000, alert_threshold: float = 0.05):
        self.scores = deque(maxlen=window_size)
        self.alert_threshold = alert_threshold  # max allowed drop from baseline
        self.baseline_mean: "float | None" = None

    def set_baseline(self, scores: "list[float]") -> None:
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


# Exercise the class: establish a baseline, then feed 60 scores where the
# last 20 have degraded quality, and confirm the monitor fires an alert.
_monitor = RollingQualityMonitor(window_size=200, alert_threshold=0.05)
_monitor.set_baseline([0.80] * 100)

_status = None
for i in range(60):
    score = 0.80 if i < 40 else 0.55  # quality drop starting at sample 40
    _status = _monitor.add_score(score)

print(_status)
assert _status["n_samples"] == 60
assert _status["alert"] is True
assert _status["reason"] is not None


print("\nAll runnable blocks in 12-production-mlops/07-online-eval-ab-testing.md executed successfully.")
