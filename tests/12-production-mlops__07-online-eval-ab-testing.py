"""
Runnability test for content/12-production-mlops/07-online-eval-ab-testing.md

Chapter has 7 heuristically CPU-runnable Python blocks:
  - block #0 (line ~112): assign_variant() + two_proportion_z_test()
  - block #1 (line ~226): InterleavingSession / compute_interleaving_win_rate()
  - block #2 (line ~300): cuped_estimate()
  - block #3 (line ~378): always_valid_ci()
  - block #5 (line ~455): should_rollback()
  - block #6 (line ~548): stratified_sampler()
  - block #7 (line ~598): RollingQualityMonitor

Skipped:
  - block #4 (line ~428): YAML Argo Rollouts config, not Python. # SKIP(non-python)

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
# Block #3 (line ~378): always-valid (anytime) confidence interval
# =========================================================================

def always_valid_ci(
    n: int,
    k: int,       # successes so far
    alpha: float = 0.05,
) -> "tuple[float, float]":
    """
    Approximate always-valid (anytime) confidence interval for a
    Bernoulli proportion using the Howard et al. (2021) normal-mixture bound.

    This CI is valid at any sample size n >= 1 without correction for peeking.
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


_widths = []
for _n, _k in [(100, 40), (1000, 400), (5000, 2000)]:
    _lo, _hi = always_valid_ci(_n, _k)
    print(f"n={_n:5d}: {_lo:.3f} - {_hi:.3f}  (width {_hi-_lo:.3f})")
    assert 0.0 <= _lo <= _hi <= 1.0
    _widths.append(_hi - _lo)
# The confidence sequence must shrink as more data accumulates.
assert _widths[0] > _widths[1] > _widths[2]

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
