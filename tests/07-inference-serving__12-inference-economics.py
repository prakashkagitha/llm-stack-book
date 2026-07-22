"""
Runs the CPU-runnable Python blocks from
content/07-inference-serving/12-inference-economics.md verbatim (concatenated
in chapter order, since later blocks may reuse names from earlier ones) and
adds minimal glue so every block actually executes.

Blocks tested:
  - block #0 (line ~115): bandwidth_breakeven / decode_step_time_ms roofline model
  - block #2 (line ~316): AutoscalerConfig / AutoscalerState / autoscale_step
  - block #3 (line ~436): classify_complexity / route_query / simulate_cost cost router
  - block #4 (line ~559): INFERENCE_METRICS / compute_cost_per_1m_tokens

Block #1 is a `text` (non-python, illustrative console output) block — SKIP,
nothing to execute.

All imports used by these blocks are stdlib only (math, dataclasses,
collections, time, typing, random) — no optional third-party imports needed.
"""

import math

# =====================================================================
# Block #0 (line ~115): illustrative scheduler — bandwidth breakeven batch
# size and decode step time model.
# =====================================================================


def bandwidth_breakeven(flops_per_sec: float, bandwidth_bytes_per_sec: float) -> int:
    """
    Return the batch size B* at which decode transitions from bandwidth-bound
    to compute-bound. Below B*, adding more sequences to the batch is free in
    terms of time (you're already waiting for weights to stream from HBM).
    Above B*, decode time grows linearly with batch size.

    Args:
        flops_per_sec: Peak BF16 FLOP/s (e.g., 989e12 for H100 SXM5)
        bandwidth_bytes_per_sec: HBM bandwidth in bytes/s (e.g., 3.35e12 for H100)

    Returns:
        B*: arithmetic intensity breakeven batch size
    """
    return int(flops_per_sec / bandwidth_bytes_per_sec)


def decode_step_time_ms(
    n_params: int,
    batch_size: int,
    flops_per_sec: float,
    bandwidth_bytes_per_sec: float,
    bytes_per_param: int = 2,  # BF16 / FP16
) -> float:
    """
    Estimate the wall-clock time for one decode step (one new token per sequence).

    The model weights must be streamed from HBM once per step regardless of batch
    size (bandwidth-bound regime). At large batch sizes the MMA units become the
    bottleneck (compute-bound regime).
    """
    # Bytes read: all parameters once (both weight read and result write)
    bytes_read = n_params * bytes_per_param
    # FLOPs: 2 MACs per parameter per batch element
    flops = 2 * n_params * batch_size

    # Time in each regime (seconds)
    t_bandwidth = bytes_read / bandwidth_bytes_per_sec
    t_compute = flops / flops_per_sec

    # Actual time is the max; report in ms
    return max(t_bandwidth, t_compute) * 1000.0


# --- Hardware constants ---
H100_FLOPS = 989e12  # BF16 tensor core FLOP/s (H100 SXM5)
H100_BW = 3.35e12  # HBM bandwidth bytes/s

# --- Model: Llama 3 70B (70e9 params, BF16) ---
N_PARAMS = 70e9

B_STAR = bandwidth_breakeven(H100_FLOPS, H100_BW)
print(f"H100 arithmetic intensity breakeven batch size: {B_STAR}")

for batch in [1, 4, 16, 64, 128, 256, B_STAR]:
    t = decode_step_time_ms(N_PARAMS, batch, H100_FLOPS, H100_BW)
    regime = "bandwidth-bound" if batch <= B_STAR else "compute-bound"
    print(f"  batch={batch:4d}  step_time={t:.2f} ms  regime={regime}")

# --- glue: verify the book's claimed numbers (B* ~= 295, flat ~41.79ms floor) ---
assert B_STAR == 295, f"expected B* = 295, got {B_STAR}"
step_time_at_1 = decode_step_time_ms(N_PARAMS, 1, H100_FLOPS, H100_BW)
step_time_at_bstar = decode_step_time_ms(N_PARAMS, B_STAR, H100_FLOPS, H100_BW)
assert math.isclose(step_time_at_1, step_time_at_bstar, rel_tol=1e-9), (
    "decode step time should be flat (bandwidth-bound) up to B*"
)
assert math.isclose(step_time_at_1, 41.79, abs_tol=0.01), (
    f"expected ~41.79ms bandwidth floor, got {step_time_at_1:.2f}"
)
# One step past B* must strictly increase step time (compute-bound regime).
step_time_past_bstar = decode_step_time_ms(N_PARAMS, B_STAR + 50, H100_FLOPS, H100_BW)
assert step_time_past_bstar > step_time_at_bstar


# =====================================================================
# Block #1 (line ~183): `text` block showing illustrative console output.
# SKIP(non-python): not a code block, nothing to execute.
# =====================================================================


# =====================================================================
# Block #2 (line ~316): minimal autoscaler mock — decides replica count
# from queue depth and current throughput, with hysteresis.
# =====================================================================

from dataclasses import dataclass, field
from collections import deque
import time


@dataclass
class AutoscalerConfig:
    min_replicas: int = 1
    max_replicas: int = 16
    target_tokens_per_sec_per_replica: float = 2000.0  # sustained decode tps/replica
    scale_up_queue_threshold: int = 100  # tokens queued -> add replica
    scale_down_idle_seconds: float = 120.0  # idle for 2 min -> remove replica
    cooldown_seconds: float = 60.0  # min time between scale events


@dataclass
class AutoscalerState:
    n_replicas: int = 1
    last_scale_time: float = field(default_factory=time.time)
    idle_since: dict = field(default_factory=dict)  # replica_id -> time became idle


def autoscale_step(
    config: AutoscalerConfig,
    state: AutoscalerState,
    queue_depth_tokens: int,  # tokens currently waiting in the request queue
    active_tokens_per_sec: float,  # current measured throughput
    now: float = None,
) -> int:
    """
    Return the desired number of replicas. Does not actually spin up/down anything —
    the orchestrator (Kubernetes, Ray Serve, etc.) handles the actual scaling.
    """
    if now is None:
        now = time.time()

    # Don't scale more often than the cooldown window
    if now - state.last_scale_time < config.cooldown_seconds:
        return state.n_replicas

    desired = state.n_replicas

    # Scale UP: queue is building faster than current replicas can drain it
    tokens_capacity = state.n_replicas * config.target_tokens_per_sec_per_replica
    if queue_depth_tokens > config.scale_up_queue_threshold:
        # Estimate replicas needed to drain queue within 30 seconds
        needed = (
            int(
                (active_tokens_per_sec + queue_depth_tokens / 30.0)
                / config.target_tokens_per_sec_per_replica
            )
            + 1
        )
        desired = min(needed, config.max_replicas)

    # Scale DOWN: we have spare capacity
    elif (
        active_tokens_per_sec < 0.5 * tokens_capacity
        and state.n_replicas > config.min_replicas
    ):
        desired = max(config.min_replicas, state.n_replicas - 1)

    if desired != state.n_replicas:
        state.n_replicas = desired
        state.last_scale_time = now

    return desired


# --- glue: the chapter defines but never calls this — drive it through a
# tiny scale-up / scale-down scenario, using explicit `now` timestamps so
# the cooldown logic is exercised deterministically (no real sleeping). ---
cfg = AutoscalerConfig(
    min_replicas=1, max_replicas=8, target_tokens_per_sec_per_replica=1000.0,
    scale_up_queue_threshold=100, cooldown_seconds=60.0,
)
state = AutoscalerState(n_replicas=1, last_scale_time=0.0)

# Heavy queue backlog at t=100 (past cooldown from t=0) -> should scale up.
desired_up = autoscale_step(
    cfg, state, queue_depth_tokens=5000, active_tokens_per_sec=900.0, now=100.0
)
assert desired_up > 1, f"expected scale-up under heavy queue, got {desired_up} replicas"
assert state.n_replicas == desired_up
assert state.last_scale_time == 100.0

# Immediately after (t=105, inside cooldown) -> replica count must not change.
desired_cooldown = autoscale_step(
    cfg, state, queue_depth_tokens=5000, active_tokens_per_sec=900.0, now=105.0
)
assert desired_cooldown == state.n_replicas

# Much later (t=300, past cooldown) with near-zero load -> should scale down.
desired_down = autoscale_step(
    cfg, state, queue_depth_tokens=0, active_tokens_per_sec=1.0, now=300.0
)
assert desired_down < desired_up, "expected scale-down once traffic drops"
assert desired_down >= cfg.min_replicas

print(
    f"autoscale demo: scaled {1} -> {desired_up} (load) -> {desired_down} (idle) replicas"
)


# =====================================================================
# Block #3 (line ~436): cost-aware router — send queries to a small or
# large model based on a heuristic complexity proxy.
# =====================================================================

from typing import Literal

ModelTier = Literal["fast_small", "capable_large"]

# Illustrative cost per 1M output tokens (update to your actual prices)
COST_PER_1M = {
    "fast_small": 0.50,  # e.g., a 7B fine-tuned model on cheap hardware
    "capable_large": 8.00,  # e.g., a 70B frontier model
}


def classify_complexity(prompt: str, n_few_shot: int = 0) -> float:
    """
    Estimate query complexity as a float in [0, 1].
    Real systems train a small classifier on human-labeled routing decisions.
    Here we use proxy features: prompt length, question words, code keywords.
    """
    words = prompt.lower().split()
    n_words = len(words)

    code_keywords = {
        "def", "class", "import", "function", "algorithm",
        "implement", "debug", "explain", "analyze",
    }
    hard_keywords = {
        "compare", "contrast", "design", "evaluate", "synthesize",
        "critique", "reason", "proof", "derive",
    }

    code_signal = sum(1 for w in words if w in code_keywords) / max(n_words, 1)
    hard_signal = sum(1 for w in words if w in hard_keywords) / max(n_words, 1)
    length_signal = min(n_words / 200.0, 1.0)  # normalize at 200 words

    return min(1.0, code_signal * 2 + hard_signal * 2 + length_signal * 0.5)


def route_query(prompt: str, complexity_threshold: float = 0.25) -> ModelTier:
    """
    Return which model tier to use for this prompt.
    Below the threshold, the small fast model suffices.
    """
    score = classify_complexity(prompt)
    if score < complexity_threshold:
        return "fast_small"
    return "capable_large"


# Simulate routing 1000 queries and compute expected cost
import random


def simulate_cost(
    n_queries: int = 1000, avg_output_tokens: int = 300, threshold: float = 0.25
) -> dict:
    random.seed(42)
    total_small = 0
    total_large = 0

    # Synthetic prompts: mix of simple and complex
    prompts = (
        ["What is 2+2?"] * 400  # trivial
        + ["List the top 5 European capitals"] * 200  # easy
        + ["Implement a red-black tree in Python with full docstrings"] * 200  # hard
        + ["Compare DPO vs PPO for RLHF alignment"] * 200  # hard
    )
    random.shuffle(prompts)

    for p in prompts[:n_queries]:
        tier = route_query(p, threshold)
        if tier == "fast_small":
            total_small += 1
        else:
            total_large += 1

    cost_small = (total_small * avg_output_tokens / 1e6) * COST_PER_1M["fast_small"]
    cost_large = (total_large * avg_output_tokens / 1e6) * COST_PER_1M["capable_large"]
    cost_all_large = (n_queries * avg_output_tokens / 1e6) * COST_PER_1M["capable_large"]

    return {
        "routed_small": total_small,
        "routed_large": total_large,
        "cost_with_routing": cost_small + cost_large,
        "cost_all_large": cost_all_large,
        "savings_pct": 100.0 * (1 - (cost_small + cost_large) / cost_all_large),
    }


result = simulate_cost()
print(f"Routed to small: {result['routed_small']} / Routed to large: {result['routed_large']}")
print(f"Cost with routing: ${result['cost_with_routing']:.4f}")
print(f"Cost all-large:    ${result['cost_all_large']:.4f}")
print(f"Savings:           {result['savings_pct']:.1f}%")

# --- glue: sanity checks on the simulation the chapter's code produces ---
assert result["routed_small"] + result["routed_large"] == 1000
assert result["cost_with_routing"] < result["cost_all_large"], (
    "routing should always be cheaper than sending everything to the large model"
)
assert result["savings_pct"] > 0
# "What is 2+2?" is trivial and must route to the small tier.
assert route_query("What is 2+2?") == "fast_small"
# A prompt loaded with hard keywords must route to the large tier.
assert (
    route_query("compare contrast design evaluate synthesize critique reason proof derive")
    == "capable_large"
)


# =====================================================================
# Block #4 (line ~559): reference metrics dict + cost-per-1M-tokens
# helper for a production dashboard.
# =====================================================================

INFERENCE_METRICS = {
    # Throughput
    "output_tokens_per_second": "gauge",  # Overall system output rate
    "input_tokens_per_second": "gauge",  # Overall system prefill rate
    # Latency
    "ttft_p50_ms": "gauge",  # Median time to first token
    "ttft_p99_ms": "gauge",  # p99 time to first token
    "tbt_p50_ms": "gauge",  # Median time between tokens
    "tbt_p99_ms": "gauge",  # p99 time between tokens
    # Efficiency
    "gpu_utilization_pct": "gauge",  # Per-GPU utilization
    "kv_cache_utilization_pct": "gauge",  # KV cache fill rate (via vLLM metrics)
    "batch_size_mean": "gauge",  # Average active batch size
    "batch_size_p99": "gauge",  # p99 active batch size
    # Cost
    "cost_per_1m_output_tokens": "gauge",  # Derived: $/1M output tokens
    "gpu_cost_per_hour": "gauge",  # Cluster cost rate (from cloud API or fixed)
    "requests_per_dollar": "gauge",  # Inverse cost efficiency
    # Quality proxy
    "generation_errors_per_min": "gauge",  # Truncations, OOM aborts, timeouts
    "queue_depth_tokens": "gauge",  # Pending tokens in request queue
}


def compute_cost_per_1m_tokens(
    gpu_cost_per_hour: float,
    output_tokens_per_second: float,
) -> float:
    """
    Compute the current effective cost per 1M output tokens from live metrics.
    This number should be tracked and alerted on if it exceeds budget.
    """
    if output_tokens_per_second <= 0:
        return float("inf")
    return (gpu_cost_per_hour * 1e6) / (output_tokens_per_second * 3600.0)


# --- glue: the chapter defines this dict/function but never calls the
# function or exercises the dict — do both with tiny fixed inputs. ---
assert len(INFERENCE_METRICS) == 15
assert all(v == "gauge" for v in INFERENCE_METRICS.values())

# 4x H100 cluster at $5/hr/GPU sustaining 8000 output tok/s.
cost_per_1m = compute_cost_per_1m_tokens(gpu_cost_per_hour=20.0, output_tokens_per_second=8000.0)
print(f"cost_per_1m_output_tokens = ${cost_per_1m:.4f}")
expected = (20.0 * 1e6) / (8000.0 * 3600.0)
assert math.isclose(cost_per_1m, expected, rel_tol=1e-9)
assert math.isclose(cost_per_1m, 0.6944, abs_tol=1e-3)

# Degenerate case: zero throughput must report infinite cost, not divide-by-zero.
assert compute_cost_per_1m_tokens(gpu_cost_per_hour=20.0, output_tokens_per_second=0.0) == float("inf")


print("\nAll blocks executed successfully.")
