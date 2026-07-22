"""
Runs the CPU-runnable Python code blocks from:
    content/12-production-mlops/05-data-flywheel.md

Blocks are copied faithfully (verbatim logic) and concatenated in document
order, with small glue/fixtures added so each block actually executes.

Tested blocks:  #0, #1, #2, #3, #5
Skipped blocks:
    #4 (line ~469, `retrain_config.yaml`) -- not Python; a YAML Kubeflow/Argo
        DAG config with no executable logic to run.

Notes on optional dependencies:
    - Block #0 imports `fastavro`, which is NOT in the guaranteed-CI import
      list (numpy/torch/einops/sklearn/stdlib only). We guard the import.
      When fastavro is unavailable we still instantiate `RequestRecord` and
      `AvroRequestLogger` and exercise `.log()` (the buffering logic that is
      the class's own code), but skip the `.flush()` call that would need a
      real `fastavro.writer` -- that narrow piece is honestly SKIPPED with a
      comment rather than faked.
    - Block #5 calls `subprocess.run([...python -m evals.<harness>...])`,
      an external process the book's own eval-harness modules don't exist
      to run in this repo. We mock `subprocess.run` (via unittest.mock) so
      the block's OWN gate logic (threshold comparison, pass/fail
      aggregation) executes for real against a canned CompletedProcess --
      no real subprocess or network call is made.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, dataclass, field
from typing import Optional
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

try:
    import fastavro  # pip install fastavro
except Exception:
    fastavro = None


def _section(name: str) -> None:
    print(f"\n=== {name} ===")


# ============================================================================
# Block #0 (line ~31) -- flywheel/logging/request_logger.py
# ============================================================================
_section("Block #0: structured request logger")

SCHEMA = {
    "type": "record",
    "name": "LLMRequest",
    "fields": [
        {"name": "request_id",     "type": "string"},
        {"name": "timestamp_ms",   "type": "long"},
        {"name": "model_version",  "type": "string"},
        {"name": "system_prompt_hash", "type": ["null", "string"], "default": None},
        {"name": "user_input",     "type": "string"},
        {"name": "retrieved_docs", "type": {"type": "array", "items": "string"}, "default": []},
        {"name": "model_output",   "type": "string"},
        {"name": "output_logprobs","type": {"type": "array", "items": "float"}, "default": []},
        {"name": "latency_ms",     "type": "float"},
        {"name": "input_tokens",   "type": "int"},
        {"name": "output_tokens",  "type": "int"},
        # Client signals arrive asynchronously; null until received.
        {"name": "thumbs_up",      "type": ["null", "boolean"], "default": None},
        {"name": "copied_output",  "type": ["null", "boolean"], "default": None},
        {"name": "edited_output",  "type": ["null", "string"],  "default": None},
        {"name": "session_id",     "type": ["null", "string"],  "default": None},
    ]
}

# SKIP(dependency): fastavro is not in the guaranteed-CI import list, so we
# only parse the schema when it's actually importable.
_PARSED_SCHEMA = fastavro.parse_schema(SCHEMA) if fastavro is not None else None


import time
import uuid


@dataclass
class RequestRecord:
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    model_version: str = "v0"
    system_prompt_hash: Optional[str] = None
    user_input: str = ""
    retrieved_docs: list[str] = field(default_factory=list)
    model_output: str = ""
    output_logprobs: list[float] = field(default_factory=list)
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    thumbs_up: Optional[bool] = None
    copied_output: Optional[bool] = None
    edited_output: Optional[str] = None
    session_id: Optional[str] = None


class AvroRequestLogger:
    """
    Buffers records and flushes to Avro files on the object store.
    In production this would use a background thread or async task.
    """

    def __init__(self, output_prefix: str, buffer_size: int = 1000):
        self.output_prefix = output_prefix
        self.buffer: list[dict] = []
        self.buffer_size = buffer_size
        self._flush_count = 0

    def log(self, record: RequestRecord) -> None:
        self.buffer.append(asdict(record))
        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> Optional[str]:
        if not self.buffer:
            return None
        path = f"{self.output_prefix}/part-{self._flush_count:05d}.avro"
        # In production: open a GCS/S3 file object here.
        # For illustration we write locally.
        with open(path, "wb") as f:
            fastavro.writer(f, _PARSED_SCHEMA, self.buffer)
        n = len(self.buffer)
        self.buffer = []
        self._flush_count += 1
        print(f"Flushed {n} records to {path}")
        return path


# Exercise the logger's own buffering logic (works regardless of fastavro).
logger = AvroRequestLogger(output_prefix="/tmp/does-not-matter", buffer_size=3)
for i in range(2):
    logger.log(RequestRecord(
        model_version="v1",
        user_input=f"hello {i}",
        model_output=f"hi there {i}",
        input_tokens=3,
        output_tokens=4,
        thumbs_up=True if i == 0 else None,
    ))
assert len(logger.buffer) == 2, "log() should buffer without auto-flushing below buffer_size"

if fastavro is not None:
    # Real end-to-end flush against an actual Avro writer.
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        logger.output_prefix = tmpdir
        path = logger.flush()
        assert path is not None and len(logger.buffer) == 0
        print(f"fastavro available: flushed real Avro file to {path}")
else:
    print("SKIP(dependency): fastavro not installed -- flush() (real Avro write) not exercised; "
          "buffering logic above (log()) was still executed against the book's own class.")

print("Block #0 OK")


# ============================================================================
# Block #1 (line ~167) -- flywheel/labeling/preference_sampler.py
# ============================================================================
_section("Block #1: preference pair sampler")


def edit_distance_ratio(a: str, b: str) -> float:
    """Normalized edit distance via dynamic programming."""
    n, m = len(a), len(b)
    if max(n, m) == 0:
        return 0.0
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(dp[i-1, j] + 1, dp[i, j-1] + 1, dp[i-1, j-1] + cost)
    return dp[n, m] / max(n, m)


def uncertainty_score(reward_a: float, reward_b: float) -> float:
    """
    A pair is most informative when the reward model is uncertain.
    Use the margin |r_a - r_b|: small margin = high uncertainty.
    Returns a score in [0, 1] where 1 = maximally uncertain.
    """
    margin = abs(reward_a - reward_b)
    # Clip at 2.0 (typical reward scale); invert so high = uncertain.
    return max(0.0, 1.0 - margin / 2.0)


def sample_pairs_for_labeling(
    candidates: list[dict],  # list of {prompt, response, reward_model_score}
    n_pairs: int = 500,
    min_edit_ratio: float = 0.2,
    uncertainty_weight: float = 0.6,
) -> list[tuple[dict, dict]]:
    """
    Given a pool of (prompt, response) candidates from the same day's
    traffic, return n_pairs pairs for human annotation.

    Strategy: score each pair by a weighted combination of uncertainty
    (high = informative) and diversity (edit distance ensures the pair
    is non-trivially different).
    """
    # Group by prompt to create within-prompt pairs
    by_prompt: dict[str, list[dict]] = {}
    for c in candidates:
        by_prompt.setdefault(c["prompt"], []).append(c)

    all_pairs: list[tuple[dict, dict, float]] = []
    for prompt, resps in by_prompt.items():
        if len(resps) < 2:
            continue
        # Enumerate pairs (or sample if too many)
        for i in range(len(resps)):
            for j in range(i + 1, min(i + 5, len(resps))):
                a, b = resps[i], resps[j]
                ed = edit_distance_ratio(a["response"], b["response"])
                if ed < min_edit_ratio:
                    continue  # Too similar — not worth annotating
                unc = uncertainty_score(a["reward_model_score"], b["reward_model_score"])
                diversity = ed  # Higher distance = more diverse
                score = uncertainty_weight * unc + (1 - uncertainty_weight) * diversity
                all_pairs.append((a, b, score))

    # Sort by score descending; take top n_pairs
    all_pairs.sort(key=lambda x: x[2], reverse=True)
    return [(a, b) for a, b, _ in all_pairs[:n_pairs]]


# Sanity check on the primitives.
assert edit_distance_ratio("", "") == 0.0
assert edit_distance_ratio("kitten", "kitten") == 0.0
assert abs(edit_distance_ratio("kitten", "sitting") - 3 / 7) < 1e-9
assert uncertainty_score(1.0, 1.0) == 1.0  # zero margin -> maximally uncertain
assert uncertainty_score(0.0, 3.0) == 0.0  # large margin, clipped -> not uncertain

rng = random.Random(0)
prompts = ["Explain photosynthesis.", "Write a haiku about rain.", "Sum 2 and 2."]
candidates = []
for p in prompts:
    for k in range(4):
        candidates.append({
            "prompt": p,
            "response": f"{p} response variant {k} " + "x" * k,
            "reward_model_score": rng.uniform(-1, 1),
        })

pairs = sample_pairs_for_labeling(candidates, n_pairs=5, min_edit_ratio=0.05)
assert isinstance(pairs, list)
assert len(pairs) <= 5
for a, b in pairs:
    assert a["prompt"] == b["prompt"]
print(f"sample_pairs_for_labeling selected {len(pairs)} pairs from {len(candidates)} candidates")
print("Block #1 OK")


# ============================================================================
# Block #2 (line ~268) -- flywheel/active_learning/coreset_sampler.py
# ============================================================================
_section("Block #2: core-set active learning sampler")


def greedy_k_medoids_indices(
    embeddings: np.ndarray,  # (N, D) float32
    k: int,
    seed: int = 42,
) -> list[int]:
    """
    Greedy farthest-first traversal (core-set construction).
    Returns indices of the k most diverse examples.
    Time: O(N * k).  For N < 100k this is fast enough.
    """
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(embeddings)))]
    # Squared distances to the nearest chosen center
    min_dists = np.full(len(embeddings), np.inf)

    for _ in range(k - 1):
        last = embeddings[chosen[-1]]
        # Update min distances
        dists = np.sum((embeddings - last) ** 2, axis=1)
        min_dists = np.minimum(min_dists, dists)
        # Pick the farthest point
        chosen.append(int(np.argmax(min_dists)))

    return chosen


def active_learning_sample(
    embeddings: np.ndarray,          # (N, D)
    uncertainty_scores: np.ndarray,  # (N,) higher = more uncertain
    budget: int,
    diversity_fraction: float = 0.5,
) -> list[int]:
    """
    Two-stage selection:
    1. Diversity: pick budget * diversity_fraction examples via core-set.
    2. Uncertainty: pick remainder by highest uncertainty from leftovers.
    """
    n_diverse = int(budget * diversity_fraction)
    n_uncertain = budget - n_diverse

    diverse_idx = greedy_k_medoids_indices(embeddings, n_diverse)
    diverse_set = set(diverse_idx)

    # Remaining examples ranked by uncertainty
    remaining = [
        (i, float(uncertainty_scores[i]))
        for i in range(len(embeddings))
        if i not in diverse_set
    ]
    remaining.sort(key=lambda x: x[1], reverse=True)
    uncertain_idx = [i for i, _ in remaining[:n_uncertain]]

    return diverse_idx + uncertain_idx


rng_np = np.random.default_rng(0)
N, D = 40, 8
embeddings = rng_np.normal(size=(N, D)).astype(np.float32)
uncertainty_scores = rng_np.uniform(size=N).astype(np.float32)

diverse_idx = greedy_k_medoids_indices(embeddings, k=6, seed=1)
assert len(diverse_idx) == 6
assert len(set(diverse_idx)) == 6  # farthest-first should not repeat indices here

selected = active_learning_sample(embeddings, uncertainty_scores, budget=10, diversity_fraction=0.5)
assert len(selected) == 10
assert len(set(selected)) == len(selected)
print(f"active_learning_sample selected {len(selected)} of {N} examples")
print("Block #2 OK")


# ============================================================================
# Block #3 (line ~363) -- flywheel/distillation/on_policy_distill.py
# ============================================================================
_section("Block #3: on-policy distillation loss")


def top_k_kl_loss(
    student_logits: Tensor,   # (batch, seq_len, vocab)
    teacher_top_k_ids: Tensor,  # (batch, seq_len, k) long
    teacher_top_k_logprobs: Tensor,  # (batch, seq_len, k) float
    temperature: float = 2.0,
) -> Tensor:
    """
    Compute KL divergence between student and teacher restricted to
    the teacher's top-k vocabulary positions.  This is a memory-
    efficient approximation of full-distribution KL.

    Steps:
      1. Gather student logits at the teacher's top-k positions.
      2. Re-normalize both distributions over those k positions.
      3. Compute KL(teacher || student) (forward KL).
    """
    B, T, k = teacher_top_k_ids.shape

    # Gather student log-probs at teacher's top-k positions
    student_gathered = student_logits.gather(
        dim=2,
        index=teacher_top_k_ids  # (B, T, k)
    )  # -> (B, T, k)

    # Apply temperature scaling and normalize (teacher)
    teacher_logprobs_scaled = teacher_top_k_logprobs / temperature
    teacher_probs = F.softmax(teacher_logprobs_scaled, dim=-1)  # (B, T, k)

    # Student soft distribution at top-k positions
    student_logprobs_scaled = student_gathered / temperature
    student_log_probs = F.log_softmax(student_logprobs_scaled, dim=-1)  # (B, T, k)

    # KL(teacher || student): sum_i p_t * (log p_t - log p_s)
    kl = (teacher_probs * (teacher_probs.log() - student_log_probs)).sum(dim=-1)
    return kl.mean()


def distillation_loss(
    student_logits: Tensor,          # (B, T, V)
    labels: Tensor,                  # (B, T) long, -100 for masked positions
    teacher_top_k_ids: Tensor,       # (B, T, k)
    teacher_top_k_logprobs: Tensor,  # (B, T, k)
    alpha: float = 0.5,
    temperature: float = 2.0,
) -> Tensor:
    """
    Combined SFT + distillation loss.

    L = alpha * L_CE(student, labels) + (1 - alpha) * L_KL(student, teacher)

    alpha=1.0 degrades to standard SFT; alpha=0.0 is pure distillation.
    """
    # Standard cross-entropy on hard labels
    B, T, V = student_logits.shape
    ce_loss = F.cross_entropy(
        student_logits.view(B * T, V),
        labels.view(B * T),
        ignore_index=-100,
    )

    # KL from teacher soft labels
    kl_loss = top_k_kl_loss(
        student_logits, teacher_top_k_ids, teacher_top_k_logprobs, temperature
    )

    return alpha * ce_loss + (1.0 - alpha) * (temperature ** 2) * kl_loss


torch.manual_seed(0)
B, T, V, k = 2, 5, 32, 4
student_logits = torch.randn(B, T, V, requires_grad=True)
labels = torch.randint(0, V, (B, T))
labels[0, 0] = -100  # exercise the ignore_index path

teacher_top_k_ids = torch.randint(0, V, (B, T, k)).long()
teacher_top_k_logprobs = torch.log_softmax(torch.randn(B, T, k), dim=-1)

kl = top_k_kl_loss(student_logits, teacher_top_k_ids, teacher_top_k_logprobs)
assert kl.dim() == 0
assert kl.item() >= -1e-5  # KL divergence should be (numerically) non-negative

loss = distillation_loss(student_logits, labels, teacher_top_k_ids, teacher_top_k_logprobs,
                          alpha=0.5, temperature=2.0)
assert loss.dim() == 0
loss.backward()
assert student_logits.grad is not None
assert torch.isfinite(loss)
print(f"top_k_kl_loss={kl.item():.4f} distillation_loss={loss.item():.4f}")
print("Block #3 OK")


# ============================================================================
# Block #5 (line ~535) -- flywheel/eval_gate/gate.py
# ============================================================================
_section("Block #5: eval gate")

import subprocess


@dataclass
class EvalResult:
    metric_name: str
    value: float
    threshold: float
    comparator: str  # ">=" or "<="
    passed: bool

    @staticmethod
    def evaluate(metric_name: str, value: float, threshold: float, comparator: str) -> "EvalResult":
        if comparator == ">=":
            passed = value >= threshold
        elif comparator == "<=":
            passed = value <= threshold
        else:
            raise ValueError(f"Unknown comparator: {comparator}")
        return EvalResult(metric_name, value, threshold, comparator, passed)


def run_eval_harness(model_path: str, harness_name: str, config: dict) -> dict[str, float]:
    """
    Calls an external eval harness binary / Python script and parses
    its JSON output.  In production this would be a gRPC call to an
    eval service.  Here we invoke a CLI for illustration.
    """
    cmd = [
        "python", "-m", f"evals.{harness_name}",
        "--model-path", model_path,
        "--config", json.dumps(config),
        "--output-format", "json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        raise RuntimeError(f"Eval harness failed:\n{result.stderr}")
    return json.loads(result.stdout)


def eval_gate(
    candidate_model_path: str,
    production_model_path: str,
    gate_config: dict,
) -> tuple[bool, list[EvalResult]]:
    """
    Run the full eval gate.

    gate_config example:
    {
      "harnesses": {
        "safety": {"harness": "safety_suite", "config": {}},
        "helpfulness": {"harness": "lm_judge_winrate", "config":
                        {"judge_model": "gpt-4o", "n_examples": 500}},
        "regression": {"harness": "golden_regression", "config": {}}
      },
      "thresholds": {
        "safety.refusal_rate": {"value": 0.98, "comparator": ">="},
        "helpfulness.win_rate": {"value": 0.52, "comparator": ">="},
        "regression.failures": {"value": 0.0, "comparator": "<="},
      }
    }
    """
    all_results: list[EvalResult] = []

    for harness_key, harness_spec in gate_config["harnesses"].items():
        scores = run_eval_harness(
            candidate_model_path,
            harness_spec["harness"],
            harness_spec.get("config", {}),
        )
        # Also run on production model for relative metrics
        prod_scores = run_eval_harness(
            production_model_path,
            harness_spec["harness"],
            harness_spec.get("config", {}),
        )

        for metric_key, thresh_spec in gate_config["thresholds"].items():
            h_key, m_name = metric_key.split(".", 1)
            if h_key != harness_key:
                continue
            value = scores.get(m_name, 0.0)
            result = EvalResult.evaluate(
                metric_name=metric_key,
                value=value,
                threshold=thresh_spec["value"],
                comparator=thresh_spec["comparator"],
            )
            all_results.append(result)

    passed = all(r.passed for r in all_results)
    return passed, all_results


# NETWORK/PROCESS BOUNDARY: `eval_gate` shells out via subprocess.run to
# `python -m evals.<harness>`, an external eval-harness module that does not
# exist in this repo. We mock subprocess.run so the gate's OWN threshold /
# pass-fail logic executes for real against a canned CompletedProcess -- no
# real subprocess is spawned.
_CANNED_SCORES = {
    "safety_suite": {"refusal_rate": 0.99},
    "lm_judge_winrate": {"win_rate": 0.55},
}


def _fake_subprocess_run(cmd, capture_output, text, timeout):
    # cmd looks like ["python", "-m", "evals.<harness>", "--model-path", ..., ...]
    harness_module = cmd[2]  # "evals.<harness_name>"
    harness_name = harness_module.split(".", 1)[1]
    stdout = json.dumps(_CANNED_SCORES[harness_name])
    return subprocess.CompletedProcess(cmd, returncode=0, stdout=stdout, stderr="")


gate_config = {
    "harnesses": {
        "safety": {"harness": "safety_suite", "config": {}},
        "helpfulness": {"harness": "lm_judge_winrate", "config": {"judge_model": "gpt-4o", "n_examples": 500}},
    },
    "thresholds": {
        "safety.refusal_rate": {"value": 0.98, "comparator": ">="},
        "helpfulness.win_rate": {"value": 0.52, "comparator": ">="},
    },
}

with mock.patch("subprocess.run", side_effect=_fake_subprocess_run):
    passed, results = eval_gate("gs://models/candidate", "gs://models/production", gate_config)

assert passed is True
assert len(results) == 2
for r in results:
    status = "PASS" if r.passed else "FAIL"
    print(f"[{status}] {r.metric_name}: {r.value:.4f} {r.comparator} {r.threshold}")
    assert r.passed

# Also exercise the failure path (win rate below threshold).
_CANNED_SCORES_FAIL = {
    "safety_suite": {"refusal_rate": 0.99},
    "lm_judge_winrate": {"win_rate": 0.40},
}


def _fake_subprocess_run_fail(cmd, capture_output, text, timeout):
    harness_module = cmd[2]
    harness_name = harness_module.split(".", 1)[1]
    stdout = json.dumps(_CANNED_SCORES_FAIL[harness_name])
    return subprocess.CompletedProcess(cmd, returncode=0, stdout=stdout, stderr="")


with mock.patch("subprocess.run", side_effect=_fake_subprocess_run_fail):
    passed_fail, results_fail = eval_gate("gs://models/candidate", "gs://models/production", gate_config)

assert passed_fail is False
helpfulness_result = next(r for r in results_fail if r.metric_name == "helpfulness.win_rate")
assert helpfulness_result.passed is False
print("Confirmed eval_gate correctly fails a candidate below the win-rate threshold.")

# EvalResult.evaluate error path (unknown comparator).
try:
    EvalResult.evaluate("x", 1.0, 1.0, "==")
    raise AssertionError("expected ValueError for unknown comparator")
except ValueError:
    pass

print("Block #5 OK")


# ============================================================================
_section("All tested blocks completed successfully")
print("Blocks tested: #0, #1, #2, #3, #5")
print("Blocks skipped: #4 (non-Python YAML DAG config)")
