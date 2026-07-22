# 12.5 Data Flywheels & Continuous Improvement

Every LLM product faces the same tension: you ship a model trained on yesterday's data into a world that keeps changing, with users who keep finding new failure modes. The teams that win long-term are not the ones who launch the best model on day one — they are the ones who have built machinery to observe failures, collect signal, retrain, and redeploy faster than their competitors. This machinery is called the **data flywheel**.

A data flywheel is a self-reinforcing loop. Better data produces a better model, which earns more users, which generates more interaction data, which feeds the next improvement. The compounding effect is not linear: each trip around the loop tends to reveal sharper edge cases and produces denser training signal than the last. This chapter walks through every component of that loop in detail — logging design, labeling pipelines, preference collection, active learning, distillation from production traffic, and eval-gated deployment — and shows how to wire them together into a live, self-improving system.

This chapter assumes you are already familiar with [Supervised Fine-Tuning & Instruction Tuning](../05-posttraining-alignment/01-sft-instruction-tuning.html), [The RLHF Pipeline & Reward Modeling](../05-posttraining-alignment/05-rlhf-reward-modeling.html), and [Observability, Logging & LLMOps](../12-production-mlops/02-observability-llmops.html). It also builds on evaluation concepts covered in [The Evaluation Problem & Benchmark Landscape](../11-evaluation/01-eval-landscape.html) and [Building Eval Harnesses](../11-evaluation/03-eval-harnesses.html).

## The Anatomy of a Data Flywheel

Before diving into individual components, let us establish the whole picture. A production data flywheel has six stages that cycle continuously:


{{fig:flywheel-six-stage-loop}}


Each stage has concrete engineering requirements. We will trace through them in order, then discuss the math underlying the value of compounding data.

### Why the flywheel is a moat

For a new entrant competing against a mature product, the challenge is not the model itself — open-weight base models make that accessible. The challenge is the data advantage. After $k$ rounds of the flywheel, a product has collected approximately $N_0 \cdot r^k$ training examples (where $r > 1$ is the per-round growth factor from a growing user base). More important than volume is *distribution shift*: after many rounds, a well-run flywheel's training set covers the long tail of real user behaviors in a way no static dataset can match. This is the moat.

## Structured Logging as the Foundation

No flywheel works without high-quality logs. Logs are not just debugging artifacts — they are raw material. Every request must produce a logged record that captures enough context to later train or evaluate a model.

### What to log

A minimal request record contains: request ID, timestamp, model version, raw user input, any retrieved context, the full model output, latency, token counts, and any client-side signals received (thumbs up/down, copy events, follow-up edits, session end). A richer record adds: the sampled probabilities of the chosen tokens (for distillation and importance weighting), the intermediate chain-of-thought if visible, and the system prompt hash.

```python
# flywheel/logging/request_logger.py
"""
Structured request logger that writes Avro records to an append-only
object store (e.g., GCS or S3).  Every field is typed to support
schema evolution without breaking downstream readers.
"""

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Optional

import fastavro  # pip install fastavro


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

_PARSED_SCHEMA = fastavro.parse_schema(SCHEMA)


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
```

### Joining async signals back to requests

Client signals (thumbs up, copy, regeneration requests) arrive seconds to minutes after the original request. You need a join service that updates the immutable log record. The cleanest approach is an event stream (Kafka or Pub/Sub): the serving tier emits request events; the client browser emits signal events keyed on request ID; a Flink or Spark Streaming job performs a session-windowed join and writes the enriched record to a curated table.


{{fig:flywheel-async-signal-join}}


Unenriched records (no explicit signal) are not useless — implicit signals such as session continuation, regeneration rate, and downstream edit distance are powerful weak-supervision labels (see Section 4 below).

## Preference Collection in Production

Human preference is the highest-quality signal for alignment, but it is expensive and sparse. Production systems use three strategies to maximize its value.

### Explicit feedback collection

Show users a simple thumbs-up / thumbs-down control. Even at a 3–5% click-through rate on a busy product, you may collect tens of thousands of labeled comparisons per day. Design rules:

1. **Show the comparison, not just a rating.** Present side-by-side outputs from two model variants when running an A/B test. Collect `(prompt, output_A, output_B, preference)` tuples directly.
2. **Attach a free-text reason.** A text field after a thumbs-down increases the signal-to-noise ratio and feeds into category analysis.
3. **Log what you did not show.** You need the counterfactual (what the other variant would have said) for offline reward model training — this requires logging the outputs of both models on every request even when only one is shown.

### Structured preference from downstream behavior

Production provides implicit preference signals that require no user action:

| Signal | Proxy for |
|---|---|
| User edits the response | Output was close but wrong |
| User regenerates | Output was clearly bad |
| User copies the output | Output was good (often strong positive) |
| User continues the conversation | Output was acceptable |
| User abandons the session immediately | Output may have been very bad |

These signals are noisy individually but highly correlated in aggregate. You can train a lightweight classifier to predict explicit thumbs-up from implicit signals, then use this **proxy reward model** to label the remaining 95% of unlabeled traffic.

{{fig:preference-signal-pyramid}}

### Pairwise preference labeling at scale

For offline reward model training (see [The RLHF Pipeline & Reward Modeling](../05-posttraining-alignment/05-rlhf-reward-modeling.html)), you need explicit pairwise preference labels. A practical pipeline:

```python
# flywheel/labeling/preference_sampler.py
"""
Sample pairs of responses to the same prompt for human preference
annotation.  Pairs are selected so that:
  - The responses differ meaningfully (filter near-identical pairs)
  - The prompt distribution is diverse (cluster-stratified sampling)
  - Hard cases are prioritized (reward model uncertainty sampling)
"""

import random
from typing import Optional
import numpy as np


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
```

## Active Learning: Which Examples to Label Next?

Not all unlabeled examples are equally informative. Active learning selects the subset of production traffic where labeling effort will produce the largest model improvement.

### Uncertainty sampling

The simplest strategy: label examples where the current model is most uncertain. For a language model, uncertainty is hard to compute exactly, but good proxies exist:

$$
H(\text{output}) \approx -\frac{1}{T} \sum_{t=1}^{T} \log p_\theta(y_t \mid y_{<t}, x)
$$

This is just the average negative log-probability per token, i.e. the per-token cross-entropy loss. High entropy outputs (low average logprob) correspond to cases where the model was uncertain. You are already computing this during inference; storing it costs a single float per request.

### Core-set / diversity sampling

Uncertainty sampling alone leads to annotation of many near-duplicate examples (the model is uncertain in a cluster around the same concept). Add a diversity constraint: after computing uncertainty scores, run k-medoids clustering on the prompt embeddings, then sample the highest-uncertainty example from each cluster.

{{fig:active-learning-uncertainty-vs-diversity}}

```python
# flywheel/active_learning/coreset_sampler.py
"""
Core-set active learning: select a diverse and uncertain subset of
production examples for annotation.  Uses prompt embeddings from a
small frozen encoder (e.g., a 100M embedding model).
"""

from typing import Optional
import numpy as np


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
```

### Hard-negative mining

A special case of active learning: examples where the model confidently produced an incorrect answer. For tasks with verifiable ground truth (math, code execution, structured extraction), you can automatically identify hard negatives using a test oracle and route them directly to the training set without human review.

!!! example "Worked example: budget allocation"
    Suppose you have 10,000 unlabeled examples from one day's traffic and a budget of 500 human labels.

    - Your reward model gives average per-token logprob scores; you compute the bottom 2,000 by score (most uncertain).
    - You embed all 2,000 with a 100M sentence encoder (takes ~30 seconds on a single A100).
    - Core-set sampling selects 250 diverse examples from this uncertain pool.
    - An additional 250 are selected from hard negatives: code examples where the generated code failed the unit tests (you run the code in a sandbox for every coding request).
    - Total: 500 labeled examples. At USD 0.10 per label (HITL vendors), cost is USD 50.

    After one week of this process at 500 labels/day, you have 3,500 high-quality examples. Fine-tuning on these (in addition to the base SFT dataset) typically improves reward model Spearman correlation by on the order of 3–8 percentage points — the exact gain depends on task difficulty and the quality of the base RM.

## Distillation from Production Traffic

Beyond labeling for reward models and SFT, you can use production traffic to distill the model's own knowledge into a smaller or faster version. See also [Distillation, Model Compression & Knowledge Transfer](../05-posttraining-alignment/12-distillation-compression.html) for the full distillation picture.

### Sequence-level knowledge distillation

Classic KD (Hinton et al.) trains a student on the teacher's soft probability distribution over tokens. For LLMs at scale you cannot store the full vocabulary distribution for every token of every production request. Two practical alternatives:

1. **Top-k logit storage.** Log the top-32 token IDs and their logprobs for each output position. This is ~5x more data than the text alone but gives a useful soft target.
2. **Speculative pseudo-labels.** Run the teacher model on the sampled output and record whether it would have chosen the same token. Use this agreement signal as a binary label for on-policy distillation.

### On-policy distillation pipeline

```python
# flywheel/distillation/on_policy_distill.py
"""
On-policy distillation: the teacher model generates responses to
production prompts; we train the student to match the teacher's
distribution using a combination of cross-entropy on the text and
KL divergence on the top-k logits.

This is a simplified illustration; in production you would use a
proper distributed training harness (e.g., TRL's SFT trainer).
"""

import torch
import torch.nn.functional as F
from torch import Tensor


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
```

The $T^2$ factor in the combined loss corrects for the fact that temperature scaling reduces gradient magnitudes (it was proved in the original Hinton et al., 2015 paper and is easy to derive: if logits are divided by $T$, the softmax output is flatter, reducing entropy by $T^2$ in expectation for Gaussian logits).

### Data flywheel for cheaper inference

A compelling use of distillation from production traffic is to progressively compress the serving model. As the product matures and the training set grows, you retrain a student model that has fewer parameters but covers the distribution well — because it was trained specifically on the traffic distribution your users actually produce. This compounds with quantization (see [Quantization I](../04-kernels-efficiency/07-quantization-ptq.html)) to reduce inference cost over time while maintaining quality.

## Retraining Pipelines

Retraining is not fine-tuning a model once. It is an automated pipeline triggered on a schedule or by a data threshold, with reproducibility as a first-class requirement.

### The retraining recipe

A typical LLM product retraining loop runs SFT followed by a preference optimization step (DPO, RLHF-PPO, or GRPO — see [Direct Preference Optimization & Its Variants](../05-posttraining-alignment/07-dpo-and-variants.html) and [GRPO, RLOO & Critic-Free RL](../05-posttraining-alignment/08-grpo-rloo.html) for the trade-offs). The new training data is mixed with a frozen replay buffer of earlier data at a ratio that prevents catastrophic forgetting:

$$
\mathcal{D}_{\text{train}} = (1 - \rho) \cdot \mathcal{D}_{\text{new}} \cup \rho \cdot \mathcal{D}_{\text{replay}}
$$

Typical values of the replay fraction $\rho$ are 0.3–0.5. Too low and the model forgets earlier capabilities; too high and the new signal is diluted.

```yaml
# flywheel/pipelines/retrain_config.yaml
#
# Example Kubeflow / Argo Workflows DAG configuration
# for a weekly retraining run.

retraining_job:
  trigger:
    schedule: "0 2 * * 1"          # Every Monday at 02:00 UTC
    data_threshold_new_examples: 5000  # Also trigger if this many new labels accumulated

  data_assembly:
    new_sft_data:
      source: "gs://my-logs/labeled/sft/"
      date_window_days: 7
    new_preference_data:
      source: "gs://my-logs/labeled/preferences/"
      date_window_days: 7
    replay_buffer:
      source: "gs://my-data/replay-buffer-v3/"
      sample_fraction: 0.40          # rho = 0.40

  sft_step:
    base_model: "gs://my-models/checkpoint-stable"  # Pinned stable base
    epochs: 1
    learning_rate: 2.0e-5
    batch_size: 128
    peft: lora                       # LoRA to keep training cheap
    lora_rank: 64
    output: "gs://my-models/sft-candidate/"

  preference_step:
    algorithm: dpo                   # or: ppo, grpo
    beta: 0.1                        # KL regularization strength
    base_model: "sft-candidate"
    output: "gs://my-models/dpo-candidate/"

  eval_gate:
    harness: "internal-eval-v2"      # See eval-gated deployment section
    pass_thresholds:
      safety_refusal_rate: ">= 0.98"
      helpfulness_win_rate: ">= 0.52"  # vs. production model in A/B judge
      regression_suite: "0 regressions on all priority-1 cases"
    on_failure: alert_slack_and_halt
    on_pass: deploy_to_canary_10pct
```

### Preventing catastrophic forgetting

The biggest practical failure mode is a new model that is better on the new data but worse on some existing capability. Three defenses:

1. **Replay buffers.** Mix old data in at ratio $\rho$ as above.
2. **EWC-style regularization.** Elastic Weight Consolidation adds a penalty proportional to the Fisher information of the old task. In practice, a simpler proxy — adding a KL divergence penalty relative to the frozen previous checkpoint — is more common for LLMs (this is essentially the PPO KL term applied during SFT).
3. **Regression test suite.** A hardcoded set of golden examples that the new model must answer identically to the old model (or better). Any regression on these blocks the deployment.

## Eval-Gated Deployment

A new model should never go to production without passing an automated evaluation gate. The gate is a decision function:

$$
\text{deploy}(v_{\text{new}}) = \begin{cases} \text{yes} & \text{if } \forall k: \text{score}_k(v_{\text{new}}) \geq \tau_k \text{ and no regression} \\ \text{no} & \text{otherwise} \end{cases}
$$

where $k$ indexes the set of evaluation dimensions and $\tau_k$ is the minimum acceptable score on dimension $k$.

### Building the eval gate

```python
# flywheel/eval_gate/gate.py
"""
Eval gate: runs a model candidate through a suite of evals and returns
a pass/fail verdict.  Designed to be called from a CI/CD pipeline
(Argo Workflow, GitHub Actions, etc.).

See also: Part XI (Evaluation) for how to build the eval harnesses
that this gate calls.
"""

import json
import sys
from dataclasses import dataclass
from typing import Optional
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--production", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    passed, results = eval_gate(args.candidate, args.production, config)

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.metric_name}: {r.value:.4f} {r.comparator} {r.threshold}")

    if not passed:
        print("\nEval gate FAILED. Blocking deployment.")
        sys.exit(1)
    else:
        print("\nEval gate PASSED. Proceeding to canary deployment.")
        sys.exit(0)
```

### The win-rate judge

The most commonly used gate metric for open-ended generation quality is the **win rate against production**: an LLM judge (often GPT-4o or an internal judge model) evaluates 500–1,000 prompt/response pairs and decides which of candidate vs. production is better. A win rate $\geq 0.52$ with statistical significance is a typical deployment gate. This approach is described in detail in [LLM-as-a-Judge & Automated Evaluation](../11-evaluation/02-llm-as-judge.html).

!!! warning "Common pitfall: eval distribution mismatch"
    If your eval suite is assembled once and never updated, it will diverge from the distribution of real user traffic over time. The model can overfit to the eval suite — passing all gates while regressing on real users (a form of Goodhart's Law). Fix this by *adding new golden examples from production traffic to the regression suite* on every release, and rotating out examples that the model has easily solved for multiple consecutive releases.

## The Compounding Data Advantage

Let us formalize the compounding effect. Suppose:
- At round $k$, the product has $N_k$ users generating $D_k$ training examples.
- Model quality at round $k$ is $Q_k$, and better quality attracts more users: $N_{k+1} = N_k \cdot (1 + \alpha \cdot \Delta Q_k)$.
- Each new user generates $d$ examples per unit time, so $D_{k+1} = D_k + d \cdot N_{k+1}$.
- Model quality improves with data: $Q_{k+1} = Q_k + \beta \cdot \log(D_{k+1} / D_k)$ (log-linear in data, consistent with scaling law intuitions from [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html)).

Even with modest values ($\alpha = 0.1$, $\beta = 0.05$, $d = 10$), after 10 rounds:

$$
\frac{N_{10}}{N_0} \approx \prod_{k=0}^{9} (1 + \alpha \cdot \Delta Q_k)
$$

The product is super-linear in the number of rounds because $\Delta Q_k$ is driven by $\log$ data growth, but user growth is multiplicative. A new entrant starting at round 10 with the same initial model but no training data faces a gap that is essentially impossible to close through model architecture improvements alone.

{{fig:compounding-flywheel-moat}}

This analysis also explains why *data quality* matters more than *data volume* past a certain scale. Once you have covered the main distribution, the marginal value of a new random example is small; the marginal value of a hard example from the tail of the distribution remains high throughout.

## Putting It All Together: A Self-Improving Product

Here is a complete end-to-end picture of how the components described above wire together in a real production system. The key insight is that this is not a sequential pipeline — it is a continuously running set of services that all interact.


{{fig:flywheel-system-architecture}}


### Operational cadence

| Cadence | Action |
|---|---|
| Real-time | Serve requests, log records, emit client signal events |
| Minutes | Async signal join, proxy reward scoring |
| Hours | Active learning sampling, annotation queue refresh |
| Days | Human annotation batch completes |
| Weekly | Retrain DAG triggers, eval gate runs, canary deploy |
| Monthly | Replay buffer refresh, annotation taxonomy review |

The weekly retraining cycle is a practical baseline. Teams with very high traffic (millions of requests/day) and strong automation can run daily cycles; teams with sparse human annotation budgets may run monthly.

!!! interview "Interview Corner"
    **Q:** You are designing the ML system for a production coding assistant used by 500,000 developers. How would you build a data flywheel to continuously improve it?

    **A:** I would structure it as a six-stage loop. First, **structured logging**: every request logs the prompt, output, per-token logprobs, and a session ID so client signals can be joined later. Second, **preference collection**: the IDE plugin emits implicit signals (code accepted/rejected, tests passing/failing after accepting, immediate edits) which a proxy reward model uses to score the 95% of traffic with no explicit feedback. Third, **active learning**: daily, I run a core-set sampler over the preceding day's uncertain requests (low logprob, high proxy-reward variance) and hard negatives (accepted code that failed CI) to fill a 500-example annotation quota routed to domain-expert annotators. Fourth, **retraining**: weekly SFT + DPO run on the new data mixed with a 40% replay buffer, using LoRA to keep compute tractable. Fifth, **eval gate**: the candidate must beat production in a GPT-4o win-rate judge on 1,000 held-out examples, pass a regression suite of 200 priority-1 golden cases, and show no safety regressions. Sixth, **canary deploy** at 10% traffic for 48 hours, monitoring regression on live task-completion rate before full rollout. The compounding effect comes from the fact that each week's better model attracts more usage, which provides higher-quality signal for the next week.

!!! sota "State of the Art & Resources (2026)"
    Data flywheels have matured from research concept to core production practice: major labs now run weekly or faster retraining loops, closed-loop preference collection, and automated eval gates as standard infrastructure. The field has converged on mixing active data selection, weak supervision from implicit signals, and replay-buffered SFT+DPO cycles as the workhorse recipe.

    **Foundational work**

    - [Sorscher et al., *Beyond Neural Scaling Laws: Beating Power Law Scaling via Data Pruning* (NeurIPS 2022)](https://arxiv.org/abs/2206.14486) — NeurIPS Outstanding Paper; proves that intelligent data pruning can shift error from power-law to exponential decay, establishing the theoretical case for quality over quantity.
    - [Hinton, Vinyals & Dean, *Distilling the Knowledge in a Neural Network* (2015)](https://arxiv.org/abs/1503.02531) — the temperature-scaled soft-label KD paper cited throughout this chapter; the $T^2$ loss correction derives from here.

    **Recent advances (2023–2026)**

    - [Luo et al., *Arena Learning: Build Data Flywheel for LLMs Post-training via Simulated Chatbot Arena* (2024)](https://arxiv.org/abs/2407.10627) — replaces expensive human arena battles with AI-judged simulated competitions to drive iterative SFT+RL improvement; introduces WizardArena for offline Elo estimation.
    - [Zhao et al., *Agent-in-the-Loop: A Data Flywheel for Continuous Improvement in LLM-based Customer Support* (2025)](https://arxiv.org/abs/2510.06674) — production case study showing four annotation types (preference, explanation, relevance, gap) fed back into weekly retraining, cutting cycles from months to weeks with +8.4% helpfulness.
    - [Nie et al., *CharacterFlywheel: Scaling Iterative Improvement of Engaging and Steerable LLMs in Production* (2026)](https://arxiv.org/abs/2603.01973) — Meta's 15-generation flywheel over real user traffic, integrating reward modeling, SFT, and RL; instruction-following accuracy rose from 59% to 85% over the run.
    - [Xia et al., *LESS: Selecting Influential Data for Targeted Instruction Tuning* (ICML 2024)](https://arxiv.org/abs/2402.04333) — gradient-similarity-based data selection; training on a LESS-selected 5% of data often outperforms training on the full set, making active learning tractable at scale.
    - [Ankner et al., *Perplexed by Perplexity: Perplexity-Based Data Pruning With Small Reference Models* (2024)](https://arxiv.org/abs/2405.20541) — a 125M proxy model scoring perplexity on training candidates improves a 3B model by up to 2 points on downstream tasks; practical guidance for flywheel data-quality filtering.
    - [Liu et al., *Online Speculative Decoding* (ICML 2024)](https://arxiv.org/abs/2310.07177) — continuously distills the production target model into the draft model using live query traffic, improving token acceptance by 10–65% with cost-neutral retraining on idle serving capacity.

    **Open-source & tools**

    - [princeton-nlp/LESS](https://github.com/princeton-nlp/LESS) — official ICML 2024 implementation of gradient-based influential-data selection for instruction tuning, with scripts for warmup, gradient collection, and LoRA fine-tuning.
    - [opendilab/awesome-RLHF](https://github.com/opendilab/awesome-RLHF) — continuously updated catalogue of RLHF papers (2020–2026), codebases, datasets, and blog posts; the best single index for tracking the preference-learning literature.

    **Go deeper**

    - [Nathan Lambert, *RLHF Book: Reinforcement Learning from Human Feedback and LLM Post-Training* (2026)](https://rlhfbook.com/) — comprehensive free book covering instruction tuning, reward modeling, rejection sampling, DPO, and online RLHF; the clearest end-to-end reference for the full preference-learning pipeline that drives every flywheel.

## Further Reading

- Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT), arXiv 2022 — the foundational RLHF-from-production-feedback paper.
- Ziegler et al., "Fine-Tuning Language Models from Human Preferences," arXiv 2019 — first demonstration of reward modeling from human preference labels.
- Settles, "Active Learning Literature Survey," University of Wisconsin, 2010 — comprehensive reference on uncertainty sampling, query by committee, and core-set methods.
- Hinton, Vinyals, and Dean, "Distilling the Knowledge in a Neural Network," NIPS Deep Learning Workshop 2015 — the temperature-scaled soft-label distillation paper.
- Kim and Rush, "Sequence-Level Knowledge Distillation," EMNLP 2016 — adapts KD to sequence-to-sequence models; the on-policy variant is widely used for LLM compression.
- Sorscher et al., "Beyond Neural Scaling Laws: Beating Power Law Scaling via Data Pruning," NeurIPS 2022 — argues that intelligent data selection can beat scaling on a fixed compute budget.
- Ankner et al., "Perplexed by Perplexity: Perplexity-Based Data Pruning With Small Reference Models," arXiv 2024 — practical guidance on using small proxy models to filter training data quality.

!!! key "Key Takeaways"
    - The data flywheel is a compounding advantage: better model → more users → more signal → better model. After enough rounds, the training distribution gap is larger than any architectural advantage a new entrant can claim.
    - Every logged request is raw material. Design schemas for schema evolution (Avro/Protobuf), join client signals asynchronously, and store per-token logprobs even if you do not use them immediately.
    - Explicit preference labels are expensive and sparse; proxy reward models trained on implicit signals (copy, edit, session continuation) can extend coverage to 100% of traffic.
    - Active learning with core-set diversity sampling is 3–5x more label-efficient than random sampling — you get coverage of the hard tail without annotation redundancy on easy clusters.
    - Distillation from production traffic with top-k logit storage lets you continuously compress the serving model, reducing inference cost while maintaining quality on the actual user distribution.
    - Replay buffers at $\rho \approx 0.3$–$0.5$ are the primary defense against catastrophic forgetting during weekly retraining cycles.
    - The eval gate is not optional: win-rate vs. production, a regression suite, and safety checks must all pass before any deployment, however small. Without this gate, the flywheel degrades via Goodhart's Law — the model optimizes for the training distribution rather than genuine quality.
    - Data quality beats data volume past a certain scale. Hard negatives, diverse examples from the long tail, and preference labels on uncertain cases have far higher marginal value than additional random samples of easy cases.

## Exercises

**1.** (Conceptual) The chapter's third design rule for explicit feedback collection is "Log what you did not show" — during an A/B test between two model variants, log the output of *both* variants on every request even though the user only ever sees one. Explain why this is required for offline reward model training, and what specifically breaks if you only log the output that was actually served.

??? note "Solution"
    Offline reward model (RM) training consumes *pairwise* comparisons of the form $(\text{prompt}, \text{output}_A, \text{output}_B, \text{preference})$. The reward model learns from the *contrast* between two responses to the same prompt — that is the training signal in the Bradley-Terry / pairwise-ranking objective used in [The RLHF Pipeline & Reward Modeling](../05-posttraining-alignment/05-rlhf-reward-modeling.html).

    If you only log the served output, you have at most $(\text{prompt}, \text{output}_{\text{served}}, \text{signal})$ — a single response with an absolute rating (e.g. thumbs-up). You cannot reconstruct what the *other* variant would have produced after the fact, because generation is stochastic (sampling temperature, seeds) and the losing variant may already be retired by the time you assemble the training set. The counterfactual output is unrecoverable.

    What breaks concretely:

    - You lose the ability to build within-prompt pairs from live A/B traffic, which is exactly the highest-value, on-distribution preference data. `sample_pairs_for_labeling` groups candidates `by_prompt` and requires `len(resps) >= 2`; with only the served output you have one response per prompt and the pair enumeration produces nothing.
    - Even the implicit signal becomes biased: the served output is chosen by the routing policy, so a plain absolute-rating dataset is confounded by which variant was routed. A pair from the *same* prompt cancels prompt difficulty out of the comparison; an absolute rating does not.

    So the counterfactual output must be captured synchronously at serve time (log both, show one). It costs one extra forward pass, and it is the only moment the counterfactual exists.

**2.** (Quantitative) You are scoring candidate response pairs for annotation with the chapter's pipeline. For one candidate response the model emitted five tokens with natural-log per-token logprobs $[-0.1, -0.2, -2.0, -0.3, -0.4]$.

   (a) Compute the uncertainty proxy $H(\text{output}) = -\frac{1}{T}\sum_{t} \log p_\theta(y_t\mid y_{<t},x)$.

   (b) For a pair whose two responses have reward-model scores $r_A = 1.3$ and $r_B = 0.5$, compute `uncertainty_score` as defined in the chapter (reward scale clipped at $2.0$).

   (c) The pair's normalized edit-distance ratio is $0.5$. Using `uncertainty_weight = 0.6`, compute the final selection `score` that `sample_pairs_for_labeling` would assign.

??? note "Solution"
    (a) $H$ is the average negative log-probability per token. Sum of logprobs $= -0.1-0.2-2.0-0.3-0.4 = -3.0$, over $T=5$ tokens:

    $$
    H = -\frac{1}{5}(-3.0) = \frac{3.0}{5} = 0.6
    $$

    The lone low-logprob token ($-2.0$) is what drives the uncertainty up; the model was confident on the other four positions.

    (b) `uncertainty_score` uses the reward margin, inverted and clipped: margin $= |r_A - r_B| = |1.3 - 0.5| = 0.8$.

    $$
    \text{unc} = \max\!\left(0,\; 1 - \frac{0.8}{2.0}\right) = 1 - 0.4 = 0.6
    $$

    (c) The selection score is a weighted blend of uncertainty and diversity (the edit-distance ratio), with `diversity = ed = 0.5`:

    $$
    \text{score} = 0.6 \cdot \text{unc} + (1-0.6)\cdot \text{diversity} = 0.6\cdot 0.6 + 0.4\cdot 0.5 = 0.36 + 0.20 = 0.56
    $$

    A higher-margin (more confidently ranked) or more near-identical pair would score lower and fall below the top-`n_pairs` cutoff.

**3.** (Quantitative) A weekly retraining run accumulates $D_{\text{new}} = 6000$ freshly labeled examples. You mix them with a frozen replay buffer using the chapter's ratio $\mathcal{D}_{\text{train}} = (1-\rho)\,\mathcal{D}_{\text{new}} \cup \rho\,\mathcal{D}_{\text{replay}}$, where the new data forms the $(1-\rho)$ fraction of the final training set. You want to use all 6000 new examples.

   (a) With $\rho = 0.4$, how large is the assembled training set, and how many replay examples are drawn?

   (b) A teammate proposes $\rho = 0.1$ to "learn faster from fresh data." How many replay examples does that draw, and which failure mode from the chapter does this invite?

??? note "Solution"
    The new examples occupy the $(1-\rho)$ fraction, so if all 6000 are used, the total size is $|\mathcal{D}_{\text{train}}| = D_{\text{new}} / (1-\rho)$, and replay count $= |\mathcal{D}_{\text{train}}| - D_{\text{new}}$.

    (a) With $\rho = 0.4$:

    $$
    |\mathcal{D}_{\text{train}}| = \frac{6000}{1 - 0.4} = \frac{6000}{0.6} = 10000, \qquad \text{replay} = 10000 - 6000 = 4000
    $$

    This sits squarely in the chapter's recommended $\rho \in [0.3, 0.5]$ band.

    (b) With $\rho = 0.1$:

    $$
    |\mathcal{D}_{\text{train}}| = \frac{6000}{0.9} \approx 6667, \qquad \text{replay} = 6667 - 6000 \approx 667
    $$

    Only ~667 old examples survive against 6000 new ones. This invites **catastrophic forgetting**: the chapter names replay buffers at $\rho \approx 0.3$-$0.5$ as the primary defense, warning that too low a replay fraction lets the model forget earlier capabilities (better on the new week's data, worse on some existing capability). The regression test suite would likely catch it at the eval gate and block deployment — wasting the training run.

**4.** (Implementation) The chapter's implicit-signal table says a proxy reward model can label the ~95% of traffic with no explicit thumbs-up/down. As a first, rule-based version of that proxy, implement a function `implicit_proxy_reward(record: RequestRecord) -> float` returning a scalar in $[-1, 1]$. Follow the chapter's table: an explicit thumbs signal dominates; otherwise a copy is a strong positive; an edit means "close but wrong" (scale the penalty by how much was edited, reusing `edit_distance_ratio`); no signal at all is neutral. Keep it consistent with the `RequestRecord` fields defined in `request_logger.py`.

??? note "Solution"
    The `RequestRecord` fields available as signals are `thumbs_up` (`Optional[bool]`), `copied_output` (`Optional[bool]`), and `edited_output` (`Optional[str]`, the user's edited text). We map them to the chapter's table: explicit feedback is the strongest and overrides everything; a copy is "often strong positive"; an edit is "close but wrong" whose severity scales with edit distance from the original `model_output`.

    ```python
    # flywheel/labeling/implicit_proxy.py
    from flywheel.logging.request_logger import RequestRecord
    from flywheel.labeling.preference_sampler import edit_distance_ratio


    def implicit_proxy_reward(record: RequestRecord) -> float:
        """
        Rule-based proxy reward in [-1, 1] derived from implicit signals.
        Priority: explicit thumbs > copy > edit > no signal (neutral).
        A later learned classifier would replace these hand-set weights,
        trained to predict thumbs_up from the same features.
        """
        # 1. Explicit feedback dominates when present.
        if record.thumbs_up is not None:
            return 1.0 if record.thumbs_up else -1.0

        # 2. A copy is a strong positive (chapter: "often strong positive").
        if record.copied_output:
            return 0.7

        # 3. An edit means "close but wrong": penalty scales with how much
        #    of the output the user had to change. Small edit -> mild
        #    negative; wholesale rewrite -> stronger negative.
        if record.edited_output is not None:
            frac_changed = edit_distance_ratio(
                record.model_output, record.edited_output
            )  # in [0, 1]
            return -0.5 * frac_changed

        # 4. No signal: neutral. (Session-continuation signals, if joined
        #    in, could nudge this slightly positive.)
        return 0.0
    ```

    Notes on the design choices, all grounded in the chapter:

    - `thumbs_up` is checked first because explicit human preference is described as "the highest-quality signal"; when it exists it should not be diluted by weaker proxies.
    - The edit penalty reuses `edit_distance_ratio` from `preference_sampler.py` so a one-character fix returns a value near $0$ (barely wrong) while a full rewrite approaches $-0.5$ (mostly wrong but still better than an outright regenerate, which the table flags as "clearly bad").
    - The output is bounded in $[-1, 1]$, so these scores can be dropped straight into `reward_model_score` slots for `uncertainty_score` / `sample_pairs_for_labeling`, or used as regression targets when you later train the *learned* proxy RM to predict `thumbs_up` from these features.

**5.** (Implementation / quantitative) Turn the chapter's "Compounding Data Advantage" model into a runnable simulator and report the moat after 10 rounds. Use this well-posed version of the recurrences (each round, update data, then quality, then users):

$$
D_{k+1} = D_k + d\cdot N_k, \qquad Q_{k+1} = Q_k + \beta\log\!\frac{D_{k+1}}{D_k}, \qquad N_{k+1} = N_k\,(1 + \alpha\,\Delta Q_{k+1})
$$

   with $\Delta Q_{k+1} = Q_{k+1}-Q_k$. Initialize $N_0 = 10000$, $D_0 = 100000$, $Q_0 = 1.0$, and use $\alpha = 0.1$, $\beta = 0.05$, $d = 10$. Compute $N_{10}/N_0$, and explain why the per-round *quality* gain $\Delta Q_k$ shrinks even as the user base keeps compounding.

??? note "Solution"
    ```python
    # flywheel/analysis/compounding_sim.py
    import math


    def simulate_flywheel(rounds=10, N0=10000.0, D0=100000.0, Q0=1.0,
                          alpha=0.1, beta=0.05, d=10.0):
        N, D, Q = N0, D0, Q0
        for _ in range(rounds):
            D_next = D + d * N                    # new examples from current users
            Q_next = Q + beta * math.log(D_next / D)  # log-linear data -> quality
            dQ = Q_next - Q
            N_next = N * (1.0 + alpha * dQ)       # better quality attracts users
            N, D, Q = N_next, D_next, Q_next
        return N, D, Q


    if __name__ == "__main__":
        N, D, Q = simulate_flywheel()
        print(f"N10/N0 = {N/10000.0:.6f}")   # -> 1.012085
        print(f"N10={N:.2f}  D10={D:.2f}  Q10={Q:.4f}")
    ```

    Running it (matching a hand-check of the first two rounds):

    - Round 1: $D_1 = 100000 + 10\cdot 10000 = 200000$; $Q_1 = 1 + 0.05\ln 2 \approx 1.03466$; $\Delta Q_1 \approx 0.03466$; $N_1 = 10000(1 + 0.1\cdot 0.03466) \approx 10034.7$.
    - Round 2: $D_2 = 200000 + 10\cdot 10034.7 \approx 300347$; $Q_2 \approx 1.05499$; $\Delta Q_2 \approx 0.02033$; $N_2 \approx 10055.1$.
    - After 10 rounds: $N_{10}/N_0 \approx \mathbf{1.0121}$, with $D_{10} \approx 1.108\times 10^{6}$ and $Q_{10} \approx 1.120$.

    Why $\Delta Q_k$ shrinks: quality grows with the *log ratio* $\ln(D_{k+1}/D_k)$, consistent with the log-linear scaling-law intuition the chapter invokes. Even though the absolute number of new examples $d\cdot N_k$ grows every round, the *ratio* $D_{k+1}/D_k = 1 + d N_k / D_k$ shrinks toward $1$ because the cumulative denominator $D_k$ grows faster than the per-round increment. So $\ln(D_{k+1}/D_k)\to 0$ and each round buys less quality than the last — diminishing returns on raw volume.

    This is exactly the chapter's punchline: past a certain scale, another random on-distribution example barely moves quality (the $\log$ is flattening), so the marginal value of *data volume* collapses while the marginal value of *hard, long-tail, high-uncertainty* examples stays high. The moat is not the modest $1.2\%$ user compounding in these tame parameters — it is that a round-10 entrant starts at $N_0, D_0$ with *zero* accumulated coverage of the long tail, a distribution gap no architecture tweak closes.
