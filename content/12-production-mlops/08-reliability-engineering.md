# 12.8 Reliability Engineering for LLM Systems: SLOs & Incident Response

Classical site reliability engineering (SRE) was designed for systems with clear binary outcomes: a request either succeeds or it fails. Large language model (LLM) systems break this assumption in at least three important ways. First, a response can be syntactically valid but semantically wrong — the request "succeeded" but the user got garbage. Second, quality degrades continuously rather than discretely; there is no connection reset, no 500 status code, no stack trace. Third, correctness depends on the entire pipeline — the prompt template, retrieval corpus, model version, and sampling parameters — so the usual "was the upstream service up?" diagnosis tree is insufficient.

This chapter adapts SRE methodology to these realities. We cover how to define service level indicators (SLIs) and objectives (SLOs) for probabilistic text systems, how to build a diagnosis tree for LLM-specific failure modes, how to execute prompt and model rollbacks safely, how to write trace-attached postmortems, and how to design graceful degradation and multi-provider failover. By the end you will have concrete runbooks you can paste into an incident wiki.

This chapter sits at the intersection of several others. Observability primitives (traces, spans, structured logs) are covered in [Observability, Logging & LLMOps](../12-production-mlops/02-observability-llmops.html). How to set up online A/B testing and guardrail metrics is in [Online Evaluation: A/B Testing, Canaries & Guardrail Metrics](../12-production-mlops/07-online-eval-ab-testing.html). Cost-based routing decisions belong in [Caching, Routing & Cost Control in Production](../12-production-mlops/03-caching-routing-cost.html).

---

## SLIs and SLOs for Probabilistic Systems

### The problem with binary success rates

In a traditional API, SLI = (successful requests) / (total requests). For an LLM API this ratio is deceptive: every response that comes back with HTTP 200 counts as a success, even if the model hallucinated a phone number, switched language mid-paragraph, or returned a blank string. A system could maintain 99.9% HTTP success while delivering value only 70% of the time.

{{fig:reliability-http200-quality-gap}}

We need a richer SLI vocabulary that covers three dimensions:

| Dimension | What it measures | Typical SLO target |
|---|---|---|
| **Availability** | Fraction of requests that receive any response | 99.9% (43 min/month downtime budget) |
| **Latency** | Time-to-first-token (TTFT) and end-to-end (E2E) at tail percentiles | TTFT p99 < 1 s; E2E p99 < 10 s |
| **Quality** | Fraction of responses meeting a quality bar (automated or sampled judge) | Quality SLO ≥ 95% on canary eval suite |

### Defining quality SLOs concretely

A *quality SLI* requires an automated judge. The judge can be a lightweight classifier trained on your labeled data, an LLM-as-a-judge rubric (see [LLM-as-a-Judge & Automated Evaluation](../11-evaluation/02-llm-as-judge.html)), or a suite of regex/heuristic checks for structural properties (valid JSON output, correct language, non-empty, within length bounds).

Define the SLO on a *sliding window*:

$$
\text{Quality SLO window} = \frac{\text{judged-good responses in last } W \text{ minutes}}{\text{total judged responses in last } W \text{ minutes}} \geq \theta
$$

Choose $W$ and $\theta$ based on traffic volume. With 10,000 requests per hour a 30-minute window gives you 5,000 samples, enough to detect a 2-percentage-point drop at high confidence. With 100 requests per hour, use a 6-hour window and supplement with a daily offline eval suite.

### Latency tail budgets

LLM latency is bimodal: most responses are fast, but long-input or long-output requests hit the tail hard. The p99 latency is often 5–10× the p50, unlike typical web APIs where the ratio is 2–3×. SLOs should track:

- **TTFT p99** — the latency until the first token appears in the client. This is the "loading" experience. Budget: on the order of 0.5–2 s for interactive use cases.
- **Tokens-per-second (TPS) p50** — decode speed during streaming.
- **Total response latency p99** — relevant for non-streaming callers.
- **Timeout rate** — fraction of requests exceeding a hard wall-clock limit.

Track these per route, not just globally. A summarization endpoint with 4,096-token outputs has very different latency characteristics than a classification endpoint returning a single token.

!!! example "Worked example: error budget arithmetic"

    Suppose your availability SLO is 99.9% over a 30-day window.

    - Total minutes in 30 days: $30 \times 24 \times 60 = 43{,}200$ minutes.
    - Allowed downtime: $43{,}200 \times 0.001 = 43.2$ minutes.
    - Your quality SLO is 95% (judged-good) on a per-hour window.
    - With 2,000 requests/hour, a quality SLO burn of 1× means $2{,}000 \times 0.05 = 100$ bad responses per hour.
    - A provider regression that drops quality to 80% burns $2{,}000 \times (0.95 - 0.80) = 300$ extra bad responses per hour, or $300/100 = 3\times$ your error budget rate.
    - At that rate, your 30-day quality error budget (assuming budget = 5% × total requests) is exhausted in $30/3 = 10$ days — a clear threshold to trigger incident escalation.

### The "gradual silent collapse" failure mode

The most dangerous LLM failure is one you do not notice for days. This happens when:

1. A provider silently rolls out a new model version that scores slightly worse on your task.
2. Prompt drift — someone edits a prompt template without a review and degrades quality by a few percent.
3. Retrieval corpus staleness — documents age out and the RAG index starts returning off-topic chunks.

None of these produce HTTP errors. Your latency dashboard looks green. Only a quality SLO with a short enough window catches them early.

**Detection strategy:** run a *canary eval* — a fixed set of 50–200 golden request/response pairs with automated scoring — every 15 minutes in production. Alert if the pass rate drops below the SLO threshold for two consecutive windows. The fixed golden set is immune to traffic distribution changes, giving you a stable signal.

---

## A Four-Root-Cause Diagnosis Tree

When a quality or latency alert fires, you need a structured way to identify the cause before you call a war room. Here is a practical four-branch tree:


{{fig:reliability-diagnosis-tree}}


This tree is a triage guide, not a decision tree in the ML sense. You should check all four branches in parallel; in practice, two or more causes can coincide.

### Branch 1: Model/provider regression

**Signals:**
- Provider API error rate increases (5xx, 429, timeout).
- TTFT or E2E latency p99 spike without change in traffic volume.
- Quality drops suddenly on canary eval but prompt/retrieval show no change.
- Provider status page (e.g., OpenAI's status.openai.com, Anthropic's status.anthropic.com) shows an incident.

**Immediate actions:**
1. Check provider status page programmatically (see runbook code below).
2. Compare quality on a fixed eval set against the last known-good baseline.
3. If quality is degraded, activate failover to secondary provider (§Multi-Provider Failover).

**Diagnostic code:**

```python
import httpx
import json
import datetime
from dataclasses import dataclass, field
from typing import Optional

PROVIDER_STATUS_URLS = {
    "openai": "https://status.openai.com/api/v2/status.json",
    "anthropic": "https://status.anthropic.com/api/v2/status.json",
    "google": "https://status.cloud.google.com/incidents.json",
}

@dataclass
class ProviderHealth:
    provider: str
    status: str          # "operational", "degraded", "outage"
    indicator: str       # raw indicator from status page
    checked_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)
    error: Optional[str] = None

async def check_provider_status(provider: str) -> ProviderHealth:
    """Fetch provider status page and parse the summary indicator."""
    url = PROVIDER_STATUS_URLS.get(provider)
    if not url:
        return ProviderHealth(provider=provider, status="unknown", indicator="no-url")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(url)
            r.raise_for_status()
            data = r.json()
        # Statuspage.io v2 format: data["status"]["indicator"]
        # values: "none" | "minor" | "major" | "critical"
        indicator = data.get("status", {}).get("indicator", "unknown")
        status = (
            "operational" if indicator == "none"
            else "degraded" if indicator in ("minor", "major")
            else "outage"
        )
        return ProviderHealth(provider=provider, status=status, indicator=indicator)
    except Exception as exc:
        # Treat connection failure as potential outage
        return ProviderHealth(
            provider=provider, status="unknown", indicator="fetch-error", error=str(exc)
        )

async def diagnose_providers() -> dict[str, ProviderHealth]:
    import asyncio
    results = await asyncio.gather(
        *[check_provider_status(p) for p in PROVIDER_STATUS_URLS],
        return_exceptions=False,
    )
    return {h.provider: h for h in results}
```

### Branch 2: Prompt change regression

Every prompt template must be versioned. The simplest versioning scheme is a SHA-256 hash of the rendered system prompt concatenated with the user prompt template. Store this hash in every request trace.

**Signals:**
- Quality SLI drop coincides with a prompt template deployment.
- The distribution of response lengths, refusal rates, or structured-output parse failures shifts.

### Branch 3: Retrieval drift

For RAG pipelines, the retrieval layer can silently degrade when:
- The corpus is not re-indexed after document updates, leaving stale or deleted documents in the index.
- Embedding model version changes alter vector representations, misaligning query and document spaces.
- The reranker's training distribution diverges from production queries.

**Signals (instrument these as SLIs):**
- Mean reciprocal rank (MRR) on a golden query set drops below threshold.
- Fraction of retrievals with cosine similarity above 0.7 drops.
- Average chunk relevance score from the reranker drops.

Cross-reference [Chunking, Reranking & Hybrid Search](../09-rag-retrieval/04-chunking-reranking-hybrid.html) for reranker architecture, and [Retrieval-Augmented Generation Architectures](../09-rag-retrieval/03-rag-architectures.html) for end-to-end pipeline design.

### Branch 4: Upstream data regression

LLM pipelines often sit downstream of data pipelines that feed feature stores, knowledge bases, or fine-tuning datasets. A schema migration, a missing backfill, or a buggy extraction job can inject corrupted context into every request.

**Signals:**
- Spike in context parse errors.
- Change in distribution of metadata fields (e.g., sudden increase in null values).
- Data pipeline DAG shows failed or late runs.

---

## Prompt and Model Rollback

### Prompt versioning infrastructure

Treat prompt templates like code: version them in git, gate deployments behind review, and make rollback a one-command operation.

```python
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Optional
import redis  # pip install redis

@dataclass
class PromptVersion:
    """A versioned prompt template stored in a fast key-value store."""
    template_id: str      # stable identifier, e.g. "customer-support-v1"
    version: str          # semantic version, e.g. "2.3.1"
    sha256: str           # hash of the rendered canonical template
    system_prompt: str
    user_prompt_template: str  # Jinja2 or f-string template
    deployed_at: float    # Unix timestamp
    deployed_by: str
    rollback_to: Optional[str] = None  # version to revert to on rollback

class PromptRegistry:
    """
    Redis-backed registry for prompt versions.
    Supports atomic deploy/rollback with audit trail.
    """
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.r = redis.from_url(redis_url, decode_responses=True)

    def _compute_sha(self, system: str, user_template: str) -> str:
        payload = json.dumps({"system": system, "user": user_template}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def deploy(self, v: PromptVersion) -> None:
        """Atomically deploy a new prompt version, saving previous for rollback."""
        key = f"prompt:{v.template_id}:current"
        prev_json = self.r.get(key)
        pipe = self.r.pipeline(transaction=True)
        if prev_json:
            # Archive previous version for rollback
            prev = json.loads(prev_json)
            pipe.set(f"prompt:{v.template_id}:previous", prev_json)
            v.rollback_to = prev["version"]
        pipe.set(key, json.dumps(v.__dict__))
        # Keep a full audit log
        pipe.lpush(f"prompt:{v.template_id}:history", json.dumps(v.__dict__))
        pipe.execute()
        print(f"Deployed {v.template_id}@{v.version} (sha={v.sha256})")

    def rollback(self, template_id: str) -> Optional[PromptVersion]:
        """Atomically revert to the previous prompt version."""
        prev_json = self.r.get(f"prompt:{template_id}:previous")
        if not prev_json:
            print(f"No previous version found for {template_id}")
            return None
        prev_data = json.loads(prev_json)
        prev = PromptVersion(**prev_data)
        # Swap current ← previous
        pipe = self.r.pipeline(transaction=True)
        pipe.set(f"prompt:{template_id}:current", prev_json)
        pipe.lpush(
            f"prompt:{template_id}:history",
            json.dumps({"event": "rollback", "to": prev.version, "at": time.time()}),
        )
        pipe.execute()
        print(f"Rolled back {template_id} to {prev.version}")
        return prev

    def get_current(self, template_id: str) -> Optional[PromptVersion]:
        data = self.r.get(f"prompt:{template_id}:current")
        if not data:
            return None
        return PromptVersion(**json.loads(data))
```

### Model rollback

For self-hosted models, model rollback means reverting the serving deployment to a previous checkpoint. For API providers, you cannot directly control model versions, but you can:

1. Pin a specific model version string (e.g., `gpt-4o-2024-08-06` instead of `gpt-4o`). Pinned versions are deprecated on a schedule, but they give you control over when to absorb a model update.
2. Maintain a *shadow model* running the new version against 5% of traffic. Monitor quality SLI on both. Only switch 100% traffic after the shadow passes.
3. If the provider offers no pinning and degrades quality, activate the secondary provider.

**Canary eval on deploy:**

```python
import asyncio
from typing import Callable, Awaitable

async def canary_eval_gate(
    template_id: str,
    new_version: PromptVersion,
    eval_fn: Callable[[str, str], Awaitable[float]],   # (prompt, response) -> [0,1]
    golden_cases: list[dict],   # list of {"input": ..., "expected_score": float}
    pass_threshold: float = 0.92,
    registry: PromptRegistry = None,
) -> bool:
    """
    Run a canary eval before fully deploying a new prompt version.
    Returns True if the new version passes the quality gate.
    """
    scores = []
    for case in golden_cases:
        # Render prompt using new template
        prompt = new_version.user_prompt_template.format(**case["input"])
        # In production replace this with your actual LLM call
        response = await mock_llm_call(prompt, new_version.system_prompt)
        score = await eval_fn(prompt, response)
        scores.append(score)

    mean_score = sum(scores) / len(scores)
    passed = mean_score >= pass_threshold
    print(
        f"Canary eval for {template_id}@{new_version.version}: "
        f"mean_score={mean_score:.3f}, threshold={pass_threshold}, passed={passed}"
    )
    if passed and registry:
        registry.deploy(new_version)
    return passed

async def mock_llm_call(prompt: str, system: str) -> str:
    """Stub — replace with actual provider call."""
    await asyncio.sleep(0.01)
    return "mock response"
```

!!! warning "Never roll forward without a quality gate"

    A common mistake is to deploy a prompt fix during an active incident without running the canary eval suite first. The "fix" can introduce a new regression. Under incident pressure, add a fast gate: run 20–30 golden cases before touching 100% of traffic. Ten minutes of testing buys you confidence that the rollout won't make things worse.

---

## Trace-Attached Postmortems

### Why traces matter for LLM postmortems

Traditional postmortems rely on logs and metrics. LLM incidents require *trace-level* evidence: the exact prompt that was sent, the exact response received, the retrieval chunks that were injected, the sampling parameters used, and the latency of each pipeline stage. Without traces, you cannot answer "did the prompt change cause the regression?" or "which requests were affected?"

Every LLM request should emit a structured trace with at least these fields:

```json
{
  "trace_id": "a3f1b2c4-...",
  "timestamp": "2026-06-04T09:42:11.123Z",
  "route": "customer-support",
  "prompt_template_id": "customer-support-v1",
  "prompt_sha": "4a7e9c12",
  "model": "gpt-4o-2024-08-06",
  "provider": "openai",
  "retrieval": {
    "query": "how do I cancel my subscription",
    "top_k": 5,
    "chunks": [
      {"doc_id": "faq-cancel-001", "score": 0.91, "text": "..."}
    ],
    "retrieval_latency_ms": 38
  },
  "llm_call": {
    "input_tokens": 812,
    "output_tokens": 143,
    "ttft_ms": 342,
    "total_latency_ms": 1204,
    "finish_reason": "stop"
  },
  "quality_judge": {
    "score": 0.88,
    "flags": []
  },
  "user_feedback": null
}
```

See [Observability, Logging & LLMOps](../12-production-mlops/02-observability-llmops.html) for the full observability infrastructure to produce these traces.

### Postmortem template

A good LLM postmortem has six sections. The key addition over traditional postmortems is the *trace evidence* section, which anchors every claim to a specific trace ID.

```text
=== LLM INCIDENT POSTMORTEM ===

Incident ID: INC-2026-0604-001
Severity: SEV-2 (Quality SLO breach, ~18% quality degradation for 2h 20m)
Author: on-call engineer
Review date: 2026-06-06

1. SUMMARY
   <2-3 sentences: what happened, how long, business impact>

2. TIMELINE (UTC)
   09:12  Quality SLO alert fires (canary eval pass rate: 77%, SLO: 95%)
   09:18  On-call acknowledges; begins diagnosis tree
   09:24  Branch 1 (provider): OpenAI status operational; latency nominal
   09:27  Branch 2 (prompt): no prompt deploy in last 48h
   09:31  Branch 3 (retrieval): MRR on golden set dropped from 0.71 → 0.54
   09:35  Root cause identified: corpus re-index job failed at 07:00;
           stale index missing 12% of documents added last month
   09:48  Re-index triggered; traffic held at current quality
   11:32  Index rebuild complete; quality SLO restored (canary: 96%)

3. ROOT CAUSE
   The nightly re-index cron job failed silently (exit code 0 despite partial
   failure). 23,000 documents added in the previous 3 weeks were absent from
   the production index. Retrieval was returning lower-relevance fallback
   documents for ~30% of queries.

4. TRACE EVIDENCE
   Affected trace sample (earliest detection):
     trace_id: a3f1b2c4-7e9d-...
     retrieval.chunks[0].score: 0.43  (normal: >0.75)
     quality_judge.score: 0.61        (SLO: >=0.95)
     quality_judge.flags: ["off-topic-context"]

5. ACTION ITEMS
   [ ] Add exit-code validation + document-count assertion to re-index job
   [ ] Alert if post-index doc count drops >5% vs pre-index count
   [ ] Add retrieval MRR to real-time SLI dashboard (was offline-only)
   [ ] Document re-index runbook in incident wiki

6. WHAT WENT WELL
   - Canary eval detected the issue within 10 min of corpus failure
   - Diagnosis tree narrowed root cause to retrieval in <25 min
```

### Mean-time-to-detect (MTTD) and mean-time-to-restore (MTTR)

Track these two metrics across all incidents:

$$
\text{MTTD} = \frac{1}{N}\sum_{i=1}^{N} (t_{\text{alert},i} - t_{\text{fault\_start},i})
$$

$$
\text{MTTR} = \frac{1}{N}\sum_{i=1}^{N} (t_{\text{restored},i} - t_{\text{alert},i})
$$

For LLM quality incidents, MTTD is dominated by the width of your quality SLI window. A 30-minute window means worst-case 30-minute MTTD. A 5-minute window with lower confidence (more noise) means faster detection at the cost of false positives. Tune this tradeoff based on your error budget burn rate.

---

## Detecting Gradual Silent Quality Collapse

### Why silent collapse is harder than hard failures

A provider outage causes an immediate spike in error rate. Silent quality collapse is insidious: it might manifest as a 2–3% drop per week in user satisfaction ratings, invisible against normal noise. By the time it is noticed, you have lost weeks of error budget and potentially user trust.

{{fig:reliability-silent-quality-collapse}}

Three instrumentation strategies combat this:

**1. Anchored canary evals.** As described above: a fixed golden set scored automatically every 15 minutes. The golden set must be *frozen* — never updated during an incident, only extended during calm periods after careful human review.

**2. Behavioral drift metrics.** Track the distribution of automated quality scores over time, not just the pass/fail rate. A shift in mean score from 0.91 to 0.87 may not breach the SLO threshold yet, but it is a leading indicator of imminent breach.

$$
\text{quality drift} = \bar{s}_{t} - \bar{s}_{t-\Delta}
$$

Alert when $|\text{quality drift}| > \epsilon$ for two consecutive windows, where $\epsilon$ is calibrated on historical variance (a common heuristic: $\epsilon = 2\sigma_{\text{historical}}$).

**3. User signal feedback loops.** Thumbs-up/down, edit-rate, copy-rate, and session abandonment are lagging but high-signal quality indicators. Join them back to trace IDs for root-cause correlation. See [Data Flywheels & Continuous Improvement](../12-production-mlops/05-data-flywheel.html) for how to build this loop.

### Detecting regression at the segment level

A global quality SLO can mask a severe regression in a specific user segment or query type. Track quality SLIs broken down by:

- **Route** (summarization, Q&A, code generation, classification)
- **Language** (quality regressions in non-English languages are commonly missed)
- **Input length bucket** (short / medium / long context)
- **Time of day** (some providers have degraded off-peak performance)

```python
from collections import defaultdict
import statistics
from typing import NamedTuple

class QualityRecord(NamedTuple):
    trace_id: str
    route: str
    language: str
    input_len_bucket: str   # "short" | "medium" | "long"
    score: float

def segment_quality_report(
    records: list[QualityRecord],
    slo_threshold: float = 0.95,
) -> dict:
    """
    Compute per-segment quality pass rates and flag segments breaching SLO.
    Returns a dict: segment_key -> {pass_rate, count, breaching}.
    """
    buckets: dict[tuple, list[float]] = defaultdict(list)
    for r in records:
        key = (r.route, r.language, r.input_len_bucket)
        buckets[key].append(r.score)

    report = {}
    for key, scores in buckets.items():
        pass_rate = sum(1 for s in scores if s >= slo_threshold) / len(scores)
        report[key] = {
            "pass_rate": pass_rate,
            "mean_score": statistics.mean(scores),
            "count": len(scores),
            "breaching": pass_rate < slo_threshold,
        }
    # Sort by pass_rate ascending so worst segments are first
    return dict(sorted(report.items(), key=lambda x: x[1]["pass_rate"]))
```

!!! interview "Interview Corner"

    **Q:** How would you design a quality SLO for an LLM-powered customer support system, and how would you detect a silent quality regression?

    **A:** I would define a multi-dimensional SLO: availability (HTTP success rate >= 99.9%), latency (TTFT p99 < 1 s), and quality (automated judge pass rate >= 95% on a rolling 30-minute window). For silent regression detection, I'd run an anchored canary eval — a frozen set of ~100 golden request/response pairs scored by a lightweight classifier — every 15 minutes. I'd also track the *distribution* of quality scores, not just the pass rate, and alert on a 2-sigma drift in mean score. Finally, I'd segment quality by route and language to catch regressions the global metric masks. The key insight is that HTTP-level metrics are insufficient; you need application-layer quality telemetry.

---

## Fallback and Degradation Design

### The degradation ladder

Rather than binary "up or down," design a ladder of degraded states, each providing less value but more reliability:


{{fig:reliability-degradation-ladder}}


Implement level transitions as a circuit-breaker pattern. Each transition should be automatic (triggered by SLI thresholds), logged, and reversible.

```python
import threading
import time
from enum import IntEnum
from typing import Callable, Optional

class DegradationLevel(IntEnum):
    NORMAL = 0
    RETRIEVAL_OFF = 1
    MODEL_FALLBACK = 2
    CACHED_ONLY = 3
    GRACEFUL_ERROR = 4

class DegradationController:
    """
    Thread-safe degradation controller.
    Advances or retreats degradation level based on SLI measurements.
    Uses exponential backoff before attempting recovery.
    """
    def __init__(self, recovery_probe_interval: float = 60.0):
        self._level = DegradationLevel.NORMAL
        self._lock = threading.Lock()
        self._last_degraded_at: Optional[float] = None
        self._recovery_probe_interval = recovery_probe_interval  # seconds

    @property
    def level(self) -> DegradationLevel:
        return self._level

    def degrade(self, reason: str) -> DegradationLevel:
        """Advance one level; returns new level."""
        with self._lock:
            if self._level < DegradationLevel.GRACEFUL_ERROR:
                self._level = DegradationLevel(self._level + 1)
                self._last_degraded_at = time.monotonic()
                print(f"[DEGRADATION] Level → {self._level.name}: {reason}")
        return self._level

    def recover(self) -> DegradationLevel:
        """Retreat one level; returns new level."""
        with self._lock:
            if self._level > DegradationLevel.NORMAL:
                # Enforce minimum dwell time before recovery attempt
                if (self._last_degraded_at is not None and
                        time.monotonic() - self._last_degraded_at < self._recovery_probe_interval):
                    return self._level
                self._level = DegradationLevel(self._level - 1)
                print(f"[RECOVERY] Level → {self._level.name}")
        return self._level

    def reset(self) -> None:
        """Force reset to NORMAL (use only in manual incident resolution)."""
        with self._lock:
            self._level = DegradationLevel.NORMAL
            print("[RESET] Degradation level reset to NORMAL")

# Usage pattern: call from your SLI monitoring loop
controller = DegradationController(recovery_probe_interval=120.0)

def handle_request(query: str) -> str:
    level = controller.level
    if level == DegradationLevel.NORMAL:
        return full_rag_pipeline(query)
    elif level == DegradationLevel.RETRIEVAL_OFF:
        return model_only_pipeline(query)
    elif level == DegradationLevel.MODEL_FALLBACK:
        return small_model_pipeline(query)
    elif level == DegradationLevel.CACHED_ONLY:
        cached = lookup_cache(query)
        return cached or graceful_error_response()
    else:
        return graceful_error_response()

def full_rag_pipeline(q): return f"[RAG] {q}"
def model_only_pipeline(q): return f"[MODEL-ONLY] {q}"
def small_model_pipeline(q): return f"[SMALL-MODEL] {q}"
def lookup_cache(q): return None
def graceful_error_response(): return "Service temporarily limited. Please try again shortly."
```

### Caching as a reliability primitive

A semantic cache (described in detail in [Caching, Routing & Cost Control in Production](../12-production-mlops/03-caching-routing-cost.html)) doubles as a reliability backstop. When the primary model is unavailable or degraded, serve cached responses for queries with cosine similarity above a threshold to a cached query.

The hit rate of your semantic cache under normal traffic is the ceiling on how many requests you can serve from cache during an outage. Maintain a minimum cache size — warm the cache proactively with your top-K most frequent query patterns.

---

## Provider Outage Runbooks and Multi-Provider Failover

### The multi-provider architecture

No single LLM provider offers five-nines availability. OpenAI, Anthropic, and Google all experience incidents several times per year. A production LLM system must have at least one secondary provider and an automatic failover mechanism.


{{fig:reliability-multiprovider-gateway}}


The gateway handles:
1. **Health probing** — lightweight synthetic request every 30 s per provider.
2. **Circuit breaking** — if a provider returns >5% errors in a 60-second window, open its circuit and route to the next provider.
3. **Latency SLO enforcement** — if provider latency p95 exceeds budget, deprioritize (soft circuit break) and increase weight on the faster provider.
4. **Model equivalence mapping** — map your internal model alias (e.g., `llm-v2`) to provider-specific model IDs (e.g., `gpt-4o-2024-08-06` or `claude-opus-4-5`).

```python
import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional
import httpx

@dataclass
class ProviderConfig:
    name: str
    api_base: str
    api_key_env: str
    model_id: str          # provider-specific model identifier
    weight: float = 1.0    # routing weight (higher = preferred)
    max_failures: int = 5  # failures in window before circuit opens
    window_seconds: float = 60.0

@dataclass
class CircuitState:
    failures: list[float] = field(default_factory=list)
    open_until: float = 0.0  # monotonic timestamp; 0 = closed

    def record_failure(self, window: float) -> None:
        now = time.monotonic()
        self.failures = [t for t in self.failures if now - t < window]
        self.failures.append(now)

    def is_open(self, max_failures: int) -> bool:
        if time.monotonic() < self.open_until:
            return True
        return len(self.failures) >= max_failures

    def trip(self, cooldown: float = 30.0) -> None:
        self.open_until = time.monotonic() + cooldown
        self.failures.clear()

class MultiProviderGateway:
    """
    Routes LLM requests across multiple providers with circuit breaking.
    Falls back automatically on error or timeout.
    """
    def __init__(self, providers: list[ProviderConfig]):
        self.providers = providers
        self.circuits: dict[str, CircuitState] = {
            p.name: CircuitState() for p in providers
        }

    def _available_providers(self) -> list[ProviderConfig]:
        """Return providers whose circuits are closed, ordered by weight desc."""
        available = [
            p for p in self.providers
            if not self.circuits[p.name].is_open(p.max_failures)
        ]
        return sorted(available, key=lambda p: p.weight, reverse=True)

    async def complete(
        self,
        messages: list[dict],
        timeout: float = 30.0,
        max_tokens: int = 1024,
    ) -> dict:
        """
        Attempt completion across providers in priority order.
        Returns the first successful response.
        Raises RuntimeError if all providers fail.
        """
        import os
        available = self._available_providers()
        if not available:
            raise RuntimeError("All providers have open circuits — no fallback available")

        last_error: Optional[Exception] = None
        for provider in available:
            try:
                result = await self._call_provider(
                    provider, messages, timeout, max_tokens,
                    api_key=os.environ.get(provider.api_key_env, ""),
                )
                # Success: record and return
                print(f"[GATEWAY] Served by {provider.name}")
                return result
            except (httpx.TimeoutException, httpx.HTTPStatusError, Exception) as e:
                last_error = e
                circuit = self.circuits[provider.name]
                circuit.record_failure(provider.window_seconds)
                if circuit.is_open(provider.max_failures):
                    circuit.trip(cooldown=30.0)
                    print(f"[CIRCUIT] Opened for {provider.name}: {e}")
                print(f"[GATEWAY] {provider.name} failed, trying next provider: {e}")

        raise RuntimeError(f"All providers failed. Last error: {last_error}")

    async def _call_provider(
        self,
        provider: ProviderConfig,
        messages: list[dict],
        timeout: float,
        max_tokens: int,
        api_key: str,
    ) -> dict:
        """Thin OpenAI-compatible API call. Extend for non-OpenAI providers."""
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(
                f"{provider.api_base}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": provider.model_id,
                    "messages": messages,
                    "max_tokens": max_tokens,
                },
            )
            r.raise_for_status()
            return r.json()
```

### Provider outage runbook

This is the step-by-step procedure for on-call engineers. Pin it in your incident wiki and Slack channel.


{{fig:reliability-provider-outage-runbook}}


### Rate-limit and quota management

Provider outages include soft outages from rate limiting. A burst of user traffic can exhaust your per-minute token quota, causing 429 errors that your monitoring might mis-classify as an outage.

Track token consumption as a first-class metric. If your gateway observes a 429, it should:
1. Apply exponential backoff with jitter before retrying on the same provider.
2. Route overflow to secondary provider if primary is consistently 429-ing.
3. Alert if token consumption is trending toward the quota limit so you can request a quota increase proactively.

```python
import random

async def retry_with_backoff(
    call_fn,
    max_retries: int = 4,
    base_delay: float = 1.0,
    jitter: float = 0.5,
) -> dict:
    """
    Exponential backoff with full jitter for rate-limited LLM calls.
    Jitter prevents the thundering-herd problem when many workers back off simultaneously.
    """
    for attempt in range(max_retries):
        try:
            return await call_fn()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                delay = base_delay * (2 ** attempt) + random.uniform(0, jitter)
                print(f"Rate limited (attempt {attempt+1}/{max_retries}), retrying in {delay:.2f}s")
                await asyncio.sleep(delay)
            else:
                raise  # Non-rate-limit errors: don't retry here, propagate
    raise RuntimeError(f"Exhausted {max_retries} retries due to rate limiting")
```

---

## The On-Call Lifecycle: Alerts, War Rooms, and Error Budget Reviews

### Alert design principles

Poorly designed alerts are the #1 cause of alert fatigue, which leads to on-call engineers ignoring alerts — defeating their purpose. Apply these principles:

1. **Alert on symptoms, not causes.** Alert on "quality SLO breach" rather than "OpenAI error rate > 1%". The symptom is what matters to users; the cause is what you investigate after.
2. **Every alert must be actionable.** If you cannot write a runbook step for an alert, delete it or convert it to a dashboard warning.
3. **Use multi-window burn rate alerts.** A burn rate of 14.4× means you exhaust your 30-day error budget in 2 days. Trigger pages at burn rate ≥ 14.4× (fast burn) and warnings at burn rate ≥ 6× (slow burn). This is the SRE Workbook recommendation.

The fast-burn formula:

$$
\text{burn rate} = \frac{1 - \text{SLI}_{\text{current}}}{1 - \text{SLO}_{\text{target}}}
$$

For example, if SLO = 0.95 and current quality rate = 0.80:

$$
\text{burn rate} = \frac{1 - 0.80}{1 - 0.95} = \frac{0.20}{0.05} = 4\times
$$

A $4\times$ burn rate exhausts the monthly budget in $30/4 = 7.5$ days — a warning-level alert.

### Error budget reviews

Hold a monthly error budget review. The agenda:

| Item | What to discuss |
|---|---|
| Budget consumed | How much of each SLO budget was spent vs. planned |
| Top incidents by budget impact | Which incidents caused the most budget burn |
| Systemic patterns | Provider > prompt > retrieval drift distribution |
| Action item completion | Were last month's postmortem items closed? |
| Budget policy | Should the team freeze feature deployments if budget < 10%? |

Budget freezes are a powerful forcing function: when the error budget is nearly exhausted, no new prompt changes or retrieval updates ship until the budget is replenished — incentivizing reliability work over feature velocity.

### War room setup

When a SEV-1 (complete outage) or SEV-2 (major SLO breach) fires:

```text
War Room Checklist
==================
[ ] Incident commander assigned (not the same as the engineer debugging)
[ ] Communications lead designated (updates to stakeholders every 30 min)
[ ] Incident channel created: #inc-YYYYMMDD-NNN
[ ] Live dashboard pinned in channel
[ ] Diagnosis tree started in shared doc (anyone can update)
[ ] Provider status pages bookmarked
[ ] Runbook link shared
[ ] Rollback authority confirmed (who can approve a prompt/model rollback?)
[ ] Start a clock: MTTD and MTTR tracking begins now
```

---

## Key Takeaways

!!! key "Key Takeaways"

    - SLOs for LLM systems must include a **quality SLO** (automated judge pass rate) in addition to availability and latency SLOs. HTTP success rate alone is blind to semantic failures.
    - Use **anchored canary evals** — a frozen golden set scored every 15 minutes — to detect gradual silent quality collapse before it becomes a user-visible incident.
    - The **four-root-cause tree** (provider regression, prompt change, retrieval drift, upstream data) gives on-call engineers a structured triage path; check all four branches in parallel.
    - **Prompt and model versioning** must be first-class infrastructure: SHA-based prompt registries with atomic rollback, pinned model version strings at the provider API, and canary eval gates before full rollout.
    - **Trace-attached postmortems** anchor every claim to a specific trace ID (prompt SHA, retrieval scores, quality judge output); without trace evidence, postmortems devolve into speculation.
    - Design a **degradation ladder** (full RAG → retrieval-off → smaller model → cache-only → graceful error) with automatic circuit-breaker transitions; never assume binary up/down.
    - **Multi-provider failover** with circuit breaking is non-negotiable for production systems; no single provider offers five-nines availability, and automatic failover should reduce mean-time-to-restore to under 2 minutes.
    - Use **burn rate alerts** (fast-burn ≥ 14.4×, slow-burn ≥ 6×) rather than raw SLI threshold alerts to give early warning proportional to the speed at which the error budget is being consumed.

---

!!! sota "State of the Art & Resources (2026)"
    Reliability engineering for LLM systems has rapidly matured from adapting classical SRE practices to a discipline in its own right: production experience now shows that quality SLOs, multi-window burn-rate alerting, and multi-provider circuit-breaker failover are table-stakes for any customer-facing deployment. The 2025–2026 literature provides the first large-scale empirical taxonomies of real LLM inference incidents.

    **Foundational work**

    - [Beyer et al., *Site Reliability Engineering* (2016)](https://sre.google/sre-book/table-of-contents/) — Google's canonical SRE text; chapters on SLOs, error budgets, and incident management are free online.
    - [Beyer, Murphy et al., *The Site Reliability Workbook* (2018)](https://sre.google/workbook/table-of-contents/) — hands-on companion covering SLO implementation and error budget policy; free online.
    - [Google SRE, *Alerting on SLOs* (SRE Workbook Chapter 5)](https://sre.google/workbook/alerting-on-slos/) — the definitive treatment of multi-window, multi-burn-rate alerting that the chapter's burn-rate formulas are drawn from.

    **Recent advances (2023–2026)**

    - [Ranganathan, Zhang, Wu, *Enhancing Reliability in AI Inference Services* (2025)](https://arxiv.org/abs/2511.07424) — empirical taxonomy of 156 high-severity LLM production incidents at Microsoft; identifies dominant failure modes and mitigation strategies.
    - [Rabanser et al., *Towards a Science of AI Agent Reliability* (2026)](https://arxiv.org/abs/2602.16666) — ICML 2026 paper evaluating 15 models across consistency, robustness, predictability, and safety; shows capability gains have not translated into proportional reliability gains.
    - [Altenbernd, Wiesner, Kao, *Exploring Silent Data Corruption as a Reliability Challenge in LLM Training* (2026)](https://arxiv.org/abs/2604.00726) — examines hardware-induced silent faults that bypass system-level detection, relevant to the "silent quality collapse" failure mode.

    **Open-source & tools**

    - [traceloop/openllmetry](https://github.com/traceloop/openllmetry) — OpenTelemetry-based instrumentation for LLM pipelines; provides standard span attributes for model calls, prompt versions, and retrieval stages across 10+ providers.
    - [BerriAI/litellm](https://github.com/BerriAI/litellm) — Python SDK and proxy gateway (49k+ stars) unifying 100+ LLM providers with built-in fallbacks, circuit-breaker-style cooldowns, and per-provider spend tracking.
    - [LangSmith](https://www.langchain.com/langsmith) — trace-level observability platform for LLM agents; supports online evaluations, quality scoring, and PagerDuty/webhook alerting on production traces.

    **Go deeper**

    - [OpenTelemetry, *LLM Observability with OpenTelemetry* (2024)](https://opentelemetry.io/blog/2024/llm-observability/) — official CNCF guide to instrumenting LLM applications with traces and metrics using the emerging semantic conventions for GenAI.
    - [Portkey, *Retries, Fallbacks, and Circuit Breakers in LLM Apps* (2024)](https://portkey.ai/blog/retries-fallbacks-and-circuit-breakers-in-llm-apps/) — practical production guide distinguishing when to use each resilience primitive in an LLM gateway.

## Further Reading

- Beyer, Jones, Petoff, Murphy. *Site Reliability Engineering.* Google, O'Reilly, 2016. The foundational SRE text; chapters on SLOs, error budgets, and incident management remain the gold standard.
- Beyer, Murphy, et al. *The Site Reliability Workbook.* Google, O'Reilly, 2018. Practical implementation of SLOs, multi-window burn rate alerting, and error budget policy.
- Kleppmann, M. *Designing Data-Intensive Applications.* O'Reilly, 2017. Chapter on reliability covers circuit breakers, timeouts, and fallback patterns applicable to LLM gateways.
- Brewer, E. "Kubernetes and the Path to Cloud Native." SOSP 2019. Discusses graceful degradation at scale.
- LangSmith (LangChain). Observability and tracing for LLM pipelines — a practical reference for trace schema design. langchain.com/langsmith.
- OpenLLMetry / Traceloop. OpenTelemetry-based instrumentation for LLM systems; provides standard span attributes for model calls, prompt versions, and retrieval stages.
- Ribeiro, Wu, Guestrin, Singh. "Beyond Accuracy: Behavioral Testing of NLP Models with CheckList." ACL 2020. Foundation for golden-set canary eval design — systematic behavioral test suites rather than held-out accuracy alone.
