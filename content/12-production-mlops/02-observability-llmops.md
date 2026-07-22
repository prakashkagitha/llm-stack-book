# 12.2 Observability, Logging & LLMOps

Deploying a language model is not the finish line — it is the starting gun. In traditional software, you ship code, watch error rates, and deploy a hotfix. With LLMs the failure modes are subtler: the model might still answer every request with HTTP 200 while silently hallucinating, gradually drifting off-topic, accumulating prompt-injection vulnerabilities, or costing ten times what you budgeted. Observability — the practice of understanding a system's internal state from its external outputs — is the discipline that closes this loop.

This chapter covers the full LLMOps observability stack: how to trace individual requests, what metrics to collect and alert on, how to log prompts and responses safely, how to run evaluations inside production traffic, how to detect quality drift, and how to wire all of this into a continuous improvement flywheel. The tools we examine (Langfuse, Phoenix/Arize, Weights & Biases, Prometheus, OpenTelemetry) are illustrative; the concepts are portable.

For the upstream serving architecture that this observability layer wraps, see [Designing an LLM Serving System](../12-production-mlops/01-serving-system-design.html). For offline evaluation techniques such as LLM-as-a-judge, see [LLM-as-a-Judge & Automated Evaluation](../11-evaluation/02-llm-as-judge.html).

## The Three Pillars: Traces, Metrics, Logs

The canonical observability model from distributed systems gives us three signals. All three are necessary for LLMs; none alone is sufficient.

**Traces** capture the causal chain of a single request. For a RAG pipeline, a trace spans the user query, the embedding call, the vector-database retrieval, the prompt construction, the LLM call, and finally the response. Each step is a *span* with a start time, duration, input, and output. Traces answer "what happened to request X?"

**Metrics** are aggregated numerical time-series: token counts, latency percentiles, error rates, cost. Metrics answer "how is the system behaving overall right now?"

**Logs** are structured records of individual events. For LLMs the most important logs are the verbatim prompt and response strings, because the model's behavior is expressed in natural language that no metric can fully capture. Logs answer "what exactly did the model say?"

{{fig:obs-three-pillars-pipeline}}

## Distributed Tracing with OpenTelemetry

OpenTelemetry (OTel) is the vendor-neutral standard for generating and propagating traces, metrics, and logs. The Python SDK makes it straightforward to instrument an LLM service.

```python
# trace_setup.py  —  one-time bootstrap for an LLM microservice
# Requirements: opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

def setup_tracing(service_name: str, otlp_endpoint: str = "http://localhost:4317") -> trace.Tracer:
    """Configure an OTel TracerProvider that exports to any OTLP-compatible backend.
    
    Compatible backends include: Langfuse (via proxy), Jaeger, Grafana Tempo,
    Google Cloud Trace, Honeycomb, and Datadog.
    """
    provider = TracerProvider()
    exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    # BatchSpanProcessor buffers spans and sends them asynchronously
    # to avoid adding latency to the critical path.
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return trace.get_tracer(service_name)


TRACER = setup_tracing("llm-chat-service")
```

```python
# rag_pipeline.py  —  a fully-instrumented RAG request handler
import time
import hashlib
from opentelemetry import trace
from trace_setup import TRACER

def handle_request(
    user_id: str,
    query: str,
    retriever,
    llm_client,
    max_tokens: int = 512,
) -> dict:
    """
    Process one user query through a RAG pipeline, emitting OTel spans for
    every stage. The root span holds the full end-to-end latency.
    """
    # Root span for the entire request.
    with TRACER.start_as_current_span("rag_request") as root_span:
        # Attach stable, low-cardinality attributes for filtering in dashboards.
        root_span.set_attribute("user.id", user_id)
        root_span.set_attribute("query.length_chars", len(query))
        root_span.set_attribute("query.hash", hashlib.md5(query.encode()).hexdigest()[:8])

        # ── Stage 1: Retrieval ────────────────────────────────────────────────
        with TRACER.start_as_current_span("retrieval") as ret_span:
            t0 = time.perf_counter()
            docs = retriever.retrieve(query, top_k=5)
            ret_span.set_attribute("retrieval.latency_ms", (time.perf_counter() - t0) * 1000)
            ret_span.set_attribute("retrieval.num_docs", len(docs))
            ret_span.set_attribute("retrieval.top_score", docs[0]["score"] if docs else 0.0)

        # ── Stage 2: Prompt construction ─────────────────────────────────────
        with TRACER.start_as_current_span("prompt_build") as pb_span:
            context_text = "\n\n".join(d["text"] for d in docs)
            system_prompt = "You are a helpful assistant. Use the provided context."
            user_message = f"Context:\n{context_text}\n\nQuestion: {query}"
            pb_span.set_attribute("prompt.system_chars", len(system_prompt))
            pb_span.set_attribute("prompt.user_chars", len(user_message))

        # ── Stage 3: LLM inference ───────────────────────────────────────────
        with TRACER.start_as_current_span("llm_inference") as llm_span:
            t0 = time.perf_counter()
            response = llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
                max_tokens=max_tokens,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            usage = response.usage
            # LLM-specific span attributes are the most valuable for cost analysis.
            llm_span.set_attribute("llm.model",               "gpt-4o-mini")
            llm_span.set_attribute("llm.prompt_tokens",       usage.prompt_tokens)
            llm_span.set_attribute("llm.completion_tokens",   usage.completion_tokens)
            llm_span.set_attribute("llm.total_tokens",        usage.total_tokens)
            llm_span.set_attribute("llm.latency_ms",          elapsed_ms)
            llm_span.set_attribute("llm.finish_reason",       response.choices[0].finish_reason)

        answer = response.choices[0].message.content
        root_span.set_attribute("response.length_chars", len(answer))
        return {"answer": answer, "docs": docs}
```

The key design principle here: put *semantic attributes* on every span so you can slice metrics by model, user cohort, or prompt template later. Avoid putting raw prompt strings in span attributes (size limits + PII risk) — instead log them separately and link by trace ID.

## Token, Latency & Cost Metrics

Metrics are aggregates. We want to observe their distributions over time, not just averages. Use Prometheus-style histograms (or their equivalents) to track percentiles.

### The four core LLM metrics

| Metric | Unit | Why It Matters |
|---|---|---|
| Time to First Token (TTFT) | milliseconds | User-perceived responsiveness for streaming |
| Time per Output Token (TPOT) | ms / token | Streaming smoothness; bottlenecks in decode |
| Total request latency | milliseconds | End-to-end SLA |
| Token cost | USD / 1M tokens | Unit economics; budget guardrails |

TTFT is dominated by prefill and queue wait. TPOT is dominated by memory bandwidth during autoregressive decode. See [The Anatomy of LLM Inference: Prefill, Decode & The KV Cache](../07-inference-serving/01-anatomy-inference.html) for the underlying mechanics, and [Inference Economics: Latency, Throughput & Cost](../07-inference-serving/12-inference-economics.html) for the cost model in full detail.

```python
# metrics.py  —  Prometheus metrics for an LLM serving endpoint
# Requirements: prometheus_client
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# ── Counters (monotonically increasing) ──────────────────────────────────────
REQUESTS_TOTAL = Counter(
    "llm_requests_total",
    "Total LLM requests",
    ["model", "status"],          # labels allow slicing by model and outcome
)
TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "Total tokens processed",
    ["model", "direction"],       # direction = 'prompt' or 'completion'
)

# ── Histograms (track distributions, not just averages) ───────────────────────
TTFT_HISTOGRAM = Histogram(
    "llm_time_to_first_token_seconds",
    "Time to first token in seconds",
    ["model"],
    # Buckets tuned for typical LLM latencies: 50ms → 10s
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
)
LATENCY_HISTOGRAM = Histogram(
    "llm_request_latency_seconds",
    "End-to-end request latency",
    ["model"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)
COST_HISTOGRAM = Histogram(
    "llm_request_cost_usd",
    "Estimated cost per request in USD",
    ["model"],
    buckets=[0.0001, 0.001, 0.01, 0.05, 0.10, 0.50, 1.0],
)

# ── Gauges (current snapshot) ─────────────────────────────────────────────────
QUEUE_DEPTH = Gauge("llm_request_queue_depth", "Current pending request count")

# Pricing table (USD per 1M tokens); update as providers change rates
PRICING = {
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "gpt-4o":      {"prompt": 5.00, "completion": 15.0},
}

def record_llm_call(model: str, prompt_tokens: int, completion_tokens: int,
                    latency_s: float, ttft_s: float, error: bool = False):
    """Call this after every LLM API response to update all metrics."""
    status = "error" if error else "ok"
    REQUESTS_TOTAL.labels(model=model, status=status).inc()
    TOKENS_TOTAL.labels(model=model, direction="prompt").inc(prompt_tokens)
    TOKENS_TOTAL.labels(model=model, direction="completion").inc(completion_tokens)
    TTFT_HISTOGRAM.labels(model=model).observe(ttft_s)
    LATENCY_HISTOGRAM.labels(model=model).observe(latency_s)

    if model in PRICING:
        cost = (prompt_tokens / 1e6 * PRICING[model]["prompt"]
                + completion_tokens / 1e6 * PRICING[model]["completion"])
        COST_HISTOGRAM.labels(model=model).observe(cost)

if __name__ == "__main__":
    start_http_server(9090)   # Prometheus scrapes :9090/metrics
```

### Alerting thresholds (example rules)

```yaml
# prometheus_alerts.yml
groups:
  - name: llm_service
    rules:
      # Alert if median TTFT exceeds 2 seconds for 5 minutes
      - alert: LLMHighTTFT
        expr: histogram_quantile(0.50, rate(llm_time_to_first_token_seconds_bucket[5m])) > 2.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "LLM median TTFT is {{ $value | humanizeDuration }}"

      # Alert if error rate exceeds 1% over 5 minutes
      - alert: LLMHighErrorRate
        expr: >
          rate(llm_requests_total{status="error"}[5m])
          / rate(llm_requests_total[5m]) > 0.01
        for: 5m
        labels:
          severity: critical

      # Alert if hourly token cost is trending toward budget overage
      - alert: LLMCostBudgetWarning
        expr: sum(rate(llm_tokens_total{direction="completion"}[1h])) * 3600 * 0.60 / 1e6 > 100
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Projected hourly completion cost exceeds $100"
```

!!! example "Worked example: Cost and budget math"
    Suppose your application sends an average of **800 prompt tokens** and receives **300 completion tokens** per request, using `gpt-4o-mini` (USD 0.15/M prompt, USD 0.60/M completion).

    Cost per request:

    $$
    c = \frac{800}{10^6} \times 0.15 + \frac{300}{10^6} \times 0.60 = 0.00012 + 0.00018 = \$0.00030
    $$

    At 10 requests per second, that is:

    $$
    0.00030 \times 10 \times 3600 \times 24 = \$259.20 \text{ per day}
    $$

    If you switch to a model that is 40% cheaper on completion tokens (USD 0.36/M), and you shorten the average prompt by 200 tokens via better context management, daily cost becomes:

    $$
    \frac{600}{10^6} \times 0.15 + \frac{300}{10^6} \times 0.36 = 0.000090 + 0.000108 = \$0.000198
    $$

    That is USD 171.07/day — a 34% reduction. Tracking token distributions per-template makes these opportunities visible; without metrics you are flying blind.

## Prompt & Response Logging

Raw prompt and response logging is the highest-fidelity signal available, but also the most sensitive. The discipline is: log what you need, protect what you log, and delete what you no longer need.

### What to log

```python
# log_schema.py  —  Pydantic model for a structured LLM log record
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import uuid

class LLMLogRecord(BaseModel):
    # Identity
    record_id:      str       = Field(default_factory=lambda: str(uuid.uuid4()))
    trace_id:       str       # OTel trace ID — links to distributed trace
    session_id:     str       # Groups turns in a multi-turn conversation
    user_id_hashed: str       # SHA-256 of user ID — never store raw PII here

    # Request context
    timestamp_utc:  datetime
    model:          str
    prompt_version: str       # e.g. "chat-v3.2" — MUST be versioned for regression analysis

    # Content — store in encrypted, access-controlled storage (not plaintext RDBMS)
    system_prompt:  Optional[str]   = None   # May contain IP; log if needed for debugging
    user_message:   str
    assistant_reply: str

    # Usage
    prompt_tokens:      int
    completion_tokens:  int
    latency_ms:         float
    finish_reason:      str   # "stop", "length", "content_filter", etc.

    # Evaluation signals (filled in asynchronously)
    thumbs_up:          Optional[bool]  = None   # User feedback if collected
    auto_eval_score:    Optional[float] = None   # From async LLM-as-judge pipeline
    safety_flagged:     Optional[bool]  = None   # From guardrail layer

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
```

### Privacy and data-handling rules

1. **Hash or pseudonymize user IDs** before logging. Use a keyed HMAC so you can re-identify for debugging under a formal process, but the logs are not linkable without the key.
2. **Classify prompts by sensitivity tier.** A coding assistant's prompts are low sensitivity; a healthcare assistant's prompts may contain PHI and must never leave a HIPAA-compliant boundary.
3. **Define retention policies.** Seven days for debugging, 90 days for eval/training, one year for compliance. Automate deletion.
4. **Separate storage from compute.** Write logs to an append-only, encrypted object store (S3 + SSE-KMS, GCS) and query with Athena or BigQuery. Do not log to a relational database that is also serving production traffic.

!!! warning "Common pitfall: logging to stdout in production"
    Structured logs emitted to stdout get mixed with framework noise, may be truncated at 64 KB by your log collector, and — critically — are often streamed to a central logging system that security teams have not reviewed for PII. Always write LLM content logs to a dedicated, permission-controlled sink. Use the trace ID to correlate with the main log stream without duplicating sensitive data there.

## Evaluation in Production

Offline evaluation with held-out benchmarks tells you how a model performs on a fixed distribution. Production evaluation tells you how it performs on your actual users, who are always stranger than your benchmark dataset. The two must both be part of your release process.

{{fig:obs-eval-in-production-async-sampling}}

### Sampling strategy

You cannot judge every response with an LLM-as-judge (it is expensive and adds latency). Instead, sample strategically:

```python
# production_eval_sampler.py
import random
from dataclasses import dataclass
from typing import Optional

@dataclass
class SamplingPolicy:
    base_rate: float          # Fraction of all traffic to evaluate (e.g. 0.05 = 5%)
    failure_rate: float       # Fraction of error/flagged requests to evaluate (e.g. 1.0 = 100%)
    new_prompt_rate: float    # Fraction of requests using a new prompt template to evaluate
    low_score_rate: float     # Fraction of requests below a score threshold to evaluate

def should_evaluate(record: dict, policy: SamplingPolicy) -> bool:
    """Return True if this record should be sent to the async eval pipeline."""
    # Always evaluate failures and safety flags
    if record.get("finish_reason") in ("content_filter", "error"):
        return random.random() < policy.failure_rate
    # Evaluate new prompt versions at higher rate to catch regressions early
    if record.get("is_new_prompt_version", False):
        return random.random() < policy.new_prompt_rate
    # Evaluate previously low-scoring records (detected drift)
    if record.get("auto_eval_score", 1.0) < 0.5:
        return random.random() < policy.low_score_rate
    # Baseline random sample for steady-state tracking
    return random.random() < policy.base_rate

POLICY = SamplingPolicy(
    base_rate=0.05,
    failure_rate=1.00,
    new_prompt_rate=0.30,
    low_score_rate=0.50,
)
```

### Async LLM-as-judge evaluation pipeline

```python
# async_eval_worker.py
# Runs as a separate process / Cloud Function; reads from an eval queue
import asyncio
import json
from openai import AsyncOpenAI

client = AsyncOpenAI()

JUDGE_SYSTEM = """You are an expert evaluator. Score the assistant reply on:
1. Faithfulness (0-2): Does the reply contradict the provided context?
2. Relevance (0-2): Does the reply address the user's question?
3. Fluency (0-1): Is the reply grammatically correct?

Output ONLY valid JSON: {"faithfulness": X, "relevance": X, "fluency": X, "reason": "..."}"""

async def judge_single(record: dict) -> dict:
    """Call the judge model and parse the score, with a 30-second timeout."""
    prompt = (
        f"Context provided to assistant:\n{record.get('context', 'none')}\n\n"
        f"User question: {record['user_message']}\n\n"
        f"Assistant reply: {record['assistant_reply']}"
    )
    response = await asyncio.wait_for(
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature=0,      # Zero temperature for reproducibility
            max_tokens=256,
        ),
        timeout=30.0,
    )
    raw = response.choices[0].message.content
    scores = json.loads(raw)
    scores["composite"] = (scores["faithfulness"] + scores["relevance"] + scores["fluency"]) / 5.0
    scores["record_id"]  = record["record_id"]
    return scores

async def eval_batch(records: list[dict]) -> list[dict]:
    """Evaluate a batch of records concurrently (up to 20 in parallel)."""
    semaphore = asyncio.Semaphore(20)
    async def bounded(r):
        async with semaphore:
            try:
                return await judge_single(r)
            except Exception as e:
                return {"record_id": r["record_id"], "error": str(e), "composite": None}
    return await asyncio.gather(*[bounded(r) for r in records])
```

For a deep dive into LLM-as-judge methodology and bias mitigation, see [LLM-as-a-Judge & Automated Evaluation](../11-evaluation/02-llm-as-judge.html).

## Drift Detection

A model that performed well at launch can degrade silently as user behaviour evolves, upstream data sources change, or the base model provider silently updates their serving stack. There are three types of drift to monitor.

{{fig:obs-drift-detection-triptych}}

### Input drift (covariate shift)

Track statistics of the prompt distribution over time. If users start asking qualitatively different questions, your eval scores from an earlier period are no longer representative.

$$
\text{PSI}(P \| Q) = \sum_{i=1}^{N} (P_i - Q_i) \ln \frac{P_i}{Q_i}
$$

The Population Stability Index (PSI) measures how much a distribution $P$ (current) has shifted from a reference $Q$ (baseline). By convention: PSI < 0.1 is stable, 0.1–0.25 is moderate shift, > 0.25 is significant. Apply it to binned token count distributions, embedding dimensions, or topic proportions.

```python
# drift_detection.py  —  PSI on prompt embedding distributions
import numpy as np
from sklearn.decomposition import PCA

def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Population Stability Index on 1-D arrays.
    
    Typical usage: call this on the first principal component of prompt embeddings,
    comparing a rolling 24-hour window against the previous 7-day baseline.
    """
    # Use reference distribution to define bin edges (important: same bins for both)
    min_val = min(reference.min(), current.min())
    max_val = max(reference.max(), current.max())
    bins = np.linspace(min_val, max_val, n_bins + 1)

    ref_counts, _ = np.histogram(reference, bins=bins)
    cur_counts, _ = np.histogram(current,   bins=bins)

    # Add small epsilon to avoid division by zero or log(0)
    eps = 1e-6
    ref_pct = (ref_counts + eps) / (ref_counts.sum() + eps * n_bins)
    cur_pct = (cur_counts + eps) / (cur_counts.sum() + eps * n_bins)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def monitor_embedding_drift(
    reference_embeddings: np.ndarray,   # shape: (N_ref, D)
    current_embeddings:   np.ndarray,   # shape: (N_cur, D)
    n_components: int = 5,
) -> dict[str, float]:
    """Project embeddings to principal components, then compute PSI on each."""
    pca = PCA(n_components=n_components).fit(reference_embeddings)
    ref_proj = pca.transform(reference_embeddings)
    cur_proj = pca.transform(current_embeddings)

    results = {}
    for i in range(n_components):
        results[f"psi_pc{i}"] = compute_psi(ref_proj[:, i], cur_proj[:, i])
    results["psi_max"] = max(results.values())
    return results
```

### Output quality drift

Track your eval composite score as a rolling mean. Use a Page-Cusum or CUSUM change-point algorithm to detect a sustained downward shift that is not noise:

$$
S_n = \max\!\left(0,\; S_{n-1} + (x_n - \mu_0 - k)\right)
$$

where $\mu_0$ is the in-control mean quality score, $k$ is the allowance parameter (typically half the smallest shift to detect), and an alert fires when $S_n > h$ (a decision threshold, commonly set by ARL — average run length — analysis).

### Provider / model version drift

A model provider can silently change model behavior behind the same API endpoint (e.g. "gpt-4o-2024-05-13" vs a later snapshot). Pin model versions explicitly in your API calls. Add a canary that runs a fixed probe set once per hour and alerts if the judge score drops more than a threshold.

## A/B Testing and Canary Releases

LLMOps borrows traffic-splitting patterns from traditional MLOps but with LLM-specific wrinkles: there is no single scalar prediction to compare; quality is a distribution; and changes compound over multi-turn conversations.

### Traffic splitting

```python
# ab_router.py  —  stateless traffic-split router for LLM variants
import hashlib
from typing import Callable

# Variant config: (weight, handler_function)
# Weights must sum to 1.0.
VARIANTS = {
    "control":     (0.80, "handle_with_gpt4o_mini"),
    "treatment_A": (0.10, "handle_with_claude3_haiku"),
    "treatment_B": (0.10, "handle_with_gpt4o_mini_v2_prompt"),
}

def route_request(session_id: str) -> str:
    """
    Route a request to a variant using a deterministic hash.
    
    Using a hash of session_id (not a random draw) ensures that the SAME
    user always gets the SAME variant across multiple turns — critical for
    multi-turn conversation quality evaluation.
    """
    # MD5 is fine here; we need bucketing, not cryptographic security.
    bucket = int(hashlib.md5(session_id.encode()).hexdigest(), 16) % 1000

    cumulative = 0
    for variant_name, (weight, _handler) in VARIANTS.items():
        cumulative += int(weight * 1000)
        if bucket < cumulative:
            return variant_name
    return "control"   # fallback
```

### Statistical significance for quality metrics

Comparing eval scores across two variants with a t-test requires careful sample size planning. Use a two-sample t-test on per-session composite scores:

$$
t = \frac{\bar{X}_A - \bar{X}_B}{\sqrt{s_A^2/n_A + s_B^2/n_B}}
$$

A typical LLM quality score (0–1 range) has standard deviation on the order of 0.15–0.25. To detect a 5-percentage-point improvement ($\delta = 0.05$) at 80% power and $\alpha = 0.05$, you need roughly:

$$
n \approx \frac{2 \sigma^2 (z_{\alpha/2} + z_\beta)^2}{\delta^2} = \frac{2 \times 0.04 \times (1.96 + 0.84)^2}{0.0025} \approx 250 \text{ sessions per variant}
$$

At 1,000 sessions/day with 10% traffic in the treatment arm, that is 100 sessions/day per variant — plan for 2–3 days before making a decision.

!!! interview "Interview Corner"
    **Q:** You deploy a new prompt template and want to know if it improves response quality. How do you set up and interpret the A/B test?

    **A:** Route a small fraction of traffic (say 10%) to the new template, using session-level hashing for consistency within multi-turn conversations. Collect per-session composite quality scores from your async LLM-as-judge pipeline. After reaching the pre-determined sample size (calculated from expected effect size, variance, and desired power), run a two-sample t-test or Mann-Whitney U test (the latter is more robust to non-normal score distributions). Guard against multiple comparisons if you test several metrics — use Bonferroni correction or pre-register a primary metric. Also track secondary guardrails: latency, cost, error rate. Report lift as both absolute (e.g. +3.2 pp) and relative (+5.1%) with confidence intervals. Only ship if you see statistically significant improvement on the primary metric AND no degradation on guardrails.

## Langfuse and the LLMOps Ecosystem

Langfuse is an open-source LLM observability platform that provides a purpose-built UI for traces, evals, prompt versioning, and datasets. It exposes an OpenAI-compatible SDK decorator and a REST API.

```python
# langfuse_integration.py  —  SDK-level Langfuse tracing
# Requirements: langfuse openai
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
import openai

# Langfuse reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST from env
lf = Langfuse()

@observe(name="rag_pipeline")          # Creates a trace named "rag_pipeline"
def run_rag(query: str, user_id: str) -> str:
    # Update the current trace with metadata (user, session, tags)
    langfuse_context.update_current_trace(
        user_id=user_id,
        tags=["production", "rag-v2"],
    )

    # Retrieval step — appears as a child span
    docs = retrieve_documents(query)

    # LLM call — use the Langfuse-wrapped OpenAI client for automatic token counting
    client = lf.get_openai_client()     # wraps openai.OpenAI transparently
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": build_prompt(query, docs)}],
    )
    return response.choices[0].message.content


@observe(name="retrieve_documents")    # Child span for retrieval
def retrieve_documents(query: str) -> list:
    # ... vector DB call ...
    return []

def build_prompt(query, docs):
    return f"Context: {docs}\n\nQuestion: {query}"
```

Beyond Langfuse, the ecosystem includes:

- **Arize Phoenix**: Open-source, strong embedding visualization and drift monitoring; excellent for RAG systems.
- **Weights & Biases Weave**: Integrates naturally if you are already using W&B for training; supports traces and evals.
- **Helicone**: Proxy-based approach — zero code changes; good for quick cost/latency dashboards.
- **Datadog LLM Observability**: Managed, enterprise-grade; integrates with existing Datadog alerts.
- **OpenLIT**: OpenTelemetry-native SDK with native GPU metric collection, useful when running self-hosted inference.

For organizations running self-hosted inference with vLLM or SGLang, expose the OpenAI-compatible `/metrics` endpoint that these servers natively provide, then scrape with Prometheus. vLLM emits metrics including `vllm:num_requests_running`, `vllm:gpu_cache_usage_perc`, and `vllm:generation_tokens_total` — see [vLLM: Architecture, PagedAttention & Internals](../07-inference-serving/03-vllm-internals.html) for details.

## The LLMOps Lifecycle and Continuous Improvement

LLMOps is the practice of operating LLM-powered systems with the same rigor applied to software services. The lifecycle has five phases that cycle continuously.

{{fig:obs-llmops-lifecycle-loop}}

### Phase 1: Ship with a canary

Route 5–10% of traffic to the new version. Monitor error rate, latency P95, and eval composite score. Use automated rollback if any guardrail metric degrades beyond a threshold within the first hour.

### Phase 2: Observe continuously

The metrics and trace pipeline described above should be always-on. Build dashboards at three granularities: real-time (1-minute granularity, for incident response), daily (for business-level KPIs), and weekly (for trend and drift analysis).

### Phase 3: Evaluate a curated sample

Maintain a *golden dataset* of 200–500 representative requests with human-authored reference answers. Re-run this dataset against every new model version or prompt change before any canary. Supplement with production samples filtered by the sampling policy above. Track score over time in your experiment tracking tool.

### Phase 4: Improve — the data flywheel

Production logs are a gold mine. Low-rated responses (thumbs-down or low judge score) become negative examples. High-rated responses become positive examples. If your platform supports it (see [Data Flywheels & Continuous Improvement](../12-production-mlops/05-data-flywheel.html)), route these curated examples back into your fine-tuning pipeline.

Prompt improvements are often the fastest wins. Langfuse's prompt management lets you maintain versioned system prompts, A/B test them against production traffic, and roll back instantly — without a code deployment.

### Phase 5: Retrain or fine-tune

After accumulating sufficient labeled production data, fine-tune the model. Track the fine-tuning run in your experiment tracking tool (W&B, MLflow) with all hyperparameters and dataset hashes. Run your golden-dataset eval before and after, and require a statistically significant improvement on the primary metric to promote the new weights.

For the mechanics of efficient fine-tuning, see [PEFT I: LoRA, QLoRA, DoRA & The Adapter Family](../05-posttraining-alignment/03-peft-lora-qlora.html). For the full data-flywheel architecture, see [Data Flywheels & Continuous Improvement](../12-production-mlops/05-data-flywheel.html).

## What to Monitor and Alert On

Consolidate your alerting policy into four layers, from raw infrastructure to business impact:

| Layer | Metric / Signal | Alert Threshold (example) |
|---|---|---|
| **Infrastructure** | GPU utilization, memory, host errors | GPU util < 40% for 10 min (underloaded) or > 95% (saturated) |
| **Serving** | Request queue depth, TTFT P95, error rate | TTFT P95 > 3s for 5 min; error rate > 1% |
| **Quality** | Rolling eval composite score, safety flag rate | Score drop > 5 pp vs 7-day baseline; safety flags > 0.5% |
| **Cost** | Hourly USD spend, cost per DAU | Projected daily cost > 120% of budget |

For multi-turn agentic pipelines (see [The Agentic Loop: ReAct, Plan-Execute & Reflection](../08-agents-harness/02-agentic-loop.html)), add:
- **Tool call success rate**: fraction of tool calls that return valid results
- **Turn count distribution**: unusually high turn counts signal loops or confused planning
- **Context utilization**: fraction of context window used (approaching the limit is a risk signal)

For RAG systems (see [Retrieval-Augmented Generation Architectures](../09-rag-retrieval/03-rag-architectures.html)), add:
- **Retrieval relevance score**: low scores mean the retriever is not finding useful context
- **No-hit rate**: fraction of queries where no document score exceeds a minimum threshold

!!! tip "Practitioner tip: use composite burn-rate alerts, not fixed thresholds"
    Fixed thresholds (alert if score < 0.7) generate alarm fatigue when traffic is noisy. Instead, compute a *burn rate*: if your error budget allows 1% of responses to be low-quality, alert when the 1-hour burn rate exceeds 2x the budget rate. This is the SRE error-budget model applied to LLM quality, and it dramatically reduces false positives while catching genuine degradations faster.

{{fig:obs-burn-rate-vs-fixed-threshold}}

!!! sota "State of the Art & Resources (2026)"
    LLMOps observability has rapidly matured from ad-hoc logging into a principled discipline: OpenTelemetry GenAI semantic conventions now provide a vendor-neutral standard for LLM spans and metrics, while purpose-built platforms (Langfuse, Arize Phoenix, MLflow Tracing) offer production-grade trace storage, eval pipelines, and drift monitoring. The key challenge in 2025–2026 is scaling these practices to multi-agent systems with complex, nested execution graphs.

    **Foundational work**

    - [Sculley et al., *Hidden Technical Debt in Machine Learning Systems* (NeurIPS 2015)](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-syst) — the canonical argument for why monitoring and observability are non-negotiable in any ML system.
    - [Shankar et al., *Who Validates the Validators?* (2024)](https://arxiv.org/abs/2404.12272) — identifies "criteria drift" in LLM-as-judge pipelines and proposes a mixed-initiative approach to aligning automated evaluators with human preferences.

    **Recent advances (2023–2026)**

    - [OpenTelemetry GenAI Semantic Conventions — Spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — the emerging CNCF-backed standard for LLM span attributes (model, token counts, finish reason), now supported by Datadog, Google Cloud, AWS, and Azure.
    - [OpenTelemetry GenAI Semantic Conventions — Metrics](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/) — companion spec covering client-side metrics including time-to-first-token and throughput, enabling consistent dashboards across providers.
    - [Jain, *An Introduction to Observability for LLM-based Applications using OpenTelemetry* (2024)](https://opentelemetry.io/blog/2024/llm-observability/) — official OpenTelemetry blog walkthrough for instrumenting LLM apps with OTel and exporting to Prometheus and Jaeger.

    **Open-source & tools**

    - [langfuse/langfuse](https://github.com/langfuse/langfuse) — open-source LLM engineering platform (YC W23) providing OTel-native tracing, prompt versioning, evals, and datasets; the most widely adopted self-hostable LLMOps stack.
    - [Arize-ai/phoenix](https://github.com/Arize-ai/phoenix) — open-source AI observability platform with strong RAG evaluation, embedding drift visualization, and support for 40+ frameworks via OpenInference traces.
    - [openlit/openlit](https://github.com/openlit/openlit) — OpenTelemetry-native observability SDK that instruments 50+ LLM providers plus NVIDIA GPU metrics with one line of code; good for self-hosted inference.
    - [wandb/weave](https://github.com/wandb/weave) — Weights & Biases toolkit for GenAI tracing and evaluation; integrates naturally with W&B experiment tracking for teams already in that ecosystem.
    - [MLflow Tracing for LLM and Agent Observability](https://mlflow.org/docs/latest/genai/tracing/) — MLflow 3.0 adds fully OTel-compatible tracing for 50+ GenAI libraries, linking traces to code, data, and prompts; 100% open source and self-hosted.

    **Go deeper**

    - [Google SRE Workbook — Alerting on SLOs](https://sre.google/workbook/alerting-on-slos/) — the multi-window, multi-burn-rate alerting model that LLMOps quality alerting adapts; essential reading for setting alert thresholds that minimize alarm fatigue.
    - [Langfuse blog — OpenTelemetry for LLM Observability (2024)](https://langfuse.com/blog/2024-10-opentelemetry-for-llm-observability) — practitioner analysis of OTel adoption in LLMOps, including the Langfuse OTel Collector and why the industry is converging on OTel as the standard transport layer.

## Further Reading

- **Sculley et al., "Hidden Technical Debt in Machine Learning Systems" (NIPS 2015)** — the foundational paper on ML system complexity and why monitoring is non-negotiable.
- **Shankar et al., "Who Validates the Validators? Aligning LLM-Assisted Evaluation of LLM Outputs with Human Preferences" (2024)** — on the reliability and calibration of LLM-as-judge pipelines.
- **Langfuse** (github.com/langfuse/langfuse) — open-source LLM observability; read the architecture docs for a practical reference implementation.
- **Arize Phoenix** (github.com/Arize-ai/phoenix) — OTel-native tracing and embedding drift for LLMs and RAG.
- **Google SRE Book, Chapter 5: Eliminating Toil** (sre.google/sre-book) — the error-budget and burn-rate alerting model that LLMOps adapts.
- **Klaise et al., "Alibi Detect: Algorithms for Outlier, Adversarial and Drift Detection" (JMLR 2022)** — statistical toolkit for production drift detection including CUSUM and PSI.
- **OpenTelemetry LLM Semantic Conventions** (opentelemetry.io/docs/specs/semconv/gen-ai/) — the emerging standard for LLM span attributes, maintained by the OTel community.

!!! key "Key Takeaways"
    - LLM observability requires all three pillars: distributed **traces** (per-request causal chains), **metrics** (aggregated time-series for TTFT, tokens, cost), and **logs** (verbatim prompts and responses for qualitative debugging).
    - Use OpenTelemetry as the vendor-neutral instrumentation layer; export to any backend (Langfuse, Jaeger, Grafana Tempo, Datadog).
    - Track TTFT, TPOT, total latency, prompt/completion token counts, and estimated cost per request as your baseline metric set. Use Prometheus histograms, not averages.
    - Prompt/response logs are uniquely valuable and uniquely sensitive: hash user IDs, classify by data sensitivity, enforce retention policies, and store in access-controlled object storage — never in plaintext RDBMS.
    - Run eval-in-production with strategic sampling (100% of failures, 5–30% of new-template traffic, ~5% of baseline); use an async LLM-as-judge pipeline so eval does not add request latency.
    - Detect three kinds of drift: input distribution shift (PSI on prompt embeddings), output quality drift (CUSUM on eval scores), and provider model drift (hourly canary probes).
    - A/B test prompt and model changes with session-level hash routing for multi-turn consistency; pre-calculate required sample sizes before launch.
    - The LLMOps lifecycle is a closed loop: Ship → Observe → Evaluate → Improve → Retrain. Production logs feed the data flywheel that makes the next version better.
    - Prefer burn-rate alerts over fixed thresholds to reduce alarm fatigue while maintaining sensitivity to genuine degradation.

## Exercises

**1.** *(Conceptual — the three pillars.)* A teammate proposes cutting costs by dropping verbatim prompt/response **logs** and keeping only **metrics** (token counts, latency, error rate) and **traces** (per-stage spans with semantic attributes). Give two concrete failure modes from this chapter that this setup would fail to catch, and explain why. Separately, the chapter says to put *semantic attributes* on spans but to keep raw prompt strings out of span attributes — give the two reasons it cites.

??? note "Solution"
    Two failure modes that metrics + traces alone miss:

    - **Silent hallucination / off-topic drift.** The introduction stresses that a model can "answer every request with HTTP 200 while silently hallucinating." Metrics would show a healthy `status="ok"` counter, low error rate, and normal token counts; traces would show every span completing successfully. Nothing numeric reveals that the *content* is wrong. Only the verbatim response log — the pillar that answers "what exactly did the model say?" — surfaces it, typically via the async LLM-as-judge pipeline that reads `assistant_reply`.
    - **Prompt-template regressions / prompt-injection.** Because logs carry the `prompt_version` and the actual `user_message`/`assistant_reply`, they are what lets you do regression analysis after a template change or spot an injection attack in the wild. A latency histogram or a span duration cannot show that a new template started leaking the system prompt.

    In short, the chapter's framing is that all three pillars are necessary and none is sufficient; the log pillar is the only one that captures behavior "expressed in natural language that no metric can fully capture."

    Why keep raw prompt strings *off* span attributes: (1) **size limits** — span attributes are meant for stable, low-cardinality values and large strings blow past backend limits; (2) **PII risk** — traces fan out to observability backends that may not be reviewed for sensitive data. The chapter's guidance is to log prompts separately in an access-controlled sink and link them to the span by trace ID.

**2.** *(Quantitative — cost math.)* Your service averages **1,200 prompt tokens** and **400 completion tokens** per request on `gpt-4o-mini` (USD 0.15 per 1M prompt tokens, USD 0.60 per 1M completion tokens). (a) Compute the cost per request. (b) At a steady **5 requests/second**, compute the projected daily cost. (c) A context-management change trims the average prompt to **700 tokens** with completions unchanged. What is the new daily cost, and what percentage reduction is that?

??? note "Solution"
    (a) Cost per request, using $c = \frac{p}{10^6}\times 0.15 + \frac{k}{10^6}\times 0.60$ with $p=1200$, $k=400$:

    $$
    c = \frac{1200}{10^6}\times 0.15 + \frac{400}{10^6}\times 0.60 = 0.00018 + 0.00024 = \$0.00042
    $$

    (b) Requests per day $= 5 \times 3600 \times 24 = 432{,}000$. Daily cost:

    $$
    0.00042 \times 432{,}000 = \$181.44 \text{ per day}
    $$

    (c) New per-request cost with $p=700$:

    $$
    c' = \frac{700}{10^6}\times 0.15 + \frac{400}{10^6}\times 0.60 = 0.000105 + 0.00024 = \$0.000345
    $$

    New daily cost $= 0.000345 \times 432{,}000 = \$149.04$. Reduction:

    $$
    \frac{181.44 - 149.04}{181.44} = \frac{32.40}{181.44} \approx 0.1786 = 17.9\%
    $$

    The prompt is the only thing that changed, and prompt tokens are the cheaper direction here, so a 42% cut in prompt tokens buys only about an 18% cost cut — exactly the kind of per-template insight the chapter says token distributions make visible.

**3.** *(Quantitative — PSI drift.)* You bin prompt-length into three buckets. The 7-day **baseline** proportions are $Q = [0.40,\ 0.35,\ 0.25]$ and the current 24-hour window gives $P = [0.25,\ 0.35,\ 0.40]$. Using the chapter's definition $\text{PSI} = \sum_i (P_i - Q_i)\ln\frac{P_i}{Q_i}$, compute the PSI and classify the shift using the chapter's thresholds.

??? note "Solution"
    Term by term (natural logs):

    - Bin 1: $(0.25 - 0.40)\ln\frac{0.25}{0.40} = (-0.15)\ln(0.625) = (-0.15)(-0.4700) = 0.07050$
    - Bin 2: $(0.35 - 0.35)\ln\frac{0.35}{0.35} = 0 \times \ln(1) = 0$
    - Bin 3: $(0.40 - 0.25)\ln\frac{0.40}{0.25} = (0.15)\ln(1.6) = (0.15)(0.4700) = 0.07050$

    $$
    \text{PSI} = 0.07050 + 0 + 0.07050 = 0.1410
    $$

    Against the chapter's convention (PSI < 0.1 stable, 0.1–0.25 moderate, > 0.25 significant), $0.141$ falls in the **moderate shift** band. Note the symmetric mass swap from bin 1 to bin 3 produces two equal positive contributions — PSI is a sum of non-negative terms, so opposite-direction movements do not cancel.

**4.** *(Quantitative — A/B sample size.)* You want to detect a $\delta = 0.04$ (4-percentage-point) improvement in the per-session composite quality score, which has standard deviation $\sigma = 0.20$. Using the chapter's formula at 80% power ($z_\beta = 0.84$) and $\alpha = 0.05$ two-sided ($z_{\alpha/2} = 1.96$), compute the required sessions **per variant**. If you run 1,000 sessions/day total and put 10% of traffic in the treatment arm, how many days until you can decide?

??? note "Solution"
    Sample size per variant, using $n \approx \dfrac{2\sigma^2 (z_{\alpha/2}+z_\beta)^2}{\delta^2}$:

    $$
    n \approx \frac{2 \times (0.20)^2 \times (1.96 + 0.84)^2}{(0.04)^2}
      = \frac{2 \times 0.04 \times (2.80)^2}{0.0016}
      = \frac{2 \times 0.04 \times 7.84}{0.0016}
    $$

    $$
    = \frac{0.6272}{0.0016} = 392 \text{ sessions per variant}
    $$

    The treatment arm receives 10% of 1,000 sessions/day $= 100$ sessions/day. Time to reach 392:

    $$
    \frac{392}{100} = 3.92 \Rightarrow \approx 4 \text{ days}
    $$

    (The control arm at 80% traffic reaches 392 far sooner, so the treatment arm is the binding constraint.) Smaller effect sizes scale as $1/\delta^2$: halving the target effect to $\delta = 0.02$ would quadruple $n$ to about 1,568 per variant, roughly 16 days in the treatment arm — which is why the chapter insists on pre-calculating sample size before launch.

**5.** *(Implementation — output-quality drift detector.)* The chapter gives an *upper* CUSUM, $S_n = \max(0,\ S_{n-1} + (x_n - \mu_0 - k))$, which accumulates *upward* deviations. Output-quality drift is a sustained *drop*, so implement a **lower** CUSUM detector class that fires when the composite score falls. It should take $\mu_0$ (in-control mean), $k$ (allowance), and $h$ (decision threshold), accept one score at a time via `update(x)`, and return `True` on the update that first crosses $h$. Then trace it by hand on $\mu_0 = 0.80$, $k = 0.02$, $h = 0.10$ with the stream $[0.80,\ 0.78,\ 0.75,\ 0.74,\ 0.73]$.

??? note "Solution"
    To detect a downward shift, flip the sign of the deviation: accumulate $(\mu_0 - k - x_n)$, which grows when $x_n$ sits below $\mu_0 - k$.

    ```python
    # quality_cusum.py  --  lower CUSUM change-point detector for eval scores
    class QualityCUSUM:
        """Detect a sustained DROP in the rolling composite quality score.

        mu0 : in-control (baseline) mean score
        k   : allowance, typically half the smallest shift you want to detect
        h   : decision threshold (set via ARL analysis)
        """
        def __init__(self, mu0: float, k: float, h: float):
            self.mu0 = mu0
            self.k = k
            self.h = h
            self.s = 0.0
            self.alerted = False

        def update(self, x: float) -> bool:
            # Lower CUSUM: accumulate how far below (mu0 - k) each score falls.
            self.s = max(0.0, self.s + (self.mu0 - self.k - x))
            fired = self.s > self.h and not self.alerted
            if fired:
                self.alerted = True   # latch so we alert once per excursion
            return fired

        def reset(self):
            self.s = 0.0
            self.alerted = False
    ```

    Hand trace with $\mu_0 = 0.80$, $k = 0.02$, so each step adds $(0.78 - x_n)$, clamped at 0:

    | $n$ | $x_n$ | $0.78 - x_n$ | $S_n = \max(0, S_{n-1} + \cdot)$ | $S_n > h$? |
    |---|---|---|---|---|
    | 1 | 0.80 | $-0.02$ | $\max(0,\ 0 - 0.02) = 0$ | no |
    | 2 | 0.78 | $0.00$ | $\max(0,\ 0 + 0.00) = 0$ | no |
    | 3 | 0.75 | $+0.03$ | $\max(0,\ 0 + 0.03) = 0.03$ | no |
    | 4 | 0.74 | $+0.04$ | $\max(0,\ 0.03 + 0.04) = 0.07$ | no |
    | 5 | 0.73 | $+0.05$ | $\max(0,\ 0.07 + 0.05) = 0.12$ | **yes** |

    The detector fires on the 5th score, when $S_5 = 0.12 > h = 0.10$. A single low reading (e.g. one noisy $0.75$) never crosses $h$ on its own; the alert requires a *sustained* run below $\mu_0 - k$, which is exactly the noise-vs-signal separation the chapter wants from a change-point algorithm rather than a fixed threshold.

**6.** *(Implementation — burn-rate quality alert.)* Implement the practitioner tip's **burn-rate** alert to replace a fixed "score < 0.7" rule. Given an error budget that allows a fraction `budget` (e.g. 0.01 = 1%) of responses to be low-quality, and the counts from a rolling window, compute the burn rate and alert when it exceeds a multiplier (e.g. 2x). Then apply it to a 1-hour window with 5,000 responses of which 130 were flagged low-quality, at `budget = 0.01`, `multiplier = 2.0`.

??? note "Solution"
    The burn rate is the *observed* bad fraction divided by the *budgeted* bad fraction; a burn rate of 1 means you are consuming budget exactly as fast as allowed, and >1 means faster.

    ```python
    # burn_rate_alert.py  --  SRE-style quality burn-rate alert
    def quality_burn_rate(low_quality: int, total: int, budget: float) -> float:
        """Return the burn rate: observed bad fraction / budgeted bad fraction.

        low_quality : count of low-quality responses in the window
        total       : total responses in the window
        budget      : allowed fraction of low-quality responses (e.g. 0.01)
        """
        if total == 0:
            return 0.0
        observed_fraction = low_quality / total
        return observed_fraction / budget

    def should_alert(low_quality: int, total: int,
                     budget: float = 0.01, multiplier: float = 2.0) -> bool:
        """Fire when the window is burning budget faster than `multiplier`x."""
        return quality_burn_rate(low_quality, total, budget) > multiplier
    ```

    Applying it to the window:

    $$
    \text{observed fraction} = \frac{130}{5000} = 0.026, \qquad
    \text{burn rate} = \frac{0.026}{0.01} = 2.6
    $$

    Since $2.6 > 2.0$, `should_alert(...)` returns `True` — the service is burning its quality budget at 2.6x the sustainable rate. The advantage over a fixed threshold (which the chapter warns causes alarm fatigue): the same absolute count of bad responses raises no alarm when traffic is high and the fraction stays under budget, but triggers quickly when the *rate* of budget consumption spikes, catching genuine degradations faster while suppressing noise-driven false positives.
