"""
Runs the CPU-runnable Python blocks from
content/12-production-mlops/02-observability-llmops.md end-to-end.

Chapter blocks:
  #0 (line ~25)  trace_setup.py            -- TESTED (guarded: needs `opentelemetry`)
  #1 (line ~51)  rag_pipeline.py           -- SKIP(network): calls a real llm_client.chat.completions.create
  #2 (line ~135) metrics.py                -- TESTED (guarded: needs `prometheus_client`)
  #3 (line ~203) prometheus_alerts.yml     -- SKIP(non-python): YAML config, not executable Python
  #4 (line ~265) log_schema.py             -- TESTED (guarded: needs `pydantic`)
  #5 (line ~324) production_eval_sampler.py-- TESTED (stdlib only)
  #6 (line ~361) async_eval_worker.py      -- SKIP(network): calls AsyncOpenAI().chat.completions.create
  #7 (line ~432) drift_detection.py        -- TESTED (numpy + sklearn)
  #8 (line ~498) ab_router.py              -- TESTED (stdlib only)
  #9 (line ~555) langfuse_integration.py   -- SKIP(network): calls Langfuse()/openai client against real services

"Guarded" blocks use libraries that are NOT in the CI-guaranteed set
(numpy, torch-cpu, einops, sklearn, stdlib). They are imported inside a
try/except at module scope; if the library is missing the block's code is
still defined verbatim but simply not exercised, and the test reports a
SKIP for that block instead of failing the whole file.
"""

import sys
import time
import json
import random
import hashlib
import asyncio
import uuid
from dataclasses import dataclass
from typing import Optional, List, Callable
from datetime import datetime, timezone

import numpy as np
from sklearn.decomposition import PCA

RESULTS = []  # (block_id, status, note)


def report(block_id, status, note=""):
    RESULTS.append((block_id, status, note))
    print(f"[{status}] block #{block_id}: {note}")


# ---------------------------------------------------------------------------
# Optional third-party libraries used by some blocks. Guard at module scope
# per the hard rules -- the file must still import cleanly without them.
# ---------------------------------------------------------------------------
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    OTEL_AVAILABLE = True
except Exception:
    trace = TracerProvider = BatchSpanProcessor = OTLPSpanExporter = None
    OTEL_AVAILABLE = False

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    PROMETHEUS_AVAILABLE = True
except Exception:
    Counter = Histogram = Gauge = start_http_server = None
    PROMETHEUS_AVAILABLE = False

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except Exception:
    BaseModel = object
    Field = None
    PYDANTIC_AVAILABLE = False


# ===========================================================================
# Block #0 (line ~25) -- trace_setup.py
# ===========================================================================
if OTEL_AVAILABLE:
    # trace_setup.py  —  one-time bootstrap for an LLM microservice
    # Requirements: opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc
    def setup_tracing(service_name: str, otlp_endpoint: str = "http://localhost:4317") -> "trace.Tracer":
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

    # Constructing the exporter/provider is local object setup (no network
    # I/O happens until spans are actually flushed by the batch processor),
    # so this is safe to execute on CPU without network access.
    TRACER = setup_tracing("llm-chat-service")
    assert TRACER is not None
    with TRACER.start_as_current_span("smoke_span") as span:
        span.set_attribute("test", True)
    report(0, "RAN", "setup_tracing() built a TracerProvider and emitted a span")
else:
    report(0, "SKIP(missing-dep)", "opentelemetry not installed; trace_setup.py defined-not-run")


# ===========================================================================
# Block #1 (line ~51) -- rag_pipeline.py
# SKIP(network): handle_request() calls llm_client.chat.completions.create(),
# a real OpenAI-compatible network call. Not included per task instructions.
# ===========================================================================


# ===========================================================================
# Block #2 (line ~135) -- metrics.py
# ===========================================================================
if PROMETHEUS_AVAILABLE:
    # metrics.py  —  Prometheus metrics for an LLM serving endpoint
    # Requirements: prometheus_client

    # ── Counters (monotonically increasing) ──────────────────────────────
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

    # ── Histograms (track distributions, not just averages) ────────────────
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

    # ── Gauges (current snapshot) ───────────────────────────────────────────
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

    # NOTE: the book's `if __name__ == "__main__": start_http_server(9090)`
    # entry point is intentionally NOT invoked here -- it binds a real
    # network port, which is out of scope for a CPU logic test and was
    # already guarded behind `__main__` in the source (i.e. import-safe).

    record_llm_call(
        model="gpt-4o-mini",
        prompt_tokens=800,
        completion_tokens=300,
        latency_s=1.2,
        ttft_s=0.35,
        error=False,
    )
    record_llm_call(
        model="gpt-4o-mini",
        prompt_tokens=650,
        completion_tokens=120,
        latency_s=0.9,
        ttft_s=0.20,
        error=True,
    )
    assert REQUESTS_TOTAL.labels(model="gpt-4o-mini", status="ok")._value.get() == 1
    assert REQUESTS_TOTAL.labels(model="gpt-4o-mini", status="error")._value.get() == 1
    report(2, "RAN", "record_llm_call() updated counters/histograms for 2 synthetic requests")
else:
    report(2, "SKIP(missing-dep)", "prometheus_client not installed; metrics.py defined-not-run")


# ===========================================================================
# Block #3 (line ~203) -- prometheus_alerts.yml
# SKIP(non-python): a YAML alerting-rules config, not executable Python.
# ===========================================================================


# ===========================================================================
# Block #4 (line ~265) -- log_schema.py
# ===========================================================================
if PYDANTIC_AVAILABLE:
    # log_schema.py  —  Pydantic model for a structured LLM log record
    from pydantic import BaseModel as _PydBaseModel, Field as _PydField
    from typing import Optional as _Optional, List as _List
    from datetime import datetime as _datetime
    import uuid as _uuid

    class LLMLogRecord(_PydBaseModel):
        # Identity
        record_id:      str       = _PydField(default_factory=lambda: str(_uuid.uuid4()))
        trace_id:       str       # OTel trace ID — links to distributed trace
        session_id:     str       # Groups turns in a multi-turn conversation
        user_id_hashed: str       # SHA-256 of user ID — never store raw PII here

        # Request context
        timestamp_utc:  _datetime
        model:          str
        prompt_version: str       # e.g. "chat-v3.2" — MUST be versioned for regression analysis

        # Content — store in encrypted, access-controlled storage (not plaintext RDBMS)
        system_prompt:  _Optional[str]   = None   # May contain IP; log if needed for debugging
        user_message:   str
        assistant_reply: str

        # Usage
        prompt_tokens:      int
        completion_tokens:  int
        latency_ms:         float
        finish_reason:      str   # "stop", "length", "content_filter", etc.

        # Evaluation signals (filled in asynchronously)
        thumbs_up:          _Optional[bool]  = None   # User feedback if collected
        auto_eval_score:    _Optional[float] = None   # From async LLM-as-judge pipeline
        safety_flagged:     _Optional[bool]  = None   # From guardrail layer

        class Config:
            json_encoders = {_datetime: lambda v: v.isoformat()}

    hashed_user = hashlib.sha256(b"user-42").hexdigest()
    record = LLMLogRecord(
        trace_id=uuid.uuid4().hex,
        session_id="sess-001",
        user_id_hashed=hashed_user,
        timestamp_utc=datetime.now(timezone.utc),
        model="gpt-4o-mini",
        prompt_version="chat-v3.2",
        user_message="What is the capital of France?",
        assistant_reply="Paris.",
        prompt_tokens=42,
        completion_tokens=3,
        latency_ms=210.5,
        finish_reason="stop",
    )
    assert record.assistant_reply == "Paris."
    payload = record.model_dump_json() if hasattr(record, "model_dump_json") else record.json()
    assert "Paris" in payload
    report(4, "RAN", "LLMLogRecord instantiated and serialized to JSON")
else:
    report(4, "SKIP(missing-dep)", "pydantic not installed; log_schema.py defined-not-run")


# ===========================================================================
# Block #5 (line ~324) -- production_eval_sampler.py
# ===========================================================================
# production_eval_sampler.py
import random as _random_unused  # noqa: F401  (book re-imports `random`; already imported above)
from dataclasses import dataclass as _dataclass_unused  # noqa: F401 (already imported above)
from typing import Optional as _Optional_unused  # noqa: F401 (already imported above)

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

random.seed(0)
# failure_rate=1.0 -> always sampled
assert should_evaluate({"finish_reason": "error"}, POLICY) is True
assert should_evaluate({"finish_reason": "content_filter"}, POLICY) is True
# low_score_rate=0.0 policy -> never sampled for a low-score record
never_policy = SamplingPolicy(base_rate=0.0, failure_rate=0.0, new_prompt_rate=0.0, low_score_rate=0.0)
assert should_evaluate({"finish_reason": "stop", "auto_eval_score": 0.2}, never_policy) is False
assert should_evaluate({"finish_reason": "stop", "auto_eval_score": 0.9}, never_policy) is False
sampled = sum(should_evaluate({"finish_reason": "stop"}, POLICY) for _ in range(2000))
# base_rate=0.05 over 2000 draws should land in a sane band around 100
assert 40 < sampled < 200, f"unexpected sample count: {sampled}"
report(5, "RAN", f"should_evaluate() exercised across policy branches ({sampled}/2000 baseline sampled)")


# ===========================================================================
# Block #6 (line ~361) -- async_eval_worker.py
# SKIP(network): judge_single() calls AsyncOpenAI().chat.completions.create().
# ===========================================================================


# ===========================================================================
# Block #7 (line ~432) -- drift_detection.py
# ===========================================================================
# drift_detection.py  —  PSI on prompt embedding distributions
# (numpy / sklearn.decomposition.PCA already imported at module scope)

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
) -> "dict[str, float]":
    """Project embeddings to principal components, then compute PSI on each."""
    pca = PCA(n_components=n_components).fit(reference_embeddings)
    ref_proj = pca.transform(reference_embeddings)
    cur_proj = pca.transform(current_embeddings)

    results = {}
    for i in range(n_components):
        results[f"psi_pc{i}"] = compute_psi(ref_proj[:, i], cur_proj[:, i])
    results["psi_max"] = max(results.values())
    return results


rng = np.random.default_rng(0)
ref_embeddings = rng.normal(loc=0.0, scale=1.0, size=(300, 16)).astype(np.float32)
# "current" window: same distribution -> PSI should be small/stable
cur_embeddings_same = rng.normal(loc=0.0, scale=1.0, size=(150, 16)).astype(np.float32)
# "current" window: shifted distribution -> PSI should be larger
cur_embeddings_shifted = rng.normal(loc=2.5, scale=1.0, size=(150, 16)).astype(np.float32)

psi_same = compute_psi(ref_embeddings[:, 0], cur_embeddings_same[:, 0])
psi_shifted = compute_psi(ref_embeddings[:, 0], cur_embeddings_shifted[:, 0])
assert psi_same < psi_shifted, (psi_same, psi_shifted)

drift_same = monitor_embedding_drift(ref_embeddings, cur_embeddings_same, n_components=5)
drift_shifted = monitor_embedding_drift(ref_embeddings, cur_embeddings_shifted, n_components=5)
assert set(drift_same.keys()) == {"psi_pc0", "psi_pc1", "psi_pc2", "psi_pc3", "psi_pc4", "psi_max"}
assert drift_shifted["psi_max"] > drift_same["psi_max"]
report(7, "RAN", f"compute_psi/monitor_embedding_drift: psi_max same={drift_same['psi_max']:.4f} "
                  f"shifted={drift_shifted['psi_max']:.4f}")


# ===========================================================================
# Block #8 (line ~498) -- ab_router.py
# ===========================================================================
# ab_router.py  —  stateless traffic-split router for LLM variants
# (hashlib already imported at module scope)

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

# Determinism: same session_id always maps to the same variant.
for sid in ["user-1", "user-2", "user-3", "session-abc"]:
    v1 = route_request(sid)
    v2 = route_request(sid)
    assert v1 == v2
    assert v1 in VARIANTS

# Distribution sanity check across many synthetic sessions.
counts = {"control": 0, "treatment_A": 0, "treatment_B": 0}
n = 5000
for i in range(n):
    counts[route_request(f"session-{i}")] += 1
assert counts["control"] > counts["treatment_A"] and counts["control"] > counts["treatment_B"]
report(8, "RAN", f"route_request() deterministic + distribution ~{counts} over {n} sessions")


# ===========================================================================
# Block #9 (line ~555) -- langfuse_integration.py
# SKIP(network): run_rag()/observe() talk to a real Langfuse server and the
# OpenAI API via lf.get_openai_client().chat.completions.create().
# ===========================================================================


if __name__ == "__main__":
    print("\n=== Summary ===")
    for block_id, status, note in RESULTS:
        print(f"block #{block_id}: {status} -- {note}")
    failed = [r for r in RESULTS if r[1] not in ("RAN", "SKIP(missing-dep)")]
    if failed:
        print(f"\n{len(failed)} block(s) in an unexpected state.")
        sys.exit(1)
    print("\nAll tested blocks ran successfully (or honestly skipped due to missing optional deps).")
