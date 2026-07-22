"""
Executable proof-of-runnability test for:
content/12-production-mlops/08-reliability-engineering.md

Blocks tested (heuristically CPU-runnable):
  - block #1 (line ~194): PromptVersion / PromptRegistry (redis-backed prompt registry)
  - block #2 (line ~278): canary_eval_gate / mock_llm_call
  - block #5 (line ~464): QualityRecord / segment_quality_report
  - block #6 (line ~522): DegradationLevel / DegradationController / handle_request

Blocks explicitly SKIPPED (per chapter-provided heuristic classification):
  - block #0 (line ~100): needs network (provider status page fetch via httpx) -- also
    flagged needs-gpu by the heuristic; either way it performs a real outbound HTTP call
    and is not exercised here.
  - block #3 (line ~333): JSON trace schema, not a Python code block.
  - block #4 (line ~371): plain-text postmortem template, not a Python code block.
  - block #7 (line ~626): MultiProviderGateway -- makes real httpx calls to provider APIs
    (network required); not exercised.
  - block #8 (line ~758): retry_with_backoff -- designed to wrap a real network call
    (httpx.HTTPStatusError-driven retry loop against a live API); not exercised.
  - block #9 (line ~828): plain-text war-room checklist, not a Python code block.

The only third-party import in the tested blocks is `redis` (block #1). It is guarded per
the hard rules: if the real `redis` package is unavailable, a tiny in-memory stand-in
implementing exactly the subset of the client interface used by PromptRegistry (get, set,
lpush, pipeline/execute) is substituted so the block's own business logic (SHA hashing,
atomic deploy/rollback bookkeeping) still executes for real, offline.
"""

import sys
import traceback

# ============================================================================
# Book block #1 (chapter line ~194): Prompt versioning infrastructure
# ============================================================================

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Optional

try:
    import redis  # pip install redis
except Exception:
    # Minimal in-memory stand-in for the redis client, implementing only the
    # methods PromptRegistry actually calls: get, set, lpush, pipeline(...).execute().
    # This lets the book's real deploy/rollback/get_current logic run offline.
    class _FakePipeline:
        def __init__(self, store):
            self._store = store
            self._ops = []

        def set(self, key, value):
            self._ops.append(("set", key, value))
            return self

        def lpush(self, key, value):
            self._ops.append(("lpush", key, value))
            return self

        def execute(self):
            for op, key, value in self._ops:
                if op == "set":
                    self._store[key] = value
                elif op == "lpush":
                    self._store.setdefault(key, [])
                    self._store[key].insert(0, value)
            self._ops = []

    class _FakeRedisClient:
        def __init__(self):
            self._store = {}

        def get(self, key):
            return self._store.get(key)

        def set(self, key, value):
            self._store[key] = value

        def lpush(self, key, value):
            self._store.setdefault(key, [])
            self._store[key].insert(0, value)

        def pipeline(self, transaction=True):
            return _FakePipeline(self._store)

    class _FakeRedisModule:
        @staticmethod
        def from_url(url, decode_responses=True):
            return _FakeRedisClient()

    redis = _FakeRedisModule()


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


# ============================================================================
# Book block #2 (chapter line ~278): Canary eval gate on deploy
# ============================================================================

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


# ============================================================================
# Book block #5 (chapter line ~464): Segment-level quality report
# ============================================================================

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


# ============================================================================
# Book block #6 (chapter line ~522): Degradation ladder / circuit breaker
# ============================================================================

import threading
from enum import IntEnum


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


# ============================================================================
# Test driver: exercise each block's classes/functions with tiny fixtures
# ============================================================================

def test_block1_prompt_registry():
    registry = PromptRegistry(redis_url="redis://fake-test-host:6379")

    v1 = PromptVersion(
        template_id="customer-support",
        version="1.0.0",
        sha256=registry._compute_sha("You are a support agent.", "Help with: {issue}"),
        system_prompt="You are a support agent.",
        user_prompt_template="Help with: {issue}",
        deployed_at=time.time(),
        deployed_by="alice",
    )
    registry.deploy(v1)
    assert registry.get_current("customer-support").version == "1.0.0"

    v2 = PromptVersion(
        template_id="customer-support",
        version="1.1.0",
        sha256=registry._compute_sha("You are a friendly support agent.", "Help with: {issue}"),
        system_prompt="You are a friendly support agent.",
        user_prompt_template="Help with: {issue}",
        deployed_at=time.time(),
        deployed_by="bob",
    )
    registry.deploy(v2)
    current = registry.get_current("customer-support")
    assert current.version == "1.1.0"
    assert current.rollback_to == "1.0.0"

    rolled = registry.rollback("customer-support")
    assert rolled is not None
    assert rolled.version == "1.0.0"
    assert registry.get_current("customer-support").version == "1.0.0"

    print("[OK] block #1 PromptVersion/PromptRegistry: deploy + rollback verified")
    return registry, v2


def test_block2_canary_eval_gate(registry):
    async def _eval_fn(prompt: str, response: str) -> float:
        # Deterministic toy scorer: reward non-empty mock responses.
        return 1.0 if response else 0.0

    golden_cases = [
        {"input": {"issue": "cancel subscription"}, "expected_score": 1.0},
        {"input": {"issue": "reset password"}, "expected_score": 1.0},
        {"input": {"issue": "billing question"}, "expected_score": 1.0},
    ]

    candidate = PromptVersion(
        template_id="customer-support",
        version="1.2.0",
        sha256=registry._compute_sha("You are a support agent.", "Help with: {issue}"),
        system_prompt="You are a support agent.",
        user_prompt_template="Help with: {issue}",
        deployed_at=time.time(),
        deployed_by="carol",
    )

    passed = asyncio.run(
        canary_eval_gate(
            template_id="customer-support",
            new_version=candidate,
            eval_fn=_eval_fn,
            golden_cases=golden_cases,
            pass_threshold=0.92,
            registry=registry,
        )
    )
    assert passed is True
    assert registry.get_current("customer-support").version == "1.2.0"

    print("[OK] block #2 canary_eval_gate: gate passed and deployed on success")


def test_block5_segment_quality_report():
    records = [
        QualityRecord("t1", "qa", "en", "short", 0.98),
        QualityRecord("t2", "qa", "en", "short", 0.97),
        QualityRecord("t3", "qa", "en", "long", 0.60),
        QualityRecord("t4", "qa", "en", "long", 0.55),
        QualityRecord("t5", "summarization", "es", "medium", 0.99),
        QualityRecord("t6", "summarization", "es", "medium", 0.40),
    ]
    report = segment_quality_report(records, slo_threshold=0.95)

    # Worst segment (qa/en/long) should be sorted first.
    worst_key = next(iter(report))
    assert worst_key == ("qa", "en", "long")
    assert report[("qa", "en", "long")]["breaching"] is True
    assert report[("qa", "en", "long")]["count"] == 2
    assert report[("qa", "en", "short")]["breaching"] is False
    assert report[("qa", "en", "short")]["pass_rate"] == 1.0

    print("[OK] block #5 segment_quality_report:", report)


def test_block6_degradation_controller():
    ctrl = DegradationController(recovery_probe_interval=0.0)
    assert ctrl.level == DegradationLevel.NORMAL
    assert handle_request_with(ctrl, "hello") == "[RAG] hello"

    ctrl.degrade("provider latency spike")
    assert ctrl.level == DegradationLevel.RETRIEVAL_OFF
    assert handle_request_with(ctrl, "hello") == "[MODEL-ONLY] hello"

    ctrl.degrade("model quality regression")
    assert ctrl.level == DegradationLevel.MODEL_FALLBACK
    assert handle_request_with(ctrl, "hello") == "[SMALL-MODEL] hello"

    ctrl.degrade("cache-only fallback triggered")
    assert ctrl.level == DegradationLevel.CACHED_ONLY
    # lookup_cache always returns None in this stub, so we hit the graceful error path.
    assert handle_request_with(ctrl, "hello") == graceful_error_response()

    ctrl.degrade("total outage")
    assert ctrl.level == DegradationLevel.GRACEFUL_ERROR
    assert handle_request_with(ctrl, "hello") == graceful_error_response()

    # recovery_probe_interval=0.0 means recover() can step down immediately.
    recovered = ctrl.recover()
    assert recovered == DegradationLevel.CACHED_ONLY

    ctrl.reset()
    assert ctrl.level == DegradationLevel.NORMAL

    print("[OK] block #6 DegradationController: full ladder traversal verified")


def handle_request_with(ctrl: DegradationController, query: str) -> str:
    """
    Thin adapter mirroring the book's module-level handle_request(), but parameterized
    on a controller instance instead of the module-global `controller`, so the test can
    exercise the state machine at every ladder level deterministically without racing
    the recovery dwell-time timer on the shared global.
    """
    level = ctrl.level
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


def test_block6_module_level_handle_request():
    # Also exercise the book's actual module-level handle_request() / global `controller`
    # verbatim, at least at the NORMAL level (its default state).
    assert controller.level == DegradationLevel.NORMAL
    assert handle_request("ping") == "[RAG] ping"
    print("[OK] block #6 module-level handle_request(): verified at NORMAL level")


def main():
    failures = []

    try:
        registry, _ = test_block1_prompt_registry()
    except Exception:
        traceback.print_exc()
        failures.append("block1")
        registry = None

    if registry is not None:
        try:
            test_block2_canary_eval_gate(registry)
        except Exception:
            traceback.print_exc()
            failures.append("block2")
    else:
        failures.append("block2 (skipped: block1 setup failed)")

    try:
        test_block5_segment_quality_report()
    except Exception:
        traceback.print_exc()
        failures.append("block5")

    try:
        test_block6_degradation_controller()
        test_block6_module_level_handle_request()
    except Exception:
        traceback.print_exc()
        failures.append("block6")

    if failures:
        print(f"\nFAILED: {failures}")
        sys.exit(1)
    else:
        print("\nAll tested blocks executed successfully: #1, #2, #5, #6")


if __name__ == "__main__":
    main()
