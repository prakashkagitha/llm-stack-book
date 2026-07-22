"""
Runs the CPU-runnable Python code blocks from:
    content/12-production-mlops/03-caching-routing-cost.md

Blocks are copied faithfully (verbatim logic) and concatenated in document
order, with small glue/fixtures added so each block actually executes.

Tested blocks:  #0, #1, #3, #4, #5, #9 (bonus), #10
Skipped blocks:
    #2  (line ~191) -- Anthropic prompt-caching API call. needs-net.
    #6  (line ~405) -- loads a real vLLM engine + downloads a GPTQ
        checkpoint ("TheBloke/Llama-2-70B-Chat-GPTQ"). The task's heuristic
        flagged this as "CPU-runnable" but on inspection it needs both
        network access (HF hub download) and a GPU (vLLM engine init),
        and `vllm` is not in the allowed-imports list. Import is guarded;
        the engine is never constructed/called. SKIP(network,gpu).
    #7  (line ~434) -- calls `call_frontier_api`, an undefined network
        call wrapped in httpx exception handling. needs-net.
    #8  (line ~468) -- Anthropic Batches API polling loop. needs-net.

Bonus: block #9 (line ~535, InferenceWorker/SIGTERM handling) was listed
as a default-skip "needs-net" block by the heuristic, but on inspection it
performs no network or GPU calls at all -- it only registers a SIGTERM
handler and toggles a boolean. It is trivially CPU-safe, so it is tested
here as well for extra coverage.

Real bug found & fixed in the book source (mirrored here):
    The `InferenceWorker.handle_request` method (block #9) raised
    `ServiceUnavailableError(...)` without that exception ever being
    defined anywhere in the chapter -- a NameError waiting to happen the
    first time a request arrives while the worker is draining. Fixed by
    defining a minimal `ServiceUnavailableError(Exception)` class in the
    same code block, both in the .md source and mirrored below.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import signal
import sys
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

import numpy as np

# Optional third-party imports used by the *actual* book blocks -- guarded
# per the hard rules so the module still loads without them installed.
try:
    import redis
except Exception:  # pragma: no cover
    redis = None

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
except Exception:  # pragma: no cover
    LogisticRegression = None
    LabelEncoder = None

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None

try:
    import anthropic
except Exception:  # pragma: no cover
    anthropic = None

try:
    from vllm import LLM, SamplingParams
except Exception:  # pragma: no cover -- vllm not in allowed imports; guarded
    LLM = None
    SamplingParams = None


def _section(name: str) -> None:
    print(f"\n=== {name} ===")


# ============================================================================
# Block #0 (line ~50) -- Exact caching (Redis + SHA-256 key)
# ============================================================================
_section("Block #0: exact_cache_get / exact_cache_set")

# The book's block does:
#     client = redis.Redis(host="localhost", port=6379, decode_responses=False)
# A real redis.Redis() constructor is lazy (no I/O), but the very first
# `.get()`/`.setex()` call would open a TCP connection to localhost:6379,
# which is a network call and is forbidden/unavailable in CI. We substitute
# a tiny in-memory fake that implements just the .get/.setex surface the
# book's functions use, so the cache-key + JSON logic below (the actual
# point of the block) runs unmodified.
class _FakeRedisClient:
    def __init__(self, *args, **kwargs):
        self._store: dict[str, bytes] = {}

    def get(self, key):
        return self._store.get(key)

    def setex(self, key, ttl_seconds, value):
        if isinstance(value, str):
            value = value.encode("utf-8")
        self._store[key] = value
        return True


client = _FakeRedisClient(host="localhost", port=6379, decode_responses=False)


def _cache_key(model: str, messages: list[dict], params: dict) -> str:
    """
    Deterministic SHA-256 key over the full request.
    sort_keys=True ensures dict ordering never matters.
    """
    payload = json.dumps(
        {"model": model, "messages": messages, **params},
        sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")
    return "llm:exact:" + hashlib.sha256(payload).hexdigest()


def exact_cache_get(model, messages, params, ttl_seconds=86400 * 30):
    key = _cache_key(model, messages, params)
    blob = client.get(key)
    if blob is not None:
        return json.loads(blob)          # cache hit: no model call
    return None


def exact_cache_set(model, messages, params, response, ttl_seconds=86400 * 30):
    key = _cache_key(model, messages, params)
    # SETEX: set with expiry atomically
    client.setex(key, ttl_seconds, json.dumps(response))


# --- exercise it ---
_model = "gpt-4o-mini"
_messages = [{"role": "user", "content": "What is the capital of France?"}]
_params = {"temperature": 0.0, "max_tokens": 50}

assert exact_cache_get(_model, _messages, _params) is None  # miss before set

exact_cache_set(_model, _messages, _params, {"text": "Paris"})
hit = exact_cache_get(_model, _messages, _params)
assert hit == {"text": "Paris"}, f"expected exact-cache hit, got {hit}"

# Dict key order in `params` must not matter (sort_keys=True).
_params_reordered = {"max_tokens": 50, "temperature": 0.0}
hit2 = exact_cache_get(_model, _messages, _params_reordered)
assert hit2 == {"text": "Paris"}, "cache key should be order-independent"

print("exact cache hit:", hit)


# ============================================================================
# Block #1 (line ~109) -- SemanticCache (cosine-similarity linear scan)
# ============================================================================
_section("Block #1: SemanticCache")


class SemanticCache:
    """
    Minimal semantic cache using cosine similarity.
    In production, replace the linear scan with a FAISS/Qdrant ANN index.
    """
    def __init__(self, embed_fn, threshold: float = 0.93):
        self.embed_fn = embed_fn      # callable: str -> np.ndarray (unit-normed)
        self.threshold = threshold
        self.index: list[tuple[np.ndarray, dict]] = []  # (embedding, entry)

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))    # assumes unit-norm inputs

    def get(self, query: str) -> Optional[dict]:
        """Return the best cached entry if similarity >= threshold."""
        if not self.index:
            return None
        q_emb = self.embed_fn(query)
        best_score, best_entry = max(
            ((self._cosine(q_emb, emb), entry) for emb, entry in self.index),
            key=lambda x: x[0],
        )
        if best_score >= self.threshold:
            return best_entry          # cache hit
        return None

    def put(self, query: str, response: dict) -> None:
        """Store a new (query, response) pair."""
        q_emb = self.embed_fn(query)
        self.index.append((q_emb, {"query": query, "response": response}))


# A tiny, deterministic, CPU-only "embedding" fixture: hash the string into
# a fixed seed, draw a Gaussian vector from it, and unit-normalise. Identical
# strings therefore always embed identically (similarity 1.0), which is
# enough to exercise the cache's admit/reject logic without a real model.
def _toy_embed_fn(text: str, dim: int = 16) -> np.ndarray:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:4], "big")
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim)
    return v / np.linalg.norm(v)


sem_cache = SemanticCache(embed_fn=_toy_embed_fn, threshold=0.93)
assert sem_cache.get("cancel my subscription") is None  # empty index -> miss

sem_cache.put("cancel my subscription", {"text": "Here's how to cancel..."})
exact_hit = sem_cache.get("cancel my subscription")  # identical string -> sim=1.0
assert exact_hit is not None and exact_hit["response"] == {"text": "Here's how to cancel..."}

miss = sem_cache.get("what is the weather in Paris")  # unrelated random embedding
assert miss is None, "unrelated query should not exceed the 0.93 threshold"

print("semantic cache exact-string hit:", exact_hit["response"])


# ============================================================================
# Block #3 (line ~235) -- Sequential quality-based cascade
# ============================================================================
_section("Block #3: cascade / confidence_gate / length_gate")


@dataclass
class ModelTier:
    name: str
    call_fn: Callable[..., Awaitable[dict]]   # async function
    cost_per_1k_tokens: float                 # illustrative combined cost
    quality_gate: Callable[[dict], bool]      # returns True iff output is good enough


async def cascade(prompt: str, tiers: list[ModelTier]) -> dict:
    """
    Try each tier in order (cheapest first).
    Return the first response that passes its quality gate,
    or the last tier's response unconditionally.
    """
    for i, tier in enumerate(tiers):
        response = await tier.call_fn(prompt)
        is_last = (i == len(tiers) - 1)
        if is_last or tier.quality_gate(response):
            response["_tier_used"] = tier.name
            return response
    # unreachable, but satisfies type checker
    raise RuntimeError("Empty tier list")


# ------- Example quality gates -------

def confidence_gate(response: dict, min_logprob: float = -0.15) -> bool:
    """
    Accept the cheap model's output if average log-probability of
    output tokens is high (model is 'confident').
    Requires logprobs=True in the API call.
    """
    logprobs = response.get("logprobs", [])
    if not logprobs:
        return False
    avg = sum(logprobs) / len(logprobs)
    return avg >= min_logprob


def length_gate(response: dict, max_tokens: int = 200) -> bool:
    """
    Reject cheap model if it hit the token limit — likely incomplete.
    """
    return response.get("finish_reason") != "length"


# --- exercise the gates directly ---
assert confidence_gate({"logprobs": [-0.05, -0.02, -0.10]}) is True
assert confidence_gate({"logprobs": [-1.0, -2.0]}) is False
assert confidence_gate({}) is False
assert length_gate({"finish_reason": "stop"}) is True
assert length_gate({"finish_reason": "length"}) is False


# --- exercise the cascade end to end ---
async def _cheap_model(prompt: str) -> dict:
    return {"text": f"cheap answer to: {prompt}", "logprobs": [-0.01, -0.02, -0.03]}


async def _expensive_model(prompt: str) -> dict:
    return {"text": f"expensive answer to: {prompt}", "logprobs": [-0.5]}


tiers = [
    ModelTier("small", _cheap_model, cost_per_1k_tokens=0.002, quality_gate=confidence_gate),
    ModelTier("large", _expensive_model, cost_per_1k_tokens=0.020, quality_gate=confidence_gate),
]
cascade_result = asyncio.run(cascade("What's your refund policy?", tiers))
assert cascade_result["_tier_used"] == "small", "confident cheap tier should short-circuit"
print("cascade result:", cascade_result)

# A low-confidence cheap response should escalate to the large tier.
async def _unconfident_cheap_model(prompt: str) -> dict:
    return {"text": "uh...", "logprobs": [-2.0, -3.0]}

tiers_escalate = [
    ModelTier("small", _unconfident_cheap_model, cost_per_1k_tokens=0.002, quality_gate=confidence_gate),
    ModelTier("large", _expensive_model, cost_per_1k_tokens=0.020, quality_gate=confidence_gate),
]
escalated = asyncio.run(cascade("Complex legal question", tiers_escalate))
assert escalated["_tier_used"] == "large", "low-confidence cheap tier should escalate"
print("escalated cascade result:", escalated)


# ============================================================================
# Block #4 (line ~287) -- Classifier-based routing (embedding + logreg)
# ============================================================================
_section("Block #4: RoutingClassifier")


class RoutingClassifier:
    """
    Maps a query embedding to a model tier label.
    Train offline on cascade logs; serve online with ~1 ms latency.
    """
    def __init__(self, embed_fn, labels: list[str]):
        self.embed_fn = embed_fn
        self.enc = LabelEncoder().fit(labels)
        self.clf = LogisticRegression(max_iter=500, C=1.0)

    def fit(self, queries: list[str], tier_labels: list[str]) -> None:
        X = np.stack([self.embed_fn(q) for q in queries])
        y = self.enc.transform(tier_labels)
        self.clf.fit(X, y)

    def predict(self, query: str) -> tuple[str, float]:
        """Returns (tier_name, confidence)."""
        emb = self.embed_fn(query).reshape(1, -1)
        proba = self.clf.predict_proba(emb)[0]
        idx = int(np.argmax(proba))
        return self.enc.inverse_transform([idx])[0], float(proba[idx])


# Bootstrap a tiny labelled set (as the chapter suggests: bootstrap labels
# from cascade logs). Toy queries -> toy tier labels.
_train_queries = [
    "what are your business hours",
    "how do i reset my password",
    "what is your return policy",
    "can you explain the pricing tiers in detail for enterprise",
    "my invoice looks wrong and i need a detailed reconciliation",
    "walk me through migrating my whole account to a new region",
]
_train_labels = ["simple", "simple", "simple", "medium", "medium", "complex"]

router = RoutingClassifier(embed_fn=_toy_embed_fn, labels=["simple", "medium", "complex"])
router.fit(_train_queries, _train_labels)

tier, confidence = router.predict("what are your business hours")
assert tier in {"simple", "medium", "complex"}
assert 0.0 <= confidence <= 1.0
print(f"routed query -> tier={tier!r}, confidence={confidence:.3f}")


# ============================================================================
# Block #5 (line ~345) -- Speculative routing (parallel dispatch + cancel)
# ============================================================================
_section("Block #5: speculative_route")


async def speculative_route(query: str, small_fn, large_fn, gate_fn):
    """
    Fires both models concurrently. Returns the small model's output if it
    passes the gate; otherwise waits for (and returns) the large model.
    Cancels the large model task if small passes early.
    """
    small_task = asyncio.create_task(small_fn(query))
    large_task = asyncio.create_task(large_fn(query))

    # Await the small model first (it should finish sooner)
    small_resp = await small_task
    if gate_fn(small_resp):
        large_task.cancel()             # stop paying for large model
        try:
            await large_task            # let cancellation propagate cleanly
        except asyncio.CancelledError:
            pass
        return small_resp, "small"

    # Small model failed quality gate; wait for large model
    large_resp = await large_task
    return large_resp, "large"


async def _fast_small_fn(query: str) -> dict:
    return {"text": "fast answer", "logprobs": [-0.01, -0.02]}


async def _slow_large_fn(query: str) -> dict:
    await asyncio.sleep(0.05)  # simulate a slower expensive model
    return {"text": "slow but thorough answer", "logprobs": [-0.05]}


spec_resp, spec_source = asyncio.run(
    speculative_route("simple lookup query", _fast_small_fn, _slow_large_fn, confidence_gate)
)
assert spec_source == "small", "confident fast response should win and cancel the large task"
print("speculative_route result:", spec_resp, spec_source)


async def _unconfident_small_fn(query: str) -> dict:
    return {"text": "not sure", "logprobs": [-3.0]}


spec_resp2, spec_source2 = asyncio.run(
    speculative_route("hard query", _unconfident_small_fn, _slow_large_fn, confidence_gate)
)
assert spec_source2 == "large", "failed gate should fall through to the large model"
print("speculative_route fallback result:", spec_resp2, spec_source2)


# ============================================================================
# Block #9 (line ~535) -- Spot-preemption-aware InferenceWorker (bonus)
# ============================================================================
# Listed as a default-skip "needs-net" candidate, but it performs no network
# I/O at all -- only signal handling and a boolean flag. Trivially CPU-safe,
# so we test it. BUG FIX (mirrored in the .md): `ServiceUnavailableError` was
# referenced but never defined anywhere in the chapter; defined here.
_section("Block #9 (bonus): InferenceWorker SIGTERM draining")


class ServiceUnavailableError(Exception):
    """Raised when the worker is draining and cannot accept new requests."""


class InferenceWorker:
    def __init__(self):
        self.accepting_new = True
        self.in_flight = 0
        # Register spot preemption handler
        signal.signal(signal.SIGTERM, self._handle_preemption)

    def _handle_preemption(self, signum, frame):
        """
        Called ~30-120s before the spot instance is reclaimed.
        Stop accepting new requests; let in-flight ones complete.
        """
        print("SIGTERM received — draining inference worker", file=sys.stderr)
        self.accepting_new = False
        # Health check endpoint will start returning 503,
        # causing the load balancer to stop sending new requests.

    async def handle_request(self, request):
        if not self.accepting_new:
            raise ServiceUnavailableError("Worker is draining")
        self.in_flight += 1
        try:
            return await self._run_inference(request)
        finally:
            self.in_flight -= 1

    # Not in the book's snippet (would be the actual model call); tiny
    # fixture so handle_request is fully exercised end to end.
    async def _run_inference(self, request):
        return {"text": f"served: {request}"}


worker = InferenceWorker()
ok_resp = asyncio.run(worker.handle_request("hello"))
assert ok_resp == {"text": "served: hello"}
assert worker.in_flight == 0

# Simulate the preemption signal handler firing directly (avoids sending a
# real OS signal to the test process) and verify draining behaviour.
worker._handle_preemption(signal.SIGTERM, None)
assert worker.accepting_new is False
try:
    asyncio.run(worker.handle_request("too late"))
    raise AssertionError("expected ServiceUnavailableError while draining")
except ServiceUnavailableError:
    print("worker correctly rejected request while draining")


# ============================================================================
# Block #10 (line ~585) -- CostControlStack (composes all layers)
# ============================================================================
_section("Block #10: CostControlStack")


@dataclass
class CostControlStack:
    exact_cache: "ExactCache"
    semantic_cache: "SemanticCache"
    router: "RoutingClassifier"
    simple_model_fn: callable
    medium_model_fn: callable
    frontier_model_fn: callable

    async def handle(self, request: dict) -> dict:
        messages = request["messages"]
        query = messages[-1]["content"]          # last user turn

        # --- Layer 1: Exact cache ---
        hit = self.exact_cache.get(request)
        if hit:
            return {**hit, "_source": "exact_cache"}

        # --- Layer 2: Semantic cache ---
        hit = self.semantic_cache.get(query)
        if hit:
            return {**hit["response"], "_source": "semantic_cache"}

        # --- Layer 3: Route to cheapest capable model ---
        tier, confidence = self.router.predict(query)

        if tier == "simple":
            resp = await self.simple_model_fn(request)
        elif tier == "medium":
            resp = await self.medium_model_fn(request)
        else:  # "complex"
            resp = await self.frontier_model_fn(request)

        resp["_tier"] = tier
        resp["_router_confidence"] = confidence

        # --- Fill both caches for future requests ---
        self.exact_cache.set(request, resp)
        self.semantic_cache.put(query, resp)

        return resp


# The chapter names an "ExactCache" type but only ever defines the two
# free functions exact_cache_get/exact_cache_set (block #0), operating on
# a global `client`. CostControlStack expects an object with .get(request)
# / .set(request, response). This thin adapter is pure glue -- it does not
# alter the cached logic itself, which is the verbatim block-#0 code above.
class ExactCache:
    def get(self, request: dict) -> Optional[dict]:
        return exact_cache_get(request["model"], request["messages"], request.get("params", {}))

    def set(self, request: dict, response: dict) -> None:
        exact_cache_set(request["model"], request["messages"], request.get("params", {}), response)


async def _stack_simple_model(request: dict) -> dict:
    return {"text": "simple-tier answer"}


async def _stack_medium_model(request: dict) -> dict:
    return {"text": "medium-tier answer"}


async def _stack_frontier_model(request: dict) -> dict:
    return {"text": "frontier-tier answer"}


stack = CostControlStack(
    exact_cache=ExactCache(),
    semantic_cache=SemanticCache(embed_fn=_toy_embed_fn, threshold=0.93),
    router=router,  # trained in block #4
    simple_model_fn=_stack_simple_model,
    medium_model_fn=_stack_medium_model,
    frontier_model_fn=_stack_frontier_model,
)

request1 = {
    "model": "stack-demo",
    "messages": [{"role": "user", "content": "what are your business hours"}],
    "params": {"temperature": 0.0},
}

# First call: both caches miss -> routed through the classifier to a model.
resp1 = asyncio.run(stack.handle(request1))
assert "_source" not in resp1, "first call should not be served from cache"
assert resp1["_tier"] in {"simple", "medium", "complex"}
print("first stack call (cache miss, routed):", resp1)

# Second, identical call: exact cache should now hit.
resp2 = asyncio.run(stack.handle(request1))
assert resp2["_source"] == "exact_cache", f"expected exact-cache hit, got {resp2}"
print("second stack call (exact cache hit):", resp2)

# Third call: different request text but same wording -> semantic cache hit
# would require the same embedding; here we directly probe a *new* request
# whose query string matches an existing semantic index entry exactly
# (identical string -> cosine similarity 1.0 with the toy embed_fn), while
# varying the model/messages so the exact-cache key misses.
request3 = {
    "model": "stack-demo-v2",
    "messages": [{"role": "user", "content": "what are your business hours"}],
    "params": {"temperature": 0.0},
}
resp3 = asyncio.run(stack.handle(request3))
assert resp3["_source"] == "semantic_cache", f"expected semantic-cache hit, got {resp3}"
print("third stack call (semantic cache hit):", resp3)

print("\nAll tested blocks executed successfully.")
