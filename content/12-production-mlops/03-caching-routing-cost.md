# 12.3 Caching, Routing & Cost Control in Production

LLM inference is expensive. A single frontier-model call can cost anywhere from a fraction of a cent to several dollars depending on context length and model tier, and at even modest traffic (say, ten thousand daily active users sending five messages each) you are looking at meaningful infrastructure spend before you have written a line of business logic. This chapter is about systematically engineering that bill down — not by compromising quality, but by routing work to the cheapest system that can do it well, avoiding redundant computation wherever possible, and squeezing utilisation out of every GPU-second you pay for.

The techniques here sit at the intersection of distributed systems, economics, and ML: semantic caching, prompt-prefix caching, model routing cascades, speculative routing, quantised fallbacks, intelligent batching, and spot/preemptible GPU scheduling. We will cover the mechanism of each, when to reach for it, and how to wire them together into a coherent cost-control stack.

Cross-references: [Inference Economics: Latency, Throughput & Cost](../07-inference-serving/12-inference-economics.html) covers the per-token cost model; [Prefix Caching & KV-Cache Reuse](../07-inference-serving/07-prefix-caching.html) covers the low-level KV-cache reuse mechanism; [Continuous Batching & Request Scheduling](../07-inference-serving/02-continuous-batching.html) covers the scheduler side of batching; [Quantization I](../04-kernels-efficiency/07-quantization-ptq.html) and [Quantization II](../04-kernels-efficiency/08-quantization-formats-qat.html) cover the quantisation methods we invoke here as fallbacks.

---

## Why Does This Problem Exist?

Token economics follow a simple formula. Let $C_\text{input}$ and $C_\text{output}$ be per-token prices (in USD), and let a request have $n_p$ prompt tokens and $n_g$ generated tokens:

$$
\text{cost per request} = C_\text{input} \cdot n_p + C_\text{output} \cdot n_g
$$

Output tokens are typically 3–5× more expensive than input tokens because decoding is memory-bandwidth-bound and fundamentally sequential (see [The Anatomy of LLM Inference: Prefill, Decode & The KV Cache](../07-inference-serving/01-anatomy-inference.html)). For a 128-token system prompt plus 500-token user message feeding a 300-token response, and illustrative prices of \$0.003/1K input and \$0.015/1K output, one call costs roughly:

$$
\frac{628 \cdot 0.003 + 300 \cdot 0.015}{1000} \approx \$0.0064
$$

That is small. But at 50,000 calls per day the monthly bill is around USD 9,600 — and that is a single, modest product. Production applications routinely run at 10× to 100× that volume, and frontier models are significantly pricier. The levers are: (a) call fewer tokens, (b) reuse previously computed results, (c) route to a cheaper model when you can, (d) spread load to reduce idle GPU time.

{{fig:cost-request-anatomy-levers}}

---

## Exact Caching

The simplest optimisation is to remember the answer to a query you have already answered. If you can guarantee that two requests are byte-for-byte identical, you can return the cached response with zero model compute.

### What to key on

The cache key must cover everything that would change the model's output:

- Model ID (and version/commit, not just name)
- Full serialised messages array (role + content)
- Sampling hyperparameters (temperature, top-p, top-k, max tokens)
- Any injected system prompt variables

A common mistake is to key only on the user message and miss that the system prompt varies per tenant, producing cross-tenant cache poisoning.

### Storage and eviction

Redis with a TTL is the standard choice. A SHA-256 hash of the canonicalised request body fits in 32 bytes; the response blob is typically 1–10 KB. With a 90-day TTL and 50,000 RPD the steady-state working set is on the order of a few hundred megabytes — trivially cacheable.

```python
import hashlib, json, redis

client = redis.Redis(host="localhost", port=6379, decode_responses=False)

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
```

Exact caching has a narrow hit rate for conversational workloads (few requests are truly identical) but extremely high value for specific patterns: FAQ bots, templated document generation, CI/CD pipeline prompts, and embedding calls (which are purely deterministic). For embeddings, exact caching can eliminate 60–90% of API calls if users resubmit the same documents.

### Handling non-determinism

If `temperature > 0`, returning a cached stale response is semantically wrong for freshness-sensitive queries. A common compromise is to cache only when `temperature == 0`, or to cache with a short TTL (a few hours) to capture burst traffic while not serving stale creative content for long.

---

## Semantic Caching

Exact caching misses near-duplicate queries. Semantic caching embeds the user query and checks whether the embedding is close enough to a previously answered query that the cached answer is still valid.

{{fig:cost-semantic-cache-flow}}

### Similarity threshold selection

Let $\hat{q}$ be the unit-normalised query embedding and $\hat{c}$ be a cached entry's embedding. The cosine similarity is $s = \hat{q} \cdot \hat{c}$. You admit a cache hit when $s \geq \tau$.

Choosing $\tau$ is a calibration problem. Too low: you return wrong answers (a FAQ about "cancel subscription" matches "delete my account" — maybe OK) or dangerously wrong ones ("What is the dosage of aspirin" matches "What is the dosage of ibuprofen" — very much not OK). Too high: the hit rate collapses toward exact matching.

A practical approach:

1. Collect a held-out set of (query, gold answer) pairs from your domain.
2. For each $\tau \in [0.85, 0.99]$ sweep, measure precision (fraction of returned hits that are semantically correct) and recall (fraction of queries served from cache).
3. Pick the $\tau$ that hits your precision floor (typically 0.97+) while maximising recall.

For general chat bots a threshold near 0.92–0.95 is common. For medical, legal, or financial applications a higher bar (0.97+) or a human-review loop on borderline hits is appropriate.

{{fig:cost-semantic-threshold-tradeoff}}

```python
import numpy as np
from typing import Optional

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
```

### Embedding model selection and latency

The embedding call itself introduces latency. A 100 ms embedding call eats into the savings if the cached path is supposed to be fast. Use a small, locally-hosted embedding model (e.g., a 22M-parameter sentence-transformer) so the embedding call completes in 1–5 ms on CPU. The ANN lookup in a vector store like Qdrant or FAISS is another 1–10 ms at typical scales. Compare that to a frontier model call at 500 ms to 5 s: the speedup is 100×.

Semantic caching integrates naturally with RAG systems (see [Retrieval-Augmented Generation Architectures](../09-rag-retrieval/03-rag-architectures.html)) — you can cache at both the retrieval step and the generation step.

---

## Prompt-Prefix Caching (Provider-Side KV Reuse)

Distinct from the application-level caches above, providers like Anthropic (prompt caching) and OpenAI (caching) offer server-side KV-cache reuse for repeated prompt prefixes. If you send a 2,000-token system prompt on every call, the provider can skip recomputing the key-value tensors for that prefix after the first request.

The economics are significant. Anthropic's prompt caching charges roughly 10% of the normal input price for cache-hit tokens (as of 2025). For a 2,000-token system prompt at \$0.003/1K tokens:

- Without caching: 2,000 tokens × \$0.003/1K = \$0.006 per call
- With caching (after first call): 2,000 tokens × \$0.0003/1K = \$0.0006 per call

At 10,000 calls per day this saves approximately USD 18 per day on the system prompt alone — over USD 6,500 per year.

!!! example "Worked example: prompt caching savings"

    Scenario: a coding assistant with a 4,000-token system prompt (instructions + code style guide) and an average 800-token user message generating 400-token responses. Traffic: 20,000 calls/day. Prices (illustrative): \$0.003/1K input, \$0.0003/1K cached input, \$0.015/1K output.

    **Without prompt caching:**

    $$
    \text{daily cost} = 20{,}000 \times \frac{4800 \times 0.003 + 400 \times 0.015}{1000}
    = 20{,}000 \times (0.0144 + 0.006) = 20{,}000 \times 0.0204 = \$408/\text{day}
    $$

    **With prompt caching** (system prompt hits cache 95% of the time):

    $$
    \text{daily cost} \approx 20{,}000 \times \frac{4000 \times 0.95 \times 0.0003 + 800 \times 0.003 + 400 \times 0.015}{1000}
    $$

    $$
    = 20{,}000 \times (0.00114 + 0.0024 + 0.006) = 20{,}000 \times 0.00954 = \$190.80/\text{day}
    $$

    Saving: roughly **\$217/day** or **\$79K/year** — just from restructuring your prompt.

To maximise prefix cache hits, keep the stable part of your prompt at the top (system instructions, few-shot examples, retrieved context) and put the variable part at the bottom (user message). This is covered in more depth in [Prefix Caching & KV-Cache Reuse](../07-inference-serving/07-prefix-caching.html) and [Context Engineering & Management](../08-agents-harness/04-context-engineering.html).

{{fig:cost-prefix-cache-reuse-layout}}

```python
# Anthropic prompt caching API usage (Python SDK, 2024+)
import anthropic

client = anthropic.Anthropic()

# Mark the static system prompt for caching.
# The provider will reuse KV tensors for this block across calls.
response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": STATIC_SYSTEM_PROMPT,          # 4000+ tokens
            "cache_control": {"type": "ephemeral"}, # request caching
        }
    ],
    messages=[
        {"role": "user", "content": user_message}  # variable part
    ],
)

# Inspect whether you got a cache hit
usage = response.usage
print(f"Input tokens: {usage.input_tokens}")
print(f"Cache read tokens: {usage.cache_read_input_tokens}")   # billed at 10%
print(f"Cache write tokens: {usage.cache_creation_input_tokens}")  # first call
```

---

## Model Routing and Cascades

Not every query needs your most capable (and most expensive) model. A cascade routes each request to the cheapest model that can answer it correctly, escalating to stronger models only when needed.

{{fig:cost-routing-cascade}}

The cascade can be implemented in two ways:

### 1. Quality-based escalation (sequential)

Call the cheap model first; if its output meets a quality gate, return it. Otherwise call the expensive model. This introduces latency for the escalated fraction, so it is best for workloads where most queries are simple (high hit rate on the cheap tier).

```python
import asyncio
from dataclasses import dataclass
from typing import Callable, Awaitable

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
```

### 2. Classifier-based routing (parallel or pre-dispatch)

Train a small classifier that predicts which model tier a query belongs to, and route before calling any LLM. This avoids the latency overhead of sequential calls but requires labelled training data (which you can bootstrap from cascade logs: label the query with the cheapest tier that gave good output in the sequential cascade).

```python
# Lightweight routing classifier using a small embedding model + logistic regression.
# In practice you might use a fine-tuned DistilBERT or even a rule-based system.

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

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
```

### Cascade economics

Let $p_s$ be the fraction of queries routed to the small model (hit rate), $c_s$ the small-model cost per query, and $c_l$ the large-model cost per query. Expected cost per query:

$$
\mathbb{E}[\text{cost}] = p_s \cdot c_s + (1 - p_s) \cdot (c_s + c_l)
= c_s + (1-p_s) \cdot c_l
$$

If the small model handles 70% of traffic (on the sequential cascade that also pays $c_s$ before escalating), and $c_s = \$0.002$, $c_l = \$0.020$:

$$
\mathbb{E}[\text{cost}] = 0.002 + 0.30 \times 0.020 = 0.002 + 0.006 = \$0.008
$$

Versus paying $c_l$ for everything: USD 0.020. A **2.5× cost reduction** for a 70% hit rate.

---

## Speculative Routing

Speculative routing is the routing analogue of speculative decoding (see [Speculative Decoding: Draft Models, Medusa, EAGLE & Lookahead](../07-inference-serving/06-speculative-decoding.html)). Instead of waiting for a sequential cascade decision, you fire both the cheap and expensive models in parallel and discard the expensive result if the cheap one passes the quality gate. This cuts latency to approximately the cheap model's latency for the common case, while guaranteeing large-model quality for the rest.

{{fig:cost-speculative-routing}}

The economics are worse than sequential cascade (you always pay both) unless the large model can be cancelled mid-generation when the small model succeeds. Streaming APIs with cancellation make this viable.

```python
import asyncio

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
```

Speculative routing works best when: (a) the cheap model is 3–10× faster than the expensive one, (b) the quality gate can be evaluated quickly (e.g., a short confidence check, not a slow LLM-as-judge), and (c) the cancellation saves meaningful tokens (longer outputs).

!!! interview "Interview Corner"

    **Q:** You are building a cost-optimised LLM API for a customer support product with 100K daily requests. The P95 latency must stay under 2 seconds. Describe the end-to-end cost control architecture you would design.

    **A:** Start with a layered cache stack: exact cache (Redis, SHA-256 key over full request) to handle repeated tickets, plus semantic cache (ANN index, threshold ~0.93) to catch near-duplicates — together these can serve 20–40% of traffic with no model call. Persist a static system prompt and FAQ context at the top of every prompt and enable provider-side prompt caching (Anthropic/OpenAI), saving 60–80% on that portion of input tokens.

    For uncached traffic, add a routing classifier: embed the query with a local 22M sentence-transformer, classify into "simple" (FAQ lookup, binary yes/no, short factual) vs. "complex" (multi-turn, policy edge cases, complaints). Route simple queries to a cheap 7B quantised (INT4) model hosted on spot instances, and complex queries to a frontier model. With a ~65% simple-route hit rate and a 10× cost gap between tiers, expected cost drops by ~6×.

    For spot/preemptible GPUs: run the cheap tier on spot instances with an on-demand fallback pool; statistically GPU preemptions are rare and requests can retry on the on-demand pool within the 2-second SLA.

    Finally, enable continuous batching on your inference server (vLLM or SGLang) to maximise GPU utilisation, target >80% GPU compute utilisation, and set up cost dashboards with per-tier, per-feature-flag breakdowns so you can detect regressions immediately.

---

## Quantised Fallbacks

Running a smaller quantised model is not just about model routing to a different API endpoint — you can also host a quantised version of the same model locally as a fallback that trades quality for cost and latency. The quantisation taxonomy is covered in depth in [Quantization I: Post-Training Quantization (GPTQ, AWQ, SmoothQuant)](../04-kernels-efficiency/07-quantization-ptq.html) and [Quantization II](../04-kernels-efficiency/08-quantization-formats-qat.html); here we focus on deployment economics.

### Memory and throughput impact

A 70B parameter model at FP16 requires approximately 140 GB of GPU memory (2 bytes per parameter). The same model quantised to INT4 (with GPTQ or AWQ) requires around 35 GB — fitting on a single 40 GB A100, versus four A100s for FP16. The cost impact of that memory reduction is dramatic:

| Quantisation | Memory (70B model) | Decode throughput (relative) | Quality loss (MMLU) |
|---|---|---|---|
| FP16 (baseline) | ~140 GB | 1.0× | 0% |
| INT8 (SmoothQuant) | ~70 GB | 1.3–1.5× | < 0.5% |
| INT4 (AWQ/GPTQ) | ~35 GB | 1.8–2.2× | 1–3% |
| INT4 + 2-bit outliers (QuIP#) | ~25 GB | similar to INT4 | 2–5% |

For a fallback tier receiving queries that were already routed away from the frontier model, the 1–3% MMLU degradation is often acceptable.

```python
# Loading a GPTQ-quantised model with vLLM for cost-effective fallback serving
from vllm import LLM, SamplingParams

# vLLM natively supports GPTQ/AWQ via the quantization parameter.
# This model fits on a single A100 (40GB) vs 4x A100 for FP16.
fallback_llm = LLM(
    model="TheBloke/Llama-2-70B-Chat-GPTQ",   # example quantised checkpoint
    quantization="gptq",
    dtype="float16",
    gpu_memory_utilization=0.92,               # leave 8% for KV cache headroom
    max_model_len=4096,
)

fallback_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
)

def run_fallback(prompts: list[str]) -> list[str]:
    outputs = fallback_llm.generate(prompts, fallback_params)
    return [o.outputs[0].text for o in outputs]
```

### When to invoke a quantised fallback

Quantised fallbacks fit into the routing cascade as the cheap-but-hosted tier, between the semantic cache and the frontier API call. You can also use them for latency degradation gracefully: when the frontier API returns a 429 rate-limit or a timeout, fall back to the local quantised model rather than returning an error to the user.

```python
import httpx

async def call_with_quantised_fallback(prompt: str) -> dict:
    try:
        # Attempt frontier model first
        resp = await call_frontier_api(prompt, timeout=3.0)
        return resp
    except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
        # Fall back to locally hosted quantised model
        text = run_fallback([prompt])[0]
        return {"text": text, "_source": "quantised_fallback"}
```

---

## Batching for Cost

Batching is the primary lever for maximising GPU utilisation and therefore amortising fixed GPU cost across more tokens. There are two distinct batching strategies relevant to cost control.

### Continuous (in-flight) batching

The inference server (vLLM, SGLang, TGI) uses continuous batching (also called iteration-level batching) to keep the GPU full across requests with different lengths. This is covered architecturally in [Continuous Batching & Request Scheduling](../07-inference-serving/02-continuous-batching.html); the cost angle is that higher effective batch size directly reduces cost per token:

$$
\text{cost per token} \approx \frac{\text{GPU-hour price}}{\text{tokens per GPU-hour}}
$$

At batch size 1 a modern GPU may generate on the order of 1,000–5,000 tokens/second for a 7B model. At batch size 32 the same GPU generates 10,000–30,000 tokens/second — a 5–6× throughput improvement for the same hardware cost. This is why GPU utilisation is the KPI: each percentage point of utilisation is free capacity.

### Offline batching for async workloads

Not all LLM workloads are latency-sensitive. Nightly report generation, bulk document summarisation, evaluation runs, and data labelling jobs can tolerate minutes of latency. For these, the cloud provider's batch inference API (Anthropic Batches API, OpenAI Batch API) offers a significant discount — on the order of 50% — in exchange for up to 24-hour turnaround.

```python
import anthropic, json, time

batch_client = anthropic.Anthropic()

def run_batch_job(requests: list[dict]) -> list[dict]:
    """
    Submit a batch of requests to Anthropic Batches API.
    Up to 50% cheaper; results available within 24 hours.
    """
    # Build the batch request list
    batch_requests = [
        {
            "custom_id": f"req-{i}",
            "params": {
                "model": "claude-opus-4-5",
                "max_tokens": 1024,
                "messages": req["messages"],
            },
        }
        for i, req in enumerate(requests)
    ]

    # Submit the batch
    batch = batch_client.messages.batches.create(requests=batch_requests)
    print(f"Batch created: {batch.id}, status: {batch.processing_status}")

    # Poll until complete (in production: use a webhook or async poller)
    while batch.processing_status == "in_progress":
        time.sleep(60)
        batch = batch_client.messages.batches.retrieve(batch.id)
        print(f"  Status: {batch.processing_status}, "
              f"succeeded: {batch.request_counts.succeeded}, "
              f"errored: {batch.request_counts.errored}")

    # Collect results
    results = []
    for result in batch_client.messages.batches.results(batch.id):
        if result.result.type == "succeeded":
            results.append({
                "custom_id": result.custom_id,
                "text": result.result.message.content[0].text,
            })
    return results
```

The batch API is the right choice for any pipeline that is not user-interactive: bulk summarisation, scheduled report generation, embedding generation for new documents, and offline evaluation suites.

---

## Spot and Preemptible GPUs

Cloud GPUs come in two flavours: on-demand (always available, full price) and spot/preemptible (heavily discounted — typically 60–80% cheaper — but can be reclaimed by the cloud provider with 30–120 seconds notice). For a well-designed LLM serving system, spot instances are highly tractable.

### Architecture for spot resilience


{{fig:cost-spot-resilience-fleet}}


Key design decisions:

1. **Stateless inference workers.** Each worker loads the model from shared storage (EFS, GCS) at startup; no local mutable state. Preemption loses nothing.
2. **Short request timeouts + retries.** Set a 10-second timeout per inference request. If a spot instance is reclaimed mid-request, the load balancer retries on another instance. For a 2-second P95 SLA, this is usually invisible.
3. **Keep a minimum on-demand floor.** Even 1 on-demand instance per deployment ensures some capacity remains during a spot shortage.
4. **Preemption signal handling.** Cloud providers send a SIGTERM (GCP) or a metadata flag (AWS) 30–120 seconds before reclamation. The worker should stop accepting new requests and drain in-flight ones.

```python
import signal, sys

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
```

### Spot instance cost modelling

Suppose you have a serving system requiring 4 GPU-hours per hour at steady state. On-demand A10G: on the order of USD 1.50/GPU-hour; spot A10G: on the order of USD 0.45/GPU-hour (70% discount). A mixed fleet of 75% spot, 25% on-demand costs:

$$
\text{cost/hour} = 4 \times (0.75 \times 0.45 + 0.25 \times 1.50) = 4 \times (0.3375 + 0.375) = 4 \times 0.7125 = \$2.85
$$

Versus all on-demand at USD 6.00/hour. A **2.1× cost reduction** purely from instance type selection, with near-transparent resilience.

---

## Putting It All Together: A Production Cost-Control Stack

The techniques above are not independent — they compose into a layered stack, and the order matters.


{{fig:cost-control-stack-layers}}


```python
import asyncio
from dataclasses import dataclass
from typing import Optional

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
```

### Monitoring and continuous improvement

No cost-control stack is set-and-forget. Wire up the following metrics:

- **Cache hit rate** (exact and semantic, separately) — alert if it drops, investigate if query distribution shifted.
- **Tier distribution** (fraction of traffic to each model tier) — regressions in routing quality show up as unexpectedly high frontier usage.
- **Cost per request** — tagged by product feature, user cohort, model tier. Allows per-feature cost attribution.
- **Quality-gate pass rate** — tracks whether the cheap tier is regressing (model update, distribution shift).

See [Observability, Logging & LLMOps](../12-production-mlops/02-observability-llmops.html) for the observability infrastructure that powers these dashboards.

!!! warning "Cache invalidation pitfalls"

    Semantic caches can serve stale content when your knowledge base changes (new policy, product update). Always include a cache-invalidation hook in your content management system: when a document is updated, delete all semantic cache entries whose source documents include that document's ID. Exact caches keyed on model version should be flushed whenever you update the model.

!!! tip "Practitioner tip: bootstrap your routing classifier cheaply"

    You do not need labelled data from day one. Deploy the sequential cascade (small-then-large) for the first week, log which tier each request ended up using, and use that as noisy supervision. After 10,000–50,000 examples, train a routing classifier offline and A/B test it against the sequential cascade. Typical result: equal quality, 30–50% lower latency for the majority of traffic.

---

## Key Takeaways

!!! key "Key Takeaways"

    - Cost per request = $C_\text{input} \cdot n_p + C_\text{output} \cdot n_g$; output tokens cost 3–5× more — reduce generated length before reducing input length.
    - Exact caching (Redis + SHA-256) is zero-latency and zero-risk; it pays off most for embedding calls and templated workloads.
    - Semantic caching adds a 10–20 ms overhead but can serve 20–40% of conversational traffic; calibrate the cosine threshold against your domain's precision floor.
    - Provider-side prompt caching (Anthropic, OpenAI) can reduce input costs by up to 90% for repeated long prefixes; restructure prompts to put stable content first.
    - Model routing cascades (sequential or classifier-based) deliver 2–6× cost reductions by routing simple queries to cheap models; bootstrap the classifier from cascade logs.
    - Speculative routing fires cheap and expensive models in parallel and cancels the expensive one on early success — optimal when cheap model latency << expensive model latency and cancellation is cheap.
    - INT4 quantisation halves GPU memory relative to INT8 and reduces hardware cost 4× relative to FP16, with 1–3% quality loss on standard benchmarks — acceptable for a fallback tier.
    - Spot/preemptible GPUs provide 60–80% cost reduction; make workers stateless, handle SIGTERM gracefully, and keep a small on-demand floor.
    - Monitor cache hit rate, tier distribution, and cost per request continuously; routing quality degrades silently with distribution shift.

---

!!! sota "State of the Art & Resources (2026)"
    LLM cost control has matured into a well-structured engineering discipline: semantic caching, model-routing cascades, and provider-side KV-prefix caching can collectively cut production API spend by 50–90% without sacrificing quality. Open frameworks such as RouteLLM and GPTCache have made these techniques production-accessible, while research on non-prefix KV reuse (CacheBlend) and learned routers continues to push the frontier.

    **Foundational work**

    - [Chen et al., *FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance* (2023)](https://arxiv.org/abs/2305.05176) — introduces the LLM cascade framework and cost-quality trade-off analysis that underlies most routing systems.
    - [Jiang et al., *LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion* (2023)](https://arxiv.org/abs/2306.02561) — foundational ACL 2023 work on ranking and fusing outputs across model tiers.
    - [Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (2023)](https://arxiv.org/abs/2309.06180) — the vLLM/PagedAttention SOSP paper; explains why continuous batching and KV-cache management are the backbone of cost-efficient serving.

    **Recent advances (2023–2026)**

    - [Ong et al., *RouteLLM: Learning to Route LLMs with Preference Data* (2024)](https://arxiv.org/abs/2406.18665) — trained routers reduce costs by up to 85% while maintaining 95% of strong-model quality across MT-Bench, MMLU, and GSM8K.
    - [Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs* (2024)](https://arxiv.org/abs/2312.07104) — introduces RadixAttention for automatic KV-prefix reuse across structured programs and multi-turn conversations.
    - [Yao et al., *CacheBlend: Fast LLM Serving for RAG with Cached Knowledge Fusion* (2024)](https://arxiv.org/abs/2405.16444) — extends prefix caching to non-prefix RAG chunks, reducing time-to-first-token by 2–3× without quality loss.

    **Open-source & tools**

    - [zilliztech/GPTCache](https://github.com/zilliztech/GPTCache) — pluggable semantic cache library for LLM APIs; supports FAISS, Qdrant, and Milvus backends with drop-in LangChain/LlamaIndex integration.
    - [lm-sys/RouteLLM](https://github.com/lm-sys/routellm) — open-source routing framework from LMSYS; drop-in OpenAI-compatible client that redirects queries to cheap or strong models based on trained preference-data routers.
    - [sgl-project/sglang](https://github.com/sgl-project/sglang) — high-performance serving framework with RadixAttention prefix caching; achieves up to 6.4× higher throughput than baseline systems.

    **Go deeper**

    - [Anthropic Prompt Caching — official docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) — canonical reference for enabling server-side KV reuse on the Claude API, including pricing, TTL options, and cache breakpoint rules.
    - [RouteLLM: An Open-Source Framework for Cost-Effective LLM Routing — LMSYS Blog (2024)](https://www.lmsys.org/blog/2024-07-01-routellm/) — practical walkthrough of training, evaluating, and deploying LLM routers in production.

## Further Reading

- **Kang et al., "LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion," ACL 2023** — foundational work on ensembling and routing across LLMs.
- **Chen et al., "FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance," 2023** — introduces the LLM cascade framework and cost-quality trade-off analysis.
- **Vllm project (Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023)** — the continuous batching and KV-cache management paper underlying most open-source serving stacks.
- **Dao et al., "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning," ICLR 2024** — understanding IO-efficient attention is prerequisite to understanding why KV-cache reuse saves so much.
- **Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration," MLSys 2024** — the quantisation method most commonly used in quantised fallback deployments.
- **SGLang RadixAttention (Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs," 2024)** — prefix cache reuse at the serving-system level, complementing provider-side caching.
- **GPTCache (GitHub: zilliztech/GPTCache)** — open-source semantic cache library with pluggable vector stores and embedding backends, useful as a reference implementation.
