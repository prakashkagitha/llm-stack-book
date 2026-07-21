# 12.1 Designing an LLM Serving System

A single model running under `vllm serve` on one GPU is a demo. A *serving system* is what stands between a million daily users and a fleet of GPUs: it terminates connections, authenticates and meters traffic, decides which replica of which model handles each request, protects the GPUs from overload, keeps tail latency inside a contractual budget, and scales the fleet up and down as load swings by 10x between 3 a.m. and 3 p.m. This chapter is about the *system* — the boxes and arrows around the model — and the quantitative reasoning that sizes each box.

It assumes you already understand what happens *inside* one engine. If prefill, decode, and the KV cache are not yet second nature, read [The Anatomy of LLM Inference: Prefill, Decode & The KV Cache](../07-inference-serving/01-anatomy-inference.html) first; the batching machinery comes from [Continuous Batching & Request Scheduling](../07-inference-serving/02-continuous-batching.html); the cost arithmetic comes from [Inference Economics: Latency, Throughput & Cost](../07-inference-serving/12-inference-economics.html). Here we zoom *out* and design the thing that wraps dozens of those engines into a service with an SLO. This chapter also doubles as a system-design interview script — see [ML System Design: A Framework](../interview/03-ml-system-design-framework.html) for the meta-framework and [ML System Design: Worked Cases](../interview/04-ml-system-design-cases.html) for adjacent cases.

## Why LLM Serving Is Not Just "Microservices With a GPU"

If you have served a REST API before, your instincts are mostly right and dangerously incomplete. Three properties make LLM serving a different animal.

**Requests are not uniform-cost.** A normal web request takes a few milliseconds and returns a fixed-size payload. An LLM request streams for *seconds to minutes* and its cost is dominated by an output length you do not know in advance. A 20-token classification and a 4,000-token essay enter the same queue. This means classical load balancers that distribute by request *count* will badly imbalance GPU *work*. You must balance by tokens, or better, by predicted compute.

**The unit of work is a long-lived, stateful stream.** Each generation holds a KV cache that grows with every token, pinning a slice of HBM (high-bandwidth memory) for the request's entire lifetime. You cannot freely move an in-flight request to another replica — its KV cache lives in one GPU's memory. Statelessness, the thing that makes web services trivially scalable, is gone.

**Latency has two numbers, not one.** Users perceive two distinct quantities: **TTFT** (time to first token), how long until words start appearing, and **TPOT / ITL** (time per output token / inter-token latency), how fast they then stream. These trade against each other and against throughput. A single "latency" SLO is meaningless; you need both.


{{fig:serving-ttft-tpot-timeline}}


The job of a serving system is to keep both of these numbers inside a budget while keeping the GPUs — the expensive part — busy. Everything below is in service of that tension.

## The Reference Architecture

Let us lay out the full stack top to bottom, then dissect each layer. Requests flow downward; tokens stream back upward.


{{fig:serving-reference-architecture}}


A few non-obvious points about this picture. The **gateway** is cheap, stateless, CPU-bound, and trivially horizontally scalable; you run many of them behind a standard L4/L7 load balancer. The **router** is the brain — it makes per-request placement decisions and is where most of the interesting policy lives. The **model pools** are groups of identical replicas of one model configuration; each replica is a full inference engine with its own continuous-batching scheduler and its own KV cache. The **autoscaler** watches metrics and changes the number of replicas per pool. Below we walk these in order.

## The Gateway: The Cheap, Stateless Front Door

The gateway does everything that is *not* model inference and *is* per-request bookkeeping. Keep it dumb and fast so it never becomes the bottleneck.

- **TLS termination** and HTTP/2 or gRPC handling. Streaming responses go out as Server-Sent Events (SSE) or chunked transfer; the gateway must support streaming without buffering the whole response.
- **Authentication and authorization**: validate API keys / OAuth tokens, resolve the caller's org and tier.
- **Rate limiting and quota**: per-key requests-per-minute *and* tokens-per-minute. LLM rate limits must be token-aware, because one request can be 1,000x another.
- **Usage metering**: count prompt and completion tokens for billing and for the data flywheel ([Data Flywheels & Continuous Improvement](../12-production-mlops/05-data-flywheel.html)).
- **Request validation and normalization**: enforce `max_tokens` caps, reject malformed JSON early, apply default sampling params.

A token-aware rate limiter is the one piece worth showing, because the naive "N requests per minute" limiter is wrong for LLMs. We use a token bucket keyed on *tokens*, refilling continuously.

```python
import time
import threading

class TokenBucketLimiter:
    """
    Token-aware rate limiter. The 'tokens' here are LLM tokens (prompt+completion),
    not generic request credits. We refill `rate` tokens/sec up to `capacity`.

    Each LLM request first RESERVES an estimate (prompt_len + max_new_tokens),
    then RECONCILES with the true completion length when the stream finishes.
    This prevents a flood of long-generation requests from blowing the budget.
    """
    def __init__(self, rate_per_sec: float, capacity: float):
        self.rate = rate_per_sec          # e.g. 50_000 tokens/sec for a tier
        self.capacity = capacity          # burst ceiling, e.g. 100_000 tokens
        self.tokens = capacity            # start full
        self.last = time.monotonic()
        self.lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last = now

    def try_reserve(self, estimated_tokens: float) -> bool:
        """Reserve up-front. Returns False (429 -> client backs off) if over budget."""
        with self.lock:
            self._refill()
            if self.tokens >= estimated_tokens:
                self.tokens -= estimated_tokens
                return True
            return False

    def reconcile(self, estimated_tokens: float, actual_tokens: float) -> None:
        """Refund or charge the difference once the true length is known."""
        with self.lock:
            self._refill()
            # If we over-estimated, give tokens back (capped at capacity).
            self.tokens = min(self.capacity, self.tokens + (estimated_tokens - actual_tokens))
```

The crucial idea is **reserve-then-reconcile**: because output length is unknown, you charge a pessimistic estimate up front (so a burst of long requests cannot overrun the budget) and refund the slack when the stream ends. This same pattern reappears in admission control and in cost accounting.

!!! tip "Practitioner tip"

    Put a hard server-side cap on `max_tokens` at the gateway and refuse requests that omit one or ask for absurd lengths. Unbounded generation is the single most common cause of a "healthy" cluster suddenly blowing its p99 — a handful of 16k-token requests can monopolize KV cache and starve everyone else. A cap is a one-line policy that prevents a class of incidents.

## The Router: Where the Intelligence Lives

The router answers two questions for every request: *which model* and *which replica of that model*. Both are policy decisions, and both matter for latency and cost.

### Model routing

Sometimes the caller names the model explicitly (`"model": "llama-3-70b"`) and routing is a lookup. More interesting is **policy routing**, where the system chooses the model: send easy queries to a cheap 8B model and hard ones to an expensive 70B, gated by a small classifier or by the prompt's characteristics. This *cascade* pattern is a major cost lever and is developed in depth in [Caching, Routing & Cost Control in Production](../12-production-mlops/03-caching-routing-cost.html); here we note only that the router is the natural home for it, and that any routing classifier must itself be cheap (sub-millisecond) or you have just added latency to every request.

### Replica selection: do not use round-robin

The seductive default is round-robin or random. For LLMs this is a mistake, because replicas are *stateful and unevenly loaded*. Two better signals:

1. **Least outstanding load.** Pick the replica with the fewest queued + running tokens (not requests). Because cost scales with tokens, balancing tokens balances work.
2. **Prefix-cache affinity.** If a request shares a long prefix (system prompt, few-shot examples, a document) with a recent request, routing it to the replica that already has those KV blocks cached turns an expensive prefill into a near-free cache hit. This is the central idea of [Prefix Caching & KV-Cache Reuse](../07-inference-serving/07-prefix-caching.html) and of SGLang's RadixAttention ([SGLang: RadixAttention & Structured Programs](../07-inference-serving/04-sglang-radixattention.html)).

These two goals conflict: affinity says "send it where the cache is," load-balancing says "send it where it's quiet." The router blends them. A simple, effective scheme is **power-of-two-choices with a cache bonus**: sample two candidate replicas at random, and pick the one with the lower *effective* load, where effective load subtracts a bonus for cached prefix overlap.

```python
import random
from dataclasses import dataclass, field

@dataclass
class ReplicaState:
    id: str
    queued_tokens: int = 0        # tokens waiting (prefill not yet started)
    running_tokens: int = 0       # tokens of active sequences (KV held)
    cached_prefixes: set = field(default_factory=set)  # hashes of cached prefix blocks

    def effective_load(self, req_prefix_hashes: set) -> float:
        # Base load = work in flight. Tokens are the right unit, not request count.
        load = self.queued_tokens + 0.5 * self.running_tokens
        # Cache bonus: each overlapping prefix block we DON'T have to recompute
        # is real prefill work saved, so we subtract it from effective load.
        overlap = len(self.cached_prefixes & req_prefix_hashes)
        CACHE_BLOCK_TOKENS = 16
        return load - overlap * CACHE_BLOCK_TOKENS

def pick_replica(replicas, req_prefix_hashes, d=2):
    """Power-of-d-choices, load- and cache-aware. O(d), no global scan needed."""
    candidates = random.sample(replicas, k=min(d, len(replicas)))
    return min(candidates, key=lambda r: r.effective_load(req_prefix_hashes))
```

Power-of-two-choices is a beautiful result from balls-into-bins theory: sampling *two* random replicas and picking the less loaded one reduces the maximum load from $\Theta(\log n / \log\log n)$ (pure random) to $\Theta(\log\log n)$ — an exponential improvement — while needing only local state for two replicas, not a global scan. It is the workhorse of large fleets precisely because it avoids the herd behavior and central-state bottleneck of "always pick the global minimum."

!!! warning "Common pitfall: the thundering herd to the least-loaded replica"

    If every router instance always sends to the *single* globally least-loaded replica, they all pick the same one simultaneously, overload it, then all stampede to the next — load oscillates instead of balancing. This is why power-of-*d*-choices (with randomization) beats "always pick the minimum" in a distributed router. Randomize, and never let all routers share one synchronous view of "the best" replica.

## SLOs: Defining "Fast Enough" Precisely

You cannot design a system without a target. For LLM serving the target is a set of **SLOs** (service-level objectives) on latency, stated as *percentiles*, because averages hide the tail that users actually feel.

The canonical four metrics:

| Metric | Definition | Why it matters |
|---|---|---|
| **TTFT** | enqueue → first output token | Perceived responsiveness; dominated by queue wait + prefill |
| **TPOT / ITL** | mean time between successive output tokens | Streaming smoothness; dominated by decode step time |
| **E2E latency** | enqueue → last token | $\approx \text{TTFT} + \text{TPOT}\times(N_{out}-1)$ |
| **Throughput** | tokens/sec across the fleet | Drives cost-per-token (the business metric) |

A real SLO names a percentile and a number, e.g. *"p50 TTFT ≤ 300 ms, p99 TTFT ≤ 1,000 ms, p90 TPOT ≤ 50 ms, for prompts ≤ 2k tokens."* The percentile matters enormously: the gap between p50 and p99 is almost entirely **queueing delay** under load, which is why batching and admission control (below) are SLO tools, not just throughput tools.

The fundamental tension is **throughput vs. latency**, mediated by batch size. Bigger batches amortize weight loads across more tokens, raising throughput (lower cost), but they make each decode step take longer (worse TPOT) and make a newly arrived request wait behind a larger in-flight batch (worse TTFT). Continuous batching ([Continuous Batching & Request Scheduling](../07-inference-serving/02-continuous-batching.html)) and chunked prefill ([Disaggregated Prefill/Decode & Chunked Prefill](../07-inference-serving/08-disaggregated-chunked-prefill.html)) exist to move this frontier outward, but the tradeoff never disappears. The serving system's batching *policy* is the knob that places you on the curve where your SLOs are met at the lowest cost.


{{fig:serving-throughput-tpot-frontier}}


## Queueing: Little's Law and Why p99 Explodes

Here is the single most useful piece of quantitative reasoning for a serving interview. Treat one model pool as a queueing system. **Little's Law** relates the three core quantities of any stable queue:

$$
L = \lambda \cdot W
$$

where $L$ is the mean number of requests in the system, $\lambda$ is the arrival rate (requests/sec), and $W$ is the mean time a request spends in the system. It holds for *any* stable system regardless of arrival or service distribution — which is what makes it so powerful for capacity planning.

Now define **utilization** $\rho = \lambda / (c\,\mu)$, where $c$ is the number of parallel "servers" (think: batch slots / replicas) and $\mu$ is the service rate per server. The brutal fact of queueing theory is that waiting time does not grow linearly as you approach saturation — it grows like $1/(1-\rho)$. For a simple M/M/1 model the mean time in system is:

$$
W = \frac{1}{\mu - \lambda} = \frac{1/\mu}{1 - \rho}
$$

{{fig:serving-queueing-cliff}}

As $\rho \to 1$, $W \to \infty$. This is the mathematical reason a cluster that looks "80% utilized and fine" falls off a cliff at 95%: the queueing term $1/(1-\rho)$ goes from $5$ to $20$ — a 4x latency increase from a 15-point utilization change. The tail percentiles explode even faster than the mean. **This is why you provision LLM clusters to run at 60–75% utilization, not 95%.** The headroom is not waste; it is the budget that keeps p99 bounded.

!!! example "Worked example: sizing a pool to a p99 TTFT SLO"

    **Goal.** Serve Llama-3-8B with **p99 TTFT ≤ 1.0 s** at a peak of **λ = 60 requests/sec**, average prompt = 800 tokens.

    **Step 1 — single-replica prefill capacity.** Suppose one A100 replica running vLLM prefills at a sustained ~10,000 tokens/sec (illustrative; measure yours). A request's prefill compute is $800$ tokens, so the *prefill service time* per request is

    $$
    \frac{800}{10{,}000} = 0.08\ \text{s}.
    $$

    Service rate per replica $\mu \approx 1/0.08 = 12.5$ requests/sec.

    **Step 2 — replicas for throughput alone.** Raw capacity needed: $\lambda/\mu = 60 / 12.5 = 4.8$, so **5 replicas** would handle the mean load. But at 5 replicas, utilization is $\rho = 60/(5 \times 12.5) = 0.96$ — deep in the danger zone.

    **Step 3 — size for the tail, not the mean.** Target $\rho \le 0.7$. Required replicas:

    $$
    c \ge \frac{\lambda}{0.7\,\mu} = \frac{60}{0.7 \times 12.5} = 6.86 \Rightarrow \textbf{7 replicas}.
    $$

    **Step 4 — sanity-check the tail.** With $c = 7$, the system behaves like an M/M/c with $\rho = 60/(7\times12.5)=0.686$. Mean queue wait is small (tens of ms); the p99 wait — roughly a few multiples of the mean service time at this $\rho$ — lands comfortably under the prefill+queue budget of 1.0 s. The 2 "extra" replicas beyond the throughput minimum are buying you the tail.

    **Takeaway:** the SLO, not the average load, sets the replica count. Sizing to the mean ($\rho \approx 0.96$) would *technically* keep up on average while violating p99 constantly.

The admission-control corollary: when a queue starts to build, it is usually better to **shed load fast** (return HTTP 429 so the client retries elsewhere or backs off) than to admit a request that will violate its SLO anyway *and* consume a KV slot that degrades everyone behind it. A queue that grows without bound is a worse outcome than a clean rejection.

```python
def admit(request, replica, slo_ttft_s=1.0, prefill_tok_per_s=10_000):
    """
    Predictive admission control. Estimate the TTFT this request WOULD see if
    admitted to `replica` right now; reject (429) if it can't meet the SLO.
    This protects the requests already in flight from a late-arriving straggler.
    """
    # Work ahead of us in the queue (tokens), plus our own prefill cost.
    work_ahead = replica.queued_tokens
    our_prefill = request.prompt_len
    predicted_ttft = (work_ahead + our_prefill) / prefill_tok_per_s
    if predicted_ttft > slo_ttft_s:
        return False   # shed load: 429, let the client retry / route elsewhere
    replica.queued_tokens += our_prefill
    return True
```

## Replica and Cluster Sizing: Memory Is the Real Constraint

Throughput sizing (above) tells you how many replicas you need for *speed*. A second, independent constraint sets how many concurrent requests *fit*: **HBM capacity**, almost entirely consumed by model weights plus KV cache. If you do not budget memory, the autoscaler will happily place a replica that OOMs on the first big batch.

The memory budget for one replica is:

$$
M_{\text{HBM}} = \underbrace{M_{\text{weights}}}_{\text{fixed}} + \underbrace{M_{\text{activations}}}_{\text{small, transient}} + \underbrace{M_{\text{KV}}}_{\text{scales with concurrency}\times\text{context}}
$$

The KV-cache term per token, for a model with $L$ layers, $H_{kv}$ key/value heads, head dimension $d_h$, in `bytes_per_elem` precision, storing both K and V, is:

$$
\text{bytes/token} = 2 \cdot L \cdot H_{kv} \cdot d_h \cdot \text{bytes\_per\_elem}
$$

The factor $H_{kv}$ (not the full head count $H$) is exactly why grouped-query attention matters so much for serving — see [Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html). Fewer KV heads means more concurrent requests fit. The number of requests you can serve concurrently is then the leftover memory divided by per-request KV footprint:

$$
N_{\text{concurrent}} \approx \frac{M_{\text{HBM}} - M_{\text{weights}} - M_{\text{reserve}}}{(\text{bytes/token}) \times \overline{L_{\text{ctx}}}}
$$

```python
def kv_bytes_per_token(num_layers, num_kv_heads, head_dim, bytes_per_elem=2):
    # 2 for K and V; bytes_per_elem=2 for fp16/bf16, 1 for fp8.
    return 2 * num_layers * num_kv_heads * head_dim * bytes_per_elem

def max_concurrent_requests(hbm_gb, weight_gb, reserve_gb,
                            num_layers, num_kv_heads, head_dim,
                            avg_ctx_tokens, bytes_per_elem=2):
    """How many simultaneous sequences fit in one replica's HBM."""
    free_bytes = (hbm_gb - weight_gb - reserve_gb) * 1e9
    per_tok = kv_bytes_per_token(num_layers, num_kv_heads, head_dim, bytes_per_elem)
    per_request = per_tok * avg_ctx_tokens
    return int(free_bytes // per_request)

# Llama-3-70B-ish on one 80GB H100 shard would not fit weights alone (140GB bf16),
# so 70B needs tensor parallelism. Let's size an 8B model on a single 80GB H100:
#   8B bf16 weights ~= 16 GB. L=32, num_kv_heads=8 (GQA), head_dim=128.
n = max_concurrent_requests(
    hbm_gb=80, weight_gb=16, reserve_gb=4,
    num_layers=32, num_kv_heads=8, head_dim=128,
    avg_ctx_tokens=2048, bytes_per_elem=2,
)
print(n)  # -> ~488 concurrent 2k-token sequences fit in KV cache
```

For a 70B model the weights alone (≈140 GB in bf16) exceed one 80 GB GPU, forcing **tensor parallelism** (TP) across GPUs — covered in [Multi-GPU & Multi-Node Inference](../07-inference-serving/11-multi-gpu-inference.html). The sizing rule then operates per-shard: with TP=4, each GPU holds a quarter of the weights and a quarter of each layer's KV, and the four GPUs together form *one replica*. PagedAttention ([PagedAttention & KV-Cache Memory Management](../04-kernels-efficiency/06-paged-attention-kv.html)) is what lets you actually pack memory to near this theoretical $N_{\text{concurrent}}$ instead of wasting it on fragmentation and worst-case pre-allocation.

!!! note "Aside: two sizing constraints, take the binding one"

    You now have two independent replica counts: one from *throughput* ($\lambda / (0.7\mu)$) and one from *concurrency/memory* ($N_{\text{requests}} / N_{\text{concurrent}}$). The real fleet size is the **maximum** of the two. Compute-bound workloads (long prompts, short outputs) are limited by prefill throughput; memory-bound workloads (many concurrent long-context chats) are limited by KV capacity. Know which regime you are in before you order GPUs.

## Batching Policy: The Throughput Engine

Inside each replica, the scheduler decides which sequences run in each forward pass. The serving system's job is to *configure* that policy, not reinvent it. Three levers, in increasing sophistication.

**Continuous (in-flight) batching.** Static batching — wait for $B$ requests, run them lockstep until all finish — wastes the GPU because short sequences sit idle waiting for the longest one. Continuous batching instead adds and evicts sequences *every decode step*: the moment one finishes, a waiting request takes its slot. This is the single biggest throughput win in modern serving and is the default in vLLM, SGLang, and TGI. The mechanism is detailed in [Continuous Batching & Request Scheduling](../07-inference-serving/02-continuous-batching.html); the serving system mainly tunes its admission limits (`max_num_seqs`, `max_num_batched_tokens`).

**Chunked prefill.** Prefill is compute-heavy and *bursty*; a single long prompt's prefill can stall the decode loop, spiking the TPOT of everyone else (a "decode stall"). Chunked prefill splits a long prefill into token-sized chunks and interleaves them with ongoing decode steps, smoothing TPOT at a small TTFT cost. See [Disaggregated Prefill/Decode & Chunked Prefill](../07-inference-serving/08-disaggregated-chunked-prefill.html).

**Prefill/decode disaggregation.** Because prefill is compute-bound and decode is memory-bandwidth-bound, they want different hardware and different batch sizes. Disaggregation runs them on *separate* replica pools and streams the KV cache between them. This eliminates prefill-vs-decode interference entirely at the cost of a KV transfer over the network — worth it at scale, overkill for small deployments.

```python
class BatchPolicy:
    """
    Serving-side knobs that bound a replica's continuous-batching scheduler.
    These are the parameters the autoscaler and SLO budget actually control.
    """
    def __init__(self,
                 max_num_seqs=256,           # concurrency cap (KV-memory bound)
                 max_num_batched_tokens=8192,# per-step token budget (TPOT bound)
                 enable_chunked_prefill=True,
                 prefill_chunk_size=512):     # split long prefills into 512-tok chunks
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.enable_chunked_prefill = enable_chunked_prefill
        self.prefill_chunk_size = prefill_chunk_size

    def step_budget(self, running_seqs, waiting_prefills):
        """
        Decide this step's work. Decode tokens are cheap (1 tok/seq); reserve the
        rest of the token budget for prefill chunks. This is the policy that
        trades TTFT (admit prefills) against TPOT (keep decode steps small).
        """
        decode_tokens = len(running_seqs)            # 1 token per running seq
        prefill_budget = self.max_num_batched_tokens - decode_tokens
        chunks = []
        for req in waiting_prefills:
            if prefill_budget <= 0 or len(running_seqs) >= self.max_num_seqs:
                break
            take = min(self.prefill_chunk_size, req.remaining_prefill, prefill_budget)
            chunks.append((req, take))
            prefill_budget -= take
        return decode_tokens, chunks
```

The deep lesson: **the batch policy is how you spend your latency budget.** A bigger `max_num_batched_tokens` raises throughput (cheaper) but lengthens each decode step (worse TPOT); enabling chunked prefill protects TPOT at the cost of slightly higher TTFT. There is no universally correct setting — there is only the setting that meets *your* SLO at the lowest cost, found by load-testing against your real traffic distribution.

## Autoscaling on GPUs: Scaling a Resource You Cannot Get Instantly

Autoscaling a stateless web service is easy: CPU goes up, add pods, pods are ready in seconds. GPU autoscaling for LLMs is hard for three reasons, and a good system designer names all three.

1. **Cold starts are minutes, not seconds.** A new replica must acquire a GPU node (possibly from a cloud quota or a cluster autoscaler that itself takes minutes), pull a multi-gigabyte container, load tens of gigabytes of weights from object storage into HBM, warm up CUDA graphs / `torch.compile`, and prime the KV allocator. A 70B replica can take several minutes to become ready.
2. **The scaling signal must be a leading indicator.** Scaling on GPU utilization is too late — by the time utilization saturates, the queue is already building and p99 is already violated. Scale on **queue depth / waiting-tokens** or on a **predicted-TTFT** signal, which rise *before* the SLO breaks.
3. **Scale-down must drain, not kill.** You cannot SIGKILL a replica with 200 in-flight streaming requests; their KV caches and partial generations vanish. Scale-down marks a replica as **draining** (router stops sending new work), waits for in-flight requests to finish (with a timeout), *then* releases the GPU.

Because cold starts are slow, the standard pattern is **predictive + buffered autoscaling**: keep a warm headroom of $k$ idle replicas (or a small pool of pre-warmed standby nodes) so demand spikes are absorbed instantly, while the slow path provisions more capacity in the background. The control law is a target-tracking loop on queue-derived load.

```python
import math

def desired_replicas(current_replicas,
                     waiting_tokens, running_tokens,
                     replica_token_capacity,   # tokens/sec one replica sustains
                     target_utilization=0.7,
                     warm_buffer=2,
                     min_replicas=2, max_replicas=64):
    """
    Target-tracking autoscaler driven by QUEUE load (a leading indicator),
    not GPU utilization (a lagging one).

    offered_load: tokens of work the fleet currently owes (queued + in-flight).
    We size so that this load sits at `target_utilization` of total capacity,
    then add a warm buffer to absorb spikes during the slow cold-start window.
    """
    offered_load = waiting_tokens + running_tokens
    total_capacity_needed = offered_load / target_utilization
    raw = math.ceil(total_capacity_needed / replica_token_capacity)
    desired = raw + warm_buffer
    # Hysteresis: clamp and let the caller apply scale-down delay separately.
    return max(min_replicas, min(max_replicas, desired))
```

Two operational guardrails make this stable in practice. **Hysteresis / asymmetric timing**: scale *up* aggressively (spikes hurt users now) but scale *down* slowly (e.g. only after load stays low for several minutes), so a brief dip does not trigger an expensive teardown-then-rebuild cycle. **Cost-aware bounds**: GPUs are expensive enough that the `min_replicas` floor and `warm_buffer` are real budget decisions, not afterthoughts — a warm H100 sitting idle still bills by the second. Scale-to-zero is attractive for rarely-used models but pays the full multi-minute cold start on the next request, so reserve it for latency-tolerant or batch workloads.

!!! warning "Common pitfall: autoscaling on the wrong metric"

    Scaling on GPU utilization or even on request count feels natural and is wrong for LLMs. Utilization saturates at 100% and stays pinned while the queue (and p99) grows underneath it — it cannot tell you *how much* you are behind. Request count ignores that one 8k-token request is worth a hundred 80-token ones. Scale on **waiting/running tokens** (or predicted TTFT). It is the only signal that is both a leading indicator and proportional to actual GPU work.

## Multi-Model Serving: Many Models, Finite GPUs

Production fleets serve dozens of models: base + fine-tunes, multiple sizes, embedding models, a moderation classifier, several customer-specific adapters. Giving each its own always-on dedicated GPUs is simple but ruinously expensive when most are lightly used. Three strategies, from cheapest-but-slowest to most-isolated.

**LoRA / adapter multiplexing (best for fine-tunes).** If twenty models are LoRA fine-tunes ([PEFT I: LoRA, QLoRA, DoRA & The Adapter Family](../05-posttraining-alignment/03-peft-lora-qlora.html)) of one base, you load the base weights *once* and swap only the small low-rank adapter matrices per request — even *within a single batch*, different requests can use different adapters (multi-LoRA batching, as in vLLM's S-LoRA-style support). One GPU's worth of base weights serves twenty "models" at near-zero marginal memory. This is the highest-leverage multi-model trick when it applies.

**Time-sharing with hot-swap (best for many independent models, spiky traffic).** Keep a pool of GPUs and load/evict *full* model weights on demand, treating HBM like a cache with an LRU policy keyed on recent traffic. The cost is the cold-start latency on a cache miss; mitigate it by keeping the top-$k$ models pinned and streaming weights fast from a local NVMe tier.

**Static partitioning (best for a few high-QPS models).** Give each big, busy model its own dedicated pool. Simplest to reason about, best isolation (no noisy-neighbor), worst utilization. Reserve it for the handful of models that are busy enough to justify full GPUs.

```python
from collections import OrderedDict

class ModelLRUCache:
    """
    HBM-as-cache for whole-model hot-swapping. Keeps total loaded weights under
    `capacity_gb`; evicts the least-recently-used model on a miss. Production
    systems pin the top-k busiest models so they are never evicted (anti-thrash).
    """
    def __init__(self, capacity_gb, pinned=()):
        self.capacity_gb = capacity_gb
        self.loaded = OrderedDict()    # model_id -> size_gb, MRU at the end
        self.pinned = set(pinned)
        self.used_gb = 0.0

    def get(self, model_id, size_gb, load_fn):
        if model_id in self.loaded:
            self.loaded.move_to_end(model_id)   # mark most-recently-used
            return  # hot path: already resident, zero cold-start
        # Cache miss: evict LRU (never a pinned model) until it fits.
        while self.used_gb + size_gb > self.capacity_gb:
            for victim in list(self.loaded):
                if victim not in self.pinned:
                    self.used_gb -= self.loaded.pop(victim)
                    break
            else:
                raise MemoryError("cannot fit model; all residents pinned")
        load_fn(model_id)                # SLOW: weights -> HBM (the cold start)
        self.loaded[model_id] = size_gb
        self.used_gb += size_gb
```

The decision tree is: *Are they adapters of one base?* → multiplex. *Are they distinct but individually low-QPS?* → time-share with hot-swap. *Are a few of them high-QPS?* → give those static pools and time-share the long tail. Most real fleets do all three at once.

## Putting It Together: An End-to-End Request Trace

To consolidate, here is the life of one streaming chat request through the whole system.


{{fig:serving-e2e-request-trace}}


Every numbered step maps to a layer we designed. Notice how the *same* request touches quota (gateway), placement and cache affinity (router), admission and SLO prediction (admit), batching policy (replica), and feeds the autoscaler and metering — these are not separate systems, they are one control loop around the GPUs.

!!! interview "Interview Corner"

    **Q:** You're designing an LLM serving system for a chat product. Peak load is 500 requests/sec, average prompt 1,000 tokens, average output 500 tokens. The SLO is p99 TTFT ≤ 800 ms and p90 TPOT ≤ 60 ms. Walk me through how you'd size the fleet and what your main failure modes are.

    **A:** I'd start by separating the two sizing constraints. **Throughput sizing:** measure single-replica prefill rate (say ~12k tok/s on the target GPU) and decode capacity; prefill load is $500 \times 1000 = 5\times10^5$ prompt tok/s, so I need ~42 replica-equivalents of prefill *at 100% util*, but I size to ~70% utilization for the p99 tail — Little's Law plus the $1/(1-\rho)$ queueing blowup means running near saturation will blow the p99 even if the mean keeps up — so ~60 replicas for throughput. **Memory sizing:** compute KV bytes/token from the model's layers, KV-head count (GQA helps a lot here), and precision; check how many concurrent (prompt+output ≈ 1.5k-token) sequences fit per GPU, and convert peak concurrency (Little's Law: $L = \lambda W$) into a second replica count. I take the **max** of the two.

    For **TTFT** I'd use chunked prefill so long prompts don't stall the decode loop, prefix caching for the shared system prompt (likely a big win in chat), and predictive admission control that sheds load with a 429 rather than admitting requests that will miss the SLO. For **TPOT** I'd cap `max_num_batched_tokens` so decode steps stay small enough to hit 60 ms.

    **Main failure modes:** (1) unbounded `max_tokens` letting a few long generations monopolize KV and wreck p99 — fix with a hard cap; (2) autoscaling on GPU utilization instead of queue depth, so we scale too late given multi-minute cold starts — fix with queue/predicted-TTFT scaling plus a warm buffer; (3) round-robin routing imbalancing token-work and ignoring prefix-cache affinity — fix with power-of-two-choices weighted by tokens and cache overlap; (4) running at 95% utilization "to save cost" and falling off the queueing cliff — provision headroom instead.

!!! key "Key Takeaways"

    - **LLM serving is stateful, long-lived, and two-dimensional in latency.** Design around TTFT *and* TPOT, not a single "latency"; the unit of work is a streaming sequence pinning a KV cache, not a stateless request.
    - **The router is the brain.** Balance by *tokens*, not request count, and use power-of-two-choices with prefix-cache affinity — never round-robin and never "always pick the global minimum."
    - **SLOs are percentiles, and p99 is dominated by queueing.** Because wait time scales as $1/(1-\rho)$, you provision LLM clusters to ~60–75% utilization; the headroom is the budget that keeps the tail bounded.
    - **Two independent sizing constraints — throughput and memory — and you take the max.** Throughput sizing comes from prefill rate and Little's Law; memory sizing comes from KV bytes/token and HBM capacity. Know whether you are compute- or memory-bound.
    - **Batching policy is how you spend the latency budget.** `max_num_batched_tokens`, chunked prefill, and prefill/decode disaggregation place you on the throughput-vs-TPOT frontier; tune them against your real traffic, not a benchmark.
    - **Autoscale on a leading indicator (queue depth / waiting tokens), not GPU utilization.** Cold starts are minutes, so keep a warm buffer, scale up fast and down slow, and *drain* before releasing a GPU.
    - **Multi-model serving is a memory-management problem.** Multiplex LoRA adapters over a shared base, hot-swap whole models with an LRU/pinned cache, and statically partition only the few high-QPS models.

!!! sota "State of the Art & Resources (2026)"
    LLM serving systems have matured into a rich stack: continuous batching (Orca/vLLM) and PagedAttention are now table-stakes defaults, while prefill/decode disaggregation, KVCache-centric architectures (Mooncake), and multi-LoRA multiplexing define the production frontier for high-throughput, SLO-compliant fleets.

    **Foundational work**

    - [Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models* (OSDI 2022)](https://www.usenix.org/conference/osdi22/presentation/yu) — introduced iteration-level (continuous) batching, the single biggest throughput step-change in LLM serving.
    - [Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention* (SOSP 2023)](https://arxiv.org/abs/2309.06180) — virtual-memory-inspired KV cache; enables near-theoretical concurrency and underpins vLLM.

    **Recent advances (2023–2026)**

    - [Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs* (2023)](https://arxiv.org/abs/2312.07104) — RadixAttention for prefix-cache-aware routing; now one of the two dominant open serving frameworks.
    - [Patel et al., *Splitwise: Efficient Generative LLM Inference Using Phase Splitting* (ISCA 2024)](https://arxiv.org/abs/2311.18677) — hardware case for running prefill and decode on separate machines; foundational disaggregation paper.
    - [Zhong et al., *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving* (OSDI 2024)](https://arxiv.org/abs/2401.09670) — co-optimizes per-phase resource allocation and parallelism to maximize requests meeting both TTFT and TPOT SLOs.
    - [Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve* (OSDI 2024)](https://arxiv.org/abs/2403.02310) — chunked prefill with stall-free scheduling; directly controls the TTFT/TPOT frontier.
    - [Qin et al., *Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving* (FAST 2025 Best Paper)](https://arxiv.org/abs/2407.00079) — production architecture (Kimi) treating KV cache as a first-class distributed resource; shows 525% throughput gain in overload scenarios.
    - [Sheng et al., *S-LoRA: Serving Thousands of Concurrent LoRA Adapters* (2023)](https://arxiv.org/abs/2311.03285) — multi-adapter batching over a shared base; the canonical reference for LoRA multiplexing.

    **Open-source & tools**

    - [vllm-project/vllm](https://github.com/vllm-project/vllm) — the leading open-source serving engine (81k+ stars); PagedAttention, continuous batching, multi-LoRA, 200+ model architectures.
    - [sgl-project/sglang](https://github.com/sgl-project/sglang) — high-performance alternative with RadixAttention, disaggregated prefill/decode, powering 400k+ GPUs in production.
    - [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) — NVIDIA's optimized inference library with custom attention kernels, speculative decoding, and Triton integration.

    **Go deeper**

    - [Databricks, *Reliable LLM Inference at Scale* (2025)](https://www.databricks.com/blog/reliable-llm-inference-scale) — production war stories: model-unit abstractions, cost-aware autoscaling, silent failure detection serving 125 trillion tokens/month.
    - [vLLM Blog, *vLLM Router: A High-Performance Prefill/Decode-Aware Load Balancer* (2025)](https://vllm.ai/blog/2025-12-13-vllm-router-release) — power-of-two-choices, consistent-hash affinity routing, and P/D disaggregation orchestration in a production Rust router.

## Further reading

- Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models* (OSDI 2022) — the paper that introduced iteration-level (continuous) batching.
- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (the vLLM paper, SOSP 2023).
- Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs* — RadixAttention and prefix-cache-aware serving.
- Patel et al., *Splitwise: Efficient Generative LLM Inference Using Phase Splitting* — the case for prefill/decode disaggregation.
- Agrawal et al., *Sarathi-Serve* (Taming Throughput-Latency Tradeoff with chunked prefill) — chunked prefill and the TTFT/TPOT tradeoff.
- Sheng et al., *S-LoRA: Serving Thousands of Concurrent LoRA Adapters* — multi-adapter batching for multi-model serving.
- Mitzenmacher, *The Power of Two Choices in Randomized Load Balancing* — the load-balancing result behind power-of-*d*-choices routing.
- Kleinrock, *Queueing Systems, Volume 1: Theory* — Little's Law and the $1/(1-\rho)$ behavior, the foundation of capacity planning.
