# 7.12 Inference Economics: Latency, Throughput & Cost

Every LLM serving system is a negotiation between three forces: latency (how fast a single request finishes), throughput (how many requests the system handles per second), and cost (how much you pay per token produced). These three forces are coupled in fundamental ways — you cannot optimize all three simultaneously, and understanding the trade-offs quantitatively is what separates engineers who ship sustainable inference systems from those who run out of GPU budget six weeks after launch.

This chapter is the capstone of Part VII. We assume you have read about the mechanics of prefill and decode in [The Anatomy of LLM Inference: Prefill, Decode & The KV Cache](../07-inference-serving/01-anatomy-inference.html), continuous batching in [Continuous Batching & Request Scheduling](../07-inference-serving/02-continuous-batching.html), and the memory machinery in [PagedAttention & KV-Cache Memory Management](../04-kernels-efficiency/06-paged-attention-kv.html). Here we focus on the economics: the math of dollars per million tokens, the hardware selection calculus, and the operational playbook for keeping costs under control.

## The Latency–Throughput–Cost Triangle

The fundamental constraint is that a GPU can do roughly a fixed number of floating-point operations per second (FLOP/s) and move a fixed number of bytes per second across its memory bus (bandwidth). Every inference workload maps to a point on the roofline (see [The Roofline Model & Performance Engineering](../04-kernels-efficiency/01-roofline-performance.html)):

$$
\text{Arithmetic Intensity} = \frac{\text{FLOPs performed}}{\text{Bytes read from/written to HBM}}
$$

Decode is almost always memory-bandwidth-bound: for a batch of size $B$ generating one token, we read all model weights once (roughly $2P$ bytes for an $FP16$ model with $P$ parameters) but do only $2PB$ FLOPs — arithmetic intensity is $B$. An H100 has roughly 3,350 GB/s of HBM bandwidth and ~989 TFLOP/s of BF16 MMA throughput. The breakeven arithmetic intensity is:

$$
\text{AI}^* = \frac{989 \times 10^{12}}{3350 \times 10^9} \approx 295
$$

Until batch size exceeds ~295, decode is bandwidth-bound, and **adding more arithmetic units does not help** — you are just waiting for weights to stream from HBM.

Prefill, by contrast, is compute-bound at large sequence lengths (the attention FLOPs scale as $O(L^2)$, eventually dominating weight-loading). This asymmetry drives the disaggregated prefill/decode architectures described in [Disaggregated Prefill/Decode & Chunked Prefill](../07-inference-serving/08-disaggregated-chunked-prefill.html).

### The Three-Way Trade-off Stated Clearly

| Goal | Lever | Downside |
|---|---|---|
| Minimize TTFT (time-to-first-token) | Small batch, high-priority prefill | Low GPU utilization, high $/token |
| Maximize throughput (tokens/s/GPU) | Large batch, full GPU utilization | Higher latency per request |
| Minimize $/1M tokens | Fill GPU to arithmetic intensity breakeven | Latency SLO may be missed |

There is no free lunch. Your job is to find the operating point that satisfies the latency SLO while maximizing GPU utilization — because utilization is the primary driver of cost efficiency.

## The Math of Dollars Per Million Tokens

Let's derive the cost formula from scratch and then plug in real numbers.

### GPU Rental Cost

Suppose you rent a node with $G$ GPUs at a price of $\$R$ per hour. The node delivers $T_{\text{sustained}}$ output tokens per second (measured at your operating batch size). In one hour you produce:

$$
\text{tokens/hour} = T_{\text{sustained}} \times 3600
$$

The cost per token is therefore:

$$
\text{cost/token} = \frac{R}{T_{\text{sustained}} \times 3600}
$$

And per million tokens:

$$
\boxed{\text{\$/1M tokens} = \frac{R \times 10^6}{T_{\text{sustained}} \times 3600}}
$$

!!! example "Worked Example: H100 cluster serving Llama-3 70B"

    **Setup:** 4× H100 SXM (8 GPU node rented at around \$20/hour, or roughly \$2.50/GPU-hour). Serving Llama-3 70B in BF16 (140 GB weights), tensor-parallel across 4 GPUs. At a busy batch of 32 concurrent requests each generating 512 tokens:

    - Sustained decode throughput (measured): roughly 1,800 output tokens/second for the 4-GPU node.
    - Rental cost: \$20/hour for the node.

    $$
    \text{\$/1M tokens} = \frac{20 \times 10^6}{1800 \times 3600} = \frac{20{,}000{,}000}{6{,}480{,}000} \approx \$3.09 \text{ per 1M output tokens}
    $$

    If we instead run a small batch of 4 (low traffic), throughput drops to ~550 tokens/s:

    $$
    \text{\$/1M tokens} = \frac{20 \times 10^6}{550 \times 3600} \approx \$10.10 \text{ per 1M output tokens}
    $$

    **Lesson:** at 1/8th the traffic, cost per token triples. Low utilization is the enemy of cost efficiency.

{{fig:econ-cost-per-token-hyperbola}}

### Input Tokens vs. Output Tokens

Output tokens are expensive because they require an autoregressive decode step — one weight-loading pass per token. Input (prefill) tokens are cheap relative to output because they are processed in parallel. As a rough rule of thumb, at large batch sizes and typical prompt/completion ratios, output tokens cost roughly 3–5× more in wall-clock GPU-time — and hence dollars — per token than input tokens (the exact ratio depends on sequence lengths and batch size). Note the asymmetry is *not* about arithmetic: the FLOPs per token are essentially identical (~2P, where P is the parameter count) for a prefill token and a decode token. It is about bandwidth. A decode step reads the entire weight matrix from HBM to emit a single token, so it is memory-bound and poorly utilized; parallel prefill reads those same weights once and amortizes them across every prompt token at near-peak FLOP utilization. Output tokens are expensive because each one pays for its own weight-loading pass, not because it does more math.

Public API providers charge differently for input and output tokens for exactly this reason. When optimizing your prompt, shortening the *completion* (e.g., via structured generation or speculative decoding — see [Speculative Decoding: Draft Models, Medusa, EAGLE & Lookahead](../07-inference-serving/06-speculative-decoding.html)) has a higher ROI than shortening the prompt.

## Batching vs. Latency SLOs

Batching is the primary mechanism by which you trade latency for efficiency. Understanding the math helps you set the right operating point.

### Continuous Batching and Effective Batch Size

With continuous batching (see [Continuous Batching & Request Scheduling](../07-inference-serving/02-continuous-batching.html)), the effective batch size at any instant is the number of sequences that are currently in the decode phase. Call this $B_{\text{eff}}$. The decode throughput scales approximately linearly with $B_{\text{eff}}$ until the arithmetic intensity breakeven, after which it saturates at compute capacity.

For a model with $P$ parameters in FP16/BF16, the time to generate one token for a batch of size $B$ is approximately:

$$
t_{\text{decode}}(B) = \max\!\left(\frac{2P}{\text{BW}},\ \frac{2PB}{\text{FLOP/s}}\right)
$$

where BW is HBM bandwidth. The first term is the bandwidth-bound floor (weight streaming), and the second is the compute-bound ceiling. The knee of the curve — where bandwidth and compute balance — is at $B^* = \text{FLOP/s} / \text{BW}$ (the arithmetic intensity breakeven from above).

### Latency SLOs in Practice

Typical production SLOs take two forms:
- **TTFT (time-to-first-token)**: often 200–500 ms for interactive chat.
- **TBT (time-between-tokens)**: often 30–80 ms/token for a streaming UX that feels "fast" (roughly 12–33 tokens/second perceived).
- **P99 end-to-end latency** for a fixed-length response.

A key insight is that *batching affects TBT but not TTFT in the same way*. TTFT is dominated by prefill time (which scales with prompt length and is bounded by compute), whereas TBT is dominated by decode speed (which scales with batch size up to the bandwidth floor).

The scheduler must balance these. The simplest heuristic: saturate batch up to $B^*$, which gives maximum throughput without exceeding the compute-bound regime. Beyond $B^*$ you pay in latency without gaining more tokens per dollar.

```python
# Illustrative scheduler: fill batch up to the bandwidth breakeven point.
# This is a simplified model; real systems use token budgets and KV cache limits.

import math

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
    t_compute   = flops / flops_per_sec

    # Actual time is the max; report in ms
    return max(t_bandwidth, t_compute) * 1000.0


# --- Hardware constants ---
H100_FLOPS = 989e12      # BF16 tensor core FLOP/s (H100 SXM5)
H100_BW    = 3.35e12     # HBM bandwidth bytes/s

# --- Model: Llama 3 70B (70e9 params, BF16) ---
N_PARAMS = 70e9

B_STAR = bandwidth_breakeven(H100_FLOPS, H100_BW)
print(f"H100 arithmetic intensity breakeven batch size: {B_STAR}")

for batch in [1, 4, 16, 64, 128, 256, B_STAR]:
    t = decode_step_time_ms(N_PARAMS, batch, H100_FLOPS, H100_BW)
    regime = "bandwidth-bound" if batch <= B_STAR else "compute-bound"
    print(f"  batch={batch:4d}  step_time={t:.2f} ms  regime={regime}")
```

Running this produces output along the lines of:

```text
H100 arithmetic intensity breakeven batch size: 295
  batch=   1  step_time=41.79 ms  regime=bandwidth-bound
  batch=   4  step_time=41.79 ms  regime=bandwidth-bound
  batch=  16  step_time=41.79 ms  regime=bandwidth-bound
  batch=  64  step_time=41.79 ms  regime=bandwidth-bound
  batch= 128  step_time=41.79 ms  regime=bandwidth-bound
  batch= 256  step_time=41.79 ms  regime=bandwidth-bound
  batch= 295  step_time=41.79 ms  regime=bandwidth-bound
```

Below $B^* \approx 295$ the decode step time is flat at ~42 ms per step (bandwidth-bound floor). Each new request added below this threshold contributes zero extra latency but produces one more output token — a pure win. Above this, each additional request adds proportional latency.

{{fig:econ-decode-step-time-knee}}

The practical lesson: on a single H100 serving a 70B BF16 model, you can pack up to ~295 concurrent decoding sequences before latency starts climbing. That is the regime where `tokens/s/GPU` is maximized.

## GPU Selection and the Hardware Cost Function

### Key GPU Metrics for Inference

When choosing a GPU for inference, the relevant specs are different from those for training:

| GPU | HBM (GB) | BW (TB/s) | BF16 TFLOP/s | $/hr (spot, approx) |
|---|---|---|---|---|
| A10G | 24 | 0.60 | 31.2 | ~\$0.70 |
| A100 40GB | 40 | 2.00 | 312 | ~\$2.50 |
| A100 80GB | 80 | 2.00 | 312 | ~\$3.50 |
| H100 SXM5 | 80 | 3.35 | 989 | ~\$5.00 |
| H200 SXM | 141 | 4.80 | 989 | ~\$7.00 |
| B200 SXM | 192 | 8.00 | ~4,500 | ~\$10.00 |

Note: prices are illustrative order-of-magnitude estimates and vary significantly by provider, region, and contract.

For inference on a fixed model, the figures of merit are:
1. **Tokens/s/dollar**: dominated by HBM bandwidth for decode-heavy workloads.
2. **Max batch size before KV-cache OOM**: dominated by HBM capacity.
3. **TTFT for long prompts**: dominated by compute (FLOP/s).

The H100's jump in bandwidth (3.35 vs. 2.00 TB/s for A100) directly translates to a 1.67× speedup in decode throughput per GPU when bandwidth-bound — before any software optimization.

### Tensor Parallelism Across GPUs

Splitting a model across $N$ GPUs with tensor parallelism (TP) reduces per-GPU memory by $1/N$ but also splits both FLOPs and bandwidth by roughly $1/N$. The bandwidth-bound decode time becomes:

$$
t_{\text{decode, TP=N}}(B) \approx \max\!\left(\frac{2P / N}{\text{BW}_{\text{GPU}}},\ \frac{2PB / N}{\text{FLOP/s}_{\text{GPU}}}\right) + t_{\text{allreduce}}
$$

The all-reduce adds a fixed per-step communication overhead (typically a few ms over NVLink). Since the weight-streaming time halves with TP=2, the effective batch size breakeven also halves — you saturate compute with fewer concurrent requests.

For very large models (>70B), TP is often mandatory to fit in memory. The key point is that **TP does not improve tokens/s/dollar beyond what's needed to fit the model** — it just allows serving. Sequence parallelism and disaggregated architectures are needed for further scaling (see [Multi-GPU & Multi-Node Inference](../07-inference-serving/11-multi-gpu-inference.html)).

## Why Decode-Heavy Workloads Are Expensive

Reasoning models, coding assistants, and long-form generation are all examples of decode-heavy workloads: the ratio of output tokens to input tokens is high (sometimes 10:1 or more). This matters economically for three reasons:

1. **Compute cost scales with output length.** Each output token requires one full forward pass through the decode stage (weight loading from HBM). A 10× longer output costs roughly 10× more GPU time.

2. **KV cache memory scales with context length.** For a model with $n_{\text{heads}}$ KV heads, head dimension $d$, $L$ layers, in BF16, the KV cache for one sequence of length $S$ is:

$$
\text{KV bytes} = 2 \times n_{\text{kv}} \times d \times L \times S \times 2 \text{ bytes}
$$

For Llama-3 70B: $n_{\text{kv}}=8$ (GQA), $d=128$, $L=80$. At $S=32{,}768$ tokens:

$$
\text{KV bytes} = 2 \times 8 \times 128 \times 80 \times 32768 \times 2 = 10.7 \text{ GB per sequence}
$$

A single 80GB H100 serving 70B at TP=2 (40 GB available for KV after weights) can hold at most about 3–4 concurrent long-context sequences. Batch size is severely limited.

3. **Speculative decoding helps but has limits.** Speculative decoding (see [Speculative Decoding: Draft Models, Medusa, EAGLE & Lookahead](../07-inference-serving/06-speculative-decoding.html)) can recover 2–3× speed for predictable outputs (code, factual answers) by verifying multiple tokens per step, but the draft model adds HBM bandwidth overhead and is less effective for creative or reasoning-heavy outputs.

### Chain-of-Thought Tax

The reasoning models popularized by OpenAI o1 and DeepSeek-R1 generate long internal reasoning traces before the final answer. If a reasoning model generates 2,000 thinking tokens before a 200-token answer, the effective output-to-answer token ratio is 11:1. At \$3/1M output tokens, reasoning for 1,000 queries costs:

$$
\text{cost} = \frac{1000 \times 2200}{10^6} \times 3 = \$6.60
$$

vs. \$0.60 for a direct-answer model with 200 tokens. The quality premium has a concrete dollar tag. See [Reasoning, Chain-of-Thought & Test-Time Compute](../05-posttraining-alignment/10-reasoning-test-time-compute.html) for the quality vs. cost trade-off in detail.

## Capacity Planning and Autoscaling

### The Capacity Planning Formula

Capacity planning starts from traffic projections. Given:
- $\lambda$ = peak requests per second
- $\bar{o}$ = average output tokens per request
- $T_{\text{decode}}$ = sustained decode tokens/s per GPU
- $U_{\text{target}}$ = target GPU utilization (e.g., 0.7 to leave headroom)

The number of GPUs required for decode is:

$$
N_{\text{GPUs}} = \lceil \frac{\lambda \cdot \bar{o}}{T_{\text{decode}} \cdot U_{\text{target}}} \rceil
$$

For prefill-heavy workloads, a similar formula applies based on TTFT SLO and prefill FLOP/s.

!!! example "Capacity planning for a mid-size API"

    Suppose your API sees peak traffic of 50 requests/second, each generating an average of 400 output tokens. You are using A100 80GB GPUs with TP=1 serving a 13B model. The 13B model fits on one GPU with plenty of KV cache headroom.

    Sustained decode throughput at batch=50 (all bandwidth-bound): roughly 8,000 tokens/s per A100 (illustrative).

    Target utilization = 0.70 (30% headroom for traffic spikes and cold starts).

    $$
    N_{\text{GPUs}} = \left\lceil \frac{50 \times 400}{8000 \times 0.70} \right\rceil = \left\lceil \frac{20000}{5600} \right\rceil = \lceil 3.57 \rceil = 4 \text{ GPUs}
    $$

    At ~\$3.50/GPU-hour (A100 80GB), this cluster costs \$14/hour or \$336/day at peak. During off-peak at 10% load, you may scale down to 1–2 GPUs and save ~70%.

### Autoscaling Strategies

Autoscaling for LLM serving is different from stateless web services because:

1. **Startup latency is high.** Loading a 70B model from disk to GPU memory takes tens of seconds. Cold-start latency makes traditional reactive autoscaling painful.

2. **The batch size effect creates sudden non-linearity.** At low traffic, fewer GPUs may be *just as fast* (since decode is bandwidth-bound and batch size doesn't affect per-step latency until saturation). The breakeven batch size gives you a natural hysteresis threshold.

3. **GPU memory is the binding constraint, not CPU.** You cannot partially load a model; you need the whole thing in HBM.

Practical approaches:

- **Predictive (proactive) scaling**: use historical traffic patterns to pre-scale 5–10 minutes ahead of predicted ramp-up.
- **Scale to zero with warm pools**: maintain 1 "warm" replica always; scale-to-zero on truly idle queues.
- **Model sharding and multiplexing**: run multiple smaller models on the same GPU (e.g., several 7B fine-tunes with different LoRA adapters via LoRA adapter hot-swap, as in [PEFT I: LoRA, QLoRA, DoRA & The Adapter Family](../05-posttraining-alignment/03-peft-lora-qlora.html)).

```python
# Minimal autoscaler mock: decides how many replicas to run based on queue depth
# and current tokens/s, with hysteresis to avoid thrashing.

from dataclasses import dataclass, field
from collections import deque
import time

@dataclass
class AutoscalerConfig:
    min_replicas: int = 1
    max_replicas: int = 16
    target_tokens_per_sec_per_replica: float = 2000.0  # sustained decode tps/replica
    scale_up_queue_threshold: int = 100    # tokens queued → add replica
    scale_down_idle_seconds: float = 120.0  # idle for 2 min → remove replica
    cooldown_seconds: float = 60.0          # min time between scale events

@dataclass
class AutoscalerState:
    n_replicas: int = 1
    last_scale_time: float = field(default_factory=time.time)
    idle_since: dict = field(default_factory=dict)  # replica_id → time became idle

def autoscale_step(
    config: AutoscalerConfig,
    state: AutoscalerState,
    queue_depth_tokens: int,     # tokens currently waiting in the request queue
    active_tokens_per_sec: float, # current measured throughput
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
        needed = int((active_tokens_per_sec + queue_depth_tokens / 30.0)
                     / config.target_tokens_per_sec_per_replica) + 1
        desired = min(needed, config.max_replicas)

    # Scale DOWN: we have spare capacity
    elif active_tokens_per_sec < 0.5 * tokens_capacity and state.n_replicas > config.min_replicas:
        desired = max(config.min_replicas, state.n_replicas - 1)

    if desired != state.n_replicas:
        state.n_replicas = desired
        state.last_scale_time = now

    return desired
```

## Quantization and Its Cost Impact

Quantization (see [Quantization I: Post-Training Quantization (GPTQ, AWQ, SmoothQuant)](../04-kernels-efficiency/07-quantization-ptq.html) and [Quantization II: INT4/INT8/FP8, GGUF, bitsandbytes & QAT](../04-kernels-efficiency/08-quantization-formats-qat.html)) is one of the most powerful tools in the inference economist's toolkit.

### How Quantization Changes the Economics

Going from BF16 to INT8 halves the number of bytes loaded from HBM per decode step — directly halving the bandwidth-bound decode time (for the same batch size) or equivalently, halving the number of GPUs needed to sustain the same throughput.

Going from BF16 to INT4 (e.g., via GPTQ or AWQ) approximately quarters the decode memory bandwidth, but introduces some quality degradation and requires dequantization overhead.

$$
\text{speedup}_{\text{BW}} = \frac{\text{bits}_{original}}{\text{bits}_{quantized}}
$$

For a 70B model:

| Precision | Weights size | Decode BW factor | GPUs for 70B (TP) |
|---|---|---|---|
| BF16 | 140 GB | 1.0× | 2× H100 |
| FP8 | 70 GB | 2.0× | 1× H100 |
| INT4 (AWQ) | 35 GB | 4.0× | 1× H100 (with KV headroom) |

FP8 inference is now supported natively on H100 and B200 hardware via the FP8 matmul units, offering near-BF16 quality with roughly 2× throughput improvement. This is a straightforward win for most production workloads.

### KV Cache Quantization

Beyond weight quantization, the KV cache itself can be quantized. From the 70B example above (10.7 GB per sequence at 32k context in BF16), switching KV cache to INT8 halves this to 5.35 GB/sequence — roughly doubling the number of concurrent long-context sequences a single node can handle.

## Optimizing the Bill: A Practitioner Playbook

Pulling all the above together, here is a concrete checklist for reducing inference costs.

### Tier 1: Cheapest wins (deploy immediately)

1. **Maximize continuous-batch utilization.** If your average GPU utilization during peak is below 70%, you have headroom to serve more traffic at zero incremental cost. Profile with `nvidia-smi dmon` or vLLM's built-in metrics.

2. **Enable FP8 or INT8 quantization** where quality is acceptable. Use AWQ or GPTQ for INT4 on older hardware. This halves or quarters your GPU memory footprint, often allowing you to cut replicas.

3. **Enable prefix caching** (see [Prefix Caching & KV-Cache Reuse](../07-inference-serving/07-prefix-caching.html)) for workloads with shared system prompts or few-shot examples. Recomputing a 2,000-token system prompt on every request wastes significant compute.

4. **Choose the right model size.** A well-fine-tuned 13B model may outperform a generic 70B on your specific task at 1/5th the compute cost. Evaluate empirically.

### Tier 2: Architectural changes (higher leverage, more work)

5. **Speculative decoding** for predictable outputs (code, templates, FAQ answers). Properly configured, a 2–3× throughput gain is achievable with no quality loss.

6. **Disaggregated prefill/decode** for mixed-length workloads. Prefill-heavy requests (long documents) run on FLOP-optimized nodes; decode runs on bandwidth-optimized nodes. This avoids decode latency spikes caused by large prefill batches blocking the decode queue.

7. **Knowledge distillation** into a smaller task-specific model. If your deployment uses a general 70B model but only for one narrow task, distilling a fine-tuned 7B model can yield 10× cost reduction. See [Distillation, Model Compression & Knowledge Transfer](../05-posttraining-alignment/12-distillation-compression.html).

### Tier 3: Systemic cost management

8. **Prompt compression** (prompt caching, summarization, retrieval rather than full context) reduces input tokens. Though cheaper per token, input tokens still consume compute and KV cache memory.

9. **Routing to model tiers.** Route simple queries (keyword lookup, formatting) to small/fast/cheap models, and complex queries to large models. See [Caching, Routing & Cost Control in Production](../12-production-mlops/03-caching-routing-cost.html).

10. **Spot/preemptible instances** for batch inference workloads (embeddings, offline scoring, eval runs). Spot pricing is typically 60–70% cheaper; use checkpoint-and-resume for fault tolerance.

```python
# Cost-aware router: send queries to a small or large model based on estimated complexity.
# In production, use a lightweight classifier; here we use a heuristic proxy.

from typing import Literal

ModelTier = Literal["fast_small", "capable_large"]

# Illustrative cost per 1M output tokens (update to your actual prices)
COST_PER_1M = {
    "fast_small":   0.50,   # e.g., a 7B fine-tuned model on cheap hardware
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

    code_keywords  = {"def", "class", "import", "function", "algorithm",
                      "implement", "debug", "explain", "analyze"}
    hard_keywords  = {"compare", "contrast", "design", "evaluate", "synthesize",
                      "critique", "reason", "proof", "derive"}

    code_signal  = sum(1 for w in words if w in code_keywords) / max(n_words, 1)
    hard_signal  = sum(1 for w in words if w in hard_keywords) / max(n_words, 1)
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

def simulate_cost(n_queries: int = 1000,
                  avg_output_tokens: int = 300,
                  threshold: float = 0.25) -> dict:
    random.seed(42)
    total_small = 0
    total_large = 0

    # Synthetic prompts: mix of simple and complex
    prompts = (
        ["What is 2+2?"] * 400              # trivial
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
```

## The Full Cost Stack: Beyond GPU Compute

GPU compute is the dominant but not the only cost. A complete cost breakdown for a production LLM API includes:

| Category | Typical fraction of total cost | Notes |
|---|---|---|
| GPU compute | 60–80% | Model serving (decode, prefill) |
| GPU memory-tied KV cache overhead | included above | But drives replica count |
| Networking | 5–10% | NVLink within node; Infiniband/Ethernet cross-node |
| Storage & I/O | 2–5% | Model checkpoints, logging, dataset serving |
| CPU/orchestration | 3–8% | Kubernetes, API servers, tokenizers, routers |
| Observability | 1–3% | Tracing, metrics, logging pipelines |
| Engineering time | Variable | Often dominates at < \$10k/month GPU spend |

At small scale, the marginal dollar usually goes further when spent on engineering (prompt compression, fine-tuning a smaller model, prefix caching implementation) than on more hardware.

At large scale (> \$1M/month), negotiated reserved instance pricing, custom silicon (Google TPUs, AWS Trainium2, NVIDIA A100/H100 3-year reservations), and distillation programs pay off significantly.

!!! interview "Interview Corner"

    **Q:** You are designing an LLM API serving a mix of chat queries (avg 200 output tokens, latency-sensitive, TTFT < 300 ms) and batch document summarization jobs (avg 1500 output tokens, latency-insensitive). Both run on the same 70B model. How would you architect the serving system, and what are the key cost-saving opportunities?

    **A:** The core insight is that these two workloads have opposite requirements: chat needs low TTFT (fast prefill, priority scheduling) while batch jobs want maximum throughput (large batches, no SLO pressure). Running them on the same fleet means batch jobs inflate queue latency for chat, and the low latency requirement of chat prevents batch jobs from filling the GPUs.

    The right architecture is **workload disaggregation**: a dedicated chat tier (2–4 GPUs per replica, continuous batching with a tight TTFT SLO, FP8 quantization for speed, prefix caching for common system prompts) and a separate batch tier (larger batch sizes, potentially INT4 quantization, spot/preemptible instances since failures can be retried). A router (based on request metadata or a flag in the API call) directs traffic to the appropriate tier.

    Key cost-saving opportunities in order of leverage: (1) INT4/FP8 quantization on both tiers — halves or quarters GPU count; (2) prefix caching on chat tier for shared system prompts; (3) spot instances on the batch tier for 60–70% compute cost reduction; (4) routing simpler chat queries to a smaller distilled model (7B or 13B fine-tune); (5) speculative decoding on the batch tier where output patterns are predictable (document templates, structured summaries).

## Putting It All Together: A Reference Dashboard

A good inference cost dashboard tracks these key metrics in real time:

```python
# Reference metrics to instrument in your serving stack (e.g., via Prometheus)
# Each metric name maps to a Prometheus gauge/histogram.

INFERENCE_METRICS = {
    # Throughput
    "output_tokens_per_second":   "gauge",    # Overall system output rate
    "input_tokens_per_second":    "gauge",    # Overall system prefill rate

    # Latency
    "ttft_p50_ms":                "gauge",    # Median time to first token
    "ttft_p99_ms":                "gauge",    # p99 time to first token
    "tbt_p50_ms":                 "gauge",    # Median time between tokens
    "tbt_p99_ms":                 "gauge",    # p99 time between tokens

    # Efficiency
    "gpu_utilization_pct":        "gauge",    # Per-GPU utilization
    "kv_cache_utilization_pct":   "gauge",    # KV cache fill rate (via vLLM metrics)
    "batch_size_mean":            "gauge",    # Average active batch size
    "batch_size_p99":             "gauge",    # p99 active batch size

    # Cost
    "cost_per_1m_output_tokens":  "gauge",    # Derived: $/1M output tokens
    "gpu_cost_per_hour":          "gauge",    # Cluster cost rate (from cloud API or fixed)
    "requests_per_dollar":        "gauge",    # Inverse cost efficiency

    # Quality proxy
    "generation_errors_per_min":  "gauge",    # Truncations, OOM aborts, timeouts
    "queue_depth_tokens":         "gauge",    # Pending tokens in request queue
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
```

Alert thresholds to set:
- `cost_per_1m_output_tokens` > 2× your baseline → something is wrong (traffic spike, failed GPU, KV cache thrashing).
- `kv_cache_utilization_pct` > 90% → risk of request preemption/OOM; scale out or reduce max context.
- `ttft_p99_ms` > SLO → prefill queue is backing up; consider [Disaggregated Prefill/Decode & Chunked Prefill](../07-inference-serving/08-disaggregated-chunked-prefill.html).
- `batch_size_mean` consistently < 10% of $B^*$ at peak hours → you are over-provisioned; scale down.

!!! key "Key Takeaways"

    - The latency–throughput–cost triangle is governed by the roofline: decode is bandwidth-bound until batch size exceeds $B^* = \text{FLOP/s} / \text{BW}$ (roughly 295 for H100 serving a 70B BF16 model). Below $B^*$, adding concurrent sequences improves throughput at zero latency cost.
    - Cost per million output tokens is $\text{\$/1M} = (R \times 10^6) / (T_{\text{sustained}} \times 3600)$. The dominant lever is sustained throughput, which is maximized by keeping GPU utilization near the saturation point.
    - Output tokens are 3–5× more expensive per token than input tokens — but in wall-clock GPU-time and dollars, not FLOPs: per-token compute (~2P) is the same for prefill and decode. The gap is bandwidth: each decode token pays for its own full weight read from HBM (memory-bound, low utilization), whereas parallel prefill amortizes one weight read across the whole prompt. Reasoning models that produce long chains-of-thought carry a significant cost multiplier.
    - FP8 and INT8 quantization are the single highest-leverage optimization: they halve or quarter decode bandwidth consumption with minimal quality degradation on modern hardware.
    - Continuous batching near $B^*$ is the core mechanism for cost efficiency; your scheduler should target this operating point subject to your TTFT/TBT SLOs.
    - For mixed workloads, disaggregate latency-sensitive (chat) and throughput-optimized (batch) serving tiers, using spot instances for the latter.
    - KV cache memory is a first-class resource: at long contexts (32k+), a single sequence can occupy 10+ GB of HBM, severely limiting concurrent requests. KV cache quantization (INT8) roughly doubles the number of long-context sequences a node can hold.
    - Always measure `cost_per_1m_output_tokens` as a first-class production metric alongside TTFT and GPU utilization. It surfaces inefficiencies that neither metric alone reveals.

!!! sota "State of the Art & Resources (2026)"
    Inference economics is a rapidly maturing discipline: by 2025 the field had moved from ad-hoc throughput maximization toward principled SLO-aware goodput optimization, hardware-native FP8 serving, and disaggregated prefill/decode architectures that are now standard in production clusters handling billions of tokens per day.

    **Foundational work**

    - [Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (2023)](https://arxiv.org/abs/2309.06180) — introduced the KV-cache paging abstraction underlying vLLM; the baseline cost and throughput benchmark the whole field references.
    - [Pope et al., *Efficiently Scaling Transformer Inference* (2022)](https://arxiv.org/abs/2211.05102) — Google's analytical roofline model for partitioning transformer inference across TPU/GPU slices; the canonical reference for hardware utilization math.

    **Recent advances (2023–2026)**

    - [Zhong et al., *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving* (2024)](https://arxiv.org/abs/2401.09670) — formalises "goodput" (SLO-attaining requests/s) and shows disaggregating prefill and decode onto separate GPU pools yields up to 4.5× better goodput.
    - [Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve* (2024)](https://arxiv.org/abs/2403.02310) — chunked prefill + stall-free scheduling; up to 6.9× higher throughput over vLLM at the same TTFT SLO on multi-GPU deployments.
    - [Qin et al., *Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving* (2024)](https://arxiv.org/abs/2407.00079) — Best Paper at FAST 2025; the production architecture behind Kimi, disaggregates KV cache across DRAM/SSD/NIC to boost throughput 59–498% in real traces.
    - [Yuan et al., *LLM Inference Unveiled: Survey and Roofline Model Insights* (2024)](https://arxiv.org/abs/2402.16363) — systematic roofline-model survey of quantization, batching, and parallelism strategies with an open-source LLM-Viewer analysis tool.

    **Open-source & tools**

    - [vllm-project/vllm](https://github.com/vllm-project/vllm) — the reference high-throughput serving engine (PagedAttention, continuous batching, FP8, LoRA hot-swap); its `/metrics` endpoint is the fastest way to observe the cost formulas in this chapter on real hardware.

    **Go deeper**

    - [DistServe Blog: Throughput is Not All You Need](https://haoailab.com/blogs/distserve/) — accessible write-up from Hao AI Lab on why goodput beats raw throughput as an operational metric.
    - [DigitalOcean: The LLM Inference Trilemma](https://www.digitalocean.com/blog/llm-inference-tradeoffs) — practitioner-oriented breakdown of the latency/throughput/cost triangle with a workload-type decision framework.
    - [Baseten: 33% Faster LLM Inference with FP8 Quantization](https://www.baseten.co/blog/33-faster-llm-inference-with-fp8-quantization/) — measured results on H100 hardware showing 33% throughput gain and 24% cost reduction per million tokens when moving from FP16 to FP8.

## Further Reading

- **Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention"**, SOSP 2023. The foundational paper behind vLLM; introduces the KV cache memory problem and paged allocation.
- **Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models"**, OSDI 2022. Introduces continuous batching (iteration-level scheduling) and quantifies the throughput gains.
- **Sheng et al., "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU"**, ICML 2023. Shows how to trade latency for throughput on memory-constrained hardware via offloading.
- **Pope et al., "Efficiently Scaling Transformer Inference"**, MLSys 2023 (Google). Detailed analysis of model parallelism strategies and hardware trade-offs for serving large models at scale.
- **Agrawal et al., "Sarathi-Serve: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills"**, OSDI 2024. Quantifies the prefill-decode interference problem and the benefit of chunked prefill for latency.
- **vLLM project** (github.com/vllm-project/vllm): The reference open-source implementation; its metrics endpoint is the best way to observe the concepts in this chapter in a real system.
- **LLM-Perf Leaderboard** (HuggingFace): Community benchmarks for tokens/s and cost across models, hardware, and quantization levels — useful for calibrating the numbers in this chapter against real measurements.
