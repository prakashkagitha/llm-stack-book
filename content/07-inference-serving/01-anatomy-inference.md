# 7.1 The Anatomy of LLM Inference: Prefill, Decode & The KV Cache

Training a large language model is a one-time (or occasional) capital expense. *Serving* it is a recurring operational cost that runs every second of every day, for every user, forever. Most of the money a deployed LLM ever consumes is spent here, in the inference loop. So if you want to understand why inference systems look the way they do — why vLLM and SGLang and TensorRT-LLM exist, why "continuous batching" is a phrase people say with reverence, why everyone obsesses over a thing called the KV cache — you have to start at the bottom, with the raw mechanics of how a transformer turns a prompt into tokens.

This chapter is the foundation for all of [Part VII](../07-inference-serving/02-continuous-batching.html). We will dissect a single forward-generation request into its two physically distinct phases — **prefill** and **decode** — and show that they have *opposite* performance characteristics. We will derive the **KV cache** from first principles, compute its size in gigabytes for real models, and prove why decode is bandwidth-bound rather than compute-bound. From there the latency vocabulary (TTFT, TPOT, ITL), the tokens-per-second ceiling, and Little's law for throughput will all fall out as consequences of two or three simple physical facts. By the end you should be able to look at a GPU spec sheet and a model config and estimate, on a napkin, how fast it can serve.

We assume you already understand the transformer forward pass; if attention itself is fuzzy, revisit [The Attention Mechanism From Scratch](../02-transformer/03-attention-from-scratch.html). We lean heavily on the [Roofline Model](../04-kernels-efficiency/01-roofline-performance.html) and on [GPU Architecture & The Memory Hierarchy](../01-foundations/08-gpu-architecture.html), so keep those mental models handy.

## Two Phases, One Model

{{fig:kv-cache-growth}}

A decoder-only LLM generates text **autoregressively**: it produces one token at a time, and each new token is conditioned on every token that came before it. Given a prompt of $S$ tokens, generating $T$ new tokens, the model effectively runs a forward pass for positions $S+1, S+2, \dots, S+T$. The crucial observation is that these positions are *not* all generated the same way.

The very first forward pass — the one that ingests the entire prompt — is special. The model sees all $S$ prompt tokens **at once** and produces the first output token. This is **prefill** (also called the "prompt" or "context encoding" phase). Every subsequent step ingests exactly **one** token (the one just generated) and produces the next. This is **decode** (the "generation" or "extension" phase).


{{fig:anatomy-prefill-decode-flow}}


Why split it this way? Because of the **causal attention mask**. Token at position $i$ may attend only to positions $\le i$. During prefill, we have all $S$ tokens in hand, so we can compute their attention in a single batched, masked matrix multiply — the whole prompt is processed in parallel, the same way a training forward pass works. During decode, we are generating position $S+t$ and *do not yet have* positions beyond it, so we are forced into a strictly sequential loop: one token, then the next, then the next.

These two phases run the *same weights* through the *same layers*. What differs is the **shape of the activation tensors** flowing through them, and that single difference — a sequence dimension of $S$ versus a sequence dimension of $1$ — flips the bottleneck from compute to memory. Everything in this chapter is a downstream consequence of that flip.

### A naive generation loop (and why it is quadratically wasteful)

Let us write the simplest possible autoregressive loop, with no cache at all, to expose the redundancy that the KV cache later eliminates.

```python
import torch

@torch.no_grad()
def generate_naive(model, input_ids, max_new_tokens):
    """
    The simplest correct autoregressive decoder: re-run the FULL forward
    pass on the entire growing sequence at every step. Correct, but O(T^2)
    in attention work because we recompute K and V for all old tokens
    every single step. This is what the KV cache fixes.
    """
    for _ in range(max_new_tokens):
        # logits over the WHOLE sequence; we only need the last position
        logits = model(input_ids).logits        # [B, seq_len, vocab]
        next_token = logits[:, -1, :].argmax(-1, keepdim=True)  # greedy
        input_ids = torch.cat([input_ids, next_token], dim=1)   # grow seq
    return input_ids
```

This is correct but catastrophically slow. At decode step $t$, the sequence has length $S + t$, and we recompute the keys and values for *all* of those positions even though only the newest token is new. The cumulative attention work scales like $\sum_{t} (S+t)^2$ — quadratic in total length and full of redundant recomputation. Almost everything from the previous step is identical and could have been saved. That observation is the entire motivation for the KV cache.

## The KV Cache: Mechanism and Math

Recall a single attention head. For query, key, and value projections, each token's hidden state $x_i \in \mathbb{R}^{d}$ is mapped to:

$$
q_i = x_i W_Q, \qquad k_i = x_i W_K, \qquad v_i = x_i W_V
$$

The attention output for query position $i$ is a softmax-weighted sum over the keys and values of all positions $j \le i$:

$$
o_i = \sum_{j \le i} \operatorname{softmax}_j\!\left(\frac{q_i \cdot k_j}{\sqrt{d_k}}\right) v_j
$$

Here is the key structural fact: **the keys $k_j$ and values $v_j$ for past positions do not depend on the current query position $i$.** Once we have computed $k_3$ and $v_3$ for token 3, those vectors are fixed forever — they are reused, unchanged, by every future query. The query $q_i$ changes each step, but the $K$ and $V$ of the past are constant.

So we cache them. The **KV cache** is exactly the running collection of all $k_j$ and $v_j$ computed so far, for every layer and every head. At each decode step we:

1. compute $q, k, v$ for the *single* new token only,
2. **append** its $k$ and $v$ to the cache,
3. compute attention of the one new query against the *entire* cached $K$ and $V$.

This turns the per-step attention from "recompute everything" into "compute one new KV and read the rest." The recomputation is gone; the price is the memory to hold the cache. We have made the classic systems trade: **spend memory to save compute.**

{{fig:kv-cache-append-vs-recompute}}

### A from-scratch KV-cached attention step

```python
import torch
import torch.nn.functional as F

class CachedSelfAttention(torch.nn.Module):
    """
    Single-head causal self-attention with an explicit KV cache.
    Strips away batching/multi-head bookkeeping to expose the mechanism.
    """
    def __init__(self, d_model, d_head):
        super().__init__()
        self.Wq = torch.nn.Linear(d_model, d_head, bias=False)
        self.Wk = torch.nn.Linear(d_model, d_head, bias=False)
        self.Wv = torch.nn.Linear(d_model, d_head, bias=False)
        self.scale = d_head ** -0.5
        self.k_cache = None  # [seq_so_far, d_head]
        self.v_cache = None

    def reset(self):
        self.k_cache = None
        self.v_cache = None

    def forward(self, x):
        # x: [n_new_tokens, d_model].
        #   - prefill: n_new_tokens == S (the whole prompt)
        #   - decode:  n_new_tokens == 1 (just the last token)
        q = self.Wq(x)                      # [n_new, d_head]
        k = self.Wk(x)                      # [n_new, d_head]
        v = self.Wv(x)                      # [n_new, d_head]

        # Append the new keys/values to the cache (the whole trick).
        if self.k_cache is None:
            self.k_cache, self.v_cache = k, v
        else:
            self.k_cache = torch.cat([self.k_cache, k], dim=0)
            self.v_cache = torch.cat([self.v_cache, v], dim=0)

        # Attend the new query(ies) against ALL cached keys/values.
        scores = (q @ self.k_cache.T) * self.scale  # [n_new, seq_so_far]

        # Causal mask: needed during prefill (n_new = S) so that token i
        # cannot see j > i. During decode (n_new = 1) the single new query
        # is allowed to see everything in the cache, so no mask is required.
        if q.shape[0] > 1:
            seq = self.k_cache.shape[0]
            past = seq - q.shape[0]
            i = torch.arange(q.shape[0]).unsqueeze(1) + past
            j = torch.arange(seq).unsqueeze(0)
            scores = scores.masked_fill(j > i, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        return attn @ self.v_cache          # [n_new, d_head]
```

Notice the asymmetry baked into the shapes. In prefill, `x` has $S$ rows and the attention score matrix is $S \times S$ — a big matmul. In decode, `x` has a single row and the score matrix is $1 \times (\text{seq\_so\_far})$ — a skinny matrix-times-matrix that is really a sequence of dot products. That shape difference is the whole story of why the two phases behave so differently, which we quantify next.

### KV cache size: the formula that governs everything

The cache must store, for every transformer layer, a key and a value vector for every token seen so far. Let:

- $L$ = number of layers,
- $H_{kv}$ = number of **key/value** heads (with Multi-Query or Grouped-Query Attention this is fewer than the number of query heads — see [Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html)),
- $d_h$ = dimension per head,
- $S$ = sequence length (tokens cached),
- $B$ = batch size (number of concurrent sequences),
- $P$ = bytes per element (2 for fp16/bf16, 1 for fp8/int8).

The factor of 2 below is for storing **both** K and V. The total KV-cache size in bytes is:

$$
\text{KV bytes} = 2 \cdot B \cdot S \cdot L \cdot H_{kv} \cdot d_h \cdot P
$$

It is worth dwelling on what this formula says and does *not* say. The KV cache grows **linearly** with sequence length and **linearly** with batch size. It does **not** depend on the model's MLP width, vocabulary, or the number of *query* heads — only the *KV* heads matter, which is precisely why GQA/MQA were invented to shrink it. And note $H_{kv} \cdot d_h$ is the total KV hidden dimension, often written $d_{kv}$.

```python
def kv_cache_bytes(batch, seq_len, n_layers, n_kv_heads, head_dim, dtype_bytes=2):
    """KV-cache size in bytes. Factor 2 = keys AND values."""
    return 2 * batch * seq_len * n_layers * n_kv_heads * head_dim * dtype_bytes

# Llama-3-style 8B: 32 layers, 8 KV heads (GQA), head_dim 128, bf16.
gib = kv_cache_bytes(batch=1, seq_len=8192, n_layers=32,
                     n_kv_heads=8, head_dim=128, dtype_bytes=2) / 2**30
print(f"{gib:.2f} GiB per 8k-token sequence")   # ~1.0 GiB
```

!!! example "Worked example: how many concurrent users fit in an A100?"
    Take a Llama-3-8B-class model with 32 layers, 8 KV heads (GQA), head dimension 128, served in bf16 (2 bytes). Per token, the cache costs:

    $$
    2 \cdot 32 \cdot 8 \cdot 128 \cdot 2 = 131{,}072 \text{ bytes} \approx 128 \text{ KiB/token}.
    $$

    A single 8,192-token context therefore needs $8192 \times 128\,\text{KiB} \approx 1.0$ GiB of KV cache.

    Now budget an 80 GB A100. The weights take $8\text{B} \times 2\,\text{bytes} = 16$ GB. Leave ~4 GB for activations, CUDA context, and fragmentation. That leaves roughly $80 - 16 - 4 = 60$ GB for KV cache. At 1 GiB per full 8k context, you can hold about **60 concurrent 8k sequences** — or many more short ones, or fewer long ones.

    Contrast with a model that used full Multi-Head Attention with 32 KV heads instead of 8: the per-token cost would be $4\times$ larger, ~512 KiB/token, and you would fit only ~15 concurrent sequences. **This single ratio is why every modern serving model uses GQA or MLA.** The KV cache, not the weights, is usually what limits how many users you can serve at once.

{{fig:kv-cache-vs-weights-budget}}

The collapse from "weights are the big thing" to "the KV cache is the big thing" is the central surprise of inference engineering. For long contexts and high concurrency, the cache can dwarf the model itself. Managing it well — packing it tightly, paging it, reusing shared prefixes — is the subject of [PagedAttention & KV-Cache Memory Management](../04-kernels-efficiency/06-paged-attention-kv.html) and [Prefix Caching & KV-Cache Reuse](../07-inference-serving/07-prefix-caching.html). This chapter just establishes *why* it matters so much.

## Why Decode Is Memory-Bound and Prefill Is Compute-Bound

This is the most important concept in the whole of inference serving, so we will derive it carefully using the **arithmetic intensity** lens from the [Roofline Model](../04-kernels-efficiency/01-roofline-performance.html). Arithmetic intensity is the ratio of floating-point operations performed to bytes of memory moved:

$$
I = \frac{\text{FLOPs}}{\text{bytes moved}} \quad \left[\frac{\text{FLOP}}{\text{byte}}\right]
$$

A GPU has two relevant peak rates: compute throughput (FLOP/s) and memory bandwidth (bytes/s). The **ridge point** is their ratio, $I^\* = \text{FLOP/s} \div \text{bytes/s}$. If a kernel's intensity $I > I^\*$ it is **compute-bound** (limited by the math units); if $I < I^\*$ it is **memory-bound** (limited by how fast you can feed data from HBM). For a modern data-center GPU the ridge point is on the order of a few hundred FLOP/byte. An A100 delivers roughly 312 TFLOP/s of bf16 tensor-core compute against roughly 2.0 TB/s of HBM bandwidth, giving a ridge near $\sim$150 FLOP/byte. An H100 is higher on both axes but the ridge is in the same ballpark. Hold that "~100–300 FLOP/byte" number in your head.

### The decisive quantity: tokens per forward pass

Consider one matrix multiply that dominates the transformer: a token's hidden vector times a weight matrix $W \in \mathbb{R}^{d \times d}$. If we process $N$ tokens at once, we compute $X W$ where $X$ is $N \times d$. The cost is:

- **FLOPs:** $2 N d^2$ (multiply-add over an $N\times d$ by $d \times d$ matmul).
- **Bytes moved:** we must read $W$ once ($d^2$ elements $\times P$ bytes) plus the activations ($N d \cdot P$). For the large weight matrices that dominate an LLM, the weight read dominates when $N$ is small, so bytes $\approx d^2 P$.

Arithmetic intensity is therefore approximately:

$$
I \approx \frac{2 N d^2}{d^2 P} = \frac{2N}{P}
$$

**The intensity scales with $N$, the number of tokens processed together.** That is the entire argument. Read it twice.

- **Prefill** processes the whole prompt at once: $N = S$, often hundreds or thousands of tokens. Even modest $S$ pushes $I = 2N/P$ well above the ridge point, so prefill is **compute-bound** — it saturates the tensor cores. Doubling your FLOPs roughly doubles prefill throughput; the weights are read once and amortized over many tokens.

- **Decode** processes exactly one token per sequence: $N = 1$ (for a single request). Then $I \approx 2/P = 1$ FLOP/byte for fp16 — *two orders of magnitude* below the ridge point. Decode is therefore deeply **memory-bound**. The GPU spends its time streaming the entire weight matrix (and the entire KV cache) from HBM to perform a tiny amount of arithmetic, and the tensor cores sit mostly idle.


{{fig:anatomy-roofline-intensity-numberline}}


This is why **batching helps decode enormously but barely helps prefill.** Prefill already saturates compute, so cramming more sequences in does not raise per-token throughput much (and can even hurt latency). Decode, by contrast, reads the *same weights* regardless of how many sequences share the step — so if you decode $B$ sequences together, you read $W$ once and reuse it across all $B$ tokens, lifting $N$ from $1$ to $B$ and intensity from $\approx 2/P$ to $\approx 2B/P$. Batching decode is essentially free FLOPs riding on memory traffic you were paying for anyway. **This single insight is the entire reason continuous batching exists** — see [Continuous Batching & Request Scheduling](../07-inference-serving/02-continuous-batching.html).

### A measurement you can run

You can *feel* the memory-bound nature of decode directly. The following microbenchmark shows that decode latency per token is nearly flat as batch size grows (because you are bandwidth-limited, not compute-limited), until the batch gets large enough to finally start saturating compute.

```python
import torch, time

def bench_decode(model, batch_sizes, d_model, n_steps=50, device="cuda"):
    """
    Single-token (decode-shaped) forward passes at several batch sizes.
    Memory-bound behavior => per-step latency stays ~flat as batch grows,
    so tokens/sec scales almost linearly with batch. Compute-bound behavior
    would instead show latency rising proportionally with batch.
    """
    model.eval()
    for B in batch_sizes:
        x = torch.randn(B, 1, d_model, device=device, dtype=torch.bfloat16)
        # warmup (CUDA lazy init, autotuning, caches)
        for _ in range(5):
            with torch.no_grad():
                model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            with torch.no_grad():
                model(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / n_steps
        print(f"B={B:4d}  {dt*1e3:6.2f} ms/step  "
              f"{B/dt:8.0f} tok/s  {dt*1e3/B:6.3f} ms/tok")
```

On real hardware you will see the per-token time barely move from $B=1$ to $B=16$ or $B=32$ — you are getting those extra tokens "for free" off the same weight reads — and only at large batch do you finally cross into the compute-bound regime where latency climbs. That flat region is the memory-bound signature, and exploiting it is the single biggest lever in throughput-oriented serving.

## The Tokens-Per-Second Ceiling

Because single-stream decode is memory-bound, we can compute a remarkably tight upper bound on generation speed using nothing but the model size and the GPU's memory bandwidth. Every decode step must read **every weight** of the model from HBM at least once (each weight participates in some matmul for the new token). So the time for one decode step is bounded below by the time to stream all the weights:

$$
t_{\text{step}} \;\ge\; \frac{\text{model bytes}}{\text{memory bandwidth}}
= \frac{N_{\text{params}} \cdot P}{\text{BW}}
$$

and the per-stream decode throughput is bounded above by the reciprocal:

$$
\text{tokens/sec} \;\le\; \frac{\text{BW}}{N_{\text{params}} \cdot P}
$$

This "weights divided by bandwidth" estimate is one of the most useful back-of-envelope tools in the field. (At large context lengths you must also add the KV-cache bytes read per step; for short contexts the weight read dominates.)

!!! example "Worked example: the speed-of-light decode rate"
    A 70B-parameter model in fp16 occupies $70 \times 10^9 \times 2 = 140$ GB of weights. On an A100 with ~2.0 TB/s of HBM bandwidth, a single decode step cannot be faster than:

    $$
    t_{\text{step}} \ge \frac{140 \times 10^9}{2.0 \times 10^{12}} = 0.070\ \text{s} = 70\ \text{ms}.
    $$

    That caps single-stream decode at about $1/0.070 \approx 14$ tokens/sec — and that is the *optimistic* ceiling assuming perfect bandwidth utilization (real kernels hit perhaps 60–80% of peak). This is exactly why a 70B model "feels slow" token-by-token and why people quantize weights to int4: dropping $P$ from 2 to 0.5 bytes shrinks the weight read 4×, lifting the decode ceiling to ~55 tokens/sec on the same GPU. Quantization buys *decode speed* primarily by shrinking memory traffic, not by adding FLOPs — see [Quantization II](../04-kernels-efficiency/08-quantization-formats-qat.html).

    Compare an 8B model: $16$ GB of weights at 2.0 TB/s gives $t_{\text{step}} \ge 8$ ms, a ceiling near 125 tokens/sec. Smaller models are not just cheaper to train — they decode proportionally faster because there is less to stream.

The flip side: this ceiling is **per stream**. If you batch $B$ independent sequences, you still read the weights only once per step and serve $B$ tokens, so **aggregate** throughput is roughly $B \times$ the single-stream rate, right up until you exhaust KV-cache memory or finally saturate compute. Single-stream latency and aggregate throughput are different axes, and batching trades one for the other. Hold this thought — it is the seed of Little's law below.

## The Latency Vocabulary: TTFT, TPOT, ITL

Streaming LLM applications (a chat UI typing tokens at you) are not characterized by a single "latency" number. The user's experience splits along the prefill/decode boundary, and the industry has settled on a small, precise vocabulary. Learn these exact definitions — they appear in every serving SLA and every interview.

| Metric | Name | What it measures | Dominated by |
|---|---|---|---|
| **TTFT** | Time To First Token | Request arrival → first output token streamed | Prefill (compute-bound) + queueing |
| **TPOT** | Time Per Output Token | Average time between subsequent tokens | Decode (memory-bound) |
| **ITL** | Inter-Token Latency | Per-step gap between tokens (often used interchangeably with TPOT, but ITL is the per-step value while TPOT is the mean) | Decode + scheduling jitter |
| **E2E** | End-to-End Latency | Arrival → last token | $\text{TTFT} + (T-1)\cdot\text{TPOT}$ |

{{fig:latency-timeline-ttft-tpot}}

The decomposition is clean and worth memorizing:

$$
\text{Latency}_{\text{E2E}} \approx \underbrace{\text{TTFT}}_{\text{prefill of } S \text{ tokens}} + \underbrace{(T-1)\cdot\text{TPOT}}_{\text{decode of } T-1 \text{ more tokens}}
$$

**TTFT** is what the user feels as "responsiveness" — how long the cursor blinks before text appears. It is governed by prefill, so it scales with prompt length $S$ (longer prompts take longer to encode) and by queueing delay under load. Because prefill is compute-bound, TTFT is improved by more FLOPs, by **chunked prefill** (slicing a long prompt so it interleaves with ongoing decodes), and by **prefix caching** (skipping prefill entirely for a shared prompt prefix). See [Disaggregated Prefill/Decode & Chunked Prefill](../07-inference-serving/08-disaggregated-chunked-prefill.html).

**TPOT** is what the user feels as "typing speed" — the steady cadence once text starts flowing. It is governed by decode, so it is memory-bound and follows the tokens/sec ceiling above. A common product target is to keep TPOT below the human reading rate (very roughly 5–10 tokens/sec is readable, so a TPOT under ~100–200 ms feels smooth). Crucially, batching *raises aggregate throughput but can raise per-stream TPOT*, because a larger decode batch takes slightly longer per step. This tension — throughput versus per-user latency — is the central knob of serving.

```python
def measure_streaming_latency(stream_fn, prompt):
    """
    Wrap a streaming generation call and report TTFT, mean TPOT, and the
    per-token ITL series. `stream_fn(prompt)` must yield one token at a time.
    """
    import time
    t_start = time.perf_counter()
    t_prev = None
    ttft = None
    itls = []                      # inter-token latencies (seconds)
    n_tokens = 0
    for _token in stream_fn(prompt):
        now = time.perf_counter()
        if ttft is None:
            ttft = now - t_start   # arrival -> first token
        else:
            itls.append(now - t_prev)
        t_prev = now
        n_tokens += 1
    tpot = sum(itls) / len(itls) if itls else float("nan")
    return {
        "ttft_ms": ttft * 1e3,
        "tpot_ms": tpot * 1e3,                # mean inter-token latency
        "p99_itl_ms": sorted(itls)[int(0.99 * len(itls)) - 1] * 1e3 if itls else None,
        "output_tokens": n_tokens,
        "decode_tps": (n_tokens - 1) / sum(itls) if itls else float("nan"),
    }
```

When you benchmark a serving system, you report these as **distributions, not point values** — p50 and p99 TTFT, p50 and p99 TPOT — because tail latency under load is what breaks SLAs. A system with great median TPOT but a terrible p99 (caused by, say, a giant prompt prefill blocking the decode batch) feels janky and stutters. This is exactly the pathology chunked prefill was designed to cure.

!!! warning "Common pitfall: optimizing the wrong metric"
    Throughput (tokens/sec aggregated across all users) and per-user latency (TTFT, TPOT) are **in tension**, and you cannot maximize both. Bigger batches and longer scheduling windows raise throughput (better USD-per-token economics) but worsen tail latency. A batch-offline summarization job should be tuned for throughput; an interactive chat assistant must protect TTFT and p99 TPOT. The classic mistake is to quote a single "tokens/sec" number with no batch size, no context length, and no latency constraint — it is almost meaningless. Always specify the operating point. We unpack the cost side of this trade in [Inference Economics: Latency, Throughput & Cost](../07-inference-serving/12-inference-economics.html).

## Little's Law: From Latency to Throughput

To reason about a *fleet* serving many users — not a single request — we borrow a result from queueing theory. **Little's law** states that for any stable system in steady state, the average number of in-flight requests equals the arrival rate times the average time each request spends in the system:

$$
L = \lambda \cdot W
$$

where $L$ is the average number of concurrent requests resident in the system, $\lambda$ is the throughput (requests completed per second, which in steady state equals the arrival rate), and $W$ is the average latency (time in system). It is astonishingly general — it assumes nothing about the arrival distribution or service discipline, only stability.

For LLM serving we apply it at the *token* level, which makes the bandwidth story precise. Let $L$ be the number of sequences we can hold concurrently (capped by KV-cache memory), let $W$ be the average time a sequence stays resident (roughly its total generation time, $\approx T \cdot \text{TPOT}$), and let $\lambda$ be the sequence completion rate. Rearranged:

$$
\lambda = \frac{L}{W} \quad\Longrightarrow\quad \text{throughput (req/s)} = \frac{\text{max concurrent sequences}}{\text{avg time per sequence}}
$$

This ties the whole chapter together. The numerator $L$ is set by **KV-cache memory** — the formula from earlier directly bounds how many sequences fit. The denominator $W$ is set by **decode speed** — the bandwidth-bound TPOT ceiling. So your serving throughput is fundamentally governed by the two physical facts we derived: how much KV cache fits in HBM, and how fast HBM bandwidth lets you decode.

!!! example "Worked example: sizing a fleet with Little's law"
    Suppose our 8B model on an 80 GB A100 holds $L \approx 60$ concurrent 8k-token sequences (from the earlier KV-cache budget). Suppose each request generates on average $T = 500$ tokens and our batched decode achieves a per-stream TPOT of about 20 ms (decode steps are fast because batching amortizes the weight read). Then the average time-in-system per sequence is:

    $$
    W \approx 500 \times 0.020\ \text{s} = 10\ \text{s (decode)} \; + \; \text{TTFT} \approx 10\text{–}11\ \text{s}.
    $$

    By Little's law the sustainable request throughput is:

    $$
    \lambda = \frac{L}{W} = \frac{60}{10.5} \approx 5.7\ \text{requests/sec}.
    $$

    At 500 output tokens each that is about **2,850 output tokens/sec** of aggregate decode throughput from one GPU. Push arrival rate above $\lambda$ and the queue grows without bound — TTFT climbs, the system becomes unstable, and you must add GPUs or shorten outputs. This back-of-envelope is exactly how capacity planners size inference fleets, and it shows concretely how KV-cache capacity ($L$) and decode bandwidth ($W$) jointly set the ceiling.

The practical consequence: to raise throughput you either **increase $L$** (shrink the KV cache via GQA/MLA, quantize the cache, page it more tightly, share prefixes) or **decrease $W$** (faster decode via quantized weights, speculative decoding, better kernels). Every serving optimization in Part VII is, underneath, an attack on one of these two terms. Continuous batching keeps $L$ as full as possible at all times; PagedAttention lets you pack $L$ tighter without fragmentation; speculative decoding shrinks $W$ by emitting multiple tokens per forward pass.

!!! interview "Interview Corner"
    **Q:** A teammate says "our 7B model only generates 30 tokens/sec on an H100, the GPU must be broken — it has 1000 TFLOPS." How do you respond?

    **A:** The GPU is almost certainly fine; single-stream decode is *memory-bandwidth-bound, not compute-bound*, so TFLOPS is the wrong number to look at. Each decode step generates one token but must stream all ~14 GB of fp16 weights from HBM. At ~3 TB/s that is a hard floor of roughly $14/3000 \approx 4.7$ ms per step, i.e. a single-stream ceiling around 200 tokens/sec; 30 tokens/sec just means we are achieving a fraction of peak bandwidth (kernel overhead, small context, or an unfused implementation). The tensor cores are mostly *idle* during decode — their 1000 TFLOPS is unused because arithmetic intensity is about 1 FLOP/byte, two orders of magnitude below the H100's ridge point. The fix is not "a faster GPU's FLOPS" but to (1) raise arithmetic intensity by **batching** many sequences so one weight read serves many tokens, (2) shrink the weight bytes via **int4/int8 quantization**, or (3) emit multiple tokens per pass via **speculative decoding**. I'd also confirm we are actually using a KV cache and a fused-attention kernel, since a naive O($T^2$) loop or an unfused path can leave bandwidth on the table.

## Putting It Together: An Annotated Generation Loop

To consolidate everything, here is a complete, cache-using generation loop with the prefill/decode split made explicit and annotated with the performance regime of each phase. This is the conceptual skeleton inside every production engine; what vLLM and SGLang add is *scheduling many of these loops together* and *managing the KV memory*, which later chapters cover.

```python
import torch

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=256,
             eos_id=None, device="cuda"):
    """
    KV-cached autoregressive generation, prefill/decode split made explicit.
    Uses a HF-style model whose forward() accepts past_key_values and
    returns updated ones (use_cache=True). Greedy for clarity; swap in
    your sampler from the decoding chapter for real use.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    S = input_ids.shape[1]

    # ---- PREFILL ----------------------------------------------------------
    # One big forward pass over all S prompt tokens. COMPUTE-BOUND: the SxS
    # attention and the S-row matmuls saturate the tensor cores. This builds
    # the initial KV cache and produces the first output token. Its cost is
    # what the user experiences as TTFT.
    out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values                      # the KV cache, all layers
    next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
    generated = [next_id]

    # ---- DECODE -----------------------------------------------------------
    # Loop, one token at a time. MEMORY-BOUND: each step feeds a single token,
    # so it streams the full weights + KV cache from HBM to do tiny math. The
    # per-step time is TPOT. We pass only the NEW token plus the cache, never
    # the whole sequence again -- that is the KV cache doing its job.
    for _ in range(max_new_tokens - 1):
        out = model(input_ids=next_id, past_key_values=past, use_cache=True)
        past = out.past_key_values                  # cache grew by one token
        next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
        generated.append(next_id)
        if eos_id is not None and next_id.item() == eos_id:
            break

    return tokenizer.decode(torch.cat(generated, dim=1)[0])
```

Three things to internalize from this skeleton. First, **prefill runs once, decode runs hundreds of times** — for a typical chat turn with a 1,000-token prompt and a 300-token answer, almost all wall-clock time is in the decode loop even though prefill touches more tokens, because decode is the slow, memory-bound part repeated 300 times. Second, **the cache (`past`) is the entire state** — its size is the formula we derived, and managing it is the hardest part of real serving. Third, **the only input to each decode step is one token plus the cache** — which is exactly why many sequences can share a decode step (continuous batching) and why the weight read can be amortized across them.

!!! tip "Practitioner tip: profile the split before optimizing"
    Before reaching for any fancy technique, measure how your real traffic divides between prefill and decode. A retrieval-augmented or long-document workload with 8k-token prompts and 200-token answers is **prefill-heavy** — your wins come from chunked prefill, prefix caching, and prefill FLOPs. A chatty agent loop with short prompts and long reasoning traces is **decode-heavy** — your wins come from batching, weight quantization, and speculative decoding. The same GPU, the same model, two completely different optimization playbooks. The ratio $S : T$ in your traffic tells you which knobs matter, and it is the first thing a good serving engineer measures.

## Key Takeaways

!!! key "Key Takeaways"
    - LLM inference has **two phases with opposite characteristics**: prefill processes the whole prompt in parallel and is **compute-bound**; decode generates one token at a time and is **memory-bandwidth-bound**. The flip comes entirely from the per-pass token count $N$ collapsing from $S$ to $1$.
    - The **KV cache** stores keys and values for all past tokens so they need not be recomputed each step, trading memory for compute. Its size is $2 \cdot B \cdot S \cdot L \cdot H_{kv} \cdot d_h \cdot P$ bytes — linear in batch and sequence length, and the usual *binding constraint* on how many users you can serve.
    - Decode's arithmetic intensity is $\approx 2N/P$ FLOP/byte; at $N=1$ it sits ~100× below a data-center GPU's ridge point, so the tensor cores idle while HBM streams the weights. **Batching raises $N$ and is therefore nearly free throughput** — the foundation of continuous batching.
    - The single-stream **decode speed ceiling** is $\text{BW} / (N_{\text{params}} \cdot P)$ — model weight bytes divided by memory bandwidth. This is why big models feel slow and why quantization (smaller $P$) primarily buys decode speed.
    - The latency vocabulary: **TTFT** (prefill-bound, the responsiveness the user feels first), **TPOT/ITL** (decode-bound, the typing cadence), and **E2E** $\approx$ TTFT $+ (T-1)\cdot$TPOT. Always report distributions (p50/p99), not point values.
    - **Throughput and per-user latency are in tension.** Larger batches and longer scheduling windows raise aggregate tokens/sec but worsen tail latency; pick the operating point for your workload.
    - **Little's law** ties it together: throughput $\approx$ (max concurrent sequences) / (avg time per sequence) — numerator set by KV-cache capacity, denominator by decode bandwidth. Every Part VII optimization attacks one of these two terms.
    - Before optimizing, **profile your prefill:decode split** ($S:T$): prefill-heavy and decode-heavy traffic need entirely different playbooks.

!!! sota "State of the Art & Resources (2026)"
    Production LLM serving in 2026 is shaped by two dominant insights from this chapter: decode is memory-bandwidth-bound, so every major system optimizes KV-cache packing and weight access; and prefill and decode have opposite bottlenecks, so disaggregating them onto separate hardware is now a mainstream strategy. The open-source ecosystem — led by vLLM and SGLang — has made these ideas broadly accessible while research continues to push KV-cache compression and speculative decoding further.

    **Foundational papers**

    - [Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (2023)](https://arxiv.org/abs/2309.06180) — introduces PagedAttention (OS-style virtual memory for KV cache), the basis of vLLM and continuous batching at scale.
    - [Leviathan et al., *Fast Inference from Transformers via Speculative Decoding* (2022)](https://arxiv.org/abs/2211.17192) — shows that a small draft model plus parallel verification can emit multiple tokens per forward pass, directly attacking the memory-bound decode bottleneck.

    **Pushing the frontier (2024–2026)**

    - [Zhong et al., *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving* (2024)](https://arxiv.org/abs/2401.09670) — assigns prefill and decode to separate GPU pools, eliminating phase interference and enabling independent scaling; accepted OSDI 2024.
    - [Patel et al., *Splitwise: Efficient Generative LLM Inference Using Phase Splitting* (2024)](https://arxiv.org/abs/2311.18677) — concurrent Microsoft Research work reaching the same disaggregation conclusion; presented at ISCA 2024.
    - [Liu et al., *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache* (2024)](https://arxiv.org/abs/2402.02750) — quantizes KV cache to 2-bit per element without fine-tuning, shrinking cache footprint 2.6× and enabling larger batch sizes; ICML 2024.
    - [Hooper et al., *KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization* (2024)](https://arxiv.org/abs/2401.18079) — per-channel and non-uniform quantization strategies enabling extreme-context inference; NeurIPS 2024.

    **Open-source & tools**

    - [vllm-project/vllm](https://github.com/vllm-project/vllm) — the de facto standard high-throughput serving engine; implements PagedAttention, continuous batching, prefix caching, and speculative decoding.
    - [sgl-project/sglang](https://github.com/sgl-project/sglang) — SGLang runtime with RadixAttention for automatic KV-cache reuse across requests sharing common prefixes.

    **Go deeper**

    - [Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs* (2023)](https://arxiv.org/abs/2312.07104) — the RadixAttention paper; shows prefix-aware KV-cache management integrated directly into the serving runtime; NeurIPS 2024.
    - [vLLM Documentation](https://docs.vllm.ai/en/latest/) — official docs covering PagedAttention internals, automatic prefix caching design, and deployment guides.

## Further Reading

- Vaswani et al., *Attention Is All You Need* (2017) — the original transformer and the attention computation whose K/V we cache.
- Pope et al., *Efficiently Scaling Transformer Inference* (Google, 2022) — the canonical analysis of inference arithmetic, memory-bound decode, and the prefill/decode distinction; the source of much of this chapter's framing.
- Dao et al., *FlashAttention* and *FlashAttention-2* — IO-aware attention kernels that make prefill (and long-context decode) memory-efficient; see [FlashAttention I](../04-kernels-efficiency/02-flash-attention-1.html).
- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM, 2023) — the paper that reframed KV-cache memory as a paging problem; see [vLLM Internals](../07-inference-serving/03-vllm-internals.html).
- Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need* (2019) — introduces Multi-Query Attention, the first major attack on KV-cache size; see [MHA, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html).
- Williams, Waterman & Patterson, *Roofline: An Insightful Visual Performance Model* (2009) — the arithmetic-intensity model underpinning the compute- vs memory-bound argument; see [The Roofline Model](../04-kernels-efficiency/01-roofline-performance.html).
- Little, *A Proof for the Queuing Formula $L = \lambda W$* (1961) — the throughput law used for fleet sizing.
