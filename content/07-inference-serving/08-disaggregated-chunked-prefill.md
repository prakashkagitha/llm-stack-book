# 7.8 Disaggregated Prefill/Decode & Chunked Prefill

Autoregressive generation has two fundamentally different compute phases: **prefill**, which processes the entire prompt in parallel, and **decode**, which generates one token at a time. These phases live inside the same GPU in almost every deployed system — but they have radically different resource profiles. Mixing them together on the same hardware leads to a class of performance problems that have quietly been the biggest source of latency waste in production LLM serving since the first continuous-batching schedulers were deployed.

This chapter tears apart the prefill/decode interference problem, explains why separating these phases onto different hardware pools can dramatically improve both latency and throughput, and covers the practical engineering — KV-cache transfer, chunked prefill scheduling, and the systems DistServe and Splitwise that have formalized these ideas. Readers already comfortable with continuous batching ([Continuous Batching & Request Scheduling](../07-inference-serving/02-continuous-batching.html)) and PagedAttention ([PagedAttention & KV-Cache Memory Management](../04-kernels-efficiency/06-paged-attention-kv.html)) will find this chapter the natural next step toward cutting-edge serving system design.

## The Two Phases Are Nothing Alike

Before we can appreciate why disaggregation helps, we need a precise picture of what each phase actually does to the hardware.

### Prefill: Compute-Bound Matrix Multiplications

During prefill, the server processes a sequence of length $T_p$ (the prompt length) in a single forward pass. The dominant cost is the projection matrices in each attention layer and each MLP layer, which see batched matrix multiplications of shape $[T_p, d_\text{model}]$ against $[d_\text{model}, d_\text{ff}]$. With a long prompt, $T_p$ is large, the matrices are fat, and modern GPUs can achieve high arithmetic intensity — often north of 100–200 FLOPs/byte — pushing utilization close to the peak TFLOP ceiling.

The arithmetic intensity for a single linear layer with input $[B \cdot T_p, d]$ and weight $[d, d']$ is roughly:

$$
I_\text{prefill} = \frac{2 \cdot B \cdot T_p \cdot d \cdot d'}{(B \cdot T_p \cdot d + d \cdot d') \cdot \text{bytes\_per\_element}}
$$

For large $T_p$, the numerator grows quadratically in $T_p$ (for attention, $O(T_p^2)$ through the $QK^\top$ product) while the weight loading cost is constant, so the operation is firmly compute-bound.

### Decode: Memory-Bandwidth-Bound Vector-Matrix Multiplications

During decode, we generate one token at a time. The batch dimension is the number of concurrent sequences, $B_d$. Each layer executes a matrix-vector multiply: shape $[B_d, d_\text{model}]$ times $[d_\text{model}, d']$. Unless $B_d$ is in the hundreds, this is entirely memory-bandwidth-bound: we stream gigabytes of weights from HBM every step just to do a handful of FLOPs per byte.

For a 70B parameter model in BF16 (140 GB of weights), each decode step reads roughly 140 GB from HBM regardless of batch size. On an A100 SXM with HBM bandwidth of roughly 2 TB/s, that's on the order of 70 ms of pure bandwidth time per step — leaving essentially no room for compute to hide. In contrast, a H100 (3.35 TB/s bandwidth) gets this to around 40 ms in theory.

The key insight: **prefill wants to be compute-bound and loves big batches; decode wants high bandwidth and is bottlenecked by memory, not FLOPs.** These constraints point toward different hardware configurations: fewer, faster GPUs (or GPUs with high TFLOP/s) for prefill, and more, bandwidth-rich GPUs for decode.

{{fig:prefill-decode-reuse-contrast}}

### The Interference Problem

Now imagine both phases run on the same GPU pool under a continuous batching scheduler (see [Continuous Batching & Request Scheduling](../07-inference-serving/02-continuous-batching.html)). Every iteration of the inference engine processes a mixed batch: some sequences are in prefill, others are in decode. This mixing has two serious consequences:

1. **Prefill preempts decode.** A long incoming prompt (say 8,192 tokens) takes tens of milliseconds to prefill. During that time, every decode-only request that was already generating is paused. The time-to-next-token for existing users spikes unexpectedly, violating service-level objectives (SLOs) on P99 latency.

2. **Decode throughput degrades under mixed batches.** When a decode step includes prefill tokens from new requests, the attention kernel must handle variable-length sequences with very different KV-cache patterns. GPU kernel efficiency drops; memory fragmentation increases; the effective batch size for decode shrinks.

This is the **prefill-decode interference problem**. It is not a theoretical concern — production teams have measured it causing P99 time-to-first-token (TTFT) and P99 inter-token latency (ITL) to exceed SLOs by 2–5x under moderate load.

## Disaggregated Prefill/Decode: The Architecture

The core idea of disaggregation is simple to state: **run prefill on a dedicated pool of instances (prefill workers), run decode on a separate pool (decode workers), and transfer the KV cache between them.**


{{fig:disagg-pd-architecture}}


### Lifecycle of a Request

1. A request arrives at the router with a prompt of $T_p$ tokens.
2. The router schedules it on a prefill worker. The prefill worker runs a single forward pass over the entire prompt, producing the first output token and — critically — the full KV cache for layers $1 \ldots L$.
3. The KV cache (a tensor of shape $[L, 2, T_p, n_\text{kv\_heads}, d_\text{head}]$) is transmitted to a decode worker via a high-speed interconnect (NVLink, RDMA over InfiniBand, or PCIe depending on cluster topology).
4. The decode worker receives the KV cache, places it into its paged KV-cache allocator, and begins generating tokens one step at a time — without interfering with any prefill computation.
5. Generated tokens stream back to the client.

### Why This Helps

Decode workers never see a long prefill stall. Every step they execute is a pure decode step over a batch of active sequences. Time-to-next-token becomes predictable and low. Meanwhile, prefill workers handle new requests at maximum compute efficiency — no decode sequences competing for memory bandwidth.

## KV Cache Transfer: The Engineering Challenge

Transferring KV caches is not free. For a single request with prompt length $T_p = 4096$, a 70B model with 80 layers, 8 KV heads (GQA), and $d_\text{head} = 128$, storing in BF16:

$$
\text{KV size} = L \times 2 \times T_p \times n_\text{kv} \times d_\text{head} \times 2\ \text{bytes}
$$

$$
= 80 \times 2 \times 4096 \times 8 \times 128 \times 2\ \text{bytes} = 1,073,741,824\ \text{bytes} \approx 1\ \text{GB}
$$

Transferring 1 GB at NVLink4 speeds (~900 GB/s bidirectional between two H100s) takes about 1 ms. Over InfiniBand HDR100 (100 Gb/s ≈ 12.5 GB/s), it takes about 80 ms — nearly as long as the prefill itself for large prompts. The transfer mechanism fundamentally shapes the architecture:

| Interconnect | Bandwidth | 1 GB KV transfer time | Notes |
|---|---|---|---|
| NVLink4 (H100 NVSwitch) | ~900 GB/s | ~1 ms | On the same NVSwitch fabric |
| NVLink3 (A100 NVSwitch) | ~600 GB/s | ~1.7 ms | |
| PCIe 5.0 x16 | ~64 GB/s | ~16 ms | Cross-node CPU path |
| InfiniBand HDR (100 Gb/s) | ~12.5 GB/s | ~80 ms | Long-haul cross-node |
| InfiniBand NDR (400 Gb/s) | ~50 GB/s | ~20 ms | State-of-the-art IB |

For real deployments, the recommendation is: keep prefill and decode workers on the same NVSwitch fabric (same node or adjacent nodes connected via NVSwitch) to keep transfer latency under a few milliseconds. If that is not possible, pipeline the transfer with decode (start decoding even as later layers' KV caches arrive) to overlap transfer and computation.

### Layer-Wise Pipelining

A practical optimization is to transfer KV caches layer by layer as they are computed during prefill, not as a single bulk transfer at the end. The decode worker can start generating once it has received all layers. With layer-wise pipelining, the decode worker can begin as soon as the last layer's KV cache arrives, reducing end-to-end latency by overlapping network transfer with later prefill layers.

```python
# Pseudocode: layer-wise KV transfer from prefill worker to decode worker
# (simplified, assumes a hypothetical RPC / RDMA abstraction)

import torch
from typing import Tuple

class PrefillWorker:
    def __init__(self, model, kv_sender):
        self.model = model
        self.kv_sender = kv_sender  # e.g. a RDMA/NVLink send handle

    def prefill_and_stream_kv(
        self,
        input_ids: torch.Tensor,   # [1, T_p]
        request_id: str,
    ) -> torch.Tensor:
        """
        Run prefill layer-by-layer, streaming each layer's KV
        to the paired decode worker as we go.
        Returns the first output token (greedy) so TTFT is fast.
        """
        x = self.model.embed(input_ids)          # [1, T_p, d_model]
        first_token = None

        for layer_idx, layer in enumerate(self.model.layers):
            # Standard attention + MLP forward
            x, kv_cache = layer.forward_with_kv(x)
            # kv_cache shape: [2, T_p, n_kv_heads, d_head]

            # Fire-and-forget async send — does NOT block prefill forward pass
            self.kv_sender.send_async(
                request_id=request_id,
                layer_idx=layer_idx,
                kv=kv_cache,
            )

        # Compute logits only for last position (first output token)
        logits = self.model.lm_head(x[:, -1, :])   # [1, vocab]
        first_token = logits.argmax(dim=-1)          # greedy; real systems sample
        return first_token


class DecodeWorker:
    def __init__(self, model, kv_receiver, paged_kv_manager):
        self.model = model
        self.kv_receiver = kv_receiver
        self.kv_mgr = paged_kv_manager

    def receive_kv_and_decode(
        self,
        request_id: str,
        first_token: torch.Tensor,
        max_new_tokens: int,
    ):
        """
        Wait for all layers' KV caches to arrive, then decode.
        In a real system this overlaps with prefill's later layers.
        """
        # Block until all L layers have been received
        kv_caches = self.kv_receiver.collect(request_id)
        # kv_caches: list of [2, T_p, n_kv_heads, d_head] tensors, one per layer

        # Allocate paged KV slots and copy into the page table
        slot = self.kv_mgr.allocate(request_id, kv_caches)

        generated = [first_token.item()]
        cur_token = first_token
        for _ in range(max_new_tokens - 1):
            # Pure decode step: append current token's KV to each layer's cache
            logits = self.model.decode_step(cur_token, slot)
            cur_token = logits.argmax(dim=-1)
            generated.append(cur_token.item())
            if cur_token.item() == self.model.eos_id:
                break

        return generated
```

## Chunked Prefill: Serving Both Phases on One Pool

Disaggregation requires separate hardware pools and a network transfer path — a significant operational complexity. **Chunked prefill** is the middle-ground technique that keeps prefill and decode on the same GPU but breaks long prefills into small chunks, interleaving them with decode steps so no single iteration monopolizes the GPU.

### The Basic Idea

Instead of processing a 16K-token prompt in one monolithic forward pass (which would stall decode for many tens of milliseconds), we split the prompt into chunks of at most $C$ tokens each, say $C = 512$. Each inference iteration processes:

- One **chunk** of the current prompt (partial prefill), contributing $C$ tokens to the KV cache.
- All **decode tokens** from in-flight sequences (one new token each).


{{fig:chunked-prefill-iterations}}


The decode sequences are never fully preempted. Their ITL (inter-token latency) increases only by the overhead of the prefill chunk, not by the full prompt length. This converts the bursty, unpredictable P99 latency problem into a smooth, bounded one.

### Choosing the Chunk Size $C$

The chunk size $C$ is a critical knob:

- **Too large:** each iteration still stalls decode sequences for too long. If $C = 4096$, the P99 ITL spikes are only 4x better than without chunking.
- **Too small:** the prefill is broken into so many chunks that the total time to complete prefill (TTFT) grows. Each chunk incurs per-iteration overhead (kernel launches, scheduling, KV-cache bookkeeping). Additionally, attention kernels are less efficient on shorter sequences — you leave FLOP/s on the table.

In practice, production systems tune $C$ per deployment based on their latency SLOs. Values in the range $C \in [256, 2048]$ are common. The scheduler can also make $C$ dynamic: use large chunks when the decode batch is empty (no one is waiting) and small chunks when many decode sequences are active.

!!! example "Worked Example: Chunked Prefill Latency"

    **Setup:** A 13B parameter model running on one A100 80GB SXM. The decode batch is $B_d = 32$ sequences. A new request arrives with a 8,192-token prompt.

    **Without chunked prefill:** The prefill runs as one monolithic forward pass. Empirically, a 13B model prefill over 8K tokens on an A100 takes roughly 800 ms (this varies with implementation; the order of magnitude is correct). During this time, all 32 decode sequences are stalled — their ITL spikes by 800 ms. If the SLO is P99 ITL ≤ 100 ms, this is an 8x violation.

    **With chunked prefill, $C = 512$:** The 8K prompt is split into 16 chunks of 512 tokens. Each chunk takes roughly $800 / 16 = 50$ ms to process (prefill scales roughly linearly in prompt length for transformer attention, $O(T_p)$ for causal self-attention over the new tokens). Each iteration, decode sequences incur ~50 ms of ITL overhead from the chunk — right at the 100 ms SLO boundary (they add their own ~10–20 ms of decode compute on top). TTFT increases from 800 ms to 16 × 50 ms + scheduling overhead ≈ 850 ms — almost unchanged.

    **Tradeoff:** Chunked prefill kept P99 ITL within SLO by paying a modest 6% TTFT penalty.

### Attention on Partial KV Caches

A subtle implementation detail: when processing chunk $k$ of a prompt, the attention layer must attend over the KV cache built from chunks $0 \ldots k-1$. This is not the same as standard decode (which attends over a complete past KV cache). It is also not the same as full prefill (which attends over the full prompt). The attention mask must reflect that:

- Tokens in chunk $k$ can attend to all previous prompt tokens (chunks $0 \ldots k-1$) that have been cached.
- Tokens in chunk $k$ can attend to earlier tokens within the same chunk (causal masking within the chunk).
- Tokens in chunk $k$ cannot attend to later prompt tokens (not yet processed).

This requires a custom attention mask and careful block layout in the KV cache. Systems like vLLM (see [vLLM: Architecture, PagedAttention & Internals](../07-inference-serving/03-vllm-internals.html)) implement chunked prefill by extending their paged KV cache manager to handle mid-prompt KV writes and mid-sequence mask construction.

```python
import torch
import torch.nn.functional as F

def chunked_prefill_attention(
    q_chunk: torch.Tensor,    # [C, n_heads, d_head] — queries for current chunk
    k_full: torch.Tensor,     # [T_past + C, n_kv_heads, d_head] — all keys so far
    v_full: torch.Tensor,     # [T_past + C, n_kv_heads, d_head]
    T_past: int,              # tokens already in KV cache (from previous chunks)
    C: int,                   # chunk size
    scale: float,
) -> torch.Tensor:
    """
    Compute attention for a chunk of prefill tokens.

    The mask allows each query at position T_past+i to attend to
    positions 0 .. T_past+i (standard causal), but NOT T_past+i+1 ..
    T_past+C-1 (future tokens in the same chunk).
    """
    n_heads, d_head = q_chunk.shape[1], q_chunk.shape[2]

    # Expand GQA: if n_kv_heads < n_heads, repeat KV heads
    # (omitted for brevity — same as decode)

    # Build causal mask for the chunk against the full context
    # Shape: [C, T_past + C]
    T_total = T_past + C
    causal_mask = torch.ones(C, T_total, dtype=torch.bool)
    for i in range(C):
        # query at position T_past + i can see positions 0 .. T_past + i
        causal_mask[i, T_past + i + 1:] = False

    # Attention scores: [n_heads, C, T_total]
    q = q_chunk.transpose(0, 1)     # [n_heads, C, d_head]
    k = k_full.transpose(0, 1)      # [n_kv_heads, T_total, d_head]
    # (assume n_heads == n_kv_heads for clarity)
    scores = torch.bmm(q, k.transpose(1, 2)) * scale   # [n_heads, C, T_total]

    # Apply causal mask (broadcast over head dim)
    scores = scores.masked_fill(
        ~causal_mask.unsqueeze(0),  # [1, C, T_total]
        float('-inf'),
    )

    attn = F.softmax(scores, dim=-1)    # [n_heads, C, T_total]
    v = v_full.transpose(0, 1)          # [n_kv_heads, T_total, d_head]
    out = torch.bmm(attn, v)            # [n_heads, C, d_head]
    return out.transpose(0, 1)          # [C, n_heads, d_head]
```

## DistServe: Formalizing Disaggregation

**DistServe** (Zhong et al., 2024) is the landmark paper that formally analyzed and implemented disaggregated prefill/decode serving. Its key contributions:

1. **Quantified the interference problem** with measurements showing that mixed batches cause P99 TTFT to grow proportionally with the longest prefill in the batch, and P99 ITL to grow as the decode batch is interrupted by prefill work.

2. **Proposed resource allocation as an optimization problem:** given a fleet of GPUs, how many should be assigned to the prefill pool vs. the decode pool? The answer depends on the workload's prompt-to-output ratio. Long-prompt workloads (RAG pipelines, document summarization) need more prefill capacity; chatbot workloads with short prompts and long outputs need more decode capacity.

3. **Implemented KV transfer via RDMA** with layer-wise pipelining to overlap transfer with computation, achieving near-zero transfer overhead on NVLink-connected nodes.

4. **Demonstrated SLO attainment:** under tight latency SLOs, disaggregated serving can serve 2–4x more requests per second than mixed serving while keeping P99 TTFT and P99 ITL within SLO bounds.

The optimization problem DistServe solves is (informally):

$$
\max_{r_P, r_D} \ \text{Throughput}(r_P, r_D) \quad \text{s.t.} \quad P99_\text{TTFT} \leq S_\text{TTFT},\ P99_\text{ITL} \leq S_\text{ITL},\ r_P + r_D = N
$$

where $r_P$ and $r_D$ are the number of replicas (GPU groups) allocated to prefill and decode, and $N$ is the total GPU budget.

## Splitwise: Heterogeneous Hardware for Each Phase

**Splitwise** (Patel et al., 2023, Microsoft Research) takes disaggregation one step further: it argues that because prefill is compute-bound and decode is memory-bandwidth-bound, you should use *different GPU models* for the two pools. Specifically:

- **Prefill workers:** use high-FLOP/s, moderately-bandwidth GPUs. In an H100/A100 world, this often means fewer GPUs with aggressive compute configurations.
- **Decode workers:** use high-bandwidth-memory GPUs — or even CPUs with large memory for small batches (CPU offloading). The H100 HBM3 memory at 3.35 TB/s shines here.

Splitwise also introduced the term **"prompt phase"** for prefill and **"token phase"** for decode, now widely adopted in the systems literature. Their key empirical finding: on commercial cloud deployments, the decode phase uses far fewer FLOPs per token than prefill but consumes a comparable fraction of total serving cost due to the time it spends waiting for memory bandwidth. Disaggregation with heterogeneous hardware can reduce per-token cost by routing each phase to its best-fit hardware.

This connects directly to inference economics (see [Inference Economics: Latency, Throughput & Cost](../07-inference-serving/12-inference-economics.html)) — the compute-to-cost frontier is different for prefill and decode, and ignoring this difference means overpaying.

## Scheduling Under Disaggregation

A disaggregated system requires a more sophisticated scheduler than continuous batching. The scheduler must:

1. **Route new requests to prefill workers** with available capacity.
2. **Match prefill worker output** to decode workers with available KV-cache pages.
3. **Handle KV transfer back-pressure:** if decode workers are full, the prefill worker must pause or queue its output — this is analogous to a producer/consumer problem.
4. **Rebalance pools dynamically** as the arrival rate shifts between short-prompt (decode-heavy) and long-prompt (prefill-heavy) workloads.

### Priority and Preemption Policies

In a pure disaggregated system, decode workers can still be preempted if they run out of KV-cache memory for long sequences. The preemption options remain the same as in standard serving: swap KV cache to CPU memory or recompute from scratch (at the cost of a prefill re-run). These policies are discussed in depth in [PagedAttention & KV-Cache Memory Management](../04-kernels-efficiency/06-paged-attention-kv.html).

Under chunked prefill (without full disaggregation), the scheduler gains a finer-grained lever: it can dynamically adjust $C$ to give more or less priority to prefill vs. decode. A simple policy:

```python
def adaptive_chunk_size(
    decode_queue_depth: int,       # number of active decode sequences
    prefill_queue_depth: int,      # number of prompts waiting to start prefill
    base_chunk_size: int = 512,
    max_chunk_size: int = 4096,
    min_chunk_size: int = 128,
    decode_pressure_threshold: int = 16,
) -> int:
    """
    Return the chunk size for the next iteration.

    When many decode sequences are active (high decode_queue_depth),
    use small chunks to minimise decode ITL inflation.
    When the prefill queue is long and decode is idle, use large chunks
    to drain prompts quickly and minimise TTFT.
    """
    if decode_queue_depth == 0:
        # No decode sequences waiting; use full large chunks for fast TTFT
        return max_chunk_size

    if decode_queue_depth >= decode_pressure_threshold:
        # Lots of decode sequences in-flight; be gentle on ITL
        return min_chunk_size

    # Interpolate linearly between min and base
    ratio = decode_queue_depth / decode_pressure_threshold
    chunk = int(base_chunk_size - ratio * (base_chunk_size - min_chunk_size))
    return max(min_chunk_size, min(chunk, base_chunk_size))
```

## Implementation in vLLM and SGLang

Both vLLM and SGLang have shipped production-quality chunked prefill implementations.

### vLLM's Chunked Prefill

vLLM introduced chunked prefill (often called **"prefill chunking"** in its documentation) as a scheduler-level feature. The key parameters:

```yaml
# vllm serve configuration (YAML or CLI flags)
enable_chunked_prefill: true
max_num_batched_tokens: 2048   # total tokens (prefill chunks + decode) per iteration
max_num_seqs: 256              # max concurrent sequences
```

Internally, the `Scheduler` class in vLLM's `vllm/core/scheduler.py` maintains a queue of "prefill sequences" with a `num_computed_tokens` counter. Each scheduling round, it allocates up to `max_num_batched_tokens - num_decode_tokens` tokens toward prefill — splitting them across waiting sequences as needed.

### SGLang's RadixAttention and Chunking

SGLang (see [SGLang: RadixAttention & Structured Programs](../07-inference-serving/04-sglang-radixattention.html)) combines chunked prefill with its RadixAttention prefix cache. A chunked prefill step can reuse prefix cache hits from earlier chunks, meaning that if two requests share a common prefix, their prefill work is shared at the chunk boundary — a multiplicative efficiency win.

### A Minimal End-to-End Chunked Prefill Scheduler

```python
import dataclasses
from collections import deque
from typing import List, Optional
import torch

@dataclasses.dataclass
class SequenceState:
    seq_id: int
    prompt_ids: List[int]                # full prompt token ids
    num_computed: int = 0                # how many prompt tokens have been processed
    kv_cache: Optional[torch.Tensor] = None  # accumulated KV cache
    output_ids: List[int] = dataclasses.field(default_factory=list)
    finished: bool = False

class ChunkedPrefillScheduler:
    """
    Minimal scheduler demonstrating chunked prefill logic.
    Does NOT implement actual model calls — shows scheduling decisions only.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        max_decode_seqs: int = 64,
        max_batched_tokens: int = 4096,
    ):
        self.chunk_size = chunk_size
        self.max_decode_seqs = max_decode_seqs
        self.max_batched_tokens = max_batched_tokens

        self.waiting: deque[SequenceState] = deque()      # not yet started
        self.prefilling: deque[SequenceState] = deque()   # partially prefilled
        self.decoding: List[SequenceState] = []           # in decode phase

    def add_request(self, seq_id: int, prompt_ids: List[int]):
        self.waiting.append(SequenceState(seq_id=seq_id, prompt_ids=prompt_ids))

    def schedule(self) -> dict:
        """
        Produce a batch descriptor for the next forward pass.
        Returns a dict describing which sequences to process and how.
        """
        budget = self.max_batched_tokens
        batch = {"decode": [], "prefill_chunks": []}

        # --- Step 1: schedule decode sequences (highest priority) ---
        for seq in self.decoding:
            if not seq.finished:
                # Each decode sequence consumes 1 token of budget
                if budget >= 1:
                    batch["decode"].append(seq.seq_id)
                    budget -= 1

        decode_budget_used = self.max_batched_tokens - budget

        # --- Step 2: schedule prefill chunks with remaining budget ---
        # First, continue any partially prefilled sequences
        next_prefilling = deque()
        for seq in self.prefilling:
            remaining = len(seq.prompt_ids) - seq.num_computed
            chunk = min(remaining, self.chunk_size, budget)
            if chunk <= 0:
                next_prefilling.append(seq)
                continue

            batch["prefill_chunks"].append({
                "seq_id": seq.seq_id,
                "token_ids": seq.prompt_ids[seq.num_computed: seq.num_computed + chunk],
                "start_pos": seq.num_computed,
            })
            seq.num_computed += chunk
            budget -= chunk

            if seq.num_computed >= len(seq.prompt_ids):
                # Prefill complete; move to decode
                self.decoding.append(seq)
            else:
                next_prefilling.append(seq)

        self.prefilling = next_prefilling

        # Then, admit new requests if budget remains and decode pool not full
        while (
            self.waiting
            and budget >= self.chunk_size
            and len(self.decoding) < self.max_decode_seqs
        ):
            seq = self.waiting.popleft()
            chunk = min(len(seq.prompt_ids), self.chunk_size, budget)
            batch["prefill_chunks"].append({
                "seq_id": seq.seq_id,
                "token_ids": seq.prompt_ids[:chunk],
                "start_pos": 0,
            })
            seq.num_computed = chunk
            budget -= chunk

            if seq.num_computed >= len(seq.prompt_ids):
                self.decoding.append(seq)
            else:
                self.prefilling.append(seq)

        return batch

    def mark_decode_finished(self, seq_id: int):
        self.decoding = [s for s in self.decoding if s.seq_id != seq_id]
```

## Comparative Analysis: Chunked vs. Disaggregated

Both chunked prefill and full disaggregation solve the interference problem, but with different tradeoffs:

| Dimension | Chunked Prefill | Full Disaggregation |
|---|---|---|
| Hardware complexity | Single pool | Two pools + network |
| KV transfer cost | None (same GPU) | Depends on interconnect |
| TTFT impact | Slight increase (more iterations) | Minimal (prefill at full speed) |
| ITL impact | Bounded by chunk size | Near-zero (no prefill on decode workers) |
| Operational complexity | Low (one scheduler) | High (routing, rebalancing, fault tolerance) |
| Cost efficiency | Moderate | High (heterogeneous hardware) |
| Best for | Moderate prompt lengths, single-cluster | Very long prompts, multi-cluster |

A key point: these techniques are not mutually exclusive. A disaggregated system can also use chunked prefill *within* each prefill worker to avoid wasting memory bandwidth on very long sequences when the prefill batch is small.

!!! interview "Interview Corner"

    **Q:** A production LLM serving system is experiencing P99 inter-token latency (ITL) violations whenever a long prompt (>4K tokens) arrives in the system, even though average latency is fine. The system uses continuous batching. What is the root cause, and what are two architectural solutions with their tradeoffs?

    **A:** The root cause is **prefill-decode interference**: when a long prompt enters continuous batching, it occupies the GPU for many milliseconds computing its KV cache. All other sequences in the decode phase are blocked — their ITL spikes by the full prefill duration. Two solutions:

    1. **Chunked prefill:** Break the incoming prompt into small chunks (e.g., 512 tokens) and interleave each chunk with a decode step. This bounds the per-iteration overhead to chunk_time, keeping P99 ITL within SLO. The tradeoff is slightly increased TTFT (more iterations to complete prefill) and minor implementation complexity in the attention kernel.

    2. **Disaggregated prefill/decode:** Move prefill to a dedicated GPU pool and decode to a separate pool, transferring KV caches over a high-speed interconnect. This eliminates interference entirely at the cost of significant infrastructure complexity (two pools, network transfer, load balancing) and potential KV transfer latency if the interconnect is slow (e.g., cross-node InfiniBand vs. NVLink).

    Choose chunked prefill for simpler deployments; disaggregation for very large clusters or workloads with extreme prompt lengths.

## Practical Deployment Guidance

### When to Enable Chunked Prefill

Enable chunked prefill whenever:

- You observe bursty P99 ITL under mixed workloads (common in production chatbots and RAG pipelines).
- Your system receives occasional long prompts mixed with ongoing conversations.
- You are using vLLM or SGLang and have not already enabled it — it is low-risk and usually improves P99 latency.

A good starting value: `max_num_batched_tokens = 2048` with `chunk_size = 512`. Profile your P99 ITL and TTFT with a production traffic replay, then tune.

### When to Build a Disaggregated System

Disaggregation is warranted when:

- Prompt lengths are consistently long (>8K tokens — document processing, code understanding, long-context RAG).
- You have strict P99 ITL SLOs (< 50 ms for real-time applications) that chunked prefill alone cannot meet.
- You operate a large enough cluster that dedicating separate GPU pools is economically justifiable.
- You can place prefill and decode workers on the same NVSwitch fabric (same node or adjacent nodes) to keep KV transfer below ~5 ms.

### Monitoring Key Metrics

```python
# Example metrics to track for a disaggregated system
# (pseudocode — plug into your Prometheus/OpenTelemetry stack)

METRICS = {
    # Latency
    "ttft_p50_ms": "Time to first token, 50th percentile",
    "ttft_p99_ms": "Time to first token, 99th percentile",
    "itl_p50_ms":  "Inter-token latency, 50th percentile",
    "itl_p99_ms":  "Inter-token latency, 99th percentile",

    # KV transfer (disaggregated only)
    "kv_transfer_latency_p99_ms": "P99 KV cache transfer time prefill->decode",
    "kv_transfer_bytes_per_sec":  "KV transfer throughput (capacity planning)",
    "kv_transfer_queue_depth":    "Number of KV caches awaiting transfer",

    # Pool utilization
    "prefill_worker_gpu_util_pct": "GPU utilization on prefill pool",
    "decode_worker_gpu_util_pct":  "GPU utilization on decode pool",
    "decode_kv_cache_fill_pct":    "Fraction of paged KV cache in use on decode workers",

    # Scheduler health
    "prefill_queue_depth":   "Requests waiting for prefill start",
    "chunked_prefill_iters": "Average iterations to complete one prefill",
}
```

## Connections to the Broader Inference Stack

Disaggregated prefill/decode does not exist in isolation. Several adjacent technologies interact with it:

**Prefix Caching** (see [Prefix Caching & KV-Cache Reuse](../07-inference-serving/07-prefix-caching.html)): If two requests share a common prefix, the prefill worker can skip recomputing that portion, and the decode worker receives a smaller KV cache. Disaggregation amplifies the value of prefix caching because the KV transfer cost is proportional to the unique (non-cached) portion of the KV cache.

**Speculative Decoding** (see [Speculative Decoding: Draft Models, Medusa, EAGLE & Lookahead](../07-inference-serving/06-speculative-decoding.html)): Speculative decoding generates draft tokens on the decode worker and verifies them in a batched forward pass. This verification pass looks like a short prefill — under disaggregation, it stays on the decode worker (it is short enough not to cause interference) rather than being sent to the prefill pool.

**Multi-GPU Inference** (see [Multi-GPU & Multi-Node Inference](../07-inference-serving/11-multi-gpu-inference.html)): Tensor parallelism and pipeline parallelism within each pool interact with KV transfer. For a tensor-parallel model (e.g., 4-way TP), each GPU holds $1/4$ of each KV head — so the KV cache transfer is split across 4 GPUs, and all four must synchronize with the corresponding 4 decode GPUs. This requires careful collective communication design.

**GPU Architecture** (see [GPU Architecture & The Memory Hierarchy](../01-foundations/08-gpu-architecture.html)): The fundamental reason prefill and decode prefer different hardware is rooted in the roofline model. The compute roof matters for prefill; the memory bandwidth wall matters for decode. Choosing hardware with the right roof height for each pool is a direct application of the roofline analysis.

!!! key "Key Takeaways"

    - Prefill is compute-bound (large matrix multiplications over long sequences); decode is memory-bandwidth-bound (vector-matrix multiplies streaming weight tensors). These phases have fundamentally different optimal hardware profiles.
    - **Prefill-decode interference** occurs in continuous batching when a long prefill stalls in-flight decode sequences, causing bursty P99 inter-token latency spikes.
    - **Chunked prefill** breaks long prompts into sub-chunks of size $C$ (typically 256–2048 tokens) interleaved with decode steps, bounding P99 ITL inflation to the chunk compute time with minimal TTFT overhead.
    - **Disaggregated prefill/decode** separates the two phases onto dedicated GPU pools, eliminating interference entirely at the cost of a KV cache transfer over the interconnect. NVLink/NVSwitch is preferred (sub-millisecond transfer); InfiniBand is viable for bulk long-context workloads.
    - KV cache transfer size scales as $O(L \times T_p \times n_\text{kv} \times d_\text{head})$; for a 70B model with a 4K-token prompt, this is on the order of 1 GB — transferable in ~1 ms on NVLink4.
    - **DistServe** (Zhong et al., 2024) formally analyzed disaggregation and showed 2–4x throughput improvement under tight SLOs. **Splitwise** (Patel et al., 2023) extended this to heterogeneous hardware, routing each phase to cost-optimal GPU types.
    - Chunked prefill and disaggregation are complementary: a disaggregated system can still chunk large prefills within the prefill pool to improve batching efficiency.
    - Chunk size $C$ should be tuned dynamically: large chunks when the decode queue is empty (to minimize TTFT), small chunks under high decode load (to protect ITL SLOs).
    - Both vLLM and SGLang support chunked prefill in production today; full disaggregation requires orchestration infrastructure but is increasingly supported via project-level extensions.

!!! sota "State of the Art & Resources (2026)"
    Disaggregated prefill/decode has moved from research into production infrastructure: every major serving framework (vLLM, SGLang, NVIDIA Dynamo) now supports separate prefill and decode pools, while chunked prefill is enabled by default in most deployments. The key open challenges are optimizing KV-cache transfer cost across cluster topologies and dynamic pool rebalancing under bursty traffic.

    **Foundational work**

    - [Zhong et al., *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving* (2024)](https://arxiv.org/abs/2401.09670) — the landmark OSDI 2024 paper that formalized P/D disaggregation, quantified interference, and showed 7.4× throughput gains under tight SLOs.
    - [Patel et al., *Splitwise: Efficient Generative LLM Inference Using Phase Splitting* (2024)](https://arxiv.org/abs/2311.18677) — ISCA 2024; introduces the "prompt phase / token phase" framing and the heterogeneous-hardware argument for routing each phase to cost-optimal GPU types.
    - [Agrawal et al., *SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills* (2023)](https://arxiv.org/abs/2308.16369) — the original chunked-prefill paper; shows decode-maximal batching with fixed-size prefill chunks eliminates decode stalls.

    **Recent advances (2023–2026)**

    - [Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve* (2024)](https://arxiv.org/abs/2403.02310) — OSDI 2024 follow-up: stall-free scheduling with chunked prefill achieves 2.6–5.6× higher serving capacity over vLLM baselines.
    - [Qin et al., *Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving* (2024)](https://arxiv.org/abs/2407.00079) — Moonshot AI's production system; disaggregates KV storage into a CPU/DRAM/SSD distributed pool, enabling 75% more requests handled under SLO in real workloads.

    **Open-source & tools**

    - [vllm-project/vllm — Disaggregated Prefilling docs](https://docs.vllm.ai/en/latest/features/disagg_prefill/) — official vLLM guide to running separate prefill and decode instances with connector-based KV transfer; covers configuration, benchmarking, and supported interconnects.
    - [microsoft/sarathi-serve](https://github.com/microsoft/sarathi-serve) — reference implementation of Sarathi-Serve (chunked prefill + stall-free scheduler); clean codebase for studying scheduling logic.
    - [kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake) — production KV-cache transfer engine (RDMA/CXL/NVMe-oF) integrated with vLLM and SGLang; useful for cross-node KV migration at scale.
    - [ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo) — NVIDIA's open-source datacenter-scale inference stack; provides independently scalable P/D pools, KV-aware routing, and multi-tier caching on top of vLLM/SGLang/TRT-LLM.

    **Go deeper**

    - [NVIDIA Developer Blog: *Introducing NVIDIA Dynamo* (2025)](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) — engineering walkthrough of disaggregated serving at datacenter scale, announced at GTC 2025.
    - [Microsoft Research: *Splitwise* publication page](https://www.microsoft.com/en-us/research/publication/splitwise-efficient-generative-llm-inference-using-phase-splitting/) — includes links to paper, slides, and the Azure LLM inference dataset used for evaluation.

## Further Reading

- **Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized Large Language Model Serving," OSDI 2024.** The foundational paper for disaggregated serving; includes the formal goodput optimization and KV transfer analysis.
- **Patel et al., "Splitwise: Efficient Generative LLM Inference Using Phase Splitting," ISCA 2024 (Microsoft Research).** Formalizes the heterogeneous hardware argument and coins "prompt phase" / "token phase" terminology.
- **Agrawal et al., "Sarathi-Serve: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills," OSDI 2024.** Demonstrates chunked prefill in a production-oriented scheduler with careful measurement of the TTFT-ITL tradeoff.
- **Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023.** The vLLM paper; PagedAttention is the KV-cache memory management substrate on which chunked prefill and disaggregation are built.
- **vLLM documentation: "Chunked Prefill" and "Disaggregated Prefill."** The vLLM project documentation covers both features with configuration examples and benchmark guidance.
- **SGLang GitHub repository (lm-sys/sglang).** SGLang's scheduler source code is an excellent reference for how chunked prefill interacts with RadixAttention prefix caching in a production system.
