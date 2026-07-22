# 4.6 PagedAttention & KV-Cache Memory Management

A trained transformer is only half the story. The other half is *serving* it: turning a static set of weights into a system that answers thousands of concurrent users with low latency and high throughput. The single largest obstacle to that goal is not compute — modern GPUs have FLOPs to spare during decoding — it is **memory**, specifically the memory consumed by the **key–value cache (KV cache)**.

This chapter is about how that memory is managed. We start from the arithmetic of how large the KV cache actually is, watch a naive serving system bleed throughput through **fragmentation**, and then develop the central idea of this chapter: borrowing **virtual memory and paging** from operating systems and applying it to attention. The result, **PagedAttention** — introduced by Kwon et al. in the vLLM paper (*Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023) — is the mechanism that turned LLM serving from "fit one request comfortably" into "pack the GPU to the rafters." It is the bridge between the kernel-level material of [FlashAttention I: IO-Awareness & The Online Softmax](../04-kernels-efficiency/02-flash-attention-1.html) and the systems-level material in [vLLM: Architecture, PagedAttention & Internals](../07-inference-serving/03-vllm-internals.html).

By the end you should be able to: compute KV-cache memory exactly; explain why contiguous allocation wastes 60–80% of it; describe block tables, copy-on-write, and the paged kernel; and reason about the throughput consequences.

## Why the KV Cache Exists, and How Big It Gets

{{tool:kv-cache-budgeter}}

Autoregressive generation produces one token at a time. To generate token $t$, a decoder-only transformer must attend over all previous tokens $1 \dots t-1$. Attention needs the **keys** and **values** of those tokens at every layer. If we recomputed them from scratch each step, generating a sequence of length $n$ would cost $O(n^2)$ work *per new token* — quadratically wasteful.

The KV cache is the standard fix: after a token is processed, we store its per-layer key and value vectors and reuse them forever. Generation then splits into two phases (covered in depth in [The Anatomy of LLM Inference: Prefill, Decode & The KV Cache](../07-inference-serving/01-anatomy-inference.html)):

- **Prefill**: process the whole prompt in one batched forward pass, producing K/V for every prompt token. Compute-bound.
- **Decode**: generate one token at a time, each step appending exactly one K and one V vector per layer to the cache. Memory-bandwidth-bound.

### The size formula

Let:

- $L$ = number of transformer layers
- $H_{kv}$ = number of **key/value** heads (with [GQA/MQA](../02-transformer/04-mha-gqa-mla.html) this is smaller than the number of query heads)
- $d_h$ = dimension per head
- $s$ = sequence length (prompt + generated tokens)
- $b$ = bytes per element (2 for fp16/bf16, 1 for fp8)

For a single sequence, the KV cache stores both K and V, across all layers, for all heads and positions:

$$
\text{Bytes}_{\text{KV}} = 2 \cdot L \cdot H_{kv} \cdot d_h \cdot s \cdot b
$$

The leading $2$ is "K and V." Note what is *not* here: the batch dimension (that just multiplies through) and the number of *query* heads (the cache scales with KV heads only — this is the entire point of GQA/MQA for serving).

Define the **per-token KV footprint** $\beta = 2 \cdot L \cdot H_{kv} \cdot d_h \cdot b$ bytes/token. Then the cache for a sequence of length $s$ is simply $\beta s$, and for a batch of $B$ sequences it is $\beta \sum_{i} s_i$. This linear-in-tokens structure is what makes paging natural: tokens are the unit of allocation.

!!! example "Worked example: KV cache for Llama-2-13B and a 70B GQA model"

    **Llama-2-13B** (multi-head, no GQA): $L=40$, $H_{kv}=40$ heads, $d_h=128$, fp16 ($b=2$).

    $$
    \beta = 2 \cdot 40 \cdot 40 \cdot 128 \cdot 2 = 819{,}200 \text{ bytes/token} \approx 0.78\ \text{MiB/token}.
    $$

    A single 2048-token sequence costs $0.78 \times 2048 \approx 1.6$ GiB. On an 80 GiB A100 holding the ~26 GiB of fp16 weights, you have ~54 GiB for cache — about **34** such sequences. The model has tens of thousands of FLOPs of headroom per step, but you run out of *memory* at a few dozen sequences. That ceiling is the throughput ceiling.

    **Llama-2-70B with GQA** ($H_{kv}=8$): $L=80$, $d_h=128$, fp16.

    $$
    \beta = 2 \cdot 80 \cdot 8 \cdot 128 \cdot 2 = 327{,}680 \text{ bytes/token} \approx 0.3125\ \text{MiB/token}.
    $$

    Despite being a far larger model, GQA shrinks per-token KV by collapsing 64 query heads onto 8 KV heads. A 4096-token sequence is only $0.3125 \times 4096 = 1.25$ GiB. GQA is a *serving* decision as much as a quality one.

A useful way to internalize this: the KV cache for a long conversation can rival or exceed the *weights themselves*. A 4M-token context at $\beta \approx 0.3$ MiB/token would need ~1.2 TiB — impossible on one GPU. KV memory, not parameters, is the binding constraint on context length and concurrency.

```python
def kv_cache_bytes(num_layers, num_kv_heads, head_dim, seq_len,
                   bytes_per_elem=2, batch=1):
    """Exact KV-cache size in bytes. The leading 2 covers K and V."""
    per_token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_elem
    return per_token * seq_len * batch

# Llama-2-13B, one 2048-token request
b = kv_cache_bytes(40, 40, 128, 2048)
print(f"{b/2**30:.2f} GiB")           # ~1.56 GiB

# Per-token footprint (the constant beta)
beta = kv_cache_bytes(40, 40, 128, 1)
print(f"{beta/2**20:.3f} MiB/token")  # ~0.781 MiB/token
```

{{fig:paged-kv-cache-vs-weights-growth}}

## The Fragmentation Problem

The naive serving system allocates one **contiguous** chunk of GPU memory per request, sized for the maximum sequence length the request could reach. This is how early systems (and a straightforward HuggingFace `generate` loop) work. It is also catastrophically wasteful, in three distinct ways. The vLLM authors named them precisely.

### Internal fragmentation: reserving for the worst case

You do not know in advance how long a generation will be. So you reserve `max_seq_len` slots up front. If a request finishes after 30 tokens but you reserved 2048, the other 2018 slots are allocated, untouched, and unavailable to anyone else for the request's lifetime. That is **internal fragmentation** — wasted space *inside* an allocation.

{{fig:paged-kv-internal-frag-bar}}

In real workloads outputs are short and highly variable, so internal fragmentation routinely consumes the majority of reserved cache.

### External fragmentation: holes between allocations

Different requests reserve different-sized contiguous blocks. As requests of varying `max_len` arrive and complete, the free memory shatters into a patchwork of gaps. A new request needing a contiguous 1.5 GiB block may be rejected even though 4 GiB is free in total — because no single *contiguous* hole is large enough. This is **external fragmentation**, the classic curse of any contiguous allocator (think `malloc` without compaction).

### Reservation waste and the inability to share

Because each request owns a private contiguous region, two requests with an *identical prompt prefix* — a shared system prompt, a few-shot template, a forked beam — each store their own complete copy of that prefix's KV. There is no mechanism to share it. For agentic and batch workloads with long fixed preambles, this duplicates gigabytes.

The vLLM paper measured that under contiguous allocation, only **20–40%** of KV memory held actual token state; the rest was lost to these three effects. Recovering that memory is, to first order, a $2\text{--}4\times$ throughput win — because throughput in the memory-bound decode regime is set by *how many sequences you can hold concurrently*.

!!! warning "Throughput is gated by batch size, which is gated by memory"

    During decode, each step reads the *entire* KV cache and all weights from HBM but does very little arithmetic per token — it is bandwidth-bound (see [The Roofline Model & Performance Engineering](../04-kernels-efficiency/01-roofline-performance.html)). The fix for bandwidth-bound work is **bigger batches**: process more sequences per memory sweep so the fixed weight-read cost is amortized. But batch size is capped by how many KV caches fit in memory. Therefore *wasted KV memory directly throttles throughput.* This causal chain — fragmentation → small batch → low throughput — is the whole motivation for paging.

## The OS-Paging Analogy

Operating systems solved exactly this problem decades ago. A process believes it owns a single contiguous **virtual address space**, but physically its memory is scattered across fixed-size **page frames** in RAM, with a **page table** mapping virtual page numbers to physical frame numbers. Benefits: no external fragmentation (any free frame works for any page), internal fragmentation bounded by at most one page per allocation, and trivial sharing (two page tables can point at the same physical frame).

PagedAttention transplants this idea wholesale:

| Operating system | PagedAttention |
|---|---|
| Process | Request / sequence |
| Virtual address space | The logical token sequence (positions $0,1,2,\dots$) |
| Page | **KV block** (fixed number of tokens, e.g. 16) |
| Physical page frame | A physical KV block in GPU HBM |
| Page table | **Block table** (per-sequence: logical block → physical block) |
| Page fault → allocate frame | Sequence grows → allocate a new physical block |
| Shared read-only pages | Shared prompt prefix blocks |
| Copy-on-write | Copy-on-write when a shared block must diverge |

The key reframing: the KV cache of a sequence need **not be contiguous in physical memory**. We chop it into fixed-size blocks of $B$ tokens (typical $B = 16$), store the blocks anywhere in a global pool, and keep a small per-sequence **block table** recording where each logical block physically lives. Logical contiguity (token positions in order) is decoupled from physical contiguity (where bytes sit in HBM).

{{fig:paged-kv}}


{{fig:paged-kv-logical-physical-mapping}}

Sequence A's three logical blocks live in physical blocks 7, 1, 4 — scattered, out of order, interleaved with B's blocks. Attention does not care, because the block table tells the kernel exactly where to gather from.

## Block Tables, the Allocator & Copy-on-Write

Let us build the bookkeeping concretely. The system maintains a global pool of physical KV blocks and a free list. Allocation is per-block, on demand, as sequences grow.

```python
from collections import defaultdict

class BlockManager:
    """Minimal model of vLLM's paged KV allocator.

    The physical KV tensors themselves live in two big preallocated
    GPU pools (one for K, one for V) shaped roughly
        [num_blocks, block_size, num_kv_heads, head_dim].
    Here we only track *which* physical block each logical block uses,
    plus reference counts to enable copy-on-write sharing.
    """
    def __init__(self, num_blocks, block_size):
        self.block_size = block_size
        self.free = list(range(num_blocks))        # free physical block ids
        self.ref_count = defaultdict(int)          # phys_block -> #sequences using it
        self.block_tables = {}                     # seq_id -> [phys_block, ...]

    def _alloc_block(self):
        if not self.free:
            raise MemoryError("KV pool exhausted — must preempt a sequence")
        blk = self.free.pop()
        self.ref_count[blk] = 1
        return blk

    def allocate(self, seq_id, num_tokens):
        """Allocate enough blocks to hold `num_tokens` (e.g. after prefill)."""
        n_blocks = (num_tokens + self.block_size - 1) // self.block_size
        self.block_tables[seq_id] = [self._alloc_block() for _ in range(n_blocks)]

    def append_token(self, seq_id, cur_len):
        """Called once per decode step. Allocate a new block only when the
        current last block is exactly full — this is the ONLY moment a paged
        system touches the allocator during decode."""
        table = self.block_tables[seq_id]
        if cur_len % self.block_size == 0:         # boundary crossed → new page
            table.append(self._alloc_block())

    def slot_index(self, seq_id, token_pos):
        """Translate a logical token position to a flat physical slot id,
        exactly what the kernel needs to write/read the K,V for that token."""
        table = self.block_tables[seq_id]
        phys_block = table[token_pos // self.block_size]
        offset = token_pos % self.block_size
        return phys_block * self.block_size + offset

    def free_seq(self, seq_id):
        for blk in self.block_tables.pop(seq_id):
            self.ref_count[blk] -= 1
            if self.ref_count[blk] == 0:           # last user → reclaim
                self.free.append(blk)
```

Three properties fall out immediately:

1. **No external fragmentation.** Every physical block is identical in size, so any free block satisfies any request. The free list never gets "stuck."
2. **Bounded internal fragmentation.** A sequence wastes at most $B-1$ token-slots in its final, partially filled block. With $B=16$ and sequences of hundreds of tokens, that is well under a few percent — versus the 60–80% of contiguous allocation.
3. **On-demand growth.** Blocks are allocated only as the sequence actually reaches them. A request that stops after 30 tokens used 2 blocks, not 128.

### Choosing the block size $B$

$B$ trades two things off. **Small $B$** (e.g. 1) minimizes internal fragmentation but bloats the block table, adds per-block gather overhead in the kernel, and increases pointer-chasing. **Large $B$** (e.g. 128) amortizes kernel overhead and shrinks the table but reintroduces internal fragmentation (up to $B-1$ wasted slots) and coarsens sharing granularity. vLLM defaults to $B=16$ as a sweet spot for typical head dims; it is a tunable knob, not a law.

### Copy-on-write for shared prefixes

Now the payoff that contiguous allocation simply cannot offer. Suppose many requests share a long system prompt, or you run **parallel sampling** (one prompt, $k$ output samples) or **beam search**. The shared prefix's KV is *identical* across all of them and is *read-only* once computed. So store it once and have every sequence's block table point at the same physical blocks, bumping the reference count.

```python
    def fork(self, parent_id, child_id):
        """Share ALL of parent's blocks with a new child (e.g. a new beam
        or sample). O(#blocks) pointer copies, zero KV data copied."""
        parent = self.block_tables[parent_id]
        self.block_tables[child_id] = list(parent)   # copy the table, not the KV
        for blk in parent:
            self.ref_count[blk] += 1                  # now shared
```

Sharing works perfectly until a shared sequence needs to **write** into a block that someone else also references — i.e. two samples diverge and the next token must be appended to a block whose `ref_count > 1`. We cannot mutate a block another sequence depends on. The classic OS answer applies: **copy-on-write (COW)**. Copy the contested block to a fresh physical block, redirect *this* sequence's table entry, decrement the original's ref count, then write.

```python
    def cow_append(self, seq_id, logical_block_idx):
        """Ensure seq_id privately owns the given logical block before writing.
        Returns (src_phys, dst_phys): if they differ, the kernel must copy
        block contents src→dst before the new token's K,V is written."""
        table = self.block_tables[seq_id]
        phys = table[logical_block_idx]
        if self.ref_count[phys] == 1:
            return phys, phys                         # privately owned: write in place
        # shared → copy on write
        new_phys = self._alloc_block()                # ref_count[new_phys] = 1
        self.ref_count[phys] -= 1                      # we leave the shared block
        table[logical_block_idx] = new_phys
        return phys, new_phys                          # kernel copies src→dst, then writes
```

Crucially, COW happens **per block, not per sequence**. Two diverging beam-search candidates that shared 200 prefix blocks copy only the *one* block where they first differ; the other 199 stay shared. For wide beams or large $k$-sampling over long prompts, this is the difference between $O(k \cdot s)$ and $O(s + k \cdot B)$ memory. This same machinery generalizes to cross-request **prefix caching**, the subject of [Prefix Caching & KV-Cache Reuse](../07-inference-serving/07-prefix-caching.html); SGLang pushes it further with a radix tree in [SGLang: RadixAttention & Structured Programs](../07-inference-serving/04-sglang-radixattention.html).

{{fig:paged-kv-copy-on-write}}

!!! example "Worked example: parallel sampling savings"

    Prompt = 1000 tokens, generate $k=8$ samples of 200 tokens each, Llama-2-70B GQA ($\beta \approx 0.3125$ MiB/token), block size 16.

    **Contiguous (no sharing):** each sample stores prompt + output = 1200 tokens.
    Total $= 8 \times 1200 \times 0.3125 = 3000$ MiB $\approx 2.93$ GiB.

    **Paged + COW:** the 1000-token prompt is stored *once* ($\lceil 1000/16\rceil = 63$ blocks), shared by all 8 samples. Each sample privately stores its 200 output tokens ($\lceil 200/16\rceil = 13$ blocks). Total blocks $= 63 + 8\times 13 = 167$ blocks $\times 16 \times 0.3125 = 835$ MiB $\approx 0.82$ GiB.

    A **3.6× reduction**, from sharing the prompt KV. That freed memory becomes more concurrent requests — i.e. more throughput.

## The PagedAttention Kernel

The bookkeeping is only useful if the attention kernel can *read* a logically-contiguous sequence whose K/V physically lives in scattered blocks. A standard attention kernel — including [FlashAttention](../04-kernels-efficiency/02-flash-attention-1.html) — assumes K and V are contiguous tensors it can stride through linearly. PagedAttention modifies the kernel to consult the **block table** and gather block-by-block.

The decode-time attention for one query token against a length-$s$ cache is, conceptually:

$$
\mathbf{o} = \sum_{j=1}^{s} \frac{\exp\!\big(\mathbf{q}^\top \mathbf{k}_j / \sqrt{d_h}\big)}{\sum_{j'} \exp\!\big(\mathbf{q}^\top \mathbf{k}_{j'}/\sqrt{d_h}\big)}\, \mathbf{v}_j
$$

The kernel computes this in a numerically stable **online softmax** (running max + running denominator, exactly the FlashAttention trick from [FlashAttention I](../04-kernels-efficiency/02-flash-attention-1.html)), but instead of one contiguous K/V tensor, it iterates over the sequence's *physical* blocks. For each logical block it reads the physical block id from the block table, loads that block's $B$ keys and values from the global pool, and accumulates. The block table is the indirection layer; the math is unchanged.

Here is a faithful, from-scratch reference implementation in PyTorch. It is *slow* (a real kernel is a fused CUDA/Triton kernel — see [Writing GPU Kernels with Triton](../04-kernels-efficiency/04-triton-kernels.html) and [CUDA Programming Essentials for ML Engineers](../04-kernels-efficiency/05-cuda-essentials.html)) but it is exact and shows precisely where the block table enters.

```python
import torch
import math

def paged_attention_decode(query,            # [num_heads, head_dim]  (one new token)
                           k_pool, v_pool,    # [num_blocks, block_size, num_kv_heads, head_dim]
                           block_table,       # list[int]: logical -> physical block id
                           context_len,       # number of valid cached tokens
                           block_size,
                           num_queries_per_kv):  # GQA group size (query heads per kv head)
    """Single-query paged attention with online (streaming) softmax.

    Demonstrates the ONE structural difference from a normal attention kernel:
    K and V are fetched block-by-block via `block_table` instead of being
    read from a single contiguous tensor. Everything else — the scaled
    dot-product, the numerically-stable streaming softmax — is standard.
    """
    num_heads, head_dim = query.shape
    scale = 1.0 / math.sqrt(head_dim)

    out  = torch.zeros_like(query)                          # [H, d]
    m    = torch.full((num_heads,), float("-inf"))          # running max of logits
    l    = torch.zeros(num_heads)                           # running softmax denominator

    num_logical_blocks = (context_len + block_size - 1) // block_size
    for lb in range(num_logical_blocks):
        phys = block_table[lb]                              # <-- INDIRECTION
        # how many tokens of this block are valid (last block may be partial)
        start = lb * block_size
        valid = min(block_size, context_len - start)

        # gather this physical block's K,V; map GQA query head -> its kv head
        for t in range(valid):
            for h in range(num_heads):
                kv_h = h // num_queries_per_kv              # GQA head mapping
                k = k_pool[phys, t, kv_h]                   # [d]
                v = v_pool[phys, t, kv_h]                   # [d]
                logit = scale * torch.dot(query[h], k)      # scalar

                # --- online softmax update (FlashAttention-style) ---
                m_new = torch.maximum(m[h], logit)
                # rescale the existing accumulator to the new max
                alpha = torch.exp(m[h] - m_new)
                p     = torch.exp(logit - m_new)
                l[h]  = l[h] * alpha + p
                out[h] = out[h] * alpha + p * v
                m[h]  = m_new

    return out / l.unsqueeze(-1)                            # normalize by denominator
```

The inner triple loop is purely illustrative; a production kernel assigns one thread block (CUDA cooperative thread array) per (sequence, KV-head), loads each physical KV block into shared memory / registers, does the dot products with vectorized loads, and reduces with warp shuffles. But the *only* algorithmic novelty over FlashAttention is the line `phys = block_table[lb]`: a gather through the page table before each block load.

### Performance considerations

The indirection is not free, but it is cheap:

- **Extra memory traffic** is one small block-table read per block — negligible against loading the block's $B \times H_{kv} \times d_h$ KV elements.
- **Non-contiguous reads** are the real cost: scattered physical blocks defeat large coalesced loads and prefetchers. PagedAttention mitigates this by keeping a *whole block* contiguous (so within a block, loads are coalesced) and by choosing $B$ large enough (16+) to amortize the per-block setup.
- The vLLM authors report the paged kernel runs within a small percentage of a perfectly-contiguous FlashAttention kernel — a tiny per-step tax that is *overwhelmingly* repaid by the larger batch sizes the freed memory enables.

A further refinement, **PagedAttention v2** (in vLLM) and related work, splits very long sequences across multiple thread blocks (a "split-K"-style reduction over the sequence dimension) so a single long request does not serialize on one streaming multiprocessor — important when concurrency is low but contexts are long.

!!! tip "Practitioner tip: gpu_memory_utilization and preemption"

    In vLLM, `gpu_memory_utilization` (default ~0.9) sets the fraction of HBM carved out for the *combined* weights + KV pool. The number of physical KV blocks is computed from whatever is left after weights and activation scratch. Set it too low and you starve the batch (low throughput); too high and you risk OOM from activation spikes during prefill. When the pool is exhausted mid-decode, vLLM **preempts** a running sequence — either *recomputing* its KV later (cheap if short) or *swapping* its blocks to CPU RAM (cheap if long) — and resumes it when blocks free up. Preemption is the paged analogue of OS swapping; watch for it in logs as a signal you are memory-bound and should lower concurrency or shorten `max_model_len`.

## How Paging Unlocks High-Throughput Serving

Paging is not just a memory optimization in isolation; it is the substrate that makes the rest of modern serving work.

**It enables large, dynamic batches.** Because each sequence grows block-by-block and reclaims blocks on completion, the system can pack many sequences into the freed space. This is the storage layer beneath **continuous batching** (also called in-flight batching), where finished sequences are evicted from the running batch and new ones admitted every step rather than waiting for the whole batch to finish — see [Continuous Batching & Request Scheduling](../07-inference-serving/02-continuous-batching.html). Continuous batching needs to add and remove sequences at *token* granularity; only a block allocator can give back a finished sequence's memory cheaply enough for that to pay off.

**It enables prefix sharing at scale.** COW blocks let a fixed system prompt, a RAG context, or a few-shot template be stored once and reused across thousands of requests, slashing both memory and prefill compute (the shared prefix need not be recomputed). This is the foundation of [Prefix Caching & KV-Cache Reuse](../07-inference-serving/07-prefix-caching.html).

**It raises effective throughput several-fold.** The vLLM paper reports $2\text{--}4\times$ higher throughput than prior contiguous-allocation systems at the same latency, precisely because the recovered memory becomes more concurrent sequences in the memory-bound decode regime.

{{fig:paged-kv-hbm-contiguous-vs-paged}}

The lesson generalizes well beyond vLLM: **TensorRT-LLM**, **TGI**, and **SGLang** all adopted paged KV management (see [TensorRT-LLM, TGI & Other Serving Stacks](../07-inference-serving/05-trtllm-tgi-stacks.html)). It composes with [quantizing the KV cache](../04-kernels-efficiency/08-quantization-formats-qat.html) to fp8/int8 (halving $b$ in the size formula, doubling block capacity), with [chunked prefill](../07-inference-serving/08-disaggregated-chunked-prefill.html), and with [multi-GPU inference](../07-inference-serving/11-multi-gpu-inference.html) where each tensor-parallel rank pages its shard of the KV cache independently.

!!! interview "Interview Corner"

    **Q:** A teammate says "FlashAttention already made attention memory-efficient, so why does vLLM need PagedAttention?" How do you respond?

    **A:** They solve different problems at different layers. FlashAttention is a *compute-kernel* optimization: it avoids materializing the $s \times s$ attention score matrix in HBM by tiling and using an online softmax, reducing the memory *of a single attention computation* from $O(s^2)$ to $O(s)$ and cutting HBM traffic. PagedAttention is a *memory-allocator* optimization for the *KV cache that persists across decode steps*. FlashAttention does nothing about the fact that a naive server reserves a contiguous `max_len` slab per request and so wastes 60–80% of KV memory to internal/external fragmentation and cannot share prefixes. PagedAttention stores the KV cache in fixed-size, non-contiguous blocks with a per-sequence block table — eliminating external fragmentation, bounding internal fragmentation to under one block, and enabling copy-on-write prefix sharing. The two are complementary: vLLM's kernel uses FlashAttention-style online softmax *and* paged block gathering. The win is throughput: recovered KV memory means larger decode batches, which is what matters because decode is memory-bandwidth-bound.

!!! note "Aside: why blocks and not a slab-allocated free list of arbitrary sizes?"

    One could imagine a general `malloc`-style allocator with arbitrary-size KV regions and compaction. Fixed-size blocks are deliberately simpler: uniform size makes allocation $O(1)$ (pop the free list), makes any free block fungible (no best-fit search, no compaction passes), makes the kernel's address arithmetic trivial (`block * B + offset`), and makes reference counting / COW clean at a single granularity. It is the same reason OSes use fixed page frames rather than variable-size segments. Uniformity buys simplicity and predictable performance.

## Putting It Together: A Minimal Paged Serving Loop

To cement the mechanics, here is a skeletal decode loop that ties the allocator, the slot mapping, and the kernel together for a small batch — the shape of what a paged engine does every step.

```python
import torch

def serve_step(seqs, block_mgr, k_pool, v_pool, model, block_size):
    """One decode step over a dynamic batch of sequences.

    `seqs`: dict seq_id -> {"tokens": [...], "len": int}
    Each step: (1) run the model to get this token's q,k,v per layer,
    (2) grow the block table if this position starts a new block, then
    write k,v into the paged pool at the right slot,
    (3) attend over the sequence's blocks, (4) sample next token.
    Sequences that emit EOS are freed, returning their blocks to the pool.
    """
    finished = []
    for seq_id, s in seqs.items():
        cur_len = s["len"]

        # --- 1: compute this token's K,V (q/k/v come from the model forward; mocked)
        q, k, v = model.qkv(s["tokens"][-1])            # per-layer tensors elided

        # --- 2: ensure a physical block exists for the slot we are about to
        # write, THEN write. The token at position cur_len starts a fresh block
        # exactly when cur_len % block_size == 0; growing the table HERE (rather
        # than after the write) also covers the prefill boundary -- e.g. a prompt
        # whose length is an exact multiple of block_size, where allocate() left
        # the table one block short. Skip this and slot_index indexes
        # table[cur_len // block_size] one past the last block -> IndexError.
        block_mgr.append_token(seq_id, cur_len)         # grow on block boundary
        slot = block_mgr.slot_index(seq_id, cur_len)    # logical pos -> physical slot
        phys_block, off = divmod(slot, block_size)
        k_pool[phys_block, off] = k
        v_pool[phys_block, off] = v

        # --- 3: paged attention over all cached tokens (uses the block table)
        table = block_mgr.block_tables[seq_id]
        attn_out = paged_attention_decode(
            q, k_pool, v_pool, table, cur_len + 1,
            block_size, num_queries_per_kv=model.gqa_group)

        # --- 4: project + sample the next token
        next_tok = model.sample(attn_out)
        s["tokens"].append(next_tok)
        s["len"] += 1

        if next_tok == model.eos_id:
            finished.append(seq_id)

    # reclaim memory immediately so admitted requests can reuse it THIS step
    for seq_id in finished:
        block_mgr.free_seq(seq_id)                       # blocks return to free list
        del seqs[seq_id]

    return finished
```

Notice the rhythm: the allocator is touched at most once per sequence per step (and usually *not at all* — only on a $B$-token boundary), the kernel always reads through the block table, and freed sequences return blocks to the pool instantly. That instant reclamation is what lets a scheduler admit a fresh request into the gap the same step a sequence finishes — the union of paging and continuous batching that defines modern high-throughput serving.

**Verify the block-boundary edge case.** The subtlety worth testing explicitly is the prefill->decode boundary when the prompt length is an *exact multiple* of the block size. There `allocate()` fills the last block precisely, so the very first decode write lands at position `prompt_len`, which begins a brand-new logical block. Growing the table *before* `slot_index` is what keeps that write in range:

```python
def test_block_boundary():
    B = 16
    mgr = BlockManager(num_blocks=8, block_size=B)
    prompt_len = 32                         # EXACT multiple of B -> the tricky case
    mgr.allocate("s", prompt_len)           # reserves ceil(32/16) = 2 blocks (pos 0..31)
    assert len(mgr.block_tables["s"]) == 2

    # emulate serve_step's write ordering for three decode steps (pos 32, 33, 34)
    for cur_len in range(prompt_len, prompt_len + 3):
        mgr.append_token("s", cur_len)      # grow BEFORE the write (the fix)
        slot = mgr.slot_index("s", cur_len) # would raise IndexError without the grow
        phys, off = divmod(slot, B)
        assert off == cur_len % B           # offset within the physical block

    # writing positions 32..34 needed exactly one new block beyond prefill's two
    assert len(mgr.block_tables["s"]) == 3
    print("boundary case OK")

test_block_boundary()                       # -> boundary case OK
```

With the old ordering (grow *after* the write, keyed on the post-increment length), `slot_index("s", 32)` evaluates `table[32 // 16] == table[2]` on a two-element table and raises `IndexError` on the very first decode step -- the bug this test pins down.

!!! key "Key Takeaways"

    - The **KV cache** stores per-layer keys and values to avoid $O(n^2)$ recomputation; its size is $2 \cdot L \cdot H_{kv} \cdot d_h \cdot s \cdot b$ bytes — linear in tokens, and it often rivals the model weights. KV memory, not FLOPs, is the binding constraint on concurrency and context length during decode.
    - **GQA/MQA** shrink the KV cache by reducing $H_{kv}$; this is a serving decision as much as a modeling one.
    - Naive **contiguous, max-length** allocation wastes 60–80% of KV memory through internal fragmentation (worst-case reservation), external fragmentation (holes between allocations), and the inability to share identical prefixes.
    - **PagedAttention** applies OS virtual-memory paging: split the KV cache into fixed-size **blocks** ($B \approx 16$ tokens), store them anywhere in a pool, and map logical → physical via a per-sequence **block table**. This eliminates external fragmentation and bounds internal fragmentation to under one block.
    - **Copy-on-write** block sharing lets requests share identical prompt prefixes (system prompts, few-shot, beams, parallel samples), copying only the one block where they diverge — large memory and prefill-compute savings.
    - The **paged kernel** is FlashAttention-style online softmax plus one indirection: read the physical block id from the block table, gather that block's K/V, accumulate. The tax is a small per-block gather; the payoff is far larger batches.
    - Paging is the substrate for **continuous batching** and **prefix caching**, delivering roughly $2\text{--}4\times$ throughput in the memory-bound decode regime. When the pool is exhausted, the engine **preempts** sequences (recompute or swap), the paged analogue of OS swapping.

!!! sota "State of the Art & Resources (2026)"
    PagedAttention (2023) is now the universal baseline for production LLM serving: virtually every major framework — vLLM, SGLang, TensorRT-LLM, TGI — adopted paged KV management, and the frontier of active research has moved to KV-cache quantization (fp8/int4), prefix-caching at radix-tree scale, and alternatives like virtual-memory-backed contiguous KV that avoid the scattered-block overhead.

    **Foundational work**

    - [Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (2023)](https://arxiv.org/abs/2309.06180) — the SOSP 2023 paper that introduced block tables, copy-on-write prefix sharing, and the 2–4× throughput wins; read it alongside the vLLM source.
    - [Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (2023)](https://arxiv.org/abs/2305.13245) — explains why reducing KV heads with GQA/MQA halves or quarters the per-token KV footprint, the key lever on the size formula.

    **Recent advances (2023–2026)**

    - [Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs* (2024)](https://arxiv.org/abs/2312.07104) — introduces RadixAttention, a radix-tree KV-cache index that generalises COW prefix sharing to arbitrary program structures, achieving up to 5× inference speedup.
    - [Prabhu et al., *vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention* (2024)](https://arxiv.org/abs/2405.04437) — leverages OS demand-paging to keep KV cache virtually contiguous, avoiding custom paged kernels while matching or beating PagedAttention throughput; accepted ASPLOS 2025.
    - [Hooper et al., *KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization* (2024)](https://arxiv.org/abs/2401.18079) — per-channel + non-uniform 4-bit/3-bit KV quantization enabling 10M-token contexts on a single A100; NeurIPS 2024.
    - [Ye et al., *FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving* (2025)](https://arxiv.org/abs/2501.01005) — composable attention kernels (paged, ragged, sparse) adopted by vLLM, SGLang, and TRT-LLM; MLSys 2025.

    **Open-source & tools**

    - [vllm-project/vllm](https://github.com/vllm-project/vllm) — the canonical production implementation of PagedAttention; the scheduler, block manager, and paged-attention CUDA kernels are the reference for everything in this chapter.
    - [sgl-project/sglang](https://github.com/sgl-project/sglang) — high-performance serving framework with RadixAttention prefix caching, powering 400k+ GPUs in production.
    - [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) — modular attention kernel library (paged, ragged, FP8) used as the attention backend inside vLLM and SGLang.

    **Go deeper**

    - [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://vllm.ai/blog/2023-06-20-vllm) — the original Berkeley blog post explaining the motivation and benchmarks; accessible first read before tackling the paper.

## Further reading

- Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang, Stoica — *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM), SOSP 2023. The paper that introduced PagedAttention; read it alongside the vLLM source.
- Dao, Fu, Ermon, Rudra, Ré — *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, 2022; and Dao — *FlashAttention-2*. The online-softmax, IO-aware kernel that PagedAttention's gather sits on top of.
- Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai — *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, 2023. Why $H_{kv}$ shrinks.
- Shoeybi, Patwary, Puri, et al. — *Megatron-LM* — for the tensor-parallel context in which each rank pages its KV shard.
- The **vLLM** and **SGLang** repositories — the canonical production implementations of paged KV management and (in SGLang) radix-tree prefix sharing.
- Operating-systems texts (e.g. Silberschatz, *Operating System Concepts*) on paging, page tables, and copy-on-write — the prior art PagedAttention borrows from.

## Exercises

**1.** *(Conceptual.)* The size formula $\text{Bytes}_{\text{KV}} = 2 \cdot L \cdot H_{kv} \cdot d_h \cdot s \cdot b$ contains the number of **key/value** heads $H_{kv}$ but *not* the number of query heads. Explain why the KV-cache size is independent of the query-head count, and why this makes GQA/MQA a "serving decision as much as a quality one."

??? note "Solution"

    The KV cache stores only what must be *reused* across decode steps: the keys and values of past tokens. Queries are never cached — at each decode step there is exactly one new query (the current token), it attends over the cached K/V, and then it is discarded; it is never needed again. So only K and V occupy the cache, and their count is set by the number of KV heads $H_{kv}$, not the number of query heads.

    GQA/MQA exploit exactly this asymmetry. They keep the full set of *query* heads (which preserves most of the model's representational capacity, hence quality) but let several query heads *share* one KV head, shrinking $H_{kv}$. Because $H_{kv}$ appears linearly in $\beta = 2 \cdot L \cdot H_{kv} \cdot d_h \cdot b$, cutting KV heads from, say, 64 to 8 shrinks the per-token KV footprint 8x with no change to the query side. As the chapter's worked example shows, this is what lets a 70B GQA model have a *smaller* per-token KV footprint (0.3125 MiB/token) than a 13B multi-head model (0.78 MiB/token). Since KV memory is the binding constraint on batch size — and batch size sets decode throughput — reducing $H_{kv}$ directly buys concurrency. Hence it is chosen for serving throughput as much as for model quality.

**2.** *(Quantitative.)* Consider a model with $L = 32$ layers, $H_{kv} = 8$ KV heads, $d_h = 128$, served in fp16 ($b = 2$). (a) Compute the per-token KV footprint $\beta$ in bytes and in MiB. (b) A single request has a 4000-token prompt and generates 2000 tokens (final length $s = 6000$). What is its KV cache in GiB? (c) If a GPU reserves 40 GiB of HBM for the KV pool, how many such 6000-token sequences can be held concurrently?

??? note "Solution"

    **(a)** Per-token footprint:

    $$
    \beta = 2 \cdot L \cdot H_{kv} \cdot d_h \cdot b = 2 \cdot 32 \cdot 8 \cdot 128 \cdot 2 = 131{,}072 \text{ bytes/token}.
    $$

    In MiB: $131072 / 2^{20} = 0.125$ MiB/token (exactly $1/8$ MiB).

    **(b)** For $s = 6000$:

    $$
    \text{Bytes}_{\text{KV}} = \beta \cdot s = 131072 \cdot 6000 = 786{,}432{,}000 \text{ bytes}.
    $$

    In GiB: $786{,}432{,}000 / 2^{30} \approx 0.732$ GiB per sequence.

    **(c)** Number of concurrent sequences:

    $$
    \left\lfloor \frac{40 \text{ GiB}}{0.732 \text{ GiB}} \right\rfloor = \left\lfloor 54.6 \right\rfloor = 54 \text{ sequences}.
    $$

    (Equivalently: $40 \cdot 2^{30} / 786{,}432{,}000 \approx 54.6$, so 54 fit.) This concurrency ceiling — not FLOPs — is what caps decode throughput.

**3.** *(Quantitative.)* With block size $B = 16$, contrast the internal fragmentation of contiguous vs. paged allocation for a request that reserves `max_seq_len = 2048` but actually stops after generating 100 tokens. (a) Under contiguous max-length allocation, how many token-slots are wasted, and what fraction of the reserved region is that? (b) Under paged allocation, how many blocks are allocated, how many token-slots are wasted in the final partial block, and what fraction of the *used* blocks' capacity is that?

??? note "Solution"

    **(a) Contiguous.** The system reserves all 2048 slots but only 100 hold real token state. Wasted slots:

    $$
    2048 - 100 = 1948 \text{ slots wasted}, \quad \frac{1948}{2048} \approx 95.1\%.
    $$

    Over 95% of the reserved region is dead space held for the request's lifetime.

    **(b) Paged.** Blocks are allocated on demand. To hold 100 tokens:

    $$
    \lceil 100 / 16 \rceil = \lceil 6.25 \rceil = 7 \text{ blocks}, \text{ total capacity } 7 \cdot 16 = 112 \text{ slots}.
    $$

    Only the final block is partial: $100 \bmod 16 = 4$ tokens used in it, so $16 - 4 = 12$ slots wasted (all other blocks are full). Wasted fraction of allocated capacity:

    $$
    \frac{12}{112} \approx 10.7\%.
    $$

    And the waste is bounded by at most $B - 1 = 15$ slots regardless of sequence length, so for longer sequences the wasted fraction shrinks toward zero — versus contiguous allocation, where the waste stays enormous whenever the actual length falls far short of `max_seq_len`.

**4.** *(Quantitative + conceptual.)* You run **parallel sampling**: one 800-token prompt, $k = 4$ samples of 150 output tokens each, on the 70B GQA model ($\beta \approx 0.3125$ MiB/token), block size $B = 16$. (a) Compute the total KV memory under naive contiguous allocation (no sharing). (b) Compute it under paged allocation with copy-on-write prefix sharing, counting blocks. (c) State the reduction factor, and explain in one sentence why COW copies the divergence block *per block* rather than duplicating the whole prefix.

??? note "Solution"

    **(a) Contiguous, no sharing.** Each of the 4 samples stores the full prompt + its output $= 800 + 150 = 950$ tokens:

    $$
    4 \cdot 950 \cdot 0.3125 \text{ MiB} = 4 \cdot 296.875 = 1187.5 \text{ MiB} \approx 1.16 \text{ GiB}.
    $$

    **(b) Paged + COW.** The 800-token prompt is stored *once* and shared by all 4 samples:

    $$
    \lceil 800 / 16 \rceil = 50 \text{ blocks (shared)}.
    $$

    Each sample privately stores its 150 output tokens:

    $$
    \lceil 150 / 16 \rceil = \lceil 9.375 \rceil = 10 \text{ blocks per sample}.
    $$

    Total blocks $= 50 + 4 \cdot 10 = 90$ blocks. Memory:

    $$
    90 \cdot 16 \cdot 0.3125 \text{ MiB} = 90 \cdot 5 = 450 \text{ MiB} \approx 0.44 \text{ GiB}.
    $$

    **(c) Reduction:** $1187.5 / 450 \approx 2.64\times$. COW copies only the single block where two sequences first differ because sharing and reference counting are tracked at *block* granularity; the identical, read-only prefix blocks stay shared and only the contested block is duplicated, turning $O(k \cdot s)$ prefix memory into one shared copy plus a per-sample tail.

**5.** *(Implementation.)* Extend the chapter's `BlockManager` with a method

    ```python
    def num_free_blocks(self): ...
    def can_allocate(self, num_tokens): ...
    ```

    where `num_free_blocks` returns the number of physical blocks currently on the free list, and `can_allocate(num_tokens)` returns `True` iff a fresh sequence of `num_tokens` tokens could be allocated right now (i.e. enough free blocks exist). Then write a short snippet that constructs a `BlockManager(num_blocks=8, block_size=16)`, checks whether a 200-token request fits, and prints the result. (Note: $\lceil 200/16 \rceil = 13$ blocks are needed, but only 8 exist.)

??? note "Solution"

    Both methods are pure bookkeeping over the existing free list; they add no state. `can_allocate` reuses the same block-count formula (`ceil(num_tokens / block_size)`) that `allocate` uses, so the check stays consistent with what allocation will actually consume.

    ```python
    def num_free_blocks(self):
        """How many physical blocks are currently reclaimable."""
        return len(self.free)

    def can_allocate(self, num_tokens):
        """True iff a NEW sequence of `num_tokens` tokens fits in free memory."""
        n_blocks = (num_tokens + self.block_size - 1) // self.block_size
        return n_blocks <= len(self.free)
    ```

    Driver snippet:

    ```python
    mgr = BlockManager(num_blocks=8, block_size=16)
    need = (200 + 16 - 1) // 16          # ceil(200/16) = 13 blocks needed
    print("free blocks:", mgr.num_free_blocks())   # 8
    print("blocks needed for 200 tokens:", need)   # 13
    print("can allocate 200-token request:", mgr.can_allocate(200))  # False
    print("can allocate 100-token request:", mgr.can_allocate(100))  # ceil(100/16)=7 <= 8 -> True
    ```

    Output:

    ```
    free blocks: 8
    blocks needed for 200 tokens: 13
    can allocate 200-token request: False
    can allocate 100-token request: True
    ```

    This is exactly the admission check a scheduler runs before accepting a new request: if `can_allocate` is `False`, the request waits (or the engine preempts a running sequence to free blocks), which is the paged analogue of the OS refusing to page in a process when no frames are free.

**6.** *(Conceptual, harder.)* The `paged_attention_decode` kernel adds exactly one operation over a standard FlashAttention decode kernel: `phys = block_table[lb]`. (a) Why does this single indirection not change the *numerical result* of attention? (b) The chapter says non-contiguous reads are "the real cost." Explain the hardware reason, and describe the two mitigations PagedAttention uses to keep the tax small.

??? note "Solution"

    **(a) Numerics are unchanged.** Attention is a permutation-order-independent reduction *in structure but not in indexing*: each cached token $j$ contributes one term $\exp(q^\top k_j/\sqrt{d_h}) v_j$ to the numerator and one to the denominator. The block table only changes *where in HBM* the kernel fetches $k_j, v_j$ from; it does not change *which* $(k_j, v_j)$ pairs are gathered, their values, or the order in which logical positions are visited (the loop still walks logical blocks $0, 1, 2, \dots$ in order and, within a block, token offsets in order). Since the online-softmax accumulation (running max $m$, denominator $l$, output $out$) is mathematically the same associative reduction over the same set of terms, the final normalized output $out / l$ is bit-for-bit the same computation as a contiguous kernel reading the identical K/V. The indirection is an address translation, not a change to the math.

    **(b) Cost and mitigations.** The hardware reason: GPUs achieve peak HBM bandwidth through *coalesced*, large, contiguous memory transactions and hardware prefetching. When consecutive logical blocks live at scattered physical addresses (blocks 7, 1, 4, ...), the kernel issues loads that jump around HBM, defeating prefetchers and preventing the wide coalesced bursts a fully contiguous tensor would allow — so effective bandwidth drops even though the total bytes read are the same.

    Two mitigations:

    1. **Keep each block internally contiguous.** A physical block stores its $B \times H_{kv} \times d_h$ K/V elements in a contiguous region, so *within* a block the loads are fully coalesced — scattering happens only at block *boundaries*, not on every element.
    2. **Choose $B$ large enough (16+).** With a larger block, each expensive "jump to a new physical location" is amortized over $B$ tokens' worth of contiguous, coalesced reads, so the per-block indirection/setup cost is a small fraction of the useful load. (This is the same trade-off discussed in "Choosing the block size $B$": too small bloats overhead, too large reintroduces internal fragmentation.)

    The net effect, per the vLLM authors, is that the paged kernel runs within a small percentage of a perfectly contiguous FlashAttention kernel — a tax overwhelmingly repaid by the larger batches the recovered memory enables.
