# 7.3 vLLM: Architecture, PagedAttention & Internals

If there is a single piece of open-source software that defined the era of practical LLM serving, it is **vLLM**. Born from a 2023 Berkeley paper (Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*), it took an idea borrowed from operating systems — paging — and applied it to the one resource that dominates LLM inference memory: the **KV cache**. The result was a throughput jump of several times over the contemporary baselines, and within a year vLLM became the default backend for an enormous fraction of self-hosted inference, RL rollout engines, and managed APIs.

This chapter is a *reference* on how vLLM actually works inside. We assume you already understand the mechanics of prefill, decode, and the KV cache from [The Anatomy of LLM Inference: Prefill, Decode & The KV Cache](../07-inference-serving/01-anatomy-inference.html), and the idea of running many requests in one forward pass from [Continuous Batching & Request Scheduling](../07-inference-serving/02-continuous-batching.html). PagedAttention itself — the kernel and the memory-management algorithm — is developed in depth in [PagedAttention & KV-Cache Memory Management](../04-kernels-efficiency/06-paged-attention-kv.html); here we recap it just enough to understand vLLM's *system*, then spend most of our time on the block manager, the scheduler, the engine/executor stack, the rewritten **V1** architecture, and the features that make vLLM a production engine: prefix caching, speculative decoding, and multi-LoRA. We close with how to actually run and tune it.

## Why the KV cache is the bottleneck (and what PagedAttention fixes)

Recall the shape of autoregressive inference. **Prefill** runs the whole prompt through the network in one parallel pass and produces a key and value vector for every layer and every prompt token; those are stored in the **KV cache**. **Decode** then generates one token at a time, and each new token appends one more K and V vector per layer. The cache grows by one token-slot per step and is read in full on every step.

The per-request KV cache size is exact and worth memorizing:

$$
\text{bytes} = 2 \times L \times H_{kv} \times d_{h} \times S \times b
$$

where $L$ is the number of layers, $H_{kv}$ the number of **key/value** heads (after GQA/MQA — see [Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html)), $d_h$ the head dimension, $S$ the sequence length, $b$ the bytes per element, and the leading $2$ counts keys and values.

The systems problem is not the total — it is the *dynamics*. A request's final length is unknown when it arrives. Pre-vLLM engines handled this by **pre-allocating a contiguous buffer** for each request sized to `max_seq_len`. This is catastrophic for memory:

- **Internal fragmentation.** A request that generates 50 tokens but was allocated 2048 slots wastes 97% of its reservation for its entire lifetime.
- **External fragmentation.** Contiguous buffers of different sizes leave unusable gaps between them, exactly like a poorly managed heap.
- **No sharing.** Two requests with an identical system prompt each keep their own full copy.

The vLLM paper measured that in such systems only ~20–40% of KV memory held actual tokens; the rest was reserved-but-empty or fragmented. Since KV memory caps the number of concurrent requests, and concurrency caps throughput, this waste translated almost linearly into lost throughput.

### PagedAttention in one paragraph

The operating-systems answer to fragmentation is **virtual memory with paging**: divide memory into fixed-size pages, let a process see a contiguous virtual address space, and map each virtual page to any physical page through a page table. PagedAttention applies this exactly. The KV cache is carved into fixed-size **blocks** (vLLM's term for pages), each holding the K and V vectors for a fixed number of tokens — the **block size**, commonly 16. A request's logical sequence of tokens maps, via a **block table**, to a list of physical blocks that need not be contiguous in GPU memory.

{{fig:vllm-paged-block-table-mapping}}

The payoff:

- **Near-zero internal waste.** Only the *last* partially-filled block of each sequence is under-utilized — at most `block_size - 1` slots, independent of `max_seq_len`. With block size 16 the worst-case waste per sequence is 15 token-slots, versus thousands before.
- **No external fragmentation.** All blocks are the same size, so any free block fits any request. Allocation is `O(1)` off a free list.
- **Copy-on-write sharing.** Two sequences can point their block tables at the *same* physical block. This is the mechanism behind shared prompts and parallel sampling.

The cost is a custom attention kernel: instead of reading K and V from one contiguous tensor, the kernel must gather them block-by-block using the block table. That gather is the PagedAttention CUDA kernel. The arithmetic of attention is unchanged — only the addressing is indirected. We develop that kernel in [PagedAttention & KV-Cache Memory Management](../04-kernels-efficiency/06-paged-attention-kv.html); here we treat it as a primitive and build the engine around it.

!!! note "Block size is a real tuning knob"
    Larger blocks mean fewer block-table entries and slightly more efficient kernels, but more internal waste in the last block and coarser-grained prefix sharing. Block size 16 is the long-standing default for most models; FP8 KV caches and certain attention backends prefer other values. It is exposed as `block_size` and you rarely need to change it.

## The block manager: paging for the KV cache

The **block manager** (in V1, the `KVCacheManager`) is vLLM's memory allocator. It owns the pool of physical KV blocks and hands them out to sequences. Think of it as `malloc`/`free` plus a page-table — but tuned for the append-only, highly-shareable access pattern of LLM decode.

### How many blocks exist?

At startup vLLM runs a **memory profiling** pass. It loads the model weights, runs a dummy forward at the configured maximum batch, and measures peak activation memory. Whatever is left of GPU memory — up to a fraction set by `gpu_memory_utilization` (default ~0.9) — is dedicated to the KV cache. Dividing that by the bytes-per-block gives the total number of physical blocks. This is why vLLM "grabs" most of your GPU on launch: it is pre-carving the entire KV pool so allocation at runtime is just popping from a free list.

```python
# Sketch of vLLM's startup KV-budget computation (numbers illustrative).
total_gpu_bytes      = 80 * 1024**3          # 80 GB GPU (e.g. A100/H100)
weight_bytes         = 14 * 1024**3          # ~14 GB for a 7B model in bf16
peak_activation      = 4  * 1024**3          # measured by a profiling forward pass
util                 = 0.90                  # gpu_memory_utilization

usable               = int(total_gpu_bytes * util)        # ~72 GB
kv_cache_bytes       = usable - weight_bytes - peak_activation

# Bytes for one block: 2 (K and V) * block_size * num_kv_heads * head_dim
#                       * num_layers * dtype_bytes
block_size, n_kv_heads, head_dim = 16, 8, 128
n_layers, dtype_bytes            = 32, 2     # bf16
bytes_per_block = 2 * block_size * n_kv_heads * head_dim * n_layers * dtype_bytes

num_gpu_blocks = kv_cache_bytes // bytes_per_block
print(f"KV pool: {kv_cache_bytes/1024**3:.1f} GB -> {num_gpu_blocks} blocks "
      f"({num_gpu_blocks * block_size} token-slots)")
```

The total token-slots (`num_gpu_blocks * block_size`) is the hard ceiling on *how many tokens of KV cache can exist across all running requests at once*. The scheduler's whole job is to keep total demand under this ceiling.

### The free list and the block table

The allocator maintains a free list of physical block IDs. Each sequence keeps a **block table**: an ordered list mapping its logical block index to a physical block ID. The core operations:

```python
class BlockManager:
    """Simplified KV-block allocator (V0-flavoured for clarity)."""

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))   # the free list
        self.ref_count = {}                          # phys_block -> refcount

    def can_allocate(self, seq_len: int) -> bool:
        n = (seq_len + self.block_size - 1) // self.block_size
        return len(self.free_blocks) >= n

    def allocate(self, seq_len: int) -> list[int]:
        n = (seq_len + self.block_size - 1) // self.block_size
        table = []
        for _ in range(n):
            blk = self.free_blocks.pop()             # O(1)
            self.ref_count[blk] = 1
            table.append(blk)
        return table                                  # this is the block table

    def can_append(self, seq) -> bool:
        """Room for one more decode token? A fresh block is only needed when
        the current tokens exactly fill the last block."""
        boundary = seq.num_tokens % self.block_size == 0
        return len(self.free_blocks) >= (1 if boundary else 0)

    def append_slot(self, block_table: list[int], cur_len: int) -> int | None:
        """Called each decode step. `cur_len` is the number of tokens already
        stored; the token about to be generated lands at position `cur_len`.
        Grow the table by one block ONLY when that position starts a fresh
        block (cur_len % block_size == 0); otherwise the last block still has
        a free slot and we return None. (Mirrors 4.6's BlockManager.append_token:
        without this guard we would pop one block per sequence *per step*, a
        block_size-fold over-allocation.)"""
        if cur_len % self.block_size != 0:
            return None                              # room in the last block
        blk = self.free_blocks.pop()                 # boundary: need a new block
        self.ref_count[blk] = 1
        block_table.append(blk)
        return blk

    def free(self, block_table: list[int]) -> None:
        for blk in block_table:
            self.ref_count[blk] -= 1
            if self.ref_count[blk] == 0:             # last owner releases it
                self.free_blocks.append(blk)
```

Two design points carry their weight:

- **Lazy, incremental growth.** A sequence allocates one block at a time as decode crosses block boundaries (`append_slot`). It never reserves space for tokens it has not yet produced. This is the direct cure for internal fragmentation.
- **Reference counting enables sharing.** A physical block can be referenced by multiple sequences. `free` only returns a block to the pool when its refcount hits zero. This single mechanism powers parallel sampling, beam search, and prefix caching.

### Copy-on-write for shared blocks

When two sequences share a block (refcount > 1) and one of them needs to *write* into it — e.g. two samples diverged and one wants to append a token into a block the other still reads — vLLM does **copy-on-write**: allocate a fresh block, copy the shared block's contents into it, point the writer's block table at the copy, and decrement the original's refcount. Identical to COW in `fork()`. It means shared prefixes cost memory only once until the moment of divergence.

```python
def append_with_cow(self, block_table, idx):
    """Ensure block_table[idx] is writable; copy if shared."""
    blk = block_table[idx]
    if self.ref_count[blk] > 1:                # shared -> must not clobber
        new_blk = self.free_blocks.pop()
        # GPU-side: copy KV contents of `blk` into `new_blk`
        self.ref_count[blk]   -= 1
        self.ref_count[new_blk] = 1
        block_table[idx] = new_blk             # writer now owns its private copy
        return ("copy", blk, new_blk)          # scheduler emits a copy op
    return ("noop",)
```

## The scheduler: continuous batching under a memory budget

vLLM's scheduler is the brain. On every engine step it decides *which* sequences run in the next forward pass, subject to the block budget. It implements **continuous batching** (also called iteration-level scheduling): requests join and leave the running batch at token granularity, so a finished sequence frees its slot mid-flight and a waiting request fills it on the very next step. The general technique is covered in [Continuous Batching & Request Scheduling](../07-inference-serving/02-continuous-batching.html); here we look at vLLM's specific policy.

### The three queues and the step loop

Sequences live in three states:

- **WAITING** — admitted but no KV blocks yet (never run, or preempted-and-recomputed).
- **RUNNING** — has KV blocks; will be in the next forward pass.
- **SWAPPED** — preempted, its KV blocks evicted to CPU RAM (V0 swap path).

A schedule step, conceptually:

```python
def schedule_step(self):
    scheduled, blocks_to_copy = [], []

    # 1. Keep RUNNING sequences going; each decode step may need +1 block.
    for seq in self.running:
        while not self.block_mgr.can_append(seq):
            # Out of memory: preempt the lowest-priority running seq.
            victim = self.running.pop()              # tail = newest/lowest prio
            self._preempt(victim)                    # recompute or swap
            if victim is seq:
                break
        if seq in self.running:
            self.block_mgr.append_slot(seq.block_table, seq.num_tokens)
            scheduled.append(seq)

    # 2. Admit WAITING sequences (prefill) if budget and token quota allow.
    while self.waiting and self._budget_left(scheduled):
        seq = self.waiting[0]
        if not self.block_mgr.can_allocate(seq.prompt_len):
            break                                    # not enough free blocks
        self.waiting.popleft()
        seq.block_table = self.block_mgr.allocate(seq.prompt_len)
        seq.state = RUNNING
        scheduled.append(seq)

    return scheduled, blocks_to_copy
```

The subtlety in step 1 is **preemption**. Decode is monotonic: every running sequence needs at most one new block per step, and once you are decoding you cannot pause a sequence without losing forward progress. So when the pool is exhausted, vLLM evicts a *whole* sequence to make room, with two recovery strategies:

- **Recomputation (default in V1).** Drop the victim's KV blocks entirely and move it back to WAITING. When rescheduled, re-run prefill over its prompt *plus tokens generated so far*. Wastes compute but frees memory instantly and needs no CPU transfer. Because prefill is compute-bound and fast, this is usually the better choice.
- **Swapping (V0).** Copy the victim's KV blocks to pinned CPU memory, free the GPU blocks, and copy back when rescheduled. Saves recompute but pays PCIe bandwidth twice.

!!! warning "Preemption storms degrade tail latency"
    If you admit more requests than the KV pool can sustain, the scheduler thrashes: requests are admitted, partly decoded, preempted, recomputed, preempted again. Throughput collapses and p99 latency explodes. The cure is `max_num_seqs` and `max_num_batched_tokens` sized so steady-state demand fits the pool — see the tuning section. Watch the `num_preemptions` counter in the logs.

### Prefill, decode, and chunked prefill

A naive scheduler runs either a batch of prefills or a batch of decodes. The trouble: a long prefill (say a 32k-token prompt) monopolizes a forward pass and stalls every decoding request, spiking inter-token latency for everyone — a head-of-line blocking problem.

**Chunked prefill** fixes this by splitting a long prompt into chunks of at most `max_num_batched_tokens` and processing one chunk per step, *interleaved with decodes of other requests in the same batch*. A single forward pass thus mixes a slice of prefill tokens with many one-token decodes. This smooths inter-token latency and keeps the GPU busy (decode alone is memory-bound and under-utilizes compute; mixing in prefill tokens raises arithmetic intensity — see the roofline view in [The Roofline Model & Performance Engineering](../04-kernels-efficiency/01-roofline-performance.html)). In V1 chunked prefill is on by default and the prefill/decode distinction largely dissolves into a single "token budget" per step. The disaggregation alternative — running prefill and decode on *separate* machines — is covered in [Disaggregated Prefill/Decode & Chunked Prefill](../07-inference-serving/08-disaggregated-chunked-prefill.html).

## The engine and executor stack

Around the scheduler sits the request-handling and execution machinery. From the outside in:

{{fig:vllm-engine-executor-stack}}

- **LLMEngine / EngineCore** is the orchestrator. It owns the scheduler and the block manager, accepts requests, drives the step loop, runs the tokenizer/detokenizer, and streams outputs. In V1 the heavy loop runs in a dedicated **EngineCore** process so Python overhead (tokenization, HTTP, scheduling) overlaps with GPU execution.

- **Executor** abstracts *where and how* the model runs: a single process for one GPU, a multiprocessing or Ray-based executor for tensor/pipeline parallelism across GPUs and nodes. It broadcasts the `ExecuteModelRequest` to every worker and gathers results. Multi-GPU details are in [Multi-GPU & Multi-Node Inference](../07-inference-serving/11-multi-gpu-inference.html).

- **Worker** owns one GPU's slice of the model and its KV blocks. It holds the **ModelRunner**, which turns the scheduler's abstract plan into concrete GPU work: it flattens token IDs and positions into tensors, assembles per-sequence block tables and slot mappings, runs the forward pass through the model with the chosen **attention backend** (FlashAttention, FlashInfer, or the PagedAttention kernel), and samples the next tokens.

### The forward step in detail

Each engine step, the ModelRunner builds inputs for a batch that may mix prefill chunks and decodes. The crucial tensors are the **slot mapping** (for each token being written, the exact physical KV slot to write its K and V into) and the **block tables** (for each sequence, where to *read* prior KV from).

```python
# Conceptual ModelRunner input assembly for one step.
def prepare_inputs(scheduled_seqs, block_size):
    input_ids, positions, slot_mapping = [], [], []
    block_tables = []                         # padded [num_seqs, max_blocks]

    for seq in scheduled_seqs:
        new_tokens = seq.tokens_to_process()  # prompt chunk (prefill) or 1 (decode)
        start = seq.num_computed_tokens
        for i, tok in enumerate(new_tokens):
            pos = start + i
            input_ids.append(tok)
            positions.append(pos)
            # Where does this token's K/V get written?
            logical_block = pos // block_size
            offset        = pos %  block_size
            phys_block    = seq.block_table[logical_block]
            slot_mapping.append(phys_block * block_size + offset)
        block_tables.append(seq.block_table)

    return {
        "input_ids":    torch.tensor(input_ids,    device="cuda"),
        "positions":    torch.tensor(positions,    device="cuda"),
        "slot_mapping": torch.tensor(slot_mapping, device="cuda"),
        "block_tables": pad_and_stack(block_tables, device="cuda"),
    }
```

Inside each attention layer, the freshly computed K and V are scattered into the cache at `slot_mapping`, then PagedAttention reads K and V for the whole context by gathering blocks via `block_tables`. The MLP and norms are ordinary dense ops. Because the same kernel handles any mix of prefill and decode tokens, the scheduler is free to pack them however it likes.

### CUDA graphs and `torch.compile`

Decode steps are tiny — one token per sequence — so per-kernel **launch overhead** (the CPU cost of dispatching each CUDA kernel) becomes a large fraction of step time. vLLM captures the decode forward pass into a **CUDA graph**: the sequence of kernel launches is recorded once for a given batch shape and replayed as a single GPU submission, slashing CPU overhead. Because graphs are shape-specialized, vLLM captures a set of graphs for a ladder of batch sizes and pads each real batch up to the nearest captured size. V1 deepens this with a `torch.compile`-based path (piecewise compilation that leaves attention as a custom op while compiling the rest), further fusing kernels. See [Kernel Fusion, torch.compile, CUDA Graphs & Compilers](../04-kernels-efficiency/09-compilers-fusion.html). The flags `enforce_eager=True` (disable graphs, easier debugging, slower) and `--cuda-graph-sizes` control this.

!!! example "Worked example: how many requests fit, and what throughput follows"
    Take Llama-3-8B in bf16 on one 80 GB GPU. The model has $L=32$ layers, GQA with $H_{kv}=8$ KV heads, $d_h=128$, and bf16 means $b=2$ bytes.

    **Per-token KV cost** (both K and V, all layers):
    $$
    2 \times L \times H_{kv} \times d_h \times b
    = 2 \times 32 \times 8 \times 128 \times 2 = 131{,}072 \text{ bytes} \approx 128\text{ KB/token}.
    $$

    **KV budget.** Weights take $\approx 16$ GB. With `gpu_memory_utilization=0.9` we use $\approx 72$ GB; reserve $\approx 4$ GB for activations, leaving $\approx 52$ GB for KV:
    $$
    \frac{52 \times 1024^3 \text{ bytes}}{128 \times 1024 \text{ bytes/token}} \approx 425{,}000 \text{ token-slots}.
    $$

    **Concurrency.** If the average request holds $\approx 2{,}000$ tokens of context (prompt + generated), that is $425{,}000 / 2{,}000 \approx 210$ concurrent requests. A pre-paging engine that reserved `max_seq_len = 8192` per request would fit only $425{,}000 / 8{,}192 \approx 52$ — a $4\times$ concurrency loss purely to fragmentation.

    **Block count.** With block size 16, one block holds $16 \times 128\text{KB} = 2$ MB, so the pool is $\approx 52\text{GB} / 2\text{MB} \approx 26{,}500$ blocks. Worst-case internal waste is 15 token-slots per sequence — about $15/2000 = 0.75\%$ — versus the old design's reservation-dominated waste.

    The throughput consequence is direct: decode is memory-bandwidth bound, so reading $\sim4\times$ more concurrent sequences' KV per unit time (because $4\times$ more fit) yields roughly $4\times$ the tokens/second, until you saturate HBM bandwidth or compute.

## V1: the rewritten architecture

In 2024–2025 vLLM was substantially re-architected as **V1**, now the default engine. V0 had accreted features over a fast-moving codebase, and its single-process design left GPU bubbles whenever the CPU was busy tokenizing, scheduling, or detokenizing. V1's themes:

- **Isolated EngineCore process.** The scheduler + block manager + model execution loop run in their own process, communicating with the API/tokenizer side over a fast IPC channel. This **overlaps CPU work with GPU execution** — while the GPU runs step $t$, the front end is already tokenizing requests and detokenizing outputs for step $t-1$/$t+1$. The result is fewer GPU idle bubbles and higher utilization, especially at high request rates.

- **Unified scheduler with a token budget.** V1 removes the rigid "prefill batch vs decode batch" split. The scheduler simply allocates a per-step **token budget** (`max_num_batched_tokens`) across whatever sequences want to run, mixing prefill chunks and decodes freely. Chunked prefill becomes the default, not an option.

- **Prefix caching on by default.** V1's KV-cache manager treats cached prefixes as a first-class citizen (next section), with low enough overhead that it is enabled out of the box rather than opt-in.

- **`torch.compile` + piecewise CUDA graphs** as the default execution path, replacing much hand-written V0 glue and improving portability across hardware backends.

- **Cleaner extension points** for new attention backends, hardware (the `Platform` abstraction for NVIDIA/AMD/TPU/CPU), speculative decoding, and structured output.

For the user, V1 is mostly transparent: the `LLM(...)` and OpenAI-server interfaces are unchanged. What you observe is higher throughput, lower overhead at small batch sizes, and prefix caching helping for free. If you need a V0-only feature during a migration you can sometimes force the old engine, but new development targets V1.

## Prefix caching: reuse KV across requests

Many requests share a prefix: a long system prompt, a few-shot exemplar block, a chat history that grows by one turn, or — in agentic and RL workloads — a giant fixed instruction reused across thousands of rollouts. Recomputing that prefix's KV every time is pure waste. **Automatic prefix caching (APC)** lets vLLM *reuse the KV blocks of a previously computed prefix* across requests, turning an expensive prefill into a cheap cache hit.

### How it works: hashing blocks

Because the KV cache is already block-structured, sharing is natural. vLLM computes a **hash for each full block** that incorporates the token IDs in that block *and the hash of all preceding blocks* — a rolling hash that makes the block ID a function of the entire prefix up to that point. (Two prefixes that diverge at token 5 get different hashes from block 0 onward; two identical prefixes get identical hashes block-for-block.)

```python
def block_hash(prev_hash, token_ids_in_block, extra=None):
    """Hash of a *full* block = (hash of everything before) + (this block's
    tokens) + optional extras (LoRA id, multimodal hash, cache salt)."""
    return hash((prev_hash, tuple(token_ids_in_block), extra))

def hash_prompt_blocks(prompt_ids, block_size, lora_id=None):
    hashes, h = [], None
    for i in range(0, len(prompt_ids) - block_size + 1, block_size):
        block = prompt_ids[i:i + block_size]      # only FULL blocks are hashable
        h = block_hash(h, block, extra=lora_id)
        hashes.append(h)
    return hashes
```

The block manager keeps a map from `block_hash -> physical_block_id` for blocks whose contents are "committed" (full and immutable). When a new request arrives:

1. Hash its prompt blocks.
2. For each leading block whose hash is already in the map, **reuse** that physical block: point the new request's block table at it and bump the refcount. No compute, no new memory.
3. At the first block that misses, stop matching; allocate fresh blocks for the remaining (uncached) tokens and prefill only those.

A new request with a 4000-token cached system prompt and 50 new tokens prefills only ~50 tokens instead of 4050 — a roughly $80\times$ reduction in prefill work for that request, and a large drop in time-to-first-token.

### Eviction and correctness

Cached blocks still occupy the pool. When the pool is full and a new allocation is needed, vLLM evicts cached blocks that no running sequence references (refcount via the cache, not active use), typically **LRU** with awareness of how deep in a prefix the block sits (evict leaves before roots so popular shared prefixes survive). Eviction is lossless: a re-request just recomputes.

Correctness hinges on the hash covering *everything that affects the KV*: token IDs, position (implied by block order), the active **LoRA adapter** (different adapters produce different KV — see multi-LoRA below), and a **cache salt** you can set to isolate tenants so one user's cached prefix can never be served to another. Multimodal inputs hash their image/audio features too. Get this wrong and you would serve one request's KV to another — a correctness and security bug — which is why the hashing is conservative. Prefix caching has its own dedicated chapter, [Prefix Caching & KV-Cache Reuse](../07-inference-serving/07-prefix-caching.html); here the point is that it falls out almost for free from the paged design plus reference counting.

!!! tip "Prefix caching is a giant win for agents and RL rollouts"
    Workloads that hammer a fixed instruction block — coding agents, ReAct loops, GRPO/PPO rollouts that sample many completions from the same prompt — see the largest gains, because the shared prefix can be many thousands of tokens reused thousands of times. This is also why vLLM is the rollout engine of choice in RL stacks like veRL ([veRL: HybridFlow & The Single-Controller Architecture](../06-rl-infra/04-verl.html)). It is on by default in V1; you can disable with `enable_prefix_caching=False` if your traffic has no shared prefixes and you want to avoid the (small) hashing overhead.

## Speculative decoding support

Decode is memory-bound: each step you load the entire model's weights from HBM to produce *one* token. The hardware could do far more arithmetic per byte loaded. **Speculative decoding** exploits this by having a cheap **drafter** propose several future tokens, then verifying them with the big model in a *single* forward pass; accepted tokens come "for free" because verifying $k$ tokens costs about the same memory traffic as generating one. The algorithms (draft models, Medusa, EAGLE, n-gram/lookahead) are the subject of [Speculative Decoding: Draft Models, Medusa, EAGLE & Lookahead](../07-inference-serving/06-speculative-decoding.html); here we note vLLM's *plumbing*.

vLLM supports several proposers behind a common interface:

- **Draft model** — a small model of the same family proposes tokens.
- **N-gram / prompt lookahead** — propose by matching recent context against the prompt; free, great for tasks with copying (summarization, code editing).
- **EAGLE / Medusa-style** — lightweight heads on the target model predict multiple future tokens.

Each step, vLLM runs the proposer to get $k$ draft tokens, then runs the target model on all $k+1$ positions at once. A **rejection-sampling verifier** accepts the longest prefix of drafts consistent with the target's distribution and corrects the first rejected token, so the output distribution is *provably identical* to plain sampling from the target — speculation changes speed, not what is generated.

```python
# Engine-level shape of one speculative step (target distribution preserved).
draft_tokens, draft_probs = proposer.propose(seq, k)          # cheap
target_logits = target_model(seq.context + draft_tokens)      # ONE big fwd, k+1 pos
accepted = rejection_sample(draft_tokens, draft_probs, target_logits)  # 0..k accepted
seq.extend(accepted)
seq.extend([sample(target_logits[len(accepted)])])            # +1 bonus/correction token
# Net: up to k+1 tokens emitted per single target forward pass.
```

The system challenge is that speculation must coexist with paged KV and continuous batching: the KV for *rejected* draft tokens must be discarded (their blocks/slots rolled back), and batches mix sequences accepting different numbers of tokens. V1 re-implemented spec decode to fit the unified scheduler cleanly. The **speedup** depends on the **acceptance rate** $\alpha$ and draft cost; expected tokens per target step is roughly $\frac{1-\alpha^{k+1}}{1-\alpha}$, so high-$\alpha$, predictable text (code, structured output) benefits most, while creative high-entropy text benefits least — and a poor drafter can even *slow you down* because of the wasted draft compute.

## Multi-LoRA serving

LoRA (low-rank adaptation — [PEFT I: LoRA, QLoRA, DoRA & The Adapter Family](../05-posttraining-alignment/03-peft-lora-qlora.html)) fine-tunes a model by adding a low-rank update $\Delta W = BA$ to selected weight matrices, where $B \in \mathbb{R}^{d\times r}$, $A \in \mathbb{R}^{r\times d}$, and the rank $r$ is tiny (8–64). The adapter is a few megabytes versus tens of gigabytes for the base model. vLLM's **multi-LoRA** serving exploits this: keep *one* copy of the base model in GPU memory and serve **many different fine-tunes simultaneously** by swapping in the small adapters, even mixing requests for different adapters in the same batch.

The key kernel insight is that you do not merge adapters into the weights (that would force one adapter per batch). Instead you keep the base forward pass shared and add each request's LoRA contribution as a **batched, gathered low-rank matmul**. For a layer's output $y = Wx$, the LoRA-augmented output for request $i$ using adapter $a(i)$ is:

$$
y_i = W x_i + \frac{\alpha}{r}\, B_{a(i)} \big(A_{a(i)}\, x_i\big).
$$

vLLM's **Punica/SGMV-style** kernels compute the $B(Ax)$ term for a whole batch where each row may use a *different* adapter, gathering the right $A, B$ per row. This makes the marginal cost of serving $N$ adapters close to serving one, as long as the adapters fit in memory.

```python
from vllm import LLM
from vllm.lora.request import LoRARequest

# Base model loaded once; enable LoRA with a max rank and a cap on
# how many distinct adapters may be active in a single batch.
llm = LLM(model="meta-llama/Llama-3-8b",
          enable_lora=True,
          max_loras=4,        # distinct adapters per *batch*
          max_lora_rank=16,   # must be >= rank of any adapter you load
          max_cpu_loras=32)   # adapters parked in CPU, paged to GPU on demand

# Each request names the adapter it wants. Different adapters can be
# batched together in the same forward pass.
out = llm.generate(
    ["Translate to French: Hello", "Summarize: ...long doc..."],
    lora_request=[
        LoRARequest("fr-translator", 1, "/adapters/fr_lora"),
        LoRARequest("summarizer",    2, "/adapters/sum_lora"),
    ],
)
```

Adapters themselves are **paged like the KV cache**: a pool of GPU adapter slots, with inactive adapters held in CPU memory (`max_cpu_loras`) and copied in on demand. `max_loras` caps distinct adapters *per batch* (kernel/memory limit); `max_cpu_loras` caps how many are kept warm. This is the backbone of multi-tenant "one base model, hundreds of customer fine-tunes" serving. Note the interaction with prefix caching: the LoRA ID is part of the block hash, so two requests can only share cached prefix blocks if they use the *same* adapter.

!!! interview "Interview Corner"
    **Q:** vLLM gets a large throughput win over a naive HuggingFace `generate` serving loop. Mechanistically, where does that win come from, and what is the single biggest lever?

    **A:** The win is overwhelmingly about **KV-cache memory efficiency translating into concurrency**. A naive loop pre-allocates a contiguous KV buffer sized to `max_seq_len` per request, so 60–80% of KV memory is reserved-but-empty (internal fragmentation) or unusable gaps (external fragmentation). Since KV memory caps how many sequences run concurrently, and decode throughput scales with concurrency (it's HBM-bandwidth bound — more sequences read per unit time = more tokens/sec), that wasted memory is wasted throughput. **PagedAttention** removes the fragmentation by paging the KV cache into fixed-size blocks mapped through a per-sequence block table, so the only waste is the last partial block (≤ `block_size − 1` slots). That can quadruple concurrency on typical workloads. **Continuous batching** then keeps that concurrency saturated by admitting and retiring requests at token granularity instead of waiting for a whole batch to finish. The single biggest lever is the paged KV enabling high concurrency; continuous batching, prefix caching, CUDA graphs, and chunked prefill are multipliers on top. A good follow-up answer names the failure mode: oversubscribing the KV pool causes preemption thrashing that destroys tail latency, so `gpu_memory_utilization`, `max_num_seqs`, and `max_num_batched_tokens` must be tuned to keep steady-state demand under the pool size.

## Running and tuning vLLM

### Two entry points

**Offline batched inference** — drive the engine directly from Python; best for evals, dataset generation, and RL rollouts:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3-8b-instruct",
    tensor_parallel_size=2,          # shard across 2 GPUs (TP)
    gpu_memory_utilization=0.90,     # fraction of GPU for weights + KV
    max_model_len=8192,              # caps KV per request; lower = more concurrency
    enable_prefix_caching=True,      # default in V1; reuse shared prefixes
    dtype="bfloat16",
)
params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=256)
for o in llm.generate(["Explain PagedAttention in one sentence."], params):
    print(o.outputs[0].text)
```

**Online OpenAI-compatible server** — a drop-in replacement for the OpenAI API, used by virtually every "self-host an LLM" deployment:

```bash
# Launch an OpenAI-compatible HTTP server.
vllm serve meta-llama/Llama-3-8b-instruct \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --max-num-seqs 256 \
    --enable-chunked-prefill \
    --port 8000

# Then call it exactly like the OpenAI API:
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3-8b-instruct",
       "messages":[{"role":"user","content":"Hello!"}]}'
```

### The knobs that actually matter

| Flag | What it controls | How to think about it |
|---|---|---|
| `gpu_memory_utilization` | Fraction of GPU for weights + KV pool | Higher = bigger KV pool = more concurrency, but risks OOM from activation spikes. 0.85–0.92 typical. |
| `max_model_len` | Max context (prompt + output) per request | Caps per-request KV; set to the largest you truly need, not the model's max — lower frees KV for more concurrency. |
| `max_num_seqs` | Max concurrent sequences per batch | Too high → preemption thrash; too low → idle GPU. Tune with the preemption counter. |
| `max_num_batched_tokens` | Token budget per forward step | Bigger favors throughput/prefill; smaller favors inter-token latency. Key chunked-prefill knob. |
| `tensor_parallel_size` | GPUs to shard one model across | Use when weights+KV don't fit on one GPU, or to cut latency. See [Multi-GPU & Multi-Node Inference](../07-inference-serving/11-multi-gpu-inference.html). |
| `enable_prefix_caching` | Reuse KV of shared prefixes | On by default in V1; huge for shared system prompts / agents / RL. |
| `quantization` | Weight quant (`awq`, `gptq`, `fp8`, …) | Shrinks weights → bigger KV pool & cheaper compute; small quality cost. See [Quantization I](../04-kernels-efficiency/07-quantization-ptq.html). |
| `kv_cache_dtype` | KV cache precision (e.g. `fp8`) | Halves KV bytes → roughly doubles concurrency, slight accuracy cost. |
| `enforce_eager` | Disable CUDA graphs | For debugging only; costs throughput. |

A practical tuning loop:

1. **Pick `max_model_len` honestly.** This single number sets the worst-case KV per request. Cutting an unnecessary 32k cap down to 8k can multiply concurrency.
2. **Push `gpu_memory_utilization` up** until you see activation OOMs, then back off a notch.
3. **Decide your objective.** Throughput-first: large `max_num_batched_tokens`, large `max_num_seqs`. Latency-first (low inter-token latency for chat): smaller `max_num_batched_tokens` with chunked prefill so long prefills don't stall decodes.
4. **Watch the metrics.** vLLM logs/Prometheus expose KV-cache utilization, running/waiting/swapped counts, `num_preemptions`, prefix-cache hit rate, and throughput. If preemptions are nonzero in steady state, you're oversubscribed — lower `max_num_seqs` or `max_model_len`, or quantize to enlarge the pool.
5. **Quantize to buy concurrency.** FP8/INT4 weights and an FP8 KV cache both enlarge the effective KV pool; for many workloads that concurrency gain outweighs the tiny quality cost. Inference economics — the latency/throughput/cost trade-off you are navigating — is the subject of [Inference Economics: Latency, Throughput & Cost](../07-inference-serving/12-inference-economics.html).

!!! warning "Common pitfall: `max_model_len` left at the model maximum"
    Leaving `max_model_len` at a model's full 128k context when your requests are 2k forces vLLM to *reason* about the KV budget conservatively for admission and (in some paths) over-reserve, throttling concurrency for no benefit. Always set `max_model_len` to the largest context you actually serve. Similarly, setting `gpu_memory_utilization` too high leaves no headroom for transient activation spikes during long prefills and triggers OOM crashes mid-traffic.

vLLM is not the only serving engine — [SGLang: RadixAttention & Structured Programs](../07-inference-serving/04-sglang-radixattention.html) pushes prefix sharing further with a radix tree, and [TensorRT-LLM, TGI & Other Serving Stacks](../07-inference-serving/05-trtllm-tgi-stacks.html) trades flexibility for hand-tuned NVIDIA kernels. But vLLM's combination of a clean paged-memory core, broad model and hardware coverage, an active community, and the OpenAI-compatible surface has made it the default. Understanding its internals — the block manager, the scheduler, the executor, and how prefix caching, speculation, and multi-LoRA bolt onto the paged core — is understanding how modern open LLM serving works.

!!! key "Key Takeaways"
    - The KV cache, not the weights, is the dynamic bottleneck in LLM serving; pre-paging engines wasted 60–80% of it to internal and external fragmentation, which directly throttled concurrency and thus throughput.
    - **PagedAttention** pages the KV cache into fixed-size blocks mapped through a per-sequence **block table**, cutting waste to at most one partial block per sequence and enabling `O(1)` allocation, copy-on-write sharing, and prefix reuse.
    - The **block manager** is a refcounted block allocator; the **scheduler** runs continuous batching under the block budget, preempting (by recomputation or swap) when the pool is exhausted — oversubscription causes preemption thrashing that wrecks tail latency.
    - **Chunked prefill** interleaves long prefills with decodes to smooth inter-token latency and raise GPU utilization; in **V1** it is default and the engine runs on a unified per-step token budget.
    - **V1** isolates the engine loop in its own process to overlap CPU and GPU work, defaults to `torch.compile` + piecewise CUDA graphs, and enables **prefix caching** out of the box.
    - **Automatic prefix caching** hashes full KV blocks (including LoRA id and a tenant salt) to reuse shared prefixes across requests — a massive win for system prompts, agents, and RL rollouts.
    - **Speculative decoding** and **multi-LoRA** bolt onto the paged core: speculation verifies $k$ drafted tokens in one target pass (output distribution preserved), and multi-LoRA serves many adapters over one base model via batched gathered low-rank matmuls, with adapters paged like KV blocks.
    - Tune via `gpu_memory_utilization`, an honest `max_model_len`, `max_num_seqs`, and `max_num_batched_tokens`; quantizing weights and the KV cache buys concurrency, and the preemption/cache-hit metrics tell you whether you're oversubscribed.

!!! sota "State of the Art & Resources (2026)"
    vLLM has become the dominant open-source LLM serving engine, with its V1 architecture (2025) now the default: an isolated EngineCore process, unified token-budget scheduler, prefix caching on by default, and `torch.compile` + piecewise CUDA graphs — delivering up to 1.7× higher throughput than V0. Active research continues on tighter disaggregated prefill/decode, speculative decoding, and multi-LoRA efficiency.

    **Foundational work**

    - [Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (SOSP 2023)](https://arxiv.org/abs/2309.06180) — the founding vLLM paper; introduces PagedAttention and demonstrates 2–4× throughput gains over prior systems.
    - [Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models* (OSDI 2022)](https://www.usenix.org/conference/osdi22/presentation/yu) — origin of iteration-level (continuous) batching; 36× throughput improvement over FasterTransformer on GPT-3 scale.

    **Recent advances (2023–2026)**

    - [Ye et al., *FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving* (MLSys 2025)](https://arxiv.org/abs/2501.01005) — block-sparse KV-cache formats and JIT-compiled attention kernels; the attention backend vLLM V1 uses alongside FlashAttention.
    - [Chen et al., *Punica: Multi-Tenant LoRA Serving* (2023)](https://arxiv.org/abs/2310.18547) — SGMV (segmented gather matmul) kernels that batch LoRA contributions across different adapters in one pass; underpins vLLM's multi-LoRA implementation.
    - [Sheng et al., *S-LoRA: Serving Thousands of Concurrent LoRA Adapters* (2023)](https://arxiv.org/abs/2311.03285) — unified memory paging for adapter weights alongside KV cache; shows near-flat marginal cost for hundreds of adapters.
    - [Xia et al., *Unlocking Efficiency in LLM Inference: A Comprehensive Survey of Speculative Decoding* (2024)](https://arxiv.org/abs/2401.07851) — broad survey of draft-model, Medusa, EAGLE, and lookahead approaches; useful for understanding acceptance-rate trade-offs and when speculation helps.

    **Open-source & tools**

    - [vllm-project/vllm](https://github.com/vllm-project/vllm) — the vLLM engine itself (~82k stars); reference implementation of PagedAttention, continuous batching, prefix caching, multi-LoRA, and speculative decoding.
    - [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) — the FlashInfer kernel library used by vLLM V1's attention backend; supports paged KV-cache in block-sparse format with JIT compilation.

    **Go deeper**

    - [vLLM V1: A Major Upgrade to vLLM's Core Architecture (vLLM Blog, Jan 2025)](https://vllm.ai/blog/2025-01-27-v1-alpha-release) — the official design writeup for V1: EngineCore isolation, unified scheduler, torch.compile path, and what changed from V0.
    - [vLLM Official Documentation](https://docs.vllm.ai/en/stable/) — production deployment guide, tuning knobs, hardware support matrix, and the V1 migration guide.

## Further reading

- Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang, Stoica — *Efficient Memory Management for Large Language Model Serving with PagedAttention* (SOSP 2023). The founding vLLM paper.
- Yu, Jeong, Kim, Kim, Chun — *Orca: A Distributed Serving System for Transformer-Based Generative Models* (OSDI 2022). Origin of iteration-level / continuous batching.
- Dao, Fu, Ermon, Rudra, Ré — *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. The attention kernel vLLM builds on.
- Chen, Borgeaud, Irving, Lespiau, Sifre, Jumper — *Accelerating Large Language Model Decoding with Speculative Sampling*; and Leviathan, Kalman, Matias — *Fast Inference from Transformers via Speculative Decoding*.
- Chen, Ye, Zheng, et al. — *Punica: Multi-Tenant LoRA Serving*, and *S-LoRA: Serving Thousands of Concurrent LoRA Adapters*. The basis of vLLM's multi-LoRA kernels.
- The **vLLM** project repository and documentation (vllm-project/vllm), including the V1 architecture design notes.
