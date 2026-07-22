# 7.7 Prefix Caching & KV-Cache Reuse

Every token the decoder emits costs one full forward pass through the model's attention layers — but only for the *new* key and value vectors produced by that token. The key-value (KV) tensors for tokens the model has already processed are immutable: they are a deterministic function of the token IDs and model weights. That mathematical fact is the entire foundation of KV-cache reuse: if two requests share a common prefix, the KV tensors for those shared tokens need only be computed once, and every subsequent request that begins with the same prefix can skip straight to the first token that differs.

At small scale this is a pleasant micro-optimization. At production scale — where a system-prompt of 2,000 tokens is prepended to every conversation, or where a 64-shot RAG document forms the head of every retrieval request — prefix caching is a first-order economic lever. It can reduce time-to-first-token (TTFT) for subsequent requests by 80–95 % and cut total GPU compute proportionally.

This chapter develops prefix caching from first principles: the algebra of KV materialization, hash-based block identification, the two major implementations (SGLang's RadixAttention and vLLM's Automatic Prefix Caching), eviction policy, cross-request sharing semantics, and the concrete deployment patterns where shared prefixes appear at scale. We close with a worked numerical cost model and practical configuration guidance.

For the low-level anatomy of the KV cache itself and the paged memory allocator that makes per-request caching tractable, first read [PagedAttention & KV-Cache Memory Management](../04-kernels-efficiency/06-paged-attention-kv.html) and [The Anatomy of LLM Inference: Prefill, Decode & The KV Cache](../07-inference-serving/01-anatomy-inference.html).

---

## 7.7.1 Why KV Tensors Are Reusable

During the prefill phase for a sequence of tokens $x_1, x_2, \dots, x_T$, every transformer layer $\ell$ computes

$$
K^{(\ell)}_t = x_t W_K^{(\ell)}, \qquad V^{(\ell)}_t = x_t W_V^{(\ell)}
$$

where $x_t$ is the hidden state of token $t$ at layer $\ell$ and $W_K^{(\ell)}, W_V^{(\ell)}$ are learned weight matrices. Because the model is autoregressive and deterministic, $K^{(\ell)}_t$ and $V^{(\ell)}_t$ depend only on the input tokens $x_1, \dots, x_t$ and the (fixed) weights. More precisely, they depend on the exact byte sequence of the token IDs up to position $t$, concatenated with any position encoding applied at that layer.

This gives us a strong guarantee: **if two requests share the token-level prefix $x_1, \dots, x_P$ byte-for-byte, they produce identical KV tensors for all positions $1 \leq t \leq P$ in every layer.** No stochastic element exists in key or value computation (unlike, say, the generation step which may involve temperature sampling). This means the KV tensors are a pure function of the prefix and can be safely cached and reused across any number of requests.

The same logic extends to the *sequence of positions*. Most modern models use relative or rotary (RoPE) positional encodings (see [Positional Encodings: Sinusoidal, Learned, RoPE & ALiBi](../02-transformer/05-positional-encoding.html)), which embed position information into the key and query vectors. Because both the token IDs *and* the positions are deterministic from the prefix, the KV cache is still reusable as long as the prefix always occupies positions $1 \dots P$ — which is the common case for system prompts that sit at the start of a context.

!!! warning "RoPE and prefix cache correctness"
    With absolute positional encodings (the original sinusoidal scheme), the position of a token in the overall sequence is baked into the KV vectors. If a prefix is appended mid-conversation rather than placed at position 0, the cached KVs from a prior run at position 0 would be **incorrect**. Most production systems sidestep this by requiring that cached prefixes always start at position 0 — which is naturally satisfied by system-prompt reuse and few-shot headers. Be cautious with implementations that try to insert a cached block at an arbitrary offset with a non-RoPE model.

---

## 7.7.2 Block-Level Identification: Content-Based Hashing

The challenge for a serving system is identifying, at request-dispatch time, which portions of the incoming context are already cached without comparing raw tensor data. The standard approach is **content-based hashing of token-ID blocks**.

The KV cache is managed in pages (also called *blocks*) of a fixed size, typically 16 or 32 tokens. Each page stores the K and V tensors for exactly those tokens at all transformer layers. To identify a page, the server computes a hash of its content — specifically, the token IDs contained in that page concatenated with the hash of the immediately preceding page:

$$
h_0 = \text{Hash}([\text{token\_ids}[0:B]])
$$

$$
h_i = \text{Hash}(h_{i-1} \,\|\, \text{token\_ids}[i \cdot B : (i+1) \cdot B])
$$

where $B$ is the block size and $\|$ denotes concatenation. This chained hash construction ensures that two blocks with identical token IDs but different preceding context get different hashes — a block containing `"the dog"` at position 16 is semantically different from the same tokens at position 48 because the model's hidden states (and thus the KV tensors) depend on all prior context.

The serving engine maintains a **prefix hash table**: a map from block hash to a reference-counted pointer to the physical GPU memory block holding the pre-computed KV tensors. On receiving a new request, the scheduler:

1. Tokenizes the prompt and segments it into blocks of size $B$.
2. Walks the block chain (recomputing hashes incrementally from the front) and looks each hash up in the table.
3. Identifies the longest matching cached prefix — a contiguous run of cache hits from position 0.
4. Schedules prefill only for the uncached suffix.


{{fig:prefixcache-block-hash-chain}}


The engine then runs prefill only for blocks 2 and 3 (32 tokens), using the cached KV blocks for blocks 0 and 1 as "prior context" passed into the attention kernel.

### Hash collision risk

SHA-256 truncated to 64 bits provides ample collision resistance for any realistic serving workload. However, some systems (e.g., vLLM's internal implementation) use simpler 64-bit hashes for speed. A hash collision would cause a request to receive incorrect KV tensors, producing subtly wrong output. In practice, the probability of a collision in a fleet serving billions of requests is negligible, but safety-critical deployments may want to add a lightweight token-ID equality check on hash hits.

---

## 7.7.3 RadixAttention: SGLang's Tree-Based Prefix Store

SGLang (Zheng et al., "Efficiently Programming Large Language Models using SGLang," 2024) introduced **RadixAttention**, which reframes the prefix cache as a radix tree (trie) of token sequences. Rather than maintaining a flat hash table keyed on individual block hashes, RadixAttention organizes cached prefixes as a tree where:

- Each edge is labeled with a sequence of tokens.
- Each node stores the KV tensors corresponding to the edge's token segment.
- Shared prefixes correspond to common root-to-node paths.


{{fig:prefixcache-radix-tree}}


This representation has several advantages over a flat hash table:

**Automatic longest-prefix matching.** A single trie lookup finds the longest cached prefix in $O(P)$ time, where $P$ is the number of tokens in the prefix, rather than hashing individual blocks separately.

**Prefix sharing across partial matches.** Two requests sharing only the first 1,000 tokens of a 2,000-token context can still reuse the first 1,000 tokens without either having previously generated the full 2,000-token prefix. The trie naturally captures this partial overlap.

**Multi-turn conversation chains.** Each turn in a conversation appends to the previous turn's path in the trie. If a user sends messages A, B, C in one session, the path `root → A → A+B → A+B+C` is built incrementally. A second session that shares message A automatically benefits from the first session's computation.

```python
from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class RadixNode:
    """
    A node in the RadixAttention trie.

    In a real implementation, kv_ptr would be a reference to a GPU
    memory block. Here we use a placeholder bytes object to represent
    the serialised KV tensors.
    """
    token_ids: list[int]      # edge label (tokens along this edge)
    kv_data: Optional[bytes]  # cached KV tensors (None if not yet materialized)
    children: dict[int, "RadixNode"] = field(default_factory=dict)
    ref_count: int = 0        # how many active requests are using this node
    last_access_time: float = 0.0

    def is_leaf(self) -> bool:
        return len(self.children) == 0


class RadixTree:
    """
    Minimal RadixAttention trie for prefix caching.
    Supports insert, longest-prefix lookup, and LRU eviction.
    """

    def __init__(self):
        self.root = RadixNode(token_ids=[], kv_data=None)
        self._clock = 0.0

    def _tick(self) -> float:
        self._clock += 1.0
        return self._clock

    def insert(self, token_ids: list[int], kv_data: bytes) -> None:
        """
        Insert a fully prefilled sequence into the trie.
        Splits existing edges as needed (standard radix tree insertion).
        """
        node = self.root
        idx = 0  # position in token_ids

        while idx < len(token_ids):
            first_token = token_ids[idx]

            if first_token not in node.children:
                # No matching child — create a new leaf.
                new_node = RadixNode(
                    token_ids=token_ids[idx:],
                    kv_data=kv_data,  # store full KV data at leaf
                    last_access_time=self._tick(),
                )
                node.children[first_token] = new_node
                return

            child = node.children[first_token]
            # Find longest common prefix between token_ids[idx:] and child.token_ids.
            lcp = 0
            while (lcp < len(child.token_ids) and
                   idx + lcp < len(token_ids) and
                   child.token_ids[lcp] == token_ids[idx + lcp]):
                lcp += 1

            if lcp == len(child.token_ids):
                # Fully matched the existing edge — descend.
                idx += lcp
                node = child
            else:
                # Partial match — split the edge.
                # Create an intermediate node with the common prefix.
                split_node = RadixNode(
                    token_ids=child.token_ids[:lcp],
                    kv_data=None,  # split node has no KV data of its own
                    last_access_time=self._tick(),
                )
                # Old child gets the suffix as its new edge label.
                child.token_ids = child.token_ids[lcp:]
                split_node.children[child.token_ids[0]] = child
                # New leaf gets the remaining tokens.
                remainder_node = RadixNode(
                    token_ids=token_ids[idx + lcp:],
                    kv_data=kv_data,
                    last_access_time=self._tick(),
                )
                split_node.children[token_ids[idx + lcp]] = remainder_node
                # Attach split_node to parent.
                node.children[first_token] = split_node
                return

    def match_prefix(self, token_ids: list[int]) -> tuple[int, Optional[bytes]]:
        """
        Find the longest cached prefix of token_ids.
        Returns (length_matched, kv_data_of_deepest_matched_node).
        """
        node = self.root
        idx = 0
        best_len = 0
        best_kv = None

        while idx < len(token_ids):
            first_token = token_ids[idx]
            if first_token not in node.children:
                break

            child = node.children[first_token]
            child.last_access_time = self._tick()  # update LRU stamp

            lcp = 0
            while (lcp < len(child.token_ids) and
                   idx + lcp < len(token_ids) and
                   child.token_ids[lcp] == token_ids[idx + lcp]):
                lcp += 1

            idx += lcp

            if child.kv_data is not None:
                best_len = idx
                best_kv = child.kv_data

            if lcp < len(child.token_ids):
                break  # partial match — descent stops here

            node = child

        return best_len, best_kv
```

The implementation above shows the essential structure. In production (e.g., the SGLang codebase), `kv_data` is replaced by a list of physical GPU memory block IDs, and the tree nodes carry reference counts so the memory manager knows which blocks are actively being read by running requests.

---

## 7.7.4 vLLM Automatic Prefix Caching (APC)

vLLM's **Automatic Prefix Caching** (APC), introduced in vLLM 0.4, takes a simpler but highly effective approach: it reuses the existing PagedAttention block table and adds a global **evictor hash table** on top. See [vLLM: Architecture, PagedAttention & Internals](../07-inference-serving/03-vllm-internals.html) for the block allocator background.

In vLLM's model:

- Every physical block of 16 tokens has a *block hash* computed as in Section 7.7.2.
- A global `prefix_cache: dict[BlockHash, PhysicalBlock]` is maintained alongside the normal free list.
- When a request completes (or when a block is fully written), the block is inserted into `prefix_cache` keyed by its hash. The block is not freed; instead it enters an LRU evictor.
- When a new request arrives, the scheduler computes the first $k$ block hashes and checks `prefix_cache`. If a hit occurs, the corresponding physical block is mapped into the new request's block table without any GPU computation.
- The LRU evictor runs when memory pressure requires freeing blocks: the least-recently-used cached blocks are evicted first, before any block that is actively referenced by a running request.

```python
# Pseudo-code: vLLM APC block-allocation logic (simplified from vLLM source)

class APCBlockAllocator:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.free_blocks: list[int] = list(range(num_blocks))
        # Map from content hash -> physical block id
        self.prefix_cache: dict[int, int] = {}
        # LRU ordering: (timestamp, block_id) — use a sortedcontainer in practice
        self.lru_order: list[tuple[float, int]] = []
        self._time = 0

    def _compute_block_hash(self, prev_hash: int, token_ids: list[int]) -> int:
        """Chain the previous block hash with current token IDs."""
        import hashlib
        data = prev_hash.to_bytes(8, 'little') + bytes(token_ids)
        return int.from_bytes(hashlib.sha256(data).digest()[:8], 'little')

    def allocate_or_reuse(
        self,
        token_ids_blocks: list[list[int]]
    ) -> tuple[list[int], int]:
        """
        Given a list of blocks (each a list of token IDs), return:
          - block_ids: physical block IDs for the full context
          - num_cached: how many leading blocks came from cache
        """
        block_ids = []
        prev_hash = 0
        num_cached = 0

        for i, block_tokens in enumerate(token_ids_blocks):
            h = self._compute_block_hash(prev_hash, block_tokens)

            if h in self.prefix_cache:
                # Cache hit: reuse this physical block.
                phys_id = self.prefix_cache[h]
                block_ids.append(phys_id)
                num_cached += 1
                self._touch(phys_id)
            else:
                # Cache miss: allocate a fresh physical block.
                # Evict if necessary.
                if not self.free_blocks:
                    self._evict_lru()
                phys_id = self.free_blocks.pop()
                # We will compute KV for this block during prefill.
                # After prefill completes, insert into cache.
                self.prefix_cache[h] = phys_id
                self._touch(phys_id)
                block_ids.append(phys_id)

            prev_hash = h

        return block_ids, num_cached

    def _touch(self, block_id: int):
        """Update LRU timestamp for a block."""
        self._time += 1
        self.lru_order.append((self._time, block_id))

    def _evict_lru(self):
        """Remove the least-recently-used cached block."""
        self.lru_order.sort()
        while self.lru_order:
            _, block_id = self.lru_order.pop(0)
            # Remove from prefix cache if it's still there and not pinned.
            for k, v in list(self.prefix_cache.items()):
                if v == block_id:
                    del self.prefix_cache[k]
                    self.free_blocks.append(block_id)
                    return
```

### RadixAttention vs vLLM APC: a comparison

| Dimension | RadixAttention (SGLang) | Automatic Prefix Caching (vLLM) |
|-----------|-------------------------|----------------------------------|
| Data structure | Radix trie | Flat hash table + LRU list |
| Prefix matching | Trie traversal, $O(P)$ | Block-by-block hash lookup, $O(P/B)$ |
| Partial-block sharing | No (block-granularity) | No (block-granularity) |
| Multi-turn chains | First-class (tree paths) | Supported via block reuse |
| Implementation complexity | Higher | Lower |
| Eviction granularity | Node (variable length) | Block (fixed size) |
| Production maturity | SGLang ≥ 0.2 | vLLM ≥ 0.4 |

Both systems share the same fundamental guarantee: **only complete blocks that are fully written (all token IDs fixed) are eligible for caching**. A partially-filled block at the trailing edge of a prefix cannot be placed in the cache because its hash would change as more tokens are appended.

---

## 7.7.5 Cache Eviction and Memory Pressure

A prefix cache competes for the same GPU DRAM as the KV memory needed by actively running requests. This creates a fundamental tension: holding too many cached prefixes starves new requests of memory; holding too few means cache misses.

### LRU eviction

Both major systems use **Least-Recently-Used (LRU)** eviction at block granularity. The LRU policy is a good fit because:

1. System prompts and few-shot headers are re-used many times and thus have high recency at any given moment.
2. Completed responses from long-ago requests are unlikely to be needed again.

The eviction trigger is memory pressure: the allocator attempts to satisfy a new allocation from free blocks first, then from the LRU tail of the cache.

### Priority tiers

Some deployments add **priority annotations** to cached blocks:

- **Pinned**: never evicted (reserved for very high-frequency system prompts).
- **Normal**: evicted LRU.
- **Ephemeral**: evicted immediately after the owning request completes.

vLLM's `--enable-prefix-caching` flag enables APC with normal-priority eviction. SGLang exposes finer-grained control through its `CacheEngine` API.

### Cache sizing heuristics

A rough rule of thumb: allocate 15–25 % of the KV cache budget to "warm" cached blocks for a production system that uses a static system prompt. If you have a 40 GB GPU KV budget and a 2,000-token system prompt that accounts for 80 % of traffic, the warm portion for that prompt is tiny (a few MB; see the worked example below). The budget concern arises with many distinct prefixes (e.g., per-user system prompts or a large set of few-shot demonstrations), where the aggregate footprint can run into gigabytes.

!!! tip "Practitioner tip"
    Monitor the metric `prefix_cache_hit_rate` (exposed by both SGLang and vLLM via their metrics endpoints). A hit rate below 50 % on a workload with a static system prompt usually indicates a configuration error — either prefix caching is not enabled or the system prompt is being varied (e.g., injecting a timestamp into it, which defeats hashing).

---

## 7.7.6 Cross-Request Sharing and Concurrency

A subtle but important aspect of prefix caching is **concurrent access** to shared blocks. When two requests simultaneously benefit from the same cached prefix, they must not interfere with each other. The key insight is that cached KV blocks are **read-only**: once a block is finalized and inserted into the cache, no running request ever modifies it (new tokens are written into fresh, unshared blocks appended after the cached prefix). This makes concurrent reading safe without locks.

The reference count on each block tracks how many active requests are currently using it. A block with `ref_count > 0` cannot be evicted, even if it is the LRU candidate.


{{fig:prefixcache-refcount-sharing}}


This design is sometimes called **copy-on-write semantics**: the cached prefix is shared as read-only, and any divergence (the user-specific suffix) is written to fresh blocks that belong exclusively to each request.

### Cross-session versus within-session caching

Most implementations cache across sessions — two entirely separate HTTP requests from different users share blocks if they have the same prefix. Within a single multi-turn conversation, caching is even simpler because the serving framework can explicitly pin the turn history in the cache.

The **security implication** is real: if two users share the same system-prompt block in GPU memory and the KV data leaks via a side channel, information about the system prompt could be recovered. In practice, the system prompt is already visible to anyone who can query the endpoint, so this is not usually a concern. For multi-tenant deployments with confidential system prompts per tenant, consider running separate model instances or using per-tenant prefix isolation.

---

## 7.7.7 The Economics of Shared Prefixes

Prefix caching produces the largest gains precisely when it is needed most: when a large, repeated prefix dominates request cost. Let us quantify this carefully.

### Cost model

For a request with prefix length $P$ (tokens) and new suffix length $S$ (tokens):

- Without caching: prefill cost $\propto (P + S)^2 / 2$ attention operations (rough $O(N^2)$ scaling) plus $P + S$ weight projections.
- With a warm cache hit on the full prefix: prefill cost $\propto S \cdot P + S^2/2$ (the suffix attends to the cached prefix) plus $S$ weight projections.

The ratio of cached to uncached prefill work is approximately

$$
\text{speedup} \approx \frac{P + S}{S + P \cdot \frac{S}{P+S}} \approx \frac{P + S}{S}\quad \text{for large } P/S
$$

For $P = 2000$, $S = 50$ (a 2,000-token system prompt with a short user query): speedup $\approx 41\times$ in prefill attention FLOPs. Real TTFT improvements are smaller due to weight-loading overhead that is shared regardless of cache state, but 5–15× TTFT improvements are commonly observed in practice.


{{fig:prefixcache-attention-cost-geometry}}

!!! example "Worked numerical example: system-prompt caching for a chatbot"
    **Setup.** A production chatbot uses:
    - Model: LLaMA-3 70B, 80 layers, 8192-dimensional hidden state, GQA with 8 KV heads, head dim 128, dtype bfloat16.
    - System prompt: 1,024 tokens.
    - Average user query: 64 tokens.
    - Serving load: 200 requests/second, 95 % share the identical system prompt.

    **KV cache size for the system prompt (per request without caching).**

    At each layer, K and V tensors for one token:
    $$
    \text{bytes per token per layer} = 2 \times n_{\text{kv\_heads}} \times d_{\text{head}} \times 2 = 2 \times 8 \times 128 \times 2 = 4096 \text{ bytes}
    $$

    For 1,024 tokens across 80 layers:
    $$
    \text{KV size} = 1024 \times 80 \times 4096 \approx 335 \text{ MB}
    $$

    Without caching, 200 req/s × 335 MB = 67 GB of KV data produced per second just for the system prompt — all of it redundant.

    **With prefix caching.** The 1,024-token system-prompt KV tensors are computed once and stored. They occupy 335 MB of GPU memory (one copy, shared across all concurrent requests). Each arriving request skips the system-prompt prefill entirely, saving:
    $$
    \text{saved prefill tokens per second} = 200 \times 0.95 \times 1024 \approx 194{,}880 \text{ tokens/sec}
    $$

    At an LLaMA-3 70B prefill throughput of roughly 5,000–15,000 tokens/sec per A100 (depending on batch size), those saved tokens correspond to freeing up one or more GPUs worth of compute that can be redeployed to serve more requests.

    **TTFT impact.** With a cold cache, prefilling 1,024 + 64 = 1,088 tokens takes on the order of 100–300 ms (depending on batch load). With a warm cache, only 64 tokens are prefilled, cutting TTFT to the order of 10–30 ms — an order-of-magnitude improvement in interactive latency.

### Where shared prefixes arise in practice

| Use case | Shared prefix content | Typical prefix length |
|----------|-----------------------|----------------------|
| Chatbot with system prompt | Role description, safety instructions | 200–4,000 tokens |
| Few-shot examples | 4–64 labeled examples | 1,000–30,000 tokens |
| RAG document header | Retrieved passage + instructions | 500–8,000 tokens |
| Code completion with repo context | File contents, imports | 2,000–50,000 tokens |
| Multi-agent tool definitions | Tool schemas + examples | 1,000–10,000 tokens |
| Batch classification | Rubric + calibration examples | 500–5,000 tokens |

The agent use case (see [The Agentic Loop: ReAct, Plan-Execute & Reflection](../08-agents-harness/02-agentic-loop.html)) deserves particular attention. In a ReAct loop, each new turn prepends the entire conversation history. After $k$ turns with average turn length $L$, the prefix is $k \cdot L$ tokens. Without prefix caching, total prefill cost scales as $O(k^2 L)$; with caching and a warm trie, each new turn costs only $O(L)$ for the suffix, so total cost scales as $O(kL)$ — a qualitative improvement in long-horizon agent tasks.

{{fig:prefixcache-agent-turn-growth}}

---

## 7.7.8 Enabling and Tuning Prefix Caching in Production

### vLLM configuration

```bash
# Enable APC in vLLM (the flag is per-engine startup)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-70b-instruct \
    --enable-prefix-caching \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --block-size 16
```

The `--block-size` parameter controls the hashing granularity. Larger blocks mean fewer hash lookups but coarser prefix alignment — a prompt that is one token short of a full block boundary gets no cache benefit for that trailing block.

```python
# Verify APC is active via vLLM metrics endpoint
import requests

stats = requests.get("http://localhost:8000/metrics").text
# Look for: vllm:gpu_prefix_cache_hit_rate
# and:      vllm:gpu_cache_usage_perc

for line in stats.splitlines():
    if "prefix_cache" in line or "cache_usage" in line:
        print(line)
```

### SGLang configuration

```bash
# SGLang enables RadixAttention by default; disable with --disable-radix-cache
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3-70b-instruct \
    --tp 4 \
    --context-length 32768 \
    --mem-fraction-static 0.85
# Metrics available at http://localhost:30000/get_server_info
```

### Maximising hit rate: best practices

```python
# Example: structuring prompts for maximum cache reuse
# The STABLE part of the prompt must come FIRST.

SYSTEM_PROMPT = """
You are a senior software engineer reviewing code for correctness,
performance, and security. For each issue found:
1. Describe the issue clearly.
2. Explain the impact.
3. Suggest a fix with example code.
""".strip()

def build_prompt(user_code: str) -> str:
    """
    Always place the system prompt at position 0.
    The user-specific code comes last, ensuring the shared prefix
    can be cached and reused across all requests.
    """
    return f"{SYSTEM_PROMPT}\n\n---\n\nCode to review:\n```python\n{user_code}\n```"

# ANTI-PATTERN: injecting dynamic content into the shared prefix
def build_prompt_bad(user_code: str, timestamp: str) -> str:
    """
    DO NOT embed timestamps or request IDs into the system prompt.
    This makes every prompt unique and destroys the shared prefix.
    """
    return (
        f"Current time: {timestamp}\n"   # ← defeats prefix caching
        f"{SYSTEM_PROMPT}\n\n"
        f"Code:\n{user_code}"
    )
```

**Alignment to block boundaries.** Because the cache operates on full blocks, prompts that are aligned to multiples of `block_size` tokens get the best coverage. If your system prompt is 1,020 tokens and `block_size=16`, the last partial block (4 tokens) will never be cached. Pad the system prompt to 1,024 tokens to get full coverage.

```python
def pad_to_block_boundary(token_ids: list[int],
                           block_size: int,
                           pad_token_id: int = 0) -> list[int]:
    """
    Pad a prefix to the next block boundary so that no tokens are
    left in an uncacheable partial block.
    """
    remainder = len(token_ids) % block_size
    if remainder == 0:
        return token_ids
    padding = block_size - remainder
    return token_ids + [pad_token_id] * padding
```

### Interaction with chunked prefill

When chunked prefill is enabled (see [Disaggregated Prefill/Decode & Chunked Prefill](../07-inference-serving/08-disaggregated-chunked-prefill.html)), a long prompt is split into multiple "chunks" and processed across several scheduler iterations. Prefix caching interacts cleanly: cached blocks are consumed in the first chunk without any compute; only uncached blocks are fed into the chunked prefill pipeline.

### Interaction with speculative decoding

Speculative decoding (see [Speculative Decoding: Draft Models, Medusa, EAGLE & Lookahead](../07-inference-serving/06-speculative-decoding.html)) accelerates the decode phase, not prefill. Prefix caching accelerates prefill. They are largely orthogonal and can be combined: enable both for workloads that have a large shared prefix followed by a multi-token generation step.

---

## 7.7.9 Advanced Topics and Open Problems

### Disk-level prefix caching

For extremely long, rarely-changing prefixes — such as entire codebases embedded as context (the "repository-level code completion" use case) — it is possible to store pre-computed KV tensors on NVMe SSDs and load them into GPU VRAM on demand. This is sometimes called **persistent KV caching** or **offline prefix materialisation**. The GPU-NVMe bandwidth of modern PCIe 5.0 systems (on the order of 12–14 GB/s) is fast enough to load a 10,000-token prefix in well under a second.

The engineering challenge is cache invalidation: if the model weights change (e.g., after a fine-tuning update), all persistent KV caches become stale and must be regenerated. Some serving systems track a model-version hash alongside each on-disk KV file.

### Multi-model and cross-layer caching

All the caching described so far applies to a single model instance. In a serving cluster running multiple replicas of the same model, caching can be extended across replicas: one node holds the "canonical" KV cache for a popular system prompt, and other nodes pull the cache over the network (RDMA or NVLink) rather than recomputing it. This is sometimes called **distributed prefix caching** or **remote KV caching**. The bandwidth requirement is on the order of seconds for a typical system-prompt KV block, so this pattern is practical only for very-high-value long prefixes.

### Quantized KV caches

Storing KV caches in INT8 or FP8 reduces their memory footprint by 2–4× (see [Quantization I: Post-Training Quantization (GPTQ, AWQ, SmoothQuant)](../04-kernels-efficiency/07-quantization-ptq.html)), which directly increases the number of distinct prefixes that can be held warm simultaneously. The tradeoff is a small accuracy loss in the attention computation for the cached tokens. Both SGLang and vLLM support quantized KV caches via the `--kv-cache-dtype` flag.

### Semantic prefix caching

Content-based hashing is exact-match only: two prompts that differ by a single space produce different hashes even if the resulting KV tensors are nearly identical. Research directions include **semantic prefix caching**, where nearby prompts (in embedding space) can share cache entries via approximate nearest-neighbour lookup. This is an active research area as of 2025–2026 and not yet deployed in mainstream serving stacks.

---

!!! interview "Interview Corner"
    **Q:** A candidate is asked: "Your LLM serving system has a 2,000-token system prompt prepended to every request. How would you reduce the latency and compute cost for repeated requests, and what are the key engineering concerns?"

    **A:** The canonical solution is prefix caching (also called KV cache reuse). Because KV tensors are a deterministic function of the token IDs and model weights, the 2,000-token system prompt produces identical KV blocks for every request. We enable Automatic Prefix Caching (vLLM) or RadixAttention (SGLang), which stores these blocks in a hash-indexed GPU memory pool and skips their recomputation on cache hits. Engineering concerns:

    1. **Hash chaining**: blocks are keyed by a chain hash so that identical token sequences at different positions don't collide.
    2. **Block alignment**: the system prompt should be padded to a multiple of the block size (e.g., 16 tokens) so no tokens fall in an uncacheable partial block.
    3. **Prompt stability**: dynamic content (timestamps, user IDs) must not be injected into the shared prefix — it should go *after* the shared portion.
    4. **Eviction policy**: the cache shares GPU DRAM with the live KV budget; a pinned entry for the most frequent system prompt prevents LRU eviction of high-value data.
    5. **Reference counting**: cached blocks must be reference-counted so they are not freed while a concurrent request is still reading them.

    In production, this commonly reduces TTFT from ~200 ms to ~20 ms for short user queries and cuts prefill GPU utilisation by a factor of $P/S$ where $P$ is the prefix length and $S$ is the average suffix length.

---

!!! key "Key Takeaways"
    - KV tensors are a pure function of token IDs and model weights, making any shared prefix's KV blocks exactly reusable across requests.
    - Content-based hashing of fixed-size blocks enables O(1) cache lookup per block; chaining hashes ensures positional correctness.
    - SGLang's RadixAttention models the prefix cache as a trie, naturally capturing multi-turn chains and partial prefix overlap. vLLM's APC takes a simpler flat-hash approach layered on top of the PagedAttention allocator.
    - LRU eviction balances warm-cache hit rate against memory pressure; reference counting prevents eviction of blocks in active use.
    - Prefix caching is most valuable for long, stable shared prefixes: system prompts, few-shot headers, RAG documents, and multi-turn agent histories.
    - For maximum hit rate: place shared content at the front of every prompt, avoid injecting dynamic data into shared prefixes, and pad system prompts to block-size boundaries.
    - Prefix caching (prefill savings) and speculative decoding (decode acceleration) are largely orthogonal and can be combined.
    - Quantised KV caches multiply the effective warm-cache capacity by 2–4× at a small accuracy cost.

---

!!! sota "State of the Art & Resources (2026)"
    Prefix caching and KV-cache reuse have become standard infrastructure in production LLM serving as of 2024–2026: vLLM's Automatic Prefix Caching and SGLang's RadixAttention ship on by default, while newer work extends reuse across disaggregated serving nodes, compresses cached tensors for storage efficiency, and explores semantic (approximate) matching beyond exact-hash lookup.

    **Foundational work**

    - [Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (2023)](https://arxiv.org/abs/2309.06180) — the block-paged KV allocator that makes per-request and cross-request cache management tractable (SOSP 2023).
    - [Gim et al., *Prompt Cache: Modular Attention Reuse for Low-Latency Inference* (2023)](https://arxiv.org/abs/2311.04934) — formalises reusable "prompt modules" and attention-state sharing across requests, 8–60× latency gains (MLSys 2024).

    **Recent advances (2023–2026)**

    - [Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs* (2023)](https://arxiv.org/abs/2312.07104) — introduces RadixAttention, a trie-based prefix cache enabling automatic KV reuse across multi-turn and multi-call workloads.
    - [Ye et al., *ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition* (2024)](https://arxiv.org/abs/2402.15220) — cross-instance prefix sharing with a two-phase attention kernel; 1.6–2.3× throughput over vLLM on shared-prefix workloads (ACL 2024).
    - [Liu et al., *CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving* (2023)](https://arxiv.org/abs/2310.07240) — custom encoder compresses KV cache 3.5–4.3× for network-efficient cross-node reuse (SIGCOMM 2024).
    - [Liu et al., *LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference* (2025)](https://arxiv.org/abs/2510.09665) — production system that stores and shares KV caches across GPU, CPU, disk, and S3; 3–10× TTFT reduction in multi-round and RAG workloads.

    **Open-source & tools**

    - [vllm-project/vllm](https://github.com/vllm-project/vllm) — production LLM serving engine; enable Automatic Prefix Caching with `--enable-prefix-caching`.
    - [sgl-project/sglang](https://github.com/sgl-project/sglang) — high-performance serving framework with RadixAttention on by default; disable with `--disable-radix-cache`.
    - [LMCache/LMCache](https://github.com/LMCache/LMCache) — drop-in vLLM/SGLang extension for multi-tier distributed KV cache offloading and peer-to-peer sharing.

    **Go deeper**

    - [Fast and Expressive LLM Inference with RadixAttention and SGLang](https://www.lmsys.org/blog/2024-01-17-sglang/) — LMSYS blog post with benchmarks and design rationale for the radix-tree cache.
    - [Automatic Prefix Caching — vLLM official docs](https://docs.vllm.ai/en/stable/design/prefix_caching/) — implementation details, hash-chaining design, and configuration guide for vLLM's APC.

## Further Reading

- Zheng, L. et al. "Efficiently Programming Large Language Models using SGLang." *arXiv:2312.07104*, 2023. — Introduces RadixAttention and the SGLang serving framework.
- Kwon, W. et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." *SOSP 2023*. — The PagedAttention paper that underpins block-level KV management in vLLM.
- Pope, R. et al. "Efficiently Scaling Transformer Inference." *MLSys 2023*. — Analysis of attention arithmetic, prefill/decode separation, and the case for batched serving.
- vLLM blog: "Automatic Prefix Caching in vLLM." Official vLLM documentation and blog posts at the [vLLM GitHub repository](https://github.com/vllm-project/vllm) — release notes for v0.4 describe APC implementation details.
- Dao, T. et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *NeurIPS 2022*. — The kernel that makes prefix-aware attention fast enough to matter; see also [FlashAttention I: IO-Awareness & The Online Softmax](../04-kernels-efficiency/02-flash-attention-1.html).
- Gim, I. et al. "Prompt Cache: Modular Attention Reuse for Low-Latency Inference." *MLSys 2024*. — Formalises the idea of caching attention states for reusable prompt modules.

---

## Exercises

**1.** *(Conceptual.)* The chapter states that KV tensors are "a pure function of the token IDs and model weights." Use that fact to answer two questions. (a) A well-meaning engineer prepends `f"Current time: {timestamp}\n"` to the front of a shared system prompt so the model "knows the current time." The `prefix_cache_hit_rate` metric collapses to near zero. Explain precisely why, referring to the chained block-hash construction of Section 7.7.2. (b) The engineer moves the timestamp to *after* the system prompt instead. Explain why this restores the hit rate for the system-prompt blocks, and state exactly which blocks now hit and which still miss.

??? note "Solution"
    **(a)** The block hash is chained: $h_0 = \text{Hash}(\text{token\_ids}[0{:}B])$ and $h_i = \text{Hash}(h_{i-1} \,\|\, \text{token\_ids}[i\cdot B:(i+1)\cdot B])$. The very first block now contains the timestamp tokens, which change on every request (a new minute, second, or millisecond produces different token IDs). Therefore $h_0$ is different for every request. Because every subsequent hash depends on $h_{i-1}$, a different $h_0$ poisons the entire chain: $h_1, h_2, \dots$ all differ too, even though the system-prompt tokens after the timestamp are byte-for-byte identical. Every block is a cache miss, so the hit rate collapses to zero.

    **(b)** With the timestamp moved to the end, the leading blocks contain only the (stable) system-prompt token IDs. Their chained hashes $h_0, h_1, \dots$ are identical across requests, so all of the *complete* system-prompt blocks hit the cache. The blocks that still miss are: (i) any block that mixes the tail of the system prompt with the start of the varying timestamp/user content, and (ii) every block at or after the timestamp, since those token IDs vary per request. This is exactly the chapter's guidance in Section 7.7.8: put the stable content first and the dynamic content last.

**2.** *(Quantitative.)* A serving engine uses `block_size = 16`. A system prompt tokenizes to **1,020 tokens**. (a) How many *complete*, cacheable blocks does the system prompt occupy, and how many tokens are stranded in an uncacheable partial block? (b) Using the chapter's `pad_to_block_boundary` helper, to what length is the prompt padded, how many pad tokens are added, and how many cacheable blocks result? (c) Briefly: what is the downside of padding, and why is it usually worth it here?

??? note "Solution"
    **(a)** Complete blocks $= \lfloor 1020 / 16 \rfloor = 63$ blocks, covering $63 \times 16 = 1008$ tokens. Remainder $= 1020 \bmod 16 = 12$ tokens. Those 12 tokens sit in a partial 64th block that is *not yet full*, so its hash is not stable and it cannot be cached. **12 tokens are stranded.**

    **(b)** `remainder = 1020 % 16 = 12`, which is nonzero, so `padding = 16 - 12 = 4`. The prompt is padded to $1020 + 4 = 1024$ tokens. Now $1024 / 16 = 64$ complete blocks, **all cacheable** — the previously stranded 12 tokens plus 4 pad tokens now form a full, hashable 64th block.

    **(c)** The downside is that the 4 pad tokens add a small amount of prefill compute and KV memory the first time the prefix is materialized (and slightly change the attention over the prompt if the pad token is not masked). But it is a one-time cost paid once and then reused on every hit; in exchange the trailing 12 real tokens of the system prompt become cacheable on every subsequent request, so for a high-traffic static prompt the padding pays for itself almost immediately.

**3.** *(Quantitative.)* A model has **80 layers**, uses GQA with **8 KV heads**, head dimension **128**, and stores the KV cache in **bfloat16** (2 bytes). A shared prefix is **512 tokens** long. (a) Compute the bytes of KV per token per layer, then the total KV footprint of the cached prefix. (b) The endpoint receives **300 requests/second**, all sharing this prefix. Without prefix caching, how many tokens of prefix prefill are recomputed per second, and how many *bytes* of redundant KV are produced per second? (c) With caching, how many copies of the prefix KV are stored in GPU memory, regardless of concurrency?

??? note "Solution"
    **(a)** Following Section 7.7.7's formula, KV per token per layer stores both K and V:
    $$
    2 \times n_{\text{kv\_heads}} \times d_{\text{head}} \times 2\text{ bytes} = 2 \times 8 \times 128 \times 2 = 4096 \text{ bytes}.
    $$
    Total footprint for 512 tokens over 80 layers:
    $$
    512 \times 80 \times 4096 = 167{,}772{,}160 \text{ bytes} = 160 \text{ MB}.
    $$

    **(b)** Redundant prefill tokens per second $= 300 \times 512 = 153{,}600$ tokens/sec. Redundant KV bytes per second $= 300 \times 160\text{ MB} = 48{,}000\text{ MB} \approx 46.9 \text{ GB/sec}$ — all of it identical and wasted.

    **(c)** Exactly **one** copy. The cached KV blocks are read-only and shared across all concurrent requests (Section 7.7.6); a reference count tracks the active readers, but only a single 160 MB physical copy is stored no matter how many requests are in flight.

**4.** *(Quantitative.)* The chapter models uncached prefill attention cost as $\propto (P+S)^2/2$ and cached prefill cost as $\propto S\cdot P + S^2/2$ (the suffix attends to the cached prefix). Take $P = 1500$ (prefix) and $S = 100$ (suffix). (a) Compute the exact attention-op speedup ratio. (b) Compute the chapter's large-$P/S$ approximation $(P+S)/S$ and explain why it overestimates the true ratio here. (c) The chapter says real TTFT gains are typically 5-15x rather than the raw FLOP ratio. Give one reason from the chapter.

??? note "Solution"
    **(a)** Uncached cost $\propto (P+S)^2/2 = 1600^2/2 = 2{,}560{,}000/2 = 1{,}280{,}000$. Cached cost $\propto S\cdot P + S^2/2 = 1500\times100 + 100^2/2 = 150{,}000 + 5{,}000 = 155{,}000$. Exact ratio:
    $$
    \frac{1{,}280{,}000}{155{,}000} \approx 8.26\times.
    $$

    **(b)** The approximation gives $(P+S)/S = 1600/100 = 16\times$, roughly double the true value. It overestimates because it drops the $S\cdot P$ term in the denominator — but with $S=100$ that term ($150{,}000$) actually dominates the cached cost. The approximation is only tight when $S \ll P$ *and* $S\cdot P$ becomes negligible relative to $(P+S)^2/2$; here $S/P = 1/15$ is not small enough, so the suffix's attention over the long cached prefix is a substantial fraction of the remaining work.

    **(c)** Prefill also involves weight-projection / weight-loading overhead that is incurred regardless of cache state (the chapter notes "weight-loading overhead that is shared regardless of cache state"). That fixed component is not eliminated by caching, so the observed TTFT speedup is smaller than the pure attention-FLOP ratio.

**5.** *(Implementation.)* Section 7.7.4 shows vLLM-style APC using a flat `prefix_cache: dict[hash, block]` with a chained block hash. Implement a standalone function

    longest_cached_prefix(token_ids, block_size, prefix_cache) -> int

that returns the number of *leading tokens* already cached, by segmenting `token_ids` into `block_size` blocks, chaining hashes exactly as `_compute_block_hash` in the chapter does (seed `prev_hash = 0`), and stopping at the **first block miss** (a cache hit after a miss must not count — the prefix must be contiguous from position 0, per Section 7.7.2 step 3). Only complete blocks are eligible; a trailing partial block never counts. Include a short test.

??? note "Solution"
    The key subtleties: (i) reproduce the chapter's chained hash so lookups match how blocks were inserted; (ii) break on the first miss so the result is a contiguous prefix from position 0; (iii) skip the trailing partial block since only fully-written blocks are cacheable (Section 7.7.4).

    ```python
    import hashlib

    def _compute_block_hash(prev_hash: int, token_ids: list[int]) -> int:
        """Chain the previous block hash with current token IDs (as in the chapter)."""
        data = prev_hash.to_bytes(8, "little") + bytes(token_ids)
        return int.from_bytes(hashlib.sha256(data).digest()[:8], "little")

    def longest_cached_prefix(
        token_ids: list[int],
        block_size: int,
        prefix_cache: dict[int, int],
    ) -> int:
        """
        Return the number of leading tokens whose (complete) blocks are all
        present in prefix_cache, walking from position 0 and stopping at the
        first miss. A trailing partial block is never counted.
        """
        num_full_blocks = len(token_ids) // block_size  # ignore partial tail
        prev_hash = 0
        cached_tokens = 0
        for i in range(num_full_blocks):
            block = token_ids[i * block_size:(i + 1) * block_size]
            h = _compute_block_hash(prev_hash, block)
            if h not in prefix_cache:
                break                      # contiguity: stop at first miss
            cached_tokens += block_size
            prev_hash = h                  # advance the chain only on a hit
        return cached_tokens

    # --- test ---
    if __name__ == "__main__":
        B = 4
        # Warm the cache with a 12-token prefix (3 full blocks).
        warm = [10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33]
        cache: dict[int, int] = {}
        ph = 0
        for i in range(len(warm) // B):
            ph = _compute_block_hash(ph, warm[i * B:(i + 1) * B])
            cache[ph] = i  # value = physical block id (arbitrary here)

        # Request sharing the first 2 blocks (8 tokens), then diverging,
        # plus a trailing partial block that must not count.
        req = [10, 11, 12, 13, 20, 21, 22, 23, 99, 98, 97, 96, 5, 5]
        assert longest_cached_prefix(req, B, cache) == 8

        # A request whose block 0 differs gets zero, even if a later
        # block would independently hash into the cache (non-contiguous).
        req2 = [0, 0, 0, 0, 30, 31, 32, 33]
        assert longest_cached_prefix(req2, B, cache) == 0

        # Exact full-prefix hit.
        assert longest_cached_prefix(warm, B, cache) == 12
        print("all tests passed")
    ```

    Note in the second assertion that block `[30,31,32,33]` was inserted into the cache while warming — but only under the chained hash that assumes blocks `[10..]` and `[20..]` preceded it. In `req2` the chain seed differs, so that block's recomputed hash misses, and even if it hit, the `break` on block 0's miss guarantees the returned prefix stays contiguous from position 0.
