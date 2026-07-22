# 7.4 SGLang: RadixAttention & Structured Programs

In [vLLM: Architecture, PagedAttention & Internals](../07-inference-serving/03-vllm-internals.html) we saw how paging the KV cache into fixed-size blocks turns memory fragmentation into a non-problem and lets an engine pack many requests onto one GPU. PagedAttention answers the question *"how do I store the KV cache without wasting memory?"* SGLang asks a sharper, complementary question: *"why am I recomputing the same prefixes over and over, and why is my Python scheduler the bottleneck?"*

SGLang (Structured Generation Language) began as a research system from the same broad lineage as vLLM and grew into one of the two dominant open-source inference engines of 2024–2026. It contributes two big ideas that this chapter is about:

1. **RadixAttention** — a runtime that keeps a *radix tree of token prefixes* and automatically reuses any KV cache that has already been computed for a shared prefix, across requests, with no user annotation. It is prefix caching turned from a hand-managed feature into an always-on property of the scheduler.
2. **The frontend language** — `gen`, `select`, `fork`, `join`, and friends, a small embedded DSL (domain-specific language) in Python that lets you express multi-call LLM programs (branching, parallel sampling, tool calls, agents) so the runtime can *see the structure* and exploit prefix sharing, parallelism, and constrained decoding.

Underneath sits a high-throughput **runtime** with a zero-overhead scheduler, a fast constrained-decoding stack, and the same continuous-batching machinery from [Continuous Batching & Request Scheduling](../07-inference-serving/02-continuous-batching.html). You have SGLang checked out locally, so throughout this chapter we will be concrete: real module paths, real class names, real CLI flags.

By the end you should be able to (a) explain RadixAttention's data structure and eviction policy at the level of a whiteboard implementation, (b) write a structured SGLang program that branches and recombines, (c) reason about when SGLang beats vLLM and when it does not, and (d) answer the interview question "how would you cache KV across requests?" without hand-waving.

## Why RadixAttention exists: the shared-prefix problem

Recall the two phases of inference from [The Anatomy of LLM Inference](../07-inference-serving/01-anatomy-inference.html). **Prefill** runs the prompt through the model once, producing one KV vector pair per layer per token; this is compute-bound and $O(L)$ in prompt length $L$. **Decode** then generates tokens one at a time, each step reading the entire KV cache; this is memory-bandwidth-bound.

The crucial observation: a huge fraction of real-world prompts *share prefixes*.

- **System prompts.** Every request to a chat assistant begins with the same multi-hundred-token system prompt. If you serve 1,000 requests/second, you re-prefill that identical block 1,000 times per second.
- **Few-shot prompts.** A classification service sends the same 8 in-context examples (maybe 2,000 tokens) before each new query.
- **Agents and tree search.** A reasoning agent forks $k$ candidate continuations from a common context (self-consistency, beam-like search, branch-and-evaluate). All $k$ branches share the entire context up to the fork point.
- **Multi-turn chat.** Turn $t+1$ is exactly turn $t$'s context plus a new user message plus the model's previous reply. The first $N$ tokens are byte-identical.

Naively, prefill recomputes the KV cache for every shared token. RadixAttention's promise is: **compute the KV for a token sequence at most once, then reuse it for any request whose prompt starts with that sequence**, automatically, as long as it is still in GPU memory.

This is the same idea as [Prefix Caching & KV-Cache Reuse](../07-inference-serving/07-prefix-caching.html), but SGLang's contribution is the *data structure* (a radix tree) and the *eviction policy* (LRU over the tree) that make it general — it works across arbitrarily overlapping prefixes from unrelated requests, not just a single pinned system prompt.

### A back-of-the-envelope motivation

Take a 70B-parameter model with grouped-query attention (GQA): 80 layers, 8 KV heads, head dimension 128, in bf16 (2 bytes). The KV cache cost *per token* is

$$
\text{bytes/token} = 2 \times n_{\text{layers}} \times n_{\text{kv heads}} \times d_{\text{head}} \times \text{bytes} = 2 \times 80 \times 8 \times 128 \times 2 = 327{,}680 \text{ bytes} \approx 320\ \text{KiB}.
$$

A 2,000-token shared system prompt therefore costs about $2000 \times 320\ \text{KiB} \approx 625\ \text{MiB}$ of KV cache — and roughly $2000 \times 2 \times P$ FLOPs of prefill compute per request, where $P$ is the parameter count. Recomputing that for 1,000 requests is $1000 \times$ wasted prefill. RadixAttention pays it **once** and serves the other 999 from cache. We will put exact numbers on this in the worked example below.

## The radix tree: data structure and operations

{{fig:radix-tree}}

A **trie** (prefix tree) stores strings by sharing common prefixes: each edge is one character, each path from root to a node spells a stored string. A **radix tree** (a.k.a. PATRICIA trie, compressed trie) is a trie where every chain of single-child nodes is collapsed into one node whose edge holds a *whole substring*. This compression is what makes it practical: a 2,000-token system prompt is one edge, not 2,000 nodes.

SGLang adapts this to KV caching with a key twist: **the "characters" are token IDs, and each node owns the KV-cache slots for its edge's tokens.** Reuse the prefix → reuse those KV slots.

```text
                       (root)
                         |
              "You are a helpful assistant."   <- shared system prompt (one edge)
                         |  [KV slots for these tokens live here]
              +----------+-----------+
              |                      |
   "Translate to French:"   "Summarize:"        <- two task templates branch
              |                      |
   "Hello world"            "The quick brown..." <- distinct user inputs
```

Each path from root to a node spells a token sequence; the node holds (a pointer to) the KV-cache entries for the tokens on the edge leading into it. When a new request arrives, we **match its prompt against the tree** to find the longest cached prefix, reuse that KV, and only prefill the *uncached suffix*.

### What a node actually holds

In SGLang's source (`python/sglang/srt/mem_cache/radix_cache.py`), the node is `TreeNode`. Stripped to essentials, its real fields are:

```python
class TreeNode:
    def __init__(self):
        self.children = defaultdict(TreeNode)   # child_key (first token / page) -> child
        self.parent = None
        self.key = None        # RadixKey: the token-id sequence on the edge INTO this node
        self.value = None      # torch.Tensor of KV-cache slot indices for those tokens
        self.lock_ref = 0      # >0  => pinned, cannot be evicted (a running req needs it)
        self.last_access_time = time.monotonic()  # for LRU eviction
        self.hit_count = 0
```

Two fields carry the whole design:

- **`value`** is *not* the KV tensors themselves; it is a tensor of **indices** into the global paged KV pool (the `token_to_kv_pool_allocator`). The radix tree is a thin index structure layered on top of the same paged memory you met in [PagedAttention & KV-Cache Memory Management](../04-kernels-efficiency/06-paged-attention-kv.html). This separation is what lets one KV block be referenced by many tree nodes.
- **`lock_ref`** is a reference count. While a request is actively using a node's KV (during its prefill or decode), the node is *locked* (`inc_lock_ref`) so eviction cannot reclaim it out from under a running kernel. When the request finishes, `dec_lock_ref` unlocks it, and the node becomes *evictable* — its KV lingers in the cache for future reuse until memory pressure forces it out.

### Matching a prefix

`match_prefix` walks down from the root, consuming as many tokens of the incoming key as it can. The core loop (`_match_prefix_helper`) is short and worth reading:

```python
def _match_prefix_helper(self, node, key):
    # key is the incoming request's token-id sequence (a RadixKey)
    child_key = key.child_key(self.page_size)   # hash of the first page of `key`
    value = []                                   # collected KV-slot index tensors

    while len(key) > 0 and child_key in node.children:
        child = node.children[child_key]
        # how many leading tokens of `key` agree with this edge's tokens?
        prefix_len = child.key.match(key, page_size=self.page_size)

        if prefix_len < len(child.key):
            # partial match: the edge agrees for `prefix_len` tokens then diverges.
            # SPLIT the edge so the shared part becomes its own node.
            new_node = self._split_node(child.key, child, prefix_len)
            value.append(new_node.value)
            node = new_node
            break
        else:
            # full edge matched: take its whole KV, descend, continue with the rest.
            value.append(child.value)
            node = child
            key = key[prefix_len:]
            if len(key):
                child_key = key.child_key(self.page_size)

    return value, node     # concatenated `value` = cached KV indices; `node` = match point
```

The returned `value` tensors, concatenated, are exactly the KV-cache slots the new request can **reuse without recomputation**. The request then only prefills `key` from the match point onward.

### Splitting and inserting

The interesting case is a **partial** match. Suppose the tree has one edge `"You are a helpful assistant. Be concise."` and a new request shares only `"You are a helpful assistant. "`. We must *split* the edge so the shared part is reusable:

```text
before:  (root) --"You are a helpful assistant. Be concise."--> (A)

after:   (root) --"You are a helpful assistant. "--> (S)
                                                       |
                                  --"Be concise."--> (A, original KV preserved)
```

{{fig:radix-match-split-insert}}

`_split_node` creates the new intermediate node `S`, gives it the shared KV slots (`child.value[:split_len]`), re-parents the old node `A` under it with the remaining slots (`child.value[split_len:]`), and rewires children. No KV is recomputed — we only re-slice index tensors. The new request then attaches its divergent suffix as a fresh child of `S`. **Insertion is just matching plus, if a suffix is left over, allocating new KV and adding a child.**

A subtlety: matching and splitting happen at **page granularity** (`page_size`, default 1 token, but commonly a small power of two). Pages are the same allocation unit as in PagedAttention; the radix key uses `child_key(page_size)` so two requests share a node only when they agree on whole pages. With `page_size > 1`, a divergence mid-page cannot be shared — a deliberate trade of a little reuse for cheaper, block-aligned bookkeeping.

### Eviction: LRU over leaves

GPU memory is finite, so the tree must shrink under pressure. SGLang evicts with an **LRU (least-recently-used) policy restricted to evictable leaves**. Because `TreeNode.__lt__` compares `last_access_time`, the eviction routine heapifies the leaves and pops the oldest first:

```python
def evict(self, num_tokens):
    leaves = self._collect_leaves()
    heapq.heapify(leaves)                 # min-heap by last_access_time (oldest first)
    num_evicted = 0
    while num_evicted < num_tokens and leaves:
        node = heapq.heappop(leaves)
        if node.lock_ref > 0:             # pinned by a running request -> skip
            continue
        # free this node's KV slots back to the paged pool
        self.token_to_kv_pool_allocator.free(node.value)
        num_evicted += len(node.value)
        self._delete_leaf(node)
        # if the parent just became a childless, unlocked leaf, it is now evictable
        if node.parent.children == {} and node.parent.lock_ref == 0:
            heapq.heappush(leaves, node.parent)
```

Three properties make this correct and effective:

- **Leaf-only eviction.** You can never evict an interior node while its children live, because evicting a prefix would orphan the suffixes that depend on it. By construction, evicting leaves peels the tree from its tips inward — exactly the cold prefixes.
- **Lock-awareness.** `lock_ref > 0` nodes are skipped, so you never reclaim KV a running kernel is reading.
- **Cache-aware order.** Recently used prefixes (your hot system prompt) have fresh `last_access_time` and sit at the bottom of the heap; they survive. SGLang also supports alternative strategies (LFU, priority-aware) via `EvictionStrategy`, but LRU is the default and the one to reason about.

{{fig:radix-lru-eviction}}

Beyond GPU, SGLang has a **hierarchical** variant (`hiradix_cache.py`, `memory_pool_host.py`) that backs evicted nodes to host/CPU memory (and even disk or a remote KV store) so a prefix that falls out of GPU can be paged back in faster than recomputing it — the same spirit as the multi-tier caches discussed in [Caching, Routing & Cost Control in Production](../12-production-mlops/03-caching-routing-cost.html).

## A from-scratch RadixAttention cache

Reading SGLang's production code is one thing; the idea sticks when you build it. Here is a self-contained, runnable token-level radix cache that captures match/insert/split/evict. It models KV slots as integers (real SGLang stores tensors of pool indices), which is all you need to understand the mechanics.

```python
import time
import heapq
from collections import defaultdict
from itertools import count

_slot_counter = count()   # stand-in for a paged KV allocator handing out slot ids

class Node:
    def __init__(self):
        self.children = {}          # first_token -> Node
        self.key = []               # token ids on the edge into this node
        self.value = []             # KV "slot ids" for those tokens (1 per token)
        self.parent = None
        self.lock_ref = 0           # pinned while a request uses this node
        self.last_access = time.monotonic()

    def __lt__(self, other):        # LRU ordering for the eviction heap
        return self.last_access < other.last_access

def _match_len(a, b):
    """Number of leading tokens shared by sequences a and b."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n

class RadixCache:
    def __init__(self):
        self.root = Node()
        self.num_tokens = 0          # total cached tokens (proxy for KV memory)

    # ---- MATCH: longest cached prefix of `key` -------------------------------
    def match_prefix(self, key):
        node, matched_value, matched_len = self.root, [], 0
        node.last_access = time.monotonic()
        while key:
            first = key[0]
            if first not in node.children:
                break
            child = node.children[first]
            child.last_access = time.monotonic()
            p = _match_len(child.key, key)
            if p < len(child.key):           # partial -> split, then stop
                child = self._split(node, child, p)
                matched_value += child.value
                matched_len += p
                node = child
                break
            else:                            # full edge consumed -> descend
                matched_value += child.value
                matched_len += p
                node = child
                key = key[p:]
        return node, matched_value, matched_len

    def _split(self, parent, child, split_len):
        mid = Node()
        mid.parent = parent
        mid.key = child.key[:split_len]
        mid.value = child.value[:split_len]      # shared prefix KV
        mid.lock_ref = child.lock_ref
        child.key = child.key[split_len:]        # remainder stays on old node
        child.value = child.value[split_len:]
        child.parent = mid
        mid.children = {child.key[0]: child}
        parent.children[mid.key[0]] = mid
        return mid

    # ---- INSERT: cache a full prompt+output, return reused-token count --------
    def insert(self, key):
        node, _, matched_len = self.match_prefix(list(key))
        suffix = key[matched_len:]
        if suffix:                                # allocate fresh KV for the new tail
            child = Node()
            child.parent = node
            child.key = list(suffix)
            child.value = [next(_slot_counter) for _ in suffix]   # "compute" KV
            node.children[suffix[0]] = child
            self.num_tokens += len(suffix)
        return matched_len                        # how many tokens we reused

    # ---- LOCK / UNLOCK: pin a path so it survives eviction -------------------
    def lock_path(self, node, delta):
        while node is not None and node is not self.root:
            node.lock_ref += delta
            node = node.parent

    # ---- EVICT: free `n` tokens, LRU over evictable leaves -------------------
    def evict(self, n):
        leaves = [c for c in self._leaves() if c.lock_ref == 0]
        heapq.heapify(leaves)
        freed = 0
        while freed < n and leaves:
            node = heapq.heappop(leaves)
            freed += len(node.value)
            self.num_tokens -= len(node.value)
            del node.parent.children[node.key[0]]
            parent = node.parent
            if parent is not self.root and not parent.children and parent.lock_ref == 0:
                heapq.heappush(leaves, parent)
        return freed

    def _leaves(self):
        out, stack = [], [self.root]
        while stack:
            nd = stack.pop()
            if nd.children:
                stack.extend(nd.children.values())
            elif nd is not self.root:
                out.append(nd)
        return out


# ---- demo: a shared system prompt across three requests ----------------------
cache = RadixCache()
sys_prompt = list("SYS:")                 # pretend each char is a token id
r1 = sys_prompt + list("hello")
r2 = sys_prompt + list("help")
r3 = sys_prompt + list("hi")

print("req1 reused:", cache.insert(r1))   # 0  (cold cache)
print("req2 reused:", cache.insert(r2))   # 7  ("SYS:hel" shared) -> triggers a split
print("req3 reused:", cache.insert(r3))   # 5  ("SYS:h" shared)
print("tokens cached:", cache.num_tokens) # far fewer than the 9+8+6 tokens across the three full prompts
print("freed by evict(3):", cache.evict(3))
```

Run it: request 1 is a cold miss; request 2 reuses the shared `"SYS:hel"` prefix and forces a node split; request 3 reuses `"SYS:h"`. The total cached-token count is well below the sum of the three prompt lengths, which is precisely the prefill compute you saved. This ~120-line toy is, structurally, what `radix_cache.py` does at scale — minus paging, GQA, page-size alignment, host offload, and CUDA.

!!! warning "Common pitfall: forgetting to lock before you compute"

    A running request's prefix must be **locked** (`lock_ref += 1` along its path) *before* you start its forward pass and unlocked only after it finishes. If you skip this, a concurrent request under memory pressure can evict KV slots your in-flight kernel is still reading, producing silent garbage or an out-of-bounds access. In the toy above, `lock_path` exists for exactly this reason; in SGLang it is `inc_lock_ref` / `dec_lock_ref`. Reference counting — not a global mutex — is what lets many requests safely share one node.

## The frontend language: structured LLM programs

The radix tree gives you *implicit* reuse: send overlapping prompts and they share KV automatically, even through the OpenAI-compatible HTTP server with zero code changes. But SGLang's second contribution is letting you *express structure explicitly* so the runtime can do even better — branch in parallel, share prefixes deliberately, and constrain outputs.

The frontend lives in `python/sglang/lang/`. You write a function decorated with `@sgl.function`; inside, a state object `s` accumulates the program, and primitives describe LLM calls.

```python
import sglang as sgl

@sgl.function
def tip_suggestion(s):
    s += sgl.system("You are an expert assistant. Give concise, correct advice.")
    s += sgl.user("Give me three tips for staying healthy.")
    # Branch: generate three independent tips IN PARALLEL, all sharing the prefix above.
    forks = s.fork(3)
    for i, f in enumerate(forks):
        f += sgl.assistant(sgl.gen(f"tip_{i}", max_tokens=64, temperature=0.7))
    # Recombine the children back into the parent state.
    forks.join()
    tips = [f["tip_" + str(i)] for i, f in enumerate(forks)]
    s += sgl.assistant("Here are three tips:\n" + "\n".join(tips))

# Connect to a running runtime (sglang.launch_server) and run it.
backend = sgl.RuntimeEndpoint("http://localhost:30000")
sgl.set_default_backend(backend)
state = tip_suggestion.run()
print(state["tip_0"], state["tip_1"], state["tip_2"])
```

The primitives (from `python/sglang/lang/api.py`):

- **`gen(name, max_tokens=..., temperature=..., stop=..., regex=..., json_schema=..., choices=...)`** — a model generation, bound to a variable `name` you can read back from the state. With `regex`/`json_schema` it constrains output (next section); with `choices` it becomes a `select`.
- **`select(name, choices=[...])`** — pick the single most likely option from a fixed list, scored by the model's own log-probabilities (length-normalized by default). This is a *constrained classification* primitive: the output is guaranteed to be one of the choices, and it costs one short scoring pass rather than open-ended generation.
- **`fork(k)`** — split the current state into `k` children that **all share the parent's prefix** (and thus its KV cache, via RadixAttention). Branches run concurrently on the runtime.
- **`join()`** — synchronize forked branches and gather their variables back, so the parent can read each child's results.
- **`system` / `user` / `assistant`** (and their `_begin`/`_end` forms) — emit role-tagged segments using the model's [chat template](../05-posttraining-alignment/02-chat-templates-packing.html).

### Why this matters beyond ergonomics

`fork` is where the frontend and RadixAttention meet. When you fork three branches off a shared context, SGLang knows — *statically, before running* — that all three share a prefix, so it computes that prefix's KV once and points all three branches at the same radix node. Compare this to issuing three independent HTTP requests: with RadixAttention the prefix is *probably* still cached, but with `fork` it is *guaranteed* shared and the branches are co-scheduled into the same batch. The frontend turns prefix sharing from a lucky cache hit into a planned execution.

The same applies to agents and chained calls. A multi-step program (extract → reason → format) keeps a single growing state; each step's prompt is the previous state plus new text, so each call reuses everything before it. Tree-of-thought search, self-consistency voting, and branch-and-evaluate harnesses are the canonical fits — the same patterns from [The Agentic Loop](../08-agents-harness/02-agentic-loop.html) and [Reasoning, Chain-of-Thought & Test-Time Compute](../05-posttraining-alignment/10-reasoning-test-time-compute.html).

### Tracing and interpretation

How does `fork` know the prefix statically? SGLang can **trace** the program (`lang/tracer.py`) into an intermediate representation (`lang/ir.py`) — a dataflow graph of `Gen`, `Select`, `Fork`, `Join` nodes — before execution. The interpreter (`lang/interpreter.py`) then walks that graph, dispatching calls to the runtime and managing the shared state. Tracing lets the system reorder independent calls, batch siblings, and reuse prefixes without you orchestrating any of it. For simple use you never see the IR; for advanced control flow it is what makes the structure analyzable.

!!! tip "Practitioner tip: you do not need the frontend to get RadixAttention"

    RadixAttention is a property of the **runtime**, so the plain OpenAI-compatible endpoint (`/v1/chat/completions`) already reuses shared prefixes across independent requests with no code changes. Use the `gen`/`fork` frontend when you want *guaranteed* co-scheduled sharing and parallel branching (agents, tree search, batch evaluation). For a stateless chat proxy, just point your existing OpenAI client at the SGLang server and enjoy free prefix caching.

## Constrained decoding: making the model obey a grammar

A recurring production need is *structured output*: valid JSON, a value from an enum, a number, a date. SGLang's frontend exposes this through `gen(..., regex=...)`, `gen(..., json_schema=...)`, and `select(...)`, backed by a fast constrained-decoding engine in `python/sglang/srt/constrained/`.

The mechanism (covered in depth in [Structured & Constrained Generation](../07-inference-serving/10-structured-generation.html)) is **logit masking against a finite-state machine (FSM)**. A regular expression — or a JSON schema compiled to a grammar — is converted to an FSM over the *token vocabulary*. At each decode step the FSM is in some state; only tokens whose first character(s) keep the FSM on a valid path are allowed. SGLang sets the logits of every disallowed token to $-\infty$ before sampling:

```python
import torch

def apply_fsm_mask(logits, allowed_token_ids):
    """Zero out probability mass on tokens that would violate the grammar."""
    mask = torch.full_like(logits, float("-inf"))
    mask[allowed_token_ids] = 0.0
    return logits + mask        # softmax over this only samples from allowed tokens
```

The expensive part is computing `allowed_token_ids` per state. The naive approach scans the whole vocabulary (e.g. 150k tokens) at every step — pure CPU overhead that can dominate decode latency. SGLang's key optimization is a **compressed FSM with jump-forward decoding**: when the grammar forces a run of tokens (e.g. after `{"name": ` the next characters *must* be `"`), the engine doesn't sample them one at a time. It *jumps forward*, emitting the forced substring in a single step and skipping the model calls entirely. For a JSON schema with many fixed keys and punctuation, jump-forward can collapse a large fraction of decode steps. SGLang integrates grammar backends (historically its own, plus `outlines` and `xgrammar`) and caches compiled FSMs so a repeated schema is compiled once.

`select` is a different, even cheaper kind of constraint: instead of decoding under a mask, it **scores** each candidate string against the prompt and returns the highest-probability one (by default length-normalized log-prob; other methods live in `lang/choices.py`). For "is this sentiment positive/negative/neutral?" `select` is both faster and strictly correct — the output cannot be off-list.

## The runtime and the zero-overhead scheduler

The frontend is the brain; the **runtime** is the muscle. Launch it from the CLI you have locally:

```bash
# Start an OpenAI-compatible SGLang server with RadixAttention on by default.
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --port 30000 \
    --tp-size 1 \                 # tensor-parallel degree (see ch 7.11)
    --mem-fraction-static 0.85 \  # fraction of GPU mem reserved for weights+KV pool
    --chunked-prefill-size 8192   # cap prefill tokens per step (ch 7.8)

# Prefix caching (RadixAttention) is ENABLED by default; disable to A/B test:
#   --disable-radix-cache
```

Internally the runtime mirrors the architecture from [vLLM: Architecture, PagedAttention & Internals](../07-inference-serving/03-vllm-internals.html): a **tokenizer manager** (HTTP front), a **scheduler** (`srt/managers/scheduler.py`) that batches requests and drives the model, and a **detokenizer manager** that streams text back. The scheduler owns the `RadixCache`, runs continuous batching, applies chunked prefill, and decides — every step — which requests to prefill, which to decode, and which to wait. Its admission/eviction logic lives in `srt/managers/schedule_policy.py`, which is *cache-aware*: it prefers to schedule requests that hit long cached prefixes, because those are cheap to admit.

### The "zero-overhead" scheduler

Here is a problem that sounds boring but dominates real throughput: **the CPU scheduler is in the critical path**. Each step the engine must, on the CPU, pick the batch, update the radix tree, build sampling metadata, prepare input tensors, and launch kernels. If the GPU forward pass for a decode step takes, say, 8 ms but the Python scheduling around it takes 4 ms, the GPU idles 33% of the time. The GPU is the expensive resource; idling it is the cardinal sin of an inference engine.

SGLang's **zero-overhead scheduler** (sometimes called *overlap scheduling*) attacks this by **overlapping CPU scheduling with GPU compute**. The trick: while the GPU executes step $t$'s forward pass, the CPU concurrently prepares the batch and metadata for step $t+1$. By the time the GPU finishes step $t$, step $t+1$'s inputs are already staged, so kernels launch back-to-back with no bubble.

```text
Without overlap (CPU and GPU alternate, GPU idles during scheduling):
  CPU: [sched t]              [sched t+1]              [sched t+2]
  GPU:          [fwd t]                  [fwd t+1]                 [fwd t+2]
       <-idle->        <-idle->                 <-idle->

With overlap scheduling (CPU step t+1 runs UNDER GPU step t):
  CPU: [sched t][sched t+1][sched t+2][sched t+3]
  GPU:          [  fwd t  ][ fwd t+1 ][ fwd t+2 ]   <- GPU never waits on the CPU
```

{{fig:overlap-scheduler-timeline}}

Combined with **CUDA graphs** (capturing the decode forward pass once and replaying it to eliminate per-kernel launch overhead — see [Kernel Fusion, torch.compile, CUDA Graphs & Compilers](../04-kernels-efficiency/09-compilers-fusion.html)), the steady-state decode loop becomes almost pure GPU work. This is why SGLang's decode throughput is competitive-to-leading: the scheduler stops being the bottleneck. The implementation realizes this by running model execution and the next batch's preparation on separate streams/threads so their timelines overlap; conceptually:

```python
# Pseudocode for the overlap idea (the real code uses CUDA streams + a worker).
prev = scheduler.prepare_batch()          # build batch for step 0
launch_forward(prev)                      # GPU starts step 0 (async)
while running:
    nxt = scheduler.prepare_batch()       # CPU builds step t+1 WHILE GPU runs step t
    sync(prev)                            # wait only for GPU step t to finish
    sample_and_update(prev)               # detokenize, update radix tree
    launch_forward(nxt)                   # immediately launch step t+1
    prev = nxt
```

The win is structural, not a micro-optimization: it removes a *serial dependency*, so it compounds with everything else (batch size, GQA, quantization).

## A worked example: how much does RadixAttention save?

!!! example "Worked example: 1,000 requests sharing a 2,000-token system prompt"

    **Setup.** We serve a 70B model (GQA: 80 layers, 8 KV heads, $d_{\text{head}}=128$, bf16) to 1,000 chat requests. Every request begins with the **same 2,000-token system prompt**, then appends a unique 50-token user message. We compare prefill cost with and without RadixAttention.

    **KV cache per token** (from the earlier formula):
    $$
    2 \times 80 \times 8 \times 128 \times 2 = 327{,}680 \text{ bytes} \approx 320\ \text{KiB/token}.
    $$
    The shared prefix costs $2000 \times 320\ \text{KiB} \approx 625\ \text{MiB}$ of KV — **stored once**.

    **Prefill FLOPs.** A forward pass costs $\approx 2P$ FLOPs per token for a dense $P$-parameter model ($P = 70 \times 10^9$), so per token $\approx 1.4 \times 10^{11}$ FLOPs.

    *Without RadixAttention* every request re-prefills all $2000 + 50 = 2050$ tokens:
    $$
    1000 \times 2050 \times 1.4\times 10^{11} \approx 2.87 \times 10^{17}\ \text{FLOPs}.
    $$

    *With RadixAttention* the 2,000-token prefix is prefilled **once**; each request prefills only its unique 50-token suffix:
    $$
    \underbrace{2000 \times 1.4\times 10^{11}}_{\text{prefix, once}} + \underbrace{1000 \times 50 \times 1.4\times 10^{11}}_{\text{suffixes}} \approx 2.8\times 10^{14} + 7.0 \times 10^{15} \approx 7.3 \times 10^{15}\ \text{FLOPs}.
    $$

    **Speedup on prefill:** $2.87\times 10^{17} / 7.3\times 10^{15} \approx \mathbf{39\times}$ less prefill compute. The shared prefix went from 97.6% of the prefill work to a one-time cost. Decode work is unchanged (each request still generates its own tokens), so the *end-to-end* speedup depends on your prefill/decode ratio — but for short-output, long-shared-prompt workloads (classification, routing, RAG with a fixed instruction block), the win is enormous.

    **Memory check.** The 625 MiB shared prefix easily fits; the 1,000 unique 50-token suffixes add $1000 \times 50 \times 320\ \text{KiB} \approx 16\ \text{GiB}$ — substantial, which is exactly why eviction and paging matter. As requests finish, their suffix nodes unlock and the LRU evictor reclaims them while keeping the hot shared prefix pinned.

The take-away: **RadixAttention converts repeated prefill into a one-time cost plus cheap suffix prefill.** The more your traffic shares prefixes, the closer you get to that 39× figure; with all-unique prompts the radix tree degenerates to a flat list of leaves and you pay roughly the same as without it (minus tiny bookkeeping).

## SGLang vs vLLM: a practical comparison

Both engines descend from the same insights (continuous batching, paged KV) and have converged feature-wise — vLLM added automatic prefix caching; SGLang added strong paged-attention kernels. They are more alike than different. The distinctions worth knowing:

| Dimension | SGLang | vLLM |
|---|---|---|
| KV reuse data structure | Radix tree of prefixes (`RadixCache`), always-on, cross-request | Automatic prefix caching via block-hash table; PagedAttention blocks |
| Headline original idea | RadixAttention + structured frontend | PagedAttention (block-paged KV) |
| Frontend programming model | `gen`/`select`/`fork`/`join` DSL for multi-call programs | Primarily request-in/text-out; LLM Python API + OpenAI server |
| Scheduler | Zero-overhead / overlap scheduler (CPU under GPU) | Continuous batching; its own scheduler optimizations |
| Constrained decoding | Compressed FSM + jump-forward; xgrammar/outlines backends | Guided decoding via outlines/xgrammar |
| Best-fit workload | Branchy/agentic programs, heavy shared prefixes, structured output | General-purpose high-throughput serving, very broad model & HW support |
| Ecosystem breadth | Fast-moving, strong on reasoning/agent serving | Largest model zoo, hardware backends, integrations |

**When to reach for SGLang:** workloads with strong prefix structure (shared system prompts, few-shot, RAG instruction blocks), agentic/tree-search programs where `fork`/`join` express parallel branches, heavy structured-output (JSON/grammar) needs, and reasoning servers where the overlap scheduler's decode throughput shines. Several RL training stacks use SGLang as their rollout engine for exactly these reasons — see [veRL: HybridFlow & The Single-Controller Architecture](../06-rl-infra/04-verl.html) and [The Generation–Training Loop & Rollout Engines](../06-rl-infra/02-generation-training-loop.html).

**When vLLM may fit better:** maximum model/hardware coverage, an existing vLLM-centric deployment, or when you simply want the broadest, most battle-tested OpenAI-compatible server. In practice many teams benchmark both on *their* traffic — the right answer is workload-dependent, and both projects move fast enough that any specific throughput claim ages within months.

!!! note "Aside: the two ideas are composable, not competing"

    RadixAttention and PagedAttention are orthogonal. PagedAttention is about *how KV blocks are allocated and addressed* (non-contiguous, paged). RadixAttention is about *which requests share which blocks* (a prefix tree over those blocks). SGLang in fact runs paged attention kernels underneath its radix tree — the tree stores *indices into paged blocks*. You can have either without the other, and the best systems have both.

## Putting it together: end-to-end usage

A complete, runnable client that exercises shared prefixes, parallel forks, and constrained output against a locally launched server:

```python
import sglang as sgl

# Assumes: python -m sglang.launch_server --model-path <m> --port 30000 is running.
sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))

@sgl.function
def classify_and_explain(s, review):
    # Shared instruction block: identical across every call -> cached once by RadixAttention.
    s += sgl.system("You are a sentiment classifier. Be precise.")
    s += sgl.user(f"Classify the sentiment of this review:\n{review}")
    # Constrained: output MUST be one of these three labels (FSM-masked / scored).
    s += sgl.assistant("Sentiment: " + sgl.gen("label",
                                               choices=["positive", "negative", "neutral"]))
    # Now branch: 2 independent one-sentence justifications, sharing everything above.
    forks = s.fork(2)
    for i, f in enumerate(forks):
        f += sgl.user("Give a one-sentence reason.")
        f += sgl.assistant(sgl.gen(f"reason_{i}", max_tokens=40, temperature=0.9))
    forks.join()
    return {
        "label": s["label"],
        "reasons": [forks[i]["reason_" + str(i)] for i in range(2)],
    }

reviews = [
    "The battery dies in two hours. Useless.",
    "Exceeded every expectation — buying another!",
    "It works. Nothing special, nothing wrong.",
]
# run_batch executes these concurrently; the shared system+instruction prefix
# is prefilled ONCE and reused across all three via the radix tree.
states = classify_and_explain.run_batch(
    [{"review": r} for r in reviews], progress_bar=True
)
for st in states:
    print(st["label"], "::", st["reasons"][0])
```

What happens under the hood, end to end:

1. The three calls in `run_batch` are admitted by the scheduler. Their shared `system` + instruction prefix matches a single radix node after the first call computes it; calls 2 and 3 **reuse** its KV (a cache hit, no re-prefill).
2. The `select` over `["positive","negative","neutral"]` scores the three candidates and returns the best — output is guaranteed on-list.
3. Each `fork(2)` creates two children sharing the parent's full KV path; the two justifications generate in parallel, co-scheduled.
4. The overlap scheduler keeps the GPU busy across decode steps; CUDA graphs replay the decode kernel; finished requests unlock their nodes; LRU eviction reclaims cold suffixes under pressure.

You wrote a branchy, constrained, prefix-sharing program in ~20 lines and the runtime exploited all three properties automatically.

!!! interview "Interview Corner"

    **Q:** You're serving a chatbot where every request shares a 1,500-token system prompt, but user messages and conversations are all different. Walk me through how you'd cache KV across requests, and what data structure you'd use. What breaks at scale, and how do you bound memory?

    **A:** I'd use a **radix tree (compressed prefix trie) over token IDs**, exactly SGLang's RadixAttention. Each node owns the KV-cache slot indices for the tokens on its incoming edge; a path from root spells a cached token sequence. On a new request I run `match_prefix` to find the longest cached prefix, reuse those KV slots, and only prefill the divergent suffix — so the 1,500-token system prompt is prefilled **once** and shared by all requests, splitting nodes when prompts diverge mid-edge.

    To bound memory I store KV in a **paged pool** and let the tree hold indices, not tensors, so one block can be referenced by many nodes. I **reference-count** nodes (`lock_ref`): a node is pinned while any running request uses its path, and becomes evictable when all finish. Under memory pressure I **evict LRU over evictable leaves** — peeling cold suffixes from the tips inward, never orphaning a live prefix, never reclaiming locked KV. The hot system prompt has a fresh access time and survives. What breaks at scale: with all-unique prompts the tree degenerates to a flat leaf list and reuse vanishes (you pay normal prefill plus tiny overhead); and the CPU bookkeeping per step can starve the GPU — which is why SGLang overlaps scheduling with compute. For a final tier I'd offload evicted prefixes to host/CPU memory so a falling-out prefix can be paged back faster than recomputed.

!!! key "Key Takeaways"

    - **RadixAttention** keeps a *radix tree of token prefixes* whose nodes own KV-cache slot indices, giving automatic, always-on, cross-request KV reuse — not just for a pinned system prompt but for any overlapping prefixes.
    - The core operations are **match (find longest cached prefix), split (share a diverging edge), insert (prefill only the suffix), and evict (LRU over unlocked leaves)** — peeling cold tips while pinning hot prefixes via reference counting (`lock_ref`).
    - RadixAttention is layered on **paged KV** (PagedAttention): the tree stores *indices into paged blocks*, so the two ideas compose rather than compete.
    - The **frontend DSL** — `gen`, `select`, `fork`, `join` — lets you express branchy multi-call programs so the runtime can *guarantee* prefix sharing and co-schedule parallel branches, instead of relying on lucky cache hits.
    - **Constrained decoding** masks logits against a grammar/FSM; SGLang's **compressed FSM + jump-forward** emits forced token runs in one step, slashing decode cost for JSON/regex outputs, while `select` scores a fixed choice list.
    - The **zero-overhead (overlap) scheduler** runs CPU batch preparation *under* the GPU forward pass, removing scheduling bubbles; with CUDA graphs the decode loop becomes near-pure GPU work.
    - **Versus vLLM:** the engines have converged; SGLang's edge is branchy/agentic programs, heavy shared prefixes, and structured output, while vLLM leads on breadth of models/hardware. Benchmark both on your own traffic.
    - The savings are real and bounded by your sharing: a long shared prefix can cut prefill compute by an order of magnitude or more; all-unique prompts see little benefit.

!!! sota "State of the Art & Resources (2026)"
    SGLang has become one of the two dominant open-source LLM inference engines (alongside vLLM), with RadixAttention now a standard technique widely replicated across serving frameworks. The zero-overhead overlap scheduler and XGrammar-backed structured generation represent the current frontier for high-throughput production serving as of 2026.

    **Foundational work**

    - [Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs* (2023)](https://arxiv.org/abs/2312.07104) — introduces RadixAttention, the frontend DSL, and compressed-FSM decoding; the primary reference for this chapter.
    - [Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (2023)](https://arxiv.org/abs/2309.06180) — the vLLM paper establishing paged KV blocks that RadixAttention indexes into.
    - [Willard & Louf, *Efficient Guided Generation for Large Language Models* (2023)](https://arxiv.org/abs/2307.09702) — FSM-based constrained decoding (the Outlines approach) that SGLang's grammar backend builds upon.

    **Recent advances (2023–2026)**

    - [Dong et al., *XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models* (2024)](https://arxiv.org/abs/2411.15100) — near-zero-overhead context-free grammar decoding; now the default structured-output backend in SGLang and vLLM.
    - [LMSYS, *SGLang v0.4: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs* (2024)](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/) — engineering post detailing the overlap scheduler and cache-aware load balancing that define the current SGLang architecture.
    - [LMSYS, *Achieving Faster Open-Source Llama3 Serving with SGLang Runtime* (2024)](https://www.lmsys.org/blog/2024-07-25-sglang-llama3/) — benchmark showing SGLang matching or exceeding TensorRT-LLM on Llama-3 at various scales.

    **Open-source & tools**

    - [sgl-project/sglang](https://github.com/sgl-project/sglang) — the main SGLang repo; see `srt/mem_cache/radix_cache.py` for RadixAttention and `lang/api.py` for the frontend DSL.
    - [mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar) — the XGrammar structured generation engine used by SGLang, vLLM, and TensorRT-LLM.

    **Go deeper**

    - [LMSYS, *Fast and Expressive LLM Inference with RadixAttention and SGLang* (2024)](https://www.lmsys.org/blog/2024-01-17-sglang/) — the original blog post walking through RadixAttention's design and throughput results, an excellent complement to the paper.
    - [LMSYS, *Fast JSON Decoding for Local LLMs with Compressed Finite State Machine* (2024)](https://www.lmsys.org/blog/2024-02-05-compressed-fsm/) — deep-dive on jump-forward decoding and why the compressed FSM cuts JSON decode latency by up to 2×.
    - [SGLang Documentation](https://sgl-project.github.io/) — official docs covering installation, server flags, structured output APIs, and deployment guides.

## Further reading

- Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs* — the paper that introduces RadixAttention, the frontend language, and the compressed-FSM constrained decoder.
- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (the vLLM paper) — the paged-KV foundation RadixAttention builds on; compare and contrast.
- Dao et al., *FlashAttention* and *FlashAttention-2* — the IO-aware attention kernels that the runtime uses under the hood; see [FlashAttention I](../04-kernels-efficiency/02-flash-attention-1.html).
- Willard & Louf, *Efficient Guided Generation for Large Language Models* (the `outlines` FSM approach) and the **xgrammar** project — grammar/FSM constrained decoding integrated by SGLang.
- The **sglang** GitHub repository (`sgl-project/sglang`) — read `python/sglang/srt/mem_cache/radix_cache.py`, `srt/managers/scheduler.py`, and `lang/api.py` for the production implementations behind this chapter.
- Related chapters: [vLLM: Architecture, PagedAttention & Internals](../07-inference-serving/03-vllm-internals.html), [Prefix Caching & KV-Cache Reuse](../07-inference-serving/07-prefix-caching.html), [Structured & Constrained Generation](../07-inference-serving/10-structured-generation.html), and [Continuous Batching & Request Scheduling](../07-inference-serving/02-continuous-batching.html).
