"""
Runs the CPU-runnable Python blocks from
content/07-inference-serving/04-sglang-radixattention.md, concatenated in
order. Each tested block is copied verbatim from the chapter; only the
minimal glue needed to make a block that merely *defines* something actually
execute is added, and is clearly marked "GLUE".

Blocks covered (the 4 heuristically CPU-runnable blocks the task named):

  #5  (line ~171) - from-scratch RadixCache (Node/RadixCache classes) + the
                     chapter's own demo (insert/evict on a shared prefix).
                     Fully self-contained (stdlib only) -- run verbatim.
  #6  (line ~313) - SGLang frontend program (`@sgl.function`, `fork`/`join`)
                     that connects to `sgl.RuntimeEndpoint("http://localhost:30000")`
                     and calls `.run()` -- SKIP, see below.
  #7  (line ~364) - `apply_fsm_mask(logits, allowed_token_ids)` -- self-contained
                     torch logit-masking helper. Run verbatim, then exercised
                     with a tiny GLUE-added call (the book shows the function
                     but never calls it).
  #11 (line ~488) - end-to-end SGLang client (`@sgl.function`, `run_batch`)
                     that connects to `sgl.RuntimeEndpoint("http://localhost:30000")`
                     -- SKIP, see below.

SKIP(network) + SKIP(missing-package): blocks #6 and #11 both `import sglang`
and then talk to a locally launched SGLang server via
`sgl.RuntimeEndpoint("http://localhost:30000")` (`.run()` / `.run_batch()`).
Two independent reasons this cannot run here: (1) `sglang` is not on the
guaranteed-available import list (numpy/torch/einops/sklearn/stdlib) and is
not installed in this environment, so the import is guarded; (2) even if it
were installed, both blocks' entire point is issuing real HTTP requests to an
inference server (`http://localhost:30000`) that does not exist in CI --
there is no meaningful boundary to mock without fabricating the DSL's
internal tracing/scheduling behaviour, which is exactly the logic these
blocks exist to demonstrate. Both blocks are therefore left defined-not-called
(wrapped in functions) and skipped outright.
"""

import time
import heapq
from collections import defaultdict
from itertools import count

import torch

try:
    import sglang as sgl
except Exception:
    sgl = None


# ============================================================================
# Block #5 (line ~171): a from-scratch RadixAttention cache -- run verbatim.
# ============================================================================

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

req1_reused = cache.insert(r1)
print("req1 reused:", req1_reused)   # 0  (cold cache)
req2_reused = cache.insert(r2)
print("req2 reused:", req2_reused)   # 7  ("SYS:hel" shared) -> triggers a split
req3_reused = cache.insert(r3)
print("req3 reused:", req3_reused)   # 5  ("SYS:h" shared)
print("tokens cached:", cache.num_tokens) # far fewer than the 9+8+6 tokens across the three full prompts
freed = cache.evict(3)
print("freed by evict(3):", freed)

# Assertions mirroring the chapter's own inline comments. NOTE: the book's
# original comments here (# 5 "SYS:h" shared / # 4 "SYS:" shared) were WRONG --
# running the code shows the actual longest-common-prefix matches are 7
# ("SYS:hel") and 5 ("SYS:h"). This was a real bug in the chapter (the
# worked-example comments didn't match what its own code computes); fixed in
# both the chapter markdown and here.
assert req1_reused == 0
assert req2_reused == 7
assert req3_reused == 5
assert cache.num_tokens < len(r1) + len(r2) + len(r3)
assert freed >= 3


# ============================================================================
# Block #6 (line ~313): SGLang frontend program hitting a live runtime.
# SKIP(network) + SKIP(missing-package): `sglang` is not installed here, and
# even if it were, `sgl.RuntimeEndpoint("http://localhost:30000").run()` is a
# real HTTP call to a server this environment does not run. Left
# defined-not-called.
# ============================================================================

def _block6_tip_suggestion_demo():
    """Book's block #6, verbatim, wrapped so it is defined but never called."""
    import sglang as sgl  # noqa: F401 (re-import inside, only used if invoked)

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

print("SKIP(network+missing-package): block #6 (sglang frontend program hitting "
      "http://localhost:30000) left defined-not-called; sglang is not installed "
      "and no live SGLang server is available in this environment.")


# ============================================================================
# Block #7 (line ~364): FSM logit masking -- self-contained, run verbatim.
# ============================================================================

import torch

def apply_fsm_mask(logits, allowed_token_ids):
    """Zero out probability mass on tokens that would violate the grammar."""
    mask = torch.full_like(logits, float("-inf"))
    mask[allowed_token_ids] = 0.0
    return logits + mask        # softmax over this only samples from allowed tokens

# --- GLUE: the book shows the function but never calls it. Exercise it with
# a tiny vocabulary so its logic actually runs on CPU. ---
_vocab_size = 10
_logits = torch.randn(_vocab_size)
_allowed = torch.tensor([2, 5, 7])
_masked = apply_fsm_mask(_logits, _allowed)

_probs = torch.softmax(_masked, dim=-1)
_nonzero_positions = (_probs > 0).nonzero(as_tuple=True)[0]
assert set(_nonzero_positions.tolist()) == set(_allowed.tolist())
assert torch.isclose(_probs.sum(), torch.tensor(1.0), atol=1e-5)
print("apply_fsm_mask: probability mass restricted to allowed tokens:",
      sorted(_nonzero_positions.tolist()))


# ============================================================================
# Block #11 (line ~488): end-to-end SGLang client hitting a live runtime.
# SKIP(network) + SKIP(missing-package): same reasons as block #6 -- `sglang`
# is not installed, and the block's entire point is real HTTP calls to
# http://localhost:30000. Left defined-not-called.
# ============================================================================

def _block11_classify_and_explain_demo():
    """Book's block #11, verbatim, wrapped so it is defined but never called."""
    import sglang as sgl  # noqa: F401

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

print("SKIP(network+missing-package): block #11 (sglang end-to-end client hitting "
      "http://localhost:30000) left defined-not-called; sglang is not installed "
      "and no live SGLang server is available in this environment.")


print("\nAll CPU-runnable blocks from 07-inference-serving/04-sglang-radixattention.md executed OK.")
