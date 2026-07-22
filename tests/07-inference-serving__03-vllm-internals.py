"""
Runs the CPU-runnable Python code blocks from:
    content/07-inference-serving/03-vllm-internals.md

Chapter has 10 Python/bash blocks total (#0-#9). Heuristic classification:
    #0 (line ~52, 19 lines)  -> TESTED: startup KV-budget arithmetic sketch (pure Python/int math).
    #1 (line ~80)             -> SKIP(fragment): `BlockManager` class defined but never
                                  instantiated/exercised in isolation in the book's flow here
                                  (it's a reference implementation, not called with concrete
                                  numbers in the prose). Not one of the 3 target blocks.
    #2 (line ~139)            -> SKIP(fragment): `append_with_cow` is a method sketch that
                                  assumes a `self` with `free_blocks`/`ref_count` already set
                                  up elsewhere; not one of the 3 target blocks.
    #3 (line ~167)            -> SKIP(fragment): `schedule_step` method sketch referencing
                                  `self.running`, `self.waiting`, `self.block_mgr`, `_preempt`,
                                  `_budget_left` that are never defined in the chapter; not one
                                  of the 3 target blocks.
    #4 (line ~231)            -> SKIP(needs-gpu): `prepare_inputs` builds `torch.tensor(...,
                                  device="cuda")` -- hard CUDA dependency; not one of the 3
                                  target blocks.
    #5 (line ~308)            -> SKIP(fragment): `block_hash` / `hash_prompt_blocks` helper
                                  functions; not one of the 3 target blocks (also not flagged
                                  needs-gpu/CPU-runnable by the harness for this run).
    #6 (line ~353, 8 lines)  -> TESTED: speculative-decoding engine step sketch. This is
                                  pseudocode referencing undefined `proposer`, `seq`, `k`,
                                  `target_model`, `rejection_sample`, `sample`. We supply
                                  minimal, honest, CPU-only fixtures/implementations for each
                                  (a toy proposer, a toy sequence object, a toy "target model"
                                  producing random logits, and a real rejection-sampling
                                  routine implementing what the prose describes) and then run
                                  the book's lines verbatim against them.
    #7 (line ~377, 21 lines) -> SKIP(needs-gpu/needs-network): imports `vllm` (not an allowed
                                  CI dependency), constructs `LLM(model="meta-llama/Llama-3-8b",
                                  ...)` which downloads real weights from the Hugging Face Hub
                                  and requires a GPU, and calls `.generate(...)` against that
                                  real model. There is no honest boundary to mock here without
                                  gutting the very thing being demonstrated (real multi-LoRA
                                  serving), so per the hard rules on network/GPU dependencies
                                  this block is skipped rather than faked. (The task's heuristic
                                  flagged it "CPU-runnable" by shape alone; the actual content
                                  needs a GPU + network + a real 8B checkpoint.)
    #8 (line ~416)            -> SKIP(needs-gpu): `vllm serve` CLI example (bash, launches a
                                  real GPU server) -- also explicitly needs-gpu per the harness.
    #9 (curl example)         -> SKIP(shell): a `curl` invocation against a running server, not
                                  Python.

Only #0 and #6 are executed here as literal book code (verbatim, modulo the tiny fixtures
block #6 needs to have names to run against). #7 is intentionally skipped, not faked.
"""

import numpy as np

# --------------------------------------------------------------------------
# Block #0 (book line ~52): "Sketch of vLLM's startup KV-budget computation"
# Pure integer/float arithmetic, no external deps -- run verbatim.
# --------------------------------------------------------------------------

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

# Sanity checks that the book's arithmetic actually executed and produced a
# sane KV budget (this is the whole point of the block: sizing the KV pool).
assert usable == int(80 * 1024**3 * 0.90)
assert kv_cache_bytes > 0, "KV budget should be positive for these illustrative numbers"
assert bytes_per_block == 2 * 16 * 8 * 128 * 32 * 2
assert num_gpu_blocks == kv_cache_bytes // bytes_per_block
assert num_gpu_blocks > 0


# --------------------------------------------------------------------------
# Block #6 (book line ~353): "Engine-level shape of one speculative step"
#
# The book presents this as pseudocode over names it never defines in this
# chapter (`proposer`, `seq`, `k`, `target_model`, `rejection_sample`,
# `sample`) -- it is illustrating the *shape* of a speculative-decoding step,
# not a self-contained snippet. To actually execute it we provide minimal,
# honest CPU fixtures:
#   - `ToySeq`     : a sequence object with `.context` (list of token ids)
#                    and `.extend(tokens)`, matching how the book's code
#                    uses `seq.context` and `seq.extend(...)`.
#   - `ToyProposer`: a tiny stand-in "drafter" with `.propose(seq, k)` that
#                    returns k draft tokens + their draft probabilities,
#                    matching the n-gram/EAGLE-style proposer interface
#                    described in the prose just above this block.
#   - `target_model`: a toy "big model" forward pass returning random logits
#                    of shape (num_positions, vocab) -- stands in for the
#                    real target model's single k+1-position forward pass.
#   - `rejection_sample`: a real implementation of the rejection-sampling
#                    verifier the prose describes ("accepts the longest
#                    prefix of drafts consistent with the target's
#                    distribution"), so the book's line that calls it is
#                    exercising real (not stubbed-out) logic.
#   - `sample`     : samples a token id from a logits vector via softmax.
# --------------------------------------------------------------------------

rng = np.random.default_rng(0)
VOCAB = 50


def sample(logits):
    """Sample a token id from logits (softmax sampling, CPU/numpy)."""
    probs = np.exp(logits - np.max(logits))
    probs = probs / probs.sum()
    return int(rng.choice(len(probs), p=probs))


def rejection_sample(draft_tokens, draft_probs, target_logits):
    """Speculative-decoding rejection sampling: walk the draft tokens in
    order and accept token i with probability min(1, p_target(tok)/p_draft(tok)),
    stopping at the first rejection -- this is the mechanism the chapter's
    prose describes ("accepts the longest prefix of drafts consistent with
    the target's distribution")."""
    accepted = []
    for i, (tok, q) in enumerate(zip(draft_tokens, draft_probs)):
        row = target_logits[i]
        target_probs = np.exp(row - np.max(row))
        target_probs = target_probs / target_probs.sum()
        p_target = float(target_probs[tok])
        if rng.random() < min(1.0, p_target / q):
            accepted.append(tok)
        else:
            break
    return accepted


class ToyProposer:
    """Stand-in cheap drafter: proposes k tokens with fabricated draft
    probabilities (a real n-gram/EAGLE proposer would instead score
    likely-continuation tokens; the interface shape is what matters here)."""

    def propose(self, seq, k):
        draft_tokens = [int(rng.integers(0, VOCAB)) for _ in range(k)]
        draft_probs = [float(rng.uniform(0.3, 0.9)) for _ in range(k)]
        return draft_tokens, draft_probs


class ToySeq:
    """Stand-in sequence object matching the `.context` / `.extend()`
    interface the book's snippet uses."""

    def __init__(self, context):
        self.context = list(context)

    def extend(self, tokens):
        self.context.extend(tokens)


def target_model(token_ids):
    """Stand-in target model forward pass: returns one row of logits per
    input position, shape (len(token_ids), VOCAB) -- mirrors the real
    target model's single k+1-position forward pass."""
    return rng.normal(size=(len(token_ids), VOCAB))


proposer = ToyProposer()
seq = ToySeq(context=[1, 2, 3, 4, 5])
k = 4
context_len_before = len(seq.context)

# ---- book code, verbatim ----
draft_tokens, draft_probs = proposer.propose(seq, k)          # cheap
target_logits = target_model(seq.context + draft_tokens)      # ONE big fwd, k+1 pos
accepted = rejection_sample(draft_tokens, draft_probs, target_logits)  # 0..k accepted
seq.extend(accepted)
seq.extend([sample(target_logits[len(accepted)])])            # +1 bonus/correction token
# Net: up to k+1 tokens emitted per single target forward pass.
# ---- end book code ----

assert len(draft_tokens) == k
assert len(draft_probs) == k
# one logits row per input position: the prior context plus the k draft tokens
assert target_logits.shape == (context_len_before + k, VOCAB)
assert 0 <= len(accepted) <= k
# exactly one bonus/correction token appended on top of however many drafts
# were accepted -- "up to k+1 tokens emitted per single target forward pass".
assert len(seq.context) == context_len_before + len(accepted) + 1
assert len(seq.context) <= context_len_before + k + 1


# --------------------------------------------------------------------------
# Block #5 (book line ~308): prefix-caching rolling block hashes.
# `block_hash` / `hash_prompt_blocks` are complete, self-contained pure-Python
# functions (only `hash` + tuples) -- run verbatim, then assert the two
# properties the prose claims: (a) identical prefixes hash block-for-block
# identically; (b) prefixes that diverge get different hashes from the block
# of divergence onward (the rolling hash chains in `prev_hash`).
# --------------------------------------------------------------------------

# ---- book code, verbatim ----
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
# ---- end book code ----

_bs = 4
_base = [10, 11, 12, 13,  20, 21, 22, 23,  30, 31, 32, 33]
# Same prefix hashed twice -> identical block hashes (cache would hit).
assert hash_prompt_blocks(_base, _bs) == hash_prompt_blocks(list(_base), _bs)
# Only full blocks are hashed: 12 tokens / block_size 4 -> 3 block hashes;
# 2 trailing partial tokens contribute no hash.
assert len(hash_prompt_blocks(_base + [99, 98], _bs)) == 3
# Divergence in block 1 (index 4) => block 0 matches, blocks 1.. differ.
_diverge = [10, 11, 12, 13,  20, 21, 22, 99,  30, 31, 32, 33]
_hb, _hd = hash_prompt_blocks(_base, _bs), hash_prompt_blocks(_diverge, _bs)
assert _hb[0] == _hd[0], "identical leading block must share its hash"
assert _hb[1] != _hd[1], "divergent block must differ"
assert _hb[2] != _hd[2], "rolling hash must propagate divergence forward"
# Different LoRA id => different KV => different hash (correctness/security).
assert hash_prompt_blocks(_base, _bs, lora_id=1) != hash_prompt_blocks(_base, _bs, lora_id=2)


# --------------------------------------------------------------------------
# Block #7 (book line ~377): multi-LoRA `vllm.LLM(...)` / `.generate(...)`
# example.
#
# SKIP(needs-gpu, needs-network): this constructs a real vLLM engine against
# "meta-llama/Llama-3-8b" (downloads multi-GB weights from the Hugging Face
# Hub) and requires a GPU to run the forward passes; `vllm` is also not in
# this test suite's allowed-import list. Mocking `vllm.LLM` down to a stub
# would just assert that Python can call a mock -- it would not exercise any
# real logic of the block (unlike, say, mocking an HTTP client around real
# retry/parsing code), so per the hard rules this is left as an honest skip
# rather than faked. No import of `vllm` is attempted anywhere in this file.
# --------------------------------------------------------------------------


print("\nAll tested blocks (#0, #5, #6) executed successfully.")
print(f"  block #0: num_gpu_blocks={num_gpu_blocks}, bytes_per_block={bytes_per_block}")
print(f"  block #5: hashed {len(hash_prompt_blocks(_base, _bs))} full blocks; "
      f"rolling-hash divergence + lora-id isolation verified")
print(f"  block #6: draft_tokens={draft_tokens}, accepted={accepted}, "
      f"final seq.context len={len(seq.context)}")
