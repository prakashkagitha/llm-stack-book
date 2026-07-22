"""
Runs the CPU-runnable Python blocks from
content/07-inference-serving/05-trtllm-tgi-stacks.md

Blocks tested (in chapter order):
  - block #1 (line ~79):  TensorRT-LLM async Executor example
      -> tensorrt_llm is a GPU-only package that also requires network access
         to obtain / build an engine. We guard the import and, since the
         package is unavailable in CI, substitute a tiny in-process fake
         GenerationExecutor / GenerationRequest that reproduces the async
         submit/aiter_tokens contract the block relies on -- so the block's
         OWN async scheduling logic still executes, offline, on CPU.
  - block #4 (line ~204):  TGIScheduler pseudocode class
      -> pure Python, CPU safe. Instantiated and exercised with a tiny
         Request dataclass (added to the chapter to fix a real bug -- see
         below).
  - block #5 (line ~265):  gguf_size_gb() model-size estimator
      -> pure Python/arithmetic, CPU safe. Called with representative
         parameter counts.

Skipped blocks:
  - #0, #2, #3, #6, #7: shell/bash blocks (trtllm-build CLI, docker run,
    curl, ollama CLI) -- not Python.
  - #8: needs a real GPU (TensorRT-LLM engine build / execution).
  - #9: non-python (C++ snippet).

Real bug found & fixed in content/07-inference-serving/05-trtllm-tgi-stacks.md:
  Block #4's `TGIScheduler.schedule` had the return-type annotation
  `-> list[Request]` but `Request` was never defined anywhere in the
  chapter. Return-type annotations are evaluated at class-definition time
  (unlike local-variable annotations inside a method body, which are not),
  so this raised `NameError: name 'Request' is not defined` the instant the
  class body executed -- i.e. the block could never even be imported, let
  alone run. Fixed by adding a minimal `Request` dataclass (with the two
  attributes the pseudocode actually reads: `current_length` and
  `max_total_tokens`) immediately before the `TGIScheduler` class in the
  chapter. Mirrored here.
"""

import asyncio
import sys

# ---------------------------------------------------------------------------
# Guarded third-party import (per HARD RULES: any non-stdlib/non-numpy/torch/
# einops/sklearn import must be guarded so the file still loads without it).
# tensorrt_llm is GPU-only and not installed in CI.
# ---------------------------------------------------------------------------
try:
    from tensorrt_llm.executor import GenerationExecutor, GenerationRequest
except Exception:
    GenerationExecutor = None
    GenerationRequest = None

if GenerationExecutor is None:
    # Offline fake reproducing the minimal async submit / aiter_tokens
    # contract the book's block relies on, so the block's own async
    # request-response logic still runs on CPU without network/GPU.
    class GenerationRequest:
        def __init__(self, input_token_ids, max_new_tokens, streaming=False):
            self.input_token_ids = input_token_ids
            self.max_new_tokens = max_new_tokens
            self.streaming = streaming

        async def aiter_tokens(self):
            # Yield a handful of fake tokens, proportional to max_new_tokens,
            # capped so the test stays fast.
            n = min(self.max_new_tokens, 5)
            for i in range(n):
                await asyncio.sleep(0)
                yield f"tok{i}"

    class GenerationExecutor:
        def __init__(self, engine_dir, executor_config):
            self.engine_dir = engine_dir
            self.executor_config = executor_config
            self.submitted = []

        @classmethod
        def create(cls, engine_dir, executor_config):
            return cls(engine_dir, executor_config)

        def submit(self, request):
            self.submitted.append(request)


# ===========================================================================
# Block #1 (line ~79): Minimal TensorRT-LLM Executor example
# ===========================================================================

collected_tokens = []  # test-side hook to prove the async loop actually ran


async def serve_requests():
    """
    The Executor runs a background thread that continuously feeds the engine.
    Requests are submitted as GenerationRequest objects and picked up
    at the next scheduling interval (default: every decode step).
    """
    executor = GenerationExecutor.create(
        engine_dir="./llama-2-7b-engine",
        executor_config={
            "max_beam_width": 1,
            "scheduler_policy": "guaranteed_no_evict",  # vs "max_utilization"
        }
    )

    # Submit two requests concurrently — they will be batched automatically
    req_a = GenerationRequest(
        input_token_ids=[1, 234, 567],
        max_new_tokens=100,
        streaming=True,
    )
    req_b = GenerationRequest(
        input_token_ids=[1, 890, 123, 456],
        max_new_tokens=50,
        streaming=True,
    )

    executor.submit(req_a)
    executor.submit(req_b)

    # Stream tokens as they arrive
    async for token in req_a.aiter_tokens():
        print(f"A: {token}", end=" ", flush=True)
        collected_tokens.append(token)

    return executor


executor_result = asyncio.run(serve_requests())
print()

assert len(executor_result.submitted) == 2, "both requests should have been submitted"
assert len(collected_tokens) > 0, "the async token stream should have yielded tokens"


# ===========================================================================
# Block #4 (line ~204): TGIScheduler pseudocode class
# (Request dataclass added to fix the NameError bug described above)
# ===========================================================================

from dataclasses import dataclass


@dataclass
class Request:
    current_length: int
    max_total_tokens: int


class TGIScheduler:
    def __init__(self, max_batch_total_tokens: int):
        self.max_batch_total_tokens = max_batch_total_tokens
        self.waiting: list[Request] = []
        self.running: list[Request] = []

    def schedule(self) -> list[Request]:
        """
        Called every decode step. Fill the running batch up to the token budget.
        New requests are added from waiting if budget permits.
        Running requests keep their slot as long as they haven't finished.
        """
        budget_used = sum(r.current_length for r in self.running)
        for req in list(self.waiting):
            needed = req.max_total_tokens  # pre-allocated worst case
            if budget_used + needed <= self.max_batch_total_tokens:
                self.running.append(req)
                self.waiting.remove(req)
                budget_used += needed
        return self.running


# Exercise TGIScheduler with a tiny fixture: a token budget of 100, three
# fresh waiting requests reserving 40 tokens (worst case) each -- only the
# first two should fit (budget_used starts at 0 since nothing is running
# yet: 0+40<=100, 40+40<=100, 80+40>100).
scheduler = TGIScheduler(max_batch_total_tokens=100)
scheduler.waiting = [
    Request(current_length=0, max_total_tokens=40),
    Request(current_length=0, max_total_tokens=40),
    Request(current_length=0, max_total_tokens=40),
]

running = scheduler.schedule()

assert len(running) == 2, f"expected 2 requests to fit in the 100-token budget, got {len(running)}"
assert len(scheduler.waiting) == 1, "the third request should remain in the waiting queue"
assert sum(r.max_total_tokens for r in running) == 80

# A later decode step: the two running requests have since generated more
# tokens (current_length grew from 0 towards their max_total_tokens), which
# is what `budget_used` is computed from. With 90 tokens of actual usage
# already, the remaining 40-token-budget waiting request no longer fits
# (90 + 40 > 100), so it correctly stays queued.
for r in scheduler.running:
    r.current_length = 45
running_2 = scheduler.schedule()
assert len(running_2) == 2, "no additional requests should fit once running requests have grown"
assert len(scheduler.waiting) == 1


# ===========================================================================
# Block #5 (line ~265): gguf_size_gb() model-size estimator
# ===========================================================================

def gguf_size_gb(params_billions: float, bits_per_weight: float = 4.5) -> float:
    """
    Rough size estimate for a GGUF-quantized LLM.
    bits_per_weight: 4.5 for Q4_K_M, 5.5 for Q5_K_M, 8.5 for Q8_0
    """
    params = params_billions * 1e9
    bytes_per_weight = bits_per_weight / 8.0
    # Embeddings and norms are typically kept in FP16, ~5% of params
    emb_fraction = 0.05
    size_bytes = (
        params * (1 - emb_fraction) * bytes_per_weight
        + params * emb_fraction * 2.0  # FP16
    )
    return size_bytes / (1024**3)


# Llama-3-70B at Q4_K_M:
size_70b_q4 = gguf_size_gb(70, 4.5)
size_70b_q5 = gguf_size_gb(70, 5.5)
size_13b_q4 = gguf_size_gb(13, 4.5)

print(f"70B Q4_K_M: {size_70b_q4:.1f} GB")   # ~41 GB
print(f"70B Q5_K_M: {size_70b_q5:.1f} GB")   # ~49 GB
print(f"13B Q4_K_M: {size_13b_q4:.1f} GB")   # ~7.7 GB

# The chapter states these approximate values in a trailing comment;
# assert the same order-of-magnitude claims (loose tolerance -- this is a
# rough arithmetic estimator, not a fitted constant).
assert 38.0 < size_70b_q4 < 44.0, size_70b_q4
assert 46.0 < size_70b_q5 < 52.0, size_70b_q5
assert 7.0 < size_13b_q4 < 8.5, size_13b_q4
# Q5 should always be larger than Q4 for the same param count.
assert size_70b_q5 > size_70b_q4
# Size should scale roughly linearly with parameter count.
assert size_70b_q4 > 5 * size_13b_q4


print("\nAll CPU-runnable blocks in 05-trtllm-tgi-stacks.md executed successfully.")
sys.exit(0)
