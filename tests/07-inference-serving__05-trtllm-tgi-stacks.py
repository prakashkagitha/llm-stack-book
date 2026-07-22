"""
Executable test for content/07-inference-serving/05-trtllm-tgi-stacks.md

Concatenates the chapter's 3 CPU-runnable Python blocks in order and exercises
each one on CPU / with tiny fixtures so the book's actual code runs end to end.

Blocks covered:
  #1 (line ~79)  TensorRT-LLM Executor async request/response example
                 (`GenerationExecutor`, `GenerationRequest`). `tensorrt_llm` is
                 not installed in CI and is not on the allowed-import list, so
                 we inject a minimal fake `tensorrt_llm.executor` module into
                 sys.modules *before* the book's `from tensorrt_llm.executor
                 import ...` line runs. The fake backend implements exactly the
                 surface the block calls (`.create()`, `.submit()`,
                 `.aiter_tokens()`) so the block's own async submission/
                 streaming logic executes for real against a canned backend,
                 rather than being skipped or stubbed out.
  #4 (line ~204) TGIScheduler class (token-budget waiting-queue scheduler).
                 Pure Python, no external deps. The class body references a
                 `Request` type only in (attribute) annotations, which Python
                 evaluates at class-definition / assignment time even though
                 it never stores them -- so a minimal `Request` fixture class
                 is defined before the block (the book itself never defines
                 `Request`; it's implied pseudocode glue).
  #5 (line ~265) gguf_size_gb() -- GGUF model file size estimator + the book's
                 worked 70B/13B examples, run verbatim. BUG FOUND & FIXED: the
                 book's inline comments claimed ~38 / ~46 / ~7 GB for the three
                 examples, but the function as written actually computes
                 ~41.4 / ~49.1 / ~7.7 GB. Corrected the comments (and a matching
                 "≈38 GB" mention in the Ollama performance-profile prose) in
                 content/07-inference-serving/05-trtllm-tgi-stacks.md to match
                 the code's real output; asserts below check the corrected values.

Blocks intentionally SKIPPED (per task spec):
  #0 -- shell (TensorRT-LLM checkpoint convert / trtllm-build / CLI run)
  #2 -- shell (FP8 trtllm-build invocation)
  #3 -- shell (TGI docker run + curl)
  #6 -- shell (Ollama install/run + curl)
  #7 -- shell (LMDeploy install/convert/serve, incl. an embedded `python -c`
        one-liner -- it's one fenced bash block in the book)
  #8 -- needs-gpu (MLC-LLM `mlc_llm.build(...)` targets real GPU backends
        such as apple/m3-gpu, cuda, rocm, vulkan, webgpu; not CPU-runnable)
  #9 -- non-python (CUDA C++ paged-attention kernel pseudocode)
"""

import asyncio
import sys
import types


# ============================================================
# Glue: fake `tensorrt_llm.executor` module (network/hardware boundary mock)
# for block #1. Only the attributes the block actually touches are provided.
# ============================================================
_fake_tensorrt_llm = types.ModuleType("tensorrt_llm")
_fake_executor_mod = types.ModuleType("tensorrt_llm.executor")


class _FakeGenerationRequest:
    """Stands in for tensorrt_llm.executor.GenerationRequest."""

    def __init__(self, input_token_ids, max_new_tokens, streaming=False):
        self.input_token_ids = input_token_ids
        self.max_new_tokens = max_new_tokens
        self.streaming = streaming
        # Canned tokens a real Executor would stream back for this request.
        self._canned_tokens = [f"tok{i}" for i in range(4)]

    async def aiter_tokens(self):
        for tok in self._canned_tokens:
            await asyncio.sleep(0)  # yield control, mimicking async streaming
            yield tok


class _FakeGenerationExecutor:
    """Stands in for tensorrt_llm.executor.GenerationExecutor."""

    def __init__(self, engine_dir, executor_config):
        self.engine_dir = engine_dir
        self.executor_config = executor_config
        self.submitted = []

    @classmethod
    def create(cls, engine_dir, executor_config):
        return cls(engine_dir, executor_config)

    def submit(self, request):
        self.submitted.append(request)


_fake_executor_mod.GenerationExecutor = _FakeGenerationExecutor
_fake_executor_mod.GenerationRequest = _FakeGenerationRequest
sys.modules["tensorrt_llm"] = _fake_tensorrt_llm
sys.modules["tensorrt_llm.executor"] = _fake_executor_mod


# ============================================================
# Block #1 (line ~79) -- Minimal TensorRT-LLM Executor example
# (C++ API wrapped in Python; async request/response model)
# ============================================================
from tensorrt_llm.executor import GenerationExecutor, GenerationRequest


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

    return executor, req_a, req_b


# ============================================================
# Glue: minimal `Request` fixture for block #4.
# The book's pseudocode annotates `self.waiting: list[Request]` /
# `self.running: list[Request]` and `def schedule(self) -> list[Request]`
# without ever defining `Request` itself (it's illustrative pseudocode for a
# real Rust struct). Attribute annotations and return annotations are
# evaluated by Python at definition time, so a tiny stand-in is required for
# the class to even be importable/executable.
# ============================================================
class Request:
    """Minimal stand-in for TGI's internal Request struct."""

    def __init__(self, current_length: int, max_total_tokens: int):
        self.current_length = current_length
        self.max_total_tokens = max_total_tokens


# ============================================================
# Block #4 (line ~204) -- Simplified pseudocode illustrating TGI's waiting
# queue logic. Real implementation is in Rust + a Python model server side.
# ============================================================
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


# ============================================================
# Block #5 (line ~265) -- Estimate GGUF model file size from parameter count
# ============================================================
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


# ============================================================
# Driver -- exercise every block above with tiny CPU-safe inputs
# ============================================================

def main():
    # --- Block #1: async Executor submit + stream, against the fake backend
    executor, req_a, req_b = asyncio.run(serve_requests())
    print()  # newline after the "A: tok0 tok1 ..." stream printed above
    assert isinstance(executor, _FakeGenerationExecutor)
    assert executor.submitted == [req_a, req_b], "both requests should have been submitted"
    assert req_a.streaming is True and req_a.max_new_tokens == 100
    assert req_b.input_token_ids == [1, 890, 123, 456]
    print(f"[OK] block #1 GenerationExecutor.submit()'d {len(executor.submitted)} requests, "
          f"streamed {len(req_a._canned_tokens)} tokens for req_a")

    # --- Block #4: TGIScheduler token-budget scheduling ---------------------
    sched = TGIScheduler(max_batch_total_tokens=100)
    # Already-running request consuming some of the budget.
    sched.running.append(Request(current_length=30, max_total_tokens=40))
    # Waiting requests: one fits within the remaining budget, one doesn't.
    fits = Request(current_length=0, max_total_tokens=50)   # 30 + 50 = 80 <= 100 -> admitted
    overflow = Request(current_length=0, max_total_tokens=60)  # 80 + 60 = 140 > 100 -> stays waiting
    sched.waiting.extend([fits, overflow])

    running = sched.schedule()
    assert fits in running, "request that fits the token budget should be admitted"
    assert overflow not in running, "request that overflows the token budget should not be admitted"
    assert overflow in sched.waiting, "overflowing request should remain in the waiting queue"
    assert len(running) == 2, f"expected 2 running requests (1 pre-existing + 1 admitted), got {len(running)}"
    print(f"[OK] block #4 TGIScheduler admitted {len(running)} running requests, "
          f"{len(sched.waiting)} still waiting")

    # --- Block #5: gguf_size_gb() + the book's worked examples --------------
    size_70b_q4 = gguf_size_gb(70, 4.5)
    size_70b_q5 = gguf_size_gb(70, 5.5)
    size_13b_q4 = gguf_size_gb(13, 4.5)
    print(f"70B Q4_K_M: {size_70b_q4:.1f} GB")   # ~41 GB
    print(f"70B Q5_K_M: {size_70b_q5:.1f} GB")   # ~49 GB
    print(f"13B Q4_K_M: {size_13b_q4:.1f} GB")   # ~7.7 GB

    # Sanity-check against the book's (corrected) stated approximations.
    # NOTE: the book originally claimed ~38 / ~46 / ~7 GB here, but those
    # inline comments didn't match what gguf_size_gb() actually computes
    # (41.4 / 49.1 / 7.7 GB) -- a real content bug, fixed in the chapter
    # alongside this test (see content/07-inference-serving/05-trtllm-tgi-stacks.md).
    assert 40.5 < size_70b_q4 < 42.0, f"expected ~41 GB for 70B Q4_K_M; got {size_70b_q4:.2f} GB"
    assert 48.5 < size_70b_q5 < 49.5, f"expected ~49 GB for 70B Q5_K_M; got {size_70b_q5:.2f} GB"
    assert 7.5 < size_13b_q4 < 7.9, f"expected ~7.7 GB for 13B Q4_K_M; got {size_13b_q4:.2f} GB"
    assert size_70b_q5 > size_70b_q4, "higher bits-per-weight should mean a larger file"
    print(f"[OK] block #5 gguf_size_gb matches book's worked examples "
          f"(70B Q4_K_M={size_70b_q4:.1f}GB, 70B Q5_K_M={size_70b_q5:.1f}GB, 13B Q4_K_M={size_13b_q4:.1f}GB)")

    print("\nAll CPU-runnable blocks in 07-inference-serving/05-trtllm-tgi-stacks.md executed OK.")


if __name__ == "__main__":
    main()
