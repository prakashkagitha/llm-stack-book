"""
Runnable-code test for content/07-inference-serving/11-multi-gpu-inference.md

Blocks tested (assembled in chapter order):
  - block #3 (line ~354): Illustrative DP request router using asyncio
    (DPReplica, LeastLoadedRouter, demo_dp) — the book's own demo is called
    verbatim (the `asyncio.run(demo_dp())` line is commented out in the
    chapter with "# uncomment to run"; we uncomment it here to actually
    execute it).

Skipped (per the task's heuristic classification):
  - block #0 (line ~47):  ColParallelLinear / RowParallelLinear — needs
    `torch.distributed` process-group init (dist.all_reduce) to actually run;
    genuinely needs a multi-GPU/multi-process collective backend.
  - block #1 (line ~154): PipelineStage / pipeline_forward_inference /
    build_pipeline_example — needs-gpu per the harness heuristic. (It is
    CPU-constructible in principle, but nn.TransformerEncoderLayer stacks of
    80 layers times running the full pipeline is flagged needs-gpu by the
    classifier and is skipped here to respect the given block classification.)
  - block #2 (line ~256): expert_parallel_forward and helpers — needs
    `torch.distributed` process group (dist.get_world_size/get_rank) to run;
    genuinely needs multi-GPU EP setup. Also relies on `_all_to_all_dispatch`
    / `_all_to_all_gather` placeholders that are explicitly non-functional
    ("simplified") stand-ins for real collectives, not runnable logic.
  - block #4 (line ~529): `from vllm import LLM, SamplingParams` — needs GPU
    + the vllm package + a real model download; also a network/model-hub call.
  - block #5 (line ~558): shell (`ray start`, `python -m vllm...` CLI) — not
    Python.
  - block #6 (line ~578): shell (`python -m sglang.launch_server` CLI) — not
    Python.

No network / external-API calls are used in the tested block (asyncio.sleep
only). No bugs were found in the book's code; the tested block runs verbatim
once its own commented-out demo invocation is uncommented.
"""

import asyncio
import random
from typing import List

random.seed(0)


# =====================================================================
# Block #3 (line ~354) — Illustrative DP request router using asyncio
# =====================================================================

class DPReplica:
    """Represents a single model replica (TP+PP shard group)."""

    def __init__(self, replica_id: int):
        self.replica_id = replica_id
        self._queue_depth = 0  # active requests

    async def generate(self, prompt: str, max_tokens: int) -> str:
        self._queue_depth += 1
        # Simulate inference latency (proportional to output tokens)
        await asyncio.sleep(max_tokens * 0.001)
        self._queue_depth -= 1
        return f"[replica={self.replica_id}] output for: {prompt[:20]}..."

    @property
    def load(self) -> int:
        return self._queue_depth


class LeastLoadedRouter:
    """Route each request to the least-loaded replica."""

    def __init__(self, replicas: List[DPReplica]):
        self.replicas = replicas

    async def route(self, prompt: str, max_tokens: int) -> str:
        # Pick the replica with the fewest in-flight requests
        replica = min(self.replicas, key=lambda r: r.load)
        return await replica.generate(prompt, max_tokens)


async def demo_dp():
    replicas = [DPReplica(i) for i in range(4)]
    router = LeastLoadedRouter(replicas)
    # Use fewer prompts / smaller max_tokens than the book's illustration so
    # the test stays well under the runtime budget (book used 20 prompts x
    # 100 max_tokens; kept identical here since it's already fast: 20 * 0.1s
    # of *simulated* async sleep across 4 replicas concurrently, ~0.5s wall).
    prompts = [f"Explain concept #{i}" for i in range(20)]
    # Dispatch all 20 requests concurrently
    tasks = [router.route(p, max_tokens=100) for p in prompts]
    results = await asyncio.gather(*tasks)
    # All replicas contribute; each individual request is fast
    print(f"Served {len(results)} requests across {len(replicas)} replicas")
    return results


def test_block_3():
    results = asyncio.run(demo_dp())  # uncommented from the book's demo
    assert len(results) == 20
    # Every result string must come from one of the 4 replicas and must
    # embed the truncated prompt, exactly matching DPReplica.generate's format.
    seen_replicas = set()
    for i, r in enumerate(results):
        assert r.startswith("[replica="), r
        assert "output for:" in r
        replica_id = int(r.split("replica=")[1].split("]")[0])
        assert 0 <= replica_id < 4
        seen_replicas.add(replica_id)
        expected_prompt_prefix = f"Explain concept #{i}"[:20]
        assert expected_prompt_prefix in r
    # Least-loaded routing across 20 concurrent same-cost requests among 4
    # replicas should spread load across more than just one replica.
    assert len(seen_replicas) > 1, seen_replicas
    print("test_block_3 (DP router demo) passed:", results[:2], "...")


if __name__ == "__main__":
    test_block_3()
    print("\nAll tests passed.")
