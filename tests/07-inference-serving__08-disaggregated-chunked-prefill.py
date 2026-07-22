"""
Runnability test for content/07-inference-serving/08-disaggregated-chunked-prefill.md

Tests the 4 heuristically CPU-runnable Python blocks from the chapter,
concatenated in chapter order (later blocks are independent of earlier ones
here, but are kept in document order for fidelity):

    - block #0 (line ~93)  -- layer-wise KV transfer pseudocode:
                              PrefillWorker / DecodeWorker
    - block #1 (line ~218) -- chunked_prefill_attention (partial-KV causal
                              attention for a prefill chunk)
    - block #4 (line ~368) -- SequenceState / ChunkedPrefillScheduler
                              (minimal end-to-end chunked-prefill scheduler)
    - block #5 (line ~528) -- METRICS dict (monitoring metric names/descriptions)

Skipped:
    - block #2 (line ~314, adaptive_chunk_size) -- heuristically flagged as a
      fragment; not in the requested set of 4 blocks to test.
    - block #3 (line ~353) -- a YAML config snippet (vllm serve flags), not
      Python.

No network or GPU is used. Block #0's PrefillWorker/DecodeWorker classes are
written against a "hypothetical RPC / RDMA abstraction" (the chapter's own
words) -- kv_sender / kv_receiver / paged_kv_manager / model are all
injected dependencies, exactly as the chapter designs them. To execute the
class bodies faithfully on CPU we supply tiny fake implementations of those
collaborators (an in-memory KV channel and a toy model with random weights)
and drive the real PrefillWorker/DecodeWorker code through them.
"""

import math

import torch

# =============================================================================
# Block #0 (line ~93): layer-wise KV transfer from prefill worker to decode
# worker. Verbatim from the chapter (pseudocode, "assumes a hypothetical
# RPC / RDMA abstraction").
# =============================================================================

# (chapter also imports `from typing import Tuple` here; Tuple is unused in
# the block body but we keep the import for fidelity)
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

            # Fire-and-forget async send -- does NOT block prefill forward pass
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


# =============================================================================
# Block #1 (line ~218): chunked_prefill_attention -- attention over a chunk
# of prefill tokens against a partially-filled KV cache. Verbatim from the
# chapter.
# =============================================================================

import torch.nn.functional as F


def chunked_prefill_attention(
    q_chunk: torch.Tensor,    # [C, n_heads, d_head] -- queries for current chunk
    k_full: torch.Tensor,     # [T_past + C, n_kv_heads, d_head] -- all keys so far
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
    # (omitted for brevity -- same as decode)

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


# =============================================================================
# Block #4 (line ~368): A Minimal End-to-End Chunked Prefill Scheduler.
# Verbatim from the chapter.
# =============================================================================

import dataclasses
from collections import deque
from typing import List, Optional


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
    Does NOT implement actual model calls -- shows scheduling decisions only.
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


# =============================================================================
# Block #5 (line ~528): Example monitoring metrics for a disaggregated
# system. Verbatim from the chapter (a plain data dict -- pseudocode for
# wiring into Prometheus/OpenTelemetry).
# =============================================================================

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


# =============================================================================
# Test glue -- fixtures + calls that actually EXECUTE the blocks above.
# =============================================================================

def test_prefill_decode_workers():
    """
    Exercises block #0: builds a tiny fake model + in-memory KV channel +
    paged KV manager (the "hypothetical RPC/RDMA abstraction" the chapter's
    pseudocode is written against), then drives the real PrefillWorker and
    DecodeWorker classes end to end on CPU.
    """
    torch.manual_seed(0)

    d_model = 8
    n_layers = 3
    n_kv_heads = 2
    d_head = 4
    vocab_size = 16
    eos_id = -1  # unreachable by argmax over vocab_size tokens -> loop runs to completion

    class FakeLayer:
        def forward_with_kv(self, x):
            # x: [1, T, d_model] -> return (unchanged x, kv_cache)
            T = x.shape[1]
            kv_cache = torch.randn(2, T, n_kv_heads, d_head)
            return x, kv_cache

    class FakeModel:
        def __init__(self):
            self.layers = [FakeLayer() for _ in range(n_layers)]
            self.eos_id = eos_id
            self.lm_head_w = torch.randn(d_model, vocab_size)

        def embed(self, input_ids):
            T = input_ids.shape[1]
            return torch.randn(1, T, d_model)

        def lm_head(self, x):
            return x @ self.lm_head_w

        def decode_step(self, cur_token, slot):
            # Pure decode step over the allocated paged-KV slot.
            assert "kv_caches" in slot and len(slot["kv_caches"]) == n_layers
            return torch.randn(1, vocab_size)

    class FakeKVChannel:
        """In-memory stand-in for the RDMA/NVLink send/receive handle."""
        def __init__(self):
            self.store = {}

        def send_async(self, request_id, layer_idx, kv):
            self.store.setdefault(request_id, {})[layer_idx] = kv

        def collect(self, request_id):
            per_layer = self.store[request_id]
            return [per_layer[i] for i in sorted(per_layer.keys())]

    class FakePagedKVManager:
        def allocate(self, request_id, kv_caches):
            return {"request_id": request_id, "kv_caches": kv_caches}

    model = FakeModel()
    channel = FakeKVChannel()

    prefill_worker = PrefillWorker(model, channel)
    input_ids = torch.randint(0, vocab_size, (1, 10))
    first_token = prefill_worker.prefill_and_stream_kv(input_ids, request_id="req-0")

    assert first_token.shape == (1,)
    # All layers' KV caches were streamed to the channel.
    assert len(channel.store["req-0"]) == n_layers

    decode_worker = DecodeWorker(model, channel, FakePagedKVManager())
    generated = decode_worker.receive_kv_and_decode(
        request_id="req-0", first_token=first_token, max_new_tokens=5,
    )

    assert generated[0] == first_token.item()
    assert len(generated) == 5  # eos_id is unreachable, so it runs to max_new_tokens
    print(f"test_prefill_decode_workers: OK (generated {len(generated)} tokens: {generated})")


def test_chunked_prefill_attention():
    """Exercises block #1 with tiny shapes and checks masking correctness."""
    torch.manual_seed(0)

    C = 4
    T_past = 6
    n_heads = 2
    d_head = 8
    T_total = T_past + C

    q_chunk = torch.randn(C, n_heads, d_head)
    k_full = torch.randn(T_total, n_heads, d_head)
    v_full = torch.randn(T_total, n_heads, d_head)
    scale = 1.0 / math.sqrt(d_head)

    out = chunked_prefill_attention(q_chunk, k_full, v_full, T_past, C, scale)

    assert out.shape == (C, n_heads, d_head)
    assert torch.isfinite(out).all()

    # Sanity-check the masking logic directly: query i can only "see" keys
    # 0 .. T_past+i. Verify by recomputing attention weights for query 0
    # restricted to the visible window equals a manual softmax over that
    # window only.
    # Manually recompute scores for query index 0 (visible keys: 0..T_past)
    q0 = q_chunk[0]  # [n_heads, d_head]
    visible = T_past + 1
    k_vis = k_full[:visible]  # [visible, n_heads, d_head]
    scores0 = torch.einsum("hd,thd->ht", q0, k_vis) * scale
    attn0 = F.softmax(scores0, dim=-1)
    v_vis = v_full[:visible]
    expected0 = torch.einsum("ht,thd->hd", attn0, v_vis)
    assert torch.allclose(out[0], expected0, atol=1e-5), (out[0], expected0)
    print("test_chunked_prefill_attention: OK (causal masking verified for query 0)")


def test_chunked_prefill_scheduler():
    """Exercises block #4: drives requests through waiting -> prefilling ->
    decoding, verifying chunking and admission logic."""
    scheduler = ChunkedPrefillScheduler(
        chunk_size=4, max_decode_seqs=8, max_batched_tokens=6,
    )

    prompt = list(range(100, 110))  # 10 tokens
    scheduler.add_request(seq_id=1, prompt_ids=prompt)

    # Round 1: budget=6, chunk_size=4 -> admits seq 1, computes 4 tokens,
    # moves to `prefilling` (4 < 10).
    batch1 = scheduler.schedule()
    assert len(batch1["prefill_chunks"]) == 1
    assert batch1["prefill_chunks"][0]["seq_id"] == 1
    assert batch1["prefill_chunks"][0]["token_ids"] == prompt[0:4]
    assert len(scheduler.prefilling) == 1
    assert len(scheduler.decoding) == 0

    # Round 2: continues the partially-prefilled sequence with another
    # 4-token chunk (6 tokens computed so far, 4 remaining).
    batch2 = scheduler.schedule()
    assert batch2["prefill_chunks"][0]["start_pos"] == 4
    assert batch2["prefill_chunks"][0]["token_ids"] == prompt[4:8]
    assert len(scheduler.prefilling) == 1
    assert len(scheduler.decoding) == 0

    # Round 3: finishes prefill (8 + 2 remaining = 10), moves to decoding.
    batch3 = scheduler.schedule()
    assert batch3["prefill_chunks"][0]["start_pos"] == 8
    assert batch3["prefill_chunks"][0]["token_ids"] == prompt[8:10]
    assert len(scheduler.prefilling) == 0
    assert len(scheduler.decoding) == 1
    assert scheduler.decoding[0].seq_id == 1

    # Round 4: sequence is now pure decode -- consumes 1 token of budget.
    batch4 = scheduler.schedule()
    assert batch4["decode"] == [1]
    assert batch4["prefill_chunks"] == []

    scheduler.mark_decode_finished(1)
    assert scheduler.decoding == []
    print("test_chunked_prefill_scheduler: OK (waiting -> prefilling -> decoding -> finished)")


def test_metrics_dict():
    """Exercises block #5: the METRICS dict is well-formed and usable."""
    assert isinstance(METRICS, dict)
    assert len(METRICS) == 12
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in METRICS.items())
    assert "ttft_p99_ms" in METRICS and "itl_p99_ms" in METRICS
    print(f"test_metrics_dict: OK ({len(METRICS)} metrics defined)")


if __name__ == "__main__":
    test_prefill_decode_workers()
    test_chunked_prefill_attention()
    test_chunked_prefill_scheduler()
    test_metrics_dict()
    print("\nAll book-code checks passed for 07-inference-serving/08-disaggregated-chunked-prefill.md.")
