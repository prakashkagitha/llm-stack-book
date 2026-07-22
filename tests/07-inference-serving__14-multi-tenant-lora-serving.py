"""
Runnability test for content/07-inference-serving/14-multi-tenant-lora-serving.md

Orchestrator heuristic flagged block #5 (line ~275, the "vLLM offline batched
multi-LoRA" snippet) as the sole CPU-runnable block. On inspection this is
wrong: that block does `from vllm import LLM` and instantiates
`LLM(model="meta-llama/Llama-2-13b-hf", ...)`, which (a) requires the `vllm`
package, (b) downloads/loads a real 13B-parameter model from the Hugging Face
Hub (network + gated-repo auth), and (c) requires a GPU to run inference. None
of that is honestly CPU-testable, so per the hard rules (no network calls, no
GPU) it is SKIPPED below with only its import boundary guarded.

The block the heuristic marked "needs-gpu" (#6, "From Scratch: A Batched
Multi-Adapter Forward", lines ~302-467) is actually pure CPU PyTorch code --
`MultiLoRALinear` defaults to `device="cpu"`, and nothing in it touches CUDA.
It is the chapter's real executable payload (the from-scratch SGMV/BGMV math
model + adapter registry with LRU eviction + a numerical correctness check
against a per-adapter reference), so it is the block actually exercised here,
copied verbatim from the chapter with a couple of explicit asserts appended.

Block #2 (the `prefetch_adapters` / `copy_stream = torch.cuda.Stream()`
sketch, lines ~139-159) is genuinely GPU-only (it calls `torch.cuda.Stream()`
at definition time) and is SKIPPED entirely -- it is not even defined here,
since merely constructing a `torch.cuda.Stream()` would fail on a CPU-only
machine.

Summary of what happens below:
  - block #5 (vLLM offline multi-LoRA generate): SKIP(network+gpu) -- import
    guarded, not called.
  - block #2 (prefetch_adapters sketch): SKIP(gpu) -- not included.
  - block #6 (MultiLoRALinear / forward_segmented / AdapterRegistry / demo):
    EXECUTED verbatim, on CPU.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from collections import OrderedDict

# --- SKIP(network+gpu): block #5 -------------------------------------------
# The chapter's vLLM offline multi-LoRA example instantiates
# LLM(model="meta-llama/Llama-2-13b-hf", ...) which downloads real model
# weights from the Hugging Face Hub and requires a GPU to run generation.
# We only guard the import so the test file still loads if `vllm` happens to
# be absent (as it is in this CI), and we never call LLM(...) or
# llm.generate(...).
try:
    from vllm import LLM, SamplingParams  # noqa: F401
    from vllm.lora.request import LoRARequest  # noqa: F401
except Exception:
    LLM = None
    SamplingParams = None
    LoRARequest = None
# Not instantiated, not called -- see docstring above.

# --- SKIP(gpu): block #2 ----------------------------------------------------
# `copy_stream = torch.cuda.Stream()` and the `prefetch_adapters` sketch at
# lines ~139-159 require an actual CUDA device to even construct the stream
# object; there is nothing CPU-portable to run here, so it is omitted.


# ---------------------------------------------------------------------------
# Block #6 (verbatim from the chapter, lines ~302-467):
# "From Scratch: A Batched Multi-Adapter Forward"
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 1. The batched multi-adapter linear layer (the SGMV/BGMV idea, vectorized).
# ---------------------------------------------------------------------------
class MultiLoRALinear:
    """A frozen base weight W0 plus a *stack* of LoRA adapters, applied
    per-row according to a per-token adapter index. This is the math model
    of Punica's SGMV; a real kernel fuses the gather + matmuls in CUDA/Triton.
    """
    def __init__(self, d_in, d_out, max_adapters, rank, device="cpu"):
        self.d_in, self.d_out, self.rank = d_in, d_out, rank
        # Frozen base weight: [d_out, d_in]
        self.W0 = torch.randn(d_out, d_in, device=device) / d_in**0.5
        # Adapter stacks. Slot 0 is reserved as the "no adapter" identity:
        # its A,B are zero so the correction is exactly zero (base-only rows).
        # A_all: [S, r, d_in]   B_all: [S, d_out, r]
        S = max_adapters + 1
        self.A_all = torch.zeros(S, rank, d_in, device=device)
        self.B_all = torch.zeros(S, d_out, rank, device=device)
        self.scale = torch.ones(S, device=device)  # alpha/r per slot; slot 0 -> 0
        self.scale[0] = 0.0
        self.device = device

    def set_adapter(self, slot, A, B, alpha):
        """Install adapter weights into a GPU slot. A:[r,d_in] B:[d_out,r]."""
        assert slot >= 1, "slot 0 is reserved as base-only"
        self.A_all[slot].copy_(A)
        self.B_all[slot].copy_(B)
        self.scale[slot] = alpha / self.rank

    def forward(self, x, lora_idx):
        """
        x:        [B, d_in]   stacked activations (one row per token)
        lora_idx: [B] long    adapter slot for each row (0 == base only)
        returns:  [B, d_out]
        """
        # (a) Base GEMM — fully batched, peak-efficiency, adapter-independent.
        y = F.linear(x, self.W0)                      # [B, d_out]

        # (b) Gather each row's adapter matrices. In a real kernel this gather
        #     is fused; here we use advanced indexing for clarity.
        A = self.A_all[lora_idx]                       # [B, r, d_in]
        B = self.B_all[lora_idx]                       # [B, d_out, r]
        s = self.scale[lora_idx].unsqueeze(-1)         # [B, 1]

        # (c) Shrink then expand:  v = A x   (per-row mat-vec), then B v.
        #     einsum expresses the per-row low-rank product without a Python loop.
        v = torch.einsum("brd,bd->br", A, x)           # [B, r]   (down-proj)
        delta = torch.einsum("bor,br->bo", B, v)       # [B, d_out](up-proj)
        y = y + s * delta                              # rows with slot 0 add 0
        return y


# ---------------------------------------------------------------------------
# 2. A segment-sorted variant: sort the batch by adapter so identical
#    adapters are contiguous (what the scheduler does to form SGMV segments).
# ---------------------------------------------------------------------------
def forward_segmented(layer, x, lora_idx):
    """Demonstrate adapter-sorting: group rows by adapter, apply per segment,
    then scatter results back to original order. Mirrors how a real server
    sorts a continuous batch by adapter id before launching SGMV."""
    order = torch.argsort(lora_idx)                    # stable grouping
    x_sorted, idx_sorted = x[order], lora_idx[order]
    y_sorted = layer.forward(x_sorted, idx_sorted)     # same result, contiguous
    # scatter back to the caller's order
    y = torch.empty_like(y_sorted)
    y[order] = y_sorted
    return y


# ---------------------------------------------------------------------------
# 3. A tiny adapter registry with GPU-slot LRU eviction + CPU warm pool.
# ---------------------------------------------------------------------------
@dataclass
class AdapterMeta:
    A: torch.Tensor
    B: torch.Tensor
    alpha: float
    ref_count: int = 0

class AdapterRegistry:
    def __init__(self, layer: "MultiLoRALinear", n_gpu_slots: int):
        self.layer = layer
        self.n_gpu_slots = n_gpu_slots
        self.cpu_pool: dict[str, AdapterMeta] = {}     # warm pool (CPU DRAM)
        self.gpu: "OrderedDict[str,int]" = OrderedDict()  # name -> slot (LRU order)
        self.free_slots = list(range(1, n_gpu_slots + 1))  # slot 0 reserved

    def register(self, name, A, B, alpha):
        """Add an adapter to the (CPU) warm pool — the source of truth here."""
        self.cpu_pool[name] = AdapterMeta(A=A, B=B, alpha=alpha)

    def ensure_resident(self, name) -> int:
        """Return the GPU slot for `name`, loading + evicting as needed."""
        if name in self.gpu:                           # GPU hit
            self.gpu.move_to_end(name)                 # mark most-recently-used
            return self.gpu[name]
        meta = self.cpu_pool[name]                     # CPU-warm hit (else KeyError)
        if not self.free_slots:                        # need to evict an LRU adapter
            victim, vslot = next(iter(self.gpu.items()))
            if self.cpu_pool[victim].ref_count > 0:
                raise RuntimeError("LRU victim is in use; need a richer policy")
            del self.gpu[victim]
            self.free_slots.append(vslot)
        slot = self.free_slots.pop()
        self.layer.set_adapter(slot, meta.A, meta.B, meta.alpha)  # H2D copy
        self.gpu[name] = slot
        return slot

    def build_index(self, request_adapter_names):
        """Map a batch's per-request adapter names to GPU slot indices,
        ensuring each is resident first."""
        return torch.tensor([self.ensure_resident(n) for n in request_adapter_names],
                            dtype=torch.long, device=self.layer.device)


# ---------------------------------------------------------------------------
# 4. End-to-end sanity check: correctness vs an explicit per-adapter reference.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    d_in, d_out, r = 64, 48, 8
    layer = MultiLoRALinear(d_in, d_out, max_adapters=4, rank=r)
    reg = AdapterRegistry(layer, n_gpu_slots=3)        # exactly enough for this batch's
                                                        # 3 concurrent adapters; a 4th
                                                        # distinct adapter in a later batch
                                                        # would force an eviction

    # Register 3 adapters into the CPU warm pool (exactly filling GPU slots).
    refs = {}
    for name in ["contracts", "game-lore", "med-notes"]:
        A = torch.randn(r, d_in) * 0.02
        B = torch.randn(d_out, r) * 0.02
        reg.register(name, A, B, alpha=16.0)
        refs[name] = (A, B, 16.0 / r)

    # A heterogeneous batch: 6 tokens over 3 adapters + 1 base-only row.
    batch_names = ["contracts", "game-lore", "contracts",
                   "med-notes", "game-lore", "__base__"]
    x = torch.randn(len(batch_names), d_in)

    # __base__ maps to reserved slot 0 (no correction); others get real slots.
    def to_slot(n):
        return 0 if n == "__base__" else reg.ensure_resident(n)
    lora_idx = torch.tensor([to_slot(n) for n in batch_names], dtype=torch.long)

    y = layer.forward(x, lora_idx)
    y_seg = forward_segmented(layer, x, lora_idx)      # sorted path, same answer

    # Reference: compute each row independently with merged math y = W0 x + (a/r) B A x
    y_ref = torch.empty_like(y)
    for i, n in enumerate(batch_names):
        base = F.linear(x[i], layer.W0)
        if n == "__base__":
            y_ref[i] = base
        else:
            A, B, s = refs[n]
            y_ref[i] = base + s * (B @ (A @ x[i]))

    print("batched  vs reference  max err:", (y - y_ref).abs().max().item())
    print("segmented vs reference max err:", (y_seg - y_ref).abs().max().item())
    # Both errors are ~1e-6 (float rounding): the fused multi-adapter forward
    # is numerically identical to per-adapter merged math.

    # --- test-harness assertions (not in the book text, added to make this
    # an actual pass/fail check rather than just a print) ---
    err_batched = (y - y_ref).abs().max().item()
    err_segmented = (y_seg - y_ref).abs().max().item()
    assert err_batched < 1e-3, f"batched forward diverged from reference: {err_batched}"
    assert err_segmented < 1e-3, f"segmented forward diverged from reference: {err_segmented}"

    # The 3 registered adapters exactly fill the 3 GPU slots -- no eviction yet.
    assert len(reg.gpu) == reg.n_gpu_slots, "GPU slot pool should be fully occupied"
    assert len(reg.free_slots) == 0, "no free slots should remain once all 3 slots are taken"
    assert set(reg.gpu) == {"contracts", "game-lore", "med-notes"}

    # Now genuinely exercise the LRU-eviction branch in AdapterRegistry: register
    # a 4th adapter and make it resident. With the pool full and no free slots,
    # ensure_resident() must evict the least-recently-used adapter. Access order
    # so far (via move_to_end on hits) makes "contracts" the LRU victim.
    reg.register("support", torch.randn(r, d_in) * 0.02, torch.randn(d_out, r) * 0.02, alpha=16.0)
    support_slot = reg.ensure_resident("support")           # forces one eviction
    assert len(reg.gpu) == reg.n_gpu_slots, "pool stays full after eviction+load"
    assert len(reg.free_slots) == 0, "the reclaimed slot is immediately reused"
    assert "support" in reg.gpu, "the newly loaded adapter must be resident"
    assert "contracts" not in reg.gpu, "the LRU adapter must have been evicted"
    # The evicted adapter's weights still live in the CPU warm pool (cheap re-load).
    assert "contracts" in reg.cpu_pool, "eviction only frees the GPU slot, not CPU DRAM"
    # The victim's slot was recycled for the newcomer.
    assert support_slot in range(1, reg.n_gpu_slots + 1)

    print("OK: block #6 (from-scratch batched multi-adapter forward) executed and verified.")
