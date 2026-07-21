# 7.14 Multi-Tenant LoRA & Adapter Serving at Scale

Imagine you run a platform where every customer fine-tunes their own model. A legal-tech firm wants a contracts assistant; a game studio wants a lore-keeping NPC; a hospital wants a discharge-summary drafter. The naïve answer is "give each customer their own GPU running their own fine-tuned model." With a 13-billion-parameter base in bf16 weighing roughly 26 GB, that means one (or several) dedicated 80 GB GPU per tenant, sitting near-idle whenever that tenant is quiet. At a thousand tenants this is economically absurd: you are paying for thousands of GPUs to serve a request rate that, aggregated, might fit on twenty.

The escape hatch is **Low-Rank Adaptation (LoRA)**. Each tenant's fine-tune is not a fresh 26 GB of weights — it is a handful of tiny low-rank matrices, often 10–200 MB, layered *on top of* a single shared base model. If we can keep one copy of the base resident on the GPU and swap in the right small adapter per request, we can serve hundreds-to-thousands of distinct fine-tuned models from a single base deployment. The catch: a production batch contains requests for *many different adapters at once*. Serving them efficiently — without looping over adapters one at a time and destroying GPU utilization — is the central systems problem of this chapter.

This chapter assumes you understand LoRA's math from [PEFT I: LoRA, QLoRA, DoRA & The Adapter Family](../05-posttraining-alignment/03-peft-lora-qlora.html) and the inference anatomy from [The Anatomy of LLM Inference: Prefill, Decode & The KV Cache](../07-inference-serving/01-anatomy-inference.html) and [Continuous Batching & Request Scheduling](../07-inference-serving/02-continuous-batching.html). We build up from the LoRA forward pass, derive the batched heterogeneous-adapter kernel (Punica's SGMV), study the S-LoRA / LoRAX architectures and their adapter registries, hot-swapping and tiered storage, unified vs. disaggregated execution, cross-model prefix-cache reuse, and the throughput/SLO/fairness trade-offs that govern real deployments. We close with a from-scratch batched multi-adapter forward in PyTorch.

---

## 7.14.1 The LoRA Forward Pass and Why Multi-Tenancy Is Hard

Recall the LoRA reparameterization. For a frozen base weight matrix $W_0 \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$, LoRA adds a low-rank update:

$$
W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B A
$$

where $A \in \mathbb{R}^{r \times d_\text{in}}$ and $B \in \mathbb{R}^{d_\text{out} \times r}$, the rank $r \ll \min(d_\text{out}, d_\text{in})$ (typically 8–64), and $\alpha$ is a scaling hyperparameter. For an input activation $x \in \mathbb{R}^{d_\text{in}}$, the layer output is

$$
y = W_0 x + \frac{\alpha}{r} B (A x).
$$

The crucial structural fact is the **parenthesization**: we never materialize $\Delta W = BA$ (that would be a full $d_\text{out} \times d_\text{in}$ matrix, as large as the base). Instead we compute $Ax$ first — a cheap projection down to dimension $r$ — then $B(Ax)$, a projection back up. The LoRA path costs $r(d_\text{in} + d_\text{out})$ multiply-adds versus $d_\text{in} d_\text{out}$ for the base; for $r=16$ on a $4096 \times 4096$ projection, that is about 0.8% extra compute.

### The single-adapter trap: merging

If you only ever serve **one** adapter, the optimal move is to *merge* it into the base: precompute $W = W_0 + \frac{\alpha}{r}BA$ once, and from then on you run a plain dense model with zero LoRA overhead. This is what `peft`'s `merge_and_unload()` does. Merging is perfect for single-tenant serving.

But merging is a trap for multi-tenancy. If a batch contains requests for adapters $\mathcal{A}_1, \mathcal{A}_2, \dots, \mathcal{A}_k$, a merged weight can only embody one of them. You would have to either (a) run $k$ separate forward passes, one per adapter, each with batch size $\tfrac{B}{k}$ — collapsing the batch and wrecking throughput — or (b) re-merge and un-merge the base weights between micro-batches, which means rewriting tens of gigabytes of GPU memory per step. Both are catastrophic. The entire field of multi-tenant LoRA serving exists to avoid them.

### The batched-heterogeneous problem, stated precisely

Continuous batching (the foundation of vLLM/SGLang/TGI) assembles a batch of tokens from many requests at different decode positions. In multi-tenant LoRA, each *request* carries an adapter ID. So a single batched matrix multiply now looks like this: we have a stacked activation tensor $X \in \mathbb{R}^{B \times d_\text{in}}$ where row $i$ belongs to adapter $a_i \in \{1, \dots, N\}$. The base contribution $X W_0^\top$ is shared and trivially batched. The LoRA contribution is *not*:

$$
y_i = x_i W_0^\top + \frac{\alpha_{a_i}}{r_{a_i}} \, x_i A_{a_i}^\top B_{a_i}^\top.
$$

Every row may use a **different** $A$, $B$, rank $r$, and scale $\alpha$. This is a *grouped* or *segmented* matrix multiply: the batch is partitioned into segments by adapter, and each segment multiplies against its own pair of small matrices. Standard `torch.bmm` cannot express it efficiently when segment sizes are wildly uneven (one popular adapter might have 200 rows in the batch; a cold one, 1). The kernel that solves this is the heart of the chapter.

!!! note "Adapter scope"
    In practice LoRA is applied to a subset of projections — most commonly the attention $q,k,v,o$ projections, and increasingly the MLP up/gate/down projections. Each adapted linear layer gets its own $A,B$ pair, so a single adapter is really a *bundle* of dozens of $(A,B)$ matrices, one per adapted layer. When we say "adapter $a$" in a kernel, we mean its slice for the specific layer being computed.

---

## 7.14.2 Punica & SGMV: The Batched Multi-Adapter Kernel

[Punica (Chen et al., *Punica: Multi-Tenant LoRA Serving*, 2023)](https://arxiv.org/abs/2310.18547) identified the grouped matmul as the bottleneck and introduced **SGMV — Segmented Gather Matrix-Vector multiplication** (more precisely a segmented gather matrix–*matrix* multiply). The insight: rather than launching one kernel per adapter (which serializes and underuses the GPU), launch **one** kernel that processes the whole batch, with each thread block told which adapter's weights to gather.

### The data layout

Punica keeps all adapter weights in a single contiguous tensor, indexed by adapter ID:

```text
  A_all:  [N_adapters, num_layers, r, d_in]   # all "down" projections, stacked
  B_all:  [N_adapters, num_layers, d_out, r]  # all "up" projections, stacked

  For a batch of B tokens, a per-token index vector:
  lora_idx: [B]  e.g. [3, 3, 7, 0, 3, 7, ...]  # which adapter each token uses
                 (value -1 or a reserved slot means "no adapter / base only")
```

Because requests for the same adapter tend to arrive together (and the scheduler can *sort* the batch by adapter to make segments contiguous), the batch decomposes into segments:

```text
  sorted batch:  [ A=3 | A=3 | A=3 | A=7 | A=7 | A=0 ]
  segment ptrs:  seg_start = [0, 3, 5]   # adapter 3: rows 0..2, adapter 7: 3..4, adapter 0: 5
  seg_adapter  = [3, 7, 0]
```

### What SGMV computes

SGMV fuses the two LoRA matmuls with the gather. Conceptually, for each segment $s$ with adapter $a$ and rows $[\text{start}_s, \text{end}_s)$:

$$
V_{[\text{start}_s:\text{end}_s]} = X_{[\text{start}_s:\text{end}_s]} \, A_a^\top \quad (\text{shrink: } d_\text{in} \to r)
$$

$$
Y_{[\text{start}_s:\text{end}_s]} \mathrel{+}= \frac{\alpha_a}{r_a} \, V_{[\text{start}_s:\text{end}_s]} \, B_a^\top \quad (\text{expand: } r \to d_\text{out})
$$

The kernel is launched as a 2-D grid: one axis over segments, one over output tiles. Each CTA (cooperative thread array / thread block) reads its segment's adapter ID, indexes into `A_all`/`B_all` to find the right weight tile, and accumulates. Crucially, **the base matmul $XW_0^\top$ is a separate, fully-batched GEMM** that runs at peak GPU efficiency; SGMV only computes the small low-rank *additive correction* and writes it into the same output buffer.

This split is what makes the scheme cheap: 99% of the FLOPs (the base GEMM) are a dense batched operation independent of adapters, and the 1% LoRA correction is handled by a specialized grouped kernel.

{{fig:sgmv-segmented-gather-matmul}}

### Two variants: BGMV vs SGMV

Punica distinguishes the decode and prefill regimes:

- **BGMV (Batched Gather Matrix-Vector)** — used in **decode**, where each request contributes exactly *one* token per step. The batch is "one row per request," so it is a batched matrix-vector product: each row gathers a different adapter. Memory-bandwidth bound (we stream each adapter's weights to multiply by a single vector).
- **SGMV (Segmented Gather Matrix-multiply)** — used in **prefill**, where each request contributes *many* tokens (its whole prompt). Now each adapter's segment has multiple rows, so we get a real (small) GEMM per segment with arithmetic intensity high enough to be compute-bound. SGMV tiles over the rows of a segment to reuse the adapter weights loaded into shared memory.

!!! example "Worked example: LoRA correction cost in a decode step"
    Take a Llama-2-13B-shaped layer: $d_\text{in}=d_\text{out}=5120$, LoRA on $q,k,v,o$ (4 projections), rank $r=16$, scale $\alpha=32$. Suppose a decode batch has $B=64$ tokens spread over 20 distinct adapters.

    **Base path (per adapted projection):** the dense GEMM is $B \times d_\text{out} \times d_\text{in} = 64 \times 5120 \times 5120 \approx 1.68 \times 10^9$ MACs. Across 4 projections, $\approx 6.7$ GMACs.

    **LoRA correction (per adapted projection):** shrink $B \times r \times d_\text{in} = 64\times 16\times 5120 \approx 5.2\times 10^6$ MACs, expand the same, so $\approx 1.0\times 10^7$ MACs per projection, $\approx 4.2\times 10^7$ across 4 — about **0.6%** of the base FLOPs.

    But in *decode* the operation is bandwidth-bound, not FLOP-bound. The base weights for the 4 projections are $4 \times 5120 \times 5120 \times 2\,\text{bytes} \approx 210\,\text{MB}$, loaded once for the whole batch. The LoRA weights are $20\,\text{adapters} \times 4\,\text{proj} \times (B{+}A)\,\text{params} = 20 \times 4 \times (5120{\cdot}16 + 16{\cdot}5120) \times 2\,\text{bytes} \approx 52\,\text{MB}$ — about a **25% bandwidth tax** on the attention-projection matmuls, even though it is only 0.6% of the FLOPs. This is exactly why decode-time LoRA hurts more than its FLOP count suggests, and why fusing the gather (avoiding redundant reloads) matters so much.

---

## 7.14.3 S-LoRA & LoRAX: Serving Thousands of Adapters

Punica gave us the kernel. [S-LoRA (Sheng et al., *S-LoRA: Serving Thousands of Concurrent LoRA Adapters*, 2023)](https://arxiv.org/abs/2311.03285) gave us the *system*: how to keep thousands of adapters straight in memory, schedule them fairly, and not run out of GPU DRAM. LoRAX (Predibase's open-source server) and the multi-LoRA paths now in vLLM and SGLang are direct descendants.

### Unified Paging: adapters live in the KV-cache pool

The signature S-LoRA idea is **Unified Paging**. The KV cache (see [PagedAttention & KV-Cache Memory Management](../04-kernels-efficiency/06-paged-attention-kv.html)) already manages GPU DRAM as a pool of fixed-size pages allocated and freed dynamically as requests come and go. S-LoRA stuffs adapter weights into the *same* paged memory pool. Both KV pages and adapter weight tensors are variable in size and lifetime; unifying them avoids fragmentation and lets the system trade memory fluidly between "more concurrent requests" (KV) and "more resident adapters" (LoRA weights).


{{fig:lora-serving-unified-paged-pool}}


Because adapters are of heterogeneous rank, S-LoRA pads them to a common rank for the kernel or, better, uses rank-aware tiling so a rank-8 adapter does not waste a rank-64 slot. Its custom kernels (an evolution of Punica's MBGMV/SGMV) handle the non-uniform ranks directly.

### Tiered storage: GPU → CPU → disk/object store

You cannot fit thousands of adapters in GPU DRAM at once, and you do not need to: at any instant only the *active* adapters (those with in-flight requests) must be resident. S-LoRA and LoRAX organize adapters in a **storage hierarchy**:


{{fig:lora-serving-tiered-storage-hierarchy}}


When a request arrives for adapter $a$, the scheduler checks whether $a$ is GPU-resident. If not, it triggers an **asynchronous prefetch** (HBM ← CPU ← SSD/object store) while the request waits in the queue, and admits the request to a batch only once the weights have landed. The cost of a cold adapter is dominated by this transfer, not by compute.

### Hot-swapping and the prefetch pipeline

The art is **overlapping** adapter transfer with ongoing compute so swaps are invisible. A good server runs a copy engine (DMA over a separate CUDA stream) that streams the next batch's needed adapters into GPU pages while the current batch is still executing on the compute stream. Because adapters are small (tens of MB) and PCIe/NVLink bandwidth is large (tens to hundreds of GB/s), an adapter swap from CPU DRAM typically takes well under a millisecond — comfortably hidden behind a decode step.

```python
# Sketch: overlap adapter prefetch with the current forward pass.
# copy_stream does H2D DMA; compute_stream runs the model.
copy_stream = torch.cuda.Stream()

def prefetch_adapters(needed_ids, registry, gpu_pool):
    """Stream weights for `needed_ids` into GPU pages on a side stream."""
    with torch.cuda.stream(copy_stream):
        for aid in needed_ids:
            if aid in gpu_pool:                  # already resident -> skip
                continue
            cpu_weights = registry.fetch_to_cpu(aid)   # CPU/SSD/object store
            slot = gpu_pool.alloc(aid)                  # may evict an LRU adapter
            slot.copy_(cpu_weights, non_blocking=True)  # async H2D into the slot

# Each scheduler step:
#   1. pick the batch (requests + their adapter IDs)
#   2. prefetch_adapters(...) on copy_stream
#   3. run base+LoRA forward on compute_stream
#   4. compute_stream.wait_stream(copy_stream)  before the LoRA kernel reads weights
```

!!! warning "The cold-adapter SLO cliff"
    A request whose adapter is *not* resident pays the full fetch latency before its first token. If your object store is S3, a cold adapter can add 50–200 ms to time-to-first-token (TTFT). Two defenses: (1) a **warm pool** in CPU DRAM sized to your working set of adapters so most "GPU misses" are CPU hits, and (2) **admission shaping** — group cold-adapter requests so you pay the transfer once for a batch, and keep a small LRU pin for your most popular adapters. Always measure TTFT *segmented by adapter cache state* (GPU-hit / CPU-hit / cold); a healthy median can hide a brutal cold tail.

---

## 7.14.4 The Adapter Registry, Eviction & Fairness

Behind every multi-LoRA server is an **adapter registry**: the control-plane component that maps a tenant-facing adapter name to its physical weights and current residency tier, reference-counts in-flight usage, and decides what to evict.

### What the registry tracks

| Field | Purpose |
|---|---|
| `adapter_id` / name | tenant-facing handle (e.g. `acme/contracts-v3`) |
| rank $r$, target modules, $\alpha$ | kernel configuration; validated against base at load |
| location | `GPU` / `CPU` / `SSD` / `OBJECT_STORE` |
| `gpu_slot` | page handle in the unified pool when resident |
| `ref_count` | number of in-flight requests currently using it |
| `last_used`, `hit_count` | LRU/LFU eviction signals |
| checksum / version | integrity + safe hot-reload of a retrained adapter |

### Eviction policy

When the GPU pool is full and a new adapter must be loaded, the registry evicts a resident adapter with `ref_count == 0` (never one in active use). LRU is the default; LFU or a cost-aware policy (weight the recency by adapter size and fetch cost) can be better when a few adapters dominate traffic. Because an evicted adapter still lives in CPU DRAM (or at least the object store), eviction is cheap — it just frees GPU pages. This is structurally identical to KV-cache eviction in [Prefix Caching & KV-Cache Reuse](../07-inference-serving/07-prefix-caching.html), and in S-LoRA's unified pool it is literally the *same* allocator.

### Per-tenant fairness

A naïve "first-come-first-served, sort by adapter" scheduler is throughput-optimal but unfair: one tenant who floods the queue can starve everyone else, and a popular adapter that is always resident gets a latency advantage over a cold one. Production servers add fairness controls:

- **Per-tenant token-bucket rate limits** to cap any single tenant's share of the batch.
- **Max-adapters-per-batch** so a single step's LoRA kernel does not degenerate into thousands of tiny segments (each segment has fixed launch overhead; too many tanks efficiency).
- **Weighted fair queuing** across tenants so batch slots are allocated proportionally to entitlements, not arrival order.
- **Reserved/pinned adapters** for premium tenants guaranteed GPU residency (a latency SLO), traded against shared pool capacity.

There is a genuine tension here: sorting the batch by adapter maximizes kernel efficiency (few, large segments) but can violate per-request latency fairness; honoring strict FCFS order maximizes fairness but can produce a batch with many tiny adapter segments. Real schedulers interpolate — they sort *within* a fairness-bounded window.

!!! tip "Cap the adapters-per-step, not just the batch size"
    The dominant inefficiency in multi-LoRA decode is not batch size — it is *adapter cardinality* in the batch. A batch of 128 tokens over 4 adapters runs a tight SGMV; the same 128 tokens over 128 adapters runs 128 microscopic segments dominated by launch and gather overhead. Configure `max_loras` (vLLM) / max concurrent adapters per step, and let the scheduler defer overflow adapters to the next step. This single knob often moves throughput more than any kernel tuning.

---

## 7.14.5 Unified vs. Disaggregated Execution & Cross-Model Prefix Reuse

### Unified execution

The default architecture — vLLM, SGLang, LoRAX, S-LoRA — is **unified**: base GEMM and LoRA correction run on the same GPU(s) in the same forward pass. The base matmul output is computed, then the SGMV/BGMV kernel adds the per-adapter correction in place before the activation flows on. One process, one weight set, one batch. This is simplest and has the lowest latency because there is no cross-machine hop.


{{fig:lora-serving-unified-forward-sgmv}}


### Disaggregated execution

A **disaggregated** design separates concerns across machines or pools. There are two distinct axes, easily conflated:

1. **Prefill/decode disaggregation** (see [Disaggregated Prefill/Decode & Chunked Prefill](../07-inference-serving/08-disaggregated-chunked-prefill.html)) — prefill (compute-bound, batches large) runs on one pool, decode (bandwidth-bound) on another, with the KV cache shipped between them. With LoRA, *both* pools must hold the relevant adapters, and the SGMV vs BGMV split maps naturally onto the two pools.
2. **Base/adapter disaggregation** — a more exotic design where a fleet of "base servers" runs the shared dense forward and a separate "adapter service" applies LoRA corrections, or where adapters are sharded across nodes. This can help when you have *so many* high-rank adapters that they exceed any single node's memory, or when adapter compute is offloaded to cheaper hardware. The cost is an extra activation round-trip per layer, which is usually prohibitive for the decode path; it is mostly of interest at extreme adapter counts.

For the overwhelming majority of deployments, **unified execution with tiered adapter storage** wins: it keeps the hot path local and pays transfer cost only on cold misses.

### Cross-model prefix-cache reuse

Here is a subtle and valuable interaction. Prefix caching reuses KV blocks for shared token prefixes (system prompts, few-shot headers). But KV tensors depend on the *weights*, and LoRA changes the weights — so in general two requests on *different adapters* produce *different* KV tensors for the same prefix tokens. Can they share a prefix cache?

The answer hinges on **which modules the adapter touches**:

- If the LoRA does **not** adapt the $k$ and $v$ projections (e.g., it only adapts $q,o$ or the MLP), then $K$ and $V$ for the prefix are computed purely from base weights and are **identical across all such adapters**. The prefix KV cache can be shared across every tenant — a large win, since system prompts are often shared.
- If the LoRA **does** adapt $k$ or $v$, the cached KV blocks are adapter-specific. They are still cacheable, but only reusable by requests on the *same* adapter (and only if the base prefix is the same). The cache must therefore be keyed by `(prefix_hash, adapter_id)`.

```text
  Prefix-cache key strategy:
    adapter touches k or v  →  key = hash(prefix_tokens) ⊕ adapter_id   (per-adapter)
    adapter leaves k,v base →  key = hash(prefix_tokens)                (shared!)
```

Designing adapters to avoid the $k,v$ projections (when accuracy permits) is therefore not just a quality choice — it directly enables cross-tenant KV reuse. This is a concrete example of co-designing the fine-tuning recipe with the serving system. See [SGLang: RadixAttention & Structured Programs](../07-inference-serving/04-sglang-radixattention.html) for the trie-based cache this plugs into; the radix tree can store per-adapter subtrees off a shared base-prefix root.

!!! interview "Interview Corner"
    **Q:** "We serve 800 customer-specific LoRA adapters over a shared 13B base. Decode throughput is fine but p99 TTFT is terrible and very spiky. Walk me through the likely causes and fixes."

    **A:** Spiky p99 TTFT with good steady-state throughput almost always points to **cold-adapter loading on the request critical path**. The fixes, in order:

    1. **Measure TTFT segmented by adapter cache state** (GPU-resident / CPU-warm / cold-from-object-store). The spikes are the cold tail.
    2. **Add or enlarge a CPU-DRAM warm pool** sized to the working set of adapters so most GPU misses are sub-millisecond CPU hits rather than 50–200 ms object-store fetches.
    3. **Prefetch on a side CUDA stream** so adapter H2D transfer overlaps the prior batch's compute and is hidden behind a decode step.
    4. **Pin the top-K most popular adapters** in GPU memory (LRU exempt) so the bulk of traffic never misses.
    5. **Cap `max_loras` per step** — if the scheduler crams too many distinct adapters into one batch, the SGMV degenerates into many tiny segments, inflating both decode time and queueing delay that shows up as TTFT.
    6. **Shape admission**: batch cold-adapter requests so each adapter is fetched once per wave, and apply per-tenant rate limits so one tenant's burst of cold adapters doesn't evict everyone else's warm ones (cache thrashing).

    The throughput being fine while TTFT spikes is the tell: the GPU is busy and efficient; the latency is being injected in the *control plane* (registry/prefetch), not the *data plane* (kernels).

---

## 7.14.6 The vLLM & SGLang Multi-LoRA Paths

Both leading open-source engines ship production multi-LoRA support built on the Punica/S-LoRA lineage.

**vLLM** exposes LoRA as a first-class serving feature. You launch with `--enable-lora`, set `--max-loras` (max distinct adapters per *batch/step*) and `--max-cpu-loras` (the CPU warm-pool size), and bound rank with `--max-lora-rank`. Adapters can be registered statically at launch (`--lora-modules name=path ...`) or **loaded dynamically at runtime** via the API, which is what makes a true multi-tenant platform possible — tenants upload adapters and route to them by name without restarting the server. Internally vLLM uses Punica-style SGMV/BGMV kernels (and Triton variants), the paged allocator holds adapter weights, and an LRU manager handles GPU↔CPU residency. Requests carry a `LoRARequest(name, id, path)` so the scheduler knows which adapter each belongs to.

**SGLang** similarly supports multi-LoRA, sorting requests by adapter to form efficient SGMV segments and integrating adapter residency with its RadixAttention KV cache (so the cross-model prefix-reuse story above is native). Its structured-program model means a single program can fan out across adapters, and its scheduler co-optimizes the LoRA batch with prefix sharing.

```bash
# vLLM: serve a base model with multi-LoRA, dynamic loading enabled.
vllm serve meta-llama/Llama-2-13b-hf \
  --enable-lora \
  --max-loras 8 \           # up to 8 distinct adapters per scheduler step
  --max-lora-rank 64 \      # kernel sized for ranks up to 64
  --max-cpu-loras 256 \     # CPU warm pool: 256 adapters resident off-GPU
  --enable-prefix-caching   # share base-prefix KV where adapters allow

# Register / route to an adapter at request time (OpenAI-compatible API):
#   "model": "acme-contracts-v3"   where that name maps to a loaded LoRA.
```

```python
# vLLM offline batched multi-LoRA: one base, many adapters, one batch.
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(model="meta-llama/Llama-2-13b-hf",
          enable_lora=True, max_loras=4, max_lora_rank=16)

sp = SamplingParams(temperature=0.0, max_tokens=64)

# Each prompt is tagged with its own adapter; vLLM batches them together
# and applies the per-row SGMV correction in a single fused forward.
prompts = [
    ("Summarize this contract clause: ...", LoRARequest("contracts", 1, "/adapters/contracts")),
    ("Generate NPC dialogue for a dragon:", LoRARequest("game-lore", 2, "/adapters/game-lore")),
    ("Draft a discharge summary for:",      LoRARequest("med-notes", 3, "/adapters/med-notes")),
]
outs = llm.generate([p for p, _ in prompts], sp,
                    lora_request=[lr for _, lr in prompts])
```

---

## 7.14.7 From Scratch: A Batched Multi-Adapter Forward

Let us build the core mechanism end to end: a single adapted linear layer that, given a batch of activations each tagged with an adapter ID, computes the base GEMM once and adds the correct per-row LoRA correction — a CPU/GPU-portable PyTorch model of SGMV/BGMV. We then wrap it in a tiny registry with eviction to show the control plane.

```python
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from collections import OrderedDict

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
    reg = AdapterRegistry(layer, n_gpu_slots=2)        # only 2 GPU slots: forces eviction

    # Register 3 adapters into the CPU warm pool (more than GPU slots).
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
```

Running this prints max errors on the order of $10^{-6}$ — float rounding — confirming that the single batched forward over a heterogeneous mix of adapters is *exactly* equivalent to merging each adapter and running it alone, but at a fraction of the cost and with one shared base in memory. The `forward_segmented` path shows the scheduler trick of sorting by adapter; the registry shows GPU-slot LRU eviction with a CPU warm pool and a reserved base-only slot.

To turn this toy into the real thing you would: (1) replace the `einsum` gather with a fused SGMV/BGMV Triton or CUDA kernel that never materializes the gathered `A`/`B` tensors; (2) apply it to all adapted projections in every transformer block, not one layer; (3) move the registry's CPU↔GPU copies onto a side stream for overlap; and (4) wire the adapter index through the continuous-batching scheduler so it is rebuilt every step as requests join and leave.

!!! warning "Don't materialize the gathered weights"
    The `self.A_all[lora_idx]` advanced-index in the toy creates a `[B, r, d_in]` tensor — it physically copies each adapter's matrix once per row that uses it. For a popular adapter with 200 rows in the batch, that is 200 redundant copies of the same weight, ballooning memory traffic. The whole point of a real SGMV kernel is that a thread block loads each adapter's weight tile into shared memory **once** and reuses it across all rows of that adapter's segment. The toy is correct but not bandwidth-optimal; the production kernel is both.

---

## 7.14.8 Throughput, SLOs & When Not to Use Multi-LoRA

Multi-tenant LoRA serving is a throughput-per-dollar machine, but it is not free. A summary of the trade-offs:

| Dimension | Win | Cost / caveat |
|---|---|---|
| GPU memory | one base resident, adapters are tiny | adapter pool competes with KV cache for DRAM |
| Throughput | thousands of tenants on one deployment | LoRA correction adds a decode bandwidth tax (§7.14.2) |
| TTFT | warm adapters add ~0 latency | cold adapters add fetch latency (the SLO cliff) |
| Fairness | shared infra, elastic | needs explicit per-tenant scheduling/limits |
| Quality | per-tenant customization | rank ceiling; high-rank adapters cost more to serve |

**When *not* to use multi-LoRA serving:** if a single tenant dominates traffic, just *merge* their adapter and serve a dedicated dense model — you avoid the LoRA tax entirely. If adapters are high-rank (hundreds) or full fine-tunes, the "tiny adapter" assumption breaks and per-model deployment may be cheaper. And if tenants need different *base* models (not just different adapters on one base), multi-LoRA doesn't apply at all — you are back to multi-model serving and routing (see [Caching, Routing & Cost Control in Production](../12-production-mlops/03-caching-routing-cost.html)). The sweet spot is **many low-rank adapters over one shared base, with skewed-but-not-degenerate traffic** — exactly the SaaS fine-tuning platform we opened with.

For the broader economics of latency vs. throughput vs. cost that frame these decisions, see [Inference Economics: Latency, Throughput & Cost](../07-inference-serving/12-inference-economics.html) and the system-design view in [Designing an LLM Serving System](../12-production-mlops/01-serving-system-design.html).

---

!!! key "Key Takeaways"
    - Merging a LoRA into the base is optimal for *one* adapter but fatal for multi-tenancy; keep the LoRA path separate so a single batch can serve many adapters at once.
    - The core kernel problem is a *grouped/segmented* matmul: one shared base GEMM plus a per-row low-rank correction where each row may use a different $A,B,r,\alpha$. Punica's SGMV (prefill) and BGMV (decode) fuse the gather with the small matmuls.
    - In decode, the LoRA correction is bandwidth-bound: it can be ~0.6% of the FLOPs yet a 20–30% bandwidth tax because each adapter's weights must be streamed — fusing the gather to load each weight once is what makes it cheap.
    - S-LoRA's Unified Paging puts adapters in the same paged DRAM pool as the KV cache; tiered storage (GPU→CPU→SSD→object store) keeps only active adapters on the GPU, with async prefetch hiding swaps behind compute.
    - The adapter registry is the control plane: name→weights mapping, residency tier, ref-counting, and LRU/LFU eviction (never evict a `ref_count>0` adapter).
    - Cold adapters create the p99 TTFT cliff; defend with a CPU warm pool, side-stream prefetch, pinned top-K adapters, and capping adapters-per-step (`max_loras`).
    - Cross-tenant prefix-cache reuse is possible when the adapter does *not* touch the $k,v$ projections — co-design the fine-tune to keep KV base-computed and share system-prompt KV across all tenants.
    - vLLM (`--enable-lora`, `--max-loras`, `--max-cpu-loras`) and SGLang ship Punica/S-LoRA-style multi-LoRA with dynamic runtime adapter loading — the foundation of a real fine-tuning SaaS.

---

!!! sota "State of the Art & Resources (2026)"
    Multi-tenant LoRA serving has matured rapidly from the 2023 Punica/S-LoRA papers into production-grade support in every major inference engine; the open frontier is disaggregated LoRA execution, cross-model KV reuse, and co-optimizing adapter eviction with KV-cache management.

    **Foundational work**

    - [Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021)](https://arxiv.org/abs/2106.09685) — the reparameterization that makes multi-tenant adapter serving tractable.
    - [Chen et al., *Punica: Multi-Tenant LoRA Serving* (2023)](https://arxiv.org/abs/2310.18547) — introduces SGMV/BGMV, the grouped gather matmul that lets one kernel serve a heterogeneous-adapter batch.

    **Recent advances (2023–2026)**

    - [Sheng et al., *S-LoRA: Serving Thousands of Concurrent LoRA Adapters* (2023)](https://arxiv.org/abs/2311.03285) — Unified Paging puts adapters and KV pages in one pool; tiered storage enables serving thousands of adapters from a single GPU cluster.
    - [Li et al., *CaraServe: CPU-Assisted and Rank-Aware LoRA Serving* (2024)](https://arxiv.org/abs/2401.11240) — early-starts prefill on CPU while the adapter streams to GPU, hiding cold-adapter load latency.
    - [Wu et al., *dLoRA: Dynamically Orchestrating Requests and Adapters for LoRA LLM Serving* (OSDI 2024)](https://www.usenix.org/conference/osdi24/presentation/wu-bingyang) — dynamically merges and unmerges adapters and migrates requests across replicas to balance load.
    - [Zhang et al., *Improving Multi-LoRA Serving via Efficient LoRA and KV Cache Management* (2025)](https://arxiv.org/abs/2505.03756) — joint adapter + KV cache placement (FASTLIBRA) cuts TTFT by ~63% over prior systems.

    **Open-source & tools**

    - [punica-ai/punica](https://github.com/punica-ai/punica) — reference SGMV/BGMV CUDA kernels and multi-LoRA serving system from the Punica paper.
    - [predibase/lorax](https://github.com/predibase/lorax) — production-ready multi-LoRA inference server with dynamic adapter loading, tiered weight caching, and OpenAI-compatible API.
    - [vLLM — LoRA Adapters](https://docs.vllm.ai/en/latest/features/lora/) — official docs for `--enable-lora`, `--max-loras`, `--max-cpu-loras`, and dynamic runtime adapter registration.

    **Go deeper**

    - [LMSYS Blog — *Recipe for Serving Thousands of Concurrent LoRA Adapters* (2023)](https://www.lmsys.org/blog/2023-11-15-slora/) — accessible walkthrough of S-LoRA's design, benchmarks, and trade-offs against PEFT/vLLM baselines.

## Further Reading

- Chen, L. et al. "Punica: Multi-Tenant LoRA Serving." *MLSys 2024 / arXiv:2310.18547*. — Introduces SGMV/BGMV, the batched gather matmul for heterogeneous adapters over a shared base.
- Sheng, Y. et al. "S-LoRA: Serving Thousands of Concurrent LoRA Adapters." *arXiv:2311.03285*, 2023. — Unified Paging, tiered adapter storage, and rank-aware kernels for thousands of adapters.
- Hu, E. et al. "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR 2022*. — The original low-rank reparameterization underpinning everything in this chapter.
- Kwon, W. et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." *SOSP 2023*. — The paged allocator S-LoRA's Unified Paging extends to adapter weights.
- Zheng, L. et al. "SGLang: Efficient Execution of Structured Language Model Programs." *arXiv:2312.07104*, 2023. — RadixAttention and the structured-program serving model that hosts SGLang's multi-LoRA path.
- Predibase, "LoRAX: Multi-LoRA inference server." Open-source repository — production reference implementation of dynamic adapter loading and tiered storage.
- vLLM documentation, "Using LoRA adapters" — configuration and the runtime dynamic-loading API for multi-tenant adapter serving.
