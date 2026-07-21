# 7.13 Serving Mixture-of-Experts: Expert Parallelism & All-to-All Inference

A dense 70-billion-parameter model and a sparse 671-billion-parameter Mixture-of-Experts (MoE) model can have *identical* per-token FLOPs. DeepSeek-V3 has 671B total parameters but activates only ~37B per token; Llama-4 Maverick has ~400B total but ~17B active; Qwen3-235B-A22B activates ~22B of 235B. The promise of MoE — explained architecturally in [Mixture-of-Experts (MoE) Architectures](../02-transformer/09-mixture-of-experts.html) — is that *quality tracks total parameters while compute tracks active parameters*. That promise is what makes these models attractive to *train*. But it creates a serving problem that has almost nothing in common with serving a dense model of equivalent speed.

The trouble is memory and movement, not arithmetic. Those 671B parameters still have to *live* on GPUs — roughly 671 GB in FP8, over 1.3 TB in BF16 — even though any single token touches a tiny fraction of them. And because experts are scattered across dozens of GPUs, every MoE layer becomes a two-phase **all-to-all** communication: send each token to the GPUs holding its chosen experts (*dispatch*), then send the results back (*combine*). A dense model with the same active parameter count does a couple of cheap all-reduces per layer; a large MoE does two cluster-wide shuffles whose cost is set by the *slowest, most overloaded* GPU in the group. Serving MoE well is the art of making that shuffle cheap and keeping every expert busy.

This chapter is the inference-specific companion to the architecture chapter and to [Multi-GPU & Multi-Node Inference](../07-inference-serving/11-multi-gpu-inference.html), which introduced expert parallelism (EP) as one of four parallelism axes. Here we go deep on EP at decode time: the all-to-all kernel itself (DeepEP), hot-expert skew and why it is worse in serving than in training, the attention-DP + expert-EP hybrid layout that frontier stacks now use, prefill-vs-decode EP asymmetry, expert and KV offloading, and the economics of "dense-equivalent throughput." We assume you know prefill/decode and the KV cache ([The Anatomy of LLM Inference](../07-inference-serving/01-anatomy-inference.html)) and collective communication ([Parallel Computing & Collective Communication](../01-foundations/09-parallel-collectives.html)).

---

## Why MoE Serving Is a Different Problem

Start from the resource that dominates decode. A single decode step is **memory-bandwidth-bound**: the GPU must stream the weights it needs through its compute units, and at batch sizes typical of low-latency serving it does very little arithmetic per byte loaded. For a *dense* model the working set is the whole weight matrix every step. For an MoE model the per-token working set is small — the router, the attention block, the shared expert (if any), and the $k$ routed experts the token selected — but the *resident* working set is enormous because **every expert must be in memory in case some token in the batch picks it**.

This is the first inversion. In a dense model, parameters loaded $\approx$ parameters resident. In an MoE model, with $E$ experts and top-$k$ routing, a single token loads $k/E$ of the expert weights but the *batch* may load all of them. Consider a batch of $B$ tokens, $E=256$ experts, $k=8$. Expected distinct experts touched is

$$
\mathbb{E}[\text{distinct experts}] = E\left(1 - \left(1 - \tfrac{k}{E}\right)^{B}\right).
$$

For $B=64$, $k=8$, $E=256$ this is $256(1-(1-1/32)^{64})\approx 256(1-0.132)\approx 222$ — already **87% of all experts** touched by a modest batch. So even though each *token* is sparse, a realistic decode *batch* is nearly dense in expert coverage. You cannot avoid keeping (almost) all experts resident and ready.

{{fig:moe-serving-sparse-token-dense-batch}}

The second inversion is communication. EP places experts on different GPUs; routing is data-dependent and changes every token. The required primitive is **all-to-all**: GPU $i$ has a variable number of tokens destined for each expert, hence for each GPU. Unlike the all-reduce of tensor parallelism (TP) — whose volume is fixed at $\sim 2\,d_{\text{model}}$ bytes per token per layer regardless of routing — all-to-all volume is $\sim k\,d_{\text{model}}$ bytes per token per layer *and* is **load-imbalanced and irregular**: the message sizes are unknown until the router fires.


{{fig:moe-serving-dense-tp-vs-moe-ep}}


The third inversion is *who is the bottleneck*. In dense decode, the bottleneck is HBM bandwidth on a balanced set of identical GPUs. In MoE decode, the bottleneck is whichever GPU happens to hold a **hot expert** that the current batch over-selects, because the all-to-all cannot complete until that GPU finishes its disproportionate share. Routing skew, which is a mild efficiency annoyance during training (you fix it with an auxiliary load-balance loss; see [Mixture-of-Experts Architectures](../02-transformer/09-mixture-of-experts.html)), becomes a **tail-latency weapon** in serving, where each decode step is a synchronization barrier.

!!! note "Aside: 'dense-equivalent' is the right mental model"

    Throughout this chapter, compare an MoE model to the *dense model with the same active parameter count*, not the same total. DeepSeek-V3 (37B active) should be benchmarked against a hypothetical dense 37B, not a dense 671B. The MoE wins on quality-per-FLOP; the question this chapter answers is how much of that win survives the serving overhead — the all-to-all, the skew, and the memory to hold all experts. A well-tuned large-EP deployment recovers most of it; a naive single-node MoE deployment can throw it all away.

---

## Expert Parallelism Mechanics: Dispatch, Compute, Combine

Let us pin down the layout. Suppose $E$ experts spread over $G$ GPUs in the EP group, $E/G$ experts per GPU (assume it divides). Each GPU also holds a replica of the non-expert parts (attention, router, shared expert, norms) — or, in the hybrid layout of the next section, a *shard* of attention. A token resident on GPU $i$ is processed as:

1. **Router** runs locally; produces top-$k$ expert ids and gate weights for each local token.
2. **Dispatch (all-to-all #1):** tokens are permuted so that all tokens destined for expert $e$ are contiguous, then sent to the GPU owning $e$. Because token-to-GPU counts vary, this is an *all-to-all-v* (variable length).
3. **Expert compute:** each GPU runs its local experts' FFNs over the tokens it received — a grouped GEMM (one matmul per expert, batched).
4. **Combine (all-to-all #2):** expert outputs are sent back to each token's home GPU and **weighted-summed** by the gate weights. (With top-$k>1$, a token's home GPU receives $k$ partial results to combine.)

Here is a minimal but correct single-process reference that makes the data movement explicit. It simulates EP across `G` virtual ranks with plain tensors so you can run it on one GPU or CPU and inspect the permutations.

```python
import torch
import torch.nn.functional as F

def moe_ep_layer(x, router_w, expert_w1, expert_w2, k, G):
    """
    Reference EP forward for ONE MoE layer, simulating G ranks in-process.
      x          : (T, d)         tokens for the whole EP group (gathered for clarity)
      router_w   : (E, d)         router weights
      expert_w1  : (E, d, h)      per-expert up-projection
      expert_w2  : (E, h, d)      per-expert down-projection
      k          : top-k          experts per token
      G          : number of EP ranks; expert e lives on rank (e // (E//G))
    Returns y : (T, d). The point is to expose dispatch/compute/combine, not speed.
    """
    T, d = x.shape
    E = router_w.shape[0]
    experts_per_rank = E // G

    # ---- 1. ROUTER (runs on each token's home rank) ----
    logits = x @ router_w.t()                      # (T, E)
    gate = F.softmax(logits, dim=-1)               # softmax over all experts
    topv, topi = gate.topk(k, dim=-1)              # (T, k) weights and expert ids
    topv = topv / topv.sum(-1, keepdim=True)       # renormalize the chosen-k weights

    y = torch.zeros_like(x)

    # ---- 2-4. For each rank, gather its tokens (DISPATCH), compute, scatter back (COMBINE)
    for r in range(G):
        lo, hi = r * experts_per_rank, (r + 1) * experts_per_rank
        # Which (token, slot) pairs route to an expert owned by rank r?
        mask = (topi >= lo) & (topi < hi)          # (T, k) boolean
        if not mask.any():
            continue
        tok_idx, slot_idx = mask.nonzero(as_tuple=True)   # variable length -> all-to-all-v
        local_eid = topi[tok_idx, slot_idx] - lo          # expert index within this rank
        gw = topv[tok_idx, slot_idx]                      # gate weight for the combine

        # Grouped expert compute: loop experts on this rank (real kernels fuse this GEMM)
        for le in range(experts_per_rank):
            sel = (local_eid == le)
            if not sel.any():
                continue
            rows = tok_idx[sel]
            xin = x[rows]                                  # tokens sent to this expert
            hmid = F.gelu(xin @ expert_w1[lo + le])        # up-proj + act
            out = hmid @ expert_w2[lo + le]                # down-proj
            # COMBINE: weighted scatter-add back to the home token
            y.index_add_(0, rows, out * gw[sel].unsqueeze(-1))
    return y

# --- tiny smoke test ---
torch.manual_seed(0)
T, d, h, E, k, G = 12, 16, 32, 8, 2, 4
x  = torch.randn(T, d)
rw = torch.randn(E, d) * 0.1
w1 = torch.randn(E, d, h) * (d ** -0.5)
w2 = torch.randn(E, h, d) * (h ** -0.5)
y = moe_ep_layer(x, rw, w1, w2, k, G)
print(y.shape)   # torch.Size([12, 16])
```

The `mask.nonzero` step is the heart of the cost model: its length on each rank is **how many tokens that rank must process**, and it varies wildly with routing. In a real distributed kernel, steps 2 and 4 are `torch.distributed.all_to_all_single` calls with per-rank split sizes computed from the router output. The grouped GEMM in step 3 is a single fused kernel (e.g., CUTLASS grouped GEMM, or Triton's `grouped_matmul`) that runs all local experts' matmuls with one launch, padding or masking ragged group sizes.

### The Two Costs: Bytes Moved and the Barrier

Per MoE layer, the all-to-all moves about $k\,d_{\text{model}}$ bytes per token *out* (dispatch) and the same back (combine), times bytes-per-element. For BF16, $d_{\text{model}}=7168$ (DeepSeek-V3), $k=8$:

$$
\text{dispatch bytes/token} = k \cdot d_{\text{model}} \cdot 2 = 8 \cdot 7168 \cdot 2 \approx 115\,\text{KB}.
$$

That is per token, per MoE layer, in *each* direction — and DeepSeek-V3 has 58 MoE layers. The volume is not the killer on fast interconnect (NVLink at ~900 GB/s, or 8×400 Gb/s InfiniBand per node); the killer is that **each all-to-all is a barrier**. Decode is one token per request per step, so the batch must cross the all-to-all 2× per MoE layer, and the step cannot finish until the last byte from the slowest rank arrives. Latency, not bandwidth, is what large-EP engineering fights — which is exactly what DeepEP is built for.

---

## DeepEP: A Production All-to-All Kernel

`all_to_all_single` from a generic NCCL build works, but it leaves a lot on the table for MoE decode: it does not overlap dispatch with router computation, it does not exploit the two-tier topology (NVLink within a node, RDMA across nodes), and it synchronizes the GPU with the host to compute split sizes. **DeepEP** (released by DeepSeek alongside DeepSeek-V3) is a purpose-built expert-parallel communication library that addresses all three. The mechanisms generalize to any large-EP serving stack (vLLM and SGLang both integrate DeepEP-style kernels), so they are worth understanding even if you never call the library directly.

DeepEP provides two families of kernels:

- **High-throughput (normal) kernels** for *prefill* and training, where batches are large and you want to saturate bandwidth. These do **NVLink+RDMA forwarding**: a token going to a remote node's expert is sent over RDMA once, then forwarded over NVLink to the right GPU inside that node, instead of doing $G$ independent point-to-point sends. This is the all-to-all analogue of a hierarchical all-reduce.
- **Low-latency kernels** for *decode*, where batches are tiny and the all-to-all latency directly sets time-per-output-token (TPOT). These use **pure RDMA with no NVLink hop** (lower latency, fewer synchronization points) and a **hook-based overlap** scheme: the kernel returns control to Python before the receive completes, lets you run unrelated compute (the attention of the next micro-step, or the router), and you call a hook later that consumes the arrived data. This hides the all-to-all latency *behind* useful work — communication and computation truly overlap rather than serialize.

{{fig:moe-serving-alltoall-overlap-timeline}}

Two further tricks matter:

1. **FP8 dispatch.** DeepEP can quantize the dispatched activations to FP8 (E4M3) on the fly, halving the bytes moved versus BF16. The combine comes back in BF16 (gate-weighted sums are precision-sensitive). This nearly halves dispatch latency at negligible quality cost — the same logic as FP8 KV cache.
2. **Communication–computation overlap via the SM-free design.** The low-latency kernels are designed to use few or no streaming multiprocessors for the RDMA path (the NIC does the work), leaving the SMs free to compute experts on already-arrived tokens. The classic DeepSeek serving picture is a software pipeline where, at any instant, one micro-batch's tokens are *in flight* over RDMA while another micro-batch's experts are *computing*.

```python
# Sketch of the low-latency decode pattern DeepEP enables (pseudo-API).
# The real library: deep_ep.Buffer.low_latency_dispatch / _combine with a hook.

def moe_decode_step(hidden, router, experts, ep_buffer):
    topk_idx, topk_w = router(hidden)                  # local routing

    # Kick off dispatch; returns immediately with a 'hook' you call later.
    recv_x, recv_layout, hook = ep_buffer.low_latency_dispatch(
        hidden, topk_idx, num_experts=experts.E, use_fp8=True
    )
    # ---- OVERLAP WINDOW: do work that doesn't depend on recv_x ----
    #   e.g., prefetch next layer's expert weights, compute shared-expert FFN,
    #         run the next request stream's attention, etc.
    shared_out = experts.shared_expert(hidden)         # dense path, no all-to-all
    # --------------------------------------------------------------
    hook()                                             # block only now, on arrival

    expert_out = experts.grouped_gemm(recv_x, recv_layout)   # local expert FFNs (FP8 in)
    # Combine: send results home and weighted-sum by gate weights.
    y = ep_buffer.low_latency_combine(expert_out, topk_idx, topk_w)
    return y + shared_out
```

The single most important idea: **on the decode critical path, the all-to-all should not be visible**. If your profiler shows the GPU idle inside the EP collective, your overlap is broken, and your effective decode throughput will be a fraction of dense-equivalent. Getting the overlap right is the difference between a large-EP deployment that hits 80%+ of dense-equivalent throughput and one that hits 30%.

---

## Hot-Expert Skew at Decode and Load Balancing

Routing is never perfectly uniform, and uniformity is harder to guarantee at inference than in training. Three things make decode skew dangerous.

**First, no auxiliary loss is acting.** During training, a load-balancing loss (or DeepSeek's auxiliary-loss-free bias-correction) actively pushes the router toward balance. At inference the weights are frozen; whatever imbalance the deployment-time traffic induces is simply *suffered*. Worse, real traffic is not the training distribution: a burst of code requests, or one language, or one prompt template repeated across a batch (think a RAG system with a fixed system prompt) can correlate routing decisions and spike a few experts.

**Second, decode batches are small, so the law of large numbers does not save you.** With $B=2048$ tokens spread over $E=256$ experts, expected tokens per expert is $\sim 8B \cdot k/E$ and the *relative* fluctuation is small. With a low-latency decode batch of $B=64$, a single popular expert can receive several times the mean, and the all-to-all barrier waits for it.

**Third, every step is a barrier.** In prefill you process thousands of tokens per request; skew averages out over the prompt and the cost is amortized over a big GEMM. In decode you cross the all-to-all once per token per layer; a hot expert taxes *every single step*.

Quantify skew with the imbalance factor — the max load over the mean load:

$$
\text{IF} = \frac{\max_g L_g}{\frac{1}{G}\sum_g L_g},
$$

where $L_g$ is the number of tokens GPU $g$'s experts must process. Because the all-to-all completes at the slowest rank, the *effective* expert-compute time scales with IF. An IF of 1.0 is perfect; an IF of 2.0 means you are wasting half your aggregate expert FLOPs waiting on one GPU.

```python
import numpy as np

def imbalance_factor(token_expert_ids, E, G):
    """token_expert_ids: (T, k) selected expert ids for a decode batch."""
    counts = np.bincount(token_expert_ids.reshape(-1), minlength=E)
    experts_per_rank = E // G
    rank_load = counts.reshape(G, experts_per_rank).sum(axis=1)
    return rank_load.max() / rank_load.mean()

# Uniform routing vs. a skewed batch (one rank's experts are 4x popular)
rng = np.random.default_rng(0)
E, G, T, k = 256, 32, 2048, 8
uniform = rng.integers(0, E, size=(T, k))
print("uniform IF:", round(imbalance_factor(uniform, E, G), 3))    # ~1.05

p = np.ones(E); p[:8] *= 4; p /= p.sum()                            # experts 0..7 hot
skewed = rng.choice(E, size=(T, k), p=p)
print("skewed  IF:", round(imbalance_factor(skewed, E, G), 3))     # > 1.5
```

The production countermeasures are layered:

- **Expert replication / redundant experts.** Identify the busiest experts offline (from traffic logs) and place an extra copy on an under-loaded GPU; the dispatch then load-balances tokens for that expert across its replicas. DeepSeek's deployment calls these *redundant experts*; a handful of replicas of the top-K hottest experts can cut IF dramatically. The cost is a little extra memory.
- **Dynamic expert rebalancing.** Periodically (every few seconds) recompute expert placement from recent load and migrate experts to flatten the histogram. This is a control loop, not a per-step decision — migrating experts mid-decode is too expensive.
- **Capacity factor and token dropping.** As in training, you can cap tokens per expert at $C = \text{cf} \cdot \frac{B k}{E}$ and drop overflow (the token skips that expert, keeping only its other top-$k$ choices, or falls back to the shared expert). Dropping bounds the worst-case all-to-all but degrades quality; in serving most stacks prefer **no drop** with replication instead, because a dropped token is a silently worse answer.
- **Shared experts as a relief valve.** Models like DeepSeek-V3 route *every* token through one or two always-on **shared experts** (a dense path) in addition to the routed ones. The shared expert carries the "common" computation, reduces what the routed experts must specialize in, and — crucially for serving — its FFN runs locally with **no all-to-all**, giving you free overlap work to hide dispatch latency behind.

!!! warning "Common pitfall: benchmarking MoE on uniform synthetic traffic"

    A favorite way to fool yourself is to benchmark an EP deployment with randomly sampled prompts that route near-uniformly, measure a beautiful IF $\approx 1.05$, and ship. Production traffic is correlated — shared system prompts, one dominant language, bursty topics — and IF in the wild is routinely 1.3–2.0. Always benchmark with *replayed production traces*, and always report tail TPOT (p99), not mean, because skew lives in the tail.

---

## The Hybrid Layout: Attention-DP + Expert-EP

Here is the layout subtlety that separates a 2024-era MoE deployment from a 2025/2026 one. A large MoE has two very different sublayers with opposite parallelism preferences:

- **Attention** has *few* parameters but a *large, per-request* KV cache. You want to shard it in a way that does not replicate the KV cache and keeps it bandwidth-efficient. With Multi-head Latent Attention (MLA), as used by DeepSeek (see [Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html)), the per-token KV is a small compressed latent, so each GPU can cheaply hold the *full* KV for *its own* requests.
- **Experts** have *most* of the parameters and need to be *spread* (EP) so they fit and so aggregate expert bandwidth is high.

If you naively apply tensor parallelism (TP) to the whole model, you replicate the KV cache across all TP ranks — wasteful — and you split attention heads, which for MLA or low-head-count attention can underutilize each GPU. The modern answer is a **hybrid**:

> **Attention runs data-parallel (attention-DP): each GPU handles a disjoint set of requests and keeps those requests' full KV cache locally. Experts run expert-parallel (EP): the FFN/MoE sublayer is sharded across the same GPUs.**


{{fig:moe-serving-attn-dp-expert-ep-layout}}


The payoff is large. Attention does **zero** cross-GPU communication (no all-reduce, no KV replication), so the only collective in the whole layer is the MoE all-to-all. The KV cache is partitioned by request, so total KV capacity scales with GPU count — you can serve more concurrent requests. And because attention is local and cheap, the GPUs have spare time to *overlap* the MoE all-to-all behind attention of the next micro-batch.

The catch is **balance across the DP attention ranks**. Each GPU now owns a different set of requests with different sequence lengths and different arrival times. If GPU0's requests are long (big prefill, big KV) and GPU7's are short, GPU0 becomes the straggler — and since the MoE all-to-all is a group barrier, *everyone* waits for GPU0. So the scheduler must **balance tokens-in-flight across attention-DP ranks**, not just balance experts. This couples request scheduling ([Continuous Batching & Request Scheduling](../07-inference-serving/02-continuous-batching.html)) to the parallelism layout in a way dense serving never has to think about.

A common refinement combines a *small* TP degree for attention (to fit MLA/attention compute on fast GEMMs and to bound per-GPU KV) with DP across TP groups, and EP for experts — written as TP×DP for attention and EP for the MoE, all multiplexed on the same physical GPUs. The exact recipe depends on model shape and interconnect; the invariant is *attention and experts are parallelized differently, on the same devices.*

---

## Prefill vs. Decode: Why EP Needs Two Modes

Prefill and decode stress EP in opposite ways, and a serious deployment configures (or even physically disaggregates; see [Disaggregated Prefill/Decode & Chunked Prefill](../07-inference-serving/08-disaggregated-chunked-prefill.html)) them differently.

| Aspect | Prefill | Decode |
|---|---|---|
| Tokens per step | thousands (whole prompt) | one per request |
| Expert GEMM | large, compute-bound, well-utilized | tiny, latency-bound, often padded |
| All-to-all goal | **bandwidth** (move bytes fast) | **latency** (finish the barrier fast) |
| DeepEP kernel | high-throughput (NVLink+RDMA forwarding) | low-latency (pure RDMA, hook overlap) |
| EP degree sweet spot | smaller EP, big batches saturate experts | **large EP**, spread experts to add bandwidth |
| Skew sensitivity | low (averages over many tokens) | high (every step is a barrier) |

**Prefill** sends many tokens per expert, so the grouped GEMM is fat and compute-bound; the all-to-all is a bandwidth problem you solve with the high-throughput kernel. Skew matters less because thousands of tokens average out per expert.

**Decode** sends ~$Bk/E$ tokens per expert per step — often a handful or even zero. The grouped GEMM is skinny and dominated by overhead and padding; the bottleneck is the all-to-all *latency* and the per-step barrier. This is why **decode wants large EP**: spreading experts over more GPUs means each GPU holds fewer experts, so the local grouped GEMM is faster *and* more aggregate HBM bandwidth feeds expert weights, while DeepEP's low-latency kernels keep the wider all-to-all from dominating. Counterintuitively, for decode of a huge MoE, *more* GPUs in the EP group can *lower* per-token latency — the opposite of dense TP, where adding ranks eventually hurts because the all-reduce grows.

This asymmetry is the core argument for **disaggregating** prefill and decode onto separate GPU pools for MoE serving: run prefill on a smaller-EP, high-throughput pool, then hand the KV cache to a large-EP, low-latency decode pool. Each pool uses the matching DeepEP kernel and a matching EP degree, and neither compromises for the other.

!!! example "Worked example: sizing a large-EP DeepSeek-V3 decode deployment"

    **Model.** DeepSeek-V3: 671B total params, ~37B active/token, 256 routed experts + 1 shared, top-8 routing, $d_{\text{model}}=7168$, 61 layers (58 MoE), MLA KV.

    **Weights, FP8 (1 byte/param).** Resident weight memory $\approx 671\text{ GB}$. On 8×H100-80GB (640 GB) it does *not* fit with room for KV and activations. On **2 nodes = 16×H100 (1.28 TB)** it fits comfortably: $\approx 42\text{ GB}$ weights/GPU, leaving ~38 GB/GPU for KV + activations + overhead.

    **EP layout.** Place 256 experts over 16 GPUs $\Rightarrow$ 16 experts/GPU. Attention runs DP (each GPU owns its requests' MLA KV); MoE runs EP=16 with all-to-all.

    **All-to-all per layer (decode, BF16 dispatch).** Per token, dispatch moves $k\,d_{\text{model}}\cdot 2 = 8\cdot7168\cdot2 \approx 115\text{ KB}$. With FP8 dispatch, ~57 KB. For a decode batch of $B=128$ tokens that is $\approx 7.3\text{ MB}$ out per MoE layer; over 58 MoE layers, $\approx 423\text{ MB}$ moved per decode step (dispatch) and similar for combine. At an effective ~150 GB/s usable bidirectional RDMA per GPU after overlap, the *exposed* (non-overlapped) fraction is what hits TPOT — the engineering goal is to drive that toward zero with the low-latency hook kernels.

    **KV budget.** MLA compresses KV to a small latent (on the order of ~70 KB/token across all layers in FP8, vs. hundreds of KB for vanilla MHA). With ~38 GB/GPU free, one GPU holds $\sim 38\text{e}9 / 70\text{e}3 \approx 540{,}000$ tokens of KV — i.e., hundreds of concurrent long-context requests *per GPU*, times 16 GPUs. KV capacity is plentiful precisely because MLA + attention-DP avoids replication.

    **Takeaway.** The binding constraint is not arithmetic (37B active is cheap) — it is fitting 671B of weights resident, keeping IF near 1.0 with redundant experts, and hiding the 58-layer all-to-all behind compute. Two H100 nodes with large EP and DeepEP overlap is a sensible decode pool; a single node cannot hold the weights.

---

## Offloading: Experts and KV When You Can't Fit

Not everyone has two H100 nodes. The defining property of MoE — most parameters are *cold* on any given step — makes MoE unusually amenable to **offloading**: keep hot/shared parameters in HBM and stream cold experts from CPU RAM (or NVMe) on demand. The math that makes this viable: with top-$k$ routing, a single token needs only $k$ experts; if you can *predict or prefetch* which experts the batch will use, you move only those.

The techniques, roughly in order of aggressiveness:

- **Keep attention + router + shared expert in HBM; offload routed experts to CPU.** Each step, after the router fires, the needed experts are copied HBM-ward over PCIe/NVLink-C2C just-in-time. This works when the per-step expert set is small (small batch, low $k$) and the PCIe transfer overlaps compute. It collapses for large batches that touch most experts.
- **Expert caching with an LRU.** Treat HBM as a cache of experts; keep the most-recently/most-frequently-used experts resident and fault in the rest from CPU. Because real traffic is skewed (the same hot experts again and again), a cache far smaller than the full expert set can achieve a high hit rate. The cache *eviction policy is the router's load histogram* — keep the hot experts pinned.
- **Pre-gating / look-ahead prefetch.** Run the router (or a cheap proxy) a step early so you know which experts the next step needs and start the CPU→GPU copy before you need them, hiding transfer latency behind the current step's compute.
- **NVMe tier (ZeRO-Inference / FlexGen style).** For truly memory-starved setups, experts live on NVMe and stream through CPU RAM to GPU. Throughput is low (this is for offline batch, not interactive serving), but it lets a single GPU "run" a model far larger than its HBM. See [Memory-Efficient Training](../04-kernels-efficiency/10-memory-efficient-training.html) for the offload hierarchy and ZeRO-Infinity mechanics.

```python
# Illustrative expert offload with an LRU resident cache (single GPU).
import torch
from collections import OrderedDict

class ExpertCache:
    """Keep `capacity` experts on GPU; fault the rest in from CPU on demand."""
    def __init__(self, cpu_experts, capacity, device="cuda"):
        self.cpu = cpu_experts                  # dict: eid -> (w1_cpu, w2_cpu) pinned
        self.capacity = capacity
        self.device = device
        self.resident = OrderedDict()           # eid -> (w1_gpu, w2_gpu), LRU order

    def get(self, eid):
        if eid in self.resident:
            self.resident.move_to_end(eid)      # mark most-recently-used
            return self.resident[eid]
        if len(self.resident) >= self.capacity:
            old, (w1, w2) = self.resident.popitem(last=False)   # evict LRU
            del w1, w2
        w1c, w2c = self.cpu[eid]
        # non_blocking copy overlaps with compute if src is pinned memory
        w1 = w1c.to(self.device, non_blocking=True)
        w2 = w2c.to(self.device, non_blocking=True)
        self.resident[eid] = (w1, w2)
        return w1, w2

    def prefetch(self, eids):
        """Look-ahead: kick off copies for next step's experts."""
        for e in eids:
            if e not in self.resident:
                self.get(e)                     # warms the cache before it's needed
```

The honest caveat: offloading trades latency for capacity. PCIe 5.0 moves ~64 GB/s; an expert FFN in DeepSeek-V3 is on the order of tens of MB, so faulting in a handful per step costs hundreds of microseconds to milliseconds of transfer — easily dominating a decode step if not overlapped, and impossible to fully hide once a large batch touches most experts. Offloading is a *cost-saving for low-QPS or batch workloads*, not a path to low-latency high-throughput serving. For that, you buy the HBM.

---

## The Economics of Dense-Equivalent Throughput

Why do labs build these monsters if serving them is this hard? Because the *unit economics* — discussed in general in [Inference Economics: Latency, Throughput & Cost](../07-inference-serving/12-inference-economics.html) — favor sparsity when you can keep the experts busy.

Frame it as cost per million output tokens. The compute per token is set by *active* parameters: DeepSeek-V3 at 37B active does roughly the arithmetic of a dense 37B. But it answers with the *quality* of something far larger than 37B dense. So if you compare at equal quality, the MoE serves each token with a fraction of the FLOPs — *provided you reach high GPU utilization*. The serving overheads we have catalogued — all-to-all, skew, padding, the memory to hold all experts — are the "tax" on that FLOP advantage. Define **dense-equivalent throughput efficiency**:

$$
\eta = \frac{\text{achieved tokens/s on the MoE}}{\text{tokens/s of a dense model with the same active params on the same GPUs}}.
$$

$\eta = 1$ means the MoE serves as fast as its dense-active twin — you got the quality of a giant model for the price of a small one. A naive single-node MoE deployment, hammered by exposed all-to-all and IF $\approx 2$, might land at $\eta \approx 0.3$. A well-engineered large-EP deployment — attention-DP, redundant experts, DeepEP overlap, FP8 dispatch, prefill/decode disaggregation — recovers most of the gap, $\eta \approx 0.7$–$0.9$. That last factor of 2–3× is the entire ballgame; it is why DeepEP, redundant experts, and the hybrid layout exist.

The other half of the economics is **batch size**. MoE *needs* large batches to be efficient: only a big batch keeps every expert's grouped GEMM fat enough to be compute-bound, and only a big batch amortizes the all-to-all overhead per token. A dense model degrades gracefully at small batch (it is just bandwidth-bound); an MoE at small batch is doubly punished — skinny expert GEMMs *and* a per-step all-to-all barrier with little to overlap. The practical consequence: **MoE serving lives at high concurrency.** If your traffic cannot fill big batches, a dense model of equal active size may be cheaper to run despite "wasting" parameters. The break-even concurrency is a real number you should compute for your traffic before committing to an MoE in production.

!!! interview "Interview Corner"

    **Q:** You're serving a 671B-parameter, 37B-active MoE on a 16-GPU cluster and observe that p50 TPOT matches a dense 37B baseline but p99 TPOT is 3× worse. The GPU utilization shows brief idle gaps inside every MoE layer. What is happening and how do you fix it?

    **A:** Two coupled problems, both visible as idle inside the MoE layer. (1) **Hot-expert skew**: the all-to-all is a barrier that completes only when the slowest rank finishes, so a batch that over-selects experts on one GPU stalls every other GPU. This explains the *tail* (p99) blowing up while p50 looks fine — skew is bursty and traffic-dependent. Fix: add **redundant replicas** of the hottest experts on under-loaded GPUs and load-balance dispatch across replicas; run a periodic **rebalancing** control loop driven by the live load histogram; ensure you are *not* dropping tokens (prefer replication to capacity-factor dropping in serving). (2) **Exposed all-to-all**: idle gaps inside the layer mean the dispatch/combine latency is on the critical path, not overlapped. Fix: use **low-latency DeepEP kernels with the hook-based overlap**, schedule the **shared-expert FFN** and the next micro-batch's attention into the overlap window, and enable **FP8 dispatch** to halve the bytes moved. If the cluster is two nodes, also confirm attention is **DP (local KV)** so attention contributes no extra collective. Together these typically pull p99 back toward p50 and lift dense-equivalent efficiency from ~0.3–0.4 to ~0.7–0.9.

---

## Putting It Together: An EP Serving Checklist

A production MoE serving stack (vLLM, SGLang, TensorRT-LLM all converge here) assembles the pieces from this chapter:

1. **Layout.** Attention-DP (local KV, possibly small TP for MLA) + expert-EP, multiplexed on the same GPUs. Size EP so all experts fit resident with headroom for KV.
2. **Communication.** DeepEP (or equivalent): high-throughput kernels for prefill, low-latency hook kernels for decode, FP8 dispatch, NVLink+RDMA forwarding across nodes.
3. **Balance.** Redundant experts for the hot ones, a rebalancing control loop, and a scheduler that balances tokens-in-flight across attention-DP ranks (not just across experts).
4. **Disaggregation.** Separate prefill (smaller EP, throughput kernels) and decode (large EP, latency kernels) pools; hand off the KV cache between them.
5. **Overlap.** Drive exposed all-to-all toward zero by scheduling shared-expert compute, attention, and weight prefetch into the dispatch/combine windows.
6. **Memory tiering.** If HBM is tight, LRU expert caching / offload — but only for low-QPS or batch workloads; interactive low-latency serving needs experts resident.
7. **Measure the right thing.** Report $\eta$ (dense-equivalent efficiency), p99 TPOT, and the routing imbalance factor on *replayed production traffic*, not synthetic uniform routing.

```python
def ep_decode_step_pipeline(batch, layers, ep, sched):
    """
    End-to-end shape of one decode step in a hybrid attention-DP + expert-EP stack.
    Each `layer` interleaves LOCAL attention with an EP MoE, overlapping the all-to-all.
    """
    sched.balance_tokens_across_dp_ranks(batch)        # keep attention ranks even
    h = batch.hidden
    for layer in layers:
        # --- attention: purely local, owns this rank's request KV (MLA) ---
        h = layer.attention(h, batch.kv_cache)         # NO cross-GPU collective
        # --- router + MoE with overlapped all-to-all ---
        topk_idx, topk_w = layer.router(h)
        recv, layout, hook = ep.low_latency_dispatch(   # async; returns a hook
            h, topk_idx, use_fp8=True)
        shared = layer.shared_expert(h)                 # overlap work, no all-to-all
        hook()                                          # block only on arrival
        expert_out = layer.grouped_gemm(recv, layout)   # local experts
        h = ep.low_latency_combine(expert_out, topk_idx, topk_w) + shared
    return layer_norm_and_lm_head(h)
```

Every line of that loop is a decision this chapter argued for: local attention because of attention-DP, FP8 dispatch and an async hook because of DeepEP overlap, a shared expert sitting in the overlap window, and a scheduler that keeps the DP ranks balanced so the all-to-all barrier never waits on a straggler. That is what it takes to make a trillion-scale sparse model serve at the speed of a small dense one — which is the whole reason MoE exists.

---

!!! key "Key Takeaways"

    - **MoE serving is a memory and movement problem, not an arithmetic one.** Per-token FLOPs equal a small dense model, but all experts must stay resident (a modest decode batch touches ~85%+ of experts), and every MoE layer needs two all-to-all collectives.
    - **The all-to-all is a barrier.** Decode crosses dispatch+combine once per token per layer; the step finishes only when the slowest, most-overloaded rank does. Latency, not bandwidth, is the enemy at decode.
    - **DeepEP-style kernels are the enabler:** high-throughput (NVLink+RDMA forwarding) for prefill, low-latency hook-overlap (pure RDMA, FP8 dispatch) for decode. The goal is *zero exposed all-to-all* on the decode critical path.
    - **Hot-expert skew is a tail-latency weapon at inference** (no auxiliary loss acting, small batches, per-step barrier). Counter with redundant replicas of hot experts, dynamic rebalancing, shared experts, and replication-over-dropping.
    - **The modern layout is hybrid: attention-DP (local KV, no replication, zero attention collective) + expert-EP.** It requires balancing tokens-in-flight across attention-DP ranks, coupling scheduling to the parallelism layout.
    - **Prefill and decode want opposite EP configs:** prefill = smaller EP + throughput kernels; decode = *large* EP + latency kernels. This argues for disaggregating the two onto separate pools.
    - **MoE is uniquely offload-friendly** (most params cold per step) via LRU expert caching and look-ahead prefetch — but offload buys capacity, not low latency; interactive serving keeps experts in HBM.
    - **Economics hinge on dense-equivalent efficiency $\eta$ and on batch size.** A naive deployment lands at $\eta\approx 0.3$; a well-engineered large-EP one at $0.7$–$0.9$. MoE needs high concurrency to win — at low QPS, a dense model of equal active size may be cheaper.

---

!!! sota "State of the Art & Resources (2026)"
    MoE serving is a fast-moving frontier: the core challenge of making per-layer all-to-all collectives invisible on the decode critical path is now well-understood, with large-EP + attention-DP hybrid layouts and purpose-built overlap kernels (DeepEP) achieving dense-equivalent throughput efficiency of 0.7–0.9 on frontier hardware. Production deployments at scale (96–128 GPUs) using disaggregated prefill/decode and redundant expert replication are now open-source and reproducible via vLLM and SGLang.

    **Foundational work**

    - [Lepikhin et al., *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding* (2020)](https://arxiv.org/abs/2006.16668) — first formalization of expert sharding with all-to-all dispatch/combine at 600B+ parameter scale on TPUs.
    - [Fedus, Zoph & Shazeer, *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity* (2021)](https://arxiv.org/abs/2101.03961) — introduced capacity factor, token dropping, and the load-balancing auxiliary loss that underpins all later MoE routing analysis.
    - [Rajbhandari et al., *DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale* (2022)](https://arxiv.org/abs/2201.05596) — first systematic treatment of MoE inference optimization, combining EP + TP, achieving 4.5× speedup over quality-equivalent dense models.

    **Recent advances (2023–2026)**

    - [DeepSeek-AI, *DeepSeek-V3 Technical Report* (2024)](https://arxiv.org/abs/2412.19437) — canonical large-EP production MoE: MLA, auxiliary-loss-free load balancing, redundant expert replication, and EP=32 serving across nodes.
    - [Jiang et al., *Mixtral of Experts* (2024)](https://arxiv.org/abs/2401.04088) — first widely-deployed open-weight sparse MoE (8×7B), establishing the modern top-2 routing + grouped-GEMM baseline for MoE serving benchmarks.
    - [Gale et al., *MegaBlocks: Efficient Sparse Training with Mixture-of-Experts* (2022)](https://arxiv.org/abs/2211.15841) — block-sparse GPU kernels for MoE that eliminate token dropping and padding waste; widely used as the grouped-GEMM backend in serving stacks.
    - [LMSYS, *Deploying DeepSeek with PD Disaggregation and Large-Scale Expert Parallelism on 96 H100 GPUs* (2025)](https://www.lmsys.org/blog/2025-05-05-large-scale-ep/) — open-source reproduction of DeepSeek's production EP serving in SGLang: 52k input tokens/s/node, 5× over vanilla TP, at ~$0.20/1M output tokens.

    **Open-source & tools**

    - [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP) — purpose-built EP communication library: high-throughput NVLink+RDMA-forwarding kernels for prefill, low-latency hook-overlap RDMA kernels for decode, and FP8 dispatch.
    - [sgl-project/sglang](https://github.com/sgl-project/sglang) — high-performance serving framework with native large-scale EP, DeepEP integration, EPLB load rebalancing, and disaggregated prefill/decode for MoE models.

    **Go deeper**

    - [vLLM Expert Parallel Deployment docs](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/) — practical guide to EP layout, the vLLM Expert Parallel Load Balancer (EPLB), and multi-node MoE deployment with disaggregated serving.

## Further reading

- DeepSeek-AI, *DeepSeek-V3 Technical Report*, 2024 — the canonical large-EP MoE: MLA, auxiliary-loss-free balancing, redundant experts, and production EP serving across nodes.
- DeepSeek-AI, *DeepEP* (open-source repository) — purpose-built expert-parallel all-to-all kernels: high-throughput NVLink+RDMA forwarding, low-latency hook-overlap kernels, and FP8 dispatch.
- Lepikhin et al., *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding*, ICLR 2021 — first formalization of expert sharding and all-to-all dispatch/combine at scale.
- Fedus, Zoph & Shazeer, *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*, JMLR 2022 — capacity factor, token dropping, and load-balancing loss that this chapter's serving countermeasures echo.
- Rajbhandari et al., *DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale*, ICML 2022 — early systematic treatment of MoE *inference* optimization and expert parallelism.
- Aminabadi et al., *DeepSpeed-Inference* / Rajbhandari et al., *ZeRO-Infinity*, SC 2021 — the offload hierarchy (HBM/CPU/NVMe) underpinning expert and KV offloading.
- vLLM (Kwon et al., 2023) and SGLang (Zheng et al., 2024) project documentation — open-source references for production EP serving, large-EP deployment, and DeepEP integration; see [vLLM: Architecture, PagedAttention & Internals](../07-inference-serving/03-vllm-internals.html) and [SGLang: RadixAttention & Structured Programs](../07-inference-serving/04-sglang-radixattention.html).
