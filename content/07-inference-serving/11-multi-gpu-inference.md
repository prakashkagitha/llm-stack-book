# 7.11 Multi-GPU & Multi-Node Inference

Modern large language models do not fit comfortably on a single GPU. A 70-billion-parameter model in BF16 requires roughly 140 GB of device memory before you account for the KV cache, activations, or framework overhead — far beyond the 80 GB of a single H100. Even when the weights *do* fit, a single GPU's bandwidth and compute can limit both latency and throughput. Multi-GPU and multi-node serving is therefore not an exotic optimization — it is the default operating mode for anything above about 13 billion parameters in production.

This chapter builds a complete picture of how inference is distributed: the four parallelism strategies available, what each costs and buys, how communication interacts with the decode loop, how Mixture-of-Experts (MoE) models introduce a fifth axis called expert parallelism, and finally how to size and configure a real deployment. We will implement minimal but correct reference code for each strategy and work through concrete numerical examples.

Before reading, you may want to review [GPU Architecture & The Memory Hierarchy](../01-foundations/08-gpu-architecture.html) and [Parallel Computing & Collective Communication](../01-foundations/09-parallel-collectives.html) for the hardware foundations. The training-side treatment of the same parallelism strategies lives in [Distributed Training II: Tensor, Pipeline, Sequence & Expert Parallelism](../03-pretraining/06-distributed-model-parallel.html); this chapter focuses exclusively on inference-specific trade-offs.

---

## The Four Parallelism Axes

Serving a large model across $N$ devices decomposes into four independent (and composable) axes:

| Axis | Abbreviation | What is sharded | Primary benefit |
|---|---|---|---|
| Tensor parallelism | TP | Weight matrices split column/row-wise within a layer | Reduces per-GPU memory; can lower latency |
| Pipeline parallelism | PP | Model split into stages of consecutive layers | Fits very deep models; primarily a throughput tool |
| Expert parallelism | EP | MoE expert weights spread across devices | Enables enormous MoE capacity at constant compute |
| Data parallel replicas | DP | Full (or TP/PP) model replicated | Pure throughput scaling; latency unchanged |

The total GPU count satisfies $N = \text{TP} \times \text{PP} \times \text{DP}$. EP is *not* an extra multiplier on top of that: expert-parallel ranks are drawn from GPUs the deployment already owns, and in vLLM and SGLang the expert-parallel group size is exactly $\text{EP} = \text{TP} \times \text{DP}$ within one model instance — the same devices, regrouped for the MoE layers only. A fifth axis, **context parallelism** (CP), shards the *sequence* rather than the weights; it is a long-context tool and we return to it at the end of the chapter.

Each axis interacts differently with the two phases of inference — prefill (compute-bound, processes the full prompt in one pass) and decode (memory-bandwidth-bound, processes one token per step). We cover both phases throughout.

None of this is needed for the ~100M-parameter model built in Part XIV — it fits in a fraction of one GPU, and the only axis that applies is DP replicas behind a load balancer (see [Evaluation & Serving: Honest Benchmarks, int4 Quantization, and Running on a Laptop](../14-capstone/11-evaluation-and-serving.html)). Everything below is what changes once the model no longer fits on one device.

{{fig:four-parallelism-axes}}

---

## Tensor Parallelism

### Mechanism

Tensor parallelism (TP), introduced at production scale by Megatron-LM (Shoeybi et al., 2019), splits individual weight matrices across GPUs so that each GPU handles a column-partition (for a linear projecting *into* the hidden dimension) or a row-partition (projecting *out* of it). Consider a column-parallel linear layer $Y = X W$ where $W \in \mathbb{R}^{d \times k}$:

$$
W = \begin{bmatrix} W_1 & W_2 & \cdots & W_T \end{bmatrix}, \quad
Y_i = X W_i \quad \text{on GPU } i
$$

After the column-parallel layer, each GPU holds $Y_i \in \mathbb{R}^{B \times k/T}$. A subsequent row-parallel layer $Z = Y W'$ is arranged as $W' = \begin{bmatrix} W'_1 \\ \vdots \\ W'_T \end{bmatrix}$ so that GPU $i$ computes $Z_i = Y_i W'_i$ and the full result is $Z = \sum_i Z_i$, recovered by an **all-reduce**.

{{fig:tensor-parallel-allreduce}}

In the attention layer, the $Q, K, V$ projections are column-parallel (each GPU owns a disjoint set of attention heads) and the output projection is row-parallel, requiring one all-reduce per attention block. The MLP block follows the same pattern: up-project column-parallel, down-project row-parallel, one all-reduce.

For a TP degree $T$, there are **two all-reduces per transformer layer** during the forward pass: one after attention, one after the MLP.

```python
# Minimal tensor-parallel linear layer (illustrative, not production)
import torch
import torch.distributed as dist

class ColParallelLinear(torch.nn.Module):
    """
    Column-parallel: each rank holds columns [start:end] of the weight matrix.
    Input X is replicated across all ranks; output is partitioned.
    No all-reduce needed here — a subsequent RowParallelLinear will reduce.
    """

    def __init__(self, in_features: int, out_features: int, rank: int, world_size: int):
        super().__init__()
        assert out_features % world_size == 0
        self.rank = rank
        self.world_size = world_size
        self.local_out = out_features // world_size
        # Each rank holds a (in_features x local_out) slice
        self.weight = torch.nn.Parameter(
            torch.empty(self.local_out, in_features)
        )
        torch.nn.init.kaiming_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, in_features) — replicated
        return torch.nn.functional.linear(x, self.weight)  # (batch, seq, local_out)


class RowParallelLinear(torch.nn.Module):
    """
    Row-parallel: each rank holds rows [start:end].
    Input is partitioned (from ColParallelLinear output); output is all-reduced.
    """

    def __init__(self, in_features: int, out_features: int, rank: int, world_size: int):
        super().__init__()
        assert in_features % world_size == 0
        self.local_in = in_features // world_size
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, self.local_in)
        )
        torch.nn.init.kaiming_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, local_in) — partitioned; compute partial output
        partial = torch.nn.functional.linear(x, self.weight)  # (batch, seq, out_features)
        # Sum partials across all tensor-parallel ranks
        dist.all_reduce(partial, op=dist.ReduceOp.SUM)
        return partial
```

### Running It For Real: `torchrun` and NCCL

The two classes above are inert until something creates a process group. In PyTorch that is `torch.distributed`, and the collective implementation underneath — the library that actually moves the bytes over NVLink, PCIe, or InfiniBand — is NVIDIA's **NCCL** (backend `"nccl"`; use `"gloo"` if you only have CPUs). Every serving stack in this chapter, vLLM and SGLang and TensorRT-LLM alike, ultimately calls NCCL for its TP all-reduces. The script below is a complete, runnable check that a column-then-row sharded MLP reproduces the unsharded result:

```python
# tp_mlp_check.py — run with: torchrun --nproc_per_node=2 tp_mlp_check.py
import torch
import torch.distributed as dist
import torch.nn.functional as F


def main():
    dist.init_process_group(backend="nccl")  # "gloo" for a CPU-only machine
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)

    # Identical seed on every rank => every rank materializes the SAME full
    # weights, so we can compare the sharded result against a local reference.
    torch.manual_seed(0)
    d_model, d_ff, B = 512, 2048, 8
    W_up = torch.randn(d_ff, d_model, device=dev) / d_model**0.5
    W_down = torch.randn(d_model, d_ff, device=dev) / d_ff**0.5
    x = torch.randn(B, d_model, device=dev)  # replicated input, as TP requires

    # Reference: the full, unsharded MLP.
    ref = F.gelu(x @ W_up.T) @ W_down.T

    # Sharded: column-parallel up-projection, row-parallel down-projection.
    assert d_ff % world == 0
    s = d_ff // world
    W_up_local = W_up[rank * s : (rank + 1) * s, :]     # this rank's d_ff columns
    W_down_local = W_down[:, rank * s : (rank + 1) * s]  # this rank's d_ff rows

    h = F.gelu(x @ W_up_local.T)   # (B, d_ff/world) — elementwise, so NO comm here
    out = h @ W_down_local.T       # (B, d_model) — a PARTIAL sum, not the answer
    dist.all_reduce(out, op=dist.ReduceOp.SUM)  # the one and only collective

    err = (out - ref).abs().max().item()
    if rank == 0:
        print(f"world={world}  max abs error vs unsharded = {err:.2e}")  # ~1e-5 in fp32
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

The error is pure floating-point reassociation noise, which is the point: TP is an exact refactoring of the same arithmetic, not an approximation. Two environment variables are worth knowing when this goes wrong on a real cluster — `NCCL_DEBUG=INFO` prints the topology NCCL discovered and the ring/tree algorithm it chose, and `NCCL_P2P_DISABLE=1` / `NCCL_IB_DISABLE=1` force it off NVLink or InfiniBand respectively, which is the standard way to confirm that a "slow TP" problem really is a fabric problem.

!!! tip "Practitioner tip: custom all-reduce for tiny decode messages"

    Decode all-reduces are small (tens of kilobytes), and at that size NCCL's ring algorithm is latency-bound rather than bandwidth-bound. vLLM therefore ships its own one-shot NVLink all-reduce kernel that beats NCCL below a size threshold and falls back to NCCL above it; `--disable-custom-all-reduce` turns it off. If you see TP=8 decode that is slower than TP=4, check this path (and CUDA graphs) before blaming the model.

### TP in the Decode Loop

During decode, the batch size is typically small (often 1 to a few hundred). A single all-reduce on a BF16 tensor of shape `[B, 1, d_model]` transfers roughly $2 \times d_{\text{model}} \times B$ bytes per rank over NVLink. For Llama-3 70B with $d_{\text{model}} = 8192$ and $B = 64$:

$$
\text{bytes per all-reduce} = 2 \times 8192 \times 64 = 1\,\text{MB}
$$

At a NVLink bandwidth of around 900 GB/s between two H100s, that is about **1 µs** per all-reduce, negligible compared with the kernel launch overhead. TP therefore has very low communication overhead on a single NVLink island. Crossing PCIe or Ethernet raises this cost by 10–100×, making TP across nodes generally inadvisable.

### Latency vs. Throughput Impact

TP reduces the per-GPU memory footprint of weights by a factor of $T$ and reduces the per-layer compute time by approximately $T$ (each GPU does $1/T$ of the matmul). For a latency-sensitive workload (single user, small batch), TP can deliver near-linear speedup up to the NVLink bandwidth wall. For a throughput-sensitive workload, TP is less efficient than data parallelism, because the all-reduce volume grows *linearly* with batch size: the cost per token is fixed and never amortizes, unlike weight loading, which is the one cost batching does amortize. DP replicas pay no such tax at all.

!!! example "TP memory saving for Llama-3 70B"

    Llama-3 70B has approximately 70 billion parameters. At BF16 (2 bytes/parameter):

    $$\text{Weight memory} = 70 \times 10^9 \times 2 = 140 \text{ GB}$$

    With TP = 4 across 4 × H100 80 GB GPUs:

    $$\text{Per-GPU weight memory} = 140/4 = 35 \text{ GB}$$

    Leaving 45 GB per GPU for the KV cache. At a KV head dimension of 128, 8 KV heads (GQA), 80 layers, and BF16:

    $$\text{KV cache per token} = 2 \times 8 \times 128 \times 80 \times 2 = 327\,680 \text{ bytes} \approx 320 \text{ KB/token}$$

    So the remaining 45 GB supports roughly $45 \times 10^9 / 327{,}680 \approx 137{,}000$ tokens of context. For a maximum context of 8 K tokens, that accommodates about 17 concurrent requests — a reasonable serving batch.

    This is the *conservative* count: it charges every GPU the full 320 KB/token. In reality TP shards the KV heads too, so with 8 KV heads at TP = 4 each GPU stores only 2 of them and the true capacity is about 4x larger. Exercise 3 works that arithmetic through.

---

## Pipeline Parallelism

### Mechanism

Pipeline parallelism (PP) divides the model's layers into consecutive **stages** assigned to different devices. GPU 0 holds layers 0–$L/P$, GPU 1 holds layers $L/P$–$2L/P$, and so on. Communication between stages consists of passing activations (the hidden state) from one stage to the next — a single peer-to-peer `send`/`recv`, not a collective.

For a model with hidden size $d$ and micro-batch size $B$:

$$
\text{inter-stage activation bytes} = B \times 1 \times d \times \text{dtype\_bytes}
$$

For Llama-3 70B with $d = 8192$, $B = 1$, BF16: $1 \times 8192 \times 2 = 16$ KB per step. Even over InfiniBand at 25 GB/s, this is about **0.6 µs** — entirely dominated by the GPU compute.

### The Pipeline Bubble and Decode

During *training*, PP creates a "bubble" — idle time while stages wait for activations. In inference, the story is different:

- **Prefill**: a single batch passes through the pipeline sequentially. For a PP degree of $P$, each stage processes $L/P$ layers, so the total latency is *roughly unchanged* from a single GPU processing all $L$ layers (ignoring inter-stage latency). There is no bubble; the pipeline has only one micro-batch.
- **Decode**: each step is sequential by nature. Stage $i$ cannot start until stage $i-1$ finishes. This makes PP a **latency-neutral** strategy for decode: it does not help per-token latency and may hurt it slightly due to inter-stage synchronization.

PP's real value is enabling models that do not fit in TP-only memory. For example, a 400B+ dense model with PP = 4 and TP = 8 on 32 GPUs keeps each GPU's memory load manageable while TP handles the per-layer distribution.

```python
# Minimal pipeline stage runner (single-process simulation)
import torch
from typing import List, Tuple

class PipelineStage(torch.nn.Module):
    """Holds a contiguous slice of transformer layers."""

    def __init__(self, layers: torch.nn.ModuleList):
        super().__init__()
        self.layers = layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def pipeline_forward_inference(
    stages: List[PipelineStage],
    x: torch.Tensor,
    devices: List[torch.device],
) -> torch.Tensor:
    """
    Sequential pipeline forward pass (no micro-batching during inference decode).
    Each stage resides on a different device; activations are moved between stages.
    """
    assert len(stages) == len(devices)
    for stage, device in zip(stages, devices):
        # Move activations to the current stage's device
        x = x.to(device)
        with torch.no_grad():
            x = stage(x)
    return x  # Final output on the last device


# Example: split an 80-layer model into 4 pipeline stages of 20 layers each
def build_pipeline_example(
    n_layers: int = 80,
    n_stages: int = 4,
    d_model: int = 512,  # small for illustration
) -> Tuple[List[PipelineStage], List[torch.device]]:
    layers_per_stage = n_layers // n_stages
    devices = [torch.device("cpu")] * n_stages  # would be cuda:i in real usage
    stages = []
    for stage_idx in range(n_stages):
        module_list = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                d_model=d_model, nhead=8, batch_first=True
            )
            for _ in range(layers_per_stage)
        ])
        stages.append(PipelineStage(module_list).to(devices[stage_idx]))
    return stages, devices
```

### When PP Helps

PP is most effective when:
1. The model is too large for TP alone to fit in per-node NVLink islands.
2. You are willing to trade some decode latency for lower per-GPU memory.
3. Throughput (tokens/second) matters more than time-to-first-token (TTFT).

For latency-critical deployments (interactive chat), avoid PP unless forced by memory constraints.

---

## Expert Parallelism for MoE Models

### MoE Inference Recap

In a Mixture-of-Experts model, each transformer block contains $E$ expert FFN sub-networks, of which each token activates $k$ (top-$k$ routing). The active compute per token is therefore $k/E$ of the total expert parameter mass. See [Mixture-of-Experts (MoE) Architectures](../02-transformer/09-mixture-of-experts.html) for the architecture details.

Expert parallelism (EP) assigns disjoint subsets of experts to different devices. For EP degree $G$ and $E$ total experts, each device hosts $E/G$ experts. After a router assigns each token to its top-$k$ experts, tokens must be **dispatched** to the GPU holding the target expert and **gathered** back after processing. This requires two all-to-all collectives per MoE layer.

$$
\text{all-to-all volume (bytes)} = B \times d_{\text{model}} \times \text{dtype\_bytes} \times k
$$

For a batch of 512 tokens, $d_{\text{model}} = 7168$ (DeepSeek-V3 style), $k = 8$ routed experts, BF16:

$$
512 \times 7168 \times 2 \times 8 = 58.7 \text{ MB per all-to-all}
$$

With two all-to-alls (dispatch and combine) per MoE layer, and 58 MoE layers in DeepSeek-V3 (the first 3 of its 61 layers use a dense FFN), that is on the order of 6.8 GB leaving each rank per forward pass — a serious amount of traffic, and the single reason wide-EP is hard. Two things cut it in practice: dispatching activations in **FP8** rather than BF16 (halving the volume, which DeepEP supports natively) and overlapping the transfer with expert compute so it never appears on the critical path.

{{fig:expert-parallel-scatter-gather}}

### Wide Expert Parallelism (Wide-EP): DeepSeek-V3 / DeepSeek-R1

DeepSeek-V3 introduced the concept of **wide EP** in the context of serving their 671B MoE model. Standard practice places EP within a single node (using NVLink for all-to-all). Wide-EP extends EP across *multiple nodes*, using InfiniBand for inter-node all-to-all, allowing a far larger EP degree.

DeepSeek-V3 has $E = 256$ fine-grained routed experts per MoE layer, plus one *shared* expert that every token uses (and which therefore never needs an all-to-all — it is replicated on every rank), with $k = 8$ routed expert activations per token. Wide-EP allows EP degrees in the tens or hundreds, spreading all 256 experts across many GPU nodes so that each rank holds only a handful. The routing then dispatches tokens across the cluster via RDMA.

Key trade-offs of wide-EP:

- **Benefit**: each GPU holds fewer experts, so expert weight memory per GPU drops proportionally. Total KV-cache memory scales with the cluster.
- **Cost**: all-to-all over InfiniBand has 10–50× higher latency than NVLink. For small batches, the network becomes the bottleneck.
- **Mitigation**: two techniques help — (1) **compute-communication overlap** (prefetch the next all-to-all while computing the current layer), and (2) **expert load balancing** to avoid hot experts that increase effective network traffic. DeepSeek-V3 also uses an auxiliary-loss-free balancing scheme during training to encourage balanced routing, which directly improves serving efficiency.

As of 2026 the intra-node/inter-node boundary is itself shifting: on NVIDIA's Blackwell **GB200/GB300 NVL72** systems a single fifth-generation NVLink domain spans 72 GPUs at 1.8 TB/s per GPU, so an entire wide-EP group can sit inside one NVLink island — collapsing much of the inter-node all-to-all penalty that motivated these mitigations on Hopper-era (H100/H200) clusters.

```python
# Simplified expert-parallel dispatch/gather (pseudocode with real shapes)
import torch
import torch.distributed as dist

def expert_parallel_forward(
    x: torch.Tensor,           # (B, d_model) — token embeddings on this rank
    router_logits: torch.Tensor,  # (B, E) — per-expert logit scores
    expert_ffns: torch.nn.ModuleList,  # E/G experts on this rank
    k: int = 2,                # top-k routing
    ep_group: dist.ProcessGroup = None,  # EP process group
) -> torch.Tensor:
    """
    Single MoE layer with expert parallelism.
    Each rank owns experts [rank * E/G : (rank+1) * E/G].
    """
    ep_size = dist.get_world_size(ep_group)
    rank = dist.get_rank(ep_group)
    E = router_logits.shape[-1]
    experts_per_rank = E // ep_size

    # Step 1: compute routing weights and expert assignments
    top_k_weights, top_k_indices = torch.topk(
        torch.softmax(router_logits, dim=-1), k, dim=-1
    )  # (B, k) each

    # Step 2: build dispatch tensors — which tokens go to which rank
    # For each token, determine which expert rank owns the selected expert
    expert_ranks = top_k_indices // experts_per_rank  # (B, k)

    # Step 3: all-to-all dispatch — send each token to owning expert rank
    # (In practice, this involves scatter_to_expert_rank via dist.all_to_all)
    # Here we simulate with a gather
    dispatched = _all_to_all_dispatch(x, expert_ranks, top_k_indices, ep_group)
    # dispatched is now on the correct rank, shaped (local_tokens, d_model)

    # Step 4: compute expert outputs on local experts
    local_expert_indices = top_k_indices % experts_per_rank  # local numbering
    # This would iterate over local experts and process their assigned tokens
    expert_out = _apply_local_experts(dispatched, local_expert_indices, expert_ffns)

    # Step 5: all-to-all gather — return results to token-owning ranks
    output = _all_to_all_gather(expert_out, expert_ranks, ep_group)

    # Step 6: weighted sum over k experts
    # output shape: (B, k, d_model); top_k_weights: (B, k)
    output = (output * top_k_weights.unsqueeze(-1)).sum(dim=1)

    return output


def _all_to_all_dispatch(x, expert_ranks, indices, group):
    """Placeholder — real implementation uses dist.all_to_all."""
    # In production frameworks (vLLM, SGLang), this is a fused CUDA kernel
    return x  # simplified


def _all_to_all_gather(x, expert_ranks, group):
    """Placeholder — real implementation uses dist.all_to_all."""
    return x.unsqueeze(1).expand(-1, 2, -1)  # simplified


def _apply_local_experts(x, local_indices, expert_ffns):
    """Apply local expert FFNs to dispatched tokens."""
    results = []
    for i, expert in enumerate(expert_ffns):
        mask = (local_indices == i).any(dim=-1)  # tokens routed to expert i
        if mask.any():
            results.append(expert(x[mask]))
    # Reassemble — simplified
    return x.unsqueeze(1).expand(-1, 2, -1)
```

### EP Load Balancing

Router collapse (all tokens routed to a few "hot" experts) kills EP performance: one GPU is overloaded while others are idle, and the all-to-all becomes unbalanced. Production systems address this with:

1. **Auxiliary load-balancing loss** during training (balance loss in Switch Transformer, DeepSeek-V2/V3's group-norm softmax).
2. **Token dropping** when an expert's capacity is exceeded.
3. **Expert duplication**: replicate popular experts on multiple GPUs at inference time, then route with load awareness.
4. **Dynamic expert offloading**: for CPU-offloaded MoE inference (useful when GPU count is limited), pre-fetch the next likely experts based on previous routing statistics.

### DP Attention: Why Wide-EP Deployments Do Not Use Wide TP

A detail that surprises people the first time they read a production MoE serving config: the attention blocks and the MoE blocks use *different* parallelism on the *same* GPUs.

The reason is the KV cache. TP shards the KV cache by attention head, so it only shrinks the per-GPU footprint while TP $\le n_{\text{kv\_heads}}$; beyond that the heads must be *replicated*, and every extra TP rank costs a full duplicate copy of the cache. Multi-head Latent Attention (MLA) makes this worse — its compressed latent KV behaves like a single head, so any TP degree above 1 replicates it outright. Meanwhile the MoE layers want a very large EP degree, because that is what makes each rank's expert weights small.

The resolution is **DP attention**: run attention data-parallel, with each rank owning a disjoint set of *requests* and its own private KV cache, while the MoE layers regroup the identical set of GPUs into one large expert-parallel group. At the boundary of each MoE layer the ranks exchange their local tokens (an all-gather or, in the fused kernels, the dispatch all-to-all itself), run the experts, and scatter the results back to the owning rank. The KV cache is never duplicated, per-rank batch stays large enough to escape the small-batch trap, and EP can be as wide as the cluster.

Both major open engines expose this directly:

```bash
# SGLang: DP attention + expert parallelism on 8 GPUs
python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 \
    --tp 8 --dp-size 8 --enable-dp-attention --enable-ep-moe --trust-remote-code

# vLLM: the data-parallel dimension is applied to attention; MoE layers use
# an expert-parallel group of size (data_parallel_size x tensor_parallel_size)
vllm serve deepseek-ai/DeepSeek-V3 \
    --tensor-parallel-size 1 --data-parallel-size 8 --enable-expert-parallel
```

Note what `--data-parallel-size` means here: unlike the independent replicas of the next section, these DP ranks are *one* model instance and must step in lockstep through the shared MoE layers — vLLM keeps them synchronized with dummy forward passes when one rank has no work. Flag names and defaults in both projects move quickly; check `--help` against your installed version rather than trusting a config copied from a blog post.

---

## Data Parallel Replicas

Data parallelism (DP) for inference is the simplest strategy: run $D$ identical copies of the model (each itself potentially TP/PP sharded), each serving independent requests. A load balancer or router distributes incoming requests across replicas.

$$
\text{throughput} \propto D, \quad \text{latency} = \text{const (per replica)}
$$

DP is the right choice when:
- A single replica can already fit in available GPU memory.
- Throughput matters more than individual request latency.
- You want fault tolerance (a replica can fail without taking down the service).

In practice, most production deployments combine all four axes. For example, Llama-3 70B on a cluster of 16 × H100 nodes (8 GPUs each, 128 GPUs total) might use TP = 8 (within a node), PP = 2 (across 2 nodes), giving 16 GPUs per model replica, and DP = 8 replicas.

```python
# Illustrative DP request router using asyncio (simplified)
import asyncio
import random
from typing import List

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
    prompts = [f"Explain concept #{i}" for i in range(20)]
    # Dispatch all 20 requests concurrently
    tasks = [router.route(p, max_tokens=100) for p in prompts]
    results = await asyncio.gather(*tasks)
    # All replicas contribute; each individual request is fast
    print(f"Served {len(results)} requests across {len(replicas)} replicas")

# asyncio.run(demo_dp())  # uncomment to run
```

---

## Communication Overhead in the Decode Step

The decode step is uniquely sensitive to communication latency because it processes **one token at a time**. Each step must complete before the next token can be generated. The total per-step time budget for a latency target of, say, 30 tokens/second is only 33 ms. Within that budget, all communication must fit.

### TP Communication Profile

For TP degree $T$ and hidden size $d$, the all-reduce each layer transfers $2d$ bytes per rank (send $d$ bytes, receive $d$ bytes in BF16). With $L$ layers and $A$ all-reduces per layer ($A = 2$ for standard TP):

$$
\text{total TP comm per decode step} = 2 \times d \times L \times A \times \text{dtype\_bytes} \times B
$$

For Llama-3 70B at batch $B = 1$ ($d = 8192$, $L = 80$, $A = 2$, 2 bytes):

$$
2 \times 8192 \times 80 \times 2 \times 2 = 5.2 \text{ MB}
$$

On NVLink (900 GB/s bidirectional), this costs roughly **5.8 µs** — negligible. Crossing PCIe (32 GB/s): about **163 µs** per step, or 6 ms per second of generation at 37 tokens/s — still small but not negligible.

Crossing InfiniBand HDR (25 GB/s effective per rank in a ring): about **208 µs** per step. At 30 tokens/s, that is 6.2 ms/s just in TP communication. This motivates the hard rule: **TP within NVLink islands only**.

### PP Communication Profile

PP communication in decode is a single `send`/`recv` of the hidden-state tensor: $B \times d$ elements. For $B = 64$ and $d = 8192$ in BF16: $64 \times 8192 \times 2 = 1$ MB, sent once per layer per stage boundary. With $P - 1$ stage boundaries, this is much smaller than TP all-reduces and is non-blocking (can overlap with compute on next stage).

### EP (Wide) Communication Profile

For wide-EP, the all-to-all in the decode step has volume approximately:

$$
\text{dispatch all-to-all} = B \times k \times d \times \text{dtype\_bytes}
$$

For $B = 1$ (single-user decode), $k = 8$, $d = 7168$, BF16: $1 \times 8 \times 7168 \times 2 = 114$ KB. Over InfiniBand (25 GB/s): about 4.5 µs. Still manageable even for single-token decode, but scales linearly with batch size, which is why wide-EP benefits greatly from *batched* decode.

!!! warning "The EP small-batch trap"

    For EP deployments, a small decode batch (e.g., B = 1) means most expert GPUs are idle while the single relevant expert GPU works. To amortize the all-to-all cost and keep all GPUs busy, you must batch enough concurrent requests that each expert receives at least one token per step. With $E = 256$ experts and $k = 8$, the expected tokens per expert is $8B/256 = B/32$. For every expert to see at least one token, you need $B \geq 32$. In practice, aim for $B \geq 4 \times E/k = 128$.

---

## When Each Strategy Helps: A Decision Framework

The right combination depends on your SLO (service-level objective), model size, cluster topology, and traffic pattern. Use the following framework:

{{fig:mgpu-decision-tree}}

| Strategy | Helps TTFT | Helps TPOT | Helps Throughput | Communication cost |
|---|---|---|---|---|
| TP (NVLink) | Yes | Yes | Modest | Very low |
| TP (PCIe/IB) | Marginal | Negative | Negative | High |
| PP | No | No | Yes (enables larger model) | Very low |
| EP (intra-node) | No | No | Yes (enables larger MoE) | Low |
| EP (wide, inter-node) | No | Negative at low batch | Yes at high batch | Medium |
| DP (independent replicas) | No | No | Linear | None |
| DP attention (+ EP) | No | No | Yes (avoids KV duplication) | Low (lockstep sync) |
| CP (context parallel) | Yes for very long prompts | Indirectly (more KV fits) | Yes at long context | Medium |

TTFT = time to first token; TPOT = time per output token.

---

## Sizing a Multi-GPU Deployment

### Memory Budget

For a dense model, the GPU memory budget decomposes as:

$$
M_{\text{total}} = M_{\text{weights}} + M_{\text{kv}} + M_{\text{activations}} + M_{\text{framework}}
$$

- **Weights**: $P \times \text{dtype\_bytes} / \text{TP}$ per GPU (divided by TP degree).
- **KV cache**: $2 \times n_{\text{kv\_heads}} \times d_{\text{head}} \times L \times \text{dtype\_bytes} \times S_{\text{max}}$ per request, where $S_{\text{max}}$ is max sequence length. Divided by TP if KV heads are sharded.
- **Activations**: during prefill, $\approx 2 \times B \times S \times d \times L \times \text{dtype\_bytes}$, transient. Negligible during decode.
- **Framework overhead**: typically 1–4 GB for CUDA context, memory allocator, etc.

### Throughput Sizing

The maximum throughput of a DP+TP+PP cluster scales as:

$$
\text{throughput} = D \times \frac{\text{model\_flops\_per\_token}}{\text{GPU\_flops} \times \text{MFU}}
$$

where $D$ is the DP replica count and MFU (Model FLOP Utilization) is the fraction of peak hardware FLOPs actually achieved. For well-tuned serving stacks (continuous batching, FlashAttention, CUDA graphs), MFU during prefill reaches 40–60% on H100s; during decode with small batches, MFU drops to 5–15% because the workload is memory-bandwidth-bound, not compute-bound (see [The Anatomy of LLM Inference: Prefill, Decode & The KV Cache](../07-inference-serving/01-anatomy-inference.html)).

!!! example "Deployment sizing for a 70B model"

    **Target**: serve 100 concurrent users, each generating up to 512 tokens, with TPOT < 50 ms.

    **Step 1: GPU count per replica**

    Weight memory (BF16): 140 GB. Use TP = 4 on 4 × H100 80 GB.
    Per-GPU weight memory: 35 GB.
    Remaining: 45 GB for KV cache.

    KV cache per token (Llama-3 70B, GQA 8 heads, head_dim 128, 80 layers, BF16):

    $$2 \times 8 \times 128 \times 80 \times 2 = 327{,}680 \text{ bytes} \approx 320 \text{ KB}$$

    At 45 GB per GPU: $45 \times 10^9 / 327{,}680 \approx 137{,}000$ tokens of KV cache per GPU.

    For 100 concurrent users × 512 token context: $100 \times 512 = 51{,}200$ tokens. Well within budget — one replica (4 GPUs) can hold all concurrent KV caches.

    **Step 2: Latency check**

    Decode is memory-bandwidth-bound, so model the step as *bytes that must cross HBM*, not FLOPs. The key point is that TP divides those bytes: the four GPUs read their shards **concurrently**, so the wall-clock step time uses per-GPU bytes over per-GPU bandwidth, not the whole 140 GB.

    - Weights per GPU: $140 / 4 = 35$ GB.
    - KV bytes read per step: with 8 KV heads at TP = 4, each GPU holds 2 of them, i.e. 80 KB/token. At 100 requests × 512 tokens of context that is $51{,}200 \times 80\,\text{KB} \approx 4.1$ GB per GPU.

    $$\text{TPOT}_{\text{ideal}} \approx \frac{(35 + 4.1) \times 10^9}{3.35 \times 10^{12}} \approx 12 \text{ ms}$$

    Real kernels sustain roughly 60–75% of peak HBM bandwidth once you include attention, the sampler, and residual launch overhead, so budget **15–20 ms** — comfortably inside the 50 ms SLO, with room to grow the batch. (Had we naively charged all 140 GB to one GPU we would have gotten 42 ms and concluded, wrongly, that the config barely fits.)

    **Step 3: Throughput target**

    Continuous batching produces 100 tokens per decode step — one per in-flight request — so throughput is $B / \text{TPOT}$:

    $$\frac{100}{0.018\ \text{s}} \approx 5{,}500 \text{ tokens/s from one replica}$$

    That nominally meets a 5,000 tokens/s SLO, but with zero headroom for traffic spikes, prefill stealing decode time, or a failed node. Deploy **DP = 2** replicas (8 GPUs) for a 2x margin and single-replica fault tolerance, and add replicas from there as traffic grows.

    **Final configuration**: 8 × H100 GPUs, TP = 4 per replica, DP = 2 replicas — then scale DP linearly with traffic.

### Practical vLLM / SGLang Configuration

```python
# vLLM multi-GPU launch example (CLI equivalent shown as Python API)
from vllm import LLM, SamplingParams

# TP = 4, single node (H100 × 4)
llm = LLM(
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    tensor_parallel_size=4,      # TP degree: each GPU holds 1/4 of each weight
    dtype="bfloat16",
    max_model_len=8192,           # max context length (determines KV cache allocation)
    gpu_memory_utilization=0.90,  # leave 10% headroom
    # For pipeline parallelism across nodes, use:
    # pipeline_parallel_size=2,   # PP = 2 across 2 nodes (requires distributed launch)
    enforce_eager=False,          # enable CUDA graphs for decode
)

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=512,
)

outputs = llm.generate(
    prompts=["Explain tensor parallelism in LLM inference."],
    sampling_params=sampling_params,
)
print(outputs[0].outputs[0].text)
```

```bash
# Multi-node launch with Ray (2 nodes, TP=8 per node, PP=2 across nodes)
# On head node:
ray start --head --port=6379

# On worker nodes:
ray start --address='<head-node-ip>:6379'

# Launch vLLM across both nodes:
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-70B-Instruct \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --port 8000
```

For MoE models like Mixtral 8×22B or DeepSeek-V3 on SGLang, the simplest configuration keeps everything inside one node (the multi-node, DP-attention variant is the one shown earlier):

```bash
# SGLang launch for a large MoE (DeepSeek-V3-style)
# 8 GPUs on one node, TP=8; expert parallelism handled internally
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --tp 8 \
    --dtype bfloat16 \
    --enable-ep-moe \
    --ep-size 8 \
    --port 30000 \
    --trust-remote-code
```

For wide-EP across nodes, vLLM and SGLang now build on DeepSeek's open-source **DeepEP** library — GPU-initiated, RDMA-based all-to-all dispatch/combine kernels, with a dedicated low-latency variant for the decode path — to perform the inter-node MoE routing. On H200 clusters this wide-EP path (DeepEP plus dual-batch compute–communication overlap) has been reported to sustain on the order of 2k output tokens/s per GPU for DeepSeek-V3-class models.

---

## Cross-Cutting Concerns

### KV Cache Sharding Under TP

When TP shards the attention heads, the KV cache is naturally sharded too: each GPU stores only the KV entries for its subset of heads. This is one reason TP improves effective KV cache capacity — the per-GPU KV footprint shrinks by factor $T$ for MHA models. GQA (Grouped-Query Attention) complicates this: if the number of KV heads is smaller than TP degree, some GPUs duplicate KV heads rather than sharding them. See [Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html) for the head-count arithmetic.

### Context Parallelism: The Fifth Axis

Head-count sharding hits a wall, and long context is where it hurts. A model with 8 KV heads gets no further KV reduction past TP = 8, and MLA gets none past TP = 1. **Context parallelism** (CP, also called sequence parallelism in this setting) sidesteps this by partitioning along the *sequence* dimension instead: rank $i$ owns tokens $[i \cdot S/C, (i+1) \cdot S/C)$ of the KV cache for *all* heads. Attention then becomes a distributed reduction — each rank computes partial attention over its own KV shard and the ranks combine partial outputs using the same log-sum-exp rescaling that makes FlashAttention's online softmax work, the same trick that makes FlashAttention's tiled softmax exact (see [FlashAttention I: IO-Awareness & The Online Softmax](../04-kernels-efficiency/02-flash-attention-1.html)) and, applied across devices in a ring, gives Ring Attention. The training-side treatment is in [Distributed Training II: Tensor, Pipeline, Sequence & Expert Parallelism](../03-pretraining/06-distributed-model-parallel.html) and [Long-Context Pretraining & Context Extension](../03-pretraining/13-long-context-pretraining.html).

For prefill this also parallelizes the quadratic attention compute over very long prompts; for decode it is purely a memory play, letting the per-GPU KV footprint keep shrinking past the KV-head limit and so raising the maximum context or concurrency a cluster can hold. vLLM exposes a decode-context-parallel size as a separate knob from TP for precisely this case; because the feature set here is moving fast in both vLLM and SGLang, verify what your installed version supports before designing a deployment around it.

### Disaggregated Prefill and Decode

A complementary approach to multi-GPU parallelism is running prefill and decode on separate GPU pools, which allows each pool to be sized independently. This is covered in [Disaggregated Prefill/Decode & Chunked Prefill](../07-inference-serving/08-disaggregated-chunked-prefill.html). In practice, TP degree may differ between the prefill cluster (benefits from larger TP due to compute-bound nature) and the decode cluster (benefits from smaller TP to avoid over-paying all-reduce cost at small batch).

### CUDA Graphs Under TP

CUDA graphs (capturing the decode step for replay without kernel launch overhead) work transparently under TP: each GPU captures its own graph, and the NCCL all-reduce is part of the captured graph. This requires that the batch size and tensor shapes are fixed at graph capture time, which is why most frameworks maintain a small set of graphs for discrete batch sizes. See [Kernel Fusion, torch.compile, CUDA Graphs & Compilers](../04-kernels-efficiency/09-compilers-fusion.html).

### Speculative Decoding Under TP

When using speculative decoding (see [Speculative Decoding: Draft Models, Medusa, EAGLE & Lookahead](../07-inference-serving/06-speculative-decoding.html)), the draft model can run on a different TP degree from the target model. Draft models are typically small enough for TP = 1. The verification step runs on the target model's TP group and incurs the usual all-reduce cost, but processes multiple tokens simultaneously, so the per-verified-token communication cost is amortized.

---

## DeepSeek-V3 Wide-EP: A Case Study

DeepSeek-V3 (671B MoE, 37B active parameters per token, 256 routed experts) uses a "multi-head latent attention" (MLA) design that dramatically compresses the KV cache, combined with aggressive wide-EP to distribute experts efficiently. The deployment described in the DeepSeek-V3 technical report is the clearest published example of every idea in this chapter used at once — and, importantly, it uses **different configurations for prefill and decode**, which are served on separate GPU pools:

- **Prefill unit**: 4 nodes / 32 GPUs. Attention runs TP = 4 with sequence parallelism, replicated DP = 8; the MoE layers regroup the same 32 GPUs into EP = 32. Redundant copies of high-load experts are deployed on top, and the expert-placement is refreshed periodically from measured routing statistics.
- **Decode unit**: 40 nodes / 320 GPUs — far wider. Attention again runs TP = 4 but with DP = 80, so each rank owns its own requests and its own KV cache; the MoE layers form a single EP = 320 group, which is roughly one expert per GPU plus capacity for redundant and shared experts.
- **Compute–communication overlap**: the report's decode strategy splits the in-flight batch into two micro-batches and overlaps one micro-batch's all-to-all with the other's expert compute, so the network time is hidden rather than added. (The training-side analogue is DualPipe.)
- **Batch size**: very large batches are required — with EP = 320 and $k = 8$, a small batch would leave most of the 320 GPUs holding experts that no token selected.

Notice how the pieces fit: attention is data-parallel because MLA's latent KV cannot be usefully sharded further; TP is capped at 4 and stays inside one NVLink island; EP is the axis that gets stretched across 40 nodes because expert weights are the memory problem. The wide-EP approach trades per-request latency for cluster-level throughput, which suits high-volume API traffic. For latency-sensitive workloads, a smaller EP degree on a single node is preferable.

!!! interview "Interview Corner"

    **Q:** You are serving a 70B dense transformer with TP = 8 on a single 8-GPU node. An interviewer tells you the TPOT is 60 ms and asks you to reduce it to 30 ms without changing the hardware. What are your options and trade-offs?

    **A:** The decode step is memory-bandwidth-bound for small batches. At TP = 8, each GPU already loads only 1/8 of the weight parameters per step, so memory bandwidth is partially amortized. To halve TPOT:

    1. **Increase batch size**: if TPOT of 60 ms corresponds to a batch size of 1, batching 2 requests together roughly halves the per-token time by amortizing weight loads — but doubles user latency if requests arrive serially.
    2. **Quantize weights to INT8 or FP8**: halving weight size halves memory traffic, roughly halving TPOT. Tools like GPTQ, AWQ, or TensorRT-LLM's FP8 mode achieve this with minimal quality loss.
    3. **Enable CUDA graphs**: eliminates kernel launch overhead (~1–5 ms per step for 80 layers), especially significant for small-batch decode.
    4. **Use speculative decoding**: draft 3–5 tokens per step with a small draft model, verify in parallel. Effective TPOT drops by the acceptance rate.
    5. **Reduce sequence length / KV cache size**: shorter context means less KV cache memory to load per attention step, slightly improving bandwidth utilization.
    6. **Upgrade to GQA/MQA** if the model variant allows, reducing KV heads loaded per step.

    The first step should be batching + CUDA graphs; quantization is the highest-impact single change if quality allows.

---

!!! key "Key Takeaways"

    - **Tensor parallelism** splits weight matrices within a layer, requires all-reduce per layer, and is only efficient on NVLink-connected GPUs (within a node). It reduces both memory and latency.
    - **Pipeline parallelism** splits layers across devices, requires only activation pass between stages, is latency-neutral for decode, and is primarily a memory-capacity tool for very large models.
    - **Expert parallelism** routes MoE tokens to GPU-resident experts via all-to-all collectives. Wide-EP extends EP across nodes using InfiniBand, enabling massive MoE capacity at the cost of small-batch efficiency.
    - **Data parallel replicas** are the purest throughput scaling mechanism — identical model copies, independently serving requests, with no communication overhead.
    - **DP attention + EP** is the standard wide-EP layout: attention runs data-parallel so the KV cache is never duplicated (TP cannot shard KV past $n_{\text{kv\_heads}}$, and not at all for MLA), while the same GPUs regroup into one large expert-parallel group for the MoE layers. **Context parallelism** shards the sequence itself and is the axis to reach for when long context, not weight size, is the memory constraint.
    - **Decode is memory-bandwidth-bound**: the all-reduce cost of TP at NVLink speeds is negligible, but TP across InfiniBand can introduce measurable per-token latency.
    - **Communication volume**: TP all-reduce ≈ 2 × d_model bytes per rank per layer; EP all-to-all ≈ B × k × d_model bytes per rank per MoE layer. Both must fit within the per-token time budget.
    - **Sizing rule of thumb**: allocate enough TP to fit model weights with 40–60% GPU memory headroom for KV cache, then add DP replicas until the throughput SLO is met.
    - **Wide-EP (DeepSeek-style)** requires large batch sizes to amortize inter-node all-to-all; it excels for high-throughput API serving of enormous MoE models but adds serving infrastructure complexity.

---

!!! sota "State of the Art & Resources (2026)"
    Multi-GPU and multi-node inference is now a mature discipline, with production systems routinely spanning 512+ GPUs using combinations of tensor, pipeline, expert, and data parallelism. The frontier has shifted toward trillion-parameter MoE models (DeepSeek-V3/R1, Kimi K2, Qwen3-MoE) where wide expert parallelism — now standardized in open frameworks via DeepSeek's open-source DeepEP all-to-all kernels — and compute–communication overlap are the defining challenges. On the hardware side, NVIDIA's Blackwell GB200/GB300 NVL72 (fifth-generation NVLink, 1.8 TB/s per GPU, up to a 72-GPU NVLink domain) is beginning to blur the old intra-node/inter-node boundary, letting far larger TP and EP groups stay on NVLink.

    **Foundational work**

    - [Shoeybi et al., *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism* (2019)](https://arxiv.org/abs/1909.08053) — introduced column/row-parallel tensor parallelism that underpins every major serving framework today.

    **Recent advances (2023–2026)**

    - [Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (2023)](https://arxiv.org/abs/2309.06180) — PagedAttention enables high-throughput multi-GPU serving by eliminating KV-cache fragmentation; foundation of vLLM.
    - [Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs* (2024)](https://arxiv.org/abs/2312.07104) — RadixAttention and compressed FSM scheduling for multi-GPU structured-output serving; NeurIPS 2024.
    - [DeepSeek-AI, *DeepSeek-V3 Technical Report* (2024)](https://arxiv.org/abs/2412.19437) — 671B MoE with EP = 64 across 8 nodes, DualPipe compute–communication overlap, and auxiliary-loss-free expert load balancing.
    - [Kimi Team, *Kimi K2: Open Agentic Intelligence* (2025)](https://arxiv.org/abs/2507.20534) — 1.04T-parameter MoE (32B active, 384 experts, MLA); a current standard-bearer for open trillion-parameter models that serve via wide EP.

    **Open-source & tools**

    - [vllm-project/vllm](https://github.com/vllm-project/vllm) — the dominant open-source LLM serving engine; supports TP, PP, and EP with PagedAttention and continuous batching.
    - [sgl-project/sglang](https://github.com/sgl-project/sglang) — high-performance serving framework with RadixAttention, wide EP support, and strong multi-node scaling.
    - [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) — production-grade inference library with FP8/NVFP4 quantization, fused kernels, and multi-node TP + PP + EP via a PyTorch-native API.
    - [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP) — DeepSeek's open-source expert-parallel communication library: GPU-initiated, RDMA-based all-to-all dispatch/combine kernels (with a low-latency decode variant) now integrated into vLLM and SGLang.
    - [deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA) — optimized CUDA kernels for Multi-head Latent Attention on Hopper GPUs, achieving up to 640 TFlops prefill and 3 TB/s decode bandwidth.

    **Go deeper**

    - [vLLM Blog, *Large Scale Serving: DeepSeek with Wide EP* (2025)](https://vllm.ai/blog/2025-12-17-large-scale-serving) — production wide-EP walkthrough reaching ~2.2k tokens/s/GPU on H200 with DeepEP and dual-batch overlap.
    - [NVIDIA Technical Blog, *GB200 NVL72 Delivers Trillion-Parameter LLM Training and Real-Time Inference*](https://developer.nvidia.com/blog/nvidia-gb200-nvl72-delivers-trillion-parameter-llm-training-and-real-time-inference/) — how the 72-GPU NVLink domain reshapes TP/EP placement for very large MoE serving.

## Further Reading

- Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism," 2019 — original TP formulation.
- Lepikhin et al., "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding," ICLR 2021 — early expert parallelism for Transformers.
- Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity," JMLR 2022 — capacity factor, expert load balancing.
- DeepSeek-AI, "DeepSeek-V3 Technical Report," 2024 — wide-EP serving, MLA, FP8 training, production deployment details.
- Kimi Team, "Kimi K2: Open Agentic Intelligence," 2025 — 1T-parameter open MoE (32B active, 384 experts, MLA) representative of the trillion-parameter models now served with wide expert parallelism.
- Rajbhandari et al., "ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning," SC 2021 — memory hierarchy for serving very large models with CPU/NVMe offload.
- vLLM project (Kwon et al., 2023) and SGLang (Zheng et al., 2024) — open-source references for multi-GPU serving implementations; see [vLLM: Architecture, PagedAttention & Internals](../07-inference-serving/03-vllm-internals.html) and [SGLang: RadixAttention & Structured Programs](../07-inference-serving/04-sglang-radixattention.html).
- Huang et al., "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism," NeurIPS 2019 — pipeline schedule analysis and bubble formulation.

---

## Exercises

**1.** Tensor parallelism must be confined to a single NVLink island, but data-parallel replicas can be spread across nodes, racks, or even datacenters with no penalty. Explain the difference in terms of *what* each strategy communicates and *when*.

??? note "Solution"

    The difference is entirely about the communication that sits on the critical path of a decode step.

    - **TP** requires two all-reduces *per transformer layer* per forward pass (one after attention, one after the MLP). An all-reduce is a blocking collective: the layer cannot proceed until every rank has contributed and received the summed partial. Its volume is roughly $2 \times d_{\text{model}}$ bytes per rank per layer and, crucially, **does not shrink with batch size** — you pay it on every single decode step. On NVLink (about 900 GB/s) the total per-step cost for Llama-3 70B is around 5.8 microseconds and is negligible; over InfiniBand HDR the same traffic costs on the order of 208 microseconds per step, large enough to eat a measurable fraction of a 33 ms token budget. Because the cost is fixed per step and blocking, it only stays cheap on the fast intra-node fabric — hence "TP within NVLink islands only."

    - **DP** replicas are fully independent copies of the model. Different requests go to different replicas, and no tensor is ever exchanged *between* replicas during inference. There is no collective on the decode critical path at all, so the interconnect between replicas is irrelevant — they can be in different datacenters. DP buys linear throughput with zero communication cost; the flip side is that it does nothing for the latency of a single request (each replica is still a full model).

**2.** Consider a dense transformer with $d_{\text{model}} = 5120$ and $L = 40$ layers served at TP with the standard $A = 2$ all-reduces per layer in BF16, decoding at batch size $B = 1$. (a) Compute the total TP all-reduce volume per decode step, per rank. (b) Estimate the wall-clock communication time if the TP group is on NVLink at 900 GB/s. (c) Estimate it if the group is forced across PCIe at 32 GB/s. (d) What does the comparison tell you about placement?

??? note "Solution"

    Use the chapter's formula $\text{total TP comm per step} = 2 \times d \times L \times A \times \text{dtype\_bytes} \times B$, with $B = 1$.

    **(a)** With $d = 5120$, $L = 40$, $A = 2$, dtype = 2 bytes:

    $$2 \times 5120 \times 40 \times 2 \times 2 = 1{,}638{,}400 \text{ bytes} \approx 1.64 \text{ MB}$$

    **(b)** On NVLink at 900 GB/s:

    $$\frac{1.638 \times 10^6}{900 \times 10^9} \approx 1.8 \times 10^{-6}\ \text{s} = 1.8\ \text{microseconds}$$

    **(c)** On PCIe at 32 GB/s:

    $$\frac{1.638 \times 10^6}{32 \times 10^9} \approx 51 \times 10^{-6}\ \text{s} = 51\ \text{microseconds}$$

    **(d)** The volume is identical; only the fabric changes, yet PCIe is about 28x slower per step. 1.8 microseconds is lost in kernel-launch noise, but 51 microseconds per step accumulates: at 30 tokens/s that is about 1.5 ms of pure communication per second of generation, and it grows with $L$ and TP degree. The lesson is that TP placement, not TP volume, is what makes or breaks decode latency — keep the TP group on NVLink.

**3.** You serve Llama-3 70B with **TP = 8** on eight H100 80 GB GPUs. The model has 8 GQA KV heads, head dimension 128, 80 layers, BF16. Recall from the chapter that the total KV cache is $2 \times 8 \times 128 \times 80 \times 2 = 327{,}680$ bytes per token, and that under TP the KV cache is sharded across ranks by attention head. (a) What is the per-GPU KV footprint per token at TP = 8? (b) After weights, suppose 45 GB per GPU remains for KV cache. How many concurrent requests of 8192-token context does one GPU support? (c) Compare with the chapter's TP = 4 example that fit only about 17 such requests, and explain the difference.

??? note "Solution"

    **(a)** There are 8 KV heads and TP = 8, so each GPU owns exactly one KV head. The per-GPU per-token footprint is the total divided by 8:

    $$\frac{327{,}680}{8} = 40{,}960 \text{ bytes/token} = 40 \text{ KB/token}$$

    **(b)** Bytes per 8192-token request on one GPU:

    $$40{,}960 \times 8192 = 335{,}544{,}320 \text{ bytes} \approx 0.335 \text{ GB}$$

    Number of requests in 45 GB:

    $$\frac{45 \times 10^9}{335{,}544{,}320} \approx 134 \text{ concurrent requests}$$

    **(c)** The chapter's TP = 4 example fit about 17 requests because it charged the *full* 320 KB/token to each GPU (it did not shard the KV cache in that estimate). Here, sharding the 8 KV heads across 8 ranks cuts the per-GPU footprint by 8x, so the same 45 GB holds roughly 8x more context — about 134 instead of 17. This is exactly the cross-cutting point that TP shrinks the per-GPU KV footprint by the TP degree (when the KV-head count is at least the TP degree); with GQA, once TP exceeds the number of KV heads the heads must be duplicated and this scaling stops.

**4.** A colleague proposes switching a latency-critical interactive-chat deployment from TP = 8 to PP = 8 (one pipeline stage per GPU) "to spread the model out the same way." Why will this fail to help — and likely hurt — the per-token decode latency (TPOT)? In your answer, contrast what happens during prefill versus decode.

??? note "Solution"

    Pipeline parallelism is **latency-neutral for decode** and is fundamentally a memory-capacity / throughput tool, not a latency tool.

    - **Prefill**: a single batch flows through the stages sequentially. Each of the 8 stages processes $L/8$ layers, so the end-to-end latency is roughly the same as one GPU doing all $L$ layers (there is only one micro-batch, so no bubble), plus a small amount of inter-stage `send`/`recv`. PP neither helps nor meaningfully hurts TTFT here.

    - **Decode**: each step generates one token and is inherently sequential across stages — stage $i$ cannot begin until stage $i-1$ has produced and shipped its hidden state. The total work per token is still all $L$ layers, now strung across 8 devices with $P-1 = 7$ inter-stage synchronizations added on top. So TPOT is at best unchanged and in practice slightly *worse* because of the added hop latency and the loss of overlap.

    By contrast TP = 8 genuinely lowers TPOT: each GPU does $1/8$ of every layer's matmul and loads $1/8$ of the weights per step (decode is memory-bandwidth-bound, so less weight traffic means faster steps), with only cheap NVLink all-reduces added. For interactive chat, keep TP; reach for PP only when the model cannot otherwise fit in a node's NVLink island.

**5.** Using the chapter's `ColParallelLinear` and `RowParallelLinear`, implement a `TensorParallelMLP` that performs the standard up-project / activation / down-project block with exactly **one** all-reduce. Explain why the activation between the two projections needs no communication, and how many all-reduces your block costs.

??? note "Solution"

    The MLP is column-parallel on the way up and row-parallel on the way down. The up-projection leaves the hidden dimension $d_{\text{ff}}$ *partitioned* across ranks; the elementwise activation acts independently on each entry, so a rank can apply it to its own slice with no knowledge of the others. Only the down-projection needs to sum partial contributions, and that single all-reduce lives inside `RowParallelLinear`. Total cost: **one all-reduce** for the whole MLP.

    ```python
    import torch
    import torch.nn.functional as F

    class TensorParallelMLP(torch.nn.Module):
        """
        Column-parallel up-projection -> elementwise activation -> row-parallel
        down-projection. Exactly one all-reduce, contributed by RowParallelLinear.
        """

        def __init__(self, d_model: int, d_ff: int, rank: int, world_size: int):
            super().__init__()
            # Up: d_model -> d_ff, columns split across ranks (no comm here).
            self.up = ColParallelLinear(d_model, d_ff, rank, world_size)
            # Down: d_ff -> d_model, rows split across ranks (all-reduce here).
            self.down = RowParallelLinear(d_ff, d_model, rank, world_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, seq, d_model), replicated across ranks
            h = self.up(x)          # (batch, seq, d_ff / world_size), partitioned
            h = F.gelu(h)           # elementwise: correct on the local partition
            y = self.down(h)        # (batch, seq, d_model), all-reduced -> replicated
            return y
    ```

    Because the activation is elementwise, `gelu(h)` on the partitioned tensor is bit-identical to slicing the full activation — no cross-rank exchange is required between the two linears. The output of `down` is already all-reduced, so it is replicated and ready to feed the next layer's (replicated) input, matching the col-then-row invariant used throughout the chapter. This is exactly why the attention and MLP blocks each cost only one all-reduce, giving the stated $A = 2$ per transformer layer.

**6.** You deploy an MoE model with $E = 128$ experts and top-$k = 4$ routing under wide expert parallelism. (a) Derive, from the expected number of tokens each expert sees, the minimum decode batch size $B$ so that on average every expert receives at least one token per step. (b) What batch does the chapter's practical rule of thumb recommend, and why is it larger than the average-case minimum? (c) What goes wrong if you run this deployment at $B = 1$?

??? note "Solution"

    **(a)** Each of $B$ tokens activates $k$ experts, so the total expert activations per step is $kB$, spread over $E$ experts. The expected activations per expert is

    $$\mathbb{E}[\text{tokens per expert}] = \frac{kB}{E}.$$

    Requiring this to be at least 1:

    $$\frac{kB}{E} \ge 1 \;\Rightarrow\; B \ge \frac{E}{k} = \frac{128}{4} = 32.$$

    **(b)** The chapter recommends $B \ge 4E/k = 4 \times 32 = 128$. The average-case bound of 32 only guarantees *one token per expert on average*; because routing is random, at $B = 32$ many experts will still receive zero tokens on a given step (and others several). Over-provisioning by roughly 4x makes it statistically likely that *every* expert is fed each step, so no GPU sits idle and the all-to-all payload stays balanced. This also amortizes the fixed all-to-all latency over more useful tokens.

    **(c)** At $B = 1$ a single token selects only $k = 4$ experts, so at most 4 of the 128 expert slots do any work while the rest of the cluster is idle. You still pay two all-to-all collectives (dispatch and gather) to move that one token across nodes over InfiniBand, whose latency is 10-50x NVLink. The result is near-zero utilization and a decode step dominated by inter-node communication rather than compute — the "EP small-batch trap." Wide-EP only pays off with large batched decode.
