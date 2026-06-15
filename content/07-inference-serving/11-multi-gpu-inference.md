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

The total GPU count satisfies $N = \text{TP} \times \text{PP} \times \text{DP}$, plus any EP expansion on top of TP. Each axis interacts differently with the two phases of inference — prefill (compute-bound, processes the full prompt in one pass) and decode (memory-bandwidth-bound, processes one token per step). We cover both phases throughout.

---

## Tensor Parallelism

### Mechanism

Tensor parallelism (TP), introduced at production scale by Megatron-LM (Shoeybi et al., 2019), splits individual weight matrices across GPUs so that each GPU handles a column-partition (for a linear projecting *into* the hidden dimension) or a row-partition (projecting *out* of it). Consider a column-parallel linear layer $Y = X W$ where $W \in \mathbb{R}^{d \times k}$:

$$
W = \begin{bmatrix} W_1 & W_2 & \cdots & W_T \end{bmatrix}, \quad
Y_i = X W_i \quad \text{on GPU } i
$$

After the column-parallel layer, each GPU holds $Y_i \in \mathbb{R}^{B \times k/T}$. A subsequent row-parallel layer $Z = Y W'$ is arranged as $W' = \begin{bmatrix} W'_1 \\ \vdots \\ W'_T \end{bmatrix}$ so that GPU $i$ computes $Z_i = Y_i W'_i$ and the full result is $Z = \sum_i Z_i$, recovered by an **all-reduce**.

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

### TP in the Decode Loop

During decode, the batch size is typically small (often 1 to a few hundred). A single all-reduce on a BF16 tensor of shape `[B, 1, d_model]` transfers roughly $2 \times d_{\text{model}} \times B$ bytes per rank over NVLink. For Llama-3 70B with $d_{\text{model}} = 8192$ and $B = 64$:

$$
\text{bytes per all-reduce} = 2 \times 8192 \times 64 = 1\,\text{MB}
$$

At a NVLink bandwidth of around 900 GB/s between two H100s, that is about **1 µs** per all-reduce, negligible compared with the kernel launch overhead. TP therefore has very low communication overhead on a single NVLink island. Crossing PCIe or Ethernet raises this cost by 10–100×, making TP across nodes generally inadvisable.

### Latency vs. Throughput Impact

TP reduces the per-GPU memory footprint of weights by a factor of $T$ and reduces the per-layer compute time by approximately $T$ (each GPU does $1/T$ of the matmul). For a latency-sensitive workload (single user, small batch), TP can deliver near-linear speedup up to the NVLink bandwidth wall. For a throughput-sensitive workload, TP is less efficient than data parallelism because the all-reduces do not scale with batch size — you pay the same communication cost regardless.

!!! example "TP memory saving for Llama-3 70B"

    Llama-3 70B has approximately 70 billion parameters. At BF16 (2 bytes/parameter):

    $$\text{Weight memory} = 70 \times 10^9 \times 2 = 140 \text{ GB}$$

    With TP = 4 across 4 × H100 80 GB GPUs:

    $$\text{Per-GPU weight memory} = 140/4 = 35 \text{ GB}$$

    Leaving 45 GB per GPU for the KV cache. At a KV head dimension of 128, 8 KV heads (GQA), 80 layers, and BF16:

    $$\text{KV cache per token} = 2 \times 8 \times 128 \times 80 \times 2 = 327\,680 \text{ bytes} \approx 320 \text{ KB/token}$$

    So the remaining 45 GB supports roughly $45 \times 10^9 / 327{,}680 \approx 137{,}000$ tokens of context. For a maximum context of 8 K tokens, that accommodates about 17 concurrent requests — a reasonable serving batch.

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

For a batch of 512 tokens, $d_{\text{model}} = 7168$ (DeepSeek-V3 style), $k = 2$ experts, BF16:

$$
512 \times 7168 \times 2 \times 2 = 14.7 \text{ MB per all-to-all}
$$

With two all-to-alls (dispatch and gather) per MoE layer, and 61 MoE layers in DeepSeek-V3, that is about 1.8 GB of network traffic per forward pass — significant but manageable on InfiniBand HDR/NDR at 400 Gb/s.

### Wide Expert Parallelism (Wide-EP): DeepSeek-V3 / DeepSeek-R1

DeepSeek-V3 introduced the concept of **wide EP** in the context of serving their 671B MoE model. Standard practice places EP within a single node (using NVLink for all-to-all). Wide-EP extends EP across *multiple nodes*, using InfiniBand for inter-node all-to-all, allowing a far larger EP degree.

DeepSeek-V3 has $E = 256$ fine-grained experts per MoE layer (plus 1–2 shared experts activated every step), with $k = 8$ routed expert activations per token. Wide-EP allows EP = 64 or more, spreading all 256 experts across many GPU nodes. The routing then dispatches tokens across the cluster via RDMA.

Key trade-offs of wide-EP:

- **Benefit**: each GPU holds fewer experts, so expert weight memory per GPU drops proportionally. Total KV-cache memory scales with the cluster.
- **Cost**: all-to-all over InfiniBand has 10–50× higher latency than NVLink. For small batches, the network becomes the bottleneck.
- **Mitigation**: two techniques help — (1) **compute-communication overlap** (prefetch the next all-to-all while computing the current layer), and (2) **expert load balancing** to avoid hot experts that increase effective network traffic. DeepSeek-V3 also uses an auxiliary loss during training to encourage balanced routing, which directly improves serving efficiency.

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
\text{total TP comm per decode step} = 2 \times d \times L \times A \times \text{dtype\_bytes}
$$

For Llama-3 70B ($d = 8192$, $L = 80$, $A = 2$, 2 bytes):

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
| DP | No | No | Linear | None |

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

    Decode TPOT for Llama-3 70B at batch size 100:
    FLOPs per token ≈ $2 \times 70 \times 10^9 = 140$ GFLOPs (weight loading dominates).
    H100 memory bandwidth: 3.35 TB/s. Weight bytes to load per token: 140 GB (all weights).

    $$\text{TPOT} \approx \frac{140 \times 10^9}{3.35 \times 10^{12}} \approx 42 \text{ ms}$$

    This meets the 50 ms budget. But this is a single-token-at-a-time estimate; continuous batching with 100 concurrent requests amortizes weight loading, achieving better effective TPOT.

    **Step 3: Throughput target**

    100 users × 512 tokens / (42 ms/token) ≈ 1,219 tokens/s from one replica.
    If the traffic SLO requires 5,000 tokens/s, deploy DP = 4 replicas (16 GPUs total).

    **Final configuration**: 16 × H100 GPUs, TP = 4 per replica, DP = 4 replicas.

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

For MoE models like Mixtral 8×22B or DeepSeek-V3 on SGLang:

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

For wide-EP across nodes, SGLang and DeepSeek's own serving infrastructure use a specialized MoE dispatch layer that performs RDMA-based all-to-all via NCCL or a custom communication library.

---

## Cross-Cutting Concerns

### KV Cache Sharding Under TP

When TP shards the attention heads, the KV cache is naturally sharded too: each GPU stores only the KV entries for its subset of heads. This is one reason TP improves effective KV cache capacity — the per-GPU KV footprint shrinks by factor $T$ for MHA models. GQA (Grouped-Query Attention) complicates this: if the number of KV heads is smaller than TP degree, some GPUs duplicate KV heads rather than sharding them. See [Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html) for the head-count arithmetic.

### Disaggregated Prefill and Decode

A complementary approach to multi-GPU parallelism is running prefill and decode on separate GPU pools, which allows each pool to be sized independently. This is covered in [Disaggregated Prefill/Decode & Chunked Prefill](../07-inference-serving/08-disaggregated-chunked-prefill.html). In practice, TP degree may differ between the prefill cluster (benefits from larger TP due to compute-bound nature) and the decode cluster (benefits from smaller TP to avoid over-paying all-reduce cost at small batch).

### CUDA Graphs Under TP

CUDA graphs (capturing the decode step for replay without kernel launch overhead) work transparently under TP: each GPU captures its own graph, and the NCCL all-reduce is part of the captured graph. This requires that the batch size and tensor shapes are fixed at graph capture time, which is why most frameworks maintain a small set of graphs for discrete batch sizes. See [Kernel Fusion, torch.compile, CUDA Graphs & Compilers](../04-kernels-efficiency/09-compilers-fusion.html).

### Speculative Decoding Under TP

When using speculative decoding (see [Speculative Decoding: Draft Models, Medusa, EAGLE & Lookahead](../07-inference-serving/06-speculative-decoding.html)), the draft model can run on a different TP degree from the target model. Draft models are typically small enough for TP = 1. The verification step runs on the target model's TP group and incurs the usual all-reduce cost, but processes multiple tokens simultaneously, so the per-verified-token communication cost is amortized.

---

## DeepSeek-V3 Wide-EP: A Case Study

DeepSeek-V3 (671B MoE, 37B active parameters per token, 256 routed experts) uses a novel "multi-head latent attention" (MLA) design that dramatically compresses the KV cache, combined with aggressive wide-EP to distribute experts efficiently. The production serving configuration reported by the DeepSeek team uses:

- **TP = 8** within each 8-GPU NVLink island.
- **EP = 64** across 64 GPU groups (512 GPUs for a single model instance).
- **Compute–communication overlap**: while each GPU computes expert FFNs on received tokens, the next all-to-all is pre-issued using asynchronous NCCL primitives.
- **Batch size**: very large batches (hundreds to thousands of requests) are required to keep all 256 × 64 / 8 = 2048 GPU-expert slots busy simultaneously.

The wide-EP approach trades per-request latency for massive cluster-level throughput, which suits DeepSeek's API traffic patterns. For latency-sensitive workloads, a smaller EP degree on a single node would be preferred.

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
    - **Decode is memory-bandwidth-bound**: the all-reduce cost of TP at NVLink speeds is negligible, but TP across InfiniBand can introduce measurable per-token latency.
    - **Communication volume**: TP all-reduce ≈ 2 × d_model bytes per rank per layer; EP all-to-all ≈ B × k × d_model bytes per rank per MoE layer. Both must fit within the per-token time budget.
    - **Sizing rule of thumb**: allocate enough TP to fit model weights with 40–60% GPU memory headroom for KV cache, then add DP replicas until the throughput SLO is met.
    - **Wide-EP (DeepSeek-style)** requires large batch sizes to amortize inter-node all-to-all; it excels for high-throughput API serving of enormous MoE models but adds serving infrastructure complexity.

---

!!! sota "State of the Art & Resources (2026)"
    Multi-GPU and multi-node inference is now a mature discipline, with production systems routinely spanning 512+ GPUs using combinations of tensor, pipeline, expert, and data parallelism. The frontier has shifted toward ultra-large MoE models (DeepSeek-V3, Mixtral, Qwen MoE) where wide expert parallelism over InfiniBand and compute–communication overlap are the defining challenges.

    **Foundational work**

    - [Shoeybi et al., *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism* (2019)](https://arxiv.org/abs/1909.08053) — introduced column/row-parallel tensor parallelism that underpins every major serving framework today.
    - [Narayanan et al., *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM* (2021)](https://arxiv.org/abs/2104.04473) — formalizes the composition of TP × PP × DP and the interleaved 1F1B pipeline schedule.

    **Recent advances (2023–2026)**

    - [Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (2023)](https://arxiv.org/abs/2309.06180) — PagedAttention enables high-throughput multi-GPU serving by eliminating KV-cache fragmentation; foundation of vLLM.
    - [Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs* (2024)](https://arxiv.org/abs/2312.07104) — RadixAttention and compressed FSM scheduling for multi-GPU structured-output serving; NeurIPS 2024.
    - [DeepSeek-AI, *DeepSeek-V3 Technical Report* (2024)](https://arxiv.org/abs/2412.19437) — 671B MoE with EP = 64 across 8 nodes, DualPipe compute–communication overlap, and auxiliary-loss-free expert load balancing.

    **Open-source & tools**

    - [vllm-project/vllm](https://github.com/vllm-project/vllm) — the dominant open-source LLM serving engine; supports TP, PP, and EP with PagedAttention and continuous batching.
    - [sgl-project/sglang](https://github.com/sgl-project/sglang) — high-performance serving framework with RadixAttention, wide EP support, and strong multi-node scaling.
    - [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) — canonical reference implementation for TP/PP/DP composition; Megatron Core provides reusable parallelism building blocks.
    - [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) — production-grade inference library with FP8/NVFP4 quantization, fused kernels, and multi-node TP + PP + EP via a PyTorch-native API.
    - [deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA) — optimized CUDA kernels for Multi-head Latent Attention on H800/H100, achieving up to 640 TFlops prefill and 3 TB/s decode bandwidth.

    **Go deeper**

    - [NVIDIA Technical Blog, *Mastering LLM Techniques: Inference Optimization*](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) — practitioner-level walkthrough of TP, PP, sequence parallelism, and quantization with concrete framework guidance.

## Further Reading

- Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism," 2019 — original TP formulation.
- Lepikhin et al., "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding," ICLR 2021 — early expert parallelism for Transformers.
- Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity," JMLR 2022 — capacity factor, expert load balancing.
- DeepSeek-AI, "DeepSeek-V3 Technical Report," 2024 — wide-EP serving, MLA, FP8 training, production deployment details.
- Rajbhandari et al., "ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning," SC 2021 — memory hierarchy for serving very large models with CPU/NVMe offload.
- vLLM project (Kwon et al., 2023) and SGLang (Zheng et al., 2024) — open-source references for multi-GPU serving implementations; see [vLLM: Architecture, PagedAttention & Internals](../07-inference-serving/03-vllm-internals.html) and [SGLang: RadixAttention & Structured Programs](../07-inference-serving/04-sglang-radixattention.html).
- Huang et al., "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism," NeurIPS 2019 — pipeline schedule analysis and bubble formulation.
