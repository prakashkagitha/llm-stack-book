# 1.9 Parallel Computing & Collective Communication

Modern large language models require compute that no single GPU can provide. Training a 70-billion-parameter model in a reasonable amount of time requires hundreds — sometimes thousands — of GPUs working in tight coordination. This chapter explains the substrate that makes that coordination possible: the programming model for parallel computation and the collective communication operations that keep thousands of accelerators synchronized. Everything in [Distributed Training I: Data Parallelism, DDP, ZeRO & FSDP](../03-pretraining/05-distributed-data-parallel.html) and [Distributed Training II: Tensor, Pipeline, Sequence & Expert Parallelism](../03-pretraining/06-distributed-model-parallel.html) builds on the foundations developed here.

We start from first principles — processes, threads, and how GPUs expose parallelism — and build up to the exact cost models engineers use to reason about whether a training job will be network-bound or compute-bound.

## Processes, Threads, and the SPMD Model

### Processes vs. Threads

A **process** is an OS-managed execution context with its own virtual address space, file descriptors, and signal state. Two processes cannot share memory by default; they must use explicit inter-process communication (IPC). A **thread** lives inside a process and shares its address space; threads communicate through shared memory and synchronize via locks, semaphores, or atomic operations.

GPU distributed training almost universally uses the **one-process-per-GPU** model rather than threading across GPUs. The reasons are practical:

1. Python's Global Interpreter Lock (GIL) prevents true CPU-thread-level parallelism, so CPU-side preprocessing and coordination would be serialized.
2. GPU driver libraries (CUDA) are not always thread-safe when the same process owns multiple GPU contexts.
3. Process isolation provides fault containment: a crash on GPU 3 does not corrupt the state of GPU 0.

Each process gets a unique integer **rank** (from 0 to $N-1$, where $N$ is the world size) and belongs to a **process group**. A process group is just a set of ranks with a shared communicator — think of it as a namespace for collective operations.

### Single-Program Multiple-Data (SPMD)

The SPMD programming model is the backbone of all distributed deep learning. Every process runs the *same program*, but operates on *different data* and uses its rank to branch when needed.

```python
# spmd_demo.py — run with:
#   torchrun --nproc_per_node=4 spmd_demo.py
import torch
import torch.distributed as dist

def main():
    # Initialize the default process group. "nccl" for GPU; "gloo" for CPU.
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()        # which GPU am I?
    world_size = dist.get_world_size()  # total GPU count

    # Every rank uses its rank as a local device index.
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Each process creates a tensor whose value equals its rank.
    # This simulates different data shards arriving at different workers.
    x = torch.tensor([float(rank)], device=device)
    print(f"[rank {rank}/{world_size}] before all-reduce: x = {x.item()}")

    # Sum x across all ranks; every rank receives the global sum.
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    print(f"[rank {rank}/{world_size}] after all-reduce:  x = {x.item()}")
    # Expected: sum(0, 1, 2, 3) = 6.0 on every rank.

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

When you launch with `torchrun --nproc_per_node=4`, the launcher forks four processes. Each process discovers its rank and world size through environment variables (`RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`) that `torchrun` sets before calling your script. This is the SPMD contract: every process runs `main()`, but the outcome differs by rank.

## NCCL: The Collective Communication Library

NVIDIA Collective Communications Library (NCCL) is the low-level library that `torch.distributed` calls under the hood when you use a GPU backend. NCCL implements a set of **collective operations**: routines where all $N$ processes participate and the result depends on contributions from all of them. NCCL is highly optimized to exploit NVLink between GPUs on the same node and InfiniBand between nodes.

### The Six Core Collectives

There are six operations you need to know cold. In all of them, let $n$ be the number of ranks and $M$ be the data size (bytes or elements) per rank before the operation.

| Operation | What happens | Output size per rank |
|---|---|---|
| **Broadcast** | rank 0 sends its buffer to all other ranks | $M$ |
| **Reduce** | all ranks contribute; only rank 0 gets the result | $M$ (root only) |
| **All-Reduce** | like Reduce, but every rank gets the result | $M$ |
| **Reduce-Scatter** | reduce across ranks, then scatter slices | $M/n$ |
| **All-Gather** | each rank contributes its slice; all collect the concatenation | $nM$ |
| **All-to-All** | each rank sends a distinct chunk to each other rank | $M$ |

Visually, for $n=4$ ranks each holding a buffer $[a_0, a_1, a_2, a_3]$:


{{fig:collectives-six-dataflow}}


The key insight used in ZeRO-style data parallelism (see [Distributed Training I](../03-pretraining/05-distributed-data-parallel.html)) is that **All-Reduce = Reduce-Scatter followed by All-Gather**. This decomposition lets us interleave communication with computation.

### torch.distributed Collective API

```python
# collectives_demo.py — minimal, heavily-commented examples
import torch
import torch.distributed as dist

def demo_collectives(rank, world_size, device):
    """Demonstrate all six collectives on a simple tensor."""

    # ── 1. Broadcast ──────────────────────────────────────────────────────────
    # Only rank 0 sets meaningful data; others receive it.
    data = torch.zeros(4, device=device)
    if rank == 0:
        data = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
    dist.broadcast(data, src=0)
    # After: every rank holds [1, 2, 3, 4].

    # ── 2. Reduce ─────────────────────────────────────────────────────────────
    x = torch.tensor([float(rank + 1)], device=device)
    dist.reduce(x, dst=0, op=dist.ReduceOp.SUM)
    # After: rank 0 holds sum(1,2,...,world_size); others unchanged.

    # ── 3. All-Reduce ─────────────────────────────────────────────────────────
    x = torch.tensor([float(rank)], device=device)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    # After: every rank holds sum(0, 1, ..., world_size-1).

    # ── 4. Reduce-Scatter ─────────────────────────────────────────────────────
    # Each rank contributes a buffer of size world_size;
    # rank i receives the sum of element i from all ranks.
    input_tensor = torch.arange(world_size, dtype=torch.float32, device=device) + rank
    output_tensor = torch.zeros(1, device=device)
    dist.reduce_scatter(output_tensor, list(input_tensor.split(1)), op=dist.ReduceOp.SUM)
    # Element i on rank r equals (i + r), so element i summed across all ranks is
    #   sum over r of (i + r) = world_size*i + world_size*(world_size-1)/2.
    # Rank i receives element i, so the outputs differ per rank (illustrating the scatter).
    # For world_size=4: rank 0 -> 6.0, rank 1 -> 10.0, rank 2 -> 14.0, rank 3 -> 18.0.

    # ── 5. All-Gather ─────────────────────────────────────────────────────────
    my_chunk = torch.tensor([float(rank)], device=device)
    gathered = [torch.zeros(1, device=device) for _ in range(world_size)]
    dist.all_gather(gathered, my_chunk)
    # After: gathered = [tensor(0), tensor(1), ..., tensor(world_size-1)] on every rank.

    # ── 6. All-to-All ─────────────────────────────────────────────────────────
    # Each rank sends a different value to each other rank.
    # input_list[j] goes TO rank j; output_list[j] comes FROM rank j.
    input_list  = [torch.tensor([rank * 10.0 + j], device=device) for j in range(world_size)]
    output_list = [torch.zeros(1, device=device) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list)
    # Rank 0 receives [0*10+0, 1*10+0, 2*10+0, 3*10+0] = [0, 10, 20, 30]
    # (what each other rank sent to slot 0)
```

### Asynchronous Operations

Every collective has an async variant that returns a `Work` handle, allowing you to overlap computation with communication:

```python
# Overlap gradient all-reduce with backward pass (simplified DDP idea)
import torch
import torch.distributed as dist

def overlapped_allreduce_example(grad_tensor, device):
    """
    Fire off the all-reduce without blocking; do other work; then wait.
    This is the heart of DDP's gradient communication overlap.
    """
    # Non-blocking: returns immediately; communication starts in background.
    handle = dist.all_reduce(grad_tensor, op=dist.ReduceOp.SUM, async_op=True)

    # ... CPU work or other GPU kernels can run here ...
    # For example, you could compute loss.backward() for the next micro-batch.

    # Block until the all-reduce is complete before using grad_tensor.
    handle.wait()

    # Scale by 1/world_size to get the average gradient.
    grad_tensor /= dist.get_world_size()
    return grad_tensor
```

## Ring All-Reduce: The Algorithm That Scaled Deep Learning

Naive all-reduce has rank 0 collect everything then broadcast — a star topology that makes rank 0 a bottleneck. Ring all-reduce eliminates this bottleneck and achieves near-optimal bandwidth utilization.

{{fig:ring-allreduce}}

### Algorithm

Arrange $n$ ranks in a logical ring. The algorithm runs in two phases, each consisting of $n-1$ steps:

**Phase 1 — Reduce-Scatter:** Each rank sends one chunk and receives one chunk per step, accumulating partial sums.

**Phase 2 — All-Gather:** Each rank sends fully-reduced chunks around the ring until every rank has every chunk.

{{fig:ring-ar-reduce-scatter-trace}}

The total data sent per rank per phase is $\frac{n-1}{n} \cdot M$ (each step sends $M/n$ bytes). Both phases together send $2 \cdot \frac{n-1}{n} \cdot M$ bytes per rank, approaching $2M$ for large $n$. This is also the lower bound — ring all-reduce is **bandwidth-optimal**.

### Tree All-Reduce

For small messages where the latency of $2(n-1)$ steps dominates, a **binary tree** or **recursive halving/doubling** algorithm is better. In $\log_2 n$ steps, a tree reduces all values. This is the regime of optimizer-state broadcast or short control messages.

The tradeoff:

| Algorithm | Latency (steps) | Bandwidth per rank | Best for |
|---|---|---|---|
| Ring | $2(n-1)$ | $\frac{2(n-1)}{n} M$ → $2M$ | Large messages (gradients) |
| Tree (binary) | $2 \log_2 n$ | $O(M \log n)$ | Small messages, latency-sensitive |
| Recursive halving | $\log_2 n$ reduce + $\log_2 n$ broadcast | Near-optimal | Medium messages |

NCCL selects the algorithm automatically based on message size and topology.

## Bandwidth and Latency Cost Models

To reason about whether your training run is compute-bound or communication-bound, you need a simple cost model. The **alpha-beta model** (also called the LogP model in the literature) approximates the time to send a message of $B$ bytes as:

$$
T(B) = \alpha + \frac{B}{\beta}
$$

where $\alpha$ is the **latency** (startup cost in seconds, independent of message size) and $\beta$ is the **bandwidth** (bytes per second, the asymptotic rate for large messages).

For a ring all-reduce of $M$ total bytes across $n$ ranks:

$$
T_{\text{ring-AR}}(M) = 2(n-1)\alpha + \frac{2(n-1)}{n} \cdot \frac{M}{\beta}
$$

As $n$ grows, the latency term $2(n-1)\alpha$ becomes painful for many small messages — this is why gradient bucketing (combining small gradients into large tensors before communicating) is critical in practice.

For very large $n$, the bandwidth term simplifies:

$$
T_{\text{ring-AR}}(M) \approx 2\alpha n + 2\frac{M}{\beta}
$$

The bandwidth term is constant in $n$ — adding more GPUs doesn't change the bandwidth cost. The latency term grows linearly, which is why ring all-reduce across thousands of GPUs requires hierarchical approaches.

{{fig:alpha-beta-cost-model}}

!!! example "Worked Example: Is a DDP Training Step Bottlenecked by Communication?"

    **Setup:** 8 GPUs on a single node connected via NVLink. We are training a 1.3B parameter model in bf16 (2 bytes/parameter). After the backward pass, we need to all-reduce gradients.

    **Gradient buffer size:**
    $$M = 1.3 \times 10^9 \times 2 \text{ bytes} = 2.6 \text{ GB}$$

    **NVLink bandwidth** (H100 SXM): on the order of 900 GB/s aggregate bidirectional, or roughly $\beta \approx 450$ GB/s per direction for a single link. With 8 GPUs in a ring, effective bandwidth is approximately $\beta_{\text{eff}} \approx 300$ GB/s (after ring inefficiency and protocol overhead — use this as an engineering estimate, not a specification).

    **Communication time:**
    $$T_{\text{comm}} \approx \frac{2M}{\beta_{\text{eff}}} = \frac{2 \times 2.6 \text{ GB}}{300 \text{ GB/s}} \approx 17 \text{ ms}$$

    **Compute time (forward + backward):** For a 1.3B model on a batch of 32 tokens × 2048 context on H100, a rough estimate is on the order of 200–400 ms total. Communication is therefore on the order of 5–10% of step time — manageable, and further reducible by overlapping all-reduce with backward pass.

    **Cross-node scenario:** If instead the 8 GPUs span 2 nodes connected by 200 Gb/s InfiniBand ($\beta \approx 25$ GB/s):
    $$T_{\text{comm}} \approx \frac{2 \times 2.6}{25} \approx 208 \text{ ms}$$

    Now communication *exceeds* compute time. The only remedies are gradient compression, ZeRO with reduce-scatter/all-gather split, or tensor/pipeline parallelism to reduce the communicated volume.

## Network Topology: NVLink, NVSwitch, and InfiniBand

Communication cost is not uniform across a cluster. You need to understand the physical topology to reason about where collectives should be placed.

### Intra-Node: NVLink and NVSwitch

**NVLink** is NVIDIA's proprietary high-bandwidth GPU-to-GPU interconnect. On an H100 SXM server, each GPU has 18 NVLink 4.0 lanes, providing roughly 900 GB/s bidirectional bandwidth per GPU — vastly higher than PCIe Gen 5 (on the order of 128 GB/s bidirectional).

**NVSwitch** is a crossbar switch that connects all GPUs on a node with full NVLink bandwidth — every GPU can communicate with every other GPU simultaneously at full speed, rather than routing through a chain. An 8-GPU DGX H100 uses four NVSwitch 3.0 chips, providing an effective 3.6 TB/s of all-to-all bandwidth within the node.

This topology means:


{{fig:nvswitch-crossbar-topology}}


NCCL exploits NVSwitch by using its own **all-reduce algorithm** that leverages the all-to-all connectivity, avoiding a sequential ring in favor of a one-shot reduce pattern across the switch fabric.

### Inter-Node: InfiniBand

Between nodes, current clusters use **InfiniBand** (IB). Common configurations:

- **HDR (200 Gb/s):** ~25 GB/s effective unidirectional per port
- **NDR (400 Gb/s):** ~50 GB/s effective unidirectional per port
- **XDR (800 Gb/s):** emerging in 2025 deployments

A cluster of nodes is connected through an IB fabric, often organized as a **fat-tree** or **dragonfly** topology, providing full bisection bandwidth in principle (but subject to hotspots in practice). The IB Host Channel Adapter (HCA) on each node connects the CPUs and GPUs to the fabric; GPU Direct RDMA (Remote Direct Memory Access) allows the NIC to read/write GPU HBM directly, bypassing the CPU.

### Hierarchical Communication

Because intra-node bandwidth is ~10–50× higher than inter-node bandwidth, production NCCL jobs use **hierarchical collectives**: first reduce within a node (fast, NVLink), then reduce across nodes (slower, IB), then broadcast back. NCCL implements this automatically via its topology detection.


{{fig:hierarchical-allreduce-stages}}


## A Complete torch.distributed Training Step

The following implements a minimal DDP-style training loop from scratch, showing exactly where each collective fires and why.

```python
# minimal_ddp.py
# Run: torchrun --nproc_per_node=4 minimal_ddp.py
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

# ──────────────────────────────────────────────────────────────────────────────
# 1. Initialize distributed environment
# ──────────────────────────────────────────────────────────────────────────────
dist.init_process_group(backend="nccl")
rank       = dist.get_rank()
world_size = dist.get_world_size()
device     = torch.device(f"cuda:{rank}")
torch.cuda.set_device(device)

# ──────────────────────────────────────────────────────────────────────────────
# 2. Build a toy model. Every rank starts with a random init;
#    we must broadcast weights from rank 0 so all ranks start identically.
# ──────────────────────────────────────────────────────────────────────────────
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 10)

    def forward(self, x):
        return self.linear(x)

model = TinyModel().to(device)

# Broadcast initial parameters so all ranks are identical.
for param in model.parameters():
    dist.broadcast(param.data, src=0)

# ──────────────────────────────────────────────────────────────────────────────
# 3. Build a toy dataset. DistributedSampler ensures each rank gets
#    a non-overlapping shard of the data, which is the "data parallel" part.
# ──────────────────────────────────────────────────────────────────────────────
N = 1024
dataset = TensorDataset(
    torch.randn(N, 128),
    torch.randint(0, 10, (N,))
)
sampler    = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# ──────────────────────────────────────────────────────────────────────────────
# 4. Training loop: forward → backward → all-reduce gradients → optimizer step
# ──────────────────────────────────────────────────────────────────────────────
for epoch in range(2):
    sampler.set_epoch(epoch)  # ensures different shuffling per epoch

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()

        # ── All-Reduce: average gradients across all ranks ──────────────────
        # Without this, each rank would update with its own local gradient,
        # causing model divergence after the first step.
        for param in model.parameters():
            if param.grad is not None:
                # SUM then divide → average; or use ReduceOp.AVG if supported.
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= world_size
        # ────────────────────────────────────────────────────────────────────

        optimizer.step()

    if rank == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.4f}")

dist.destroy_process_group()
```

In production, `torch.nn.parallel.DistributedDataParallel` (DDP) wraps this loop and uses a more sophisticated gradient bucketing scheme to maximize overlap between backward computation and all-reduce communication. The principle is identical to the manual loop above.

## Process Groups and Custom Communicators

So far we've used the **default process group** containing all ranks. For pipeline parallelism and tensor parallelism (see [Distributed Training II](../03-pretraining/06-distributed-model-parallel.html)), you need **sub-groups** where only a subset of ranks communicate.

```python
# process_groups.py — illustrating tensor-parallel sub-groups
import torch.distributed as dist

def build_tp_dp_groups(tp_size: int):
    """
    Construct tensor-parallel (TP) and data-parallel (DP) process groups.

    For world_size=8 and tp_size=4:
      TP groups (communicate tensor shards):  [0,1,2,3] and [4,5,6,7]
      DP groups (communicate gradients):      [0,4], [1,5], [2,6], [3,7]

    Within a TP group, all-gather/reduce-scatter move activation shards.
    Across DP groups, all-reduce moves gradient updates.
    """
    world_size = dist.get_world_size()
    rank       = dist.get_rank()
    assert world_size % tp_size == 0
    dp_size = world_size // tp_size

    # ── Tensor-parallel groups ────────────────────────────────────────────────
    tp_group = None
    for i in range(dp_size):
        ranks_in_group = list(range(i * tp_size, (i + 1) * tp_size))
        group = dist.new_group(ranks=ranks_in_group)
        if rank in ranks_in_group:
            tp_group = group

    # ── Data-parallel groups ──────────────────────────────────────────────────
    dp_group = None
    for j in range(tp_size):
        ranks_in_group = list(range(j, world_size, tp_size))
        group = dist.new_group(ranks=ranks_in_group)
        if rank in ranks_in_group:
            dp_group = group

    return tp_group, dp_group

# Usage:
# tp_group, dp_group = build_tp_dp_groups(tp_size=4)
# dist.all_reduce(tensor_shard, group=tp_group)   # fast, NVLink
# dist.all_reduce(grad,         group=dp_group)   # slower, IB
```

NCCL treats each `new_group` call as a new communicator; it internally runs its topology detection and algorithm selection within that group.

## Collective Performance: Benchmarking and Profiling

Understanding actual achieved bandwidth versus theoretical peak is essential for diagnosing training slowdowns.

```python
# benchmark_allreduce.py — measure achieved bandwidth for a ring all-reduce
import time
import torch
import torch.distributed as dist

def benchmark_all_reduce(message_bytes: int, n_iters: int = 50):
    """
    Measure the achieved bandwidth of all-reduce for a given message size.
    Returns (latency_ms, bandwidth_GBps).
    """
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    device     = torch.device(f"cuda:{rank}")

    # Allocate a float32 buffer of the requested size.
    n_elements = message_bytes // 4
    buf = torch.randn(n_elements, device=device)

    # Warm up: let NCCL initialize its internal state.
    for _ in range(5):
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    dist.barrier()

    # Timed runs.
    start = time.perf_counter()
    for _ in range(n_iters):
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    dist.barrier()
    elapsed = time.perf_counter() - start

    latency_ms = elapsed / n_iters * 1000
    # Busbw formula: 2*(n-1)/n * message_bytes / time  (ring all-reduce)
    busbw_GBps = (2 * (world_size - 1) / world_size * message_bytes) / (elapsed / n_iters) / 1e9

    if rank == 0:
        print(f"Message: {message_bytes/1e6:.1f} MB | "
              f"Latency: {latency_ms:.2f} ms | "
              f"Bus BW: {busbw_GBps:.1f} GB/s")
    return latency_ms, busbw_GBps


# Example results on an 8-GPU DGX H100 (NVLink):
# Message:   1.0 MB | Latency:  0.15 ms | Bus BW:  52.3 GB/s
# Message:  64.0 MB | Latency:  1.23 ms | Bus BW: 413.8 GB/s
# Message: 512.0 MB | Latency:  7.84 ms | Bus BW: 521.0 GB/s
# (Illustrative figures; actual results depend on driver version and cluster state)
```

The **busbw** (bus bandwidth) formula $\frac{2(n-1)}{n} \cdot M / T$ is the standard metric because it accounts for the ring algorithm's traffic pattern and is comparable across cluster sizes. Compare busbw to the theoretical NVLink bandwidth to understand efficiency.

For production profiling, use PyTorch Profiler with NCCL tracing enabled:

```python
from torch.profiler import profile, ProfilerActivity, schedule

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=5),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./logs/profiler"),
    record_shapes=True,
    with_stack=True,
) as prof:
    for step, (x, y) in enumerate(dataloader):
        train_step(x, y, model, optimizer)
        prof.step()

# In TensorBoard → Distributed → you will see the NCCL ops timeline,
# showing exactly how much time is spent in AllReduce vs compute.
```

!!! interview "Interview Corner"

    **Q:** You're training a 70B-parameter model in bf16 across 64 GPUs (8 nodes × 8 GPUs) with pure data parallelism. Each training step takes 1 second of compute. Your inter-node InfiniBand provides 400 Gb/s per link (one link per node). Is your job communication-bound? What would you do?

    **A:** Gradient buffer size = $70 \times 10^9 \times 2$ bytes $= 140$ GB. For a ring all-reduce across $n = 64$ ranks, the bandwidth term is approximately $\frac{2(n-1)}{n} M \approx 2 \times 140 = 280$ GB of data per rank moved through the slowest link (IB). At 400 Gb/s = 50 GB/s effective, that's $\frac{280}{50} \approx 5.6$ seconds of communication — more than 5× the compute time, so yes, badly communication-bound.

    Remedies in priority order — the goal is to cut communication *volume* on the slow inter-node hop, not just memory: (1) adopt **tensor parallelism** (TP-4 within node, over fast NVLink): each data-parallel rank then holds only 1/4 of the parameters, so its gradient buffer shrinks from 140 GB to ~35 GB and the inter-node all-reduce moves ~4x less data per rank (~70 GB instead of ~280 GB), while TP's own all-reduces stay on-node where bandwidth is 10-50x higher; (2) use **hierarchical / hybrid sharding** (e.g., FSDP `HYBRID_SHARD` / HSDP): shard within a node and replicate across nodes, keeping the heavy reduce-scatter and all-gather on NVLink and sending only reduced gradient shards over InfiniBand; (3) apply **gradient compression** (PowerSGD, Top-K sparsification) if convergence is acceptable; (4) **increase per-GPU batch size or use gradient accumulation** to raise the compute-to-communication ratio (more compute per all-reduce). Note that plain **ZeRO Stage 1/2** does *not* fix this bottleneck: its reduce-scatter + all-gather sums to essentially the same total volume as DDP's all-reduce (see [Distributed Training I](../03-pretraining/05-distributed-data-parallel.html)) — it removes the memory redundancy of replicated optimizer and gradient state, not the communication.

## All-to-All and Expert Parallelism

The **all-to-all** collective is less common in vanilla data parallelism but is critical for **Mixture-of-Experts (MoE)** models (see [Mixture-of-Experts (MoE) Architectures](../02-transformer/09-mixture-of-experts.html)). In MoE, tokens are routed to expert sub-networks that may live on different GPUs. Sending each token to its assigned expert is exactly an all-to-all operation.

```python
# all_to_all_moe_sketch.py — illustrating token dispatch in MoE
import torch
import torch.distributed as dist

def moe_dispatch(tokens: torch.Tensor, expert_ids: torch.Tensor, n_experts: int):
    """
    tokens: [seq_len, hidden_dim]  — tokens on this GPU
    expert_ids: [seq_len]          — which expert each token should go to
    n_experts: total number of experts, one per GPU

    Returns: tokens sorted and dispatched to the correct expert GPU.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert n_experts == world_size  # one expert per GPU for simplicity

    seq_len, hidden = tokens.shape

    # Count how many tokens go to each expert.
    counts = torch.bincount(expert_ids, minlength=world_size)  # [world_size]

    # All-to-all the counts so each rank knows what it will receive.
    recv_counts = torch.zeros_like(counts)
    dist.all_to_all_single(recv_counts, counts)

    # Sort tokens by destination expert.
    sort_idx = expert_ids.argsort()
    sorted_tokens = tokens[sort_idx]  # [seq_len, hidden], grouped by expert

    # Build send/recv lists for all-to-all.
    send_splits = counts.tolist()
    recv_splits = recv_counts.tolist()

    # Allocate receive buffer.
    total_recv = recv_counts.sum().item()
    recv_buf = torch.zeros(total_recv, hidden, device=tokens.device, dtype=tokens.dtype)

    # Perform the actual all-to-all data transfer.
    dist.all_to_all_single(
        recv_buf,
        sorted_tokens,
        output_split_sizes=recv_splits,
        input_split_sizes=send_splits,
    )

    # recv_buf now contains tokens that belong to this GPU's expert.
    return recv_buf, recv_counts
```

The all-to-all cost is $O(M)$ where $M$ is the total token volume, and unlike all-reduce it does not benefit from the ring structure — it is inherently limited by the bisection bandwidth of the network.

{{fig:all-to-all-moe-dispatch}}

## Key Considerations for Choosing Collectives

Different distributed training strategies use different collectives as their communication primitive:

| Strategy | Key Collective | Why |
|---|---|---|
| Data Parallelism (DDP) | All-Reduce (gradients) | Every rank needs the globally averaged gradient |
| ZeRO Stage 1 | Reduce-Scatter (grads) + All-Gather (params) | Shard optimizer state; materialize only on demand |
| ZeRO Stage 2 | Reduce-Scatter (grads) + All-Gather (params) | Also shard gradients |
| ZeRO Stage 3 / FSDP | All-Gather (forward), Reduce-Scatter (backward) | Full parameter sharding |
| Tensor Parallelism | All-Reduce (within TP group) | Reconstruct activation shards after matmul split |
| Expert Parallelism | All-to-All | Route tokens to their expert GPUs |
| Pipeline Parallelism | Point-to-Point (send/recv) | Pass activations between pipeline stages |

Understanding this table is what separates an engineer who can debug a distributed training job from one who cannot. If your profiler shows an all-gather is slow, you know FSDP is materializing parameters and the bottleneck is memory bandwidth for the parameter gather, not compute. If all-to-all is slow, you have an MoE routing or load-imbalance problem.

!!! warning "Common Pitfall: Collective Deadlock"

    Every collective is a **barrier** — all participating ranks must call it before any rank can proceed. A common bug is a conditional collective:

    ```python
    # WRONG — only rank 0 calls all-reduce; all others hang forever.
    if rank == 0:
        dist.all_reduce(tensor)  # deadlock!

    # CORRECT — every rank calls every collective, always.
    dist.all_reduce(tensor)
    if rank == 0:
        process(tensor)  # branch after the collective, not before.
    ```

    Similarly, mismatched `new_group` calls will deadlock because `new_group` internally calls a barrier across all ranks to initialize the communicator.

!!! tip "Practitioner Tip: Gradient Bucketing"

    By default, PyTorch DDP groups parameters into 25 MB buckets and fires an all-reduce for each bucket as soon as all gradients in that bucket are ready during backward. This pipeline — compute gradients for later layers while earlier-layer all-reduces are in flight — is responsible for most of the overlap benefit. You can tune the bucket size with `DDP(model, bucket_cap_mb=50)`. Larger buckets reduce the number of all-reduce calls (lower latency overhead) but delay when communication starts (less overlap). Optimal bucket size depends on your model's backward time per layer.

!!! key "Key Takeaways"

    - Distributed GPU training uses the SPMD model: one process per GPU, all running the same program, differentiated by rank.
    - The six core collectives are: Broadcast, Reduce, All-Reduce, Reduce-Scatter, All-Gather, and All-to-All. All-Reduce is equivalent to Reduce-Scatter followed by All-Gather.
    - Ring all-reduce achieves near-optimal bandwidth by distributing load uniformly across $n$ ranks; its bandwidth cost is $\approx 2M$ bytes per rank regardless of $n$.
    - The alpha-beta cost model $T = \alpha + B/\beta$ separates latency (per-call startup) from bandwidth (asymptotic rate). Large messages are bandwidth-bound; small messages are latency-bound.
    - NVLink/NVSwitch provides ~10–50× more bandwidth than InfiniBand, making intra-node collectives much cheaper than inter-node ones. Hierarchical collectives exploit this.
    - Different parallelism strategies use different collectives: data parallelism uses all-reduce, ZeRO/FSDP uses reduce-scatter+all-gather, tensor parallelism uses all-reduce within a sub-group, and MoE uses all-to-all.
    - Process groups (`dist.new_group`) let you create communicators over subsets of ranks, enabling the 3D parallelism (DP × TP × PP) used by Megatron-LM.
    - Every collective is a synchronization barrier — missing a collective call or making it conditional causes deadlock.

!!! sota "State of the Art & Resources (2026)"
    Collective communication is a mature but rapidly evolving field: ring all-reduce remains the workhorse for large-message gradient synchronization, while NVLink/NVSwitch fabrics, hierarchical collectives, and algorithm-selection heuristics in NCCL 2.x continue to push practical bandwidth efficiency close to hardware limits at scales of 10,000+ GPUs.

    **Foundational work**

    - [Thakur, Rabenseifner & Gropp, *Optimization of Collective Communication Operations in MPICH* (2005)](https://journals.sagepub.com/doi/10.1177/1094342005051521) — introduces the ring, recursive halving/doubling, and binary-tree algorithms with the alpha-beta cost model that every distributed training engineer still uses.
    - [Shoeybi et al., *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism* (2019)](https://arxiv.org/abs/1909.08053) — canonical description of tensor-parallel all-reduce patterns within a transformer layer.
    - [Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* (2020)](https://arxiv.org/abs/1910.02054) — shows how reduce-scatter + all-gather replaces all-reduce to enable full optimizer/gradient/parameter sharding.

    **Recent advances (2021–2024)**

    - [Li et al., *PyTorch Distributed: Experiences on Accelerating Data Parallel Training* (2020)](https://arxiv.org/abs/2006.15704) — details DDP gradient bucketing, hook-based overlap, and the design decisions in `torch.distributed`.
    - [Narayanan et al., *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM* (2021)](https://arxiv.org/abs/2104.04473) — demonstrates composing data, tensor, and pipeline parallelism across 3,072 GPUs; defines the 3D-parallelism collective pattern.
    - [Jiang et al., *MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs* (2024)](https://arxiv.org/abs/2402.15627) — production engineering report covering hierarchical collectives, IB topology tuning, and fault recovery at extreme scale.

    **Open-source & tools**

    - [NVIDIA/nccl](https://github.com/NVIDIA/nccl) — the authoritative implementation of GPU collective communication; topology-aware algorithm selection, NVLink and InfiniBand support.
    - [NVIDIA/nccl-tests](https://github.com/NVIDIA/nccl-tests) — benchmarking suite for measuring achieved bus-bandwidth across all NCCL collective operations; standard tool for cluster acceptance testing.
    - [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) — reference implementation of 3D-parallel transformer training; shows exactly which collectives fire in each parallelism dimension.

    **Go deeper**

    - [PyTorch torch.distributed API docs](https://docs.pytorch.org/docs/2.7/distributed.html) — full API reference for process groups, backends, collective calls, and async operations.
    - [NVIDIA NCCL Documentation — Overview](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html) — official guide covering algorithm selection, topology detection, environment variables, and tuning.
    - [Understanding NCCL Tuning to Accelerate GPU-to-GPU Communication (NVIDIA Blog, 2025)](https://developer.nvidia.com/blog/understanding-nccl-tuning-to-accelerate-gpu-to-gpu-communication/) — explains NCCL's cost model, dynamic scheduler, and tuner-plugin interface for cluster-specific optimization.

## Further Reading

- **Thakur, Rabenseifner, Gropp (2005):** "Optimization of Collective Communication Operations in MPICH" — the foundational paper on ring and tree all-reduce algorithms and their cost models.
- **Li et al. (2020):** "PyTorch Distributed: Experiences on Accelerating Data Parallel Training" — covers DDP bucketing, hook-based gradient compression, and the design choices in `torch.distributed`.
- **NVIDIA NCCL Documentation and Source** (github.com/NVIDIA/nccl) — the authoritative reference for NCCL algorithm selection, topology detection, and tuning knobs.
- **Rajbhandari et al. (2020):** "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (DeepSpeed) — explains how reduce-scatter + all-gather enables full optimizer/gradient/parameter sharding.
- **Jiang et al. (2022):** "Megascale: Scaling Large Language Model Training to More Than 10,000 GPUs" — describes hierarchical collectives, network topology design, and reliability engineering at cluster scale.
- **Shoeybi et al. (2019):** "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" — the canonical reference for tensor-parallel all-reduce patterns within a transformer layer.
