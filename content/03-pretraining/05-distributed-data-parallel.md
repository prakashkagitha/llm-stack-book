# 3.5 Distributed Training I: Data Parallelism, DDP, ZeRO & FSDP

A single modern GPU can hold a few billion parameters of model state, but the models we actually want to train have tens or hundreds of billions of parameters — and even when the *parameters* fit, the *optimizer states* and *activations* usually do not. Data parallelism is the first and most important answer to the question "how do I use more than one GPU?" It is conceptually simple — replicate the model, split the batch — but the engineering required to make it fast (overlapping communication with computation) and memory-efficient (sharding state across devices) is the difference between a training run that finishes and one that runs out of memory on the first step.

This chapter develops data parallelism from first principles. We start with the math of gradient averaging, build a from-scratch Distributed Data Parallel (DDP) wrapper to expose its bucketing-and-overlap machinery, then confront the memory problem head-on with a precise accounting of where every byte goes. That accounting motivates the **ZeRO** family of optimizations and its PyTorch-native cousin **FSDP** (Fully Sharded Data Parallel), which shard optimizer states, gradients, and finally parameters themselves across the data-parallel group. We assume you have read [Parallel Computing & Collective Communication](../01-foundations/09-parallel-collectives.html) — we lean heavily on the collectives (all-reduce, reduce-scatter, all-gather) and the alpha-beta cost model developed there. The orthogonal forms of parallelism (tensor, pipeline, expert) are the subject of [Distributed Training II: Tensor, Pipeline, Sequence & Expert Parallelism](../03-pretraining/06-distributed-model-parallel.html).

## Data Parallelism From First Principles

### The Core Idea: Replicate the Model, Shard the Batch

In **data parallelism (DP)**, every one of the $N$ workers (one per GPU) holds a complete copy of the model. We take a global mini-batch $B$ and split it into $N$ disjoint **local batches** of size $b = B / N$. Each worker $i$ computes a forward and backward pass on its own local batch, producing a **local gradient** $g_i$. The key fact that makes this correct is that the gradient of a sum-of-losses objective is the sum of per-example gradients.

Consider the standard average loss over a mini-batch:

$$
\mathcal{L}(\theta) = \frac{1}{B} \sum_{j=1}^{B} \ell(x_j; \theta)
$$

Because differentiation is linear, the gradient decomposes exactly across any partition of the batch into $N$ shards $S_1, \dots, S_N$:

$$
\nabla_\theta \mathcal{L} = \frac{1}{B} \sum_{j=1}^{B} \nabla_\theta \ell(x_j; \theta) = \frac{1}{N} \sum_{i=1}^{N} \underbrace{\left( \frac{1}{b} \sum_{j \in S_i} \nabla_\theta \ell(x_j; \theta) \right)}_{\text{local gradient } g_i}
$$

So if each worker computes the *average* gradient over its own local batch of size $b$, then the **mean of the local gradients equals the true full-batch gradient**. This is why the canonical DP gradient synchronization is an **all-reduce with a mean** (equivalently, sum-then-divide-by-$N$): after the all-reduce, every worker holds the identical averaged gradient $\bar g = \frac{1}{N}\sum_i g_i$, applies the identical optimizer update, and so every replica stays bit-for-bit in lockstep.

{{fig:ddp-dp-allreduce-mean}}

### Two Invariants You Must Maintain

Data parallelism is only correct if two invariants hold. Both are easy to violate, and violations produce silent divergence rather than crashes.

1. **Identical initialization.** All replicas must start from the same parameters. In practice you either construct the model with a fixed seed on every rank, or you build it on rank 0 and `broadcast` the weights (and any registered buffers, like BatchNorm running statistics) to all other ranks. If replicas start different, averaging gradients does nothing to bring them back together.

2. **Identical gradients after sync.** Every parameter's gradient must be all-reduced before the optimizer step. Forget one parameter (a common bug when you add a new module and its grad isn't in the bucket), and that parameter drifts independently on each rank.

There is a third, subtler concern: the *effective batch size* grows with $N$. Going from 1 GPU to 64 GPUs at fixed local batch size multiplies your global batch by 64, which changes the optimization dynamics. You typically must re-tune the learning rate (the linear scaling rule, warmup) — that interaction is the subject of [Learning Rate Schedules, Warmup, Batch Size & Hyperparameters](../03-pretraining/10-lr-schedules-hparams.html). Data parallelism is a *systems* technique that has *statistical* consequences.

!!! note "Aside: DP scales throughput, not model size"

    Plain data parallelism does **not** let you train a bigger model — every GPU still holds the full model, so the largest trainable model is bounded by one GPU's memory. DP scales *throughput* (tokens/sec) by processing more data in parallel. The memory-saving variants later in this chapter (ZeRO, FSDP) are what let DP also scale *model size*, by sharding the replicated state across the group.

### The Naive Implementation and Why It's Slow

A correct but slow data-parallel loop simply all-reduces each gradient after `loss.backward()` completes. We saw exactly this in [Parallel Computing & Collective Communication](../01-foundations/09-parallel-collectives.html). The problem is twofold:

- **No overlap.** The backward pass finishes *entirely*, then communication starts. Compute and network sit idle waiting for each other. With backward and all-reduce each taking, say, 200 ms, you've serialized them into 400 ms.
- **Too many tiny messages.** A transformer has thousands of parameter tensors, many small (layer norms, biases). All-reducing each one separately pays the latency cost $\alpha$ thousands of times per step. Recall the alpha-beta model $T(B) = \alpha + B/\beta$: for tiny $B$, the fixed $\alpha$ dominates and bandwidth is wasted.

Fixing both is exactly what production DDP does, and it's worth building from scratch to understand.

## DDP Internals: Bucketing and Communication Overlap

PyTorch's `DistributedDataParallel` (DDP) is the workhorse of single-model-fits-on-one-GPU training. Its two central ideas are **gradient bucketing** and **autograd-hook-driven overlap**. The result is that the gradient all-reduce of layer $L$ runs on the network *while the backward pass is still computing gradients for layer $L-1$ on the GPU*.

{{fig:data-parallel}}

### The Backward Pass Produces Gradients in Reverse Order

Backpropagation computes gradients layer by layer from the output back to the input. The gradient for the *last* layer is ready first; the gradient for the *first* layer is ready last. This ordering is the key opportunity: as soon as the last layer's gradient is computed, we can start its all-reduce immediately — there is no reason to wait for the rest of the backward pass.

DDP registers an **autograd hook** on every parameter. When that parameter's `.grad` is populated during backward, the hook fires. The hook's job is to mark the gradient "ready" and, when enough gradients are ready, kick off communication.

### Bucketing: Coalescing Gradients to Amortize Latency

Rather than all-reduce each parameter's gradient individually (thousands of tiny messages), DDP groups parameters into **buckets** — contiguous flat buffers of, by default, about 25 MB. When *all* gradients in a bucket have been produced, DDP fires a single all-reduce for the whole bucket. This trades thousands of latency-bound small all-reduces for a few dozen bandwidth-bound large ones.

Buckets are assigned in (roughly) **reverse order of the forward pass**, so that the bucket containing the last-computed-in-forward / first-computed-in-backward parameters is filled and ready to communicate earliest. The ordering matters: a poor ordering would leave a bucket waiting on one straggler gradient, stalling overlap.

{{fig:ddp-bucketed-overlap-timeline}}

The win is large: in the ideal case, the only exposed (non-overlapped) communication is the *last* bucket's all-reduce, which has no remaining backward compute to hide behind. Everything else is free.

### A From-Scratch DDP Wrapper

Here is a minimal but faithful reimplementation of DDP's core mechanics — broadcast-on-init, per-parameter hooks, bucketing, and async all-reduce overlap. It is runnable and heavily commented so you can see every moving part.

```python
# tiny_ddp.py — a from-scratch DistributedDataParallel.
# Run: torchrun --nproc_per_node=4 tiny_ddp.py
import torch
import torch.nn as nn
import torch.distributed as dist


class TinyDDP(nn.Module):
    """
    A minimal DistributedDataParallel that demonstrates the two core ideas:
      1. Broadcast parameters at init so all replicas start identical.
      2. Bucket gradients and all-reduce each bucket asynchronously as soon
         as it is full during the backward pass, overlapping comm with compute.
    """

    def __init__(self, module: nn.Module, bucket_cap_mb: float = 25.0):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.bucket_bytes_cap = int(bucket_cap_mb * 1024 * 1024)

        # ── Invariant 1: identical initialization ───────────────────────────────
        # Broadcast params AND buffers (e.g. BN running stats) from rank 0.
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)
        for b in self.module.buffers():
            dist.broadcast(b.data, src=0)

        # ── Assign parameters to buckets in REVERSE registration order ──────────
        # Reverse order approximates reverse-of-forward = order-of-backward,
        # so the first bucket to fill is the first to be communicable.
        params = [p for p in self.module.parameters() if p.requires_grad]
        params = list(reversed(params))

        self._buckets = []          # list of dicts describing each bucket
        cur, cur_bytes = [], 0
        for p in params:
            pbytes = p.numel() * p.element_size()
            if cur and cur_bytes + pbytes > self.bucket_bytes_cap:
                self._buckets.append(cur)
                cur, cur_bytes = [], 0
            cur.append(p)
            cur_bytes += pbytes
        if cur:
            self._buckets.append(cur)

        # Map each parameter -> (bucket_index) and track readiness counters.
        self._param_to_bucket = {}
        for bidx, bucket in enumerate(self._buckets):
            for p in bucket:
                self._param_to_bucket[p] = bidx

        self._pending_work = []     # async all-reduce handles to wait on
        self._ready_counts = [0] * len(self._buckets)
        self._register_hooks()

    def _register_hooks(self):
        """Attach a post-accumulate-grad hook to every parameter."""
        for p in self.module.parameters():
            if not p.requires_grad:
                continue
            # This hook fires AFTER p.grad has been accumulated in backward.
            p.register_post_accumulate_grad_hook(self._make_hook(p))

    def _make_hook(self, param):
        def hook(p):
            bidx = self._param_to_bucket[p]
            self._ready_counts[bidx] += 1
            # When every parameter in this bucket has a gradient, communicate.
            if self._ready_counts[bidx] == len(self._buckets[bidx]):
                self._all_reduce_bucket(bidx)
        return hook

    def _all_reduce_bucket(self, bidx):
        """Flatten the bucket's grads, async all-reduce, store the handle."""
        grads = [p.grad for p in self._buckets[bidx]]
        flat = torch._utils._flatten_dense_tensors(grads)  # one contiguous buffer
        # async_op=True returns immediately; NCCL runs in the background while
        # the backward pass keeps producing earlier-layer gradients.
        handle = dist.all_reduce(flat, op=dist.ReduceOp.SUM, async_op=True)
        self._pending_work.append((handle, flat, self._buckets[bidx]))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        """
        Call once after loss.backward(). Waits on all in-flight all-reduces,
        scatters the reduced flat buffers back into each .grad, and averages.
        """
        for handle, flat, bucket in self._pending_work:
            handle.wait()                      # block until this bucket is reduced
            flat /= self.world_size            # SUM -> MEAN
            # Unflatten the contiguous buffer back into per-parameter grads.
            synced = torch._utils._unflatten_dense_tensors(
                flat, [p.grad for p in bucket]
            )
            for p, g in zip(bucket, synced):
                p.grad.copy_(g)
        # Reset for next iteration.
        self._pending_work.clear()
        self._ready_counts = [0] * len(self._buckets)


# ── Usage ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    model = nn.Sequential(
        nn.Linear(1024, 4096), nn.GELU(),
        nn.Linear(4096, 4096), nn.GELU(),
        nn.Linear(4096, 1024),
    ).to(device)
    ddp = TinyDDP(model, bucket_cap_mb=25.0)
    opt = torch.optim.AdamW(ddp.parameters(), lr=1e-3)

    for step in range(10):
        x = torch.randn(32, 1024, device=device)   # different data per rank
        y = ddp(x).pow(2).mean()                    # toy loss
        opt.zero_grad(set_to_none=True)
        y.backward()                                # hooks fire, comm overlaps
        ddp.finish_gradient_synchronization()       # wait + average
        opt.step()
        if rank == 0:
            print(f"step {step}: loss={y.item():.4f}")

    dist.destroy_process_group()
```

The production DDP differs from this toy in important ways — it rebuilds bucket order after the first iteration based on the *observed* backward order, handles unused parameters (the `find_unused_parameters` flag), uses a dedicated CUDA stream for communication, and overlaps the bucket *copy-out* too — but the essence is exactly what's above: hooks fire as grads are produced; full buckets are all-reduced asynchronously; one final wait synchronizes everything.

!!! tip "Practitioner tip: the no_sync() context for gradient accumulation"

    When doing gradient accumulation (several micro-batches per optimizer step), you do **not** want an all-reduce after every micro-batch — only after the last one. Real DDP provides `model.no_sync()`, a context manager that suppresses the hooks' communication, letting `.grad` accumulate locally. Run the first $k-1$ micro-batches under `no_sync()` and the last one normally. This cuts communication volume by a factor of $k$ at the cost of holding the accumulated gradient locally. Forgetting it is a classic "why is my 4-step accumulation 4× slower than expected" bug.

!!! warning "Common pitfall: DDP with unused parameters"

    If your forward pass conditionally skips a sub-module (e.g. an auxiliary head used only some steps), that module's parameters never receive a gradient, their hooks never fire, and the corresponding bucket never completes — so DDP hangs at the `wait()` forever. The fix is `DistributedDataParallel(model, find_unused_parameters=True)`, which traverses the autograd graph to detect which parameters were used and marks the rest "ready" immediately. It costs a graph traversal per step, so only enable it when you actually have conditionally-unused parameters.

## The Memory Problem: Where Every Byte Goes

Plain DDP replicates *everything* on every GPU. To see why that's a problem — and what ZeRO/FSDP fix — we need a precise accounting of training memory. There are four consumers: parameters, gradients, optimizer states, and activations. The first three are what ZeRO shards; activations are handled separately (activation checkpointing, covered in [Memory-Efficient Training: Checkpointing, Offloading & LoRA Math](../04-kernels-efficiency/10-memory-efficient-training.html)).

### Mixed-Precision Adam: The Standard Memory Model

The dominant training recipe is **mixed-precision with Adam/AdamW** (see [Mixed Precision, bf16 & FP8 Training](../03-pretraining/08-mixed-precision-fp8.html) and [Optimizers: SGD, Adam, Adafactor, Lion, Muon & Shampoo](../03-pretraining/09-optimizers.html)). With $\Psi$ parameters, the per-GPU memory for the *persistent* model state (everything except activations) in the canonical bf16-compute / fp32-master-weights setup is:

| State | Precision | Bytes per parameter |
|---|---|---|
| Parameters (bf16, for fwd/bwd) | 2 bytes | $2\Psi$ |
| Gradients (bf16) | 2 bytes | $2\Psi$ |
| Master weights (fp32 copy) | 4 bytes | $4\Psi$ |
| Adam first moment $m$ (fp32) | 4 bytes | $4\Psi$ |
| Adam second moment $v$ (fp32) | 4 bytes | $4\Psi$ |

Summing: $2\Psi + 2\Psi + 4\Psi + 4\Psi + 4\Psi = 16\Psi$ bytes. This is the famous **"16 bytes per parameter"** rule for mixed-precision Adam. (Some accountings put gradients in fp32 too, giving $18\Psi$; the exact split depends on the framework, but the order of magnitude is fixed.)

The crucial observation behind ZeRO: of those 16 bytes, **12 bytes — the fp32 master weights and the two Adam moments — are "optimizer state" that is only touched once per step, during `optimizer.step()`**. There is no reason for every GPU to store a full copy. Only the 2 bytes of bf16 params (needed every forward/backward) and 2 bytes of gradient (produced every backward) genuinely need to be present on each GPU at all times — and even those can be sharded if we're willing to gather them on demand.

!!! example "Worked example: a 7.5B model on 80 GB GPUs"

    Take $\Psi = 7.5 \times 10^9$ parameters (a GPT-2-XL-to-Llama-7B-class model). Persistent state under mixed-precision Adam:

    $$
    16 \times \Psi = 16 \times 7.5\times10^9 = 120 \times 10^9 \text{ bytes} = 120 \text{ GB}
    $$

    That already **exceeds a single 80 GB H100**, before we've allocated a single byte for activations. Pure DDP simply cannot train this model — every replica needs 120 GB. Now spread it over $N = 64$ GPUs:

    - **DDP:** 120 GB per GPU. **Does not fit. Impossible.**
    - **ZeRO-1** (shard the 12 bytes of optimizer state): per-GPU $= 2\Psi + 2\Psi + 12\Psi/N = 30 + 1.5\,\text{GB-ish}$. Concretely $4\Psi + 12\Psi/64 = 30\text{ GB} + 1.4\text{ GB} \approx 31.4$ GB. **Fits with room for activations.**
    - **ZeRO-2** (also shard the 2 bytes of gradients): $2\Psi + (2\Psi + 12\Psi)/N = 15\text{ GB} + 14\Psi/64 = 15 + 1.6 \approx 16.6$ GB.
    - **ZeRO-3 / FSDP** (also shard the 2 bytes of params): $16\Psi/N = 120/64 \approx 1.9$ GB per GPU.

    The progression $120 \to 31 \to 17 \to 1.9$ GB per GPU is the entire point of ZeRO. As $N \to \infty$, ZeRO-3 drives per-GPU model-state memory toward zero. This is what makes 100B+ parameter training on commodity 80 GB GPUs possible.

### Activations Are the Other Half of the Story

The $16\Psi$ figure is *persistent* state. **Activations** — the intermediate tensors saved during the forward pass for use in the backward pass — are transient but can dwarf the parameters for long sequences and large batches. Roughly, activation memory scales as $O(\text{batch} \times \text{seq\_len} \times \text{layers} \times \text{hidden})$ and is *not* sharded by data parallelism (each GPU has its own local batch's activations). ZeRO/FSDP do not directly reduce activation memory; that's the job of activation checkpointing (recompute instead of store) and sequence/tensor parallelism. Keep this in mind: ZeRO-3 can shrink model state to nearly nothing yet still OOM on activations. See [Memory-Efficient Training: Checkpointing, Offloading & LoRA Math](../04-kernels-efficiency/10-memory-efficient-training.html).

## ZeRO: Sharding the Redundancy Away

**ZeRO** (Zero Redundancy Optimizer), introduced by Rajbhandari et al. in the DeepSpeed project, is the observation that data parallelism stores $N$ identical copies of state that is mostly idle. ZeRO partitions that state across the $N$ data-parallel workers so each owns only a $1/N$ slice, and reconstructs the full tensors on demand via collectives — trading memory for a modest amount of extra communication. It comes in three cumulative stages.

### ZeRO-1: Shard Optimizer States

Stage 1 partitions only the **optimizer states** (the fp32 master weights and Adam's $m$, $v$) — the 12-bytes-per-parameter that are touched only at `optimizer.step()`. Each GPU $i$ owns the optimizer state for parameter-shard $i$ (a contiguous $1/N$ slice of the flattened parameter space).

The step proceeds:

1. Forward/backward as in DDP — every GPU has full bf16 params and computes full local gradients.
2. **Reduce-scatter** the gradients: instead of an all-reduce (which gives every GPU the full averaged gradient), reduce-scatter gives GPU $i$ only the averaged gradient for *its* shard. This is cheaper — recall all-reduce = reduce-scatter + all-gather — so stage 1 already halves gradient communication relative to a naive all-reduce.
3. Each GPU updates *only its shard's* fp32 master weights using its $m, v$, producing updated bf16 params for its shard.
4. **All-gather** the updated bf16 parameters so every GPU again has the full model for the next forward pass.

Memory per GPU: $2\Psi + 2\Psi + 12\Psi/N$. Communication volume is essentially the same as DDP ($2\Psi$ worth: a reduce-scatter plus an all-gather), so **ZeRO-1 gives a large memory saving for free**.

### ZeRO-2: Also Shard Gradients

Stage 2 additionally partitions the **gradients**. The insight: once GPU $i$ has reduce-scattered its gradient shard and only needs *that* shard to update *its* optimizer state, there's no reason to keep the full gradient buffer materialized everywhere. As each bucket's gradient is reduced, GPU $i$ keeps only its slice and frees the rest.

Memory per GPU: $2\Psi + (2\Psi + 12\Psi)/N = 2\Psi + 14\Psi/N$. Communication is again the same $2\Psi$ (reduce-scatter of grads + all-gather of params). ZeRO-2 is the sweet spot for many setups: it shards everything except the bf16 parameters (which must be present for forward/backward), at the same communication cost as DDP.

### ZeRO-3: Also Shard Parameters

Stage 3 takes the final step and partitions the **parameters themselves**. Now no GPU holds the full model at rest — GPU $i$ permanently stores only its $1/N$ slice of the bf16 parameters (plus its grad shard and optimizer shard). To run the forward pass, parameters must be reconstructed *layer by layer, just in time*:

- **Forward:** before computing layer $\ell$, **all-gather** layer $\ell$'s parameters from all GPUs, compute the layer, then **free** the gathered parameters immediately. Only one (or a few prefetched) layers' worth of full parameters is materialized at any instant.
- **Backward:** **all-gather** the layer's parameters again (they were freed), compute gradients, **reduce-scatter** the gradients to their owner, free everything.

Memory per GPU: $16\Psi/N$ for model state — it shards linearly with the number of GPUs. The cost is an **extra all-gather**: ZeRO-3 communicates roughly $3\Psi$ worth (all-gather params in forward, all-gather params in backward, reduce-scatter grads) versus $2\Psi$ for ZeRO-1/2 — about **1.5× the communication** of DDP. That extra all-gather is the price of never storing the full model.

{{fig:zero-stages-sharding-ladder}}

### ZeRO-Offload and ZeRO-Infinity

DeepSpeed extends ZeRO with **offload**: push the sharded optimizer states (and optionally gradients/params) to CPU RAM or NVMe SSD, fetching them only when needed. ZeRO-Infinity orchestrates GPU↔CPU↔NVMe movement to train models far larger than aggregate GPU memory. The tradeoff is bandwidth: PCIe/CPU-memory bandwidth is an order of magnitude below HBM, so offload is for "fits nowhere else" regimes, not for speed. We treat offload mechanics in [Megatron-LM, DeepSpeed & Parallelism in Practice](../03-pretraining/07-megatron-deepspeed.html).

!!! note "Aside: ZeRO is still data parallelism"

    A common confusion is to call ZeRO-3 "model parallelism." It is not. Every GPU still processes a *different data shard* and the *full computation* of every layer (after gathering that layer's params) — that's data parallelism. The model is *sharded for storage* but *replicated for compute*. Contrast with tensor parallelism (Chapter 3.6), where each GPU computes a *different slice of every layer's math* on the *same* data. ZeRO shards memory; TP shards compute. They are orthogonal and routinely combined.

## FSDP: PyTorch-Native Fully Sharded Data Parallel

**FSDP** (Fully Sharded Data Parallel) is PyTorch's native implementation of the ZeRO-3 idea, built into `torch.distributed`. Conceptually it is ZeRO-3; in practice it has its own vocabulary and knobs worth knowing because FSDP (and its successor FSDP2) is what most PyTorch-native large-model training uses today.

### The FSDP Unit and the Flat Parameter

FSDP groups parameters into **FSDP units** (a unit is typically one transformer block, set via an *auto-wrap policy*). Within a unit, all parameters are flattened and concatenated into a single 1-D **FlatParameter**, which is then sharded evenly across the $N$ ranks: rank $i$ owns the $i$-th contiguous slice. Sharding a flat buffer (rather than each tensor individually) means the all-gather/reduce-scatter operate on big contiguous messages — bandwidth-efficient, exactly the bucketing lesson from DDP applied to sharding.

The lifecycle of a unit during a step:

{{fig:fsdp-unit-lifecycle}}

Because only one unit (plus prefetched neighbors) is unsharded at a time, peak parameter memory is roughly (one block's full params) + (this rank's $1/N$ shard of the whole model), not the whole model. The smaller the unit, the lower the peak — but the more frequent and smaller the all-gathers, hurting bandwidth. Wrapping per transformer block is the standard compromise.

### Prefetching, Communication Streams, and Overlap

FSDP overlaps just like DDP, but in *both* directions. It runs collectives on a separate CUDA stream and **prefetches**: while computing block $\ell$, it issues the all-gather for block $\ell+1$ (forward) or $\ell-1$ (backward) so the params are ready by the time compute needs them. This hides the all-gather latency behind compute. The reduce-scatter of gradients in backward overlaps with the backward compute of the next block — the same compute/comm pipeline as DDP, now with an extra gather to hide.

### A Complete, Runnable FSDP Example

Here is an end-to-end FSDP training script for a small transformer, using the modern API. It shows the auto-wrap policy, mixed precision, activation checkpointing, sharded optimizer, and — critically — how to save a checkpoint when no rank has the full model.

```python
# fsdp_train.py — end-to-end FSDP on a toy transformer.
# Run: torchrun --nproc_per_node=4 fsdp_train.py
import functools
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper, CheckpointImpl, apply_activation_checkpointing,
)


# ── A tiny transformer block — the unit FSDP will shard around ───────────────────
class Block(nn.Module):
    def __init__(self, d=1024, h=16):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class ToyTransformer(nn.Module):
    def __init__(self, vocab=50_000, d=1024, n_layers=12):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d) for _ in range(n_layers)])
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        x = self.emb(idx)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Build the model on meta/CPU; FSDP will shard it onto the GPUs.
    model = ToyTransformer()

    # ── Auto-wrap policy: make every `Block` its own FSDP unit ──────────────────
    # Each Block becomes a FlatParameter sharded across ranks. Params are
    # all-gathered just-in-time per block and freed right after.
    auto_wrap = functools.partial(
        transformer_auto_wrap_policy, transformer_layer_cls={Block}
    )

    # ── Mixed precision: bf16 compute, fp32 reductions for numerical safety ─────
    mp = MixedPrecision(
        param_dtype=torch.bfloat16,      # params gathered/used in bf16
        reduce_dtype=torch.float32,      # grad reduce-scatter accumulates in fp32
        buffer_dtype=torch.bfloat16,
    )

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        mixed_precision=mp,
        sharding_strategy=ShardingStrategy.FULL_SHARD,   # ZeRO-3 semantics
        device_id=device,
        use_orig_params=True,            # play nicely with torch.compile & optims
    )

    # ── Activation checkpointing: recompute block activations in backward to
    #    save activation memory (orthogonal to FSDP param sharding) ─────────────
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ),
        check_fn=lambda m: isinstance(m, Block),
    )

    # The optimizer sees only this rank's parameter shards (sharded optim state).
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))

    model.train()
    for step in range(20):
        idx = torch.randint(0, 50_000, (8, 512), device=device)   # local batch
        logits = model(idx)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), idx.view(-1)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()        # all-gathers + reduce-scatters happen inside, overlapped
        # Gradient clipping across shards: FSDP provides a sharded-aware clip.
        model.clip_grad_norm_(1.0)
        opt.step()
        if rank == 0:
            print(f"step {step:3d} | loss {loss.item():.4f}")

    # ── Checkpointing: no single rank holds the full model under FULL_SHARD.
    #    Gather a full (unsharded) state dict ONLY on rank 0 to write to disk. ──
    save_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_cfg):
        full_sd = model.state_dict()       # all-gathers params, materialized on rank 0
        if rank == 0:
            torch.save(full_sd, "ckpt_full.pt")
    # For large models prefer SHARDED_STATE_DICT (each rank writes its own shard),
    # which avoids materializing the whole model on one GPU/host.

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

A few load-bearing details in that script:

- **`ShardingStrategy.FULL_SHARD`** is ZeRO-3. FSDP also offers `SHARD_GRAD_OP` (ZeRO-2: shard grads + optimizer state but keep params resident — less communication, more memory) and `NO_SHARD` (plain DDP). `HYBRID_SHARD` shards *within* a node and replicates *across* nodes, which avoids slow inter-node all-gathers for the parameter reconstruction — a critical optimization on clusters where intra-node NVLink is far faster than inter-node InfiniBand.
- **`reduce_dtype=torch.float32`** keeps the gradient reduce-scatter in fp32 even though params are bf16 — bf16 gradient summation across many ranks loses precision (bf16 has only 8 mantissa bits), so reducing in fp32 protects convergence at negligible cost.
- **`use_orig_params=True`** exposes the original parameter tensors (not just the opaque FlatParameter), which is required for `torch.compile`, parameter-group-specific learning rates, and selective freezing.
- **Checkpointing is genuinely different under sharding.** No rank has the whole model, so saving requires either an all-gather to materialize a full state dict on rank 0 (simple, but a memory spike and a bottleneck for huge models) or a *sharded* state dict where each rank writes its own slice (scalable). This is covered in depth in [Checkpointing, Fault Tolerance & Long-Running Jobs](../03-pretraining/12-checkpointing-fault-tolerance.html).

!!! warning "Common pitfall: clipping and gradient norms under sharding"

    Never call the plain `torch.nn.utils.clip_grad_norm_` on FSDP parameters. Each rank only sees its *shard* of every gradient, so a naive per-rank norm is the norm of a slice, not the global norm — clipping with it scales gradients incorrectly and silently destabilizes training. Use the FSDP-aware `model.clip_grad_norm_(...)`, which all-reduces the squared norm across ranks before computing the global norm. The same care applies to any logging of gradient statistics: reduce across ranks first.

### FSDP2: Per-Parameter Sharding with DTensor

The newer **FSDP2** redesigns sharding around per-parameter `DTensor` (distributed tensor) representations instead of the monolithic FlatParameter. Each parameter is individually a sharded `DTensor`, which composes cleanly with tensor parallelism (you can have a parameter that is both TP-sharded along one mesh dimension and FSDP-sharded along another), removes FlatParameter's awkward edge cases around mixed dtypes and frozen parameters, and integrates better with `torch.compile`. The mental model — all-gather a unit's params before compute, free after, reduce-scatter grads — is unchanged; FSDP2 mostly makes the *composition* with other parallelism dimensions (the "$N$-D parallelism" of Chapter 3.6 and 3.7) far cleaner via a unified `DeviceMesh`.

## Choosing and Combining Strategies

### A Decision Guide

The right choice is a function of how the model and its state fit relative to one GPU's memory:

| Situation | Recommended strategy | Rationale |
|---|---|---|
| Model + Adam state ($16\Psi$) fits comfortably on one GPU | **DDP** | Lowest communication, simplest, fastest |
| Model fits but optimizer state is tight | **ZeRO-1** / FSDP `SHARD_GRAD_OP`-lite | Free memory win, same comm as DDP |
| Model fits but grads + optim state too big | **ZeRO-2** / FSDP `SHARD_GRAD_OP` | Shard grads+optim, params stay resident |
| Model state ($16\Psi$) exceeds one GPU | **ZeRO-3** / FSDP `FULL_SHARD` | Only way to fit; pay ~1.5× comm |
| Even sharded params exceed aggregate GPU memory | **ZeRO-Infinity** (NVMe offload) or add tensor/pipeline parallelism | Spill to CPU/NVMe, or shard *compute* too |
| Inter-node network is the bottleneck | **HYBRID_SHARD** | Shard within node, replicate across nodes |

The general principle: **use the least sharding that fits.** Each step from DDP toward ZeRO-3 trades communication for memory. If DDP fits, sharding only adds overhead. The moment it doesn't fit, climb the ladder exactly as far as needed.

### Composing With Model Parallelism

For the largest models, data parallelism alone is insufficient and is combined with tensor and pipeline parallelism into **N-D parallelism**. A typical layout assigns a `DeviceMesh` with axes for data-parallel (FSDP), tensor-parallel, and pipeline-parallel dimensions, with FSDP sharding the data-parallel axis and TP/PP sharding the model axis. The interaction — for instance, FSDP-sharding the *tensor-parallel* parameters so that you shard along two mesh dimensions at once — is exactly what FSDP2 + DTensor makes ergonomic. We develop these combinations in [Distributed Training II: Tensor, Pipeline, Sequence & Expert Parallelism](../03-pretraining/06-distributed-model-parallel.html) and [Megatron-LM, DeepSpeed & Parallelism in Practice](../03-pretraining/07-megatron-deepspeed.html).

!!! example "Worked example: communication cost of FSDP on a slow network"

    Suppose a 13B model trained with FSDP `FULL_SHARD` across $N = 16$ GPUs spanning 2 nodes, where the inter-node link is the bottleneck at an effective $\beta \approx 25$ GB/s (200 Gb/s InfiniBand). Per optimizer step, FSDP moves roughly $3\Psi$ worth of bf16 traffic (all-gather params in fwd + all-gather in bwd + reduce-scatter grads), where each "$\Psi$ worth" is $\Psi \times 2$ bytes $= 26$ GB.

    Naively that's $3 \times 26 = 78$ GB of traffic per rank, but the *per-rank* volume of a reduce-scatter or all-gather over $N$ ranks is $\frac{N-1}{N}\Psi \cdot 2 \approx 26$ GB each, so:

    $$
    T_{\text{comm}} \approx \frac{3 \times 26 \text{ GB}}{25 \text{ GB/s}} \approx 3.1 \text{ s/step}
    $$

    If forward+backward compute is, say, 1.5 s, the job is badly communication-bound — the extra all-gather of ZeRO-3 is exposed because the inter-node link is slow. **Switching to `HYBRID_SHARD`** keeps the bandwidth-heavy parameter all-gathers *inside* each node (fast NVLink, $\beta \approx 300$ GB/s) and only does a once-per-step inter-node all-reduce of gradients — roughly $2\Psi$ over the slow link $\approx \frac{2 \times 26}{25} \approx 2.1$ s, partly overlappable, and the all-gathers nearly vanish from the critical path. This is why hybrid sharding is the default for multi-node FSDP whenever the model *fits* in a single node's aggregate memory.

!!! interview "Interview Corner"

    **Q:** You're training a 30B-parameter model with mixed-precision AdamW on 32 GPUs (4 nodes × 8). With DDP you immediately OOM. Walk me through what's consuming the memory, which sharding strategy you'd pick, and the communication tradeoff you're accepting.

    **A:** Persistent state under mixed-precision Adam is ~16 bytes/param: 2 (bf16 params) + 2 (bf16 grads) + 4 (fp32 master) + 4 (Adam $m$) + 4 (Adam $v$). For 30B that's $16 \times 30\text{B} = 480$ GB per replica — impossible on an 80 GB GPU, so DDP (which replicates all of it) OOMs as expected.

    I'd use **FSDP `FULL_SHARD` (ZeRO-3)**, which shards all three categories across the 32 GPUs: per-GPU model state drops to $480/32 = 15$ GB, leaving headroom for activations (which I'd further cut with activation checkpointing). The tradeoff is communication: ZeRO-3 adds an all-gather of parameters in *both* forward and backward (to reconstruct each layer just-in-time), so total comm is ~$3\Psi$ versus DDP's ~$2\Psi$ — about 1.5×. FSDP hides most of it by prefetching the next layer's all-gather on a separate stream during the current layer's compute, so on fast intra-node NVLink it's nearly free.

    Since I'm multi-node and the inter-node InfiniBand is much slower than NVLink, the parameter all-gathers across nodes would be the bottleneck. If the model *fits within one node's* 8×80=640 GB, I'd switch to **`HYBRID_SHARD`**: shard within each node (fast) and only all-reduce gradients across nodes once per step (the cheap part), keeping the expensive all-gathers on NVLink. If it doesn't fit in a node, I'd combine FSDP with tensor parallelism (TP=8 within node, FSDP across nodes) to shrink the per-GPU footprint further while keeping cross-node traffic to gradient sync.

!!! key "Key Takeaways"

    - **Data parallelism** replicates the model and shards the batch; correctness rests on the linearity of gradients (mean of local gradients = full-batch gradient) plus two invariants: identical init and identical post-sync gradients (an all-reduce/mean).
    - **DDP** makes DP fast via **gradient bucketing** (coalesce many tiny tensors into ~25 MB buffers to amortize latency) and **autograd-hook-driven overlap** (all-reduce a bucket as soon as it fills, while the backward pass keeps computing). Use `no_sync()` for gradient accumulation.
    - The memory problem: mixed-precision Adam costs **~16 bytes/param** (2 bf16 params + 2 bf16 grads + 4 fp32 master + 4+4 Adam moments). Plain DDP replicates all 16 on every GPU; 12 of them are idle except at `optimizer.step()`.
    - **ZeRO** shards the redundant state across the DP group: **Stage 1** shards optimizer states ($16 \to 4 + 12/N$ B/param), **Stage 2** also shards gradients ($2 + 14/N$), **Stage 3** also shards parameters ($16/N$). Stages 1–2 cost the *same* communication as DDP; stage 3 adds an all-gather (~1.5× comm).
    - **FSDP** is PyTorch-native ZeRO-3: it shards a per-unit **FlatParameter**, all-gathers each unit's params just-in-time for forward/backward, frees them after, and reduce-scatters gradients to their owner. `FULL_SHARD`=ZeRO-3, `SHARD_GRAD_OP`=ZeRO-2, `NO_SHARD`=DDP, `HYBRID_SHARD`=shard-in-node/replicate-across.
    - ZeRO/FSDP shard *storage*, not *compute* — they are still data parallelism, orthogonal to (and combinable with) tensor and pipeline parallelism. They do **not** reduce activation memory; pair them with activation checkpointing.
    - **Choose the least sharding that fits.** Each rung from DDP to ZeRO-3 trades communication for memory; on slow inter-node networks, `HYBRID_SHARD` keeps the heavy all-gathers on fast intra-node links.
    - Sharding changes auxiliary operations: use FSDP-aware **gradient clipping** (global norm across shards) and **sharded checkpoints** (each rank writes its slice) rather than their single-GPU equivalents.

!!! sota "State of the Art & Resources (2026)"
    Data parallelism with sharded optimizer state (ZeRO/FSDP) is now the standard baseline for any LLM pretraining run; virtually every frontier model is trained with some combination of ZeRO-3 / FSDP2, tensor parallelism, and pipeline parallelism. The frontier in 2025–2026 is compiler-driven overlap (SimpleFSDP, torch.compile) and tight float8 + FSDP2 co-design pushing throughput 50%+ beyond baseline bf16 FSDP1.

    **Foundational work**

    - [Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* (2020)](https://arxiv.org/abs/1910.02054) — introduces the three-stage sharding of optimizer state, gradients, and parameters that underpins all modern large-model training.
    - [Li et al., *PyTorch Distributed: Experiences on Accelerating Data Parallel Training* (2020)](https://arxiv.org/abs/2006.15704) — the design paper for PyTorch DDP: gradient bucketing, autograd hooks, and overlap that this chapter reconstructs from scratch.
    - [Zhao et al., *PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel* (2023)](https://arxiv.org/abs/2304.11277) — the FSDP design paper covering FlatParameter sharding, prefetching, hybrid sharding, and mixed precision.

    **Recent advances (2023–2026)**

    - [Rajbhandari et al., *ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning* (2021)](https://arxiv.org/abs/2104.07857) — heterogeneous CPU/NVMe offload extending ZeRO to models too large for aggregate GPU memory.
    - [Liang et al., *TorchTitan: One-stop PyTorch native solution for production ready LLM pre-training* (2024)](https://arxiv.org/abs/2410.06511) — production reference showing FSDP2 + 3D parallelism + DTensor on Llama 3.1 up to 405B; accepted ICLR 2025.
    - [Zhang et al., *SimpleFSDP: Simpler Fully Sharded Data Parallel with torch.compile* (2024)](https://arxiv.org/abs/2411.00284) — compiler-based FSDP that traces the full compute-communication graph, enabling IR-level bucketing and up to 68% throughput improvement over FSDP2 eager.

    **Open-source & tools**

    - [deepspeedai/DeepSpeed](https://github.com/deepspeedai/DeepSpeed) — the production reference implementation of ZeRO stages 1–3, ZeRO-Infinity (NVMe offload), and ZeRO++ communication compression.
    - [pytorch/torchtitan](https://github.com/pytorch/torchtitan) — PyTorch-native training platform using FSDP2, DTensor, and torch.compile; the canonical example of composing all parallelism axes cleanly.

    **Go deeper**

    - [PyTorch: *Supercharging Training using float8 and FSDP2* (2024)](https://pytorch.org/blog/training-using-float8-fsdp2/) — official blog post showing float8 all-gathers + FSDP2 + torch.compile delivering ~50% throughput gains on 70B–405B Llama models.
    - [PyTorch Tutorials: *What is Distributed Data Parallel (DDP)*](https://docs.pytorch.org/tutorials/beginner/ddp_series_theory.html) — official beginner tutorial series covering DDP mechanics, multi-GPU, and multi-node setups with working code.

## Further Reading

- **Rajbhandari, Rajbhandari, Ruwase, He (2020):** "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" — the foundational paper introducing the three-stage sharding of optimizer states, gradients, and parameters, with the per-parameter memory accounting this chapter builds on.
- **Ren et al. (2021):** "ZeRO-Offload: Democratizing Billion-Scale Model Training" and **Rajbhandari et al. (2021):** "ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning" — CPU/NVMe offload extensions.
- **Li et al. (2020):** "PyTorch Distributed: Experiences on Accelerating Data Parallel Training" — the design of DDP, including bucketing, gradient hooks, and the overlap strategy reconstructed in this chapter.
- **Zhao et al. (2023):** "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel" — the FSDP design paper covering FlatParameter sharding, prefetching, hybrid sharding, and mixed precision.
- **PyTorch documentation:** `torch.distributed.fsdp` and the FSDP2 / `DTensor` / `DeviceMesh` tutorials — the authoritative, version-current reference for the APIs used in this chapter.
- **DeepSpeed documentation and repository** (github.com/microsoft/DeepSpeed) — the production reference implementation of ZeRO stages 1–3, offload, and their configuration.
