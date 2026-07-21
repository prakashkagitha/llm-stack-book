# 6.7 Colocated vs Disaggregated RL & Weight Synchronization

Every reinforcement-learning-for-LLM system is, at heart, a tug-of-war between two very different programs fighting over the same scarce resource: GPUs. On one side is the **trainer** — a backward-pass-heavy, memory-hungry process that holds the policy weights, the optimizer state, gradients, and activations, and runs at full tensor-core utilization in short, dense bursts. On the other side is the **rollout engine** (also called the *generator* or *actor*) — an autoregressive decoder that is memory-bandwidth bound, runs for a long time per step, and spends most of its life waiting on the KV cache. These two workloads have nearly opposite hardware appetites, and how you place them on your cluster is the single most consequential infrastructure decision in an RL run.

This chapter is about that decision. We cover the two dominant placement strategies — **colocated** (trainer and rollout share GPUs) and **disaggregated** (they live in separate GPU pools) — and the family of tricks in between (time-slicing, offloading). Then we get to the hard part that ties both worlds together: **weight synchronization**. After every (or every few) optimizer steps, the freshly updated policy weights must reach the inference engine, which is laid out in a completely different parallelism scheme. We dissect the three transport mechanisms (NCCL broadcast, CUDA IPC, checkpoint reload), the **resharding problem** between training and inference layouts, and the latency/utilization math that decides which design wins.

This is the chapter where the [anatomy of an RL system](../06-rl-infra/01-anatomy-rl-system.html) and the [generation–training loop](../06-rl-infra/02-generation-training-loop.html) become a concrete engineering problem. We assume you know what GRPO and PPO are ([GRPO, RLOO & Critic-Free RL](../05-posttraining-alignment/08-grpo-rloo.html), [Policy Gradients & PPO for Language Models](../05-posttraining-alignment/06-ppo-for-llms.html)); here we care only about *where the bytes live and how they move*.

## The Two Workloads and Why Placement Matters

Start with a single RL iteration. Ignoring the reward step, the loop is:

{{fig:colodis-rl-iteration-loop}}

The rollout and the trainer touch *the same model*, but they want it in different forms. Let us be precise about the differences, because the entire chapter follows from them.

**Memory profile.** The trainer holds, for a model with $P$ parameters trained in bf16 with the Adam optimizer:

$$
M_\text{train} = \underbrace{2P}_{\text{weights}} + \underbrace{2P}_{\text{grads}} + \underbrace{12P}_{\text{Adam m, v, fp32 master}} + M_\text{act}
$$

That well-known $\approx 16P$ bytes (plus activations) is why training a 7B model needs on the order of 112 GB of optimizer-related state. The rollout engine, by contrast, holds only weights and a KV cache:

$$
M_\text{infer} = \underbrace{2P}_{\text{weights}} + M_\text{KV}
$$

A bf16 7B policy is 14 GB of weights; the rest of an 80 GB GPU is free for KV cache, which is exactly what an inference engine like [vLLM](../07-inference-serving/03-vllm-internals.html) or [SGLang](../07-inference-serving/04-sglang-radixattention.html) wants — more KV cache means larger rollout batches and higher decode throughput.

**Compute profile.** Training is *compute bound*: a forward+backward pass is $\approx 6P$ FLOPs per token and saturates the tensor cores. Decoding is *memory-bandwidth bound*: each generated token reads the entire weight matrix from HBM but does only $\approx 2P$ FLOPs, so the GPU's arithmetic units idle while the memory bus is the bottleneck. (This is the prefill-vs-decode distinction from [The Anatomy of LLM Inference](../07-inference-serving/01-anatomy-inference.html), and the roofline picture from [The Roofline Model & Performance Engineering](../04-kernels-efficiency/01-roofline-performance.html).)

**Temporal profile.** Generation is *long and bursty per sample but parallelizable across samples*; a long chain-of-thought rollout can take tens of seconds. Training is *short and dense*: once the rollouts and advantages are ready, a handful of minibatch gradient steps finish in seconds.

These three mismatches are the crux. If you put both workloads on the same GPU at the same time, they contend for HBM and you must shrink both. If you put them on separate GPUs, one pool sits idle while the other works (unless you pipeline). The art of RL infrastructure is hiding that idle time.

!!! note "Aside: parallelism layouts differ too"

    It is not just memory and compute that differ — the *parallelism strategy* differs. A trainer typically uses FSDP/ZeRO ([Distributed Training I](../03-pretraining/05-distributed-data-parallel.html)) or Megatron-style tensor + pipeline parallelism ([Distributed Training II](../03-pretraining/06-distributed-model-parallel.html)) tuned for backward throughput. The inference engine picks a *different* tensor-parallel degree tuned for decode latency and KV-cache headroom. The mismatch between these two layouts is the **resharding problem** of [§5](#the-resharding-problem-training-vs-inference-layouts), and it is the reason weight sync is hard rather than a simple memcpy.

## Colocated: Sharing GPUs Between Training and Rollout

{{fig:colocated-disagg}}

In a **colocated** design, the same physical GPUs run both the trainer and the rollout engine. This is the default in [TRL](../06-rl-infra/03-trl.html), the classic configuration in [veRL](../06-rl-infra/04-verl.html)'s "hybrid engine," and the simplest thing to reason about. The fundamental constraint is that an 80 GB GPU cannot simultaneously hold $16P$ bytes of training state *and* $2P$ bytes of inference weights *and* a useful KV cache. Something has to give, and the three strategies for making it give are **time-slicing**, **offloading**, and **memory partitioning**.

### Time-slicing (the hybrid engine)

The cleanest colocated pattern runs the two phases sequentially on the same GPUs and swaps which one owns memory. During the rollout phase, the inference engine is "live": weights are resident, KV cache is allocated, the trainer's optimizer state and gradients are pushed out of the way (to CPU or freed). During the training phase, the inference engine releases its KV cache and the trainer reclaims the memory for activations and gradients.

{{fig:colodis-memory-timeline}}

veRL calls this the **hybrid engine**: a single set of workers that wears two hats, flipping between a training role (FSDP or Megatron) and a generation role (vLLM/SGLang) on command from the single controller. The key implementation detail is that the inference engine must support *sleep/wake*: it tears down and rebuilds its KV-cache memory pool around the training phase rather than holding it for the whole iteration.

```python
# Sketch of the time-sliced colocated loop (single-controller view, veRL-style).
# Each "worker" actually holds BOTH an FSDP-wrapped trainer and a vLLM engine
# bound to the same CUDA devices. We flip between them every iteration.

for step in range(num_steps):
    # ---- PHASE 1: ROLLOUT ----------------------------------------------------
    # Wake the inference engine: allocate the KV-cache pool on the GPUs.
    rollout_engine.wake_up()                  # cudaMalloc the paged KV blocks
    # Trainer state is offloaded so the KV cache has room (see §offloading).
    trainer.offload_optimizer_to_cpu()        # async D2H copy of Adam m, v, master
    trainer.offload_grads()

    batch = sample_prompts(dataset, n=global_batch)
    # generate() runs continuous batching over all prompts (G samples each for GRPO)
    completions = rollout_engine.generate(batch, sampling_params)

    # ---- PHASE 2: REWARD -----------------------------------------------------
    rewards = reward_fn(batch.prompts, completions)        # may call a sandbox/verifier
    advantages = compute_group_advantages(rewards)         # GRPO: r_i - mean(r_group)

    # ---- PHASE 3: TRAIN ------------------------------------------------------
    rollout_engine.sleep()                    # free the KV-cache pool back to the allocator
    trainer.reload_optimizer_to_gpu()         # bring Adam state back (async H2D)

    for micro in make_microbatches(batch, completions, advantages):
        loss = trainer.compute_loss(micro)    # fwd + bwd under FSDP
        loss.backward()
    trainer.optimizer.step()
    trainer.optimizer.zero_grad()

    # ---- PHASE 4: WEIGHT SYNC ------------------------------------------------
    # Push the just-updated weights into the (sleeping) inference engine's
    # parameter buffers so the NEXT rollout uses the new policy. See §4.
    sync_weights(trainer, rollout_engine)
```

The beauty of time-slicing is **GPU utilization**: every GPU is busy in every phase, because there is only one pool. The cost is **latency** — the phases are strictly serial, so the rollout GPUs are doing inference work while the (same) training capability sits idle, and vice versa. There is no overlap. For a run where rollout dominates wall-clock (long reasoning traces), this means your expensive training-capable cluster spends most of its time doing memory-bound decode at low FLOP utilization.

### Offloading

Time-slicing only works if you can *make room*. The mechanism is offloading: moving tensors that are not needed in the current phase off the GPU. There are two flavors:

1. **Optimizer-state offload (host RAM).** Adam's $m$, $v$, and fp32 master copy are $12P$ bytes — the largest single consumer. During rollout they are dead weight, so we copy them to pinned CPU memory over PCIe/NVLink-C2C and free the GPU tensors. Before the optimizer step we copy them back. This is exactly ZeRO-Offload / FSDP CPU-offload ([Memory-Efficient Training](../04-kernels-efficiency/10-memory-efficient-training.html)), reused here on a per-phase cadence.

2. **Weight offload / KV-cache release.** The inference engine's KV pool (tens of GB) is released between rollouts. Inference *weights* may be kept resident (they are the same bytes the trainer needs) or, in the most aggressive colocation, the inference engine and trainer literally alias the *same* weight tensors — there is one copy of the parameters and a "view" of them for each role.

The offload transfer is not free. A 7B model's Adam state is $12 \times 7\text{e}9 = 84$ GB. Over a 64 GB/s PCIe 4.0 x16 link that is $84 / 64 \approx 1.3$ s each way, or 2.6 s of pure copy per iteration. Over NVLink-C2C (Grace-Hopper, hundreds of GB/s) it is a fraction of that. This copy time is the hidden tax of colocated time-slicing, and it is why fast host interconnects (NVLink-C2C, large pinned buffers, double-buffered async copies) matter so much.

!!! tip "Practitioner tip"

    Always use **pinned (page-locked) host memory** for offload buffers and issue the copies on a dedicated CUDA stream so they overlap with the tail of the previous phase. A naïve `tensor.cpu()` allocates pageable memory and serializes the copy, turning a 0.5 s overlap-able transfer into a 2 s stall. In PyTorch: pre-allocate with `torch.empty(..., pin_memory=True)` and use `non_blocking=True` on `.to()`.

### Memory partitioning (true simultaneous colocation)

The third option is to *not* time-slice at all: carve the GPU's HBM into a training partition and an inference partition that coexist. This is rare in pure RL because the combined footprint rarely fits, but it appears in two guises: (a) very small models or heavily LoRA-fied policies where $16P$ is tiny, and (b) **multiplexing via MPS** (NVIDIA Multi-Process Service) or MIG, where two processes share SMs and memory under a hardware scheduler. The win is true overlap; the risk is that the two contend for HBM bandwidth and *both* slow down. In practice, time-slicing + offload dominates for full-parameter RL; partitioning shows up mainly with LoRA-based RL where the policy delta is small.

## Disaggregated: Separate Pools for Training and Rollout

The **disaggregated** design gives up on sharing. You provision two distinct GPU pools: a *training cluster* and an *inference (rollout) cluster*, each with its own optimal parallelism layout, connected by a network and a weight-sync channel. This is the design of [OpenRLHF](../06-rl-infra/05-openrlhf-nemo-ray.html), the async architecture of [Prime-RL](../06-rl-infra/06-prime-rl-async.html), and the throughput-optimized mode of large-scale veRL deployments. It mirrors the [disaggregated prefill/decode](../07-inference-serving/08-disaggregated-chunked-prefill.html) idea from serving — split a workload whose phases have different hardware profiles onto specialized hardware.

{{fig:colodis-disagg-topology}}

The two pools can be sized independently. If a run is rollout-bound (long generations, small policy updates), you give the rollout pool more GPUs — say 16 inference GPUs feeding 8 training GPUs. If it is training-bound (huge model, short completions), you flip the ratio. This **independent scaling** is the headline advantage of disaggregation and is impossible under strict colocation, where the ratio is forced to 1:1.

The second advantage is **pipelining**. Because the pools are separate, the rollout cluster can be generating the *next* batch of experience while the training cluster is still doing gradient steps on the *current* batch. This overlap is exactly what colocated time-slicing cannot do. But it comes with a subtle cost: the rollouts generated by the inference pool were produced by a policy that is now *one step stale* relative to the trainer. We have crossed from on-policy into **off-policy** territory.

### On-policy vs off-policy: the staleness knob

If the trainer waits for the rollout pool to finish before each update, and the rollout pool always uses the latest weights, the system is **synchronous** and **on-policy** — exactly equivalent to colocated time-slicing, just on separate hardware (and with one pool idle while the other works). To get overlap you must let the rollout pool run ahead, producing experience under policy $\pi_{\theta_{t-k}}$ while the trainer is at $\theta_t$. The lag $k$ is the **staleness**.

{{fig:colodis-sync-vs-async-timeline}}

Staleness is a knob, not a free lunch. A small lag ($k=1$) is usually harmless and is corrected by the importance-sampling ratio already present in PPO/GRPO ($\rho = \pi_\theta / \pi_{\theta_\text{old}}$); the off-policy data is reweighted toward the current policy. Large lag degrades the gradient signal and can destabilize training. Managing this — importance correction, clipping, partial rollouts, and how much asynchrony is safe — is the subject of [Prime-RL, Async RL & Decentralized Training](../06-rl-infra/06-prime-rl-async.html) and the stability tricks in [Advantage Estimation, KL Control & Stability Tricks](../06-rl-infra/09-advantage-kl-tricks.html). For our purposes the point is structural: **disaggregation buys overlap at the price of off-policyness**, and the weight-sync frequency is what controls the staleness.

### Cost of disaggregation

The disadvantages are real. First, **idle GPUs under perfect balance are impossible** — there is always slack on one side, and you pay for both pools whether or not they are saturated. Second, you now ship two large things across the network every step: the **experience** (rollouts) from inference to training, and the **weights** from training to inference. The experience is cheap (token IDs and logprobs, megabytes). The weights are not (gigabytes), and getting them across efficiently is the whole next section. Third, the system is operationally more complex: two fleets, a queue/orchestrator (usually Ray, see [OpenRLHF, NeMo-Aligner & Ray-Based Systems](../06-rl-infra/05-openrlhf-nemo-ray.html)), and a weight-sync protocol that must survive the resharding mismatch.

## Weight Synchronization Mechanisms

After the optimizer step, the trainer holds the new policy $\theta_{t+1}$. The rollout engine still holds $\theta_t$. We must move the new weights into the inference engine's parameter buffers before (or for async, soon after) the next rollout. There are three transport mechanisms, in increasing order of speed and decreasing order of robustness.

{{fig:colodis-weight-sync-three-paths}}

### Mechanism 1: Checkpoint reload (slow, simple, robust)

The trainer writes a full checkpoint to shared storage; the inference engine reloads it. This is the "obviously correct" baseline and the only mechanism that trivially handles *any* layout mismatch, because the checkpoint is a layout-independent `state_dict` and the inference engine loads it with its own sharding logic — the same path it uses at startup.

```python
# Mechanism 1: checkpoint reload. Correct, slow, layout-agnostic.
def sync_via_checkpoint(trainer, rollout_engines, path):
    # 1. Gather the FULL (unsharded) fp32/bf16 state dict on rank 0.
    #    Under FSDP this triggers an all-gather of every shard.
    full_state = trainer.get_full_state_dict()          # heavy: materializes 2P bytes
    if trainer.is_rank0:
        torch.save(full_state, path)                    # write 2P bytes to disk/NFS
    barrier()
    # 2. Every inference engine reloads from disk and re-shards to its own TP layout.
    for engine in rollout_engines:
        engine.load_weights(path)                       # read 2P bytes, reshard
```

For a 7B model this is 14 GB written and re-read; on NFS that can be tens of seconds. For a 70B model (140 GB) it is minutes. Checkpoint reload is fine for *occasional* sync (e.g., DPO-style offline data refresh, or recovery) but far too slow to run every RL step. Its one irreplaceable virtue: it is the fallback that always works, including across nodes that share nothing but a filesystem, and it doubles as your [fault-tolerance checkpoint](../03-pretraining/12-checkpointing-fault-tolerance.html).

### Mechanism 2: NCCL broadcast (fast, in-band, the workhorse)

The standard fast path is to keep the weights on the GPU the entire time and move them GPU-to-GPU over NVLink/InfiniBand using **NCCL collectives** ([Parallel Computing & Collective Communication](../01-foundations/09-parallel-collectives.html)). The trainer ranks that own each parameter shard `broadcast` it directly into the inference ranks' parameter tensors, never touching disk or host memory.

The trick is that the trainer and inference processes are *different process groups*. To make NCCL move a tensor from one to the other you build a **shared communicator** that spans both — a process group whose members are (training rank 0 …) ∪ (inference rank 0 …). NCCL needs a rendezvous: rank 0 creates a unique ID, distributes it (over Ray, a file, or a socket), and every participant calls `init_process_group` / `ncclCommInitRank` with it.

```python
# Mechanism 2: NCCL broadcast over a process group that SPANS trainer + inference.
# Run on every rank in the combined group. We assume the param NAMES match and
# we send one parameter at a time, broadcasting from the trainer rank that owns it.

import torch
import torch.distributed as dist

class WeightSyncGroup:
    def __init__(self, rank, world_size, master_addr, master_port):
        # ONE communicator that includes BOTH the training ranks and the
        # inference ranks. The trainer is the "source" of every broadcast.
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            rank=rank,
            world_size=world_size,   # = num_train_ranks + num_infer_ranks
        )
        self.src_rank = 0            # trainer rank 0 is the broadcast root

    def push(self, named_params):
        # Called on the TRAINER. Broadcasts each weight to all inference ranks.
        for name, tensor in named_params:
            # tensor must be a full (gathered) parameter in the layout the
            # inference engine expects. Resharding (§5) happens BEFORE this call.
            dist.broadcast(tensor, src=self.src_rank)

    def pull(self, param_buffers):
        # Called on each INFERENCE rank. Receives into its parameter buffers.
        for name, buf in param_buffers.items():
            dist.broadcast(buf, src=self.src_rank)     # buf filled in place
```

In real systems (vLLM's `RLHFWorker` / collective-RPC interface, SGLang's weight-update API, OpenRLHF's `vllm_engine.update_weight`) this is wrapped so the trainer calls one RPC and the inference workers receive into their internal parameter storage. The transfer runs at NVLink/IB bandwidth: 14 GB of bf16 weights over a 900 GB/s NVLink fabric is $14 / 900 \approx 16$ ms — three orders of magnitude faster than NFS reload. This is why NCCL broadcast is the default in-band sync for every serious RL framework.

Two practical wrinkles. First, **bf16 vs fp32**: broadcast in the dtype the inference engine consumes (usually bf16), halving the bytes versus fp32 master weights. Second, **bucketing**: broadcasting thousands of tiny tensors one at a time is latency-bound by kernel-launch and handshake overhead; frameworks flatten parameters into large contiguous buckets and broadcast a few big buffers instead, which is bandwidth-bound and far faster.

### Mechanism 3: CUDA IPC (zero-copy, same-node, fastest)

When the trainer and inference engine are *different processes on the same node* sharing the same GPUs (the colocated case), you can avoid copying the weights at all. **CUDA IPC** (Inter-Process Communication) lets one process export a handle to a GPU allocation and another process map that exact device memory into its own address space. No bytes move over any link — the inference engine simply reads from the trainer's weight tensors.

```python
# Mechanism 3: CUDA IPC handle export. The trainer process exports a handle to
# its weight tensor's device memory; the inference process imports it and gets a
# tensor backed by the SAME physical GPU memory. Zero copy.

# ---- TRAINER PROCESS ----
import torch
from torch.multiprocessing.reductions import reduce_tensor

def export_handles(named_params):
    handles = {}
    for name, tensor in named_params:
        # reduce_tensor produces a picklable IPC handle (device ptr + metadata).
        # Send `handles` to the inference process over a pipe / Ray object store.
        handles[name] = reduce_tensor(tensor.detach())
    return handles

# ---- INFERENCE PROCESS ----
def import_handles(handles):
    mapped = {}
    for name, handle in handles.items():
        rebuild_fn, args = handle
        # Rebuilds a torch.Tensor whose storage points at the trainer's memory.
        mapped[name] = rebuild_fn(*args)     # NO copy; aliases trainer's weights
    return mapped
```

In the most tightly colocated designs the inference engine does not even re-import after every step — the parameter tensors are *aliased once* at startup, and because the trainer writes its updates in place (`optimizer.step()` mutates the same `param.data`), the inference engine automatically "sees" the new weights with literally zero sync work. The catch is correctness: you must ensure the trainer's optimizer step has fully completed (CUDA stream synchronized) before the inference engine reads, and you must guarantee the two are not writing/reading the buffer concurrently. CUDA IPC is the fastest possible sync — effectively free — but only available on a single node and requiring that both processes can map the same memory.

The decision tree:

| Situation | Mechanism | Order-of-magnitude cost (7B) |
|---|---|---|
| Same node, shared GPUs, aliased weights | CUDA IPC | ~0 ms (no copy) |
| Same/multi node, GPUs reachable by NCCL | NCCL broadcast | ~10–50 ms |
| Cross-cluster, no NCCL path, or recovery | Checkpoint reload | seconds–minutes |

!!! warning "Common pitfall: name and shape mismatch"

    NCCL broadcast and IPC both assume the trainer's parameter you send and the inference buffer you receive into have the **same name, shape, and dtype**. They very often do not, because the inference engine fuses QKV projections, fuses gate+up in the MLP, transposes for its kernels, or uses a different tensor-parallel split. If you broadcast a 4096×4096 `q_proj` into an engine that expects a fused 12288×4096 `qkv_proj`, you get silent garbage (or a crash). A correct sync layer maintains an explicit **name+shape mapping** and reshapes/concatenates on the trainer side *before* the broadcast. This is the resharding problem, next.

## The Resharding Problem: Training vs Inference Layouts

Here is the part that makes weight sync genuinely hard rather than a memcpy. The trainer and the inference engine almost never store the parameters the same way. They differ along three axes, and the sync layer must reconcile all three.

**Axis 1 — Sharding (parallelism layout).** The trainer might use FSDP, which flattens every parameter and shards it into equal slices across all data-parallel ranks; the byte at offset $i$ on rank $r$ has no semantic meaning on its own. The inference engine uses **tensor parallelism**: it splits attention heads and MLP columns across its TP ranks in a *structured*, head-aligned way. These two shardings are incompatible. Even if both used tensor parallelism, the *degree* differs: the trainer might run TP=8 (to fit optimizer state) while inference runs TP=2 (to minimize decode latency and maximize KV cache). Going from TP=8 to TP=2 means each inference rank must *gather* the slices of four training ranks.

**Axis 2 — Fusion.** Inference engines fuse operations for kernel efficiency. The three separate projections $W_Q, W_K, W_V$ that the trainer stores are concatenated into one `qkv_proj` tensor in vLLM; `gate_proj` and `up_proj` become one `gate_up_proj`. Sync must concatenate (or split) accordingly. With GQA/MQA ([Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html)) the concatenation is not even uniform — K and V have fewer heads than Q, so the fused tensor is laid out in a specific interleaved order that the sync code must replicate exactly.

**Axis 3 — Dtype and quantization.** The trainer holds bf16 (and fp32 master); the inference engine might run **quantized** weights — FP8, INT8, or INT4 ([Quantization II](../04-kernels-efficiency/08-quantization-formats-qat.html)). Sync then includes an on-the-fly quantization step: take the bf16 master, compute scales, pack to INT4, and write into the engine's quantized buffers. This is expensive and changes the sync from "copy bytes" to "recompute the quantized representation," which is one reason RL on quantized inference engines is fiddly.


{{fig:colodis-reshard-tp8-to-tp2}}


The general recipe a correct resharding sync layer follows:

```python
# Resharding weight sync: from a sharded trainer layout to a fused, re-sharded
# inference layout. Conceptual; real impls (veRL, OpenRLHF) are more optimized.

def reshard_and_sync(trainer, infer_engines, name_map, infer_tp):
    """
    name_map: tells us how trainer param names map to inference param names,
              including which trainer tensors fuse into one inference tensor.
    infer_tp: tensor-parallel degree of the inference engine.
    """
    for infer_name, spec in name_map.items():
        # 1) GATHER: materialize the FULL (unsharded) trainer tensor(s).
        #    Under FSDP this all-gathers the flat shards and reshapes to 2D.
        full_tensors = [trainer.gather_full_param(tn) for tn in spec.trainer_names]

        # 2) FUSE: concatenate Q/K/V (or gate/up) into the engine's fused tensor,
        #    honoring GQA head grouping if K/V have fewer heads than Q.
        fused = fuse_along(full_tensors, dim=spec.fuse_dim, head_layout=spec.heads)

        # 3) (optional) QUANTIZE: if the engine runs FP8/INT4, compute scales & pack.
        if spec.quantized:
            fused = quantize(fused, scheme=spec.quant_scheme)

        # 4) RE-SHARD: split the fused tensor into `infer_tp` slices, one per
        #    inference TP rank, along the engine's tensor-parallel dimension.
        shards = split_for_tp(fused, tp=infer_tp, dim=spec.tp_dim)

        # 5) SEND: broadcast/scatter each shard to its inference rank (Mechanism 2),
        #    or copy in place if same-node aliasing (Mechanism 3).
        scatter_to_infer_ranks(shards, infer_name, infer_engines)
```

Steps 1 and 2 are the expensive ones: gathering the full parameter undoes the trainer's sharding (an all-gather of $2P$ bytes), and you need transient memory to hold the full fused tensor. Good implementations do this **layer by layer**, streaming one fused tensor at a time so peak extra memory is one layer's worth, not the whole model. veRL's resharding manager and OpenRLHF's `update_weight` path both implement a version of this name-mapped, layer-streamed gather-fuse-reshard-broadcast pipeline. The single-controller architecture of [veRL](../06-rl-infra/04-verl.html) is in part *designed* to make this resharding step expressible as one orchestrated collective rather than ad-hoc message passing.

!!! note "Why not just make the layouts identical?"

    You can, sometimes: run the inference engine at the same TP degree as the trainer and disable fusion, so sync is a plain shard-to-shard broadcast with no gather. This is the fastest sync but the *slowest* everything-else — you have handcuffed your inference engine to the trainer's parallelism, losing the decode-latency and KV-cache tuning that motivated disaggregation. The resharding cost is the price of letting each engine pick its optimal layout. The right call depends on whether sync time or rollout time dominates your iteration.

## Latency, Utilization, and a Worked Example

We now have all the pieces to reason quantitatively. Define one RL iteration's wall-clock time. For a **synchronous** system (colocated time-sliced, or synchronous disaggregated):

$$
T_\text{iter} = T_\text{rollout} + T_\text{reward} + T_\text{train} + T_\text{sync} + T_\text{offload}
$$

For an **asynchronous** disaggregated system with perfect pipelining, rollout overlaps training, so the iteration is bounded by the slower stage:

$$
T_\text{iter}^\text{async} \approx \max\!\left(T_\text{rollout},\; T_\text{train}\right) + T_\text{sync}
$$

The **GPU utilization** of the colocated synchronous design, measured as useful-work fraction on the training-capable GPUs, is roughly the share of time spent in compute-bound work:

$$
U_\text{colo} = \frac{T_\text{train}}{T_\text{rollout} + T_\text{reward} + T_\text{train} + T_\text{sync} + T_\text{offload}}
$$

When rollout dominates (long reasoning traces), $U_\text{colo}$ is small — your training cluster spends most of its life doing memory-bound decode. That is the central argument for disaggregation: move the long, cheap rollout onto cheaper/more-numerous inference GPUs and keep the expensive training GPUs busy with training.

!!! example "Worked example: colocated vs disaggregated for a 7B reasoning run"

    Setup: a 7B policy, GRPO with $G=8$ samples per prompt, average completion length 4000 tokens (long chain-of-thought). We have 16 H100-80GB GPUs and an NVLink fabric at ~900 GB/s intra-node. Per RL iteration we process a global batch of 256 prompts (= 2048 rollout sequences). Suppose the measured per-iteration stage times are:

    | Stage | Time | Notes |
    |---|---|---|
    | $T_\text{rollout}$ | 40 s | decode 2048 × 4000 tokens, memory-bound |
    | $T_\text{reward}$ | 3 s | math/code verifiers, partly overlappable |
    | $T_\text{train}$ | 8 s | a few GRPO minibatch fwd+bwd steps |
    | $T_\text{offload}$ | 3 s | Adam state out+in over PCIe per iter |
    | $T_\text{sync}$ | 0.02 s | NCCL broadcast of 14 GB at 900 GB/s |

    **Colocated time-sliced (all 16 GPUs do everything, serially):**

    $$T_\text{iter}^\text{colo} = 40 + 3 + 8 + 3 + 0.02 \approx 54\ \text{s}$$

    Training-GPU utilization:

    $$U_\text{colo} = \frac{8}{54} \approx 15\%$$

    For 39 of every 54 seconds, 16 training-capable H100s are doing memory-bound decode — a poor use of tensor cores.

    **Disaggregated async (split 16 GPUs as 4 train + 12 rollout):** With 12 GPUs on rollout instead of 16, raw rollout throughput drops to $16/12$ of before, so $T_\text{rollout}' \approx 40 \times 16/12 \approx 53$ s for the same batch. But rollout now overlaps training, and we can let the rollout pool stay full. With 4 training GPUs, $T_\text{train}' \approx 8 \times 16/4 \approx 32$ s (fewer GPUs, more time) — but training overlaps rollout. The iteration is bounded by the slower stage plus sync:

    $$T_\text{iter}^\text{async} \approx \max(53,\ 32) + 0.02 \approx 53\ \text{s}$$

    That looks no better on wall-clock — and indeed for *this* split it is not, because we starved training. The real win is **rebalancing**: because rollout dominates, push more GPUs to rollout *and* size training so $T_\text{train}' \le T_\text{rollout}'$. With, say, 6 train + 10 rollout: $T_\text{rollout}' \approx 40 \times 16/10 = 64$ s, $T_\text{train}' \approx 8 \times 16/6 \approx 21$ s, iteration $\approx 64$ s — still rollout-bound, training fully hidden. The lesson: disaggregation's value is not automatic speedup, it is the *freedom to size the two pools so the cheap, dominant stage is the only thing on the critical path*, while the training GPUs run at near-100% utilization instead of 15%. On a real cost model where rollout can run on cheaper inference GPUs, that reallocation is a large effective-cost win even when wall-clock is similar.

The example also exposes the staleness cost we hid: in the async case, the rollouts feeding each update were generated under weights one iteration old. If that staleness hurts sample efficiency by, say, 10%, you must weigh it against the utilization gain. This is the perennial RL-infra trade and why the choice is workload-dependent rather than universal.

!!! interview "Interview Corner"

    **Q:** You are designing an RL system to post-train a 32B model with very long (8k-token) reasoning rollouts. You have a fixed budget of 64 H100s. Walk me through whether you colocate or disaggregate, how you sync weights, and what the resharding problem forces you to handle.

    **A:** With 8k-token rollouts, the run is heavily rollout-bound — decode will dominate wall-clock — so I disaggregate. I split the 64 GPUs into a small training pool and a large rollout pool, e.g. 16 train + 48 rollout, sized so training time hides under rollout time. The trainer runs FSDP or Megatron TP+PP to fit optimizer state for 32B (~512 GB of Adam state alone); the rollout pool runs vLLM/SGLang at a low TP degree (TP=2 or 4) to maximize KV-cache headroom for the long sequences. I run **asynchronous** with staleness 1, relying on the PPO/GRPO importance ratio plus clipping to correct the off-policy lag.

    For weight sync I use **NCCL broadcast** over a process group spanning both pools — 64 GB of bf16 weights at NVLink/IB bandwidth is well under a second, so I can afford to sync every step. The **resharding problem** forces me to (1) gather the FSDP-sharded params into full tensors layer by layer, (2) fuse Q/K/V and gate/up into the engine's fused layout, honoring GQA head grouping, and (3) re-split from the trainer's TP degree to the inference TP degree before broadcasting. I keep an explicit name+shape map between the two layouts and stream one layer at a time to cap transient memory. If the inference engine runs FP8, sync also re-quantizes from the bf16 master on each push. I'd checkpoint to disk periodically as the robust fallback and for fault tolerance, but never use checkpoint-reload on the hot sync path because it is orders of magnitude too slow.

## Putting It Together: A Decision Framework

There is no universally best design; there is a set of questions whose answers pick one for you.

**Is the run rollout-bound or training-bound?** Profile $T_\text{rollout}$ vs $T_\text{train}$. Long generations (reasoning, agentic multi-turn rollouts, see [Agentic & Multi-Turn RL](../06-rl-infra/10-agentic-multiturn-rl.html)) are rollout-bound and favor disaggregation, where the cheap rollout pool can be scaled independently and the training GPUs stay saturated. Short completions with a huge model are training-bound and tolerate colocation, since rollout is a small slice of the iteration.

**How much hardware do you have?** On a single node (one machine, ≤8 GPUs), colocated time-slicing with CUDA-IPC or aliased weights is simplest and the sync cost is essentially zero. Disaggregation makes sense at multi-node scale where you can afford to dedicate fleets and the network supports fast NCCL sync.

**How tolerant is your algorithm of staleness?** Strictly on-policy methods, or runs where you have seen instability from off-policy data, push you toward synchronous designs (colocated, or synchronous disaggregated with the rollout pool idling during updates). If your importance correction and KL control ([Advantage Estimation, KL Control & Stability Tricks](../06-rl-infra/09-advantage-kl-tricks.html)) are solid, async disaggregation unlocks far better utilization.

**What is your weight-sync budget?** If sync is cheap (same-node IPC, or fast NVLink/IB NCCL), you can sync every step and keep staleness at 1. If sync is expensive (cross-cluster, must reload checkpoints), you sync rarely, which *forces* higher staleness — the sync mechanism and the algorithm's off-policyness are coupled.


{{fig:colodis-decision-flow}}


The frameworks map onto this flow. [TRL](../06-rl-infra/03-trl.html) is colocated by default (with an optional vLLM server that nudges toward disaggregation). [veRL](../06-rl-infra/04-verl.html) supports both: the hybrid engine for colocation and a disaggregated mode, with a single controller orchestrating the resharding sync. [OpenRLHF](../06-rl-infra/05-openrlhf-nemo-ray.html) is Ray-native and leans disaggregated, with vLLM rollout workers receiving NCCL-broadcast weights. [Prime-RL](../06-rl-infra/06-prime-rl-async.html) takes async disaggregation to its conclusion with deliberately decoupled, possibly geographically separated, trainer and rollout fleets. The scaling tricks that make the rollout pool itself efficient — load balancing across uneven-length generations, in-flight weight updates — are the subject of [Scaling RL: Throughput, Load Balancing & The Latest Tricks](../06-rl-infra/11-scaling-rl-tricks.html).

!!! key "Key Takeaways"

    - The trainer (compute-bound, $\approx 16P$ bytes of state, short bursts) and the rollout engine (memory-bandwidth-bound, $\approx 2P$ bytes + KV cache, long bursts) have opposite hardware appetites; how you place them is the central RL-infra decision.
    - **Colocated** shares GPUs via time-slicing and offloading (the "hybrid engine"): maximal per-GPU utilization within a phase, but strictly serial phases mean training-capable GPUs spend most of a rollout-bound run doing memory-bound decode at low FLOP utilization.
    - **Disaggregated** uses separate training and rollout pools that can be sized independently and pipelined to overlap; this buys utilization and throughput at the price of **off-policy staleness** (rollouts are produced under slightly stale weights).
    - Weight sync has three mechanisms: **CUDA IPC** (zero-copy, same node, fastest), **NCCL broadcast** (GPU-to-GPU over NVLink/IB, the everyday workhorse at ~10s of ms for a 7B model), and **checkpoint reload** (seconds–minutes, layout-agnostic, the robust fallback and fault-tolerance path).
    - The **resharding problem** is what makes sync hard: trainer and inference layouts differ in sharding (FSDP vs TP, and TP degree), fusion (separate vs fused QKV / gate-up, with GQA head grouping), and dtype (bf16 master vs FP8/INT4 inference). A correct sync layer gathers, fuses, optionally quantizes, re-shards, and broadcasts, streamed layer by layer.
    - Sync mechanism and algorithm staleness are coupled: cheap sync lets you update the rollout policy every step (staleness 1, corrected by the PPO/GRPO importance ratio); expensive sync forces higher staleness.
    - The right design is workload-dependent: rollout-bound + multi-node + staleness-tolerant ⇒ async disaggregation; small/single-node or strictly on-policy ⇒ colocated time-slicing. The win from disaggregation is not automatic wall-clock speedup but the freedom to size pools so the dominant stage is the only thing on the critical path.

!!! sota "State of the Art & Resources (2026)"
    Colocated vs disaggregated RL placement, the hybrid-engine, and weight-synchronization resharding are now well-defined systems problems with production-grade open-source tooling. The dominant trend through 2025–2026 is fully disaggregated, asynchronous training with NCCL-broadcast (or increasingly RDMA P2P) weight sync, driven by the long reasoning rollouts of GRPO-style training that make rollout-bound iteration times the bottleneck.

    **Foundational work**

    - [Sheng et al., *HybridFlow: A Flexible and Efficient RLHF Framework* (2024)](https://arxiv.org/abs/2409.19256) — introduces the single-controller hybrid engine, the 3D-HybridEngine for zero-redundancy weight resharding between FSDP training and TP inference, and the colocated vs disaggregated design space; the canonical systems paper for this chapter.
    - [Hu et al., *OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework* (2024)](https://arxiv.org/abs/2405.11143) — Ray-native disaggregated architecture with vLLM rollout workers and NCCL weight broadcast; the reference implementation of the disaggregated pattern.

    **Recent advances (2023–2026)**

    - [Noukhovitch et al., *Asynchronous RLHF: Faster and More Efficient Off-Policy RL for Language Models* (2024)](https://arxiv.org/abs/2410.18252) — ICLR 2025; rigorous treatment of staleness, importance-sampling correction, and the safety of off-policy data in async disaggregated training.
    - [Zhong et al., *StreamRL: Scalable, Heterogeneous, and Elastic RL for LLMs with Disaggregated Stream Generation* (2025)](https://arxiv.org/abs/2504.15930) — tackles pipeline bubbles and long-tail skewness in disaggregated RL via stream generation and skewness-aware dispatching.
    - [Fu et al., *AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning* (2025)](https://arxiv.org/abs/2505.24298) — fully decouples generation from training and reports up to 2.77× speedup over synchronous baselines.

    **Open-source & tools**

    - [verl-project/verl](https://github.com/verl-project/verl) — the HybridFlow implementation; supports colocated hybrid engine and disaggregated mode, FSDP/Megatron training backends, vLLM/SGLang rollout, and a built-in resharding manager for gather-fuse-reshard-broadcast weight sync.
    - [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — Ray + vLLM disaggregated RL framework supporting PPO, GRPO, REINFORCE++; the most widely adopted open-source disaggregated RL codebase.
    - [sgl-project/sglang](https://github.com/sgl-project/sglang) — high-performance inference engine with first-class RL support: sleep/wake APIs, three weight-refit strategies (disk, tensor, distributed NCCL), and generation pause/resume for weight updates; see also the [SGLang for RL docs](https://sgl-project.github.io/advanced_features/sglang_for_rl.html).

    **Go deeper**

    - [*Keep the Tokens Flowing: Lessons from 16 Open-Source RL Libraries* — Hugging Face Blog (2026)](https://huggingface.co/blog/async-rl-training-landscape) — surveys 16 frameworks across seven axes (orchestration, rollout buffer, weight sync protocol, staleness management, partial rollouts, LoRA support, training backend); the best single reference for comparing the async disaggregated landscape.
    - [*Updating 1T Parameters in Seconds: P2P Weight Transfer in Large-Scale Distributed RL* — LMSYS Blog (2026)](https://www.lmsys.org/blog/2026-04-29-p2p-update/) — RDMA-based P2P weight sync via Mooncake TransferEngine reduces a 1T-model broadcast from 53 s to 7.2 s, pointing toward the next generation of weight-sync infrastructure.
    - [*Accelerating RLHF with vLLM: Best Practices from OpenRLHF* — vLLM Blog (2025)](https://vllm.ai/blog/2025-04-23-openrlhf-vllm) — practical walkthrough of colocated IPC and disaggregated NCCL weight sync in the OpenRLHF + vLLM stack, with code examples.

## Further Reading

- **Sheng et al., "HybridFlow: A Flexible and Efficient RLHF Framework" (veRL, 2024)** — the single-controller architecture, the hybrid engine for colocation, and the resharding manager that reconciles training and inference layouts; the most directly relevant systems paper for this chapter.
- **Hu et al., "OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework"** — Ray-based disaggregated rollout workers with vLLM and NCCL weight broadcast; a clear reference implementation of the disaggregated pattern.
- **Ren et al., "ZeRO-Offload: Democratizing Billion-Scale Model Training"** — the optimizer-state-offload mechanism reused on a per-phase cadence in colocated time-slicing.
- **Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (vLLM, 2023)** — the rollout engine and its KV-cache memory model that determine what colocation must make room for.
- **Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"** and **Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"** — the two training-side layouts (sharded data parallel and tensor/pipeline parallel) whose mismatch with inference TP creates the resharding problem.
- **NVIDIA NCCL documentation and the CUDA IPC runtime API** — the collective-communication and zero-copy-handle primitives underlying weight-sync Mechanisms 2 and 3.
- **DeepSeek-AI, "DeepSeek-R1" (2025)** — a large-scale RLVR run whose long reasoning rollouts are the canonical rollout-bound workload motivating disaggregation.
