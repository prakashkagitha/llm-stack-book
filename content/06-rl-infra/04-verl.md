# 6.4 veRL: HybridFlow & The Single-Controller Architecture

In [The Anatomy of an RL-for-LLM System](../06-rl-infra/01-anatomy-rl-system.html) we drew the box diagram: a policy that generates rollouts, a reward source that scores them, and a trainer that turns rewards into gradients. In [The Generation–Training Loop & Rollout Engines](../06-rl-infra/02-generation-training-loop.html) we saw that the *generation* half wants an inference engine (vLLM/SGLang) with PagedAttention and continuous batching, while the *training* half wants a distributed-training engine (FSDP or Megatron) with sharded optimizer state. And in [TRL: HuggingFace's RL Library](../06-rl-infra/03-trl.html) we used a library that runs both halves inside a *single Python process* on each GPU — simple, hackable, and a perfect teaching tool, but one that hits a wall when you want to mix tensor parallelism in the trainer with a different parallelism in the rollout engine, or scale a 70B-policy GRPO run across 64 GPUs.

This chapter is about the framework that broke that wall and became the default substrate for serious RL-for-LLM research and production in 2024–2026: **veRL** (the "volcano engine Reinforcement Learning" library, often stylized `verl`), built on the ideas in the **HybridFlow** paper (Sheng et al., 2024). veRL's central thesis is deceptively small and enormously consequential: **the *control flow* of an RL algorithm and the *computation* of each stage live at different scales, so they should be programmed at different scales.** The high-level dataflow — "generate, score, compute advantages, update" — is written *once*, as ordinary single-threaded Python on a driver process (the **single controller**). Each heavy stage — generation, the actor forward/backward, the critic, the reference log-probs — runs as a SPMD (single-program-multiple-data) program across many GPUs (the **multi-controller** workers). HybridFlow is the glue that lets a single line of driver code like `advantages = compute_advantage(rewards, values)` dispatch onto, and gather results from, hundreds of GPU workers, *and* lets the system **reshard** a model's weights between the layout the trainer wants and the layout the rollout engine wants — without ever round-tripping through disk.

By the end you will understand: why naive single-controller and naive multi-controller architectures each fail; what the `WorkerGroup` / `ResourcePool` / `@register(dispatch=...)` machinery actually does; how the **3D-HybridEngine** reshards FSDP/Megatron training weights into vLLM's tensor-parallel inference layout with near-zero redundant memory; how Ray placement groups colocate the actor, critic, and rollout engine on the same GPUs; and why this design is what lets veRL scale where a TRL-style monolith cannot. We will write a miniature single-controller dispatcher and a from-scratch resharding routine so the mechanism is concrete, not magic.

## Two architectures, two failure modes

Before veRL, every distributed RL system sat at one of two extremes. Understanding *why both are wrong* is the fastest route to understanding why HybridFlow's hybrid is right.

### The single-controller extreme

A **single-controller** system has one process — call it the *driver* — that holds the algorithm logic and issues every operation as a remote call. This is how classic RL frameworks (and a lot of Ray code) are written: the driver says "actor, generate"; "reward model, score"; "trainer, take a step", and each call ships data to a remote actor and waits for the result. The dataflow reads exactly like the math:

```python
# Single-controller pseudo-code: the WHOLE algorithm in one readable loop.
for batch in prompts:
    responses = rollout_worker.generate(batch)          # remote call
    rewards   = reward_worker.score(batch, responses)   # remote call
    values    = critic_worker.value(batch, responses)   # remote call
    advantages, returns = compute_gae(rewards, values)  # LOCAL on the driver
    actor_worker.update(batch, responses, advantages)   # remote call
    critic_worker.update(batch, responses, returns)     # remote call
```

This is gorgeous to read and trivial to modify — swap GAE for GRPO's group baseline by editing one line. The problem is **what flows through the driver**. For a single update of a 7B policy with a batch of, say, 1024 prompts × 8 samples × 2048 tokens, the per-token log-probs, the hidden-state-free tensors, the response token ids, and the masks are *hundreds of megabytes to gigabytes per step*, and in a naive single-controller design they are gathered to the driver, sliced, and re-scattered. The driver becomes a serialization bottleneck and a single point of network congestion. Worse, the *intra-stage* parallelism (the all-reduces inside the actor's backward, the all-gathers inside generation) cannot be expressed at all — the driver only knows how to call "the actor," not how 8 actor shards talk to each other. So pure single-controller systems are either slow (everything funnels through the driver) or they cheat by hiding the parallelism inside opaque workers, losing the ability to compose parallelisms.

### The multi-controller extreme

A **multi-controller** system is the opposite: there is *no* central driver. Every GPU runs the *same* program (SPMD), and the RL algorithm is expressed as collective operations that all ranks execute in lockstep. This is exactly how Megatron-LM and DeepSpeed pretraining work (see [Megatron-LM, DeepSpeed & Parallelism in Practice](../03-pretraining/07-megatron-deepspeed.html)), and it is *maximally efficient*: there is no driver to bottleneck, and tensor/pipeline/data parallelism are native because every rank knows its place in the mesh.

The catch is that **the RL control flow now lives inside the SPMD program**, replicated across every rank, and it is a nightmare to write and change. Consider what "compute GRPO advantages, then take a PPO step" looks like when there is no privileged process: every rank must agree on which prompts are in the batch, the group reductions for the advantage must be done with explicit collectives, and crucially, the *boundaries between stages* — where generation ends and training begins, where the rollout engine's parallelism (TP=2) differs from the trainer's (FSDP across 8) — must be hand-coded as resharding collectives interleaved with the algorithm. A researcher who wants to try a new advantage estimator must now edit SPMD code full of `torch.distributed.all_gather` calls and rank arithmetic. Multi-controller is fast but rigid; every new algorithm is a distributed-systems project.


{{fig:verl-single-vs-multi-controller}}


### HybridFlow's resolution: single-controller *between* stages, multi-controller *within*

HybridFlow's insight is that these two regimes operate at **different granularities**, and you can have both:

- **Between stages** (generate → score → advantage → update) the data volume is large but the number of "operations" is tiny (a handful per step) and the logic changes constantly as researchers experiment. Use a **single controller**: one driver runs the readable Python loop.
- **Within a stage** (the actor's forward/backward, vLLM generation) the operation count is enormous (millions of CUDA kernels) and the logic is stable (it's just a transformer). Use **multi-controller SPMD**: each stage is a `WorkerGroup` of ranks running an efficient parallel program.

The driver issues *one* logical instruction per stage — `actor.generate_sequences(batch)` — and a **dispatch layer** fans that single call out to all ranks of the actor's `WorkerGroup`, each of which runs its slice of the SPMD program, then a **collect layer** gathers the results back to the driver. The driver never touches the intra-stage collectives; the workers never touch the inter-stage control flow. The data that flows to the driver is only the *logical batch* (prompts, responses, rewards, advantages) — not the activations or gradients, which stay sharded inside the workers. This is the whole architecture in one sentence: **a single-controller dataflow program orchestrating multi-controller SPMD computation graphs.**

## The HybridFlow programming model

Let us make the abstraction concrete with the actual building blocks veRL exposes. There are four that you must understand: `ResourcePool`, `Worker` / `WorkerGroup`, the `@register(dispatch=...)` decorator, and the `DataProto` container.

### Resource pools and worker groups

A **`ResourcePool`** is a reservation of GPUs — for example "8 GPUs, as 1 node of 8." A **`Worker`** is a class whose instances each occupy one GPU (one rank). A **`WorkerGroup`** binds a `Worker` class to a `ResourcePool`: it spins up one Ray actor per rank, sets each rank's `RANK`, `WORLD_SIZE`, `MASTER_ADDR`, etc., and initializes `torch.distributed` so the group can run SPMD. Critically, **multiple worker groups can share the same resource pool** — this is how veRL *colocates* the actor, the rollout engine, the critic, and the reference model on the *same* physical GPUs, time-slicing them. We will return to colocation; for now the mental model is:


{{fig:verl-resourcepool-colocation}}


The single driver holds *handles* to these worker groups and calls methods on them. Here is the shape of veRL's main PPO/GRPO loop, lightly simplified to show the single-controller structure. Read it as ordinary Python — that is the point.

```python
# The driver process (single controller). This is the ENTIRE RL algorithm.
# Each `*.method(...)` is one logical instruction that fans out to all ranks
# of that worker group and gathers the result back as a DataProto.

def fit(self):
    for step, prompt_batch in enumerate(self.dataloader):
        # 1) ROLLOUT: actor weights are resharded into the vLLM layout and we
        #    generate G samples per prompt. Returns responses + per-token logprobs.
        batch = self.actor_rollout_wg.generate_sequences(prompt_batch)

        # 2) RECOMPUTE old log-probs under the TRAIN-layout actor (numerical
        #    consistency with the trainer; see the inference/train mismatch note).
        batch = batch.union(self.actor_rollout_wg.compute_log_prob(batch))

        # 3) REFERENCE log-probs for the KL term (frozen model).
        if self.use_reference_policy:
            batch = batch.union(self.ref_policy_wg.compute_ref_log_prob(batch))

        # 4) REWARD: rule-based verifier and/or a reward model.
        batch = batch.union(self.reward_fn(batch))

        # 5) VALUES (PPO only; GRPO skips this — no critic).
        if self.use_critic:
            batch = batch.union(self.critic_wg.compute_values(batch))

        # 6) ADVANTAGE: pure LOCAL computation on the driver. Tiny tensors.
        batch = compute_advantage(batch, adv_estimator=self.config.adv_estimator)

        # 7) UPDATE: scatter the batch to the actor (and critic) workers; each
        #    runs its sharded forward/backward; gradients stay inside the group.
        if self.use_critic:
            self.critic_wg.update_critic(batch)
        self.actor_rollout_wg.update_actor(batch)
```

Notice that step 6 — the part researchers most want to change — is *local, single-threaded Python on small tensors*. Swapping PPO's GAE for GRPO's group baseline, or for RLOO, or for a brand-new estimator, is a one-function edit with no distributed code in sight. That is the productivity win of the single controller. Steps 1, 2, 5, and 7 are the heavy SPMD stages, each hidden behind one method call.

### The dispatch decorator: how one call becomes N

The magic that turns `self.actor_rollout_wg.generate_sequences(prompt_batch)` (one call on the driver) into "run the SPMD generate program on all 8 ranks and gather" is the **`@register(dispatch=...)`** decorator on the worker method. It declares *how the input should be split across ranks* and *how the outputs should be combined*. veRL ships a small set of dispatch modes; the important ones:

| Dispatch mode | Input handling | Output handling | Used for |
|---|---|---|---|
| `ONE_TO_ALL` | broadcast the same args to every rank | take rank 0's result | broadcasts, barriers, config |
| `ALL_TO_ALL` | pass args through unchanged | return list from all ranks | generic collectives |
| `DP_COMPUTE_PROTO` | **shard the batch along the data-parallel dim**, one slice per DP rank | **concatenate** the per-rank `DataProto`s back into one | the workhorse: generation, log-prob, update |
| `MEGATRON_COMPUTE_PROTO` | shard over DP, replicate within TP/PP groups | gather from DP ranks only | Megatron-backed workers |

The cleverness of `DP_COMPUTE_PROTO` is that it understands the worker's *parallelism topology*: it splits the batch only across the **data-parallel** ranks and *replicates* it across the **tensor/pipeline-parallel** ranks (which all need the same data to cooperate on one micro-batch). The driver does not know or care about TP — it just hands over a batch and gets back a batch. Here is a from-scratch sketch of what the decorator does, so the mechanism is not a black box:

```python
# A miniature reconstruction of veRL's dispatch mechanism. The real one is more
# careful about padding, async futures, and TP/PP replication, but this captures
# the single-controller -> multi-controller fan-out/fan-in exactly.

import functools

# Dispatch functions: given a WorkerGroup and the call's args, return a LIST of
# (args, kwargs) — one entry per rank.
def dispatch_dp_compute_proto(worker_group, batch):
    dp_size = worker_group.dp_size            # number of data-parallel groups
    tp_size = worker_group.tp_size            # ranks per DP group (TP * PP)
    chunks = batch.chunk(dp_size)             # split the BATCH across DP groups only
    per_rank = []
    for dp_rank in range(dp_size):
        for _ in range(tp_size):              # replicate the chunk to every TP rank
            per_rank.append((chunks[dp_rank],))   # in this DP group
    return per_rank                            # length == world_size

# Collect functions: given the list of per-rank outputs, fold them into one result.
def collect_dp_compute_proto(worker_group, outputs):
    dp_size = worker_group.dp_size
    tp_size = worker_group.tp_size
    # Keep only ONE representative per DP group (TP ranks computed identical batch
    # outputs); concatenate across DP groups to reconstruct the full batch order.
    reps = [outputs[dp_rank * tp_size] for dp_rank in range(dp_size)]
    return DataProto.concat(reps)

DISPATCH = {
    "DP_COMPUTE_PROTO": (dispatch_dp_compute_proto, collect_dp_compute_proto),
}

def register(dispatch):
    """Decorator placed on Worker methods. Records the dispatch mode so the
    WorkerGroup proxy knows how to fan out / fan in when the DRIVER calls it."""
    def decorator(fn):
        fn._dispatch_mode = dispatch
        @functools.wraps(fn)
        def inner(self, *args, **kwargs):     # runs ON the worker (one rank)
            return fn(self, *args, **kwargs)
        return inner
    return decorator

class WorkerGroupProxy:
    """Lives on the DRIVER. `self.workers` are Ray actor handles (one per rank)."""
    def __init__(self, workers, dp_size, tp_size):
        self.workers, self.dp_size, self.tp_size = workers, dp_size, tp_size

    def call(self, method_name, batch):
        dispatch_fn, collect_fn = DISPATCH[
            getattr(WorkerClass, method_name)._dispatch_mode]
        per_rank_args = dispatch_fn(self, batch)            # split + replicate
        # Launch the SAME method on every rank in parallel (Ray remote calls).
        futures = [w.__getattr__(method_name).remote(*a)
                   for w, a in zip(self.workers, per_rank_args)]
        outputs = ray.get(futures)                          # gather all ranks
        return collect_fn(self, outputs)                    # fold into one DataProto
```

So when the driver writes `actor_rollout_wg.generate_sequences(batch)`, under the hood the proxy (1) consults the dispatch mode registered on `generate_sequences`, (2) splits the batch across DP groups and replicates within TP, (3) launches the method on all ranks via Ray, (4) gathers, and (5) concatenates. The driver wrote one line; 8 GPUs ran an SPMD generation. **That is HybridFlow.**

{{fig:verl-dispatch-fanout}}

### DataProto: the typed batch that flows between stages

The object passed around — `DataProto` — is veRL's batch container: a dict of named tensors (`prompts`, `responses`, `attention_mask`, `old_log_probs`, `ref_log_probs`, `rewards`, `advantages`, …) plus non-tensor metadata. It supports `chunk` (for dispatch), `concat` (for collect), and `union` (to merge in new columns each stage produces). It is the *only* thing that crosses the single-/multi-controller boundary, and it is deliberately small: logical batch data, never activations or optimizer state. Keeping the boundary data minimal is what prevents the single-controller bottleneck that sinks naive designs.

!!! note "Aside: why Ray and not just torchrun"
    veRL uses **Ray** as the actor/scheduling substrate, not because Ray is faster at collectives (it isn't — the heavy collectives still go through NCCL inside each worker group), but because Ray gives you *named, addressable, individually-callable* worker processes with **placement groups** for GPU pinning. The single-controller driver needs to hold a handle to "the actor worker group" and call methods on it heterogeneously (generate, then update), interleaved with calls to the critic and reward groups. `torchrun` gives you a flat SPMD world with no such addressability. Ray is the abstraction that makes the *between-stage* single-controller possible; NCCL remains the abstraction that makes the *within-stage* multi-controller fast. See [OpenRLHF, NeMo-Aligner & Ray-Based Systems](../06-rl-infra/05-openrlhf-nemo-ray.html) for the broader Ray-RL ecosystem.

## The 3D-HybridEngine: resharding between train and rollout

We now reach the part of veRL that is the most technically interesting and the source of much of its performance: the **3D-HybridEngine**. The problem it solves is unavoidable in any colocated RL system. The actor model must live in *two different parallel layouts*:

- **Training layout.** During the update, the actor is sharded for training — typically **FSDP** (parameters, gradients, and optimizer state sharded across all data-parallel ranks; see [Distributed Training I: Data Parallelism, DDP, ZeRO & FSDP](../03-pretraining/05-distributed-data-parallel.html)) or **Megatron 3D parallelism** (tensor × pipeline × data; see [Distributed Training II: Tensor, Pipeline, Sequence & Expert Parallelism](../03-pretraining/06-distributed-model-parallel.html)). Optimizer state dominates memory: Adam needs two moments per parameter, so the training engine holds roughly $4\times$–$6\times$ the raw parameter bytes.
- **Rollout layout.** During generation, the *same* weights must feed **vLLM** or **SGLang**, which shard the model purely with **tensor parallelism** (TP) for low-latency decode and use PagedAttention for the KV cache (see [vLLM: Architecture, PagedAttention & Internals](../07-inference-serving/03-vllm-internals.html) and [PagedAttention & KV-Cache Memory Management](../04-kernels-efficiency/06-paged-attention-kv.html)). The rollout engine needs *no* optimizer state and *no* gradients — just the bf16 weights and a big KV cache.

Every RL step must convert the actor weights from the training layout to the rollout layout (before generation) and conceptually back (the trainer just keeps its own copy and updates it in place). The naive way to do this is catastrophic: gather the full weights to every rank, write a checkpoint, and have vLLM load it — disk round-trips of tens of gigabytes per step, plus a full unsharded copy of the model materialized in memory. The 3D-HybridEngine does it **in GPU memory, in place, with a clever rank arrangement that minimizes redundant copies.**

### Why resharding is non-trivial: a TP-degree mismatch

The core difficulty is that the training TP degree and the rollout TP degree usually differ. Suppose training uses Megatron with tensor-parallel degree $p$ (each weight matrix is split column- or row-wise across $p$ ranks) and rollout uses vLLM with tensor-parallel degree $q$, where often $q < p$ (inference wants fewer, fatter shards for decode efficiency). To form the rollout shards you must **re-partition** the parameter tensors: gather the $p$ training shards of each layer and re-split them into $q$ rollout shards. Done naively across all GPUs, this is an all-gather of the *entire model* followed by a re-scatter — $O(\text{model size})$ communication and a transient full copy on each rank.

The HybridFlow trick is to arrange the training and rollout parallel *groups* so that resharding is a **local regrouping** within small sets of GPUs rather than a global all-gather. Concretely, the 3D-HybridEngine keeps the training and inference TP groups *nested on the same physical GPUs* so that the gather-and-resplit only ever happens among the $\max(p,q)$ GPUs that already share those parameters — never across the whole cluster. The redundant memory is bounded by one TP group's worth of weights, not the whole model, and the communication is one *intra-node* all-gather per layer, overlapped with compute.

### A from-scratch resharding routine

To demystify it, here is a self-contained reconstruction of the central operation: take a weight matrix that is tensor-parallel-sharded $p$ ways (training) and re-shard it $q$ ways (rollout). This is exactly the per-layer kernel the 3D-HybridEngine runs, minus the cross-rank choreography.

```python
import torch
import torch.distributed as dist

def reshard_column_parallel(local_shard: torch.Tensor,
                            train_tp_group, train_tp_size: int,
                            rollout_tp_size: int, rank_in_group: int):
    """
    Reshard one COLUMN-parallel weight (e.g. an MLP up-projection, split along the
    OUTPUT dim) from train_tp_size shards to rollout_tp_size shards, IN GPU MEMORY.

    local_shard : this rank's slice of the full weight, shape (in_dim, out_dim/p).
    Returns this rank's rollout slice, shape (in_dim, out_dim/q), or None if this
    rank is not used by the rollout engine (q < p case).

    Mechanism:
      1. all-gather the p training shards WITHIN the (small) TP group -> full weight.
      2. re-split the full weight into q rollout shards.
      3. each rank keeps the rollout shard it is responsible for.
    The all-gather is bounded by ONE TP group (intra-node), not the whole cluster.
    """
    p, q = train_tp_size, rollout_tp_size

    # --- Step 1: gather the p column-shards into the full output dimension. ---
    gathered = [torch.empty_like(local_shard) for _ in range(p)]
    dist.all_gather(gathered, local_shard, group=train_tp_group)
    full_weight = torch.cat(gathered, dim=1)         # (in_dim, out_dim) full, on every rank

    # --- Step 2: re-split into q rollout shards along the same (output) dim. ---
    out_dim = full_weight.shape[1]
    assert out_dim % q == 0, "output dim must be divisible by rollout TP degree"
    rollout_shards = full_weight.chunk(q, dim=1)     # list of q tensors

    # --- Step 3: which rollout shard does THIS physical GPU own? ---
    # The 3D-HybridEngine maps rank_in_group -> a rollout shard id so that the
    # first q ranks of the training group become the q rollout ranks. Ranks >= q
    # are idle during rollout (their weights were gathered above and can be freed).
    if rank_in_group < q:
        my_rollout_shard = rollout_shards[rank_in_group].contiguous()
        return my_rollout_shard                      # (in_dim, out_dim/q)
    else:
        return None                                  # not a rollout rank

# Row-parallel weights (split along the INPUT dim) are symmetric: all-gather along
# dim=0 instead of dim=1, then re-chunk along dim=0. Attention QKV and o_proj need
# head-aware regrouping so that whole heads stay together after re-sharding.
```

The thing to internalize: resharding is **gather-within-a-small-group, then re-split** — purely arithmetic on the partition boundaries — and the *engineering* is making sure (a) the gather is intra-node, (b) only one TP group's worth of extra memory is ever live, and (c) the optimizer state and gradients are *not* gathered (they stay in the training layout untouched; only the bf16 parameters are copied into the rollout engine's weight buffers). After resharding, vLLM's weight tensors are *updated in place* via its `load_weights` / `update_weights` API — no disk, no checkpoint.

{{fig:verl-reshard-p-to-q}}

### Weight synchronization, in place

Once weights are resharded into the rollout layout, they must be *injected* into the running vLLM engine. vLLM exposes a path to overwrite its model parameters from in-memory tensors (in recent versions via a collective `update_weights` or by directly assigning into the model's parameters and refreshing any quantization/caches). veRL's actor-rollout worker calls this every step, so the rollout engine always generates with the *latest* policy. Because the rollout engine is colocated on the same GPUs, the transfer is a device-to-device copy (or even a no-copy view when layouts align), not a network or PCIe transfer.


{{fig:verl-colocated-rl-step}}


The disaggregated alternative — putting the rollout engine on *separate* GPUs and streaming weights over the network — is covered in [Colocated vs Disaggregated RL & Weight Synchronization](../06-rl-infra/07-colocated-vs-disaggregated.html). veRL supports both; the colocated path with the 3D-HybridEngine is its signature and is what most single-cluster runs use.

## Placement, colocation, and Ray mechanics

We have said the actor, critic, reference, and rollout engine *colocate* on the same GPUs. Let us make that precise, because it is where Ray placement groups earn their keep and where most real-world veRL configuration bugs hide.

### Placement groups: pinning ranks to GPUs

Ray's **placement group** is a reservation of "bundles," where each bundle requests resources (e.g. `{"GPU": 1, "CPU": 8}`) and a *strategy* governs how bundles are packed onto nodes. veRL uses these to lay out worker groups deterministically:

- **`STRICT_PACK` / `PACK`** — put all bundles on as few nodes as possible. Good for tensor parallelism (TP ranks must be on the same node for fast NVLink all-reduces).
- **`STRICT_SPREAD` / `SPREAD`** — distribute bundles across nodes (for data parallelism that tolerates inter-node bandwidth).

A typical single-node 8-GPU setup creates one placement group with 8 GPU bundles, packed on the node. The `ActorRolloutWorkerGroup`, `CriticWorkerGroup`, and `RefWorkerGroup` are all bound to *that same placement group* — meaning their rank-$i$ Ray actors all land on GPU $i$. Because they share GPUs, they cannot all be active at once; veRL **time-slices** them: during rollout the actor-rollout group's vLLM engine owns the GPU compute and the bulk of memory (KV cache), then it is paused/offloaded and the actor's training forward/backward runs, then the critic, then the reference. This is the colocated design, and its big win is **GPU utilization**: no GPU sits idle waiting for another stage, because every stage runs on every GPU.

```yaml
# Sketch of veRL's resource/placement configuration (conceptual).
# One node, 8 GPUs. Actor+rollout colocated; critic and ref share the same GPUs.
resource_pool:
  process_on_nodes: [8]          # 8 ranks on 1 node -> one placement group, PACK
actor_rollout_ref:
  actor:
    strategy: fsdp               # training engine: FSDP (or "megatron")
    fsdp_config: { param_offload: false, optimizer_offload: false }
  rollout:
    name: vllm                   # rollout engine: vLLM (or "sglang")
    tensor_model_parallel_size: 2   # rollout TP degree q (train uses FSDP, "p"=1 view)
    gpu_memory_utilization: 0.5     # fraction of GPU mem vLLM may use for weights+KV
    n: 8                            # G: samples per prompt (the group size)
  ref:
    fsdp_config: { param_offload: true }   # ref is frozen -> offload to CPU when idle
critic:
  strategy: fsdp
```

### The memory budget of colocation

Colocation is powerful but it forces a tight memory budget: at any instant the GPU must hold whatever the *currently active* stage needs, and weights/state for the *inactive* stages either stay resident (cheap, e.g. bf16 actor params) or get **offloaded to CPU** (the `param_offload`/`optimizer_offload` flags above). The single biggest knob is `gpu_memory_utilization` for vLLM — it tells the rollout engine how much VRAM to claim for weights plus KV cache, and it must leave room for the trainer's optimizer state when training resumes.

!!! example "Worked example: will a 7B GRPO run fit on 8×80GB?"
    Take a 7B-parameter actor, bf16 weights, trained with FSDP across 8×A100-80GB, rollout via vLLM with TP=2, group size $G=8$, GRPO (so **no critic**). Let us budget per GPU.

    **Training-side resident memory (FSDP, sharded across 8 ranks):**

    - Parameters (bf16): $7\text{B}\times 2\text{ B} = 14\text{ GB}$ total $\Rightarrow 14/8 = 1.75\text{ GB}$ per rank.
    - Gradients (bf16): another $1.75\text{ GB}$ per rank.
    - Adam optimizer state in fp32 (master weights + two moments): roughly $7\text{B}\times(4+4+4)\text{ B} = 84\text{ GB}$ total $\Rightarrow 84/8 = 10.5\text{ GB}$ per rank.
    - Subtotal training state: $\approx 14\text{ GB}$ per rank, *persistently sharded*.

    **Rollout-side memory (vLLM, TP=2):** the rollout engine needs a *full* bf16 copy of the model split across its TP group. With TP=2, each rollout rank holds $14\text{ GB}/2 = 7\text{ GB}$ of weights. The KV cache then uses whatever `gpu_memory_utilization` leaves. With a 32k-token context budget across the batched group samples, the KV cache can easily want **tens of GB** — this is usually the dominant rollout cost. (For the KV-cache size formula, see [The Anatomy of LLM Inference: Prefill, Decode & The KV Cache](../07-inference-serving/01-anatomy-inference.html).)

    **The resharding transient:** when the 3D-HybridEngine gathers a layer to re-split it, the live extra memory is bounded by *one TP group's* weights for *one layer* at a time (a few hundred MB), not the whole 14 GB model — that bound is the entire point of the nested-group arrangement.

    **Does it fit?** Per 80 GB GPU during rollout: $\sim 14\text{ GB}$ (training state, kept resident) $+ \sim 7\text{ GB}$ (rollout weights) $\approx 21\text{ GB}$ baseline, leaving $\sim 59\text{ GB}$ for the KV cache and activations — comfortable. During the update stage, vLLM's KV cache is released, freeing tens of GB for training activations. The colocation works because *the two stages' peak memories don't coincide*: rollout's peak is KV cache, training's peak is activations, and they happen at different times. **If you instead ran PPO with a 7B critic**, you would add another $\sim 14\text{ GB}$/rank of critic training state, and the budget tightens enough that you would likely enable `optimizer_offload`. This is one concrete, mechanical reason the field prefers critic-free GRPO/RLOO for large policies (see [GRPO, RLOO & Critic-Free RL](../05-posttraining-alignment/08-grpo-rloo.html)).

### Hybrid sharding for very large models

For models too large for a single node, veRL combines parallelisms: FSDP or Megatron across nodes for training, and vLLM TP/PP within nodes for rollout, with the 3D-HybridEngine resharding between them. The "3D" in 3D-HybridEngine refers precisely to managing the *three* parallel dimensions (data, tensor, pipeline) consistently across the train↔rollout boundary so that a parameter living at training coordinate $(\text{dp}, \text{tp}, \text{pp})$ knows exactly which rollout coordinate it must land at. This is the generalization of the single-matrix resharding routine above to the full 3D mesh.

## Why veRL scales: the throughput argument

Put the pieces together and you can state precisely *why* veRL scales where a TRL-style monolith stalls.

1. **No driver bottleneck.** Only the small logical batch (prompts, responses, rewards, advantages) crosses the single-controller boundary; activations, gradients, and optimizer state never leave the worker groups. The driver's per-step data is megabytes, not the hundreds of GB that move *inside* the SPMD stages.
2. **Native parallelism composition.** Because each stage is its own SPMD `WorkerGroup`, you can independently choose the trainer's parallelism (FSDP, or Megatron TP×PP×DP) and the rollout engine's parallelism (vLLM TP), and the 3D-HybridEngine bridges them. A monolith that runs train and generate in one process is stuck with one parallel layout for both — the wrong layout for at least one of them.
3. **Colocation kills idle GPUs.** With actor, rollout, critic, and reference time-sliced on the same GPUs via placement groups, no GPU waits on another machine. In a *disaggregated* design, generation GPUs idle while training GPUs work and vice versa unless you carefully pipeline; colocation sidesteps that for single-cluster runs.
4. **In-memory resharding, no disk.** Weight sync between train and rollout is a device-to-device regroup, not a checkpoint write/read. At 70B scale, the difference between an in-memory regroup and a disk round-trip per step is the difference between a step taking seconds versus minutes.
5. **The expensive engines are best-of-breed and unmodified.** veRL does not reimplement attention or a sampler; it *orchestrates* vLLM/SGLang for generation and FSDP/Megatron for training. You inherit PagedAttention, continuous batching, FlashAttention, and FSDP's sharding for free, and upgrade them independently.

### What still bottlenecks veRL

No system is free. The dominant cost in synchronous veRL is the **generation stage's tail latency**: a batched group rollout finishes only when its *slowest, longest* sample finishes, so a few very long responses stall the whole step ("straggler" or "long-tail" effect). Mitigations include the async and disaggregated designs of [Prime-RL, Async RL & Decentralized Training](../06-rl-infra/06-prime-rl-async.html) and load-balancing tricks in [Scaling RL: Throughput, Load Balancing & The Latest Tricks](../06-rl-infra/11-scaling-rl-tricks.html). veRL has steadily added asynchronous rollout and partial-rollout features to attack exactly this. The second cost is the **train/inference numerical mismatch**: vLLM's generation kernels and the trainer's forward pass compute log-probs with different code paths, so the `old_log_prob` used in the importance ratio can disagree with what the sampler actually used. veRL recomputes `old_log_prob` under the *training* engine (step 2 of the driver loop) to keep the ratio self-consistent; this is a subtle but load-bearing correctness detail discussed in [Advantage Estimation, KL Control & Stability Tricks](../06-rl-infra/09-advantage-kl-tricks.html).

## A worked example: writing a custom reward and advantage in veRL

The single-controller design's biggest practical payoff is that the parts you want to customize — the **reward** and the **advantage** — are plain Python on small tensors, requiring zero distributed code. Here is a realistic sketch of plugging a verifiable math reward and a custom advantage into veRL's driver-side hooks. This is the shape of an actual veRL extension.

```python
# 1) A custom reward function. veRL calls this on the DRIVER with a DataProto of
#    decoded responses. No GPUs, no collectives — just scoring strings. In a real
#    run the verifier might call a sandboxed code runner (see chapter 6.8).
import re

def math_verifiable_reward(data, tokenizer, **kwargs):
    """Return a per-sample scalar reward tensor for a DataProto batch."""
    rewards = []
    responses = tokenizer.batch_decode(data.batch["responses"],
                                       skip_special_tokens=True)
    golds = data.non_tensor_batch["ground_truth"]      # parallel list of answers
    for resp, gold in zip(responses, golds):
        m = re.search(r"\\boxed\{([^}]*)\}", resp)     # parse the boxed answer
        pred = m.group(1).strip() if m else None
        correct = (pred is not None) and is_math_equiv(pred, gold)  # symbolic check
        fmt = 1.0 if ("<think>" in resp and "</think>" in resp) else 0.0
        rewards.append(1.0 * correct + 0.1 * fmt)      # correctness dominates
    return torch.tensor(rewards, dtype=torch.float32)

# 2) A custom advantage estimator, registered with veRL's advantage dispatcher.
#    This is the GRPO group-baseline, written as ordinary local Python. Because it
#    runs on the DRIVER over tiny per-sample reward tensors, there is NOTHING
#    distributed here — contrast with a multi-controller system where this same
#    computation would require explicit all-gathers across ranks.
def grpo_group_advantage(rewards, group_size, eps=1e-6, normalize_std=False):
    """rewards: (B,) with B = n_prompts * group_size, grouped contiguously."""
    g = rewards.view(-1, group_size)                   # (n_prompts, G)
    adv = g - g.mean(dim=1, keepdim=True)              # group-mean baseline
    if normalize_std:                                  # Dr.GRPO recommends FALSE
        adv = adv / (g.std(dim=1, keepdim=True) + eps)
    return adv.reshape(-1)                             # one scalar advantage / sample
# (See chapter 5.8 for the full GRPO/RLOO/Dr.GRPO derivations and trade-offs.)

# 3) Wiring: in the YAML config you point veRL at these. The driver loop's
#    step 4 (reward) and step 6 (advantage) call them; steps 1/2/5/7 (the heavy
#    SPMD stages) are untouched. You changed the algorithm without writing a single
#    line of torch.distributed code. THAT is the HybridFlow productivity win.
```

Compare the effort: in a pure multi-controller system, changing the advantage estimator means editing SPMD code, getting the group reductions right with collectives, and worrying about which rank holds which sample. In veRL, it is the function above, run once on the driver over a `(B,)` tensor. The heavy lifting (generation across 8 GPUs, FSDP backward across 8 GPUs) is unchanged and invisible to you.

!!! tip "Practitioner tip: match the rollout TP degree to your decode shape, not your train shape"
    A common veRL misconfiguration is setting the rollout `tensor_model_parallel_size` equal to whatever the trainer uses. They are independent knobs. The rollout TP degree should be chosen for *decode efficiency and KV-cache headroom*: too high and you pay all-reduce latency on every decode step for a model that already fits; too low and the KV cache can't hold your group-of-$G$ long generations. Start with the smallest TP that fits the model plus a generous KV budget, and let the 3D-HybridEngine handle the train→rollout regroup. Watching the vLLM "KV cache usage" and "preemption" metrics tells you immediately if TP is too small.

!!! warning "Common pitfall: gpu_memory_utilization starves the trainer"
    `rollout.gpu_memory_utilization` tells vLLM how much VRAM to *claim and hold*. Set it too high (say 0.9) and vLLM's KV cache pool eats memory that the trainer needs for activations and optimizer state when the step switches to the update stage — you get an OOM *after* a successful rollout, which is confusing because generation worked. In colocated runs, leave clear headroom (often 0.4–0.6) so the trainer's peak fits alongside vLLM's reserved pool, or enable `param_offload`/`optimizer_offload` to spill training state to CPU during rollout. The two stages share one GPU; budget for the *max* of their peaks at every switch point, not the sum, but only if you actually free each stage's memory at the boundary.

## How veRL relates to the rest of the ecosystem

It helps to place veRL on the same map as the other frameworks in this Part.

| Framework | Controller model | Train engine | Rollout engine | Resharding | Best at |
|---|---|---|---|---|---|
| **TRL** (ch. 6.3) | single process per GPU (monolith) | Accelerate/FSDP, DeepSpeed | vLLM (colocated or server) | simple weight load | hackability, small/medium scale, research velocity |
| **veRL** (this ch.) | **single-controller + multi-controller (HybridFlow)** | FSDP **or** Megatron | vLLM **or** SGLang | **3D-HybridEngine, in-memory** | scaling, composable parallelism, large policies |
| **OpenRLHF** (ch. 6.5) | Ray multi-controller, disaggregated | DeepSpeed ZeRO | vLLM | weight broadcast over Ray/NCCL | Ray-native disaggregated PPO |
| **NeMo-Aligner** (ch. 6.5) | Megatron-centric | Megatron-Core | TRT-LLM / in-framework | Megatron-aware | NVIDIA stack, max Megatron scale |
| **Prime-RL** (ch. 6.6) | async, decentralized | FSDP | vLLM | async weight sync | async / cross-datacenter RL |

The throughline: TRL optimizes for *velocity at modest scale* by keeping everything in one process; veRL optimizes for *scale with flexibility* by separating the single-controller dataflow from multi-controller SPMD computation and bridging layouts with the 3D-HybridEngine. If you outgrow TRL because you need Megatron-scale training *and* a different vLLM rollout layout *and* 70B policies, veRL is where you go. If you need fully asynchronous or geographically distributed training, you reach further to the systems in [Prime-RL, Async RL & Decentralized Training](../06-rl-infra/06-prime-rl-async.html).

!!! interview "Interview Corner"
    **Q:** Explain HybridFlow's "single-controller + multi-controller" hybrid. Why is *neither* extreme good enough on its own, and give one concrete operation that lives on each side of the boundary.

    **A:** The two design extremes each fail in a complementary way. A pure **single-controller** system (one driver issues every op as a remote call) makes the RL algorithm trivially readable and editable, but it funnels all the batch data — log-probs, masks, response ids — through the driver, which becomes a serialization and network bottleneck, and it cannot express the *intra-stage* collectives (the all-reduces inside the actor's backward), so it can't compose parallelism. A pure **multi-controller** SPMD system (every GPU runs the same program, like Megatron) is maximally efficient and composes tensor/pipeline/data parallelism natively, but it buries the RL control flow inside replicated SPMD code, so changing the advantage estimator means editing distributed code full of explicit collectives and rank arithmetic. HybridFlow uses **single-controller between stages** (one driver runs the readable `generate → score → advantage → update` loop; only the small logical batch crosses this boundary) and **multi-controller within each stage** (each stage is a `WorkerGroup` of SPMD ranks; the heavy collectives stay inside). A `@register(dispatch=DP_COMPUTE_PROTO)` decorator fans one driver call out to all ranks (splitting the batch across data-parallel ranks, replicating within tensor-parallel ranks) and gathers results back. Concrete examples: **on the single-controller side**, `advantages = compute_advantage(rewards, values)` — tiny tensors, local Python, no collectives. **On the multi-controller side**, `actor.update_actor(batch)` — a sharded FSDP/Megatron forward-backward whose all-reduces never touch the driver. The 3D-HybridEngine additionally reshards the actor between the trainer's FSDP/Megatron layout and vLLM's tensor-parallel rollout layout *in GPU memory*, so the same weights serve both stages without a disk round-trip.

!!! interview "Interview Corner"
    **Q:** In a colocated veRL run, the actor trains with FSDP and generates with vLLM at TP=2. Walk through what physically happens to the weights and optimizer state in one RL step, and where the memory peaks are.

    **A:** The optimizer state (fp32 master weights + Adam moments) and gradients live the *entire* step in the **FSDP training layout**, sharded across all data-parallel ranks, and are *never moved* — only the bf16 parameters are copied for rollout. At the start of the step, the **3D-HybridEngine** reshards those bf16 params from the FSDP/training partition into vLLM's TP=2 rollout partition: it gathers each layer's shards *within a small TP group* (intra-node, bounded by one TP group's weights, not the whole model) and re-splits them into the TP=2 layout, then injects them into vLLM's parameter buffers in place via `update_weights` — no disk. vLLM then generates $G$ samples per prompt; here the memory peak is the **PagedAttention KV cache**, which can be tens of GB. After generation, the KV cache is released; the driver computes rewards and advantages locally; then `update_actor` runs the FSDP forward/backward, whose memory peak is **training activations**. The key is that the rollout peak (KV cache) and the training peak (activations) occur at *different times*, so colocation fits as long as you don't hold both — which is why `gpu_memory_utilization` must leave headroom and why you may offload optimizer state to CPU during rollout. With GRPO there is no critic, so you avoid a second full set of training state; that is part of why critic-free methods scale better in colocated setups.

!!! key "Key Takeaways"
    - **HybridFlow = single-controller *between* stages + multi-controller *within* stages.** The driver runs the readable `generate → score → advantage → update` loop on small logical batches; each heavy stage runs as an SPMD `WorkerGroup`. This beats both pure single-controller (driver bottleneck, can't compose parallelism) and pure multi-controller (RL logic buried in distributed code).
    - **The `@register(dispatch=...)` decorator** turns one driver method call into a fan-out across all ranks of a worker group and a fan-in of the results. `DP_COMPUTE_PROTO` shards the batch across data-parallel ranks and replicates within tensor-parallel ranks — the driver never sees the intra-stage collectives.
    - **The 3D-HybridEngine reshards the actor in GPU memory** between the trainer's FSDP/Megatron layout and the rollout engine's vLLM/SGLang tensor-parallel layout. It gathers-within-a-TP-group then re-splits, bounding the transient memory to one TP group's weights and avoiding any disk checkpoint round-trip per step.
    - **Optimizer state never moves.** Only bf16 parameters are copied for rollout; gradients and Adam state stay in the training layout. The train peak (activations) and rollout peak (KV cache) happen at different times, which is what makes colocation fit.
    - **Ray provides placement groups and addressable workers**, enabling colocation of the actor, critic, reference, and rollout engine on the *same* GPUs (time-sliced). This maximizes GPU utilization. NCCL still does the fast collectives inside each worker group.
    - **veRL scales because** no large data crosses the driver, parallelisms compose freely across stages, colocation eliminates idle GPUs, resharding is in-memory, and best-of-breed engines (vLLM, FSDP, Megatron) are orchestrated unmodified.
    - **Customizing reward and advantage is plain local Python on tiny tensors** — the single-controller payoff. Changing the algorithm requires zero `torch.distributed` code, unlike a multi-controller system.
    - **Known bottlenecks:** synchronous rollout tail-latency (stragglers) and the train/inference log-prob mismatch (mitigated by recomputing `old_log_prob` under the training engine). Async and disaggregated designs address the former.

!!! sota "State of the Art & Resources (2026)"
    veRL / HybridFlow has become the dominant open-source substrate for serious RL-for-LLM work; its single-controller + multi-controller hybrid and 3D-HybridEngine are now the reference architecture, with the framework scaling past 671B models and hundreds of GPUs as of 2026.

    **Foundational work**

    - [Sheng et al., *HybridFlow: A Flexible and Efficient RLHF Framework* (2024)](https://arxiv.org/abs/2409.19256) — the EuroSys 2025 paper introducing the single/multi-controller hybrid and 3D-HybridEngine; the direct foundation of veRL.
    - [Moritz et al., *Ray: A Distributed Framework for Emerging AI Applications* (2018)](https://arxiv.org/abs/1712.05889) — the actor-model and placement-group substrate that makes veRL's single controller possible; still the canonical Ray reference.

    **Recent advances (2023–2026)**

    - [DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* (2025)](https://arxiv.org/abs/2501.12948) — the high-profile GRPO run that popularized veRL-scale critic-free RL for reasoning; demonstrated pure-RL emergent self-reflection.
    - [Yu et al., *DAPO: An Open-Source LLM Reinforcement Learning System at Scale* (2025)](https://arxiv.org/abs/2503.14476) — fully open-source GRPO variant (Decoupled Clip + Dynamic Sampling) built on veRL; achieves 50 pts on AIME 2024 with Qwen2.5-32B.
    - [Cui et al., *Process Reinforcement through Implicit Rewards* (2025)](https://arxiv.org/abs/2502.01456) — PRIME: dense process rewards from policy rollouts alone, implemented as a veRL extension; shows +15% reasoning improvement from a 7B base.
    - [Hu et al., *OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework* (2024)](https://arxiv.org/abs/2405.11143) — the main Ray-based disaggregated alternative to veRL; useful for comparing colocated vs. disaggregated design choices.

    **Open-source & tools**

    - [volcengine/verl](https://github.com/volcengine/verl) — the production veRL library: WorkerGroup, ResourcePool, DataProto, FSDP/Megatron actor backends, vLLM/SGLang rollout backends, and all supported algorithms (PPO, GRPO, DAPO, PRIME, RLOO, …).
    - [PRIME-RL/PRIME](https://github.com/PRIME-RL/PRIME) — open-source online RL with implicit process rewards; built on veRL and scales to multi-node clusters.

    **Go deeper**

    - [veRL HybridFlow Programming Guide](https://verl.readthedocs.io/en/latest/hybrid_flow.html) — official docs walking through ResourcePool, WorkerGroup, and the dispatch decorator with code examples.
    - [veRL v0.7 Release Blog](https://verl.readthedocs.io/en/latest/blog/v0.7.html) — covers the Hybrid-Controller architecture evolution, rollout-server mode, TransferQueue, and async pipeline support added in 2026.
    - [SkyPilot Blog: *How to train and scale AI math/coding agents using VeRL* (2025)](https://blog.skypilot.co/verl-rl-training/) — practical end-to-end tutorial for launching veRL RL training on any cloud or Kubernetes cluster.

## Further reading

- Sheng, Zhang, Ye, et al., **HybridFlow: A Flexible and Efficient RLHF Framework** (2024) — the paper that introduces the single-controller/multi-controller hybrid and the 3D-HybridEngine; the foundation of veRL.
- The **`volcengine/verl`** repository — the production implementation: `WorkerGroup`, `ResourcePool`, `DataProto`, the `@register` dispatch layer, FSDP/Megatron actor backends, and vLLM/SGLang rollout backends.
- Rajbhandari, Rasley, Ruwase, He, **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models** (2020) — the sharding ideas behind FSDP that veRL's training engine uses; see [Distributed Training I: Data Parallelism, DDP, ZeRO & FSDP](../03-pretraining/05-distributed-data-parallel.html).
- Shoeybi, Patwary, Puri, et al., **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism** (2019) — the tensor/pipeline parallelism that the Megatron actor backend and the resharding logic build on.
- Kwon, Li, Zhuang, et al., **Efficient Memory Management for Large Language Model Serving with PagedAttention** (vLLM, 2023) — the rollout engine veRL drives; see [vLLM: Architecture, PagedAttention & Internals](../07-inference-serving/03-vllm-internals.html).
- Moritz, Nishihara, Wang, et al., **Ray: A Distributed Framework for Emerging AI Applications** (2018) — the actor and placement-group substrate underneath veRL's single controller.
- Shao, Wang, Zhu, et al., **DeepSeekMath** (2024) and DeepSeek-AI, **DeepSeek-R1** (2025) — the GRPO algorithm most veRL runs implement; see [GRPO, RLOO & Critic-Free RL](../05-posttraining-alignment/08-grpo-rloo.html).

## Exercises

**1.** (Conceptual) A colleague proposes building an RL-for-LLM trainer as a *pure single-controller* system: one driver process issues every operation as a remote call and holds all the algorithm logic. They argue this is the most readable design. State the two distinct ways this pure design fails at scale, and then identify which *one* line of veRL's driver loop (`fit`) genuinely belongs on the single-controller side and why it is safe to keep it there.

??? note "Solution"
    The pure single-controller design fails in two complementary ways described in the chapter:

    1. **Driver bottleneck.** Everything the algorithm touches — per-token log-probs, response token ids, attention masks — is gathered to the driver, sliced, and re-scattered. For a 7B policy update with 1024 prompts x 8 samples x 2048 tokens these tensors are hundreds of MB to GB per step, so the driver becomes a serialization and network-congestion bottleneck.
    2. **Cannot express or compose intra-stage parallelism.** The driver only knows how to call "the actor"; it has no way to express the all-reduces inside the actor's backward or the all-gathers inside generation. So the parallelism must either be absent (slow) or hidden inside opaque workers, which forfeits the ability to compose the trainer's parallelism with a different rollout parallelism.

    The line that genuinely belongs on the single-controller side is the advantage computation:

    ```python
    batch = compute_advantage(batch, adv_estimator=self.config.adv_estimator)
    ```

    It is safe there because it operates only on *tiny* per-sample tensors (rewards, values, advantages) with no collectives — it is local single-threaded Python. This is precisely the part researchers change most often, and keeping it on the driver is the productivity win of HybridFlow: swapping GAE for GRPO's group baseline is a one-function edit with no distributed code. The heavy stages (`generate_sequences`, `compute_log_prob`, `update_actor`, `update_critic`) do *not* belong on the driver — they must run as multi-controller SPMD `WorkerGroup`s so their activations and gradients never cross the boundary.

**2.** (Conceptual) veRL's driver loop calls `compute_log_prob` under the *training-layout* actor immediately after generation, even though vLLM already produced per-token log-probs during rollout (step 2 recomputes what step 1 seemingly gave you). Why is this recomputation not redundant? What correctness quantity depends on it?

??? note "Solution"
    It is not redundant because vLLM's generation kernels and the trainer's forward pass are *different code paths* that compute log-probs with different numerics (different kernels, fusion, and precision handling). The log-prob vLLM used while sampling can therefore disagree with the log-prob the training engine would assign to the same token — the **train/inference numerical mismatch**.

    The quantity that depends on getting this right is the **importance-sampling ratio** in the PPO/GRPO objective, $r_t = \exp(\log \pi_{\text{new}}(a_t) - \log \pi_{\text{old}}(a_t))$. The `old_log_prob` in the denominator must be computed by the *same* engine that will later compute `new_log_prob` during the update; otherwise the ratio is systematically biased at step zero (it would not equal 1 even before any gradient step), corrupting the clipping and the gradient. By recomputing `old_log_prob` under the training engine, veRL keeps the ratio self-consistent. Note this is a *correctness* fix, not a performance one — it costs an extra forward pass every step.

**3.** (Quantitative) An actor `WorkerGroup` occupies a world of 8 ranks configured with tensor-parallel degree 2 (and no pipeline parallelism), so in the dispatch code `tp_size = 2` and `dp_size = 4`. The driver calls a method registered with `DP_COMPUTE_PROTO` on a `DataProto` batch of `B = 1024` samples. Using the chapter's `dispatch_dp_compute_proto` / `collect_dp_compute_proto` sketch: (a) how many samples does each of the 8 ranks receive? (b) which ranks' outputs does the collect function keep, and (c) how many samples are in the final reassembled `DataProto`? (d) If you had *mistakenly* concatenated all 8 ranks' outputs instead of one representative per DP group, how many samples would you get and why is that wrong?

??? note "Solution"
    (a) `DP_COMPUTE_PROTO` splits the batch across the **data-parallel** dimension only. With `dp_size = 4`, the batch is chunked into 4 pieces of $1024 / 4 = 256$ samples each. Each chunk is then *replicated* to every rank in its TP group (`tp_size = 2`). So all 8 ranks receive **256 samples** — ranks 0 and 1 get chunk 0, ranks 2 and 3 get chunk 1, ranks 4 and 5 get chunk 2, ranks 6 and 7 get chunk 3.

    (b) `collect_dp_compute_proto` keeps one representative per DP group: `outputs[dp_rank * tp_size]` for `dp_rank` in $\{0,1,2,3\}$ with `tp_size = 2`, i.e. ranks **0, 2, 4, 6**.

    (c) Concatenating those 4 representatives of 256 samples each gives $4 \times 256 = \mathbf{1024}$ samples — the original batch order is reconstructed.

    (d) Concatenating all 8 outputs would give $8 \times 256 = 2048$ samples. That is wrong because the two ranks in each TP group computed on the *identical* replicated chunk — their outputs are duplicates, not new data. Keeping both double-counts every sample, so you would train on each example twice with corrupted batch alignment.

**4.** (Quantitative) The chapter budgets a 7B GRPO run on 8xA100-80GB and finds it fits comfortably. You instead want to run **PPO with a 7B critic**, still on 8xA100-80GB, FSDP for both actor and critic, vLLM rollout at TP=2, bf16 weights, Adam. Compute the per-GPU *baseline* resident memory during the rollout stage (training state of both models, plus the rollout weight copy), and state how much is left for the vLLM KV cache and activations. Use the chapter's convention that Adam fp32 master weights + two moments cost 12 bytes/parameter.

??? note "Solution"
    Work per rank, with FSDP sharding all training state across 8 ranks.

    **Actor training state (resident all step):**

    - Parameters (bf16): $7\text{B} \times 2\,\text{B} = 14\,\text{GB}$ total $\Rightarrow 14/8 = 1.75\,\text{GB}$/rank.
    - Gradients (bf16): $1.75\,\text{GB}$/rank.
    - Adam fp32 (master + 2 moments), $12\,\text{B/param}$: $7\text{B} \times 12 = 84\,\text{GB}$ total $\Rightarrow 84/8 = 10.5\,\text{GB}$/rank.
    - Actor subtotal: $1.75 + 1.75 + 10.5 = 14\,\text{GB}$/rank.

    **Critic training state:** a 7B critic has the same shape, so another $\approx 14\,\text{GB}$/rank.

    **Rollout weights (vLLM, TP=2):** a full bf16 copy of the 7B actor split across the TP=2 group $\Rightarrow 14\,\text{GB}/2 = 7\,\text{GB}$/rank. (The critic does not generate, so it contributes no rollout copy.)

    **Baseline resident during rollout:** $14 + 14 + 7 = \mathbf{35\,\text{GB}}$/rank.

    **Left for KV cache + activations:** $80 - 35 = \mathbf{45\,\text{GB}}$/rank.

    So it still *nominally* fits, but the headroom fell from $\sim 59\,\text{GB}$ (GRPO, no critic) to $\sim 45\,\text{GB}$ — the extra $14\,\text{GB}$ of critic training state is exactly the tightening the chapter warns about. With long group-of-$G$ generations the KV cache can want tens of GB, so you would likely enable `optimizer_offload` (spilling the $2 \times 10.5 = 21\,\text{GB}$/rank of Adam state to CPU during rollout) to restore comfortable headroom. This mechanical squeeze is one reason the field prefers critic-free GRPO/RLOO for large policies.

**5.** (Implementation) The chapter's `grpo_group_advantage` uses the *group mean* of all $G$ samples as the baseline. Implement `rloo_group_advantage(rewards, group_size)` instead, using the **leave-one-out** baseline: each sample's baseline is the mean of the *other* $G-1$ samples in its group, so $A_i = r_i - \frac{1}{G-1}\sum_{j \ne i} r_j$. Keep it as pure local Python on a `(B,)` tensor in the chapter's style (fully vectorized, no Python loop over samples), and note the one edge case that must hold on `group_size`.

??? note "Solution"
    The leave-one-out sum for sample $i$ is (group total minus $r_i$), so the leave-one-out mean is $(\text{sum} - r_i)/(G-1)$. Everything vectorizes over the `(n_prompts, G)` view:

    ```python
    def rloo_group_advantage(rewards, group_size):
        """rewards: (B,) with B = n_prompts * group_size, grouped contiguously.
        RLOO leave-one-out baseline: A_i = r_i - mean_{j != i} r_j."""
        assert group_size > 1, "RLOO needs G > 1 (divide-by-(G-1))"
        g = rewards.view(-1, group_size)                 # (n_prompts, G)
        group_sum = g.sum(dim=1, keepdim=True)           # (n_prompts, 1)
        loo_mean = (group_sum - g) / (group_size - 1)    # each sample excluded
        adv = g - loo_mean                               # (n_prompts, G)
        return adv.reshape(-1)                           # one advantage / sample
    ```

    The load-bearing edge case is **`group_size > 1`**: the denominator is $G-1$, so a group size of 1 would divide by zero (and is meaningless anyway — you cannot form a leave-one-out baseline from a single sample). Like `grpo_group_advantage`, this runs on the *driver* over a tiny `(B,)` tensor with zero distributed code; a multi-controller system would need explicit cross-rank all-gathers to form the same group reductions.

**6.** (Implementation) The chapter's `reshard_column_parallel` reshards a *column*-parallel weight (split along the output dim, `dim=1`). Implement the symmetric `reshard_row_parallel` for a **row-parallel** weight (e.g. an MLP down-projection, split along the *input* dim, `dim=0`), resharding from `train_tp_size` shards to `rollout_tp_size` shards in GPU memory. Keep the same three-step structure and the same rank-to-rollout-shard mapping, and state which dimension changes versus the column-parallel case.

??? note "Solution"
    The only change is the axis: a row-parallel weight is split along the **input** dimension (`dim=0`), so both the all-gather reconstruction and the re-split happen along `dim=0` instead of `dim=1`. The gather-within-a-small-TP-group then re-split logic, and the `rank_in_group < q` ownership mapping, are identical.

    ```python
    import torch
    import torch.distributed as dist

    def reshard_row_parallel(local_shard: torch.Tensor,
                             train_tp_group, train_tp_size: int,
                             rollout_tp_size: int, rank_in_group: int):
        """
        Reshard one ROW-parallel weight (split along the INPUT dim, dim=0) from
        train_tp_size shards to rollout_tp_size shards, IN GPU MEMORY.

        local_shard : this rank's slice, shape (in_dim/p, out_dim).
        Returns this rank's rollout slice, shape (in_dim/q, out_dim), or None if
        this rank is not used by the rollout engine (q < p case).
        """
        p, q = train_tp_size, rollout_tp_size

        # --- Step 1: gather the p row-shards into the full input dimension. ---
        gathered = [torch.empty_like(local_shard) for _ in range(p)]
        dist.all_gather(gathered, local_shard, group=train_tp_group)
        full_weight = torch.cat(gathered, dim=0)         # (in_dim, out_dim) full

        # --- Step 2: re-split into q rollout shards along the SAME (input) dim. ---
        in_dim = full_weight.shape[0]
        assert in_dim % q == 0, "input dim must be divisible by rollout TP degree"
        rollout_shards = full_weight.chunk(q, dim=0)     # list of q tensors

        # --- Step 3: which rollout shard does THIS physical GPU own? ---
        if rank_in_group < q:
            return rollout_shards[rank_in_group].contiguous()  # (in_dim/q, out_dim)
        else:
            return None                                  # not a rollout rank
    ```

    Versus the column-parallel case, the changed dimension is `dim=1 -> dim=0` in both the `torch.cat` (step 1) and the `chunk` (step 2), and the divisibility assertion is now on the input dim rather than the output dim. As the chapter notes, attention QKV/`o_proj` weights need additional *head-aware* regrouping so whole heads stay intact after re-sharding, which neither the pure column nor pure row routine handles on its own.
