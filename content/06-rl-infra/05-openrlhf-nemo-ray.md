# 6.5 OpenRLHF, NeMo-Aligner & Ray-Based Systems

Reinforcement learning from human feedback (RLHF) for large language models is not just an algorithmic problem — it is primarily a distributed systems engineering problem. When the policy, critic, and reward model together exceed the memory of a single GPU node, when rollout throughput must match gradient-update throughput, and when checkpoint synchronization needs to happen dozens of times per hour across hundreds of GPUs, the implementation framework becomes the bottleneck, not the math.

This chapter examines two mature, production-grade open-source frameworks — **OpenRLHF** and **NVIDIA NeMo-Aligner** — and the **Ray** distributed actor system that underpins them both (in different ways). By the end of this chapter you will understand how each system assigns roles to processes, how they move tensors between actors, where their design choices diverge, and how to make the right choice for your cluster and model size.

We assume familiarity with the PPO objective (see [Policy Gradients & PPO for Language Models](../05-posttraining-alignment/06-ppo-for-llms.html)) and with vLLM's PagedAttention engine (see [vLLM: Architecture, PagedAttention & Internals](../07-inference-serving/03-vllm-internals.html)). We also assume you have seen the four-role decomposition of an RL-for-LLM system in [The Anatomy of an RL-for-LLM System](../06-rl-infra/01-anatomy-rl-system.html).

---

## 6.5.1 Ray as an Orchestration Substrate

Before studying OpenRLHF or NeMo-Aligner, we need to understand **Ray** — the distributed computing framework developed at UC Berkeley and now maintained by Anyscale — because it is the runtime that ties everything together in the OpenRLHF design.

### Ray's Programming Model

Ray exposes three primitives:

1. **Remote functions** (`@ray.remote` decorated): stateless tasks that execute on any available worker.
2. **Actors** (`@ray.remote` decorated classes): stateful processes with their own GPU, CPU, and memory allocations, addressable by handle.
3. **Object store**: a per-node shared-memory store (originally Arrow's Plasma, since Ray 1.x an in-tree implementation). Objects put into the store are zero-copy readable by any process on the same node.

The key insight for RL training is that **actors map perfectly onto the roles in RLHF**: the rollout engine is one actor (or a group), the critic is another, the reference model is a third. Each actor owns its model shards and optimizer state. The controller — typically a small CPU process — choreographs them by submitting tasks and awaiting futures.

```python
import ray
import torch

# --- Minimal Ray actor example: a stateful GPU model holder ---
@ray.remote(num_gpus=1)
class ModelActor:
    """
    A simple Ray actor that holds a model on a single GPU.
    In a real RLHF system, this would be a policy, critic, or RM.
    """
    def __init__(self, hidden_size: int):
        self.device = torch.device("cuda:0")
        # Tiny toy model; in practice this is a multi-billion-parameter LLM
        self.model = torch.nn.Linear(hidden_size, hidden_size).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

    def forward(self, x_ref):
        """Receive a Ray object reference, run forward pass, return a new ref."""
        x = ray.get(x_ref)           # deserialize tensor from object store
        x = x.to(self.device)
        with torch.no_grad():
            out = self.model(x)
        return ray.put(out.cpu())    # put result back into object store

    def update(self, loss_ref):
        """Apply a gradient update given a remote loss tensor."""
        loss = ray.get(loss_ref).to(self.device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return "updated"

# --- Controller (runs on CPU driver) ---
ray.init()
actor = ModelActor.remote(hidden_size=256)

x = ray.put(torch.randn(8, 256))   # put input into object store
out_ref = actor.forward.remote(x)  # async dispatch; returns a future
out = ray.get(out_ref)             # block until done
print(out.shape)                   # torch.Size([8, 256])
ray.shutdown()
```

### Placement Groups and Gang Scheduling

When a single actor spans multiple GPUs (e.g., a 70B policy sharded with tensor parallelism across 8 GPUs), Ray uses **placement groups** to reserve a "bundle" of resources that are co-located and launched atomically. This prevents the partial-allocation deadlock that plagues naive multi-GPU resource requests.

```python
import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

ray.init()

# Reserve 8 GPUs on the same node for a tensor-parallel actor group
pg = placement_group(
    bundles=[{"GPU": 1, "CPU": 4} for _ in range(8)],
    strategy="STRICT_PACK",   # all 8 bundles MUST land on one node
)
ray.get(pg.ready())           # wait for allocation

@ray.remote(num_gpus=1)
class TPShard:
    """One shard of a tensor-parallel model."""
    def __init__(self, rank: int, world_size: int):
        self.rank, self.world_size = rank, world_size

    def rendezvous_endpoint(self):
        """Rank 0 publishes an address the other ranks can dial."""
        import socket
        import ray
        s = socket.socket()
        s.bind(("", 0))                     # ask the OS for a free port
        port = s.getsockname()[1]
        s.close()
        return ray.util.get_node_ip_address(), port

    def init_pg(self, master_addr: str, master_port: int):
        """Form the NCCL communicator. Ray sets no launcher env vars, so
        `init_method="env://"` would hang here — we must pass the address
        explicitly. This rendezvous-then-init split is exactly what
        OpenRLHF and vLLM do when wiring GPU groups under Ray."""
        import torch.distributed as dist
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            rank=self.rank,
            world_size=self.world_size,
        )
        return f"shard {self.rank} alive"

shards = [
    TPShard.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=i,
        )
    ).remote(rank=i, world_size=8)
    for i in range(8)
]
addr, port = ray.get(shards[0].rendezvous_endpoint.remote())
# All 8 must call init_process_group concurrently — it is a collective.
print(ray.get([s.init_pg.remote(addr, port) for s in shards]))
# ['shard 0 alive', 'shard 1 alive', ..., 'shard 7 alive']
ray.shutdown()
```

### Object Store and Zero-Copy Tensors

A critical performance property of Ray's object store is **zero-copy reads for numpy arrays on the same node**. When rollout tokens and log-probabilities are placed in the object store, a training worker on that node maps the shared-memory pages directly instead of copying. The caveat, and the source of a warning message every RLHF engineer eventually sees: the mapped buffer is **read-only**, so `torch.from_numpy(ray.get(ref))` yields a tensor you must `.clone()` before writing to (PyTorch emits "The given NumPy array is not writable" otherwise). Torch tensors themselves round-trip through numpy in Ray's serializer, and CUDA tensors are always copied to host memory first — so keep experience batches on CPU as numpy, and never route GPU tensors through the object store. For a batch of 1024 sequences of length 2048, the per-token logprob array is approximately:

$$
1024 \times 2048 \times 4 \text{ bytes} \approx 8 \text{ GB}
$$

Eliminating even one copy of that tensor per step saves gigabytes of PCIe or NVLink bandwidth per iteration.

Across nodes there is no such trick: Ray serializes the object (cloudpickle, with numpy buffers handled out-of-band) and copies it over the network, so cross-node object transfers should be minimized — keep rollout workers and the training workers that consume their output on the same node when possible.

---

## 6.5.2 OpenRLHF: Architecture and Design Philosophy

**OpenRLHF** (originally open-sourced in late 2023 by the OpenLLMAI community) is built around a clean separation of concerns: every RLHF role — **policy actor**, **reference actor**, **critic**, and **reward model** — lives in its own Ray actor group. The rollout engine is vLLM. Gradient updates use DeepSpeed ZeRO (see [Distributed Training I: Data Parallelism, DDP, ZeRO & FSDP](../03-pretraining/05-distributed-data-parallel.html)).

### Role Decomposition


{{fig:openrlhf-ray-role-decomposition}}


The key design insight: **the vLLM engine holds a *read-only copy* of policy weights for generation**. After each PPO update, the policy training actor must push its updated weights to the vLLM engine. This weight-sync step is where OpenRLHF differs from single-process approaches.

### Weight Synchronization: the Object-Store Path and the NCCL Path

There are two ways to get updated parameters from the training actor into the vLLM engine, and the difference between them is the single most important performance decision in a Ray-based RLHF system.

**Path A — through the Ray object store.** The training actor gathers its ZeRO-3 shards, moves them to CPU, `ray.put()`s them, and the vLLM workers `load_weights()` from the materialized dict. Simple, backend-agnostic, works across nodes with no extra setup — but every byte makes a GPU→CPU→(serialize)→CPU→GPU round trip. OpenRLHF's `--vllm_sync_backend gloo` is the CPU-collective cousin of this: same host-memory detour, without the serialization.

**Path B — direct NCCL broadcast (`--vllm_sync_backend nccl`, the default in practice).** At startup, the training ranks and the vLLM workers join a *second, dedicated* process group. At sync time the training rank broadcasts each tensor GPU-to-GPU; the vLLM worker receives into a pre-allocated buffer and calls `load_weights` on it. Nothing touches CPU or the object store. Ray is used only to *name* the participants and to invoke the RPC — the bytes travel over NVLink/InfiniBand.

Path B is what every serious framework does today. Modern vLLM exposes exactly the hooks needed for it: you attach a **worker extension class** to the engine and drive it with **`collective_rpc`**, which runs a named method on *every* vLLM worker rank simultaneously (see vLLM's `examples/offline_inference/rlhf.py`, the reference implementation that OpenRLHF, veRL and NeMo-RL all mirror).

```python
# --- vLLM side: a worker extension, loaded into every vLLM worker rank ---
# (saved as e.g. rlhf_utils.py so vLLM can import it by string name)
import torch
from vllm.distributed.utils import stateless_init_process_group


class WeightSyncExtension:
    """Methods here become callable on every vLLM worker via collective_rpc.
    `self` is the vLLM Worker, so `self.model_runner.model` is the live model."""

    def init_weight_update_group(self, master_addr, master_port,
                                 rank_offset, world_size):
        # vLLM worker i takes global rank `rank_offset + i`; the trainer holds
        # rank 0. This is a SEPARATE process group from vLLM's own TP group.
        my_rank = torch.distributed.get_rank() + rank_offset
        self.weight_update_group = stateless_init_process_group(
            master_addr, master_port, my_rank, world_size,
            torch.device(f"cuda:{torch.cuda.current_device()}"),
        )

    def update_weight(self, name, dtype, shape):
        """Receive ONE parameter by broadcast and load it in place."""
        buf = torch.empty(shape, dtype=dtype, device="cuda")
        self.weight_update_group.broadcast(buf, src=0, stream=torch.cuda.current_stream())
        self.model_runner.model.load_weights(weights=[(name, buf)])
        del buf   # vLLM copied it into the model's own storage


# --- Driver side: the engine must live in a Ray actor, not in this process ---
import ray
from vllm import LLM


@ray.remote(num_gpus=0)   # the engine's own workers claim the GPUs
class VLLMEngineActor:
    def __init__(self):
        self.llm = LLM(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            tensor_parallel_size=2,
            distributed_executor_backend="ray",
            enforce_eager=True,                 # weights change every step
            worker_extension_cls="rlhf_utils.WeightSyncExtension",
        )

    def rpc(self, method, *args):
        return self.llm.collective_rpc(method, args=args)

    def reset_prefix_cache(self):
        return self.llm.reset_prefix_cache()


engine = VLLMEngineActor.remote()

# Once, at startup: trainer is rank 0, the 2 vLLM workers are ranks 1 and 2.
ray.get(engine.rpc.remote("init_weight_update_group",
                          master_addr, master_port, 1, 1 + 2))
# ...and the trainer joins the SAME group as rank 0 (world_size=3), symmetrically.

def push_weights(model, engine, weight_update_group):
    """Broadcast every parameter, one at a time, trainer -> vLLM workers."""
    for name, p in model.named_parameters():
        # Fire the RPC WITHOUT waiting: the workers must be sitting in their
        # matching `broadcast` when we call ours, or both sides deadlock.
        # This is why the engine has to be a Ray actor — a plain in-process
        # `LLM.collective_rpc` blocks, and the broadcast below never happens.
        handle = engine.rpc.remote("update_weight", name, p.dtype, p.shape)
        torch.distributed.broadcast(p.data, src=0, group=weight_update_group)
        ray.get(handle)                        # now the recv is complete
    ray.get(engine.reset_prefix_cache.remote())  # old-weight prefixes must go
```

Three details that bite people:

1. **Parameters must be un-sharded before broadcast.** Under ZeRO-3 (or FSDP) each rank holds a slice. The trainer wraps the loop in DeepSpeed's `GatheredParameters` (or FSDP's `summon_full_params`) so that rank 0 has the full tensor, one parameter at a time — which is why sync is a *parameter-by-parameter* loop and not one giant broadcast, and why peak extra memory is only the size of the largest tensor (usually the embedding matrix).
2. **The KV cache is fine; the prefix cache is not.** Blocks holding KV for in-flight requests are dropped when generation finishes, so nothing stale survives. But vLLM's *automatic prefix caching* persists KV across requests, and those blocks were computed with the old weights — reusing them silently mixes two policies into one rollout. Reset it after every sync.
3. **`enforce_eager=True`** (or careful use of CUDA graphs) — captured graphs bake in weight pointers; in-place `load_weights` into the same storage is safe, but reallocating is not.

For reference, here is the same two-actor skeleton at the level of Ray plumbing, using the simpler object-store path:

```python
# Simplified weight-sync routine, object-store path (Path A)
import ray
import torch
from typing import Dict

@ray.remote(num_gpus=1)
class PolicyTrainingActor:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def get_named_params(self) -> Dict[str, torch.Tensor]:
        """Return state dict on CPU for broadcasting."""
        return {k: v.cpu() for k, v in self.model.named_parameters()}

    def ppo_step(self, batch):
        """Run one PPO gradient update."""
        # ... (compute loss, backward, step) ...
        pass


@ray.remote(num_gpus=1)
class VLLMRolloutActor:
    def __init__(self, vllm_engine):
        self.engine = vllm_engine

    def load_weights(self, state_dict):
        """Receive new policy weights and reload into vLLM.

        Note: `state_dict` arrives already materialized. Ray auto-dereferences
        top-level ObjectRef arguments passed to a `.remote()` call before the
        method body runs, so no additional `ray.get()` is needed (or valid)
        here — calling `ray.get()` on an already-resolved dict raises a
        ValueError.
        """
        # Reach the live model on every worker rank. Do NOT poke at
        # `llm_engine.model_executor.driver_worker` — that path only existed in
        # vLLM's legacy V0 engine, where the driver ran in-process. Since V1 the
        # workers are separate processes and `collective_rpc` is the only
        # supported way in.
        self.engine.collective_rpc("load_weights_from_dict", args=(state_dict,))
        self.engine.reset_prefix_cache()
        return "weights_loaded"

    def rollout(self, prompts, sampling_params):
        """Generate responses using updated weights."""
        from vllm import SamplingParams
        outputs = self.engine.generate(prompts, SamplingParams(**sampling_params))
        return [(o.prompt, o.outputs[0].text, o.outputs[0].logprobs)
                for o in outputs]
```


{{fig:weight-sync-two-copies-bandwidth}}


### The PPO Training Loop in OpenRLHF

Here is the high-level controller logic. Notice how it is a plain Python loop — no custom distributed runtime, just Ray futures:

```python
import ray
from dataclasses import dataclass
from typing import List

@dataclass
class ExperienceBatch:
    prompts: List[str]
    responses: List[str]
    # All four below are per-token, ragged: outer list = batch, inner = time.
    rewards: List[List[float]]
    advantages: List[List[float]]
    old_logprobs: List[List[float]]   # from vLLM at generation time
    ref_logprobs: List[List[float]]   # from reference model


def run_ppo_training(
    policy_actor,
    vllm_actor,
    ref_actor,
    reward_actor,
    critic_actor,
    prompt_dataset,
    num_epochs: int = 1,
    rollout_batch_size: int = 256,
    ppo_epochs: int = 4,
    gamma: float = 1.0,
    lam: float = 0.95,
):
    """
    High-level PPO controller. Runs entirely on CPU driver process.
    Each .remote() call is non-blocking; ray.get() blocks until done.
    """
    for epoch in range(num_epochs):
        for prompt_batch in prompt_dataset.batches(rollout_batch_size):

            # --- Phase 1: Generate rollouts ---
            rollout_ref = vllm_actor.rollout.remote(
                prompt_batch, {"temperature": 1.0, "max_tokens": 512}
            )

            # --- Phase 2: Score with RM (can overlap with rollout) ---
            # (actually waits on rollout_ref internally)
            reward_ref = reward_actor.score.remote(rollout_ref)

            # --- Phase 3: Compute reference log-probs ---
            ref_logp_ref = ref_actor.log_probs.remote(rollout_ref)

            # --- Phase 4: Estimate values and advantages ---
            value_ref = critic_actor.value.remote(rollout_ref)

            # Gather everything; build experience batch
            rollouts, rewards, ref_logps, values = ray.get(
                [rollout_ref, reward_ref, ref_logp_ref, value_ref]
            )
            batch = build_experience_batch(
                rollouts, rewards, ref_logps, values, gamma, lam
            )
            batch_ref = ray.put(batch)  # into shared object store

            # --- Phase 5: PPO gradient updates (multiple epochs) ---
            for _ in range(ppo_epochs):
                policy_loss_ref = policy_actor.ppo_step.remote(batch_ref)
                critic_loss_ref = critic_actor.update.remote(batch_ref)
                ray.get([policy_loss_ref, critic_loss_ref])

            # --- Phase 6: Sync updated weights to vLLM ---
            params_ref = policy_actor.get_named_params.remote()
            ray.get(vllm_actor.load_weights.remote(params_ref))


def build_experience_batch(rollouts, rewards, ref_logps, values, gamma, lam):
    """
    Compute GAE advantages (lambda-return) and pack into ExperienceBatch.
    See: Schulman et al. 'High-Dimensional Continuous Control Using
         Generalized Advantage Estimation', 2015.

    Shapes: rewards[i] and values[i] are per-TOKEN sequences of length T_i for
    rollout i (the RM score lands on the final token, the KL penalty on every
    token). GAE recurses along the time axis *inside* one sequence and never
    across the batch: separate rollouts are independent episodes, so
    bootstrapping sequence i+1's value into sequence i is simply wrong —
    a classic bug when a batch tensor is flattened before this loop.
    """
    advantages = []
    for r_seq, v_seq in zip(rewards, values):
        T = len(r_seq)
        adv_seq = [0.0] * T
        last_gae = 0.0
        for t in reversed(range(T)):
            # Past EOS there is no successor state, so V(s_{T}) = 0.
            next_val = v_seq[t + 1] if t + 1 < T else 0.0
            delta = r_seq[t] + gamma * next_val - v_seq[t]
            last_gae = delta + gamma * lam * last_gae
            adv_seq[t] = last_gae
        advantages.append(adv_seq)
    return ExperienceBatch(
        prompts=[r[0] for r in rollouts],
        responses=[r[1] for r in rollouts],
        rewards=rewards,
        advantages=advantages,
        old_logprobs=[r[2] for r in rollouts],
        ref_logprobs=ref_logps,
    )
```

### Practical Configuration: OpenRLHF Launch Script

OpenRLHF is not a library you import — it is a set of CLI entrypoints under `openrlhf.cli`, launched as a **Ray job** so that the driver itself runs inside the cluster. The distinctive feature of the interface is that *GPU allocation per role is an explicit command-line argument*: you literally spell out how many nodes and GPUs the actor, critic, reference and reward models each get.

```bash
#!/bin/bash
# Launch OpenRLHF PPO training on 4 nodes, 8 GPUs each (32 GPUs total)
# Policy: LLaMA-3-70B  |  RM: LLaMA-3-8B  |  Rollout: vLLM
#
# GPU accounting (this MUST sum to <= 32, and it is easy to get wrong):
#   actor  2 nodes x 8 = 16   <- reference COLOCATED here, +0 GPUs
#   critic 1 node  x 8 =  8   <- reward model COLOCATED here, +0 GPUs
#   vLLM   2 engines x TP=4 = 8
#   total = 32
# Colocated roles must declare the SAME node/GPU counts as their host role,
# which is why --ref_* mirrors --actor_* and --reward_* mirrors --critic_*.

pip install openrlhf[vllm]

# Start Ray cluster (typically done via Kubernetes/KubeRay or a Slurm prolog)
ray start --head --num-gpus=8 --num-cpus=64
# On worker nodes:  ray start --address=<head_ip>:6379 --num-gpus=8

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json='{"working_dir": "."}' \
  -- python3 -m openrlhf.cli.train_ppo_ray \
  --pretrain meta-llama/Meta-Llama-3-70B-Instruct \
  --reward_pretrain meta-llama/Meta-Llama-3-8B \
  --save_path ./checkpoints/llama3-70b-ppo \
  --prompt_data OpenRLHF/prompt-collection-v0.1 \
  --input_key context_messages --apply_chat_template \
  --num_episodes 1 \
  --prompt_max_len 1024 --generate_max_len 1024 \
  --rollout_batch_size 1024 \
  --micro_rollout_batch_size 8 \
  --train_batch_size 128 \
  --micro_train_batch_size 4 \
  --vllm_num_engines 2 \
  --vllm_tensor_parallel_size 4 \
  --vllm_sync_backend nccl \
  --actor_num_nodes 2 --actor_num_gpus_per_node 8 \
  --critic_num_nodes 1 --critic_num_gpus_per_node 8 \
  --ref_num_nodes 2   --ref_num_gpus_per_node 8 \
  --reward_num_nodes 1 --reward_num_gpus_per_node 8 \
  --colocate_actor_ref --colocate_critic_reward \
  --zero_stage 3 \
  --bf16 --flash_attn --gradient_checkpointing \
  --actor_learning_rate 5e-7 \
  --critic_learning_rate 9e-6 \
  --init_kl_coef 0.01 \
  --normalize_reward \
  --adam_offload      # offload optimizer state to CPU to save GPU memory
```

Two switches on this command line are worth more than the rest combined:

- **`--colocate_actor_ref` / `--colocate_critic_reward` / `--colocate_all_models`.** These place two roles in the *same* placement-group bundles, time-slicing the GPUs rather than dedicating separate ones. The reference model is idle except for one forward pass per batch, so colocating it with the actor typically buys you a whole node back at a few percent throughput cost. `--colocate_all_models` goes further and puts the vLLM engines on the training GPUs too, offloading each model's weights while the other runs — the "colocated" regime analysed in [Colocated vs Disaggregated RL & Weight Synchronization](../06-rl-infra/07-colocated-vs-disaggregated.html).
- **`--advantage_estimator`.** The default `gae` is the critic-based PPO described above. Setting it to `group_norm` switches to **GRPO**-style group-relative advantages, and `reinforce_baseline` to REINFORCE++ — both of which are *critic-free*. That is not a small algorithmic knob: it deletes an entire actor group from the topology. In the 32-GPU layout above, dropping `--critic_num_nodes` frees 8 GPUs and removes the critic's parameters, gradients and optimizer state from the memory budget. Since 2025 most reasoning-RL runs take this path (see [RL with Verifiable Rewards (RLVR) & The Reasoning Recipe](../05-posttraining-alignment/09-rlvr-reasoning.html)), which is why the "four-role" picture is increasingly a *three*-role picture.

---

## 6.5.3 Memory Budget Analysis for OpenRLHF

Understanding how OpenRLHF allocates memory across its four actor groups is essential for capacity planning.

!!! example "Worked Memory Example: LLaMA-3-70B PPO on 32 GPUs"

    **Model size:** 70B parameters at bf16 = $70 \times 10^9 \times 2$ bytes = 140 GB.

    **Policy training actor (16 GPUs, ZeRO-3):**
    - Model parameters (bf16, sharded): $140 / 16 = 8.75$ GB per GPU
    - Gradients (bf16, also sharded by ZeRO-3): $8.75$ GB per GPU
    - Optimizer state (Adam $m$ and $v$ in fp32, plus the fp32 master copy of the weights): $70\times10^9 \times (4+4+4) = 840$ GB total, i.e. $840/16 = 52.5$ GB per GPU. This single term is larger than everything else combined and *does not fit* alongside the rest on an 80 GB card — which is exactly why `--adam_offload` (DeepSpeed's CPU Adam) is not optional at this scale. Offloaded, the GPU cost of this term is ~0.
    - Activations (per micro-batch=8, len=2048): roughly $2-4$ GB per GPU with gradient checkpointing
    - **Total per GPU (with CPU Adam offload):** $8.75 + 8.75 + \sim3 \approx 20$–$22$ GB on an 80 GB A100/H100. Comfortable — but note the host now needs 840 GB of pinned CPU RAM across the 2 nodes, and every step pays a PCIe round trip for the optimizer update.

    **vLLM rollout engine (8 GPUs, 2 engines × TP=4):**
    - Each engine holds a **full copy** of the weights, sharded only across its own TP group — not $1/n$ of them. So the per-GPU weight cost is $140 / \text{TP}$, independent of how many engines you run. This is the single most common sizing error: with TP=2 the figure would be 70 GB per GPU, leaving nothing for KV cache on an 80 GB card. TP=4 gives $140/4 = 35$ GB per GPU and fits.
    - KV cache (remaining ~40 GB per GPU after weights and workspace): LLaMA-3-70B uses GQA with 8 KV heads × 128 dims over 80 layers, so one token of KV costs $2 \times 80 \times 8 \times 128 \times 2 = 0.33$ MB across the whole model, or ~82 KB per GPU at TP=4. That is roughly $40\,\text{GB} / 82\,\text{KB} \approx 490$k tokens in flight per engine — a couple of hundred concurrent 2k-token rollouts, enough to keep generation throughput-bound rather than concurrency-starved.

    **Reference actor (4 GPUs, ZeRO-3, no optimizer, no gradients):**
    - Parameters only: $140 / 4 = 35$ GB per GPU. Frozen, so no gradient or optimizer storage — this is the cheapest role per parameter, and the natural candidate for `--colocate_actor_ref`.

    **Reward model (2 GPUs, 8B params at bf16):**
    - $8 \times 10^9 \times 2 / 2 = 8$ GB per GPU. Very comfortable.

    **Roll-up:** 16 (policy) + 8 (vLLM) + 4 (reference) + 2 (RM) = **30 of the 32 GPUs**, holding roughly $16 \times 21 + 8 \times 35 + 4 \times 35 + 2 \times 8 \approx 770$ GB of *resident* state out of the cluster's 2560 GB. The remaining ~1.8 TB is not slack to celebrate — it is KV-cache and activation headroom, and a well-tuned run consumes most of it. Note also what is missing from this budget: the **critic**, which under PPO is a second 70B model with its own gradients and optimizer state and would need another 16 GPUs. On 32 GPUs a 70B PPO run with a 70B critic does not fit; you either shrink the critic, or switch to `--advantage_estimator group_norm` (GRPO) and delete it.

---

## 6.5.4 NeMo-Aligner: Megatron-Based RLHF

**NVIDIA NeMo-Aligner** (released in 2023 as part of the NeMo framework) takes a fundamentally different approach: instead of Ray, it uses **Megatron-LM's** native 3D parallelism and **NCCL** collective communication as the backbone. TensorRT-LLM (TRT-LLM) serves as the rollout engine rather than vLLM. We study it here as the canonical Megatron-native design point; note that NVIDIA archived NeMo-Aligner in late 2025 and re-architected it into **NeMo-RL**, which keeps the Megatron Core kernels but adds Ray scheduling and Hugging Face integration (the design lessons below carry over directly).

### Megatron-LM as the Foundation

Recall from [Megatron-LM, DeepSpeed & Parallelism in Practice](../03-pretraining/07-megatron-deepspeed.html) that Megatron-LM implements:
- **Tensor parallelism (TP):** column and row linear splits within a transformer layer.
- **Pipeline parallelism (PP):** layer-level splits across devices with micro-batch pipelining.
- **Data parallelism (DP):** replicated shards with ZeRO-style optimizer partitioning.

NeMo-Aligner leverages all three for both the policy and critic. The communication topology is known at initialization time and uses only NCCL — there is no dynamic task graph as in Ray. This provides lower latency for gradient synchronization but less flexibility for heterogeneous role layouts.


{{fig:nemo-aligner-megatron-3d-layout}}


### The Reward Model and Critic in NeMo-Aligner

In NeMo-Aligner's PPO implementation, the critic shares the same Megatron parallelism configuration as the policy (same TP, PP, DP degrees). The reward model can be a separate Megatron model or an external scoring function. The key difference from OpenRLHF: **all four components communicate via NCCL point-to-point sends**, not via Ray's object store.

```python
# NeMo-Aligner PPO trainer (conceptual — simplified from actual source)
# Actual code lives in nemo_aligner/algorithms/ppo.py

from nemo_aligner.utils.train_utils import clip_gradients
from nemo_aligner.utils.distributed import masked_mean
import torch

class MegatronPPOTrainer:
    """
    PPO trainer built on Megatron-LM. All communication uses NCCL.
    Policy, critic, RM are all Megatron GPTModel instances.
    """
    def __init__(self, policy, critic, rm, ref_policy, cfg):
        self.policy = policy      # MegatronGPTModel
        self.critic = critic      # MegatronGPTModel with value head
        self.rm = rm              # frozen reward model
        self.ref = ref_policy     # frozen reference policy
        self.cfg = cfg

    @torch.no_grad()
    def compute_rewards_and_advantages(self, rollout_batch):
        """
        Given generated sequences, compute per-token advantages.
        Runs on the same Megatron process group — no Ray involved.
        """
        # rm_scores: [batch, 1] scalar reward per sequence
        rm_scores = self.rm.infer(rollout_batch["tokens"])

        # values: [batch, seq_len] value estimates
        values = self.critic.infer(rollout_batch["tokens"])

        # ref_logprobs: [batch, seq_len] for KL penalty
        ref_logprobs = self.ref.log_probs(rollout_batch["tokens"])

        # KL divergence penalty (token-level)
        kl_penalty = (
            rollout_batch["logprobs"] - ref_logprobs
        ).clamp(min=-10, max=10)

        # Combine: reward = RM score (at EOS) - kl_coef * KL
        rewards = -self.cfg.kl_coef * kl_penalty
        rewards[:, -1] += rm_scores.squeeze(-1)  # add RM reward at last token

        # GAE advantage estimation
        advantages = self._gae(rewards, values)
        return advantages, values

    def _gae(self, rewards, values, gamma=1.0, lam=0.95):
        """
        Generalized Advantage Estimation (Schulman et al., 2015).
        rewards, values: [batch, seq_len]
        """
        B, T = rewards.shape
        adv = torch.zeros_like(rewards)
        last_gae = torch.zeros(B, device=rewards.device)
        for t in reversed(range(T)):
            next_val = values[:, t + 1] if t + 1 < T else torch.zeros(B, device=values.device)
            delta = rewards[:, t] + gamma * next_val - values[:, t]
            last_gae = delta + gamma * lam * last_gae
            adv[:, t] = last_gae
        return adv

    def ppo_policy_loss(self, rollout_batch, advantages):
        """
        Clipped PPO objective with value-function loss.
        L^CLIP(θ) = E[min(r_t A_t, clip(r_t, 1-ε, 1+ε) A_t)]
        """
        logprobs = self.policy.log_probs(rollout_batch["tokens"])
        old_logprobs = rollout_batch["logprobs"]
        mask = rollout_batch["response_mask"]  # 1 for response tokens only

        # Importance-sampling ratio
        ratio = torch.exp(logprobs - old_logprobs)

        eps = self.cfg.cliprange   # typically 0.2
        clipped_ratio = ratio.clamp(1 - eps, 1 + eps)

        # Clipped surrogate loss (negative because we maximize)
        policy_loss = -masked_mean(
            torch.min(ratio * advantages, clipped_ratio * advantages),
            mask
        )
        return policy_loss
```

### TRT-LLM Rollouts in NeMo-Aligner

NeMo-Aligner uses **TensorRT-LLM** for the generation phase rather than vLLM. TRT-LLM compiles the model into a highly optimized TensorRT engine with INT8/INT4 weight quantization and fused kernels. This can yield significantly higher throughput than vLLM for fixed batch shapes — an important property during RLHF where rollout prompts are typically drawn from a fixed distribution.

The trade-off: TRT-LLM engines must be recompiled (or use dynamic shapes carefully) if the model architecture changes. Reloading weights into a compiled TRT engine after a gradient update involves an "engine reload" API call that is more expensive than vLLM's in-place `load_weights` call.

```bash
# Convert a NeMo policy checkpoint to TRT-LLM for rollout
# (Runs before training or after each N update steps)
python scripts/nemo_aligner/convert_nemo_to_trtllm.py \
  --nemo_checkpoint /checkpoints/policy/step_0 \
  --output_dir /tmp/trtllm_engine \
  --dtype bfloat16 \
  --tp_size 4 \
  --max_batch_size 64 \
  --max_input_len 1024 \
  --max_output_len 512
```

---

## 6.5.5 Design Comparison: OpenRLHF vs. NeMo-Aligner

The two frameworks represent different points in a fundamental design space. Let us make this concrete.

| Dimension | OpenRLHF | NeMo-Aligner |
|-----------|----------|--------------|
| Orchestration | Ray actors + Python controller | Megatron-LM 3D parallelism + NCCL |
| Rollout engine | vLLM (PagedAttention) | TRT-LLM (compiled TensorRT engine) |
| Gradient backend | DeepSpeed ZeRO-1/2/3 | Megatron-LM native (with ZeRO-1 optional) |
| Weight sync method | NCCL broadcast over a side process group via vLLM `collective_rpc` (object store as fallback) | NCCL broadcast between Megatron process groups |
| Flexibility | High (swap any actor, add roles) | Lower (fixed Megatron topology) |
| Communication overhead | Higher (Python coordination + serialization) | Lower (direct NCCL, pinned buffers) |
| Multi-model (actor ≠ critic model) | First-class support | Supported but same parallelism config |
| Cluster management | Ray cluster (Kubernetes/Slurm) | Slurm + MPI launch |
| Best fit | Research, heterogeneous models, rapid iteration | Production, large homogeneous clusters, throughput-optimized |

### The Fundamental Tension: Flexibility vs. Communication Efficiency

The core trade-off is between **dynamic task graphs** (Ray) and **static SPMD execution** (Megatron). Ray's actor model allows you to assign different GPU counts to each role — say, 16 GPUs for the policy, 4 for the critic, 2 for the RM — with no requirement that they share an NCCL communicator. This makes it easy to balance compute across heterogeneous models. The downside: every tensor that crosses actor boundaries must be serialized (even via shared memory), and the Python controller becomes a latency bottleneck for fine-grained coordination.

Megatron's static NCCL topology amortizes this: all processes are launched together, all-reduce and send-recv operations are pre-planned, and there is no Python GIL involvement during the hot path. But every role must conform to the same parallelism configuration, which is restrictive when the policy and critic are different model sizes.

### Where veRL Sits

For completeness: **veRL** (Volcano Engine RL, covered in [veRL: HybridFlow & The Single-Controller Architecture](../06-rl-infra/04-verl.html)) attempts a middle ground with its "HybridFlow" design — a single-controller architecture that uses Ray for orchestration but NCCL for intra-group communication, with explicit "resource pools" that can mix tensor-parallel groups from different model instances.


{{fig:ray-dynamic-vs-megatron-static-orchestration}}


---

## 6.5.6 Implementing a Minimal Ray-Based RLHF Loop from Scratch

To cement understanding, let us build a minimal but complete RLHF loop using Ray actors. We use a toy GPT-2-scale model and a synthetic reward function (string length, normalized) to keep it runnable without a GPU cluster. This illustrates every mechanism: rollout, scoring, advantage estimation, PPO update, weight sync.

```python
"""
Minimal Ray-based RLHF loop (runnable on a single machine with 1-2 GPUs).
Uses HuggingFace Transformers + Ray + simple PPO.
NOT production code — for pedagogical clarity.
"""

import ray
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Dict
import numpy as np


# ── Reward function ────────────────────────────────────────────────────────
def reward_fn(responses: List[str]) -> List[float]:
    """
    Toy reward: prefers responses of length 50–150 chars.
    Real RLHF uses a trained reward model or verifier.
    """
    rewards = []
    for r in responses:
        length = len(r)
        # Gaussian-like reward centered at 100 chars
        reward = float(np.exp(-((length - 100) ** 2) / (2 * 50 ** 2)))
        rewards.append(reward)
    return rewards


# ── Rollout Actor ──────────────────────────────────────────────────────────
@ray.remote(num_gpus=0.5)   # share GPU for toy demo
class RolloutActor:
    """Generates responses given prompts."""
    def __init__(self, model_name: str):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32
        )
        self.model.eval()
        self.device = "cpu"   # CPU for toy demo; change to "cuda" in practice

    def generate(self, prompts: List[str], max_new_tokens: int = 64
                 ) -> List[Tuple[str, List[float]]]:
        """Returns (response_text, token_logprobs) for each prompt."""
        results = []
        for prompt in prompts:
            inputs = self.tok(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.9,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            # Decode generated tokens (excluding prompt)
            gen_tokens = out.sequences[0, inputs["input_ids"].shape[1]:]
            response = self.tok.decode(gen_tokens, skip_special_tokens=True)

            # Compute per-token log-probabilities
            logprobs = []
            for step, score in enumerate(out.scores):
                lp = F.log_softmax(score, dim=-1)
                tok_id = gen_tokens[step].item()
                logprobs.append(lp[0, tok_id].item())

            results.append((response, logprobs))
        return results

    def update_weights(self, state_dict_ref):
        """Load new weights from Ray object store."""
        state_dict = ray.get(state_dict_ref)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        return True


# ── Policy Training Actor ──────────────────────────────────────────────────
@ray.remote(num_gpus=0.5)
class PolicyActor:
    """Holds the trainable policy and runs PPO updates."""
    def __init__(self, model_name: str, lr: float = 1e-5):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32
        )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.device = "cpu"

    def ppo_step(
        self,
        prompts: List[str],
        responses: List[str],
        old_logprobs: List[List[float]],   # per-token, from rollout
        advantages: List[float],           # per-sequence (scalar GAE)
        clip_eps: float = 0.2,
        kl_coef: float = 0.05,
    ) -> Dict[str, float]:
        """
        PPO clipped policy gradient update.
        For simplicity, uses sequence-level advantages (not token-level GAE).
        """
        total_loss = 0.0
        self.model.train()
        self.optimizer.zero_grad()

        for prompt, resp, old_lp_seq, adv in zip(
            prompts, responses, old_logprobs, advantages
        ):
            # Tokenize full sequence (prompt + response)
            full_text = prompt + resp
            tokens = self.tok(full_text, return_tensors="pt").to(self.device)
            prompt_len = self.tok(prompt, return_tensors="pt")["input_ids"].shape[1]

            # Forward pass
            with torch.enable_grad():
                logits = self.model(**tokens).logits  # [1, T, vocab]

            # Log-probs for response tokens only
            resp_logits = logits[0, prompt_len - 1:-1, :]  # [resp_len, vocab]
            resp_token_ids = tokens["input_ids"][0, prompt_len:]
            new_lps = F.log_softmax(resp_logits, dim=-1)
            new_lp_seq = new_lps[
                torch.arange(len(resp_token_ids)), resp_token_ids
            ].tolist()

            # Clip old_lp_seq to match actual response length
            seq_len = min(len(old_lp_seq), len(new_lp_seq))
            old_lp_t = torch.tensor(old_lp_seq[:seq_len])
            new_lp_t = torch.stack([
                new_lps[i, resp_token_ids[i]] for i in range(seq_len)
            ])

            # Importance sampling ratio: exp(new - old)
            ratio = torch.exp(new_lp_t - old_lp_t)
            adv_t = torch.tensor(adv)  # broadcast over tokens

            # Clipped surrogate
            surr1 = ratio * adv_t
            surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_t
            loss = -torch.min(surr1, surr2).mean()
            total_loss += loss

        (total_loss / len(prompts)).backward()

        # Gradient clipping to prevent instability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.model.eval()
        return {"policy_loss": (total_loss / len(prompts)).item()}

    def get_state_dict(self):
        """Return CPU state dict for broadcasting to rollout actor."""
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}


# ── Main Training Loop ─────────────────────────────────────────────────────
def main():
    ray.init(ignore_reinit_error=True)

    MODEL_NAME = "gpt2"   # small model; swap for LLaMA-3-8B in practice
    prompts = [
        "Tell me about the water cycle:",
        "Explain gradient descent:",
        "What is a transformer model?",
        "Describe how the internet works:",
    ] * 4   # 16 prompts total

    rollout_actor = RolloutActor.remote(MODEL_NAME)
    policy_actor = PolicyActor.remote(MODEL_NAME)

    for step in range(5):
        # 1. Generate rollouts
        results = ray.get(rollout_actor.generate.remote(prompts))
        responses = [r[0] for r in results]
        old_logprobs = [r[1] for r in results]

        # 2. Score with reward function
        rewards = reward_fn(responses)

        # 3. Compute advantages (simple: advantage = reward - mean(reward))
        mean_r = np.mean(rewards)
        std_r = np.std(rewards) + 1e-8
        advantages = [(r - mean_r) / std_r for r in rewards]

        # 4. PPO update
        stats = ray.get(policy_actor.ppo_step.remote(
            prompts, responses, old_logprobs, advantages
        ))
        print(f"Step {step}: loss={stats['policy_loss']:.4f}, "
              f"mean_reward={mean_r:.4f}")

        # 5. Sync weights to rollout actor
        sd_ref = policy_actor.get_state_dict.remote()
        ray.get(rollout_actor.update_weights.remote(sd_ref))

    ray.shutdown()


if __name__ == "__main__":
    main()
```

---

## 6.5.7 Performance Engineering and Scaling Considerations

### Throughput Bottleneck Analysis

In any Ray-based RLHF system, throughput is determined by the slowest stage in the pipeline. The typical bottleneck hierarchy (from most to least common):

1. **Rollout (generation):** autoregressive decoding is memory-bandwidth-bound. vLLM's PagedAttention and continuous batching partially amortize this, but for long sequences it dominates.
2. **Reward model scoring:** if the RM is large (e.g., 70B), its forward pass can match or exceed policy update time.
3. **Weight synchronization:** transferring 140 GB of bf16 parameters over PCIe (16 GB/s) takes ~9 seconds per sync. This motivates reducing sync frequency or using NVLink.
4. **Python controller overhead:** for very short update steps (small models), the Ray controller's Python overhead can be significant — on the order of 10–50 ms per step.

!!! example "Weight Sync Bandwidth Example"

    Consider syncing a 70B parameter policy (140 GB bf16) from 8 training GPUs to 4 vLLM engines (2 GPUs each):

    - **NVLink intra-node** (600 GB/s bidirectional): $140 \text{ GB} / 600 \text{ GB/s} \approx 0.23$ seconds.
    - **PCIe Gen4** (64 GB/s bidirectional): $140 \text{ GB} / 64 \text{ GB/s} \approx 2.2$ seconds.
    - **InfiniBand HDR** (25 GB/s per link, 8 links): $140 \text{ GB} / 200 \text{ GB/s} \approx 0.7$ seconds.

    If PPO update epochs take roughly 30 seconds, weight sync adds 1–7% overhead depending on interconnect — acceptable. But for smaller models where updates are faster (say, 5 seconds for a 7B model), sync overhead can reach 20–40% without careful optimization (e.g., overlapping sync with the next rollout batch).

### Reducing Weight Sync Overhead

Three strategies used in practice:

1. **Lazy sync:** only sync every $N$ PPO epochs rather than every step. Trades off staleness of the rollout policy for reduced communication cost.
2. **Sync only what changes:** with full fine-tuning every parameter is dirty after every Adam step, so there is nothing to skip. But under **LoRA** only the adapter matrices are trainable — a few hundred MB rather than 140 GB — and vLLM can serve them through its LoRA path or merge them on receipt. OpenRLHF exposes this via `--lora_rank`; it turns weight sync from a first-order cost into a rounding error, at the price of the capacity limits discussed in [PEFT I: LoRA, QLoRA, DoRA & The Adapter Family](../05-posttraining-alignment/03-peft-lora-qlora.html).
3. **In-place NCCL broadcast:** instead of going through the Ray object store, use a direct NCCL broadcast from training GPU ranks to vLLM GPU ranks (`--vllm_sync_backend nccl`, Path B above). Requires pre-established communicators at startup — the approach NeMo-Aligner uses natively, and now the default everywhere.
4. **Overlap sync with the next rollout's prefill.** The broadcast is a parameter-by-parameter loop; vLLM cannot start generating until the *last* tensor lands, but the trainer's other ranks are idle throughout. Frameworks hide most of this by issuing the broadcast on a side CUDA stream while the controller assembles the next prompt batch.

### Async Rollout Pipelines

An advanced optimization (covered more fully in [Prime-RL, Async RL & Decentralized Training](../06-rl-infra/06-prime-rl-async.html)) is to decouple rollout from training: while the policy is being updated on one batch, the vLLM engine generates the *next* batch of rollouts using the previous policy weights. This "off-policy" approach introduces staleness but keeps both GPU pools busy; Noukhovitch et al. (2024) report roughly 40% wall-clock savings at one step of staleness.

OpenRLHF supports this through an asynchronous-training mode that launches rollout as a background Ray task while training proceeds (the flag has been spelled differently across releases — check `python3 -m openrlhf.cli.train_ppo_ray --help` for the one your version exposes). The controller pattern is the same regardless:

```python
# Conceptual async pipeline
import ray
from queue import Queue

def async_ppo_loop(policy_actor, vllm_actor, reward_actor, prompts,
                    batch_size, num_steps, sync_every=1):
    rollout_queue = Queue(maxsize=2)  # prefetch up to 2 batches

    # Kick off first rollout
    rollout_future = vllm_actor.rollout.remote(prompts[:batch_size])

    for step in range(num_steps):
        # Overlap: while we update on current batch, prefetch next rollout
        next_rollout_future = vllm_actor.rollout.remote(
            prompts[step * batch_size : (step + 1) * batch_size]
        )

        # Get current batch (may already be ready)
        current_rollout = ray.get(rollout_future)
        rewards = ray.get(reward_actor.score.remote(ray.put(current_rollout)))

        # Train on current batch (this is the slow step)
        ray.get(policy_actor.ppo_step.remote(ray.put(current_rollout), ray.put(rewards)))

        # Sync weights only every K steps to amortize cost
        if step % sync_every == 0:
            sd_ref = policy_actor.get_state_dict.remote()
            ray.get(vllm_actor.load_weights.remote(sd_ref))

        rollout_future = next_rollout_future   # advance the pipeline
```

---

## 6.5.8 Choosing a Framework: Decision Guide

Here is a practical guide for selecting between OpenRLHF, NeMo-Aligner, and alternatives (TRL, veRL) based on your constraints.

| Scenario | Recommendation |
|----------|----------------|
| ~100M–1B model, single GPU, GRPO/RLVR | No framework — a single-process loop with an in-process vLLM engine |
| Research prototype, 7B–13B model, single node | TRL (simplest, no Ray needed) |
| Research, 70B+ model, heterogeneous roles | OpenRLHF (flexible Ray decomposition) |
| Production, NVIDIA DGX cluster, 70B+ model | NeMo-RL (Megatron Core kernels + Ray; successor to NeMo-Aligner) |
| Custom RL algorithm with unique role topology | veRL (HybridFlow, explicit resource pools) |
| Async / decentralized training across commodity | Prime-RL (see Chapter 6.6) |
| GRPO or critic-free RL | TRL or veRL (no critic actor needed) |

!!! tip "When you do *not* need any of this"

    Every mechanism in this chapter exists to solve one problem: the policy, critic, reference and reward models do not fit on one GPU together. Below roughly 1B parameters they do. For **Stack-100M** — our ~100M-parameter capstone model — the policy is ~200 MB in bf16, the reference is another 200 MB, and a verifiable reward function is a Python assertion with no parameters at all. Everything lives in one process on one card: you instantiate a vLLM `LLM` for generation and a `nn.Module` for training in the same script, and "weight sync" is a `load_weights` call on a state dict that never leaves the GPU. No Ray, no placement groups, no NCCL rendezvous. The capstone's GRPO loop does exactly this — see [Post-Training: SFT, DPO, and Narrow RLVR (GRPO) That Works at 100M](../14-capstone/09-post-training.html).

    Read this chapter for the *shape* of the problem, not because you need Ray at 100M. The value of knowing it is that the single-process loop and the 32-GPU OpenRLHF job are the **same five phases** — generate, score, estimate advantage, update, sync — and the moment your model outgrows one GPU, each phase becomes an actor in the diagram above. Ray's overhead (10–50 ms of Python coordination per step) is invisible when a step takes 30 s and ruinous when it takes 200 ms, which is precisely the regime a 100M model runs in.

The choice between OpenRLHF and NeMo-Aligner is often less about algorithmic capability and more about **operational familiarity**: teams already running Megatron-LM pretraining will find NeMo-Aligner's configuration files familiar; teams comfortable with Ray (e.g., those using Ray Serve for inference) will find OpenRLHF's programming model more natural.

One practical note: as of 2026, OpenRLHF and veRL have the largest and most active open-source communities, with extensive documented examples for GRPO and RLVR (see [RL with Verifiable Rewards (RLVR) & The Reasoning Recipe](../05-posttraining-alignment/09-rlvr-reasoning.html)) and broad model support. On the NVIDIA side, NeMo-RL (the successor to the now-archived NeMo-Aligner) is the more tightly hardware-integrated option and the framework of choice for production RLHF at NVIDIA scale.

!!! interview "Interview Corner"

    **Q:** You need to run PPO on a 70B policy and a 70B critic simultaneously. Both require ZeRO-3. Your cluster has 64 × H100 80 GB GPUs. How would you lay out the actors in OpenRLHF, and what is the main engineering risk?

    **A:** A reasonable layout: 16 GPUs for the policy training actor (ZeRO-3 + CPU Adam offload, ~8.75 GB params + ~8.75 GB grads per GPU), 16 GPUs for the critic (same), 8 GPUs for the reference model (frozen, no optimizer or grads, ~17.5 GB per GPU), 8 GPUs for two vLLM engines at TP=4 (35 GB of weights per GPU — TP=2 would put 70 GB on each card and leave no room for KV cache, which is the trap in this question), 2 GPUs for the reward model (8B). Total: 50 GPUs, leaving 14 spare for headroom or a second RM instance.

    The main engineering risk is **weight synchronization latency**: after each PPO epoch, 140 GB of bf16 parameters must transfer from the policy training actor to the vLLM engines. Over InfiniBand or NVLink this takes on the order of 0.5–2 seconds per sync — tolerable if PPO updates take 10+ seconds, but a significant fraction of runtime for small batches. The mitigation is to sync less frequently (every N update epochs) or to pre-establish direct NCCL communicators between training and vLLM ranks to bypass the Ray object store.

---

!!! key "Key Takeaways"
    - Ray provides a **distributed actor model** that maps cleanly onto RLHF roles: each actor (policy, critic, RM, reference) owns its GPUs and communicates asynchronously via futures and the object store.
    - **OpenRLHF** uses Ray + vLLM + DeepSpeed ZeRO. Its strength is flexibility: roles can have different model sizes, parallelism degrees, and GPU counts, making it ideal for research and heterogeneous model configurations.
    - **NeMo-Aligner** uses Megatron-LM 3D parallelism + TRT-LLM rollouts. Its strength is throughput on homogeneous NVIDIA clusters: all-reduce and point-to-point transfers use pre-established NCCL communicators with no Python coordination overhead.
    - **Weight synchronization** (from training actor to rollout engine) is a key engineering cost. On a 70B model, even NVLink sync takes ~0.2 seconds; PCIe can take 2+ seconds. Sync frequency should be tuned to balance policy staleness against communication overhead.
    - Ray's **placement groups** with `STRICT_PACK` are essential for multi-GPU actors: they guarantee that all shards of a tensor-parallel group land on the same node, enabling fast NVLink communication within the group.
    - The **object store** provides zero-copy reads for co-located processes — critical for large experience batches (rollout tokens + log-probs) that can reach 8+ GB per batch.
    - The Megatron-native design (NeMo-Aligner, now succeeded by **NeMo-RL**) is the better choice when your cluster is NVIDIA DGX-based, your models are homogeneous in size, and you want NVIDIA's kernel optimizations and TRT-LLM throughput. OpenRLHF is the better choice when you need flexibility, fast iteration, or heterogeneous actor sizes.
    - Switching `--advantage_estimator` from `gae` to `group_norm` (**GRPO**) or `reinforce_baseline` (REINFORCE++) deletes the critic actor entirely — a topology change, not just an algorithm change, freeing an entire model's worth of parameters, gradients and optimizer state. This is why most 2026 reasoning-RL runs are critic-free.
    - Both frameworks support **async rollout pipelines** where generation and training overlap; published results report on the order of 40% wall-clock reduction, at the cost of off-policy staleness.

---

!!! sota "State of the Art & Resources (2026)"
    Ray-based RLHF systems have matured rapidly: OpenRLHF and veRL are now the dominant open-source frameworks for large-scale PPO/GRPO training, while NVIDIA replaced NeMo-Aligner with NeMo-RL (2025) — adding Ray orchestration and Hugging Face integration alongside Megatron Core kernels. Async rollout pipelines (decoupling generation from gradient updates) have become standard practice, delivering ~40% throughput gains with minimal policy staleness.

    **Foundational work**

    - [Moritz et al., *Ray: A Distributed Framework for Emerging AI Applications* (2018)](https://arxiv.org/abs/1712.05889) — the OSDI paper that introduced Ray's actor/task model, now the de-facto orchestration substrate for RLHF.
    - [Schulman et al., *Proximal Policy Optimization Algorithms* (2017)](https://arxiv.org/abs/1707.06347) — the PPO objective that all frameworks in this chapter implement.

    **Recent advances (2023–2026)**

    - [Hu et al., *OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework* (2024)](https://arxiv.org/abs/2405.11143) — the paper describing OpenRLHF's Ray + vLLM + DeepSpeed ZeRO architecture with 1.2–1.7× speedups over prior systems.
    - [Shen et al., *NeMo-Aligner: Scalable Toolkit for Efficient Model Alignment* (2024)](https://arxiv.org/abs/2405.01481) — NVIDIA's Megatron-LM + TRT-LLM RLHF design; now superseded by NeMo-RL.
    - [Sheng et al., *HybridFlow: A Flexible and Efficient RLHF Framework* (2024)](https://arxiv.org/abs/2409.19256) — the veRL paper introducing the 3D-HybridEngine that reshards weights between training and generation with zero memory redundancy (EuroSys '25).
    - [Noukhovitch et al., *Asynchronous RLHF: Faster and More Efficient Off-Policy RL for Language Models* (2024)](https://arxiv.org/abs/2410.18252) — ICLR 2025 paper demonstrating ~40% wall-clock speedup by decoupling rollout generation from policy updates.
    - [Yu et al., *DAPO: An Open-Source LLM Reinforcement Learning System at Scale* (2025)](https://arxiv.org/abs/2503.14476) — ByteDance's production-scale GRPO variant built on veRL, reaching 50 pts on AIME 2024 with Qwen2.5-32B.

    **Open-source & tools**

    - [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — production-ready Ray + vLLM + DeepSpeed framework supporting PPO, GRPO, REINFORCE++, async rollout, and VLMs.
    - [verl-project/verl](https://github.com/verl-project/verl) — HybridFlow's open-source implementation; supports FSDP, Megatron-LM, vLLM, and SGLang backends.
    - [NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL) — NeMo-RL, NVIDIA's 2025 successor to NeMo-Aligner with Ray scheduling, Megatron Core, and HuggingFace integration.
    - [PrimeIntellect-ai/prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) — fully async, decentralized RL training framework scaling to 1000+ GPUs across heterogeneous nodes, used to train INTELLECT-2.

    **Go deeper**

    - [NeMo-RL Documentation](https://docs.nvidia.com/nemo/rl/latest/index.html) — official NVIDIA docs covering DTensor and Megatron Core backends, GRPO/DPO recipes, and multi-node deployment.

## Further Reading

- **OpenRLHF** — Jian Hu et al., "OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework", 2024. (GitHub: OpenLLMAI/OpenRLHF)
- **NeMo-Aligner** — Gerald Shen et al., "NeMo-Aligner: Scalable Toolkit for Efficient Model Alignment", 2024. (GitHub: NVIDIA/NeMo-Aligner)
- **Ray** — Moritz et al., "Ray: A Distributed Framework for Emerging AI Applications", OSDI 2018.
- **DeepSpeed ZeRO** — Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models", SC 2020.
- **vLLM** — Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023.
- **Megatron-LM** — Narayanan et al., "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM", SC 2021.
- **GAE** — Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation", ICLR 2016.
- **veRL HybridFlow** — Sheng et al., "HybridFlow: A Flexible and Efficient RLHF Framework", 2024. (GitHub: volcengine/verl)

---

## Exercises

**1.** In OpenRLHF the vLLM engine holds a *read-only* copy of the policy weights, and after every batch of PPO updates the controller runs a `sync_weights_to_vllm()` step (Phase 6 of `run_ppo_training`). A naive single-process PPO implementation has no such step. Explain *why* the extra copy and the sync step exist in OpenRLHF, and state precisely which correctness property of PPO would silently degrade if you simply never called the sync.

??? note "Solution"
    OpenRLHF decomposes the system into independent Ray actors: the **policy training actor** (DeepSpeed ZeRO-3) and the **vLLM rollout actor** are *different processes on different GPUs* with *different weight representations*. The training actor holds sharded, trainable bf16 parameters plus gradients and optimizer state; vLLM holds a dense, inference-optimized copy laid out for PagedAttention decoding. They cannot share the same tensors, so vLLM necessarily works from a copy. A single-process PPO uses one model object for both generation and the gradient step, so "sync" is implicit — the same weights that generated the rollout are the ones being updated.

    The property that degrades without sync is the **on-policy-ness** of the data, i.e. the validity of the importance-sampling ratio $r_t(\theta) = \exp(\log \pi_\theta(a_t\mid s_t) - \log \pi_{\theta_{\text{old}}}(a_t\mid s_t))$. PPO assumes the rollout was drawn from a policy close to the current one, and it corrects the small remaining mismatch with $r_t$ and the clip range $[1-\epsilon,\,1+\epsilon]$. If you never sync, vLLM keeps generating from the *initial* weights forever while the training actor drifts arbitrarily far away. The behavior policy $\pi_{\theta_{\text{old}}}$ and the target policy $\pi_\theta$ diverge without bound, the ratio blows past the clip range on essentially every token (so the clipped surrogate stops providing a useful gradient signal), and every update becomes heavily off-policy. Training does not crash — it silently learns from stale data and the KL-to-reference and reward curves stop tracking what the policy is actually doing. Sync is what keeps the generation policy $\approx$ the training policy so PPO's near-on-policy assumption holds.

**2.** The placement-group example reserves 8 bundles with `strategy="STRICT_PACK"` for a tensor-parallel (TP=8) actor group. (a) What partial-allocation failure does using a placement group prevent in the first place? (b) Why specifically `STRICT_PACK` rather than the default spread, given the actor is tensor-parallel? (c) What would go wrong at runtime if 4 of the 8 shards landed on node A and 4 on node B?

??? note "Solution"
    **(a)** Without a placement group, a request for 8 single-GPU workers is granted greedily and independently. On a busy cluster you can acquire, say, 5 GPUs and then block indefinitely waiting for the other 3 while still holding the 5 — and if several jobs do this simultaneously they deadlock, each pinning GPUs the others need. A placement group reserves the whole "bundle" of 8 GPUs **atomically (gang scheduling)**: `ray.get(pg.ready())` returns only when all 8 are secured, so you never hold a partial, unusable allocation.

    **(b)** TP splits a single transformer layer's matmuls across all 8 ranks, so every layer requires all-reduce / all-gather collectives among the 8 shards *on the critical path of every forward and backward pass*. That traffic must ride the fast intra-node fabric (NVLink). `STRICT_PACK` forces all 8 bundles onto **one node**; the default spread strategy would scatter them across nodes, pushing the per-layer collectives onto slow cross-node links.

    **(c)** The group still initializes (the NCCL communicator forms across both nodes), but every TP collective now traverses the inter-node network (InfiniBand/TCP at ~25-200 GB/s) instead of NVLink (~600 GB/s) for *each layer of every step*. Since these collectives are on the hot path and happen many times per token, generation and training throughput collapse — you pay a cross-node round trip repeatedly where you expected NVLink. It is a correctness-preserving but catastrophic performance regression, which is exactly what `STRICT_PACK` exists to prevent.

**3.** Use the chapter's Weight Sync Bandwidth Example. A 70B policy is 140 GB in bf16. Assume one PPO update phase takes $T = 30$ s, and define sync overhead as $\text{sync}/(\text{sync} + T)$. (a) Compute the sync time and the overhead fraction for NVLink (600 GB/s), PCIe Gen4 (64 GB/s), and InfiniBand HDR (200 GB/s aggregate). (b) On the PCIe path, below what update time $T$ does sync overhead exceed 25%? (c) You cannot change the interconnect but you can apply *lazy sync* (sync once every $N$ update phases). With PCIe and $T = 6$ s, what $N$ brings the *amortized* overhead below 10%?

??? note "Solution"
    **(a)** Sync time is $140\,\text{GB}$ divided by bandwidth:

    - NVLink: $140/600 = 0.233$ s. Overhead $= 0.233/(0.233+30) = 0.77\%$.
    - PCIe Gen4: $140/64 = 2.19$ s. Overhead $= 2.19/(2.19+30) = 6.8\%$.
    - InfiniBand HDR: $140/200 = 0.70$ s. Overhead $= 0.70/(0.70+30) = 2.3\%$.

    These land in the chapter's stated "1-7% at $T\approx30$ s" range.

    **(b)** With PCIe, $\text{sync} = 2.19$ s. Solve $\dfrac{2.19}{2.19 + T} > 0.25$:
    $$
    2.19 + T < \frac{2.19}{0.25} = 8.76 \quad\Rightarrow\quad T < 6.57 \text{ s}.
    $$
    So once an update phase drops below about $6.6$ s, PCIe sync overhead exceeds 25%. This is why small/fast models suffer most.

    **(c)** Lazy sync amortizes one $2.19$ s transfer over $N$ updates of $6$ s each. Amortized overhead:
    $$
    \frac{2.19}{2.19 + N\cdot 6} < 0.10 \quad\Rightarrow\quad 2.19 + 6N > 21.9 \quad\Rightarrow\quad N > 3.28.
    $$
    So $N = 4$ suffices (overhead $= 2.19/(2.19 + 24) = 8.4\%$; $N=3$ gives $10.8\%$, just over). The cost is that the rollout policy is now up to 4 update phases stale — more off-policy data, the trade-off the "Reducing Weight Sync Overhead" section flags.

**4.** The Worked Memory Example places the 70B policy training actor on **16** GPUs (ZeRO-3, Adam offloaded to CPU) and reports ~20-22 GB per 80 GB GPU. Suppose you instead give the policy only **8** GPUs, keeping ZeRO-3 and CPU Adam offload. Estimate per-GPU memory for parameters, partitioned gradients, and activations, give a total, and say whether it still fits and how "comfortable" it is compared with the 16-GPU layout.

??? note "Solution"
    ZeRO-3 shards parameters and gradients evenly across the data-parallel group, so halving the GPU count doubles each shard.

    - **Parameters (bf16):** $140\,\text{GB} / 8 = 17.5$ GB per GPU (was $140/16 = 8.75$).
    - **Gradients (bf16, ZeRO-3 partitioned, same size as params):** $17.5$ GB per GPU (was $8.75$).
    - **Optimizer state (Adam):** offloaded to CPU, so $\approx 0$ GB on the GPU (same as the 16-GPU case — offload makes this term vanish from the GPU budget regardless of GPU count). Note what happens *without* offload: the $m$, $v$ and fp32 master copies total 840 GB, which is $840/8 = 105$ GB per GPU — more than the whole card. On 8 GPUs, offload is not a tuning knob, it is the only thing that makes the layout possible at all.
    - **Activations (micro-batch 8, len 2048, gradient checkpointing):** the chapter's ~2-4 GB per GPU still applies; call it ~3 GB.

    **Total:** $17.5 + 17.5 + 3 \approx 38$ GB per GPU.

    It still fits on an 80 GB A100/H100 with room to spare, but it is markedly less comfortable than the 16-GPU layout (~21 GB). You have nearly doubled the per-GPU footprint (~21 GB → ~38 GB), cutting free headroom from ~59 GB to ~42 GB (about 29% less), which directly shrinks the room for larger micro-batches, longer contexts, or activation memory spikes. The parameter and gradient shards scale as $1/(\text{\#GPUs})$; the offloaded optimizer term does not scale, which is exactly why CPU Adam offload is what keeps the 8-GPU layout viable at all.

**5.** Modify the minimal Ray RLHF loop (Section 6.5.6) to use **lazy weight sync**: pass a `sync_every=K` argument to `main()` so the policy weights are pushed to the rollout actor only every $K$ steps instead of every step. Show the changed `main()` loop, and explain in one or two sentences what becomes stale and why the `old_logprobs` used inside `ppo_step` are still the correct ones to use.

??? note "Solution"
    Only the tail of the loop changes: gate the `get_state_dict` / `update_weights` sync on the step index. Everything else (rollout, reward, advantage, `ppo_step`) is unchanged.

    ```python
    def main(sync_every: int = 1):
        ray.init(ignore_reinit_error=True)

        MODEL_NAME = "gpt2"
        prompts = [
            "Tell me about the water cycle:",
            "Explain gradient descent:",
            "What is a transformer model?",
            "Describe how the internet works:",
        ] * 4

        rollout_actor = RolloutActor.remote(MODEL_NAME)
        policy_actor = PolicyActor.remote(MODEL_NAME)

        for step in range(5):
            # 1. Generate rollouts (from the LAST synced weights)
            results = ray.get(rollout_actor.generate.remote(prompts))
            responses = [r[0] for r in results]
            old_logprobs = [r[1] for r in results]

            # 2. Score
            rewards = reward_fn(responses)

            # 3. Advantages
            mean_r = np.mean(rewards)
            std_r = np.std(rewards) + 1e-8
            advantages = [(r - mean_r) / std_r for r in rewards]

            # 4. PPO update
            stats = ray.get(policy_actor.ppo_step.remote(
                prompts, responses, old_logprobs, advantages
            ))
            print(f"Step {step}: loss={stats['policy_loss']:.4f}, "
                  f"mean_reward={mean_r:.4f}")

            # 5. LAZY sync: only push weights every `sync_every` steps
            if step % sync_every == 0:
                sd_ref = policy_actor.get_state_dict.remote()
                ray.get(rollout_actor.update_weights.remote(sd_ref))

    if __name__ == "__main__":
        main(sync_every=2)
    ```

    What goes stale is the **rollout actor's copy of the policy**: between syncs it keeps generating from weights that are up to $K-1$ steps behind the training actor, so the collected data is increasingly off-policy (the trade-off named in "Reducing Weight Sync Overhead"). The `old_logprobs` remain correct because PPO's importance ratio is defined relative to *the policy that actually produced the tokens* ($\pi_{\theta_{\text{old}}}$), and those log-probs were returned by the rollout actor *at generation time* — they always describe the exact distribution the sampled tokens came from, regardless of how many steps ago that policy was synced.

**6.** Extend the minimal loop with a **reference model and a KL penalty**, mirroring the reward construction in the NeMo-Aligner section (`reward = RM score - kl_coef * KL`). Implement a `ReferenceActor` whose `log_probs(prompts, responses)` returns the per-sequence summed log-probability under a frozen model, then show how to fold a sequence-level KL term into the reward before computing advantages in `main()`. Use the chapter's toy `reward_fn` as the "RM".

??? note "Solution"
    The reference actor loads the same base model as the policy but is frozen; it recomputes log-probs of the *already generated* responses with a teacher-forced forward pass — the same slicing trick `PolicyActor.ppo_step` uses (`prompt_len - 1 : -1`). We return one scalar per sequence (the summed response log-prob) so it lines up with the toy loop's sequence-level advantages.

    ```python
    @ray.remote(num_gpus=0.5)
    class ReferenceActor:
        """Frozen reference model: scores responses for the KL penalty."""
        def __init__(self, model_name: str):
            self.tok = AutoTokenizer.from_pretrained(model_name)
            self.tok.pad_token = self.tok.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float32
            )
            self.model.eval()
            self.device = "cpu"

        def log_probs(self, prompts, responses):
            """Per-sequence summed log-prob of the response tokens."""
            seq_logps = []
            for prompt, resp in zip(prompts, responses):
                tokens = self.tok(prompt + resp, return_tensors="pt").to(self.device)
                prompt_len = self.tok(
                    prompt, return_tensors="pt"
                )["input_ids"].shape[1]
                with torch.no_grad():
                    logits = self.model(**tokens).logits          # [1, T, vocab]
                resp_logits = logits[0, prompt_len - 1:-1, :]      # [resp_len, vocab]
                resp_ids = tokens["input_ids"][0, prompt_len:]
                lps = F.log_softmax(resp_logits, dim=-1)
                tok_lps = lps[torch.arange(len(resp_ids)), resp_ids]
                seq_logps.append(tok_lps.sum().item())
            return seq_logps
    ```

    In `main()`, create the actor once and adjust the reward with a sequence-level KL estimate $\text{KL}_{\text{seq}} \approx \sum_t (\log \pi_{\text{old}}(a_t) - \log \pi_{\text{ref}}(a_t))$, which is just `sum(old_logprobs) - ref_logprob_sum` per sequence. Insert between the scoring and advantage steps:

    ```python
    KL_COEF = 0.05
    ref_actor = ReferenceActor.remote(MODEL_NAME)   # created alongside the others

    # ... inside the loop, after `rewards = reward_fn(responses)` ...

    # Reference log-probs (per-sequence sum) for the KL penalty
    ref_logps = ray.get(ref_actor.log_probs.remote(prompts, responses))

    # Sequence-level KL: sum of per-token (old - ref); old_logprobs is per-token
    kl_seq = [sum(old_lp) - ref_lp
              for old_lp, ref_lp in zip(old_logprobs, ref_logps)]

    # reward = RM score (toy reward_fn) - kl_coef * KL      [NeMo-Aligner form]
    rewards = [r - KL_COEF * kl for r, kl in zip(rewards, kl_seq)]

    # advantages are then computed from these KL-penalized rewards, unchanged:
    mean_r = np.mean(rewards)
    std_r = np.std(rewards) + 1e-8
    advantages = [(r - mean_r) / std_r for r in rewards]
    ```

    This reproduces the NeMo-Aligner recipe `rewards = -kl_coef * kl_penalty` with the RM score added on top, but at sequence granularity to fit the toy loop. Because $\pi_{\text{old}}$ (the rollout policy) and $\pi_{\text{ref}}$ start identical, $\text{KL}_{\text{seq}} \approx 0$ at step 0 and grows as the policy drifts from the reference, penalizing responses that stray too far — exactly the role of the KL term in RLHF. Note the reference actor is frozen and holds no optimizer state, matching the chapter's memory budget (parameters only, ~17.5 GB/GPU for a 70B reference).
