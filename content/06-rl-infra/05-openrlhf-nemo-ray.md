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
3. **Object store**: shared memory backed by Apache Plasma. Objects put into the store are zero-copy readable by any process on the same node.

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
        import torch.distributed as dist
        dist.init_process_group(
            backend="nccl",
            init_method="env://",   # env vars set by launcher
            rank=rank,
            world_size=world_size,
        )
        self.rank = rank

    def ping(self):
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
print(ray.get([s.ping.remote() for s in shards]))
# ['shard 0 alive', 'shard 1 alive', ..., 'shard 7 alive']
ray.shutdown()
```

### Object Store and Zero-Copy Tensors

A critical performance property of Ray's object store is **zero-copy reads for numpy arrays and tensors on the same node**. When rollout tokens and log-probabilities are serialized into the object store, the training worker can read them directly from shared memory without a copy. For a batch of 1024 sequences of length 2048, the logprob tensor is approximately:

$$
1024 \times 2048 \times 4 \text{ bytes} \approx 8 \text{ GB}
$$

Eliminating even one copy of that tensor per step saves gigabytes of PCIe or NVLink bandwidth per iteration.

Across nodes, Ray serializes via Apache Arrow and ships over TCP, so cross-node object transfers should be minimized — keep rollout workers and training workers co-located when possible.

---

## 6.5.2 OpenRLHF: Architecture and Design Philosophy

**OpenRLHF** (originally open-sourced in late 2023 by the OpenLLMAI community) is built around a clean separation of concerns: every RLHF role — **policy actor**, **reference actor**, **critic**, and **reward model** — lives in its own Ray actor group. The rollout engine is vLLM. Gradient updates use DeepSpeed ZeRO (see [Distributed Training I: Data Parallelism, DDP, ZeRO & FSDP](../03-pretraining/05-distributed-data-parallel.html)).

### Role Decomposition


{{fig:openrlhf-ray-role-decomposition}}


The key design insight: **the vLLM engine holds a *read-only copy* of policy weights for generation**. After each PPO update, the policy training actor must push its updated weights to the vLLM engine. This weight-sync step is where OpenRLHF differs from single-process approaches.

### Weight Synchronization via Distributed Checkpointing

OpenRLHF implements weight sync by having the policy actor broadcast updated parameters directly to the vLLM workers using NCCL collectives or shared-memory copies, bypassing disk entirely. The mechanism:

1. Policy actor finishes a gradient step and calls `sync_weights_to_vllm()`.
2. For each parameter tensor, the actor puts it into the Ray object store.
3. vLLM workers pull the tensors and call `model.load_state_dict(...)` in-place.
4. vLLM's KV cache is not invalidated by weight changes (generation restarts fresh).

```python
# Simplified weight-sync routine (OpenRLHF-style)
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

    def load_weights(self, state_dict_ref):
        """Receive new policy weights and reload into vLLM."""
        state_dict = ray.get(state_dict_ref)
        # vLLM exposes model.llm_engine.model_executor.driver_worker.model_runner.model
        model = self.engine.llm_engine.model_executor.driver_worker \
                           .model_runner.model
        missing, unexpected = model.load_weights(state_dict.items())
        assert not missing and not unexpected, \
            f"Weight mismatch: missing={missing}, unexpected={unexpected}"
        return "weights_loaded"

    def rollout(self, prompts, sampling_params):
        """Generate responses using updated weights."""
        from vllm import SamplingParams
        outputs = self.engine.generate(prompts, SamplingParams(**sampling_params))
        return [(o.prompt, o.outputs[0].text, o.outputs[0].logprobs)
                for o in outputs]
```

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
    rewards: List[float]
    advantages: List[float]
    old_logprobs: List[float]    # from vLLM at generation time
    ref_logprobs: List[float]    # from reference model


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
    """
    advantages = []
    last_gae = 0.0
    T = len(rewards)
    # For language model RL the 'value' is typically a scalar per sequence
    for t in reversed(range(T)):
        next_val = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_val - values[t]
        last_gae = delta + gamma * lam * last_gae
        advantages.insert(0, last_gae)
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

```bash
#!/bin/bash
# Launch OpenRLHF PPO training on 4 nodes, 8 GPUs each (32 GPUs total)
# Policy: LLaMA-3-70B  |  RM: LLaMA-3-8B  |  Rollout: vLLM

# Start Ray cluster (typically done via Kubernetes or Slurm integration)
ray start --head --num-gpus=8 --num-cpus=64

# On worker nodes:
# ray start --address=<head_ip>:6379 --num-gpus=8

python train_ppo.py \
  --pretrain meta-llama/Meta-Llama-3-70B-Instruct \
  --reward_pretrain meta-llama/Meta-Llama-3-8B \
  --save_path ./checkpoints/llama3-70b-ppo \
  --num_episodes 1 \
  --rollout_batch_size 1024 \
  --micro_rollout_batch_size 8 \
  --num_rollout_workers 4 \
  --vllm_num_engines 4 \
  --vllm_tensor_parallel_size 2 \
  --actor_num_nodes 2 \
  --actor_num_gpus_per_node 8 \
  --critic_num_nodes 1 \
  --critic_num_gpus_per_node 8 \
  --ref_num_nodes 1 \
  --ref_num_gpus_per_node 8 \
  --reward_num_nodes 1 \
  --reward_num_gpus_per_node 2 \
  --zero_stage 3 \
  --bf16 \
  --learning_rate 5e-7 \
  --kl_target 0.05 \
  --init_kl_coef 0.01 \
  --normalize_reward \
  --adam_offload      # offload optimizer state to CPU to save GPU memory
```

---

## 6.5.3 Memory Budget Analysis for OpenRLHF

Understanding how OpenRLHF allocates memory across its four actor groups is essential for capacity planning.

!!! example "Worked Memory Example: LLaMA-3-70B PPO on 32 GPUs"

    **Model size:** 70B parameters at bf16 = $70 \times 10^9 \times 2$ bytes = 140 GB.

    **Policy training actor (16 GPUs, ZeRO-3):**
    - Model parameters: $140 / 16 = 8.75$ GB per GPU
    - Gradients (same size as params with ZeRO-3): $8.75$ GB per GPU
    - Optimizer state (Adam: 2 moments, fp32): $140 \times 2 \times 4 / 16 = 70$ GB across 16 GPUs = $4.375$ GB per GPU in fp32 (but Adam can be offloaded to CPU)
    - Activations (per micro-batch=8, len=2048): roughly $2-4$ GB per GPU with gradient checkpointing
    - **Total per GPU (with CPU Adam offload):** ~14–18 GB on an 80 GB A100. Comfortable.

    **vLLM rollout engine (8 GPUs, TP=2, 4 engines × 2 GPUs):**
    - Model weights loaded as bfloat16: $140 / 4$ engines = 35 GB per engine = 17.5 GB per GPU
    - KV cache (remaining ~60 GB): supports ~100k tokens in flight per engine
    - **Total per GPU:** ~18 GB weights + KV cache fills the rest. Tight on 40 GB, comfortable on 80 GB.

    **Reference actor (8 GPUs, ZeRO-3, no optimizer):**
    - Parameters only: $140 / 8 = 17.5$ GB per GPU. No gradient storage needed.

    **Reward model (2 GPUs, 8B params at bf16):**
    - $8 \times 10^9 \times 2 / 2 = 8$ GB per GPU. Very comfortable.

    **Total cluster GPU memory used:** roughly $8 \times 18.75 + 8 \times 17.5 + 8 \times 17.5 + 2 \times 8 \approx$ 437 GB across 26 GPUs. On 32 × 80 GB = 2560 GB total, there is substantial headroom for batch sizes, long contexts, or larger models.

---

## 6.5.4 NeMo-Aligner: Megatron-Based RLHF

**NVIDIA NeMo-Aligner** (released in 2023 as part of the NeMo framework) takes a fundamentally different approach: instead of Ray, it uses **Megatron-LM's** native 3D parallelism and **NCCL** collective communication as the backbone. TensorRT-LLM (TRT-LLM) serves as the rollout engine rather than vLLM.

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
| Weight sync method | Ray object store + in-place load | NCCL broadcast between process groups |
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
2. **Differential sync:** track which parameter blocks changed since the last sync (using gradient sparsity in ZeRO-3) and only transfer modified shards.
3. **In-place NCCL broadcast:** instead of going through the Ray object store, use a direct NCCL broadcast from training GPU ranks to vLLM GPU ranks. Requires pre-established NCCL communicators at startup — the approach NeMo-Aligner uses natively.

### Async Rollout Pipelines

An advanced optimization (covered more fully in [Prime-RL, Async RL & Decentralized Training](../06-rl-infra/06-prime-rl-async.html)) is to decouple rollout from training: while the policy is being updated on one batch, the vLLM engine generates the *next* batch of rollouts using the previous policy weights. This "off-policy" approach introduces staleness but can nearly double throughput by keeping both GPU pools busy.

OpenRLHF supports this via the `--async_rollout` flag, which launches rollout as a background Ray task while training proceeds:

```python
# Conceptual async pipeline
import ray
from queue import Queue

def async_ppo_loop(policy_actor, vllm_actor, reward_actor, prompts):
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
| Research prototype, 7B–13B model, single node | TRL (simplest, no Ray needed) |
| Research, 70B+ model, heterogeneous roles | OpenRLHF (flexible Ray decomposition) |
| Production, NVIDIA DGX cluster, 70B+ model | NeMo-Aligner (optimized Megatron kernels, TRT-LLM) |
| Custom RL algorithm with unique role topology | veRL (HybridFlow, explicit resource pools) |
| Async / decentralized training across commodity | Prime-RL (see Chapter 6.6) |
| GRPO or critic-free RL | TRL or veRL (no critic actor needed) |

The choice between OpenRLHF and NeMo-Aligner is often less about algorithmic capability and more about **operational familiarity**: teams already running Megatron-LM pretraining will find NeMo-Aligner's configuration files familiar; teams comfortable with Ray (e.g., those using Ray Serve for inference) will find OpenRLHF's programming model more natural.

One practical note: as of mid-2025, OpenRLHF has a larger and more active open-source community, more documented examples for GRPO and RLVR (see [RL with Verifiable Rewards (RLVR) & The Reasoning Recipe](../05-posttraining-alignment/09-rlvr-reasoning.html)), and broader model support. NeMo-Aligner is more tightly integrated with NVIDIA's hardware optimizations and is the framework of choice for production RLHF at NVIDIA scale.

!!! interview "Interview Corner"

    **Q:** You need to run PPO on a 70B policy and a 70B critic simultaneously. Both require ZeRO-3. Your cluster has 64 × H100 80 GB GPUs. How would you lay out the actors in OpenRLHF, and what is the main engineering risk?

    **A:** A reasonable layout: 16 GPUs for the policy training actor (ZeRO-3, ~8.75 GB params per GPU), 16 GPUs for the critic (same), 8 GPUs for the reference model (no optimizer, ~17.5 GB per GPU), 8 GPUs across 4 vLLM engines (2 GPU TP each), 2 GPUs for the reward model (8B). Total: 50 GPUs, leaving 14 spare for headroom or a second RM instance.

    The main engineering risk is **weight synchronization latency**: after each PPO epoch, 140 GB of bf16 parameters must transfer from the policy training actor to the vLLM engines. Over InfiniBand or NVLink this takes on the order of 0.5–2 seconds per sync — tolerable if PPO updates take 10+ seconds, but a significant fraction of runtime for small batches. The mitigation is to sync less frequently (every N update epochs) or to pre-establish direct NCCL communicators between training and vLLM ranks to bypass the Ray object store.

---

!!! key "Key Takeaways"
    - Ray provides a **distributed actor model** that maps cleanly onto RLHF roles: each actor (policy, critic, RM, reference) owns its GPUs and communicates asynchronously via futures and the object store.
    - **OpenRLHF** uses Ray + vLLM + DeepSpeed ZeRO. Its strength is flexibility: roles can have different model sizes, parallelism degrees, and GPU counts, making it ideal for research and heterogeneous model configurations.
    - **NeMo-Aligner** uses Megatron-LM 3D parallelism + TRT-LLM rollouts. Its strength is throughput on homogeneous NVIDIA clusters: all-reduce and point-to-point transfers use pre-established NCCL communicators with no Python coordination overhead.
    - **Weight synchronization** (from training actor to rollout engine) is a key engineering cost. On a 70B model, even NVLink sync takes ~0.2 seconds; PCIe can take 2+ seconds. Sync frequency should be tuned to balance policy staleness against communication overhead.
    - Ray's **placement groups** with `STRICT_PACK` are essential for multi-GPU actors: they guarantee that all shards of a tensor-parallel group land on the same node, enabling fast NVLink communication within the group.
    - The **object store** provides zero-copy reads for co-located processes — critical for large experience batches (rollout tokens + log-probs) that can reach 8+ GB per batch.
    - NeMo-Aligner is the better choice when your cluster is NVIDIA DGX-based, your models are homogeneous in size, and you want NVIDIA's kernel optimizations and TRT-LLM throughput. OpenRLHF is the better choice when you need flexibility, fast iteration, or heterogeneous actor sizes.
    - Both frameworks support **async rollout pipelines** where generation and training overlap, roughly doubling end-to-end throughput at the cost of off-policy staleness.

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
