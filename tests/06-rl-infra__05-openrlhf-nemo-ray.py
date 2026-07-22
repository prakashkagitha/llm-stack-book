"""
Executable test for content/06-rl-infra/05-openrlhf-nemo-ray.md

Concatenates the chapter's 3 CPU-runnable Python blocks in order and exercises
each one on CPU / with tiny fixtures so the book's actual code runs end to end.

Blocks covered:
  #2 (line ~152) Simplified weight-sync routine (OpenRLHF-style):
                 PolicyTrainingActor + VLLMRolloutActor Ray actor classes.
  #3 (line ~206) The PPO training loop controller: ExperienceBatch dataclass,
                 run_ppo_training(), build_experience_batch() (GAE math).
  #8 (line ~781) Conceptual async rollout pipeline: async_ppo_loop().

Blocks intentionally SKIPPED (per task spec):
  #0 -- needs GPU (ray.remote(num_gpus=1) ModelActor, real CUDA device)
  #1 -- needs GPU (placement_group of 8 GPU bundles, NCCL process group)
  #4 -- shell (OpenRLHF launch script)
  #5 -- needs GPU (MegatronPPOTrainer -- imports nemo_aligner, a package that
        does not exist on PyPI / is not installable in this environment)
  #6 -- shell (TRT-LLM engine conversion command)
  #7 -- needs GPU (full HuggingFace Transformers PPO loop: RolloutActor /
        PolicyActor download+run "gpt2" via transformers -- both a real model
        download and >60s of CPU generation, unsuitable for this harness)

`ray` is a third-party dependency that is NOT in the guaranteed-available list
for this test suite (only numpy/torch/einops/sklearn/stdlib are guaranteed),
so the import is guarded. If `ray` is unavailable the Ray-actor exercises are
skipped explicitly and only the pure-Python logic (the GAE math in
build_experience_batch) is verified.

Real bugs found & fixed in the book:
  1. Block #2's `VLLMRolloutActor.load_weights(self, state_dict_ref)` called
     `ray.get(state_dict_ref)` internally. But Ray auto-dereferences
     top-level ObjectRef arguments passed to a `.remote()` call *before* the
     method body runs -- exactly how it's invoked in this chapter
     (`vllm_actor.load_weights.remote(params_ref)`, both here and in block
     #3's Phase 6). So `state_dict_ref` is already the materialized dict by
     the time `load_weights` runs, and calling `ray.get()` on a plain dict
     raises `ValueError: Invalid type of object refs, <class 'dict'>`.
     Reproduced live against real Ray (see below) and fixed by dropping the
     redundant `ray.get()` and using the argument directly. Mirrored below.
  2. Block #8's `async_ppo_loop(policy_actor, vllm_actor, reward_actor, prompts)`
     referenced free variables `batch_size`, `num_steps`, and `sync_every` that
     were never parameters, module globals, or locals -- calling the function as
     written raises NameError. Fixed by adding them as parameters:
     `async_ppo_loop(policy_actor, vllm_actor, reward_actor, prompts,
                      batch_size, num_steps, sync_every=1)`. Mirrored below.
"""

import sys
from dataclasses import dataclass
from queue import Queue
from typing import Dict, List, Tuple
from unittest.mock import MagicMock

import numpy as np
import torch
import torch.nn.functional as F

try:
    import ray
except Exception:
    ray = None

RAY_AVAILABLE = ray is not None


# ============================================================
# Block #2 (line ~152) -- Simplified weight-sync routine (OpenRLHF-style)
# ============================================================
if RAY_AVAILABLE:

    # NOTE: the book decorates both actors with num_gpus=1. This test runs on
    # CPU-only hardware, so we request num_gpus=0 -- the only change from the
    # book's code, and it affects Ray's resource scheduling, not any of the
    # actors' logic.
    @ray.remote(num_gpus=0)
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

    @ray.remote(num_gpus=0)
    class VLLMRolloutActor:
        def __init__(self, vllm_engine):
            self.engine = vllm_engine

        def load_weights(self, state_dict):
            """Receive new policy weights and reload into vLLM.

            Note: `state_dict` arrives already materialized. Ray
            auto-dereferences top-level ObjectRef arguments passed to a
            `.remote()` call before the method body runs, so no additional
            `ray.get()` is needed (or valid) here.
            """
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


# ============================================================
# Block #3 (line ~206) -- The PPO training loop controller
# ============================================================
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


# ============================================================
# Block #8 (line ~781) -- Conceptual async rollout pipeline
# ============================================================
# BUG FIX mirrored from the .md: the book's signature was
#   def async_ppo_loop(policy_actor, vllm_actor, reward_actor, prompts):
# but the body references batch_size / num_steps / sync_every, which were
# never defined anywhere -- a real NameError bug. Fixed by adding them as
# parameters (logic below is otherwise verbatim).
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


# ============================================================
# Test harness / glue
# ============================================================

def test_build_experience_batch_gae_math():
    """Pure-Python GAE check for block #3's build_experience_batch (no Ray)."""
    rollouts = [("p1", "r1", 0.1), ("p2", "r2", 0.2)]
    rewards = [1.0, 2.0]
    ref_logps = [0.05, 0.06]
    values = [0.5, 0.5]
    gamma, lam = 1.0, 0.95

    batch = build_experience_batch(rollouts, rewards, ref_logps, values, gamma, lam)

    # Hand-computed GAE (reverse recursion, matches block #3's loop):
    # t=1: next_val=0.0 -> delta=2.0+0-0.5=1.5   -> last_gae=1.5
    # t=0: next_val=0.5 -> delta=1.0+0.5-0.5=1.0 -> last_gae=1.0+0.95*1.5=2.425
    expected_advantages = [2.425, 1.5]

    assert isinstance(batch, ExperienceBatch)
    assert batch.prompts == ["p1", "p2"]
    assert batch.responses == ["r1", "r2"]
    assert batch.rewards == rewards
    assert batch.old_logprobs == [0.1, 0.2]
    assert batch.ref_logprobs == ref_logps
    for got, want in zip(batch.advantages, expected_advantages):
        assert abs(got - want) < 1e-9, (got, want)
    print(f"[block #3] build_experience_batch GAE OK: advantages={batch.advantages}")


def _make_mock_vllm_engine():
    """Nested MagicMock standing in for a real vLLM LLMEngine, so block #2's
    VLLMRolloutActor.load_weights runs its actual attribute-chain + assert
    logic against a canned 'load succeeded' response, offline."""
    engine = MagicMock()
    model = engine.llm_engine.model_executor.driver_worker.model_runner.model
    model.load_weights.return_value = ([], [])  # (missing, unexpected)
    return engine


def test_block2_weight_sync_actors():
    """Instantiate + exercise block #2's PolicyTrainingActor and VLLMRolloutActor."""
    tiny_model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)

    policy_actor = PolicyTrainingActor.remote(tiny_model, optimizer)
    vllm_actor = VLLMRolloutActor.remote(_make_mock_vllm_engine())

    # get_named_params(): actor pulls its own model's params onto CPU
    params_ref = policy_actor.get_named_params.remote()
    params = ray.get(params_ref)
    assert set(params.keys()) == {k for k, _ in tiny_model.named_parameters()}
    assert all(v.device.type == "cpu" for v in params.values())

    # load_weights(): vLLM actor pulls the ref from the object store and
    # loads it into the (mocked) engine's model via model.load_weights(...)
    status = ray.get(vllm_actor.load_weights.remote(params_ref))
    assert status == "weights_loaded"

    # ppo_step() is an elided stub in the book ("... pass"); confirm it still
    # executes cleanly as an actor method call.
    result = ray.get(policy_actor.ppo_step.remote(batch=None))
    assert result is None
    print("[block #2] PolicyTrainingActor + VLLMRolloutActor weight-sync OK")
    return policy_actor  # reused by block #3's Ray-orchestration test


if RAY_AVAILABLE:

    @ray.remote(num_gpus=0)
    class _ToyVLLMActor:
        """Minimal glue standing in for a real vLLM engine actor in block #3's
        orchestration test -- same .rollout()/.load_weights() interface as the
        book's VLLMRolloutActor, but without a real vLLM dependency."""
        # sampling_params defaults to None: block #8's "conceptual" async
        # pipeline calls `vllm_actor.rollout.remote(prompts[...])` with only
        # prompts (no sampling_params) -- consistent with that snippet being
        # illustrative pseudocode of the control flow, not a fully wired
        # call. Block #3's run_ppo_training (a complete example) does pass
        # sampling_params.
        def rollout(self, prompts, sampling_params=None):
            return [(p, f"response-to-{p}", 0.1 * (i + 1)) for i, p in enumerate(prompts)]

        def load_weights(self, state_dict):
            # `state_dict` is already resolved -- see the block #2 bug-fix
            # note at the top of this file re: Ray's argument
            # auto-dereferencing.
            assert isinstance(state_dict, dict)
            return "weights_loaded"

    @ray.remote(num_gpus=0)
    class _ToyRefActor:
        def log_probs(self, rollouts):
            return [0.05 * (i + 1) for i in range(len(rollouts))]

    @ray.remote(num_gpus=0)
    class _ToyRewardActor:
        def score(self, rollouts):
            return [float(len(r[1])) / 10.0 for r in rollouts]

    @ray.remote(num_gpus=0)
    class _ToyCriticActor:
        def value(self, rollouts):
            return [0.5 for _ in rollouts]

        def update(self, batch_ref):
            return {"critic_loss": 0.0}


class _TinyPromptDataset:
    def __init__(self, prompts):
        self.prompts = prompts

    def batches(self, batch_size):
        for i in range(0, len(self.prompts), batch_size):
            yield self.prompts[i:i + batch_size]


def test_block3_run_ppo_training(policy_actor):
    """Drive block #3's run_ppo_training end to end with tiny Ray actors."""
    vllm_actor = _ToyVLLMActor.remote()
    ref_actor = _ToyRefActor.remote()
    reward_actor = _ToyRewardActor.remote()
    critic_actor = _ToyCriticActor.remote()
    dataset = _TinyPromptDataset(["Tell me about X.", "Explain Y."])

    run_ppo_training(
        policy_actor=policy_actor,
        vllm_actor=vllm_actor,
        ref_actor=ref_actor,
        reward_actor=reward_actor,
        critic_actor=critic_actor,
        prompt_dataset=dataset,
        num_epochs=1,
        rollout_batch_size=2,
        ppo_epochs=1,
    )
    print("[block #3] run_ppo_training orchestration executed without error")


if RAY_AVAILABLE:

    @ray.remote(num_gpus=0)
    class _ToyAsyncPolicyActor:
        """Glue actor for block #8: needs ppo_step(rollout, rewards) and
        get_state_dict(), an interface distinct from block #2/#3's
        PolicyTrainingActor (which the book's own async snippet does not
        reuse)."""
        def ppo_step(self, rollout_ref, rewards_ref):
            return {"loss": 0.0}

        def get_state_dict(self):
            return {"w": torch.zeros(2)}


def test_block8_async_ppo_loop():
    """Exercise the (bug-fixed) block #8 async pipeline end to end."""
    policy_actor = _ToyAsyncPolicyActor.remote()
    vllm_actor = _ToyVLLMActor.remote()
    reward_actor = _ToyRewardActor.remote()
    prompts = ["p1", "p2", "p3", "p4"]

    async_ppo_loop(
        policy_actor, vllm_actor, reward_actor, prompts,
        batch_size=2, num_steps=2, sync_every=1,
    )
    print("[block #8] async_ppo_loop executed without error (bug-fixed signature)")


def main():
    # Block #3's GAE math needs no Ray at all -- always runs.
    test_build_experience_batch_gae_math()

    if not RAY_AVAILABLE:
        print("SKIP(dependency): ray is not installed -- skipping block #2, "
              "the Ray-orchestration half of block #3, and block #8 "
              "(all require ray.remote actors).")
        return

    # num_cpus is generous (rather than exactly the actor count) because each
    # @ray.remote actor reserves 1 CPU by default and this test creates
    # several small actors across blocks #2/#3/#8; too few CPUs deadlocks
    # ray.get() waiting on an actor that can never be scheduled.
    ray.init(ignore_reinit_error=True, num_cpus=8, include_dashboard=False,
              logging_level="ERROR", log_to_driver=False)
    try:
        policy_actor = test_block2_weight_sync_actors()
        test_block3_run_ppo_training(policy_actor)
        test_block8_async_ppo_loop()
    finally:
        ray.shutdown()

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
