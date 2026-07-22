"""
Runnable-code check for content/06-rl-infra/06-prime-rl-async.md

Blocks tested (verbatim from the chapter, concatenated in chapter order):
  - block #0 (line ~114): async_ppo_loss  -- pure torch, exercised on tiny tensors.
  - block #1 (line ~181): RolloutQueue / inference_worker / trainer_loop (asyncio) --
      exercised end-to-end with a tiny in-process fake inference engine, fake reward
      function, and a real (tiny) torch model/optimizer. `compute_loss` is not defined
      in the chapter (the book leaves it as "uses async_ppo_loss internally"); we supply
      a minimal, honest implementation that literally calls the book's async_ppo_loss.
  - block #2 (line ~298): toploc_commit / toploc_verify -- pure torch, exercised with an
      honest (noisy-but-consistent) recompute and an adversarial (forged) recompute.

Skipped:
  - block #3 (YAML config block, line ~358): non-Python, not applicable.
"""

import asyncio
from types import SimpleNamespace

import torch
import torch.nn as nn

# =====================================================================================
# Block #0 (chapter line ~114): async_ppo_loss
# =====================================================================================

def async_ppo_loss(
    logp_train,          # (B, T) log pi_theta(o_t | ...) recomputed by the TRAINER (current weights)
    logp_behavior,       # (B, T) log pi_theta_old(o_t | ...) LOGGED at generation by the INFERENCE engine
    advantages,          # (B,)   group-relative advantage per response (GRPO-style), detached
    response_mask,       # (B, T) 1.0 on response tokens, 0.0 on prompt/pad
    staleness,            # (B,)   integer s = k - v for each rollout (trainer_step - gen_version)
    eps_low=0.2,         # PPO clip lower
    eps_high=0.28,       # PPO clip-higher (DAPO-style asymmetric clip aids exploration)
    tis_cap=4.0,         # truncated importance-sampling cap C on the behavior-side ratio
    s_max=4,             # drop rollouts staler than this
):
    """
    Off-policy, token-level PPO surrogate for ASYNC RL.
    Returns (loss, metrics). Designed so that at s=0 and matched engines it equals on-policy GRPO.
    """
    B, T = logp_train.shape

    # --- 1. Hard staleness gate: drop rollouts older than s_max (zero their contribution) ----
    fresh = (staleness <= s_max).float().unsqueeze(1)          # (B, 1)
    mask = response_mask * fresh                                # (B, T)

    # --- 2. Per-token PPO ratio rho_t = pi_theta / pi_theta_old (computed in log-space) -------
    log_ratio = logp_train - logp_behavior                     # (B, T)
    ratio = torch.exp(log_ratio)                               # rho_t

    # --- 3. Truncated importance sampling: cap the ratio so a few outlier tokens can't blow up.
    #        We cap the *raw* ratio used as an IS weight; the clip below provides the trust region.
    ratio = torch.clamp(ratio, max=tis_cap)

    # --- 4. PPO clipped surrogate, token level. advantages broadcast across the sequence. -----
    adv = advantages.unsqueeze(1)                              # (B, 1) -> broadcast to (B, T)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high) * adv
    per_token = torch.min(unclipped, clipped)                  # (B, T)

    # --- 5. Token-level aggregation (Dr. GRPO / DAPO style): sum over all tokens, divide by ---
    #        the total number of valid tokens. NOT a per-response mean (that biases length).
    loss = -(per_token * mask).sum() / mask.sum().clamp(min=1.0)

    # --- diagnostics that you MUST watch in async runs ----------------------------------------
    with torch.no_grad():
        approx_kl = ((ratio - 1.0) - log_ratio)                # k3 estimator of KL(old||new), per token
        approx_kl = (approx_kl * mask).sum() / mask.sum().clamp(min=1.0)
        clipfrac = (((ratio < 1 - eps_low) | (ratio > 1 + eps_high)).float() * mask).sum() / mask.sum().clamp(min=1.0)
        dropped = 1.0 - fresh.mean()
    return loss, {"approx_kl": approx_kl.item(),
                  "clipfrac": clipfrac.item(),
                  "frac_dropped_stale": dropped.item(),
                  "mean_staleness": staleness.float().mean().item()}


def test_async_ppo_loss():
    torch.manual_seed(0)
    B, T = 4, 6
    logp_behavior = -torch.rand(B, T) * 2.0                 # plausible log-probs in [-2, 0]
    logp_train = logp_behavior + 0.05 * torch.randn(B, T)    # nearby policy -> small ratios
    logp_train.requires_grad_(True)
    advantages = torch.tensor([1.0, -1.0, 0.5, -0.3])
    response_mask = torch.ones(B, T)
    response_mask[0, 4:] = 0.0                                # a short response with padding
    staleness = torch.tensor([0.0, 1.0, 3.0, 9.0])            # last one exceeds s_max=4 -> dropped

    loss, metrics = async_ppo_loss(
        logp_train, logp_behavior, advantages, response_mask, staleness,
        eps_low=0.2, eps_high=0.28, tis_cap=4.0, s_max=4,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert logp_train.grad is not None
    assert metrics["frac_dropped_stale"] == 0.25  # exactly 1 of 4 rollouts is stale (s=9 > s_max=4)
    print("[block0] async_ppo_loss loss=%.4f metrics=%s" % (loss.item(), metrics))


# =====================================================================================
# Block #1 (chapter line ~181): RolloutQueue / inference_worker / trainer_loop
# =====================================================================================

class RolloutQueue:
    """A bounded async queue of finished rollouts, each stamped with its policy version."""
    def __init__(self, maxsize=4096):
        self.q = asyncio.Queue(maxsize=maxsize)

    async def put(self, rollout):     # rollout = dict(prompt, tokens, logp_behavior, reward, version)
        await self.q.put(rollout)

    async def drain_batch(self, batch_size, trainer_step, s_max):
        """Pull up to batch_size FRESH rollouts; skip (and discard) any too stale."""
        batch, attempts = [], 0
        while len(batch) < batch_size and attempts < batch_size * 8:
            r = await self.q.get()
            attempts += 1
            if trainer_step - r["version"] <= s_max:
                batch.append(r)
            # else: silently drop the stale rollout; it is off-distribution noise
        return batch

async def inference_worker(worker_id, engine, prompts, weight_box, rq: RolloutQueue, reward_fn):
    """One generator: pull a prompt, sample with CURRENT in-memory weights, score, enqueue. Repeat forever."""
    idx = 0
    while True:
        prompt = prompts[idx % len(prompts)]; idx += 1
        version = weight_box.version                      # the policy version we are about to sample under
        out = await engine.generate(prompt)               # continuous-batched sampling; returns tokens + logprobs
        reward = await reward_fn(prompt, out.text)        # verifier / reward model (may be remote & async)
        await rq.put({"prompt": prompt, "tokens": out.token_ids,
                      "logp_behavior": out.logprobs, "reward": reward,
                      "version": version})
        # Hot-swap to newer weights at this safe boundary if the trainer published some.
        if weight_box.version > version:
            await engine.load_weights(weight_box.state_dict)   # in-place; next sample uses new weights

async def trainer_loop(model, optimizer, rq: RolloutQueue, weight_box, steps, batch_size, s_max, publish_every):
    for step in range(steps):
        batch = await rq.drain_batch(batch_size, trainer_step=step, s_max=s_max)
        loss, metrics = compute_loss(model, batch, trainer_step=step)   # uses async_ppo_loss internally
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % publish_every == 0:
            weight_box.publish(model.state_dict(), version=step)        # generators pick this up lazily


# ---- minimal, honest glue: the chapter leaves `compute_loss`, the inference engine,
# ---- the weight store, and the reward function abstract ("engine", "weight_box",
# ---- "reward_fn" are parameters, not chapter code). We supply tiny concrete versions
# ---- so the async pipeline above actually runs end-to-end on CPU.

ROLLOUT_T = 6      # fixed response length used by the fake engine (keeps tensor shapes aligned)
VOCAB = 50


class TinyPolicy(nn.Module):
    """A trivial 'policy': a per-token scalar log-prob contribution from a token embedding."""
    def __init__(self, vocab_size=VOCAB):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 1)

    def forward(self, token_ids):          # (B, T) long -> (B, T) float "log-prob"
        return self.embed(token_ids).squeeze(-1)


def compute_loss(model, batch, trainer_step):
    """Glue: assembles a batch of rollout dicts into tensors and calls the book's async_ppo_loss."""
    tokens = torch.stack([r["tokens"] for r in batch])                # (B, T) long
    logp_behavior = torch.stack([r["logp_behavior"] for r in batch])  # (B, T)
    rewards = torch.tensor([r["reward"] for r in batch])
    versions = torch.tensor([r["version"] for r in batch])
    staleness = (trainer_step - versions).float()

    logp_train = model(tokens)                                        # (B, T), differentiable
    advantages = rewards - rewards.mean()
    response_mask = torch.ones_like(logp_train)

    return async_ppo_loss(
        logp_train, logp_behavior, advantages, response_mask, staleness,
        eps_low=0.2, eps_high=0.28, tis_cap=4.0, s_max=4,
    )


class FakeEngine:
    """Stand-in for a vLLM/SGLang continuous-batching engine."""
    def __init__(self, T=ROLLOUT_T, vocab=VOCAB):
        self.T = T
        self.vocab = vocab
        self.load_calls = 0

    async def generate(self, prompt):
        await asyncio.sleep(0)  # yield control, like a real async engine call
        tokens = torch.randint(0, self.vocab, (self.T,))
        logprobs = -torch.rand(self.T) * 2.0
        return SimpleNamespace(text=f"response-to-{prompt}", token_ids=tokens, logprobs=logprobs)

    async def load_weights(self, state_dict):
        await asyncio.sleep(0)
        self.load_calls += 1


class WeightBox:
    """Stand-in for the versioned weight store / broadcaster."""
    def __init__(self):
        self.version = 0
        self.state_dict = None

    def publish(self, state_dict, version):
        self.state_dict = state_dict
        self.version = version


async def fake_reward_fn(prompt, text):
    await asyncio.sleep(0)
    return float(len(text) % 5)


async def run_async_pipeline_demo():
    rq = RolloutQueue(maxsize=64)
    weight_box = WeightBox()
    model = TinyPolicy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    prompts = ["p1", "p2", "p3"]
    engine = FakeEngine()

    workers = [
        asyncio.create_task(inference_worker(i, engine, prompts, weight_box, rq, fake_reward_fn))
        for i in range(2)
    ]
    trainer_task = asyncio.create_task(
        trainer_loop(model, optimizer, rq, weight_box, steps=3, batch_size=4, s_max=4, publish_every=1)
    )

    await trainer_task

    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)

    return weight_box, engine


def test_async_pipeline():
    weight_box, engine = asyncio.run(asyncio.wait_for(run_async_pipeline_demo(), timeout=30))
    # trainer published weights every step for 3 steps -> weight_box should reflect the last step
    assert weight_box.version == 2
    assert weight_box.state_dict is not None
    print("[block1] async pipeline ran 3 trainer steps; final weight_box.version=%d, "
          "engine.load_weights calls=%d" % (weight_box.version, engine.load_calls))


# =====================================================================================
# Block #2 (chapter line ~298): toploc_commit / toploc_verify
# =====================================================================================

def toploc_commit(hidden_states, k=128):
    """
    PROVER side (runs on the untrusted worker during generation).
    hidden_states: (T, d) activations at the committed layer for the generated tokens.
    Returns a compact commitment: the top-k magnitude components per token.
    The real scheme adds a cryptographic hash + polynomial encoding over these;
    this captures the locality-sensitive core that makes it robust yet discriminative.
    """
    T, d = hidden_states.shape
    vals, idx = torch.topk(hidden_states.abs(), k=k, dim=-1)        # (T, k) largest-|h| components
    signs = torch.sign(torch.gather(hidden_states, 1, idx))         # keep sign of each
    # Quantize values coarsely so benign low-bit noise doesn't change the commitment.
    q_vals = (vals * 16).round().to(torch.int16)                    # coarse magnitude buckets
    return {"idx": idx.to(torch.int32), "qval": q_vals, "sign": signs.to(torch.int8)}

def toploc_verify(commitment, recomputed_hidden, k=128, tol_frac=0.90, mag_tol=2):
    """
    VERIFIER side (trusted). recomputed_hidden: (T, d) from a CHEAP batched prefill
    on trusted hardware over the SAME claimed tokens. Returns True if the worker's
    commitment is consistent with an honest run of THIS model, within numeric tolerance.
    """
    ref = toploc_commit(recomputed_hidden, k=k)
    T = recomputed_hidden.shape[0]
    passes = 0
    for t in range(T):
        # How many of the prover's top-k indices also appear in our top-k for this token?
        shared = len(set(commitment["idx"][t].tolist()) & set(ref["idx"][t].tolist()))
        # And do the shared components' coarse magnitudes roughly agree?
        ok_idx = shared / k >= tol_frac
        # (a full check also compares q_vals on the shared indices within mag_tol buckets)
        passes += int(ok_idx)
    return passes / T >= tol_frac     # accept only if MOST tokens are consistent


def test_toploc():
    torch.manual_seed(0)
    T, d, k = 10, 256, 16

    hidden_states = torch.randn(T, d)
    commitment = toploc_commit(hidden_states, k=k)
    assert commitment["idx"].shape == (T, k)

    # Honest case: benign hardware/kernel noise -> top-k structure should still agree.
    noisy = hidden_states + torch.randn(T, d) * 1e-4
    honest_ok = toploc_verify(commitment, noisy, k=k)
    assert honest_ok is True

    # Adversarial case: a completely different activation tensor (forged/wrong model)
    # should NOT reproduce the same dominant top-k structure.
    forged = torch.randn(T, d) * 10.0
    forged_ok = toploc_verify(commitment, forged, k=k)
    assert forged_ok is False

    print("[block2] toploc: honest_ok=%s forged_ok=%s" % (honest_ok, forged_ok))


# =====================================================================================
# Block #3 (chapter line ~358): YAML config
# SKIP(non-python): this block is a YAML config snippet, not Python code.
# =====================================================================================


if __name__ == "__main__":
    test_async_ppo_loss()
    test_async_pipeline()
    test_toploc()
    print("OK: all runnable blocks in 06-rl-infra/06-prime-rl-async.md executed successfully.")
