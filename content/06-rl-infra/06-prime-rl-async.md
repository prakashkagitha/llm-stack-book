# 6.6 Prime-RL, Async RL & Decentralized Training

Every RL-for-LLM system we have built so far has a hidden synchronization barrier at its heart. In [The Generation–Training Loop & Rollout Engines](../06-rl-infra/02-generation-training-loop.html) we drew the loop as a strict alternation: the policy generates a batch of rollouts, *everyone waits*, the trainer computes a gradient step, *everyone waits* for the new weights, and only then does generation resume. That barrier is the single largest source of wasted GPU-hours in the entire stack. Generation is autoregressive, memory-bound, and dominated by a long tail of slow samples; training is compute-bound and bursty. When you couple them on the same clock, the trainer sits idle while the slowest rollout finishes, and the generators sit idle while the optimizer steps. On a real reasoning-RL run where responses range from 200 to 16,000 tokens, this *straggler* effect alone can leave half your fleet idle.

This chapter is about breaking that barrier. We will make generation and training run as **decoupled, continuously-busy services** that exchange rollouts and weights over a queue rather than a lock-step. Doing so turns on-policy RL into *off-policy* RL — the data a gradient step consumes was produced by a slightly older policy — which forces us to confront **staleness** and to import **importance-sampling corrections** from the policy-gradient literature. We will then push the idea to its logical extreme: if generation and training need only exchange a queue of rollouts and an occasional weight broadcast, the two can live in *different data centers*, on *different continents*, on *heterogeneous, untrusted hardware*. That is the bet Prime Intellect makes with **prime-rl** and the **INTELLECT** models, and it requires one more ingredient we will build carefully: a cryptographic-style **proof that a remote, untrusted GPU actually ran the inference it claims to have run** — TOPLOC.

This is the async frontier of RL infrastructure. It pairs with [Colocated vs Disaggregated RL & Weight Synchronization](../06-rl-infra/07-colocated-vs-disaggregated.html) (the placement question), [Advantage Estimation, KL Control & Stability Tricks](../06-rl-infra/09-advantage-kl-tricks.html) (the estimator math), and [Scaling RL: Throughput, Load Balancing & The Latest Tricks](../06-rl-infra/11-scaling-rl-tricks.html) (the throughput view).

## The synchronous barrier and why it wastes the fleet

### Anatomy of the stall

Recall the synchronous RL step. Let $T_{\text{gen}}$ be the wall-clock time to generate one batch of rollouts and $T_{\text{train}}$ the time for the forward/backward/optimizer step on those rollouts. In a colocated system (generation and training share the GPUs) the two phases are serialized on the *same* devices, so the step time is simply

$$
T_{\text{step}}^{\text{sync}} = T_{\text{gen}} + T_{\text{train}} + T_{\text{sync}},
$$

where $T_{\text{sync}}$ is the cost of moving updated weights from the trainer into the inference engine's memory (a `load_weights` into vLLM/SGLang; see [vLLM: Architecture, PagedAttention & Internals](../07-inference-serving/03-vllm-internals.html)). The problem is not the sum — it is that during $T_{\text{gen}}$ the *training* kernels are idle and during $T_{\text{train}}$ the *generation* engine is idle. You bought one set of GPUs and are using it for one job at a time.

In a *disaggregated* system (separate generator and trainer pools) you can at least run them on different hardware, but the synchronous barrier still couples their clocks: the trainer cannot start step $k$ until the generators finish producing batch $k$, and the generators cannot start batch $k+1$ until they receive weights $\theta_k$. The two pools ping-pong, and at any instant exactly one is working.

### The straggler tax

Autoregressive generation makes this far worse than the naïve picture. The time to generate a response is roughly linear in its token length, and reasoning-RL response-length distributions are heavy-tailed. If a batch of $B$ prompts produces responses whose lengths $\ell_i$ vary widely, the batch is not done until the *longest* one finishes:

$$
T_{\text{gen}} \approx t_{\text{tok}} \cdot \max_i \ell_i,
$$

even though the *mean* work is $t_{\text{tok}} \cdot \frac{1}{B}\sum_i \ell_i$. With a long tail, $\max_i \ell_i$ can be $5\!-\!20\times$ the mean. The generators that finished their short samples early sit idle, waiting on one or two outliers, and the trainer waits on all of them.


{{fig:primerl-sync-vs-async-timeline}}


The fix is conceptually simple: **let the generators keep generating while the trainer trains.** Don't synchronize on every step. Push fresh weights to the generators *occasionally* and let them continue producing rollouts from whatever weights they currently hold. This is asynchronous, off-policy RL — and it is the dominant paradigm for large-scale reasoning RL in 2025.

## Decoupling generation from training: async off-policy RL

### The queue-based architecture

The async design replaces the barrier with a **rollout queue** and a **weight-update channel**. Three logical components run concurrently:

1. **Inference workers** (generators): a pool of vLLM/SGLang engines that pull prompts, sample completions with the *current* policy weights they hold, compute rewards (or ship completions to a verifier), and push finished rollouts onto the queue. They never block on the trainer.
2. **Trainer**: pulls a batch of rollouts off the queue, computes the loss, steps the optimizer, and — every $N$ steps — publishes new weights to a versioned store.
3. **Weight broadcaster**: pushes published weights into the inference workers. A worker swaps to the new weights at a safe boundary (between requests, or via an atomic in-place update) and tags every subsequent rollout with the **policy version** it was generated under.


{{fig:primerl-queue-async-architecture}}


The single most important new piece of bookkeeping is the **policy version stamped on every rollout**. A rollout generated under weights $\theta_v$ but used to update weights $\theta_k$ (with $k > v$) is **stale** by $s = k - v$ steps. Staleness is the central quantity of this chapter; everything else is engineering around keeping it bounded and correcting for it.

### Staleness, the off-policy gap, and the async hyperparameter

Define the **maximum allowed staleness** $s_{\max}$ (often called `async_level` or `max_off_policy_age`). The trainer refuses to consume any rollout older than $s_{\max}$ steps, and the inference workers throttle so they don't run more than $s_{\max}$ versions ahead of the trainer. Two limits anchor the design space:

- $s_{\max} = 0$: fully synchronous, on-policy. The generators must use *exactly* the weights the trainer just produced. Maximum statistical fidelity, maximum idle time.
- $s_{\max} = \infty$: fully off-policy. Generators run an arbitrarily old policy. Maximum throughput, but the data may be so off-distribution that the gradient is meaningless.

The interesting regime is small but nonzero, typically $s_{\max} \in \{1,2,4\}$. The classic empirical result, established by the **AReaL** (Ant Research) and related async-RL systems and echoed across the field, is that **one step of staleness is essentially free** — the policy moves so little per step that $\theta_{k-1}$ and $\theta_k$ induce nearly identical sampling distributions — and even moderate staleness is tolerable *if you apply an importance-sampling correction*. The throughput win is large: published async systems report on the order of **2–4× end-to-end speedups** over the synchronous baseline at matched final quality.

There is a subtle but critical degenerate case to design out, sometimes called the **one-step-staleness trap**. If you naïvely allow exactly one batch in flight, a single straggler in the *generation* of batch $k+1$ can still stall the trainer once it exhausts batch $k$. Robust systems therefore decouple at the *sample* level, not the batch level: the queue holds individual finished rollouts, the trainer assembles a training batch from whatever is available, and inference workers continuously refill. This is the **pipelined async rollout** pattern (next section).

### Why off-policy needs a correction: the distribution mismatch

Under the hood, async RL is plain off-policy policy-gradient. We want the gradient of the expected reward under the *current* policy $\pi_\theta$, but our samples came from an *older behavior policy* $\pi_{\theta_{\text{old}}}$ (the one the generator held). The score-function estimator is only unbiased if samples come from the policy being differentiated. To fix the mismatch we reweight each sample by the **importance ratio** between the target and behavior policies — exactly the ratio at the heart of PPO (see [Policy Gradients & PPO for Language Models](../05-posttraining-alignment/06-ppo-for-llms.html)).

## The math of staleness: importance sampling for off-policy LLM RL

### The token-level importance ratio

For language models the action at each step is a token, so the natural unit is the per-token importance ratio. For response $o = (o_1,\dots,o_{|o|})$ generated under behavior policy $\pi_{\theta_{\text{old}}}$ and being optimized under target policy $\pi_\theta$, define

$$
\rho_t(\theta) \;=\; \frac{\pi_\theta(o_t \mid q, o_{<t})}{\pi_{\theta_{\text{old}}}(o_t \mid q, o_{<t})}.
$$

The importance-weighted policy-gradient surrogate (the objective whose gradient is the off-policy REINFORCE gradient) is

$$
J^{\text{IS}}(\theta) \;=\; \mathbb{E}_{o\sim\pi_{\theta_{\text{old}}}}\!\Big[\sum_{t} \rho_t(\theta)\, \hat A_t \Big],
$$

where $\hat A_t$ is the advantage (a group-relative score for GRPO; see [GRPO, RLOO & Critic-Free RL](../05-posttraining-alignment/08-grpo-rloo.html)). When $\theta = \theta_{\text{old}}$ every $\rho_t = 1$ and this collapses to the on-policy objective — exactly what you want at zero staleness.

The danger of importance sampling is **variance explosion**. The ratio is a product of per-token ratios over the whole sequence; for a 4,000-token response, even tiny per-token discrepancies compound multiplicatively. A sequence-level ratio $\prod_t \rho_t$ can be astronomically large or small, and its variance is unbounded. Three standard defenses, in increasing sophistication:

**1. PPO clipping.** Replace the raw ratio with the clipped surrogate. This is *the same machinery* that makes PPO multi-epoch-safe, now doing double duty as a staleness corrector:

$$
J^{\text{clip}}(\theta) = \mathbb{E}\Big[\sum_t \min\big(\rho_t \hat A_t,\; \operatorname{clip}(\rho_t, 1-\varepsilon, 1+\varepsilon)\,\hat A_t\big)\Big].
$$

Clipping caps how far the target policy can move per step on any single token, bounding the per-token contribution to $[1-\varepsilon, 1+\varepsilon]\cdot|\hat A_t|$. This is the workhorse correction; for $s=1$ staleness it is usually sufficient by itself.

**2. Truncated importance sampling (TIS).** Clipping the *PPO* ratio is not enough when the gap between the generation engine and the training engine is large or systematic. A second ratio appears in async/disaggregated RL that practitioners often miss: the **inference engine** (vLLM, in fp16/fp8 with fused kernels) and the **training engine** (PyTorch FSDP, possibly bf16 with different kernels) do **not** produce numerically identical log-probs for the same tokens and weights. The behavior log-prob you logged at generation time, $\pi_{\theta_{\text{old}}}^{\text{infer}}$, differs from what the trainer would compute, $\pi_{\theta_{\text{old}}}^{\text{train}}$. Truncated importance sampling caps the behavior-side ratio at a constant $C$:

$$
w_t = \min\!\left(\frac{\pi_\theta^{\text{train}}(o_t)}{\pi_{\theta_{\text{old}}}^{\text{infer}}(o_t)},\; C\right), \qquad C \in [2, 10] \text{ typically.}
$$

This single correction, highlighted by veRL/AReaL practitioners, is frequently the difference between a stable async run and one that silently collapses — the engine-mismatch bias is real and accumulates.

**3. Sequence-level masking / regularized IS.** Some systems (e.g. GSPO-style sequence-level objectives) avoid the multiplicative product entirely by using a *geometric-mean* or length-normalized sequence ratio, or by simply **dropping** rollouts whose sequence-level ratio falls outside a trust band. Dropping is crude but robust: a rollout that is wildly off-policy contributes more noise than signal, so discard it.

### A from-scratch async off-policy loss

Here is a compact, correct implementation of the corrected loss. It takes log-probs from *both* the behavior policy (logged at generation, from the inference engine) and the current policy (recomputed by the trainer), applies PPO clipping plus truncated importance sampling, and masks stale or degenerate samples.

```python
import torch

def async_ppo_loss(
    logp_train,          # (B, T) log pi_theta(o_t | ...) recomputed by the TRAINER (current weights)
    logp_behavior,       # (B, T) log pi_theta_old(o_t | ...) LOGGED at generation by the INFERENCE engine
    advantages,          # (B,)   group-relative advantage per response (GRPO-style), detached
    response_mask,       # (B, T) 1.0 on response tokens, 0.0 on prompt/pad
    staleness,           # (B,)   integer s = k - v for each rollout (trainer_step - gen_version)
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
```

Three design notes worth internalizing. First, **you must recompute `logp_train` in the trainer** — you cannot reuse the inference engine's log-probs as the numerator, because then `log_ratio` would mix two engines' numerics on *both* sides and the gradient would be wrong (the numerator must be differentiable w.r.t. $\theta$). Second, **the behavior log-prob is logged once at generation** and travels with the rollout through the queue; storing it is cheap (one float per token) and essential. Third, the **clip-higher** asymmetry (`eps_high > eps_low`, from DAPO) matters more in async settings because off-policy data tends to be lower-entropy; the higher upper clip preserves exploration. See [Advantage Estimation, KL Control & Stability Tricks](../06-rl-infra/09-advantage-kl-tricks.html) for the KL-control side.

!!! warning "Common pitfall: the silent engine-mismatch bias"
    The most insidious async bug is *not* staleness — it is the numerical gap between your inference engine and your training engine computing log-probs for the *same* weights and tokens. vLLM in fp8 with paged kernels and your FSDP trainer in bf16 will disagree, sometimes by enough that the importance ratio is systematically biased away from 1 even at zero staleness. Symptoms: a nonzero, *drifting* `approx_kl` on the very first epoch over fresh rollouts; reward that climbs then collapses. Always log both log-probs and monitor their mean absolute difference. Truncated importance sampling (`tis_cap`) and recomputing the trainer-side log-prob are the fixes; never assume the two engines agree.

## Pipelined async rollouts: keeping every GPU busy

### Sample-level streaming instead of batch-level barriers

The decisive engineering move is to stream at the granularity of *individual rollouts*. Instead of "generate batch $k$, then train on batch $k$," the inference workers run a continuous loop, and the trainer assembles each training batch from whatever finished rollouts are sitting in the queue. Long stragglers no longer block short samples — a worker that finishes a 200-token response immediately starts the next prompt; the 16,000-token outlier finishes whenever it finishes and joins a *later* training batch (with its staleness stamped accordingly).

This requires the inference engine to support **continuous batching** (in-flight requests of different lengths, new prompts admitted as slots free up; see [Continuous Batching & Request Scheduling](../07-inference-serving/02-continuous-batching.html)). vLLM and SGLang provide exactly this, which is why they are the standard rollout engines for async RL.

```python
import asyncio, time
from collections import deque

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
```

`weight_box` is the versioned weight store. Its `publish` is the broadcaster's job; in a single-node setup it is a shared object, but at scale it is a sharded broadcast — discussed in [Colocated vs Disaggregated RL & Weight Synchronization](../06-rl-infra/07-colocated-vs-disaggregated.html). The key property of this loop: **neither the trainer nor any worker ever blocks on the other.** Workers always have weights to sample with; the trainer always has rollouts to consume. The only coupling is the soft staleness gate.

### Throughput bookkeeping

How fast can the trainer go before it starves? Let the inference fleet produce rollouts at rate $r_{\text{gen}}$ (rollouts/sec) and let the trainer consume a batch of size $B$ in $T_{\text{train}}$ seconds. The trainer is *generation-bound* if $B / T_{\text{train}} > r_{\text{gen}}$ (it eats faster than the fleet produces) and *training-bound* otherwise. The async sweet spot provisions the fleet so the two rates are close, with a small generation surplus to absorb straggler variance. When generation-bound, add inference workers (cheap, embarrassingly parallel); when training-bound, the trainer is the bottleneck and async buys you nothing — you are already saturated.

!!! example "Worked example: how much does async actually save?"
    Take a 7B reasoning-RL run. Suppose for a batch of $B=512$ prompts:

    - Mean response length is 1,200 tokens, but the **max** in a batch is ~9,000 tokens (heavy tail).
    - Inference throughput per worker is ~3,000 tok/s; you have 16 inference workers → ~48,000 tok/s aggregate.
    - Training step (forward+backward+optimizer on the 512 rollouts) takes $T_{\text{train}} = 14\text{ s}$.

    **Synchronous step time.** Generation must wait for the longest sample. Even with continuous batching across 16 workers, the batch is not "done" until the 9,000-token straggler finishes. Total response tokens $\approx 512 \times 1200 = 614{,}400$, but the *critical path* is set by the straggler queued behind others. A realistic synchronous gen time is dominated by the tail; call it $T_{\text{gen}}^{\text{sync}} \approx 614{,}400 / 48{,}000 \approx 12.8\text{ s}$ of useful work, but tail effects stretch wall-clock to ~$18\text{ s}$. Plus weight sync $T_{\text{sync}} \approx 2\text{ s}$. So

    $$
    T_{\text{step}}^{\text{sync}} \approx 18 + 14 + 2 = 34\text{ s}, \quad\text{GPU utilization} \approx \frac{14}{34} \approx 41\% \text{ (trainer)}.
    $$

    **Async step time.** The trainer never waits: while it spends $14\text{ s}$ on step $k$, the 16 workers produce $48{,}000 \times 14 = 672{,}000$ tokens $\approx 560$ rollouts — *more* than the 512 it needs. The trainer is generation-fed continuously, so

    $$
    T_{\text{step}}^{\text{async}} \approx \max(T_{\text{train}},\, B/r_{\text{gen}}) \approx \max(14, \; 614{,}400/48{,}000) \approx \max(14, 12.8) = 14\text{ s}.
    $$

    **Speedup** $\approx 34 / 14 \approx 2.4\times$, with trainer utilization near 100%. The straggler tax and the weight-sync stall both vanish because nothing is on the critical path but the trainer's own compute. This is exactly the 2–4× regime async systems report — and note the savings come *entirely* from eliminating idle time, not from any algorithmic change.

## Prime Intellect's prime-rl and decentralized RL

### What changes when the workers leave the data center

Everything so far assumed your inference workers and trainer share a fast interconnect (NVLink/InfiniBand inside one cluster). **prime-rl**, Prime Intellect's RL framework, takes the async architecture and asks: what if the inference workers are scattered across the *internet* — community GPUs, spot instances in different clouds, machines you neither own nor trust? The async, queue-based design is precisely what makes this feasible, because the only thing that crosses the slow, unreliable wide-area link is (a) prompts going out, (b) rollouts coming back, and (c) occasional weight broadcasts. None of these is on a tight latency-critical loop.

Three properties of async RL make decentralization tractable where synchronous RL would be hopeless:

1. **Loose coupling tolerates latency.** A rollout that takes an extra 500 ms to traverse the WAN just arrives a little later and carries slightly more staleness — which the importance-sampling correction already handles. A *synchronous* barrier across the internet would be catastrophic; an async queue barely notices.
2. **Generation is the parallel, fault-tolerant part.** Inference workers are stateless w.r.t. each other. One dropping offline mid-rollout costs you one rollout. The trainer — the stateful, hard-to-replicate part — stays centralized on reliable hardware. This is the same disaggregation logic as [Colocated vs Disaggregated RL & Weight Synchronization](../06-rl-infra/07-colocated-vs-disaggregated.html), pushed to a global scale.
3. **Weight broadcast is infrequent and one-way.** Publishing weights every $N$ steps over the internet is a bandwidth problem, not a latency problem, and it is solvable with sharding, quantized weight deltas, and BitTorrent-style fan-out.

This is the lineage of the **INTELLECT** models. **INTELLECT-1** (a ~10B-parameter base model) demonstrated globally-distributed *pretraining* across continents using Prime Intellect's **OpenDiLoCo** (an open implementation of DeepMind's DiLoCo — Distributed Low-Communication training), which performs many *local* optimizer steps between rare global synchronizations to slash communication. **INTELLECT-2** then applied the same decentralized philosophy to **RL**: globally-distributed, asynchronous reinforcement learning for a reasoning model, where permissionless, geographically-spread inference nodes contribute rollouts. prime-rl is the framework that orchestrates that RL.


{{fig:primerl-decentralized-trust-pipeline}}


### The new problem: you cannot trust the workers

Centralized async RL trusts its inference workers implicitly — they are your own GPUs. The moment workers are *permissionless* (anyone can join and contribute rollouts, possibly for a reward/incentive), you inherit two adversarial problems that do not exist inside a single trusted cluster:

1. **Reward forgery.** A worker could claim its completion solved the math problem (reward = 1) when it didn't. *This* is handled by running the **verifier** on trusted hardware: the verifier independently re-checks the answer (re-runs the unit tests, re-evaluates the math equivalence). The worker sends the *completion*, not the *reward*; the trusted side computes the reward. See [Reward Engineering, Verifiers & Sandboxes](../06-rl-infra/08-reward-verifiers-sandboxes.html).
2. **Inference forgery.** Far subtler: a worker could *lie about what it generated*. It could run a *smaller/cheaper* model, or a different model entirely, or just fabricate tokens, then submit a plausible-looking completion. Re-verifying the reward catches a wrong *answer*, but a forged *correct-looking* completion that was never actually produced by the policy poisons the training distribution — the trainer would update toward tokens the policy never assigned probability to. We need a way to prove that **this specific completion was generated by this specific model on this specific input.** That is what TOPLOC provides.

## TOPLOC: verifying that an untrusted GPU really ran the model

### The verification problem, precisely

The naïve verification of an inference claim is to **re-run it**: take the prompt and the claimed output tokens, run a forward pass on trusted hardware, and check that the model assigns them the probabilities the worker claimed. But a full re-run on trusted hardware is as expensive as generating in the first place — it defeats the entire purpose of offloading generation to cheap external GPUs. We need verification that is **much cheaper than generation**, yet still catches a worker that swapped the model, changed the precision, or fabricated tokens.

The core difficulty is *non-determinism*. Re-running the exact same model on the exact same tokens on *different* hardware does **not** give bit-identical activations: GPU floating-point reductions are non-associative, kernels differ across vLLM versions and GPU architectures, and tensor-parallel reductions reorder sums. So you cannot just demand a hash of the activations to match — an honest worker on an A100 and an honest verifier on an H100 would disagree, yet both are legitimate. The verification scheme must be **robust to benign numerical noise** while still **sensitive to adversarial changes** (a different model, wrong precision, fabricated tokens).

### How TOPLOC works (the mechanism)

**TOPLOC** (Prime Intellect's *Locality-sensitive hashing for inference verification*) solves this with a compact, **locality-sensitive commitment** to the model's intermediate activations. The intuition is to commit not to the exact floating-point activations (too brittle) but to a *robust fingerprint* of them that survives benign perturbation but breaks under a real model change. The mechanism, at a useful level of detail:

1. **The prover (worker)** runs the forward pass during generation. For the committed layer(s), it computes a small set of features from the activations — concretely, it identifies the **top-$k$ activation values and their indices** for each token's hidden state (the largest-magnitude components dominate the geometry and are the most stable), and forms a compact commitment over them. This commitment is tiny — on the order of bytes per token, not the full hidden state — and ships alongside the completion.
2. **The verifier (trusted)** takes the prompt and the claimed output tokens and runs a forward pass *in a single batched prefill* — crucially, this is **far cheaper than autoregressive generation**, because it processes all tokens in parallel (one big matmul) rather than one-at-a-time with a growing KV cache. The verifier recomputes the same top-$k$ activation features and checks them against the prover's commitment.
3. **The robust comparison.** The verifier does not demand exact equality. It checks that the prover's top-$k$ indices/values **agree within a tolerance**: the same large components show up in roughly the same places with roughly the same magnitudes. Benign hardware/kernel noise perturbs the low bits and occasionally swaps near-ties in the ranking, but the *dominant structure* is preserved. A different model, a different precision (fp8 vs bf16), or fabricated tokens shifts the activation geometry enough that the top-$k$ structure diverges beyond tolerance — the proof fails.

```python
import torch

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
```

The economics are what make it work: the **commitment is tiny** (top-$k$ ints, a few hundred bytes per token at most, often compressed much further), and **verification is a single batched prefill** — cheaper than the original autoregressive generation by the ratio of "one parallel forward over $T$ tokens" to "$T$ sequential forwards with KV-cache growth." So the trainer can afford to verify **every** rollout (or a random audited subset) before admitting it to the training stream, at a cost that is a small fraction of generation. A worker that submits a forged completion fails the check and its rollout is rejected (and, in an incentivized network, the worker is penalized).

!!! note "Aside: why locality-sensitive, not cryptographic, hashing"
    A normal cryptographic hash (SHA-256) of the activations is useless here: flip one low bit and the hash is completely different, so an honest worker on different hardware would always fail. TOPLOC's insight is to hash a *quantity that is stable under benign perturbation* — the identity and coarse magnitude of the dominant activation components — so the commitment is **locality-sensitive**: nearby activation tensors produce consistent commitments, distant ones (different model/precision) do not. It trades the exactness of a cryptographic proof for *robust statistical confidence* that is appropriate for floating-point ML, and that is exactly the right tradeoff for verifying inference on heterogeneous GPUs.

### Putting it together: the prime-rl trust pipeline

A rollout from a permissionless worker is admitted to training only after passing **two** independent trusted checks:

1. **TOPLOC** confirms the *completion was genuinely produced by the current policy* on the given prompt (anti-inference-forgery).
2. The **verifier** independently recomputes the *reward* from the completion (anti-reward-forgery), per [Reward Engineering, Verifiers & Sandboxes](../06-rl-infra/08-reward-verifiers-sandboxes.html).

Only then does the rollout — with its behavior log-probs, reward, and **staleness stamp** — join the queue feeding the async off-policy loss we built earlier. Staleness is generally *larger* here than in a single cluster (weight broadcasts over the WAN are infrequent), so the importance-sampling and truncated-IS corrections do real, load-bearing work. The full system is async off-policy RL (Sections 2–4) plus a trust layer (Sections 5–6).

!!! interview "Interview Corner"
    **Q:** You're designing globally-distributed, permissionless RL: a central trainer plus thousands of untrusted inference GPUs across the internet. Walk me through the two things that break versus a single-cluster async setup, and how you'd fix each.

    **A:** Two categories break. **(1) Statistical / systems:** the WAN makes weight broadcast slow and infrequent, so rollouts arrive *staler* and from a possibly-different inference engine. I keep the queue-based async architecture (nothing latency-critical crosses the WAN — only prompts out, rollouts in, periodic sharded weight broadcasts), stamp every rollout with its policy version, gate on a max-staleness $s_{\max}$, and correct the off-policy gap with the PPO clipped ratio plus **truncated importance sampling** to bound the engine-mismatch ratio. I log both behavior (inference-engine) and recomputed (trainer) log-probs and watch `approx_kl` and `clipfrac`. **(2) Trust / adversarial:** workers are untrusted, so they can forge the *reward* or forge the *inference itself*. Reward forgery I kill by never trusting a worker-reported reward — the trusted verifier recomputes it from the completion. Inference forgery (running a smaller model, wrong precision, or fabricating tokens) I catch with **TOPLOC**: each worker ships a tiny locality-sensitive commitment to its top-$k$ activations; the trusted side re-verifies with a single *cheap batched prefill* (much cheaper than generation) and accepts only if the dominant activation structure matches within tolerance — robust to benign GPU numerical noise but sensitive to a real model swap. A rollout enters training only after passing both checks. I keep the *trainer* (stateful, hard to replicate) centralized on reliable hardware and push only the *stateless, fault-tolerant generation* to the edge.

## Practical configuration and failure modes

A real prime-rl-style async run lives or dies by a handful of knobs and monitors. The config below is representative (not a verbatim copy of any repo) and annotated with *why* each value matters.

```yaml
# Async off-policy RL — representative configuration
async:
  s_max: 4                 # max staleness (steps). 1-2 for tight clusters; up to 8 over WAN.
  batch_size: 512          # training batch assembled from the rollout queue
  publish_weights_every: 1 # broadcast new weights every N trainer steps (raise over slow links)
  queue_maxsize: 4096      # backpressure: workers throttle if the queue is full (too far ahead)

importance_sampling:
  ppo_eps_low: 0.2
  ppo_eps_high: 0.28       # clip-higher (DAPO): preserves exploration on lower-entropy off-policy data
  tis_cap: 4.0             # truncated importance sampling on the behavior-side ratio (engine mismatch)
  drop_if_seq_ratio_gt: 8  # discard rollouts whose sequence-level ratio is wildly off-policy

loss:
  aggregation: token_level # Dr. GRPO / DAPO: sum over tokens / total tokens (NOT per-response mean)
  kl_coef: 0.0             # R1-style runs often drop the explicit KL; clip provides the trust region

decentralized:             # only when workers are permissionless / remote
  verify_inference: toploc # require a TOPLOC proof per rollout
  toploc_topk: 128
  recompute_reward: true   # never trust a worker-reported reward; verifier recomputes it
  audit_fraction: 1.0      # fraction of rollouts to verify (1.0 = all; lower to save verifier compute)
```

The dashboard you watch, in priority order:

| Metric | Healthy | Red flag | What it means |
|---|---|---|---|
| `mean_staleness` | $\le s_{\max}$, stable | climbing | trainer is generation-starved or workers fell behind |
| `frac_dropped_stale` | low, steady | rising | $s_{\max}$ too tight or fleet too slow; add workers / raise $s_{\max}$ |
| `approx_kl` (fresh data) | $\approx 0$ | nonzero & drifting | **engine mismatch** — trainer vs inference log-probs disagree |
| `clipfrac` | moderate (5–25%) | $>50\%$ | data too off-policy; lower $s_{\max}$ or publish weights more often |
| `toploc_reject_rate` | low | spiking | a worker (or cohort) is forging inference; quarantine it |
| trainer GPU util | $\approx 100\%$ | low | async isn't helping — you're training-bound, not generation-bound |

!!! tip "Practitioner tip: start synchronous, then turn the staleness dial up"
    Bring up your RL run **fully synchronous** ($s_{\max}=0$) first and confirm reward climbs and `approx_kl` on fresh data is essentially zero. *Then* introduce async by raising $s_{\max}$ to 1, then 2, watching that reward curves overlay the sync baseline. This isolates async/staleness bugs from ordinary RL bugs: if the sync run is healthy and the $s_{\max}=1$ run diverges, your importance-sampling correction or engine-mismatch handling is wrong — not your reward, advantage, or learning rate. Debugging async-and-RL simultaneously from a cold start is how teams lose a week.

!!! warning "Common pitfall: backpressure deadlock and version skew"
    Two async-specific footguns. **(1) No backpressure:** if inference workers race arbitrarily far ahead of the trainer, the queue fills with rollouts that are *already* staler than $s_{\max}$ by the time the trainer reaches them — you generate a mountain of garbage that gets dropped. Bound the queue and have workers *block* (or slow their prompt intake) when it is full, so generation rate self-throttles to the trainer's consumption rate. **(2) Version skew on weight swap:** if a worker hot-swaps weights *mid-response*, the first half of the completion came from $\theta_v$ and the second half from $\theta_{v+1}$, and your behavior log-probs are now inconsistent — the importance ratio is meaningless. Only ever swap weights at a *request boundary* (between completions), never mid-generation, and stamp the rollout with the single version that produced it end-to-end.

## Where the frontier is heading

Async off-policy RL is now the default for serious reasoning-RL at scale: veRL, AReaL, SLIME, and prime-rl all ship asynchronous pipelines, and the open question has shifted from "*should* we go async" to "*how much* staleness can we absorb and *how* do we correct for it." The research edges, briefly:

- **Bigger staleness budgets with smarter corrections.** Sequence-level importance weighting (GSPO-style), adaptive $s_{\max}$ that grows as the policy stabilizes, and per-token confidence weighting to down-weight high-variance ratios.
- **Fully disaggregated, elastic fleets.** Inference and training pools that scale independently and tolerate node churn — the topic of [Colocated vs Disaggregated RL & Weight Synchronization](../06-rl-infra/07-colocated-vs-disaggregated.html) and [Scaling RL: Throughput, Load Balancing & The Latest Tricks](../06-rl-infra/11-scaling-rl-tricks.html).
- **Cheaper, stronger inference verification.** TOPLOC is one point in a design space that includes trusted execution environments (TEEs) and zero-knowledge ML proofs; the prize is verification cost approaching zero with cryptographic-grade guarantees.
- **Permissionless training as an economic system.** INTELLECT-style networks turn idle global GPU capacity into a substrate for frontier RL, with incentive design (who gets paid for a verified rollout) becoming as important as the optimizer.

The throughline of the whole chapter: **the async barrier-break is a systems idea, but it forces an algorithmic correction (off-policy importance sampling) and, at global scale, a trust mechanism (TOPLOC).** Get all three right and you can run frontier RL on hardware you neither own nor trust.

!!! key "Key Takeaways"
    - **The synchronous RL barrier wastes the fleet.** Lock-step generation→train→sync leaves the trainer idle during generation and the generators idle during training; heavy-tailed response lengths add a brutal *straggler tax* where the batch waits on its single longest sample.
    - **Async off-policy RL decouples the two via a rollout queue.** Generators sample continuously with whatever weights they hold; the trainer consumes rollouts and publishes weights every $N$ steps. Each rollout is stamped with a **policy version**, and **staleness** $s = k - v$ becomes the central quantity.
    - **Off-policy means importance sampling.** Reweight each token by $\rho_t = \pi_\theta/\pi_{\theta_{\text{old}}}$. Tame the variance with **PPO clipping** (the per-step trust region), **truncated importance sampling** (caps the behavior-side ratio — critical because the inference and training engines compute different log-probs), and dropping wildly off-policy rollouts.
    - **One step of staleness is nearly free; $s_{\max}\in\{1,2,4\}$ buys ~2–4× throughput** at matched quality. The win is *eliminating idle time*, not any algorithmic change.
    - **Pipeline at the sample level, not the batch level.** Stream individual finished rollouts through a bounded queue (with backpressure) so stragglers never block short samples; swap weights only at request boundaries, never mid-generation.
    - **Decentralized RL (prime-rl, INTELLECT-2) exploits loose coupling.** Only prompts, rollouts, and infrequent weight broadcasts cross the WAN — none latency-critical — so stateless generation can run on permissionless, globally-distributed GPUs while the stateful trainer stays centralized.
    - **Untrusted workers create two new attacks.** *Reward forgery* is killed by recomputing rewards on a trusted **verifier**; *inference forgery* (wrong model/precision/fabricated tokens) is caught by **TOPLOC**, a locality-sensitive commitment to top-$k$ activations that re-verifies via a cheap batched prefill — robust to benign GPU numerics, sensitive to real model changes.
    - **Monitor staleness, `approx_kl` on fresh data, `clipfrac`, and TOPLOC reject rate.** A drifting `approx_kl` on the first epoch over fresh rollouts is the signature of engine mismatch — the most common silent async bug. Bring the run up synchronously, then dial $s_{\max}$ up.

!!! sota "State of the Art & Resources (2026)"
    Async, off-policy RL is now the default paradigm for large-scale reasoning-model training. The open frontier has shifted from *whether* to decouple generation from training to *how much staleness is tolerable*, how to correct for it rigorously, and how to push generation onto permissionless, globally-distributed hardware — the domain pioneered by Prime Intellect's INTELLECT-2 and prime-rl.

    **Foundational work**

    - [Espeholt et al., *IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures* (2018)](https://arxiv.org/abs/1802.01561) — introduced the decoupled actor-learner architecture and V-trace off-policy correction that is the direct conceptual ancestor of async LLM RL.
    - [DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* (2025)](https://arxiv.org/abs/2501.12948) — the landmark reasoning-RL recipe whose scale and heavy-tailed response lengths made async infrastructure a practical necessity.

    **Recent advances (2023–2026)**

    - [Fu et al., *AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning* (2025)](https://arxiv.org/abs/2505.24298) — full decoupling of generation and training with staleness-aware PPO; reports up to 2.77× speedup over synchronous baselines (NeurIPS 2025).
    - [Prime Intellect Team, *INTELLECT-2: A Reasoning Model Trained Through Globally Decentralized Reinforcement Learning* (2025)](https://arxiv.org/abs/2505.07291) — first 32B model trained via permissionless, globally-distributed async RL using prime-rl, TOPLOC, and SHARDCAST.
    - [Ong et al., *TOPLOC: A Locality Sensitive Hashing Scheme for Trustless Verifiable Inference* (2025)](https://arxiv.org/abs/2501.16007) — compact top-k activation commitments enable cheap, hardware-robust verification that an untrusted GPU ran the claimed model.
    - [Yu et al., *DAPO: An Open-Source LLM Reinforcement Learning System at Scale* (2025)](https://arxiv.org/abs/2503.14476) — introduces clip-higher asymmetric PPO, token-level loss aggregation, and dynamic sampling; key stabilization techniques for async off-policy runs.
    - [Sheng et al., *HybridFlow: A Flexible and Efficient RLHF Framework* (2024)](https://arxiv.org/abs/2409.19256) — the single-controller architecture underlying veRL; covers truncated importance sampling to handle inference-vs-training engine log-prob mismatch.

    **Open-source & tools**

    - [PrimeIntellect-ai/prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) — async RL training framework used for INTELLECT-2; supports FSDP2, vLLM, TOPLOC verification, and multi-node deployment at 1000+ GPUs.
    - [inclusionAI/AReaL](https://github.com/inclusionAI/AReaL) — production async RL system from Ant Group / Tsinghua IIIS; flexible, sample-level streaming with staleness control.
    - [verl-project/verl](https://github.com/verl-project/verl) — widely-used HybridFlow-based RL post-training library integrating FSDP, Megatron, vLLM, and SGLang; 21k+ GitHub stars.

    **Go deeper**

    - [Prime Intellect, *INTELLECT-2: The First Globally Distributed RL Training of a 32B Model* (blog, 2025)](https://www.primeintellect.ai/blog/intellect-2) — engineering walkthrough of the full decentralized async pipeline, TOPLOC integration, and SHARDCAST weight broadcast.

## Further reading

- DeepSeek-AI, **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning** (2025) — the reasoning-RL recipe whose scale motivates async infrastructure.
- Mei, Fu, Zhang, et al. (Ant Research / IIIS Tsinghua), **AReaL: A Fully Asynchronous Reinforcement Learning System for Language Reasoning** (2025) — staleness control and async pipelining in practice.
- Yu, et al. (Qwen / ByteDance Seed), **DAPO: An Open-Source LLM Reinforcement Learning System at Scale** (2025) — clip-higher, token-level loss, dynamic sampling.
- Sheng, Zhang, et al., **HybridFlow (veRL): A Flexible and Efficient RLHF Framework** (2024) — the single-controller architecture and truncated-IS for engine mismatch; see [veRL: HybridFlow & The Single-Controller Architecture](../06-rl-infra/04-verl.html).
- Prime Intellect, **INTELLECT-1: Launching the First Decentralized Training of a 10B Parameter Model** and **OpenDiLoCo** (2024) — globally-distributed low-communication pretraining.
- Prime Intellect, **INTELLECT-2** and the **prime-rl** framework (2025) — globally-distributed asynchronous RL for reasoning models.
- Douillard, Feng, Rusu, et al. (DeepMind), **DiLoCo: Distributed Low-Communication Training of Language Models** (2023) — many local steps between rare global syncs.
- Ong, et al. (Prime Intellect), **TOPLOC: A Locality-Sensitive Hashing Scheme for Trustless Verifiable Inference** (2024) — verifying that an untrusted GPU ran the claimed model.
- Schulman, Wolski, Dhariwal, et al., **Proximal Policy Optimization Algorithms** (2017) — the clipped surrogate that doubles as the staleness corrector.
- Espeholt, et al. (DeepMind), **IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures** (2018) — the classic decoupled actor-learner with V-trace off-policy correction, the conceptual ancestor of async LLM RL.
