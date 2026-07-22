# 6.11 Scaling RL: Throughput, Load Balancing & The Latest Tricks

By now you have seen the *anatomy* of an RL-for-LLM system ([The Anatomy of an RL-for-LLM System](../06-rl-infra/01-anatomy-rl-system.html)), the generation–training loop ([The Generation–Training Loop & Rollout Engines](../06-rl-infra/02-generation-training-loop.html)), and the controllers that orchestrate it ([veRL: HybridFlow & The Single-Controller Architecture](../06-rl-infra/04-verl.html)). This chapter is about a brutal, unglamorous fact: **most of the wall-clock time in an RL run is spent waiting**, and most of that waiting is *self-inflicted*. A GRPO step that should take 90 seconds takes 240 because seven of your eight rollout workers finished early and sat idle while one chewed through a 16,000-token chain-of-thought. Your H100s, which cost real money, idle at 30% MFU not because the math is hard but because the *schedule* is bad.

Scaling RL is therefore overwhelmingly a problem of **filling bubbles**: keeping the generation engines saturated, keeping the gradient device busy, and not throwing away samples that cost you a forward pass to produce. On top of that scheduling substrate sits a second layer — a fast-moving body of *algorithmic* tricks (DAPO's dynamic sampling and clip-higher, Dr. GRPO's bias fixes, VAPO's value-model stabilization, length penalties and over-long filtering) that change *which* samples you keep and *how much* each token contributes to the loss. These two layers interact: dynamic sampling, for instance, is simultaneously a statistics fix and a throughput disaster if you implement it naively.

We will treat them together because in practice you tune them together. We assume you know GRPO and the RLVR recipe ([GRPO, RLOO & Critic-Free RL](../05-posttraining-alignment/08-grpo-rloo.html); [RL with Verifiable Rewards (RLVR) & The Reasoning Recipe](../05-posttraining-alignment/09-rlvr-reasoning.html)) and continuous batching from the serving side ([Continuous Batching & Request Scheduling](../07-inference-serving/02-continuous-batching.html)).

## Where the time actually goes: a throughput model

Before optimizing anything, write down where the seconds go. A synchronous (on-policy) RL step has four phases that run *serially* in the simplest design:

{{fig:scalingrl-sync-step-phases}}

For reasoning workloads the generation phase dominates — frequently 70–85% of step time — because you generate long sequences autoregressively (memory-bandwidth bound, one token per forward pass) but train on them in a few large, compute-bound matmuls. So the single highest-leverage question in scaling RL is: **what fraction of generation wall-clock is useful work?**

Define, for a single generation phase processing a batch of $N$ prompts each sampled $G$ times (so $B = NG$ sequences):

$$
T_{\text{gen}} \;=\; \underbrace{\max_{i} \, t_i}_{\text{slowest sequence}} \;+\; \underbrace{T_{\text{sched}}}_{\text{scheduling / KV churn}}
$$

The $\max_i t_i$ term is the killer. If sequence lengths $L_i$ are heavy-tailed (and reasoning-trace lengths absolutely are), the longest sequence can be $5$–$10\times$ the median. Every other worker that drains its queue before the tail finishes contributes a *bubble*. Define generation efficiency as useful token-time over total token-time:

$$
\eta_{\text{gen}} \;=\; \frac{\sum_{i=1}^{B} L_i}{B \cdot \max_i L_i}
$$

This is just $\bar{L}/L_{\max}$ — mean over max. If your mean trace is 2,000 tokens and one outlier hits 16,000, then $\eta_{\text{gen}} \approx 0.125$ for a batch that runs to completion as one unit on one engine. *That is an 8× slowdown from one sequence.* Everything in the load-balancing section below is an attack on this single ratio.

!!! note "Aside: why generation, not training, dominates"
    A 7B model decoding one token does roughly $2 \cdot 7\times10^9 \approx 1.4\times10^{10}$ FLOPs but must read all 7B weights from HBM — it is bandwidth-bound and runs far below peak. Training on the same tokens packs them into dense matmuls that hit tensor cores near peak. So a token costs *far more wall-clock to generate than to train on*, even though the FLOP counts are comparable. This asymmetry is the entire reason RL-for-LLM infra looks the way it does. See [The Anatomy of LLM Inference: Prefill, Decode & The KV Cache](../07-inference-serving/01-anatomy-inference.html).

## Generation load imbalance & continuous batching in RL

### The problem: static batching wastes the tail

The naive rollout loop hands each engine a fixed shard of prompts and waits for *all* of them. Inside a single engine, if it uses **static batching** — pad every sequence to the longest in the batch, run them lock-step — you pay $B \cdot L_{\max}$ token-forwards even though only $\sum_i L_i$ are real. With reasoning traces this is catastrophic.

**Continuous batching** (a.k.a. in-flight / iteration-level batching), the standard in vLLM and SGLang, fixes the *intra-engine* part: as soon as one sequence emits its EOS, its slot is freed and the next queued prompt is admitted *mid-step*. No padding to a global max; the engine processes a rolling population of active sequences sized to fit the KV-cache budget. Within one engine, $\eta$ recovers to near 1 as long as there is always more work queued.

```python
# Sketch of why continuous batching helps. Two regimes, same 8 prompts.
import numpy as np

lengths = np.array([200, 220, 240, 260, 300, 1800, 320, 280])  # one outlier

# Static batching: every step processes ALL live slots padded to max length.
static_token_forwards = len(lengths) * lengths.max()           # 8 * 1800

# Continuous batching: a slot is freed at its own EOS; total real work only.
# (Assume the engine always has the 8 in flight; freed slots stay idle here
#  because there is nothing else queued -> tail still hurts, but no padding.)
continuous_token_forwards = lengths.sum()

print(static_token_forwards)      # 14400
print(continuous_token_forwards)  # 3620  -> ~4x less compute on the same batch
```

But note the comment: continuous batching removes *padding* waste, not *tail* waste. If those 8 prompts are the entire batch and nothing else is queued, the engine still runs until the 1,800-token outlier finishes, with 7 idle slots. To kill the tail you must either (a) feed *more prompts than slots* so freed slots get refilled, or (b) stop the tail early, or (c) overlap the tail with training. We will do all three.

### Lever 1: oversubscribe the engines

The cleanest fix: make the generation batch *larger than the number of concurrent slots* and let continuous batching pull from the queue. If an engine holds 64 concurrent sequences but you give it 512 prompts to produce, the 7 freed slots from short sequences immediately admit new prompts, and the engine stays full until the queue drains. Now the only bubble is the *final* tail — the last few long sequences with no replacements. With a deep queue, $\eta_{\text{gen}}$ climbs from $\bar L/L_{\max}$ toward $\approx 1 - (\text{slots}\cdot L_{\max})/\sum_i L_i$.

In veRL/OpenRLHF this is just: don't shard prompts 1:1 to slots; pour the whole rollout batch into the inference pool and let the scheduler balance. The practical constraint is the **KV-cache budget** ([PagedAttention & KV-Cache Memory Management](../04-kernels-efficiency/06-paged-attention-kv.html)): more concurrent long sequences need more cache pages.

{{fig:scalingrl-gen-bubble-oversubscribe}}

### Lever 2: length-aware dispatch across engines

When you have $K$ inference engines (data-parallel replicas of the policy), how you *partition* prompts across them matters. If you round-robin, one engine may by bad luck receive several outliers and become the straggler that gates the whole step. You cannot know lengths in advance, but you can balance *adaptively*: pull from a shared queue rather than pre-partitioning, so a fast engine that finishes its short sequences immediately grabs the next prompt. A shared global queue is strictly better than static per-engine assignment for heavy-tailed work — it is the classic "single queue, many servers" result from queueing theory.

```python
# Length-balanced greedy dispatch (LPT, "longest processing time first") when
# you DO have a length estimate (e.g., from a cheap length-predictor or from a
# previous epoch's trace for the same prompt). Minimizes makespan = max load.
import heapq

def lpt_dispatch(prompt_len_estimates, num_engines):
    """Assign prompts to engines to minimize the max per-engine total length.
    prompt_len_estimates: list of (prompt_id, estimated_gen_len)."""
    # Sort longest-first so big jobs are placed before small ones can imbalance.
    jobs = sorted(prompt_len_estimates, key=lambda x: -x[1])
    # Min-heap of (current_load, engine_id); always feed the least-loaded engine.
    heap = [(0, e) for e in range(num_engines)]
    heapq.heapify(heap)
    assignment = {e: [] for e in range(num_engines)}
    for pid, est in jobs:
        load, eng = heapq.heappop(heap)
        assignment[eng].append(pid)
        heapq.heappush(heap, (load + est, eng))
    return assignment

# LPT is a 4/3-approximation to optimal makespan -- in practice near-perfect
# balance when you have many small jobs and a few large ones.
```

In on-policy RL you usually do *not* have length estimates, so the shared-queue (pull) model is the workhorse and LPT shows up only when you cache per-prompt length statistics across epochs.

### Lever 3: overlap generation with training

Even a perfectly balanced generation phase leaves the *training* GPUs idle while it runs, and vice versa. The fix is to pipeline: while the trainer consumes batch $t$, the rollout engines already generate batch $t+1$. This is **one-step-off-policy** (sometimes "staleness-1") RL — the data the trainer sees was produced by weights one update old. Empirically a staleness of 1–2 is harmless for GRPO-style objectives if you apply the importance-sampling correction (PPO's ratio handles exactly this; see [Policy Gradients & PPO for Language Models](../05-posttraining-alignment/06-ppo-for-llms.html)). This is the gateway to fully async RL.

{{fig:scalingrl-overlap-pipeline}}

Fully async / disaggregated designs push this further — separate the inference and training fleets entirely and let them run continuously, reconciling via periodic weight broadcast. That is the subject of [Prime-RL, Async RL & Decentralized Training](../06-rl-infra/06-prime-rl-async.html) and [Colocated vs Disaggregated RL & Weight Synchronization](../06-rl-infra/07-colocated-vs-disaggregated.html); here we just note that overlap is the *third and largest* lever against the bubble, and it is the one that turns a 50%-idle system into a ~90%-utilized one.

## Partial rollout: don't pay for the tail twice

Overlap helps the *fleet* but a single 16k-token sequence still takes a long time to produce, and in a synchronous step it gates everything. **Partial rollout** (popularized by the Kimi k1.5 report and adopted widely) addresses the tail directly: cap how long any generation phase runs, *truncate the unfinished sequences mid-generation, save their KV state and partial text*, train on what completed, and **resume the truncated sequences in the next rollout phase** from where they left off.

The trick is that a partially generated sequence is not wasted — it is a *checkpoint*. You amortize one very long trajectory across multiple RL steps instead of letting it stall a single step.

{{fig:scalingrl-partial-rollout}}

The subtlety is **off-policyness**: a trajectory resumed at step $k+3$ has a prefix sampled by the weights at step $k$. The continuation is on newer weights. This is exactly the partial-trajectory staleness async RL must handle, and you correct for it with the per-token importance ratio just as PPO does — each token is reweighted by $\pi_{\text{new}}(a_t)/\pi_{\text{behavior}}(a_t)$. In practice you store, per resumed token, the log-prob under the *behavior* policy that produced it, and use that as the denominator. Partial rollout without that correction silently biases the gradient.

```python
@dataclass
class PartialTrajectory:
    prompt_ids: list[int]
    generated_ids: list[int]          # tokens emitted so far across phases
    behavior_logprobs: list[float]    # per-token logprob under the policy that
                                      # actually sampled each token (may differ
                                      # across phases -> needed for IS correction)
    kv_handle: object | None          # opaque engine-side KV cache handle, or
                                      # None if KV was dropped and must be re-prefilled
    done: bool

def run_phase(engine, trajs, token_budget):
    """Generate up to `token_budget` new tokens per trajectory, then truncate."""
    for tr in trajs:
        if tr.done:
            continue
        # Resume from saved KV if the engine kept it; else re-prefill the prefix.
        out = engine.generate(
            prefix_ids=tr.prompt_ids + tr.generated_ids,
            kv_handle=tr.kv_handle,
            max_new_tokens=token_budget,
            return_logprobs=True,
        )
        tr.generated_ids.extend(out.token_ids)
        tr.behavior_logprobs.extend(out.logprobs)  # tag with THIS phase's policy
        tr.kv_handle = out.kv_handle
        tr.done = out.hit_eos or len(tr.generated_ids) >= GLOBAL_MAX
    return trajs
```

**KV trade-off.** You can either keep the truncated sequence's KV cache resident on the inference engine between phases (fast resume, but the KV pages sit pinned and reduce the slots available for fresh prompts) or drop it and re-prefill the prefix on resume (frees memory, costs a prefill). Prefix caching ([Prefix Caching & KV-Cache Reuse](../07-inference-serving/07-prefix-caching.html)) softens the re-prefill cost. The right choice depends on your KV pressure; under heavy oversubscription, dropping + re-prefill often wins because slots are the scarce resource.

## Dynamic sampling & filtering: DAPO and the zero-gradient problem

Now we shift from *infrastructure* throughput to *statistical* throughput — making each produced sample actually contribute signal. The flagship example is **DAPO** (Decoupled clip and Dynamic sAmpling Policy Optimization), an open recipe that stacks four ideas on GRPO. Two of them — dynamic sampling and over-long filtering — are squarely about the rollout/throughput layer, so we treat DAPO here.

### Why GRPO wastes whole prompts

Recall GRPO computes a group-relative advantage by standardizing rewards *within* the $G$ samples for one prompt:

$$
\hat A_{i} \;=\; \frac{r_i - \operatorname{mean}(\{r_j\}_{j=1}^{G})}{\operatorname{std}(\{r_j\}_{j=1}^{G})}
$$

Consider a verifiable-reward task where $r \in \{0, 1\}$. If a prompt is *too easy*, all $G$ samples are correct: $r = (1,1,\dots,1)$, so $\operatorname{mean}=1$, every $\hat A_i = 0$. If *too hard*, all are wrong: again every $\hat A_i = 0$. **A prompt with all-same rewards contributes exactly zero gradient.** You paid $G$ full generations for it and got nothing. As training proceeds and the model masters the easy prompts, the *fraction of all-correct groups grows* — so the fraction of your rollout budget that is pure waste *increases over training*. This is the zero-gradient (or "vanishing-advantage") problem, and it is a throughput problem dressed as a statistics problem.

### DAPO's dynamic sampling: oversample, then filter to a full effective batch

DAPO's fix: **keep sampling new prompts and discarding all-same-reward groups until you have a full batch of groups that actually have non-zero advantage variance.** Concretely, set a target of $N$ "useful" prompt-groups per step; generate groups, drop any with $\operatorname{std}(r)=0$ (all correct or all wrong), and keep going until $N$ survive.

```python
def dynamic_sampling_step(engine, reward_fn, prompts_iter,
                          target_groups, G, max_oversample_factor=4):
    """Return `target_groups` prompt-groups, each with non-degenerate rewards.
    Oversamples to compensate for filtered-out all-same groups (DAPO-style)."""
    kept = []
    generated_groups = 0
    while len(kept) < target_groups:
        if generated_groups >= max_oversample_factor * target_groups:
            # Safety valve: if too many groups are degenerate, the curriculum
            # is mismatched (model too strong/weak). Stop to avoid runaway cost.
            break
        prompt = next(prompts_iter)
        completions = engine.generate(prompt, n=G)            # G samples
        rewards = [reward_fn(prompt, c) for c in completions]
        generated_groups += 1
        # Degenerate group => every group-relative advantage is 0 => no gradient.
        if min(rewards) == max(rewards):
            continue                                          # discard, don't train
        kept.append((prompt, completions, rewards))
    return kept
```

The throughput consequence is double-edged. *Statistically* it is a huge win: every group in the batch now moves the policy. *Mechanically* it is dangerous — the oversampling factor is **data-dependent and grows over training**, so a naive synchronous implementation generates a variable, ballooning number of sequences per step and reintroduces tail bubbles plus unpredictable step times. The production fix is to run dynamic sampling *on top of* a continuous-batching, oversubscribed, async generation pool: keep a deep queue of prompts, generate continuously, filter on the fly, and snapshot a training batch the moment $N$ useful groups have accumulated — never blocking the generators. Filtering and throughput are thus coupled: dynamic sampling is only cheap if your generation layer is already async and oversubscribed.

!!! warning "Common pitfall: dynamic sampling on a synchronous engine"
    If you bolt DAPO's dynamic sampling onto a synchronous "generate-batch-then-train" loop, your step time becomes a random variable that *increases* as the model improves (more groups get filtered, so you oversample more). You will see step times creep up and GPU utilization sag mid-run and blame the optimizer. The fix is architectural: dynamic sampling *requires* a decoupled, continuously-fed generation pool so the extra sampling overlaps training instead of serializing with it.

{{fig:scalingrl-dynamic-sampling-zero-gradient}}

### Over-long filtering & length control

The second DAPO throughput idea concerns **truncated** generations. If you cap generation at, say, 8k tokens and a sequence hits the cap without emitting EOS, what reward does it get? If you assign it the *task reward* (often 0 for an unfinished answer), you punish the model for a length-budget artifact and inject noise — the trajectory might have been on track to a correct answer. DAPO's **over-long filtering** simply *masks the loss* on sequences that were truncated by the length cap, so they neither reward nor penalize. A gentler variant is an **over-long soft punishment**: a small, length-graded penalty that ramps up as the sequence approaches the cap, nudging the model toward concision without the cliff of a hard zero.

```python
def length_shaped_reward(task_reward, gen_len, soft_start, hard_cap, lam=1.0):
    """DAPO-style soft overlong penalty.
    No penalty below soft_start; linearly ramps to -lam at hard_cap.
    Truncated (>= hard_cap, no EOS) sequences are handled by masking elsewhere."""
    if gen_len <= soft_start:
        return task_reward
    if gen_len >= hard_cap:
        # over-long: mask this trajectory's loss instead of training on noise
        return None   # sentinel -> caller drops/masks it
    frac = (gen_len - soft_start) / (hard_cap - soft_start)
    return task_reward - lam * frac
```

Length control matters beyond throughput: unfiltered RLVR tends to make traces *longer and longer* (a known reward-coupled length bias), inflating your generation cost every step in a vicious loop — longer traces, slower rollouts, more tail, lower MFU. Explicit length shaping is partly a *cost-control* mechanism. We return to the length-bias *cause* in the next section, because Dr. GRPO argues part of it is an artifact of the loss normalization itself.

## The latest algorithmic tricks: Dr. GRPO, VAPO, and the clip family

### DAPO's other two: clip-higher and token-level loss

The two DAPO ingredients we deferred are pure objective changes:

- **Clip-Higher (decoupled clipping).** PPO/GRPO clip the importance ratio symmetrically to $[1-\varepsilon, 1+\varepsilon]$. DAPO observes this *suppresses exploration*: for a low-probability but promising token, the upside is capped at $1+\varepsilon$, so the model can never aggressively up-weight a rare good token, and entropy collapses. DAPO **decouples** the bounds into $\varepsilon_{\text{low}}$ and $\varepsilon_{\text{high}}$ with $\varepsilon_{\text{high}} > \varepsilon_{\text{low}}$ (e.g. raise the upper clip), giving rare-but-good tokens more room to grow while keeping the downside tight. The clipped surrogate becomes

$$
\mathcal{L} = -\,\mathbb{E}\Big[\min\big(\rho_t \hat A_t,\; \operatorname{clip}(\rho_t,\, 1-\varepsilon_{\text{low}},\, 1+\varepsilon_{\text{high}})\,\hat A_t\big)\Big], \quad \rho_t = \frac{\pi_\theta(a_t)}{\pi_{\text{old}}(a_t)}
$$

- **Token-level policy-gradient loss.** GRPO as originally written averages the loss *per sequence first, then over sequences* — so every sequence gets equal weight regardless of length, which means tokens in long sequences are *down-weighted*. DAPO computes the loss as a flat **mean over all tokens in the batch**, giving each token equal weight. This couples directly to the length-normalization bug that Dr. GRPO formalizes next.

### Dr. GRPO: removing GRPO's two normalization biases

**Dr. GRPO** ("GRPO Done Right") makes a sharp, surgical claim: vanilla GRPO contains two normalization terms that introduce *optimization bias*, and one of them directly causes the runaway-length pathology.

1. **Length normalization in the loss.** GRPO divides each sequence's summed token loss by its length $|o_i|$. Combined with the way advantages are shared across tokens, this makes the *per-token* gradient magnitude depend on sequence length: for a *negative*-advantage (wrong) trajectory, dividing by a larger length *shrinks the penalty per token* — so the model learns that making wrong answers *longer* reduces the per-token penalty. That is a direct gradient incentive to ramble when wrong. Dr. GRPO **removes the length division**, using a constant normalizer (or token-level mean) so per-token gradients are length-independent.

2. **Standard-deviation normalization in the advantage.** Dividing the group-relative advantage by $\operatorname{std}(\{r_j\})$ over-weights prompts that happen to have *low* reward variance (a small denominator inflates the advantage). This is a question-level difficulty bias. Dr. GRPO **drops the std division**, keeping only the mean-subtraction baseline:

$$
\hat A_i^{\text{Dr.GRPO}} = r_i - \operatorname{mean}(\{r_j\}_{j=1}^{G})
\qquad\text{(no } /\operatorname{std}\text{, no } /|o_i|\text{)}
$$

```python
import torch

def grpo_advantage(rewards):                       # rewards: (G,)
    mean = rewards.mean()
    std = rewards.std() + 1e-6
    return (rewards - mean) / std                  # vanilla GRPO

def dr_grpo_advantage(rewards):                    # Dr. GRPO: mean-only baseline
    return rewards - rewards.mean()

def sequence_loss_vanilla(token_logp_ratio, adv, mask):
    # per-sequence mean over tokens (divides by length) -> length bias
    per_tok = -(token_logp_ratio * adv.unsqueeze(-1)) * mask
    return (per_tok.sum(-1) / mask.sum(-1).clamp(min=1)).mean()

def token_loss_drgrpo(token_logp_ratio, adv, mask):
    # flat mean over ALL valid tokens (no per-length division) -> length-neutral
    per_tok = -(token_logp_ratio * adv.unsqueeze(-1)) * mask
    return per_tok.sum() / mask.sum().clamp(min=1)
```

The two fixes interact with DAPO's token-level loss: both camps converge on "don't normalize per-sequence length; treat tokens uniformly." If you take one practical lesson from this section, it is **be deliberate about your loss normalizer** — it is not a cosmetic constant, it encodes a length policy and a difficulty policy, and getting it wrong is a common silent cause of length blow-up and instability ([Reward Hacking, Over-Optimization & Alignment Failures](../05-posttraining-alignment/13-reward-hacking-failures.html)).

{{fig:scalingrl-length-norm-bias}}

### VAPO: making the value model work for long CoT

GRPO/DAPO/Dr. GRPO are **critic-free** — they replace the learned value function with a group baseline ([Advantage Estimation, KL Control & Stability Tricks](../06-rl-infra/09-advantage-kl-tricks.html)). **VAPO** (Value-model-based Augmented PPO) argues the opposite: a *well-tuned value model* gives lower-variance, per-token advantages that beat group baselines on long-chain reasoning — *if* you fix the things that usually make value models fail there. Its main ingredients are engineering hardening for the critic: a **length-adaptive GAE** $\lambda$ (the bias–variance knob of generalized advantage estimation must scale with sequence length, because a fixed $\lambda$ that works at 500 tokens over-smooths at 8,000), value-pretraining/warmup so the critic is not garbage at step 0 when it would otherwise inject huge advantage noise, and DAPO-style clip-higher and token-level losses layered on top. VAPO is the strongest evidence that critic-free is a *convenience*, not a law: with enough care the value model wins, at the cost of training and hosting a second network (more memory, another forward/backward) — a classic throughput-vs-sample-efficiency trade.

The practical takeaway for *scaling*: critic-free (GRPO family) is cheaper per step and far simpler to host (no second model on the training GPUs), which is why it dominates open infra; value-based (VAPO/PPO) can be more *sample*-efficient but costs you memory and a second model to keep in sync. Pick based on whether your bottleneck is GPUs-per-step or samples-to-converge.

## Replay, curriculum & sample reuse

Two more levers raise *sample* efficiency — getting more learning per generated token, which is the ultimate throughput because generation is the expensive part.

**Replay / sample reuse.** On-policy purists generate fresh samples every step and throw them away. But each sample cost a generation; reusing it for $k$ minibatch passes (PPO epochs) amortizes that cost. The catch is staleness: after one gradient step the data is off-policy, so you *must* use the clipped importance ratio (which is exactly what PPO/GRPO already provide). A modest replay — 1–4 inner epochs over each rollout batch — is nearly free sample-efficiency. Go too far and the ratios drift outside the clip range, gradients get clipped to zero, and you waste compute on dead samples. A **prioritized** replay can preferentially revisit high-advantage-magnitude (most informative) trajectories.

**Curriculum.** Dynamic sampling already implements an *implicit* curriculum: it discards prompts that are all-correct (mastered) or all-wrong (hopeless), so the model trains on prompts in its **zone of proximal development** — those it gets right sometimes. You can make this explicit: bin prompts by historical pass-rate, schedule from easy to hard, or up-sample prompts near the 50% success boundary where group-relative advantage variance (hence gradient signal) is maximized. For a Bernoulli group with success probability $p$, the reward variance is $p(1-p)$, maximized at $p=0.5$ — so a prompt the model solves half the time gives the strongest learning signal, and one it never or always solves gives none. Curriculum is just steering the rollout budget toward $p \approx 0.5$.

```python
def informativeness(pass_rate):
    """Expected per-group reward variance for Bernoulli reward; gradient signal
    peaks at pass_rate = 0.5 and vanishes at 0 or 1 (the DAPO-filtered cases)."""
    return pass_rate * (1.0 - pass_rate)

# Schedule rollout budget proportional to informativeness (a soft curriculum).
def allocate_budget(prompt_pass_rates, total_budget):
    w = {pid: informativeness(pr) + 1e-3 for pid, pr in prompt_pass_rates.items()}
    z = sum(w.values())
    return {pid: int(total_budget * wi / z) for pid, wi in w.items()}
```

This unifies the chapter's two halves: the *infra* trick (dynamic sampling) and the *learning* trick (curriculum) are the same idea — **spend generation on prompts that move the policy** — viewed from throughput and from statistics respectively.

## A worked throughput example

!!! example "Worked example: filling the bubble on an 8×H100 GRPO step"
    Setup: GRPO on a 7B reasoning model. Per step we want $N = 256$ useful prompt-groups, $G = 8$ samples each, so the target is $256 \times 8 = 2{,}048$ trained sequences. Inference fleet: 4 H100s running vLLM (the other 4 train, colocated time-sliced). Trace lengths are heavy-tailed: median 1,500 tokens, mean 2,200, max outliers ~14,000. Assume an effective decode rate of ~2,500 tokens/s per H100 at this batch concurrency.

    **Baseline — static-ish synchronous, no oversubscription, no filtering.**
    Generation runs until the longest of 2,048 sequences finishes. With per-engine concurrency = 512 and 4 engines, the engines drain to the tail. Effective generation efficiency $\eta_{\text{gen}} = \bar L / L_{\max} = 2200 / 14000 \approx 0.157$. Real token-work $= 2048 \times 2200 = 4.5\text{M}$ tokens, but the phase runs as if $2048 \times 14000 = 28.7\text{M}$ token-times across the fleet. At $4 \times 2500 = 10{,}000$ tok/s fleet decode, the *wall-clock* is gated by the tail engine: roughly $14000 / 2500 \approx 5.6$ s just to drain the outlier, plus the bulk $\approx 4.5\text{M}/10{,}000 = 450$ s for real work — but because the tail engine cannot parallelize one sequence, total $\approx 450 / \eta_{\text{effective}}$. Take a measured stand-in: **generation ≈ 200 s**, training ≈ 60 s, **step ≈ 260 s**, train GPUs idle during all 200 s of generation.

    **+ Continuous batching & oversubscription (Lever 1).** Deep queue keeps all engines full until the final tail. Bubble shrinks to the last handful of long sequences: generation ≈ **120 s** (≈1.7× faster).

    **+ Overlap generation with training (Lever 3, 1-step off-policy).** Training's 60 s now hides *inside* the next generation phase. Effective step ≈ $\max(120, 60) =$ **120 s**, and train GPUs go from 23% to ~100% busy.

    **+ Partial rollout, 4k budget/phase (tail control).** The 14k outliers no longer gate a step; they amortize across ~4 phases. Per-phase generation drops to ≈ **80 s**. Step ≈ 80 s.

    **+ Dynamic sampling (DAPO).** Statistically, suppose 35% of groups are all-correct/all-wrong and contribute zero gradient. Without filtering you *trained* on 2,048 sequences but only $0.65 \times 2048 \approx 1{,}331$ carried signal — your *effective* useful throughput was 65% of nominal. With dynamic sampling you oversample ~1.54× to refill the batch with useful groups; because generation is already async+oversubscribed, the extra sampling overlaps and costs little wall-clock, so you now deliver **2,048 fully-useful groups** in ≈ **95 s**.

    Net: **260 s → ~95 s per step (≈2.7× wall-clock)**, *and* every trained sample now moves the policy — the *learning-per-step* improved on top of the speedup. The single biggest win was overlap (Lever 3); the single biggest *quality* win was dynamic sampling.

## Putting it together: a state-of-the-art recipe

The 2024–2026 open-source consensus stack for scaling RL on reasoning models looks roughly like this — note how throughput tricks and algorithmic tricks interleave:

{{fig:scalingrl-sota-recipe-stack}}

The meta-point: **throughput in RL is won by overlap and by not wasting samples, not by faster kernels.** Kernel-level speedups (FlashAttention, FP8, CUDA graphs — Part IV) matter, but they shave the *useful* work; the order-of-magnitude wins come from eliminating bubbles (overlap, oversubscription, partial rollout) and from eliminating *useless* work (dynamic sampling, over-long filtering, curriculum). Get the scheduling right first.

!!! tip "Practitioner tip: instrument the bubble before you optimize"
    Log, per step: total generated tokens, $\bar L$, $L_{\max}$, $\eta_{\text{gen}} = \bar L/L_{\max}$, the fraction of degenerate (all-same-reward) groups, train-GPU idle %, and infer-GPU idle %. These six numbers tell you *which* lever to pull. High $L_{\max}/\bar L$ → partial rollout. High train-GPU idle → overlap. High degenerate-group fraction → dynamic sampling. Most teams optimize blind; a one-screen dashboard of these is worth a week of guesswork.

!!! interview "Interview Corner"
    **Q:** Your GRPO run on reasoning data gets only ~30% GPU utilization and step time keeps creeping up as training proceeds. Walk me through diagnosing and fixing it.

    **A:** Two distinct symptoms. (1) *Low utilization* is almost always generation-side bubbles. In a synchronous loop the training GPUs idle during generation and vice versa, and within generation a heavy-tailed length distribution means most workers finish early and idle waiting on the longest trace. I'd measure $\eta_{\text{gen}} = \bar L / L_{\max}$ and the train/infer idle percentages. Fixes in order of impact: overlap generation with training (1-step off-policy pipelining) so neither fleet idles; oversubscribe the inference engines with a deep prompt queue feeding continuous batching so freed slots refill; and partial rollout to cap how long any single trace gates a step. (2) *Step time creeping up over training* is the signature of dynamic sampling on a synchronous engine: as the model improves, more prompt-groups become all-correct (zero advantage variance), so DAPO-style filtering discards more and the oversampling factor grows — each step generates more sequences. The fix isn't to drop dynamic sampling (you need it, or you train on zero-gradient batches) but to make generation async and oversubscribed so the extra sampling overlaps training instead of serializing. I'd also check loss normalization — per-sequence length normalization (vanilla GRPO) incentivizes longer wrong answers, inflating $L_{\max}$ over time and feeding the bubble; switching to a token-level loss (Dr. GRPO / DAPO) removes that length-blowup driver.

!!! key "Key Takeaways"
    - **The dominant cost in RL-for-LLM is generation, and the dominant waste is the bubble** — idle GPUs from heavy-tailed sequence lengths and from serializing generation against training. Optimize the *schedule* before the kernels.
    - **$\eta_{\text{gen}} = \bar L / L_{\max}$** is the one number to watch; a single 8× outlier trace can cut a run-to-completion batch's efficiency to ~12%.
    - **Three levers fill the bubble:** oversubscribe engines + continuous batching (refill freed slots), length-aware/shared-queue dispatch (no straggler engine), and overlap generation with training via 1-step off-policy pipelining (the biggest single win).
    - **Partial rollout** caps the tail by truncating long traces, stashing their KV/text, and resuming next phase — but you must importance-correct for the per-token behavior policy.
    - **DAPO's dynamic sampling** discards all-same-reward groups (zero gradient) and oversamples to a full *useful* batch — a statistics fix that is only cheap on an async, oversubscribed generation layer.
    - **Dr. GRPO** removes two silent biases: per-sequence length normalization (which incentivizes longer wrong answers) and std-normalization of the advantage (which over-weights low-variance prompts). Be deliberate about your loss normalizer.
    - **DAPO's clip-higher** decouples the PPO clip bounds to preserve exploration; **VAPO** shows a hardened value model (length-adaptive GAE, warmup) can beat critic-free RL at the cost of a second network.
    - **Curriculum and dynamic sampling are the same idea** — spend the rollout budget on prompts near pass-rate 0.5, where reward variance $p(1-p)$ and thus gradient signal is maximal.

!!! sota "State of the Art & Resources (2026)"
    Scaling RL for LLMs is a fast-moving engineering discipline where the biggest wins come from eliminating generation bubbles (overlap, oversubscription, partial rollout) and from eliminating useless samples (dynamic sampling, curriculum). The 2025 open-source ecosystem has largely converged on DAPO-style dynamic sampling and token-level loss normalization as standard practice, while value-based methods (VAPO) and fully-async disaggregated fleets represent the frontier.

    **Foundational work**

    - [DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* (2025)](https://arxiv.org/abs/2501.12948) — introduced GRPO and large-scale RLVR, the baseline that all tricks in this chapter build upon.
    - [Sheng et al., *HybridFlow: A Flexible and Efficient RLHF Framework* (2024)](https://arxiv.org/abs/2409.19256) — the veRL paper; describes 3D-HybridEngine, hybrid single/multi-controller design, and 1.5–20× throughput gains over prior systems.

    **Recent advances (2023–2026)**

    - [Yu et al., *DAPO: An Open-Source LLM Reinforcement Learning System at Scale* (2025)](https://arxiv.org/abs/2503.14476) — dynamic sampling, clip-higher, token-level loss, and over-long filtering; 50 pts on AIME 2024 with Qwen2.5-32B.
    - [Liu et al., *Understanding R1-Zero-Like Training: A Critical Perspective* / Dr. GRPO (2025)](https://arxiv.org/abs/2503.20783) — identifies length-normalization and std-normalization biases in GRPO; simple fix prevents runaway trace length.
    - [Yue et al., *VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks* (2025)](https://arxiv.org/abs/2504.05118) — first value-model-based RL framework to outperform critic-free methods on long-CoT; length-adaptive GAE and critic warm-up are key.
    - [Kimi Team, *Kimi k1.5: Scaling Reinforcement Learning with LLMs* (2025)](https://arxiv.org/abs/2501.12599) — introduces partial rollout (truncate-and-resume) and long-context RL scaling to 128k tokens.
    - [Noukhovitch et al., *Asynchronous RLHF: Faster and More Efficient Off-Policy RL for Language Models* (ICLR 2025)](https://arxiv.org/abs/2410.18252) — formal treatment of overlapping generation and training; ~40–70% wall-clock speedup with provably small staleness cost.

    **Open-source & tools**

    - [verl-project/verl](https://github.com/verl-project/verl) — production-grade flexible RL post-training framework; supports GRPO, PPO, FSDP/Megatron-LM, vLLM/SGLang, scales to 671B parameters.
    - [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — Ray + vLLM distributed RLHF framework; supports PPO, GRPO, RLOO, REINFORCE++ with multi-turn and VLM training.
    - [sgl-project/sglang](https://github.com/sgl-project/sglang) — high-throughput LLM serving engine widely used as the rollout backend in RL stacks; RadixAttention, continuous batching, CUDA graphs.
    - [PrimeIntellect-ai/prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) — fully async RL framework for agentic training; scales to 1000+ GPUs with FSDP2 + vLLM and FP8 inference.

## Further reading

- Yu et al., *DAPO: An Open-Source LLM Reinforcement Learning System at Scale* — dynamic sampling, clip-higher, token-level loss, over-long filtering.
- Liu et al., *Understanding R1-Zero-Like Training: A Critical Perspective* (Dr. GRPO) — the length- and std-normalization bias analysis and fixes.
- Yue et al., *VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks* — value-model-based long-CoT RL with length-adaptive GAE.
- Kimi Team, *Kimi k1.5: Scaling Reinforcement Learning with LLMs* — partial rollout and long-context RL infrastructure.
- DeepSeek-AI, *DeepSeekMath* (GRPO) and *DeepSeek-R1* — the group-relative baseline and the RLVR reasoning recipe these tricks build on.
- Schulman et al., *Proximal Policy Optimization Algorithms* and *High-Dimensional Continuous Control Using Generalized Advantage Estimation (GAE)* — the clipping and advantage machinery underneath.
- The **veRL (HybridFlow)** and **OpenRLHF** repositories — production implementations of oversubscription, continuous batching, and async overlap for RL.

## Exercises

**1.** (Conceptual) In GRPO with a verifiable binary reward $r \in \{0,1\}$, explain why a prompt whose $G$ samples all receive the *same* reward (all correct, or all wrong) contributes *exactly zero* gradient. Then explain why the fraction of your rollout budget wasted this way tends to *increase* as training proceeds, and name the DAPO mechanism that fixes it.

??? note "Solution"
    GRPO's advantage standardizes rewards *within* the group:
    $$
    \hat A_i = \frac{r_i - \operatorname{mean}(\{r_j\}_{j=1}^{G})}{\operatorname{std}(\{r_j\}_{j=1}^{G})}.
    $$
    If every $r_j$ is identical (all $1$ or all $0$), then $r_i = \operatorname{mean}(\{r_j\})$ for every $i$, so the numerator $r_i - \operatorname{mean} = 0$ for all samples. Every $\hat A_i = 0$, and since the policy-gradient loss is a sum of terms each proportional to $\hat A_i$, the whole group's contribution to the gradient is zero. You paid for $G$ full generations and got no learning signal. (The $\operatorname{std}=0$ denominator is degenerate too, but the numerator already kills it regardless of how you regularize the denominator.)

    Why the waste *grows*: early in training the model gets many prompts wrong-but-sometimes-right (pass-rate near the middle), so groups have reward variance and non-zero advantage. As training proceeds the model *masters* the easy prompts, pushing their pass-rate to $\approx 1$ — those groups become all-correct and therefore degenerate. So the fraction of all-same-reward groups climbs over the run, and an ever-larger share of the generation budget produces zero gradient. It is a throughput problem wearing a statistics costume.

    The fix is **DAPO's dynamic sampling**: keep sampling fresh prompt-groups and discard any with $\operatorname{std}(r)=0$, continuing until you have accumulated a full batch of $N$ groups that all have non-zero advantage variance. Every trained group then moves the policy.

**2.** (Quantitative) Take the chapter's 8-sequence batch with generated lengths
$$
L = [200,\ 220,\ 240,\ 260,\ 300,\ 1800,\ 320,\ 280].
$$
(a) Compute the generation efficiency $\eta_{\text{gen}} = \bar L / L_{\max}$ for running this batch to completion as one unit. (b) Compute the static-batching token-forwards $B\cdot L_{\max}$ and the continuous-batching token-forwards $\sum_i L_i$, and the ratio between them. (c) In one sentence, say which kind of waste (padding vs tail) the ratio in (b) measures, and which it does *not*.

??? note "Solution"
    (a) Sum: $200+220+240+260+300+1800+320+280 = 3620$. Mean: $\bar L = 3620/8 = 452.5$. Max: $L_{\max}=1800$.
    $$
    \eta_{\text{gen}} = \frac{452.5}{1800} \approx 0.251.
    $$
    So run-to-completion, only about 25% of the fleet's token-time on this batch is useful work; the single 1800-token outlier drags the rest down.

    (b) Static batching pads all $B=8$ live slots to $L_{\max}=1800$ every step:
    $$
    B\cdot L_{\max} = 8 \times 1800 = 14400 \ \text{token-forwards}.
    $$
    Continuous batching pays only for real tokens:
    $$
    \sum_i L_i = 3620 \ \text{token-forwards}.
    $$
    Ratio $= 14400 / 3620 \approx 3.98$, i.e. roughly $4\times$ less compute on the same batch (matching the chapter's sketch).

    (c) That $\approx 4\times$ ratio measures the elimination of **padding** waste (no more padding every sequence up to the global max). It does *not* capture **tail** waste: if these 8 are the whole batch and nothing else is queued, continuous batching still runs until the 1800-token sequence finishes with 7 slots idle. Killing the tail needs oversubscription, partial rollout, or overlap — not just continuous batching.

**3.** (Quantitative) Rewards are Bernoulli: for a prompt the model solves with probability $p$, each of the $G$ independent samples is correct w.p. $p$. (a) Using the chapter's `informativeness(p) = p(1-p)`, compute the per-group reward variance at $p = 0.1,\ 0.5,\ 0.9$ and state where signal is maximal. (b) A group is *degenerate* (filtered by dynamic sampling) exactly when all $G$ samples are correct or all wrong. For $G=8$, compute the degenerate probability at $p=0.5$ and at $p=0.9$, and interpret. (c) If a fraction $f=0.35$ of generated groups turn out degenerate, what oversampling factor do you need to still deliver $N$ useful groups?

??? note "Solution"
    (a) $p(1-p)$:
    - $p=0.1$: $0.1 \times 0.9 = 0.09$.
    - $p=0.5$: $0.5 \times 0.5 = 0.25$.
    - $p=0.9$: $0.9 \times 0.1 = 0.09$.

    Signal (reward variance, hence group-relative advantage magnitude) is maximal at $p=0.5$ — the prompt the model solves half the time — and symmetric, vanishing toward $p=0$ or $p=1$. This is why curriculum aims the rollout budget at $p \approx 0.5$.

    (b) Degenerate probability $= p^G + (1-p)^G$.
    - $p=0.5,\ G=8$: $0.5^8 + 0.5^8 = 2 \times (1/256) = 2/256 = 1/128 \approx 0.0078$.
    - $p=0.9,\ G=8$: $0.9^8 + 0.1^8 \approx 0.4305 + 10^{-8} \approx 0.430$.

    Interpretation: a mid-difficulty prompt ($p=0.5$) is almost never filtered (< 1% degenerate), so it reliably yields signal. An easy prompt ($p=0.9$) is degenerate ~43% of the time — nearly half its groups are all-correct and wasted. As the model masters prompts (pushing $p$ toward 1), the degenerate rate climbs, which is precisely the growing-waste effect from Exercise 1 and the reason dynamic sampling's oversampling factor grows over training.

    (c) You keep a fraction $(1-f)$ of generated groups, so to net $N$ useful ones you must generate
    $$
    \frac{N}{1-f} = \frac{N}{0.65} \approx 1.54\,N,
    $$
    an oversampling factor of about $1.54\times$ (matching the worked example). Because the generation layer is async and oversubscribed, this extra sampling overlaps training rather than serializing with it.

**4.** (Implementation) The chapter's `run_phase` stores, per resumed token, the log-prob under the *behavior* policy in `behavior_logprobs`. Implement a function `is_corrected_surrogate(new_logprobs, behavior_logprobs, advantages, eps_low, eps_high)` that computes DAPO's decoupled clip-higher surrogate *loss* (a scalar to minimize) for one partial trajectory, using the per-token importance ratio $\rho_t = \pi_\theta(a_t)/\pi_{\text{behavior}}(a_t)$. Use the chapter's asymmetric clip bounds $[1-\varepsilon_{\text{low}},\ 1+\varepsilon_{\text{high}}]$ and a flat mean over tokens (token-level loss). Then briefly state why the *behavior* log-prob — not the current-policy log-prob — must go in the denominator.

??? note "Solution"
    The ratio uses log-probs, so $\rho_t = \exp(\log\pi_\theta(a_t) - \log\pi_{\text{behavior}}(a_t))$. The clip-higher surrogate takes the pessimistic $\min$ of the unclipped and clipped terms, then we negate and take a token-level mean (equal weight per token, per DAPO / Dr. GRPO):

    ```python
    import torch

    def is_corrected_surrogate(new_logprobs, behavior_logprobs, advantages,
                               eps_low=0.2, eps_high=0.28):
        """Clip-higher, token-level surrogate LOSS for one partial trajectory.
        new_logprobs, behavior_logprobs, advantages: 1-D tensors, per token.
        Returns a scalar to MINIMIZE."""
        # rho_t = pi_theta(a_t) / pi_behavior(a_t), computed in log space.
        ratio = torch.exp(new_logprobs - behavior_logprobs)
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high) * advantages
        # min() = PPO's pessimistic bound; asymmetric bounds = DAPO clip-higher.
        surrogate = torch.minimum(unclipped, clipped)
        # Flat mean over all tokens (token-level loss, no per-length division).
        return -surrogate.mean()
    ```

    Why the behavior policy is the denominator: a partial trajectory resumed several steps later has a prefix that was *sampled* by older weights, and importance sampling corrects the mismatch between the distribution that *produced* the tokens and the distribution you are *optimizing*. The correct ratio is $\pi_{\text{new}}/\pi_{\text{behavior}}$, where $\pi_{\text{behavior}}$ is the policy that actually drew each token (stored at sampling time in `behavior_logprobs`). Putting the current policy in the denominator would give $\rho_t \equiv 1$, discarding the correction entirely and silently biasing the gradient — exactly the failure the chapter warns about for partial rollout without IS correction.

**5.** (Conceptual + short calculation) Dr. GRPO claims vanilla GRPO's per-sequence *length normalization* gives a direct gradient incentive to make *wrong* answers longer. Make this concrete: two wrong trajectories share the same negative advantage $\hat A = -1$, but have lengths $|o_1| = 100$ and $|o_2| = 1000$ tokens. Assuming the summed per-token loss magnitude for a trajectory is proportional to $|\hat A| \cdot |o_i|$ before normalization, compute the *per-token* penalty each trajectory receives (a) under vanilla GRPO's divide-by-length normalizer and (b) under Dr. GRPO's token-level (flat) mean. Explain what the model learns from (a).

??? note "Solution"
    Model the pre-normalization summed loss magnitude of a trajectory as $|\hat A|\cdot|o_i|$ (each of its $|o_i|$ tokens carries a $|\hat A|$-sized push). The *per-token* penalty is what drives each token's gradient.

    (a) **Vanilla GRPO — divide by sequence length $|o_i|$.** GRPO's per-sequence mean multiplies every token's loss by the normalizer $1/|o_i|$. The summed magnitude $|\hat A|\cdot|o_i|$ therefore collapses to a per-sequence loss of $|\hat A|\cdot|o_i|/|o_i| = |\hat A|$, which spread across the sequence's $|o_i|$ tokens gives a per-token penalty of
    $$
    \frac{|\hat A|}{|o_i|}.
    $$
    Numerically, with $|\hat A|=1$:
    - $|o_1|=100$: per-token penalty $= 1/100 = 0.010$.
    - $|o_2|=1000$: per-token penalty $= 1/1000 = 0.001$.

    The long wrong trajectory gets a $10\times$ *smaller* penalty per token. So gradient descent finds a cheap way to reduce the per-token loss on wrong answers: *make them longer*. That is a direct incentive to ramble when wrong, and it compounds — longer wrong traces inflate $L_{\max}$, worsen the tail bubble, and lower MFU over the run.

    (b) **Dr. GRPO / DAPO — flat token-level mean, no per-length division.** Every valid token in the batch gets equal weight, so the per-token penalty is the same constant for both trajectories:
    - $|o_1|=100$: per-token penalty $= |\hat A| = 1$ (up to the shared global $1/(\text{total tokens})$ factor).
    - $|o_2|=1000$: per-token penalty $= |\hat A| = 1$ (same).

    Now lengthening a wrong answer gives *no* per-token relief — each extra wrong token adds its own full penalty rather than diluting the existing one. Removing the length division makes per-token gradients length-independent and eliminates the runaway-length incentive, which is Dr. GRPO's central fix (paired with dropping the std-normalization difficulty bias).
