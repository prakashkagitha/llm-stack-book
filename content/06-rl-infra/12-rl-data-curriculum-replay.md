# 6.12 RL Data, Curriculum & Replay Management

Every chapter in this part so far has treated the *prompts* as a given — a static `.jsonl` of math problems or coding tasks that flows into the rollout engine. That is a convenient fiction, and it is the single most expensive fiction in RL-for-LLMs. The dirty secret of reinforcement learning with verifiable rewards (RLVR) is that **most of your rollout budget is wasted on prompts that teach the policy nothing.** A prompt the model already solves 8/8 times produces a group with zero reward variance and therefore zero gradient under GRPO. A prompt the model solves 0/8 times produces the same zero-variance, zero-gradient group from the other end. You paid for $G$ full autoregressive rollouts — the most expensive operation in the entire stack ([The Anatomy of an RL-for-LLM System](../06-rl-infra/01-anatomy-rl-system.html)) — and got an all-zeros advantage vector for your trouble.

This chapter is about the **data side** of RL: the levers that determine *which* prompts the policy sees, *when* it sees them, and *whether a rollout you already paid for gets reused.* These levers — dataset construction and quality control, difficulty estimation, difficulty-targeted online selection, prompt curriculum, dynamic sampling, and rollout replay buffers — are collectively the largest free lunch in RL sample efficiency. They are also where theory (the variance of the policy-gradient estimator) and systems (keeping a heavy-tailed generation fleet saturated) collide most violently.

We assume you know GRPO and the RLVR recipe ([GRPO, RLOO & Critic-Free RL](../05-posttraining-alignment/08-grpo-rloo.html); [RL with Verifiable Rewards (RLVR) & The Reasoning Recipe](../05-posttraining-alignment/09-rlvr-reasoning.html)), the throughput model from the previous chapter ([Scaling RL: Throughput, Load Balancing & The Latest Tricks](../06-rl-infra/11-scaling-rl-tricks.html)), and the reward/verifier machinery that turns a completion into a scalar ([Reward Engineering, Verifiers & Sandboxes](../06-rl-infra/08-reward-verifiers-sandboxes.html)). This is the curriculum-and-data complement to those chapters.

## Why difficulty is the master variable

Start from the gradient. For a prompt $x$ with a binary verifiable reward $r\in\{0,1\}$, sampled $G$ times under the current policy, the empirical pass rate is $\hat p = \frac{1}{G}\sum_{i} r_i$. GRPO centers each completion's reward by the group mean and (in vanilla form) divides by the group standard deviation:

$$
A_i \;=\; \frac{r_i - \bar r}{\operatorname{std}(r) + \varepsilon}, \qquad \bar r = \hat p, \qquad \operatorname{std}(r) = \sqrt{\hat p (1-\hat p)}.
$$

Two facts fall out immediately. First, if every $r_i$ is identical (all $0$ or all $1$), then $\bar r = r_i$ for all $i$, every $A_i = 0$, and the prompt contributes **exactly zero** to the policy gradient. The group is dead. Second, the amount of *usable signal* in a group is governed by the variance of the reward, which for a Bernoulli is

$$
\operatorname{Var}(r) \;=\; p(1-p).
$$

This is a downward parabola, maximized at $p = 0.5$ where $\operatorname{Var}(r) = 0.25$, and falling to zero at both ends. The expected number of *informative pairs* — completions with different rewards inside a group, which is what GRPO's centering actually exploits — is proportional to $p(1-p)$ as well. So the gradient signal-to-noise of a prompt is a single-peaked function of one scalar: **its pass rate under the current policy.**

{{fig:rldata-signal-parabola}}

The practical consequence: **you want to spend rollout compute on prompts whose current pass rate is near 0.5.** A prompt at $p=0.9$ gives variance $0.09$; a prompt at $p=0.5$ gives $0.25$ — roughly $2.8\times$ the signal per rollout. A prompt at $p=0.99$ gives $0.0099$, essentially nothing. This is the quantitative heart of curriculum learning in RL, and it is why "difficulty" is not a soft pedagogical nicety but the master variable controlling your estimator's efficiency.

There is a subtlety the variance argument hides. Pass rate is **non-stationary**: it is a property of the *prompt and the current policy together*, and the policy moves every step. A prompt that sits at $p=0.5$ at step 0 may be at $p=0.95$ by step 200 because the model learned it. A static difficulty label computed once, offline, decays. The whole architecture of online difficulty-targeted selection exists to track this moving target.

!!! note "Aside: why not just always train at p=0.5?"
    Because a policy that only ever sees 50%-pass prompts never gets pulled into new regimes. The frontier of "things I can do half the time" advances only if some signal leaks from harder prompts (occasional lucky successes that the gradient amplifies) and is consolidated on easier ones. In practice you target a *band* — say $p\in[0.2, 0.8]$ — not a razor at $0.5$, and you let the band's contents drift as the policy improves. The band is the curriculum; the policy walks up it.

## Building the dataset: construction and quality control

Before any online selection can help, you need a corpus of prompts that is (a) *verifiable*, (b) *de-duplicated against your eval sets*, and (c) *spread across a usable difficulty range*. Garbage in the prompt pool is more corrosive in RL than in SFT, because RL will ruthlessly exploit any defect in the reward signal ([Reward Hacking, Over-Optimization & Alignment Failures](../05-posttraining-alignment/13-reward-hacking-failures.html)).

### The non-negotiables of an RLVR prompt set

For each task you need a **prompt** and a **checker**. The checker is the verifier — a deterministic (or near-deterministic) function that maps a completion to a reward. For math it is symbolic/numeric answer-equivalence; for code it is a unit-test harness in a sandbox; for agentic tasks it is an environment that returns a terminal success bit ([Reward Engineering, Verifiers & Sandboxes](../06-rl-infra/08-reward-verifiers-sandboxes.html)). The data record therefore carries more than text:

```python
from dataclasses import dataclass, field
from typing import Callable, Optional
import hashlib

@dataclass
class RLTask:
    task_id: str                      # stable hash, used for dedup + replay keys
    prompt: str                       # the rendered chat prompt fed to the policy
    answer: Optional[str] = None      # ground-truth answer for math-style checking
    tests: Optional[list] = None      # unit tests for code tasks
    domain: str = "unknown"           # math / code / logic / agentic / ...
    # difficulty state, maintained ONLINE (see later sections):
    pass_rate_ema: float = 0.5        # exponential moving avg of empirical pass rate
    n_attempts: int = 0               # how many groups we've spent on this task
    n_solved_groups: int = 0          # groups where at least one sample passed
    static_difficulty: Optional[float] = None  # offline estimate, optional prior

    def fingerprint(self) -> str:
        # Normalize before hashing so trivial whitespace/casing diffs collapse.
        norm = " ".join(self.prompt.lower().split())
        return hashlib.sha256(norm.encode()).hexdigest()[:16]


def make_checker(task: RLTask) -> Callable[[str], float]:
    """Return a verifier closure. In production the code path runs in a sandbox;
    here we sketch the math path. The checker MUST be robust to extraction noise."""
    if task.domain == "math":
        gold = normalize_math(task.answer)
        def check(completion: str) -> float:
            pred = extract_boxed_answer(completion)   # parse \boxed{...}
            return 1.0 if pred is not None and math_equal(pred, gold) else 0.0
        return check
    raise NotImplementedError(f"no checker for domain {task.domain}")
```

A few rules that separate a usable RL corpus from a frustrating one:

1. **Verifiability over volume.** Ten thousand prompts with airtight checkers beat a million with flaky ones. A checker that returns a false positive 2% of the time teaches the policy to *find* that 2% — RL is an adversary against your verifier. Audit checkers on known-good and known-bad completions before trusting them.

2. **Answer extraction is part of the reward.** If your checker only accepts `\boxed{}` and the model writes "The answer is 42," you will mislabel correct completions as failures, depressing pass rates and corrupting your difficulty estimates. Make extraction permissive *for grading correctness* but be careful it cannot be gamed.

3. **Decontaminate against evals — twice.** Run n-gram and embedding-level overlap checks between your prompt pool and every benchmark you will report on. Then do it again after any synthetic augmentation. Contamination inflates eval and is the most common cause of "great training curve, flat real-world gains." This mirrors pretraining decontamination ([Data Cleaning, Deduplication & Quality Filtering](../03-pretraining/02-data-cleaning-dedup.html)).

4. **Dedup near-duplicates inside the pool.** Two prompts that differ only in variable names are one prompt's worth of signal but two prompts' worth of compute. MinHash/LSH or embedding clustering collapses them.

5. **Track provenance.** Keep `domain`, source, and a difficulty prior per task. You will want to re-weight domains later, exactly as in pretraining data mixing ([Data Mixing, Domain Weighting & Curriculum](../03-pretraining/14-data-mixing-curriculum.html)).

### Where the prompts come from

Most strong RLVR runs blend several sources: curated competition/benchmark-style problems with known answers; **synthetically generated** variants (a stronger model or a generator produces new problems plus checkable answers, then a solver filters for solvability — see [Synthetic Data for Pre- and Post-Training](../03-pretraining/15-synthetic-data.html)); and **mined** failures from a previous policy (prompts the last checkpoint got wrong become the next run's curriculum, a data flywheel — [Data Flywheels & Continuous Improvement](../12-production-mlops/05-data-flywheel.html)). For agentic tasks the "prompt" is a whole environment seed (a repo state, a browser task, a tool sandbox), and construction means building reproducible, resettable environments ([Agentic & Multi-Turn RL](../06-rl-infra/10-agentic-multiturn-rl.html)).

## Difficulty estimation: offline priors and online truth

You ultimately want, for every task, an estimate of its pass rate under the *current* policy. There are two regimes: an **offline prior** computed once before the run, and the **online estimate** maintained during it.

### Offline difficulty estimation

Before the run, sample the base policy (or a reference model) $k$ times on each prompt and record $\hat p_0 = (\text{successes})/k$. This is the maximum-likelihood estimate of the pass rate, with the obvious caveat that for small $k$ it is coarse — with $k=8$ you can only distinguish pass rates in steps of $1/8 = 0.125$, and the standard error is

$$
\operatorname{SE}(\hat p) = \sqrt{\frac{\hat p(1-\hat p)}{k}} \;\le\; \frac{1}{2\sqrt{k}}.
$$

For $k=8$ that worst-case SE is $\approx 0.18$ — so a prompt you measured at $\hat p_0 = 0.5$ might truly be anywhere in roughly $[0.32, 0.68]$. Offline difficulty is a *prior*, not a label. Its value is in **bucketing** and in **discarding the unusable tails** (prompts the base model solves $0/k$ — possibly impossible or mis-checkered — and $k/k$ — already mastered). A common offline pipeline:

```python
import numpy as np

def estimate_offline_difficulty(tasks, policy, checker_fn, k=8, batch_gen=None):
    """One pass over the pool: k rollouts each, record empirical pass rate.
    Returns buckets and prunes the dead tails. batch_gen() should call your
    rollout engine (vLLM/SGLang) — batch ALL prompts*k together for throughput."""
    for t in tasks:
        completions = batch_gen(t.prompt, n=k)            # k samples
        rewards = [checker_fn(t)(c) for c in completions]
        t.static_difficulty = float(np.mean(rewards))     # p_hat_0 in [0,1]
        t.pass_rate_ema = t.static_difficulty             # seed the online EMA

    keep, pruned = [], []
    for t in tasks:
        # Prune the dead tails: never-solved (maybe broken) and always-solved.
        if 0.0 < t.static_difficulty < 1.0:
            keep.append(t)
        else:
            pruned.append(t)
    return keep, pruned


def difficulty_bucket(p, edges=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)):
    """Map a pass rate to a coarse bucket index. Buckets, not raw p, because
    raw p from small k is too noisy to act on at fine granularity."""
    for b in range(len(edges) - 1):
        if edges[b] <= p < edges[b + 1]:
            return b
    return len(edges) - 2  # p == 1.0 edge case
```

The $0/k$-and-$k/k$ pruning typically removes 20–50% of a raw competition-math pool: a chunk is trivial for a strong base model, and another chunk is currently impossible (genuinely too hard, or the checker is wrong). Both contribute zero gradient and pure rollout waste, so removing them is a direct sample-efficiency win *before training begins*.

### Online difficulty: the EMA and Beta views

During training the truth is the online pass rate. Each time task $t$ is rolled out as a group of $G$, you observe $\hat p_t = (\text{successes})/G$ and update an exponential moving average:

$$
\bar p \;\leftarrow\; (1-\alpha)\,\bar p \;+\; \alpha\,\hat p_t,
$$

with $\alpha$ around $0.1$–$0.3$. The EMA tracks the moving policy: a prompt that was hard becomes easy as the policy learns, and the EMA follows. A cleaner Bayesian alternative is a **Beta–Bernoulli** posterior per task: maintain counts $(s, f)$ of successes and failures (optionally discounted over time so old observations decay), and the posterior pass rate is $\text{Beta}(s+1, f+1)$ with mean $\frac{s+1}{s+f+2}$. The Beta view gives you not just a point estimate but a *credible interval*, which is exactly what you need to decide whether a prompt is "near 0.5" with confidence or just under-sampled — and it plugs naturally into a Thompson-sampling selection policy (below).

!!! warning "Common pitfall: stale difficulty labels"
    The most common difficulty-curriculum bug is computing pass rates *once* and treating them as fixed. Within a few hundred steps a well-chosen 50%-band collapses toward 100% as the policy masters it, and if you keep sampling the same "medium" bucket you are now training on solved prompts — zero gradient, wasted compute, and a training curve that mysteriously plateaus. Difficulty MUST be re-estimated online (EMA or decayed Beta). The whole point is that difficulty is a function of the *current* policy, which is a moving target.

## Difficulty-targeted online selection: keeping prompts near p≈0.5

Now the central mechanism. Given a pool with online difficulty estimates, how do you choose the next batch of prompts to roll out so that the *realized* groups cluster near $p\approx0.5$ and almost none come back zero-variance?

### Selection as a bandit over difficulty

Frame it as a multi-armed bandit where each task (or each difficulty bucket) is an arm and the "reward" is the *informativeness* of the group it produces — for example $\hat p_t(1-\hat p_t)$, the realized Bernoulli variance, or simply $\mathbb{1}[\text{group had non-zero variance}]$. You want to pull arms that are currently informative while exploring arms whose difficulty you are unsure about (because the policy moved). Two clean policies:

- **Greedy-to-target:** score each task by closeness of its estimated pass rate to the target $p^\star$ (usually $0.5$), e.g. $\text{score}(t) = -\,|\bar p_t - p^\star|$, and sample the top-scoring tasks (with noise / temperature so you do not over-commit to a handful). Simple, effective, but blind to estimate uncertainty.
- **Thompson sampling on the Beta posterior:** draw $\tilde p_t \sim \text{Beta}(s_t+1, f_t+1)$ for each candidate and select tasks whose *sampled* pass rate is nearest the target. This automatically explores under-sampled tasks (wide posteriors get drawn far from their mean), so freshly-promoted-difficulty prompts get re-evaluated. This is the principled version and is what I reach for in practice.

A worked implementation appears in the code section. The key design choice is **the band, not the point**: select for $\bar p \in [p_{\text{lo}}, p_{\text{hi}}]$ (say $[0.2, 0.8]$) rather than exactly $0.5$, so the batch retains some easier prompts (stability, anti-forgetting) and some harder prompts (frontier signal, exploration). Within the band you can still weight toward the center.

### The relationship to dynamic sampling

Difficulty-targeted selection and **dynamic sampling** (DAPO-style) attack the same enemy — zero-variance groups — from opposite ends of the loop. Selection works *before* generation: pick prompts likely to land in-band. Dynamic sampling works *after* generation: discard the groups that came back all-same-reward anyway and oversample to refill the batch. You want both, because difficulty estimates are noisy (small-$k$ SE is large) and the policy moves between selection and rollout, so even a well-targeted batch will produce some dead groups. Selection reduces how many you have to throw away; dynamic sampling guarantees the batch you train on is fully informative.

## Dynamic sampling: never train on a zero-variance batch

Dynamic sampling, popularized by DAPO, is mechanically simple and statistically important. After generating $G$ completions per prompt and scoring them, **drop every prompt whose group has zero reward variance** (all correct or all wrong), then keep generating *more* prompts until you have collected a full target batch of $B_{\text{keep}}$ prompts that all have non-zero variance.

$$
\text{keep prompt } x \iff 0 < \hat p(x) < 1 \quad\Longleftrightarrow\quad \operatorname{std}\big(r(x)\big) > 0.
$$

The benefit is that **every** gradient step now operates on a batch where every prompt contributes signal — no dead weight diluting the update, no wasted optimizer step. The cost is *throughput*: you must oversample. If a fraction $\rho$ of generated groups survive the filter, you must generate $B_{\text{keep}}/\rho$ groups to fill the batch — and $\rho$ shrinks as the policy improves and more prompts saturate to $p=1$. This is the throughput-vs-statistics tension flagged in the scaling chapter ([Scaling RL: Throughput, Load Balancing & The Latest Tricks](../06-rl-infra/11-scaling-rl-tricks.html)): dynamic sampling is *cheap on an async, oversubscribed generation layer* and *brutal on a synchronous one*, where the extra rollouts serialize against training.

Difficulty-targeted selection is the throughput rescue for dynamic sampling: by feeding the generator prompts that are *already likely* to be in-band, you raise the survival fraction $\rho$, so you oversample less to fill the batch. The two are complementary — selection raises $\rho$, dynamic sampling guarantees correctness when $\rho<1$.

!!! example "Worked example: the oversampling tax, with and without targeting"
    Suppose you want a training batch of $B_{\text{keep}} = 256$ informative prompts, $G = 8$ samples each.

    **Naive uniform sampling (mid-training).** Your pool's pass-rate distribution, *under the current policy*, is roughly: 40% of prompts at $p\approx0.95$ (nearly mastered), 25% at $p\approx0.05$ (nearly impossible), 35% spread in $[0.2,0.8]$. What survives the zero-variance filter? A group at $p=0.95$ comes back all-correct with probability $0.95^8 \approx 0.66$, i.e. it is *dead* 66% of the time, surviving only 34%. A group at $p=0.05$ is dead $0.95^8\approx0.66$ of the time too (all-wrong), surviving 34%. The in-band prompts survive almost always ($1 - 0.8^8 - 0.2^8 \approx 0.83$ at $p=0.5$; higher near the band edges is similar). Survival fraction:

    $$
    \rho \approx 0.40(0.34) + 0.25(0.34) + 0.35(0.83) \approx 0.136 + 0.085 + 0.29 \approx 0.51.
    $$

    To fill 256 you must generate $256 / 0.51 \approx 502$ groups — roughly **2× the rollout compute** thrown away as zero-variance, $502 \times 8 \approx 4016$ completions for 2048 kept.

    **Difficulty-targeted selection.** Now you pre-select prompts whose online EMA sits in $[0.2,0.8]$. Even accounting for estimate noise and policy drift (so realized in-band fraction is, say, 80% rather than 100%), survival climbs to roughly

    $$
    \rho \approx 0.80(0.83) + 0.20(0.34) \approx 0.66 + 0.07 \approx 0.73.
    $$

    Now you generate $256/0.73 \approx 351$ groups — you have cut the oversampling tax from $\sim$2× down to $\sim$1.37×, saving about **30% of generation compute** for the same informative batch. On a run where generation is 75% of wall-clock, that is a $\sim$20% end-to-end speedup, for free, from data-side bookkeeping.

## Prompt curriculum: walking the policy up the difficulty ladder

Difficulty-targeted selection keeps you near $p=0.5$ *at each step*. A **curriculum** is the trajectory of that band over the whole run. Three flavors, increasingly automatic:

### Static / staged curriculum

Sort prompts into difficulty tiers offline and present them in stages: easy first, then medium, then hard, switching tiers on a schedule or when an aggregate pass-rate threshold is hit. This is the RL analogue of pretraining curriculum ([Data Mixing, Domain Weighting & Curriculum](../03-pretraining/14-data-mixing-curriculum.html)). It is simple and sometimes helps warm-start, but it is brittle: the offline tiers decay (the master-variable problem), and a hard stage switch can destabilize the policy.

### Automatic / online curriculum

Let the curriculum *emerge* from online difficulty targeting. Because difficulty is re-estimated every step and you always select for the band, the contents of the band naturally drift from easy to hard as the policy masters the easier material — the curriculum is an emergent property, not a hand-authored schedule. This is the modern default: you do not write a curriculum, you write a *selection rule* and the curriculum is its trajectory.

### Self-paced / regret-based curriculum

A more aggressive variant (rooted in the unsupervised-environment-design and automatic-curriculum-learning literature) selects prompts by **learning progress** or **regret**: prioritize tasks where the policy is improving fastest, or where there is the largest gap between achievable and current performance. Practically, score a task by the *recent change* in its pass-rate EMA (a prompt whose $\bar p$ is climbing is where learning is happening) and up-weight it. This pushes compute exactly to the frontier of competence. It is powerful but adds estimator complexity and can be unstable if the progress signal is noisy.

{{fig:rldata-curriculum-band-drift}}

The figure is the whole idea: each tier rises and saturates; the *band* (the prompts currently near $p^\star$) slides from easy through medium to hard. A good online curriculum keeps the training batch riding the diagonal where the tiers cross $p^\star$.

## Replay buffers: reusing rollouts you already paid for

Generation is the dominant cost ([Scaling RL: Throughput, Load Balancing & The Latest Tricks](../06-rl-infra/11-scaling-rl-tricks.html)). A rollout you used once and discarded is money burned. **Replay** reuses past rollouts — but RL-for-LLMs is *on-policy* by nature (the policy gradient is an expectation under the *current* policy), so naive replay is off-policy and biased. Replay in this setting comes in two distinct flavors that solve different problems.

### Prompt-level replay (revisiting tasks)

The cheap, safe kind: replay *which prompts to attempt*, not the old completions. A **prioritized prompt buffer** stores tasks with a priority equal to their current informativeness (e.g. $|\bar p_t - p^\star|^{-1}$, or recent learning progress) and re-samples high-priority prompts more often — this is exactly the prioritized-experience-replay idea (Schaul et al.) applied at the *task* level rather than the transition level. The completions are always freshly generated under the current policy, so it stays on-policy. This is just difficulty-targeted selection with persistence and is essentially free.

### Trajectory-level replay (reusing completions)

The expensive, dangerous kind: store the *actual completions* and their per-token log-probs $\pi_{\text{old}}(a_t\mid s_t)$, and reuse them for a few extra gradient steps. Because they were generated under an older policy, you must **importance-correct** with the PPO ratio and rely on clipping to bound the bias ([Policy Gradients & PPO for Language Models](../05-posttraining-alignment/06-ppo-for-llms.html); [Advantage Estimation, KL Control & Stability Tricks](../06-rl-infra/09-advantage-kl-tricks.html)):

$$
\rho_t = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\text{old}}(a_t\mid s_t)}, \qquad
L = \min\!\big(\rho_t A_t,\; \operatorname{clip}(\rho_t, 1-\epsilon, 1+\epsilon)A_t\big).
$$

This is the mechanism behind **asynchronous / off-policy RL** (the generator runs ahead of the trainer; rollouts are 1–4 steps stale by the time they are consumed — [Prime-RL, Async RL & Decentralized Training](../06-rl-infra/06-prime-rl-async.html)). The replay "buffer" here is shallow — a few steps of staleness, not a DQN-style million-transition reservoir — because the importance weights blow up and the clipped gradient goes to zero once $\pi_\theta$ has drifted too far from $\pi_{\text{old}}$. **Staleness is the half-life of a stored trajectory.** Beyond a few steps the IS weights are so far from 1 that clipping zeroes the contribution, so the trajectory is dead weight. A practical buffer evicts trajectories older than a staleness bound $\tau_{\max}$ (e.g. 2–4 policy versions) and tracks the *fraction of tokens being clipped* as a health metric — if most tokens are clipped, your buffer is too stale and you are training on noise.

A third, increasingly important pattern is the **experience / success buffer for agentic RL**: store *successful* trajectories (especially for hard, rarely-solved tasks) and replay them as on-policy-ish positive examples or as SFT-style anchors, to keep the policy from forgetting a hard-won capability. This blurs into expert iteration / rejection-sampling fine-tuning (the policy generates, you keep the winners, you train on them) and is a robust way to bank progress on sparse-reward agentic tasks ([Agentic & Multi-Turn RL](../06-rl-infra/10-agentic-multiturn-rl.html)).

!!! tip "Practitioner tip: keep three buffers, not one"
    In a mature RL stack you typically maintain (1) a **prioritized prompt buffer** (which tasks to attempt next — on-policy, free, the curriculum), (2) a **shallow staleness buffer** of recent completions with stored log-probs for async overlap (off-policy, importance-corrected, half-life of a few steps), and (3) a **success/hard-case buffer** of banked winning trajectories for anti-forgetting on sparse tasks. They are different objects with different correctness constraints — do not conflate them. The first is about *throughput and signal*, the second about *latency overlap*, the third about *retention*.

## Code: a dynamic-sampling + difficulty-bucketing rollout loop

Here is a self-contained, heavily-commented loop that ties the chapter together: difficulty-targeted selection via Thompson sampling, generation, zero-variance filtering with oversampling (dynamic sampling), online difficulty (Beta posterior) updates, and a prioritized prompt buffer. The rollout engine and trainer are stubbed so the *data logic* is in focus; in a real system `engine.generate` calls vLLM/SGLang and `trainer.step` calls your GRPO update.

```python
import numpy as np
from dataclasses import dataclass, field

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# Task state: a Beta(s+1, f+1) posterior over the *current-policy* pass rate.
# We decay old counts so the posterior tracks the moving policy (not lifetime).
# ---------------------------------------------------------------------------
@dataclass
class TaskState:
    task_id: int
    true_p: float                 # SIMULATION ONLY: the latent pass rate
    s: float = 1.0                # decayed success pseudo-count (+1 prior)
    f: float = 1.0                # decayed failure pseudo-count (+1 prior)
    n_groups: int = 0             # how many groups we've spent here (priority/age)

    def posterior_mean(self):
        return self.s / (self.s + self.f)

    def sample_p(self):
        # Thompson draw: sample a plausible pass rate from the posterior.
        return rng.beta(self.s, self.f)

    def update(self, successes, G, decay=0.9):
        # Decay then add this group's evidence. Decay makes the posterior
        # forget stale (old-policy) observations so it tracks current p.
        self.s = decay * self.s + successes
        self.f = decay * self.f + (G - successes)
        self.n_groups += 1


# ---------------------------------------------------------------------------
# Stubs standing in for the real rollout engine and trainer.
# ---------------------------------------------------------------------------
class FakeEngine:
    """Simulates generating G samples for a task; returns #successes ~ Binomial.
    A real engine returns text completions; the checker turns them into r in {0,1}.
    We also let true_p drift UP a touch each time a task is trained on, to mimic
    the policy mastering material (the non-stationarity the EMA/Beta must track)."""
    def rollout(self, task: TaskState, G: int) -> int:
        succ = int(rng.binomial(G, task.true_p))
        return succ

    def learn_drift(self, task: TaskState, lr=0.02):
        # Mastering nudges pass rate toward 1; harder material drifts slower.
        task.true_p = min(0.999, task.true_p + lr * (1.0 - task.true_p))


# ---------------------------------------------------------------------------
# Difficulty-targeted selection by Thompson sampling toward a target pass rate.
# ---------------------------------------------------------------------------
def select_candidates(tasks, n_select, target_p=0.5):
    scored = []
    for t in tasks:
        p_tilde = t.sample_p()              # explore via posterior uncertainty
        score = -abs(p_tilde - target_p)    # prefer tasks near the target band
        scored.append((score, t))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:n_select]]


# ---------------------------------------------------------------------------
# One RL step: select -> generate -> DYNAMIC SAMPLING filter -> update -> train.
# Dynamic sampling: keep only non-zero-variance groups; oversample to refill.
# ---------------------------------------------------------------------------
def rl_step(tasks, engine, B_keep=64, G=8, target_p=0.5,
            oversample_cap=6, buckets=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)):
    kept = []                               # (task, successes) that have signal
    generated_groups = 0
    rounds = 0
    while len(kept) < B_keep and rounds < oversample_cap:
        rounds += 1
        # Oversample a generous candidate set so we can refill after filtering.
        need = B_keep - len(kept)
        cands = select_candidates(tasks, n_select=2 * need, target_p=target_p)
        for t in cands:
            successes = engine.rollout(t, G)
            generated_groups += 1
            t.update(successes, G)          # online difficulty (Beta) update
            # DYNAMIC SAMPLING: drop zero-variance groups (all pass / all fail).
            if 0 < successes < G:
                kept.append((t, successes))
                if len(kept) >= B_keep:
                    break

    # --- "Train" on the kept (informative) groups: here we just record stats
    #     and apply the simulated learning drift so difficulty is non-stationary.
    bucket_counts = np.zeros(len(buckets) - 1, dtype=int)
    for t, successes in kept:
        engine.learn_drift(t)               # policy improves -> p drifts up
        p_hat = successes / G
        b = min(np.searchsorted(buckets, p_hat, side="right") - 1, len(buckets) - 2)
        bucket_counts[max(b, 0)] += 1

    survival = len(kept) / max(generated_groups, 1)
    return {
        "kept": len(kept),
        "generated_groups": generated_groups,
        "survival_rho": survival,           # what fraction survived the filter
        "oversample_factor": generated_groups / max(len(kept), 1),
        "bucket_counts": bucket_counts,     # distribution of kept difficulties
    }


# ---------------------------------------------------------------------------
# Run it. Watch survival rho recover toward 1 (selection feeds in-band prompts)
# and the kept-difficulty histogram concentrate near the target band.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # A pool spanning the full difficulty range, incl. the dead tails we must avoid.
    pool = [TaskState(i, true_p=rng.uniform(0.02, 0.98)) for i in range(4000)]
    for step in range(8):
        stats = rl_step(pool, FakeEngine(), B_keep=64, G=8, target_p=0.5)
        print(f"step {step:2d} | rho={stats['survival_rho']:.2f} "
              f"| oversample={stats['oversample_factor']:.2f}x "
              f"| buckets[0.0-1.0]={stats['bucket_counts']}")
```

Two behaviors to watch when you run this. First, the **survival fraction $\rho$** is well above the uniform-sampling baseline because Thompson selection feeds in-band prompts — that is the oversampling-tax saving from the worked example, made mechanical. Second, the **kept-difficulty histogram concentrates around the middle buckets** (near $p=0.5$): even though the pool spans $[0.02, 0.98]$, the batch you actually train on is the informative middle, by construction. As `learn_drift` pushes mastered tasks toward $p=1$, the Beta posteriors follow, those tasks stop being selected, and harder tasks rotate into the band — the emergent curriculum, in code.

```python
# Bolt-on: a prioritized PROMPT buffer (PER at the task level). Priority = how
# close a task is to the target band, with a small age bonus so we revisit
# under-sampled tasks. This is the on-policy, FREE kind of replay.
def prompt_priority(t: TaskState, target_p=0.5, age_w=0.05):
    closeness = 1.0 / (abs(t.posterior_mean() - target_p) + 0.05)  # near band -> high
    uncertainty = (t.s * t.f) / ((t.s + t.f) ** 2 * (t.s + t.f + 1))  # Beta variance
    return closeness + age_w * uncertainty  # exploit band + explore uncertain tasks

def sample_from_buffer(tasks, n, target_p=0.5, temperature=1.0):
    pr = np.array([prompt_priority(t, target_p) for t in tasks])
    probs = (pr ** (1.0 / temperature))
    probs /= probs.sum()
    idx = rng.choice(len(tasks), size=n, replace=False, p=probs)
    return [tasks[i] for i in idx]
```

The prioritized prompt buffer is the persistent, sampling-without-replacement-per-step version of the Thompson selector: priority rewards proximity to the band (exploit) plus posterior variance (explore under-sampled tasks). Note this buffer stores *tasks*, never old completions — it is strictly on-policy and therefore free of importance-weighting concerns, unlike the staleness buffer discussed earlier.

## Putting it together: the data-side knobs that move sample efficiency

Step back. The levers in this chapter form a pipeline, and each one removes a different category of wasted rollout:

{{fig:rldata-sample-efficiency-pipeline}}

Each numbered stage is a multiplier on sample efficiency, and they compound: decontamination protects the *validity* of everything downstream; tail-pruning removes the prompts that can *never* contribute; targeting raises survival $\rho$ so dynamic sampling oversamples less; dynamic sampling guarantees the gradient is never diluted by dead groups; and online difficulty closes the loop so the whole thing tracks the policy as it improves. A run with all six tuned can be *several times* more sample-efficient — in rollouts-per-unit-improvement — than the same algorithm reading a static shuffled `.jsonl`, with no change to the loss function at all.

!!! interview "Interview Corner"
    **Q:** You're running GRPO on a math corpus and notice your training reward climbs for 200 steps then plateaus, while GPU utilization stays high and the loss is non-zero but tiny. What's happening, and how do you diagnose and fix it on the *data* side — without touching the RL algorithm?

    **A:** This is the classic *difficulty-saturation* failure. Early on, many prompts sit near $p=0.5$ where reward variance $p(1-p)$ — and thus the policy-gradient signal — is maximal. As the policy improves, those prompts drift toward $p=1$; their groups increasingly come back all-correct, contribute zero advantage variance, and the *effective* batch (the prompts that actually carry gradient) shrinks even though every GPU is busy generating. Utilization stays high because you're still rolling out; the loss is tiny because most groups are near-zero-variance.

    To diagnose, I'd log the **per-group pass-rate histogram** and the **fraction of zero-variance groups per step**; the smoking gun is that fraction climbing over training. The fix is entirely data-side: (1) turn on **dynamic sampling** so you only train on $0<\hat p<1$ groups and oversample to refill — this immediately restores a fully-informative batch; (2) maintain an **online difficulty estimate** (EMA or decayed Beta posterior) and **target selection** toward the $p\approx0.5$ band, so you stop spending rollouts on mastered prompts and the *emergent curriculum* slides toward harder material; (3) if survival fraction $\rho$ is dropping, that confirms saturation and also tells you the oversampling tax you're paying. If the pool itself is exhausted of in-band prompts (everything is now easy), that's a *corpus* problem — mine the current policy's failures or add harder synthetic prompts to extend the curriculum. Crucially none of this changes the GRPO objective; it changes *which prompts the objective sees*.

## Key Takeaways

!!! key "Key Takeaways"
    - **Difficulty is the master variable.** For a binary verifiable reward, gradient signal per prompt scales as $p(1-p)$, peaking at pass rate $p=0.5$ and vanishing at both ends — so a prompt's *current* pass rate decides its worth.
    - **Pass rate is non-stationary.** It's a property of the prompt *and the current policy*; a static difficulty label decays as the policy learns. Estimate difficulty **online** (EMA or a decayed Beta–Bernoulli posterior), never once-and-forever.
    - **Construction and QC dominate validity.** Verifiable checkers, eval decontamination, and near-duplicate removal matter more in RL than SFT because RL adversarially exploits any reward defect; prune the $0/k$ and $k/k$ tails before training to delete prompts that can never contribute.
    - **Difficulty-targeted online selection** (greedy-to-target or Thompson sampling on the Beta posterior) keeps realized groups near a $p\approx0.5$ *band*, raising the dynamic-sampling survival fraction $\rho$ and cutting the oversampling tax — often a ~20–30% end-to-end win, free.
    - **Dynamic sampling** drops zero-variance groups and oversamples to refill, guaranteeing every gradient step trains on signal — cheap on an async/oversubscribed generator, brutal on a synchronous one.
    - **Curriculum is emergent, not authored:** with online targeting, the band's contents drift from easy to hard automatically as the policy masters material; regret/learning-progress weighting pushes compute to the competence frontier.
    - **Three buffers, three jobs:** a prioritized *prompt* buffer (on-policy, free, the curriculum); a *shallow staleness* buffer of completions with stored log-probs for async overlap (off-policy, importance-corrected, half-life of a few steps); and a *success/hard-case* buffer for anti-forgetting on sparse agentic tasks.
    - **Trajectory replay has a half-life:** stored completions go stale as the policy drifts; importance weights blow up and clipping zeroes their gradient, so evict beyond a staleness bound and monitor the clipped-token fraction.

!!! sota "State of the Art & Resources (2026)"
    Data-side RL has converged on a recognizable stack: verifiable corpora with aggressive decontamination, offline difficulty bucketing to prune dead tails, online difficulty tracking, difficulty-targeted selection toward a ~50%-pass band, and DAPO-style dynamic sampling as default practice. The frontier is automatic curriculum (learning-progress/regret weighting) and principled off-policy replay for fully-async, agentic RL.

    **Foundational work**

    - [Schaul et al., *Prioritized Experience Replay* (2016)](https://arxiv.org/abs/1511.05952) — the priority-and-importance-weight machinery that prompt/trajectory buffers adapt to the LLM setting.
    - [Graves et al., *Automated Curriculum Learning for Neural Networks* (2017)](https://arxiv.org/abs/1704.03003) — learning-progress signals for ordering tasks; the conceptual root of regret-based curricula.

    **Recent advances (2024–2026)**

    - [Yu et al., *DAPO: An Open-Source LLM Reinforcement Learning System at Scale* (2025)](https://arxiv.org/abs/2503.14476) — dynamic sampling (drop zero-variance groups, oversample to refill) as standard RLVR practice.
    - [DeepSeek-AI, *DeepSeek-R1* (2025)](https://arxiv.org/abs/2501.12948) and *DeepSeekMath* — the GRPO/RLVR baseline whose advantage structure makes pass rate the master variable.
    - [Kimi Team, *Kimi k1.5* (2025)](https://arxiv.org/abs/2501.12599) — curriculum and prioritized sampling at scale alongside partial-rollout infrastructure.

    **Open-source & tools**

    - [verl-project/verl](https://github.com/verl-project/verl) and [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — production RL frameworks with dynamic sampling, dataset/curriculum hooks, and async replay.
    - [PrimeIntellect-ai/prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) — fully-async agentic RL where shallow staleness replay and importance correction are first-class.

## Further reading

- Schaul, Quan, Antonoglou & Silver, *Prioritized Experience Replay* — priority and importance weighting; the buffer mechanics this chapter ports to prompts and trajectories.
- Graves et al., *Automated Curriculum Learning for Neural Networks* — learning-progress-driven task ordering; the basis for regret/progress curricula.
- Yu et al., *DAPO: An Open-Source LLM Reinforcement Learning System at Scale* — dynamic sampling and the zero-variance-group filter in an RLVR context.
- DeepSeek-AI, *DeepSeekMath* (GRPO) and *DeepSeek-R1* — the group-relative advantage that makes pass-rate variance $p(1-p)$ the signal-to-noise of a prompt.
- Kimi Team, *Kimi k1.5: Scaling Reinforcement Learning with LLMs* — curriculum, prioritized sampling, and long-context RL data infrastructure.
- Noukhovitch et al., *Asynchronous RLHF: Faster and More Efficient Off-Policy RL for Language Models* — the staleness/importance-weight regime that bounds trajectory-replay reuse.
- Schulman et al., *Proximal Policy Optimization Algorithms* — the clipped importance-sampling objective that makes shallow off-policy replay safe.
- Dennis et al., *Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design* (PAIRED) — regret-based automatic curriculum, the theoretical cousin of difficulty-targeted selection.
