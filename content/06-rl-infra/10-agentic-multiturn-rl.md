# 6.10 Agentic & Multi-Turn RL

Everything in Part VI up to this point has assumed a comfortably simple RL setting: a prompt goes in, the policy emits one completion, a reward function scores that completion, and you compute a policy gradient. That is **single-turn RL**. It is the setting of the DeepSeek-R1 math recipe, of [GRPO, RLOO & Critic-Free RL](../05-posttraining-alignment/08-grpo-rloo.html), and of most of what people mean when they say "RLVR." One prompt, one response, one scalar.

Agentic RL breaks that assumption in the most consequential way possible. Now the model does not emit a single answer — it emits an **action**, the action hits an **environment** (a Python interpreter, a search engine, a web browser, a shell, a game state), the environment emits an **observation**, and the model must condition on that observation to decide its next action. The episode is a *trajectory* of interleaved model tokens and environment tokens, possibly dozens of turns long, and the reward may not arrive until the very end. This is the regime that produces tool-using coding agents, deep-research agents, and computer-use agents — and it is where almost every hard problem in RL infrastructure shows up at once.

This chapter is about the mechanics of doing RL when the rollout is a *conversation with an environment* rather than a single generation. We will build the trajectory data structure from scratch, work out exactly which tokens get a loss and which do not (the single most common source of silent bugs in agentic RL), derive turn-level versus trajectory-level credit assignment, walk through an environment in the RAGEN/ReAct style, and confront the genuinely hard problems: long-horizon credit assignment, partial rollouts, and the brutal throughput tax of interleaving generation with environment execution. The downstream payoff — what these agents actually *do* — is the subject of [Part VIII](../08-agents-harness/02-agentic-loop.html); here we focus on how to *train* them.

## From Single-Turn to Multi-Turn: The Trajectory as the Unit

In single-turn RL the unit of data is a `(prompt, response, reward)` triple. The policy defines a distribution $\pi_\theta(y \mid x)$ over the response $y$, and the response tokens are exactly the tokens we differentiate.

In multi-turn agentic RL the unit is a **trajectory** $\tau$:

$$
\tau = (s_0, a_0, o_1, a_1, o_2, a_2, \dots, o_{T-1}, a_{T-1}, r)
$$

Here $s_0$ is the initial system/user prompt (the task), each $a_t$ is an **action** the model generates (which may be free-form reasoning, a tool call, or a final answer), each $o_t$ is an **observation** returned by the environment after executing $a_{t-1}$, and $r$ is the reward — usually a single terminal scalar, sometimes a vector of per-turn rewards. The whole trajectory is flattened into one token sequence that the model sees as its context window.

The critical structural fact: **the model authored some of those tokens and the environment authored the rest.** The actions $a_t$ are sampled from the policy — they are on-policy, differentiable, and carry a `log_prob`. The observations $o_t$ are *injected* into the context by the environment — they were never sampled from $\pi_\theta$, they have no meaningful log-prob under the policy, and back-propagating a policy-gradient loss through them is simply *wrong*. We will return to this masking question in its own section because it is the number-one footgun.

Let us make the data structure concrete. A clean way to represent a trajectory is as a list of *segments*, each tagged with who produced it and whether it should receive a loss.

```python
from dataclasses import dataclass, field
from enum import Enum
import torch

class Role(Enum):
    SYSTEM = "system"          # task description, tool schemas (no loss)
    ASSISTANT = "assistant"    # model-generated action (LOSS — this is the policy)
    OBSERVATION = "observation"  # tool output / env feedback (no loss, MASKED)

@dataclass
class Segment:
    role: Role
    token_ids: list[int]       # tokens for this segment
    logprobs: list[float] | None = None  # only for ASSISTANT segments (from rollout)

@dataclass
class Trajectory:
    segments: list[Segment] = field(default_factory=list)
    reward: float = 0.0                 # terminal reward
    turn_rewards: list[float] | None = None  # optional per-turn dense rewards
    metadata: dict = field(default_factory=dict)  # task_id, success flag, n_turns...

    def flatten(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Concatenate all segments into one token sequence plus a 0/1
        'loss mask' that is 1 exactly on ASSISTANT tokens. This mask is
        the single most important object in agentic RL — get it wrong
        and you train the model to predict tool outputs (catastrophic).
        """
        all_ids: list[int] = []
        loss_mask: list[int] = []
        for seg in self.segments:
            all_ids.extend(seg.token_ids)
            is_action = 1 if seg.role == Role.ASSISTANT else 0
            loss_mask.extend([is_action] * len(seg.token_ids))
        return (
            torch.tensor(all_ids, dtype=torch.long),
            torch.tensor(loss_mask, dtype=torch.float),
        )
```

This `Trajectory` object is the in-memory representation that flows from the rollout engine to the trainer. Everything else in this chapter is about how it is produced, how the reward is assigned to its tokens, and how it is batched. Compare this to the single-turn rollout buffer described in [The Anatomy of an RL-for-LLM System](../06-rl-infra/01-anatomy-rl-system.html): the difference is precisely the multiplicity of segments and the loss mask.

### Why not just treat the whole trajectory as one big completion?

A tempting shortcut: flatten the trajectory into one string and run ordinary GRPO on it as if it were a single completion. This is wrong for two reasons.

First, the **loss mask**. The flattened string contains tool outputs the model never generated. If you compute the policy-gradient loss over those tokens, you are pushing the policy to *predict* the environment's responses — to predict the output of a search engine or a Python interpreter from its own weights. At best this wastes capacity; at worst it makes the model hallucinate tool outputs and stop actually calling tools. You must zero the loss on observation tokens.

First and a half, even the *log-probabilities under the old policy* must be computed correctly: the importance ratio in PPO/GRPO is $\pi_\theta(a_t)/\pi_{\text{old}}(a_t)$ over action tokens only, conditioned on everything before — including observations — as context.

Second, **the rollout was not generated in one autoregressive pass.** The model generated $a_0$, then *stopped*, the environment ran, the observation was appended, and the model resumed. There may be sampling-distribution discontinuities at each boundary (different temperature for the final answer, forced formatting tokens, truncation). Treating it as one smooth generation hides those seams and corrupts the log-probs. We must track segment boundaries explicitly.

## The Rollout Loop: Interleaving Generation and Environment

The heart of agentic RL is a loop that alternates between *generating* an action and *stepping* an environment. This is structurally identical to the inference-time agentic loop of [The Agentic Loop: ReAct, Plan-Execute & Reflection](../08-agents-harness/02-agentic-loop.html) and [Tool Use & Function Calling](../08-agents-harness/01-tool-use-function-calling.html) — the difference is that during RL we *record* every token, its log-prob, and the segment boundaries so we can compute gradients later.

Here is a from-scratch rollout loop. It uses a generic environment interface (`reset`, `step`) modeled on the Gym/RAGEN convention, and an inference engine that exposes per-token log-probs (vLLM and SGLang both do).

```python
from typing import Protocol

class Environment(Protocol):
    """RAGEN/Gym-style environment interface for tool-using agents."""
    def reset(self, task) -> str:
        """Return the initial prompt (system + task) as a string."""
        ...
    def step(self, action_text: str) -> tuple[str, float, bool]:
        """
        Execute the model's action against the world.
        Returns (observation_text, turn_reward, done).
        - observation_text: what the model sees next (tool result, error, etc.)
        - turn_reward: dense per-turn reward (often 0.0 until terminal)
        - done: True if the episode has ended (success, failure, or give-up)
        """
        ...

def rollout(policy_engine, tokenizer, env: Environment, task,
            max_turns: int = 8, max_action_tokens: int = 512) -> Trajectory:
    """
    Run one agentic episode and record a fully-tagged Trajectory.
    `policy_engine.generate` returns (text, token_ids, logprobs) and STOPS
    at a tool-call boundary (a stop string like '</tool_call>' or an EOS).
    """
    traj = Trajectory()
    prompt = env.reset(task)

    # The initial prompt is context the model conditions on but never produced:
    # role=SYSTEM => loss_mask 0.
    sys_ids = tokenizer.encode(prompt)
    traj.segments.append(Segment(Role.SYSTEM, sys_ids))

    turn_rewards: list[float] = []
    running_text = prompt

    for turn in range(max_turns):
        # ---- 1. GENERATE an action (on-policy, differentiable) ----
        text, action_ids, logprobs = policy_engine.generate(
            running_text,
            max_tokens=max_action_tokens,
            stop=["</tool_call>", tokenizer.eos_token],
        )
        traj.segments.append(Segment(Role.ASSISTANT, action_ids, logprobs))
        running_text += text

        # ---- 2. STEP the environment with the action ----
        obs_text, turn_reward, done = env.step(text)
        turn_rewards.append(turn_reward)

        if done:
            traj.reward = sum(turn_rewards)          # terminal credit
            traj.turn_rewards = turn_rewards
            traj.metadata["n_turns"] = turn + 1
            return traj

        # ---- 3. INJECT the observation (NOT generated => loss_mask 0) ----
        # Wrap so the model can parse it, e.g. <tool_response>...</tool_response>
        obs_wrapped = f"<tool_response>\n{obs_text}\n</tool_response>"
        obs_ids = tokenizer.encode(obs_wrapped)
        traj.segments.append(Segment(Role.OBSERVATION, obs_ids))
        running_text += obs_wrapped

    # Hit max_turns without finishing: episode truncated.
    traj.reward = sum(turn_rewards)
    traj.turn_rewards = turn_rewards
    traj.metadata["n_turns"] = max_turns
    traj.metadata["truncated"] = True
    return traj
```

Three details deserve emphasis.

**Stop strings define turn boundaries.** The policy engine must stop generation exactly when the model emits a complete tool call (`</tool_call>`) or decides to give a final answer (EOS). This is why agentic RL is sensitive to the inference engine's stop-string handling — a missed stop string means the model keeps "generating" what should have been an observation, which poisons the trajectory. In practice you co-design the chat template, the tool-call format, and the stop strings together.

**Observations can themselves be enormous.** A single web-page fetch or a verbose stack trace can be thousands of tokens. Those tokens consume context budget and inference FLOPs on every subsequent turn but never receive a gradient. This is the agentic version of the context-management problem and connects directly to [Context Engineering & Management](../08-agents-harness/04-context-engineering.html); for RL it means trajectories have wildly variable, observation-dominated lengths.

**The same engine that serves inference must report log-probs.** In single-turn RL you can sometimes get away with recomputing log-probs in the trainer. In multi-turn RL, recomputation must replay the *exact* interleaving — same observations, same boundaries — or the ratios are wrong. Storing the rollout log-probs per action token (as we do in `Segment.logprobs`) and reconciling them against a recomputed forward pass is the safe pattern; see the discussion of inference/training log-prob mismatch in [The Generation–Training Loop & Rollout Engines](../06-rl-infra/02-generation-training-loop.html).

{{fig:agentic-trajectory-loss-mask}}

## Masking: The Single Most Important Detail

We have flagged it repeatedly; now we make it precise and show how the mask propagates all the way into the loss.

The policy gradient for a trajectory, in its REINFORCE-with-baseline form, is

$$
\nabla_\theta J = \mathbb{E}_\tau\!\left[\sum_{t} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, \hat{A}_t\right]
$$

where the sum runs over **action tokens only**. The observations $o_t$ appear in $s_t$ (they are part of the context the next action conditions on) but they are *never* arguments to $\log \pi_\theta$. The loss mask is exactly the indicator that token $i$ is an action token.

In a token-level PPO/GRPO objective this becomes

$$
\mathcal{L} = -\frac{1}{\sum_i m_i}\sum_{i} m_i \cdot \min\!\Big(\rho_i \hat{A}_i,\; \operatorname{clip}(\rho_i, 1-\epsilon, 1+\epsilon)\,\hat{A}_i\Big)
$$

with $m_i \in \{0,1\}$ the loss mask and $\rho_i = \pi_\theta(t_i)/\pi_{\text{old}}(t_i)$ the per-token importance ratio. Every term where $m_i = 0$ — every observation token — drops out. Here is the masked loss in code, written to mirror exactly what veRL or TRL's experimental multi-turn paths do internally.

```python
import torch
import torch.nn.functional as F

def masked_grpo_loss(logits, token_ids, loss_mask, advantages,
                     old_logprobs, epsilon=0.2):
    """
    Token-level clipped surrogate with a loss mask.

    logits:       [B, L, V] current-policy logits (teacher-forced on the trajectory)
    token_ids:    [B, L]    the realized tokens (actions AND observations)
    loss_mask:    [B, L]    1.0 on action tokens, 0.0 on observation/system tokens
    advantages:   [B, L]    per-token advantage (broadcast from trajectory or turn)
    old_logprobs: [B, L]    log-probs recorded at rollout time (behaviour policy)
    """
    # Shift for next-token prediction: logits at position i predict token i+1.
    logp_all = F.log_softmax(logits[:, :-1, :], dim=-1)        # [B, L-1, V]
    target = token_ids[:, 1:]                                   # [B, L-1]
    new_logprobs = torch.gather(
        logp_all, dim=-1, index=target.unsqueeze(-1)
    ).squeeze(-1)                                               # [B, L-1]

    # Align the other tensors to the shifted positions.
    mask = loss_mask[:, 1:]
    adv  = advantages[:, 1:]
    old  = old_logprobs[:, 1:]

    ratio = torch.exp(new_logprobs - old)                      # ρ_i
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * adv
    per_token_loss = -torch.min(unclipped, clipped)            # PPO surrogate

    # Mask out observation/system tokens, then average over ACTION tokens only.
    denom = mask.sum().clamp(min=1.0)
    loss = (per_token_loss * mask).sum() / denom
    return loss
```

!!! warning "Common pitfall"

    The most insidious masking bug is *off-by-one*. Because of the next-token shift, the loss at position $i$ trains the prediction of token $i{+}1$. If your mask marks "this token is an action," but you apply it *before* the shift, you will (a) compute a loss on the last system/observation token that precedes the first action token's first prediction, and (b) drop a loss on the boundary between the last action token of a turn and the first observation token. Always shift the mask the same way you shift the targets, and unit-test it: feed a trajectory where you *know* exactly which positions should be live, and assert `mask.sum()` equals your hand count.

A second, subtler masking issue is the **loss-normalization denominator**. Should you divide by the number of action tokens in the *trajectory*, in the *batch*, or in the *turn*? Dividing by the per-trajectory token count gives every trajectory equal weight regardless of length; dividing by the batch total gives every *token* equal weight, which over-weights long trajectories. This is the exact same length-bias controversy discussed for single-turn GRPO in [GRPO, RLOO & Critic-Free RL](../05-posttraining-alignment/08-grpo-rloo.html), but it bites harder here because trajectory lengths vary by an order of magnitude (a 1-turn success versus an 8-turn flailing failure). Most production systems normalize per-trajectory and then average over trajectories, which neutralizes length bias.

## Credit Assignment: Trajectory-Level vs Turn-Level

The reward usually lands at the end: did the agent solve the bug, find the answer, win the game? But the trajectory has many actions, only some of which were good. **Credit assignment** is the problem of deciding how much of the terminal reward each action deserves. There are three regimes in practice.

### Outcome reward, broadcast to all actions (trajectory-level)

The simplest scheme: compute one scalar reward $r$ for the whole trajectory and assign the *same* advantage to every action token. With a GRPO-style group baseline, you sample $G$ trajectories for the same task, compute each trajectory's terminal reward, and use the group mean as the baseline:

$$
\hat{A}^{(g)} = \frac{r^{(g)} - \mu}{\sigma + \delta}, \qquad
\mu = \frac{1}{G}\sum_{g} r^{(g)}, \qquad
\sigma = \sqrt{\tfrac{1}{G}\sum_g (r^{(g)}-\mu)^2}
$$

Every action token in trajectory $g$ — across all its turns — receives $\hat{A}^{(g)}$. This is **trajectory-level credit assignment**: the policy is told "this whole episode was above/below average," and it is up to the credit-assignment magic of stochastic gradient descent over many trajectories to figure out which turns mattered. It is robust, requires no per-turn reward signal, and is what most RLVR-style agentic recipes use. Its weakness is high variance and slow learning on long horizons: a single bad action in an otherwise good 8-turn trajectory still gets a positive advantage.

```python
def assign_trajectory_advantage(group: list[Trajectory], delta: float = 1e-6):
    """
    GRPO-style group-relative advantage at the TRAJECTORY level.
    All trajectories in `group` are rollouts for the SAME task/prompt.
    Returns each trajectory annotated with a scalar advantage that will
    be broadcast onto every one of its action tokens.
    """
    rewards = torch.tensor([t.reward for t in group])
    mu = rewards.mean()
    sigma = rewards.std(unbiased=False)
    for t, r in zip(group, rewards):
        adv = (r - mu) / (sigma + delta)
        t.metadata["advantage"] = adv.item()   # one number for the whole trajectory
    return group
```

### Turn-level / process rewards (per-turn credit)

If the environment can emit a meaningful reward *per turn* — a unit test that now passes, a sub-goal reached, a retrieved document that contained the answer — we can assign credit at turn granularity. Two ways to do this:

**Dense turn rewards with discounting.** Treat each turn as a step in an MDP and compute a discounted return-to-go for each action:

$$
G_t = \sum_{k=t}^{T-1} \gamma^{\,k-t}\, r_k
$$

The action at turn $t$ gets advantage based on $G_t$ rather than the full-trajectory $r$. Discounting $\gamma < 1$ means later actions get more credit for the rewards near them. You can run full GAE (Generalized Advantage Estimation) over turns if you have a value head, exactly as in PPO; the difference from token-level GAE is that the "step" is a whole turn, not a token. See [Advantage Estimation, KL Control & Stability Tricks](../06-rl-infra/09-advantage-kl-tricks.html) for the GAE machinery.

**Turn-level group baselines.** A nice critic-free variant: define the advantage of a turn as the difference its action made to the *outcome*, estimated by branching. Roll out $G$ continuations *from the same intermediate state* and use their mean outcome as the value of that state. The advantage of an action is then the group baseline of the *next* state minus the baseline of the current state — a Monte-Carlo, critic-free temporal-difference estimate. This is expensive (you branch at every turn) but gives genuinely per-turn credit without a learned value function.

!!! example "Worked example: trajectory-level vs turn-level credit"

    A code-fixing agent attempts a bug across 3 turns. The terminal reward is the fraction of unit tests passing at the end: $r = 0.75$ (3 of 4 tests pass). The per-turn signals the environment can emit are: tests passing *after each turn* = $[0.25, 0.25, 0.75]$, so the per-turn *deltas* (new tests fixed) are $r_0 = 0.25$, $r_1 = 0.00$, $r_2 = 0.50$.

    **Trajectory-level.** We sampled $G = 4$ trajectories for this bug with terminal rewards $[0.75, 0.50, 0.00, 0.25]$. Mean $\mu = 0.375$, std $\sigma \approx 0.275$. Our trajectory's advantage is

    $$\hat{A} = \frac{0.75 - 0.375}{0.275 + 10^{-6}} \approx +1.36$$

    Every action token in all 3 turns gets $+1.36$ — including turn 1, which fixed *nothing*. The signal is correct on average (this was a good trajectory) but rewards the wasted middle turn.

    **Turn-level with $\gamma = 0.9$.** Returns-to-go:

    $$G_0 = 0.25 + 0.9(0.00) + 0.9^2(0.50) = 0.25 + 0 + 0.405 = 0.655$$
    $$G_1 = 0.00 + 0.9(0.50) = 0.450$$
    $$G_2 = 0.50$$

    Now turn 1's action carries $G_1 = 0.45$ — still positive (it set up the winning turn 2) but visibly *less* credit than turn 0's $0.655$ or the decisive turn 2. After group-normalizing these per-turn returns against the same-state baselines, turn 1's advantage would be near zero, correctly identifying it as neutral. The turn-level scheme has lower variance and assigns blame/credit far more precisely — at the cost of needing a per-turn reward signal the trajectory-level scheme does not require.

### When to use which

{{fig:credit-assignment-traj-vs-turn}}

Use **trajectory-level** credit when the only honest reward is terminal (did the final answer match? did the PR merge?) and you cannot decompose it — this is the common case and it is what verifiable-reward agentic RL defaults to. Reach for **turn-level** credit when the environment naturally exposes intermediate progress (test pass counts, game score, retrieval hits) and trajectories are long enough that broadcast advantage learns too slowly. Process reward models (PRMs), which *learn* to score intermediate steps, are a third option discussed in [The RLHF Pipeline & Reward Modeling](../05-posttraining-alignment/05-rlhf-reward-modeling.html); they are powerful but reward-hackable, so verifiable per-turn signals are preferred when available.

## Environment Design: ReAct and the RAGEN Pattern

An agentic RL system is only as good as its environments. An environment is a tiny piece of software with a `reset`/`step` contract that (a) renders the world into text the model can read, (b) parses the model's action text into a structured operation, (c) executes it safely, and (d) emits a reward. The ReAct ("Reason + Act") format — interleaving `Thought:` reasoning with `Action:` tool calls and `Observation:` results — is the dominant interaction protocol, and frameworks like RAGEN generalize it into a Gym-style API specifically built for training rather than just inference.

Here is a complete, runnable environment for a multi-turn tool-using search-and-answer task, written in the RAGEN/ReAct style.

```python
import re

class SearchQAEnv:
    """
    A multi-turn ReAct environment: the agent must answer a question by
    issuing <search>query</search> calls and finally <answer>...</answer>.
    Terminal reward = 1.0 for a correct answer, with a small format bonus
    and a per-turn step penalty to discourage dithering.
    """
    SYSTEM_PROMPT = (
        "You are a research agent. To find information, emit "
        "<search>your query</search>. When you know the answer, emit "
        "<answer>your final answer</answer>. Think step by step before acting."
    )

    def __init__(self, retriever, max_turns: int = 6, step_penalty: float = 0.02):
        self.retriever = retriever          # callable: query -> list[str] passages
        self.max_turns = max_turns
        self.step_penalty = step_penalty

    def reset(self, task: dict) -> str:
        # task = {"question": ..., "answer": "<gold>"}
        self.gold = task["answer"]
        self.turns = 0
        return f"{self.SYSTEM_PROMPT}\n\nQuestion: {task['question']}\n"

    def step(self, action_text: str) -> tuple[str, float, bool]:
        self.turns += 1
        step_cost = -self.step_penalty       # discourage long episodes

        # --- Parse a final answer first ---
        ans = re.search(r"<answer>(.*?)</answer>", action_text, re.DOTALL)
        if ans:
            pred = ans.group(1).strip()
            correct = self._match(pred, self.gold)
            reward = (1.0 if correct else 0.0) + step_cost
            return ("", reward, True)        # terminal

        # --- Parse a search action ---
        srch = re.search(r"<search>(.*?)</search>", action_text, re.DOTALL)
        if srch:
            query = srch.group(1).strip()
            passages = self.retriever(query)[:3]
            obs = "\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
            done = self.turns >= self.max_turns
            # If we ran out of turns mid-search, that's a failure (reward 0).
            return (obs if not done else "Out of turns.", step_cost, done)

        # --- Malformed action: tell the model, penalize lightly ---
        obs = ("Invalid action. Use <search>...</search> or "
               "<answer>...</answer>.")
        done = self.turns >= self.max_turns
        return (obs, step_cost, done)

    @staticmethod
    def _match(pred: str, gold: str) -> bool:
        norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
        return norm(gold) in norm(pred)      # lenient containment match
```

This tiny class encodes several environment-design lessons that matter enormously for training dynamics.

**The reward shapes behavior, and the model will exploit every loophole.** The step penalty discourages dithering, but set it too high and the model rushes to a guess; set it to zero and the model learns to "search forever" because searching is free and occasionally helps. The lenient `_match` invites reward hacking — the model can pad its answer with many candidate strings hoping one matches. Reward design for agents is the most leverage-heavy and most dangerous part of the system; the failure modes are catalogued in [Reward Hacking, Over-Optimization & Alignment Failures](../05-posttraining-alignment/13-reward-hacking-failures.html), and the engineering of verifiers and sandboxes in [Reward Engineering, Verifiers & Sandboxes](../06-rl-infra/08-reward-verifiers-sandboxes.html).

**Action parsing must be forgiving but deterministic.** Early in training the policy emits malformed actions constantly. If a malformed action crashes the environment, the rollout dies and you lose the gradient signal. Instead, return a structured error observation (as above) so the model gets a learning signal that teaches it the format. This single design choice — *errors are observations, not exceptions* — is what makes format-following emerge from RL rather than requiring exhaustive SFT.

**Safety and isolation.** A code-execution or shell environment runs *model-generated code*. During RL the model will, with certainty, eventually generate `rm -rf /`, an infinite loop, or a fork bomb — not maliciously but because it is exploring. Every such environment must run in a sandbox (container, gVisor, firejail, ephemeral VM) with CPU/memory/time limits and no network unless required. This is non-negotiable infrastructure; see [Reward Engineering, Verifiers & Sandboxes](../06-rl-infra/08-reward-verifiers-sandboxes.html) and [Harness Engineering: Building a Coding Agent](../08-agents-harness/03-harness-coding-agent.html).

### Statefulness and reset semantics

Single-turn environments are stateless: each `step` is independent. Agentic environments are **stateful** — a filesystem accumulates edits, a database accumulates writes, a browser accumulates history. This has two consequences for the RL system. First, `reset` must truly reset: a leaked file from a previous episode is a silent source of reward leakage and non-reproducibility. Second, for group sampling (GRPO with $G$ rollouts of the same task) you need $G$ *independent* environment instances, because the rollouts interact with their own copy of the world. A single shared environment would let trajectory 3's writes corrupt trajectory 4's reads. The cost of $G$ parallel sandboxes is a real budget line in agentic RL.

## The Hard Problems

Multi-turn RL concentrates almost every open problem in RL-for-LLMs. We treat the four that dominate engineering effort.

### 1. Long-horizon credit assignment

With 8 turns and 500 action tokens each, a trajectory has ~4000 action tokens, and a single terminal reward must be smeared across all of them. The variance of the gradient grows with horizon, and the signal-to-noise per action plummets. Mitigations, in rough order of how often they are used: (a) **group baselines** to reduce variance for free; (b) **turn-level or process rewards** to localize credit; (c) **return discounting** $\gamma<1$ so distant rewards do not dominate nearby actions; (d) **curriculum** — start with short-horizon tasks and grow the horizon as the policy improves; (e) **hindsight relabeling** — if a "failed" trajectory accidentally achieved *some* goal, relabel it as a success for that goal (powerful but tricky to apply to language tasks). The honest state of the art is that long-horizon agentic RL is *fragile*, and the most reliable lever is a denser, more verifiable reward rather than a cleverer estimator.

### 2. Partial rollouts and length skew

Trajectories have wildly different lengths: a lucky 1-turn success finishes in 200 ms while an 8-turn failure with three 4000-token web fetches takes 30 s. In a synchronous batch, the whole training step waits for the slowest trajectory — the **straggler problem**, magnified relative to single-turn RL because the length distribution is far heavier-tailed. Two structural responses:

**Partial rollouts (continuation across steps).** Instead of waiting for every trajectory to finish, cap the *generation budget* per step. Trajectories that hit the cap are paused mid-episode, their state (KV cache, conversation so far) is saved, and they are *resumed* in the next iteration — possibly under updated weights. This keeps every rollout worker busy and bounds step latency, at the cost of trajectories that span multiple policy versions (a mild off-policyness, handled by importance sampling). This is the agentic generalization of the partial-rollout idea in async RL systems; see [Prime-RL, Async RL & Decentralized Training](../06-rl-infra/06-prime-rl-async.html) and [The Generation–Training Loop & Rollout Engines](../06-rl-infra/02-generation-training-loop.html).

**Length-balanced batching and continuous batching.** Group trajectories so that the per-microbatch length is roughly constant, and use the inference engine's continuous batching so that finished sequences are evicted and new ones admitted without draining the batch — exactly the mechanism described in [Continuous Batching & Request Scheduling](../07-inference-serving/02-continuous-batching.html). The interaction between continuous batching and the *interleaved* generation of agentic rollouts (each sequence repeatedly pausing to call a tool) is one of the genuinely hard scheduling problems in this space.

### 3. The throughput tax of environment interaction

In single-turn RL, the GPU generates continuously. In agentic RL, every turn the GPU *stops*, waits for the environment (a Python sandbox spin-up, a search API round-trip, a 10-second test suite), then resumes. During that wait the expensive accelerators are idle. If environment latency is comparable to generation latency — and for code execution or web tools it often *exceeds* it — you can lose half your GPU-hours to waiting.

{{fig:rollout-throughput-sync-vs-async}}

The standard fix is **asynchronous, decoupled rollout**: run many environment instances concurrently so that while one trajectory waits on its tool, the GPU generates the next action for another trajectory. This requires an async rollout engine (vLLM's and SGLang's async APIs) feeding a pool of environment workers, with a scheduler that interleaves their requests. Architecturally this is the colocated-vs-disaggregated question of [Colocated vs Disaggregated RL & Weight Synchronization](../06-rl-infra/07-colocated-vs-disaggregated.html) applied to environments: you want the inference engine saturated by a *fan-out* of environments, not blocked on any one of them.

!!! tip "Practitioner tip"

    Profile your rollout with two numbers: **GPU-generate time** and **environment-wait time** per trajectory. If env-wait is more than ~20% of generate time, you are leaving real money on the table and should add environment concurrency (more parallel env workers per inference replica) before you touch any algorithmic knob. A common, embarrassing discovery is that a single un-pooled sandbox `docker run` per turn (cold-start ~1–2 s) dominates the entire training budget; pre-warming a pool of containers can cut wall-clock time by several-fold with zero ML changes.

### 4. Inference/training distribution mismatch, compounded

Single-turn RL already suffers a gap between the inference engine's sampling distribution (vLLM in bf16 with its kernels) and the trainer's recomputed log-probs (FSDP forward, possibly different precision). In multi-turn RL this gap **compounds across turns**: a tiny per-token discrepancy at turn 0 changes the action, which changes the observation, which changes the entire downstream trajectory. The model that "generated" the trajectory under the inference engine is not exactly the model the trainer thinks generated it. The defenses — recompute log-probs in the trainer for the importance ratio, keep precision consistent, clip ratios aggressively, and monitor the inference-vs-train log-prob gap as a first-class metric — are detailed in [The Generation–Training Loop & Rollout Engines](../06-rl-infra/02-generation-training-loop.html) and [Advantage Estimation, KL Control & Stability Tricks](../06-rl-infra/09-advantage-kl-tricks.html). The agentic-specific advice is to *store and reconcile* per-action-token log-probs at segment boundaries, never to trust a single end-to-end recomputation of a multi-turn string.

## Putting It Together: A Minimal Agentic GRPO Step

Let us assemble the pieces into one training step that ties rollout, masking, credit assignment, and the masked loss together. This is the skeleton that frameworks like veRL's multi-turn path and TRL's experimental agentic trainers implement; see [veRL: HybridFlow & The Single-Controller Architecture](../06-rl-infra/04-verl.html) and [TRL: HuggingFace's RL Library](../06-rl-infra/03-trl.html) for the production versions.

```python
import torch

def agentic_grpo_step(policy, ref_policy, policy_engine, tokenizer,
                      env_factory, tasks, G=8, beta=0.02, optimizer=None):
    """
    One GRPO step over agentic trajectories.
      policy        : trainable model (FSDP-wrapped); provides logits for loss
      ref_policy    : frozen reference for KL (see [9])
      policy_engine : inference engine that generates actions + logprobs
      env_factory   : callable -> fresh, isolated Environment instance
      tasks         : list of tasks; we sample G rollouts per task
    """
    all_trajs: list[Trajectory] = []

    # ---- 1. ROLLOUT: G independent episodes per task (group sampling) ----
    for task in tasks:
        group = []
        for _ in range(G):
            env = env_factory()                  # ISOLATED env per rollout
            traj = rollout(policy_engine, tokenizer, env, task)
            group.append(traj)
        # ---- 2. CREDIT ASSIGNMENT: group-relative advantage per trajectory ----
        assign_trajectory_advantage(group)       # sets t.metadata['advantage']
        all_trajs.extend(group)

    # ---- 3. BATCH: flatten + pad trajectories, build per-token tensors ----
    seqs, masks, advs, old_lps = [], [], [], []
    max_len = max(sum(len(s.token_ids) for s in t.segments) for t in all_trajs)
    for t in all_trajs:
        ids, mask = t.flatten()                  # [L], [L]
        A = t.metadata["advantage"]
        adv = mask * A                           # broadcast scalar adv onto ACTION tokens
        # gather rollout-time logprobs aligned to every token (0 where masked)
        lp = torch.zeros_like(ids, dtype=torch.float)
        cursor = 0
        for s in t.segments:
            n = len(s.token_ids)
            if s.role == Role.ASSISTANT and s.logprobs is not None:
                lp[cursor:cursor+n] = torch.tensor(s.logprobs)
            cursor += n
        pad = max_len - len(ids)
        seqs.append(torch.nn.functional.pad(ids, (0, pad)))
        masks.append(torch.nn.functional.pad(mask, (0, pad)))
        advs.append(torch.nn.functional.pad(adv, (0, pad)))
        old_lps.append(torch.nn.functional.pad(lp, (0, pad)))

    token_ids   = torch.stack(seqs).cuda()       # [B, L]
    loss_mask   = torch.stack(masks).cuda()      # [B, L]
    advantages  = torch.stack(advs).cuda()       # [B, L]
    old_logprobs = torch.stack(old_lps).cuda()   # [B, L]

    # ---- 4. FORWARD + MASKED LOSS (teacher-forced over the trajectory) ----
    logits = policy(token_ids).logits            # [B, L, V]
    pg_loss = masked_grpo_loss(
        logits, token_ids, loss_mask, advantages, old_logprobs, epsilon=0.2,
    )

    # ---- 5. KL penalty to the reference, masked to ACTION tokens ----
    with torch.no_grad():
        ref_logits = ref_policy(token_ids).logits
    kl = masked_token_kl(logits, ref_logits, token_ids, loss_mask)
    loss = pg_loss + beta * kl

    # ---- 6. OPTIMIZE ----
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    return {"pg_loss": pg_loss.item(), "kl": kl.item(),
            "mean_reward": sum(t.reward for t in all_trajs) / len(all_trajs),
            "mean_turns": sum(t.metadata["n_turns"] for t in all_trajs) / len(all_trajs)}


def masked_token_kl(logits, ref_logits, token_ids, loss_mask):
    """Per-token KL(π_θ || π_ref) on realized tokens, averaged over ACTION tokens."""
    lp  = torch.log_softmax(logits[:, :-1], dim=-1)
    rlp = torch.log_softmax(ref_logits[:, :-1], dim=-1)
    tgt = token_ids[:, 1:].unsqueeze(-1)
    lp_tok  = lp.gather(-1, tgt).squeeze(-1)
    rlp_tok = rlp.gather(-1, tgt).squeeze(-1)
    # k3 estimator: exp(Δ) - Δ - 1, low-variance and non-negative
    diff = rlp_tok - lp_tok
    per_token_kl = torch.exp(diff) - diff - 1.0
    m = loss_mask[:, 1:]
    return (per_token_kl * m).sum() / m.sum().clamp(min=1.0)
```

Read this against the single-turn GRPO loop in [TRL: HuggingFace's RL Library](../06-rl-infra/03-trl.html). The algorithm is the same — group baseline, clipped surrogate, KL to reference. Everything that is *new* is bookkeeping: the trajectory data structure, the loss mask that survives flattening and padding, the broadcast of one advantage onto many action tokens across many turns, and the storage/alignment of rollout-time log-probs across segment boundaries. That bookkeeping is where agentic RL implementations live or die.

!!! interview "Interview Corner"

    **Q:** You are training a tool-using agent with GRPO. Your reward steadily climbs, but when you deploy the model it has *stopped calling tools* and just hallucinates plausible-looking tool outputs inline. Training reward looks great. What most likely went wrong, and how would you diagnose and fix it?

    **A:** This is the classic missing-loss-mask bug. If observation (tool-output) tokens are *not* masked from the policy-gradient loss, the model is trained to predict the tool outputs as if it had generated them — so it learns to produce fluent fake observations and skip the actual tool call, which conveniently also lets it "match" the reward pattern it saw in successful trajectories. **Diagnose:** (1) log `loss_mask.sum()` versus total tokens per batch — if the mask is live on observation spans, that is the smoking gun; (2) inspect a few flattened trajectories and assert that the mask is zero over every `<tool_response>` span (a unit test with a hand-counted trajectory); (3) check the off-by-one — the mask must be shifted identically to the next-token targets. **Fix:** zero the loss (and the KL, and the importance ratio's contribution) on all observation and system tokens, normalize over action tokens only, and re-verify the mask with the unit test. A secondary contributor can be a reward that does not *require* a real tool round-trip (e.g., a verifier that only checks the final answer string), which lets the model get away with faking; tighten the environment so that a final answer is only accepted after genuine tool interactions, or add a verifier that checks tool calls actually executed.

## Key Takeaways

!!! key "Key Takeaways"

    - In multi-turn agentic RL the unit of data is a **trajectory** of interleaved model actions and environment observations; the model authored only the action tokens, and those are the only tokens that get a policy-gradient loss.
    - The **loss mask** (1 on action tokens, 0 on observations/system) is the single most important and most bug-prone object; mask the surrogate loss, the KL, and the importance ratio, normalize over action tokens only, and watch the next-token off-by-one.
    - **Credit assignment** ranges from trajectory-level (one terminal reward broadcast to all actions, robust but high-variance) to turn-level/process rewards (discounted returns-to-go or branched group baselines, lower variance but needs an intermediate signal).
    - Use **group-relative (GRPO-style) baselines** to cut variance for free; reach for turn-level credit only when the environment exposes honest intermediate progress.
    - **Environment design is the lever**: errors should be observations not exceptions (so format-following emerges), every code/shell environment must be sandboxed, and group sampling needs $G$ *independent* environment instances to avoid state leakage.
    - The hard problems are **long-horizon credit assignment**, **partial rollouts / length skew** (straggler-dominated, heavy-tailed lengths), the **throughput tax** of GPUs idling on environment latency, and a **compounded inference/training distribution mismatch** across turns.
    - The biggest practical wins are usually infrastructural — pooled/pre-warmed sandboxes, async decoupled rollout to hide environment latency, partial rollouts to bound step time — not algorithmic cleverness.
    - The training algorithm is the same GRPO/PPO as single-turn RL; what is new is the trajectory bookkeeping, the mask, and the credit broadcast. The downstream agent behavior these methods produce is the subject of [Part VIII](../08-agents-harness/02-agentic-loop.html).

!!! sota "State of the Art & Resources (2026)"
    Agentic and multi-turn RL has advanced rapidly since 2023: frameworks such as veRL, RAGEN, and TRL now ship production-ready multi-turn trainers, SWE-bench Verified has become the canonical coding-agent leaderboard, and the field is converging on trajectory-level GRPO with masked policy gradients as the baseline recipe while actively exploring turn-level credit assignment and async rollout architectures.

    **Foundational work**

    - [Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* (2022)](https://arxiv.org/abs/2210.03629) — establishes the interleaved Thought/Action/Observation loop that every tool-using agent and agentic RL environment now follows.
    - [Lightman et al., *Let's Verify Step by Step* (2023)](https://arxiv.org/abs/2305.20050) — introduces process reward models (PRMs) that assign credit at each reasoning step, motivating turn-level reward designs in agentic RL.
    - [Andrychowicz et al., *Hindsight Experience Replay* (2017)](https://arxiv.org/abs/1707.01495) — canonical technique for sparse long-horizon rewards via goal relabeling; the language-agent analogue is discussed in credit-assignment mitigations.

    **Recent advances (2023–2026)**

    - [Wang et al., *RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning* (2025)](https://arxiv.org/abs/2504.20073) — proposes StarPO for trajectory-level multi-turn RL, identifies the "Echo Trap" instability, and provides a Gym-style training framework.
    - [Wang & Ammanabrolu, *A Practitioner's Guide to Multi-Turn Agentic Reinforcement Learning* (2025)](https://arxiv.org/abs/2510.01132) — empirical map of design choices across environment, reward, and policy axes; the closest thing to a field-wide recipe guide.
    - [Wei et al., *Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Reward Design* (2025)](https://arxiv.org/abs/2505.11821) — first systematic study of turn-level reward signals for GRPO/PPO in multi-turn settings, directly relevant to the credit-assignment section.
    - [Jimenez et al., *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* (2024)](https://arxiv.org/abs/2310.06770) — the canonical benchmark driving multi-turn coding-agent RL, making long-horizon credit assignment a practical necessity.
    - [Yao et al., *τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains* (2024)](https://arxiv.org/abs/2406.12045) — evaluates agents on multi-turn tool-calling with a simulated user, exposing reliability failures that motivate better trajectory-level training.

    **Open-source & tools**

    - [verl-project/verl](https://github.com/verl-project/verl) — production RL post-training library (HybridFlow) with multi-turn rollout support, integrates FSDP, Megatron-LM, vLLM, and SGLang.
    - [RAGEN-AI/RAGEN](https://github.com/RAGEN-AI/RAGEN) — StarPO-based multi-turn agentic RL framework with 10 built-in Gym-compatible environments and trajectory filtering for training stability.

    **Go deeper**

    - [Xi et al., *AgentGym: Evolving LLM-based Agents across Diverse Environments* (2024)](https://arxiv.org/abs/2406.04151) — unified ReAct-format benchmark spanning 14 environments (web, games, coding, embodied) used to study generalist agent RL at scale.

## Further Reading

- **Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (2023)** — the Reason+Act interaction format that nearly every tool-using environment adopts; foundational for the trajectory structure in this chapter.
- **Wang et al., "RAGEN: Training Agents by Reinforcing Reasoning" / the RAGEN framework** — a StarPO-style multi-turn agentic RL framework with a Gym-like environment API built for training (not just inference); a concrete reference implementation of the patterns here.
- **Schulman et al., "Proximal Policy Optimization Algorithms" (2017)** and **Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (2016)** — the PPO clipped surrogate and GAE that underlie turn-level advantage estimation.
- **Shao et al., "DeepSeekMath" (2024)** — introduces GRPO and the group-relative advantage that the trajectory-level credit scheme builds on.
- **Andrychowicz et al., "Hindsight Experience Replay" (2017)** — the relabeling idea for sparse, long-horizon rewards referenced under long-horizon credit assignment.
- **veRL / HybridFlow (Sheng et al.)** and the **TRL** repository — production frameworks whose multi-turn rollout and masking code paths implement the bookkeeping described here; read their multi-turn trainers alongside [veRL: HybridFlow & The Single-Controller Architecture](../06-rl-infra/04-verl.html).
- **SWE-bench (Jimenez et al., 2024)** and **tau-bench / agentic tool-use benchmarks** — the evaluation targets that motivate multi-turn coding and tool-using RL; see [Agent Evaluation & Benchmarks](../08-agents-harness/08-agent-evaluation.html).

## Exercises

**1.** *(Conceptual.)* In the flattened trajectory, observation tokens (the `<tool_response>...</tool_response>` spans) are placed in the context window and the model conditions on them, yet the chapter insists they must receive `loss_mask = 0`. Explain precisely what the policy is trained to do if you leave the loss *on* over observation tokens, and connect this to the deployment-time failure described in the Interview Corner (a model that stops calling tools and hallucinates their outputs).

??? note "Solution"

    The policy-gradient / clipped-surrogate loss over a token $t_i$ pushes $\pi_\theta$ to increase (for positive advantage) or decrease (for negative advantage) the probability of *generating* $t_i$ given its context. Observation tokens were **not** sampled from $\pi_\theta$ — they were injected by the environment (a search engine, a Python interpreter). If those tokens are unmasked, the gradient trains the model to *predict the environment's output from its own weights*: to reproduce the text of a search result or a stack trace autoregressively.

    Two damaging consequences follow. First, it wastes capacity forcing the model to memorize/approximate tool outputs it cannot actually know. Second, and worse, it makes hallucinating a plausible observation a rewarded behavior: in successful trajectories the observation spans co-occur with high reward, so the unmasked loss reinforces "emit fluent fake tool output." At deployment the model then produces convincing but fabricated `<tool_response>` blocks inline and skips the real tool call entirely — exactly the Interview Corner failure. The fix is to zero the loss (and the KL term, and the importance-ratio contribution) on every observation and system token, and to normalize only over action tokens. The one place observations legitimately appear is *inside the context* $s_t$ that the next action conditions on — never as an argument to $\log \pi_\theta$.

**2.** *(Quantitative.)* You run GRPO with a group of $G = 4$ trajectories for one task. Their terminal rewards are $[1.0,\ 0.5,\ 0.5,\ 0.0]$. Using the chapter's trajectory-level advantage formula (population std, $\delta = 10^{-6}$), compute the group mean $\mu$, the std $\sigma$, and the scalar advantage $\hat{A}^{(g)}$ assigned to *each* of the four trajectories. Then state how many of the ~4000 action tokens in the first trajectory (8 turns, ~500 tokens each) receive that advantage, and why the middle turns get it whether or not they helped.

??? note "Solution"

    Mean: $\mu = \tfrac{1}{4}(1.0 + 0.5 + 0.5 + 0.0) = 0.5$.

    Deviations: $0.5,\ 0,\ 0,\ -0.5$; squared: $0.25,\ 0,\ 0,\ 0.25$; sum $= 0.5$.

    Population variance: $0.5 / 4 = 0.125$, so $\sigma = \sqrt{0.125} \approx 0.35355$.

    Advantages $\hat{A} = (r - \mu)/(\sigma + \delta)$:

    $$
    \hat{A}^{(1)} = \frac{1.0 - 0.5}{0.35355 + 10^{-6}} \approx +1.414, \quad
    \hat{A}^{(2)} = \hat{A}^{(3)} = \frac{0.5 - 0.5}{\ldots} \approx 0.000, \quad
    \hat{A}^{(4)} = \frac{0.0 - 0.5}{\ldots} \approx -1.414.
    $$

    Under trajectory-level (outcome) credit, the *single* scalar $\hat{A}^{(1)} \approx +1.414$ is broadcast onto **every** action token of trajectory 1 — all ~4000 of them across all 8 turns (system and observation tokens get 0 via the mask). Every action token gets the identical advantage because the scheme has no per-turn signal to distinguish turns; it only knows the whole episode finished above the group average. A wasted or even harmful middle turn is therefore reinforced exactly as much as the decisive turn. That indiscriminate broadcast is the source of the high variance and slow long-horizon learning the chapter warns about, and it is what turn-level credit (Exercise 3) is designed to fix.

**3.** *(Quantitative.)* An agent solves a task in 3 turns. The environment emits per-turn (delta) rewards $r_0 = 0.2$, $r_1 = 0.0$, $r_2 = 0.5$. Using the discounted return-to-go $G_t = \sum_{k=t}^{T-1}\gamma^{\,k-t} r_k$ with $\gamma = 0.8$, compute $G_0$, $G_1$, $G_2$. Which turn receives the most credit, and how does this compare with the trajectory-level scheme that would give all three turns the same advantage?

??? note "Solution"

    Work backwards ($T = 3$):

    $$G_2 = r_2 = 0.50$$
    $$G_1 = r_1 + \gamma\, G_2 = 0.0 + 0.8 \times 0.50 = 0.40$$
    $$G_0 = r_0 + \gamma\, G_1 = 0.2 + 0.8 \times 0.40 = 0.2 + 0.32 = 0.52$$

    So $G_0 = 0.52$, $G_1 = 0.40$, $G_2 = 0.50$.

    Turn 0 carries the most return-to-go ($0.52$) because it collects its own immediate reward *plus* the discounted value of everything that follows, including the decisive turn 2. The idle turn 1, whose own delta was $0.0$, still receives $0.40$ purely as the discounted credit for setting up turn 2 — but it is visibly the smallest of the three, correctly marking it as the least individually productive step.

    Contrast with trajectory-level credit: it would assign the *same* advantage (derived from the single terminal reward, e.g. $r = 0.7$ against a group baseline) to all three turns, giving turn 1 exactly as much credit as turns 0 and 2. The turn-level scheme localizes credit and lowers gradient variance — at the cost of needing the honest per-turn signal $[0.2, 0.0, 0.5]$ that the trajectory-level scheme does not require.

**4.** *(Quantitative.)* You profile one rollout worker: generating a full trajectory costs **4 s** of GPU time, and the environment interaction (sandbox spin-up + tool round-trips) costs **6 s** of wall-clock during which the GPU is idle, for a serial episode wall-clock of 10 s. (a) What fraction of GPU time is wasted, and does the practitioner-tip threshold say to act? (b) If you run several trajectories concurrently through one inference replica so the GPU serves other episodes' actions while any one waits on its tool, how many concurrent trajectories are needed to keep the GPU continuously busy, and what is the resulting throughput speedup?

??? note "Solution"

    (a) With a serial episode the GPU generates for 4 s and sits idle for 6 s out of every 10 s, so GPU utilization is $4/(4+6) = 40\%$ and **60% of GPU time is wasted**. The practitioner tip says to add environment concurrency whenever env-wait exceeds ~20% of generate time; here env-wait/generate $= 6/4 = 150\%$, far above threshold, so yes — fix the concurrency before touching any algorithmic knob.

    (b) To keep the GPU busy, while one trajectory spends its 6 s on the environment the GPU must have other trajectories' actions to generate. Each trajectory occupies the GPU $4$ s out of every $10$ s cycle, so the number of concurrent trajectories needed to fill the pipe is

    $$
    N = \left\lceil \frac{\text{generate} + \text{wait}}{\text{generate}} \right\rceil = \left\lceil \frac{10}{4} \right\rceil = \lceil 2.5 \rceil = 3.
    $$

    With 3 concurrent trajectories the GPU is essentially always generating (utilization $\to 100\%$), and throughput rises from one 4 s-of-GPU episode per 10 s to roughly one per 4 s — about a **2.5x speedup** (utilization $40\% \to 100\%$) with zero ML changes. This is exactly the async, decoupled-rollout fan-out the chapter recommends for hiding environment latency.

**5.** *(Implementation.)* The chapter implements `assign_trajectory_advantage` (one scalar broadcast to all action tokens). Implement its turn-level counterpart. Write `assign_turn_advantage(traj, gamma)` that (i) computes the discounted return-to-go $G_t$ for each turn from `traj.turn_rewards`, and (ii) returns a per-token advantage tensor, aligned to `traj.flatten()`, where each `ASSISTANT` segment's tokens carry that turn's $G_t$ and all `SYSTEM`/`OBSERVATION` tokens carry $0$. Assume `traj.turn_rewards[k]` is the reward for the $k$-th `ASSISTANT` action, in order. Keep it consistent with the chapter's `Trajectory`/`Segment`/`Role` classes.

??? note "Solution"

    The key alignment fact is that the `ASSISTANT` segments appear in the same order as `turn_rewards`, so we walk the segments while incrementing a turn counter only on assistant segments, and paint each action span with its return-to-go. Returns-to-go are computed with a single backward pass.

    ```python
    import torch

    def assign_turn_advantage(traj: Trajectory, gamma: float = 0.9) -> torch.Tensor:
        """
        Turn-level credit: discounted return-to-go G_t broadcast onto that
        turn's ACTION tokens; SYSTEM/OBSERVATION tokens get 0.
        Returns a [L] float tensor aligned to traj.flatten().
        """
        r = traj.turn_rewards or []
        T = len(r)

        # --- (i) returns-to-go G_t = sum_{k>=t} gamma^{k-t} r_k, one backward pass ---
        G = [0.0] * T
        running = 0.0
        for t in reversed(range(T)):
            running = r[t] + gamma * running
            G[t] = running

        # --- (ii) paint each ASSISTANT span with its turn's return-to-go ---
        ids, mask = traj.flatten()                 # ids:[L], mask:[L]
        adv = torch.zeros_like(mask)               # float [L], 0 everywhere
        cursor = 0
        turn = 0
        for seg in traj.segments:
            n = len(seg.token_ids)
            if seg.role == Role.ASSISTANT:
                # guard against a missing/short turn_rewards list
                g = G[turn] if turn < T else 0.0
                adv[cursor:cursor + n] = g
                turn += 1
            cursor += n
        return adv
    ```

    Sanity checks worth asserting in a test: `adv` has the same length as `ids`; `adv` is nonzero exactly where `mask == 1` (assuming all $G_t \neq 0$); and `(adv != 0).sum()` equals the total number of action tokens. This tensor can be fed straight into `masked_grpo_loss` as the `advantages` argument in place of the broadcast scalar `mask * A` used in `agentic_grpo_step`, giving genuinely per-turn credit rather than one trajectory-wide number.

**6.** *(Implementation, hard.)* The chapter's warning admonition says the most insidious masking bug is an *off-by-one*: the mask must be shifted the same way as the next-token targets, and you should unit-test it against a hand-counted trajectory. Build a trajectory with segment token counts `SYSTEM=3, ASSISTANT=2, OBSERVATION=4, ASSISTANT=2` (11 tokens total). By hand, (a) give the pre-shift `loss_mask`, (b) give the shifted mask `loss_mask[:, 1:]` that aligns with the shifted targets in `masked_grpo_loss`, and (c) write a runnable unit test that constructs this trajectory and asserts both the pre-shift and post-shift live-token counts.

??? note "Solution"

    (a) `flatten()` emits `mask = 1` exactly on `ASSISTANT` tokens. With order SYSTEM(3), ASSISTANT(2), OBSERVATION(4), ASSISTANT(2):

    ```
    index :  0 1 2 | 3 4 | 5 6 7 8 | 9 10
    role  :  S S S | A A | O O O O | A A
    mask  :  0 0 0 | 1 1 | 0 0 0 0 | 1 1     ->  mask.sum() = 4
    ```

    (b) `masked_grpo_loss` shifts targets to `token_ids[:, 1:]` (logits at position $i$ predict token $i{+}1$) and shifts the mask identically to `loss_mask[:, 1:]`, i.e. drop index 0:

    ```
    predicts token:  1 2 3 4 5 6 7 8 9 10
    shifted mask  :  0 0 1 1 0 0 0 0 1 1     ->  shifted sum = 4
    ```

    The count is preserved (still 4) because we dropped a masked (`0`) position at the front. The crucial correctness point: after the shift, the live positions are those whose *predicted* token (index $i{+}1$) is an action token — position 2 predicts token 3 (first action token), position 3 predicts token 4, position 8 predicts token 9, position 9 predicts token 10. If instead you had masked *before* shifting and then sliced the targets, you would leave a stray loss on the last pre-action token and drop the loss at the action-to-observation boundary — the exact off-by-one the admonition describes.

    (c) Runnable unit test:

    ```python
    import torch

    def make_test_traj() -> Trajectory:
        t = Trajectory()
        t.segments = [
            Segment(Role.SYSTEM,      [10, 11, 12]),        # 3
            Segment(Role.ASSISTANT,   [20, 21], [-0.1, -0.2]),  # 2 (action)
            Segment(Role.OBSERVATION, [30, 31, 32, 33]),    # 4
            Segment(Role.ASSISTANT,   [40, 41], [-0.3, -0.4]),  # 2 (action)
        ]
        return t

    def test_loss_mask_shift():
        traj = make_test_traj()
        ids, mask = traj.flatten()

        # (a) pre-shift: 11 tokens, exactly 4 action tokens
        assert ids.shape[0] == 11
        assert mask.tolist() == [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1]
        assert mask.sum().item() == 4.0

        # (b) post-shift (as used inside masked_grpo_loss): drop index 0
        shifted = mask[1:]                       # [B,L] -> here 1-D: [L-1]
        assert shifted.tolist() == [0, 0, 1, 1, 0, 0, 0, 0, 1, 1]
        assert shifted.sum().item() == 4.0

        # count is preserved because the dropped position was masked (0)
        assert mask.sum().item() == shifted.sum().item()
        print("mask shift OK: 4 live action tokens before and after shift")

    test_loss_mask_shift()
    ```

    Running it prints `mask shift OK: 4 live action tokens before and after shift`. Batching this trajectory just adds a leading dimension, turning `mask[1:]` into `loss_mask[:, 1:]` as written in `masked_grpo_loss`; the per-trajectory hand count of 4 is the ground truth you assert against.
