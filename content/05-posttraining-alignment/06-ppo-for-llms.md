# 5.6 Policy Gradients & PPO for Language Models

In [The RLHF Pipeline & Reward Modeling](../05-posttraining-alignment/05-rlhf-reward-modeling.html) we built two of the three pieces of Reinforcement Learning from Human Feedback (RLHF): a supervised fine-tuned (SFT) policy that can follow instructions, and a reward model (RM) that maps a prompt-response pair to a scalar "how good is this." This chapter is about the third piece — the part that actually *changes the policy's weights using the reward*. That part is reinforcement learning, and for years the default algorithm was **Proximal Policy Optimization (PPO)**. It is the optimizer behind InstructGPT and the original ChatGPT, and even though critic-free methods like GRPO ([GRPO, RLOO & Critic-Free RL](../05-posttraining-alignment/08-grpo-rloo.html)) and offline methods like DPO ([Direct Preference Optimization & Its Variants](../05-posttraining-alignment/07-dpo-and-variants.html)) have eaten much of its market share, PPO remains the canonical, most-general, and most-instructive RL algorithm for language models. Everything that came after is best understood as "PPO, but we deleted/changed *this* part."

The intellectual problem PPO solves is deceptively hard. Supervised fine-tuning has a differentiable target: for each token there is a "correct" next token, and we minimize cross-entropy. But "produce a response a human would prefer" gives you a single scalar score at the *end* of a whole generated sequence, with no per-token target and no gradient through the sampling step. We cannot backpropagate through "sample a token from a categorical distribution." Reinforcement learning is precisely the toolkit for optimizing an expected reward when the only thing you can do is *sample actions, observe a scalar, and nudge the action distribution.* This chapter develops that toolkit from first principles — the score-function (REINFORCE) estimator, baselines and the advantage, Generalized Advantage Estimation (GAE) — and then assembles the full PPO-for-LLM loss and training loop, with runnable code and a worked numerical example. We end with an honest accounting of *why PPO is finicky*, which is the reason the rest of Part V exists.

This chapter assumes you are comfortable with the transformer forward pass ([Building a GPT From Scratch](../02-transformer/07-build-gpt-from-scratch.html)), autodiff ([Automatic Differentiation & PyTorch Internals](../01-foundations/07-autodiff-pytorch.html)), and the probability background in [Probability, Statistics & Information Theory](../01-foundations/02-probability-information.html).

## Generation as a Markov decision process

To run RL on a language model we first have to *see* generation as a sequential decision problem. The mapping is exact and worth internalizing because every term in the PPO loss corresponds to one of these objects.

A **Markov decision process (MDP)** is a tuple $(\mathcal{S}, \mathcal{A}, P, r, \gamma)$: states, actions, a transition kernel $P(s'\mid s,a)$, a reward $r(s,a)$, and a discount $\gamma$. For autoregressive text generation, the correspondence is:

- **State** $s_t$ = the prompt plus all tokens generated so far: $s_t = (q, o_{<t})$ where $q$ is the prompt and $o_{<t} = (o_1, \dots, o_{t-1})$. The state is the full context window the model conditions on.
- **Action** $a_t = o_t$ = the next **token** emitted, drawn from the vocabulary $\mathcal{V}$. So the action space is huge but discrete: $|\mathcal{A}| = |\mathcal{V}|$, on the order of $10^5$.
- **Policy** $\pi_\theta(a_t \mid s_t) = \pi_\theta(o_t \mid q, o_{<t})$ = exactly the model's next-token softmax distribution. *The language model **is** the policy.* This is the single most important identification in the chapter.
- **Transition** $P(s_{t+1}\mid s_t, a_t)$ is **deterministic**: appending token $o_t$ to $s_t$ gives $s_{t+1} = (q, o_{\le t})$ with probability 1. There is no environment stochasticity; all randomness is in the policy's own sampling. This is a special, friendly property of the text MDP.
- **Reward.** In RLHF the reward is **terminal and sparse**: zero for every token until the sequence ends (at the EOS token or length cap), and then a single scalar $R(q, o)$ from the reward model. There is no per-token reward signal. (Contrast this with RL with verifiable rewards, where the terminal reward is a programmatic checker; see [RL with Verifiable Rewards & The Reasoning Recipe](../05-posttraining-alignment/09-rlvr-reasoning.html).)
- **Episode / trajectory** $\tau = (s_0, a_0, s_1, a_1, \dots)$ = one full generated response. The "horizon" is the response length $T = |o|$.

{{fig:ppo-mdp-trajectory}}

The objective we want to maximize is the **expected terminal reward** of responses sampled from our policy, over the distribution of prompts $\mathcal{D}$:

$$
J(\theta) = \mathbb{E}_{q \sim \mathcal{D}}\;\mathbb{E}_{o \sim \pi_\theta(\cdot \mid q)}\big[ R(q, o) \big].
$$

The whole game is: nudge $\theta$ so that responses the reward model likes become more probable. We almost always set the discount $\gamma = 1$ for LLMs (we care about the whole response equally; there is no infinite horizon to tame), with one important caveat about GAE we will return to.

!!! note "Aside: why determinism matters"
    Because the transition is deterministic and the only randomness is the policy's sampling, the text MDP is *much* nicer than robotics or Atari. There is no transition model to learn, no environment noise to average over, and a trajectory's probability is *exactly* the product of the policy's token probabilities — which we can compute in one forward pass. This is why "RL for LLMs" is, mechanically, mostly bookkeeping over log-probs rather than the harder problems classic RL faces.

## REINFORCE: the score-function gradient

### Deriving the policy gradient

We want $\nabla_\theta J(\theta)$, but $J$ is an expectation over a distribution that *depends on $\theta$* — we cannot just push the gradient inside, because the thing we average over is itself changing. The **score-function estimator** (a.k.a. the likelihood-ratio trick, a.k.a. REINFORCE) resolves this with one identity. Write the expectation as a sum (or integral) over trajectories, where $p_\theta(\tau)$ is the probability of trajectory $\tau$ under the policy:

$$
J(\theta) = \sum_\tau p_\theta(\tau)\, R(\tau), \qquad
\nabla_\theta J(\theta) = \sum_\tau \nabla_\theta p_\theta(\tau)\, R(\tau).
$$

Now use the **log-derivative identity** $\nabla_\theta p_\theta(\tau) = p_\theta(\tau)\,\nabla_\theta \log p_\theta(\tau)$ (which is just the chain rule on $\log$). Substituting turns the sum back into an expectation we can estimate by sampling:

$$
\nabla_\theta J(\theta) = \sum_\tau p_\theta(\tau)\,\nabla_\theta \log p_\theta(\tau)\, R(\tau)
= \mathbb{E}_{\tau \sim \pi_\theta}\big[ R(\tau)\, \nabla_\theta \log p_\theta(\tau) \big].
$$

This is the crucial move: the gradient of an expectation became an expectation of a gradient, *and the troublesome $\nabla R$ never appears* — we do not need the reward to be differentiable. Now expand $\log p_\theta(\tau)$. The trajectory probability factorizes; the transition terms $P(s_{t+1}\mid s_t,a_t)$ do not depend on $\theta$ (the environment is fixed), so their gradient is zero and they drop out:

$$
\log p_\theta(\tau) = \underbrace{\log p(s_0)}_{\nabla=0} + \sum_{t} \log \pi_\theta(a_t\mid s_t) + \underbrace{\sum_t \log P(s_{t+1}\mid s_t,a_t)}_{\nabla=0}.
$$

So $\nabla_\theta \log p_\theta(\tau) = \sum_t \nabla_\theta \log \pi_\theta(a_t\mid s_t)$, and we arrive at the **REINFORCE estimator**:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[ R(\tau) \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(o_t \mid q, o_{<t}) \right].
$$

In words: **sample a response, then push up the log-probability of every token in it, scaled by the whole-trajectory reward.** Good responses (high $R$) make all their tokens more likely; bad responses make all theirs less likely. It is almost embarrassingly simple, and it is unbiased.

```python
import torch
import torch.nn.functional as F

def reinforce_loss(token_logprobs, response_mask, rewards):
    """
    Vanilla REINFORCE for a batch of sampled responses.

    token_logprobs : (B, T) log pi_theta(o_t | q, o_<t) for the SAMPLED tokens.
    response_mask  : (B, T) 1.0 on generated tokens, 0.0 on prompt/padding.
    rewards        : (B,)   scalar terminal reward per response.

    We MAXIMIZE E[R * sum_t logprob], so we MINIMIZE its negative.
    rewards are detached: they are weights, not differentiable targets.
    """
    seq_logprob = (token_logprobs * response_mask).sum(dim=-1)   # (B,) sum over tokens
    loss = -(rewards.detach() * seq_logprob).mean()
    return loss
```

### Why vanilla REINFORCE is unusable as-is

REINFORCE is unbiased but has *catastrophic variance*, for two reasons that we fix in the next two sections.

**Problem 1 — the common-mode offset.** Suppose your reward model outputs scores in, say, $[0, 10]$ (all positive). Then *every* sampled response has positive reward, so *every* response's tokens get pushed *up* — the gradient says "make everything more likely," which is incoherent (probabilities sum to 1). The only useful information is the *relative* differences between responses, and that signal is a tiny ripple on top of a huge positive DC offset. The estimator is unbiased, so in infinite samples the offset cancels (because $\sum_o \pi_\theta(o)=1$ makes a constant baseline contribute zero in expectation), but with a finite batch the variance is enormous. The fix is a **baseline** (next section).

**Problem 2 — credit assignment.** Multiplying *every* token's gradient by the *whole-trajectory* reward means a token early in the sequence is credited for rewards earned by tokens after it (fine, it caused them) but also blamed/credited symmetrically with no notion of which tokens actually mattered. For LLMs with a single terminal reward this is less acute than in dense-reward RL, but the **value function** and **advantage** still let us assign credit far more sharply. The fix is the actor-critic advantage and GAE (two sections down).

!!! example "Worked example: variance of REINFORCE with vs. without a baseline"
    Take a toy single-token "bandit" with three possible responses and current policy probabilities $\pi = (0.5, 0.3, 0.2)$ and rewards $R = (10.0, 9.5, 9.0)$ (note: all large and positive — a typical reward-model failure mode). The "gradient signal" for response $i$ is proportional to $R_i$ (no baseline) vs. $R_i - b$ (with baseline $b = \bar R$).

    Mean reward $\bar R = 0.5(10) + 0.3(9.5) + 0.2(9.0) = 5.0 + 2.85 + 1.8 = 9.65$.

    | response | $R_i$ | $R_i - \bar R$ |
    |---|---|---|
    | A | $10.0$ | $+0.35$ |
    | B | $9.5$  | $-0.15$ |
    | C | $9.0$  | $-0.65$ |

    Without the baseline, all three "signals" are around $+9$ to $+10$ — the estimator is dominated by a meaningless common offset of $\approx 9.65$, and the useful spread ($\pm 0.5$) is $\sim 5\%$ of the magnitude. Subtracting the baseline collapses the offset to zero and leaves *only* the discriminative signal: response A is above average (push up), B and C below (push down). The variance of the per-sample gradient estimate drops by roughly $(\bar R/\text{spread})^2 \approx 20^2 = 400\times$ here. This is why **no one runs REINFORCE without a baseline**, and why the baseline is the conceptual seed of both the PPO critic and the GRPO group mean.

## Baselines, the value function, and the advantage

### The baseline trick

We can subtract from the reward *any quantity $b(s_t)$ that does not depend on the action $a_t$* without biasing the gradient. The proof is one line: the expected contribution of the baseline term is

$$
\mathbb{E}_{a_t \sim \pi_\theta}\!\big[ b(s_t)\, \nabla_\theta \log \pi_\theta(a_t\mid s_t) \big]
= b(s_t)\, \nabla_\theta \sum_{a_t} \pi_\theta(a_t \mid s_t)
= b(s_t)\, \nabla_\theta\, 1 = 0,
$$

using the same log-derivative identity in reverse ($\sum_a \pi \nabla \log\pi = \sum_a \nabla\pi = \nabla\sum_a\pi = \nabla 1 = 0$). So baselines change variance but not the mean — they are a free lunch. The **variance-minimizing** baseline is (approximately) the expected reward from state $s_t$, which is exactly the *value function*.

{{fig:reinforce-baseline-variance}}

### The value function and the advantage

Define the **state-value function** as the expected return (sum of future rewards) from state $s_t$ under the current policy:

$$
V^\pi(s_t) = \mathbb{E}_{\pi}\!\left[ \sum_{k \ge t} \gamma^{k-t} r_k \;\middle|\; s_t \right].
$$

For RLHF with terminal reward only, $V^\pi(s_t)$ is "the reward model score we *expect* to earn if we continue generating from this partial response with the current policy." Define the **action-value** $Q^\pi(s_t, a_t)$ analogously but committing to action $a_t$ first. The **advantage** is their difference:

$$
A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t).
$$

The advantage answers the precise question we care about: *was taking token $a_t$ in state $s_t$ better or worse than the policy's average behavior there?* Positive advantage → make that token more likely; negative → less likely. Using $V$ as the baseline, the policy gradient becomes the **actor-critic** form:

$$
\nabla_\theta J(\theta) = \mathbb{E}_\pi\!\left[ \sum_t A^\pi(s_t, a_t)\, \nabla_\theta \log \pi_\theta(a_t\mid s_t) \right].
$$

This is the engine of PPO. The **actor** is the policy $\pi_\theta$; the **critic** is a learned estimate $V_\phi \approx V^\pi$. In an LLM, the critic is typically a *separate value head* on top of a transformer (often a copy of the policy or the reward model with a scalar output per token) — a second model nearly as expensive as the policy itself. That expense is exactly what GRPO later deletes.

{{fig:ppo-four-model-passes}}

We will estimate $V_\phi$ by regression (the critic loss) and combine it with the observed return to estimate $A_t$. The cleanest, lowest-variance way to do that combination is **GAE**.

## Generalized Advantage Estimation (GAE)

### The bias-variance dial

We need an estimate $\hat A_t$ of the advantage from sampled data plus the critic $V_\phi$. There is a spectrum of estimators trading bias against variance, indexed by how many real reward steps you use before falling back on the critic's guess.

Define the **TD residual** (temporal-difference error) at step $t$:

$$
\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t).
$$

This is "the reward you actually got plus the critic's value of where you landed, minus the critic's prediction for where you were" — a one-step correction to the value estimate. The two extremes:

- **One-step TD** ($\hat A_t = \delta_t$): low variance (uses only one real reward), but biased whenever $V_\phi$ is wrong.
- **Monte Carlo** ($\hat A_t = \sum_{k\ge t}\gamma^{k-t} r_k - V_\phi(s_t)$): unbiased (uses the *actual* full return), but high variance.

**GAE** (Schulman et al., 2016) interpolates with an exponential weighting controlled by $\lambda \in [0,1]$:

$$
\hat A_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l \ge 0} (\gamma\lambda)^l\, \delta_{t+l}
= \delta_t + \gamma\lambda\,\delta_{t+1} + (\gamma\lambda)^2\,\delta_{t+2} + \cdots
$$

$\lambda = 0$ recovers one-step TD (low variance, high bias); $\lambda = 1$ recovers the Monte Carlo advantage (high variance, unbiased — the $V_\phi$ terms telescope away). In practice $\lambda \approx 0.95$ and $\gamma \approx 1.0$ for LLMs. The advantage and the critic's regression target (the **return**) are computed together:

$$
\hat R_t = \hat A_t + V_\phi(s_t) \quad\text{(the "returns" the critic regresses toward).}
$$

GAE is computed by a single **backward recursion** over the sequence — start at the last token and accumulate:

$$
\hat A_t = \delta_t + \gamma\lambda\, \hat A_{t+1}.
$$

{{fig:gae-bias-variance-dial}}

```python
import torch

def compute_gae(rewards, values, response_mask, gamma=1.0, lam=0.95):
    """
    Generalized Advantage Estimation, computed per token by a backward pass.

    rewards       : (B, T) per-token reward. For RLHF this is mostly zeros,
                    with the reward-model score (and any KL penalty) placed
                    at the last response token of each sequence.
    values        : (B, T) critic estimates V_phi(s_t) for each token.
    response_mask : (B, T) 1.0 on response tokens, 0.0 elsewhere.
    Returns:
      advantages (B, T), returns (B, T) = advantages + values.
    """
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(B, device=rewards.device)
    # V(s_{T}) for the bootstrap of the final step is 0 (episode ends at EOS).
    next_value = torch.zeros(B, device=rewards.device)

    for t in reversed(range(T)):
        mask_t = response_mask[:, t]
        # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_adv = delta + gamma * lam * last_adv
        # zero out advantage/last_adv outside the response so padding can't leak
        advantages[:, t] = last_adv * mask_t
        last_adv = last_adv * mask_t
        next_value = values[:, t] * mask_t  # carry V(s_t) leftward as V(s_{t+1})

    returns = advantages + values
    # Whitening (mean 0, std 1) over response tokens stabilizes the scale.
    flat = advantages[response_mask.bool()]
    advantages = (advantages - flat.mean()) / (flat.std() + 1e-8)
    advantages = advantages * response_mask
    return advantages, returns
```

!!! example "Worked example: hand-checking compute_gae on a 4-token toy"
    Take one sequence (B=1) with four positions and `response_mask = [1, 1, 1, 0]` — position 3 is padding. Use `gamma=1.0` and `lam=0.5` (the code defaults to `lam=0.95`; we pick 0.5 here purely for clean arithmetic — the recursion is byte-identical either way). The reward-model score $R=10$ is placed on the last *unmasked* token (position 2), zeros elsewhere: `rewards = [0, 0, 10, 0]`. Critic values are `values = [1, 2, 3, 5]`, where the 5 at the padding slot is deliberately garbage and must not leak into any real advantage.

    The recursion being applied at each step, walking $t$ from 3 down to 0, is $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ then $\hat A_t = \delta_t + \gamma\lambda\, \hat A_{t+1}$, with `next_value` and `last_adv` reset to 0 whenever the mask is 0.

    | t | mask | $r_t$ | $V_t$ | $V(s_{t+1})$ | $\delta_t$ | $\hat A_t$ (raw) | $R_t$ (return) |
    |---|------|-------|-------|--------------|------------|-------------------|-----------------|
    | 3 | 0 | 0  | 5 | 0 (init) | -5 | 0 (masked out) | 5 (masked, ignore) |
    | 2 | 1 | 10 | 3 | 0        | 7  | 7               | 10 |
    | 1 | 1 | 0  | 2 | 3        | 1  | 4.5             | 6.5 |
    | 0 | 1 | 0  | 1 | 2        | 1  | 3.25            | 4.25 |

    Three subtle points the table makes concrete:

    - **Mask boundary / terminal bootstrap.** At $t=2$ the incoming `next_value` is 0, *not* the padding value 5 — stepping past position 3 set `next_value = values * mask = 0`, so the last real token correctly bootstraps from $V(s_T)=0$. Hence $\hat A_2 = \delta_2 = 7$, with no leakage from the garbage padding value.
    - **Backward recursion.** $\hat A_1 = 1 + 0.5 \times 7 = 4.5$, then $\hat A_0 = 1 + 0.5 \times 4.5 = 3.25$ — each step folds in the discounted future advantage exactly as the formula prescribes.
    - **Returns.** $R_t = \hat A_t + V_t$, computed *before* whitening: $10 = 7+3$, $6.5 = 4.5+2$, $4.25 = 3.25+1$.

    Whitening then normalizes over the three unmasked raw advantages $[3.25, 4.5, 7]$: mean $= 4.9167$, std $= 1.9094$ (`torch.std` defaults to unbiased, $\text{ddof}=1$). The returned, whitened, mask-applied advantages are approximately $[-0.873, -0.218, 1.091, 0.000]$ — the padding slot is forced to exactly 0 by the final `* response_mask`.

    As a unit test: calling `compute_gae(rewards, values, mask, gamma=1.0, lam=0.5)` on these exact tensors should reproduce raw advantages `[3.25, 4.5, 7, 0]`, returns `[4.25, 6.5, 10, 5]`, and whitened advantages `[-0.873, -0.218, 1.091, 0]`. Note the return at the padding slot is a meaningless 5 (= 0 + 5); that's harmless because every downstream consumer multiplies by the mask, but a good test also asserts the nonzero reward landed on the last unmasked token.

!!! warning "Common pitfall: where exactly does the reward go?"
    The reward model produces *one* scalar per response, but GAE wants a per-token `rewards` tensor. The standard convention is to place the RM score (and the per-token KL penalty, see below) on the **last response token** (the token before/at EOS), with zeros elsewhere. A frequent bug is putting the reward on the padding position, on the prompt's last token, or one step off — which silently shifts all advantages by one and quietly wrecks training. Always assert that the nonzero terminal reward lands on the final *unmasked* token of each sequence.

!!! note "Aside: with terminal-only reward, GAE is close to Monte Carlo"
    Because LLM reward is sparse (zero until the end), the TD residuals $\delta_t$ for interior tokens are just $\gamma V(s_{t+1}) - V(s_t)$ — pure critic bootstrapping — and the single real reward only enters at the last step. With $\lambda \to 1$ and $\gamma=1$, GAE collapses to "(terminal reward) $- V_\phi(s_t)$ for every token," i.e. the same scalar advantage (minus the critic's per-token baseline) shared across the response. This is precisely why critic-free methods that assign one group-relative scalar to every token (GRPO/RLOO) lose so little: in the terminal-reward regime the critic's main job is per-token baselining, not genuine multi-step credit assignment.

## The PPO objective

{{fig:ppo-clip}}

### From policy gradient to a surrogate we can reuse

REINFORCE/actor-critic is an **on-policy** estimator: the gradient is only valid for data generated by the *current* $\theta$. But generating rollouts is the expensive part (it's autoregressive decoding on a server; see [The Generation–Training Loop & Rollout Engines](../06-rl-infra/02-generation-training-loop.html)), so we badly want to take *several* gradient steps on each batch of rollouts. The moment we update $\theta$ once, the data is off-policy and the naive gradient is biased.

The principled fix is **importance sampling**. If data was generated by a *behavior* policy $\pi_{\theta_{\text{old}}}$ but we want the gradient for the current $\pi_\theta$, we reweight by the ratio:

$$
r_t(\theta) = \frac{\pi_\theta(o_t \mid s_t)}{\pi_{\theta_{\text{old}}}(o_t \mid s_t)}, \qquad
\mathbb{E}_{\pi_{\theta_{\text{old}}}}\!\big[ r_t(\theta)\, \hat A_t \big] = \mathbb{E}_{\pi_\theta}\!\big[ \hat A_t \big].
$$

The corresponding **surrogate objective** $L^{\text{IS}}(\theta) = \mathbb{E}[ r_t(\theta)\,\hat A_t ]$ has the right gradient at $\theta = \theta_{\text{old}}$. But importance sampling is dangerous: if $\pi_\theta$ moves far from $\pi_{\theta_{\text{old}}}$, the ratio $r_t$ can blow up (a token that was rare under old policy becomes likely under new), the estimator variance explodes, and a single huge ratio times a large advantage can destroy the policy in one step. The whole *point* of PPO is to **bound how far the update can exploit this ratio.**

### The clipped surrogate

PPO's predecessor, TRPO (Trust Region Policy Optimization), enforced a hard KL constraint between old and new policy via a second-order optimization — correct but heavy. **PPO** (Schulman et al., 2017) achieves a similar trust-region effect with a stunningly simple first-order trick: *clip the ratio.* The PPO clipped objective per token is

$$
L^{\text{CLIP}}_t(\theta) = \min\!\Big( r_t(\theta)\,\hat A_t,\;\; \operatorname{clip}\big(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon\big)\,\hat A_t \Big),
$$

and we **maximize** $\mathbb{E}_t[L^{\text{CLIP}}_t]$. Typical $\epsilon = 0.2$. Let us read this carefully, because the `min` and the sign of $\hat A$ interact in a way that trips people up.

- **When $\hat A_t > 0$** (good token, we want $r_t \uparrow$): the unclipped term $r_t \hat A$ grows with $r_t$, but the clipped term caps $r_t$ at $1+\epsilon$. The `min` picks the smaller, so once $r_t > 1+\epsilon$ the objective is flat at $(1+\epsilon)\hat A$ — its gradient is **zero**. Translation: *you get full credit for boosting a good token up to $1+\epsilon$, but no extra reward for boosting it further.* No incentive to take a huge step.
- **When $\hat A_t < 0$** (bad token, we want $r_t \downarrow$): the unclipped term $r_t\hat A$ becomes *more negative* as $r_t$ grows and *less negative* as $r_t$ shrinks. The clipped term floors $r_t$ at $1-\epsilon$. The `min` of two negatives picks the more negative one... which means the clip engages on the *lower* side: once $r_t < 1-\epsilon$, the objective is flat at $(1-\epsilon)\hat A$, zero gradient. Translation: *you get full credit for suppressing a bad token down to $1-\epsilon$, but no extra reward for crushing it further.*

The asymmetry is the key insight: the `min` makes the clip into a **one-sided trust region that only ever removes the incentive to move *too far in the rewarding direction*.** It never clips a step that is *correcting* an overshoot back toward the old policy. The objective is a pessimistic (lower) bound on the unclipped surrogate.


{{fig:ppo-clip-surrogate-curves}}


```python
def ppo_policy_loss(new_logprobs, old_logprobs, advantages, mask, clip_eps=0.2):
    """
    PPO clipped policy (actor) loss for one minibatch.

    new_logprobs : (B, T) log pi_theta(o_t|s_t)  -- requires grad
    old_logprobs : (B, T) log pi_old(o_t|s_t)    -- detached, from rollout time
    advantages   : (B, T) detached GAE advantages (already whitened)
    mask         : (B, T) response mask
    """
    log_ratio = new_logprobs - old_logprobs            # log(π_θ / π_old)
    ratio = log_ratio.exp()                            # the importance ratio r_t

    unclipped = ratio * advantages
    clipped   = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    # maximize min(...)  ==  minimize -min(...)
    per_token_loss = -torch.min(unclipped, clipped)

    loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)

    # Diagnostics every practitioner watches:
    with torch.no_grad():
        clipfrac = ((ratio - 1.0).abs() > clip_eps)[mask.bool()].float().mean()
        approx_kl = ((ratio - 1) - log_ratio)[mask.bool()].mean()  # k3 KL(old||new)
    return loss, clipfrac, approx_kl
```

### The value (critic) loss

The critic $V_\phi$ is trained by regression toward the GAE returns $\hat R_t$. PPO uses a **clipped value loss** too — to stop the critic from moving too far per update, mirroring the policy clip:

$$
L^{V}_t(\phi) = \max\!\Big( \big(V_\phi(s_t) - \hat R_t\big)^2,\;\; \big(\operatorname{clip}(V_\phi(s_t),\, V_{\text{old}}\!-\!\epsilon_v,\, V_{\text{old}}\!+\!\epsilon_v) - \hat R_t\big)^2 \Big).
$$

The `max` makes this a *pessimistic* (larger) loss, so the critic is penalized for both under- and over-shooting beyond the clip window. Many modern implementations find the value-clip optional or even slightly harmful and use plain MSE; we include it because the canonical PPO-RLHF recipe (and TRL) uses it.

```python
def ppo_value_loss(values, old_values, returns, mask, clip_eps_v=0.2):
    """Clipped value-function regression loss (critic)."""
    v_clipped = old_values + torch.clamp(values - old_values, -clip_eps_v, clip_eps_v)
    loss_unclipped = (values   - returns) ** 2
    loss_clipped   = (v_clipped - returns) ** 2
    per_token = 0.5 * torch.max(loss_unclipped, loss_clipped)   # pessimistic
    return (per_token * mask).sum() / mask.sum().clamp(min=1.0)
```

## The KL penalty to the reference policy

### Why we need it

Maximize reward-model score alone and the policy will **over-optimize**: it finds adversarial responses that the (imperfect, finite-data) reward model scores highly but humans hate — repetitive sycophancy, weird formatting, exploitation of RM blind spots. This is **reward hacking** ([Reward Hacking, Over-Optimization & Alignment Failures](../05-posttraining-alignment/13-reward-hacking-failures.html)), and it is the dominant failure mode of unconstrained RLHF. The standard guardrail is a **KL penalty** that keeps the policy close to the frozen reference (the SFT model $\pi_{\text{ref}}$): drift too far and you pay.

We penalize the per-token KL divergence $\mathbb{D}_{\text{KL}}[\pi_\theta(\cdot\mid s_t)\,\|\,\pi_{\text{ref}}(\cdot\mid s_t)]$. The full RLHF objective is therefore the **KL-regularized expected reward**:

$$
\max_\theta\;\; \mathbb{E}_{q\sim\mathcal{D},\, o\sim\pi_\theta}\!\left[ R(q,o) - \beta \sum_{t} \mathbb{D}_{\text{KL}}\big[\pi_\theta(\cdot\mid s_t)\,\|\,\pi_{\text{ref}}(\cdot\mid s_t)\big] \right].
$$

This regularized objective is *exactly* what DPO solves in closed form ([Direct Preference Optimization & Its Variants](../05-posttraining-alignment/07-dpo-and-variants.html)) — DPO and PPO optimize the same target by different means. The coefficient $\beta$ is the single most consequential hyperparameter in PPO-RLHF: too small and you reward-hack; too large and the policy barely moves and you get no improvement.

### Folding KL into the per-token reward

The dominant implementation trick (from InstructGPT) is to *not* add KL as a separate loss term but to **bake it into the per-token reward** before computing GAE. At each response token we set

$$
\tilde r_t = -\beta\,\big(\log \pi_\theta(o_t\mid s_t) - \log \pi_{\text{ref}}(o_t\mid s_t)\big),
$$

a per-token KL penalty, and we add the reward-model score $R(q,o)$ only at the final token:

$$
\text{token reward}_t = \underbrace{\tilde r_t}_{\text{KL, every token}} + \underbrace{R(q,o)\cdot \mathbb{1}[t = T]}_{\text{RM score, last token only}}.
$$

This is elegant: GAE then propagates the KL penalty through the value function and advantage automatically, so the policy is shaped at every token by "stay near the reference" while being pulled at the end toward "score well." Note $\log\pi_\theta - \log\pi_{\text{ref}}$ for the *sampled* token is the **k1** single-sample KL estimator; it is unbiased but can be negative for individual tokens. Some recipes instead use Schulman's always-positive **k3** estimator $\rho - \log\rho - 1$ (with $\rho = \pi_{\text{ref}}/\pi_\theta$) for lower variance; see the discussion in [GRPO, RLOO & Critic-Free RL](../05-posttraining-alignment/08-grpo-rloo.html).

```python
def make_token_rewards(policy_logprobs, ref_logprobs, rm_scores, mask, kl_beta=0.05):
    """
    Build the per-token reward tensor fed to GAE: a per-token KL penalty
    everywhere, plus the scalar reward-model score at the last response token.

    policy_logprobs, ref_logprobs : (B, T) log-probs of the SAMPLED tokens.
    rm_scores : (B,) one scalar per response from the reward model.
    """
    # k1 per-token KL penalty: -beta * (logπ_θ - logπ_ref)
    kl = policy_logprobs - ref_logprobs                  # (B, T)
    token_rewards = -kl_beta * kl * mask                 # KL on every response token

    # Index of the last response token per row, and drop the RM score there.
    last_idx = (mask.sum(dim=1) - 1).long().clamp(min=0) # (B,)
    rows = torch.arange(token_rewards.size(0), device=mask.device)
    token_rewards[rows, last_idx] += rm_scores           # terminal RM reward
    return token_rewards
```

!!! tip "Practitioner tip: adaptive KL control"
    Rather than a fixed $\beta$, InstructGPT used an **adaptive KL controller**: pick a target KL (e.g. the running KL should sit near some value), and after each step multiply $\beta$ up if the measured KL exceeded the target, down if below. This keeps the policy at a roughly constant "distance budget" from the reference even as the reward landscape changes through training. It is a cheap PID-like loop and it prevents both the "ran away from the SFT model" and the "never moved" failure modes. Watch the realized KL as a first-class metric, not just the reward.

## The full PPO-for-LLM loop

We now assemble everything. The total per-minibatch loss combines the three pieces — clipped policy loss, value loss, and an **entropy bonus** $S[\pi_\theta]$ that discourages premature collapse to a deterministic policy (it keeps exploration alive):

$$
L^{\text{PPO}}(\theta,\phi) = \underbrace{-\,\mathbb{E}_t\big[L^{\text{CLIP}}_t\big]}_{\text{policy}} \;+\; c_v\,\underbrace{\mathbb{E}_t\big[L^{V}_t\big]}_{\text{value}} \;-\; c_e\,\underbrace{\mathbb{E}_t\big[S[\pi_\theta](s_t)\big]}_{\text{entropy bonus}}.
$$

(The KL-to-reference is *already inside* $\hat A_t$ via the per-token reward; the entropy term here is a different, intra-policy regularizer.) Typical coefficients: $c_v \approx 0.1$–$1.0$, $c_e \approx 0$–$0.01$.

The training loop has a characteristic **two-phase rhythm** — a *rollout phase* (generate, score, compute advantages) and an *optimization phase* (several epochs of minibatch gradient steps on the frozen rollouts), which is the structure every RL-infra system in Part VI is built around:

{{fig:ppo-iteration-two-phase-loop}}

Here is a compact but complete PPO step that ties the helper functions together. It is written for clarity over speed; a production system (TRL's `PPOTrainer`, OpenRLHF, veRL) disaggregates generation onto an inference engine and shards the four models, but the math is identical.

```python
import torch
import torch.nn.functional as F

# Assume: policy (with value head), ref_model, reward_model, tokenizer, optimizer.
# policy(input_ids) returns logits AND a per-token scalar value (value head).

PPO_EPOCHS   = 4
MINIBATCHES  = 4
CLIP_EPS     = 0.2
KL_BETA      = 0.05
VF_COEF      = 0.5
ENT_COEF     = 0.0
TARGET_KL    = 0.02   # early-stop the epoch loop if approx_kl exceeds this

def token_logprobs_and_values(model, input_ids):
    """Per-token log-prob of the realized next token, plus value-head output."""
    out = model(input_ids)                                   # logits (B,T,V), values (B,T)
    logits, values = out.logits[:, :-1], out.values[:, :-1]
    logp = F.log_softmax(logits.float(), dim=-1)
    targets = input_ids[:, 1:].unsqueeze(-1)
    token_lp = logp.gather(-1, targets).squeeze(-1)          # (B, T-1)
    return token_lp, values

@torch.no_grad()
def ppo_rollout(prompts):
    # 1-2: generate responses with the current (behavior) policy.
    input_ids, resp_mask = generate_batch(policy, prompts)   # your decode fn
    # 3: reward-model scores (one scalar per sequence).
    rm_scores = reward_model.score(input_ids, resp_mask)     # (B,)
    # 4: cache behavior log-probs, reference log-probs, and values.
    old_lp,  old_values = token_logprobs_and_values(policy,    input_ids)
    ref_lp,  _          = token_logprobs_and_values(ref_model, input_ids)
    mask = resp_mask[:, 1:]                                   # align with shifted targets
    # 5: per-token rewards = KL penalty + terminal RM score.
    token_rewards = make_token_rewards(old_lp, ref_lp, rm_scores, mask, KL_BETA)
    # 6: GAE advantages and returns.
    advantages, returns = compute_gae(token_rewards, old_values, mask)
    return dict(input_ids=input_ids, mask=mask, old_lp=old_lp,
                old_values=old_values, advantages=advantages, returns=returns,
                mean_reward=rm_scores.mean().item())

def ppo_update(buf):
    B = buf["input_ids"].size(0)
    idx = torch.randperm(B)
    mb_size = B // MINIBATCHES
    for _ in range(PPO_EPOCHS):
        for m in range(MINIBATCHES):
            sl = idx[m*mb_size:(m+1)*mb_size]
            new_lp, new_values = token_logprobs_and_values(policy, buf["input_ids"][sl])

            p_loss, clipfrac, approx_kl = ppo_policy_loss(
                new_lp, buf["old_lp"][sl], buf["advantages"][sl], buf["mask"][sl], CLIP_EPS)
            v_loss = ppo_value_loss(
                new_values, buf["old_values"][sl], buf["returns"][sl], buf["mask"][sl])
            entropy = policy_entropy(policy, buf["input_ids"][sl], buf["mask"][sl])

            loss = p_loss + VF_COEF * v_loss - ENT_COEF * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
        if approx_kl > 1.5 * TARGET_KL:        # KL early-stopping safeguard
            break

# Driver:
# for it in range(NUM_ITERS):
#     buf = ppo_rollout(sample_prompts(batch_size))
#     ppo_update(buf)
```

!!! example "Worked example: tracing one token through the PPO step"
    A prompt asks for a polite refusal. We sample a response; the reward model gives it $R = +1.5$ (good refusal). Take one interior token $o_t = $ "however". Suppose at rollout time the behavior policy had $\pi_{\text{old}}(o_t\mid s_t) = 0.10$ (so $\log\pi_{\text{old}} = -2.303$), the reference (SFT) model had $\pi_{\text{ref}} = 0.08$ ($\log\pi_{\text{ref}} = -2.526$), and after some optimization the current policy has $\pi_\theta(o_t\mid s_t) = 0.14$ ($\log\pi_\theta = -1.966$).

    **Per-token KL penalty** (k1), with $\beta = 0.05$:
    $$\tilde r_t = -\beta(\log\pi_\theta - \log\pi_{\text{ref}}) = -0.05\,(-1.966 - (-2.526)) = -0.05 \times 0.560 = -0.028.$$
    The token is slightly *more* likely than the reference, so it incurs a small negative reward — the KL leash. (At rollout time we'd actually use $\pi_{\text{old}}$ for the reward; we use $\pi_\theta$ here to show the live penalty.)

    **Suppose GAE returned** $\hat A_t = +0.8$ for this token (it sits on a trajectory that ended well, and the critic predicted slightly lower). The token is "good": we want to push it up.

    **Importance ratio:** $r_t = \pi_\theta/\pi_{\text{old}} = 0.14/0.10 = 1.40$. With $\epsilon = 0.2$, the clip ceiling is $1.20$.
    - unclipped: $r_t \hat A_t = 1.40 \times 0.8 = 1.12$
    - clipped: $1.20 \times 0.8 = 0.96$
    - $\min(1.12, 0.96) = 0.96$ → **the clip engages**; gradient w.r.t. this token is zeroed.

    Interpretation: the optimizer already moved this token's probability up by $40\%$ since rollout — past the $20\%$ trust region. PPO refuses to reward going further this epoch. The token will get another chance after the *next* rollout, when $\pi_{\text{old}}$ is reset to the current policy and the ratio starts back at $1.0$. This is the trust region in action: bounded, incremental, safe steps. The `clipfrac` diagnostic counts what fraction of tokens hit this clip; a healthy run sits around $0.1$–$0.3$. A `clipfrac` near $0$ means your learning rate or advantages are tiny (no movement); near $1$ means you're taking wild steps and should lower the LR or $\epsilon$.

## Why PPO is finicky

PPO works, and InstructGPT proved it scales. But practitioners universally describe it as *brittle and operationally heavy*, and understanding **why** is exactly what motivates the rest of Part V and Part VI. The difficulties stack:

**1. Four models in memory at once.** A PPO step needs the **policy** (trains), the **critic/value net** (trains, often as large as the policy), the **reward model** (frozen, large), and the **reference model** (frozen). On a 7B policy that is roughly $4\times$ the parameter footprint plus the policy's and critic's optimizer states (Adam keeps two moments per trainable parameter; see [Optimizers](../03-pretraining/09-optimizers.html)). Fitting this requires sharding (FSDP/ZeRO; see [Distributed Training I](../03-pretraining/05-distributed-data-parallel.html)) and careful memory choreography — and you still have to *generate* with the policy, which competes for the same GPUs ([Colocated vs Disaggregated RL & Weight Synchronization](../06-rl-infra/07-colocated-vs-disaggregated.html)).

**2. The critic is hard to train and a major instability source.** A value head regressing sparse terminal rewards through a 7B transformer is itself a finicky learning problem. If $V_\phi$ is biased, your advantages are biased, the policy chases a wrong gradient, and the whole thing diverges quietly — reward looks fine, then collapses. Many practitioners spend more time debugging the critic than the policy. This single pain point is the entire reason **GRPO and RLOO delete the critic** ([GRPO, RLOO & Critic-Free RL](../05-posttraining-alignment/08-grpo-rloo.html)).

**3. A thicket of coupled hyperparameters.** $\beta$ (KL), $\epsilon$ (clip), $\lambda$ and $\gamma$ (GAE), $c_v$ and $c_e$, learning rates for policy *and* critic, number of PPO epochs, minibatch size, the reward-normalization scheme, the target KL for adaptive control. These interact non-linearly: a too-large LR with too-small $\beta$ reward-hacks; a too-large $\beta$ stalls; too many PPO epochs makes the data badly off-policy and the clip can't save you. The viable region is narrow and problem-dependent, and there is no clean loss curve telling you you're in it — you must watch *reward, KL, clipfrac, value loss, and entropy together*.

**4. Reward over-optimization is always lurking.** Even with the KL leash, push long enough and the policy finds RM blind spots; reward climbs while true quality falls (the "Goodharting" curve, often visible as reward up / human-eval down). The KL coefficient is a blunt instrument against a sharp problem. See [Reward Hacking, Over-Optimization & Alignment Failures](../05-posttraining-alignment/13-reward-hacking-failures.html).

**5. Generation–training skew.** In production you generate on a fast inference engine (vLLM/SGLang) and train on a different stack; numerical differences (kernels, precision, KV-cache layout) mean the log-probs you *think* the behavior policy had can differ slightly from what the trainer recomputes — biasing the importance ratio. Keeping the sampler and trainer log-probs consistent is a real engineering tax ([The Generation–Training Loop & Rollout Engines](../06-rl-infra/02-generation-training-loop.html)).

**6. Sample inefficiency and cost.** Each iteration regenerates fresh rollouts (you can only reuse them for a few epochs before the clip stops being valid). Generation dominates wall-clock. This is why DPO — which needs *no* online generation — is so attractive when you already have preference pairs, and why so much RL-infra work targets rollout throughput ([Scaling RL: Throughput, Load Balancing & The Latest Tricks](../06-rl-infra/11-scaling-rl-tricks.html)).

The honest summary: PPO is the *right general algorithm* and the most *educational* one — every term has a clean theoretical justification — but it asks a lot of the engineer. The field's trajectory since 2023 has been a steady search for things that keep PPO's good behavior (trust-region stability, online exploration) while removing its pain (the critic, the hyperparameter thicket, the offline-impossibility). DPO removed the RL loop entirely; GRPO/RLOO removed the critic; RLVR removed the reward model in favor of verifiers. You cannot understand *why* any of those exist without first understanding the PPO they are simplifying — which is this chapter.

!!! interview "Interview Corner"
    **Q:** Walk me through the PPO clipped objective. Why the `min`, and what specifically does clipping prevent? Why do we even need importance sampling here?

    **A:** We need importance sampling because we generate rollouts once (expensive autoregressive decoding) but want to take several gradient steps on them. After the first update the data is off-policy, so we reweight each token by the ratio $r_t = \pi_\theta/\pi_{\text{old}}$; the surrogate $\mathbb{E}[r_t \hat A_t]$ then has the correct gradient at $\theta = \theta_{\text{old}}$. The danger is that a large ratio times a large advantage can take a catastrophic step and blow up the policy. PPO bounds this by clipping: the objective is $\min(r_t\hat A_t,\ \operatorname{clip}(r_t, 1{-}\epsilon, 1{+}\epsilon)\hat A_t)$. The `min` makes it a **pessimistic lower bound** that creates a one-sided trust region. For a *good* token ($\hat A>0$) it stops rewarding you once $r_t > 1+\epsilon$ — no incentive to over-boost; for a *bad* token ($\hat A<0$) it stops rewarding you once $r_t < 1-\epsilon$ — no incentive to over-suppress. Crucially, because of the `min`, clipping only ever *removes* incentive to move further in the rewarding direction; it never blocks a step that corrects an overshoot back toward $\pi_{\text{old}}$. The net effect is small, stable, incremental policy updates without TRPO's expensive second-order KL constraint. I'd also mention the `clipfrac` diagnostic — fraction of tokens being clipped, healthy around 0.1–0.3 — and that the *separate* KL-to-reference penalty (a different mechanism from the clip) is what prevents reward hacking, while the clip just prevents per-step instability.

!!! interview "Interview Corner"
    **Q:** In PPO-RLHF there are two different "KL"s and two different "clips." Distinguish them.

    **A:** **Two KLs:** (1) the **KL-to-reference penalty** $\beta\,\mathbb{D}_{\text{KL}}[\pi_\theta\|\pi_{\text{ref}}]$, between the *trained policy and the frozen SFT model*, which keeps the policy from drifting into reward-hacked nonsense and defines the regularized objective (the same one DPO solves in closed form) — usually folded into the per-token reward before GAE. (2) the **approximate KL between $\pi_{\text{old}}$ and $\pi_\theta$** *within* an optimization phase, which we only *monitor* (and use for epoch early-stopping) to detect when the data has gone too off-policy. **Two clips:** (1) the **policy ratio clip** at $1\pm\epsilon$ that defines the trust region; (2) the **value-function clip** $V_{\text{old}}\pm\epsilon_v$ that limits how far the critic moves per update (a pessimistic `max` on the squared error). They serve different roles: the policy clip is for trust-region stability, the value clip is critic-stability hygiene (and is the more optional of the two). Confusing the reference-KL (alignment leash) with the old-vs-new KL (off-policy monitor) is the classic mistake.

!!! key "Key Takeaways"
    - **The LM is the policy.** Text generation is an MDP where states are partial sequences, actions are tokens, the transition is deterministic, and the reward is *terminal and sparse* — a single reward-model scalar at EOS.
    - **REINFORCE / the score-function estimator** turns "gradient of an expected reward" into "expected reward times $\nabla\log\pi$" using the log-derivative trick — no differentiable reward needed. Unbiased but catastrophically high-variance on its own.
    - **A baseline** $b(s_t)$ that doesn't depend on the action reduces variance without bias; the best one is the **value function** $V^\pi$, giving the **advantage** $A = Q - V$ ("was this token better than average?"). The learned critic $V_\phi$ is the second model PPO carries.
    - **GAE** with parameters $(\gamma, \lambda)$ dials bias vs. variance between one-step TD ($\lambda{=}0$) and Monte Carlo ($\lambda{=}1$), computed by one backward recursion of TD residuals $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$.
    - **PPO's clipped surrogate** $\min(r_t\hat A_t,\ \operatorname{clip}(r_t,1{-}\epsilon,1{+}\epsilon)\hat A_t)$ is a first-order trust region: it lets you reuse rollouts for several epochs via importance sampling while preventing any single step from exploiting the ratio too far. `clipfrac` $\approx 0.1$–$0.3$ is healthy.
    - **The KL-to-reference penalty** $\beta\,\mathbb{D}_{\text{KL}}[\pi_\theta\|\pi_{\text{ref}}]$ — usually folded into the per-token reward — is the alignment leash against reward hacking. $\beta$ is the most consequential knob; adaptive KL control targets a fixed "distance budget."
    - **The full loop is two-phase:** a rollout phase (generate → score with RM → forward ref/value → KL-reward → GAE) and an optimization phase (several clipped-objective epochs on the frozen buffer), with policy + value + entropy terms.
    - **PPO is finicky** because it juggles four models, a hard-to-train critic, a thicket of coupled hyperparameters, ever-present reward over-optimization, generation–training skew, and high generation cost — which is precisely the motivation for DPO (no RL loop), GRPO/RLOO (no critic), and RLVR (no reward model).

!!! sota "State of the Art & Resources (2026)"
    PPO remains the canonical, most general RL algorithm for RLHF, but since 2023 the field has largely moved toward critic-free variants (GRPO, RLOO, REINFORCE++) that preserve PPO's trust-region stability while eliminating the expensive value network; PPO's four-model loop is now mainly seen in large-scale or multi-turn agent training where the full generality is warranted.

    **Foundational work**

    - [Schulman et al., *Proximal Policy Optimization Algorithms* (2017)](https://arxiv.org/abs/1707.06347) — introduces the clipped surrogate objective and the two-phase rollout/optimize loop that every LLM RL recipe inherits.
    - [Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation* (2016)](https://arxiv.org/abs/1506.02438) — defines GAE and the $(\gamma,\lambda)$ bias-variance dial used verbatim in PPO-RLHF.
    - [Schulman et al., *Trust Region Policy Optimization* (2015)](https://arxiv.org/abs/1502.05477) — the second-order predecessor PPO's clipping approximates; essential for understanding *why* the clip is sufficient.
    - [Ouyang et al., *Training Language Models to Follow Instructions with Human Feedback* (InstructGPT, 2022)](https://arxiv.org/abs/2203.02155) — the canonical PPO-for-RLHF recipe: per-token KL reward, adaptive KL controller, and empirical evidence that PPO scales to 175B.

    **Recent advances (2023–2026)**

    - [Zheng et al., *Secrets of RLHF in Large Language Models Part I: PPO* (2023)](https://arxiv.org/abs/2307.04964) — systematic ablation of every PPO-RLHF component; identifies policy-constraint tuning as the key stability factor and releases reproducible code.
    - [Hu et al., *REINFORCE++: Stabilizing Critic-Free Policy Optimization with Global Advantage Normalization* (2025)](https://arxiv.org/abs/2501.03262) — drops the critic entirely, adds global advantage normalization from PPO; matches PPO quality at lower compute cost.

    **Open-source & tools**

    - [huggingface/trl](https://github.com/huggingface/trl) — HuggingFace's full-stack post-training library with PPOTrainer, GRPOTrainer, DPOTrainer, and SFTTrainer; the most widely used entry point for RLHF experiments.
    - [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — Ray+vLLM-based production RLHF framework supporting PPO, GRPO, REINFORCE++, and async rollouts; designed for 7B–70B scale.
    - [verl-project/verl](https://github.com/verl-project/verl) — ByteDance's HybridFlow open-source implementation; flexible hybrid-controller design supporting FSDP, Megatron-LM, vLLM, and SGLang backends.

    **Go deeper**

    - [Lilian Weng, *Policy Gradient Algorithms* (Lil'Log, 2018, updated)](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/) — comprehensive reference covering REINFORCE, actor-critic, TRPO, PPO, and SAC with clean derivations; the go-to written companion to this chapter.

## Further reading

- Sutton & Barto, **Reinforcement Learning: An Introduction** (2nd ed.) — the canonical text for MDPs, policy gradients, value functions, and the actor-critic framework.
- Williams, **Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning** (1992) — the original REINFORCE / score-function estimator.
- Schulman, Levine, Moritz, Jordan, Abbeel, **Trust Region Policy Optimization** (2015) — the trust-region predecessor PPO approximates.
- Schulman, Moritz, Levine, Jordan, Abbeel, **High-Dimensional Continuous Control Using Generalized Advantage Estimation** (2016) — GAE.
- Schulman, Wolski, Dhariwal, Radford, Klimov, **Proximal Policy Optimization Algorithms** (2017) — the clipped objective itself.
- Ouyang, Wu, Jiang, et al. (OpenAI), **Training Language Models to Follow Instructions with Human Feedback** (InstructGPT, 2022) — the canonical PPO-for-RLHF recipe, per-token KL reward, and adaptive KL control.
- Ziegler, Stiennon, Wu, et al., **Fine-Tuning Language Models from Human Preferences** (2019) — the earlier blueprint for KL-regularized RLHF.
- Huang et al., **The N Implementation Details of RLHF with PPO** / the **CleanRL** and **TRL** PPO implementations — concrete, debuggable reference code; see [TRL: HuggingFace's RL Library](../06-rl-infra/03-trl.html).
- John Schulman, **Approximating KL Divergence** (blog note) — the k1/k2/k3 KL estimators used for the penalty.
