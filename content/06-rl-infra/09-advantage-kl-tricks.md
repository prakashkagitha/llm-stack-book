# 6.9 Advantage Estimation, KL Control & Stability Tricks

Reinforcement learning for language models lives or dies on a handful of scalar quantities computed per token: the **advantage** $A_t$ that tells the optimizer which tokens to push up and which to push down, the **KL divergence** that keeps the policy from drifting into gibberish, and the **clipping** and **normalization** machinery that prevents a single noisy update from blowing up the run. Get these right and a 7B model learns to reason on GSM8K in a few hundred steps. Get them subtly wrong — a sign flip in the advantage, a KL estimator with the wrong variance, a loss averaged over the wrong axis — and the run either flatlines or collapses into repeated tokens and `nan` gradients.

This chapter is the engineering reference for that machinery. We assume you already know the algorithms at the level of [Policy Gradients & PPO for Language Models](../05-posttraining-alignment/06-ppo-for-llms.html) and [GRPO, RLOO & Critic-Free RL](../05-posttraining-alignment/08-grpo-rloo.html); here we go one level deeper into the *implementation details that the papers gloss over and the codebases argue about*. Every formula gets plugged with real numbers, and every trick comes with the code that implements it. By the end you should be able to read a `verl` or `trl` advantage-computation function line by line and know exactly what each `clamp`, `detach`, and `masked_mean` is doing and why.

## The Credit-Assignment Problem and the Advantage

The policy-gradient theorem says the gradient of expected return $J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]$ is

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\!\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t\mid s_t)\, \Psi_t \right]
$$

where $\Psi_t$ is *some* measure of how good action $a_t$ was. The whole art of variance reduction is choosing $\Psi_t$. The naive choice $\Psi_t = R(\tau)$ (the full trajectory return) is unbiased but catastrophically high-variance: it credits every token in a 500-token response with the *entire* reward, including the reward earned by tokens that came after it. The causality trick replaces it with the **reward-to-go** $\sum_{t'\ge t}r_{t'}$, and subtracting a state-dependent **baseline** $b(s_t)$ leaves the gradient unbiased while shrinking variance. The optimal baseline is the value function $V(s_t)$, and the resulting quantity is the **advantage**:

$$
A_t = Q(s_t,a_t) - V(s_t).
$$

For LLMs the "state" $s_t$ is the prompt plus all tokens generated so far, the "action" $a_t$ is the next token, and rewards are almost always **sparse and terminal**: a verifier or reward model emits a single scalar $r$ at the end of sequence (EOS), and all intermediate rewards are zero. This sparsity is the defining feature of RL-for-LLMs and it shapes every estimator below. There is no dense per-token signal to bootstrap from; the entire learning signal is one number per rollout, possibly with a small per-token KL penalty bolted on.


{{fig:advkl-terminal-reward-rollout}}


The three families of advantage estimators you will meet in production differ entirely in *how they turn that one terminal number into a per-token $A_t$*:

1. **Monte-Carlo / REINFORCE-style**: $A_t = R - b$ for a scalar baseline $b$. No value network, no per-token resolution.
2. **GAE (Generalized Advantage Estimation)**: uses a learned critic $V_\phi(s_t)$ to produce a smooth per-token advantage trading bias against variance via $\lambda$.
3. **Group-relative (GRPO / RLOO)**: replaces the critic with the empirical mean reward over a *group* of samples for the same prompt.

We take them in turn.

## Monte-Carlo, GAE, and Group-Relative Advantages

{{fig:advkl-three-advantage-families}}

### Monte-Carlo / REINFORCE with a baseline

The simplest estimator assigns the same advantage to every token in a sequence:

$$
A_t = R(\tau) - b
$$

The baseline $b$ can be a running mean of returns, a batch mean, or — in **RLOO (REINFORCE Leave-One-Out)** — the mean reward of the *other* $K-1$ samples drawn for the same prompt:

$$
A_i = R_i - \frac{1}{K-1}\sum_{j\ne i} R_j.
$$

Leave-one-out makes the baseline an unbiased estimate of $V(s_0)$ that does not depend on sample $i$ itself, which is exactly the property an optimal baseline needs to keep the gradient unbiased. The cost is $K$ generations per prompt. The benefit is zero critic, zero critic-training instability, and a baseline that is provably variance-reducing. RLOO and its cousins are covered in depth in [GRPO, RLOO & Critic-Free RL](../05-posttraining-alignment/08-grpo-rloo.html).

### Generalized Advantage Estimation (GAE)

When you *do* have a critic $V_\phi$, GAE (Schulman et al., 2015) gives the canonical bias–variance knob. Define the one-step **TD residual**

$$
\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t),
$$

then the GAE advantage is the exponentially weighted sum of future residuals:

$$
A_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{T-1-t} (\gamma\lambda)^l\, \delta_{t+l}.
$$

$\lambda=0$ collapses to the high-bias, low-variance one-step TD advantage $\delta_t$; $\lambda=1$ recovers the unbiased, high-variance Monte-Carlo advantage $\sum_{l}\gamma^l r_{t+l} - V(s_t)$. The value target for training the critic is $V_t^{\text{target}} = A_t + V_\phi(s_t)$ (the "returns"). GAE is computed by a backward recurrence — this is the single most-implemented loop in RL infrastructure:

```python
import torch

def compute_gae(rewards, values, gamma=1.0, lam=0.95, mask=None):
    """Generalized Advantage Estimation, computed token-by-token, backward.

    Args:
        rewards: (B, T) per-token rewards. For LLMs this is usually all zeros
                 except the last valid token, which carries r - beta*KL.
        values:  (B, T) critic estimates V(s_t) for each token position.
        gamma:   discount. For LLM RL almost always 1.0 (episodes are short
                 and we do not want to discount a correct final answer).
        lam:     GAE lambda. 0.95-1.0 typical.
        mask:    (B, T) 1.0 for real tokens, 0.0 for padding.
    Returns:
        advantages (B, T), returns (B, T)
    """
    B, T = rewards.shape
    if mask is None:
        mask = torch.ones_like(rewards)
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(B, device=rewards.device)
    # Append a bootstrap value of 0 past the end (terminal state).
    for t in reversed(range(T)):
        next_value = values[:, t + 1] if t + 1 < T else torch.zeros(B, device=rewards.device)
        # TD residual delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        delta = rewards[:, t] + gamma * next_value * mask[:, t] - values[:, t]
        # Recurrence: A_t = delta_t + gamma*lam*A_{t+1}
        last_gae = delta + gamma * lam * last_gae * mask[:, t]
        advantages[:, t] = last_gae
    returns = advantages + values          # value-function targets
    return advantages * mask, returns * mask
```

Two LLM-specific subtleties bite here. First, **`gamma` is almost always 1.0** for reasoning RL: a 300-token chain of thought is one logical episode, and discounting would arbitrarily penalize the final answer token relative to early scratch-work. Second, because rewards are terminal-only, with $\gamma=1, \lambda=1$ every token in a sequence gets advantage $R - V(s_t)$ — the critic's *only* job is to learn a position-dependent baseline. Many practitioners ask, reasonably, whether a per-token critic is worth its cost when the reward is a single terminal number; the rise of GRPO is largely an answer of "no, not for verifiable-reward reasoning tasks."

### Group-relative advantages (GRPO)

GRPO (Shao et al., *DeepSeekMath*, 2024) throws away the critic entirely. For each prompt $q$ it samples a **group** of $G$ completions, scores each with the reward function, and standardizes within the group:

$$
A_i = \frac{R_i - \operatorname{mean}(\{R_1,\dots,R_G\})}{\operatorname{std}(\{R_1,\dots,R_G\}) + \varepsilon}.
$$

Every token in completion $i$ receives this same scalar $A_i$ (in the original "outcome-supervised" form). The group mean is the baseline; the group std is a built-in per-prompt normalization. This is elegant and cheap, but it has well-documented pathologies that later work fixed:

- **The std-normalization bias.** Dividing by the group std up-weights prompts that happen to be *easy or hard enough* to produce low-reward-variance groups. A prompt where 7/8 samples are correct has tiny std, so its few informative gradients get amplified; a prompt with 50/50 split has large std and gets damped. **Dr. GRPO** (Liu et al., 2025) argues this is a bias and recommends dropping the std divisor (subtract the mean only). Many codebases now expose a `scale_rewards` / `norm_adv` flag for exactly this.
- **The all-same-reward group.** If all $G$ samples get identical reward (all correct or all wrong), the advantage is $0/\varepsilon=0$ and the group contributes *no gradient*. This is fine in principle but wastes compute; **dynamic sampling** (DAPO, Yu et al., 2025) over-samples and filters out zero-variance groups so every batch is full of informative gradients.
- **Length bias.** Because the same scalar advantage multiplies the log-prob of every token, longer correct responses accumulate larger total gradient — encouraging length inflation. The loss-aggregation choice (token-mean vs sequence-mean, below) interacts directly with this.

```python
def grpo_advantages(rewards, group_size, eps=1e-4, scale_by_std=True):
    """Group-relative advantages. rewards: (B,) with B = num_prompts*group_size,
    laid out as [p0_s0, p0_s1, ..., p0_s{G-1}, p1_s0, ...]."""
    rewards = rewards.view(-1, group_size)            # (num_prompts, G)
    mean = rewards.mean(dim=1, keepdim=True)
    adv = rewards - mean                               # subtract group baseline
    if scale_by_std:                                  # standard GRPO
        std = rewards.std(dim=1, keepdim=True)
        adv = adv / (std + eps)
    # else: Dr. GRPO — mean-only, avoids the std up-weighting bias
    return adv.view(-1)                               # broadcast to tokens later
```

The table below is the cheat-sheet you want in front of you when picking an estimator.

| Estimator | Critic? | Per-token resolution | Variance | Bias | Best for |
|---|---|---|---|---|---|
| MC / REINFORCE | No | No (constant per seq) | High | Unbiased | Quick baselines, debugging |
| RLOO | No | No | Medium | Unbiased | Preference/RM rewards, small $G$ |
| GAE | Yes | Yes | Tunable via $\lambda$ | Tunable | Dense rewards, classic PPO-RLHF |
| GRPO | No | No (outcome) | Medium | Std-norm bias | Verifiable-reward reasoning |

## Advantage Normalization and Whitening

Whatever estimator you use, the raw advantages are on an arbitrary scale that drifts as the reward distribution shifts during training. Feeding un-normalized advantages straight into the loss makes the effective learning rate a moving target. The standard remedy is **whitening**: shift to zero mean and scale to unit variance across the batch.

$$
\hat{A}_t = \frac{A_t - \mu_A}{\sigma_A + \varepsilon},\qquad \mu_A = \frac{\sum_t m_t A_t}{\sum_t m_t},\quad \sigma_A^2 = \frac{\sum_t m_t (A_t-\mu_A)^2}{\sum_t m_t}.
$$

The mask $m_t$ matters enormously: you must compute the mean and variance over **real tokens only**, never over padding, or the statistics are silently corrupted by however much padding happens to be in the batch. This is the single most common advantage-normalization bug.

```python
def masked_whiten(advantages, mask, shift_mean=True, eps=1e-8):
    """Zero-mean, unit-variance advantages over the masked (non-pad) tokens.

    shift_mean=False keeps the mean (some recipes only rescale variance, because
    subtracting a batch mean is itself a baseline change that can be undesirable
    when you already subtracted a per-prompt group baseline)."""
    n = mask.sum()
    mean = (advantages * mask).sum() / n
    var = ((advantages - mean) ** 2 * mask).sum() / n
    whitened = (advantages - mean) * torch.rsqrt(var + eps)
    if not shift_mean:
        whitened = whitened + mean
    return whitened * mask
```

There are three live debates here that you will encounter in code review:

**Batch-level vs group-level normalization.** GRPO *already* normalizes within each prompt's group. Applying a *second*, batch-level whitening on top changes the relative weighting between prompts. Some recipes do both, some do only group-level, some (Dr. GRPO) do mean-subtraction only. There is no universally correct answer — it depends on whether you want every prompt to contribute equally (batch-norm) or each group's internal signal preserved (group-only).

**Whitening couples examples within a batch.** After whitening, the gradient for example $i$ depends on the rewards of *all* other examples in the batch (through $\mu_A$ and $\sigma_A$). This is usually benign but means your effective objective changes with batch size and with the distributed all-reduce that computes global statistics. If you whiten **per-GPU** instead of globally, small per-rank batches give noisy statistics; if you whiten **globally**, you pay an all-reduce. Most serious frameworks whiten globally across data-parallel ranks.

**Don't whiten when the baseline is already exact.** RLOO's leave-one-out baseline is already an unbiased per-prompt baseline. Subtracting an additional batch mean is redundant and can *increase* variance. A good rule: normalize *scale* (divide by std) freely, but be deliberate about *shifting* the mean more than once.

!!! warning "Common pitfall"

    Normalizing advantages over the padded tensor instead of the masked one. If your batch is 40% padding and you call `advantages.mean()` / `advantages.std()` on the full `(B, T)` tensor, the zeros from padding pull the mean toward zero and shrink the std, silently scaling your real advantages by an arbitrary, batch-composition-dependent factor. Always reduce with the loss mask. The same bug lurks in entropy and KL reductions.

## KL Control: Estimators, Placement, and Adaptation

KL divergence $D_{\mathrm{KL}}(\pi_\theta \| \pi_{\text{ref}})$ between the current policy and a frozen reference (usually the SFT model) is the leash that keeps RL from destroying the model's general capabilities and from reward-hacking into degenerate text. There are three independent engineering decisions: **how to estimate it**, **where to apply it** (in the reward or in the loss), and **how to set its coefficient** (fixed or adaptive). The relationship between KL control and the broader failure mode of over-optimization is the subject of [Reward Hacking, Over-Optimization & Alignment Failures](../05-posttraining-alignment/13-reward-hacking-failures.html).

### The k1 / k2 / k3 estimators

We almost never compute the exact KL. Instead we have, per token, the log-probabilities under the policy and the reference, and we form a Monte-Carlo estimate from the single sampled token. Let $r = \log\frac{\pi_{\text{ref}}(a_t)}{\pi_\theta(a_t)}$ (note the direction: ref over policy). John Schulman's well-known note on approximating KL gives three estimators:

$$
k_1 = -r = \log\frac{\pi_\theta}{\pi_{\text{ref}}},\qquad
k_2 = \frac{1}{2}r^2,\qquad
k_3 = (e^{r} - 1) - r = \frac{\pi_{\text{ref}}}{\pi_\theta} - 1 - \log\frac{\pi_{\text{ref}}}{\pi_\theta}.
$$

- **k1** is the naive, unbiased estimator. $\mathbb{E}[k_1]=D_{\mathrm{KL}}$ exactly, but it has high variance and — crucially — can be **negative** for an individual token (when the policy assigns lower probability than the reference to the sampled token). A per-token KL penalty that is sometimes negative is a strange reward signal.
- **k2** $=\frac12 r^2$ is always non-negative, lower-variance, and biased. It is a good estimator of KL when the two distributions are close (it is the second-order Taylor term), which is exactly the regime KL control operates in.
- **k3** is the modern default (used by GRPO/TRL). It is **unbiased *and* always non-negative**, because $e^r - 1 - r \ge 0$ for all $r$. It achieves this by being a control-variate-corrected version of $k_1$. This is the estimator to use unless you have a specific reason not to.

```python
import torch

def kl_estimators(logp_policy, logp_ref):
    """Per-token KL(pi_theta || pi_ref) estimators from sampled-token logprobs.
    logp_policy, logp_ref: (B, T) log pi_theta(a_t) and log pi_ref(a_t)."""
    logr = logp_ref - logp_policy                 # r = log(pi_ref / pi_theta)
    k1 = -logr                                    # unbiased, can be negative
    k2 = 0.5 * logr.pow(2)                         # biased, always >= 0
    k3 = torch.expm1(logr) - logr                 # = exp(logr) - 1 - logr; unbiased, >= 0
    return k1, k2, k3
```

{{fig:advkl-kl-estimators-k1k2k3}}

!!! note "Aside"

    Why is $k_3 = e^r - 1 - r$ unbiased? Because $\mathbb{E}_{a\sim\pi_\theta}\!\left[\frac{\pi_{\text{ref}}(a)}{\pi_\theta(a)}\right] = \sum_a \pi_\theta(a)\frac{\pi_{\text{ref}}(a)}{\pi_\theta(a)} = \sum_a \pi_{\text{ref}}(a) = 1$, so $\mathbb{E}[e^r - 1] = 0$, and $\mathbb{E}[-r] = \mathbb{E}[k_1] = D_{\mathrm{KL}}$. The $e^r-1$ term is a zero-mean control variate that cancels variance while preserving the expectation. This is the same control-variate trick used throughout variance-reduced estimation.

### KL in the reward vs KL in the loss

There are two structurally different places to put the KL penalty, and they are *not* equivalent.

**KL-in-reward (per-token penalty).** Subtract $\beta\, k(s_t)$ from the reward at each token *before* computing advantages:

$$
\tilde{r}_t = \underbrace{r\cdot\mathbb{1}[t=T]}_{\text{terminal task reward}} - \beta\, k_{\text{KL}}(s_t).
$$

The per-token KL then flows through GAE/advantage computation and gets *credit-assigned and bootstrapped* like any other reward. This is the classic PPO-RLHF formulation (Ziegler et al., 2019; InstructGPT). Its advantage: the KL is treated as part of the return, so the critic learns to predict it and the discounting/GAE machinery shapes it temporally. Its subtlety: the KL gets entangled with advantage normalization — whitening the advantages also rescales the KL penalty's effective strength.

**KL-in-loss (direct penalty term).** Compute advantages from the task reward *only*, then add the KL as a separate, explicit term in the loss:

$$
\mathcal{L} = -\mathbb{E}\!\left[\min(\rho_t \hat{A}_t,\ \text{clip}(\rho_t)\hat{A}_t)\right] + \beta\, \mathbb{E}[k_{\text{KL}}(s_t)].
$$

This is the GRPO formulation. The KL is a clean regularizer on the policy with its own gradient, not laundered through the advantage. Its advantage: clean separation, the KL coefficient means exactly what you think, and there's no interaction with advantage whitening. Its subtlety: you backpropagate through the KL term, so you need a *differentiable* KL estimate (k3 is differentiable in `logp_policy`), whereas KL-in-reward only needs the *value* of the KL (it's a constant added to the reward, no gradient through it).

```python
def policy_loss_kl_in_loss(logp, logp_old, logp_ref, advantages, mask,
                           clip_eps=0.2, beta=0.04):
    """GRPO-style: advantages already computed from task reward only;
    KL is an explicit, differentiable loss term (k3)."""
    ratio = torch.exp(logp - logp_old)                       # importance ratio rho_t
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    pg_loss = -torch.min(unclipped, clipped)                 # per-token PPO surrogate

    logr = logp_ref - logp                                   # differentiable in logp
    kl = torch.expm1(logr) - logr                            # k3, >= 0
    loss_per_tok = pg_loss + beta * kl
    return (loss_per_tok * mask).sum() / mask.sum()          # token-mean (see below)
```

!!! tip "Practitioner tip"

    A recent and increasingly common choice in **RLVR / reasoning** runs is to drop the KL penalty entirely ($\beta=0$). When the reward is a *verifiable* correctness signal (the answer is right or wrong), there is no reward model to hack, and the main job of KL — preventing exploitation of a flawed reward model — disappears. DeepSeek-R1-Zero and several follow-ups report better reasoning gains with no KL term, letting the policy move far from the reference. Keep KL when your reward is a *learned* RM (it can be hacked); consider dropping it when your reward is a *verifier*. See [RL with Verifiable Rewards (RLVR) & The Reasoning Recipe](../05-posttraining-alignment/09-rlvr-reasoning.html).

### Adaptive KL control

With a fixed $\beta$, the realized KL wanders: too small and the policy bolts away from the reference; too large and learning stalls. **Adaptive KL** (Ziegler et al., 2019) steers $\beta$ toward a target KL $D_{\text{targ}}$ with a simple proportional controller:

$$
\beta \leftarrow \beta\cdot\left(1 + K_\beta\cdot\operatorname{clip}\!\left(\frac{D_{\mathrm{KL}}-D_{\text{targ}}}{D_{\text{targ}}},\ -0.2,\ 0.2\right)\right).
$$

If measured KL exceeds the target, $\beta$ grows to push it back down; if it's below target, $\beta$ shrinks to allow more movement. This is a thermostat for policy drift.

```python
class AdaptiveKLController:
    """Proportional controller that nudges beta toward a target KL.
    From the InstructGPT/PPO-RLHF lineage (Ziegler et al., 2019)."""
    def __init__(self, init_beta=0.2, target_kl=6.0, horizon=10000):
        self.beta = init_beta
        self.target = target_kl
        self.horizon = horizon          # adaptation speed (larger = slower)

    def update(self, current_kl, n_steps):
        # proportional error, clipped to +-20% so a single noisy batch
        # cannot swing beta wildly
        proportional_error = max(-0.2, min(0.2, current_kl / self.target - 1.0))
        mult = 1.0 + proportional_error * n_steps / self.horizon
        self.beta *= mult
        return self.beta
```

## Clipping: PPO Ratio, Clip-Higher, and Dual-Clip

The importance ratio $\rho_t = \frac{\pi_\theta(a_t)}{\pi_{\theta_{\text{old}}}(a_t)}$ corrects for the fact that we optimize on data sampled from a slightly stale policy $\pi_{\theta_{\text{old}}}$ (the rollout policy) while updating $\pi_\theta$ over several gradient steps. Left unconstrained, a single large ratio can produce an enormous, destabilizing update. PPO's clipped surrogate (Schulman et al., 2017) bounds it:

$$
\mathcal{L}^{\text{clip}}_t = \min\!\Big(\rho_t \hat{A}_t,\ \operatorname{clip}(\rho_t, 1-\varepsilon_{\text{low}}, 1+\varepsilon_{\text{high}})\,\hat{A}_t\Big).
$$

The `min` is the asymmetry that makes PPO work. Walk through the four cases — they are a favorite whiteboard question:


{{fig:advkl-ppo-clip-quadrant}}


The clip only ever *removes* incentive to move further in a direction you've already moved a lot — it never reverses the sign. When $A_t>0$ and the ratio has already grown past $1+\varepsilon$, the gradient is zeroed so you don't keep inflating that token's probability on stale data.

### Clip-higher (decoupled clipping)

DAPO (Yu et al., 2025) observed that the *symmetric* clip $\varepsilon_{\text{low}}=\varepsilon_{\text{high}}=\varepsilon$ quietly throttles exploration. Consider a token the policy currently assigns probability 0.9 and one it assigns 0.01. To grow the rare token's probability by the same *ratio* headroom $1+\varepsilon$, the absolute increase it's allowed is tiny; meanwhile common tokens hit the ceiling easily. The clip preferentially suppresses the up-weighting of *low-probability* tokens — exactly the exploratory, diverse tokens you want a reasoning model to learn. Their fix, **clip-higher**, decouples the bounds and *raises only the upper one* (e.g. $\varepsilon_{\text{low}}=0.2,\ \varepsilon_{\text{high}}=0.28$):

```python
def ppo_clip_loss(ratio, adv, mask, eps_low=0.2, eps_high=0.28):
    """Decoupled (clip-higher) PPO surrogate. eps_high > eps_low gives
    low-probability/exploratory tokens more room to grow (DAPO)."""
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * adv
    loss = -torch.min(unclipped, clipped)
    # diagnostic: fraction of tokens where the clip was active
    clipfrac = ((ratio > 1 + eps_high) | (ratio < 1 - eps_low)).float()
    clipfrac = (clipfrac * mask).sum() / mask.sum()
    return (loss * mask).sum() / mask.sum(), clipfrac
```

### Dual-clip

When $\hat{A}_t < 0$ *and* the ratio $\rho_t$ becomes large (which happens when the policy has drifted a lot and a bad action's probability has spiked), the term $\rho_t\hat A_t$ becomes a large *negative* number multiplied by a large ratio — a huge negative surrogate whose gradient can explode. **Dual-clip PPO** (Ye et al., 2020) adds a *lower bound* of $c\hat A_t$ (with constant $c>1$, e.g. $c=3$) for the negative-advantage case:

$$
\mathcal{L}^{\text{dual}}_t =
\begin{cases}
\max\!\big(\min(\rho_t\hat A_t,\ \operatorname{clip}(\rho_t)\hat A_t),\ c\,\hat A_t\big) & \hat A_t < 0\\[4pt]
\min\!\big(\rho_t\hat A_t,\ \operatorname{clip}(\rho_t)\hat A_t\big) & \hat A_t \ge 0
\end{cases}
$$

The extra $\max(\cdot, c\hat A_t)$ floors how negative the surrogate can get, capping the gradient magnitude from pathological ratios. Dual-clip matters most in long-horizon or off-policy-heavy settings (multiple epochs over the same rollouts, async RL with stale weights). In on-policy single-epoch reasoning runs it rarely fires.

```python
def dual_clip_loss(ratio, adv, mask, eps=0.2, c=3.0):
    """Dual-clip PPO (Ye et al. 2020). Adds a lower floor c*adv for adv<0."""
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * adv
    standard = torch.min(ratio * adv, clipped)              # usual PPO surrogate
    # For negative advantages, floor the (negative) surrogate at c*adv:
    neg = torch.max(standard, c * adv)                      # only binds when adv<0
    surrogate = torch.where(adv < 0, neg, standard)
    return -(surrogate * mask).sum() / mask.sum()
```

!!! example "Worked example: clipping with real numbers"

    Suppose after a few gradient steps a particular good token ($\hat A_t = +2.0$) has ratio $\rho_t = 1.5$, and we use $\varepsilon_{\text{low}}=0.2,\ \varepsilon_{\text{high}}=0.28$.

    - Unclipped surrogate: $\rho_t \hat A_t = 1.5\times 2.0 = 3.0$.
    - Clipped ratio: $\operatorname{clip}(1.5,\ 0.8,\ 1.28)=1.28$, so clipped surrogate $=1.28\times2.0=2.56$.
    - PPO takes $\min(3.0,\ 2.56)=2.56$. The loss is $-2.56$; the gradient *through the ratio* is **zero** because the clipped branch (a constant w.r.t. the live $\rho$ at the boundary) was selected. The optimizer gets no incentive to push this already-overgrown token further.

    Now a bad token: $\hat A_t = -1.0$, ratio spiked to $\rho_t = 4.0$ (heavy drift). With dual-clip $c=3$:

    - $\rho_t\hat A_t = 4.0\times(-1.0) = -4.0$.
    - Clipped: $\operatorname{clip}(4.0, 0.8, 1.2)\times(-1.0)=1.2\times(-1.0)=-1.2$. Standard PPO $\min(-4.0,-1.2)=-4.0$.
    - Dual-clip floor: $\max(-4.0,\ c\hat A_t)=\max(-4.0,\ -3.0)=-3.0$. The surrogate is floored at $-3.0$ instead of $-4.0$, shrinking the gradient magnitude by 25% and preventing a single drifted token from dominating the batch update.

    A healthy run keeps the **clip fraction** (tokens where the clip is active) in the rough range of 5–20%. A clip fraction climbing past ~40% means the policy is moving faster than PPO's trust region allows — lower the LR, lower the number of inner epochs, or refresh $\pi_{\theta_{\text{old}}}$ more often.

## Entropy Bonus, Loss Aggregation, and the Bag of Tricks

### Entropy regularization

To prevent premature collapse onto a single high-probability continuation (which kills exploration and tanks reasoning diversity), PPO adds an **entropy bonus**: maximize the policy's per-token entropy $H(\pi_\theta(\cdot\mid s_t))$ with a small coefficient $\alpha$:

$$
\mathcal{L} = \mathcal{L}^{\text{clip}} - \alpha\, H(\pi_\theta) + \beta\,\text{KL}.
$$

Note the sign: we *subtract* entropy from the (minimized) loss, i.e. we reward high entropy. The full-distribution entropy $H = -\sum_v p_v\log p_v$ is computed from the *logits over the whole vocabulary*, not from the sampled token, so it needs the logits tensor — a non-trivial memory cost at vocab sizes of 128k+.

```python
def entropy_from_logits(logits, mask):
    """Exact per-token entropy H = -sum_v p_v log p_v over the full vocab,
    averaged over real tokens. logits: (B, T, V)."""
    logp = torch.log_softmax(logits, dim=-1)
    p = logp.exp()
    ent = -(p * logp).sum(dim=-1)                 # (B, T)
    return (ent * mask).sum() / mask.sum()
```

Entropy regularization is a double-edged sword. Too little and the model collapses to greedy, repetitive outputs (you'll see entropy plummet and generation diversity die). Too much and the model never commits, reward stalls, and you get word-salad. A common modern finding in RLVR is that the *natural* entropy dynamics matter: entropy that collapses too fast predicts a stuck run, and several recipes monitor entropy as a primary health metric rather than aggressively forcing it with a large $\alpha$. Typical $\alpha$ is small ($10^{-3}$ to $10^{-2}$) or zero.

### Loss aggregation: token-mean vs sequence-mean

This is the most underappreciated and most consequential implementation choice in the chapter, because it silently changes *what objective you are optimizing*. You have per-token losses $\ell_{i,t}$ for sequence $i$, token $t$, with valid lengths $L_i$. There are two ways to reduce to a scalar:

**Token-level mean** (each token weighted equally across the whole batch):

$$
\mathcal{L}_{\text{token}} = \frac{\sum_i \sum_t m_{i,t}\,\ell_{i,t}}{\sum_i\sum_t m_{i,t}} = \frac{\sum_i\sum_t m_{i,t}\ell_{i,t}}{\sum_i L_i}.
$$

**Sequence-level mean** (each *sequence* weighted equally, regardless of length):

$$
\mathcal{L}_{\text{seq}} = \frac{1}{N}\sum_i \frac{\sum_t m_{i,t}\ell_{i,t}}{L_i}.
$$

The difference is *how much a long sequence contributes*. Under token-mean, a 600-token response contributes 6x the gradient of a 100-token one — so **token-mean rewards length** when advantages are positive, and is implicated in the length-explosion failure mode of GRPO. Under sequence-mean, every response counts once, so per-token gradients in long sequences are *down-weighted* — which can under-train the very long reasoning chains you care about. DAPO's "token-level loss" and the surrounding discussion argue for token-mean (with a fixed global denominator) precisely to give long correct reasoning its due weight, while controlling length through *other* means (explicit length penalties, dynamic sampling).

```python
def aggregate_loss(per_token_loss, mask, mode="token_mean"):
    """Reduce (B, T) per-token loss to a scalar. The mode silently changes
    the objective — choose deliberately."""
    if mode == "token_mean":
        # every token equal -> long sequences contribute proportionally more
        return (per_token_loss * mask).sum() / mask.sum().clamp_min(1.0)
    elif mode == "seq_mean":
        # every sequence equal -> per-sequence mean, then mean over sequences
        seq_len = mask.sum(dim=1).clamp_min(1.0)              # (B,) valid lengths
        seq_loss = (per_token_loss * mask).sum(dim=1) / seq_len
        return seq_loss.mean()
    elif mode == "token_mean_fixed":
        # DAPO-style: divide by a fixed constant (e.g. max_len) so the
        # denominator does not depend on batch composition
        return (per_token_loss * mask).sum() / per_token_loss.shape[1]
    else:
        raise ValueError(mode)
```

{{fig:advkl-loss-aggregation-length}}

!!! example "Worked example: why aggregation changes the answer"

    Batch of two completions for the same prompt, both *correct* (advantage $+1$ per token after sign): one is 10 tokens, one is 200 tokens. Say each token's surrogate loss is $-1.0$ (we want to push them up).

    - **Token-mean:** total $= (10 + 200)\times(-1.0) = -210$; denominator $= 210$; loss $=-1.0$. The 200-token sequence supplied 200/210 ≈ **95% of the gradient signal**. The model learns "long correct answers are very good" — length creeps up over training.
    - **Sequence-mean:** seq-1 mean $=-1.0$, seq-2 mean $=-1.0$; batch mean $=-1.0$. Both sequences contributed **equally** despite the 20x length difference. Per-token, the long sequence's tokens were down-weighted 20x.

    Same data, same per-token losses, *different gradient* and *different length dynamics*. When you read a codebase, find the loss-reduction line first — it tells you more about the run's behavior than any hyperparameter.

### The stabilization checklist

Beyond the big three (advantage, KL, clip), production RL leans on a long tail of smaller tricks. Here is the working engineer's checklist, each with the one-line reason it exists.

- **Importance-ratio sanity clamp.** Clamp `logp - logp_old` to e.g. $[-20, 20]$ before `exp` so a numerical blip can't produce `inf` ratios. Cheap insurance.
- **Recompute old logprobs with the *training* engine.** Rollouts come from vLLM/SGLang; the training forward pass uses a different kernel/precision. The `logp_old` used in the ratio must be consistent — either recompute it under the trainer, or correct for the train/inference logprob mismatch. This mismatch is a leading cause of "my GRPO ratio is systematically off 1.0." See [The Generation–Training Loop & Rollout Engines](../06-rl-infra/02-generation-training-loop.html).
- **Gradient clipping by global norm.** `clip_grad_norm_(params, max_norm=1.0)` after backward; a single bad batch can produce a huge gradient that one clip absorbs. Log the *pre-clip* grad norm — sudden spikes are your earliest warning of instability (see [Training Stability, Loss Spikes & Debugging Large Runs](../03-pretraining/11-training-stability.html)).
- **Reward / advantage clipping.** Clip raw rewards to a sane range (e.g. $[-10, 10]$) so a reward-function bug or an outlier verifier score can't dominate.
- **Mask everything consistently.** Prompt tokens, padding, and (for multi-turn) tool-output tokens must be masked out of the loss, the KL, the entropy, *and* the advantage normalization. An inconsistent mask between these is a silent, slow-bleeding bug.
- **Overlong-sequence handling.** Truncated (length-capped) generations have no EOS and an ambiguous reward. DAPO's "overlong filtering / soft punishment" either drops them or applies a soft length penalty rather than a hard zero, avoiding a noisy negative signal.
- **Reference-model refresh / removal.** For long RLVR runs some recipes periodically reset the reference to the current policy (so KL is measured against a moving anchor) or drop it entirely once the model is stable.
- **bf16 for the policy, fp32 for the reductions.** Compute log-softmax, KL, entropy, and advantage statistics in fp32 even when the model runs in bf16 — these sums over long sequences and large vocabularies lose precision badly in bf16. See [Mixed Precision, bf16 & FP8 Training](../03-pretraining/08-mixed-precision-fp8.html) and [Numerical Computing, Floating Point & Precision](../01-foundations/04-numerics-precision.html).

Here is a compact training step that wires these together so you can see the data flow end to end.

```python
def grpo_train_step(policy_logp, old_logp, ref_logp, full_logits,
                    rewards, group_size, mask,
                    eps_low=0.2, eps_high=0.28, beta=0.0, ent_coef=0.0):
    """One GRPO-style update. All log-prob tensors are (B, T); rewards (B,).
    Returns scalar loss and a metrics dict for logging."""
    # 1) Group-relative advantage from terminal reward, broadcast to tokens.
    adv_seq = grpo_advantages(rewards, group_size, scale_by_std=False)   # (B,)
    adv = adv_seq.unsqueeze(1).expand_as(mask)                           # (B, T)
    # 2) Optional batch-level whitening of *scale* only (mean already removed).
    adv = masked_whiten(adv, mask, shift_mean=False)
    # 3) Importance ratio with a numerical safety clamp.
    log_ratio = (policy_logp - old_logp).clamp(-20.0, 20.0)
    ratio = torch.exp(log_ratio)
    # 4) Clip-higher PPO surrogate (per token).
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * adv
    pg = -torch.min(unclipped, clipped)
    # 5) KL-in-loss (k3, differentiable) and entropy bonus.
    logr = (ref_logp - policy_logp)
    kl = torch.expm1(logr) - logr                                       # (B, T), >=0
    ent = entropy_from_logits(full_logits, mask)                        # scalar
    per_tok = pg + beta * kl
    loss = (per_tok * mask).sum() / mask.sum().clamp_min(1.0) - ent_coef * ent
    # 6) Diagnostics — log these every step.
    with torch.no_grad():
        clipfrac = (((ratio > 1 + eps_high) | (ratio < 1 - eps_low)).float()
                    * mask).sum() / mask.sum()
        approx_kl = (kl * mask).sum() / mask.sum()
    return loss, {"clipfrac": clipfrac.item(), "kl": approx_kl.item(),
                  "entropy": ent.item(), "ratio_mean": ratio.mean().item()}
```

!!! interview "Interview Corner"

    **Q:** In a GRPO run your reward climbs steadily for 300 steps, then collapses, generations become repetitive, and KL to the reference spikes. The clip fraction has crept from 10% to 55%. Walk me through your diagnosis and the *minimal* set of changes you'd try, in order.

    **A:** The symptoms are a textbook **trust-region breach driven by entropy collapse**. A rising clip fraction (10% → 55%) means the policy is moving far faster per update than PPO's clip can absorb — most tokens are now on the clipped branch, so the effective gradient is being thrown away and the updates that *do* land are the unclipped, high-ratio ones, which are exactly the destabilizing ones. The repetition + entropy collapse + KL spike confirm the policy has run away from the reference into a degenerate mode (often reward-hacking a flaw in the reward, or just collapsing onto one high-probability continuation).

    Minimal changes, cheapest first: **(1)** lower the learning rate (the ratio is moving too fast — this directly shrinks the clip fraction); **(2)** reduce the number of inner PPO epochs per rollout batch (more epochs = more off-policy drift = larger ratios), ideally to 1 for on-policy GRPO; **(3)** add or raise the entropy bonus, or at least *monitor* entropy as the leading indicator — its collapse preceded the reward collapse. If KL was off ($\beta=0$): **(4)** re-introduce a small KL-in-loss term (k3) or lower the target KL on the adaptive controller; **(5)** check the reward function for a hackable shortcut, since RLVR collapses are frequently the model finding a degenerate way to score (e.g. emitting the answer format without reasoning). I would *not* first reach for dual-clip or advantage-whitening tweaks — those are second-order here. The root cause is too-large effective steps, and LR / epochs / entropy address it directly.

    Follow-up they'll ask: *why does a high clip fraction make things worse rather than safer?* Because the clip zeroing gradients is not symmetric protection — it removes the *corrective* gradients on already-moved tokens while the rare unclipped tokens (often the destabilizing low-probability ones) still update, so a high clip fraction means you're increasingly training on a biased, adversarial subset of your own batch.

## Key Takeaways

!!! key "Key Takeaways"

    - The advantage $A_t = Q(s_t,a_t)-V(s_t)$ is the per-token learning signal; the three production families are Monte-Carlo/RLOO (scalar baseline, no critic), GAE ($\lambda$ trades bias for variance, needs a critic), and GRPO (group mean/std as baseline, critic-free). For terminal-reward reasoning tasks, group-relative methods have largely displaced the critic.
    - Always normalize/whiten advantages over the **masked** (non-pad) tokens, in fp32, ideally globally across data-parallel ranks. Be deliberate about *shifting* the mean more than once — RLOO/GRPO already subtract a baseline. Dr. GRPO argues for dropping the group-std divisor to remove a difficulty-weighting bias.
    - Use the **k3** KL estimator $e^r-1-r$ by default: unbiased *and* non-negative. k1 is unbiased but can go negative; k2 ($\frac12 r^2$) is biased but stable when distributions are close.
    - **KL-in-reward** (per-token, flows through GAE) and **KL-in-loss** (explicit differentiable term) are not equivalent; GRPO uses KL-in-loss. For *verifiable* rewards, dropping KL entirely ($\beta=0$) is now common; keep KL when the reward is a *learned* RM that can be hacked.
    - PPO clipping is asymmetric by design (the `min`): it only removes incentive to over-move, never reverses sign. **Clip-higher** ($\varepsilon_{\text{high}}>\varepsilon_{\text{low}}$) restores exploration on low-probability tokens; **dual-clip** floors the surrogate for large-ratio negative-advantage tokens.
    - **Loss aggregation silently changes the objective**: token-mean rewards length (long sequences dominate the gradient), sequence-mean weights each response equally. Pick deliberately; it drives the length-inflation failure mode.
    - Monitor `clip_fraction` (healthy ~5–20%), KL, entropy, and pre-clip grad norm every step. A clip fraction climbing past ~40% with collapsing entropy is the canonical "trust region breached" signature — respond with lower LR, fewer inner epochs, and entropy/KL pressure before anything fancier.
    - The unglamorous tricks carry the run: ratio clamping before `exp`, global-norm grad clipping, reward clipping, consistent masking across loss/KL/entropy/normalization, and reconciling rollout-engine vs trainer logprobs.

!!! sota "State of the Art & Resources (2026)"
    Advantage estimation, KL control, and PPO clipping for LLMs are now a mature engineering discipline: the foundational algorithms (GAE, PPO, adaptive KL) date to 2015–2019, while 2023–2025 work has refined them specifically for sparse-reward, long-chain-of-thought settings — delivering clip-higher, Dr. GRPO, REINFORCE++, and the token-vs-sequence loss debate. The main open frameworks (verl, TRL, OpenRLHF) implement nearly all variants and are the best place to read production-grade code.

    **Foundational work**

    - [Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation* (2015)](https://arxiv.org/abs/1506.02438) — introduces GAE and the γ–λ bias/variance knob; the backward recurrence every RL codebase implements.
    - [Schulman et al., *Proximal Policy Optimization Algorithms* (2017)](https://arxiv.org/abs/1707.06347) — the clipped surrogate objective and trust-region intuition behind every PPO implementation.
    - [Ziegler et al., *Fine-Tuning Language Models from Human Preferences* (2019)](https://arxiv.org/abs/1909.08593) — first application of KL-penalized PPO to LLMs; introduces KL-in-reward and the adaptive KL controller.

    **Recent advances (2023–2026)**

    - [Shao et al., *DeepSeekMath* (2024)](https://arxiv.org/abs/2402.03300) — introduces GRPO and group-relative advantages, removing the critic for sparse verifiable-reward tasks.
    - [Liu et al., *Understanding R1-Zero-Like Training: A Critical Perspective* (2025)](https://arxiv.org/abs/2503.20783) — diagnoses the std-normalization and length biases in GRPO; proposes Dr. GRPO (mean-only normalization).
    - [Yu et al., *DAPO: An Open-Source LLM RL System at Scale* (2025)](https://arxiv.org/abs/2503.14476) — clip-higher, dynamic sampling, token-level loss, and overlong reward shaping; achieves SOTA on AIME 2024.
    - [Hu et al., *REINFORCE++* (2025)](https://arxiv.org/abs/2501.03262) — critic-free RL with global advantage normalization; more stable than GRPO, faster than PPO.

    **Open-source & tools**

    - [volcengine/verl](https://github.com/volcengine/verl) — ByteDance's production RL framework (FSDP + Megatron + vLLM/SGLang); DAPO and GRPO reference implementation.
    - [huggingface/trl](https://github.com/huggingface/trl) — HuggingFace's post-training library; GRPOTrainer and PPOTrainer with all major advantage/KL variants.
    - [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — Ray-based scalable RLHF framework supporting PPO, GRPO, RLOO, and REINFORCE++.

    **Go deeper**

    - [Yihua Zhang, *Re-understanding KL Approximation from an RL-for-LLM Lens* (2025)](https://huggingface.co/blog/NormalUhr/kl-divergence-estimator-rl-llm) — clear walk-through of why k3 is unbiased and non-negative, with LLM-specific context.

## Further reading

- **Schulman, Moritz, Levine, Jordan, Abbeel, "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (2015)** — the GAE paper; the $\gamma$–$\lambda$ bias/variance derivation and the backward recurrence.
- **Schulman, Wolski, Dhariwal, Radford, Klimov, "Proximal Policy Optimization Algorithms" (2017)** — the clipped surrogate objective and the trust-region intuition.
- **John Schulman, "Approximating KL Divergence" (blog note)** — the source of the k1/k2/k3 estimators and the control-variate argument for k3.
- **Ziegler et al., "Fine-Tuning Language Models from Human Preferences" (2019)** and **Ouyang et al., "Training Language Models to Follow Instructions with Human Feedback" (InstructGPT, 2022)** — KL-in-reward and the adaptive KL controller in the RLHF setting.
- **Shao et al., "DeepSeekMath" (2024)** — introduces GRPO and group-relative advantages.
- **Liu et al., "Understanding R1-Zero-Like Training: A Critical Perspective" / Dr. GRPO (2025)** — the std-normalization and length-bias critique of GRPO.
- **Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning System at Scale" (2025)** — clip-higher, dynamic sampling, token-level loss, and overlong handling.
- **Ye et al., "Mastering Complex Control in MOBA Games with Deep Reinforcement Learning" (2020)** — the dual-clip PPO variant.
- **The `verl`, `trl`, and `OpenRLHF` repositories** — read their advantage-computation and policy-loss functions side by side; the differences in masking, whitening, and aggregation are the real curriculum.

## Exercises

**1.** (Conceptual) A colleague computes GRPO advantages, then runs `advantages.mean()` and `advantages.std()` over the full padded `(B, T)` tensor to whiten them, ignoring the loss mask. Their batch happens to be 40% padding. Explain precisely what goes wrong: what happens to the estimated mean and std, and in which direction are the real tokens' advantages mis-scaled?

??? note "Solution"

    Padding positions carry advantage $0$. With 40% of the tensor being zeros:

    - **Mean is pulled toward zero.** The true masked mean is $\mu_{\text{real}} = \frac{1}{N_{\text{real}}}\sum_{\text{real}} A_t$. The padded mean is $\mu_{\text{pad}} = \frac{N_{\text{real}}}{N_{\text{real}}+N_{\text{pad}}}\,\mu_{\text{real}} = 0.6\,\mu_{\text{real}}$. So the mean is shrunk by the fraction of real tokens.
    - **Std/variance is shrunk.** The padded variance mixes in $0.4$ worth of points sitting at value $0$ (near or below the real mean), which reduces the spread. Concretely the padded second moment about the padded mean is smaller than the real variance, so $\sigma_{\text{pad}} < \sigma_{\text{real}}$.

    After whitening, each real token gets $\hat A_t = (A_t - \mu_{\text{pad}})/(\sigma_{\text{pad}}+\varepsilon)$. Because $\sigma_{\text{pad}}$ is too small, the real advantages are **divided by too small a number and therefore scaled UP** (inflated) relative to correct whitening, and they are shifted by the wrong (too-small) mean. Worse, the scale factor depends on *how much padding happened to land in the batch* — so the effective learning rate silently varies with batch composition. This is exactly the pitfall the chapter warns about: always reduce with the loss mask (`masked_whiten`).

**2.** (Quantitative) A prompt is sampled $G=4$ times with rewards $R = [1, 1, 1, 0]$ (three correct, one wrong). Compute the GRPO advantages (a) with standard std-normalization ($\varepsilon = 0$, using the population std with divisor $G$) and (b) Dr. GRPO style (mean subtraction only). Then a second, harder prompt gives $R = [1, 0, 1, 0]$ (a 50/50 split). Compare the magnitude of the advantage assigned to a *correct* sample across the two prompts under standard GRPO, and state which prompt's gradients get up-weighted *relative to mean-only (Dr. GRPO) normalization*.

??? note "Solution"

    **Prompt A, $R=[1,1,1,0]$.** Mean $= 3/4 = 0.75$. Mean-subtracted advantages: $[0.25, 0.25, 0.25, -0.75]$ (this is the Dr. GRPO answer, part b).

    Population variance $= \frac{1}{4}\big(3(0.25)^2 + (-0.75)^2\big) = \frac{1}{4}(3\cdot0.0625 + 0.5625) = \frac{1}{4}(0.1875+0.5625)=\frac{0.75}{4}=0.1875$. Std $= \sqrt{0.1875} \approx 0.4330$.

    Standard GRPO advantages: divide by $0.4330$:
    $$[0.25/0.4330,\ \dots,\ -0.75/0.4330] \approx [0.577,\ 0.577,\ 0.577,\ -1.732].$$
    A correct sample gets advantage $\approx +0.577$.

    **Prompt B, $R=[1,0,1,0]$.** Mean $= 0.5$. Mean-subtracted: $[0.5, -0.5, 0.5, -0.5]$.

    Population variance $= \frac{1}{4}\big(4\cdot(0.5)^2\big)=\frac{1}{4}(1.0)=0.25$. Std $=0.5$.

    Standard GRPO advantages: $[0.5/0.5,\dots] = [1.0, -1.0, 1.0, -1.0]$. A correct sample gets advantage $\approx +1.0$.

    **Comparison.** A correct sample gets $+0.577$ in prompt A (the easier 3/4 group) but $+1.0$ in prompt B (the 50/50 group). Taken at face value this looks like the harder prompt's correct samples get the bigger signal — but that is mostly the *mean-subtraction* talking (prompt B's raw mean-subtracted signal is $0.5$ vs prompt A's $0.25$), **not** the std bias. To isolate the std effect, look at the multiplicative factor $1/\sigma$ that the divisor applies: prompt A is scaled by $1/0.433 \approx 2.31$, prompt B by only $1/0.5 = 2.0$. The **lower-variance (easier) prompt A is amplified by the larger factor** — exactly the chapter's claim that dividing by std up-weights low-variance groups.

    Equivalently, compare the *ratio* of the two correct-sample advantages before and after the divisor. Under mean-only (Dr. GRPO): $0.25/0.5 = 0.50$. Under standard GRPO: $0.577/1.0 = 0.577$. Std-normalization has shifted weight **toward prompt A** relative to what mean-only gives. Dr. GRPO drops the divisor precisely so that a group's reward variance (a proxy for difficulty) cannot re-weight its gradients this way — the natural difficulty signal ($0.25$ for the easy prompt vs $0.5$ for the hard one) is left intact.

**3.** (Quantitative) You have a single sampled token where the reference assigns it higher probability than the policy: $\log\pi_\theta(a) = -3.0$ and $\log\pi_{\text{ref}}(a) = -2.0$. Compute the three KL estimators $k_1$, $k_2$, $k_3$ for this token. Which one(s) are negative, and what does that say about using $k_1$ as a per-token penalty?

??? note "Solution"

    Recall $r = \log\frac{\pi_{\text{ref}}}{\pi_\theta} = \log\pi_{\text{ref}} - \log\pi_\theta = -2.0 - (-3.0) = +1.0$.

    - $k_1 = -r = -1.0$.
    - $k_2 = \tfrac12 r^2 = \tfrac12 (1.0)^2 = 0.5$.
    - $k_3 = e^r - 1 - r = e^{1.0} - 1 - 1.0 = 2.71828 - 2.0 = 0.71828$.

    So $k_1 = -1.0$ is **negative**, while $k_2 = 0.5$ and $k_3 \approx 0.718$ are both non-negative (as they must be: $k_2 = \tfrac12 r^2 \ge 0$ and $e^r - 1 - r \ge 0$ for all $r$).

    A negative $k_1$ occurs exactly when the policy assigns *lower* probability than the reference to the sampled token ($\pi_\theta < \pi_{\text{ref}}$, i.e. $r>0$). If you use $k_1$ directly as a per-token KL penalty subtracted from the reward, that token would get a *negative penalty* — i.e. a small *bonus* — for being off-reference, which is a nonsensical, high-variance signal even though $\mathbb{E}[k_1] = D_{\mathrm{KL}} \ge 0$ over many samples. This is why $k_3$ (unbiased AND always $\ge 0$) is the modern default: it never hands out a spurious per-token bonus.

**4.** (Conceptual) The chapter says KL-in-reward only needs the *value* of the KL, whereas KL-in-loss needs a *differentiable* KL estimate. Explain why, in terms of where each KL term sits in the computation graph and what gets backpropagated. What would break if you used a `.detach()`-ed (non-differentiable) $k_3$ in the KL-in-loss formulation?

??? note "Solution"

    **KL-in-reward.** The KL is subtracted from the scalar reward *before* advantages are computed: $\tilde r_t = r\cdot\mathbb{1}[t{=}T] - \beta\,k_{\text{KL}}(s_t)$. Advantages are then computed from $\tilde r_t$ and enter the loss only *multiplied against* $\log\pi_\theta$ (the policy-gradient term). The KL contributes to the loss purely as a numeric constant folded into $\hat A_t$; the gradient w.r.t. $\theta$ flows through the $\log\pi_\theta(a_t)$ factor, **not** through the KL value. So the KL only needs to be a number — it is treated as part of the (constant, detached) return/advantage. This is why in the classic formulation the KL is computed under `no_grad` and just added to the reward.

    **KL-in-loss.** The KL is an explicit additive term: $\mathcal{L} = \mathcal{L}^{\text{clip}} + \beta\,\mathbb{E}[k_{\text{KL}}]$. Here we *want* $\nabla_\theta$ of the KL term itself to push the policy back toward the reference. That requires $k_3 = e^{\text{logr}} - \text{logr} - 1$ with $\text{logr} = \log\pi_{\text{ref}} - \log\pi_\theta$ to remain connected to $\theta$ through $\log\pi_\theta$.

    **If you `.detach()` $k_3$ here:** the KL term becomes a constant w.r.t. $\theta$, so $\nabla_\theta(\beta\,k_3) = 0$. The regularizer would contribute nothing to the gradient — the policy would receive *no* pull back toward the reference from the KL term, and $\beta$ would effectively be zero for optimization purposes (it would still change the reported loss value, misleadingly). The policy would be free to drift exactly as if there were no KL penalty.

**5.** (Implementation) Extend the chapter's `masked_whiten` into `masked_whiten_global` that computes the mean and variance across **all data-parallel ranks** (global whitening) instead of per-GPU, using `torch.distributed` all-reduces. Assume a process group is initialized. Keep the fp32-reduction and masking discipline. Explain in one line why this matters.

??? note "Solution"

    Global whitening needs the *global* sums $\sum m_t$, $\sum m_t A_t$, and $\sum m_t A_t^2$, each all-reduced with `SUM`, then combined into mean and variance. Do the reductions in fp32.

    ```python
    import torch
    import torch.distributed as dist

    def masked_whiten_global(advantages, mask, shift_mean=True, eps=1e-8):
        """Zero-mean, unit-variance advantages over non-pad tokens, with mean/var
        computed GLOBALLY across all data-parallel ranks. Reductions in fp32."""
        adv32 = advantages.float()
        m = mask.float()
        # Local partial sums for count, sum, and sum-of-squares.
        local = torch.stack([
            m.sum(),                       # n
            (adv32 * m).sum(),             # sum A
            (adv32 * adv32 * m).sum(),     # sum A^2
        ])
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local, op=dist.ReduceOp.SUM)
        n, s1, s2 = local[0], local[1], local[2]
        n = n.clamp_min(1.0)
        mean = s1 / n
        var = s2 / n - mean * mean          # E[A^2] - E[A]^2, over real tokens
        var = var.clamp_min(0.0)            # guard tiny negative from fp error
        whitened = (adv32 - mean) * torch.rsqrt(var + eps)
        if not shift_mean:
            whitened = whitened + mean
        return (whitened * m).to(advantages.dtype)
    ```

    **Why it matters (one line):** per-GPU whitening computes mean/std over each rank's small local batch, giving noisy, rank-dependent statistics; global whitening makes the normalization independent of how the batch was sharded, so the effective objective does not change with data-parallel degree.

**6.** (Implementation, hard) Implement `dual_clip_higher_loss(ratio, adv, mask, eps_low, eps_high, c)` that combines **clip-higher** (asymmetric $\varepsilon_{\text{low}} \ne \varepsilon_{\text{high}}$) with the **dual-clip** floor $c\hat A_t$ for negative-advantage tokens. Then verify it on the chapter's worked-example numbers: (i) a good token $\hat A = +2.0$, $\rho = 1.5$, $\varepsilon_{\text{low}}=0.2$, $\varepsilon_{\text{high}}=0.28$; (ii) a bad token $\hat A = -1.0$, $\rho = 4.0$, $c = 3$ (with symmetric $\varepsilon = 0.2$ for the clip on this token). Report the per-token surrogate value for each.

??? note "Solution"

    ```python
    import torch

    def dual_clip_higher_loss(ratio, adv, mask, eps_low=0.2, eps_high=0.28, c=3.0):
        """Clip-higher PPO surrogate with a dual-clip floor for adv < 0.
        Returns (scalar_loss, per_token_surrogate)."""
        clipped = torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * adv
        standard = torch.min(ratio * adv, clipped)         # clip-higher surrogate
        # For negative advantages, floor the (negative) surrogate at c*adv:
        neg = torch.max(standard, c * adv)                 # only binds when adv < 0
        surrogate = torch.where(adv < 0, neg, standard)
        loss = -(surrogate * mask).sum() / mask.sum().clamp_min(1.0)
        return loss, surrogate
    ```

    **Verification.**

    (i) Good token, $\hat A = +2.0$, $\rho = 1.5$, $\varepsilon_{\text{low}}=0.2$, $\varepsilon_{\text{high}}=0.28$:
    - $\rho\hat A = 1.5\times 2.0 = 3.0$.
    - $\operatorname{clip}(1.5,\,0.8,\,1.28) = 1.28 \Rightarrow$ clipped $= 1.28\times 2.0 = 2.56$.
    - `standard` $= \min(3.0, 2.56) = 2.56$. Since $\hat A \ge 0$, the dual-clip branch is not taken. **Surrogate $= 2.56$** (loss contribution $-2.56$), matching the chapter's worked example. Gradient through $\rho$ is zero (clipped branch selected).

    (ii) Bad token, $\hat A = -1.0$, $\rho = 4.0$, $\varepsilon = 0.2$, $c = 3$:
    - $\rho\hat A = 4.0\times(-1.0) = -4.0$.
    - $\operatorname{clip}(4.0,\,0.8,\,1.2) = 1.2 \Rightarrow$ clipped $= 1.2\times(-1.0) = -1.2$.
    - `standard` $= \min(-4.0, -1.2) = -4.0$.
    - Since $\hat A < 0$: `neg` $= \max(-4.0,\ c\hat A) = \max(-4.0,\ 3\times(-1.0)) = \max(-4.0, -3.0) = -3.0$. **Surrogate $= -3.0$**, exactly the dual-clip floor from the chapter — the surrogate is capped at $-3.0$ instead of $-4.0$, shrinking the gradient magnitude by 25% and preventing one drifted token from dominating the update.
