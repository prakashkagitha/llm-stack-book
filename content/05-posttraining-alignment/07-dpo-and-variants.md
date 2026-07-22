# 5.7 Direct Preference Optimization & Its Variants

In [The RLHF Pipeline & Reward Modeling](../05-posttraining-alignment/05-rlhf-reward-modeling.html) and [Policy Gradients & PPO for Language Models](../05-posttraining-alignment/06-ppo-for-llms.html) we built the classical alignment machine: collect human preference pairs, fit a reward model $r_\phi$ to them, then run PPO to maximize that reward minus a KL penalty that keeps the policy near a frozen reference. It works — it is how InstructGPT and the first ChatGPT were aligned — but it is a *lot* of moving parts. You train a reward model, you spin up online generation, you carry a value network (the critic), you tune clip ranges and KL coefficients, and you pray the whole feedback loop stays stable. A single misbehaving component can quietly wreck the run.

In 2023, **Direct Preference Optimization (DPO)** (Rafailov, Sharma, Mitchell, Manning, Ermon, Finn, *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*) collapsed that pipeline into a single supervised-style loss. The headline result is almost too clean to believe: the constrained RLHF objective has a *closed-form* optimal policy, and if you substitute that closed form back into the reward model's own loss, the reward model **disappears**. What's left is a simple classification loss over preference pairs that you train with ordinary backprop — no reward model, no sampling, no critic, no rollouts. DPO turned alignment from an RL problem into a fine-tuning problem.

This chapter derives DPO from first principles, builds the intuition for its *implicit reward* and the temperature $\beta$, implements the loss from scratch, and then tours the now-large family of descendants — **IPO**, **KTO**, **ORPO**, **SimPO**, **CPO** — each of which fixes a specific weakness. We close with the **DPO-vs-PPO debate**, which is one of the hottest topics an interviewer can hand you. This is among the most interview-dense chapters in the book; read it with a pen.

## From RLHF to a closed-form policy

### The objective we are actually optimizing

The RLHF objective (developed in [The RLHF Pipeline & Reward Modeling](../05-posttraining-alignment/05-rlhf-reward-modeling.html)) is: maximize reward while staying close, in KL divergence, to a reference policy $\pi_{\text{ref}}$ (usually the SFT model). For a prompt $x$ and a response $y$ sampled from policy $\pi_\theta$:

$$
\max_{\pi_\theta}\;\; \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta(\cdot\mid x)}\big[r(x,y)\big]
\;-\; \beta\, \mathbb{D}_{\mathrm{KL}}\!\big(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big).
$$

The KL term is not a nicety — without it the policy collapses onto whatever degenerate output maximizes the reward model, exploiting every crack in $r$ (see [Reward Hacking, Over-Optimization & Alignment Failures](../05-posttraining-alignment/13-reward-hacking-failures.html)). The coefficient $\beta$ controls how hard we are willing to move away from $\pi_{\text{ref}}$.

### The key move: this has a closed-form solution

Here is the pivot the DPO authors exploit. For a *fixed* reward function $r$, this KL-regularized objective is solved exactly — we do not need RL at all. Write the objective per prompt $x$ and treat $\pi_\theta(\cdot\mid x)$ as a probability distribution we optimize freely (one number per possible response $y$). Expanding the KL:

$$
\max_{\pi}\;\sum_y \pi(y\mid x)\Big[r(x,y) - \beta\log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}\Big],
\qquad \text{s.t. } \sum_y \pi(y\mid x)=1.
$$

Divide by $\beta$ and rearrange the bracket: maximizing $\tfrac{1}{\beta}r - \log\frac{\pi}{\pi_{\text{ref}}}$ is the same as *minimizing* the KL between $\pi$ and a distribution proportional to $\pi_{\text{ref}}(y\mid x)\exp\!\big(\tfrac1\beta r(x,y)\big)$. A KL is minimized (to zero) when the two distributions are equal, so the optimum is:

$$
\boxed{\;\pi^{*}(y\mid x) \;=\; \frac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,\exp\!\Big(\tfrac{1}{\beta}\,r(x,y)\Big)\;}
\qquad
Z(x)=\sum_{y}\pi_{\text{ref}}(y\mid x)\exp\!\Big(\tfrac{1}{\beta}r(x,y)\Big).
$$

This is a **Gibbs / Boltzmann distribution**: tilt the reference policy by the exponentiated reward, then renormalize. It is intuitive — the optimal aligned policy is the reference *reweighted* toward high reward, with $\beta$ as a temperature that controls how aggressively. The only problem is $Z(x)$: it sums over *all possible responses*, an astronomically large set, so we cannot compute $\pi^*$ directly. That intractable partition function is exactly why classical RLHF resorts to sampling-based RL (PPO) instead of just writing down the answer.

{{fig:dpo-boltzmann-tilt}}

### Inverting the relationship: the implicit reward

DPO's trick is to refuse to compute $Z(x)$ and instead **solve for the reward in terms of the policy.** Take the log of the boxed equation and rearrange:

$$
r(x,y) \;=\; \beta\,\log\frac{\pi^{*}(y\mid x)}{\pi_{\text{ref}}(y\mid x)} \;+\; \beta\log Z(x).
$$

Read that carefully. It says: *any* reward function induces an optimal policy, and conversely, given the optimal policy we can recover the reward — up to the prompt-dependent constant $\beta\log Z(x)$. So instead of parameterizing a reward model $r_\phi$ and then training a policy to match it, we can **parameterize the reward implicitly through the policy** $\pi_\theta$ that we ultimately want:

$$
r_\theta(x,y) \;=\; \beta\,\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)} \;+\; \beta\log Z(x).
$$

This $r_\theta$ is the **implicit reward** (sometimes "DPO reward"). It is just $\beta$ times the log-ratio of the trained policy to the reference — a quantity we can compute from two forward passes, with no separate reward network at all. The term $\beta\log Z(x)$ is still intractable, but in a moment it will cancel.

## Deriving the DPO loss

### The Bradley–Terry bridge

We need one more ingredient from reward modeling: how preferences relate to rewards. The **Bradley–Terry model** says the probability that a response $y_w$ ("winner") is preferred over $y_l$ ("loser") for prompt $x$ is a logistic function of their reward *difference*:

$$
P(y_w \succ y_l \mid x) \;=\; \sigma\big(r(x,y_w) - r(x,y_l)\big)
\;=\; \frac{1}{1+\exp\!\big(-(r(x,y_w)-r(x,y_l))\big)}.
$$

A reward model is normally trained by maximizing the likelihood of the observed human preferences under this model — a binary cross-entropy on reward *differences*. The crucial structural fact: **only the reward difference matters**, never the absolute reward.

{{fig:dpo-reward-model-cancellation}}

Now substitute the implicit reward into the Bradley–Terry difference. Watch the magic:

$$
r_\theta(x,y_w) - r_\theta(x,y_l)
= \beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}
+ \cancel{\beta\log Z(x)}
- \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}
- \cancel{\beta\log Z(x)}.
$$

The intractable $\beta\log Z(x)$ depends only on $x$, so it is **identical for the winner and loser** and cancels in the difference. This is the whole reason DPO works. We are left with a reward difference expressed purely in policy log-probabilities:

$$
r_\theta(x,y_w) - r_\theta(x,y_l)
= \beta\Big[\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\Big].
$$

### The loss

Plug this into the Bradley–Terry negative log-likelihood and we get the DPO loss. For a dataset $\mathcal{D}$ of preference triples $(x, y_w, y_l)$:

$$
\boxed{\;
\mathcal{L}_{\text{DPO}}(\theta)
= -\,\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}
\Big[\log \sigma\Big(
\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}
-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}
\Big)\Big]\;}
$$

That is it. No reward model, no sampling, no RL. You need exactly four log-probabilities per training example — $\pi_\theta$ and $\pi_{\text{ref}}$ on both the chosen and rejected responses — and a single sigmoid/cross-entropy. The reference log-probs are constant, so in practice you precompute them once or run a frozen reference in a second (no-grad) forward pass. Define the convenient shorthand

$$
\Delta \;=\; \beta\big(\underbrace{\log\pi_\theta(y_w\mid x)-\log\pi_{\text{ref}}(y_w\mid x)}_{\text{chosen log-ratio}} - \underbrace{(\log\pi_\theta(y_l\mid x)-\log\pi_{\text{ref}}(y_l\mid x))}_{\text{rejected log-ratio}}\big),
$$

so $\mathcal{L}_{\text{DPO}} = -\log\sigma(\Delta)$. The loss is small when $\Delta$ is large and positive — i.e., when the policy has raised the chosen response's log-ratio above the rejected one's, *relative to the reference*.

### What the gradient does (and why the reference matters)

Differentiate $-\log\sigma(\Delta)$. Using $\frac{d}{dz}\log\sigma(z)=\sigma(-z)$:

$$
\nabla_\theta \mathcal{L}_{\text{DPO}}
= -\,\beta\,\underbrace{\sigma(-\Delta)}_{\text{weight}}\;
\Big[\nabla_\theta\log\pi_\theta(y_w\mid x) - \nabla_\theta\log\pi_\theta(y_l\mid x)\Big].
$$

This is beautifully interpretable. The gradient **increases** the log-probability of the chosen response and **decreases** that of the rejected one — a contrastive push. The scalar weight $\sigma(-\Delta)$ is large precisely when the model is *wrong* about a pair (the implicit reward of the loser exceeds the winner, so $\Delta<0$ and $\sigma(-\Delta)\to 1$) and small when the model already gets the pair right ($\Delta\gg0$, weight $\to 0$). So DPO automatically focuses its gradient on the examples it currently mis-ranks. The reference appears only inside $\Delta$ (and hence inside the weight), gating *how much* to push but not the direction of the push.


{{fig:dpo-margin-gradient-regimes}}


## DPO from scratch

Here is the canonical implementation. The only non-obvious part is computing the *sequence* log-probability — the sum of per-token log-probabilities over the response tokens (masking out the prompt). Everything else is the boxed loss verbatim.

```python
import torch
import torch.nn.functional as F

def sequence_logprob(logits, labels, loss_mask):
    """
    Sum of per-token log p(label_t | context) over response tokens.

    logits    : (B, T, V) raw model logits for positions 0..T-1.
    labels    : (B, T)    token ids; for autoregressive LMs label at position t
                          is the token that should be predicted AT t (already shifted
                          by the caller, see below).
    loss_mask : (B, T)    1.0 for response tokens we score, 0.0 for prompt/padding.
    Returns   : (B,)      summed log-prob of each response.
    """
    logprobs = F.log_softmax(logits, dim=-1)                      # (B, T, V)
    # Gather the log-prob of the actual next token at each position.
    token_logp = torch.gather(
        logprobs, dim=2, index=labels.unsqueeze(-1)
    ).squeeze(-1)                                                 # (B, T)
    return (token_logp * loss_mask).sum(dim=-1)                   # (B,)


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps,   ref_rejected_logps,
             beta=0.1):
    """
    The DPO loss. Inputs are SEQUENCE log-probs (already summed over tokens),
    one scalar per example, for the policy and frozen reference on both responses.

    Returns
      loss          : scalar to call .backward() on
      chosen_reward : β·(logπθ(yw) − logπref(yw))   — the implicit reward of the winner
      reject_reward : β·(logπθ(yl) − logπref(yl))   — the implicit reward of the loser
    """
    # Log-ratios π_θ / π_ref for chosen and rejected responses.
    chosen_logratio   = policy_chosen_logps   - ref_chosen_logps      # (B,)
    rejected_logratio = policy_rejected_logps - ref_rejected_logps    # (B,)

    # Δ = β · (chosen_logratio − rejected_logratio).
    logits = beta * (chosen_logratio - rejected_logratio)            # (B,)

    # −log σ(Δ) == softplus(−Δ); softplus is numerically stable.
    loss = F.softplus(-logits).mean()

    # Implicit rewards, handy to LOG (detached — diagnostics only).
    chosen_reward = (beta * chosen_logratio).detach()
    reject_reward = (beta * rejected_logratio).detach()
    return loss, chosen_reward, reject_reward
```

And the training step that wires the reference model in. The reference is frozen, so its forward pass runs under `torch.no_grad()`; in production you usually *precompute* `ref_*_logps` once over the whole dataset and cache them, which removes the reference model from the hot loop entirely and halves memory.

```python
def dpo_training_step(policy, ref_model, batch, beta=0.1):
    """
    batch contains, for chosen and rejected responses:
      *_input_ids (B, T), *_labels (B, T) [shifted], *_loss_mask (B, T).
    """
    def logps(model, ids, labels, mask, grad):
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            logits = model(ids).logits[:, :-1, :]    # predict next token
        return sequence_logprob(logits, labels[:, 1:], mask[:, 1:])

    # Policy: needs gradients. Two forward passes (chosen, rejected).
    pol_chosen = logps(policy, batch["chosen_input_ids"],
                       batch["chosen_labels"], batch["chosen_loss_mask"], grad=True)
    pol_reject = logps(policy, batch["rejected_input_ids"],
                       batch["rejected_labels"], batch["rejected_loss_mask"], grad=True)

    # Reference: frozen, no gradients (or precomputed & cached offline).
    ref_chosen = logps(ref_model, batch["chosen_input_ids"],
                       batch["chosen_labels"], batch["chosen_loss_mask"], grad=False)
    ref_reject = logps(ref_model, batch["rejected_input_ids"],
                       batch["rejected_labels"], batch["rejected_loss_mask"], grad=False)

    loss, r_chosen, r_reject = dpo_loss(pol_chosen, pol_reject,
                                        ref_chosen, ref_reject, beta=beta)

    # The single most useful training metric: preference accuracy =
    # fraction of pairs where implicit reward of chosen > rejected.
    reward_acc = (r_chosen > r_reject).float().mean()
    margin = (r_chosen - r_reject).mean()
    return loss, {"reward_acc": reward_acc.item(), "reward_margin": margin.item()}
```

!!! warning "Common pitfall: forgetting to mask the prompt and right-shift"

    Two off-by-one bugs cause most broken DPO runs. (1) **Score only response tokens.** The `loss_mask` must zero out the prompt; otherwise you're optimizing the log-prob of the *prompt* (identical for chosen and rejected, so it partly cancels, but it pollutes the gradient and the length statistics). (2) **Shift labels.** For a causal LM, `logits[:, t]` predicts token `t+1`, so labels must be the input shifted left by one — hence the `[:, 1:]` / `[:, :-1]` slicing above. Get either wrong and your reward accuracy hovers near 0.5 forever. Always log `reward_acc` and `reward_margin`; if accuracy isn't climbing above 0.5 in the first hundred steps, you have a masking/shift bug, not a hyperparameter problem.

### The role of $\beta$

$\beta$ is the same KL temperature from the original objective, and it controls the *strength of the trust region* around $\pi_{\text{ref}}$. In the loss it scales $\Delta$, so it sets how hard the policy is allowed to push the chosen/rejected log-ratios apart before the sigmoid saturates.

- **Small $\beta$** (e.g. 0.01): weak regularization. The policy can drift far from $\pi_{\text{ref}}$; you fit preferences hard but risk degeneration, reward over-optimization, and the implicit-reward magnitude exploding.
- **Large $\beta$** (e.g. 0.5): strong regularization. The policy stays close to $\pi_{\text{ref}}$; safer but may underfit the preferences and barely move.
- **Typical**: $\beta \in [0.1, 0.5]$, with **0.1** the de-facto default from the paper and TRL. Tune it like a learning rate; it is the most important DPO hyperparameter after the LR itself.

!!! example "Worked example: one DPO update by hand"

    Take $\beta=0.1$ and one preference pair. Suppose for the *chosen* response the policy assigns summed log-prob $\log\pi_\theta(y_w)=-12.0$ and the reference $\log\pi_{\text{ref}}(y_w)=-12.5$. For the *rejected* response, $\log\pi_\theta(y_l)=-10.0$ and $\log\pi_{\text{ref}}(y_l)=-11.0$.

    Chosen log-ratio: $-12.0-(-12.5)=+0.5$. Rejected log-ratio: $-10.0-(-11.0)=+1.0$.

    Implicit rewards: $r_w=\beta\cdot 0.5 = 0.05$, $r_l=\beta\cdot 1.0 = 0.10$. The model currently thinks the **rejected** response is better ($r_l>r_w$) — it is wrong about this pair.

    Margin: $\Delta = 0.1\,(0.5-1.0) = -0.05$. Loss $=-\log\sigma(-0.05)=-\log(0.4875)=0.718$ nats — *above* $\log 2\approx0.693$, the loss of a coin-flip, confirming the model mis-ranks this pair. Gradient weight $=\sigma(-\Delta)=\sigma(0.05)=0.5125$ (near maximal), so this example contributes a strong corrective gradient that raises $\log\pi_\theta(y_w)$ and lowers $\log\pi_\theta(y_l)$.

    Now imagine after some training the chosen log-ratio rises to $+1.5$ and rejected falls to $0.0$. Then $\Delta=0.1(1.5-0.0)=0.15$, loss $=-\log\sigma(0.15)=0.62$ nats, weight $=\sigma(-0.15)=0.46$ — smaller. The pair is now ranked correctly and contributes less. This is the self-curriculum: solved pairs fade, hard pairs dominate.

## The variants: fixing what DPO gets wrong

DPO is elegant but not perfect. Three weaknesses drive the entire variant zoo: **(1)** it overfits / pushes margins to infinity on deterministic preferences (no explicit margin cap); **(2)** it needs *paired* data $(y_w,y_l)$ for the same prompt, which is expensive; and **(3)** it needs a frozen reference model, doubling memory and adding a forward pass. Each variant attacks one or more of these.

### IPO — capping the margin

**Identity Preference Optimization (IPO)** (Azar et al., *A General Theoretical Paradigm to Understand Learning from Human Preferences*) diagnoses a subtle DPO flaw. Bradley–Terry assumes preferences come from underlying real-valued rewards, but human preference data is often **near-deterministic**: for a clearly better $y_w$, the empirical preference probability is essentially 1. When $P(y_w\succ y_l)\to 1$, the BT log-likelihood is minimized by driving the reward *difference* to $+\infty$ — i.e., DPO keeps pushing $\Delta$ ever larger with no natural stopping point, and the $\sigma$ saturates so the KL regularization stops biting. The policy overfits, collapsing $\pi_\theta(y_l)\to 0$ and drifting arbitrarily far from $\pi_{\text{ref}}$.

IPO replaces the log-sigmoid with a **squared-error loss around a target margin** $\tfrac{1}{2\beta}$, which has a finite optimum:

$$
\mathcal{L}_{\text{IPO}} = \mathbb{E}\Big[\Big(\underbrace{h_\theta(x,y_w,y_l)}_{\text{log-ratio difference}} - \tfrac{1}{2\beta}\Big)^2\Big],
\quad
h_\theta = \log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}.
$$

```python
def ipo_loss(pol_chosen, pol_reject, ref_chosen, ref_reject, beta=0.1):
    # h = (chosen log-ratio) − (rejected log-ratio); note: NO β multiplier inside.
    h = (pol_chosen - ref_chosen) - (pol_reject - ref_reject)
    # Regress h toward the finite target 1/(2β) instead of pushing it to +∞.
    return ((h - 1.0 / (2.0 * beta)) ** 2).mean()
```

Because the target is finite, IPO cannot push the margin to infinity; here $\beta$ literally sets the *target gap* between chosen and rejected log-ratios. IPO is more robust to label noise and to repeated/over-represented pairs, at some cost in peak performance on clean data.

### KTO — unpaired, prospect-theory alignment

**Kahneman–Tversky Optimization (KTO)** (Ethayarajh, Xu, Muennighoff, Jurafsky, Kiela, *KTO: Model Alignment as Prospect Theory Optimization*) removes the **pairing requirement** entirely. Instead of $(x,y_w,y_l)$ triples, KTO consumes individual examples each labeled simply **desirable** or **undesirable** (a thumbs-up/thumbs-down dataset). This is enormously practical: production feedback is almost always unpaired — users rate single completions, they don't rank pairs.

KTO is grounded in **prospect theory**: humans are loss-averse (a loss hurts more than an equivalent gain pleases) and evaluate outcomes relative to a reference point. KTO builds a **value function** $v$ on the implicit reward $r_\theta(x,y)=\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$, measured against a reference point $z_0$ (a running estimate of the average KL/reward, computed in-batch). Desirable examples should have reward above $z_0$; undesirable ones below it, with separate weights $\lambda_D,\lambda_U$ that let you encode loss-aversion and correct for class imbalance:

$$
\mathcal{L}_{\text{KTO}} = \mathbb{E}\big[\, \lambda_y - v(x,y)\,\big],\qquad
v(x,y)=\begin{cases}
\lambda_D\,\sigma\!\big(r_\theta - z_0\big) & y \text{ desirable}\\[2pt]
\lambda_U\,\sigma\!\big(z_0 - r_\theta\big) & y \text{ undesirable}
\end{cases}
$$

```python
def kto_loss(policy_logps, ref_logps, labels_desirable, kl_z0,
             beta=0.1, lam_desirable=1.0, lam_undesirable=1.0):
    """
    policy_logps, ref_logps : (B,) sequence log-probs (single, UNPAIRED responses).
    labels_desirable        : (B,) bool — True = thumbs up, False = thumbs down.
    kl_z0                    : scalar reference point z0 on the IMPLICIT-REWARD scale,
                              i.e. beta times the detached running-mean in-batch KL
                              (same units as r = beta * log(pi/pi_ref)). beta is
                              already folded into r and z0, so it is NOT reapplied
                              inside the sigmoid below.
    """
    r = beta * (policy_logps - ref_logps)                 # implicit reward, (B,)
    margin = r - kl_z0
    # Desirable: want margin > 0  → value = σ(margin); loss = λ_D · (1 − value).
    # Undesirable: want margin < 0 → value = σ(−margin); loss = λ_U · (1 − value).
    desirable_loss   = lam_desirable   * (1.0 - torch.sigmoid(margin))
    undesirable_loss = lam_undesirable * (1.0 - torch.sigmoid(-margin))
    loss = torch.where(labels_desirable, desirable_loss, undesirable_loss)
    return loss.mean()
```

KTO often **matches or beats DPO** despite using strictly weaker (unpaired) supervision, and it degrades gracefully under extreme class imbalance — making it a strong default when your feedback is a stream of single-completion ratings rather than curated pairs.

### ORPO — alignment with no reference model

**Odds Ratio Preference Optimization (ORPO)** (Hong, Lee, Thorne, *ORPO: Monolithic Preference Optimization without Reference Model*) goes after the **reference model** itself. DPO needs $\pi_{\text{ref}}$ both to define the implicit reward and to anchor the KL. ORPO observes that if you fold preference learning *directly into SFT*, the SFT term anchors the model and a reference becomes unnecessary. ORPO's loss is the standard SFT negative log-likelihood on the chosen response **plus** an odds-ratio penalty that pushes the *odds* of the chosen above the rejected:

$$
\mathcal{L}_{\text{ORPO}} = \underbrace{-\log\pi_\theta(y_w\mid x)}_{\text{SFT on the winner}}
\;-\;\lambda\,\log\sigma\Big(\log\frac{\mathrm{odds}_\theta(y_w\mid x)}{\mathrm{odds}_\theta(y_l\mid x)}\Big),
\qquad
\mathrm{odds}=\frac{\pi}{1-\pi}.
$$

The odds ratio (rather than a probability ratio) is the deliberate choice: $\log$-odds penalizes the rejected response more gently as it gets unlikely, avoiding the runaway suppression that a raw probability ratio would cause. The win is operational: **one model, one stage.** You merge SFT and alignment into a single pass — no separate preference-tuning phase, no reference forward pass, roughly half the memory of DPO.

```python
def orpo_loss(pol_chosen_logps, pol_rejected_logps,
              chosen_len, rejected_len, chosen_nll, lam=0.1):
    """
    pol_chosen_logps / pol_rejected_logps : (B,) SEQUENCE log-probs (sum over tokens)
        of chosen/rejected under the policy -- e.g. the output of sequence_logprob.
    chosen_len / rejected_len : (B,) number of scored response tokens in each
        response (the per-example sum of the loss_mask). Used to length-normalize
        the log-probs BEFORE the odds so exp(mean_logp) is a valid per-token
        probability in (0, 1); matches TRL ORPOTrainer's average_log_prob=True.
    chosen_nll : (B,) standard SFT negative log-likelihood on the chosen response.
    """
    # Length-normalize to a per-token MEAN log-prob. Without this, a summed logp
    # is very negative (e.g. -12), exp(-12) ~ 6e-6, log1p(-exp) ~ 0, and the odds
    # term collapses to a plain (mean_chosen - mean_rejected) log-prob difference.
    mean_chosen   = pol_chosen_logps   / chosen_len       # (B,)
    mean_rejected = pol_rejected_logps / rejected_len     # (B,)

    # log-odds(y) = log[ p/(1-p) ] = logp - log(1 - exp(logp))  (log1mexp form).
    def log_odds(logp):
        return logp - torch.log1p(-torch.exp(logp.clamp(max=-1e-6)))

    log_or = log_odds(mean_chosen) - log_odds(mean_rejected)   # (B,)
    or_term = -torch.nn.functional.logsigmoid(log_or)          # push chosen odds up
    return (chosen_nll + lam * or_term).mean()
```

ORPO is the method of choice when you want a *single*, simple training stage and cannot afford a reference model in memory — common for from-scratch fine-tunes on modest hardware.

### SimPO — reference-free, length-normalized

**SimPO** (Meng, Xia, Chen, *SimPO: Simple Preference Optimization with a Reference-Free Reward*) keeps DPO's contrastive log-sigmoid shape but makes two changes. First, drop the reference model: define the reward as the **length-normalized average log-probability**, $r(x,y)=\tfrac{\beta}{|y|}\log\pi_\theta(y\mid x)$. Length normalization is the key insight — DPO's *summed* log-prob is biased by length (longer sequences have more negative summed log-prob), which SimPO argues is a major cause of DPO's tendency to inflate response length. Second, add a **target reward margin** $\gamma$ that the chosen must beat the rejected by:

$$
\mathcal{L}_{\text{SimPO}} = -\,\mathbb{E}\Big[\log\sigma\Big(
\frac{\beta}{|y_w|}\log\pi_\theta(y_w\mid x) - \frac{\beta}{|y_l|}\log\pi_\theta(y_l\mid x) - \gamma\Big)\Big].
$$

```python
def simpo_loss(pol_chosen_logps, pol_rejected_logps,
               chosen_len, rejected_len, beta=2.0, gamma=1.0):
    # Length-normalized (per-token average) log-prob = the SimPO reward / β.
    r_chosen   = beta * (pol_chosen_logps   / chosen_len)
    r_rejected = beta * (pol_rejected_logps / rejected_len)
    # Require the chosen reward to exceed rejected by at least the margin γ.
    return -torch.nn.functional.logsigmoid(r_chosen - r_rejected - gamma).mean()
```

SimPO removes the reference forward pass *and* the length bias, often matching or beating DPO with less compute. The cost: $\gamma$ and $\beta$ now interact and need tuning, and without the reference anchor you lean entirely on the data and $\gamma$ to prevent degeneration. (SimPO typically uses a larger $\beta$, e.g. 2.0–2.5, because the reward is now an average rather than a sum.)

### CPO — contrastive, memory-light, from machine translation

**Contrastive Preference Optimization (CPO)** (Xu et al., *Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation*) was developed for MT but is general. CPO approximates DPO with a **uniform reference** (equivalently, drops $\pi_{\text{ref}}$) — saving the reference memory and forward pass — and adds a **behavior-cloning (SFT) regularizer** on the chosen response so the policy doesn't drift while it contrasts:

$$
\mathcal{L}_{\text{CPO}} = \underbrace{-\log\sigma\big(\beta\log\pi_\theta(y_w\mid x) - \beta\log\pi_\theta(y_l\mid x)\big)}_{\text{reference-free contrastive term}}
\;\;\underbrace{-\;\lambda\log\pi_\theta(y_w\mid x)}_{\text{SFT anchor}}.
$$

CPO is essentially "DPO without the reference, plus an explicit SFT term to compensate" — the same intuition as ORPO and CPO converging from different directions: *fold the anchor into the loss so you can delete the reference model.*

### A unified view

{{fig:dpo-variant-family-map}}

Almost every method here is a choice of **(a)** what reward to read off the policy, **(b)** what loss shape contrasts winner vs. loser, and **(c)** whether/how to anchor to a reference.

| Method | Reward / score | Loss shape | Reference model? | Paired data? | Killer feature |
|---|---|---|---|---|---|
| **DPO** | $\beta\log\frac{\pi_\theta}{\pi_{\text{ref}}}$ (summed) | log-sigmoid (Bradley–Terry) | yes | yes | exact RLHF optimum, no RL |
| **IPO** | same log-ratio | squared error to $\tfrac{1}{2\beta}$ | yes | yes | finite margin, noise-robust |
| **KTO** | $\beta\log\frac{\pi_\theta}{\pi_{\text{ref}}}$ vs. $z_0$ | prospect-theory value | yes | **no** | unpaired thumbs up/down |
| **ORPO** | log-odds ratio | SFT NLL + log-sigmoid(OR) | **no** | yes | single-stage, fused with SFT |
| **SimPO** | $\frac{\beta}{|y|}\log\pi_\theta$ (length-norm) | log-sigmoid with margin $\gamma$ | **no** | yes | reference-free, no length bias |
| **CPO** | $\beta\log\pi_\theta$ (summed) | log-sigmoid + SFT anchor | **no** | yes | memory-light contrastive |

The practitioner's decision tree: *paired data + can afford a reference* → start with **DPO** ($\beta=0.1$). *Noisy or near-deterministic labels* → **IPO**. *Only unpaired ratings* → **KTO**. *Want one stage / no reference model / tight memory* → **ORPO** or **SimPO**. There is no universal winner; results are dataset- and base-model-dependent, and you should treat the choice as a hyperparameter to sweep.

## The DPO-vs-PPO debate

This is the single most likely "deep" alignment question in an interview, so let's make it crisp.

{{fig:dpo-vs-ppo}}

### Why DPO is attractive

DPO is **offline, stable, and cheap.** No reward model to train, no online generation, no critic, no reward-hacking feedback loop, no sampling hyperparameters. It is a supervised loss — it trains like SFT. For most teams, that operational simplicity is decisive, and DPO + good preference data gets you most of the way to a well-aligned chat model. Open recipes like Zephyr and many Llama/Mistral fine-tunes used DPO precisely because it is so much easier to get right than PPO.

### Why PPO can still win

The catch is that DPO is **off-policy and offline.** It learns only from the fixed preference pairs in your dataset — responses generated by *some other* policy. It never sees its own current outputs. This has real consequences:

- **Distribution shift / out-of-distribution behavior.** DPO's loss only constrains behavior on $(y_w,y_l)$ pairs in the data. The mechanism that *lowers* $\pi_\theta(y_l)$ can, as a side effect, *raise* probability mass on completely unseen responses — including bad ones not represented in the data. Several analyses (notably *Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study*, Xu et al.) show DPO can find policies that exploit out-of-distribution regions, and that on the hardest benchmarks well-tuned **PPO outperforms DPO**.
- **No exploration.** PPO generates fresh samples and scores them with a reward model, so it can discover and reinforce *new* high-reward behaviors. DPO can only re-rank what's already in the dataset. For tasks where the best behavior isn't well-represented in the preference data — competitive coding, math — online RL's exploration matters.
- **A reward model generalizes; a dataset doesn't.** PPO's reward model can score *any* generated response, giving dense signal across the whole output space. DPO's "reward" is implicit and only ever evaluated on the dataset pairs.

### The honest synthesis

The community's rough 2024–2025 consensus: **DPO is the better default** — simpler, cheaper, stable, and good enough for most alignment — while **well-tuned online RL (PPO, and increasingly GRPO) achieves the highest ceiling**, especially on hard, verifiable tasks. The gap narrows or vanishes if you make DPO *more online*: **iterative / online DPO** regenerates fresh pairs from the current policy (labeled by a reward model or judge) and re-runs DPO each round, recovering much of PPO's exploration benefit while keeping DPO's simple loss. This is why the frontier moved toward online and verifiable-reward RL ([GRPO, RLOO & Critic-Free RL](../05-posttraining-alignment/08-grpo-rloo.html), [RL with Verifiable Rewards (RLVR) & The Reasoning Recipe](../05-posttraining-alignment/09-rlvr-reasoning.html)) rather than abandoning either family.

!!! interview "Interview Corner"

    **Q:** Derive why DPO doesn't need a reward model. Where exactly does the reward model "go," and what assumption makes it disappear?

    **A:** Start from the KL-regularized RLHF objective. For a fixed reward $r$ it has a closed-form optimum $\pi^*(y\mid x)\propto \pi_{\text{ref}}(y\mid x)\exp(\tfrac1\beta r(x,y))$. Invert it to express the reward in terms of the policy: $r(x,y)=\beta\log\frac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)}+\beta\log Z(x)$. So the *implicit reward* is just $\beta$ times the policy-to-reference log-ratio. Now plug this into the **Bradley–Terry** preference model, which depends only on the reward *difference* $r(x,y_w)-r(x,y_l)$. The intractable partition term $\beta\log Z(x)$ depends only on $x$, so it is identical for winner and loser and **cancels in the difference**. That cancellation — enabled by the Bradley–Terry "only differences matter" assumption plus the closed-form optimal policy — is what lets us train the policy directly on preferences with a simple log-sigmoid loss, no reward model and no RL. The reward model didn't vanish; it was *re-parameterized into the policy itself*.

!!! interview "Interview Corner"

    **Q:** Your DPO run shows reward accuracy climbing nicely, but the chosen *and* rejected log-probabilities are *both* falling, and the model's outputs are getting worse / more repetitive. What's going on?

    **A:** This is the well-known DPO failure where the *margin* improves while *both* absolute log-probs decline — DPO only constrains the *difference* $\Delta$, so it can satisfy the loss by pushing the rejected log-prob down faster than the chosen, dragging the chosen down too. The policy is moving probability mass off *both* in-distribution responses and onto unseen (often degenerate) outputs, which the loss never penalizes. Fixes: (1) **raise $\beta$** to tighten the KL leash to $\pi_{\text{ref}}$; (2) add an **SFT / NLL term on the chosen** response (this is exactly what CPO and ORPO bake in, and what DPO+SFT mixing does) so the chosen log-prob is held up in absolute terms; (3) switch to a loss with an absolute anchor like **KTO** (value relative to a reference point) or **IPO** (finite target margin prevents the runaway); (4) check data quality — if many "rejected" responses are actually fine, you're teaching the model to suppress good behavior. Always log absolute chosen/rejected log-probs, not just reward accuracy.

{{fig:dpo-degeneration-footgun}}

!!! tip "Practitioner tip: precompute reference log-probs and watch your metrics"

    Run the frozen reference over your entire preference dataset *once*, cache the chosen/rejected sequence log-probs to disk, and drop the reference model from the training loop — this halves memory and removes a forward pass per step. During training, the three metrics that actually diagnose DPO health are: **reward accuracy** (should climb past ~0.6–0.7), **reward margin** $r_w-r_l$ (should grow but not explode), and **absolute chosen log-prob** (should *not* crater). If margin grows while chosen log-prob crashes, you're in the degeneration regime above.

## Where this sits in the stack

DPO and its variants are the **offline preference-tuning** layer of post-training. They consume the preference data and reward-modeling concepts from [The RLHF Pipeline & Reward Modeling](../05-posttraining-alignment/05-rlhf-reward-modeling.html), they are the lightweight alternative to [Policy Gradients & PPO for Language Models](../05-posttraining-alignment/06-ppo-for-llms.html), and they sit upstream of the online critic-free methods in [GRPO, RLOO & Critic-Free RL](../05-posttraining-alignment/08-grpo-rloo.html). In practice DPO almost always runs *after* [Supervised Fine-Tuning & Instruction Tuning](../05-posttraining-alignment/01-sft-instruction-tuning.html) (the SFT model is your $\pi_{\text{ref}}$), is frequently combined with [PEFT I: LoRA, QLoRA, DoRA & The Adapter Family](../05-posttraining-alignment/03-peft-lora-qlora.html) to fit on a single GPU, and shares its reward-hacking failure modes with everything in [Reward Hacking, Over-Optimization & Alignment Failures](../05-posttraining-alignment/13-reward-hacking-failures.html). Production implementations of every loss in this chapter live in [TRL: HuggingFace's RL Library](../06-rl-infra/03-trl.html).

!!! key "Key Takeaways"

    - **DPO's central identity:** the KL-regularized RLHF objective has the closed-form optimum $\pi^*\propto\pi_{\text{ref}}\exp(\tfrac1\beta r)$. Inverting it gives the **implicit reward** $r_\theta=\beta\log\frac{\pi_\theta}{\pi_{\text{ref}}}$ — $\beta$ times a log-ratio you compute from two forward passes.
    - **Why the reward model disappears:** plug the implicit reward into the **Bradley–Terry** model; the intractable partition $\beta\log Z(x)$ depends only on the prompt, is the same for winner and loser, and **cancels in the difference**. What's left is $\mathcal{L}=-\log\sigma(\Delta)$, a supervised classification loss.
    - **The gradient is contrastive:** raise $\log\pi_\theta(y_w)$, lower $\log\pi_\theta(y_l)$, weighted by $\sigma(-\Delta)$ — large on mis-ranked pairs, small on solved ones (an automatic curriculum). $\beta\in[0.1,0.5]$ is the KL leash; **0.1** is the default.
    - **DPO's three weaknesses** drive the variant zoo: unbounded margins on deterministic labels, the need for **paired** data, and the need for a **reference** model.
    - **IPO** caps the margin (squared error to $\tfrac{1}{2\beta}$, noise-robust); **KTO** drops pairing (prospect-theory value on unpaired thumbs up/down); **ORPO** drops the reference (SFT + odds-ratio, single stage); **SimPO** drops the reference and the length bias (length-normalized reward + margin $\gamma$); **CPO** is reference-free contrastive + SFT anchor.
    - **A core DPO footgun:** the loss only constrains the *difference*, so the chosen *and* rejected log-probs can both fall (degeneration). Watch **absolute** chosen log-prob, not just reward accuracy; raise $\beta$ or add an SFT/NLL anchor.
    - **DPO vs PPO:** DPO is offline, stable, cheap, and the right default; well-tuned online RL (PPO/GRPO) explores and reaches a higher ceiling on hard, verifiable tasks. **Iterative/online DPO** closes much of the gap by regenerating fresh pairs each round.

!!! sota "State of the Art & Resources (2026)"
    DPO and its variants have become the dominant offline alignment paradigm; most open frontier models (Llama 3, Mistral, Gemma) use DPO or a close descendant as a post-SFT step, while the cutting edge has shifted toward online and iterative variants that close the gap with PPO on hard reasoning benchmarks.

    **Foundational work**

    - [Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (2023)](https://arxiv.org/abs/2305.18290) — the original paper deriving the closed-form RLHF optimum and the log-sigmoid loss.
    - [Azar et al., *A General Theoretical Paradigm to Understand Learning from Human Preferences / IPO* (2023)](https://arxiv.org/abs/2310.12036) — the ΨPO framework and Identity Preference Optimization, which caps DPO's unbounded margin with a squared-error target.

    **Recent advances (2023–2026)**

    - [Ethayarajh et al., *KTO: Model Alignment as Prospect Theoretic Optimization* (2024)](https://arxiv.org/abs/2402.01306) — drops the pairing requirement entirely, training from individual thumbs-up/thumbs-down labels via prospect theory; matches DPO on many benchmarks.
    - [Hong, Lee & Thorne, *ORPO: Monolithic Preference Optimization without Reference Model* (2024)](https://arxiv.org/abs/2403.07691) — fuses SFT and preference alignment into a single stage with an odds-ratio term, eliminating the reference model.
    - [Meng, Xia & Chen, *SimPO: Simple Preference Optimization with a Reference-Free Reward* (2024)](https://arxiv.org/abs/2405.14734) — length-normalized reward plus a target margin; NeurIPS 2024; eliminates the reference and the length bias simultaneously.
    - [Xu et al., *Contrastive Preference Optimization (CPO)* (2024)](https://arxiv.org/abs/2401.08417) — reference-free contrastive loss with an SFT anchor, developed for machine translation; ICML 2024.
    - [Xu et al., *Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study* (2024)](https://arxiv.org/abs/2404.10719) — empirical and theoretical analysis showing well-tuned PPO outperforms DPO on hard tasks; the definitive entry point for the DPO-vs-PPO debate; ICML 2024.
    - [Xiong et al., *Iterative Preference Learning from Human Feedback* (2024)](https://arxiv.org/abs/2312.11456) — formalizes online/iterative DPO as KL-regularized contextual bandit, recovering much of PPO's exploration benefit within the DPO framework.

    **Open-source & tools**

    - [huggingface/trl](https://github.com/huggingface/trl) — the standard production library; implements `DPOTrainer`, `KTOTrainer`, `ORPOTrainer`, `CPOTrainer`, `OnlineDPOTrainer`, `GRPOTrainer`, and more, all integrated with PEFT/LoRA.
    - [princeton-nlp/SimPO](https://github.com/princeton-nlp/SimPO) — official SimPO training code, evaluation scripts, and fine-tuned Mistral/Llama-3 checkpoints on AlpacaEval 2 and Arena-Hard.

    **Go deeper**

    - [HuggingFace TRL DPO Trainer docs](https://huggingface.co/docs/trl/en/dpo_trainer) — covers all loss types (sigmoid, IPO, hinge, SimPO, APO, DiscoPOP…), logging metrics, and PEFT integration in one place.
    - [HuggingFace Blog, *Preference Tuning LLMs with DPO Methods* (2024)](https://huggingface.co/blog/pref-tuning) — empirical comparison of DPO, IPO, and KTO on 7B models with MT-Bench sweeps.

## Further reading

- Rafailov, Sharma, Mitchell, Manning, Ermon, Finn, **Direct Preference Optimization: Your Language Model is Secretly a Reward Model** (2023) — the original DPO derivation and experiments.
- Azar, Guo, Piot, Munos, et al., **A General Theoretical Paradigm to Understand Learning from Human Preferences** (2023) — the $\Psi$PO framework and **IPO**.
- Ethayarajh, Xu, Muennighoff, Jurafsky, Kiela, **KTO: Model Alignment as Prospect Theory Optimization** (2024) — unpaired, prospect-theory alignment.
- Hong, Lee, Thorne, **ORPO: Monolithic Preference Optimization without Reference Model** (2024) — odds-ratio, single-stage, reference-free.
- Meng, Xia, Chen, **SimPO: Simple Preference Optimization with a Reference-Free Reward** (2024) — length-normalized reward with a target margin.
- Xu, Sharaf, Chen, et al., **Contrastive Preference Optimization** (CPO, 2024) — reference-free contrastive with an SFT anchor (developed for machine translation).
- Xu, Fu, Gao, et al., **Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study** (2024) — the empirical core of the DPO-vs-PPO debate.
- Bradley & Terry, **Rank Analysis of Incomplete Block Designs** (1952) — the preference model underlying all of the above.
- HuggingFace **TRL** (`DPOTrainer`, `KTOTrainer`, `ORPOTrainer`, `CPOTrainer`) — production implementations; see [TRL: HuggingFace's RL Library](../06-rl-infra/03-trl.html).

## Exercises

**1.** (Conceptual) DPO never trains, evaluates, or even stores a reward model, yet the chapter insists it is *implicitly* optimizing the same Bradley–Terry objective a reward model would. Explain precisely (a) what the "implicit reward" $r_\theta(x,y)$ is in terms of the policy and reference, (b) why the intractable partition term $\beta\log Z(x)$ never has to be computed, and (c) what would break if the loss compared *absolute* implicit rewards $r_\theta(x,y)$ instead of the *difference* $r_\theta(x,y_w)-r_\theta(x,y_l)$.

??? note "Solution"
    (a) Inverting the closed-form optimum $\pi^*(y\mid x)\propto \pi_{\text{ref}}(y\mid x)\exp(\tfrac1\beta r(x,y))$ gives $r(x,y)=\beta\log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}+\beta\log Z(x)$. Parameterizing the reward through the policy we actually want to train yields the **implicit reward**

    $$
    r_\theta(x,y)=\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}+\beta\log Z(x),
    $$

    i.e. $\beta$ times the policy-to-reference log-ratio, plus a prompt-only constant. The log-ratio is computable from two forward passes (policy and frozen reference); no separate reward network exists.

    (b) The Bradley–Terry model depends only on the reward *difference* $r_\theta(x,y_w)-r_\theta(x,y_l)$. Since $\beta\log Z(x)$ is a sum over *all* responses of a given prompt $x$, it is **identical for the winner and the loser** and cancels exactly in the difference. So the one intractable quantity is precisely the one that disappears — that cancellation is the whole reason DPO works.

    (c) $\beta\log Z(x)$ would no longer cancel, and it is intractable (an astronomically large sum), so an absolute-reward loss would be uncomputable. Even setting that aside, the reference/partition offset is unidentifiable from preference data — preferences only ever pin down *differences* — so any absolute-reward objective is ill-posed. DPO is well-defined exactly because it is a loss on differences.

**2.** (Quantitative) Take $\beta=0.1$ and one preference pair. The policy assigns summed log-probs $\log\pi_\theta(y_w)=-8.0$ and $\log\pi_\theta(y_l)=-7.0$; the reference assigns $\log\pi_{\text{ref}}(y_w)=-9.0$ and $\log\pi_{\text{ref}}(y_l)=-7.2$. Compute (a) the two log-ratios, (b) the implicit rewards $r_w,r_l$ and whether the model currently ranks the pair correctly, (c) the margin $\Delta$ and the DPO loss $-\log\sigma(\Delta)$ in nats, and (d) the gradient weight $\sigma(-\Delta)$. Is the loss above or below the coin-flip value $\log 2$?

??? note "Solution"
    (a) Chosen log-ratio: $-8.0-(-9.0)=+1.0$. Rejected log-ratio: $-7.0-(-7.2)=+0.2$.

    (b) $r_w=\beta\cdot 1.0=0.1$ and $r_l=\beta\cdot 0.2=0.02$. Since $r_w>r_l$, the model **ranks this pair correctly** (chosen implicit reward exceeds rejected).

    (c) $\Delta=\beta(1.0-0.2)=0.1\times0.8=0.08$. Then $\sigma(0.08)=\frac{1}{1+e^{-0.08}}=\frac{1}{1+0.92312}=0.5200$, so the loss is $-\log(0.5200)=0.654$ nats.

    (d) Weight $=\sigma(-\Delta)=\sigma(-0.08)=1-0.5200=0.480$.

    The loss $0.654$ is **below** $\log 2\approx0.693$, consistent with the pair already being (weakly) ranked correctly. The margin is small, so the weight $0.480$ is still large — this pair still contributes a substantial corrective gradient that raises $\log\pi_\theta(y_w)$ and lowers $\log\pi_\theta(y_l)$.

**3.** (Conceptual) During a DPO run your `reward_acc` climbs steadily past 0.7, but generations get shorter and more repetitive, and you notice the *absolute* chosen log-prob $\log\pi_\theta(y_w)$ is falling throughout training. (a) Explain how reward accuracy can improve while the chosen response's own probability drops. (b) Where is the displaced probability mass going, and why does the loss not punish it? (c) Give two fixes drawn from this chapter's variants and say what each one changes.

??? note "Solution"
    (a) The DPO loss constrains only the *difference* $\Delta=\beta[(\log\tfrac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)})-(\log\tfrac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)})]$. The model can grow $\Delta$ (and thus `reward_acc`, which just checks $r_w>r_l$) by pushing the *rejected* log-ratio down faster than the chosen — dragging the chosen log-prob down too, as long as it falls more slowly than the rejected. Accuracy measures ranking, not absolute likelihood, so it looks healthy.

    (b) Probability is conserved, so lowering both $\pi_\theta(y_w)$ and $\pi_\theta(y_l)$ moves mass onto **other, unseen responses** — often degenerate ones (short, repetitive). The loss is evaluated only on the dataset's $(y_w,y_l)$ pairs, so it never sees or penalizes this off-distribution mass. This is the degeneration footgun.

    (c) Any two of: (1) **raise $\beta$** to tighten the KL leash to $\pi_{\text{ref}}$, limiting how far the policy can drift. (2) Add an **SFT/NLL anchor on the chosen response** — exactly what **CPO** and **ORPO** bake in — which directly holds $\log\pi_\theta(y_w)$ up in absolute terms. (3) Switch to **KTO**, whose prospect-theory value scores each response against a reference point $z_0$, giving an absolute anchor rather than a pure difference. (4) Switch to **IPO**, whose finite target margin $\tfrac{1}{2\beta}$ removes the incentive to push the margin (and the rejected log-prob) toward infinity. In every case: always log absolute chosen log-prob, not just reward accuracy.

**4.** (Quantitative) SimPO argues DPO's *summed* log-prob reward is length-biased. Consider two responses to the same prompt that are equally fluent per token — both have an average log-prob of $-1.0$ per token — but response $A$ has length $10$ (summed log-prob $-10$) and response $B$ has length $30$ (summed log-prob $-30$). (a) Using a summed-log-prob reward $r=\beta\sum_t\log\pi_\theta$ with $\beta=0.1$ (take the reference term as identical for both, so it drops out), compute $r_A$ and $r_B$ and say which the summed reward prefers. (b) Using SimPO's length-normalized reward $r=\frac{\beta}{|y|}\log\pi_\theta$ with $\beta=2.0$, compute $r_A$ and $r_B$. (c) In one sentence, state what this shows about a length-blind summed reward.

??? note "Solution"
    (a) $r_A=0.1\times(-10)=-1.0$ and $r_B=0.1\times(-30)=-3.0$. The summed reward strongly **prefers the shorter response $A$** ($r_A>r_B$ by $2.0$), even though the two are equally fluent per token — the preference is entirely an artifact of $B$ having more tokens to sum over.

    (b) SimPO first divides by length: mean log-prob is $-10/10=-1.0$ for $A$ and $-30/30=-1.0$ for $B$. So $r_A=2.0\times(-1.0)=-2.0$ and $r_B=2.0\times(-1.0)=-2.0$. The two rewards are **equal** — SimPO is length-invariant here.

    (c) A summed-log-prob reward conflates *quality* with *length*: because every extra token adds a negative term, longer sequences look worse (and shorter ones better) independent of per-token quality, which is the length bias SimPO removes by normalizing — and which the chapter cites as a driver of DPO's tendency to distort response length.

**5.** (Implementation) The chapter gives the CPO loss as a formula but no code. CPO is "reference-free DPO plus an SFT anchor":

    $$
    \mathcal{L}_{\text{CPO}} = -\log\sigma\big(\beta\log\pi_\theta(y_w\mid x) - \beta\log\pi_\theta(y_l\mid x)\big)\;-\;\lambda\log\pi_\theta(y_w\mid x).
    $$

    Implement `cpo_loss` in the chapter's style. It takes the policy sequence log-probs for the chosen and rejected responses (outputs of `sequence_logprob`) and the chosen SFT negative log-likelihood `chosen_nll` (as `orpo_loss` does), plus `beta` and `lam`. Return the mean loss. Note that no reference log-probs appear anywhere — that is the point.

??? note "Solution"
    Two observations. First, the contrastive term is exactly the DPO loss with the reference dropped, so it is $-\log\sigma(\beta(\text{chosen}-\text{rejected}))$ on the *raw* policy log-probs. Second, the SFT anchor $-\lambda\log\pi_\theta(y_w\mid x)$ equals $\lambda$ times the chosen negative log-likelihood, so we reuse `chosen_nll` just like `orpo_loss`.

    ```python
    import torch
    import torch.nn.functional as F

    def cpo_loss(pol_chosen_logps, pol_rejected_logps, chosen_nll,
                 beta=0.1, lam=1.0):
        """
        pol_chosen_logps / pol_rejected_logps : (B,) SEQUENCE log-probs (sum over
            tokens) of chosen/rejected under the policy -- output of sequence_logprob.
            NOTE: no reference log-probs -- CPO is reference-free.
        chosen_nll : (B,) standard SFT negative log-likelihood on the chosen response
            (= -pol_chosen_logps if you score the full chosen sequence).
        """
        # Reference-free contrastive term: DPO's log-sigmoid on RAW policy log-probs.
        logits = beta * (pol_chosen_logps - pol_rejected_logps)      # (B,)
        contrastive = -F.logsigmoid(logits)                         # -log sigma(.)

        # SFT anchor -lambda * log pi(yw) == lambda * chosen_nll, holds chosen up.
        return (contrastive + lam * chosen_nll).mean()
    ```

    Sanity checks consistent with the chapter: (1) dropping the reference means CPO saves the reference model's memory and its forward pass, matching the "memory-light" claim in the variants table. (2) The SFT anchor is the same absolute-log-prob anchor recommended in Exercise 3 as a degeneration fix, so CPO cannot satisfy its loss by cratering the chosen log-prob — the $\lambda\,\text{chosen\_nll}$ term directly penalizes that. (3) With $\lambda=0$ this reduces to reference-free DPO on summed log-probs.
