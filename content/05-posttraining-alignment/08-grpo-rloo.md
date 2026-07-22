# 5.8 GRPO, RLOO & Critic-Free RL

In [Policy Gradients & PPO for Language Models](../05-posttraining-alignment/06-ppo-for-llms.html) we built the canonical RLHF optimizer: a policy network, a frozen reference model for the KL penalty, a reward model, **and** a separate value network — the *critic* — trained to predict the expected future reward of a partial generation so we can compute low-variance advantages via Generalized Advantage Estimation (GAE). That critic is expensive. It is a second transformer, often the same size as the policy, with its own optimizer state, its own forward and backward passes, and its own failure modes. On a 7B policy, the critic roughly *doubles* the trainable-parameter memory footprint and adds a whole training subsystem that can silently diverge.

This chapter is about a simple, radical question that the field answered emphatically in 2024–2025: **what if we just delete the critic?** It turns out that for the LLM setting — where we generate a *complete* response, score it once at the end, and care about the whole trajectory — we can estimate the advantage we need using nothing but a small *group* of sampled responses to the same prompt. The group itself becomes the baseline. No value network, no GAE, no bootstrapping. This idea, in two closely related forms — **RLOO** (REINFORCE Leave-One-Out) and **GRPO** (Group Relative Policy Optimization) — is what drove DeepSeek-R1 and most of the open reasoning-model wave. By the end you will be able to derive both, implement GRPO from scratch, reason about its now-famous bugs, and apply the 2025 fixes (Dr. GRPO, token-level loss, clip-higher).

This chapter pairs tightly with [RL with Verifiable Rewards (RLVR) & The Reasoning Recipe](../05-posttraining-alignment/09-rlvr-reasoning.html) (where the reward becomes a binary correctness checker) and with [Advantage Estimation, KL Control & Stability Tricks](../06-rl-infra/09-advantage-kl-tricks.html) (the infrastructure-level view).

## Why the critic is optional: the baseline as variance reduction

### The REINFORCE skeleton

Strip RLHF down to its core. The policy $\pi_\theta$ is an autoregressive language model; a "trajectory" is a sampled response $o$ to a prompt $q$. We observe a single scalar reward $R(q,o)$ at the end (from a reward model, a verifier, or a rule). We want to maximize expected reward,

$$
J(\theta) = \mathbb{E}_{q\sim\mathcal{D},\, o\sim\pi_\theta(\cdot\mid q)}\big[R(q,o)\big].
$$

The REINFORCE / score-function estimator (see [Policy Gradients & PPO for Language Models](../05-posttraining-alignment/06-ppo-for-llms.html)) gives the gradient

$$
\nabla_\theta J(\theta) = \mathbb{E}\Big[ R(q,o)\,\nabla_\theta \log \pi_\theta(o\mid q) \Big],
\qquad
\log \pi_\theta(o\mid q) = \sum_{t=1}^{|o|}\log \pi_\theta(o_t \mid q, o_{<t}).
$$

This is an unbiased estimator, but its variance is brutal. If every reward is positive (say rewards live in $[0,5]$), then *every* sampled response gets pushed up — good ones harder, bad ones less hard, but all in the "more likely" direction. The signal we actually care about — *which responses were better than typical* — is buried in a large common-mode offset.

### Baselines: subtract anything that doesn't depend on the action

The classic fix is a **baseline** $b(q)$ subtracted from the reward:

$$
\nabla_\theta J(\theta) = \mathbb{E}\Big[ \big(R(q,o) - b(q)\big)\,\nabla_\theta \log \pi_\theta(o\mid q) \Big].
$$

This remains unbiased for *any* $b$ that does not depend on the sampled action $o$, because $\mathbb{E}_{o}\big[b(q)\,\nabla_\theta \log \pi_\theta(o\mid q)\big] = b(q)\,\nabla_\theta \underbrace{\textstyle\sum_o \pi_\theta(o\mid q)}_{=1} = 0$. The variance, however, changes a lot. The variance-minimizing baseline is (approximately) the expected reward $\mathbb{E}_o[R(q,o)]$ — exactly what PPO's *value function* tries to learn. The quantity $A = R - b$ is the **advantage**: how much better this response was than the baseline.

The critic-free insight is that we do not need a *learned* estimate of $\mathbb{E}_o[R(q,o)]$. We can get a **Monte Carlo** estimate of it for free by sampling several responses to the same prompt and averaging their rewards. The group of samples is the baseline. That is the whole trick — everything below is engineering around it.

{{fig:grpo-critic-free-vs-ppo}}

### Why this is *especially* natural for LLMs

In classic RL (robotics, Atari) the reward is dense and per-step, the horizon is long, and bootstrapping a value function is essential to assign credit across thousands of steps. The LLM-with-verifiable-reward setting is different in three ways that make the critic almost vestigial:

1. **The reward is terminal.** You generate the whole answer, then score it once. There is no intermediate reward to bootstrap from. GAE with $\lambda\to 1$ collapses toward plain Monte Carlo returns anyway.
2. **You can cheaply resample.** Unlike a robot, you can draw $G=8$ or $16$ completions for the same prompt in one batched generation call. A Monte Carlo baseline is therefore essentially free relative to the cost you already pay for generation.
3. **The value function is hard and unstable here.** Training a per-token value head on sparse terminal rewards through a 7B transformer is finicky; it is one of the main sources of RLHF instability. Removing it removes a whole class of bugs.

So we trade a *learned, low-variance, biased-if-wrong* baseline (the critic) for a *Monte Carlo, slightly-higher-variance, unbiased* baseline (the group). For terminal-reward LLM RL, that trade is almost always a win.

## RLOO: REINFORCE with a Leave-One-Out baseline

### The estimator

REINFORCE Leave-One-Out (RLOO; popularized for LLMs by Ahmadian et al., *Back to Basics*, 2024, building on earlier LOO baselines) sharpens the Monte Carlo baseline with a subtle correctness fix. Suppose for prompt $q$ we sample $G$ responses $o_1,\dots,o_G$ i.i.d. from $\pi_\theta$ with rewards $R_1,\dots,R_G$. The naive baseline is the full group mean $\bar R = \frac1G\sum_j R_j$. But $\bar R$ *includes* $R_i$, so the baseline for sample $i$ is correlated with $R_i$ — and a baseline correlated with the action you are evaluating introduces bias. RLOO removes exactly the offending term: the baseline for response $i$ is the mean of the *other* $G-1$ responses:

$$
b_i \;=\; \frac{1}{G-1}\sum_{j\neq i} R_j,
\qquad
A_i \;=\; R_i - b_i \;=\; \frac{G}{G-1}\Big(R_i - \bar R\Big).
$$

The second equality is a tidy identity worth knowing: leave-one-out is just the full-group-centered advantage rescaled by $G/(G-1)$. The RLOO gradient estimate over the group is

$$
\hat g_{\text{RLOO}} \;=\; \frac{1}{G}\sum_{i=1}^{G} \Big(R_i - b_i\Big)\,\nabla_\theta \log \pi_\theta(o_i\mid q).
$$

This is unbiased (each $b_i$ is independent of $o_i$), low-variance for modest $G$, and — crucially — uses **no value network and no clipping**. It is "back to basics": plain policy gradient with a clever, free baseline. RLOO treats the whole response as a single action; the per-token log-probs are simply summed.

### A tiny RLOO implementation

```python
import torch
import torch.nn.functional as F

def rloo_loss(logprobs_per_token, response_mask, rewards, group_size):
    """
    Minimal RLOO loss for one batch of prompts, each with `group_size` responses.

    logprobs_per_token : (B, T) log pi_theta(o_t | q, o_<t) for the SAMPLED tokens.
                         B = num_prompts * group_size, laid out so that every
                         contiguous block of `group_size` rows shares a prompt.
    response_mask      : (B, T) 1.0 for response tokens, 0.0 for prompt/padding.
    rewards            : (B,)   scalar terminal reward per response.
    group_size         : G, number of responses per prompt.
    Returns a scalar loss to call .backward() on.
    """
    B, T = logprobs_per_token.shape
    G = group_size
    n_prompts = B // G

    # Sequence log-prob = sum of per-token log-probs over response tokens.
    seq_logprob = (logprobs_per_token * response_mask).sum(dim=-1)      # (B,)

    # Group rewards: (n_prompts, G)
    r = rewards.view(n_prompts, G)
    group_sum = r.sum(dim=1, keepdim=True)                             # (n_prompts, 1)
    # Leave-one-out baseline: (sum of others) / (G - 1)
    loo_baseline = (group_sum - r) / (G - 1)                           # (n_prompts, G)
    advantage = (r - loo_baseline).view(B)                            # (B,)

    # Policy-gradient loss. We MAXIMIZE advantage * logprob, so MINIMIZE its negative.
    # advantage is detached: it is a weight, not something we differentiate through.
    loss = -(advantage.detach() * seq_logprob).mean()
    return loss
```

Two things to internalize. First, the advantage is `detach()`-ed: gradients flow only through `seq_logprob`, never through the reward. Second, RLOO as written is a *single-step* estimator — it assumes the data was generated by the *current* $\pi_\theta$. If you reuse the same rollouts for several gradient steps (which you do for efficiency), the estimator becomes off-policy and you need importance weighting and clipping. That is exactly the bridge to GRPO.

!!! note "Aside: RLOO vs. GRPO in one sentence"
    RLOO and GRPO use the *same group-baseline idea*; GRPO additionally (a) normalizes the advantage by the group's reward standard deviation, (b) wraps it in a PPO-style clipped surrogate so you can take multiple epochs over the same rollouts, and (c) historically applied the loss per token. The 2025 "fixes" mostly walk GRPO *back toward* RLOO's simpler choices.

## GRPO: Group Relative Policy Optimization

{{fig:grpo-advantage}}

GRPO (introduced by Shao et al. in *DeepSeekMath*, 2024, and made famous by *DeepSeek-R1*, 2025) is the workhorse of modern reasoning-model training. It keeps PPO's machinery you actually want — the clipped surrogate objective that makes multi-epoch updates safe — and throws away the part you don't — the value network. The baseline becomes the standardized group reward.

### The advantage

For a prompt $q$, sample a group of $G$ outputs $\{o_1,\dots,o_G\}$ from the *behavior* policy $\pi_{\theta_{\text{old}}}$ (the weights used to generate). Compute rewards $\{R_1,\dots,R_G\}$. The GRPO advantage assigned to *every token* of response $i$ is the group-standardized reward:

$$
\hat A_i \;=\; \frac{R_i - \operatorname{mean}(\{R_1,\dots,R_G\})}{\operatorname{std}(\{R_1,\dots,R_G\}) + \varepsilon}.
$$

Note what this is: a **z-score of the reward within its group**. Subtracting the mean is the baseline (variance reduction, exactly as above). Dividing by the std is a normalization that makes the advantage scale-free, so a prompt where rewards happen to be large and a prompt where they are small contribute comparably. Every token in $o_i$ shares the same scalar $\hat A_i$ — GRPO does no per-token credit assignment, because with a terminal reward there is nothing to assign per token. (We will see in the "fixes" section that the std-normalization is the most controversial design choice in the whole method.)

### The clipped surrogate objective

Let $\pi_\theta$ be the current policy and $\pi_{\theta_{\text{old}}}$ the policy that generated the rollouts. Define the per-token importance ratio

$$
r_{i,t}(\theta) \;=\; \frac{\pi_\theta(o_{i,t}\mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}\mid q, o_{i,<t})}.
$$

The GRPO objective (in its original DeepSeekMath form) is the PPO clipped surrogate, averaged over tokens within a response and then over responses, plus a KL penalty to a reference model:

$$
\mathcal{J}_{\text{GRPO}}(\theta)
= \mathbb{E}\!\left[
\frac{1}{G}\sum_{i=1}^{G}
\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}
\Big(
\min\big( r_{i,t}\hat A_i,\;
\operatorname{clip}(r_{i,t}, 1-\epsilon, 1+\epsilon)\,\hat A_i \big)
- \beta\, \mathbb{D}_{\text{KL}}\!\big[\pi_\theta \,\|\, \pi_{\text{ref}}\big]_{i,t}
\Big)
\right].
$$

We *maximize* $\mathcal J$, equivalently minimize $-\mathcal J$. Three pieces:

- **The clipped term** $\min(r\hat A,\ \operatorname{clip}(r,1-\epsilon,1+\epsilon)\hat A)$ is the PPO trust region (see [Policy Gradients & PPO for Language Models](../05-posttraining-alignment/06-ppo-for-llms.html)). When the new policy has moved too far on a token (ratio outside $[1-\epsilon,1+\epsilon]$) *in a direction that would keep increasing the surrogate*, the gradient is clipped off. This is what makes it safe to take several gradient epochs on one batch of rollouts. Typical $\epsilon \approx 0.2$.
- **The KL penalty** $-\beta\,\mathbb{D}_{\text{KL}}[\pi_\theta\|\pi_{\text{ref}}]$ keeps the policy from drifting too far from the SFT model $\pi_{\text{ref}}$, preserving fluency and preventing reward hacking ([Reward Hacking, Over-Optimization & Alignment Failures](../05-posttraining-alignment/13-reward-hacking-failures.html)). Note that GRPO puts the KL as a *penalty term in the loss*, not folded into the reward as PPO often does.
- **The double average** — mean over tokens *within* a response ($\frac{1}{|o_i|}\sum_t$), then mean over responses ($\frac1G\sum_i$). This per-response length normalization is, as we'll see, the source of GRPO's length bias.

### The k3 KL estimator

GRPO does not compute the exact KL (you would need the full vocabulary distribution from the reference at every position, which is expensive to log). It uses the **k3** unbiased, always-positive, low-variance estimator from John Schulman's well-known note on approximating KL. For a token where $\rho = \dfrac{\pi_{\text{ref}}(o_t\mid\cdot)}{\pi_\theta(o_t\mid\cdot)}$,

$$
\mathbb{D}_{\text{KL}}[\pi_\theta\|\pi_{\text{ref}}]_t \;\approx\; \rho - \log \rho - 1.
$$

This is computed only on the sampled tokens (so just one log-prob from each of the policy and reference). It is non-negative for all $\rho>0$ (since $x-\log x-1\ge 0$), which is a nice property for a penalty.

```python
def k3_kl(logprob_policy, logprob_ref):
    """
    Schulman's k3 estimator of KL(pi_theta || pi_ref), per token, always >= 0.
    Inputs are log pi(o_t | ...) for the SAMPLED token o_t under each model.
    """
    log_ratio = logprob_ref - logprob_policy        # log(pi_ref / pi_theta)
    ratio = log_ratio.exp()                         # pi_ref / pi_theta
    return ratio - log_ratio - 1.0                  # rho - log rho - 1
```

## GRPO from scratch: a complete, runnable training step

Below is a self-contained GRPO trainer for a small Hugging Face causal LM. It is deliberately explicit: rollout, reward, advantage, the clipped loss with KL, and a multi-epoch update over the same rollouts. It runs on a single GPU with a tiny model and a toy verifiable reward. This is the code an interviewer might ask you to sketch on a whiteboard, fleshed out to actually run.

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# 0. Setup. Use a tiny instruct model so this runs on a laptop GPU.
# ---------------------------------------------------------------------------
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
policy = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device)
ref    = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device)
ref.eval()
for p in ref.parameters():
    p.requires_grad_(False)
opt = torch.optim.AdamW(policy.parameters(), lr=1e-6)

GROUP_SIZE   = 8       # G: responses sampled per prompt
CLIP_EPS_LOW = 0.2     # lower clip (1 - eps_low)
CLIP_EPS_HIGH= 0.28    # upper clip (1 + eps_high); "clip-higher", see below
KL_BETA      = 0.0     # many R1-style recipes drop KL entirely; set >0 to use it
PPO_EPOCHS   = 2       # gradient epochs over one batch of rollouts
MAX_NEW      = 200

# ---------------------------------------------------------------------------
# 1. A toy verifiable reward: 1.0 if the response contains the right number,
#    plus a small format reward for using <answer>...</answer>. In a real
#    RLVR run this is a math/code/unit-test checker (see chapter 5.9).
# ---------------------------------------------------------------------------
def reward_fn(question, response, gold):
    r = 0.0
    if f"<answer>{gold}</answer>" in response:
        r += 1.0                                   # correctness
    if "<answer>" in response and "</answer>" in response:
        r += 0.2                                   # format bonus
    return r

# ---------------------------------------------------------------------------
# 2. Rollout: sample GROUP_SIZE responses per prompt with the OLD policy.
#    In production this is a vLLM/SGLang server, not model.generate; see
#    chapter 6.2 (the generation-training loop).
# ---------------------------------------------------------------------------
@torch.no_grad()
def rollout(prompts, golds):
    seqs, masks, prompt_lens, rewards = [], [], [], []
    policy.eval()
    for q, gold in zip(prompts, golds):
        chat = [{"role": "user", "content": q}]
        ids = tok.apply_chat_template(chat, add_generation_prompt=True,
                                      return_tensors="pt").to(device)
        plen = ids.shape[1]
        # Sample G completions in one batched call (num_return_sequences=G).
        out = policy.generate(ids, do_sample=True, temperature=1.0, top_p=1.0,
                              max_new_tokens=MAX_NEW, num_return_sequences=GROUP_SIZE,
                              pad_token_id=tok.eos_token_id)
        for g in range(GROUP_SIZE):
            full = out[g]                          # (plen + gen_len,)
            text = tok.decode(full[plen:], skip_special_tokens=True)
            rewards.append(reward_fn(q, text, gold))
            seqs.append(full)
            prompt_lens.append(plen)
    policy.train()
    # Pad to a common length and build a response mask (1 on generated tokens).
    maxlen = max(s.shape[0] for s in seqs)
    B = len(seqs)
    input_ids = torch.full((B, maxlen), tok.eos_token_id, dtype=torch.long, device=device)
    resp_mask = torch.zeros((B, maxlen), dtype=torch.float, device=device)
    for i, (s, plen) in enumerate(zip(seqs, prompt_lens)):
        input_ids[i, :s.shape[0]] = s
        resp_mask[i, plen:s.shape[0]] = 1.0        # 1.0 on generated tokens
        # generate(num_return_sequences=G) RIGHT-PADS early-finishing completions
        # with pad_token_id (== eos here) up to the group's longest sequence.
        # Those trailing pad-eos tokens were NEVER sampled by the policy; leaving
        # them in resp_mask feeds advantage-weighted gradient and KL into
        # positions the policy never chose. Keep exactly ONE eos (the true stop
        # token the policy did emit) and mask everything after it.
        gen = s[plen:]
        eos_hits = (gen == tok.eos_token_id).nonzero(as_tuple=True)[0]
        if eos_hits.numel() > 0:
            first_eos = plen + int(eos_hits[0])
            resp_mask[i, first_eos + 1:] = 0.0     # drop pad-eos after true stop
    rewards = torch.tensor(rewards, dtype=torch.float, device=device)
    return input_ids, resp_mask, rewards

# ---------------------------------------------------------------------------
# 3. Per-token log-prob of the SAMPLED tokens under a given model.
# ---------------------------------------------------------------------------
def token_logprobs(model, input_ids, resp_mask):
    out = model(input_ids).logits[:, :-1, :]       # predict token t+1 from t
    logprobs = F.log_softmax(out.float(), dim=-1)
    targets = input_ids[:, 1:]                     # the actually-sampled next tokens
    tok_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    return tok_lp, resp_mask[:, 1:]                # shift mask to align with targets

# ---------------------------------------------------------------------------
# 4. GRPO advantage: z-score the reward within each group of GROUP_SIZE.
# ---------------------------------------------------------------------------
def grpo_advantage(rewards, group_size, normalize_std=True):
    g = rewards.view(-1, group_size)               # (n_prompts, G)
    adv = g - g.mean(dim=1, keepdim=True)          # subtract group mean (baseline)
    if normalize_std:
        adv = adv / (g.std(dim=1, keepdim=True, correction=0) + 1e-4)  # population std (matches worked example); torch default is SAMPLE std
    return adv.reshape(-1)                          # (B,) one scalar per response

# ---------------------------------------------------------------------------
# 5. The GRPO loss for ONE gradient epoch over a fixed batch of rollouts.
# ---------------------------------------------------------------------------
def grpo_loss(input_ids, resp_mask, advantage, old_lp, ref_lp,
              token_level=True):
    new_lp, mask = token_logprobs(policy, input_ids, resp_mask)   # (B, T-1)
    A = advantage.unsqueeze(1)                     # (B, 1) -> broadcast over tokens

    ratio = (new_lp - old_lp).exp()                # pi_theta / pi_old, per token
    unclipped = ratio * A
    clipped   = torch.clamp(ratio, 1 - CLIP_EPS_LOW, 1 + CLIP_EPS_HIGH) * A
    pg = -torch.min(unclipped, clipped)            # negative surrogate (we minimize)

    if KL_BETA > 0:
        kl = k3_kl(new_lp, ref_lp)                 # per-token KL(theta || ref)
        pg = pg + KL_BETA * kl

    if token_level:
        # Dr.GRPO / token-level: sum over ALL tokens, divide by ALL tokens.
        loss = (pg * mask).sum() / mask.sum().clamp(min=1.0)
    else:
        # Original GRPO: mean over tokens WITHIN a response, then mean over responses.
        per_resp = (pg * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        loss = per_resp.mean()
    return loss

# ---------------------------------------------------------------------------
# 6. One full GRPO outer step: rollout -> advantage -> PPO_EPOCHS updates.
# ---------------------------------------------------------------------------
def grpo_step(prompts, golds):
    input_ids, resp_mask, rewards = rollout(prompts, golds)
    advantage = grpo_advantage(rewards, GROUP_SIZE, normalize_std=True)

    # Cache OLD (behavior) and REF log-probs once; they are fixed for this batch.
    with torch.no_grad():
        old_lp, _ = token_logprobs(policy, input_ids, resp_mask)
        ref_lp, _ = token_logprobs(ref,    input_ids, resp_mask) if KL_BETA > 0 \
                    else (torch.zeros_like(old_lp), None)

    for _ in range(PPO_EPOCHS):
        loss = grpo_loss(input_ids, resp_mask, advantage, old_lp, ref_lp)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
    return rewards.mean().item(), loss.item()

# Example driver (toy data):
# prompts = ["What is 17 + 26? Put the final number in <answer></answer>."]
# golds   = ["43"]
# for it in range(100):
#     mean_r, l = grpo_step(prompts * 4, golds * 4)
#     print(it, "mean_reward", round(mean_r, 3), "loss", round(l, 4))
```

!!! note "Expected behavior: what a healthy toy run looks like"
    - **Trajectory.** On this single-prompt arithmetic toy with `Qwen/Qwen2.5-0.5B-Instruct` and `GROUP_SIZE=8`, `mean_reward` should climb from roughly `0.2`-`0.5` at step 0 (the base instruct model already answers `17+26` some of the time and often emits the tags) to `>1.0` within about `30`-`80` outer steps. It will not sit exactly at the `1.2` ceiling because sampling stays stochastic. Wall-clock is a few minutes on one consumer GPU (24 GB, e.g. RTX 3090/4090); generation dominates the time, not the backward pass.
    - **Healthy diagnostics.** The **fraction of non-degenerate groups** (groups whose `G` rewards are not all equal) should be clearly `>0` in the early steps -- that is the *only* source of gradient. It naturally decays toward `0` as the policy saturates to always-correct, at which point `mean_reward` plateaus near the ceiling (expected, not a bug). Token-level **entropy** should stay positive (the policy keeps exploring).
    - **Failure signatures.** `mean_reward` flat near `0` with all-wrong groups -> reward/parsing broken or task too hard (check the exact `<answer>{gold}</answer>` string match). `mean_reward` stuck mid-range while the non-degenerate-group fraction is already `0` -> dead groups (raise `G`, vary the prompts, add curriculum). Reward rising while decoded samples turn into repetitive gibberish and entropy collapses -> the policy is diverging: lower the learning rate, set `KL_BETA>0`, and confirm the EOS-mask fix in `rollout` is in place.
    - **Beyond the toy.** For a non-trivial signal, swap the single repeated prompt for a small GSM8K slice and track pass@1 over a few hundred steps rather than one arithmetic fact.

A few engineering notes that matter in practice:

- **`generate` pads with `pad_token_id`, so the response mask must stop at the true EOS.** With `num_return_sequences=G`, completions that finish early are right-padded with `pad_token_id` (here the EOS id) up to the group's longest sequence. The rollout loop above therefore masks everything after the *first* EOS, so the ratio, KL, and advantage-weighted loss are computed only on tokens the policy actually sampled. Forgetting this silently injects advantage-weighted gradient on repeated pad-EOS positions — a classic, hard-to-spot GRPO bug.
- **`old_lp` is recomputed, not reused from generation.** In the toy code we recompute log-probs with a forward pass. In production, generation happens on a separate inference engine (vLLM/SGLang) and you must be careful that the log-probs used for the ratio come from a *consistent* policy. Mismatch between the sampler's numerics and the trainer's numerics is a real, subtle source of bias — see [The Generation–Training Loop & Rollout Engines](../06-rl-infra/02-generation-training-loop.html).
- **KL is often set to zero in R1-style recipes.** DeepSeek-R1-Zero used essentially no KL/penalty and let the model drift far from the base, which is *desired* when you want emergent long reasoning. Keep $\beta>0$ for chat alignment where you must preserve the SFT persona.
- **PPO_EPOCHS > 1 is why we need the ratio and clip at all.** If you only ever take one gradient step on each rollout batch ($\pi_\theta=\pi_{\text{old}}$, ratio $=1$), GRPO collapses into RLOO-with-std-normalization. The clip earns its keep only when you reuse rollouts.

## A worked numerical example

Let's make the magnitudes concrete. Take one prompt, a group of $G=4$ sampled responses, and a verifiable reward in $\{0, 1\}$ (wrong / right), plus the format bonus from our `reward_fn`.

!!! example "Worked example: computing GRPO advantages and a token's loss"
    Four responses are sampled. Their rewards (correctness + format) come out to:

    | response | correct? | format? | reward $R_i$ |
    |---|---|---|---|
    | $o_1$ | yes | yes | $1.2$ |
    | $o_2$ | no  | yes | $0.2$ |
    | $o_3$ | yes | yes | $1.2$ |
    | $o_4$ | no  | no  | $0.0$ |

    **Group mean:** $\bar R = (1.2+0.2+1.2+0.0)/4 = 0.65$.

    **Group std (population):** deviations are $\{0.55, -0.45, 0.55, -0.65\}$; their squares sum to $0.3025+0.2025+0.3025+0.4225 = 1.23$; variance $=1.23/4 = 0.3075$; $\operatorname{std}=\sqrt{0.3075}\approx 0.5545$.

    **GRPO advantages** $\hat A_i = (R_i-\bar R)/(\operatorname{std}+\varepsilon)$, with $\varepsilon=10^{-4}$:

    $$
    \hat A_1 = \hat A_3 = \frac{0.55}{0.5546} \approx +0.992,\quad
    \hat A_2 = \frac{-0.45}{0.5546} \approx -0.811,\quad
    \hat A_4 = \frac{-0.65}{0.5546} \approx -1.172.
    $$

    Every token of $o_1$ and $o_3$ gets advantage $\approx +0.99$ (push up); every token of $o_2$ gets $\approx -0.81$, of $o_4$ gets $\approx -1.17$ (push down). Notice the model is told nothing about *which* tokens were good — it just makes the whole correct trajectory more likely and the whole wrong trajectory less likely. Over many prompts the shared substrings cancel and the genuinely discriminative tokens (the reasoning steps that lead to the right answer) get net-reinforced. This is the "weak credit assignment, strong averaging" character of GRPO.

    **Now one token's contribution.** Take a token in $o_1$ ($\hat A_1=+0.992$). Suppose after a gradient step the policy raised this token's probability so the ratio is $r_{1,t}=1.35$. With $\epsilon_{\text{high}}=0.28$, the clip ceiling is $1.28$. Since $\hat A>0$:

    - unclipped surrogate $= 1.35\times 0.992 = 1.339$
    - clipped surrogate $= 1.28\times 0.992 = 1.270$
    - $\min(1.339, 1.270) = 1.270$ — **clipped**, so this token's gradient is zeroed out (no further push). Good: the trust region kicked in.

    Take instead a token in $o_4$ ($\hat A_4=-1.172$) whose ratio dropped to $r=0.6$ (policy already suppressed it). Lower clip floor is $1-0.2=0.8$. With $\hat A<0$:

    - unclipped $= 0.6\times(-1.172)=-0.703$
    - clipped $= 0.8\times(-1.172)=-0.938$
    - $\min(-0.703,-0.938)=-0.938$ → the **clipped** branch is selected; the surrogate is the more-negative value, so its gradient is also zeroed. (For negative advantage, `min` selects the clipped term whenever the ratio fell below the floor — the trust region prevents over-suppressing already-suppressed tokens.)

    The point of the worked numbers: the advantage magnitudes are $O(1)$, clipping engages exactly when a single step moved a token's probability by more than ~20–28%, and the std-normalization made a $1.2$-vs-$0.0$ reward gap into a clean $\pm 1$ advantage regardless of the raw reward scale.

!!! note "Population vs. sample std: a convention gotcha"
    The worked numbers above use the **population** standard deviation (divide the summed squared deviations by $G=4$), giving $\operatorname{std}\approx 0.5545$ and advantages $+0.992,\,-0.811,\,+0.992,\,-1.172$. PyTorch's `Tensor.std()` **defaults to the Bessel-corrected sample std** (divide by $G-1=3$), which gives $\operatorname{std}\approx 0.6403$ and the slightly smaller advantages $+0.859,\,-0.703,\,+0.859,\,-1.015$. Both conventions appear in production code — TRL's `GRPOTrainer` uses the PyTorch default (sample std) — and either is fine as long as your code and your expected values agree. The `grpo_advantage` code above pins `correction=0` (population std) specifically so it reproduces the numbers in this example; drop that argument if you want to match TRL.

## The DeepSeek-R1 recipe

GRPO is the optimizer; DeepSeek-R1 is the *recipe* that made it iconic. Two variants matter, and the contrast is instructive.

### R1-Zero: pure RL from a base model

R1-Zero starts from the **base** (pretrained, not even SFT'd) DeepSeek-V3 model and applies GRPO directly with a **rule-based verifiable reward** — no reward model, no human preferences. The reward has two parts:

1. **Accuracy reward:** for math, did the boxed final answer match the gold? For code, did it pass the unit tests in a sandbox ([Reward Engineering, Verifiers & Sandboxes](../06-rl-infra/08-reward-verifiers-sandboxes.html))? This is binary and uncheatable in the usual reward-model sense.
2. **Format reward:** did the model put its reasoning between `<think>...</think>` and its answer between `<answer>...</answer>`? A small shaping reward that teaches the *structure*, not the content.

```text
   R1-Zero reward = accuracy_reward(answer, gold)        # 0 or 1 (the real signal)
                  + format_reward(uses <think></think>)  # small, e.g. +0.x
   (no reward model, no learned value model, no human labels)
```

The striking empirical result was that, with nothing but GRPO and this rule reward, the model spontaneously learned to **think longer**: response length grew over training, and behaviors like self-verification and backtracking ("wait, let me reconsider…") *emerged* without ever being demonstrated. The famous "aha moment" — the model catching its own error mid-reasoning — appeared from pure RL pressure on correctness. This is the central evidence that long chain-of-thought reasoning ([Reasoning, Chain-of-Thought & Test-Time Compute](../05-posttraining-alignment/10-reasoning-test-time-compute.html)) can be *grown* by RL rather than only *imitated* from data.

R1-Zero had warts: poor readability, language mixing (switching between languages mid-thought), and ugly formatting — precisely because there was no KL anchor to a polished SFT model and the reward only cared about correctness.

### R1: the multi-stage pipeline

To get a deployable model, DeepSeek-R1 wraps the same GRPO core in a multi-stage pipeline (described qualitatively; we do not cite exact figures):

{{fig:grpo-r1-pipeline}}

The takeaways for an engineer: (1) a tiny **cold-start SFT** dramatically stabilizes RL and improves the final model's polish; (2) **rejection sampling + SFT** ("STaR-like" self-distillation) is a cheap, stable way to bank RL gains and broaden coverage; (3) you eventually need a **reward model** again for the non-verifiable parts (helpfulness, safety) — GRPO is reward-source-agnostic, it just needs *some* scalar per response. The DeepSeek team also showed that **distilling** R1's outputs into smaller dense models often beats running GRPO on those small models directly — a result echoed in [Distillation, Model Compression & Knowledge Transfer](../05-posttraining-alignment/12-distillation-compression.html).

!!! tip "Practitioner tip: pass@k as your north star, not pass@1"
    GRPO can only learn from a prompt if the group contains *both* successes and failures — otherwise every advantage is zero and the gradient vanishes. So the right difficulty band for your prompt set is "hard but not impossible at $G$ samples." Track the fraction of prompts with mixed outcomes; if most groups are all-correct or all-wrong, the run is wasting compute. Curriculum (start easy, raise difficulty) and large $G$ both help keep groups informative.

{{fig:grpo-dead-groups}}

## The 2025 fixes: Dr. GRPO, token-level loss, and clip-higher

GRPO as originally written has two now-well-documented **optimization biases** — places where the loss does not faithfully estimate the policy gradient and instead silently rewards or punishes *length*. The 2025 literature (notably Liu et al., *Understanding R1-Zero-like Training* / "Dr. GRPO", and the Qwen team's *DAPO*) diagnosed and fixed them. These are favorite interview topics because they require you to actually look at the loss algebra.

### Bias 1: the response-level length normalization

{{fig:grpo-length-bias}}

Recall the inner term $\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}(\cdot)$. Dividing each response's summed token loss by its own length $|o_i|$ means **each response contributes equally regardless of length**, which sounds fair but isn't, gradient-wise. Consider two responses with the *same* positive advantage. The gradient signal per token is scaled by $1/|o_i|$, so a *long correct* response gets a *smaller per-token* push than a *short correct* one. Conversely, for negative advantage, long *wrong* responses are penalized *less per token* than short wrong ones. The net effect of this asymmetry is a systematic pressure that, combined with std-normalization, **inflates response length** — the model learns that rambling is cheap when wrong and reinforced when right. This is a major driver of the "GRPO models get longer and longer" phenomenon, separate from genuine reasoning gains.

**The fix (Dr. GRPO / token-level loss):** drop the per-response division and instead sum the loss over *all* tokens in the batch and divide by a *constant* (or by the total token count). Every token gets equal weight; length no longer modulates the per-token gradient:

$$
\mathcal{J}_{\text{token-level}}(\theta)
= \frac{1}{\sum_{i}|o_i|}\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}
\min\big( r_{i,t}\hat A_i,\;\operatorname{clip}(r_{i,t},1-\epsilon,1+\epsilon)\hat A_i\big).
$$

In the code above, this is the `token_level=True` branch: `(pg * mask).sum() / mask.sum()`. DAPO frames it as "token-level policy-gradient loss"; Dr. GRPO removes the length term entirely (dividing by a constant max length). Both kill the length bias.

### Bias 2: the standard-deviation normalization

The other half of "Dr. GRPO" (the name puns on "Done Right" / removing the *bias*) targets the $\div\,\operatorname{std}(\{R\})$ in the advantage. Dividing by the group std means **prompts with low reward variance get their advantages amplified**. A prompt where the group rewards are $\{1,1,1,0\}$ (std small) produces *larger-magnitude* advantages than a prompt with $\{1,1,0,0\}$ (std large), even though the latter is arguably more informative. This re-weights prompts by the inverse of their reward spread — an artifact, not a feature. It particularly distorts very easy and very hard prompts (where std is tiny) and interacts badly with the length bias.

**The fix:** simply **don't divide by the std**. Keep the mean-subtraction (the legitimate baseline) and drop the normalization:

$$
\hat A_i^{\text{Dr.GRPO}} = R_i - \operatorname{mean}(\{R_1,\dots,R_G\}).
$$

This is *exactly the group-mean baseline of RLOO* (up to the $G/(G-1)$ factor). In the code, set `normalize_std=False`. The empirical claim from the Dr. GRPO work is that removing both biases yields more *token-efficient* models — they reason just as well with shorter chains, because the optimizer is no longer secretly paying them to be verbose.

### Fix 3: clip-higher (decoupled clipping, from DAPO)

PPO's symmetric clip $[1-\epsilon, 1+\epsilon]$ has a subtle pathology for exploration noted by the DAPO authors. The *upper* clip $1+\epsilon$ caps how much a *low-probability but good* token can be boosted in one update. Tokens that are currently rare (say $\pi_{\text{old}}=0.01$) but turn out to be useful can only be lifted to at most $1.01\times$ their probability per step — so the policy's **entropy collapses**: it doubles down on already-likely tokens and stops exploring. DAPO's fix is **clip-higher**: decouple the lower and upper clip ranges and raise the ceiling,

$$
\operatorname{clip}\big(r_{i,t},\; 1-\epsilon_{\text{low}},\; 1+\epsilon_{\text{high}}\big),\qquad \epsilon_{\text{high}} > \epsilon_{\text{low}}.
$$

For example $\epsilon_{\text{low}}=0.2,\ \epsilon_{\text{high}}=0.28$. This gives more headroom to promote promising rare tokens (better exploration / sustained entropy) while keeping the lower clip tight to avoid catastrophic suppression. In the code, that is `CLIP_EPS_LOW` vs `CLIP_EPS_HIGH`. DAPO pairs this with two more ideas worth knowing:

- **Dynamic sampling:** discard prompts whose entire group is all-correct or all-wrong (zero advantage, zero gradient) and resample, so every gradient step has informative groups. This directly addresses the "mixed-outcome" requirement from the practitioner tip above.
- **Overlong filtering / soft length penalty:** mask the loss (or apply a graded penalty) on responses that hit the generation length cap, so truncated garbage doesn't get treated as a real (and mis-scored) sample.

### Putting the fixes together

| Component | Original GRPO | Dr. GRPO | DAPO |
|---|---|---|---|
| Baseline | group mean | group mean | group mean |
| Std normalization | yes ($\div\operatorname{std}$) | **no** | **no** |
| Loss aggregation | per-response mean, then mean | token-level (÷ constant) | **token-level** (÷ total tokens) |
| Clipping | symmetric $\epsilon$ | symmetric | **clip-higher** ($\epsilon_{\text{low}}\!\neq\!\epsilon_{\text{high}}$) |
| Dynamic sampling | no | no | **yes** |
| KL penalty | yes ($\beta$) | optional | often **dropped** |

The arc is clear: each "fix" removes an artificial scaling from the loss until what remains is, essentially, **a clean token-level REINFORCE with a group-mean baseline and a PPO clip for off-policy safety** — RLOO's spirit with PPO's trust region. If you remember one thing: *the legitimate parts of GRPO are the group-mean baseline and the clipped ratio; the std-normalization and per-response length normalization are the parts that caused trouble.*

!!! warning "Common pitfall: the all-equal-reward dead group"
    If every response in a group gets the same reward (all correct, all wrong, or a degenerate reward function), then `mean` equals every $R_i$, every advantage is $0$, and that group contributes *exactly zero gradient*. With std-normalization you additionally divide $0$ by a near-zero std — the $\varepsilon$ in the denominator saves you from NaNs, but the group is still dead. This is not a bug to fix in the loss; it is a signal that your prompts are mis-calibrated in difficulty (or your reward is too coarse). Monitor the fraction of non-degenerate groups as a first-class training metric, and use dynamic sampling to refill the batch.

## Length, format, and reward shaping

GRPO's reward can be anything that returns a scalar per response. In practice for reasoning models it is a *sum of shaped components*, and getting the shaping right is most of the art.

- **Correctness (the real signal).** Binary or graded, from a verifier: exact-match on a boxed answer, a math equivalence checker (e.g., symbolic comparison so $\frac12$ and $0.5$ both count), or unit-test pass-rate for code. This is the only component you truly trust; everything else is a guardrail. Covered in depth in [RL with Verifiable Rewards (RLVR) & The Reasoning Recipe](../05-posttraining-alignment/09-rlvr-reasoning.html).
- **Format reward.** A small bonus for emitting the required structure (`<think>`/`<answer>` tags, a `\boxed{}`). Keep it *small* relative to correctness, or the model learns to nail the format on wrong answers — a cheap local optimum. A common pattern: format reward only *if* the answer is also parseable, so it cannot be farmed independently of attempting the task.
- **Length control.** Because uncorrected GRPO inflates length, you may add an explicit length term: a mild penalty above a target budget, or DAPO's overlong-soft-punishment that ramps a penalty as you approach the max-token cap. The cleaner solution, per Dr. GRPO, is to *remove the biases that cause length inflation in the first place* rather than counter-shaping against them.
- **Language consistency.** R1 added a reward for keeping the whole response in the target language, trading a small amount of raw accuracy for readability — a deliberate, documented tradeoff.

A subtle but important point: **shaped rewards are where reward hacking enters GRPO.** The model optimizes the *sum*, and any component that is easier to satisfy than "be correct" becomes a target. Format bonuses get farmed; length bonuses produce padding; an LLM-judge helpfulness reward gets sycophancy. The mean-subtraction in the advantage does *not* protect you here — if hacking the format raises *every* response's reward, the baseline rises too and you're fine, but if it raises *some* responses (the ones that stumble into the exploit), those get positive advantage and the exploit is reinforced. Treat every non-correctness reward component as a liability to be minimized. See [Reward Hacking, Over-Optimization & Alignment Failures](../05-posttraining-alignment/13-reward-hacking-failures.html).

## Where GRPO/RLOO sit relative to DPO and PPO

It helps to place these methods on a single map (full PPO treatment in [Policy Gradients & PPO for Language Models](../05-posttraining-alignment/06-ppo-for-llms.html); DPO in [Direct Preference Optimization & Its Variants](../05-posttraining-alignment/07-dpo-and-variants.html)).

| Method | Needs reward model? | Needs value net (critic)? | Online generation? | Baseline / advantage | Multi-epoch safe? |
|---|---|---|---|---|---|
| **DPO** | no (uses preference pairs directly) | no | no (offline) | implicit, closed-form | n/a |
| **PPO** | yes (or a reward fn) | **yes** | yes | learned $V$ + GAE | yes (clip) |
| **RLOO** | reward fn (any scalar) | no | yes | leave-one-out group mean | no (single step) |
| **GRPO** | reward fn (any scalar) | no | yes | group z-score | yes (clip) |

The mental model: **DPO** is the cheapest (offline, no generation) but is limited to pairwise preferences and can't exploit a *programmatic* reward. **PPO** is the most general and lowest-variance but carries the critic. **GRPO/RLOO** hit the sweet spot for *verifiable-reward* reasoning RL: online (so the policy explores its own current behavior), critic-free (cheap), and able to consume any scalar reward — which is exactly what a unit-test runner or math checker provides. That is why the entire 2025 reasoning-model wave runs on GRPO and its descendants rather than PPO or DPO.

!!! interview "Interview Corner"
    **Q:** PPO and GRPO both use the same clipped surrogate objective. What exactly does GRPO remove, why is that valid for LLM RLHF, and what new failure mode does the replacement introduce?

    **A:** GRPO removes the **value network (critic)** and the **GAE** that PPO uses to estimate the advantage. PPO computes $A_t = \delta_t + \gamma\lambda\delta_{t+1}+\dots$ from a learned $V(s)$; GRPO replaces the entire advantage with a **group-relative score**: sample $G$ responses to the same prompt, and set every token's advantage to the standardized reward $\hat A_i=(R_i-\operatorname{mean})/\operatorname{std}$ within the group. This is valid for the LLM setting because the reward is **terminal** (scored once at the end of a full generation), so there's no intermediate reward to bootstrap — the Monte Carlo group mean is a perfectly good, unbiased baseline, and resampling $G$ completions is cheap. We keep PPO's clipped ratio only so we can take multiple gradient epochs on the same rollouts. The new failure modes are **optimization biases in the loss**: the per-response $1/|o_i|$ length normalization and the $\div\operatorname{std}$ normalization both secretly reweight the gradient and inflate response length and over-weight low-variance prompts — which is exactly what Dr. GRPO/DAPO fix by going token-level, dropping the std, and using clip-higher. A strong answer also notes the **dead-group problem**: if all $G$ rewards are equal the advantage is zero and the group contributes no gradient, so prompt difficulty must be tuned so groups have mixed outcomes.

!!! interview "Interview Corner"
    **Q:** Your GRPO run's reward is climbing but average response length is exploding and eval accuracy is flat. What's happening and what knobs do you turn?

    **A:** Classic GRPO length inflation. The likely culprits, in order: (1) **per-response length normalization** in the loss — switch to a **token-level** loss (sum over all tokens, divide by total token count or a constant) so long responses don't get a smaller per-token gradient; (2) **std-normalization** of the advantage amplifying low-variance prompts — drop the $\div\operatorname{std}$ and keep just the group-mean baseline (Dr. GRPO); (3) a **format or length reward** being farmed — make the format bonus contingent on a correctness attempt and small relative to accuracy; (4) **no length cap handling** — add overlong filtering so truncated samples aren't mis-scored. If exploration also looks dead (entropy collapsing), add **clip-higher** ($\epsilon_{\text{high}}>\epsilon_{\text{low}}$). The fact that accuracy is flat while reward rises is the tell-tale sign that the model is optimizing the *artifacts* of the loss rather than the *correctness* signal.

!!! key "Key Takeaways"
    - **The critic is optional for terminal-reward LLM RL.** Because we score a full response once, a Monte Carlo baseline from a *group* of sampled responses replaces PPO's learned value network — no second transformer, no GAE.
    - **RLOO** = plain REINFORCE with a **leave-one-out** group-mean baseline ($b_i=\frac{1}{G-1}\sum_{j\ne i}R_j$); unbiased, simple, single-step, no clipping.
    - **GRPO** = group-relative advantage $\hat A_i=(R_i-\operatorname{mean})/\operatorname{std}$ assigned to *every token*, optimized with PPO's **clipped surrogate** so you can take multiple epochs on one batch of rollouts. KL to a reference is an optional penalty (often dropped in R1-style runs).
    - **DeepSeek-R1-Zero** showed that GRPO + a pure **rule-based reward** (accuracy + format, no reward model) on a *base* model spontaneously grows long chain-of-thought, self-verification, and "aha" backtracking. **R1** wraps this in cold-start SFT → reasoning RL → rejection-sampling SFT → final RL.
    - **GRPO has two real biases:** per-response **length normalization** ($1/|o_i|$) and **std-normalization** ($\div\operatorname{std}$). Both silently inflate length and reweight prompts. **Dr. GRPO** removes both (token-level loss, mean-only advantage); **DAPO** adds **clip-higher**, dynamic sampling, and overlong filtering.
    - **A group is dead if all its rewards are equal** (advantage $=0$, zero gradient). Tune prompt difficulty and use dynamic sampling so groups have mixed pass/fail outcomes.
    - **Reward shaping is the attack surface.** Keep correctness dominant; make format/length bonuses small and contingent; treat every non-correctness component as a reward-hacking liability.
    - **Mental model:** the well-behaved core of GRPO is *token-level REINFORCE with a group-mean baseline plus a PPO clip* — i.e., RLOO's spirit with a trust region for off-policy reuse.

!!! sota "State of the Art & Resources (2026)"
    Critic-free group-baseline RL (GRPO/RLOO) is now the dominant post-training optimizer for open reasoning models; virtually every 2025–2026 open reasoning model—DeepSeek-R1, QwQ, Skywork-o, and their descendants—trains with GRPO or a direct derivative, and the main active research front is removing the remaining optimization biases (length inflation, std-normalization artifacts) identified by Dr. GRPO and DAPO.

    **Foundational work**

    - [Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (2024)](https://arxiv.org/abs/2402.03300) — introduces GRPO as a critic-free, group-relative PPO variant for LLM math reasoning.
    - [Ahmadian et al., *Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs* (2024)](https://arxiv.org/abs/2402.14740) — establishes RLOO for LLMs; shows it matches or beats PPO at 3× speed and 70% less RAM.
    - [DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* (2025)](https://arxiv.org/abs/2501.12948) — the landmark recipe demonstrating that GRPO + rule-based rewards on a base model spontaneously grows chain-of-thought and "aha" self-correction.

    **Recent advances (2023–2026)**

    - [Liu et al., *Understanding R1-Zero-Like Training: A Critical Perspective* (2025)](https://arxiv.org/abs/2503.20783) — diagnoses GRPO's length-inflation and std-normalization biases; introduces Dr. GRPO (token-level loss, mean-only advantage) to fix both.
    - [Yu et al., *DAPO: An Open-Source LLM Reinforcement Learning System at Scale* (2025)](https://arxiv.org/abs/2503.14476) — adds clip-higher, dynamic sampling, and overlong filtering; achieves 50 pts on AIME 2024 with Qwen2.5-32B.
    - [Hu et al., *REINFORCE++: Stabilizing Critic-Free Policy Optimization with Global Advantage Normalization* (2025)](https://arxiv.org/abs/2501.03262) — global batch-level advantage normalization that improves stability over both GRPO and RLOO without a critic.

    **Open-source & tools**

    - [huggingface/trl](https://github.com/huggingface/trl) — production `GRPOTrainer` and `RLOOTrainer` with full HuggingFace ecosystem integration; the most widely used entry point for GRPO experiments.
    - [verl-project/verl](https://github.com/verl-project/verl) — high-throughput GRPO/RLOO/DAPO/DrGRPO training via HybridFlow; integrates vLLM/SGLang for rollouts and FSDP/Megatron for training.
    - [BytedTsinghua-SIA/DAPO](https://github.com/BytedTsinghua-SIA/DAPO) — official open-source DAPO system built on verl, including training code and curated datasets.

    **Go deeper**

    - [John Schulman, *Approximating KL Divergence* (2020)](http://joschu.net/blog/kl-approx.html) — the canonical note deriving the k1/k2/k3 estimators used for the per-token KL penalty in GRPO.
    - [Cameron Wolfe, *Group Relative Policy Optimization (GRPO)* (2025)](https://cameronrwolfe.substack.com/p/grpo) — accessible deep-dive connecting GRPO theory to DeepSeekMath and DeepSeek-R1 empirics.

## Further reading

- Shao, Wang, Zhu, et al., **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models** (2024) — introduces GRPO.
- DeepSeek-AI, **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning** (2025) — R1-Zero, the multi-stage R1 recipe, distillation results.
- Ahmadian, Cremer, Gallé, et al., **Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs** (2024) — RLOO for LLMs.
- Liu, Chen, et al., **Understanding R1-Zero-Like Training: A Critical Perspective** (2025) — the "Dr. GRPO" analysis of GRPO's length and std biases.
- Yu, et al. (Qwen / ByteDance Seed), **DAPO: An Open-Source LLM Reinforcement Learning System at Scale** (2025) — token-level loss, clip-higher, dynamic sampling, overlong filtering.
- John Schulman, **Approximating KL Divergence** (blog note) — the k1/k2/k3 estimators used for the GRPO KL term.
- Williams, **Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning** (1992) — the original REINFORCE.
- HuggingFace **TRL** (`GRPOTrainer`, `RLOOTrainer`) and **veRL** repositories — production implementations of everything in this chapter; see [TRL: HuggingFace's RL Library](../06-rl-infra/03-trl.html) and [veRL: HybridFlow & The Single-Controller Architecture](../06-rl-infra/04-verl.html).

## Exercises

**1.** (Conceptual) The chapter claims that subtracting a baseline $b(q)$ from the reward leaves the policy-gradient estimator *unbiased for any $b$ that does not depend on the sampled action $o$*, yet RLOO goes to the trouble of excluding $R_i$ from the baseline of sample $i$. If any action-independent baseline is unbiased, why is the naive full-group mean $\bar R = \frac1G\sum_j R_j$ a *biased* baseline for sample $i$, while the leave-one-out mean $b_i=\frac1{G-1}\sum_{j\ne i}R_j$ is not? State the exact property the baseline must have and explain which one has it.

??? note "Solution"
    The required property is that the baseline used for sample $i$ must be **statistically independent of the sampled action $o_i$** (equivalently, it must not depend on $o_i$). The unbiasedness proof relies on this: for a baseline $b$ that does not depend on $o_i$,

    $$
    \mathbb{E}_{o_i}\big[b\,\nabla_\theta \log\pi_\theta(o_i\mid q)\big]
    = b\,\nabla_\theta\!\!\sum_{o_i}\pi_\theta(o_i\mid q)
    = b\,\nabla_\theta 1 = 0,
    $$

    so subtracting $b$ adds zero in expectation. The step $\mathbb{E}_{o_i}[b\,\nabla\log\pi]=b\,\mathbb{E}_{o_i}[\nabla\log\pi]$ is only valid when $b$ can be pulled out of the expectation over $o_i$ — i.e. when $b$ is not a function of $o_i$.

    The full-group mean $\bar R = \frac1G\big(R_i + \sum_{j\ne i}R_j\big)$ **contains $R_i$**, which is a function of $o_i$. It is therefore correlated with the very action it is scoring and cannot be factored out of the expectation, so subtracting it introduces bias. The leave-one-out mean $b_i=\frac1{G-1}\sum_{j\ne i}R_j$ is built only from the *other* $G-1$ i.i.d. responses; it is independent of $o_i$ and preserves unbiasedness. (Note this is about the baseline of a *specific* sample: the global constant $\bar R$ subtracted uniformly is also fine, but RLOO wants the tighter per-sample baseline, and the correct independent version of it is the leave-one-out mean.)

**2.** (Quantitative) For one prompt you sample $G=4$ responses with rewards $R=\{1.2,\;0.2,\;1.2,\;0.0\}$. (a) Compute the leave-one-out baseline $b_i$ and advantage $A_i=R_i-b_i$ for each response *directly* from the definition. (b) Verify each $A_i$ against the identity $A_i=\frac{G}{G-1}(R_i-\bar R)$. (c) What is $\sum_i A_i$, and why does this value hold for RLOO in general?

??? note "Solution"
    Group sum $=1.2+0.2+1.2+0.0=2.6$, so $\bar R = 2.6/4 = 0.65$. With $G-1=3$, $b_i=(2.6-R_i)/3$.

    (a) Direct leave-one-out:

    | $i$ | $R_i$ | $b_i=(2.6-R_i)/3$ | $A_i=R_i-b_i$ |
    |---|---|---|---|
    | 1 | $1.2$ | $1.4/3 = 0.4667$ | $+0.7333$ |
    | 2 | $0.2$ | $2.4/3 = 0.8000$ | $-0.6000$ |
    | 3 | $1.2$ | $1.4/3 = 0.4667$ | $+0.7333$ |
    | 4 | $0.0$ | $2.6/3 = 0.8667$ | $-0.8667$ |

    (b) Identity check with $\frac{G}{G-1}=\frac43$ and deviations $R_i-\bar R=\{0.55,-0.45,0.55,-0.65\}$:

    $$
    \tfrac43(0.55)=0.7333,\quad \tfrac43(-0.45)=-0.6000,\quad \tfrac43(0.55)=0.7333,\quad \tfrac43(-0.65)=-0.8667.
    $$

    Every value matches (a). The leave-one-out advantage is exactly the group-centered reward rescaled by $G/(G-1)$.

    (c) $\sum_i A_i = 0.7333-0.6000+0.7333-0.8667 = 0$. This is exact for RLOO in general: $\sum_i A_i=\frac{G}{G-1}\sum_i(R_i-\bar R)=\frac{G}{G-1}\big(\sum_i R_i - G\bar R\big)=0$, since $\sum_i R_i = G\bar R$ by definition of the mean. The positive pushes on better-than-average responses are exactly balanced by the negative pushes on worse-than-average ones.

**3.** (Quantitative) A prompt yields $G=4$ binary rewards $R=\{1,1,0,0\}$. (a) Compute the GRPO advantages $\hat A_i$ using the **population** std (the convention pinned by `grpo_advantage` in the chapter, `correction=0`), taking $\varepsilon\to 0$. (b) Recompute using PyTorch's default **sample** (Bessel-corrected) std and give the resulting advantages. (c) Take a token in a correct response ($\hat A>0$, use your part-(a) value) whose importance ratio after one update is $r_{i,t}=1.4$, with $\epsilon_{\text{low}}=0.2,\ \epsilon_{\text{high}}=0.28$. Is its surrogate clipped? (d) A different prompt yields $R=\{1,1,1,1\}$. What advantage does every token get, and what does this group contribute to the gradient?

??? note "Solution"
    (a) $\bar R = (1+1+0+0)/4 = 0.5$. Deviations $\{0.5,0.5,-0.5,-0.5\}$; squared sum $=1.0$. Population variance $=1.0/4=0.25$, so population $\operatorname{std}=0.5$. Advantages $\hat A_i=(R_i-0.5)/0.5$:

    $$
    \hat A = \{+1,\;+1,\;-1,\;-1\}.
    $$

    (b) Sample variance divides by $G-1=3$: $1.0/3=0.3333$, so $\operatorname{std}=\sqrt{0.3333}\approx 0.5774$. Advantages $=(R_i-0.5)/0.5774$:

    $$
    \hat A = \{+0.866,\;+0.866,\;-0.866,\;-0.866\}.
    $$

    Same signs and structure, just smaller magnitude — this is exactly the population-vs-sample convention gotcha from the chapter.

    (c) Use $\hat A=+1$. Since $\hat A>0$ the ceiling is $1+\epsilon_{\text{high}}=1.28$. Unclipped surrogate $=r\hat A=1.4\times 1=1.40$; clipped $=\operatorname{clip}(1.4,0.8,1.28)\times 1 = 1.28$. Then $\min(1.40,1.28)=1.28$ — the **clipped** branch wins, so this token is clipped and its gradient is zeroed. The single update moved the token's probability by $+40\%$, beyond the $+28\%$ trust-region ceiling.

    (d) $\bar R = 1$, every deviation is $0$, so every $\hat A_i = 0/(0+\varepsilon)=0$. This is a **dead group**: every token's advantage is zero, so it contributes *exactly zero gradient*. The $\varepsilon$ only prevents a divide-by-zero NaN; it does not resurrect any signal. Such all-equal groups are the motivation for DAPO's dynamic sampling.

**4.** (Conceptual) GRPO's original loss has two "optimization biases" that Dr. GRPO removes. (a) Explain how the per-response factor $\frac1{|o_i|}$ biases the gradient with respect to response *length*, and state the token-level fix. (b) Explain how dividing the advantage by $\operatorname{std}(\{R\})$ re-weights *prompts*, and state the fix. (c) Separately, DAPO's clip-higher raises only the *upper* clip. What failure mode does the symmetric upper clip cause, and why does raising it help?

??? note "Solution"
    (a) **Length-normalization bias.** The inner term $\frac1{|o_i|}\sum_t(\cdot)$ divides a response's summed token loss by its own token count, so each *response* contributes equally but each *token* in a long response gets a $1/|o_i|$-smaller gradient than a token in a short response with the same advantage. For positive advantage this means long correct responses are reinforced *less per token* than short correct ones; for negative advantage, long wrong responses are penalized *less per token* than short wrong ones. That asymmetry creates a standing pressure toward longer responses (rambling is "cheap" when wrong, and under-penalized). **Fix (token-level loss):** stop dividing per response; sum the loss over *all* tokens in the batch and divide by the total token count (or a constant), $\frac{1}{\sum_i|o_i|}\sum_i\sum_t(\cdot)$, so every token carries equal weight and length no longer modulates the per-token gradient. In the chapter code this is the `token_level=True` branch, `(pg*mask).sum()/mask.sum()`.

    (b) **Std-normalization bias.** Dividing by $\operatorname{std}(\{R\})$ makes the advantage magnitude inversely proportional to the group's reward spread, so a low-variance group (e.g. $\{1,1,1,0\}$) gets its advantages *amplified* relative to a high-variance group (e.g. $\{1,1,0,0\}$). This silently re-weights prompts by the inverse of their reward spread — an artifact, worst for very easy/very hard prompts where std is tiny. **Fix:** drop the division and keep only the mean-subtraction, $\hat A_i^{\text{Dr.GRPO}}=R_i-\operatorname{mean}(\{R\})$ (the legitimate baseline, i.e. the RLOO group-mean up to the $G/(G-1)$ factor). In the code, `normalize_std=False`.

    (c) The symmetric upper clip $1+\epsilon$ caps how much a *currently-low-probability but useful* token can be boosted in a single update (a token at $\pi_{\text{old}}=0.01$ can only be lifted to $\le 1.01\times$ its probability). Repeatedly, the policy can never quickly promote promising rare tokens, so it doubles down on already-likely tokens and **entropy collapses** (exploration dies). **Clip-higher** decouples the bounds and raises the ceiling ($\epsilon_{\text{high}}>\epsilon_{\text{low}}$, e.g. $0.28$ vs $0.2$), giving rare-but-good tokens more headroom to grow per step and sustaining entropy, while the tight lower clip still guards against catastrophically over-suppressing tokens.

**5.** (Implementation) DAPO's *dynamic sampling* discards groups whose rewards are all equal (dead groups contribute zero gradient) so every update sees only informative groups. Building on the chapter's code, (a) write a function `dynamic_sampling_mask(rewards, group_size)` that returns a per-*response* boolean mask (`True` for responses whose group is non-degenerate) and the *fraction of non-degenerate groups* (the diagnostic the chapter says to monitor). (b) Show how to use it in `grpo_loss`'s token-level branch so dead-group tokens are excluded from the loss. Keep it consistent with the chapter's tensor conventions.

??? note "Solution"
    A group is non-degenerate iff not all its rewards are equal, i.e. iff its reward range (max minus min) exceeds a small tolerance. We compute one boolean per group, expand it to per-response, and report the group-level fraction.

    ```python
    import torch

    def dynamic_sampling_mask(rewards, group_size, tol=1e-8):
        """
        DAPO dynamic sampling: flag responses in NON-degenerate groups.

        rewards    : (B,) scalar terminal reward per response; every contiguous
                     block of `group_size` rows shares a prompt (chapter layout).
        group_size : G.
        Returns
          keep_resp : (B,) bool, True where the response's group has mixed rewards.
          frac      : float, fraction of groups that are non-degenerate.
        """
        G = group_size
        g = rewards.view(-1, G)                                   # (n_prompts, G)
        # Non-degenerate <=> rewards are not all identical in the group.
        alive = (g.max(dim=1).values - g.min(dim=1).values) > tol  # (n_prompts,) bool
        frac = alive.float().mean().item()
        keep_resp = alive.unsqueeze(1).expand(-1, G).reshape(-1)   # (B,) bool
        return keep_resp, frac
    ```

    Combine this per-response mask with the existing token-level response mask so dead-group tokens drop out of both the numerator and the denominator of the loss. Because the aggregation divides by the surviving token count, discarding dead groups does not shrink the loss magnitude — it just removes zero-advantage tokens (which contribute nothing anyway) and keeps the diagnostic honest:

    ```python
    def grpo_loss_dyn(input_ids, resp_mask, advantage, old_lp, ref_lp,
                      rewards, group_size):
        new_lp, mask = token_logprobs(policy, input_ids, resp_mask)  # (B, T-1)
        A = advantage.unsqueeze(1)                                   # (B, 1)

        ratio = (new_lp - old_lp).exp()
        unclipped = ratio * A
        clipped   = torch.clamp(ratio, 1 - CLIP_EPS_LOW, 1 + CLIP_EPS_HIGH) * A
        pg = -torch.min(unclipped, clipped)
        if KL_BETA > 0:
            pg = pg + KL_BETA * k3_kl(new_lp, ref_lp)

        # Zero out tokens belonging to dead (all-equal-reward) groups.
        keep_resp, frac = dynamic_sampling_mask(rewards, group_size)  # (B,), float
        mask = mask * keep_resp.unsqueeze(1).float()                  # (B, T-1)

        loss = (pg * mask).sum() / mask.sum().clamp(min=1.0)          # token-level
        return loss, frac                                            # frac = diagnostic
    ```

    Notes: (1) the dead-group tokens already had advantage $0$, so masking them changes the loss only through the denominator — the practical payoff of *true* dynamic sampling is that you would **resample fresh prompts** to refill the batch with informative groups rather than merely dropping them; this function is the filter that decision is built on. (2) `frac` is exactly the "fraction of non-degenerate groups" the chapter flags as a first-class training metric; log it every step, since it is the *only* source of gradient and naturally decays toward $0$ as the policy saturates. (3) The range-based test matches the population/sample-std discussion: it needs no std at all and never divides by a near-zero denominator.
