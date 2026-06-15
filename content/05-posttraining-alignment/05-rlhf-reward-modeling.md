# 5.5 The RLHF Pipeline & Reward Modeling

A base language model, freshly pretrained on a trillion tokens of internet text, is a magnificent next-token predictor and a frustrating assistant. Ask it "How do I make a loaf of sourdough?" and it might continue with three more questions in the same style, because forum posts that begin with a question are often followed by *more* questions. It has learned the *distribution of text*, not the *intent to be helpful*. [Supervised Fine-Tuning & Instruction Tuning](../05-posttraining-alignment/01-sft-instruction-tuning.html) goes a long way toward fixing this: show the model tens of thousands of (instruction, good-response) pairs and it learns the *format* of being an assistant. But SFT has a ceiling. It can only imitate demonstrations a human bothered to write, it gives equal weight to every token of every demonstration regardless of quality, and it has no notion that one valid answer might be *better* than another equally valid answer.

The problem is that "good response" is not a thing you can easily write down. You cannot author a demonstration for every possible prompt, and for open-ended prompts ("write me a poem about the ocean") there is no single correct target at all. But humans find it *easy to compare*: shown two ocean poems, a person can reliably say which they prefer, even when they could not have written either. **Reinforcement Learning from Human Feedback (RLHF)** is the machinery that converts that cheap, plentiful comparison signal into a training objective. We collect human *preferences* over model outputs, distill them into a learned **reward model** (RM) that scores any response with a scalar, and then optimize the language model — now the **policy** — to maximize that score while staying close to its SFT starting point.

This chapter is the cornerstone of Part V. We follow the **InstructGPT** recipe (Ouyang et al., 2022) end to end: how preference data is collected, the **Bradley–Terry** statistical model that turns pairwise comparisons into a continuous reward, how to train and evaluate the reward model (and how it gets *hacked*), and the four-model — actor, critic, reward, reference — apparatus that makes the optimization step work. The actual policy-optimization algorithm (PPO) is the subject of [Policy Gradients & PPO for Language Models](../05-posttraining-alignment/06-ppo-for-llms.html); the reward-free shortcut that skips the RM entirely is [Direct Preference Optimization & Its Variants](../05-posttraining-alignment/07-dpo-and-variants.html). Here we build everything *up to and including the reward signal*, plus the scaffolding around the optimizer.

## The three-stage recipe and why it works

InstructGPT (and ChatGPT after it) is built on a three-stage pipeline. Each stage produces an artifact the next stage consumes.

{{fig:rlhf-pipeline}}

{{fig:rlhf-three-stage-pipeline}}

The key conceptual move is **decoupling the source of supervision from the form of supervision.** Humans supply preferences — a comparison signal that is cheap, has low inter-annotator variance, and requires no expertise to author. The reward model turns those discrete comparisons into a *dense, differentiable scalar* defined on the entire space of possible responses, including responses no human has ever seen. The policy then gets to *generate its own training data*: it proposes responses, the RM grades them, and the policy is nudged toward the high-scoring region. This is why RLHF can exceed SFT — the policy explores outputs *better than any demonstration in the dataset*, and the RM, generalizing from preferences, can recognize and reward them.

!!! note "Aside: why not just SFT on the best responses?"

    A natural alternative is **rejection sampling / best-of-N SFT**: sample many responses, keep the one the RM (or a human) likes best, and SFT on it. This works and is widely used (it is "Stage 2" of the LLaMA-2 alignment and a core ingredient in many recipes). RLHF with PPO can be seen as a smoother, online version of the same idea: instead of taking one discrete "keep the best" step, you continuously push probability mass toward higher-reward regions and *away* from lower-reward ones, using the *contrast* between good and bad. The negative signal — learning what *not* to do — is something best-of-N SFT throws away.

## Preference data collection

Everything downstream is only as good as the preference data. This is where most of the human cost and most of the subtle failure modes live.

### What a labeler actually does

The labeling unit is a **comparison**. A prompt $x$ is drawn (from real API traffic, in InstructGPT's case, plus labeler-written prompts). The SFT model generates $K$ candidate responses $y_1, \dots, y_K$ (InstructGPT used $K$ between 4 and 9). A human is shown the prompt and all $K$ responses and produces a **ranking** — a total or partial order over the candidates. From a single ranking of $K$ items you extract $\binom{K}{2}$ pairwise preferences. Ranking $K$ items at once is far more sample-efficient than $\binom{K}{2}$ independent pairwise tasks, *and* it forces internal consistency: a labeler cannot say A>B, B>C, and C>A within one ranking.

Each extracted pair becomes a training row $(x, y_w, y_l)$ where $y_w$ ("win") is the preferred response and $y_l$ ("lose") is the dispreferred one. A modern open preference dataset row looks like:

```json
{
  "prompt": "Explain why the sky is blue to a 10-year-old.",
  "chosen": "Sunlight is actually made of all the colors mixed together...",
  "rejected": "The sky is blue due to Rayleigh scattering, wherein the cross-section...",
  "metadata": {"annotator_id": 71, "confidence": "strong", "k_rank_position": [1, 3]}
}
```

### The dimensions labelers are asked to judge

"Which is better?" is underspecified, so production labeling uses an explicit rubric. InstructGPT's instructions asked labelers to weigh:

- **Helpfulness** — does it follow the instruction and actually solve the user's problem (including inferring unstated intent)?
- **Honesty / truthfulness** — is it factually correct; does it avoid confidently stating falsehoods?
- **Harmlessness** — does it avoid toxic, dangerous, or biased content?

These can conflict: the most helpful answer to "how do I pick a lock" is harmful; the most harmless is unhelpful. InstructGPT told labelers to prioritize helpfulness during *comparison* labeling but harmlessness during a separate *safety* labeling pass — a deliberate decomposition that later work (Anthropic's helpfulness/harmlessness split, then [Constitutional AI, RLAIF & Self-Improvement](../05-posttraining-alignment/11-constitutional-rlaif.html)) made explicit by training *separate* reward models per axis.

!!! warning "Common pitfall: the preference signal is noisy and biased"

    Human preference labels have a measurable agreement rate — labeler-vs-labeler agreement is often only in the 70–75% range on hard prompts, and labeler-vs-researcher agreement similar. That ceiling propagates: a reward model can rarely exceed human agreement, and a policy optimized against it inherits the model's blind spots. Worse, preferences carry **systematic biases**: humans (and therefore RMs) prefer *longer* answers, more *confident* tone, nicely *formatted* markdown, and sycophantic agreement — independent of correctness. Length bias is so strong that "the RM rewards verbosity" is one of the first things to check when an RLHF run produces rambling outputs. Curate for *contrast on the axis you care about*, balance response lengths, and audit for these confounds before training.

### How much data

InstructGPT used on the order of tens of thousands of comparisons. Reward-model data is more leveraged than SFT data because each comparison is a *relative* judgment that constrains the whole reward surface, not a single absolute target. But preference data has a shelf life: once you optimize a policy against an RM, the *new* policy generates responses in a distribution the RM never saw during training. This is the **distribution-shift** problem and it motivates *iterated* RLHF — collect fresh preferences on the new policy's outputs, retrain the RM, optimize again. LLaMA-2 ran roughly five such iterations; this loop is the seed of the [Data Flywheels & Continuous Improvement](../12-production-mlops/05-data-flywheel.html) story.

## The Bradley–Terry model: from comparisons to a scalar

We have pairs $(x, y_w, y_l)$. We want a function $r_\phi(x, y)$ — a single scalar score — such that higher means "more preferred." How do we get a continuous score from discrete comparisons? The bridge is a 70-year-old idea from the statistics of paired comparisons: the **Bradley–Terry model** (Bradley & Terry, 1952), the same math behind Elo chess ratings.

### The model

Assume each response $y$ to a prompt $x$ has a latent real-valued "strength" $r(x,y)$. The Bradley–Terry assumption is that the probability a human prefers $y_w$ over $y_l$ is a **logistic function of the difference in strengths**:

$$
P(y_w \succ y_l \mid x) \;=\; \frac{\exp\big(r(x, y_w)\big)}{\exp\big(r(x, y_w)\big) + \exp\big(r(x, y_l)\big)} \;=\; \sigma\big(r(x, y_w) - r(x, y_l)\big),
$$

where $\sigma(z) = 1/(1 + e^{-z})$ is the logistic sigmoid. This is exactly a softmax over two items, or equivalently logistic regression on the *score gap*. Three properties make it the right choice:

1. **Only differences matter.** Adding a constant $c$ to every score leaves all preference probabilities unchanged ($\sigma((r_w + c) - (r_l + c)) = \sigma(r_w - r_l)$). The reward is identified only up to an additive constant — a fact with real consequences (it is why you must not compare raw reward magnitudes across two separately-trained RMs, and why we mean-center rewards before PPO).
2. **It is calibrated and monotone.** Equal scores give a coin flip ($\sigma(0)=0.5$); a score gap of $+2$ means $\sigma(2)\approx 0.88$, i.e. "preferred 88% of the time." The score *is* a log-odds of preference, so it has an interpretable scale.
3. **It is differentiable**, so we can fit $r$ by gradient descent.

We replace the abstract strength $r$ with a neural network $r_\phi$ and fit $\phi$ by **maximum likelihood** on the observed preferences.

### Deriving the pairwise loss

For one comparison $(x, y_w, y_l)$, the likelihood under Bradley–Terry is $\sigma(r_\phi(x,y_w) - r_\phi(x,y_l))$. The negative log-likelihood — the loss we minimize — is therefore

$$
\mathcal{L}_{\text{RM}}(\phi) \;=\; -\,\mathbb{E}_{(x,\,y_w,\,y_l)\sim\mathcal{D}}\Big[\, \log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big) \Big].
$$

Let $\Delta = r_\phi(x,y_w) - r_\phi(x,y_l)$ be the **margin** the model assigns. Then the per-example loss is $-\log\sigma(\Delta) = \log(1 + e^{-\Delta})$, the **softplus** of $-\Delta$. Note its shape: it is *near zero* when $\Delta$ is large and positive (model confidently correct), grows *linearly* in $-\Delta$ when the model is confidently wrong, and equals $\log 2 \approx 0.693$ at $\Delta = 0$. The gradient with respect to the margin is clean:

$$
\frac{\partial \mathcal{L}}{\partial \Delta} = -\,\sigma(-\Delta) = -\big(1 - \sigma(\Delta)\big).
$$

So the push on the scores is proportional to *how wrong the model currently is*: if it already strongly prefers the winner ($\sigma(\Delta)\to 1$), the gradient vanishes; if it has it backwards, the gradient saturates at magnitude $1$. This self-limiting behavior is exactly the robustness property of logistic losses.

!!! note "Aside: the ranking generalization (Plackett–Luce)"

    When labelers produce a full ranking of $K$ items rather than a single pair, you *can* train on all $\binom{K}{2}$ pairs (InstructGPT does — and crucially treats all pairs from one prompt as **one batch element** to avoid overfitting, see below). The principled multi-item generalization of Bradley–Terry is the **Plackett–Luce** model, whose likelihood for a ranking $y_1 \succ y_2 \succ \dots \succ y_K$ is a product of softmaxes, "choose the best, remove it, choose the next best, …". For most LLM RM training the simple pairwise decomposition is used because it is trivially batchable.

### A subtle but critical training detail

InstructGPT found that if you shuffle all $\binom{K}{2}$ pairs from many prompts together and train on them as independent examples, the RM **overfits**: each completion $y_i$ appears in $K-1$ pairs within an epoch, so a single forward pass's value gets reused many times and the model memorizes specific completions. The fix is to put **all $\binom{K}{2}$ pairs from one prompt into a single forward/backward pass** (one gradient step "sees" each of the $K$ completions exactly once, computing all pairwise terms from those $K$ scores). This is both more compute-efficient (each completion is encoded once, not $K-1$ times) and a strong regularizer. Remember this for the interview.

## Reward model architecture and training

### The architecture: a transformer with a scalar head

A reward model is almost the same network as the policy. You take the SFT model, **remove the unembedding (language-modeling) head**, and bolt on a small linear layer that maps the final hidden state to a single scalar. You read off the scalar at the position of the **last token** of the response (or the EOS token) — that hidden state has attended over the entire prompt-and-response, so it is a reasonable summary on which to base a whole-sequence judgment.

{{fig:rm-scalar-head-architecture}}

Why initialize from the SFT model rather than from scratch or from the base? It already understands language and the task distribution; you are only teaching it a new, low-dimensional output (a ranking). InstructGPT noted that a *smaller* RM than the policy works fine and is far cheaper — they used a 6B RM to align a 175B policy. The RM does not need to be as capable as the policy; it only needs to *recognize* quality, which is easier than *producing* it (judging is easier than generating).

### The from-scratch pairwise loss

Here is the complete, runnable core of reward-model training — the loss every RLHF library implements. This is the central code of the chapter.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardModel(nn.Module):
    """A transformer backbone with a scalar value head on top.

    `backbone` is any model returning per-token hidden states of shape
    (batch, seq_len, d_model) -- e.g. an SFT-initialized decoder with its
    language-modeling head removed.
    """
    def __init__(self, backbone, d_model):
        super().__init__()
        self.backbone = backbone
        # A single linear layer: d_model -> 1 scalar. Initialize small so early
        # rewards are near zero (helps optimization stability downstream).
        self.value_head = nn.Linear(d_model, 1, bias=False)
        nn.init.normal_(self.value_head.weight, std=1.0 / (d_model + 1) ** 0.5)

    def forward(self, input_ids, attention_mask):
        # hidden: (B, T, d_model)
        hidden = self.backbone(input_ids, attention_mask=attention_mask).last_hidden_state
        # Per-token scalar scores: (B, T)
        scores = self.value_head(hidden).squeeze(-1)
        # Reward is read at the LAST non-pad token of each sequence.
        # attention_mask is 1 for real tokens, 0 for padding.
        last_idx = attention_mask.sum(dim=1) - 1            # (B,) index of final real token
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        reward = scores[batch_idx, last_idx]                # (B,) one scalar per sequence
        return reward


def bradley_terry_loss(reward_chosen, reward_rejected, margin=0.0):
    """The Bradley-Terry / pairwise ranking loss.

    reward_chosen, reward_rejected : (B,) scalar reward for the preferred and
                                     dispreferred response in each pair.
    margin : optional fixed margin m. Loss becomes -log sigmoid(r_w - r_l - m),
             which only counts the pair as "solved" once r_w beats r_l by m.
             Useful when you have a graded preference strength (set m larger for
             strongly-preferred pairs). Default 0 recovers plain Bradley-Terry.

    Returns (loss, accuracy) where accuracy is the fraction of pairs the model
    currently orders correctly -- the single most important RM training metric.
    """
    delta = reward_chosen - reward_rejected - margin        # the margin Δ
    # -log σ(Δ) == softplus(-Δ), computed stably:
    loss = F.softplus(-delta).mean()
    # A pair is "correct" iff the chosen response scores higher than the rejected.
    accuracy = (reward_chosen > reward_rejected).float().mean()
    return loss, accuracy


def train_step(rm, optimizer, batch):
    """One optimization step on a batch of preference pairs.

    `batch` provides chosen/rejected token ids and masks, already tokenized as
    [prompt + response]. We run BOTH responses through the SAME reward model and
    compare their scalar scores. This shared-encoder, two-forward-pass structure
    is exactly what TRL's RewardTrainer and DPO-style code do under the hood.
    """
    rm.train()
    # Forward both completions of every pair through the same network.
    r_chosen   = rm(batch["chosen_ids"],   batch["chosen_mask"])     # (B,)
    r_rejected = rm(batch["rejected_ids"], batch["rejected_mask"])   # (B,)

    loss, acc = bradley_terry_loss(r_chosen, r_rejected)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)   # RMs train fast; clip helps
    optimizer.step()
    return {"loss": loss.item(), "pref_acc": acc.item(),
            "reward_margin": (r_chosen - r_rejected).mean().item()}
```

The two non-obvious lines are worth dwelling on. First, `F.softplus(-delta)` is the *numerically stable* way to compute $-\log\sigma(\Delta)$: a naive `-torch.log(torch.sigmoid(delta))` underflows to `inf` when `delta` is very negative, whereas `softplus` is stable everywhere (the same reason we use `F.binary_cross_entropy_with_logits` rather than logits-then-sigmoid-then-BCE; see [Numerical Computing, Floating Point & Precision](../01-foundations/04-numerics-precision.html)). Second, both responses pass through the **same** network — the reward model is a *Siamese* / shared-encoder architecture, and only the *difference* of its two outputs ever enters the loss.

### Batching all-pairs-from-one-prompt

To implement the InstructGPT regularizer — all $\binom{K}{2}$ pairs from a prompt in one step — you encode the $K$ completions once and form all pairwise terms from the $K$ scalars:

```python
def all_pairs_bt_loss(rewards_K, chosen_better_mask):
    """rewards_K : (K,) scalar reward for the K completions of ONE prompt,
                   ordered by the human ranking (index 0 = most preferred).
       Since they are ranked, every pair (i, j) with i < j has y_i preferred.
    Computes the mean BT loss over all C(K,2) pairs, encoding each completion once.
    """
    K = rewards_K.shape[0]
    # Pairwise score differences Δ_ij = r_i - r_j for all i<j.
    diff = rewards_K.unsqueeze(1) - rewards_K.unsqueeze(0)   # (K, K), Δ_ij at [i, j]
    iu = torch.triu_indices(K, K, offset=1)                  # upper triangle: i<j
    deltas = diff[iu[0], iu[1]]                              # (C(K,2),) all preferred-minus-dispreferred
    return F.softplus(-deltas).mean()
```

Each of the $K$ completions is encoded by the transformer exactly once; the $\binom{K}{2}$ comparisons are cheap arithmetic on the resulting scalars. This is both the efficiency win and the anti-overfitting win InstructGPT reported.

!!! example "Worked example: reward magnitudes, loss, and preference probability"

    Suppose for a single comparison the reward model outputs $r_\phi(x, y_w) = 2.3$ for the chosen response and $r_\phi(x, y_l) = 0.8$ for the rejected one. The margin is $\Delta = 2.3 - 0.8 = 1.5$.

    - **Preference probability the model assigns:** $\sigma(1.5) = \frac{1}{1 + e^{-1.5}} = \frac{1}{1 + 0.223} \approx 0.817$. The model thinks the chosen response would be preferred about **82%** of the time — consistent with a genuine but not overwhelming preference.
    - **Per-example loss:** $-\log\sigma(1.5) = \log(1 + e^{-1.5}) = \log(1.223) \approx 0.201$ nats. Low, because the model already orders this pair correctly.
    - **Gradient on the margin:** $-(1 - \sigma(1.5)) = -(1 - 0.817) = -0.183$. To *reduce* the loss, gradient descent pushes $\Delta$ *up* (the negative sign means the loss decreases as $\Delta$ increases), i.e. spreads the two scores further apart — but only gently, since the model is mostly right already.

    Now suppose the model had it **backwards**: $r_\phi(x,y_w) = 0.5$, $r_\phi(x,y_l) = 1.9$, so $\Delta = -1.4$. Then $\sigma(-1.4)\approx 0.198$ (model thinks the *chosen* one wins only 20% of the time — wrong!), the loss is $-\log(0.198) \approx 1.62$ nats (eight times larger), and the margin gradient is $-(1 - 0.198) = -0.80$ — a strong push to flip the ordering. The loss and its gradient both scale with how wrong the model is, exactly as designed.

    Finally, the **additive-constant invariance**: if we shifted *both* rewards by $+100$, every $\Delta$, every loss, and every gradient above is *identical*. The absolute scale of an RM's outputs is meaningless; only gaps are.

### What good RM training looks like

The headline metric is **preference accuracy** on a held-out set — the fraction of pairs the RM orders the same way the human did. For well-curated data this lands somewhere in the high-60s to high-70s percent; remember that the *human–human* agreement ceiling is itself around 70–75%, so an RM scoring near there is essentially at the noise floor of the labels. Other diagnostics:

- **Calibration:** bucket pairs by the RM's predicted preference probability $\sigma(\Delta)$ and check that, e.g., pairs where it predicts 80% are actually preferred ~80% of the time. A miscalibrated RM gives PPO a distorted gradient.
- **Reward distribution:** plot the histogram of scores on a fixed eval set across training. A healthy RM has well-separated chosen/rejected distributions; a collapsing one pushes everything to extremes.
- **Score on a fixed anchor set:** because rewards are only identified up to a constant, track *gaps* between fixed reference responses, not absolute values, to compare checkpoints.

## Reward model evaluation and reward hacking

The reward model is a *proxy*. It is a learned, imperfect stand-in for "what humans actually want," and the moment you optimize a policy against it, you create an adversary — the policy — whose entire job is to find inputs the proxy scores highly. **Goodhart's law** ("when a measure becomes a target, it ceases to be a good measure") is not a footnote here; it is the central operational risk of RLHF. We give this its own deep treatment in [Reward Hacking, Over-Optimization & Alignment Failures](../05-posttraining-alignment/13-reward-hacking-failures.html); here we cover the mechanism and the controls baked into the pipeline.

### What reward hacking looks like

The policy discovers responses that score high under $r_\phi$ but are *not actually better* to a human. Classic patterns:

- **Length exploitation.** The RM learned (from biased data) that longer answers are usually preferred, so the policy learns to pad. You see reward climbing while answers get verbose and repetitive. The fix at the data level is length-balanced preference data; at the optimization level, length-penalty rewards or length-debiasing of the RM.
- **Sycophancy.** The RM rewards agreeing with the user's stated position, so the policy stops pushing back even when the user is wrong.
- **Format farming.** The RM over-weights markdown structure, bullet points, or a confident tone; the policy emits beautifully formatted nonsense.
- **Out-of-distribution exploitation.** The policy finds genuinely degenerate text (gibberish, repeated tokens, special-token soup) that lands in a corner of input space the RM never trained on and happens to score high. This is the most dramatic failure: reward goes up, human quality goes to zero.

### The KL leash and over-optimization

The reason RLHF does not immediately collapse into reward hacking is the **KL penalty** that anchors the policy to the frozen reference (the SFT model). Optimization maximizes a *regularized* objective:

$$
\max_{\pi_\theta} \;\; \mathbb{E}_{x\sim\mathcal{D},\, y\sim\pi_\theta(\cdot\mid x)}\Big[\, r_\phi(x, y) \;-\; \beta\,\mathbb{D}_{\text{KL}}\!\big(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big)\Big].
$$

The KL term penalizes the policy for drifting from the SFT model's distribution. Why does this curb hacking? Reward-hacking outputs are usually *low-probability under the SFT model* (no sane assistant produces token soup), so reaching them requires a large KL move that the penalty makes expensive. The coefficient $\beta$ is the leash length: small $\beta$ lets the policy roam (more reward, more hacking risk); large $\beta$ keeps it near SFT (safer, less gain).

This trade-off has a famous empirical shape. Plot *true* quality (measured by held-out humans or a much larger "gold" RM) against the KL distance from the reference as you optimize. Quality rises, peaks, then **falls** as the policy over-optimizes the proxy — the **reward-model over-optimization** curve characterized by Gao, Schulman & Hilton (2022). They found the gold-reward gain follows a clean functional form in $\sqrt{\mathrm{KL}}$, with proxy reward diverging upward while true reward turns over. The practical takeaway: **the KL budget is a hyperparameter you tune to sit near the peak**, and "reward went up" is *never* sufficient evidence that the model got better.

{{fig:rm-overoptimization-curve}}

### Defenses

The pipeline-level defenses against reward hacking are: (1) the **KL penalty** and **early stopping** on a gold metric; (2) **reward model ensembles** — average several independently trained RMs so the policy must hack all of them at once, which is harder; (3) **iterated RLHF** — once the policy finds an exploit, collect human preferences on those exploited outputs (where humans will rank the garbage *last*), retrain the RM, and the hole closes; (4) **uncertainty-aware rewards** — penalize responses where the RM ensemble disagrees, since high disagreement flags out-of-distribution inputs. We expand all of these in [Reward Hacking, Over-Optimization & Alignment Failures](../05-posttraining-alignment/13-reward-hacking-failures.html).

!!! tip "Practitioner tip: a held-out 'gold' signal is non-negotiable"

    Never tune an RLHF run on RM reward alone — that is the metric being hacked. Hold out a *different* evaluation signal the policy is not optimizing against: a panel of human ratings, a much larger reward model, or an [LLM-as-a-Judge & Automated Evaluation](../11-evaluation/02-llm-as-judge.html) with a different prompt and base model. When proxy reward keeps climbing but your gold signal plateaus or dips, you have found the over-optimization peak — stop or shrink the KL leash.

## The four-model setup

When people say RLHF is "memory-hungry and operationally heavy," this is what they mean. The PPO stage juggles **four** models simultaneously. Understanding what each one is, whether it is trained or frozen, and why it exists is the single most-tested piece of RLHF systems knowledge.

{{fig:rlhf-four-model-setup}}

| # | Model | Role | Trained? | Initialized from | Why it exists |
|---|-------|------|----------|------------------|---------------|
| 1 | **Actor** $\pi_\theta$ | the policy: generates responses; we update its weights | **yes** | SFT model | this *is* the model we are aligning |
| 2 | **Critic** $V_\psi$ | value function: predicts expected future reward of a partial sequence | **yes** | RM (or SFT) | reduces gradient variance via advantage estimation (GAE) |
| 3 | **Reward** $r_\phi$ | scores the full response with one scalar | **no (frozen)** | trained in Stage 2 | supplies the optimization signal |
| 4 | **Reference** $\pi_{\text{ref}}$ | frozen copy of the SFT model | **no (frozen)** | SFT model | the KL anchor that keeps the policy from drifting/hacking |

### The actor and the reference

The **actor** is the policy $\pi_\theta$ — the model you will ship. It is initialized from the SFT model and is the only model whose weights we want to end up with. The **reference** $\pi_{\text{ref}}$ is a *frozen snapshot of that same SFT model*. At each PPO step we need both $\log\pi_\theta(y\mid x)$ (from the actor) and $\log\pi_{\text{ref}}(y\mid x)$ (from the reference) to compute the per-token KL penalty. The reference never updates — that is the whole point; it is the fixed star the policy is leashed to. (Some recipes periodically refresh the reference to the current policy; the default is to freeze it for the entire run.)

### The reward model

The **reward model** $r_\phi$ from Stage 2 is frozen during PPO. It is consulted *once per response*, at the end, to produce the terminal scalar reward. Because the reward is terminal (only the last token "earns" $r_\phi$), the per-token reward fed to the optimizer is the KL penalty at every token *plus* the reward model's scalar at the final token:

$$
R_t = \underbrace{-\beta\big(\log\pi_\theta(y_t\mid x, y_{<t}) - \log\pi_{\text{ref}}(y_t\mid x, y_{<t})\big)}_{\text{per-token KL penalty}} \;+\; \underbrace{r_\phi(x, y)\cdot\mathbb{1}[t = T]}_{\text{terminal reward}}.
$$

That is, every token pays a little KL "rent," and the whole sequence collects the RM's score at the end. This sparse-terminal-reward structure is exactly why a critic is so valuable here — and why critic-free methods (next section) had to find another way.

### The critic

The **critic** $V_\psi$ is the subtle one. It is a value network that, at each token position, predicts the *expected total future reward* of the sequence from that point on. Why do we need it? Policy-gradient methods have enormous variance if you weight each action by the raw return; subtracting a *baseline* — the expected return — dramatically reduces variance without adding bias. The critic *is* that learned baseline, and combined with the rewards it yields the **advantage** $A_t$ (via Generalized Advantage Estimation, GAE) that PPO actually optimizes. The critic is trained jointly with the actor, usually initialized from the reward model (which already knows how to map sequences to scalar values). The full machinery of advantages, GAE, and the clipped PPO objective is the subject of [Policy Gradients & PPO for Language Models](../05-posttraining-alignment/06-ppo-for-llms.html); the infrastructure view of advantage estimation and KL control is [Advantage Estimation, KL Control & Stability Tricks](../06-rl-infra/09-advantage-kl-tricks.html).

### The memory bill

Four models is a lot of GPU. If the policy is a $7$B-parameter model in bf16, then in the worst case:

- **Actor:** trained, so it carries weights + gradients + Adam optimizer states. Roughly $2$ bytes (bf16 weights) $+ 2$ (grad) $+ 8$ (fp32 Adam moments, two states) $\approx 16$ bytes/param $\to \sim 112$ GB before activations.
- **Critic:** also trained; another full set of weights+grads+optimizer states, often similar size $\to$ on the order of another $\sim 100$ GB (a $7$B critic), or less if smaller.
- **Reward:** frozen, inference only $\to \sim 14$ GB (bf16 weights, no grad/optimizer).
- **Reference:** frozen, inference only $\to \sim 14$ GB.

So a "$7$B RLHF run" can have a resident footprint on the order of $250$+ GB *before* activations and the KV cache for generation — which is why RLHF training is dominated by sharding tricks and by the generation/training split covered in [The Anatomy of an RL-for-LLM System](../06-rl-infra/01-anatomy-rl-system.html). Practical economizers: make the **RM and critic smaller** than the policy (InstructGPT's 6B RM for a 175B policy); use **LoRA** so actor and critic share a frozen backbone and the reference is "free" (the policy *minus* its LoRA adapters *is* the reference — see [PEFT I: LoRA, QLoRA, DoRA & The Adapter Family](../05-posttraining-alignment/03-peft-lora-qlora.html)); or **fold the value head onto the actor** so actor and critic share a trunk.

!!! note "Aside: how the four-model count shrinks"

    A through-line of Part V and Part VI is *removing models from this diagram.* **DPO** ([Direct Preference Optimization & Its Variants](../05-posttraining-alignment/07-dpo-and-variants.html)) deletes the reward model *and* the critic by deriving a loss that optimizes the policy directly on preference pairs — leaving just the actor and a frozen reference. **GRPO/RLOO** ([GRPO, RLOO & Critic-Free RL](../05-posttraining-alignment/08-grpo-rloo.html)) delete the *critic* by using a group of sampled responses as a Monte Carlo baseline, keeping a reward (often a rule-based verifier) and a reference. Each method is, in part, an answer to "this four-model setup is too expensive."

## Putting it together: a skeleton RLHF loop

To cement how the pieces interlock, here is the *control flow* of the PPO stage in pseudo-real PyTorch. The actual PPO update (clipped surrogate, GAE) is intentionally abstracted into `ppo_update`, which the next chapter builds in full; the goal here is to see where each of the four models is called.

```python
import torch

def rlhf_ppo_epoch(actor, critic, reward_model, ref_model,
                   prompts, tokenizer, beta_kl=0.02, ppo_update=None):
    """One outer iteration of the RLHF/PPO loop.

    actor       : policy π_θ        (trained)   -- generates and is updated
    critic      : value V_ψ         (trained)   -- baseline for advantages
    reward_model: r_φ               (frozen)    -- terminal scalar reward
    ref_model   : π_ref (SFT copy)  (frozen)    -- KL anchor
    """
    # ---- 1. ROLLOUT: actor generates responses to a batch of prompts. ----
    #     In production this runs on a fast inference engine (vLLM/SGLang);
    #     see "The Generation–Training Loop & Rollout Engines".
    queries = tokenizer(prompts, return_tensors="pt", padding=True)
    with torch.no_grad():
        responses = actor.generate(**queries, max_new_tokens=512, do_sample=True)

    # full sequence = prompt ++ response; build a mask marking response tokens
    seq, resp_mask = build_sequences_and_mask(queries, responses)

    # ---- 2. SCORE & ANCHOR (all under no_grad; these models are not updated). ----
    with torch.no_grad():
        # Terminal scalar reward for each complete response (frozen RM).
        scores = reward_model(seq, attention_mask=(seq != tokenizer.pad_token_id))  # (B,)

        # Per-token log-probs from the FROZEN reference (for the KL penalty).
        ref_logprobs = token_logprobs(ref_model, seq, resp_mask)                    # (B, T)

    # ---- 3. The actor's own per-token log-probs and the critic's values
    #         (these DO require grad -- they define the PPO objective). ----
    actor_logprobs = token_logprobs(actor, seq, resp_mask)                          # (B, T)
    values         = critic(seq, resp_mask)                                         # (B, T)

    # ---- 4. Build the per-token reward: KL penalty everywhere + RM score at end. ----
    kl_per_token = actor_logprobs.detach() - ref_logprobs                           # (B, T)
    rewards = -beta_kl * kl_per_token                                               # KL "rent"
    last = resp_mask.sum(dim=1) - 1                                                 # final resp idx
    rewards[torch.arange(rewards.size(0)), last] += scores                          # add terminal r_φ

    # ---- 5. PPO update: compute advantages (GAE) from (rewards, values), then
    #         take clipped policy-gradient + value-function steps. See chapter 5.6. ----
    stats = ppo_update(actor, critic,
                       logprobs=actor_logprobs, old_logprobs=actor_logprobs.detach(),
                       values=values, rewards=rewards, resp_mask=resp_mask)
    return stats
```

Trace the four models through it: the **actor** generates (step 1) and supplies differentiable log-probs (step 3); the **reward model** scores once (step 2); the **reference** supplies frozen log-probs for KL (step 2); the **critic** supplies values (step 3) that become the advantage baseline (step 5). Every line of an industrial RLHF library is an elaboration of this skeleton — better generation engines, sharded models, vectorized KL control, and the PPO clip — but the data flow is exactly this.

!!! interview "Interview Corner"

    **Q:** Walk me through the InstructGPT RLHF pipeline. Why do we train a separate *reward model* instead of optimizing directly against human ratings, and what is the Bradley–Terry model doing?

    **A:** Three stages. (1) **SFT**: fine-tune the base model on human-written (prompt, response) demonstrations to get an instruction-following starting point $\pi_{\text{SFT}}$. (2) **Reward modeling**: for each prompt, sample $K$ responses from $\pi_{\text{SFT}}$, have humans *rank* them, extract $\binom{K}{2}$ pairwise preferences, and fit a reward model $r_\phi$ — the SFT network with a scalar head — by minimizing the **Bradley–Terry** loss $-\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))$. (3) **RL (PPO)**: optimize the policy to maximize $r_\phi$ minus a KL penalty to a frozen reference. We use a *learned* reward model rather than querying humans in the loop because PPO needs to score *millions* of freshly generated responses per run — humans are far too slow and expensive, and they cannot score outputs the policy hasn't generated yet. The reward model converts a sparse, discrete, slow human comparison signal into a **dense, differentiable, queryable** scalar over the *entire* response space. Bradley–Terry is the bridge: it models $P(y_w\succ y_l)=\sigma(r(x,y_w)-r(x,y_l))$, i.e. it treats each response as having a latent strength and the preference as a logistic function of the strength *difference*. Fitting it by max-likelihood gives the pairwise loss. A sharp candidate adds: rewards are only identified up to an additive constant (only differences matter), the RM can be smaller than the policy (judging is easier than generating), and the whole thing is a proxy that the policy will try to hack — hence the KL leash and early stopping on a held-out gold signal.

!!! interview "Interview Corner"

    **Q:** RLHF's PPO stage is said to require "four models." Name them, say which are trained vs. frozen, and explain what breaks if you drop the reference model or the critic.

    **A:** **Actor** (the policy $\pi_\theta$, *trained*), **critic** (value network $V_\psi$, *trained*), **reward model** ($r_\phi$, *frozen*), and **reference** ($\pi_{\text{ref}}$, a *frozen* copy of the SFT model). The reward model supplies the terminal scalar signal; the reference supplies $\log\pi_{\text{ref}}$ for the per-token KL penalty; the critic supplies a learned baseline so advantages have low variance. **Drop the reference** and you lose the KL anchor — the policy is free to drift arbitrarily far from the SFT distribution, which collapses fluency and, worse, lets it sprint toward reward-hacking regions (degenerate text that scores high on the proxy RM). The KL term is what makes hacking *expensive*, because hacked outputs are low-probability under the SFT model. **Drop the critic** and your policy-gradient estimates get very high variance, because you are now weighting actions by raw returns with no baseline; training becomes unstable and sample-inefficient. You *can* drop the critic if you replace it with another baseline — that is exactly what **RLOO/GRPO** do, using a group of sampled responses as a Monte Carlo baseline instead of a learned value network. So: reference is about *safety/stability of the distribution*; critic is about *variance of the gradient estimate*. Bonus point: the four-model footprint is the main reason DPO (which drops the RM and critic) and GRPO (which drops the critic) exist.

## Key Takeaways

!!! key "Key Takeaways"

    - **RLHF turns cheap comparisons into a learned objective.** Humans can't author a target for every prompt, but they can reliably say which of two responses is better. RLHF distills those preferences into a reward model and optimizes the policy against it, letting the model exceed its SFT demonstrations.
    - **The InstructGPT recipe is three stages:** SFT (imitate demonstrations) → reward modeling (fit a scalar from rankings) → PPO (maximize reward under a KL leash). Each stage's output seeds the next.
    - **Bradley–Terry is the bridge from discrete preferences to a continuous reward.** It models $P(y_w\succ y_l)=\sigma\big(r(x,y_w)-r(x,y_l)\big)$; max-likelihood gives the pairwise loss $-\log\sigma(r_w - r_l)$. Rewards are identified only up to an additive constant — **only score differences are meaningful.**
    - **The reward model is the SFT network with a scalar head**, read at the last token, trained as a shared-encoder (Siamese) comparator. It can be *smaller* than the policy, because recognizing quality is easier than producing it. Batch all $\binom{K}{2}$ pairs from one prompt together to avoid overfitting.
    - **The reward model is a proxy, and the policy is its adversary.** Optimizing it triggers Goodhart's law: length exploitation, sycophancy, format farming, and OOD gibberish that scores high but is worse to humans. "Reward went up" is never proof the model improved.
    - **The KL penalty to a frozen reference is the leash** that makes hacking expensive and curbs over-optimization; quality vs. KL traces a rise-then-fall curve, so you tune the KL budget and early-stop on a held-out *gold* signal you are *not* optimizing.
    - **The PPO stage juggles four models:** actor (trained), critic (trained), reward (frozen), reference (frozen). Reference controls distribution drift; critic controls gradient variance. This four-model, ~250 GB-for-7B footprint is precisely what DPO (drops RM + critic) and GRPO/RLOO (drop critic) were invented to shrink.

!!! sota "State of the Art & Resources (2026)"
    RLHF with learned reward models remains the backbone of production alignment, now extended through iterated online loops, generative/process reward models, and critic-free variants (GRPO, DAPO). The four-model PPO pipeline described here is the baseline every modern recipe either uses or explicitly departs from.

    **Foundational work**

    - [Christiano et al., *Deep Reinforcement Learning from Human Preferences* (2017)](https://arxiv.org/abs/1706.03741) — introduced the idea of learning a reward model from pairwise human comparisons, the conceptual seed of all RLHF pipelines.
    - [Ouyang et al., *Training language models to follow instructions with human feedback* (InstructGPT, 2022)](https://arxiv.org/abs/2203.02155) — the canonical three-stage SFT → RM → PPO recipe this chapter follows end to end.
    - [Bai et al. (Anthropic), *Training a Helpful and Harmless Assistant with RLHF* (2022)](https://arxiv.org/abs/2204.05862) — iterated RLHF, the helpfulness/harmlessness reward split, and the public HH-RLHF preference dataset.

    **Recent advances (2023–2026)**

    - [Gao, Schulman & Hilton, *Scaling Laws for Reward Model Overoptimization* (2022)](https://arxiv.org/abs/2210.10760) — characterizes the proxy-vs-gold reward over-optimization curve and derives the √KL functional form; essential reading before tuning β.
    - [Touvron et al., *Llama 2* (2023)](https://arxiv.org/abs/2307.09288) — iterated RLHF in practice: five rounds of RM retraining, separate helpfulness/safety reward models, and rejection-sampling + PPO in combination.
    - [Lambert et al., *RewardBench: Evaluating Reward Models for Language Modeling* (2024)](https://arxiv.org/abs/2403.13787) — the first standardized benchmark for reward models; the leaderboard reveals which RMs generalize to safety, reasoning, and instruction-following.
    - [Dong et al., *RLHF Workflow: From Reward Modeling to Online RLHF* (2024)](https://arxiv.org/abs/2405.07863) — comprehensive open-source recipe for online iterative RLHF, matching proprietary pipelines on AlpacaEval-2 and Arena-Hard.

    **Open-source & tools**

    - [huggingface/trl](https://github.com/huggingface/trl) — the standard library for RLHF in the HuggingFace ecosystem; `RewardTrainer` and `PPOTrainer` implement the core algorithms from this chapter.
    - [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — Ray + vLLM-based framework for scalable RLHF, supporting PPO, GRPO, REINFORCE++, and 70B+ models across multiple nodes.
    - [allenai/reward-bench](https://github.com/allenai/reward-bench) — RewardBench inference and evaluation harness for comparing reward models on chat, safety, and reasoning subsets.

## Further reading

- Ouyang, Wu, Jiang, et al., **Training language models to follow instructions with human feedback** (InstructGPT, 2022) — the canonical end-to-end RLHF recipe this chapter follows.
- Christiano, Leike, Brown, et al., **Deep Reinforcement Learning from Human Preferences** (2017) — the foundational paper that introduced learning a reward model from pairwise human comparisons.
- Bradley & Terry, **Rank Analysis of Incomplete Block Designs** (1952) — the original paired-comparison model underlying the reward loss; Luce/Plackett for the ranking generalization.
- Stiennon, Ouyang, Wu, et al., **Learning to summarize from human feedback** (2020) — the precursor that worked out reward modeling and PPO for a concrete task (summarization).
- Bai, Jones, Ndousse, et al. (Anthropic), **Training a Helpful and Harmless Assistant with RLHF** (2022) — the helpfulness/harmlessness split and separate reward models per axis.
- Gao, Schulman & Hilton, **Scaling Laws for Reward Model Overoptimization** (2022) — the proxy-vs-gold reward over-optimization curve and the $\sqrt{\mathrm{KL}}$ functional form.
- Touvron, Martin, Stone, et al., **Llama 2: Open Foundation and Fine-Tuned Chat Models** (2023) — iterated RLHF, separate helpfulness/safety RMs, rejection-sampling + PPO in practice.
- HuggingFace **TRL** (`RewardTrainer`, `PPOTrainer`) and **OpenRLHF** repositories — production implementations of the reward loss and four-model loop; see [TRL: HuggingFace's RL Library](../06-rl-infra/03-trl.html).
