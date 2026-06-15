# 1.8 The 10-Day Study Plan & Final Checklist

You have ten days. Maybe seven. Maybe the recruiter called yesterday and you have a long weekend. Whatever your window, the failure mode is always the same: you re-read the chapters you already know, panic-skim the ones you don't, and walk into the interview having *recognized* a thousand concepts but *retrieved* none of them under pressure. Recognition is not retrieval. An interview is a retrieval test conducted by a stranger while you are nervous.

This chapter is the antidote. It is a concrete, day-by-day operating plan that maps directly onto the chapters of this book, a spaced-repetition flashcard set of the must-knows, a final-48-hours checklist, and a short page to read on the morning of the interview to reset your nerves. It is deliberately prescriptive. When you are stressed, decision-making is expensive; a plan you can follow on autopilot is worth more than a perfect plan you have to design each morning.

The plan assumes you have already worked through the earlier interview-prep chapters — particularly [The Google ML Domain Interview: Format & Strategy](../13-interview-prep/01-google-ml-interview-format.html) for what the loop looks like, and [ML Breadth: Rapid-Fire Concepts & Model Answers](../13-interview-prep/02-ml-breadth-rapidfire.html) for the breadth bank. Here we sequence the whole book into a study cadence and turn it into muscle memory.

## The Philosophy: Retrieval, Spacing, and Interleaving

Three findings from cognitive science should drive every hour of your prep, because they are the cheapest performance wins available to you.

**The testing effect.** Trying to recall an answer *before* you check it strengthens memory far more than re-reading the answer. The classic Roediger & Karpicke studies showed that learners who studied once and then *tested* themselves repeatedly outperformed learners who *studied* the material four times — and the gap *widened* at a one-week delay. So: do not re-read chapters. Quiz yourself on them, then re-read only the gaps the quiz exposes.

**Spacing.** The same total study time produces dramatically better long-term retention when spread across days rather than massed into one block. The forgetting curve is roughly exponential. If $R(t)$ is the probability you still recall an item at time $t$ (in days) since the last review, a simple model is

$$
R(t) = e^{-t/S},
$$

where $S$ is the *memory strength* — a time constant. Each successful recall *increases* $S$, so the curve flattens. That is why a card reviewed on days 1, 3, and 7 is far stickier than the same card hammered three times on day 9. The entire reason for a 10-day plan rather than a 10-hour cram is to let $S$ grow between reviews.

**Interleaving.** Practicing related-but-different problems in a shuffled order — a coding problem, then a system-design prompt, then a breadth question — feels harder and slower than blocking, but produces better transfer. Interviews are interleaved by nature: you do not get to study one topic per interviewer in advance. Practice the way you will perform.

!!! note "Aside: why flashcards beat highlighting"

    Highlighting and re-reading produce a strong *illusion of competence*: the material feels familiar, so you believe you know it. Familiarity is a fluency signal, not a retrieval signal. Flashcards and self-testing break the illusion by forcing *production*. If a card makes you wince because you *almost* knew it — that wince is the single most valuable signal in your entire prep. Mark it and come back to it.

We will operationalize all three: the daily plan spaces topics, every day ends with self-testing, and the back half of the plan interleaves mock interviews across domains.

## A Minimal Spaced-Repetition Engine You Can Actually Run

You can use Anki, and you probably should. But it is worth understanding the algorithm — both because the mechanism *is itself a plausible interview topic* (it is a tiny online scheduling problem) and because a 40-line script you control beats an app you fight. Below is a from-scratch implementation of a simplified SM-2 scheduler — the algorithm behind Anki — that you can load with the flashcards later in this chapter.

```python
"""
A minimal spaced-repetition scheduler (simplified SM-2).

Each card has:
  - ease (E): how "easy" the card is; scales the interval. Starts at 2.5.
  - interval (I): days until the next review.
  - reps: number of consecutive successful recalls.

You grade each recall 0..5 (0 = blackout, 5 = perfect). A grade >= 3 is a pass.
On a pass, the interval grows multiplicatively by the ease factor, so well-known
cards rapidly drift to weeks apart while struggling cards stay daily.
"""
from dataclasses import dataclass, field
from datetime import date, timedelta
import json


@dataclass
class Card:
    front: str
    back: str
    ease: float = 2.5          # E-factor; SM-2 floor is 1.3
    interval: int = 0          # days
    reps: int = 0
    due: date = field(default_factory=date.today)

    def review(self, grade: int, today: date) -> None:
        """Update scheduling state given a recall grade in 0..5."""
        if grade < 3:
            # Failed: reset the streak, see it again tomorrow.
            self.reps = 0
            self.interval = 1
        else:
            # Passed: grow the interval.
            if self.reps == 0:
                self.interval = 1
            elif self.reps == 1:
                self.interval = 6
            else:
                self.interval = round(self.interval * self.ease)
            self.reps += 1

        # Update ease. Hard passes (grade 3) shrink ease; perfect passes (5) keep it.
        # This is the classic SM-2 update; ease never drops below 1.3.
        self.ease = max(1.3, self.ease + (0.1 - (5 - grade) * (0.08 + (5 - grade) * 0.02)))
        self.due = today + timedelta(days=self.interval)


def due_cards(deck, today):
    """Return cards due today or earlier, hardest-first (lowest ease)."""
    return sorted([c for c in deck if c.due <= today], key=lambda c: c.ease)


# --- Example session -------------------------------------------------------
if __name__ == "__main__":
    deck = [
        Card("Chinchilla optimal tokens-per-param?", "~20 tokens / parameter"),
        Card("Why sqrt(d_k) in attention?", "Keeps logit variance ~1 so softmax "
             "doesn't saturate; dot product of d_k unit-var dims has variance d_k."),
        Card("LoRA: what is delta W?", "B @ A, with A in R^{r x d}, B in R^{d x r}, "
             "r << d; only A,B are trained, W0 frozen."),
    ]
    today = date.today()
    for card in due_cards(deck, today):
        print("Q:", card.front)
        # In real use you'd prompt for input(); here we simulate a confident pass.
        card.review(grade=5, today=today)
        print("   ->", card.back, f"(next in {card.interval}d)\n")

    # Persist so the schedule survives across the 10 days.
    with open("deck.json", "w") as f:
        json.dump([{**c.__dict__, "due": c.due.isoformat()} for c in deck], f, indent=2)
```

Run this once a day, grade honestly, and the scheduler surfaces exactly the cards you are about to forget — which, by the forgetting-curve logic above, is precisely when reviewing them yields the most strength gain per minute.

!!! warning "Common pitfall: grading yourself too kindly"

    The entire value of spaced repetition collapses if you grade a card a "4" because you *recognized* the answer the instant you saw it. The test is: could you have *produced* the full answer, out loud, with no prompt, in the five seconds before flipping the card? If not, it is a fail (grade < 3) and you see it tomorrow. Honest grading is uncomfortable and non-negotiable.

## The 10-Day Plan, Day by Day

The plan is built around four-hour study days (scale the hours to your situation — the *sequence* matters more than the volume). Each day has a **theme**, a set of **chapters to actively recall** (not re-read cover to cover — skim, then self-test), a **build/code block**, and a **review block** that runs your due flashcards. Days are front-loaded with the highest-leverage, most-frequently-asked material, so that if life eats your last days, you have already covered the topics most likely to appear.

The ordering principle: *breadth and fundamentals first* (they anchor everything and are the most common bar-raiser), *the transformer and inference next* (the densest cluster of LLM-specific questions), *training and alignment in the middle*, and *system design plus behavioral last* (they integrate everything, so they reward being studied after the parts are fresh).

### Days 1–2: Foundations and Breadth

The goal of the first two days is to make your mathematical and classical-ML reflexes sharp, because a missed bias–variance or a botched cross-entropy derivation early in a loop colors every later interviewer's read of you.

| Day | Theme | Active-recall chapters | Build/code block |
|-----|-------|------------------------|------------------|
| 1 | Math + ML fundamentals | [Probability, Statistics & Information Theory](../01-foundations/02-probability-information.html), [Calculus, Optimization & Convexity](../01-foundations/03-calculus-optimization.html), [Machine Learning Fundamentals](../01-foundations/05-ml-fundamentals.html) | Logistic regression + its gradient from scratch in NumPy |
| 2 | Nets, autodiff, breadth bank | [Neural Networks From Scratch: MLPs & Backprop](../01-foundations/06-neural-nets-from-scratch.html), [Automatic Differentiation & PyTorch Internals](../01-foundations/07-autodiff-pytorch.html), [ML Breadth: Rapid-Fire Concepts](../13-interview-prep/02-ml-breadth-rapidfire.html) | A 30-line reverse-mode autodiff `Value` class, checked against `torch.autograd` |

On day 1, the code block is the single highest-yield exercise in classical ML: derive and implement binary cross-entropy and its gradient. The loss for one example with label $y \in \{0,1\}$ and predicted probability $\hat{y} = \sigma(z)$ is

$$
\mathcal{L} = -\big[\,y \log \hat{y} + (1-y)\log(1-\hat{y})\,\big],
$$

and the gradient with respect to the logit $z$ collapses to the famously clean

$$
\frac{\partial \mathcal{L}}{\partial z} = \hat{y} - y.
$$

Being able to *derive* that on a whiteboard — showing that the $\sigma'(z) = \sigma(z)(1-\sigma(z))$ term cancels the $1/\hat{y}$ from the log — is a classic warm-up interviewers love, because it separates people who memorized the result from people who understand it.

```python
import numpy as np

def sigmoid(z):
    # Numerically stable sigmoid: avoid exp overflow for large |z|.
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))

def bce_loss_and_grad(X, y, w, b):
    z = X @ w + b
    p = sigmoid(z)
    eps = 1e-12                                   # guard log(0)
    loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    # dL/dz = p - y  (the clean result you derive on the whiteboard)
    dz = (p - y) / len(y)
    grad_w = X.T @ dz
    grad_b = dz.sum()
    return loss, grad_w, grad_b
```

End each day by running your flashcard deck. On days 1–2 you are *seeding* the deck — most cards are brand new, so expect everything to be due tomorrow.

### Days 3–4: The Transformer and Attention

This is the densest, most-asked cluster in any LLM interview. If you can derive scaled dot-product attention, explain the KV cache, and reason about MQA/GQA trade-offs cold, you have covered a large fraction of the LLM-specific surface area.

| Day | Theme | Active-recall chapters | Build/code block |
|-----|-------|------------------------|------------------|
| 3 | Attention + heads | [The Attention Mechanism From Scratch](../02-transformer/03-attention-from-scratch.html), [Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html), [Positional Encodings (RoPE, ALiBi)](../02-transformer/05-positional-encoding.html) | Single-head causal attention; then a KV cache |
| 4 | Block + a whole GPT | [The Transformer Block](../02-transformer/06-transformer-block.html), [Building a GPT From Scratch](../02-transformer/07-build-gpt-from-scratch.html), [Tokenization](../02-transformer/01-tokenization.html) | Annotate nanoGPT's forward pass; count its parameters by hand |

The must-build for day 3 is causal self-attention. Type it from memory; the goal is to reach the point where the shapes flow without thinking.

```python
import torch
import torch.nn.functional as F

def causal_self_attention(x, W_q, W_k, W_v, W_o):
    """
    x: (B, T, d_model). One head, for clarity.
    Returns (B, T, d_model).
    """
    B, T, d = x.shape
    q = x @ W_q                       # (B, T, d_k)
    k = x @ W_k                       # (B, T, d_k)
    v = x @ W_v                       # (B, T, d_k)
    d_k = q.shape[-1]

    # Scaled scores. The 1/sqrt(d_k) keeps logit variance ~O(1) so softmax
    # doesn't saturate into a near-one-hot distribution with tiny gradients.
    scores = (q @ k.transpose(-2, -1)) / (d_k ** 0.5)   # (B, T, T)

    # Causal mask: position t may attend to <= t only. Set future to -inf
    # so softmax assigns them exactly zero weight.
    mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
    scores = scores.masked_fill(mask, float("-inf"))

    attn = F.softmax(scores, dim=-1)  # (B, T, T)
    out = attn @ v                    # (B, T, d_k)
    return out @ W_o                  # (B, T, d_model)
```

Day 4's pen-and-paper exercise — counting a GPT's parameters — deserves its own worked example, because the magnitudes recur in *every* memory and cost question you will get.

!!! example "Worked example: parameter count and KV-cache size of a 7B-class model"

    Take a decoder-only model with $L = 32$ layers, hidden size $d = 4096$, $h = 32$ heads, vocabulary $V = 32000$, and a feed-forward expansion of $4d$.

    **Parameters per layer.** Attention projections (Q, K, V, O) are four $d \times d$ matrices: $4 d^2 = 4 \cdot 4096^2 \approx 67.1\text{M}$. The MLP has an up-projection $d \times 4d$ and a down-projection $4d \times d$: $2 \cdot 4 d^2 = 8 d^2 \approx 134.2\text{M}$. Per layer $\approx 201.3\text{M}$.

    **All layers.** $32 \times 201.3\text{M} \approx 6.44\text{B}$.

    **Embeddings.** $V \cdot d = 32000 \cdot 4096 \approx 131\text{M}$ (plus a tied or separate output head of the same size). Total lands around $6.7\text{B}$ — hence "7B."

    **KV cache.** For a sequence of $S = 4096$ tokens, batch $B = 1$, in fp16 (2 bytes), the cache stores K and V for every layer:

    $$
    \text{bytes} = 2\,(\text{K,V}) \times L \times S \times d \times 2\,(\text{fp16}) = 2 \times 32 \times 4096 \times 4096 \times 2 \approx 4.3\text{ GB}.
    $$

    That single number — roughly **4 GB of KV cache for one 4k-token sequence** — explains why GQA, paged attention, and quantized KV exist. If an interviewer asks "why can't I just batch 100 of these on one 80 GB GPU?", you now answer with arithmetic, not hand-waving. See [PagedAttention & KV-Cache Memory Management](../04-kernels-efficiency/06-paged-attention-kv.html).

### Days 5–6: Pretraining, Scaling, and Efficiency

| Day | Theme | Active-recall chapters | Build/code block |
|-----|-------|------------------------|------------------|
| 5 | Objective + scaling + parallelism | [The Pretraining Objective & Loss](../03-pretraining/03-pretraining-objective.html), [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html), [Distributed Training I: DDP, ZeRO & FSDP](../03-pretraining/05-distributed-data-parallel.html) | Estimate training FLOPs ($C \approx 6ND$) for a target model |
| 6 | Kernels + quantization | [The Roofline Model](../04-kernels-efficiency/01-roofline-performance.html), [FlashAttention I: IO-Awareness](../04-kernels-efficiency/02-flash-attention-1.html), [Quantization I: PTQ (GPTQ, AWQ)](../04-kernels-efficiency/07-quantization-ptq.html) | Arithmetic intensity of a matmul; locate it on the roofline |

The day-5 reflex to drill is the *Chinchilla* compute relation. Training compute is approximately

$$
C \approx 6 N D
$$

FLOPs, where $N$ is parameter count and $D$ is the number of training tokens (the factor 6 is 2 for the forward multiply-add and 4 for the backward pass). Chinchilla's headline finding is that compute-optimal training uses roughly $D \approx 20 N$ — about **20 tokens per parameter**. Memorize "6ND" and "20 tokens per param"; together they let you sanity-check any training proposal in your head. See [Scaling Laws](../03-pretraining/04-scaling-laws.html).

The day-6 reflex is *arithmetic intensity*: a kernel is compute-bound if its FLOPs-per-byte exceeds the hardware's ratio of peak FLOP/s to peak memory bandwidth, and memory-bound otherwise. LLM *decode* is memory-bound — one token at a time streams the whole weight matrix — which is the root cause of why batching, speculative decoding, and quantization all help. This one idea unifies half of Part IV and Part VII.

### Days 7–8: Alignment, Inference, and Agents

| Day | Theme | Active-recall chapters | Build/code block |
|-----|-------|------------------------|------------------|
| 7 | Post-training + RLHF | [Supervised Fine-Tuning](../05-posttraining-alignment/01-sft-instruction-tuning.html), [PEFT I: LoRA, QLoRA](../05-posttraining-alignment/03-peft-lora-qlora.html), [The RLHF Pipeline & Reward Modeling](../05-posttraining-alignment/05-rlhf-reward-modeling.html), [Direct Preference Optimization](../05-posttraining-alignment/07-dpo-and-variants.html) | Implement the DPO loss in ~10 lines |
| 8 | Inference + agents + RAG | [Anatomy of LLM Inference](../07-inference-serving/01-anatomy-inference.html), [Continuous Batching](../07-inference-serving/02-continuous-batching.html), [Speculative Decoding](../07-inference-serving/06-speculative-decoding.html), [RAG Architectures](../09-rag-retrieval/03-rag-architectures.html) | Implement top-p (nucleus) sampling |

The day-7 build is the DPO objective: short, conceptually deep, and a magnet for follow-ups. Given a preferred completion $y_w$ and a dispreferred $y_l$ for prompt $x$, with policy $\pi_\theta$ and frozen reference $\pi_\text{ref}$,

$$
\mathcal{L}_\text{DPO} = -\log \sigma\!\Big(\beta \big[\log\tfrac{\pi_\theta(y_w\mid x)}{\pi_\text{ref}(y_w\mid x)} - \log\tfrac{\pi_\theta(y_l\mid x)}{\pi_\text{ref}(y_l\mid x)}\big]\Big).
$$

```python
import torch.nn.functional as F

def dpo_loss(pi_logp_w, pi_logp_l, ref_logp_w, ref_logp_l, beta=0.1):
    """
    Each arg is a (batch,) tensor of *summed* token log-probs of a full
    completion under the policy (pi_) or frozen reference (ref_).
    The model implicitly *is* its own reward model: r(x,y) = beta * log(pi/ref).
    """
    pi_logratio  = pi_logp_w  - pi_logp_l     # how much the policy prefers w over l
    ref_logratio = ref_logp_w - ref_logp_l    # the reference's preference
    # Train the policy to prefer w more strongly than the reference does.
    logits = beta * (pi_logratio - ref_logratio)
    return -F.logsigmoid(logits).mean()
```

Be ready for the killer follow-up: *why is there no separate reward model or PPO loop in DPO?* The answer — that the optimal RLHF policy has a closed form which lets you re-parameterize the reward as a function of the policy's own log-ratio, turning a two-stage RL problem into a single classification loss — is one of the cleanest "do you actually understand this" tests in alignment. See [Direct Preference Optimization & Its Variants](../05-posttraining-alignment/07-dpo-and-variants.html).

### Days 9–10: Integration — System Design, Coding, and Behavioral

The last two days are pure interleaving and mock interviews. You stop learning new material — the deadline for new facts has passed, and cramming new content now displaces consolidation of what you already know — and instead *integrate* under realistic conditions.

| Day | Theme | Active-recall chapters | Build/code block |
|-----|-------|------------------------|------------------|
| 9 | System design + coding | [ML System Design: A Framework](../13-interview-prep/03-ml-system-design-framework.html), [ML System Design: Worked Cases](../13-interview-prep/04-ml-system-design-cases.html), [Coding for ML Interviews](../13-interview-prep/05-coding-for-ml-interviews.html), [Designing an LLM Serving System](../12-production-mlops/01-serving-system-design.html) | Two timed system-design mocks (45 min each), out loud |
| 10 | Behavioral + light review | [Behavioral, Leadership & Project Deep-Dive](../13-interview-prep/07-behavioral-googleyness.html), [LLM-Specific Deep-Dive Questions](../13-interview-prep/06-llm-deepdive-questions.html), the *Final 48 Hours* checklist below | Rehearse 3 STAR stories out loud; one timed coding problem |

Do the day-9 mocks *out loud*, ideally with a peer or by recording yourself. Speaking the design — "let me start by clarifying requirements: what is the QPS, the latency SLO, and the acceptable cost per query?" — exposes the gaps that silent reading hides. The single biggest difference between candidates who *know* system design and candidates who *pass* it is whether they have rehearsed talking through one. See [ML System Design: A Framework](../13-interview-prep/03-ml-system-design-framework.html).

## The Spaced-Repetition Flashcard Set: The Must-Knows

Below is the core deck — the facts and one-liners that should be *instant* by interview day. Seed them into the scheduler on day 1. They are grouped by area, but you should review them *shuffled* (interleaving). Format: **front** → *back*.

**Foundations & math**

- Bias–variance decomposition? → $\mathbb{E}[(\hat{f}-y)^2] = \text{Bias}^2 + \text{Variance} + \sigma^2_\text{noise}$; high bias = underfit, high variance = overfit.
- Cross-entropy gradient w.r.t. the logit? → $\hat{y} - y$ (softmax/sigmoid + CE collapses cleanly).
- Why does Adam use $\hat{m}, \hat{v}$ bias correction? → $m, v$ start at 0, so early estimates are biased toward 0; dividing by $1-\beta^t$ corrects it.
- L1 vs L2 regularization? → L1 induces sparsity (corners of the diamond constraint); L2 shrinks smoothly toward 0.

**Transformer**

- Why $1/\sqrt{d_k}$ in attention? → A dot product of $d_k$ unit-variance dims has variance $d_k$; scaling restores variance $\approx 1$ so softmax doesn't saturate.
- MHA vs MQA vs GQA? → MQA shares one K,V across all heads (small cache, some quality loss); GQA shares K,V across *groups* — the practical middle ground.
- RoPE in one sentence? → Rotates Q,K by a position-dependent angle so the dot product depends only on *relative* position; extrapolates, and is the modern default.
- Pre-norm vs post-norm? → Pre-norm (LayerNorm/RMSNorm *inside* the residual branch) is far more stable at depth; the modern default.

**Pretraining & efficiency**

- Training FLOPs rule? → $C \approx 6ND$ ($N$ params, $D$ tokens).
- Chinchilla optimal? → ~20 tokens per parameter for compute-optimal training.
- ZeRO stages? → Stage 1 shards optimizer states, 2 adds gradients, 3 adds parameters (= FSDP).
- Memory-bound vs compute-bound decode? → Decode is memory-bound (streams all weights per token); prefill is compute-bound.

**Alignment & inference**

- DPO vs PPO? → DPO removes the explicit reward model + RL loop by re-parameterizing reward as $\beta\log(\pi_\theta/\pi_\text{ref})$; a single classification loss.
- KL term in RLHF, why? → Penalizes drift from the SFT reference so the policy doesn't reward-hack into degenerate text.
- Continuous (in-flight) batching? → Add/evict requests at the *token* level instead of waiting for a whole batch to finish; a big throughput win.
- Speculative decoding correctness? → A cheap draft proposes $k$ tokens; the target verifies in one pass and accepts via a rejection rule that *preserves the target's distribution exactly*.

**System design & production**

- First move in any ML system-design question? → Clarify requirements: scale (QPS), latency SLO, cost budget, quality bar, and what "good" means.
- RAG vs long-context vs fine-tuning? → RAG for fresh/large knowledge + citations; long-context for whole-document reasoning; fine-tuning for behavior/format/style.

!!! tip "Practitioner tip: make your own cards from your own mistakes"

    The deck above is a starting kit, not the whole deck. Every time a mock interview or a chapter exposes something you couldn't produce, write a card *in your own words* immediately. Cards you author from your own failures are worth ten cards copied from a list, because phrasing the question is itself a retrieval rehearsal — and the card is targeted at *your* specific gap.

## Interview Corner

!!! interview "Interview Corner"

    **Q:** You have one week before an ML interview and you feel weakest on inference and serving. Do you spend the whole week there?

    **A:** No — that is the classic trap of optimizing the topic that *feels* scariest rather than the one with the highest expected value. I'd allocate by *probability of being asked × current gap × recoverability*. Inference shows up often, so I'd give it solid coverage — KV cache, continuous batching, the prefill/decode memory-vs-compute split, speculative decoding — but I would *not* let it crowd out fundamentals (bias–variance, backprop, optimization) or system design, because those are the bar-raiser topics that appear in nearly every loop and that interviewers weight heavily. Concretely: I'd front-load fundamentals and the transformer in days 1–4 (most-asked, anchors everything), give inference a dedicated day, and reserve the final two days for integrated mocks. Spending a whole week on one topic also violates spacing — I'd retain less of everything. The meta-skill being probed here is *prioritization under a deadline*, which is exactly what the job demands, so I'd narrate that reasoning out loud.

## The Final 48 Hours Checklist

The last two days are about *consolidation and logistics*, not new learning. Cramming new material in the final 48 hours is actively counterproductive: it raises anxiety, displaces sleep, and interferes with the consolidation of what you already know. Work this checklist instead.

**T-minus 48 hours**

- [ ] Do one full timed mock per remaining interview type (coding, ML design, breadth). Score yourself honestly; note the top 3 gaps only.
- [ ] Run your *entire* flashcard deck once, even cards not due. Mark every wince — these few cards, not the whole book, are your final-day review list.
- [ ] Re-read your own STAR stories (situation, task, action, result). Say each one out loud once. Three to five rehearsed stories cover most behavioral prompts.
- [ ] Confirm logistics: interview time *in your timezone*, video link, interviewer names/roles, and the coding environment (shared doc? IDE? which languages are allowed?).

**T-minus 24 hours**

- [ ] Skim *only* your wince-cards and the Key Takeaways boxes of the chapters you study from. No new topics.
- [ ] Prepare your environment: water, charged laptop, backup internet (phone hotspot), pen and blank paper, a quiet room, a "do not disturb" sign.
- [ ] Write your three clarifying-question openers on a sticky note: *"What's the scale? What's the latency SLO? What does 'good' mean here?"* — you'll use these in design and ambiguous coding problems alike.
- [ ] Stop studying by early evening. Light exercise. **Protect your sleep** — see the next section on why this is non-negotiable.

**Morning of**

- [ ] No new material. Glance once at the must-know one-liners (FLOPs $= 6ND$, 20 tokens/param, $\hat{y}-y$, $1/\sqrt{d_k}$, KV cache $\approx$ 4 GB / 4k seq).
- [ ] Eat something with protein. Hydrate. Arrive or log in 10 minutes early.
- [ ] Re-read the "Reset Your Nerves" page below.
- [ ] Have your questions-for-the-interviewer ready. Asking good questions signals seniority and genuine interest.

!!! warning "Common pitfall: the all-nighter"

    Sleep is when memory consolidates — the hippocampus replays the day's learning into cortical long-term storage — and it is also when working memory and fluid reasoning are restored. A single night of short sleep measurably degrades exactly the faculties an interview taxes: recall speed, attention, and reasoning under pressure. An extra two hours of cramming at the cost of two hours of sleep is a *negative-expected-value trade*. The candidate who slept and reviewed 20 wince-cards will outperform the candidate who skipped sleep to re-read three chapters. Every time.

## Reset Your Nerves: The Morning-Of Page

Read this on the morning of the interview. It is short on purpose.

**You are not being asked to know everything.** No one does. The interviewer is not running a checklist of trivia; they are sampling how you *think* — how you decompose an ambiguous problem, how you reason about trade-offs, whether you say "I'm not sure, but here's how I'd figure it out" instead of bluffing. A confident "I don't know that specific number, but I can derive a bound" beats a confident wrong answer every time. Not knowing one thing does not sink the interview. *Pretending* to know it might.

**Reframe the nerves.** The racing heart and the adrenaline are not fear — they are your body mobilizing resources for a hard task. The physiological signature of "anxiety" and "excitement" is nearly identical; the difference is the label you attach. Tell yourself "I'm excited," not "I'm nervous." It sounds glib; it is supported by real research (Brooks, *Get Excited*, 2014) and it works.

**Use the box-breath before you start.** If your heart is pounding, do three rounds of: inhale 4 seconds, hold 4, exhale 4, hold 4. This down-regulates the sympathetic nervous system in under a minute and clears the cognitive fog that panic creates.

**Think out loud — it is a gift to you, not just the interviewer.** Narrating your reasoning slows you to a sustainable pace, surfaces partial credit even when you don't finish, and lets the interviewer nudge you back before you waste ten minutes down a wrong path. Silence helps no one. When stuck, say what you're considering and why; when you make an assumption, state it.

**Clarify before you solve.** The most common avoidable failure is solving the wrong problem fast. Spend the first two minutes asking questions. Interviewers *want* you to — designing the problem with you is part of the signal.

**It is a conversation between two engineers, not an interrogation.** The person across the table (or screen) was once where you are. Most interviewers are quietly rooting for you; a candidate who is a pleasure to think alongside is a candidate they want to hire. Be warm. Be curious. Treat their hints as collaboration, not as a verdict.

**One bad question is not a bad loop.** Interviews are scored holistically, often across multiple interviewers. If one segment goes sideways, mentally close that tab and start the next one fresh. Candidates routinely pass loops in which they bombed a question. The story you tell yourself between rounds matters more than any single round.

You have done the work. The ten days of spaced, interleaved, retrieval-based practice are *in* you now — that is exactly why we front-loaded the plan and protected your sleep. Trust the preparation, breathe, clarify, and think out loud. Go.

!!! key "Key Takeaways"

    - **Recognition is not retrieval.** Self-test against flashcards and quiz yourself; do not re-read chapters. The wince when you *almost* knew something is your highest-value signal.
    - **Space and interleave.** Ten short days beat one long cram because each successful recall increases memory strength $S$; shuffle topics so practice matches the interleaved reality of a real loop.
    - **Front-load the plan.** Fundamentals and the transformer first (most-asked, they anchor everything), training/efficiency/alignment in the middle, integrated system-design and behavioral mocks last.
    - **Memorize the load-bearing numbers:** $\partial\mathcal{L}/\partial z = \hat{y}-y$, attention scale $1/\sqrt{d_k}$, training FLOPs $\approx 6ND$, Chinchilla ~20 tokens/param, and "KV cache $\approx$ 4 GB per 4k-token sequence" for a 7B model.
    - **Drill must-build code cold:** causal attention + KV cache, a reverse-mode autodiff node, the DPO loss, top-p sampling, and logistic-regression-from-scratch.
    - **The final 48 hours are for consolidation, not new content.** Run mocks, review only your wince-cards, rehearse STAR stories out loud, fix logistics — and protect your sleep, which is when learning is actually stored.
    - **On the day: clarify before solving, think out loud, reframe nerves as excitement, and remember it's a conversation between two engineers.** A grounded "here's how I'd figure it out" beats a confident bluff.

## Further reading

- Roediger & Karpicke, *Test-Enhanced Learning: Taking Memory Tests Improves Long-Term Retention*, 2006 — the foundational paper on the testing effect.
- Ebbinghaus, *Memory: A Contribution to Experimental Psychology*, 1885 — the original forgetting curve.
- Brown, Roediger & McDaniel, *Make It Stick: The Science of Successful Learning*, 2014 — the practitioner's synthesis of spacing, interleaving, and retrieval practice.
- Brooks, *Get Excited: Reappraising Pre-Performance Anxiety as Excitement*, 2014 — the anxiety-reframing result.
- Wozniak, *SuperMemo / SM-2 algorithm* — the spaced-repetition scheduler reconstructed in this chapter, and the basis for Anki.
- Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla), 2022 — the source of the "20 tokens per parameter" rule you will quote.
- Rafailov et al., *Direct Preference Optimization*, 2023 — the derivation behind the DPO loss you should be able to write from memory.
