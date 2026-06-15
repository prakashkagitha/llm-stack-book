# 5.11 Constitutional AI, RLAIF & Self-Improvement

Human preference labels are expensive, inconsistent, and difficult to scale to the throughput that modern post-training pipelines demand. A single RLHF training run for a frontier model can consume millions of human-written comparisons, each requiring minutes of careful reading by a skilled annotator. Constitutional AI (CAI), Reinforcement Learning from AI Feedback (RLAIF), and the family of self-improvement techniques — STaR, ReST, self-rewarding language models, and weak-to-strong generalization — all attack this bottleneck from different angles. This chapter explains each mechanism in depth, shows working code for the critical components, and discusses where alignment data actually comes from at scale.

Before diving in, orient yourself: the underlying RLHF pipeline that CAI feeds into is covered in [The RLHF Pipeline & Reward Modeling](../05-posttraining-alignment/05-rlhf-reward-modeling.html), and the PPO/GRPO trainers that consume AI-generated reward signals live in [Policy Gradients & PPO for Language Models](../05-posttraining-alignment/06-ppo-for-llms.html) and [GRPO, RLOO & Critic-Free RL](../05-posttraining-alignment/08-grpo-rloo.html). Reward hacking risks that arise when an AI grades itself are treated carefully in [Reward Hacking, Over-Optimization & Alignment Failures](../05-posttraining-alignment/13-reward-hacking-failures.html).

---

## The Problem: Human Labeling at Scale

Human annotation has three compounding problems when you try to scale it.

**Volume.** A typical preference dataset for a chat model contains on the order of hundreds of thousands to a few million examples. The generation side of an RL run continuously produces new rollouts that need scoring — at the rate of millions per day for a large run. Human throughput simply does not match generation throughput.

**Consistency.** Human raters disagree, especially on nuanced safety and helpfulness trade-offs. Inter-annotator agreement on a 1–5 helpfulness scale can be surprisingly low. A reward model trained on noisy labels learns noisy preferences — it may generalize to behaviors the annotators never intended.

**Latency.** A human annotation job takes hours to days to return labels. RL algorithms like PPO want reward signals in seconds. The asynchronous mismatch forces practitioners to freeze the reward model during an RL run, which limits the feedback loop.

AI feedback sidesteps all three by replacing the human rater with a language model. The AI can score millions of samples per hour with deterministic temperatures, and the feedback loop closes in milliseconds. The trade-off is that the AI rater may embed its own biases and errors — which is exactly the problem that a *constitution* is designed to constrain.

---

## Constitutional AI (CAI)

### Core Idea: A Principle Set as a Steering Document

Constitutional AI, introduced by Bai et al. (Anthropic, 2022), gives the model an explicit list of principles — the "constitution" — and teaches it to evaluate its own outputs against those principles before producing a final response. The pipeline has two major stages.

**Stage 1 — Supervised Learning from Self-Critique (SL-CAI).** Given a prompt that would normally elicit a harmful response, the model first generates a harmful draft, then critiques the draft with reference to a specific constitutional principle, then revises the draft to be more aligned. This critique-revision loop can be iterated several times. The final revised response becomes the supervised fine-tuning (SFT) target.

**Stage 2 — RL from AI Feedback (RLAIF).** The model (or a larger "preference model") scores pairs of responses by asking which response better satisfies a given constitutional principle. These AI-generated preference labels train a preference model. The preference model then provides reward signals for an RL fine-tuning loop (e.g., PPO), yielding the final Constitutional AI model (CAI).

{{fig:cai-two-stage-pipeline}}

### The Constitution

A constitution is a list of natural-language principles. Anthropic's public constitution references ideas such as:

- "Choose the response that is least likely to contain information that could be used for harm."
- "Choose the response that is more honest and does not mislead the user."
- "Choose the response that is most consistent with being a helpful, honest, and harmless AI assistant."

Each principle is a short English sentence. During SL-CAI, one principle is sampled per example to construct the critique prompt. During RLAIF, one principle is sampled to construct the comparison prompt. Cycling through the full principle set creates diverse training signal that covers many alignment dimensions without requiring a human to write a rubric for each.

```python
import random
from dataclasses import dataclass
from typing import Optional

# A minimal, illustrative constitution (not the full Anthropic version)
CONSTITUTION = [
    "Choose the response that is least likely to cause physical or psychological harm.",
    "Choose the response that is most honest and does not deceive the user.",
    "Choose the response that avoids discrimination based on race, gender, or religion.",
    "Choose the response that does not assist with illegal activities.",
    "Choose the response that best respects the user's autonomy and right to make decisions.",
    "Choose the response that is most helpful while remaining safe.",
    "Choose the response that avoids generating explicit or adult content.",
    "Choose the response that protects user privacy and does not solicit personal data.",
]

@dataclass
class ConstitutionExample:
    prompt: str
    initial_response: str
    principle: str
    critique_prompt: str
    revision_prompt: str

def build_critique_prompt(prompt: str, response: str, principle: str) -> str:
    """
    Construct the critique prompt used in SL-CAI Stage 1.
    The model is asked to identify problems in `response` according to `principle`.
    """
    return (
        f"Human: {prompt}\n\n"
        f"Assistant: {response}\n\n"
        f"Please critique the above response with respect to the following principle:\n"
        f"  '{principle}'\n\n"
        f"Identify specific problems and explain briefly why they violate the principle.\n"
        f"Critique:"
    )

def build_revision_prompt(
    prompt: str,
    response: str,
    principle: str,
    critique: str,
) -> str:
    """
    Construct the revision prompt: given the critique, rewrite the response
    to satisfy the principle.
    """
    return (
        f"Human: {prompt}\n\n"
        f"Assistant: {response}\n\n"
        f"Critique (for principle '{principle}'):\n{critique}\n\n"
        f"Now rewrite the response to address the critique and satisfy the principle.\n"
        f"Revised response:"
    )

def build_preference_prompt(
    prompt: str,
    response_a: str,
    response_b: str,
    principle: str,
) -> str:
    """
    RLAIF Stage 2: ask an AI judge which response better satisfies the principle.
    Returns a prompt whose completion should be 'A' or 'B'.
    """
    return (
        f"Consider the following principle:\n  '{principle}'\n\n"
        f"Human turn: {prompt}\n\n"
        f"Response A:\n{response_a}\n\n"
        f"Response B:\n{response_b}\n\n"
        f"Which response better satisfies the principle above? Answer with a single letter, "
        f"either 'A' or 'B', and nothing else.\nAnswer:"
    )

def sample_principle() -> str:
    return random.choice(CONSTITUTION)
```

### Iterative Critique-Revision in Practice

The most powerful insight in CAI is that a model can meaningfully improve its own outputs through critique, even when the initial response was misaligned. In practice, one to four critique-revision rounds provide most of the benefit; further iterations show diminishing returns and risk over-sanitizing helpful content.

```python
from typing import Callable

def critique_revise_loop(
    model_generate: Callable[[str], str],
    prompt: str,
    num_rounds: int = 3,
    verbose: bool = False,
) -> str:
    """
    Run the SL-CAI critique-revision loop.

    Args:
        model_generate: a function that calls the LLM and returns a completion string.
        prompt:         the user's original request.
        num_rounds:     how many critique+revision cycles to run.

    Returns:
        The final revised response after `num_rounds` iterations.
    """
    # Step 1: generate an initial (potentially misaligned) response
    current_response = model_generate(f"Human: {prompt}\n\nAssistant:")

    for round_idx in range(num_rounds):
        # Sample a different principle each round for diversity
        principle = sample_principle()

        # Step 2: critique the current response
        critique_prompt = build_critique_prompt(prompt, current_response, principle)
        critique = model_generate(critique_prompt).strip()

        if verbose:
            print(f"--- Round {round_idx + 1}: principle='{principle[:60]}...' ---")
            print(f"Critique: {critique[:200]}")

        # Step 3: revise based on the critique
        revision_prompt = build_revision_prompt(
            prompt, current_response, principle, critique
        )
        current_response = model_generate(revision_prompt).strip()

        if verbose:
            print(f"Revision preview: {current_response[:200]}\n")

    return current_response


# ------------------------------------------------------------------
# Toy invocation (replace `model_generate` with a real API call):
# ------------------------------------------------------------------
def mock_model(prompt: str) -> str:
    """Stub — in production replace with an API call to your LLM."""
    if "Critique:" in prompt:
        return "The response could encourage harm by providing step-by-step instructions."
    if "Revised response:" in prompt:
        return "I'm unable to help with that request. Here's what I can offer instead..."
    return "Sure, here's exactly how to do that dangerous thing..."  # initial harmful stub

final = critique_revise_loop(mock_model, "How do I hack into my neighbor's WiFi?", verbose=True)
print("Final response:", final)
```

### Building the AI Preference Dataset

For Stage 2, we generate multiple candidate responses for each prompt (often two, but sometimes more) and use the AI judge to assign a preference label. A useful addition is asking the judge for a *chain-of-thought* explanation before the final label, which improves calibration.

```python
import re
from typing import NamedTuple

class PreferencePair(NamedTuple):
    prompt: str
    chosen: str      # the preferred response
    rejected: str    # the less-preferred response
    principle: str
    judge_rationale: str

def ai_label_pair(
    model_generate: Callable[[str], str],
    prompt: str,
    response_a: str,
    response_b: str,
    use_cot: bool = True,
) -> PreferencePair:
    """
    Use an AI judge to label which of response_a / response_b is preferred.
    With `use_cot=True`, the judge reasons before giving its final verdict,
    which empirically increases accuracy.
    """
    principle = sample_principle()

    if use_cot:
        judge_prompt = (
            f"Consider the following principle:\n  '{principle}'\n\n"
            f"Human turn: {prompt}\n\n"
            f"Response A:\n{response_a}\n\n"
            f"Response B:\n{response_b}\n\n"
            f"First, briefly explain which response better satisfies the principle and why.\n"
            f"Then, on a new line, write exactly: VERDICT: A  or  VERDICT: B\n\n"
            f"Reasoning:"
        )
    else:
        judge_prompt = build_preference_prompt(prompt, response_a, response_b, principle)

    judge_output = model_generate(judge_prompt)

    # Parse the verdict
    verdict_match = re.search(r"VERDICT:\s*([AB])", judge_output, re.IGNORECASE)
    if verdict_match is None:
        # Fall back to looking for a standalone letter
        verdict_match = re.search(r"\b([AB])\b\s*$", judge_output.strip())

    verdict = verdict_match.group(1).upper() if verdict_match else "A"  # default tie-break

    chosen, rejected = (response_a, response_b) if verdict == "A" else (response_b, response_a)

    return PreferencePair(
        prompt=prompt,
        chosen=chosen,
        rejected=rejected,
        principle=principle,
        judge_rationale=judge_output,
    )
```

---

## RLAIF: Replacing Human Raters with AI

RLAIF (Lee et al., Google, 2023) is a direct, controlled comparison between human feedback and AI feedback for training a reward model. The key finding is that, for many helpfulness tasks, a preference model trained on AI labels achieves win rates against human-preference-trained baselines that are roughly comparable, at a fraction of the cost.

### The Labeling Protocol

The AI judge is typically a model that is at least as capable as the policy being trained — often the same scale or larger. The judge is prompted with the comparison task and asked to reason before deciding. Two important practical choices:

1. **Position bias mitigation.** LLMs systematically prefer the first response they read. Swap the order of A and B for a fraction of examples, and take the average (or the majority vote).

2. **Self-consistency filtering.** Run the judge multiple times with temperature > 0 and only keep examples where the judge gives a consistent verdict. This removes ambiguous cases that would add label noise to the reward model.

```python
import torch
from collections import Counter

def robust_ai_label(
    model_generate: Callable[[str], str],
    prompt: str,
    response_a: str,
    response_b: str,
    n_votes: int = 5,
    consistency_threshold: float = 0.7,
) -> Optional[PreferencePair]:
    """
    Multi-vote AI labeling with position-debiasing and consistency filtering.

    - n_votes: number of independent judge calls.
    - consistency_threshold: discard examples where the majority is below this fraction.
    """
    votes = []
    rationales = []

    for i in range(n_votes):
        # Alternate order to cancel position bias
        if i % 2 == 0:
            pair = ai_label_pair(model_generate, prompt, response_a, response_b)
        else:
            # Swap positions, then flip the verdict back
            pair_swapped = ai_label_pair(model_generate, prompt, response_b, response_a)
            # After swap, 'chosen' refers to the winner in swapped order;
            # re-map: if swapped chosen == response_b, verdict was "A" in swapped = B in original
            if pair_swapped.chosen == response_b:
                # original A won after swap-correction
                pair = PreferencePair(
                    prompt=prompt, chosen=response_a, rejected=response_b,
                    principle=pair_swapped.principle, judge_rationale=pair_swapped.judge_rationale
                )
            else:
                pair = PreferencePair(
                    prompt=prompt, chosen=response_b, rejected=response_a,
                    principle=pair_swapped.principle, judge_rationale=pair_swapped.judge_rationale
                )

        votes.append(pair.chosen)
        rationales.append(pair.judge_rationale)

    vote_counter = Counter(votes)
    majority_response, majority_count = vote_counter.most_common(1)[0]
    consistency = majority_count / n_votes

    if consistency < consistency_threshold:
        return None  # discard ambiguous example

    rejected_response = response_b if majority_response == response_a else response_a
    return PreferencePair(
        prompt=prompt,
        chosen=majority_response,
        rejected=rejected_response,
        principle=sample_principle(),
        judge_rationale=rationales[0],  # representative rationale
    )
```

### RLAIF vs Human Preference: Trade-offs

| Dimension | Human feedback | AI feedback (RLAIF) |
|---|---|---|
| Cost per label | USD 0.05–0.50 | USD 0.0001–0.01 (API cost) |
| Throughput | ~1 M/month (large team) | ~100 M/day |
| Consistency | Inter-annotator ~60–80 % | High (same prompt → same output) |
| Cultural bias | Rater pool bias | Model's training data bias |
| Safety sensitivity | Good with training | Requires explicit constitution |
| Scalability | Linear in headcount | Linear in compute |

The biggest practical risk of RLAIF is *bias amplification*: the judge model's preferences are learned, not ground truth, and those preferences can encode harmful biases invisibly. A constitution provides a partial check by making the evaluation criterion explicit and human-auditable.

---

## Self-Rewarding Language Models

Self-rewarding LMs (Yuan et al., Meta, 2024) push the RLAIF idea further: instead of a separate judge model, the *same model* acts as both policy and reward model. The training loop alternates between (1) generating candidate responses and (2) scoring them using the model itself in "judge mode."

The model is fine-tuned on a joint objective: it must be good at answering user queries *and* good at evaluating responses according to rubrics. Concretely, during the scoring step the model is prompted with a rubric:

```text
Review the following response to a user's query.
Score it on a scale from 1 to 5 on each of:
  - Instruction following
  - Accuracy
  - Helpful completeness
  - Safety

Response: {response}

Scores (format: IFollow=X, Accuracy=X, Completeness=X, Safety=X):
```

The training loop is:

1. **SFT initialization.** Fine-tune on a seed dataset of high-quality (prompt, response) pairs to give the model a strong starting point in both answer quality and judging ability.
2. **LLM-as-a-judge (LaaJ) data generation.** For each training prompt, sample $k$ responses from the current policy at some temperature. Score each response using the model in judge mode. This yields a ranked set of responses.
3. **Preference dataset construction.** Treat the highest-scored response as chosen and the lowest as rejected. (Or use all $\binom{k}{2}$ pairs.)
4. **DPO/IPO update.** Fine-tune on the new preference pairs using [Direct Preference Optimization & Its Variants](../05-posttraining-alignment/07-dpo-and-variants.html).
5. **Repeat.** The updated model is used as the judge in the next round.

The virtuous cycle is that as the policy improves, the judge also improves (since they share weights), leading to progressively better preference data — *self-improvement through self-evaluation*.

!!! warning "Self-reward collapse"
    If the model's judge mode becomes too lenient (grading its own outputs too highly), the preference pairs collapse to near-zero margin and DPO training stagnates. Regularize by mixing in external human preference pairs or by adding entropy bonuses to the judge scoring distribution.

---

## Rejection Sampling, STaR & ReST

### Rejection Sampling Fine-Tuning (RFT)

The simplest self-improvement technique is rejection sampling fine-tuning. Given a seed policy $\pi_0$ and a verifiable reward $r$ (a unit-test pass, a math answer checker, etc.), the loop is:

1. For each training problem $x$, sample $k$ completions $y_1, \ldots, y_k \sim \pi_\theta(\cdot | x)$.
2. Keep only those $y_i$ for which $r(x, y_i) = 1$ (correct answer).
3. Fine-tune $\pi_\theta$ on the kept (x, y) pairs.
4. Repeat.

Each iteration increases the probability of correct solutions because the training set now contains only correct demonstrations. This is related to [RL with Verifiable Rewards (RLVR) & The Reasoning Recipe](../05-posttraining-alignment/09-rlvr-reasoning.html), which covers the verifiable-reward approach in depth.

```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

class RejectionSamplingDataset(Dataset):
    """
    Dynamically built rejection-sampling dataset.
    Stores only (prompt, completion) pairs that passed the reward filter.
    """
    def __init__(self, accepted_pairs: list[tuple[str, str]], tokenizer, max_len: int = 512):
        self.pairs = accepted_pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        prompt, completion = self.pairs[idx]
        text = prompt + completion + self.tokenizer.eos_token
        tokens = self.tokenizer(
            text, truncation=True, max_length=self.max_len, return_tensors="pt"
        )
        input_ids = tokens["input_ids"].squeeze(0)
        # Mask the prompt tokens in the loss (train only on the completion)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].squeeze(0)
        labels = input_ids.clone()
        labels[: len(prompt_ids)] = -100  # ignore_index
        return {"input_ids": input_ids, "labels": labels}


def rejection_sampling_iteration(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    reward_fn,          # reward_fn(prompt, completion) -> bool
    k: int = 16,        # number of samples per prompt
    temperature: float = 0.8,
) -> list[tuple[str, str]]:
    """
    One round of rejection sampling.
    Returns a list of (prompt, completion) pairs that earned reward == 1.
    """
    accepted = []
    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
            # Sample k completions
            outputs = model.generate(
                input_ids,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=256,
                num_return_sequences=k,
                pad_token_id=tokenizer.eos_token_id,
            )
            for seq in outputs:
                # Decode only the newly generated tokens
                new_tokens = seq[input_ids.shape[-1]:]
                completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
                if reward_fn(prompt, completion):
                    accepted.append((prompt, completion))
    return accepted
```

### STaR: Self-Taught Reasoner

Zeiler et al. (2022) introduced STaR (Self-Taught Reasoner), which addresses a key problem: when $k$ is small and the problem is hard, the model may produce zero correct completions for many prompts, yielding no training signal. STaR adds a *rationalization hint*: when the model fails, show it the ground-truth answer and ask it to construct a chain-of-thought that leads to that answer, then use that chain-of-thought as an additional training example.

$$
\mathcal{D}_\text{STaR} = \underbrace{\{(x, r_{\text{sampled}}, y^*) : r \text{ correct}\}}_{\text{self-generated rationales}} \cup \underbrace{\{(x \| y^*, r_{\text{hint}}, y^*) : r \text{ incorrect}\}}_{\text{hint-conditioned rationales}}
$$

where $x$ is the question, $r$ is the rationale, and $y^*$ is the correct answer. The hint set provides training signal even when the model cannot solve problems cold.

### ReST: Reinforced Self-Training

ReST (Gulcehre et al., Google, 2023) separates the data generation step ("Grow") from the fine-tuning step ("Improve") and repeats them:

- **Grow:** sample a large offline dataset $\mathcal{D}^t$ from the current policy $\pi_t$. Filter by reward threshold $\tau_t$.
- **Improve:** fine-tune $\pi_{t+1}$ on $\mathcal{D}^t$ starting from $\pi_t$.

Critically, the threshold $\tau_t$ is raised each iteration — the bar gets higher. Early rounds accept solutions with partial credit; later rounds only accept near-perfect solutions. This curriculum prevents the model from plateauing on easy examples.

!!! example "Worked example: rejection-sampling pass rates"
    Consider a math benchmark where the seed policy $\pi_0$ solves 30 % of problems correctly when sampling $k=1$ (greedy). With $k=32$ samples and temperature 0.8, pass@32 might be around 70 %: 70 % of problems yield at least one correct completion that becomes a training example.

    After one RFT iteration on this data, the new policy $\pi_1$ achieves, say, 42 % at $k=1$. A second iteration with $k=32$ now hits pass@32 ≈ 85 %, and $k=1$ accuracy climbs to ~55 %. Each round narrows the gap between greedy accuracy and best-of-$k$ accuracy, which is the core mechanism: RFT converts sampling-time compute into model capability.

    Concrete magnitudes: a 7B model on MATH training problems might consume about 200 MB of training text per iteration (a few tens of thousands of accepted completions at ~1,000 tokens each). Each GPU-hour of generation produces on the order of 50,000 completions at temperature 0.8 on an H100.

---

## Self-Play & Iterative Self-Improvement

Self-play is familiar from game-playing AI (AlphaGo, AlphaStar): two copies of the model compete, and the stronger copy's move choices become training data. In the language domain, the analogies are:

- **Red-teaming self-play.** One copy of the model generates adversarial prompts designed to elicit unsafe outputs from the other copy. Successful jailbreaks become SFT safety-training examples.
- **Debate.** Two model instances argue opposing positions on a factual question; a third model (or human) judges. The winning side's arguments reinforce correct reasoning.
- **Constitutional self-play (in SPIN, Chen et al., 2024).** The policy at iteration $t$ generates responses that are used as *rejected* examples, while real high-quality human data provides *chosen* examples. This forces the model to distinguish its own (current) output distribution from the gold distribution, directly optimizing the gap.

```python
# Simplified SPIN (Self-Play Fine-Tuning) data construction
# Reference: Chen et al., "Self-Play Fine-Tuning Converts Weak Language Models to Strong", 2024

def build_spin_pairs(
    model_generate: Callable[[str], str],
    human_data: list[tuple[str, str]],  # (prompt, gold_response)
) -> list[PreferencePair]:
    """
    For each (prompt, gold) pair in the human dataset:
      - Generate a response from the current policy (the 'player' copy).
      - Use gold_response as 'chosen', policy response as 'rejected'.
    This creates a preference signal that pushes the policy distribution
    toward the human data distribution.
    """
    pairs = []
    for prompt, gold_response in human_data:
        # The 'main player' generates the rejected sample
        policy_response = model_generate(f"Human: {prompt}\n\nAssistant:")
        pairs.append(PreferencePair(
            prompt=prompt,
            chosen=gold_response,
            rejected=policy_response,
            principle="Match the quality and style of high-quality human responses.",
            judge_rationale="Human data is used as gold reference.",
        ))
    return pairs
```

---

## Weak-to-Strong Generalization

Weak-to-strong generalization (Burns et al., OpenAI, 2023) addresses a different but related problem: what happens when the student model is *stronger* than the supervisor? This is directly relevant to the scalable oversight problem: future models will likely surpass human raters in many domains, making human feedback unreliable.

The experimental setup is:

1. Choose a "weak supervisor" — a small model or a model with degraded capability — and elicit its labels.
2. Fine-tune a "strong student" (a much larger model) on those weak labels.
3. Ask: does the strong student exceed the weak supervisor's performance? Does it recover the "ceiling" set by a strong-supervised baseline?

Empirically, the strong student trained on weak labels does significantly *better* than the weak supervisor — it generalizes beyond the noisy signal. The gap between weak-supervised and strong-supervised performance (the "elicitation gap") shrinks with certain interventions:

- **Bootstrapping.** Train an intermediate "medium" model on weak labels, then use the medium model to relabel data for the strong model.
- **Consistency regularization.** Penalize the strong model for confidently disagreeing with the weak supervisor.
- **Auxiliary confidence loss.** Train the strong model to predict *where* the weak supervisor's labels are likely correct, then weight training examples accordingly.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeakToStrongTrainer:
    """
    Illustrative trainer that implements the 'bootstrapping' weak-to-strong strategy.
    In practice you would use your distributed training stack, but this
    sketch shows the key loss computation.
    """

    def __init__(
        self,
        strong_model: nn.Module,
        weak_labels: torch.Tensor,   # shape [N], float in [0,1]
        confidence_weight: float = 0.1,
    ):
        self.model = strong_model
        self.weak_labels = weak_labels
        self.conf_w = confidence_weight

    def compute_loss(
        self,
        logits: torch.Tensor,     # shape [B, 2] for binary classification
        indices: torch.Tensor,    # which examples in the batch
    ) -> torch.Tensor:
        """
        Main loss = cross-entropy with weak labels.
        Auxiliary loss = learn the *confidence* of weak labels
                         (treat examples with extreme weak probabilities as high-confidence).
        """
        weak = self.weak_labels[indices]  # [B] floats
        probs = torch.sigmoid(logits[:, 1])  # binary case: P(positive)

        # Primary BCE against weak labels
        ce_loss = F.binary_cross_entropy(probs, weak)

        # Confidence signal: |weak - 0.5| is high when weak supervisor is confident
        weak_confidence = (2 * (weak - 0.5).abs())  # maps [0,1] to [0,1]
        # Encourage the strong model to be right where the weak label is confident
        conf_loss = F.binary_cross_entropy(probs, weak, weight=weak_confidence)

        return ce_loss + self.conf_w * conf_loss
```

!!! interview "Interview Corner"
    **Q:** What is Constitutional AI, and how does it reduce the cost of alignment data compared to vanilla RLHF?

    **A:** Constitutional AI (Bai et al., Anthropic 2022) replaces human preference labels with AI-generated labels guided by an explicit principle set — the "constitution." Instead of asking human raters which of two responses is better, CAI prompts a language model to judge pairs of responses against a stated principle ("choose the response that avoids harm"). This yields two wins over standard RLHF: (1) AI can label millions of pairs per day at a small fraction of the cost of a human annotation round, and (2) the evaluation criterion is explicit and auditable — the constitution is a human-readable document that specifies exactly what the judge is measuring. The pipeline also includes a supervised stage where the model critiques and revises its own potentially harmful drafts before ever involving RL, which gives clean SFT training signal without human annotators. The main trade-off is that the AI judge inherits its own model's biases, so the constitution must be carefully designed and periodically audited.

---

## Sourcing Alignment Data at Scale: A Practical View

Where does alignment data actually come from in a production post-training pipeline? The honest answer is: from many sources in parallel, with AI-labeling taking an increasingly large share.

### The Alignment Data Flywheel

{{fig:cai-alignment-data-flywheel}}

Key stages:

1. **Seed human data (O(10K–100K) examples).** Expert annotators write high-quality demonstrations and preference comparisons for the most safety-critical and capability-critical cases. This is expensive but small.

2. **SFT warmup.** The seed data trains a first supervised model. This model generates plausible candidate responses for the RLAIF stage.

3. **RLAIF at scale (O(1M–100M) labels).** A judge model (often a larger model or the SFT model itself) labels AI-generated comparison pairs. Constitutional principles guide the labeling. The resulting preference dataset trains a reward model.

4. **RL fine-tuning.** PPO or DPO on the AI-labeled preference data updates the policy. See [Policy Gradients & PPO for Language Models](../05-posttraining-alignment/06-ppo-for-llms.html).

5. **Human spot-check QA.** A sample of the RL model's outputs is audited by humans — not to re-label everything, but to catch systematic failures or reward hacking. This is the human bottleneck that remains, kept small but targeted.

6. **Iterative upgrade.** The RL policy becomes the next seed for another generation of RLAIF data, creating a flywheel.

### What to Prioritize with Limited Human Budget

Given a fixed annotation budget, practitioners should spend it on:

- **Hard cases.** Easy prompts can be labeled by AI with high confidence. Hard, nuanced, or safety-critical prompts are where human judgment adds the most value over AI.
- **Adversarial red-teaming.** Human red-teamers find jailbreaks that AI self-play may miss.
- **Calibration examples.** A small set of "anchor" examples with known correct preferences helps calibrate the AI judge and catches drift between iterations.
- **Low-agreement filtering.** Run the AI judge multiple times; spend human budget on the examples where the judge is inconsistent.

```python
# Practical data-budget allocation sketch

def allocate_annotation_budget(
    total_examples: int,
    ai_labeling_confidence: list[float],  # per-example confidence in [0,1]
    human_budget_fraction: float = 0.05,  # spend 5% of budget on humans
) -> tuple[list[int], list[int]]:
    """
    Given a list of per-example AI confidence scores, route the
    least-confident examples to human annotators.
    Returns (human_indices, ai_indices).
    """
    n_human = int(total_examples * human_budget_fraction)
    # Sort by ascending confidence: least confident go to humans
    ranked = sorted(range(len(ai_labeling_confidence)),
                    key=lambda i: ai_labeling_confidence[i])
    human_indices = ranked[:n_human]
    ai_indices = ranked[n_human:]
    return human_indices, ai_indices
```

---

## Putting It All Together: A CAI + RLAIF + RFT Recipe

A modern self-improving alignment pipeline combines all the techniques above. Here is a concrete recipe and the key hyperparameters to tune.

```python
"""
End-to-end sketch of a CAI + RLAIF + Rejection-Sampling pipeline.
This is pseudocode-level but shows the data flow clearly.
In production each step would run on a distributed cluster.
"""

from dataclasses import dataclass, field

@dataclass
class AlignmentPipelineConfig:
    # SL-CAI (critique-revision)
    cai_num_critique_rounds: int = 3       # k rounds of critique+revision
    cai_harmful_prompt_fraction: float = 0.3  # fraction of prompts to use CAI on

    # RLAIF preference labeling
    rlaif_judge_model: str = "anthropic/claude-3-opus"  # or your own larger model
    rlaif_n_votes: int = 5                 # majority-vote ensemble
    rlaif_consistency_threshold: float = 0.7

    # Rejection sampling (RFT)
    rft_k: int = 32                        # completions per prompt
    rft_temperature: float = 0.8
    rft_num_iterations: int = 3

    # DPO fine-tuning
    dpo_beta: float = 0.1                  # KL penalty
    dpo_learning_rate: float = 5e-7
    dpo_epochs: int = 1

    # Human spot-check
    human_qa_fraction: float = 0.02       # 2% of final outputs reviewed by humans


def run_alignment_pipeline(
    seed_policy,
    tokenizer,
    harmful_prompts: list[str],
    general_prompts: list[str],
    verifiable_prompts: list[str],
    reward_fn,
    config: AlignmentPipelineConfig,
):
    """
    Full pipeline (abbreviated, showing the structure):

    1. SL-CAI: generate clean responses for harmful prompts via critique-revision.
    2. SFT warmup on SL-CAI outputs + existing SFT data.
    3. RLAIF: label general-prompt pairs with AI judge.
    4. Train reward model on RLAIF pairs.
    5. RFT on verifiable-reward tasks.
    6. DPO on RLAIF preference pairs.
    7. Human QA gate.
    """

    # === STAGE 1: SL-CAI ===
    cai_training_pairs = []
    for prompt in harmful_prompts:
        safe_response = critique_revise_loop(
            seed_policy.generate_text,
            prompt,
            num_rounds=config.cai_num_critique_rounds,
        )
        cai_training_pairs.append((prompt, safe_response))

    # === STAGE 2: SFT warmup ===
    # (sft_train(seed_policy, cai_training_pairs) — omitted for brevity)

    # === STAGE 3: RLAIF ===
    preference_dataset = []
    for prompt in general_prompts:
        # Sample two responses at moderate temperature
        resp_a = seed_policy.generate_text(prompt, temperature=0.8)
        resp_b = seed_policy.generate_text(prompt, temperature=0.8)
        pair = robust_ai_label(
            seed_policy.generate_text, prompt, resp_a, resp_b,
            n_votes=config.rlaif_n_votes,
            consistency_threshold=config.rlaif_consistency_threshold,
        )
        if pair is not None:
            preference_dataset.append(pair)

    # === STAGE 4: Reward model training ===
    # (reward_model_train(preference_dataset) — omitted)

    # === STAGE 5: Rejection sampling (verifiable tasks) ===
    rft_training_data = []
    for iteration in range(config.rft_num_iterations):
        accepted = rejection_sampling_iteration(
            seed_policy, tokenizer, verifiable_prompts,
            reward_fn, k=config.rft_k, temperature=config.rft_temperature,
        )
        rft_training_data.extend(accepted)
        # (sft_finetune(seed_policy, accepted) — omitted; updates seed_policy in place)

    # === STAGE 6: DPO on RLAIF pairs ===
    # (dpo_train(seed_policy, preference_dataset, beta=config.dpo_beta) — omitted)

    # === STAGE 7: Human QA spot-check ===
    # (sample config.human_qa_fraction of outputs, route to human review)

    return seed_policy  # the aligned model
```

---

!!! key "Key Takeaways"
    - Constitutional AI replaces the human rater with an AI judge steered by an explicit principle set. The pipeline has two stages: supervised critique-revision (SL-CAI) and RL from AI feedback (RLAIF), each eliminating a different bottleneck.
    - RLAIF can produce preference labels orders of magnitude faster and cheaper than human annotation. Position bias and label noise are mitigated by response-order randomization and multi-vote consistency filtering.
    - Self-rewarding language models use the same model as both policy and judge. The virtuous cycle improves both simultaneously, but risks self-grade inflation — mitigation includes mixing in external data and entropy bonuses.
    - Rejection sampling fine-tuning (RFT) is the simplest form of self-improvement: sample $k$ responses, keep the correct ones, fine-tune. STaR adds hint-conditioned rationalization to provide signal even when the model fails; ReST adds a curriculum threshold that rises each iteration.
    - SPIN (self-play fine-tuning) uses the model's own outputs as rejected examples, directly optimizing the gap between the current policy distribution and the target human data distribution.
    - Weak-to-strong generalization shows that a capable student trained on noisy weak labels can exceed the supervisor. Bootstrapping (intermediate model, then strong model) and confidence-weighted loss close the elicitation gap further.
    - In production, a small but carefully targeted human annotation budget — spent on hard cases, adversarial examples, and calibration anchors — provides the quality floor that AI labeling alone cannot guarantee.
    - The alignment data flywheel: seed human data → SFT → RLAIF → RL → better model → better RLAIF labels → repeat. Each round should raise the reward threshold (curriculum) to avoid stagnation.

---

!!! sota "State of the Art & Resources (2026)"
    Constitutional AI and RLAIF have become the default scaling strategy for alignment data in frontier labs: AI-generated preference labels now outnumber human labels by orders of magnitude, with constitutions providing auditable constraints on what the AI judge measures. Self-improvement loops (STaR, ReST, SPIN, self-rewarding LMs) are now standard post-training components, and the scalable-oversight question — how to supervise models that exceed human competence — is an active research frontier.

    **Foundational work**

    - [Bai et al., *Constitutional AI: Harmlessness from AI Feedback* (2022)](https://arxiv.org/abs/2212.08073) — original CAI paper defining the two-stage critique-revision + RLAIF pipeline that replaced human labelers at Anthropic.
    - [Zelikman et al., *STaR: Bootstrapping Reasoning With Reasoning* (2022)](https://arxiv.org/abs/2203.14465) — self-taught reasoner that generates rationales, filters by correctness, and fine-tunes iteratively; foundational for verifiable-reward self-improvement.

    **Recent advances (2023–2026)**

    - [Lee et al., *RLAIF vs. RLHF* (2023)](https://arxiv.org/abs/2309.00267) — controlled Google study showing AI-labeled preferences match human labels on summarization and dialogue; introduces position-debiasing protocol.
    - [Gulcehre et al., *Reinforced Self-Training (ReST) for Language Modeling* (2023)](https://arxiv.org/abs/2308.08998) — Grow/Improve curriculum with rising reward thresholds; shows offline rejection sampling is more compute-efficient than online RL.
    - [Burns et al., *Weak-to-Strong Generalization* (2023)](https://arxiv.org/abs/2312.09390) — OpenAI study demonstrating that a strong student trained on weak labels exceeds its supervisor; central paper on scalable oversight.
    - [Yuan et al., *Self-Rewarding Language Models* (2024)](https://arxiv.org/abs/2401.10020) — same model acts as policy and judge; iterative DPO updates improve both roles simultaneously.
    - [Chen et al., *Self-Play Fine-Tuning (SPIN)* (2024)](https://arxiv.org/abs/2401.01335) — uses the model's own previous-iteration outputs as the rejected baseline in DPO, directly closing the gap to human data.
    - [Wu et al., *Meta-Rewarding Language Models* (2024)](https://arxiv.org/abs/2407.19594) — adds a meta-judge role to fix reward saturation in self-rewarding loops; improves AlpacaEval 2 win rate from 22.9 % to 39.4 % on Llama-3-8B.

    **Open-source & tools**

    - [huggingface/trl](https://huggingface.co/docs/trl/index) — full post-training library with SFT, DPO, PPO, GRPO, and reward-modeling trainers; the standard implementation base for RLAIF pipelines.
    - [openai/weak-to-strong](https://github.com/openai/weak-to-strong) — reference codebase for weak-to-strong generalization experiments across NLP and vision tasks.

    **Go deeper**

    - [Anthropic, *Claude's Constitution*](https://www.anthropic.com/constitution) — the publicly released principle set used in Claude's CAI training; CC0 licensed and directly actionable as a starting point for your own constitution.

## Further Reading

- Bai et al., "Constitutional AI: Harmlessness from AI Feedback," Anthropic, 2022. The original CAI paper defining the critique-revision and RLAIF pipeline.
- Lee et al., "RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback," Google, 2023. Controlled comparison of human vs. AI preference labels; introduces position-debiasing protocol.
- Zeiler et al., "STaR: Bootstrapping Reasoning With Reasoning," NeurIPS 2022. Self-taught reasoner with hint-conditioned rationalization.
- Gulcehre et al., "Reinforced Self-Training (ReST) for Language Modeling," Google DeepMind, 2023. Grow-Improve curriculum for offline rejection sampling.
- Yuan et al., "Self-Rewarding Language Models," Meta AI, 2024. The joint policy+judge training loop with DPO updates.
- Chen et al., "Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models," 2024. SPIN: using the model's own outputs as the rejected baseline.
- Burns et al., "Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision," OpenAI, 2023. Empirical study of the scalable oversight setting with bootstrapping and confidence-weighting interventions.
- Anthropic Model Card and Constitution — publicly released at anthropic.com, describing the principles used in Claude's CAI training.
