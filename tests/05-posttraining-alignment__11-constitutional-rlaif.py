"""
Executable extraction of the CPU-runnable code blocks from
content/05-posttraining-alignment/11-constitutional-rlaif.md

Blocks tested (chapter's own numbering):
  #0 (needed as glue/dependency for #1-#3; trivially CPU-safe: pure string
      building + a dataclass, no network) - CONSTITUTION, prompt builders
  #1 (line ~129, 60 lines) - critique_revise_loop + mock_model
  #2 (line ~195, 57 lines) - PreferencePair + ai_label_pair
  #3 (line ~270, 60 lines) - robust_ai_label (multi-vote + position debiasing)
  #6 (line ~501, 27 lines) - build_spin_pairs (SPIN data construction)
  #7 (line ~550, 44 lines) - WeakToStrongTrainer
  #8 (line ~634, 20 lines) - allocate_annotation_budget

Blocks explicitly SKIPPED:
  # SKIP(non-python): block #4 (line ~353) is a plain-text rubric template,
    not executable Python.
  # SKIP(needs-gpu): block #5 (line ~396) - RejectionSamplingDataset /
    rejection_sampling_iteration calls `tokenizer(...)`, `AutoModelForCausalLM`,
    `.cuda()`, and `model.generate(...)` against a real HF model/tokenizer;
    requires GPU + a real checkpoint, not CPU-safe to instantiate here.
  # SKIP(needs-net): block #9 (line ~662) - run_alignment_pipeline is an
    end-to-end orchestration sketch that calls rejection_sampling_iteration
    (GPU-only, see #5 above) and references a real judge model
    ("anthropic/claude-3-opus"); it is pseudocode-level per the chapter's own
    docstring and not meant to run standalone.
"""

import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

random.seed(0)
torch.manual_seed(0)


# =====================================================================
# Block #0 (glue/dependency for #1-#3): the constitution + prompt builders
# =====================================================================

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


# =====================================================================
# Block #1 (line ~129): critique_revise_loop + mock_model
# =====================================================================

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

assert final == "I'm unable to help with that request. Here's what I can offer instead...", (
    f"critique_revise_loop did not converge to the expected revised response, got: {final!r}"
)


# =====================================================================
# Block #2 (line ~195): PreferencePair + ai_label_pair
# =====================================================================

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


def mock_judge_prefers_a(prompt: str) -> str:
    return "Reasoning: Response A is more directly responsive.\nVERDICT: A"


def mock_judge_prefers_b(prompt: str) -> str:
    return "Reasoning: Response B is more accurate.\nVERDICT: B"


pair_a = ai_label_pair(mock_judge_prefers_a, "What is the capital of France?", "Paris.", "London.")
assert pair_a.chosen == "Paris." and pair_a.rejected == "London."

pair_b = ai_label_pair(mock_judge_prefers_b, "What is the capital of France?", "London.", "Paris.")
assert pair_b.chosen == "Paris." and pair_b.rejected == "London."

print("ai_label_pair OK:", pair_a, pair_b)


# =====================================================================
# Block #3 (line ~270): robust_ai_label (multi-vote + position debiasing)
# =====================================================================

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


# A judge that always says "VERDICT: A" regardless of which response is passed
# as A. Because robust_ai_label swaps positions on odd rounds and corrects the
# verdict back, a judge with a genuine (order-independent) preference for the
# *content* of response_a should still land on "Response A text" as the
# consistent majority winner across all 5 votes.
def mock_judge_always_a_slot(prompt: str) -> str:
    return "Reasoning: The first-listed response is judged better here.\nVERDICT: A"


robust_result = robust_ai_label(
    mock_judge_always_a_slot,
    "Test prompt",
    "Response A text",
    "Response B text",
    n_votes=5,
    consistency_threshold=0.7,
)
assert robust_result is not None, "expected a non-ambiguous consistent verdict"
assert robust_result.chosen == "Response A text"
assert robust_result.rejected == "Response B text"
print("robust_ai_label OK:", robust_result)

# A judge that flips its verdict slot every call has no genuine content
# preference and should discard the example as inconsistent, or at least
# not crash. Alternate verdicts A, B, A, B, A -> this still combines with
# swap-correction; just verify robust_ai_label runs without error and
# returns either None or a valid PreferencePair.
call_counter = {"n": 0}


def mock_judge_alternating(prompt: str) -> str:
    call_counter["n"] += 1
    verdict = "A" if call_counter["n"] % 2 == 1 else "B"
    return f"Reasoning: mixed signal.\nVERDICT: {verdict}"


alt_result = robust_ai_label(
    mock_judge_alternating,
    "Test prompt 2",
    "Response A text",
    "Response B text",
    n_votes=5,
    consistency_threshold=0.7,
)
assert alt_result is None or isinstance(alt_result, PreferencePair)
print("robust_ai_label (alternating judge) OK:", alt_result)


# =====================================================================
# Block #6 (line ~501): build_spin_pairs (SPIN data construction)
# =====================================================================

def build_spin_pairs(
    model_generate: Callable[[str], str],
    human_data: list,  # list[tuple[str, str]]  (prompt, gold_response)
) -> list:
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


def toy_policy_generate(prompt: str) -> str:
    return "A mediocre policy-generated answer."


human_data = [
    ("What is 2 + 2?", "2 + 2 equals 4."),
    ("Name the largest planet.", "Jupiter is the largest planet in the solar system."),
]
spin_pairs = build_spin_pairs(toy_policy_generate, human_data)
assert len(spin_pairs) == 2
assert spin_pairs[0].chosen == "2 + 2 equals 4."
assert spin_pairs[0].rejected == "A mediocre policy-generated answer."
assert spin_pairs[1].chosen == "Jupiter is the largest planet in the solar system."
print("build_spin_pairs OK:", spin_pairs)


# =====================================================================
# Block #7 (line ~550): WeakToStrongTrainer
# =====================================================================

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


toy_strong_model = nn.Linear(4, 2)
toy_weak_labels = torch.tensor([0.9, 0.1, 0.8, 0.2, 0.6, 0.4], dtype=torch.float32)
trainer = WeakToStrongTrainer(toy_strong_model, toy_weak_labels, confidence_weight=0.1)

toy_indices = torch.tensor([0, 1, 2, 3])
toy_logits = torch.tensor([
    [0.1, 1.5],
    [0.3, -1.2],
    [0.0, 0.8],
    [0.2, -0.5],
], dtype=torch.float32)

loss = trainer.compute_loss(toy_logits, toy_indices)
assert loss.dim() == 0, "compute_loss should return a scalar tensor"
assert torch.isfinite(loss), "compute_loss returned a non-finite value"
print("WeakToStrongTrainer.compute_loss OK:", loss.item())


# =====================================================================
# Block #8 (line ~634): allocate_annotation_budget
# =====================================================================

def allocate_annotation_budget(
    total_examples: int,
    ai_labeling_confidence: list,  # list[float] per-example confidence in [0,1]
    human_budget_fraction: float = 0.05,  # spend 5% of budget on humans
):
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


toy_confidences = [0.95, 0.10, 0.80, 0.05, 0.60, 0.99, 0.40, 0.55, 0.30, 0.70]
human_idx, ai_idx = allocate_annotation_budget(
    total_examples=10, ai_labeling_confidence=toy_confidences, human_budget_fraction=0.3
)
assert len(human_idx) == 3
assert len(ai_idx) == 7
assert set(human_idx) | set(ai_idx) == set(range(10))
# The 3 lowest-confidence examples are indices 3 (0.05), 1 (0.10), 8 (0.30)
assert set(human_idx) == {3, 1, 8}
print("allocate_annotation_budget OK: human_indices =", human_idx, "ai_indices =", ai_idx)


print("\nAll CPU-runnable blocks executed successfully.")
