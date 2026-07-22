"""
Executable test for content/11-evaluation/01-eval-landscape.md

Tests the 4 CPU-runnable Python blocks from the chapter:
  - block #1 (~line 131): n-gram contamination detection
  - block #2 (~line 179): completion_probe memorization test
  - block #3 (~line 319): multiple-choice answer extraction + batch scoring
  - block #4 (~line 409): position-bias variant generation

All four blocks are pure standard-library Python (collections, difflib,
re, itertools) -- no network, no GPU, no third-party deps. They are
concatenated in chapter order (later blocks are independent of earlier
ones here, but we preserve order per the instructions) and then each is
exercised with a tiny, honest fixture that actually calls every
function/class the block defines.
"""

from collections import Counter
import sys


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
# Block #1 (verbatim from chapter, ~line 131): n-gram contamination detection
# ---------------------------------------------------------------------------

from collections import Counter
from typing import List


def build_ngram_set(text: str, n: int = 13) -> set:
    """Extract all n-grams (by word) from a string."""
    tokens = text.lower().split()
    return set(
        " ".join(tokens[i : i + n])
        for i in range(len(tokens) - n + 1)
    )


def flag_contaminated(
    test_questions: List[str],
    training_docs: List[str],
    n: int = 13,
    threshold: float = 0.5,
) -> List[bool]:
    """
    Returns True for each test question that has >= threshold fraction of
    its n-grams appearing in any training document.

    NOTE: This is O(|train| * |test|) in the worst case; in practice you
    build a Bloom filter or inverted index over training n-grams.
    """
    # Build the full training n-gram set (expensive; in practice use a Bloom filter)
    train_ngrams: set = set()
    for doc in training_docs:
        train_ngrams |= build_ngram_set(doc, n)

    contaminated = []
    for q in test_questions:
        q_ngrams = build_ngram_set(q, n)
        if len(q_ngrams) == 0:
            contaminated.append(False)
            continue
        overlap = len(q_ngrams & train_ngrams) / len(q_ngrams)
        contaminated.append(overlap >= threshold)
    return contaminated


# ---------------------------------------------------------------------------
# Block #2 (verbatim from chapter, ~line 179): memorization completion probe
# ---------------------------------------------------------------------------

from difflib import SequenceMatcher
from typing import Callable, Dict, List


def _token_lcs_similarity(reference: str, candidate: str) -> float:
    """Longest common contiguous token run, normalized by reference length.

    Lowercased whitespace tokenization so trivial case/spacing differences
    do not hide a near-verbatim match. Returns a value in [0, 1].
    """
    ref = reference.lower().split()
    cand = candidate.lower().split()
    if not ref:
        return 0.0
    sm = SequenceMatcher(a=ref, b=cand, autojunk=False)
    match = sm.find_longest_match(0, len(ref), 0, len(cand))
    return match.size / len(ref)


def completion_probe(
    items: List[str],
    generate: Callable[[str], str],
    split_frac: float = 0.5,
    threshold: float = 0.75,
) -> Dict[str, float]:
    """
    Memorization probe by first-half completion.

    For each benchmark item, split it into a prefix (the first `split_frac`
    of its whitespace tokens) and a reference suffix (the rest). Ask the
    model to continue the prefix, then measure how close the continuation is
    to the true suffix. A high near-verbatim rate is evidence the item was
    in the training corpus.

    `generate(prefix) -> continuation` wraps your backend (HF
    `model.generate`, vLLM, an API); decode GREEDILY (temperature 0) so the
    probe is reproducible. An item counts as 'memorized' when the token-LCS
    similarity between the continuation and the reference suffix is
    >= threshold. threshold ~0.75 flags near-verbatim recall while
    tolerating a few paraphrased tokens; raise it toward 0.9 for a stricter
    verbatim-only test.

    Returns {'memorized_rate', 'mean_similarity', 'n'}.
    """
    sims: List[float] = []
    memorized = 0
    for text in items:
        toks = text.split()
        if len(toks) < 4:
            continue  # too short to split meaningfully
        cut = max(1, int(len(toks) * split_frac))
        prefix = " ".join(toks[:cut])
        reference = " ".join(toks[cut:])
        continuation = generate(prefix)
        sim = _token_lcs_similarity(reference, continuation)
        sims.append(sim)
        if sim >= threshold:
            memorized += 1
    n = len(sims)
    return {
        "memorized_rate": memorized / n if n else 0.0,
        "mean_similarity": sum(sims) / n if n else 0.0,
        "n": n,
    }


# Verification (no model needed): an oracle that returns the exact suffix
# gives similarity 1.0 on every item -> memorized_rate 1.0. A generator
# that returns unrelated text gives memorized_rate ~ 0.0. Calibrate the
# threshold on a KNOWN-CLEAN control set (e.g. freshly written items the
# model cannot have seen) and treat a benchmark whose memorized_rate is far
# above that control baseline as contaminated.
#
# oracle = lambda prefix: "placeholder"  # replace with real second half
# items = ["the quick brown fox jumps over the lazy dog"]
# def _oracle(prefix, full=items[0]):
#     return full[len(prefix):]
# assert completion_probe(items, _oracle, threshold=0.75)["memorized_rate"] == 1.0
# assert completion_probe(items, lambda p: "zzz", threshold=0.75)["memorized_rate"] == 0.0


# ---------------------------------------------------------------------------
# Block #3 (verbatim from chapter, ~line 319): MC answer extraction + scoring
# ---------------------------------------------------------------------------

import re
from typing import Optional


def extract_mc_answer(output: str) -> Optional[str]:
    """
    Robustly extract a multiple-choice answer letter from model output.
    Handles common formats: "A", "(A)", "Answer: A", "The answer is A.", etc.
    Returns None if no single unambiguous letter is found.
    """
    # Try explicit "answer is X" pattern first (most reliable)
    m = re.search(r"\bthe answer is\s+\(?([A-Da-d])\)?", output, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Try "Answer: X" pattern
    m = re.search(r"\banswer\s*:\s*\(?([A-Da-d])\)?", output, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Fall back to a lone letter on its own line or at end of output
    lines = output.strip().splitlines()
    for line in reversed(lines):  # check from end first
        m = re.fullmatch(r"\s*\(?([A-Da-d])\)?\s*[.]?", line.strip())
        if m:
            return m.group(1).upper()

    # Last resort: find any capital A-D that appears
    letters = re.findall(r"\b([A-D])\b", output)
    if len(set(letters)) == 1:
        return letters[0]

    return None  # ambiguous or missing


def evaluate_mc_batch(
    predictions: list[str],
    gold_labels: list[str],
) -> dict:
    """
    Evaluate a batch of multiple-choice predictions.

    Args:
        predictions: raw model output strings.
        gold_labels: correct answer letters, e.g. ["A", "C", "B", ...].

    Returns:
        dict with 'accuracy', 'null_rate' (fraction where answer was not
        extracted), and 'corrected_accuracy' (guessing-corrected, k=4).

    NOTE: this scores predictions that ALREADY exist, so it cannot measure
    position bias. Position bias is a property of how the model responds
    when the SAME options are presented in a different order; detecting it
    requires re-rendering each question with permuted choices and calling
    the model again (see make_position_bias_variants below). Shuffling the
    (prediction, gold) pairs post hoc only reorders examples -- every
    pred/gold pairing, and therefore the accuracy, stays identical.
    """
    correct = 0
    wrong = 0
    null = 0

    for pred, gold in zip(predictions, gold_labels):
        extracted = extract_mc_answer(pred)
        if extracted is None:
            null += 1
        elif extracted == gold.upper():
            correct += 1
        else:
            wrong += 1

    total = len(predictions)
    non_null = total - null
    accuracy = correct / total if total > 0 else 0.0
    corrected = (correct - wrong / 3) / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "corrected_accuracy": corrected,
        "null_rate": null / total if total > 0 else 0.0,
        "n": total,
        "correct": correct,
        "wrong": wrong,
        "null": null,
    }


# ---------------------------------------------------------------------------
# Block #4 (verbatim from chapter, ~line 409): position-bias variants
# ---------------------------------------------------------------------------

import itertools
from typing import List, Optional, Tuple


def make_position_bias_variants(
    question: str,
    choices: List[str],          # option TEXTS in original order: choices[0] == "A"
    gold_index: int,             # index into choices of the correct option
    max_variants: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """
    Build re-rendered prompts with the answer options permuted, so a caller
    can re-run the model on each and see whether accuracy depends on WHERE
    the correct option sits. This is how you actually test position bias --
    it needs fresh model calls, not post-hoc relabeling.

    Returns a list of (prompt, gold_letter) pairs, one per permutation:
      - prompt: the question with choices relabeled A., B., ... in the
        permuted order
      - gold_letter: the letter the correct option now occupies

    Usage: for each (prompt, gold_letter), call your model, run
    extract_mc_answer on the output, and tally accuracy grouped by
    gold_letter. A model with no position bias scores the same for every
    gold_letter; a large spread (e.g. much higher when the answer is 'A')
    is position bias.
    """
    letters = "ABCDEFGHIJ"
    perms = list(itertools.permutations(range(len(choices))))
    if max_variants is not None:
        perms = perms[:max_variants]
    variants: List[Tuple[str, str]] = []
    for perm in perms:
        lines = [question, ""]
        new_gold = None
        for new_pos, orig_idx in enumerate(perm):
            lines.append(f"{letters[new_pos]}. {choices[orig_idx]}")
            if orig_idx == gold_index:
                new_gold = letters[new_pos]
        variants.append(("\n".join(lines), new_gold))
    return variants


# Sanity check: 4 options -> 24 permutations; the correct option lands in
# each of A/B/C/D exactly 6 times, so an unbiased model scores equally.
# vs = make_position_bias_variants("Q?", ["w", "x", "y", "z"], gold_index=0)
# assert len(vs) == 24
# from collections import Counter
# assert Counter(g for _, g in vs) == {"A": 6, "B": 6, "C": 6, "D": 6}


# ===========================================================================
# Execution / fixtures: actually run every function above with tiny inputs.
# ===========================================================================


def test_block1_ngram_contamination():
    _section("Block #1: n-gram contamination detection")

    # Craft a 13-word test question that appears verbatim in a training doc
    # (contaminated) and one that does not (clean).
    shared_sentence = (
        "the quick brown fox jumps over the lazy dog while the sun sets slowly"
    )
    assert len(shared_sentence.split()) >= 13

    training_docs = [
        "Some unrelated preamble text goes here for padding purposes only today.",
        f"A blog post discussing math problems: {shared_sentence} and then some more discussion follows after that.",
    ]
    test_questions = [
        shared_sentence,  # should be flagged contaminated (verbatim overlap)
        "a completely novel sentence about zebras painting watercolor murals under moonlight tonight quietly",  # clean
    ]

    ngrams = build_ngram_set(shared_sentence, n=13)
    assert len(ngrams) >= 1, "expected at least one 13-gram from a 14-word sentence"

    flags = flag_contaminated(test_questions, training_docs, n=13, threshold=0.5)
    assert flags == [True, False], f"unexpected contamination flags: {flags}"
    print("build_ngram_set + flag_contaminated OK ->", flags)


def test_block2_completion_probe():
    _section("Block #2: completion_probe memorization test")

    items = ["the quick brown fox jumps over the lazy dog today"]

    def _oracle(prefix, full=items[0]):
        # Returns the exact true suffix -> should be detected as memorized.
        toks = full.split()
        cut = max(1, int(len(toks) * 0.5))
        reference = " ".join(toks[cut:])
        return reference

    def _unrelated(prefix):
        return "zzz completely different words here"

    oracle_result = completion_probe(items, _oracle, threshold=0.75)
    assert oracle_result["memorized_rate"] == 1.0, oracle_result
    assert oracle_result["n"] == 1

    unrelated_result = completion_probe(items, _unrelated, threshold=0.75)
    assert unrelated_result["memorized_rate"] == 0.0, unrelated_result

    print("completion_probe (oracle) ->", oracle_result)
    print("completion_probe (unrelated) ->", unrelated_result)

    # also exercise _token_lcs_similarity directly
    sim_exact = _token_lcs_similarity("hello world foo", "hello world foo")
    assert sim_exact == 1.0
    sim_none = _token_lcs_similarity("hello world foo", "totally different text")
    assert sim_none == 0.0
    print("_token_lcs_similarity OK -> exact:", sim_exact, "none:", sim_none)


def test_block3_mc_extraction_and_scoring():
    _section("Block #3: MC answer extraction + batch scoring")

    # Exercise every branch of extract_mc_answer.
    assert extract_mc_answer("The answer is B.") == "B"
    assert extract_mc_answer("Answer: C") == "C"
    assert extract_mc_answer("Some reasoning...\n(D)") == "D"
    assert extract_mc_answer("I believe the correct choice is A somewhere in here") == "A"
    assert extract_mc_answer("I am not sure at all, this is rambling text") is None

    predictions = [
        "The answer is A.",   # correct (gold A)
        "Answer: B",           # wrong (gold C)
        "no clear letter here",  # null
        "(D)",                 # correct (gold D)
    ]
    gold_labels = ["A", "C", "B", "D"]

    result = evaluate_mc_batch(predictions, gold_labels)
    assert result["n"] == 4
    assert result["correct"] == 2
    assert result["wrong"] == 1
    assert result["null"] == 1
    assert abs(result["accuracy"] - 0.5) < 1e-9
    assert abs(result["null_rate"] - 0.25) < 1e-9
    expected_corrected = (2 - 1 / 3) / 4
    assert abs(result["corrected_accuracy"] - expected_corrected) < 1e-9
    print("evaluate_mc_batch ->", result)


def test_block4_position_bias_variants():
    _section("Block #4: position-bias variant generation")

    variants = make_position_bias_variants(
        "Which of these is a fruit?", ["wrench", "apple", "hammer", "screwdriver"],
        gold_index=1,
    )
    # 4 choices -> 4! = 24 permutations
    assert len(variants) == 24
    gold_letter_counts = Counter(g for _, g in variants)
    assert gold_letter_counts == {"A": 6, "B": 6, "C": 6, "D": 6}, gold_letter_counts

    # spot-check structure of one variant
    prompt, gold_letter = variants[0]
    assert "Which of these is a fruit?" in prompt
    assert gold_letter in "ABCD"
    # the gold letter's line in the prompt must contain "apple"
    gold_line = [
        line for line in prompt.splitlines() if line.startswith(f"{gold_letter}.")
    ][0]
    assert "apple" in gold_line

    # max_variants truncation
    truncated = make_position_bias_variants(
        "Q?", ["w", "x", "y", "z"], gold_index=0, max_variants=5
    )
    assert len(truncated) == 5

    print("make_position_bias_variants OK -> 24 permutations, balanced:", gold_letter_counts)


def main():
    test_block1_ngram_contamination()
    test_block2_completion_probe()
    test_block3_mc_extraction_and_scoring()
    test_block4_position_bias_variants()
    print("\nAll blocks executed successfully.")


if __name__ == "__main__":
    main()
