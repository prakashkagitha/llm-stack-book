"""
Test harness for content/11-evaluation/03-eval-harnesses.md

Runs the CPU-runnable Python blocks from the chapter, in order, with
minimal glue/fixtures so each one actually executes (not just parses).

Blocks tested (verbatim from the book, faithfully reproduced):
  - block #1  (line ~108): doc_to_text / process_results (generation task)
  - block #2  (line ~137): build_prompt (K-shot prompt builder pseudocode)
  - block #5  (line ~265): normalize_answer / extract_mc_answer
  - block #9  (line ~364): entity_f1 (+ MedNERTask, guarded -- needs lm_eval)
  - block #16 (line ~948): mcnemar_test (scipy.stats, guarded)
  - block #17 (line ~998): bootstrap_ci

Blocks explicitly SKIPPED (per the harness's heuristic classification):
  #0, #3, #8, #14  = shell snippets (pip install / lm_eval CLI / helm CLI /
                      results directory listing) -- not python.
  #4               = needs-net (chat-template example calls
                      AutoTokenizer.from_pretrained("meta-llama/..."), a
                      gated model requiring network + auth).
  #6, #7, #13, #18 = non-python (JSON data row, YAML task config,
                      JSON sample-log row, GitHub Actions YAML).
  #10, #11         = needs real transformers models. minimal_harness.py's
                      evaluate()/score_choices() and smoke_test.py both call
                      AutoModelForCausalLM.from_pretrained("gpt2") /
                      AutoTokenizer.from_pretrained("gpt2"), which requires a
                      network fetch (no local cache guaranteed in CI) --
                      forbidden by the no-network rule. transformers is also
                      not in the guaranteed-import list. Skipped.
  #12              = needs-net (the independent-recompute snippet reruns
                      smoke_test(), same transformers/network dependency
                      as #10/#11).
  #15              = non-python (reproducibility checklist, plain text).
"""

import random
import re
import string
from typing import Callable

import numpy as np


# ============================================================================
# Block #1 (line ~108): generation-based task definition snippet
# ============================================================================

def doc_to_text(doc):
    """Format a single document into a prompt string."""
    return (
        "Question: " + doc["question"] +
        "\nLet's think step by step.\nAnswer:"
    )


def process_results(doc, results):
    """Extract the numeric answer from a chain-of-thought generation."""
    import re
    # results[0] is the generated string
    gen = results[0]
    # Look for the last number in the generation
    matches = re.findall(r"[-+]?\d*\.?\d+", gen)
    if matches:
        predicted = float(matches[-1])
        gold = float(doc["answer"])
        return {"exact_match": predicted == gold}
    return {"exact_match": 0.0}


def _test_block1():
    doc = {"question": "What is 3 plus 4?", "answer": "7"}
    prompt = doc_to_text(doc)
    assert prompt == (
        "Question: What is 3 plus 4?\nLet's think step by step.\nAnswer:"
    )

    # Correct chain-of-thought generation
    gen_correct = "3 + 4 = 7. The answer is 7."
    r = process_results(doc, [gen_correct])
    assert r == {"exact_match": True}

    # Wrong final number
    gen_wrong = "3 + 4 = 6. The answer is 6."
    r_wrong = process_results(doc, [gen_wrong])
    assert r_wrong == {"exact_match": False}

    # No number at all
    r_empty = process_results(doc, ["I don't know."])
    assert r_empty == {"exact_match": 0.0}

    print("block #1 OK:", r, r_wrong, r_empty)


# ============================================================================
# Block #2 (line ~137): K-shot prompt builder (pseudocode of harness internals)
# ============================================================================

def build_prompt(task, doc, k, fewshot_docs):
    """Build a K-shot prompt for a document."""
    parts = []

    # System prompt (if the task defines one)
    if task.has_system_prompt():
        parts.append(task.system_prompt())

    # K few-shot examples
    for fewshot_doc in fewshot_docs[:k]:
        # doc_to_text formats the question
        # doc_to_target formats the gold answer
        parts.append(task.doc_to_text(fewshot_doc) +
                     task.doc_to_target(fewshot_doc))

    # The actual test document (no answer appended)
    parts.append(task.doc_to_text(doc))

    return task.fewshot_delimiter().join(parts)


class _ToyTask:
    """Minimal stand-in for the harness's Task interface -- just enough
    surface area (has_system_prompt/system_prompt/doc_to_text/doc_to_target/
    fewshot_delimiter) to drive build_prompt() end to end."""

    def has_system_prompt(self):
        return True

    def system_prompt(self):
        return "You are a helpful assistant."

    def doc_to_text(self, doc):
        return f"Question: {doc['question']}\nAnswer:"

    def doc_to_target(self, doc):
        return f" {doc['answer']}"

    def fewshot_delimiter(self):
        return "\n\n"


def _test_block2():
    task = _ToyTask()
    fewshot_docs = [
        {"question": "2+2?", "answer": "4"},
        {"question": "3+3?", "answer": "6"},
    ]
    test_doc = {"question": "5+5?", "answer": "10"}

    prompt_2shot = build_prompt(task, test_doc, k=2, fewshot_docs=fewshot_docs)
    assert "You are a helpful assistant." in prompt_2shot
    assert "2+2?" in prompt_2shot and " 4" in prompt_2shot
    assert "3+3?" in prompt_2shot and " 6" in prompt_2shot
    assert prompt_2shot.endswith("Question: 5+5?\nAnswer:")

    prompt_0shot = build_prompt(task, test_doc, k=0, fewshot_docs=fewshot_docs)
    assert "2+2?" not in prompt_0shot
    assert prompt_0shot.endswith("Question: 5+5?\nAnswer:")

    print("block #2 OK, 2-shot prompt:\n", prompt_2shot)


# ============================================================================
# Block #5 (line ~265): normalize_answer / extract_mc_answer
# ============================================================================

def normalize_answer(s: str) -> str:
    """Normalize a string answer for comparison.

    Follows the same normalization as the SQuAD evaluation script,
    used widely in QA benchmarks.
    """
    # Lowercase
    s = s.lower()
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Remove punctuation
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    # Collapse whitespace
    s = " ".join(s.split())
    return s


def extract_mc_answer(generation: str, choices: list[str]) -> str | None:
    """Extract a multiple-choice answer from a free-form generation.

    Handles both letter-based ("A", "B") and text-based answers.
    Returns the matched choice text, or None if no match.
    """
    gen = generation.strip()

    # Try to match a leading letter like "A" or "A."
    letter_match = re.match(r"^([A-Da-d])[\.\):\s]?", gen)
    if letter_match:
        idx = ord(letter_match.group(1).upper()) - ord("A")
        if 0 <= idx < len(choices):
            return choices[idx]

    # Try to match the choice text directly
    gen_norm = normalize_answer(gen)
    for choice in choices:
        if normalize_answer(choice) in gen_norm:
            return choice

    return None


def _test_block5():
    assert normalize_answer("  The Paris.  ") == "paris"
    assert normalize_answer("A diffusion") == "diffusion"

    choices = ["London", "Berlin", "Paris", "Madrid"]
    # Letter-based
    assert extract_mc_answer("C) Paris", choices) == "Paris"
    assert extract_mc_answer("c", choices) == "Paris"
    # Text-based
    assert extract_mc_answer("I think the answer is Paris.", choices) == "Paris"
    # No match
    assert extract_mc_answer("I have no idea.", choices) is None

    print("block #5 OK")


# ============================================================================
# Block #9 (line ~364): custom NER task -- entity_f1 metric (+ MedNERTask,
# which subclasses lm_eval's ConfigurableTask; lm_eval is not a guaranteed
# CI dependency so that import and the class using it are guarded/optional).
# ============================================================================

try:
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.api.metrics import mean
except Exception:
    ConfigurableTask = None
    mean = None


def entity_f1(items):
    """Compute macro-averaged entity-level F1 over a list of (pred, gold) pairs.

    Args:
        items: list of (prediction_set, gold_set) tuples, where each set
               contains normalized entity strings.

    Returns:
        Macro-averaged F1 score in [0, 1].
    """
    f1_scores = []
    for pred_set, gold_set in items:
        if not gold_set:
            # No gold entities: perfect if model also predicts nothing
            f1_scores.append(1.0 if not pred_set else 0.0)
            continue

        true_pos = len(pred_set & gold_set)
        precision = true_pos / len(pred_set) if pred_set else 0.0
        recall    = true_pos / len(gold_set)

        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))

    return sum(f1_scores) / len(f1_scores)


if ConfigurableTask is not None:
    class MedNERTask(ConfigurableTask):
        """Medical NER task with entity-level F1 metric."""

        VERSION = 1
        DATASET_PATH = "json"
        DATASET_NAME = None

        def doc_to_text(self, doc):
            return f"Extract all medical entities from this text:\n{doc['text']}\nEntities:"

        def doc_to_target(self, doc):
            # Return the list as a comma-separated string for generation
            return ", ".join(doc["entities"])

        def process_results(self, doc, results):
            # Parse the generation into a set of entities
            generation = results[0].strip()
            pred_entities = {
                e.strip().lower()
                for e in generation.split(",")
                if e.strip()
            }
            gold_entities = {e.lower() for e in doc["entities"]}
            return {
                "entity_f1": (pred_entities, gold_entities)
            }

        def aggregation(self):
            return {"entity_f1": entity_f1}

        def higher_is_better(self):
            return {"entity_f1": True}
else:
    MedNERTask = None  # SKIP(optional-dep): lm_eval not installed in CI


def _test_block9():
    # entity_f1 is standalone logic (no lm_eval dependency) -- exercise it
    # directly with tiny fixtures shaped like what process_results() would
    # produce: (predicted_entity_set, gold_entity_set) pairs.
    pred1 = {"aspirin", "ibuprofen"}
    gold1 = {"aspirin", "ibuprofen"}
    pred2 = {"aspirin"}
    gold2 = {"aspirin", "ibuprofen"}
    pred3 = set()
    gold3 = set()

    score = entity_f1([(pred1, gold1), (pred2, gold2), (pred3, gold3)])
    # item1: F1=1.0 (exact match)
    # item2: precision=1/1=1.0, recall=1/2=0.5 -> F1=2*1*0.5/1.5=2/3
    # item3: no gold, no pred -> F1=1.0
    expected = (1.0 + (2 * 1.0 * 0.5 / 1.5) + 1.0) / 3
    assert abs(score - expected) < 1e-9

    if MedNERTask is None:
        print("block #9: entity_f1 OK (score=%.4f); MedNERTask class "
              "SKIPPED(optional-dep): lm_eval not installed" % score)
    else:
        print("block #9: entity_f1 OK (score=%.4f); MedNERTask class also "
              "available (lm_eval installed)" % score)


# ============================================================================
# Block #16 (line ~948): McNemar's test (scipy.stats -- guarded optional dep,
# not in the guaranteed-import list even though sklearn commonly pulls it in)
# ============================================================================

try:
    from scipy.stats import chi2, binomtest
except Exception:
    chi2 = None
    binomtest = None


def mcnemar_test(results_a: list[int], results_b: list[int]) -> dict:
    """McNemar's test for two paired binary result sequences.

    Uses the exact two-sided binomial test when discordant pairs are few
    (< 25) and the continuity-corrected chi-square approximation otherwise.
    See 11.6 (Statistical Rigor) for the full derivation and reference code.
    """
    assert len(results_a) == len(results_b), "Must be paired on the same docs"

    n_01 = sum(a == 0 and b == 1 for a, b in zip(results_a, results_b))  # B right, A wrong
    n_10 = sum(a == 1 and b == 0 for a, b in zip(results_a, results_b))  # A right, B wrong
    n_disc = n_01 + n_10

    if n_disc == 0:                    # models never disagree -> no evidence
        statistic, p_value = None, 1.0
    elif n_disc < 25:                  # too few pairs for chi-square: exact test
        # Under H0 each discordant pair is a fair coin: n_01 ~ Binomial(n_disc, 0.5).
        statistic = None
        p_value = binomtest(n_01, n_disc, 0.5, alternative="two-sided").pvalue
    else:
        # Yates continuity correction, clamped at 0 so n_01 == n_10 gives
        # exactly 0 (an uncorrected (|0|-1)**2 would report a spurious 1/n_disc).
        statistic = max(abs(n_01 - n_10) - 1, 0) ** 2 / n_disc
        p_value = float(chi2.sf(statistic, df=1))

    return {
        "chi2": round(statistic, 4) if statistic is not None else None,
        "p_value": round(p_value, 4),
        "n_01": n_01,  # B correct, A wrong
        "n_10": n_10,  # A correct, B wrong
        "acc_a": sum(results_a) / len(results_a),
        "acc_b": sum(results_b) / len(results_b),
    }


def _test_block16():
    if binomtest is None or chi2 is None:
        print("block #16 SKIPPED(optional-dep): scipy not available")
        return

    # Example usage (verbatim from the book)
    results_a = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]  # model A correct/wrong
    results_b = [1, 1, 1, 0, 0, 1, 1, 0, 1, 0]  # model B correct/wrong
    out = mcnemar_test(results_a, results_b)
    # Only 4 discordant pairs (n_disc = 4 < 25), so the exact binomial test runs
    # and no chi-square statistic is reported:
    assert out["chi2"] is None
    assert out["p_value"] == 1.0
    assert out["n_01"] == 2
    assert out["n_10"] == 2
    assert out["acc_a"] == 0.6
    assert out["acc_b"] == 0.6
    print("block #16 (exact-test branch) OK:", out)

    # Exercise the chi-square branch too (n_disc >= 25), with a real gap
    # between the two models.
    rng = random.Random(0)
    big_a = [1 if rng.random() < 0.60 else 0 for _ in range(200)]
    big_b = [1 if rng.random() < 0.75 else 0 for _ in range(200)]
    out_big = mcnemar_test(big_a, big_b)
    assert out_big["chi2"] is not None
    assert 0.0 <= out_big["p_value"] <= 1.0
    print("block #16 (chi-square branch) OK:", out_big)


# ============================================================================
# Block #17 (line ~998): bootstrap confidence intervals
# ============================================================================

def bootstrap_ci(
    scores: list[float],
    metric_fn: Callable[[list[float]], float] = np.mean,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute a bootstrap confidence interval for any metric.

    Args:
        scores: per-example metric values (e.g., list of 0/1 for accuracy)
        metric_fn: aggregation function (default: mean)
        n_bootstrap: number of bootstrap resamples
        alpha: significance level (0.05 -> 95% CI)
        seed: random seed for reproducibility

    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(seed)
    arr = np.array(scores)

    point_estimate = metric_fn(arr)

    # Resample with replacement n_bootstrap times
    boot_stats = []
    for _ in range(n_bootstrap):
        resample = rng.choice(arr, size=len(arr), replace=True)
        boot_stats.append(metric_fn(resample))

    boot_stats = np.array(boot_stats)
    lower = np.percentile(boot_stats, 100 * alpha / 2)
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))

    return point_estimate, lower, upper


def _test_block17():
    # Example: accuracy scores for 500 examples (book's exact fixture).
    # n_bootstrap is reduced from the book's 10_000 to 2_000 to keep the
    # test fast; the logic exercised is identical.
    scores = [1] * 370 + [0] * 130  # 74% accuracy
    pt, lo, hi = bootstrap_ci(scores, n_bootstrap=2000, seed=42)
    assert abs(pt - 0.740) < 1e-9
    assert lo < pt < hi
    # Book's reported 95% CI is roughly [0.703, 0.776]; allow slack since we
    # use fewer bootstrap resamples for speed.
    assert 0.65 < lo < 0.75
    assert 0.72 < hi < 0.82
    print(f"block #17 OK: acc = {pt:.3f}  95% CI: [{lo:.3f}, {hi:.3f}]")


# ============================================================================
# Run all
# ============================================================================

if __name__ == "__main__":
    _test_block1()
    _test_block2()
    _test_block5()
    _test_block9()
    _test_block16()
    _test_block17()
    print("\nAll testable blocks from 03-eval-harnesses.md executed successfully.")
