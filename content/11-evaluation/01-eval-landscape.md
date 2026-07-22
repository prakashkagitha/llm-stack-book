# 11.1 The Evaluation Problem & Benchmark Landscape

Measuring whether an LLM is good is surprisingly hard. Harder than training it. When a model scores 90 % on a knowledge benchmark, you want to know: does it *know* more, or does it exploit test artifacts? When a newer model beats a prior one on MMLU, should you deploy it? When a model aces GSM8K but fails on a slightly rephrased version of the same arithmetic problem, what does that tell you?

This chapter builds the conceptual toolkit for evaluation literacy. We cover *why* eval is a genuinely unsolved problem, survey the major benchmarks a practitioner will encounter, dissect the failure modes (contamination, saturation, multiple-choice shortcuts), and discuss the gap between benchmark numbers and actual usefulness. The chapter that follows — [LLM-as-a-Judge & Automated Evaluation](../11-evaluation/02-llm-as-judge.html) — covers the judge-model paradigm; [Building Eval Harnesses](../11-evaluation/03-eval-harnesses.html) covers the engineering side; and [Reasoning, Coding & Agentic Evals](../11-evaluation/04-reasoning-coding-agentic-evals.html) goes deep on code and agent evaluation.

---

## The Core Difficulty of LLM Evaluation

### What Makes a Good Evaluator?

A good evaluator for a statistical model has three properties: it is *valid* (it measures what you care about), *reliable* (two runs or two evaluators give the same answer), and *sensitive* (it can detect real differences at the scale you care about). LLM benchmarks routinely fail at least one of these.

**Validity failures.** A question asking "What is the capital of France?" tests *whether the answer appears in training data* more than it tests any deeper capability. Multiple-choice benchmarks let models succeed via elimination and surface-form pattern matching rather than genuine reasoning. Open-ended generation benchmarks require a judge, and the judge introduces its own biases.

**Reliability failures.** LLM outputs are stochastic. Sampling with temperature > 0 produces different answers on repeated runs. A single-run evaluation with 500 questions has high variance. The standard error of a proportion $p$ estimated from $n$ samples is approximately:

$$
\text{SE}(p) = \sqrt{\frac{p(1-p)}{n}}
$$

For $p = 0.80$ and $n = 500$, that is $\sqrt{0.80 \times 0.20 / 500} \approx 0.018$, meaning the 95 % confidence interval is roughly $\pm 3.6$ percentage points. Two models differing by 1 % on MMLU are likely within noise.

**Sensitivity failures.** Benchmarks saturate: once frontier models exceed 90 %, the remaining 10 % is dominated by ambiguous or poorly-worded questions, not model capability. New benchmarks replace old ones, breaking longitudinal comparisons.

### The Objective Mismatch

Training optimizes next-token prediction on a distribution of text. Evaluation wants to measure something else entirely — helpfulness, factual accuracy, reasoning ability, safety. These are related but not identical. A model trained on a trillion tokens of internet text develops representations that *correlate* with useful behaviors, but the correspondence is imperfect and the eval design determines what you actually measure.

Consider the chain of proxies:

```text
True goal: "Is this model useful for my tasks?"
         ↓
Operationalization: "Does it score well on a suite of benchmarks?"
         ↓
Measurement: accuracy on multiple-choice / code pass rates / human judgments
         ↓
Training signal: cross-entropy loss on next token
```

Each arrow introduces a gap. Evaluation research tries to close those gaps; this chapter is about understanding where they lie.

{{fig:eval-proxy-gap-ladder}}

---

## The Benchmark Zoo

The field has produced dozens of benchmarks. We organize them by the capability they primarily test, then discuss their mechanics, strengths, and weaknesses.

### Knowledge & Reasoning: MMLU and MMLU-Pro

**MMLU** (Massive Multitask Language Understanding, Hendrycks et al., 2021) contains around 14,000 four-choice questions spanning 57 subjects: STEM, humanities, social sciences, and professional domains (law, medicine, finance). Each question has one correct answer and three distractors.

The appeal is breadth: a single number summarizes performance across many domains. The weakness is the same breadth: average accuracy conflates wildly different skills, and a model that is superb at general knowledge but weak at formal mathematics and strong on medicine can land at the same number as one with a different profile.

**Scoring.** Most harnesses report normalized accuracy (fraction correct). Log-likelihood scoring — choosing the answer whose completion has highest $\log p$ under the model — is common and more reproducible than generation-based scoring.

$$
\hat{y} = \arg\max_{c \in \{A,B,C,D\}} \log p_\theta(\text{answer text of } c \mid \text{question})
$$

**MMLU-Pro** (Wang et al., 2024) adds harder, ten-choice questions with more complex reasoning requirements, reducing the chance of guessing correctly to 10 % versus 25 %. This makes it more discriminative at the frontier but harder to interpret for smaller models.

### Mathematical Reasoning: GSM8K and MATH

**GSM8K** (Grade School Math 8K, Cobbe et al., 2021) is a dataset of roughly 8,500 school-level word problems. A correct answer requires multi-step arithmetic (integer or simple decimal, no calculus). The canonical evaluation uses a model's generated chain-of-thought followed by a numerical extraction regex, and accuracy is the fraction of problems where the extracted number matches the reference answer.

**MATH** (Hendrycks et al., 2021) covers competition mathematics in seven subject categories — Prealgebra, Algebra, Intermediate Algebra, Counting & Probability, Geometry, Number Theory, and Precalculus — at five difficulty levels. Problems require symbolic manipulation, not just numerical calculation, which makes automated verification harder.

**Why GSM8K saturated fast.** Frontier models exceeded 90 % on GSM8K by 2023. The problems are short and the arithmetic is tractable for any model that learned the step-by-step format. Once a model learns *how to write out* arithmetic reasoning, the benchmark mainly tests whether that format was in training data. This led to MATH and then to AIME (below).

**AIME** (American Invitational Mathematics Examination). Problems from the real AIME competition test, where top high-school students in the US solve 15 very difficult integer-answer problems. A score of 4–5 out of 15 would historically have been considered noteworthy for language models; frontier reasoning models began surpassing that threshold around 2024–2025. Because new AIME exams are released yearly, contamination can be partially controlled by only evaluating on the most recent year's exam.

### Code: HumanEval and MBPP

**HumanEval** (Chen et al., 2021, Codex paper) is 164 hand-written Python programming problems, each consisting of a function signature, a docstring, and a set of unit tests. The primary metric is **pass@k**: the probability that at least one of $k$ generated samples passes all tests for a given problem.

$$
\text{pass@k} = \mathbb{E}_{\text{problem}} \left[ 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}} \right]
$$

where $n$ is the number of samples drawn and $c$ is the number that pass. For unbiased estimation one draws $n \geq k$ samples and applies this formula. In practice, many papers report pass@1 (single sample, greedy decoding) for simplicity.

**MBPP** (Mostly Basic Programming Problems, Austin et al., 2021) is 974 crowd-sourced Python problems (the commonly evaluated test split is 500 problems; a hand-verified "sanitized" subset has 427), similarly evaluated with unit tests. Both HumanEval and MBPP are now largely saturated at the frontier; models score above 90 % pass@1.

**Limitations.** Only functional correctness is tested; code style, efficiency, and security are ignored. The test suites are thin — typically 3–8 tests per problem — so it is possible to pass with a solution that is logically wrong but happens to match the test values.

### Scientific Reasoning: GPQA

**GPQA** (Graduate-Level Google-Proof Q&A, Rein et al., 2023) is a set of questions in biology, physics, and chemistry that are explicitly designed to be difficult for non-expert humans even with internet access ("Google-proof"). Domain experts answer correctly at rates in the 60–70 % range; non-experts with internet access are closer to 30–40 %. This makes it one of the few benchmarks where human expert performance is a meaningful, non-trivial ceiling.

GPQA exists because general knowledge benchmarks are too easy for frontier models and rely on information that was almost certainly in training data. GPQA forces genuine inference from principles, though contamination remains possible for any questions that leaked online before the cutoff.

### Instruction Following: IFEval

**IFEval** (Instruction Following Evaluation, Zhou et al., 2023) measures whether a model follows *verifiable* formatting constraints: "respond in exactly N words," "use bullet points," "include the word X," "write N paragraphs," "do not use commas." Because compliance can be checked programmatically, no judge model is needed.

IFEval distinguishes two metrics: **prompt-level accuracy** (did the model satisfy all instructions in the prompt?) and **instruction-level accuracy** (across all individual instructions, what fraction were satisfied?). It is a rare example of an eval where the ground truth is a deterministic function of the output text.

### Complex Reasoning: BBH (BIG-Bench Hard)

**BIG-Bench** was a massive collaborative benchmark containing hundreds of tasks. **BBH** (Suzgun et al., 2022) is the subset of 23 tasks on which, at the time, language models performed below human-rater agreement using few-shot prompting. Tasks include multi-step arithmetic, tracking shuffled objects, causal reasoning, and logical deduction. The motivation was to focus on tasks that remained genuinely challenging.

BBH is typically reported with chain-of-thought prompting, making it a measure of *multi-step reasoning ability given an appropriate scaffold* rather than raw knowledge retrieval.

### Abstraction: ARC-AGI

**ARC-AGI** (Abstraction and Reasoning Corpus for Artificial General Intelligence, Chollet, 2019) is qualitatively different from all the above. Each task presents 3–5 input/output grid pairs where the grids contain colored cells, and the model must infer the transformation rule and apply it to a new test grid. The rules cannot be memorized from training data because they are novel compositions of simple visual primitives (reflection, tiling, counting, etc.).

The benchmark is designed to test *fluid intelligence* — generalizing from a handful of examples to a novel rule — rather than crystallized knowledge or skill learned from large datasets. It proved very difficult for standard LLMs (scores in single digits for years); it attracted community attention during the ARC Prize challenge in 2024, where a combination of program synthesis and test-time compute approaches pushed scores meaningfully higher.

ARC-AGI is important conceptually: it is a benchmark that *resists* the standard scaling law playbook, because more parameters and more tokens do not straightforwardly help if the capability is principled rule induction rather than pattern completion.

---

## Benchmark Contamination

### The Problem

**Test contamination** occurs when the evaluation data appears in a model's training corpus, so the model can retrieve answers rather than generate them. This is the single most important validity threat for any benchmark result.

The internet contains solutions to many benchmarks. GSM8K problems and their step-by-step solutions are widely discussed in blog posts and GitHub repositories. MMLU questions appear in study guides and forums. HumanEval solutions are on Leetcode and other sites. Any model trained on a large, lightly-filtered web crawl will have seen some fraction of these.

### Detection Methods

**N-gram overlap.** The simplest detection computes the fraction of test questions that appear as near-verbatim substrings in a training document. A common heuristic: if a 13-gram from a test question appears in any training document, flag that example as potentially contaminated.

```python
# Minimal contamination detection using n-gram overlap
# Assumes training_docs is a list of strings, test_questions similarly.

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
```

**Memorization probes.** A stronger test presents the first half of a benchmark example and asks the model to complete it. If it produces the exact second half verbatim, that is strong evidence of memorization. Carlini et al. (2021, 2022) formalized this kind of training-data extraction and showed memorization grows with model scale and with how often a string is duplicated in the corpus; Oren et al. (2023) proposed an exchangeability test — a benchmark's examples are exchangeable, so a model that assigns systematically higher likelihood to the benchmark's canonical ordering than to shuffled orderings has likely seen the exact set; and Golchin & Surdeanu (2024) use guided prompting to coax verbatim continuations out of a suspected-contaminated model. The `completion_probe` below implements the simplest of these.

```python
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
```

Like all these probes, the completion probe needs a clean control set to interpret: an absolute memorized_rate is meaningless without a contamination-free baseline to compare against.

**Canary strings.** Benchmark authors embed unique, random token sequences into the evaluation set. If a model reproduces these during completion, the training corpus was contaminated.

**Behavioral tests.** Surprisingly, you can sometimes detect contamination by varying the surface form. If a model's accuracy drops sharply when questions are paraphrased or answer choices are shuffled, the model may be recognizing the original form rather than reasoning from the content.

### The Severity Gradient

Contamination is not binary. There is a spectrum:

1. **Verbatim memorization**: exact training data match; the model can reproduce the answer without reasoning.
2. **Near-verbatim**: training data contains a slightly different version; small generalization still required.
3. **Topic exposure**: the topic, domain, or reasoning pattern appears in training but not the specific question; this is unavoidable and arguably acceptable.
4. **Format priming**: the model has seen the *format* of the benchmark (e.g., four-choice Q&A) during training or fine-tuning; this gives an advantage unrelated to the capability being measured.

!!! warning "Common pitfall"
    Reporting a "decontaminated" score using only 13-gram overlap misses format priming and topic exposure. A more honest evaluation removes any training document whose topic domain significantly overlaps the test question domain — or uses held-out benchmark versions (e.g., AIME 2025 evaluated right after release).

{{fig:contamination-severity-gradient}}

---

## Benchmark Saturation

### What Saturation Means

Saturation occurs when frontier model performance clusters near the ceiling (90–100 %) so that benchmark scores no longer distinguish models. At that point, score differences are dominated by noise, ambiguous questions, and evaluation methodology choices rather than genuine capability differences.

GSM8K crossed this threshold around 2023; HumanEval in 2024. MMLU approaches it: multiple frontier models score in the 85–90 % range. When a benchmark saturates, the community moves to harder variants (MMLU → MMLU-Pro) or new benchmarks (GSM8K → MATH → AIME).

### Floor Effects in Smaller Models

The mirror problem affects evaluations of smaller models on very hard benchmarks. But there is a subtlety worth internalizing: a *genuine* floor on GPQA — a four-choice benchmark — is the random-chance rate of about 25 %, which a model reaches by emitting parseable answers that carry no signal. So if a 7B-parameter model scores 5 %, do not read that as "very weak scientific reasoning": 5 % is far *below* chance, and a model almost never performs worse than random guessing. A score that sits well under 25 % is a red flag that the answer extractor is failing — the model is producing answers in a format the regex misses (or refusing, or rambling without committing to a letter), so parsed answers default to wrong. Before drawing any capability conclusion, inspect the null/unparsed rate and a sample of raw generations; a true floor result clusters near 25 %, not near 0 %.

The rule of thumb: **a benchmark is most informative when average performance is between roughly 20 % and 80 %.** Below 20 % you are measuring noise; above 80 % you are measuring ceiling effects and question quality.

{{fig:benchmark-score-informative-band}}

---

## Multiple-Choice Pitfalls

### Why Multiple-Choice Underestimates and Overestimates

Multiple-choice benchmarks introduce specific artifacts:

**Inflation from guessing.** A four-choice benchmark has a random-chance baseline of 25 %. Models that cannot answer a question can still guess correctly 25 % of the time. Some benchmarks apply a **correction for guessing**:

$$
\text{score}_\text{corrected} = \text{correct} - \frac{\text{wrong}}{k-1}
$$

where $k = 4$ for four-choice. This is equivalent to the scoring rule for tests where you must mark a question wrong to guess, and makes the random-chance score 0.

**Sensitivity to option ordering.** Multiple experiments have shown that simply swapping which letter (A/B/C/D) carries the correct answer can change model accuracy by several percentage points. Models have biases toward certain positions (often A or the first option) that are purely artifacts of training data distribution.

**Sensitivity to answer format.** Whether the model is asked to output "A", "(A)", "Answer: A", or the full text of the answer all yield different accuracy estimates from the same underlying model knowledge.

```python
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
```

Measuring position bias itself is a separate experiment from scoring existing predictions, and it requires making NEW model calls: you re-render each question with the answer options permuted into every ordering, run the model on each permutation, and check whether accuracy depends on which letter the correct option happens to land on. A model with no position bias scores equally regardless of where the gold answer sits; a model that scores much higher when the answer is "A" (for instance) has a position bias.

```python
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
```

**Log-likelihood versus generation scoring.** Log-likelihood scoring (ranking answers by $\log p$ of their text given the question) does not require a well-calibrated answer extractor, but it has its own artifacts: the probability of a completion depends on its *length*, so a short correct answer ("Yes") and a long incorrect answer ("No, because of the following reasons...") are not comparable without length normalization.

A common normalization is to divide by the number of tokens in the completion:

$$
\text{score}(c) = \frac{1}{|c|} \sum_{t=1}^{|c|} \log p_\theta(c_t \mid \text{question}, c_{<t})
$$

Different harnesses make different choices here, making cross-harness comparisons unreliable.

---

## Worked Example: Confidence Intervals and Minimum Detectable Differences

!!! example "Worked example: when is a score difference real?"
    Suppose model A scores 82.4 % on MMLU (n = 14,042 questions) and model B scores 83.1 %.
    Is the 0.7 % difference statistically significant?

    The standard error for each proportion:

    $$
    \text{SE}_A = \sqrt{\frac{0.824 \times 0.176}{14042}} \approx \sqrt{0.00001032} \approx 0.00321
    $$

    $$
    \text{SE}_B = \sqrt{\frac{0.831 \times 0.169}{14042}} \approx \sqrt{0.00001001} \approx 0.00316
    $$

    The standard error of the *difference* (assuming independent samples, same questions):

    $$
    \text{SE}_{A-B} = \sqrt{\text{SE}_A^2 + \text{SE}_B^2} \approx \sqrt{(0.00321)^2 + (0.00316)^2} \approx 0.00452
    $$

    A 95 % confidence interval for the difference covers $\pm 1.96 \times 0.00452 \approx \pm 0.009$, or roughly $\pm 0.9$ percentage points.

    The observed difference of 0.7 % falls *within* the 95 % CI, meaning we cannot reject the null hypothesis that the models are equally capable on MMLU.

    Moral: **a difference smaller than about 1 % on the full MMLU set is not statistically significant without multiple runs or a stricter significance test.** On smaller subsets (e.g., a 500-question domain slice), the CI is roughly $\pm 3.5$ points, so differences below 7 % are noise.

---

## The Gap Between Benchmarks and Usefulness

### Why High Benchmark Scores Do Not Guarantee Useful Models

Benchmark performance and practical usefulness diverge for several reasons:

**Distribution shift.** Benchmarks test a fixed distribution of questions curated at one point in time. User queries are drawn from a different, continuously shifting distribution. A model can dominate a benchmark while failing on common user requests that the benchmark did not anticipate.

**Capability ≠ behavior.** A model may *be able* to answer a question correctly but refuse to do so because of overly conservative safety training, choose a verbose format the user dislikes, or give a technically correct but practically useless answer. Benchmarks usually measure capability, not the full response quality a user experiences.

**Task coverage.** No benchmark captures everything users actually want. Coding benchmarks do not test the ability to explain code to a junior developer. Knowledge benchmarks do not test the ability to write a persuasive essay. Instruction-following benchmarks test atomic constraints, not the fluency of a 2,000-word response that also follows constraints.

**Goodhart's Law.** "When a measure becomes a target, it ceases to be a good measure." Models fine-tuned specifically on benchmark-adjacent data can inflate benchmark scores without improving on user tasks. The field has seen multiple cases where targeted post-training sharply improved benchmark rankings without corresponding improvements in blind human evaluation.

### The Role of Human Evaluation

Human evaluation is expensive, slow, and noisy, but it remains the gold standard for usefulness. The standard paradigm is A/B comparison: present two model outputs to human raters and ask which is better (or ask for a score on several dimensions). The LMSYS Chatbot Arena (Chiang et al., 2024) is the most widely used public version of this: users chat with two anonymous models, vote for the better response, and ratings are aggregated into an Elo-style ranking.

Human evals have their own problems: rater fatigue, verbosity bias (longer answers tend to win even when shallower), positional bias (the first response tends to win slightly), and the fact that casual chatbot preferences may not correlate with performance on professional tasks.

The evolving practice is to use human evaluation as a calibration signal for automated evaluators — see [LLM-as-a-Judge & Automated Evaluation](../11-evaluation/02-llm-as-judge.html) — so that the cheap, scalable automated signal can be shown to correlate with expensive human judgment.

### Leaderboard Gaming and the Meta-Benchmark Problem

When benchmarks become public leaderboards (as MMLU, HumanEval, and others have), they attract optimization pressure. Labs run many ablations and select the checkpoint and prompting strategy that maximizes the leaderboard score, which is a form of implicit overfitting to the test distribution even without literal data contamination.

The most rigorous evals are:
1. **Held-out or private**: the test set is never released publicly (e.g., private portions of safety evals at major labs).
2. **Freshly generated**: questions are generated after the model's training cutoff (e.g., new AIME problems each year).
3. **Diverse enough to resist targeted optimization**: a large, heterogeneous suite where gaming one task hurts others.

The [Machine Learning Fundamentals](../01-foundations/05-ml-fundamentals.html) chapter discusses the train-test split principle that underlies this; the same logic applies at benchmark scale.

---

## Critical Literacy: Reading Benchmark Claims

When you encounter a model card or paper reporting benchmark results, apply this checklist:

```text
BENCHMARK CLAIM CHECKLIST
==========================
1. WHICH version of the benchmark?
   - MMLU vs. MMLU-Pro?  5-shot vs. 0-shot?
   - Log-likelihood or generation scoring?  Which normalization?

2. WHAT prompting strategy?
   - 0-shot, few-shot (N-shot), chain-of-thought, zero-shot-CoT?
   - System prompt present?  Format instructions?

3. CONTAMINATION: did training data predate the benchmark release?
   - Was decontamination performed?  What method?
   - Does the paper report post-decontamination scores?

4. COMPARISON FAIRNESS:
   - Are baseline numbers from the same harness with the same settings?
   - Were baselines re-run or taken from prior papers?

5. STATISTICAL SIGNIFICANCE:
   - Is the benchmark large enough to support the claimed difference?
   - Are confidence intervals reported?

6. WHAT'S NOT REPORTED?
   - Benchmarks where this model underperformed?
   - Human evaluation?  Real task performance?
```

Reproducibility is a live problem. The same model evaluated with different harnesses (lm-evaluation-harness, HELM, BigBench Lite, etc.) can show multi-percentage-point swings. The [Building Eval Harnesses](../11-evaluation/03-eval-harnesses.html) chapter covers how to standardize evaluations.

---

## The Benchmark Landscape at a Glance

| Benchmark | Domain | Format | Questions | Primary Metric | Saturation? |
|---|---|---|---|---|---|
| MMLU | Knowledge (57 domains) | 4-choice | ~14,000 | Accuracy | Approaching |
| MMLU-Pro | Knowledge (harder) | 10-choice | ~12,000 | Accuracy | No |
| GSM8K | Grade-school math | Open answer | 8,500 | Exact match | Yes |
| MATH | Competition math | Open answer | 12,500 | Exact match | Partial |
| AIME | Hard competition math | Integer answer | 15/yr | Exact match | No |
| HumanEval | Python coding | Code gen + unit tests | 164 | pass@1 | Yes |
| MBPP | Python coding | Code gen + unit tests | 974 (500 test) | pass@1 | Yes |
| GPQA | PhD science | 4-choice | ~450 | Accuracy | No |
| BBH | Complex reasoning | Open/4-choice | 6,511 | Accuracy | Partial |
| IFEval | Instruction following | Open (rule-checked) | 541 | Prompt/instr accuracy | No |
| ARC-AGI | Visual rule induction | Grid matching | 400 | Accuracy | No |

---

## Interview Corner

!!! interview "Interview Corner"
    **Q:** A paper reports that their new model achieves state-of-the-art performance on MMLU, improving from 85.2 % to 86.1 %. Should you believe the result, and why or why not?

    **A:** Be skeptical for several reasons. First, do a quick statistical sanity check: MMLU has ~14,000 questions, so the SE of each estimate is about 0.3 %, giving a 95 % CI on the difference of roughly ±0.85 %. The claimed 0.9 % improvement is only marginally outside that CI and could be noise from a single run. Second, check whether the same harness and prompting strategy was used for both models — different log-likelihood normalization or few-shot count can easily shift scores by 1–2 %. Third, check for contamination: if the training data postdates MMLU's release (2021) and no decontamination was done, the gain may reflect memorization. Fourth, check whether MMLU is now near saturation for this class of model — at 85–86 %, the remaining questions may be ambiguous or malformed, so the "improvement" may be spurious. A more credible evaluation would also report performance on harder benchmarks (MMLU-Pro, GPQA), include human evals or a judge-based comparison, and provide confidence intervals.

---

## Key Takeaways

!!! key "Key Takeaways"
    - Benchmark validity, reliability, and sensitivity all fail in practice: benchmarks measure what is easy to measure, not necessarily what you care about, and confidence intervals are wider than practitioners assume.
    - The standard error of a proportion on a benchmark of $n$ items is $\sqrt{p(1-p)/n}$; two models differing by less than ~1 % on MMLU (n ≈ 14,000) are likely within noise.
    - Contamination is a spectrum from verbatim memorization to format priming; 13-gram overlap detection catches the obvious cases but misses subtler ones.
    - Saturation renders benchmarks uninformative at the frontier: once performance exceeds ~80–85 %, the remaining errors are dominated by question quality, not model capability, and the field moves to harder benchmarks.
    - Multiple-choice benchmarks introduce artifacts: positional bias, sensitivity to answer formatting, and inflation from guessing. Log-likelihood scoring reduces extraction noise but introduces length-normalization choices.
    - High benchmark scores are necessary but not sufficient for usefulness: capability does not equal behavior, benchmarks sample a fixed distribution, and Goodhart's Law applies as soon as a benchmark becomes a leaderboard target.
    - Critical benchmark literacy means checking: which version, which prompting strategy, which harness, contamination protocol, statistical significance, and what was *not* reported.
    - ARC-AGI is qualitatively distinct: it tests fluid rule induction from a handful of examples, which resists the standard scaling-law solution and requires principled generalization.

---

!!! sota "State of the Art & Resources (2026)"
    LLM evaluation remains an open research problem: benchmarks saturate within months, contamination undermines validity, and no single metric reliably predicts real-world usefulness. The field is converging on dynamic, contamination-resistant benchmarks, crowdsourced human preference signals, and multi-dimensional evaluation suites as partial solutions to these challenges.

    **Foundational work**

    - [Hendrycks et al., *Measuring Massive Multitask Language Understanding* (2021)](https://arxiv.org/abs/2009.03300) — introduced MMLU, still the most widely cited knowledge benchmark, defining the 57-domain evaluation paradigm.
    - [Suzgun et al., *Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them* (2022)](https://arxiv.org/abs/2210.09261) — distilled BIG-Bench into 23 hard reasoning tasks (BBH) and showed CoT dramatically improved performance.
    - [Liang et al., *Holistic Evaluation of Language Models* (2022)](https://arxiv.org/abs/2211.09110) — HELM framework; argued for multi-metric, multi-scenario evaluation beyond single-number leaderboards.

    **Recent advances (2023–2026)**

    - [Rein et al., *GPQA: A Graduate-Level Google-Proof Q&A Benchmark* (2023)](https://arxiv.org/abs/2311.12022) — 448 PhD-level science questions where non-expert humans with internet access score ~34%; one of the last benchmarks not yet saturated by frontier models.
    - [Wang et al., *MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark* (2024)](https://arxiv.org/abs/2406.01574) — ten-choice questions with heavier reasoning load; NeurIPS 2024 Spotlight, replacing MMLU as the standard knowledge eval.
    - [Gema et al., *Are We Done with MMLU?* (2024)](https://arxiv.org/abs/2406.04127) — systematic audit finding pervasive label errors in MMLU; introduced MMLU-Redux (5,700 re-annotated questions) and showed reported scores are inflated.
    - [White et al., *LiveBench: A Challenging, Contamination-Limited LLM Benchmark* (2024)](https://arxiv.org/abs/2406.19314) — monthly-updated questions drawn from recent arXiv papers and news; auto-scored with objective ground truth, eliminating judge bias.
    - [Chiang et al., *Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference* (2024)](https://arxiv.org/abs/2403.04132) — crowdsourced pairwise Elo-style human evaluation with 240K+ votes; the dominant real-world usefulness signal.
    - [Chollet et al., *ARC Prize 2024: Technical Report* (2024)](https://arxiv.org/abs/2412.04604) — documents the ARC Prize competition where test-time training pushed ARC-AGI scores from 33 % to 55.5 %, revealing the limits of pure scaling for fluid-intelligence tasks.

    **Open-source & tools**

    - [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — de facto standard eval framework backing the HuggingFace Open LLM Leaderboard; supports 60+ benchmarks across HuggingFace, vLLM, and API backends.

## Further Reading

- Hendrycks et al., *Measuring Massive Multitask Language Understanding* (MMLU), 2021.
- Cobbe et al., *Training Verifiers to Solve Math Word Problems* (GSM8K), 2021.
- Hendrycks et al., *Measuring Mathematical Problem Solving With the MATH Dataset*, 2021.
- Chen et al., *Evaluating Large Language Models Trained on Code* (HumanEval / Codex), 2021.
- Rein et al., *GPQA: A Graduate-Level Google-Proof Q&A Benchmark*, 2023.
- Suzgun et al., *Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them* (BBH), 2022.
- Zhou et al., *Instruction-Following Evaluation for Large Language Models* (IFEval), 2023.
- Chollet, *On the Measure of Intelligence* (ARC), 2019.
- Wang et al., *MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark*, 2024.
- Liang et al., *Holistic Evaluation of Language Models* (HELM), 2022.
- Chiang et al., *Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference*, 2024.
- EleutherAI, *lm-evaluation-harness* (GitHub repository).

---

## Exercises

**1.** A 7B model is evaluated on GPQA (a four-choice benchmark) and scores **6 %**. A colleague concludes the model has "essentially no scientific reasoning ability." Explain why this conclusion is unjustified given only that number, what the *genuine* floor of the benchmark is, and what you would inspect before drawing any capability conclusion.

??? note "Solution"
    On a four-choice benchmark, a model that emits parseable but content-free answers scores at the random-chance rate of about $1/4 = 25\%$. A model almost never performs *worse* than random guessing on the underlying task, so a score of 6 % is far *below* the genuine floor and cannot reflect the model's scientific reasoning at all — a truly weak model bottoms out near 25 %, not near 0 %.

    A score well under chance is a red flag that the **answer extractor**, not the model, is failing. The likely causes are that the model produces answers in a format the extraction regex misses, refuses, or rambles without committing to a letter — so unparsed generations default to "wrong."

    Before concluding anything about capability, inspect:

    - the **null / unparsed rate** (`null_rate` from `evaluate_mc_batch`) — a high value confirms extraction failure;
    - a **sample of raw generations** to see what format the model actually uses;

    then fix the extractor (or switch to log-likelihood scoring) and re-run. A true floor result clusters near 25 %.

**2.** You evaluate two models on a **200-question** domain slice of a benchmark. Model A scores **60 %** and model B scores **68 %** on the *same* questions. Treating the two proportions as independent (as the chapter's worked example does), compute the standard error of each estimate, the standard error of the difference, and the 95 % confidence interval of the difference. Is the 8-point gap statistically significant?

??? note "Solution"
    Standard error of a proportion is $\text{SE}(p) = \sqrt{p(1-p)/n}$ with $n = 200$.

    $$
    \text{SE}_A = \sqrt{\frac{0.60 \times 0.40}{200}} = \sqrt{0.00120} \approx 0.0346
    $$

    $$
    \text{SE}_B = \sqrt{\frac{0.68 \times 0.32}{200}} = \sqrt{0.001088} \approx 0.0330
    $$

    Standard error of the difference:

    $$
    \text{SE}_{A-B} = \sqrt{\text{SE}_A^2 + \text{SE}_B^2} = \sqrt{0.00120 + 0.001088} \approx \sqrt{0.002288} \approx 0.0478
    $$

    The 95 % CI of the difference is $\pm 1.96 \times 0.0478 \approx \pm 0.0938$, i.e. about $\pm 9.4$ percentage points.

    The observed gap is $68\% - 60\% = 8$ points, which falls *inside* the $\pm 9.4$-point interval. So the difference is **not statistically significant** — on a 200-question slice you cannot distinguish these models from a single run. This matches the chapter's rule of thumb that on a ~500-question slice the CI is roughly $\pm 3.5$ points (differences below ~7 points are noise), and shrinking to 200 questions widens it further.

**3.** A four-choice benchmark ($k = 4$) has 100 questions. A model produces a parseable answer to every question, getting **40 correct** and **60 wrong**. Compute its raw accuracy and its guessing-corrected score using the chapter's correction-for-guessing formula. Then verify that a pure random guesser (25 correct, 75 wrong) gets a corrected score of 0.

??? note "Solution"
    Raw accuracy is simply $40/100 = 0.40$, i.e. 40 %.

    The correction-for-guessing formula is

    $$
    \text{score}_\text{corrected} = \text{correct} - \frac{\text{wrong}}{k-1}, \qquad k = 4.
    $$

    For this model: $40 - \dfrac{60}{3} = 40 - 20 = 20$ effective correct answers, or $20/100 = 0.20$ as a fraction.

    For a pure random guesser on 100 four-choice questions, the expected split is 25 correct / 75 wrong:

    $$
    25 - \frac{75}{3} = 25 - 25 = 0.
    $$

    The correction rescales the scale so that guessing scores 0 and perfect scores 1, removing the 25 % free-guessing floor. Note the corrected score (20 %) is below the raw accuracy (40 %) because much of the raw accuracy was attributable to lucky guessing.

**4.** On a HumanEval-style problem you draw $n = 5$ samples and $c = 2$ of them pass all unit tests. Using the chapter's unbiased pass@k estimator, compute pass@1 and pass@2 for this problem. Show that the pass@1 estimate equals $c/n$.

??? note "Solution"
    The unbiased single-problem estimator is

    $$
    \text{pass@k} = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}, \qquad n = 5,\ c = 2.
    $$

    **pass@1** ($k = 1$):

    $$
    1 - \frac{\binom{3}{1}}{\binom{5}{1}} = 1 - \frac{3}{5} = 0.40.
    $$

    In general for $k = 1$: $1 - \dfrac{\binom{n-c}{1}}{\binom{n}{1}} = 1 - \dfrac{n-c}{n} = \dfrac{c}{n}$, which here is $2/5 = 0.40$. So pass@1 is exactly the fraction of samples that pass.

    **pass@2** ($k = 2$):

    $$
    1 - \frac{\binom{3}{2}}{\binom{5}{2}} = 1 - \frac{3}{10} = 0.70.
    $$

    pass@2 (0.70) exceeds pass@1 (0.40) because drawing two samples gives two chances for at least one to pass. The term $\binom{n-c}{k}/\binom{n}{k}$ is the probability that *all* $k$ drawn samples come from the $n-c$ failing ones, so one minus it is the probability at least one passes.

**5.** Implement `estimate_pass_at_k(n, c, k)` for the chapter's pass@k formula, in the chapter's code style. It must return the single-problem estimate $1 - \binom{n-c}{k}/\binom{n}{k}$, be numerically stable for large $n$ (do not form giant factorials), and handle the edge cases $c = 0$ (return 0.0), $c \geq$ enough that every draw passes (return 1.0), and $k > n$ (invalid). Verify it against Exercise 4.

??? note "Solution"
    The stable trick, used by the Codex paper, is to write $\dfrac{\binom{n-c}{k}}{\binom{n}{k}}$ as a product $\prod_{i=n-c+1}^{n} \dfrac{i-k}{i}$, which stays in $[0, 1]$ and never builds a large factorial. When $n - c < k$ every $k$-subset must include a passing sample, so pass@k $= 1$; when $c = 0$ the product is 1 and pass@k $= 0$.

    ```python
    def estimate_pass_at_k(n: int, c: int, k: int) -> float:
        """
        Unbiased single-problem pass@k estimate: 1 - C(n-c, k) / C(n, k),
        where n = samples drawn, c = samples that pass, k = budget.

        Computed as a running product prod_{i=n-c+1..n} (i - k) / i to avoid
        forming large binomial coefficients. Stays in [0, 1].
        """
        if not (0 <= c <= n):
            raise ValueError("need 0 <= c <= n")
        if not (1 <= k <= n):
            raise ValueError("need 1 <= k <= n")
        if c == 0:
            return 0.0
        if n - c < k:
            # fewer than k failing samples => every k-subset contains a pass
            return 1.0
        prob_all_fail = 1.0
        for i in range(n - c + 1, n + 1):
            prob_all_fail *= (i - k) / i
        return 1.0 - prob_all_fail


    # Verification against Exercise 4 (n=5, c=2):
    assert abs(estimate_pass_at_k(5, 2, 1) - 0.40) < 1e-9   # == c/n
    assert abs(estimate_pass_at_k(5, 2, 2) - 0.70) < 1e-9
    # Edge cases:
    assert estimate_pass_at_k(5, 0, 1) == 0.0               # nothing passes
    assert estimate_pass_at_k(5, 4, 2) == 1.0               # only 1 failing < k
    ```

    A full pass@k over a benchmark averages `estimate_pass_at_k(n, c_problem, k)` across problems, matching the expectation $\mathbb{E}_{\text{problem}}[\cdot]$ in the chapter's formula.

**6.** You want to *measure* position bias, not just score existing predictions. Using `make_position_bias_variants` from the chapter, you run a model on all permutations of a four-choice question and, aggregating over many questions, obtain the following accuracies grouped by which letter the gold answer landed on: A = 0.74, B = 0.61, C = 0.58, D = 0.55. (a) For a single four-choice question, how many permutations does `make_position_bias_variants` produce, and how many times does the gold option land on each letter? (b) Write a small function `position_bias_spread(acc_by_letter)` that returns the max-minus-min spread, compute it here, and explain why post-hoc shuffling of `(prediction, gold)` pairs could *not* have produced these numbers.

??? note "Solution"
    **(a)** For four options the function enumerates `itertools.permutations(range(4))`, which is $4! = 24$ permutations. By symmetry the gold option occupies each of A, B, C, D in exactly $24 / 4 = 6$ of them (the chapter's sanity check asserts `Counter(...) == {"A": 6, "B": 6, "C": 6, "D": 6}`). So every letter gets an equal, fair number of trials.

    **(b)**

    ```python
    def position_bias_spread(acc_by_letter: dict) -> float:
        """Max minus min accuracy across gold-letter positions.
        0.0 means no measurable position bias; larger means more bias."""
        vals = list(acc_by_letter.values())
        return max(vals) - min(vals)

    acc = {"A": 0.74, "B": 0.61, "C": 0.58, "D": 0.55}
    assert abs(position_bias_spread(acc) - 0.19) < 1e-9
    ```

    The spread is $0.74 - 0.55 = 0.19$, i.e. 19 percentage points — a large position bias favouring answer "A," exactly the "bias toward the first option" the chapter describes.

    Post-hoc shuffling of `(prediction, gold)` pairs could never produce these numbers because reordering the list of already-generated predictions leaves every individual pred/gold pairing unchanged; accuracy is invariant under permuting examples. Detecting position bias requires **fresh model calls**: you must re-render each question with the options permuted (`make_position_bias_variants`) and query the model again for each ordering, because the bias lives in *how the model responds* to a given ordering, not in the static scoring of fixed outputs.
