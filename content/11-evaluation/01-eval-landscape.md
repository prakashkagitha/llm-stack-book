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

**MATH** (Hendrycks et al., 2021) covers competition mathematics in seven categories — algebra, geometry, number theory, probability, pre-calculus, calculus, and combinatorics — at five difficulty levels. Problems require symbolic manipulation, not just numerical calculation, which makes automated verification harder.

**Why GSM8K saturated fast.** Frontier models exceeded 90 % on GSM8K by 2023. The problems are short and the arithmetic is tractable for any model that learned the step-by-step format. Once a model learns *how to write out* arithmetic reasoning, the benchmark mainly tests whether that format was in training data. This led to MATH and then to AIME (below).

**AIME** (American Invitational Mathematics Examination). Problems from the real AIME competition test, where top high-school students in the US solve 15 very difficult integer-answer problems. A score of 4–5 out of 15 would historically have been considered noteworthy for language models; frontier reasoning models began surpassing that threshold around 2024–2025. Because new AIME exams are released yearly, contamination can be partially controlled by only evaluating on the most recent year's exam.

### Code: HumanEval and MBPP

**HumanEval** (Chen et al., 2021, Codex paper) is 164 hand-written Python programming problems, each consisting of a function signature, a docstring, and a set of unit tests. The primary metric is **pass@k**: the probability that at least one of $k$ generated samples passes all tests for a given problem.

$$
\text{pass@k} = \mathbb{E}_{\text{problem}} \left[ 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}} \right]
$$

where $n$ is the number of samples drawn and $c$ is the number that pass. For unbiased estimation one draws $n \geq k$ samples and applies this formula. In practice, many papers report pass@1 (single sample, greedy decoding) for simplicity.

**MBPP** (Mostly Basic Programming Problems, Austin et al., 2021) is 374 crowd-sourced Python problems, similarly evaluated with unit tests. Both HumanEval and MBPP are now largely saturated at the frontier; models score above 90 % pass@1.

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

**Memorization probes.** A stronger test presents the first half of a benchmark example and asks the model to complete it. If it produces the exact second half verbatim, that is strong evidence of memorization. The *MeMo* and *DataComp* lines of work developed more principled probes.

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

---

## Benchmark Saturation

### What Saturation Means

Saturation occurs when frontier model performance clusters near the ceiling (90–100 %) so that benchmark scores no longer distinguish models. At that point, score differences are dominated by noise, ambiguous questions, and evaluation methodology choices rather than genuine capability differences.

GSM8K crossed this threshold around 2023; HumanEval in 2024. MMLU approaches it: multiple frontier models score in the 85–90 % range. When a benchmark saturates, the community moves to harder variants (MMLU → MMLU-Pro) or new benchmarks (GSM8K → MATH → AIME).

### Floor Effects in Smaller Models

The mirror problem affects evaluations of smaller models on very hard benchmarks. If a 7B-parameter model scores 5 % on GPQA, that is near random chance (25 % for four-choice), and we cannot distinguish "cannot reason scientifically" from "cannot read the question." Performance at the floor is equally uninformative.

The rule of thumb: **a benchmark is most informative when average performance is between roughly 20 % and 80 %.** Below 20 % you are measuring noise; above 80 % you are measuring ceiling effects and question quality.

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
    shuffle_seed: Optional[int] = None,
) -> dict:
    """
    Evaluate a batch of multiple-choice predictions.

    Args:
        predictions: raw model output strings.
        gold_labels: correct answer letters, e.g. ["A", "C", "B", ...].
        shuffle_seed: if set, randomly shuffles answer order to test position bias.

    Returns:
        dict with 'accuracy', 'null_rate' (fraction where answer was not extracted),
        and 'corrected_accuracy' (guessing-corrected, k=4).
    """
    import random

    if shuffle_seed is not None:
        rng = random.Random(shuffle_seed)
        paired = list(zip(predictions, gold_labels))
        rng.shuffle(paired)
        predictions, gold_labels = zip(*paired)

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
| MBPP | Python coding | Code gen + unit tests | 374 | pass@1 | Yes |
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
