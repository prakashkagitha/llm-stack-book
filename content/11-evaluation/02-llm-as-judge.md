# 11.2 LLM-as-a-Judge & Automated Evaluation

Human evaluation of LLM outputs is the gold standard — but it is slow, expensive, and hard to reproduce. A single preference study across a few thousand examples might cost weeks of annotator time and thousands of dollars. When you are running ablations, A/B testing prompts, or iterating on a fine-tuned model daily, that cadence is untenable.

The insight behind **LLM-as-a-Judge** (LLMaaJ) is simple: a sufficiently capable language model can replicate many of the judgments a human makes about quality, correctness, and helpfulness — and it can do so at the speed of inference. Since roughly 2023, this approach has become the backbone of nearly every large-scale automated evaluation pipeline, including the reward modeling loop described in [The RLHF Pipeline & Reward Modeling](../05-posttraining-alignment/05-rlhf-reward-modeling.html).

This chapter is a rigorous treatment of the machinery: how to prompt a judge, which biases infect its scores, how to calibrate and audit it against human labels, and how to build the Chatbot Arena Elo system from scratch.

---

## Why Human Evaluation Alone Does Not Scale

Before diving into the mechanics, it is worth being precise about why we need automation at all.

**Throughput.** A single annotator, given a pair of model responses, might take 2–3 minutes to read, compare, and label. At that rate, 10,000 comparisons require roughly 300–500 annotator-hours. With a team of 10, that is multiple weeks. Modern development cycles need daily evaluation loops.

**Consistency.** Even trained annotators disagree. Inter-annotator agreement on open-ended quality often sits at Cohen's κ in the range of 0.4–0.6 — barely moderate. Annotator mood, fatigue, and framing effects introduce noise that is hard to control.

**Coverage.** Benchmarks like MMLU or HumanEval cover well-defined task categories with right/wrong answers — see [The Evaluation Problem & Benchmark Landscape](../11-evaluation/01-eval-landscape.html). Open-ended generation (summaries, essays, conversational responses) cannot be evaluated this way. LLMaaJ fills that gap.

**Reproducibility.** A judge prompt is code. It can be versioned, diffed, and re-run. Human panels cannot.

The tradeoff is that LLM judges have their own systematic biases — and the chapter's central skill is understanding and mitigating those biases so the judge is trustworthy.

---

## The Two Evaluation Paradigms: Pointwise vs Pairwise

All LLMaaJ setups fall into one of two modes, and the choice has deep implications for reliability and use case.

### Pointwise (Absolute) Scoring

A single response is scored on an absolute scale — commonly 1–5 or 1–10 — against a rubric.

```text
Judge prompt (pointwise):

  [System]
  You are a rigorous and calibrated evaluator. Score the following response
  on a scale of 1 to 5 for the criterion "helpfulness", where:
    1 = completely unhelpful or harmful
    2 = partially addresses the request but has major gaps
    3 = adequate response with some flaws
    4 = good, nearly complete response
    5 = excellent, comprehensive, directly useful response

  [User]
  Question: {question}
  Response: {response}

  Reply with ONLY a JSON object: {"score": <int 1-5>, "rationale": "<1-2 sentences>"}
```

Advantages: can compare any number of models without re-running pairs; scores are additive (you can average across criteria); efficient at O(N) queries per N responses.

Disadvantages: the absolute scale is hard to calibrate across prompt types. A model might give a score of 4 for responses that a human would rate quite differently, because the judge has no anchor.

### Pairwise (Comparative) Scoring

Two responses A and B are shown simultaneously. The judge picks a winner (or declares a tie).

```text
Judge prompt (pairwise):

  [System]
  You are a fair, expert evaluator. Given a question and two responses (A and B),
  decide which response is more helpful, accurate, and well-written.
  Do not let response length, formatting style, or order influence your decision.

  [User]
  Question: {question}

  Response A:
  {response_a}

  Response B:
  {response_b}

  Output ONLY JSON: {"winner": "A" | "B" | "tie", "rationale": "<2-3 sentences>"}
```

Advantages: relative judgments are cognitively easier and better calibrated than absolute ones — the judge (like a human) can often tell which of two things is better even when scoring each on an absolute scale is hard. Consistency within a comparison is high.

Disadvantages: requires O(N²) queries to compare all pairs; does not yield an absolute quality estimate; susceptible to **position bias** (discussed next).

### When to Use Each

| Criterion | Pointwise | Pairwise |
|---|---|---|
| Comparing many models | Efficient | Expensive (O(N²)) |
| Ranking two models head-to-head | Noisy anchor | Natural |
| Computing absolute quality scores | Direct | Needs Elo conversion |
| Agreement with human preference | Moderate | High |
| Susceptibility to position bias | Low | High |

In practice, production eval pipelines often use **pointwise scoring for monitoring** (cheap, daily runs) and **pairwise comparison for model selection** (milestone decisions).

{{fig:judge-pointwise-vs-pairwise}}

---

## Rubric Design: Decomposing Quality

Open-ended "quality" is multidimensional. A response can be accurate but terse, or verbose but partially wrong. Collapsing everything into a single score loses signal. The solution is **rubric-based multi-criteria evaluation**.

### The G-Eval Framework (Liu et al.)

G-Eval (Liu et al., 2023) pioneered using chain-of-thought (CoT) to generate evaluation criteria and score steps, then computing a weighted score. The key idea: ask the judge to first produce a rationale, then emit a score — this dramatically improves calibration vs. asking for only a score.

A production rubric typically includes several dimensions:

```python
RUBRIC_CRITERIA = {
    "correctness": (
        "Is every factual claim in the response accurate? "
        "Does it avoid hallucinations and unsupported assertions?"
    ),
    "helpfulness": (
        "Does the response directly address the user's question? "
        "Is the answer complete?"
    ),
    "safety": (
        "Does the response avoid harmful, toxic, or policy-violating content?"
    ),
    "conciseness": (
        "Is the response appropriately concise without omitting important information?"
    ),
    "instruction_following": (
        "Did the response follow all explicit instructions (format, length, language, etc.)?"
    ),
}
```

The judge evaluates each criterion independently (or in one pass with separate scores), then we aggregate. Separating criteria also makes it easier to **debug regressions**: if a new model drop has lower helpfulness but stable safety, you know exactly where to look.

### Weighted Aggregation

If criteria have different business importance, use a weighted sum:

$$
S_{\text{overall}} = \sum_{c \in \mathcal{C}} w_c \cdot s_c
$$

where $w_c$ is the weight for criterion $c$ and $s_c \in [1, 5]$ is the score. Weights should be set based on product requirements and validated against human preference labels, not chosen arbitrarily.

---

## Biases in LLM Judges: The Menagerie

The central challenge of LLMaaJ is systematic bias — the judge deviates from human ground truth in predictable, correctable ways. You must know these biases to build a trustworthy system.

### Position Bias

In pairwise evaluation, the judge systematically favors whichever response appears first (or second, depending on the model). This has been empirically documented across multiple judge models. The effect can be as large as 10–15 percentage points on win rate depending on task difficulty.

**Mitigation: always run both orderings.** Present (A, B) and (B, A), then aggregate:

```python
def pairwise_judge_debiased(judge_fn, question, response_a, response_b):
    """
    Run both orderings and combine results to cancel position bias.
    Returns: "A", "B", or "tie"
    """
    # First ordering: A before B
    result_ab = judge_fn(question, response_a, response_b)  # "A","B","tie"
    
    # Second ordering: B before A (we swap and interpret the result)
    result_ba = judge_fn(question, response_b, response_a)  # returns in terms of B,A
    # Flip result_ba back to A/B space
    flip = {"A": "B", "B": "A", "tie": "tie"}
    result_ba_flipped = flip[result_ba]
    
    # Resolve
    if result_ab == result_ba_flipped:
        return result_ab          # Both orderings agree
    else:
        return "tie"              # Disagree → call it a tie
```

This doubles inference cost but nearly eliminates position bias.

{{fig:judge-position-swap-debias}}

### Verbosity Bias (Length Bias)

LLM judges tend to prefer longer responses, independent of quality. A response that elaborates at length — even if it contains more hedging and filler — often beats a tight, accurate response. This has been measured in MT-Bench and other benchmarks.

**Mitigation strategies:**

1. Explicitly instruct the judge: *"Do not let response length or formatting influence your decision. A concise, accurate answer is better than a verbose one with the same information."*
2. Include a "conciseness" criterion in the rubric to counterbalance length preference.
3. Normalize scores by response length in your analysis pipeline to detect length artifacts.
4. Test your judge's calibration by sampling response pairs where one is a padded version of the other.

### Self-Preference Bias

When the judge model is the same as (or closely related to) one of the evaluated models, it tends to prefer its own outputs. GPT-4 as judge overrates GPT-4 responses. This is sometimes called the "narcissism" problem.

**Mitigation:** Use a **different model family** for judging than for generation when possible. If you are evaluating GPT-4-class models, use Claude or Gemini as judge, and vice versa. Track self-preference rate in your calibration set.

### Sycophancy Bias

If the judge is shown a hint about which response a human preferred (or if one response includes flattery or authoritative-sounding language), it may defer to that cue rather than evaluating on merit. This extends to the judge agreeing with whatever opinion is asserted confidently in a response.

**Mitigation:** Never leak labels into the judge context. Use neutral prompt framing.

### Instruction-Following Bias

Some judges over-weight instruction following and under-weight factual accuracy. A response that perfectly follows format instructions but contains factual errors may score higher than one with a minor format deviation but accurate content.

**Mitigation:** Use separate rubric criteria. Weight them explicitly.

### The Bias Calibration Protocol

Build a **gold calibration set** of ~200–500 examples with trusted human labels (e.g., majority vote from 3+ annotators). Regularly run your judge on this set and compute:

- Agreement rate (what fraction match the human majority label?)
- Bias profile per ordering (does position A win more than expected?)
- False positive / false negative rates per criterion

If calibration drops, it usually signals that your judge prompt needs updating for the new model family or that the task distribution has drifted.

{{fig:judge-bias-menagerie}}

---

## Agreement with Humans: Measuring Judge Quality

A judge is only trustworthy if it agrees with humans at a high enough rate. The standard metric is **agreement rate** on a held-out human annotation set.

### Cohen's Kappa

Raw agreement is confounded by chance. Cohen's kappa adjusts for it:

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

where $p_o$ is observed agreement and $p_e$ is expected agreement under the null hypothesis that both raters assign labels randomly according to their marginals.

For a binary (A wins / B wins / tie) pairwise task, if the judge labels A=60%, B=30%, tie=10% and humans label A=50%, B=35%, tie=15%, the expected agreement $p_e$ is the dot product of the marginals, and $\kappa$ will typically be in the 0.5–0.7 range for a good judge.

### Spearman's Rank Correlation

For pointwise scores, compare judge scores to human quality ratings using Spearman's $\rho$ (rank correlation, preferred to Pearson because scores are ordinal):

$$
\rho = 1 - \frac{6 \sum d_i^2}{n(n^2-1)}
$$

where $d_i$ is the rank difference for the $i$-th example. A good LLM judge achieves $\rho$ in the range of 0.7–0.9 on well-defined criteria.

### Practical Calibration Pipeline

```python
import json
from collections import Counter
from scipy.stats import spearmanr

def calibrate_judge(judge_fn, calibration_set):
    """
    calibration_set: list of dicts with keys:
      - "question": str
      - "response": str
      - "human_score": int (1-5)
    
    Returns calibration report dict.
    """
    judge_scores = []
    human_scores = []
    
    for ex in calibration_set:
        result = judge_fn(ex["question"], ex["response"])
        judge_scores.append(result["score"])
        human_scores.append(ex["human_score"])
    
    # Spearman correlation
    rho, pvalue = spearmanr(judge_scores, human_scores)
    
    # Score distribution
    judge_dist = Counter(judge_scores)
    human_dist = Counter(human_scores)
    
    # Mean absolute error
    mae = sum(abs(j - h) for j, h in zip(judge_scores, human_scores)) / len(judge_scores)
    
    return {
        "spearman_rho": round(rho, 3),
        "p_value": round(pvalue, 4),
        "mae": round(mae, 3),
        "judge_score_distribution": dict(sorted(judge_dist.items())),
        "human_score_distribution": dict(sorted(human_dist.items())),
        "n_examples": len(calibration_set),
    }
```

---

## A Complete Judge Implementation

Here is a production-ready pointwise judge with CoT-before-score, retries, and structured output parsing.

```python
import json
import re
import time
from typing import Optional
from openai import OpenAI  # or any chat completion API

client = OpenAI()  # set OPENAI_API_KEY in environment

JUDGE_SYSTEM_PROMPT = """\
You are a rigorous, calibrated evaluator of AI assistant responses.

Your task: evaluate the RESPONSE to the QUESTION on the following criteria.
For each criterion, think step-by-step before scoring.

Criteria (each scored 1-5):
  correctness    – factual accuracy; 1=major errors, 5=fully accurate
  helpfulness    – directly addresses the user's need; 1=unhelpful, 5=maximally helpful
  conciseness    – appropriate length; 1=bloated/terse to detriment, 5=perfectly sized
  safety         – avoids harm; 1=harmful content, 5=completely safe

Rules:
- Score based on content quality, NOT on length, formatting polish, or style.
- A short accurate answer is better than a long inaccurate one.
- Provide a brief rationale (1 sentence per criterion) before each score.

Output a JSON object with this exact structure:
{
  "correctness":   {"rationale": "...", "score": <1-5>},
  "helpfulness":   {"rationale": "...", "score": <1-5>},
  "conciseness":   {"rationale": "...", "score": <1-5>},
  "safety":        {"rationale": "...", "score": <1-5>}
}
Do not output anything outside the JSON block.
"""

def judge_response(
    question: str,
    response: str,
    model: str = "gpt-4o",
    max_retries: int = 3,
) -> Optional[dict]:
    """
    Evaluate a single (question, response) pair.
    Returns a dict with per-criterion {rationale, score} and an overall score,
    or None if parsing fails after retries.
    """
    user_message = f"QUESTION:\n{question}\n\nRESPONSE:\n{response}"
    
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                temperature=0.0,   # Deterministic for reproducibility
                max_tokens=512,
                response_format={"type": "json_object"},  # GPT-4o feature
            )
            raw = completion.choices[0].message.content
            result = json.loads(raw)
            
            # Validate structure
            criteria = ["correctness", "helpfulness", "conciseness", "safety"]
            for c in criteria:
                assert c in result
                assert "score" in result[c]
                assert 1 <= int(result[c]["score"]) <= 5
            
            # Compute a weighted overall score
            weights = {"correctness": 0.4, "helpfulness": 0.4,
                       "conciseness": 0.1, "safety": 0.1}
            overall = sum(weights[c] * result[c]["score"] for c in criteria)
            result["overall"] = round(overall, 2)
            return result
        
        except (json.JSONDecodeError, KeyError, AssertionError) as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"Judge parsing failed after {max_retries} attempts: {e}")
                return None


# ---------- Usage example ----------
if __name__ == "__main__":
    q = "What is the capital of France?"
    r_good = "Paris is the capital and largest city of France."
    r_verbose = (
        "Great question! France, a beautiful country in Western Europe, "
        "has many wonderful cities. The capital city is Paris, which is "
        "located in the north of the country along the Seine River. "
        "Paris is famous for the Eiffel Tower and amazing cuisine!"
    )
    
    score_good    = judge_response(q, r_good)
    score_verbose = judge_response(q, r_verbose)
    
    print(f"Concise response overall:  {score_good['overall']}")
    print(f"Verbose response overall:  {score_verbose['overall']}")
    # A well-calibrated judge should penalize verbosity here
```

---

## Chatbot Arena and the Elo Rating System

Chatbot Arena (Zheng et al., LMSYS, 2023) is the crowd-sourced human preference platform that pits two anonymous models against each other on a user-provided prompt. Millions of such pairwise battles are aggregated into a single ranking via the **Elo rating system**, originally designed for chess.

### The Elo Model

Each model $i$ has a latent skill rating $\theta_i$. Given two models with ratings $\theta_A$ and $\theta_B$, the probability that A beats B follows the logistic function:

$$
P(A \succ B) = \frac{1}{1 + 10^{(\theta_B - \theta_A)/400}}
$$

The constant 400 (from chess convention) sets the scale so that a 400-point rating gap corresponds to a 10:1 win-rate advantage.

After each game with outcome $s_A \in \{1, 0.5, 0\}$ (win, tie, loss for A):

$$
\theta_A \leftarrow \theta_A + K \cdot (s_A - P(A \succ B))
$$

$$
\theta_B \leftarrow \theta_B + K \cdot (s_B - P(B \succ A))
$$

where $K$ is a step size (typically 32–64).

### From LLMaaJ to Arena Elo

We can run the same arena protocol with an LLM judge instead of humans. Each evaluation prompt becomes a "battle": run two model responses through the judge, record the outcome, and update Elo ratings. This is sometimes called **automated arena** or **judge-based ranking**.

```python
import math
import random
from collections import defaultdict

class EloRanker:
    """
    Maintains Elo ratings for a set of LLM models.
    Ratings start at 1000 (chess convention baseline).
    """
    
    def __init__(self, k: float = 32.0, base: float = 10.0, scale: float = 400.0):
        self.k = k
        self.base = base
        self.scale = scale
        self.ratings: dict[str, float] = defaultdict(lambda: 1000.0)
        self.game_counts: dict[str, int] = defaultdict(int)
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """P(A wins) under Elo model."""
        return 1.0 / (1.0 + self.base ** ((rating_b - rating_a) / self.scale))
    
    def update(self, model_a: str, model_b: str, outcome: str) -> None:
        """
        outcome: "A" (A wins), "B" (B wins), or "tie"
        Updates ratings in place.
        """
        score_a = {"A": 1.0, "B": 0.0, "tie": 0.5}[outcome]
        score_b = 1.0 - score_a
        
        ra, rb = self.ratings[model_a], self.ratings[model_b]
        ea = self.expected_score(ra, rb)
        eb = 1.0 - ea
        
        self.ratings[model_a] += self.k * (score_a - ea)
        self.ratings[model_b] += self.k * (score_b - eb)
        self.game_counts[model_a] += 1
        self.game_counts[model_b] += 1
    
    def leaderboard(self) -> list[tuple[str, float, int]]:
        """Returns [(model_name, rating, n_games)] sorted by rating desc."""
        return sorted(
            [(m, round(r, 1), self.game_counts[m]) for m, r in self.ratings.items()],
            key=lambda x: -x[1],
        )


def run_automated_arena(
    judge_fn,       # callable(question, resp_a, resp_b) -> "A"|"B"|"tie"
    prompts: list[str],
    models: dict[str, callable],  # name -> generate_fn(prompt) -> str
    n_battles: int = 500,
    k: float = 32.0,
) -> EloRanker:
    """
    Run an automated arena: sample random prompt/model pairs,
    call judge, update Elo.
    """
    ranker = EloRanker(k=k)
    model_names = list(models.keys())
    
    for battle_idx in range(n_battles):
        # Sample a random prompt and two distinct models
        prompt = random.choice(prompts)
        a, b = random.sample(model_names, 2)
        
        # Generate responses
        resp_a = models[a](prompt)
        resp_b = models[b](prompt)
        
        # Judge (debiased: run both orderings)
        out_ab = judge_fn(prompt, resp_a, resp_b)
        out_ba_raw = judge_fn(prompt, resp_b, resp_a)
        flip = {"A": "B", "B": "A", "tie": "tie"}
        out_ba = flip[out_ba_raw]
        
        outcome = out_ab if out_ab == out_ba else "tie"
        
        ranker.update(a, b, outcome)
        
        if (battle_idx + 1) % 100 == 0:
            print(f"\n--- Battle {battle_idx+1} ---")
            for name, rating, n in ranker.leaderboard():
                print(f"  {name:<25} {rating:>7.1f}  ({n} games)")
    
    return ranker
```

!!! example "Worked example: Elo update after a single battle"

    Suppose model A has rating 1050 and model B has rating 980, and the judge declares A wins.

    $$
    P(A \succ B) = \frac{1}{1 + 10^{(980-1050)/400}} = \frac{1}{1 + 10^{-0.175}} = \frac{1}{1 + 0.668} \approx 0.599
    $$

    With $K = 32$:

    $$
    \theta_A' = 1050 + 32 \times (1.0 - 0.599) = 1050 + 12.8 = 1062.8
    $$

    $$
    \theta_B' = 980 + 32 \times (0.0 - 0.401) = 980 - 12.8 = 967.2
    $$

    The win was somewhat expected (A was already rated higher), so the rating shift is moderate — about 13 points each way.

    If instead B (the weaker model) had won, the update would be larger in magnitude ($32 \times (1.0 - 0.401) \approx 19.2$ for B), because upsets carry more information.

{{fig:elo-battles-to-ranking}}

---

## Reward Models as Judges

The LLMaaJ setup using a chat model as judge has a sibling: the dedicated **reward model** (RM). An RM is a transformer with a scalar head trained on human preference data (typically Bradley-Terry logistic regression over chosen/rejected pairs). For full background see [The RLHF Pipeline & Reward Modeling](../05-posttraining-alignment/05-rlhf-reward-modeling.html).

### When to Use an RM vs. a Chat Judge

| Dimension | Chat-model judge | Reward model |
|---|---|---|
| Latency | 1–5 s / query (full generation) | 10–50 ms / query (single forward pass) |
| Throughput | Low (API rate limits) | High (can run at training throughput) |
| Calibration | Strong out-of-the-box | Requires careful training data curation |
| Interpretability | Returns rationale text | Returns a scalar |
| Customizability | Prompt engineering | Requires re-training |
| Self-preference risk | High (same family) | Low (separate model) |

In practice, large RL training loops (see [Policy Gradients & PPO for Language Models](../05-posttraining-alignment/06-ppo-for-llms.html)) use reward models because they must call the judge millions of times per training run. Chat-model judges are used for held-out offline evaluation at lower throughput requirements.

### Reward Model as a Pointwise Judge

A reward model produces a scalar $r \in \mathbb{R}$ (after the scalar head), higher is better. To use it as a judge:

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class RewardModelJudge:
    """
    Wraps a Bradley-Terry reward model as a pointwise judge.
    Model must output a scalar logit (or use a single-logit head).
    """
    
    def __init__(self, model_name: str = "OpenAssistant/reward-model-deberta-v3-large-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
    
    @torch.no_grad()
    def score(self, question: str, response: str) -> float:
        """
        Returns a reward scalar. Higher = better.
        The model expects a concatenated [question, response] input
        in a specific format (model-dependent).
        """
        # Format as the model expects (typically question + sep + response)
        text = f"{question}\n\n{response}"
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        logits = self.model(**tokens).logits  # shape: (1, 1) for scalar head
        return logits.squeeze().item()
    
    def pairwise_judge(self, question: str, resp_a: str, resp_b: str) -> str:
        """Returns 'A', 'B', or 'tie' based on reward scores."""
        score_a = self.score(question, resp_a)
        score_b = self.score(question, resp_b)
        margin = abs(score_a - score_b)
        
        if margin < 0.05:   # Small margin → call it a tie
            return "tie"
        return "A" if score_a > score_b else "B"
```

### Reward Model Calibration Issues

Reward models are susceptible to **reward hacking** — subtle distributional artifacts that inflate scores without genuine quality improvement. Common failure modes:

- **Length hacking:** RM scores correlate with response length independently of quality.
- **Sycophancy reward:** RM was trained on data where agreeable responses were preferred, so it over-rewards agreement.
- **Distribution shift:** RM generalizes poorly to prompts outside its training distribution.

These issues are covered in depth in [Reward Hacking, Over-Optimization & Alignment Failures](../05-posttraining-alignment/13-reward-hacking-failures.html). From an evaluation standpoint: always compute a calibration Spearman $\rho$ against held-out human labels before trusting RM scores as ground truth.

---

## Judge Calibration, Auditing, and Trust

Deploying a judge in production without calibration is the most common mistake. Here is a systematic protocol.

### The Judge Calibration Checklist

1. **Build a gold set.** Collect 300–500 (prompt, response) examples spanning the full quality range. Get majority-vote labels from at least 3 annotators. Include adversarial examples (long-but-wrong, short-but-right, off-topic, etc.).

2. **Compute baseline metrics.** Run the judge; compute Spearman $\rho$, MAE, bias profile, and per-criterion calibration.

3. **Measure known biases.** For each of the biases in the previous section, construct minimal test pairs (e.g., identical content with different lengths to test verbosity bias).

4. **Iterate on the judge prompt.** If verbosity bias is measured at >5 pp win-rate difference between length-matched pairs, strengthen the anti-length instruction in the prompt.

5. **Set acceptance thresholds.** Before deploying a new judge or judge model, require Spearman $\rho > 0.75$ and agreement rate >70% on the gold set.

6. **Re-calibrate on model updates.** When you upgrade the judge model (e.g., GPT-4o to GPT-4o-mini), re-run calibration. Score distributions can shift.

{{fig:judge-calibration-trust-loop}}

### Confidence Intervals for Elo Rankings

A single Elo rating is not a stable estimate — it has uncertainty depending on the number of games played. Use bootstrap resampling to get confidence intervals:

```python
import numpy as np
from copy import deepcopy

def bootstrap_elo_ci(battle_log: list[dict], n_bootstrap: int = 200, k: float = 32.0):
    """
    battle_log: list of {"model_a": str, "model_b": str, "outcome": str}
    Returns a dict of model -> (mean_rating, lower_95, upper_95)
    """
    all_ratings = defaultdict(list)
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        sample = random.choices(battle_log, k=len(battle_log))
        
        ranker = EloRanker(k=k)
        for battle in sample:
            ranker.update(battle["model_a"], battle["model_b"], battle["outcome"])
        
        for model, rating, _ in ranker.leaderboard():
            all_ratings[model].append(rating)
    
    results = {}
    for model, ratings in all_ratings.items():
        arr = np.array(ratings)
        results[model] = {
            "mean":   round(float(np.mean(arr)), 1),
            "lower":  round(float(np.percentile(arr, 2.5)), 1),
            "upper":  round(float(np.percentile(arr, 97.5)), 1),
        }
    return results
```

If the 95% confidence intervals of two models overlap, you do not have statistically significant evidence that one outperforms the other — more battles are needed.

---

## Production LLMaaJ: Design Considerations

### Caching and Idempotency

Judge calls are expensive (API cost + latency). Cache results keyed on `(judge_model, judge_prompt_hash, question_hash, response_hash)`. This makes re-runs cheap and evaluation truly reproducible.

```python
import hashlib, json, sqlite3

def cache_key(judge_model, system_prompt, question, response):
    payload = json.dumps([judge_model, system_prompt, question, response], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()
```

### Parallelism

For bulk evaluation (thousands of queries), use async API calls with rate-limit-aware batching:

```python
import asyncio
from openai import AsyncOpenAI

async_client = AsyncOpenAI()

async def judge_async(question: str, response: str, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        completion = await async_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user",   "content": f"QUESTION:\n{question}\n\nRESPONSE:\n{response}"},
            ],
            temperature=0.0,
            max_tokens=512,
            response_format={"type": "json_object"},
        )
        return json.loads(completion.choices[0].message.content)

async def bulk_judge(examples: list[dict], max_concurrency: int = 20) -> list[dict]:
    """
    examples: list of {"question": str, "response": str}
    max_concurrency: tune to stay under API rate limits
    """
    sem = asyncio.Semaphore(max_concurrency)
    tasks = [judge_async(ex["question"], ex["response"], sem) for ex in examples]
    return await asyncio.gather(*tasks)
```

### Judge Cost Model

For a daily evaluation run of 10,000 examples using GPT-4o (on the order of USD 5 per million input tokens as of mid-2025):

- Average prompt tokens: ~500 (system + question + response)
- Average output tokens: ~150 (rationale + JSON)
- Total tokens: 10,000 × 650 = 6.5M tokens
- Rough cost: on the order of USD 30–50 per run

For cost-sensitive pipelines, use a smaller judge model (GPT-4o-mini, Claude Haiku) for daily monitoring and reserve GPT-4o or Claude Opus for milestone evaluations. Always validate that the smaller judge maintains acceptable calibration.

!!! interview "Interview Corner"

    **Q:** You are building an LLM judge to evaluate daily model updates. Your colleague argues that because the LLM judge isn't perfect, you should just do human evaluation for all comparisons. How do you respond, and what would you do to ensure the judge is trustworthy?

    **A:** Human evaluation doesn't scale to a daily cadence across thousands of examples — it is too slow and expensive. The right answer is not to choose one over the other, but to layer them: use the LLM judge for high-frequency, lower-stakes monitoring (detecting regressions, comparing ablations), and use human evaluation for milestone decisions and for building the calibration set that validates the judge. To make the judge trustworthy: (1) build a gold calibration set with majority-vote human labels, (2) measure and mitigate known biases (position, verbosity, self-preference) with debiasing protocols like running both orderings, (3) require Spearman ρ > 0.75 and >70% agreement before deployment, (4) use a different model family for judging than for generation to reduce self-preference, and (5) re-calibrate when the judge model is upgraded. The judge is a tool with known failure modes, not a black box to trust blindly.

---

!!! key "Key Takeaways"

    - LLM-as-a-Judge (LLMaaJ) enables scalable automated evaluation at inference speed, but must be treated as an approximate human proxy — not ground truth.
    - Pointwise evaluation is efficient (O(N) queries) and good for monitoring; pairwise is more calibrated for head-to-head comparisons but costs O(N²) and has strong position bias.
    - The five principal biases are: position, verbosity, self-preference, sycophancy, and instruction-following bias — each has a specific mitigation (run both orderings, explicit anti-length instructions, use different model family, neutral prompts, separate rubric criteria).
    - Rubric-based multi-criteria evaluation (e.g., correctness, helpfulness, conciseness, safety weighted separately) provides more diagnostic signal than a single aggregate score.
    - Chain-of-thought before scoring (G-Eval style) significantly improves calibration versus asking for a bare score.
    - The Elo rating system converts pairwise win/loss outcomes into a global ranking; bootstrap resampling gives confidence intervals around ratings.
    - Reward models are faster judges (10–50 ms vs. 1–5 s) suitable for training-loop reward signals; chat judges are more flexible and interpretable for offline evaluation.
    - Always validate any judge — LLM or RM — against a gold human-labeled calibration set and track Spearman ρ and bias profile before trusting it in production.
    - Cache judge outputs deterministically (hash of judge model + prompt + inputs) to make evaluations cheap, reproducible, and diffable.

---

!!! sota "State of the Art & Resources (2026)"
    LLM-as-a-Judge has become the standard backbone of automated evaluation pipelines since 2023, with a rich body of work characterizing its biases and open-source tooling now mature enough for production use. Dedicated open-source judge models (Prometheus) and length-debiased benchmarks (LC-AlpacaEval) have largely addressed early criticism about reliability, while crowd-sourced platforms like Chatbot Arena provide large-scale human-preference ground truth.

    **Foundational work**

    - [Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* (2023)](https://arxiv.org/abs/2306.05685) — the paper that named and systematized the LLMaaJ paradigm, introduced MT-Bench, and measured position/verbosity/self-preference biases.
    - [Liu et al., *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment* (2023)](https://arxiv.org/abs/2303.16634) — showed that chain-of-thought before scoring dramatically improves correlation with human judgments on summarization and dialogue.
    - [Wang et al., *Large Language Models are not Fair Evaluators* (2023)](https://arxiv.org/abs/2305.17926) — systematic measurement of position bias; proposed swap-ordering calibration (the "run both orderings" mitigation used in production).

    **Recent advances (2023–2026)**

    - [Kim et al., *Prometheus: Inducing Fine-grained Evaluation Capability in Language Models* (2023)](https://arxiv.org/abs/2310.08491) — first open-source judge model matching GPT-4 on rubric-based evaluation; avoids proprietary API dependency.
    - [Kim et al., *Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models* (2024)](https://arxiv.org/abs/2405.01535) — extends Prometheus to both absolute and pairwise grading; achieves 72–85% agreement with human judgments across benchmarks.
    - [Dubois et al., *Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators* (2024)](https://arxiv.org/abs/2404.04475) — regression-based length normalization that removes the dominant verbosity bias from LLM-based win-rate estimates.
    - [Chiang et al., *Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference* (2024)](https://arxiv.org/abs/2403.04132) — describes the statistical backbone (Bradley-Terry model) of the Arena platform and its use as a large-scale human-preference ground truth.

    **Open-source & tools**

    - [prometheus-eval/prometheus](https://github.com/prometheus-eval/prometheus) — official repo for the Prometheus judge model family; includes training code, the Feedback Collection dataset, and inference utilities.
    - [tatsu-lab/alpaca_eval](https://github.com/tatsu-lab/alpaca_eval) — AlpacaEval harness with length-controlled win-rate; achieves 0.98 Spearman correlation with Chatbot Arena at under $10 per run.
    - [openai/evals](https://github.com/openai/evals) — OpenAI's eval framework with model-graded eval templates and a community registry of benchmarks.
    - [confident-ai/deepeval](https://github.com/confident-ai/deepeval) — production-ready pytest-style LLM evaluation framework with 40+ built-in metrics including G-Eval, RAG faithfulness, and hallucination detection.

## Further Reading

- **Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (LMSYS, 2023)** — the foundational paper defining the LLMaaJ paradigm and MT-Bench; analyzes position and verbosity biases.
- **Liu et al., "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment" (2023)** — introduces CoT-before-score and shows that using GPT-4 with structured criteria significantly correlates with human judgments on summarization and dialogue.
- **Wang et al., "Large Language Models are not Fair Evaluators" (2023)** — systematic measurement of position bias and the swap-ordering mitigation.
- **Dubois et al., "AlpacaEval: An Automatic Evaluator for Instruction-following Language Models" (2023)** — the AlpacaEval benchmark; open-source automated evaluation harness with length-controlled win rate.
- **Chiang & Lee, "Can Large Language Models Be an Alternative to Human Evaluations?" (ACL 2023)** — empirical study of agreement rates and failure modes.
- **Ouyang et al., "Training language models to follow instructions with human feedback" (InstructGPT, 2022)** — reward modeling methodology that underpins RM-as-judge.
- **Chatbot Arena Leaderboard (LMSYS)** — live crowdsourced human preference ranking; a model for large-scale Elo-based evaluation.
- **OpenAI Evals** (github.com/openai/evals) — open-source eval framework with LLMaaJ templates and model-graded eval types.

---

## Exercises

**1.** Pairwise judging achieves higher agreement with human preference than pointwise scoring (see the "When to Use Each" table), yet the same table lists pairwise as having *high* susceptibility to position bias while pointwise is *low*. Explain, in terms of what each paradigm asks the judge to do, why relative comparison is both better calibrated *and* newly vulnerable to position bias — a failure mode pointwise scoring essentially does not have.

??? note "Solution"
    The two properties come from the same source: pairwise judging places both responses in a single context and asks a *relative* question ("which is better?").

    - **Why better calibrated.** As the chapter notes, relative judgments are cognitively easier and better anchored than absolute ones — "the judge (like a human) can often tell which of two things is better even when scoring each on an absolute scale is hard." Pointwise scoring forces the judge to map a response onto an absolute 1-5 scale with no anchor, so its notion of what a "4" means drifts across prompt types. Pairwise removes the need for a stable absolute scale: the two responses anchor each other.

    - **Why newly vulnerable to position bias.** Position bias is defined only for pairwise setups: the judge "systematically favors whichever response appears first (or second)." This can only happen because both candidates share one prompt and occupy distinct slots (A vs B). Pointwise scoring evaluates a *single* response in isolation — there is no "first" or "second" slot for the model to anchor on, so ordering cannot bias it (the table lists pointwise position-bias susceptibility as low for exactly this reason). The very structure that lets the two responses anchor each other for better calibration is the structure that introduces a slot the judge can spuriously favor.

    The chapter's mitigation (run both orderings and aggregate) directly targets this: it cancels the slot preference while preserving the relative-comparison advantage.

**2.** Use the weighted aggregation from the chapter's `judge_response` implementation, with weights `correctness=0.4, helpfulness=0.4, conciseness=0.1, safety=0.1`. A judge returns these per-criterion scores for the two responses in the chapter's usage example:

- Concise answer ("Paris is the capital..."): correctness 5, helpfulness 5, conciseness 5, safety 5.
- Verbose answer ("Great question! France..."): correctness 5, helpfulness 4, conciseness 2, safety 5.

Compute the `overall` score for each. Does the aggregation reward the concise answer, as the code comment ("A well-calibrated judge should penalize verbosity here") predicts? Which single criterion drives the gap, and by how much of the overall difference?

??? note "Solution"
    Apply $S_{\text{overall}} = \sum_c w_c \cdot s_c$.

    Concise answer:
    $$
    S = 0.4(5) + 0.4(5) + 0.1(5) + 0.1(5) = 2.0 + 2.0 + 0.5 + 0.5 = 5.00
    $$

    Verbose answer:
    $$
    S = 0.4(5) + 0.4(4) + 0.1(2) + 0.1(5) = 2.0 + 1.6 + 0.2 + 0.5 = 4.30
    $$

    The concise answer scores higher (5.00 vs 4.30), so yes, the aggregation penalizes verbosity as intended.

    Decompose the gap of $5.00 - 4.30 = 0.70$ by criterion:

    - helpfulness: $0.4(5-4) = 0.40$
    - conciseness: $0.1(5-2) = 0.30$
    - correctness and safety: $0$ each (tied at 5).

    The **helpfulness** criterion contributes the larger share ($0.40$ of the $0.70$ gap), even though conciseness is the criterion "about" length. This illustrates a subtlety from the chapter: because conciseness carries only weight $0.1$, a large 3-point conciseness deficit ($0.30$) is worth less than a 1-point helpfulness deficit at weight $0.4$. If verbosity manifests mainly as padding that dilutes helpfulness, the low conciseness weight still catches it indirectly; but if you want length itself to dominate the penalty, you would need to raise $w_{\text{conciseness}}$ (weights "should be set based on product requirements and validated against human preference labels").

**3.** Two models enter an automated arena using the chapter's `EloRanker` (start rating 1000, $K = 32$, base 10, scale 400).

(a) In the very first battle, model A and model B are both at 1000 and the judge declares A the winner. Compute both new ratings.

(b) Later, model A sits at 1200 and model B at 1000. Compute $P(A \succ B)$, the judge-free expected win probability.

(c) The chapter says a 400-point gap corresponds to a 10:1 win-rate advantage. What rating gap corresponds to a 3:1 advantage (i.e. $P(A \succ B) = 0.75$)?

??? note "Solution"
    **(a)** With equal ratings, $P(A \succ B) = \frac{1}{1 + 10^{0}} = 0.5$. Outcome A wins, so $s_A = 1$, $s_B = 0$:
    $$
    \theta_A' = 1000 + 32(1 - 0.5) = 1000 + 16 = 1016
    $$
    $$
    \theta_B' = 1000 + 32(0 - 0.5) = 1000 - 16 = 984
    $$
    A rises to **1016**, B falls to **984**. With a completely uninformative prior (equal ratings), the update is the maximum symmetric $\pm K/2 = \pm 16$.

    **(b)** Gap $= \theta_A - \theta_B = 200$:
    $$
    P(A \succ B) = \frac{1}{1 + 10^{(1000-1200)/400}} = \frac{1}{1 + 10^{-0.5}} = \frac{1}{1 + 0.3162} = \frac{1}{1.3162} \approx 0.760
    $$
    So about a **76%** expected win rate for A.

    **(c)** A 3:1 advantage means $P = 0.75$, i.e. odds of 3. From the logistic model, odds $= 10^{\Delta/400}$ where $\Delta$ is the gap. Solve:
    $$
    10^{\Delta/400} = 3 \;\Rightarrow\; \frac{\Delta}{400} = \log_{10} 3 = 0.4771 \;\Rightarrow\; \Delta = 400 \times 0.4771 \approx 191
    $$
    A gap of about **191 points** gives a 3:1 (75%) win-rate advantage. (Sanity check: the 200-point gap in part (b) gave 0.760, just above 0.75, consistent with the required gap being slightly under 200.)

**4.** You audit a pairwise judge against a human-labeled gold set. The judge's label marginals are A $= 60\%$, B $= 30\%$, tie $= 10\%$; the human majority-vote marginals are A $= 50\%$, B $= 35\%$, tie $= 15\%$ (these are exactly the numbers in the chapter's Cohen's-kappa section). Suppose the judge and human labels agree on $72\%$ of examples ($p_o = 0.72$). Compute Cohen's $\kappa$. Interpret the result against the chapter's stated range for a "good judge," and explain why raw agreement of $72\%$ overstates the judge's quality.

??? note "Solution"
    Expected chance agreement $p_e$ is the dot product of the two label marginals (probability both raters land on the same category by chance):
    $$
    p_e = (0.60)(0.50) + (0.30)(0.35) + (0.10)(0.15) = 0.300 + 0.105 + 0.015 = 0.420
    $$
    Then:
    $$
    \kappa = \frac{p_o - p_e}{1 - p_e} = \frac{0.72 - 0.42}{1 - 0.42} = \frac{0.30}{0.58} \approx 0.517
    $$
    So $\kappa \approx 0.52$, which sits at the low end of the chapter's "0.5-0.7 range for a good judge" — acceptable but not strong.

    Raw agreement of $72\%$ overstates quality because both raters lean heavily toward label A (judge 60%, human 50%), so they would collide on A frequently *even if the judge answered randomly according to its marginals*. That chance floor is $p_e = 42\%$. Kappa rescales the observed agreement into the band above chance: of the $1 - 0.42 = 0.58$ "room" for non-chance agreement, the judge captured $0.30$, i.e. about $52\%$ of the achievable non-chance agreement. Reporting $72\%$ alone hides the fact that nearly six-tenths of that agreement is attributable to shared marginal bias rather than genuine matching judgment.

**5.** The chapter's `pairwise_judge_debiased` runs both orderings and returns `"tie"` whenever they disagree — but it *discards* the fact that a disagreement happened, which is exactly the signal you need for the calibration protocol's "bias profile per ordering." Implement a function

    ```python
    def measure_position_bias(judge_fn, pairs):
        ...
    ```

    that runs both orderings for every pair and estimates the judge's *first-position preference rate*: among all non-tie decisions across both orderings, the fraction in which the judge picked whichever response was presented **first**. Under an unbiased judge this rate should be about $0.50$; a value well above $0.50$ indicates first-position bias. `pairs` is a list of `(question, response_a, response_b)` tuples; `judge_fn(question, first, second)` returns `"A"`, `"B"`, or `"tie"`, where `"A"` always means "the first-presented response won." Return the rate (or `None` if there are no decisive judgments).

??? note "Solution"
    The key observation is that `judge_fn` reports its verdict in *slot* terms, not identity terms: `"A"` always means the first-presented response won and `"B"` means the second won. So we do **not** flip results back into A/B identity space (as the debiasing helper does) — we count slot wins directly. Each pair yields two independent decisions (one per ordering); we pool all decisive ones.

    ```python
    def measure_position_bias(judge_fn, pairs):
        """
        pairs: list of (question, response_a, response_b)
        judge_fn(question, first, second) -> "A" | "B" | "tie"
            where "A" means the FIRST-presented response won.

        Returns the first-position preference rate: among all non-tie
        decisions (2 per pair), the fraction won by the first slot.
        ~0.5 => unbiased; >0.5 => favors first position. None if no
        decisive judgments.
        """
        first_wins = 0
        decisive = 0

        for question, resp_a, resp_b in pairs:
            # Ordering 1: resp_a first, resp_b second
            r1 = judge_fn(question, resp_a, resp_b)
            # Ordering 2: swap so resp_b is now first
            r2 = judge_fn(question, resp_b, resp_a)

            for r in (r1, r2):
                if r == "tie":
                    continue
                decisive += 1
                if r == "A":        # "A" == first-presented slot won
                    first_wins += 1

        if decisive == 0:
            return None
        return first_wins / decisive
    ```

    Why this measures position bias cleanly: for any given pair, if the judge were driven purely by content, the *same underlying response* would win in both orderings, so its slot-win would land in the first slot once and the second slot once — contributing one first-win and one second-win, i.e. a net of $0.5$. Only a genuine slot preference pushes the pooled rate away from $0.5$. Averaged over many pairs, a rate of, say, $0.62$ means the judge favors the first-presented response about $62\%$ of the time among decisive calls -- a first-position bias of roughly $12$ percentage points, in the range the chapter cites ("10-15 percentage points on win rate").

    This is the per-ordering "bias profile" the calibration protocol asks for. If the rate exceeds an acceptance threshold you would either strengthen the judge prompt or rely on the both-orderings aggregation in `pairwise_judge_debiased` for all production calls.
