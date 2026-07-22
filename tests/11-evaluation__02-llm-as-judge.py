"""
Runs the CPU-runnable Python code blocks from:
    content/11-evaluation/02-llm-as-judge.md

Blocks are copied faithfully (verbatim logic) and concatenated in document
order. Minimal glue (tiny mocked judge functions / fixture data) is added so
each block actually EXECUTES on CPU with no network access.

Tested blocks:
    #2 (line ~112, RUBRIC_CRITERIA dict)
    #4 (line ~255, calibrate_judge -- Spearman correlation calibration report)
    #6 (line ~440, EloRanker + run_automated_arena)
    #8 (line ~655, bootstrap_elo_ci)
    #9 (line ~698, cache_key)

Skipped blocks:
    #0 (line ~35)  -- ```text``` pointwise judge prompt, not Python.
    #1 (line ~62)  -- ```text``` pairwise judge prompt, not Python.
    #3 (line ~158) -- pairwise_judge_debiased(): a bare utility-function
                       fragment (needs an external judge_fn injected by the
                       caller) with no standalone exercise/assert in the book
                       itself; marked "fragment" per the task brief.
    #5 (line ~304) -- needs-net: imports `openai.OpenAI`, calls the real
                       Chat Completions API (client.chat.completions.create).
    #7 (line ~576) -- needs-net: `RewardModelJudge` downloads
                       "OpenAssistant/reward-model-deberta-v3-large-v2" from
                       the HF hub via AutoTokenizer/AutoModelForSequenceClassification.
    #10 (line ~711) -- needs-net: `AsyncOpenAI` + real async API calls.

Third-party dependency note: Block #4 uses `scipy.stats.spearmanr`. scipy is
not in this task's guaranteed-available list (numpy/torch/einops/sklearn/
stdlib only), so the import is guarded; if scipy is unavailable, Block #4's
`calibrate_judge` function is still defined but not called (skipped at
runtime) rather than erroring the whole file.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from copy import deepcopy

import numpy as np

try:
    from scipy.stats import spearmanr
    _HAVE_SCIPY = True
except Exception:
    spearmanr = None
    _HAVE_SCIPY = False


def _section(name: str) -> None:
    print(f"\n=== {name} ===")


# ============================================================================
# Block #2 (line ~112) -- RUBRIC_CRITERIA
# ============================================================================
_section("Block #2: RUBRIC_CRITERIA")

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

assert set(RUBRIC_CRITERIA.keys()) == {
    "correctness", "helpfulness", "safety", "conciseness", "instruction_following",
}
assert all(isinstance(v, str) and len(v) > 0 for v in RUBRIC_CRITERIA.values())
print(f"{len(RUBRIC_CRITERIA)} rubric criteria defined: {list(RUBRIC_CRITERIA)}")

print("Block #2 OK")


# ============================================================================
# Block #4 (line ~255) -- calibrate_judge (Spearman calibration pipeline)
# SKIP(scipy unavailable): calibrate_judge is defined but not invoked if
# scipy.stats.spearmanr could not be imported.
# ============================================================================
_section("Block #4: calibrate_judge")


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


if _HAVE_SCIPY:
    # Tiny fixture calibration set + a mock judge that tracks human_score with
    # some deterministic per-example jitter, so the correlation is realistic
    # (neither perfectly 1.0 nor uncorrelated).
    _fixture_calibration_set = [
        {"question": "Q1", "response": "R1", "human_score": 1},
        {"question": "Q2", "response": "R2", "human_score": 2},
        {"question": "Q3", "response": "R3", "human_score": 2},
        {"question": "Q4", "response": "R4", "human_score": 3},
        {"question": "Q5", "response": "R5", "human_score": 4},
        {"question": "Q6", "response": "R6", "human_score": 4},
        {"question": "Q7", "response": "R7", "human_score": 5},
        {"question": "Q8", "response": "R8", "human_score": 5},
    ]
    _jitter = {1: 0, 2: 1, 3: 0, 4: -1, 5: 0, 6: 1, 7: -1, 8: 0}

    def _mock_judge_fn(question, response):
        idx = int(question[1:])  # "Q3" -> 3
        human = next(ex["human_score"] for ex in _fixture_calibration_set if ex["question"] == question)
        score = min(5, max(1, human + _jitter[idx]))
        return {"score": score, "rationale": f"mock rationale for {question}"}

    report = calibrate_judge(_mock_judge_fn, _fixture_calibration_set)
    print(json.dumps(report, indent=2))

    assert report["n_examples"] == 8
    assert -1.0 <= report["spearman_rho"] <= 1.0
    assert report["mae"] >= 0.0
    assert sum(report["judge_score_distribution"].values()) == 8
    assert sum(report["human_score_distribution"].values()) == 8
    # With only +/-1 jitter around the human scores, correlation should be strongly positive.
    assert report["spearman_rho"] > 0.5, f"expected strong positive correlation, got {report['spearman_rho']}"

    print("Block #4 OK")
else:
    print("Block #4 SKIPPED: scipy not available")


# ============================================================================
# Block #6 (line ~440) -- EloRanker + run_automated_arena
# ============================================================================
_section("Block #6: EloRanker + run_automated_arena")


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


# --- worked example verification (from the book's "Worked example" callout) ---
_wr = EloRanker(k=32.0)
_wr.ratings["A"] = 1050.0
_wr.ratings["B"] = 980.0
_p_a_wins = _wr.expected_score(1050.0, 980.0)
assert abs(_p_a_wins - 0.599) < 1e-3, f"expected P(A>B)~0.599, got {_p_a_wins}"
_wr.update("A", "B", "A")
assert abs(_wr.ratings["A"] - 1062.8) < 0.1
assert abs(_wr.ratings["B"] - 967.2) < 0.1

# --- tiny fixture arena: 3 "models" are deterministic string generators, and
# the judge prefers whichever response is (deterministically) longer, so a
# fixed ranking should emerge (fake_gpt > fake_mid > fake_small). ---
random.seed(0)

_toy_models = {
    "fake_small": lambda prompt: "ok",
    "fake_mid": lambda prompt: "a reasonably useful answer",
    "fake_gpt": lambda prompt: "a thorough, detailed, and carefully reasoned answer to " + prompt,
}
_toy_prompts = ["What is 2+2?", "Explain gravity.", "Name a fruit."]


def _length_judge(question: str, resp_a: str, resp_b: str) -> str:
    if len(resp_a) > len(resp_b):
        return "A"
    if len(resp_b) > len(resp_a):
        return "B"
    return "tie"


_ranker = run_automated_arena(
    judge_fn=_length_judge,
    prompts=_toy_prompts,
    models=_toy_models,
    n_battles=40,
    k=32.0,
)
_board = _ranker.leaderboard()
print("Final leaderboard:", _board)

assert len(_board) == 3
_names_in_order = [name for name, _, _ in _board]
assert _names_in_order[0] == "fake_gpt", f"expected fake_gpt to top the board, got {_names_in_order}"
assert _names_in_order[-1] == "fake_small", f"expected fake_small to be last, got {_names_in_order}"
assert sum(n for _, _, n in _board) == 40 * 2  # each battle updates 2 models' game_counts

print("Block #6 OK")


# ============================================================================
# Block #8 (line ~655) -- bootstrap_elo_ci
# Continues Block #6's EloRanker class.
# n_bootstrap reduced from the book's illustrative default (200) to 20 to
# keep runtime small; same bootstrap logic, fewer resamples.
# ============================================================================
_section("Block #8: bootstrap_elo_ci")


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


# Build a small deterministic battle log where fake_gpt reliably beats the
# other two, so we can check the bootstrap CIs reflect that separation.
_battle_log = []
for i in range(30):
    if i % 3 == 0:
        _battle_log.append({"model_a": "fake_gpt", "model_b": "fake_small", "outcome": "A"})
    elif i % 3 == 1:
        _battle_log.append({"model_a": "fake_gpt", "model_b": "fake_mid", "outcome": "A"})
    else:
        _battle_log.append({"model_a": "fake_mid", "model_b": "fake_small", "outcome": "A"})

random.seed(1)
_ci = bootstrap_elo_ci(_battle_log, n_bootstrap=20, k=32.0)
print(json.dumps(_ci, indent=2))

assert set(_ci.keys()) == {"fake_gpt", "fake_mid", "fake_small"}
for m, stats in _ci.items():
    assert stats["lower"] <= stats["mean"] <= stats["upper"]
# fake_gpt won every battle it was in -> should rank clearly above fake_small.
assert _ci["fake_gpt"]["mean"] > _ci["fake_small"]["mean"]
# `deepcopy` is imported by the book's block; confirm it's usable (not just imported).
assert deepcopy(_ci) == _ci

print("Block #8 OK")


# ============================================================================
# Block #9 (line ~698) -- cache_key
# ============================================================================
_section("Block #9: cache_key")


def cache_key(judge_model, system_prompt, question, response):
    payload = json.dumps([judge_model, system_prompt, question, response], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


_k1 = cache_key("gpt-4o", "You are a judge.", "What is 2+2?", "4")
_k2 = cache_key("gpt-4o", "You are a judge.", "What is 2+2?", "4")
_k3 = cache_key("gpt-4o", "You are a judge.", "What is 2+2?", "five")

assert _k1 == _k2, "cache_key must be deterministic for identical inputs"
assert _k1 != _k3, "cache_key must differ when the response differs"
assert len(_k1) == 64 and all(c in "0123456789abcdef" for c in _k1), "sha256 hexdigest expected"
print(f"cache_key example: {_k1}")

print("Block #9 OK")


print("\nALL BLOCKS PASSED" if _HAVE_SCIPY else "\nALL BLOCKS PASSED (scipy-dependent Block #4 skipped)")
