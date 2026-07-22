"""
Executable test for content/08-agents-harness/08-agent-evaluation.md

Concatenates the chapter's 5 CPU-runnable Python blocks in order and exercises
each one with tiny fixtures so the book's actual code runs end to end.

Blocks covered:
  #0 (line ~46)  run_swebench_task            -- local git apply + subprocess test runner
  #1 (line ~174) Trajectory / outcome & efficiency scoring dataclasses+functions
  #2 (line ~302) pass_at_k / aggregate_pass_at_k estimator
  #3 (line ~421) agent_benchmark_ci (Wilson CI) + mcnemar_test  (scipy guarded)
  #4 (line ~545) EvalConfig + run_eval (minimal reproducible eval harness)

No network calls anywhere in this chapter's runnable blocks. Block #0 uses only
local `git` via subprocess against a throwaway temp repo -- no network involved.
"""

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# scipy is not in the guaranteed-available list (numpy, torch, einops, sklearn,
# stdlib), so guard it per the hard rules. Block #3 defines its functions either
# way; the example call at the bottom of block #3 is skipped if scipy is absent.
try:
    from scipy import stats
except Exception:
    stats = None


# ============================================================
# Block #0 (line ~46) -- Minimal SWE-bench task runner
# ============================================================
import subprocess, tempfile, os, json

def run_swebench_task(repo_path: str, patch: str, test_cmd: str) -> bool:
    """
    Apply a patch to a repo clone and run the test suite.
    Returns True if all tests pass.

    repo_path: path to a fresh checkout at the pre-patch commit
    patch:     git-diff-style string produced by the agent
    test_cmd:  e.g. "pytest tests/test_models.py -x -q"
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
        f.write(patch)
        patch_file = f.name

    try:
        # Step 1: apply the patch
        result = subprocess.run(
            ["git", "apply", "--check", patch_file],
            cwd=repo_path,
            capture_output=True, text=True
        )
        if result.returncode != 0:
            # patch doesn't apply cleanly → immediate FAIL
            return False

        subprocess.run(["git", "apply", patch_file], cwd=repo_path, check=True)

        # Step 2: run the test suite
        result = subprocess.run(
            test_cmd.split(),
            cwd=repo_path,
            capture_output=True, text=True,
            timeout=120  # hard wall-clock limit per task
        )
        # pytest returns 0 only when all tests pass
        return result.returncode == 0

    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(patch_file)
        # Reset the repo for the next task
        subprocess.run(["git", "checkout", "--", "."], cwd=repo_path)


def _test_block0():
    """Glue: build a throwaway local git repo, craft a real patch via `git diff`,
    and a tiny checker script, then exercise run_swebench_task's success and
    failure paths. All local subprocess calls; no network."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = os.path.join(tmp, "repo")
        os.makedirs(repo)
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)

        file_path = os.path.join(repo, "file.txt")
        with open(file_path, "w") as f:
            f.write("line1\n")
        subprocess.run(["git", "add", "file.txt"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)

        # Craft a real patch: modify the file, capture `git diff`, then revert
        # working tree so run_swebench_task applies it from a clean pre-patch state.
        with open(file_path, "a") as f:
            f.write("line2\n")
        diff_result = subprocess.run(
            ["git", "diff"], cwd=repo, capture_output=True, text=True, check=True
        )
        patch = diff_result.stdout
        assert "line2" in patch
        subprocess.run(["git", "checkout", "--", "."], cwd=repo, check=True)

        # Checker script: exits 0 iff "line2" is present in file.txt (i.e. patch applied)
        check_script = os.path.join(tmp, "check.py")
        with open(check_script, "w") as f:
            f.write(
                "import sys\n"
                f"content = open({file_path!r}).read()\n"
                "sys.exit(0 if 'line2' in content else 1)\n"
            )
        test_cmd = f"{sys.executable} {check_script}"

        # Success path: valid patch applies, test then passes.
        ok = run_swebench_task(repo, patch, test_cmd)
        assert ok is True, "expected run_swebench_task to report success on a valid patch"

        # Repo must be reset by the `finally` block for the next task.
        with open(file_path) as f:
            assert f.read() == "line1\n", "repo was not reset after run_swebench_task"

        # Failure path: patch that doesn't apply cleanly (context doesn't match).
        bad_patch = patch.replace("line1", "totally-different-context")
        failed = run_swebench_task(repo, bad_patch, test_cmd)
        assert failed is False, "expected run_swebench_task to report failure on a bad patch"

    print("[block #0] run_swebench_task: success path True, bad-patch path False -- OK")


_test_block0()


# ============================================================
# Block #1 (line ~174) -- Trajectory vs. outcome scoring
# ============================================================
import json
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class TrajectoryStep:
    action: str          # e.g. "read_file", "edit_file", "run_tests"
    action_input: dict
    observation: str     # what the environment returned
    reward: float = 0.0  # step-level reward if using process reward model

@dataclass
class Trajectory:
    task_id: str
    steps: List[TrajectoryStep] = field(default_factory=list)
    final_outcome: bool = False   # did the task succeed?

def compute_outcome_score(trajectories: List[Trajectory]) -> float:
    """Binary pass rate — the standard SWE-bench metric."""
    return sum(t.final_outcome for t in trajectories) / len(trajectories)

def compute_trajectory_efficiency(traj: Trajectory) -> dict:
    """
    Compute trajectory-level metrics beyond binary pass/fail.
    Returns a dict with step count, token estimate, and action diversity.
    """
    step_count = len(traj.steps)

    # Rough token estimate: action_input + observation for each step
    total_chars = sum(
        len(json.dumps(s.action_input)) + len(s.observation)
        for s in traj.steps
    )
    token_estimate = total_chars // 4  # ~4 chars/token as approximation

    # How many distinct action types were used?
    action_types = {s.action for s in traj.steps}

    return {
        "task_id": traj.task_id,
        "success": traj.final_outcome,
        "step_count": step_count,
        "token_estimate": token_estimate,
        "distinct_actions": len(action_types),
        "tokens_per_step": token_estimate / max(step_count, 1),
    }

def compare_trajectories(
    model_a: List[Trajectory],
    model_b: List[Trajectory]
) -> dict:
    """
    Compare two sets of agent trajectories on both outcome and efficiency.
    Assumes matching task order between model_a and model_b.
    """
    outcome_a = compute_outcome_score(model_a)
    outcome_b = compute_outcome_score(model_b)

    # Only compare trajectories where outcomes differ (diagnostic cases)
    a_pass_b_fail = [(a, b) for a, b in zip(model_a, model_b)
                     if a.final_outcome and not b.final_outcome]
    b_pass_a_fail = [(a, b) for a, b in zip(model_a, model_b)
                     if not a.final_outcome and b.final_outcome]

    return {
        "outcome_A": outcome_a,
        "outcome_B": outcome_b,
        "delta": outcome_a - outcome_b,
        "A_wins": len(a_pass_b_fail),
        "B_wins": len(b_pass_a_fail),
        "n_tasks": len(model_a),
    }


def _test_block1():
    step_a1 = TrajectoryStep(
        action="read_file", action_input={"path": "a.py"},
        observation="def foo(): pass", reward=0.1,
    )
    step_a2 = TrajectoryStep(
        action="edit_file", action_input={"path": "a.py", "diff": "+def bar(): pass"},
        observation="patch applied", reward=0.4,
    )
    traj_a = Trajectory(task_id="task-1", steps=[step_a1, step_a2], final_outcome=True)

    step_b1 = TrajectoryStep(
        action="read_file", action_input={"path": "a.py"},
        observation="def foo(): pass", reward=0.1,
    )
    traj_b = Trajectory(task_id="task-1", steps=[step_b1], final_outcome=False)

    outcome = compute_outcome_score([traj_a, traj_b])
    assert outcome == 0.5, outcome

    eff = compute_trajectory_efficiency(traj_a)
    assert eff["step_count"] == 2
    assert eff["distinct_actions"] == 2
    assert eff["success"] is True

    cmp = compare_trajectories([traj_a], [traj_b])
    assert cmp["A_wins"] == 1
    assert cmp["B_wins"] == 0
    assert math.isclose(cmp["delta"], 1.0 - 0.0)

    print(f"[block #1] outcome={outcome}, efficiency={eff}, compare={cmp}")


_test_block1()


# ============================================================
# Block #2 (line ~302) -- pass@k estimator
# ============================================================
import math
from typing import Sequence

def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased estimator of pass@k.

    n: total samples drawn per task
    c: number of correct samples among n
    k: number of samples to select (k <= n)

    Returns probability that at least one of k samples is correct.
    Uses the combinatorial formula from the Codex paper (Chen et al. 2021).
    """
    if n < k:
        raise ValueError(f"Cannot compute pass@{k} with only {n} samples")
    if c == 0:
        return 0.0
    if c == n:
        return 1.0
    if n - c < k:
        # Fewer "wrong" samples than k means every k-subset includes a correct one.
        # (Real bug fix mirrored from the book: without this, the log-space
        # computation below hits math.log(0) / math.log(negative) whenever
        # k exceeds the number of incorrect samples, e.g. n=8, c=3, k=8.)
        return 1.0
    # 1 - P(all k selected are wrong) = 1 - C(n-c, k) / C(n, k)
    # Compute in log space for numerical stability with large n
    log_num = sum(math.log(n - c - i) for i in range(k))
    log_den = sum(math.log(n - i) for i in range(k))
    return 1.0 - math.exp(log_num - log_den)

def aggregate_pass_at_k(
    results: Sequence[tuple[int, int]],  # list of (n, c) per task
    k: int
) -> float:
    """
    Aggregate pass@k over multiple tasks (unweighted mean).
    results: list of (n_samples, n_correct) per task.
    """
    scores = [pass_at_k(n, c, k) for n, c in results]
    return sum(scores) / len(scores)

# Example: 500 tasks, each run 8 times
import random
random.seed(42)
task_results = [(8, random.randint(0, 3)) for _ in range(500)]

_block2_scores = {}
for k in [1, 2, 4, 8]:
    score = aggregate_pass_at_k(task_results, k)
    _block2_scores[k] = score
    print(f"pass@{k}: {score:.3f}")
# Example output (will vary by random seed):
# pass@1: 0.188
# pass@2: 0.340
# pass@4: 0.560
# pass@8: 0.737


def _test_block2():
    # Worked example from the chapter text: n=8, c=2 -> pass@1=0.25, pass@2~0.464, pass@4~0.786
    assert math.isclose(pass_at_k(8, 2, 1), 0.25, rel_tol=1e-9)
    assert math.isclose(pass_at_k(8, 2, 2), 1 - 15 / 28, rel_tol=1e-9)
    assert math.isclose(pass_at_k(8, 2, 4), 1 - 15 / 70, rel_tol=1e-9)
    assert pass_at_k(5, 0, 3) == 0.0
    assert pass_at_k(5, 5, 3) == 1.0

    # Regression test for a real book bug: n=8, c=3, k=8 used to raise
    # "math domain error" from math.log(0) because n-c (=5) < k (=8) was
    # not short-circuited before the log-space binomial computation.
    assert pass_at_k(8, 3, 8) == 1.0
    assert pass_at_k(8, 1, 8) == 1.0

    try:
        pass_at_k(3, 1, 5)
        raise AssertionError("expected ValueError for k > n")
    except ValueError:
        pass

    # pass@k must be monotonically non-decreasing in k for the same task set.
    ks = [1, 2, 4, 8]
    vals = [_block2_scores[k] for k in ks]
    assert all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1)), vals

    print(f"[block #2] pass@k worked example verified; aggregate scores={_block2_scores}")


_test_block2()


# ============================================================
# Block #3 (line ~421) -- Confidence intervals & McNemar's test
# ============================================================
import numpy as np
# `from scipy import stats` is guarded at module top (scipy is not in the
# guaranteed-available package list for this test suite).

def agent_benchmark_ci(n_tasks: int, n_correct: int, confidence: float = 0.95) -> dict:
    """
    Wilson score interval for a binary agent benchmark.
    More accurate than normal approximation at extreme proportions.
    """
    p_hat = n_correct / n_tasks
    alpha = 1 - confidence
    z = stats.norm.ppf(1 - alpha / 2)  # e.g. 1.96 for 95% CI

    # Wilson score interval
    denom = 1 + z**2 / n_tasks
    center = (p_hat + z**2 / (2 * n_tasks)) / denom
    margin = (z * np.sqrt(p_hat * (1 - p_hat) / n_tasks + z**2 / (4 * n_tasks**2))) / denom

    return {
        "estimate": p_hat,
        "ci_lower": max(0.0, center - margin),
        "ci_upper": min(1.0, center + margin),
        "n": n_tasks,
        "n_correct": n_correct,
    }

def mcnemar_test(outcomes_a: list[bool], outcomes_b: list[bool]) -> dict:
    """
    McNemar's test for paired binary outcomes.
    Tests whether model A and model B have significantly different success rates
    on the SAME task set (the right comparison, vs. unpaired z-test).

    outcomes_a[i], outcomes_b[i]: True/False for task i under model A/B.
    """
    assert len(outcomes_a) == len(outcomes_b)
    # Count concordant and discordant pairs
    b = sum(1 for a, bb in zip(outcomes_a, outcomes_b) if a and not bb)   # A passes, B fails
    c = sum(1 for a, bb in zip(outcomes_a, outcomes_b) if not a and bb)   # B passes, A fails

    # McNemar's statistic (with continuity correction)
    chi2 = (abs(b - c) - 1)**2 / (b + c) if (b + c) > 0 else 0.0
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return {
        "A_only": b,    # tasks only A solves
        "B_only": c,    # tasks only B solves
        "chi2": chi2,
        "p_value": p_value,
        "significant_at_05": p_value < 0.05,
    }

if stats is not None:
    # Example: 500 tasks, model A gets 160 right, model B gets 175 right
    # but 30 are tasks A solves that B doesn't, and 45 are tasks B solves that A doesn't
    outcomes_a = [True] * 130 + [True] * 30 + [False] * 45 + [False] * 295
    outcomes_b = [True] * 130 + [False] * 30 + [True] * 45 + [False] * 295
    result = mcnemar_test(outcomes_a, outcomes_b)
    print(result)
    # {"A_only": 30, "B_only": 45, "chi2": ..., "p_value": ..., "significant_at_05": ...}
else:
    print("[block #3] SKIP(scipy unavailable): mcnemar_test/agent_benchmark_ci example not run")


def _test_block3():
    if stats is None:
        print("[block #3] SKIP(scipy unavailable): defined but not exercised")
        return

    ci = agent_benchmark_ci(500, 150)
    assert math.isclose(ci["estimate"], 0.3, rel_tol=1e-9)
    assert ci["ci_lower"] < ci["estimate"] < ci["ci_upper"]
    # sanity-check against the chapter's worked SE ~= 0.020 -> CI roughly [0.26, 0.34]
    assert 0.24 < ci["ci_lower"] < 0.28
    assert 0.32 < ci["ci_upper"] < 0.36

    assert result["A_only"] == 30
    assert result["B_only"] == 45
    assert result["chi2"] > 0
    assert 0.0 <= result["p_value"] <= 1.0

    print(f"[block #3] agent_benchmark_ci(500,150)={ci}, mcnemar={result}")


_test_block3()


# ============================================================
# Block #4 (line ~545) -- Minimal reproducible eval harness
# ============================================================
import json, hashlib, datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Callable, Any

@dataclass
class EvalConfig:
    benchmark: str         # e.g. "swe-bench-verified"
    model: str             # e.g. "gpt-4o-2024-08-06"
    scaffold: str          # e.g. "acr-v2.1"
    n_seeds: int = 3
    max_iterations: int = 5
    file_localization: str = "bm25-top10"  # or "oracle", "agent-only"
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.datetime.utcnow().isoformat()

    def fingerprint(self) -> str:
        """Stable hash of this config for deduplication."""
        s = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(s.encode()).hexdigest()[:12]


def run_eval(
    config: EvalConfig,
    tasks: list[dict],
    agent_fn: Callable[[dict, int], bool],  # (task, seed) -> success
    output_dir: Path,
) -> dict:
    """
    Run an agent eval with multiple seeds, log results, and compute statistics.

    agent_fn: your agent callable. Takes a task dict and a random seed int,
              returns True if the task was resolved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fp = config.fingerprint()
    results_path = output_dir / f"eval_{fp}.jsonl"

    all_per_task = []  # list of (n_correct, n_total) per task

    with open(results_path, "w") as out_f:
        for task in tasks:
            task_results = []
            for seed in range(config.n_seeds):
                success = agent_fn(task, seed)
                record = {
                    "task_id": task["id"],
                    "seed": seed,
                    "success": success,
                    "config_fingerprint": fp,
                }
                out_f.write(json.dumps(record) + "\n")
                task_results.append(success)

            n_correct = sum(task_results)
            all_per_task.append((config.n_seeds, n_correct))

    # Aggregate
    pass1 = aggregate_pass_at_k(all_per_task, k=1)
    passN = aggregate_pass_at_k(all_per_task, k=config.n_seeds)
    ci = agent_benchmark_ci(len(tasks), int(pass1 * len(tasks)))

    summary = {
        "config": asdict(config),
        "n_tasks": len(tasks),
        "pass_at_1": pass1,
        f"pass_at_{config.n_seeds}": passN,
        "ci_95_lower": ci["ci_lower"],
        "ci_95_upper": ci["ci_upper"],
        "results_file": str(results_path),
    }

    summary_path = output_dir / f"summary_{fp}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def _test_block4():
    if stats is None:
        print("[block #4] SKIP(scipy unavailable via block #3's agent_benchmark_ci): defined but not run")
        return

    config = EvalConfig(
        benchmark="toy-bench",
        model="toy-model-1",
        scaffold="toy-scaffold-v1",
        n_seeds=4,
        max_iterations=2,
        file_localization="oracle",
    )
    tasks = [{"id": f"t{i}"} for i in range(6)]

    def dummy_agent_fn(task: dict, seed: int) -> bool:
        # Deterministic toy policy: succeeds on even (task_index + seed) sums.
        idx = int(task["id"][1:])
        return (idx + seed) % 3 != 0

    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "eval_out"
        summary = run_eval(config, tasks, dummy_agent_fn, output_dir)

        assert 0.0 <= summary["pass_at_1"] <= 1.0
        assert summary["n_tasks"] == len(tasks)
        assert Path(summary["results_file"]).exists()

        results_path = Path(summary["results_file"])
        lines = results_path.read_text().strip().splitlines()
        assert len(lines) == len(tasks) * config.n_seeds
        for line in lines:
            record = json.loads(line)
            assert set(record.keys()) == {"task_id", "seed", "success", "config_fingerprint"}

        # fingerprint is stable/deterministic given the same config (minus timestamp changes)
        fp1 = config.fingerprint()
        fp2 = config.fingerprint()
        assert fp1 == fp2

        print(f"[block #4] run_eval summary: {summary}")


_test_block4()

print("\nAll runnable blocks in 08-agent-evaluation.md executed successfully.")
