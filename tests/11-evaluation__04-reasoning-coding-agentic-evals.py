"""
Executable extraction of the CPU-runnable Python code blocks from
content/11-evaluation/04-reasoning-coding-agentic-evals.md

Blocks tested (chapter's own numbering), assembled in order so later
blocks can rely on names defined by earlier blocks:
  #0 (line ~47,  56 lines)  - pass_at_k / aggregate_pass_at_k (unbiased
      Pass@k estimator). BUG FOUND & FIXED (mirrored in content/): the
      original code crashed with "math domain error" whenever k > n - c
      (e.g. n=20, c=19, k=20) because C(n-c, k) is 0 in that regime and
      the log-space formula computed log(0)/log(negative). Added the
      missing `if n - c < k: return 1.0` guard. This also corrects the
      chapter's stated Pass@10 output (0.770 -> 0.689) and its incorrect
      claim that "Pass@20 is lower than Pass@10" -- Pass@k is provably
      non-decreasing in k for fixed n, so that could never be true.
  #2 (line ~135, 106 lines) - run_in_sandbox / ExecutionResult: a real
      subprocess.run(["python3", ...]) sandbox. No network involved --
      it just spawns a local python3 interpreter, which is allowed.
      BUG FOUND & FIXED (mirrored in content/): the original code wrapped
      its f-string template in `textwrap.dedent(...)`, but dedent runs
      *after* `{code}`/`{test_body}` interpolation. Since `code` is itself
      unindented (starts "\ndef two_sum(...):"), the combined string's
      common leading whitespace becomes empty, so dedent is a no-op and
      "import resource, sys" stays at 8-space indent -- an
      IndentationError before the generated script can even run. Fixed by
      building the template at column 0 and dropping dedent entirely.
  #3 (line ~263, 48 lines)  - math_answers_equivalent (sympy symbolic
      equivalence). sympy is not in the CI-guaranteed import set, so the
      import is guarded; the block is defined-and-called only if sympy
      imports successfully. Additionally, its symbolic-equality path
      (`sympy.parsing.latex.parse_latex`) needs the separate
      `antlr4-python3-runtime` package at call time, which is not
      installed and not CI-guaranteed either -- so even with sympy
      present, we only execute+assert on this block if antlr4 is
      importable; otherwise it's defined-not-called with an explicit
      SKIP (see below). This is an environment/dependency gap, not a
      bug in the book's logic.
  #5 (line ~381, 123 lines) - minimal agentic eval harness: AgentTask,
      TrajectoryStep, Trajectory, AgentEnvironment (ABC), AgentModel (ABC),
      collect_trajectory, evaluate_agent (asyncio). We supply tiny concrete
      subclasses (a scripted echo environment + a scripted agent) so the
      abstract classes and the full async loop actually execute.
  #6 (line ~542, 55 lines)  - majority_vote_accuracy (self-consistency /
      majority voting over k samples)
  #7 (line ~629, 56 lines)  - detects_hardcoded_solution (unit-test-hacking
      heuristic detector using ast + re)

Blocks explicitly SKIPPED (per the harness's heuristic classification):
  #1 = non-python (sandbox architecture figure placeholder / prose).
  #4 = non-python (plain-text SWE-bench instance schema block).

No network or external API calls occur anywhere in this file. The only
"process" spawned is a local `python3` subprocess (block #2), which is the
book's own minimal-sandbox demonstration and requires no network access.
"""

import ast
import asyncio
import math
import os
import re
import statistics
import subprocess
import tempfile
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

import numpy as np

# --- Guarded optional third-party import (not in the CI-guaranteed set) ---
try:
    from sympy import sympify, simplify, Eq, solve, latex
    from sympy.parsing.latex import parse_latex
    _HAVE_SYMPY = True
except Exception:
    sympify = simplify = Eq = solve = latex = None
    parse_latex = None
    _HAVE_SYMPY = False

# sympy.parsing.latex.parse_latex additionally needs the separate
# `antlr4-python3-runtime` package at call time (it's a lazy internal
# import, so the `from sympy.parsing.latex import parse_latex` line above
# can succeed even when antlr4 itself is missing). Neither sympy nor
# antlr4 are in the CI-guaranteed import set, so probe for antlr4
# explicitly -- without it, every parse_latex() call raises and the
# block's actual symbolic-equivalence logic never runs.
_HAVE_ANTLR4 = False
if _HAVE_SYMPY:
    try:
        import antlr4  # noqa: F401
        _HAVE_ANTLR4 = True
    except Exception:
        _HAVE_ANTLR4 = False


# ============================================================================
# Block #0 (line ~47): Pass@k unbiased estimator
# ============================================================================

def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased Pass@k estimator from Chen et al. (HumanEval, 2021).

    Args:
        n: total number of generated samples for this problem
        c: number of samples that pass all tests
        k: the k in Pass@k (e.g. 1, 10, 100)

    Returns:
        estimated probability that at least one of k random draws passes.

    Note: uses log-space arithmetic to avoid combinatorial overflow.
    """
    if n < k:
        raise ValueError(f"n={n} must be >= k={k}")
    if c > n:
        raise ValueError(f"c={c} cannot exceed n={n}")
    if c == n:
        # All samples pass — all possible draws of k pass.
        return 1.0
    if n - c < k:
        # BUG FIX (mirrors content/11-evaluation/04-reasoning-coding-agentic-evals.md):
        # Fewer failing samples than k: any draw of k must include a
        # passing sample. C(n-c, k) is 0 by definition (can't choose k
        # items from a pool of n-c), so the failure probability is 0.
        # Without this guard, math.log(n - c - i) hits log(0) or a
        # negative argument and raises "math domain error".
        return 1.0
    # Numerically stable: compute log(C(n-c, k) / C(n, k))
    # = sum(log((n-c-i)/(n-i)) for i in range(k))
    log_prob_fail = sum(
        math.log(n - c - i) - math.log(n - i)
        for i in range(k)
    )
    return 1.0 - math.exp(log_prob_fail)


def aggregate_pass_at_k(results: List[dict], k: int) -> float:
    """
    Average Pass@k across problems.

    Each entry in results: {'n': int, 'c': int}
    """
    scores = [pass_at_k(r['n'], r['c'], k) for r in results]
    return sum(scores) / len(scores)


def _test_block0():
    # Example: HumanEval-style evaluation
    problems = [
        {'n': 20, 'c': 6},    # moderate difficulty
        {'n': 20, 'c': 19},   # easy
        {'n': 20, 'c': 0},    # failed completely
        {'n': 20, 'c': 2},    # hard
    ]

    scores = {}
    for k in [1, 10, 20]:
        score = aggregate_pass_at_k(problems, k)
        scores[k] = score
        print(f"Pass@{k:>2d}: {score:.3f}")

    # Cross-check against the book's (corrected) stated output text block.
    assert abs(scores[1] - 0.338) < 1e-3
    assert abs(scores[10] - 0.689) < 1e-3
    assert abs(scores[20] - 0.750) < 1e-3
    # Pass@k must be non-decreasing in k for fixed n (drawing more samples
    # cannot lower the chance of at least one pass).
    assert scores[1] <= scores[10] <= scores[20]

    # Worked example from the chapter text: n=20, c=6.
    naive_k10 = 1 - (1 - 6 / 20) ** 10
    assert abs(naive_k10 - 0.972) < 1e-3
    unbiased_k10 = pass_at_k(20, 6, 10)
    assert abs(unbiased_k10 - 0.9946) < 1e-3
    # At k=1 the unbiased estimator matches the empirical pass rate exactly.
    assert abs(pass_at_k(20, 6, 1) - 0.30) < 1e-9

    # Error paths documented in the docstring/behavior.
    try:
        pass_at_k(5, 1, 10)
        raise AssertionError("expected ValueError for n < k")
    except ValueError:
        pass
    try:
        pass_at_k(5, 6, 1)
        raise AssertionError("expected ValueError for c > n")
    except ValueError:
        pass


_test_block0()
print("[ok] block #0 pass_at_k / aggregate_pass_at_k")


# ============================================================================
# Block #2 (line ~135): A minimal sandbox in Python
# ============================================================================

@dataclass
class ExecutionResult:
    passed: bool
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    oom_killed: bool


def run_in_sandbox(
    code: str,
    test_cases: list,
    timeout_seconds: float = 5.0,
    memory_mb: int = 256,
) -> ExecutionResult:
    """
    Executes model-generated Python code against unit tests in a subprocess
    with resource limits.

    In production, replace the subprocess call with a Docker/gVisor invocation.
    Here we use resource limits via `resource` module for demonstration.
    """
    # Build a self-contained test script: solution + assertions
    test_body = "\n".join(
        f"assert {tc['input']} == {tc['expected']!r}, "
        f"'Failed: {tc['input']}'"
        for tc in test_cases
    )
    # BUG FIX (mirrors content/11-evaluation/04-reasoning-coding-agentic-evals.md):
    # the original code wrapped this f-string in textwrap.dedent(...), but
    # dedent runs on the string *after* {code}/{test_body} interpolation.
    # Since `code` is itself unindented (starts "\ndef two_sum(...):"),
    # the combined string's common leading whitespace becomes empty, so
    # dedent is a no-op and "import resource, sys" stays at 8-space
    # indent -- an IndentationError before the script can even run.
    # Fix: build the template at column 0 and skip dedent entirely.
    full_script = f"""
import resource, sys
# Hard memory limit (Linux only)
resource.setrlimit(
    resource.RLIMIT_AS,
    ({memory_mb * 1024 * 1024}, {memory_mb * 1024 * 1024})
)
# --- Model-generated solution below ---
{code}
# --- Test assertions ---
{test_body}
print("PASS")
"""

    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.py', delete=False
    ) as f:
        f.write(full_script)
        script_path = f.name

    try:
        result = subprocess.run(
            ["python3", script_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            # Additional isolation: no inherited env vars
            env={"PATH": "/usr/bin:/bin"},
        )
        passed = (result.returncode == 0 and "PASS" in result.stdout)
        return ExecutionResult(
            passed=passed,
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            timed_out=False,
            oom_killed="MemoryError" in result.stderr or result.returncode == -9,
        )
    except subprocess.TimeoutExpired:
        return ExecutionResult(
            passed=False, exit_code=-1,
            stdout="", stderr="Timeout",
            timed_out=True, oom_killed=False,
        )
    finally:
        os.unlink(script_path)


def _test_block2():
    # Example usage (verbatim from the book)
    solution_code = """
def two_sum(nums, target):
    seen = {}
    for i, n in enumerate(nums):
        diff = target - n
        if diff in seen:
            return [seen[diff], i]
        seen[n] = i
    return []
"""

    test_cases = [
        {"input": "two_sum([2,7,11,15], 9)", "expected": [0, 1]},
        {"input": "two_sum([3,2,4], 6)",     "expected": [1, 2]},
        {"input": "two_sum([3,3], 6)",        "expected": [0, 1]},
    ]

    result = run_in_sandbox(solution_code, test_cases)
    print(f"Passed: {result.passed}, exit={result.exit_code}")
    assert result.passed is True
    assert result.exit_code == 0
    assert "PASS" in result.stdout

    # Also exercise the failure path: a solution that is simply wrong.
    bad_solution = """
def two_sum(nums, target):
    return []
"""
    bad_result = run_in_sandbox(bad_solution, test_cases)
    assert bad_result.passed is False
    assert bad_result.exit_code != 0
    assert "AssertionError" in bad_result.stderr


_test_block2()
print("[ok] block #2 run_in_sandbox / ExecutionResult")


# ============================================================================
# Block #3 (line ~263): math_answers_equivalent (sympy symbolic equivalence)
# ============================================================================

if _HAVE_SYMPY:
    def math_answers_equivalent(model_answer: str, gold_answer: str) -> bool:
        """
        Check if two math answer strings are symbolically equivalent.
        Handles: numbers, fractions, expressions, simple equations.

        Caveats: does not handle set answers, multiple solutions, or
        answers requiring context (e.g., 'x > 0').
        """
        # Try numeric equality first (fast path)
        try:
            a = float(sympify(model_answer))
            b = float(sympify(gold_answer))
            return abs(a - b) < 1e-6
        except Exception:
            pass

        # Symbolic equality (slower path)
        try:
            expr_a = parse_latex(model_answer)
            expr_b = parse_latex(gold_answer)
            # simplify(A - B) == 0 iff A == B
            diff = simplify(expr_a - expr_b)
            return diff == 0
        except Exception:
            pass

        # Fallback: normalized string match
        def normalize(s):
            return s.strip().replace(" ", "").replace("\\,", "").lower()

        return normalize(model_answer) == normalize(gold_answer)

    def _test_block3():
        # Test (verbatim pairs from the book)
        pairs = [
            ("\\frac{1}{2}", "0.5"),             # should be True
            ("x^2 + 2x + 1", "(x+1)^2"),         # should be True
            ("3.14159", "\\pi"),                   # should be False
            ("\\sqrt{4}", "2"),                    # should be True
        ]

        expected = [True, True, False, True]
        actual = []
        for a, b in pairs:
            r = math_answers_equivalent(a, b)
            print(f"  '{a}' == '{b}': {r}")
            actual.append(r)

        assert actual == expected, f"expected {expected}, got {actual}"

    if _HAVE_ANTLR4:
        _test_block3()
        print("[ok] block #3 math_answers_equivalent")
    else:
        # math_answers_equivalent is defined and reachable above (so the
        # block's logic exists in the module), but its symbolic-equality
        # path depends on sympy.parsing.latex.parse_latex, which itself
        # requires the separate `antlr4-python3-runtime` package at call
        # time. That package is not installed here and is not in the
        # CI-guaranteed import set (numpy/torch/einops/sklearn/stdlib),
        # so we cannot honestly assert on parse_latex's output -- doing so
        # would just be testing "does antlr4 exist", not the book's logic.
        print(
            "[skip] block #3 math_answers_equivalent -- "
            "antlr4-python3-runtime unavailable (required by "
            "sympy.parsing.latex.parse_latex, not a CI-guaranteed import)"
        )
else:
    print("[skip] block #3 math_answers_equivalent -- sympy unavailable")


# ============================================================================
# Block #5 (line ~381): Minimal agentic eval harness
# ============================================================================

@dataclass
class AgentTask:
    task_id: str
    description: str           # Natural-language task specification
    initial_state: dict        # Starting environment state
    success_fn: Callable       # fn(final_state) -> bool
    max_steps: int = 30


@dataclass
class TrajectoryStep:
    step_idx: int
    action: str               # e.g., "bash('ls -la')"
    observation: str          # environment response
    success_so_far: bool      # did this step crash?


@dataclass
class Trajectory:
    task_id: str
    steps: list = field(default_factory=list)
    final_state: dict = field(default_factory=dict)
    completed: bool = False
    error: str = ""


class AgentEnvironment(ABC):
    @abstractmethod
    async def reset(self, initial_state: dict) -> str:
        """Reset to initial state, return initial observation."""

    @abstractmethod
    async def step(self, action: str):
        """Execute action; return (observation, done, final_state)."""


class AgentModel(ABC):
    @abstractmethod
    async def act(self, task: str, history: list) -> str:
        """Given task description and history, return next action string."""


async def collect_trajectory(
    task: AgentTask,
    agent: AgentModel,
    env: AgentEnvironment,
) -> Trajectory:
    """Run one agent episode and collect the full trajectory."""
    traj = Trajectory(task_id=task.task_id)
    obs = await env.reset(task.initial_state)
    history = []

    for step_idx in range(task.max_steps):
        try:
            # Model chooses next action
            action = await agent.act(task.description, history)
            # Environment executes it
            obs, done, final_state = await env.step(action)

            step = TrajectoryStep(
                step_idx=step_idx,
                action=action,
                observation=obs,
                success_so_far=True,
            )
            history.append(step)
            traj.steps.append(step)

            if done:
                traj.final_state = final_state
                break
        except Exception as e:
            traj.error = str(e)
            break

    # Score the trajectory
    traj.completed = task.success_fn(traj.final_state)
    return traj


async def evaluate_agent(
    tasks: list,
    agent: AgentModel,
    env_factory: Callable[[], AgentEnvironment],
    rollouts_per_task: int = 3,
) -> dict:
    """
    Evaluate an agent across all tasks with multiple rollouts.
    Returns summary statistics.
    """
    results = {t.task_id: [] for t in tasks}

    for task in tasks:
        for rollout in range(rollouts_per_task):
            env = env_factory()
            traj = await collect_trajectory(task, agent, env)
            results[task.task_id].append(traj.completed)

    # Task is "solved" if at least one rollout succeeded (Pass@rollouts_per_task)
    solved = sum(any(r) for r in results.values())
    total = len(tasks)
    pass_at_1 = sum(r[0] for r in results.values()) / total
    pass_at_k = solved / total

    return {
        "pass_at_1": pass_at_1,
        f"pass_at_{rollouts_per_task}": pass_at_k,
        "per_task_results": results,
        "mean_steps": None,  # compute from trajectories
    }


# --- Minimal glue: tiny concrete env + agent to actually exercise the ABCs ---

class _CounterEnvironment(AgentEnvironment):
    """A toy environment: 'increment' bumps a counter; task succeeds
    once the counter reaches a target. Deterministic, seeded by
    initial_state so different rollouts can behave differently."""

    async def reset(self, initial_state: dict) -> str:
        self._counter = initial_state.get("start", 0)
        self._target = initial_state.get("target", 3)
        self._max_ok_step = initial_state.get("max_ok_step", 100)
        self._step_num = 0
        return f"counter={self._counter}"

    async def step(self, action: str):
        self._step_num += 1
        if action == "increment":
            self._counter += 1
        done = self._counter >= self._target
        final_state = {"counter": self._counter}
        return f"counter={self._counter}", done, final_state


class _ScriptedAgent(AgentModel):
    """Always proposes 'increment' -- deterministic, no external calls."""

    async def act(self, task: str, history: list) -> str:
        return "increment"


def _success_at_target(final_state: dict) -> bool:
    return final_state.get("counter", 0) >= 3


def _test_block5():
    tasks = [
        AgentTask(
            task_id="reach-3",
            description="Increment a counter until it reaches 3.",
            initial_state={"start": 0, "target": 3},
            success_fn=_success_at_target,
            max_steps=10,
        ),
        AgentTask(
            task_id="reach-3-again",
            description="Increment a counter until it reaches 3.",
            initial_state={"start": 0, "target": 3},
            success_fn=_success_at_target,
            max_steps=1,  # too few steps -> should fail
        ),
    ]

    agent = _ScriptedAgent()
    summary = asyncio.run(
        evaluate_agent(tasks, agent, _CounterEnvironment, rollouts_per_task=2)
    )
    print(summary)

    # Task 1 has enough max_steps -> every rollout should complete.
    assert all(summary["per_task_results"]["reach-3"])
    # Task 2 has only 1 max_step -> counter never reaches 3 -> never completes.
    assert not any(summary["per_task_results"]["reach-3-again"])
    assert summary["pass_at_1"] == 0.5
    assert summary["pass_at_2"] == 0.5


_test_block5()
print("[ok] block #5 agentic eval harness (AgentTask/Trajectory/collect_trajectory/evaluate_agent)")


# ============================================================================
# Block #6 (line ~542): majority_vote_accuracy (self-consistency)
# ============================================================================

def majority_vote_accuracy(
    generate_fn: Callable[[str, int], list],
    verify_fn: Callable[[str, str], bool],
    problems: list,
    k_values: list = [1, 4, 16, 64],
) -> dict:
    """
    Measure accuracy under majority voting for different sample budgets k.

    Args:
        generate_fn(problem, n) -> list of n answer strings
        verify_fn(model_answer, gold_answer) -> bool
        problems: list of {'input': str, 'gold': str}
        k_values: list of k's to evaluate
    """
    max_k = max(k_values)
    per_problem_samples = []

    # Generate max_k samples per problem once; reuse for smaller k
    for prob in problems:
        samples = generate_fn(prob['input'], max_k)
        per_problem_samples.append({
            'gold': prob['gold'],
            'samples': samples,
        })

    results = {}
    for k in k_values:
        correct = 0
        for entry in per_problem_samples:
            # Take first k samples
            answers = entry['samples'][:k]
            gold = entry['gold']

            # Majority vote: pick the most frequent answer
            from collections import Counter
            vote_counts = Counter(answers)
            majority_answer = vote_counts.most_common(1)[0][0]

            if verify_fn(majority_answer, gold):
                correct += 1

        results[k] = correct / len(problems)

    return results


def _test_block6():
    # Deterministic toy "sampler": a noisy model that answers correctly with
    # probability proportional to how many of the first n draws we mark
    # correct, seeded by a fixed RNG so results are reproducible on CPU.
    rng = np.random.RandomState(0)

    def generate_fn(problem: str, n: int) -> list:
        gold = problem  # the "problem" string doubles as the gold answer
        wrong = "WRONG"
        # 70% chance of the correct answer each draw -> majority vote should
        # converge to the correct answer as k grows.
        return [gold if rng.rand() < 0.7 else wrong for _ in range(n)]

    def verify_fn(model_answer: str, gold_answer: str) -> bool:
        return model_answer == gold_answer

    problems = [{'input': f'problem-{i}', 'gold': f'problem-{i}'} for i in range(20)]

    results = majority_vote_accuracy(
        generate_fn, verify_fn, problems, k_values=[1, 4, 16, 64]
    )
    print(results)

    assert set(results.keys()) == {1, 4, 16, 64}
    for v in results.values():
        assert 0.0 <= v <= 1.0
    # With p=0.7 correct per draw, majority-vote accuracy should trend
    # upward (or at least not collapse) as k grows from 1 to 64.
    assert results[64] >= results[1] - 0.15


_test_block6()
print("[ok] block #6 majority_vote_accuracy")


# ============================================================================
# Block #7 (line ~629): detects_hardcoded_solution (unit-test hacking check)
# ============================================================================

def detects_hardcoded_solution(code: str, test_inputs: list) -> bool:
    """
    Heuristic check for hardcoded solutions.
    Flags:
      1. If return statements contain only literals (no computation).
      2. If test input values appear as literals in if-conditions.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False  # Can't parse, not our problem here

    # Check 1: only-literal returns (no variable references)
    for node in ast.walk(tree):
        if isinstance(node, ast.Return):
            if node.value and isinstance(node.value, ast.Constant):
                # A single constant return in a function is suspicious
                return True

    # Check 2: test input values hardcoded in conditions
    for test_input in test_inputs:
        # Extract numeric literals from the test input
        nums = re.findall(r'\b\d+\b', test_input)
        for num in nums:
            # If this exact number appears in an if-condition return structure
            if_hardcode_pattern = rf'if.*\b{num}\b.*:\s*return'
            if re.search(if_hardcode_pattern, code):
                return True

    return False


def _test_block7():
    # Example of a hardcoded (hacked) solution (verbatim from the book)
    hacked_solution = """
def two_sum(nums, target):
    if nums == [2, 7, 11, 15] and target == 9:
        return [0, 1]
    if nums == [3, 2, 4] and target == 6:
        return [1, 2]
    return []
"""

    test_inputs = ["two_sum([2, 7, 11, 15], 9)", "two_sum([3, 2, 4], 6)"]
    flagged = detects_hardcoded_solution(hacked_solution, test_inputs)
    print(flagged)
    # True — suspicious pattern detected
    assert flagged is True

    # A genuine (non-hardcoded) solution should not be flagged.
    genuine_solution = """
def two_sum(nums, target):
    seen = {}
    for i, n in enumerate(nums):
        diff = target - n
        if diff in seen:
            return [seen[diff], i]
        seen[n] = i
    return []
"""
    not_flagged = detects_hardcoded_solution(genuine_solution, test_inputs)
    assert not_flagged is False


_test_block7()
print("[ok] block #7 detects_hardcoded_solution")


print("\nAll 11-evaluation/04-reasoning-coding-agentic-evals.md blocks executed successfully.")
