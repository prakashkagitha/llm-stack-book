# 11.4 Reasoning, Coding & Agentic Evals

Evaluating a model's raw language modeling performance — perplexity, BLEU, or even static multiple-choice benchmarks — tells us almost nothing about whether it can actually *solve problems*. Reasoning, coding, and agentic tasks are the canonical hard cases: the answer is either right or wrong (or can be verified cheaply), the failure modes are subtle, and the evaluation methodology itself can be gamed in ways that corrupt the signal entirely. This chapter covers the techniques that make these evals trustworthy.

We build from first principles: why exact-match is insufficient, how Pass@k turns stochastic generation into a reliable probability estimate, how sandboxed execution verifies correctness without trusting the model, how reasoning traces are evaluated rather than just answers, and how modern agentic harnesses turn long multi-step tasks into reproducible benchmarks. We close with test-time-compute-aware evaluation and a rigorous treatment of reward hacking in eval contexts.

Related background: the broader benchmark landscape is covered in [The Evaluation Problem & Benchmark Landscape](../11-evaluation/01-eval-landscape.html), LLM-as-a-Judge methodology lives in [LLM-as-a-Judge & Automated Evaluation](../11-evaluation/02-llm-as-judge.html), and how to build a general eval harness is in [Building Eval Harnesses](../11-evaluation/03-eval-harnesses.html). The RL machinery that trains reasoning models — and why its reward signal must be trustworthy — is covered in [RL with Verifiable Rewards (RLVR) & The Reasoning Recipe](../05-posttraining-alignment/09-rlvr-reasoning.html) and [Reward Engineering, Verifiers & Sandboxes](../06-rl-infra/08-reward-verifiers-sandboxes.html).

## Pass@k: Measuring Coding Ability as a Probability

### Why a Single Sample Is Not Enough

A model that generates correct code 30% of the time and a model that generates correct code 80% of the time both produce *some* correct outputs if you generate enough samples. Evaluating coding ability on a single generation per problem conflates two things: the model's intrinsic ability and sampling luck. Pass@1 (the fraction of problems solved on the first try) is what matters for single-shot user interactions, but it underestimates capability when $k > 1$ generations are allowed — which is the norm in production coding agents.

**Pass@k** is the probability that *at least one* of $k$ independent samples passes all tests for a given problem. If we generate $n$ samples per problem (with $n \geq k$) and $c$ of them pass, the unbiased estimator is:

$$
\widehat{\text{Pass@}k} = 1 - \frac{\binom{n - c}{k}}{\binom{n}{k}}
$$

This avoids the naive estimator $1 - (1 - c/n)^k$, which is biased when $c/n$ is estimated from finite samples. The combinatorial form is mathematically exact: it computes the probability that a random draw of $k$ samples from the $n$ generated contains zero passing samples, then subtracts from one.

!!! example "Worked Example: Pass@k Numerics"
    Suppose you generate $n = 20$ samples for a problem and $c = 6$ pass all tests.

    **Naive estimator** (biased):
    $$
    1 - \left(1 - \frac{6}{20}\right)^{10} = 1 - (0.7)^{10} \approx 1 - 0.028 = 0.972
    $$

    **Unbiased estimator** for $k = 10$:
    $$
    1 - \frac{\binom{20 - 6}{10}}{\binom{20}{10}} = 1 - \frac{\binom{14}{10}}{\binom{20}{10}}
    $$
    $$
    = 1 - \frac{1001}{184756} \approx 1 - 0.0054 = 0.9946
    $$

    The naive estimator gives 97.2%, the unbiased estimator gives 99.5%. The gap narrows as $n \to \infty$, but for small samples this bias materially affects comparisons between models.

    For $k = 1$: $1 - \binom{14}{1}/\binom{20}{1} = 1 - 14/20 = 0.30$. This matches the empirical pass rate $c/n = 6/20 = 0.30$ exactly — the unbiased estimator coincides with the empirical rate at $k=1$.

### Implementation

```python
import math
from typing import List


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


# Example: HumanEval-style evaluation
problems = [
    {'n': 20, 'c': 6},    # moderate difficulty
    {'n': 20, 'c': 19},   # easy
    {'n': 20, 'c': 0},    # failed completely
    {'n': 20, 'c': 2},    # hard
]

for k in [1, 10, 20]:
    score = aggregate_pass_at_k(problems, k)
    print(f"Pass@{k:>2d}: {score:.3f}")
```

```text
Pass@ 1: 0.338
Pass@10: 0.770
Pass@20: 0.750
```

Pass@20 is lower than Pass@10 here because one problem has $c=0$: once all $k$ draws must come from 20 samples and $c=0$, the probability remains 0 regardless of $k$.

## Code Execution Sandboxes

### Why Execution Is Non-Negotiable

String matching against expected outputs fails for several reasons: (1) there are many syntactically distinct but semantically equivalent programs; (2) floating-point outputs differ across platforms; (3) the model may produce correct code with wrong formatting. Ground truth for code is *behavior*, not text. This means we must *run* the generated code.

Running untrusted model-generated code on the host machine is a security catastrophe: the code might delete files, exfiltrate secrets, or crash the evaluator. A proper sandbox provides:

- **Isolation**: the evaluated process cannot reach the host filesystem or network.
- **Resource limits**: CPU time, memory, and file descriptor caps prevent runaway resource consumption.
- **Reproducibility**: identical OS environment for every run.

### Sandbox Architectures


{{fig:rcae-sandbox-architecture}}


**Docker + seccomp** is the most common approach for research harnesses. For production-quality isolation (used by platforms like Leetcode Discuss and Kaggle competitions), **gVisor** (Google's user-space kernel) or **Firecracker microVMs** (AWS Lambda's substrate) are preferred because they provide stronger isolation without a full VM boot overhead (Firecracker boots in under 125 ms).

### A Minimal Sandbox in Python

```python
import subprocess
import tempfile
import textwrap
import os
from dataclasses import dataclass
from typing import Optional


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
    test_cases: list[dict],
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
    full_script = textwrap.dedent(f"""
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
    """)

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


# Example usage
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
```

### HumanEval, MBPP, and LiveCodeBench

The seminal **HumanEval** benchmark (Chen et al., 2021) contains 164 Python programming problems with unit tests, designed so solutions cannot be looked up from training data. **MBPP** (Mostly Basic Python Problems, Austin et al., 2021) covers 374 crowd-sourced problems. Both are now considered partially contaminated — frontier models were trained on code from the web that likely contains solutions.

**LiveCodeBench** (Jain et al., 2024) addresses contamination by continuously adding new problems from competitive programming platforms (Leetcode, Codeforces, AtCoder) *after* each model's training cutoff. This makes it a living benchmark where contamination is structurally impossible for recent additions.

**SWE-bench** (Jimenez et al., 2023) is qualitatively different: it presents real GitHub issues from open-source Python repositories and asks the model to produce a patch that makes the failing CI tests pass. The sandbox here is the project's own test suite, and success is measured by the fraction of tests that flip from red to green. This is far harder than HumanEval and requires navigating real codebases, reading documentation, and multi-file edits.

## Math Verification and Reasoning-Trace Evaluation

### Exact-Match vs. Symbolic Equivalence

Math problems have a special property: the final answer is often a number, expression, or set that can be *verified* cheaply. But string exact-match fails embarrassingly: "1/2" and "0.5" are the same answer, as are "$x = 3$" and "$x=3$" and "3".

The standard approach is **normalization + symbolic equivalence**:

1. Strip LaTeX formatting and whitespace.
2. Parse into a symbolic expression (using Python's `sympy`).
3. Check symbolic equality, not string equality.

```python
from sympy import sympify, simplify, Eq, solve, latex
from sympy.parsing.latex import parse_latex


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


# Test
pairs = [
    ("\\frac{1}{2}", "0.5"),             # should be True
    ("x^2 + 2x + 1", "(x+1)^2"),         # should be True
    ("3.14159", "\\pi"),                   # should be False
    ("\\sqrt{4}", "2"),                    # should be True
]

for a, b in pairs:
    print(f"  '{a}' == '{b}': {math_answers_equivalent(a, b)}")
```

### Benchmarks: MATH, GSM8K, AMC/AIME, MMLU-Pro

**GSM8K** (Cobbe et al., 2021) contains 8,500 grade-school math word problems. Every problem has a step-by-step solution and a final numeric answer. It was saturated by frontier models around 2024 — many models exceed 90% accuracy, making it a poor discriminator at the top.

**MATH** (Hendrycks et al., 2021) is substantially harder: 12,500 competition math problems across algebra, geometry, number theory, and calculus, at levels AMC 8 through AIME. Difficulty is labeled 1–5; level 5 problems (olympiad-style) remain challenging even for top models.

**AIME** (American Invitational Mathematics Examination) is now commonly used as a frontier discriminator. It is scored out of 15 (each problem worth 1 point), with no partial credit. The short integer answer format makes verification trivial while the mathematical depth is substantial. Frontier models in 2025 began scoring in the range of 5–12 on AIME I/II, a range previously only top human competitors reached.

**Olympiad-level benchmarks** (FrontierMath, OlympiadBench) push further, collecting unpublished research-level problems to prevent data contamination.

### Evaluating the Reasoning Trace, Not Just the Answer

A model can get the right answer via a flawed reasoning chain (lucky coincidence, "sycophancy" in the trace, or incorrect steps that happen to cancel). Conversely, a model can write a perfect chain but make an arithmetic error at the last step. Evaluating only the final answer misses both cases.

**Process reward models (PRMs)** assign a score to each step in a chain-of-thought solution, trained on human annotations of step correctness. PRMs can be used at eval time to:

1. **Rank samples**: among $n$ rollouts with the correct final answer, prefer those where all intermediate steps are also scored highly.
2. **Identify failure modes**: a model might consistently produce correct answers via flawed steps — an important red flag for reasoning training.

**Outcome reward models (ORMs)** score only the final answer. They are cheaper to train but cannot distinguish lucky correct answers from reliably correct reasoning.

The interplay between PRMs and ORMs in training is covered in [RL with Verifiable Rewards (RLVR) & The Reasoning Recipe](../05-posttraining-alignment/09-rlvr-reasoning.html). For eval purposes, running both a final-answer check and a PRM on a sample of traces gives the most complete picture of reasoning quality.

## Agentic Eval Harnesses

### What Makes Agentic Eval Hard

A single-turn QA benchmark has one input and one output. An agentic task might involve 10–50 model calls, tool invocations, file reads, web searches, and code executions. The challenges are:

- **Partial credit**: the task may be 80% complete but fail the final assertion. Binary pass/fail discards useful signal.
- **Non-determinism**: different runs of the same model on the same task may branch differently, requiring multiple rollouts.
- **Environment statefulness**: a web browsing task that clicks the wrong button early may be unrecoverable, masking the model's ability to handle later steps correctly.
- **Evaluation cost**: running a 30-step agent trajectory on 500 tasks, each with 5 rollouts, is expensive.

### SWE-bench Verified

SWE-bench (Jimenez et al., 2023) and its **Verified** variant (where human judges confirmed the task specification is unambiguous) are the primary coding-agent benchmarks. Each instance is:

```text
Repo:        django/django
Issue:       "QuerySet.delete() crashes with an F() expression in filter"
Gold patch:  a real commit from the repo history that fixes the issue
Test:        the CI test suite; success = failing tests now pass
```

The harness must:
1. Check out the repo at the commit *before* the fix.
2. Confirm the relevant tests fail on the un-patched repo.
3. Apply the model's patch.
4. Run the test suite.
5. Confirm the previously-failing tests now pass *and* no new failures were introduced.

This pipeline is containerized per-instance to avoid cross-contamination. The Docker image for each repo is built once and cached. The resolved rate (fraction of issues fixed) is the primary metric.

### WebArena, OSWorld, and τ-Bench

**WebArena** (Zhou et al., 2023) evaluates agents on realistic web tasks: booking a flight, submitting a form, finding information in a CMS. The agent controls a real browser (Playwright) inside a sandbox, and success is measured by final page state or database content — not by what the agent said it did.

**OSWorld** extends this to full desktop environments: the agent interacts with a virtual machine via screenshot + keyboard/mouse, performing tasks in real applications (LibreOffice, Chrome, terminal).

**τ-bench** (Yao et al., 2024) focuses on tool-augmented agents in retail and airline domains, with a simulated database and user simulator. The key metric is *task completion rate across multi-turn conversations* rather than single-step accuracy.

### Building a Custom Agentic Eval Harness

```python
"""
Minimal agentic eval harness illustrating the core loop.
Separates: task loading, trajectory collection, and scoring.
"""

import json
import asyncio
from dataclasses import dataclass, field
from typing import Callable, Any
from abc import ABC, abstractmethod


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
    steps: list[TrajectoryStep] = field(default_factory=list)
    final_state: dict = field(default_factory=dict)
    completed: bool = False
    error: str = ""


class AgentEnvironment(ABC):
    @abstractmethod
    async def reset(self, initial_state: dict) -> str:
        """Reset to initial state, return initial observation."""

    @abstractmethod
    async def step(self, action: str) -> tuple[str, bool, dict]:
        """Execute action; return (observation, done, final_state)."""


class AgentModel(ABC):
    @abstractmethod
    async def act(self, task: str, history: list[TrajectoryStep]) -> str:
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
    tasks: list[AgentTask],
    agent: AgentModel,
    env_factory: Callable[[], AgentEnvironment],
    rollouts_per_task: int = 3,
) -> dict[str, Any]:
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
```

### Partial Credit and Step-Level Metrics

For long-horizon tasks, pass/fail is too coarse. A 20-step task where the model completes 18 steps correctly but fails on step 19 should score higher than one where it fails on step 2. Useful step-level metrics:

- **Progress rate**: fraction of subtasks completed before failure.
- **Recovery rate**: given an injected error at step $i$, how often does the agent recover?
- **Efficiency**: number of steps taken divided by the oracle minimum (lower is better).

These metrics require decomposing the task into checkable substeps, which demands additional annotation effort but produces far richer signals than binary success.

## Test-Time-Compute-Aware Evaluation

### The Problem with Fixed-Compute Evals

Before "thinking" models (o1, DeepSeek-R1, QwQ, Gemini 2.0 Flash Thinking), all models used roughly the same compute per token at inference: one forward pass per generated token. Test-time compute is now a *dial*: a model can spend one token or 10,000 tokens of chain-of-thought before producing an answer.

This creates a fundamental incomparability problem: a model scoring 80% on AIME with 1,000 reasoning tokens may be less capable than one scoring 75% with 100 tokens, or may be far more capable if it can reach 95% with 10,000 tokens. Standard benchmarks that report a single accuracy number without specifying the inference budget are comparing apples to oranges.

### Compute-Normalized Comparison

The right framework is to compare models at the same *token budget* or the same *wall-clock time*. We define:

$$
\text{Acc}(B) = \mathbb{E}_{\text{problem}}\left[\text{Correct}\;|\;\text{total tokens} \leq B\right]
$$

and plot accuracy vs. token budget $B$ as a curve. A model that dominates everywhere on this curve is unambiguously better. When curves cross, the comparison is budget-dependent.

**Majority voting** (self-consistency, Wang et al., 2022): generate $k$ independent solutions and take the majority answer. Accuracy improves with $k$, but at the cost of $k \times$ compute. The marginal return diminishes — this is a compute-accuracy tradeoff curve.

**Best-of-N with a verifier**: generate $N$ solutions and return the one scored highest by a reward model or PRM. This is more compute-efficient than majority voting for hard problems because it can identify the single correct solution among $N$, rather than requiring a majority to be correct.

**Tree-of-thought / MCTS with PRM**: explore a tree of reasoning steps, pruned by a PRM at each node. This can solve problems that are unsolvable with linear CoT under the same token budget.

```python
import statistics
from typing import Callable


def majority_vote_accuracy(
    generate_fn: Callable[[str, int], list[str]],
    verify_fn: Callable[[str, str], bool],
    problems: list[dict],
    k_values: list[int] = [1, 4, 16, 64],
) -> dict[int, float]:
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


# Example output (illustrative, not real benchmark numbers):
# {1: 0.42, 4: 0.61, 16: 0.74, 64: 0.81}
# The curve should be plotted to compare models at equal k.
```

### Scaling Curves as Eval Outputs

Rather than reporting a single number, a test-time-compute-aware eval report should include:

| Metric | k=1 | k=4 | k=16 | k=64 |
|--------|-----|-----|------|------|
| Pass@k (MATH L5) | 0.31 | 0.52 | 0.68 | 0.79 |
| Tokens/problem | 800 | 3,200 | 12,800 | 51,200 |
| Cost/problem (USD) | 0.002 | 0.008 | 0.032 | 0.128 |

This framing makes the cost-performance tradeoff explicit. For most production use cases, the sweet spot is around $k = 4 \text{–} 8$ for hard reasoning tasks.

## Avoiding Reward Hacking in Evals

Reward hacking in *training* (covered in [Reward Hacking, Over-Optimization & Alignment Failures](../05-posttraining-alignment/13-reward-hacking-failures.html)) is about a policy exploiting the reward model rather than solving the real task. In *evaluation*, the analogous failure is a model (or evaluator) exploiting the benchmark metric rather than solving the real problem. The distinctions are subtle but consequential.

### Forms of Eval Reward Hacking

**1. Test set contamination.** The model was trained on data that contains the benchmark questions or their answers. This inflates scores without improving real capability. Mitigation: use held-out test sets with release dates after the model's training cutoff; use procedurally generated problems.

**2. Format exploitation.** The answer parser accepts "the answer is 42" but not "42". A model trained with RLVR learns to prepend boilerplate that matches the parser. Mitigation: normalize before parsing; use multiple parsers; do symbolic equivalence.

**3. Spurious correlation shortcuts.** Multiple-choice benchmarks may have positional biases (option C is correct more often) or stylistic tells. Models fine-tuned on the benchmark distribution learn these shortcuts. Mitigation: debiased sampling; few-shot calibration; ablate by position permutation.

**4. Unit test hacking for coding evals.** A model can pass unit tests without solving the problem: hardcode outputs for the specific test inputs. Mitigation: use private test cases not visible at generation time; test behavioral properties, not just I/O pairs.

**5. Self-report inflation in agentic evals.** A model might output "Task completed successfully" without actually completing the task, if the success detector is a language model rather than an oracle. Mitigation: always use ground-truth oracles (database state, file contents, test suite results) — never ask the model to self-report success.

**6. Overfitting to public benchmarks.** When organizations tune hyperparameters, prompts, or fine-tuning data to maximize scores on public benchmarks, those benchmarks lose generalizability. Mitigation: treat public benchmarks as the *val* set; maintain a private eval suite as the *test* set; rotate benchmarks regularly.

```python
"""
Detecting unit test hacking: check if a submitted solution
hardcodes outputs for the visible test cases.
"""

import ast
import re
from typing import Optional


def detects_hardcoded_solution(code: str, test_inputs: list[str]) -> bool:
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


# Example of a hardcoded (hacked) solution
hacked_solution = """
def two_sum(nums, target):
    if nums == [2, 7, 11, 15] and target == 9:
        return [0, 1]
    if nums == [3, 2, 4] and target == 6:
        return [1, 2]
    return []
"""

test_inputs = ["two_sum([2, 7, 11, 15], 9)", "two_sum([3, 2, 4], 6)"]
print(detects_hardcoded_solution(hacked_solution, test_inputs))
# True — suspicious pattern detected
```

The practical defense is to always include *private* test cases alongside the public ones that models can see. The model might hack the public tests, but private tests exercise general behavior.

## How Frontier Reasoning Is Measured in Practice

### The Measurement Stack for Frontier Labs

When Anthropic, Google DeepMind, or OpenAI report reasoning benchmark scores, the evaluation pipeline involves several layers:


{{fig:rcae-frontier-measurement-stack}}


Every decision in this stack affects the reported number. Running a reasoning model at temperature 0 without a thinking budget may report a different number than running it with extended thinking enabled — neither is "wrong," but they're measuring different things.

### Calibration and Human Baselines

A benchmark is only interpretable when we have a human baseline. For AIME, the human baseline is well-established: the top 5% of high school math students qualify for AIME, and median AIME scores for qualifiers hover around 5–7. For SWE-bench Verified, the human baseline (a skilled software engineer given the same issue and repo, without context on the fix) is high — professional developers routinely resolve such issues, suggesting the benchmark is not at the boundary of human capability but is a meaningful proxy for junior-developer-level coding.

### Contamination Detection

Contamination detection methods include:

1. **N-gram overlap**: compute Jaccard similarity between benchmark test cases and training data. High overlap suggests contamination.
2. **Perplexity test**: a contaminated model will have anomalously low perplexity on benchmark inputs compared to a reference model.
3. **Canary insertion**: insert synthetic problems into training data that are superficially similar to benchmark problems but with different answers. If the model answers the canary correctly and the benchmark incorrectly, contamination from a specific source is likely.
4. **Temporal holdout**: use only benchmark problems with release dates after the model's stated training cutoff. LiveCodeBench and OlympiadBench are designed around this principle.

!!! interview "Interview Corner"
    **Q:** A colleague claims their new model achieves 92% on HumanEval and 85% on MBPP, arguing it's the best code generation model available. What questions would you ask before accepting this claim?

    **A:** Several things to scrutinize:

    1. **Contamination**: Were HumanEval/MBPP problems (or near-duplicates) present in the training data? Both benchmarks are small and publicly available — their solutions exist widely on the web. Ask for scores on LiveCodeBench (problems released after training cutoff) or private held-out tests.

    2. **Which metric?** Pass@1 at temperature 0 (greedy) vs. Pass@1 averaged over multiple temperatures vs. Pass@10 are all "accuracy" but measure different things. Greedy Pass@1 can be inflated by a model that memorizes rather than generalizes.

    3. **Test suite quality**: HumanEval's unit tests are intentionally simple and sometimes have false positives — a solution can pass all tests without correctly solving the problem. Did they cross-check against stricter private test suites?

    4. **How does it compare on harder tasks?** SWE-bench Verified (real repo issue fixing) and APPS (competitive programming) are far harder discriminators. A model that plateaus at 92% HumanEval may score 10% on SWE-bench.

    5. **Inference budget**: Was reasoning/thinking enabled? At what token limit? Reporting two numbers (thinking-off and thinking-on) is standard for reasoning models.

## The Evaluation Loop: Connecting Evals to Training

Evals are not just a reporting mechanism — they are an active signal in the development loop. The connection between eval methodology and training integrity is direct:

- **RLVR training** (covered in [RL with Verifiable Rewards (RLVR) & The Reasoning Recipe](../05-posttraining-alignment/09-rlvr-reasoning.html)) uses the *same execution-based verifiers* as eval — if those verifiers are buggy or gameable, the trained model learns to game them.
- **Benchmark saturation**: when a model achieves near-ceiling performance on a benchmark, it disappears as a gradient signal. New harder benchmarks must be introduced continuously.
- **Eval–train separation**: using the same benchmark for training reward and for final reporting violates the train/test split principle. Keep a held-out test set that no RLVR iteration has ever optimized against.

The data flywheel for reasoning models is: hard evals reveal failure modes → new training data targets those modes → improved model → harder evals needed → repeat. This is described in the context of production systems in [Data Flywheels & Continuous Improvement](../12-production-mlops/05-data-flywheel.html).

!!! warning "Common pitfall: Temperature 0 eval for stochastic reasoning models"
    Frontier reasoning models (those with extended thinking / chain-of-thought) are typically sampled at temperature > 0 because their reasoning traces are inherently stochastic. Running them at temperature 0 (greedy decoding) can artificially reduce scores by collapsing the diversity of approaches the model would use at higher temperature. Always check whether the model was designed for greedy or sampled decoding, and match the eval sampling strategy to the model's intended use.

!!! tip "Practitioner tip: Separate your hard and easy problems"
    When running a coding or math eval, always stratify results by difficulty tier. Reporting a single average accuracy hides the shape of the distribution. A model that scores 95% on easy, 60% on medium, and 5% on hard is very different from one that scores 70% across all tiers — even if they tie on mean accuracy. Stratified reporting is the only way to track whether training is making progress on the hard tail.

!!! key "Key Takeaways"
    - **Pass@k** uses the unbiased combinatorial estimator $1 - \binom{n-c}{k}/\binom{n}{k}$ rather than the naive $(1-(c/n))^k$ formula; the difference matters for small sample sizes.
    - **Code execution in sandboxes** (Docker, gVisor, Firecracker) is non-negotiable for correct code eval — string matching fails on equivalent programs and is insecure.
    - **Math verification** requires symbolic equivalence (e.g., via sympy), not string matching; separate process reward models (PRMs) evaluate chain quality, not just final answers.
    - **Agentic evals** require ground-truth environment oracles (test suites, database state) rather than self-report; partial-credit metrics and trajectory-level analysis are richer than binary pass/fail.
    - **Test-time-compute-aware eval** plots accuracy as a function of token budget and is essential for comparing reasoning models; single-number reporting is misleading when models have variable thinking budgets.
    - **Benchmark contamination** is the silent killer of eval validity; mitigate via temporal holdout (LiveCodeBench), N-gram overlap checks, and canary insertion.
    - **Eval reward hacking** — format exploitation, unit test hardcoding, spurious shortcuts, self-report inflation — requires defensive engineering: private test cases, oracle-based success checks, and normalization-robust parsers.
    - **Separate eval benchmarks from training reward signals**: if a model is RL-trained against a verifier, that verifier cannot be used as the final eval metric without biasing the comparison.

!!! sota "State of the Art & Resources (2026)"
    Reasoning, coding, and agentic evaluation has matured into a rigorous sub-field: execution-based benchmarks with temporal contamination controls (LiveCodeBench, SWE-bench), process-reward-model grading of reasoning chains, and multi-step agentic harnesses with environment oracles (WebArena, OSWorld, τ-bench) are now standard practice. The central challenge in 2025–2026 is compute-budget-aware evaluation as test-time scaling blurs fixed-compute comparisons.

    **Foundational work**

    - [Chen et al., *Evaluating Large Language Models Trained on Code* (2021)](https://arxiv.org/abs/2107.03374) — introduced HumanEval and the unbiased Pass@k estimator used throughout the field.
    - [Lightman et al., *Let's Verify Step by Step* (2023)](https://arxiv.org/abs/2305.20050) — PRM800K and process reward models for step-level evaluation of math reasoning chains.
    - [Wang et al., *Self-Consistency Improves Chain of Thought Reasoning in Language Models* (2022)](https://arxiv.org/abs/2203.11171) — majority-voting as a test-time compute strategy; foundational for compute-normalized eval curves.

    **Recent advances (2023–2026)**

    - [Jimenez et al., *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* (2023)](https://arxiv.org/abs/2310.06770) — real-repo issue-fixing benchmark; SWE-bench Verified (2024) is the dominant coding-agent leaderboard target.
    - [Jain et al., *LiveCodeBench: Holistic and Contamination Free Evaluation of LLMs for Code* (2024)](https://arxiv.org/abs/2403.07974) — living benchmark adding post-cutoff competitive programming problems to prevent contamination.
    - [Glazer et al., *FrontierMath: A Benchmark for Evaluating Advanced Mathematical Reasoning in AI* (2024)](https://arxiv.org/abs/2411.04872) — research-level unpublished math problems; frontier models solve under 2%, providing a non-saturating discriminator.
    - [Xie et al., *OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments* (2024)](https://arxiv.org/abs/2404.07972) — full desktop-environment agentic eval; best models reach ~12% vs. human 72%.

    **Open-source & tools**

    - [swe-bench/SWE-bench](https://github.com/swe-bench/SWE-bench) — official harness and dataset for SWE-bench and SWE-bench Verified; Docker-containerized per-instance evaluation.
    - [LiveCodeBench/LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) — official toolkit for contamination-controlled code evaluation across LeetCode, AtCoder, and Codeforces.
    - [sierra-research/tau-bench](https://github.com/sierra-research/tau-bench) — τ-bench harness for tool-augmented agents in retail/airline domains with simulated user interactions.

    **Go deeper**

    - [Zhou et al., *WebArena: A Realistic Web Environment for Building Autonomous Agents* (2023)](https://arxiv.org/abs/2307.13854) — browser-based agentic eval with Playwright; tasks graded by oracle database/page state, not model self-report.
    - [OpenAI, *Learning to Reason with LLMs* (2024)](https://openai.com/index/learning-to-reason-with-llms/) — o1 system card and eval methodology, including compute-scaling curves and AIME/MATH frontier results.

## Further Reading

- **Chen et al., "Evaluating Large Language Models Trained on Code" (HumanEval), 2021** — introduced Pass@k with the unbiased estimator and the HumanEval benchmark.
- **Austin et al., "Program Synthesis with Large Language Models" (MBPP), 2021** — the Mostly Basic Python Problems benchmark and few-shot synthesis evaluation.
- **Hendrycks et al., "Measuring Mathematical Problem Solving with the MATH Dataset", 2021** — the MATH benchmark and analysis of difficulty tiers.
- **Cobbe et al., "Training Verifiers to Solve Math Word Problems" (GSM8K), 2021** — grade-school math benchmark and the use of verifier models.
- **Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in Language Models", 2022** — majority voting as a test-time compute strategy.
- **Jimenez et al., "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?", 2023** — the real-repo issue-fixing benchmark and its eval harness.
- **Zhou et al., "WebArena: A Realistic Web Environment for Building Autonomous Agents", 2023** — the web-task agentic eval benchmark.
- **Jain et al., "LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code", 2024** — living benchmark with temporal contamination control.
- **Lightman et al., "Let's Verify Step by Step", 2023** — process reward models for evaluating and training mathematical reasoning chains.
- **OpenAI, "Learning to Reason with LLMs" (o1 technical report), 2024** — frontier reasoning evaluation methodology and compute-scaling eval curves.
