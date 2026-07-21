# 6.8 Reward Engineering, Verifiers & Sandboxes

The training signal in reinforcement learning from human feedback (RLHF) and in RL with verifiable rewards (RLVR) is just a scalar $r \in \mathbb{R}$ returned for each model completion. Everything the model learns — its reasoning habits, its code style, its factual accuracy — is shaped by what that scalar measures. Getting the reward function right is therefore the most consequential engineering decision in your entire post-training pipeline.

This chapter is a practical deep-dive into reward engineering: how to build rule-based verifiers that check math answers or run code tests, how to sandbox arbitrary execution safely at scale, how to use a second LLM as a judge, how to combine multiple reward signals without one dominating, and how to instrument and serve the whole system at training throughput. We assume familiarity with the basic RLHF/RLVR pipeline described in [RL with Verifiable Rewards (RLVR) & The Reasoning Recipe](../05-posttraining-alignment/09-rlvr-reasoning.html) and the training loop mechanics from [The Generation–Training Loop & Rollout Engines](../06-rl-infra/02-generation-training-loop.html).

## The Anatomy of a Reward Function

Every reward function takes a (prompt, completion) pair and optionally a reference answer or test suite, and returns a scalar. In practice they come in three families:

| Family | Examples | Pros | Cons |
|---|---|---|---|
| **Rule-based verifier** | Math equivalence, regex match, code unit test | Deterministic, zero cost at training time | Only defined for tasks with ground truth |
| **LLM-judge** | GPT-4o / Claude scoring correctness, helpfulness | Works for open-ended tasks | Expensive, noisy, gameable |
| **Learned reward model** | Bradley-Terry model trained on preference data | Smooth signal, general-purpose | Reward hacking, needs fresh data |

In RLVR (the paradigm that produced DeepSeek-R1 and related reasoning models), the goal is to rely on the first family as much as possible. The key insight is that for tasks with a checkable ground truth, a rule-based verifier is both cheaper and harder to hack than a learned reward model. We still need LLM judges and learned reward models for tasks like "write a good essay," but we should use them sparingly.

### Reward Signal Flow


{{fig:reward-signal-flow}}


The weighted combination step is itself a design choice (see section on multi-objective rewards below). For now, note that different reward sources run at different latencies and different costs, which matters a lot for training throughput.

## Rule-Based Verifiers

### Math Equivalence Checking

The canonical RLVR task is mathematical reasoning: generate a chain-of-thought and a boxed final answer, then check if the answer is mathematically equivalent to the ground-truth. "Mathematically equivalent" is harder than string equality.

Consider: `1/2`, `0.5`, `\frac{1}{2}`, `50\%`, and `0.50` all represent the same value. A naive string comparison gives sparse reward and introduces arbitrary formatting bias. We need symbolic or numeric normalization.

{{fig:math-equivalence-ladder}}

```python
"""
math_verifier.py — robust math answer checker for RL training.

Handles:
  - Fraction strings: "1/2", "3 1/4" (mixed numbers)
  - LaTeX: r"\frac{3}{4}", r"\sqrt{2}", r"2^{10}"
  - Percentages: "50%" -> 0.5 comparison
  - Sets/tuples: "{1, 2}" == "{2, 1}"
  - Symbolic via sympy with numeric fallback
"""

import re
import math
from fractions import Fraction
from typing import Optional, Union
import sympy
from sympy.parsing.latex import parse_latex
from sympy import simplify, N


def normalize_latex_number(s: str) -> str:
    """Strip LaTeX boilerplate that doesn't change value."""
    s = s.strip()
    # Remove \boxed{...}
    s = re.sub(r'\\boxed\{(.+?)\}', r'\1', s)
    # Remove dollar signs
    s = s.replace('$', '').strip()
    # Normalize whitespace
    s = re.sub(r'\s+', ' ', s)
    return s


def try_numeric(s: str) -> Optional[float]:
    """
    Try to evaluate s as a number.
    Returns float or None if not parseable as a pure number.
    """
    s = normalize_latex_number(s)
    # Handle percentages: "50%" → 0.5
    if s.endswith('%'):
        try:
            return float(s[:-1]) / 100.0
        except ValueError:
            pass
    # Handle fractions: "3/4"
    try:
        return float(Fraction(s))
    except (ValueError, ZeroDivisionError):
        pass
    # Plain float
    try:
        return float(s)
    except ValueError:
        pass
    return None


def try_sympy(s: str) -> Optional[sympy.Expr]:
    """Parse a LaTeX string with sympy."""
    try:
        expr = parse_latex(normalize_latex_number(s))
        return expr
    except Exception:
        return None


def math_equivalent(pred: str, gold: str, rtol: float = 1e-6) -> bool:
    """
    Return True iff pred and gold represent the same mathematical value.
    
    Strategy:
      1. Exact string match after normalization
      2. Both are numeric → compare with relative tolerance
      3. Both parse as sympy expressions → simplify(pred - gold) == 0
      4. Sympy + numeric fallback (N(expr))
    """
    pred = normalize_latex_number(pred)
    gold = normalize_latex_number(gold)

    # 1. Exact match
    if pred == gold:
        return True

    # 2. Numeric
    pn = try_numeric(pred)
    gn = try_numeric(gold)
    if pn is not None and gn is not None:
        if gn == 0:
            return abs(pn) < 1e-9
        return abs(pn - gn) / (abs(gn) + 1e-12) < rtol

    # 3. Symbolic
    pe = try_sympy(pred)
    ge = try_sympy(gold)
    if pe is not None and ge is not None:
        diff = simplify(pe - ge)
        if diff == 0:
            return True
        # Numeric fallback: evaluate to float
        try:
            pf = float(N(pe, 30))
            gf = float(N(ge, 30))
            if gf == 0:
                return abs(pf) < 1e-9
            return abs(pf - gf) / (abs(gf) + 1e-12) < rtol
        except Exception:
            pass

    return False


def extract_answer_from_completion(completion: str) -> str:
    """
    Extract the model's final answer.
    Expects the model to write \\boxed{answer} or <answer>...</answer>.
    Returns empty string if not found.
    """
    # Try \boxed{...} (LaTeX math)
    m = re.search(r'\\boxed\{([^}]+)\}', completion)
    if m:
        return m.group(1).strip()
    # Try <answer>...</answer> tag
    m = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: last number on a "The answer is X" line
    m = re.search(r'[Tt]he answer is[:\s]+([^\n.]+)', completion)
    if m:
        return m.group(1).strip()
    return ""


def math_reward(prompt: str, completion: str, gold_answer: str) -> float:
    """
    Main entry point: return reward in {0.0, 1.0} for a math response.
    A partial reward of 0.5 is returned when the format is correct
    but the answer is wrong, to encourage the model to use \boxed{}.
    """
    pred = extract_answer_from_completion(completion)
    if not pred:
        return 0.0  # No answer extracted

    if math_equivalent(pred, gold_answer):
        return 1.0
    else:
        # Small reward for correct format, zero for content
        return 0.1  # format reward (optional — see section on reward shaping)
```

!!! warning "Floating-point equality traps"
    Never use `pred == gold` on raw strings. `"0.333"` and `"1/3"` are mathematically the same. `"3.0"` and `"3"` differ as strings. Always normalize before comparing, and use a tolerance for floats (relative tolerance around $10^{-6}$ works for competition math).

### Code Execution Verifiers

For coding tasks, the verifier runs unit tests inside a sandbox and returns pass/fail. The reward can be binary (all tests pass = 1.0) or proportional (fraction of tests passed).

```python
"""
code_verifier.py — unit-test-based reward for code generation.

This module orchestrates sandboxed execution (see sandboxes section)
and returns a float reward based on test outcomes.
"""

import subprocess
import tempfile
import os
import json
import textwrap
from dataclasses import dataclass


@dataclass
class TestResult:
    passed: int         # number of tests that passed
    total: int          # total tests
    timed_out: bool     # did any test time out?
    error: str          # compilation/import error if any


def run_tests_in_subprocess(
    code: str,
    test_suite: str,
    timeout_seconds: float = 5.0,
) -> TestResult:
    """
    Execute generated code + test suite in a temporary directory.
    Returns TestResult.

    IMPORTANT: For production use, wrap this in a proper sandbox
    (see the Sandboxed Execution section). This bare subprocess
    approach is only safe on air-gapped machines or inside a container.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write generated code
        code_path = os.path.join(tmpdir, "solution.py")
        with open(code_path, "w") as f:
            f.write(code)

        # Write test runner that imports solution.py
        runner = textwrap.dedent(f"""
            import sys, json, traceback
            sys.path.insert(0, {repr(tmpdir)})
            
            results = {{"passed": 0, "total": 0, "error": ""}}
            try:
                from solution import *
                {textwrap.indent(test_suite, "                ")}
                # Each test function is test_<name>; discover and run
                test_fns = [v for k, v in globals().items() if k.startswith("test_")]
                results["total"] = len(test_fns)
                for fn in test_fns:
                    try:
                        fn()
                        results["passed"] += 1
                    except Exception:
                        pass
            except Exception as e:
                results["error"] = traceback.format_exc()
                results["total"] = 1  # At least one test failed
            
            print(json.dumps(results))
        """)
        runner_path = os.path.join(tmpdir, "runner.py")
        with open(runner_path, "w") as f:
            f.write(runner)

        try:
            proc = subprocess.run(
                ["python", runner_path],
                capture_output=True, text=True,
                timeout=timeout_seconds,
            )
            if proc.returncode != 0 and not proc.stdout:
                return TestResult(0, 1, False, proc.stderr[:500])
            data = json.loads(proc.stdout.strip())
            return TestResult(
                passed=data["passed"],
                total=max(data["total"], 1),
                timed_out=False,
                error=data.get("error", ""),
            )
        except subprocess.TimeoutExpired:
            return TestResult(0, 1, timed_out=True, error="timeout")
        except json.JSONDecodeError as e:
            return TestResult(0, 1, False, f"JSON decode error: {e}")


def code_reward(
    prompt: str,
    completion: str,
    test_suite: str,
    partial_credit: bool = True,
) -> float:
    """
    Reward for code generation:
      1.0  if all tests pass
      k/n  if partial_credit and k of n tests pass
      0.0  if syntax error, timeout, or all tests fail
    """
    # Strip markdown code fences if present
    code = completion
    m = __import__("re").search(r"```(?:python)?\n(.*?)```", completion, __import__("re").DOTALL)
    if m:
        code = m.group(1)

    result = run_tests_in_subprocess(code, test_suite)

    if result.timed_out:
        return 0.0
    if result.error and result.passed == 0:
        return 0.0
    if partial_credit:
        return result.passed / result.total
    else:
        return 1.0 if result.passed == result.total else 0.0
```

## Sandboxed Execution: Security, Isolation & Throughput

Running arbitrary model-generated code is a serious security risk. A model that has learned to manipulate the reward function will eventually generate code that:

- Reads its own test file and hard-codes the expected output.
- Calls `os.kill(os.getpid(), signal.SIGTERM)` to time out, then retries with correct answers.
- Escapes to the host via container breakouts.
- Consumes unbounded CPU/memory, starving other workers.

A production sandbox must enforce: **no network access, no filesystem writes outside tmpdir, CPU and memory limits, syscall filtering, and process tree isolation**.

### Sandbox Architectures


{{fig:sandbox-arch-options}}


For training at scale (on the order of thousands of completions per second), Docker-per-sample is too slow. The standard approach is a warm pool of microVMs or containers that are snapshotted after initialization, then restored to a clean state between runs.

### Building a Sandboxed Execution Service

```python
"""
sandbox_pool.py — reusable sandbox pool using Docker containers.

In production you would use Firecracker or gVisor microVMs;
Docker containers are shown here for clarity.

Each sandbox is a long-lived container. We send code + tests
over stdin and get results back over stdout, avoiding the per-run
container startup cost (~200ms) by reusing warm containers.
"""

import threading
import queue
import subprocess
import json
import time
from dataclasses import dataclass, field
from typing import Optional


SANDBOX_IMAGE = "python:3.11-slim"
# Dockerfile for the sandbox image would add no extra packages
# and run as a non-root user, but we keep it simple here.

SANDBOX_RUNNER = """
import sys, json, traceback, signal, resource

# Hard memory limit: 256 MB
resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))

# Read tasks from stdin until EOF
for line in sys.stdin:
    task = json.loads(line)
    code = task["code"]
    tests = task["tests"]
    result = {"passed": 0, "total": 0, "error": ""}
    
    try:
        exec_globals = {}
        exec(compile(code, "<generated>", "exec"), exec_globals)
        exec(compile(tests, "<tests>", "exec"), exec_globals)
        fns = [v for k, v in exec_globals.items() if k.startswith("test_")]
        result["total"] = len(fns)
        for fn in fns:
            try:
                fn()
                result["passed"] += 1
            except Exception:
                pass
    except Exception:
        result["error"] = traceback.format_exc()[-500:]
        result["total"] = 1
    
    print(json.dumps(result), flush=True)
"""


@dataclass
class SandboxWorker:
    proc: subprocess.Popen
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_used: float = field(default_factory=time.time)


class SandboxPool:
    """
    A pool of warm sandbox processes.
    Thread-safe: acquire() blocks until a worker is free.
    """

    def __init__(self, pool_size: int = 8, timeout: float = 5.0):
        self.timeout = timeout
        self._available: queue.Queue = queue.Queue()
        self._all_workers = []
        for _ in range(pool_size):
            w = self._spawn()
            self._all_workers.append(w)
            self._available.put(w)

    def _spawn(self) -> SandboxWorker:
        """Start a sandbox process that reads JSON tasks from stdin."""
        proc = subprocess.Popen(
            [
                "docker", "run", "--rm", "--interactive",
                "--network=none",           # No network
                "--memory=256m",            # 256 MB RAM limit
                "--cpus=1",                 # 1 vCPU
                "--pids-limit=50",          # No fork bombs
                "--read-only",              # Read-only root filesystem
                "--tmpfs=/tmp:size=64m",    # Small writable /tmp
                SANDBOX_IMAGE,
                "python", "-c", SANDBOX_RUNNER,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return SandboxWorker(proc=proc)

    def execute(self, code: str, tests: str) -> dict:
        """
        Run code + tests in a sandbox. Blocks until a worker is available.
        Returns {"passed": int, "total": int, "error": str}.
        """
        worker: SandboxWorker = self._available.get(timeout=30.0)
        try:
            task = json.dumps({"code": code, "tests": tests}) + "\n"
            worker.proc.stdin.write(task)
            worker.proc.stdin.flush()
            # Read response with timeout
            worker.proc.stdout._sock.settimeout(self.timeout)
            line = worker.proc.stdout.readline()
            return json.loads(line)
        except Exception as e:
            # Worker is dead; replace it
            worker.proc.kill()
            worker = self._spawn()
            return {"passed": 0, "total": 1, "error": str(e)}
        finally:
            worker.last_used = time.time()
            self._available.put(worker)  # Return worker to pool
```

!!! tip "Practical sandbox tuning"
    For RLVR training at scale (e.g., 4096 rollout completions per training step), you typically need a pool of 64-256 sandbox workers per GPU node, each handling ~20 requests/second. The bottleneck shifts from process startup (eliminated by warm pools) to test suite I/O. Keep test suites small (under 1 KB) and use in-memory temp files rather than disk I/O.

### Throughput Arithmetic

!!! example "Sandbox throughput sizing"
    Suppose your training run generates 4096 completions per step, each requiring ~50 ms of sandbox execution (10 unit tests, each 5 ms), and a new step begins every 30 seconds.
    
    Required throughput: $4096 / 30 \approx 137$ completions/second.
    
    Per worker capacity: $1000 \text{ ms} / 50 \text{ ms} = 20$ completions/second.
    
    Workers needed: $\lceil 137 / 20 \rceil = 7$ workers.
    
    In practice you want $2\text{–}3\times$ headroom for variance and retries, so **~20 sandbox workers** suffices for this configuration. At 256 MB RAM each, that is 5 GB of sandbox overhead on the reward server — negligible compared to the GPU memory.
    
    If test suites are more expensive (e.g., compiling C++ or running integration tests at 2 seconds each), the same arithmetic gives ~820 workers, which requires a dedicated cluster of CPU machines, exactly the architecture used by competitive coding RL systems.

## LLM-Judge Rewards

For open-ended tasks where there is no crisp ground truth — writing quality, helpfulness, reasoning coherence — we need a second LLM to score completions. This is sometimes called RLAIF (RL from AI Feedback), covered in depth in [Constitutional AI, RLAIF & Self-Improvement](../05-posttraining-alignment/11-constitutional-rlaif.html).

### Designing a Judge Prompt

A good judge prompt:
1. Specifies the evaluation criteria precisely (correctness, clarity, step-by-step validity).
2. Uses a reference answer or rubric when available.
3. Asks the judge to reason before scoring (chain-of-thought in the judge improves calibration).
4. Outputs a structured response to make score extraction reliable.

```python
"""
llm_judge.py — LLM-as-a-judge reward for open-ended tasks.

Uses an OpenAI-compatible API. Replace with your preferred provider.
"""

import re
import asyncio
import openai
from typing import Optional


JUDGE_SYSTEM = """
You are an expert evaluator for mathematical reasoning and problem solving.
You will be given a problem, a reference solution, and a model response.
Your task is to evaluate the model response on a scale of 0 to 10.

Evaluation criteria:
- Correctness (5 pts): Is the final answer correct?
- Reasoning quality (3 pts): Is the chain-of-thought valid, with each step justified?
- Clarity (2 pts): Is the solution well-organized and readable?

Output format:
<thinking>
[Your analysis of the response]
</thinking>
<score>N</score>

Where N is an integer from 0 to 10.
""".strip()


JUDGE_USER_TEMPLATE = """
PROBLEM:
{problem}

REFERENCE SOLUTION:
{reference}

MODEL RESPONSE:
{response}

Please evaluate the model response using the criteria above.
""".strip()


async def judge_single(
    problem: str,
    response: str,
    reference: str,
    judge_model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> float:
    """
    Returns a normalized reward in [0, 1] for a single (problem, response) pair.
    Calls the judge LLM asynchronously.
    """
    client = openai.AsyncOpenAI()
    
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
            problem=problem,
            reference=reference,
            response=response,
        )},
    ]
    
    resp = await client.chat.completions.create(
        model=judge_model,
        messages=messages,
        temperature=temperature,
        max_tokens=512,
    )
    
    text = resp.choices[0].message.content
    m = re.search(r"<score>(\d+(?:\.\d+)?)</score>", text)
    if m:
        raw_score = float(m.group(1))
        return min(max(raw_score / 10.0, 0.0), 1.0)
    
    # Fallback: try to parse any integer at the end of the response
    nums = re.findall(r"\b(\d+)\b", text)
    if nums:
        raw = float(nums[-1])
        return min(max(raw / 10.0, 0.0), 1.0)
    
    return 0.5  # Uncertain — return neutral score


async def batch_judge(
    problems: list[str],
    responses: list[str],
    references: list[str],
    judge_model: str = "gpt-4o-mini",
    concurrency: int = 32,
) -> list[float]:
    """
    Evaluate a batch of completions concurrently.
    concurrency limits simultaneous API calls to avoid rate limits.
    """
    sem = asyncio.Semaphore(concurrency)
    
    async def bounded_judge(p, r, ref):
        async with sem:
            return await judge_single(p, r, ref, judge_model)
    
    tasks = [
        bounded_judge(p, r, ref)
        for p, r, ref in zip(problems, responses, references)
    ]
    return await asyncio.gather(*tasks)
```

### Judge Calibration and Positional Bias

LLM judges suffer from **positional bias** (they prefer whichever answer appears first), **verbosity bias** (longer responses score higher regardless of quality), and **self-preference bias** (a judge from the same family as the policy gives spuriously high scores). Mitigation strategies:

- Swap the order of responses in pairwise comparisons and average the scores.
- Use a judge from a different model family than the policy.
- Normalize scores within each prompt batch: $\tilde{r}_i = (r_i - \mu_{\text{batch}}) / \sigma_{\text{batch}}$.
- Include a "null" baseline completion (e.g., "I don't know") to anchor the scale.

## Reward Shaping and the Format vs. Correctness Tradeoff

Raw binary rewards (correct/incorrect) create sparse gradients, especially early in training when the model rarely reaches the right answer. Reward shaping introduces auxiliary rewards to guide the model toward behaviors that are likely to lead to the correct answer.

### Format Rewards

A common technique in RLVR is to give a small positive reward for using the expected output format (e.g., a `<think>...</think>` reasoning block followed by a `\boxed{answer}`), even when the answer is wrong. This solves a cold-start problem: the model must first learn to produce structured output before it can receive meaningful correctness rewards.

$$
r_{\text{total}} = r_{\text{correctness}} + \lambda_f \cdot r_{\text{format}}
$$

{{fig:format-vs-correctness-shaping}}

where $r_{\text{format}} \in \{0, 1\}$ indicates whether the output satisfies the format, and $\lambda_f$ is typically small (on the order of 0.1 to 0.2) to avoid the model gaming format at the expense of correctness.

!!! warning "Format reward gaming"
    If $\lambda_f$ is too large, the model learns to produce syntactically correct but semantically empty responses. For example, it may always write `<think>Let me solve this step by step.</think>\boxed{0}` — perfect format, always wrong answer, but nonzero reward. Monitor the correlation between format reward and correctness reward throughout training; if they decouple, reduce $\lambda_f$.

### Process Reward Models (PRMs)

Instead of rewarding only the final answer, a Process Reward Model (PRM) assigns a reward to each step in the chain-of-thought. This is covered in depth in [RL with Verifiable Rewards (RLVR) & The Reasoning Recipe](../05-posttraining-alignment/09-rlvr-reasoning.html); here we focus on the infrastructure interface.

```python
"""
prm_reward.py — step-level reward using a process reward model.

The PRM is a separate model (often a smaller classifier fine-tuned
on step-level correctness annotations) that scores each reasoning step.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class ProcessRewardModel:
    """
    Wraps a PRM that scores individual reasoning steps.
    Input: a (problem, partial_solution_so_far) pair.
    Output: a scalar score in [0, 1] for the most recent step.
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1,
        ).to(device).eval()
        self.device = device

    @torch.no_grad()
    def score_steps(
        self,
        problem: str,
        steps: list[str],
        batch_size: int = 32,
    ) -> list[float]:
        """
        Score each step in a chain-of-thought.
        steps: list of individual reasoning steps (not cumulative).
        Returns a list of scalar rewards, one per step.
        """
        scores = []
        # Build cumulative context for each step
        contexts = []
        for i in range(len(steps)):
            # Include all steps up to and including step i
            partial = problem + "\n\n" + "\n".join(steps[:i+1])
            contexts.append(partial)

        # Batch inference
        for start in range(0, len(contexts), batch_size):
            batch = contexts[start:start + batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.device)
            logits = self.model(**enc).logits.squeeze(-1)
            # Sigmoid to get probability of "step is correct"
            probs = torch.sigmoid(logits).cpu().tolist()
            if isinstance(probs, float):
                probs = [probs]
            scores.extend(probs)

        return scores

    def aggregate(
        self,
        step_scores: list[float],
        method: str = "min",
    ) -> float:
        """
        Aggregate step scores into a single episode reward.
        
        'min': reward = min step score (weakest-link; used in Math-Shepherd)
        'last': reward = score of final step (common in practice)
        'mean': reward = mean of all step scores
        """
        if not step_scores:
            return 0.0
        if method == "min":
            return min(step_scores)
        elif method == "last":
            return step_scores[-1]
        elif method == "mean":
            return sum(step_scores) / len(step_scores)
        raise ValueError(f"Unknown aggregation method: {method}")
```

## Multi-Objective and Weighted Rewards

Most production RLVR runs combine multiple reward signals:

| Signal | Weight | Purpose |
|---|---|---|
| Correctness (rule-based) | 1.0 | Primary learning signal |
| Format compliance | 0.1–0.2 | Cold-start / structure |
| Reasoning length penalty | -0.01 per token over limit | Control verbosity |
| Safety classifier | -5.0 if flagged | Hard constraint |
| LLM judge (helpfulness) | 0.3 | Open-ended quality |

The weighted combination is:

$$
r_{\text{total}} = \sum_{k} w_k \cdot r_k
$$

This is simple but has pitfalls. If reward magnitudes differ by orders of magnitude (e.g., correctness ∈ {0, 1} but a continuous fluency score ∈ [0, 100]), one term will dominate the gradient. Always normalize rewards to similar scales before combining.

### Reward Normalization Per Prompt

A robust pattern used in practice is to normalize rewards within the batch of completions for a single prompt (a "prompt group"):

$$
\tilde{r}_{i} = \frac{r_i - \mu_{\text{group}}}{\sigma_{\text{group}} + \epsilon}
$$

This is exactly the normalization used in GRPO (Group Relative Policy Optimization), discussed in [GRPO, RLOO & Critic-Free RL](../05-posttraining-alignment/08-grpo-rloo.html). It ensures that the advantage signal is zero-mean per prompt regardless of the absolute reward scale.

```python
"""
reward_combiner.py — multi-objective reward combining and normalization.
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass
class RewardComponent:
    name: str
    fn: Callable                  # (prompt, completion, metadata) -> float
    weight: float                 # Relative importance
    clip_min: float = -10.0       # Clip extreme values
    clip_max: float = 10.0


class MultiObjectiveReward:
    """
    Combines multiple reward signals, normalizes per group,
    and returns the final scalar reward for each completion.
    """

    def __init__(self, components: list[RewardComponent]):
        self.components = components

    def compute(
        self,
        prompts: list[str],
        completions: list[str],
        metadata: list[dict],
        normalize_per_group: bool = True,
    ) -> tuple[list[float], dict[str, list[float]]]:
        """
        Returns (total_rewards, per_component_rewards).
        
        Assumes each (prompt, completions) group has the same prompt
        and completions are ordered: all completions for prompt 0,
        then all for prompt 1, etc.
        
        metadata: list of dicts with per-sample info (e.g., gold_answer).
        normalize_per_group: apply Z-score normalization within groups.
        """
        n = len(completions)
        component_rewards = {c.name: np.zeros(n) for c in self.components}
        
        # Evaluate each component
        for comp in self.components:
            for i, (prompt, completion, meta) in enumerate(
                zip(prompts, completions, metadata)
            ):
                raw = comp.fn(prompt, completion, meta)
                clipped = np.clip(raw, comp.clip_min, comp.clip_max)
                component_rewards[comp.name][i] = clipped

        # Weighted sum
        total = np.zeros(n)
        for comp in self.components:
            total += comp.weight * component_rewards[comp.name]

        # Normalize per group if requested
        if normalize_per_group:
            # Group by unique prompt (assumes they come in contiguous blocks)
            unique_prompts = []
            seen = {}
            for p in prompts:
                if p not in seen:
                    seen[p] = len(unique_prompts)
                    unique_prompts.append(p)
            for prompt in unique_prompts:
                idx = [i for i, p in enumerate(prompts) if p == prompt]
                group = total[idx]
                mu, sigma = group.mean(), group.std()
                total[idx] = (group - mu) / (sigma + 1e-8)

        return total.tolist(), {k: v.tolist() for k, v in component_rewards.items()}
```

!!! interview "Interview Corner"
    **Q:** In an RLVR training run for math reasoning, the model's correctness reward plateaus at 30% after 500 steps, but format reward is 100%. What are your next steps?

    **A:** Several failure modes are consistent with this pattern:
    
    1. **The format reward is too large relative to correctness.** If $\lambda_f = 0.5$ and correctness rewards are sparse (0 or 1), the model may have found a local optimum where format reward alone gives a good expected return. Try reducing $\lambda_f$ to 0.05-0.1 and restarting.
    
    2. **The problem distribution is too hard.** If the base model never produces correct answers during rollout (the oracle pass rate is near zero), there is no positive reward signal to latch onto. Solutions: use a warmer sampling temperature (e.g., 0.9 instead of 0.6) to increase exploration, curriculum with easier problems first, or start from a stronger SFT checkpoint.
    
    3. **The math verifier has bugs.** Check the false-negative rate of your verifier on known-correct answers. A buggy normalizer that rejects `\frac{1}{2}` when the gold is `0.5` creates a phantom ceiling. Log verifier inputs/outputs during training.
    
    4. **KL penalty is too strong.** If the KL coefficient against the reference policy is large, the policy cannot move far enough from the SFT model to find new correct solutions. Check the KL term in the loss; see [Advantage Estimation, KL Control & Stability Tricks](../06-rl-infra/09-advantage-kl-tricks.html).

## Reward Server Architecture

A reward server decouples reward computation from training, allowing:

- Independent scaling of reward workers (especially important when rewards involve expensive sandbox execution).
- Reward caching (identical (prompt, completion) pairs return cached results).
- Async reward computation to hide latency behind the next rollout batch.


{{fig:reward-server-arch}}


Here is a minimal FastAPI reward server:

```python
"""
reward_server.py — minimal HTTP reward server for RL training.

Deploy behind a load balancer; run multiple instances for throughput.
"""

import asyncio
import hashlib
import json
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from math_verifier import math_reward
from code_verifier import code_reward, SandboxPool
from llm_judge import batch_judge

app = FastAPI(title="Reward Server")

# Global sandbox pool — initialized once at startup
_sandbox_pool: Optional[SandboxPool] = None
# Simple in-memory cache (use Redis in production)
_cache: dict[str, float] = {}


@app.on_event("startup")
async def startup():
    global _sandbox_pool
    _sandbox_pool = SandboxPool(pool_size=16, timeout=10.0)


class RewardRequest(BaseModel):
    task_type: str          # "math", "code", "judge"
    prompt: str
    completion: str
    metadata: dict          # gold_answer, test_suite, reference, etc.


class RewardResponse(BaseModel):
    reward: float
    cached: bool
    breakdown: dict         # Per-component rewards for logging


def _cache_key(req: RewardRequest) -> str:
    raw = json.dumps({
        "type": req.task_type,
        "prompt": req.prompt,
        "completion": req.completion,
        "meta": req.metadata,
    }, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


@app.post("/reward", response_model=RewardResponse)
async def compute_reward(req: RewardRequest):
    key = _cache_key(req)
    if key in _cache:
        return RewardResponse(reward=_cache[key], cached=True, breakdown={})

    breakdown = {}
    total = 0.0

    if req.task_type == "math":
        gold = req.metadata.get("gold_answer", "")
        r = math_reward(req.prompt, req.completion, gold)
        breakdown["correctness"] = r
        total = r

    elif req.task_type == "code":
        tests = req.metadata.get("test_suite", "")
        r = await asyncio.get_event_loop().run_in_executor(
            None,  # Default thread pool
            lambda: code_reward(req.prompt, req.completion, tests)
        )
        breakdown["code_tests"] = r
        total = r

    elif req.task_type == "judge":
        reference = req.metadata.get("reference", "")
        rewards = await batch_judge(
            [req.prompt], [req.completion], [reference]
        )
        r = rewards[0]
        breakdown["llm_judge"] = r
        total = r

    else:
        raise HTTPException(400, f"Unknown task_type: {req.task_type}")

    _cache[key] = total
    return RewardResponse(reward=total, cached=False, breakdown=breakdown)


@app.post("/reward/batch")
async def compute_rewards_batch(requests: list[RewardRequest]):
    """Process a batch of reward requests concurrently."""
    tasks = [compute_reward(req) for req in requests]
    return await asyncio.gather(*tasks)
```

## Reward Hacking and Mitigation

Reward hacking occurs when the model finds a high-reward policy that does not correspond to the intended behavior. This is covered extensively in [Reward Hacking, Over-Optimization & Alignment Failures](../05-posttraining-alignment/13-reward-hacking-failures.html), but from an infrastructure perspective the key defenses are:

**Adversarial testing of your verifier.** Before training, run a fuzzer that generates edge-case strings (e.g., `\boxed{}`, `\boxed{\infty}`, extremely long LaTeX expressions, Unicode look-alikes for digits) and check that your verifier handles them correctly. A buggy verifier is itself a reward hacking surface.

**Hold-out test suites.** For code tasks, separate the visible test cases (used for reward) from hidden test cases (used for eval). This mirrors competitive programming practice. If the model achieves 90% on visible tests but 30% on hidden tests, it has overfit to the visible tests — reward hacking through test memorization.

**Reward variance monitoring.** Track $\text{Var}(r)$ within each prompt group over training. Healthy training shows decreasing variance as the model converges. A variance spike often indicates the model discovered a reward shortcut.

**KL-constrained reward.** Adding a KL penalty $-\beta \cdot D_{\text{KL}}(\pi \| \pi_{\text{ref}})$ to the reward limits how far the policy can deviate from the reference model. This is not a perfect defense, but it makes dramatic behavioral changes more expensive in reward terms. See [Advantage Estimation, KL Control & Stability Tricks](../06-rl-infra/09-advantage-kl-tricks.html).

## Putting It Together: A Complete Reward Pipeline

Here is how the components connect in a real RLVR system:

```python
"""
rl_reward_pipeline.py — end-to-end reward pipeline for a math RLVR run.

This orchestrates: rollout → reward computation → advantage normalization.
Intended to run on the CPU reward server, called from the training loop.
"""

import asyncio
import numpy as np
from typing import NamedTuple


class RolloutBatch(NamedTuple):
    prompts: list[str]            # One per prompt group
    completions: list[list[str]]  # completions[i] = list of G completions for prompt i
    gold_answers: list[str]       # Reference answers for verifier


class RewardBatch(NamedTuple):
    rewards: np.ndarray           # Shape (N,) flat — all completions
    advantages: np.ndarray        # Shape (N,) after group normalization
    component_log: dict           # For logging / dashboards


async def compute_rewards_for_batch(
    batch: RolloutBatch,
    format_lambda: float = 0.1,
) -> RewardBatch:
    """
    Main pipeline:
    1. Compute correctness reward (rule-based math verifier)
    2. Compute format reward
    3. Combine with weights
    4. Normalize per group to get advantages
    """
    from math_verifier import math_reward, extract_answer_from_completion

    flat_prompts = []
    flat_completions = []
    flat_golds = []
    group_ids = []  # Which group each completion belongs to

    for g, (prompt, comps, gold) in enumerate(
        zip(batch.prompts, batch.completions, batch.gold_answers)
    ):
        for comp in comps:
            flat_prompts.append(prompt)
            flat_completions.append(comp)
            flat_golds.append(gold)
            group_ids.append(g)

    n = len(flat_completions)
    correctness = np.zeros(n)
    format_r = np.zeros(n)

    for i, (prompt, comp, gold) in enumerate(
        zip(flat_prompts, flat_completions, flat_golds)
    ):
        # Format reward: 1 if \boxed{} is present, 0 otherwise
        has_boxed = bool(__import__("re").search(r'\\boxed\{', comp))
        format_r[i] = 1.0 if has_boxed else 0.0
        # Correctness reward
        correctness[i] = math_reward(prompt, comp, gold)

    combined = correctness + format_lambda * format_r

    # Normalize per group → advantages
    advantages = np.zeros(n)
    num_groups = len(batch.prompts)
    for g in range(num_groups):
        idx = [i for i, gid in enumerate(group_ids) if gid == g]
        group = combined[idx]
        mu, sigma = group.mean(), group.std()
        advantages[idx] = (group - mu) / (sigma + 1e-8)

    return RewardBatch(
        rewards=combined,
        advantages=advantages,
        component_log={
            "correctness": correctness.tolist(),
            "format": format_r.tolist(),
        },
    )
```

!!! key "Key Takeaways"
    - Rule-based verifiers (math equivalence, code unit tests) are the gold standard for RLVR: deterministic, free at inference time, and hard to hack compared to learned reward models.
    - Math verifiers must do symbolic/numeric normalization, not string equality. Sympy with a numeric fallback handles most competition math formats correctly.
    - Code verifiers need sandboxed execution to prevent reward hacking via filesystem or process manipulation. Warm container or microVM pools eliminate per-sample startup overhead.
    - LLM-judge rewards are necessary for open-ended tasks but introduce positional and verbosity bias; mitigate by swapping order, using cross-family judges, and normalizing within batches.
    - Format rewards (small $\lambda \approx 0.1$) solve the cold-start problem by encouraging structured output before correctness rewards become dense; too large a $\lambda$ invites gaming.
    - Multi-objective rewards should be combined on similar scales and normalized per prompt group (GRPO-style) to produce zero-mean advantages independent of absolute reward magnitude.
    - The reward server is a critical piece of RL infrastructure: decouple it from training, cache results, and size your sandbox pool to match rollout throughput (workers = throughput / per-worker capacity, plus 2-3x headroom).
    - Monitor reward hacking via verifier fuzzing, hold-out test suites, and within-group reward variance throughout training.

!!! sota "State of the Art & Resources (2026)"
    Reward engineering for RLVR has matured rapidly since DeepSeek-R1 demonstrated that rule-based verifiers alone—without any learned reward model—can drive state-of-the-art reasoning. The active frontier now spans more robust symbolic verifiers, step-level process reward models that scale test-time compute, and production-grade sandboxing infrastructure for safe code execution at training throughput.

    **Foundational work**

    - [Lightman et al., *Let's Verify Step by Step* (2023)](https://arxiv.org/abs/2305.20050) — introduces process supervision (PRMs) and the PRM800K dataset; the paper that established step-level reward modeling as a research area.
    - [Gao, Schulman & Hilton, *Scaling Laws for Reward Model Overoptimization* (2022)](https://arxiv.org/abs/2210.10760) — characterizes how reward model overoptimization degrades true performance; foundational analysis for KL-constrained reward design.

    **Recent advances (2023–2026)**

    - [DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* (2025)](https://arxiv.org/abs/2501.12948) — canonical RLVR recipe using rule-based math and format verifiers without any learned reward model.
    - [Wang et al., *Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations* (2023)](https://arxiv.org/abs/2312.08935) — automated PRM training via Monte Carlo rollouts, removing the need for human step-level labels.
    - [Khalifa et al., *Process Reward Models That Think* (ThinkPRM, 2025)](https://arxiv.org/abs/2504.16828) — PRMs that generate verification chain-of-thought, outperforming LLM-as-a-Judge with only 1% of the process labels.
    - [Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* (2023)](https://arxiv.org/abs/2306.05685) — systematic analysis of positional, verbosity, and self-preference biases in LLM judges; standard reference for calibrating judge rewards.

    **Open-source & tools**

    - [verl-project/verl](https://github.com/verl-project/verl) — flexible, high-throughput RL post-training framework (PPO, GRPO, DAPO) with pluggable reward functions; used to reproduce and extend DeepSeek-R1.
    - [e2b-dev/E2B](https://github.com/e2b-dev/E2B) — open-source Firecracker-backed cloud sandbox SDK for safely executing AI-generated code; Python and TypeScript APIs.
    - [openai/prm800k](https://github.com/openai/prm800k) — 800 K step-level correctness labels on MATH solutions plus the SymPy-based answer-grading logic from "Let's Verify Step by Step."
    - [opendilab/awesome-RLVR](https://github.com/opendilab/awesome-RLVR) — curated, actively updated reading list of RLVR papers, codebases, and tutorials (2024–2026).

    **Go deeper**

    - [Lilian Weng, *Reward Hacking in Reinforcement Learning* (Lil'Log, 2024)](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/) — comprehensive 37-minute survey of reward hacking failure modes in RL and LLMs, with mitigation strategies; essential reading before shipping a reward pipeline.

## Further Reading

- Lightman et al., *Let's Verify Step by Step* (OpenAI, 2023) — introduces process reward models (PRMs) for math reasoning and the PRM800K dataset.
- Wang et al., *Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations* (2023) — automated PRM training via Monte Carlo rollouts.
- DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* (2025) — the canonical RLVR recipe with format + correctness rewards.
- Chen et al., *Evaluating Large Language Models Trained on Code* (OpenAI, 2021) — introduces HumanEval and the pass@k metric for code reward evaluation.
- Guo et al., *Deepseek-Coder: When the Large Language Model Meets Programming* (2024) — discusses code-execution reward pipelines at scale.
- Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* (2023) — systematic analysis of LLM judge biases and calibration.
- Shen et al., *Loose lips sink ships: Mitigating Length Bias in Reinforcement Learning from Human Feedback* (2023) — reward shaping to control response length.
- `google/evals`, `openai/evals` — open-source eval harness libraries with reward function implementations.
