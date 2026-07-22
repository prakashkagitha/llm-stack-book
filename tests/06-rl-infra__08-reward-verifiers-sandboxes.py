"""
Executable extraction of the CPU-runnable code blocks from
content/06-rl-infra/08-reward-verifiers-sandboxes.md

Blocks tested (chapter's own numbering):
  #0 (line ~37,  148 lines) - math_verifier.py: normalize_latex_number,
      try_numeric, try_sympy, math_equivalent, extract_answer_from_completion,
      math_reward
  #1 (line ~194, 120 lines) - code_verifier.py: run_tests_in_subprocess,
      code_reward (uses a real subprocess, no docker)
  #2 (line ~337, 123 lines) - sandbox_pool.py: SandboxWorker, SandboxPool.
      The book wraps sandboxes in `docker run ...`; we don't require a live
      Docker daemon/image pull (that would be a hidden network dependency in
      CI), so we monkeypatch only `SandboxPool._spawn` to launch the exact
      same `SANDBOX_RUNNER` script directly via `sys.executable` instead of
      through `docker run`. Every other line of the pool's logic (queue,
      locking, JSON stdin/stdout protocol, timeout handling) is the book's
      real code, exercised against a real subprocess.
  #5 (line ~757, 78 lines)  - reward_combiner.py: RewardComponent,
      MultiObjectiveReward (weighting + per-group normalization)
  #6 (line ~864, 102 lines) - reward_server.py: FastAPI reward server.
      `fastapi` is not in the guaranteed CI import set, so the import is
      guarded; if unavailable the block is defined but not exercised
      (see SKIP note below). math_verifier/code_verifier/llm_judge imports
      are dropped since those names already live in this same concatenated
      module (per the harness's own "later blocks depend on earlier blocks"
      convention).
  #7 (line ~984, 84 lines)  - rl_reward_pipeline.py: RolloutBatch,
      RewardBatch, compute_rewards_for_batch (full rollout -> reward ->
      advantage pipeline)

Blocks explicitly SKIPPED:
  # SKIP(needs-net): block #3 (line ~492, llm_judge.py) calls a real OpenAI
    API (`openai.AsyncOpenAI()...chat.completions.create`). Defined here
    with `openai` import guarded (module stays importable without the
    `openai` package) but never invoked — no network calls are made.
  # SKIP(needs-gpu): block #4 (line ~639, prm_reward.py) loads a real HF
    checkpoint via `AutoModelForSequenceClassification.from_pretrained(...)`
    and defaults to `device="cuda"`. `transformers`/`torch` model download
    + GPU are both out of scope for a CPU-only smoke test; import guarded.
  # SKIP(missing-dependency): within block #6, the actual FastAPI app
    construction/route execution only runs `if _HAS_FASTAPI` — `fastapi` is
    not in the CI-guaranteed import list (numpy/torch/einops/sklearn/stdlib
    only) and is not installed in this environment, so that portion is
    defined-but-not-called and the pure-Python parts (`_cache_key`) are
    exercised directly against a lightweight duck-typed stand-in instead of
    a pydantic `BaseModel`.

Real bugs found in the book's code and fixed in both the .md and here:
  1. code_verifier.py's `run_tests_in_subprocess` built the generated
     runner script with textwrap.dedent() wrapping an f-string whose body
     embedded textwrap.indent(test_suite, ...) mid-line.
     `dedent()` computes a single common leading-whitespace prefix over the
     *whole* string, but only the first line of an embedded multi-line
     f-string value inherits the template line's own indentation —
     subsequent lines of `test_suite` don't — so the inserted test
     function(s) end up misaligned relative to the surrounding `try:`
     block. Confirmed with a standalone repro: this raised a real
     `IndentationError` for a plain one-test-function suite, meaning the
     block as printed in the book cannot run at all. Fixed by building the
     template flush-left and explicitly indenting the embedded block with
     `textwrap.indent(test_suite, "    ")` before insertion.
  2. code_verifier.py's `run_tests_in_subprocess` invoked the bare `"python"`
     executable (`subprocess.run(["python", runner_path], ...)`). Many
     Linux/CI images only provide `python3` on PATH, so this call raises
     FileNotFoundError there. Fixed to use `sys.executable`.
  3. sandbox_pool.py's `SandboxPool.execute` called
     `worker.proc.stdout._sock.settimeout(self.timeout)` to bound the read
     from a sandbox worker. `subprocess.Popen(..., stdout=PIPE, text=True)`
     returns a `TextIOWrapper` over an OS pipe, which has no `_sock`
     attribute at all (that's a socket-specific API) — this line always
     raises `AttributeError` on every call, meaning the pool as written
     never returns a real result; it silently kills and respawns the worker
     on every single `execute()`. Confirmed with a standalone repro
     (`hasattr(proc.stdout, "_sock")` is `False`). Fixed by using
     `select.select([worker.proc.stdout], [], [], self.timeout)` to bound
     the wait for a response line, which works correctly on OS pipes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import queue
import re
import select
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, NamedTuple, Optional, Union

import numpy as np

# --- Guarded optional third-party imports (not in the CI-guaranteed set) ---

try:
    import sympy
    from sympy.parsing.latex import parse_latex
    from sympy import simplify, N
except Exception:
    sympy = None
    parse_latex = None
    simplify = None
    N = None

try:
    import openai  # noqa: F401  (block #3 - defined but never called, SKIP(needs-net))
except Exception:
    openai = None

try:
    import torch  # noqa: F401
    from transformers import AutoTokenizer, AutoModelForSequenceClassification  # noqa: F401
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    _HAS_FASTAPI = True
except Exception:
    FastAPI = None
    HTTPException = None
    BaseModel = None
    _HAS_FASTAPI = False


# =====================================================================
# Block #0 (line ~37) - math_verifier.py
# =====================================================================

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


def try_sympy(s: str) -> Optional["sympy.Expr"]:
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
    but the answer is wrong, to encourage the model to use \\boxed{}.
    """
    pred = extract_answer_from_completion(completion)
    if not pred:
        return 0.0  # No answer extracted

    if math_equivalent(pred, gold_answer):
        return 1.0
    else:
        # Small reward for correct format, zero for content
        return 0.1  # format reward (optional — see section on reward shaping)


def test_block_0_math_verifier():
    # Exact numeric match via different textual encodings
    assert math_equivalent("0.5", "1/2") is True
    assert math_equivalent("50%", "0.5") is True
    assert math_equivalent(r"\frac{1}{2}", "0.5") in (True, False)  # sympy-parse-dependent, must not raise
    assert math_equivalent("3.0", "3") is True
    assert math_equivalent("2", "3") is False

    completion_ok = r"We compute step by step. The final answer is \boxed{1/2}."
    assert extract_answer_from_completion(completion_ok) == "1/2"
    assert math_reward("What is 1/2?", completion_ok, "0.5") == 1.0

    completion_wrong = r"\boxed{3}"
    assert math_reward("What is 1/2?", completion_wrong, "0.5") == 0.1

    completion_no_answer = "I don't know."
    assert math_reward("What is 1/2?", completion_no_answer, "0.5") == 0.0

    if sympy is not None and parse_latex is not None:
        # Only exercised when sympy's LaTeX parser (antlr4) actually works
        try:
            parse_latex(r"\frac{3}{4}")
            has_latex_backend = True
        except Exception:
            has_latex_backend = False
        if has_latex_backend:
            assert math_equivalent(r"\frac{3}{4}", "0.75") is True

    print("[block #0] math_verifier.py: OK")


# =====================================================================
# Block #1 (line ~194) - code_verifier.py
# =====================================================================

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

        # Write test runner that imports solution.py.
        # BUGFIX: the book originally built this with
        # `textwrap.dedent(f"""...{textwrap.indent(test_suite, ...)}...""")`.
        # dedent() computes one common leading-whitespace prefix over the
        # *entire* string, but only the first line of an embedded
        # multi-line f-string value inherits the template's own
        # indentation — later lines of test_suite don't — so the inserted
        # test functions ended up misaligned relative to the surrounding
        # `try:` block and raised IndentationError (confirmed via repro).
        # Fix: build the template flush-left and indent the test_suite
        # block explicitly.
        indented_tests = textwrap.indent(test_suite, "    ")
        runner = f'''import sys, json, traceback
sys.path.insert(0, {repr(tmpdir)})

results = {{"passed": 0, "total": 0, "error": ""}}
try:
    from solution import *
{indented_tests}
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
'''
        runner_path = os.path.join(tmpdir, "runner.py")
        with open(runner_path, "w") as f:
            f.write(runner)

        try:
            proc = subprocess.run(
                # BUGFIX (book used bare "python", which is absent from PATH
                # on many Linux/CI images that only ship "python3"):
                [sys.executable, runner_path],
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


def test_block_1_code_verifier():
    completion = """```python
def add(a, b):
    return a + b
```"""
    test_suite = "def test_add():\n    assert add(2, 3) == 5\n"
    r = code_reward("Write add(a, b)", completion, test_suite)
    assert r == 1.0, f"expected 1.0, got {r}"

    # Partial credit: one of two tests fails
    completion2 = """```python
def add(a, b):
    return a + b
```"""
    test_suite2 = (
        "def test_add_ok():\n    assert add(2, 3) == 5\n"
        "def test_add_bad():\n    assert add(2, 3) == 999\n"
    )
    r2 = code_reward("Write add(a, b)", completion2, test_suite2)
    assert abs(r2 - 0.5) < 1e-9, f"expected 0.5, got {r2}"

    # Syntax error in generated code -> 0.0
    bad_completion = "```python\ndef add(a, b)\n    return a + b\n```"
    r3 = code_reward("Write add(a, b)", bad_completion, test_suite)
    assert r3 == 0.0, f"expected 0.0, got {r3}"

    print("[block #1] code_verifier.py: OK")


# =====================================================================
# Block #2 (line ~337) - sandbox_pool.py
# =====================================================================

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
            # BUGFIX (book used `worker.proc.stdout._sock.settimeout(...)`:
            # a subprocess.Popen(text=True) pipe is a TextIOWrapper over an
            # OS pipe, which has no `_sock` attribute at all — that line
            # always raised AttributeError, silently killing/respawning the
            # worker on every call. select() works correctly on pipe fds.
            ready, _, _ = select.select([worker.proc.stdout], [], [], self.timeout)
            if not ready:
                raise TimeoutError(f"sandbox worker timed out after {self.timeout}s")
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


def test_block_2_sandbox_pool():
    # We don't require a live Docker daemon/image pull in a CPU-only CI
    # test (that would be a hidden network dependency). We monkeypatch
    # only `_spawn` to launch the *exact same* SANDBOX_RUNNER script
    # directly with sys.executable instead of wrapping it in `docker run`.
    # Every other line of SandboxPool (queue, locking, JSON stdin/stdout
    # protocol, select()-based timeout) is the book's real, unmodified code.
    class DirectSandboxPool(SandboxPool):
        def _spawn(self) -> SandboxWorker:
            proc = subprocess.Popen(
                [sys.executable, "-c", SANDBOX_RUNNER],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            return SandboxWorker(proc=proc)

    pool = DirectSandboxPool(pool_size=2, timeout=5.0)
    try:
        result = pool.execute(
            code="def add(a, b):\n    return a + b\n",
            tests="def test_add():\n    assert add(2, 3) == 5\n",
        )
        assert result == {"passed": 1, "total": 1, "error": ""}, result

        # A second call reuses a warm worker from the pool.
        result2 = pool.execute(
            code="def add(a, b):\n    return a + b\n",
            tests="def test_add_fail():\n    assert add(2, 3) == 999\n",
        )
        assert result2 == {"passed": 0, "total": 1, "error": ""}, result2
    finally:
        for w in pool._all_workers:
            try:
                w.proc.kill()
            except Exception:
                pass

    print("[block #2] sandbox_pool.py: OK (docker replaced by direct subprocess spawn)")


# =====================================================================
# Block #3 (line ~492) - llm_judge.py — SKIP(needs-net)
# Defined so later blocks (#6) can reference `batch_judge` by name, but
# never invoked: it would make a real OpenAI network call.
# =====================================================================

JUDGE_SYSTEM = """
You are an expert evaluator for mathematical reasoning and problem solving.
You will be given a problem, a reference solution, and a model response.
Your task is to evaluate the model response on a scale of 0 to 10.
""".strip()

JUDGE_USER_TEMPLATE = """
PROBLEM:
{problem}

REFERENCE SOLUTION:
{reference}

MODEL RESPONSE:
{response}
""".strip()


async def judge_single(
    problem: str,
    response: str,
    reference: str,
    judge_model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> float:
    """SKIP(needs-net): real network call to OpenAI. Not invoked in this test."""
    client = openai.AsyncOpenAI()
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
            problem=problem, reference=reference, response=response,
        )},
    ]
    resp = await client.chat.completions.create(
        model=judge_model, messages=messages, temperature=temperature, max_tokens=512,
    )
    text = resp.choices[0].message.content
    m = re.search(r"<score>(\d+(?:\.\d+)?)</score>", text)
    if m:
        return min(max(float(m.group(1)) / 10.0, 0.0), 1.0)
    nums = re.findall(r"\b(\d+)\b", text)
    if nums:
        return min(max(float(nums[-1]) / 10.0, 0.0), 1.0)
    return 0.5


async def batch_judge(
    problems: list,
    responses: list,
    references: list,
    judge_model: str = "gpt-4o-mini",
    concurrency: int = 32,
) -> list:
    """SKIP(needs-net): real network calls to OpenAI. Not invoked in this test."""
    sem = asyncio.Semaphore(concurrency)

    async def bounded_judge(p, r, ref):
        async with sem:
            return await judge_single(p, r, ref, judge_model)

    tasks = [bounded_judge(p, r, ref) for p, r, ref in zip(problems, responses, references)]
    return await asyncio.gather(*tasks)


# =====================================================================
# Block #4 (line ~639) - prm_reward.py — SKIP(needs-gpu)
# Defined only to keep the chapter's code inventory complete; never
# instantiated (requires downloading + running a real HF checkpoint).
# =====================================================================

if AutoTokenizer is not None and torch is not None:
    class ProcessRewardModel:
        """SKIP(needs-gpu): not instantiated — requires a real HF checkpoint."""

        def __init__(self, model_name: str, device: str = "cuda"):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=1,
            ).to(device).eval()
            self.device = device

        @torch.no_grad()
        def score_steps(self, problem: str, steps: list, batch_size: int = 32) -> list:
            scores = []
            contexts = []
            for i in range(len(steps)):
                partial = problem + "\n\n" + "\n".join(steps[:i + 1])
                contexts.append(partial)
            for start in range(0, len(contexts), batch_size):
                batch = contexts[start:start + batch_size]
                enc = self.tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=True, max_length=2048,
                ).to(self.device)
                logits = self.model(**enc).logits.squeeze(-1)
                probs = torch.sigmoid(logits).cpu().tolist()
                if isinstance(probs, float):
                    probs = [probs]
                scores.extend(probs)
            return scores

        def aggregate(self, step_scores: list, method: str = "min") -> float:
            if not step_scores:
                return 0.0
            if method == "min":
                return min(step_scores)
            elif method == "last":
                return step_scores[-1]
            elif method == "mean":
                return sum(step_scores) / len(step_scores)
            raise ValueError(f"Unknown aggregation method: {method}")


# =====================================================================
# Block #5 (line ~757) - reward_combiner.py
# =====================================================================

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

    def __init__(self, components: list):
        self.components = components

    def compute(
        self,
        prompts: list,
        completions: list,
        metadata: list,
        normalize_per_group: bool = True,
    ):
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


def test_block_5_reward_combiner():
    def correctness_fn(prompt, completion, meta):
        return 1.0 if completion.strip() == meta["gold"] else 0.0

    def format_fn(prompt, completion, meta):
        return 1.0 if completion.startswith("<think>") else 0.0

    components = [
        RewardComponent(name="correctness", fn=correctness_fn, weight=1.0),
        RewardComponent(name="format", fn=format_fn, weight=0.1),
    ]
    combiner = MultiObjectiveReward(components)

    prompts = ["2+2?", "2+2?", "3+3?", "3+3?"]
    completions = [
        "<think>ok</think>4",   # format ok, wrong content (gold is "4" exactly)
        "4",                    # correct content, no format
        "<think>ok</think>6",   # both correct
        "5",                    # both wrong
    ]
    metadata = [
        {"gold": "4"}, {"gold": "4"},
        {"gold": "<think>ok</think>6"}, {"gold": "6"},
    ]

    totals, breakdown = combiner.compute(prompts, completions, metadata, normalize_per_group=True)
    assert len(totals) == 4
    assert set(breakdown.keys()) == {"correctness", "format"}
    # Per-group z-score normalization -> each group of 2 sums to ~0
    assert abs(totals[0] + totals[1]) < 1e-6
    assert abs(totals[2] + totals[3]) < 1e-6

    totals_raw, _ = combiner.compute(prompts, completions, metadata, normalize_per_group=False)
    # completion[2] matches gold exactly (correctness=1) and starts with <think> (format=1)
    assert abs(totals_raw[2] - 1.1) < 1e-9

    print("[block #5] reward_combiner.py: OK")


# =====================================================================
# Block #6 (line ~864) - reward_server.py
# =====================================================================

def _cache_key(req) -> str:
    raw = json.dumps({
        "type": req.task_type,
        "prompt": req.prompt,
        "completion": req.completion,
        "meta": req.metadata,
    }, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


if _HAS_FASTAPI:
    app = FastAPI(title="Reward Server")
    _sandbox_pool = None
    _cache: dict = {}

    class RewardRequest(BaseModel):
        task_type: str
        prompt: str
        completion: str
        metadata: dict

    class RewardResponse(BaseModel):
        reward: float
        cached: bool
        breakdown: dict

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
                None, lambda: code_reward(req.prompt, req.completion, tests)
            )
            breakdown["code_tests"] = r
            total = r
        else:
            raise HTTPException(400, f"Unknown task_type: {req.task_type}")

        _cache[key] = total
        return RewardResponse(reward=total, cached=False, breakdown=breakdown)


def test_block_6_reward_server():
    # Pure-Python part of the block (_cache_key) is duck-typeable and does
    # not need fastapi/pydantic at all.
    class FakeReq:
        def __init__(self, task_type, prompt, completion, metadata):
            self.task_type = task_type
            self.prompt = prompt
            self.completion = completion
            self.metadata = metadata

    r1 = FakeReq("math", "2+2?", r"\boxed{4}", {"gold_answer": "4"})
    r2 = FakeReq("math", "2+2?", r"\boxed{4}", {"gold_answer": "4"})
    r3 = FakeReq("math", "2+2?", r"\boxed{5}", {"gold_answer": "4"})
    assert _cache_key(r1) == _cache_key(r2)
    assert _cache_key(r1) != _cache_key(r3)

    if not _HAS_FASTAPI:
        print("[block #6] reward_server.py: SKIP(missing-dependency) — "
              "fastapi not installed / not in the CI-guaranteed import "
              "list; only the dependency-free _cache_key logic was "
              "exercised.")
        return

    async def _run():
        req = RewardRequest(
            task_type="math", prompt="2+2?", completion=r"\boxed{4}",
            metadata={"gold_answer": "4"},
        )
        resp = await compute_reward(req)
        assert resp.reward == 1.0
        assert resp.cached is False
        resp2 = await compute_reward(req)
        assert resp2.cached is True

    asyncio.run(_run())
    print("[block #6] reward_server.py: OK")


# =====================================================================
# Block #7 (line ~984) - rl_reward_pipeline.py
# =====================================================================

class RolloutBatch(NamedTuple):
    prompts: list            # One per prompt group
    completions: list        # completions[i] = list of G completions for prompt i
    gold_answers: list       # Reference answers for verifier


class RewardBatch(NamedTuple):
    rewards: np.ndarray      # Shape (N,) flat — all completions
    advantages: np.ndarray   # Shape (N,) after group normalization
    component_log: dict      # For logging / dashboards


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
    # (math_reward, extract_answer_from_completion already defined above
    # in this concatenated module — block #0's math_verifier.py)

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


def test_block_7_rl_reward_pipeline():
    batch = RolloutBatch(
        prompts=["2+2?", "3+3?"],
        completions=[
            # prompt 0: correct+fmt, correct-but-no-boxed (unextractable ->
            # 0.0), wrong+fmt
            [r"\boxed{4}", "4", r"\boxed{5}"],
            [r"\boxed{6}", r"\boxed{7}"],        # prompt 1: correct+fmt, wrong+fmt
        ],
        gold_answers=["4", "6"],
    )

    result = asyncio.run(compute_rewards_for_batch(batch, format_lambda=0.1))
    assert isinstance(result, RewardBatch)
    assert result.rewards.shape == (5,)
    assert result.advantages.shape == (5,)

    # rewards: [1.1, 0.0, 0.2,  1.1, 0.2]  (correctness + 0.1*format).
    # Note completion[1] = "4" has no \boxed{}/<answer> marker, so
    # extract_answer_from_completion() returns "" and math_reward is 0.0
    # even though "4" is the correct value — this is the book's own
    # cold-start/format-reward argument in action, not a bug.
    expected_rewards = np.array([1.1, 0.0, 0.2, 1.1, 0.2])
    assert np.allclose(result.rewards, expected_rewards), result.rewards

    # per-group z-scored advantages sum to ~0 within each group
    assert abs(result.advantages[0:3].sum()) < 1e-6
    assert abs(result.advantages[3:5].sum()) < 1e-6

    print("[block #7] rl_reward_pipeline.py: OK")


# =====================================================================
# Run all tests
# =====================================================================

if __name__ == "__main__":
    test_block_0_math_verifier()
    test_block_1_code_verifier()
    test_block_2_sandbox_pool()
    test_block_5_reward_combiner()
    test_block_6_reward_server()
    test_block_7_rl_reward_pipeline()
    print("\nAll CPU-runnable blocks in 06-rl-infra/08-reward-verifiers-sandboxes.md executed successfully.")
