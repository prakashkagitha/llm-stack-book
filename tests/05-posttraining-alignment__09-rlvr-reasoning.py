"""
Runnability test for content/05-posttraining-alignment/09-rlvr-reasoning.md

Tests the 3 heuristically CPU-runnable Python blocks from the chapter,
concatenated in chapter order so later blocks can use names defined by
earlier ones:

    - block #0 (line ~84)  -- math-equivalence verifier: extract_boxed_answer,
                               normalize_numeric, math_is_correct (+ the
                               chapter's own inline sanity asserts, verbatim)
    - block #1 (line ~186) -- sandboxed code execution: _set_limits,
                               run_in_sandbox, code_reward, extract_code_block
    - block #3 (line ~290) -- the assembled rlvr_reward function that routes
                               to math_is_correct / code_reward, plus the
                               _is_degenerate anti-hacking guard

Block #2 (line ~267) is a Lean 4 theorem-prover snippet -- not Python, SKIP.
Block #4 (line ~378, `mixed_reward`) is a non-standalone fragment: it calls
`reward_model_score`, a function from an *earlier* chapter (5.5, the reward
model) that is never defined in this chapter -- SKIP(fragment): calling it
would just be testing a NameError, not the book's logic.

No network or third-party (non-stdlib/numpy/torch/einops/sklearn) dependency
is used anywhere in this chapter's runnable blocks: the code sandbox spawns a
local `python -I` subprocess (no network, temp-dir cwd, POSIX rlimits) to run
tiny model-generated-style programs -- this is exactly the mechanism the book
demonstrates, exercised offline and unmodified.
"""

import re
import sys

# =============================================================================
# Block #0 (line ~84): math equivalence verifier. Verbatim from the chapter,
# including its own inline sanity-check asserts.
# =============================================================================

from fractions import Fraction

def extract_boxed_answer(text: str) -> str | None:
    r"""
    Pull the LAST \boxed{...} content from a chain-of-thought response.
    We take the last one because the model often writes intermediate
    \boxed expressions before its final answer. Handles nested braces.
    """
    idx = text.rfind(r"\boxed")
    if idx == -1:
        # Fallback: try an <answer>...</answer> delimiter.
        m = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
        return m[-1].strip() if m else None
    # Walk braces to find the matching close for \boxed{ ... }.
    i = text.find("{", idx)
    if i == -1:
        return None
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j].strip()
        j += 1
    return None  # unbalanced braces

def normalize_numeric(s: str):
    """
    Try to coerce a string answer to an exact rational or a float.
    Handles fractions ('3/4'), LaTeX \frac, percentages, commas,
    surrounding $ signs, and trailing units-free numbers. Returns
    a Fraction/float, or None if it isn't a clean number.
    """
    if s is None:
        return None
    s = s.strip()
    s = s.replace("$", "").replace(",", "").replace("\\!", "").strip()
    s = re.sub(r"\\text\{.*?\}", "", s)              # drop \text{...} units
    s = s.replace("\\%", "").replace("%", "")        # treat percent as a number
    # LaTeX \frac{a}{b}  or  \dfrac{a}{b}
    m = re.fullmatch(r"\\d?frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if m:
        return Fraction(int(m.group(1)), int(m.group(2)))
    # plain a/b
    m = re.fullmatch(r"(-?\d+)\s*/\s*(-?\d+)", s)
    if m:
        return Fraction(int(m.group(1)), int(m.group(2)))
    try:
        return Fraction(s)            # exact for integers / decimals like '0.50'
    except (ValueError, ZeroDivisionError):
        pass
    try:
        return float(s)               # last resort, lossy
    except ValueError:
        return None

def math_is_correct(response: str, gold: str, atol: float = 1e-6) -> float:
    """
    Verifiable math reward: 1.0 if the model's final boxed answer is
    numerically equivalent to gold, else 0.0. Falls back to a
    normalized string compare for non-numeric answers (e.g. '(2, 3)').
    """
    pred = extract_boxed_answer(response)
    if pred is None:
        return 0.0
    a, b = normalize_numeric(pred), normalize_numeric(gold)
    if a is not None and b is not None:
        # Exact when both are Fractions; tolerant when a float is involved.
        if isinstance(a, Fraction) and isinstance(b, Fraction):
            return 1.0 if a == b else 0.0
        return 1.0 if abs(float(a) - float(b)) <= atol else 0.0
    # Non-numeric: compare normalized strings (whitespace/case-insensitive).
    norm = lambda x: re.sub(r"\s+", "", x).lower()
    return 1.0 if norm(pred) == norm(gold) else 0.0

# --- quick sanity checks (these all return 1.0) ---
assert math_is_correct(r"... so the answer is \boxed{1/2}.", "0.5") == 1.0
assert math_is_correct(r"first \boxed{7} then \boxed{0.50}", "1/2") == 1.0
assert math_is_correct(r"<answer>42</answer>", "42") == 1.0
assert math_is_correct(r"\boxed{\frac{3}{4}}", "0.75") == 1.0
assert math_is_correct(r"\boxed{8}", "9") == 0.0

print("[block #0] math verifier sanity asserts passed")


# =============================================================================
# Block #1 (line ~186): sandboxed code execution. Verbatim from the chapter.
# Runs a local `python -I` subprocess -- no network, temp-dir cwd, POSIX
# rlimits -- exactly the mechanism the book demonstrates.
# =============================================================================

import subprocess, tempfile, os, textwrap, resource, json

def _set_limits():
    """Called in the child via preexec_fn: cap CPU, memory, and file size."""
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))            # 5s CPU time
    mem = 512 * 1024 * 1024                                    # 512 MB
    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))         # address space
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))  # 1 MB files
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))       # cap fork-bombs

def run_in_sandbox(source_code: str, stdin: str = "", timeout: float = 6.0):
    """
    Execute untrusted Python in an isolated subprocess with rlimits and a
    wall-clock timeout. Returns (ok, stdout, stderr). Network is NOT blocked
    here (do that with a namespace/seccomp in production); we run with a
    minimal env and a temp CWD so there is nothing useful to touch.
    """
    with tempfile.TemporaryDirectory() as workdir:
        path = os.path.join(workdir, "prog.py")
        with open(path, "w") as f:
            f.write(source_code)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", path],   # -I = isolated mode (ignore env/PYTHONPATH)
                input=stdin.encode(),
                capture_output=True,
                timeout=timeout,                # wall-clock kill
                cwd=workdir,                    # sandboxed working dir
                preexec_fn=_set_limits,         # apply rlimits in child (POSIX)
                env={"PATH": "/usr/bin", "OPENBLAS_NUM_THREADS": "1"},
            )
            return (proc.returncode == 0, proc.stdout.decode(errors="replace"),
                    proc.stderr.decode(errors="replace"))
        except subprocess.TimeoutExpired:
            return (False, "", "TIMEOUT")

def code_reward(completion: str, test_cases: list[dict],
                entry_point: str = "solve") -> float:
    """
    Verifiable code reward = fraction of hidden unit tests passed.
    `test_cases` is a list of {"input": "...", "expected": "..."} dicts.
    The model's `completion` is expected to define a function `entry_point`
    that reads from stdin and prints to stdout. We assemble a harness so the
    model's code NEVER sees the test inputs as data it can inspect.
    """
    program = extract_code_block(completion)
    if program is None:
        return 0.0
    passed = 0
    for tc in test_cases:
        # Harness runs the model code, then calls it on this test's stdin.
        harness = program + "\n\nif __name__ == '__main__':\n    " + entry_point + "()\n"
        ok, out, err = run_in_sandbox(harness, stdin=tc["input"])
        if ok and out.strip() == tc["expected"].strip():
            passed += 1
    return passed / len(test_cases)   # graded reward in [0, 1]

def extract_code_block(text: str) -> str | None:
    """Pull the last ```python ... ``` fenced block (the final solution)."""
    import re
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL)
    return blocks[-1].strip() if blocks else None

# --- exercise the sandbox on a tiny "model completion": an echo-sum program.
_sample_completion = (
    "Here is my solution:\n"
    "```python\n"
    "def solve():\n"
    "    a, b = map(int, input().split())\n"
    "    print(a + b)\n"
    "```\n"
)
_test_cases = [
    {"input": "2 3", "expected": "5"},
    {"input": "10 -4", "expected": "6"},
    {"input": "0 0", "expected": "1"},   # deliberately wrong expected -> should fail
]
_score = code_reward(_sample_completion, _test_cases)
assert 0.0 < _score < 1.0, f"expected a graded (partial) score, got {_score}"
# exactly 2 of 3 test cases should pass (the third has a wrong 'expected')
assert abs(_score - (2 / 3)) < 1e-9, f"expected 2/3 pass rate, got {_score}"
print(f"[block #1] code_reward sandbox executed, graded score = {_score:.3f}")

# also sanity-check the timeout / crash-safety path with a program that raises.
_bad_completion = "```python\ndef solve():\n    raise RuntimeError('boom')\n```\n"
_bad_score = code_reward(_bad_completion, _test_cases)
assert _bad_score == 0.0
print("[block #1] code_reward correctly scores a crashing program as 0.0")


# =============================================================================
# Block #3 (line ~290): the assembled RLVR reward function. Verbatim from
# the chapter, built on top of math_is_correct / code_reward /
# extract_boxed_answer defined in blocks #0 and #1 above.
# =============================================================================

def rlvr_reward(question: str, response: str, gold: str,
                domain: str, test_cases=None) -> dict:
    """
    Full RLVR reward, returned as a breakdown so we can log each component.
    Correctness is the real signal (weight 1.0). Format is a small shaping
    bonus that is ONLY granted if the model also attempted a parseable answer,
    so it can never be farmed independently of trying to solve the task.
    """
    # 1. Correctness (the only component we truly trust).
    if domain == "math":
        accuracy = math_is_correct(response, gold)          # {0, 1}
    elif domain == "code":
        accuracy = code_reward(response, test_cases)         # [0, 1] graded
    else:
        accuracy = 0.0

    # 2. Format shaping (tiny, and CONTINGENT on a parseable answer existing).
    has_think = "<think>" in response and "</think>" in response
    has_answer = extract_boxed_answer(response) is not None
    format_bonus = 0.1 if (has_think and has_answer) else 0.0

    # 3. Anti-hacking guard: zero out everything if the response is degenerate
    #    (e.g. empty, or repeats one token -- catches a known length-hack mode).
    if _is_degenerate(response):
        return {"accuracy": 0.0, "format": 0.0, "total": 0.0}

    total = accuracy + format_bonus
    return {"accuracy": accuracy, "format": format_bonus, "total": total}

def _is_degenerate(text: str) -> bool:
    toks = text.split()
    if len(toks) < 3:
        return True
    # crude repetition check: >60% of tokens are the single most common token
    from collections import Counter
    most = Counter(toks).most_common(1)[0][1]
    return most / len(toks) > 0.6

# --- exercise rlvr_reward across the worked-example-style cases in the book ---

# o_1: correct, with <think> tags -> accuracy 1.0, format 0.1, total 1.1
r1 = rlvr_reward(
    "Compute the integral.",
    "<think>3x^2 integrates to x^3, evaluated 0 to 1 gives 1.</think> "
    r"so the answer is \boxed{1}.",
    gold="1", domain="math",
)
assert r1 == {"accuracy": 1.0, "format": 0.1, "total": 1.1}, r1

# o_2: correct, no think tags -> accuracy 1.0, format 0.0, total 1.0
r2 = rlvr_reward("Compute the integral.", r"the answer is \boxed{1}.",
                  gold="1", domain="math")
assert r2 == {"accuracy": 1.0, "format": 0.0, "total": 1.0}, r2

# o_3: wrong answer, with think tags -> accuracy 0.0, format 0.1, total 0.1
r3 = rlvr_reward("Compute the integral.",
                  r"<think>forgot to evaluate</think> \boxed{x^3}",
                  gold="1", domain="math")
assert r3 == {"accuracy": 0.0, "format": 0.1, "total": 0.1}, r3

# o_6: degenerate response (repeats one token) -> everything zeroed
r6 = rlvr_reward("Compute the integral.", "the the the the the the the the",
                  gold="1", domain="math")
assert r6 == {"accuracy": 0.0, "format": 0.0, "total": 0.0}, r6

print("[block #3] rlvr_reward (math domain) matches the book's worked example:",
      r1, r2, r3, r6)

# also route through the code domain to exercise the code_reward branch.
rc = rlvr_reward(
    "Sum two numbers.", _sample_completion, gold="", domain="code",
    test_cases=_test_cases,
)
assert abs(rc["accuracy"] - (2 / 3)) < 1e-9
assert rc["format"] == 0.0   # no <think>/boxed tags in the code completion
print(f"[block #3] rlvr_reward (code domain) = {rc}")

print("\nAll RLVR reasoning-chapter blocks executed successfully.")
