"""CI-tested extracts of content/08-agents-harness/03-harness-coding-agent.md

Runs the chapter's CPU-runnable Python blocks verbatim and exercises each one
(functions called, classes instantiated). Blocks needing a live API / GPU /
extra framework are skipped with an explicit SKIP(...) reason.

Inventory (block idx -> disposition):
  0  ci  ToolResult dataclass + edit_file (constrained string-replace edit) -> TESTED
  1  ci  read_file() line-numbered read with offset/limit windowing + truncation
         note -> TESTED
  2  net SKIP(network): agent_loop() constructs `Anthropic()` and calls
         `client.messages.create(...)` -- a real LLM API call. We substitute a
         deterministic offline fake (`agent_loop`) at the call boundary so that
         block #7's own orchestration logic (run_coding_agent) still executes.
  3  ci  DENY/ALLOW regex classify() + gated_execute() permission gate -> TESTED
  4  ci  run_bash() sandboxed subprocess wrapper (timeout, output cap)   -> TESTED
  5  ci  TODO_STATE + todo_write() externalized plan tracker            -> TESTED
  6  ci  verify_before_done() ground-truth verification gate            -> TESTED
  7  ci  build_system_prompt() + run_coding_agent() orchestration       -> TESTED
         (run_coding_agent's call to agent_loop is satisfied by the block #2
         offline fake described above; verify_before_done's hardcoded default
         `pytest -q` is resolved through a PATH-shimmed stub script rather than
         invoking a real pytest run, since pytest is not a guaranteed CI dep and
         we must not let a bare `pytest -q` in an arbitrary cwd recurse into the
         book's own test suite.)
"""

import os
import re
import shlex
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Block #0 (line ~61, 42 lines) -- ToolResult + edit_file, verbatim from chapter
# ---------------------------------------------------------------------------
@dataclass
class ToolResult:
    """Uniform return type for every tool. `is_error` flips the model into
    repair mode; `content` is what gets serialized back into the context."""
    content: str
    is_error: bool = False

def edit_file(path: str, old_string: str, new_string: str) -> ToolResult:
    """Replace an exact, unique occurrence of `old_string` with `new_string`.

    Design contract (this is the whole point of the tool):
      1. The file must already have been read this session (enforced elsewhere).
      2. `old_string` must occur EXACTLY ONCE. Zero -> the model hallucinated
         the context. Two+ -> the edit is ambiguous. Both are hard errors the
         model must fix, NOT something we silently guess at.
    """
    p = Path(path)
    if not p.exists():
        return ToolResult(f"Error: {path} does not exist.", is_error=True)

    text = p.read_text()
    count = text.count(old_string)
    if count == 0:
        return ToolResult(
            f"Error: old_string not found in {path}. Re-read the file; the "
            f"text you supplied does not match the current contents.",
            is_error=True,
        )
    if count > 1:
        return ToolResult(
            f"Error: old_string appears {count} times in {path}. Provide more "
            f"surrounding context so the match is unique.",
            is_error=True,
        )

    p.write_text(text.replace(old_string, new_string))
    # Return a tiny confirmation, NOT the whole file: save context tokens.
    return ToolResult(f"Edited {path}: 1 replacement.")


# ---------------------------------------------------------------------------
# Block #1 (line ~108, 15 lines) -- read_file, verbatim from chapter
# ---------------------------------------------------------------------------
def read_file(path: str, offset: int = 0, limit: int = 2000) -> ToolResult:
    """Read a file with 1-based line numbers, like `cat -n`. Line numbers are
    not cosmetic: they give the model a coordinate system to reason about
    spans, jump to errors from a traceback, and describe edits precisely."""
    p = Path(path)
    if not p.exists():
        return ToolResult(f"Error: {path} does not exist.", is_error=True)
    lines = p.read_text().splitlines()
    window = lines[offset : offset + limit]
    numbered = "\n".join(f"{offset + i + 1:6d}\t{ln}"
                         for i, ln in enumerate(window))
    truncated = "" if offset + limit >= len(lines) else \
        f"\n... ({len(lines) - offset - limit} more lines; use offset to continue)"
    return ToolResult(numbered + truncated)


# ---------------------------------------------------------------------------
# Block #3 (line ~268, 30 lines) -- permission gate, verbatim from chapter
# ---------------------------------------------------------------------------
# Patterns that are NEVER run without explicit human confirmation.
DENY = [r"\brm\s+-rf\b", r"\bgit\s+push\b", r"\bcurl\b.*\|\s*sh\b",
        r"\bsudo\b", r":\(\)\s*\{", r"\bdd\b\s+if="]
# Read-only commands that are safe to auto-allow.
ALLOW = [r"^ls\b", r"^cat\b", r"^grep\b", r"^pytest\b", r"^git status\b",
         r"^git diff\b", r"^python -m pytest\b"]

def classify(cmd: str) -> str:
    if any(re.search(p, cmd) for p in DENY):
        return "deny"                       # hard block, or require confirm
    if any(re.match(p, cmd.strip()) for p in ALLOW):
        return "allow"
    return "ask"                            # unknown -> human in the loop

def gated_execute(name, args, fn):
    """Wrap every tool call in the permission gate."""
    if name == "bash":
        decision = classify(args["command"])
        if decision == "deny":
            return ToolResult(
                f"Blocked: '{args['command']}' matches a denied pattern. "
                f"Explain why you need it and propose a safer alternative.",
                is_error=True)
        if decision == "ask" and not human_approves(args["command"]):
            return ToolResult("User declined to run this command.",
                              is_error=True)
    return fn(**args)

# `human_approves` is referenced by gated_execute() but its implementation is
# not shown in this chapter (it's a UI-layer prompt in the real harness). We
# add the minimal glue stub here so the "ask" branch of the book's own logic
# is exercised offline and deterministically.
def human_approves(cmd: str) -> bool:
    return False


# ---------------------------------------------------------------------------
# Block #4 (line ~304, 24 lines) -- sandboxed bash tool, verbatim from chapter
# ---------------------------------------------------------------------------
def run_bash(command: str, timeout: int = 120, cwd: str = ".") -> ToolResult:
    """Execute a shell command with the mechanistic guardrails that complement
    the policy gate: a timeout (kills runaway loops), output capping (protects
    the context window), and a fixed cwd (path containment). In production this
    process would also run inside a container/sandbox with no network egress."""
    try:
        proc = subprocess.run(
            command, shell=True, cwd=cwd, timeout=timeout,
            capture_output=True, text=True,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            f"Command timed out after {timeout}s and was killed.", is_error=True)

    out = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
    # Cap output to protect the context window; keep head and tail.
    MAX = 8000
    if len(out) > MAX:
        out = out[:MAX // 2] + "\n... [truncated] ...\n" + out[-MAX // 2:]
    status = "" if proc.returncode == 0 else f"\n[exit code {proc.returncode}]"
    return ToolResult(out + status, is_error=proc.returncode != 0)


# ---------------------------------------------------------------------------
# Block #5 (line ~341, 12 lines) -- todo tracker, verbatim from chapter
# ---------------------------------------------------------------------------
# A todo list is just state the harness owns and re-injects each turn.
# Keeping it at the END of the context exploits the recency the model attends
# to most, and forces the model to externalize its plan instead of holding it
# in fragile working memory.
TODO_STATE = []   # list of {"id", "task", "status"} dicts

def todo_write(items: list) -> ToolResult:
    global TODO_STATE
    TODO_STATE = items
    pending = sum(1 for i in items if i["status"] != "completed")
    return ToolResult(f"Updated plan: {len(items)} items, {pending} remaining.")


# ---------------------------------------------------------------------------
# Block #6 (line ~382, 16 lines) -- verification gate, verbatim from chapter
# ---------------------------------------------------------------------------
def verify_before_done(messages, verify_cmd="pytest -q"):
    """A gate the loop can call before accepting a 'done' (no-tool-call) turn.
    If verification hasn't passed, inject the failure as an observation and
    force another iteration. This single mechanism -- closing the loop on a
    ground-truth check -- is what most cleanly separates reliable harnesses
    from unreliable ones."""
    result = run_bash(verify_cmd)
    if result.is_error:
        messages.append({"role": "user", "content": [{
            "type": "text",
            "text": (f"Verification `{verify_cmd}` FAILED before completion:\n"
                     f"{result.content}\nFix the cause and re-verify."),
        }]})
        return False, messages       # do NOT terminate; keep working
    return True, messages


# ---------------------------------------------------------------------------
# Block #2 (line ~155): SKIP(network) -- `agent_loop` calls the real
# Anthropic API (`from anthropic import Anthropic`; `client.messages.create`).
# We substitute a deterministic offline fake with the SAME call signature so
# that block #7's `run_coding_agent`, which calls `agent_loop(...)`, still
# exercises its own real control flow (context assembly -> loop -> verify).
# ---------------------------------------------------------------------------
TOOL_SCHEMAS = []  # normally JSON schemas for read_file/edit_file/bash/search

def agent_loop(system_prompt: str, user_task: str,
               tool_schemas: list, max_turns: int = 50,
               token_budget: int = 500_000):
    # Offline fake standing in for the real (network) block #2: the model
    # immediately emits a no-tool-call "done" message, structurally ending
    # the loop -- exactly the natural-stop branch of the real loop.
    fake_response = {"role": "assistant", "content": "Task complete (fake)."}
    messages = [{"role": "user", "content": user_task},
                {"role": "assistant", "content": fake_response["content"]}]
    return fake_response, messages


# ---------------------------------------------------------------------------
# Block #7 (line ~409, 37 lines) -- top-level orchestration, verbatim
# ---------------------------------------------------------------------------
def build_system_prompt(repo_root: str) -> str:
    """Context assembly for the static preamble. Byte-stable across turns so
    the API-side prefix cache stays warm."""
    import subprocess, datetime
    git_status = subprocess.run(["git", "-C", repo_root, "status", "--short"],
                                capture_output=True, text=True).stdout
    memory = (Path(repo_root) / "CLAUDE.md")
    memory_text = memory.read_text() if memory.exists() else "(none)"
    return f"""You are a coding agent operating in a terminal.

Working directory: {repo_root}
Date: {datetime.date.today()}
Git status:
{git_status or '(clean)'}

Project instructions (CLAUDE.md):
{memory_text}

Rules:
- Read a file before editing it. Make the smallest correct change.
- Use `search` to locate code; do not guess file contents.
- After editing, ALWAYS run the project's tests to verify before finishing.
- Be concise. Finish by emitting a short summary with no tool calls.
"""

def run_coding_agent(repo_root: str, task: str):
    system = build_system_prompt(repo_root)
    schemas = TOOL_SCHEMAS                 # JSON schemas for the 5 tools
    final, messages = agent_loop(system, task, schemas)

    # Final verification gate: do not accept 'done' until tests pass.
    ok, messages = verify_before_done(messages)
    while not ok:
        final, messages = agent_loop(system, "", schemas)  # continue working
        ok, messages = verify_before_done(messages)
    return final


# ===========================================================================
# Tests
# ===========================================================================

def test_block0_edit_file():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "foo.py"
        p.write_text("def f():\n    return OLD_VALUE\n")

        # Successful unique-match edit.
        res = edit_file(str(p), "OLD_VALUE", "NEW_VALUE")
        assert not res.is_error
        assert "1 replacement" in res.content
        assert p.read_text() == "def f():\n    return NEW_VALUE\n"

        # Zero matches -> loud, recoverable error.
        res2 = edit_file(str(p), "DOES_NOT_EXIST", "X")
        assert res2.is_error
        assert "not found" in res2.content

        # Ambiguous (2+ matches) -> loud, recoverable error.
        p2 = Path(d) / "bar.py"
        p2.write_text("x = 1\nx = 1\n")
        res3 = edit_file(str(p2), "x = 1", "x = 2")
        assert res3.is_error
        assert "appears 2 times" in res3.content

        # Missing file.
        res4 = edit_file(str(Path(d) / "nope.py"), "a", "b")
        assert res4.is_error
        assert "does not exist" in res4.content

    # ToolResult defaults.
    tr = ToolResult("ok")
    assert tr.is_error is False


def test_block1_read_file():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "src.py"
        p.write_text("a\nb\nc\nd\ne\n")

        # Full read: 1-based line numbers, no truncation note (limit covers all).
        full = read_file(str(p))
        assert not full.is_error
        assert "     1\ta" in full.content
        assert "     5\te" in full.content
        assert "more lines" not in full.content

        # Windowed read: offset+limit smaller than file -> truncation note, and
        # the numbering is offset-relative (1-based on the absolute line).
        win = read_file(str(p), offset=1, limit=2)
        assert "     2\tb" in win.content
        assert "     3\tc" in win.content
        assert "     1\ta" not in win.content
        assert "(2 more lines; use offset to continue)" in win.content

        # Missing file -> loud error.
        miss = read_file(str(Path(d) / "nope.py"))
        assert miss.is_error
        assert "does not exist" in miss.content


def test_block3_permission_gate():
    assert classify("rm -rf /tmp/build") == "deny"
    assert classify("git push origin main --force") == "deny"
    assert classify("ls -la") == "allow"
    assert classify("git status --short") == "allow"
    assert classify("npm install left-pad") == "ask"

    with tempfile.TemporaryDirectory() as d:
        # Allowed command actually executes.
        out = gated_execute("bash", {"command": "ls"}, run_bash)
        assert not out.is_error

        # Denied command never reaches the tool function.
        out2 = gated_execute("bash", {"command": "rm -rf /"}, run_bash)
        assert out2.is_error
        assert "Blocked" in out2.content

        # Unknown command escalates to human approval; our stub declines.
        out3 = gated_execute("bash", {"command": "npm install left-pad"}, run_bash)
        assert out3.is_error
        assert "declined" in out3.content


def test_block4_run_bash():
    ok = run_bash("echo hello_book_test")
    assert not ok.is_error
    assert "hello_book_test" in ok.content

    bad = run_bash("exit 7")
    assert bad.is_error
    assert "[exit code 7]" in bad.content

    timed_out = run_bash("sleep 2", timeout=1)
    assert timed_out.is_error
    assert "timed out" in timed_out.content


def test_block5_todo():
    global TODO_STATE
    items = [
        {"id": 1, "task": "write tests", "status": "pending"},
        {"id": 2, "task": "fix bug", "status": "completed"},
    ]
    res = todo_write(items)
    assert TODO_STATE == items
    assert "2 items, 1 remaining." in res.content


def test_block6_verify_before_done():
    msgs = []
    ok, msgs = verify_before_done(msgs, verify_cmd='python3 -c "import sys; sys.exit(0)"')
    assert ok is True
    assert msgs == []

    msgs2 = []
    ok2, msgs2 = verify_before_done(msgs2, verify_cmd='python3 -c "import sys; sys.exit(1)"')
    assert ok2 is False
    assert len(msgs2) == 1
    assert "FAILED before completion" in msgs2[0]["content"][0]["text"]


def test_block7_orchestration():
    with tempfile.TemporaryDirectory() as repo:
        (Path(repo) / "CLAUDE.md").write_text("Always run `make test` before finishing.\n")

        prompt = build_system_prompt(repo)
        assert f"Working directory: {repo}" in prompt
        assert "Always run `make test`" in prompt
        assert "(clean)" in prompt or "Git status" in prompt

        # Shim a `pytest` stub onto PATH so verify_before_done's hardcoded
        # default `pytest -q` resolves deterministically and offline, instead
        # of either failing (pytest not installed) or recursing into the
        # book's real, potentially large test suite.
        with tempfile.TemporaryDirectory() as bindir:
            stub = Path(bindir) / "pytest"
            stub.write_text("#!/bin/sh\nexit 0\n")
            stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

            old_path = os.environ.get("PATH", "")
            os.environ["PATH"] = bindir + os.pathsep + old_path
            try:
                final = run_coding_agent(repo, "Fix the failing test")
            finally:
                os.environ["PATH"] = old_path

        assert final is not None
        assert final["content"] == "Task complete (fake)."


# ---------------------------------------------------------------------------
def main():
    tested = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
            tested += 1
    skipped = 1  # block #2 (network, offline-faked)
    print(f"\n{tested} test functions passed; {skipped} chapter blocks SKIPPED.")


if __name__ == "__main__":
    main()
