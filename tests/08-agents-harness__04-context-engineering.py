"""
Runs the CPU-runnable Python code blocks from:
    content/08-agents-harness/04-context-engineering.md

Blocks are copied faithfully (verbatim logic) and concatenated in document
order, with small glue/fixtures added so each block actually executes.

Tested blocks:
    #1 (line ~92)  -- needle-in-haystack harness (build_haystack, eval_grid)
    #2 (line ~161) -- just-in-time context tools (grep_tool, read_window)
    #3 (line ~217) -- token budget manager (Segment, ContextBuilder, ntok)
    #4 (line ~315) -- compaction (compact, maybe_compact)
    #5 (line ~377) -- externalized scratchpad (Scratchpad)

Skipped blocks (not in scope per task spec):
    #0 -- non-python (ASCII diagram of the context-window "stack of segments")
    #6 -- non-python (ASCII diagram of cache-friendly vs cache-busting ordering)
    #7 -- SKIP(network): `build_cached_request` assembles a request dict for a
          real Anthropic-style API call; it makes no network call itself, but
          it exists purely to be *sent* to a live API and has no interesting
          standalone logic beyond dict/list construction already exercised by
          block #3's ContextBuilder. Not executed here.

No network or GPU calls anywhere below. `tiktoken` (used by block #3/#4 for
real tokenization) is optional and already guarded by the book's own
try/except -- if unavailable, the heuristic `ntok` fallback is used instead,
exactly as the book intends.

Real bug found and fixed (mirrored in content/08-agents-harness/04-context-engineering.md):
    In block #4's `compact()`, the safety-net hard-trim used
        toks = _enc.encode(summary)[:max_tokens] if 'tiktoken' in dir() else None
    `dir()` with no arguments returns names in the *current local scope*, not
    module globals -- inside a function body, the module-level `tiktoken`
    import is never a local name, so `'tiktoken' in dir()` is ALWAYS False,
    even when tiktoken imported successfully and `_enc` exists. The hard-trim
    safety net therefore never fired. Fixed to a try/except NameError around
    the `_enc` reference, which correctly detects whether the real tokenizer
    is available.
"""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import tempfile
from dataclasses import dataclass, field


def _section(name: str) -> None:
    print(f"\n=== {name} ===")


# ============================================================================
# Block #1 (line ~92) -- needle-in-haystack harness to measure rot/position
# effects. Copied verbatim from the book.
# ============================================================================
_section("Block #1: needle-in-haystack harness")

# A "needle in a haystack" harness to MEASURE rot/position effects for your stack.
# Run this against your own model+prompt; do not trust generic claims.
# (import random already at module scope above)

def build_haystack(needle: str, filler_tokens: list[str],
                   total_tokens: int, needle_frac: float) -> str:
    """Embed `needle` at a fractional depth in a haystack of `total_tokens`.

    needle_frac=0.0 -> needle at start, 0.5 -> middle, 1.0 -> end.
    `filler_tokens` is a large pool of irrelevant words to pad with.
    """
    n_pad = max(0, total_tokens - len(needle.split()))
    insert_at = int(n_pad * needle_frac)
    body = [random.choice(filler_tokens) for _ in range(n_pad)]
    words = body[:insert_at] + needle.split() + body[insert_at:]
    return " ".join(words)

def eval_grid(call_model, needle, question, expected,
              lengths=(2000, 8000, 32000, 128000),
              depths=(0.0, 0.25, 0.5, 0.75, 1.0),
              filler_pool=None, trials=3):
    """Sweep (length x depth). Returns accuracy per cell.
    `call_model(prompt) -> str`. We score by substring match on `expected`."""
    filler_pool = filler_pool or ["the","system","logs","report","value",
                                  "module","request","status","cache","node"]
    results = {}
    for L in lengths:
        for d in depths:
            hits = 0
            for _ in range(trials):
                hay = build_haystack(needle, filler_pool, L, d)
                prompt = (f"Read the following text and answer the question.\n\n"
                          f"{hay}\n\nQuestion: {question}\nAnswer:")
                out = call_model(prompt)
                hits += int(expected.lower() in out.lower())
            results[(L, d)] = hits / trials
    return results

# Reading the result grid you will typically see:
#   - accuracy falling as L grows (context rot),
#   - within a row, a dip around d=0.5 (lost-in-the-middle).
# Use it to choose YOUR effective window, not the advertised one.

# --- glue: exercise build_haystack + eval_grid with a tiny, deterministic,
# offline "model" (no network). The fake model does perfect substring
# retrieval so we can assert the harness's own bookkeeping is correct,
# independent of any real LLM's accuracy. ---
random.seed(0)

def _fake_perfect_retrieval_model(prompt: str) -> str:
    # Simulates a model that always finds the needle if it's in the prompt.
    m = re.search(r"Question:.*", prompt, re.DOTALL)
    return "42" if "the secret number is 42" in prompt.lower() else "unknown"

hay_sample = build_haystack("the secret number is 42",
                             ["foo", "bar", "baz", "qux"],
                             total_tokens=50, needle_frac=0.5)
assert "42" in hay_sample.split()
print(f"haystack sample ({len(hay_sample.split())} words): {hay_sample[:80]}...")

grid = eval_grid(_fake_perfect_retrieval_model,
                  needle="the secret number is 42",
                  question="What is the secret number?",
                  expected="42",
                  lengths=(20, 60),
                  depths=(0.0, 0.5, 1.0),
                  trials=2)
assert len(grid) == 2 * 3
assert all(v == 1.0 for v in grid.values()), grid  # perfect model -> perfect grid
print(f"eval_grid ran {len(grid)} cells, all accuracy=1.0 as expected: {grid}")


# ============================================================================
# Block #2 (line ~161) -- just-in-time context via tools (grep_tool,
# read_window). Copied verbatim from the book.
# ============================================================================
_section("Block #2: just-in-time context tools")

# Just-in-time context via tools. The agent pulls minimal slices on demand
# instead of us pre-stuffing whole files. Note that we return *windows*
# around hits, not whole files, to protect the budget.
# (import subprocess, re already at module scope above)

def grep_tool(pattern: str, path: str = ".", max_hits: int = 20) -> str:
    """Return file:line: matches. Cheap to put in context; points the agent
    at *where* to look without dumping whole files."""
    try:
        out = subprocess.run(["grep", "-rni", pattern, path],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception as e:
        return f"grep error: {e}"
    lines = out.splitlines()[:max_hits]
    return "\n".join(lines) if lines else "(no matches)"

def read_window(path: str, center_line: int, radius: int = 40) -> str:
    """Read only lines [center-radius, center+radius]. Returning a 80-line
    window instead of a 4000-line file is the difference between a 600-token
    and a 30000-token tool result."""
    with open(path) as f:
        lines = f.readlines()
    lo = max(0, center_line - radius)
    hi = min(len(lines), center_line + radius)
    body = "".join(f"{i+1:>5} | {lines[i]}" for i in range(lo, hi))
    return f"# {path} lines {lo+1}-{hi}\n{body}"

# The agent's loop: grep to localize, read_window to inspect, then act.
# At no point does a whole 4000-line file enter the context.

# --- glue: a tiny fixture file to grep and window-read, standing in for the
# "server.py" the book refers to later (block #3 reuses read_window on it). ---
_tmpdir = tempfile.mkdtemp(prefix="ctxeng_")
SERVER_PY = os.path.join(_tmpdir, "server.py")
with open(SERVER_PY, "w") as _f:
    for _i in range(1, 301):
        if _i == 120:
            _f.write(f"def paginate(items, page):  # TODO off-by-one bug here (line {_i})\n")
        else:
            _f.write(f"# line {_i}: filler code\n")

grep_result = grep_tool("TODO", _tmpdir)
assert "server.py" in grep_result and "off-by-one" in grep_result
print(f"grep_tool found: {grep_result}")

window_result = read_window(SERVER_PY, 120)
assert "lines 81-160" in window_result  # lo=120-40=80 (0-idx) -> displayed as lo+1
assert "off-by-one" in window_result
print(f"read_window returned {len(window_result.splitlines())} lines, header: "
      f"{window_result.splitlines()[0]}")


# ============================================================================
# Block #3 (line ~217) -- token budget manager (Segment, ContextBuilder).
# Copied verbatim from the book.
# ============================================================================
_section("Block #3: token budget manager")

# A budget manager that assembles a prompt under a hard token ceiling.
# It counts with the REAL tokenizer and enforces per-segment caps,
# evicting from the middle of history (oldest-but-not-first) on overflow.
# (from dataclasses import dataclass, field already at module scope above)

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")  # stand-in tokenizer
    def ntok(s: str) -> int: return len(_enc.encode(s))
except Exception:
    # Fallback heuristic ~4 chars/token; ONLY for environments without tiktoken.
    def ntok(s: str) -> int: return max(1, len(s) // 4)

@dataclass
class Segment:
    name: str
    budget: int                 # max tokens this segment may occupy
    items: list[str] = field(default_factory=list)
    pinned: list[bool] = field(default_factory=list)  # never-evict flags

    def add(self, text: str, pin: bool = False):
        self.items.append(text); self.pinned.append(pin)

    def tokens(self) -> int:
        return sum(ntok(x) for x in self.items)

    def fit(self):
        """Evict UNPINNED items, oldest first, until under budget.
        Keeps pinned items (e.g., the system prompt, the current task)."""
        while self.tokens() > self.budget:
            # find oldest unpinned item to drop
            idx = next((i for i, p in enumerate(self.pinned) if not p), None)
            if idx is None:
                break  # everything pinned; nothing legal to drop
            dropped = self.items.pop(idx); self.pinned.pop(idx)
            # leave a breadcrumb so the model knows something was removed
            self.items.insert(idx, f"[elided {ntok(dropped)} tokens]")
            self.pinned.insert(idx, True)

class ContextBuilder:
    def __init__(self, w_eff: int, output_reserve: int = 4000):
        self.w_eff = w_eff
        self.output_reserve = output_reserve
        self.segments: list[Segment] = []

    def segment(self, name: str, budget: int) -> Segment:
        s = Segment(name, budget); self.segments.append(s); return s

    def build(self) -> str:
        # 1) enforce per-segment budgets
        for s in self.segments:
            s.fit()
        # 2) enforce the GLOBAL ceiling: if total still too big,
        #    squeeze the largest elastic segment further.
        ceiling = self.w_eff - self.output_reserve
        while sum(s.tokens() for s in self.segments) > ceiling:
            big = max(self.segments, key=lambda s: s.tokens())
            big.budget = int(big.budget * 0.85)   # tighten and re-fit
            big.fit()
            if all(s.tokens() <= 1 for s in self.segments):
                break
        return "\n\n".join("\n".join(s.items) for s in self.segments)

# --- usage ---------------------------------------------------------------
cb = ContextBuilder(w_eff=32_000, output_reserve=4000)
sys_seg   = cb.segment("system",    budget=2_000)
ret_seg   = cb.segment("retrieved", budget=8_000)
hist_seg  = cb.segment("history",   budget=16_000)
task_seg  = cb.segment("task",      budget=2_000)

sys_seg.add("You are a coding agent. Follow the plan. Be terse.", pin=True)
ret_seg.add(read_window(SERVER_PY, 120))           # only the relevant slice
for turn in []:  # ... append conversation turns here ...
    hist_seg.add(turn)
task_seg.add("Fix the off-by-one in pagination.", pin=True)

prompt = cb.build()
print(f"assembled {ntok(prompt)} tokens (ceiling {32_000-4000})")

# --- glue: exercise the eviction path too (an oversized segment must shrink
# under its budget, leaving an "[elided ...]" breadcrumb). ---
overflow_seg = Segment("overflow_test", budget=50)
overflow_seg.add("this is an old, unpinned block of text " * 20)   # big
overflow_seg.add("current task", pin=True)
assert overflow_seg.tokens() > overflow_seg.budget
overflow_seg.fit()
assert overflow_seg.tokens() <= overflow_seg.budget
assert any(item.startswith("[elided") for item in overflow_seg.items)
print(f"eviction test: segment fit to {overflow_seg.tokens()} tokens, "
      f"items={overflow_seg.items}")


# ============================================================================
# Block #4 (line ~315) -- compaction (compact, maybe_compact).
# Copied from the book, with one bug fix (see module docstring): the
# `'tiktoken' in dir()` check inside `compact()` can never be True because
# `dir()` with no arguments only sees the function's LOCAL scope, never the
# module-level `tiktoken` import. Replaced with a try/except NameError on
# `_enc`, which is the actual thing being conditionally available.
# ============================================================================
_section("Block #4: compaction")

# Compaction: collapse old turns into a structured snapshot via a model call.
# The summary is itself a context artifact, so we constrain its length hard.
COMPACTION_PROMPT = """\
You are compacting an agent's working memory. Produce a STRUCTURED state \
snapshot under {max_tokens} tokens. Preserve only what is needed to CONTINUE \
the task. Use exactly these sections; omit a section if empty.

## Goal
<the user's original objective, one or two sentences>
## Decisions
<bulleted: decision -> rationale>
## Files changed
<path: what changed and current state>
## Constraints discovered
<bulleted facts that constrain future actions>
## Open TODOs
<ordered, actionable>
## Current hypothesis / next step
<one short paragraph>

Do NOT include raw logs, superseded plans, or verbose reasoning.
--- CONVERSATION TO COMPACT ---
{history}
"""

def compact(call_model, turns: list[str], max_tokens: int = 1200) -> str:
    history = "\n\n".join(turns)
    prompt = COMPACTION_PROMPT.format(max_tokens=max_tokens, history=history)
    summary = call_model(prompt)
    # Hard-trim as a safety net in case the model over-runs.
    # (BUG FIX: `'tiktoken' in dir()` is always False inside a function body
    #  -- dir() with no args sees only local names, never module globals.
    #  The correct check is whether the real-tokenizer encoder `_enc` is
    #  actually available, which we detect via NameError.)
    try:
        toks = _enc.encode(summary)[:max_tokens]
    except NameError:
        toks = None
    return _enc.decode(toks) if toks is not None else summary

def maybe_compact(hist_seg: "Segment", call_model, keep_recent: int = 6):
    """If history exceeds 75% of budget, compact everything except the most
    recent `keep_recent` turns (recency is high-value; never summarize it)."""
    if hist_seg.tokens() < 0.75 * hist_seg.budget:
        return
    old, recent = hist_seg.items[:-keep_recent], hist_seg.items[-keep_recent:]
    snapshot = compact(call_model, old)
    # Replace the old span with one pinned snapshot; keep recent verbatim.
    hist_seg.items = [f"## COMPACTED STATE\n{snapshot}"] + recent
    hist_seg.pinned = [True] + [False] * len(recent)

# --- glue: a canned, offline "model" for compaction (no network) plus a
# history segment pushed well past its 75% high-water mark. ---
def _fake_compaction_model(prompt: str) -> str:
    return ("## Goal\nFix pagination off-by-one.\n"
            "## Decisions\n- Use half-open ranges -> matches slicing convention\n"
            "## Files changed\nserver.py: patched paginate()\n"
            "## Constraints discovered\n- API rate-limits at 10 rps\n"
            "## Open TODOs\n1. Add regression test\n"
            "## Current hypothesis / next step\nVerify with unit tests.\n")

# 1) Directly test compact(): the fixed hard-trim safety net.
short_summary = compact(_fake_compaction_model, ["turn A", "turn B"], max_tokens=500)
assert isinstance(short_summary, str) and len(short_summary) > 0
print(f"compact() (no trim needed) -> {len(short_summary)} chars")

def _fake_verbose_model(prompt: str) -> str:
    return "word " * 5000  # deliberately far over max_tokens

trimmed = compact(_fake_verbose_model, ["turn A"], max_tokens=20)
assert isinstance(trimmed, str)
try:
    import tiktoken as _tk_check  # noqa: F401
    _has_tiktoken = True
except Exception:
    _has_tiktoken = False
if _has_tiktoken:
    # With a real tokenizer available, the safety net must actually fire and
    # cut the oversized summary down to (approximately) max_tokens tokens.
    assert ntok(trimmed) <= 25, f"hard-trim did not fire: {ntok(trimmed)} tokens"
    print(f"compact() hard-trim fired: {ntok(trimmed)} tokens (<=20 requested)")
else:
    print("tiktoken unavailable -> compact() fell back to untrimmed summary "
          "(expected fallback behavior)")

# 2) Exercise maybe_compact end-to-end on the hist_seg built in block #3.
for i in range(80):
    hist_seg.add(f"[turn {i}] tool ran command and produced verbose output " * 20,
                 pin=False)
assert hist_seg.tokens() >= 0.75 * hist_seg.budget, \
    f"fixture didn't reach high-water mark: {hist_seg.tokens()} tokens"
n_before = len(hist_seg.items)

maybe_compact(hist_seg, _fake_compaction_model, keep_recent=6)

assert len(hist_seg.items) == 7, hist_seg.items  # 1 compacted + 6 recent
assert hist_seg.items[0].startswith("## COMPACTED STATE")
assert hist_seg.pinned[0] is True
assert hist_seg.pinned[1:] == [False] * 6
print(f"maybe_compact collapsed {n_before} turns -> {len(hist_seg.items)} "
      f"items (1 pinned snapshot + 6 recent verbatim)")

# A second call, now under the high-water mark, must be a no-op.
items_snapshot = list(hist_seg.items)
maybe_compact(hist_seg, _fake_compaction_model, keep_recent=6)
assert hist_seg.items == items_snapshot
print("maybe_compact is a no-op once below the 75% high-water mark")


# ============================================================================
# Block #5 (line ~377) -- externalized scratchpad (Scratchpad).
# Copied verbatim from the book.
# ============================================================================
_section("Block #5: externalized scratchpad")

# An externalized scratchpad. The plan lives on disk, NOT only in the window.
# Each turn we re-read the (small) current plan into a pinned segment, so it
# survives compaction and the model always sees an up-to-date task structure.
# (import json, os already at module scope above)

class Scratchpad:
    def __init__(self, path="PLAN.md"):
        self.path = path
        if not os.path.exists(path):
            self._write({"goal": "", "todos": [], "notes": [], "done": []})

    def _write(self, state: dict):
        with open(self.path, "w") as f:
            f.write("# PLAN\n")
            f.write(f"## Goal\n{state['goal']}\n\n")
            f.write("## TODO\n" + "".join(f"- [ ] {t}\n" for t in state["todos"]))
            f.write("\n## Done\n" + "".join(f"- [x] {d}\n" for d in state["done"]))
            f.write("\n## Notes\n" + "".join(f"- {n}\n" for n in state["notes"]))
            # Keep a machine-readable mirror for reliable parsing.
            with open(self.path + ".json", "w") as g:
                json.dump(state, g)

    def state(self) -> dict:
        with open(self.path + ".json") as f:
            return json.load(f)

    def render(self) -> str:
        """Small, dense view to pin into context every turn."""
        with open(self.path) as f:
            return f.read()

    # Tool surface the agent calls:
    def set_goal(self, goal: str):
        s = self.state(); s["goal"] = goal; self._write(s)
    def add_todo(self, item: str):
        s = self.state(); s["todos"].append(item); self._write(s)
    def complete(self, item: str):
        s = self.state()
        if item in s["todos"]:
            s["todos"].remove(item); s["done"].append(item); self._write(s)
    def note(self, text: str):
        s = self.state(); s["notes"].append(text); self._write(s)

# Each turn:
#   plan_seg.items = [pad.render()]; plan_seg.pinned = [True]
# The plan is ~300 tokens, always fresh, and immune to history compaction.

# --- glue: instantiate against a real temp file and drive the tool surface. ---
plan_path = os.path.join(_tmpdir, "PLAN.md")
pad = Scratchpad(plan_path)
pad.set_goal("Fix the off-by-one in pagination.")
pad.add_todo("Reproduce the bug with a failing test")
pad.add_todo("Patch paginate() in server.py")
pad.note("API rate-limits at 10 rps")
pad.complete("Reproduce the bug with a failing test")

state = pad.state()
assert state["goal"] == "Fix the off-by-one in pagination."
assert state["todos"] == ["Patch paginate() in server.py"]
assert state["done"] == ["Reproduce the bug with a failing test"]
assert state["notes"] == ["API rate-limits at 10 rps"]

rendered = pad.render()
assert "## TODO" in rendered and "Patch paginate() in server.py" in rendered
assert "## Done" in rendered and "Reproduce the bug" in rendered
assert os.path.exists(plan_path + ".json")

plan_seg = Segment("plan", budget=2_000)
plan_seg.add(pad.render(), pin=True)
assert plan_seg.pinned == [True]
print(f"Scratchpad round-tripped through disk at {plan_path}; "
      f"rendered plan is {ntok(pad.render())} tokens, pinned into a segment.")


print("\nAll 5 CPU-runnable blocks executed successfully.")
