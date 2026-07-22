"""
Runs the CPU-runnable Python blocks from content/08-agents-harness/02-agentic-loop.md.

Blocks tested (verbatim from the chapter, with tiny calling glue added):
    - block #8 (line ~641): MAX_CONTEXT_FRACTION / context_full()  — token-budget termination check
    - block #9 (line ~671): detect_loop()                          — infinite-loop detection heuristic

All other blocks in this chapter are prose fragments or require a real/mocked
network-calling LLM client (openai.OpenAI chat completions) whose *point* is
the API interaction itself (ReActAgent, make_plan, execute_plan, reflect,
critic_gate, beam_react_agent) — those are SKIPPED per instructions, see the
notes at the bottom of this file.
"""

import sys


# ---------------------------------------------------------------------------
# Block #8 (line ~641) — verbatim from the chapter
# ---------------------------------------------------------------------------

MAX_CONTEXT_FRACTION = 0.85  # stop at 85% of context window

def context_full(messages: list, model_context_limit: int = 128_000) -> bool:
    """Rough check: count characters as a proxy for tokens (4 chars ≈ 1 token)."""
    total_chars = sum(len(m["content"]) for m in messages)
    estimated_tokens = total_chars / 4
    return estimated_tokens > MAX_CONTEXT_FRACTION * model_context_limit


# ---------------------------------------------------------------------------
# Block #9 (line ~671) — verbatim from the chapter
# ---------------------------------------------------------------------------

from collections import Counter

def detect_loop(action_history: list[str], window: int = 4, threshold: int = 3) -> bool:
    """
    Returns True if the same action appears >= threshold times
    in the last 'window' steps.
    """
    recent = action_history[-window:]
    counts = Counter(recent)
    return any(c >= threshold for c in counts.values())


# ---------------------------------------------------------------------------
# Glue: actually exercise both functions on tiny fixtures.
# ---------------------------------------------------------------------------

def main() -> None:
    # --- context_full() ---
    # Small conversation: well under the 85% threshold for a small model context.
    small_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "Thought: trivial arithmetic.\nAction: finish(4)"},
    ]
    assert context_full(small_messages, model_context_limit=1000) is False, (
        "Small conversation should not be considered 'full'."
    )

    # Force the fraction to be exceeded: total_chars/4 > 0.85 * limit.
    # Use a tiny limit so a short string easily exceeds it.
    big_content = "x" * 100  # 100 chars -> 25 estimated tokens
    big_messages = [{"role": "user", "content": big_content}]
    assert context_full(big_messages, model_context_limit=10) is True, (
        "Long conversation relative to a tiny context limit should be 'full'."
    )

    # Sanity-check the module constant is what the book states.
    assert MAX_CONTEXT_FRACTION == 0.85

    print("context_full(): OK")
    print(f"  small_messages -> full={context_full(small_messages, 1000)}")
    print(f"  big_messages   -> full={context_full(big_messages, 10)}")

    # --- detect_loop() ---
    # Case 1: the same action repeated >= threshold times within the window -> True
    repeating_history = [
        "search('Python syntax')",
        "search('Python syntax')",
        "search('Python syntax')",
    ]
    assert detect_loop(repeating_history, window=4, threshold=3) is True, (
        "Three identical actions should be detected as a loop."
    )

    # Case 2: varied actions -> False
    varied_history = [
        "search('Tokyo population')",
        "search('New York population')",
        "calculator('37 / 20')",
        "finish('1.85')",
    ]
    assert detect_loop(varied_history, window=4, threshold=3) is False, (
        "Four distinct actions should not be flagged as a loop."
    )

    # Case 3: repetition outside the window shouldn't count.
    mixed_history = [
        "search('a')", "search('a')", "search('a')",  # outside the last-4 window below
        "search('b')", "calculator('1+1')", "finish('2')", "search('c')",
    ]
    assert detect_loop(mixed_history, window=4, threshold=3) is False, (
        "Repeats outside the trailing window should not trigger detection."
    )

    print("detect_loop(): OK")
    print(f"  repeating_history -> loop={detect_loop(repeating_history)}")
    print(f"  varied_history    -> loop={detect_loop(varied_history)}")
    print(f"  mixed_history     -> loop={detect_loop(mixed_history)}")

    print("\nAll CPU-runnable blocks executed successfully.")


# ---------------------------------------------------------------------------
# SKIP notes (blocks not executed here, per task instructions):
#
#   #0  (text fence, not python) — the `while not done: ...` pseudocode loop.
#   #1  REACT_SYSTEM_PROMPT string constant — fragment, nothing to execute.
#   #2  minimal_react.py (ReActAgent + tools) — SKIP(network): instantiates
#       `openai.OpenAI(api_key=...)` and calls `client.chat.completions.create`,
#       a real network/API call. Also depends on OPENAI_API_KEY being set.
#   #3  PLAN_PROMPT string constant — fragment, nothing to execute.
#   #4  make_plan()/execute_plan() — SKIP(network): calls
#       `client.chat.completions.create(...)` against a live OpenAI client and
#       also calls `agent.run()` from the network-bound ReActAgent (#2).
#   #5  reflect() — SKIP(network): calls `client.chat.completions.create(...)`.
#   #6  critic_gate() — SKIP(network): calls `client.chat.completions.create(...)`.
#   #7  beam_react_agent() — SKIP(network): calls `client.chat.completions.create(...)`
#       in a loop, and additionally uses `ReActAgent._parse_action(None, text)` /
#       `ReActAgent._execute_action(None, ...)` in an unusual "static-style" call
#       pattern that depends on class #2.
#
# Blocks #8 and #9 are pure, self-contained, deterministic Python with no I/O
# or network dependency, so they are executed above verbatim.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    main()
    sys.exit(0)
