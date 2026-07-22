"""
Runnability test for content/05-posttraining-alignment/10-reasoning-test-time-compute.md

Tests the 2 heuristically CPU-runnable Python blocks from the chapter,
concatenated in chapter order:

    - block #5 (line ~419, 70 lines) -- MCTSNode dataclass + mcts_search()
      (minimal Monte Carlo Tree Search for LLM reasoning)
    - block #6 (line ~539, 24 lines) -- EXAMPLE_REASONING_TRACE, an
      illustrative R1-style reasoning trace string constant

Other blocks are SKIPPED:
    - #0: non-python (figure/prose reference, not a code block)
    - #1, #2, #3: needs-net (PRM training / API-calling snippets that talk
      to a model server or HF hub)
    - #4: needs-gpu (PRM scoring with a real transformer model on device)
    - #7: needs-net (forced_budget_inference() takes an API `client` object
      and makes a real network call to a hosted model; the chapter itself
      presents it purely as an illustration of the budget-forcing control
      pattern, not as something to run offline)

Blocks #5 and #6 have no third-party dependencies beyond the stdlib, so
they are copied verbatim from the chapter and exercised directly.
"""

import math
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# Block #5 (line ~419): MCTS for LLM reasoning. Verbatim from the chapter.
# =============================================================================

@dataclass
class MCTSNode:
    state: str            # reasoning context so far
    parent: Optional["MCTSNode"] = None
    children: list["MCTSNode"] = field(default_factory=list)
    N: int = 0            # visit count
    Q: float = 0.0        # mean value
    P: float = 1.0        # prior from LM (log-prob of this branch)

    def ucb_score(self, c_puct: float = 2.0) -> float:
        if self.N == 0:
            return float("inf")
        parent_N = self.parent.N if self.parent else self.N
        return self.Q + c_puct * self.P * math.sqrt(parent_N) / (1 + self.N)

    def is_leaf(self) -> bool:
        return len(self.children) == 0


def mcts_search(
    lm_policy,       # callable(state) -> list[(next_thought, log_prob)]
    value_fn,        # callable(state) -> float  (PRM or rollout)
    root_state: str,
    n_iterations: int = 50,
    expansion_width: int = 3,
    c_puct: float = 2.0,
) -> MCTSNode:
    """
    Minimal MCTS for LLM reasoning. Returns the root node; caller can
    extract the best path by following max-Q children.
    """
    root = MCTSNode(state=root_state)

    for _ in range(n_iterations):
        # --- SELECT ---
        node = root
        path = [node]
        while not node.is_leaf():
            node = max(node.children, key=lambda c: c.ucb_score(c_puct))
            path.append(node)

        # --- EXPAND ---
        if node.N > 0 or node == root:  # expand visited nodes or root
            candidates = lm_policy(node.state)  # [(thought, log_prob), ...]
            for thought, log_prob in candidates[:expansion_width]:
                child = MCTSNode(
                    state=node.state + "\n" + thought,
                    parent=node,
                    P=math.exp(log_prob),
                )
                node.children.append(child)
            if node.children:
                # Descend into first unexplored child
                node = node.children[0]
                path.append(node)

        # --- SIMULATE (evaluate) ---
        value = value_fn(node.state)

        # --- BACKPROP ---
        for n in reversed(path):
            n.N += 1
            n.Q += (value - n.Q) / n.N  # running mean update

    return root


# =============================================================================
# Block #6 (line ~539): Illustration of an R1-style reasoning trace.
# Verbatim from the chapter.
# =============================================================================

# Illustration: a reasoning trace from an R1-style model
# (simplified; real traces are much longer and more varied)

EXAMPLE_REASONING_TRACE = """
<think>
Let me work through this step by step.

The problem asks for the number of ways to arrange 5 distinct books
on a shelf such that two specific books (A and B) are never adjacent.

First, let me count total arrangements: 5! = 120.

Now, let me count arrangements where A and B ARE adjacent.
Treat A+B as a single "super-book": 4! arrangements × 2 (AB or BA) = 48.

Wait, let me double-check: treating [AB] as one unit, we have 4 units total.
4! = 24 arrangements, and we can have AB or BA so ×2 = 48. Yes, that's right.

Therefore, arrangements where A and B are NOT adjacent = 120 - 48 = 72.
</think>

The answer is **72**.
"""


# =============================================================================
# Glue: tiny toy fixtures to actually EXERCISE the code above on CPU.
# =============================================================================

def _toy_lm_policy(state: str):
    """
    Fake LM policy: deterministically proposes a small, finite set of
    "next thought" candidates with log-probabilities, derived from how
    deep the current state already is (measured by newline count). This
    keeps the search tree small and bounded so the toy run completes fast.
    """
    depth = state.count("\n")
    if depth >= 3:
        # Terminal-ish states: offer a single "conclude" move so the tree
        # doesn't grow unboundedly.
        return [("Therefore, the answer is 42.", math.log(0.9))]
    return [
        ("Consider approach A.", math.log(0.5)),
        ("Consider approach B.", math.log(0.3)),
        ("Consider approach C.", math.log(0.2)),
    ]


def _toy_value_fn(state: str) -> float:
    """
    Fake PRM/value function: rewards states that mention "42" (the
    "correct" toy answer) and otherwise scores by how far the search has
    progressed, giving MCTS a meaningful (if trivial) signal to optimize.
    """
    if "42" in state:
        return 1.0
    return 0.1 * state.count("\n")


def main():
    # --- Exercise MCTSNode directly ---
    root = MCTSNode(state="root")
    assert root.is_leaf()
    assert root.ucb_score() == float("inf"), "unvisited node must have infinite UCB"

    child = MCTSNode(state="root\nchild", parent=root, P=0.5)
    root.children.append(child)
    assert not root.is_leaf()
    root.N = 4
    child.N = 1
    child.Q = 0.7
    score = child.ucb_score(c_puct=2.0)
    expected = 0.7 + 2.0 * 0.5 * math.sqrt(4) / (1 + 1)
    assert math.isclose(score, expected), f"ucb_score mismatch: {score} vs {expected}"

    # --- Exercise mcts_search end-to-end with toy policy/value fns ---
    search_root = mcts_search(
        lm_policy=_toy_lm_policy,
        value_fn=_toy_value_fn,
        root_state="Q: what is the answer?",
        n_iterations=30,
        expansion_width=3,
        c_puct=2.0,
    )

    assert isinstance(search_root, MCTSNode)
    assert search_root.N == 30, f"root should be visited once per iteration, got {search_root.N}"
    assert len(search_root.children) > 0, "root should have been expanded"

    # Extract best path by following max-Q children (as the chapter suggests).
    node = search_root
    path_states = [node.state]
    while not node.is_leaf():
        node = max(node.children, key=lambda c: c.Q)
        path_states.append(node.state)
    assert len(path_states) >= 2, "search should have descended past the root"
    print("MCTS best path depth:", len(path_states))
    print("MCTS best leaf value estimate (Q):", node.Q)

    # --- Exercise the EXAMPLE_REASONING_TRACE constant ---
    assert isinstance(EXAMPLE_REASONING_TRACE, str)
    assert "<think>" in EXAMPLE_REASONING_TRACE and "</think>" in EXAMPLE_REASONING_TRACE
    assert "72" in EXAMPLE_REASONING_TRACE
    think_start = EXAMPLE_REASONING_TRACE.index("<think>")
    think_end = EXAMPLE_REASONING_TRACE.index("</think>")
    assert think_start < think_end, "reasoning trace must open <think> before closing it"

    print("All block #5 (MCTS) and #6 (reasoning trace) checks passed.")


if __name__ == "__main__":
    main()
