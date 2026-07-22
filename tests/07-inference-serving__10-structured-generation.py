"""
Runnable-code check for content/07-inference-serving/10-structured-generation.md

Chapter has 8 fenced code blocks (0-indexed):
  #0  (line ~109)  ```text  PDA stack-state example              -> SKIP: not Python
  #1  (line ~138)  ```python build_fsm_index (Outlines-style)    -> TESTED
  #2  (line ~206)  ```       mask = precomputed | evaluate(...)  -> SKIP: not Python (pseudocode line)
  #3  (line ~247)  ```python regex_sampler.py                    -> SKIP(network/gpu): the __main__ demo
                      downloads gpt2 from the HF hub via
                      `transformers.GPT2Tokenizer.from_pretrained` / `GPT2LMHeadModel.from_pretrained`
                      and runs a real generation loop -- forbidden network call, no CI download.
                      We DO reuse its tiny pure-tensor helper `apply_mask_to_logits` verbatim below
                      because block #7's `safe_apply_mask` depends on that name (same relationship
                      the book text has between the two blocks).
  #4  (line ~498)  ```python outlines_demo.py                    -> SKIP(network/gpu): requires the
                      `outlines` package and `device="cuda"` plus a real Llama checkpoint download.
  #5  (line ~568)  ```       tool-call JSON shape fragment        -> SKIP: not Python (JSON fragment
                      inside a numbered list, not a runnable snippet)
  #6  (line ~677)  ```python is_token_valid_BUGGY / _CORRECT     -> TESTED (deliberately-broken demo)
  #7  (line ~705)  ```python safe_apply_mask                     -> TESTED
"""

import re
import warnings
from typing import Dict, Set, List, Tuple

import torch


# =============================================================================
# Block #1 (line ~138): Outlines-style FSM index construction
# Verbatim from the book, except `regex_to_dfa` -- which the book itself calls
# "conceptual pseudocode" / "standard automaton ops" and never implements --
# is supplied here as a tiny fixture DFA so the block's own loop logic (the
# thing actually being taught) executes for real.
# =============================================================================

def regex_to_dfa(regex_pattern: str):
    """
    Test fixture standing in for the book's unimplemented `regex_to_dfa`.
    Builds a fixed toy DFA that accepts exactly two decimal digits
    (states: 0=start -> 1=one digit seen -> 2=accept, no transitions out of 2).
    `regex_pattern` is accepted for interface compatibility but not parsed --
    the book's own code never specifies what `regex_to_dfa` should look like
    beyond this (states, transitions, start, accepts) shape.
    """

    class _DFA:
        pass

    dfa = _DFA()
    dfa.states = [0, 1, 2]
    dfa.start = 0
    dfa.accepts = {2}
    transitions = {}
    for d in "0123456789":
        transitions[(0, d)] = 1
        transitions[(1, d)] = 2
    dfa.transitions = transitions
    return dfa


# --- verbatim from the chapter (block #1, line ~138) -----------------------

def build_fsm_index(
    regex_pattern: str,
    vocab: Dict[int, str],    # token_id -> decoded string
) -> Dict[int, Set[int]]:
    """
    Returns a dict: fsm_state -> set of valid token_ids
    Pre-computes the full transition table for constrained decoding.
    """
    # Step 1: compile regex to NFA, convert to DFA (standard automaton ops)
    fsm = regex_to_dfa(regex_pattern)  # returns (states, transitions, start, accepts)

    index: Dict[int, Set[int]] = {}
    dead_state = -1

    for state in fsm.states:
        valid_tokens: Set[int] = set()
        for token_id, token_str in vocab.items():
            # Simulate the DFA over the token string, starting from `state`
            current = state
            reachable = True
            for char in token_str:
                next_state = fsm.transitions.get((current, char), dead_state)
                if next_state == dead_state:
                    reachable = False
                    break
                current = next_state

            if reachable:
                # Token is valid: consuming it does not kill the FSM
                valid_tokens.add(token_id)

        index[state] = valid_tokens

    return index

# --- end verbatim block #1 --------------------------------------------------


def test_build_fsm_index():
    # Tiny toy vocabulary mixing digit and non-digit tokens.
    vocab = {
        0: "1",
        1: "9",
        2: "ab",
        3: "12",
        4: "5",
    }

    index = build_fsm_index("[0-9]{2}", vocab)

    # From the start state, any token beginning with a digit is "reachable"
    # (does not immediately die), including single digits and "12".
    assert index[0] == {0, 1, 3, 4}, index[0]
    # After one digit (state 1), only tokens whose *every* char keeps the DFA
    # alive survive: "ab" dies immediately, "12" dies on its 2nd char because
    # the accept state (2) has no outgoing transitions in our toy DFA.
    assert index[1] == {0, 1, 4}, index[1]
    # The accept state has no outgoing transitions at all in this toy DFA, so
    # every non-empty token dies on its first character.
    assert index[2] == set(), index[2]

    print("[block #1] build_fsm_index: index =", index, "-> OK")


# =============================================================================
# Dependency needed by block #7: `apply_mask_to_logits` is defined in block #3
# (regex_sampler.py), which is SKIPPED above because its __main__ section
# performs a real network download of gpt2. The helper itself is a tiny,
# network-free, GPU-free pure-tensor function, copied verbatim so that block
# #7's `safe_apply_mask` -- which calls it by name in the book text -- has
# something real to call, exactly mirroring the chapter's own block-to-block
# dependency.
# =============================================================================

def apply_mask_to_logits(
    logits: torch.Tensor,           # shape [vocab_size] or [batch, vocab_size]
    mask: torch.Tensor,             # bool tensor, same last dim as logits
) -> torch.Tensor:
    """
    Sets logits of forbidden tokens to -inf. Operates in-place on a clone.
    """
    masked = logits.clone()
    masked[~mask] = float('-inf')
    return masked


# =============================================================================
# Block #6 (line ~677): tokenization-boundary bug illustration
# Verbatim from the book. `is_token_valid_BUGGY` is explicitly presented as
# broken ("BUG", "WRONG") -- we do not fix it, we only demonstrate that it
# misbehaves exactly as the surrounding prose describes, and that the
# CORRECT version does not share the flaw.
# =============================================================================

# --- verbatim from the chapter (block #6, line ~677) ------------------------

def is_token_valid_BUGGY(fsm_state, token_str, fsm):
    """BUG: only checks if the token_str matches as a prefix of a word,
    not whether each character advances the FSM correctly."""
    return token_str.startswith(fsm.expected_prefix)   # WRONG

def is_token_valid_CORRECT(fsm_state, token_str, fsm):
    """Simulates FSM transitions for every character in the token."""
    current = fsm_state
    for char in token_str:
        current = fsm.transition(current, char)
        if current is None:      # dead state
            return False
    return True  # reached a live (non-dead) state; token is valid

# --- end verbatim block #6 ---------------------------------------------------


class _ToyFSM:
    """
    Minimal fixture FSM accepting exactly the string "ab":
    state 0 --'a'--> state 1 --'b'--> state 2 (accept); anything else -> dead (None).
    `expected_prefix` is what the buggy checker naively compares against.
    """
    expected_prefix = "a"

    def transition(self, state, char):
        if state == 0 and char == "a":
            return 1
        if state == 1 and char == "b":
            return 2
        return None


def test_tokenization_boundary_bug():
    fsm = _ToyFSM()

    # A genuinely valid token: both checkers should agree it's valid.
    assert is_token_valid_CORRECT(0, "ab", fsm) is True
    assert is_token_valid_BUGGY(0, "ab", fsm) is True

    # The book's failure scenario: token "ax" starts with the expected
    # prefix "a" but its 2nd character ('x') is not accepted after 'a'.
    # The buggy checker wrongly says it's valid; the correct one rejects it.
    assert is_token_valid_BUGGY(0, "ax", fsm) is True   # WRONG answer, as documented
    assert is_token_valid_CORRECT(0, "ax", fsm) is False  # correct rejection

    print("[block #6] is_token_valid_BUGGY vs _CORRECT: bug reproduced as documented -> OK")


# =============================================================================
# Block #7 (line ~705): safe fallback when the constraint mask is all-zero
# Verbatim from the book.
# =============================================================================

# --- verbatim from the chapter (block #7, line ~705) -------------------------

def safe_apply_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply mask with fallback if all tokens are forbidden."""
    if not mask.any():
        # All tokens forbidden: grammar is unsatisfiable at this point.
        # Log a warning and fall back to unconstrained sampling.
        import warnings
        warnings.warn(
            "Constrained generation: no valid tokens in current FSM state. "
            "Falling back to unconstrained sampling for this step.",
            RuntimeWarning,
            stacklevel=2,
        )
        return logits   # Return unmasked logits
    return apply_mask_to_logits(logits, mask)

# --- end verbatim block #7 ----------------------------------------------------


def test_safe_apply_mask():
    logits = torch.tensor([1.0, 2.0, 3.0])

    # Normal case: mask has at least one allowed token -> masking applied.
    mask = torch.tensor([True, False, True])
    out = safe_apply_mask(logits, mask)
    assert out[0].item() == 1.0
    assert out[1].item() == float('-inf')
    assert out[2].item() == 3.0

    # Pathological case: grammar unsatisfiable at this step -> all-zero mask.
    empty_mask = torch.zeros(3, dtype=torch.bool)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out2 = safe_apply_mask(logits, empty_mask)
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)
    assert torch.equal(out2, logits)  # fell back to unmasked logits

    print("[block #7] safe_apply_mask: normal + all-zero fallback -> OK")


if __name__ == "__main__":
    test_build_fsm_index()
    test_tokenization_boundary_bug()
    test_safe_apply_mask()
    print("\nAll CPU-runnable blocks in 07-inference-serving/10-structured-generation.md executed successfully.")
