"""
Runnable-code test for content/99-appendix/03-papers-reading-list.md

Block #0 (line ~235): Minimal GRPO advantage computation (from scratch, educational)
    - Self-contained: defines grpo_advantages() and calls it on an example
      8-completion reward vector, then prints the result.

Block #1: non-python (SKIP)
"""

import torch

# ---------------------------------------------------------------------------
# Block #0 (line ~235) — verbatim from the chapter
# ---------------------------------------------------------------------------

# Minimal GRPO advantage computation (from scratch, educational)
import torch

def grpo_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """
    Compute group-normalized advantages for a batch of completions.

    Args:
        rewards: Tensor of shape [G] — one scalar reward per completion
                 in the group for a single prompt.
    Returns:
        advantages: Tensor of shape [G] — zero-mean, unit-variance advantages.
    """
    # Group mean and std across the G completions for this prompt
    mu = rewards.mean()          # scalar
    sigma = rewards.std() + 1e-8 # small epsilon for numerical stability

    # Normalized advantage: each completion is compared to the group average
    # Positive means "this completion did better than the group median"
    advantages = (rewards - mu) / sigma  # shape [G]
    return advantages


# Example: 8 completions for one math problem
# Reward = 1.0 if final answer is correct, 0.0 otherwise
rewards = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
advantages = grpo_advantages(rewards)
# rewards: [1, 0, 1, 1, 0, 0, 1, 0]  → 4 correct out of 8
# mu = 0.5, sigma ≈ 0.535
# advantages ≈ [0.94, -0.94, 0.94, 0.94, -0.94, -0.94, 0.94, -0.94]
print(advantages)
# The four correct completions get positive advantage ~+0.94
# The four incorrect completions get negative advantage ~-0.94
# This signal trains the policy to increase probability of correct completions


# ---------------------------------------------------------------------------
# Block #1: non-python — SKIP(non-python): the chapter's other fenced block is
# not a Python code sample (it is prose/reference material), so there is
# nothing CPU-runnable to extract.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Sanity checks (mirroring the book's own worked example/comments)
# ---------------------------------------------------------------------------

def _run_checks():
    assert advantages.shape == (8,)
    # Zero mean up to floating point error
    assert abs(advantages.mean().item()) < 1e-5
    # Correct completions (reward=1) should have positive advantage,
    # incorrect completions (reward=0) should have negative advantage.
    correct_mask = rewards == 1.0
    assert torch.all(advantages[correct_mask] > 0)
    assert torch.all(advantages[~correct_mask] < 0)
    # Value should match the book's stated approximation (~0.94)
    assert torch.allclose(advantages[correct_mask], torch.full((4,), 0.9354), atol=0.01)
    print("All checks passed.")


if __name__ == "__main__":
    _run_checks()
