"""
Runs the CPU-runnable Python blocks from:
content/13-interview-prep/08-ten-day-study-plan.md

Block #0 (line ~35): minimal spaced-repetition scheduler (simplified SM-2) - Card/due_cards + demo.
Block #1 (line ~145): numerically-stable sigmoid + BCE loss and gradient from scratch.
Block #2 (line ~178): causal self-attention (single head) from scratch in torch.
Block #3 (line ~257): DPO loss in ~10 lines.

All four blocks are copied faithfully from the chapter. Glue code (marked GLUE) instantiates
and calls each one with tiny CPU-only inputs so every block actually executes.
"""

import os
import tempfile

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Block #0 (line ~35, 73 lines): minimal spaced-repetition scheduler (SM-2)
# ---------------------------------------------------------------------------
"""
A minimal spaced-repetition scheduler (simplified SM-2).

Each card has:
  - ease (E): how "easy" the card is; scales the interval. Starts at 2.5.
  - interval (I): days until the next review.
  - reps: number of consecutive successful recalls.

You grade each recall 0..5 (0 = blackout, 5 = perfect). A grade >= 3 is a pass.
On a pass, the interval grows multiplicatively by the ease factor, so well-known
cards rapidly drift to weeks apart while struggling cards stay daily.
"""
from dataclasses import dataclass, field
from datetime import date, timedelta
import json


@dataclass
class Card:
    front: str
    back: str
    ease: float = 2.5          # E-factor; SM-2 floor is 1.3
    interval: int = 0          # days
    reps: int = 0
    due: date = field(default_factory=date.today)

    def review(self, grade: int, today: date) -> None:
        """Update scheduling state given a recall grade in 0..5."""
        if grade < 3:
            # Failed: reset the streak, see it again tomorrow.
            self.reps = 0
            self.interval = 1
        else:
            # Passed: grow the interval.
            if self.reps == 0:
                self.interval = 1
            elif self.reps == 1:
                self.interval = 6
            else:
                self.interval = round(self.interval * self.ease)
            self.reps += 1

        # Update ease. Hard passes (grade 3) shrink ease; perfect passes (5) keep it.
        # This is the classic SM-2 update; ease never drops below 1.3.
        self.ease = max(1.3, self.ease + (0.1 - (5 - grade) * (0.08 + (5 - grade) * 0.02)))
        self.due = today + timedelta(days=self.interval)


def due_cards(deck, today):
    """Return cards due today or earlier, hardest-first (lowest ease)."""
    return sorted([c for c in deck if c.due <= today], key=lambda c: c.ease)


# --- Example session -------------------------------------------------------
# GLUE: the book's own `if __name__ == "__main__":` block, run directly here
# (rather than gated on __name__) so it actually executes as part of the test.
# Persist to a scratch/tmp dir instead of the repo's cwd.
deck = [
    Card("Chinchilla optimal tokens-per-param?", "~20 tokens / parameter"),
    Card("Why sqrt(d_k) in attention?", "Keeps logit variance ~1 so softmax "
         "doesn't saturate; dot product of d_k unit-var dims has variance d_k."),
    Card("LoRA: what is delta W?", "B @ A, with A in R^{r x d}, B in R^{d x r}, "
         "r << d; only A,B are trained, W0 frozen."),
]
today = date.today()
for card in due_cards(deck, today):
    print("Q:", card.front)
    # In real use you'd prompt for input(); here we simulate a confident pass.
    card.review(grade=5, today=today)
    print("   ->", card.back, f"(next in {card.interval}d)\n")

# Persist so the schedule survives across the 10 days.
with tempfile.TemporaryDirectory() as _tmpdir:
    _deck_path = os.path.join(_tmpdir, "deck.json")
    with open(_deck_path, "w") as f:
        json.dump([{**c.__dict__, "due": c.due.isoformat()} for c in deck], f, indent=2)
    assert os.path.exists(_deck_path)

assert all(c.reps == 1 for c in deck), "every card should have been reviewed once"
assert all(c.interval == 1 for c in deck), "first pass -> interval should be 1 day"
print("[block 0] SM-2 scheduler: OK\n")


# ---------------------------------------------------------------------------
# Block #1 (line ~145, 18 lines): sigmoid + BCE loss and gradient from scratch
# ---------------------------------------------------------------------------
def sigmoid(z):
    # Numerically stable sigmoid: avoid exp overflow for large |z|.
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


def bce_loss_and_grad(X, y, w, b):
    z = X @ w + b
    p = sigmoid(z)
    eps = 1e-12                                   # guard log(0)
    loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    # dL/dz = p - y  (the clean result you derive on the whiteboard)
    dz = (p - y) / len(y)
    grad_w = X.T @ dz
    grad_b = dz.sum()
    return loss, grad_w, grad_b


# GLUE: tiny toy binary-classification batch, checked against a numeric gradient.
rng = np.random.default_rng(0)
X_toy = rng.normal(size=(8, 3))
y_toy = (rng.uniform(size=8) > 0.5).astype(np.float64)
w_toy = rng.normal(size=3)
b_toy = 0.1

loss0, gw, gb = bce_loss_and_grad(X_toy, y_toy, w_toy, b_toy)
assert np.isfinite(loss0)
assert gw.shape == w_toy.shape

# One gradient-descent step should not increase the loss (sanity check the sign/scale).
loss1, _, _ = bce_loss_and_grad(X_toy, y_toy, w_toy - 0.5 * gw, b_toy - 0.5 * gb)
assert loss1 < loss0, f"gradient step should decrease loss: {loss1} vs {loss0}"

# Finite-difference check on grad_b against the analytic gradient.
eps_fd = 1e-6
loss_plus, _, _ = bce_loss_and_grad(X_toy, y_toy, w_toy, b_toy + eps_fd)
loss_minus, _, _ = bce_loss_and_grad(X_toy, y_toy, w_toy, b_toy - eps_fd)
numeric_gb = (loss_plus - loss_minus) / (2 * eps_fd)
assert abs(numeric_gb - gb) < 1e-4, f"grad_b mismatch: analytic={gb} numeric={numeric_gb}"
print(f"[block 1] BCE loss/grad: OK (loss={loss0:.4f} -> {loss1:.4f} after one GD step)\n")


# ---------------------------------------------------------------------------
# Block #2 (line ~178, 27 lines): causal self-attention (single head)
# ---------------------------------------------------------------------------
import torch.nn.functional as F


def causal_self_attention(x, W_q, W_k, W_v, W_o):
    """
    x: (B, T, d_model). One head, for clarity.
    Returns (B, T, d_model).
    """
    B, T, d = x.shape
    q = x @ W_q                       # (B, T, d_k)
    k = x @ W_k                       # (B, T, d_k)
    v = x @ W_v                       # (B, T, d_k)
    d_k = q.shape[-1]

    # Scaled scores. The 1/sqrt(d_k) keeps logit variance ~O(1) so softmax
    # doesn't saturate into a near-one-hot distribution with tiny gradients.
    scores = (q @ k.transpose(-2, -1)) / (d_k ** 0.5)   # (B, T, T)

    # Causal mask: position t may attend to <= t only. Set future to -inf
    # so softmax assigns them exactly zero weight.
    mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
    scores = scores.masked_fill(mask, float("-inf"))

    attn = F.softmax(scores, dim=-1)  # (B, T, T)
    out = attn @ v                    # (B, T, d_k)
    return out @ W_o                  # (B, T, d_model)


# GLUE: tiny (B, T, d_model) input, call the function, and sanity-check causality
# by verifying position 0's output only depends on x[:, 0, :].
torch.manual_seed(0)
B, T, d_model, d_k = 2, 5, 8, 8
x_toy = torch.randn(B, T, d_model)
W_q = torch.randn(d_model, d_k) * 0.1
W_k = torch.randn(d_model, d_k) * 0.1
W_v = torch.randn(d_model, d_k) * 0.1
W_o = torch.randn(d_k, d_model) * 0.1

out = causal_self_attention(x_toy, W_q, W_k, W_v, W_o)
assert out.shape == (B, T, d_model)
assert torch.isfinite(out).all()

# Perturb a *future* token (position 4) and confirm position 0's output is unchanged
# (this is exactly what the causal mask guarantees).
x_perturbed = x_toy.clone()
x_perturbed[:, 4, :] += 10.0
out_perturbed = causal_self_attention(x_perturbed, W_q, W_k, W_v, W_o)
assert torch.allclose(out[:, 0, :], out_perturbed[:, 0, :], atol=1e-6), \
    "causal mask violated: position 0 output changed when a future token changed"
print("[block 2] causal_self_attention: OK (causality verified)\n")


# ---------------------------------------------------------------------------
# Block #3 (line ~257, 14 lines): DPO loss
# ---------------------------------------------------------------------------
def dpo_loss(pi_logp_w, pi_logp_l, ref_logp_w, ref_logp_l, beta=0.1):
    """
    Each arg is a (batch,) tensor of *summed* token log-probs of a full
    completion under the policy (pi_) or frozen reference (ref_).
    The model implicitly *is* its own reward model: r(x,y) = beta * log(pi/ref).
    """
    pi_logratio  = pi_logp_w  - pi_logp_l     # how much the policy prefers w over l
    ref_logratio = ref_logp_w - ref_logp_l    # the reference's preference
    # Train the policy to prefer w more strongly than the reference does.
    logits = beta * (pi_logratio - ref_logratio)
    return -F.logsigmoid(logits).mean()


# GLUE: toy batch of summed log-probs for "preferred"/"dispreferred" completions.
torch.manual_seed(1)
batch = 6
ref_logp_w_toy = torch.randn(batch) * 2 - 5
ref_logp_l_toy = torch.randn(batch) * 2 - 5
# Policy has already learned to prefer w over l relative to the reference.
pi_logp_w_toy = ref_logp_w_toy + 0.5
pi_logp_l_toy = ref_logp_l_toy - 0.5

loss_pref = dpo_loss(pi_logp_w_toy, pi_logp_l_toy, ref_logp_w_toy, ref_logp_l_toy)
# If policy == reference exactly, logits are 0 everywhere -> loss == log(2).
loss_equal = dpo_loss(ref_logp_w_toy, ref_logp_l_toy, ref_logp_w_toy, ref_logp_l_toy)
assert torch.isfinite(loss_pref)
assert abs(loss_equal.item() - np.log(2)) < 1e-5
assert loss_pref.item() < loss_equal.item(), \
    "a policy that already prefers w over l should have lower DPO loss than pi==ref"
print(f"[block 3] dpo_loss: OK (loss_pref={loss_pref.item():.4f} < "
      f"loss_equal={loss_equal.item():.4f} = log 2)\n")

print("All 4 blocks from 13-interview-prep/08-ten-day-study-plan.md executed successfully.")
