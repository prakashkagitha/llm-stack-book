"""
Runnable-code test for content/00-frontmatter/00-preface.md

Block #0 (line ~36): the three-line LM loss snippet (import torch,
import torch.nn.functional as F, and the lm_loss function).
"""

import torch
import torch.nn.functional as F


# ---- Block #0 (verbatim from the chapter) ----------------------------
# logits: (batch, seq_len, vocab)   targets: (batch, seq_len) of token ids
def lm_loss(logits, targets):
    # shift so that position t predicts token t+1
    logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
    targets = targets[:, 1:].reshape(-1)
    return F.cross_entropy(logits, targets)   # mean negative log-likelihood
# ------------------------------------------------------------------------


def test_lm_loss():
    torch.manual_seed(0)
    batch, seq_len, vocab = 2, 5, 17

    logits = torch.randn(batch, seq_len, vocab)
    targets = torch.randint(0, vocab, (batch, seq_len))

    loss = lm_loss(logits, targets)

    assert loss.dim() == 0, "lm_loss should return a scalar (mean) loss"
    assert torch.isfinite(loss), "loss should be finite"
    assert loss.item() > 0, "cross-entropy loss on random logits should be positive"

    # Sanity check: manually compute the same shifted cross-entropy and
    # confirm lm_loss matches it exactly.
    shifted_logits = logits[:, :-1, :].reshape(-1, vocab)
    shifted_targets = targets[:, 1:].reshape(-1)
    expected = F.cross_entropy(shifted_logits, shifted_targets)
    assert torch.allclose(loss, expected)

    print(f"lm_loss OK: loss={loss.item():.4f}")


if __name__ == "__main__":
    test_lm_loss()
    print("All preface blocks executed successfully.")
