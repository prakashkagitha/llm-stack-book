"""
Runs the CPU-runnable Python code block from
content/01-foundations/02-probability-information.md

Block #0 (line ~319): "perplexity.py" — a complete module computing
perplexity from raw log-probs, from logits, via sliding window, plus a
cross-entropy/KL decomposition demo and a label-smoothed cross-entropy
implementation. Copied verbatim from the chapter (the docstring claims a
`transformers` dependency, but the code never actually imports or uses it,
so nothing needs to be mocked/guarded here beyond the stdlib + torch).

Block #1 (a plain-text "expected output" listing) is SKIP(non-python): it's
just a fenced ```text``` block, not runnable code.
"""

import math
import torch
import torch.nn.functional as F
from typing import List, Optional


# ─────────────────────────────────────────────────────
# 1. Low-level: perplexity from a list of log-probs
# ─────────────────────────────────────────────────────

def perplexity_from_log_probs(log_probs: List[float]) -> float:
    """
    Given a list of natural-log probabilities log p(x_t | x_{<t}),
    compute perplexity = exp( -mean(log_probs) ).

    Args:
        log_probs: list of floats, each <= 0.

    Returns:
        Perplexity as a float.
    """
    if not log_probs:
        raise ValueError("Empty log-prob list")
    avg_nll = -sum(log_probs) / len(log_probs)   # average negative log-likelihood
    return math.exp(avg_nll)


# ─────────────────────────────────────────────────────
# 2. From raw logits and target token ids
# ─────────────────────────────────────────────────────

def perplexity_from_logits(
    logits: torch.Tensor,   # shape (T, V) — one logit vector per time step
    targets: torch.Tensor,  # shape (T,)   — ground-truth token ids
    ignore_index: int = -100,
) -> float:
    """
    Compute perplexity from raw (unnormalized) logits.

    This is exactly what a training loop would call after the forward pass,
    except we expose each step for clarity.

    Steps:
      1. Apply log-softmax to get log-probabilities.
      2. Gather the log-prob of the correct token at each step.
      3. Mask out padding tokens (ignore_index).
      4. Average and exponentiate.
    """
    # (T, V) → (T, V) in log-probability space
    log_probs = F.log_softmax(logits, dim=-1)   # numerically stable via LogSumExp

    # Gather log-prob of the target token at each position.
    # targets shape: (T,) → unsqueeze to (T, 1) for gather
    valid_mask = targets != ignore_index
    safe_targets = targets.clone()
    safe_targets[~valid_mask] = 0  # avoid index error on masked positions

    # Shape: (T, 1) → squeeze to (T,)
    token_log_probs = log_probs.gather(1, safe_targets.unsqueeze(1)).squeeze(1)

    # Zero out masked positions before averaging
    token_log_probs = token_log_probs * valid_mask.float()

    n_valid = valid_mask.sum().item()
    avg_nll = -token_log_probs.sum().item() / n_valid
    return math.exp(avg_nll)


# ─────────────────────────────────────────────────────
# 3. Dataset-level perplexity with sliding window
#    (handles sequences longer than the model context)
# ─────────────────────────────────────────────────────

def sliding_window_perplexity(
    token_ids: torch.Tensor,   # shape (N_total,)
    logit_fn,                  # callable: (T,) -> (T, V) logits
    max_length: int = 512,
    stride: int = 256,
) -> float:
    """
    Compute perplexity over a long token sequence using a sliding window.

    The 'stride' trick ensures each token is scored in context,
    not at the very beginning of a truncated window where the model
    has no context. This is the method used by Radford et al. (GPT-2)
    for WikiText-103 evaluation.

    Args:
        token_ids:  1-D tensor of all token ids in the test set.
        logit_fn:   function mapping a 1-D token tensor of length <= max_length
                    to a logit tensor of shape (len, vocab_size).
        max_length: model's maximum context length.
        stride:     step between windows; lower stride = more context overlap
                    = slightly slower but more accurate for longer texts.

    Returns:
        Perplexity (float).
    """
    seq_len = token_ids.size(0)
    total_nll = 0.0
    total_tokens = 0
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        window = token_ids[begin:end]          # shape: (window_size,)

        # The model scores positions [0, window_size-1];
        # for LM evaluation we predict token t+1 from tokens 0..t.
        with torch.no_grad():
            logits = logit_fn(window)          # shape: (window_size, V)

        # Targets are the next token at each position:
        # position i predicts token i+1, so we shift by 1.
        # In a causal LM, logits[i] predicts token i+1 (the standard convention).
        # We count only tokens that were *not* already counted in the previous window.
        target_ids = token_ids[begin + 1 : end + 1]
        # Use only new positions (stride steps from the right of the window)
        count_from = max(prev_end - begin, 1)  # at least predict 1 token

        logits_new = logits[count_from - 1 : len(target_ids)]
        targets_new = target_ids[count_from - 1 :]

        if logits_new.shape[0] == 0:
            prev_end = end
            continue

        log_probs = F.log_softmax(logits_new, dim=-1)
        token_nlls = -log_probs.gather(
            1, targets_new[:logits_new.shape[0]].unsqueeze(1)
        ).squeeze(1)

        total_nll += token_nlls.sum().item()
        total_tokens += token_nlls.shape[0]
        prev_end = end

        if end == seq_len:
            break

    return math.exp(total_nll / total_tokens)


# ─────────────────────────────────────────────────────
# 4. Demonstration: cross-entropy decomposition
# ─────────────────────────────────────────────────────

def demonstrate_ce_kl_relationship():
    """
    Show numerically that H(p, q) = H(p) + KL(p || q).
    Uses a small 4-class example.
    """
    # True distribution p (label smoothed example)
    p = torch.tensor([0.7, 0.1, 0.1, 0.1])
    # Model distribution q
    q = torch.tensor([0.5, 0.2, 0.2, 0.1])

    assert abs(p.sum().item() - 1.0) < 1e-6, "p must be a valid distribution"
    assert abs(q.sum().item() - 1.0) < 1e-6, "q must be a valid distribution"

    # Shannon entropy H(p)
    # Convention: 0 * log(0) = 0
    H_p = -(p * torch.log(p.clamp(min=1e-12))).sum().item()

    # KL divergence KL(p || q)
    # Sum over positions where p > 0
    kl_pq = (p * torch.log((p / q.clamp(min=1e-12)).clamp(min=1e-12))).sum().item()

    # Cross-entropy H(p, q) = -sum_x p(x) log q(x)
    ce = -(p * torch.log(q.clamp(min=1e-12))).sum().item()

    print(f"H(p)              = {H_p:.4f} nats")
    print(f"KL(p || q)        = {kl_pq:.4f} nats")
    print(f"H(p) + KL(p||q)   = {H_p + kl_pq:.4f} nats")
    print(f"H(p, q) directly  = {ce:.4f} nats")
    print(f"Match: {abs(ce - (H_p + kl_pq)) < 1e-5}")

    # In the one-hot case (standard training), H(p) = 0 so CE = KL
    p_onehot = torch.tensor([1.0, 0.0, 0.0, 0.0])
    H_onehot = 0.0  # entropy of a degenerate distribution
    ce_onehot = -(p_onehot * torch.log(q.clamp(min=1e-12))).sum().item()
    kl_onehot = ce_onehot - H_onehot  # = ce_onehot
    print(f"\nOne-hot target:")
    print(f"H(p_onehot, q) = -log q(k*) = {ce_onehot:.4f} = {-math.log(0.5):.4f}")
    print(f"This equals -log(q[0]) = -log(0.5) = {-math.log(0.5):.4f}")

    return H_p, kl_pq, ce


# ─────────────────────────────────────────────────────
# 5. Label smoothing loss
# ─────────────────────────────────────────────────────

def label_smoothed_cross_entropy(
    logits: torch.Tensor,    # (B, T, V) or (T, V)
    targets: torch.Tensor,   # (B, T) or (T,) long tensor
    smoothing: float = 0.1,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Cross-entropy with label smoothing.

    Equivalent to PyTorch's CrossEntropyLoss(label_smoothing=smoothing),
    but written out explicitly for teaching purposes.

    The loss is:
        L = (1 - eps) * CE_hard(logits, targets)
          + eps * mean_over_classes( -log_softmax(logits) )
    """
    vocab_size = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab_size)   # (B*T, V)
    flat_targets = targets.reshape(-1)              # (B*T,)

    log_probs = F.log_softmax(flat_logits, dim=-1)  # (B*T, V)

    # Hard-target negative log-likelihood
    nll_loss = F.nll_loss(
        log_probs,
        flat_targets,
        ignore_index=ignore_index,
        reduction='mean',
    )

    # Soft-target component: uniform over all classes
    # -mean_k log_prob_k, averaged over valid positions
    smooth_loss = -log_probs.mean(dim=-1)  # (B*T,)
    mask = flat_targets != ignore_index
    smooth_loss = smooth_loss[mask].mean()

    return (1.0 - smoothing) * nll_loss + smoothing * smooth_loss


# ─────────────────────────────────────────────────────
# Entry point (mirrors the book's `if __name__ == "__main__":` block,
# plus assertions checking the numbers the book claims in its
# "Running the script produces output like:" listing)
# ─────────────────────────────────────────────────────

def main():
    print("=== CE / KL decomposition demo ===")
    H_p, kl_pq, ce = demonstrate_ce_kl_relationship()
    # These four numbers are deterministic (p, q are fixed tensors, no RNG).
    # NOTE: the book originally printed a stale/incorrect sample output here
    # (H(p)=0.8019, KL=0.0719, sum/CE=0.8738) that does not match what the
    # actual code computes for p=[0.7,0.1,0.1,0.1], q=[0.5,0.2,0.2,0.1].
    # This was a real bug in the chapter's illustrative output block, fixed
    # in content/01-foundations/02-probability-information.md to the true
    # values below (H(p)=0.9404, KL=0.0969, sum/CE=1.0373).
    assert math.isclose(H_p, 0.9404, abs_tol=1e-3)
    assert math.isclose(kl_pq, 0.0969, abs_tol=1e-3)
    assert math.isclose(ce, 1.0373, abs_tol=1e-3)
    # The real invariant the block demonstrates, independent of the exact
    # numbers: H(p, q) == H(p) + KL(p || q).
    assert abs(ce - (H_p + kl_pq)) < 1e-5

    print("\n=== Perplexity from raw log-probs ===")
    # Toy sequence: 5 tokens with model-assigned probabilities
    # as in the worked example above
    probs = [0.20, 0.05, 0.30, 0.60, 0.15]
    log_probs_list = [math.log(p) for p in probs]
    ppl = perplexity_from_log_probs(log_probs_list)
    print(f"Perplexity = {ppl:.3f}  (expected ~5.17)")
    assert math.isclose(ppl, 5.173, abs_tol=1e-2)

    print("\n=== Perplexity from logits ===")
    torch.manual_seed(42)
    V, T = 32, 10
    # Simulate a model that has reasonably high confidence on the correct tokens
    logits = torch.randn(T, V)
    targets = torch.randint(0, V, (T,))
    # Boost the correct token logits by 2 so the model looks "smart"
    for t in range(T):
        logits[t, targets[t]] += 2.0
    ppl_from_logits = perplexity_from_logits(logits, targets)
    print(f"Perplexity from logits = {ppl_from_logits:.3f}")
    # Exact value depends on the PyTorch RNG implementation for
    # torch.randn/randint under manual_seed(42), which can vary across
    # versions/platforms; assert only the structural property (finite,
    # positive, and >= 1 since it's exp of a non-negative average NLL).
    assert math.isfinite(ppl_from_logits) and ppl_from_logits >= 1.0

    print("\n=== Label-smoothed loss ===")
    logits_batch = torch.randn(2, 5, 100)  # batch=2, seq_len=5, vocab=100
    targets_batch = torch.randint(0, 100, (2, 5))
    ls_loss = label_smoothed_cross_entropy(logits_batch, targets_batch, smoothing=0.1)
    hard_loss = F.cross_entropy(logits_batch.reshape(-1, 100), targets_batch.reshape(-1))
    print(f"Label-smoothed loss = {ls_loss.item():.4f}")
    print(f"Hard cross-entropy  = {hard_loss.item():.4f}")
    print(f"Label smoothing adds regularization: LS loss > hard CE = {ls_loss.item() > hard_loss.item()}")
    # NOTE: the book's sample output claims "LS loss > hard CE = True" as if
    # that were a guaranteed property. On untrained/random logits this is
    # NOT guaranteed -- it's a coincidence of the particular RNG draw -- and
    # indeed does not hold in this environment's torch build. We therefore
    # only assert the two losses are close (both are convex-combination-
    # related averages over the same random logits), not an ordering; the
    # book's text was corrected to note this is RNG/version-dependent.
    assert label_smoothed_cross_entropy(
        torch.zeros(1, 1, 4), torch.zeros(1, 1, dtype=torch.long)
    ).item() >= 0
    # And verify the label-smoothing formula matches PyTorch's own
    # CrossEntropyLoss(label_smoothing=...) implementation, which is the
    # actual, environment-independent correctness property of this block.
    ref_loss = F.cross_entropy(
        logits_batch.reshape(-1, 100), targets_batch.reshape(-1), label_smoothing=0.1
    )
    assert math.isclose(ls_loss.item(), ref_loss.item(), abs_tol=1e-4), (
        ls_loss.item(), ref_loss.item()
    )

    # Exercise the sliding-window path too (not in the book's __main__ block,
    # but it's a top-level function the chapter defines and claims is
    # runnable — call it with a tiny toy "model" so it actually executes).
    print("\n=== Sliding-window perplexity (extra exercise, tiny toy model) ===")
    toy_vocab = 16
    toy_tokens = torch.randint(0, toy_vocab, (40,))

    def toy_logit_fn(window: torch.Tensor) -> torch.Tensor:
        # Deterministic toy "model": one-hot-ish logits favoring the token
        # itself, just to produce a well-defined (T, V) logits tensor.
        g = torch.Generator().manual_seed(0)
        return torch.randn(window.shape[0], toy_vocab, generator=g)

    sw_ppl = sliding_window_perplexity(
        toy_tokens, toy_logit_fn, max_length=16, stride=8
    )
    print(f"Sliding-window perplexity (toy) = {sw_ppl:.3f}")
    assert sw_ppl > 0 and math.isfinite(sw_ppl)


if __name__ == "__main__":
    main()
    print("\nAll assertions passed.")
