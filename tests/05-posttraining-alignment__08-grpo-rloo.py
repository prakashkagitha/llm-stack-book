"""
Runs the CPU-runnable Python block(s) from
content/05-posttraining-alignment/08-grpo-rloo.md.

Blocks in the chapter (per the task's heuristic scan):
  #0 (line ~75, 34 lines) - rloo_loss(): a minimal RLOO loss for a batch of
                             prompts each with `group_size` sampled responses.
                             Pure torch, no GPU, no network -- CPU-runnable.
                             TESTED below, code copied verbatim from the book.
  #1 - k3_kl(): a tiny standalone function (Schulman's k3 KL estimator).
       Pure torch, no GPU, no network -- trivially CPU-runnable, so it is
       copied verbatim and TESTED below (block #0 does not call it, but the
       "SKIP unless trivially CPU-safe" default means a self-contained pure-
       torch function like this is exercised, not skipped). We check the
       chapter's stated properties: >= 0, exactly 0 at pi_theta == pi_ref,
       and equal to rho - log(rho) - 1.
  #2 - The full GRPOish trainer: loads Qwen/Qwen2.5-0.5B-Instruct from the
       HF hub, calls .generate() on `cuda` if available. SKIP(needs-gpu) +
       SKIP(network): it downloads model weights from the HF hub and is
       written to run on a GPU. Left untouched/not executed.
  #3 - The "R1-Zero reward = ..." block is a ```text``` pseudocode fragment,
       not Python. SKIP(non-python).

So blocks #0 and #1 are assembled into a runnable module below. Block #0 is
copied verbatim from the chapter and then exercised with tiny CPU fixtures: 2
prompts x group_size=3 responses, a handful of response tokens each, and a
`.backward()` call to prove the loss is actually differentiable end-to-end
(the book's own point: gradients must flow through `seq_logprob`, not through
the detached advantage).
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Block #0 (content/05-posttraining-alignment/08-grpo-rloo.md, line ~75)
# Copied verbatim from the chapter.
# ---------------------------------------------------------------------------
def rloo_loss(logprobs_per_token, response_mask, rewards, group_size):
    """
    Minimal RLOO loss for one batch of prompts, each with `group_size` responses.

    logprobs_per_token : (B, T) log pi_theta(o_t | q, o_<t) for the SAMPLED tokens.
                         B = num_prompts * group_size, laid out so that every
                         contiguous block of `group_size` rows shares a prompt.
    response_mask      : (B, T) 1.0 for response tokens, 0.0 for prompt/padding.
    rewards            : (B,)   scalar terminal reward per response.
    group_size         : G, number of responses per prompt.
    Returns a scalar loss to call .backward() on.
    """
    B, T = logprobs_per_token.shape
    G = group_size
    n_prompts = B // G

    # Sequence log-prob = sum of per-token log-probs over response tokens.
    seq_logprob = (logprobs_per_token * response_mask).sum(dim=-1)      # (B,)

    # Group rewards: (n_prompts, G)
    r = rewards.view(n_prompts, G)
    group_sum = r.sum(dim=1, keepdim=True)                             # (n_prompts, 1)
    # Leave-one-out baseline: (sum of others) / (G - 1)
    loo_baseline = (group_sum - r) / (G - 1)                           # (n_prompts, G)
    advantage = (r - loo_baseline).view(B)                            # (B,)

    # Policy-gradient loss. We MAXIMIZE advantage * logprob, so MINIMIZE its negative.
    # advantage is detached: it is a weight, not something we differentiate through.
    loss = -(advantage.detach() * seq_logprob).mean()
    return loss


# ---------------------------------------------------------------------------
# Block #1 (content/05-posttraining-alignment/08-grpo-rloo.md, line ~172)
# Copied verbatim from the chapter. Pure torch, CPU-runnable, so TESTED (not
# skipped) below despite not being called by block #0.
# ---------------------------------------------------------------------------
def k3_kl(logprob_policy, logprob_ref):
    """
    Schulman's k3 estimator of KL(pi_theta || pi_ref), per token, always >= 0.
    Inputs are log pi(o_t | ...) for the SAMPLED token o_t under each model.
    """
    log_ratio = logprob_ref - logprob_policy        # log(pi_ref / pi_theta)
    ratio = log_ratio.exp()                         # pi_ref / pi_theta
    return ratio - log_ratio - 1.0                  # rho - log rho - 1


# ---------------------------------------------------------------------------
# GLUE: tiny CPU fixtures + demo driver (not in the book; minimal harness
# to actually execute rloo_loss and check its documented properties).
# ---------------------------------------------------------------------------
def _make_fixture(seed=0):
    torch.manual_seed(seed)
    n_prompts, G, T = 2, 3, 5
    B = n_prompts * G

    # Fake per-token log-probs, requiring grad so we can call .backward().
    logprobs_per_token = (-torch.rand(B, T, dtype=torch.float32) - 0.1).requires_grad_(True)

    # Response mask: first 2 tokens are "prompt" (masked out), last 3 are "response".
    response_mask = torch.zeros(B, T, dtype=torch.float32)
    response_mask[:, 2:] = 1.0

    # Rewards: within each group of G=3, vary so the group is non-degenerate.
    rewards = torch.tensor([1.0, 0.0, 0.5, 0.2, 0.2, 0.8], dtype=torch.float32)
    assert rewards.shape[0] == B

    return logprobs_per_token, response_mask, rewards, G, n_prompts


def main():
    logprobs_per_token, response_mask, rewards, G, n_prompts = _make_fixture()

    loss = rloo_loss(logprobs_per_token, response_mask, rewards, group_size=G)
    assert loss.dim() == 0, "rloo_loss must return a scalar"
    assert torch.isfinite(loss), "loss must be finite"

    # Gradients should flow through seq_logprob (hence logprobs_per_token),
    # exactly as the book states, and NOT depend on rewards (advantage is
    # detached).
    loss.backward()
    assert logprobs_per_token.grad is not None
    assert torch.isfinite(logprobs_per_token.grad).all()
    # Gradient should be exactly zero on masked-out (prompt) positions, since
    # response_mask zeroes those tokens out of seq_logprob before the loss.
    assert torch.allclose(
        logprobs_per_token.grad[:, :2], torch.zeros(logprobs_per_token.shape[0], 2)
    ), "no gradient should flow to masked (prompt) positions"
    # And nonzero on the response positions (advantages are non-degenerate here).
    assert logprobs_per_token.grad[:, 2:].abs().sum() > 0

    # Cross-check the book's identity: A_i = R_i - loo_baseline_i =
    # (G/(G-1)) * (R_i - group_mean_i). Recompute independently and compare
    # against the loss the function produced.
    r = rewards.view(n_prompts, G)
    group_mean = r.mean(dim=1, keepdim=True)
    advantage_identity = (G / (G - 1)) * (r - group_mean)
    group_sum = r.sum(dim=1, keepdim=True)
    loo_baseline = (group_sum - r) / (G - 1)
    advantage_direct = r - loo_baseline
    assert torch.allclose(advantage_identity, advantage_direct, atol=1e-6), (
        "RLOO leave-one-out baseline must equal the G/(G-1)*(R - mean) identity "
        "stated in the chapter"
    )

    with torch.no_grad():
        seq_logprob = (logprobs_per_token * response_mask).sum(dim=-1)
        expected_loss = -(advantage_direct.view(-1).detach() * seq_logprob).mean()
    assert torch.allclose(loss.detach(), expected_loss, atol=1e-6)

    # Sanity: a fully degenerate group (all rewards equal) must contribute
    # exactly zero advantage -- the "dead group" property the chapter
    # describes for GRPO applies identically to RLOO's baseline.
    degenerate_rewards = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    T_local = logprobs_per_token.shape[1]
    logprobs2 = (-torch.rand(6, T_local, dtype=torch.float32) - 0.1).requires_grad_(True)
    mask2 = torch.zeros(6, T_local, dtype=torch.float32)
    mask2[:, 2:] = 1.0
    dead_loss = rloo_loss(logprobs2, mask2, degenerate_rewards, group_size=G)
    dead_loss.backward()
    assert torch.allclose(
        logprobs2.grad, torch.zeros_like(logprobs2.grad), atol=1e-7
    ), "an all-equal-reward group must produce exactly zero gradient (dead group)"

    # -----------------------------------------------------------------------
    # Block #1 (k3_kl): the chapter's claimed properties -- always >= 0, equals
    # 0 exactly when pi_theta == pi_ref, and matches rho - log(rho) - 1.
    # -----------------------------------------------------------------------
    torch.manual_seed(1)
    lp_policy = -torch.rand(64) - 0.05
    lp_ref = -torch.rand(64) - 0.05
    kl = k3_kl(lp_policy, lp_ref)
    assert torch.isfinite(kl).all()
    assert (kl >= -1e-6).all(), "k3 estimator must be non-negative (rho - log rho - 1 >= 0)"
    # Exact zero when the two models agree on the sampled token.
    same = k3_kl(lp_policy, lp_policy)
    assert torch.allclose(same, torch.zeros_like(same), atol=1e-6), (
        "k3 KL must be exactly 0 when pi_theta == pi_ref"
    )
    # Cross-check against the closed form rho - log(rho) - 1 with rho = pi_ref/pi_theta.
    rho = (lp_ref - lp_policy).exp()
    expected_kl = rho - (lp_ref - lp_policy) - 1.0
    assert torch.allclose(kl, expected_kl, atol=1e-6)
    print("block #1 (k3_kl): OK")
    print("  non-negative, zero at equality, matches rho - log rho - 1: OK")

    print("block #0 (rloo_loss): OK")
    print(f"  loss = {loss.item():.6f}")
    print(f"  advantage (per response) = {advantage_direct.view(-1).tolist()}")
    print("  masked-position grad is exactly zero: OK")
    print("  RLOO G/(G-1)*(R-mean) identity matches direct leave-one-out: OK")
    print("  degenerate (all-equal-reward) group produces zero gradient: OK")


if __name__ == "__main__":
    main()
    print("\nAll checks passed.")
