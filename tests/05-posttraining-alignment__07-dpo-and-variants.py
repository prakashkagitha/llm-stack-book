"""
Runnability test for content/05-posttraining-alignment/07-dpo-and-variants.md

Every fenced python block in this chapter is a complete, self-contained function.
All six are copied VERBATIM below and exercised end-to-end on tiny CPU fixtures:

  Block #0 (~line 133): sequence_logprob() + dpo_loss()   — canonical from-scratch DPO
  Block #1 (~line 186): dpo_training_step()               — reference-wiring glue
  Block #2 (~line 260): ipo_loss()                        — squared-error to 1/(2β)
  Block #3 (~line 284): kto_loss()                        — unpaired prospect-theory
  Block #4 (~line 321): orpo_loss()                       — SFT + odds-ratio, ref-free
  Block #5 (~line 359): simpo_loss()                      — length-norm, ref-free

No block is skipped: each function is called, gradients are checked, and the
DPO worked-example arithmetic from the chapter is reproduced numerically.
"""

import torch
import torch.nn.functional as F
from types import SimpleNamespace


# ============================================================================
# Block #0 (chapter line ~133) — verbatim from the book
# ============================================================================

def sequence_logprob(logits, labels, loss_mask):
    """
    Sum of per-token log p(label_t | context) over response tokens.

    logits    : (B, T, V) raw model logits for positions 0..T-1.
    labels    : (B, T)    token ids; for autoregressive LMs label at position t
                          is the token that should be predicted AT t (already shifted
                          by the caller, see below).
    loss_mask : (B, T)    1.0 for response tokens we score, 0.0 for prompt/padding.
    Returns   : (B,)      summed log-prob of each response.
    """
    logprobs = F.log_softmax(logits, dim=-1)                      # (B, T, V)
    # Gather the log-prob of the actual next token at each position.
    token_logp = torch.gather(
        logprobs, dim=2, index=labels.unsqueeze(-1)
    ).squeeze(-1)                                                 # (B, T)
    return (token_logp * loss_mask).sum(dim=-1)                   # (B,)


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps,   ref_rejected_logps,
             beta=0.1):
    """
    The DPO loss. Inputs are SEQUENCE log-probs (already summed over tokens),
    one scalar per example, for the policy and frozen reference on both responses.

    Returns
      loss          : scalar to call .backward() on
      chosen_reward : β·(logπθ(yw) − logπref(yw))   — the implicit reward of the winner
      reject_reward : β·(logπθ(yl) − logπref(yl))   — the implicit reward of the loser
    """
    # Log-ratios π_θ / π_ref for chosen and rejected responses.
    chosen_logratio   = policy_chosen_logps   - ref_chosen_logps      # (B,)
    rejected_logratio = policy_rejected_logps - ref_rejected_logps    # (B,)

    # Δ = β · (chosen_logratio − rejected_logratio).
    logits = beta * (chosen_logratio - rejected_logratio)            # (B,)

    # −log σ(Δ) == softplus(−Δ); softplus is numerically stable.
    loss = F.softplus(-logits).mean()

    # Implicit rewards, handy to LOG (detached — diagnostics only).
    chosen_reward = (beta * chosen_logratio).detach()
    reject_reward = (beta * rejected_logratio).detach()
    return loss, chosen_reward, reject_reward


# ============================================================================
# Block #1 (chapter line ~186) — verbatim from the book
# ============================================================================

def dpo_training_step(policy, ref_model, batch, beta=0.1):
    """
    batch contains, for chosen and rejected responses:
      *_input_ids (B, T), *_labels (B, T) [shifted], *_loss_mask (B, T).
    """
    def logps(model, ids, labels, mask, grad):
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            logits = model(ids).logits[:, :-1, :]    # predict next token
        return sequence_logprob(logits, labels[:, 1:], mask[:, 1:])

    # Policy: needs gradients. Two forward passes (chosen, rejected).
    pol_chosen = logps(policy, batch["chosen_input_ids"],
                       batch["chosen_labels"], batch["chosen_loss_mask"], grad=True)
    pol_reject = logps(policy, batch["rejected_input_ids"],
                       batch["rejected_labels"], batch["rejected_loss_mask"], grad=True)

    # Reference: frozen, no gradients (or precomputed & cached offline).
    ref_chosen = logps(ref_model, batch["chosen_input_ids"],
                       batch["chosen_labels"], batch["chosen_loss_mask"], grad=False)
    ref_reject = logps(ref_model, batch["rejected_input_ids"],
                       batch["rejected_labels"], batch["rejected_loss_mask"], grad=False)

    loss, r_chosen, r_reject = dpo_loss(pol_chosen, pol_reject,
                                        ref_chosen, ref_reject, beta=beta)

    # The single most useful training metric: preference accuracy =
    # fraction of pairs where implicit reward of chosen > rejected.
    reward_acc = (r_chosen > r_reject).float().mean()
    margin = (r_chosen - r_reject).mean()
    return loss, {"reward_acc": reward_acc.item(), "reward_margin": margin.item()}


# ============================================================================
# Block #2 (chapter line ~260) — verbatim from the book
# ============================================================================

def ipo_loss(pol_chosen, pol_reject, ref_chosen, ref_reject, beta=0.1):
    # h = (chosen log-ratio) − (rejected log-ratio); note: NO β multiplier inside.
    h = (pol_chosen - ref_chosen) - (pol_reject - ref_reject)
    # Regress h toward the finite target 1/(2β) instead of pushing it to +∞.
    return ((h - 1.0 / (2.0 * beta)) ** 2).mean()


# ============================================================================
# Block #3 (chapter line ~284) — verbatim from the book
# ============================================================================

def kto_loss(policy_logps, ref_logps, labels_desirable, kl_z0,
             beta=0.1, lam_desirable=1.0, lam_undesirable=1.0):
    """
    policy_logps, ref_logps : (B,) sequence log-probs (single, UNPAIRED responses).
    labels_desirable        : (B,) bool — True = thumbs up, False = thumbs down.
    kl_z0                    : scalar reference point z0 on the IMPLICIT-REWARD scale,
                              i.e. beta times the detached running-mean in-batch KL
                              (same units as r = beta * log(pi/pi_ref)). beta is
                              already folded into r and z0, so it is NOT reapplied
                              inside the sigmoid below.
    """
    r = beta * (policy_logps - ref_logps)                 # implicit reward, (B,)
    margin = r - kl_z0
    # Desirable: want margin > 0  → value = σ(margin); loss = λ_D · (1 − value).
    # Undesirable: want margin < 0 → value = σ(−margin); loss = λ_U · (1 − value).
    desirable_loss   = lam_desirable   * (1.0 - torch.sigmoid(margin))
    undesirable_loss = lam_undesirable * (1.0 - torch.sigmoid(-margin))
    loss = torch.where(labels_desirable, desirable_loss, undesirable_loss)
    return loss.mean()


# ============================================================================
# Block #4 (chapter line ~321) — verbatim from the book
# ============================================================================

def orpo_loss(pol_chosen_logps, pol_rejected_logps,
              chosen_len, rejected_len, chosen_nll, lam=0.1):
    """
    pol_chosen_logps / pol_rejected_logps : (B,) SEQUENCE log-probs (sum over tokens)
        of chosen/rejected under the policy -- e.g. the output of sequence_logprob.
    chosen_len / rejected_len : (B,) number of scored response tokens in each
        response (the per-example sum of the loss_mask). Used to length-normalize
        the log-probs BEFORE the odds so exp(mean_logp) is a valid per-token
        probability in (0, 1); matches TRL ORPOTrainer's average_log_prob=True.
    chosen_nll : (B,) standard SFT negative log-likelihood on the chosen response.
    """
    # Length-normalize to a per-token MEAN log-prob. Without this, a summed logp
    # is very negative (e.g. -12), exp(-12) ~ 6e-6, log1p(-exp) ~ 0, and the odds
    # term collapses to a plain (mean_chosen - mean_rejected) log-prob difference.
    mean_chosen   = pol_chosen_logps   / chosen_len       # (B,)
    mean_rejected = pol_rejected_logps / rejected_len     # (B,)

    # log-odds(y) = log[ p/(1-p) ] = logp - log(1 - exp(logp))  (log1mexp form).
    def log_odds(logp):
        return logp - torch.log1p(-torch.exp(logp.clamp(max=-1e-6)))

    log_or = log_odds(mean_chosen) - log_odds(mean_rejected)   # (B,)
    or_term = -torch.nn.functional.logsigmoid(log_or)          # push chosen odds up
    return (chosen_nll + lam * or_term).mean()


# ============================================================================
# Block #5 (chapter line ~359) — verbatim from the book
# ============================================================================

def simpo_loss(pol_chosen_logps, pol_rejected_logps,
               chosen_len, rejected_len, beta=2.0, gamma=1.0):
    # Length-normalized (per-token average) log-prob = the SimPO reward / β.
    r_chosen   = beta * (pol_chosen_logps   / chosen_len)
    r_rejected = beta * (pol_rejected_logps / rejected_len)
    # Require the chosen reward to exceed rejected by at least the margin γ.
    return -torch.nn.functional.logsigmoid(r_chosen - r_rejected - gamma).mean()


# ============================================================================
# Glue: tiny CPU fixtures that exercise every block above end-to-end
# ============================================================================

def test_block0_dpo_from_scratch():
    torch.manual_seed(0)

    B, T, V = 4, 6, 13   # batch, sequence length, vocab size

    # --- Fake "policy" and "reference" logits for chosen/rejected responses ---
    policy_chosen_logits   = torch.randn(B, T, V, requires_grad=True)
    policy_rejected_logits = torch.randn(B, T, V, requires_grad=True)
    ref_chosen_logits      = torch.randn(B, T, V)
    ref_rejected_logits    = torch.randn(B, T, V)

    # Random token-id labels (already "shifted" as the docstring expects).
    chosen_labels   = torch.randint(0, V, (B, T))
    rejected_labels = torch.randint(0, V, (B, T))

    # loss_mask: zero out first 2 positions (pretend they're prompt tokens),
    # score the remaining 4 as "response" tokens.
    loss_mask = torch.zeros(B, T)
    loss_mask[:, 2:] = 1.0

    # --- Exercise sequence_logprob() on all four (policy/ref x chosen/rejected) ---
    pol_chosen_logps   = sequence_logprob(policy_chosen_logits, chosen_labels, loss_mask)
    pol_rejected_logps = sequence_logprob(policy_rejected_logits, rejected_labels, loss_mask)
    ref_chosen_logps   = sequence_logprob(ref_chosen_logits, chosen_labels, loss_mask)
    ref_rejected_logps = sequence_logprob(ref_rejected_logits, rejected_labels, loss_mask)

    assert pol_chosen_logps.shape == (B,)
    assert pol_rejected_logps.shape == (B,)
    # Sequence log-probs must be <= 0 (sums of log-softmax outputs).
    assert (pol_chosen_logps <= 0).all()
    assert (ref_chosen_logps <= 0).all()

    # --- Exercise dpo_loss() ---
    loss, chosen_reward, reject_reward = dpo_loss(
        pol_chosen_logps, pol_rejected_logps,
        ref_chosen_logps, ref_rejected_logps,
        beta=0.1,
    )

    assert loss.dim() == 0, "dpo_loss should return a scalar"
    assert loss.item() > 0, "softplus(-logits) is strictly positive"
    assert chosen_reward.shape == (B,)
    assert reject_reward.shape == (B,)
    assert not chosen_reward.requires_grad, "implicit rewards are detached diagnostics"

    # Loss must actually be differentiable w.r.t. the policy logits.
    loss.backward()
    assert policy_chosen_logits.grad is not None
    assert policy_rejected_logits.grad is not None
    assert torch.isfinite(policy_chosen_logits.grad).all()
    assert torch.isfinite(policy_rejected_logits.grad).all()

    # --- Sanity-check against the worked-example arithmetic in the chapter ---
    # beta=0.1, chosen log-ratio +0.5, rejected log-ratio +1.0 => Delta = -0.05
    # loss = -log(sigmoid(-0.05)) ~= 0.71812
    pcl = torch.tensor([-12.0])
    rcl = torch.tensor([-12.5])
    prl = torch.tensor([-10.0])
    rrl = torch.tensor([-11.0])
    worked_loss, worked_cr, worked_rr = dpo_loss(pcl, prl, rcl, rrl, beta=0.1)
    assert abs(worked_loss.item() - 0.71812) < 1e-3, worked_loss.item()
    assert abs(worked_cr.item() - 0.05) < 1e-6
    assert abs(worked_rr.item() - 0.10) < 1e-6

    print("Block #0 (sequence_logprob, dpo_loss): OK")
    print(f"  batch loss = {loss.item():.4f}, chosen_reward = {chosen_reward.tolist()}")
    print(f"  worked-example loss = {worked_loss.item():.5f} (book says 0.718 nats)")


class _TinyLM(torch.nn.Module):
    """Minimal stand-in for a HF causal LM: returns an object with `.logits`
    of shape (B, T, V). Used to drive dpo_training_step end-to-end on CPU."""
    def __init__(self, vocab):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, vocab)

    def forward(self, ids):
        return SimpleNamespace(logits=self.emb(ids))


def test_block1_dpo_training_step():
    torch.manual_seed(1)
    B, T, V = 3, 7, 11

    policy = _TinyLM(V)
    ref_model = _TinyLM(V)
    for p in ref_model.parameters():
        p.requires_grad_(False)

    def ids():
        return torch.randint(0, V, (B, T))

    mask = torch.zeros(B, T)
    mask[:, 3:] = 1.0   # first 3 tokens are "prompt", rest scored

    batch = {
        "chosen_input_ids": ids(),   "chosen_labels": ids(),   "chosen_loss_mask": mask.clone(),
        "rejected_input_ids": ids(), "rejected_labels": ids(), "rejected_loss_mask": mask.clone(),
    }

    loss, metrics = dpo_training_step(policy, ref_model, batch, beta=0.1)

    assert loss.dim() == 0
    assert loss.item() > 0
    assert set(metrics) == {"reward_acc", "reward_margin"}
    assert 0.0 <= metrics["reward_acc"] <= 1.0

    # Real gradients must flow into the POLICY only; the frozen reference must not.
    loss.backward()
    assert policy.emb.weight.grad is not None
    assert torch.isfinite(policy.emb.weight.grad).all()
    assert ref_model.emb.weight.grad is None, "reference model must stay frozen"

    print("Block #1 (dpo_training_step): OK")
    print(f"  loss = {loss.item():.4f}, metrics = {metrics}")


def test_block2_ipo():
    torch.manual_seed(2)
    B = 5
    pol_chosen = torch.randn(B, requires_grad=True)
    pol_reject = torch.randn(B, requires_grad=True)
    ref_chosen = torch.randn(B)
    ref_reject = torch.randn(B)

    loss = ipo_loss(pol_chosen, pol_reject, ref_chosen, ref_reject, beta=0.1)
    assert loss.dim() == 0
    assert loss.item() >= 0.0, "squared error is non-negative"
    loss.backward()
    assert torch.isfinite(pol_chosen.grad).all()

    # When h hits the target 1/(2β), the loss is exactly zero (finite optimum).
    beta = 0.1
    # h = (chosen ratio) - (rejected ratio); set chosen ratio = 1/(2β), rejected = 0.
    pc = torch.tensor([1.0 / (2.0 * beta)])
    rc = torch.tensor([0.0])
    pr = torch.tensor([0.0])
    rr = torch.tensor([0.0])
    zero = ipo_loss(pc, pr, rc, rr, beta=beta)
    assert abs(zero.item()) < 1e-10, zero.item()

    print("Block #2 (ipo_loss): OK")
    print(f"  loss = {loss.item():.4f}, at-target loss = {zero.item():.2e}")


def test_block3_kto():
    torch.manual_seed(3)
    B = 6
    policy_logps = torch.randn(B, requires_grad=True)
    ref_logps    = torch.randn(B)
    labels_desirable = torch.tensor([True, False, True, False, True, False])
    kl_z0 = torch.tensor(0.05)

    loss = kto_loss(policy_logps, ref_logps, labels_desirable, kl_z0, beta=0.1)
    assert loss.dim() == 0
    assert loss.item() > 0
    loss.backward()
    assert torch.isfinite(policy_logps.grad).all()

    # Monotonicity check: a desirable example with a much higher implicit reward
    # should incur strictly lower loss than one with a much lower reward.
    ref1 = torch.tensor([0.0])
    des = torch.tensor([True])
    z0 = torch.tensor(0.0)
    hi = kto_loss(torch.tensor([10.0]), ref1, des, z0, beta=0.1)   # r >> z0 -> good
    lo = kto_loss(torch.tensor([-10.0]), ref1, des, z0, beta=0.1)  # r << z0 -> bad
    assert hi.item() < lo.item(), (hi.item(), lo.item())

    print("Block #3 (kto_loss): OK")
    print(f"  loss = {loss.item():.4f}, desirable hi/lo loss = {hi.item():.4f}/{lo.item():.4f}")


def test_block4_orpo():
    torch.manual_seed(4)
    B = 4
    # Summed sequence log-probs (negative), scored-token counts, and SFT NLL.
    pol_chosen_logps   = torch.tensor([-12.0, -8.0, -20.0, -5.0], requires_grad=True)
    pol_rejected_logps = torch.tensor([-15.0, -9.0, -18.0, -6.0], requires_grad=True)
    chosen_len   = torch.tensor([4.0, 3.0, 6.0, 2.0])
    rejected_len = torch.tensor([5.0, 3.0, 5.0, 2.0])
    chosen_nll   = torch.tensor([3.0, 2.5, 3.3, 2.4], requires_grad=True)

    loss = orpo_loss(pol_chosen_logps, pol_rejected_logps,
                     chosen_len, rejected_len, chosen_nll, lam=0.1)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(pol_chosen_logps.grad).all()
    assert torch.isfinite(chosen_nll.grad).all()

    # The odds-ratio term should be smaller when the chosen per-token logprob is
    # relatively higher (better than rejected) than when it is worse.
    def or_only(pc, pr):
        # lam huge, nll zero -> isolate the odds-ratio contribution
        return orpo_loss(pc, pr, torch.tensor([4.0]), torch.tensor([4.0]),
                         torch.tensor([0.0]), lam=1.0)
    better  = or_only(torch.tensor([-4.0]), torch.tensor([-12.0]))  # chosen >> rejected
    worse   = or_only(torch.tensor([-12.0]), torch.tensor([-4.0]))  # chosen << rejected
    assert better.item() < worse.item(), (better.item(), worse.item())

    print("Block #4 (orpo_loss): OK")
    print(f"  loss = {loss.item():.4f}, or-term better/worse = {better.item():.4f}/{worse.item():.4f}")


def test_block5_simpo():
    torch.manual_seed(5)
    pol_chosen_logps   = torch.tensor([-12.0, -8.0, -20.0, -5.0], requires_grad=True)
    pol_rejected_logps = torch.tensor([-15.0, -9.0, -18.0, -6.0], requires_grad=True)
    chosen_len   = torch.tensor([4.0, 3.0, 6.0, 2.0])
    rejected_len = torch.tensor([5.0, 3.0, 5.0, 2.0])

    loss = simpo_loss(pol_chosen_logps, pol_rejected_logps,
                      chosen_len, rejected_len, beta=2.0, gamma=1.0)
    assert loss.dim() == 0
    assert loss.item() > 0
    loss.backward()
    assert torch.isfinite(pol_chosen_logps.grad).all()

    # A larger target margin gamma should never decrease the loss (it demands more).
    small_g = simpo_loss(pol_chosen_logps.detach(), pol_rejected_logps.detach(),
                         chosen_len, rejected_len, beta=2.0, gamma=0.0)
    large_g = simpo_loss(pol_chosen_logps.detach(), pol_rejected_logps.detach(),
                         chosen_len, rejected_len, beta=2.0, gamma=3.0)
    assert large_g.item() > small_g.item(), (small_g.item(), large_g.item())

    print("Block #5 (simpo_loss): OK")
    print(f"  loss = {loss.item():.4f}, gamma 0 vs 3 = {small_g.item():.4f}/{large_g.item():.4f}")


def main():
    test_block0_dpo_from_scratch()
    test_block1_dpo_training_step()
    test_block2_ipo()
    test_block3_kto()
    test_block4_orpo()
    test_block5_simpo()
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
