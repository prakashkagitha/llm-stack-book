"""
Runnable-code test for content/06-rl-infra/09-advantage-kl-tricks.md

Every code block in the chapter is a self-contained, pure-CPU PyTorch function
(no network, no GPU, no external state). This test copies each block verbatim
and exercises it -- calling the functions and asserting on their outputs,
including against the chapter's own worked-example numbers.

Blocks tested (all of them):
    - block #0 (line ~71):  compute_gae            -- GAE backward recurrence
    - block #1 (line ~121): grpo_advantages        -- group-relative advantage
    - block #2 (line ~154): masked_whiten          -- masked zero-mean/unit-var
    - block #3 (line ~200): kl_estimators          -- k1/k2/k3 KL estimators
    - block #4 (line ~239): policy_loss_kl_in_loss -- GRPO-style KL-in-loss
    - block #5 (line ~269): AdaptiveKLController    -- proportional KL controller
    - block #6 (line ~307): ppo_clip_loss          -- clip-higher PPO surrogate
    - block #7 (line ~334): dual_clip_loss         -- dual-clip PPO
    - block #8 (line ~373): entropy_from_logits    -- full-vocab entropy
    - block #9 (line ~403): aggregate_loss         -- token/seq/fixed reductions
    - block #10 (line ~449): grpo_train_step       -- end-to-end wiring
"""

import torch

# ---------------------------------------------------------------------------
# Block #0 (line ~71): GAE backward recurrence
# ---------------------------------------------------------------------------

def compute_gae(rewards, values, gamma=1.0, lam=0.95, mask=None):
    """Generalized Advantage Estimation, computed token-by-token, backward.

    Args:
        rewards: (B, T) per-token rewards. For LLMs this is usually all zeros
                 except the last valid token, which carries r - beta*KL.
        values:  (B, T) critic estimates V(s_t) for each token position.
        gamma:   discount. For LLM RL almost always 1.0 (episodes are short
                 and we do not want to discount a correct final answer).
        lam:     GAE lambda. 0.95-1.0 typical.
        mask:    (B, T) 1.0 for real tokens, 0.0 for padding.
    Returns:
        advantages (B, T), returns (B, T)
    """
    B, T = rewards.shape
    if mask is None:
        mask = torch.ones_like(rewards)
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(B, device=rewards.device)
    # Append a bootstrap value of 0 past the end (terminal state).
    for t in reversed(range(T)):
        next_value = values[:, t + 1] if t + 1 < T else torch.zeros(B, device=rewards.device)
        # TD residual delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        delta = rewards[:, t] + gamma * next_value * mask[:, t] - values[:, t]
        # Recurrence: A_t = delta_t + gamma*lam*A_{t+1}
        last_gae = delta + gamma * lam * last_gae * mask[:, t]
        advantages[:, t] = last_gae
    returns = advantages + values          # value-function targets
    return advantages * mask, returns * mask


# ---------------------------------------------------------------------------
# Block #1 (line ~121): group-relative advantages (GRPO)
# ---------------------------------------------------------------------------

def grpo_advantages(rewards, group_size, eps=1e-4, scale_by_std=True):
    """Group-relative advantages. rewards: (B,) with B = num_prompts*group_size,
    laid out as [p0_s0, p0_s1, ..., p0_s{G-1}, p1_s0, ...]."""
    rewards = rewards.view(-1, group_size)            # (num_prompts, G)
    mean = rewards.mean(dim=1, keepdim=True)
    adv = rewards - mean                               # subtract group baseline
    if scale_by_std:                                  # standard GRPO
        std = rewards.std(dim=1, keepdim=True)
        adv = adv / (std + eps)
    # else: Dr. GRPO — mean-only, avoids the std up-weighting bias
    return adv.view(-1)                               # broadcast to tokens later


# ---------------------------------------------------------------------------
# Block #2 (line ~154): masked whitening
# ---------------------------------------------------------------------------

def masked_whiten(advantages, mask, shift_mean=True, eps=1e-8):
    """Zero-mean, unit-variance advantages over the masked (non-pad) tokens.

    shift_mean=False keeps the mean (some recipes only rescale variance, because
    subtracting a batch mean is itself a baseline change that can be undesirable
    when you already subtracted a per-prompt group baseline)."""
    n = mask.sum()
    mean = (advantages * mask).sum() / n
    var = ((advantages - mean) ** 2 * mask).sum() / n
    whitened = (advantages - mean) * torch.rsqrt(var + eps)
    if not shift_mean:
        whitened = whitened + mean
    return whitened * mask


# ---------------------------------------------------------------------------
# Block #3 (line ~200): k1/k2/k3 KL estimators
# ---------------------------------------------------------------------------

def kl_estimators(logp_policy, logp_ref):
    """Per-token KL(pi_theta || pi_ref) estimators from sampled-token logprobs.
    logp_policy, logp_ref: (B, T) log pi_theta(a_t) and log pi_ref(a_t)."""
    logr = logp_ref - logp_policy                 # r = log(pi_ref / pi_theta)
    k1 = -logr                                    # unbiased, can be negative
    k2 = 0.5 * logr.pow(2)                         # biased, always >= 0
    k3 = torch.expm1(logr) - logr                 # = exp(logr) - 1 - logr; unbiased, >= 0
    return k1, k2, k3


# ---------------------------------------------------------------------------
# Block #4 (line ~239): GRPO-style KL-in-loss policy loss
# ---------------------------------------------------------------------------

def policy_loss_kl_in_loss(logp, logp_old, logp_ref, advantages, mask,
                           clip_eps=0.2, beta=0.04):
    """GRPO-style: advantages already computed from task reward only;
    KL is an explicit, differentiable loss term (k3)."""
    ratio = torch.exp(logp - logp_old)                       # importance ratio rho_t
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    pg_loss = -torch.min(unclipped, clipped)                 # per-token PPO surrogate

    logr = logp_ref - logp                                   # differentiable in logp
    kl = torch.expm1(logr) - logr                            # k3, >= 0
    loss_per_tok = pg_loss + beta * kl
    return (loss_per_tok * mask).sum() / mask.sum()          # token-mean (see below)


# ---------------------------------------------------------------------------
# Block #5 (line ~269): adaptive KL controller
# ---------------------------------------------------------------------------

class AdaptiveKLController:
    """Proportional controller that nudges beta toward a target KL.
    From the InstructGPT/PPO-RLHF lineage (Ziegler et al., 2019)."""
    def __init__(self, init_beta=0.2, target_kl=6.0, horizon=10000):
        self.beta = init_beta
        self.target = target_kl
        self.horizon = horizon          # adaptation speed (larger = slower)

    def update(self, current_kl, n_steps):
        # proportional error, clipped to +-20% so a single noisy batch
        # cannot swing beta wildly
        proportional_error = max(-0.2, min(0.2, current_kl / self.target - 1.0))
        mult = 1.0 + proportional_error * n_steps / self.horizon
        self.beta *= mult
        return self.beta


# ---------------------------------------------------------------------------
# Block #6 (line ~307): clip-higher PPO surrogate
# ---------------------------------------------------------------------------

def ppo_clip_loss(ratio, adv, mask, eps_low=0.2, eps_high=0.28):
    """Decoupled (clip-higher) PPO surrogate. eps_high > eps_low gives
    low-probability/exploratory tokens more room to grow (DAPO)."""
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * adv
    loss = -torch.min(unclipped, clipped)
    # diagnostic: fraction of tokens where the clip was active
    clipfrac = ((ratio > 1 + eps_high) | (ratio < 1 - eps_low)).float()
    clipfrac = (clipfrac * mask).sum() / mask.sum()
    return (loss * mask).sum() / mask.sum(), clipfrac


# ---------------------------------------------------------------------------
# Block #7 (line ~334): dual-clip PPO
# ---------------------------------------------------------------------------

def dual_clip_loss(ratio, adv, mask, eps=0.2, c=3.0):
    """Dual-clip PPO (Ye et al. 2020). Adds a lower floor c*adv for adv<0."""
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * adv
    standard = torch.min(ratio * adv, clipped)              # usual PPO surrogate
    # For negative advantages, floor the (negative) surrogate at c*adv:
    neg = torch.max(standard, c * adv)                      # only binds when adv<0
    surrogate = torch.where(adv < 0, neg, standard)
    return -(surrogate * mask).sum() / mask.sum()


# ---------------------------------------------------------------------------
# Block #8 (line ~373): full-vocab entropy
# ---------------------------------------------------------------------------

def entropy_from_logits(logits, mask):
    """Exact per-token entropy H = -sum_v p_v log p_v over the full vocab,
    averaged over real tokens. logits: (B, T, V)."""
    logp = torch.log_softmax(logits, dim=-1)
    p = logp.exp()
    ent = -(p * logp).sum(dim=-1)                 # (B, T)
    return (ent * mask).sum() / mask.sum()


# ---------------------------------------------------------------------------
# Block #9 (line ~403): loss aggregation
# ---------------------------------------------------------------------------

def aggregate_loss(per_token_loss, mask, mode="token_mean"):
    """Reduce (B, T) per-token loss to a scalar. The mode silently changes
    the objective — choose deliberately."""
    if mode == "token_mean":
        # every token equal -> long sequences contribute proportionally more
        return (per_token_loss * mask).sum() / mask.sum().clamp_min(1.0)
    elif mode == "seq_mean":
        # every sequence equal -> per-sequence mean, then mean over sequences
        seq_len = mask.sum(dim=1).clamp_min(1.0)              # (B,) valid lengths
        seq_loss = (per_token_loss * mask).sum(dim=1) / seq_len
        return seq_loss.mean()
    elif mode == "token_mean_fixed":
        # DAPO-style: divide by a fixed constant (e.g. max_len) so the
        # denominator does not depend on batch composition
        return (per_token_loss * mask).sum() / per_token_loss.shape[1]
    else:
        raise ValueError(mode)


# ---------------------------------------------------------------------------
# Block #10 (line ~449): end-to-end GRPO train step
# ---------------------------------------------------------------------------

def grpo_train_step(policy_logp, old_logp, ref_logp, full_logits,
                    rewards, group_size, mask,
                    eps_low=0.2, eps_high=0.28, beta=0.0, ent_coef=0.0):
    """One GRPO-style update. All log-prob tensors are (B, T); rewards (B,).
    Returns scalar loss and a metrics dict for logging."""
    # 1) Group-relative advantage from terminal reward, broadcast to tokens.
    adv_seq = grpo_advantages(rewards, group_size, scale_by_std=False)   # (B,)
    adv = adv_seq.unsqueeze(1).expand_as(mask)                           # (B, T)
    # 2) Optional batch-level whitening of *scale* only (mean already removed).
    adv = masked_whiten(adv, mask, shift_mean=False)
    # 3) Importance ratio with a numerical safety clamp.
    log_ratio = (policy_logp - old_logp).clamp(-20.0, 20.0)
    ratio = torch.exp(log_ratio)
    # 4) Clip-higher PPO surrogate (per token).
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * adv
    pg = -torch.min(unclipped, clipped)
    # 5) KL-in-loss (k3, differentiable) and entropy bonus.
    logr = (ref_logp - policy_logp)
    kl = torch.expm1(logr) - logr                                       # (B, T), >=0
    ent = entropy_from_logits(full_logits, mask)                        # scalar
    per_tok = pg + beta * kl
    loss = (per_tok * mask).sum() / mask.sum().clamp_min(1.0) - ent_coef * ent
    # 6) Diagnostics — log these every step.
    with torch.no_grad():
        clipfrac = (((ratio > 1 + eps_high) | (ratio < 1 - eps_low)).float()
                    * mask).sum() / mask.sum()
        approx_kl = (kl * mask).sum() / mask.sum()
    return loss, {"clipfrac": clipfrac.item(), "kl": approx_kl.item(),
                  "entropy": ent.item(), "ratio_mean": ratio.mean().item()}


# ---------------------------------------------------------------------------
# Exercise the blocks
# ---------------------------------------------------------------------------

def test_compute_gae():
    # --- compute_gae: a tiny B=2, T=5 rollout with a terminal-only reward ---
    B, T = 2, 5
    rewards = torch.zeros(B, T)
    rewards[:, -1] = torch.tensor([1.0, -1.0])     # terminal reward only, per book's setup
    values = torch.randn(B, T) * 0.1                # small critic estimates
    mask = torch.ones(B, T)
    mask[1, 3:] = 0.0                                # second sequence padded after t=2

    advantages, returns = compute_gae(rewards, values, gamma=1.0, lam=0.95, mask=mask)

    assert advantages.shape == (B, T)
    assert returns.shape == (B, T)
    assert torch.isfinite(advantages).all()
    assert torch.isfinite(returns).all()
    # Padded positions must be exactly zero (masked out).
    assert torch.all(advantages[1, 3:] == 0.0)
    assert torch.all(returns[1, 3:] == 0.0)
    # returns = advantages + values on the real tokens, per the book's definition.
    real = mask.bool()
    assert torch.allclose(returns[real], (advantages + values * mask)[real])

    # Chapter claim (line ~105): with gamma=1, lambda=1 and terminal-only reward,
    # every token gets advantage R - V(s_t). Verify against a direct computation.
    adv1, _ = compute_gae(rewards, values, gamma=1.0, lam=1.0)
    R = rewards[:, -1:].expand_as(values)           # terminal reward broadcast
    assert torch.allclose(adv1, R - values, atol=1e-5)

    print("compute_gae: advantages =")
    print(advantages)
    print("compute_gae: returns =")
    print(returns)


def test_grpo_advantages():
    # Two prompts, group_size=2. p0 = [1,0] (informative), p1 = [1,1] (all-same).
    rewards = torch.tensor([1.0, 0.0, 1.0, 1.0])
    adv_mean_only = grpo_advantages(rewards, group_size=2, scale_by_std=False)
    # p0: subtract mean 0.5 -> [+0.5, -0.5]; p1: all-same -> [0, 0] (no gradient).
    assert torch.allclose(adv_mean_only, torch.tensor([0.5, -0.5, 0.0, 0.0]))

    # Standard GRPO divides by group std; the all-same group -> 0/eps = 0.
    adv_std = grpo_advantages(rewards, group_size=2, scale_by_std=True)
    assert torch.allclose(adv_std[2:], torch.zeros(2))
    # Non-degenerate group is scaled up (|adv| > mean-only version).
    assert adv_std[0] > 0 and adv_std[1] < 0
    print("grpo_advantages (mean-only):", adv_mean_only)
    print("grpo_advantages (std-scaled):", adv_std)


def test_masked_whiten():
    adv = torch.tensor([[1.0, 3.0, 5.0, 7.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])     # last token is padding
    w = masked_whiten(adv, mask, shift_mean=True)
    real = mask.bool()
    # Over the 3 real tokens the whitened values are zero-mean, unit-variance.
    vals = w[real]
    assert torch.allclose(vals.mean(), torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(vals.var(unbiased=False), torch.tensor(1.0), atol=1e-4)
    # Padding is untouched (zeroed) and the pad value did NOT enter the stats:
    assert w[0, 3] == 0.0
    # shift_mean=False keeps the (masked) mean instead of centering at 0.
    w2 = masked_whiten(adv, mask, shift_mean=False)
    assert w2[real].mean() > 1.0
    print("masked_whiten:", w)


def test_kl_estimators():
    # --- kl_estimators: tiny (B, T) logprob tensors ---
    logp_policy = torch.log(torch.tensor([[0.5, 0.3, 0.8], [0.9, 0.2, 0.4]]))
    logp_ref = torch.log(torch.tensor([[0.5, 0.6, 0.2], [0.9, 0.5, 0.1]]))

    k1, k2, k3 = kl_estimators(logp_policy, logp_ref)

    assert k1.shape == k2.shape == k3.shape == logp_policy.shape
    assert torch.isfinite(k1).all() and torch.isfinite(k2).all() and torch.isfinite(k3).all()
    # k2 and k3 must always be non-negative, as the chapter states.
    assert (k2 >= 0).all(), "k2 = 0.5*r^2 must be non-negative"
    assert (k3 >= -1e-6).all(), "k3 = e^r - 1 - r must be non-negative"
    # k1 (=-logr) can go negative when the policy assigns lower prob than ref
    # to the sampled token -- confirm that's actually exercised by this fixture.
    assert (k1 < 0).any(), "expected at least one negative k1 in this fixture"
    # Sanity-check the algebra: k3 should equal expm1(logr) - logr exactly (same formula).
    logr = logp_ref - logp_policy
    assert torch.allclose(k3, torch.expm1(logr) - logr)
    assert torch.allclose(k1, -logr)
    assert torch.allclose(k2, 0.5 * logr.pow(2))

    print("kl_estimators: k1 =", k1)
    print("kl_estimators: k2 =", k2)
    print("kl_estimators: k3 =", k3)


def test_policy_loss_kl_in_loss():
    torch.manual_seed(1)
    B, T = 2, 3
    logp_old = torch.randn(B, T) - 1.0
    logp = logp_old.clone().requires_grad_(True)     # on-policy: logp == logp_old
    logp_ref = torch.randn(B, T) - 1.0
    adv = torch.randn(B, T)
    mask = torch.ones(B, T)

    loss = policy_loss_kl_in_loss(logp, logp_old, logp_ref, adv, mask,
                                  clip_eps=0.2, beta=0.04)
    assert loss.dim() == 0 and torch.isfinite(loss)
    # The KL term is differentiable in logp (chapter's key point about k3).
    loss.backward()
    assert logp.grad is not None and torch.isfinite(logp.grad).all()
    assert (logp.grad != 0).any(), "KL-in-loss must produce gradient w.r.t. policy logp"
    print("policy_loss_kl_in_loss:", loss.item())


def test_adaptive_kl_controller():
    # KL above target -> beta grows; below target -> beta shrinks.
    ctrl = AdaptiveKLController(init_beta=0.2, target_kl=6.0, horizon=10000)
    b0 = ctrl.beta
    b1 = ctrl.update(current_kl=12.0, n_steps=100)   # KL >> target
    assert b1 > b0, "beta must grow when KL exceeds target"
    # proportional error clipped to +0.2 -> mult = 1 + 0.2*100/10000 = 1.002
    assert abs(b1 - 0.2 * 1.002) < 1e-9

    ctrl2 = AdaptiveKLController(init_beta=0.2, target_kl=6.0, horizon=10000)
    b2 = ctrl2.update(current_kl=0.0, n_steps=100)    # KL well below target
    assert b2 < 0.2, "beta must shrink when KL is below target"
    print("adaptive KL: up ->", b1, " down ->", b2)


def test_ppo_clip_loss():
    # Chapter worked example (line ~347): adv=+2.0, ratio=1.5, eps_low=0.2, eps_high=0.28.
    ratio = torch.tensor([[1.5]])
    adv = torch.tensor([[2.0]])
    mask = torch.ones(1, 1)
    loss, clipfrac = ppo_clip_loss(ratio, adv, mask, eps_low=0.2, eps_high=0.28)
    # clip(1.5, 0.8, 1.28)=1.28 -> clipped surrogate 2.56; min(3.0,2.56)=2.56; loss=-2.56.
    assert torch.allclose(loss, torch.tensor(-2.56), atol=1e-5)
    # ratio 1.5 > 1.28 => clip is active on this token.
    assert torch.allclose(clipfrac, torch.tensor(1.0))
    print("ppo_clip_loss worked example: loss =", loss.item(),
          "clipfrac =", clipfrac.item())


def test_dual_clip_loss():
    # Chapter worked example (line ~353): adv=-1.0, ratio=4.0, c=3.
    ratio = torch.tensor([[4.0]])
    adv = torch.tensor([[-1.0]])
    mask = torch.ones(1, 1)
    loss = dual_clip_loss(ratio, adv, mask, eps=0.2, c=3.0)
    # surrogate floored at max(-4.0, -3.0) = -3.0 -> loss = +3.0.
    assert torch.allclose(loss, torch.tensor(3.0), atol=1e-5)

    # For positive advantage dual-clip == standard PPO surrogate (floor never binds).
    ratio_p = torch.tensor([[1.5]])
    adv_p = torch.tensor([[2.0]])
    loss_p = dual_clip_loss(ratio_p, adv_p, mask, eps=0.2, c=3.0)
    # standard: clip(1.5,0.8,1.2)*2=2.4; min(3.0,2.4)=2.4; loss=-2.4.
    assert torch.allclose(loss_p, torch.tensor(-2.4), atol=1e-5)
    print("dual_clip_loss: neg-adv floored loss =", loss.item(),
          "pos-adv loss =", loss_p.item())


def test_entropy_from_logits():
    # Uniform logits over V -> entropy = log(V) at every position.
    B, T, V = 2, 3, 8
    logits = torch.zeros(B, T, V)                    # uniform distribution
    mask = torch.ones(B, T)
    ent = entropy_from_logits(logits, mask)
    assert torch.allclose(ent, torch.log(torch.tensor(float(V))), atol=1e-5)

    # A near-deterministic distribution has entropy ~ 0 (< uniform).
    peaked = torch.zeros(B, T, V)
    peaked[..., 0] = 20.0
    ent2 = entropy_from_logits(peaked, mask)
    assert ent2 < ent and ent2 >= 0
    print("entropy_from_logits: uniform =", ent.item(), "peaked =", ent2.item())


def test_aggregate_loss():
    # seq0: 2 valid tokens of loss 1.0; seq1: 4 valid tokens of loss 3.0.
    per_tok = torch.tensor([[1.0, 1.0, 0.0, 0.0],
                            [3.0, 3.0, 3.0, 3.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0],
                         [1.0, 1.0, 1.0, 1.0]])
    tm = aggregate_loss(per_tok, mask, "token_mean")
    sm = aggregate_loss(per_tok, mask, "seq_mean")
    tmf = aggregate_loss(per_tok, mask, "token_mean_fixed")
    # token_mean: (2*1 + 4*3)/6 = 14/6; the long seq dominates (12/14 of the sum).
    assert torch.allclose(tm, torch.tensor(14.0 / 6.0))
    # seq_mean: mean([1.0, 3.0]) = 2.0; each sequence counts once.
    assert torch.allclose(sm, torch.tensor(2.0))
    # token_mean_fixed: 14 / T(=4) = 3.5.
    assert torch.allclose(tmf, torch.tensor(3.5))
    # The chapter's whole point: same data, different objective.
    assert not torch.allclose(tm, sm)
    try:
        aggregate_loss(per_tok, mask, "bogus")
        assert False, "unknown mode must raise"
    except ValueError:
        pass
    print("aggregate_loss token/seq/fixed:", tm.item(), sm.item(), tmf.item())


def test_grpo_train_step():
    torch.manual_seed(2)
    B, T, V, G = 4, 5, 16, 2                          # 2 prompts x group_size 2
    old_logp = torch.randn(B, T) - 1.0
    policy_logp = old_logp + 0.01 * torch.randn(B, T)
    ref_logp = torch.randn(B, T) - 1.0
    full_logits = torch.randn(B, T, V)
    rewards = torch.tensor([1.0, 0.0, 1.0, 1.0])
    mask = torch.ones(B, T)
    mask[3, 3:] = 0.0

    loss, metrics = grpo_train_step(policy_logp, old_logp, ref_logp, full_logits,
                                    rewards, group_size=G, mask=mask,
                                    beta=0.04, ent_coef=0.01)
    assert loss.dim() == 0 and torch.isfinite(loss)
    for key in ("clipfrac", "kl", "entropy", "ratio_mean"):
        assert key in metrics and torch.isfinite(torch.tensor(metrics[key]))
    assert 0.0 <= metrics["clipfrac"] <= 1.0
    assert metrics["kl"] >= 0.0            # k3 is non-negative
    assert metrics["entropy"] >= 0.0
    print("grpo_train_step: loss =", loss.item(), "metrics =", metrics)


def main():
    torch.manual_seed(0)
    test_compute_gae()
    test_grpo_advantages()
    test_masked_whiten()
    test_kl_estimators()
    test_policy_loss_kl_in_loss()
    test_adaptive_kl_controller()
    test_ppo_clip_loss()
    test_dual_clip_loss()
    test_entropy_from_logits()
    test_aggregate_loss()
    test_grpo_train_step()
    print("\nAll tested blocks executed successfully.")


if __name__ == "__main__":
    main()
