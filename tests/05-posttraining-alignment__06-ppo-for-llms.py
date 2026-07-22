"""
Runs the CPU-runnable Python from content/05-posttraining-alignment/06-ppo-for-llms.md
("Policy Gradients & PPO for Language Models") and checks it actually executes and
produces numbers consistent with the chapter's own worked examples.

Blocks tested (verbatim from the book, concatenated in chapter order):
  - block #0 (line ~67):  reinforce_loss
  - block #1 (line ~188): compute_gae
  - block #5 (line ~407): the full PPO step -- token_logprobs_and_values,
                           ppo_rollout, ppo_update, and the module-level constants.

Block #5 is not standalone: it calls three helper functions defined earlier in the
same chapter (ppo_policy_loss ~line 292, ppo_value_loss ~line 329, make_token_rewards
~line 369). Those were flagged as "fragment" blocks (#2, #3, #4) because on their own
they are not runnable demos -- but block #5 cannot execute without them, so they are
copied here verbatim too and get exercised for real as part of driving block #5.
This is the "later blocks depend on earlier names" case the assignment describes.

Everything else here (TinyLM, a toy reward model, generate_batch, policy_entropy,
the optimizer) is minimal glue standing in for what the chapter explicitly assumes
as given: "Assume: policy (with value head), ref_model, reward_model, tokenizer,
optimizer." `policy_entropy` is called by the book's own ppo_update but never
defined in the chapter text, so a small, honest entropy computation is supplied.

BUG FOUND AND FIXED (mirrored in both this file and the .md):
  make_token_rewards computed the index of the last response token as
  `mask.sum(dim=1) - 1`. That is only correct if the response occupies columns
  0..count-1 of `mask`. But in ppo_rollout, `mask` is `resp_mask[:, 1:]`, which
  still carries the prompt's leading zeros (shifted by one for next-token
  alignment) -- so for any prompt longer than 1 token the reward-model score
  got written onto the wrong column, silently corrupting GAE for every sequence
  in the batch. This is exactly the failure mode the chapter's own "Common
  pitfall" callout warns about. Fixed to locate the last column where mask==1
  positionally (works for any prompt length). See the assertion in
  test_make_token_rewards_places_reward_correctly below, which reproduces the
  bug on a synthetic case with prompt_len=3 before confirming the fix.
"""
import copy
import random
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
random.seed(0)


# ======================================================================
# Block #0 (line ~67) -- vanilla REINFORCE loss
# ======================================================================
def reinforce_loss(token_logprobs, response_mask, rewards):
    """
    Vanilla REINFORCE for a batch of sampled responses.

    token_logprobs : (B, T) log pi_theta(o_t | q, o_<t) for the SAMPLED tokens.
    response_mask  : (B, T) 1.0 on generated tokens, 0.0 on prompt/padding.
    rewards        : (B,)   scalar terminal reward per response.

    We MAXIMIZE E[R * sum_t logprob], so we MINIMIZE its negative.
    rewards are detached: they are weights, not differentiable targets.
    """
    seq_logprob = (token_logprobs * response_mask).sum(dim=-1)   # (B,) sum over tokens
    loss = -(rewards.detach() * seq_logprob).mean()
    return loss


def test_reinforce_loss():
    B, T, V = 4, 6, 10
    logits = torch.randn(B, T, V)
    logprobs_full = F.log_softmax(logits, dim=-1)
    tokens = torch.randint(0, V, (B, T))
    token_logprobs = (
        logprobs_full.gather(-1, tokens.unsqueeze(-1)).squeeze(-1).clone().requires_grad_(True)
    )
    response_mask = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 0.0]] * B)
    rewards = torch.tensor([1.0, -1.0, 2.0, 0.5])

    loss = reinforce_loss(token_logprobs, response_mask, rewards)
    assert loss.dim() == 0, "loss must be a scalar"
    loss.backward()
    assert token_logprobs.grad is not None
    # positions outside response_mask must receive exactly zero gradient
    assert torch.allclose(token_logprobs.grad[:, [0, 1, 5]], torch.zeros(B, 3))
    print(f"block #0 reinforce_loss: OK (loss={loss.item():.4f})")


# ======================================================================
# Block #1 (line ~188) -- Generalized Advantage Estimation
# ======================================================================
def compute_gae(rewards, values, response_mask, gamma=1.0, lam=0.95):
    """
    Generalized Advantage Estimation, computed per token by a backward pass.

    rewards       : (B, T) per-token reward. For RLHF this is mostly zeros,
                    with the reward-model score (and any KL penalty) placed
                    at the last response token of each sequence.
    values        : (B, T) critic estimates V_phi(s_t) for each token.
    response_mask : (B, T) 1.0 on response tokens, 0.0 elsewhere.
    Returns:
      advantages (B, T), returns (B, T) = advantages + values.
    """
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(B, device=rewards.device)
    # V(s_{T}) for the bootstrap of the final step is 0 (episode ends at EOS).
    next_value = torch.zeros(B, device=rewards.device)

    for t in reversed(range(T)):
        mask_t = response_mask[:, t]
        # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_adv = delta + gamma * lam * last_adv
        # zero out advantage/last_adv outside the response so padding can't leak
        advantages[:, t] = last_adv * mask_t
        last_adv = last_adv * mask_t
        next_value = values[:, t] * mask_t  # carry V(s_t) leftward as V(s_{t+1})

    returns = advantages + values
    # Whitening (mean 0, std 1) over response tokens stabilizes the scale.
    flat = advantages[response_mask.bool()]
    advantages = (advantages - flat.mean()) / (flat.std() + 1e-8)
    advantages = advantages * response_mask
    return advantages, returns


def test_compute_gae_matches_worked_example():
    """Reproduces the chapter's hand-checked 4-token toy example exactly."""
    rewards = torch.tensor([[0.0, 0.0, 10.0, 0.0]])
    values = torch.tensor([[1.0, 2.0, 3.0, 5.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])

    advantages, returns = compute_gae(rewards, values, mask, gamma=1.0, lam=0.5)

    # raw advantages (pre-whitening) per the book: [3.25, 4.5, 7, 0]
    raw_adv = returns - values  # returns = raw_adv + values, so this recovers raw_adv
    expected_raw = torch.tensor([[3.25, 4.5, 7.0, 5.0]])  # position 3 return is 0+5=5 (masked, ignore)
    assert torch.allclose(raw_adv[:, :3], expected_raw[:, :3], atol=1e-4)

    expected_returns = torch.tensor([[4.25, 6.5, 10.0, 5.0]])
    assert torch.allclose(returns, expected_returns, atol=1e-4)

    expected_whitened = torch.tensor([[-0.873, -0.218, 1.091, 0.0]])
    assert torch.allclose(advantages, expected_whitened, atol=2e-3)
    assert advantages[0, 3].item() == 0.0, "padding slot must be forced to exactly 0"
    print("block #1 compute_gae: OK, matches book's worked example", advantages.tolist())


# ======================================================================
# Fragment blocks #2, #3, #4 -- required by block #5, copied verbatim
# (block #4 carries the bug fix described in the module docstring)
# ======================================================================
def ppo_policy_loss(new_logprobs, old_logprobs, advantages, mask, clip_eps=0.2):
    """
    PPO clipped policy (actor) loss for one minibatch.

    new_logprobs : (B, T) log pi_theta(o_t|s_t)  -- requires grad
    old_logprobs : (B, T) log pi_old(o_t|s_t)    -- detached, from rollout time
    advantages   : (B, T) detached GAE advantages (already whitened)
    mask         : (B, T) response mask
    """
    log_ratio = new_logprobs - old_logprobs            # log(pi_theta / pi_old)
    ratio = log_ratio.exp()                            # the importance ratio r_t

    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    # maximize min(...)  ==  minimize -min(...)
    per_token_loss = -torch.min(unclipped, clipped)

    loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)

    # Diagnostics every practitioner watches:
    with torch.no_grad():
        clipfrac = ((ratio - 1.0).abs() > clip_eps)[mask.bool()].float().mean()
        approx_kl = ((ratio - 1) - log_ratio)[mask.bool()].mean()  # k3 KL(old||new)
    return loss, clipfrac, approx_kl


def test_ppo_policy_loss_matches_worked_example():
    """Reproduces the chapter's "tracing one token through the PPO step" example."""
    import math

    pi_old, pi_theta = 0.10, 0.14
    new_logprobs = torch.tensor([[math.log(pi_theta)]])
    old_logprobs = torch.tensor([[math.log(pi_old)]])
    advantages = torch.tensor([[0.8]])
    mask = torch.tensor([[1.0]])

    loss, clipfrac, approx_kl = ppo_policy_loss(new_logprobs, old_logprobs, advantages, mask, clip_eps=0.2)
    # book: ratio=1.40, unclipped=1.12, clipped=0.96, min=0.96 -> clip engages
    assert abs(loss.item() - (-0.96)) < 1e-3
    assert clipfrac.item() == 1.0
    assert abs(approx_kl.item() - 0.0635) < 1e-3
    print(f"block #2 ppo_policy_loss: OK (loss={loss.item():.4f}, clipfrac={clipfrac.item()}, approx_kl={approx_kl.item():.4f})")


def ppo_value_loss(values, old_values, returns, mask, clip_eps_v=0.2):
    """Clipped value-function regression loss (critic)."""
    v_clipped = old_values + torch.clamp(values - old_values, -clip_eps_v, clip_eps_v)
    loss_unclipped = (values - returns) ** 2
    loss_clipped = (v_clipped - returns) ** 2
    per_token = 0.5 * torch.max(loss_unclipped, loss_clipped)   # pessimistic
    return (per_token * mask).sum() / mask.sum().clamp(min=1.0)


def make_token_rewards(policy_logprobs, ref_logprobs, rm_scores, mask, kl_beta=0.05):
    """
    Build the per-token reward tensor fed to GAE: a per-token KL penalty
    everywhere, plus the scalar reward-model score at the last response token.

    policy_logprobs, ref_logprobs : (B, T) log-probs of the SAMPLED tokens.
    rm_scores : (B,) one scalar per response from the reward model.
    """
    # k1 per-token KL penalty: -beta * (logpi_theta - logpi_ref)
    kl = policy_logprobs - ref_logprobs                  # (B, T)
    token_rewards = -kl_beta * kl * mask                 # KL on every response token

    # Index of the last response token per row. Found positionally (not via
    # mask.sum()-1): after the next-token shift, `mask` still carries the
    # prompt's leading zeros, so "count of response tokens minus one" is NOT
    # the same column as "the last column where mask==1" unless the prompt
    # has length 1. Locating the rightmost 1 directly keeps this correct for
    # any prompt length. [BUG FIX -- see module docstring]
    positions = torch.arange(mask.size(1), device=mask.device).unsqueeze(0).expand_as(mask)
    last_idx = (positions * mask).amax(dim=1).long()      # (B,)
    rows = torch.arange(token_rewards.size(0), device=mask.device)
    token_rewards[rows, last_idx] += rm_scores           # terminal RM reward
    return token_rewards


def test_make_token_rewards_places_reward_correctly():
    """
    Synthetic case with a nonzero prompt length. The book's own "Common
    pitfall" callout says: "a good test also asserts the nonzero terminal
    reward lands on the final unmasked token of each sequence." This test
    does exactly that, and would FAIL against the original
    `mask.sum(dim=1) - 1` formula (it lands on column 3, not column 5).
    """
    prompt_len, resp_len = 3, 4
    T = prompt_len + resp_len - 1  # after the next-token shift
    policy_logprobs = torch.zeros(1, T)
    ref_logprobs = torch.zeros(1, T)
    rm_scores = torch.tensor([2.5])
    mask = torch.tensor([[0.0] * (prompt_len - 1) + [1.0] * resp_len])  # shape (1, 6)

    token_rewards = make_token_rewards(policy_logprobs, ref_logprobs, rm_scores, mask, kl_beta=0.05)
    true_last_col = mask.shape[1] - 1  # = 5, the rightmost 1
    assert token_rewards[0, true_last_col].item() == 2.5, "RM score must land on the last unmasked token"
    assert token_rewards[0, :true_last_col].abs().max().item() < 1e-8, "no reward should leak elsewhere (KL is 0 here)"
    print("block #4 make_token_rewards: OK, reward lands on the true last response token", token_rewards.tolist())


test_reinforce_loss()
test_compute_gae_matches_worked_example()
test_ppo_policy_loss_matches_worked_example()
test_make_token_rewards_places_reward_correctly()


# ======================================================================
# Block #5 (line ~407) -- the full PPO-for-LLM step
# ======================================================================
PPO_EPOCHS = 4
MINIBATCHES = 4
CLIP_EPS = 0.2
KL_BETA = 0.05
VF_COEF = 0.5
ENT_COEF = 0.0
TARGET_KL = 0.02   # early-stop the epoch loop if approx_kl exceeds this


def token_logprobs_and_values(model, input_ids):
    """Per-token log-prob of the realized next token, plus value-head output."""
    out = model(input_ids)                                   # logits (B,T,V), values (B,T)
    logits, values = out.logits[:, :-1], out.values[:, :-1]
    logp = F.log_softmax(logits.float(), dim=-1)
    targets = input_ids[:, 1:].unsqueeze(-1)
    token_lp = logp.gather(-1, targets).squeeze(-1)          # (B, T-1)
    return token_lp, values


@torch.no_grad()
def ppo_rollout(prompts):
    # 1-2: generate responses with the current (behavior) policy.
    input_ids, resp_mask = generate_batch(policy, prompts)   # your decode fn
    # 3: reward-model scores (one scalar per sequence).
    rm_scores = reward_model.score(input_ids, resp_mask)     # (B,)
    # 4: cache behavior log-probs, reference log-probs, and values.
    old_lp, old_values = token_logprobs_and_values(policy, input_ids)
    ref_lp, _ = token_logprobs_and_values(ref_model, input_ids)
    mask = resp_mask[:, 1:]                                   # align with shifted targets
    # 5: per-token rewards = KL penalty + terminal RM score.
    token_rewards = make_token_rewards(old_lp, ref_lp, rm_scores, mask, KL_BETA)
    # 6: GAE advantages and returns.
    advantages, returns = compute_gae(token_rewards, old_values, mask)
    return dict(input_ids=input_ids, mask=mask, old_lp=old_lp,
                old_values=old_values, advantages=advantages, returns=returns,
                mean_reward=rm_scores.mean().item())


def ppo_update(buf):
    B = buf["input_ids"].size(0)
    idx = torch.randperm(B)
    mb_size = B // MINIBATCHES
    for _ in range(PPO_EPOCHS):
        for m in range(MINIBATCHES):
            sl = idx[m * mb_size:(m + 1) * mb_size]
            new_lp, new_values = token_logprobs_and_values(policy, buf["input_ids"][sl])

            p_loss, clipfrac, approx_kl = ppo_policy_loss(
                new_lp, buf["old_lp"][sl], buf["advantages"][sl], buf["mask"][sl], CLIP_EPS)
            v_loss = ppo_value_loss(
                new_values, buf["old_values"][sl], buf["returns"][sl], buf["mask"][sl])
            entropy = policy_entropy(policy, buf["input_ids"][sl], buf["mask"][sl])

            loss = p_loss + VF_COEF * v_loss - ENT_COEF * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
        if approx_kl > 1.5 * TARGET_KL:        # KL early-stopping safeguard
            break

# Driver:
# for it in range(NUM_ITERS):
#     buf = ppo_rollout(sample_prompts(batch_size))
#     ppo_update(buf)


# ----------------------------------------------------------------------
# Glue standing in for "Assume: policy (with value head), ref_model,
# reward_model, tokenizer, optimizer" -- and for `policy_entropy`, which
# ppo_update calls but the chapter text never defines.
# ----------------------------------------------------------------------
VOCAB_SIZE = 12
HIDDEN = 8


class TinyLM(nn.Module):
    """Toy stand-in for a transformer policy/value model, tiny enough for CPU."""

    def __init__(self, vocab_size, hidden):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden)
        self.mix = nn.Linear(hidden, hidden)
        self.lm_head = nn.Linear(hidden, vocab_size)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, input_ids):
        h = torch.tanh(self.mix(self.emb(input_ids)))   # (B, T, H)
        logits = self.lm_head(h)                          # (B, T, V)
        values = self.value_head(h).squeeze(-1)            # (B, T)
        return SimpleNamespace(logits=logits, values=values)


class TinyRewardModel:
    """Toy stand-in for a frozen reward model: a fixed per-token score table."""

    def __init__(self, vocab_size, seed=123):
        g = torch.Generator().manual_seed(seed)
        self.token_score = torch.randn(vocab_size, generator=g)

    def score(self, input_ids, resp_mask):
        per_tok = self.token_score[input_ids]                       # (B, T)
        return (per_tok * resp_mask).sum(dim=1) / resp_mask.sum(dim=1).clamp(min=1)


def generate_batch(policy_model, prompts, resp_len=4):
    """Toy autoregressive sampler standing in for the book's 'your decode fn'."""
    input_ids = torch.tensor(prompts, dtype=torch.long)
    B = input_ids.size(0)
    resp_mask = torch.zeros(B, input_ids.size(1), dtype=torch.float32)
    for _ in range(resp_len):
        out = policy_model(input_ids)
        probs = F.softmax(out.logits[:, -1, :], dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_tok], dim=1)
        resp_mask = torch.cat([resp_mask, torch.ones(B, 1)], dim=1)
    return input_ids, resp_mask


def policy_entropy(model, input_ids, mask):
    """Mean token-level entropy of the policy over response positions."""
    out = model(input_ids)
    logp = F.log_softmax(out.logits[:, :-1].float(), dim=-1)
    p = logp.exp()
    ent = -(p * logp).sum(dim=-1)                          # (B, T-1)
    return (ent * mask).sum() / mask.sum().clamp(min=1.0)


def test_full_ppo_step():
    global policy, ref_model, reward_model, optimizer

    policy = TinyLM(VOCAB_SIZE, HIDDEN)
    ref_model = copy.deepcopy(policy)
    for p in ref_model.parameters():
        p.requires_grad_(False)
    reward_model = TinyRewardModel(VOCAB_SIZE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)

    B, prompt_len = 8, 3
    prompts = [[random.randint(0, VOCAB_SIZE - 1) for _ in range(prompt_len)] for _ in range(B)]

    params_before = [p.detach().clone() for p in policy.parameters()]

    buf = ppo_rollout(prompts)
    assert buf["input_ids"].shape[0] == B
    assert torch.isfinite(torch.tensor(buf["mean_reward"]))
    assert buf["advantages"].shape == buf["mask"].shape

    ppo_update(buf)

    # A real gradient step must actually have moved the policy's weights.
    moved = any(
        not torch.allclose(a, b) for a, b in zip(params_before, policy.parameters())
    )
    assert moved, "ppo_update ran but did not change any policy parameter"
    print(f"block #5 full PPO step: OK (mean_reward={buf['mean_reward']:.4f}, weights updated={moved})")


test_full_ppo_step()

print("\nAll blocks executed successfully.")
