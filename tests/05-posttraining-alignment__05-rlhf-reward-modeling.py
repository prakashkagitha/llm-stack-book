"""
Executable test for content/05-posttraining-alignment/05-rlhf-reward-modeling.md

Assembles the chapter's CPU-runnable Python blocks, in the order they appear
in the chapter, into one runnable module.

Blocks in the chapter:
    #0 (line ~34)  - a JSON snippet (non-Python). SKIP(non-python).
    #1 (line ~120) - RewardModel, bradley_terry_loss, train_step. TESTED.
    #2 (line ~207) - all_pairs_bt_loss, the InstructGPT all-C(K,2)-pairs-from-
                      one-prompt regularizer. TESTED. It is a fully
                      self-contained, CPU-runnable function, so it is executed
                      here (verbatim from the book) on a ranked set of scores.
    #3 (line ~328) - rlhf_ppo_epoch, the four-model PPO control-flow skeleton.
                      TESTED. The chapter explicitly says the real PPO update
                      (`ppo_update`) is "intentionally abstracted" and left
                      for the next chapter, and it never defines the small
                      utilities `build_sequences_and_mask` / `token_logprobs`
                      it calls -- they are assumed obvious plumbing. This test
                      supplies minimal, clearly-labeled glue for exactly those
                      three names (a toy tokenizer/actor/critic/reward model
                      and a toy `ppo_update`) so `rlhf_ppo_epoch`'s OWN code
                      (copied verbatim from the book) actually executes and
                      every one of the four models gets called.

Design choice that keeps the glue honest and shape-consistent with the book's
own indexing (`last = resp_mask.sum(dim=1) - 1` used directly as an index into
a (B, T) reward tensor): every prompt is tokenized to the SAME length and the
toy actor always generates the full `max_new_tokens`, so `resp_mask` is simply
all-ones and response tokens are exactly the trailing R positions of `seq` for
every batch element -- i.e. response-token position i inside the (B, R)
tensors returned by `token_logprobs`/`critic` IS position i of the response,
with no padding to reconcile. This mirrors block #1's own trick (a strictly
right-padded sequence where `attention_mask.sum(dim=1) - 1` is the last real
token) rather than reinterpreting the chapter's code.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Block #1 (line ~120): the reward-model core -- RewardModel, bradley_terry_loss,
# train_step. Copied verbatim from the chapter.
# =============================================================================

class RewardModel(nn.Module):
    """A transformer backbone with a scalar value head on top.

    `backbone` is any model returning per-token hidden states of shape
    (batch, seq_len, d_model) -- e.g. an SFT-initialized decoder with its
    language-modeling head removed.
    """
    def __init__(self, backbone, d_model):
        super().__init__()
        self.backbone = backbone
        # A single linear layer: d_model -> 1 scalar. Initialize small so early
        # rewards are near zero (helps optimization stability downstream).
        self.value_head = nn.Linear(d_model, 1, bias=False)
        nn.init.normal_(self.value_head.weight, std=1.0 / (d_model + 1) ** 0.5)

    def forward(self, input_ids, attention_mask):
        # hidden: (B, T, d_model)
        hidden = self.backbone(input_ids, attention_mask=attention_mask).last_hidden_state
        # Per-token scalar scores: (B, T)
        scores = self.value_head(hidden).squeeze(-1)
        # Reward is read at the LAST non-pad token of each sequence.
        # attention_mask is 1 for real tokens, 0 for padding.
        last_idx = attention_mask.sum(dim=1) - 1            # (B,) index of final real token
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        reward = scores[batch_idx, last_idx]                # (B,) one scalar per sequence
        return reward


def bradley_terry_loss(reward_chosen, reward_rejected, margin=0.0):
    """The Bradley-Terry / pairwise ranking loss.

    reward_chosen, reward_rejected : (B,) scalar reward for the preferred and
                                     dispreferred response in each pair.
    margin : optional fixed margin m. Loss becomes -log sigmoid(r_w - r_l - m),
             which only counts the pair as "solved" once r_w beats r_l by m.
             Useful when you have a graded preference strength (set m larger for
             strongly-preferred pairs). Default 0 recovers plain Bradley-Terry.

    Returns (loss, accuracy) where accuracy is the fraction of pairs the model
    currently orders correctly -- the single most important RM training metric.
    """
    delta = reward_chosen - reward_rejected - margin        # the margin Δ
    # -log σ(Δ) == softplus(-Δ), computed stably:
    loss = F.softplus(-delta).mean()
    # A pair is "correct" iff the chosen response scores higher than the rejected.
    accuracy = (reward_chosen > reward_rejected).float().mean()
    return loss, accuracy


def train_step(rm, optimizer, batch):
    """One optimization step on a batch of preference pairs.

    `batch` provides chosen/rejected token ids and masks, already tokenized as
    [prompt + response]. We run BOTH responses through the SAME reward model and
    compare their scalar scores. This shared-encoder, two-forward-pass structure
    is exactly what TRL's RewardTrainer and DPO-style code do under the hood.
    """
    rm.train()
    # Forward both completions of every pair through the same network.
    r_chosen   = rm(batch["chosen_ids"],   batch["chosen_mask"])     # (B,)
    r_rejected = rm(batch["rejected_ids"], batch["rejected_mask"])   # (B,)

    loss, acc = bradley_terry_loss(r_chosen, r_rejected)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)   # RMs train fast; clip helps
    optimizer.step()
    return {"loss": loss.item(), "pref_acc": acc.item(),
            "reward_margin": (r_chosen - r_rejected).mean().item()}


# --- Test-only glue for block #1: a tiny backbone standing in for "the SFT
# model with its LM head removed", exposing the `.last_hidden_state` interface
# RewardModel.forward expects. ---------------------------------------------

class _TinyBackboneOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class TinyBackbone(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, input_ids, attention_mask=None):
        h = self.proj(self.embed(input_ids))
        return _TinyBackboneOutput(h)


def test_reward_model_block():
    torch.manual_seed(0)
    vocab_size, d_model = 24, 8

    backbone = TinyBackbone(vocab_size, d_model)
    rm = RewardModel(backbone, d_model)
    optimizer = torch.optim.SGD(rm.parameters(), lr=0.5)

    # A tiny batch of 4 preference pairs, sequence length 6, with VARYING real
    # lengths per row (via attention_mask) to actually exercise the
    # "last non-pad token" indexing logic in RewardModel.forward.
    B, T = 4, 6
    chosen_ids = torch.randint(1, vocab_size, (B, T))
    rejected_ids = torch.randint(1, vocab_size, (B, T))
    chosen_lens = [6, 5, 4, 6]
    rejected_lens = [6, 6, 3, 5]
    chosen_mask = torch.zeros(B, T, dtype=torch.long)
    rejected_mask = torch.zeros(B, T, dtype=torch.long)
    for i, L in enumerate(chosen_lens):
        chosen_mask[i, :L] = 1
    for i, L in enumerate(rejected_lens):
        rejected_mask[i, :L] = 1

    batch = {
        "chosen_ids": chosen_ids, "chosen_mask": chosen_mask,
        "rejected_ids": rejected_ids, "rejected_mask": rejected_mask,
    }

    losses = []
    for _ in range(5):
        stats = train_step(rm, optimizer, batch)
        losses.append(stats["loss"])
        assert stats["loss"] == stats["loss"], "loss is NaN"
        assert 0.0 <= stats["pref_acc"] <= 1.0

    # Training on a fixed batch should not blow up, and typically pushes the
    # loss down (not asserted strictly -- 5 SGD steps on a tiny random init
    # aren't guaranteed to be monotone, but it must stay finite and bounded).
    assert all(l < 50.0 for l in losses)

    # --- Verify the book's own worked example (the "Worked example" box). ---
    # r_w = 2.3, r_l = 0.8  ->  delta = 1.5
    loss_ex1, acc_ex1 = bradley_terry_loss(torch.tensor([2.3]), torch.tensor([0.8]))
    assert acc_ex1.item() == 1.0
    assert abs(loss_ex1.item() - 0.201) < 0.01           # book: ~0.201 nats
    prob_ex1 = torch.sigmoid(torch.tensor(1.5)).item()
    assert abs(prob_ex1 - 0.817) < 0.005                  # book: sigma(1.5) ~= 0.817

    # r_w = 0.5, r_l = 1.9 (backwards)  ->  delta = -1.4
    loss_ex2, acc_ex2 = bradley_terry_loss(torch.tensor([0.5]), torch.tensor([1.9]))
    assert acc_ex2.item() == 0.0
    assert abs(loss_ex2.item() - 1.62) < 0.01             # book: ~1.62 nats, 8x larger

    # Additive-constant invariance: shifting both rewards by +100 changes nothing.
    loss_shift, _ = bradley_terry_loss(torch.tensor([102.3]), torch.tensor([100.8]))
    assert abs(loss_shift.item() - loss_ex1.item()) < 1e-5

    print("block #1 (RewardModel/bradley_terry_loss/train_step) OK ->",
          {"losses": [round(l, 4) for l in losses]})


# =============================================================================
# Block #2 (line ~207): the InstructGPT all-pairs-from-one-prompt regularizer.
# Copied verbatim from the chapter.
# =============================================================================

def all_pairs_bt_loss(rewards_K, chosen_better_mask):
    """rewards_K : (K,) scalar reward for the K completions of ONE prompt,
                   ordered by the human ranking (index 0 = most preferred).
       Since they are ranked, every pair (i, j) with i < j has y_i preferred.
    Computes the mean BT loss over all C(K,2) pairs, encoding each completion once.
    """
    K = rewards_K.shape[0]
    # Pairwise score differences Δ_ij = r_i - r_j for all i<j.
    diff = rewards_K.unsqueeze(1) - rewards_K.unsqueeze(0)   # (K, K), Δ_ij at [i, j]
    iu = torch.triu_indices(K, K, offset=1)                  # upper triangle: i<j
    deltas = diff[iu[0], iu[1]]                              # (C(K,2),) all preferred-minus-dispreferred
    return F.softplus(-deltas).mean()


def test_all_pairs_bt_loss_block():
    # A perfectly-ordered (descending) score vector -- every one of the C(K,2)
    # pairs is ranked correctly, so the mean BT loss should be small and
    # strictly less than log 2 (the loss at a tie).
    K = 5
    ordered = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0])
    loss_ordered = all_pairs_bt_loss(ordered, None)
    import math
    assert loss_ordered.item() < math.log(2.0)

    # All-equal scores -> every pair contributes exactly softplus(0) = log 2.
    tied = torch.zeros(K)
    loss_tied = all_pairs_bt_loss(tied, None)
    assert abs(loss_tied.item() - math.log(2.0)) < 1e-6

    # Reversing the ranking (model has every pair backwards) must be strictly
    # worse than both the ordered and the tied case.
    reversed_scores = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    loss_reversed = all_pairs_bt_loss(reversed_scores, None)
    assert loss_reversed.item() > loss_tied.item() > loss_ordered.item()

    # Additive-constant invariance also holds for the all-pairs form.
    loss_shift = all_pairs_bt_loss(ordered + 100.0, None)
    assert abs(loss_shift.item() - loss_ordered.item()) < 1e-5

    # Sanity: C(K,2) pairs actually feed the mean (K*(K-1)/2 upper-tri entries).
    diff = ordered.unsqueeze(1) - ordered.unsqueeze(0)
    iu = torch.triu_indices(K, K, offset=1)
    assert iu.shape[1] == K * (K - 1) // 2

    print("block #2 (all_pairs_bt_loss) OK ->",
          {"ordered": round(loss_ordered.item(), 4),
           "tied": round(loss_tied.item(), 4),
           "reversed": round(loss_reversed.item(), 4)})


# =============================================================================
# Block #3 (line ~328): the four-model RLHF/PPO control-flow skeleton.
# Copied verbatim from the chapter (only the two helper calls
# `build_sequences_and_mask` / `token_logprobs` and the `ppo_update` argument
# are left as the deliberately-abstracted names the chapter itself uses).
# =============================================================================

def rlhf_ppo_epoch(actor, critic, reward_model, ref_model,
                   prompts, tokenizer, beta_kl=0.02, ppo_update=None):
    """One outer iteration of the RLHF/PPO loop.

    actor       : policy π_θ        (trained)   -- generates and is updated
    critic      : value V_ψ         (trained)   -- baseline for advantages
    reward_model: r_φ               (frozen)    -- terminal scalar reward
    ref_model   : π_ref (SFT copy)  (frozen)    -- KL anchor
    """
    # ---- 1. ROLLOUT: actor generates responses to a batch of prompts. ----
    #     In production this runs on a fast inference engine (vLLM/SGLang);
    #     see "The Generation–Training Loop & Rollout Engines".
    queries = tokenizer(prompts, return_tensors="pt", padding=True)
    with torch.no_grad():
        responses = actor.generate(**queries, max_new_tokens=512, do_sample=True)

    # full sequence = prompt ++ response; build a mask marking response tokens
    seq, resp_mask = build_sequences_and_mask(queries, responses)

    # ---- 2. SCORE & ANCHOR (all under no_grad; these models are not updated). ----
    with torch.no_grad():
        # Terminal scalar reward for each complete response (frozen RM).
        scores = reward_model(seq, attention_mask=(seq != tokenizer.pad_token_id))  # (B,)

        # Per-token log-probs from the FROZEN reference (for the KL penalty).
        ref_logprobs = token_logprobs(ref_model, seq, resp_mask)                    # (B, T)

    # ---- 3. The actor's own per-token log-probs and the critic's values
    #         (these DO require grad -- they define the PPO objective). ----
    actor_logprobs = token_logprobs(actor, seq, resp_mask)                          # (B, T)
    values         = critic(seq, resp_mask)                                         # (B, T)

    # ---- 4. Build the per-token reward: KL penalty everywhere + RM score at end. ----
    kl_per_token = actor_logprobs.detach() - ref_logprobs                           # (B, T)
    rewards = -beta_kl * kl_per_token                                               # KL "rent"
    last = resp_mask.sum(dim=1) - 1                                                 # final resp idx
    rewards[torch.arange(rewards.size(0)), last] += scores                          # add terminal r_φ

    # ---- 5. PPO update: compute advantages (GAE) from (rewards, values), then
    #         take clipped policy-gradient + value-function steps. See chapter 5.6. ----
    stats = ppo_update(actor, critic,
                       logprobs=actor_logprobs, old_logprobs=actor_logprobs.detach(),
                       values=values, rewards=rewards, resp_mask=resp_mask)
    return stats


# --- Test-only glue for block #3 ------------------------------------------
#
# The chapter explicitly abstracts `ppo_update` away ("the next chapter
# builds in full") and never defines `build_sequences_and_mask` /
# `token_logprobs` -- they're treated as obvious utility functions. None of
# this glue reimplements or bypasses the logic the block under test
# (`rlhf_ppo_epoch`) is demonstrating; it only supplies toy actor/critic/
# reward/reference models and a tokenizer small enough to run on CPU in
# milliseconds, plus the named helper functions with the simplest behavior
# consistent with how `rlhf_ppo_epoch` uses their outputs.

class _TinyLMOutput:
    def __init__(self, logits):
        self.logits = logits


class TinyLM(nn.Module):
    """Toy causal-LM stand-in used for both the actor and the reference model.
    No attention (predicts a distribution from each token's own embedding) --
    this keeps `.generate` cheap enough to run 512 autoregressive steps on CPU
    while still exercising real forward/backward passes through nn.Linear.
    """
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        h = self.embed(input_ids)
        logits = self.head(h)
        return _TinyLMOutput(logits)

    def generate(self, input_ids, attention_mask=None, max_new_tokens=50, do_sample=True):
        ids = input_ids
        for _ in range(max_new_tokens):
            logits = self.forward(ids).logits[:, -1, :].clone()
            logits[:, 0] = -1e9  # reserve id 0 for pad_token_id; never generate it
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
            else:
                nxt = logits.argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
        return ids


class TinyCritic(nn.Module):
    """Toy value network: scalar-per-position head over the response segment."""
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, seq, resp_mask):
        R = resp_mask.shape[1]
        L = seq.shape[1]
        h = self.embed(seq[:, L - R:])                # (B, R, d) response-only hidden states
        v = self.value_head(h).squeeze(-1)             # (B, R)
        return v * resp_mask


class TinyTokenizer:
    """Deterministic whitespace tokenizer. Test prompts are chosen to have
    equal word counts so `padding=True` never actually pads -- this keeps the
    "response = trailing R tokens of seq" convention exact, matching how
    `rlhf_ppo_epoch` indexes `resp_mask.sum(dim=1) - 1` directly into the
    (B, T) reward tensor.
    """
    def __init__(self, vocab_size, pad_token_id=0):
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

    def _tok(self, word):
        return (sum(ord(c) for c in word) % (self.vocab_size - 1)) + 1  # never 0 (pad)

    def __call__(self, prompts, return_tensors="pt", padding=True):
        tokenized = [[self._tok(w) for w in p.split()] for p in prompts]
        maxlen = max(len(t) for t in tokenized)
        input_ids = torch.full((len(prompts), maxlen), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(prompts), maxlen), dtype=torch.long)
        for i, t in enumerate(tokenized):
            input_ids[i, :len(t)] = torch.tensor(t, dtype=torch.long)
            attention_mask[i, :len(t)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def build_sequences_and_mask(queries, responses):
    """TEST GLUE standing in for the chapter's unshown utility. `responses` is
    already the full (prompt ++ generated) sequence, HF-`generate`-style.
    `resp_mask` marks the newly-generated suffix; since every prompt in this
    test tokenizes to the same length and generation always runs the full
    `max_new_tokens`, the response occupies exactly the trailing R positions
    for every row, so a simple all-ones (B, R) mask is exact (no padding to
    reconcile).
    """
    P = queries["input_ids"].shape[1]
    seq = responses
    R = seq.shape[1] - P
    B = seq.shape[0]
    resp_mask = torch.ones(B, R, dtype=torch.long)
    return seq, resp_mask


def token_logprobs(model, seq, resp_mask):
    """TEST GLUE standing in for the chapter's unshown utility: per-token
    log-prob of each actual response token, restricted to the response
    segment (the trailing R positions of `seq`, R = resp_mask.shape[1]).
    """
    R = resp_mask.shape[1]
    L = seq.shape[1]
    logits = model(seq).logits                       # (B, L, V)
    logp_all = F.log_softmax(logits, dim=-1)
    resp_tokens = seq[:, L - R:]                      # (B, R) actual response token ids
    resp_logp = logp_all[:, L - R:, :]                # (B, R, V)
    logp = torch.gather(resp_logp, 2, resp_tokens.unsqueeze(-1)).squeeze(-1)
    return logp * resp_mask


def make_ppo_update(actor_opt, critic_opt):
    """TEST GLUE: a minimal, honest stand-in for the real clipped-PPO/GAE
    update, which the chapter explicitly defers to the next chapter
    ("the actual PPO update ... is intentionally abstracted into
    `ppo_update`"). This does NOT claim to be PPO -- it is just enough of a
    real advantage-weighted policy-gradient + value-regression step to prove
    `rlhf_ppo_epoch`'s data (rewards, values, logprobs, resp_mask) flows into
    a working optimizer step with correct shapes.
    """
    def ppo_update(actor, critic, logprobs, old_logprobs, values, rewards, resp_mask):
        mask = resp_mask.float()
        denom = mask.sum().clamp(min=1.0)
        advantages = (rewards - values.detach()) * mask
        policy_loss = -(advantages * logprobs * mask).sum() / denom
        value_loss = (((values - rewards) * mask) ** 2).sum() / denom
        loss = policy_loss + 0.5 * value_loss

        actor_opt.zero_grad()
        critic_opt.zero_grad()
        loss.backward()
        actor_opt.step()
        critic_opt.step()
        return {"policy_loss": policy_loss.item(), "value_loss": value_loss.item(),
                "mean_reward": rewards.sum().item() / denom.item()}
    return ppo_update


def test_rlhf_ppo_epoch_block():
    torch.manual_seed(0)
    vocab_size, d_model = 24, 8

    actor = TinyLM(vocab_size, d_model)
    ref_model = copy.deepcopy(actor)              # frozen SFT snapshot
    for p in ref_model.parameters():
        p.requires_grad_(False)

    critic = TinyCritic(vocab_size, d_model)

    # Reuse the chapter's OWN RewardModel (block #1) as the frozen reward
    # model in the four-model setup, exactly as the chapter describes it.
    reward_backbone = TinyBackbone(vocab_size, d_model)
    reward_model = RewardModel(reward_backbone, d_model)
    for p in reward_model.parameters():
        p.requires_grad_(False)

    tokenizer = TinyTokenizer(vocab_size)
    prompts = ["explain gravity briefly", "write a poem"]  # 3 words each -> equal length

    actor_opt = torch.optim.SGD(actor.parameters(), lr=0.01)
    critic_opt = torch.optim.SGD(critic.parameters(), lr=0.01)
    ppo_update = make_ppo_update(actor_opt, critic_opt)

    # rlhf_ppo_epoch hardcodes max_new_tokens=512 internally (left untouched,
    # verbatim from the book); the toy models are tiny enough that 512
    # autoregressive steps still finish in a couple of seconds on CPU.
    stats = rlhf_ppo_epoch(actor, critic, reward_model, ref_model,
                            prompts, tokenizer, beta_kl=0.02, ppo_update=ppo_update)

    assert set(stats.keys()) == {"policy_loss", "value_loss", "mean_reward"}
    for k, v in stats.items():
        assert v == v, f"{k} is NaN"          # NaN != NaN
        assert abs(v) < 1e6, f"{k} exploded: {v}"

    print("block #3 (rlhf_ppo_epoch, four-model PPO skeleton) OK ->", stats)


if __name__ == "__main__":
    test_reward_model_block()
    test_all_pairs_bt_loss_block()
    test_rlhf_ppo_epoch_block()
    print("ALL BLOCKS EXECUTED OK")
