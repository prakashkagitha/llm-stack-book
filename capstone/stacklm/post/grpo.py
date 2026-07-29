"""Narrow RLVR with GRPO (Shao et al., 2024) on a verifiable arithmetic task.

Reward = exact-match correctness. Group-relative, std-normalized advantage
(GRPO, Shao et al. 2024), a clipped surrogate with decoupled bounds
("clip-higher", DAPO, Yu et al. 2025), token-level policy-gradient
normalization (also DAPO -- NOT Dr.GRPO's constant normalizer), and a k3 KL
penalty against a frozen reference. See Ch. 14.9 for the Dr.GRPO ablation
(drop the std division AND the per-sequence normalizer -- both, or neither).

Minimal but real: single process, no KV cache, synchronous generate-then-train.
"""
import copy
import random
import re

import torch

from ..optim import build_optimizer
from ..train.loop import autocast_ctx
from .chat import render_conversation, Turn, SPECIAL


def make_arithmetic_prompt(rng, max_val=99):
    a, b = rng.randint(2, max_val), rng.randint(2, max_val)
    op = rng.choice(["+", "-", "*"])
    ans = {"+": a + b, "-": a - b, "*": a * b}[op]
    question = f"Compute {a} {op} {b}. Give the final integer after '####'."
    return question, ans


_FINAL = re.compile(r"####\s*(-?\d+)")


def exact_match_reward(completion_text, gold_answer):
    m = _FINAL.search(completion_text)
    if m is None:
        return 0.0, None
    try:
        pred = int(m.group(1))
    except ValueError:
        return 0.0, None
    return (1.0 if pred == gold_answer else 0.0), pred


_ANY_INT = re.compile(r"-?\d+")


def shaped_reward(completion_text, gold_answer, fmt_weight=0.1):
    """Exact-match reward plus a small format bonus for emitting the '####'
    marker and any integer -- gives RLVR a denser early signal (Ch. 14.9 exercise)."""
    correct, pred = exact_match_reward(completion_text, gold_answer)
    bonus = 0.0
    if "####" in completion_text:
        bonus += fmt_weight
    if _ANY_INT.search(completion_text):
        bonus += fmt_weight
    return correct + bonus, pred


@torch.no_grad()
def sample_group(model, tok, prompt_ids, G, max_new=32, temperature=1.0, device="cpu"):
    model.eval()
    end_id = tok.special_token_id(SPECIAL["end"])
    eos_id = tok.special_token_id(SPECIAL["eos"])
    x = prompt_ids.to(device).unsqueeze(0).repeat(G, 1)     # (G, Tp)
    Tp = x.size(1)
    cap = model.cfg.max_seq_len                              # keep seqs within RoPE cache
    done = torch.zeros(G, dtype=torch.bool, device=device)
    for _ in range(max_new):
        if x.size(1) >= cap:
            break
        logits, _ = model(x)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits.float() / temperature, dim=-1)
        nxt = torch.multinomial(probs, 1)                    # (G, 1)
        nxt[done] = eos_id
        x = torch.cat([x, nxt], dim=1)
        done |= (nxt.squeeze(1) == end_id) | (nxt.squeeze(1) == eos_id)
        if done.all():
            break
    gen_mask = torch.zeros_like(x, dtype=torch.float)
    gen_mask[:, Tp:] = 1.0                                   # completion tokens
    # Zero the mask past each row's FIRST stop token: the stop itself is
    # supervised (we want the model to emit it), everything after it is eos
    # padding and must earn neither advantage nor KL penalty.
    for i in range(G):
        row = x[i, Tp:]
        stop = ((row == end_id) | (row == eos_id)).nonzero()
        if len(stop):
            gen_mask[i, Tp + int(stop[0].item()) + 1:] = 0.0
    return x, gen_mask, Tp


def token_logprobs(model, seqs):
    # seqs are guaranteed <= max_seq_len (sample_group caps growth), so no slicing.
    logits, _ = model(seqs)
    logits = logits[:, :-1, :]
    logp = torch.log_softmax(logits.float(), dim=-1)
    return torch.gather(logp, 2, seqs[:, 1:].unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def collect_nondegenerate_groups(policy, tok, rng, *, target_groups, group_size,
                                 reward_fn=exact_match_reward, max_new=32,
                                 temperature=1.0, oversample=3.0, device="cpu"):
    """DAPO-style dynamic sampling (Yu et al., 2025).

    Keep drawing prompts and DISCARD every group whose samples all score the
    same (accuracy 0 or 1), until we have `target_groups` groups that actually
    carry gradient. Returns (groups, tries); each group is (seqs, gmask,
    rewards, Tp). `tries / len(groups)` is the single most informative RLVR
    health metric: ~1.0 in the sweet spot, blowing up as the task drifts out of
    reach. Budget-capped by `oversample` so a hopeless task fails fast.
    """
    kept, tries, budget = [], 0, int(target_groups * oversample)
    while len(kept) < target_groups and tries < budget:
        tries += 1
        q, gold = make_arithmetic_prompt(rng)
        p_ids, _ = render_conversation([Turn("user", q)], tok,
                                       add_generation_prompt=True)
        p_ids = p_ids[: max(1, policy.cfg.max_seq_len - max_new - 1)]
        p_ids = torch.tensor(p_ids, dtype=torch.long)
        seqs, gmask, Tp = sample_group(policy, tok, p_ids, group_size,
                                       max_new=max_new, temperature=temperature,
                                       device=device)
        rewards = torch.tensor(
            [reward_fn(tok.decode(seqs[i, Tp:].tolist()), gold)[0]
             for i in range(group_size)], device=device)
        if rewards.max() == rewards.min():
            continue                       # degenerate: zero std, zero gradient
        kept.append((seqs, gmask, rewards, Tp))
    return kept, tries


def grpo_train(sft_model, tok, *, iterations=5, group_size=6, prompts_per_iter=4,
               inner_epochs=1, lr=1e-6, clip_eps_low=0.2, clip_eps_high=0.28,
               kl_beta=0.02, temperature=1.0, max_new=32, device="cpu", seed=0,
               log_every=1, reward_fn=exact_match_reward):
    rng = random.Random(seed)
    device = torch.device(device)
    policy = sft_model.to(device)
    ref = copy.deepcopy(policy).to(device).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = build_optimizer(policy, lr=lr, weight_decay=0.0, betas=(0.9, 0.95))

    accs, losses, degen = [], [], []
    for it in range(iterations):
        batch_seqs, batch_gmask, batch_adv, batch_oldlp = [], [], [], []
        n_correct, n_total, n_degenerate = 0, 0, 0
        for _ in range(prompts_per_iter):
            q, gold = make_arithmetic_prompt(rng)
            p_ids, _ = render_conversation([Turn("user", q)], tok,
                                           add_generation_prompt=True)
            # leave room for generation within the RoPE cache
            p_ids = p_ids[: max(1, policy.cfg.max_seq_len - max_new - 1)]
            p_ids = torch.tensor(p_ids, dtype=torch.long)
            seqs, gmask, Tp = sample_group(policy, tok, p_ids, group_size,
                                           max_new=max_new, temperature=temperature,
                                           device=device)
            rewards = torch.zeros(group_size, device=device)
            for i in range(group_size):
                text = tok.decode(seqs[i, Tp:].tolist())
                r, _ = reward_fn(text, gold)
                rewards[i] = r
            # count exact matches (a shaped reward adds a <1.0 format bonus)
            n_correct += int((rewards >= 1.0).sum().item())
            n_total += group_size
            if rewards.max() == rewards.min():
                n_degenerate += 1        # zero std -> zero advantage -> no gradient
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)   # (G,)
            with torch.no_grad():
                old_lp = token_logprobs(policy, seqs)                   # (G, T-1)
            batch_seqs.append(seqs)
            batch_gmask.append(gmask)
            batch_adv.append(adv)
            batch_oldlp.append(old_lp)

        policy.train()
        last_loss, clip_frac, logp_tok = 0.0, 0.0, 0.0
        for _ in range(inner_epochs):
            for seqs, gmask, adv, old_lp in zip(batch_seqs, batch_gmask,
                                                batch_adv, batch_oldlp):
                opt.zero_grad(set_to_none=True)
                with autocast_ctx(device):
                    new_lp = token_logprobs(policy, seqs)              # (G, T-1)
                    m = gmask[:, 1:]
                    ratio = torch.exp(new_lp - old_lp)
                    a = adv.unsqueeze(1)                               # (G, 1)
                    unclipped = ratio * a
                    # clip-higher (DAPO): the UPPER bound is wider, leaving room
                    # to raise the probability of rare-but-correct tokens.
                    clipped = torch.clamp(ratio, 1 - clip_eps_low,
                                          1 + clip_eps_high) * a
                    surrogate = torch.min(unclipped, clipped)
                    with torch.no_grad():
                        ref_lp = token_logprobs(ref, seqs)
                    logr = ref_lp - new_lp
                    kl = torch.exp(logr) - logr - 1.0
                    per_tok = -(surrogate - kl_beta * kl)
                    # token-level normalization over GENERATED tokens (DAPO)
                    loss = (per_tok * m).sum() / m.sum().clamp(min=1)
                    with torch.no_grad():                  # health metrics
                        denom = m.sum().clamp(min=1).item()
                        clip_frac = (((ratio > 1 + clip_eps_high) |
                                      (ratio < 1 - clip_eps_low)).float()
                                     * m).sum().item() / denom
                        logp_tok = (new_lp * m).sum().item() / denom
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
                last_loss = loss.item()
        acc = n_correct / max(1, n_total)
        accs.append(acc)
        losses.append(last_loss)
        degen.append(n_degenerate)
        if log_every and it % log_every == 0:
            print(f"  [grpo] it{it} train_acc {acc:.3f} loss {last_loss:.4f} "
                  f"degen {n_degenerate}/{prompts_per_iter} "
                  f"clip {clip_frac:.3f} logp/tok {logp_tok:.3f}")
    return policy, {"acc_history": accs, "loss_history": losses,
                    "degenerate_history": degen}
