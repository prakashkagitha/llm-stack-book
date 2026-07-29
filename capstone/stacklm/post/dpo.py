"""Direct Preference Optimization (Rafailov et al., 2023) -- feasible at 100M/$.
A frozen deep-copied reference model supplies the KL anchor; the policy is nudged
to prefer chosen over rejected responses.
"""
import copy

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..optim import build_optimizer
from ..train.loop import autocast_ctx
from .chat import SPECIAL, Turn, render_conversation
from .grpo import exact_match_reward, sample_group


def _strip_stop(text):
    """Cut a decoded completion at its first stop marker; drop the marker."""
    for stop in (SPECIAL["end"], SPECIAL["eos"]):
        text = text.split(stop)[0]
    return text.strip()


def arithmetic_score(text, gold):
    """A verifiable score_fn for on-policy pair mining.

    Returns (primary, tiebreak):
      primary  -- a DISCRETE quality level: 1.5 exact match, 0.5 parseable but
                  wrong, 0.0 unparseable. This is the only component the
                  degeneracy filter looks at.
      tiebreak -- -len(text), used only to order samples that TIE on `primary`.

    Keeping the length term out of `primary` matters: a continuous shaping term
    almost never ties, which would silently disable any epsilon-based
    degeneracy filter and train DPO on 'shorter is better' (Ch. 14.9).
    """
    r, pred = exact_match_reward(text, gold)
    return r + (0.5 if pred is not None else 0.0), -len(text)


@torch.no_grad()
def mine_onpolicy_pairs(model, tok, prompts, score_fn, *, k=4, max_new=96,
                        temperature=1.0, device="cpu"):
    """Generate k completions per prompt from the CURRENT policy, score them,
    and keep (prompt, best, worst) whenever the PRIMARY scores actually differ.

    prompts  : iterable of (prompt_text, meta); `meta` is whatever score_fn needs.
    score_fn : (completion_text, meta) -> (primary, tiebreak), higher is better.
    """
    model.eval()
    pairs = []
    for prompt_text, meta in prompts:
        p_ids, _ = render_conversation([Turn("user", prompt_text)], tok,
                                       add_generation_prompt=True)
        p_ids = torch.tensor(p_ids, dtype=torch.long)
        seqs, _gmask, Tp = sample_group(model, tok, p_ids, k, max_new=max_new,
                                        temperature=temperature, device=device)
        texts = [_strip_stop(tok.decode(seqs[i, Tp:].tolist())) for i in range(k)]
        scores = [score_fn(t, meta) for t in texts]
        primary = [s[0] for s in scores]
        if max(primary) - min(primary) < 1e-6:
            continue                       # no verifiable signal in this prompt
        hi = max(range(k), key=lambda i: scores[i])   # lexicographic: primary,
        lo = min(range(k), key=lambda i: scores[i])   # then the length tiebreak
        if not texts[hi] or not texts[lo]:
            continue
        pairs.append({"prompt": prompt_text,
                      "chosen": texts[hi], "rejected": texts[lo]})
    return pairs


class PreferenceDataset(Dataset):
    """Render {"prompt","chosen","rejected"} dicts into two masked sequences.

    The mask is render_conversation's assistant_mask, so DPO log-probabilities
    are summed over RESPONSE tokens only. Over-long pairs are DROPPED, not
    truncated: a truncated winner teaches that good answers stop abruptly.
    """

    def __init__(self, pairs, tok, max_len=1024, system=None):
        self.items = []
        for p in pairs:
            rec = {}
            for key in ("chosen", "rejected"):
                turns = ([Turn("system", system)] if system else []) + [
                    Turn("user", p["prompt"]), Turn("assistant", p[key])]
                ids, mask = render_conversation(turns, tok,
                                                add_generation_prompt=False)
                if len(ids) > max_len:
                    rec = None
                    break
                rec[key] = (ids, mask)
            if rec:
                self.items.append(rec)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate_preferences(items, pad_id):
    """Right-pad chosen and rejected (independently) to the batch max.

    No attention mask is needed: attention is causal, so pads sitting strictly
    after every real token cannot influence any real token's representation,
    and their own logits are discarded by the loss mask.
    """
    out = {}
    for key in ("chosen", "rejected"):
        seqs = [it[key] for it in items]
        T = max(len(ids) for ids, _ in seqs)
        ids_b = torch.full((len(seqs), T), pad_id, dtype=torch.long)
        msk_b = torch.zeros((len(seqs), T), dtype=torch.float)
        for r, (ids, mask) in enumerate(seqs):
            ids_b[r, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            msk_b[r, :len(mask)] = torch.tensor(mask, dtype=torch.float)
        out[f"{key}_ids"] = ids_b
        out[f"{key}_mask"] = msk_b
    return out


def sequence_logprob(model, input_ids, loss_mask, seg=None):
    logits, _ = model(input_ids, seq_ids=seg)
    logits = logits[:, :-1, :]
    targets = input_ids[:, 1:]
    mask = loss_mask[:, 1:]
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok_logp = torch.gather(logp, 2, targets.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    return (tok_logp * mask).sum(dim=-1)                                 # (B,)


def dpo_loss(policy, ref, batch, beta=0.1, device="cpu"):
    ch_ids = batch["chosen_ids"].to(device)
    ch_m = batch["chosen_mask"].to(device)
    rj_ids = batch["rejected_ids"].to(device)
    rj_m = batch["rejected_mask"].to(device)

    pi_ch = sequence_logprob(policy, ch_ids, ch_m)
    pi_rj = sequence_logprob(policy, rj_ids, rj_m)
    with torch.no_grad():
        ref_ch = sequence_logprob(ref, ch_ids, ch_m)
        ref_rj = sequence_logprob(ref, rj_ids, rj_m)

    chosen_logratio = pi_ch - ref_ch
    rejected_logratio = pi_rj - ref_rj
    logits = beta * (chosen_logratio - rejected_logratio)
    loss = -F.logsigmoid(logits).mean()

    with torch.no_grad():
        acc = (logits > 0).float().mean()               # implicit-reward accuracy
        chosen_reward = beta * chosen_logratio.mean()   # should trend UP
        rejected_reward = beta * rejected_logratio.mean()   # should trend DOWN
        margin = chosen_reward - rejected_reward
    # Report the two rewards SEPARATELY: a falling margin-with-both-rewards-
    # falling is degradation dressed as progress (Ch. 14.9).
    return loss, {"acc": acc.item(), "chosen_r": chosen_reward.item(),
                  "rejected_r": rejected_reward.item(), "margin": margin.item()}


def dpo_train(sft_model, batches, *, epochs=1, lr=5e-7, beta=0.1, grad_accum=1,
              max_grad_norm=1.0, device="cpu", log_every=5):
    device = torch.device(device)
    policy = sft_model.to(device)
    ref = copy.deepcopy(sft_model).to(device).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = build_optimizer(policy, lr=lr, weight_decay=0.0, betas=(0.9, 0.95))

    history = []
    step = 0
    for ep in range(epochs):
        for micro, batch in enumerate(batches):
            with autocast_ctx(device):
                loss, stats = dpo_loss(policy, ref, batch, beta=beta, device=device)
                loss = loss / grad_accum
            loss.backward()
            if (micro + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                history.append(loss.item() * grad_accum)
                if log_every and step % log_every == 0:
                    print(f"  [dpo] step{step} loss {loss.item()*grad_accum:.3f} "
                          f"acc {stats['acc']:.2f} "
                          f"chosen_r {stats['chosen_r']:+.3f} "
                          f"rejected_r {stats['rejected_r']:+.3f}")
    return policy, {"loss_history": history}
