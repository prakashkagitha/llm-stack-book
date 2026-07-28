"""Direct Preference Optimization (Rafailov et al., 2023) -- feasible at 100M/$.
A frozen deep-copied reference model supplies the KL anchor; the policy is nudged
to prefer chosen over rejected responses.
"""
import copy

import torch
import torch.nn.functional as F

from ..optim import build_optimizer
from ..train.loop import autocast_ctx


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
        acc = (logits > 0).float().mean()
        margin = beta * (chosen_logratio - rejected_logratio).mean()
    return loss, {"acc": acc.item(), "margin": margin.item()}


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
                          f"acc {stats['acc']:.2f} margin {stats['margin']:.3f}")
    return policy, {"loss_history": history}
