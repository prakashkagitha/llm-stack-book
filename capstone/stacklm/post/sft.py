"""Supervised fine-tuning on the chat template (Ch. 14.9). Loss is masked to
assistant tokens only via `IGNORE = -100` labels; the causal shift is applied in
the training step.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..optim import build_optimizer
from ..train.loop import autocast_ctx
from .chat import render_conversation

IGNORE = -100


class PackedSFTDataset(Dataset):
    def __init__(self, conversations, tok, block=256):
        self.block = block
        ids_buf, lbl_buf, seg_buf = [], [], []
        seg = 0
        for turns in conversations:
            ids, mask = render_conversation(turns, tok, add_generation_prompt=False)
            labels = [tid if m == 1 else IGNORE for tid, m in zip(ids, mask)]
            ids_buf.extend(ids)
            lbl_buf.extend(labels)
            seg_buf.extend([seg] * len(ids))
            seg += 1
        n = max(1, (len(ids_buf) // block)) * block
        # pad to a whole number of blocks
        while len(ids_buf) < n:
            ids_buf.append(tok.pad_id)
            lbl_buf.append(IGNORE)
            seg_buf.append(seg)
        self.ids = np.array(ids_buf[:n], dtype=np.int64).reshape(-1, block)
        self.lbl = np.array(lbl_buf[:n], dtype=np.int64).reshape(-1, block)
        self.seg = np.array(seg_buf[:n], dtype=np.int64).reshape(-1, block)

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, i):
        return (torch.from_numpy(self.ids[i]),
                torch.from_numpy(self.lbl[i]),
                torch.from_numpy(self.seg[i]))


def sft_train(model, loader, *, epochs=1, lr=2e-5, grad_accum=2,
              max_grad_norm=1.0, device="cpu", log_every=5):
    device = torch.device(device)
    model.to(device).train()
    opt = build_optimizer(model, lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    step = 0
    history = []
    # Running sums for the CURRENT accumulation window. Under assistant-only
    # masking the supervised-token count varies wildly between microbatches, so
    # per-microbatch means (the usual `loss / grad_accum`) silently reweight
    # short windows up. We accumulate SUMS and divide the grads once, at the
    # accumulation boundary, by the window's true supervised-token count.
    win_loss, win_tok = 0.0, 0
    for ep in range(epochs):
        for micro, (ids, labels, seg) in enumerate(loader):
            ids, labels, seg = ids.to(device), labels.to(device), seg.to(device)
            with autocast_ctx(device):
                logits, _ = model(ids, seq_ids=seg)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()   # already IGNORE-masked
                n_tok = int((shift_labels != IGNORE).sum().item())
                if n_tok > 0:
                    loss_sum = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)).float(),
                        shift_labels.view(-1),
                        ignore_index=IGNORE,
                        reduction="sum",             # SUM, not mean
                    )
            if n_tok > 0:                            # a window with no assistant
                loss_sum.backward()                  # tokens would give 0/0 = NaN
                win_loss += loss_sum.item()
                win_tok += n_tok
            if (micro + 1) % grad_accum == 0:
                if win_tok > 0:
                    inv = 1.0 / win_tok              # exact per-token mean grad
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.mul_(inv)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    opt.step()
                    step += 1
                    history.append(win_loss / win_tok)
                    if log_every and step % log_every == 0:
                        print(f"  [sft] ep{ep} step{step} "
                              f"loss {win_loss / win_tok:.3f} ntok {win_tok}")
                opt.zero_grad(set_to_none=True)
                win_loss, win_tok = 0.0, 0
    return {"loss_history": history}
