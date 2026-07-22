"""
Runnable-code test for content/13-interp-safety-gov/02-knowledge-editing-unlearning.md

Blocks tested (assembled in chapter order):
  - block #0 (line ~146): grace_layer.py — GraceAdapter, a stripped-down
    GRACE-style deferral adapter (epsilon-ball codebook lookup over
    activations). Instantiated and exercised with add_edit() + forward().
  - block #1 (line ~241): unlearn_step.py — unlearn_step() / seq_logprob(),
    a gradient-difference + NPO-style unlearning loss. Called against tiny
    fake "model"/"ref_model" stand-ins that reproduce the exact HuggingFace
    calling convention the book's code expects (model(**batch) -> object
    with .logits and, for the retain call, .loss), so the block's OWN loss
    arithmetic executes verbatim and on CPU with no network/model download.

Skipped:
  - block #2 (line ~313, minimal_rome.py): SKIP(needs-gpu / needs-network) —
    downloads GPT-2 weights from the HuggingFace hub (`GPT2LMHeadModel.
    from_pretrained(...)`), which requires network access unavailable in
    CI. The task's own block manifest also marks this "needs-gpu". Per the
    hard rules, network/model-download blocks are skipped rather than
    faked with a fabricated model, since the point of that block IS the
    real GPT-2 forward/backward passes.

No real bugs were found in the two tested blocks; both run verbatim as
written in the chapter, with only minimal glue: a tiny standalone
`FakeCausalLM` class (not from the book) that satisfies the `model(**batch)`
calling convention `unlearn_step` and `seq_logprob` rely on, built from
plain torch so no `transformers` import is needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# =====================================================================
# Block #0 (line ~146) — grace_layer.py: GraceAdapter
# =====================================================================
# grace_layer.py — a stripped-down GRACE-style deferral adapter (concept demo).

class GraceAdapter(nn.Module):
    """Wrap one hidden layer: replace its output with a stored value
    when the input activation lands inside a stored epsilon-ball."""
    def __init__(self, dim, init_eps=3.0):
        super().__init__()
        self.keys, self.vals, self.eps = [], [], []  # the editable codebook
        self.init_eps = init_eps

    def add_edit(self, key_act: torch.Tensor, target_val: torch.Tensor):
        # Store the activation we want to intercept and what to emit instead.
        self.keys.append(key_act.detach())
        self.vals.append(target_val.detach())
        self.eps.append(self.init_eps)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if not self.keys:
            return h
        K = torch.stack(self.keys)                       # [n_edits, dim]
        d = torch.cdist(h.reshape(-1, h.shape[-1]), K)   # L2 distance to each key
        nearest = d.argmin(dim=-1)                        # closest codebook entry
        eps = torch.tensor(self.eps, device=h.device)[nearest]
        inside = d.gather(-1, nearest[:, None]).squeeze(-1) < eps   # within ball?
        out = h.reshape(-1, h.shape[-1]).clone()
        V = torch.stack(self.vals)
        out[inside] = V[nearest[inside]]                 # defer: overwrite activation
        return out.reshape_as(h)


def test_block_0_grace_adapter():
    dim = 8
    adapter = GraceAdapter(dim=dim, init_eps=3.0)

    # Before any edit: passthrough, since the codebook is empty.
    h0 = torch.randn(4, dim)
    out0 = adapter(h0)
    assert torch.equal(out0, h0)

    # Add one edit: intercept an activation near `key_act`, emit `target_val`.
    key_act = torch.ones(dim) * 5.0
    target_val = torch.zeros(dim)
    adapter.add_edit(key_act, target_val)

    # A batch with one activation very close to the stored key (should be
    # deferred to target_val) and one far away (should pass through).
    near = key_act + 0.01 * torch.randn(dim)          # well inside eps=3.0
    far = torch.ones(dim) * -50.0                      # nowhere near the key
    h = torch.stack([near, far])
    out = adapter(h)

    print("near -> deferred output (should be ~0):", out[0].round(decimals=3).tolist())
    print("far  -> passthrough (unchanged):", torch.equal(out[1], far))

    assert torch.allclose(out[0], target_val, atol=1e-6)
    assert torch.equal(out[1], far)

    # A second edit exercises the multi-entry codebook / nearest-key routing.
    key_act2 = torch.ones(dim) * -5.0
    target_val2 = torch.ones(dim) * 9.0
    adapter.add_edit(key_act2, target_val2)
    h2 = torch.stack([key_act + 0.01, key_act2 - 0.01, far])
    out2 = adapter(h2)
    assert torch.allclose(out2[0], target_val, atol=1e-2)
    assert torch.allclose(out2[1], target_val2, atol=1e-2)
    assert torch.equal(out2[2], far)


# =====================================================================
# Block #1 (line ~241) — unlearn_step.py: unlearn_step / seq_logprob
# =====================================================================
# unlearn_step.py — gradient-difference + NPO-style unlearning step (sketch).

def unlearn_step(model, ref_model, forget_batch, retain_batch,
                 beta=0.1, retain_lambda=1.0, method="npo"):
    # ---- retain term: ordinary LM loss keeps general ability intact ----
    r_out = model(**retain_batch, labels=retain_batch["input_ids"])
    retain_loss = r_out.loss

    # ---- forget term ----
    f_logp = seq_logprob(model, forget_batch)          # log P_theta(forget seq)
    if method == "grad_ascent":
        # Raw ascent: push forget log-prob down. Simple but unstable.
        forget_loss = f_logp.mean()                    # minimizing this = ascending NLL
    elif method == "npo":
        # NPO: forget set as 'rejected' in a DPO-style ratio vs frozen reference.
        with torch.no_grad():
            ref_logp = seq_logprob(ref_model, forget_batch)
        ratio = beta * (f_logp - ref_logp)             # how much more likely than ref
        # -log sigmoid(-ratio): drives P_theta below the reference, self-limiting.
        forget_loss = -F.logsigmoid(-ratio).mean() * (2.0 / beta)
    loss = forget_loss + retain_lambda * retain_loss
    return loss

def seq_logprob(model, batch):
    out = model(**batch)
    logp = F.log_softmax(out.logits[:, :-1], dim=-1)
    tgt = batch["input_ids"][:, 1:]
    tok_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    mask = batch["attention_mask"][:, 1:]
    return (tok_logp * mask).sum(-1) / mask.sum(-1)    # mean log-prob per sequence


# ---------------------------------------------------------------------
# Minimal glue (not from the book): a tiny causal-LM stand-in that honors
# the exact HuggingFace-style calling convention `unlearn_step`/`seq_logprob`
# rely on: `model(**batch)` and, when `labels` is passed, `model(input_ids=,
# attention_mask=, labels=)` returning an object with `.logits` and `.loss`.
# This lets the book's own unlearning-loss logic execute verbatim on CPU
# with no `transformers` import and no network/model download.
# ---------------------------------------------------------------------

class _CausalLMOutput:
    def __init__(self, logits, loss=None):
        self.logits = logits
        self.loss = loss


class FakeCausalLM(nn.Module):
    def __init__(self, vocab_size=16, dim=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.proj = nn.Linear(dim, vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None):
        h = self.embed(input_ids)
        logits = self.proj(h)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].reshape(-1, logits.shape[-1])
            shift_labels = labels[:, 1:].reshape(-1)
            loss = F.cross_entropy(shift_logits, shift_labels)
        return _CausalLMOutput(logits=logits, loss=loss)


def test_block_1_unlearn_step():
    torch.manual_seed(0)
    vocab_size, dim = 16, 8
    model = FakeCausalLM(vocab_size, dim)
    ref_model = FakeCausalLM(vocab_size, dim)
    ref_model.load_state_dict(model.state_dict())  # ref starts identical to model
    for p in ref_model.parameters():
        p.requires_grad_(False)

    def make_batch(seq_len=6, batch=3):
        input_ids = torch.randint(0, vocab_size, (batch, seq_len))
        attention_mask = torch.ones(batch, seq_len, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    forget_batch = make_batch()
    retain_batch = make_batch()

    # seq_logprob on its own: mean log-prob per sequence, one scalar per row.
    lp = seq_logprob(model, forget_batch)
    assert lp.shape == (3,)
    assert torch.isfinite(lp).all()

    # method="npo" (the chapter's recommended, stable default)
    loss_npo = unlearn_step(model, ref_model, forget_batch, retain_batch,
                             beta=0.1, retain_lambda=1.0, method="npo")
    print("NPO unlearn loss:", loss_npo.item())
    assert loss_npo.dim() == 0
    assert torch.isfinite(loss_npo)

    # Since model == ref_model at this point, ratio == 0 for every sequence,
    # so forget_loss == -log(sigmoid(0)) * (2/beta) == log(2) * (2/beta).
    expected_forget = torch.log(torch.tensor(2.0)) * (2.0 / 0.1)
    r_out = model(**retain_batch, labels=retain_batch["input_ids"])
    expected_loss = expected_forget + r_out.loss
    assert torch.allclose(loss_npo, expected_loss, atol=1e-4), (loss_npo, expected_loss)

    # method="grad_ascent" (the naive, unstable baseline the chapter warns about)
    loss_ascent = unlearn_step(model, ref_model, forget_batch, retain_batch,
                                beta=0.1, retain_lambda=1.0, method="grad_ascent")
    print("grad-ascent unlearn loss:", loss_ascent.item())
    assert torch.isfinite(loss_ascent)

    # The loss must be differentiable wrt model params (it's used to step
    # an optimizer in the chapter's use case) -- exercise backward() too.
    loss_npo.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert all(g >= 0 for g in grad_norms)


if __name__ == "__main__":
    test_block_0_grace_adapter()
    test_block_1_unlearn_step()
    print("ALL TESTS PASSED")
