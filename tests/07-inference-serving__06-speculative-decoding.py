"""
Executable test for content/07-inference-serving/06-speculative-decoding.md

Assembles the chapter's CPU-runnable Python blocks, in the order they appear
in the chapter, into one runnable module. Each block is copied verbatim from
the book, with minimal glue (a tiny fake "base model" fixture, and a call at
the bottom of each block) so the book's actual code executes on CPU.

Blocks tested (chapter order):
    #2 (line ~372) - MedusaHead / MedusaModel (parallel decoding heads that
                      reuse the frozen base model's LM head)
    #3 (line ~438) - build_tree_attn(): tree-attention mask + position ids
                      from a parent-pointer tree

Blocks skipped:
    #0, #1 - SKIP(needs-gpu): naive / KV-cached speculative-decoding drivers
              that load a real target+draft model pair onto .cuda() and call
              .generate(); not CPU-runnable and not standalone (they assume
              real pretrained `target`/`draft`/`tok` objects).
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Block #2 (line ~372): Medusa parallel decoding heads.
# ---------------------------------------------------------------------------

class MedusaHead(nn.Module):
    """One Medusa head: a residual MLP block reusing the base model's LM head.
    Predicts a token several positions ahead from the SAME hidden state."""
    def __init__(self, hidden_size, lm_head):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()
        self.lm_head = lm_head            # SHARED with the frozen base model
    def forward(self, h):                 # h: [B, T, hidden]
        h = h + self.act(self.linear(h))  # residual connection
        return self.lm_head(h)            # [B, T, vocab]

class MedusaModel(nn.Module):
    def __init__(self, base_model, num_heads=4):
        super().__init__()
        self.base = base_model            # frozen target
        hidden = base_model.config.hidden_size
        lm_head = base_model.get_output_embeddings()
        self.heads = nn.ModuleList(
            MedusaHead(hidden, lm_head) for _ in range(num_heads)
        )
    @torch.no_grad()
    def forward(self, input_ids):
        out = self.base(input_ids, output_hidden_states=True)
        h_last = out.hidden_states[-1]            # [B, T, hidden]
        base_logits = self.base.get_output_embeddings()(h_last)
        # head k predicts token t+k+2; base predicts t+1.
        head_logits = [head(h_last) for head in self.heads]
        return base_logits, head_logits          # use last position to draft


# --- glue: a tiny fake "base model" so MedusaModel's forward can actually run.
# It only needs to expose .config.hidden_size, .get_output_embeddings(), and
# a __call__ that accepts (input_ids, output_hidden_states=True) and returns
# an object with a .hidden_states tuple, exactly like a HF model would.

class _TinyConfig:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

class _TinyOutput:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states

class _TinyBaseModel(nn.Module):
    """Minimal stand-in for a HF causal-LM: embedding -> 1 layer -> shared LM head."""
    def __init__(self, vocab_size=32, hidden_size=16):
        super().__init__()
        self.config = _TinyConfig(hidden_size)
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layer = nn.Linear(hidden_size, hidden_size)
        self._lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_output_embeddings(self):
        return self._lm_head

    def forward(self, input_ids, output_hidden_states=True):
        h0 = self.embed(input_ids)             # [B, T, hidden]
        h1 = torch.tanh(self.layer(h0))         # one "transformer layer"
        return _TinyOutput(hidden_states=(h0, h1))


def _run_medusa_smoke_test():
    torch.manual_seed(0)
    base = _TinyBaseModel(vocab_size=32, hidden_size=16)
    base.eval()
    medusa = MedusaModel(base, num_heads=4)
    medusa.eval()

    input_ids = torch.randint(0, 32, (2, 6))   # [B=2, T=6]
    base_logits, head_logits = medusa(input_ids)

    assert base_logits.shape == (2, 6, 32), base_logits.shape
    assert len(head_logits) == 4
    for hl in head_logits:
        assert hl.shape == (2, 6, 32), hl.shape

    # Sanity: last-position logits give a valid next-token distribution per head.
    next_token_candidates = [base_logits[:, -1, :].argmax(-1)] + [
        hl[:, -1, :].argmax(-1) for hl in head_logits
    ]
    assert len(next_token_candidates) == 5
    for cand in next_token_candidates:
        assert cand.shape == (2,)
        assert (cand >= 0).all() and (cand < 32).all()

    print("Medusa smoke test OK:",
          "base logits", base_logits.shape,
          "num heads", len(head_logits))

_run_medusa_smoke_test()


# ---------------------------------------------------------------------------
# Block #3 (line ~438): tree attention mask + position ids.
# ---------------------------------------------------------------------------

def build_tree_attn(parents):
    """
    parents: list where parents[i] is the index of node i's parent
             (parents[0] = -1 for the root). Nodes are in any order such
             that a parent appears before its children.
    Returns (mask, position_ids):
      mask[i, j] = True  iff node j is an ancestor of node i, or j == i.
      position_ids[i] = depth of node i (root = 0).
    """
    n = len(parents)
    mask = torch.zeros(n, n, dtype=torch.bool)
    depth = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        # Walk from node i up to the root, marking every ancestor as visible.
        j, d = i, 0
        while j != -1:
            mask[i, j] = True
            if j != i:
                d += 1
            j = parents[j]
        depth[i] = d
    return mask, depth

# Example tree: [root, A, B, A1, A2, B1, B2] from the diagram above.
parents = [-1, 0, 0, 1, 1, 2, 2]
mask, pos = build_tree_attn(parents)
print(pos.tolist())          # [0, 1, 1, 2, 2, 2, 2]  (depths)
# Feed `mask` as the attention mask and `pos` as position_ids to the target
# in ONE forward pass; then verify each root->leaf path with accept/reject.

assert pos.tolist() == [0, 1, 1, 2, 2, 2, 2]
assert mask.shape == (7, 7)
# Root (node 0) is its own only ancestor.
assert mask[0].tolist() == [True, False, False, False, False, False, False]
# Node 3 (A1, child of A=1, child of root=0): ancestors are {0, 1, 3}.
assert mask[3].tolist() == [True, True, False, True, False, False, False]
# Node 6 (B2, child of B=2, child of root=0): ancestors are {0, 2, 6}.
assert mask[6].tolist() == [True, False, True, False, False, False, True]
# Sibling branches must not see each other: node 3 (under A) must not see node 5/6 (under B).
assert not mask[3, 5] and not mask[3, 6]
assert not mask[5, 3] and not mask[5, 4]

print("build_tree_attn smoke test OK")


if __name__ == "__main__":
    print("All speculative-decoding blocks executed successfully.")
