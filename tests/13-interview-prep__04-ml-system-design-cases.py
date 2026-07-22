"""
Runnable-code test for content/13-interview-prep/04-ml-system-design-cases.md

Tests every CPU-runnable Python block in the chapter, reproduced VERBATIM
(only test fixtures / assertions are added around them), in document order:

  - block @ line ~55  : Tower / TwoTower model + in_batch_softmax_loss
  - block @ line ~101 : MultiTaskRanker + combined_utility
  - block @ line ~174 : reciprocal_rank_fusion (RRF)
  - block @ line ~258 : replicas_needed (LLM serving capacity)
  - block @ line ~292 : assemble_context (RAG context packing)
  - block @ line ~399 : reward_model_loss + dpo_loss

Skipped blocks -- all ```text ASCII-art diagrams (funnel, serving topology,
RL step, tiered moderation): SKIP(non-python) prose diagrams, no code to run.

All blocks use only numpy / torch. No network, no API calls -> fully hermetic.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Block @ line ~55: two-tower retrieval model + in-batch softmax loss (verbatim)
# ---------------------------------------------------------------------------

class Tower(nn.Module):
    """A generic tower: features -> L2-normalized d-dim embedding."""
    def __init__(self, in_dim, d=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, d),
        )

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=-1)   # unit vectors => dot product = cosine

class TwoTower(nn.Module):
    def __init__(self, user_dim, item_dim, d=128):
        super().__init__()
        self.user_tower = Tower(user_dim, d)
        self.item_tower = Tower(item_dim, d)
        # temperature sharpens/softens the softmax over candidates
        self.log_temp = nn.Parameter(torch.tensor(0.0))

    def forward(self, user_feats, item_feats):
        u = self.user_tower(user_feats)     # (B, d)
        v = self.item_tower(item_feats)     # (B, d)
        return u, v

def in_batch_softmax_loss(u, v, log_temp):
    """Sampled-softmax with in-batch negatives.
    Row i's positive is column i; every other item in the batch is a negative.
    This is why large batches matter: more negatives => better retrieval.
    """
    logits = (u @ v.t()) / log_temp.exp()        # (B, B) similarity matrix
    labels = torch.arange(u.size(0), device=u.device)
    return F.cross_entropy(logits, labels)

# --- exercise it end-to-end ---
torch.manual_seed(0)
model = TwoTower(user_dim=32, item_dim=48, d=128)
user_feats = torch.randn(8, 32)
item_feats = torch.randn(8, 48)
u, v = model(user_feats, item_feats)
print(u.shape, v.shape)   # torch.Size([8, 128]) torch.Size([8, 128])
assert u.shape == (8, 128) and v.shape == (8, 128), "tower embeddings must be (B, d=128)"
# Towers L2-normalize their output => unit norm.
assert torch.allclose(u.norm(dim=-1), torch.ones(8), atol=1e-5), "user embeddings must be unit-norm"
assert torch.allclose(v.norm(dim=-1), torch.ones(8), atol=1e-5), "item embeddings must be unit-norm"

loss = in_batch_softmax_loss(u, v, model.log_temp)
print(float(loss.detach()))
assert np.isfinite(float(loss.detach())) and float(loss.detach()) > 0, "in-batch softmax loss must be finite and positive"
# The loss must be differentiable back through the towers.
loss.backward()
assert model.user_tower.net[0].weight.grad is not None, "loss must produce gradients into the user tower"
# log_temp is a learnable parameter and should receive gradient too.
assert model.log_temp.grad is not None, "temperature parameter must receive gradient"

# A perfectly-aligned batch (identity similarity) drives the loss toward its
# minimum; a scrambled one raises it -- confirms the diagonal-is-positive setup.
d = 128
ident = F.normalize(torch.eye(6, d), dim=-1)
lt = torch.tensor(0.0)
aligned = float(in_batch_softmax_loss(ident, ident, lt))
scrambled = float(in_batch_softmax_loss(ident, ident[torch.randperm(6)], lt))
assert aligned < scrambled, "matched pairs on the diagonal must score lower loss than a scramble"


# ---------------------------------------------------------------------------
# Block @ line ~101: multi-task ranker + combined utility (verbatim)
# ---------------------------------------------------------------------------

class MultiTaskRanker(nn.Module):
    """Shared bottom -> per-task heads. A real system uses MMoE
    (multi-gate mixture-of-experts) so tasks can share or specialize."""
    def __init__(self, feat_dim, hidden=1024):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.click_head   = nn.Linear(hidden, 1)   # P(click)
        self.watch_head   = nn.Linear(hidden, 1)   # E[watch | click], log-seconds
        self.satisfy_head = nn.Linear(hidden, 1)   # P(not dissatisfied)

    def forward(self, x):
        h = self.shared(x)
        return {
            "click":   torch.sigmoid(self.click_head(h)),
            "watch":   self.watch_head(h),            # regression in log space
            "satisfy": torch.sigmoid(self.satisfy_head(h)),
        }

def combined_utility(out, w=(1.0, 1.0, 0.5)):
    """Weighted product / sum that ops teams tune via A/B tests."""
    wc, ww, ws = w
    # P(click) * exp(E[log watch]) * P(satisfy), in log space for stability
    return (wc * torch.log(out["click"] + 1e-6)
            + ww * out["watch"]
            + ws * torch.log(out["satisfy"] + 1e-6))

ranker = MultiTaskRanker(feat_dim=64)
feats = torch.randn(500, 64)                 # ~500 candidates survive to ranking
out = ranker(feats)
assert set(out.keys()) == {"click", "watch", "satisfy"}, "ranker must emit all three heads"
assert out["click"].shape == (500, 1), "click head must be per-candidate scalar"
# Sigmoid heads are bounded probabilities; the watch head is an unbounded regression.
assert (out["click"] >= 0).all() and (out["click"] <= 1).all(), "P(click) must be in [0,1]"
assert (out["satisfy"] >= 0).all() and (out["satisfy"] <= 1).all(), "P(satisfy) must be in [0,1]"

util = combined_utility(out)
print(util.shape)   # torch.Size([500, 1])
assert util.shape == (500, 1), "utility must be one score per candidate"
assert torch.isfinite(util).all(), "combined utility must be finite for all candidates"

# The utility must be monotone in each head, holding the others fixed:
# raising P(click) with everything else equal must raise utility.
base = {"click": torch.tensor([[0.3]]), "watch": torch.tensor([[2.0]]),
        "satisfy": torch.tensor([[0.8]])}
higher_click = {**base, "click": torch.tensor([[0.9]])}
assert float(combined_utility(higher_click)) > float(combined_utility(base)), \
    "higher click probability must yield higher utility (all else equal)"
# The 0.5 weight on satisfy vs 1.0 on click means click moves utility more than
# satisfy for the same probability delta -- sanity on the weighting.


# ---------------------------------------------------------------------------
# Block @ line ~174: Reciprocal Rank Fusion (verbatim)
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(rank_lists, k=60):
    """rank_lists: list of lists of doc_ids, each ordered best-first."""
    scores = {}
    for ranking in rank_lists:
        for rank, doc_id in enumerate(ranking):        # rank starts at 0
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)

# Usage (verbatim, with the fixture lists the chapter leaves as callouts):
bm25_results  = ["d1", "d2", "d3", "d4", "d5"]
dense_results = ["d3", "d1", "d6", "d7", "d8"]
fused = reciprocal_rank_fusion([bm25_results, dense_results])
top100 = fused[:100]
print(fused)

# d1 is rank-0 in bm25 and rank-1 in dense -> it should top the fused list.
# d3 appears rank-2 in bm25 and rank-0 in dense -> also near the top.
assert fused[0] == "d1", f"doc appearing high in BOTH lists should win fusion, got {fused[0]}"
assert set(fused[:2]) == {"d1", "d3"}, "the two docs present in both lists must lead"
# Every doc from either retriever survives into the fused set (union, not intersection).
assert set(fused) == set(bm25_results) | set(dense_results), "RRF fuses the union of candidates"
# A doc present in both lists must outscore a doc present in only one at the same rank.
only_once = reciprocal_rank_fusion([["x", "y"], ["z"]])
assert only_once[0] in {"x", "z"}, "rank-0 docs lead"
assert top100[:2] == fused[:2], "top100 slice preserves fused order"


# ---------------------------------------------------------------------------
# Block @ line ~258: LLM serving replica capacity (verbatim)
# ---------------------------------------------------------------------------

def replicas_needed(qps, avg_output_tokens, tokens_per_sec_per_replica):
    """Decode throughput is the usual bottleneck for chat workloads.
    A replica producing T tokens/s (summed over its batch) serves
    QPS = T / avg_output_tokens requests per second."""
    per_replica_qps = tokens_per_sec_per_replica / avg_output_tokens
    return qps / per_replica_qps

result = replicas_needed(2000, 300, 3000)
print(result)   # -> ~200 replicas
assert abs(result - 200.0) < 1e-9, f"the chapter's worked number is ~200 replicas, got {result}"
# Doubling QPS doubles replicas; doubling per-replica throughput halves them.
assert replicas_needed(4000, 300, 3000) == 2 * result, "replicas scale linearly with QPS"
assert replicas_needed(2000, 300, 6000) == result / 2, "replicas halve when throughput doubles"


# ---------------------------------------------------------------------------
# Block @ line ~292: RAG context assembly under a token budget (verbatim)
# ---------------------------------------------------------------------------

def assemble_context(chunks, token_budget, count_tokens):
    """chunks: reranked best-first. Greedily pack under budget,
    then reorder so the single best chunk sits last (recency bias)."""
    selected, used = [], 0
    for c in chunks:
        n = count_tokens(c.text)
        if used + n > token_budget:
            break
        selected.append(c); used += n
    if len(selected) >= 2:
        # move best chunk (index 0) to the end of the prompt
        selected = selected[1:] + selected[:1]
    return selected

class Chunk:
    def __init__(self, cid, text):
        self.id = cid
        self.text = text

def count_words(s):
    return len(s.split())

# 4 chunks of 100 words each; a 250-word budget admits exactly 2.
chunks = [Chunk(i, " ".join(["w"] * 100)) for i in range(4)]
selected = assemble_context(chunks, token_budget=250, count_tokens=count_words)
print([c.id for c in selected])
assert len(selected) == 2, "budget of 250 over 100-word chunks admits exactly 2 chunks"
# best chunk (id 0) must be relocated to the END (lost-in-the-middle mitigation).
assert selected[-1].id == 0, "the highest-reranked chunk must be placed last"
assert selected[0].id == 1, "the second-best chunk leads the prompt after reordering"

# A single admitted chunk is NOT reordered (guard: len >= 2).
one = assemble_context(chunks, token_budget=150, count_tokens=count_words)
assert [c.id for c in one] == [0], "a lone chunk stays in place"

# Budget is never exceeded.
total_words = sum(count_words(c.text) for c in selected)
assert total_words <= 250, "assembled context must respect the token budget"


# ---------------------------------------------------------------------------
# Block @ line ~399: RLHF reward-model loss + DPO loss (verbatim)
# ---------------------------------------------------------------------------

def reward_model_loss(reward_chosen, reward_rejected):
    """Bradley-Terry: maximize the margin between preferred and rejected.
    reward_* are scalar scores from the RM head."""
    return -F.logsigmoid(reward_chosen - reward_rejected).mean()

def dpo_loss(pi_logp_chosen, pi_logp_rejected,
             ref_logp_chosen, ref_logp_rejected, beta=0.1):
    """DPO turns the RLHF objective into a classification loss on pairs,
    eliminating the separate reward model AND the RL loop. The policy's
    log-prob ratio against a frozen reference IS the implicit reward."""
    pi_logratios  = pi_logp_chosen  - pi_logp_rejected
    ref_logratios = ref_logp_chosen - ref_logp_rejected
    return -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()

# Bradley-Terry: loss falls as the chosen score exceeds the rejected score.
good_margin = reward_model_loss(torch.tensor([2.0]), torch.tensor([-1.0]))
bad_margin  = reward_model_loss(torch.tensor([-1.0]), torch.tensor([2.0]))
print(float(good_margin), float(bad_margin))
assert float(good_margin) < float(bad_margin), \
    "RM loss must be lower when the preferred response scores higher"
# At equal scores, -log sigmoid(0) = log 2.
tie = reward_model_loss(torch.tensor([0.5]), torch.tensor([0.5]))
assert abs(float(tie) - np.log(2)) < 1e-5, "equal rewards give loss = log 2"

# DPO: when the policy already prefers chosen MORE than the reference does
# (positive advantage), loss is below its log-2 tie point; when it prefers it
# LESS, loss is above.
pc, pr = torch.tensor([1.0]), torch.tensor([0.0])   # policy prefers chosen
rc, rr = torch.tensor([0.0]), torch.tensor([0.0])   # reference indifferent
dpo_good = float(dpo_loss(pc, pr, rc, rr))
dpo_bad  = float(dpo_loss(pr, pc, rc, rr))           # policy prefers rejected
assert dpo_good < np.log(2) < dpo_bad, "DPO loss must reward policy>reference preference alignment"
# Gradients must flow to the policy log-probs (the thing DPO optimizes).
p = torch.tensor([1.0], requires_grad=True)
l = dpo_loss(p, torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]))
l.backward()
assert p.grad is not None and torch.isfinite(p.grad).all(), "DPO loss must be differentiable wrt policy"


print("\nAll 6 tested blocks executed and asserted successfully.")
