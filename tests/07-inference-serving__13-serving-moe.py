"""
Runs the CPU-runnable Python blocks from content/07-inference-serving/13-serving-moe.md.

Blocks tested (verbatim from the chapter, in order):
  - block #0 (line ~50):  moe_ep_layer()      — reference expert-parallel forward + smoke test
  - block #1 (line ~142): moe_decode_step()   — DeepEP-style low-latency decode pattern (pseudo-API
                            in the book; here it is called against tiny toy stand-ins for
                            router/experts/ep_buffer so the function's OWN logic actually executes)
  - block #2 (line ~190): imbalance_factor()  — routing-skew metric + uniform vs. skewed smoke test

Blocks intentionally SKIPPED (per the assignment's default heuristics):
  - block #3 (line ~293, ExpertCache): SKIP(needs-gpu) — default device="cuda", and the whole
    point of the class is HBM<->CPU offload with pinned/non_blocking transfers; there is nothing
    honest to demonstrate about it on CPU-only (no .to("cuda") is available in CI).
  - block #4 (line ~365, ep_decode_step_pipeline): SKIP(fragment) — references undefined
    `layers`, `batch.hidden`, `batch.kv_cache`, `sched`, `layer_norm_and_lm_head`, etc.; it is an
    illustrative end-to-end pipeline sketch, not a standalone runnable unit.
"""

import numpy as np
import torch
import torch.nn.functional as F

# ===========================================================================
# Block #0 (line ~50): moe_ep_layer — reference EP forward, verbatim from the book
# ===========================================================================


def moe_ep_layer(x, router_w, expert_w1, expert_w2, k, G):
    """
    Reference EP forward for ONE MoE layer, simulating G ranks in-process.
      x          : (T, d)         tokens for the whole EP group (gathered for clarity)
      router_w   : (E, d)         router weights
      expert_w1  : (E, d, h)      per-expert up-projection
      expert_w2  : (E, h, d)      per-expert down-projection
      k          : top-k          experts per token
      G          : number of EP ranks; expert e lives on rank (e // (E//G))
    Returns y : (T, d). The point is to expose dispatch/compute/combine, not speed.
    """
    T, d = x.shape
    E = router_w.shape[0]
    experts_per_rank = E // G

    # ---- 1. ROUTER (runs on each token's home rank) ----
    logits = x @ router_w.t()                      # (T, E)
    gate = F.softmax(logits, dim=-1)               # softmax over all experts
    topv, topi = gate.topk(k, dim=-1)              # (T, k) weights and expert ids
    topv = topv / topv.sum(-1, keepdim=True)       # renormalize the chosen-k weights

    y = torch.zeros_like(x)

    # ---- 2-4. For each rank, gather its tokens (DISPATCH), compute, scatter back (COMBINE)
    for r in range(G):
        lo, hi = r * experts_per_rank, (r + 1) * experts_per_rank
        # Which (token, slot) pairs route to an expert owned by rank r?
        mask = (topi >= lo) & (topi < hi)          # (T, k) boolean
        if not mask.any():
            continue
        tok_idx, slot_idx = mask.nonzero(as_tuple=True)   # variable length -> all-to-all-v
        local_eid = topi[tok_idx, slot_idx] - lo          # expert index within this rank
        gw = topv[tok_idx, slot_idx]                      # gate weight for the combine

        # Grouped expert compute: loop experts on this rank (real kernels fuse this GEMM)
        for le in range(experts_per_rank):
            sel = (local_eid == le)
            if not sel.any():
                continue
            rows = tok_idx[sel]
            xin = x[rows]                                  # tokens sent to this expert
            hmid = F.gelu(xin @ expert_w1[lo + le])        # up-proj + act
            out = hmid @ expert_w2[lo + le]                # down-proj
            # COMBINE: weighted scatter-add back to the home token
            y.index_add_(0, rows, out * gw[sel].unsqueeze(-1))
    return y


# --- tiny smoke test ---
torch.manual_seed(0)
T, d, h, E, k, G = 12, 16, 32, 8, 2, 4
x = torch.randn(T, d)
rw = torch.randn(E, d) * 0.1
w1 = torch.randn(E, d, h) * (d ** -0.5)
w2 = torch.randn(E, h, d) * (h ** -0.5)
y = moe_ep_layer(x, rw, w1, w2, k, G)
print(y.shape)   # torch.Size([12, 16])

assert y.shape == (T, d)
assert torch.isfinite(y).all()
print("[block 0] moe_ep_layer OK")


# ===========================================================================
# Block #1 (line ~142): moe_decode_step — DeepEP-style low-latency decode sketch,
# verbatim from the book. It is written against a pseudo-API (router/experts/ep_buffer);
# below we supply tiny toy stand-ins that implement that exact interface on CPU with
# plain tensors (no real all-to-all, no network), so the block's own control flow —
# dispatch -> overlap window -> hook -> grouped_gemm -> combine — genuinely executes.
# ===========================================================================


def moe_decode_step(hidden, router, experts, ep_buffer):
    topk_idx, topk_w = router(hidden)                  # local routing

    # Kick off dispatch; returns immediately with a 'hook' you call later.
    recv_x, recv_layout, hook = ep_buffer.low_latency_dispatch(
        hidden, topk_idx, num_experts=experts.E, use_fp8=True
    )
    # ---- OVERLAP WINDOW: do work that doesn't depend on recv_x ----
    #   e.g., prefetch next layer's expert weights, compute shared-expert FFN,
    #         run the next request stream's attention, etc.
    shared_out = experts.shared_expert(hidden)         # dense path, no all-to-all
    # --------------------------------------------------------------
    hook()                                             # block only now, on arrival

    expert_out = experts.grouped_gemm(recv_x, recv_layout)   # local expert FFNs (FP8 in)
    # Combine: send results home and weighted-sum by gate weights.
    y = ep_buffer.low_latency_combine(expert_out, topk_idx, topk_w)
    return y + shared_out


# --- minimal honest glue: toy router/experts/ep_buffer implementing the pseudo-API on CPU ---
class _ToyRouter:
    """Local top-k router, same math as block #0's router step."""

    def __init__(self, d, E, k):
        self.w = torch.randn(E, d) * 0.1
        self.k = k

    def __call__(self, hidden):
        logits = hidden @ self.w.t()
        gate = F.softmax(logits, dim=-1)
        topv, topi = gate.topk(self.k, dim=-1)
        topv = topv / topv.sum(-1, keepdim=True)
        return topi, topv


class _ToyExperts:
    """Grouped-GEMM experts + a shared (dense, no all-to-all) expert."""

    def __init__(self, d, h, E):
        self.E = E
        self.w1 = torch.randn(E, d, h) * (d ** -0.5)
        self.w2 = torch.randn(E, h, d) * (h ** -0.5)
        self.shared_w = torch.randn(d, d) * (d ** -0.5)

    def shared_expert(self, hidden):
        return hidden @ self.shared_w

    def grouped_gemm(self, recv_x, recv_layout):
        out = torch.zeros_like(recv_x)
        for eid in recv_layout.unique().tolist():
            sel = (recv_layout == eid)
            xin = recv_x[sel]
            hmid = F.gelu(xin @ self.w1[eid])
            out[sel] = hmid @ self.w2[eid]
        return out


class _ToyEPBuffer:
    """
    Single-process stand-in for deep_ep.Buffer's low_latency_dispatch/_combine.
    No real network/all-to-all: it just reshapes (T, k, d) -> (T*k, d) and back,
    which is exactly what the real dispatch/combine achieve across ranks.
    """

    def low_latency_dispatch(self, hidden, topk_idx, num_experts, use_fp8=False):
        T, kk = topk_idx.shape
        d = hidden.shape[-1]
        recv_x = hidden.unsqueeze(1).expand(T, kk, d).reshape(T * kk, d).clone()
        recv_layout = topk_idx.reshape(-1)          # expert id owning each row
        state = {"arrived": False}

        def hook():
            state["arrived"] = True                # in the real kernel: blocks on RDMA completion

        return recv_x, recv_layout, hook

    def low_latency_combine(self, expert_out, topk_idx, topk_w):
        T, kk = topk_idx.shape
        d = expert_out.shape[-1]
        expert_out = expert_out.reshape(T, kk, d)
        return (expert_out * topk_w.unsqueeze(-1)).sum(dim=1)


torch.manual_seed(1)
T2, d2, h2, E2, k2 = 10, 16, 24, 8, 2
hidden = torch.randn(T2, d2)
router = _ToyRouter(d2, E2, k2)
experts = _ToyExperts(d2, h2, E2)
ep_buffer = _ToyEPBuffer()

decode_out = moe_decode_step(hidden, router, experts, ep_buffer)
assert decode_out.shape == (T2, d2)
assert torch.isfinite(decode_out).all()
print("[block 1] moe_decode_step OK, out shape:", tuple(decode_out.shape))


# ===========================================================================
# Block #2 (line ~190): imbalance_factor — routing-skew metric, verbatim from the book
# ===========================================================================


def imbalance_factor(token_expert_ids, E, G):
    """token_expert_ids: (T, k) selected expert ids for a decode batch."""
    counts = np.bincount(token_expert_ids.reshape(-1), minlength=E)
    experts_per_rank = E // G
    rank_load = counts.reshape(G, experts_per_rank).sum(axis=1)
    return rank_load.max() / rank_load.mean()


# Uniform routing vs. a skewed batch (one rank's experts are 4x popular)
rng = np.random.default_rng(0)
E, G, T, k = 256, 32, 2048, 8
uniform = rng.integers(0, E, size=(T, k))
uniform_if = imbalance_factor(uniform, E, G)
print("uniform IF:", round(uniform_if, 3))    # ~1.05

p = np.ones(E); p[:8] *= 4; p /= p.sum()                            # experts 0..7 hot
skewed = rng.choice(E, size=(T, k), p=p)
skewed_if = imbalance_factor(skewed, E, G)
print("skewed  IF:", round(skewed_if, 3))     # > 1.5

assert 1.0 <= uniform_if < 1.3, f"expected near-uniform IF, got {uniform_if}"
assert skewed_if > 1.5, f"expected a clearly skewed IF, got {skewed_if}"
print("[block 2] imbalance_factor OK")

print("\nAll CPU-runnable blocks executed successfully.")
