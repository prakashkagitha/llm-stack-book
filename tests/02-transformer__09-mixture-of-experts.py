"""
Runnable-code test for content/02-transformer/09-mixture-of-experts.md

Blocks tested (assembled in chapter order, later blocks may reuse names from
earlier ones exactly as in the chapter):
  - block #0 (line ~60):  softmax-then-topk vs topk-then-softmax gating
                           (route_softmax_then_topk, route_topk_then_softmax)
  - block #1 (line ~104): Toy sparse MoE layer (Expert, SparseMoE) with the
                           book's own smoke test (moe(x) -> y, aux_loss)
  - block #3 (line ~339): ExpertLoadMonitor + the healthy-vs-collapsed demo

Skipped:
  - block #2 (line ~220): fragment — `switch_aux_loss` is a standalone helper
    function with no call site of its own in the chapter (the worked
    auxiliary-loss computation that IS exercised lives inside SparseMoE.forward
    in block #1, which already tests the identical f/P hard-soft-mix logic).
    Per the task spec, non-called fragments default to SKIP. Since it only
    needs names already imported in block #1 (torch, F), it would run fine if
    called, but we do not fabricate a call site the book itself doesn't show.

No network / external-API calls are used anywhere in this chapter's code, so
no mocking is required. No bugs were found in the book's code; all three
tested blocks run verbatim.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# =====================================================================
# Block #0 (line ~60) — softmax-then-topk vs topk-then-softmax gating
# =====================================================================

def route_softmax_then_topk(logits, k):
    # GShard/Switch style: softmax over ALL experts, then keep the top-k slice.
    # The kept weights do NOT sum to 1 (they are a slice of a full distribution).
    probs = F.softmax(logits, dim=-1)                  # (tokens, E)
    weights, idx = torch.topk(probs, k, dim=-1)        # (tokens, k)
    return weights, idx

def route_topk_then_softmax(logits, k):
    # Mixtral style: pick top-k logits, then softmax over JUST those k.
    # The kept weights DO sum to 1 -> stable output magnitude.
    topk_logits, idx = torch.topk(logits, k, dim=-1)   # (tokens, k)
    weights = F.softmax(topk_logits, dim=-1)           # (tokens, k), rows sum to 1
    return weights, idx


def test_block_0():
    logits = torch.tensor([[2.0, 1.0, 0.1, -1.0, 0.5, 3.0, -2.0, 0.2]])  # 1 token, 8 experts
    w1, i1 = route_softmax_then_topk(logits, k=2)
    w2, i2 = route_topk_then_softmax(logits, k=2)
    print("softmax-then-topk:", i1.tolist(), w1.round(decimals=3).tolist())  # weights < 1, no sum-to-1
    print("topk-then-softmax:", i2.tolist(), w2.round(decimals=3).tolist())  # weights sum to 1

    # The highest two logits are index 5 (3.0) and index 0 (2.0) -> both methods
    # must select the same experts, since top-k on logits == top-k on softmax(logits)
    # (softmax is monotonic).
    assert set(i1[0].tolist()) == {5, 0}
    assert set(i2[0].tolist()) == {5, 0}

    # topk-then-softmax weights renormalize to sum to 1 per the chapter's claim.
    assert torch.allclose(w2.sum(dim=-1), torch.ones(1), atol=1e-5)
    # softmax-then-topk weights are a slice of a full distribution -> do NOT sum to 1.
    assert not torch.allclose(w1.sum(dim=-1), torch.ones(1), atol=1e-5)
    assert w1.sum(dim=-1).item() < 1.0


# =====================================================================
# Block #1 (line ~104) — Toy sparse MoE layer from scratch
# =====================================================================

class Expert(nn.Module):
    """A single expert: a standard 2-layer FFN (SwiGLU-free for clarity)."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):                      # x: (n_tokens_for_this_expert, d_model)
        return self.w2(F.gelu(self.w1(x)))

class SparseMoE(nn.Module):
    """Top-k sparse MoE layer with capacity-based token dropping and an
    auxiliary load-balancing loss (Switch Transformer style)."""
    def __init__(self, d_model, d_ff, n_experts, k=2, capacity_factor=1.25):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.k = k
        self.capacity_factor = capacity_factor
        self.router = nn.Linear(d_model, n_experts, bias=False)   # the gate W_g
        self.experts = nn.ModuleList(
            [Expert(d_model, d_ff) for _ in range(n_experts)]
        )

    def forward(self, x):
        # x: (batch, seq, d_model). Flatten to a flat token list -- routing is per-token.
        B, T, D = x.shape
        x = x.reshape(-1, D)                       # (N, D), N = B*T
        N = x.shape[0]

        # 1) ROUTE: logits -> top-k experts -> renormalized gate weights (Mixtral style).
        logits = self.router(x)                    # (N, E)
        topk_logits, topk_idx = torch.topk(logits, self.k, dim=-1)   # (N, k)
        gate = F.softmax(topk_logits, dim=-1)      # (N, k), each row sums to 1

        # 2) CAPACITY: max tokens any single expert will accept this batch.
        #    capacity = capacity_factor * (tokens * k / E), rounded up.
        capacity = int(self.capacity_factor * (N * self.k) / self.n_experts)
        capacity = max(capacity, 1)

        out = torch.zeros_like(x)                  # accumulate expert outputs here
        # Counters for the auxiliary loss and for diagnostics.
        # f_e = fraction of tokens dispatched to expert e (hard count, top-1 view).
        # P_e = mean router probability mass on expert e (soft).
        router_prob = F.softmax(logits, dim=-1)    # (N, E) full distribution for aux loss
        load = torch.zeros(self.n_experts, device=x.device)

        # 3) DISPATCH: loop over experts; process the tokens routed to each.
        for e in range(self.n_experts):
            # Which (token, slot) pairs picked expert e? slot in {0..k-1}.
            sel = (topk_idx == e)                  # (N, k) bool
            token_ids, slot_ids = sel.nonzero(as_tuple=True)   # variable length
            load[e] = token_ids.numel()            # tokens wanting expert e (pre-drop)

            if token_ids.numel() == 0:
                continue
            # CAPACITY DROP: keep only the first `capacity` tokens for this expert.
            if token_ids.numel() > capacity:
                token_ids = token_ids[:capacity]
                slot_ids = slot_ids[:capacity]

            ex_in = x[token_ids]                    # (n_e, D) gather
            ex_out = self.experts[e](ex_in)         # (n_e, D) run THIS expert only
            w = gate[token_ids, slot_ids].unsqueeze(-1)        # (n_e, 1) gate weight
            out.index_add_(0, token_ids, w * ex_out)           # scatter-add back

        # 4) AUXILIARY LOAD-BALANCING LOSS (Switch): encourage uniform routing.
        f = load / load.sum().clamp(min=1)         # dispatched fraction per expert
        P = router_prob.mean(dim=0)                # mean router prob per expert
        aux_loss = self.n_experts * torch.sum(f * P)   # minimized when both are uniform

        return out.reshape(B, T, D), aux_loss


def test_block_1():
    # --- smoke test: it runs and shapes are right ---
    torch.manual_seed(0)
    moe = SparseMoE(d_model=32, d_ff=64, n_experts=8, k=2, capacity_factor=1.25)
    x = torch.randn(4, 16, 32)              # batch 4, seq 16
    y, aux = moe(x)
    print(y.shape, float(aux))             # torch.Size([4, 16, 32]) <some positive scalar>

    assert y.shape == (4, 16, 32)
    assert aux.item() > 0.0
    # Every token gets a real (non-degenerate) output row -- with capacity_factor
    # 1.25 and only 64 tokens across 8 experts (avg load 16, capacity 20), no
    # dropping should occur, so no output row should be exactly zero.
    assert not torch.any((y.reshape(-1, 32).abs().sum(dim=-1) == 0.0))


# =====================================================================
# Block #3 (line ~339) — ExpertLoadMonitor: routing-collapse diagnostics
# =====================================================================

class ExpertLoadMonitor:
    """
    Tracks per-expert dispatch load and router entropy across MoE layers.
    Wire it by having each SparseMoE layer stash its pre-softmax router
    logits (shape (N, E), N = tokens in the batch, E = n_experts) into a
    dict keyed by layer name, then call log_step(logits_by_layer) once
    per training step.
    """
    def __init__(self, n_experts: int, k: int, log_every: int = 10,
                 starve_frac: float = 0.2, patience: int = 3):
        self.n_experts = n_experts
        self.k = k
        self.log_every = log_every
        self.threshold = starve_frac / n_experts  # e.g. 0.2/8 = 0.025
        self.patience = patience
        self.step = 0
        self.starve_streak = {}  # (layer_name, expert_idx) -> consecutive windows below threshold

    @torch.no_grad()
    def log_step(self, logits_by_layer: dict) -> dict:
        self.step += 1
        if self.step % self.log_every != 0:
            return {}

        metrics = {}
        alerts = []
        for name, logits in logits_by_layer.items():
            N, E = logits.shape
            probs = torch.softmax(logits, dim=-1)                       # (N, E)
            tok_H = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1)  # (N,) nats
            norm_entropy = (tok_H.mean() / math.log(E)).item()          # in [0, 1]

            idx = torch.topk(logits, self.k, dim=-1).indices            # (N, k)
            counts = torch.bincount(idx.reshape(-1), minlength=E).float()
            total = counts.sum()
            if total == 0:
                continue  # empty batch edge case
            f = counts / total                                          # (E,), sums to 1
            fair = 1.0 / E
            starved = (f < self.threshold).nonzero(as_tuple=True)[0].tolist()

            for e in range(E):
                key = (name, e)
                if e in starved:
                    self.starve_streak[key] = self.starve_streak.get(key, 0) + 1
                    if self.starve_streak[key] >= self.patience:
                        alerts.append(
                            f"{name}: expert {e} starved for {self.starve_streak[key]} "
                            f"windows (f={f[e]:.4f} < {self.threshold:.4f})"
                        )
                else:
                    self.starve_streak[key] = 0

            metrics[f"moe/{name}/router_entropy"] = norm_entropy
            metrics[f"moe/{name}/max_load_ratio"] = (f.max() / fair).item()  # >~2-3x => imbalance
            metrics[f"moe/{name}/min_load_ratio"] = (f.min() / fair).item()  # ->0 => a dying expert
            metrics[f"moe/{name}/n_starved"] = len(starved)

        metrics["moe/alerts"] = alerts
        for a in alerts:
            print("[MoE ALERT]", a)
        return metrics


def test_block_3():
    # --- Demo: a healthy layer vs. a collapsed layer ---
    E, k, N = 8, 2, 4096
    monitor = ExpertLoadMonitor(n_experts=E, k=k, log_every=1, starve_frac=0.2, patience=3)

    torch.manual_seed(0)
    healthy_logits = torch.randn(N, E)          # uniform-ish routing

    torch.manual_seed(1)
    collapsed_logits = torch.randn(N, E)
    collapsed_logits[:, 0:4] += 4.0             # experts 0-3 dominate
    collapsed_logits[:, 4:8] -= 4.0             # experts 4-7 starve

    out = {}
    for _ in range(3):  # 3 windows == patience -> alert fires on the last call
        out = monitor.log_step({
            "layer0.healthy": healthy_logits,
            "layer1.collapsed": collapsed_logits,
        })

    print(out)

    # The healthy layer should show near-uniform load, no starved experts, and
    # therefore raise no alerts on it.
    assert out["moe/layer0.healthy/n_starved"] == 0
    assert not any("layer0.healthy" in a for a in out["moe/alerts"])

    # The collapsed layer starves exactly the 4 experts we deliberately biased
    # against (indices 4-7), and after `patience` consecutive windows the
    # monitor must fire one alert per starved expert.
    assert out["moe/layer1.collapsed/n_starved"] == 4
    collapsed_alerts = [a for a in out["moe/alerts"] if "layer1.collapsed" in a]
    assert len(collapsed_alerts) == 4
    for e in range(4, 8):
        assert any(f"expert {e} starved" in a for a in collapsed_alerts)

    # Entropy sanity: the collapsed layer's routing distribution is far less
    # uniform than the healthy layer's.
    assert out["moe/layer1.collapsed/router_entropy"] < out["moe/layer0.healthy/router_entropy"]


if __name__ == "__main__":
    test_block_0()
    test_block_1()
    test_block_3()
    print("ALL TESTS PASSED")
