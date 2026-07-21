# 2.9 Mixture-of-Experts (MoE) Architectures

There is a brutal fact at the center of the scaling story: in a dense Transformer, **every parameter does work on every token**. If you want a model that "knows more" — more facts, more languages, more code idioms — you grow the feed-forward networks (FFNs), and the cost of *each forward pass* grows in lockstep. A 70-billion-parameter dense model spends roughly 70 billion parameters' worth of floating-point operations (FLOPs) on the word "the" and on a subtle line of CUDA, equally. That is wasteful, and it is the wall that Mixture-of-Experts (MoE) is built to break.

The MoE idea is almost embarrassingly simple to state: **decouple the number of parameters from the FLOPs per token.** Replace the single FFN in a Transformer block with a *bank* of FFNs called experts, and add a small learned **router** that, for each token, picks just a few of them. A token is processed by, say, 2 experts out of 256. The model now *contains* the knowledge of 256 FFNs but *pays* for only 2 of them per token. You get the memory footprint and representational capacity of a giant model with the compute bill of a small one. This is **conditional computation** — the network decides, per input, which sub-networks to activate — and it is the architectural choice behind GShard, Switch Transformer, Mixtral, DeepSeek-MoE, Qwen-MoE, and most of the frontier "trillion-parameter" models that nonetheless run at the speed of a much smaller dense model.

The catch — and this chapter is largely about the catch — is that *picking* is hard. The router is a discrete, non-differentiable decision sitting in the middle of a network we train with smooth gradients. Tokens must be routed in balanced batches so that GPUs stay busy and no expert is starved or overwhelmed. Experts must be spread across devices, which turns every MoE layer into a communication problem (the [expert parallelism](../03-pretraining/06-distributed-model-parallel.html) we preview at the end). Get the load balancing wrong and the model collapses to using one expert, throwing away the whole point. We will build the mechanism from the FFN up, write a correct toy MoE layer you can run, work the numbers on capacity and parameter counts, and tour the landmark designs that got each piece right.

This chapter assumes you know the [Transformer block](../02-transformer/06-transformer-block.html) — especially that the FFN is two linear layers with a nonlinearity — and the softmax from [Probability, Statistics & Information Theory](../01-foundations/02-probability-information.html). It pairs closely with [Scaling Laws](../03-pretraining/04-scaling-laws.html) (why sparsity changes the compute-optimal frontier) and with [Distributed Training II](../03-pretraining/06-distributed-model-parallel.html) (how experts are sharded in practice).

## From One FFN to a Bank of Experts

Recall the Transformer block: attention mixes information *across* tokens, then a position-wise FFN transforms *each* token independently. For a token representation $x \in \mathbb{R}^{d_\text{model}}$ the standard FFN is

$$
\operatorname{FFN}(x) = W_2\,\phi\!\left(W_1 x + b_1\right) + b_2,
\qquad W_1 \in \mathbb{R}^{d_\text{ff} \times d_\text{model}},\; W_2 \in \mathbb{R}^{d_\text{model} \times d_\text{ff}},
$$

with $\phi$ a nonlinearity (GELU, or a gated SwiGLU as in modern models — see [The Transformer Block](../02-transformer/06-transformer-block.html)). Typically $d_\text{ff} = 4\,d_\text{model}$, so the FFN holds about $8\,d_\text{model}^2$ parameters and is usually **two-thirds of the parameters in the whole model**. The FFN is also where most of the "knowledge" is thought to live: keys-and-values memory, factual associations, n-gram-like circuits. So if you want to add capacity cheaply, the FFN is exactly where you want to do it.

A **MoE layer** replaces that one FFN with $E$ experts, each an independent FFN with its own weights $\{\operatorname{FFN}_e\}_{e=1}^{E}$, plus a **router** (also called the **gate**) $g: \mathbb{R}^{d_\text{model}} \to \mathbb{R}^{E}$ that scores how relevant each expert is to the token. The layer output is a weighted combination of the experts the router selects:

$$
\operatorname{MoE}(x) = \sum_{e \in \mathcal{T}(x)} g_e(x)\, \operatorname{FFN}_e(x),
$$

where $\mathcal{T}(x) \subseteq \{1,\dots,E\}$ is the (small) set of selected experts for token $x$ and $g_e(x)$ is its gate weight. In a **dense MoE** (the original 1991 Jacobs–Jordan formulation, and a "soft" baseline today) $\mathcal{T}(x) = \{1,\dots,E\}$: you run *all* experts and blend them. That is differentiable and clean but defeats the purpose — you pay for all $E$ FFNs. The breakthrough that made MoE useful at LLM scale is **sparsity**: make $|\mathcal{T}(x)| = k$ with $k \ll E$ (typically $k=1$ or $k=2$), so each token touches only a handful of experts.

{{fig:moe-dense-vs-sparse-block}}

Two vocabulary items you will see constantly. **Total parameters** counts every expert's weights — this is the model's "size on disk" and its capacity. **Active parameters** counts only what a single token actually uses — this drives the FLOPs and therefore latency. Mixtral 8×7B, for instance, has on the order of 47B total parameters but activates roughly 13B per token (it is *not* 8×7=56B, because attention and embeddings are shared and only the FFNs are replicated). That ratio — total-to-active — is the sparsity dial, and it is the headline of every MoE model card. The whole bet of MoE is that **a model's quality tracks total parameters while its cost tracks active parameters**, so you want that ratio large.

{{fig:moe-total-vs-active-resource-split}}

!!! note "Aside: 'expert' is a misnomer worth unlearning"
    The word *expert* suggests each FFN specializes in a human-legible domain — one for French, one for Python, one for poetry. Reality is messier. Learned experts specialize along axes that are mostly *not* interpretable: token frequency, surface n-gram patterns, position, sometimes syntactic role. You will occasionally find an expert that fires on numbers or on whitespace, but do not expect a "biology expert." Think of routing as a *learned, load-balanced hash* of tokens to sub-networks, not a panel of domain specialists. This reframing predicts the real failure modes (imbalance, collapse) far better than the marketing name does.

## The Router: Gating, Top-k, and the Softmax-Then-Select Question

The router is a single linear layer — astonishingly small relative to what it controls. It maps each token to one logit per expert:

$$
h(x) = W_g\, x \in \mathbb{R}^{E}, \qquad W_g \in \mathbb{R}^{E \times d_\text{model}}.
$$

For $E=256$ experts and $d_\text{model}=4096$, that is about a million parameters governing the routing of a model with hundreds of billions. The router's job is to turn the logit vector $h(x)$ into (a) a *choice* of which $k$ experts to run and (b) a set of *weights* with which to combine their outputs. There are two design knobs, and the order in which you pull them matters.

{{fig:moe-routing}}

**Top-k selection.** Take the indices of the $k$ largest logits: $\mathcal{T}(x) = \operatorname{top\text{-}k}\big(h(x)\big)$. This is the discrete, non-differentiable heart of the layer. Crucially, gradients flow only through the experts that were *selected* — an expert that loses the top-k comparison gets no gradient signal this step. (This is also why the gate weights, not the indices, must carry gradient: the selected experts' weights $g_e$ scale their outputs, so $\partial \mathcal{L}/\partial W_g$ flows through them. The `top-k` operator itself is treated as a stop-gradient on the index decision; the network learns routing only through the magnitudes of the kept weights and the load-balancing loss we add later.)

**Softmax: before or after top-k?** Two conventions exist and they are *not* equivalent:

- **Softmax-then-top-k** (GShard, Switch): compute $p = \operatorname{softmax}(h(x))$ over *all* $E$ experts, then keep the $k$ largest probabilities as the gate weights. The kept weights are a slice of a full distribution; they do **not** sum to 1.
- **Top-k-then-softmax** (Mixtral): select the top-$k$ logits first, then softmax *only over those $k$*. The gate weights now sum to 1, which keeps the output magnitude stable regardless of how confident the router was.

Mixtral's choice — renormalizing over the chosen experts — is the more common modern default because it decouples "how much total signal flows" from "how spread the router's confidence was." With softmax-then-top-k, a router that is uncertain (flat distribution) produces small gate weights and a weak FFN contribution; renormalizing fixes that. Here is the gate, both ways, so you can see the difference is a single line:

```python
import torch
import torch.nn.functional as F

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

logits = torch.tensor([[2.0, 1.0, 0.1, -1.0, 0.5, 3.0, -2.0, 0.2]])  # 1 token, 8 experts
w1, i1 = route_softmax_then_topk(logits, k=2)
w2, i2 = route_topk_then_softmax(logits, k=2)
print("softmax-then-topk:", i1.tolist(), w1.round(decimals=3).tolist())  # weights < 1, no sum-to-1
print("topk-then-softmax:", i2.tolist(), w2.round(decimals=3).tolist())  # weights sum to 1
```

**Noisy / top-k gating.** The original sparse-MoE paper (Shazeer et al., 2017) added tunable Gaussian noise to the logits *before* top-k:

$$
h(x) = W_g x + \varepsilon \odot \operatorname{softplus}(W_\text{noise}\, x),\qquad \varepsilon \sim \mathcal{N}(0, I).
$$

The noise does two jobs: it **breaks ties** so different tokens explore different experts early in training, and it acts as a regularizer that spreads load. Most modern models drop the learned noise (they rely on the auxiliary balancing loss instead) but the lesson — that routing needs *exploration* to avoid premature collapse onto a few experts — recurs throughout the field. DeepSeek-V3 reintroduced a different exploration mechanism: a per-expert **bias** added to the *routing scores used for selection only* (not for the gate weights), nudged up or down each step to equalize load without polluting the gradient — an "auxiliary-loss-free" balancing trick we return to below.

{{fig:moe-router-dispatch-pipeline}}

## A Toy MoE Layer From Scratch

Let us build a correct, runnable sparse MoE FFN layer. The pedagogical subtlety in *every* MoE implementation is the same: top-k routing is **ragged** — different tokens go to different experts, so you cannot just run a clean batched matmul over the whole sequence. There are two ways to handle this, and you should know both.

1. **The dense/masked loop** (simple, what we write first): loop over experts; for each expert, find the tokens routed to it, run that expert on them, scatter the results back. Clear and correct; not the fastest because of the gather/scatter, but it is exactly what grouped-GEMM kernels (Megablocks, the [Triton](../04-kernels-efficiency/04-triton-kernels.html) grouped matmul) optimize under the hood.
2. **The all-experts-then-mask trick** (only for tiny $E$): run *every* expert on *every* token and zero out the ones you did not select. Trivial to write, but it throws away the FLOPs savings — useful only for unit tests and teaching.

We implement the loop version, which is what scales:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

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

# --- smoke test: it runs and shapes are right ---
torch.manual_seed(0)
moe = SparseMoE(d_model=32, d_ff=64, n_experts=8, k=2, capacity_factor=1.25)
x = torch.randn(4, 16, 32)              # batch 4, seq 16
y, aux = moe(x)
print(y.shape, float(aux))             # torch.Size([4, 16, 32]) <some positive scalar>
```

A few details in that code are load-bearing and worth dwelling on. The `index_add_` is what makes top-$k>1$ correct: a token selected by two experts contributes *both* weighted outputs to the same output row, and because the gate weights were renormalized to sum to 1, the result is a true convex combination. The capacity clamp `token_ids[:capacity]` is the **token dropping** mechanism — tokens beyond an expert's quota silently get *no* FFN contribution this layer (their output stays whatever the residual stream carried in, since we add to a zero buffer that then joins the residual). In a real implementation the dropped token still flows through the residual connection, so it is degraded, not destroyed. And the auxiliary loss combines a *hard* count `f` with a *soft* probability `P` on purpose — we explain exactly why next.

!!! warning "Common pitfall: forgetting that token dropping makes the layer batch-dependent"
    Because capacity is computed from the current batch and excess tokens are dropped, an MoE layer's output for a given token can change depending on *which other tokens are in the batch* — a property dense layers never have. At training time this is fine (it is a form of noise). At **inference** it is a correctness hazard: two identical prompts batched differently can produce different logits. Production serving either uses a generous capacity factor, drops the capacity limit entirely (dynamic-size grouped GEMM), or pads/recomputes so results are batch-invariant. Always know which regime your serving stack is in.

## Load Balancing: The Make-or-Break Problem

Left to its own devices, a sparse router **collapses**. Early in training, one expert is — by random initialization — slightly better. The router sends it more tokens, so it gets more gradient, so it improves faster, so it attracts even more tokens. This is a textbook rich-get-richer feedback loop, and its fixed point is a model that routes (almost) everything to one expert while the rest atrophy, never receiving gradient. You have then paid for $E$ experts and trained one. Worse, in a distributed setting, the favored expert's GPU is saturated while the others idle — the slowest device sets the pace, so imbalance directly *wastes wall-clock time*. Load balancing is therefore not a nicety; it is the difference between MoE working and MoE failing.

{{fig:moe-router-collapse-and-aux-loss}}

### The auxiliary load-balancing loss

The dominant fix, from GShard and refined by Switch, is to add a differentiable penalty that pushes routing toward uniform. Define, over a batch of $N$ tokens and $E$ experts:

- $f_e$ = the **fraction of tokens dispatched** to expert $e$ (a hard count, $\sum_e f_e = 1$ for top-1; for top-$k$ it sums to $k$ before normalization).
- $P_e = \frac{1}{N}\sum_{i} p_e(x_i)$ = the **mean router probability** assigned to expert $e$ (soft, from the full softmax).

The Switch auxiliary loss is

$$
\mathcal{L}_\text{aux} = \alpha \cdot E \cdot \sum_{e=1}^{E} f_e\, P_e,
$$

with a small coefficient $\alpha$ (on the order of $10^{-2}$). Two questions: why the *product* $f_e P_e$, and why mix a hard count with a soft probability?

The product is minimized, subject to $\sum_e P_e = 1$, exactly when the load is **uniform** ($P_e = 1/E$ for all $e$). You can see this with Lagrange multipliers, or intuitively: $\sum_e f_e P_e$ is like an expectation that is smallest when probability mass avoids the heavily-loaded experts. The factor $E$ rescales it so the loss is $\approx 1$ at perfect balance, making $\alpha$ interpretable across different $E$.

The hard/soft mix is the clever part. We *want* to penalize the hard dispatch counts $f_e$ — those are what actually cause imbalance — but $f_e$ comes from a `top-k`/`argmax` and is **non-differentiable**. So we keep $f_e$ as a constant (it provides the *direction*: which experts are overloaded) and multiply by the differentiable $P_e$ (which provides the *gradient*: nudging the router's probabilities down on overloaded experts and up on starved ones). The loss "reads" imbalance from the hard counts and "writes" the correction through the soft probabilities. It is a beautiful little gradient-routing trick and it is why MoE trains at all.

```python
def switch_aux_loss(router_logits, expert_idx, n_experts, alpha=1e-2):
    """router_logits: (N, E) pre-softmax. expert_idx: (N,) top-1 choice per token."""
    P = F.softmax(router_logits, dim=-1).mean(dim=0)        # (E,) soft mass, differentiable
    # f: hard dispatch fraction (a count; we detach so only P carries gradient).
    f = torch.bincount(expert_idx, minlength=n_experts).float()
    f = (f / f.sum()).detach()                              # (E,), constant direction
    return alpha * n_experts * torch.sum(f * P)
```

A second, smaller term often accompanies it: the **router z-loss** (from ST-MoE), $\mathcal{L}_z = \beta \cdot \frac{1}{N}\sum_i \big(\log\sum_e e^{h_e(x_i)}\big)^2$, which penalizes large router logits. This keeps the router from saturating into a near-one-hot regime where small numerical perturbations flip the top-k choice, dramatically improving training stability in bf16/fp8 (see [Mixed Precision](../03-pretraining/08-mixed-precision-fp8.html)).

### Expert-choice routing: invert the problem

There is an elegant alternative that makes imbalance *impossible by construction*. In standard **token-choice** routing, each token picks its top-$k$ experts — and nothing stops every token from picking the same expert. **Expert-choice** routing (Zhou et al., 2022) flips it: each *expert* picks its top-$C$ tokens, where $C$ is exactly its capacity. Now every expert processes precisely $C$ tokens — perfect balance, no auxiliary loss, no dropping in the usual sense. The trade-off: a token might be chosen by *many* experts or by *none*, so some tokens get more compute than others (the model decides which tokens are "hard"), and the scheme is awkward for autoregressive *inference* because it needs to see the whole batch of tokens to do the selection, which breaks causal, one-token-at-a-time decoding. Expert-choice is therefore most natural in encoder-style or training-time settings; token-choice with an aux loss remains the default for decoder LLMs.

{{fig:moe-token-vs-expert-choice}}

## Capacity Factor, Token Dropping, and the Worked Numbers

The **capacity** of an expert is the maximum number of tokens it will accept in a batch. With $N$ tokens, top-$k$ routing, and $E$ experts, the *average* tokens per expert is $Nk/E$. We multiply by a **capacity factor** $C_f \ge 1$ to leave slack for imbalance:

$$
\text{capacity} = \left\lceil C_f \cdot \frac{N\,k}{E} \right\rceil.
$$

Why does capacity exist at all? Because GPUs want **static, rectangular tensors**. To run expert $e$ as one efficient matmul, you need a fixed-size buffer of tokens for it. Capacity is that buffer size. If more than `capacity` tokens route to an expert, the overflow is **dropped** (skips the FFN, passes through on the residual). If fewer arrive, the buffer is **padded** with zeros (wasted FLOPs). So $C_f$ trades two evils: too low and you drop many tokens (hurting quality); too high and you waste compute and memory on padding. Typical training values are $C_f \in [1.0, 2.0]$; Switch used around 1.0–1.25, GShard often 2.0. Modern systems increasingly use **dropless** MoE (Megablocks) with grouped/block-sparse GEMMs that handle ragged sizes directly, eliminating both dropping and padding — but the capacity concept is essential for understanding the classics and most serving paths.

!!! example "Worked example: capacity, dropping, and the parameter/FLOP split"
    Take a configuration close to a small Mixtral-style layer: $E = 8$ experts, top-$k = 2$, $d_\text{model} = 4096$, $d_\text{ff} = 14336$, processing a batch of $N = 8192$ tokens (e.g. 16 sequences of length 512). Use a capacity factor $C_f = 1.25$.

    **Average load:** $Nk/E = 8192 \times 2 / 8 = 2048$ tokens per expert on average.

    **Capacity per expert:** $\lceil 1.25 \times 2048 \rceil = 2560$ tokens. Each expert's dispatch buffer holds 2560 token-slots; the layer reserves $8 \times 2560 = 20480$ slots for $8192 \times 2 = 16384$ routed token-instances — about 25% headroom for imbalance.

    **What if one expert is hot?** Suppose the router (mid-training, imperfectly balanced) sends 3000 tokens to expert 3. Capacity is 2560, so $3000 - 2560 = 440$ tokens are **dropped** at this layer — they get no FFN update and pass through on the residual. Across 32 MoE layers, a token has many chances to be dropped somewhere; this is why a too-tight capacity factor visibly hurts loss.

    **Parameter count.** Each expert FFN (SwiGLU has 3 matrices; use 2 here for simplicity) holds $2 \times d_\text{model} \times d_\text{ff} = 2 \times 4096 \times 14336 \approx 1.17 \times 10^{8}$ params. Eight experts: $\approx 9.4 \times 10^{8}$ params in this one MoE layer. A dense FFN would be just $1.17 \times 10^{8}$. So the MoE layer holds **8× the parameters** of a dense FFN.

    **FLOP count (the payoff).** Per token, only $k = 2$ experts run. The MoE FFN does $\approx 2 \times (2 \times d_\text{model} \times d_\text{ff}) \times k$ FLOPs $= 2 \times 1.17\times10^8 \times 2 \approx 9.4 \times 10^{8}$ FLOPs per token — i.e. **2 experts' worth**, not 8. The model *contains* 8 experts but each token *pays* for 2. Total-to-active ratio $= 8/2 = 4\times$. That 4× is the lever: roughly 4× the parameters (capacity) at 1× the per-token FLOPs of a 2-expert dense model.

The takeaway in one line: **capacity factor governs the dropping-vs-padding trade-off; the total-to-active ratio governs the capacity-vs-cost trade-off.** Both are knobs you tune, and both show up directly in training loss and serving latency.

## The Landmark Designs: GShard, Switch, Mixtral, DeepSeek-MoE

The ideas above were not discovered all at once; each landmark model fixed a specific pain point. Knowing the lineage lets you reason about *why* a given model made its choices.

### GShard (2020): MoE goes to scale

GShard (Lepikhin et al., Google) put MoE into a 600B-parameter multilingual translation Transformer and established the recipe that everyone still uses: **top-2 routing**, an **auxiliary load-balancing loss**, a **capacity factor with token dropping**, and — critically — **expert parallelism** via an all-to-all communication primitive that shuffles tokens to wherever their experts live. GShard is where MoE stopped being a curiosity and became a scaling tool, and where the systems problem (the all-to-all, previewed below) was first confronted head-on.

### Switch Transformer (2021): top-1 is enough

Switch (Fedus, Zoph, Shazeer, Google) made a deliberately aggressive simplification: route each token to exactly **one** expert ($k = 1$). The conventional wisdom had been that you needed top-2 so the router could *compare* two experts and get a gradient signal for the choice; Switch showed top-1 trains fine with the right tricks. The payoff is roughly halved communication and compute per token. Switch also contributed the cleaner **auxiliary loss** formulation above, careful **bf16 stability** work (selectively casting the router to fp32), and the **router z-loss**. It scaled to 1.6T parameters. If you implement MoE once from scratch, implement Switch; it is the minimal complete design.

### Mixtral 8×7B (2023): MoE for open decoder LLMs

Mixtral (Mistral AI) was the model that made MoE mainstream for the open-source decoder-only LLM. Eight experts per layer, **top-2** routing, **top-k-then-softmax** gating (renormalized weights), with attention and embeddings **shared** across experts so only the FFNs are replicated. The naming "8×7B" is a friendly fiction: it is *not* 56B parameters but $\approx 47$B total (because the non-FFN parts are not multiplied), activating $\approx 13$B per token. Mixtral demonstrated that an MoE could match or beat a much larger dense model (Llama-2 70B) at a fraction of the active compute, on standard chat/reasoning benchmarks — the existence proof that sold the architecture to practitioners.

### DeepSeek-MoE and the fine-grained + shared-expert idea (2024)

DeepSeek-MoE introduced two refinements that the frontier has largely adopted, and they are worth understanding precisely because they attack a real limitation of the GShard/Switch design: **coarse experts force redundancy**. If you only have 8 big experts and you want to combine, say, "code knowledge" with "math knowledge" on a given token, your only options are coarse. Two changes fix this:

1. **Fine-grained experts.** Instead of $E$ big experts, use $mE$ experts each $1/m$ the size, and increase $k$ proportionally. The *active* parameter count is unchanged, but the number of *combinations* of experts a token can express explodes combinatorially ($\binom{mE}{mk}$ grows fast). More, smaller experts give the router a richer, more composable palette and empirically improve specialization. DeepSeek-V3 takes this far: **256 routed experts**, top-8 per token.

2. **Shared experts.** Designate a small number of experts (often 1) that **every** token always uses, in addition to its routed experts. The intuition: common, universal knowledge ("how to form a plural," "basic syntax") should not have to be redundantly relearned inside many specialized experts. The shared expert absorbs the common case; the routed experts then specialize on the residual. This both improves quality and reduces the redundancy that pure routed-MoE suffers.

DeepSeek-V3 further replaced the auxiliary loss with the **auxiliary-loss-free** bias-adjustment scheme mentioned earlier: a per-expert bias $b_e$ is added to the routing logits *for selection only*, and after each step $b_e$ is decremented for overloaded experts and incremented for underloaded ones. This achieves balance without the aux loss's gradient interference, which can slightly degrade quality by fighting the language-modeling objective. The architecture also pairs MoE with **Multi-head Latent Attention (MLA)** for KV-cache compression — see [Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html).

{{fig:moe-dense-gshard-deepseek-lineage}}

A small comparison table to anchor the lineage:

| Model | Experts $E$ | Top-$k$ | Gating | Balancing | Notable idea |
|---|---|---|---|---|---|
| Shazeer et al. 2017 | up to 1000s | k (e.g. 4) | softmax-then-topk + noise | aux + importance loss | first sparse LSTM MoE |
| GShard 2020 | per-layer, sharded | 2 | softmax-then-topk | aux loss, capacity drop | expert parallel all-to-all |
| Switch 2021 | up to thousands | 1 | softmax-then-topk | aux + z-loss | top-1 simplicity, bf16 stability |
| Mixtral 2023 | 8 | 2 | topk-then-softmax | aux loss | open decoder LLM, shared attn |
| DeepSeek-V3 2024 | 256 routed (+shared) | 8 | sigmoid/softmax + bias | aux-loss-free bias | fine-grained + shared experts |

## Expert Parallelism: A Systems Preview

So far we have treated all experts as living on one device. At frontier scale they cannot — 256 experts of 100M+ parameters each will not fit on one GPU, and even if they did, you would want their compute spread out. **Expert parallelism (EP)** shards the experts across devices: GPU 0 holds experts 0–31, GPU 1 holds 32–63, and so on. This creates the defining systems challenge of MoE, and it is worth previewing here even though [Distributed Training II](../03-pretraining/06-distributed-model-parallel.html) covers it in depth.

The problem: a token on GPU 0 may route to an expert on GPU 5. Before the MoE layer can run, **every token must be sent to the device holding its chosen expert**, and after the experts run, **every result must be sent back** to the token's home device for the residual add and the next layer. These two shuffles are **all-to-all** collectives (see [Parallel Computing & Collective Communication](../01-foundations/09-parallel-collectives.html)) — the most communication-intensive pattern there is, because in the worst case every device talks to every other device.

{{fig:moe-expert-parallel-all-to-all}}

This is exactly why **load balancing maps directly to throughput**. If expert 3 (on GPU 5) is overloaded, GPU 5 does more work than the others; because the all-to-all is a barrier, *every* GPU waits for the slowest. Imbalance does not just hurt quality — it idles your cluster. The all-to-all is also bandwidth-hungry, so production MoE training overlaps it with computation, places experts to minimize cross-node traffic, and tunes the capacity factor to bound the message sizes. DeepSeek-V3's auxiliary-loss-free balancing and its node-limited routing (restricting each token to experts on a bounded number of nodes) are, at bottom, *communication* optimizations dressed as routing rules. The architecture and the systems are inseparable in MoE; you cannot reason about one without the other.

For inference, EP interacts with the [serving stack](../07-inference-serving/01-anatomy-inference.html): MoE models have *low* active-parameter FLOPs (great for decode) but *high* total-parameter memory (you must keep all experts resident in HBM, since any token might need any of them) and *high* all-to-all traffic. This makes MoE serving a memory-bandwidth and communication problem more than a compute one — the opposite balance from dense models — and motivates specialized kernels and expert-placement strategies in vLLM and SGLang.

!!! interview "Interview Corner"
    **Q:** A Mixtral-style model is "8×7B" but reported as ~47B total parameters activating ~13B per token. Reconcile those three numbers, and explain why the total-to-active ratio, not the raw parameter count, is what you should reason about for both quality and serving cost.

    **A:** The "8×7B" name suggests $8 \times 7 = 56$B, but only the **FFNs** are replicated 8×; **attention, embeddings, norms, and the output head are shared** across experts. So total $\approx$ (shared params) + 8 × (one expert's FFN params) $\approx$ 47B, not 56B. **Active** parameters count what one token uses: the shared params + the FFNs of just its top-$k=2$ experts $\approx$ 13B. For **quality**, model capacity (and thus knowledge) tracks *total* parameters — 47B-class. For **serving cost**: *latency/FLOPs* track *active* parameters — 13B-class, so it decodes about as fast as a 13B dense model. But *memory* tracks *total* — you must hold all 47B in HBM because any token can route to any expert, so it has the memory footprint of a 47B model. The total-to-active ratio ($\approx 3.6\times$ here) is the sparsity dial: it tells you you get ~47B of capacity at ~13B of compute, while paying ~47B of memory and extra all-to-all communication. Reasoning with the ratio, not a single number, is what keeps you from mispredicting either latency or VRAM.

!!! warning "Common pitfall: comparing MoE and dense models by total parameters"
    A 47B MoE and a 47B dense model are *not* comparable: the MoE does the FLOPs of a ~13B model per token, so it trains and serves far cheaper but is generally *less capable per total-parameter* than an equally-sized dense model (each parameter is exercised less, on a fraction of tokens). The fair comparisons are MoE-vs-dense at equal *active* params (MoE wins on quality, because it has more total capacity) or at equal *training/serving FLOPs* (MoE usually wins). Quoting only total parameters is a marketing apples-to-oranges; always ask for the active count and the training token budget.

## Training, Stability, and When to Reach for MoE

A few practitioner realities round out the picture. MoE models are **less FLOP-efficient to *train* per step but reach a given loss in fewer FLOPs** — the [scaling-laws](../03-pretraining/04-scaling-laws.html) story is that, at a fixed compute budget, sparse models sit on a better loss-vs-FLOPs frontier than dense ones, which is the whole economic argument for them. But they come with sharp edges:

- **Training instability.** Routers are prone to spiky losses; the router z-loss, fp32 router computation, and careful initialization (small router weights so early routing is near-uniform) are standard mitigations. See [Training Stability & Loss Spikes](../03-pretraining/11-training-stability.html).
- **Fine-tuning is finicky.** Sparse models can overfit downstream tasks faster and route inconsistently on small datasets; freezing the router or using a higher capacity factor during fine-tuning helps. Distilling a sparse model into a dense one ([Distillation & Compression](../05-posttraining-alignment/12-distillation-compression.html)) is a common deployment path when you want MoE's training efficiency but dense serving simplicity.
- **Memory dominates serving.** As above, all experts must be resident, so MoE shifts the bottleneck from compute to HBM capacity and bandwidth. Expert offloading (CPU↔GPU paging of cold experts) and quantizing experts more aggressively than attention are active areas.
- **When to use it.** Reach for MoE when you are *compute-bound* in training or *latency-bound* in serving but have *memory to spare* — exactly the frontier-pretraining and high-throughput-serving regimes. If memory is your scarce resource (edge, single-GPU), a dense model of the same *total* size is simpler and serves with less machinery.

!!! tip "Practitioner tip: watch the routing entropy, not just the loss"
    The single most useful MoE training diagnostic is the **per-expert token distribution** (and its entropy) over time. Healthy training shows it rising toward uniform early and then *gently* specializing — not collapsing to a spike. A sudden drop in routing entropy, or one expert's load climbing past ~2–3× the mean, is an early warning of collapse that precedes any loss spike. Log it every few steps; it will save you a wasted multi-day run.

### Instrumenting the tip: an `ExpertLoadMonitor`

The tip above is only useful if you actually instrument it. Below is a drop-in `ExpertLoadMonitor` that mirrors the `TrainingMonitor` pattern from [Training Stability & Loss Spikes](../03-pretraining/11-training-stability.html) — same `log_every` throttling, same `log_step` returning a metrics dict — but computes the three signals that predict MoE collapse before the loss does: normalized router entropy, the per-expert dispatch-fraction histogram, and a starvation alert.

```python
import math
import torch

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


# --- Demo: a healthy layer vs. a collapsed layer ---
E, k, N = 8, 2, 4096
monitor = ExpertLoadMonitor(n_experts=E, k=k, log_every=1, starve_frac=0.2, patience=3)

torch.manual_seed(0)
healthy_logits = torch.randn(N, E)          # uniform-ish routing

torch.manual_seed(1)
collapsed_logits = torch.randn(N, E)
collapsed_logits[:, 0:4] += 4.0             # experts 0-3 dominate
collapsed_logits[:, 4:8] -= 4.0             # experts 4-7 starve

for _ in range(3):  # 3 windows == patience -> alert fires on the last call
    out = monitor.log_step({
        "layer0.healthy": healthy_logits,
        "layer1.collapsed": collapsed_logits,
    })

print(out)
```

Expected output: `layer0.healthy` reports `router_entropy ~= 0.83`, `max_load_ratio ~= 1.0`, `n_starved = 0`; `layer1.collapsed` reports `router_entropy ~= 0.54`, experts 4-7 at `min_load_ratio = 0.0`, `n_starved = 4`, and on the third call four `[MoE ALERT]` lines fire for `layer1.collapsed` (none for `layer0.healthy`). The starvation rule in words: an expert holding fewer than `starve_frac/E = 0.2/8 = 0.025` of dispatched tokens — 20% of its fair share of `1/E = 0.125` — counts as starving; `patience` consecutive logging windows below that threshold trigger the alert.

!!! key "Key Takeaways"
    - **MoE decouples parameters from FLOPs.** Replace the FFN with $E$ experts and a router; each token uses only $k \ll E$ of them, so the model has the *capacity* of a giant model at the *compute* of a small one. The total-to-active ratio is the sparsity dial.
    - **The router is a tiny linear gate** producing per-expert logits; you select **top-$k$** (the non-differentiable step) and combine outputs with softmax weights — either softmax-then-topk (Switch/GShard) or topk-then-softmax with renormalized weights (Mixtral).
    - **Load balancing is make-or-break.** Without it, routing collapses to one expert (rich-get-richer). The **auxiliary loss** $\alpha E \sum_e f_e P_e$ mixes a *hard* dispatch count (direction) with a *soft* probability (gradient); the **z-loss** stabilizes router logits; **expert-choice** routing balances by construction but breaks causal decode.
    - **Capacity factor and token dropping** exist because GPUs want rectangular tensors: capacity $= \lceil C_f \cdot Nk/E\rceil$ bounds each expert's buffer; overflow is dropped (quality cost), underflow is padded (compute cost). Dropless grouped-GEMM kernels (Megablocks) avoid both.
    - **The lineage:** Shazeer 2017 (sparse gating) → GShard (top-2, expert parallelism) → Switch (top-1 simplicity, z-loss) → Mixtral (open decoder LLM, renormalized gating) → DeepSeek-MoE (**fine-grained + shared experts**, auxiliary-loss-free bias balancing).
    - **Expert parallelism makes MoE a communication problem:** tokens are shuffled to their experts' devices via **all-to-all** and back. Load imbalance directly idles GPUs because the all-to-all is a barrier — architecture and systems are inseparable.
    - **MoE serving is memory- and bandwidth-bound,** not compute-bound: low active FLOPs (fast decode) but all experts must stay resident in HBM (high VRAM) with heavy all-to-all traffic.
    - **Compare MoE to dense by *active* params or by *FLOPs*, never by total params** — total-parameter comparisons are marketing apples-to-oranges.

!!! sota "State of the Art & Resources (2026)"
    Sparse Mixture-of-Experts is now the dominant architecture for frontier-scale LLMs: virtually every model above ~100B parameters (Gemini, GPT-4, DeepSeek-V3, Grok) uses some form of MoE, with fine-grained expert counts in the hundreds and auxiliary-loss-free load balancing becoming the new standard. The open-source ecosystem has matured around dropless grouped-GEMM kernels and integrated serving stacks.

    **Foundational work**

    - [Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer* (2017)](https://arxiv.org/abs/1701.06538) — introduced noisy top-k gating and the modern sparse MoE recipe for language modeling.
    - [Lepikhin et al., *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding* (2020)](https://arxiv.org/abs/2006.16668) — first MoE at 600B+ parameters; established top-2 routing, capacity factor, and expert-parallel all-to-all.
    - [Fedus, Zoph, Shazeer, *Switch Transformers: Scaling to Trillion Parameter Models* (2021)](https://arxiv.org/abs/2101.03961) — showed top-1 routing suffices; introduced the clean auxiliary loss formulation and bf16 stability tricks.

    **Recent advances (2023–2026)**

    - [Zoph et al., *ST-MoE: Designing Stable and Transferable Sparse Expert Models* (2022)](https://arxiv.org/abs/2202.08906) — introduced the router z-loss that stabilizes training; thorough fine-tuning study at 269B parameters.
    - [Zhou et al., *Mixture-of-Experts with Expert Choice Routing* (2022)](https://arxiv.org/abs/2202.09368) — inverts routing so each expert picks its top-C tokens, guaranteeing perfect load balance by construction.
    - [Jiang et al. (Mistral AI), *Mixtral of Experts* (2024)](https://arxiv.org/abs/2401.04088) — open decoder-only MoE with top-k-then-softmax renormalized gating; existence proof for MoE in the open-source stack.
    - [Dai et al. (DeepSeek), *DeepSeekMoE: Towards Ultimate Expert Specialization* (2024)](https://arxiv.org/abs/2401.06066) — fine-grained experts plus always-on shared experts; richer routing combinations at identical active-parameter cost.
    - [DeepSeek-AI, *DeepSeek-V3 Technical Report* (2024)](https://arxiv.org/abs/2412.19437) — 671B total / 37B active parameters; auxiliary-loss-free bias-adjustment balancing and node-limited routing as communication optimizations.

    **Open-source & tools**

    - [databricks/megablocks](https://github.com/databricks/megablocks) — dropless MoE training via block-sparse grouped-GEMM kernels; eliminates token dropping and padding, up to 40% faster than prior frameworks.
    - [deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) — open inference and training code for the DeepSeek-V3 MoE architecture including the auxiliary-loss-free balancing implementation.

    **Go deeper**

    - [Hugging Face, *Mixture of Experts Explained* (2023)](https://huggingface.co/blog/moe) — comprehensive practitioner guide covering routing, capacity, load balancing, serving tradeoffs, and the MegaBlocks/FasterMoE ecosystem.

## Further reading

- Shazeer, Mirhoseini, Maziarz, Davis, Le, Hinton, Dean — *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer* (2017). The paper that introduced noisy top-k gating and modern sparse MoE.
- Lepikhin, Lee, Xu, Chen, Firat, Huang, Krikun, Shazeer, Chen — *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding* (2020). Top-2 routing, capacity, and expert-parallel all-to-all at scale.
- Fedus, Zoph, Shazeer — *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity* (2021). Top-1 routing, the auxiliary loss formulation, and bf16 stability tricks.
- Zoph, Bello, Kumar, et al. — *ST-MoE: Designing Stable and Transferable Sparse Expert Models* (2022). The router z-loss and a careful study of stability and fine-tuning.
- Zhou, Lei, Liu, et al. — *Mixture-of-Experts with Expert Choice Routing* (2022). The expert-choice scheme that balances load by construction.
- Jiang, Sablayrolles, Roux, et al. (Mistral AI) — *Mixtral of Experts* (2024). The open decoder-only MoE with top-2 renormalized gating.
- Dai, Deng, Zhao, et al. (DeepSeek) — *DeepSeekMoE: Towards Ultimate Expert Specialization* (2024), and the *DeepSeek-V3 Technical Report* (2024). Fine-grained and shared experts; auxiliary-loss-free balancing.
- Gale, Narayanan, Zaharia, et al. — *MegaBlocks: Efficient Sparse Training with Mixture-of-Experts* (2022). Block-sparse / grouped-GEMM kernels that make MoE dropless.
- Jacobs, Jordan, Nowlan, Hinton — *Adaptive Mixtures of Local Experts* (1991). The original mixture-of-experts formulation that started it all.
