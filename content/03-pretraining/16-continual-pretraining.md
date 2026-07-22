# 3.16 Continual & Domain-Adaptive Pretraining

The most expensive thing in this book is a from-scratch pretraining run. A frontier base model represents tens of millions of dollars of compute and months of wall-clock time. So the question that dominates real-world LLM engineering is rarely "how do I train a model from scratch?" — it is "I have a good base model and a new requirement (a new domain, a new language, fresher data, a bigger model); how do I get there *without paying the full pretraining bill again*?"

That is **continual pretraining (CPT)** — also called *continued pretraining*, *continued pretraining*, or, when the new data is concentrated in one domain, **domain-adaptive pretraining (DAPT)**. You take a converged checkpoint and keep optimizing the language-modeling objective on a new data distribution. It sounds trivial — "just keep training" — and that is exactly why so many teams get it wrong. Naively resuming training on new data triggers two opposing failure modes at once: if you use a small learning rate, the model barely *learns* the new domain (the *plasticity* problem); if you use a large one, it *forgets* everything it knew (the *catastrophic forgetting* problem, also called the *stability* problem). The entire craft of CPT is navigating this stability–plasticity dilemma at a cost that is a single-digit percentage of the original run.

This chapter covers the full toolkit. We start with the learning-rate dynamics that make or break a CPT run (re-warming and re-decay), then the data-side defense against forgetting (replay), then the specific recipes for domain, language, and streaming adaptation. The second half is about a more aggressive idea: not just continuing to train the *same* weights, but **growing the model** — adding depth and width, upcycling a dense model into a Mixture-of-Experts (MoE), and transplanting an old model's knowledge into a new, larger vocabulary. We close with how to *plan* a CPT run: predicting the loss trajectory and budgeting compute before you spend it.

Read this chapter alongside [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html), which sets the compute-vs-data tradeoffs CPT exploits; [Learning Rate Schedules, Warmup, Batch Size & Hyperparameters](../03-pretraining/10-lr-schedules-hparams.html), whose schedule machinery we reuse; [Data Mixing, Domain Weighting & Curriculum](../03-pretraining/14-data-mixing-curriculum.html) for the data-mix knobs; and [Mixture-of-Experts (MoE) Architectures](../02-transformer/09-mixture-of-experts.html) for the upcycling target.

## The Stability–Plasticity Dilemma

Let $\theta_0$ be the weights of a base model converged on data distribution $\mathcal{D}_{\text{base}}$ (say, general web text). We want a model that performs well on a new distribution $\mathcal{D}_{\text{new}}$ (say, medical literature) **without** materially regressing on $\mathcal{D}_{\text{base}}$. Formally we are minimizing a two-term objective,

$$
\mathcal{L}(\theta) = \underbrace{\mathbb{E}_{x \sim \mathcal{D}_{\text{new}}}[-\log p_\theta(x)]}_{\text{plasticity: learn the new}} \;+\; \lambda\,\underbrace{\mathbb{E}_{x \sim \mathcal{D}_{\text{base}}}[-\log p_\theta(x)]}_{\text{stability: don't forget the old}},
$$

but with two complications: we usually do not have $\mathcal{D}_{\text{base}}$ exactly (the original mix is proprietary or lost), and we want the *new* term to dominate the compute. **Catastrophic forgetting** is the empirical fact that gradient descent on the first term alone drives the second term up sharply — the parameters that encoded general knowledge are overwritten by the new-domain gradient signal.

Why does forgetting happen so violently in transformers? The loss landscape near a converged minimum is *not* flat in the directions that matter for old tasks. SGD on new data moves $\theta$ along whatever directions reduce the new loss fastest; those directions are generally **not orthogonal** to the directions that the old task cares about. The Fisher information matrix $F$ of the base task quantifies this — high-curvature directions (large $F_{ii}$) are precisely the parameters whose perturbation most damages old performance. Elastic Weight Consolidation (EWC, Kirkpatrick et al., 2017) added an explicit penalty $\sum_i F_{ii}(\theta_i - \theta_{0,i})^2$, but for LLM-scale CPT the dominant practical tools turn out to be *learning-rate control* and *replay*, not curvature penalties — they are cheaper and more robust. We will see why.

{{fig:cpt-stability-plasticity-tradeoff}}

The good news, established by Gupta et al. (2023) and the Ibrahim et al. (2024) "Simple and Scalable Strategies to Continually Pre-train LLMs" study, is that a *simple* recipe — re-warm the learning rate, re-decay it, and mix in a small fraction of replay data — recovers nearly the performance of an idealized from-scratch run on the union of the data, at a fraction of the cost. The rest of this chapter unpacks each ingredient.

## Learning-Rate Re-Warming and Re-Decaying

The single most important CPT knob is the learning-rate schedule. The base model finished training at its minimum LR $\eta_{\min}$ (the floor of a cosine decay, see [Learning Rate Schedules, Warmup, Batch Size & Hyperparameters](../03-pretraining/10-lr-schedules-hparams.html)). If you simply resume at $\eta_{\min}$, the model is in "polish" mode — gradients are tiny, and it will absorb almost nothing from $\mathcal{D}_{\text{new}}$. To actually *learn*, you must raise the LR back up. This is **re-warming**.

### The mechanics

A CPT schedule has three phases, mirroring a from-scratch run but compressed:

1. **Re-warmup** ($T_w$ steps): linearly ramp from a low value (often $\eta_{\min}$ of the base run, or even lower) up to a re-warm peak $\eta_{\text{cpt}}$.
2. **Stable/decay body**: hold or cosine-decay over the CPT token budget.
3. **Re-decay** ($T_d$ steps): cosine or linear decay down to a new floor.

Two decisions dominate outcomes: **how high to re-warm** ($\eta_{\text{cpt}}$) and **how to decay**.

{{fig:cpt-rewarm-schedule-and-spike}}

**How high?** The peak controls the stability–plasticity tradeoff directly. Empirically, re-warming to the *original pretraining peak* learns the new domain fastest but forgets the most. Re-warming to roughly $10\%$–$30\%$ of the original peak is the common sweet spot for domain adaptation; for a large *distribution shift* (e.g., a brand-new language) you push higher, toward $50\%$+; for a gentle *freshness* update (new web crawl, same distribution) you stay lower. A crucial subtlety from the Ibrahim et al. study: the *transient* loss spike caused by re-warming is temporary — the loss on both old and new data jumps up when you raise the LR, then recovers below the starting point. Do not panic at the spike; judge the run by where it settles after re-decay.

**How to decay?** You must re-decay to a low floor at the end, because — exactly as in the Warmup-Stable-Decay (WSD) schedule — most of the *committed* loss reduction happens during the decay phase. A CPT run held at high LR and never decayed looks unconverged. If you plan to do *several* CPT rounds (a streaming scenario), the WSD-style "decay only at the end" or "infinite LR schedule" of Ibrahim et al. is attractive: hold a constant high LR across rounds and only spend a short decay when you need a deployable checkpoint.

### Schedule code

Here is a from-scratch CPT scheduler with explicit re-warm and re-decay, written as a plain multiplier on the optimizer's base LR.

```python
import math
from torch.optim.lr_scheduler import LambdaLR

def make_cpt_schedule(
    optimizer,
    base_peak_lr: float,        # the ORIGINAL pretraining peak LR
    rewarm_fraction: float,     # eta_cpt = rewarm_fraction * base_peak_lr
    num_rewarm_steps: int,      # T_w: linear re-warmup
    num_total_steps: int,       # full CPT step budget
    num_decay_steps: int,       # T_d: final cosine re-decay
    start_lr_frac: float = 0.0, # LR at step 0 as a fraction of eta_cpt
    final_lr_frac: float = 0.05 # floor as a fraction of eta_cpt
):
    """
    Three-phase CPT LR schedule (re-warm -> stable -> re-decay).
    Returns a multiplier in [0, 1] RELATIVE to base_peak_lr, so set the
    optimizer's base lr == base_peak_lr.

        |       ____________________
        |      /                    \\
        |     /                      \\___
        |    /                           (floor)
        +---/------------------------------------>
          T_w        stable           T_d
    """
    eta_cpt_mult = rewarm_fraction          # peak as fraction of base peak
    stable_end   = num_total_steps - num_decay_steps

    def lr_lambda(step: int) -> float:
        if step < num_rewarm_steps:
            # Phase 1: linear re-warm from start_lr_frac*eta_cpt -> eta_cpt
            frac = step / max(1, num_rewarm_steps)
            mult = start_lr_frac + frac * (1.0 - start_lr_frac)
            return eta_cpt_mult * mult
        elif step < stable_end:
            # Phase 2: hold at the re-warm peak
            return eta_cpt_mult
        else:
            # Phase 3: cosine re-decay from eta_cpt -> final_lr_frac*eta_cpt
            prog = (step - stable_end) / max(1, num_decay_steps)
            prog = min(prog, 1.0)
            cos = 0.5 * (1.0 + math.cos(math.pi * prog))   # 1 -> 0
            mult = final_lr_frac + (1.0 - final_lr_frac) * cos
            return eta_cpt_mult * mult

    return LambdaLR(optimizer, lr_lambda)

# Example: base model trained at peak 3e-4, finished at 3e-5.
# We re-warm to 20% of peak (6e-5), over 2% of the CPT budget, then
# decay over the final 20%.
# optimizer base lr is set to 3e-4 (base_peak_lr).
# sched = make_cpt_schedule(opt, base_peak_lr=3e-4, rewarm_fraction=0.20,
#             num_rewarm_steps=400, num_total_steps=20_000,
#             num_decay_steps=4_000, final_lr_frac=0.05)
```

!!! warning "Common pitfall: skipping re-warmup causes a worse spike than doing it"
    It is tempting to think "the model is already trained, so I can jump straight to my target CPT learning rate with no warmup." Don't. Resuming a converged checkpoint and immediately applying a 10x-higher LR than it finished at produces a large gradient step into a region the optimizer's stale Adam second-moment estimates ($v_t$) are not calibrated for — you get a sharp loss spike and sometimes divergence. Always re-warm over at least a few hundred steps, and re-initialize or carefully load the optimizer state (see below).

### What about the optimizer state?

A subtle decision: do you resume the Adam optimizer state ($m_t$, $v_t$) from the base checkpoint, or reset it? Both are defensible. Resetting forces a fresh re-warmup (the $v_t$ estimates rebuild from scratch, so warmup is mandatory). Resuming preserves per-parameter learning-rate adaptation but the stale $v_t$ can mis-scale the first new gradients. The common practical choice is to **reset the optimizer state and re-warm**, which is also what you must do anyway if the model architecture changed (depth/width growth, new vocab) — the optimizer state has the wrong shape. We discuss precision and checkpoint-format issues in [Checkpointing, Fault Tolerance & Long-Running Jobs](../03-pretraining/12-checkpointing-fault-tolerance.html).

## Replay: Fighting Catastrophic Forgetting With Data

Re-warming makes the model *plastic*; **replay** keeps it *stable*. The idea is dead simple and remarkably effective: mix a fraction of *old-distribution* data back into the CPT data stream so that the gradient never sees pure $\mathcal{D}_{\text{new}}$. The **replay ratio** $r$ is the fraction of tokens drawn from $\mathcal{D}_{\text{base}}$ (or a proxy for it):

$$
\mathcal{D}_{\text{cpt}} = (1 - r)\,\mathcal{D}_{\text{new}} + r\,\mathcal{D}_{\text{replay}}.
$$

Typical values are $r \in [0.01, 0.30]$. The Ibrahim et al. study found that even **5% replay** dramatically reduces forgetting on the original distribution while costing almost nothing in new-domain learning, and that the marginal benefit of replay above ~25–50% is small. The intuition: a small but steady gradient signal pointing back toward $\mathcal{D}_{\text{base}}$ is enough to keep the model anchored, because forgetting is driven by *unopposed* drift away from the old minimum.

### What if you don't have the original data?

You usually don't have the exact $\mathcal{D}_{\text{base}}$. Three options, in rough order of preference:

1. **Proxy replay**: use a high-quality public general corpus (web text, books, Wikipedia, code) as a stand-in. It need not match the original mix exactly — it just needs to broadly cover the capabilities you want to preserve.
2. **Self-generated replay** (pseudo-rehearsal): sample text *from the base model itself* and replay that. The base model's own distribution is, by definition, the thing you're trying to preserve. This is the LLM analogue of generative replay and connects to [Synthetic Data for Pre- and Post-Training](../03-pretraining/15-synthetic-data.html).
3. **No replay, low LR**: if you genuinely cannot replay, lean harder on a *low* re-warm peak to limit drift — but accept more forgetting.

### A replay-aware data sampler

```python
import random
from itertools import cycle

def replay_mixed_stream(new_iter, replay_iter, replay_ratio: float, seed: int = 0):
    """
    Yield documents from a CPT stream that is `replay_ratio` old-distribution
    and the rest new-distribution. Token-level ratios are approximated at the
    document level here; for exact token budgets, weight by document length.

    new_iter    : iterator over NEW-domain documents (the target)
    replay_iter : iterator over BASE/REPLAY documents (anti-forgetting)
    replay_ratio: probability a given document comes from replay (e.g. 0.05)
    """
    rng = random.Random(seed)
    new_pool    = cycle(new_iter)      # in practice these are sharded streams,
    replay_pool = cycle(replay_iter)   # not in-memory; cycle is illustrative
    while True:
        if rng.random() < replay_ratio:
            yield next(replay_pool)
        else:
            yield next(new_pool)

# For *exact* token accounting you instead track running token counts and
# pull from whichever source is behind its target share -- this matters when
# documents vary wildly in length (code files vs. tweets).
def token_balanced_stream(new_iter, replay_iter, replay_ratio, len_fn=len):
    new_pool, replay_pool = cycle(new_iter), cycle(replay_iter)
    seen_new = seen_replay = 0
    while True:
        total = seen_new + seen_replay
        # current replay share; pull replay if we are below target, else new
        cur_replay_share = seen_replay / total if total else 0.0
        if cur_replay_share < replay_ratio:
            doc = next(replay_pool); seen_replay += len_fn(doc)
        else:
            doc = next(new_pool); seen_new += len_fn(doc)
        yield doc
```

!!! tip "Practitioner tip: measure forgetting on a frozen held-out base set"
    Before you start CPT, carve out a held-out evaluation set from the *base* distribution (general-knowledge MMLU-style tasks, a perplexity set on web text) and a held-out *new*-domain set. Log both every N steps. A healthy CPT run shows new-domain loss dropping steadily while base loss rises only slightly and then plateaus. If base loss climbs monotonically, raise the replay ratio or lower the re-warm peak. This two-curve plot is your single best diagnostic.

{{fig:cpt-replay-two-curve-diagnostic}}

## Domain, Language & Streaming Adaptation

The re-warm + replay recipe is the *engine*. The *application* — which domain, how big the shift, how the data arrives — sets the dials. Below are the canonical regimes.

### Domain adaptation (code, medical, legal, finance)

The classic motivation (Gururangan et al., *Don't Stop Pretraining*, 2020) is that even a strong general model benefits from a continued pass over in-domain text before any task fine-tuning. Modern recipes (StarCoder-style code adaptation, the BloombergGPT finance model, PMC/Meditron-style medical models, SaulLM legal models) follow the same arc:

- **Token budget**: a domain pass is typically **10–100 B tokens** — single-digit percent of the base run. Below ~5 B tokens you mostly polish; above ~100 B you risk drifting into a *new* base model with all the forgetting that implies.
- **Re-warm peak**: $\sim 10$–$30\%$ of base peak. Code adaptation often goes higher because code is structurally far from prose.
- **Replay**: $5$–$30\%$ general text, plus — importantly — **keep some of the base domain's neighbors**. For medical, replay general English *and* general science so you don't forget how to write fluently.
- **Tokenizer**: domain text may tokenize poorly (chemical formulae, legal citations, code symbols). You may need vocabulary extension — covered in the tokenizer-transfer section below.

The order matters too: domain CPT comes *before* instruction tuning and alignment ([Supervised Fine-Tuning & Instruction Tuning](../05-posttraining-alignment/01-sft-instruction-tuning.html)). CPT injects *knowledge*; SFT/RLHF inject *behavior*. Doing CPT after alignment tends to wash out the alignment.

### Language adaptation (adding or strengthening a language)

Extending a predominantly-English model to, say, Hindi or Swahili is the hardest CPT regime because the shift is large *and* the tokenizer is usually a bottleneck: an English-centric BPE tokenizer fragments other scripts into many tokens (high *fertility*), so the model wastes context and compute, and the new-language embeddings barely exist. The recipe:

1. **Extend the tokenizer** with new-language merges/tokens (see vocabulary transfer below), initializing new embeddings sensibly.
2. **Re-warm higher** ($30$–$60\%$ of base peak) — you are teaching genuinely new statistics.
3. **Heavy replay of the strong language(s)** ($\geq 25\%$) so the model retains English reasoning, which often *transfers* to the new language.
4. **Larger budget** ($50$–$300$ B tokens) than a same-language domain pass.

A counterintuitive but well-documented effect: cross-lingual transfer means a model that is strong in English and given even modest target-language CPT often reasons better in the target language than a model trained on the target language alone, because abstract reasoning circuits are largely language-agnostic.

### Streaming / temporal adaptation (fresh data over time)

Models go stale: a 2024-trained model does not know 2026 events. The streaming regime does CPT *repeatedly* as new data arrives. Key differences from a one-shot pass:

- **Schedule**: a one-shot cosine that decays to a floor "ends" the model. For repeated rounds, use a WSD/infinite schedule — hold a constant LR across rounds, decay only to produce a release checkpoint, then *resume from the pre-decay weights* (Ibrahim et al. show the decayed checkpoint is a worse starting point than the high-LR one). 
- **Replay across all past rounds**, not just the original base, to avoid forgetting *intermediate* knowledge.
- **Distribution-shift detection**: monitor per-domain perplexity to decide *when* a refresh is warranted.

This is the pretraining-side analogue of the production [Data Flywheels & Continuous Improvement](../12-production-mlops/05-data-flywheel.html) loop.

!!! example "Worked example: budgeting a medical domain-adaptive pass"
    We have a 7B-parameter base model trained on 2 T tokens at peak LR $3\times10^{-4}$, decayed to $3\times10^{-5}$. We want a medical variant.

    **Token budget.** Pick 40 B CPT tokens — that is $40/2000 = 2\%$ of the original token budget, a typical DAPT magnitude.

    **Compute.** Using the standard $C \approx 6ND$ FLOPs estimate (see [Scaling Laws](../03-pretraining/04-scaling-laws.html)), with $N = 7\times10^9$ params and $D = 4\times10^{10}$ tokens:

    $$
    C \approx 6 \times 7\times10^9 \times 4\times10^{10} = 1.68\times10^{21}\ \text{FLOPs}.
    $$

    On 64 H100s at, say, an achieved $4\times10^{14}$ FLOP/s each (≈40% of bf16 peak), aggregate is $2.56\times10^{16}$ FLOP/s, so wall-clock $\approx 1.68\times10^{21} / 2.56\times10^{16} \approx 6.6\times10^{4}\ \text{s} \approx 18$ hours. The *original* 2 T-token run was 50x this — **CPT costs ~2% of pretraining**, exactly the leverage we wanted.

    **Data mix.** $r = 0.15$ replay: $34$ B medical tokens + $6$ B general/science tokens. **Schedule:** re-warm to $0.20 \times 3\times10^{-4} = 6\times10^{-5}$ over 1% of steps, hold, then cosine re-decay over the final 20% to $3\times10^{-6}$. **Tokenizer:** add ~2k medical/biochemical tokens; initialize new embeddings as the mean of their BPE sub-token embeddings (see below).

## Model Growth & Weight Reuse

So far we kept the architecture fixed. The more ambitious form of "don't retrain from scratch" *changes the architecture* but **reuses the trained weights** as initialization. The premise — validated by Net2Net (Chen et al., 2015), bert2BERT, Gong et al.'s progressive stacking, and the dense-to-MoE *sparse upcycling* of Komatsuzaki et al. (2022) — is that a smaller trained model is a far better initialization for a larger one than random init, so you can "grow" into a bigger model and finish with a short CPT pass.

{{fig:cpt-function-preserving-growth}}

### Depth growth (layer stacking)

To go from $L$ to $2L$ layers, **duplicate** existing layers (interleaved or stacked) so the initial deeper network computes *nearly the same function* as the shallow one. The cleanest trick is **function-preserving** growth: add new layers initialized so their contribution is zero at step 0. Because transformer blocks are residual, $x \mapsto x + f(x)$, you can make a new block an identity by zeroing the output projection of its attention and MLP sublayers — then $f(x)=0$ and the block passes its input through untouched. Training then "wakes up" the new layers gradually.

```python
import torch, torch.nn as nn, copy

def grow_depth_identity(model, insert_after: list[int]):
    """
    Insert function-preserving (identity-at-init) transformer blocks.
    `model.layers` is a ModuleList of residual transformer blocks, each of
    the form: x = x + attn(ln1(x)); x = x + mlp(ln2(x)).
    We clone an existing block and ZERO its residual-output projections so the
    new block is the identity map at initialization -> the grown model
    computes the SAME function as the original at step 0 (loss is preserved).
    """
    new_layers = []
    for i, layer in enumerate(model.layers):
        new_layers.append(layer)
        if i in insert_after:
            twin = copy.deepcopy(layer)
            with torch.no_grad():
                # zero the output projection of attention (o_proj) and MLP
                # (down_proj). Names depend on your block; adapt accordingly.
                twin.attn.o_proj.weight.zero_()
                if twin.attn.o_proj.bias is not None:
                    twin.attn.o_proj.bias.zero_()
                twin.mlp.down_proj.weight.zero_()
                if getattr(twin.mlp.down_proj, "bias", None) is not None:
                    twin.mlp.down_proj.bias.zero_()
            new_layers.append(twin)
    model.layers = nn.ModuleList(new_layers)
    model.config.num_layers = len(new_layers)
    return model
```

The alternative, **progressive stacking** (train shallow, copy-stack to deeper, continue), was shown by Gong et al. to reach a target deep model faster in total FLOPs than training the deep model from scratch — the shallow phase is cheap and the warm start is good.

### Width growth

Widening a hidden dimension $d \to d'$ reuses Net2Net's *net2wider*: copy existing neurons and **split their outgoing weights** so the function is preserved. If neuron $j$ is duplicated $k$ times, divide each copy's outgoing weights by $k$ so the summed contribution is unchanged. To break the symmetry (otherwise the copies receive identical gradients forever and never diverge), add tiny noise. bert2BERT applies exactly this "function-preserving" widening to initialize a wide BERT from a narrow one and recovers most of the from-scratch performance with far less compute.

```python
@torch.no_grad()
def net2wider_linear(W_in: torch.Tensor, W_out: torch.Tensor, new_width: int,
                     noise: float = 1e-3):
    """
    Function-preserving width expansion of one hidden layer.
      W_in : (hidden, in)   -> produces the hidden activations
      W_out: (out, hidden)  -> consumes them
    Returns expanded (W_in', W_out') with hidden -> new_width such that the
    composed map is unchanged (up to small symmetry-breaking noise).
    """
    hidden, _in = W_in.shape
    assert new_width >= hidden
    # choose which existing neurons to duplicate (uniform with replacement)
    idx = torch.randint(0, hidden, (new_width - hidden,))
    pick = torch.cat([torch.arange(hidden), idx])         # (new_width,)
    # count copies of each original neuron, for the divide-by-k correction
    counts = torch.bincount(pick, minlength=hidden).float()
    W_in_new  = W_in[pick].clone()                        # replicate rows
    W_in_new += noise * torch.randn_like(W_in_new)        # break symmetry
    # outgoing weights: divide each replica by the # of copies of its source
    W_out_new = (W_out[:, pick] / counts[pick].unsqueeze(0)).clone()
    return W_in_new, W_out_new
```

### Dense-to-MoE upcycling

**Sparse upcycling** turns a trained *dense* model into a *sparse* Mixture-of-Experts model, reusing the dense weights. The recipe (Komatsuzaki et al., 2022; used at scale for several production MoEs): replace each (or every other) dense MLP block with an MoE layer whose $E$ experts are each **initialized as a copy of the original dense MLP**. The router is added fresh (small random init). At step 0, every expert is identical, so — if the router is roughly uniform — the MoE layer computes approximately the original dense MLP's output, preserving the function. Continued pretraining then *differentiates* the experts. This buys you a high-capacity MoE (more parameters, same or modestly higher FLOPs per token) for the cost of a CPT pass, rather than training an MoE from scratch.

```python
import torch.nn as nn, copy

class UpcycledMoE(nn.Module):
    """
    Replace a dense MLP with an MoE whose experts are clones of that MLP.
    At init all experts are identical, so with a near-uniform router the layer
    approximately reproduces the dense MLP -> function-preserving upcycle.
    See chapter 2.9 for routing, load balancing, and capacity factors.
    """
    def __init__(self, dense_mlp, num_experts=8, top_k=2):
        super().__init__()
        d_model = dense_mlp.up_proj.in_features
        # one router (gate) added fresh; small init so logits start ~uniform
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        nn.init.normal_(self.gate.weight, std=1e-3)
        # each expert is a deep copy of the trained dense MLP
        self.experts = nn.ModuleList(copy.deepcopy(dense_mlp)
                                     for _ in range(num_experts))
        self.top_k = top_k

    def forward(self, x):                       # x: (tokens, d_model)
        logits = self.gate(x)                   # (tokens, E)
        w, idx = torch.topk(logits.softmax(-1), self.top_k, dim=-1)
        w = w / w.sum(-1, keepdim=True)         # renormalize top-k weights
        out = torch.zeros_like(x)
        for slot in range(self.top_k):          # gather-scatter over experts
            for e, expert in enumerate(self.experts):
                mask = idx[:, slot] == e
                if mask.any():
                    out[mask] += w[mask, slot:slot+1] * expert(x[mask])
        return out
```

See [Mixture-of-Experts (MoE) Architectures](../02-transformer/09-mixture-of-experts.html) for routing, auxiliary load-balancing losses, and capacity factors — all of which you must tune during the upcycling CPT, because freshly-added routers are prone to collapse (routing all tokens to one expert).

### Vocabulary & tokenizer transfer

When you change the tokenizer — extending the vocab for a new domain/language, or swapping to a different tokenizer entirely — the embedding matrix $E \in \mathbb{R}^{|V|\times d}$ and the output (unembedding) matrix must change shape, and the new rows must be initialized. Random init for new tokens is wasteful; the trained model already "knows" the meaning of the *pieces* of a new token. Two standard moves:

- **Mean-of-subtokens init**: a new token (e.g., a whole word that the old tokenizer split into pieces) gets its embedding initialized to the **mean of its old sub-token embeddings**. This is the heuristic behind FOCUS and the widely-used embedding-init utilities, and it dramatically shortens the CPT needed to make the new tokens useful.
- **Shared-token preservation**: tokens present in *both* vocabularies keep their trained embeddings exactly. Only genuinely new tokens are initialized.

```python
import torch

@torch.no_grad()
def init_new_embeddings(old_emb, new_vocab, old_tokenizer, mean_init=True):
    """
    Build a new embedding matrix for an extended/changed vocabulary.
      old_emb       : (|V_old|, d) trained embedding matrix
      new_vocab     : dict {new_token_str -> new_id}
      old_tokenizer : can encode a string into OLD ids
    Shared tokens copy their trained vector; new tokens are initialized to the
    mean of the OLD sub-token embeddings of their surface string (FOCUS-style).
    Falls back to the overall mean (a safe centroid) when no sub-tokens exist.
    """
    d = old_emb.shape[1]
    new_emb = torch.empty(len(new_vocab), d)
    overall_mean = old_emb.mean(0)
    old_vocab = old_tokenizer.get_vocab()       # {token_str -> old_id}
    for tok, new_id in new_vocab.items():
        if tok in old_vocab:                     # shared: copy trained vector
            new_emb[new_id] = old_emb[old_vocab[tok]]
        elif mean_init:                          # new: mean of OLD sub-tokens
            sub_ids = old_tokenizer.encode(tok, add_special_tokens=False)
            if sub_ids:
                new_emb[new_id] = old_emb[torch.tensor(sub_ids)].mean(0)
            else:
                new_emb[new_id] = overall_mean
        else:
            new_emb[new_id] = overall_mean
    return new_emb
```

After re-initializing embeddings you almost always **CPT the whole model** (not just the embeddings) so the body learns to use the new vocabulary; a common warm-up is to first train *only* the new embedding rows for a few hundred steps with the rest frozen, then unfreeze everything. See [Tokenization: BPE, WordPiece, Unigram & Byte-Level](../02-transformer/01-tokenization.html) for how the merges that define new tokens are produced, and [Embeddings & The Input Pipeline](../02-transformer/02-embeddings-input.html) for embedding/unembedding tying.

## Planning CPT Compute With Loss-Trajectory Models

The final discipline is *planning*: before spending the compute, predict where the loss will land and how to split the budget. Three questions recur.

**1. How many tokens?** CPT obeys a scaling-law-like diminishing-returns curve. The new-domain loss after $D$ CPT tokens is well-modeled by a shifted power law,

$$
\mathcal{L}_{\text{new}}(D) \approx \mathcal{L}_\infty + \frac{A}{(D_0 + D)^{\alpha}},
$$

where $D_0$ encodes the "head start" the base model already has on the new domain, and $A,\alpha,\mathcal{L}_\infty$ are fit from a short pilot. Run a small pilot at, say, 1 B, 2 B, 4 B tokens, fit the curve, and extrapolate to find the *knee* where additional tokens stop paying. The same machinery is developed in [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html); CPT just adds the $D_0$ offset that represents transferred knowledge.

**2. How much forgetting?** Model the *base*-domain loss as *rising* with CPT tokens at a rate damped by the replay ratio $r$. A serviceable empirical form is

$$
\mathcal{L}_{\text{base}}(D, r) \approx \mathcal{L}_{\text{base}}(0) + B\,(1-r)\,D^{\beta},
$$

so larger replay $r$ flattens forgetting. Fitting both curves from the pilot lets you pick the $(D, r, \eta_{\text{cpt}})$ that hits a target — e.g., "maximize medical gain subject to base-loss regression $\le 2\%$."

**3. Grow now or later?** If you also plan model growth, the loss-trajectory model tells you whether to grow first (cheaper if the growth's function-preserving init starts near the base loss) or finish CPT on the small model first.

Here is a tiny fitter you can run on pilot points.

```python
import numpy as np
from scipy.optimize import curve_fit

def fit_cpt_trajectory(tokens, losses):
    """
    Fit L(D) = L_inf + A / (D0 + D)**alpha to pilot (tokens, loss) points,
    then return a predictor and the token count to hit a target loss.
    tokens : array of CPT token counts (e.g. [1e9, 2e9, 4e9])
    losses : measured new-domain loss at each.
    """
    def law(D, L_inf, A, D0, alpha):
        return L_inf + A / np.power(D0 + D, alpha)
    p0 = [min(losses) * 0.9, 1.0, 1e9, 0.3]            # rough initial guess
    popt, _ = curve_fit(law, np.asarray(tokens), np.asarray(losses),
                        p0=p0, maxfev=20000)
    L_inf, A, D0, alpha = popt

    def predict(D):                                    # loss at D tokens
        return L_inf + A / (D0 + D) ** alpha

    def tokens_for_target(target_loss):                # invert the law
        if target_loss <= L_inf:
            return float("inf")                        # unreachable: below L_inf
        return (A / (target_loss - L_inf)) ** (1.0 / alpha) - D0

    return predict, tokens_for_target, popt

# Example usage:
# predict, tokens_for, params = fit_cpt_trajectory(
#     [1e9, 2e9, 4e9], [2.41, 2.30, 2.22])
# print(predict(40e9))           # extrapolated loss at the full 40B budget
# print(tokens_for(2.10))        # tokens needed to reach loss 2.10
```

!!! interview "Interview Corner"
    **Q:** You have a strong 13B English base model and need a Japanese version on a fixed, modest compute budget. Walk me through your continual-pretraining plan and the top three things that will go wrong if you're careless.

    **A:** First, fix the **tokenizer**: an English BPE fragments Japanese into many tokens (high fertility), wasting context and compute. I'd extend the vocab with Japanese merges and initialize the new embedding rows as the **mean of their old sub-token embeddings**, copying the embeddings of any shared tokens unchanged. Second, the **schedule**: reset the optimizer state and **re-warm** to ~30–50% of the original peak LR (a language shift is large, so I want plasticity), then **re-decay** to a low floor at the end. Third, **replay**: keep ~25%+ English in the mix — English reasoning transfers cross-lingually, so heavy replay both prevents forgetting *and* lifts Japanese reasoning. Budget on the order of 50–200 B tokens, validated by a pilot-fit power law so I stop at the knee.

    The three failure modes: **(1) Catastrophic forgetting of English** if I re-warm too hot with no replay — the model becomes a mediocre monolingual model. **(2) The re-warm loss spike scaring me into aborting** — it's transient and recovers below baseline after decay; judge by the settled loss, not the spike. **(3) A broken tokenizer** — if I randomly init the new embeddings or don't fix fertility, I burn most of the budget just teaching the model to read, and the new tokens stay near-useless. I'd also CPT *before* any instruction tuning, since alignment applied first gets washed out.

## Key Takeaways

!!! key "Key Takeaways"
    - Continual pretraining (CPT) extends a converged base model on new data for a few percent of the original compute; the whole craft is navigating the **stability–plasticity dilemma** — learning the new without forgetting the old.
    - **Re-warm and re-decay** the learning rate: resume too cold and the model learns nothing; resume too hot and it forgets. ~10–30% of the original peak suits domain adaptation; push higher (~30–60%) for large shifts like a new language. Always re-warm over a few hundred steps and reset/recalibrate the optimizer state.
    - The re-warm **loss spike is transient** — judge a run by where loss settles after re-decay, not by the spike.
    - **Replay** a small fraction (often just 5%, up to ~30%) of old-distribution (or self-generated, or proxy) data to anchor the model against forgetting; even tiny replay buys most of the benefit.
    - **Domain DAPT** is ~10–100 B tokens before alignment; **language adaptation** needs tokenizer extension + heavy strong-language replay; **streaming** updates use a WSD/infinite schedule and replay across all past rounds.
    - **Grow, don't retrain**: function-preserving depth (zero the new block's residual output), width (Net2Net split-and-normalize), and **dense→MoE sparse upcycling** (clone the dense MLP into experts) let a trained model initialize a larger one, finished with a short CPT pass.
    - **Tokenizer/vocabulary transfer** initializes new token embeddings as the **mean of their old sub-tokens** and preserves shared tokens — this drastically shortens the CPT needed for the new vocab to become useful.
    - **Plan with loss-trajectory models**: fit a shifted power law $\mathcal{L}\approx\mathcal{L}_\infty + A/(D_0+D)^\alpha$ to a short pilot to choose token budget, replay ratio, and the forgetting/learning tradeoff before committing compute.

!!! sota "State of the Art & Resources (2026)"
    Continual and domain-adaptive pretraining is now a standard pillar of every production LLM pipeline: the re-warm + replay recipe from Ibrahim et al. (2024) has become the de-facto baseline, dense-to-MoE sparse upcycling is used at scale by several labs, and temporal-continual pretraining benchmarks (TiC-LM, 2025) are pushing the field toward rigorous evaluation of knowledge-freshness tradeoffs.

    **Foundational work**

    - [Gururangan et al., *Don't Stop Pretraining* (2020)](https://arxiv.org/abs/2004.10964) — the empirical case for domain-adaptive and task-adaptive pretraining (DAPT/TAPT) that established the practice.
    - [Ke et al., *Continual Pre-training of Language Models* (ICLR 2023)](https://arxiv.org/abs/2302.03241) — introduces soft-masking over important parameters to prevent forgetting while enabling knowledge transfer across sequential domains.
    - [Komatsuzaki et al., *Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints* (2022)](https://arxiv.org/abs/2212.05055) — the canonical dense-to-MoE upcycling recipe showing ~50% compute savings vs. training an MoE from scratch.

    **Recent advances (2023–2026)**

    - [Gupta et al., *Continual Pre-Training of LLMs: How to (re)warm your model?* (2023)](https://arxiv.org/abs/2308.04014) — systematic study of LR re-warming dynamics; shows the transient loss spike is harmless and rewarming consistently outperforms cold-resume.
    - [Ibrahim et al., *Simple and Scalable Strategies to Continually Pre-train LLMs* (2024)](https://arxiv.org/abs/2403.08763) — the definitive large-scale ablation of re-warming, re-decaying, replay ratios, and the WSD infinite-LR schedule for streaming CPT.
    - [Chen et al., *MEDITRON-70B: Scaling Medical Pretraining for LLMs* (2023)](https://arxiv.org/abs/2311.16079) — end-to-end DAPT recipe for medicine (48 B tokens from PubMed + guidelines on Llama-2), outperforming GPT-3.5 on medical benchmarks.
    - [Li et al., *TiC-LM: A Web-Scale Benchmark for Time-Continual LLM Pretraining* (ACL 2025)](https://arxiv.org/abs/2504.02107) — 114 Common Crawl dumps as a temporal benchmark; shows meta-schedules + fixed replay achieve comparable loss to retraining from scratch at 2.6× less compute.

    **Open-source & tools**

    - [apple/ml-tic-lm](https://github.com/apple/ml-tic-lm) — official code for TiC-LM: dataset pipelines, training scripts, and evaluation for time-continual pretraining experiments.
    - [Wang-ML-Lab/llm-continual-learning-survey](https://github.com/Wang-ML-Lab/llm-continual-learning-survey) — living paper list for the ACM CSUR 2025 comprehensive survey on continual learning of LLMs; organized by CPT, DAPT, and continual fine-tuning.

    **Go deeper**

    - [Dobler & de Melo, *FOCUS: Effective Embedding Initialization for Monolingual Specialization* (EMNLP 2023)](https://arxiv.org/abs/2305.14481) — sub-token-mean embedding init for vocabulary transfer; the standard reference for initializing new tokens when extending a tokenizer.
    - [AMD ROCm Blog, *Continued Pretraining: A Practical Playbook for Language-Specific LLM Adaptation* (2024)](https://rocm.blogs.amd.com/artificial-intelligence/multilingual-continued-pretraining/README.html) — end-to-end walkthrough of building a Finnish Llama 3.1 variant with data mixing, schedule tuning, and alignment.

## Further Reading

- **Gururangan et al., "Don't Stop Pretraining: Adapt Language Models to Domains and Tasks" (2020)** — the foundational case for domain-adaptive and task-adaptive continued pretraining (DAPT/TAPT).
- **Ibrahim et al., "Simple and Scalable Strategies to Continually Pre-train Large Language Models" (2024)** — the definitive modern recipe: re-warming, re-decaying, replay ratios, and infinite LR schedules, with large-scale ablations.
- **Gupta et al., "Continual Pre-Training of Large Language Models: How to (re)warm your model?" (2023)** — focused study of LR re-warming dynamics and the transient spike.
- **Kirkpatrick et al., "Overcoming Catastrophic Forgetting in Neural Networks" (EWC, 2017)** — the Fisher-information curvature penalty; the classic framing of forgetting that CPT replay sidesteps cheaply.
- **Chen et al., "Net2Net: Accelerating Learning via Knowledge Transfer" (2015)** — function-preserving net2wider/net2deeper transformations underlying model growth.
- **Gong et al., "Efficient Training of BERT by Progressively Stacking" (2019)** and **Chen et al., "bert2BERT: Towards Reusable Pretrained Language Models" (2021)** — depth/width growth with weight reuse for transformers.
- **Komatsuzaki et al., "Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints" (2022)** — the dense-to-MoE upcycling recipe.
- **Wu et al., "FOCUS: Effective Embedding Initialization for Monolingual Specialization of Multilingual Models" (2023)** — sub-token-mean embedding initialization for vocabulary/tokenizer transfer.
- **Wu et al., "BloombergGPT" (2023)** and **Chen et al., "Meditron" / SaulLM legal-LM reports (2023–2024)** — real domain-adaptive pretraining recipes for finance, medicine, and law.

## Exercises

**1.** *(Conceptual.)* A colleague wants to specialize your base model on legal text. The base model finished its cosine schedule at its floor LR $\eta_{\min} = 3\times10^{-5}$, and your colleague proposes simply resuming training on the legal corpus at exactly $\eta_{\min}$, "since that's where the model left off." (a) Which horn of the stability–plasticity dilemma does this proposal fall on, and what will the legal-domain loss curve look like? (b) Your colleague, told to "raise the LR," instead jumps straight to $3\times10^{-4}$ with no warmup. What goes wrong mechanically? (c) After you finally run a proper re-warm, the loss on *both* the legal set and the held-out base set spikes upward in the first few hundred steps. Should you abort? Why or why not?

??? note "Solution"
    **(a)** Resuming at $\eta_{\min}$ falls on the **plasticity** (learning) horn: the LR is in "polish" mode, gradients are tiny, and the model absorbs almost nothing from the new legal distribution. The legal-domain loss curve will be nearly flat — it barely drops below where it started. You paid compute and learned little.

    **(b)** Jumping straight to $3\times10^{-4}$ (a $10\times$ increase over the floor) with no re-warmup applies a large gradient step into a region for which the optimizer's stale Adam second-moment estimates $v_t$ are miscalibrated (they were tuned to the tiny end-of-run gradients). The result is a sharp loss spike and sometimes outright divergence. The fix is to re-warm over at least a few hundred steps and to reset (or carefully recalibrate) the optimizer state so $v_t$ rebuilds against the new data.

    **(c)** **Do not abort.** The transient spike on both old and new data is the expected, documented consequence of raising the LR (Ibrahim et al.); the loss recovers and typically settles *below* where it started once you re-decay. The correct diagnostic is the *settled* loss after re-decay, not the spike. Aborting here would throw away a run that is behaving exactly as designed.

**2.** *(Quantitative — replay accounting.)* You must ensure the model actually *sees* at least $50$ B tokens of new-domain (biomedical) text, and you want a replay ratio of $r = 0.2$ general-text tokens to fight forgetting. (a) What is the total CPT token budget $D$, and how many replay tokens does that imply? (b) Relative to a hypothetical replay-free run that also sees $50$ B new tokens, what percentage of *extra* compute does the replay cost? (c) Ibrahim et al. found the marginal benefit of replay above ~25–50% is small, and even 5% helps a lot. Given that, is $r=0.2$ a defensible choice here?

??? note "Solution"
    **(a)** New tokens are the $(1-r)$ share of the stream: $50 = (1-r)\,D = 0.8\,D$, so
    $$
    D = \frac{50}{0.8} = 62.5\ \text{B tokens}.
    $$
    Replay tokens are $r\,D = 0.2 \times 62.5 = 12.5$ B. Check: $50 + 12.5 = 62.5$ B. 

    **(b)** The replay-free run processes $50$ B tokens; this run processes $62.5$ B. Since compute scales with tokens ($C \approx 6ND$ at fixed $N$), the extra cost is
    $$
    \frac{62.5 - 50}{50} = 25\%\ \text{more compute}.
    $$
    (Equivalently, adding replay at ratio $r$ on top of a fixed new-token target multiplies compute by $1/(1-r)$.)

    **(c)** Yes. $r=0.2$ sits comfortably in the useful band: well above the 5% that already buys most of the anti-forgetting benefit, and below the ~25–50% region where returns flatten. For a large biomedical shift, spending 25% extra compute to substantially protect general-language ability is a reasonable trade — and you could drop toward $r=0.05$–$0.1$ if the compute budget were tighter, accepting somewhat more forgetting.

**3.** *(Quantitative — compute budgeting.)* Your base model has $N = 3\times10^{9}$ parameters and was pretrained on $1.5$ T tokens. You plan a $30$ B-token domain-adaptive pass on a cluster of $16$ H100s, each sustaining an *achieved* $4\times10^{14}$ FLOP/s. Using the $C \approx 6ND$ estimate: (a) What is the CPT FLOP cost and the wall-clock time? (b) What fraction of the original pretraining *token* budget is this pass?

??? note "Solution"
    **(a)** With $N = 3\times10^{9}$ and $D = 3\times10^{10}$,
    $$
    C \approx 6ND = 6 \times 3\times10^{9} \times 3\times10^{10} = 5.4\times10^{20}\ \text{FLOPs}.
    $$
    Aggregate throughput is $16 \times 4\times10^{14} = 6.4\times10^{15}$ FLOP/s, so
    $$
    t \approx \frac{5.4\times10^{20}}{6.4\times10^{15}} = 8.44\times10^{4}\ \text{s} \approx 23.4\ \text{hours}.
    $$

    **(b)** $D_{\text{cpt}} / D_{\text{base}} = 30\ \text{B} / 1{,}500\ \text{B} = 0.02 = 2\%$. This is squarely in the typical DAPT magnitude (single-digit percent of the base run) — the leverage that makes CPT worthwhile: a day of compute versus the months-long original run.

**4.** *(Conceptual — why upcycling preserves the function.)* Look at the `UpcycledMoE.forward` code. At initialization every expert is a deep copy of the same dense MLP. (a) Show that at step 0 the MoE layer reproduces the dense MLP's output *exactly*, and note precisely which line makes this independent of the router's logits. (b) The chapter's prose says the layer computes the dense output "approximately, if the router is roughly uniform." Reconcile that hedge with your exact result in (a). (c) Name the failure mode the chapter warns about for freshly-added routers, and explain why it becomes a real risk only *after* step 0.

??? note "Solution"
    **(a)** Let $g(x)$ be the shared expert function (all experts are identical clones, so $\text{expert}_e(x) = g(x)$ for every $e$). The forward pass selects the top-$k$ experts, renormalizes their gate weights with `w = w / w.sum(-1, keepdim=True)` so that $\sum_{\text{slot}} w_{\text{slot}} = 1$, then accumulates $\sum_{\text{slot}} w_{\text{slot}}\,\text{expert}(x)$. Since every expert returns the same $g(x)$,
    $$
    \text{out}(x) = \sum_{\text{slot}=1}^{k} w_{\text{slot}}\, g(x) = g(x)\sum_{\text{slot}} w_{\text{slot}} = g(x).
    $$
    The result equals the dense MLP output *exactly*, and the **renormalization line** (`w = w / w.sum(...)`) is what makes it independent of the router logits: whatever top-$k$ experts are chosen and with whatever raw softmax weights, the renormalized weights sum to 1 and multiply the single shared $g(x)$.

    **(b)** The "approximately / roughly uniform" hedge is the *general* statement for an upcycle where the top-$k$ selection could differ or where only a subset of experts are exact clones; it also covers implementations that do **not** renormalize the top-$k$ weights (there the output is $g(x)\sum w_{\text{slot}}$, which depends on how much softmax mass the top-$k$ captured, i.e. on router uniformity). For *this specific code* — identical experts *and* renormalized top-$k$ weights — the preservation is exact, so the hedge is conservative here.

    **(c)** **Router collapse**: the freshly-initialized router learns to send all tokens to a single expert (or a few), leaving the others untrained. It is not a risk at step 0 precisely because all experts are identical — routing is then irrelevant to the output, and the small-init router keeps logits near-uniform. Once CPT begins and experts start to *differentiate*, the router's choices start to matter, and without an auxiliary load-balancing loss (chapter 2.9) the positive feedback loop toward one expert can take over. Hence you must tune load balancing and capacity factors *during* the upcycling CPT.

**5.** *(Quantitative — loss-trajectory planning.)* You ran a short pilot and `fit_cpt_trajectory` returned the shifted power law $\mathcal{L}_{\text{new}}(D) = \mathcal{L}_\infty + A/(D_0 + D)^{\alpha}$ with $\mathcal{L}_\infty = 2.00$, $A = 200$, $D_0 = 1\times10^{9}$, $\alpha = 0.30$ (tokens in absolute counts). (a) Predict the new-domain loss at the planned full budget of $D = 40$ B tokens. (b) Using `tokens_for_target`, how many tokens would you need to reach a loss of $2.10$? (c) What does the answer to (b), compared to the $40$ B plan, tell you about where the "knee" of this curve is?

??? note "Solution"
    **(a)** At $D = 4\times10^{10}$, $D_0 + D = 4.1\times10^{10}$. Then
    $$
    (4.1\times10^{10})^{0.3} = \exp(0.3\ln 4.1\times10^{10}) = \exp(0.3 \times 24.44) = \exp(7.33) \approx 1.53\times10^{3},
    $$
    so
    $$
    \mathcal{L}_{\text{new}}(40\text{B}) = 2.00 + \frac{200}{1.53\times10^{3}} \approx 2.00 + 0.131 = 2.13.
    $$

    **(b)** Inverting the law (the `tokens_for_target` branch, valid since $2.10 > \mathcal{L}_\infty = 2.00$):
    $$
    D = \left(\frac{A}{\text{target} - \mathcal{L}_\infty}\right)^{1/\alpha} - D_0 = \left(\frac{200}{0.10}\right)^{1/0.3} - 10^{9} = (2000)^{3.333} - 10^{9}.
    $$
    $(2000)^{3.333} = \exp(3.333 \times \ln 2000) = \exp(3.333 \times 7.601) = \exp(25.34) \approx 1.01\times10^{11}$, so
    $$
    D \approx 1.01\times10^{11} - 1\times10^{9} \approx 1.0\times10^{11} = 100\ \text{B tokens}.
    $$

    **(c)** Going from the settled $2.13$ at $40$ B down to $2.10$ costs $\sim 100$ B tokens — **2.5$\times$ the budget for a $0.03$-nat gain**. Because loss is $\mathcal{L}_\infty + A/(D_0+D)^\alpha$ with small $\alpha$, returns diminish sharply: you are already past the knee at $40$ B. The rational move is to stop near the current plan (or reallocate the extra compute to replay / a larger model) rather than chase $2.10$.

**6.** *(Implementation — verify function-preserving depth growth.)* The `grow_depth_identity` function claims the grown model computes the *same* function as the original at step 0. Write a small runnable harness that builds a toy residual transformer, records its output on a batch, grows it, and asserts the output is unchanged. Then explain in one sentence which lines of `grow_depth_identity` are responsible for the invariance.

??? note "Solution"
    A minimal harness using blocks of the residual form the function assumes ($x \mapsto x + \text{attn}(...)$; $x \mapsto x + \text{mlp}(...)$), with the projection names `attn.o_proj` and `mlp.down_proj` that `grow_depth_identity` zeroes:

    ```python
    import torch, torch.nn as nn

    class ToyAttn(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.qkv    = nn.Linear(d, d)
            self.o_proj = nn.Linear(d, d)      # residual output projection
        def forward(self, x):
            return self.o_proj(torch.relu(self.qkv(x)))

    class ToyMLP(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.up_proj   = nn.Linear(d, 4 * d)
            self.down_proj = nn.Linear(4 * d, d)   # residual output projection
        def forward(self, x):
            return self.down_proj(torch.relu(self.up_proj(x)))

    class ToyBlock(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.attn, self.mlp = ToyAttn(d), ToyMLP(d)
        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x

    class ToyModel(nn.Module):
        def __init__(self, d, L):
            super().__init__()
            self.layers = nn.ModuleList(ToyBlock(d) for _ in range(L))
            self.config = type("cfg", (), {"num_layers": L})()
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    torch.manual_seed(0)
    model = ToyModel(d=16, L=4)
    x = torch.randn(2, 5, 16)

    with torch.no_grad():
        before = model(x)
    grow_depth_identity(model, insert_after=[1, 3])   # from the chapter
    with torch.no_grad():
        after = model(x)

    print("layers:", model.config.num_layers)                 # 4 -> 6
    print("max abs diff:", (before - after).abs().max().item())
    assert torch.allclose(before, after, atol=1e-6)
    print("function preserved at init: OK")
    ```

    Running this prints `layers: 6` and a max abs difference at the level of floating-point noise (`~1e-7`), and the assertion passes.

    **Why it works:** the lines `twin.attn.o_proj.weight.zero_()` and `twin.mlp.down_proj.weight.zero_()` (and their bias zeroing) make each inserted block's attention and MLP sublayers output exactly $0$, so — because the block is residual, $x \mapsto x + f(x)$ with $f(x) = 0$ — the twin is the identity map and passes its input through unchanged, leaving the whole network's function identical at step 0.
