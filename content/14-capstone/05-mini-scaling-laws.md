# 14.5 Mini Scaling Laws: Fit Your Own Law Before Spending the Budget

You are about to spend roughly **20 billion tokens** training `Stack-100M` — the plan budgets 15 to 25 A100-hours at USD 40 to 100, and the exact config is fixed in [The Architecture](../14-capstone/04-architecture.html) (we reproduce it below). Before you commit that budget you should be able to answer one question with a *number*, not a shrug: **what final loss should this run reach, and is the 100M/20B-token allocation actually the right one?** If your prediction and your run disagree by more than a couple of tenths of a nat, something is broken — a data-pipeline bug, a mistuned learning rate, a tokenizer mismatch — and you want to know that on hour two, not hour twenty.

The general theory of how loss scales with parameters $N$ and tokens $D$ is developed in [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html). That chapter gives you the *form* of the law and the folk constants. This chapter does the thing the theory chapter tells you to do but does not do for your specific setup: **it fits the law to your own recipe, on your own data, with your own tokenizer, and extrapolates to the exact model you are about to train.** The constants $E$, $A$, $B$, $\alpha$, $\beta$ are *not* universal — they depend on the corpus (FineWeb-Edu + Cosmopedia + a little code and math), the vocab (32,768), and every architecture choice frozen into the recipe. Importing another lab's numbers is how you mis-budget a run.

The plan of attack, following Hoffmann et al.'s Chinchilla methodology at miniature scale: train a **ladder of four tiny models** (about 4M, 9M, 19M, 44M non-embedding parameters) under the *identical* recipe, each at several token budgets, fit $L(N,D)=E+A/N^{\alpha}+B/D^{\beta}$ to the 17 measured losses, cross-check with **IsoFLOP profiles**, then **extrapolate** to 84.5M and predict `Stack-100M`'s loss. The ladder costs about **5 GPU-hours (~USD 7)** including a data-mixture screen — roughly 16% of the flagship run, and the cheapest insurance you will ever buy. Then we use the fitted law to justify the plan's most counterintuitive decision: **deliberately over-training to ~200 tokens/parameter**, an order of magnitude past compute-optimal.

Along the way we will fix three things that quietly wreck small-scale ladders and that most treatments skip: the $6ND$ FLOP rule is **wrong by 1.6×–3.6× at this scale and wrong by a *different* factor at every rung**; the learning rate **does not transfer across width**; and a fixed batch size gives your cheapest runs **too few optimizer steps for their own schedule to mean anything**.

---

## A Ladder Under One Frozen Recipe

A scaling-law fit is only as trustworthy as the *invariance* of everything you are not varying. The single rule that makes the ladder valid is this: **hold the entire recipe fixed and change only $N$ and $D$.** Same tokenizer, same data mixture and ordering, same optimizer (Muon + AdamW), same [WSD schedule](../03-pretraining/10-lr-schedules-hparams.html) shape, same RMSNorm/RoPE/NoPE/GQA/SwiGLU/QK-norm stack from the plan. If you let the architecture drift between rungs, you are no longer fitting a law — you are fitting noise, which is precisely the confound that broke Kaplan et al.'s original allocation (a learning-rate schedule that was not matched to each run's token count).

"Hold the recipe fixed" is subtler than it sounds, because two things you *must* change with width are the learning rate and the batch size. We come back to both after the parameter accounting.

### Scaling the ladder: deep-and-thin at every rung

`Stack-100M` is **deep-and-thin** (30 layers × 512 wide), following the small-model result that at fixed parameters, more layers × narrower width beats the reverse (MobileLLM, Liu et al. 2024). We scale the ladder the same way — growing `d_model` and `n_layers` *together* so every rung keeps roughly the same depth-to-width aspect ratio as the target. Head dimension stays pinned at 64; the small rungs collapse GQA to a single KV head (MQA), which is the natural small-model limit of the 8:2 grouping the target uses.

| Rung | `d_model` | `n_layers` | `n_heads` | `n_kv_heads` | `intermediate` | **N (non-embed)** | N (total) |
|---|---|---|---|---|---|---|---|
| **S1** | 192 | 10 | 3 | 1 | 512 | **3.93M** | 10.22M |
| **S2** | 256 | 13 | 4 | 1 | 704 | **9.16M** | 17.55M |
| **S3** | 320 | 17 | 5 | 1 | 896 | **18.80M** | 29.29M |
| **S4** | 448 | 21 | 7 | 1 | 1216 | **43.95M** | 58.63M |
| *target* | *512* | *30* | *8* | *2* | *1408* | *84.54M* | *101.32M* |

The ladder spans about **1.05 decades** in $N$ (3.93M → 43.95M) and the target sits just **0.28 decades** above the top rung — a short, honest extrapolation. That closeness is a feature of the USD-100 project: you are not predicting a 400B model from 100M runs (a 3.5-decade leap of faith), you are predicting 84.5M from 44M. The law barely has to reach.

### The embedding subtlety: which $N$ do you fit?

Here is a trap that is unique to *small* models and that the frontier-scale literature glosses over. The `Stack-100M` recipe uses a 32,768-token vocabulary, and the **tied** input/output embedding is a $32768 \times 512 = 16.78\text{M}$-parameter table (Press & Wolf 2017; the vocab-size tradeoff is analyzed in [Tokenization: BPE, WordPiece, Unigram & Byte-Level](../02-transformer/01-tokenization.html)). At the 4M rung, that embedding is *larger than the entire rest of the model* — 62% of total parameters at S1, versus 17% at the target.

There are exactly two defensible conventions, and **the literature is split, which is the single biggest source of confusion in this area**:

- **Kaplan et al. (2020)** fit on **non-embedding** parameters, $N_{\text{nonembed}}$, and paired that with $C = 6 N_{\text{nonembed}} D$.
- **Hoffmann et al. (Chinchilla, 2022)** fit and report **total** parameters. Chinchilla's famous "20 tokens per parameter" is 20 tokens per *total* parameter.

This is not a stylistic difference. **Pearce & Song, *Reconciling Kaplan and Chinchilla Scaling Laws* (arXiv 2406.12907, TMLR 2024)**, show that counting non-embedding rather than total parameters — combined with running the study at small scale — is a *primary* driver of the discrepancy between Kaplan's $N^\star \propto C^{0.73}$ and Chinchilla's $N^\star \propto C^{0.50}$. When they simulate the Chinchilla study under Kaplan's conventions, they reproduce Kaplan-like biased coefficients. And the effect is largest precisely at sub-1B scale — where we live.

So which do we use, and why is it safe?

!!! warning "The convention is only half the decision; the FLOP model is the other half"
    The bias Pearce & Song diagnose is not caused by counting non-embedding parameters *per se*. It is caused by the **mismatch** between using $N_{\text{nonembed}}$ as the capacity variable and using $6 N_{\text{nonembed}} D$ as the compute variable — because at small scale the embedding/output head and the attention term are a large, *size-dependent* fraction of the real compute that $6N_{\text{nonembed}}D$ throws away. Change one without the other and you rotate your exponents.

    `Stack-100M`'s ladder uses **$N_{\text{nonembed}}$ as the capacity variable** — the transformer blocks are what actually do the depth-driven sequence-mixing and feature-transformation work the $A/N^\alpha$ term is meant to describe, and the embedding's contribution to capacity is a lookup that does not deepen with the model — **paired with a full FLOP accounting that includes the attention and tied-head terms** (next section). That pairing is self-consistent. If you would rather follow Chinchilla and fit on $N_{\text{total}}$, do it — but then read the compute-optimal answers as total-parameter counts. We refit our own data under both conventions later in the chapter and show what changes.

{{fig:nonembed-vs-embedding-trap}}

```python
# stacklm/scaling/ladder.py  -- ladder configs + parameter accounting.
# The nonembed_params() arithmetic mirrors stacklm.model.StackConfig (Ch. 14.4).
from dataclasses import dataclass

@dataclass(frozen=True)
class LadderConfig:
    """A single rung. Every field except (d_model, n_layers) is *derived* so the
    recipe stays frozen: head_dim pinned at 64, SwiGLU width ~= 2.75*d_model
    rounded to a multiple of 64, KV heads = 1 for the small rungs (MQA limit)."""
    name: str
    d_model: int
    n_layers: int
    head_dim: int = 64
    n_kv_heads: int = 1
    vocab_size: int = 32768
    seq_len: int = 2048            # pretraining context; enters the FLOP count

    @property
    def n_heads(self) -> int:
        assert self.d_model % self.head_dim == 0
        return self.d_model // self.head_dim

    @property
    def intermediate(self) -> int:                 # SwiGLU hidden width
        raw = 2.75 * self.d_model
        return int(round(raw / 64) * 64)           # round to multiple of 64

    def nonembed_params(self) -> int:
        """Parameters in the transformer blocks (the capacity variable we fit).
        Per block: attention Q,O are d*d; K,V are d*(n_kv*head_dim) under GQA;
        SwiGLU MLP is 3*d*intermediate (gate, up, down). Norms are negligible."""
        d, kv, hd, inter = self.d_model, self.n_kv_heads, self.head_dim, self.intermediate
        attn = 2 * d * d + 2 * d * (kv * hd)       # Q,O  +  K,V
        mlp  = 3 * d * inter                       # SwiGLU: gate, up, down
        return self.n_layers * (attn + mlp)

    def embed_params(self) -> int:
        return self.vocab_size * self.d_model      # tied: counted once

    def total_params(self) -> int:
        return self.nonembed_params() + self.embed_params()

LADDER = [
    LadderConfig("S1", d_model=192, n_layers=10),
    LadderConfig("S2", d_model=256, n_layers=13),
    LadderConfig("S3", d_model=320, n_layers=17),
    LadderConfig("S4", d_model=448, n_layers=21),
]
TARGET = LadderConfig("Stack-100M", d_model=512, n_layers=30, n_kv_heads=2)

if __name__ == "__main__":
    for c in LADDER + [TARGET]:
        print(f"{c.name:10s} d={c.d_model:3d} L={c.n_layers:2d} "
              f"heads={c.n_heads} kv={c.n_kv_heads} inter={c.intermediate:4d}  "
              f"N_nonemb={c.nonembed_params()/1e6:6.2f}M  "
              f"total={c.total_params()/1e6:7.2f}M")
    # Stack-100M prints N_nonemb=84.54M, total=101.32M  -- matches the plan's ~101M.
```

Run it and the target line reads `N_nonemb=84.54M  total=101.32M`, reproducing the plan's parameter budget exactly (16.78M tied embedding + 84.54M in blocks). That arithmetic reproducibility is the point: if your own counter disagrees, fix it *now*, because every downstream FLOP and dollar estimate rides on it.

### The other confound: the learning rate does not transfer across width

Checklist item 3 below says every rung must get its own decayed schedule. There is a second, equally silent confound that most ladder write-ups omit: **the optimal learning rate depends on width**, and our rungs span $d_{\text{model}}$ from 192 to 512. Train them all at one LR and you systematically under-train the wide rungs (LR too low for them) or destabilize the narrow ones (LR too high) — a *width-correlated* error, which is exactly the kind that rotates $\alpha$ rather than just adding noise.

The principled fix is **muP / µTransfer** (Yang et al., *Tensor Programs V*, arXiv 2203.03466): reparameterize initialization scales, per-group learning rates, and the output multiplier so that the optimal hyperparameters become width-invariant, then tune once on the cheapest model and transfer. Microsoft's `mup` package (`pip install mup`) implements this for PyTorch by recording *base shapes* from a reference model and rescaling optimizer groups accordingly; Cerebras-GPT (Dey et al., 2023) is the public example of a Chinchilla-style ladder trained under muP. The mechanics are derived in [Learning Rate Schedules, Warmup, Batch Size & Hyperparameters](../03-pretraining/10-lr-schedules-hparams.html); here we only need the operational consequence.

Our optimizer is the Muon + AdamW hybrid from [Chapter 14.6](../14-capstone/06-optimizer-and-schedule.html), which splits the problem neatly:

- **Muon (2D hidden matrices).** Muon orthogonalizes the momentum and then rescales by $0.2\sqrt{\max(m,n)}$ (the "Muon is Scalable" RMS-matching trick, Liu et al., 2025). Because the orthogonalized update has a spectral norm set by the *shape* and the rescaling divides that shape dependence out, Muon's peak LR is far closer to width-invariant than Adam's. We hold it **constant at 0.02 across every rung**, and verify with a short sweep at S1 and S3 rather than assuming.
- **AdamW (embeddings, norms, 1D, the tied head).** These are the groups muP genuinely rescales. We apply the standard muP width scaling for matrix-like Adam groups: $\eta \propto 1/d_{\text{model}}$, anchored at the target's $3\times10^{-3}$ with base width $d_{\text{base}}=512$.

| Rung | `d_model` | Muon peak LR | AdamW peak LR $=3\times10^{-3}\cdot\frac{512}{d}$ |
|---|---|---|---|
| S1 | 192 | 0.02 | $8.0\times10^{-3}$ |
| S2 | 256 | 0.02 | $6.0\times10^{-3}$ |
| S3 | 320 | 0.02 | $4.8\times10^{-3}$ |
| S4 | 448 | 0.02 | $3.4\times10^{-3}$ |
| *target* | *512* | *0.02* | $3.0\times10^{-3}$ |

Everything else in the schedule is fixed by the run's own step count: **linear warmup for `min(2000, 5% of steps)`**, constant stable phase, then the WSD **$1-\sqrt{\cdot}$ decay over the final 20%** to ~0, exactly as in Chapter 14.6. Weight decay 0.1, betas $(0.9,0.95)$, grad-clip 1.0, bf16.

!!! tip "Practitioner tip: spend 2% of the ladder budget proving LR transfer"
    Before you launch the sweep, run S1 at one token budget at three LRs (say $0.5\times$, $1\times$, $2\times$ the table value) and do the same at S3. If the loss-vs-LR curves have their minima at the same *multiplier*, your parameterization transfers and you may run the whole ladder on one setting. If the minima drift with width, your scaling is wrong and every exponent you fit afterwards is contaminated. Six extra S1/S3 runs at low token counts cost well under a GPU-hour and convert an assumption into a measurement.

---

## Compute Accounting: What $6ND$ Leaves Out

To lay out the sweep we need to convert each `(N, D)` run into FLOPs and dollars. The workhorse everyone quotes is the dense-transformer rule derived in the [scaling-laws chapter](../03-pretraining/04-scaling-laws.html): **2 FLOPs per parameter per token forward, 4 backward, so $C \approx 6ND$.** It is an excellent rule at 7B. At 100M it is not, and at 4M it is off by a factor of 3.6.

Three terms make up the real per-token training cost of a decoder-only transformer (all "×3" factors are forward + backward $\approx 3\times$ forward):

1. **Block matmuls.** $2 N_{\text{nonembed}}$ FLOPs/token forward $\Rightarrow$ $6 N_{\text{nonembed}}$. This is the $6ND$ term.
2. **Attention scores and the value-weighted sum.** Per layer, $QK^\top$ over a length-$s$ sequence costs $2 s^2 d_{\text{model}}$ and $AV$ another $2 s^2 d_{\text{model}}$ — but causal masking halves both, so per *token* it is $2 s\, d_{\text{model}}$ per layer forward. Over $L$ layers, ×3: **$6 L s\, d_{\text{model}}$.** (GQA does not change this: the K/V heads are broadcast back to all query heads, so the score matmul still sums over the full $d_{\text{model}}$.)
3. **The tied output head.** A $d_{\text{model}} \times V$ matmul, $2 d_{\text{model}} V$ forward, ×3: **$6 d_{\text{model}} V$.** The input embedding is a gather and costs no meaningful FLOPs; the head does *all* of the vocabulary work.

$$
\text{FLOPs/token} \;=\; \underbrace{6 N_{\text{nonembed}}}_{\text{blocks}} \;+\; \underbrace{6\, n_{\text{layers}}\, s\, d_{\text{model}}}_{\text{causal attention}} \;+\; \underbrace{6\, d_{\text{model}} V}_{\text{tied head}}
$$

Working this rule out for our own ladder at $s=2048$, $V=32768$ is sobering:

| Rung | blocks | attention | head | **total FLOPs/token** | share blocks / attn / head | **total ÷ $6N$** |
|---|---|---|---|---|---|---|
| **S1** | $2.36\times10^{7}$ | $2.36\times10^{7}$ | $3.77\times10^{7}$ | $8.49\times10^{7}$ | 28% / 28% / **44%** | **3.60×** |
| **S2** | $5.50\times10^{7}$ | $4.09\times10^{7}$ | $5.03\times10^{7}$ | $1.46\times10^{8}$ | 38% / 28% / 34% | **2.66×** |
| **S3** | $1.13\times10^{8}$ | $6.68\times10^{7}$ | $6.29\times10^{7}$ | $2.43\times10^{8}$ | 47% / 28% / 26% | **2.15×** |
| **S4** | $2.64\times10^{8}$ | $1.16\times10^{8}$ | $8.81\times10^{7}$ | $4.67\times10^{8}$ | 56% / 25% / 19% | **1.77×** |
| *target* | $5.07\times10^{8}$ | $1.89\times10^{8}$ | $1.01\times10^{8}$ | $7.97\times10^{8}$ | 64% / 24% / 13% | **1.57×** |

Read the last column again. **$6ND$ undercounts by a factor that varies by more than 2× across the ladder.** At S1 the vocabulary head alone costs *more than the entire block stack*. Two runs you naively call "exactly $C$" can differ by 2.3× in real compute — which means IsoFLOP slices built on $6ND$ are not iso-FLOP at all, and their vertices are pulled toward larger $N$ (the rung whose true cost you underestimated least). It also means a ladder costed with $6ND$ is 2–3× cheaper on paper than in the cloud.

Keep $6ND$ as the mental rule of thumb — it is right about the *shape* of the cost and it is what everyone quotes. Just do not let it be your sweep's cost model.

```python
# stacklm/scaling/flops.py
def flops_per_token(cfg) -> dict:
    """Full training FLOPs per token: blocks + causal attention + tied head.
    Each term is (forward MACs x 2) x 3 for forward+backward.
    - blocks:    2 * N_nonembed  fwd  -> 6 * N_nonembed
    - attention: QK^T and AV are 2*s^2*d each; causal masking halves both,
                 so per TOKEN it is 2*s*d per layer fwd -> 6 * L * s * d
    - head:      the d x V logits matmul, 2*d*V fwd -> 6 * d * V
      (the input embedding is a gather: no meaningful FLOPs)"""
    blocks = 6.0 * cfg.nonembed_params()
    attn   = 6.0 * cfg.n_layers * cfg.seq_len * cfg.d_model
    head   = 6.0 * cfg.d_model * cfg.vocab_size
    return dict(blocks=blocks, attn=attn, head=head, total=blocks + attn + head)

def training_flops(cfg, n_tokens: float) -> float:
    """Total training FLOPs -- the number you budget and schedule against."""
    return flops_per_token(cfg)["total"] * n_tokens

def training_flops_6nd(cfg, n_tokens: float) -> float:
    """The 6ND approximation, kept ONLY for contrast and for reproducing
    published numbers that were quoted this way."""
    return 6.0 * cfg.nonembed_params() * n_tokens

def gpu_hours(flops: float, peak_flops_per_s: float = 312e12, mfu: float = 0.40) -> float:
    """Wall-clock on ONE accelerator. 312 TFLOP/s ~ A100 bf16 peak. MFU here is
    measured against the FULL model FLOPs above -- the only honest denominator
    (see Ch. 14.7, which measures it on the real run)."""
    return flops / (peak_flops_per_s * mfu) / 3600.0
```

!!! example "What the correction does to the flagship budget"
    `Stack-100M` at 200 tokens/param is $D = 1.691\times10^{10}$ tokens.

    - $6ND$ says $C = 6 \times 8.454\times10^{7} \times 1.691\times10^{10} = 8.58\times10^{18}$ FLOPs.
    - The full accounting says $C = 7.97\times10^{8} \times 1.691\times10^{10} = \mathbf{1.35\times10^{19}}$ FLOPs — **1.57× more**.

    On one A100 at 40% MFU (measured against full model FLOPs) that is **≈30 GPU-hours**; at the ~50% a well-compiled bf16 run with FlashAttention can reach at `seq_len=2048`, **≈24**. The plan's headline "15–25 A100-hours" corresponds to the $6ND$ figure (19 GPU-hr at 40% MFU). With the honest denominator, **budget about a day of A100 time**, USD 25–60 at USD 1–2/GPU-hr, and let [Chapter 14.7](../14-capstone/07-pretraining-run.html)'s measured throughput be the arbiter. This is the first thing the corrected FLOP model changes: your invoice.

    The other thing it changes is the *allocation* answer, because a fixed-size vocabulary head is a tax that big models amortize and small ones do not. We quantify that at the end of the IsoFLOP section.

### Designing the sweep: an IsoFLOP backbone plus off-diagonal points

Two forces pull against each other. To **identify** the law you want spread in *both* directions — some runs param-limited (few tokens per param), some data-limited (many). To keep the sweep **cheap** you cannot run the 44M rung to 400 tokens/param (that single run is $8.2\times10^{18}$ FLOPs — about 60% of the entire flagship). The resolution is a hybrid grid:

- **An IsoFLOP backbone.** Pick four compute budgets a factor of 3 apart, $C \in \{10^{16},\,3\times10^{16},\,9\times10^{16},\,2.7\times10^{17}\}$. Inside each budget, run every ladder rung whose implied $D = C/\text{FLOPs-per-token}$ lands at a sane tokens/param (roughly 6–500). Because every run in a slice costs the *same true* $C$, these slices give the IsoFLOP method its raw material — the compute-optimal $N$ at each budget — though with only four rungs each slice is thinly populated (we return to that limitation below).
- **A few off-diagonal fixed-model points.** IsoFLOP slices all lie on lines of constant compute — a degenerate direction that cannot separate $\alpha$ from $\beta$ in the additive form. Adding a handful of extra runs (every rung pushed down to 12 tokens/param, plus the two cheap rungs pushed to 400 and 150) breaks that degeneracy and lets the parametric fit pin the exponents.

{{fig:ladder-sweep-nd-plane}}

### Batch size: every rung needs enough steps for its schedule to mean anything

A fixed `batch_tokens = 2**19` for all runs is the other silent ladder-killer. The cheapest run in our sweep sees $4.7\times10^{7}$ tokens; at 0.5M tokens per step that is **90 optimizer steps** including warmup and the WSD decay tail — not a measurement, a rounding error. Worse, a 0.5M-token batch is far above the **critical batch size** for a 4M-parameter model at that token count (McCandlish et al., 2018; Zhang et al., *How Does Critical Batch Size Scale in Pre-training?*, arXiv 2410.21676, ICLR 2025, report a fit in which critical batch size grows roughly as $D^{0.47}$ and depends only weakly on $N$). Past the critical batch size the extra data parallelism buys almost no per-step progress, so those runs land *systematically high* in loss — and they are exactly the low-tokens-per-param points you added to pin $\beta$. The bias lands on the exponent you were trying to measure.

So we scale the batch with the run, under two constraints — **at least ~2000 optimizer steps** and **at or below the critical batch size** — and round down to a power of two:

```python
# stacklm/scaling/sweep.py -- design the ladder sweep, size its batches, cost it out.
import math
from stacklm.scaling.ladder import LADDER, TARGET
from stacklm.scaling.flops import flops_per_token, training_flops, gpu_hours

BY_NAME = {c.name: c for c in LADDER}

# (1) IsoFLOP backbone: (true_compute_budget_C, rung). Every run in a slice
#     costs exactly C under the FULL FLOP model -- so the slices really are iso-FLOP.
ISO = [(1e16, "S1"), (1e16, "S2"),
       (3e16, "S1"), (3e16, "S2"), (3e16, "S3"),
       (9e16, "S1"), (9e16, "S2"), (9e16, "S3"),
       (2.7e17, "S2"), (2.7e17, "S3"), (2.7e17, "S4")]
# (2) Off-diagonal fixed-model points: (rung, tokens_per_param).
EXTRA = [("S1", 12), ("S2", 12), ("S3", 12), ("S4", 12), ("S1", 400), ("S2", 150)]

MIN_STEPS = 2000

def critical_batch_tokens(D: float) -> float:
    """Empirical critical-batch-size fit (Zhang et al. 2025): B* grows ~ D^0.47
    and depends only weakly on N. Illustrative constants -- re-fit on your own
    runs if you care; the point is that B* is a function of D, not a constant."""
    return 22.91 * D ** 0.47

def batch_tokens(D: float) -> int:
    """Tokens per optimizer step: small enough for >= MIN_STEPS steps AND at or
    below the critical batch size, rounded DOWN to a power of two, clipped to
    [2**14, 2**19]. Sanity check: at the flagship's D this returns 2**19 =
    524288 tokens -- exactly the ~0.5M-token batch the plan specifies."""
    b = min(D / MIN_STEPS, critical_batch_tokens(D))
    b = 2 ** int(math.floor(math.log2(b)))
    return int(min(max(b, 2 ** 14), 2 ** 19))

def build_runs():
    runs = []                                  # each run is a dict the harness consumes
    for C, name in ISO:
        c = BY_NAME[name]
        D = C / flops_per_token(c)["total"]    # tokens s.t. TRUE cost == C exactly
        runs.append(dict(cfg=c, N=c.nonembed_params(), D=D, C=C,
                         tpp=D / c.nonembed_params(), kind="iso"))
    for name, tpp in EXTRA:
        c = BY_NAME[name]
        D = float(tpp * c.nonembed_params())
        runs.append(dict(cfg=c, N=c.nonembed_params(), D=D,
                         C=training_flops(c, D), tpp=float(tpp), kind="fixed"))
    for r in runs:                             # size the batch, then the schedule
        r["batch_tokens"] = batch_tokens(r["D"])
        r["steps"] = int(r["D"] / r["batch_tokens"])
        r["warmup_steps"] = min(2000, max(50, round(0.05 * r["steps"])))
    return runs

if __name__ == "__main__":
    runs = build_runs()
    for r in runs:
        print(f'{r["cfg"].name:3s} {r["kind"]:5s} C={r["C"]:.2e} '
              f'D={r["D"]/1e9:7.4f}B tpp={r["tpp"]:6.1f} '
              f'batch={r["batch_tokens"]:7d} steps={r["steps"]:6d}')
    ladder = sum(r["C"] for r in runs)
    D_big  = 200 * TARGET.nonembed_params()               # ~200 tok/param => 16.9B
    big    = training_flops(TARGET, D_big)
    print(f"{len(runs)} runs; ladder = {ladder:.3e} FLOPs "
          f"= {100*ladder/big:4.1f}% of the flagship ({big:.3e})")
    print(f"ladder wall-clock (1xA100, 40% MFU) = {gpu_hours(ladder):.2f} GPU-hr "
          f"~ USD {gpu_hours(ladder)*1.5:.2f}")
    print(f"flagship = {gpu_hours(big):.1f} GPU-hr @40% MFU, "
          f"{gpu_hours(big, mfu=0.50):.1f} @50%")
```

The printout is the sweep you are about to run — 17 jobs, none of them degenerate:

```text
S1  iso   C=1.00e+16 D= 0.1177B tpp=  29.9  batch=  32768 steps=  3592
S2  iso   C=1.00e+16 D= 0.0684B tpp=   7.5  batch=  32768 steps=  2087
S1  iso   C=3.00e+16 D= 0.3532B tpp=  89.8  batch= 131072 steps=  2694
S2  iso   C=3.00e+16 D= 0.2052B tpp=  22.4  batch=  65536 steps=  3131
S3  iso   C=3.00e+16 D= 0.1237B tpp=   6.6  batch=  32768 steps=  3774
S1  iso   C=9.00e+16 D= 1.0596B tpp= 269.5  batch= 262144 steps=  4042
S2  iso   C=9.00e+16 D= 0.6157B tpp=  67.2  batch= 262144 steps=  2349
S3  iso   C=9.00e+16 D= 0.3710B tpp=  19.7  batch= 131072 steps=  2830
S2  iso   C=2.70e+17 D= 1.8471B tpp= 201.7  batch= 262144 steps=  7046
S3  iso   C=2.70e+17 D= 1.1131B tpp=  59.2  batch= 262144 steps=  4246
S4  iso   C=2.70e+17 D= 0.5777B tpp=  13.1  batch= 262144 steps=  2203
S1  fixed C=4.01e+15 D= 0.0472B tpp=  12.0  batch=  16384 steps=  2880
S2  fixed C=1.61e+16 D= 0.1099B tpp=  12.0  batch=  32768 steps=  3354
S3  fixed C=5.47e+16 D= 0.2256B tpp=  12.0  batch=  65536 steps=  3442
S4  fixed C=2.47e+17 D= 0.5275B tpp=  12.0  batch= 262144 steps=  2012
S1  fixed C=1.34e+17 D= 1.5729B tpp= 400.0  batch= 262144 steps=  6000
S2  fixed C=2.01e+17 D= 1.3738B tpp= 150.0  batch= 262144 steps=  5241
17 runs; ladder = 1.846e+18 FLOPs = 13.7% of the flagship (1.347e+19)
ladder wall-clock (1xA100, 40% MFU) = 4.11 GPU-hr ~ USD 6.16
flagship = 30.0 GPU-hr @40% MFU, 24.0 @50%
```

Every rung gets 2,000–7,000 steps, so its WSD warmup/stable/decay phases all mean something; the batch never exceeds the critical size; and the same `batch_tokens()` rule, evaluated at the flagship's 16.9B tokens, returns exactly the plan's 0.5M-token batch. **The sweep is ~14% of the flagship compute — about 4 GPU-hours, roughly USD 6.** Note the honest scale effect: at the frontier a scaling ladder is well under 1% of the target run. Here the top rung (44M) is *half* the target (84.5M), so the ladder is an unavoidably larger fraction — but 14% of a day-long run to know the answer before you spend the other 86% is still the best trade in the project.

!!! warning "Match the LR decay to each run's own token count"
    The single most common way to poison a scaling ladder is to reuse one long learning-rate schedule and read off intermediate losses. A run whose [WSD or cosine decay](../03-pretraining/10-lr-schedules-hparams.html) has not finished evaluates *worse than it truly is*, which inflates the high-token losses and biases the fit toward "make the model bigger" — the exact Kaplan confound Chinchilla diagnosed. Every rung in the sweep gets its **own** schedule that decays to zero at *its* token budget $D$. In WSD terms (Ch. 14.6) that means a short run gets a short stable phase and its own decay tail — which is why `build_runs()` computes `steps` and `warmup_steps` per run and passes them down.

### The sweep harness

The harness is thin — it drives the real training loop from Chapter 14.7 and records one JSON line per run, plus the full loss curve (we need those curves later, for the live monitor). In continuous integration the loop is stubbed to a few steps on a synthetic corpus (hermetic, CPU-only); on the A100 it runs for real.

```python
# stacklm/scaling/run_sweep.py
import json, os
from stacklm.scaling.sweep import build_runs
from stacklm.train import train_run        # thin convenience wrapper over the Ch. 14.7 loop
from stacklm.model import StackConfig, Stack100M
from stacklm.data import PackedDataset      # Ch. 14.2 memmap shards

D_BASE = 512                                # muP base width = the target's d_model

def run_one(run, data_dir, out_path):
    c = run["cfg"]
    model_cfg = StackConfig(                # frozen recipe -- only d_model/n_layers vary
        vocab_size=c.vocab_size, d_model=c.d_model, n_layers=c.n_layers,
        n_heads=c.n_heads, n_kv_heads=c.n_kv_heads, head_dim=c.head_dim,
        intermediate=c.intermediate, rope_theta=10000.0,
        tie_embeddings=True, nope_every=4, qk_norm=True,     # <- identical across rungs
    )
    model = Stack100M(model_cfg)
    data = PackedDataset(data_dir, seq_len=c.seq_len)
    # Muon's LR is width-invariant under the 0.2*sqrt(max(m,n)) RMS matching;
    # the AdamW group gets the muP 1/width scaling anchored at the target.
    out = train_run(
        model, data,
        total_tokens=int(run["D"]),
        batch_tokens=run["batch_tokens"],
        optimizer="muon+adamw",
        lr_muon=0.02,
        lr_adamw=3e-3 * D_BASE / c.d_model,
        schedule="wsd", warmup_steps=run["warmup_steps"], decay_frac=0.20,
        weight_decay=0.1, grad_clip=1.0, seed=1234,
        log_every_tokens=run["D"] / 200,    # ~200 curve points, for the live monitor
    )
    row = dict(name=c.name, N=run["N"], D=run["D"], C=run["C"],
               tpp=run["tpp"], kind=run["kind"], steps=run["steps"],
               batch_tokens=run["batch_tokens"],
               val_loss=out.final_val_loss,          # LR fully decayed to THIS run's D
               curve=out.curve)                      # [(tokens, val_loss), ...]
    with open(out_path, "a") as f:
        f.write(json.dumps(row) + "\n")
    return row

if __name__ == "__main__":
    out = "ladder_results.jsonl"
    if os.path.exists(out):
        os.remove(out)
    for run in build_runs():
        r = run_one(run, data_dir=os.environ["STACKLM_DATA"], out_path=out)
        print(f"{r['name']:4s} N={r['N']/1e6:5.1f}M D={r['D']/1e9:5.3f}B "
              f"tpp={r['tpp']:6.1f} steps={r['steps']:5d} -> val_loss={r['val_loss']:.4f}")
```

### Designing your own ladder: a checklist

If you adapt this to a different target than `Stack-100M`, the design choices above generalize into a short checklist. Every item exists to defeat a specific failure mode.

1. **Span at least a decade in $N$.** Our rungs cover 3.9M → 44M (1.05 decades). Less spread and the exponent $\alpha$ is unconstrained; the power law needs leverage.
2. **Keep the top rung within ~0.3 decades of the target.** We extrapolate 44M → 84.5M. Extrapolating a factor of 2 is honest; extrapolating a factor of 100 is a prayer. The further the reach, the more the loose $E,A,B$ offset hurts you.
3. **Give every rung its own decayed schedule.** Non-negotiable — this is the Kaplan confound, and it is a one-line bug (a shared `total_steps`) that silently rotates your whole fit.
4. **Cost the sweep with a full FLOP model, not $6ND$.** Include the causal-attention and output-head terms. At sub-1B scale the correction is 1.6×–3.6× and *rung-dependent*, so $6ND$ slices are not iso-compute.
5. **Scale the learning rate with width (muP), or prove you do not have to.** Then state the per-rung LR in the config, not in a comment. A width-correlated LR error rotates $\alpha$ just as surely as a schedule error does.
6. **Size the batch per run: $\ge$ ~2000 steps and $\le$ the critical batch size.** Echo `steps` in the sweep printout so a 90-step run is visible and gets rejected before it poisons $\beta$.
7. **Freeze everything else byte-for-byte.** Same tokenizer, data order, `nope_every`, `qk_norm`, optimizer split. If you must change one thing (say, test MLA vs GQA), that is a *separate* ladder, not a mixed one.
8. **Budget 10–20% of the flagship compute at this scale** (well under 1% at frontier scale, where the ladder is a rounding error). Ours is ~14% for the law plus ~3% for the data-mixture screen.
9. **Add a few seeds at the cheapest rung.** Two or three reruns of S1 at one token budget measure your noise floor directly, which tells you the Huber `delta` to use and whether two rungs *really* differ or just wiggled.
10. **Hold out the top rung, refit, and check the prediction.** The only success criterion that matters is extrapolation: fit on S1–S3, predict S4, and demand agreement before you trust the leap to 84.5M.

The through-line: a scaling ladder is a *measurement instrument*, and like any instrument it is worthless if uncalibrated. Every item above is a calibration step.

---

## Fitting Your Own L(N, D)

With `ladder_results.jsonl` in hand we fit the Chinchilla parametric form:

$$
L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}
$$

$E$ is the irreducible floor (the entropy of the mixture under this tokenizer), $A/N^{\alpha}$ the finite-capacity penalty, $B/D^{\beta}$ the finite-data penalty. Two robustness tricks — both from Chinchilla's appendix and explained in the [scaling-laws chapter](../03-pretraining/04-scaling-laws.html) — separate a real fit from a toy one: (1) build $\log L$ via **`logsumexp`** of the three log-terms, which is numerically stable across the orders of magnitude the terms span; (2) minimize the **Huber loss** of the residual in log space, so the few noisiest runs cannot dominate.

The code below is self-contained: because you cannot run 17 GPU jobs inside a textbook, it *synthesizes* the loss table from a plausible ground-truth law (the numbers a real Stack-recipe ladder would reveal) and then fits it — proving the machinery recovers what it should. In practice you delete the synthesis block and `json.load` your real results; the fitting code is byte-for-byte identical.

```python
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

# --- Load the measured ladder. In production: read ladder_results.jsonl. ---------
# Here we synthesize from a ground-truth law so the demo is hermetic. These
# constants are what a real Stack-recipe ladder on this mix would look like:
# a higher floor E than web text (small vocab, educational corpus) and a strong
# data term (tiny models are data-hungry at these token counts).
GROUND_TRUTH = dict(E=2.45, A=124.0, alpha=0.33, B=234.0, beta=0.30)
def _law(N, D, p): return p["E"] + p["A"]*N**(-p["alpha"]) + p["B"]*D**(-p["beta"])

from stacklm.scaling.sweep import build_runs
rng = np.random.default_rng(0)
runs = build_runs()
# dtype=float matters: an integer N array silently truncates the fit's E term.
N_obs = np.array([r["N"] for r in runs], dtype=float)
D_obs = np.array([r["D"] for r in runs], dtype=float)
# ~1% multiplicative noise mimics seed / data-order variation between runs:
L_obs = _law(N_obs, D_obs, GROUND_TRUTH) * (1.0 + 0.01*rng.standard_normal(len(runs)))

# --- The fit. Parameterize positives via logs: E=exp(e), A=exp(a), B=exp(b). -----
def predict_log_loss(theta, N, D):
    e, a, b, alpha, beta = theta
    terms = np.stack([
        np.full_like(N, e),            # log E
        a - alpha * np.log(N),         # log(A * N^-alpha)
        b - beta  * np.log(D),         # log(B * D^-beta)
    ])
    return logsumexp(terms, axis=0)    # = log(E + A N^-alpha + B D^-beta), stably

def huber(r, delta=1e-3):
    ar = np.abs(r)
    return np.where(ar <= delta, 0.5*r**2, delta*(ar - 0.5*delta))

def fit(N, D, L, rng, bounds_exp=(0.25, 0.42), n_starts=60):
    """Multi-start L-BFGS-B on a non-convex surface. We CONSTRAIN the exponents
    to the band the scaling literature consistently reports -- with only 4 rungs
    spanning ~1 decade in N, an unconstrained fit will happily rail alpha to an
    absurd 0.1 or 0.6 and take the extrapolation with it. Bounding the exponents
    to physically-sane values is the single most important regularizer here."""
    def objective(theta):
        return np.sum(huber(predict_log_loss(theta, N, D) - np.log(L)))
    best, best_val = None, np.inf
    for _ in range(n_starts):
        x0 = np.array([rng.uniform(0.4, 1.1), rng.uniform(3, 7), rng.uniform(3, 7),
                       rng.uniform(*bounds_exp), rng.uniform(*bounds_exp)])
        res = minimize(objective, x0, method="L-BFGS-B",
                       bounds=[(0.2, 1.3), (0, 12), (0, 12), bounds_exp, bounds_exp])
        if res.fun < best_val:
            best, best_val = res.x, res.fun
    return best

best = fit(N_obs, D_obs, L_obs, rng)
e, a, b, alpha, beta = best
print(f"fitted:  E={np.exp(e):.3f}  A={np.exp(a):.1f}  alpha={alpha:.3f}  "
      f"B={np.exp(b):.1f}  beta={beta:.3f}")
print(f"allocation exponent  beta/(alpha+beta) = {beta/(alpha+beta):.3f}")
```

Across ten noise seeds this grid returns an allocation exponent $\beta/(\alpha+\beta)$ scattered over **0.37–0.60** (median ≈0.45) around the ground-truth $0.30/0.63 = 0.476$, while the *individual* constants swing far more wildly: $E$ from ~2.13 to ~2.69 against a true 2.45, and $A$ from ~58 to ~430 against a true 124. That instability is not a bug; it is the whole lesson of the next section.

### Read the fit honestly: the loss extrapolates, the constants do not

Compare any single seed's fitted constants to the ground truth (`E=2.45, A=124, alpha=0.33, B=234, beta=0.30`) and the mismatch is glaring — $A$ routinely comes out **2× too small or 3.5× too large**, $E$ wanders a third of a nat — and *the constants change every time you re-run with different noise*. This is *not* a bug, and it is the single most important thing to internalize about scaling-law fits: **$E$, $A$, and $B$ are strongly correlated and only weakly identified.** The additive surface is nearly flat near the optimum, so many $(E,A,B)$ triples fit the same 17 points almost equally well; they trade off against each other and only their *combination* is pinned. If your colleague's fit reports different constants, neither of you is wrong. Epoch AI's replication of Chinchilla (Besiroglu et al., 2024) found the original parametric constants were fragile for exactly this reason.

Be equally honest about the exponents. It is tempting to say "the allocation exponent is robust even though the constants are not," and at frontier-scale ladders that is roughly true — but at *our* ladder size, with 17 points and 1% run-to-run noise, $\beta/(\alpha+\beta)$ still wanders by ±0.1. Quote it as "roughly one-for-one, Chinchilla-like," never to three digits.

What *is* reliably pinned is the quantity you actually care about — the **extrapolated loss**. Because the fit interpolates the measured surface faithfully, and the target sits only 0.28 decades beyond the top rung, the compensating errors in $E$, $A$, $B$ largely cancel where it matters:

```python
def predicted_loss(theta, N, D):
    return float(np.exp(predict_log_loss(theta, np.array([N]), np.array([D])))[0])

N100 = 84.54e6
for tpp in (20, 200):
    D = tpp * N100
    print(f"Stack-100M @ {tpp:3d} tok/param (D={D/1e9:5.2f}B): "
          f"predicted L={predicted_loss(best, N100, D):.3f}  "
          f"(ground truth {_law(N100, D, GROUND_TRUTH):.3f})")
# across 10 noise seeds:
# @ 20  tok/param: predicted L in 3.08-3.21   (ground truth 3.149)
# @ 200 tok/param: predicted L in 2.85-3.07   (ground truth 2.950)
# The ~0.1-nat BAND is the stable thing; the individual digits are not.
```

Across noise seeds the extrapolation stays **within about 0.1 nats** of the truth, with no consistent bias — some seeds land a little high, some a little low. That is exactly the resolution you should expect from a four-rung ladder — and exactly enough. The lesson, straight from the theory chapter: *validate a scaling-law fit by extrapolation, never by admiring the raw constants.* Report the predicted loss with an honest ±0.1-nat band, not five decimal places of $A$. A ±0.1-nat band is useless for splitting hairs between two good runs but perfect for its real job: **a broken run misses by 0.3+ nats**, and that you will catch on hour two.

The sharpest version of that validation is checklist item 10 — fit without the top rung and predict it:

```python
# Hold out S4 entirely, refit on S1-S3, predict the two held-out S4 points.
mask = np.array([r["cfg"].name != "S4" for r in runs])
theta_ho = fit(N_obs[mask], D_obs[mask], L_obs[mask], np.random.default_rng(7))
pred = np.exp(predict_log_loss(theta_ho, N_obs[~mask], D_obs[~mask]))
print("held-out S4:", np.round(pred, 3), "vs actual", np.round(L_obs[~mask], 3))
# Across seeds the held-out S4 predictions land within 0.02-0.06 nats of the
# measured values. THAT is the number that licenses the leap to 84.5M --
# not the fitted A, and not the R^2 on the training points.
```

{{fig:loss-extrapolates-constants-dont}}

### Which parameter count? Refit under both conventions

We chose $N_{\text{nonembed}}$ above and promised evidence. Here it is — the same 17 measurements, fit twice, once with each convention, with the exponent bounds widened to $[0.15,0.60]$ so we can *see* the exponent move rather than watch it hit a wall:

```python
N_total = np.array([r["cfg"].total_params() for r in runs], dtype=float)

for seed in range(6):
    rng_s = np.random.default_rng(seed)
    L_s = _law(N_obs, D_obs, GROUND_TRUTH) * (1 + 0.01*rng_s.standard_normal(len(runs)))
    t_ne = fit(N_obs,   D_obs, L_s, np.random.default_rng(1000+seed), bounds_exp=(0.15, 0.60))
    t_to = fit(N_total, D_obs, L_s, np.random.default_rng(1000+seed), bounds_exp=(0.15, 0.60))
    print(f"seed{seed}: non-embedding alpha={t_ne[3]:.3f} | total-param alpha={t_to[3]:.3f}")
# non-embedding: alpha in 0.29-0.60, median ~0.43  (ground truth 0.33)
# total params : alpha in 0.43-0.60, median ~0.59, sitting at or within 0.02 of
#                the upper bound in four of six seeds
```

The non-embedding fits scatter *around* the truth. The total-parameter fits sit systematically **higher** and pile up against the upper bound. The mechanism is the one Exercise 1 works through: the embedding is $32768 \cdot d$, growing only linearly in width, while block parameters grow like $n_{\text{layers}} \cdot d^{2}$. So the embedding's *share* of $N_{\text{total}}$ falls monotonically up the ladder (62% → 48% → 36% → 25% → 17%), which flattens the low-$N$ end of the curve and forces the fitter to rotate $\alpha$ to compensate. That rotation is Kaplan's $C^{0.73}$ in miniature, and it is exactly what Pearce & Song reconstruct at full scale.

Two honest caveats. First, this demo is rigged in the non-embedding convention's favour, because the synthetic ground truth was *generated* on $N_{\text{nonembed}}$ — it shows that mixing conventions between generation and fitting corrupts $\alpha$, not that nature prefers one. Second, the deeper point stands regardless: **whichever variable you fit, say so, and make your compute variable consistent with it.** Fit $N_{\text{total}}$ and you should also stop pretending the head is free; fit $N_{\text{nonembed}}$ and you must count the head's FLOPs separately, which is exactly what our `flops_per_token()` does.

### IsoFLOP Profiles: Chinchilla's Second Method

The parametric fit is Chinchilla's Approach 3 (fit the whole surface, differentiate). Its independent cross-check is **Approach 2, the IsoFLOP method**, which never commits to the parametric form and is therefore robust to its misspecification. The idea: at each fixed compute budget $C$, the loss as a function of $\log N$ (with $D$ forced by the FLOP constraint) is a **U-shaped valley** — too-small models are param-limited, too-large models are data-starved. Fit a **parabola in $\log N$**, read off the vertex, and you have the compute-optimal $N^\star(C)$ for that budget without ever assuming a power law. Do it for several budgets and fit a line through the valleys: its slope is the allocation exponent $a$ in $N^\star \propto C^{a}$.

There is a catch at *our* scale, and it is worth stating plainly: **a parabola needs enough points, bracketing the minimum on both arms, or its vertex is garbage.** Our four-rung ladder puts only two to four points in each IsoFLOP slice — too few. So we do the honest thing: demonstrate the method on a properly dense grid (7 models per slice, the way Hoffmann et al. actually ran it), and use the real ladder's coarse valleys only as a loose sanity check. A production ladder would simply include more rungs.

Because the slices must be genuinely iso-compute, the demo searches over *real configurations* in the ladder's deep-and-thin family and forces $D$ from the full FLOP model:

```python
import math, numpy as np
from stacklm.scaling.ladder import LadderConfig
from stacklm.scaling.flops import flops_per_token

def family(d_model: int, n_kv_heads: int = 2) -> LadderConfig:
    """A smooth continuation of the ladder's deep-and-thin aspect ratio,
    n_layers ~= d_model/19.2 (S1..S4 give 10/13/17/23). This is the SEARCH
    SPACE for the allocation question -- the target itself is pinned by the
    plan at d=512/L=30, a little deeper than the family's L=27."""
    return LadderConfig(f"d{d_model}", d_model=d_model,
                        n_layers=max(4, round(d_model / 19.2)), n_kv_heads=n_kv_heads)

def slice_configs(C, n=7, span=0.6):
    """Seven real configs for one IsoFLOP slice: coarse-scan for the valley,
    then span +/- `span` decades of N around it. CENTERING MATTERS -- an
    off-centre grid is one of the documented biases of the parabola method."""
    coarse = [family(d) for d in range(64, 1345, 64)]
    losses = [_law(c.nonembed_params(), C / flops_per_token(c)["total"], GROUND_TRUTH)
              for c in coarse]
    N_c = coarse[int(np.argmin(losses))].nonembed_params()
    return [min((family(d) for d in range(64, 2049, 32)),
                key=lambda c: abs(math.log(c.nonembed_params() / t)))
            for t in N_c * np.logspace(-span, span, n)]

rng2 = np.random.default_rng(1)
C_slices = np.array([1e16, 3e16, 9e16, 2.7e17])
N_star = []
for C in C_slices:
    cfgs = slice_configs(C)
    Ns = np.array([c.nonembed_params() for c in cfgs], dtype=float)
    Ds = np.array([C / flops_per_token(c)["total"] for c in cfgs])   # TRUE iso-FLOP
    Ls = _law(Ns, Ds, GROUND_TRUTH) * (1 + 0.01*rng2.standard_normal(len(cfgs)))
    c2, c1, c0 = np.polyfit(np.log(Ns), Ls, 2)      # parabola in log N
    N_star.append(np.exp(-c1 / (2.0 * c2)))         # vertex = valley = optimal N
    print(f"C={C:.1e}: N*={N_star[-1]/1e6:6.2f}M  D*={C/flops_per_token(cfgs[3])['total']/1e9:.3f}B")

a_iso, _ = np.polyfit(np.log(C_slices), np.log(N_star), 1)   # N* ~ C^a
print(f"IsoFLOP allocation exponent  a = {a_iso:.3f}")
```

Run it across a few noise seeds and the vertices march monotonically upward — roughly 7–11M, 12–17M, 19–30M, 38–50M — giving $a$ in the range **0.44–0.57**. The *true* minima of this family under this law — computable exactly, since we know the ground truth — sit at 7.05M, 13.55M, 24.33M, 41.59M, i.e. $a = 0.538$. So the parabola method is both **noisy** (±0.06 across seeds from 1% loss noise alone) and, on five of six seeds, **biased low**. That is not our sloppiness: it is the documented pathology of Approach 2. Czech et al. (2026) show the parabolic approximation carries systematic bias even on *noise-free* data, driven by grid width, off-centre sampling, and the asymmetry of the loss surface when $\alpha \ne \beta$ — and that Approach 3 removes it if you exploit the objective's partially-linear structure (variable projection) instead of throwing a generic optimizer at it. Note also how much work `slice_configs` is doing: change `span` to 1.2 and the vertices move, because a wider grid makes the quadratic Taylor approximation worse. If you take one operational rule from this section, take that one — **centre the slice on the valley and keep the span tight.**

The practical reading: **run both methods and require them to agree on the story, not the digit.** Ours do — parametric $\beta/(\alpha+\beta) \approx 0.45$, IsoFLOP $a \approx 0.44$–$0.57$, true $0.538$; all of them say "grow $N$ and $D$ together, roughly one-for-one." That is the Chinchilla result, reproduced from scratch, with an honest error bar.

!!! note "Aside: why our exponent is ~0.54 and not Chinchilla's 0.50"
    Under a pure $6ND$ cost model, minimizing $L$ at fixed $C$ gives the textbook answer $N^\star \propto C^{\beta/(\alpha+\beta)}$ — here $0.30/0.63 = 0.476$. But our cost is not $6ND$. Write the real per-token cost as $c(N) \propto N^{\gamma}$: because the head term $6 d V$ and the attention term $6 L s d$ grow more slowly than $N$, we measure $\gamma \approx 0.65$ at the bottom of the ladder and $\gamma \approx 0.83$–$0.89$ around 100–400M, versus $\gamma = 1$ in the $6ND$ world. Substituting $D = C/c(N)$ into the law and setting the derivative to zero gives

    $$
    N^{\star} \;\propto\; C^{\,\beta/(\alpha + \gamma\beta)}.
    $$

    With $\gamma \approx 0.73$ over our budget range that is $0.30/(0.33 + 0.219) = 0.546$ — matching the 0.538 we measured directly. **Interpretation:** at small scale, a fixed-size vocabulary head is a compute tax that large models amortize and small models pay in full, so compute-optimal training pushes you toward *larger* models than Chinchilla's 0.5 would suggest. As $N$ grows and the blocks dominate, $\gamma \to 1$ and the exponent relaxes back to Chinchilla's. This is the same mechanism, viewed from the cost side, that Pearce & Song identify on the parameter-counting side — and it is invisible if you cost your sweep with $6ND$.

??? note "Optional: Chinchilla's third method (the loss envelope) and why we skip it"
    Hoffmann et al. actually used three approaches. Approach 1 is the **loss envelope**: train many models to convergence, plot every run's loss against its FLOPs, trace the lower-left frontier (the best loss achievable at each compute), and fit power laws to the *frontier* points' $N$ and $D$. It is the most data-hungry of the three — you need a dense scatter of runs whose convex hull is well-populated — and at a 17-run ladder the hull is defined by a handful of points, so its exponents are even noisier than the IsoFLOP parabolas. We omit it here for that reason, but the mental model is worth keeping: Approach 1 reads the frontier, Approach 2 reads the valleys at fixed compute, Approach 3 reads the whole surface. All three should agree on the allocation exponent; when they do not, distrust the extrapolation, not just one method.

---

## Extrapolating to Stack-100M and Predicting Its Loss

Now the payoff. We have a fitted law; we plug in the target's non-embedding count, $N=84.54\text{M}$, and read off the loss at the two allocations that matter — the compute-optimal one and the plan's over-trained one.

!!! example "Predicting Stack-100M's pretraining loss before spending the budget"
    **Setup.** Fitted law (constants above), target $N = 84.54\text{M}$ non-embedding params, $s = 2048$, $V = 32768$.

    **Chinchilla-style compute-optimal point (~20 tok/param).** $D = 20N \approx 1.69\text{B}$ tokens.
    $$
    L \approx E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}} \;\approx\; 3.15\ \text{nats/token} \quad (\pm 0.1).
    $$
    This is the ballpark the [scaling-laws chapter's](../03-pretraining/04-scaling-laws.html) loss-milestone table gives for a ~100M model at ~20 tok/param (~3.2–3.4 nats on web text; a touch lower here because the educational mix is cleaner).

    **The plan's over-trained point (~200 tok/param).** $D \approx 16.9\text{B}$ tokens (the ~20B-token budget).
    $$
    L \approx 2.95\ \text{nats/token} \quad (\pm 0.1\text{ from the fit's absolute-level uncertainty}).
    $$

    **What this buys you.** Over-training the *same* 100M model from 1.69B to 16.9B tokens — a **10× increase in training compute** — moves the loss from ~3.15 to ~2.95, a gain of roughly **0.20 nats**. At this scale that is a large, real improvement in text quality (the difference between a model that frequently loses the thread and one that mostly holds it), and this *difference* is far better pinned than either endpoint because it depends on the exponents and not on the loose offset. Critically, **the run you are about to launch should land at ≈ 2.9–3.0 nats/token held-out.** If your live curve is projecting somewhere 0.3+ nats away, stop and debug — that is the entire reason you built the ladder, and the next subsection makes "projecting" concrete.

    **Cross-check with the FLOP budget.** The flagship run costs $C = 7.97\times10^{8} \times 1.691\times10^{10} \approx 1.35\times10^{19}$ FLOPs. Ask the fitted law what the *compute-optimal* configuration for that same budget would be — searching over real deep-and-thin configs with $D$ forced by the full FLOP model — and it prefers $d_{\text{model}}=768$, 40 layers, $N^\star \approx 250\text{M}$, $D^\star \approx 6.65\text{B}$ (~27 tok/param). So for the very same compute, a compute-optimal model would be **~3× larger** than the one we are choosing to build. Which raises the obvious question the next section answers: *why are we deliberately building the smaller, over-trained model?*

Use the book's interactive Scaling-Law-Optimal tool to poke at these tradeoffs — drop in the fitted $E,A,B,\alpha,\beta$ and slide the compute budget to watch $(N^\star, D^\star)$ move:

{{tool:scaling-law-optimal}}

### Using the ladder as a live monitor

The chapter opened with a promise: catch a mismatch on hour two. Delivering it requires one more step, because **the fitted law predicts the final, LR-decayed loss, while at hour two your run is mid-stable-phase at full LR and therefore sitting well above its eventual value.** Comparing a live number to a final prediction will make every healthy run look broken. Two ingredients fix it, and you already paid for both when you saved the ladder's loss curves.

**1. Measure your own WSD decay drop.** The anneal at the end of a WSD run buys a characteristic extra drop (MiniCPM's headline observation; see [Chapter 14.6](../14-capstone/06-optimizer-and-schedule.html)). Its size depends on your schedule shape and data, so *measure* it across rungs rather than importing a number:

```python
# stacklm/scaling/monitor.py
import numpy as np

def measure_decay_drop(curves, decay_frac=0.20) -> float:
    """Median (loss at start of decay) - (final loss) across the ladder rungs.
    `curves` is a list of (tokens, val_loss) arrays from ladder_results.jsonl.
    At 100M scale expect this to be on the order of a tenth of a nat -- but the
    whole point is that YOUR ladder tells you, and it costs nothing extra."""
    drops = []
    for tok, loss in curves:
        tok, loss = np.asarray(tok, float), np.asarray(loss, float)
        i = int(np.searchsorted(tok, tok[-1] * (1.0 - decay_frac)))
        drops.append(loss[i] - loss[-1])
    return float(np.median(drops))
```

**2. Project the live stable-phase curve to its endpoint.** Within the stable phase the loss falls like the law's data term, so it is *linear in* $D^{-\beta}$ with the $\beta$ your ladder just measured. Fit a straight line to the last stretch of the live curve in that coordinate, extrapolate to the full token budget, then subtract the decay drop:

```python
def project_final_loss(tokens, losses, total_tokens, beta, decay_drop, warmup_skip=0.25):
    """Project a live STABLE-PHASE curve to the run's final, decayed loss.

    tokens, losses : the live run's logged curve so far (nats/token, held-out).
    beta           : the data exponent from YOUR ladder fit.
    decay_drop     : measure_decay_drop() over the ladder curves.
    warmup_skip    : drop the first fraction of points; the warmup transient
                     is not on the power law and will drag the line."""
    t, l = np.asarray(tokens, float), np.asarray(losses, float)
    keep = t > t[-1] * warmup_skip
    x = t[keep] ** (-beta)                        # law is linear in D^-beta ...
    slope, intercept = np.polyfit(x, l[keep], 1)  # ... so a straight line fits
    l_stable_end = intercept + slope * total_tokens ** (-beta)
    return l_stable_end - decay_drop

# Hour two of the flagship: ~1.2B tokens in (4% of the run).
projected = project_final_loss(live_tokens, live_losses,
                               total_tokens=1.691e10, beta=beta,
                               decay_drop=measure_decay_drop(ladder_curves))
print(f"projected final loss {projected:.3f}  vs  ladder prediction 2.95 +/- 0.10")
```

**3. Compare *shapes*, not just endpoints.** Overlay the S4 rung's curve on the flagship's with **tokens-per-parameter** on the x-axis. Under a frozen recipe the two curves should lie almost on top of each other over their shared range; a healthy flagship simply continues where S4 stopped. A *shape* divergence — a kink, a plateau, a different slope — localizes the bug in time far better than a single scalar does, and it shows up in the first hour.

When the check fails, triage before you restart:

| What you see at hour two | Most likely cause | Where to look |
|---|---|---|
| Loss pinned near $\ln 32768 = 10.40$ | Model is learning nothing: labels fully masked, targets not shifted by one, or LR effectively 0 | Ch. 14.7 loss/label plumbing |
| Smooth curve, projects **0.3–0.8 nats high** | Tokenizer mismatch (data tokenized with a different vocab), or val shard drawn from a different mixture than the ladder's | Ch. 14.3, Ch. 14.2 manifest |
| Projects **below** the prediction | Too good to be true: val contamination, or documents leaking across a pack boundary so the model sees val text in train | Ch. 14.2 dedup + doc-aware masking |
| High **and noisy**, frequent spikes | LR too high for this width (check the muP scaling), or QK-norm/QK-clip disabled | Ch. 14.6, [Training Stability](../03-pretraining/11-training-stability.html) |
| Fine, then diverges | Attention-logit blow-up under Muon — the failure MuonClip/QK-clip exists to stop | Ch. 14.6 |
| On-prediction but 3× slow | Not a scaling-law problem at all: an MFU/throughput problem | Ch. 14.7 |

The instrument you built to *choose* the run is the same instrument that *supervises* it. That is the whole return on the 14%.

---

## Second Use of the Ladder: Choosing the Data Mixture

The plan fixes an approximate mixture — **70% FineWeb-Edu, 15% Cosmopedia v2, 10% code (StarCoder), 5% math (FineMath/OpenWebMath)** — and explicitly delegates the tuning of those weights to this chapter. A ladder is exactly the right instrument, because the *same* frozen-recipe discipline that lets you vary $(N, D)$ lets you vary the mixture instead: **fix $N$ and $D$, vary the weights, rank by held-out loss.** This is the small-proxy-model methodology behind DoReMi (Xie et al., 2023), RegMix (Liu et al., 2024), and Aioli (Chen et al., 2024), and it is the cheapest lever on final quality you have.

Two design rules make the ranking mean something.

**Rule 1: hold the evaluation set fixed and matched to your deployment target.** If you score each candidate on a validation set drawn from *its own* mixture, you are measuring "how predictable is this mix by a model trained on it," which rewards low-entropy mixes (crank the code fraction and watch aggregate loss fall). Instead, fix one validation set whose composition reflects what the model must actually do, and *also* report per-domain losses so you can see the trade you are making.

**Rule 2: screen cheap, confirm one rung up.** Ai2's **DataDecide** (Magnusson et al., ICML 2025) is the definitive study here: 25 corpora × 14 model sizes from 4M to 1B, over 30k checkpoints. Its headline finding is both encouraging and cautionary — ranking candidate corpora at a *single small size* predicts the best corpus at 1B in roughly 80% of pairwise comparisons, using ~0.01% of the compute. Eighty percent is a wonderful screen and a terrible oracle. So: screen at S1, confirm the finalists at S2, and never let a mixture decision rest on one 4M-parameter run.

```python
# stacklm/scaling/mixture.py -- second use of the ladder: rank data mixtures.
from stacklm.scaling.ladder import LADDER
from stacklm.scaling.flops import training_flops, gpu_hours
from stacklm.scaling.sweep import batch_tokens

BY = {c.name: c for c in LADDER}

# Candidate weights over (fineweb_edu, cosmopedia, code, math). The plan's
# default is first; the others probe one axis at a time so a win is attributable.
CANDIDATES = {
    "plan":        (0.70, 0.15, 0.10, 0.05),   # PLAN sec.2 default
    "more_synth":  (0.55, 0.30, 0.10, 0.05),   # is Cosmopedia worth more weight?
    "less_synth":  (0.82, 0.03, 0.10, 0.05),   # ...or is it hurting diversity?
    "more_code":   (0.60, 0.15, 0.20, 0.05),   # code as a reasoning/structure prior
    "more_math":   (0.60, 0.15, 0.10, 0.15),   # math for the downstream agent
    "web_only":    (1.00, 0.00, 0.00, 0.00),   # the null hypothesis -- always include it
}

# A FIXED validation set, composed for the deployment target, NOT per-candidate.
# Ch. 14.10's agent needs grounded QA over clean prose plus a little arithmetic.
VAL_WEIGHTS = {"fineweb_edu": 0.55, "cosmopedia": 0.20, "code": 0.10, "math": 0.15}

def screen(rung="S1", tokens_per_param=100, data_root=None):
    """Train one model per candidate at a FIXED (N, D) and score them all on the
    SAME held-out set. Returns rows sorted by the target-weighted loss."""
    from stacklm.data import build_mixture          # Ch. 14.2 shard sampler
    from stacklm.train import train_run
    from stacklm.model import StackConfig, Stack100M
    from stacklm.eval import per_domain_loss        # Ch. 14.11 eval harness

    cfg = BY[rung]
    D = int(tokens_per_param * cfg.nonembed_params())
    bt = batch_tokens(D)
    rows = []
    for name, w in CANDIDATES.items():
        data = build_mixture(data_root, weights=w, seq_len=cfg.seq_len, seed=1234)
        model = Stack100M(StackConfig(
            vocab_size=cfg.vocab_size, d_model=cfg.d_model, n_layers=cfg.n_layers,
            n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads, head_dim=cfg.head_dim,
            intermediate=cfg.intermediate, tie_embeddings=True,
            nope_every=4, qk_norm=True))
        out = train_run(model, data, total_tokens=D, batch_tokens=bt,
                        optimizer="muon+adamw", lr_muon=0.02,
                        lr_adamw=3e-3 * 512 / cfg.d_model,
                        schedule="wsd", warmup_steps=min(2000, max(50, (D//bt)//20)),
                        decay_frac=0.20, seed=1234)          # identical for all mixes
        per_dom = per_domain_loss(out.model, split="val")    # dict domain -> nats
        score = sum(VAL_WEIGHTS[k] * per_dom[k] for k in VAL_WEIGHTS)
        rows.append(dict(name=name, weights=w, score=score, per_domain=per_dom))
    return sorted(rows, key=lambda r: r["score"])

if __name__ == "__main__":
    cfg = BY["S1"]; D = 100 * cfg.nonembed_params()
    screen_flops = len(CANDIDATES) * training_flops(cfg, D)
    cfg2 = BY["S2"]; D2 = 60 * cfg2.nonembed_params()
    confirm_flops = 2 * training_flops(cfg2, D2)             # top-2 finalists at S2
    total = screen_flops + confirm_flops
    print(f"screen {screen_flops:.2e} + confirm {confirm_flops:.2e} = {total:.2e} FLOPs")
    print(f"= {gpu_hours(total):.2f} GPU-hr ~ USD {gpu_hours(total)*1.5:.2f}")
    # -> 2.00e17 + 1.61e17 = 3.61e17 FLOPs = 0.80 GPU-hr ~ USD 1.21
    #    i.e. 2.7% of the flagship run to make the mixture decision on evidence.
```

**Six candidates at S1 (393M tokens each, 3,000 steps) plus two confirmations at S2 (549M tokens, ~2,100 steps) cost $3.6\times10^{17}$ FLOPs — 2.7% of the flagship, about 0.8 GPU-hours, roughly USD 1.20.** That is the entire price of replacing "70/15/10/5 felt right" with a measurement.

Three practical notes. **Build the candidate mixes with real tooling**: HuggingFace `datatrove` handles the streaming, filtering, dedup and shard-writing pipeline of [Chapter 14.2](../14-capstone/02-data-pipeline.html), and re-weighting is a sampler config, not a re-download. **Always include the null hypothesis** (`web_only`) — if the fancy mix does not beat 100% FineWeb-Edu on your fixed validation set, you have learned something valuable and cheap. And **read the per-domain columns, not just the score**: a mix that wins overall while doubling your math loss is the wrong mix for a capstone whose downstream agent must do arithmetic. If you want to go further than static weights, the online methods (DoReMi's group-DRO proxy, Aioli's on-the-fly mixing-law estimation) and the broader curriculum question are covered in [Data Mixing, Domain Weighting & Curriculum](../03-pretraining/14-data-mixing-curriculum.html).

---

## The Over-Training Decision: Compute-Optimal ≠ Deployment-Optimal

Chinchilla answers "what minimizes *training* loss for a *training*-compute budget." That is almost never the real objective. `Stack-100M` will be **trained once and served indefinitely** — quantized to int4 and run on a laptop (Ch. 14.11). For a model you deploy, the quantity to minimize is **lifetime compute**, training *plus* all future inference:

$$
C_{\text{lifetime}} = \underbrace{c(N)\, D_{\text{train}}}_{\text{train once}} \;+\; \underbrace{2 N D_{\text{infer}}}_{\text{serve forever}}
$$

subject to hitting a **target loss** $L^\star$, where $c(N)$ is the full per-token training cost from the table above. Every generated token costs $\approx 2N$ FLOPs and the KV cache scales with $N$ (see [The Anatomy of LLM Inference](../07-inference-serving/01-anatomy-inference.html) and [Inference Economics](../07-inference-serving/12-inference-economics.html)). So the larger $D_{\text{infer}}$ is, the more it pays to **shrink $N$** (cheap inference forever) and **grow $D_{\text{train}}$** to buy back the loss you gave up. This is the inference-aware regime formalized by Sardana, Frankle et al. (*Beyond Chinchilla-Optimal*, 2024), and it is why Llama-3-8B saw ~15T tokens (~1900 tok/param), two orders of magnitude past Chinchilla.

Here is the decision made concrete with the ladder's own numbers — and note that the corrected FLOP accounting makes the case *stronger*, not weaker, because it pushes the compute-optimal model up to 250M:

!!! example "Two ways to spend the same 1.35e19 FLOPs"
    | | config | model $N$ | tokens $D$ | tok/param | predicted loss | inference cost/token |
    |---|---|---|---|---|---|---|
    | **compute-optimal** | $d$=768, $L$=40 | 249.7M | 6.65B | ~27 | **~2.92** | $2 \times 249.7\text{M}$ |
    | **our choice** | $d$=512, $L$=30 | 84.5M | 16.91B | 200 | **~2.95** | $2 \times 84.5\text{M}$ |

    Same training budget ($1.35\times10^{19}$ FLOPs under the full cost model, by construction). The over-trained 84.5M model is **only ~0.03 nats/token worse** in loss — a difference you would struggle to detect in generated text — yet it is **less than a third of the size**: every future forward pass costs $84.5/249.7 \approx 0.34\times$ as much, a **~66% permanent cut** to inference FLOPs, latency, activation memory, and KV-cache footprint. You pay the extra training tokens **once**; you collect the inference savings on **every request for the life of the model**. For a model destined to run quantized on a laptop, that trade is not close. (The absolute loss *levels* carry the fit's ±0.1-nat uncertainty; the **~0.03-nat penalty and the 0.34× size ratio do not** — they are set by the exponents and the fixed FLOP constraint, so they hold whether the true floor is 2.9 or 3.0.)

    Put differently: compute-optimal is the right target if you will train a model and (nearly) never run it. The instant you plan to *serve* it at any scale, you should slide down the size axis and over-train — which is precisely the ~200 tok/param, ~20B-token budget fixed in the [capstone overview](../14-capstone/01-overview-and-landscape.html).

{{fig:overtrain-vs-compute-optimal-headtohead}}

```python
import numpy as np
from stacklm.scaling.flops import flops_per_token

def lifetime_optimal(L_target, D_infer, E, A, alpha, B, beta):
    """Pick a REAL config (and its D_train) to hit L_target while minimizing
    TRAIN+INFER FLOPs. We search the deep-and-thin family so the training cost
    is the honest c(N) -- blocks + attention + head -- not 6ND."""
    best = None
    for d in range(128, 2049, 64):
        cfg = family(d)                              # from the IsoFLOP section
        N   = cfg.nonembed_params()
        budget = L_target - E - A * N**(-alpha)      # loss left for the data term
        if budget <= 0:
            continue                                 # this N alone overshoots L_target
        D_train = (B / budget) ** (1.0 / beta)
        total = flops_per_token(cfg)["total"] * D_train + 2 * N * D_infer
        if best is None or total < best["total"]:
            best = dict(d=d, N=N, D_train=D_train, total=total, tpp=D_train/N)
    return best

E, A, alpha, B, beta = 2.45, 124.0, 0.33, 234.0, 0.30    # the fitted (here, GT) law
for D_infer in (1e10, 1e12, 1e14):                       # light -> heavy serving
    r = lifetime_optimal(2.95, D_infer, E, A, alpha, B, beta)
    print(f"D_infer={D_infer:.0e}: d={r['d']:4d}  N*={r['N']/1e6:6.1f}M  "
          f"D_train={r['D_train']/1e9:8.2f}B  tok/param={r['tpp']:6.0f}")
# D_infer=1e+10: d= 640  N*= 146.0M  D_train=    8.10B  tok/param=    55
# D_infer=1e+12: d= 448  N*=  49.5M  D_train=   53.31B  tok/param=  1078
# D_infer=1e+14: d= 384  N*=  31.5M  D_train=  305.11B  tok/param=  9699
# As planned serving load rises, the optimal model SHRINKS and tok/param CLIMBS
# -- the quantitative engine behind "over-train a small model for deployment."
```

One honest caveat before you crank tokens/param to the moon: the Chinchilla form assumes every token is *fresh*. High-quality data is finite, and past a few epochs of repetition returns collapse (Muennighoff et al.'s data-constrained law; see [scaling laws](../03-pretraining/04-scaling-laws.html) and [Data Cleaning & Dedup](../03-pretraining/02-data-cleaning-dedup.html)). At 16.9B tokens on our ~20B-token deduplicated mix we are inside one epoch, so the fresh-token law holds — but the 1078- and 9699-tok/param rows above are *fantasy* at our corpus size, and that is exactly why the plan invests so heavily in dedup and synthetic Cosmopedia data rather than simply looping the corpus.

!!! interview "Interview Corner"
    **Q:** You fit a scaling law from a ladder of tiny models and it predicts a 100M model will reach 2.95 nats at 200 tokens/param and 3.15 at 20. Your manager asks why you would ever train past the compute-optimal point — isn't that wasting compute?

    **A:** It wastes *training* compute but saves *lifetime* compute, and lifetime is what we pay. Compute-optimal (Chinchilla) minimizes training loss for a training-FLOP budget — the right objective only if you train a model and never serve it. Our model is trained once and served indefinitely, so the objective is $c(N)D_{\text{train}} + 2ND_{\text{infer}}$ subject to a target loss. Because inference cost scales with $N$, the optimum shifts toward a *smaller* model trained on *more* tokens. Concretely, for our fixed $1.35\times10^{19}$-FLOP budget the compute-optimal model is ~250M at ~6.6B tokens; we instead train ~85M at ~17B tokens. We give up ~0.03 nats of loss — imperceptible — to make every future forward pass ~66% cheaper, permanently.

    Two things I'd want to say before anyone quotes those numbers back at me. First, that budget is $1.35\times10^{19}$, not the $8.6\times10^{18}$ you get from $6ND$: at 100M scale the tied vocabulary head and causal attention are 36% of the real FLOPs, and the correction is *rung-dependent* — 3.6× at 4M, 1.6× at 100M — so a ladder costed with $6ND$ has slices that are not iso-FLOP and vertices biased toward large $N$. Second, the penalty and the size ratio are robust even though our tiny-ladder fit only pins the *absolute* loss to ±0.1 nats, because they depend on the exponents and the FLOP constraint, not the loose offset. Two guardrails on the over-training: keep total tokens within a few epochs of unique data (repetition returns collapse past ~4–16 epochs), and remember the law predicts *loss*, not *capabilities* — downstream abilities are threshold-y and should be validated with smooth proxy metrics, not extrapolated from the loss curve.

---

## Key Takeaways

!!! key "Key Takeaways"
    - **Fit your own law; don't import constants.** $E$, $A$, $B$, $\alpha$, $\beta$ depend on your corpus, tokenizer, and frozen recipe. A four-rung ladder (~4M/9M/19M/44M) under the *identical* Stack recipe costs ~4 GPU-hours (~USD 6, ~14% of the flagship) and de-risks the whole 20B-token commitment.
    - **$6ND$ is a 40%-error approximation at 100M and a 260%-error one at 4M.** Cost your sweep with $6N_{\text{nonembed}} + 6 L s\, d + 6 d V$ per token — blocks + causal attention + tied head. The correction is *rung-dependent* (3.60× → 1.57× up our ladder), so $6ND$ "IsoFLOP" slices are not iso-compute and their vertices are biased.
    - **Say which parameter count you fit, and make compute consistent with it.** Kaplan used non-embedding $N$; Chinchilla used total $N$; Pearce & Song (2024) show that mismatch is a primary driver of the $C^{0.73}$ vs $C^{0.50}$ discrepancy, worst below 1B. We fit $N_{\text{nonembed}}$ *and* count the head's FLOPs explicitly.
    - **Freeze everything but $N$ and $D$ — except LR and batch, which must move.** Give every rung a schedule that decays to zero at *its own* token count (the Kaplan confound), scale the AdamW group's LR as $1/d_{\text{model}}$ (muP; Muon's RMS-matched update is near width-invariant), and size the batch for $\ge$ ~2000 steps and $\le$ the critical batch size.
    - **Fit in log space with `logsumexp` + Huber loss**, constrain the exponents to the sane ~0.25–0.42 band, multi-start the non-convex objective, and judge the fit by *held-out extrapolation*, not the raw constants — $E$, $A$, $B$ are weakly identified (our $A$ ranged 58–430 against a true 124) while the extrapolated loss lands within ~0.1 nats and a held-out top rung within ~0.05.
    - **Two methods, one story.** Parametric $\beta/(\alpha+\beta) \approx 0.45$ and IsoFLOP $a \approx 0.37$–$0.54$ both say "grow $N$ and $D$ roughly together." IsoFLOP parabolas are noisy *and* biased low at this density (Czech et al., 2026); require agreement on the story, not the digit. Under the real cost model the exponent is $\beta/(\alpha+\gamma\beta) \approx 0.54$, not 0.48, because a fixed vocab head taxes small models hardest.
    - **Predicted Stack-100M loss: ~3.15 nats (±0.1) at ~20 tok/param, ~2.95 at the over-trained 200 tok/param (16.9B tokens).** Project your live stable-phase curve in $D^{-\beta}$ coordinates, subtract the ladder's measured WSD decay drop, and compare; a >0.3-nat miss means a bug, and the triage table tells you which one.
    - **The ladder also picks your data mixture.** Six candidate mixes at S1 plus two confirmations at S2 cost ~0.8 GPU-hr (~2.7% of the flagship). Score on a *fixed* target-domain validation set with per-domain breakdowns, include a web-only null hypothesis, and confirm one rung up — DataDecide (2025) shows single-small-size rankings transfer only ~80% of the time.
    - **Compute-optimal ≠ deployment-optimal.** For the same $1.35\times10^{19}$ FLOPs, compute-optimal wants ~250M params; we deliberately build ~85M and over-train, trading ~0.03 nats for a ~66% permanent cut in inference cost — the deployment economics behind the plan's 200-tok/param budget.
    - **The law predicts loss, not capabilities.** Extrapolate loss with confidence; treat capability thresholds as uncertain and stay within a few epochs of unique data.

!!! sota "State of the Art & Resources (2026)"
    The Chinchilla parametric/IsoFLOP machinery this chapter miniaturizes is still the working standard, but 2024–2026 work has formalized the deployment-aware extension this chapter builds toward, explained *why* Kaplan and Chinchilla disagreed, and exposed real fragility in the fitting methodology itself — worth knowing before you trust any single fit, including your own.

    **Foundational work**

    - [Kaplan, McCandlish, Henighan, et al., *Scaling Laws for Neural Language Models* (2020)](https://arxiv.org/abs/2001.08361) — the original power-law form for loss vs. $N$, $D$, $C$, the non-embedding parameter convention, and the LR-schedule confound this chapter's checklist is built to avoid.
    - [Hoffmann, Borgeaud, Mensch, et al., *Training Compute-Optimal Large Language Models* (Chinchilla, 2022)](https://arxiv.org/abs/2203.15556) — the three fitting approaches (loss envelope, IsoFLOP, parametric $L(N,D)$) this chapter's ladder reproduces at 1/1000th the scale, fit on *total* parameters.
    - [Yang, Hu, Babuschkin, et al., *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer* (muP, 2022)](https://arxiv.org/abs/2203.03466) — why the optimal LR moves with width and how to make it stop; the `mup` package is the reference PyTorch implementation.
    - [McCandlish, Kaplan, Amodei, et al., *An Empirical Model of Large-Batch Training* (2018)](https://arxiv.org/abs/1812.06162) — the critical-batch-size / gradient-noise-scale framework behind this chapter's per-run batch rule.

    **Recent advances (2023–2026)**

    - [Pearce & Song, *Reconciling Kaplan and Chinchilla Scaling Laws* (TMLR 2024)](https://arxiv.org/abs/2406.12907) — shows that counting non-embedding rather than total parameters, at small scale, largely explains Kaplan's $C^{0.73}$; the single most important paper for anyone fitting laws below 1B.
    - [Sardana, Portes, Doubov & Frankle, *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws* (2024)](https://arxiv.org/abs/2401.00448) — formalizes the lifetime-compute (train + serve) objective behind this chapter's over-training decision.
    - [Zhang, Morwani, Vyas, et al., *How Does Critical Batch Size Scale in Pre-training?* (ICLR 2025)](https://arxiv.org/abs/2410.21676) — critical batch size grows with *data* size and only weakly with model size; the basis for scaling batch tokens per run instead of fixing them.
    - [Magnusson, Tai, et al., *DataDecide: How to Predict Best Pretraining Data with Small Experiments* (Ai2, ICML 2025)](https://arxiv.org/abs/2504.11393) — 25 corpora × 14 sizes (4M–1B), 30k checkpoints; small-scale rankings predict the 1B winner ~80% of the time, the caveat that shapes this chapter's screen-then-confirm mixture protocol.
    - [Muennighoff, Rush, Barak, et al., *Scaling Data-Constrained Language Models* (2023)](https://arxiv.org/abs/2305.16264) — the repetition/data-wall correction that bounds how far over-training can go before epoching returns collapse.
    - [Besiroglu, Erdil, Barnett & You, *Chinchilla Scaling: A Replication Attempt* (2024)](https://arxiv.org/abs/2404.10102) — reconstructs Hoffmann et al.'s data and shows the original parametric constants fit poorly and carry implausibly narrow confidence intervals, the empirical basis for this chapter's "trust the extrapolated loss, not the raw constants" rule.
    - [Czech, Xu, Elmatad, Wang & Held, *Problems with Chinchilla Approach 2: Systematic Biases in IsoFLOP Parabola Fits* (2026)](https://arxiv.org/abs/2603.22339) — the parabola-vertex method is biased even on noise-free data (grid width, off-centre sampling, $\alpha \ne \beta$ asymmetry); Approach 3 fixes it if you exploit the objective's partially-linear structure.
    - [Liu, Su, et al., *Muon is Scalable for LLM Training* (Moonshot AI, 2025)](https://arxiv.org/abs/2502.16982) — the $0.2\sqrt{\max(m,n)}$ update-RMS matching that lets one LR govern the Muon and AdamW groups, and makes Muon's LR far more width-portable than Adam's.

    **Open-source & tools**

    - [Open-Athena/vpnls](https://github.com/Open-Athena/vpnls) — fits $L=E+A/N^\alpha+B/D^\beta$ by **variable projection**, reducing the nonlinear search to the two exponents so a dense grid search is tractable; Cython, SciPy L-BFGS-B and JAX backends, MSE and Huber losses. The production answer to the multi-start fragility of the `scipy` fit above.
    - [apple/ml-scalefit](https://github.com/apple/ml-scalefit) — a JAX package for fitting parametric scaling laws with basin-hopping optimization and bootstrapped uncertainty, shipping example data from the Chinchilla and data-mixture-law papers.
    - [microsoft/mup](https://github.com/microsoft/mup) — `pip install mup`; records base shapes from a reference model and rescales optimizer groups so one tuned LR transfers across the ladder.
    - [huggingface/datatrove](https://github.com/huggingface/datatrove) — the streaming filter/dedup/tokenize/shard pipeline used to build the candidate data mixtures for the mixture screen (and the FineWeb corpora themselves).
    - [shehper/scaling_laws](https://github.com/shehper/scaling_laws) — an open, from-scratch reproduction of Kaplan-style scaling laws on nanoGPT at sub-10M-parameter scale, the same "prove it on tiny models first" spirit as this chapter's ladder.

    **Go deeper**

    - [Epoch AI, *Chinchilla Scaling: A Replication Attempt*](https://epoch.ai/publications/chinchilla-scaling-a-replication-attempt) — the visual, blog-form companion to the Besiroglu et al. paper, with the reconstructed data and refit curves plotted out.
    - [Open Athena, *Problems with Chinchilla Approach 2*](https://openathena.ai/blog/problems-with-chinchilla-approach-2/) — the blog companion to the 2026 paper, with the IsoFLOP experiment data released on HuggingFace.

## Further Reading

- Hoffmann, Borgeaud, Mensch, et al. *Training Compute-Optimal Large Language Models* (Chinchilla). arXiv 2022. — The three fitting methods (loss-envelope, IsoFLOP, parametric) this chapter miniaturizes.
- Kaplan, McCandlish, Henighan, et al. *Scaling Laws for Neural Language Models*. arXiv 2020. — The original power-law form and the LR-schedule confound to avoid.
- Pearce & Song. *Reconciling Kaplan and Chinchilla Scaling Laws*. TMLR 2024. — Why the parameter-counting convention, at small scale, is the primary source of the two papers' disagreement.
- Yang, Hu, et al. *Tensor Programs V: Zero-Shot Hyperparameter Transfer* (muP). arXiv 2022. — The width-scaling rules behind this chapter's per-rung learning rates.
- Sardana, Frankle, et al. *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws*. arXiv 2024. — Formalizes lifetime-compute and inference-aware over-training.
- Besiroglu, Erdil, Barnett, et al. *Chinchilla Scaling: A Replication Attempt* (Epoch AI). 2024. — Why the parametric constants are fragile and the extrapolated loss is the thing to trust.
- Muennighoff, Rush, Barak, et al. *Scaling Data-Constrained Language Models*. NeurIPS 2023. — The repetition/data-wall limit on how far you can over-train.
- Xie, Pham, et al. *DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining*. NeurIPS 2023. — Proxy-model domain reweighting, the ancestor of this chapter's mixture screen.
- Liu, Zheng, et al. *RegMix: Data Mixture as Regression for Language Model Pre-training*. ICLR 2025. — Many small runs plus a regression surface to predict the best unseen mixture.
- Magnusson, Tai, et al. *DataDecide: How to Predict Best Pretraining Data with Small Experiments*. ICML 2025. — What does and does not transfer from 4M-parameter proxy runs.
- Liu, Chang, et al. *MobileLLM: Optimizing Sub-billion Parameter Language Models*. 2024. — The deep-and-thin small-model result the ladder inherits.

## Exercises

**1.** *(Conceptual.)* This chapter fits on **non-embedding** parameters; Chinchilla fit on **total** parameters. Suppose a colleague fits $L(N_{\text{total}}, D) = E + A/N_{\text{total}}^{\alpha} + B/D^{\beta}$ on our 17 ladder points (blocks + the tied $32768 \times d$ embedding). Explain concretely why this rotates the fitted exponent $\alpha$, why the problem is *specific to small models*, and what you would have to change about the compute variable to make the total-parameter convention self-consistent.

??? note "Solution"
    The additive term $A/N^{\alpha}$ is meant to capture the *depth-driven, sequence-mixing capacity* of the transformer blocks. The embedding table does a fixed amount of lookup work per token that does **not** scale with depth, so folding it into $N$ mixes two quantities that grow at different rates as you climb the ladder.

    Why it hits *small* models specifically: the embedding size is $32768 \cdot d$, growing only linearly in width $d$, while the block parameters grow like $n_{\text{layers}} \cdot d^2$ — and depth grows with the ladder too. The embedding's share of $N_{\text{total}}$ therefore falls monotonically:

    - S1: blocks $3.93\text{M}$, embedding $32768 \times 192 = 6.29\text{M}$ — the embedding is **larger than the entire rest of the model** ($\approx 62\%$ of total params).
    - S2: $48\%$. S3: $36\%$. S4: $25\%$.
    - Target: blocks $84.54\text{M}$, embedding $16.78\text{M}$ — only $\approx 17\%$.

    That shrinking near-constant baseline flattens the low-$N$ end of the curve — the small rungs look "less penalized than their block count deserves" — and the fitter compensates by rotating $\alpha$ upward. In the chapter's refit, the total-parameter convention pushed $\alpha$ to a median of ~0.59 (railing at the 0.60 bound in four of six seeds) versus ~0.43 for non-embedding. Scaled up, this is precisely the mechanism Pearce & Song (2024) identify behind Kaplan's $N^\star \propto C^{0.73}$ versus Chinchilla's $C^{0.50}$. At frontier scale the embedding is a rounding error at every rung, so the confound largely vanishes — which is why the frontier literature can gloss over it and small-model builders cannot.

    **To make the total-parameter convention self-consistent** you must keep the compute variable matched to the capacity variable. Fitting $N_{\text{total}}$ while costing runs at $6 N_{\text{nonembed}} D$ is the worst of both worlds. Use the full per-token cost either way — $6N_{\text{nonembed}} + 6 L s\, d + 6 d V$ — and simply be explicit about which $N$ appears in the *law*. Then a reader can convert: at the target, $N_{\text{total}} = 1.20\, N_{\text{nonembed}}$; at S1, $2.60\times$.

**2.** *(Conceptual.)* The warning admonition says every rung must get "its **own** schedule that decays to zero at *its* token budget $D$." Suppose instead you train S4 **once** for its largest sweep budget (0.578B tokens, its $2.7\times10^{17}$-FLOP IsoFLOP point) on a single WSD schedule, and to obtain S4's loss at 0.2B and 0.4B tokens you simply read the value off that one curve at those step counts. Why are those numbers biased, what is the name of the historical mistake this reproduces, and how does the bias interact with the batch-size rule?

??? note "Solution"
    A WSD (or cosine) schedule sized for 0.578B tokens is still at **full learning rate** at the 0.2B and 0.4B marks — its decay does not begin until 80% of the way through. A model evaluated mid-stable-phase, before the LR has annealed, sits at a noisier, higher point on its loss surface: it "evaluates *worse than it truly is*." So the 0.2B and 0.4B losses you read off are **inflated** relative to what runs that *decayed to zero at 0.2B* (respectively 0.4B) would actually reach.

    The damage is that this bias is not uniform across your grid — it corrupts intermediate-budget reads while leaving the properly-decayed endpoint alone — so it silently rotates the whole fitted surface and biases the allocation toward "make the model bigger." This is precisely the **Kaplan confound**: Kaplan et al.'s original scaling study reused schedules that were not matched to each run's token count, which is why their recommended allocation over-weighted parameters relative to tokens. Hoffmann et al. (Chinchilla) fixed it by giving every run a schedule matched to its own $D$ — the one-line discipline (never a shared `total_steps`) this chapter's checklist item 3 calls non-negotiable.

    **Interaction with the batch rule:** the bias compounds, because the two shortcuts fail in the same direction. A shared long schedule *and* a shared large batch both hurt the short, low-tokens-per-param runs hardest — mid-schedule reads are worst early, and a fixed 0.5M-token batch is furthest above the critical batch size when $D$ is smallest. Those runs are exactly the off-diagonal points you added to pin $\beta$, so both errors land on the same exponent. The chapter's fix — per-run `steps`, `warmup_steps`, and `batch_tokens` derived from $D$ — removes both at once. (There is one legitimate use of a shared curve: the WSD *branching* trick of Ch. 14.6, where you fork multiple decay runs from one stable checkpoint. That is fine precisely because each branch gets its own full decay.)

**3.** *(Quantitative.)* Using the exact arithmetic in `LadderConfig.nonembed_params()` and `flops_per_token()`, compute rung **S2** ($d_{\text{model}}=256$, $n_{\text{layers}}=13$, `head_dim`$=64$, `n_kv_heads`$=1$, $s=2048$, $V=32768$) by hand: (a) its SwiGLU `intermediate` width, the per-block attention and MLP parameter counts, and the total non-embedding count; (b) its tied embedding and the embedding's share of total parameters; (c) the three FLOPs-per-token terms and the ratio of the total to $6N$.

??? note "Solution"
    **(a) Parameters.** $2.75 \times 256 = 704$; $704/64 = 11.0$, so `intermediate` $= 11 \times 64 = 704$.

    $$
    \text{attn} = 2 d^2 + 2 d (kv \cdot hd) = 2(256^2) + 2(256)(64) = 131072 + 32768 = 163840,
    $$
    $$
    \text{mlp} = 3 d \cdot \text{inter} = 3 \times 256 \times 704 = 540672.
    $$

    Per block $163840 + 540672 = 704512$; all 13 blocks: $N = 13 \times 704512 = 9{,}158{,}656 \approx 9.16\text{M}$. $\checkmark$

    **(b) Embedding.** $32768 \times 256 = 8{,}388{,}608 \approx 8.39\text{M}$. Total $= 9.16 + 8.39 = 17.55\text{M}$, so the embedding is $8.39/17.55 \approx \mathbf{48\%}$ of S2's parameters — a vivid illustration of why the parameter-counting convention matters at this scale.

    **(c) FLOPs per token.**
    $$
    \text{blocks} = 6N = 6 \times 9{,}158{,}656 = 5.495\times10^{7},
    $$
    $$
    \text{attention} = 6 \, n_{\text{layers}}\, s\, d = 6 \times 13 \times 2048 \times 256 = 4.089\times10^{7},
    $$
    $$
    \text{head} = 6\, d\, V = 6 \times 256 \times 32768 = 5.033\times10^{7}.
    $$

    Total $= 1.462\times10^{8}$ FLOPs/token, which is $1.462\times10^{8} / 5.495\times10^{7} = \mathbf{2.66\times}$ the $6ND$ figure. Note that the *head alone* costs slightly more than the entire block stack at this rung. A sweep costed with $6ND$ would under-budget every S2 run by a factor of 2.66 — and every S4 run by only 1.77, which is why the two rungs' "equal-$C$" runs are not equal at all.

**4.** *(Quantitative.)* Cost the **flagship** run two ways: `Stack-100M` ($d$=512, $L$=30, $N = 84.54\text{M}$ non-embedding, $s$=2048, $V$=32768) over-trained to 200 tokens/param. Compute (a) $D$; (b) $C$ under $6ND$ and under the full model; (c) single-A100 wall-clock at 40% and 50% MFU (peak 312 TFLOP/s); (d) the dollar cost at USD 1–2/GPU-hour. Reconcile your answer with the plan's "15 to 25 A100-hours."

??? note "Solution"
    **(a) Tokens.** $D = 200 \times 84.54\text{M} = 1.6908 \times 10^{10} \approx 16.9\text{B}$.

    **(b) FLOPs.** The $6ND$ rule gives
    $$
    C_{6ND} = 6 \times (8.454\times10^{7}) \times (1.6908\times10^{10}) = 8.58 \times 10^{18}.
    $$
    The full per-token cost is $6N + 6Lsd + 6dV = 5.072\times10^{8} + 1.887\times10^{8} + 1.007\times10^{8} = 7.966\times10^{8}$, so
    $$
    C_{\text{full}} = 7.966\times10^{8} \times 1.6908\times10^{10} = \mathbf{1.347 \times 10^{19}}\ \text{FLOPs},
    $$
    i.e. $1.57\times$ the $6ND$ figure. The blocks are only 64% of the real cost; attention is 24% and the tied head 13%.

    **(c) Wall-clock.** At 40% MFU the effective throughput is $312\times10^{12} \times 0.40 = 1.248\times10^{14}$ FLOP/s, so
    $$
    t = \frac{1.347\times10^{19}}{1.248\times10^{14}} = 1.079\times10^{5}\ \text{s} \approx \mathbf{30.0\ \text{GPU-hours}}.
    $$
    At 50% MFU, $t \approx 24.0$ GPU-hours.

    **(d) Dollars.** 24–30 GPU-hr at USD 1–2/hr is **USD 24–60**.

    **Reconciliation.** The plan's "15–25 A100-hours" band is the $6ND$ number: $8.58\times10^{18} / 1.248\times10^{14} / 3600 = 19.1$ GPU-hr at 40% MFU, which sits squarely inside it. Both are defensible statements *if you say which FLOP model you used*; only the full one predicts your actual bill. The practical takeaway is to budget "about a day of A100 time" and treat Ch. 14.7's measured throughput as ground truth — which is also why MFU must be quoted against full model FLOPs, or a 1.57× accounting error hides inside a number that looks like a hardware metric.

**5.** *(Implementation.)* The prediction example claims that for the flagship budget $C = 1.347\times10^{19}$ FLOPs, the *compute-optimal* configuration is $d_{\text{model}}=768$, 40 layers, $N^\star \approx 250\text{M}$, $D^\star \approx 6.65\text{B}$ (~27 tok/param). Write `compute_optimal_allocation(C, E, A, alpha, B, beta)` that finds this by searching over **real configurations** in the deep-and-thin family, with $D$ forced by the **full** FLOP model. Run it with the ground-truth law, confirm the numbers, and then re-run it with a $6ND$ constraint to see how much the answer moves.

??? note "Solution"
    On an IsoFLOP slice $D$ is not free: fixing $C$ forces $D = C/c(N)$, so the two-variable law collapses to a one-variable function we minimize directly. The subtlety is that $c$ depends on the *shape* of the model (layers, width, vocab), not just $N$ — so we grid over configurations, not over an abstract $N$.

    ```python
    import numpy as np
    from stacklm.scaling.ladder import LadderConfig
    from stacklm.scaling.flops import flops_per_token

    def family(d_model, n_kv_heads=2):
        """The ladder's deep-and-thin aspect ratio, continued smoothly."""
        return LadderConfig(f"d{d_model}", d_model=d_model,
                            n_layers=max(4, round(d_model / 19.2)),
                            n_kv_heads=n_kv_heads)

    def compute_optimal_allocation(C, E, A, alpha, B, beta, use_6nd=False):
        """Minimize L(N, D) = E + A N^-alpha + B D^-beta subject to the FLOP
        constraint, searching real configs. D is forced by the constraint."""
        best = None
        for d in range(128, 2049, 64):
            cfg = family(d)
            N = cfg.nonembed_params()
            D = C / (6.0 * N) if use_6nd else C / flops_per_token(cfg)["total"]
            L = E + A * N**(-alpha) + B * D**(-beta)
            if best is None or L < best["L"]:
                best = dict(d=d, layers=cfg.n_layers, N=N, D=D, L=L, tpp=D / N)
        return best

    E, A, alpha, B, beta = 2.45, 124.0, 0.33, 234.0, 0.30     # the ground-truth law
    C_full = 7.966e8 * (200 * 84.54e6)                        # ~1.347e19
    r = compute_optimal_allocation(C_full, E, A, alpha, B, beta)
    print(f"d={r['d']} L={r['layers']} N*={r['N']/1e6:.1f}M D*={r['D']/1e9:.2f}B "
          f"tok/param={r['tpp']:.1f} L*={r['L']:.4f}")
    # -> d=768 L=40 N*=249.7M D*=6.65B tok/param=26.6 L*=2.9245
    ```

    The valley is genuinely interior and genuinely flat, which is why you should never quote $N^\star$ to three digits: at $d$=640 ($N$=146.0M) $L=2.9302$; at $d$=704 (193.4M) $L=2.9258$; at $d$=768 (249.7M) $L=2.9245$; at $d$=832 (316.0M) $L=2.9260$; at $d$=896 (393.5M) $L=2.9299$. A 2.7× range of model sizes is within 0.006 nats of the minimum — the same weak-identification story the fitting section told, now visible in the allocation itself.

    Re-running with `use_6nd=True` on the *same* $C_{\text{full}}$ still lands at $d$=768 but reports 36.0 tok/param instead of 26.6, because $6ND$ credits the model with 1.6× more tokens than it can actually afford. And if you pair the $6ND$ *budget* $8.58\times10^{18}$ with the $6ND$ *constraint* — two errors that partly cancel — you get $d$=704, $N^\star \approx 193\text{M}$ at $38$ tok/param, the familiar "a few tens of tokens per parameter" folk answer. That partial cancellation is exactly why the $6ND$ rule survives in practice and exactly why you should not rely on it when the numbers matter: it is roughly right about the *allocation ratio* and badly wrong about the *bill*.

    Either way, the compute-optimal model is ~3× larger than the 84.5M the plan actually builds — the tension the over-training section resolves.

**6.** *(Quantitative, hard.)* From the "Two ways to spend the same $1.35\times10^{19}$ FLOPs" table: compute-optimal is $d$=768, $N=249.7\text{M}$, $D=6.65\text{B}$; the plan's choice is $d$=512, $N=84.5\text{M}$, $D=16.91\text{B}$. (a) Verify both **training** runs cost essentially the same FLOPs under the full model. (b) Using inference cost $\approx 2N$ FLOPs/token, compute the ratio of the two models' per-token inference cost and the percentage cut. (c) Given (a) and the fact that the two models reach nearly the same loss ($\sim 2.92$ vs $\sim 2.95$), argue why the over-trained model is the better deployment choice *for any positive serving load* $D_{\text{infer}}$ — and identify the one assumption that argument rests on.

??? note "Solution"
    **(a) Training FLOPs match.** The compute-optimal config ($d$=768, 40 layers, $N=249.7\text{M}$) has per-token cost $6N + 6Lsd + 6dV = 1.498\times10^{9} + 3.775\times10^{8} + 1.510\times10^{8} = 2.027\times10^{9}$, so
    $$
    C_{\text{opt}} = 2.027\times10^{9} \times 6.65\times10^{9} = 1.347\times10^{19}.
    $$
    Ours is $7.966\times10^{8} \times 1.6908\times10^{10} = 1.347\times10^{19}$. Equal by construction — this is a single IsoFLOP slice, which it only *is* because we used the full cost model on both sides.

    **(b) Inference cost ratio.** Per-token cost is $2N$, so
    $$
    \frac{2 \times 84.5\text{M}}{2 \times 249.7\text{M}} = \frac{84.5}{249.7} \approx 0.338,
    $$
    i.e. every future forward pass costs $\approx 0.34\times$ as much — a permanent cut of $1 - 0.338 \approx \mathbf{66\%}$ to inference FLOPs, and correspondingly to latency, activation memory, and KV-cache footprint (all of which scale with $N$; the KV cache also scales with $n_{\text{layers}} \times n_{\text{kv}} \times \text{head\_dim}$, which shrinks here too, from 40 layers to 30).

    **(c) Why over-training wins for any $D_{\text{infer}} > 0$.** Lifetime compute is
    $$
    C_{\text{lifetime}} = \underbrace{c(N) D_{\text{train}}}_{\text{once}} + \underbrace{2 N D_{\text{infer}}}_{\text{forever}}.
    $$
    The training terms are equal (part a) and the achieved losses are within $\sim 0.03$ nats. So the two options are essentially tied on the training term and on quality, and the *only* remaining difference is the inference term, which is strictly smaller for the 84.5M model at every $D_{\text{infer}} > 0$:
    $$
    C_{\text{lifetime}}^{\text{ours}} - C_{\text{lifetime}}^{\text{opt}} \approx 0 + \big(2\cdot84.5\text{M} - 2\cdot249.7\text{M}\big) D_{\text{infer}} = -(3.30\times10^{8})\, D_{\text{infer}} < 0 .
    $$
    Because the training budgets are matched, there is effectively no break-even to clear: the over-trained small model is cheaper from the first served token onward. And the $0.03$-nat penalty and the $0.34\times$ ratio are set by the exponents and the fixed FLOP constraint, so they survive the fit's $\pm 0.1$-nat uncertainty on the absolute level.

    **The assumption.** That $0.03$ nats of loss is worth *nothing* — i.e. that the two models are behaviourally equivalent. Loss is not capability. A 250M model may cross a downstream threshold (multi-step arithmetic, instruction-following consistency) that an 85M model does not, and no amount of over-training buys a capability that requires depth. The argument is airtight on *compute* and merely plausible on *quality*, which is why Ch. 14.11 validates capabilities directly rather than inferring them from the loss curve.

**7.** *(Quantitative.)* The chapter rejects `batch_tokens = 2**19` for all runs. (a) For the cheapest sweep run — S1 at 12 tokens/param — compute $D$ and the number of optimizer steps a 0.5M-token batch would give. (b) With WSD warmup at 5% of steps and decay over the final 20%, how many steps land in each phase? Comment on whether the measurement is usable. (c) Apply the chapter's `batch_tokens()` rule and recompute. (d) Verify that the same rule returns $2^{19}$ at the flagship's token count.

??? note "Solution"
    **(a)** $D = 12 \times 3.93216\times10^{6} = 4.719\times10^{7}$ tokens. At $2^{19} = 524288$ tokens/step that is
    $$
    4.719\times10^{7} / 524288 = \mathbf{90\ \text{steps}}.
    $$

    **(b)** Warmup $= 0.05 \times 90 \approx 5$ steps; decay $= 0.20 \times 90 = 18$ steps; stable $\approx 67$ steps. This is not a measurement of anything. Ninety Adam/Muon updates is barely past initialization — the model is still in the regime where the loss is dominated by learning the unigram distribution, the WSD "stable phase" has no time to be stable, and a 5-step warmup on a fresh model at $8\times10^{-3}$ AdamW LR is itself a stability hazard. Worse, the point is not merely noisy: a 0.5M-token batch is roughly $5\times$ above the critical batch size at $D \approx 4.7\times10^{7}$ ($22.91 \times (4.719\times10^{7})^{0.47} \approx 9.3\times10^{4}$ tokens), so the run lands *systematically* high, and it is one of the low-tokens-per-param anchors that pin $\beta$.

    **(c)** The rule takes $\min(D/2000,\, B^\star(D)) = \min(2.36\times10^{4},\, 9.3\times10^{4}) = 2.36\times10^{4}$, rounds *down* to a power of two ($2^{14} = 16384$, which is also the floor), giving
    $$
    4.719\times10^{7} / 16384 = \mathbf{2880\ \text{steps}}
    $$
    — 144 warmup, ~2160 stable, 576 decay. Now the schedule means something, and 16384 tokens is 8 sequences of 2048, entirely comfortable on one A100 for a 4M-parameter model.

    **(d) Flagship.** $D = 1.6908\times10^{10}$. Then $D/2000 = 8.45\times10^{6}$ and $B^\star(D) = 22.91 \times (1.6908\times10^{10})^{0.47} \approx 1.47\times10^{6}$, so the binding constraint is the critical batch size; rounding down gives $2^{20}$, which the clip caps at $2^{19} = 524288$ — **exactly the ~0.5M-token batch the plan specifies**, reached in 32,250 steps. A rule that reproduces your hand-chosen production setting from first principles is a rule you can trust on the runs you have not hand-checked.

**8.** *(Design.)* You have USD 2 of A100 time (about 1.3 GPU-hours) to decide between the plan's default mixture and one alternative that doubles the math fraction. Design the experiment: which rung, how many tokens, what validation set, what you would report, and what result would make you *reject* the alternative even if its aggregate loss is lower. Then state the one thing this experiment cannot tell you.

??? note "Solution"
    **Design.** 1.3 GPU-hr at 40% MFU is $\approx 5.8\times10^{17}$ FLOPs. Spend it as a screen plus a confirmation rather than one big run:

    - **Screen at S1, 100 tok/param** ($D = 3.93\times10^{8}$ tokens, 3,000 steps at a 131,072-token batch): $3.34\times10^{16}$ FLOPs per candidate. Run *three* arms — plan, more-math, and a `web_only` null — for $1.00\times10^{17}$.
    - **Confirm at S2, 60 tok/param** ($D = 5.49\times10^{8}$, ~2,100 steps): $8.03\times10^{16}$ per arm. Confirm the top two: $1.61\times10^{17}$.
    - **Seed repeats**: rerun the winning and runner-up arms at S1 with a second data-order seed ($2 \times 3.34\times10^{16} = 6.7\times10^{16}$). Without a noise floor you cannot tell a real 0.01-nat gap from a shuffle.
    - Total $\approx 3.3\times10^{17}$ FLOPs $\approx 0.73$ GPU-hr — inside the USD 2 budget with room to spare for a failed job.

    Everything else is frozen: same tokenizer, same architecture, same per-rung LR from the muP table, same WSD shape, same seed policy.

    **Validation set.** One *fixed* set, composed for the deployment target (say 55% FineWeb-Edu / 20% Cosmopedia / 10% code / 15% math), never per-candidate — otherwise you are measuring the entropy of each mixture rather than its usefulness. Verify it is disjoint from every training shard (Ch. 14.2's dedup manifest).

    **Report.** The target-weighted score *and* the four per-domain losses *and* the seed-to-seed spread, as a table. One number is not a result.

    **When to reject the winner anyway.** If `more_math` wins on the aggregate score but its **FineWeb-Edu per-domain loss rises by more than the seed noise**, reject it: the capstone's agent must read and summarize prose (Ch. 14.10), and buying arithmetic with fluency is the wrong trade. Same verdict if the aggregate win is smaller than the seed-to-seed spread — that is not a win, it is a shuffle. And if `web_only` is within noise of both, take it: fewer moving parts, less pipeline to maintain.

    **What it cannot tell you.** Whether the ranking *transfers* to 100M. DataDecide (Magnusson et al., 2025) measured this directly across 25 corpora and 14 sizes: single-small-size rankings pick the 1B winner in roughly 80% of pairwise comparisons — excellent for screening, but one comparison in five flips. Nor can it tell you anything about *capabilities*: mixture effects on downstream arithmetic or instruction-following are threshold-y and may not appear at 4M or 9M parameters at all. Screen here, confirm at the top rung, and re-check after mid-training (Ch. 14.8), where the WSD decay phase re-weights the mixture anyway.
