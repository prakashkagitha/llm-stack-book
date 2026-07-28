# 14.5 Mini Scaling Laws: Fit Your Own Law Before Spending the Budget

You are about to spend roughly **20 billion tokens** — 15 to 25 A100-hours, USD 40 to 100 — training `Stack-100M` (the exact config is fixed in [The Architecture](../14-capstone/04-architecture.html); we reproduce it below). Before you commit that budget you should be able to answer one question with a *number*, not a shrug: **what final loss should this run reach, and is the 100M/20B-token allocation actually the right one?** If your prediction and your run disagree by more than a couple of tenths of a nat, something is broken — a data-pipeline bug, a mistuned learning rate, a tokenizer mismatch — and you want to know that on hour two, not hour twenty.

The general theory of how loss scales with parameters $N$ and tokens $D$ is developed in [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html). That chapter gives you the *form* of the law and the folk constants. This chapter does the thing the theory chapter tells you to do but does not do for your specific setup: **it fits the law to your own recipe, on your own data, with your own tokenizer, and extrapolates to the exact model you are about to train.** The constants $E$, $A$, $B$, $\alpha$, $\beta$ are *not* universal — they depend on the corpus (FineWeb-Edu + Cosmopedia + a little code and math), the vocab (32,768), and every architecture choice frozen into the recipe. Importing another lab's numbers is how you mis-budget a run.

The plan of attack, following Hoffmann et al.'s Chinchilla methodology at miniature scale: train a **ladder of four tiny models** (about 4M, 9M, 19M, 44M non-embedding parameters) under the *identical* recipe, each at several token budgets, fit $L(N,D)=E+A/N^{\alpha}+B/D^{\beta}$ to the ~17 measured losses, cross-check with **IsoFLOP profiles**, then **extrapolate** to 84.5M and predict `Stack-100M`'s loss. The whole ladder costs on the order of **USD 3** — about 10% of the flagship run — and it is the cheapest insurance you will ever buy. Then we use the fitted law to justify the plan's most counterintuitive decision: **deliberately over-training to ~200 tokens/parameter**, an order of magnitude past compute-optimal.

---

## A Ladder Under One Frozen Recipe

A scaling-law fit is only as trustworthy as the *invariance* of everything you are not varying. The single rule that makes the ladder valid is this: **hold the entire recipe fixed and change only $N$ and $D$.** Same tokenizer, same data mixture and ordering, same optimizer (Muon + AdamW), same [WSD schedule](../03-pretraining/10-lr-schedules-hparams.html) shape, same RMSNorm/RoPE/NoPE/GQA/SwiGLU/QK-norm stack from the plan. If you let the architecture drift between rungs, you are no longer fitting a law — you are fitting noise, which is precisely the confound that broke Kaplan et al.'s original allocation (a learning-rate schedule that was not matched to each run's token count).

### Scaling the ladder: deep-and-thin at every rung

`Stack-100M` is **deep-and-thin** (30 layers × 512 wide), following the small-model result that at fixed parameters, more layers × narrower width beats the reverse (MobileLLM, Liu et al. 2024). We scale the ladder the same way — growing `d_model` and `n_layers` *together* so every rung keeps roughly the same depth-to-width aspect ratio as the target. Head dimension stays pinned at 64; the small rungs collapse GQA to a single KV head (MQA), which is the natural small-model limit of the 8:2 grouping the target uses.

| Rung | `d_model` | `n_layers` | `n_heads` | `n_kv_heads` | `intermediate` | **N (non-embed)** |
|---|---|---|---|---|---|---|
| **S1** | 192 | 10 | 3 | 1 | 512 | **3.93M** |
| **S2** | 256 | 13 | 4 | 1 | 704 | **9.16M** |
| **S3** | 320 | 17 | 5 | 1 | 896 | **18.80M** |
| **S4** | 448 | 21 | 7 | 1 | 1216 | **43.95M** |
| *target* | *512* | *30* | *8* | *2* | *1408* | *84.54M* |

The ladder spans about **1.05 decades** in $N$ (3.93M → 43.95M) and the target sits just **0.28 decades** above the top rung — a short, honest extrapolation. That closeness is a feature of the $100 project: you are not predicting a 400B model from 100M runs (a 3.5-decade leap of faith), you are predicting 84.5M from 44M. The law barely has to reach.

### The embedding subtlety you must not ignore

Here is a trap that is unique to *small* models and that the frontier-scale literature glosses over. The `Stack-100M` recipe uses a 32,768-token vocabulary, and the **tied** input/output embedding is a $32768 \times 512 = 16.78\text{M}$-parameter table (Press & Wolf 2017; the vocab-size tradeoff is analyzed in [Tokenization: BPE, WordPiece, Unigram & Byte-Level](../02-transformer/01-tokenization.html)). At the 4M rung, that embedding is *four times larger than the entire rest of the model*. If you fit the law on **total** parameters, the embedding — which does a fixed amount of work per token regardless of depth — swamps the signal and corrupts your exponent.

The standard fix, used by Kaplan and Chinchilla alike, is to fit on **non-embedding parameters**: the parameters in the transformer blocks that actually do the sequence-mixing and feature-transformation work whose FLOPs the $6ND$ rule counts. Every $N$ in this chapter is a non-embedding count. (The embedding still costs memory and its own small forward FLOPs — but it does not participate in the depth-driven capacity scaling the law describes.)

```python
# stacklm/scaling/ladder.py  -- ladder configs + non-embedding parameter accounting.
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

    @property
    def n_heads(self) -> int:
        assert self.d_model % self.head_dim == 0
        return self.d_model // self.head_dim

    @property
    def intermediate(self) -> int:                 # SwiGLU hidden width
        raw = 2.75 * self.d_model
        return int(round(raw / 64) * 64)           # round to multiple of 64

    def nonembed_params(self) -> int:
        """Parameters in the N transformer blocks (what the 6ND rule counts).
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
              f"total={c.total_params()/1e6:6.2f}M")
    # Stack-100M prints N_nonemb=84.54M, total=101.32M  -- matches the plan's ~101M.
```

Run it and the target line reads `N_nonemb=84.54M  total=101.32M`, reproducing the plan's parameter budget exactly (16.78M tied embedding + 84.54M in blocks). That arithmetic reproducibility is the point: if your own counter disagrees, fix it *now*, because every downstream FLOP and dollar estimate rides on it.

---

## Compute Accounting: The 6ND Budget for the Ladder

To lay out the sweep we need to convert each `(N, D)` run into FLOPs and dollars. The workhorse is the dense-transformer rule derived in the [scaling-laws chapter](../03-pretraining/04-scaling-laws.html): **2 FLOPs per parameter per token forward, 4 backward, so $C \approx 6ND$**, with $N$ the non-embedding parameter count.

```python
# stacklm/scaling/flops.py
def training_flops(n_nonembed: int, n_tokens: int) -> float:
    """Total training FLOPs (the 6ND rule): 2 fwd MAC + 4 bwd, per param per token."""
    return 6.0 * n_nonembed * n_tokens

def gpu_hours(flops: float, peak_flops_per_s: float = 312e12, mfu: float = 0.40) -> float:
    """Wall-clock on ONE accelerator. 312 TFLOP/s ~ A100 bf16 peak; MFU ~0.40 is
    a realistic single-GPU number for a 100M-class model (see Ch. 14.7)."""
    return flops / (peak_flops_per_s * mfu) / 3600.0
```

### Designing the sweep: an IsoFLOP backbone plus off-diagonal points

Two forces pull against each other. To **identify** the law you want spread in *both* directions — some runs param-limited (few tokens per param), some data-limited (many). To keep the sweep **cheap** you cannot run the 44M rung to 400 tokens/param (that single run would cost more than the flagship). The resolution is a hybrid grid:

- **An IsoFLOP backbone.** Pick four compute budgets $C \in \{6\times10^{15},\,1.5\times10^{16},\,4\times10^{16},\,10^{17}\}$. Inside each budget, run every ladder rung whose implied $D=C/(6N)$ lands at a sane tokens/param (roughly 6–700). Because every run in a slice costs the *same* $C$, these slices give the IsoFLOP method its raw material — the compute-optimal $N$ at each budget — though with only four rungs each slice is thinly populated (we return to that limitation below).
- **A few off-diagonal fixed-model points.** IsoFLOP slices all lie on lines of constant $6ND$ — a degenerate direction that cannot separate $\alpha$ from $\beta$ in the additive form. Adding a handful of extra runs (the cheap rungs pushed to very low and very high token counts) breaks that degeneracy and lets the parametric fit pin the exponents.

```python
# stacklm/scaling/sweep.py -- design the ladder sweep and cost it out.
from stacklm.scaling.ladder import LADDER, TARGET
from stacklm.scaling.flops import training_flops, gpu_hours

BY_NAME = {c.name: c for c in LADDER}

# (1) IsoFLOP backbone: (compute_budget_C, rung). Every run in a slice costs exactly C.
ISO = [(6e15, "S1"), (6e15, "S2"),
       (1.5e16, "S1"), (1.5e16, "S2"), (1.5e16, "S3"),
       (4e16, "S1"), (4e16, "S2"), (4e16, "S3"),
       (1e17, "S2"), (1e17, "S3"), (1e17, "S4")]
# (2) Off-diagonal fixed-model points: (rung, tokens_per_param).
EXTRA = [("S1", 12), ("S2", 12), ("S3", 12), ("S4", 20), ("S1", 300), ("S2", 120)]

def build_runs():
    runs = []  # each run is a dict the training harness consumes
    for C, name in ISO:
        c = BY_NAME[name]; N = c.nonembed_params()
        D = C / (6.0 * N)                          # tokens s.t. 6*N*D == C exactly
        runs.append(dict(cfg=c, N=N, D=D, C=6*N*D, tpp=D/N, kind="iso"))
    for name, tpp in EXTRA:
        c = BY_NAME[name]; N = c.nonembed_params()
        D = float(tpp * N)
        runs.append(dict(cfg=c, N=N, D=D, C=6*N*D, tpp=float(tpp), kind="fixed"))
    return runs  # keep D as a float here; the harness rounds to int tokens at train time

if __name__ == "__main__":
    runs = build_runs()
    ladder_flops = sum(r["C"] for r in runs)
    # The flagship run: Stack-100M over-trained to ~200 tok/param (~16.9B tokens).
    big_flops = training_flops(TARGET.nonembed_params(), 200 * TARGET.nonembed_params())
    print(f"{len(runs)} ladder runs")
    print(f"ladder compute = {ladder_flops:.2e} FLOPs "
          f"= {100*ladder_flops/big_flops:4.1f}% of the flagship run")
    print(f"ladder wall-clock (1xA100, 40% MFU) = {gpu_hours(ladder_flops):.2f} GPU-hr "
          f"~ ${gpu_hours(ladder_flops)*1.5:.2f}")
    print(f"flagship run = {gpu_hours(big_flops):.1f} GPU-hr")
```

This prints **17 runs, ≈9.7% of the flagship compute, about 1.9 GPU-hours (~USD 3)**. Note the honest scale effect: at the frontier a scaling ladder is well under 1% of the target run, because the target is thousands of times bigger than the ladder. Here the top rung (44M) is *half* the target (84.5M), so the ladder is an unavoidably larger fraction — but 10% of a $30 run to know the answer before you spend the other 90% is still the best trade in the project.

!!! warning "Match the LR decay to each run's own token count"
    The single most common way to poison a scaling ladder is to reuse one long learning-rate schedule and read off intermediate losses. A run whose [WSD or cosine decay](../03-pretraining/10-lr-schedules-hparams.html) has not finished evaluates *worse than it truly is*, which inflates the high-token losses and biases the fit toward "make the model bigger" — the exact Kaplan confound Chinchilla diagnosed. Every rung in the sweep gets its **own** schedule that decays to zero at *its* token budget $D$. In WSD terms (Ch. 14.6) that means a short run gets a short stable phase and its own decay tail.

### The sweep harness

The harness is thin — it drives the real training loop from Chapter 14.7 and records one JSON line per run. In continuous integration the loop is stubbed to a few steps on a synthetic corpus (hermetic, CPU-only); on the A100 it runs for real.

```python
# stacklm/scaling/run_sweep.py
import json, os
from stacklm.scaling.sweep import build_runs
from stacklm.train import train_run        # thin convenience wrapper over the Ch. 14.7 loop
from stacklm.model import StackConfig, Stack100M
from stacklm.data import PackedDataset      # Ch. 14.2 memmap shards

def run_one(run, data_dir, out_path):
    c = run["cfg"]
    model_cfg = StackConfig(                # frozen recipe -- only d_model/n_layers vary
        vocab_size=c.vocab_size, d_model=c.d_model, n_layers=c.n_layers,
        n_heads=c.n_heads, n_kv_heads=c.n_kv_heads, head_dim=c.head_dim,
        intermediate=c.intermediate, rope_theta=10000.0,
        tie_embeddings=True, nope_every=4, qk_norm=True,     # <- identical across rungs
    )
    model = Stack100M(model_cfg)
    data = PackedDataset(data_dir, seq_len=2048)
    # train_run() returns final held-out loss (nats/token), LR decayed to THIS run's D
    # (WSD stable+decay from Ch. 14.6; Muon+AdamW hybrid; same seed policy for all rungs).
    val_loss = train_run(model, data, total_tokens=int(run["D"]),
                         batch_tokens=2**19, optimizer="muon+adamw",
                         schedule="wsd", seed=1234)
    row = dict(name=c.name, N=run["N"], D=run["D"], C=run["C"],
               tpp=run["tpp"], kind=run["kind"], val_loss=val_loss)
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
              f"tpp={r['tpp']:6.1f} -> val_loss={r['val_loss']:.4f}")
```

### Designing your own ladder: a checklist

If you adapt this to a different target than `Stack-100M`, the design choices above generalize into a short checklist. Every item exists to defeat a specific failure mode.

1. **Span at least a decade in $N$.** Our rungs cover 3.9M → 44M (1.05 decades). Less spread and the exponent $\alpha$ is unconstrained; the power law needs leverage.
2. **Keep the top rung within ~0.3 decades of the target.** We extrapolate 44M → 84.5M. Extrapolating a factor of 2 is honest; extrapolating a factor of 100 is a prayer. The further the reach, the more the loose $E,A,B$ offset hurts you.
3. **Give every rung its own decayed schedule.** Non-negotiable — this is the Kaplan confound, and it is a one-line bug (a shared `total_steps`) that silently rotates your whole fit.
4. **Freeze the recipe byte-for-byte.** Same tokenizer, data order, `nope_every`, `qk_norm`, optimizer split. If you must change one thing (say, test MLA vs GQA), that is a *separate* ladder, not a mixed one.
5. **Budget 5–15% of the flagship compute.** At frontier scale a ladder is <1%; at 100M the top rung is a large fraction of the target, so 5–15% is honest. Ours is ~10% (~USD 3).
6. **Add a few seeds at the cheapest rung.** Two or three reruns of S1 at one token budget measure your noise floor directly, which tells you the Huber `delta` to use and whether two rungs *really* differ or just wiggled.
7. **Hold out the top rung, refit, and check the prediction.** The only success criterion that matters is extrapolation: fit on S1–S3, predict S4, and demand agreement within a couple of percent before you trust the leap to 84.5M.

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

def objective(theta):
    return np.sum(huber(predict_log_loss(theta, N_obs, D_obs) - np.log(L_obs)))

# Multi-start (the surface is non-convex). We CONSTRAIN the exponents to the
# 0.25-0.42 band that the scaling literature consistently reports -- with only 4
# rungs spanning ~1 decade in N, an unconstrained fit will happily rail alpha to
# an absurd 0.1 or 0.5 and take the extrapolation with it. Bounding the exponents
# to physically-sane values is the single most important regularizer at this scale.
best, best_val = None, np.inf
for _ in range(60):
    x0 = np.array([rng.uniform(0.4, 1.1), rng.uniform(3, 7), rng.uniform(3, 7),
                   rng.uniform(0.25, 0.42), rng.uniform(0.25, 0.42)])
    res = minimize(objective, x0, method="L-BFGS-B",
                   bounds=[(0.2, 1.3), (0, 10), (0, 10), (0.25, 0.42), (0.25, 0.42)])
    if res.fun < best_val:
        best, best_val = res.x, res.fun

e, a, b, alpha, beta = best
E_, A_, B_ = np.exp(e), np.exp(a), np.exp(b)
print(f"fitted:  E={E_:.3f}  A={A_:.1f}  alpha={alpha:.3f}  B={B_:.1f}  beta={beta:.3f}")
print(f"alloc exponent  a = beta/(alpha+beta) = {beta/(alpha+beta):.3f}")
```

On this grid the fit returns an allocation exponent $\beta/(\alpha+\beta) \approx 0.44\text{–}0.49$ — near Chinchilla's $\approx0.5$, and bracketing the ground-truth $0.30/0.63 = 0.476$ — that stays in that narrow band across noise seeds, while the *individual* constants swing wildly from seed to seed: $E$ lands anywhere from ~2.3 to ~2.9, $A$ from ~120 to ~480. That instability is not a bug; it is the whole lesson of the next paragraph.

### Read the fit honestly: the loss extrapolates, the constants do not

Compare any single seed's fitted constants to the ground truth (`E=2.45, A=124, alpha=0.33, B=234, beta=0.30`) and the mismatch is glaring — $A$ routinely comes out **3–4× too large** or too small, $E$ wanders half a nat — and *the constants change every time you re-run with different noise*. This is *not* a bug, and it is the single most important thing to internalize about scaling-law fits: **$E$, $A$, and $B$ are strongly correlated and only weakly identified.** The additive surface is nearly flat near the optimum, so many $(E,A,B)$ triples fit the same 17 points almost equally well; they trade off against each other and only their *combination* is pinned. If your colleague's fit reports different constants, neither of you is wrong. Epoch AI's replication of Chinchilla (Besiroglu et al., 2024) found the original parametric constants were fragile for exactly this reason.

What *is* well identified are the two quantities you actually care about — the **allocation exponent** (recovered as $\approx0.44\text{–}0.49$ regardless of seed) and the **extrapolated loss**. Because the fit interpolates the measured surface faithfully, and the target sits only 0.28 decades beyond the top rung, the compensating errors in $E$, $A$, $B$ largely cancel where it matters:

```python
def predicted_loss(N, D):
    return float(np.exp(predict_log_loss(best, np.array([N]), np.array([D])))[0])

N100 = 84.54e6
for tpp in (20, 200):
    D = tpp * N100
    print(f"Stack-100M @ {tpp:3d} tok/param (D={D/1e9:4.1f}B): "
          f"predicted L={predicted_loss(N100, D):.3f}  "
          f"(ground truth {_law(N100, D, GROUND_TRUTH):.3f})")
# @ 20  tok/param: predicted L ~3.1-3.2   (ground truth 3.149)
# @ 200 tok/param: predicted L ~2.9-3.05  (ground truth 2.950)
# (exact digits vary with the noise seed; the ~0.1-nat BAND is what's stable.)
```

Across noise seeds the extrapolation stays **within about 0.1 nats** of the truth, with no consistent bias — some seeds land a little high, some a little low (the constrained fit trades a sliver of absolute accuracy for stability). That is exactly the resolution you should expect from a four-rung ladder — and exactly enough. The lesson, straight from the theory chapter: *validate a scaling-law fit by extrapolation, never by admiring the raw constants.* Report the predicted loss with an honest ±0.1-nat band, not five decimal places of $A$. A ±0.1-nat band is useless for splitting hairs between two good runs but perfect for its real job: **a broken run misses by 0.3+ nats**, and that you will catch on hour two.

### IsoFLOP Profiles: Chinchilla's Second Method

The parametric fit is Chinchilla's Approach 3 (fit the whole surface, differentiate). Its independent cross-check is **Approach 2, the IsoFLOP method**, which never commits to the parametric form and is therefore robust to its misspecification. The idea: at each fixed compute budget $C$, the loss as a function of $\log N$ (with $D=C/6N$ forced) is a **U-shaped valley** — too-small models are param-limited, too-large models are data-starved. Fit a **parabola in $\log N$**, read off the vertex, and you have the compute-optimal $N^\star(C)$ for that budget without ever assuming a power law. Do it for several budgets and fit a line through the valleys: its slope is the allocation exponent $a$ in $N^\star \propto C^{a}$.

There is a catch at *our* scale, and it is worth stating plainly: **a parabola needs enough points, bracketing the minimum on both arms, or its vertex is garbage.** Our four-rung ladder puts only two or three points in each IsoFLOP slice — too few. Fit a three-point parabola to noisy losses and the vertex jitters wildly (across seeds our ladder's slices scatter the exponent anywhere from ~0.3 to ~0.6). So we do the honest thing: demonstrate the method on a properly dense grid (7 models per slice, the way Hoffmann et al. actually ran it), and use the real ladder's coarse valleys only as a loose sanity check. A production ladder would simply include more rungs.

```python
import numpy as np

# Dense IsoFLOP demo: 7 models per slice so the parabola vertex is well-determined.
# (In a real project you'd TRAIN these 7-per-slice; here we read the fitted law.)
rng2 = np.random.default_rng(1)
C_slices = np.array([6e15, 1.5e16, 4e16, 1e17])
N_star = []
for C in C_slices:
    N_center = np.sqrt(C / 120.0)                      # ~20x-rule optimum for the slice
    Ns = N_center * np.logspace(-0.6, 0.6, 7)          # 7 models spanning ~1.2 decades
    Ds = C / (6.0 * Ns)                                # D set so 6*N*D == C for every run
    Ls = _law(Ns, Ds, GROUND_TRUTH) * (1 + 0.01 * rng2.standard_normal(len(Ns)))
    c2, c1, c0 = np.polyfit(np.log(Ns), Ls, 2)         # parabola in log N
    N_star.append(np.exp(-c1 / (2.0 * c2)))            # vertex = valley = compute-optimal N
    print(f"C={C:.1e}: N*={N_star[-1]/1e6:5.2f}M  D*={C/(6*N_star[-1])/1e9:.3f}B")

a_iso, _ = np.polyfit(np.log(C_slices), np.log(N_star), 1)   # N* ~ C^a
print(f"IsoFLOP allocation exponent  a = {a_iso:.3f}")
print(f"parametric  beta/(alpha+beta) = {beta/(alpha+beta):.3f}")
```

The dense valleys march monotonically upward (≈6M → 9M → 15M → 23M) and recover $a \approx 0.48$ — meaning at the compute-optimal point you split each additional decade of compute with $N^\star \propto C^{0.48}$ and $D^\star \propto C^{0.52}$, i.e. **roughly equal growth**, Chinchilla's headline result reproduced from scratch. That the IsoFLOP $a\approx0.5$ and the parametric $\beta/(\alpha+\beta)\approx0.47$ tell the *same* allocation story — two methods with different failure modes agreeing — is the real signal that the law is trustworthy enough to extrapolate. The reassuring fact is that this *slope* is robust even when the absolute loss level (the $E,A,B$ offset) is not: the valley position depends only on where the minima sit, not on how high the surface floats.

??? note "Optional: Chinchilla's third method (the loss envelope) and why we skip it"
    Hoffmann et al. actually used three approaches. Approach 1 is the **loss envelope**: train many models to convergence, plot every run's loss against its FLOPs, trace the lower-left frontier (the best loss achievable at each compute), and fit power laws to the *frontier* points' $N$ and $D$. It is the most data-hungry of the three — you need a dense scatter of runs whose convex hull is well-populated — and at a 17-run ladder the hull is defined by a handful of points, so its exponents are even noisier than the IsoFLOP parabolas. We omit it here for that reason, but the mental model is worth keeping: Approach 1 reads the frontier, Approach 2 reads the valleys at fixed compute, Approach 3 reads the whole surface. All three should agree on the allocation exponent; when they do not, distrust the extrapolation, not just one method.

---

## Extrapolating to Stack-100M and Predicting Its Loss

Now the payoff. We have a fitted law; we plug in the target's non-embedding count, $N=84.54\text{M}$, and read off the loss at the two allocations that matter — the compute-optimal one (~20 tokens/param) and the plan's over-trained one (~200 tokens/param).

!!! example "Predicting Stack-100M's pretraining loss before spending the budget"
    **Setup.** Fitted law (constants above), target $N = 84.54\text{M}$ non-embedding params.

    **Compute-optimal point (Chinchilla ~20 tok/param).** $D = 20N \approx 1.69\text{B}$ tokens.
    $$
    L \approx E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}} \;\approx\; 3.15\ \text{nats/token} \quad (\pm 0.1).
    $$
    This is what a *compute-minimizing* 100M model would reach — and it is exactly the ballpark the [scaling-laws chapter's](../03-pretraining/04-scaling-laws.html) loss-milestone table gives for a ~100M model at ~20 tok/param (~3.2–3.4 nats on web text; a touch lower here because the educational mix is cleaner).

    **The plan's over-trained point (~200 tok/param).** $D \approx 16.9\text{B}$ tokens (the ~20B-token budget).
    $$
    L \approx 2.95\ \text{nats/token} \quad (\pm 0.1\text{ from the fit's absolute-level uncertainty}).
    $$

    **What this buys you.** Over-training the *same* 100M model from 1.69B to 16.9B tokens — a **10× increase in training compute** — moves the loss from ~3.15 to ~2.95, a gain of roughly **0.20 nats**. At this scale that is a large, real improvement in text quality (the difference between a model that frequently loses the thread and one that mostly holds it), and this *difference* is far better pinned than either endpoint because it depends on the well-identified exponents, not the loose offset. Critically, **the run you are about to launch should land at ≈ 2.9–3.0 nats/token held-out.** If hour-two extrapolation of your live loss curve is heading somewhere 0.3+ nats away, stop and debug — that is the entire reason you built the ladder.

    **Cross-check with the FLOP budget.** The flagship run is $C = 6 \times 84.54\text{M} \times 16.9\text{B} \approx 8.6\times10^{18}$ FLOPs. Ask the fitted law what the *compute-optimal* allocation for that same budget would be (minimize $L$ s.t. $6ND=C$): it prefers $N^\star \approx 190\text{M}$, $D^\star \approx 7.5\text{B}$ (~40 tok/param). So for the very same compute, a compute-optimal model would be **~2.2× larger** than the one we are choosing to build. Which raises the obvious question the next section answers: *why are we deliberately building the smaller, over-trained model?*

Use the book's interactive Scaling-Law-Optimal tool to poke at these tradeoffs — drop in the fitted $E,A,B,\alpha,\beta$ and slide the compute budget to watch $(N^\star, D^\star)$ move:

{{tool:scaling-law-optimal}}

---

## The Over-Training Decision: Compute-Optimal ≠ Deployment-Optimal

Chinchilla answers "what minimizes *training* loss for a *training*-compute budget." That is almost never the real objective. `Stack-100M` will be **trained once and served indefinitely** — quantized to int4 and run on a laptop (Ch. 14.11). For a model you deploy, the quantity to minimize is **lifetime compute**, training *plus* all future inference:

$$
C_{\text{lifetime}} = \underbrace{6 N D_{\text{train}}}_{\text{train once}} \;+\; \underbrace{2 N D_{\text{infer}}}_{\text{serve forever}}
$$

subject to hitting a **target loss** $L^\star$. Every inference token costs $\approx 2N$ FLOPs and the KV cache scales with $N$ (see [The Anatomy of LLM Inference](../07-inference-serving/01-anatomy-inference.html) and [Inference Economics](../07-inference-serving/12-inference-economics.html)). So the larger $D_{\text{infer}}$ is, the more it pays to **shrink $N$** (cheap inference forever) and **grow $D_{\text{train}}$** to buy back the loss you gave up. This is the inference-aware regime formalized by Sardana, Frankle et al. (*Beyond Chinchilla-Optimal*, 2024), and it is why Llama-3-8B saw ~15T tokens (~1900 tok/param), two orders of magnitude past Chinchilla.

Here is the decision made concrete with the ladder's own numbers:

!!! example "Two ways to spend the same 8.6e18 FLOPs"
    | | model $N$ | tokens $D$ | tok/param | train loss | inference cost/token |
    |---|---|---|---|---|---|
    | **compute-optimal** | 190M | 7.5B | ~40 | **~2.94** | $2 \times 190\text{M}$ |
    | **our choice** | 84.5M | 16.9B | 200 | **~2.95** | $2 \times 84.5\text{M}$ |

    Same training budget. The over-trained 84.5M model is **only ~0.01 nats/token worse** in loss — a difference you would struggle to detect in generated text — yet it is **less than half the size**: every future forward pass costs $84.5/190 \approx 0.45\times$ as much, a **~55% permanent cut** to inference FLOPs, latency, memory, and KV-cache footprint. You pay the extra training tokens **once**; you collect the inference savings on **every request for the life of the model**. For a model destined to run on a laptop, that trade is not close. (The absolute loss *levels* here carry the fit's ±0.1-nat uncertainty; the **~0.01-nat penalty and the 0.45× size ratio do not** — they are set by the well-identified exponents and the fixed FLOP constraint, so they hold whether the true floor is 2.9 or 3.0.)

    Put differently: compute-optimal is the right target if you will train a model and (nearly) never run it. The instant you plan to *serve* it at any scale, you should slide down the size axis and over-train — which is precisely the ~200 tok/param, ~20B-token budget fixed in the [capstone overview](../14-capstone/01-overview-and-landscape.html).

```python
import numpy as np
def lifetime_optimal(L_target, D_infer, E, A, alpha, B, beta):
    """Pick (N, D_train) to hit L_target while minimizing TRAIN+INFER FLOPs.
    Grid over N; for each N, solve the tokens needed to reach L_target, then score."""
    best = None
    for N in np.logspace(7, 9, 600):                 # 10M .. 1B non-embed params
        budget = L_target - E - A * N**(-alpha)      # loss left for the data term
        if budget <= 0:
            continue                                 # this N alone overshoots L_target
        D_train = (B / budget) ** (1.0 / beta)
        total = 6*N*D_train + 2*N*D_infer
        if best is None or total < best["total"]:
            best = dict(N=N, D_train=D_train, total=total, tpp=D_train/N)
    return best

E, A, alpha, B, beta = 2.45, 124.0, 0.33, 234.0, 0.30    # the fitted (here, GT) law
for D_infer in (1e10, 1e12, 1e14):                       # light -> heavy serving
    r = lifetime_optimal(2.95, D_infer, E, A, alpha, B, beta)
    print(f"D_infer={D_infer:.0e}: N*={r['N']/1e6:6.1f}M  "
          f"D_train={r['D_train']/1e9:6.2f}B  tok/param={r['tpp']:6.0f}")
# As you plan to serve more tokens, the optimal model SHRINKS and tok/param CLIMBS
# -- the quantitative engine behind "over-train a small model for deployment."
```

Run it: as expected serving load rises from $10^{10}$ to $10^{14}$ tokens, the lifetime-optimal $N$ *falls* and tokens/param *climbs* well past 200 — the mathematics of why the plan over-trains. One honest caveat before you crank tokens/param to the moon: the Chinchilla form assumes every token is *fresh*. High-quality data is finite, and past a few epochs of repetition returns collapse (Muennighoff et al.'s data-constrained law; see [scaling laws](../03-pretraining/04-scaling-laws.html) and [Data Cleaning & Dedup](../03-pretraining/02-data-cleaning-dedup.html)). At 16.9B tokens on our ~20B-token deduplicated mix we are inside one epoch, so the fresh-token law holds — but this is exactly why the plan invests so heavily in dedup and synthetic Cosmopedia data rather than simply looping the corpus.

!!! interview "Interview Corner"
    **Q:** You fit a scaling law from a ladder of tiny models and it predicts a 100M model will reach 2.95 nats at 200 tokens/param and 3.15 at 20. Your manager asks why you would ever train past the compute-optimal 20 tokens/param — isn't that wasting compute?

    **A:** It wastes *training* compute but saves *lifetime* compute, and lifetime is what we pay. Compute-optimal (Chinchilla) minimizes training loss for a training-FLOP budget — the right objective only if you train a model and never serve it. Our model is trained once and served indefinitely, so the objective is $6ND_{\text{train}} + 2ND_{\text{infer}}$ subject to a target loss. Because inference cost scales with $N$, the optimum shifts toward a *smaller* model trained on *more* tokens. Concretely, for our fixed FLOP budget the compute-optimal model is ~190M at ~7.5B tokens; we instead train ~85M at ~17B tokens. We give up only ~0.01 nats of loss — imperceptible — to make every future forward pass ~55% cheaper, permanently. And note that penalty is robust even though our tiny-ladder fit only pins the *absolute* loss to ±0.1 nats: the penalty depends on the well-identified exponents and the FLOP constraint, not the loose offset. The break-even is tiny — past a few hundred million inference tokens the over-trained small model is strictly cheaper overall. Two guardrails: keep total tokens within a few epochs of unique data (repetition returns collapse past ~4–16 epochs), and remember the law predicts *loss*, not *capabilities* — downstream abilities are threshold-y and should be validated with smooth proxy metrics, not extrapolated from the loss curve.

---

## Key Takeaways

!!! key "Key Takeaways"
    - **Fit your own law; don't import constants.** $E$, $A$, $B$, $\alpha$, $\beta$ depend on your corpus, tokenizer, and frozen recipe. A four-rung ladder (~4M/9M/19M/44M) under the *identical* Stack recipe costs on the order of USD 3 (~10% of the flagship run) and de-risks the whole 20B-token commitment.
    - **Freeze everything but $N$ and $D$**, and give every rung a learning-rate schedule that decays to zero at *its own* token count — reusing one long schedule is the Kaplan confound that biases the fit toward "bigger model."
    - **Fit on non-embedding parameters.** At vocab 32,768 the 16.8M tied embedding dwarfs the small rungs; counting it corrupts the exponent. The $6ND$ FLOP rule uses non-embedding $N$ too.
    - **Fit in log space with `logsumexp` + Huber loss**, constrain the exponents to the sane ~0.25–0.42 band (with 4 rungs an unconstrained fit rails), multi-start the non-convex objective, and judge the fit by *extrapolation*, not the raw constants — $E$, $A$, $B$ are weakly identified and correlated (our $A$ came out ~3–4× off) while the extrapolated loss lands within ~0.1 nats.
    - **Two methods, one answer.** The parametric $L(N,D)$ fit and Chinchilla's IsoFLOP-parabola method agree on an allocation exponent $a\approx0.5$ ($N^\star, D^\star$ grow roughly together, the Chinchilla result) — and IsoFLOP needs a *dense* slice (6+ models), not our coarse 4-rung ladder, to pin it.
    - **Predicted Stack-100M loss: ~3.15 nats (±0.1) at compute-optimal (20 tok/param, 1.7B tokens), ~2.95 at the over-trained 200 tok/param (16.9B tokens).** Your live run must track this; a >0.3-nat miss means a bug, not bad luck.
    - **Compute-optimal ≠ deployment-optimal.** For the same FLOP budget, compute-optimal wants ~190M params; we deliberately build ~85M and over-train, trading ~0.01 nats of loss for a ~55% permanent cut in inference cost — the deployment economics behind the plan's 200-tok/param budget.
    - **The law predicts loss, not capabilities.** Extrapolate loss with confidence; treat capability thresholds as uncertain and stay within a few epochs of unique data.

## Further Reading

- Hoffmann, Borgeaud, Mensch, et al. *Training Compute-Optimal Large Language Models* (Chinchilla). arXiv 2022. — The three fitting methods (loss-envelope, IsoFLOP, parametric) this chapter miniaturizes.
- Kaplan, McCandlish, Henighan, et al. *Scaling Laws for Neural Language Models*. arXiv 2020. — The original power-law form and the LR-schedule confound to avoid.
- Sardana, Frankle, et al. *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws*. arXiv 2023/2024. — Formalizes lifetime-compute and inference-aware over-training.
- Besiroglu, Erdil, Barnett, et al. *Chinchilla Scaling: A Replication Attempt* (Epoch AI). 2024. — Why the parametric constants are fragile and the extrapolated loss is the thing to trust.
- Muennighoff, Rush, Barak, et al. *Scaling Data-Constrained Language Models*. NeurIPS 2023. — The repetition/data-wall limit on how far you can over-train.
- Liu, Chang, et al. *MobileLLM: Optimizing Sub-billion Parameter Language Models*. 2024. — The deep-and-thin small-model result the ladder inherits.
