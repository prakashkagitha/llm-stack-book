# 3.4 Scaling Laws: Kaplan, Chinchilla & Beyond

Suppose your manager hands you a fixed compute budget — say, a cluster-month that works out to roughly $10^{22}$ floating-point operations (FLOPs) — and asks one question: *"What is the best language model we can train with this?"* You have two main dials to turn. You can make the model **bigger** (more parameters $N$), or you can train it on **more data** (more tokens $D$). Crank $N$ too high and you starve the model of data; it never sees enough text to learn the parameters it has. Crank $D$ too high and you waste compute pushing more tokens through a model too small to absorb them. Somewhere in between is the sweet spot.

Remarkably, this is not guesswork. The relationship between compute, model size, data, and the loss a model achieves turns out to follow clean, predictable **power laws** that hold across more than seven orders of magnitude of compute. These *scaling laws* are arguably the single most consequential empirical discovery in modern LLM engineering: they let you extrapolate from a handful of cheap small-scale runs to predict the loss of a model you have not yet trained, and they tell you exactly how to allocate a budget so that no FLOP is wasted. The entire frontier-model industry — GPT-4, Gemini, Llama, Claude — is built on top of these curves.

This chapter develops scaling laws from first principles. We will derive the power-law form, contrast the original **Kaplan et al.** prescription with the corrected **Chinchilla** prescription (and the famous "~20 tokens per parameter" rule), work through real budget-planning arithmetic, write code that *fits* a scaling law from synthetic data, and then confront the messier frontier: inference-aware over-training, the emergent-abilities debate, and where these laws break down. Scaling laws are an interview favorite at every level — by the end you should be able to derive the compute-optimal allocation on a whiteboard.

This chapter builds directly on [The Pretraining Objective & Loss](../03-pretraining/03-pretraining-objective.html) (what "loss" means here) and [Pretraining Data: Sources, Crawling & The Data Pipeline](../03-pretraining/01-pretraining-data.html) (where the tokens come from). The compute estimates lean on the FLOP accounting from [GPU Architecture & The Memory Hierarchy](../01-foundations/08-gpu-architecture.html).

---

## The Shape of the Curve: Power Laws in Loss

### What we are measuring

Throughout, "loss" $L$ means the **cross-entropy of next-token prediction**, in nats per token, on held-out data — the same objective covered in [The Pretraining Objective & Loss](../03-pretraining/03-pretraining-objective.html). It is the negative log-likelihood the model assigns to the true next token, averaged over the corpus:

$$
L = -\frac{1}{D}\sum_{t=1}^{D} \log p_\theta(x_t \mid x_{<t})
$$

Lower is better. A perfect model would achieve the **entropy of the language itself** — the irreducible uncertainty that no model can remove (the actual next word genuinely is not fully determined by the context). We will call that floor $E$.

### The empirical observation

If you train a family of transformers and plot the loss against model size $N$ (with data effectively unlimited), against data $D$ (with the model effectively unlimited), or against compute $C$, you find something striking: on a **log-log plot, the curve is a straight line**. A straight line on log-log axes is the signature of a **power law**. Concretely, holding everything else non-bottlenecking:

{{fig:scaling-law}}

$$
L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N},
\qquad
L(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D}
$$

where $N_c, D_c$ are constants with units of parameters and tokens, and $\alpha_N, \alpha_D$ are small positive exponents (empirically on the order of $0.05$–$0.1$). The exponents being *small* is the whole story of why training is expensive: to halve the loss-above-floor, you need to multiply $N$ by roughly $2^{1/\alpha_N}$, which for $\alpha_N \approx 0.07$ is about $2^{14} \approx 16000\times$ more parameters. Returns diminish, but they never stop.

### The joint form: the Chinchilla parameterization

The cleanest and most useful parameterization — introduced by Hoffmann et al. (the Chinchilla paper) — treats $N$ and $D$ jointly and adds the irreducible floor:

$$
\boxed{\;L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}\;}
$$

Read this term by term:

- $E$ — the **irreducible loss**, the entropy of natural text. Even an infinitely large model trained on infinite data cannot beat this. Empirically $E$ is somewhere around $1.6$–$1.7$ nats/token for web text, though the exact value depends entirely on the tokenizer and corpus.
- $\frac{A}{N^{\alpha}}$ — the **finite-model penalty**. A model with limited parameters cannot represent the ideal predictor; this term shrinks as you add parameters.
- $\frac{B}{D^{\beta}}$ — the **finite-data penalty**. With limited tokens, the model cannot estimate its parameters well (and overfits); this term shrinks as you add data.

The two penalty terms add. That additivity is what makes the math tractable: it says the param bottleneck and the data bottleneck are (approximately) independent, so we can reason about them separately and then trade them off against a shared compute budget.

{{fig:scaling-loss-floor-curve}}

!!! note "Why a power law and not something else?"
    There is no single agreed first-principles derivation, but the leading intuition comes from data manifolds. If the data lives on a manifold of intrinsic dimension $d$, and a model of size $N$ effectively tiles that manifold with $\sim N$ pieces, then the approximation error per piece scales like $N^{-1/d}$ in the relevant norm — giving a power law whose exponent is set by the *intrinsic dimensionality of language*, not by the architecture. Sharma & Kaplan formalized a version of this argument. The practical upshot: the exponents are remarkably architecture-insensitive. Swap GELU for SwiGLU, add RoPE, change the aspect ratio within reason — the *slope* barely moves, even though the *offset* (the constant $A$) does.

---

## Compute, FLOPs, and the $C \approx 6ND$ Rule

To trade $N$ against $D$ we need a common currency: **compute**, measured in FLOPs. The foundational accounting rule for dense transformers is:

$$
\boxed{\;C \approx 6 \, N \, D\;}
$$

where $C$ is total training FLOPs, $N$ is the number of (non-embedding) parameters, and $D$ is the number of training tokens. This is worth deriving because it shows up in every capacity-planning conversation.

### Deriving the factor of 6

Consider one parameter in a matrix multiply. Processing one token through it requires:

- **Forward pass:** one multiply and one add — that is $2$ FLOPs per parameter per token. (Every weight in a `Linear` layer participates in exactly one multiply-accumulate for each token.)
- **Backward pass:** roughly twice the forward cost — $4$ FLOPs per parameter per token. The backward pass computes *two* sets of products: the gradient with respect to the inputs (to propagate further back) and the gradient with respect to the weights (to update them). Each is about as expensive as the forward matmul.

Add them: $2 + 4 = 6$ FLOPs per parameter per token. Multiply by $N$ parameters and $D$ tokens:

$$
C \approx 6 N D
$$

```python
def training_flops(n_params: int, n_tokens: int) -> float:
    """Total training FLOPs for a dense transformer (the 6ND rule).

    6 = 2 (forward MAC) + 4 (backward: grad wrt input + grad wrt weight).
    This ignores attention's quadratic term, which is small relative to the
    MLP/projection matmuls until context length is very large. It also ignores
    embeddings (use NON-embedding params for N).
    """
    return 6.0 * n_params * n_tokens


# Inference (forward only) is ~2ND -- a useful sanity check:
def inference_flops(n_params: int, n_tokens: int) -> float:
    return 2.0 * n_params * n_tokens
```

!!! warning "Caveats to 6ND that interviewers love"
    The $6ND$ rule counts only the dense matmul FLOPs. Two corrections matter in practice. (1) **Attention** adds a term proportional to $L \cdot d \cdot T^2$ per layer ($T$ = sequence length); for short contexts it is negligible, but for very long contexts it dominates and $6ND$ undercounts — see [Long-Context Pretraining & Context Extension](../03-pretraining/13-long-context-pretraining.html). (2) For **Mixture-of-Experts** models, $N$ in the FLOP formula is the *active* (per-token) parameter count, not the *total* parameter count, because each token routes to only a few experts — see [Mixture-of-Experts (MoE) Architectures](../02-transformer/09-mixture-of-experts.html). MoE breaks the tidy coupling between "model capacity" and "compute," which is precisely why it is attractive.

### Hardware FLOPs vs. model FLOPs

There is a second, sneaky gap. The $6ND$ number is **model FLOPs** — useful arithmetic. The FLOPs your GPUs actually *deliver* are lower because of memory stalls, communication, and bubbles. The ratio is the **Model FLOPs Utilization (MFU)**:

$$
\text{MFU} = \frac{6 N D}{(\text{peak hardware FLOP/s}) \times (\text{wall-clock seconds})}
$$

A well-tuned large pretraining run lands somewhere in the 0.3–0.55 MFU range on modern accelerators. You need MFU to convert a FLOP budget into a *time* and *dollar* budget. We return to this in the worked example.

---

## Kaplan vs. Chinchilla: The Great Correction

{{tool:scaling-law-optimal}}

### Kaplan et al. (2020): scaling laws are born

The first systematic scaling study — Kaplan et al., *Scaling Laws for Neural Language Models* — established the power-law form and, crucially, asked the **compute-optimal allocation** question: given a fixed budget $C$, how should you split it between $N$ and $D$?

Their answer, derived by fitting $L(N, D)$ and minimizing under the constraint $C = 6ND$, was that **most of the budget should go into model size**. As compute grows, you should grow $N$ fast and $D$ slowly. Their fitted relationship was roughly $N_{\text{opt}} \propto C^{0.73}$ and $D_{\text{opt}} \propto C^{0.27}$ — model size soaking up nearly three-quarters of every additional decade of compute. This is why the models of 2020–2021 (GPT-3 at 175B params trained on ~300B tokens, Gopher at 280B on ~300B tokens, MT-NLG at 530B) were **enormous but relatively data-starved**: roughly 1–2 tokens per parameter.

### The bug: a confounded learning-rate schedule

Two years later, Hoffmann et al. (the **Chinchilla** paper, *Training Compute-Optimal Large Language Models*) re-ran the analysis far more carefully and reached a dramatically different conclusion. The headline diagnosis of where Kaplan went wrong: **the learning-rate schedule was not matched to the token budget**.

Recall from [Learning Rate Schedules, Warmup, Batch Size & Hyperparameters](../03-pretraining/10-lr-schedules-hparams.html) that a cosine schedule should decay to its minimum *exactly at the end of training*. Kaplan's experiments largely reused a single long schedule and read off intermediate losses — so the shorter runs (fewer tokens) were evaluated mid-decay, with their learning rate still too high, making them look worse than they truly were. That systematically **understated the value of data**, biasing the optimal allocation toward parameters. Chinchilla retrained every point with a schedule decayed to its own endpoint and used three independent estimation methods that all agreed.

### The Chinchilla result: scale $N$ and $D$ in lockstep

Chinchilla found that the compute-optimal exponents are **approximately equal**:

$$
N_{\text{opt}} \propto C^{a}, \qquad D_{\text{opt}} \propto C^{b}, \qquad a \approx b \approx 0.5
$$

In words: **every time you multiply your compute budget by 10, you should multiply *both* the model size and the data by roughly $\sqrt{10} \approx 3.16$.** They grow together. This is the opposite of Kaplan's "mostly grow the model" prescription, and it implied that essentially every large model trained before 2022 was badly *undertrained* — too big for the amount of data it saw.

The proof of the pudding: Chinchilla (70B parameters, ~1.4T tokens) **outperformed Gopher (280B parameters, ~300B tokens)** despite being 4× smaller — and using the same compute. A smaller model trained on far more data won. That single result reset the entire field's defaults.

{{fig:scaling-kaplan-vs-chinchilla}}

### The "$\approx 20$ tokens per parameter" rule

Because $a \approx b$, the optimal **ratio** $D_{\text{opt}} / N_{\text{opt}}$ is approximately *constant* across compute scales. Chinchilla's fit puts that constant near **20 tokens per parameter**:

$$
\boxed{\;D_{\text{opt}} \approx 20 \, N_{\text{opt}}\;}
$$

This is the single number most people remember from the entire scaling literature. It is a back-of-the-envelope heuristic, not a law of nature — the precise multiplier depends on the dataset, tokenizer, and exact fitted exponents, and reasonable re-derivations land anywhere from ~15 to ~25. But "20× tokens" is the right order of magnitude and a perfectly defensible interview answer.

!!! interview "Interview Corner"
    **Q:** Kaplan said "make the model bigger"; Chinchilla said "scale data and params together." What changed, and what is the practical takeaway?

    **A:** Nothing about the underlying power-law *form* changed — both fit $L(N,D)$ as a sum of power laws. What changed was the *measurement*. Kaplan's smaller-token runs were evaluated with a learning-rate schedule that hadn't finished decaying, so they looked artificially under-performing, which made data look less valuable than it is. Chinchilla matched the cosine decay to each run's actual token count and used three independent methods (fixed-model loss curves, IsoFLOP profiles, and directly fitting the parametric $L(N,D)$). All three agreed that the compute-optimal exponents for $N$ and $D$ are roughly equal, $\approx 0.5$ each. The practical heuristic is **about 20 training tokens per parameter** at the compute-optimal point. Concretely: a compute-optimal 70B model wants ~1.4T tokens, not the ~300B that GPT-3-era models used. The deeper lesson is methodological — scaling-law conclusions are extremely sensitive to whether your hyperparameters (especially the LR schedule and batch size) are tuned *at each scale*, so always validate the experimental setup before trusting the slope.

---

## Deriving the Compute-Optimal Allocation

Let us actually do the optimization Chinchilla did, because it is a clean Lagrange-multiplier problem and a very common whiteboard exercise. We want to minimize the loss subject to a fixed compute budget.

**Problem.** Minimize $L(N,D) = E + A N^{-\alpha} + B D^{-\beta}$ subject to $C = 6 N D$ (fixed).

Since $E$ is constant, drop it. Substitute the constraint $D = C / (6N)$ into the loss:

$$
\tilde{L}(N) = A N^{-\alpha} + B \left(\frac{C}{6N}\right)^{-\beta}
= A N^{-\alpha} + B \left(\frac{6}{C}\right)^{\beta} N^{\beta}
$$

Take the derivative with respect to $N$ and set it to zero:

$$
\frac{d\tilde{L}}{dN} = -\alpha A N^{-\alpha - 1} + \beta B \left(\frac{6}{C}\right)^{\beta} N^{\beta - 1} = 0
$$

Rearranging gives the balance condition: at the optimum, the **marginal loss reduction per FLOP is equal** for parameters and for data. Solving for $N$:

$$
\alpha A N^{-\alpha} = \beta B \left(\frac{6}{C}\right)^{\beta} N^{\beta}
\;\Longrightarrow\;
N^{\alpha + \beta} = \frac{\alpha A}{\beta B}\left(\frac{C}{6}\right)^{\beta}
$$

$$
\boxed{\;N_{\text{opt}} = \left[\frac{\alpha A}{\beta B}\right]^{\frac{1}{\alpha+\beta}} \left(\frac{C}{6}\right)^{\frac{\beta}{\alpha+\beta}}\;}
$$

and by the constraint $D_{\text{opt}} = \frac{C}{6 N_{\text{opt}}} \propto C^{\frac{\alpha}{\alpha+\beta}}$. So the exponents are:

$$
a = \frac{\beta}{\alpha + \beta}, \qquad b = \frac{\alpha}{\alpha + \beta}, \qquad a + b = 1.
$$

The two exponents must sum to 1 (because $C = 6ND$ forces it), and they are *equal* exactly when $\alpha = \beta$. Chinchilla's fitted $\alpha$ and $\beta$ came out close to each other, which is *why* $a \approx b \approx 0.5$ and *why* the token-per-parameter ratio is roughly constant. The whole "20× rule" falls out of $\alpha \approx \beta$.

```python
import numpy as np

def chinchilla_optimal(C, A, B, alpha, beta, E=1.69):
    """Compute-optimal (N*, D*) for budget C given fitted scaling-law params.

    Minimizes L(N,D) = E + A*N^-alpha + B*D^-beta  s.t.  C = 6*N*D.
    Returns the optimal params, tokens, predicted loss, and tokens/param.
    """
    exp = beta / (alpha + beta)                  # exponent on C for N
    coef = (alpha * A / (beta * B)) ** (1.0 / (alpha + beta))
    N_opt = coef * (C / 6.0) ** exp
    D_opt = C / (6.0 * N_opt)
    L_opt = E + A * N_opt ** (-alpha) + B * D_opt ** (-beta)
    return {
        "N": N_opt, "D": D_opt, "loss": L_opt,
        "tokens_per_param": D_opt / N_opt,
        "N_exponent": exp, "D_exponent": 1.0 - exp,
    }

# Illustrative coefficients in the *spirit* of Chinchilla's fit (not exact):
A, B, alpha, beta, E = 406.4, 410.7, 0.34, 0.28, 1.69
for C in [1e19, 1e21, 1e23, 1e25]:
    r = chinchilla_optimal(C, A, B, alpha, beta, E)
    print(f"C={C:.0e}  N={r['N']:.2e}  D={r['D']:.2e}  "
          f"tok/param={r['tokens_per_param']:.1f}  L={r['loss']:.3f}")
```

Running this prints (note the ratio is *not* constant):

```text
C=1e+19  N=2.28e+08  D=7.31e+09  tok/param=32.1  L=2.986
C=1e+21  N=1.82e+09  D=9.14e+10  tok/param=50.1  L=2.329
C=1e+23  N=1.46e+10  D=1.14e+12  tok/param=78.2  L=2.005
C=1e+25  N=1.17e+11  D=1.43e+13  tok/param=122.1  L=1.845
```

With these *Approach-3* constants alpha != beta (0.34 vs 0.28), so the ratio D/N scales as C^{(alpha-beta)/(alpha+beta)} = C^{0.097} -- it grows about 1.25x per decade of compute, drifting from ~32 to ~122 tokens/param across six decades and sitting well above the folk "20x" number at every frontier budget. The clean, *constant* ~20x rule is a property of the regime alpha ~ beta: Chinchilla's Approaches 1 and 2 fitted allocation exponents a ~ b ~ 0.5 (equivalently alpha ~ beta), which is exactly what makes tokens/param scale-invariant. Set alpha = beta = 0.34 in the call above and the tok/param column collapses to a single constant; the slow drift you see here is the honest signature of the Approach-3 parametric fit, not a contradiction of Chinchilla.

---

## Fitting a Scaling Law From Scratch

Reading about scaling laws is one thing; **fitting** one is the skill that actually transfers to the job. The workflow is: (1) run a grid of small models at varying $N$ and $D$, (2) record final losses, (3) fit the parametric form, (4) extrapolate to the target scale, (5) read off the compute-optimal allocation. Here we generate synthetic but realistic data and fit it with a robust loss. The single most important practical trick — which Chinchilla used — is to fit in **log space with the Huber loss**, because raw least-squares on $L$ is dominated by the largest-loss (smallest) runs and is sensitive to outliers.

```python
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

# ---------------------------------------------------------------------------
# Step 0: synthesize a grid of (N, D, observed_loss) "experiments".
# In reality these rows come from actual training runs; here we fabricate them
# from a known ground-truth law plus noise so we can check we recover it.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
TRUE = dict(E=1.69, A=406.0, B=410.0, alpha=0.34, beta=0.28)

def true_loss(N, D, p=TRUE):
    return p["E"] + p["A"] * N ** (-p["alpha"]) + p["B"] * D ** (-p["beta"])

Ns = np.array([1e7, 3e7, 1e8, 3e8, 1e9, 3e9])          # 10M .. 3B params
Ds = np.array([1e8, 3e8, 1e9, 3e9, 1e10, 3e10, 1e11])  # 0.1B .. 100B tokens
grid = [(N, D) for N in Ns for D in Ds]
N_obs = np.array([g[0] for g in grid])
D_obs = np.array([g[1] for g in grid])
# multiplicative ~2% noise, as you'd see from seed/data-order variation:
L_obs = true_loss(N_obs, D_obs) * (1.0 + 0.02 * rng.standard_normal(len(grid)))

# ---------------------------------------------------------------------------
# Step 1: the model. We fit E, A, B, alpha, beta. To keep all params positive
# and the optimizer well-conditioned, we parameterize via logs:
#   E = exp(e), A = exp(a), B = exp(b), alpha and beta directly.
# Chinchilla's key trick: minimize the HUBER loss of the LSE residual in
# LOG-loss space, which is robust to the few noisy outlier runs.
# ---------------------------------------------------------------------------
def predict_log_loss(theta, N, D):
    e, a, b, alpha, beta = theta
    # Build log L via log-sum-exp of the three log-terms (Chinchilla's Appendix
    # trick): numerically stable across many orders of magnitude because it never
    # forms the raw sum E + A*N^-alpha + B*D^-beta before taking the log.
    terms = np.stack([
        np.full_like(N, e),                 # log of E
        a - alpha * np.log(N),              # log of A*N^-alpha
        b - beta * np.log(D),               # log of B*D^-beta
    ])
    return logsumexp(terms, axis=0)         # = log(E + A*N^-alpha + B*D^-beta)

def huber(r, delta=1e-3):
    a = np.abs(r)
    return np.where(a <= delta, 0.5 * r**2, delta * (a - 0.5 * delta))

def objective(theta):
    resid = predict_log_loss(theta, N_obs, D_obs) - np.log(L_obs)
    return np.sum(huber(resid))

# ---------------------------------------------------------------------------
# Step 2: fit. Multi-start because the objective is non-convex; keep the best.
# ---------------------------------------------------------------------------
best, best_val = None, np.inf
for _ in range(40):
    x0 = np.array([
        rng.uniform(0.0, 1.0),     # log E   (E ~ 1..2.7)
        rng.uniform(4.0, 8.0),     # log A
        rng.uniform(4.0, 8.0),     # log B
        rng.uniform(0.1, 0.6),     # alpha
        rng.uniform(0.1, 0.6),     # beta
    ])
    res = minimize(objective, x0, method="L-BFGS-B",
                   bounds=[(-2, 2), (0, 12), (0, 12), (0.01, 1.0), (0.01, 1.0)])
    if res.fun < best_val:
        best, best_val = res.x, res.fun

e, a, b, alpha, beta = best
print(f"Recovered:  E={np.exp(e):.3f}  A={np.exp(a):.1f}  B={np.exp(b):.1f}"
      f"  alpha={alpha:.3f}  beta={beta:.3f}")
print(f"Ground truth: E={TRUE['E']}  A={TRUE['A']}  B={TRUE['B']}"
      f"  alpha={TRUE['alpha']}  beta={TRUE['beta']}")

# ---------------------------------------------------------------------------
# Step 3: use the fit to predict the compute-optimal allocation at a NEW,
# much larger budget than any run in the grid -- the whole point of fitting.
# ---------------------------------------------------------------------------
def optimal_alloc(C, e, a, b, alpha, beta):
    A_, B_, E_ = np.exp(a), np.exp(b), np.exp(e)
    exp = beta / (alpha + beta)
    N = (alpha * A_ / (beta * B_)) ** (1 / (alpha + beta)) * (C / 6) ** exp
    D = C / (6 * N)
    L = E_ + A_ * N ** (-alpha) + B_ * D ** (-beta)
    return N, D, L

for C in [1e23, 1e24, 1e25]:
    N, D, L = optimal_alloc(C, e, a, b, alpha, beta)
    print(f"C={C:.0e} -> N*={N:.2e}  D*={D:.2e}  tok/param={D/N:.1f}  L*={L:.3f}")
```

Before the engineering notes, it is worth being honest about what this seed actually recovers: for `rng = np.random.default_rng(0)` the fit above returns approximately $E=1.83$, $A=879$, $B=471$, $\alpha=0.390$, $\beta=0.290$ against ground truth $E=1.69$, $A=406$, $B=410$, $\alpha=0.34$, $\beta=0.28$. If your run lands there too, that is expected — not a bug. $E$, $A$, and $B$ are strongly correlated and only weakly constrained by the data: the additive power-law surface is nearly flat near the optimum, so many $(E, A, B)$ triples fit the observed grid almost equally well, which is why $A$ can come out $2\times$ off. The exponents $\alpha, \beta$ — and especially the *allocation* exponent $\beta/(\alpha+\beta)$, recovered here as $0.426$ against a true $0.452$ — recover far more tightly, because it is the *slope* of the surface, not its offset, that the spread of runs actually pins down. The right accuracy check is therefore not the raw constants but (i) the implied allocation exponent and (ii) the extrapolated compute-optimal loss. This fit exhibits a classic compensating-error pattern: because $E$, $A$, $B$ trade off against each other, it interpolates the grid well, yet the extrapolated loss at $C=10^{25}$ comes out around $1.94$ nats versus the true $1.85$ — about $0.1$ nats high — while the allocation $N^\star(C)$, which depends only on the well-identified exponent ratio, tracks the truth far better than the absolute loss *level* does. This is not a toy artifact: Epoch AI's replication (Besiroglu et al., 2024, *"Chinchilla Scaling: A replication attempt"*) found the original Chinchilla parametric estimates were fragile for exactly this reason.

You can quantify the identifiability gap directly with a bootstrap over the fitting grid:

```python
import numpy as np

def fit_once(Nx, Dx, Lx, n_starts=15, seed=0):
    rng_local = np.random.default_rng(seed)
    best, best_val = None, np.inf
    for _ in range(n_starts):
        x0 = np.array([rng_local.uniform(0.0, 1.0), rng_local.uniform(4.0, 8.0),
                        rng_local.uniform(4.0, 8.0), rng_local.uniform(0.1, 0.6),
                        rng_local.uniform(0.1, 0.6)])
        res = minimize(lambda th: np.sum(huber(predict_log_loss(th, Nx, Dx) - np.log(Lx))),
                        x0, method="L-BFGS-B",
                        bounds=[(-2, 2), (0, 12), (0, 12), (0.01, 1.0), (0.01, 1.0)])
        if res.fun < best_val:
            best, best_val = res.x, res.fun
    return best

boot_rng = np.random.default_rng(2)
n_boot, n_rows = 200, len(N_obs)
records = []
for _ in range(n_boot):
    idx = boot_rng.integers(0, n_rows, n_rows)          # resample WITH replacement
    e_b, a_b, b_b, alpha_b, beta_b = fit_once(N_obs[idx], D_obs[idx], L_obs[idx])
    _, _, L25 = optimal_alloc(1e25, e_b, a_b, b_b, alpha_b, beta_b)
    records.append((alpha_b, beta_b, beta_b / (alpha_b + beta_b), np.exp(a_b), L25))
records = np.array(records)
names = ["alpha", "beta", "alloc_exp (beta/(a+b))", "A", "L*(1e25)"]
for j, name in enumerate(names):
    lo, mid, hi = np.percentile(records[:, j], [2.5, 50, 97.5])
    print(f"{name:>24}: [{lo:.3g}, {mid:.3g}, {hi:.3g}]  (2.5% / median / 97.5%)")
# Expect: alpha ~ [0.30, 0.45], beta ~ [0.25, 0.34], alloc-exp ~ [0.38, 0.50] -- tight.
# A spans roughly [230, 2300] and E ~ [1.57, 1.95] -- an order of magnitude wider,
# visually confirming allocation is identifiable but the raw offsets are not.
```

A few engineering notes that separate a real fit from a toy one:

- **Use enough points and enough spread.** Chinchilla's IsoFLOP method runs many models at each of several fixed compute budgets, then fits a parabola in $\log N$ to find the valley (the optimal $N$) at each budget, and finally fits a power law through those valleys. Our parametric fit above is the third of Chinchilla's three methods.
- **Exclude under-converged runs.** A run whose LR schedule did not finish, or that hit a loss spike (see [Training Stability, Loss Spikes & Debugging Large Runs](../03-pretraining/11-training-stability.html)), pollutes the fit. This is exactly the failure mode that biased Kaplan.
- **Sanity-check extrapolation, not interpolation.** The whole value is predicting *outside* your grid. Hold out your largest run, fit on the rest, and verify the prediction lands within a percent or two.

### The IsoFLOP Method (Chinchilla Approach 2)

The parametric fit above is Chinchilla's Approach 3 — fit the full surface $L(N,D)$ and differentiate. Approach 2, the **IsoFLOP method**, is more robust to misspecification of that parametric form because it never commits to it directly. Instead, at each of several *fixed* compute budgets $C$ you sweep $N$ (forcing $D = C/(6N)$ so every run in the slice costs exactly $C$), fit a quadratic (parabola) in $\log N$ to the resulting losses, and read off the valley — the loss-minimizing $N_{\text{opt}}(C)$ — for that slice. Repeat across several budgets and fit a power law through the valleys to recover the allocation exponent $a$ directly. This is the isoFLOP-slice picture illustrated in {{fig:scaling-kaplan-vs-chinchilla}} above.

{{fig:isoflop-valley-method}}

Grid design matters: use 4–6 fixed-$C$ slices spaced roughly half a decade apart, and at least 6 models per slice spanning about an order of magnitude in $N$, centered on the expected optimum $\sqrt{C/120}$ — a parabola needs points bracketing the minimum on *both* arms, or the fitted vertex is biased.

```python
import numpy as np

TRUE = dict(E=1.69, A=406.0, B=410.0, alpha=0.34, beta=0.28)
def true_loss(N, D, p=TRUE):
    return p['E'] + p['A']*N**(-p['alpha']) + p['B']*D**(-p['beta'])

rng = np.random.default_rng(1)
C_slices = np.array([1e18, 3e18, 1e19, 3e19, 1e20])   # 5 isoFLOP budgets
N_opt = []
for C in C_slices:
    N_center = np.sqrt(C / 120.0)                     # ~20x-rule optimum for slice
    Ns = N_center * np.logspace(-0.6, 0.6, 7)         # 7 models, span ~1.5 dex
    Ds = C / (6.0 * Ns)                               # D fixed so 6*N*D == C
    Ls = true_loss(Ns, Ds) * (1.0 + 0.01*rng.standard_normal(len(Ns)))
    c2, c1, c0 = np.polyfit(np.log(Ns), Ls, 2)        # parabola in log N
    logN_star = -c1 / (2.0 * c2)                      # vertex = valley
    N_opt.append(np.exp(logN_star))
N_opt = np.array(N_opt)
D_opt = C_slices / (6.0 * N_opt)
for C, N, D in zip(C_slices, N_opt, D_opt):
    print(f'C={C:.0e}  N_opt={N:.3e}  D_opt={D:.3e}  tok/param={D/N:.0f}')

a, _ = np.polyfit(np.log(C_slices), np.log(N_opt), 1)  # log N_opt = a*log C + k
print(f'IsoFLOP exponent a (N_opt ~ C^a) = {a:.3f}')
print(f'parametric beta/(alpha+beta)      = {TRUE["beta"]/(TRUE["alpha"]+TRUE["beta"]):.3f}')
```

The recovered exponent prints $a \approx 0.447$, agreeing to within about 0.005 of the analytic $\beta/(\alpha+\beta) = 0.452$ that the parametric fit targets — two independent methods landing on the same allocation slope. A caution: the extracted $N_{\text{opt}}$ is only as good as the parabola fit through it — too few points, or points sitting all on one arm of the valley, and the vertex estimate is badly biased, which is why you want at least 6 points bracketing the minimum in every slice.

---

## Designing the Sweep Under a Budget

Before you can fit anything you must decide which runs to launch. A good sweep spends roughly 1–2% of the FLOPs of the final run it is meant to inform, and it brackets the compute-optimal valley at several scales rather than clustering around a single guess.

Worked example: you have a $2\times10^{21}$-FLOP final run planned and a 1–2% experiment budget. The following generates a concrete run table (verified: it produces 24 runs totalling $2.64\times10^{19}$ FLOPs, i.e. 1.32% of the target):

```python
import numpy as np

C_final = 2e21                          # the run the sweep is designed to inform
slices = [1e17, 3e17, 1e18, 3e18]       # 4 isoFLOP budgets, ~0.5 dex apart
models_per_slice = 6
rows = []
for Cs in slices:
    N_star = np.sqrt(Cs / 120.0)                       # 20x-rule optimum for slice
    Ns = N_star * np.logspace(-0.5, 0.5, models_per_slice)  # span ~1 dex in N
    for N in Ns:
        D = Cs / (6.0 * N)                             # tokens for this run
        rows.append(dict(C=Cs, N=N, D=D, tpp=D/N, flops=Cs))  # per-run FLOPs == Cs
total = sum(r['flops'] for r in rows)
print(f'{len(rows)} runs, total = {total:.2e} FLOPs = {100*total/C_final:.2f}% of final')
for r in rows[:6]:
    print(f"  C={r['C']:.0e}  N={r['N']:.2e}  D={r['D']:.2e}  tok/param={r['tpp']:.0f}")
```

4 slices $\times$ 6 models = 24 runs. Every run inside a slice costs exactly that slice's budget $C$ (because $D = C/6N$ keeps $6ND = C$ fixed), so the whole sweep costs $6\sum_{\text{slices}} C_s = 2.64\times10^{19}$ FLOPs — 1.32% of the $2\times10^{21}$ target, comfortably inside the 1–2% envelope, and you verify it is by construction rather than by hoping. The $N$ grid spans tokens/param from roughly 200 down to 2 within each slice, deliberately bracketing the ~20 optimum so the IsoFLOP parabola has points on both arms of the valley.

A per-run hyperparameter note: tune learning rate and batch size *once*, at the smallest width in the sweep, and transfer with $\mu$P (maximal-update parameterization makes the optimal peak LR approximately scale-invariant, so you can read it straight off the cheapest run) — or, failing that, fall back to the empirical rules from [Learning Rate Schedules, Warmup, Batch Size & Hyperparameters](../03-pretraining/10-lr-schedules-hparams.html): peak LR shrinking roughly as $1/\text{width}$, batch size set from the critical-batch-size heuristic and growing as the loss falls. Crucially, every run's cosine schedule must decay to *its own* token endpoint $D$, never a shared long schedule — that is the exact Kaplan confound from earlier, now a checklist item rather than a war story.

**Designing your sweep — checklist:**

1. At least 2 orders of magnitude of spread in $N$ across the whole sweep.
2. At least 5–6 models per isoFLOP slice, bracketing the expected optimum on both sides.
3. 4–6 slices, spaced roughly half a decade apart.
4. Per-run cosine decay matched to that run's own token count — not a shared schedule.
5. Fixed data distribution and tokenizer across *all* runs.
6. A few extra seeds at the smallest scale, to estimate the noise floor (you need this to set the Huber delta and to know whether two runs really differ).
7. Hold out your largest run(s) for extrapolation validation rather than folding them into the fit.

Finally, checkpoint that last item against the fit from the previous section — refit with the largest-$N$ runs removed and see how well the held-out point is recovered:

```python
mask = N_obs != 3e9                      # hold out the largest-N runs
# (continues the fitting-block variables N_obs, D_obs, L_obs, objective, minimize, huber)
held_N, held_D, held_L = N_obs[~mask], D_obs[~mask], L_obs[~mask]
theta_ho = fit_once(N_obs[mask], D_obs[mask], L_obs[mask])
pred_L = np.exp(predict_log_loss(theta_ho, held_N, held_D))
err_pct = 100 * np.abs(pred_L - held_L) / held_L
print(f"held-out predicted vs observed: {list(zip(pred_L.round(3), held_L.round(3)))}")
print(f"held-out error: {err_pct.round(2)}%")
```

Expect the held-out loss recovered within roughly 1–2% — that is the real success criterion for a scaling-law fit: extrapolation, not interpolation.

---

## A Worked Budget-Planning Example

Let us plan an actual run end to end.

!!! example "Planning a compute-optimal run on a fixed budget"
    **Setup.** You have 256 accelerators, each with a peak of roughly $1.0 \times 10^{15}$ bf16 FLOP/s (1 PFLOP/s — in the ballpark of a modern training GPU). You budget **21 days** of wall-clock for the run and expect a Model FLOPs Utilization (MFU) of **0.45**. How big a model should you train, and on how many tokens?

    **Step 1 — Total compute budget.**

    Raw machine-seconds of FLOPs:
    $$
    256 \times 10^{15}\,\tfrac{\text{FLOP}}{\text{s}} \times (21 \times 86400\,\text{s}) \approx 4.64 \times 10^{23}\ \text{FLOP (peak)}
    $$
    Apply MFU = 0.45 to get *usable* model FLOPs:
    $$
    C \approx 0.45 \times 4.64 \times 10^{23} \approx 2.1 \times 10^{23}\ \text{FLOP}.
    $$

    **Step 2 — Compute-optimal split via the 20× rule.**

    At the optimum, $D \approx 20 N$ and $C = 6ND$, so:
    $$
    C = 6 N (20 N) = 120 \, N^2
    \;\Longrightarrow\;
    N = \sqrt{\frac{C}{120}} = \sqrt{\frac{2.1\times 10^{23}}{120}} \approx 4.2 \times 10^{10}.
    $$
    So $N \approx 42\text{B}$ parameters, and:
    $$
    D \approx 20 N \approx 8.4 \times 10^{11} = 840\text{B tokens}.
    $$

    **Step 3 — Sanity-check the FLOPs.** $6 N D = 6 \times 4.2\times10^{10} \times 8.4\times10^{11} \approx 2.1 \times 10^{23}$ FLOP. It closes.

    **Step 4 — Cross-check the throughput.** Tokens per second = $\dfrac{C/6}{N \times \text{seconds}}$. With usable FLOP/s $= 0.45 \times 256 \times 10^{15} \approx 1.15\times 10^{17}$, token throughput $= \frac{1.15\times10^{17}}{6 \times 4.2\times10^{10}} \approx 4.6\times10^{5}$ tokens/s $\approx$ 460k tok/s, which over 21 days yields $\approx 8.3\times10^{11}$ tokens — consistent with $D$. Everything is self-consistent.

    **Takeaway.** A 21-day, 256-GPU run at 45% MFU buys you roughly a **42B-parameter Chinchilla-optimal model on ~840B tokens**. If instead you only had ~300B tokens of acceptable-quality data, you would be *data-limited*: you should train a smaller model (or repeat data, with care — see below) rather than a 42B model starved of tokens.

This is the arithmetic that frontier labs run before every campaign. Notice it has exactly four inputs — GPU count, peak FLOP/s, wall-clock, and MFU — and the 20× rule. You can do it on a napkin.

!!! note "Is my real run on-curve? Loss milestones"
    The toy check -- first loss ~ ln(V) at init -- from [The Pretraining Objective & Loss](../03-pretraining/03-pretraining-objective.html) tells you training *started* correctly. To know a real BPE run is *on-curve*, anchor to these order-of-magnitude held-out targets at the Chinchilla-optimal point (English web text, GPT-2-style BPE at ~4 bytes/token; exact numbers depend on tokenizer and corpus, so treat as ballparks, not pass/fail):

    - **125M params, ~2.5B tokens (~20 tok/param):** ~3.2-3.4 nats/token held-out, BPB ~1.0-1.2. For reference, a 124M GPT-2 pushed to ~300B tokens (heavily *over*-trained) reaches ~2.85 nats/token on OpenWebText -- the Chinchilla-optimal point is higher because it sees far fewer tokens.
    - **1B params, ~20B tokens (~20 tok/param):** ~2.6-2.8 nats/token, BPB ~0.85-0.95.
    - **125M *over-trained* to ~15B tokens (~120 tok/param):** ~3.0 nats/token -- a real but small gain over the 20x point, exactly the diminishing return the flat loss surface predicts.

    Convert with BPB ~ (nats/token) / (ln 2 * bytes-per-token). If your run sits >0.3 nats above these at the same (N, D), suspect an LR-schedule (decay not matched to token count), data-pipeline, or tokenizer-mismatch bug before blaming the architecture.

---

## Beyond Chinchilla: Inference, Over-Training & Data Repetition

Chinchilla answers "what minimizes *training* loss for a *training* budget." But that is almost never the real objective. Two large corrections dominate modern practice.

### Inference-aware scaling: why you deliberately over-train

{{fig:inference-aware-overtraining}}

Chinchilla optimizes training compute. But a deployed model is *trained once and served billions of times*. If you will serve a huge number of inference tokens, it is rational to **shrink the model below Chinchilla-optimal and train it on far more data than 20× tokens** — accepting a slightly higher training loss in exchange for a permanently cheaper, faster model at inference (every forward pass costs $\approx 2N$ FLOPs and the KV cache scales with $N$; see [The Anatomy of LLM Inference: Prefill, Decode & The KV Cache](../07-inference-serving/01-anatomy-inference.html)).

This is exactly why **Llama-style** models are trained on token counts vastly exceeding 20× their parameter count. Llama-2-7B saw ~2T tokens (~280 tokens/param); Llama-3-8B saw ~15T tokens (~1900 tokens/param) — roughly **two orders of magnitude past Chinchilla-optimal**. These models are *worse than Chinchilla-optimal for their training FLOPs* but *far better per inference FLOP*, which is what their deployers actually pay for.

You can formalize this. Let $D_{\text{inf}}$ be the expected number of tokens you will serve. The objective becomes total lifetime compute:

$$
C_{\text{total}} = \underbrace{6 N D_{\text{train}}}_{\text{train once}} + \underbrace{2 N D_{\text{inf}}}_{\text{serve forever}}
$$

minimized subject to hitting a *target loss* $L^\star$. The larger $D_{\text{inf}}$ is, the more the optimum shifts toward small $N$ (cheap inference) and large $D_{\text{train}}$ (to recover the loss you gave up by shrinking $N$). This "inference-aware" or "over-trained" regime was analyzed in follow-up work (e.g., the *Beyond Chinchilla-Optimal* line of analysis by Sardana, Frankle, and collaborators).

```python
def lifetime_optimal(L_target, n_inference_tokens,
                     A=406.0, B=410.0, alpha=0.34, beta=0.28, E=1.69):
    """Pick (N, D_train) to hit a target loss while minimizing TRAIN+INFERENCE
    compute. Coarse grid search over N; for each N solve the data term needed
    to reach L_target, then score total lifetime FLOPs.
    """
    import numpy as np
    best = None
    for N in np.logspace(8, 12, 400):          # 0.1B .. 1T params
        # residual loss budget left for the data term after the param term:
        loss_from_data = L_target - E - A * N ** (-alpha)
        if loss_from_data <= 0:
            continue                            # this N alone already over-shoots
        D_train = (B / loss_from_data) ** (1.0 / beta)
        train = 6 * N * D_train
        infer = 2 * N * n_inference_tokens
        total = train + infer
        if best is None or total < best["total"]:
            best = dict(N=N, D_train=D_train, train=train,
                        infer=infer, total=total, tok_per_param=D_train / N)
    return best

# Light serving vs. heavy serving -> the optimal model SHRINKS as you serve more:
for D_inf in [1e11, 1e13, 1e15]:
    r = lifetime_optimal(L_target=1.95, n_inference_tokens=D_inf)
    print(f"D_inf={D_inf:.0e}: N={r['N']:.2e}  D_train={r['D_train']:.2e}  "
          f"tok/param={r['tok_per_param']:.0f}")
```

Run it and you will see the optimal $N$ *decrease* and tokens-per-param *increase* as inference demand grows — the quantitative justification for over-training.

### Data repetition and the data wall

The Chinchilla form assumes every token is fresh. Real high-quality data is finite — the web has only so much good text. When you exhaust unique tokens, you start **repeating epochs**, and repeated tokens are worth *less* than fresh ones. Muennighoff et al. (*Scaling Data-Constrained Language Models*) fit a modified law in which the effective data $D_{\text{eff}} < D$ once you repeat, with the value of each additional epoch decaying. The empirical rule of thumb from that work: up to **~4 epochs** of repetition costs you very little (repeated tokens are nearly as good as fresh), but beyond ~16 epochs the returns collapse toward zero. This "data wall" is a central reason the field cares so much about data quality, dedup, and synthetic data — see [Data Cleaning, Deduplication & Quality Filtering](../03-pretraining/02-data-cleaning-dedup.html).

!!! tip "Practitioner tip: decide your regime before you fit anything"
    Before applying any scaling law, classify your situation. **Compute-limited** (plenty of data, limited FLOPs): use Chinchilla, target ~20 tokens/param. **Inference-limited** (you will serve a lot): over-train deliberately, push tokens/param up by 10–100×, shrink $N$. **Data-limited** (good tokens are the bottleneck): use the data-constrained law, cap repetition near a few epochs, and invest in data quality and synthetic data instead of more raw scrape. Most production teams are in the second or third regime, *not* the first — yet the first is what the textbook "20× rule" assumes.

---

## The Emergent Abilities Debate

{{fig:emergent-mirage-metric}}

Scaling laws describe the *loss* — a smooth, continuous quantity. But practitioners care about **capabilities**: can the model do 3-digit arithmetic, pass a coding test, follow multi-step instructions? A celebrated and contested claim is that some capabilities are **emergent** — absent in small models, then appearing *abruptly* past a scale threshold, producing a sharp "phase transition" rather than a smooth curve (Wei et al., *Emergent Abilities of Large Language Models*).

The counter-argument, by Schaeffer et al. (*Are Emergent Abilities of Large Language Models a Mirage?*), is sharp and worth internalizing because it is a favorite interview trap. Their claim: **the apparent emergence is largely an artifact of the metric**, not of the model. Many benchmarks use **discontinuous or all-or-nothing scoring** — exact-match accuracy on a multi-step problem, where you get credit only if *every* token is right. Under such a metric, a smoothly improving per-token loss translates into a sharp jump in the discontinuous score:

- If a task needs $k$ tokens all correct, and per-token error probability falls smoothly with scale, then exact-match $\approx (1 - p_{\text{err}})^k$ — which stays near zero, then rises steeply once $p_{\text{err}}$ crosses a threshold. The *underlying* improvement was smooth all along; the metric *created* the cliff.

Swap the discontinuous metric for a smooth one — token-level edit distance, per-token accuracy, or a calibrated log-likelihood — and many "emergent" curves straighten out into the same gentle power-law improvement that the loss shows.

```python
import numpy as np

def smooth_per_token_accuracy(N):
    """Per-token correctness improves SMOOTHLY (power-law) with scale."""
    # Clip to a valid probability: the raw power law goes negative below
    # N ~ 4.2e8, so clamp to [0, 1]. Per-token correctness still improves
    # SMOOTHLY (power-law) with scale.
    return np.clip(1.0 - 0.9 * (N / 1e9) ** (-0.12), 0.0, 1.0)

def exact_match(N, k_tokens):
    """All-or-nothing metric: need ALL k tokens correct simultaneously."""
    p = np.clip(smooth_per_token_accuracy(N), 1e-9, 1 - 1e-9)
    return p ** k_tokens

Ns = np.logspace(8, 12, 9)
print(f"{'N':>10} {'per-token':>10} {'EM(k=1)':>9} {'EM(k=30)':>10}")
for N in Ns:
    print(f"{N:10.1e} {smooth_per_token_accuracy(N):10.3f} "
          f"{exact_match(N,1):9.3f} {exact_match(N,30):10.3f}")
# per-token rises gently; EM(k=30) stays ~0 then "emerges" sharply -- same model!
```

The synthesis most researchers now hold: **emergence is real as a phenomenon of how we measure and use models** (a model genuinely *can* suddenly do a multi-step task once its per-step reliability crosses a threshold), but it is **not a discontinuity in the underlying learning** — the loss was improving smoothly the whole time. For *planning*, the takeaway is reassuring: you can predict loss reliably from scaling laws, but you **cannot** reliably predict the exact scale at which a specific downstream capability will "click," because that depends on the metric's threshold and the task's token-length. Predict loss; treat capability thresholds as uncertain.

!!! warning "Don't over-extrapolate a single benchmark"
    Because downstream metrics are noisy and threshold-sensitive, never plan a multi-million-dollar run around the promise that "ability X emerges at scale Y." Loss extrapolates; capabilities are lumpy. Validate capability claims with smooth proxy metrics (per-token accuracy, Brier score, log-likelihood of correct answers) and several seeds before betting the budget.

---

## Limits, Caveats & The Frontier

Scaling laws are powerful but not omniscient. Keep these failure modes in mind:

- **Constants are not universal.** $E$, $A$, $B$ depend on the tokenizer, the data distribution, and the eval set. Refit for *your* corpus; do not import another lab's coefficients. The *exponents* transfer better than the *offsets*.
- **Architecture changes shift the offset.** Better architectures (SwiGLU, RoPE, good norm placement — see [The Transformer Block: Norms, Residuals, MLPs & Activations](../02-transformer/06-transformer-block.html)) and better data move the curve *down* without changing the slope much. That vertical shift is "free loss," and it compounds: a one-time architectural win is equivalent to a constant multiplier on compute forever.
- **MoE rewrites the FLOP accounting.** With sparse experts the relevant $N$ for FLOPs is *active* params; scaling laws have been re-derived for MoE in terms of active params, total params, and the number of experts. The "20× rule" does not transfer unmodified — see [Mixture-of-Experts (MoE) Architectures](../02-transformer/09-mixture-of-experts.html).
- **Hyperparameters must scale too.** Optimal learning rate, batch size, and warmup all drift with scale. A scaling law fit on mistuned runs is worthless — this was Kaplan's lesson. Techniques like $\mu$P (maximal-update parameterization) aim to make the optimal LR *scale-invariant* so you can tune small and transfer; see [Learning Rate Schedules, Warmup, Batch Size & Hyperparameters](../03-pretraining/10-lr-schedules-hparams.html).
- **Test-time compute is a new axis.** Recent work shows you can buy capability by spending more *inference* compute (longer chains of thought, search, self-consistency) instead of more *training* compute — a different scaling curve entirely. See [Reasoning, Chain-of-Thought & Test-Time Compute](../05-posttraining-alignment/10-reasoning-test-time-compute.html). The frontier question of 2025–2026 is how to *jointly* optimize pretraining, post-training, and test-time compute under one budget.
- **The data wall is real.** High-quality human text is finite. Past a few epochs, repetition stops helping, which is why synthetic data, multimodal data, and curation are now first-class scaling levers.

The grand picture: scaling laws turned LLM development from alchemy into engineering. They are the reason a lab can commit hundreds of millions of dollars to a training run *before* it starts and be confident about the loss it will hit. But they predict *loss*, and loss is a proxy. The art that remains is choosing the right regime (compute-, inference-, or data-limited), refitting the constants for your own setup, and remembering that the cliff in your benchmark might be a feature of your ruler, not your model.

---

## Exercises

1. **Extrapolation, not interpolation.** Take the synthetic grid from *Fitting a Scaling Law From Scratch*, hold out the single largest run ($N=3\times10^9$, $D=10^{11}$), refit on the remaining 41 rows, and predict that held-out loss. Verify the prediction lands within ~2% of the observed value. Then use the refit to predict the compute-optimal $(N^\star, D^\star)$ at $10\times$ the largest grid budget and report tokens/param.
2. **Reproduce the Kaplan bias.** Deliberately include under-converged runs: multiply the loss of every run with $D \geq 10^{10}$ by 1.03 (simulating a cosine schedule that never finished decaying at high token counts), refit, and report how $\alpha$ and $\beta$ shift. Explain in one sentence why inflating the high-token losses biases the fitted allocation toward larger $N$ / fewer tokens — the exact confound Chinchilla diagnosed.
3. **Loss penalty of a mis-sized model.** Fix a compute slice $C=10^{20}$. Using the ground-truth constants ($E=1.69$, $A=406$, $B=410$, $\alpha=0.34$, $\beta=0.28$), compute the optimal $N^\star$ (you should get $N^\star \approx 6.5\times10^8$, $D^\star \approx 2.6\times10^{10}$, ~40 tok/param) and the loss there, then the loss at $2\times$ and $0.5\times$ $N^\star$ (with $D=C/6N$ each). You should find a penalty of only ~0.02 nats in each direction, roughly symmetric in $\log N$ — the reason IsoFLOP valleys are shallow and $N^\star$ is only weakly identified.
4. **Full loop (capstone).** Design a sweep for a $10^{21}$-FLOP target run under a ~1% experiment budget (choose your slices and models/slice, and verify the total), fit BOTH the IsoFLOP method and the parametric method on synthetic data generated from the constants above, and confirm the two recovered allocation exponents agree to within 0.03.

---

## Key Takeaways

!!! key "Key Takeaways"
    - Language-model loss follows clean **power laws** in parameters and data: $L(N,D) = E + A N^{-\alpha} + B D^{-\beta}$, with a small irreducible floor $E$ (the entropy of text) and small exponents, so returns diminish but never vanish.
    - Training compute for a dense transformer is **$C \approx 6ND$** (2 forward + 4 backward FLOPs per parameter per token); inference is $\approx 2ND$. Convert to wall-clock/dollars via **MFU** (typically 0.3–0.55).
    - **Kaplan (2020)** concluded "grow the model fast" ($N \propto C^{0.73}$); **Chinchilla (2022)** corrected a learning-rate-schedule confound and found you should **scale $N$ and $D$ together** ($a \approx b \approx 0.5$).
    - The famous heuristic is **$\approx 20$ tokens per parameter** at the compute-optimal point — it falls out of $\alpha \approx \beta$ in the Lagrange-multiplier optimization, and Chinchilla beat the 4×-larger Gopher to prove it.
    - **Fit scaling laws in log space with a robust (Huber) loss**, exclude under-converged runs, and validate by *extrapolating* to a held-out large run — not just interpolating.
    - **Inference-aware over-training** (Llama-style: hundreds to thousands of tokens/param) is rational when you will serve the model heavily: accept higher training loss for a permanently cheaper, smaller model.
    - The **data wall** means repeated tokens are worth less; up to ~4 epochs is roughly free, beyond ~16 epochs returns collapse — so data quality and synthetic data become scaling levers.
    - **Emergent abilities** are largely an artifact of discontinuous metrics: loss improves smoothly, but all-or-nothing scores jump. Predict loss reliably; treat capability thresholds as uncertain.

!!! sota "State of the Art & Resources (2026)"
    Scaling laws remain the foundational planning tool for frontier LLM training, with the field actively extending them into the inference and test-time compute regimes. The Chinchilla "20 tokens per parameter" rule is now understood as one point on a spectrum: real deployments routinely over-train by 100–2000× for inference efficiency, and 2024–2026 research is beginning to unify pre-training and test-time compute into a single joint budget.

    **Foundational work**

    - [Kaplan et al., *Scaling Laws for Neural Language Models* (2020)](https://arxiv.org/abs/2001.08361) — the original paper establishing power-law loss curves across seven orders of magnitude of compute.
    - [Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla, 2022)](https://arxiv.org/abs/2203.15556) — corrected the LR-schedule confound; established the equal-exponent result and the ~20 tokens/param rule.
    - [Sharma & Kaplan, *A Neural Scaling Law from the Dimension of the Data Manifold* (2020)](https://arxiv.org/abs/2004.10802) — first-principles derivation of why loss follows a power law, via data-manifold intrinsic dimension.
    - [Henighan et al., *Scaling Laws for Autoregressive Generative Modeling* (2020)](https://arxiv.org/abs/2010.14701) — shows the same power-law form holds across image, video, math, and multimodal domains.

    **Recent advances (2023–2026)**

    - [Sardana et al., *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws* (2024)](https://arxiv.org/abs/2401.00448) — formalizes inference-aware over-training; shows that at ~1B inference requests, models should be trained far smaller and longer than Chinchilla prescribes.
    - [Muennighoff et al., *Scaling Data-Constrained Language Models* (NeurIPS 2023)](https://arxiv.org/abs/2305.16264) — modifies scaling laws for finite data; finds up to ~4 epochs of repetition is nearly free, beyond ~16 epochs returns collapse.
    - [Snell et al., *Scaling LLM Test-Time Compute Optimally* (2024)](https://arxiv.org/abs/2408.03314) — demonstrates test-time compute scaling can outperform a 14× larger model; introduces a new compute axis orthogonal to training scale.
    - [Schaeffer et al., *Are Emergent Abilities of Large Language Models a Mirage?* (NeurIPS 2023)](https://arxiv.org/abs/2304.15004) — shows that apparent capability phase-transitions are largely an artifact of discontinuous metrics, not discontinuities in the underlying loss.

    **Go deeper**

    - [Austin et al., *How to Scale Your Model* (Google DeepMind, 2025)](https://jax-ml.github.io/scaling-book/) — practical systems guide to scaling transformers on TPUs/GPUs, covering hardware, parallelism, and the engineering side of capacity planning.

## Further Reading

- Kaplan, McCandlish, Henighan, et al. *Scaling Laws for Neural Language Models*. arXiv 2020. (The origin of LLM scaling laws.)
- Hoffmann, Borgeaud, Mensch, et al. *Training Compute-Optimal Large Language Models* (Chinchilla). arXiv 2022. (The correction and the 20× rule.)
- Henighan, Kaplan, Katz, et al. *Scaling Laws for Autoregressive Generative Modeling*. arXiv 2020. (Scaling across modalities.)
- Sharma, Kaplan. *A Neural Scaling Law from the Dimension of the Data Manifold*. arXiv 2020. (A first-principles "why power laws" argument.)
- Sardana, Frankle, et al. *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws*. arXiv 2023. (Inference-aware over-training.)
- Muennighoff, Rush, Barak, et al. *Scaling Data-Constrained Language Models*. NeurIPS 2023. (The data wall and repetition.)
- Wei, Tay, Bommasani, et al. *Emergent Abilities of Large Language Models*. TMLR 2022.
- Schaeffer, Miranda, Koyejo. *Are Emergent Abilities of Large Language Models a Mirage?*. NeurIPS 2023.
- Hestness, Narang, Ardalani, et al. *Deep Learning Scaling is Predictable, Empirically*. arXiv 2017. (An early, prescient empirical scaling study.)
