# 3.14 Data Mixing, Domain Weighting & Curriculum

Two teams are handed the same compute budget, the same architecture, and access to the same pile of raw tokens: a few trillion tokens of web text, a few hundred billion of code, tens of billions of math, some books, and a multilingual long tail. Team A throws everything into one shuffled stream in proportion to how much of each they happen to have on disk. Team B spends a week deciding *what fraction of each batch* should be web, code, math, books, and other languages — and then schedules those fractions to change as training proceeds. At the end, Team B's model is meaningfully better at coding and reasoning, no worse at general language, and used the exact same number of FLOPs.

That gap is the subject of this chapter. Once you have cleaned and deduplicated your corpus (see [Data Cleaning, Deduplication & Quality Filtering](../03-pretraining/02-data-cleaning-dedup.html)) and decided how big a model your budget supports (see [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html)), you still face a deceptively deep optimization: the **data mixture**. What proportion of each domain do you sample? Do you upsample a small high-quality source or just deduplicate harder? Do you keep the mixture fixed, or change it over training — easy-to-hard curricula, context-window ramps, a high-quality "annealing" phase near the end? These choices routinely move benchmark scores by amounts that would otherwise cost you a 2x model-size increase.

We will develop the topic in four movements: (1) the **mixing problem** itself — what a mixture is, why it matters, and the upsampling-vs-dedup trade-off; (2) **how to choose weights** — manual ablations, and the proxy-model methods DoReMi and Group-DRO that learn weights automatically; (3) a **from-scratch DoReMi-style reweighting toy** you can run; and (4) **scheduling over time** — curriculum, context-length ramps, and mid-training annealing. The data pipeline that produces these domains is covered in [Pretraining Data: Sources, Crawling & The Data Pipeline](../03-pretraining/01-pretraining-data.html); here we assume the domains exist and ask how to *blend* them.

---

## The Mixing Problem: What a Mixture Is and Why It Matters

### Domains, weights, and the sampling distribution

Partition your corpus into $k$ **domains** (also called *groups* or *sources*) $\mathcal{D}_1, \dots, \mathcal{D}_k$ — for example web, code, math, books, encyclopedic, and a multilingual bucket. Each domain $i$ has a natural size $n_i$ (its token count after cleaning). A **mixture** is a probability vector

$$
w = (w_1, \dots, w_k), \qquad w_i \ge 0, \qquad \sum_{i=1}^{k} w_i = 1,
$$

where $w_i$ is the probability that the *next token drawn for a training batch* comes from domain $i$. Equivalently, over a run of $D$ total tokens, domain $i$ contributes $w_i D$ tokens of gradient signal. The mixture is a knob entirely separate from how much data you *have*: you can set $w_{\text{code}} = 0.20$ even if code is only 5% of your corpus, by **upsampling** (revisiting code tokens multiple times), or set $w_{\text{web}} = 0.40$ even though web is 80% of your corpus, by **downsampling** (skipping most web tokens).

Define the **natural** or **proportional** mixture as $w_i^{\text{nat}} = n_i / \sum_j n_j$ — sample each token uniformly from the pool. Team A above used $w^{\text{nat}}$. The central empirical fact of this chapter is that $w^{\text{nat}}$ is almost never optimal. Why? Because the loss you actually care about is not "loss on the corpus you happen to have"; it is loss on a *target distribution* of downstream uses, which weights code, math, and reasoning far more heavily than their raw token counts.

### The effective-epochs view

The cleanest way to think about a mixture is in terms of **epochs per domain**. If domain $i$ has $n_i$ unique tokens and you allocate it $w_i D$ training tokens, then domain $i$ is seen for

$$
e_i = \frac{w_i D}{n_i}
$$

epochs. A small high-quality domain (say 20 B tokens of curated math) inside a 2 T-token run with $w_{\text{math}} = 0.05$ is seen $e_{\text{math}} = (0.05 \times 2{,}000\text{B}) / 20\text{B} = 5$ times. Meanwhile web, with 4 T unique tokens and $w_{\text{web}} = 0.6$, is seen $e_{\text{web}} = (0.6 \times 2{,}000) / 4{,}000 = 0.3$ epochs — less than once. This asymmetry is the crux of every mixing decision: **upweighting a small domain forces you to repeat it**, and repetition has a cost. Past roughly 4 epochs of a domain, returns to repeated data decay sharply, and past a dozen or so, repeated data can actively hurt (memorization, reduced generalization). The Muennighoff et al. *"Scaling Data-Constrained Language Models"* work quantified this: repeated tokens are worth progressively less than fresh ones, with the value of an epoch decaying roughly geometrically.

So a mixture choice is implicitly an **epoch-budget choice**. You are deciding, for each domain, "how many times is it worth re-reading this?"

{{fig:mixing-epoch-budget-tree}}

### Upsampling vs. deduplication: the trade-off

A subtle but important interaction: **upsampling and deduplication push in opposite directions on the same axis — how many times a token is effectively seen.**

- **Deduplication** (covered in [Data Cleaning, Deduplication & Quality Filtering](../03-pretraining/02-data-cleaning-dedup.html)) removes *near-duplicate* documents so the model does not waste capacity memorizing boilerplate and does not silently see the same passage 50 times. It *reduces* effective epochs on duplicated content.
- **Upsampling** deliberately *increases* effective epochs on a chosen domain.

The trap is doing both blindly: if you under-deduplicate web and then also upsample a small domain, you can end up with a model that has seen the upsampled domain 5x (intended) but also seen the most common web boilerplate 50x (unintended). The principled recipe is: **deduplicate aggressively first, to make "one epoch" mean one genuine pass over unique content; then upsample deliberately, with full visibility into the resulting epoch counts.** Deduplication makes the epoch math honest; upsampling spends that honest budget where it helps.

There is also a quality-vs-quantity tension. Suppose you have a small, very high-quality math set. You can:

1. **Upsample** it (repeat it, raising $e_{\text{math}}$) — risks memorization but injects more high-quality gradient signal.
2. **Leave it at one epoch** and accept its small weight — safe but under-uses a great source.
3. **Generate synthetic data** in its style to enlarge the *unique* pool (see [Synthetic Data for Pre- and Post-Training](../03-pretraining/15-synthetic-data.html)) — best of both worlds when feasible, since fresh unique tokens beat repeats.

A common modern practice is a hybrid: upsample high-quality small domains to 2–4 epochs (where repetition is still net-positive), and use synthetic generation rather than pushing past ~4–6 epochs.

!!! warning "Common pitfall: upsampling before deduplicating"

    If you upsample a domain that still contains internal near-duplicates, you multiply the duplication. A 3x upsample of a set that already contains 4x internal duplication of some passages yields 12 effective views of those passages — enough to trigger verbatim memorization (a privacy and generalization problem; see [Privacy, Memorization & Differential Privacy for LLMs](../13-interp-safety-gov/03-privacy-memorization-dp.html)). Always deduplicate *within* a domain before deciding its upsample factor.

{{fig:mixing-dedup-upsample-multiply}}

### Why the mixture is not separable from everything else

Two important couplings make mixing harder than "tune five numbers once":

- **Mixture interacts with model scale.** The *optimal* mixture is not scale-invariant. Small models are capacity-limited and benefit from a "cleaner," narrower diet; large models can absorb a more diverse, heavier-tailed mixture and turn it into capability. A mixture tuned on a 100 M-parameter proxy may be subtly wrong for a 70 B target. This is the central risk of all proxy-based methods, and we return to it below.
- **Mixture interacts with the schedule.** A *fixed* mixture is a special case of a *schedule*. As we will see in the curriculum section, the best results often come from changing $w$ over time — e.g., more diverse/noisier data early, then concentrating high-quality math/code/instruction-like data in a final annealing phase.

---

## Choosing the Weights I: Manual Ablations and the Target-Loss View

### The objective: minimize a target-weighted loss

Make the goal explicit. You have a **target distribution** over domains $p^{\star} = (p_1^{\star}, \dots, p_k^{\star})$ encoding how much you *care* about each — perhaps uniform over domains (every domain equally important), or skewed toward code/math if that is your product, or matched to a downstream eval suite. Your true objective is to minimize the target-weighted held-out loss:

$$
\mathcal{L}_{\text{target}}(\theta) = \sum_{i=1}^{k} p_i^{\star} \, \ell_i(\theta), \qquad \ell_i(\theta) = \mathbb{E}_{x \sim \mathcal{D}_i}\big[-\log p_\theta(x)\big],
$$

where $\ell_i$ is the per-domain held-out cross-entropy (in nats/token; see [The Pretraining Objective & Loss](../03-pretraining/03-pretraining-objective.html)). Crucially, the *training* mixture $w$ is a separate object from the *target* weights $p^{\star}$. You choose $w$ — the sampling distribution — to minimize $\mathcal{L}_{\text{target}}$. Setting $w = p^{\star}$ is the naive guess, and it is usually wrong: domains differ in difficulty and in how much they transfer to one another, so the training weights that minimize a target loss are generally *not* equal to the target weights themselves. Easy domains need less weight to reach low loss; hard or high-transfer domains may deserve more.

### Manual mixture ablations: the workhorse

Before any fancy method, the industry standard is **mixture ablation at small scale**: train many small models (say 100 M–1 B parameters, for a few billion tokens each) under different candidate mixtures, evaluate each on a fixed held-out suite, and pick the winner — then *scale it up* and hope the ranking holds. The Llama, GPT-3, and Gopher reports all describe variants of this. The procedure:

1. **Fix a proxy scale** small enough to run dozens of configs cheaply but large enough to be predictive (a few hundred million parameters is typical).
2. **Define a small set of candidate mixtures** — e.g., a grid or a few hand-designed points: "web-heavy," "code-heavy," "balanced," "math-upsampled."
3. **Train one proxy per mixture** for a fixed token budget.
4. **Evaluate** each on per-domain held-out loss and on a basket of downstream tasks.
5. **Pick** the mixture optimizing your target metric, then **scale**, ideally re-checking the ranking at one intermediate scale.

This works and is robust, but it is expensive (cost grows linearly in the number of mixtures tried) and it explores only the handful of points you thought to try. It also bakes in the scale-transfer assumption. The methods in the next section automate the search and, in DoReMi's case, find the weights with a *single* extra proxy run instead of a grid.

!!! example "Worked example: setting weights from an epoch budget"

    You have a 1 T-token training budget and these cleaned, deduplicated pools: web 3 T, code 250 B, math 30 B, books 80 B, multilingual 400 B (total 3.76 T unique tokens). You decide on these target epoch counts based on quality and repetition tolerance: web 0.25 epochs (it is plentiful and lower-value per token), code 1.5, math 4, books 2, multilingual 0.6.

    Tokens allocated per domain $= e_i \times n_i$:

    - web: $0.25 \times 3000 = 750$ B
    - code: $1.5 \times 250 = 375$ B
    - math: $4 \times 30 = 120$ B
    - books: $2 \times 80 = 160$ B
    - multilingual: $0.6 \times 400 = 240$ B

    Total $= 1645$ B, but our budget is 1000 B. So normalize: scale every allocation by $1000/1645 = 0.608$. The resulting **mixture weights** $w_i = (\text{alloc}_i \times 0.608)/1000$:

    - web: $0.456$, code: $0.228$, math: $0.073$, books: $0.097$, multilingual: $0.146$.

    Sanity check: they sum to $1.0$. Note code, at 6.6% of unique tokens, gets **22.8%** of training weight — a 3.4x upweight — while web, 80% of unique tokens, gets only 45.6%. The realized epochs after normalization are $0.608\times$ the targets: web 0.15, code 0.91, math 2.4, books 1.2, multilingual 0.36 — all comfortably under the danger zone for repetition. This back-of-envelope is exactly how practitioners turn "how much do I trust each source and how often can I repeat it" into concrete sampling probabilities.

---

## Choosing the Weights II: Proxy-Model Methods (DoReMi & Group-DRO)

Manual ablation searches a few points. **DoReMi** (Domain Reweighting with Minimax Optimization; Xie et al., 2023) instead *learns* a mixture in one shot, using a small **reference model** and a small **proxy model**, then transfers the learned weights to the large target run. It is built on **Group Distributionally Robust Optimization (Group-DRO)**, so we develop that first.

### Distributionally Robust Optimization in one paragraph

Ordinary empirical risk minimization (ERM) minimizes the *average* loss over the training distribution. **DRO** instead minimizes the *worst-case* loss over a set of distributions — it is risk-averse, optimizing for the hardest reweighting an adversary could pick. In the **group** version, the adversary is restricted to reweighting the $k$ domains:

$$
\min_{\theta} \; \max_{w \in \Delta_k} \; \sum_{i=1}^{k} w_i \, \ell_i(\theta),
$$

where $\Delta_k$ is the probability simplex over domains. The inner $\max$ puts all weight on whichever domain currently has the highest loss; the outer $\min$ trains $\theta$ to bring that worst domain down. The fixed point is a model that is *uniformly good across domains* — no domain is left behind. This is exactly the property we want from a pretraining mixture: do not let math or low-resource languages collapse just because they are small.

### The DoReMi trick: excess loss, not raw loss

A naive Group-DRO objective has a problem for pretraining: some domains are *intrinsically harder* (higher irreducible entropy) than others. The adversary would dump all weight on the hardest-to-model domain forever (e.g., noisy multilingual text), regardless of whether more weight actually *helps*. DoReMi's key idea is to measure each domain not by raw loss but by **excess loss** relative to a fixed **reference model** trained once on the natural mixture:

$$
\text{excess}_i(\theta) = \underbrace{\ell_i(\theta)}_{\text{proxy loss on domain } i} - \underbrace{\ell_i(\theta_{\text{ref}})}_{\text{reference loss on domain } i}.
$$

Excess loss answers a sharper question: *"On which domain is the proxy still far from what's achievable?"* A domain with high irreducible entropy will have high loss for *both* proxy and reference, so its excess is small once the proxy catches up — the adversary stops over-investing in it. A domain where the proxy lags the reference (lots of headroom) gets upweighted. DoReMi clamps excess at zero (you cannot do better than reference "for free") and uses it to drive the weights.

{{fig:mixing-excess-loss-decomposition}}

### The algorithm

DoReMi runs three steps:

1. **Train a reference model** $\theta_{\text{ref}}$ (small, e.g. 280 M) on the natural mixture $w^{\text{nat}}$. Record its per-domain losses $\ell_i(\theta_{\text{ref}})$. (Used only to define the excess-loss baseline.)
2. **Train a proxy model** $\theta$ of the *same small size* with **online Group-DRO**: at each step, evaluate the proxy's per-domain excess loss, multiplicatively update domain weights toward high-excess domains, and use those weights to draw the next batch. Average the weights over all steps to get $\bar{w}$.
3. **Train the large target model** on the *fixed* averaged mixture $\bar{w}$ from step 2. The expensive run uses a static mixture; all the adaptivity happened cheaply in the proxy.

The online update in step 2 is **exponentiated gradient ascent** on the weights (multiplicative weights / Hedge). Let $\lambda_i^{(t)}$ be the clamped excess loss of domain $i$ at step $t$. The weight update with step size $\eta$ is

$$
\tilde{w}_i^{(t+1)} = w_i^{(t)} \exp\!\big(\eta \, \lambda_i^{(t)}\big), \qquad w_i^{(t+1)} = (1-c)\,\frac{\tilde{w}_i^{(t+1)}}{\sum_j \tilde{w}_j^{(t+1)}} + c\,\frac{1}{k},
$$

where the renormalization keeps $w^{(t+1)}$ on the simplex and the small **smoothing** constant $c$ (mixing in the uniform distribution) guarantees every domain keeps a floor of weight so it is never starved to zero. The proxy parameters $\theta$ are trained *normally* (minimizing weighted loss) while the weights chase the excess — a two-player game whose averaged weights are the output.

{{fig:doremi-reference-proxy-loop}}

The payoff: DoReMi reports faster convergence to a given loss and better downstream performance than the natural mixture, found with one small reference + one small proxy run — far cheaper than a grid of full-size ablations. The reported weights also transfer across an order of magnitude of scale reasonably well, though, as noted, transfer is not perfect and is the method's main caveat.

### Online / adaptive mixing during the real run

DoReMi freezes the mixture for the target run. An alternative is to keep adapting *during* the large run — **online data mixing**. The appeal is that the optimal mixture genuinely changes over training (a model that has mastered easy web text may benefit from shifting weight to math later). The risk is instability and the cost of computing per-domain signals on the fly. Practical online schemes (e.g., Albalak et al.'s *Online Data Mixing*, and bandit-style approaches) treat each domain as an arm of a multi-armed bandit and use a reward signal — typically the *rate of loss decrease* on that domain (its learning *velocity*) — to shift weight toward domains where the model is currently learning fastest, while a smoothing/exploration term keeps every domain sampled. This connects directly to RL-style curriculum (see [RL Data, Curriculum & Replay Management](../06-rl-infra/12-rl-data-curriculum-replay.html)), where the same "train on what you're learning from right now" intuition drives sample selection.

!!! note "Group-DRO vs. online bandit mixing — same family, different reward"

    Both treat domains as the thing to reweight and both use multiplicative-weights updates. The difference is the signal: **DoReMi's Group-DRO uses excess loss (a level)** — "how far is this domain from achievable?" — and is run on a cheap proxy to produce a static target mixture. **Online bandit mixing uses loss velocity (a derivative)** — "where am I improving fastest right now?" — and is run during the real training. Excess loss says "fix what's broken"; velocity says "ride what's working." They can disagree: a domain can have high excess loss yet near-zero velocity (stuck), in which case more weight wastes compute.

{{fig:mixing-level-vs-velocity}}

---

## A DoReMi-Style Reweighting Toy You Can Run

Let us make all of this concrete with a small, self-contained simulation. We will *not* train real transformers — that would obscure the mechanism. Instead we model each domain's loss with a realistic **learning curve**: per-domain loss falls as a power law in the number of tokens that domain has received, toward a domain-specific floor. This captures the two things that matter for mixing — domains have different floors (difficulty) and different learning rates (transfer/headroom) — and lets us watch the Group-DRO weights respond. The same multiplicative-weights loop transfers directly to a real proxy run; only the loss source changes.

```python
import numpy as np

rng = np.random.default_rng(0)

# -----------------------------------------------------------------------------
# 1. Define k domains. Each has:
#    - floor:  irreducible loss (entropy of that domain's text), in nats/token
#    - rate:   power-law exponent for how fast loss falls with tokens seen
#    - scale:  loss = floor + scale * (tokens_seen + 1)^(-rate)
#    Domains differ in BOTH difficulty (floor) and headroom/speed (scale,rate).
# -----------------------------------------------------------------------------
domains = ["web", "code", "math", "books", "multi"]
floor = np.array([1.70, 1.10, 1.55, 1.80, 2.30])   # math & multi are "hard"
scale = np.array([2.0, 3.5, 4.0, 1.8, 2.2])        # code & math have big headroom
rate  = np.array([0.32, 0.28, 0.22, 0.30, 0.18])   # math & multi learn slowly
k = len(domains)

def domain_loss(tokens_seen):
    """Per-domain held-out loss given cumulative tokens seen in that domain."""
    return floor + scale * np.power(tokens_seen + 1.0, -rate)

# -----------------------------------------------------------------------------
# 2. Reference model: train once on the NATURAL mixture (proportional to pool
#    sizes), then read off its per-domain loss. This defines the excess-loss
#    baseline. We simulate "training" simply by accumulating tokens per domain.
# -----------------------------------------------------------------------------
pool = np.array([3000., 250., 30., 80., 400.])     # unique tokens (in B), illustrative
w_nat = pool / pool.sum()                          # natural mixture
REF_TOKENS = 5.0e4                                 # arbitrary proxy-scale token units

ref_tokens_per_domain = w_nat * REF_TOKENS
ref_loss = domain_loss(ref_tokens_per_domain)
print("natural mixture  :", np.round(w_nat, 3))
print("reference loss   :", np.round(ref_loss, 3))

# -----------------------------------------------------------------------------
# 3. Proxy model with online Group-DRO (DoReMi step 2).
#    - w:        current sampling weights over domains (the adversary's play)
#    - seen:     cumulative tokens per domain (the proxy's "knowledge")
#    Each step we draw a batch split by w, accumulate tokens, recompute the
#    proxy's per-domain loss, form CLAMPED excess loss vs the reference, and
#    apply an exponentiated-gradient (multiplicative-weights) update to w.
# -----------------------------------------------------------------------------
STEPS        = 4000
BATCH_TOKENS = 10.0          # tokens added per step (proxy-scale units)
ETA          = 1.0           # weight learning rate (exp-gradient step size)
SMOOTH_C     = 0.05          # uniform smoothing -> every domain keeps a floor weight

w    = w_nat.copy()          # start from the natural mixture
seen = np.zeros(k)           # proxy has seen nothing yet
w_history = []

for t in range(STEPS):
    # (a) Draw this step's batch according to current weights and "train":
    #     allocate BATCH_TOKENS across domains in proportion to w.
    seen += w * BATCH_TOKENS

    # (b) Proxy's current per-domain loss and CLAMPED excess loss vs reference.
    proxy_loss = domain_loss(seen)
    excess = np.maximum(proxy_loss - ref_loss, 0.0)   # cannot beat ref "for free"

    # (c) Exponentiated-gradient ascent on weights toward high-excess domains.
    w = w * np.exp(ETA * excess)
    w = w / w.sum()                                   # back onto the simplex
    w = (1.0 - SMOOTH_C) * w + SMOOTH_C / k           # uniform smoothing (floor)

    w_history.append(w.copy())

w_bar = np.mean(w_history, axis=0)   # DoReMi outputs the AVERAGED weights
print("\nfinal-step weights:", np.round(w, 3))
print("AVERAGED weights  :", np.round(w_bar, 3))
print("vs natural        :", np.round(w_nat, 3))
print("upweight factor   :", np.round(w_bar / w_nat, 2))
```

Running this prints something like:

```text
natural mixture  : [0.798 0.066 0.008 0.021 0.106]
reference loss   : [1.767 1.461 2.621 2.022 2.77 ]
final-step weights: [0.305 0.174 0.174 0.174 0.174]
AVERAGED weights  : [0.361 0.172 0.133 0.136 0.198]
vs natural        : [0.798 0.066 0.008 0.021 0.106]
upweight factor   : [ 0.45  2.59 16.62  6.4   1.86]
```

Read the result. The natural mixture is 80% web; DoReMi-style reweighting collapses web from 80% to ~36% and dramatically upweights the small high-headroom domains — code ~2.6x, math ~17x, books ~6x. This is exactly the qualitative behavior reported for real DoReMi: it pulls weight *out* of the abundant, lower-headroom domain (web) and *into* domains where the proxy still has the most room to improve relative to the reference. The math domain, despite a high floor (it is genuinely hard), gets heavily upweighted because its *excess* — the gap the proxy can still close — is large. Note the multilingual domain, which has the highest floor *and* the slowest rate (small headroom relative to its difficulty), is upweighted only modestly: high raw loss alone does not earn weight; **closeable** loss does. That separation is the entire reason DoReMi uses excess loss instead of raw loss.

Two experiments to build intuition (left as exercises you can run in seconds):

- **Set `ref_loss = 0`** (i.e., use *raw* loss instead of excess). You will see the adversary dump weight onto `multi`, the highest-floor domain, and refuse to leave — illustrating exactly the pathology excess loss was invented to fix.
- **Sweep `SMOOTH_C` from `0.0` to `0.3`.** At `0.0`, weights can drive a domain's effective weight toward zero (starvation); larger values keep every domain alive but blur the signal. The DoReMi default lives near the small end.

!!! tip "Practitioner tip: validate proxy weights at one intermediate scale"

    Because the optimal mixture drifts with scale, do not blindly ship proxy-derived weights to your 70 B run. Run *one* confirmatory training at an intermediate scale (e.g., 1–7 B for a few hundred billion tokens) comparing the proxy weights against the natural mixture and one hand-tuned alternative. If the proxy weights win there too, scale with confidence; if the ranking flips, your proxy was too small to be predictive — increase it. This single check is far cheaper than discovering the problem at full scale.

---

## Scheduling Over Time: Curriculum, Context Ramps & Annealing

So far $w$ has been a single fixed vector. But the *order* in which a model sees data, and *when* it sees the best data, matters too. We now let the mixture be a function of training progress, $w(t)$, and consider three orthogonal scheduling axes: difficulty (curriculum), sequence length (context ramp), and quality (annealing/mid-training).

### Curriculum learning: easy-to-hard

**Curriculum learning** (Bengio et al., 2009) orders examples from easy to hard, mirroring how humans learn. For LLM pretraining the evidence is genuinely mixed — large transformers on shuffled data are remarkably robust, and a naive curriculum often yields little — but specific, *targeted* curricula do help. The practical recipes that survive contact with reality:

- **Length-based difficulty:** start with shorter or simpler documents, introduce longer/denser ones later. This overlaps with the context ramp below.
- **Quality/complexity-based:** introduce highly technical content (dense math, advanced code) after the model has basic linguistic competence, so it does not waste early capacity on tokens it cannot yet model. This is the curriculum intuition behind putting hard reasoning data later in training.
- **Skill-staged for code/math:** in domains with a natural difficulty gradient (e.g., basic syntax → algorithms → competition problems), staging by difficulty can outperform a uniform shuffle.

The mechanism, when it works: early in training the model's gradients on very hard examples are high-variance and poorly aligned (it lacks the prerequisites), so those tokens are inefficiently used. Delaying them spends early compute on tokens with cleaner learning signal. The risk: a *too-narrow* early diet can cause the model to over-specialize and then struggle to adapt (a mild form of the loss-of-plasticity problem; see [Continual & Domain-Adaptive Pretraining](../03-pretraining/16-continual-pretraining.html)).

!!! warning "Curriculum is not free lunch for pretraining"

    Many published "curriculum helps" results fail to replicate at scale or vanish once the baseline is a well-shuffled, well-mixed corpus. Treat curriculum as a *targeted* tool — most valuable for context-length scheduling and for the high-quality annealing phase below — rather than a universal "sort everything by difficulty" prescription. The robust wins are at the *boundaries* of training (length ramps, final-phase quality), not in fine-grained per-example ordering of the bulk.

### Context-window scheduling

Training directly at long context (e.g., 128 K tokens) from step one is wasteful: attention cost grows with sequence length (quadratically for vanilla attention), most early learning needs only local context, and long-document data is scarce. The dominant recipe is a **context-length ramp**: train the bulk of the run at a short context (e.g., 4 K), then *extend* to long context in a final phase with appropriate positional-encoding adjustments (RoPE base/theta scaling, etc.). This is a curriculum over *sequence length*, and it is one of the few curricula with near-universal adoption. Because it is a deep topic of its own — RoPE scaling, position interpolation, data selection for the long phase — we treat it fully in [Long-Context Pretraining & Context Extension](../03-pretraining/13-long-context-pretraining.html). For mixing purposes, the key point is that the *mixture itself changes* in the long-context phase: you upweight naturally long documents (books, repositories, multi-turn transcripts) because they are the only sources that exercise long-range dependencies.

### Mid-training and annealing: save the best for last

The single most impactful scheduling idea in modern pretraining is the **high-quality annealing phase** (sometimes called *mid-training* or the *cooldown* phase). The recipe, popularized by the MiniCPM team's analysis and adopted widely (Llama 3, OLMo, and others describe variants):

1. Train the vast majority of tokens on a broad, web-heavy mixture with a roughly constant (or slowly decaying) learning rate.
2. In the **final fraction** of training (often the last ~10–20% of tokens), simultaneously **(a) decay the learning rate sharply toward zero** and **(b) shift the data mixture toward the highest-quality, most target-relevant data** — curated math, code, textbooks, instruction-formatted and reasoning-heavy data.

The interaction between the two is the whole point. **Data seen while the learning rate is high and falling fastest has the largest, most lasting effect on the final weights**, because those late steps with a decaying LR are where the model settles into its final basin. Putting your best, most capability-dense data exactly there — when each gradient step still moves the weights but the model is no longer being yanked around — imprints those capabilities most strongly. The MiniCPM "Warmup-Stable-Decay" (WSD) schedule makes this explicit: a long stable-LR phase on the broad mixture, then a short decay phase on upweighted high-quality data. It also has a delicious practical benefit: because the stable phase uses a constant LR, you can *branch* multiple annealing experiments from a single stable checkpoint and try different final mixtures cheaply, instead of re-running from scratch.

{{fig:mixing-wsd-anneal-schedule}}

This connects naturally to the boundary between pretraining and post-training: the annealing-phase mixture often *resembles* instruction/SFT data (see [Supervised Fine-Tuning & Instruction Tuning](../05-posttraining-alignment/01-sft-instruction-tuning.html)), and a strong annealing phase reduces how much later fine-tuning is needed. It also overlaps with continual pretraining for domain adaptation (see [Continual & Domain-Adaptive Pretraining](../03-pretraining/16-continual-pretraining.html)).

!!! example "Worked example: how much does annealing 'cost' in the mixture?"

    A 2 T-token run reserves its final 15% (300 B tokens) for annealing. During the stable 1.7 T phase the mixture is web-heavy: math gets $w_{\text{math}}=0.04$, so it sees $0.04 \times 1700 = 68$ B math tokens. During the 300 B annealing phase the team raises $w_{\text{math}}$ to $0.25$, adding $0.25 \times 300 = 75$ B more math tokens — **more math in the final 15% than in the entire first 85%.** With only 30 B unique math tokens, total math exposure is $(68+75)/30 \approx 4.8$ epochs, most of it concentrated late when the LR is decaying and each token leaves the deepest imprint. The lesson: annealing is not just "a bit more quality data" — it can *dominate* a small domain's effective exposure and place that exposure at the most influential point in the schedule. Plan your epoch budget (the danger zone near ~4–6 epochs) with the annealing contribution included, or you will silently over-repeat your best data.

### Putting the schedule together

A modern end-to-end pretraining schedule, combining all three axes, looks like:

{{fig:mixing-end-to-end-schedule-phases}}

Each phase is a different $w(t)$, a different context length, and a different point on the LR schedule (see [Learning Rate Schedules, Warmup, Batch Size & Hyperparameters](../03-pretraining/10-lr-schedules-hparams.html)). The art of pretraining data is, in large part, the art of designing this table — and then validating it with the proxy and intermediate-scale checks from earlier in the chapter.

!!! interview "Interview Corner"

    **Q:** You're told a new pretraining run will reserve the last 10% of tokens for a "high-quality annealing phase" with a sharp learning-rate decay. A colleague proposes simply upsampling that same high-quality data uniformly across the *entire* run instead, arguing it's "the same total exposure, less complexity." Why might the annealing approach still win, and what's the main risk you'd flag?

    **A:** They are *not* equivalent even at equal total exposure, because the **effect of data depends on when it's seen relative to the learning-rate schedule**. Late steps, where the LR is decaying toward zero, are where the model settles into its final weights; gradients there produce the most lasting changes and are not subsequently washed out. Concentrating the best, most capability-dense data exactly in that window imprints those capabilities most strongly — the empirical basis for WSD/annealing schedules. Spreading the same data uniformly dilutes it across early high-LR steps whose contributions are largely overwritten later. The annealing approach also enables cheap experimentation: branch several decay runs from one stable checkpoint. The main risk to flag is **over-repetition / overfitting of the small high-quality set**: because annealing concentrates a small domain's exposure (and may push it past ~4–6 effective epochs at the most impactful moment), you can memorize it or narrow the model. Mitigate by counting epochs *including* the annealing contribution, capping repetition, and supplementing with fresh synthetic data rather than re-reading the same tokens.

---

## Bringing It Together: A Mixing & Scheduling Playbook

The decisions in this chapter compose into a repeatable workflow:

1. **Partition and deduplicate.** Define domains; deduplicate *within* each so "one epoch" is honest ([Data Cleaning, Deduplication & Quality Filtering](../03-pretraining/02-data-cleaning-dedup.html)).
2. **Set a target distribution** $p^{\star}$ reflecting what you care about downstream — not what you happen to have.
3. **Find base weights.** Either run manual mixture ablations at a proxy scale, or run a DoReMi-style reference+proxy pass to get $\bar{w}$ via excess-loss Group-DRO.
4. **Convert to an epoch budget** and check no domain exceeds the repetition danger zone (~4–6 epochs); upsample small high-value domains, downsample abundant low-value ones.
5. **Validate at one intermediate scale** before committing the full run.
6. **Design the schedule** $w(t)$: broad stable phase, optional online velocity-based nudges, a long-context phase that upweights long documents, and a final annealing phase that concentrates the highest-quality data as the LR decays.
7. **Account for annealing in the epoch math** so your best small domains are not silently over-repeated.

Get these right and you buy capability gains that would otherwise cost a substantial model-size increase — at zero extra FLOPs. Data mixing is one of the highest-leverage, lowest-cost levers in the entire pretraining stack.

!!! key "Key Takeaways"

    - A **mixture** is a sampling distribution $w$ over domains; the natural (proportional) mixture is almost never optimal because you care about a *target* distribution of downstream uses, not raw token counts.
    - Mixing is fundamentally an **epoch-budget** decision: $e_i = w_i D / n_i$. **Deduplicate first** (to make epochs honest), **then upsample deliberately** (small high-value domains to ~2–4 epochs; beyond ~4–6, returns to repeated data decay and memorization rises).
    - The **training** mixture $w$ is a separate object from the **target** weights $p^{\star}$; setting $w=p^{\star}$ is usually suboptimal because domains differ in difficulty and transfer.
    - **Manual ablations** at a proxy scale are the robust workhorse; **DoReMi** automates this with a reference + proxy run using **Group-DRO on excess loss** (closeable loss, not raw loss) to avoid over-investing in intrinsically hard domains.
    - **Online/adaptive mixing** uses loss *velocity* ("ride what's improving") rather than DoReMi's excess-loss *level* ("fix what's broken"); both use multiplicative-weights updates with smoothing to avoid starving any domain.
    - The optimal mixture **drifts with model scale and with training progress** — always validate proxy-derived weights at an intermediate scale before the full run.
    - **Annealing / mid-training** is the highest-impact schedule trick: in the final ~10–20% of tokens, decay the LR sharply *and* shift the mixture to the best, most target-relevant data — late, low-LR steps imprint capabilities most strongly (the WSD recipe).
    - **Context-length ramps** are a near-universal curriculum over sequence length; the mixture upweights long documents in the long-context phase.
    - Count epochs **including the annealing contribution** — annealing can dominate a small domain's total exposure and silently push it into the over-repetition zone.

!!! sota "State of the Art & Resources (2026)"
    Data mixing and curriculum design remain active research areas in 2026. Automated mixture-optimization methods (DoReMi, DoGE, RegMix) have largely supplanted manual grid search at proxy scale, while annealing-phase schedules are now near-universal in frontier pretraining runs. The open challenge is reliable scale-transfer: mixture weights learned on small proxies still shift in non-trivial ways at 70B+.

    **Foundational work**

    - [Sagawa, Koh, Hashimoto, Liang, *Distributionally Robust Neural Networks for Group Shifts* (2020)](https://arxiv.org/abs/1911.08731) — the Group-DRO minimax optimization framework that DoReMi builds on.
    - [Muennighoff et al., *Scaling Data-Constrained Language Models* (2023)](https://arxiv.org/abs/2305.16264) — quantifies how repeated tokens lose value; establishes the ~4-epoch threshold and geometric decay of repeat utility.

    **Recent advances (2023–2026)**

    - [Xie et al., *DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining* (2023)](https://arxiv.org/abs/2305.10429) — reference + proxy excess-loss Group-DRO method; finds mixture weights with a single extra small run, reaching 8B baseline accuracy 2.6x faster.
    - [Fan, Pagliardini, Jaggi, *DoGE: Domain Reweighting with Generalization Estimation* (2023)](https://arxiv.org/abs/2310.15393) — alternative proxy-model approach using generalization estimation rather than excess loss.
    - [Albalak, Pan, Raffel, Wang, *Efficient Online Data Mixing For Language Model Pre-Training* (2023)](https://arxiv.org/abs/2312.02406) — bandit-style (EXP3) adaptive mixing during training; reaches final perplexity with 19% fewer steps.
    - [Liu et al., *RegMix: Data Mixture as Regression for Language Model Pre-training* (2024)](https://arxiv.org/abs/2407.01492) — frames mixture selection as regression over small proxy runs; ICLR 2025 Spotlight; matches or beats DoReMi at 10% of its FLOPs.
    - [Hu et al., *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies* (2024)](https://arxiv.org/abs/2404.06395) — introduces the Warmup-Stable-Decay (WSD) schedule enabling cheap branched annealing experiments from a single stable checkpoint.
    - [OLMo Team, *OLMo 2* (2025)](https://arxiv.org/abs/2501.00656) — fully open model with detailed documentation of two-stage pretraining, domain-weighted annealing, and the Dolmino high-quality annealing mix.

    **Open-source & tools**

    - [sangmichaelxie/doremi](https://github.com/sangmichaelxie/doremi) — official PyTorch DoReMi implementation with HuggingFace Trainer integration and a fast domain-weighted dataloader.
    - [sail-sg/regmix](https://github.com/sail-sg/regmix) — RegMix code for generating mixture configs, training proxy models, and fitting the regression predictor.
    - [huggingface/datablations](https://github.com/huggingface/datablations) — code and experiments from "Scaling Data-Constrained Language Models"; useful for studying epoch-repetition tradeoffs.

    **Go deeper**

    - [Stanford CRFM: DoReMi blog post (2023)](https://crfm.stanford.edu/2023/09/14/doremi) — accessible walkthrough of the DoReMi algorithm with practical guidance on running the proxy pipeline.

## Further reading

- Xie, Pham, Dong, et al., *DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining* (2023) — the reference/proxy excess-loss Group-DRO method central to this chapter.
- Sagawa, Koh, Hashimoto, Liang, *Distributionally Robust Neural Networks for Group Shifts* (Group-DRO, 2020) — the optimization framework DoReMi builds on.
- Muennighoff, Rush, et al., *Scaling Data-Constrained Language Models* (2023) — the value of repeated tokens and the limits of upsampling.
- Albalak et al., *Online Data Mixing for Language Model Pre-training* — bandit-style adaptive mixing during the real run.
- Hu et al. (MiniCPM), *MiniCPM: Unveiling the Potential of Small Language Models* — the Warmup-Stable-Decay schedule and the high-quality annealing phase.
- Bengio, Louradour, Collobert, Weston, *Curriculum Learning* (2009) — the original easy-to-hard framing.
- Gao et al., *The Pile* — an early explicit, documented domain-weighted pretraining mixture.
- The Llama, Gopher, and OLMo technical reports — real-world descriptions of mixture ablations, annealing, and data-schedule design at frontier scale.
