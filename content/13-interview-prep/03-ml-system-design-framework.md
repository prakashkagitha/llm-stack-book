# 13.3 ML System Design: A Framework

The ML system design interview is the one round where a strong candidate most often *underperforms relative to their ability*. The reason is not a lack of knowledge — it is a lack of **structure**. Faced with "Design YouTube recommendations" or "Build a system to detect toxic comments," people dive straight into model architecture, forget the metric, never mention the data, and run out of time before they reach serving. The interviewer is left unable to assess whole categories of skill.

This chapter gives you a single, repeatable spine you can impose on *any* ML-design prompt:

```text
clarify → requirements/SLOs → data → features → model → training → eval → serving → monitoring → iteration
```

That is the whole framework. Memorize it as a chant. Walk the interviewer through it out loud, drawing boxes on the whiteboard as you go. Each stage below comes with the questions to ask, the decisions to make, the tradeoffs to name, and the code or numbers that turn hand-waving into engineering. The companion chapter [ML System Design: Worked Cases](../13-interview-prep/04-ml-system-design-cases.html) applies this exact spine to four full prompts; here we build the spine itself. For interview *format and timing*, see [The Google ML Domain Interview: Format & Strategy](../13-interview-prep/01-google-ml-interview-format.html).

## The Mental Model: A Loop, Not a Pipeline

Beginners draw ML system design as a left-to-right assembly line. Senior engineers draw a **loop**. Models decay, data drifts, labels arrive late, and your first metric is almost always slightly wrong. The system that wins is the one designed so that signals from production flow back into the next training run.

```text
              ┌──────────────────────────────────────────────────┐
              │                  ITERATION                        │
              ▼                                                   │
  ┌─────────┐   ┌──────────────┐   ┌──────┐   ┌──────────┐   ┌───────┐
  │ Clarify │ → │ Requirements │ → │ Data │ → │ Features │ → │ Model │
  └─────────┘   │   & SLOs     │   └──────┘   └──────────┘   └───────┘
                └──────────────┘                                  │
                                                                  ▼
  ┌────────────┐   ┌─────────┐   ┌──────────┐   ┌─────────────┐
  │ Monitoring │ ← │ Serving │ ← │   Eval   │ ← │  Training   │
  └────────────┘   └─────────┘   └──────────┘   └─────────────┘
        │                                                ▲
        └──────────── feedback / data flywheel ──────────┘
```

The most important arrow on the whiteboard is the dashed one from **Monitoring** back to **Data**: the *data flywheel*. Drawing it early signals seniority. We unpack flywheels in [Data Flywheels & Continuous Improvement](../12-production-mlops/05-data-flywheel.html).

A practical rule of thumb for how to spend a 45-minute round:

| Phase | Time | What you produce |
|---|---|---|
| Clarify + requirements | ~8 min | Scope, primary metric, scale, constraints |
| Data + features + labels | ~10 min | Where supervision comes from, leakage map |
| Model + training | ~10 min | Baseline → proposed model, loss, eval split |
| Serving + monitoring | ~10 min | Latency budget, candidate generation, online metrics |
| Iteration + Q&A | ~7 min | Failure modes, next experiments |

If you spend 30 minutes on model architecture, you have failed the round regardless of how clever the architecture is.

## Stage 1 — Clarify the Problem

Never start designing. **Spend the first five minutes turning a vague prompt into a crisp ML problem statement.** Interviewers deliberately under-specify so they can watch you scope. Your job is to convert "Design X" into a single sentence of the form:

> *Predict $y$ for entity $z$ in context $c$, optimizing $M$, subject to constraints $K$.*

### The clarifying-question checklist

Ask these, in roughly this order:

1. **Business objective.** What outcome are we actually trying to move? (Watch time? Revenue? Harm reduction? Retention?) The *business* metric is almost never the same as the *ML* metric, and the gap between them is where most of the interesting design lives.
2. **User and surface.** Who sees the output, where, and what action can they take? "A ranked feed on mobile" implies very different latency and slate sizes than "a nightly batch email."
3. **Prediction target.** Classification, regression, ranking, generation, retrieval? Binary or multi-class? Point estimate or full distribution?
4. **Scale.** How many users, items, queries per second (QPS), events per day? This single number drives almost every later decision — candidate generation, embedding table size, whether you can afford a cross-encoder.
5. **Latency budget.** Real-time (<100 ms), near-real-time (seconds), or batch (hours)? This is a hard constraint, not a preference.
6. **Existing system.** Is there a baseline? A heuristic? Logs? "We currently rank by recency" tells you both your baseline and your initial training data source.
7. **Constraints.** Privacy, fairness, regulatory, cold-start, multilingual, on-device.

Frame the prompt as a function you are approximating. For a recommendation system:

$$
\hat{y} = f_\theta(\text{user},\ \text{item},\ \text{context}) \approx P(\text{engage} \mid \text{user},\ \text{item},\ \text{context})
$$

State explicitly what "engage" means — a click, a 30-second watch, a like — because that *label definition* silently determines everything downstream.

!!! tip "Practitioner tip"
    State your assumptions out loud and write them in a corner of the board: "I'll assume ~100M daily active users, ~10M items, a 200 ms p99 budget for ranking, and that we have 6 months of impression/click logs." If the interviewer disagrees, they will correct you — which is *exactly* the signal you want. Silent assumptions are the number-one way candidates design the wrong system.

## Stage 2 — Requirements, Metrics & SLOs

Now translate the clarified problem into measurable targets. This stage separates engineers from theoreticians.

### Functional vs non-functional requirements

- **Functional:** what the system does. "Return a ranked list of 20 videos personalized to the user."
- **Non-functional:** how well it must do it — latency, throughput, availability, cost, freshness, fairness. These are your **SLOs (Service Level Objectives)**.

### Choosing the metric — the three-layer stack

Every ML system has a metric *hierarchy*. Name all three layers explicitly; conflating them is a classic mistake.

| Layer | Question | Examples | Measured by |
|---|---|---|---|
| **Business / North Star** | Does the company win? | DAU, revenue, retention, harm rate | A/B test, long horizon |
| **Online / system** | Does the product surface behave? | CTR, watch time, session length, complaint rate | A/B test, real traffic |
| **Offline / model** | Is the model good in isolation? | AUC, log loss, NDCG, recall@k, F1 | Held-out logs |

The art is choosing an offline metric that **correlates** with the online metric, which in turn moves the business metric. Optimizing offline AUC is worthless if AUC gains do not translate to watch time. Always say: *"I'll validate the offline→online correlation before trusting offline numbers."*

### Picking the offline metric to match the task

- **Binary classification, balanced consequences:** ROC-AUC, log loss.
- **Binary classification, heavy class imbalance** (fraud, abuse, ads click): **PR-AUC** and precision/recall at a fixed threshold. ROC-AUC is misleadingly optimistic under imbalance.
- **Ranking / recommendation:** NDCG@k, MAP, recall@k for the candidate-generation stage, AUC or calibrated log loss for the ranking stage.
- **Calibrated probabilities needed** (ads bidding, expected-value decisions): log loss + a calibration metric like Expected Calibration Error (ECE). A well-ranked but mis-calibrated model is useless for an auction.
- **Regression:** RMSE (penalizes large errors), MAE (robust), or quantile loss if you need intervals.
- **Generation / LLM:** task-specific — exact-match, pass@k, win-rate vs a baseline, or an LLM-as-judge score. See [The Evaluation Problem & Benchmark Landscape](../11-evaluation/01-eval-landscape.html) and [LLM-as-a-Judge & Automated Evaluation](../11-evaluation/02-llm-as-judge.html).

NDCG (Normalized Discounted Cumulative Gain) is worth knowing cold for ranking rounds:

$$
\text{DCG@}k = \sum_{i=1}^{k} \frac{2^{\text{rel}_i} - 1}{\log_2(i+1)},
\qquad
\text{NDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}
$$

where $\text{rel}_i$ is the graded relevance of the item at rank $i$ and $\text{IDCG@}k$ is the DCG of the ideal (perfectly sorted) ranking, which normalizes the score into $[0,1]$.

```python
import numpy as np

def dcg_at_k(relevances, k):
    """Discounted Cumulative Gain over the top-k items.
    relevances: list of graded relevance scores in the *predicted* order."""
    rel = np.asarray(relevances, dtype=float)[:k]
    # rank i is 1-indexed; discount = 1 / log2(i + 1)
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    gains = (2.0 ** rel) - 1.0           # exponential gain rewards highly-relevant hits
    return float(np.sum(gains * discounts))

def ndcg_at_k(relevances, k):
    """NDCG: DCG of predicted order divided by DCG of the ideal order."""
    actual = dcg_at_k(relevances, k)
    ideal = dcg_at_k(sorted(relevances, reverse=True), k)
    return actual / ideal if ideal > 0 else 0.0

# Predicted ranking put a "1" before a "3": NDCG penalizes that inversion.
print(round(ndcg_at_k([3, 1, 2, 0], 4), 4))   # 1.0  (already ideal)
print(round(ndcg_at_k([1, 3, 2, 0], 4), 4))   # ~0.79 (good item ranked 2nd)
```

### SLOs and the latency budget

Write down a concrete budget and decompose it. Latency is *additive* across stages, so the budget must be split:

```text
End-to-end p99 budget: 200 ms
  ├─ network + gateway        : 20 ms
  ├─ feature fetch (store)    : 30 ms
  ├─ candidate generation     : 40 ms   (ANN over 10M items → 1000 candidates)
  ├─ ranking model            : 80 ms   (score 1000 candidates)
  └─ post-processing/dedup    : 30 ms
```

Always reason about **p99, not the mean** — tail latency is what users feel and what pages the on-call engineer. A model with a great median but a fat tail will blow your SLO.

!!! warning "Common pitfall"
    Optimizing a single offline number with no online plan. If you say "I'd maximize AUC" and stop, you have skipped the entire reason the system exists. Always close the loop: offline metric → A/B test → online metric → business metric. And always name a **guardrail metric** (e.g., "improve CTR *without* increasing reported-content rate or p99 latency") so you do not win the battle and lose the war.

## Stage 3 — Data & Labels

Data is where most real ML projects succeed or fail, and where most candidates spend too little time. Slow down here.

### The four questions of data

1. **Where does supervision come from?** Explicit labels (human annotation), implicit feedback (clicks, watches, purchases), or self-supervision? Most large-scale systems run on *implicit* feedback because it is free and abundant — but it is biased.
2. **How much do we have, and how fresh?** Volume sets model capacity; freshness sets retraining cadence.
3. **What is the label, exactly?** "Did the user like this video?" Is a 2-second view a positive? A skip a negative? Pin down the positive/negative definition — it is a modeling decision disguised as a data question.
4. **What are the biases?** Position bias, presentation bias, selection bias, feedback loops. You can only learn from items you *showed*, and you showed them because a previous model thought they were good. This is **survivorship/exposure bias** and it is everywhere in recsys and search.

### Implicit feedback and its traps

Position bias is the canonical example: users click the top result more *because* it is on top, not because it is more relevant. If you train naively on clicks, you teach the model to predict *position*, not relevance.

Two standard remedies, both worth naming:

- **Inverse Propensity Weighting (IPW).** Weight each training example by the inverse probability that it was *shown* at its position. If an item at position 5 (rarely examined) got a click, that click is strong evidence and should count more:

$$
\mathcal{L}_{\text{IPW}} = \frac{1}{N}\sum_{i=1}^{N} \frac{1}{p_i}\, \ell(\hat{y}_i, y_i),
\qquad p_i = P(\text{examined} \mid \text{position}_i)
$$

- **Position as a feature at train time, zeroed at serve time.** Feed position into the model during training so it can "explain away" the position effect, then set it to a constant (e.g., position 1) at inference. This is the "position-debias tower" trick used in production ranking systems.

### Negative sampling

Most large-scale systems have abundant positives (clicks) and an astronomically large *implicit* negative set (everything not clicked). You cannot train on all of it. Strategies:

- **Random negatives** — cheap, but mostly trivially easy; the model learns little.
- **In-batch negatives** — use other positives in the mini-batch as negatives for this example. Standard in two-tower retrieval; nearly free because the embeddings are already computed.
- **Hard negatives** — items the current model ranks highly but that were not engaged. These sharpen the decision boundary but, if overused, cause instability and false-negative noise. A common recipe mixes ~majority random with a small fraction of hard negatives.

```python
import torch
import torch.nn.functional as F

def in_batch_softmax_loss(user_emb, item_emb, temperature=0.07):
    """Two-tower retrieval loss with in-batch negatives (a la sampled softmax).
    user_emb, item_emb: (B, d) L2-normalized embeddings; row i is a positive pair.
    Every *other* item in the batch acts as a negative for user i.
    """
    logits = user_emb @ item_emb.t() / temperature   # (B, B) similarity matrix
    labels = torch.arange(user_emb.size(0), device=user_emb.device)  # diagonal = positives
    # Cross-entropy pulls the matched pair together, pushes the B-1 negatives apart.
    return F.cross_entropy(logits, labels)

# Sanity check on random data
torch.manual_seed(0)
u = F.normalize(torch.randn(4, 16), dim=1)
v = F.normalize(torch.randn(4, 16), dim=1)
print(float(in_batch_softmax_loss(u, v)))   # a finite positive scalar
```

### The leakage map

Before any modeling, mentally trace every feature back in time and ask: *was this knowable at prediction time?* **Target leakage** — a feature that encodes the label or post-event information — produces gorgeous offline metrics and catastrophic production failure. Classic leaks: using a "final purchase amount" to predict whether a purchase happens; using aggregate stats computed over a window that includes the prediction instant; joining a label table that has been updated since the event.

The cleanest defense is a **point-in-time correct join**: every feature value must reflect the state of the world *as of the event timestamp*, never later. We return to this in features and again in [Caching, Routing & Cost Control in Production](../12-production-mlops/03-caching-routing-cost.html) for the serving side.

## Stage 4 — Features & Representation

Features are how you encode the world into tensors. Discuss them by **category**, not as a flat list — categories show you understand the *kinds* of signal available.

### Feature taxonomy

- **User features:** demographics, long-term history (embeddings of past interactions), aggregated counts, account age.
- **Item features:** content embeddings (text/image/audio), category, age, historical engagement stats.
- **Context features:** time of day, day of week, device, locale, network type, position in feed.
- **Cross / interaction features:** user-item affinity (has this user engaged with this creator before?), query-document match. These are often the highest-signal features and the hardest to compute online.

### Encoding choices

| Feature type | Encoding |
|---|---|
| Low-cardinality categorical | One-hot or learned embedding |
| High-cardinality categorical (user IDs, item IDs) | Embedding table (possibly hashed) |
| Numerical, skewed | Log-transform, then standardize or bucketize |
| Numerical, heavy-tailed counts | Quantile/percentile bucketing |
| Text | Tokenize → embed (see [Embeddings & Representation Learning](../09-rag-retrieval/01-embeddings-representation.html)) |
| Timestamps | Cyclical encoding: $\sin/\cos$ of hour, day-of-week |

For high-cardinality IDs, embedding tables dominate memory. A worked sizing:

!!! example "Worked example — embedding-table memory and the latency budget"
    Suppose 100M users and 10M items, each with a 64-dimensional embedding in fp32 (4 bytes).

    User table: $100{,}000{,}000 \times 64 \times 4 = 2.56 \times 10^{10}$ bytes $\approx$ **25.6 GB**.

    Item table: $10{,}000{,}000 \times 64 \times 4 = 2.56 \times 10^{9}$ bytes $\approx$ **2.56 GB**.

    Total $\approx$ **28 GB** just for two embedding tables — already exceeding a single 24 GB GPU. Mitigations: drop to fp16 (halve to ~14 GB), reduce dim to 32, **hash** user IDs into a smaller table (accepting collisions), or shard the table across hosts via a parameter server. This is exactly the kind of magnitude reasoning interviewers reward — it forces a real architectural decision rather than a hand-wave.

    Now the latency side. Suppose ranking must score $N=1000$ candidates within an 80 ms budget on one machine. That is $80\text{ ms} / 1000 = 80\ \mu\text{s}$ per candidate end-to-end. A heavy cross-encoder costing 5 ms each would need $5000$ ms — **62× over budget**. Conclusion drawn *from the numbers*: cross-encoders are infeasible at the candidate-generation stage; reserve them for a final re-rank of the top ~20, and use a cheap two-tower dot product for the 1000.

### Feature stores and train/serve skew

The single most common production ML bug is **train/serve skew**: a feature is computed one way in the offline training pipeline (in SQL over a data warehouse) and a subtly different way in the online serving path (in Python over a streaming store). The model sees a distribution at serve time it never trained on.

The standard fix is a **feature store** that guarantees the *same transformation code* and *point-in-time correctness* on both paths:

```text
                    ┌─────────────────────────┐
   raw events ─────▶│   Feature transforms     │  (ONE definition of each feature)
                    │   (shared library)       │
                    └──────────┬───────┬───────┘
                               │       │
                    offline    │       │   online
              (point-in-time)  ▼       ▼ (low-latency KV)
                    ┌──────────────┐ ┌──────────────┐
                    │ Offline store│ │ Online store │
                    │ (warehouse)  │ │ (Redis/etc.) │
                    └──────┬───────┘ └──────┬───────┘
                           ▼                ▼
                       training          serving
```

State explicitly: *"I'd use a feature store so the exact same transformation defines a feature offline and online, eliminating train/serve skew, and I'd materialize features with a point-in-time join keyed on the event timestamp."* That one sentence demonstrates production experience.

## Stage 5 — Model Selection & Training

Only now do we choose a model — and even here, **start with a baseline**.

### Always propose a baseline first

State a simple baseline before anything fancy:

- **Heuristic / non-ML:** most-popular, recency, rules. Surprisingly hard to beat and free to ship.
- **Linear / logistic regression** on hand-crafted features. Interpretable, fast, a real yardstick.
- **Gradient-boosted trees (GBDT)** for tabular features — often the strongest non-deep baseline and frequently the production winner for structured data.

Saying "I'd ship a popularity baseline in week one and measure it" signals that you optimize for *impact per unit effort*, not novelty. Many interviewers consider failure to propose a baseline a red flag.

### The complexity ladder

Climb only as far as the data and latency budget justify:

1. Logistic regression / GBDT on tabular features.
2. **Two-tower** (dual-encoder) model for retrieval: a user tower and an item tower producing embeddings whose dot product is the score. Enables sub-linear ANN retrieval over millions of items.
3. **Deep ranking network** (e.g., a wide-and-deep or DCN-style cross network) for the final ranking of a few hundred candidates, where you can afford richer cross features.
4. **Sequence models / transformers** when order and long histories matter (session-based recommendation, next-action prediction).
5. **LLM-based** approaches when the input is language and you need generalization or generation — see [Supervised Fine-Tuning & Instruction Tuning](../05-posttraining-alignment/01-sft-instruction-tuning.html) and, for retrieval-augmented designs, [Retrieval-Augmented Generation Architectures](../09-rag-retrieval/03-rag-architectures.html).

### The two-stage (candidate generation → ranking) pattern

This is *the* canonical large-scale recommendation/search architecture and you should be able to draw it from memory. You cannot run an expensive model over millions of items per request, so you split the problem:

```text
  10M items
     │
     ▼
┌─────────────────────┐   cheap, high-recall
│ Candidate generation │   (two-tower + ANN index)   →  ~1000 candidates
└─────────────────────┘
     │
     ▼
┌─────────────────────┐   expensive, high-precision
│      Ranking         │   (deep net w/ cross feats)  →  ~100 scored
└─────────────────────┘
     │
     ▼
┌─────────────────────┐   business rules, diversity,
│   Re-ranking / policy │   freshness, dedup, fairness →  top 20 shown
└─────────────────────┘
```

The candidate generator optimizes **recall** (don't lose the good item); the ranker optimizes **precision/calibration** (order the survivors correctly); the re-ranker injects **business logic** (diversity, freshness, policy). Each stage trades compute for items: cheap over many, expensive over few. The ANN retrieval underpinning candidate generation is covered in [Vector Databases & Approximate Nearest Neighbor Search](../09-rag-retrieval/02-vector-databases-ann.html).

A minimal two-tower model, end to end:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Tower(nn.Module):
    """Maps sparse + dense features to an L2-normalized embedding."""
    def __init__(self, n_ids, id_dim=32, n_dense=8, out_dim=64):
        super().__init__()
        self.id_emb = nn.Embedding(n_ids, id_dim)        # hashed ID embedding
        self.mlp = nn.Sequential(
            nn.Linear(id_dim + n_dense, 128), nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, ids, dense):
        x = torch.cat([self.id_emb(ids), dense], dim=-1)
        return F.normalize(self.mlp(x), dim=-1)          # unit-norm → dot product == cosine

class TwoTower(nn.Module):
    def __init__(self, n_users, n_items):
        super().__init__()
        self.user_tower = Tower(n_users)
        self.item_tower = Tower(n_items)

    def forward(self, u_ids, u_dense, i_ids, i_dense):
        ue = self.user_tower(u_ids, u_dense)
        ie = self.item_tower(i_ids, i_dense)
        return ue, ie

# Training step uses the in-batch softmax loss defined earlier.
# At SERVE time: precompute ALL item embeddings nightly, build an ANN index,
# embed the user online, and retrieve top-k by approximate nearest neighbor.
model = TwoTower(n_users=1 << 20, n_items=1 << 20)
u_ids = torch.randint(0, 1 << 20, (8,))
i_ids = torch.randint(0, 1 << 20, (8,))
ue, ie = model(u_ids, torch.randn(8, 8), i_ids, torch.randn(8, 8))
print(ue.shape, ie.shape)   # torch.Size([8, 64]) torch.Size([8, 64])
```

The key serving insight, which you should say out loud: **item embeddings are precomputed offline and indexed; only the user tower runs online.** That is what makes retrieval over 10M items feasible inside 40 ms.

### Loss functions — match the loss to the metric

- **Pointwise** (binary cross-entropy): predict $P(\text{click})$ per item. Simple, gives calibrated probabilities, but ignores relative order.
- **Pairwise** (BPR, RankNet): learn that item A should rank above item B. Directly targets ordering.
- **Listwise** (LambdaRank/LambdaMART, softmax/sampled-softmax): optimize a whole ranked list, often a surrogate for NDCG. Best ranking quality, more complex.

If the *online* metric is ordering quality (NDCG), a pointwise log-loss can be a fine proxy at the ranking stage because well-calibrated probabilities sort well — but name the option to go pairwise/listwise.

### Training-data hygiene

- **Split by time, not at random**, for any system with temporal structure. Random splits leak the future into the training set and inflate offline metrics. Train on weeks 1–8, validate on week 9, test on week 10.
- **Handle class imbalance** with class weights, focal loss, or down-sampling negatives (then calibrate the threshold back).
- **Watch for distribution shift** between train and serve; schedule periodic retraining.

For the deep-learning machinery underneath — optimizers, schedules, regularization, mixed precision — defer to Part III, e.g. [Optimizers: SGD, Adam, Adafactor, Lion, Muon & Shampoo](../03-pretraining/09-optimizers.html). In an interview, mention Adam/AdamW, a warmup+decay schedule, early stopping on the validation metric, and L2/dropout — then move on. Architecture is *not* where you should spend your minutes.

## Stage 6 — Evaluation: Offline and Online

Evaluation deserves its own stage because a system you cannot measure is a system you cannot improve.

### Offline evaluation done right

- **Temporal hold-out**, as above. Report the metric chosen in Stage 2, plus calibration if probabilities matter.
- **Slice the metric.** A single aggregate number hides failures. Report performance per segment: new vs returning users, head vs tail items, by locale, by device. A model that is great on average but broken for new users (cold start) is often a non-starter.
- **Counterfactual / off-policy evaluation.** Because logs were collected under the *old* policy, naive replay is biased. Estimators like **Inverse Propensity Scoring (IPS)** and the **doubly-robust** estimator estimate how a *new* policy would have performed, using logged propensities. Naming this is a strong senior signal.

The IPS estimator of a new policy's value from logs collected under behavior policy $\pi_b$:

$$
\hat{V}_{\text{IPS}}(\pi_e) = \frac{1}{N}\sum_{i=1}^{N} \frac{\pi_e(a_i \mid x_i)}{\pi_b(a_i \mid x_i)}\, r_i
$$

where $r_i$ is the logged reward, $a_i$ the logged action, and the ratio re-weights each logged event by how much more (or less) likely the *new* policy was to take that action.

### Online evaluation: the A/B test

Offline gains are a hypothesis; the **A/B test** is the verdict. The structure to recite:

1. **Hypothesis:** "The new ranker increases watch time without raising the report rate."
2. **Randomization unit:** usually the user (not the request), to avoid contamination and capture within-session effects.
3. **Primary metric + guardrails:** one metric to move, several that must *not* regress (latency p99, revenue, complaint rate).
4. **Power and duration:** size the experiment so it can detect the minimum effect you care about.

A back-of-envelope sample-size formula for comparing two proportions (e.g., CTR) at 80% power and 5% significance:

$$
n \approx \frac{16\,\bar{p}(1-\bar{p})}{\delta^2} \quad \text{per arm}
$$

where $\bar p$ is the baseline rate and $\delta$ the absolute lift you want to detect.

!!! example "Worked example — sizing an A/B test"
    Baseline CTR $\bar p = 0.10$, and we want to detect an absolute lift of $\delta = 0.005$ (i.e., 10.0% → 10.5%, a 5% relative gain).

    $$
    n \approx \frac{16 \times 0.10 \times 0.90}{(0.005)^2} = \frac{16 \times 0.09}{0.000025} = \frac{1.44}{0.000025} = 57{,}600 \text{ users per arm.}
    $$

    So roughly **115k users total**. At 100M DAU that is a tiny fraction — easy to run. But flip it: to detect a $\delta = 0.0005$ (0.5% relative) lift you need $100\times$ more users — **5.76M per arm**, which at lower traffic could take weeks. The takeaway to voice: *small effects require large samples or long runs*, and that constraint is exactly why offline screening and good guardrails matter before you spend traffic.

!!! warning "Common pitfall"
    **Peeking.** Repeatedly checking a running A/B test and stopping the moment it crosses significance massively inflates the false-positive rate — you will "discover" wins that are noise. Fix the sample size and duration in advance, or use a sequential-testing method (e.g., always-valid p-values / mSPRT) designed for continuous monitoring. Mention this and you signal experimentation maturity.

## Stage 7 — Serving, Scaling & Infrastructure

A trained model is worthless until it serves predictions within budget at scale. Cover four things: topology, latency, scaling, and freshness.

### Online vs batch vs streaming

- **Batch (offline) inference:** precompute predictions on a schedule, store in a KV store, serve by lookup. Lowest online cost; staleness is the price. Great for "users who bought X also bought Y" tables or nightly item embeddings.
- **Online (real-time) inference:** run the model per request. Necessary when the input is only known at request time (current query, session context). Bounded by your latency SLO.
- **Streaming / near-real-time:** update features (and sometimes the model) continuously from an event stream — important for freshness-sensitive systems (trending, fraud).

Most production systems are **hybrid**: batch-precompute the heavy parts (item embeddings, candidate pools), do the light, personalized part online.

### Serving topology for the two-stage system

```text
   request
      │
      ▼
 ┌──────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────┐
 │ Gateway  │──▶ │ Feature store │──▶ │ Candidate gen │──▶ │ Ranker   │──▶ response
 │ (auth,   │    │ (online KV)   │    │ (ANN index)   │    │ (deep nn │
 │  rate-lim)│   └──────────────┘    └───────────────┘    │  on GPU/ │
 └──────────┘                                              │  CPU)    │
                                                           └──────────┘
        ▲                                                       │
        └────────────────── logging / metrics ──────────────────┘
```

For the LLM-specific serving stack — continuous batching, KV-cache management, autoscaling, cost — see [Designing an LLM Serving System](../12-production-mlops/01-serving-system-design.html) and [Inference Economics: Latency, Throughput & Cost](../07-inference-serving/12-inference-economics.html). The principles below apply to any model.

### Scaling levers

- **Horizontal scaling + load balancing:** stateless model servers behind a balancer; scale replicas with QPS.
- **Caching:** cache features, candidate sets, even full responses for hot/popular queries. A cache hit is the cheapest prediction there is. See [Caching, Routing & Cost Control in Production](../12-production-mlops/03-caching-routing-cost.html).
- **Model compression:** quantization and distillation shrink the serving footprint and cut latency. See [Distillation, Model Compression & Knowledge Transfer](../05-posttraining-alignment/12-distillation-compression.html) and [Quantization I: Post-Training Quantization (GPTQ, AWQ, SmoothQuant)](../04-kernels-efficiency/07-quantization-ptq.html).
- **Hardware:** CPU is fine for trees and small nets; GPU/accelerators for deep models and LLMs.
- **Batching:** group concurrent requests to amortize fixed cost — a latency/throughput tradeoff to state explicitly.

### A tiny capacity calculation

```python
def replicas_needed(qps, per_request_ms, target_utilization=0.7,
                    concurrency_per_replica=1):
    """How many model-server replicas to hold p99 under load.
    qps: expected peak queries/sec
    per_request_ms: model latency per request (ms)
    """
    # One replica with C-way concurrency serves: 1000/latency * C requests/sec.
    throughput_per_replica = (1000.0 / per_request_ms) * concurrency_per_replica
    raw = qps / throughput_per_replica
    return int(raw / target_utilization) + 1   # headroom so we don't run hot

# 50k QPS, 80 ms per request, single-threaded, run at 70% utilization:
print(replicas_needed(50_000, 80))     # -> ~5715 replicas (!) ⇒ must batch or compress
```

That eye-watering number is itself the lesson: at 50k QPS and 80 ms, single-request serving is absurdly expensive, which *forces* you to batch, cache, distill, or push work into the offline path. Let the arithmetic drive the architecture.

### Cold start

Always raise it unprompted — it is a favorite follow-up. **New users** (no history): fall back to popularity, context-only features, or an onboarding flow that collects preferences. **New items** (no engagement): rely on content features (text/image embeddings) so the item tower can place them in embedding space before any clicks exist. Content-based features bridge the gap until collaborative signal accumulates.

## Stage 8 — Monitoring, Iteration & The Flywheel

Shipping is the *start*, not the end. The system must watch itself and feed the next iteration.

### What to monitor — three tiers

1. **System health:** latency (p50/p99), QPS, error rate, saturation. Standard SRE signals.
2. **Data & feature health:** feature distribution drift, null/NaN rates, schema changes, training-vs-serving feature-value mismatches. Most silent model failures are upstream *data* failures.
3. **Model quality:** online metric vs baseline, prediction-score distribution drift, calibration drift, and per-slice performance. Watch for **feedback loops** — a recommender that only ever shows popular items starves the tail of training data and ossifies.

A cheap, deployable drift detector using **Population Stability Index (PSI)**:

$$
\text{PSI} = \sum_{b=1}^{B} (a_b - e_b)\,\ln\!\frac{a_b}{e_b}
$$

where $e_b$ and $a_b$ are the expected (training) and actual (live) fractions of traffic in bin $b$. Rules of thumb: $\text{PSI} < 0.1$ stable, $0.1$–$0.25$ moderate shift, $> 0.25$ significant — investigate and consider retraining.

```python
import numpy as np

def psi(expected, actual, bins=10, eps=1e-6):
    """Population Stability Index between a reference and a live sample.
    Quantile bins from the *reference* distribution; compare mass in each bin."""
    quantiles = np.quantile(expected, np.linspace(0, 1, bins + 1))
    quantiles[0], quantiles[-1] = -np.inf, np.inf      # open the outer edges
    e_counts, _ = np.histogram(expected, bins=quantiles)
    a_counts, _ = np.histogram(actual, bins=quantiles)
    e = e_counts / e_counts.sum() + eps                 # avoid log(0) / div-by-0
    a = a_counts / a_counts.sum() + eps
    return float(np.sum((a - e) * np.log(a / e)))

rng = np.random.default_rng(0)
ref = rng.normal(0, 1, 10_000)
print(round(psi(ref, rng.normal(0.0, 1, 10_000)), 4))  # ~0.00  no shift
print(round(psi(ref, rng.normal(0.6, 1, 10_000)), 4))  # large  ⇒ drift, alert
```

### The data flywheel

Close the loop you drew at the start. Production interactions become *labels* for the next model: shown items + engagement become training data, corrected predictions become a curated set, and hard cases (where the model was confident and wrong) get prioritized for human labeling. This compounding advantage — better model → better product → more/better data → better model — is the moat. Detail in [Data Flywheels & Continuous Improvement](../12-production-mlops/05-data-flywheel.html). Beware degenerate loops: if you only train on what the model chose to show, you reinforce its blind spots; inject **exploration** (a small fraction of randomized or epsilon-greedy traffic) to keep collecting unbiased signal.

### Retraining cadence

State a trigger policy, not just a schedule: retrain on a fixed cadence (daily/weekly) **and** on a drift trigger (PSI breach, online-metric drop). Always **shadow** or **canary** a new model — serve it to a small slice or in log-only mode — before full rollout, and keep an instant rollback path.

!!! interview "Interview Corner"
    **Q:** Your offline AUC jumped from 0.82 to 0.86 with a new feature, but the A/B test shows *no* change in click-through and a slight rise in p99 latency. What's going on and what do you do?

    **A:** Several hypotheses, which I'd test in order. (1) **Offline/online mismatch / leakage** — a +4 point AUC jump from a single feature is suspiciously large; the feature may leak label information unavailable at serve time, or be computed differently online (train/serve skew). I'd audit it for point-in-time correctness and confirm the serving path reproduces the offline value. (2) **Metric mismatch** — AUC measures ranking of the *whole* candidate pool, but users only see the top few; gains may be on items that never get impressed, so they don't move CTR. NDCG@k or a top-slot metric would reflect the user experience better. (3) **Saturation / ceiling** — the existing model may already rank the top slots well, so improving the tail is invisible online. (4) **Latency regression** — the new feature added p99 latency, and slower responses can *depress* engagement enough to cancel a real quality gain; I'd check whether a latency-neutral version recovers the win. Decision: do **not** ship on offline AUC alone. Diagnose the leakage/skew first (most likely culprit), align the offline metric to the served surface, and only roll out if a latency-controlled A/B shows a real, guardrail-safe lift.

### A reusable checklist

Keep this in your head as the closing sweep of any design round:

```text
[ ] Restated the problem as predict-y-for-z-optimizing-M-subject-to-K
[ ] Named business, online, and offline metrics + their correlation plan
[ ] Stated scale (users/items/QPS) and a p99 latency budget, decomposed
[ ] Said where labels come from; addressed bias (position/selection)
[ ] Drew a leakage/point-in-time map; chose a time-based split
[ ] Proposed a baseline before the fancy model
[ ] Used two-stage retrieval→rank if items are large
[ ] Matched loss to metric; matched model to data size & latency
[ ] Offline eval with slices + an A/B plan with guardrails & power
[ ] Serving topology, caching, cold-start, capacity numbers
[ ] Monitoring (system/data/model), drift detection, retraining triggers
[ ] Drew the data-flywheel arrow with an exploration mechanism
```

## Tradeoff Talking Points (Say These Out Loud)

Interviewers grade *how you reason about tradeoffs*. Have these ready as reflexes:

- **Latency vs accuracy:** bigger models rank better but blow the budget → two-stage retrieval, distillation, quantization.
- **Precision vs recall:** candidate generation maximizes recall; ranking maximizes precision. The *threshold* is a business decision tied to the cost of a false positive vs false negative.
- **Freshness vs cost:** real-time features and frequent retraining cost compute and complexity; batch is cheap but stale. Pick per-feature.
- **Bias vs variance:** simple models (high bias) are robust and cheap; deep models (high variance) need more data and regularization. More on this in [Machine Learning Fundamentals](../01-foundations/05-ml-fundamentals.html).
- **Build vs buy:** a heuristic shipped this week often beats a transformer shipped next quarter — measure the gap before paying for it.
- **Exploration vs exploitation:** pure exploitation ossifies the model and starves the data flywheel; budget a slice of traffic for exploration.
- **Personalization vs privacy:** richer user features improve predictions but raise privacy/regulatory exposure; consider on-device, aggregation, or differential privacy.
- **Global vs per-segment models:** one model is simpler to operate; per-segment (or multi-task) models fit heterogeneity better at higher operational cost.

The meta-point: **there is rarely a single right answer.** The signal you send is that you can enumerate the options, state the axis of tradeoff, and justify a choice *given the stated constraints*. When in doubt, tie every decision back to the metric and SLOs from Stage 2.

!!! key "Key Takeaways"
    - Impose one spine on every prompt: **clarify → requirements/SLOs → data → features → model → training → eval → serving → monitoring → iteration** — and budget your time so you actually reach serving and monitoring.
    - **Clarify before you design.** Convert a vague prompt into "predict $y$ for $z$, optimizing $M$, subject to constraints $K$," and write your assumptions (scale, latency, baseline) on the board.
    - Name **all three metric layers** — business, online, offline — and commit to validating the offline→online correlation rather than trusting AUC alone.
    - Spend real time on **data and labels**: where supervision comes from, position/selection bias and its fixes (IPW, in-batch/hard negatives), and a **point-in-time leakage map** to kill train/serve skew.
    - **Propose a baseline first** (popularity, logistic, GBDT), then climb the complexity ladder only as far as data and latency justify; reach for the **two-stage retrieval→rank** pattern whenever the item set is large.
    - Evaluate offline with **time-based splits and per-slice metrics**, then decide with a **properly powered A/B test** that has guardrails and no peeking.
    - **Do the arithmetic** — embedding-table GB, per-candidate microseconds, replica counts, A/B sample sizes — and let the magnitudes force architectural decisions.
    - Close the loop: monitor system/data/model health, detect drift (PSI), retrain on cadence *and* trigger, and draw the **data-flywheel** arrow with an exploration budget.
    - Verbalize **tradeoffs** (latency↔accuracy, precision↔recall, freshness↔cost, explore↔exploit) and justify each choice against the stated SLOs.

## Further reading

- Covington, Adams & Sargin, *Deep Neural Networks for YouTube Recommendations* (2016) — the canonical two-stage candidate-generation/ranking architecture.
- Cheng et al., *Wide & Deep Learning for Recommender Systems* (2016) — memorization vs generalization in ranking models.
- Joachims, Swaminathan & Schnabel, *Unbiased Learning-to-Rank with Biased Feedback* — propensity weighting for position bias.
- Kohavi, Tang & Xu, *Trustworthy Online Controlled Experiments* — the practitioner's bible on A/B testing, guardrails, and pitfalls.
- Sculley et al., *Hidden Technical Debt in Machine Learning Systems* (2015) — why the model is the small part of a real ML system.
- Huyen, *Designing Machine Learning Systems* — an end-to-end treatment of the lifecycle this chapter frames.
- Google, *Rules of Machine Learning* (Zinkevich) — battle-tested heuristics, especially "do the simple thing first."
