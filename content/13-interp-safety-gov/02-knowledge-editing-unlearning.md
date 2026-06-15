# 13.2 Knowledge Editing & Machine Unlearning

A deployed language model is, among other things, a lossy compression of a snapshot of the world. The snapshot rots. A prime minister loses an election; a company rebrands; a fact you trained on turns out to be wrong; a user invokes their right to be forgotten and demands that their leaked phone number stop appearing in completions. The brute-force fix — gather corrected data and retrain from scratch — costs millions of dollars and weeks of wall-clock time for a frontier model. We would like a scalpel instead of a sledgehammer: a way to change one specific thing the model "knows" while leaving the other billions of facts, the fluency, and the reasoning untouched.

This is the domain of **knowledge editing** (deliberately overwriting a target fact) and its safety-critical cousin **machine unlearning** (provably removing the *influence* of specific training data, e.g. for copyright, privacy, or dangerous-capability removal). Both ask the same uncomfortable question: where, physically, inside a stack of transformer weights, does a fact live — and can we surgically rewrite it without collateral damage?

We build up from the locate-then-edit hypothesis (causal tracing, ROME, MEMIT, AlphaEdit), through memory- and adapter-based editors that sidestep weight surgery entirely (GRACE, WISE), to the failure modes that make all of this hard in practice (ripple effects, locality violations, forgetting at scale). We then turn to unlearning for compliance — the TOFU and MUSE benchmarks, gradient-ascent recipes, editing-as-unlearning, and the gap between "the model won't say it" and "the model provably never knew it." A from-scratch rank-one ROME-style edit anchors the mechanism in runnable code.

This chapter assumes the transformer internals from [The Transformer Block: Norms, Residuals, MLPs & Activations](../02-transformer/06-transformer-block.html) and the circuit-level view from [Mechanistic Interpretability & Model Internals](../13-interp-safety-gov/01-mechanistic-interpretability.html). It connects forward to the legal framing in [Privacy, Memorization & Differential Privacy for LLMs](../13-interp-safety-gov/03-privacy-memorization-dp.html) and [AI Governance, Compliance & Regulation](../13-interp-safety-gov/06-governance-compliance.html).

---

## 1. The Locate-Then-Edit Hypothesis

The foundational empirical claim, due to Meng et al. in *Locating and Editing Factual Associations in GPT* (ROME, 2022), is that simple subject–relation–object facts in autoregressive transformers are stored in a **localized, additive, and editable** way inside the feed-forward (MLP) sub-layers of the *middle* layers. If that claim holds even approximately, editing reduces to (a) finding the right weight matrix and (b) computing a small, targeted update to it.

### 1.1 The MLP as a key–value memory

Recall the transformer MLP at layer $\ell$ acting on a residual-stream vector $h \in \mathbb{R}^{d}$:

$$
m = W_{\text{down}}\,\sigma\!\left(W_{\text{up}}\, h\right),
$$

where $W_{\text{up}} \in \mathbb{R}^{d_{\text{mlp}} \times d}$, $W_{\text{down}} \in \mathbb{R}^{d \times d_{\text{mlp}}}$, and $\sigma$ is the nonlinearity (GELU/SwiGLU). Geva et al. (*Transformer Feed-Forward Layers Are Key-Value Memories*, 2021) reinterpret this: the rows of $W_{\text{up}}$ are **keys** that fire on particular input patterns, and the columns of $W_{\text{down}}$ are the **values** they write into the residual stream. The activation $k = \sigma(W_{\text{up}} h)$ is a sparse "which memories fired" vector, and the output is a key-weighted sum of value vectors,

$$
m = W_{\text{down}}\,k = \sum_{i} k_i \,\big(W_{\text{down}}\big)_{:,i}.
$$

ROME zooms in on $W_{\text{down}}$ (it calls it $W$) and treats it as a **linear associative memory**: a single matrix that maps a set of key vectors $\{k_1,\dots,k_n\}$ to value vectors $\{v_1,\dots,v_n\}$ via $v \approx W k$. To insert a *new* association $(k_*, v_*)$ — "when you see the key for *the Space Needle is located in the city of* ___, write the value that produces *Paris*" — we modify $W$ minimally.

### 1.2 Causal tracing: finding the layer

Before editing you must know *where*. ROME's localization tool is **causal tracing** (a form of activation patching; see [Mechanistic Interpretability & Model Internals](../13-interp-safety-gov/01-mechanistic-interpretability.html)). The recipe:

1. **Clean run.** Feed the factual prompt ("The Space Needle is in downtown ___"); record the probability the model assigns to the correct token ("Seattle") and *cache every hidden state*.
2. **Corrupted run.** Add Gaussian noise to the *subject* token embeddings ("The Space Needle") so the model loses the fact; the correct-token probability collapses.
3. **Restoration sweep.** Re-run corrupted, but at each (layer, token) position *patch in* the clean cached hidden state, one at a time. Measure how much the correct probability is restored.

The cells that restore the most are where the fact is causally mediated. The robust finding: a band of **middle-layer MLP outputs at the last subject token** carries an outsized share of the causal effect. That last-subject-token position is exactly where the edit will be applied — the model has finished "reading" the subject and is about to look up its properties.


{{fig:kedit-causal-tracing-heatmap}}


!!! note "Aside: this is a hypothesis, not a law"
    Locate-then-edit works *remarkably well on simple (subject, relation, object) facts* in GPT-2/GPT-J-scale models. Hase et al. (*Does Localization Inform Editing?*, 2023) showed a subtle and important caveat: the layer that causal tracing fingers as most "causal" is **not necessarily** the layer where an edit is most effective — you can often edit a *different* layer just as well. Localization tells you where a fact is *read*, not the unique place it must be *written*. Treat causal tracing as a strong prior, then validate empirically.

---

## 2. ROME: A Rank-One Edit

ROME makes a single rank-one update to one MLP down-projection. Two questions: *what* value to write ($v_*$), and *how* to write it without disturbing everything else.

### 2.1 Computing the target value $v_*$

ROME does **not** hand-pick $v_*$. It optimizes it. Freeze all weights; introduce a free vector $\delta$ added to the layer-$\ell$ MLP output at the subject's last token; and minimize the cross-entropy of the *desired* object $o_*$ over a few prompt templates $\{p_j\}$ that elicit the relation:

$$
v_* = \arg\min_{z}\;\frac{1}{N}\sum_{j=1}^{N} -\log P_{\,m_\ell \mathrel{+}= (z - m_\ell)}\big(o_* \mid p_j\big) \;+\; \lambda\, \text{KL}\big(P(\cdot\mid p') \,\|\, P_{\text{edited}}(\cdot\mid p')\big).
$$

The first term drags the model toward emitting the new object; the KL term (on a neutral prompt $p'$ such as "{subject} is a") is an **essence-preservation** regularizer that stops the edit from mangling the model's general sense of the subject. The minimization is a short Adam loop (typically 20–25 steps) over $z$ only — cheap, because gradients flow through a single forward pass and touch no weights.

### 2.2 The key $k_*$ and the closed-form rank-one update

The key is the *input* to $W_{\text{down}}$ at the edit site — the post-nonlinearity activation $k_* = \sigma(W_{\text{up}} h)$ at the subject's last token, averaged over the same templates (and, in practice, over a sample of prefixes for robustness).

Now the constrained update. We have an existing memory $W_0$ that already satisfies $W_0 K \approx V$ for a large set of "preserved" keys $K = [k_1,\dots,k_n]$. We want a new $W = W_0 + \Delta$ that additionally maps $k_* \mapsto v_*$ **while minimally perturbing the preserved keys**. ROME solves

$$
\min_{W}\;\lVert W K - V\rVert^2 \quad \text{subject to}\quad W k_* = v_*,
$$

and the Lagrangian gives a clean rank-one solution. Define $C = K K^\top$ (an uncentered covariance of keys — a statistic of "what this layer normally sees", precomputed once from a corpus like Wikipedia). The update is

$$
\boxed{\;\Delta = \frac{(v_* - W_0 k_*)\,\big(C^{-1} k_*\big)^\top}{\big(C^{-1} k_*\big)^\top k_*}\;}
$$

This is **rank one** — an outer product of two vectors — so it costs $d \times d_{\text{mlp}}$ extra storage at most and is trivially invertible (subtract it to undo the edit). The numerator's left factor $(v_* - W_0 k_*)$ is the *residual* we need to add at the key; the right factor $C^{-1} k_*$ steers the update along the direction that is least used by other keys (it is large where $C$ is small), which is precisely what minimizes collateral damage.

!!! example "Worked example: the magnitudes of one edit"
    Take GPT-J (6B), where the MLP hidden width is $d_{\text{mlp}} = 16384$ and the model width is $d = 4096$. ROME edits a single layer's $W_{\text{down}} \in \mathbb{R}^{4096 \times 16384}$ — about 67M parameters, but the *update* $\Delta$ is rank one, so its "size" is just the two vectors: $4096 + 16384 = 20480$ numbers, roughly **0.03%** of that one matrix and about **0.0003%** of the model's 6B parameters.

    The covariance $C = KK^\top$ is $16384 \times 16384 \approx 2.7\times 10^8$ entries; inverting it once costs $O(d_{\text{mlp}}^3) \approx 4.4\times10^{12}$ FLOPs — a few seconds on a GPU, amortized across *all* future edits to that layer because $C$ is fact-independent. The per-edit cost is then dominated by the ~25-step Adam optimization of $v_*$: ~25 forward/backward passes through the model on a handful of short prompts, i.e. **single-digit seconds**. Contrast with retraining GPT-J: thousands of GPU-hours. The asymmetry — milliseconds of linear algebra vs. weeks of training — is the whole reason the field exists.

---

## 3. From One Fact to Thousands: MEMIT and AlphaEdit

ROME edits **one** fact. Real applications need to inject hundreds or thousands at once (a knowledge refresh, a batch of corrections). Doing ROME sequentially compounds error: each rank-one bump shifts $W_0$ for the next edit, and after a few hundred edits the model degrades into incoherence. Two evolutions fix this.

### 3.1 MEMIT: mass-editing across multiple layers

MEMIT (Meng et al., *Mass-Editing Memory in a Transformer*, 2022) generalizes ROME along two axes:

1. **Many facts at once.** Instead of a rank-one update for one key, solve a *least-squares* update for a whole batch of key–value pairs $(K_1, V_1)$ simultaneously. The closed form generalizes to

    $$
    \Delta = R\,K_1^\top\big(C + K_1 K_1^\top\big)^{-1},
    $$

    where $R = V_1 - W_0 K_1$ is the matrix of residuals (one column per fact) and $C$ is again the preserved-key covariance. This is a **higher-rank** (but still low-rank) update that distributes thousands of associations across the matrix in one shot.

2. **Spread across a range of layers.** Rather than dumping the full update into a single layer, MEMIT spreads it over a *band* of critical middle layers (e.g. layers 3–8 in GPT-J). The target residual is amortized: each layer absorbs a fraction $1/L$ of the needed change, so no single matrix is perturbed violently. This is the key to scaling to ~10,000 edits while preserving fluency.

### 3.2 The drift problem and AlphaEdit

Even MEMIT degrades under *sequential* batches (edit a batch, then another, then another). Each update changes the very key distribution that the *next* update's $C$ assumed. The model's preserved knowledge drifts. AlphaEdit (Fang et al., 2024) addresses this with a **null-space projection**: before applying an update, project it onto the null space of the preserved knowledge's key covariance, so that

$$
\Delta\,K_{\text{preserved}} \approx 0.
$$

Concretely, let $P$ be the projector onto the null space of $C_{\text{preserved}} = K_p K_p^\top$ (computed from the SVD: keep the directions with near-zero singular values). Apply the MEMIT-style solve, then left-multiply by $P$:

$$
\Delta_{\text{AlphaEdit}} = P \,\Delta_{\text{MEMIT}}.
$$

Because $\Delta$ now lives in directions orthogonal to what preserved keys excite, applying it leaves their outputs (almost) exactly unchanged — the update "doesn't talk to" old facts. Empirically this dramatically reduces the catastrophic forgetting that plagues long sequential editing runs, letting the same matrix absorb far more edits before collapse.


{{fig:kedit-rome-memit-alphaedit-progression}}


!!! warning "Common pitfall: sequential editing is not batch editing"
    A 1,000-fact batch edit and 1,000 single edits applied one-after-another are **not** equivalent, even with the same algorithm. The batch solve sees all keys jointly and balances them; the sequential version lets early edits corrupt the statistics that late edits rely on. If you must edit incrementally over time, prefer AlphaEdit-style null-space methods or the memory-based editors of Section 4 — do not just loop ROME.

---

## 4. Memory and Adapter Editors: GRACE and WISE

Weight surgery is invasive: every edit permanently changes shared parameters, risks fluency, and is hard to audit. An alternative family keeps the base weights **frozen** and routes edited behavior through an external, addressable memory. This trades a clean separation (edits are data, not weight deltas) for an inference-time lookup.

### 4.1 GRACE: a discrete codebook of activations

GRACE (Hartvigsen et al., *Aging with GRACE*, 2023) inserts an adapter at one layer that holds a small **codebook** of (key, value) entries. At inference, the layer's incoming activation $h$ is compared to stored keys; if it falls within a learned radius $\epsilon$ of a key (an $\epsilon$-ball), the adapter **replaces** the activation with the stored value; otherwise it passes $h$ through untouched. New edits add codebook entries; conflicting edits split or shrink $\epsilon$-balls. Because the base model is frozen and the codebook is consulted only inside a deferral region, GRACE excels at **lifelong sequential editing** — thousands of edits over time with bounded interference, since each edit is a localized memory cell rather than a global weight perturbation.

```python
# grace_layer.py — a stripped-down GRACE-style deferral adapter (concept demo).
import torch, torch.nn as nn

class GraceAdapter(nn.Module):
    """Wrap one hidden layer: replace its output with a stored value
    when the input activation lands inside a stored epsilon-ball."""
    def __init__(self, dim, init_eps=3.0):
        super().__init__()
        self.keys, self.vals, self.eps = [], [], []  # the editable codebook
        self.init_eps = init_eps

    def add_edit(self, key_act: torch.Tensor, target_val: torch.Tensor):
        # Store the activation we want to intercept and what to emit instead.
        self.keys.append(key_act.detach())
        self.vals.append(target_val.detach())
        self.eps.append(self.init_eps)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if not self.keys:
            return h
        K = torch.stack(self.keys)                       # [n_edits, dim]
        d = torch.cdist(h.reshape(-1, h.shape[-1]), K)   # L2 distance to each key
        nearest = d.argmin(dim=-1)                        # closest codebook entry
        eps = torch.tensor(self.eps, device=h.device)[nearest]
        inside = d.gather(-1, nearest[:, None]).squeeze(-1) < eps   # within ball?
        out = h.reshape(-1, h.shape[-1]).clone()
        V = torch.stack(self.vals)
        out[inside] = V[nearest[inside]]                 # defer: overwrite activation
        return out.reshape_as(h)
```

### 4.2 WISE: side memory with routing

GRACE's weakness is **generalization**: it intercepts activations it has literally seen, so a paraphrase of the edited prompt may sail past the $\epsilon$-ball unedited. WISE (Wang et al., 2024) keeps the base ("main") FFN memory frozen and adds a **side memory** — a trainable copy of one FFN — plus a routing mechanism that decides, per token, whether to read from the main memory or the side memory based on an activation-norm gate. Edits are written into the side memory; a *knowledge-sharding* scheme spreads many edits across subspaces and merges them, mitigating interference. WISE was designed to attack a specific impossible-triangle in lifelong editing: jointly achieving **reliability** (the edit sticks), **generalization** (paraphrases get it too), and **locality** (unrelated facts untouched) — which pure weight-editing or pure memory methods each fail on one corner.

!!! tip "Practitioner tip: pick the editor for the workload"
    For a **one-off correction** in a model you control, a single ROME edit is simplest. For a **batch knowledge refresh** (a new data cutoff), MEMIT/AlphaEdit. For **continual, never-ending streams of edits** in production where you must not touch base weights (auditability, easy rollback), prefer GRACE/WISE-style memory adapters. And always ask first: would **retrieval-augmented generation** ([Retrieval-Augmented Generation Architectures](../09-rag-retrieval/03-rag-architectures.html)) solve this more robustly? If the "fact" changes weekly, a retrieval store you can edit with a database UPDATE beats any parametric edit.

---

## 5. Why It's Hard: Ripple Effects, Locality & Forgetting

Editing looks deceptively clean until you measure what *else* moved. The evaluation of an edit is itself a research problem, captured by four metrics that pull against each other.

### 5.1 The four axes of a good edit

| Metric | Question | Measured on |
|---|---|---|
| **Efficacy / Reliability** | Did the edit take? | the exact edited prompt |
| **Generalization** | Do paraphrases get it too? | rephrasings, related templates |
| **Locality / Specificity** | Are unrelated facts untouched? | neighborhood + random prompts |
| **Fluency / Consistency** | Is the model still coherent? | perplexity, n-gram entropy |

A trivial editor that hard-codes the output token maxes efficacy and destroys generalization; an aggressive weight blast maxes generalization and destroys locality. The art is the frontier.

### 5.2 Ripple effects: facts have neighbors

The deepest failure mode is the **ripple effect** (Cohen et al., *Evaluating the Ripple Effects of Knowledge Editing*, 2023). Knowledge is relational. If you edit "*Lionel Messi plays for* → *Inter Miami*", a correct world model implies a cascade of consequents: *Messi's league is now MLS*, *Messi's country of work is now the USA*, *the player wearing #10 for Inter Miami is Messi*. A surgical token-level edit changes the head fact but leaves the **2-hop and multi-hop consequences inconsistent** — the model will happily say Messi plays for Inter Miami *and* that he plays in the Spanish league. The RippleEdits benchmark formalizes this with logical-implication, composition, and subject-aliasing tests. The brutal lesson: a single rank-one bump cannot install a *belief*; it installs an *association*, and the model's other associations don't update to stay consistent.

### 5.3 Forgetting and drift at scale

As Section 3 foreshadowed, editing **accumulates damage**. Empirically, after enough sequential edits a model's general benchmark scores (perplexity, downstream accuracy) degrade — sometimes sharply, a "model collapse" cliff. The mechanisms: (a) each edit's $\Delta$ slightly perturbs preserved keys despite the $C^{-1}$ steering; (b) the covariance statistics go stale; (c) edits interfere with each other in shared parameter space. This is the editing-specific face of **catastrophic forgetting** — see the continual-learning treatment in [Continual & Domain-Adaptive Pretraining](../03-pretraining/16-continual-pretraining.html). Null-space projection (AlphaEdit) and memory adapters (GRACE/WISE) are the field's two main answers.

!!! warning "Common pitfall: confusing 'output changed' with 'knowledge changed'"
    An edit that makes the model *complete* "The capital of Australia is ___" with "Sydney" has not necessarily changed what the model knows. Probe sideways: ask in another language, ask the inverse ("Of which country is Sydney the capital?"), ask a multi-hop question. If those still say Canberra, you patched a surface association, not a belief. This same gap is what makes *unlearning* (Section 6) so treacherous: suppressing one phrasing is not removing the knowledge.

---

## 6. Machine Unlearning: Removing What a Model Knows

Editing *changes* a fact; **unlearning** *removes the influence* of specific training data. The motivations are legal and safety-critical: the **right to be forgotten** (GDPR Article 17, the CCPA right to delete) may require that a user's data no longer influence a deployed model; copyright takedowns may demand a book's text be expunged; and *dangerous-capability removal* (e.g. unlearning bioweapon-synthesis knowledge, as in the WMDP benchmark) is a frontier-safety lever. See [AI Governance, Compliance & Regulation](../13-interp-safety-gov/06-governance-compliance.html) and [Privacy, Memorization & Differential Privacy for LLMs](../13-interp-safety-gov/03-privacy-memorization-dp.html).

### 6.1 The gold standard and why we approximate it

The *exact* definition of unlearning is operational: a model has unlearned a "forget set" $D_f$ if it is **indistinguishable from a model retrained from scratch on $D \setminus D_f$**. That retrained model is the gold standard — and the reason approximate unlearning exists, because retraining is exactly the cost we are trying to avoid. SISA (Bourtoule et al., *Machine Unlearning*, 2021) makes *exact* unlearning cheaper by **sharding** training so that deleting a datum requires retraining only its shard — but sharding a trillion-token pretraining run is impractical for foundation models, so LLM unlearning is almost always *approximate*: cheap weight updates that *behave* like the retrained model on the tests we can run.

### 6.2 Gradient ascent and its discontents

The simplest recipe: do gradient **ascent** on the forget set — maximize loss on the data you want gone — usually balanced by gradient **descent** on a retain set to preserve utility:

$$
\mathcal{L}_{\text{unlearn}} = \underbrace{-\,\mathbb{E}_{x \sim D_f}\big[\log P_\theta(x)\big]}_{\text{push forget-set down}} \;+\; \lambda\,\underbrace{\mathbb{E}_{x \sim D_r}\big[-\log P_\theta(x)\big]}_{\text{keep retain-set up}}.
$$

Naively ascending loss is unstable — it diverges, blows up perplexity, and damages unrelated capabilities (the ascent gradient has no natural floor). Practical variants tame it:

- **Gradient Difference**: the loss above, subtracting forget-loss while adding retain-loss in one objective.
- **KL minimization**: instead of pure ascent on $D_f$, minimize KL to a reference (e.g. the original model) on $D_r$ while suppressing $D_f$, anchoring utility.
- **Preference-style unlearning (NPO, Zhang et al. 2024)**: *Negative Preference Optimization* treats forget samples as dis-preferred in a DPO-style loss (see [Direct Preference Optimization & Its Variants](../05-posttraining-alignment/07-dpo-and-variants.html)). NPO's gradient has a built-in adaptive weight that decays as the sample's probability drops, so it does **not** diverge the way raw gradient ascent does — a much more stable forget signal.

```python
# unlearn_step.py — gradient-difference + NPO-style unlearning step (sketch).
import torch, torch.nn.functional as F

def unlearn_step(model, ref_model, forget_batch, retain_batch,
                 beta=0.1, retain_lambda=1.0, method="npo"):
    # ---- retain term: ordinary LM loss keeps general ability intact ----
    r_out = model(**retain_batch, labels=retain_batch["input_ids"])
    retain_loss = r_out.loss

    # ---- forget term ----
    f_logp = seq_logprob(model, forget_batch)          # log P_theta(forget seq)
    if method == "grad_ascent":
        # Raw ascent: push forget log-prob down. Simple but unstable.
        forget_loss = f_logp.mean()                    # minimizing this = ascending NLL
    elif method == "npo":
        # NPO: forget set as 'rejected' in a DPO-style ratio vs frozen reference.
        with torch.no_grad():
            ref_logp = seq_logprob(ref_model, forget_batch)
        ratio = beta * (f_logp - ref_logp)             # how much more likely than ref
        # -log sigmoid(-ratio): drives P_theta below the reference, self-limiting.
        forget_loss = -F.logsigmoid(-ratio).mean() * (2.0 / beta)
    loss = forget_loss + retain_lambda * retain_loss
    return loss

def seq_logprob(model, batch):
    out = model(**batch)
    logp = F.log_softmax(out.logits[:, :-1], dim=-1)
    tgt = batch["input_ids"][:, 1:]
    tok_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    mask = batch["attention_mask"][:, 1:]
    return (tok_logp * mask).sum(-1) / mask.sum(-1)    # mean log-prob per sequence
```

### 6.3 Editing-as-unlearning

The locate-then-edit machinery (Sections 2–3) doubles as an unlearning tool. Two flavors:

- **Overwrite**: edit the target fact to a refusal or an "I don't know" / null answer, so the forget query resolves to a benign value (ROME/MEMIT with $o_*$ = "[redacted]").
- **Redirect / corrupt the representation**: methods like **RMU** (*Representation Misdirection for Unlearning*, the WMDP paper) push the hidden activations on forget-topic inputs toward random noise in a chosen layer while keeping retain activations fixed — degrading the model's *internal representation* of the dangerous topic rather than just its output token. This is harder to recover than an output-only patch.

### 6.4 Benchmarks: TOFU and MUSE

Because "the model won't say it" is not "the model unlearned it," the field built adversarial benchmarks.

- **TOFU** (Maini et al., *Task of Fictitious Unlearning*, 2024) fine-tunes a model on **synthetic author biographies that exist nowhere else**, then asks you to unlearn a subset. Because the facts are fictitious, there is no leakage from pretraining or the web — a clean test bed. It scores both **forget quality** (does the model still know the forgotten authors?) and **model utility** (retained authors + general ability), and crucially uses a **retrained-from-scratch reference** to define the target, plus a *truth-ratio* statistic that compares the model's probability on correct vs. perturbed (false) answers.

- **MUSE** (Shi et al., 2024) targets *realistic* corpora (news, books) and defines **six** desiderata, including ones the gradient-ascent crowd routinely fails: **no verbatim memorization**, **no knowledge memorization** (can't answer Q&A about forgotten text), **no privacy leakage via membership inference**, **utility preservation**, **scalability** to large forget sets, and **sequential** unlearning robustness.

### 6.5 The auditor's view: did it really forget?

The decisive test is adversarial, not behavioral. Three probes that routinely catch "fake" unlearning:

1. **Membership inference (MIA)**: can an attacker tell, from loss/perplexity, that $x \in D_f$ was once in training? If forget-set perplexity is still anomalously low, the data's fingerprint remains. (See [Privacy, Memorization & Differential Privacy for LLMs](../13-interp-safety-gov/03-privacy-memorization-dp.html).)
2. **Relearning / fine-tuning attacks**: a few gradient steps on a *tiny* sample of the forgotten data. If the model snaps back to full recall almost instantly, the knowledge was suppressed, not removed — a damning result, since true removal should require relearning from near-scratch.
3. **Jailbreak / paraphrase elicitation**: ask in another language, via role-play, or with an indirect prompt. Output-only unlearning leaks under these constantly.

!!! interview "Interview Corner"
    **Q:** Your team ran gradient-ascent unlearning to comply with a deletion request. The forget-set perplexity went up and the model refuses the direct question. Legal asks: "Is the data gone?" What do you tell them, and what tests do you run?

    **A:** I'd say *not yet demonstrated*. Higher perplexity and a refusal show the **output** changed, not that the data's **influence** was removed — those are different claims, and the legal standard (GDPR-style erasure) is about influence. I'd run three adversarial audits before signing off: (1) a **membership-inference attack** — if the forget examples are still distinguishable from never-seen data by their loss, the fingerprint persists; (2) a **relearning attack** — fine-tune on a handful of the forgotten samples; if recall returns in a few steps, we suppressed rather than removed; and (3) **paraphrase/cross-lingual/jailbreak elicitation** to check the knowledge isn't reachable by another path. I'd benchmark against a **retrained-from-scratch reference** (the TOFU/MUSE gold standard) where feasible. I'd also caution that approximate unlearning carries **no formal guarantee**; if the compliance bar is strict, we may need data-sharding (SISA) or differential-privacy training so deletion has provable semantics, and I'd document the residual risk honestly rather than overclaim.

---

## 7. From Scratch: A Minimal ROME-Style Rank-One Edit

Here is a complete, runnable rank-one editor on a small GPT-2 from HuggingFace. It implements the ROME recipe end to end: gather the key, optimize the value, solve the closed-form update, splice it into $W_{\text{down}}$, and verify reliability *and* locality. It is deliberately compact (no covariance estimation from a corpus — we approximate $C$ with a regularized identity, which is the well-known "ROME without statistics" simplification) so the mechanism is legible.

```python
# minimal_rome.py — a from-scratch rank-one factual edit on GPT-2.
# pip install torch transformers
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()

# We edit the down-projection (c_proj) of one middle MLP block.
LAYER = 6                                   # a middle layer (GPT-2 small has 12)
mlp = model.transformer.h[LAYER].mlp        # has c_fc (up) and c_proj (down)
# NOTE: GPT-2 Conv1D stores weight as [in, out]; W_down maps d_mlp -> d_model.
W_down = mlp.c_proj.weight                  # shape [d_mlp=3072, d_model=768]

@torch.no_grad()
def generate(prompt, n=12):
    ids = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(**ids, max_new_tokens=n, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=True)

# ---------------------------------------------------------------------------
# 1) Capture the KEY k* : the input to c_proj at the subject's last token.
#    The input to c_proj is exactly act = gelu(c_fc(h)), i.e. the MLP hidden.
# ---------------------------------------------------------------------------
SUBJECT = "The Eiffel Tower"
PROMPT  = "The Eiffel Tower is located in the city of"
TARGET  = " Rome"                            # the (false) object we inject

captured = {}
def hook_capture(module, inp, out):
    captured["k"] = inp[0].detach()          # input to c_proj: [batch, seq, d_mlp]
h = mlp.c_proj.register_forward_hook(hook_capture)
ids = tok(PROMPT, return_tensors="pt").to(device)
with torch.no_grad():
    model(**ids)
h.remove()
# last subject token = last token of "The Eiffel Tower" within the prompt.
subj_len = tok(SUBJECT, return_tensors="pt")["input_ids"].shape[1]
k_star = captured["k"][0, subj_len - 1].clone()     # [d_mlp]

# ---------------------------------------------------------------------------
# 2) Optimize the VALUE v* : find the c_proj-output vector at that position
#    that makes the model emit TARGET, via a free delta added at the edit site.
# ---------------------------------------------------------------------------
target_id = tok(TARGET, return_tensors="pt")["input_ids"][0, 0].to(device)
delta = torch.zeros(W_down.shape[1], device=device, requires_grad=True)  # d_model
opt = torch.optim.Adam([delta], lr=5e-1)

edit_pos = subj_len - 1
def hook_add_delta(module, inp, out):
    out = out.clone()
    out[0, edit_pos] = out[0, edit_pos] + delta          # add delta to c_proj output
    return out

for step in range(25):
    hd = mlp.c_proj.register_forward_hook(hook_add_delta)
    logits = model(**ids).logits
    hd.remove()
    # cross-entropy of TARGET at the final prompt position
    loss = F.cross_entropy(logits[0, -1:].float(), target_id.view(1))
    loss = loss + 1e-3 * delta.pow(2).sum()              # light norm penalty
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 6 == 0:
        print(f"  v* opt step {step:2d}  loss={loss.item():.3f}")

# v* is the *desired output* of c_proj at the edit position = current out + delta.
with torch.no_grad():
    hd = mlp.c_proj.register_forward_hook(hook_capture)  # reuse to grab current out
    # capture c_proj OUTPUT this time:
    def hook_out(module, inp, out): captured["o"] = out.detach()
    h2 = mlp.c_proj.register_forward_hook(hook_out)
    model(**ids); h2.remove(); hd.remove()
    v_star = (captured["o"][0, edit_pos] + delta).detach()   # [d_model]

# ---------------------------------------------------------------------------
# 3) Closed-form RANK-ONE update of W_down.
#    Conv1D weight is [d_mlp, d_model] and computes h @ W, so W maps k* (d_mlp)
#    to an output of size d_model via  out = k*^T W.  We want  k*^T (W+Δ) = v*.
#    With C ≈ (covariance), use the ROME solution Δ = (C^{-1} k*) (v* - W^T k*)^T
#    / ((C^{-1} k*)·k*).  We approximate C^{-1} ≈ I / (||k*||^2-scale).
# ---------------------------------------------------------------------------
with torch.no_grad():
    Cinv_k = k_star / (k_star.dot(k_star) + 1e-4)        # I-approx of C^{-1} k*
    Wk = W_down.t() @ k_star                             # current output for k*  [d_model]
    residual = v_star - Wk                               # what we must add        [d_model]
    denom = Cinv_k.dot(k_star)                           # scalar
    update = torch.outer(Cinv_k, residual) / denom       # [d_mlp, d_model], rank 1
    print(f"\nrank-1 update: ||Δ||_F = {update.norm().item():.3f}, "
          f"||W||_F = {W_down.norm().item():.1f}")
    W_down.add_(update)                                  # SPLICE THE EDIT

# ---------------------------------------------------------------------------
# 4) Verify: reliability (edit took) + locality (an unrelated fact survives).
# ---------------------------------------------------------------------------
print("\n[edited]   ", generate("The Eiffel Tower is located in the city of"))
print("[generalize]", generate("Where is the Eiffel Tower? It is in"))
print("[locality] ", generate("The Colosseum is located in the city of"))
```

What to watch when you run it: the **[edited]** line should now say *Rome*; the **[generalize]** line *often* but not always follows (rank-one edits generalize imperfectly — that's the Section 5 lesson live); and the **[locality]** line should still correctly say *Rome* for the Colosseum was already Rome, so pick an unrelated subject like "The Statue of Liberty is located in the city of" to confirm it still says *New York*. The Frobenius norm of $\Delta$ printed in step 3 will be tiny relative to $\lVert W\rVert_F$ — the edit is a whisper to the weight matrix, which is exactly why locality is even possible.

!!! note "Aside: production editors use real covariance"
    The identity-approximation above is the textbook simplification. Real ROME/MEMIT precompute $C = KK^\top$ from tens of thousands of Wikipedia activations so the $C^{-1}k_*$ term genuinely steers the update into *under-used* directions. The HuggingFace-ecosystem library **EasyEdit** (Zhang et al.) packages ROME, MEMIT, GRACE, WISE, and more behind one API and ships these covariance statistics — reach for it before hand-rolling anything for real use.

---

## 8. When (Not) to Edit: A Decision Guide

Parametric editing is one tool among several, and frequently the wrong one. A short decision tree:


{{fig:kedit-when-not-to-edit-tree}}


The recurring meta-lesson: **editing changes associations, not beliefs, and suppresses outputs, not knowledge.** Treat every edit as a hypothesis to be falsified by ripple-effect and adversarial-elicitation tests, not as a fact you have installed. For anything with legal weight, document the residual risk — approximate methods come with *no* formal guarantee, and an auditor with a relearning attack can often prove it.

!!! key "Key Takeaways"
    - **Locate-then-edit** rests on the finding that simple (subject, relation, object) facts are stored locally in **middle-layer MLPs** at the last subject token; **causal tracing** localizes them, but localization tells you where a fact is *read*, not the unique place it must be *written*.
    - **ROME** makes a **closed-form rank-one** update to one MLP down-projection: optimize a target value $v_*$, capture the key $k_*$, and add $\Delta = (v_* - W_0 k_*)(C^{-1}k_*)^\top / \big((C^{-1}k_*)^\top k_*\big)$, where $C$ steers the edit into under-used directions to protect locality.
    - **MEMIT** scales to thousands of facts by solving a low-rank least-squares update spread across a *band* of layers; **AlphaEdit** adds a **null-space projection** so updates don't disturb preserved knowledge, enabling long sequential editing.
    - **Memory/adapter editors** (**GRACE**'s activation codebook, **WISE**'s routed side memory) keep base weights frozen — better for lifelong editing, auditability, and rollback, at the cost of an inference-time lookup.
    - The hard part is **side effects**: **ripple effects** (multi-hop consequences stay inconsistent), **locality** violations (unrelated facts move), and **forgetting/collapse** under accumulated edits. A good edit must satisfy reliability, generalization, locality, *and* fluency at once.
    - **Machine unlearning** removes the *influence* of data; the gold standard is **indistinguishability from a model retrained without it**. Practical recipes — gradient ascent, gradient difference, **NPO**, **RMU**, editing-as-unlearning — are all *approximate* with no formal guarantee.
    - "The model won't say it" ≠ "the model unlearned it." Audit with **membership inference**, **relearning/fine-tuning attacks**, and **paraphrase/cross-lingual elicitation**; benchmark on **TOFU** and **MUSE**.
    - When a fact changes often or you must *prove* deletion, prefer **retrieval** or **SISA/DP training** over parametric editing — editing changes associations and suppresses outputs, not beliefs and knowledge.

!!! sota "State of the Art & Resources (2026)"
    Knowledge editing and machine unlearning have matured from proof-of-concept weight surgery (ROME, 2022) into a field with dedicated benchmarks, multi-method frameworks, and open safety applications — but the gap between "the model won't say it" and "the model provably forgot it" remains an active frontier with no formal guarantees.

    **Foundational work**

    - [Geva et al., *Transformer Feed-Forward Layers Are Key-Value Memories* (2021)](https://arxiv.org/abs/2012.14913) — the reinterpretation of MLP layers as key–value stores that underpins all locate-then-edit methods.
    - [Meng et al., *Locating and Editing Factual Associations in GPT* (ROME, 2022)](https://arxiv.org/abs/2202.05262) — introduced causal tracing and the rank-one closed-form weight update; the foundational locate-then-edit paper.
    - [Meng et al., *Mass-Editing Memory in a Transformer* (MEMIT, 2022)](https://arxiv.org/abs/2210.07229) — scales ROME to thousands of simultaneous edits spread across a band of layers.

    **Recent advances (2023–2026)**

    - [Hartvigsen et al., *Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adaptors* (NeurIPS 2023)](https://arxiv.org/abs/2211.11031) — memory-adapter approach that enables thousands of sequential edits while keeping base weights frozen.
    - [Cohen et al., *Evaluating the Ripple Effects of Knowledge Editing in Language Models* (TACL 2024)](https://arxiv.org/abs/2307.12976) — introduced the RippleEdits benchmark showing that editing one fact leaves multi-hop consequences inconsistent.
    - [Fang et al., *AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models* (2024)](https://arxiv.org/abs/2410.02355) — projects weight updates onto the null space of preserved-knowledge keys, dramatically reducing drift under long sequential editing.
    - [Li et al., *The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning* (ICML 2024)](https://arxiv.org/abs/2403.03218) — benchmark for hazardous-capability removal and source of the RMU representation-misdirection unlearning method.
    - [Maini et al., *TOFU: A Task of Fictitious Unlearning for LLMs* (2024)](https://arxiv.org/abs/2401.06121) — clean-room unlearning benchmark using synthetic author biographies with retrained-from-scratch gold standard.
    - [Shi et al., *MUSE: Machine Unlearning Six-Way Evaluation for Language Models* (ICML 2024)](https://arxiv.org/abs/2407.06460) — six-desiderata evaluation (verbatim memorization, knowledge memorization, privacy, utility, scalability, sequential robustness) on realistic corpora.

    **Open-source & tools**

    - [zjunlp/EasyEdit](https://github.com/zjunlp/EasyEdit) — ACL 2024 framework unifying ROME, MEMIT, GRACE, WISE, and others behind one API, including precomputed layer statistics; the standard starting point for practitioners.

    **Go deeper**

    - [Zhang et al., *A Comprehensive Study of Knowledge Editing for Large Language Models* (2024)](https://arxiv.org/abs/2401.01286) — large-scale empirical comparison of 12 editing methods with the KnowEdit benchmark; essential reading before choosing an approach.

## Further reading

- Meng, Bau, Andonian & Belinkov — *Locating and Editing Factual Associations in GPT* (ROME), 2022.
- Meng, Sharma, Andonian, Belinkov & Bau — *Mass-Editing Memory in a Transformer* (MEMIT), 2022.
- Geva, Schuster, Berant & Levy — *Transformer Feed-Forward Layers Are Key-Value Memories*, 2021.
- Hase, Bansal et al. — *Does Localization Inform Editing?*, 2023.
- Cohen, Biran, Yoran, Globerson & Geva — *Evaluating the Ripple Effects of Knowledge Editing in Language Models* (RippleEdits), 2023.
- Hartvigsen, Sankaranarayanan et al. — *Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adaptors*, 2023.
- Wang et al. — *WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of LLMs*, 2024.
- Fang et al. — *AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models*, 2024.
- Bourtoule et al. — *Machine Unlearning* (SISA), 2021.
- Maini, Feng, Schwarzschild, Lipton & Kolter — *TOFU: A Task of Fictitious Unlearning for LLMs*, 2024.
- Shi et al. — *MUSE: Machine Unlearning Six-Way Evaluation for Language Models*, 2024.
- Zhang et al. — *Negative Preference Optimization* (NPO), 2024.
- Li et al. — *The WMDP Benchmark* (and RMU for unlearning), 2024.
- Zhang et al. — *EasyEdit: An Easy-to-use Knowledge Editing Framework for LLMs*.
