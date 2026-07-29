# 13.3 Privacy, Memorization & Differential Privacy for LLMs

Train a 7-billion-parameter model on a few trillion tokens of web text and you have built, among other things, a *lossy database of its training set*. Ask it the right way and it will hand back verbatim chunks of that data: a stranger's email signature with their phone number, a leaked API key from a GitHub commit, a paragraph of a copyrighted novel, a patient's discharge summary that slipped into a scraped forum. This is not a bug in any single line of code. It is a direct, *quantifiable* consequence of fitting a high-capacity function to data with a maximum-likelihood objective. The model that generalizes best is, all else equal, the model that has also memorized the most.

That tension — generalization requires fitting the data, privacy requires *not* fitting any single example too well — is the subject of this chapter. We will make the threat concrete first: what memorization is, when and why it happens, and the family of attacks (extraction, membership inference, reconstruction) that turn it into a privacy breach. Then we audit it: canaries and the secret-sharer methodology that let you *measure* leakage before an attacker does. Then we defend it across the whole lifecycle — data sanitization, deduplication, differentially private training (DP-SGD), and inference-time mitigations — being honest about the utility and compute costs each one extracts. Finally we frame it for the lawyers: threat models and the compliance scaffolding (GDPR, HIPAA) that turns "the model leaked PII" into a reportable incident.

This chapter sits next to several others. The mechanics of *removing* knowledge from an already-trained model live in [Knowledge Editing & Machine Unlearning](../13-interp-safety-gov/02-knowledge-editing-unlearning.html); the production redaction and guardrail layer is in [Safety, Guardrails & Content Moderation](../12-production-mlops/04-safety-guardrails.html); the data-pipeline deduplication that is your cheapest privacy defense is in [Data Cleaning, Deduplication & Quality Filtering](../03-pretraining/02-data-cleaning-dedup.html); and the legal/regulatory superstructure is in [AI Governance, Compliance & Regulation](../13-interp-safety-gov/06-governance-compliance.html).

---

## 1. What Memorization Is, and Why Models Do It

Let us be precise. We say a model $f_\theta$ has **memorized** a training string $s$ if knowledge of $s$ being in the training set changes the model's behavior on $s$ in a way that knowledge of "similar" strings does not. Two operational definitions dominate the literature.

**Eidetic / verbatim memorization (extractable memorization).** A string $s$ of length $\ell$ tokens is *$k$-eidetic memorized* if there exists a prompt $p$ of length at most some budget such that greedy decoding of $f_\theta(\cdot \mid p)$ reproduces $s$ exactly, and $s$ appears in at most $k$ training documents. The smaller the $k$, the more alarming: a string that appeared *once* and is still emitted verbatim is unambiguous memorization, not a learned statistical regularity. This is the definition behind the canonical extraction attacks.

**Counterfactual memorization.** Following the influence-function view, the memorization of example $z=(x,y)$ is the gap between how well a model predicts $z$ when $z$ is in the training set versus when it is held out:

$$
\operatorname{mem}(z) \;=\; \underbrace{\mathbb{E}_{\theta \sim \mathcal{A}(D \cup \{z\})}\big[\,\text{perf}(f_\theta, z)\,\big]}_{\text{trained on }z} \;-\; \underbrace{\mathbb{E}_{\theta \sim \mathcal{A}(D \setminus \{z\})}\big[\,\text{perf}(f_\theta, z)\,\big]}_{\text{trained without }z}
$$

where $\mathcal{A}$ is the (randomized) training algorithm and $\text{perf}$ is, say, the negative loss. A high gap means the model's competence on $z$ comes *specifically* from having seen $z$, not from generalization. This is the quantity differential privacy is designed to bound.

### 1.1 Why memorization is inevitable under maximum likelihood

The pretraining loss is the per-token cross-entropy (see [The Pretraining Objective & Loss](../03-pretraining/03-pretraining-objective.html)):

$$
\mathcal{L}(\theta) = -\frac{1}{N}\sum_{i=1}^{N} \log p_\theta(x_i \mid x_{<i}).
$$

Gradient descent drives $p_\theta(x_i \mid x_{<i}) \to 1$ for tokens it can fit. For a *unique, low-entropy, high-surprise* sequence — a random 16-digit credit-card number, a UUID, a GitHub token — there is no generalizable rule that predicts the next digit; the only way to drive its loss down is to *store* it in the weights. Modern LLMs are heavily over-parameterized relative to the information content of any single rare sequence, so there is ample capacity to do exactly that. Memorization is therefore not pathological overfitting that early stopping cures; it is the *optimal* behavior of the objective on rare sequences, and it appears well before the model overfits in the classical (rising-validation-loss) sense.

### 1.2 The three things that drive memorization

Empirically, three factors dominate how much a given string gets memorized:

1. **Duplication.** This is the single biggest lever. The *fraction* of a sequence that is extractable grows roughly **log-linearly** in its duplication count $n$ (Carlini et al., 2022), while the *rate* at which the model spontaneously regurgitates it grows far faster than linearly: Kandpal et al. (2022) found that a sequence appearing on the order of ten times in the corpus is generated on the order of a thousand times more often than one appearing once. Deduplicating the training set is the highest-ROI privacy intervention you have, and it is free (you were going to dedup for quality anyway).
2. **Model scale.** Larger models memorize more, at every duplication level. Roughly, the fraction of extractable training data grows with $\log(\text{parameters})$. A 6B model memorizes substantially more than a 125M model trained on the identical data.
3. **Context length (prompt length).** The longer the prefix you give the model that matches the training context, the more likely it completes the memorized continuation. Memorization is "unlocked" by sufficiently long, in-distribution prompts.

{{fig:privacy-memorization-scaling}}

!!! note "Aside: memorization ≠ overfitting"
    A model can have a perfectly healthy, still-decreasing validation loss and yet have memorized thousands of unique sequences verbatim. The two phenomena are decoupled because memorization concerns the *tail* of rare examples while validation loss is an *average* over the bulk. This is why you cannot rely on standard regularization (weight decay, dropout) or early stopping to make privacy go away — they target the wrong statistic.

---

## 2. The Attack Surface: Extraction, Membership Inference, Reconstruction

Memorization is only a *privacy* problem when an adversary can exploit it. There are three canonical attack families, ordered roughly by how much they extract.

### 2.1 Training-data extraction

The goal: recover *some* verbatim training strings without knowing them in advance. The Carlini et al. extraction attack on GPT-2 is the template:

1. **Generate** a large pool of candidate completions by sampling from the model (optionally with diverse prompts, temperature schedules, or internet-scraped prefixes).
2. **Score / rank** candidates by a *membership signal* — a heuristic that a string is memorized rather than fluent-but-generic.
3. **Verify** the top-ranked candidates against the (or a) training corpus, or by checking they are low-$k$.

The crucial insight is the *ranking step*. Raw model perplexity over-selects generic, high-likelihood text ("the the the"). The fix is to compare the target model's likelihood against a *reference*: a string is suspicious if the model finds it *much* more likely than a baseline expects. Common membership signals:

- **zlib ratio:** $\dfrac{\log p_\theta(s)}{\text{zlib\_entropy}(s)}$ — penalizes strings that are intrinsically compressible (and so trivially low-perplexity).
- **Reference-model ratio:** $\log p_\theta(s) - \log p_{\theta_{\text{ref}}}(s)$, where $\theta_{\text{ref}}$ is a smaller or differently-trained model. Memorized strings score high because the target memorized them and the reference did not.
- **Lowercase / windowing ratios:** compare $p_\theta(s)$ to $p_\theta$ of a perturbed version of $s$.

A 2023 result extended this to *aligned* production models with a "divergence attack": prompting an aligned chat model to repeat a token forever (`"poem poem poem ..."`) caused it to diverge from its alignment and emit memorized pretraining data. The lesson: alignment training masks but does not remove memorization.

### 2.2 Membership inference attacks (MIA)

The goal here is narrower but more general: given a *specific* candidate record $z$, decide whether $z$ was in the training set. This is the workhorse privacy attack — it is the empirical instantiation of the differential-privacy adversary, and "did my data train this model?" is exactly the question GDPR and copyright litigation care about.

The simplest MIA is the **loss threshold (LOSS attack):** members have lower loss than non-members, so predict "member" if $\mathcal{L}(f_\theta, z) < \tau$. This is weak because loss depends heavily on the *intrinsic* difficulty of $z$ — an easy sentence has low loss whether or not it was trained on.

State-of-the-art MIAs *calibrate away* that intrinsic difficulty. The **Likelihood Ratio Attack (LiRA)** is the gold standard: train many "shadow" models, half including $z$ (IN) and half excluding it (OUT), fit Gaussians to the per-example loss distributions, and run a likelihood-ratio test:

$$
\Lambda(z) = \frac{p\big(\ell(z)\mid \mathcal{N}(\mu_{\text{in}},\sigma_{\text{in}}^2)\big)}{p\big(\ell(z)\mid \mathcal{N}(\mu_{\text{out}},\sigma_{\text{out}}^2)\big)}.
$$

The key methodological point — and a frequent interview trap — is that **MIA success must be measured by true-positive rate at very low false-positive rate (TPR @ low FPR)**, plotted on a log-log ROC, *not* by average accuracy. A privacy attack that is correct 50.5% of the time on average but achieves 100× the chance TPR at 0.1% FPR is a serious breach for the few people in that high-confidence tail. Average accuracy hides exactly the worst-case leakage that matters.

{{fig:mia-tpr-low-fpr-roc}}

You rarely need to reimplement the attack zoo to get a baseline. The open-source **MIMIR** suite (Duan et al., 2024) packages the standard LLM MIAs — LOSS, reference/ratio, zlib, min-$k$% probability, and neighbourhood attacks — against Pythia checkpoints with *known* train/test splits, and is the right thing to benchmark your own attack against. Its headline result is itself a rigor lesson: against a real pretrained LLM (huge corpus, roughly one epoch, deduplicated), most published MIAs perform **close to chance**, and a large share of the apparent success in the literature comes from *distribution shift* between the "member" and "non-member" sets (members scraped earlier, non-members later) rather than from memorization. When you build your own audit, draw members and non-members from the same distribution and time window — otherwise you are measuring a topic classifier, not a privacy leak.

### 2.3 Reconstruction and attribute inference

Reconstruction is the strongest attack: recover an input *given partial knowledge*. In LLMs this shows up as **prompt-completion reconstruction** — give the model `"John Smith's social security number is "` and see if it completes the real digits. **Attribute inference** is the inverse: infer a sensitive attribute of a known individual from the model's behavior. These are the attacks that turn "the model memorized a document" into "the model deanonymized a person."

{{fig:privacy-attack-ladder}}

!!! warning "Common pitfall: reporting average-case MIA accuracy"
    If a paper or your own audit reports "our MIA gets 51% accuracy, so the model is basically private," be skeptical. The privacy-relevant question is the *low-FPR tail*. Always compute and plot TPR at FPR ∈ {0.001, 0.01} on a log-scaled ROC. A model can look private on average and still confidently leak membership for the most vulnerable (most-outlier, least-deduplicated) records — which are exactly the records a regulator or plaintiff will surface.

---

## 3. Auditing Privacy: Canaries and the Secret Sharer

You cannot fix what you cannot measure. The dominant methodology for *measuring* a training pipeline's leakage is **canary insertion**, from Carlini et al.'s "The Secret Sharer."

A **canary** is a synthetic, out-of-distribution secret you deliberately insert into the training data — for example, `"The access code is 7-2-9-3-5-1-8-4"` with a *randomly chosen* numeric body drawn from a known format $R$ (here, 8 digits, so $|R| = 10^8$). After training, you measure how much the model has memorized the canary using its **exposure**:

$$
\operatorname{exposure}_\theta(s) \;=\; \log_2 |R| \;-\; \log_2 \operatorname{rank}_\theta(s),
$$

where $\operatorname{rank}_\theta(s)$ is the rank of the true canary $s$ among all $|R|$ candidate fill-ins, sorted by model log-likelihood (lowest perplexity = rank 1). Intuitively:

- If the model has *not* memorized the canary, its rank is roughly uniform in $[1, |R|]$, so $\mathbb{E}[\operatorname{rank}] \approx |R|/2$ and exposure $\approx 1$ bit.
- If the model has *fully* memorized it, $\operatorname{rank} = 1$ and exposure $= \log_2 |R|$ bits (e.g. $\approx 26.6$ bits for $10^8$). **The canary is now extractable by brute-force enumeration.**

Exposure is wonderful because it is a *continuous, calibrated* signal — you do not need to wait until the model emits the canary verbatim; you can watch exposure climb during training and catch the leak early. And computing exact rank does not require enumerating all $|R|$ candidates: under a log-normal model of the perplexity distribution you can estimate rank from a small sample.

!!! example "Worked example: from exposure to extractability"
    Insert a canary `"my secret key is XXXXXXXXX"` whose body is 9 random digits, so $|R| = 10^9$ and $\log_2 |R| \approx 29.9$ bits. We train, then rank the true body among all $10^9$ candidates by model perplexity.

    - **Case A — not memorized:** true body lands at rank $\approx 5\times10^8$. Exposure $= 29.9 - \log_2(5\times10^8) = 29.9 - 28.9 = 1.0$ bit. Safe.
    - **Case B — partial:** rank $= 1000$. Exposure $= 29.9 - \log_2(1000) = 29.9 - 9.97 = 19.9$ bits. An attacker who can make $1000$ guesses extracts it. Alarming.
    - **Case C — full:** rank $= 1$. Exposure $= 29.9$ bits. Greedy decoding emits the secret. Breach.

    Now the duplication knob: insert the same canary **once** vs. **nine times**. In the original Secret Sharer experiments, inserting a canary a handful of times in a large corpus already pushed exposure from $\approx 1$ bit toward the full $\log_2|R|$, while a single insertion in a well-deduplicated corpus often stayed near baseline. This is the empirical core of "dedup is your best cheap defense."

```python
import math
import torch
import torch.nn.functional as F

@torch.no_grad()
def sequence_logprob(model, tokenizer, text, device="cuda"):
    """Total log p_theta(text) in nats, summed over the conditional next-token terms."""
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    logits = model(ids).logits                      # (1, T, V)
    logprobs = F.log_softmax(logits[:, :-1], dim=-1)  # predict token t from <t
    target = ids[:, 1:]                               # shifted targets
    tok_lp = logprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # (1, T-1)
    return tok_lp.sum().item()                         # scalar, nats

def canary_exposure(model, tokenizer, template, true_body, body_space_size,
                    n_samples=20000, rng=None, device="cuda"):
    """
    Estimate exposure of a canary by Monte-Carlo rank estimation.

    template:        e.g. "the access code is {}"
    true_body:       the actual inserted secret, e.g. "72935184"
    body_space_size: |R|, the number of possible bodies (e.g. 10**8)

    We sample candidate bodies, score them, and estimate the rank of the true
    body via a log-normal fit of the perplexity distribution (Carlini et al.).
    """
    import random
    rng = rng or random.Random(0)
    n_digits = len(true_body)

    def body_logprob(body):
        return sequence_logprob(model, tokenizer, template.format(body), device)

    true_lp = body_logprob(true_body)

    # Sample a reference set of random bodies and collect their log-probs.
    sample_lps = []
    for _ in range(n_samples):
        rand_body = "".join(str(rng.randint(0, 9)) for _ in range(n_digits))
        sample_lps.append(body_logprob(rand_body))

    sample = torch.tensor(sample_lps)
    mu, sigma = sample.mean().item(), sample.std().item() + 1e-9

    # Fraction of the WHOLE space expected to score >= the true canary, under a
    # Gaussian fit to log-prob. P(X >= true_lp) = 1 - Phi((true_lp - mu)/sigma).
    z = (true_lp - mu) / sigma
    frac_higher = 0.5 * math.erfc(z / math.sqrt(2.0))   # upper-tail of normal
    est_rank = max(1.0, frac_higher * body_space_size)

    exposure = math.log2(body_space_size) - math.log2(est_rank)
    return exposure, est_rank, true_lp, mu

# Usage sketch (pseudo — plug in a real HF model/tokenizer you trained with a canary):
#   exp, rank, _, _ = canary_exposure(model, tok,
#                                     template="the access code is {}",
#                                     true_body="72935184",
#                                     body_space_size=10**8)
#   print(f"exposure = {exp:.1f} bits, estimated rank = {rank:.0f} of 1e8")
#   # exposure near 1.0  -> safe;  near 26.6 -> fully memorized / extractable.
```

The audit recipe in production: insert a *family* of canaries (different formats, different duplication counts, placed at different points in training) and report the exposure distribution. Rising exposure on low-duplication canaries is your early-warning system that the pipeline memorizes too aggressively — *before* a real secret leaks.

This is one of the few privacy experiments you can genuinely run end to end at hobby scale, and the capstone is the place to do it: inject the canary family while the corpus is already streaming through the filter/dedup stages of [Data: Sourcing, Filtering, Dedup, Tokenize & Pack ~20B Tokens](../14-capstone/02-data-pipeline.html) (record each canary's insertion count), then score exposure at every checkpoint written by [The Pretraining Run: A Complete Single-GPU Training Loop](../14-capstone/07-pretraining-run.html). At ~100M parameters the whole sweep costs minutes of GPU time, and the exposure-vs-duplication curve you get out is a direct, personally-verified measurement of the log-linear duplication law from §1.2 — plus proof that your dedup actually did something.

{{fig:canary-exposure-rank-ladder}}

---

## 4. Defenses Across the Lifecycle

There is no single switch. Privacy is defended in layers, each at a different point in the pipeline, each with a different cost. We go from cheapest-and-most-effective to most-expensive-and-most-rigorous.

### 4.1 Data-level: deduplication and PII scrubbing

**Deduplication** is the highest-ROI defense, full stop. Because memorization scales with duplication, removing near-duplicate documents collapses the most extractable tail. Use suffix-array exact substring dedup and MinHash/LSH near-dedup (the full machinery is in [Data Cleaning, Deduplication & Quality Filtering](../03-pretraining/02-data-cleaning-dedup.html)). Deduplicating a corpus can reduce extractable memorization by an order of magnitude at essentially zero utility cost — indeed it usually *improves* quality.

**PII scrubbing** removes the secrets before training. Detect-and-redact pipelines combine high-recall regexes (emails, phone numbers, SSNs, credit cards, API-key formats) with a named-entity-recognition (NER) model for free-form PII (names, addresses). This is the same Presidio-style stack used at inference time in [Safety, Guardrails & Content Moderation](../12-production-mlops/04-safety-guardrails.html).

```python
import re

# High-recall regexes for the structured tail of PII. These over-redact on
# purpose: false positives (a redacted non-secret) are far cheaper than false
# negatives (a leaked SSN) when training a model whose weights are forever.
PII_PATTERNS = {
    "EMAIL":  re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "PHONE":  re.compile(r"\b(?:\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"),
    "SSN":    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CCARD":  re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    # Common secret-key shapes (AWS, generic 32-64 hex, Slack/GitHub-ish tokens):
    "AWSKEY": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "HEX32":  re.compile(r"\b[0-9a-fA-F]{32,64}\b"),
    "GHPAT":  re.compile(r"\bghp_[0-9A-Za-z]{36}\b"),
}

def scrub_pii(text: str) -> tuple[str, dict]:
    """Replace structured PII with typed placeholders. Returns (clean, counts).

    Order matters: redact the most specific patterns first so a credit-card
    number is not partially eaten by the phone regex.
    """
    counts = {}
    for label in ("AWSKEY", "GHPAT", "SSN", "CCARD", "EMAIL", "PHONE", "HEX32"):
        pattern = PII_PATTERNS[label]
        text, n = pattern.subn(f"<{label}>", text)
        if n:
            counts[label] = counts.get(label, 0) + n
    return text, counts

# For free-form PII (person names, locations, orgs) layer an NER model on top:
#   import spacy; nlp = spacy.load("en_core_web_trf")
#   doc = nlp(text)
#   for ent in reversed(doc.ents):        # reverse so offsets stay valid
#       if ent.label_ in {"PERSON", "GPE", "ORG"}:
#           text = text[:ent.start_char] + f"<{ent.label_}>" + text[ent.end_char:]

sample = "Contact john.doe@acme.io or 415-555-0199; key AKIAIOSFODNN7EXAMPLE; ssn 123-45-6789."
clean, counts = scrub_pii(sample)
print(clean)   # Contact <EMAIL> or <PHONE>; key <AWSKEY>; ssn <SSN>.
print(counts)  # {'AWSKEY': 1, 'SSN': 1, 'EMAIL': 1, 'PHONE': 1}
```

In production you do not hand-roll this. **Microsoft Presidio** is the standard open-source PII stack: a set of recognizers (regex + checksum validators + spaCy/transformer NER + context words) feeding a confidence score per detected entity, and a separate anonymizer that applies a per-entity-type operator (replace, mask, hash, encrypt).

```python
# pip install presidio-analyzer presidio-anonymizer && python -m spacy download en_core_web_lg
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer, anonymizer = AnalyzerEngine(), AnonymizerEngine()

sample = "Contact john.doe@acme.io or 415-555-0199; ssn 123-45-6789."
results = analyzer.analyze(text=sample, language="en")   # list of (entity_type, start, end, score)
results = [r for r in results if r.score >= 0.6]         # tune the precision/recall knob
redacted = anonymizer.anonymize(text=sample, analyzer_results=results).text
print(redacted)   # entities replaced by <EMAIL_ADDRESS>, <PHONE_NUMBER>, <US_SSN>, ...
```

Two things Presidio buys you that a bare regex cannot: **validators** (a 16-digit run is only reported as a credit card if it passes the Luhn checksum, killing most false positives) and a **thresholdable confidence score**, so you can pick your operating point on the precision/recall curve instead of accepting whatever a regex happens to match. Add custom recognizers for your own secret formats (internal ticket IDs, API-key prefixes). At corpus scale, run the scrubber *inside* the pipeline you already stream through rather than as a separate pass — `datatrove` ships a PII-masking formatter block (emails, IP addresses) that slots in next to the dedup and quality-filter stages of [Data Cleaning, Deduplication & Quality Filtering](../03-pretraining/02-data-cleaning-dedup.html), so scrubbing costs one extra map over data you are already touching.

The limitation is fundamental: scrubbing only removes the PII *you can detect*. It cannot catch a phone number written as "four one five, five five five..." or a name that is also a common word. Scrubbing is necessary, high-value, and **not sufficient**. It also gives no formal guarantee — which is what motivates the next layer.

### 4.2 Training-level: differential privacy and DP-SGD

**Differential privacy (DP)** is the only defense with a *formal, worst-case, composable* guarantee. A randomized training algorithm $\mathcal{A}$ is **$(\varepsilon,\delta)$-differentially private** if for any two datasets $D, D'$ differing in a single record, and any set of outcomes $S$:

$$
\Pr[\mathcal{A}(D) \in S] \;\le\; e^{\varepsilon}\,\Pr[\mathcal{A}(D') \in S] \;+\; \delta.
$$

Read it operationally: *no attacker, however powerful, with any side information, can tell whether any one record was in the training set with confidence beyond a factor of $e^\varepsilon$* (up to failure probability $\delta$). This directly upper-bounds the success of *every* membership-inference attack — DP is precisely a worst-case bound on the MIA adversary of §2.2. Smaller $\varepsilon$ = more privacy. As a rule of thumb $\varepsilon \le 1$ is strong, $\varepsilon \approx 8$ is "meaningful but loose," and $\varepsilon \gg 50$ is mostly cosmetic. $\delta$ should be $\ll 1/N$ (smaller than one over the dataset size).

The definition hides one decision you must make explicitly: **what counts as "a single record."** The clipping is per *example*, so the guarantee is example-level. If one user contributed $k$ documents to the fine-tune set, group privacy degrades their protection to roughly $k\varepsilon$ — often the number that actually matters legally, since GDPR protects the *person*, not the row. **User-level DP** fixes this by clipping each user's *aggregate* contribution (concatenate or average a user's per-example gradients, then clip that once, and sample users rather than examples); it costs more utility per unit $\varepsilon$ because there are fewer units to average over, and it is the regime Google's production DP fine-tuning work targets.

The algorithm that delivers DP for deep learning is **DP-SGD** (Abadi et al., 2016). It modifies ordinary SGD with two operations per step:

1. **Per-example gradient clipping.** Compute the gradient *for each example separately* and clip its $L_2$ norm to a bound $C$. This caps the influence any single example can have on the update:
   $$
   \tilde{g}_i \;=\; g_i \cdot \min\!\Big(1, \frac{C}{\lVert g_i \rVert_2}\Big).
   $$
2. **Gaussian noise addition.** Sum the clipped gradients, add calibrated Gaussian noise, and average:
   $$
   \hat{g} \;=\; \frac{1}{B}\Big(\sum_{i=1}^{B} \tilde{g}_i \;+\; \mathcal{N}\big(0,\; \sigma^2 C^2 \mathbf{I}\big)\Big).
   $$

The noise multiplier $\sigma$ together with the **sampling rate** $q = B/N$ and the number of steps $T$ determines $\varepsilon$ via a *privacy accountant* (the Rényi-DP / moments accountant, or the tighter PRV accountant). Each step "spends" privacy budget; composition adds it up over training.

!!! warning "Common pitfall: shuffled batches silently invalidate the reported ε"
    Every DP-SGD accountant analyses the **subsampled** Gaussian mechanism, and the standard analysis assumes **Poisson sampling**: each example is included independently with probability $q = B/N$, so the batch size is a *random* variable. The usual shuffle-and-chunk `DataLoader` is sampling *without* replacement with a fixed batch size — a different mechanism, and pairing it with a Poisson accountant means the $\varepsilon$ you report is an under-estimate of what you actually spent. Opacus handles this for you by swapping in a Poisson sampler (`UniformWithReplacementSampler`) inside `make_private*`; if you write the loop yourself, draw the batch with a Bernoulli($q$) mask and make sure your code tolerates the occasional empty batch.

??? note "Optional: computing ε yourself, so the accountant is not a black box"
    The Gaussian mechanism with $L_2$ sensitivity $C$ and noise std $\sigma C$ satisfies $\big(\alpha,\ \alpha/(2\sigma^2)\big)$-**Rényi DP** for every order $\alpha>1$, and RDP composes by simply *adding* the second argument. So $T$ steps give $\big(\alpha,\ T\alpha/(2\sigma^2)\big)$-RDP, and the standard RDP $\to(\varepsilon,\delta)$ conversion optimized over $\alpha$ gives a closed form:

    $$
    \varepsilon(\delta) \;=\; \min_{\alpha>1}\left[\frac{T\alpha}{2\sigma^2} + \frac{\log(1/\delta)}{\alpha-1}\right] \;=\; \frac{T}{2\sigma^2} \;+\; 2\sqrt{\frac{T}{2\sigma^2}\,\log\frac{1}{\delta}}\,.
    $$

    ```python
    import math

    def eps_no_subsampling(T, sigma, delta):
        """(eps, delta)-DP for T composed FULL-BATCH Gaussian steps, via RDP.
        Valid but very loose for DP-SGD: it ignores amplification by subsampling."""
        a = T / (2.0 * sigma ** 2)                      # RDP epsilon per unit alpha
        return a + 2.0 * math.sqrt(a * math.log(1.0 / delta))

    print(round(eps_no_subsampling(T=1465, sigma=0.8, delta=1e-6)))   # ~1396
    ```

    An $\varepsilon$ of ~1400 is meaningless — and that is exactly the point. Essentially *all* of DP-SGD's usable privacy comes from **amplification by subsampling**: when each example participates only with probability $q$, the per-step RDP drops from $O(\alpha/\sigma^2)$ to roughly $O(q^2\alpha/\sigma^2)$, a factor of order $q^2$ cheaper. The subsampled Gaussian has no clean closed form, so libraries evaluate it numerically: Opacus's `rdp` accountant integrates the Mironov et al. bound, while its `prv` accountant convolves the privacy-loss random variables directly and is typically tighter (Google's `dp_accounting` library implements the same family). That $q^2$ is why the worked example below lands at $\varepsilon\approx7.3$ with $q\approx2\times10^{-3}$ instead of at four digits — and why "just use a bigger batch" is not free: raising $B$ raises $q$ too.

```python
import torch

def dp_sgd_step(model, batch, loss_fn, optimizer,
                clip_norm=1.0, noise_multiplier=1.0, device="cuda"):
    """
    One DP-SGD step done explicitly (microbatch=1) to expose the mechanism.
    In production use Opacus (PrivacyEngine), which gets per-sample gradients
    via module hooks (GradSampleModule) or torch.func.vmap(grad(...)) instead
    of this Python loop, and tracks the privacy accountant for you. This loop
    is pedagogical, not fast.
    """
    xs, ys = batch                       # xs: (B, ...), ys: (B, ...)
    B = xs.size(0)
    # Accumulator for the summed, clipped gradients.
    summed = [torch.zeros_like(p) for p in model.parameters()]

    for i in range(B):                                   # PER-EXAMPLE gradients
        model.zero_grad(set_to_none=True)
        out = model(xs[i:i+1])
        loss = loss_fn(out, ys[i:i+1])
        loss.backward()                                  # grad of a single example

        # L2 norm of this example's full gradient (flattened across params).
        sq = sum((p.grad.detach() ** 2).sum() for p in model.parameters())
        total_norm = torch.sqrt(sq)
        scale = min(1.0, clip_norm / (total_norm + 1e-6))  # clip to C

        for acc, p in zip(summed, model.parameters()):
            acc.add_(p.grad.detach() * scale)            # accumulate clipped grad

    # Add Gaussian noise calibrated to the clip bound, then average.
    model.zero_grad(set_to_none=True)
    for acc, p in zip(summed, model.parameters()):
        noise = torch.normal(mean=0.0,
                             std=noise_multiplier * clip_norm,
                             size=acc.shape, device=device)
        p.grad = (acc + noise) / B                        # noisy mean gradient

    optimizer.step()
    return loss.item()

# Real usage with Opacus (per-sample grads, Poisson sampling, accountant):
#   from opacus import PrivacyEngine
#   from opacus.utils.batch_memory_manager import BatchMemoryManager
#   privacy_engine = PrivacyEngine(accountant="prv")     # "prv" is tighter than "rdp"
#   model, optimizer, loader = privacy_engine.make_private_with_epsilon(
#       module=model, optimizer=optimizer, data_loader=loader,
#       target_epsilon=8.0, target_delta=1e-7, epochs=3, max_grad_norm=1.0)
#   # ^ solves for the noise multiplier that hits your epsilon, and replaces the
#   #   sampler with a Poisson one so the accounting is actually valid.
#   with BatchMemoryManager(data_loader=loader, max_physical_batch_size=8,
#                           optimizer=optimizer) as safe_loader:
#       for batch in safe_loader:                        # huge LOGICAL batch,
#           ...                                          # small PHYSICAL batch
#   eps = privacy_engine.get_epsilon(delta=1e-7)         # report the spent budget
#
# For HF Transformers + LoRA, microsoft/dp-transformers wires Opacus into the HF
# Trainer. Freeze the base weights first: per-sample gradients are then only ever
# materialized for the (tiny) adapter tensors, which is what makes DP fine-tuning
# of a multi-billion-parameter model fit on a single node.
```

{{fig:dp-sgd-clip-then-noise}}

#### The costs of DP-SGD, honestly

DP-SGD is not free. Three taxes:

- **Utility tax.** Clipping + noise hurt accuracy, and the damage falls hardest on the *tail* — rare classes, minority dialects, long-tail facts — because those examples have large, distinctive gradients that clipping flattens. This is the privacy/fairness tension: DP can disproportionately degrade underrepresented groups.
- **Compute and memory tax.** Per-example gradients break the standard batched backward pass. Naively you store a gradient per example — a $B\times$ blowup in optimizer-state-sized memory. **Ghost clipping** (Li et al., 2022) is the trick that makes this tractable: for linear/embedding layers you can compute each example's gradient *norm* from the layer's saved activations and output gradients alone, never materializing the per-example gradient, then run a second backward with the per-example clipping factors folded into the loss weights. Memory ends up close to non-private training at roughly 2× the time; `private-transformers` was the original implementation and recent Opacus releases expose fast-gradient/ghost clipping as an alternative grad-sample mode. Large *logical* batch sizes are still essential for DP to work well (see below), which is what `BatchMemoryManager`-style gradient accumulation is for.
- **Hyperparameter sensitivity.** DP-SGD wants very large batches, a carefully tuned clip norm $C$, and more epochs than you would expect; it is finicky.

The pragmatic finding that makes DP usable for LLMs: **DP fine-tuning works far better than DP pretraining.** Pretrain non-privately on public/web data, then *fine-tune with DP-SGD on the sensitive dataset*. The public pretraining gives a strong prior so the private phase only needs a small, noisy nudge — DP fine-tuning of large models recovers most of the non-private accuracy at $\varepsilon$ in the single digits, whereas DP *pretraining* from scratch is brutally lossy. Parameter-efficient methods (LoRA, prompt tuning — see [PEFT I: LoRA, QLoRA, DoRA & The Adapter Family](../05-posttraining-alignment/03-peft-lora-qlora.html)) pair especially well with DP because there are fewer parameters to noise.

!!! example "Worked example: reading a privacy budget"
    You DP-fine-tune with batch $B = 4096$, dataset $N = 2{,}000{,}000$ (so sampling rate $q = B/N = 2.048\times10^{-3}$), noise multiplier $\sigma = 0.8$, for $T = 3$ epochs $\approx 1465$ steps, targeting $\delta = 10^{-6}$. Feed $(q, \sigma, T, \delta)$ to a PRV/RDP accountant (e.g. Opacus `get_epsilon`) and it returns, say, $\varepsilon \approx 7.3$.

    Interpretation: an attacker's posterior odds that any given record was a member can shift by at most a factor of $e^{7.3} \approx 1480$. That sounds large, but it caps the *worst case over all adversaries and all records*; empirically the strongest MIA against such a model gets TPR @ 1% FPR only marginally above chance. Want $\varepsilon \approx 1$? Raise $\sigma$ (more noise, lower utility) or cut steps. The dial is explicit: **privacy, utility, compute — pick two, and the accountant tells you the exchange rate.**

### 4.3 Inference-level mitigations

When you cannot retrain (the weights already memorized), you defend at serving time. These are *mitigations*, not guarantees — they raise the cost of an attack without bounding it:

- **Output filtering / memorization detection.** Run generated text against a Bloom filter or n-gram index of known-sensitive strings (or the training corpus itself) and block verbatim emissions. Cheap and effective against *exact* extraction; defeated by paraphrase.
- **MEMFREE / "min-$k$" decoding.** Constrain decoding so the model cannot emit a span that exactly matches a forbidden n-gram, forcing a divergent token whenever a memorized continuation is about to be produced.
- **PII output guardrails.** The same Presidio-style detect-and-redact stack on the *output* side ([Safety, Guardrails & Content Moderation](../12-production-mlops/04-safety-guardrails.html)).
- **Refusal / rate-limiting on extraction-shaped prompts.** Detect the "repeat the word X forever" divergence attack and the long-prefix-completion pattern, and refuse or rate-limit.
- **Sampling temperature.** Greedy decoding maximizes verbatim emission; higher temperature reduces exact extraction but does not prevent a determined attacker who samples many times and ranks.
- **Machine unlearning.** If a specific record must be provably removed post-hoc (a GDPR erasure request, a copyright takedown), targeted unlearning is the tool — covered in depth in [Knowledge Editing & Machine Unlearning](../13-interp-safety-gov/02-knowledge-editing-unlearning.html). Note that unlearning is hard to *certify*; DP during training is the only thing that gives a guarantee without re-running.

```python
# Output-side memorization guard: block verbatim emission of known-sensitive
# n-grams. Use a Bloom filter so the index of "things we must never emit"
# (training secrets, copyrighted spans, prior leaked PII) is memory-cheap.
from pybloom_live import BloomFilter

class MemorizationGuard:
    def __init__(self, n=8, capacity=10_000_000, error_rate=1e-6):
        self.n = n                                  # n-gram length (tokens/words)
        self.bf = BloomFilter(capacity=capacity, error_rate=error_rate)

    def index(self, sensitive_texts):
        for text in sensitive_texts:                # build the forbidden set
            toks = text.split()
            for i in range(len(toks) - self.n + 1):
                self.bf.add(" ".join(toks[i:i + self.n]))

    def violates(self, generated: str) -> bool:
        toks = generated.split()
        for i in range(len(toks) - self.n + 1):
            if " ".join(toks[i:i + self.n]) in self.bf:   # verbatim overlap
                return True                                # block / regenerate
        return False

# guard = MemorizationGuard(n=8)
# guard.index(known_secrets_and_copyrighted_corpus)
# if guard.violates(model_output): regenerate_with_higher_temp_or_refuse()
```

---

## 5. End-to-End: A Membership-Inference + Canary-Extraction Demo

Let us tie it together with a runnable harness you could point at a model you trained yourself. It does two things: (1) a calibrated, reference-model membership-inference attack reported at low FPR, and (2) a canary-extraction sweep that recovers an inserted secret by ranked enumeration. This is the audit you run *before* you ship.

```python
"""
audit_privacy.py — membership inference (reference-calibrated) + canary extraction.

Run against a target model you trained, with a reference model that did NOT see
the same private data (e.g. a base checkpoint, or a smaller public model).
"""
import math
import torch
import torch.nn.functional as F
import numpy as np

@torch.no_grad()
def avg_nll(model, tokenizer, text, device="cuda"):
    """Mean per-token negative log-likelihood (the model's 'loss' on text)."""
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    logits = model(ids).logits[:, :-1]
    lp = F.log_softmax(logits, dim=-1)
    tgt = ids[:, 1:]
    nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return nll.mean().item()

# ----------------------------------------------------------------------------
# (1) MEMBERSHIP INFERENCE, reference-calibrated, scored at low FPR.
# ----------------------------------------------------------------------------
def membership_scores(target, ref, tokenizer, records, device="cuda"):
    """
    Score = ref_loss - target_loss  (a.k.a. the 'Ratio' / LiRA-lite signal).
    A MEMBER should have target_loss << ref_loss  ->  large positive score.
    Calibrating by the reference removes the intrinsic difficulty of the text.
    """
    scores = []
    for text in records:
        s = avg_nll(ref, tokenizer, text, device) - avg_nll(target, tokenizer, text, device)
        scores.append(s)
    return np.array(scores)

def roc_tpr_at_fpr(scores_member, scores_nonmember, fpr_targets=(0.001, 0.01, 0.1)):
    """
    The privacy-correct metric: TPR at fixed (low) FPR. We threshold so that at
    most `fpr` of NON-members are flagged, then measure how many MEMBERS we catch.
    """
    out = {}
    for fpr in fpr_targets:
        # threshold = the (1-fpr) quantile of non-member scores
        thresh = np.quantile(scores_nonmember, 1.0 - fpr)
        tpr = np.mean(scores_member >= thresh)
        out[fpr] = (tpr, thresh)
    # AUC for reference (NOT the headline number — the low-FPR TPR is).
    all_s = np.concatenate([scores_member, scores_nonmember])
    y = np.concatenate([np.ones_like(scores_member), np.zeros_like(scores_nonmember)])
    order = np.argsort(-all_s)
    y = y[order]
    tps = np.cumsum(y); fps = np.cumsum(1 - y)
    tpr_curve = tps / tps[-1]; fpr_curve = fps / fps[-1]
    trapz = getattr(np, "trapezoid", None) or np.trapz   # np.trapz removed in NumPy 2.0
    auc = trapz(tpr_curve, fpr_curve)
    return out, auc

# ----------------------------------------------------------------------------
# (2) CANARY EXTRACTION by ranked enumeration (small body space).
# ----------------------------------------------------------------------------
@torch.no_grad()
def extract_canary(model, tokenizer, template, true_body, n_digits, device="cuda",
                   max_enum=200000):
    """
    Enumerate (a sample of) the body space, rank candidates by model log-prob,
    and report the rank of the TRUE body + exposure. If rank==1 the secret is
    greedily extractable. For tiny spaces (<= max_enum) we enumerate exactly.
    """
    space = 10 ** n_digits
    def body_lp(body):
        ids = tokenizer(template.format(body), return_tensors="pt").input_ids.to(device)
        logits = model(ids).logits[:, :-1]
        lp = F.log_softmax(logits, dim=-1)
        tgt = ids[:, 1:]
        return lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum().item()

    true_lp = body_lp(true_body)
    rng = np.random.default_rng(0)

    if space <= max_enum:                       # exact rank by full enumeration
        better = 0
        for v in range(space):
            body = str(v).zfill(n_digits)
            if body == true_body:
                continue
            if body_lp(body) > true_lp:
                better += 1
        rank = better + 1
    else:                                        # Monte-Carlo rank estimate
        sample = [body_lp("".join(map(str, rng.integers(0, 10, n_digits))))
                  for _ in range(20000)]
        sample = np.array(sample)
        z = (true_lp - sample.mean()) / (sample.std() + 1e-9)
        frac = 0.5 * math.erfc(z / math.sqrt(2))
        rank = max(1, int(frac * space))

    exposure = math.log2(space) - math.log2(rank)
    extractable = (rank == 1)
    return dict(rank=rank, space=space, exposure=exposure,
                extractable=extractable, true_lp=true_lp)

# ----------------------------------------------------------------------------
# Driver (sketch): assumes you have `target`, `ref`, `tokenizer`, and that
# `members` were in target's training set and `nonmembers` were held out.
# ----------------------------------------------------------------------------
def run_audit(target, ref, tokenizer, members, nonmembers,
              canary_template="the access code is {}", canary_body="72935",
              device="cuda"):
    m = membership_scores(target, ref, tokenizer, members, device)
    n = membership_scores(target, ref, tokenizer, nonmembers, device)
    tpr_table, auc = roc_tpr_at_fpr(m, n)

    print("=== Membership Inference (reference-calibrated) ===")
    print(f"AUC (context only): {auc:.3f}")
    for fpr, (tpr, thr) in tpr_table.items():
        ratio = tpr / fpr if fpr > 0 else float('inf')
        print(f"  TPR @ {fpr:6.3%} FPR = {tpr:6.3%}   ({ratio:.1f}x chance)")

    print("\n=== Canary Extraction ===")
    res = extract_canary(target, tokenizer, canary_template, canary_body,
                         n_digits=len(canary_body), device=device)
    print(f"  rank {res['rank']} of {res['space']} | "
          f"exposure {res['exposure']:.1f} bits | "
          f"extractable={res['extractable']}")
    return tpr_table, auc, res
```

What "good" looks like on this harness:

```text
=== Membership Inference (reference-calibrated) ===
AUC (context only): 0.58
  TPR @  0.100% FPR =  0.180%   (1.8x chance)
  TPR @  1.000% FPR =  2.400%   (2.4x chance)
  TPR @ 10.000% FPR = 18.000%   (1.8x chance)

=== Canary Extraction ===
  rank 1 of 100000 | exposure 16.6 bits | extractable=True
```

The MIA numbers above are *illustrative* but show the right shape: a model that is only mildly vulnerable still leaks a few-× over chance in the low-FPR tail — and the canary, inserted with even modest duplication, is fully extractable (rank 1, exposure = $\log_2 10^5 \approx 16.6$ bits). Re-run after deduplication and DP fine-tuning and you want to see the canary's exposure collapse toward ~1 bit and the TPR @ 0.1% FPR fall toward chance.

!!! interview "Interview Corner"
    **Q:** A product team wants to fine-tune an LLM on internal support tickets containing customer PII and ship it to customers. Walk me through how you would reason about privacy, and where differential privacy does and does not help.

    **A:** I'd start with a threat model: the adversary is an external user of the deployed model who can issue arbitrary prompts; the asset is any single customer's PII; the attacks are extraction (verbatim PII in a completion) and membership inference (confirming a person is a customer). Then defenses in layers. (1) **Dedup + PII scrubbing** of the tickets — cheapest, biggest win, but no guarantee and incomplete recall. (2) **DP-SGD fine-tuning** on top of a non-privately-pretrained base, targeting a stated $\varepsilon$ (say single digits) and $\delta \ll 1/N$. DP gives a *formal, worst-case* bound on membership inference: it caps how much any record changes the model, which is exactly the leakage we fear; it's the right tool *for training-set membership*. (3) What DP does **not** help with: PII the model learned during *public* pretraining (DP only protects the private fine-tune set), and the *utility/fairness* hit — DP clipping disproportionately degrades rare ticket types, so I'd measure tail performance, not just aggregate. (4) **Inference guardrails** — output PII redaction and a verbatim-emission Bloom filter — as defense in depth. (5) **Audit**: insert canaries at known duplication counts, run a reference-calibrated MIA, and *report TPR at 0.1% FPR*, not average accuracy. Finally I'd note the compliance framing: under GDPR the fine-tune set is personal data with purpose-limitation and erasure obligations, so I'd want either DP (which makes "is X a member?" provably hard) or a clean retraining path for erasure requests, because unlearning a specific record from a shipped model is hard to certify.

---

## 6. Threat Modeling & Compliance Framing

Privacy engineering decisions are downstream of a *threat model* and a *legal regime*. Both must be explicit.

### 6.1 A threat model template for LLM privacy

State four things before choosing defenses:

| Dimension | Questions to answer |
|---|---|
| **Adversary** | External API user? Insider with logits/weights? Nation-state with the full model? White-box (gradients) vs black-box (text only)? |
| **Asset** | PII of a specific person? Membership (was X a customer)? Copyrighted text? A trade secret in the fine-tune set? |
| **Attack** | Extraction, membership inference, reconstruction, attribute inference, model inversion? |
| **Access** | Black-box top-1 token? Full logits/log-probs? Many queries (extraction needs volume)? Weights (white-box MIA is far stronger)? |

The access level is decisive: a **white-box** adversary with weights and gradients runs LiRA-grade attacks and can read memorization directly, so for an open-weights release DP-during-training is essentially mandatory if the training set is sensitive. A **black-box, rate-limited** API adversary is far weaker, and inference guardrails + dedup may suffice. Never design defenses without naming the adversary.

### 6.2 GDPR, HIPAA, and friends

The compliance layer (covered fully in [AI Governance, Compliance & Regulation](../13-interp-safety-gov/06-governance-compliance.html)) reframes the technical risk as legal obligation:

- **GDPR (EU).** Training data containing personal data is *processing of personal data*. Implicated principles: **lawful basis** for using it, **purpose limitation**, **data minimization** (don't train on PII you don't need — scrub it), and the **right to erasure** (Article 17). Erasure is the sharp one: if a data subject demands deletion, can you remove their influence from a shipped model? Honest options are (a) retrain without them (expensive), (b) certified machine unlearning (hard to prove), or (c) having trained with DP, which weakens the "their data is *in* the model" claim because no single record provably changed it. Many regulators also treat *strong anonymization/DP* as moving data outside GDPR's scope, which is part of DP's appeal.
- **HIPAA (US healthcare).** Protected Health Information (PHI) demands de-identification — the Safe Harbor method enumerates 18 identifiers to strip (names, dates finer than year, record numbers, etc.). A model that *regurgitates* a PHI-bearing training note is a reportable breach. HIPAA pushes you toward aggressive scrubbing *and* output guardrails, and toward keeping the model inside the covered entity's controlled environment.
- **Copyright & the "memorization = infringement" question.** Verbatim emission of copyrighted training text is the technical fact underlying ongoing litigation. Extraction audits and output Bloom-filtering are, in part, *legal* risk-reduction.
- **CCPA/CPRA, the EU AI Act, sectoral rules.** Add transparency, opt-out, and (for high-risk systems) documentation and risk-assessment duties.

The engineering upshot is a single sentence you can take to a design review: **deduplicate and scrub to minimize, train with DP when the fine-tune set is sensitive and especially when weights are released, guard the output at inference, and audit with canaries + low-FPR MIA so you can prove — to yourself and to a regulator — what your model does and does not leak.**

---

!!! key "Key Takeaways"
    - **Memorization is a feature of the objective, not a bug.** Maximum-likelihood training *stores* rare, high-surprise sequences verbatim; it scales up with model size and especially with **data duplication**, and it appears long before classical overfitting.
    - **The attack ladder is MIA → extraction → reconstruction.** Always evaluate membership inference at **TPR @ low FPR** on a log-ROC, never average accuracy — average accuracy hides the worst-case leakage that regulators and plaintiffs care about.
    - **Canary exposure is your measuring stick.** Insert synthetic secrets, compute $\operatorname{exposure} = \log_2|R| - \log_2 \operatorname{rank}$; an exposure near $\log_2|R|$ means the secret is brute-force extractable. Watch it during training as an early-warning signal.
    - **Deduplication is the highest-ROI privacy defense** — order-of-magnitude reduction in extractable memorization at zero (often negative) utility cost. PII scrubbing is necessary but incomplete and gives no formal guarantee.
    - **Differential privacy (DP-SGD) is the only formal guarantee.** Per-example gradient clipping + calibrated Gaussian noise yields an $(\varepsilon,\delta)$ bound that caps *every* membership-inference adversary. Read $\varepsilon$ as a bound on the attacker's odds shift of $e^{\varepsilon}$. Almost all of that privacy comes from **amplification by subsampling**, so the guarantee is only valid if you actually **Poisson-sample** your batches — and it is *example*-level unless you clip per *user*.
    - **DP fine-tuning ≫ DP pretraining.** Pretrain non-privately, then DP-fine-tune the sensitive set (ideally with PEFT) to recover most utility at single-digit $\varepsilon$; Opacus (+ `dp-transformers` for the HF `Trainer`) and ghost clipping make this practical. DP's costs are real: utility hit on the tail, higher compute/memory, and finicky hyperparameters.
    - **Inference-time mitigations are mitigations, not guarantees** — output PII redaction, verbatim-emission Bloom filters, MemFree-style decoding, and refusal of extraction-shaped prompts raise attack cost but don't bound leakage.
    - **Defenses follow the threat model and the law.** White-box / open-weights releases of sensitive-data models essentially require DP-during-training; GDPR's right-to-erasure and HIPAA's de-identification push you toward minimization, DP, and provable audits.

!!! sota "State of the Art & Resources (2026)"
    Privacy and memorization in LLMs is an active research frontier: the field has moved from demonstrating that extraction is possible to rigorous empirical auditing frameworks, efficient DP fine-tuning at scale, and user-level privacy guarantees for production systems. Deduplication + DP fine-tuning with PEFT is the current best-practice recipe for deploying sensitive-data models.

    **Foundational work**

    - [Carlini et al., *The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks* (2019)](https://arxiv.org/abs/1802.08232) — introduces canaries and the exposure metric; the foundation of quantitative privacy auditing for generative models.
    - [Abadi et al., *Deep Learning with Differential Privacy* (2016)](https://arxiv.org/abs/1607.00133) — the DP-SGD algorithm (per-example gradient clipping + Gaussian noise + moments accountant) that remains the gold standard for formal privacy in deep learning.
    - [Dwork & Roth, *The Algorithmic Foundations of Differential Privacy* (2014)](https://www.nowpublishers.com/article/Details/TCS-042) — the canonical textbook for the (ε, δ) definition, composition theorems, and core mechanisms.

    **Recent advances (2023–2026)**

    - [Carlini et al., *Extracting Training Data from Large Language Models* (2021)](https://arxiv.org/abs/2012.07805) — the canonical GPT-2 extraction attack; establishes the generate-rank-verify pipeline and zlib/reference-model membership signals.
    - [Carlini et al., *Quantifying Memorization Across Neural Language Models* (2022)](https://arxiv.org/abs/2202.07646) — establishes the log-linear scaling of memorization with model size, duplication, and context length.
    - [Carlini et al., *Membership Inference Attacks From First Principles* (LiRA, 2022)](https://arxiv.org/abs/2112.03570) — shadow-model likelihood-ratio attack and the case for TPR @ low FPR as the correct privacy metric.
    - [Nasr et al., *Scalable Extraction of Training Data from (Production) Language Models* (2023)](https://arxiv.org/abs/2311.17035) — the "divergence attack" showing aligned models still leak pretraining data; extraction at scale from ChatGPT and other production models.
    - [Nasr et al., *Tight Auditing of Differentially Private Machine Learning* (2023)](https://arxiv.org/abs/2302.07956) — empirical auditing that nearly matches provable DP guarantees using only two training runs; current state of the art in DP auditing.
    - [Ganesh & Charles, *Fine-tuning LLMs with user-level differential privacy* (Google Research, 2025)](https://research.google/blog/fine-tuning-llms-with-user-level-differential-privacy/) — extends DP fine-tuning to protect entire user histories rather than individual examples; reflects production deployment at Google.

    **Open-source & tools**

    - [pytorch/opacus](https://github.com/pytorch/opacus) — Meta's production-grade DP-SGD library for PyTorch; handles per-sample gradients (hooks or `torch.func`), Poisson sampling, RDP/PRV accounting, `BatchMemoryManager` gradient accumulation, and LoRA-compatible fine-tuning with minimal code changes.
    - [microsoft/dp-transformers](https://github.com/microsoft/dp-transformers) — thin integration of Opacus with the Hugging Face `Trainer`, including PEFT/LoRA recipes; the shortest path from a non-private fine-tune script to a DP one.
    - [google/differential-privacy](https://github.com/google/differential-privacy) — Google's DP library, including the `dp_accounting` package (RDP and PRV accountants) used to price subsampled-Gaussian training.
    - [microsoft/presidio](https://github.com/microsoft/presidio) — open-source PII detection and anonymization framework (regex + checksum validators + NER + context words, with thresholdable confidence scores) for scrubbing training data and filtering model outputs.
    - [iamgroot42/mimir](https://github.com/iamgroot42/mimir) — reference implementations of the standard LLM membership-inference attacks (LOSS, ratio, zlib, min-k% prob, neighbourhood) with controlled member/non-member splits; the right baseline to beat before believing your own MIA.

    **Go deeper**

    - [Lee et al., *Deduplicating Training Data Makes Language Models Better* (ACL 2022)](https://arxiv.org/abs/2107.06499) — quantifies the order-of-magnitude memorization reduction from deduplication, with released suffix-array tooling for large corpora.
    - [Li et al., *Large Language Models Can Be Strong Differentially Private Learners* (ICLR 2022)](https://arxiv.org/abs/2110.05679) — ghost clipping: per-example gradient *norms* without materializing per-example gradients, which is what makes DP fine-tuning of large transformers memory-feasible.

## Further reading

- **Carlini, Liu, Erlingsson, Kos, Song, "The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks," USENIX Security 2019** — introduces canaries and the exposure metric; the foundation of privacy auditing.
- **Carlini et al., "Extracting Training Data from Large Language Models," USENIX Security 2021** — the canonical GPT-2 extraction attack (generate, rank by membership signal, verify).
- **Carlini, Ippolito, Jagielski, Lee, Tramèr, Zhang, "Quantifying Memorization Across Neural Language Models," 2022** — establishes the scaling of memorization with model size, duplication, and context length.
- **Abadi, Chu, Goodfellow, McMahan, Mironov, Talwar, Zhang, "Deep Learning with Differential Privacy," ACM CCS 2016** — the DP-SGD algorithm (per-example clipping + Gaussian noise + moments accountant).
- **Dwork & Roth, "The Algorithmic Foundations of Differential Privacy," 2014** — the standard reference for the $(\varepsilon,\delta)$ definition, composition, and mechanisms.
- **Carlini, Chien, Nasr, Song, Terzis, Tramèr, "Membership Inference Attacks From First Principles" (LiRA), IEEE S&P 2022** — the shadow-model likelihood-ratio attack and the case for low-FPR evaluation.
- **Lee, Ippolito et al., "Deduplicating Training Data Makes Language Models Better," ACL 2022** — quantifies how deduplication reduces memorization (and improves quality).
- **Yu, Naik, Backurs, Gopi, Inan, Kamath, Kulkarni, Lee, Manoel, Wutschitz, Zanella-Béguelin et al., "Differentially Private Fine-tuning of Language Models," ICLR 2022** — shows DP fine-tuning of large LMs recovers most utility at single-digit $\varepsilon$.
- **Nasr et al., "Scalable Extraction of Training Data from (Production) Language Models," 2023** — the divergence ("repeat this word") attack on aligned models.
- **Kandpal, Wallace, Raffel, "Deduplicating Training Data Mitigates Privacy Risks in Language Models," ICML 2022** — the super-linear relationship between duplication count and regeneration rate quoted in §1.2.
- **Li, Tramèr, Liang, Hashimoto, "Large Language Models Can Be Strong Differentially Private Learners," ICLR 2022** — ghost clipping and the DP fine-tuning hyperparameter recipe (huge batches, large clipping-normalized updates).
- **Duan et al., "Do Membership Inference Attacks Work on Large Language Models?" 2024 (the MIMIR benchmark)** — most published LLM MIAs are near chance once member/non-member distribution shift is controlled.
- **microsoft/presidio**, **pytorch/opacus**, **microsoft/dp-transformers**, **google/differential-privacy (`dp_accounting`)** — production-grade open-source libraries for PII detection/anonymization, DP-SGD training, DP fine-tuning with Hugging Face, and privacy accounting.

---

## Exercises

**1.** *(Conceptual.)* A colleague reports that their fine-tuned model has a healthy, still-*decreasing* validation loss and no gap between train and validation loss, and concludes "there's no overfitting, so there's no privacy problem — and if there were, I'd just add weight decay and early stopping." Explain, using the chapter's definitions, why this reasoning is wrong on two counts: (a) why memorization can be severe even when validation loss looks perfectly healthy, and (b) why the classic regularizers they name are the wrong tools.

??? note "Solution"
    **(a) Memorization is decoupled from overfitting.** Validation loss is an *average* over the bulk of the data distribution, whereas verbatim memorization concerns the *tail* of rare, high-surprise examples (a unique credit-card number, a UUID, a one-off secret). A model can drive the loss of thousands of unique sequences to near zero — storing them in its weights — while the *average* validation loss keeps falling smoothly, because those memorized tail examples are a vanishing fraction of the average. So "no train/val gap" tells you nothing about the tail. As the chapter puts it, the two phenomena "are decoupled because memorization concerns the tail of rare examples while validation loss is an average over the bulk."

    Moreover, memorization is not pathological overfitting at all: for a unique, low-entropy, high-surprise sequence there is *no generalizable rule* that predicts the next token, so the only way maximum likelihood can lower that sequence's loss is to store it verbatim. Memorization is therefore the *optimal* behavior of the cross-entropy objective on rare sequences, and it appears *well before* the model overfits in the classical rising-validation-loss sense.

    **(b) Weight decay, dropout, and early stopping target the wrong statistic.** These regularizers are designed to close the train/val *average* gap — exactly the statistic that is already healthy here. They do not specifically suppress the storage of individual rare sequences; early stopping in particular is useless because memorization happens early, before any validation-loss turnaround it could trigger on. The right tools operate on the tail and on individual-record influence: **deduplication** (collapses the highest-duplication, most-extractable tail), **PII scrubbing** (removes the secrets pre-training), and **DP-SGD** (formally bounds how much any single record changes the model). Only DP gives a guarantee; the standard regularizers give none.

**2.** *(Quantitative.)* You insert a canary of the form `"the vault code is XXXXXX"` whose body is 6 random digits, so the body space is $|R| = 10^6$. After training you rank the true body among all $10^6$ candidates by model log-likelihood (rank 1 = lowest perplexity).

   (a) What is $\log_2|R|$, and what exposure would a *non-memorized* canary have?
   (b) You measure the true body at rank $256$. Compute its exposure in bits.
   (c) How many brute-force guesses does an attacker who enumerates from most-to-least likely need to recover the secret, and is this alarming?
   (d) You now re-run the audit after deduplicating the corpus and measure rank $\approx 4.9\times10^5$. What is the new exposure, and what does this demonstrate?

??? note "Solution"
    **(a)** $\log_2 10^6 = 6\log_2 10 \approx 6 \times 3.3219 = 19.93$ bits. A non-memorized canary has a roughly *uniform* rank, so $\mathbb{E}[\operatorname{rank}] \approx |R|/2 = 5\times10^5$, giving
    $$\operatorname{exposure} = 19.93 - \log_2(5\times10^5) = 19.93 - 18.93 = 1.0 \text{ bit}.$$
    Baseline exposure is about **1 bit** — the "safe" reading.

    **(b)** With rank $= 256 = 2^8$:
    $$\operatorname{exposure} = \log_2|R| - \log_2\operatorname{rank} = 19.93 - \log_2 256 = 19.93 - 8 = 11.93 \text{ bits}.$$

    **(c)** An attacker ranking candidates by the model's own likelihood finds the true body at position 256, so at most **256 guesses** recover it. That is trivially cheap — well within a scripted brute force — so yes, this is **alarming**: an exposure of ~12 bits on a low-duplication canary means the pipeline is memorizing far too aggressively.

    **(d)** $\operatorname{exposure} = 19.93 - \log_2(4.9\times10^5) = 19.93 - 18.90 = 1.03$ bits — essentially back to the ~1-bit baseline. The secret's rank is now near $|R|/2$, i.e. indistinguishable from a body the model never saw. This demonstrates the chapter's central cheap-defense claim: **deduplication collapses exposure toward baseline at essentially zero utility cost**, turning an extractable secret back into noise.

**3.** *(Quantitative.)* In DP-SGD the noisy averaged gradient is
   $$\hat{g} = \frac{1}{B}\Big(\sum_{i=1}^{B}\tilde{g}_i + \mathcal{N}(0,\sigma^2 C^2 \mathbf{I})\Big),$$
   with clip norm $C = 1.0$ and noise multiplier $\sigma = 1.1$. Consider one coordinate.

   (a) What is the standard deviation of the added noise *per coordinate* on the **summed** (pre-average) gradient, and does it depend on the batch size $B$?
   (b) After dividing by $B$, what is the noise std per coordinate on $\hat{g}$ for $B = 256$ versus $B = 4096$?
   (c) Assume the clipped per-example gradients are roughly aligned so the *signal* per coordinate of the averaged gradient is about constant at $g \approx 0.02$ regardless of $B$. Compute the signal-to-noise ratio (SNR) for each batch size. Explain in one sentence why the chapter says "large *logical* batch sizes are essential for DP to work well."

??? note "Solution"
    **(a)** The Gaussian noise added to the *sum* is $\mathcal{N}(0, \sigma^2 C^2 \mathbf{I})$, so the per-coordinate std on the summed gradient is
    $$\sigma C = 1.1 \times 1.0 = 1.1.$$
    This is **independent of $B$** — the same amount of noise is added to the sum whether the batch has 256 or 4096 examples.

    **(b)** Averaging divides by $B$, so the per-coordinate noise std on $\hat{g}$ is $\sigma C / B$:
    - $B = 256$: $1.1 / 256 \approx 4.30\times10^{-3}$.
    - $B = 4096$: $1.1 / 4096 \approx 2.69\times10^{-4}$.

    **(c)** $\text{SNR} = g \,/\, (\sigma C / B) = g\,B/(\sigma C)$:
    - $B = 256$: $0.02 \times 256 / 1.1 \approx 4.65$.
    - $B = 4096$: $0.02 \times 4096 / 1.1 \approx 74.5$.

    The SNR grows *linearly* in $B$ (a $16\times$ larger batch gives a $16\times$ better SNR here) because the injected noise is fixed on the sum while the averaged useful signal is constant — so the only way to drown out the DP noise and get a usable gradient direction is to average over a very large batch. That is why DP-SGD "wants very large batches," and it compounds the compute/memory cost of per-example gradients.

    The catch, at fixed $\varepsilon$: raising $B$ also raises the sampling rate $q = B/N$, and the accountant charges roughly $O(q^2/\sigma^2)$ per step, so a bigger batch buys SNR only if you also raise $\sigma$ (or take fewer steps). The net is still favourable — the noise-to-signal ratio $\sigma C/(gB)$ improves faster than the accounting cost — which is why published DP fine-tuning recipes use batches in the thousands with noise multipliers around 1, and why you should always let the accountant, not intuition, price the trade.

**4.** *(Quantitative / conceptual.)* Your DP accountant reports the fine-tune achieved $(\varepsilon, \delta) = (8.0, 10^{-6})$ on a dataset of $N = 2\times10^6$ records.

   (a) By what multiplicative factor can an adversary's posterior odds that a given record was a member shift, in the worst case? Compare to a target of $\varepsilon = 1.0$.
   (b) Is the chosen $\delta$ appropriate for this $N$? State the rule of thumb and check it.
   (c) One reviewer says "$e^{8} \approx 2981$ is a huge odds shift, so this model is basically not private." Give the two-part rebuttal the chapter supports.

??? note "Solution"
    **(a)** The $(\varepsilon,\delta)$ guarantee bounds the odds shift by $e^\varepsilon$:
    - $\varepsilon = 8.0$: $e^{8.0} \approx 2981$.
    - $\varepsilon = 1.0$: $e^{1.0} \approx 2.72$.

    So at $\varepsilon = 8$ an attacker's membership odds can move by up to ~$2981\times$ in the worst case, versus only ~$2.7\times$ at $\varepsilon = 1$. The chapter's rules of thumb: $\varepsilon \le 1$ is strong, $\varepsilon \approx 8$ is "meaningful but loose," $\varepsilon \gg 50$ is mostly cosmetic.

    **(b)** The rule of thumb is $\delta \ll 1/N$. Here $1/N = 1/(2\times10^6) = 5\times10^{-7}$. The chosen $\delta = 10^{-6}$ is *larger* than $1/N$, so it is **not** comfortably below the threshold — you would want $\delta$ at least an order of magnitude under $5\times10^{-7}$ (say $10^{-7}$ or smaller). A $\delta$ above $1/N$ is dangerous because the $+\delta$ slack can, in principle, permit leaking a whole record.

    **(c) Rebuttal, two parts.** (i) $e^\varepsilon$ is a **worst-case bound over all adversaries, all side information, and all records simultaneously** — it is a ceiling, not the leakage of a typical attack. Empirically, the strongest membership-inference attack against a model trained to single-digit $\varepsilon$ gets TPR @ 1% FPR only marginally above chance; the realized privacy is far better than the loose bound suggests (this is exactly what tight DP auditing shows). (ii) The bound still *guarantees* that no attacker can ever do better than that ceiling — unlike scrubbing or inference guardrails, which give no guarantee at all. So $\varepsilon = 8$ is a meaningful-but-loose guarantee, not "basically not private." If a tighter guarantee is required, raise $\sigma$ (more noise) or cut steps to push toward $\varepsilon \approx 1$, at a utility cost the accountant makes explicit.

**5.** *(Implementation.)* The reference-calibrated MIA in `audit_privacy.py` scores membership as `ref_loss - target_loss`. The chapter lists an alternative membership signal, the **zlib ratio**, which divides the model's log-likelihood by the string's `zlib` compression entropy so that intrinsically compressible strings (which are trivially low-perplexity) are penalized. Implement a `zlib_membership_scores` function, in the harness's style, that scores each record by $\dfrac{\log p_\theta(s)}{\text{zlib\_entropy}(s)}$ (a *larger* score should indicate a likely member), and can be dropped into `roc_tpr_at_fpr` in place of the reference signal. Note in one line why this signal is attractive when you have *no* reference model.

??? note "Solution"
    We need the model's *total* log-probability of the string (in bits, to match zlib's byte/bit units the ratio is scale-free anyway), and the zlib entropy = size in bytes of the compressed string. A member should have unusually high model likelihood *relative to how compressible the text intrinsically is*, so a larger ratio flags membership. Because $\log p_\theta(s) < 0$ and $\text{zlib\_entropy}(s) > 0$, the ratio is negative and *less negative* (larger) for memorized strings.

    ```python
    import zlib
    import numpy as np
    import torch
    import torch.nn.functional as F

    @torch.no_grad()
    def total_logprob(model, tokenizer, text, device="cuda"):
        """Sum of per-token log p_theta(text) in NATS (negative number)."""
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        logits = model(ids).logits[:, :-1]
        lp = F.log_softmax(logits, dim=-1)
        tgt = ids[:, 1:]
        return lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum().item()

    def zlib_entropy(text: str) -> float:
        """Compressed size in BYTES; a proxy for the string's intrinsic entropy.
        Larger for high-entropy (random-looking) strings, small for repetitive ones."""
        return len(zlib.compress(text.encode("utf-8")))

    def zlib_membership_scores(target, tokenizer, records, device="cuda"):
        """
        Score = log p_theta(s) / zlib_entropy(s).   No reference model needed.
        log p_theta(s) is negative; dividing by the (positive) compressed size
        normalizes away 'this text is just cheap to model because it's compressible'.
        A MEMORIZED string is much more likely than its compressibility predicts,
        so its (still-negative) score is comparatively LARGE -> flagged member.
        """
        scores = []
        for text in records:
            lp = total_logprob(target, tokenizer, text, device)   # nats, < 0
            z = zlib_entropy(text) + 1e-9                          # bytes, > 0
            scores.append(lp / z)
        return np.array(scores)

    # Drop-in: swap the reference signal for the zlib signal, same low-FPR metric.
    #   m = zlib_membership_scores(target, tokenizer, members)
    #   n = zlib_membership_scores(target, tokenizer, nonmembers)
    #   tpr_table, auc = roc_tpr_at_fpr(m, n)   # report TPR @ 0.1%, 1% FPR
    ```

    **Why it is attractive with no reference model:** the reference-model signal `ref_loss - target_loss` needs a second trained model that did *not* see the private data; the zlib ratio replaces that reference with a *free, model-independent* baseline (a general-purpose compressor), so you can calibrate away a string's intrinsic difficulty without training or hosting any shadow/reference model. As with every MIA in this chapter, evaluate it at TPR @ low FPR on a log-ROC, not by average accuracy.
