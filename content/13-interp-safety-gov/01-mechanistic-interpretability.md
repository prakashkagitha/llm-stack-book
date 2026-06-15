# 13.1 Mechanistic Interpretability & Model Internals

A trained transformer is a 70-billion-parameter function that maps token sequences to next-token distributions, and for most of its history we have treated it as a black box: feed it text, read off logits, judge it by its outputs. That stance is comfortable until the model lies, leaks a secret, develops a backdoor, or refuses for reasons you cannot articulate. At that point "the eval went up" stops being an explanation and you want to ask a sharper question: *what computation, inside the network, produced this behavior?* **Mechanistic interpretability** (often "mech interp") is the program of answering that question by reverse-engineering the learned algorithms a network implements — not statistical summaries of inputs and outputs, but the actual circuits, features, and intermediate representations that carry information from layer to layer.

This chapter is a working engineer's tour of that toolkit. We build the central mental model first — the **residual stream** as a shared communication channel, and **superposition** as the reason features are tangled — and then develop the practical instruments in roughly increasing order of causal force: **probing** (does the information exist?), the **logit lens** (what does the model "believe" mid-stack?), **activation patching** and **causal tracing** (which components *cause* a behavior?), **circuit discovery** (how do components compose into an algorithm?), and **sparse autoencoders** (how do we pull monosemantic features out of superposition?). We close with the applied payoff — debugging, **activation steering**, feature-based monitoring for safety — the tooling ecosystem (TransformerLens, SAE frameworks), and an honest accounting of what this field cannot yet do. Two from-scratch code experiments anchor the theory: a logit lens you can run on any HuggingFace model, and a tiny activation-patching study that localizes a factual association to a specific layer and token.

This material sits downstream of the architecture you already know. If the residual stream, attention heads, or the MLP block are hazy, revisit [The Transformer Block: Norms, Residuals, MLPs & Activations](../02-transformer/06-transformer-block.html) and [The Attention Mechanism From Scratch](../02-transformer/03-attention-from-scratch.html). Interpretability is also a precondition for the editing and safety techniques in [Knowledge Editing & Machine Unlearning](../13-interp-safety-gov/02-knowledge-editing-unlearning.html) and [AI Safety: Scalable Oversight, Dangerous-Capability Evals & Frontier Safety](../13-interp-safety-gov/05-ai-safety-oversight.html).

---

## 1. The Residual Stream and Superposition: The Core Mental Model

### 1.1 The residual stream as a shared bus

Take a decoder-only transformer with model dimension $d_\text{model}$ and $L$ layers. Strip away everything but the data flow and you find a strikingly simple skeleton. After embedding, every token position carries a vector $x \in \mathbb{R}^{d_\text{model}}$. Each attention sub-layer and each MLP sub-layer *reads* from this vector, computes something, and *adds* its output back:

$$
x_{\ell+1} = x_\ell + \operatorname{Attn}_\ell\!\big(\operatorname{LN}(x_\ell)\big) + \operatorname{MLP}_\ell\!\big(\operatorname{LN}(x_\ell)\big).
$$

Because every contribution is *added*, the value at layer $\ell$ is exactly the sum of the embedding plus every sub-layer output so far:

$$
x_\ell = x_\text{embed} + \sum_{k < \ell} \big(\operatorname{Attn}_k + \operatorname{MLP}_k\big).
$$

This running sum is the **residual stream**. The Anthropic "Mathematical Framework for Transformer Circuits" (Elhage et al., 2021) made this view canonical, and it reorganizes how you think about the network. The residual stream is not a transient activation — it is a *persistent communication channel*, a bus that all layers read from and write to. A head in layer 2 can write a vector that a head in layer 9 reads, by *aligning their weight matrices along a shared subspace* of $\mathbb{R}^{d_\text{model}}$. The intermediate layers simply leave that subspace untouched. We call such an arrangement a **virtual weight** connection: the effective linear map from the writer to the reader, even though no single weight matrix connects them.


{{fig:mechinterp-residual-stream-bus}}


Two structural facts fall out immediately and we will use both. First, **attention moves information *between* positions; the MLP processes information *within* a position.** Attention is the only sub-layer that mixes across the sequence dimension. Second, the linearity of the residual stream makes contributions **decomposable**: you can take any downstream quantity — a logit, an attention pattern, a neuron's preactivation — and write it as a sum of additive contributions from each upstream component. This is the foundation of *direct logit attribution* and of the logit lens in §3.

### 1.2 Features, directions, and the linear representation hypothesis

The working hypothesis of the whole field is the **linear representation hypothesis**: high-level, human-meaningful *features* are represented as approximately linear directions in activation space. "This token is inside a quotation," "the subject is a US state," "the sentiment is negative," "we are writing Python" — each is hypothesized to correspond to a direction $v \in \mathbb{R}^{d_\text{model}}$, such that the feature is active to the degree that the activation $x$ has a large component along $v$, i.e. the dot product $\langle x, v\rangle$ is large. Word-embedding analogies (`king - man + woman ≈ queen`) were the first hint; modern probing and SAE results are strong evidence that the hypothesis holds widely, if imperfectly.

If features were *orthogonal* directions, life would be easy: $d_\text{model}$ dimensions could hold $d_\text{model}$ features, and we could read each off with a clean projection. They are not, and they cannot be, because of **superposition**.

### 1.3 Superposition: why neurons are polysemantic

A transformer "wants" to represent far more features than it has dimensions. A language model plausibly tracks tens or hundreds of thousands of distinct features — specific entities, syntactic roles, topics, code constructs — but $d_\text{model}$ is only a few thousand. The resolution, demonstrated cleanly in Anthropic's "Toy Models of Superposition" (Elhage et al., 2022), is **superposition**: the network packs $n \gg d_\text{model}$ features into $d_\text{model}$ dimensions by representing them as *non-orthogonal* directions, exploiting the fact that real features are **sparse** — only a handful are active on any given token.

The geometry is the Johnson–Lindenstrauss insight: in high dimensions you can fit exponentially many *almost*-orthogonal unit vectors. With pairwise interference bounded by $\langle v_i, v_j\rangle \le \epsilon$, the number of features you can pack grows roughly like $\exp(\epsilon^2 d_\text{model})$. The price is **interference**: each feature reads a little noise from every other co-active feature. Sparsity keeps that noise manageable — if only $k$ of $n$ features fire at once, the expected squared interference scales with $k$, not $n$.

The direct, painful consequence for interpretability is **polysemanticity**: an individual neuron (one coordinate of the MLP hidden activation) responds to a *mix* of unrelated features — it might fire for academic citations, for HTTP headers, *and* for Korean text. The neuron is not the unit of computation; the *feature direction*, which is spread across many neurons, is. This is precisely why we cannot interpret a network by reading neurons one at a time, and precisely the problem sparse autoencoders (§6) are built to solve — they attempt to **decompose superposition** back into a larger set of sparse, monosemantic features.

!!! note "Aside: privileged vs. non-privileged bases"
    The residual stream has a *non-privileged* basis — nothing makes its coordinate axes special, because every operation that reads it (`LN`, then a linear map) is free to rotate. So residual-stream features can point in any direction. MLP *hidden* activations have a *privileged* basis because the element-wise nonlinearity (GELU, etc.) acts coordinate-by-coordinate, giving neurons a slight tendency to align with features — but superposition still wins, so they remain polysemantic. This is why SAEs are usually trained on residual-stream or MLP activations rather than assuming neurons are already interpretable.

---

## 2. Probing: Does the Information Exist?

The lightest-weight question you can ask is existential: **is feature $X$ linearly decodable from the activations at layer $\ell$?** A **linear probe** answers it. Freeze the model, collect activations $h^{(\ell)} \in \mathbb{R}^{d_\text{model}}$ at some layer for a labeled dataset, and train a small classifier — usually just logistic regression — to predict the label from $h^{(\ell)}$. High accuracy means the information is present and linearly accessible at that layer.

$$
\hat{y} = \sigma\!\big(w^\top h^{(\ell)} + b\big), \qquad
\mathcal{L} = -\sum_i \big[ y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\big].
$$

Probing is cheap, scalable, and produces a clean layer-by-layer picture: train one probe per layer and you can watch *where* in the stack a concept becomes linearly available. Classic results show part-of-speech and syntax peaking in early-middle layers of BERT-style models, with more abstract/semantic features peaking later.

The crucial caveat — drilled into students by the "control task" / "selectivity" work of Hewitt & Liang (2019) — is that **probing measures correlation, not use**. A probe with 95% accuracy tells you the information is *there*; it does **not** tell you the model *uses* it for any downstream computation. A sufficiently expressive probe can even fit structure the model never represents (it memorizes from the probe's own capacity). Guardrails:

- **Use the weakest probe that works** (linear, low-capacity). If a linear probe succeeds, the feature is linearly available — a strong, architecture-relevant claim. If only a 3-layer MLP probe succeeds, you have learned more about the probe than the model.
- **Run a control task.** Hewitt–Liang assign random-but-consistent labels and measure how well the probe fits *those*; high control accuracy means the probe is too powerful. Report **selectivity** = (real accuracy − control accuracy).
- **For causal claims, you must intervene** — which is exactly the jump from probing to activation patching (§4). Probing is reconnaissance; patching is proof.

```python
import torch
from sklearn.linear_model import LogisticRegression

# h_train: (N, d_model) activations cached at a chosen layer/position; y_train: (N,) labels
# We deliberately use a LINEAR probe with strong L2 regularization — low capacity by design.
probe = LogisticRegression(C=0.5, max_iter=2000)
probe.fit(h_train.numpy(), y_train.numpy())
real_acc = probe.score(h_val.numpy(), y_val.numpy())

# Control task (Hewitt & Liang): assign each *input type* a fixed random label, refit, re-score.
import numpy as np
rng = np.random.default_rng(0)
ctrl_labels = rng.integers(0, 2, size=y_train.shape)          # random but held fixed per example
ctrl = LogisticRegression(C=0.5, max_iter=2000).fit(h_train.numpy(), ctrl_labels)
ctrl_acc = ctrl.score(h_val.numpy(), rng.integers(0, 2, size=y_val.shape))

print(f"probe acc={real_acc:.3f}  control acc={ctrl_acc:.3f}  selectivity={real_acc-ctrl_acc:.3f}")
# A trustworthy linear probe has HIGH real accuracy and LOW control accuracy → high selectivity.
```

---

## 3. The Logit Lens: Reading the Residual Stream Directly

Because the residual stream is the running sum that gets unembedded into logits, we can ask a beautiful question: *what would the model predict if we stopped here?* The **logit lens** (introduced by nostalgebraist, 2020) does exactly this. Take the residual-stream vector at an intermediate layer $\ell$, apply the model's final layer-norm and unembedding matrix $W_U$ to it — skipping the remaining layers — and read off the resulting "early" next-token distribution:

$$
\text{logits}^{(\ell)} = W_U \cdot \operatorname{LN}_f\!\big(x_\ell\big), \qquad
p^{(\ell)} = \operatorname{softmax}\big(\text{logits}^{(\ell)}\big).
$$

What you typically see is the prediction **sharpening across depth**: early layers produce diffuse, often nonsensical distributions; somewhere in the middle the correct token starts climbing the ranking; the final layers commit. It is a moving picture of the model "making up its mind." The logit lens is the simplest causal-flavored tool because it uses the model's *own* unembedding — no auxiliary training — so the directions it reads are exactly the ones the model uses.

It has known failure modes. Some models (and some layers) have residual-stream activations that the unembedding does not read well until later normalization adjusts them, so a raw logit lens looks garbled. **Tuned Lens** (Belrose et al., 2023) fixes this by training a tiny per-layer affine "translator" $A_\ell x_\ell + b_\ell$ that maps each layer's residual into the space the unembed expects, giving far smoother, more faithful trajectories. Think of the logit lens as the zero-training baseline and the tuned lens as its calibrated upgrade.

### 3.1 From-scratch logit lens

Here is a complete, runnable logit lens using plain PyTorch hooks on a HuggingFace GPT-2 — no special libraries. It caches each block's residual-stream output, unembeds each one, and prints the top token predicted at every layer.

```python
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()

# ── 1. Hook every transformer block to capture its residual-stream OUTPUT. ──────────
# In GPT2, model.transformer.h[i] outputs (hidden_states, ...); hidden_states is the
# residual stream AFTER block i, i.e. x_{i+1}. We stash one per layer.
resid_by_layer = {}
def make_hook(idx):
    def hook(_module, _inp, out):
        # out is a tuple; out[0] = residual stream of shape (batch, seq, d_model)
        resid_by_layer[idx] = out[0].detach()
    return hook

handles = [blk.register_forward_hook(make_hook(i))
           for i, blk in enumerate(model.transformer.h)]

# ── 2. Run the model once. ─────────────────────────────────────────────────────────
prompt = "The Eiffel Tower is located in the city of"
ids = tok(prompt, return_tensors="pt").input_ids.to(device)
with torch.no_grad():
    _ = model(ids)
for h in handles:
    h.remove()

# ── 3. Apply final LN + unembed to EACH layer's residual at the last position. ──────
ln_f = model.transformer.ln_f          # final layer norm
W_U  = model.lm_head                    # unembedding (tied to embeddings in GPT2)

print(f"prompt: {prompt!r}\n")
print(f"{'layer':>5} | {'top-1 token':<14} | prob   | rank of ' Paris'")
print("-" * 50)
paris_id = tok(" Paris").input_ids[0]
for layer in range(model.config.n_layer):
    x = resid_by_layer[layer][0, -1]            # (d_model,) last-token residual
    logits = W_U(ln_f(x))                        # logit lens: skip remaining layers
    probs = torch.softmax(logits, dim=-1)
    top_id = int(probs.argmax())
    top_tok = tok.decode([top_id])
    # rank of the gold answer token among all vocab logits (0 = most likely)
    rank = int((logits > logits[paris_id]).sum())
    print(f"{layer:>5} | {top_tok!r:<14} | {probs[top_id]:.3f} | {rank}")
```

```text
prompt: 'The Eiffel Tower is located in the city of'

layer | top-1 token    | prob   | rank of ' Paris'
--------------------------------------------------
    0 | ' the'         | 0.04   | 8122
    3 | ' a'           | 0.05   | 1190
    6 | ' London'      | 0.07   | 14
    8 | ' Paris'       | 0.10   | 0
   10 | ' Paris'       | 0.31   | 0
   11 | ' Paris'       | 0.58   | 0
```

The exact numbers depend on the GPT-2 checkpoint, but the *shape* is the lesson and it reproduces reliably: the gold token ` Paris` is buried near rank 8000 at the embedding, climbs as middle layers retrieve the association, and by the last layers dominates the distribution. You have just watched a fact get *recalled* layer by layer — and we will now localize exactly *where* that recall happens.

---

## 4. Activation Patching & Causal Tracing: From Correlation to Cause

Probing and the logit lens are observational. To make a *causal* claim — "this component is responsible for this behavior" — you must intervene. **Activation patching** (also called *interchange intervention*, and *causal tracing* in the ROME paper, Meng et al., 2022) is the workhorse, and it is conceptually a clean controlled experiment.

### 4.1 The recipe

You need two inputs that differ in the behavior of interest:

- A **clean** run that produces the correct/interesting behavior (e.g. prompt → ` Paris`).
- A **corrupted** run that does not (e.g. the same prompt with the subject tokens replaced by noise, or swapped to a different entity → ` Rome`).

Run the model on the corrupted input, but **patch in** one activation (a specific component at a specific layer and token position) from the *clean* run, then let the forward pass finish. Measure how much the output recovers toward the clean answer. If patching a single component restores most of the correct behavior, that component **carries the information** that distinguishes the two runs.

We quantify recovery with a metric — typically the logit difference between the two candidate answers, or the log-prob of the gold token:

$$
\text{recovery}(\text{component } c) = \frac{m_\text{patched}(c) - m_\text{corrupt}}{m_\text{clean} - m_\text{corrupt}} \in [0, 1],
$$

where $m$ is, e.g., $\log p(\text{Paris}) - \log p(\text{Rome})$. A recovery of $1.0$ means patching $c$ fully restores the clean behavior; $0$ means it does nothing. Sweep $c$ over (layer × position × component) and you get a **causal heatmap** localizing the computation.

### 4.2 Noising vs. denoising, and why direction matters

There are two directions, and they answer different questions:

- **Denoising** (corrupted run, patch *in* a clean activation): "Is this clean activation *sufficient* to recover the behavior?" This finds where the critical information *lives* in the clean run.
- **Noising** (clean run, patch *in* a corrupted activation): "Is this activation *necessary* — does destroying it break the behavior?"

The two can disagree, and the disagreement is informative; a component that is sufficient when denoised but whose ablation barely hurts (because of redundancy/backup) is a real phenomenon (see "backup heads" in the IOI work, §5). The corruption method also matters: **Gaussian noising** of embeddings (ROME's original choice) can push activations off-distribution, so the modern preference is **symmetric/interchange patching** — swap to a genuine alternative prompt (`Paris`↔`Rome`) so both runs stay on-distribution. The "Towards Automated Circuit Discovery" and Neel Nanda's exposition both stress this point.

!!! warning "Common pitfall: patching with LayerNorm and the residual stream"
    Two traps bite newcomers. (1) **Position alignment.** Clean and corrupted prompts must be the *same length and tokenization* or patched activations land at the wrong position and the result is noise. Pad or design prompts to match exactly. (2) **LayerNorm folding.** A patched residual changes the LN statistics (mean/variance) at that position, so the effect you measure includes LN's renormalization. For attribution that should be linear, libraries like TransformerLens offer *LN-frozen* or *folded* modes. If your patching results look bizarre, suspect length mismatch or LN before you suspect a deep finding.

### 4.3 A tiny activation-patching experiment from scratch

This self-contained script patches the **residual stream** of GPT-2 at every (layer, position) to localize where the "Eiffel Tower → Paris" fact is stored. Clean prompt resolves to ` Paris`; corrupted swaps the subject to "Colosseum," which resolves to ` Rome`. We measure recovery of the `Paris − Rome` logit difference.

```python
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()

# Same-length prompts so positions align token-for-token.
clean_text     = "The Eiffel Tower is in the city of"
corrupt_text   = "The Colosseum  is in the city of"   # crafted to tokenize to equal length
clean_ids   = tok(clean_text,   return_tensors="pt").input_ids.to(device)
corrupt_ids = tok(corrupt_text, return_tensors="pt").input_ids.to(device)
assert clean_ids.shape == corrupt_ids.shape, "prompts must align; adjust spacing/tokens"

paris = tok(" Paris").input_ids[0]
rome  = tok(" Rome").input_ids[0]
n_layer = model.config.n_layer
seq_len = clean_ids.shape[1]

def logit_diff(logits):
    # logits: (vocab,) at the final position; positive => model prefers Paris over Rome
    return (logits[paris] - logits[rome]).item()

# ── 1. Cache the CLEAN residual stream after every block, at every position. ────────
clean_cache = {}
def cache_hook(idx):
    def hook(_m, _i, out):
        clean_cache[idx] = out[0].detach().clone()   # (1, seq, d_model)
    return hook
handles = [blk.register_forward_hook(cache_hook(i)) for i, blk in enumerate(model.transformer.h)]
with torch.no_grad():
    clean_logits = model(clean_ids).logits[0, -1]
for h in handles: h.remove()
m_clean = logit_diff(clean_logits)

# ── 2. Baseline corrupted run (no patching). ───────────────────────────────────────
with torch.no_grad():
    corrupt_logits = model(corrupt_ids).logits[0, -1]
m_corrupt = logit_diff(corrupt_logits)
print(f"clean logit-diff   (Paris-Rome): {m_clean:+.2f}")
print(f"corrupt logit-diff (Paris-Rome): {m_corrupt:+.2f}\n")

# ── 3. Patch: for each (layer, position) overwrite the corrupted residual with the
#       clean one at that single site, run forward, measure recovery. ───────────────
def patch_hook(layer_idx, pos):
    def hook(_m, _i, out):
        h = out[0]
        h[:, pos, :] = clean_cache[layer_idx][:, pos, :]   # denoising: inject clean info
        return (h,) + tuple(out[1:])
    return hook

recovery = torch.zeros(n_layer, seq_len)
for layer in range(n_layer):
    for pos in range(seq_len):
        hk = model.transformer.h[layer].register_forward_hook(patch_hook(layer, pos))
        with torch.no_grad():
            m_patched = logit_diff(model(corrupt_ids).logits[0, -1])
        hk.remove()
        recovery[layer, pos] = (m_patched - m_corrupt) / (m_clean - m_corrupt + 1e-9)

# ── 4. Print the (layer × position) recovery heatmap as text. ──────────────────────
toks = [tok.decode([t]) for t in corrupt_ids[0].tolist()]
print("recovery of Paris-Rome logit diff (1.0 = clean fully restored):\n")
print("layer\\pos " + " ".join(f"{t.strip()[:6]:>7}" for t in toks))
for layer in range(n_layer):
    row = " ".join(f"{recovery[layer,p].item():7.2f}" for p in range(seq_len))
    print(f"   L{layer:<2}    {row}")
```


{{fig:mechinterp-patching-recovery-heatmap}}


The numbers are illustrative but the structure is the well-replicated **two-bump pattern** of factual recall (this is exactly the "early-site / late-site" signature ROME identified). Patching is most effective at the **subject token** ("Tower") in the **early-middle MLP layers** (here L5) — that is where the entity's attributes are looked up and written into the residual stream. Then effectiveness migrates to the **final position** ("of") in **later layers** (L9–L11), where attention has *moved* the recalled fact to the position that produces the next token. You have just causally localized a fact to a *layer and a token*, which is precisely the handle that knowledge-editing methods like ROME and MEMIT grab — see [Knowledge Editing & Machine Unlearning](../13-interp-safety-gov/02-knowledge-editing-unlearning.html).

!!! example "Worked example: reading the recovery metric"
    Suppose at $(L5,\ \text{Tower})$ the patched logit difference is $m_\text{patched} = +4.10$. With $m_\text{clean} = +6.40$ and $m_\text{corrupt} = -4.10$, recovery is
    $$
    \frac{m_\text{patched} - m_\text{corrupt}}{m_\text{clean} - m_\text{corrupt}} = \frac{4.10 - (-4.10)}{6.40 - (-4.10)} = \frac{8.20}{10.50} \approx 0.78.
    $$
    Restoring one residual vector at one position recovered ~78% of a 10.5-nat swing in the answer — strong evidence that this single site carries most of the distinguishing information. Compare that to $(L0,\ \text{The})$ at $0.00$: patching there does nothing, so no relevant computation has happened yet.

---

## 5. Circuit Discovery: How Components Compose into Algorithms

A single causal hot-spot is a clue; a **circuit** is the explanation. A circuit is a subgraph of the model's components — specific attention heads and MLP layers, plus the connections between them — that together implement a human-understandable algorithm, with the rest of the network ablated or shown irrelevant. Finding circuits is the most ambitious end of mech interp.

### 5.1 Two landmark circuits

**Induction heads** are the canonical example, from the "In-context Learning and Induction Heads" work (Olsson et al., 2022). An induction circuit implements the rule *"if the sequence `[A][B] ... [A]` appeared, predict `[B]`"* — basic copy-from-context that underlies much of in-context learning. It is a two-head, two-layer composition: a **previous-token head** in an early layer writes "the token before me was `A`" into each position; then an **induction head** in a later layer attends *back* to the position right after the earlier `A` (using that written signal as its key) and copies its value forward. The two heads compose through the residual stream — a textbook **K-composition** (one head's output becomes another's key). Strikingly, the formation of induction heads coincides with a phase change in the loss curve during training and with the emergence of in-context learning, tying a circuit to a capability.

**The IOI circuit** ("Interpretability in the Wild," Wang et al., 2022) reverse-engineered how GPT-2 small solves *indirect object identification* — completing "When Mary and John went to the store, John gave a drink to ___" with " Mary." The full circuit involves ~26 heads in classes that the authors named functionally: *duplicate-token heads* and *induction heads* detect that "John" is repeated, *S-inhibition heads* suppress the repeated name, and *name-mover heads* attend to and copy the correct name. It also revealed **backup name-mover heads** that activate only when the primary movers are ablated — built-in redundancy that explains why naive ablation can under-estimate a component's role, and why denoising and noising disagree (§4.2).


{{fig:mechinterp-ioi-circuit}}


### 5.2 How circuits are found

The methodology is the toolkit of §§2–4 applied systematically:

1. **Define a crisp task** with a metric (logit difference between the two plausible answers).
2. **Localize** with activation patching over heads/layers/positions to find the components that matter.
3. **Classify** each important head by what it attends to and writes (attention-pattern inspection, direct logit attribution, ablation of single heads).
4. **Verify composition** — show how outputs of upstream heads become inputs (queries/keys/values) of downstream ones via path patching.
5. **Validate** by *knockout*: ablate everything outside the proposed circuit and confirm the behavior survives; ablate inside and confirm it breaks.

Doing this by hand for 26 heads is heroic and does not scale. **Automated Circuit Discovery (ACDC)** (Conmy et al., 2023) automates the edge-pruning search: it greedily removes connections in the computational graph whose removal least hurts the metric, leaving a minimal subgraph. **Attribution patching** (a gradient-based linear approximation to patching, Nanda 2023; **AtP\*** is a refined variant) makes the search tractable on large models by estimating every component's patching effect from a *single* backward pass instead of one forward pass per component — the effect of patching component $c$ is approximated by the first-order term

$$
\Delta m_c \;\approx\; \big\langle\, \nabla_{a_c}\, m,\;\; a_c^\text{clean} - a_c^\text{corrupt} \,\big\rangle,
$$

the dot product of the metric's gradient w.r.t. the activation with the clean-minus-corrupt activation difference. One backward pass yields the gradients for *all* components at once, turning an $O(\text{components})$ sweep into $O(1)$ passes — at the cost of being a linear approximation that can mislead where the true effect is nonlinear.

---

## 6. Sparse Autoencoders & Transcoders: Decomposing Superposition

Patching and circuits operate on the model's *given* units — heads, layers, neurons. But §1.3 told us neurons are polysemantic because of superposition, so a circuit drawn over neurons is drawn over the wrong primitives. **Sparse autoencoders (SAEs)** attack the root problem: they try to *recover the underlying monosemantic features* that the model packed into superposition.

### 6.1 The architecture and objective

An SAE is a wide, sparse autoencoder trained to reconstruct a model's activations $x \in \mathbb{R}^{d_\text{model}}$ from a much larger but sparsely-active hidden layer of $d_\text{sae} \gg d_\text{model}$ features (an *expansion factor* of 8×, 32×, even higher):

$$
f = \operatorname{ReLU}\!\big(W_\text{enc}(x - b_\text{dec}) + b_\text{enc}\big) \in \mathbb{R}^{d_\text{sae}}, \qquad
\hat{x} = W_\text{dec}\, f + b_\text{dec}.
$$

$$
\mathcal{L} = \underbrace{\lVert x - \hat{x}\rVert_2^2}_{\text{reconstruction}} \;+\; \lambda \underbrace{\lVert f \rVert_1}_{\text{sparsity}}.
$$

The $\ell_1$ penalty forces the code $f$ to be sparse — only a few of the thousands of features fire per token. The columns of $W_\text{dec}$ are the **feature directions** in residual-stream space; the rows of $W_\text{enc}$ detect them. The bet, vindicated by Anthropic's "Towards Monosemanticity" (Bricken et al., 2023) and "Scaling Monosemanticity" (Templeton et al., 2024) and by Cunningham et al. (2023), is that the learned features are dramatically **more monosemantic** than neurons: individual features correspond to crisp concepts — "the Golden Gate Bridge," "DNA sequences," "code that is buggy," "deception," "sycophancy" — and they activate exactly where you'd expect.

The $\ell_1$ penalty has a known flaw: it penalizes feature *magnitude*, shrinking activations and biasing reconstruction. Two important fixes: **gated SAEs** (Rajamanoharan et al., 2024) split the "which features fire" decision from "how much," and **TopK / JumpReLU SAEs** enforce sparsity directly — TopK keeps the $k$ largest activations per token (no magnitude penalty at all), giving a cleaner reconstruction–sparsity frontier. **Transcoders** generalize the idea: instead of reconstructing one layer's activations, a transcoder learns a sparse, interpretable *replacement for the MLP* — it reads the MLP's input and predicts its output through a sparse feature bottleneck, making the MLP's computation itself legible and enabling cross-layer circuit analysis in feature space ("sparse feature circuits," Marks et al., 2024).

```python
import torch, torch.nn as nn, torch.nn.functional as F

class TopKSAE(nn.Module):
    """A minimal Top-K sparse autoencoder for residual-stream activations."""
    def __init__(self, d_model, d_sae, k):
        super().__init__()
        self.k = k
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.W_enc = nn.Parameter(torch.randn(d_model, d_sae) / d_model**0.5)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        # Tie decoder to a *unit-norm* dictionary so feature directions are comparable.
        self.W_dec = nn.Parameter(F.normalize(torch.randn(d_sae, d_model), dim=1))

    def encode(self, x):
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc          # (B, d_sae)
        # Top-K: keep only the k largest activations per token; zero the rest.
        topv, topi = pre.topk(self.k, dim=-1)
        f = torch.zeros_like(pre).scatter_(-1, topi, F.relu(topv))
        return f

    def forward(self, x):
        f = self.encode(x)
        x_hat = f @ self.W_dec + self.b_dec
        # No L1 term needed: Top-K enforces sparsity structurally.
        recon = F.mse_loss(x_hat, x)
        return x_hat, f, recon

# Training loop sketch (activations streamed from a frozen LM at one hook point):
# sae = TopKSAE(d_model=768, d_sae=768*32, k=32)
# for x in activation_loader:               # x: (B, 768), L2-normalized is common
#     x_hat, f, loss = sae(x)
#     loss.backward(); opt.step(); opt.zero_grad()
#     sae.W_dec.data = F.normalize(sae.W_dec.data, dim=1)   # re-project to unit norm
```

### 6.2 Evaluating and using SAEs

You judge an SAE on three axes that trade against each other:

- **Reconstruction fidelity** — typically the *fraction of variance explained*, or the *cross-entropy loss recovered* when you splice $\hat{x}$ back into the model in place of $x$ and re-run it. A good SAE loses little task performance.
- **Sparsity** — the average number of active features per token ($L_0$). Lower is more interpretable but harder to reconstruct from.
- **Interpretability** — do features correspond to crisp human concepts? Measured via auto-interpretation (an LLM labels each feature from its top-activating examples, then predicts activations on held-out text).

The applied payoff is large. SAE features give you a **monitoring vocabulary**: you can watch a "deception" or "refers to a known exploit" feature fire during generation, a far more targeted signal than an output classifier — covered as a safety control in §7. They also enable precise **steering**: add a multiple of a feature's decoder direction to the residual stream and the model's behavior shifts along that concept (Anthropic's public "Golden Gate Claude" demo amplified a single bridge feature). SAE-based **sparse feature circuits** let you draw circuits over interpretable features rather than polysemantic neurons.

!!! warning "SAEs are not a solved oracle"
    SAEs are the most exciting tool here and also the most over-hyped. Real limitations: **feature splitting** (one concept fractures into many near-duplicate features as you widen the dictionary, so the "right" number of features is undefined); **dead features** (many never activate and waste capacity); **reconstruction is lossy** (the residual error often contains task-relevant signal the SAE discarded); and there is no guarantee the learned dictionary matches the model's *actual* computational primitives rather than a convenient basis for reconstruction. Recent work questions whether SAE features improve downstream tasks over baselines. Treat SAE features as *useful hypotheses to be causally validated* (with patching), not as ground truth.

---

## 7. Practical Interpretability: Debugging, Steering, Monitoring & Tooling

### 7.1 Activation steering

The most directly useful product of mech interp for an engineer is **activation steering** (a.k.a. *representation engineering* / activation addition). The idea: find a direction $v$ in the residual stream that corresponds to a behavior, then at inference *add* $\alpha v$ to the residual at chosen layers to push the model toward (or, with $\alpha<0$, away from) that behavior — no fine-tuning, no weight changes.

How to get $v$? Three common routes: (1) **contrastive / difference-of-means** — average the residual activations on prompts that exhibit the behavior, subtract the average on prompts that don't (the "ActAdd" and "representation engineering" recipes; the **CAA**, contrastive activation addition, variant uses paired examples); (2) a **probe weight vector** from §2; (3) an **SAE feature** decoder direction from §6. A widely-reproduced result: the **refusal behavior of chat models is mediated by a single direction** (Arditi et al., 2024) — subtracting it largely disables refusals (a jailbreak), adding it makes the model refuse benign requests. Steering is also used constructively: suppressing a sycophancy or hallucination direction, enforcing a persona, or dialing honesty.

```python
# Steering by adding a precomputed direction `v` (unit norm) to the residual stream.
def steering_hook(v, alpha, positions=slice(None)):
    v = v.to(dtype=torch.float32)
    def hook(_m, _i, out):
        h = out[0]
        h[:, positions, :] = h[:, positions, :] + alpha * v   # broadcast add along d_model
        return (h,) + tuple(out[1:])
    return hook

# h = model.transformer.h[LAYER].register_forward_hook(steering_hook(v, alpha=+8.0))
# ... generate ...  then h.remove()
# alpha is in units of residual-stream norm; sweep it — too small = no effect,
# too large = incoherent text. The usable window is model/layer specific.
```

Steering's appeal is that it is cheap, reversible, and composable; its danger is that it is *blunt* — a direction tuned on one distribution can degrade fluency or have side effects off-distribution, and "one direction = one concept" is an approximation. Validate steered models on a broad eval suite, not just the target behavior.

### 7.2 Feature-based monitoring for safety

Interpretability promises a monitoring layer that watches *internal* state, not just I/O. Concretely: train a probe or select an SAE feature for a property you care about — "the model is about to comply with a harmful request," "the chain-of-thought is unfaithful," "the output references CBRN content" — and fire an alarm when it activates during generation. This complements the output classifiers in [Safety, Guardrails & Content Moderation](../12-production-mlops/04-safety-guardrails.html): an internal monitor can catch a problem *before* the token is emitted, and can in principle detect *intent* the output conceals (a model that "knows" it is deceiving). Anthropic and others have shown probes that detect when a model is being deceptive or sandbagging. The honest caveat: such monitors can be evaded if the model's representations drift or if it learns to route the computation around the monitored direction, so they are one layer of defense-in-depth, not a guarantee. This connects directly to the scalable-oversight agenda in [AI Safety: Scalable Oversight, Dangerous-Capability Evals & Frontier Safety](../13-interp-safety-gov/05-ai-safety-oversight.html).

### 7.3 Debugging workflows

Day-to-day, the techniques compose into a debugging discipline. Model emits a surprising token? **Direct logit attribution** (decompose the final logit into per-component contributions via the residual stream's linearity) tells you which heads/MLPs pushed it. A prompt that should trigger a behavior doesn't? **Logit lens** shows at which layer the prediction diverges from expectation. A backdoor or spurious trigger suspected? **Activation patching** between triggered and clean inputs localizes the responsible component. A fine-tune that regressed? Compare logit-lens trajectories or probe accuracies before/after to see *where* the representation changed.

### 7.4 Tooling

You rarely write hooks by hand for serious work:

- **TransformerLens** (Neel Nanda et al.) is the de facto standard for mech interp on open models. It loads models into a uniform, hookable format (`HookedTransformer`), exposes every internal activation by a consistent name (`blocks.5.attn.hook_z`, `blocks.5.hook_resid_post`), folds LayerNorm for clean attribution, and provides `run_with_cache` and `run_with_hooks` so patching is a few lines instead of the boilerplate we wrote above. Our from-scratch code is for understanding; TransformerLens is for getting work done.
- **SAELens** and **dictionary_learning** train, store, and load SAEs; **Neuronpedia** hosts pretrained SAEs and an interactive feature browser with auto-interpretation labels for many open models.
- **nnsight / NDIF** expose internals of very large models (including remote execution) with a clean intervention API; **captum** and **pyvene** cover attribution and intervention more broadly.
- **Gemma Scope** (a large open suite of SAEs for Gemma 2) and the SAEs released for GPT-2 and Llama give you pretrained dictionaries so you need not train your own to start.

!!! tip "Practitioner tip: start observational, escalate to causal, always validate"
    A reliable order of operations: (1) **logit lens / probing** to form a hypothesis cheaply; (2) **activation patching** to test it causally on a crisp metric; (3) **circuit/SAE analysis** only if you need mechanism, not just localization; (4) **knockout validation** — ablate your proposed circuit and confirm the behavior breaks, ablate everything else and confirm it survives. Never ship a claim that rests on a heatmap alone; correlation-flavored tools (probes, lens) must be backed by an intervention before you believe them.

---

## 8. Honest Limits and the State of the Field

Mech interp is the most intellectually thrilling corner of LLM research and it is **far from solved**. Stating the limits plainly is part of using it responsibly.

- **It does not scale (yet).** The clean, fully reverse-engineered circuits — IOI, induction, modular addition (the "grokking" work, Nanda et al., 2023) — are on *small* models and *narrow* tasks. We do not have an end-to-end mechanistic account of any frontier model's general behavior. Automated methods (ACDC, attribution patching, SAEs) push the frontier but trade rigor for tractability.
- **Superposition is only partially tamed.** SAEs are progress, not closure: feature splitting, lossy reconstruction, and the absence of a ground-truth feature set mean we cannot yet claim to have "the" decomposition of a model's representations.
- **Faithfulness is hard to verify.** A clean-looking explanation can be wrong — the model may use a different mechanism that happens to correlate with your story. The field's defense is intervention (patching, ablation), but interventions can be confounded (off-distribution activations, LN effects, backup circuits masking necessity).
- **Findings can be illusory.** Apparent features or circuits sometimes reflect the *probe's* or *SAE's* inductive bias rather than the model's computation. Control tasks, baselines, and replication across seeds are not optional.
- **Cost and labor.** Serious circuit analysis is research-grade effort; for many production problems an output classifier or a behavioral eval is the right tool, and interpretability is reserved for high-stakes debugging, safety monitoring, or science.

None of this is a reason for cynicism. The trajectory — from word2vec analogies, to the logit lens, to causal tracing, to SAEs scaling to production models — is steep, and interpretability has already produced shippable tools: steering vectors, refusal-direction analysis, factual-recall localization that powers knowledge editing, and internal monitors for safety. The realistic stance is the engineer's stance: these are sharp, imperfect instruments; use the right one, validate causally, and report your uncertainty.

!!! interview "Interview Corner"
    **Q:** You suspect a model has a backdoor: when the token "SolidGoldMagikarp" appears, it produces unsafe completions. Walk me through how you would *localize* the mechanism, and how you'd be sure your explanation is causal rather than correlational.

    **A:** I'd treat it as an activation-patching problem with a crisp metric. First, two aligned inputs: a **clean** prompt containing the trigger (unsafe behavior) and a **corrupted** one where I swap the trigger for a benign token of equal token-length (safe behavior). Pick a metric — logit difference between an unsafe and a safe continuation. Then **denoising-patch**: run the corrupted prompt but inject the clean residual stream at one (layer, position) at a time, sweeping all sites, and measure recovery $(m_\text{patched}-m_\text{corrupt})/(m_\text{clean}-m_\text{corrupt})$. The (layer, position) where recovery spikes localizes where the trigger's effect enters and where it acts. To go from localization to *mechanism*, classify the implicated attention heads by their attention pattern and use direct logit attribution to see what they write. Crucially, to make it **causal not correlational**, I validate by **knockout**: ablate the proposed component(s) and confirm the unsafe behavior disappears on triggered inputs while clean behavior is intact; conversely confirm that ablating everything *outside* my proposed circuit leaves the behavior intact. I'd also run *both* noising and denoising — if denoising says a head is sufficient but ablating it doesn't hurt, I should suspect a **backup circuit** and widen my analysis. A probe that merely predicts "trigger present" from activations is *not* sufficient evidence — that's correlation; the patching/knockout interventions are what license a causal claim.

!!! key "Key Takeaways"
    - The **residual stream** is an additive, shared communication bus: every sub-layer reads from and writes to it, which makes downstream quantities (logits, attention scores) *linearly decomposable* into per-component contributions — the basis of every attribution method here.
    - **Superposition** lets a model pack far more sparse features than it has dimensions into non-orthogonal directions; the cost is **polysemantic neurons**, which is why you cannot interpret a network neuron-by-neuron and why SAEs exist.
    - **Probing** and the **logit lens** are cheap *observational* tools — they reveal what information is present and where predictions sharpen — but they measure correlation; use control tasks for probes and remember a positive probe does not imply the model *uses* the feature.
    - **Activation patching / causal tracing** is the workhorse for *causal* claims: patch a clean activation into a corrupted run (denoising) or vice-versa (noising) and measure recovery to localize a behavior to a (layer, position, component). Mind LayerNorm and position alignment.
    - **Circuits** (induction heads, the IOI circuit) are reverse-engineered algorithms over heads/MLPs; finding them combines patching, attention-pattern analysis, and knockout validation, with **ACDC** and **attribution patching** automating the search.
    - **Sparse autoencoders / transcoders** decompose superposition into a larger set of sparser, more monosemantic features, enabling feature-level monitoring, steering, and circuits — but suffer feature splitting, lossy reconstruction, and no ground-truth, so features are hypotheses to validate, not oracles.
    - Applied payoffs are real today: **activation steering** (e.g. the single refusal direction), **feature-based safety monitoring** that inspects internal state before a token is emitted, and **factual-recall localization** that powers knowledge editing.
    - Use **TransformerLens** for hooks/caching/patching, **SAELens / Neuronpedia / Gemma Scope** for SAEs; reach for from-scratch hooks only to learn. Always escalate from observational to causal and finish with knockout validation.

!!! sota "State of the Art & Resources (2026)"
    Mechanistic interpretability has evolved from manually tracing toy circuits in small models to automated methods and sparse autoencoders that operate at the scale of frontier systems; the field is moving fast but reverse-engineering a full frontier model's general behavior remains an open challenge.

    **Foundational work**

    - [Elhage et al., *A Mathematical Framework for Transformer Circuits* (2021)](https://transformer-circuits.pub/2021/framework/index.html) — establishes the residual-stream / virtual-weights view that underpins every method here.
    - [Elhage et al., *Toy Models of Superposition* (2022)](https://transformer-circuits.pub/2022/toy_model/index.html) — clean proof that networks exploit superposition to pack more features than dimensions; explains polysemanticity.
    - [Olsson et al., *In-context Learning and Induction Heads* (2022)](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — the canonical two-head induction circuit and its link to the emergence of in-context learning.
    - [Meng et al., *Locating and Editing Factual Associations in GPT* / ROME (2022)](https://arxiv.org/abs/2202.05262) — introduced causal tracing; the two-site factual-recall pattern exploited by knowledge-editing methods.

    **Recent advances (2023–2026)**

    - [Wang et al., *Interpretability in the Wild: IOI Circuit in GPT-2 Small* (2022)](https://arxiv.org/abs/2211.00593) — full reverse-engineering of indirect-object identification; revealed backup heads and the limits of ablation.
    - [Cunningham et al., *Sparse Autoencoders Find Highly Interpretable Features* (2023)](https://arxiv.org/abs/2309.08600) — early empirical evidence that SAEs recover monosemantic features at scale.
    - [Conmy et al., *Towards Automated Circuit Discovery* / ACDC (2023)](https://arxiv.org/abs/2304.14997) — greedy edge-pruning algorithm that automates circuit finding in GPT-2.
    - [Templeton et al., *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet* (2024)](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) — SAEs at frontier model scale; features map to concepts including "deception" and specific entities.
    - [Lieberum et al., *Gemma Scope* (2024)](https://arxiv.org/abs/2408.05147) — open suite of hundreds of JumpReLU SAEs for Gemma 2 2B/9B/27B; community baseline for SAE research.
    - [Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction* (2024)](https://arxiv.org/abs/2406.11717) — shows refusal is a one-dimensional subspace in 13 open chat models; a concrete applied result of mech interp.
    - [Marks et al., *Sparse Feature Circuits* (2024)](https://arxiv.org/abs/2403.19647) — circuits drawn over SAE features rather than polysemantic neurons; enables interpretable causal graphs.

    **Open-source & tools**

    - [TransformerLensOrg/TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) — de facto standard library for hookable, activation-patching-friendly transformer analysis; 50+ supported models.
    - [jbloomAus/SAELens](https://github.com/jbloomAus/SAELens) — train, load, and analyze sparse autoencoders; deep TransformerLens integration and pre-trained dictionaries.
    - [Neuronpedia](https://www.neuronpedia.org) — hosted interactive feature browser with auto-interpretation labels, steering controls, and circuit tracing for dozens of open models including Gemma Scope.

## Further reading

- Elhage, Nanda, Olsson, et al., *A Mathematical Framework for Transformer Circuits* (Anthropic, 2021) — the residual-stream/virtual-weights view.
- Elhage et al., *Toy Models of Superposition* (Anthropic, 2022) — superposition and polysemanticity from first principles.
- Olsson et al., *In-context Learning and Induction Heads* (Anthropic, 2022) — the induction circuit and its link to in-context learning.
- Wang, Variengien, Conmy, Shlegeris & Steinhardt, *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small* (2022) — the IOI circuit.
- Meng, Bau, Andonian & Belinkov, *Locating and Editing Factual Associations in GPT* (ROME, 2022) — causal tracing of factual recall.
- nostalgebraist, *interpreting GPT: the logit lens* (2020); Belrose et al., *Eliciting Latent Predictions from Transformers with the Tuned Lens* (2023).
- Bricken et al., *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning* (Anthropic, 2023); Templeton et al., *Scaling Monosemanticity* (Anthropic, 2024); Cunningham et al., *Sparse Autoencoders Find Highly Interpretable Features* (2023).
- Conmy et al., *Towards Automated Circuit Discovery* (ACDC, 2023); Nanda, *Attribution Patching* (2023); Rajamanoharan et al., *Gated / JumpReLU SAEs* (DeepMind, 2024); *Gemma Scope* (2024).
- Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction* (2024); Marks et al., *Sparse Feature Circuits* (2024).
- **TransformerLens** (Neel Nanda) and **SAELens / Neuronpedia** — the standard open tooling.
