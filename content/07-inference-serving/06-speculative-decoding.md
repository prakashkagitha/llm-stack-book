# 7.6 Speculative Decoding: Draft Models, Medusa, EAGLE & Lookahead

Autoregressive decoding has a humiliating property: to produce one token, a 70-billion-parameter model must move every one of its weights from HBM (high-bandwidth memory) to the compute units, do a tiny amount of arithmetic, and throw the weights away — only to load them all again for the next token. As we saw in [The Anatomy of LLM Inference: Prefill, Decode & The KV Cache](../07-inference-serving/01-anatomy-inference.html), the decode phase is **memory-bandwidth bound**. The GPU spends almost all of its time shuttling parameters, and almost none of it doing math. A single decode step on an A100 might touch 140 GB of weights (bf16, 70B params) to compute logits for *one* position. The arithmetic intensity is laughably low.

Here is the key observation that makes speculative decoding possible: a forward pass over a batch of $K$ tokens costs almost exactly the same wall-clock time as a forward pass over $1$ token, because in both cases you load all the weights once. The cost of decode is dominated by weight movement, not by the per-token FLOPs. So if we could somehow *guess* the next $K$ tokens cheaply, and then **verify all $K$ of them in a single forward pass** of the big model, we would get up to $K$ tokens for the price of one — without changing the output distribution at all.

That is exactly what speculative decoding does. A cheap **drafter** proposes a short continuation; the expensive **target** model verifies it in one parallel pass; a clever accept/reject rule guarantees the accepted tokens are distributed *exactly* as if the target had sampled them one at a time. This chapter derives that guarantee from scratch, implements it, and then surveys the family of modern drafters — separate draft models, **Medusa** heads, the **EAGLE** line, and **lookahead decoding** — along with the tree-attention machinery that makes verifying many candidates cheap. We will be precise about what determines the speedup: the **acceptance rate** and the **cost ratio** between drafter and target.

## The Core Idea: Draft, Then Verify

{{fig:spec-decoding}}

Let the target model define a next-token distribution $p(\cdot \mid x_{<t})$ over the vocabulary, given the prefix $x_{<t}$. Ordinary autoregressive sampling draws $x_t \sim p(\cdot \mid x_{<t})$, appends it, and repeats — one expensive forward pass per token.

Speculative decoding introduces a second, much cheaper model $q$, the **draft model** (or *drafter*). One full speculative step looks like this:


{{fig:specdec-draft-verify-pipeline}}


Two facts make this a win:

1. **The drafter is cheap.** Generating $\gamma$ draft tokens with $q$ costs $\gamma$ forward passes of a small model. If $q$ is, say, 30–70× smaller than $p$, those $\gamma$ steps are negligible.
2. **Verification is parallel.** The target $p$ scores all $\gamma+1$ candidate positions in **one** forward pass. Because decode is memory-bound, scoring $\gamma+1$ positions costs about the same as scoring one — we load the weights once.

If we accept $n$ draft tokens plus one bonus token, we advanced the sequence by $n+1$ positions while paying for **one** target forward pass and $\gamma$ cheap drafter passes. The expected number of tokens per target pass is the engine of the speedup.

The non-obvious part — and the reason this is *exact* rather than an approximation — is the accept/reject rule. We need it to guarantee that the emitted tokens have *exactly* the distribution $p$, even though they were proposed by $q$. We derive that next.

## The Accept/Reject Rule: Preserving the Target Distribution

Consider a single position. The drafter proposes $x \sim q(x)$ (we drop the conditioning on the prefix for brevity; everything is conditional on the current context). We want the *emitted* token to be distributed as $p(x)$, the target distribution. The mechanism is **modified rejection sampling**.

### The rule

Given a drafted token $x \sim q$:

- **Accept** $x$ with probability $\min\!\left(1, \dfrac{p(x)}{q(x)}\right)$.
- If **rejected**, sample a replacement token from the **residual distribution**

$$
p_{\text{res}}(x) \;=\; \frac{\max\!\big(0,\; p(x) - q(x)\big)}{\sum_{x'} \max\!\big(0,\; p(x') - q(x')\big)}.
$$

We write $(\cdot)_+ = \max(0,\cdot)$ for the positive part, so $p_{\text{res}}(x) \propto (p(x)-q(x))_+$. The intuition: where the drafter under-covers the target ($p > q$), we top up the probability mass; where it over-covers ($p < q$), the accept probability $p/q < 1$ trims the excess.

### The proof that this is exact

We must show the probability of emitting any specific token $x$ equals $p(x)$. The emitted token comes from one of two mutually exclusive events: *(a)* the drafter proposed $x$ and we accepted it, or *(b)* the drafter proposed something, we rejected it, and the residual sample landed on $x$.

**Event (a): drafted $x$ and accepted.**

$$
P(\text{draft } x \text{ and accept}) \;=\; q(x)\cdot \min\!\left(1, \frac{p(x)}{q(x)}\right) \;=\; \min\big(q(x),\, p(x)\big).
$$

**Event (b): some token rejected, then residual produced $x$.** First, the total probability of *any* rejection. Conditioned on a draft $x'$, rejection probability is $1 - \min(1, p(x')/q(x'))$, so

$$
\beta \;\equiv\; P(\text{reject}) \;=\; \sum_{x'} q(x')\Big(1 - \min\big(1, \tfrac{p(x')}{q(x')}\big)\Big) \;=\; \sum_{x'} \big(q(x') - \min(q(x'), p(x'))\big) \;=\; \sum_{x'} \big(q(x') - p(x')\big)_+ .
$$

The last equality holds because $q - \min(q,p) = (q-p)_+$. By symmetry of total mass ($\sum p = \sum q = 1$), $\sum_{x'} (q-p)_+ = \sum_{x'}(p-q)_+$, so $\beta$ is also the normalizer of the residual distribution. Therefore the probability of rejecting *and then* producing $x$ from $p_{\text{res}}$ is

$$
P(\text{reject and residual}=x) \;=\; \beta \cdot p_{\text{res}}(x) \;=\; \beta \cdot \frac{(p(x)-q(x))_+}{\beta} \;=\; \big(p(x)-q(x)\big)_+ .
$$

The $\beta$ cancels — a clean and satisfying step. Now add the two events:

$$
P(\text{emit } x) \;=\; \min\big(q(x), p(x)\big) \;+\; \big(p(x)-q(x)\big)_+ .
$$

Split on the sign of $p(x)-q(x)$. If $p(x) \ge q(x)$: $\min(q,p)=q(x)$ and $(p-q)_+ = p(x)-q(x)$, summing to $p(x)$. If $p(x) < q(x)$: $\min(q,p)=p(x)$ and $(p-q)_+ = 0$, summing to $p(x)$. In **both** cases $P(\text{emit }x) = p(x)$. $\blacksquare$

This is the whole theoretical foundation. The emitted token is *exactly* a sample from the target $p$, regardless of how good or bad the drafter $q$ is. The drafter only affects **speed**, never correctness — a property worth stating loudly because it is what lets production systems turn speculation on by default.

!!! note "Aside: where the acceptance probability comes from"
    The expected acceptance probability at one position is $\alpha = \sum_x \min(p(x), q(x))$, which equals $1 - \tfrac{1}{2}\lVert p - q\rVert_1$ — one minus the total variation distance between target and draft. So acceptance is high exactly when the drafter's distribution is close to the target's in TV distance. This is why a well-aligned small model, or feature-level drafters like EAGLE, accept so much more than a random tiny model. The same quantity $\alpha$ drives the speedup formula in the last section.

### Extending to a chain of $\gamma$ draft tokens

The single-position rule chains naturally. We walk the draft left to right. At each drafted position we apply the accept/reject test against the target distribution *at that same position* (which the single parallel target pass already gave us). The moment a token is **rejected**, we sample its replacement from the residual and **stop** — every draft token after a rejection is discarded, because it was conditioned on a token we just changed.

There is one more free token. If *all* $\gamma$ draft tokens are accepted, the target's forward pass also produced logits for the position *after* the last draft token (position $\gamma+1$). We can sample that token directly from $p$ as a **bonus token** — a genuinely free extra token, since that logit was computed anyway. So a step emits between $1$ (immediate rejection, replaced token) and $\gamma+1$ (all accepted plus bonus) tokens.

The greedy/argmax special case is even simpler: with temperature 0, $p$ and $q$ are point masses, accept iff the drafter's argmax equals the target's argmax. "Accept the longest matching prefix" is exactly the rejection rule specialized to deterministic distributions.

## A From-Scratch Implementation

Below is a complete, runnable implementation of speculative sampling with a real target and draft model from the same family (so they share a tokenizer — a hard requirement). It is written for clarity, not maximum throughput; production engines fuse these steps into batched CUDA kernels with tree attention (next section). Read the comments carefully: the accept/reject loop is the heart of the algorithm.

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Target = big, accurate. Draft = small, fast. They MUST share a tokenizer.
# (Any GPT-2 size pair works as a demo; swap in a 70B target / 7B draft in prod.)
TARGET_NAME = "gpt2-large"   # ~774M params
DRAFT_NAME  = "gpt2"         # ~124M params

tok    = AutoTokenizer.from_pretrained(TARGET_NAME)
target = AutoModelForCausalLM.from_pretrained(TARGET_NAME).eval().cuda()
draft  = AutoModelForCausalLM.from_pretrained(DRAFT_NAME).eval().cuda()


def apply_temperature(logits, temperature):
    """Return a probability distribution from logits with temperature.
    temperature == 0 collapses to a one-hot argmax (greedy)."""
    if temperature == 0:
        probs = torch.zeros_like(logits)
        probs[..., logits.argmax(-1)] = 1.0
        return probs
    return F.softmax(logits / temperature, dim=-1)


@torch.no_grad()
def speculative_step(prefix_ids, gamma=4, temperature=1.0):
    """
    Run ONE speculative step.
      prefix_ids: LongTensor [1, T] current context (on cuda)
      gamma:      number of draft tokens to propose
      temperature: sampling temperature for BOTH models
    Returns: new_ids [1, T+n+1] and the number of *accepted draft* tokens n.
    """
    device = prefix_ids.device
    T = prefix_ids.shape[1]

    # ---- 1) DRAFT: γ cheap autoregressive steps with the small model. ----
    draft_ids   = prefix_ids.clone()
    draft_probs = []  # q-distribution at each drafted position
    for _ in range(gamma):
        d_logits = draft(draft_ids).logits[:, -1, :]      # [1, V]
        q = apply_temperature(d_logits, temperature)      # [1, V]
        x = torch.multinomial(q, num_samples=1)           # [1, 1] sampled draft token
        draft_probs.append(q)
        draft_ids = torch.cat([draft_ids, x], dim=1)
    # draft_ids is now [1, T+γ]; the last γ tokens are the proposals d_1..d_γ.

    # ---- 2) VERIFY: ONE parallel target pass over all T+γ positions. ----
    # The target scores every position at once. We need its distribution at
    # positions T-1 .. T+γ-1 (the predictions for d_1..d_γ AND the bonus slot).
    t_logits = target(draft_ids).logits                   # [1, T+γ, V]
    target_probs = apply_temperature(
        t_logits[:, T - 1 : T + gamma, :].squeeze(0),     # [γ+1, V]
        temperature,
    )  # rows 0..γ-1 verify d_1..d_γ; row γ is the bonus-token distribution.

    # ---- 3) ACCEPT/REJECT loop, left to right. ----
    accepted = []
    n = 0
    for i in range(gamma):
        d_tok = draft_ids[0, T + i].item()                # the i-th drafted token
        p_i = target_probs[i]                             # target dist at this position
        q_i = draft_probs[i].squeeze(0)                   # draft  dist at this position
        # Accept with prob min(1, p(d)/q(d)).
        r = torch.rand(1, device=device).item()
        accept_prob = min(1.0, (p_i[d_tok] / q_i[d_tok]).item())
        if r < accept_prob:
            accepted.append(d_tok)
            n += 1
        else:
            # REJECT: sample replacement from residual (p - q)_+ , then STOP.
            residual = torch.clamp(p_i - q_i, min=0.0)
            residual = residual / residual.sum()
            repl = torch.multinomial(residual, num_samples=1).item()
            accepted.append(repl)
            break
    else:
        # No break => all γ accepted. Sample the FREE bonus token from row γ.
        bonus = torch.multinomial(target_probs[gamma], num_samples=1).item()
        accepted.append(bonus)

    new_tokens = torch.tensor([accepted], device=device, dtype=prefix_ids.dtype)
    new_ids = torch.cat([prefix_ids, new_tokens], dim=1)
    return new_ids, n


@torch.no_grad()
def generate(prompt, max_new_tokens=128, gamma=4, temperature=1.0):
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    start_len = ids.shape[1]
    total_accepted, total_steps = 0, 0
    while ids.shape[1] - start_len < max_new_tokens:
        ids, n = speculative_step(ids, gamma=gamma, temperature=temperature)
        total_accepted += n
        total_steps += 1
    # Average draft tokens accepted per target pass is the key efficiency metric.
    avg_accept = total_accepted / max(total_steps, 1)
    print(f"target passes: {total_steps}, mean accepted draft/step: {avg_accept:.2f}")
    return tok.decode(ids[0, start_len:], skip_special_tokens=True)


# print(generate("The future of machine learning is", temperature=0.8))
```

A few engineering notes the code glosses over but production systems must handle:

- **KV-cache reuse.** Re-running `draft(draft_ids)` and `target(draft_ids)` from scratch each step is wasteful. Real implementations pass cached KV and only feed the new tokens, so the drafter's $\gamma$ steps each see one new position and the target's verify pass sees $\gamma$ new positions. After a step, both caches must be **rolled back** to the accepted length — the rejected/unused positions' KV entries are evicted. See [PagedAttention & KV-Cache Memory Management](../04-kernels-efficiency/06-paged-attention-kv.html) for how paged caches make this cheap.
- **Numerical stability.** Compute $p/q$ carefully; clamp $q$ away from zero. With temperature 0, special-case the comparison to argmax to avoid $0/0$.
- **Batching.** Different requests in a batch accept different numbers of tokens, so the "advance" is ragged. This interacts non-trivially with [Continuous Batching & Request Scheduling](../07-inference-serving/02-continuous-batching.html); engines pad to the max accepted length or use variable-length kernels.

### KV-Cache Reuse and Rollback

The `speculative_step` above re-runs `draft(draft_ids)` and `target(draft_ids)` over the *entire* sequence every single step — an $O(T)$ recompute that gets more wasteful the longer the generation runs. A production implementation instead keeps a KV cache for each model and feeds only the *new* tokens at every step, then **rolls each cache back** to the accepted length with `DynamicCache.crop(...)` once accept/reject is resolved. The invariant to hold in your head: at the start of a step, both caches hold KV for the committed prefix *except* the final committed token, which is left "pending" and re-fed as the first input of the next step. Equivalently, `cache.get_seq_length()` is always one less than the committed sequence length, and each step's first move is simply to feed `prefix_ids[:, cache.get_seq_length():]` to catch the cache up. This is exactly the bookkeeping Hugging Face's assisted-generation implementation performs internally.

```python
import torch
from transformers import DynamicCache  # transformers >= 4.44

# reuses apply_temperature, target, draft, tok from the previous listing


@torch.no_grad()
def speculative_step_cached(prefix_ids, draft_cache, target_cache, gamma=4, temperature=1.0):
    """
    Same contract as speculative_step, but threads and mutates two
    DynamicCache objects (one per model) instead of recomputing from scratch.
    Returns: new_ids, n, draft_cache, target_cache
    """
    device = prefix_ids.device
    T = prefix_ids.shape[1]

    # ---- 1) DRAFT: γ cheap steps, feeding only the tokens not yet cached. ----
    draft_ids   = prefix_ids.clone()
    draft_probs = []
    for i in range(gamma):
        if i == 0:
            # Gap-prefill: catch the draft cache up to the current prefix.
            # Normally this is just the one pending token, but see the
            # all-accepted note below — it can legitimately be more.
            cur = draft_ids[:, draft_cache.get_seq_length():]
        else:
            cur = draft_ids[:, -1:]                        # one new token
        out = draft(cur, past_key_values=draft_cache, use_cache=True)
        draft_cache = out.past_key_values                  # reassign (in-place DynamicCache)
        q = apply_temperature(out.logits[:, -1, :], temperature)   # [1, V]
        x = torch.multinomial(q, num_samples=1)             # [1, 1]
        draft_probs.append(q)
        draft_ids = torch.cat([draft_ids, x], dim=1)
    # draft_cache now holds KV for positions 0..T+gamma-2 — i.e. everything
    # up to (but not including) the last draft token d_gamma, which is
    # intentionally left uncached.

    # ---- 2) VERIFY: ONE target pass over only the uncached tail. ----
    # In steady state this tail is the pending token plus d_1..d_gamma:
    # gamma+1 tokens.
    t_in = draft_ids[:, target_cache.get_seq_length():]
    out = target(t_in, past_key_values=target_cache, use_cache=True)
    target_cache = out.past_key_values
    # Use the LAST gamma+1 logits, not absolute indices: on the very first
    # step t_in also contains the whole prompt, so its length varies.
    target_probs = apply_temperature(
        out.logits[0, -(gamma + 1):, :], temperature
    )  # [gamma+1, V]; row k predicts position T+k: rows 0..gamma-1 verify
       # d_1..d_gamma, row gamma is the bonus-token distribution.

    # ---- 3) ACCEPT/REJECT loop — identical logic to speculative_step. ----
    accepted = []
    n = 0
    for i in range(gamma):
        d_tok = draft_ids[0, T + i].item()
        p_i = target_probs[i]
        q_i = draft_probs[i].squeeze(0)
        accept_prob = min(1.0, (p_i[d_tok] / q_i[d_tok]).item())
        if torch.rand(1, device=device).item() < accept_prob:
            accepted.append(d_tok)
            n += 1
        else:
            residual = torch.clamp(p_i - q_i, min=0.0)
            residual = residual / residual.sum()
            accepted.append(torch.multinomial(residual, num_samples=1).item())
            break
    else:
        accepted.append(torch.multinomial(target_probs[gamma], num_samples=1).item())

    new_ids = torch.cat(
        [prefix_ids, torch.tensor([accepted], device=device, dtype=prefix_ids.dtype)],
        dim=1,
    )

    # ---- 4) ROLLBACK: crop both caches to the accepted length. ----
    # `keep` is the number of context tokens whose KV is valid going into the
    # next step — one less than the new committed length T+n+1, matching the
    # "pending final token" invariant described above.
    keep = T + n
    # DynamicCache.crop(max_length) truncates every layer's K/V tensors to
    # max_length along the sequence dimension. It is a NO-OP when
    # max_length >= cache.get_seq_length(). Two cases:
    #   (1) Some draft token rejected (n < gamma): the draft cache currently
    #       holds T+gamma-1 positions; crop(keep) truncates it down to T+n,
    #       discarding the speculative-but-now-invalid positions.
    #   (2) ALL gamma drafts accepted (n == gamma): keep = T+gamma, which
    #       exceeds the draft cache's current length T+gamma-1, so
    #       draft_cache.crop is a no-op — the draft cache legitimately LAGS
    #       the committed prefix by one token (d_gamma was never fed to the
    #       drafter). This is fine and self-healing: next step's gap-prefill
    #       `draft_ids[:, draft_cache.get_seq_length():]` will simply feed
    #       TWO tokens (d_gamma and the bonus) instead of one. The target
    #       cache never lags this way, since the verify pass always
    #       processes through position T+gamma.
    target_cache.crop(keep)
    draft_cache.crop(keep)

    return new_ids, n, draft_cache, target_cache


@torch.no_grad()
def generate_cached(prompt, max_new_tokens=128, gamma=4, temperature=1.0):
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    start = ids.shape[1]
    draft_cache = DynamicCache()   # empty: first step's gap-prefill feeds the
    target_cache = DynamicCache()  # whole prompt to both models.
    total_accepted, total_steps = 0, 0
    while ids.shape[1] - start < max_new_tokens:
        ids, n, draft_cache, target_cache = speculative_step_cached(
            ids, draft_cache, target_cache, gamma=gamma, temperature=temperature
        )
        total_accepted += n
        total_steps += 1
    # Target cache always trails the committed length by exactly one (the
    # pending-token invariant). The draft cache may trail by two right after
    # an all-accepted step, so we only bound it, not assert equality.
    assert target_cache.get_seq_length() == ids.shape[1] - 1
    assert draft_cache.get_seq_length() <= ids.shape[1] - 1
    avg_accept = total_accepted / max(total_steps, 1)
    print(f"target passes: {total_steps}, mean accepted draft/step: {avg_accept:.2f}")
    return tok.decode(ids[0, start:], skip_special_tokens=True)
```

Correctness is easy to check reproducibly. At `temperature=0`, speculative decoding is deterministic and lossless — the accept/reject rule collapses to argmax comparison — so `generate_cached` must produce a byte-identical continuation to both the naive `generate(...)` above and plain Hugging Face greedy decoding: `tok.decode(target.generate(tok(prompt, return_tensors="pt").input_ids.cuda(), max_new_tokens=128, do_sample=False)[0, start:])`. Concretely: `assert generate_cached("The future of machine learning is", temperature=0) == generate("The future of machine learning is", temperature=0)`. At `temperature > 0` the two only match if the same RNG draws happen in the same order (seed with `torch.manual_seed` before each call), since sampling consumes randomness — the greedy-equivalence check is the robust invariant to test. As a speed sanity check, `generate_cached` should be several times faster wall-clock than `generate` for long generations, since it drops the $O(T)$ per-step recompute entirely.

Hugging Face Transformers implements exactly this rollback in `transformers/generation/utils.py` (the assisted-generation path and `_speculative_sampling`), backed by `DynamicCache.crop` in `transformers/cache_utils.py`. For **tree verification** (see the Tree Attention section below), crop-by-length no longer applies — instead of truncating a contiguous suffix, you keep exactly the accepted root-to-leaf path's nodes with a per-layer gather: `key_cache[l] = key_cache[l].index_select(-2, keep_idx)` (and likewise for `value_cache`), where `keep_idx` is a `LongTensor` of the flattened node indices on the accepted path and `-2` is the sequence dimension of the `[batch, heads, seq, head_dim]` KV tensors. This `index_select` mechanics is what paged/tree engines like vLLM, SGLang, and gpt-fast's speculative-decoding worker use under the hood.

## Choosing and Training a Draft Model

The accept/reject proof says correctness is free; the drafter only governs throughput. So drafter design is purely an optimization problem with two competing knobs:

1. **Acceptance rate $\alpha$** — how often the target accepts a draft token. Higher is better; it grows with the TV-closeness of $q$ to $p$. A drafter from the same model family, trained on the same data, accepts far more than an unrelated small model.
2. **Draft cost ratio $c$** — the drafter's per-token cost as a fraction of the target's. Lower is better. A 70B target with a 7B drafter has $c \approx 0.1$; with a 1B drafter, $c \approx 0.014$.

These trade off: a bigger drafter raises $\alpha$ but also raises $c$. The sweet spot for a separate-model drafter is usually a model 10–50× smaller than the target, from the same family (e.g., Llama-70B target with a Llama-1B/7B draft).

### Sources of drafters

- **Smaller sibling in the same family.** The classic setup from the original Google/DeepMind speculative-decoding papers (Leviathan et al.; Chen et al., 2023). Easy if the family ships multiple sizes with a shared tokenizer. The shared tokenizer is non-negotiable: token IDs must mean the same thing in both models for the accept test to be coherent.
- **Distilled drafter.** Train a small model specifically to mimic the target via [Distillation, Model Compression & Knowledge Transfer](../05-posttraining-alignment/12-distillation-compression.html). Distilling on the target's *own* outputs (sequence-level or logit-level KD) maximizes TV-closeness and thus $\alpha$ — far better than an off-the-shelf small model.
- **Self-drafting heads.** Medusa and EAGLE (below) attach lightweight heads to the target itself, eliminating the separate model entirely and getting the drafter's features "for free" from the target's own hidden states.
- **Retrieval / n-gram drafters.** No neural drafter at all: propose continuations from the prompt itself or a corpus via string matching (prompt lookup decoding). Astonishingly effective for tasks with high input-output overlap — summarization, code editing, RAG where the answer quotes the context. Acceptance on copied spans can approach 1.

!!! tip "Practitioner tip: match the drafter to the workload"
    Speculative decoding's payoff is workload-dependent. For *predictable* text — code, structured output, repetitive boilerplate, quoting from context — acceptance is high and speedups are large. For *high-entropy* creative text at high temperature, the target's distribution is genuinely diffuse, the drafter can't track it, and speedups shrink. Cheap **prompt-lookup** drafting (n-gram from the context) is often the highest-ROI option for copy-heavy workloads and costs essentially nothing to add.

## Medusa: Parallel Prediction Heads

Maintaining a *separate* draft model is operationally annoying: extra weights to load, a second model to serve, a second KV cache. **Medusa** (Cai et al., 2024) removes the separate model. It freezes the target and bolts on a small number of extra **decoding heads** — typically 4 or 5 — each a one- or two-layer MLP with a residual connection feeding into the target's existing unembedding (LM head). The base model predicts token $t{+}1$ as usual; Medusa head $k$ predicts token $t{+}k{+}1$ *directly from the same final hidden state* $h_t$.


{{fig:specdec-medusa-heads}}


So from one forward pass at position $t$, Medusa produces candidate distributions for the next *five* tokens at once. The catch: head $k$ predicts token $t{+}k{+}1$ **without conditioning on the intermediate tokens** $t{+}1,\dots,t{+}k$ — it only sees $h_t$. That is a strictly harder prediction problem, so the heads are individually weaker than true autoregressive drafting. Medusa compensates by proposing *many* candidates per head and verifying them all together with **tree attention** (next section): rather than betting on one continuation, it forms a tree of the top-few choices from each head and lets the target pick the best matching path.

Because the heads predict farther positions less reliably, Medusa pairs naturally with a relaxed acceptance criterion ("typical acceptance") that, at temperature > 0, accepts any token whose target probability exceeds a threshold rather than running exact rejection sampling. This trades a small, bounded deviation from the exact target distribution for higher acceptance and bigger speedups. (You can also run Medusa with the exact rule; it's a configuration choice.)

Training Medusa is cheap: freeze the backbone, train only the heads (Medusa-1) — sometimes a few GPU-hours — or jointly fine-tune backbone and heads for higher acceptance (Medusa-2). Since only the small heads are trainable, this is feasible even for very large targets and is closely related to the [PEFT](../05-posttraining-alignment/03-peft-lora-qlora.html) philosophy of adding tiny trainable modules to a frozen base.

```python
import torch
import torch.nn as nn

class MedusaHead(nn.Module):
    """One Medusa head: a residual MLP block reusing the base model's LM head.
    Predicts a token several positions ahead from the SAME hidden state."""
    def __init__(self, hidden_size, lm_head):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()
        self.lm_head = lm_head            # SHARED with the frozen base model
    def forward(self, h):                 # h: [B, T, hidden]
        h = h + self.act(self.linear(h))  # residual connection
        return self.lm_head(h)            # [B, T, vocab]

class MedusaModel(nn.Module):
    def __init__(self, base_model, num_heads=4):
        super().__init__()
        self.base = base_model            # frozen target
        hidden = base_model.config.hidden_size
        lm_head = base_model.get_output_embeddings()
        self.heads = nn.ModuleList(
            MedusaHead(hidden, lm_head) for _ in range(num_heads)
        )
    @torch.no_grad()
    def forward(self, input_ids):
        out = self.base(input_ids, output_hidden_states=True)
        h_last = out.hidden_states[-1]            # [B, T, hidden]
        base_logits = self.base.get_output_embeddings()(h_last)
        # head k predicts token t+k+2; base predicts t+1.
        head_logits = [head(h_last) for head in self.heads]
        return base_logits, head_logits          # use last position to draft
```

## The EAGLE Line: Feature-Level Autoregression

Medusa's weakness is that each head predicts in isolation from a single hidden state, ignoring the tokens drafted in between. **EAGLE** (Li et al., 2024 — *Extrapolation Algorithm for Greater Language-model Efficiency*) fixes this with a sharp insight: do the autoregressive drafting in **feature space** instead of token space, and condition on the tokens already drafted.

EAGLE adds one small **autoregression head** — a single transformer-style decoder layer — that runs its own little autoregressive loop, but over the target's *second-to-top-layer hidden features* rather than over tokens. Concretely, to draft the next feature it takes as input both the previous feature vector $f_{t}$ (from the target's penultimate layer) **and** the embedding of the previously sampled token. It predicts the next feature $\hat f_{t+1}$, runs it through the target's frozen LM head to get a token distribution, samples a token, embeds it, and feeds it back — a true autoregressive draft, but each step is one tiny layer instead of the whole target.


{{fig:specdec-eagle-feature-loop}}


Why is the feature-space target easier to model than the token-space target? Features are *continuous and lower-entropy* than the sampled token sequence. Much of a token's unpredictability is the irreducible sampling noise of softmax; the underlying feature trajectory is far more deterministic and so is more learnable by a tiny extrapolator. The result: EAGLE's drafts track the target much more closely than Medusa's, yielding substantially higher acceptance lengths at similar drafter cost. It remains a *lossless* drafter — verification still uses the exact accept/reject rule, so the output distribution is unchanged.

The line then iterates:

- **EAGLE-2** keeps the same drafter but makes the **draft tree dynamic and context-aware.** Instead of a fixed tree shape, it uses the drafter's own confidence (the running product of draft probabilities along each branch) to expand promising branches and prune unlikely ones. The acceptance probability of a draft token is well-approximated by the drafter's confidence, so EAGLE-2 spends its fixed token budget on the branches most likely to be accepted — more accepted tokens per target pass at the same verification cost.
- **EAGLE-3** removes a "feature-prediction" training constraint that limited how much extra training data helped, and **fuses features from multiple target layers** (low/mid/high) rather than just the penultimate one, giving the drafter a richer view. It trains directly to predict tokens (via the test-time-style multi-step objective) rather than to reconstruct a single feature, which scales better with data and pushes acceptance lengths higher still.

The throughline across the EAGLE family: keep the drafter tiny and *reuse the target's own representations*, do the cheap autoregression in feature space, and shape the verification tree dynamically.

## Tree Attention: Verifying Many Candidates at Once

Linear drafting commits to a single continuation $d_1,\dots,d_\gamma$; one early rejection wastes the rest. **Tree drafting** instead proposes *many* candidate continuations arranged as a token tree — for example the top-2 choices at depth 1, and for each of those the top-2 at depth 2, and so on — and verifies the **entire tree in a single target forward pass.** The target then accepts the longest path through the tree that survives the accept/reject test. This dramatically raises expected accepted length per pass and is essential to Medusa and EAGLE.

The trick that makes it cheap is **tree attention**: we flatten the tree into one sequence and use a custom attention mask so that each node attends only to its **ancestors** (and itself), never to nodes on sibling branches. That way, a single packed forward pass computes the correct hidden state for every node *as if* each path had been fed in separately.


{{fig:specdec-tree-attention}}


Each node must also carry the correct **position id** equal to its depth (not its index in the flattened sequence), so that positional encodings such as RoPE behave as if the path were contiguous (see [Positional Encodings](../02-transformer/05-positional-encoding.html)). Here is a compact construction of the mask and position ids from a parent-pointer tree:

```python
import torch

def build_tree_attn(parents):
    """
    parents: list where parents[i] is the index of node i's parent
             (parents[0] = -1 for the root). Nodes are in any order such
             that a parent appears before its children.
    Returns (mask, position_ids):
      mask[i, j] = True  iff node j is an ancestor of node i, or j == i.
      position_ids[i] = depth of node i (root = 0).
    """
    n = len(parents)
    mask = torch.zeros(n, n, dtype=torch.bool)
    depth = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        # Walk from node i up to the root, marking every ancestor as visible.
        j, d = i, 0
        while j != -1:
            mask[i, j] = True
            if j != i:
                d += 1
            j = parents[j]
        depth[i] = d
    return mask, depth

# Example tree: [root, A, B, A1, A2, B1, B2] from the diagram above.
parents = [-1, 0, 0, 1, 1, 2, 2]
mask, pos = build_tree_attn(parents)
print(pos.tolist())          # [0, 1, 1, 2, 2, 2, 2]  (depths)
# Feed `mask` as the attention mask and `pos` as position_ids to the target
# in ONE forward pass; then verify each root->leaf path with accept/reject.
```

After the single masked forward pass, you have the target's distribution at every node. You verify paths from the root: walk down, applying the accept/reject test at each edge, and take the longest accepted path; append the bonus token from the deepest accepted node. The verification cost is one target pass over $n$ tree nodes — and because decode is memory-bound, $n$ on the order of tens of nodes still costs roughly one token's worth of weight movement. The art is choosing the tree shape (depth, branching, total node budget) so the marginal node keeps paying for itself; EAGLE-2's dynamic, confidence-driven trees are the state of the art here.

## Lookahead Decoding: Drafting Without a Drafter

**Lookahead decoding** (Fu et al., 2024) is a different beast: it needs **no draft model and no extra trained heads at all**. It exploits the fact that the target's *own* parallel forward pass can be coaxed into generating and verifying n-grams simultaneously, by solving for fixed points of the autoregressive map using a **Jacobi-iteration** view of decoding.

The idea: think of decoding $N$ future tokens as solving a system of nonlinear equations $x_i = \arg\max p(\cdot \mid x_{<i})$ for $i = 1,\dots,N$. Ordinary decoding solves these one at a time (Gauss–Seidel order). **Jacobi decoding** instead guesses all $N$ at once and refines them in parallel: each forward pass updates every position based on the current guesses, and the sequence provably converges to the same greedy output in at most $N$ steps — but often far fewer, because many positions lock in early.

Lookahead decoding runs two things in every target forward pass, fused:

1. **The lookahead branch** performs several parallel Jacobi updates over a 2-D window (a *window size* $W$ of parallel positions over a *lookback* of $N$ steps), generating many candidate **n-grams** as a byproduct.
2. **The verification branch** checks promising n-grams collected so far (stored in an n-gram pool / cache) against the target — exactly like a draft — accepting any that match.

Both branches share **one** forward pass via a custom attention mask, much like tree attention. The payoff: the number of decoding steps to generate a sequence drops well below the sequence length, trading the GPU's spare parallel FLOPs (remember, decode is memory-bound, so FLOPs are nearly free) for fewer sequential steps. Crucially, with greedy decoding it produces *exactly* the same output as standard autoregressive decoding — it is lossless by construction, because it only ever accepts tokens that match the target's own argmax.

Trade-offs versus drafter-based methods:

- **Pro:** zero extra models, zero training, drop-in. Great when you can't or won't train a drafter.
- **Con:** the extra Jacobi/verification positions consume FLOPs and KV-cache slots, so at large batch sizes — where the GPU is already compute-saturated and *not* memory-bound — the spare FLOPs vanish and lookahead can *hurt*. This is the general rule for **all** speculative methods: they convert spare compute into fewer sequential steps, and they only help when there is spare compute, i.e., at small-to-moderate batch sizes where decode is memory-bound.

## Acceptance Rate, Speedup & When It Pays Off

Let us make the economics precise. Define, for a single-chain (non-tree) drafter of length $\gamma$:

- $\alpha$ = the per-token acceptance probability (assume i.i.d. across positions for a clean model).
- $c$ = the drafter's cost per token, as a fraction of one target forward pass.

The number of accepted draft tokens before the first rejection is a truncated geometric variable. The **expected number of tokens emitted per speculative step** (accepted drafts plus the one guaranteed corrected/bonus token) is

$$
\mathbb{E}[\text{tokens per step}] \;=\; \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}.
$$

This is the expected length of the accepted run plus the trailing token; it ranges from $1$ (when $\alpha=0$) up to $\gamma+1$ (when $\alpha\to 1$). The **cost per step** is one target pass plus $\gamma$ drafter passes: $1 + \gamma c$ target-equivalents. Hence the idealized **speedup** over plain decoding (which emits exactly one token per target pass) is

$$
\text{speedup} \;=\; \frac{\mathbb{E}[\text{tokens per step}]}{1 + \gamma c} \;=\; \frac{1 - \alpha^{\gamma+1}}{(1-\alpha)\,(1+\gamma c)} .
$$

Two regimes fall right out of this formula. As the drafter cost $c \to 0$ (e.g., cheap self-drafting heads), the denominator's drafter term vanishes and speedup $\to (1-\alpha^{\gamma+1})/(1-\alpha)$, which keeps growing with $\gamma$ toward $1/(1-\alpha)$. But with a real $c>0$, there is an **optimal $\gamma^\*$**: pushing $\gamma$ too high wastes drafter passes on tokens that will be rejected anyway. You differentiate or sweep to find it.

!!! example "Worked example: how big is the win?"
    Take a 70B target with a 1.5B drafter, so $c \approx 1.5/70 \approx 0.021$. Suppose on a coding workload the drafter achieves acceptance $\alpha = 0.8$, and we draft $\gamma = 5$ tokens per step.

    Expected tokens per step:
    $$\frac{1 - 0.8^{6}}{1 - 0.8} = \frac{1 - 0.262}{0.2} = \frac{0.738}{0.2} \approx 3.69 \text{ tokens.}$$

    Cost per step: $1 + 5(0.021) = 1.105$ target-equivalent passes.

    Speedup $\approx 3.69 / 1.105 \approx 3.3\times$. So we emit ~3.7 tokens for the wall-clock price of ~1.1 target passes — a 3.3× decode speedup, *with identical output distribution*.

    Now stress-test the assumption. If the workload is high-entropy chat at temperature 1.0 and acceptance drops to $\alpha = 0.4$, expected tokens per step is $(1 - 0.4^6)/0.6 = 0.9959/0.6 \approx 1.66$, and speedup $\approx 1.66/1.105 \approx 1.5\times$. Same machinery, very different payoff — acceptance rate is everything. And note the sweep on $\gamma$: at $\alpha=0.4$, raising $\gamma$ to 8 barely changes the numerator ($\approx 1.66$, since the geometric tail is tiny) while the denominator grows to $1.17$, so the speedup *falls*. The optimal $\gamma$ is workload-dependent.

A second, system-level caveat the formula hides: **batch size.** All these speedups assume the decode step is memory-bandwidth bound, so the extra verification FLOPs are free. At large batch sizes the GPU becomes **compute-bound** (you've already amortized weight loading across many requests), the verification tokens compete for real FLOPs, and the speedup collapses — speculation can even slow you down. Production engines therefore gate speculation on batch size and turn it off under heavy load. This is the same memory-vs-compute boundary analyzed in [The Roofline Model & Performance Engineering](../04-kernels-efficiency/01-roofline-performance.html) and [Inference Economics: Latency, Throughput & Cost](../07-inference-serving/12-inference-economics.html).

!!! warning "Common pitfall: tokenizer and distribution mismatch"
    For drafter-based speculation, the draft and target **must share an identical tokenizer**; a token ID has to mean the same string in both, or the accept test is incoherent and outputs garble. Two more subtle traps: (1) the drafter and target must use the **same temperature and sampling transform** at verification time, or the $p/q$ ratio is computed against the wrong distribution; and (2) "typical acceptance" / threshold-based acceptance (used by some Medusa configs) is *not* the exact rule — it deviates slightly from the target distribution by design. If you need bit-exact equivalence to the target's sampler, use the exact rejection rule, not a relaxed one.

!!! interview "Interview Corner"
    **Q:** Speculative decoding uses a small, inaccurate draft model, yet its output is provably identical in distribution to the large model's. How is that possible, and what *exactly* does the draft model's quality affect?

    **A:** The guarantee comes from **modified rejection sampling** at each position. A drafted token $x \sim q$ is accepted with probability $\min(1, p(x)/q(x))$; on rejection we resample from the residual $p_{\text{res}}(x) \propto (p(x)-q(x))_+$ and stop the chain. Summing the two ways a token $x$ can be emitted — drafted-and-accepted, giving $\min(p(x),q(x))$, plus rejected-then-residual, giving $(p(x)-q(x))_+$ — yields exactly $p(x)$ for every $x$, independent of $q$. So correctness holds for *any* drafter. The draft model's quality affects only **speed**: the expected acceptance probability is $\alpha = \sum_x \min(p(x), q(x)) = 1 - \mathrm{TV}(p,q)$, so a drafter closer to the target in total-variation distance accepts more tokens per verification pass and yields a larger speedup, per $(1-\alpha^{\gamma+1})/[(1-\alpha)(1+\gamma c)]$. A bad drafter just means $\alpha$ is low and you fall back toward 1× — never wrong, only slow.

    **Follow-up Q:** When would you *not* use it?

    **A:** When the decode step is already compute-bound — large batch sizes, or very small models — the extra verification FLOPs aren't free and speculation can slow you down. Also for very high-entropy sampling where acceptance is low, the win shrinks. Gate it on batch size and workload.

!!! key "Key Takeaways"
    - Decode is memory-bandwidth bound, so verifying $K$ tokens in one target pass costs about the same as one token — this is the entire reason speculative decoding works.
    - The **accept/reject rule** — accept $x\sim q$ with prob $\min(1,p(x)/q(x))$, else resample from $(p-q)_+$ — makes emitted tokens *exactly* distributed as the target $p$, for **any** drafter. Correctness is free; the drafter only buys speed.
    - Expected acceptance is $\alpha = \sum_x \min(p,q) = 1 - \mathrm{TV}(p,q)$: a drafter close to the target in total-variation distance accepts more.
    - **Separate draft models** (10–50× smaller, same tokenizer/family, ideally distilled from the target) are the classic recipe; **prompt-lookup/n-gram** drafting is near-free and excellent for copy-heavy workloads.
    - **Medusa** adds parallel prediction heads to the frozen target (cheap to train, but each head predicts in isolation); **EAGLE/EAGLE-2/3** do autoregression in **feature space** conditioned on drafted tokens, achieving much higher acceptance, with dynamic confidence-driven draft trees.
    - **Tree attention** verifies many candidate continuations in one pass via an ancestor-only mask and depth-based position ids, raising accepted tokens per step.
    - **Lookahead decoding** needs no drafter or training — it uses Jacobi iteration to generate and verify n-grams in the target's own parallel passes; lossless for greedy decoding.
    - Speedup $= (1-\alpha^{\gamma+1})/[(1-\alpha)(1+\gamma c)]$: there is an optimal draft length $\gamma^\*$, and all methods only help while decode is **memory-bound** (small/moderate batch). Gate speculation on batch size.

!!! sota "State of the Art & Resources (2026)"
    Speculative decoding is now a production staple in every major serving engine (vLLM, TensorRT-LLM, SGLang). The EAGLE family — especially EAGLE-3 (NeurIPS 2025) — represents the current speed frontier for self-speculative methods, with 3–6× latency reductions on popular models.

    **Foundational work**

    - [Leviathan, Kalman & Matias, *Fast Inference from Transformers via Speculative Decoding* (2022)](https://arxiv.org/abs/2211.17192) — the original draft-then-verify algorithm with the accept/reject proof; 2–3× speedup on T5-XXL.
    - [Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling* (2023)](https://arxiv.org/abs/2302.01318) — concurrent DeepMind formulation using modified rejection sampling; benchmarked on Chinchilla-70B.

    **Recent advances (2023–2026)**

    - [Cai et al., *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads* (2024)](https://arxiv.org/abs/2401.10774) — parallel prediction heads bolted onto a frozen target; introduces tree attention and the typical-acceptance criterion.
    - [Li et al., *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty* (2024)](https://arxiv.org/abs/2401.15077) — autoregression in feature (penultimate-layer) space; 2.7–3.5× speedup by matching the target's own representations.
    - [Li et al., *EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees* (2024)](https://arxiv.org/abs/2406.16858) — confidence-driven dynamic draft trees that allocate the token budget to the most-likely branches; 20–40% faster than EAGLE-1.
    - [Li et al., *EAGLE-3: Scaling up Inference Acceleration via Training-Time Test* (2025)](https://arxiv.org/abs/2503.01840) — multi-layer feature fusion and a training-time test objective; 3.3–6.5× speedup and now integrated into vLLM/SGLang/TRT-LLM.
    - [Fu et al., *Break the Sequential Dependency of LLM Inference Using Lookahead Decoding* (2024)](https://arxiv.org/abs/2402.02057) — Jacobi-iteration drafting with no extra model or training; lossless for greedy decoding.

    **Open-source & tools**

    - [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE) — official reference implementation of EAGLE-1, EAGLE-2, and EAGLE-3 (ICML'24, EMNLP'24, NeurIPS'25).
    - [FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa) — Medusa training and inference code; includes recipes for both Medusa-1 (heads only) and Medusa-2 (joint fine-tune).
    - [vLLM speculative decoding docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/) — production guide covering draft models, EAGLE, n-gram, and the `--speculative-config` API.

    **Go deeper**

    - [Xia et al., *Unlocking Efficiency in LLM Inference: A Comprehensive Survey of Speculative Decoding* (2024)](https://arxiv.org/abs/2401.07851) — systematic taxonomy of drafter types, verification strategies, and Spec-Bench comparisons across methods.

## Further reading

- Leviathan, Kalman & Matias, *Fast Inference from Transformers via Speculative Decoding* (Google, 2023) — the accept/reject derivation and draft-then-verify framing.
- Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling* (DeepMind, 2023) — concurrent formulation with the modified rejection-sampling proof.
- Cai et al., *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads* (2024) — parallel heads, tree attention, typical acceptance.
- Li et al., *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty* (2024), plus the **EAGLE-2** (dynamic draft trees) and **EAGLE-3** (multi-layer feature fusion, training-time test) follow-ups.
- Fu et al., *Break the Sequential Dependency of LLM Inference Using Lookahead Decoding* (2024) — Jacobi-iteration drafting without a draft model.
- Stern, Shazeer & Uszkoreit, *Blockwise Parallel Decoding for Deep Autoregressive Models* (2018) — an early precursor to multi-token prediction and verification.
- The **vLLM** and **TensorRT-LLM** repositories — production implementations of draft-model, Medusa, and EAGLE speculation; see [vLLM: Architecture, PagedAttention & Internals](../07-inference-serving/03-vllm-internals.html) and [TensorRT-LLM, TGI & Other Serving Stacks](../07-inference-serving/05-trtllm-tgi-stacks.html).
