# 2.12 Diffusion & Non-Autoregressive Language Models

Every model we have built so far in this book generates text the same way: one token at a time, strictly left to right, each token conditioned on all the tokens before it. This is the **autoregressive (AR)** factorization of the joint distribution over a sequence $x = (x_1, \dots, x_L)$:

$$
p_\theta(x) = \prod_{t=1}^{L} p_\theta(x_t \mid x_{<t})
$$

It is a beautiful, exact factorization — and it is also the source of a stubborn latency wall. Generating a 1000-token response requires **1000 sequential forward passes**, each of which must finish before the next can begin, because $x_t$ is an *input* to the computation of $x_{t+1}$. No amount of GPU parallelism removes that serial dependency. The hardware is busy but starved: at decode time you are matrix-vector limited, reading the entire model's weights from memory to produce a single token (see [The Anatomy of LLM Inference: Prefill, Decode & The KV Cache](../07-inference-serving/01-anatomy-inference.html) and [Inference Economics: Latency, Throughput & Cost](../07-inference-serving/12-inference-economics.html)).

This chapter is about a different bet: **non-autoregressive (NAR)** language models, and in particular the **discrete diffusion / masked diffusion** family that has, since roughly 2024–2025, produced the first genuinely competitive non-AR LLMs — LLaDA, Dream, and the commercial Mercury system from Inception Labs. Instead of emitting tokens left-to-right, these models start from a fully masked sequence and **iteratively denoise all positions in parallel**, refining the whole sequence over a small number of steps. The promise is a fundamentally different point on the latency–quality curve: tokens-per-second numbers that can be several times higher than AR models of comparable size, plus the ability to reason *bidirectionally* and fill text in any order.

We will develop the absorbing-state diffusion objective from first principles, contrast it sharply with next-token prediction (covered in [The Pretraining Objective & Loss](../03-pretraining/03-pretraining-objective.html)), implement a tiny masked-diffusion sampler from scratch, and then engineer our way toward the production tricks — semi-autoregressive **block diffusion**, KV-cache reuse, remasking schedules, and flexible-length generation — that make these systems practical. The mathematical machinery overlaps with continuous image diffusion ([Diffusion Models & Generative Modeling (Breadth)](../10-multimodal-and-arch/04-diffusion-generative.html)), but the discrete-token setting has its own elegant simplifications that are worth seeing carefully.

---

## Why Non-Autoregressive? The Latency Argument

Let us quantify the wall we are trying to climb over. In the decode phase, an AR transformer produces one token per forward pass. The wall-clock time to generate $L$ tokens is

$$
T_{\text{AR}} \approx L \cdot t_{\text{step}}
$$

where $t_{\text{step}}$ is the per-step latency, dominated on modern GPUs by the time to stream the model weights from HBM into the compute units. For a dense model with $P$ parameters at precision $b$ bytes/param and memory bandwidth $\beta$ bytes/s, the floor is roughly $t_{\text{step}} \gtrsim P b / \beta$ — and crucially this floor is *independent of how clever your kernel is*, because decode is memory-bandwidth bound, not compute bound.

A non-autoregressive diffusion model instead runs $N$ denoising steps, where **each step is one forward pass over the entire length-$L$ sequence in parallel**, and ideally $N \ll L$:

$$
T_{\text{NAR}} \approx N \cdot t_{\text{step}}'
$$

Here $t_{\text{step}}'$ is somewhat larger than $t_{\text{step}}$ because each step processes all $L$ positions at once (a *compute*-bound batched operation that uses the hardware far more efficiently), but the win is that $N$ can be 16, 32, or 64 instead of $L = 1000$. If a diffusion model produces acceptable text in $N = 64$ steps for a 512-token output, it has done $64$ sequential forward passes instead of $512$ — an $8\times$ reduction in *serial depth*, even before accounting for the fact that each diffusion forward pass keeps the matrix multiply units saturated.

!!! example "Worked example: serial depth and tokens/sec"

    Suppose a 7B-parameter model in bf16 ($b = 2$ bytes) runs on a GPU with $\beta \approx 3.0$ TB/s of HBM bandwidth. The decode-step memory floor is

    $$
    t_{\text{step}} \gtrsim \frac{P \cdot b}{\beta} = \frac{7 \times 10^9 \times 2}{3.0 \times 10^{12}} \approx 4.7\ \text{ms}.
    $$

    - **Autoregressive**, 512 output tokens: $T \approx 512 \times 4.7\ \text{ms} \approx 2.4\ \text{s}$, i.e. about **213 tokens/s** for a single sequence.
    - **Diffusion**, 512 tokens in $N = 32$ steps: each step reads the weights once *and* processes all 512 positions, so it is compute-heavier; say $t_{\text{step}}' \approx 12\ \text{ms}$. Then $T \approx 32 \times 12\ \text{ms} \approx 0.38\ \text{s}$ — about **1340 tokens/s** for the same single sequence.

    The diffusion model is ~6× faster *for one sequence* here, precisely because its serial depth (32) is far below the output length (512). The catch, which we will keep returning to, is whether 32 steps is enough to hit the quality the AR model gets "for free" by conditioning each token on exact predecessors. These magnitudes are illustrative; real numbers depend heavily on batch size, where AR throughput catches up because continuous batching ([Continuous Batching & Request Scheduling](../07-inference-serving/02-continuous-batching.html)) fills the compute units across many concurrent requests.

There is a second, qualitatively different motivation. AR models are **causally blind to the future**: when predicting $x_t$ they cannot see $x_{>t}$. This makes some tasks awkward — infilling, editing, satisfying global constraints, and the famous **reversal curse** (a model trained on "A is B" often fails to answer "what is B? → A"). A bidirectional denoiser conditions every position on every other position at every step, which structurally sidesteps these issues. Non-AR models are not just "AR but faster"; they have a different inductive bias.

---

## Discrete & Masked Diffusion From First Principles

Continuous diffusion (DDPM, score-based models) gradually adds Gaussian noise to a real-valued vector and learns to reverse it. Text is discrete — there is no meaningful "add a little Gaussian noise to the token `cat`." We need a corruption process that lives on a finite vocabulary. The family that has proven both simple and powerful is **absorbing-state (masked) diffusion**.

### The forward (corruption) process

Fix a special absorbing token `[MASK]`, with index $m$, that is *not* a normal vocabulary token. The forward process is defined on a continuous time variable $t \in [0, 1]$. At $t = 0$ the sequence is clean data; at $t = 1$ it is entirely `[MASK]`. For each token independently, define a monotone masking schedule $\alpha_t$ with $\alpha_0 = 1$ and $\alpha_1 = 0$. Conditioned on the clean token $x_0^i$ at position $i$, the corrupted token at time $t$ is

$$
q(x_t^i \mid x_0^i) =
\begin{cases}
\alpha_t & \text{keep the original token } x_0^i,\\[4pt]
1 - \alpha_t & \text{replace with } \texttt{[MASK]}.
\end{cases}
$$

So $\alpha_t$ is literally the probability that a given token survives unmasked at time $t$. The absorbing property is what makes this clean: once a token becomes `[MASK]` it stays `[MASK]` for all later times — the only transition is data $\to$ mask, never mask $\to$ data and never data $\to$ different-data. This is far simpler than the general discrete-diffusion transition matrices of Austin et al.'s D3PM, where any token can flip to any other.

### The reverse (denoising) process and the objective

The model is a bidirectional transformer (an encoder-style stack with **no causal mask** — see [Architecture Variants: Encoder-Decoder, Decoder-Only & Prefix-LM](../02-transformer/08-architecture-variants.html)) that takes a partially-masked sequence $x_t$ and predicts a distribution over the *clean* token at every masked position simultaneously. Call this $p_\theta(x_0^i \mid x_t)$.

The remarkable simplification of absorbing diffusion is that, after the full variational derivation, the training loss collapses to a **weighted cross-entropy on the masked positions only** — essentially a continuous-time generalization of BERT's masked language modeling. For a clean sequence $x_0$, sample a time $t \sim \mathcal{U}(0,1)$, mask each token independently with probability $1 - \alpha_t$ to get $x_t$, and minimize

$$
\mathcal{L}(\theta) = \mathbb{E}_{t \sim \mathcal{U}(0,1)} \; \mathbb{E}_{x_t \sim q(\cdot \mid x_0)} \left[ \frac{1}{1 - \alpha_t} \sum_{i : x_t^i = \texttt{[MASK]}} -\log p_\theta\!\left(x_0^i \mid x_t\right) \right].
$$

Read this carefully, because three ideas are packed into it:

1. **Only masked positions contribute.** The sum runs over $i$ where $x_t^i$ is `[MASK]`. Unmasked positions provide *context* but no loss — the model is rewarded only for recovering what was hidden.
2. **The weight $1/(1-\alpha_t)$ corrects for the masking rate.** When $t$ is near 1, almost everything is masked, $1-\alpha_t \approx 1$, and the model must reconstruct from almost nothing — a hard denoising problem. When $t$ is near 0, only a few tokens are masked and the weight is large, so each rare masked position counts heavily. This weighting is exactly what makes the masked-LM loss a valid (upper bound on the) negative log-likelihood, i.e. a true generative objective, not just a representation-learning trick. The MDLM and RADD analyses (Sahoo et al.; Ou et al., 2024) show this weighted form is a tight Evidence Lower Bound.
3. **There is no left-to-right ordering.** Unlike next-token prediction, where the chain rule dictates the factorization order, here the model learns to fill *any* subset of positions given *any* other subset. This is the source of the bidirectional advantage.

Contrast with next-token prediction, where the loss is $-\sum_t \log p_\theta(x_t \mid x_{<t})$ over *every* position with a strict causal mask. AR predicts the next token from a left context; masked diffusion predicts a random subset from a bidirectional context. AR gives you an exact autoregressive likelihood and trivially correct sampling order; diffusion gives you parallelism and bidirectionality at the cost of an approximate, order-agnostic factorization.

!!! note "Aside: why this is not just BERT"

    BERT also predicts masked tokens bidirectionally, so why can't we just sample from BERT? BERT masks a *fixed* small fraction (~15%) of tokens and is never trained to generate from a fully-masked sequence, nor to handle the high-masking-rate regime. It also lacks the time-conditioning and the $1/(1-\alpha_t)$ weighting that turn the masked-LM loss into a proper likelihood bound across *all* masking rates. Masked diffusion is "BERT trained at every masking rate from 0% to 100%, with the correct loss weighting, then sampled iteratively." That generalization is exactly what lets it generate coherent long text, which BERT cannot.

### Time conditioning, or the lack of it

Continuous diffusion models almost always feed the timestep $t$ into the network (via sinusoidal embeddings added to the input). A pleasant surprise in the discrete-masked setting is that **$t$ is largely redundant**: the *number of `[MASK]` tokens in the input already tells the model roughly where it is in the denoising process*. Several strong masked-diffusion LLMs (including LLaDA) drop explicit time conditioning entirely and rely on the mask count as an implicit clock. This is one fewer thing to get right and lets the architecture stay a vanilla bidirectional transformer.

---

## Iterative Parallel Denoising: A Tiny Sampler

Time to make this concrete. The inference loop for masked diffusion is conceptually simple: start with everything masked, and repeatedly (a) predict all masked positions in parallel, (b) *commit* some of those predictions by unmasking them, and (c) leave the rest masked for the next round. The art is entirely in **which positions to commit each step** — the *remasking schedule*.

Here is a complete, from-scratch sampler. The "model" is abstracted behind a `denoiser` callable so the loop is crystal clear; in practice it is a bidirectional transformer returning logits of shape `(L, V)`.

```python
import torch
import torch.nn.functional as F

MASK_ID = 0  # reserve index 0 in the vocab for the [MASK] absorbing token

@torch.no_grad()
def masked_diffusion_sample(
    denoiser,            # callable: (LongTensor[L]) -> FloatTensor[L, V] logits
    L: int,              # sequence length to generate
    num_steps: int,      # number of denoising iterations N (the serial depth)
    vocab_size: int,
    temperature: float = 0.0,   # 0.0 = greedy/argmax per position
    prompt: torch.Tensor = None,   # optional LongTensor of clamped prefix tokens
    device: str = "cpu",
):
    """
    Absorbing-state masked diffusion sampler.

    Strategy ("confidence-based remasking", a la LLaDA's low-confidence remasking):
    at each step we predict ALL masked positions, but only *keep* (unmask) the
    most confident predictions, sized so that the fraction of masked tokens
    follows a linear schedule from 1.0 down to 0.0 over num_steps. Less confident
    positions are returned to [MASK] and revisited in later steps.
    """
    # 1. Start fully masked.
    x = torch.full((L,), MASK_ID, dtype=torch.long, device=device)

    # 2. Clamp the prompt (these positions are never masked and never predicted).
    is_prompt = torch.zeros(L, dtype=torch.bool, device=device)
    if prompt is not None:
        x[: prompt.numel()] = prompt.to(device)
        is_prompt[: prompt.numel()] = True

    # 3. Denoising schedule: target number of STILL-masked tokens after step k.
    #    Linear from "all generatable positions masked" down to 0.
    n_gen = int((~is_prompt).sum().item())   # positions we actually generate
    # masked_target[k] = how many gen-positions remain masked AFTER step k+1
    masked_target = [
        round(n_gen * (1.0 - (k + 1) / num_steps)) for k in range(num_steps)
    ]

    for step in range(num_steps):
        masked = (x == MASK_ID) & (~is_prompt)   # positions still to fill
        if masked.sum() == 0:
            break

        logits = denoiser(x)                     # (L, V) — full bidirectional pass
        logits[:, MASK_ID] = -float("inf")       # never predict [MASK] itself

        if temperature > 0.0:
            probs = F.softmax(logits / temperature, dim=-1)
            pred = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (L,)
            conf = probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1)     # (L,)
        else:
            probs = F.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)       # greedy + its confidence

        # Candidate fill: tentatively set every masked position to its prediction.
        x_candidate = x.clone()
        x_candidate[masked] = pred[masked]

        # Decide how many of the currently-masked positions to KEEP unmasked.
        n_keep = int(masked.sum().item()) - masked_target[step]
        n_keep = max(0, n_keep)

        # Rank masked positions by confidence; keep the top-n_keep, remask the rest.
        conf_masked = conf.clone()
        conf_masked[~masked] = -float("inf")     # only compete among masked positions
        if n_keep > 0:
            keep_idx = torch.topk(conf_masked, n_keep).indices
            new_x = x.clone()
            new_x[masked] = MASK_ID              # provisionally remask all
            new_x[keep_idx] = x_candidate[keep_idx]  # commit the confident ones
            x = new_x
        # if n_keep == 0 we commit nothing this step (rare; only at the very start)

    # Final cleanup: fill any leftover masks greedily (last step should handle this).
    leftover = (x == MASK_ID) & (~is_prompt)
    if leftover.any():
        logits = denoiser(x)
        logits[:, MASK_ID] = -float("inf")
        x[leftover] = logits[leftover].argmax(dim=-1)

    return x
```

The whole behavior of the model lives in **how `n_keep` is chosen each step**. A few canonical schedules:

- **Random remasking.** Pick the positions to keep uniformly at random. Simple, unbiased, but wastes steps committing to low-confidence guesses early.
- **Confidence-based (low-confidence remasking).** Keep the highest-confidence predictions, return the rest to `[MASK]` (what the code above does). This is LLaDA's default and works well because the model "locks in" the tokens it is sure about and keeps reconsidering the uncertain ones with more context each round.
- **Top-$k$ / greedy decoding orders.** Variants that commit a fixed number of tokens per step, or use entropy rather than max-prob as the confidence measure.

The crucial subtlety is that **already-committed tokens become context for the next step.** When the model unmasks "The capital of France is" it makes the later positions far easier to predict — the bidirectional pass now sees both left and right context flowing into each remaining mask. This is the iterative-refinement engine: each round of commitments sharpens the conditional distribution for everything still masked.

!!! warning "Common pitfall: the conditional-independence trap"

    In a single denoising step the model predicts every masked position **independently** — it factorizes $p_\theta(x_0 \mid x_t) = \prod_i p_\theta(x_0^i \mid x_t)$. That independence is *wrong* for natural language: the joint over masked tokens is highly correlated. If you tried to fill *all* masks in one step, you would get incoherent output where each position is individually plausible but jointly contradictory (e.g. subject and verb disagreeing). Iterative denoising with remasking is precisely the fix: by committing only a few high-confidence tokens per step and re-conditioning, you recover the correlations the single-step factorization throws away. **More steps trade compute for coherence.** This is the diffusion-LM analog of why you can't sample all pixels of an image at once.

### How many steps do you actually need?

This is the central quality–latency knob. With $N = L$ steps and a one-token-per-step schedule, masked diffusion essentially reduces to an (order-flexible) autoregressive model and matches AR quality — but you have thrown away the speed advantage. With $N$ very small (say 8), you get blazing speed but the conditional-independence error bites and quality drops. Production systems live in the interesting middle: enough steps that each commits a small block of tokens, few enough that serial depth stays well below $L$. The empirical finding across LLaDA and Dream is that you can often reach AR-comparable quality at $N$ on the order of $L/4$ to $L/2$, and trade further quality for speed below that.

---

## Semi-Autoregressive Block Diffusion

Pure diffusion over a fixed-length block has two weaknesses that show up immediately in production: (1) it commits to a **fixed sequence length** up front, which is unnatural for open-ended generation, and (2) it cannot reuse a KV cache across steps, because every position is recomputed every step. **Block diffusion** (Arriola et al., 2025) is the hybrid that fixes both by *interpolating between autoregressive and diffusion modeling*.

The idea: split the sequence into contiguous **blocks** of $B$ tokens each. Generate the blocks **autoregressively** (left to right, one block after another), but generate the tokens *within* each block by **diffusion** (parallel denoising). Formally, with blocks $x^{(1)}, x^{(2)}, \dots$,

$$
p_\theta(x) = \prod_{b} p_\theta^{\text{diff}}\!\left(x^{(b)} \mid x^{(1)}, \dots, x^{(b-1)}\right),
$$

where each block factor is itself a small masked-diffusion model conditioned on all *previously finalized* blocks. Set $B = 1$ and you recover a pure autoregressive model; set $B = L$ (one block) and you recover pure diffusion. Block diffusion is the dial between them.

```text
   Block 1            Block 2            Block 3
 [denoise B tok] -> [denoise B tok] -> [denoise B tok] -> ...
   (parallel)         (parallel)         (parallel)
   |__________________|__________________|
        previous blocks are FINALIZED and cached in the KV cache;
        the current block attends to them causally (block-causal mask)
```

This buys three things at once:

- **KV-cache reuse.** Because earlier blocks are finalized before the current block starts, their keys and values are *fixed*. We compute them once and cache them, exactly like AR decoding ([Prefix Caching & KV-Cache Reuse](../07-inference-serving/07-prefix-caching.html), [PagedAttention & KV-Cache Memory Management](../04-kernels-efficiency/06-paged-attention-kv.html)). Within-block denoising recomputes only the $B$ positions of the current block — a huge saving versus pure diffusion, which recomputes all $L$ positions every step.
- **Flexible / arbitrary length.** Generation proceeds block by block until an end-of-sequence condition fires, so you are no longer locked into a length chosen before you start. You can keep emitting blocks until the model produces `[EOS]`, just like AR.
- **Parallelism where it pays.** Inside a block you get the diffusion speedup (parallel denoising of $B$ tokens in a few steps); across blocks you get the AR structure that captures long-range left-to-right dependencies and gives you the cache.

The attention mask is the key implementation detail. Within a block, attention is **bidirectional** (every position sees every other position in the block — that is what gives diffusion its power). Across blocks, attention is **causal at block granularity**: tokens in block $b$ can attend to all tokens of blocks $1..b$ but not $b+1..$. This **block-causal** mask is what makes the KV cache valid.

```python
def block_causal_mask(L: int, block_size: int) -> torch.Tensor:
    """
    Build an additive attention mask for block diffusion.
      - bidirectional WITHIN a block (no causal restriction)
      - causal ACROSS blocks (block b sees blocks 1..b only)
    Returns (L, L) with 0.0 where attention is allowed, -inf where blocked.
    """
    idx = torch.arange(L)
    block_of = idx // block_size                       # (L,) block index per position
    # allowed if key's block <= query's block
    allowed = block_of[None, :] <= block_of[:, None]   # (L_query, L_key) boolean
    mask = torch.zeros(L, L)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask
```

Block diffusion is the architecture that most directly maps onto the existing high-performance AR serving stack: you keep continuous batching, paged KV cache, and prefix caching, and you simply swap the inner decode kernel from "one token per step" to "a block of tokens per few diffusion steps." This is a large part of why the first commercially viable diffusion LLM, Mercury, is built on this semi-autoregressive structure rather than pure parallel diffusion.

!!! tip "Practitioner tip: block size is your latency–quality dial"

    In a block-diffusion deployment you have two coupled knobs: block size $B$ and per-block denoising steps $N_B$. Small $B$ with few steps behaves like AR (high quality, lower parallelism). Large $B$ exposes more parallelism per block but raises the conditional-independence burden, demanding more denoising steps to stay coherent. A common sweet spot is a moderate block (e.g. 16–32 tokens) with a handful of denoising steps per block, so that each block commits a few tokens at a time while the cache amortizes the cost of all earlier blocks. Tune $B$ and $N_B$ jointly against your target time-to-first-token and inter-token latency.

---

## Production Systems & Inference Economics

We now have the pieces — masked-diffusion objective, iterative denoising, block structure, KV reuse — to understand the real systems and their economics.

### The landscape

- **LLaDA** (Nie et al., 2025) is the proof of concept at scale: an 8B-parameter masked-diffusion LLM trained from scratch with the absorbing objective, with an instruction-tuned chat variant. Its headline result is that a pure (non-block) masked-diffusion model can be *competitive with similarly-sized autoregressive LLaMA-class models* on standard benchmarks — the first time a from-scratch diffusion LLM closed most of that gap. It also concretely demonstrated the bidirectional/reversal advantage (below).
- **Dream** (2025) is a 7B masked-diffusion LLM that, rather than training from scratch, *initializes from an existing autoregressive checkpoint* and adapts it to the diffusion objective — a clever way to reuse the enormous compute already spent on AR pretraining. It uses context-adaptive token-level noise and confidence-based decoding orders to push quality.
- **Mercury** (Inception Labs, 2025) is the commercial diffusion-LLM system, marketed around coding (Mercury Coder) and built on a block-diffusion-style architecture for flexible length and KV reuse. Its public pitch is order-of-magnitude higher tokens/sec than comparable autoregressive code models on the same hardware, by exploiting parallel decode. (Treat any specific throughput figure as vendor-reported; the *mechanism* — parallel denoising of blocks with cache reuse — is the durable takeaway.)

### Where the speed actually comes from — and where it doesn't

The diffusion speedup is a **serial-depth** win, not a FLOPs win. A diffusion model often does *more* total floating-point work than an AR model (it recomputes positions across steps), but it organizes that work into fewer sequential, more parallel steps. This has two important consequences for the economics ([Inference Economics: Latency, Throughput & Cost](../07-inference-serving/12-inference-economics.html)):

1. **Single-stream latency: diffusion wins big.** When you have one request and want the answer fast — interactive coding, low-batch agentic loops — cutting serial depth from $L$ to $N$ is a direct latency reduction. This is diffusion's home turf.
2. **High-throughput, large-batch serving: the gap narrows.** AR decoding is memory-bandwidth bound at small batch but becomes compute-efficient at large batch, because continuous batching packs many requests' single-token steps into one fat matrix multiply that saturates the GPU. Diffusion already saturates the GPU per step, so it has less headroom to gain from batching. At very high concurrency, an AR system's aggregate tokens/sec can rival or exceed a diffusion system's, because the diffusion model's extra FLOPs per token now compete for the same saturated compute. **Diffusion's advantage is largest at low-to-moderate batch and long single outputs.**

The honest summary: diffusion LLMs are not a free lunch that beats AR everywhere. They occupy a different region of the latency–throughput–quality surface — superb single-stream latency and bidirectional capability, in exchange for an approximate factorization, more total compute per token, and (today) a smaller ecosystem of training data, tooling, and post-training recipes than the mature AR stack.

!!! example "Worked example: the FLOPs–latency trade in block diffusion"

    Take a block-diffusion model, $L = 512$ output tokens, block size $B = 32$ (so 16 blocks), and $N_B = 8$ denoising steps per block. Count *forward passes over a block's worth of positions*:

    - Total block-denoising passes $= 16 \text{ blocks} \times 8 \text{ steps} = 128$ passes, each over 32 positions.
    - The equivalent AR model needs $512$ sequential single-token passes.

    Serial depth drops from **512 → 128** (a 4× reduction in the number of dependent steps), which is roughly the single-stream latency win. But total positions processed by the denoiser $\approx 128 \times 32 = 4096$ versus AR's $512$ — about **8× more position-evaluations**. With KV-cache reuse the earlier blocks aren't recomputed, so the *attention* cost is bounded, but the within-block MLP work is genuinely ~8× larger. That extra compute is the price of the parallelism: cheap when the GPU was idle (low batch), expensive when it was already full (high batch). This is the whole economic story in one example.

### Sampling controls carry over — mostly

Temperature, top-$k$, and top-$p$ ([Sampling Strategies & Decoding Algorithms](../07-inference-serving/09-sampling-decoding.html)) all apply *per position* inside a denoising step, exactly as in AR. What is new is the **remasking schedule** as a first-class decoding hyperparameter: how many tokens to commit per step, and by what confidence criterion. Constrained/structured generation ([Structured & Constrained Generation](../07-inference-serving/10-structured-generation.html)) is actually *more* natural in some respects — because the model sees the whole sequence, you can pin known tokens (a closing brace, a required JSON key) as permanently-unmasked context and let denoising fill around them, which is exactly the infilling capability AR models lack.

---

## The Bidirectional Advantage: Infilling, Constraints & the Reversal Curse

The most intellectually interesting reason to care about non-AR LLMs is not speed — it is **bidirectionality**. Because a masked-diffusion model conditions every position on every other position, it can do things that are structurally hard for a left-to-right model.

**Infilling and editing.** Given a document with a hole in the middle, an AR model must either be specially trained with fill-in-the-middle objectives or awkwardly re-prompted. A diffusion model treats infilling as its *native* operation: clamp the known prefix and suffix as unmasked context, mask the hole, and denoise. The hole is filled conditioned on *both* sides, which is exactly what coherent editing requires.

**Global constraints.** Tasks like "write a sentence that ends with this exact word" or "produce code with this signature and this return statement" require coordinating the beginning and end of the output. Left-to-right generation can paint itself into a corner; bidirectional denoising can place the constrained tokens first and grow the rest around them.

**The reversal curse.** AR models trained on "A is B" famously struggle to answer the reversed query "who/what is B?" because the training gradient only ever flowed in the A→B direction. A bidirectional diffusion model is trained to reconstruct *any* masked subset from any context, so it sees both directions during training and is markedly more robust to reversal. LLaDA's authors highlighted exactly this: on reversal-style tasks the masked-diffusion model can outperform a same-scale autoregressive baseline, sometimes by a wide margin, precisely because its objective is not directionally biased.

```python
# Infilling with the same sampler: clamp prefix AND suffix, denoise the middle.
def infill_example(denoiser, prefix_ids, suffix_ids, hole_len, vocab_size):
    L = len(prefix_ids) + hole_len + len(suffix_ids)
    x = torch.full((L,), MASK_ID, dtype=torch.long)
    is_clamped = torch.zeros(L, dtype=torch.bool)

    # left context
    x[: len(prefix_ids)] = torch.tensor(prefix_ids)
    is_clamped[: len(prefix_ids)] = True
    # right context (note: this is the future, which AR cannot use!)
    x[len(prefix_ids) + hole_len :] = torch.tensor(suffix_ids)
    is_clamped[len(prefix_ids) + hole_len :] = True

    # Reuse the diffusion loop, but treat ALL clamped positions like a "prompt"
    # so they are never masked and never overwritten. The hole is denoised
    # conditioned on BOTH sides simultaneously.
    return masked_diffusion_sample(
        denoiser, L=L, num_steps=hole_len, vocab_size=vocab_size,
        prompt=None,
    ), is_clamped  # (in practice, pass is_clamped into the sampler as is_prompt)
```

This is the capability that AR models have to *bolt on* and diffusion models get *for free*, and it is the strongest argument that non-AR LMs are a genuinely different tool rather than a faster clone.

!!! interview "Interview Corner"

    **Q:** A masked-diffusion LM predicts all masked positions in parallel within a single denoising step. If that step models the masked tokens as conditionally independent given the context, how can the final generated text be coherent — and what determines the quality/speed trade-off?

    **A:** The single-step conditional-independence assumption *is* wrong for language — the joint over masked tokens is strongly correlated — but the sampler never relies on filling everything in one step. It commits only a subset of high-confidence predictions per step (confidence-based remasking) and **re-conditions** the next step's predictions on those newly committed tokens. Iterating $N$ steps recovers the inter-token correlations that any single step's factorization discards, because each committed token becomes bidirectional context that sharpens the conditionals for the still-masked positions. The trade-off is set by $N$: at $N = L$ (one token per step) it degenerates to an order-flexible autoregressive model and matches AR coherence but loses the speed; at small $N$ each step commits many tokens, the independence error grows, and quality drops. Production systems pick the smallest $N$ (often roughly $L/4$ to $L/2$, or a few steps per block in block diffusion) that holds quality, which is where the tokens/sec win comes from — it's a reduction in *serial depth*, not in total FLOPs. The key insight to land is that the win is parallelism in the *sampling order*, paid for with extra compute and an approximate factorization, and that bidirectionality is a separate, structural benefit (infilling, reversal-robustness) independent of the speed argument.

---

## Practicalities, Limits & When to Reach for This

A few engineering realities to keep the picture honest:

- **Likelihood is a bound, not exact.** AR models give you an exact log-likelihood (useful for perplexity, ranking, watermarking). Masked diffusion gives a variational *bound*; reported perplexities use the ELBO and are not directly comparable to AR perplexity. Be careful when comparing benchmark tables across the two paradigms.
- **Post-training is younger.** The instruction-tuning, RLHF, and preference-optimization stack (Part V) was built around AR generation. Adapting SFT and RL — especially methods that need per-token log-probs and a well-defined generation order ([Policy Gradients & PPO for Language Models](../05-posttraining-alignment/06-ppo-for-llms.html), [GRPO, RLOO & Critic-Free RL](../05-posttraining-alignment/08-grpo-rloo.html)) — to diffusion LLMs is active research, not a solved recipe.
- **Length handling.** Pure diffusion needs a length up front; block diffusion fixes this but adds the block-size hyperparameter and a block-causal mask. Either way, length control is more involved than AR's natural "generate until `[EOS]`."
- **Tooling maturity.** vLLM, SGLang, speculative decoding ([Speculative Decoding: Draft Models, Medusa, EAGLE & Lookahead](../07-inference-serving/06-speculative-decoding.html)), and the entire kernel ecosystem are tuned for AR decode. Diffusion-specific serving is catching up but is not yet at parity.

So when should you reach for a diffusion LLM today? When **single-stream latency dominates your objective** (interactive coding assistants, low-concurrency agents), when the task is **natively bidirectional** (infilling, editing, constraint satisfaction), or when you want robustness to **reversal-style** queries. When you need exact likelihoods, the mature post-training/RL stack, maximal throughput at very high batch, or simply the lowest-risk path, autoregressive transformers remain the default — for now.

Diffusion and non-AR language models are best understood as a third major branch of the sequence-modeling tree, alongside the attention transformers of this Part and the recurrent/SSM alternatives of [Beyond Attention: SSMs, Mamba, RWKV & Linear Attention](../02-transformer/11-ssm-and-alternatives.html). All three are attempts to escape the $O(L)$-serial, $O(L^2)$-attention cost of the vanilla transformer; diffusion attacks the *serial-depth* axis specifically, and pays in approximate factorization and extra compute for the privilege.

---

## Key Takeaways

!!! key "Key Takeaways"

    - Autoregressive generation has a hard serial-depth floor of one forward pass per token; non-autoregressive **masked (absorbing-state) diffusion** instead denoises all positions in parallel over $N \ll L$ iterative steps, cutting serial depth and boosting single-stream tokens/sec.
    - The absorbing-diffusion objective reduces to a **time-conditioned, mask-rate-weighted cross-entropy on masked positions only** — a continuous-time generalization of BERT-style masked LM that is a valid likelihood bound; the weight $1/(1-\alpha_t)$ is what makes it generative rather than just representation-learning.
    - A single denoising step models masked tokens as **conditionally independent**, which is wrong for language; **iterative remasking** (commit high-confidence tokens, re-condition, repeat) recovers inter-token correlations. More steps trade compute for coherence; the remasking schedule is a first-class decoding hyperparameter.
    - **Block diffusion** interpolates between AR and diffusion: generate blocks left-to-right (autoregressive, with a block-causal mask) but tokens within a block in parallel (diffusion). This unlocks **KV-cache reuse** and **flexible/arbitrary output length**, mapping cleanly onto existing AR serving stacks.
    - The speedup is a **serial-depth (latency) win, not a FLOPs win**: diffusion often does more total compute per token but in fewer, more parallel steps. The advantage is largest at low-to-moderate batch and long single outputs; at very high concurrency, batched AR throughput catches up.
    - **Bidirectionality** is a separate, structural benefit: native infilling/editing, global-constraint satisfaction, and robustness to the **reversal curse** — capabilities AR models must bolt on but diffusion gets for free.
    - Real systems: **LLaDA** (8B, from scratch) showed diffusion LLMs can rival same-size AR models; **Dream** (7B) adapts an AR checkpoint to the diffusion objective; **Mercury** (Inception Labs) is the commercial block-diffusion system pitched on parallel-decode throughput for coding.
    - Caveats: likelihoods are ELBO bounds (not directly comparable to AR perplexity), length handling and the post-training/RL stack are less mature, and the kernel/serving ecosystem is still AR-centric. Reach for diffusion when single-stream latency, infilling, or reversal-robustness matter most.

---

!!! sota "State of the Art & Resources (2026)"
    Masked-diffusion language models moved from research curiosity to scaled, partly-commercial systems in 2024–2025. The frontier is semi-autoregressive **block diffusion** (for cache reuse and flexible length), AR-to-diffusion adaptation (reusing AR pretraining compute), and the early diffusion-native post-training stack.

    **Foundational work**

    - Austin et al., *Structured Denoising Diffusion Models in Discrete State-Spaces (D3PM)* (2021) — general discrete-diffusion transition matrices; the absorbing-state special case is the ancestor of masked diffusion.
    - Hoogeboom et al., *Argmax Flows and Multinomial Diffusion* (2021) — early discrete diffusion for categorical data.
    - Lou, Meng & Ermon, *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution (SEDD)* (2023) — score-entropy objective that made discrete diffusion competitive with AR on likelihood.

    **The simplified masked-diffusion objective (2024)**

    - Sahoo et al., *Simple and Effective Masked Diffusion Language Models (MDLM)* (2024) — shows the masked-diffusion ELBO reduces to a clean weighted masked cross-entropy.
    - Shi et al., *Simplified and Generalized Masked Diffusion for Discrete Data* (2024) — parallel derivation of the same simplification.
    - Ou et al., *Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data (RADD)* (2024) — explains why explicit time-conditioning is largely redundant.

    **Scaled systems & semi-AR (2025)**

    - Nie et al., *Large Language Diffusion Models (LLaDA)* (2025) — 8B from-scratch masked-diffusion LLM competitive with same-size AR baselines; demonstrates the reversal-curse advantage.
    - Dream team, *Dream 7B* (2025) — diffusion LLM adapted from an autoregressive checkpoint with context-adaptive noise and confidence-based decoding.
    - Arriola et al., *Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models (BD3-LM)* (2025) — the block-causal hybrid enabling KV-cache reuse and arbitrary-length generation.
    - Inception Labs, *Mercury* (2025) — commercial diffusion-LLM family (incl. Mercury Coder) built on a block-diffusion architecture for high parallel-decode throughput.

    **Go deeper**

    - The image/continuous-diffusion foundations live in [Diffusion Models & Generative Modeling (Breadth)](../10-multimodal-and-arch/04-diffusion-generative.html); the serving-side economics are in [Inference Economics: Latency, Throughput & Cost](../07-inference-serving/12-inference-economics.html).

## Further Reading

- Austin, J., et al. (2021). **Structured Denoising Diffusion Models in Discrete State-Spaces (D3PM).** NeurIPS 2021.
- Lou, A., Meng, C., & Ermon, S. (2023). **Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution (SEDD).** ICML 2024.
- Sahoo, S., et al. (2024). **Simple and Effective Masked Diffusion Language Models (MDLM).** NeurIPS 2024.
- Shi, J., et al. (2024). **Simplified and Generalized Masked Diffusion for Discrete Data.** NeurIPS 2024.
- Ou, J., et al. (2024). **Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data (RADD).**
- Nie, S., et al. (2025). **Large Language Diffusion Models (LLaDA).**
- Arriola, M., et al. (2025). **Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models (BD3-LM).** ICLR 2025.
- Inception Labs (2025). **Mercury / Mercury Coder** — commercial diffusion LLM.
- Gong, S., et al. (2025). **Dream 7B** — autoregressive-to-diffusion adaptation.
