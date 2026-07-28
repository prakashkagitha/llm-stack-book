# 14.8 Mid-Training: Quality Annealing, Long-Context Extension & Capability Injection

By the end of [Chapter 14.7](../14-capstone/07-pretraining-run.html) our `Stack-100M` model has consumed roughly **18 billion tokens** during the long **stable** phase of a Warmup-Stable-Decay (WSD) schedule, at a constant learning rate, on the broad 70/15/10/5 FineWeb-Edu / Cosmopedia / code / math mix. The train loss has flattened somewhere in the ballpark of the low-3s nats/token, and we have a checkpoint — `ckpt_stable.pt` — that is *unfinished on purpose*. We deliberately did **not** decay the learning rate to zero, because the most valuable compute in the whole project is still in front of us.

This chapter spends it. **Mid-training** is the phase that sits *between* raw pretraining and post-training — a term the OLMo 2 report (Allen Institute for AI, 2024–2025) helped crystallize, though the idea is now near-universal across open models. We make three moves, all sharing one checkpoint and one continued-training loop:

1. **WSD decay-phase annealing** — run the short, sharp LR-decay phase on a *higher-quality* data mix (more Cosmopedia, instruction-flavored text, more math and code). Because the decay phase is where most of the *committed* loss reduction lands, upgrading the data here buys a large, cheap quality jump.
2. **Long-context extension** — continue training at `seq_len = 8192` (a 4× jump from the pretrain `2048`) using **RoPE base rescaling** (NTK-aware / YaRN-style), so the model can actually attend across a document instead of a paragraph.
3. **Capability injection** — concentrate math and code in the final sub-phase, so the narrow tool-using agent we build in [Chapter 14.10](../14-capstone/10-agentic-narrow.html) has an arithmetic-and-structure floor to stand on.

This is where "narrow but real" specialization begins. We are still doing self-supervised next-token prediction — no SFT, no chat template yet (that is [Chapter 14.9](../14-capstone/09-post-training.html)). Mid-training changes *what data* the model sees and *how far* it can see, not *what objective* it optimizes.

This chapter builds directly on [Continual & Domain-Adaptive Pretraining](../03-pretraining/16-continual-pretraining.html), [Learning Rate Schedules, Warmup, Batch Size & Hyperparameters](../03-pretraining/10-lr-schedules-hparams.html), [Long-Context Pretraining & Context Extension](../03-pretraining/13-long-context-pretraining.html), [Positional Encodings: Sinusoidal, Learned, RoPE & ALiBi](../02-transformer/05-positional-encoding.html), and [Data Mixing, Domain Weighting & Curriculum](../03-pretraining/14-data-mixing-curriculum.html). We cross-link them rather than re-derive; here we *apply* them to one concrete model.

## Why a Separate Mid-Training Phase Exists

The instinct of a first-time trainer is to pick one data mix and one cosine schedule, train to convergence, and call it done. That is the Karpathy-nanoGPT recipe, and it is a fine baseline. But two facts about modern small models push us to split the run.

**Fact 1: the decay phase is disproportionately valuable.** In a WSD schedule (MiniCPM, Hu et al., 2024; used in spirit by DeepSeek's multi-step decay), the learning rate is warmed up, held constant for the bulk of training (the *stable* phase), then decayed sharply over a short final window (the *decay* phase). Empirically — and this is the whole reason WSD exists — the model looks *unconverged* at the end of the stable phase and then drops steeply the moment you begin decaying. Loss that was stuck falls by a visible margin over the last few percent of tokens. The decay phase is where the model *commits* the representations it has been loosely exploring at high LR.

That timing is a gift. If the last few percent of tokens matter most, then the *quality* of those specific tokens matters most too. Feeding premium data precisely into the decay window is the highest-leverage data decision in the entire pipeline. This is the core insight of mid-training and the reason WSD and annealing are described together in [Chapter 14.6](../14-capstone/06-optimizer-and-schedule.html).

**Fact 2: quality and context length are cheap to change late, expensive to change early.** Pretraining on 8192-token contexts from step zero would have roughly quadrupled attention cost for 20B tokens — a large tax paid over the *entire* run for a capability we only need at the *end*. Similarly, high-quality synthetic data (Cosmopedia v2) and curated math/code are scarcer and more expensive per token than filtered web text; spending 20B tokens of it would be wasteful when ~2B tokens of it, placed correctly, captures most of the benefit. Mid-training is the economically correct place to pay for both.

!!! note "Aside: mid-training vs. continual pretraining"

    These overlap but are not identical. **Continual pretraining** (Ch. 3.16) usually means adapting an *already-finished, fully-decayed* model to a new domain, and its central headache is catastrophic forgetting under LR re-warming. **Mid-training** is planned *before* the model is finished: we never fully decayed, so we resume from a high-LR checkpoint and there is nothing to "re-warm" and little to forget. The Ibrahim et al. (2024) observation — that re-warming and re-decaying a fully-decayed checkpoint costs you loss you have to claw back — is exactly why we saved `ckpt_stable.pt` *pre-decay* and resume from it with the LR still at its plateau.

### The mid-training budget

We slice a ~2B-token window off the ~20B-token budget for all three moves combined — about **10%** of total tokens. A representative split (illustrative; tune per [Chapter 14.5](../14-capstone/05-mini-scaling-laws.html)):

| Sub-phase | Tokens | `seq_len` | Purpose |
|---|---|---|---|
| A. Anneal (premium mix) | ~1.2B | 2048 | quality jump, LR peak → ~10% of peak |
| B. Long-context extend | ~0.6B | 8192 | RoPE rescale, LR ~10% → floor |
| C. Capability injection | ~0.2B | 8192 | concentrated math/code at the LR floor |

The LR decays *monotonically across all three* sub-phases — mid-training is one continuous WSD decay, just with the data mix and sequence length changing underneath it.

## Move 1 — WSD Decay-Phase Annealing

### The annealing mix

We shift the sampling weights toward denser, cleaner, more instruction-shaped text. The pretrain mix was tuned for *breadth*; the anneal mix is tuned for *the model's final impression*.

| Source | Pretrain weight | Anneal weight | Why the shift |
|---|---|---|---|
| FineWeb-Edu (Penedo et al., 2024) | 70% | 40% | still the base, but we lean less on raw web |
| Cosmopedia v2 (HuggingFace) | 15% | 30% | synthetic textbooks: dense, clean, knowledge-rich |
| StarCoder subset (BigCode) | 10% | 15% | structure and reasoning scaffolding |
| FineMath / OpenWebMath | 5% | 10% | arithmetic and symbolic reasoning |
| Instruction-flavored (QA, how-to) | 0% | 5% | a gentle nudge toward the format SFT will use |

The instruction-flavored slice is deliberately small and is **not** SFT: it is raw text that *happens* to be shaped like questions-and-answers or explanations, so the model warms to the register without us yet imposing a chat template or masking losses. Full instruction tuning is [Chapter 14.9](../14-capstone/09-post-training.html).

The data plumbing — streaming shards, document-aware packing, the `PackedDataset` memmap reader — was all built in [Chapter 14.2](../14-capstone/02-data-pipeline.html). Here we only re-weight the mixture. We reuse the same `mix_stream` sampler from `stacklm.data`; annealing is a config change, not new machinery.

```python
# stacklm/data.py  (established in Ch. 14.2 — reproduced here for the mix change)
import numpy as np

# Each source is a directory of uint16 .bin shards produced by the tokenizer
# (Ch. 14.3). The keys must match the manifest written during data prep.
ANNEAL_MIX = {
    "fineweb_edu":   0.40,
    "cosmopedia_v2": 0.30,
    "starcoder":     0.15,
    "finemath":      0.10,
    "instruct_flav": 0.05,
}
assert abs(sum(ANNEAL_MIX.values()) - 1.0) < 1e-9, "mixture weights must sum to 1"

def mix_stream(sources: dict[str, "PackedDataset"], weights: dict[str, float],
               seq_len: int, batch_size: int, rng: np.random.Generator):
    """Yield (B, seq_len) uint16 token batches by sampling *sources* per *weights*.

    Each value in `sources` is a PackedDataset (Ch. 14.2) exposing
    `sample_sequence(seq_len, rng)`. Document-aware packing and intra-doc
    position resets are handled inside each source; here we only choose which
    source each sequence in the batch is drawn from. Sampling at the *sequence*
    level (not the token level) keeps whole documents intact within a source.
    """
    names = list(weights)
    probs = np.array([weights[n] for n in names], dtype=np.float64)
    probs /= probs.sum()
    while True:
        # Choose a source per sequence in the batch, so a single batch can mix
        # web + code + math — this decorrelates gradients across domains.
        picks = rng.choice(len(names), size=batch_size, p=probs)
        batch = np.empty((batch_size, seq_len), dtype=np.uint16)
        for i, p in enumerate(picks):
            batch[i] = sources[names[p]].sample_sequence(seq_len, rng)
        yield batch
```

### Resuming the WSD decay from the stable checkpoint

The learning-rate mechanics are the crux. During the stable phase the effective schedule multiplier was pinned at `1.0` (peak LR). Mid-training runs the decay leg: multiplier goes `1.0 → 0` over the mid-training token budget, using a **1−sqrt** shape (the MiniCPM default; sharper early, gentle tail). We compute it against the mid-training step count, *not* the global step count, because the stable phase already happened.

```python
# stacklm/schedule.py
import math

def wsd_decay_multiplier(mid_step: int, num_decay_steps: int,
                         final_frac: float = 0.0, shape: str = "1-sqrt") -> float:
    """LR multiplier for the WSD *decay* leg, indexed from the start of mid-training.

    mid_step        : steps taken *since* resuming from ckpt_stable (0-indexed).
    num_decay_steps : total mid-training steps (all three sub-phases together).
    final_frac      : LR floor as a fraction of peak (we use ~0.0; some use 0.1).
    shape           : "1-sqrt" (MiniCPM) or "linear" or "cosine".
    Returns a value in [final_frac, 1.0]; multiply by peak_lr to get the LR.
    """
    t = min(mid_step, num_decay_steps) / num_decay_steps   # progress in [0, 1]
    if shape == "1-sqrt":
        decayed = 1.0 - math.sqrt(t)          # steep at first, long gentle tail
    elif shape == "linear":
        decayed = 1.0 - t
    elif shape == "cosine":
        decayed = 0.5 * (1.0 + math.cos(math.pi * t))
    else:
        raise ValueError(shape)
    return final_frac + (1.0 - final_frac) * decayed

# Quick sanity check of the 1-sqrt shape:
if __name__ == "__main__":
    N = 10_000
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        s = int(frac * N)
        print(f"progress {frac:>4.0%}  ->  lr_mult {wsd_decay_multiplier(s, N):.4f}")
    # progress   0%  ->  lr_mult 1.0000
    # progress  25%  ->  lr_mult 0.5000
    # progress  50%  ->  lr_mult 0.2929
    # progress  75%  ->  lr_mult 0.1340
    # progress 100%  ->  lr_mult 0.0000
```

Notice the shape: half the LR is gone by the 25% mark and the last three-quarters of the decay window runs at under 30% of peak. That long low-LR tail is where the premium data gets *committed*. Feeding math and code into that tail (sub-phase C) is why we schedule capability injection last.

### Why 1−sqrt, and how to watch the anneal working

The `1-sqrt` shape is the MiniCPM default for a concrete reason: it drops the LR aggressively at the *start* of decay — where the model still has enough plasticity to re-organize — then spends a long, patient tail near the floor consolidating on the premium tokens. Linear decay spreads the drop evenly and tends to leave a little loss on the table; cosine is smoother but its slow early descent wastes some of the high-plasticity window. If you only run one anneal, `1-sqrt` is the safe pick. The one knob worth ablating (Ch. 14.5) is `final_frac`: decaying to a hard `0.0` squeezes out the last fraction of loss but leaves you a checkpoint that is *finished* and awkward to continue-train; decaying to `~0.1` of peak keeps the door open for a second anneal or for post-training that prefers a non-zero starting LR. We decay to the floor for the release checkpoint and keep a pre-final checkpoint (`ckpt_mid_longctx.pt`) around as the resumable one.

You should *watch* the anneal rather than trust it. The single most informative diagnostic is a **held-out loss curve on a fixed, frozen validation set** sampled from the *pretrain* distribution, evaluated every few hundred steps throughout mid-training. Because the validation set never changes, the curve is directly comparable to the stable-phase plateau, and you will see the characteristic WSD "elbow": a visible downward bend the moment decay begins, steepening through the low-LR tail. If that elbow does *not* appear, something is wrong — usually the resume loaded weights but not the optimizer state, or the LR multiplier is being computed against the global step instead of `mid_step`. A second, cheaper signal is the gradient norm: it should *shrink* smoothly as the LR falls; a rising grad norm during decay means the premium mix is too far out-of-distribution (dial back the instruction-flavored slice) or the batch is too small for the long sequences.

## Move 2 — Long-Context Extension to 8192

Stack-100M was pretrained at `max_seq_len = 2048` with RoPE base $\theta = 10000$ (see the config in [Chapter 14.4](../14-capstone/04-architecture.html)). We now want it to attend over 8192 tokens. The obstacle is entirely in the positional encoding, and it is worth being precise about *why*.

### The extrapolation problem, concretely

RoPE rotates each 2-D slice of a query/key by an angle $m\theta_k$ that grows linearly with position $m$, where $\theta_k = \theta^{-2k/d}$ for pair index $k = 0,1,\dots,d/2-1$ and $d$ is the head dimension. For Stack-100M, $d = 64$, so there are $32$ frequency pairs. The dot product $q_m \cdot k_n$ depends only on the relative offset $m - n$ — that is RoPE's defining property, derived from scratch in [Positional Encodings: Sinusoidal, Learned, RoPE & ALiBi](../02-transformer/05-positional-encoding.html).

The problem: during pretraining, $m$ never exceeded $2048$. The low-frequency pairs (large $k$, tiny $\theta_k$) barely complete a fraction of a rotation within that window. If we naively run the model at position $8191$, those slow dimensions swing into angular ranges the model has **never seen**, and attention scores go pathological. The high-frequency (local) pairs are fine — they complete many turns even within 2048 — but the long-range pairs break.

### RoPE base rescaling (NTK-aware)

The fix is to *stretch the frequency ladder* so that positions up to 8192 land in the same angular range the model already learned for positions up to 2048. NTK-aware scaling (the "increase the base" trick, popularized on LocalLLaMA by bloc97, 2023) does this by raising the RoPE base $\theta$:

$$
\theta' = \theta \cdot s^{\,d/(d-2)}, \qquad s = \frac{L_{\text{new}}}{L_{\text{old}}}
$$

A larger base makes every $\theta_k$ *smaller*, so each dimension rotates more slowly — exactly compensating for the longer positions. The $d/(d-2)$ exponent is the NTK correction that leaves the highest-frequency pair essentially untouched (preserving local resolution) while stretching the low-frequency pairs (which needed the range). YaRN (Peng et al., 2023) refines this per-wavelength and adds an attention-temperature ("length scaling") correction that divides the pre-softmax logits by a factor $\propto \log$ of the scale, compensating for the entropy growth of attention over longer contexts; we cross-link its full treatment in [Long-Context Pretraining & Context Extension](../03-pretraining/13-long-context-pretraining.html) and use the simpler base rescale here, because **we are going to continue-train** — a small amount of training at 8192 repairs any residual mismatch far more cheaply than getting the zero-shot formula perfect.

!!! example "Worked example — rescaling Stack-100M's RoPE base for 4× context"

    Stack-100M: head dimension $d = 64$, pretrain length $L_{\text{old}} = 2048$, target $L_{\text{new}} = 8192$, original base $\theta = 10000$.

    Scale factor: $s = 8192 / 2048 = 4$.

    Exponent: $d/(d-2) = 64/62 = 1.0323$.

    $$
    \theta' = 10000 \cdot 4^{1.0323} = 10000 \cdot e^{1.0323 \cdot \ln 4} = 10000 \cdot e^{1.431} \approx 10000 \cdot 4.18 \approx \mathbf{41{,}800}.
    $$

    So we bump the RoPE base from $10000$ to about $41{,}800$ (round to $42000$). Check the slowest pair, $k = 31$, whose exponent is $2k/d = 62/64$: originally $\theta_{31} = 10000^{-62/64} \approx 1.33\times10^{-4}$ rad/token, wavelength $2\pi/\theta_{31} \approx 47{,}000$ tokens. After rescaling, $\theta'_{31} = 41800^{-62/64} \approx 3.3\times10^{-5}$ rad/token, wavelength $\approx 188{,}000$ tokens — the slow dimension now turns ~4× more slowly, so the phase at position 8192 ($8192 \cdot \theta'_{31} \approx 0.27$ rad) matches the phase the model learned at position ~2000 ($2048 \cdot \theta_{31} \approx 0.27$ rad). The fastest pair ($k=0$, $\theta_0 = \theta^0 = 1$) is exactly unchanged by any base change: local resolution is preserved.

Here is the rescale, wired to rebuild the model's RoPE cache in place. Because Stack-100M uses **NoPE on every 4th layer** (SmolLM3, HuggingFace 2025; Kazemnejad et al., 2023), those layers have *no* positional encoding and are untouched by rescaling — they already generalize across length by construction, which is precisely why we included them.

```python
# stacklm/rope.py  (build_rope_cache established in Ch. 14.1; rescale added here)
import torch

def build_rope_cache(seq_len: int, head_dim: int, base: float,
                     device=None, dtype=torch.float32):
    """Precompute (cos, sin) tables of shape (seq_len, head_dim) for RoPE.

    Identical to the pretrain cache builder — we only ever change `base` and
    `seq_len` at mid-training. inv_freq uses even indices 0,2,...,head_dim-2.
    """
    k = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = base ** (-k / head_dim)                     # (head_dim/2,)
    pos = torch.arange(seq_len, dtype=torch.float32, device=device)
    ang = torch.outer(pos, inv_freq)                       # (seq_len, head_dim/2)
    ang = torch.cat([ang, ang], dim=-1)                    # (seq_len, head_dim)
    return ang.cos().to(dtype), ang.sin().to(dtype)

def ntk_rescaled_base(base: float, head_dim: int,
                      old_len: int, new_len: int) -> float:
    """NTK-aware RoPE base rescaling: theta' = theta * s^(d/(d-2))."""
    s = new_len / old_len
    return base * (s ** (head_dim / (head_dim - 2)))

@torch.no_grad()
def extend_context(model, new_seq_len: int, device) -> float:
    """Rescale RoPE and rebuild the cache for long-context mid-training.

    Returns the new base (for logging / the config manifest). NoPE layers hold
    no rope cache and are left alone. Call this once, at the start of sub-phase B.
    """
    cfg = model.config
    new_base = ntk_rescaled_base(cfg.rope_theta, cfg.head_dim,
                                 old_len=cfg.max_seq_len, new_len=new_seq_len)
    cos, sin = build_rope_cache(new_seq_len, cfg.head_dim, new_base,
                                device=device, dtype=torch.float32)
    # Every rotary (non-NoPE) attention block shares one cache via buffers.
    for blk in model.blocks:
        if getattr(blk.attn, "use_rope", True):            # NoPE layers: False
            blk.attn.rope_cos, blk.attn.rope_sin = cos, sin
    # Update the live config so checkpoints record the new geometry.
    cfg.rope_theta = new_base
    cfg.max_seq_len = new_seq_len
    return new_base
```

Two practical notes that matter more than the formula:

- **You must actually train at 8192.** Base rescaling gets the geometry right, but the model has never *attended* across 8192 tokens; a few hundred million tokens at the long length lets attention heads learn to use the new range. Zero-shot rescaling degrades gracefully but continued training is what makes long-context real.
- **Your data must contain long documents.** If every packed sequence is a pile of 400-token web snippets separated by document-mask boundaries (see the document-aware masking from [Chapter 14.2](../14-capstone/02-data-pipeline.html)), the model never sees a genuine 8192-token dependency and long-context training does nothing. For sub-phase B we up-weight naturally long sources (long Cosmopedia articles, whole code files from StarCoder, multi-page FineWeb-Edu documents) and relax packing so long documents stay contiguous.

!!! warning "Common pitfall: forgetting the KV-cache and memory blow-up at 8192"

    Attention memory and compute are quadratic in sequence length. Going 2048 → 8192 is 4× longer, so the attention score matrix is **16×** larger and activation memory jumps. On the single A100 (80GB) flagship tier this is fine only if you *also* cut the micro-batch by ~4× and lean harder on gradient accumulation to keep the ~0.5M-token global batch (Ch. 14.6), plus [FlashAttention](../04-kernels-efficiency/02-flash-attention-1.html) (which never materializes the full score matrix) and optional [activation checkpointing](../04-kernels-efficiency/10-memory-efficient-training.html). GQA with 2 KV heads (Ch. 14.1) already shrinks the KV cache 4×, which is a large part of why 8192 fits at all. Do the arithmetic *before* you launch, not after the OOM.

### Validating that the long-context actually works

Perplexity averaged over a sequence can *fall* even when the model still cannot use position 8000 — a couple of well-predicted local tokens hide a broken tail. Two cheap probes catch this. First, **loss-versus-position**: bin the per-token loss by its position within the 8192-window and plot the curve. A healthy extension shows loss that keeps *decreasing* (or at worst flattening) as position grows — the model is using more context to predict better. A curve that *rises* past ~2048 means the rescale-plus-training has not taken and the far positions are still noise. Second, a tiny **needle-in-a-haystack** check: plant a short unique fact ("the passcode is 4713") at a random depth in a long filler document and ask the model, via next-token prediction, to complete "the passcode is". If it recovers the needle only when it sits in the first 2048 tokens, you have geometry without capability — train sub-phase B longer or on longer documents. The honest, full evaluation belongs in [Chapter 14.11](../14-capstone/11-evaluation-and-serving.html); these are the two you run *during* mid-training to know the elbow is real.

!!! tip "Practitioner tip: rescale once, at the boundary — not per step"

    `extend_context` is called exactly once, when sub-phase B begins, and it mutates `cfg.rope_theta` and `cfg.max_seq_len` in place so the change is recorded in every subsequent checkpoint. Do **not** re-derive the base from the *original* 10000 each phase — after the first rescale the config already holds ~42000, and re-applying the $s^{d/(d-2)}$ factor to it would over-stretch the ladder. If you later want to reproduce the geometry from a checkpoint, read `rope_theta` and `max_seq_len` straight from the saved config; they are the ground truth, not the pretrain defaults.

## Move 3 — Capability Injection

The final sub-phase (C) runs at the LR floor and pushes the mixture hard toward the capabilities our downstream narrow agent needs: arithmetic, symbolic manipulation, and code structure.

| Source | Sub-phase C weight |
|---|---|
| FineMath / OpenWebMath | 30% |
| StarCoder subset | 30% |
| Cosmopedia v2 (STEM-heavy slice) | 25% |
| FineWeb-Edu | 15% |

This is a small budget (~0.2B tokens) at a low, still-nonzero LR — enough to sharpen number-and-symbol handling without over-fitting or wrecking general fluency. It is honest to call this what it is: **not** turning Stack-100M into a math model, but giving a 100M model *just enough* arithmetic and code grounding that the RLVR run in [Chapter 14.9](../14-capstone/09-post-training.html) (narrow GRPO on verifiable integer arithmetic, following GRPO from DeepSeekMath, Shao et al., 2024) and the ReAct agent (Yao et al., 2022) in [Chapter 14.10](../14-capstone/10-agentic-narrow.html) have a base to reinforce. At 100M params you cannot inject a capability the model has no capacity for; you can only make sure the capacity you have is pointed at the right target. That is "narrow but real."

Capability injection reuses the exact same loop and schedule — it is simply sub-phase C's mixture, applied while the LR finishes its decay to the floor. No new code beyond a mixture dict.

### Why this ordering

The three moves are sequenced deliberately, and the order is not arbitrary. Annealing runs *first*, at 2048, because the general quality jump wants the highest LR of the decay leg and the largest, most diverse token budget — you consolidate broad representations before you specialize. Long-context extension runs *second* because it is a targeted change to a subsystem (the positional geometry) that benefits from a settled model: rescaling RoPE on a mid-anneal checkpoint whose attention patterns have already sharpened means the few hundred million long-context tokens teach the heads to *use* the new range rather than fighting a still-shifting representation. Capability injection runs *last*, at the LR floor, because it is the narrowest and most over-fitting-prone mix; keeping it in the low-LR tail limits how far it can pull the model off the general manifold while still committing arithmetic and code structure. Reverse any two of these and you either waste the high-plasticity window on narrow data or extend context on a model that is still moving. A useful mental model: **broad → long → narrow**, riding one decreasing learning rate down.

## The Mid-Training Loop: One Continuous Decay

Now we assemble everything into a single continued-training routine that resumes from `ckpt_stable.pt` and runs all three sub-phases back to back. It reuses `StackLM`, the Muon+AdamW hybrid optimizer, and the checkpoint helpers established in Chapters 14.1, 14.6, and 14.7 — mid-training adds *no new model code*, only orchestration.

```python
# stacklm/midtrain.py
"""Mid-training: resume from the pre-decay stable checkpoint and run the WSD
decay across three sub-phases (anneal @2048 -> long-context @8192 -> capability
injection @8192), on progressively higher-quality / narrower data mixes."""
import numpy as np
import torch
from dataclasses import dataclass

from stacklm.model import StackLM, StackConfig          # Ch. 14.1
from stacklm.optim import build_optimizers               # Muon+AdamW hybrid, Ch. 14.6
from stacklm.checkpoint import load_checkpoint, save_checkpoint  # Ch. 14.7
from stacklm.data import PackedDataset, mix_stream        # Ch. 14.2
from stacklm.schedule import wsd_decay_multiplier         # this chapter
from stacklm.rope import extend_context                   # this chapter


@dataclass
class SubPhase:
    name: str
    tokens: int                 # token budget for this sub-phase
    seq_len: int                # 2048 or 8192
    mix: dict                   # source -> weight
    extend_now: bool = False    # rescale RoPE at the start of this sub-phase


# The three moves of mid-training, in order. Token budgets are illustrative
# (~2B total = ~10% of the 20B pretrain budget); tune per Ch. 14.5.
PHASES = [
    SubPhase("anneal", tokens=1_200_000_000, seq_len=2048,
             mix={"fineweb_edu":0.40, "cosmopedia_v2":0.30, "starcoder":0.15,
                  "finemath":0.10, "instruct_flav":0.05}),
    SubPhase("longctx", tokens=600_000_000, seq_len=8192, extend_now=True,
             mix={"cosmopedia_v2":0.35, "starcoder":0.30, "fineweb_edu":0.25,
                  "finemath":0.10}),                       # long-doc-heavy sources
    SubPhase("capability", tokens=200_000_000, seq_len=8192,
             mix={"finemath":0.30, "starcoder":0.30, "cosmopedia_v2":0.25,
                  "fineweb_edu":0.15}),
]

GLOBAL_BATCH_TOKENS = 512_000    # ~0.5M-token batch, same as pretrain (Ch. 14.6)
PEAK_LR = 3e-3                    # Muon peak LR the stable phase ran at (Ch. 14.6)
GRAD_CLIP = 1.0
Z_LOSS_W = 1e-4                  # keep the pretrain z-loss (Ch. 14.1)


def steps_for(sub: SubPhase) -> int:
    """Optimizer steps to consume `sub.tokens` at the global batch size."""
    return sub.tokens // GLOBAL_BATCH_TOKENS


def mid_train(stable_ckpt: str, out_dir: str, device="cuda"):
    # ---- 1. Resume the *pre-decay* model + optimizer state --------------------
    model = StackLM(StackConfig()).to(device)
    optimizers = build_optimizers(model, peak_lr=PEAK_LR)  # [muon, adamw]
    start = load_checkpoint(stable_ckpt, model, optimizers, map_location=device)
    print(f"resumed from {stable_ckpt} @ global step {start['step']} "
          f"(stable phase complete, LR still at peak)")

    total_decay_steps = sum(steps_for(p) for p in PHASES)  # decay spans ALL phases
    rng = np.random.default_rng(start.get("data_seed", 1234) + 1)  # fresh stream
    mid_step = 0                                            # index within decay

    # ---- 2. Walk the three sub-phases; the LR decays continuously across them --
    for sub in PHASES:
        if sub.extend_now:
            new_base = extend_context(model, sub.seq_len, device)
            print(f"[{sub.name}] RoPE base rescaled 10000 -> {new_base:.0f}, "
                  f"seq_len -> {sub.seq_len}")

        # Micro-batch shrinks with sequence length to hold activation memory
        # roughly constant; grad-accum makes up the global batch. (see warning)
        seqs_per_global = GLOBAL_BATCH_TOKENS // sub.seq_len
        micro_bs = max(1, 32 * 2048 // sub.seq_len)        # ~4x smaller at 8192
        accum = max(1, seqs_per_global // micro_bs)

        sources = {name: PackedDataset(f"data/{name}", seq_len=sub.seq_len)
                   for name in sub.mix}
        stream = mix_stream(sources, sub.mix, sub.seq_len, micro_bs, rng)

        for _ in range(steps_for(sub)):
            lr = PEAK_LR * wsd_decay_multiplier(mid_step, total_decay_steps)
            for opt in optimizers:
                for g in opt.param_groups:
                    g["lr"] = lr * g.get("lr_scale", 1.0)   # per-group scaling

            # ---- gradient accumulation over the global batch -----------------
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            loss_acc = 0.0
            for _micro in range(accum):
                batch = torch.from_numpy(next(stream).astype(np.int64)).to(device)
                x, y = batch[:, :-1], batch[:, 1:]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, loss = model(x, targets=y, z_loss_weight=Z_LOSS_W)
                (loss / accum).backward()
                loss_acc += loss.item() / accum
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            for opt in optimizers:
                opt.step()

            if mid_step % 50 == 0:
                print(f"[{sub.name}] mid_step {mid_step:>6} "
                      f"lr {lr:.2e}  loss {loss_acc:.4f}  seq {sub.seq_len}")
            mid_step += 1

        # Checkpoint at every sub-phase boundary (model+opt+step+rng+config).
        save_checkpoint(f"{out_dir}/ckpt_mid_{sub.name}.pt", model, optimizers,
                        step=start["step"] + mid_step, config=model.config,
                        data_seed=int(rng.integers(1 << 30)))
        print(f"[{sub.name}] done -> saved ckpt_mid_{sub.name}.pt")

    save_checkpoint(f"{out_dir}/ckpt_mid_final.pt", model, optimizers,
                    step=start["step"] + mid_step, config=model.config)
    print("mid-training complete -> ckpt_mid_final.pt "
          "(ready for SFT in Ch. 14.9)")
    return model
```

A few design points worth flagging:

- **One decay across three phases.** `total_decay_steps` spans all sub-phases, and `mid_step` runs continuously, so the LR falls smoothly from peak to floor *through* the sequence-length change and the mixture changes. We do not reset the schedule at each boundary — that would create three little decay cliffs instead of one clean anneal.
- **We resume the optimizer state, not just the weights.** Muon's momentum buffers and AdamW's moments carry over from the stable phase (`load_checkpoint` restores them). Throwing them away would inject a transient the fresh decay does not need. This is the concrete payoff of saving `ckpt_stable.pt` *with* optimizer state in Ch. 14.7.
- **Micro-batch shrinks, global batch holds.** At 2048, `micro_bs = 32` and `accum = 250 // 32 = 7`; at 8192, `micro_bs = 8` and `accum = 62 // 8 = 7`. Both yield ~0.46M tokens per optimizer step, so the LR/batch relationship (Ch. 3.10) is unchanged across the sequence-length jump.
- **z-loss stays on.** The small `logsumexp` penalty (Ch. 14.1) keeps logits well-scaled through the low-LR tail; dropping it late can let logits drift.

The CI smoke test (per the PLAN's deliverable standard) runs this exact routine at toy scale: a tiny `StackConfig` (a few layers, `d_model` ~64, vocab ~256), an in-process synthetic corpus, `seq_len` 32 → 64, and a handful of steps per sub-phase, CPU-only and hermetic — proving `mid_train` *runs and resumes* end to end, while the prose above documents the real full-run magnitudes.

## What Mid-Training Costs and Buys

!!! example "Worked example — the compute bill for the anneal sub-phase"

    Sub-phase A anneals on ~1.2B tokens. Using the **6ND** FLOP rule (Ch. 3.4) with $N = 101\text{M}$ params and $D = 1.2\text{B}$ tokens:

    $$
    C = 6 N D = 6 \times 1.01\times10^{8} \times 1.2\times10^{9} \approx 7.3\times10^{17}\ \text{FLOPs} = 0.73\ \text{EFLOP}.
    $$

    On the flagship 1×A100 (80GB) at bf16 peak ≈ 312 TFLOP/s, assume a realistic ~40% MFU ≈ 125 TFLOP/s:

    $$
    t = \frac{7.3\times10^{17}}{1.25\times10^{14}} \approx 5.8\times10^{3}\ \text{s} \approx \mathbf{1.6\ GPU\text{-}hours}.
    $$

    At ~USD 1.5/GPU-hr that is about **USD 2.5**. The full ~2B-token mid-training (all three sub-phases) lands on the order of **2.5–4 GPU-hours ≈ USD 4–8** — a small fraction of the ~15–25 GPU-hour, ~USD 40–100 total budget. For that outlay you get the sharpest single quality jump in the project (the decay-phase drop), a 4× context window, and a math/code floor. This is the best marginal return on compute anywhere in the pipeline — which is exactly why mid-training is worth its own chapter.

You should expect the held-out loss to fall visibly across sub-phase A — on the order of a couple tenths of a nat below where the stable phase plateaued — with most of the drop concentrated in the low-LR tail. Long-context sub-phase B will *raise* the average loss slightly (8192-token prediction on long documents is genuinely harder than 2048-token snippets), which is expected and correct, not a regression; the loss-versus-position curve is the metric that shows the extension worked even as the scalar average ticks up. Capability sub-phase C nudges arithmetic and code perplexity down at the cost of a hair of general-web perplexity — the trade we are deliberately making. Report these as *illustrative* movements; never quote a fabricated benchmark. The honest evaluation lives in [Chapter 14.11](../14-capstone/11-evaluation-and-serving.html).

!!! interview "Interview Corner"

    **Q:** Why do practitioners run learning-rate *decay* on a different (higher-quality) data mixture than the stable phase, and why not just use that better mixture for the whole run?

    **A:** Two reasons, both economic. First, in a WSD schedule most of the *committed* loss reduction happens during the short decay phase — the model consolidates representations it explored loosely at high LR. So the data seen during decay disproportionately shapes the final model; putting your cleanest, densest, most task-relevant tokens there gives the largest quality-per-token return. Second, premium data (synthetic textbooks, curated math/code) is scarce and expensive per token, and high-quality-but-narrow data used for the *entire* run can hurt breadth and diversity. Annealing captures most of the upside for a few percent of the token budget while the long stable phase keeps broad coverage. It is a curriculum: broad-and-cheap to build general representations, then narrow-and-premium to sharpen and commit them. The same logic explains why you extend context and inject capabilities *late* — you pay for those only over the tokens where they matter, not across all 20B.

## Key Takeaways

!!! key "Key Takeaways"

    - **Mid-training is the phase between pretraining and post-training** (OLMo 2): still self-supervised next-token prediction, but on upgraded data, at longer context, with concentrated capabilities. It resumes from a *pre-decay* stable checkpoint — never a fully-decayed one.
    - **The WSD decay phase is where you spend your best data.** Most committed loss reduction happens during decay, so annealing on a premium mix (more Cosmopedia, math, code, instruction-flavored text) buys a large quality jump for ~10% of the token budget.
    - **Run one continuous decay across all sub-phases.** Let the LR fall smoothly from peak to floor *through* the mixture and sequence-length changes; do not reset the schedule at each boundary or you create decay cliffs.
    - **Long-context extension is a positional-encoding fix plus a little training.** Rescale the RoPE base with the NTK rule $\theta' = \theta\, s^{d/(d-2)}$ (10000 → ~42000 for Stack-100M's 2048→8192, $d{=}64$), then continue-train at 8192 on genuinely long documents. NoPE layers need no rescaling.
    - **Validate with loss-versus-position and a needle probe.** A falling scalar perplexity can hide a broken tail; confirm the model predicts *better* deep in the 8192-window before trusting the extension.
    - **Mind the quadratic.** 2048→8192 makes attention 16× heavier; shrink the micro-batch ~4×, keep the global batch fixed with grad accumulation, and rely on GQA + FlashAttention to fit on one A100.
    - **Capability injection is honest, not magic.** Concentrated math/code at the LR floor gives a 100M model an arithmetic-and-structure floor for the downstream narrow agent — it points existing capacity at the target; it cannot create capacity that is not there.
    - **The bill is tiny for the return.** ~2–4 GPU-hours (~USD 4–8) for the sharpest quality jump, a 4× context window, and a task-relevant base — the best marginal compute return in the whole capstone.
    - **Resume optimizer state, keep z-loss, checkpoint every boundary.** Mid-training adds orchestration, not model code; reuse `StackLM`, the Muon+AdamW hybrid, and the checkpoint helpers unchanged.

## Further Reading

- OLMo 2 Team, *2 OLMo 2 Furious* (Allen Institute for AI, 2024–2025): the open report that named and detailed the mid-training / annealing stage.
- Hu et al., *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies* (2024): the WSD schedule and decay-phase data annealing.
- Penedo et al., *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale* (HuggingFace, 2024): FineWeb-Edu and the Cosmopedia synthetic-data recipe.
- Peng et al., *YaRN: Efficient Context Window Extension of Large Language Models* (2023): per-wavelength RoPE scaling and the attention-temperature correction.
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021): the original RoPE.
- Chen et al., *Extending Context Window of Large Language Models via Positional Interpolation* (2023): position interpolation, the baseline NTK/YaRN build on.
- Ibrahim et al., *Simple and Scalable Strategies to Continually Pre-train Large Language Models* (2024): LR re-warming/re-decaying and the cost of resuming a decayed checkpoint.
- Kazemnejad et al., *The Impact of Positional Encoding on Length Generalization in Transformers* (2023), and the SmolLM3 report (HuggingFace, 2025): NoPE and length generalization.
