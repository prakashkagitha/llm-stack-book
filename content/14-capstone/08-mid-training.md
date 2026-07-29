# 14.8 Mid-Training: Quality Annealing, Long-Context Extension & Capability Injection

By the end of [Chapter 14.7](../14-capstone/07-pretraining-run.html) our `Stack-100M` model has consumed roughly **18 billion tokens** during the long **stable** phase of a Warmup-Stable-Decay (WSD) schedule, at a constant learning rate, on the broad 70/15/10/5 FineWeb-Edu / Cosmopedia / code / math mix. The train loss has flattened somewhere in the ballpark of the low-3s nats/token, and we have a checkpoint — `ckpt_stable.pt` — that is *unfinished on purpose*. We deliberately did **not** decay the learning rate to zero, because the most valuable compute in the whole project is still in front of us.

This chapter spends it. **Mid-training** is the phase that sits *between* raw pretraining and post-training — a term the OLMo 2 report (Allen Institute for AI, 2024–2025) helped crystallize, though the idea is now near-universal across open models. We make three moves, all sharing one checkpoint and one continued-training loop:

1. **WSD decay-phase annealing** — run the short, sharp LR-decay phase on a *higher-quality* data mix (more Cosmopedia, instruction-flavored text, more math and code). Because the decay phase is where most of the *committed* loss reduction lands, upgrading the data here buys a large, cheap quality jump.
2. **Long-context extension** — continue training at `seq_len = 8192` (a 4× jump from the pretrain `2048`) using **RoPE base rescaling** (NTK-aware / YaRN-style), so the model can actually attend across a document instead of a paragraph.
3. **Capability injection** — concentrate math and code in the final sub-phase, so the narrow tool-using agent we build in [Chapter 14.10](../14-capstone/10-agentic-narrow.html) has an arithmetic-and-structure floor to stand on.

This is where "narrow but real" specialization begins. We are still doing self-supervised next-token prediction — no SFT, no chat template yet (that is [Chapter 14.9](../14-capstone/09-post-training.html)). Mid-training changes *what data* the model sees and *how far* it can see, not *what objective* it optimizes.

Two of those three moves look like config changes and are not. Long-context extension in particular has two failure modes that are invisible until you check for them: **your packed shards may contain no positions past 2048**, in which case the RoPE rescale trains nothing; and **the dense document mask that worked at 2048 does not fit at 8192**, in which case you either OOM or silently fall off the FlashAttention path. This chapter cashes both of those out with code, because they are exactly the things that get hand-waved.

This chapter builds directly on [Continual & Domain-Adaptive Pretraining](../03-pretraining/16-continual-pretraining.html), [Learning Rate Schedules, Warmup, Batch Size & Hyperparameters](../03-pretraining/10-lr-schedules-hparams.html), [Long-Context Pretraining & Context Extension](../03-pretraining/13-long-context-pretraining.html), [Positional Encodings: Sinusoidal, Learned, RoPE & ALiBi](../02-transformer/05-positional-encoding.html), and [Data Mixing, Domain Weighting & Curriculum](../03-pretraining/14-data-mixing-curriculum.html). We cross-link them rather than re-derive; here we *apply* them to one concrete model.

## Why a Separate Mid-Training Phase Exists

The instinct of a first-time trainer is to pick one data mix and one cosine schedule, train to convergence, and call it done. That is the Karpathy-nanoGPT recipe, and it is a fine baseline. But two facts about modern small models push us to split the run.

**Fact 1: the decay phase is disproportionately valuable.** In a WSD schedule (MiniCPM, Hu et al., 2024; used in spirit by DeepSeek's multi-step decay), the learning rate is warmed up, held constant for the bulk of training (the *stable* phase), then decayed sharply over a short final window (the *decay* phase). Empirically — and this is the whole reason WSD exists — the model looks *unconverged* at the end of the stable phase and then drops steeply the moment you begin decaying. Loss that was stuck falls by a visible margin over the last few percent of tokens. The decay phase is where the model *commits* the representations it has been loosely exploring at high LR.

That timing is a gift. If the last few percent of tokens matter most, then the *quality* of those specific tokens matters most too. Feeding premium data precisely into the decay window is the highest-leverage data decision in the entire pipeline. This is the core insight of mid-training and the reason WSD and annealing are described together in [Chapter 14.6](../14-capstone/06-optimizer-and-schedule.html).

**Fact 2: quality and context length are cheap to change late, expensive to change early.** Pretraining on 8192-token contexts from step zero would have quadrupled the attention FLOP term for all 20B tokens — and for a deep-and-thin model like Stack-100M that term is *not* a rounding error (we do the arithmetic later in this chapter: at 8192 it is more than half of all training FLOPs). That is a large tax paid over the *entire* run for a capability we only need at the *end*. Similarly, high-quality synthetic data (Cosmopedia v2) and curated math/code are scarcer and more expensive per token than filtered web text; spending 20B tokens of it would be wasteful when ~2B tokens of it, placed correctly, captures most of the benefit. Mid-training is the economically correct place to pay for both.

!!! note "Aside: mid-training vs. continual pretraining"

    These overlap but are not identical. **Continual pretraining** (Ch. 3.16) usually means adapting an *already-finished, fully-decayed* model to a new domain, and its central headache is catastrophic forgetting under LR re-warming. **Mid-training** is planned *before* the model is finished: we never fully decayed, so we resume from a high-LR checkpoint and there is nothing to "re-warm" and little to forget. The Ibrahim et al. (2024) observation — that re-warming and re-decaying a fully-decayed checkpoint costs you loss you have to claw back — is exactly why we saved `ckpt_stable.pt` *pre-decay* and resume from it with the LR still at its plateau.

### The mid-training budget

We slice a ~2B-token window off the ~20B-token budget for all three moves combined — about **10%** of total tokens. A representative split (illustrative; tune per [Chapter 14.5](../14-capstone/05-mini-scaling-laws.html)):

| Sub-phase | Tokens | `seq_len` | Purpose |
|---|---|---|---|
| A. Anneal (premium mix) | ~1.2B | 2048 | quality jump, LR peak → ~23% of peak |
| B. Long-context extend | ~0.6B | 8192 | RoPE rescale, ~23% → ~5% of peak |
| C. Capability injection | ~0.2B | 8192 | concentrated math/code at the LR floor |

The LR decays *monotonically across all three* sub-phases — mid-training is one continuous WSD decay, just with the data mix and sequence length changing underneath it. (The exact boundary multipliers, 0.2253 and 0.0513, fall out of the $1-\sqrt{t}$ shape; Exercise 3 derives them.)

{{fig:midtrain-continuous-decay-spine}}

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

### Building the mixture loader

The data plumbing — streaming shards, document-aware packing, the `PackedMemmapDataset` memmap reader — was all built in [Chapter 14.2](../14-capstone/02-data-pipeline.html). Mid-training only re-weights the mixture, and the cleanest way to express a weighted mixture over per-source datasets in PyTorch is a `ConcatDataset` plus a `WeightedRandomSampler`: it samples at the **sequence** level, so one micro-batch can contain web, code, and math side by side (which decorrelates gradients across domains), and it needs no custom sampler code.

```python
# capstone/stacklm/mid/mixture.py
"""Weighted source mixtures over the Ch. 14.2 packed shards.

Each source lives in its own shard directory, packed at the sequence length the
sub-phase will train at (`data/mid/<source>_<seq_len>/`). Sampling is per
*sequence*, not per micro-batch, so a single forward pass mixes domains.
"""
import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from stacklm.data import PackedMemmapDataset          # Ch. 14.2

# Sub-phase A: the annealing mix. Keys are shard-directory names written by the
# tokenize+pack pass (Ch. 14.2) and recorded in the data manifest.
ANNEAL_MIX = {
    "fineweb_edu":   0.40,
    "cosmopedia_v2": 0.30,
    "starcoder":     0.15,
    "finemath":      0.10,
    "instruct_flav": 0.05,
}

# Sub-phases B and C are defined in the same module; their sources are motivated
# in "Repacking for sub-phase B" and "Move 3" below.
LONGCTX_MIX = {
    "starcoder_repo":   0.35,
    "books_pg19":       0.25,
    "arxiv_proofpile2": 0.15,
    "fineweb_edu_long": 0.15,
    "cosmopedia_v2":    0.10,   # deliberately SHORT: the anti-drift anchor
}
CAPABILITY_MIX = {
    "finemath":         0.30,
    "starcoder_repo":   0.30,
    "cosmopedia_v2":    0.25,
    "fineweb_edu_long": 0.15,
}

for _m in (ANNEAL_MIX, LONGCTX_MIX, CAPABILITY_MIX):
    assert abs(sum(_m.values()) - 1.0) < 1e-9, "mixture weights must sum to 1"


def build_mixture_loader(mix: dict, seq_len: int, micro_bs: int, *,
                         root: str = "data/mid", seed: int = 1234,
                         num_workers: int = 4):
    """Return an INFINITE iterator of batch dicts drawn from `mix`.

    Each yielded dict has `input_ids`, `position_ids`, `seq_ids`, `targets`
    (shapes (micro_bs, seq_len - 1)) -- exactly what `Stack100M.forward` and the
    document-aware mask consume.
    """
    datasets, weights = [], []
    for name, w in mix.items():
        d = PackedMemmapDataset(f"{root}/{name}_{seq_len}")
        # Guard against the single most common mid-training bug: reading shards
        # that were packed at the PRETRAIN length while claiming to train long.
        assert d.seq_len == seq_len, (
            f"{name} shards are packed at {d.seq_len}, not {seq_len}; "
            f"re-run the repack pass (see 'Repacking for sub-phase B')")
        datasets.append(d)
        # Per-sequence probability ∝ source weight, spread evenly inside a source.
        weights.extend([w / len(d)] * len(d))

    concat = ConcatDataset(datasets)
    g = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(weights, num_samples=len(concat),
                                    replacement=True, generator=g)
    loader = DataLoader(concat, batch_size=micro_bs, sampler=sampler,
                        drop_last=True, num_workers=num_workers,
                        pin_memory=True, persistent_workers=num_workers > 0)

    def infinite():
        while True:
            yield from loader
    return infinite()
```

!!! tip "Practitioner tip: the same mixture, three ways"

    Three real tools express "sample sources with these probabilities," and which one you want depends on where your data lives.

    - **Local packed shards (what we do):** `ConcatDataset` + `WeightedRandomSampler`, as above. Exact, resumable via the sampler's generator state, zero network.
    - **Streaming from the Hub:** `datasets.interleave_datasets([ds_a, ds_b, ...], probabilities=[0.4, 0.3, ...], stopping_strategy="all_exhausted")` from HuggingFace `datasets` — the right choice when you are still iterating on the mix and do not want to re-pack shards for every experiment.
    - **Production pretraining frameworks:** [`allenai/OLMo-core`](https://github.com/allenai/OLMo-core), [`huggingface/nanotron`](https://github.com/huggingface/nanotron), and [`pytorch/torchtitan`](https://github.com/pytorch/torchtitan) all take a declarative source-weight config and handle the mixture, resumption, and sharding for you. Read OLMo-core's anneal configs in particular — they are the closest public analogue of this chapter.

    Whichever you use, **write the realized mixture to the run manifest** (Ch. 14.12): "which weights did this checkpoint actually see" is the first question you will ask when a mid-training run underperforms.

### Resuming the WSD decay from the stable checkpoint

The learning-rate mechanics are the crux. During the stable phase the effective schedule multiplier was pinned at `1.0` (peak LR). Mid-training runs the decay leg: the multiplier goes `1.0 → 0` over the mid-training token budget, using a **1−sqrt** shape (the MiniCPM default; sharper early, gentle tail). We compute it against the mid-training step count, *not* the global step count, because the stable phase already happened.

This function is not new code — it already ships in `capstone/stacklm/optim/schedule.py` from [Chapter 14.6](../14-capstone/06-optimizer-and-schedule.html), and mid-training **imports** it rather than redefining it. It is reproduced here because its shape is the thing you need in your head for the rest of the chapter.

```python
# capstone/stacklm/optim/schedule.py  (Ch. 14.6 — reproduced, not redefined)
import math

def wsd_decay_multiplier(mid_step: int, num_decay_steps: int,
                         final_frac: float = 0.0, shape: str = "1-sqrt") -> float:
    """LR multiplier for the WSD *decay* leg, indexed from the start of mid-training.

    mid_step        : steps taken *since* resuming from ckpt_stable (0-indexed).
    num_decay_steps : total mid-training steps (all three sub-phases together).
    final_frac      : LR floor as a fraction of peak (we use ~0.0; some use 0.1).
    shape           : "1-sqrt" (MiniCPM) or "linear" or "cosine".
    Returns a value in [final_frac, 1.0]; multiply by EACH GROUP's peak LR.
    """
    t = min(mid_step, num_decay_steps) / max(1, num_decay_steps)   # progress in [0, 1]
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

!!! warning "Common pitfall: one multiplier, two peak learning rates"

    Stack-100M trains with a **hybrid optimizer** (Ch. 14.6): Muon on the 2-D hidden matrices, AdamW on the tied embedding, RMSNorm gains, and 1-D params. `build_optimizers(model, muon_lr=0.02, adamw_lr=3e-3)` gives the two groups *different* peaks — Muon's update is orthogonalized and spectrally normalized, so its natural LR is an order of magnitude larger than AdamW's. Mid-training must therefore scale **each group by the same multiplier**, never set both groups to one shared LR:

    ```python
    mult = wsd_decay_multiplier(mid_step, total_decay_steps)
    for opt, peak in ((muon, MUON_PEAK_LR), (adamw, ADAMW_PEAK_LR)):
        for g in opt.param_groups:
            g["lr"] = peak * mult
    ```

    Collapsing them to a single value (say, running Muon at AdamW's `3e-3`) drops the Muon group to roughly one-seventh of the LR the stable phase ran at, and you will see exactly the symptom this chapter tells you to look for: the WSD "elbow" never appears, because the decay silently started from a cliff instead of a plateau. Safest of all: do not hard-code the peaks at all — read them back out of `ckpt_stable.pt`, which stores the training config (Ch. 14.7). The checkpoint is ground truth; a number typed on this page is not.

### Why 1−sqrt, and how to watch the anneal working

The `1-sqrt` shape is the MiniCPM default for a concrete reason: it drops the LR aggressively at the *start* of decay — where the model still has enough plasticity to re-organize — then spends a long, patient tail near the floor consolidating on the premium tokens. Linear decay spreads the drop evenly and tends to leave a little loss on the table; cosine is smoother but its slow early descent wastes some of the high-plasticity window. If you only run one anneal, `1-sqrt` is the safe pick. The one knob worth ablating (Ch. 14.5) is `final_frac`: decaying to a hard `0.0` squeezes out the last fraction of loss but leaves you a checkpoint that is *finished* and awkward to continue-train; decaying to `~0.1` of peak keeps the door open for a second anneal or for post-training that prefers a non-zero starting LR. We decay to the floor for the release checkpoint and keep a pre-final checkpoint (`ckpt_mid_longctx.pt`) around as the resumable one.

You should *watch* the anneal rather than trust it. The single most informative diagnostic is a **held-out loss curve on a fixed, frozen validation set** sampled from the *pretrain* distribution, evaluated every few hundred steps throughout mid-training. Because the validation set never changes, the curve is directly comparable to the stable-phase plateau, and you will see the characteristic WSD "elbow": a visible downward bend the moment decay begins, steepening through the low-LR tail. If that elbow does *not* appear, something is wrong — the usual suspects, in order of frequency, are: the resume loaded weights but not the optimizer state; the LR multiplier is being computed against the global step instead of `mid_step`; or the two optimizer groups were collapsed onto one LR (see the warning above). A second, cheaper signal is the gradient norm: it should *shrink* smoothly as the LR falls; a rising grad norm during decay means the premium mix is too far out-of-distribution (dial back the instruction-flavored slice) or the batch is too small for the long sequences.

## Move 2 — Long-Context Extension to 8192

Stack-100M was pretrained at `max_seq_len = 2048` with RoPE base $\theta = 10000$ (see the config in [Chapter 14.4](../14-capstone/04-architecture.html)). We now want it to attend over 8192 tokens. Three things must change together, and skipping any one of them silently wastes the sub-phase: **the positional geometry** (RoPE base), **the data** (shards that actually contain positions past 2048), and **the attention mask** (a dense `(B, 1, T, T)` mask does not survive 8192). We take them in that order.

### The extrapolation problem, concretely

RoPE rotates each 2-D slice of a query/key by an angle $m\theta_k$ that grows linearly with position $m$, where $\theta_k = \theta^{-2k/d}$ for pair index $k = 0,1,\dots,d/2-1$ and $d$ is the head dimension. For Stack-100M, $d = 64$, so there are $32$ frequency pairs. The dot product $q_m \cdot k_n$ depends only on the relative offset $m - n$ — that is RoPE's defining property, derived from scratch in [Positional Encodings: Sinusoidal, Learned, RoPE & ALiBi](../02-transformer/05-positional-encoding.html).

The problem: during pretraining, $m$ never exceeded $2048$. The low-frequency pairs (large $k$, tiny $\theta_k$) barely complete a fraction of a rotation within that window. If we naively run the model at position $8191$, those slow dimensions swing into angular ranges the model has **never seen**, and attention scores go pathological. The high-frequency (local) pairs are fine — they complete many turns even within 2048 — but the long-range pairs break.

### RoPE base rescaling (NTK-aware)

The fix is to *stretch the frequency ladder* so that positions up to 8192 land in the same angular range the model already learned for positions up to 2048. NTK-aware scaling (the "increase the base" trick, popularized on LocalLLaMA by bloc97, 2023) does this by raising the RoPE base $\theta$:

$$
\theta' = \theta \cdot s^{\,d/(d-2)}, \qquad s = \frac{L_{\text{new}}}{L_{\text{old}}}
$$

A larger base makes every $\theta_k$ *smaller*, so each dimension rotates more slowly — exactly compensating for the longer positions. The $d/(d-2)$ exponent is the NTK correction that leaves the highest-frequency pair essentially untouched (preserving local resolution) while stretching the low-frequency pairs (which needed the range). YaRN (Peng et al., 2023) refines this per-wavelength and adds an attention-temperature ("length scaling") correction that divides the pre-softmax logits by a factor growing with $\log$ of the scale, compensating for the entropy growth of attention over longer contexts; we cross-link its full treatment in [Long-Context Pretraining & Context Extension](../03-pretraining/13-long-context-pretraining.html) and use the simpler base rescale here, because **we are going to continue-train** — a small amount of training at 8192 repairs any residual mismatch far more cheaply than getting the zero-shot formula perfect. (If you were extending *without* continued training — the zero-shot case — YaRN is clearly the better choice, and it is what `transformers` exposes at the config level: `rope_scaling={"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 2048}` alongside `linear`, `dynamic`, `longrope`, and `llama3` variants.)

{{fig:rope-base-rescale-frequency-ladder}}

!!! example "Worked example — rescaling Stack-100M's RoPE base for 4× context"

    Stack-100M: head dimension $d = 64$, pretrain length $L_{\text{old}} = 2048$, target $L_{\text{new}} = 8192$, original base $\theta = 10000$.

    Scale factor: $s = 8192 / 2048 = 4$.

    Exponent: $d/(d-2) = 64/62 = 1.0323$.

    $$
    \theta' = 10000 \cdot 4^{1.0323} = 10000 \cdot e^{1.0323 \cdot \ln 4} = 10000 \cdot e^{1.431} \approx 10000 \cdot 4.18 \approx \mathbf{41{,}800}.
    $$

    So we bump the RoPE base from $10000$ to about $41{,}800$ (round to $42000$). Check the slowest pair, $k = 31$, whose exponent is $2k/d = 62/64$: originally $\theta_{31} = 10000^{-62/64} \approx 1.33\times10^{-4}$ rad/token, wavelength $2\pi/\theta_{31} \approx 47{,}000$ tokens. After rescaling, $\theta'_{31} = 41800^{-62/64} \approx 3.3\times10^{-5}$ rad/token, wavelength $\approx 188{,}000$ tokens — the slow dimension now turns ~4× more slowly, so the phase at position 8192 ($8192 \cdot \theta'_{31} \approx 0.27$ rad) matches the phase the model learned at position ~2000 ($2048 \cdot \theta_{31} \approx 0.27$ rad). The fastest pair ($k=0$, $\theta_0 = \theta^0 = 1$) is exactly unchanged by any base change: local resolution is preserved.

Here is the rescale, wired to rebuild the model's RoPE cache in place. `Stack100M` holds **one** pair of `rope_cos` / `rope_sin` buffers and passes them into every block's `Attention.forward(x, cos, sin, ...)`, so there is a single cache to rebuild — `rebuild_rope` does it and updates `cfg` in the same breath, which is what makes the new geometry land in every subsequent checkpoint. Because Stack-100M uses **NoPE on every 4th layer** (SmolLM3, HuggingFace 2025; Kazemnejad et al., 2023), those layers ignore `cos`/`sin` entirely and are untouched by rescaling — they already generalize across length by construction, which is precisely why we included them.

```python
# capstone/stacklm/model/rope.py  (Ch. 14.4 — signatures reproduced for reference)
import torch

def build_rope_cache(head_dim: int, max_seq: int, theta: float,
                     device=None, dtype=torch.float32):
    """Precompute cos/sin tables of shape (max_seq, head_dim). Mid-training changes
    only `max_seq` and `theta`; the code path is identical to pretraining."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float()
                                / head_dim))                  # (head_dim/2,)
    t = torch.arange(max_seq, device=device).float()           # positions
    freqs = torch.outer(t, inv_freq)                           # (max_seq, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)                    # (max_seq, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)

def ntk_rescaled_base(base: float, head_dim: int,
                      old_len: int, new_len: int) -> float:
    """NTK-aware RoPE base rescaling: theta' = theta * s^(d/(d-2))."""
    s = new_len / old_len
    return base * (s ** (head_dim / (head_dim - 2)))
```

```python
# capstone/stacklm/mid/continue_training.py  (this chapter)
import torch
from ..model.rope import ntk_rescaled_base


def _unwrap(model):
    """`torch.compile` wraps the module; reach the real one for cfg/buffer surgery."""
    return model._orig_mod if hasattr(model, "_orig_mod") else model


@torch.no_grad()
def extend_context(model, new_seq_len: int, device="cpu") -> float:
    """Rescale RoPE base for a longer context and rebuild the cache. Returns the
    new theta. NoPE layers are unaffected (they never consult the cache).

    Call this ONCE, at the start of sub-phase B. `rebuild_rope` swaps the model's
    single (rope_cos, rope_sin) buffer pair and updates cfg.max_seq_len /
    cfg.rope_theta, so the new geometry is recorded in every later checkpoint.
    """
    m = _unwrap(model)
    cfg = m.cfg
    new_base = ntk_rescaled_base(cfg.rope_theta, cfg.head_dim,
                                 old_len=cfg.max_seq_len, new_len=new_seq_len)
    m.rebuild_rope(new_seq_len, new_base, device=device)
    return new_base
```

!!! tip "Practitioner tip: rescale once, at the boundary — and mind `torch.compile`"

    `extend_context` is called exactly once, when sub-phase B begins, and it mutates `cfg.rope_theta` and `cfg.max_seq_len` in place so the change is recorded in every subsequent checkpoint. Do **not** re-derive the base from the *original* 10000 each phase — after the first rescale the config already holds ~42000, and re-applying the $s^{d/(d-2)}$ factor to it would over-stretch the ladder. If you later want to reproduce the geometry from a checkpoint, read `rope_theta` and `max_seq_len` straight from the saved config; they are the ground truth, not the pretrain defaults.

    Two mechanical notes. First, if the model was wrapped with `torch.compile` (Ch. 14.7), `_unwrap` is mandatory before touching buffers and `state_dict` keys — everything under a compiled module is namespaced `_orig_mod.*`. Second, changing the sequence length **will** trigger a recompilation on the first 8192-token batch (Dynamo specializes on shape). That is a one-time cost of tens of seconds, but it is easiest to sidestep entirely by calling `extend_context` *before* compiling, or by accepting exactly one recompile at the A→B boundary.

### Repacking for sub-phase B: where the long documents actually come from

This is the step most write-ups skip, and it is the one that decides whether sub-phase B does anything at all.

Recall two facts from [Chapter 14.2](../14-capstone/02-data-pipeline.html). (1) The corpus was packed into **fixed 2048-token windows** (`SEQ_LEN = 2048`), so `PackedMemmapDataset("data/fineweb_edu")` physically cannot hand you an 8192-token row. (2) Every document's **position ids restart at 0** inside the packed window. Put those together and the consequence is stark:

> With per-document position resets, the largest position id the model ever sees equals the length of the longest *document*, not the length of the packed window. If your longest document is 900 tokens, rescaling RoPE for 8192 and training 0.6B tokens on it teaches the model **nothing** about positions 2048–8191. The rescale and the compute are both wasted.

So sub-phase B needs two things: shards repacked at `seq_len = 8192`, and documents long enough to fill them. The first is a re-run of Ch. 14.2's `build_shards` with a different `seq_len`. The second requires actually *sourcing* long documents, because the capstone's pretrain mix (PLAN §2) has none: Cosmopedia v2 is short-form synthetic textbook and story generations, FineWeb-Edu's document-length distribution is heavily skewed short, and FineMath samples are short by construction. Only code has natural length, and only if you stop treating a *file* as the unit.

Three sources, in decreasing order of yield:

1. **Repo-level code.** Concatenate all files within a single repository, in a deterministic order, into one "document." This is the standard trick — StarCoder2 (Lozhkov et al., 2024) and DeepSeek-Coder (Guo et al., 2024) both do repo-level pretraining precisely because it produces genuinely long, genuinely coherent sequences with real cross-file dependencies.
2. **Books.** PG-19 (Project Gutenberg books, introduced with the Compressive Transformer, Rae et al., 2019; on the Hub as `deepmind/pg19`) is public-domain, long by construction (whole books), and clean.
3. **Papers.** The arXiv subset of `EleutherAI/proof-pile-2` (Llemma, Azerbayev et al., 2024) gives long, math-dense documents — a nice double-duty with sub-phase C.

Plus a **length filter** over the sources you already have, which will recover *some* long FineWeb-Edu documents. Do not guess the yield; measure it on your own shards, because it depends entirely on your dump and your filter settings.

```python
# capstone/scripts/repack_long.py
"""Build the sub-phase-B shards: seq_len=8192, long documents only.

Run once, between sub-phase A and sub-phase B:
    python capstone/scripts/repack_long.py --out data/mid --seq-len 8192
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from stacklm.data import DataMixEntry, build_shards, stream_source    # Ch. 14.2
from stacklm.tokenizer import StackTokenizer                          # Ch. 14.3

MIN_DOC_TOKENS = 4096          # half the target window; see the assertion below


def length_filtered(docs, tok, min_tokens: int = MIN_DOC_TOKENS):
    """Keep only documents that can actually exercise positions past 2048.

    Tokenizing twice (here and in `pack_documents`) is wasteful; at 20B tokens you
    would instead carry a `n_tokens` field through the Ch. 14.2 pipeline, or use
    HuggingFace `datatrove`'s TokensCounter + a LambdaFilter to do it in one pass.
    """
    for doc in docs:
        if len(tok.encode(doc["text"])) >= min_tokens:
            yield doc


def repo_level_documents(files, sep: str = "\n\n# ==== file: {path} ====\n\n"):
    """Concatenate a repository's files into ONE document (StarCoder2 / DeepSeek-Coder).

    `files` is a stream of dicts with `repo_name`, `path`, `content`. Sorting by
    path makes the concatenation deterministic (and puts headers near sources,
    which is what a human reading the repo would do).
    """
    by_repo = defaultdict(list)
    for f in files:
        by_repo[f["repo_name"]].append(f)
    for repo, fs in by_repo.items():
        fs.sort(key=lambda f: f["path"])
        body = "".join(sep.format(path=f["path"]) + f["content"] for f in fs)
        yield {"text": body, "source": "starcoder_repo", "repo": repo}


# The sub-phase-B sources, as Ch. 14.2 `DataMixEntry` records (name, hf_path,
# weight, domain). The first three are genuinely long; the last two are
# length-FILTERED slices of the pretrain sources. `weight` is the sub-phase-B
# mixture weight consumed later by `build_mixture_loader`.
LONG_SOURCES = [
    (DataMixEntry("starcoder_repo",   "bigcode/starcoderdata",     0.35, "code"),  True),
    (DataMixEntry("books_pg19",       "deepmind/pg19",             0.25, "web"),   False),
    (DataMixEntry("arxiv_proofpile2", "EleutherAI/proof-pile-2",   0.15, "math"),  False),
    (DataMixEntry("fineweb_edu_long", "HuggingFaceFW/fineweb-edu", 0.15, "web"),   False),
    (DataMixEntry("cosmopedia_v2",    "HuggingFaceTB/cosmopedia",  0.10, "synthetic"), False),
]


def main(out_root: str, seq_len: int):
    tok = StackTokenizer.load("artifacts/tokenizer.json")     # Ch. 14.3, vocab 32768
    for entry, repo_level in LONG_SOURCES:
        raw = stream_source(entry)                            # Ch. 14.2 streaming reader
        docs = repo_level_documents(raw) if repo_level else raw
        if entry.name != "cosmopedia_v2":     # the short-form anchor stays unfiltered
            docs = length_filtered(docs, tok)
        out = f"{out_root}/{entry.name}_{seq_len}"
        n = build_shards(docs, tok, out, seq_len=seq_len,
                         tokens_per_shard=100_000_000)
        verify_positions(out, seq_len,
                         floor=4096 if entry.name != "cosmopedia_v2" else 0)
        print(f"{entry.name}: {n} shard(s) -> {out}")


def verify_positions(shard_dir: str, seq_len: int, floor: int = 4096):
    """The one-line check that decides whether sub-phase B is real or theatre."""
    hi = 0
    for p in sorted(Path(shard_dir).glob("shard_*.pos.bin")):
        hi = max(hi, int(np.memmap(p, dtype=np.uint16, mode="r").max()))
    print(f"  max position id in {shard_dir}: {hi} (window {seq_len})")
    assert hi > floor, (
        f"{shard_dir} contains no document longer than {floor} tokens: RoPE "
        f"rescaling would train on positions the data never reaches.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/mid")
    ap.add_argument("--seq-len", type=int, default=8192)
    main(ap.parse_args().out, ap.parse_args().seq_len)
```

`uint16` still suffices: the largest position id is `8191 < 65535`, so the on-disk format from Ch. 14.2 is unchanged.

With those shards in hand, sub-phase B's mixture is honest about which sources are long:

| Source | Sub-phase B weight | Genuinely long? |
|---|---|---|
| `starcoder_repo` (repo-level concatenation) | 35% | yes — whole repositories |
| `books_pg19` (Project Gutenberg) | 25% | yes — whole books |
| `arxiv_proofpile2` (arXiv slice) | 15% | yes — full papers |
| `fineweb_edu_long` (filtered ≥ 4096 tokens) | 15% | yes, by construction of the filter |
| `cosmopedia_v2` (short-form, unfiltered) | 10% | **no** — deliberately short |

That last row is not an oversight. Training *only* on long documents is a known way to degrade short-context quality, and both Llama 3's long-context stage (Grattafiori et al., 2024) and ProLong (Gao et al., Princeton, 2024) explicitly keep short-form data mixed in while upsampling long documents. The 10% Cosmopedia slice is the anchor that keeps the model from drifting off the distribution sub-phase A just spent 1.2B tokens sharpening. If you read one paper alongside this section, make it ProLong: it is the closest open, reproducible reference for exactly this step — data recipe, length schedule, and evaluation.

!!! note "Aside: what if you genuinely have no long documents?"

    Sometimes you cannot source them (proprietary domain, tiny corpus). Two honest fallbacks, both worse than real long data, both better than pretending:

    - **Relax the position reset for concatenated windows.** Pack related short documents contiguously — same repository, same book chapter, same topic cluster — and let the position clock run across them without resetting. Attention is still document-masked, but RoPE now sees positions up to 8191. You get positional coverage without semantic long-range dependency, which repairs the *geometry* but not the *capability*.
    - **Shrink the target.** Extend to 4096 instead of 8192. A 4× context you can actually train is worth more than a 4× context you can only claim.

    What does *not* work is rescaling RoPE, training on 2048-token-max documents, and reporting an 8192 context window. That is the failure mode the `verify_positions` assertion exists to catch.

### Masking at 8192: FlexAttention and varlen kernels

Now the systems half. Stack-100M implements document-aware attention by materializing a boolean mask:

```python
# capstone/stacklm/model/transformer.py  (Ch. 14.4 — the PRETRAIN-scale path)
def _attn_mask(self, seq_ids, T, device):
    if seq_ids is None:
        return None
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
    same = seq_ids[:, :, None] == seq_ids[:, None, :]      # (B, T, T)
    return (causal[None] & same).unsqueeze(1)              # (B, 1, T, T)
```

At `micro_bs = 32`, `T = 2048` that mask is $32 \times 2048^2 = 1.34\times10^8$ bools = **134 MB** — annoying but survivable. At `micro_bs = 8`, `T = 8192` it is $8 \times 8192^2 = 5.37\times10^8$ bools = **537 MB**, allocated fresh on *every* forward, and it must persist through the backward pass. Worse, passing any dense `attn_mask` to `F.scaled_dot_product_attention` **disqualifies the FlashAttention backend** — PyTorch falls back to the memory-efficient or math kernel, so you simultaneously lose the speed and lose the "never materializes the $T \times T$ score matrix" property that made 8192 affordable in the first place. Chapter 14.2 flagged this as a promise to cash later; here is where it comes due.

There are two production answers in 2026, and you should know both.

**Option 1 — FlexAttention (PyTorch-native).** `torch.nn.attention.flex_attention` (PyTorch 2.5+) lets you express the mask as a *predicate* on indices, `mask_mod(b, h, q_idx, kv_idx) -> bool`, and compiles it into a fused, block-sparse Triton kernel. `create_block_mask` evaluates the predicate once per $128 \times 128$ block and stores only which blocks are non-empty — so memory goes from $O(T^2)$ to $O((T/128)^2)$, and *fully masked blocks are never computed at all*. Document masking is the canonical example, and it makes short documents **faster**, not slower.

```python
# capstone/stacklm/model/doc_attention.py
import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# Compile once at import; `dynamic=False` keeps one specialization per shape.
flex_attention = torch.compile(flex_attention, dynamic=False)
_create_block_mask = torch.compile(create_block_mask, dynamic=False)


def document_block_mask(seq_ids: torch.Tensor):
    """Block-sparse causal + intra-document mask from Ch. 14.2's `seq_ids`.

    seq_ids : (B, T) int tensor; tokens of the same packed document share an id.
    Returns a BlockMask consumable by flex_attention. Broadcast over heads (H=None).
    """
    B, T = seq_ids.shape

    def mask_mod(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx                       # no peeking ahead
        same_doc = seq_ids[b, q_idx] == seq_ids[b, kv_idx]
        return causal & same_doc

    return _create_block_mask(mask_mod, B=B, H=None, Q_LEN=T, KV_LEN=T,
                              device=seq_ids.device)


def doc_attention(q, k, v, block_mask):
    """q, k, v: (B, n_heads, T, head_dim) -- GQA heads already expanded (Ch. 14.4).
    Scaling is 1/sqrt(head_dim) by default, matching SDPA."""
    return flex_attention(q, k, v, block_mask=block_mask)
```

Wiring it in is a two-line change to `Attention.forward`: build the `BlockMask` **once per micro-batch** in `Stack100M.forward` (not once per layer — that is 30× the mask-construction cost for an identical result) and thread it down in place of `attn_mask`, then call `doc_attention` instead of `F.scaled_dot_product_attention`. At `T = 8192` the block mask is $64 \times 64 = 4096$ block entries per batch element — kilobytes instead of half a gigabyte.

**Option 2 — varlen FlashAttention.** The `flash-attn` library (Dao et al.) exposes `flash_attn_varlen_func`, which takes *unpadded* concatenated tokens plus a cumulative-sequence-length index and runs a separate causal attention per document, never computing a cross-document score at all. This is what Megatron-LM and most production stacks use for packed pretraining.

```python
# Requires: pip install flash-attn --no-build-isolation   (CUDA, Ampere or newer)
import torch
from flash_attn import flash_attn_varlen_func

def varlen_attention(q, k, v, doc_lens: torch.Tensor):
    """q, k, v: (total_tokens, n_heads, head_dim) -- the batch is FLATTENED and
    unpadded. doc_lens: (n_docs,) int32 lengths, summing to total_tokens.

    cu_seqlens is the standard 'cumulative sequence lengths' index: the exclusive
    prefix sum with a leading 0, i.e. document i occupies [cu[i], cu[i+1]).
    """
    cu = torch.cat([torch.zeros(1, dtype=torch.int32, device=doc_lens.device),
                    doc_lens.cumsum(0).to(torch.int32)])
    max_len = int(doc_lens.max())
    return flash_attn_varlen_func(q, k, v,
                                  cu_seqlens_q=cu, cu_seqlens_k=cu,
                                  max_seqlen_q=max_len, max_seqlen_k=max_len,
                                  causal=True)
```

Which to pick? **FlexAttention**, for this capstone: it is PyTorch-native (no extra CUDA build), it composes with `torch.compile` and with the rest of Stack-100M unchanged, and it keeps the `(B, T)` batched layout the Ch. 14.2 dataset already produces. Varlen FlashAttention is marginally faster and is the right answer inside a framework that already flattens batches (Megatron-LM, and xFormers' `BlockDiagonalCausalMask.from_seqlens` for the same idea with a different API) — but adopting it means rewriting the data path to emit `cu_seqlens`. Use it when you graduate to a multi-GPU framework; use FlexAttention today.

!!! warning "Common pitfall: the memory *and* the kernel, at 8192"

    Attention memory and compute are quadratic in sequence length. Going 2048 → 8192 is 4× longer, so the score matrix is **16×** larger. Three things must line up before you launch:

    - **Micro-batch.** Cut it ~4× (32 → 8) and raise gradient accumulation to hold the ~0.5M-token global batch fixed (Ch. 14.6). Exercise 5 works the arithmetic.
    - **Mask.** Use `FlexAttention` (or varlen), never a dense `(B, 1, T, T)` bool — 537 MB per micro-batch *and* a silent fallback off the [FlashAttention](../04-kernels-efficiency/02-flash-attention-1.html) backend.
    - **Activations.** GQA with 2 KV heads (Ch. 14.4) already shrinks the KV cache 4×; add [activation checkpointing](../04-kernels-efficiency/10-memory-efficient-training.html) on the 30 blocks if you are still tight.

    Do the arithmetic *before* you launch, not after the OOM. And note what is *not* needed at this scale: sequence/context parallelism. Ring Attention (Liu et al., 2023), DeepSpeed-Ulysses (Jacobs et al., 2023), and Megatron-LM's context parallelism exist for the regime where a single sequence's activations do not fit on one device — 8192 tokens at `d_model=512` is nowhere near that. Know they exist; do not reach for them here.

{{fig:seqlen-quadratic-attention-budget}}

### Validating that the long-context actually works

Perplexity averaged over a sequence can *fall* even when the model still cannot use position 8000 — a couple of well-predicted local tokens hide a broken tail. Two cheap probes catch this.

First, **loss-versus-position**: bin the per-token loss by its position within the 8192-window and plot the curve. A healthy extension shows loss that keeps *decreasing* (or at worst flattening) as position grows — the model is using more context to predict better. A curve that *rises* past ~2048 means the rescale-plus-training has not taken and the far positions are still noise. Crucially, run this on the **long-document subset only**: on document-masked packed windows full of short documents, position within the window is not position within the document, and the curve tells you nothing.

Second, a tiny **needle-in-a-haystack** check: plant a short unique fact ("the passcode is 4713") at a random depth in a long filler document and ask the model, via next-token prediction, to complete "the passcode is". If it recovers the needle only when it sits in the first 2048 tokens, you have geometry without capability — train sub-phase B longer or on longer documents.

The 2026 standard long-context benchmarks are **RULER** (Hsieh et al., NVIDIA, 2024), which generalizes needle-in-a-haystack into 13 synthetic tasks with controllable length, and **HELMET** (Yen et al., Princeton, 2024–2025), which evaluates on realistic downstream long-context applications. Both are the right tools for a frontier model and both will read near floor for a 100M model — quote them as the destination, not as evidence. The honest, full evaluation belongs in [Chapter 14.11](../14-capstone/11-evaluation-and-serving.html); loss-versus-position and the needle probe are the two you run *during* mid-training to know the elbow is real.

{{fig:long-context-loss-vs-position-validation}}

## Move 3 — Capability Injection

The final sub-phase (C) runs at the LR floor and pushes the mixture hard toward the capabilities our downstream narrow agent needs: arithmetic, symbolic manipulation, and code structure.

| Source | Sub-phase C weight |
|---|---|
| FineMath / OpenWebMath | 30% |
| StarCoder (repo-level, from sub-phase B's shards) | 30% |
| Cosmopedia v2 (STEM-heavy slice) | 25% |
| FineWeb-Edu | 15% |

This is a small budget (~0.2B tokens) at a low, still-nonzero LR — enough to sharpen number-and-symbol handling without over-fitting or wrecking general fluency. It is honest to call this what it is: **not** turning Stack-100M into a math model, but giving a 100M model *just enough* arithmetic and code grounding that the RLVR run in [Chapter 14.9](../14-capstone/09-post-training.html) (narrow GRPO on verifiable integer arithmetic, following GRPO from DeepSeekMath, Shao et al., 2024) and the ReAct agent (Yao et al., 2022) in [Chapter 14.10](../14-capstone/10-agentic-narrow.html) have a base to reinforce. At 100M params you cannot inject a capability the model has no capacity for; you can only make sure the capacity you have is pointed at the right target. That is "narrow but real."

Capability injection reuses the exact same loop and schedule — it is simply sub-phase C's mixture, applied while the LR finishes its decay to the floor. Because it runs at `seq_len = 8192` on the sub-phase-B shards, it needs no new repack: only a mixture dict.

### Why this ordering

The three moves are sequenced deliberately, and the order is not arbitrary. Annealing runs *first*, at 2048, because the general quality jump wants the highest LR of the decay leg and the largest, most diverse token budget — you consolidate broad representations before you specialize. Long-context extension runs *second* because it is a targeted change to a subsystem (the positional geometry) that benefits from a settled model: rescaling RoPE on a mid-anneal checkpoint whose attention patterns have already sharpened means the few hundred million long-context tokens teach the heads to *use* the new range rather than fighting a still-shifting representation. Capability injection runs *last*, at the LR floor, because it is the narrowest and most over-fitting-prone mix; keeping it in the low-LR tail limits how far it can pull the model off the general manifold while still committing arithmetic and code structure. Reverse any two of these and you either waste the high-plasticity window on narrow data or extend context on a model that is still moving. A useful mental model: **broad → long → narrow**, riding one decreasing learning rate down.

## The Mid-Training Loop: One Continuous Decay

Now we assemble everything into a single continued-training routine that resumes from `ckpt_stable.pt` and runs all three sub-phases back to back. It reuses `Stack100M`, the Muon+AdamW hybrid optimizer, and the checkpoint helpers established in Chapters 14.4, 14.6, and 14.7 — mid-training adds *no new model code*, only orchestration.

The driver below is the shipped `run_mid_training` in `capstone/stacklm/mid/continue_training.py`.

```python
# capstone/stacklm/mid/continue_training.py  (the phase driver)
from dataclasses import dataclass, field

import torch

from ..optim import build_optimizers                   # Muon+AdamW hybrid, Ch. 14.6
from ..optim.schedule import wsd_decay_multiplier      # Ch. 14.6
from ..train.loop import autocast_ctx                  # bf16 on CUDA, fp32 on CPU


@dataclass
class SubPhase:
    """One mid-training sub-phase. `steps` overrides the token-budget arithmetic
    (used by CI, which runs a handful of steps instead of ~1e9 tokens)."""
    name: str
    tokens: int
    seq_len: int
    mix: dict = field(default_factory=dict)
    extend_now: bool = False       # rescale RoPE at the start of this sub-phase
    steps: int = 0                 # 0 => derive from `tokens`


def steps_for(sub: SubPhase, global_batch_tokens: int) -> int:
    return sub.steps or (sub.tokens // global_batch_tokens)


def run_mid_training(model, phases, loader_fn, *, device="cpu",
                     global_batch_tokens=512_000, micro_batch_tokens=65_536,
                     muon_lr=0.02, adamw_lr=3e-3, grad_clip=1.0,
                     optimizers=None, use_seq_ids=True, log_every=50, seed=1234,
                     checkpoint_fn=None):
    """Walk `phases` back to back under ONE WSD decay leg.

    `loader_fn(sub, micro_batch_size)` returns an *infinite* iterator of batch
    dicts (`input_ids`, `targets`, `seq_ids`) at `sub.seq_len`.
    Pass `optimizers=[muon, adamw]` to CONTINUE the stable phase's optimizer state
    (Muon momentum, AdamW moments) restored from `ckpt_stable.pt`.
    """
    torch.manual_seed(seed)
    device = torch.device(device)
    model.to(device).train()

    if optimizers is None:
        muon, adamw = build_optimizers(model, muon_lr=muon_lr, adamw_lr=adamw_lr)
        optimizers = [muon, adamw]
    else:
        muon, adamw = optimizers[0], optimizers[-1]
    peaks = {id(muon): muon_lr, id(adamw): adamw_lr}     # two groups, two peaks

    # The decay spans ALL sub-phases: it is never reset at a boundary.
    total_decay_steps = sum(steps_for(p, global_batch_tokens) for p in phases)
    mid_step, history = 0, []

    for sub in phases:
        if sub.extend_now:
            new_base = extend_context(model, sub.seq_len, device=device)
            print(f"  [{sub.name}] RoPE base -> {new_base:.0f}, seq_len -> {sub.seq_len}")

        # Micro-batch shrinks with sequence length to hold activation memory
        # roughly constant; grad-accum makes up the global batch.
        micro_bs = max(1, micro_batch_tokens // sub.seq_len)
        accum = max(1, (global_batch_tokens // sub.seq_len) // micro_bs)
        it = loader_fn(sub, micro_bs)

        for _ in range(steps_for(sub, global_batch_tokens)):
            mult = wsd_decay_multiplier(mid_step, total_decay_steps)
            for opt in optimizers:
                for g in opt.param_groups:
                    g["lr"] = peaks[id(opt)] * mult      # SAME multiplier, own peak
                opt.zero_grad(set_to_none=True)

            loss_acc = 0.0
            for _micro in range(accum):
                batch = {k: v.to(device) for k, v in next(it).items()}
                with autocast_ctx(device):
                    # z-loss is inside the model (cfg.z_loss_coef, Ch. 14.4) and
                    # stays on through the low-LR tail: it keeps logits scaled.
                    _, loss = model(batch["input_ids"], targets=batch["targets"],
                                    seq_ids=batch["seq_ids"] if use_seq_ids else None)
                    loss = loss / accum
                loss.backward()
                loss_acc += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            for opt in optimizers:
                opt.step()
            history.append(loss_acc)
            if log_every and mid_step % log_every == 0:
                print(f"  [{sub.name}] mid_step {mid_step:>5} loss {loss_acc:.4f} "
                      f"lr_muon {peaks[id(muon)] * mult:.2e} seq {sub.seq_len}")
            mid_step += 1

        if checkpoint_fn is not None:
            checkpoint_fn(sub.name, model, optimizers, mid_step)

    return {"loss_history": history, "mid_steps": mid_step,
            "total_decay_steps": total_decay_steps}
```

And the full-run entry point that supplies the three sub-phases, the mixtures, and the resume:

```python
# capstone/scripts/midtrain.py
"""Full-run mid-training: resume ckpt_stable.pt and run the WSD decay across
anneal @2048 -> long-context @8192 -> capability injection @8192.

    python capstone/scripts/midtrain.py \
        --stable artifacts/ckpt_stable.pt --out artifacts/ --device cuda
"""
import argparse
from dataclasses import asdict

from stacklm.config import StackConfig
from stacklm.model import Stack100M
from stacklm.optim import build_optimizers
from stacklm.train.loop import load_checkpoint, save_checkpoint     # Ch. 14.7
from stacklm.mid import SubPhase, run_mid_training
from stacklm.mid.mixture import (build_mixture_loader, ANNEAL_MIX,
                                 LONGCTX_MIX, CAPABILITY_MIX)

GLOBAL_BATCH_TOKENS = 512_000    # ~0.5M-token batch, same as pretrain (Ch. 14.6)
MICRO_BATCH_TOKENS  = 65_536     # 32 x 2048 at pretrain length; 8 x 8192 when long
MUON_PEAK_LR        = 0.02       # Muon group's stable-phase peak (Ch. 14.6)
ADAMW_PEAK_LR       = 3e-3       # AdamW group's stable-phase peak (Ch. 14.6)

# The three moves of mid-training, in order. Token budgets are illustrative
# (~2B total = ~10% of the 20B pretrain budget); tune per Ch. 14.5.
PHASES = [
    SubPhase("anneal", tokens=1_200_000_000, seq_len=2048, mix=ANNEAL_MIX),
    SubPhase("longctx", tokens=600_000_000, seq_len=8192, extend_now=True,
             mix=LONGCTX_MIX),
    SubPhase("capability", tokens=200_000_000, seq_len=8192, mix=CAPABILITY_MIX),
]


def main(stable_ckpt: str, out_dir: str, device: str):
    # ---- 1. Resume the *pre-decay* model AND optimizer state ------------------
    model = Stack100M(StackConfig()).to(device)
    optimizers = list(build_optimizers(model, muon_lr=MUON_PEAK_LR,
                                       adamw_lr=ADAMW_PEAK_LR))   # (muon, adamw)
    step, extra = load_checkpoint(stable_ckpt, model, optimizers,
                                  map_location=device)
    # `extra` is the payload Ch. 14.7 stores alongside the tensors: tokens_seen,
    # the PackedDataset cursor, and the training config. Trust it over this file.
    muon_peak = extra.get("muon_lr", MUON_PEAK_LR)
    adamw_peak = extra.get("adamw_lr", ADAMW_PEAK_LR)
    print(f"resumed {stable_ckpt} @ global step {step} "
          f"({extra.get('tokens_seen', 0)/1e9:.1f}B tokens, LR still at peak); "
          f"peaks muon={muon_peak} adamw={adamw_peak}")

    # ---- 2. One loader factory per sub-phase, from that sub-phase's mixture ---
    def loader_fn(sub, micro_bs):
        return build_mixture_loader(sub.mix, sub.seq_len, micro_bs,
                                    root="data/mid", seed=1234 + len(sub.name))

    def checkpoint_fn(name, model, opts, mid_step):
        save_checkpoint(f"{out_dir}/ckpt_mid_{name}.pt", model, opts,
                        step=step + mid_step,
                        extra={**extra, "phase": name,
                               "cfg": asdict(model.cfg),   # records the NEW rope geometry
                               "mid_step": mid_step})
        print(f"[{name}] done -> ckpt_mid_{name}.pt")

    # ---- 3. Walk the three sub-phases under one continuous decay --------------
    result = run_mid_training(
        model, PHASES, loader_fn, device=device,
        global_batch_tokens=GLOBAL_BATCH_TOKENS,
        micro_batch_tokens=MICRO_BATCH_TOKENS,
        muon_lr=muon_peak, adamw_lr=adamw_peak,
        optimizers=optimizers,          # CONTINUE Muon momentum + AdamW moments
        log_every=50, checkpoint_fn=checkpoint_fn)

    save_checkpoint(f"{out_dir}/ckpt_mid_final.pt", model, optimizers,
                    step=step + result["mid_steps"],
                    extra={**extra, "cfg": asdict(model.cfg)})
    print(f"mid-training complete: {result['mid_steps']} steps "
          f"-> ckpt_mid_final.pt (ready for SFT in Ch. 14.9)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stable", default="artifacts/ckpt_stable.pt")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    main(a.stable, a.out, a.device)
```

A few design points worth flagging:

- **One decay across three phases.** `total_decay_steps` spans all sub-phases and `mid_step` runs continuously, so the LR falls smoothly from peak to floor *through* the sequence-length change and the mixture changes. We do not reset the schedule at each boundary — that would create three little decay cliffs instead of one clean anneal. This is the single most important line in the file.
- **Two peaks, one multiplier.** `peaks[id(opt)] * mult` scales Muon and AdamW by the same schedule while preserving the order-of-magnitude gap between their natural learning rates.
- **We resume the optimizer state, not just the weights.** Muon's momentum buffers and AdamW's moments carry over from the stable phase — which is why `optimizers` is built *before* `load_checkpoint` and passed *into* `run_mid_training` rather than being rebuilt inside it. Throwing that state away injects a transient the fresh decay does not need. This is the concrete payoff of saving `ckpt_stable.pt` *with* optimizer state in Ch. 14.7.
- **Micro-batch shrinks, global batch holds.** At 2048, `micro_bs = 32` and `accum = 250 // 32 = 7`; at 8192, `micro_bs = 8` and `accum = 62 // 8 = 7`. Both yield ~0.46M tokens per optimizer step, so the LR/batch relationship (Ch. 3.10) is unchanged across the sequence-length jump.
- **z-loss stays on.** The small `logsumexp` penalty lives inside `Stack100M.forward` via `cfg.z_loss_coef` (Ch. 14.4); it keeps logits well-scaled through the low-LR tail, and there is nothing to re-enable.
- **`seq_ids` are threaded through every forward.** Document-aware masking is not optional and does not lapse at mid-training; it is exactly as load-bearing at 8192 as at 2048 — more so, because a packed 8192-window holds four times as many unrelated documents.

### What CI actually covers

Per the PLAN's deliverable standard, `capstone/smoke_test.py` exercises this code at toy scale, CPU-only and hermetic — a 2-layer `d_model=64` model, an in-process synthetic corpus, and a handful of steps. Concretely it runs, in this order:

1. `mid_train(model, ds, steps=4, micro_batch_size=4, grad_accum=2, peak_lr=1.5e-3, extend_to=2*max_seq_len)` — the single-phase entry point — and asserts the losses are finite and `model.cfg.max_seq_len` doubled.
2. `run_mid_training(model, phases, toy_loader, ...)` over three `SubPhase`s (`anneal` → `longctx` with `extend_now=True` → `capability`), asserting that `mid_steps == total_decay_steps` (the decay was never reset), that all three boundary checkpoints fired in order, that `cfg.max_seq_len` grew again, and that `cfg.rope_theta` strictly *increased* — the machine-checkable statement of "we rescaled the base, once, upward."

What CI does **not** cover, and cannot: the real 8192 shapes, the FlexAttention kernel (CUDA-only), and the actual data mixtures. Those are documented here with full-run magnitudes, exactly as the PLAN's "CI proves it *runs*; the prose documents the expected numbers" standard requires. Do not read a green smoke test as evidence that your long-context extension worked — that is what `verify_positions` and the loss-versus-position curve are for.

## What Mid-Training Costs and Buys

!!! example "Worked example — the compute bill, with the attention term you cannot drop"

    The usual **6ND** shortcut (Ch. 3.4) counts only the dense parameter FLOPs. It is a fine approximation for a wide, short-context model — and a bad one for Stack-100M, which is deliberately **deep-and-thin** (`d_model = 512`, `n_layers = 30`) and about to run at 8192. The attention term is

    $$
    \text{FLOPs/token}_{\text{attn}} \approx 6 \cdot n_{\text{layers}} \cdot d_{\text{model}} \cdot T
    $$

    for a *causal* kernel (the $12\,n_L d T$ dense-attention figure, halved because FlashAttention skips the masked upper triangle). With $N = 101.4\text{M}$, the dense term is $6N = 6.08\times10^{8}$ FLOPs/token, and:

    | | $T$ | attn FLOPs/token | total FLOPs/token | attn share |
    |---|---|---|---|---|
    | pretrain / sub-phase A | 2048 | $1.89\times10^{8}$ | $7.97\times10^{8}$ | 24% |
    | sub-phases B, C | 8192 | $7.55\times10^{8}$ | $1.36\times10^{9}$ | **55%** |

    At 8192, attention is *more than half* of all training FLOPs, and the 6ND estimate is off by 2.2×. Now the bill, at ~40% MFU on the flagship 1×A100 (80GB) — bf16 peak 312 TFLOP/s, so ~125 TFLOP/s achieved:

    $$
    C_A = 7.97\times10^{8} \times 1.2\times10^{9} = 9.57\times10^{17},\quad
    C_B = 1.36\times10^{9} \times 6\times10^{8} = 8.18\times10^{17},
    $$

    $$
    C_C = 1.36\times10^{9} \times 2\times10^{8} = 2.73\times10^{17}
    \;\Longrightarrow\; C_{\text{total}} \approx 2.05\times10^{18}\ \text{FLOPs}.
    $$

    $$
    t = \frac{2.05\times10^{18}}{1.25\times10^{14}} \approx 1.64\times10^{4}\ \text{s} \approx \mathbf{4.6\ GPU\text{-}hours}.
    $$

    At ~USD 1–2/GPU-hr that is roughly **USD 5–9**. The 6ND-only estimate would have predicted 2.7 GPU-hours — a **1.7× miss**, and the single easiest place in this project to blow a compute forecast. Two honest caveats in *both* directions: document-block-diagonal masking makes the attention term *smaller* than the table says whenever the packed window holds short documents (FlexAttention skips fully-masked blocks entirely), while an MFU below 40% — likely at 8192 before you tune the micro-batch — makes the wall-clock larger. Budget ~5 GPU-hours and measure.

    Even at the corrected number this is a small slice of the ~15–25 GPU-hour, ~USD 40–100 total project budget, and for it you get the sharpest single quality jump in the run (the decay-phase drop), a 4× context window, and a math/code floor. It remains the best marginal return on compute anywhere in the pipeline — which is exactly why mid-training is worth its own chapter.

You should expect the held-out loss to fall visibly across sub-phase A — on the order of a couple tenths of a nat below where the stable phase plateaued — with most of the drop concentrated in the low-LR tail. Long-context sub-phase B will *raise* the average loss slightly (8192-token prediction on books and whole repositories is genuinely harder than 2048-token snippets, and the mix itself changed), which is expected and correct, not a regression; the loss-versus-position curve is the metric that shows the extension worked even as the scalar average ticks up. Capability sub-phase C nudges arithmetic and code perplexity down at the cost of a hair of general-web perplexity — the trade we are deliberately making. Report these as *illustrative* movements; never quote a fabricated benchmark. The honest evaluation lives in [Chapter 14.11](../14-capstone/11-evaluation-and-serving.html).

!!! interview "Interview Corner"

    **Q:** Why do practitioners run learning-rate *decay* on a different (higher-quality) data mixture than the stable phase, and why not just use that better mixture for the whole run?

    **A:** Two reasons, both economic. First, in a WSD schedule most of the *committed* loss reduction happens during the short decay phase — the model consolidates representations it explored loosely at high LR. So the data seen during decay disproportionately shapes the final model; putting your cleanest, densest, most task-relevant tokens there gives the largest quality-per-token return. Second, premium data (synthetic textbooks, curated math/code) is scarce and expensive per token, and high-quality-but-narrow data used for the *entire* run can hurt breadth and diversity. Annealing captures most of the upside for a few percent of the token budget while the long stable phase keeps broad coverage. It is a curriculum: broad-and-cheap to build general representations, then narrow-and-premium to sharpen and commit them. The same logic explains why you extend context and inject capabilities *late* — you pay for those only over the tokens where they matter, not across all 20B.

    **Q:** You extend a model from 2K to 8K by rescaling the RoPE base and continuing training. The loss goes down, but a needle-in-a-haystack probe still fails past 2K. Name the two most likely causes and how you would confirm each in under ten minutes.

    **A:** *(1) The data never reaches those positions.* If packing resets position ids per document and every document is short, the largest position id in the shards may still be ~1–2K, so the extension is geometry with nothing to train it. Confirm by memory-mapping the `.pos.bin` shards and printing `max()` — one line, seconds — and assert it exceeds half the target window. *(2) The mask silently changed the kernel or the semantics.* Passing a dense `(B,1,T,T)` mask to `scaled_dot_product_attention` disqualifies the FlashAttention backend and costs half a gigabyte per micro-batch at 8K; conversely, dropping `seq_ids` entirely re-enables cross-document attention, so the model learns to attend across boundaries that will not exist at inference. Confirm by printing which SDPA backend ran (`torch.backends.cuda.sdp_kernel` / the profiler's kernel names) and by asserting `seq_ids` is non-`None` in the training loop. A third, cheaper thing to rule out first: that you re-applied the $s^{d/(d-2)}$ rescale to an already-rescaled base, over-stretching the ladder — check `cfg.rope_theta` in the checkpoint is ~42000, not ~175000.

## Key Takeaways

!!! key "Key Takeaways"

    - **Mid-training is the phase between pretraining and post-training** (OLMo 2): still self-supervised next-token prediction, but on upgraded data, at longer context, with concentrated capabilities. It resumes from a *pre-decay* stable checkpoint — never a fully-decayed one.
    - **The WSD decay phase is where you spend your best data.** Most committed loss reduction happens during decay, so annealing on a premium mix (more Cosmopedia, math, code, instruction-flavored text) buys a large quality jump for ~10% of the token budget.
    - **Run one continuous decay across all sub-phases, scaling each optimizer group by the same multiplier.** Muon (0.02) and AdamW (3e-3) have different peaks; apply `wsd_decay_multiplier` to each, and never reset the schedule at a sub-phase boundary or you create decay cliffs.
    - **Long-context extension is three changes, not one.** Rescale the RoPE base with the NTK rule $\theta' = \theta\, s^{d/(d-2)}$ (10000 → ~42000 for 2048→8192, $d{=}64$); **repack shards at 8192 from genuinely long documents**; and **swap the dense mask for FlexAttention or varlen FlashAttention**. Skip any one and the sub-phase is theatre.
    - **Check the data before you launch.** With per-document position resets, the largest position the model ever sees is the longest *document*. `assert max(position_ids) > 4096` on the sub-phase-B shards is the cheapest bug-catch in the chapter.
    - **Long documents must be sourced, not assumed.** Repo-level StarCoder concatenation, PG-19 books, and arXiv from proof-pile-2 supply real length; keep ~10% short-form data mixed in (ProLong, Llama 3) so short-context quality does not drift.
    - **Validate with loss-versus-position and a needle probe**, on the long-document subset. A falling scalar perplexity can hide a broken tail; RULER and HELMET are the real benchmarks but will read near floor at 100M.
    - **Count the attention FLOPs.** For a deep-and-thin model at 8192, attention is ~55% of training FLOPs; 6ND alone under-predicts mid-training by ~1.7×. Budget ~4.6 GPU-hours (~USD 5–9), not ~2.7.
    - **Capability injection is honest, not magic.** Concentrated math/code at the LR floor gives a 100M model an arithmetic-and-structure floor for the downstream narrow agent — it points existing capacity at the target; it cannot create capacity that is not there.
    - **Resume optimizer state, keep z-loss, thread `seq_ids`, checkpoint every boundary.** Mid-training adds orchestration, not model code; reuse `Stack100M`, the Muon+AdamW hybrid, and the Ch. 14.7 checkpoint helpers unchanged.

!!! sota "State of the Art & Resources (2026)"
    WSD-style annealing and RoPE-rescale-plus-continue-train are now the default recipe for squeezing a quality jump and a long-context window out of a small model's last few percent of tokens — the open reports below (MiniCPM, OLMo 2, SmolLM3, ProLong) all converge on the same broad-then-narrow, short-then-long shape this chapter walks through. The systems half has converged too: block-sparse document masking via FlexAttention or varlen FlashAttention is now the standard way to train on packed long sequences.

    **Foundational work**

    - [Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021)](https://arxiv.org/abs/2104.09864) — the original RoPE derivation this chapter's frequency-ladder math builds on.
    - [Kazemnejad et al., *The Impact of Positional Encoding on Length Generalization in Transformers* (2023)](https://arxiv.org/abs/2305.19466) — the NoPE result motivating Stack-100M's every-4th-layer no-position-encoding design.
    - [Chen et al., *Extending Context Window of Large Language Models via Positional Interpolation* (2023)](https://arxiv.org/abs/2306.15595) — position interpolation, the baseline NTK-aware scaling and YaRN build on.

    **Recent advances (2023–2026)**

    - [Peng et al., *YaRN: Efficient Context Window Extension of Large Language Models* (2023)](https://arxiv.org/abs/2309.00071) — per-wavelength NTK-by-parts scaling plus attention-temperature correction; the fuller cousin of the base rescale used here, and the `rope_scaling` type you get in `transformers`.
    - [Hu et al., *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies* (2024)](https://arxiv.org/abs/2404.06395) — introduces the WSD schedule and the 1−sqrt decay shape this chapter's `wsd_decay_multiplier` implements.
    - [Ibrahim et al., *Simple and Scalable Strategies to Continually Pre-train Large Language Models* (2024)](https://arxiv.org/abs/2403.08763) — quantifies the LR re-warm/re-decay tax that motivates resuming from a *pre-decay* checkpoint rather than a finished one.
    - [Gao et al., *How to Train Long-Context Language Models (Effectively)* (Princeton NLP / ProLong, 2024)](https://arxiv.org/abs/2410.02660) — the closest open reference for sub-phase B: a reproducible long-context continued-training recipe built on code repositories and books, with short data retained to protect short-context quality.
    - [Hsieh et al., *RULER: What's the Real Context Size of Your Long-Context Language Models?* (NVIDIA, 2024)](https://arxiv.org/abs/2404.06654) and [Yen et al., *HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly* (Princeton, 2024)](https://arxiv.org/abs/2410.02694) — the two benchmarks that replaced ad-hoc needle-in-a-haystack for real long-context evaluation.
    - [OLMo 2 Team, *2 OLMo 2 Furious* (Allen Institute for AI, 2024–2025)](https://arxiv.org/abs/2501.00656) — the fully open report that named and detailed the mid-training stage as a distinct phase between pretraining and post-training.
    - [Hugging Face, *SmolLM3: smol, multilingual, long-context reasoner* (2025)](https://huggingface.co/blog/smollm3) — a public 3B model whose recipe mirrors this chapter almost move-for-move: a decay-phase mix upsampling math/code, every-4th-layer NoPE, and YaRN extrapolation from a 64K training length out to 128K.

    **Open-source & tools**

    - [PyTorch, *FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention* (2024)](https://pytorch.org/blog/flexattention/) — `torch.nn.attention.flex_attention` and `create_block_mask`: the PyTorch-native, `torch.compile`-friendly way to express document masking as a block-sparse kernel.
    - [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) — `flash_attn_varlen_func` and the `cu_seqlens` convention used by Megatron-LM and most production stacks for packed sequences.
    - [allenai/OLMo-core](https://github.com/allenai/OLMo-core) — AI2's current training codebase for the OLMo family, including the staged pretrain/mid-train/anneal pipeline and its source-weighted data mixer.
    - [princeton-nlp/ProLong](https://github.com/princeton-nlp/ProLong) — code and data recipe for the long-context continued-training stage this chapter's sub-phase B follows.
    - [OpenBMB/MiniCPM](https://github.com/OpenBMB/MiniCPM) — reference implementation and checkpoints from the team that popularized WSD annealing at small scale.
    - [huggingface/smollm](https://github.com/huggingface/smollm) — training configs and data recipes for the SmolLM/SmolLM3 family, a concrete public example of NoPE + YaRN + decay-phase capability injection.
    - [huggingface/datatrove](https://github.com/huggingface/datatrove) — the filtering/dedup pipeline framework to use for the length filter and repo-level grouping at 20B-token scale.

    **Go deeper**

    - [HuggingFaceTB/cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) — the synthetic-textbook dataset this chapter's anneal mix leans on for dense, knowledge-rich tokens (and, being short-form, deliberately *not* the long-context source).
    - [deepmind/pg19](https://huggingface.co/datasets/deepmind/pg19) and [EleutherAI/proof-pile-2](https://huggingface.co/datasets/EleutherAI/proof-pile-2) — the public-domain books and arXiv/math corpora that supply sub-phase B's genuinely long documents.
    - [Aman Arora, *How LLMs Scaled from 512 to 2M Context: A Technical Deep Dive* (2025)](https://amaarora.github.io/posts/2025-09-21-rope-context-extension.html) — a worked, visual walkthrough of position interpolation, NTK-aware scaling, and YaRN that pairs well with this chapter's RoPE-rescale derivation.

## Further Reading

- OLMo 2 Team, *2 OLMo 2 Furious* (Allen Institute for AI, 2024–2025): the open report that named and detailed the mid-training / annealing stage.
- Hu et al., *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies* (2024): the WSD schedule and decay-phase data annealing.
- Gao et al., *How to Train Long-Context Language Models (Effectively)* (ProLong, Princeton NLP, 2024): the long-context continued-training data recipe — code repositories and books, with short data retained.
- Penedo et al., *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale* (HuggingFace, 2024): FineWeb-Edu and the Cosmopedia synthetic-data recipe.
- Peng et al., *YaRN: Efficient Context Window Extension of Large Language Models* (2023): per-wavelength RoPE scaling and the attention-temperature correction.
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021): the original RoPE.
- Chen et al., *Extending Context Window of Large Language Models via Positional Interpolation* (2023): position interpolation, the baseline NTK/YaRN build on.
- Ibrahim et al., *Simple and Scalable Strategies to Continually Pre-train Large Language Models* (2024): LR re-warming/re-decaying and the cost of resuming a decayed checkpoint.
- Lozhkov et al., *StarCoder 2 and The Stack v2* (BigCode, 2024) and Guo et al., *DeepSeek-Coder* (2024): repo-level code pretraining, the source of sub-phase B's longest documents.
- Rae et al., *Compressive Transformers for Long-Range Sequence Modelling* (2019): introduces PG-19, the book-length benchmark corpus.
- Hsieh et al., *RULER* (NVIDIA, 2024) and Yen et al., *HELMET* (Princeton, 2024): modern long-context evaluation beyond needle-in-a-haystack.
- Kazemnejad et al., *The Impact of Positional Encoding on Length Generalization in Transformers* (2023), and the SmolLM3 report (HuggingFace, 2025): NoPE and length generalization.
- Dao et al., *FlashAttention* and the PyTorch FlexAttention blog post (2024): the two kernel-level answers to document masking on packed long sequences.

## Exercises

**1.** In Chapter 14.7 we deliberately saved `ckpt_stable.pt` *before* decaying the learning rate, and mid-training resumes from that pre-decay checkpoint with the LR still at its plateau. Explain why resuming from a *fully-decayed* checkpoint instead would be worse, and connect your answer to the distinction between mid-training and continual pretraining.

??? note "Solution"
    A WSD schedule holds the LR at its peak through the long stable phase, then decays it sharply over a short final window; most of the *committed* loss reduction happens in that decay window. If we had already decayed the LR to (near) zero, the model would be *finished*: to keep training it we would have to **re-warm** the LR back up and then **re-decay** it. Ibrahim et al. (2024) show that re-warming and re-decaying a fully-decayed checkpoint costs you loss you then have to claw back — the model gets bumped off the low-loss basin it settled into and has to re-descend.

    By saving `ckpt_stable.pt` *pre-decay*, there is nothing to re-warm (the LR is already at its plateau) and little to forget, so mid-training is just the natural continuation of one WSD decay leg — the premium data gets fed into exactly the high-value decay window the schedule was designed around.

    This is precisely what separates **mid-training** from **continual pretraining** (Ch. 3.16). Continual pretraining adapts an *already-finished, fully-decayed* model to a new domain, and its central headache *is* the re-warming/catastrophic-forgetting problem. Mid-training is planned *before* the model is finished: we never fully decayed, so we resume from the high-LR plateau and pay none of that re-warming tax.

**2.** The chapter sequences the three moves as **broad -> long -> narrow**: anneal (general premium mix @2048) first, long-context extension second, capability injection (concentrated math/code) last, all riding one decreasing LR. Give the reason each move sits where it does, and name one concrete failure you would expect if you swapped the order to **narrow -> long -> broad**.

??? note "Solution"
    - **Anneal first** because the general quality jump wants the *highest* LR of the decay leg (the model still has enough plasticity to re-organize broad representations) and the *largest, most diverse* token budget. You consolidate broad representations before specializing.
    - **Long-context extension second** because it is a targeted change to one subsystem (the RoPE positional geometry) that benefits from a *settled* model. Rescaling RoPE on a mid-anneal checkpoint whose attention patterns have already sharpened means the long-context tokens teach the heads to *use* the new range rather than fighting a still-shifting representation.
    - **Capability injection last**, at the LR floor, because it is the narrowest and most over-fitting-prone mix; keeping it in the low-LR tail limits how far it can pull the model off the general manifold while still committing arithmetic/code structure.

    Swapping to **narrow -> broad** would put the concentrated math/code mix into the *highest-LR, high-plasticity* window. With the LR near peak, that narrow mix would pull the model hard off the general manifold and hurt breadth/diversity — you would waste the most valuable, most plastic part of the decay leg on a narrow specialization, and then the later broad phase would have to partially undo it at a *lower* LR where it has less leverage to recover. You would end up with worse general fluency and a weaker (not stronger) net capability floor.

**3.** Use the illustrative budgets from the chapter: sub-phase A = 1.2B tokens, B = 0.6B, C = 0.2B, with `GLOBAL_BATCH_TOKENS = 512_000`. The LR decays as one continuous `1-sqrt` WSD leg across all three phases (`final_frac = 0`), with `MUON_PEAK_LR = 0.02` and `ADAMW_PEAK_LR = 3e-3` (Ch. 14.6). Compute (a) `total_decay_steps`, and (b) the learning rate of *each optimizer group* at the A->B boundary and at the B->C boundary. Work the arithmetic to final numbers.

??? note "Solution"
    **(a) Steps per sub-phase** (floor division `tokens // GLOBAL_BATCH_TOKENS`, as in `steps_for`):

    $$
    \text{A} = \left\lfloor \tfrac{1.2\times10^9}{512000} \right\rfloor = \lfloor 2343.75 \rfloor = 2343,\quad
    \text{B} = \lfloor 1171.875 \rfloor = 1171,\quad
    \text{C} = \lfloor 390.625 \rfloor = 390.
    $$

    $$
    \text{total\_decay\_steps} = 2343 + 1171 + 390 = \mathbf{3904}.
    $$

    **(b)** The multiplier is $1 - \sqrt{t}$ with $t = \text{mid\_step}/\text{total\_decay\_steps}$, and each group's LR is *its own* peak times that one multiplier.

    A->B boundary: `mid_step` = 2343.

    $$
    t = \tfrac{2343}{3904} = 0.6002,\quad \text{mult} = 1-\sqrt{0.6002} = 1 - 0.7747 = 0.2253.
    $$

    $$
    \text{LR}_{\text{Muon}} = 0.02 \times 0.2253 \approx \mathbf{4.51\times10^{-3}},\qquad
    \text{LR}_{\text{AdamW}} = 3\times10^{-3} \times 0.2253 \approx \mathbf{6.76\times10^{-4}}.
    $$

    B->C boundary: `mid_step` = 2343 + 1171 = 3514.

    $$
    t = \tfrac{3514}{3904} = 0.9001,\quad \text{mult} = 1-\sqrt{0.9001} = 1 - 0.9487 = 0.0513.
    $$

    $$
    \text{LR}_{\text{Muon}} = 0.02 \times 0.0513 \approx \mathbf{1.03\times10^{-3}},\qquad
    \text{LR}_{\text{AdamW}} = 3\times10^{-3} \times 0.0513 \approx \mathbf{1.54\times10^{-4}}.
    $$

    So long-context extension (B) runs at roughly 23% of peak and capability injection (C) starts at roughly 5% of peak — exactly the "narrow, over-fitting-prone mix at the LR floor" the chapter describes. Note that the *ratio* between the two groups (0.02 / 3e-3 ≈ 6.7×) is preserved at every step, which is the whole point of scaling by a shared multiplier rather than assigning a shared LR.

**4.** Suppose a *different* model, Stack-Big, has head dimension $d = 128$, was pretrained at $L_{\text{old}} = 2048$ with RoPE base $\theta = 10000$, and you want to extend it to $L_{\text{new}} = 16384$. (a) Compute the NTK-rescaled base $\theta'$. (b) The fastest frequency pair ($k=0$) is supposed to be left essentially untouched — show that it is *exactly* unchanged by any base rescale, and say in one sentence why that matters.

??? note "Solution"
    **(a)** Scale factor $s = L_{\text{new}}/L_{\text{old}} = 16384/2048 = 8$. Exponent $d/(d-2) = 128/126 = 1.01587$.

    $$
    \theta' = \theta \cdot s^{\,d/(d-2)} = 10000 \cdot 8^{1.01587}
    = 10000 \cdot e^{1.01587\,\ln 8}
    = 10000 \cdot e^{1.01587 \times 2.07944}.
    $$

    $$
    = 10000 \cdot e^{2.1125} = 10000 \cdot 8.269 \approx \mathbf{82{,}700}.
    $$

    So bump the base from $10000$ to about $82{,}700$ (an 8x context needs a bigger stretch than Stack-100M's 4x, which only reached ~42000).

    **(b)** The per-pair frequency is $\theta_k = \theta^{-2k/d}$. For $k = 0$ the exponent is $-2\cdot 0/d = 0$, so

    $$
    \theta_0 = \theta^{0} = 1 \quad\text{and}\quad \theta'_0 = (\theta')^{0} = 1,
    $$

    identical for *any* base. The fastest pair rotates one radian per token regardless of rescaling. This matters because the highest-frequency pair encodes **local** relative position; leaving it untouched preserves the model's fine-grained local resolution while only the slow, long-range pairs get stretched to cover the new positions.

**5.** The mid-training driver shrinks the micro-batch as the sequence length grows so the *global* batch stays fixed. Using `micro_bs = max(1, MICRO_BATCH_TOKENS // seq_len)` with `MICRO_BATCH_TOKENS = 65_536`, `seqs_per_global = 512_000 // seq_len`, and `accum = seqs_per_global // micro_bs`: (a) compute `micro_bs`, `accum`, and tokens-per-optimizer-step at `seq_len = 2048` and at `seq_len = 8192`, and confirm the global batch is preserved; (b) by what factor does the attention *score matrix* grow from 2048 to 8192, and what three mechanisms keep that from causing an OOM?

??? note "Solution"
    **(a)** At `seq_len = 2048`:

    $$
    \text{micro\_bs} = \lfloor 65536 / 2048 \rfloor = 32,\quad
    \text{seqs\_per\_global} = \lfloor 512000/2048 \rfloor = 250,\quad
    \text{accum} = \lfloor 250/32 \rfloor = 7.
    $$

    Tokens/step $= \text{accum}\times\text{micro\_bs}\times\text{seq\_len} = 7\times 32\times 2048 = \mathbf{458{,}752}$.

    At `seq_len = 8192`:

    $$
    \text{micro\_bs} = \lfloor 65536 / 8192 \rfloor = 8,\quad
    \text{seqs\_per\_global} = \lfloor 512000/8192 \rfloor = 62,\quad
    \text{accum} = \lfloor 62/8 \rfloor = 7.
    $$

    Tokens/step $= 7\times 8\times 8192 = \mathbf{458{,}752}$.

    Both land at ~0.46M tokens/step, so the LR/batch relationship (Ch. 3.10) is unchanged across the sequence-length jump — the micro-batch dropped 4x (32 -> 8) exactly as the sequence grew 4x.

    **(b)** Attention score cost is quadratic in sequence length, so the score matrix grows by $(8192/2048)^2 = 4^2 = \mathbf{16\times}$. Three mechanisms keep it fitting: (1) the micro-batch drops 4x, so fewer sequences are resident at once (trading time for memory via grad accumulation); (2) a fused attention kernel — FlashAttention through SDPA, or FlexAttention's block-sparse Triton kernel — never materializes the full $T\times T$ score matrix, and FlexAttention additionally skips blocks that document masking has emptied; (3) GQA with 2 KV heads already shrinks the KV/activation footprint 4x. Activation checkpointing on the 30 blocks is the fourth lever if you are still tight. Note the *mask* is a separate trap: a dense `(B,1,T,T)` bool mask is 537 MB at `micro_bs=8, T=8192` and also disqualifies the FlashAttention backend — which is why the chapter replaces it with a `BlockMask`.

**6.** Implement the **loss-versus-position** diagnostic the chapter recommends for validating long-context extension. Write `bin_loss_by_position(per_token_loss, num_bins)` that takes a `(B, T)` tensor of per-position next-token losses and returns the mean loss in each of `num_bins` equal-width position bins across the sequence. Then state, in one sentence each, what a *healthy* curve and a *broken* curve look like, why the scalar average perplexity can hide the breakage, and why the diagnostic must be run on the long-document subset.

??? note "Solution"
    ```python
    import torch

    def bin_loss_by_position(per_token_loss: torch.Tensor, num_bins: int = 16) -> torch.Tensor:
        """Mean next-token loss binned by position within the sequence.

        per_token_loss : (B, T) tensor of per-position NLL (targets already shifted),
                         e.g. F.cross_entropy(logits, y, reduction="none") reshaped to (B, T).
        Returns         : (num_bins,) tensor; entry j = mean loss over positions in bin j.
        Bin j spans positions [j*T/num_bins, (j+1)*T/num_bins). Average over the batch
        first, then scatter-add into bins so every position contributes equally.
        """
        B, T = per_token_loss.shape
        per_pos = per_token_loss.mean(dim=0)                       # (T,) average over batch
        pos = torch.arange(T, device=per_token_loss.device)
        bin_idx = (pos * num_bins) // T                            # 0 .. num_bins-1
        sums = torch.zeros(num_bins, device=per_token_loss.device)
        counts = torch.zeros(num_bins, device=per_token_loss.device)
        sums.index_add_(0, bin_idx, per_pos)
        counts.index_add_(0, bin_idx, torch.ones_like(per_pos))
        return sums / counts.clamp(min=1)                          # (num_bins,)
    ```

    Usage during mid-training: run the frozen validation set through the model at `seq_len = 8192`, get per-token losses with `reduction="none"`, and plot `bin_loss_by_position(losses)`.

    - **Healthy curve:** loss keeps *decreasing* (or at worst flattening) as position grows — the model is genuinely using more context to predict better deep in the 8192-window.
    - **Broken curve:** loss is fine up to ~2048 and then *rises* past it — the RoPE rescale-plus-training has not taken and the far positions are still noise (train sub-phase B longer or on genuinely longer documents).
    - **Why the scalar average hides it:** a couple of well-predicted local tokens (the early, high-frequency-pair positions the model always handled) pull the *mean* down, so overall perplexity can *fall* even while the tail is broken. Only binning by position — or the needle-in-a-haystack probe — exposes that the far positions never learned to attend.
    - **Why the long-document subset:** with document-aware masking and per-document position resets, a token at *window* position 7000 inside a packed pile of 500-token documents has *document* position ~200 and can only attend to ~200 predecessors. Its loss says nothing about long-range ability, and mixing such windows in flattens the curve into uninformative noise. Bin only over windows drawn from documents that actually span the window — the same shards `verify_positions` asserted on.

**7.** (Systems) At `micro_bs = 8`, `T = 8192`, `n_heads = 8`, compute (a) the memory of the dense `(B, 1, T, T)` boolean document mask that `Stack100M._attn_mask` builds, and (b) the number of $128\times128$ blocks a FlexAttention `BlockMask` tracks per batch element. (c) A packed 8192-window holds documents averaging 1024 tokens. Estimate the factor by which block-sparse document masking reduces attention FLOPs versus a plain causal mask, and explain why the saving is *not* free at sub-phase B specifically.

??? note "Solution"
    **(a)** $8 \times 1 \times 8192 \times 8192 = 5.369\times10^{8}$ elements. A PyTorch `bool` tensor is 1 byte per element, so **537 MB** — allocated on every forward and kept alive across the backward pass. (At `micro_bs = 32, T = 2048` the same expression gives 134 MB, which is why the problem only becomes fatal at long context.) The second cost is not memory at all: handing a dense `attn_mask` to `F.scaled_dot_product_attention` rules out the FlashAttention backend, so you also lose the fused kernel.

    **(b)** $\lceil 8192/128 \rceil = 64$ blocks per side, so $64 \times 64 = \mathbf{4096}$ block entries per batch element (broadcast over heads with `H=None`). Stored as indices plus counts this is on the order of tens of kilobytes for the whole batch — five orders of magnitude smaller than the dense mask, and it is $O((T/128)^2)$ rather than $O(T^2)$ in bytes.

    **(c)** With causal masking alone, attention computes the lower triangle: $\tfrac{1}{2}T^2 = 3.36\times10^{7}$ score entries per head. With block-diagonal document masking and documents of length $\ell = 1024$, there are $T/\ell = 8$ documents, each contributing $\tfrac{1}{2}\ell^2$: total $8 \times \tfrac{1}{2}(1024)^2 = 4.19\times10^{6}$ — an **8× reduction**, i.e. a factor of $T/\ell$ in general. Block granularity costs a little of that back (partially-filled $128\times128$ blocks on the diagonal are computed in full), so expect somewhat less than 8× in practice.

    Why it is not free at sub-phase B: that saving is *proportional to how short your documents are*. Sub-phase B deliberately trains on documents that fill the window — whole repositories, whole books — so $\ell \approx T$, the block mask is nearly the full lower triangle, and the FLOP saving collapses toward 1×. That is the correct outcome (you are paying for real long-range attention, which is the entire point of the sub-phase), but it means the compute estimate in "What Mid-Training Costs and Buys" must use the *dense causal* attention term for B and C — you do not get to bank the block-sparsity discount on exactly the phase you introduced it for.
