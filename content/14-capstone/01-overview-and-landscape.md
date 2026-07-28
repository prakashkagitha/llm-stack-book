# 14.1 The Capstone: Building Stack-100M, and the 2026 Small-Model Landscape

In his `nanoGPT` and `llm.c` projects, Andrej Karpathy showed that with a clean implementation and a few tricks, you could reproduce GPT-2 (124M parameters) from scratch for a startlingly small amount of compute — on the order of a single 8×A100 node-hour for the 124M size, and roughly USD 100–200 for a careful, well-tuned reproduction on rented hardware. It was a landmark moment: training a real, GPT-class language model stopped being something only a well-funded lab could do, and became a weekend project with a credit card. That single demonstration re-anchored an entire generation of engineers' intuition for what training actually costs.

This capstone is the 2026 update to that demonstration — and the model you will build is *categorically* better than the 2019 GPT-2 it descends from, at the same parameter count. Since Karpathy's reproduction, the field learned three things that changed what a small model can do: train it on far more data than "optimal," curate that data far more aggressively, and borrow the architecture and optimizer tricks that frontier labs spent hundreds of millions of dollars discovering. This chapter lays out *why* those three forces work, surveys the 2025–2026 small-model landscape that proves it (SmolLM2/SmolLM3, Qwen3-0.6B, MobileLLM, and the component transfer from DeepSeek, Kimi, GLM, and Liquid AI), and gives you the exact specification, budget, and repository map for the model we will build together across the next eleven chapters: **Stack-100M**.

Every number in this chapter — and every chapter after it — is fixed by one specification. We will not re-derive the architecture here; we cite it, verify its parameter count by hand, and point forward to where each piece is taught in depth. If you want the single source of truth for the whole capstone, it is the config in this chapter's parameter-accounting section; every later chapter imports it unchanged.

---

## Why Build a 100M-Parameter Model From Scratch

It is worth asking directly: in a world of one-API-call frontier models, why spend eleven chapters training a *small* one yourself?

**Because ownership of every layer is the only way to actually understand the stack.** You can read this book's chapters on tokenization, attention, GQA, RoPE, Muon, WSD schedules, DPO, and GRPO in isolation, and each will make sense on its own. But nothing forces you to *reconcile* them — to discover that your tokenizer's vocabulary size trades off against your embedding parameter budget, that your data mix determines what your mid-training decay phase can inject, that your optimizer choice changes how sensitive your model is to the learning rate you picked in a chapter you read three weeks ago — the way building one coherent artifact does. A production LLM has hundreds of interacting design decisions; a 100M model, built by you, has all the same *kinds* of decisions at a scale you can hold in your head and debug on a single GPU.

**Because the economics are now real, not hypothetical.** The flagship path in this capstone trains Stack-100M on a single rented A100 for on the order of 15–25 GPU-hours — call it USD 40–100 at typical cloud rental prices. That is a number you can actually spend. Every code block in this project is complete and runnable; you are not reading pseudocode about what a training run *would* look like, you are running the training run.

**Because "narrow but real" is a more honest and more useful target than "impressive demo."** We are not going to pretend Stack-100M is a general-purpose chatbot — at this scale, that would be a lie, and this book does not lie to you about capability. Instead we aim it at something a 100M model can genuinely do well: answer questions grounded in a small retrieval corpus, call a calculator correctly, solve narrow verifiable tasks. That target is modest, but it is *real*, and getting there end to end teaches you more than a bigger model you only ever fine-tuned.

By the end of Part XIV you will have taken one model through the entire lifecycle this book covers: raw text in, a tool-using agent out, running on a laptop.

### What "end-to-end" concretely means here

To make that concrete, "end-to-end" is not a slogan — it is this exact list of things you will have built with your own hands and your own compute, in order:

1. A byte-level BPE tokenizer, trained from scratch on your own data sample.
2. A streaming data pipeline: download, quality-filter, deduplicate, tokenize, pack into fixed-length training shards.
3. A deep-and-thin transformer implementation with GQA, RoPE+NoPE, QK-norm, and SwiGLU, matching a fixed ~101M-parameter budget you can verify by hand.
4. A miniature scaling-law study — your own Chinchilla-style ladder of tiny models — used to *justify* the final config rather than accept it on faith.
5. A hybrid Muon+AdamW optimizer with a Warmup-Stable-Decay schedule, trained with a real pretraining loop: bf16 autocast, gradient accumulation, checkpoint/resume, and measured MFU.
6. A mid-training phase that anneals on higher-quality data during the WSD decay and extends the context window.
7. Post-training: SFT with a chat template, DPO on preference pairs, and a narrow RLVR/GRPO run on a verifiable task.
8. A ReAct tool-use agent, taught by distilling trajectories from a larger teacher model, that can search a small corpus and answer questions grounded in it.
9. Honest evaluation, int4 quantization, and a model that generates text on your own laptop's CPU.

That is the map for the rest of Part XIV. This chapter is the only one that does not build a piece of the model — it builds your understanding of *why* this is the right model to build, and *what* every later chapter is going to hand you.

---

## The Finished Artifact and the Map of Part XIV

The finished artifact has a name — **Stack-100M** — and it lives in a repository laid out as the Python package `stacklm`, in the `capstone/` directory. Every later chapter extends the same package; nothing is thrown away and rewritten. Here is the roadmap, with each chapter's focus taken directly from the capstone specification, so you can see how the pieces fit before you build any of them:

| Ch. | Focus | Produces |
|---|---|---|
| 14.1 | Overview & landscape (this chapter) | Spec, budget, repo skeleton |
| 14.2 | Pretraining data | FineWeb-Edu/Cosmopedia/code/math mix, dedup, packed `.bin` shards |
| 14.3 | Tokenizer | 32,768-vocab byte-level BPE, trained from scratch |
| 14.4 | Architecture | The Stack-100M transformer: GQA, RoPE+NoPE, QK-norm, SwiGLU, MLA/MTP as options |
| 14.5 | Mini scaling laws | A fitted $L(N,D)$ from a small model ladder, justifying the 101M/20B choice |
| 14.6 | Optimizer & schedule | Muon (2D weights) + AdamW hybrid, MuonClip/QK-clip, WSD schedule |
| 14.7 | The pretraining loop | Full single-GPU training run, checkpointing, MFU measurement |
| 14.8 | Mid-training | WSD decay-phase annealing, long-context extension to 8192 |
| 14.9 | Post-training | SFT, DPO, narrow RLVR/GRPO |
| 14.10 | Agentic | ReAct loop, tool distillation, the "auto-research" demo |
| 14.11 | Evaluation & serving | Honest capability report, int4 quantization, laptop inference |
| 14.12 | Retrospective | Cost accounting, reproducibility checklist, scaling to 1B |

Each of those chapters cross-links back to the relevant deep-dive chapter elsewhere in the book — the capstone teaches *integration*, the main chapters teach *mechanism*. Where this chapter references a mechanism in passing, you will find the full treatment linked.

{{fig:capstone-lifecycle-arc}}

### Narrow but real: setting honest expectations

We use the phrase "narrow but real" throughout Part XIV, so it is worth defining precisely, because it is the single most important expectation-setting sentence in this capstone.

**Narrow** means we do not claim Stack-100M is a general-purpose assistant. It will not reliably discuss history, write correct code for novel problems, or reason robustly across arbitrary domains — a 100M-parameter model, even over-trained on 20B tokens, does not have the representational capacity for that, and no amount of clever post-training changes the underlying scaling laws covered in [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html). Anyone who tells you a 100M model is a general chatbot is either confused or selling something.

**Real** means that within a scaffolded, narrow domain — arithmetic with a calculator tool, question-answering grounded in a small local corpus via retrieval, following a fixed chat format — the model's outputs are genuinely correct, not merely plausible-sounding. "Real" is a claim about *verifiability*: when the agent in Chapter 14.10 answers a question, you can check the answer against the source it retrieved, or against a verifiable reward function, exactly the way Chapter 14.9's RLVR run and Chapter 14.10's ReAct agent are built. That is a much stronger and more honest claim than "reads like a chatbot."

The entire arc of Part XIV — over-training, data quality, mid-training capability injection, narrow post-training, tool distillation — is aimed at maximizing the *real* half of that phrase without ever pretending away the *narrow* half.

---

## The 2025–2026 Small-Model Landscape

Karpathy's GPT-2 reproduction trained a 124M-parameter model on roughly 10B tokens of WebText — about 80 tokens per parameter, on data that was scraped and lightly filtered but not aggressively curated, with an architecture (learned absolute positions, LayerNorm, GELU MLP, full multi-head attention) that has since been improved on almost every axis. Stack-100M is architecturally, algorithmically, and data-wise a different animal at a similar parameter count. The proof that this is not just our project's optimism is that the same recipe has been demonstrated repeatedly by frontier labs and open-source teams over 2024–2026, at parameter counts from a few hundred million down to well under a billion:

- **SmolLM2 and SmolLM3** (HuggingFace, 2024–2025) are a family of small dense models trained on aggressively curated, heavily over-trained data mixes (built on FineWeb-Edu and Cosmopedia, the same families Stack-100M uses — see [Pretraining Data: Sources, Crawling & The Data Pipeline](../03-pretraining/01-pretraining-data.html)). SmolLM3 in particular popularized **NoPE** — omitting rotary position embeddings on a subset of layers — as a length-generalization trick at small scale, which Stack-100M adopts directly (every 4th layer, per the config below).
- **Qwen3-0.6B** (Alibaba, 2025) demonstrates that a sub-billion dense model, trained with the same modern component stack used at the frontier — GQA, RMSNorm, SwiGLU, a large, carefully mixed pretraining corpus — can be a genuinely capable base for downstream fine-tuning, not a toy.
- **MobileLLM** (Liu et al., Meta, 2024) is the paper that made the "deep-and-thin" case explicit and quantitative: at a *fixed* parameter budget, more layers with a narrower hidden dimension consistently outperforms fewer, wider layers, plus embedding-sharing tricks to spend the parameter budget on depth instead of vocabulary. The same deep-and-thin bias shows up in the GLM family and in Qwen3-0.6B's aspect ratio. This is the single design decision behind Stack-100M's 30 layers at `d_model=512` rather than, say, 12 layers at `d_model=1024`.
- **Component transfer from frontier labs** is the third leg. Techniques originally built to make 100B+-parameter training runs stable and efficient turn out to work — and matter — at 100M scale too, because the underlying arithmetic (attention-logit blow-ups, KV-cache size, gradient conditioning) does not care how many parameters your model has. Stack-100M borrows: **MLA** (Multi-head Latent Attention) from **DeepSeek-V2**, taught as an alternative to GQA; **MuonClip** (QK-clip), the fix that let **Kimi K2** (Moonshot AI, 2025) train stably with the Muon optimizer at scale; the **WSD** (Warmup-Stable-Decay) schedule popularized by **MiniCPM** and used by **DeepSeek**; and the gated short-convolution-plus-attention hybrid block design explored by **Liquid AI's LFM2** (2025), which we implement as an optional block variant.

The common thread across all of these: none of them are exotic research curiosities. They are the *current default choices* of teams training small models for real deployment, and they are choices you can implement, in full, in a single evening, in the code that follows.

!!! note "Small models did not get smarter by accident"
    It is tempting to read the small-model renaissance as "GPUs got a bit better, so small models got a bit better too." That is not what happened. Hardware improved training *throughput*, but the loss-per-parameter improvement at fixed size over 2019–2026 comes almost entirely from the three forces in the next section — training regime, data, and architecture — which are algorithmic choices, not hardware. You could run the exact recipe below on 2019-era GPUs (just more slowly) and still get a dramatically better model than 2019-era GPT-2 at the same parameter count.

---

## Three Forces That Made Small Models Good

### Force 1 — massive over-training past Chinchilla

The Chinchilla scaling law (Hoffmann et al., 2022; developed fully in [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html)) tells you the *compute-optimal* token budget for a given model size: roughly 20 tokens per parameter. For a 101M-parameter model, that is about 2B tokens — you would minimize training loss *per unit of training compute spent* by stopping there.

Stack-100M trains on **~20B tokens** — roughly **200 tokens per parameter**, about **10× past Chinchilla-optimal**. This is deliberate, and it is the single most important economic idea in this capstone: **compute-optimal is not the same as deployment-optimal.**

Chinchilla's calculation only accounts for the cost of *training*. It says nothing about what happens after you ship the model — every one of the millions or billions of inference calls a deployed model serves. Training compute is a one-time cost; inference compute recurs forever. If over-training a smaller model past its Chinchilla-optimal point buys you a meaningfully lower loss *at a fixed, cheap-to-serve parameter count*, that trade is almost always worth it for anything you intend to actually deploy, because you pay the extra training FLOPs once and save the larger model's extra inference FLOPs on every single call thereafter. This is exactly the logic Meta used to justify Llama's ratios and that essentially every subsequent open small-model release (SmolLM2/3, Qwen3, MobileLLM) has followed. The data-scaling term $B/D^\beta$ in the Chinchilla loss curve keeps paying off well past the "optimal" point — it just pays off *less per FLOP* than growing $N$ would, which is precisely the trade you are willing to make when parameter count (and therefore serving cost) is the constraint you actually care about, not training FLOPs.

{{fig:compute-vs-deployment-optimal}}

### Force 2 — data quality over raw quantity

GPT-2's WebText was scraped, filtered for outbound Reddit links with a minimum karma, and deduplicated — a reasonable 2019 pipeline, but crude by 2026 standards. The modern pretraining data stack (built out fully in [Data Cleaning, Deduplication & Quality Filtering](../03-pretraining/02-data-cleaning-dedup.html) and [Synthetic Data for Pre- and Post-Training](../03-pretraining/15-synthetic-data.html)) does two things GPT-2's data did not:

- **Classifier-based educational filtering.** FineWeb-Edu (Penedo et al., HuggingFace, 2024) trains a classifier to score web documents by educational value and keeps only the high-scoring tail of Common Crawl. The result is a corpus with dramatically higher information density per token than raw web text — fewer boilerplate pages, navigation menus, and low-content spam diluting every gradient step.
- **Synthetic, dense, on-topic text.** Cosmopedia (HuggingFace) generates textbook- and story-style synthetic content with a larger teacher model, specifically to give a small model clean, well-structured exposition of concepts that natural web text states only in passing, if at all.

At fixed token count, cleaner and denser data means every one of your 20B training tokens is doing more work — you are not spending gradient steps learning to ignore boilerplate.

### Force 3 — architecture and optimizer advances

The third force is the collection of small, individually modest architectural and optimization changes since the original Transformer that compound into a large improvement when stacked together: RMSNorm instead of LayerNorm (cheaper, and empirically as stable — see [The Transformer Block: Norms, Residuals, MLPs & Activations](../02-transformer/06-transformer-block.html)), SwiGLU instead of a plain ReLU/GELU MLP, RoPE instead of learned absolute positions (see [Positional Encodings: Sinusoidal, Learned, RoPE & ALiBi](../02-transformer/05-positional-encoding.html)), GQA to shrink the KV cache (see [Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html)), and — newer still — optimizers like Muon that condition the update geometry of 2D weight matrices far better than plain AdamW (see [Optimizers: SGD, Adam, Adafactor, Lion, Muon & Shampoo](../03-pretraining/09-optimizers.html)), paired with schedules like WSD that decouple "how long do I train" from "when do I decay," making mid-run data-mix changes cheap.

None of these changes individually is worth more than a few percent of loss. Stacked, and combined with over-training and data quality, they are the difference between a 2019-quality 100M model and a 2026-quality one.

{{fig:three-forces-2019-to-2026}}

---

## The Canonical Stack-100M Configuration

Every number below is fixed for the rest of Part XIV. This is a **deep-and-thin** decoder-only transformer — many layers, a narrow hidden dimension — following the MobileLLM insight that at fixed parameter count, depth beats width.

| Hyperparameter | Value | Notes |
|---|---|---|
| `vocab_size` | 32768 | byte-level BPE, trained from scratch (Ch. 14.3) |
| `d_model` | 512 | narrow — the "thin" in deep-and-thin |
| `n_layers` | 30 | deep |
| `n_heads` | 8 | query heads |
| `n_kv_heads` | 2 | GQA, 4:1 query-to-KV ratio |
| `head_dim` | 64 | $8 \times 64 = 512 = d_{\text{model}}$ |
| `intermediate_size` | 1408 | SwiGLU hidden size, $\approx 2.75 \cdot d_{\text{model}}$ |
| `max_seq_len` (pretrain) | 2048 | extended to 8192 in mid-training (Ch. 14.8) |
| `rope_theta` | 10000 | rescaled for long-context in mid-training |
| tied embeddings | yes | input embedding = output projection |
| total params | ≈ 101M | verified by hand below |

The components, each cited to where it comes from and cross-linked to the chapter that develops it fully:

- **Pre-norm residual blocks, RMSNorm** (Zhang & Sennrich, 2019) — cheaper than LayerNorm, no re-centering statistic, empirically just as stable. See [The Transformer Block: Norms, Residuals, MLPs & Activations](../02-transformer/06-transformer-block.html).
- **RoPE with NoPE on every 4th layer** — rotary position embeddings (Su et al., 2021) on most layers, and no positional encoding at all (Kazemnejad et al., 2023) on the interleaved subset, following SmolLM3. Interleaving improves length generalization: the NoPE layers let attention be exactly translation-invariant, which turns out to help extrapolation beyond the training context length. See [Positional Encodings: Sinusoidal, Learned, RoPE & ALiBi](../02-transformer/05-positional-encoding.html).
- **GQA — grouped-query attention**, 2 KV heads shared across 8 query heads (Ainslie et al., 2023) — a 4× reduction in KV-cache size versus full multi-head attention at essentially no quality cost at this scale. See [Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html).
- **QK-norm** — RMSNorm applied to queries and keys before the attention dot product, which keeps attention logits well-scaled and stabilizes training at the higher learning rates the Muon optimizer enables (Henry et al., 2020).
- **SwiGLU** gated MLP (Shazeer, 2020) in place of a plain feed-forward block.
- **Tied input/output embeddings** (Press & Wolf, 2017) — the same 32768×512 table serves as both the token embedding and the final logit projection, saving 16.8M parameters that would otherwise be nearly a sixth of the model.
- **z-loss and optional logit soft-cap** — a small penalty on the logsumexp of the output logits (z-loss), plus an optional Gemma-2-style tanh soft-cap, both purely for numerical stability of the softmax at the extremes of training.

Two further components are implemented and taught, but are **options**, not the default path — you will build them, understand the trade-off, and can swap them in:

- **MLA (Multi-head Latent Attention)**, from DeepSeek-V2 (2024) — compresses the KV cache into a low-rank latent representation, a further reduction beyond what GQA buys, at the cost of a more involved attention computation.
- **MTP (Multi-Token Prediction)**, from DeepSeek-V3 (2024) and Gloeckle et al. (2024) — an auxiliary head trained to predict the token *after* the next one, giving a denser training signal per forward pass.

Both are covered as explicit "if you want DeepSeek's trick" upgrades in Chapter 14.4 once you have the base architecture working, alongside a Liquid LFM2-style gated-convolution mixer block as a third option.

### Verifying the parameter count by hand

The whole point of a capstone is that you should never take a number like "≈101M parameters" on faith. Here is the exact accounting, and code that reproduces it.

$$
N_{\text{total}} = \underbrace{V d}_{\text{tied embed}} \;+\; L\Big(\underbrace{2d^2 + 2\, d\, d_{kv}}_{\text{attn (GQA)}} + \underbrace{3\, d\, d_{\text{ff}}}_{\text{SwiGLU}} + \underbrace{2d + 2 d_h}_{\text{norms + QK-norm}}\Big) \;+\; \underbrace{d}_{\text{final norm}}
$$

with $V=32768$, $d=512$, $L=30$, $d_{\text{ff}}=1408$, $d_{kv} = n_{\text{kv}}\cdot d_h = 2\times 64 = 128$, and $d_h = 64$. The attention term is written out explicitly because it is **not** simply $4d^2$: GQA makes the K and V projections only $d_{kv}=128$ wide, while Q and the output projection O stay full-width, so the four attention matrices sum to $d\cdot d + d\cdot d_{kv} + d\cdot d_{kv} + d\cdot d = 2d^2 + 2 d\, d_{kv}$, not $4d^2$. The $2 d_h$ term is the pair of tiny QK-norm scale vectors (over `head_dim`), which the code counts and the spec table rounds away.

```python
"""
stacklm/config.py — the canonical configuration for Stack-100M.

This is the single source of truth for the model's shape. Every later
chapter (architecture, pretraining loop, mid-training, post-training,
serving) imports StackConfig from here — nothing is redefined downstream.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class StackConfig:
    # --- core shape (fixed for the whole capstone, Ch. 14.1 spec) ---
    vocab_size: int = 32768          # byte-level BPE, trained from scratch (Ch. 14.3)
    d_model: int = 512               # narrow -- the "thin" in deep-and-thin
    n_layers: int = 30               # deep
    n_heads: int = 8                 # query heads
    n_kv_heads: int = 2              # GQA: 4 query heads share each KV head
    head_dim: int = 64               # n_heads * head_dim == d_model
    intermediate_size: int = 1408    # SwiGLU hidden size, ~2.75 * d_model
    max_seq_len: int = 2048          # pretrain context; -> 8192 in mid-training (Ch. 14.8)
    rope_theta: float = 10000.0

    # --- stability / small-model tricks (Ch. 14.4) ---
    tie_embeddings: bool = True      # input embed == output projection (Press & Wolf, 2017)
    use_qk_norm: bool = True         # RMSNorm on Q, K before the attention dot product
    nope_every: int = 4              # every 4th layer skips RoPE entirely (SmolLM3-style)
    z_loss_weight: float = 1e-4      # penalty on logsumexp(logits) for softmax stability
    logit_soft_cap: Optional[float] = None  # Gemma-2-style tanh soft-cap; off by default

    # --- optional efficiency variants, OFF by default (Ch. 14.4 "DeepSeek's trick") ---
    use_mla: bool = False            # Multi-head Latent Attention (DeepSeek-V2) instead of GQA
    mtp_heads: int = 0               # Multi-Token Prediction aux heads (DeepSeek-V3); 0 = off


def count_params(cfg: StackConfig) -> dict:
    """Exact, hand-verifiable parameter accounting for a StackConfig.

    This function is the executable version of the arithmetic in the
    Ch. 14.1 spec table. Run it and you should see ~101M total.
    """
    # Tied embedding table: one V x d matrix does double duty as the
    # input embedding lookup AND the final logit projection.
    embed = cfg.vocab_size * cfg.d_model

    # --- attention projections, GQA-shaped: Q is full-width, K/V are
    # only n_kv_heads wide, O projects the full-width attention output
    # back down to d_model. ---
    q_width = cfg.n_heads * cfg.head_dim      # = d_model by construction (8*64=512)
    kv_width = cfg.n_kv_heads * cfg.head_dim  # = 128 (GQA shrinks this vs. q_width)
    q_proj = cfg.d_model * q_width
    k_proj = cfg.d_model * kv_width
    v_proj = cfg.d_model * kv_width
    o_proj = q_width * cfg.d_model
    attn_per_block = q_proj + k_proj + v_proj + o_proj

    # --- SwiGLU MLP: three d_model <-> intermediate_size matrices
    # (gate, up, down) -- SwiGLU needs three, not two like a plain MLP. ---
    mlp_per_block = 3 * cfg.d_model * cfg.intermediate_size

    # --- norms: RMSNorm has one learnable scale vector per normalized
    # dimension. Two per block (pre-attn, pre-mlp) over d_model, plus
    # one final model-level norm. QK-norm adds two tiny ones over head_dim. ---
    rmsnorm_per_block = 2 * cfg.d_model
    qk_norm_per_block = (2 * cfg.head_dim) if cfg.use_qk_norm else 0
    final_norm = cfg.d_model

    per_block = attn_per_block + mlp_per_block + rmsnorm_per_block + qk_norm_per_block
    all_blocks = per_block * cfg.n_layers

    total = embed + all_blocks + final_norm
    # NOTE: if tie_embeddings were False, add another `embed` here for
    # an untied LM head. We do not -- that's a deliberate 16.8M-param
    # saving, see the worked example below.

    return {
        "embedding (tied)": embed,
        "attn_per_block": attn_per_block,
        "mlp_per_block": mlp_per_block,
        "norms_per_block": rmsnorm_per_block + qk_norm_per_block,
        "per_block_total": per_block,
        "all_blocks (x n_layers)": all_blocks,
        "final_norm": final_norm,
        "total": total,
    }


if __name__ == "__main__":
    cfg = StackConfig()
    for name, n in count_params(cfg).items():
        print(f"{name:26s} {n:>12,}  ({n / 1e6:7.3f}M)")
```

Running this prints an embedding of 16,777,216 (16.78M), an attention block of 655,360 (0.655M), an MLP block of 2,162,688 (2.163M), a per-block total (with norms) of ≈2.819M, all 30 blocks summing to ≈84.58M, and a grand total of **≈101.35M parameters** — matching the ≈101M figure in the spec table, with the small residual difference from the spec's own hand-rounding of the per-block figure to 2.82M.

!!! example "Worked example: where does 26% come from?"
    Suppose instead we had picked the more conventional 50,000-token vocabulary size (roughly what GPT-2/GPT-3-family tokenizers use) instead of 32,768. The tied embedding table would be $50{,}000 \times 512 = 25{,}600{,}000$ parameters — **≈25.6M**, versus 16.78M at our chosen vocab size. That is an extra **8.8M parameters**, all spent on the embedding table before a single transformer block has processed anything. Measured against Stack-100M's ≈101M budget, that 25.6M table is **roughly a quarter of the whole model** (about 25%, versus 16.6% for our 32,768 table) — a huge fraction of your parameter count spent on lookup rather than computation, and if you kept the transformer blocks fixed it would push the total to ≈110M. This is exactly why vocabulary size is a first-class architectural decision at small scale, not an afterthought copied from a larger model's tokenizer; Chapter 14.3 works through the full tradeoff curve when we build the tokenizer.

!!! interview "Interview Corner"
    **Q:** You are given a fixed parameter budget and told to train the best model you can, to be deployed and served many times. Explain why you would deliberately train past the Chinchilla-optimal token count, and quantify what you are trading away.

    **A:** Chinchilla-optimal minimizes loss *per unit of training compute* — it answers "how do I get the best model for a fixed training FLOP budget," treating training as the only cost. But a deployed model's cost is dominated by *inference*, which recurs on every call, while training is a one-time cost. Given a fixed parameter count (which determines serving cost and latency), you want to minimize loss *for that fixed size*, and the data-scaling term of the loss curve, $B/D^\beta$, keeps improving well past the Chinchilla ratio — it just costs more training FLOPs per unit of improvement than it would to instead grow $N$. You are trading extra, one-time training compute (in Stack-100M's case, roughly 10× the Chinchilla-optimal token count) for a lower loss at a serving cost you have already fixed. The trade only makes sense if you actually plan to serve the model enough times that the one-time training cost is amortized — which is the deployment case, not the "just want the lowest-loss model for this FLOP budget" case Chinchilla itself answers.

{{fig:stack100m-param-budget}}

---

## Budget, Hardware Tiers, and the Repository

### The compute tiers

Stack-100M is designed to be trained at three tiers of hardware, with the same code and the same architecture — only the token budget, batch size, and precision change.

| Tier | Hardware | Wall-clock | Cost | Role |
|---|---|---|---|---|
| **Flagship** | 1× A100 (80GB), rented | ≈15–25 GPU-hours | ≈USD 40–100 | The documented, full-fidelity walkthrough |
| **Consumer** | RTX 4090 / 3090 (24GB) | ≈2–4× flagship wall-clock | ≈USD 0 if owned | Same recipe, more gradient accumulation |
| **On-ramp** | Free Colab (T4, 16GB) | scaled-down run | USD 0 | Smaller model, fewer tokens — explicitly a subset, not the full run |

The flagship number is what we derive and defend below. The 4090 tier runs the identical recipe: less HBM means a smaller micro-batch and more gradient-accumulation steps to reach the same effective batch size, and no bf16-Tensor-Core throughput advantage lost — the 4090 has full bf16 support — so the main cost is wall-clock, not correctness. The Colab T4 tier is different in kind, not just degree: the T4 is a 2018 Turing-generation part with no native bf16 tensor-core path, so the on-ramp config trains in fp16 mixed precision instead of bf16 (with the corresponding loss-scaling care this requires — see [Mixed Precision, bf16 & FP8 Training](../03-pretraining/08-mixed-precision-fp8.html)) and deliberately shrinks the model and token budget so a meaningful run finishes inside Colab's free-tier session limits.

### From FLOPs to GPU-hours: a worked derivation of "$40–100"

The flagship budget is not a marketing number — it falls straight out of the $6ND$ training-FLOPs rule (developed in [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html)) and the A100's published bf16 Tensor Core throughput (see [GPU Architecture & The Memory Hierarchy](../01-foundations/08-gpu-architecture.html)).

!!! example "Worked example: FLOPs to dollars"
    **Setup.** $N \approx 101.35 \times 10^6$ parameters (from `count_params` above), $D = 20 \times 10^9$ tokens (the flagship token budget — note $D/N \approx 197$ tokens/parameter, matching the spec's "≈200 tokens/param, ≈10× past Chinchilla").

    **Total training compute**, using the standard dense-transformer approximation of 2 forward + 4 backward FLOPs per parameter per token:

    $$
    C = 6ND = 6 \times (101.35\times 10^6) \times (20\times 10^9) \approx 1.216 \times 10^{19} \text{ FLOPs}
    $$

    **Convert to wall-clock.** The A100's bf16 Tensor Core peak is $\pi \approx 312$ TFLOP/s $= 3.12\times10^{14}$ FLOP/s. A well-tuned small dense transformer with a fused attention kernel and no cross-node communication typically achieves **MFU (Model FLOPs Utilization) around 0.35–0.55** on a single GPU (the same MFU range this book uses for large multi-GPU runs, developed in [Megatron-LM, DeepSpeed & Parallelism in Practice](../03-pretraining/07-megatron-deepspeed.html)). Taking a representative MFU of 0.50:

    $$
    \text{achieved throughput} = 0.50 \times 3.12\times10^{14} = 1.56\times10^{14} \text{ FLOP/s}
    $$

    $$
    t = \frac{C}{\text{achieved throughput}} = \frac{1.216\times10^{19}}{1.56\times10^{14}} \approx 77{,}950 \text{ s} \approx 21.7 \text{ hours}
    $$

    **Convert to dollars.** At a representative on-demand A100 rental price of roughly USD 1.50–2/GPU-hour, 21.7 hours costs on the order of **USD 33–43** for the pretraining run itself. The documented USD 40–100 flagship figure leaves headroom above that core number for the tokenizer-training pass, the data pipeline run, periodic held-out evaluation during training, and — realistically — at least one restart from a checkpoint after a bad hyperparameter choice or a spot-instance preemption. Real MFU also varies; 21.7 hours assumed 50% MFU, and a less-tuned first attempt closer to 35% MFU pushes the pretraining pass alone to roughly 31 hours, comfortably inside the documented 15–25-hour *tuned* range plus the realistic slop of a first attempt.

    This is exactly the kind of napkin calculation you should be able to reproduce for any training run before you spend the money — it has four inputs (parameter count, token budget, peak hardware FLOP/s, and MFU) and one multiplication.

### Repository tour: `capstone/`, package `stacklm`

Every later chapter adds files to the same tree. Here is the layout you will build toward — nothing here is exotic, and each directory maps directly onto one of the PLAN sections summarized in the roadmap table above.

```text
capstone/
├── pyproject.toml
├── README.md
├── configs/
│   ├── stack100m.yaml          # the canonical config as YAML (mirrors StackConfig)
│   └── toy.yaml                 # tiny CI-scale config: d_model=32, n_layers=2, vocab=256
├── stacklm/
│   ├── __init__.py
│   ├── config.py                # StackConfig, count_params()          -- this chapter
│   ├── tokenizer/
│   │   └── bpe.py                # byte-level BPE trainer + encode/decode -- Ch. 14.3
│   ├── data/
│   │   ├── download.py           # streaming FineWeb-Edu/Cosmopedia/StarCoder/FineMath pull
│   │   ├── filter.py             # quality filtering
│   │   ├── dedup.py               # exact + MinHash near-dup removal
│   │   └── pack.py                # tokenize, pack to seq_len, doc-aware masking, .bin shards
│   ├── model/
│   │   ├── attention.py          # GQA + optional MLA, QK-norm, RoPE/NoPE
│   │   ├── mlp.py                 # SwiGLU
│   │   ├── block.py               # pre-norm residual block, optional LFM2-style conv mixer
│   │   └── transformer.py         # full StackLM model, tied embeddings, MTP heads
│   ├── optim/
│   │   ├── muon.py                # Newton-Schulz orthogonalized Muon + MuonClip
│   │   └── schedule.py            # WSD (warmup-stable-decay) schedule
│   ├── train.py                   # flagship pretraining loop                -- Ch. 14.7
│   ├── midtrain.py                # WSD decay-phase annealing, long-context   -- Ch. 14.8
│   ├── sft.py                     # supervised fine-tuning on chat template  -- Ch. 14.9
│   ├── dpo.py                     # direct preference optimization           -- Ch. 14.9
│   ├── rlvr.py                    # GRPO on a verifiable narrow task         -- Ch. 14.9
│   ├── agent/
│   │   ├── tools.py               # calculator, tiny-corpus retriever
│   │   ├── react.py               # think -> act -> observe loop
│   │   └── distill.py             # teacher-trajectory generation + filtering -- Ch. 14.10
│   ├── eval/
│   │   ├── probes.py              # perplexity, arithmetic, MC, retrieval-QA
│   │   └── harness.py
│   └── serve/
│       ├── quantize.py            # int8 -> int4 post-training quantization
│       └── cli.py                 # `stacklm generate "..."` on CPU
├── scripts/
│   ├── estimate_budget.py         # the FLOPs -> GPU-hours -> $ calculator above
│   └── count_params.py            # CLI wrapper around config.count_params
└── tests/
    └── test_toy_pipeline.py       # hermetic, network-free, toy-scale smoke test
```

Two design decisions in this tree are worth calling out now, because they recur in every later chapter:

**Everything is CI-testable at toy scale, without a GPU or the network.** `configs/toy.yaml` is not a rounding-error version of `stack100m.yaml` — it deliberately uses a tiny vocabulary (256, effectively raw bytes), two layers, and a synthetic in-process corpus, so `tests/test_toy_pipeline.py` can run the *entire* pipeline shape — tokenize, pack, forward pass, one optimizer step, checkpoint, reload — on a laptop CPU in seconds. This is what "the book's CI smoke-tests the whole pipeline" means concretely: the tests prove the *code* runs correctly end-to-end; the prose in each chapter documents the *expected numbers* from the real, full-scale run, which you run yourself when you're ready to spend the GPU-hours.

```python
"""
configs/toy.py — the toy-scale config used by the CI smoke tests.

Same StackConfig class as the flagship, drastically shrunk. This is the
config CI actually instantiates; the flagship stack100m config is only
ever exercised by a human running the real GPU pipeline. configs/toy.yaml
is just the YAML mirror of TOY_CONFIG below, exactly as configs/stack100m.yaml
mirrors the flagship StackConfig -- the tests import this Python object directly.
"""
from stacklm.config import StackConfig

TOY_CONFIG = StackConfig(
    vocab_size=256,       # raw bytes, no BPE merges needed for a smoke test
    d_model=32,
    n_layers=2,
    n_heads=4,
    n_kv_heads=2,          # still exercise the GQA code path (2:4 ratio, not 1:1)
    head_dim=8,
    intermediate_size=64,
    max_seq_len=64,
    use_qk_norm=True,      # exercise every stability feature the real config uses
    nope_every=4,
)

assert TOY_CONFIG.n_heads * TOY_CONFIG.head_dim == TOY_CONFIG.d_model
```

**The single-GPU path is the default, not an afterthought.** A 101M-parameter model fits comfortably, with room to spare for activations and optimizer state, on a single 80GB A100 — you do not need ZeRO sharding, tensor parallelism, or pipeline parallelism to train Stack-100M, and Chapter 14.7's loop is deliberately a *single-process* training script. We include a scale-out note pointing to DDP and then FSDP for the reader who wants to reproduce the recipe on a bigger model across multiple GPUs (cross-linking [Distributed Training I: Data Parallelism, DDP, ZeRO & FSDP](../03-pretraining/05-distributed-data-parallel.html) and [Distributed Training II: Tensor, Pipeline, Sequence & Expert Parallelism](../03-pretraining/06-distributed-model-parallel.html)), but it is explicitly *not* the flagship path — one of the lessons of this capstone is that a huge amount of the modern LLM stack (distributed training, expert parallelism, multi-node checkpointing) becomes unnecessary machinery at 100M scale, and knowing what you can skip is as valuable as knowing how to use it.

!!! tip "Practitioner tip: start on the toy config, always"
    Before you spend a single GPU-hour of rented compute, run the *entire* pipeline — tokenizer training, data packing, model forward/backward, one optimizer step, a checkpoint save and reload — on `configs/toy.yaml`, on your laptop CPU, with the tiny synthetic corpus. Every real bug you will hit in a training run (a shape mismatch in the GQA reshape, an off-by-one in document-boundary masking, a checkpoint that does not actually restore the optimizer state) reproduces identically at toy scale, in seconds, for free. Debugging a shape error after 40 minutes of A100 time is an expensive way to learn something a CPU smoke test would have told you in half a second.

---

## Key Takeaways

!!! key "Key Takeaways"
    - Stack-100M is a 2026 update to Karpathy's nanoGPT/llm.c GPT-2 reproduction: same "train it yourself, on a budget" spirit, but built with the architecture, data, and training-regime advances discovered since — the result is a categorically stronger model at a similar parameter count.
    - Three forces explain the gap: **massive over-training** past Chinchilla-optimal (deployment cost, not training-FLOP cost, is what you're minimizing), **data quality** (FineWeb-Edu/Cosmopedia-style curation over raw scale), and **architecture/optimizer transfer** from frontier labs (GQA, RoPE+NoPE, QK-norm, SwiGLU, Muon, WSD).
    - The 2025–2026 small-model landscape — SmolLM2/3, Qwen3-0.6B, MobileLLM, and component borrowing from DeepSeek, Kimi K2, GLM, and Liquid LFM2 — is not a research curiosity; it is the current default recipe for small models, and Stack-100M implements it directly.
    - The canonical config is fixed: `d_model=512`, `n_layers=30`, `n_heads=8`, `n_kv_heads=2` (GQA), `intermediate=1408` (SwiGLU), `vocab_size=32768`, tied embeddings, ≈101M total parameters — verified by hand, not asserted.
    - "Deep-and-thin" (30 narrow layers rather than fewer wide ones) is a deliberate, MobileLLM-motivated choice, not a default; vocabulary size is likewise a first-class parameter-budget decision at this scale, not copied from a larger model.
    - The flagship budget — 1×A100, ≈15–25 GPU-hours, ≈USD 40–100 — falls directly out of the $6ND$ FLOP rule and the A100's ≈312 TFLOP/s bf16 peak at a realistic 35–55% MFU; you can and should reproduce this arithmetic yourself before spending the money.
    - "Narrow but real" is the capstone's honest target: not a general chatbot, but a model whose narrow-domain, tool-scaffolded outputs (arithmetic, retrieval-grounded QA) are genuinely verifiable, not merely plausible.
    - The entire project lives in one coherent, growing repository — package `stacklm` under `capstone/` — with a hermetic, network-free toy-scale config that exercises every code path in seconds, so every real bug is caught before a GPU-hour is spent.

## Further Reading

- Hoffmann et al., *Training Compute-Optimal Large Language Models* ("Chinchilla"), 2022.
- Liu et al., *MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases*, 2024.
- Penedo et al., *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale*, HuggingFace, 2024.
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (RoPE), 2021.
- Kazemnejad et al., *The Impact of Positional Encoding on Length Generalization in Transformers* (NoPE), 2023.
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, 2023.
- Shazeer, *GLU Variants Improve Transformer* (SwiGLU), 2020.
- Zhang & Sennrich, *Root Mean Square Layer Normalization* (RMSNorm), 2019.
- Press & Wolf, *Using the Output Embedding to Improve Language Models* (weight tying), 2017.
- DeepSeek-AI, *DeepSeek-V2* (Multi-head Latent Attention) and *DeepSeek-V3* (Multi-Token Prediction) technical reports, 2024.
- Gloeckle et al., *Better & Faster Large Language Models via Multi-token Prediction*, 2024.
- Jordan et al., *Muon: An Optimizer for Hidden Layers in Neural Networks*, 2024; Moonshot AI, *Kimi K2* technical report (MuonClip / QK-clip), 2025.
- Hu et al., *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies* (WSD schedule), 2024.
- Karpathy, `nanoGPT` and `llm.c` repositories — the original small-budget GPT-2 reproduction this capstone updates.

Everything in this chapter is fixed; the next one puts it to work. Chapter 14.2 builds the data pipeline — streaming, filtering, deduplicating, and packing the 20B tokens this model will actually learn from — starting from the pretraining-data mechanics in [Pretraining Data: Sources, Crawling & The Data Pipeline](../03-pretraining/01-pretraining-data.html) and [Data Mixing, Domain Weighting & Curriculum](../03-pretraining/14-data-mixing-curriculum.html).

## Exercises

**1.** A colleague argues that since Chinchilla proved 20 tokens per parameter is *compute-optimal*, training Stack-100M on ~200 tokens per parameter is simply wasteful and you should either stop at ~2B tokens or make the model bigger. In two or three sentences, explain the flaw in this argument, and name the one condition under which your colleague would actually be right.

??? note "Solution"
    The flaw is that Chinchilla-optimal minimizes loss *per unit of training compute* — it treats training as the only cost. A deployed model's dominant cost is *inference*, which recurs on every one of potentially billions of calls, whereas training is paid once. When parameter count (hence serving cost and latency) is the fixed constraint you care about, you want the lowest loss *at that fixed size*, and the data-scaling term $B/D^{\beta}$ keeps improving well past 20 tokens/param — just less per training FLOP than growing $N$ would. Paying ~10x the Chinchilla-optimal token count once, to lower loss at a cheap-to-serve 101M parameters forever, is the right trade for anything you will actually deploy.

    Your colleague is right only in the case Chinchilla itself answers: you have a fixed *training*-FLOP budget and you are *not* going to serve the model enough times to amortize extra training compute (e.g. a one-off research run whose only goal is the lowest loss for that FLOP budget). Then growing $N$ toward the compute-optimal frontier beats over-training a smaller model.

**2.** Using only the spec table (`d_model=512`, `n_heads=8`, `n_kv_heads=2`, `head_dim=64`), compute the number of parameters in the four attention projection matrices of a single Stack-100M block. Then compute what that block's attention would cost if it used full multi-head attention (all heads keep their own K and V) instead of GQA, and state how many parameters GQA saves across all 30 layers.

??? note "Solution"
    Query width is $n_{\text{heads}}\cdot d_h = 8\times 64 = 512 = d$, and KV width under GQA is $n_{\text{kv}}\cdot d_h = 2\times 64 = 128$. The four projections are $Q: d\times d$, $K: d\times d_{kv}$, $V: d\times d_{kv}$, $O: d\times d$:

    $$
    q = 512\times 512 = 262{,}144,\quad k = v = 512\times 128 = 65{,}536,\quad o = 512\times 512 = 262{,}144
    $$

    $$
    \text{attn}_{\text{GQA}} = 262{,}144 + 65{,}536 + 65{,}536 + 262{,}144 = 655{,}360 \;(\approx 0.655\text{M}).
    $$

    Full MHA gives K and V the full query width (512), so all four matrices are $512\times 512$:

    $$
    \text{attn}_{\text{MHA}} = 4\times 512^2 = 4\times 262{,}144 = 1{,}048{,}576 = 4d^2 \;(\approx 1.049\text{M}).
    $$

    Per-block saving is $1{,}048{,}576 - 655{,}360 = 393{,}216$. Across all 30 layers:

    $$
    30 \times 393{,}216 = 11{,}796{,}480 \approx 11.8\text{M parameters}.
    $$

    That is roughly 11.6% of the whole ~101M model saved on parameters alone — and the KV-cache saving at inference time (2 KV heads instead of 8) is the 4x reduction that actually motivates GQA.

**3.** Reproduce the flagship budget arithmetic, but for a *first, untuned* run that achieves only **35% MFU** instead of the 50% used in the chapter's worked example. Use $N = 101.35\times 10^6$, $D = 20\times 10^9$ tokens, the $6ND$ FLOP rule, and the A100 bf16 peak of $312$ TFLOP/s. Give the wall-clock hours and, at USD 1.75/GPU-hour, the dollar cost of the pretraining pass.

??? note "Solution"
    Total training compute:

    $$
    C = 6ND = 6\times(101.35\times 10^6)\times(20\times 10^9) \approx 1.216\times 10^{19}\ \text{FLOPs}.
    $$

    Achieved throughput at 35% MFU:

    $$
    0.35 \times 3.12\times 10^{14} = 1.092\times 10^{14}\ \text{FLOP/s}.
    $$

    Wall-clock:

    $$
    t = \frac{1.216\times 10^{19}}{1.092\times 10^{14}} \approx 1.114\times 10^{5}\ \text{s} \approx 30.9\ \text{hours}.
    $$

    Cost: $30.9 \times 1.75 \approx \text{USD } 54$. This matches the chapter's remark that a less-tuned ~35% MFU attempt pushes the pretraining pass alone to roughly 31 hours — comfortably explaining why the documented flagship envelope (USD 40–100) leaves headroom above the ~USD 33–43 tuned/50%-MFU core number for restarts, evaluation, and the data/tokenizer passes.

**4.** The spec chooses `vocab_size=32768`. Suppose you instead reused a 65,536-token vocabulary from a larger model's tokenizer, keeping the 30 transformer blocks and `d_model=512` unchanged and keeping embeddings tied. Compute the new tied-embedding parameter count, the new model total, and the fraction of the model the embedding table now occupies. Comment on why this makes vocabulary a first-class decision at 100M scale.

??? note "Solution"
    Tied embedding table at the new vocab:

    $$
    V d = 65{,}536 \times 512 = 33{,}554{,}432 \approx 33.55\text{M}.
    $$

    The transformer blocks and final norm are unchanged. From the chapter's accounting, all 30 blocks plus the final norm total $101{,}353{,}728 - 16{,}777{,}216 = 84{,}576{,}512$ parameters (i.e. everything except the original 16.78M embedding). New total:

    $$
    84{,}576{,}512 + 33{,}554{,}432 = 118{,}130{,}944 \approx 118.1\text{M}.
    $$

    Embedding fraction:

    $$
    \frac{33{,}554{,}432}{118{,}130{,}944} \approx 0.284 = 28.4\%.
    $$

    Doubling the vocab from 32,768 to 65,536 added ~16.8M parameters — more than a sixth of the *original* model — all spent on a lookup table before any block computes anything, and pushing the embedding from 16.6% to 28.4% of the total. At 100M scale the embedding is a large, fixed fraction of the budget, so vocabulary size directly trades against how much capacity is left for actual computation (depth/width). It cannot be copied thoughtlessly from a larger model, where the same table is a rounding error against billions of block parameters. Chapter 14.3 develops the full tradeoff curve.

**5.** The `count_params` function assumes tied embeddings. Extend it so it correctly reports the total when `cfg.tie_embeddings` is `False` (an untied model adds a separate $V\times d$ output projection / LM head). Add the branch, and report the untied total for the default Stack-100M shape. Your code should stay consistent with the chapter's style.

??? note "Solution"
    Only the final assembly needs to change: when embeddings are untied, add one more $V\times d$ matrix for the LM head. A minimal, drop-in edit to the accounting block:

    ```python
    def count_params(cfg: StackConfig) -> dict:
        embed = cfg.vocab_size * cfg.d_model

        q_width = cfg.n_heads * cfg.head_dim
        kv_width = cfg.n_kv_heads * cfg.head_dim
        attn_per_block = (
            cfg.d_model * q_width      # Q
            + cfg.d_model * kv_width   # K
            + cfg.d_model * kv_width   # V
            + q_width * cfg.d_model     # O
        )
        mlp_per_block = 3 * cfg.d_model * cfg.intermediate_size
        rmsnorm_per_block = 2 * cfg.d_model
        qk_norm_per_block = (2 * cfg.head_dim) if cfg.use_qk_norm else 0
        final_norm = cfg.d_model

        per_block = attn_per_block + mlp_per_block + rmsnorm_per_block + qk_norm_per_block
        all_blocks = per_block * cfg.n_layers

        # Untied models pay for a separate V x d output projection (LM head);
        # tied models reuse the input embedding, saving exactly `embed` params.
        lm_head = 0 if cfg.tie_embeddings else cfg.vocab_size * cfg.d_model

        total = embed + all_blocks + final_norm + lm_head
        return {
            "embedding": embed,
            "lm_head (untied)": lm_head,
            "all_blocks (x n_layers)": all_blocks,
            "final_norm": final_norm,
            "total": total,
        }
    ```

    For the default shape with `tie_embeddings=False`, `lm_head = 32768 * 512 = 16,777,216`, so:

    $$
    N_{\text{untied}} = 101{,}353{,}728 + 16{,}777{,}216 = 118{,}130{,}944 \approx 118.1\text{M}.
    $$

    Untying costs an extra 16.8M parameters (~16.6% of the tied model) for no change in the transformer's computation — which is exactly the saving the spec's `tie_embeddings=True` default captures. (Note this untied total coincides with Exercise 4's number: adding a second 32,768-row table costs the same 16.78M as widening one shared table to 65,536 rows.)

**6.** The default config sets `nope_every=4`, meaning every 4th layer omits rotary position embeddings entirely (NoPE), following SmolLM3. Conceptually, (a) explain *why* interleaving NoPE layers is expected to help the model generalize to sequences longer than the 2048-token pretraining context, and (b) explain why this is essentially free in parameters — i.e. why omitting RoPE on a layer does not change that layer's parameter count.

??? note "Solution"
    (a) RoPE encodes position by rotating each query and key by an angle proportional to its absolute position, so an attention score depends on the *relative* offset between tokens — but the rotation frequencies are calibrated to the range of positions seen during training (up to 2048). Past that range, the phases enter a regime the model never saw, hurting extrapolation. A NoPE layer applies *no* positional rotation at all, so its attention is exactly translation-invariant: it depends only on content, not on absolute index, and therefore behaves identically whether the sequence is length 2048 or 8192. Interleaving such layers (every 4th) gives the network a positional-signal-free pathway that generalizes cleanly beyond the training length, while the RoPE layers still supply the fine-grained relative-position information needed for local ordering — the combination extrapolates better than either alone, which is why SmolLM3 adopted it and why Stack-100M's context can be extended to 8192 in mid-training (Ch. 14.8).

    (b) RoPE is a *parameter-free* operation: it rotates Q and K by fixed, precomputed sinusoidal angles that depend only on position and the fixed `rope_theta`, with no learnable weights. Omitting it on a layer simply skips that rotation; the layer's Q/K/V/O projections, MLP, and norms — the only things that carry parameters — are unchanged. So `nope_every` shifts *which* layers rotate their queries and keys, but every layer, RoPE or NoPE, has exactly the same `attn_per_block` count computed in Exercise 2. That is why `count_params` never references `nope_every`.
