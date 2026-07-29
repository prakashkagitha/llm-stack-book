# 14.1 The Capstone: Building Stack-100M, and the 2026 Small-Model Landscape

In his `nanoGPT` and `llm.c` projects, Andrej Karpathy showed that with a clean implementation and a few tricks, you could reproduce GPT-2 (124M parameters) from scratch for a startlingly small amount of compute — on the order of a single 8×A100 node-hour for the 124M size, and roughly USD 100–200 for a careful, well-tuned reproduction on rented hardware. It was a landmark moment: training a real, GPT-class language model stopped being something only a well-funded lab could do, and became a weekend project with a credit card. That single demonstration re-anchored an entire generation of engineers' intuition for what training actually costs.

This capstone is the 2026 update to that demonstration — and the model you will build is *categorically* better than the 2019 GPT-2 it descends from, at the same parameter count. Since Karpathy's reproduction, the field learned three things that changed what a small model can do: train it on far more data than "optimal," curate that data far more aggressively, and borrow the architecture and optimizer tricks that frontier labs spent hundreds of millions of dollars discovering. This chapter lays out *why* those three forces work, surveys the 2025–2026 small-model landscape that proves it (SmolLM2/SmolLM3, Qwen3-0.6B, MobileLLM, Karpathy's own `nanochat`, the `modded-nanogpt` speedrun, and the component transfer from DeepSeek, Kimi, GLM, and Liquid AI), and gives you the exact specification, budget, and repository map for the model we will build together across the next eleven chapters: **Stack-100M**.

Every number in this chapter — and every chapter after it — is fixed by one specification. We will not re-derive the architecture here; we cite it, verify its parameter count by hand, verify its compute budget by hand, and point forward to where each piece is taught in depth. If you want the single source of truth for the whole capstone, it is `stacklm/config.py`, reproduced in full in this chapter's parameter-accounting section; every later chapter imports it unchanged.

---

## Why Build a 100M-Parameter Model From Scratch

It is worth asking directly: in a world of one-API-call frontier models, why spend eleven chapters training a *small* one yourself?

**Because ownership of every layer is the only way to actually understand the stack.** You can read this book's chapters on tokenization, attention, GQA, RoPE, Muon, WSD schedules, DPO, and GRPO in isolation, and each will make sense on its own. But nothing forces you to *reconcile* them — to discover that your tokenizer's vocabulary size trades off against your embedding parameter budget, that your data mix determines what your mid-training decay phase can inject, that your optimizer choice changes how sensitive your model is to the learning rate you picked in a chapter you read three weeks ago — the way building one coherent artifact does. A production LLM has hundreds of interacting design decisions; a 100M model, built by you, has all the same *kinds* of decisions at a scale you can hold in your head and debug on a single GPU.

**Because the economics are now real, not hypothetical.** The flagship path in this capstone trains Stack-100M on a single rented A100 for on the order of 30–45 GPU-hours — call it USD 40–100 for the pretraining pass at typical 2026 cloud rates, and on the order of USD 100–150 for the whole project including data, scaling ladder, mid-training, post-training, and the agent. That is a number you can actually spend. We derive it from first principles later in this chapter, including the term most napkin calculations quietly drop.

**Because "narrow but real" is a more honest and more useful target than "impressive demo."** We are not going to pretend Stack-100M is a general-purpose chatbot — at this scale, that would be a lie, and this book does not lie to you about capability. Instead we aim it at something a 100M model can genuinely do well: answer questions grounded in a small retrieval corpus, call a calculator correctly, solve narrow verifiable tasks. That target is modest, but it is *real*, and getting there end to end teaches you more than a bigger model you only ever fine-tuned.

**Because knowing what to hand-roll and what to import is a skill.** Every stage of this project has a mature open-source library behind it — HF `tokenizers`, `datatrove`, `torch.compile`, FlashAttention, TRL, veRL, vLLM, `lm-evaluation-harness`, `llama.cpp`. We build from scratch to learn the mechanism, then name the library you would actually reach for at work, and say exactly what that library does that our 200-line version does not. There is a table for this later in the chapter; it is one of the most practically useful things in Part XIV.

By the end of Part XIV you will have taken one model through the entire lifecycle this book covers: raw text in, a tool-using agent out, running on a laptop.

### What "end-to-end" concretely means here

To make that concrete, "end-to-end" is not a slogan — it is this exact list of things you will have built with your own hands and your own compute, in order:

1. A byte-level BPE tokenizer, trained from scratch on your own data sample, with nine special tokens reserved up front for the chat and tool stages.
2. A streaming data pipeline: download, quality-filter, deduplicate, tokenize, pack into fixed-length training shards with document-aware masking.
3. A deep-and-thin transformer implementation with GQA, RoPE+NoPE, QK-norm, and SwiGLU, matching a fixed ~101M-parameter budget you can verify by hand.
4. A miniature scaling-law study — your own Chinchilla-style ladder of tiny models — used to *justify* the final config rather than accept it on faith.
5. A hybrid Muon+AdamW optimizer with a Warmup-Stable-Decay schedule, trained with a real pretraining loop: bf16 autocast, gradient accumulation, checkpoint/resume, and measured MFU.
6. A mid-training phase that anneals on higher-quality data during the WSD decay and extends the context window to 8192.
7. Post-training: SFT with a chat template, DPO on preference pairs, and a narrow RLVR/GRPO run on a verifiable task.
8. A ReAct tool-use agent, taught by distilling trajectories from a larger teacher model, that can search a small corpus and answer questions grounded in it.
9. Honest evaluation, int4 quantization, and a model that generates text on your own laptop's CPU.

That is the map for the rest of Part XIV. This chapter is the only one that does not build a piece of the model — it builds your understanding of *why* this is the right model to build, and *what* every later chapter is going to hand you.

---

## The Finished Artifact and the Map of Part XIV

The finished artifact has a name — **Stack-100M** — and it lives in a repository laid out as the Python package `stacklm`, in the `capstone/` directory. Every later chapter extends the same package; nothing is thrown away and rewritten. Here is the roadmap, with each chapter's focus taken directly from the capstone specification, so you can see how the pieces fit before you build any of them:

| Ch. | Focus | Produces | Package path |
|---|---|---|---|
| 14.1 | Overview & landscape (this chapter) | Spec, budget, repo skeleton | `stacklm/config.py` |
| 14.2 | Pretraining data | FineWeb-Edu/Cosmopedia/code/math mix, dedup, packed `.bin` shards | `stacklm/data/` |
| 14.3 | Tokenizer | 32,768-vocab byte-level BPE, trained from scratch | `stacklm/tokenizer/` |
| 14.4 | Architecture | The Stack-100M transformer: GQA, RoPE+NoPE, QK-norm, SwiGLU, MLA/MTP as options | `stacklm/model/` |
| 14.5 | Mini scaling laws | A fitted $L(N,D)$ from a small model ladder, justifying the 101M/20B choice | `stacklm/scaling/` |
| 14.6 | Optimizer & schedule | Muon (2D weights) + AdamW hybrid, MuonClip/QK-clip, WSD schedule | `stacklm/optim/` |
| 14.7 | The pretraining loop | Full single-GPU training run, checkpointing, MFU measurement | `stacklm/train/` |
| 14.8 | Mid-training | WSD decay-phase annealing, long-context extension to 8192 | `stacklm/mid/` |
| 14.9 | Post-training | SFT, DPO, narrow RLVR/GRPO | `stacklm/post/` |
| 14.10 | Agentic | ReAct loop, tool distillation, the "auto-research" demo | `stacklm/agent/` |
| 14.11 | Evaluation & serving | Honest capability report, int4 quantization, laptop inference | `stacklm/eval/`, `stacklm/serve/` |
| 14.12 | Retrospective | Cost accounting, reproducibility checklist, scaling to 1B | — |

Each of those chapters cross-links back to the relevant deep-dive chapter elsewhere in the book — the capstone teaches *integration*, the main chapters teach *mechanism*. Where this chapter references a mechanism in passing, you will find the full treatment linked.

{{fig:capstone-lifecycle-arc}}

### Narrow but real: setting honest expectations

We use the phrase "narrow but real" throughout Part XIV, so it is worth defining precisely, because it is the single most important expectation-setting sentence in this capstone.

**Narrow** means we do not claim Stack-100M is a general-purpose assistant. It will not reliably discuss history, write correct code for novel problems, or reason robustly across arbitrary domains — a 100M-parameter model, even over-trained on 20B tokens, does not have the representational capacity for that, and no amount of clever post-training changes the underlying scaling laws covered in [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html). Anyone who tells you a 100M model is a general chatbot is either confused or selling something.

**Real** means that within a scaffolded, narrow domain — arithmetic with a calculator tool, question-answering grounded in a small local corpus via retrieval, following a fixed chat format — the model's outputs are genuinely correct, not merely plausible-sounding. "Real" is a claim about *verifiability*: when the agent in Chapter 14.10 answers a question, you can check the answer against the source it retrieved, or against a verifiable reward function, exactly the way Chapter 14.9's RLVR run and Chapter 14.10's ReAct agent are built. That is a much stronger and more honest claim than "reads like a chatbot."

The entire arc of Part XIV — over-training, data quality, mid-training capability injection, narrow post-training, tool distillation — is aimed at maximizing the *real* half of that phrase without ever pretending away the *narrow* half.

### If you are using this book as a CS336 companion

Stanford's CS336, *Language Modeling from Scratch*, is organized around a sequence of build-it-yourself assignment units: tokenizer and transformer basics; systems (custom kernels, FlashAttention, distributed training); scaling laws; data curation; and alignment/RL. Part XIV walks the same ground as one continuous project rather than five detached assignments — Ch. 14.3 is the tokenizer unit, Ch. 14.4/14.7 the transformer-and-training unit, Ch. 14.5 the scaling-laws unit, Ch. 14.2 the data unit, and Ch. 14.9 the alignment unit — while the deep-dive parts of this book (Parts I–IV especially, including [Writing GPU Kernels with Triton](../04-kernels-efficiency/04-triton-kernels.html) and [FlashAttention I: IO-Awareness & The Online Softmax](../04-kernels-efficiency/02-flash-attention-1.html)) supply the systems material the capstone deliberately consumes as a library rather than reimplements. If you want the kernels-from-scratch experience, do Part IV's exercises *before* Ch. 14.7 and swap your own kernel into the training loop.

---

## The 2025–2026 Small-Model Landscape

Karpathy's GPT-2 reproduction trained a 124M-parameter model on roughly 10B tokens of WebText — about 80 tokens per parameter, on data that was scraped and lightly filtered but not aggressively curated, with an architecture (learned absolute positions, LayerNorm, GELU MLP, full multi-head attention) that has since been improved on almost every axis. Stack-100M is architecturally, algorithmically, and data-wise a different animal at a similar parameter count. The proof that this is not just our project's optimism is that the same recipe has been demonstrated repeatedly by frontier labs and open-source teams over 2024–2026, at parameter counts from a few hundred million down to well under a billion:

- **SmolLM2 and SmolLM3** (HuggingFace, 2024–2025) are a family of small dense models trained on aggressively curated, heavily over-trained data mixes (built on FineWeb-Edu and Cosmopedia, the same families Stack-100M uses — see [Pretraining Data: Sources, Crawling & The Data Pipeline](../03-pretraining/01-pretraining-data.html)). SmolLM3 in particular popularized **NoPE** — omitting rotary position embeddings on a subset of layers — as a length-generalization trick at small scale, which Stack-100M adopts directly (every 4th layer, per the config below).
- **Qwen3-0.6B** (Alibaba, 2025) demonstrates that a sub-billion dense model, trained with the same modern component stack used at the frontier — GQA, RMSNorm, SwiGLU, a large, carefully mixed pretraining corpus — can be a genuinely capable base for downstream fine-tuning, not a toy.
- **MobileLLM** (Liu et al., Meta, 2024) is the paper that made the "deep-and-thin" case explicit and quantitative: at a *fixed* parameter budget, more layers with a narrower hidden dimension consistently outperforms fewer, wider layers, plus embedding-sharing tricks to spend the parameter budget on depth instead of vocabulary. The same deep-and-thin bias shows up in the GLM family and in Qwen3-0.6B's aspect ratio. This is the single design decision behind Stack-100M's 30 layers at `d_model=512` rather than, say, 12 layers at `d_model=1024`.
- **`modded-nanogpt`** (Keller Jordan and contributors, 2024–2025) is the NanoGPT *speedrun*: a public, continuously-beaten leaderboard for reaching a fixed GPT-2-grade validation loss on FineWeb in minimum wall-clock on a fixed 8-GPU node. It matters here because it is the empirical provenance of much of Stack-100M's stability stack — the **Muon** optimizer was developed there, and the record-holding configurations accumulated **QK-norm**, **logit soft-capping**, and value/embedding residual tricks precisely because they let the run survive a much higher learning rate. When Chapter 14.6 tells you QK-norm is what *buys* the aggressive LR, that is the lineage.
- **`nanochat`** (Karpathy, 2025) is the closest existing analogue to this capstone, and you should read it alongside Part XIV. It is a single dependency-light repository that runs the *whole* pipeline — train a BPE tokenizer, pretrain on FineWeb-Edu, mid-train on conversational/multiple-choice/tool-use data, SFT, an optional RL pass, then serve a web chat UI — with a headline "speedrun" tier costing on the order of USD 100 on a rented 8×H100 node.
- **Component transfer from frontier labs** is the last leg. Techniques originally built to make 100B+-parameter training runs stable and efficient turn out to work — and matter — at 100M scale too, because the underlying arithmetic (attention-logit blow-ups, KV-cache size, gradient conditioning) does not care how many parameters your model has. Stack-100M borrows: **MLA** (Multi-head Latent Attention) from **DeepSeek-V2**, taught as an alternative to GQA; **MuonClip** (QK-clip), the fix that let **Kimi K2** (Moonshot AI, 2025) train stably with the Muon optimizer at scale; the **WSD** (Warmup-Stable-Decay) schedule popularized by **MiniCPM** and used by **DeepSeek**; and the gated short-convolution-plus-attention hybrid block design explored by **Liquid AI's LFM2** (2025), which we implement as an optional block variant.

!!! note "Aside: how Stack-100M differs from nanochat, deliberately"
    `nanochat` and Stack-100M share a thesis, so it is worth being precise about where they diverge — the differences are the reason Part XIV exists rather than a pointer to a repo.

    - **Hardware floor.** `nanochat`'s documented tier assumes an 8×H100 node. Stack-100M's flagship is *one* rented A100, and the consumer/Colab tiers go lower still. Nothing in Part XIV requires a multi-GPU box, and Chapter 14.7 explains exactly which parts of the distributed-training stack you get to skip at this scale.
    - **Size and shape.** `nanochat`'s speedrun model is several hundred million parameters and conventionally proportioned; Stack-100M is ~101M and deliberately deep-and-thin (30 × 512), which is a different — and, at fixed parameters, better-motivated — point on the aspect-ratio curve.
    - **We fit our own scaling law.** Chapter 14.5 trains a ladder of tiny models and fits $L(N,D)$ before spending the flagship budget, instead of taking the shape on faith. That is the single biggest pedagogical addition.
    - **Optional frontier components.** MLA, MTP, and an LFM2-style gated-convolution mixer are implemented as switchable variants, so you can measure what DeepSeek's tricks actually buy at 100M.
    - **Different end point.** `nanochat` ends at a web chat UI. Stack-100M ends at a narrow, verifiable, retrieval-and-calculator agent quantized to int4 and running on your laptop CPU — because that is what a 100M model can honestly do well.
    - **Exposition.** Every mechanism here is derived in prose and math before it appears in code; `nanochat` is (excellently) code-first.

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

{{tool:scaling-law-optimal}}

### Force 2 — data quality over raw quantity

GPT-2's WebText was scraped, filtered for outbound Reddit links with a minimum karma, and deduplicated — a reasonable 2019 pipeline, but crude by 2026 standards. The modern pretraining data stack (built out fully in [Data Cleaning, Deduplication & Quality Filtering](../03-pretraining/02-data-cleaning-dedup.html) and [Synthetic Data for Pre- and Post-Training](../03-pretraining/15-synthetic-data.html)) does two things GPT-2's data did not:

- **Classifier-based educational filtering.** FineWeb-Edu (Penedo et al., HuggingFace, 2024) trains a classifier to score web documents by educational value and keeps only the high-scoring tail of Common Crawl. The result is a corpus with dramatically higher information density per token than raw web text — fewer boilerplate pages, navigation menus, and low-content spam diluting every gradient step.
- **Synthetic, dense, on-topic text.** Cosmopedia (HuggingFace) generates textbook- and story-style synthetic content with a larger teacher model, specifically to give a small model clean, well-structured exposition of concepts that natural web text states only in passing, if at all.

At fixed token count, cleaner and denser data means every one of your 20B training tokens is doing more work — you are not spending gradient steps learning to ignore boilerplate. Note that the *libraries* matter as much as the idea: FineWeb itself was produced with HuggingFace's `datatrove`, a distributed text-processing framework whose MinHash, URL-dedup, and quality-filter blocks are the reference implementations of everything Chapter 14.2 builds by hand.

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
| `intermediate` | 1408 | SwiGLU hidden size, $\approx 2.75 \cdot d_{\text{model}}$, a multiple of 64 |
| `max_seq_len` (pretrain) | 2048 | extended to 8192 in mid-training (Ch. 14.8) |
| `rope_theta` | 10000 | rescaled for long-context in mid-training |
| tied embeddings | yes | input embedding = output projection |
| total params | ≈ 101.35M | verified by hand below |

The components, each cited to where it comes from and cross-linked to the chapter that develops it fully:

- **Pre-norm residual blocks, RMSNorm** (Zhang & Sennrich, 2019) — cheaper than LayerNorm, no re-centering statistic, empirically just as stable. See [The Transformer Block: Norms, Residuals, MLPs & Activations](../02-transformer/06-transformer-block.html).
- **RoPE with NoPE on every 4th layer** — rotary position embeddings (Su et al., 2021) on most layers, and no positional encoding at all (Kazemnejad et al., 2023) on the interleaved subset, following SmolLM3. Interleaving improves length generalization: the NoPE layers let attention be exactly translation-invariant, which turns out to help extrapolation beyond the training context length. See [Positional Encodings: Sinusoidal, Learned, RoPE & ALiBi](../02-transformer/05-positional-encoding.html).
- **GQA — grouped-query attention**, 2 KV heads shared across 8 query heads (Ainslie et al., 2023) — a 4× reduction in KV-cache size versus full multi-head attention at essentially no quality cost at this scale. See [Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html).
- **QK-norm** — RMSNorm applied to queries and keys before the attention dot product, which keeps attention logits well-scaled and stabilizes training at the higher learning rates the Muon optimizer enables (Henry et al., 2020; heavily validated in the `modded-nanogpt` speedruns).
- **SwiGLU** gated MLP (Shazeer, 2020) in place of a plain feed-forward block.
- **Tied input/output embeddings** (Press & Wolf, 2017) — the same 32768×512 table serves as both the token embedding and the final logit projection, saving 16.8M parameters that would otherwise be nearly a sixth of the model.
- **z-loss and optional logit soft-cap** — a small penalty on the logsumexp of the output logits (z-loss), plus an optional Gemma-2-style tanh soft-cap, both purely for numerical stability of the softmax at the extremes of training.
- **Bias-free linear layers throughout.** Every `nn.Linear` in the model — Q, K, V, O, and all three SwiGLU matrices — is constructed with `bias=False`, following essentially every modern open decoder. This is a real design decision, not an oversight, and the parameter accounting below depends on it.

Two further components are implemented and taught, but are **options**, not the default path — you will build them, understand the trade-off, and can swap them in:

- **MLA (Multi-head Latent Attention)**, from DeepSeek-V2 (2024) — compresses the KV cache into a low-rank latent representation, a further reduction beyond what GQA buys, at the cost of a more involved attention computation.
- **MTP (Multi-Token Prediction)**, from DeepSeek-V3 (2024) and Gloeckle et al. (2024) — an auxiliary head trained to predict the token *after* the next one, giving a denser training signal per forward pass.

Both are covered as explicit "if you want DeepSeek's trick" upgrades in Chapter 14.4 once you have the base architecture working, alongside a Liquid LFM2-style gated-convolution mixer block as a third option.

### Verifying the parameter count by hand

The whole point of a capstone is that you should never take a number like "≈101M parameters" on faith. Here is the exact accounting, and code that reproduces it.

$$
N_{\text{total}} = \underbrace{V d}_{\text{tied embed}} \;+\; L\Big(\underbrace{2d^2 + 2\, d\, d_{kv}}_{\text{attn (GQA)}} + \underbrace{3\, d\, d_{\text{ff}}}_{\text{SwiGLU}} + \underbrace{2d + 2 d_h}_{\text{norms + QK-norm}}\Big) \;+\; \underbrace{d}_{\text{final norm}}
$$

with $V=32768$, $d=512$, $L=30$, $d_{\text{ff}}=1408$, $d_{kv} = n_{\text{kv}}\cdot d_h = 2\times 64 = 128$, and $d_h = 64$. The attention term is written out explicitly because it is **not** simply $4d^2$: GQA makes the K and V projections only $d_{kv}=128$ wide, while Q and the output projection O stay full-width, so the four attention matrices sum to $d\cdot d + d\cdot d_{kv} + d\cdot d_{kv} + d\cdot d = 2d^2 + 2 d\, d_{kv}$, not $4d^2$. The $2 d_h$ term is the pair of tiny QK-norm scale vectors (over `head_dim`), which the code counts and the spec table rounds away. There are no bias vectors anywhere, per the design decision above.

```python
"""
stacklm/config.py -- the canonical configuration for Stack-100M.

This is the single source of truth for the model's shape. Every later
chapter (architecture, pretraining loop, mid-training, post-training,
serving) imports StackConfig from here -- nothing is redefined downstream.
"""
from dataclasses import dataclass


@dataclass
class StackConfig:
    # --- core shape (fixed for the whole capstone, Ch. 14.1 spec) ---
    vocab_size: int = 32768          # byte-level BPE, trained from scratch (Ch. 14.3)
    d_model: int = 512               # narrow -- the "thin" in deep-and-thin
    n_layers: int = 30               # deep
    n_heads: int = 8                 # query heads
    n_kv_heads: int = 2              # GQA: 4 query heads share each KV head
    head_dim: int = 64               # n_heads * head_dim == d_model (8*64=512)
    intermediate: int = 1408         # SwiGLU hidden size, ~2.75 * d_model
    max_seq_len: int = 2048          # pretrain context; -> 8192 in mid-training (Ch. 14.8)
    rope_theta: float = 10000.0      # RoPE base; rescaled for long-context (Ch. 14.8)

    # --- stability / small-model tricks (Ch. 14.4) ---
    tie_embeddings: bool = True      # input embed == output projection (Press & Wolf, 2017)
    qk_norm: bool = True             # RMSNorm on Q, K before the attention dot product
    nope_every: int = 4              # every 4th layer skips RoPE entirely (SmolLM3-style)
    norm_eps: float = 1e-5
    z_loss_coef: float = 1e-4        # penalty on logsumexp(logits) for softmax stability
    logit_soft_cap: float = 0.0      # Gemma-2-style tanh soft-cap on logits; 0.0 = off
    attn_soft_cap: float = 0.0       # optional soft-cap on attention logits; 0.0 = off

    # --- optional efficiency variants, OFF by default (Ch. 14.4 "DeepSeek's trick") ---
    use_mla: bool = False            # Multi-head Latent Attention (DeepSeek-V2) instead of GQA
    mtp_heads: int = 0               # Multi-Token Prediction aux heads (DeepSeek-V3); 0 = off

    def head_groups(self) -> int:
        """How many query heads share each KV head (4 for the flagship config)."""
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"
        return self.n_heads // self.n_kv_heads


def count_params(cfg: StackConfig) -> dict:
    """Exact, hand-verifiable parameter accounting for a StackConfig.

    This function is the executable version of the arithmetic in the
    Ch. 14.1 spec table. Run it and you should see ~101.35M total.

    It assumes the default path: GQA attention, no MTP heads, and
    bias-free nn.Linear everywhere. Those assumptions are enforced, not
    silently ignored -- a parameter counter that lies is worse than none.
    """
    if cfg.use_mla:
        raise NotImplementedError(
            "count_params() covers the GQA path only; MLA replaces Q/K/V with "
            "down/up latent projections. Use Stack100M(cfg).num_params() (Ch. 14.4)."
        )
    if cfg.mtp_heads:
        raise NotImplementedError(
            "count_params() covers mtp_heads=0 only; each MTP head adds an extra "
            "transformer block plus its head. Use Stack100M(cfg).num_params()."
        )

    # Tied embedding table: one V x d matrix does double duty as the
    # input embedding lookup AND the final logit projection.
    embed = cfg.vocab_size * cfg.d_model

    # --- attention projections, GQA-shaped: Q is full-width, K/V are
    # only n_kv_heads wide, O projects the full-width attention output
    # back down to d_model. No biases. ---
    q_width = cfg.n_heads * cfg.head_dim      # = d_model by construction (8*64=512)
    kv_width = cfg.n_kv_heads * cfg.head_dim  # = 128 (GQA shrinks this vs. q_width)
    q_proj = cfg.d_model * q_width
    k_proj = cfg.d_model * kv_width
    v_proj = cfg.d_model * kv_width
    o_proj = q_width * cfg.d_model
    attn_per_block = q_proj + k_proj + v_proj + o_proj

    # --- SwiGLU MLP: three d_model <-> intermediate matrices
    # (gate, up, down) -- SwiGLU needs three, not two like a plain MLP. ---
    mlp_per_block = 3 * cfg.d_model * cfg.intermediate

    # --- norms: RMSNorm has one learnable scale vector per normalized
    # dimension. Two per block (pre-attn, pre-mlp) over d_model, plus
    # one final model-level norm. QK-norm adds two tiny ones over head_dim. ---
    rmsnorm_per_block = 2 * cfg.d_model
    qk_norm_per_block = (2 * cfg.head_dim) if cfg.qk_norm else 0
    final_norm = cfg.d_model

    per_block = attn_per_block + mlp_per_block + rmsnorm_per_block + qk_norm_per_block
    all_blocks = per_block * cfg.n_layers

    total = embed + all_blocks + final_norm
    # NOTE: if tie_embeddings were False, add another `embed` here for
    # an untied LM head. We do not -- that's a deliberate 16.8M-param
    # saving, see the worked example below and Exercise 5.

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

Running this (`python3 -m stacklm.config`) prints an embedding of 16,777,216 (16.78M), an attention block of 655,360 (0.655M), an MLP block of 2,162,688 (2.163M), a per-block total (with norms) of 2,819,200, all 30 blocks summing to 84,576,000, and a grand total of **101,353,728 ≈ 101.35M parameters** — matching the ≈101M figure in the spec table. Chapter 14.4 asserts this exact integer against `Stack100M(cfg).num_params()`, so if you ever change the architecture and forget to update the accounting, the test fails.

{{tool:param-flop-counter}}

!!! example "Worked example: what a 50k vocabulary would cost you"
    Suppose instead we had picked the more conventional 50,000-token vocabulary size (roughly what GPT-2/GPT-3-family tokenizers use) instead of 32,768. The tied embedding table would be $50{,}000 \times 512 = 25{,}600{,}000$ parameters — **25.6M**, versus 16.78M at our chosen vocab size. That is an extra **8.8M parameters**, all spent on the embedding table before a single transformer block has processed anything.

    Keeping the 30 blocks and the final norm fixed at 84,576,512 parameters, the new total is $84{,}576{,}512 + 25{,}600{,}000 = 110{,}176{,}512 \approx 110.2\text{M}$, and the embedding is now

    $$
    \frac{25{,}600{,}000}{110{,}176{,}512} \approx 0.232 = 23.2\%
    $$

    of the model, versus $16{,}777{,}216 / 101{,}353{,}728 \approx 16.6\%$ for our 32,768 table. (You will also see this quoted as "≈26%," which is 25.6M measured against a *nominal* 100M budget rather than against the model you would actually get; always say which denominator you mean.) Nearly a quarter of your parameter count spent on lookup rather than computation is a lot. This is exactly why vocabulary size is a first-class architectural decision at small scale, not an afterthought copied from a larger model's tokenizer; Chapter 14.3 works through the full tradeoff curve when we build the tokenizer.

!!! interview "Interview Corner"
    **Q:** You are given a fixed parameter budget and told to train the best model you can, to be deployed and served many times. Explain why you would deliberately train past the Chinchilla-optimal token count, and quantify what you are trading away.

    **A:** Chinchilla-optimal minimizes loss *per unit of training compute* — it answers "how do I get the best model for a fixed training FLOP budget," treating training as the only cost. But a deployed model's cost is dominated by *inference*, which recurs on every call, while training is a one-time cost. Given a fixed parameter count (which determines serving cost and latency), you want to minimize loss *for that fixed size*, and the data-scaling term of the loss curve, $B/D^\beta$, keeps improving well past the Chinchilla ratio — it just costs more training FLOPs per unit of improvement than it would to instead grow $N$. You are trading extra, one-time training compute (in Stack-100M's case, roughly 10× the Chinchilla-optimal token count) for a lower loss at a serving cost you have already fixed. The trade only makes sense if you actually plan to serve the model enough times that the one-time training cost is amortized — which is the deployment case, not the "just want the lowest-loss model for this FLOP budget" case Chinchilla itself answers.

{{fig:stack100m-param-budget}}

---

## Budget: FLOPs, MFU, GPU-Hours and Dollars

### The compute tiers

Stack-100M is designed to be trained at three tiers of hardware, with the same code and the same architecture — only the token budget, batch size, and precision change.

| Tier | Hardware | Pretrain wall-clock | Cost | Role |
|---|---|---|---|---|
| **Flagship** | 1× A100 (80GB), rented | ≈30–45 GPU-hours | ≈USD 40–100 | The documented, full-fidelity walkthrough |
| **Consumer** | RTX 4090 / 3090 (24GB) | ≈2–4× flagship wall-clock | ≈USD 0 if owned | Same recipe, more gradient accumulation |
| **On-ramp** | Free Colab (T4, 16GB) | scaled-down run | USD 0 | Smaller model, fewer tokens — explicitly a subset, not the full run |

The flagship number is what we derive and defend below. The 4090 tier runs the identical recipe: less HBM means a smaller micro-batch and more gradient-accumulation steps to reach the same effective batch size, and no bf16-Tensor-Core throughput advantage is lost — the 4090 has full bf16 support — so the main cost is wall-clock, not correctness. The Colab T4 tier is different in kind, not just degree: the T4 is a 2018 Turing-generation part with no native bf16 tensor-core path, so the on-ramp config trains in fp16 mixed precision instead of bf16 (with the corresponding loss-scaling care this requires — see [Mixed Precision, bf16 & FP8 Training](../03-pretraining/08-mixed-precision-fp8.html)) and deliberately shrinks the model and token budget so a meaningful run finishes inside Colab's free-tier session limits.

### From FLOPs to GPU-hours: the derivation, including the term people drop

The flagship budget is not a marketing number, but it is also not *quite* the one-line $6ND$ calculation you will see quoted everywhere. Stack-100M is deep, thin, and trained at a fairly long context — which is exactly the regime where the attention-score term stops being a rounding error. We will do it properly.

!!! example "Worked example: FLOPs to GPU-hours to dollars"
    **Setup.** $N = 101{,}353{,}728$ parameters (from `count_params`), $D = 20 \times 10^9$ tokens ($D/N \approx 197$ tokens/parameter, ≈10× past Chinchilla), $L = 30$ layers, $s = 2048$ context, $d_q = n_{\text{heads}} \cdot d_h = 512$.

    **Term 1 — the dense matmuls.** Every parameter participates in one multiply-accumulate per token in the forward pass (2 FLOPs) and roughly twice that in the backward pass (4 FLOPs), giving the familiar rule:

    $$
    C_{\text{dense}} / \text{token} = 6N = 6 \times 101{,}353{,}728 \approx 6.081 \times 10^{8}\ \text{FLOPs}
    $$

    (A subtlety worth noticing: the tied embedding table is counted once in $N$, and that is exactly right here — the *lookup* costs no FLOPs, but the same matrix is used as the output projection, which is a genuine $V \times d$ GEMM per token. One matmul use, one count.)

    **Term 2 — the attention scores, which $6ND$ ignores.** The $QK^\top$ and $AV$ matmuls do not involve parameters at all, so no per-parameter rule can see them. Per layer over a length-$s$ sequence, the forward pass costs $2s^2 d_q$ for $QK^\top$ plus $2s^2 d_q$ for $AV$; **causal masking halves both** (a FlashAttention-style kernel simply never computes the masked blocks), and the backward pass is about twice the forward. So per token, per layer, $6 s d_q$, and over the whole model:

    $$
    C_{\text{attn}} / \text{token} = 6 L s\, d_q = 6 \times 30 \times 2048 \times 512 = 1.887 \times 10^{8}\ \text{FLOPs}
    $$

    That is **+31% on top of $6N$** — not negligible. The ratio is $\frac{6Lsd_q}{6N} \approx \frac{L s d_q}{N}$, which grows linearly in the context length and, at fixed $N$, grows with depth-over-width. Stack-100M is the highest-attention-fraction model in this book precisely *because* it is deep, thin, and long-context. (At the 8192-token mid-training context of Chapter 14.8, the same term becomes $7.55\times10^8$ per token — **+124%**, i.e. attention costs more than every weight matrix combined. Exercise 7 makes you redo the budget there.)

    **Total training compute:**

    $$
    C = (6N + 6Lsd_q)\, D \approx (7.969 \times 10^{8}) \times (2\times 10^{10}) \approx 1.59 \times 10^{19}\ \text{FLOPs}
    $$

    versus $1.216 \times 10^{19}$ from the naive $6ND$ — a 31% under-estimate you would have paid for in wall-clock.

    **Convert to wall-clock.** The A100's bf16 Tensor Core dense peak is $\pi = 312$ TFLOP/s $= 3.12\times10^{14}$ FLOP/s. Define MFU (Model FLOPs Utilization) as achieved model FLOP/s divided by $\pi$, where "model FLOPs" is exactly the $C$ we just derived. For a `d_model=512` model, a *well-tuned* single-GPU run — `torch.compile`, a FlashAttention SDPA backend, a micro-batch large enough to keep the GEMMs fat — lands in roughly the **0.30–0.45** band (see the reality check below for why it is not higher). Taking 0.35:

    $$
    \text{achieved} = 0.35 \times 3.12\times10^{14} = 1.092\times10^{14}\ \text{FLOP/s}
    $$

    $$
    t = \frac{1.59\times10^{19}}{1.092\times10^{14}} \approx 1.46\times10^{5}\ \text{s} \approx 40.5\ \text{hours}
    $$

    At the optimistic end (MFU 0.45) it is ≈31.5 hours; at a mediocre 0.30 it is ≈47 hours.

    **Convert to dollars.** A100-80GB on-demand rents for roughly USD 1.50–2.00/GPU-hour in 2026; interruptible/spot capacity on GPU marketplaces is often nearer USD 0.80–1.20. So 31–47 hours is on the order of **USD 35–85** for the pretraining pass — the documented "≈USD 40–100" flagship figure, arrived at honestly.

    This is exactly the kind of napkin calculation you should be able to reproduce for any training run before you spend the money. It has five inputs — parameter count, token budget, context length × depth, peak hardware FLOP/s, and MFU — and no step you cannot check.

{{tool:train-compute-estimator}}

!!! warning "Common pitfall: quoting 6ND at 50% MFU and believing it"
    You will see this capstone's budget quoted elsewhere (including in `capstone/PLAN.md` and the repo README) as **15–25 GPU-hours**. That number is $6ND$ alone at 50% MFU: $1.216\times10^{19} / (0.50 \times 3.12\times10^{14}) \approx 21.7$ hours. It is the *optimistic end*, and it is optimistic twice over — it drops the +31% attention term and it assumes an MFU this shape will not reach on an A100.

    Treat **≈35 GPU-hours** as the planning number and **30–45** as the honest band. Chapter 14.7 instruments the loop and reports the *measured* tokens/second and MFU, which is the only number that actually settles the argument. The dollar headline survives either way, because the pretraining pass is the one stage where you can trade wall-clock for spot pricing.

### MFU reality check: deep-and-thin costs you throughput

The chapter has now told you twice that depth-over-width is the right call at fixed parameters. Here is the bill for it, because the capstone does not hide trade-offs.

At `d_model=512`, the largest GEMM in a block is the SwiGLU up/gate projection: $(B\cdot s, 512) \times (512, 1408)$. The $K$ dimension of 512 is small by tensor-core standards, so each matmul spends a larger fraction of its time on prologue/epilogue and memory traffic than the same FLOPs would in a $d_{\text{model}}=4096$ model. Then multiply the *count* of kernels by 30 layers: 30 RMSNorm pairs, 30 RoPE applications, 30 SiLU-and-multiply gates, 30 residual adds, all of them memory-bandwidth-bound and all of them paying kernel-launch latency. A 10-layer, 1024-wide model of the same parameter count would do the same arithmetic in a third as many launches. **Deep-and-thin buys loss-per-parameter and pays for it in MFU** — which is fine, because you train once and serve forever, but you should not be surprised by the wall-clock.

The levers that actually move MFU here, in rough order of payoff (all developed in [Kernel Fusion, torch.compile, CUDA Graphs & Compilers](../04-kernels-efficiency/09-compilers-fusion.html) and [The Roofline Model & Performance Engineering](../04-kernels-efficiency/01-roofline-performance.html)):

1. **`torch.compile(model)`** — TorchInductor fuses the norm/RoPE/SiLU/residual elementwise chains into a handful of kernels per block. At 30 narrow layers this is worth more than at any other point in the book; `mode="max-autotune"` additionally tunes the small GEMM tiles.
2. **`torch.nn.functional.scaled_dot_product_attention`** — PyTorch 2.x dispatches this to a FlashAttention-2 kernel when the shapes and dtype allow, giving you IO-aware, causal-skipping attention without writing CUDA. Use `is_causal=True` rather than materializing a mask, or you lose exactly the causal halving we counted above. (The `flash-attn` package, or your own Triton kernel from Part IV, are the drop-in alternatives.)
3. **A bigger micro-batch.** Throughput at small $d_{\text{model}}$ is mostly about making $M = B \cdot s$ large enough that the GEMM is not launch-latency-bound. Fill the 80GB, then use gradient accumulation to reach the target ≈0.5M-token effective batch.
4. **bf16 autocast plus a fused optimizer** (`torch.optim.AdamW(..., fused=True)` for the AdamW half of the hybrid), so the optimizer step does not become a visible fraction of a fast step.
5. **CUDA graphs** (via `torch.compile(mode="reduce-overhead")`) if, after all of the above, you can still see launch gaps in a profile.

Dimensions are already chosen to cooperate: 512, 1408, and 64 are all multiples of 64, so no tensor-core tile is padded.

!!! tip "Practitioner tip: measure MFU on day one, not on day two"
    Chapter 14.7's loop prints tokens/second and MFU every logging interval. Run 200 steps at the real config before you launch the real job, compute your own $t = C / (\text{MFU} \times \pi)$, and decide *then* whether to spend another hour on `torch.compile` or just start. An hour of tuning that lifts MFU from 0.25 to 0.35 saves roughly 17 GPU-hours on this run — a ~20× return.

### The full-project envelope: everything the pretraining number leaves out

The pretraining pass is the headline, but it is not the bill. Here is the whole project, order-of-magnitude, at an assumed USD 1.80/GPU-hour for A100-80GB. Chapter 14.12 reconciles this against what you actually spent.

| Stage (chapter) | Resource | Rough cost |
|---|---|---|
| Data acquisition, filter, dedup (14.2) | CPU-bound; streams ~150–250 GB of raw text, keeps ~40 GB of `uint16` shards | USD 0–10 (CPU box + egress) |
| Tokenizer training (14.3) | CPU; BPE merges over a ~1–2 GB sample, a few CPU-hours | ~USD 1 |
| **Encoding 20B tokens** (14.2/14.3) | CPU, embarrassingly parallel — see the warning below | USD 1–5 with a fast encoder; *days* without |
| Scaling ladder {4M, 9M, 19M, 43M} (14.5) | ~4–5 GPU-hours | ~USD 8 |
| **Pretrain, 20B tokens (14.7)** | **≈30–45 GPU-hours** | **≈USD 55–85** |
| Mid-training: anneal + 8192 context, ~3B tokens (14.8) | ~8–12 GPU-hours (attention is +124% at 8192 — see Exercise 7) | ~USD 15–22 |
| SFT + DPO (14.9) | ~2–3 GPU-hours | ~USD 4–6 |
| RLVR / GRPO (14.9) | ~3–5 GPU-hours, **rollout-dominated**, not gradient-dominated | ~USD 6–9 |
| Teacher trajectories for distillation (14.10) | An 8B-class open model (e.g. a Qwen3-8B-class checkpoint) served locally with **vLLM** on the same rented GPU for ~1–2 hours, or a hosted API | ~USD 2–10 |
| Agent SFT on filtered traces (14.10) | ~1 GPU-hour | ~USD 2 |
| Eval, int8/int4 quantization, export (14.11) | ~1–2 GPU-hours; artifact shrinks ~406 MB fp32 → ~63 MB int4 | ~USD 3 |
| Storage + egress, one month, ~200 GB | — | ~USD 5 |
| **Total** | **≈50–70 GPU-hours** | **≈USD 100–150 on-demand; nearer USD 70–90 on spot** |

Two rows deserve their own warnings, because they are the two places readers actually get stuck.

!!! warning "Common pitfall: your Python BPE encoder is slower than your A100"
    Chapter 14.3 has you implement byte-level BPE from scratch, and you *should* — training the merges is where the mechanism lives, and a pure-Python trainer on a 1–2 GB sample is perfectly tolerable. But **encoding 20B tokens with a pure-Python inner loop is not**: at a realistic few hundred KB/s per process it is a multi-day job that can genuinely cost more wall-clock than the training run it feeds.

    The fix is standard practice, not a cop-out: train the merges yourself, then hand the merge table to a fast encoder for the bulk pass. Either load your merges into HuggingFace `tokenizers` (a Rust `ByteLevel` BPE model with a `merges` file — its `encode_batch` is multi-threaded and roughly two orders of magnitude faster than pure Python), or keep your own encoder and shard the corpus across a `multiprocessing.Pool` of all your cores. Chapter 14.3 shows both. The mechanism is yours; the throughput is the library's.

!!! warning "Common pitfall: disk, not dollars, is what kills the data stage"
    20B `uint16` tokens is 40 GB of packed shards — fine. The raw text you stream to *produce* those tokens is several times larger, and a rented box often ships with 100 GB of local disk. Stream and tokenize in one pass (HF `datasets` in `streaming=True` mode never materializes the full corpus), write shards incrementally, and delete raw text as you go. Chapter 14.2's pipeline is written this way for exactly this reason.

---

## The Repository, and What We Hand-Roll vs. What You Would Use

Every later chapter adds files to the same tree. Here is the actual layout of `capstone/` — no `configs/` directory, no YAML indirection: the configuration *is* a Python dataclass, because a config you can `assert` on is worth more than a config you can lint.

```text
capstone/
├── PLAN.md                       # canonical Stack-100M spec (source of truth)
├── README.md                     # runbook: full-run commands per stage
├── requirements.txt              # hermetic smoke set + (commented) real-run extras
├── smoke_test.py                 # end-to-end toy-scale pipeline, CPU-only, hermetic
└── stacklm/
    ├── config.py                 # StackConfig, count_params, toy_config  -- this chapter
    ├── tokenizer/
    │   └── bpe.py                # byte-level BPE trainer + encode/decode  -- Ch. 14.3
    ├── data/
    │   ├── synthetic.py          # data mix spec, streaming sources, in-process toy corpus
    │   ├── pack.py               # pack to seq_len, document-aware causal masking
    │   ├── shard.py              # uint16 .bin shard writer
    │   └── dataset.py            # memmap-backed packed dataset            -- Ch. 14.2
    ├── model/
    │   ├── rmsnorm.py            # RMSNorm
    │   ├── rope.py               # rotary embeddings + the NoPE layer rule
    │   ├── swiglu.py             # SwiGLU MLP
    │   ├── attention.py          # GQA + QK-norm + RoPE/NoPE dispatch
    │   ├── mla.py                # optional Multi-head Latent Attention (DeepSeek-V2)
    │   ├── mtp.py                # optional Multi-Token Prediction heads
    │   ├── block.py              # pre-norm residual block (+ LFM2-style conv mixer)
    │   └── transformer.py        # full Stack100M model, tied embeddings   -- Ch. 14.4
    ├── scaling/
    │   └── fit.py                # fit L(N,D)=E+A/N^a+B/D^b, IsoFLOP profiles -- Ch. 14.5
    ├── optim/
    │   ├── muon.py               # Newton-Schulz orthogonalized Muon
    │   ├── qk_clip.py            # MuonClip / QK-clip                      -- Ch. 14.6
    │   ├── schedule.py           # WSD (warmup-stable-decay) and cosine
    │   └── build.py              # the Muon/AdamW param-group split
    ├── train/
    │   └── loop.py               # single-GPU pretraining loop, ckpt/resume -- Ch. 14.7
    ├── mid/
    │   └── continue_training.py  # decay-phase anneal + RoPE rescale to 8192 -- Ch. 14.8
    ├── post/
    │   ├── chat.py               # chat template over the reserved specials
    │   ├── sft.py                # supervised fine-tuning, assistant-only loss
    │   ├── dpo.py                # direct preference optimization
    │   └── grpo.py               # GRPO on a verifiable narrow task        -- Ch. 14.9
    ├── agent/
    │   ├── tools.py              # calculator, BM25 + hash-embedding retriever
    │   ├── react.py              # think -> act -> observe parsing and rendering
    │   ├── react_run.py          # the agent loop driver
    │   ├── distill.py            # teacher-trajectory generation + filtering
    │   └── stub_teacher.py       # offline teacher so CI never calls a network -- Ch. 14.10
    ├── eval/
    │   └── metrics.py            # perplexity, arithmetic, retrieval-QA probes
    └── serve/
        ├── quantize.py           # int8 -> int4 round-to-nearest quantization
        └── generate.py           # CPU generation                          -- Ch. 14.11
```

### Build it yourself to learn it; name the library you would actually ship

This is the pedagogical rule for the whole capstone, stated once: **we hand-roll to learn the mechanism, and we name the library you would reach for at work.** A from-scratch implementation you understand plus knowledge of which battle-tested package supersedes it is strictly more valuable than either alone — and at 100M scale the from-scratch version is genuinely fast enough to finish, which is why this project is possible at all.

| Stage | What we build from scratch | What you would use in production |
|---|---|---|
| BPE training + encoding | `stacklm/tokenizer/bpe.py`, pure stdlib | HF **`tokenizers`** (Rust BPE trainer, multi-threaded `encode_batch`), **`tiktoken`** for inference-side encoding, **SentencePiece** for unigram |
| Corpus acquisition | `stream_source()` over HF **`datasets`** in streaming mode | Same, plus **`datatrove`** (the pipeline that built FineWeb) or NVIDIA **NeMo-Curator** for distributed filtering |
| Dedup | exact-hash + MinHash-LSH in `stacklm/data/` | `datatrove`'s MinHash blocks, or **`text-dedup`** |
| Attention kernel | explicit `softmax(QK^T/√d)V` math, then `F.scaled_dot_product_attention` | **`F.scaled_dot_product_attention`** (FlashAttention-2 backend), **`flash-attn`**, **xFormers**; **Triton** if you need a custom variant |
| Fusion / compilation | eager PyTorch first, then one `torch.compile` call | **`torch.compile`** (TorchInductor), CUDA graphs, **TransformerEngine** for FP8 on Hopper+ |
| Optimizer | `stacklm/optim/muon.py` (Newton–Schulz) | **`KellerJordan/Muon`** reference implementation; `torch.optim.AdamW(fused=True)` for the 1D groups |
| Scaling-law fit | least-squares fit of $L(N,D)$ in NumPy | same idea; the analysis is bespoke everywhere |
| Distributed training | *not needed at 100M* — single process, by design | PyTorch **FSDP2** / DDP, **DeepSpeed** ZeRO, **Megatron-LM** for 3D parallelism |
| Checkpoint format | `torch.save` of model+optimizer+step+RNG | **`safetensors`**, `torch.distributed.checkpoint` for sharded state |
| Experiment tracking | stdout + a JSONL log | **Weights & Biases**, **TensorBoard**, MLflow |
| SFT / DPO | `stacklm/post/sft.py`, `dpo.py` | **TRL** (`SFTTrainer`, `DPOTrainer`), **Axolotl**, **LLaMA-Factory**, **Unsloth** — see [TRL: HuggingFace's RL Library](../06-rl-infra/03-trl.html) |
| RLVR / GRPO | `stacklm/post/grpo.py` | TRL `GRPOTrainer`, **veRL**, **OpenRLHF**, **Prime-RL** — see [veRL: HybridFlow & The Single-Controller Architecture](../06-rl-infra/04-verl.html) |
| Rollout generation for RL | our own batched sampler | **vLLM** or **SGLang** as the rollout engine (this is what veRL/TRL drive under the hood) — see [vLLM: Architecture, PagedAttention & Internals](../07-inference-serving/03-vllm-internals.html) |
| Teacher trajectories | injected `Callable` + offline stub for CI | **vLLM** serving an open 8B-class model locally, or a hosted API client |
| Agent scaffolding | `stacklm/agent/react.py`, ~200 lines | **LangChain** / **LlamaIndex** / **smolagents**; **MCP** for tool wiring — see [The Model Context Protocol (MCP)](../08-agents-harness/06-mcp.html) |
| Retrieval | BM25 + a hash-embedding retriever | **FAISS** / a vector DB, plus a real embedding model — see [Vector Databases & Approximate Nearest Neighbor Search](../09-rag-retrieval/02-vector-databases-ann.html) |
| Evaluation | `stacklm/eval/metrics.py` probes | EleutherAI **`lm-evaluation-harness`**, **HELM** — see [Building Eval Harnesses](../11-evaluation/03-eval-harnesses.html) |
| Quantization | round-to-nearest int8/int4 in `serve/quantize.py` | **GPTQ** / **AWQ** (AutoGPTQ, AutoAWQ, `llm-compressor`), **`bitsandbytes`**, **GGUF** via `llama.cpp` |
| Serving | `serve/generate.py`, CPU, no batching | **vLLM** / **SGLang** / **TensorRT-LLM** on GPU; **`llama.cpp`** or **ONNX Runtime** on a laptop |

Two things to notice. First, the "not needed at 100M" row is a real lesson: a huge amount of the modern LLM stack (ZeRO sharding, tensor/pipeline parallelism, multi-node checkpointing) becomes unnecessary machinery at this scale, and knowing what you can skip is as valuable as knowing how to use it. Chapter 14.7 includes a scale-out note pointing at [Distributed Training I: Data Parallelism, DDP, ZeRO & FSDP](../03-pretraining/05-distributed-data-parallel.html) and [Distributed Training II: Tensor, Pipeline, Sequence & Expert Parallelism](../03-pretraining/06-distributed-model-parallel.html) for the reader who wants to reproduce the recipe on a bigger model — but the flagship path is deliberately a single process.

Second, the dependency list is short *on purpose*. The full run needs `torch`, `numpy`, `datasets`, `huggingface_hub`, `safetensors`, and `tqdm`; CI needs even less. Everything else in the right-hand column is something you *could* adopt, and Part XIV tells you when it starts to pay.

### Everything is CI-testable at toy scale, without a GPU or the network

`smoke_test.py` runs the *entire* pipeline shape — train a tiny BPE, pack a synthetic corpus, build the model, assert the parameter count, pretrain, mid-train with a RoPE rescale, SFT, DPO, GRPO, distill an agent trajectory from an offline stub teacher, quantize to int4, and generate — on a laptop CPU in seconds, with no network access. The tests prove the *code* runs correctly end to end; the prose in each chapter documents the *expected numbers* from the real, full-scale run.

That only works if the toy config genuinely exercises the same code paths, which takes a little care:

```python
def toy_config() -> StackConfig:
    """Tiny config for the CPU smoke test.

    The point of a toy config is not "small"; it is "small AND still hits
    every branch". Two of these numbers are load-bearing for that:
      * n_kv_heads=2 with n_heads=4 keeps the GQA reshape non-trivial
        (a 1:1 ratio would silently pass even if the grouping were wrong).
      * nope_every=2, NOT 4: the layer rule is `(layer_idx + 1) % nope_every`,
        so with only 2 layers a nope_every=4 toy would take the RoPE branch
        on BOTH layers and never execute the NoPE path at all.
    """
    cfg = StackConfig(
        vocab_size=256,        # placeholder: the smoke test overrides this (see below)
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,          # exercise the GQA code path (4:2, not 1:1)
        head_dim=16,           # 4 * 16 == 64 == d_model
        intermediate=64,
        max_seq_len=64,
        qk_norm=True,          # exercise the QK-norm branch
        nope_every=2,          # layer 1 skips RoPE -> the NoPE branch runs
    )
    assert cfg.n_heads * cfg.head_dim == cfg.d_model
    return cfg
```

The `vocab_size` deserves a note, because it is the one place a toy config can quietly break a downstream chapter. Chapter 14.3 reserves **nine** special tokens up front — `<|bos|>`, `<|eos|>`, `<|pad|>`, the four chat-role markers, and the two tool tokens — and their integer ids are load-bearing for the chat template (Ch. 14.9) and the ReAct format (Ch. 14.10). A 256-entry "raw bytes" vocabulary has no room for them. So the smoke test trains a real toy tokenizer at `vocab_size=384` (256 byte tokens + 9 specials + a handful of learned merges) and then rebuilds the config from the tokenizer that actually exists:

```python
tok = StackTokenizer()
tok.train(text, vocab_size=384, special_tokens=SPECIAL_TOKENS)   # 256 bytes + 9 specials + merges
cfg = dataclasses.replace(toy_config(), vocab_size=tok.vocab_size, max_seq_len=96)
```

Deriving the model's `vocab_size` from the trained tokenizer rather than hard-coding it is the same discipline you want at full scale: the tokenizer is upstream of the model, so the model's embedding table should be *told* how big it is, never assume.

!!! tip "Practitioner tip: start on the toy config, always"
    Before you spend a single GPU-hour of rented compute, run the *entire* pipeline — tokenizer training, data packing, model forward/backward, one optimizer step, a checkpoint save and reload — on `toy_config()`, on your laptop CPU, with the tiny synthetic corpus. Every real bug you will hit in a training run (a shape mismatch in the GQA reshape, an off-by-one in document-boundary masking, a checkpoint that does not actually restore the optimizer state) reproduces identically at toy scale, in seconds, for free. Debugging a shape error after 40 minutes of A100 time is an expensive way to learn something a CPU smoke test would have told you in half a second.

---

## Key Takeaways

!!! key "Key Takeaways"
    - Stack-100M is a 2026 update to Karpathy's nanoGPT/llm.c GPT-2 reproduction — and a single-GPU, deeper-exposition sibling of his 2025 `nanochat`: same "train it yourself, on a budget" spirit, built with the architecture, data, and training-regime advances discovered since.
    - Three forces explain the gap: **massive over-training** past Chinchilla-optimal (deployment cost, not training-FLOP cost, is what you're minimizing), **data quality** (FineWeb-Edu/Cosmopedia-style curation over raw scale), and **architecture/optimizer transfer** from frontier labs and from the public `modded-nanogpt` speedruns (GQA, RoPE+NoPE, QK-norm, SwiGLU, Muon, WSD).
    - The canonical config is fixed: `d_model=512`, `n_layers=30`, `n_heads=8`, `n_kv_heads=2` (GQA), `intermediate=1408` (SwiGLU), `vocab_size=32768`, tied embeddings, bias-free linears, **101,353,728** parameters — verified by hand and asserted in code, not asserted in prose.
    - "Deep-and-thin" is a deliberate MobileLLM-motivated choice with a real cost: 30 narrow layers give better loss-per-parameter and *worse* MFU than 12 wide ones, because small-$K$ GEMMs and 30 layers of memory-bound elementwise work do not saturate an A100.
    - The training-compute rule is $C = (6N + 6Lsd_q)D$, not $6ND$. At $s=2048$ the attention term adds **+31%** for this shape; at the 8192 mid-training context it adds **+124%**. Any budget that quotes $6ND$ alone is an under-estimate, and it is worst exactly for deep, thin, long-context models.
    - The honest flagship envelope is ≈**30–45 GPU-hours** at 0.30–0.45 MFU (≈USD 40–100), and the *whole project* — data, tokenizer, scaling ladder, mid-training, post-training, distillation teacher, eval — is ≈50–70 GPU-hours and on the order of USD 100–150. The two non-GPU traps are pure-Python tokenization of 20B tokens and local disk during the data pass.
    - "Narrow but real" is the capstone's honest target: not a general chatbot, but a model whose narrow-domain, tool-scaffolded outputs (arithmetic, retrieval-grounded QA) are genuinely verifiable, not merely plausible.
    - We hand-roll every stage to learn the mechanism and name the production library that supersedes it — `tokenizers`, `datatrove`, `torch.compile`, FlashAttention/SDPA, TRL, veRL, vLLM, `lm-evaluation-harness`, `llama.cpp`. Knowing both, and knowing which parts of the distributed stack you can skip at 100M, is the point.

!!! sota "State of the Art & Resources (2026)"
    The "train a real GPT-class model on a budget" tradition Karpathy started now has a 2025–2026 generation of open small-model recipes to learn directly from — the same data mixes, architectures, and optimizers Stack-100M borrows are all publicly documented and reproducible.

    **Foundational work**

    - [Hoffmann et al., *Training Compute-Optimal Large Language Models* (2022)](https://arxiv.org/abs/2203.15556) — the Chinchilla scaling law this chapter's "over-train past compute-optimal" argument is built against.
    - [Liu et al., *MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases* (2024)](https://arxiv.org/abs/2402.14905) — the paper that made the "deep-and-thin" case quantitative; the direct justification for Stack-100M's 30×512 shape.
    - [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — the original small-budget GPT-2 reproduction this capstone updates; simplest reference for the base training loop shape.
    - [Chowdhery et al., *PaLM* (2022)](https://arxiv.org/abs/2204.02311) — the paper that defined **Model FLOPs Utilization** (MFU) as the honest efficiency metric, including the attention term this chapter refuses to drop.

    **Recent advances (2023–2026)**

    - [Ben Allal et al., *SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model* (2025)](https://arxiv.org/abs/2502.02737) — HuggingFace's own writeup of exactly the aggressive-curation, heavily-over-trained recipe this chapter cites.
    - [*SmolLM3: smol, multilingual, long-context reasoner*](https://huggingface.co/blog/smollm3) — the official HuggingFace blog post that introduced interleaved NoPE (every 4th layer) for length generalization, which Stack-100M adopts directly.
    - [Qwen Team, *Qwen3 Technical Report* (2025)](https://arxiv.org/abs/2505.09388) — the frontier-lab report behind Qwen3-0.6B, the sub-billion dense model this chapter cites as proof a small model can be a real base, not a toy.
    - [Kimi Team, *Kimi K2: Open Agentic Intelligence* (2025)](https://arxiv.org/abs/2507.20534) — the technical report documenting MuonClip (QK-clip), the fix that lets Muon train stably at scale, which Stack-100M's optimizer chapter reuses.
    - [Hu et al., *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies* (2024)](https://arxiv.org/abs/2404.06395) — introduces the Warmup-Stable-Decay (WSD) schedule Stack-100M's pretraining and mid-training phases use.
    - [*Introducing LFM2: The Fastest On-Device Foundation Models on the Market*](https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models) — Liquid AI's official announcement of the gated short-convolution-plus-attention hybrid block Stack-100M implements as an optional mixer variant.

    **Open-source & tools**

    - [karpathy/nanochat](https://github.com/karpathy/nanochat) — the closest existing end-to-end analogue to this capstone: tokenizer → pretrain → mid-train → SFT → RL → web serve, in one dependency-light repo, at a documented ~USD 100 tier on an 8×H100 node. Read it next to Part XIV.
    - [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) — the NanoGPT speedrun leaderboard; the empirical origin of the Muon + QK-norm + logit-soft-cap stack Stack-100M's config uses.
    - [KellerJordan/Muon](https://github.com/KellerJordan/Muon) — the reference implementation of the Muon optimizer (Newton–Schulz orthogonalized updates for 2D hidden-layer weights) that Stack-100M's hybrid optimizer is built on.
    - [karpathy/llm.c](https://github.com/karpathy/llm.c) — pure C/CUDA GPT-2/GPT-3 reproduction; the low-level companion to nanoGPT for understanding what a training step costs on real hardware.
    - [huggingface/datatrove](https://github.com/huggingface/datatrove) — the distributed text-processing library FineWeb was actually built with; the production version of Chapter 14.2's filtering and MinHash dedup.
    - [huggingface/tokenizers](https://github.com/huggingface/tokenizers) — the Rust BPE trainer and multi-threaded encoder you should hand your from-scratch merge table to when it is time to encode 20B tokens.
    - [huggingface/trl](https://github.com/huggingface/trl) — `SFTTrainer` / `DPOTrainer` / `GRPOTrainer`, the production counterparts to Chapter 14.9's from-scratch loops.
    - [vllm-project/vllm](https://github.com/vllm-project/vllm) — the rollout engine behind most 2026 RL stacks and the obvious way to serve the Chapter 14.10 teacher model locally.
    - [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — the standard harness for the benchmarks Chapter 14.11's honest capability report is measured against.
    - [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — GGUF quantization and CPU inference; the production path for the "runs on your laptop" payoff.
    - [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) — the classifier-filtered, high-educational-value web corpus that anchors Stack-100M's pretraining mix.

    **Go deeper**

    - [*Cosmopedia: how to create large-scale synthetic data for pre-training Large Language Models*](https://huggingface.co/blog/cosmopedia) — HuggingFace's own walkthrough of generating the dense, textbook-style synthetic data Stack-100M mixes in alongside FineWeb-Edu.
    - [Stanford CS336, *Language Modeling from Scratch*](https://stanford-cs336.github.io/) — the course whose assignment arc (tokenizer → transformer → systems → scaling laws → data → alignment) Part XIV walks as one continuous project.

## Further Reading

- Hoffmann et al., *Training Compute-Optimal Large Language Models* ("Chinchilla"), 2022.
- Chowdhery et al., *PaLM: Scaling Language Modeling with Pathways*, 2022 — the MFU definition used throughout this chapter.
- Liu et al., *MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases*, 2024.
- Penedo et al., *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale*, HuggingFace, 2024.
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (RoPE), 2021.
- Kazemnejad et al., *The Impact of Positional Encoding on Length Generalization in Transformers* (NoPE), 2023.
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, 2023.
- Shazeer, *GLU Variants Improve Transformer* (SwiGLU), 2020.
- Zhang & Sennrich, *Root Mean Square Layer Normalization* (RMSNorm), 2019.
- Press & Wolf, *Using the Output Embedding to Improve Language Models* (weight tying), 2017.
- Dao et al., *FlashAttention* and *FlashAttention-2* — the IO-aware attention kernels behind `F.scaled_dot_product_attention`.
- DeepSeek-AI, *DeepSeek-V2* (Multi-head Latent Attention) and *DeepSeek-V3* (Multi-Token Prediction) technical reports, 2024.
- Gloeckle et al., *Better & Faster Large Language Models via Multi-token Prediction*, 2024.
- Jordan et al., *Muon: An Optimizer for Hidden Layers in Neural Networks*, 2024; Moonshot AI, *Kimi K2* technical report (MuonClip / QK-clip), 2025.
- Hu et al., *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies* (WSD schedule), 2024.
- Karpathy, `nanoGPT`, `llm.c`, and `nanochat` repositories; Jordan et al., `modded-nanogpt` — the small-budget training lineage this capstone updates.

Everything in this chapter is fixed; the next one puts it to work. Chapter 14.2 builds the data pipeline — streaming, filtering, deduplicating, and packing the 20B tokens this model will actually learn from — starting from the pretraining-data mechanics in [Pretraining Data: Sources, Crawling & The Data Pipeline](../03-pretraining/01-pretraining-data.html) and [Data Mixing, Domain Weighting & Curriculum](../03-pretraining/14-data-mixing-curriculum.html).

## Exercises

**1.** A colleague argues that since Chinchilla proved 20 tokens per parameter is *compute-optimal*, training Stack-100M on ~200 tokens per parameter is simply wasteful and you should either stop at ~2B tokens or make the model bigger. In two or three sentences, explain the flaw in this argument, and name the one condition under which your colleague would actually be right.

??? note "Solution"
    The flaw is that Chinchilla-optimal minimizes loss *per unit of training compute* — it treats training as the only cost. A deployed model's dominant cost is *inference*, which recurs on every one of potentially billions of calls, whereas training is paid once. When parameter count (hence serving cost and latency) is the fixed constraint you care about, you want the lowest loss *at that fixed size*, and the data-scaling term $B/D^{\beta}$ keeps improving well past 20 tokens/param — just less per training FLOP than growing $N$ would. Paying ~10x the Chinchilla-optimal token count once, to lower loss at a cheap-to-serve 101M parameters forever, is the right trade for anything you will actually deploy.

    Your colleague is right only in the case Chinchilla itself answers: you have a fixed *training*-FLOP budget and you are *not* going to serve the model enough times to amortize extra training compute (e.g. a one-off research run whose only goal is the lowest loss for that FLOP budget). Then growing $N$ toward the compute-optimal frontier beats over-training a smaller model.

**2.** Using only the spec table (`d_model=512`, `n_heads=8`, `n_kv_heads=2`, `head_dim=64`), compute the number of parameters in the four attention projection matrices of a single Stack-100M block. Then compute what that block's attention would cost if it used full multi-head attention (all heads keep their own K and V) instead of GQA, and state how many parameters GQA saves across all 30 layers.

??? note "Solution"
    Query width is $n_{\text{heads}}\cdot d_h = 8\times 64 = 512 = d$, and KV width under GQA is $n_{\text{kv}}\cdot d_h = 2\times 64 = 128$. The four projections are $Q: d\times d$, $K: d\times d_{kv}$, $V: d\times d_{kv}$, $O: d\times d$ (no biases):

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

    That is roughly 11.6% of the whole ~101M model saved on parameters alone — and the KV-cache saving at inference time (2 KV heads instead of 8) is the 4x reduction that actually motivates GQA. Note that GQA does *not* reduce the attention-score FLOPs: the $QK^\top$ and $AV$ matmuls still run over all 8 query heads, with the 2 KV heads broadcast. GQA is a memory optimization, not a FLOP one.

**3.** Reproduce the flagship budget arithmetic — including the attention term — for a *first, untuned* run that achieves only **25% MFU**. Use $N = 101{,}353{,}728$, $D = 20\times 10^9$ tokens, $L=30$, $s=2048$, $d_q=512$, and the A100 bf16 peak of $312$ TFLOP/s. Give the wall-clock hours and, at USD 1.75/GPU-hour, the dollar cost of the pretraining pass. Then state how many hours you would have predicted using $6ND$ alone at 50% MFU, and explain the whole gap.

??? note "Solution"
    Per-token compute:

    $$
    6N = 6\times 101{,}353{,}728 \approx 6.081\times 10^{8},\qquad
    6Lsd_q = 6\times 30\times 2048\times 512 = 1.887\times 10^{8}
    $$

    $$
    C = (6.081\times10^{8} + 1.887\times10^{8})\times 2\times10^{10} \approx 1.594\times 10^{19}\ \text{FLOPs}.
    $$

    Achieved throughput at 25% MFU: $0.25 \times 3.12\times 10^{14} = 7.80\times 10^{13}$ FLOP/s.

    $$
    t = \frac{1.594\times 10^{19}}{7.80\times 10^{13}} \approx 2.04\times 10^{5}\ \text{s} \approx 56.8\ \text{hours}.
    $$

    Cost: $56.8 \times 1.75 \approx \text{USD } 99$ — the whole "USD 40–100 flagship" budget consumed by one untuned pretraining pass.

    The naive prediction was $1.216\times10^{19} / (0.50\times3.12\times10^{14}) \approx 21.7$ hours. The 2.6× gap decomposes cleanly into two independent factors: **1.31×** from the attention-score FLOPs that $6ND$ cannot see, and **2.0×** from the MFU assumption (0.50 vs 0.25); $1.31 \times 2.0 = 2.62$. The moral is that both factors are yours to control — the attention term shrinks if you shorten the context, and the MFU factor is bought back with `torch.compile`, an SDPA/FlashAttention backend, and a fatter micro-batch.

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

    Doubling the vocab from 32,768 to 65,536 added ~16.8M parameters — more than a sixth of the *original* model — all spent on a lookup table before any block computes anything, and pushing the embedding from 16.6% to 28.4% of the total. (Use the same convention as the chapter's 50k worked example: the new embedding over the *new* total.) At 100M scale the embedding is a large, fixed fraction of the budget, so vocabulary size directly trades against how much capacity is left for actual computation (depth/width). It cannot be copied thoughtlessly from a larger model, where the same table is a rounding error against billions of block parameters. Chapter 14.3 develops the full tradeoff curve.

**5.** The `count_params` function assumes tied embeddings. Extend it so it correctly reports the total when `cfg.tie_embeddings` is `False` (an untied model adds a separate $V\times d$ output projection / LM head). Add the branch, and report the untied total for the default Stack-100M shape. Your code should stay consistent with the chapter's style.

??? note "Solution"
    Only the final assembly needs to change: when embeddings are untied, add one more $V\times d$ matrix for the LM head. A minimal, drop-in edit to the accounting block:

    ```python
    def count_params(cfg: StackConfig) -> dict:
        if cfg.use_mla or cfg.mtp_heads:
            raise NotImplementedError("GQA / mtp_heads=0 path only; see Ch. 14.4.")

        embed = cfg.vocab_size * cfg.d_model

        q_width = cfg.n_heads * cfg.head_dim
        kv_width = cfg.n_kv_heads * cfg.head_dim
        attn_per_block = (
            cfg.d_model * q_width      # Q
            + cfg.d_model * kv_width   # K
            + cfg.d_model * kv_width   # V
            + q_width * cfg.d_model     # O
        )
        mlp_per_block = 3 * cfg.d_model * cfg.intermediate
        rmsnorm_per_block = 2 * cfg.d_model
        qk_norm_per_block = (2 * cfg.head_dim) if cfg.qk_norm else 0
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

**6.** The default config sets `nope_every=4`, meaning every 4th layer omits rotary position embeddings entirely (NoPE), following SmolLM3. Conceptually, (a) explain *why* interleaving NoPE layers is expected to help the model generalize to sequences longer than the 2048-token pretraining context, and (b) explain why this is essentially free in parameters — i.e. why omitting RoPE on a layer does not change that layer's parameter count. (c) The toy config in this chapter sets `nope_every=2` rather than 4. Why?

??? note "Solution"
    (a) RoPE encodes position by rotating each query and key by an angle proportional to its absolute position, so an attention score depends on the *relative* offset between tokens — but the rotation frequencies are calibrated to the range of positions seen during training (up to 2048). Past that range, the phases enter a regime the model never saw, hurting extrapolation. A NoPE layer applies *no* positional rotation at all, so its attention is exactly translation-invariant: it depends only on content, not on absolute index, and therefore behaves identically whether the sequence is length 2048 or 8192. Interleaving such layers (every 4th) gives the network a positional-signal-free pathway that generalizes cleanly beyond the training length, while the RoPE layers still supply the fine-grained relative-position information needed for local ordering — the combination extrapolates better than either alone, which is why SmolLM3 adopted it and why Stack-100M's context can be extended to 8192 in mid-training (Ch. 14.8).

    (b) RoPE is a *parameter-free* operation: it rotates Q and K by fixed, precomputed sinusoidal angles that depend only on position and the fixed `rope_theta`, with no learnable weights. Omitting it on a layer simply skips that rotation; the layer's Q/K/V/O projections, MLP, and norms — the only things that carry parameters — are unchanged. So `nope_every` shifts *which* layers rotate their queries and keys, but every layer, RoPE or NoPE, has exactly the same `attn_per_block` count computed in Exercise 2. That is why `count_params` never references `nope_every`.

    (c) Coverage. The layer rule is `use_rope = ((layer_idx + 1) % nope_every) != 0`. With `n_layers=2` and `nope_every=4`, the two layers evaluate `1 % 4` and `2 % 4`, both non-zero, so *both* take the RoPE branch and the NoPE code path is never executed by CI — the smoke test would happily pass with a broken NoPE implementation. Setting `nope_every=2` makes layer index 1 evaluate `2 % 2 == 0` and take the NoPE branch, so both paths are covered. A toy config's job is not to be small; it is to be small *and* branch-complete.

**7.** Chapter 14.8 extends the context from 2048 to 8192 for a mid-training pass of roughly 3B tokens. (a) Compute the attention-score FLOPs per token at $s=8192$ and express them as a fraction of $6N$. (b) Compute the total compute $C$ for the 3B-token pass and the wall-clock at 35% MFU on an A100. (c) Compare against what $6ND$ alone would have predicted, and say what this implies about which kernel you must be using.

??? note "Solution"
    (a) $6 L s d_q = 6 \times 30 \times 8192 \times 512 = 754{,}974{,}720 \approx 7.55\times10^{8}$ FLOPs/token, versus $6N \approx 6.081\times10^{8}$. The ratio is $7.55/6.081 \approx 1.24$, i.e. **+124%** — at 8192 the attention scores cost *more* than every weight matrix in the model combined. (The ratio is linear in $s$: quadrupling the context from 2048 quadruples 31% to 124%.)

    (b) Per token, $6.081\times10^{8} + 7.550\times10^{8} = 1.363\times10^{9}$ FLOPs. Over $D = 3\times10^{9}$ tokens:

    $$
    C = 1.363\times10^{9} \times 3\times10^{9} \approx 4.09\times10^{18}\ \text{FLOPs}.
    $$

    At 35% MFU: $4.09\times10^{18} / 1.092\times10^{14} \approx 3.75\times10^{4}$ s $\approx$ **10.4 hours**.

    (c) $6ND$ alone gives $6.081\times10^8 \times 3\times10^9 = 1.824\times10^{18}$ FLOPs, i.e. ≈4.6 hours — **less than half** the real figure. That gap is the whole reason long-context extension is expensive, and it is why you must run this phase through a memory- and IO-aware causal kernel: `F.scaled_dot_product_attention(..., is_causal=True)` on its FlashAttention-2 backend (or `flash-attn` directly). A naive implementation that materializes the full $8192\times8192$ score matrix per head does *twice* the FLOPs (no causal skipping) and, worse, allocates $O(s^2)$ activation memory per head, which will simply OOM an 80GB card at any useful batch size. See [FlashAttention I: IO-Awareness & The Online Softmax](../04-kernels-efficiency/02-flash-attention-1.html) and [Long-Context Pretraining & Context Extension](../03-pretraining/13-long-context-pretraining.html).

**8.** You have finished Stack-100M and your team asks you to rebuild the same pipeline as a maintained internal product, where "I wrote it myself" is a liability rather than a virtue. For each of these five stages, name the open-source library you would adopt and state, in one sentence, the specific thing it gives you that the from-scratch version does not: (a) tokenizer encoding of the pretraining corpus, (b) the attention kernel, (c) DPO training, (d) RLVR rollout generation, (e) laptop deployment.

??? note "Solution"
    (a) **HuggingFace `tokenizers`** — a Rust `ByteLevel` BPE implementation with multi-threaded `encode_batch`, roughly two orders of magnitude faster than a pure-Python inner loop, plus a stable serialized `tokenizer.json` format that every downstream tool can load. You can import your own trained merges into it, so you keep the vocabulary you designed.

    (b) **`torch.nn.functional.scaled_dot_product_attention`**, which dispatches to a **FlashAttention-2** kernel — tiled, IO-aware attention that never materializes the $s \times s$ score matrix (turning $O(s^2)$ activation memory into $O(s)$) and skips masked blocks under `is_causal=True`. `flash-attn` or a Triton kernel are the drop-in alternatives when you need a variant PyTorch does not dispatch.

    (c) **TRL** (`DPOTrainer`) — a maintained implementation of DPO and its variants (IPO, KTO, ORPO, SimPO) with reference-model handling, PEFT/LoRA integration, sequence packing, and multi-GPU support already correct, so preference-data plumbing is configuration rather than code. **Axolotl** or **LLaMA-Factory** wrap it in a YAML-driven runner.

    (d) **vLLM** (or **SGLang**) as the rollout engine, driven by **veRL**, **OpenRLHF**, or TRL's `GRPOTrainer` — continuous batching plus PagedAttention makes generation, which dominates RLVR wall-clock, several times faster than a naive `model.generate` loop, and the RL frameworks handle the weight-synchronization dance between the trainer and the rollout workers.

    (e) **`llama.cpp`** with a **GGUF** export — mature CPU/Metal/AVX kernels, a family of k-quant int4/int5 schemes far better than round-to-nearest, memory-mapped weights, and a portable single-binary runtime. **ONNX Runtime** is the alternative when you need to embed the model in a non-C++ host application.
