# 14.1 The Capstone: Building Stack-100M, and the 2026 Small-Model Landscape

In his `nanoGPT` and `llm.c` projects, Andrej Karpathy showed that with a clean implementation and a few tricks, you could reproduce GPT-2 (124M parameters) from scratch for a startlingly small amount of compute — on the order of a single 8×A100 node-hour for the 124M size, and roughly USD 100–200 for a careful, well-tuned reproduction on rented hardware. It was a landmark moment: training a real, GPT-class language model stopped being something only a well-funded lab could do, and became a weekend project with a credit card. That single demonstration re-anchored an entire generation of engineers' intuition for what training actually costs.

This capstone is the 2026 update to that demonstration — and the model you will build is *categorically* better than the 2019 GPT-2 it descends from, at the same parameter count. Since Karpathy's reproduction, the field learned three things that changed what a small model can do: train it on far more data than "optimal," curate that data far more aggressively, and borrow the architecture and optimizer tricks that frontier labs spent hundreds of millions of dollars discovering. This chapter lays out *why* those three forces work, surveys the 2025–2026 small-model landscape that proves it (SmolLM2/SmolLM3, Qwen3-0.6B, MobileLLM, Karpathy's own `nanochat`, the `modded-nanogpt` speedrun, and the component transfer from DeepSeek, Kimi, GLM, and Liquid AI), and gives you the exact specification, budget, and repository map for the model we will build together across the next eleven chapters: **Stack-100M**.

Every number in this chapter — and every chapter after it — is fixed by one specification. We will not re-derive the architecture here; we cite it, verify its parameter count by hand, verify its memory and compute budgets by hand, and point forward to where each piece is taught in depth. If you want the single source of truth for the whole capstone, it is `stacklm/config.py`, reproduced verbatim in this chapter's parameter-accounting section; every later chapter imports it unchanged.

---

## Why Build a 100M-Parameter Model From Scratch

It is worth asking directly: in a world of one-API-call frontier models, why spend eleven chapters training a *small* one yourself?

**Because ownership of every layer is the only way to actually understand the stack.** You can read this book's chapters on tokenization, attention, GQA, RoPE, Muon, WSD schedules, DPO, and GRPO in isolation, and each will make sense on its own. But nothing forces you to *reconcile* them — to discover that your tokenizer's vocabulary size trades off against your embedding parameter budget, that your data mix determines what your mid-training decay phase can inject, that your optimizer choice changes how sensitive your model is to the learning rate you picked in a chapter you read three weeks ago — the way building one coherent artifact does. A production LLM has hundreds of interacting design decisions; a 100M model, built by you, has all the same *kinds* of decisions at a scale you can hold in your head and debug on a single GPU.

**Because the economics are now real, not hypothetical.** The flagship path trains Stack-100M's 18B-token stable phase on a single rented A100 in **≈22–29 GPU-hours** — on the order of USD 25–50 at 2026 cloud rates — and the *whole project*, including data, the scaling ladder, mid-training, post-training, the distillation teacher and the storage bill, lands near **USD 90–100**. We derive both numbers from first principles later in this chapter, including the FLOP term most napkin calculations quietly drop, and [Chapter 14.12](../14-capstone/12-retrospective-and-scaleup.html) reconciles them against the itemized invoice.

**Because "narrow but real" is a more honest and more useful target than "impressive demo."** We are not going to pretend Stack-100M is a general-purpose chatbot — at this scale, that would be a lie, and this book does not lie to you about capability. Instead we aim it at something a 100M model can genuinely do well: answer questions grounded in a small retrieval corpus, call a calculator correctly, solve narrow verifiable tasks. That target is modest, but it is *real*, and getting there end to end teaches you more than a bigger model you only ever fine-tuned.

**Because knowing what to hand-roll and what to import is a skill.** Every stage of this project has a mature open-source library behind it — HF `tokenizers`, `datatrove`, `torch.compile`, FlashAttention, TRL, veRL, vLLM, `lm-evaluation-harness`, `llama.cpp`. We build from scratch to learn the mechanism, then name the library you would actually reach for at work, and say exactly what that library does that our 200-line version does not. There is a table for this later in the chapter; it is one of the most practically useful things in Part XIV.

By the end of Part XIV you will have taken one model through the entire lifecycle this book covers: raw text in, a tool-using agent out, running on a laptop.

### What "end-to-end" concretely means here

"End-to-end" is not a slogan — it is this exact list of things you will have built with your own hands and your own compute, in order:

1. A byte-level BPE tokenizer, trained from scratch on your own data sample, with nine special tokens reserved up front for the chat and tool stages.
2. A streaming data pipeline: download, quality-filter, deduplicate, tokenize, pack into fixed-length training shards with document-aware masking.
3. A deep-and-thin transformer with GQA, RoPE+NoPE, QK-norm, and SwiGLU, matching a fixed ~101M-parameter budget you can verify by hand.
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

The Chinchilla scaling law (Hoffmann et al., 2022; developed fully in [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html)) tells you the *compute-optimal* token budget for a given model size: roughly 20 tokens per parameter. For Stack-100M that is about 2B tokens counted against all 101M parameters, or ~1.7B counted against the 84.5M non-embedding parameters, the convention Chapter 14.12 uses. Either way, you would minimize training loss *per unit of training compute spent* by stopping around 2B.

Stack-100M trains on **~20B tokens** — roughly **200 tokens per parameter**, about **10× past Chinchilla-optimal**. This is deliberate, and it is the single most important economic idea in this capstone: **compute-optimal is not the same as deployment-optimal.**

Chinchilla's calculation only accounts for the cost of *training*. It says nothing about what happens after you ship the model — every one of the millions or billions of inference calls a deployed model serves. Training compute is a one-time cost; inference compute recurs forever. If over-training a smaller model past its Chinchilla-optimal point buys you a meaningfully lower loss *at a fixed, cheap-to-serve parameter count*, that trade is almost always worth it for anything you intend to actually deploy, because you pay the extra training FLOPs once and save the larger model's extra inference FLOPs on every single call thereafter. This is exactly the logic Meta used to justify Llama's ratios, that Sardana et al. (*Beyond Chinchilla-Optimal*, 2024) formalized as inference-aware scaling, and that essentially every subsequent open small-model release (SmolLM2/3, Qwen3, MobileLLM) has followed. The data-scaling term $B/D^\beta$ in the Chinchilla loss curve keeps paying off well past the "optimal" point — it just pays off *less per FLOP* than growing $N$ would, which is precisely the trade you are willing to make when parameter count (and therefore serving cost) is the constraint you actually care about, not training FLOPs. Chapter 14.12 puts a break-even request count on it.

{{fig:compute-vs-deployment-optimal}}

{{tool:scaling-law-optimal}}

### Force 2 — data quality over raw quantity

GPT-2's WebText was scraped, filtered for outbound Reddit links with a minimum karma, and deduplicated — a reasonable 2019 pipeline, but crude by 2026 standards. The modern pretraining data stack (built out fully in [Data Cleaning, Deduplication & Quality Filtering](../03-pretraining/02-data-cleaning-dedup.html) and [Synthetic Data for Pre- and Post-Training](../03-pretraining/15-synthetic-data.html)) does two things GPT-2's data did not:

- **Classifier-based educational filtering.** FineWeb-Edu (Penedo et al., HuggingFace, 2024) trains a classifier to score web documents by educational value and keeps only the high-scoring tail of Common Crawl. The result is a corpus with dramatically higher information density per token than raw web text — fewer boilerplate pages, navigation menus, and low-content spam diluting every gradient step.
- **Synthetic, dense, on-topic text.** Cosmopedia (HuggingFace) generates textbook- and story-style synthetic content with a larger teacher model, specifically to give a small model clean, well-structured exposition of concepts that natural web text states only in passing, if at all.

At fixed token count, cleaner and denser data means every one of your 20B training tokens is doing more work — you are not spending gradient steps learning to ignore boilerplate. Note that the *libraries* matter as much as the idea: FineWeb itself was produced with HuggingFace's `datatrove`, a distributed text-processing framework whose MinHash, URL-dedup, and quality-filter blocks are the reference implementations of everything Chapter 14.2 builds by hand — and which `capstone/scripts/dedup_datatrove.py` shows you how to hand your corpus off to once the hand-rolled version stops scaling.

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
| total params | 101,353,728 | verified by hand below |

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
- **MTP (Multi-Token Prediction)**, from DeepSeek-V3 (2024) and Gloeckle et al. (2024) — an auxiliary module trained to predict the token *after* the next one, giving a denser training signal per forward pass.

Both are covered as explicit "if you want DeepSeek's trick" upgrades in Chapter 14.4 once you have the base architecture working, alongside a Liquid LFM2-style gated-convolution mixer block as a third option.

### Verifying the parameter count by hand

The whole point of a capstone is that you should never take a number like "≈101M parameters" on faith. Here is the exact accounting, and the code that reproduces it.

$$
N_{\text{total}} = \underbrace{V d}_{\text{tied embed}} \;+\; L\Big(\underbrace{2d^2 + 2\, d\, d_{kv}}_{\text{attn (GQA)}} + \underbrace{3\, d\, d_{\text{ff}}}_{\text{SwiGLU}} + \underbrace{2d + 2 d_h}_{\text{norms + QK-norm}}\Big) \;+\; \underbrace{d}_{\text{final norm}}
$$

with $V=32768$, $d=512$, $L=30$, $d_{\text{ff}}=1408$, $d_{kv} = n_{\text{kv}}\cdot d_h = 2\times 64 = 128$, and $d_h = 64$. The attention term is written out explicitly because it is **not** simply $4d^2$: GQA makes the K and V projections only $d_{kv}=128$ wide, while Q and the output projection O stay full-width, so the four attention matrices sum to $d\cdot d + d\cdot d_{kv} + d\cdot d_{kv} + d\cdot d = 2d^2 + 2 d\, d_{kv}$, not $4d^2$. The $2 d_h$ term is the pair of tiny QK-norm scale vectors (over `head_dim`). There are no bias vectors anywhere, per the design decision above.

This is `capstone/stacklm/config.py` in full, exactly as shipped — every later chapter imports `StackConfig` from here and redefines nothing:

```python
"""Canonical Stack-100M configuration (see capstone/PLAN.md sec. 1).

Every number here is FROZEN by the spec. The whole capstone package derives its
shapes from `StackConfig`. `count_params` reproduces the ~101M param arithmetic
that Ch. 14.4 asks the reader to be able to do by hand.
"""
from dataclasses import dataclass


@dataclass
class StackConfig:
    # --- core shape (fixed for the whole capstone, PLAN.md sec. 1) ---
    vocab_size: int = 32768          # byte-level BPE we train ourselves (Ch. 14.3)
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
    logit_soft_cap: float = 0.0      # Gemma-2-style tanh soft-cap; 0.0 = off
    loss_chunk: int = 0              # >0 = chunked fused lm_head+CE (Ch. 14.4)
    attn_soft_cap: float = 0.0       # optional attention-logit soft-cap; 0.0 = off

    # --- optional efficiency variants, OFF by default (Ch. 14.4 "DeepSeek's trick") ---
    use_mla: bool = False            # Multi-head Latent Attention (DeepSeek-V2) instead of GQA
    mtp_heads: int = 0               # Multi-Token Prediction aux heads (DeepSeek-V3); 0 = off

    def head_groups(self) -> int:
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"
        return self.n_heads // self.n_kv_heads

    def uses_rope(self, layer_idx: int) -> bool:
        """RoPE on every layer except every `nope_every`-th (SmolLM3). 0 disables
        the interleave entirely, which is what makes the checkpoint exportable as a
        stock Qwen3 architecture (Ch. 14.4 "ecosystem" section)."""
        return self.nope_every <= 0 or ((layer_idx + 1) % self.nope_every) != 0


def count_params(cfg: StackConfig) -> dict:
    """Analytic parameter accounting -- matches `Stack100M.num_params()` exactly.

    Reproduces the Ch. 14.4 arithmetic: tied embedding counted once, per-block
    attention (Q/K/V/O with GQA-shrunk K,V), SwiGLU MLP, and the norm gains.
    """
    # This accounting is exact ONLY for the default (GQA, no-MTP, bias-free) path.
    # MLA replaces Q/K/V with down/up latent projections and MTP adds a whole extra
    # block + head -- both change the count, so refuse to report a wrong number.
    if cfg.use_mla:
        raise NotImplementedError(
            "count_params() covers the GQA path only; MLA re-shapes the attention "
            "projections. Use Stack100M(cfg).num_params() (Ch. 14.4) for MLA."
        )
    if cfg.mtp_heads:
        raise NotImplementedError(
            "count_params() covers mtp_heads=0 only; each MTP head adds an extra "
            "transformer block. Use Stack100M(cfg).num_params() (Ch. 14.4)."
        )

    embed = cfg.vocab_size * cfg.d_model

    q_width = cfg.n_heads * cfg.head_dim      # = d_model by construction (8*64=512)
    kv_width = cfg.n_kv_heads * cfg.head_dim  # = 128 (GQA shrinks this vs. q_width)
    q_proj = cfg.d_model * q_width
    k_proj = cfg.d_model * kv_width
    v_proj = cfg.d_model * kv_width
    o_proj = q_width * cfg.d_model
    attn_per_block = q_proj + k_proj + v_proj + o_proj

    mlp_per_block = 3 * cfg.d_model * cfg.intermediate

    rmsnorm_per_block = 2 * cfg.d_model                    # attn_norm + mlp_norm gains
    qk_norm_per_block = (2 * cfg.head_dim) if cfg.qk_norm else 0
    final_norm = cfg.d_model

    per_block = attn_per_block + mlp_per_block + rmsnorm_per_block + qk_norm_per_block
    all_blocks = per_block * cfg.n_layers

    lm_head = 0 if cfg.tie_embeddings else cfg.vocab_size * cfg.d_model
    total = embed + all_blocks + final_norm + lm_head

    return {
        "embedding (tied)": embed,
        "attn_per_block": attn_per_block,
        "mlp_per_block": mlp_per_block,
        "norms_per_block": rmsnorm_per_block + qk_norm_per_block,
        "per_block_total": per_block,
        "all_blocks (x n_layers)": all_blocks,
        "final_norm": final_norm,
        "lm_head (untied)": lm_head,
        "total": total,
    }


def toy_config() -> StackConfig:
    """Tiny CONFIG for the CPU smoke test -- exercises every code path (GQA 4:2,
    QK-norm, NoPE-every-4) at a scale that trains in seconds."""
    cfg = StackConfig(
        vocab_size=256,       # raw-byte-ish; the toy tokenizer trains a small vocab
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,          # still exercise the GQA code path (2:4 ratio, not 1:1)
        head_dim=16,           # 4 * 16 == 64
        intermediate=64,
        max_seq_len=64,
        qk_norm=True,
        # nope_every=2, NOT 4: with only 2 layers, `(layer_idx+1) % 4 != 0` is true
        # for BOTH layers, so a nope_every=4 toy would never execute the NoPE branch.
        # At 2 it does (layer 1 skips RoPE), so CI really covers both code paths.
        nope_every=2,
    )
    assert cfg.n_heads * cfg.head_dim == cfg.d_model
    return cfg


if __name__ == "__main__":
    cfg = StackConfig()
    for name, n in count_params(cfg).items():
        print(f"{name:26s} {n:>12,}  ({n / 1e6:7.3f}M)")
```

Running this (`python3 -m stacklm.config`) prints an embedding of 16,777,216 (16.78M), an attention block of 655,360 (0.655M), an MLP block of 2,162,688 (2.163M), a per-block total (with norms) of 2,819,200, all 30 blocks summing to 84,576,000, and a grand total of **101,353,728 ≈ 101.35M parameters**. Chapter 14.4 asserts this exact integer against `Stack100M(cfg).num_params()`, so if you ever change the architecture and forget to update the accounting, the test fails.

{{tool:param-flop-counter}}

!!! example "Worked example: what a 50k vocabulary would cost you"
    Suppose instead we had picked the more conventional 50,000-token vocabulary (roughly what GPT-2/GPT-3-family tokenizers use) instead of 32,768. The tied embedding table would be $50{,}000 \times 512 = 25{,}600{,}000$ parameters — **25.6M**, versus 16.78M at our chosen vocab size: an extra **8.8M parameters** spent on the lookup table before a single transformer block has processed anything.

    Keeping the 30 blocks and the final norm fixed at 84,576,512 parameters, the new total is $84{,}576{,}512 + 25{,}600{,}000 = 110{,}176{,}512 \approx 110.2\text{M}$, and the embedding is now $25{,}600{,}000 / 110{,}176{,}512 \approx 23.2\%$ of the model, versus $16{,}777{,}216 / 101{,}353{,}728 \approx 16.6\%$ for our 32,768 table. (You will also see this quoted as "≈26%," which is 25.6M measured against a *nominal* 100M budget rather than against the model you would actually get; always say which denominator you mean.) Nearly a quarter of your parameter count spent on lookup rather than computation is a lot. This is exactly why vocabulary size is a first-class architectural decision at small scale, not an afterthought copied from a larger model's tokenizer; Chapter 14.3 works through the full tradeoff curve, and Exercise 4 makes you pay for a bigger vocabulary in layers.

{{fig:stack100m-param-budget}}

### What the model costs in bytes

Parameters are the number people quote; **bytes are the number that decides which GPU you can rent**. Three separate pools compete for HBM during training, and they are not remotely the same size. All figures below are for the flagship micro-batch of 32 sequences × 2048 tokens; Chapter 14.7 itemizes them tensor by tensor and Chapter 14.12 re-derives them with Korthikanti et al.'s per-layer coefficient.

| Pool | Size | Why |
|---|---|---|
| Weights, bf16 | ≈203 MB | $101{,}353{,}728 \times 2$ B |
| Weights, fp32 master copy | ≈406 MB | mixed precision keeps an fp32 master (see [Mixed Precision, bf16 & FP8 Training](../03-pretraining/08-mixed-precision-fp8.html)) |
| **Weights + optimizer state** | **≈1.28 GB** | Muon on the 84.54M 2D hidden matrices at **12 B/param** (one momentum buffer), AdamW on the 16.81M embedding + 1D params at 16 B/param ($m$ *and* $v$) |
| Block activations saved for backward | ≈15–34 GB | linear in micro-batch; this is what actually fills the card |
| …with full activation checkpointing | ≈2.0 GB | only block-boundary tensors survive; ~33% more hardware FLOPs |
| …with **no** IO-aware attention kernel | ≈195 GB | the materialized $s\times s$ score matrices; fits on nothing |
| Logit/loss head, unchunked | ≈30 GB | $B{\cdot}s \times V$ in bf16 *and* fp32 — Ch. 14.4's chunked loss cuts it to ~1 GB |
| KV cache at inference, 2048 ctx, bf16 | ≈31.5 MB / sequence | $2 \cdot L \cdot n_{kv} \cdot d_h \cdot s \cdot 2$ B; full MHA would be 126 MB |

Four things fall out of that table, and they are the reason it belongs in the *first* chapter rather than the seventh.

**Weights are not the problem; activations are.** Optimizer state is under 4% of the memory in play. The naive folklore figure of 16 bytes/param would have said 1.62 GB — 27% too high, because Muon carries a single momentum buffer rather than Adam's two moments. That correction matters directly for checkpoint retention (Ch. 14.12) and for every "will it fit?" estimate you make later.

**The 15–34 GB spread is honest, not sloppy.** Chapter 14.7 counts the tensors an eager PyTorch graph actually retains and lands at 15–25 GB; Chapter 14.12 applies Korthikanti's $s\,b\,h\,(34 + 5as/h)$ per-layer coefficient and lands at 34 GB. Fusion under `torch.compile` recomputes some intermediates and shrinks the true figure; the loss head's shape dominates whichever way you count. **Measure it** with `torch.cuda.max_memory_allocated()` on 20 steps before you trust either number — that is the whole discipline.

**This is what derives the consumer tier.** A 24 GB RTX 4090 holds the 1.28 GB of state easily and cannot hold ~34 GB of activations at all. Cut the micro-batch 4× to 8 sequences and activations fall to ~8.5 GB; to keep the same ≈524,288-token effective batch you need $524{,}288 / (8 \times 2048) = 32$ gradient-accumulation steps instead of the flagship's 8 — exactly the "4× more accumulation" the tier table asserts. Combine that with the 4090's ~165 TFLOP/s bf16 dense peak (≈1.9× below the A100's 312) and the lower MFU of a smaller micro-batch, and the 2–4× wall-clock factor is derived rather than guessed.

**Without FlashAttention, this project does not exist.** The 195 GB row is not a curiosity: a naive attention implementation materializes $B \cdot n_{\text{heads}} \cdot s^2$ scores per layer, and at 30 layers that exceeds any GPU ever built. Every memory number in this book's capstone assumes `F.scaled_dot_product_attention` dispatching to a FlashAttention-2 backend. The full levers — checkpointing, offloading, 8-bit optimizer states — are in [Memory-Efficient Training: Checkpointing, Offloading & LoRA Math](../04-kernels-efficiency/10-memory-efficient-training.html).

---

## Budget: FLOPs, MFU, GPU-Hours and Dollars

### The compute tiers

Stack-100M is designed to be trained at three tiers of hardware, with the same code and the same architecture — only the micro-batch, accumulation count, and precision change.

| Tier | Hardware | Pretrain wall-clock | Cost | Role |
|---|---|---|---|---|
| **Flagship** | 1× A100 (80GB), rented | ≈22–29 GPU-hours (≈25 planning) | ≈USD 25–50 | The documented, full-fidelity walkthrough |
| **Consumer** | RTX 4090 / 3090 (24GB) | ≈2–4× flagship wall-clock | ≈USD 0 if owned | Same recipe, micro-batch 8 × accum 32 |
| **On-ramp** | Free Colab (T4, 16GB) | scaled-down run | USD 0 | Smaller model, fewer tokens — explicitly a subset, not the full run |

The flagship number is what we derive and defend below. The 4090 tier runs the identical recipe and loses no bf16 Tensor-Core throughput advantage — the 4090 has full bf16 support — so the main cost is wall-clock, not correctness. The Colab T4 tier is different in kind, not just degree: the T4 is a 2018 Turing-generation part with no native bf16 tensor-core path, so the on-ramp config trains in fp16 mixed precision instead of bf16 (with the corresponding loss-scaling care this requires — see [Mixed Precision, bf16 & FP8 Training](../03-pretraining/08-mixed-precision-fp8.html)) and deliberately shrinks the model and token budget so a meaningful run finishes inside Colab's free-tier session limits.

### From FLOPs to GPU-hours: the derivation, including the term people drop

The flagship budget is not a marketing number, but it is also not *quite* the one-line $6ND$ calculation you will see quoted everywhere. Stack-100M is deep, thin, and trained at a fairly long context — exactly the regime where the attention-score term stops being a rounding error.

One bookkeeping point before the arithmetic. The ~20B-token budget is **not** 20B of pretraining plus something else: Chapter 14.7 spends ~18B tokens in the WSD stable phase at `seq_len=2048`, and Chapter 14.8 slices the remaining ~2B off the same budget for the decay-phase anneal and the long-context extension. "Pretraining" below means the 18B-token stable phase.

!!! example "Worked example: FLOPs to GPU-hours to dollars"
    **Setup.** $N = 101{,}353{,}728$ parameters (from `count_params`), $D = 18 \times 10^9$ stable-phase tokens, $L = 30$ layers, $s = 2048$ context, $d_q = n_{\text{heads}} \cdot d_h = 512$.

    **Term 1 — the dense matmuls.** Every parameter participates in one multiply-accumulate per token in the forward pass (2 FLOPs) and roughly twice that in the backward pass (4 FLOPs), giving the familiar rule:

    $$
    C_{\text{dense}} / \text{token} = 6N = 6 \times 101{,}353{,}728 \approx 6.081 \times 10^{8}\ \text{FLOPs}
    $$

    (A subtlety worth noticing: the tied embedding table is counted once in $N$, and that is exactly right here — the *lookup* costs no FLOPs, but the same matrix is used as the output projection, which is a genuine $V \times d$ GEMM per token. One matmul use, one count.)

    **Term 2 — the attention scores, which $6ND$ ignores.** The $QK^\top$ and $AV$ matmuls do not involve parameters at all, so no per-parameter rule can see them. Per layer over a length-$s$ sequence, the forward pass costs $2s^2 d_q$ for $QK^\top$ plus $2s^2 d_q$ for $AV$; **causal masking halves both** (a FlashAttention-style kernel simply never computes the masked blocks), and the backward pass is about twice the forward. So per token, per layer, $6 s d_q$, and over the whole model:

    $$
    C_{\text{attn}} / \text{token} = 6 L s\, d_q = 6 \times 30 \times 2048 \times 512 = 1.887 \times 10^{8}\ \text{FLOPs}
    $$

    That is **+31% on top of $6N$**. The ratio is $\frac{6Lsd_q}{6N} \approx \frac{L s d_q}{N}$, which grows linearly in the context length and, at fixed $N$, grows with depth-over-width. Stack-100M is the highest-attention-fraction model in this book precisely *because* it is deep, thin, and long-context. (At Chapter 14.8's 8192-token mid-training context the same term becomes $7.55\times10^8$ per token — attention then costs more than every weight matrix combined; Exercise 7 makes you redo the budget there.)

    **Total training compute for the stable phase:**

    $$
    C = (6N + 6Lsd_q)\, D \approx (7.969 \times 10^{8}) \times (1.8\times 10^{10}) \approx 1.434 \times 10^{19}\ \text{FLOPs}
    $$

    versus $1.095 \times 10^{19}$ from the naive $6ND$ — a 31% under-estimate you would have paid for in wall-clock.

    **Convert to wall-clock.** The A100's bf16 Tensor Core dense peak is $\pi = 312$ TFLOP/s $= 3.12\times10^{14}$ FLOP/s. Define MFU (Model FLOPs Utilization) as achieved model FLOP/s divided by $\pi$, where "model FLOPs" is exactly the attention-inclusive $C$ above. Chapter 14.7's instrumented run sustains ≈228k tokens/s on a `torch.compile`d loop with an SDPA/FlashAttention backend, which is **MFU ≈ 0.58** in this convention (0.45 under 6ND-only). A less-tuned loop — no compile, a smaller micro-batch, an unfused attention path — sits near **0.45** (0.34 under 6ND). So:

    $$
    t_{\text{tuned}} = \frac{1.434\times10^{19}}{0.59 \times 3.12\times10^{14}} \approx 7.8\times10^{4}\ \text{s} \approx 21.6\ \text{hours}
    $$

    $$
    t_{\text{untuned}} = \frac{1.434\times10^{19}}{0.45 \times 3.12\times10^{14}} \approx 1.02\times10^{5}\ \text{s} \approx 28.4\ \text{hours}
    $$

    **≈22–29 GPU-hours is the honest band; ≈25 is the planning number.**

    **Convert to dollars.** A100-80GB on-demand rents for roughly USD 1.50–2.00/GPU-hour in 2026; interruptible/spot capacity on GPU marketplaces is often nearer USD 0.80–1.20. At USD 1.20–1.80 the pretraining pass is **≈USD 26–51** — call it USD 25–50, arrived at honestly.

    This is exactly the kind of napkin calculation you should be able to reproduce for any training run before you spend the money. It has five inputs — parameter count, token budget, context length × depth, peak hardware FLOP/s, and MFU — and no step you cannot check.

{{tool:train-compute-estimator}}

!!! warning "Common pitfall: an MFU number without its FLOP convention is not a number"
    The *same* Chapter 14.7 run reports **44.5% MFU** and **58.2% MFU**. Neither is wrong: the first counts $6ND$ only, the second adds the $6Lsd_q$ attention term this chapter derives — a 31% gap at Stack-100M's shape. Compare your 6ND-only figure against someone else's attention-inclusive figure and your loop looks 30% worse than it is.

    Two further conventions bite. **PaLM's original MFU definition does not credit causal masking**: its model-FLOPs formula uses $12\,L\,H\,Q\,T$ for attention, exactly twice our $6Lsd_q$ (which would be +62% here). We halve it because a causal kernel genuinely computes only the lower triangle — but that means our MFU is not directly comparable to a PaLM-convention number, so *say which you mean*. And **MFU is not HFU**: with activation checkpointing on, the hardware executes ~4/3 of the model FLOPs, so $\text{HFU} \approx 1.33\times\text{MFU}$ (Korthikanti et al., 2022). Chapter 14.7 logs all of them.

### MFU reality check: deep-and-thin costs you throughput

The chapter has now told you twice that depth-over-width is the right call at fixed parameters. Here is the bill for it, because the capstone does not hide trade-offs.

At `d_model=512`, the largest GEMM in a block is the SwiGLU up/gate projection: $(B\cdot s, 512) \times (512, 1408)$. The $K$ dimension of 512 is small by tensor-core standards, so each matmul spends a larger fraction of its time on prologue/epilogue and memory traffic than the same FLOPs would in a $d_{\text{model}}=4096$ model. Then multiply the *count* of kernels by 30 layers: 30 RMSNorm pairs, 30 RoPE applications, 30 SiLU-and-multiply gates, 30 residual adds, all of them memory-bandwidth-bound and all of them paying kernel-launch latency. A 10-layer, 1024-wide model of the same parameter count would do the same arithmetic in a third as many launches. **Deep-and-thin buys loss-per-parameter and pays for it in launch count** — which is fine, because you train once and serve forever, but it is why a tuned run lands near 0.58 rather than the 0.7+ a wide model can reach, and why an untuned one collapses to 0.45 or below.

The levers that actually move MFU here, in rough order of payoff (all developed in [Kernel Fusion, torch.compile, CUDA Graphs & Compilers](../04-kernels-efficiency/09-compilers-fusion.html) and [The Roofline Model & Performance Engineering](../04-kernels-efficiency/01-roofline-performance.html)):

1. **`torch.compile(model)`** — TorchInductor fuses the norm/RoPE/SiLU/residual elementwise chains into a handful of kernels per block. At 30 narrow layers this is worth more than at any other point in the book; `mode="max-autotune"` additionally tunes the small GEMM tiles.
2. **`torch.nn.functional.scaled_dot_product_attention`** — PyTorch 2.x dispatches this to a FlashAttention-2 kernel when the shapes and dtype allow, giving you IO-aware, causal-skipping attention without writing CUDA. Use `is_causal=True` rather than materializing a mask, or you lose exactly the causal halving we counted above. (The `flash-attn` package, or your own Triton kernel from Part IV, are the drop-in alternatives.)
3. **A bigger micro-batch.** Throughput at small $d_{\text{model}}$ is mostly about making $M = B \cdot s$ large enough that the GEMM clears the roofline ridge point rather than being launch-latency-bound. Fill the 80GB — which the chunked loss head of Ch. 14.4 makes possible — then use gradient accumulation to reach the target ≈0.5M-token effective batch.
4. **bf16 autocast plus a fused optimizer** (`torch.optim.AdamW(..., fused=True)` for the AdamW half of the hybrid), so the optimizer step does not become a visible fraction of a fast step.
5. **CUDA graphs** (via `torch.compile(mode="reduce-overhead")`) if, after all of the above, you can still see launch gaps in a profile.

Dimensions are already chosen to cooperate: 512, 1408, and 64 are all multiples of 64, so no tensor-core tile is padded.

!!! tip "Practitioner tip: measure MFU on day one, not on day two"
    Chapter 14.7's loop prints tokens/second and MFU every logging interval. Run 200 steps at the real config before you launch the real job, compute your own $t = C / (\text{MFU} \times \pi)$, and decide *then* whether to spend another hour on `torch.compile` or just start. An hour of tuning that lifts attention-inclusive MFU from 0.35 to 0.58 saves roughly 15 GPU-hours on this run — a ~15× return on the hour.

### The full-project envelope in four lines

The pretraining pass is the headline, but it is not the bill. There is exactly **one** itemized cost table in this book — [Chapter 14.12's](../14-capstone/12-retrospective-and-scaleup.html), which derives every line and re-prices it onto 2026 hardware. Here is its shape so you know what you are signing up for:

- **Pretraining (Ch. 14.7):** 18B tokens, ≈22–29 GPU-hr, ≈USD 25–50. Roughly half the project.
- **Everything else on a GPU:** scaling ladder (Ch. 14.5), mid-training's ~2B tokens split 1.2B @ 2048 and ~0.8B @ 8192 (Ch. 14.8, ~3.5 GPU-hr), SFT + DPO + GRPO (Ch. 14.9), agent distillation (Ch. 14.10), eval + quantization (Ch. 14.11) — together **≈13 GPU-hr, ≈USD 25**.
- **Non-GPU:** the teacher-model API for ReAct trajectories (~USD 8) plus object storage and egress for ~200 GB of shards and checkpoints (~USD 5) — **≈USD 13**, and at 1B scale this category *overtakes* the GPU bill.
- **Re-run reality tax:** ~25% of the GPU spend for OOMs, bad launches, and restarts. Grand total **≈USD 90–100** — which is why the sticker says "the ~USD 100 model."

Two line items deserve their own warnings, because they are the two places readers actually get stuck, and neither is a GPU problem.

!!! warning "Common pitfall: your Python BPE encoder is slower than your A100"
    Chapter 14.3 has you implement byte-level BPE from scratch, and you *should* — training the merges is where the mechanism lives, and a pure-Python trainer on a 1–2 GB sample is perfectly tolerable. But **encoding 20B tokens with a pure-Python inner loop is not**: at a realistic few hundred KB/s per process it is a multi-day job that can genuinely cost more wall-clock than the training run it feeds.

    The fix is standard practice, not a cop-out: train the merges yourself, then hand the merge table to a fast encoder for the bulk pass. Either load your merges into HuggingFace `tokenizers` (a Rust `ByteLevel` BPE model with a `merges` file — its `encode_batch` is multi-threaded and roughly two orders of magnitude faster than pure Python), or keep your own encoder and shard the corpus across a `multiprocessing.Pool` of all your cores. Chapter 14.3 shows both. The mechanism is yours; the throughput is the library's.

!!! warning "Common pitfall: disk, not dollars, is what kills the data stage"
    20B `uint16` tokens is 40 GB of packed shards — fine. The raw text you stream to *produce* those tokens is several times larger, and a rented box often ships with 100 GB of local disk. Stream and tokenize in one pass (HF `datasets` in `streaming=True` mode never materializes the full corpus), write shards incrementally, and delete raw text as you go. Chapter 14.2's pipeline is written this way for exactly this reason.

!!! interview "Interview Corner"
    **Q:** Two engineers review the same training log. One reports 44.5% MFU; the other reports 58.2%. Neither made an arithmetic mistake. What happened, which number goes in the report, and what does either number tell you about what to fix?

    **A:** They used different FLOP conventions in the numerator. The 44.5% figure counts only the dense per-parameter work, $6ND$; the 58.2% figure adds the attention score/value matmuls, $6\,L\,s\,d_q$ per token, which for a deep, thin, 2048-context model like this one is a **+31%** term that $6ND$ structurally cannot see (those matmuls involve no parameters). Neither is "the" MFU — MFU is a ratio whose numerator is a *convention*, so the report must state which, and ideally give both. A third convention exists: PaLM's original definition does not credit causal masking and would double the attention term again.

    What neither number changes is the bill: wall-clock comes from measured tokens/second, not from a utilization ratio. What MFU tells you is *headroom*. A low attention-inclusive MFU with high `nvidia-smi` utilization means the GPU is busy but memory-bound — fuse with `torch.compile`, raise the micro-batch to clear the roofline ridge point, confirm SDPA is hitting its FlashAttention backend rather than the math fallback. And distinguish MFU from HFU: with activation checkpointing on, hardware FLOPs are ~4/3 of model FLOPs, so a "low" MFU may simply be recompute you deliberately bought memory with.

---

## The Repository, and What We Hand-Roll vs. What You Would Use

Every later chapter adds files to the same tree. Here is the layout of `capstone/` — no `configs/` directory, no YAML indirection: the configuration *is* a Python dataclass, because a config you can `assert` on is worth more than a config you can lint.

```text
capstone/
├── PLAN.md                       # canonical Stack-100M spec (source of truth)
├── README.md                     # runbook: full-run commands per stage
├── requirements.txt              # hermetic smoke set + (commented) real-run extras
├── smoke_test.py                 # end-to-end toy-scale pipeline, CPU-only, hermetic
├── scripts/
│   ├── dedup_datatrove.py        # hand the corpus off to datatrove's MinHash blocks -- Ch. 14.2
│   ├── repack_long.py            # repack shards to 8192 for long-context extension -- Ch. 14.8
│   └── midtrain.py               # mid-training driver (anneal -> long ctx -> capability)
└── stacklm/
    ├── config.py                 # StackConfig, count_params, toy_config  -- this chapter
    ├── tokenizer/
    │   └── bpe.py                # byte-level BPE trainer + encode/decode  -- Ch. 14.3
    ├── data/
    │   ├── synthetic.py          # data mix spec, streaming sources, in-process toy corpus
    │   ├── filters.py            # quality/heuristic filters (length, symbol ratio, ...)
    │   ├── dedup.py              # exact-hash + MinHash-LSH near-duplicate removal
    │   ├── pack.py               # pack to seq_len, document-aware causal masking
    │   ├── shard.py              # uint16 .bin shard writer
    │   ├── dataset.py            # memmap-backed packed dataset
    │   └── build_corpus.py       # driver: budgets, interleave, holdout, manifest -- Ch. 14.2
    ├── model/
    │   ├── rmsnorm.py            # RMSNorm
    │   ├── rope.py               # rotary embeddings + the NoPE layer rule
    │   ├── swiglu.py             # SwiGLU MLP
    │   ├── attention.py          # GQA + QK-norm + RoPE/NoPE dispatch
    │   ├── loss.py               # z-loss + chunked fused lm_head/cross-entropy
    │   ├── kv_cache.py           # GQA KV cache for incremental decoding
    │   ├── sampling.py           # temperature / top-k / top-p decoding
    │   ├── mla.py                # optional Multi-head Latent Attention (DeepSeek-V2)
    │   ├── mtp.py                # optional Multi-Token Prediction module
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
    │   ├── mixture.py            # premium anneal mix + long-doc upsampling
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

This is the pedagogical rule for the whole capstone, stated once: **we hand-roll to learn the mechanism, and we name the library you would reach for at work.** A from-scratch implementation you understand plus knowledge of which battle-tested package supersedes it is strictly more valuable than either alone — and at 100M scale the from-scratch version is genuinely fast enough to finish, which is why this project is possible at all. `scripts/dedup_datatrove.py` is the thesis in miniature: hand-roll MinHash in `data/dedup.py` to learn what a band collision is, then hand the corpus to the library that actually built FineWeb.

| Stage | What we build from scratch | What you would use in production |
|---|---|---|
| BPE training + encoding | `stacklm/tokenizer/bpe.py`, pure stdlib | HF **`tokenizers`** (Rust BPE trainer, multi-threaded `encode_batch`), **`tiktoken`** for inference-side encoding, **SentencePiece** for unigram |
| Corpus acquisition | `stream_source()` over HF **`datasets`** in streaming mode | Same, plus **`datatrove`** (the pipeline that built FineWeb) or NVIDIA **NeMo-Curator** for distributed filtering |
| Dedup | exact-hash + MinHash-LSH in `stacklm/data/dedup.py` | `datatrove`'s MinHash blocks (see `scripts/dedup_datatrove.py`), or **`text-dedup`** |
| Attention kernel | explicit `softmax(QK^T/√d)V` math, then `F.scaled_dot_product_attention` | **`F.scaled_dot_product_attention`** (FlashAttention-2 backend), **`flash-attn`**, **xFormers**; **Triton** if you need a custom variant |
| Fusion / compilation | eager PyTorch first, then one `torch.compile` call | **`torch.compile`** (TorchInductor), CUDA graphs, **TransformerEngine** for FP8 on Hopper+ |
| Optimizer | `stacklm/optim/muon.py` (Newton–Schulz) | **`KellerJordan/Muon`** reference implementation; `torch.optim.AdamW(fused=True)` for the 1D groups |
| Scaling-law fit | least-squares fit of $L(N,D)$ in NumPy | same idea; the analysis is bespoke everywhere |
| Distributed training | *not needed at 100M* — single process, by design | PyTorch **FSDP2** (`torch.distributed.fsdp.fully_shard`) / DDP, **DeepSpeed** ZeRO, **Megatron-LM** for 3D parallelism |
| Checkpoint format | `torch.save` of model+optimizer+step+RNG | **`safetensors`**, `torch.distributed.checkpoint` for sharded state |
| Experiment tracking | stdout + a JSONL log | **Weights & Biases**, **TensorBoard**, MLflow |
| SFT / DPO | `stacklm/post/sft.py`, `dpo.py` | **TRL** (`SFTTrainer`, `DPOTrainer`), **Axolotl**, **LLaMA-Factory**, **Unsloth** — see [TRL: HuggingFace's RL Library](../06-rl-infra/03-trl.html) |
| RLVR / GRPO | `stacklm/post/grpo.py` | TRL `GRPOTrainer`, **veRL**, **OpenRLHF**, **Prime-RL** — see [veRL: HybridFlow & The Single-Controller Architecture](../06-rl-infra/04-verl.html) |
| Rollout generation for RL | our own batched sampler (`model/sampling.py`) | **vLLM** or **SGLang** as the rollout engine (this is what veRL/TRL drive under the hood) — see [vLLM: Architecture, PagedAttention & Internals](../07-inference-serving/03-vllm-internals.html) |
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

That only works if the toy config genuinely exercises the same code paths, which is why two of `toy_config()`'s numbers (printed above) are load-bearing rather than arbitrary:

- **`n_kv_heads=2` with `n_heads=4`** keeps the GQA reshape non-trivial. A 1:1 ratio would silently pass even if the head-grouping broadcast were wrong.
- **`nope_every=2`, not 4.** The layer rule is `uses_rope(i) == ((i + 1) % nope_every) != 0`. With only 2 layers, a `nope_every=4` toy evaluates `1 % 4` and `2 % 4` — both non-zero — so *both* layers take the RoPE branch and the NoPE path is never executed by CI. At 2, layer index 1 hits `2 % 2 == 0` and takes it. A toy config's job is not to be small; it is to be small **and branch-complete**.

The `vocab_size=256` placeholder deserves a note, because it is the one place a toy config can quietly break a downstream chapter. Chapter 14.3 reserves **nine** special tokens up front — `<|bos|>`, `<|eos|>`, `<|pad|>`, the four chat-role markers, and the two tool tokens — and their integer ids are load-bearing for the chat template (Ch. 14.9) and the ReAct format (Ch. 14.10). A 256-entry "raw bytes" vocabulary has no room for them. So the smoke test trains a real toy tokenizer and rebuilds the config from the tokenizer that actually exists:

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
    - **Bytes, not parameters, decide your hardware.** Weights + Muon/AdamW optimizer state is only ≈1.28 GB (12 B/param for Muon's single momentum buffer, not the folklore 16); block activations at micro-batch 32×2048 are ≈15–34 GB, and ~195 GB without an IO-aware attention kernel. That single fact derives the 24 GB tier's 4× gradient accumulation and its 2–4× wall-clock.
    - The training-compute rule is $C = (6N + 6Lsd_q)D$, not $6ND$. At $s=2048$ the attention term adds **+31%** for this shape, and it grows linearly with context — the whole reason Chapter 14.8's 8192-token pass is expensive. Any budget quoting $6ND$ alone is an under-estimate, worst exactly for deep, thin, long-context models.
    - The flagship pretraining pass is **≈22–29 GPU-hours** (≈25 planning) at MFU 0.45–0.59 attention-inclusive, equivalently 0.34–0.45 under $6ND$ — **≈USD 25–50**. The whole project lands near **USD 90–100**; Chapter 14.12 owns the itemized bill. Always quote MFU *with* its FLOP convention: 44.5% and 58.2% can describe the same run.
    - "Narrow but real" is the capstone's honest target: not a general chatbot, but a model whose narrow-domain, tool-scaffolded outputs (arithmetic, retrieval-grounded QA) are genuinely verifiable, not merely plausible.
    - We hand-roll every stage to learn the mechanism and name the production library that supersedes it — `tokenizers`, `datatrove`, `torch.compile`, FlashAttention/SDPA, TRL, veRL, vLLM, `lm-evaluation-harness`, `llama.cpp`. Knowing both, and knowing which parts of the distributed stack you can skip at 100M, is the point.

!!! sota "State of the Art & Resources (2026)"
    The "train a real GPT-class model on a budget" tradition Karpathy started now has a 2025–2026 generation of open small-model recipes to learn directly from — the same data mixes, architectures, and optimizers Stack-100M borrows are all publicly documented and reproducible.

    **Foundational work**

    - [Hoffmann et al., *Training Compute-Optimal Large Language Models* (2022)](https://arxiv.org/abs/2203.15556) — the Chinchilla scaling law this chapter's "over-train past compute-optimal" argument is built against.
    - [Liu et al., *MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases* (2024)](https://arxiv.org/abs/2402.14905) — the paper that made the "deep-and-thin" case quantitative; the direct justification for Stack-100M's 30×512 shape.
    - [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — the original small-budget GPT-2 reproduction this capstone updates; simplest reference for the base training loop shape.
    - [Chowdhery et al., *PaLM* (2022)](https://arxiv.org/abs/2204.02311) — the paper that defined **Model FLOPs Utilization** (MFU), including the attention term this chapter refuses to drop; note its $12\,L\,H\,Q\,T$ convention does not credit causal masking, so it is exactly twice ours.
    - [Korthikanti et al., *Reducing Activation Recomputation in Large Transformer Models* (2022)](https://arxiv.org/abs/2205.05198) — the per-layer activation-memory formula behind this chapter's byte table, and the source of the MFU/HFU distinction.

    **Recent advances (2023–2026)**

    - [Ben Allal et al., *SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model* (2025)](https://arxiv.org/abs/2502.02737) — HuggingFace's own writeup of exactly the aggressive-curation, heavily-over-trained recipe this chapter cites.
    - [*SmolLM3: smol, multilingual, long-context reasoner*](https://huggingface.co/blog/smollm3) — the official HuggingFace blog post that introduced interleaved NoPE (every 4th layer) for length generalization, which Stack-100M adopts directly.
    - [Qwen Team, *Qwen3 Technical Report* (2025)](https://arxiv.org/abs/2505.09388) — the frontier-lab report behind Qwen3-0.6B, the sub-billion dense model this chapter cites as proof a small model can be a real base, not a toy.
    - [Kimi Team, *Kimi K2: Open Agentic Intelligence* (2025)](https://arxiv.org/abs/2507.20534) — the technical report documenting MuonClip (QK-clip), the fix that lets Muon train stably at scale, which Stack-100M's optimizer chapter reuses.
    - [Hu et al., *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies* (2024)](https://arxiv.org/abs/2404.06395) — introduces the Warmup-Stable-Decay (WSD) schedule Stack-100M's pretraining and mid-training phases use.
    - [Sardana et al., *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws* (2024)](https://arxiv.org/abs/2401.00448) — the formal version of this chapter's Force 1: once you price inference, the optimal model is smaller and the optimal token budget much larger.
    - [*Introducing LFM2: The Fastest On-Device Foundation Models on the Market*](https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models) — Liquid AI's official announcement of the gated short-convolution-plus-attention hybrid block Stack-100M implements as an optional mixer variant.

    **Open-source & tools**

    - [karpathy/nanochat](https://github.com/karpathy/nanochat) — the closest existing end-to-end analogue to this capstone: tokenizer → pretrain → mid-train → SFT → RL → web serve, in one dependency-light repo, at a documented ~USD 100 tier on an 8×H100 node. Read it next to Part XIV.
    - [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) — the NanoGPT speedrun leaderboard; the empirical origin of the Muon + QK-norm + logit-soft-cap stack Stack-100M's config uses.
    - [KellerJordan/Muon](https://github.com/KellerJordan/Muon) — the reference implementation of the Muon optimizer (Newton–Schulz orthogonalized updates for 2D hidden-layer weights) that Stack-100M's hybrid optimizer is built on.
    - [karpathy/llm.c](https://github.com/karpathy/llm.c) — pure C/CUDA GPT-2/GPT-3 reproduction; the low-level companion to nanoGPT for understanding what a training step costs on real hardware.
    - [huggingface/datatrove](https://github.com/huggingface/datatrove) — the distributed text-processing library FineWeb was actually built with; the production version of Chapter 14.2's filtering and MinHash dedup, and the target of `capstone/scripts/dedup_datatrove.py`.
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
- Sardana et al., *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws*, 2024 — the inference-aware version of the over-training argument.
- Chowdhery et al., *PaLM: Scaling Language Modeling with Pathways*, 2022 — the MFU definition, and the causal-masking convention caveat.
- Korthikanti et al., *Reducing Activation Recomputation in Large Transformer Models*, 2022 — activation-memory accounting and MFU vs. HFU.
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
    The flaw is that Chinchilla-optimal minimizes loss *per unit of training compute* — it treats training as the only cost. A deployed model's dominant cost is *inference*, which recurs on every one of potentially billions of calls, whereas training is paid once. When parameter count (hence serving cost and latency) is the fixed constraint you care about, you want the lowest loss *at that fixed size*, and the data-scaling term $B/D^{\beta}$ keeps improving well past 20 tokens/param — just less per training FLOP than growing $N$ would. Paying ~10× the Chinchilla-optimal token count once, to lower loss at a cheap-to-serve 101M parameters forever, is the right trade for anything you will actually deploy; Sardana et al. (2024) formalize it as inference-aware scaling.

    Your colleague is right only in the case Chinchilla itself answers: you have a fixed *training*-FLOP budget and you are *not* going to serve the model enough times to amortize extra training compute (e.g. a one-off research run whose only goal is the lowest loss for that FLOP budget). Then growing $N$ toward the compute-optimal frontier beats over-training a smaller model.

**2.** Using only the spec table (`d_model=512`, `n_heads=8`, `n_kv_heads=2`, `head_dim=64`), compute the number of parameters in the four attention projection matrices of a single Stack-100M block. Then compute what that block's attention would cost under full multi-head attention (all heads keep their own K and V), and state how many parameters GQA saves across all 30 layers.

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
    \text{attn}_{\text{MHA}} = 4\times 512^2 = 1{,}048{,}576 = 4d^2 \;(\approx 1.049\text{M}).
    $$

    Per-block saving is $1{,}048{,}576 - 655{,}360 = 393{,}216$; across all 30 layers, $30 \times 393{,}216 = 11{,}796{,}480 \approx 11.8\text{M}$ — roughly 11.6% of the whole ~101M model. The KV-cache saving at inference is the 4× reduction that actually motivates GQA (31.5 MB vs 126 MB per 2048-token sequence, from the byte table). Note GQA does *not* reduce attention-score FLOPs: $QK^\top$ and $AV$ still run over all 8 query heads with the 2 KV heads broadcast. GQA is a memory optimization, not a FLOP one.

**3.** You have a 24 GB RTX 4090 and want to run the flagship recipe. Take the chapter's byte table as given: weights + optimizer state ≈1.28 GB; block activations ≈34 GB at a micro-batch of 32 sequences × 2048 tokens with FlashAttention-backed SDPA; the effective batch must stay at 524,288 tokens. (a) Compute activation memory per sequence and the largest micro-batch that leaves ≥4 GB of headroom for the loss head, workspace, and fragmentation. (b) Give the gradient-accumulation count needed to preserve the effective batch, and compare it to the flagship's. (c) Compute the KV cache for one 2048-token sequence in bf16 under this config, and say why it is irrelevant to (a).

??? note "Solution"
    (a) Activations are linear in micro-batch: $34.2\ \text{GB} / 32 \approx 1.07$ GB per sequence of 2048 tokens. Budget: $24 - 1.28 - 4 \approx 18.7$ GB for activations, so $\lfloor 18.7 / 1.07 \rfloor = 17$ sequences in principle. In practice you would set the micro-batch to a power of two with real headroom — **8** — because the 34 GB figure is itself an estimate (Ch. 14.7's tensor-by-tensor count gives 15–25 GB) and because allocator fragmentation on a card this small is unforgiving. Micro-batch 8 costs ≈8.5 GB, leaving ~14 GB of slack; if you want 16, turn on activation checkpointing (≈2.0 GB total) and accept the ~33% hardware-FLOP surcharge.

    (b) $524{,}288 / (8 \times 2048) = 32$ accumulation steps, versus the flagship's $524{,}288 / (32 \times 2048) = 8$ — exactly **4×** more, which is the tier table's claim, now derived. Wall-clock then follows from two independent factors: the 4090's ~165 TFLOP/s bf16 dense peak is ~1.9× below the A100's 312, and a 4× smaller micro-batch lowers arithmetic intensity and therefore MFU. Together, ≈2–4×.

    (c) $2 \times L \times n_{kv} \times d_h \times s \times 2\ \text{B} = 2 \times 30 \times 2 \times 64 \times 2048 \times 2 = 31{,}457{,}280\ \text{B} \approx 31.5$ MB per sequence (126 MB under full MHA — the 4× GQA wins back). It is irrelevant to (a) because the KV cache exists only during *incremental decoding*: training does a single parallel forward over the whole sequence and never caches K/V across steps. Confusing the two is one of the most common memory-budgeting errors — see [Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html).

**4.** Your organization standardizes on a shared 65,536-token tokenizer and you must adopt it, while keeping the total parameter budget at Stack-100M's 101,353,728 and holding `d_model=512`, the SwiGLU width, and tied embeddings fixed. (a) How many transformer layers can you afford? (b) What is the resulting total, and what fraction is now embedding? (c) In one sentence, say what you have actually traded, referencing the MobileLLM result.

??? note "Solution"
    (a) The tied embedding becomes $65{,}536 \times 512 = 33{,}554{,}432$. Subtracting it and the 512-parameter final norm from the budget leaves $101{,}353{,}728 - 33{,}554{,}432 - 512 = 67{,}798{,}784$ parameters for blocks. From the chapter's accounting each block costs $655{,}360 + 2{,}162{,}688 + 1{,}152 = 2{,}819{,}200$, so

    $$
    \left\lfloor \frac{67{,}798{,}784}{2{,}819{,}200} \right\rfloor = \lfloor 24.05 \rfloor = 24 \text{ layers}.
    $$

    (b) Total $= 33{,}554{,}432 + 24 \times 2{,}819{,}200 + 512 = 101{,}215{,}744 \approx 101.2$M, just inside budget. The embedding is now $33{,}554{,}432 / 101{,}215{,}744 \approx 33.2\%$ — double the 16.6% of the 32,768-vocab design.

    (c) You paid **six layers of depth (20% of the network's sequence-mixing capacity) for a lookup table you did not design**, which is precisely the trade MobileLLM (Liu et al., 2024) shows is a losing one at sub-billion scale: at fixed parameters, depth is what buys loss. If the shared tokenizer is non-negotiable, the honest options are to raise the budget, or to untie and factorize the embedding (a $V \times r$ and $r \times d$ pair) so the vocabulary stops charging full price per row.

**5.** Extend `count_params` to support `mtp_heads >= 1` instead of refusing. Read the module docstring of `stacklm/model/mtp.py`: a Multi-Token Prediction module (DeepSeek-V3, 2024) concatenates the trunk hidden state with the embedding of the already-known next token, projects the $2d$-wide pair back to $d$, and runs one transformer block; it **shares** the token embedding and the LM head with the trunk. Write the branch, give the exact per-head cost and the new total for `mtp_heads=1`, and say why sharing matters.

??? note "Solution"
    Each MTP module owns: one full transformer block, one $2d \times d$ projection $M_k$, and two RMSNorm gains (one on the hidden state, one on the embedding). It owns *no* embedding and *no* LM head — those are references to the trunk's tensors.

    ```python
    # inside count_params(), replacing the `if cfg.mtp_heads: raise` guard:
    #
    # An MTP module (DeepSeek-V3) = 1 block + M_k (2d x d) + 2 RMSNorm gains.
    # tok_emb and lm_head are SHARED with the trunk, so they are NOT re-counted.
    mtp_proj = 2 * cfg.d_model * cfg.d_model          # M_k: concat(h, emb) -> d_model
    mtp_norms = 2 * cfg.d_model                       # RMSNorm(h) + RMSNorm(emb)
    mtp_per_head = per_block + mtp_proj + mtp_norms
    mtp_total = mtp_per_head * cfg.mtp_heads

    total = embed + all_blocks + final_norm + lm_head + mtp_total
    ```

    Numerically, with $d = 512$: $M_k = 2 \times 512 \times 512 = 524{,}288$, norms $= 1{,}024$, block $= 2{,}819{,}200$, so

    $$
    \text{mtp\_per\_head} = 2{,}819{,}200 + 524{,}288 + 1{,}024 = 3{,}344{,}512,
    $$

    and for `mtp_heads=1` the total is $101{,}353{,}728 + 3{,}344{,}512 = 104{,}698{,}240 \approx 104.7$M — a **3.30%** training-time overhead, and *zero* at inference, because the MTP module is discarded (or repurposed as a self-speculative draft head).

    Sharing is what keeps that number small. If each module carried its own $V \times d$ embedding and head it would cost an extra 33.55M parameters — ten times the module itself, and a third of the whole model — which is why a "generic extra depth head" implementation is not MTP. Verify your branch against `Stack100M(cfg).num_params()` with `mtp_heads=1`, exactly as Ch. 14.4 does for the default path.

**6.** The default config sets `nope_every=4`, meaning every 4th layer omits rotary position embeddings entirely (NoPE), following SmolLM3. Conceptually, (a) explain *why* interleaving NoPE layers is expected to help the model generalize to sequences longer than the 2048-token pretraining context, and (b) explain why this is essentially free in parameters. (c) The toy config sets `nope_every=2` rather than 4. Why?

??? note "Solution"
    (a) RoPE encodes position by rotating each query and key by an angle proportional to its absolute position, so an attention score depends on the *relative* offset between tokens — but the rotation frequencies are calibrated to the range of positions seen during training (up to 2048). Past that range the phases enter a regime the model never saw, hurting extrapolation. A NoPE layer applies *no* positional rotation, so its attention is exactly translation-invariant: it depends only on content, not on absolute index, and behaves identically whether the sequence is 2048 or 8192 tokens long. Interleaving such layers gives the network a positional-signal-free pathway that generalizes cleanly beyond the training length, while the RoPE layers still supply the fine-grained relative-position information needed for local ordering — the combination extrapolates better than either alone, which is why SmolLM3 adopted it and why Stack-100M's context can be extended to 8192 in mid-training (Ch. 14.8).

    (b) RoPE is a *parameter-free* operation: it rotates Q and K by fixed, precomputed sinusoidal angles that depend only on position and `rope_theta`, with no learnable weights. Omitting it on a layer simply skips that rotation; the layer's Q/K/V/O projections, MLP, and norms — the only things that carry parameters — are unchanged. So `nope_every` shifts *which* layers rotate their queries and keys, but every layer has exactly the same `attn_per_block` count from Exercise 2. That is why `count_params` never references `nope_every`.

    (c) Branch coverage. The rule is `uses_rope(i) == (nope_every <= 0 or ((i + 1) % nope_every) != 0)`. With `n_layers=2` and `nope_every=4`, the two layers evaluate `1 % 4` and `2 % 4`, both non-zero, so *both* take the RoPE branch and the NoPE path is never executed by CI — the smoke test would happily pass with a broken NoPE implementation. Setting `nope_every=2` makes layer index 1 evaluate `2 % 2 == 0` and take the NoPE branch, covering both. (A third branch exists: `nope_every <= 0` disables the interleave entirely, which is what makes the checkpoint exportable as a stock Qwen3 architecture in Ch. 14.4 — worth a separate assertion.)

**7.** Chapter 14.8 extends the context from 2048 to 8192 for the long-context portion of mid-training, roughly 0.8B tokens. (a) Compute the attention-score FLOPs per token at $s=8192$ and express them as a fraction of $6N$. (b) Compute the total compute $C$ for a 0.8B-token pass and the wall-clock at an attention-inclusive MFU of 0.50. (c) Compare against what $6ND$ alone would have predicted, and say what this implies about which kernel you must be using.

??? note "Solution"
    (a) $6 L s d_q = 6 \times 30 \times 8192 \times 512 = 754{,}974{,}720 \approx 7.55\times10^{8}$ FLOPs/token, versus $6N \approx 6.081\times10^{8}$. The ratio is $7.55/6.081 \approx 1.24$, i.e. **+124%** — at 8192 the attention scores cost *more* than every weight matrix in the model combined, and attention is ~55% of all training FLOPs. (The ratio is linear in $s$: quadrupling the context from 2048 quadruples 31% to 124%.)

    (b) Per token, $6.081\times10^{8} + 7.550\times10^{8} = 1.363\times10^{9}$ FLOPs. Over $D = 0.8\times10^{9}$ tokens:

    $$
    C = 1.363\times10^{9} \times 0.8\times10^{9} \approx 1.09\times10^{18}\ \text{FLOPs}.
    $$

    At MFU 0.50 on an A100: $1.09\times10^{18} / (0.50 \times 3.12\times10^{14}) \approx 6.99\times10^{3}$ s $\approx$ **1.9 hours** — which, added to the ~1.2B-token anneal at 2048, reproduces Chapter 14.12's ~3.5 GPU-hour mid-training line.

    (c) $6ND$ alone gives $6.081\times10^8 \times 0.8\times10^9 = 4.86\times10^{17}$ FLOPs — **less than half** the real figure, a 1.7× forecasting miss and the single easiest place in this project to blow a compute estimate. It is also why you must run this phase through a memory- and IO-aware causal kernel: `F.scaled_dot_product_attention(..., is_causal=True)` on its FlashAttention-2 backend (or `flash-attn` directly). A naive implementation that materializes the full $8192\times8192$ score matrix per head does *twice* the FLOPs (no causal skipping) and allocates $O(s^2)$ activation memory per head, which will OOM an 80GB card at any useful batch size. See [FlashAttention I: IO-Awareness & The Online Softmax](../04-kernels-efficiency/02-flash-attention-1.html) and [Long-Context Pretraining & Context Extension](../03-pretraining/13-long-context-pretraining.html).

**8.** You have finished Stack-100M and your team asks you to rebuild the same pipeline as a maintained internal product, where "I wrote it myself" is a liability rather than a virtue. For each of these five stages, name the open-source library you would adopt and state, in one sentence, the specific thing it gives you that the from-scratch version does not: (a) tokenizer encoding of the pretraining corpus, (b) the attention kernel, (c) DPO training, (d) RLVR rollout generation, (e) laptop deployment.

??? note "Solution"
    (a) **HuggingFace `tokenizers`** — a Rust `ByteLevel` BPE implementation with multi-threaded `encode_batch`, roughly two orders of magnitude faster than a pure-Python inner loop, plus a stable serialized `tokenizer.json` that every downstream tool can load. You can import your own trained merges, so you keep the vocabulary you designed.

    (b) **`torch.nn.functional.scaled_dot_product_attention`**, dispatching to a **FlashAttention-2** kernel — tiled, IO-aware attention that never materializes the $s \times s$ score matrix (turning $O(s^2)$ activation memory into $O(s)$; the 195 GB → 34 GB row in this chapter's byte table) and skips masked blocks under `is_causal=True`. `flash-attn` or a Triton kernel are the drop-in alternatives when you need a variant PyTorch does not dispatch.

    (c) **TRL** (`DPOTrainer`) — a maintained implementation of DPO and its variants (IPO, KTO, ORPO, SimPO) with reference-model handling, PEFT/LoRA integration, sequence packing, and multi-GPU support already correct, so preference-data plumbing is configuration rather than code. **Axolotl** or **LLaMA-Factory** wrap it in a YAML-driven runner.

    (d) **vLLM** (or **SGLang**) as the rollout engine, driven by **veRL**, **OpenRLHF**, or TRL's `GRPOTrainer` — continuous batching plus PagedAttention makes generation, which dominates RLVR wall-clock, several times faster than a naive `model.generate` loop, and the RL frameworks handle the weight-synchronization dance between the trainer and the rollout workers.

    (e) **`llama.cpp`** with a **GGUF** export — mature CPU/Metal/AVX kernels, a family of k-quant int4/int5 schemes far better than round-to-nearest, memory-mapped weights, and a portable single-binary runtime. **ONNX Runtime** is the alternative when you need to embed the model in a non-C++ host application.
