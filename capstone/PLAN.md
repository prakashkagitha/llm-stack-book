# Capstone Spec — Build **Stack-100M** End-to-End

> **This file is the single source of truth for Part 14 (the Capstone).**
> Every capstone chapter MUST stay consistent with the names, numbers, component
> choices, and citations fixed here. If a chapter needs a number or a design
> decision, take it from this file — do not invent a different one. The goal is a
> coherent, reproducible project a reader can actually run, not 12 disconnected essays.

The capstone threads the *entire* book into one concrete project: **train, align, and
serve a ~100M-parameter language model from scratch on a single GPU for roughly the cost
of a nice dinner**, using *current* (2025–2026) state-of-the-art components — then push it
as far as a 100M model can honestly go: a narrow, tool-using, retrieval-augmented
"auto-research" agent.

This is the modern update to Andrej Karpathy's "reproduce GPT-2 for ~\$100–\$200"
(nanoGPT / llm.c, 2024). Since then, small models got *dramatically* better at fixed size —
driven by (1) **massive over-training** far past Chinchilla-optimal, (2) **data quality**
(FineWeb-Edu, Cosmopedia), and (3) **architecture/optimizer advances** from the frontier
labs. We adopt those.

---

## 0. The finished artifact & the repo

- **Model name:** `Stack-100M`. **Repo/package:** `capstone/` (Python package `stacklm`).
- **What the reader ends with:** a ~100M-param base model, a mid-trained + instruction-tuned
  chat model, a narrow tool-using agent, all runnable on a laptop after int4 quantization.
- **Primary compute tier (the flagship walkthrough):** a **single NVIDIA A100 (80GB)**,
  rented at ~\$1–2/GPU-hr. The 18B-token WSD stable phase costs
  ≈ **22–29 GPU-hours (≈25 planning) ⇒ ~\$25–\$50**, derived in Ch. 14.1 from
  `C = (6N + 6Lsd_q)D` at an attention-inclusive MFU of 0.45–0.59 (equivalently
  0.34–0.45 under 6ND-only). The **whole project** — data, ladder, mid-training,
  post-training, teacher API, storage, and a ~25% re-run tax — lands at
  **≈\$90–\$100** (itemized in Ch. 14.12). We call this "the ~\$100 model."
- **Also documented (not the flagship path):**
  - **Consumer 24GB GPU** (RTX 4090 / 3090): same recipe, more gradient accumulation,
    ~2–4× wall-clock, ~\$0 if owned.
  - **Free Colab (T4, 16GB):** an on-ramp config (smaller model / fewer tokens) so anyone
    can start with zero dollars. Explicitly a scaled-down subset, not the full run.
- **Deliverable standard:** every code block is complete and copy-paste-runnable. The book's
  CI smoke-tests the whole pipeline at **toy scale** (tiny vocab/model/data, a handful of
  steps, CPU-only, hermetic) — proving the code *runs*; the prose documents the *expected*
  full-run numbers.

## 1. The canonical `Stack-100M` architecture (FIX THESE NUMBERS)

Decoder-only transformer, **deep-and-thin** (a small-model insight: at fixed params, more
layers × narrower width beats shallow × wide — MobileLLM, Liu et al. 2024; also GLM, Qwen3-0.6B).

| Hyperparameter | Value | Notes |
|---|---|---|
| `vocab_size` | **32768** | byte-level BPE we train ourselves |
| `d_model` | **512** | narrow (deep-thin) |
| `n_layers` | **30** | deep |
| `n_heads` | **8** | query heads |
| `n_kv_heads` | **2** | GQA (4:1 ratio) |
| `head_dim` | **64** | 8×64 = 512 |
| `intermediate` (SwiGLU) | **1408** | ≈ 2.75·d, rounded to a multiple of 64 |
| `max_seq_len` (pretrain) | **2048** | extended to 8192 in mid-training |
| `rope_theta` | **10000** | base; rescaled for long-context in mid-training |
| tie embeddings | **yes** | input = output embedding (Press & Wolf 2017) |
| **total params** | **≈ 101M** | embed 16.8M (tied, counted once) + 30 blocks × ~2.82M |

**Parameter accounting the reader must be able to reproduce** (this exact arithmetic appears
in Ch. 14.4): tied embedding `32768×512 = 16.78M`; per block: attention
`Q 512×512 + K 512×128 + V 512×128 + O 512×512 = 0.655M`, SwiGLU MLP `3×512×1408 = 2.163M`,
norms negligible ⇒ `~2.82M/block × 30 = 84.6M`; total `≈ 101.4M`.

### Components — use these, cite the source (ablate ONLY where it teaches something)

- **Pre-norm** residual blocks; **RMSNorm** (Zhang & Sennrich, 2019).
- **RoPE** rotary position embeddings (Su et al., 2021), `θ=10000`. **NoPE** (no positional
  encoding) on a subset of layers — every 4th layer — following **SmolLM3** (HuggingFace, 2025)
  and Kazemnejad et al. (NoPE, 2023): improves length generalization.
- **GQA** grouped-query attention, 2 KV heads (Ainslie et al., 2023) — shrinks the KV cache 4×.
- **QK-norm**: RMSNorm applied to Q and K before attention, for training stability at high LR
  (used widely in recent open models; Henry et al., 2020 "query-key normalization").
- **SwiGLU** gated MLP (Shazeer, 2020).
- **Tied input/output embeddings** (Press & Wolf, 2017) — saves 16.8M params, big at this scale.
- **z-loss** (small penalty on `logsumexp` of logits) and optional **logit soft-cap**
  (Gemma-2 style) for stability.

### Efficiency variants taught as options (implemented, cited, not the default path)

- **MLA — Multi-head Latent Attention** (DeepSeek-V2, 2024): compresses KV into a low-rank
  latent, a further KV-cache win beyond GQA. Taught in Ch. 14.4 as the "if you want DeepSeek's
  trick" upgrade; cross-links the attention/KV-cache chapters.
- **MTP — Multi-Token Prediction** (DeepSeek-V3, 2024; Gloeckle et al., 2024): an auxiliary head
  predicting the next-2 tokens during training for a denser signal and faster convergence.
- **Liquid LFM2** (Liquid AI, 2025) gated-short-convolution + attention **hybrid** — referenced
  as a design point in Ch. 14.4 (the model can interleave a cheap conv-mixer block), cited, with
  an optional block implementation; not the default.

## 2. Data (Ch. 14.2)

- **Budget: ~20B tokens** for the flagship run (≈ **200 tokens/param**, ~10× past Chinchilla's
  ~20 tokens/param). This deliberate **over-training** is the central modern lesson: for a model
  you will *serve*, compute-optimal ≠ deployment-optimal; you pay training compute once and save
  inference forever. (Chinchilla-optimal for 100M is only ~2B tokens.)
- **Mix (cite the SmolLM / FineWeb recipe):**
  - **FineWeb-Edu** (Penedo et al., HuggingFace, 2024) — the bulk; classifier-filtered educational web.
  - **Cosmopedia v2** (HuggingFace) — synthetic textbooks/stories for clean, dense knowledge.
  - **StarCoder** subset (BigCode) — a little code for reasoning/structure.
  - **FineMath / OpenWebMath** — a little math.
  - Approx weights: 70% FineWeb-Edu, 15% Cosmopedia, 10% code, 5% math (tune per Ch. 14.5).
- **Pipeline:** streaming download → quality filter → **dedup** (exact + MinHash near-dup) →
  tokenize → **pack** to `2048` with **document-aware attention masking** (no cross-document
  attention; also intra-doc via reset of position ids) → shard to `.bin` (uint16) memmap files.
- CI/toy path: a tiny synthetic corpus generated in-process (no network), so tests are hermetic.

## 3. Tokenizer (Ch. 14.3)

- **Byte-level BPE**, `vocab_size=32768`, trained from scratch on a data sample (real, from-scratch
  implementation in the chapter; ties to the book's Tokenizer Playground tool).
- Why 32768 and not 50k: at 100M params a 50k×512 embedding is ~26M params (**26% of the model**);
  vocab size is a real design lever at small scale. Show the tradeoff table.
- **Special tokens** reserved up front for later stages: `<|bos|> <|eos|> <|pad|>`,
  chat roles `<|system|> <|user|> <|assistant|> <|end|>`, tool tokens `<|tool_call|> <|tool_result|>`.

## 4. Mini scaling laws (Ch. 14.5) — the differentiator

- Train a **ladder** of tiny models under a shared recipe to fit *your own* scaling law before
  spending the big-run budget. Two complementary methods, both implemented:
  - **Fixed-model / token sweep** at small sizes, and
  - **IsoFLOP profiles** (Hoffmann et al., "Chinchilla", 2022): for several compute budgets, vary
    N vs D, find the min-loss `(N*, D*)`, fit `N*, D* ∝ C^a, C^b`.
- Ladder sizes: e.g. `{4M, 9M, 19M, 43M}` params (scale `d_model`/`n_layers` together), each on a
  matched token budget; fit `L(N,D)=E + A/N^α + B/D^β`; **extrapolate** to pick the final 100M
  config and predict its loss. Then deliberately **over-train** past the compute-optimal point and
  explain why (deployment economics).
- Uses the **6ND** FLOP rule and the book's scaling-laws chapter + the Scaling-Law-Optimal tool.

## 5. Optimizer & schedule (Ch. 14.6)

- **Muon** (Jordan et al., 2024) for the 2D hidden weight matrices (Newton–Schulz orthogonalization
  of the momentum update) + **AdamW** for embeddings, norms, and 1D params — the standard hybrid.
  Muon has driven recent speed records and was used at scale by **Kimi K2** (Moonshot, 2025).
- **MuonClip / QK-clip** (Kimi K2, 2025): clip Q·K logits / rescale to prevent attention-logit
  blow-ups — the stability fix that made Muon work at scale. Implement a simple QK-clip.
- **WSD schedule** — Warmup-Stable-Decay (MiniCPM, Hu et al., 2024; used by DeepSeek): long constant
  "stable" phase, then a short sqrt/linear **decay** phase. Pairs perfectly with **mid-training**:
  the decay phase *is* where we anneal on premium data (Ch. 14.8). Contrast with cosine.
- Also: weight decay 0.1, grad-clip 1.0, `β=(0.9,0.95)`, bf16, batch ≈ 0.5M tokens via grad accum.

## 6. Pretraining loop (Ch. 14.7)

- Full single-GPU loop: bf16 autocast, gradient accumulation, gradient clipping, **activation
  checkpointing** (optional), **checkpoint + resume** (model+opt+step+rng), **throughput & MFU**
  measurement (vs the 6ND / peak-FLOPs), periodic held-out eval + sample generation, loss logging.
- **Scale-out note** (not the flagship path): DDP then FSDP for multi-GPU / the 8×H100 box, and a
  one-paragraph multinode pointer — cross-link the distributed-training chapters. Emphasize the
  single-GPU path is sufficient for 100M.
- Documented **expected curve**: e.g. final train loss in the ~2.8–3.2 nats/token range on this mix
  (give as illustrative "on the order of", never fabricate a precise benchmark).

## 7. Mid-training (Ch. 14.8)

Modern "mid-training" = a phase *between* raw pretraining and post-training (OLMo 2, 2024; used
across recent open models):
- **WSD decay-phase annealing**: run the LR-decay phase on a *higher-quality* mix (more Cosmopedia,
  instruction-flavored data, math/code) — a big quality jump for little compute.
- **Long-context extension**: continue training at `seq_len 8192` with **RoPE base rescaling**
  (θ scaling / YaRN-style, cross-link the long-context chapter).
- **Capability injection**: concentrated math/code so the narrow downstream agent has something to
  stand on. This is where "narrow but real" specialization begins.

## 8. Post-training (Ch. 14.9)

- **SFT**: a chat template (ChatML-like using the reserved special tokens), packed instruction data,
  loss masked to assistant tokens only.
- **DPO** (Rafailov et al., 2023): preference optimization from pairs — feasible at 100M/$;
  contrast with full PPO-RLHF (too heavy at this budget; cross-link the RLHF chapter).
- **Narrow RLVR / GRPO** (RL with Verifiable Rewards; GRPO from DeepSeekMath, Shao et al., 2024):
  a small run on a *verifiable* narrow task (integer arithmetic / simple word problems) where the
  reward is exact-match correctness — showing RLVR works even at 100M *when the task is narrow*.
- Honest framing: what post-training does and does NOT buy at 100M.

## 9. Agentic — narrow "auto-research" (Ch. 14.10)

- **ReAct** loop (Yao et al., 2022): interleave thought → tool-call → observation. Tools: a
  **calculator** and a **retriever** over a small local corpus (BM25 / embedding-lite).
- **Distillation**: generate ReAct trajectories from a *large teacher* model, filter to successful
  ones, format with the tool special tokens, and **SFT the 100M model on the traces** — the only way
  a 100M model does multi-step tool use. (In CI the teacher is stubbed/offline; the prose shows the
  real teacher call.)
- The narrow **"auto-research"** demo: given a question, the agent searches the tiny corpus, reads,
  and synthesizes a short grounded answer. **Brutally honest** about the ceiling: coherent only in
  a narrow, scaffolded domain; this is the frontier of what 100M can do.
- Optional: RLVR on tool-use success to sharpen the loop (cross-link Ch. 14.9 + the agents part).

## 10. Evaluation & serving (Ch. 14.11)

- **Eval**: held-out perplexity; a few lightweight capability probes (arithmetic accuracy, a
  tiny multiple-choice set, retrieval-QA exact-match); an **honest capability report** — a 100M
  model is a narrow tool, not a chatbot oracle. Cross-link the evaluation part; warn on contamination.
- **Serve**: **int8 then int4** post-training quantization (round-to-nearest baseline; explain
  GPTQ/AWQ, Frantar et al. / Lin et al.), export, and **run on CPU/laptop** with measured latency
  and memory. The payoff: your own model generating text locally. Cross-link the quantization &
  inference chapters + the KV-cache tool.

## 11. Retrospective & scale-up (Ch. 14.12)

- Full **cost accounting** table (GPU-hours × price per stage), **reproducibility checklist**
  (seeds, config hashes, data manifest, env), what **breaks** and what to **change to reach 1B**
  (data, parallelism, LR, batch), and a curated **further-reading** list (all the real works cited
  above). Land the plane: the reader has built the whole stack, once, small, and real.

---

## Cross-chapter consistency contract (do not drift)

- Model = **Stack-100M**, exact config in §1. Package = `stacklm`, repo dir = `capstone/`.
- Token budget = **~20B** (over-trained; Chinchilla-optimal ≈ 2B — always frame over-training as
  deliberate deployment economics).
- Primary compute = **1×A100; stable phase 22–29 GPU-hr (~\$25–\$50); whole project ~\$90–\$100**.
  Secondary = 4090 / Colab. Always quote MFU **with its FLOP convention** (6ND-only vs.
  attention-inclusive `6N + 6Lsd_q` — a 31% difference at this shape).
- Components & citations exactly as listed (RoPE+NoPE, RMSNorm, QK-norm, GQA, SwiGLU, tied-emb,
  Muon+MuonClip, WSD, MLA/MTP as options, DeepSeek/Kimi/GLM/SmolLM/Qwen3/Liquid/MobileLLM sources).
- Never fabricate exact benchmark numbers/dates/quotes; illustrative magnitudes are "on the order of".
- Every chapter cross-links the deeper book chapter it builds on (scaling laws, attention, RoPE,
  GQA, optimizers, LR schedules, mixed precision, distributed training, long-context, SFT, DPO,
  RLHF, GRPO, ReAct/agents, RAG, quantization, inference, evaluation) using LINKMAP.md targets.
