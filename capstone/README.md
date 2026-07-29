# Stack-100M — the runnable capstone (`stacklm`)

This is the companion repo for **Part XIV** of *The LLM Stack*. It builds, aligns,
and serves **Stack-100M**: a ~101M-parameter decoder-only language model, trained
from scratch on a single GPU for roughly the cost of a nice dinner, then pushed as
far as a 100M model honestly goes — a narrow, tool-using, retrieval-augmented
"auto-research" agent.

It is the modern update to Karpathy's "reproduce GPT-2 for ~$100" (nanoGPT /
llm.c): same spirit, but with the 2025–2026 recipe — deliberate **over-training**
(~200 tokens/param), **data quality** (FineWeb-Edu / Cosmopedia), and current
architecture/optimizer advances (RoPE+NoPE, GQA, QK-norm, SwiGLU, tied
embeddings, **Muon+MuonClip**, **WSD** schedule, mid-training, DPO, GRPO).

> The canonical spec lives in [`PLAN.md`](PLAN.md). Every number below comes from
> there. The prose chapters are `content/14-capstone/01…12`.

## What Stack-100M is (the frozen config)

| | value | | value |
|---|---|---|---|
| `vocab_size` | 32768 | `n_kv_heads` (GQA) | 2 |
| `d_model` | 512 (deep-thin) | `head_dim` | 64 |
| `n_layers` | 30 | `intermediate` (SwiGLU) | 1408 |
| `n_heads` | 8 | `max_seq_len` | 2048 → 8192 (mid-train) |
| tie embeddings | yes | `rope_theta` | 10000 (rescaled long-ctx) |
| **total params** | **≈ 101.35M** | NoPE | every 4th layer |

Reproduce the parameter count yourself:

```bash
python3 -m stacklm.config
# ... total  101,353,728  (101.354M)
```

## Repo layout

```
capstone/
├── PLAN.md               # canonical Stack-100M spec (source of truth)
├── README.md             # this runbook
├── requirements.txt      # hermetic smoke set + (commented) real-run extras
├── smoke_test.py         # end-to-end toy-scale pipeline, CPU-only, hermetic
└── stacklm/
    ├── config.py         # StackConfig, count_params, toy_config
    ├── tokenizer/        # from-scratch byte-level BPE (pure stdlib) + specials
    ├── model/            # rmsnorm, rope(+NoPE), swiglu, attention(GQA+QK-norm),
    │                     #   block, transformer; optional mla.py, mtp.py
    ├── data/             # source registry + synthetic corpus, quality filters, exact +
    │                     #   MinHash dedup, pack (doc-aware), uint16 shards, memmap ds,
    │                     #   build_corpus driver (budgets/interleave/holdout/manifest)
    ├── scaling/          # fit L(N,D)=E+A/N^a+B/D^b (numpy, no scipy) + IsoFLOP
    ├── optim/            # muon (Newton-Schulz), qk_clip, schedule (WSD/cosine), build
    ├── train/            # single-GPU pretrain loop (CPU-safe bf16 guard), ckpt/resume
    ├── mid/              # continued training: anneal + long-context RoPE rescale
    ├── post/             # chat template, sft, dpo, grpo (RLVR arithmetic)
    ├── agent/            # tools (calc + BM25), react loop/parser, distill, fake teacher
    ├── eval/             # perplexity + arithmetic / retrieval-QA probes
    └── serve/            # int8/int4 RTN quantization + CPU generate
```

## Run the smoke test (CPU, seconds, offline)

The smoke test exercises the **whole pipeline** at toy scale (tiny vocab/model,
a few dozen synthetic docs, a handful of optimizer steps per stage). It proves the
code *runs*; the prose documents the *expected* full-run numbers.

```bash
# from the repo root
python3 capstone/smoke_test.py                       # prints "SMOKE OK"

# certify hermeticity (simulates the CPU-only CI env: blocks packages CI lacks)
python3 scripts/ci_sim_run.py capstone/smoke_test.py # must also print "SMOKE OK"
```

It runs, in order: build toy config → train a byte-BPE on synthetic text → pack a
synthetic corpus → build the model and assert the **real** param count → pretrain →
mid-train (with RoPE rescale) → SFT → DPO → GRPO on arithmetic → ReAct agent with
the offline fake teacher → int8/int4 quantize + CPU generate.

**Hermetic contract:** the only third-party imports anywhere on the smoke path are
`numpy`, `torch`, `einops`, `scikit-learn`, and the stdlib. No
network/API/dataset downloads; every `.cuda()`/bf16 path is guarded so CPU works.

## The real full run — commands per stage

At full scale, each module has a real entry point. The flagship path is a single
**A100 (80GB)**; the config numbers are frozen in `PLAN.md`.

```bash
# 0. Tokenizer — train the 32,768-vocab byte-BPE on a data sample (mostly CPU)
python3 -c "from stacklm.tokenizer import StackTokenizer, SPECIAL_TOKENS; \
  t=StackTokenizer(); t.train(open('data/sample.txt').read(), 32768, SPECIAL_TOKENS); \
  t.save('tokenizer/stack100m-32768.json')"

# 1. Scaling ladder — fit YOUR OWN law before spending the big budget
python3 -c "from stacklm.scaling import fit_scaling_law, compute_optimal_allocation; ..."

# 2. Pretrain — 20B tokens, WSD stable phase (the "~\$100" step)
python3 -c "from stacklm.train import pretrain; ..."   # or wire a config + shards

# 3. Mid-train — WSD decay-phase anneal on premium mix + 8192 ctx RoPE rescale
python3 -c "from stacklm.mid import mid_train; ..."

# 4. Post-train — SFT (chat template), then DPO, then narrow GRPO/RLVR
python3 -c "from stacklm.post import sft_train, dpo_train, grpo_train; ..."

# 5. Agent — distill ReAct traces from a real teacher, SFT on the traces
python3 -c "from stacklm.agent import distill, make_stub_teacher; ..."  # swap teacher

# 6. Eval + serve — perplexity/probes, int4 quantize, run on a laptop
python3 -c "from stacklm.serve import quantize_stacklm, generate; ..."
```

The training loops (`stacklm.train.pretrain`, `stacklm.mid.mid_train`) take a
`PackedMemmapDataset` built by `stacklm.data.build_shards` (single stream) or, for a
real corpus, by `stacklm.data.build_corpus` — which enforces the per-source token
budgets, interleaves the four sources by weight, shuffles, splits off a held-out set,
and writes `manifest.json`. Pass `offline=False` to stream the real datasets (lazily
importing `datasets`); note `bigcode/starcoderdata` is gated and needs
`huggingface_hub.login()`. Near-dedup at full 20B-token scale should be delegated to
`capstone/scripts/dedup_datatrove.py` (HuggingFace `datatrove`) rather than the
from-scratch `near_dedup_stream`, which is there to make MinHash legible.

```bash
python3 -m stacklm.data.build_corpus       # tiny offline demo: shards + manifest
```

## Compute tiers

| tier | GPU | pretrain wall-clock | approx cost |
|---|---|---|---|
| **Flagship** | 1× A100-80GB (~$1.80/hr spot) | ~22–29 GPU-hr (MFU 45% 6ND / 58% attn) | **~$25–$50** |
| Consumer | RTX 4090 / 3090 (24GB) | ~2–4× A100 (more grad-accum) | ~$0 if owned |
| Free | Colab T4 (16GB) | scaled-down on-ramp config only | $0 |

The T4/4090 tiers use the same recipe with more gradient accumulation (and fp16 +
GradScaler instead of bf16 on non-Ampere); the T4 tier is an explicit scaled-down
subset, not the full 20B-token run.

## Cost accounting (flagship, $1.80/GPU-hr A100-80GB spot)

| Stage (chapter) | GPU-hr | Stage $ |
|---|---:|---:|
| Tokenizer BPE (14.3, mostly CPU) | 0.3 | 0.54 |
| Scaling-law ladder {4M,9M,19M,43M} (14.5) | 4.5 | 8.10 |
| **Pretrain, 20B tokens, WSD stable (14.7)** | 22.0 | 39.60 |
| Mid-training: 8192 ctx + anneal, ~3B tok (14.8) | 4.5 | 8.10 |
| SFT (14.9) | 1.0 | 1.80 |
| DPO (14.9) | 1.2 | 2.16 |
| GRPO / narrow RLVR (14.9) | 3.0 | 5.40 |
| Agent distillation: teacher traces + SFT (14.10) | 1.0 | 9.80 (incl. teacher API) |
| Eval + int8/int4 quant + export (14.11) | 1.2 | 2.16 |
| Storage + egress (~200 GB, one month) | — | 5.00 |
| **Subtotal** | **38.7** | **82.66** |
| Re-run reality tax (~25% of GPU $) | — | 17.40 |
| **Grand total** | | **≈ $100** |

**Over-training thesis:** Chinchilla-optimal for 100M ≈ 2B tokens (~$4 pretrain);
we train ~20B (~200 tok/param, ~10× compute-optimal). You pay training compute
*once* and save inference *forever* — for a model you will serve, compute-optimal
≠ deployment-optimal. Break-even (serving FLOPs = pretrain FLOPs) is ~234M requests.

**Memory ladder (quantization):** fp32 ≈406MB → bf16 ≈203MB → int8 (row) ≈102MB →
int4 (group=64) ≈63MB. Model + 2048-token KV cache (GQA) ≈ 100MB — runs on a laptop.

## Back to the chapters

Each stage above maps to a Part-XIV chapter (`content/14-capstone/`): 01 overview,
02 data, 03 tokenizer, 04 architecture, 05 scaling laws, 06 optimizer/schedule,
07 pretraining, 08 mid-training, 09 post-training (SFT/DPO/GRPO), 10 agentic,
11 evaluation & serving, 12 retrospective & scale-up. The chapters carry the
citations (RoPE, NoPE/SmolLM3, GQA, QK-norm, SwiGLU, Muon/MuonClip/Kimi-K2, WSD/
MiniCPM, OLMo-2 mid-training, DPO, GRPO/DeepSeekMath, ReAct, GPTQ/AWQ, MLA/MTP).
