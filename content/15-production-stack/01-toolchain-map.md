# 15.1 The Production Toolchain: A Map From Scratch to Shipping

Part XIV was a workshop. We stripped the LLM stack down to bolts and springs and rebuilt every piece of **Stack-100M** by hand: a byte-level BPE tokenizer with its merge loop written out, a training step where you can see the `loss.backward()`, a Muon update spelled as a Newton–Schulz iteration, a GRPO advantage computed with a `for` loop over a group. The package is called `stacklm`, and its whole reason to exist is that *nothing is hidden*. You can set a breakpoint anywhere and read the state.

That is exactly what you should **not** ship.

Part XV rebuilds the identical journey — same architecture, same ~20B-token budget, same single-A100 economics — but this time with the libraries a working team actually reaches for in 2026. The thesis of this whole part fits on a bumper sticker:

> **Hand-roll to learn; reach for the library to ship.**

The from-scratch version taught you the mechanism so that when a production library throws an assertion at 3 a.m., you know what it *means*. The library gives you the twelve person-years of edge-case handling, kernel tuning, distributed correctness, and format compatibility that you should never re-derive under deadline. Both halves are load-bearing. An engineer who only knows the library is helpless when it breaks; an engineer who only knows the from-scratch code ships a training run that silently NaNs at step 40,000 because they never implemented sharded-checkpoint resharding.

This chapter is the map. It lays out the end-to-end pipeline, gives you a single crosswalk table (**stage → our `stacklm` module → the production library → the Part-XIV chapter**), states honestly where "the library" is really three competing libraries, and warns you up front about the one thing that will bite you hardest: **version drift**. By the end you should be able to point at any box in the pipeline and name both the toy version you understand and the real tool you would install.

## The Pipeline, End to End

Everything in this book has been assembling one artifact along one conveyor belt. Here is the whole belt in one diagram, annotated with the library that owns each station.

```text
                         THE STACK-100M PRODUCTION PIPELINE
                         (each box: what it does | the 2026 library)

 ┌─────────────┐   ┌──────────────┐   ┌───────────────┐   ┌────────────────────┐
 │ RAW WEB     │   │ CURATED      │   │ TOKENIZER     │   │ PACKED TOKEN SHARDS │
 │ FineWeb-Edu │──▶│ filter+dedup │──▶│ byte-BPE 32k  │──▶│ .bin uint16 memmaps │
 │ Cosmopedia  │   │  datatrove   │   │  tokenizers   │   │  (datatrove writer) │
 │ code, math  │   │ HF datasets  │   │ sentencepiece │   │                     │
 └─────────────┘   └──────────────┘   └───────────────┘   └─────────┬──────────┘
                        Ch 15.2            Ch 15.3                   │
                                                                    ▼
 ┌────────────────────────────────────────────────────────────────────────────┐
 │                        PRETRAIN  (Ch 15.4)                                    │
 │  Stack-100M: 30L × 512d, GQA, RoPE+NoPE, SwiGLU, RMSNorm, QK-norm            │
 │  Muon+AdamW · WSD schedule · bf16 · FSDP2 · activation checkpoint · MFU log  │
 │                    torchtitan  /  nanotron  (Megatron/DeepSpeed at scale)    │
 └───────────────────────────────────┬────────────────────────────────────────┘
                                      ▼  base checkpoint (safetensors)
 ┌────────────────────────────────────────────────────────────────────────────┐
 │              MID-TRAIN → POST-TRAIN  (Ch 15.5)                                │
 │  WSD decay anneal on premium mix · long-ctx (θ rescale)                       │
 │  SFT (chat template, assistant-only loss) → DPO → narrow GRPO (vLLM rollouts) │
 │                    TRL  ·  alignment-handbook  ·  peft/unsloth  ·  veRL       │
 └───────────────────────────────────┬────────────────────────────────────────┘
                                      ▼  aligned checkpoint
 ┌──────────────────────┐   ┌──────────────────────┐   ┌────────────────────────┐
 │ QUANTIZE             │   │ SERVE                │   │ EVALUATE               │
 │ GPTQ/AWQ → int4      │──▶│ OpenAI-compat API    │   │ perplexity + probes    │
 │ GGUF for laptop      │   │ continuous batching  │   │ contamination check    │
 │ llm-compressor       │   │ vLLM · SGLang        │   │ lm-evaluation-harness  │
 │ AutoAWQ · llama.cpp  │   │ llama.cpp (CPU)      │   │                        │
 └──────────────────────┘   └──────────────────────┘   └────────────────────────┘
                                      Ch 15.6
```

Read it left to right, top to bottom. Tokens flow one way; each stage consumes the previous stage's on-disk artifact and produces the next. The interfaces between stages are *files in standard formats* — this is the single most important design property of the whole pipeline, and the reason you can swap any one library without rewriting the others:

- Data → tokenizer: a stream of documents (JSONL or Arrow).
- Tokenizer → pretrain: a `tokenizer.json` plus packed `.bin` shards of `uint16` token IDs.
- Pretrain → post-train: a checkpoint directory in **safetensors**.
- Post-train → serve: another safetensors checkpoint, optionally a quantized one.
- Serve/eval: an **OpenAI-compatible HTTP API** or a direct model load.

Because the seams are standard formats, the crosswalk is not "rip out `stacklm` and install a monolith." It is "at each seam, replace the `stacklm` module with the library that produces or consumes the same file." That is what makes the from-scratch code pedagogically honest: it targets the exact same interfaces the real tools do.

## Hand-Roll to Learn, Reach for the Library to Ship

Why did we write ~4,000 lines of `stacklm` at all, if the plan was always to replace it? Because the two versions answer two different questions, and you need both answers to be employable.

**The from-scratch version answers "what is happening?"** When you wrote the BPE merge loop, you learned that a tokenizer is a deterministic greedy merge over byte pairs ranked by training frequency — so you are never mystified by why a space-prefixed token differs from a non-prefixed one, or why adding a special token can shift every downstream ID. When you wrote the GRPO loop, you learned that the "advantage" is just a within-group z-score of rewards, so you know exactly which knob to turn when your RL run collapses to a single response.

**The library version answers "how do I do this correctly at scale, today, with the rest of the team?"** The Hugging Face `tokenizers` trainer is written in Rust and will train on a multi-gigabyte sample in minutes with parallelism your Python loop cannot touch. `torchtitan` implements FSDP2 sharding, distributed checkpointing that can *reshard* across a different GPU count on resume, and per-step MFU logging — each of which is a subtle systems project in its own right. `vLLM` implements PagedAttention and continuous batching that lift serving throughput by an order of magnitude over a naive generate loop.

There is a real cost boundary here, and it is worth naming precisely: **the library's value is highest exactly where correctness is subtle and invisible.** A tokenizer that is 5% slower is fine; a distributed checkpoint that silently drops an optimizer-state shard on a resharded resume corrupts a $50 training run and you will not notice until the loss curve looks wrong 10,000 steps later. So the rule is not "libraries good, from-scratch bad." The rule is:

- **Reach for the library** whenever the stage involves *distributed correctness, kernel performance, format compatibility, or edge-case-dense data handling* — data curation, tokenizer training, distributed pretraining, quantization, and serving. These are where a decade of other people's bug fixes live.
- **Keep it simple / from-scratch** for the parts that are genuinely small and where transparency beats abstraction: your scaling-law fit, your eval harness glue, your agent's ReAct loop. A library here often adds more surface area than it removes.

The middle case is post-training. `TRL`'s `SFTTrainer`/`DPOTrainer`/`GRPOTrainer` are the standard, and at Stack-100M scale they run comfortably on one GPU — but the trainers are thin enough that your from-scratch understanding maps almost line-for-line onto their configs. That is the sweet spot Part XV is built to exploit: you will recognize every argument.

!!! tip "Practitioner tip"

    The order in which you should trust a library is: (1) does it read/write the standard format at the seam? (2) is it actively maintained (commits this quarter)? (3) does the upstream repo have a runnable example matching your exact task? If all three are yes, use it and read its example, not its docs — examples drift less than prose. If only the docs exist and the last commit was 14 months ago, treat the library as a reference implementation to copy from, not a dependency to pin.

## The Crosswalk Table

This is the spine of the whole part. Every row is a stage of the pipeline. Column 2 is the module you built by hand in Part XIV. Column 3 is the production library (or libraries) you install to ship it. Column 4 points back to the from-scratch chapter so you can hold the mechanism and the tool side by side. Later chapters of Part XV expand one or two rows each.

| Stage | `stacklm` module (from scratch) | Production library (2026) | Built by hand in |
|---|---|---|---|
| Data curation & dedup | `stacklm.data.pipeline` | **`datatrove`** (+ HF `datasets` streaming); `nemo-curator`, `dolma` as alternatives | [Ch 14.2 Data Pipeline](../14-capstone/02-data-pipeline.html) |
| Tokenizer training | `stacklm.tokenizer.bpe` | **HF `tokenizers`** (Rust `BpeTrainer`); `sentencepiece` | [Ch 14.3 Tokenizer](../14-capstone/03-tokenizer.html) |
| Model definition | `stacklm.model` | HF `transformers` `PretrainedModel` / `torchtitan` model defs | [Ch 14.4 Architecture](../14-capstone/04-architecture.html) |
| Scaling-law fit | `stacklm.scaling` | (mostly bespoke: `numpy`/`scipy` fit — no monolith) | [Ch 14.5 Mini Scaling Laws](../14-capstone/05-mini-scaling-laws.html) |
| Optimizer & schedule | `stacklm.optim.muon` | `torch` AdamW + Muon (reference impl); trainer-provided | [Ch 14.6 Optimizer & Schedule](../14-capstone/06-optimizer-and-schedule.html) |
| Pretraining loop | `stacklm.train.loop` | **`torchtitan`** (FSDP2); `nanotron`, `Megatron-LM`/`DeepSpeed`; `accelerate` for small | [Ch 14.7 Pretraining Run](../14-capstone/07-pretraining-run.html) |
| Mid-training | `stacklm.train.midtrain` | same trainer, new config (WSD decay anneal, θ rescale) | [Ch 14.8 Mid-Training](../14-capstone/08-mid-training.html) |
| SFT / DPO / GRPO | `stacklm.post.*` | **`TRL`** (`SFTTrainer`/`DPOTrainer`/`GRPOTrainer`); `alignment-handbook`; `peft`/`unsloth`; `veRL`/`OpenRLHF` at scale | [Ch 14.9 Post-Training](../14-capstone/09-post-training.html) |
| Narrow agent | `stacklm.agent.react` | (bespoke ReAct loop; distill traces via a teacher API) | [Ch 14.10 Narrow Agent](../14-capstone/10-agentic-narrow.html) |
| Quantization | `stacklm.quant.rtn` | **`llm-compressor`** (GPTQ), `AutoAWQ`, GGUF via `llama.cpp` | [Ch 14.11 Eval & Serving](../14-capstone/11-evaluation-and-serving.html) |
| Serving | `stacklm.serve.generate` | **`vLLM`** (`vllm serve`), `SGLang`, `TensorRT-LLM`, `llama.cpp` (CPU) | [Ch 14.11 Eval & Serving](../14-capstone/11-evaluation-and-serving.html) |
| Evaluation | `stacklm.eval.probes` | **`lm-evaluation-harness`** | [Ch 14.11 Eval & Serving](../14-capstone/11-evaluation-and-serving.html) |

A few rows deserve a note now, because they set expectations for the rest of the part.

**Not every stage has a "the library."** Scaling-law fitting, the ReAct agent loop, and the glue of an eval report are genuinely bespoke. There is no `pip install scaling-laws` that will fit *your* recipe on *your* mix; you run a small ladder and fit `L(N,D)=E + A/N^{\alpha} + B/D^{\beta}` with `scipy.optimize`. Part XV is honest about this: where the from-scratch version *is* the production practice, we say so and move on rather than inventing a dependency.

**Some rows are one library with several names.** "Pretraining trainer" is `torchtitan` *or* `nanotron` *or* `Megatron-LM`+`DeepSpeed`, and which one a team uses is a cultural/vendor choice more than a technical one. At Stack-100M scale (single GPU) you often use none of them — plain `accelerate` around your loop is enough, and the big trainers only earn their complexity when you cross into multi-node. We lead with `torchtitan` because it is PyTorch-native and its config maps most cleanly onto the hand-written loop of [Ch 14.7](../14-capstone/07-pretraining-run.html).

**The serving/quantization/eval trio collapses into one Part-XIV chapter (14.11) but three production tools.** That asymmetry is deliberate: doing it by hand, quantize-serve-eval is a couple hundred lines; doing it for real pulls in vLLM, llm-compressor, and lm-evaluation-harness, each a substantial system. Chapter 15.6 gives them the room the from-scratch chapter could not.

## Version Drift Is the Real Enemy

Here is the uncomfortable truth about writing (or reading) a book that names specific library versions: **the code will rot.** Not the ideas — PagedAttention will still be PagedAttention — but the *surface*. Flags get renamed, arguments move from the constructor to a config object, a trainer splits into two classes, a default flips. If you copy a command from a 2024 blog post into a 2026 environment, there is a real chance it errors on an unknown keyword argument.

This is not a reason to avoid pinning versions. It is a reason to pin them *and* to teach the reader how to survive when the pin is stale. Three habits:

**1. Pin, and record the pin next to the artifact.** Every training run should emit a `requirements.txt` (or a `uv.lock`) into its output directory alongside the checkpoint. Reproducibility is not "I used vLLM"; it is "I used exactly these resolved versions."

```bash
# Pin everything, and freeze the resolved set INTO the run's output dir.
# (illustrative versions — check the upstream release page for current ones)
pip install \
  "datatrove==0.6.*" \
  "tokenizers==0.21.*" \
  "torch==2.7.*" \
  "torchtitan==0.1.*" \
  "trl==0.19.*" \
  "peft==0.14.*" \
  "llmcompressor==0.5.*" \
  "lm-eval==0.4.*"

# vLLM hard-pins the exact torch it was built against (0.8.x -> torch 2.6.0,
# 0.9.x -> torch 2.7.0), so it fights any torch pin you make yourself. Give the
# serving engine its own virtualenv; the seam between train and serve is a file.
pip install "vllm==0.9.*"     # separate env, matches torch 2.7 above

# Freeze the ACTUAL resolved versions next to the checkpoint, not the loose pins above.
pip freeze > runs/stack100m/requirements.lock.txt
```

Treat the numbers above as illustrative, not gospel — they reflect the shape of releases in this era, and you should read the current version off the upstream release page rather than trust a printed book. That honesty *is* the practice: a book cannot ship you a working `pip install` line that survives two years; it can teach you to check.

**2. Prefer the API's stable core over its fashionable edge.** Libraries have a stable spine and a churning surface. `SFTTrainer(model, args=SFTConfig(...), train_dataset=ds)` has been stable in shape for a long time (note the second *positional* slot is `args`, inherited from `transformers.Trainer` — pass the dataset by keyword); whether packing is `packing=True` on the config or a separate collator has moved around. When you write code you intend to keep, lean on the spine and isolate the churny bits behind a thin adapter so a rename touches one line.

**3. When in doubt, read the upstream example, pinned to your version's git tag.** Every serious library ships runnable examples. If your installed version is `trl==0.19.1`, check out the repo at tag `v0.19.1` and read `examples/` there — not `main`, which may already be two breaking changes ahead of you.

```bash
# The single most reliable way to get a WORKING snippet for your exact version:
git clone https://github.com/huggingface/trl && cd trl
git checkout v0.19.1          # match the version you actually pip-installed
ls examples/scripts/          # sft.py, dpo.py, … — runnable, version-matched
ls trl/scripts/               # the CLI entry points (grpo.py lives HERE, not in examples/)
```

Throughout Part XV, whenever we show a library call, we will (a) hedge the version, (b) show the *mechanism* the call performs so you can recognize its renamed cousin, and (c) point at the upstream example directory rather than pretending our snippet is eternal. If a snippet in this book errors on an argument, the fix is almost always "an argument moved" — not "the concept changed." Fix the argument; the concept is what you came for.

!!! warning "Common pitfall"

    Do not mix versions across the seam without checking the format contract. The classic failure: you pretrain with one `transformers` version that writes a config with a new field (say a `rope_scaling` sub-key), then load into an older `vLLM` that does not understand it and either errors or — worse — silently ignores it and serves a model with the wrong context handling. The seams are standard formats, but "standard" evolves. When you cross a seam, confirm both sides agree on the format version, not just that both sides "support safetensors."

## A Worked Costing of the Whole Belt

The map is not just topological; it has a budget attached, and the budget is what makes Stack-100M a real project rather than a toy. Let us walk the compute cost of the flagship path stage by stage, using the fixed numbers from the capstone spec so the arithmetic is reproducible. This also shows you *which* stages the production libraries are actually saving you time on versus saving you money.

!!! example "Worked example: where the ~$100 goes"

    **Compute model.** Training FLOPs follow the standard rule $C \approx 6ND$ for $N$ parameters and $D$ tokens (attention-inclusive it is $C=(6N+6Lsd_{\text{model}})D$, which at Stack-100M's $L=30$, $s=2048$, $d_{\text{model}}=512$ adds 31% on top, but $6ND$ is the headline). Stack-100M has $N \approx 1.01\times10^{8}$ params and the stable phase burns $D \approx 1.8\times10^{10}$ tokens.

    **Pretraining FLOPs (the dominant cost):**

    $$
    C_{\text{stable}} \approx 6 \times (1.01\times10^{8}) \times (1.8\times10^{10}) \approx 1.09\times10^{19}\ \text{FLOPs}.
    $$

    An A100 (80GB) delivers on the order of $3\times10^{14}$ bf16 FLOP/s peak. At a realistic **MFU of ~0.45** — quoted against the same $6ND$ numerator we just used, which is ~0.59 of peak once the attention term is counted (see [Ch 14.12](../14-capstone/12-retrospective-and-scaleup.html) for why the convention must be stated) — effective throughput is $\approx 1.35\times10^{14}$ FLOP/s. Wall-clock:

    $$
    t \approx \frac{1.09\times10^{19}}{1.35\times10^{14}} \approx 8.1\times10^{4}\ \text{s} \approx 22\ \text{GPU-hours}.
    $$

    That is the well-tuned end of the spec's **22–29 GPU-hour** band — the top of the band is the same run at MFU(6ND) ~0.34 — and at ~$1.20–$1.80/GPU-hr it is **~$25–$50**, the biggest single line item.

    **Now the crosswalk insight.** The production library (`torchtitan`) does *not* change the $6ND$ FLOP count — physics is physics. What it changes is the **MFU multiplier**. A naive from-scratch loop with no fused kernels, eager attention, and a micro-batch too small to fill the tensor cores might run at MFU ~0.20 instead of ~0.45. (Activation checkpointing is *not* on that list: recompute buys memory headroom at the price of ~30% extra hardware FLOPs that the $6ND$ numerator never counts, so it lowers MFU even as it raises HFU — the flagship run uses none.) That is not a rounding error: it is the difference between 22 and ~50 GPU-hours — it **doubles the bill**. So on the pretraining row, the library's payoff is measured in *dollars*, directly, through hardware utilization.

    **The other stages are cheap by comparison.** Tokenizer training, data tokenization, SFT (a few hundred million tokens), DPO (tens of thousands of pairs), and a narrow GRPO run are each a small fraction of the pretraining burn — on the order of single-digit GPU-hours combined. Here the library's payoff is measured in *engineer-hours and correctness*, not compute dollars: `datatrove`'s MinHash dedup is not saving you GPU time, it is saving you from training on 30% duplicated garbage.

    **Whole project.** Add data processing (CPU-heavy, cheap), the scaling-law ladder (a few small runs), mid-training, post-training, a teacher-API bill for agent-trace distillation, storage, and a ~25% re-run tax, and the total lands at the spec's **~$90–$100**. The pretraining GPU burn is the anchor; everything else is the tax.

The lesson of the costing is the same as the lesson of the crosswalk: **the library earns its keep differently at different stages.** At the pretraining stage it earns *money* (MFU). At the data and serving stages it earns *correctness and throughput*. At the scaling-law and agent stages it earns almost nothing, which is why we keep those from-scratch.

## Reading Part XV: What Each Chapter Delivers

With the map in hand, here is how the rest of the part cashes out each region of the diagram. Every chapter follows the same contract from the spec: name the real library and why it is standard in 2026, show the runnable command, cross-link the Part-XIV chapter where we hand-rolled it, and be honest about version drift.

- **15.2 — Data at Scale (mirrors [Ch 14.2](../14-capstone/02-data-pipeline.html)).** The FineWeb-style pipeline in `datatrove`: readers, URL/language/quality filters, MinHash near-dedup, a tokenizing writer, and the `LocalPipelineExecutor` → `SlurmPipelineExecutor` scale-out. Plus HF `datasets` streaming, and `nemo-curator`/`dolma` as alternatives. It produces the exact packed shards `stacklm` consumes — same seam, industrial pipe.

- **15.3 — Tokenizer Training (mirrors [Ch 14.3](../14-capstone/03-tokenizer.html)).** Train a real byte-level BPE with the Rust `BpeTrainer` and `ByteLevel` pre-tokenizer, wrap it in `PreTrainedTokenizerFast`, reserve the chat/tool special tokens up front, and export. The `sentencepiece` path and when a unigram model beats BPE. Cross-links the [tokenization chapter](../02-transformer/01-tokenization.html).

- **15.4 — Pretraining With a Real Trainer (mirrors [Ch 14.7](../14-capstone/07-pretraining-run.html)).** `torchtitan` as primary: the config file, FSDP2 and tensor parallelism, activation checkpointing, the WSD schedule, distributed checkpointing, and MFU logging — every knob mapped onto the hand-written loop. `nanotron`, `Megatron-LM`/`DeepSpeed` at scale; `accelerate` for the single-GPU path. Cross-links [FSDP](../03-pretraining/05-distributed-data-parallel.html) and [mixed precision](../03-pretraining/08-mixed-precision-fp8.html).

- **15.5 — Post-Training With `TRL` (mirrors [Ch 14.9](../14-capstone/09-post-training.html)).** `SFTTrainer` with a chat template and assistant-only loss, `DPOTrainer`, and `GRPOTrainer` with vLLM rollouts — real configs mapped onto the from-scratch versions. The `alignment-handbook` recipes, `peft`/`unsloth` for cheap adapters, `veRL`/`OpenRLHF` for scale, `lm-evaluation-harness` as the gate. Cross-links [SFT](../05-posttraining-alignment/01-sft-instruction-tuning.html), [DPO](../05-posttraining-alignment/07-dpo-and-variants.html), and [GRPO](../05-posttraining-alignment/08-grpo-rloo.html).

- **15.6 — Serving, Quantizing & Evaluating (mirrors [Ch 14.11](../14-capstone/11-evaluation-and-serving.html)).** `vllm serve` for the OpenAI-compatible API with continuous batching and metrics; GPTQ/AWQ via `llm-compressor`/`AutoAWQ`; GGUF export for `llama.cpp` CPU/laptop serving; the honest eval with `lm-evaluation-harness`. Cross-links [vLLM internals](../07-inference-serving/03-vllm-internals.html) and [PTQ](../04-kernels-efficiency/07-quantization-ptq.html).

A concrete taste of the substitution — the same three seams (train → align → serve) as they look with the real tools, so you can see how thin the glue actually is once each library owns its stage:

```bash
# ── SEAM 1: pretrain (torchtitan owns FSDP2, WSD, checkpointing, MFU) ───────────
# Launch from a config; the config IS the hand-written loop's knobs.
# nproc_per_node = the GPUs you actually have: 1 for the single-A100 Stack-100M run,
# 8 on a full node (torchrun errors out if you ask for more devices than exist).
torchrun --nproc_per_node=1 -m torchtitan.train \
    --job.config_file ./stack100m.toml  # d_model=512, n_layers=30, WSD, bf16, no recompute
# → writes runs/stack100m/checkpoint/step-XXXXX/ in safetensors + distributed format

# ── SEAM 2: post-train (TRL owns the SFT/DPO/GRPO trainers) ─────────────────────
python -m trl.scripts.sft \
    --model_name_or_path runs/stack100m/hf \
    --dataset_name your/sft-mix \
    --packing --assistant_only_loss \
    --output_dir runs/stack100m-sft
# then dpo.py, then grpo.py (grpo spins up vLLM for on-policy rollouts) — see Ch 15.5

# ── SEAM 3: quantize + serve (llm-compressor + vLLM own this) ───────────────────
python quantize_w4a16.py    # llm-compressor GPTQ recipe → runs/stack100m-int4
vllm serve runs/stack100m-int4 --max-model-len 8192 --port 8000
# → OpenAI-compatible endpoint; curl it exactly like you'd curl a frontier API
```

Notice what is *not* in those commands: no attention kernel, no sharding logic, no PagedAttention block table, no MinHash/LSH dedup banding. Every one of those is something you *understand* from Part XIV and *do not maintain* in Part XV. That is the entire point of the map.

!!! interview "Interview Corner"

    **Q:** You clearly know how to write a training loop and a tokenizer from scratch — you built Stack-100M by hand. So why would you *ever* bring in `torchtitan`, `TRL`, and `vLLM` instead of just shipping your own code? Isn't that just adding dependencies you don't control?

    **A:** Because the from-scratch code and the library are optimizing different objectives, and I want both. My hand-written loop optimizes *understanding* — it exists so I can reason about failures. The libraries optimize *correctness at scale, hardware utilization, and format compatibility*, which is where multi-year, multi-contributor bug-fixing lives and where I have no business re-deriving under deadline. Concretely: at the pretraining stage the library is worth real money — a naive eager loop might run at ~0.20 MFU where torchtitan hits ~0.45, which literally doubles a $50 GPU bill through nothing but utilization. At the data and serving stages it is worth correctness and throughput — MinHash dedup and PagedAttention are exactly the kind of subtle, invisible logic where a hand-rolled version quietly corrupts a run or serves 10× fewer requests. The tell of a strong engineer is knowing *where* to draw the line: I keep the scaling-law fit and the ReAct agent loop from-scratch because a library there adds more surface than it removes, and I reach for the library precisely where correctness is subtle and invisible. And I can only make that judgment call *because* I built the from-scratch version first — otherwise the library is a black box I can't debug when it throws an assertion at 3 a.m.

!!! note "Aside: why the seams are files, not function calls"

    You might expect a "pipeline" to be one Python process passing tensors between stages. It is not, and that is deliberate. Each stage is a separate job — often on a separate machine, sometimes days apart — communicating through on-disk artifacts in standard formats (JSONL/Arrow, `tokenizer.json`, safetensors, an HTTP API). This file-based seam is what lets a team run data curation on a CPU cluster, pretraining on a GPU box, and serving on yet another fleet, each with its own pinned environment. It is also what lets you swap `datatrove` for `dolma` or `vLLM` for `SGLang` without touching the rest. When you design your own stack, resist the urge to fuse stages into one process for "efficiency" — the loose coupling is worth far more than the saved serialization.

## Key Takeaways

!!! key "Key Takeaways"

    - **One belt, two implementations.** Part XIV built Stack-100M by hand (`stacklm`) so every mechanism is transparent; Part XV rebuilds the same artifact with production libraries so you can ship it. The thesis: *hand-roll to learn, reach for the library to ship* — both halves are load-bearing.
    - **The crosswalk is the spine.** Each stage maps *from-scratch module → production library → Part-XIV chapter*: `datatrove`, HF `tokenizers`, `torchtitan`/`nanotron`, `TRL`, `llm-compressor`/`vLLM`/`llama.cpp`, and `lm-evaluation-harness`.
    - **The seams are files in standard formats** (JSONL/Arrow, `tokenizer.json`, safetensors, an OpenAI-compatible API), which is exactly why you can swap any single library without rewriting its neighbors.
    - **The library earns its keep differently per stage.** At pretraining it earns *dollars* through MFU (a naive loop at ~0.20 vs ~0.45 MFU can double a $50 run); at data/serving it earns *correctness and throughput*; at scaling-laws and the agent loop it earns little, so keep those from-scratch.
    - **Not every stage has "a library."** Scaling-law fitting, the ReAct agent, and eval glue are genuinely bespoke; the honest move is to say so rather than invent a dependency.
    - **Version drift is the real enemy.** Pin versions, freeze the *resolved* set next to the checkpoint, lean on each library's stable API spine, and read the upstream example at your installed version's git tag — not `main`.
    - **When a snippet breaks, an argument moved, not the concept.** The mechanism you learned from scratch is what makes the renamed flag findable in thirty seconds.

## Further reading

- Penedo et al., *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale* (Hugging Face, 2024) — the data recipe Stack-100M borrows, and the motivation for `datatrove`.
- Karpathy, *nanoGPT* and *llm.c* — the from-scratch lineage this whole capstone updates; read them to feel how much the from-scratch/library boundary has moved since 2024.
- Wolf et al., *Transformers: State-of-the-Art Natural Language Processing* (Hugging Face, 2020) — the library and the safetensors/`PretrainedModel` conventions that define most of our seams.
- *torchtitan* (PyTorch team) and *nanotron* (Hugging Face) repositories — PyTorch-native distributed pretraining with FSDP2; the primary trainers of Ch 15.4.
- von Werra et al., *TRL: Transformer Reinforcement Learning* (Hugging Face) repository — the SFT/DPO/GRPO trainers of Ch 15.5; read `examples/scripts/` at your pinned tag.
- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM, 2023) — the serving engine of Ch 15.6.
- *lm-evaluation-harness* (EleutherAI) repository — the standard evaluation harness that gates every checkpoint in this part.
- Frantar et al., *GPTQ*, and Lin et al., *AWQ* — the post-training quantization methods behind `llm-compressor` and `AutoAWQ`.
