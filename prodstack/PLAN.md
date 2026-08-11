# Part XV Spec — The Production Stack: Stack-100M With the Real Toolchain

> **Companion to Part XIV (the from-scratch capstone).** Part XIV built Stack-100M *by hand*
> (`stacklm`) so every mechanism is transparent. Part XV rebuilds the SAME journey with the
> **real open-source libraries a practitioner reaches for at work** — so the reader learns both
> the mechanism *and* the production tool, and can transfer directly to a real job.

**Guiding rule for every chapter:** for each stage, (1) name the real library and why it is the
standard choice in 2026, (2) show the actual, runnable command/code to do the stage with it,
(3) explicitly cross-link the Part-XIV chapter where we built that piece by hand ("we hand-rolled
this in Ch. 14.x; here is the library, and here is what it does for you that the from-scratch
version did not"), (4) be honest about version drift — pin versions, say flags move between
releases, point at the upstream example. Same Stack-100M target (config in `capstone/PLAN.md`).

Never fabricate benchmark numbers, dates, or APIs. Prefer showing the *mechanism* of the library
call; hedge any figure as "on the order of". Keep code copy-paste-runnable (note the GPU/network
it needs). This part is the answer to "great, I understand it from scratch — now what do people
actually use?"

## Chapters

- **15.1 — The Production Toolchain: A Map From Scratch to Shipping.** The whole-stack diagram:
  each layer, the from-scratch component (Part XIV) vs. the production library, and how they wire
  together end to end (data -> tokenizer -> pretrain -> post-train -> serve -> eval). The "you
  hand-roll to learn, you reach for the library to ship" thesis. A one-table crosswalk: stage ->
  our module -> the library -> the Part-XIV chapter. Sets up the rest of the part.

- **15.2 — Data at Scale: `datatrove` + Hugging Face `datasets`.** The FineWeb pipeline with
  `datatrove` (readers, URL/quality/language filters, MinHash dedup, tokenizing writer,
  `LocalPipelineExecutor` -> `SlurmPipelineExecutor`); `datasets` streaming; `nemo-curator` and
  `dolma` as alternatives. Mirrors Ch. 14.2 / 3.1 / 3.2. Show the real pipeline that produces the
  same packed shards `stacklm` consumes.

- **15.3 — Tokenizer Training: Hugging Face `tokenizers` + `sentencepiece`.** Train a real
  byte-level BPE with the `tokenizers` Rust trainer (`BpeTrainer`, `ByteLevel`), wrap in
  `PreTrainedTokenizerFast`, reserve chat/tool special tokens, export; the `sentencepiece` path and
  when to use it. Mirrors Ch. 14.3 / 2.1. Ties to the Tokenizer Playground tool.

- **15.4 — Pretraining With a Real Trainer: `torchtitan` / `nanotron` (FSDP2).** Configure and
  launch a real distributed pretraining run: the config file, FSDP2/tensor-parallel, activation
  checkpointing, the WSD schedule, checkpointing, and MFU logging — mapping every knob onto the
  hand-written loop of Ch. 14.7. `torchtitan` (PyTorch-native) as primary, `nanotron` and
  `Megatron-LM`/`DeepSpeed` as alternatives; `accelerate` for the single-GPU/small path. Cross-link
  Ch. 3.5-3.8, 14.6, 14.7.

- **15.5 — Post-Training With `TRL` (and the alignment-handbook).** SFT (`SFTTrainer`, chat
  template, `assistant_only_loss`, packing), DPO (`DPOTrainer`), and GRPO (`GRPOTrainer` + vLLM
  rollouts) with real configs, mapped onto the from-scratch SFT/DPO/GRPO of Ch. 14.9; the
  `alignment-handbook` recipes; `veRL`/`OpenRLHF` for scale; `peft`/`unsloth` for cheap adapters;
  `lm-evaluation-harness` gating. Cross-link Ch. 5.x, 6.x, 14.9.

- **15.6 — Serving, Quantizing & Evaluating: `vLLM` + `lm-eval-harness` + `llama.cpp`.** Serve the
  model with `vllm serve` (OpenAI-compatible API, continuous batching, the metrics), quantize with
  GPTQ/AWQ (`llm-compressor`/`AutoAWQ`) and to GGUF for `llama.cpp` CPU/laptop serving, and run the
  honest eval with `lm-evaluation-harness`. Maps onto Ch. 14.11 (int4 + laptop) and Ch. 7.x. Ends
  the part: the same Stack-100M, now shipped with the tools a team actually uses.
