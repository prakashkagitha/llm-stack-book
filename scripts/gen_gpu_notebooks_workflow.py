#!/usr/bin/env python3
"""Emit a Workflow that AUTHORS the GPU-tier notebooks (draft: Sonnet 5 -> verify: Opus).

Each agent writes a jupytext 'percent'-format source to notebooks-gpu/src/<part>__<slug>.py.
Notebooks are AUTHORED for 1x H100 (80GB) or 2x A100 and NOT executed here; expected
throughput/MFU/speedup numbers are documented as rough orders of magnitude (never fabricated).
scripts/make_gpu_notebooks.py then converts sources -> .ipynb.

Usage: python3 scripts/gen_gpu_notebooks_workflow.py [--ids slug1 slug2 ...]
"""
import argparse, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = "https://prakashkagitha.github.io/llm-stack-book/"

# key = <part>__<slug> (filename); chapter = site chapter path; hw target; brief.
SPECS = [
 {"key":"04-kernels-efficiency__flash-attention-benchmark","chapter":"04-kernels-efficiency/02-flash-attention-1","hw":"1x H100 80GB","title":"FlashAttention vs Naive Attention: A GPU Benchmark",
  "brief":"Benchmark scaled-dot-product attention on the GPU: a naive materialized-softmax implementation vs torch.nn.functional.scaled_dot_product_attention under each backend (math, mem-efficient, and the FLASH backend via torch.nn.attention.sdpa_kernel), in bf16, causal. Sweep sequence length (e.g. 512..16384) at fixed batch/heads/head_dim; measure median latency (torch.cuda.Event) and peak memory (torch.cuda.max_memory_allocated); show the naive path OOMs / blows up quadratically in memory while flash stays linear. Optionally show the flash-attn package path as a commented alternative. Document expected shape of results (flash ~2-4x faster at long seq, dramatically less memory) as orders of magnitude."},
 {"key":"04-kernels-efficiency__triton-fused-kernel","chapter":"04-kernels-efficiency/04-triton-kernels","hw":"1x H100 80GB","title":"Writing a Fused Triton Kernel (Fused Softmax) and Benchmarking It",
  "brief":"Write a real Triton kernel (a numerically-stable fused row softmax, or fused RMSNorm) with @triton.jit, a grid, program_id, block pointers/masking, and tl.load/tl.store. Verify correctness against torch.softmax (torch.allclose). Benchmark vs the eager torch op and torch.compile across row widths; explain the memory-bound argument (one kernel launch, one HBM round trip) and occupancy/BLOCK_SIZE. triton is a standard torch dependency. Document expected speedup as an order of magnitude."},
 {"key":"04-kernels-efficiency__torch-compile-speedup","chapter":"04-kernels-efficiency/09-compilers-fusion","hw":"1x H100 80GB","title":"torch.compile & CUDA Graphs: Measuring Real Speedups",
  "brief":"Take a small transformer block (or GPTBlock) in bf16 on GPU and measure eager vs torch.compile(model, mode='max-autotune') forward+backward latency after proper warmup (compile is lazy; exclude the first few steps). Show the fusion/graph-break story (torch._dynamo.explain or TORCH_LOGS), the effect of dynamic vs static shapes, and CUDA-graph capture for the decode step. Document expected 1.2-2x speedup as an order of magnitude and note where compile helps most (many small ops)."},
 {"key":"04-kernels-efficiency__int4-int8-quantization","chapter":"04-kernels-efficiency/08-quantization-formats-qat","hw":"1x H100 80GB","title":"INT8 & NF4 Quantization: Memory and Latency on GPU",
  "brief":"Quantize a small from-scratch transformer (or a small HF model) to int8 and 4-bit NF4 with bitsandbytes (pip install bitsandbytes; note torchao as an alternative), measuring the memory footprint (bytes/param: 16 -> 8 -> ~4.x) and decode latency vs bf16. Show round-to-nearest vs the value of calibration (GPTQ/AWQ named, not reimplemented). Include a correctness sanity check (perplexity / logit MSE stays close). Document expected ~2x/~4x memory reduction and the accuracy-vs-size tradeoff; keep numbers hedged."},
 {"key":"04-kernels-efficiency__memory-efficient-training","chapter":"04-kernels-efficiency/10-memory-efficient-training","hw":"1x H100 80GB","title":"Fitting a Bigger Model: Activation Checkpointing + 8-bit Optimizer",
  "brief":"On one GPU, measure the memory breakdown (params, grads, optimizer state, activations) while training a small transformer, then show three levers reducing peak memory: gradient/activation checkpointing (torch.utils.checkpoint), bf16 mixed precision, and an 8-bit AdamW (bitsandbytes.optim.Adam8bit; note the 16 bytes/param Adam state math). Print torch.cuda.max_memory_allocated before/after each. Document expected memory savings (checkpointing trades ~30% compute for a large activation-memory cut) as orders of magnitude."},
 {"key":"03-pretraining__bf16-vs-fp8-throughput","chapter":"03-pretraining/08-mixed-precision-fp8","hw":"1x H100 80GB","title":"bf16 vs FP8 Matmul Throughput on Hopper",
  "brief":"On an H100, compare large matmul / linear-layer throughput in bf16 vs FP8. Use torch._scaled_mm with per-tensor scales (E4M3) if available, else clearly show the transformer_engine.pytorch path (te.Linear, fp8_autocast with a DelayedScaling recipe) as the recommended route; explain dynamic scaling and why FP8 needs it (3-bit mantissa). Measure TFLOP/s achieved and compare to the roofline. Be careful and correct with the FP8 API; document expected ~1.3-1.6x speedup on Hopper as an order of magnitude, hedged."},
 {"key":"03-pretraining__train-small-gpt-mfu","chapter":"03-pretraining/17-end-to-end-pretrain-recipe","hw":"1x H100 80GB","title":"Train a Small GPT on One H100: Loss Curve, Throughput & MFU",
  "brief":"A complete single-GPU pretraining loop for a small (~30-120M) GPT on a tiny real or synthetic token stream: bf16 autocast, AdamW, cosine/WSD schedule, gradient clipping, gradient accumulation to a target token batch, and — the point — measure tokens/s and MFU (model FLOPs util = 6*N*tokens_per_sec / peak_bf16_FLOPs, with H100 ~990 TFLOP/s bf16 dense). Plot the loss curve (matplotlib). Document the expected MFU band (~35-50% is healthy on one GPU) and loss trend as hedged orders of magnitude. Cross-link the scaling-laws chapter."},
 {"key":"03-pretraining__optimizers-wallclock","chapter":"03-pretraining/09-optimizers","hw":"1x H100 80GB","title":"AdamW vs Lion vs Muon: Wall-Clock & Memory on GPU",
  "brief":"On one GPU, train the same small model with AdamW, Lion (sign-based, 8 bytes/param state), and Muon (Newton-Schulz orthogonalized momentum for 2D weights + AdamW for the rest). Compare optimizer-state memory (12/8/... bytes/param), per-step wall-clock (Muon's NS iterations add compute), and loss-vs-step / loss-vs-wallclock. Reference the real papers (Kingma&Ba, Chen et al. Lion, Jordan et al. Muon). Document expected differences qualitatively; no fabricated final numbers."},
 {"key":"14-capstone__stack100m-single-gpu-h100","chapter":"14-capstone/07-pretraining-run","hw":"1x H100 80GB","title":"Capstone: Pretrain Stack-100M on a Single H100 (Real Run)",
  "brief":"THE flagship GPU notebook. Using the capstone package `stacklm` (clone the repo / pip install -e capstone; import StackConfig, Stack100M, the WSD schedule, Muon+AdamW hybrid, the packed data loader and synthetic corpus), run a REAL single-H100 pretraining of the full Stack-100M config (d_model=512, 30 layers, GQA, ~101M params) for a bounded number of steps: bf16, grad-accum to ~0.5M-token batches, checkpoint/resume, and live tokens/s + MFU + loss logging with a plotted curve, then a sample generation. Give the full-run extrapolation (the ~20B-token / ~$100 / 15-25 H100-hr framing from the capstone) and the exact command. Document expected MFU/throughput/loss as hedged orders of magnitude. This is the 'you can actually train it' proof; keep it faithful to capstone/PLAN.md."},
 {"key":"07-inference-serving__speculative-decoding-speedup","chapter":"07-inference-serving/06-speculative-decoding","hw":"1x H100 80GB","title":"Speculative Decoding: Measuring the Real Wall-Clock Speedup",
  "brief":"With two real small HF models on one H100 (a larger target, e.g. a ~1-1.5B instruct model, and a tiny draft, e.g. a ~70-160M model from the same tokenizer family; note the weight download + license), implement (or use transformers' assisted_generation / assistant_model=) speculative decoding and measure end-to-end tokens/s vs plain autoregressive decode at greedy and low temperature. Print the acceptance rate and explain how it (and the target/draft latency ratio) governs the speedup. Document the expected ~1.5-2.5x band as an order of magnitude; be explicit that it needs network to fetch weights."},
 {"key":"03-pretraining__ddp-two-gpu-scaling","chapter":"03-pretraining/05-distributed-data-parallel","hw":"2x A100","title":"Data-Parallel Training Across 2 GPUs (DDP) & Scaling Efficiency",
  "brief":"Real DistributedDataParallel training on 2 GPUs. Because a notebook can't spawn torchrun inline, use the standard idiom: %%writefile a self-contained ddp_train.py (init_process_group nccl, DistributedSampler, DDP-wrap, per-rank logging, all_reduce of loss) then launch it with !torchrun --nproc_per_node=2 ddp_train.py. Measure single-GPU vs 2-GPU tokens/s to show scaling efficiency (<2x due to gradient all-reduce), explain gradient bucketing / overlap and the global-vs-per-GPU batch size. Document expected ~1.7-1.9x scaling as an order of magnitude."},
 {"key":"03-pretraining__fsdp-two-gpu-sharding","chapter":"03-pretraining/07-megatron-deepspeed","hw":"2x A100","title":"Sharding a Model with FSDP (ZeRO) Across 2 GPUs",
  "brief":"Show ZeRO-style sharding with PyTorch FSDP (fully_shard / FullyShardedDataParallel) on 2 GPUs via the %%writefile + !torchrun idiom. Print per-GPU memory with and without sharding to demonstrate parameters/gradients/optimizer-state are split (ZeRO-1/2/3 mapping), use an auto-wrap policy on transformer blocks, mixed precision, and a sharded checkpoint save. Explain the communication (all-gather params in fwd, reduce-scatter grads in bwd) and when FSDP beats DDP (model too big for one GPU). Document expected per-GPU memory drop as an order of magnitude; reference DeepSpeed ZeRO (Rajbhandari et al.)."},
 {"key":"03-pretraining__tensor-parallel-from-scratch","chapter":"03-pretraining/06-distributed-model-parallel","hw":"2x A100","title":"Tensor Parallelism From Scratch: Splitting a Layer Across 2 GPUs",
  "brief":"Build a minimal tensor-parallel MLP/attention across 2 GPUs (Megatron-style column-parallel then row-parallel linear with an all_reduce), via %%writefile + !torchrun. Show the forward all_reduce and the correctness check that the TP output matches the single-GPU output (allclose). Explain the communication volume and why TP is intra-node (NVLink-bound), contrasting with DDP (data) and FSDP (sharding). Reference Megatron-LM (Shoeybi et al.). Keep it a correct toy that actually runs on 2 GPUs."},
 {"key":"07-inference-serving__multi-gpu-tp-inference","chapter":"07-inference-serving/11-multi-gpu-inference","hw":"2x A100","title":"Multi-GPU Inference: Tensor-Parallel Serving of a Model Too Big for One GPU",
  "brief":"Demonstrate splitting inference across 2 GPUs. Either (a) a from-scratch tensor-parallel forward of a toy model (column/row-parallel + all_reduce) via %%writefile + !torchrun, measuring per-GPU memory and decode latency; and/or (b) show the production path with a commented vLLM snippet (LLM(model=..., tensor_parallel_size=2)) explaining KV-cache and weight sharding. Explain when TP inference is needed (weights+KV exceed one GPU) vs pipeline/replication, and the NVLink latency cost per token. Document expectations qualitatively."},
]


AUTHOR = r"""You are authoring ONE GPU-tier notebook for a definitive LLM-systems textbook. It targets real datacenter GPUs and must contain correct, runnable code — but you will NOT execute it (no GPU here). Never fabricate specific benchmark numbers; give expected results as hedged orders of magnitude ("roughly", "on the order of", "you should see ~X").

Notebook: {TITLE}
Hardware target: {HW}
Source chapter (link it): {CHAPTER_URL}

WRITE the notebook as a jupytext 'percent'-format Python file to EXACTLY:
  {SRC_PATH}
Cell format (render-critical):
  - `# %% [markdown]` starts a markdown cell; put every markdown line as a comment ('# ...').
  - `# %%` starts a code cell; write literal Python below it.
Structure:
  1) A markdown title cell: `# {TITLE}` then a blockquote hardware banner (e.g. `> **Hardware:** {HW}. Runtime: a few minutes. Not executed in the book — run it to get your own numbers.`), one sentence on what the reader will learn, and a link: `See [the chapter]({CHAPTER_URL}) for the full explanation.`
  2) A code cell with the pip install(s) needed (use `%pip install ...`; torch is preinstalled; add flash-attn/bitsandbytes/transformers/etc. ONLY if the notebook uses them), and imports + a device/bf16 setup + seed.
  3) Alternating markdown (explain the step + the expected result) and code cells implementing the experiment below.
  4) A final markdown cell: "What you should see" (hedged expected numbers) + 2-4 key takeaways + a next-step pointer.

EXPERIMENT TO IMPLEMENT (make it real, correct, and idiomatic for {HW}):
{BRIEF}

HARD RULES:
  - Code must be correct and actually runnable on {HW} as-is (correct shapes, dtypes, CUDA/bf16 usage, real library APIs). For multi-GPU, use the `%%writefile <script>.py` + `!torchrun --nproc_per_node=N <script>.py` idiom (a notebook cannot init a process group inline). Guard optional imports.
  - Measure with torch.cuda.Event (not time.time for GPU) and torch.cuda.max_memory_allocated; include warmup + torch.cuda.synchronize.
  - NO fabricated precise benchmark numbers, FLOP counts you didn't derive, dates, or citations. Name real papers/tools by author/title only.
  - Heavily comment the code. Keep it focused (roughly 8-16 cells).
Return a short JSON summary."""

VERIFY = r"""You are the Opus verifier for ONE GPU-tier notebook source (jupytext percent format) that targets {HW} and was authored WITHOUT execution. Make it correct and safe to ship, then rewrite it in place.

File: {SRC_PATH}
Chapter: {CHAPTER_URL}

READ the file, then verify hard and FIX in place (rewrite the whole file with the Write tool):
  - CORRECTNESS: every code cell must be runnable on {HW} as-is. Check tensor shapes/dtypes, CUDA/bf16 usage, and REAL library APIs (torch SDPA + torch.nn.attention.sdpa_kernel, triton jit/grid/tl.*, torch.compile, bitsandbytes Linear8bitLt/NF4 + Adam8bit, torch._scaled_mm / transformer_engine fp8_autocast, FSDP fully_shard/auto_wrap, DDP init_process_group/DistributedSampler, transformers assisted generation). Fix wrong/hallucinated API usage. Multi-GPU MUST use the %%writefile + !torchrun idiom, not inline process-group init.
  - MEASUREMENT: timing uses torch.cuda.Event + synchronize + warmup; memory uses max_memory_allocated. Fix if not.
  - HONESTY: no fabricated precise numbers/FLOPs/dates/citations; expected results are hedged orders of magnitude. Reduce any invented specifics to hedged language.
  - FORMAT: valid percent cells (# %% and # %% [markdown]); markdown cells are '#'-commented; title cell has the H1 + hardware banner + chapter link; a pip cell exists with correct deps; a final "what you should see" cell exists.
Rewrite the corrected file to EXACTLY {SRC_PATH}. Return a JSON verdict (issues_fixed, runnable, notes)."""

ASCHEMA = {"type":"object","additionalProperties":False,"required":["file","code_cells","summary"],
           "properties":{"file":{"type":"string"},"code_cells":{"type":"integer"},"summary":{"type":"string"}}}
VSCHEMA = {"type":"object","additionalProperties":False,"required":["file","runnable","notes"],
           "properties":{"file":{"type":"string"},"issues_fixed":{"type":"integer"},"runnable":{"type":"boolean"},"notes":{"type":"string"}}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "scripts", "wf_gpu_nb.js"))
    ap.add_argument("--ids", nargs="*", default=[])
    args = ap.parse_args()
    specs = [s for s in SPECS if not args.ids or s["key"] in args.ids]
    src_dir = os.path.join(ROOT, "notebooks-gpu", "src")

    jobs = []
    for s in specs:
        src_path = os.path.join(src_dir, s["key"] + ".py")
        ch_url = SITE + s["chapter"] + ".html"
        rep = {"{TITLE}": s["title"], "{HW}": s["hw"], "{CHAPTER_URL}": ch_url,
               "{SRC_PATH}": src_path, "{BRIEF}": s["brief"]}
        def fill(t):
            for k, v in rep.items():
                t = t.replace(k, v)
            return t
        jobs.append({"key": s["key"], "author": fill(AUTHOR), "verify": fill(VERIFY)})

    js = f"""export const meta = {{
  name: 'gpu-notebooks',
  description: 'Author {len(jobs)} GPU-tier notebooks (1xH100 / 2xA100): Sonnet draft -> Opus verify',
  phases: [{{ title: 'Author' }}, {{ title: 'Verify' }}],
}}
const JOBS = {json.dumps(jobs, ensure_ascii=True)};
const ASCHEMA = {json.dumps(ASCHEMA)};
const VSCHEMA = {json.dumps(VSCHEMA)};
log('Authoring ' + JOBS.length + ' GPU notebooks (draft -> Opus verify)…');
const results = await pipeline(
  JOBS,
  function (j) {{
    return agent(j.author, {{ label: 'author:' + j.key, phase: 'Author', model: 'sonnet', schema: ASCHEMA }})
      .then(function (r) {{ return j; }});
  }},
  async function (j) {{
    if (!j) return null;
    const v = await agent(j.verify, {{ label: 'verify:' + j.key, phase: 'Verify', model: 'opus', schema: VSCHEMA }});
    return {{ key: j.key, verify: v }};
  }}
);
const done = results.filter(Boolean);
log('GPU notebooks: ' + done.length + '/' + JOBS.length + ' authored+verified.');
return {{ total: JOBS.length, done: done.length, results: results }};
"""
    os.makedirs(src_dir, exist_ok=True)
    open(args.out, "w").write(js)
    print(f"Wrote {args.out}: {len(jobs)} notebooks")
    for s in specs:
        print(f"  [{s['hw']:10}] {s['key']}")


if __name__ == "__main__":
    main()
