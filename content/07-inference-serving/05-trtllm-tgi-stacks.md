# 7.5 TensorRT-LLM, TGI & Other Serving Stacks

The serving stack you choose determines whether your model runs at 10 tokens/second on a laptop or 10,000 tokens/second on a data-center cluster. The transformer forward pass is the same mathematics in both cases — what differs is how aggressively the serving runtime compiles kernels, manages memory, schedules requests, and exposes hardware capabilities. This chapter maps the complete landscape: NVIDIA's TensorRT-LLM for peak GPU throughput, HuggingFace's Text Generation Inference (TGI) for portable production deployments, and llama.cpp / Ollama / LMDeploy / MLC for the long tail of hardware and use cases.

Before diving in, make sure you are comfortable with how the prefill and decode phases work and why the KV cache is central to serving economics — see [The Anatomy of LLM Inference: Prefill, Decode & The KV Cache](../07-inference-serving/01-anatomy-inference.html). Continuous batching and in-flight request scheduling, covered in [Continuous Batching & Request Scheduling](../07-inference-serving/02-continuous-batching.html), underpin every production system we discuss here.

---

## The Performance Hierarchy

Before benchmarking any serving framework, it helps to understand the ceiling imposed by hardware. The roofline model (see [The Roofline Model & Performance Engineering](../04-kernels-efficiency/01-roofline-performance.html)) says decode throughput is bounded by memory bandwidth: each decode step reads the full model weight matrix for every generated token. For an A100-80 GB with ~2 TB/s of HBM bandwidth and a 70 B parameter model stored in FP16 (140 GB — so sharded across multiple GPUs), each token costs on the order of 70 GB of reads. That gives a raw ceiling of roughly 14 tokens/second per GPU — before batching amortizes those reads across requests.

$$
\text{tokens\_per\_sec} \leq \frac{\text{HBM bandwidth (bytes/s)}}{\text{model size (bytes per token read)}}
$$

Every framework we discuss is fighting to get close to this ceiling. The closer a system gets, the more it relies on compiled, fused, hardware-specific kernels — and the less portable it becomes. That tension is the through-line of this chapter.

{{fig:trtllm-perf-portability-spectrum}}

---

## TensorRT-LLM: The Compiled Engine Approach

### What TensorRT-LLM Is

TensorRT-LLM (TRTLLM) is NVIDIA's open-source library for building highly optimized TensorRT engines from transformer checkpoints. A TensorRT engine is a serialized, hardware-specific computation graph with all kernels pre-selected and fused at build time. Think of it as AOT (ahead-of-time) compilation for the GPU.

The key insight: PyTorch runs an interpreter that dispatches individual CUDA kernels at every step. TensorRT traces the full forward pass, applies a library of graph optimizations — layer fusion, constant folding, precision calibration — and emits a binary engine that the CUDA runtime can execute with minimal host overhead. For a model that runs millions of requests per day, eliminating Python-level dispatch overhead is material.

### Build and Run Pipeline

```bash
# Step 1: Convert a HuggingFace checkpoint to TensorRT-LLM format
# (here: Llama-2-7B in bfloat16, single GPU)
python tensorrt_llm/examples/llama/convert_checkpoint.py \
    --model_dir ./llama-2-7b-hf \
    --output_dir ./llama-2-7b-trtllm \
    --dtype bfloat16

# Step 2: Build the TensorRT engine
# max_batch_size and max_input_len are compile-time constants —
# choose them to match your production workload envelope.
trtllm-build \
    --checkpoint_dir ./llama-2-7b-trtllm \
    --output_dir ./llama-2-7b-engine \
    --max_batch_size 32 \
    --max_input_len 2048 \
    --max_output_len 512 \
    --use_inflight_batching \
    --paged_kv_cache enable \
    --gemm_plugin bfloat16 \
    --gpt_attention_plugin bfloat16

# Step 3: Run inference via the Python API
python -c "
import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner
import tensorrt as trt

runner = ModelRunner.from_dir('./llama-2-7b-engine')
# Tokenize and run — runner handles batching and KV cache internally
outputs = runner.generate(
    batch_input_ids=[[1, 2, 3, 4, 5]],
    max_new_tokens=50,
)
print(outputs)
"
```

The `--gemm_plugin` and `--gpt_attention_plugin` flags activate NVIDIA's hand-tuned GEMM and attention kernels, which are substantially faster than the TensorRT auto-scheduler can find on its own.

### In-Flight Batching in TensorRT-LLM

TensorRT-LLM implements in-flight batching (also called continuous batching) through its `Executor` API. New requests are inserted into the active batch at the token boundary — pausing requests that have finished, inserting new ones, and continuing generation — without ever stopping the GPU. This is the same principle as vLLM (see [vLLM: Architecture, PagedAttention & Internals](../07-inference-serving/03-vllm-internals.html)), but TensorRT-LLM's version is implemented inside a compiled TensorRT engine rather than in PyTorch.

```python
# Minimal TensorRT-LLM Executor example (C++ API wrapped in Python)
# This illustrates the async request/response model.

from tensorrt_llm.executor import GenerationExecutor, GenerationRequest
import asyncio

async def serve_requests():
    """
    The Executor runs a background thread that continuously feeds the engine.
    Requests are submitted as GenerationRequest objects and picked up
    at the next scheduling interval (default: every decode step).
    """
    executor = GenerationExecutor.create(
        engine_dir="./llama-2-7b-engine",
        executor_config={
            "max_beam_width": 1,
            "scheduler_policy": "guaranteed_no_evict",  # vs "max_utilization"
        }
    )

    # Submit two requests concurrently — they will be batched automatically
    req_a = GenerationRequest(
        input_token_ids=[1, 234, 567],
        max_new_tokens=100,
        streaming=True,
    )
    req_b = GenerationRequest(
        input_token_ids=[1, 890, 123, 456],
        max_new_tokens=50,
        streaming=True,
    )

    executor.submit(req_a)
    executor.submit(req_b)

    # Stream tokens as they arrive
    async for token in req_a.aiter_tokens():
        print(f"A: {token}", end=" ", flush=True)

asyncio.run(serve_requests())
```

### Paged KV Cache and Memory Management

TensorRT-LLM implements a paged KV cache that operates analogously to vLLM's PagedAttention: KV tensors are stored in fixed-size "blocks" (on the order of 16 or 32 tokens per block), and a block manager allocates/frees blocks as sequences grow and terminate. This means a server can hold many more concurrent sequences than naïve pre-allocation would allow.

The key parameter is `--kv_cache_free_gpu_mem_fraction` (default 0.9): the fraction of free GPU memory TensorRT-LLM may use for KV blocks after the model weights are loaded.

### Quantization in TensorRT-LLM

TensorRT-LLM has first-class support for INT8 weight-only quantization, INT8 SmoothQuant, FP8 (on H100/H200), and GPTQ/AWQ. These are configured at engine build time:

```bash
# FP8 engine for H100 — uses calibration data to determine per-tensor scales
trtllm-build \
    --checkpoint_dir ./llama-2-7b-trtllm \
    --output_dir ./llama-2-7b-fp8-engine \
    --strongly_typed \
    --use_fp8_context_fmha enable \
    --max_batch_size 64 \
    --max_input_len 4096 \
    --max_output_len 1024
```

Because quantization scales are baked into the engine at build time, there is zero overhead at serving time — the quantized GEMM kernels simply execute with pre-computed scales.

For a deeper treatment of the quantization formats, see [Quantization II: INT4/INT8/FP8, GGUF, bitsandbytes & QAT](../04-kernels-efficiency/08-quantization-formats-qat.html).

### Multi-GPU Tensor Parallelism

TensorRT-LLM supports tensor parallelism and pipeline parallelism at build time. Specify `--tp_size 4` (for 4-way tensor parallel) at the `convert_checkpoint.py` and `trtllm-build` stages; the library handles the all-reduce communication using NCCL. At inference time, the executor launches one process per GPU and co-ordinates automatically. See [Multi-GPU & Multi-Node Inference](../07-inference-serving/11-multi-gpu-inference.html) for the broader parallelism strategies.

### The Triton Inference Server Integration

NVIDIA's standard production path is to serve TensorRT-LLM engines via Triton Inference Server using the `tensorrtllm_backend`. Triton handles HTTP/gRPC front-end, dynamic batching at the server layer, health checks, and metrics. TensorRT-LLM handles the GPU-side scheduling and execution.

{{fig:trtllm-triton-serving-pipeline}}

### TensorRT-LLM: What You Give Up

The engine is platform-specific and must be rebuilt for each GPU SKU (A100 vs H100 vs L40S). Build times for large models can be 30–60 minutes. The `max_batch_size` and `max_input_len` are fixed at build time — you cannot exceed them at runtime. If your traffic suddenly shifts to very long prompts, you need a different engine. These constraints mean TensorRT-LLM is most suitable for dedicated GPU fleets with predictable workloads.

---

## Text Generation Inference (TGI)

### Architecture Overview

Hugging Face's Text Generation Inference is a Rust-based server with a Python model-runner process. The Rust front-end handles HTTP/gRPC routing and request queuing; the Python process runs the model using PyTorch + custom CUDA kernels. Importantly, TGI ships its own custom attention kernels (including a FlashAttention-based implementation) and its own continuous batching scheduler.

{{fig:tgi-router-modelserver-architecture}}

### Running TGI

```bash
# Launch TGI with Docker — NVIDIA GPU required for full throughput
docker run --gpus all \
    -p 8080:80 \
    -v $PWD/model_cache:/data \
    ghcr.io/huggingface/text-generation-inference:2.4 \
    --model-id meta-llama/Llama-3-8B-Instruct \
    --quantize bitsandbytes-nf4 \
    --max-input-tokens 4096 \
    --max-total-tokens 6144 \
    --max-batch-prefill-tokens 16384 \
    --num-shard 1  # tensor parallel degree

# Send a request — TGI implements the Messages API compatible with OpenAI
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tgi",
    "messages": [{"role": "user", "content": "Explain KV cache in one paragraph."}],
    "max_tokens": 200,
    "stream": true
  }'
```

TGI exposes a `/generate` endpoint (its native API), a `/v1/chat/completions` OpenAI-compatible endpoint, and a `/v1/completions` endpoint. The OpenAI-compatible surface makes it easy to drop TGI behind any client that speaks the OpenAI protocol.

### TGI's Continuous Batching Scheduler

TGI implements a "token budget" scheduler: each request is assigned a `max_total_tokens` budget, and requests are grouped into batches where the total token budget fits within the KV cache. The scheduler preempts requests if memory pressure grows (evicting KV blocks and later recomputing them). The Rust router enforces these constraints before passing requests to the Python runner, which means request validation and queueing happen with very low latency.

```python
# Simplified pseudocode illustrating TGI's waiting queue logic
# Real implementation is in Rust + a Python model server side

class TGIScheduler:
    def __init__(self, max_batch_total_tokens: int):
        self.max_batch_total_tokens = max_batch_total_tokens
        self.waiting: list[Request] = []
        self.running: list[Request] = []

    def schedule(self) -> list[Request]:
        """
        Called every decode step. Fill the running batch up to the token budget.
        New requests are added from waiting if budget permits.
        Running requests keep their slot as long as they haven't finished.
        """
        budget_used = sum(r.current_length for r in self.running)
        for req in list(self.waiting):
            needed = req.max_total_tokens  # pre-allocated worst case
            if budget_used + needed <= self.max_batch_total_tokens:
                self.running.append(req)
                self.waiting.remove(req)
                budget_used += needed
        return self.running
```

### TGI vs TensorRT-LLM: Key Differences

| Dimension | TGI | TensorRT-LLM |
|---|---|---|
| Build step | None (model loaded at startup) | Engine must be compiled (30–60 min for large models) |
| GPU portability | Any CUDA GPU supported by PyTorch | Specific to GPU SKU; must rebuild per SKU |
| Custom kernels | FlashAttention, custom GEMM from Torch | Full engine compilation, per-op tuning |
| Peak throughput | ~80–90% of hardware ceiling (typical) | ~95–100% (typical on target hardware) |
| Model support | Any HF model | Supported architectures only |
| Quantization | bitsandbytes NF4, GPTQ, AWQ | INT8, FP8, GPTQ, AWQ (baked at build time) |
| Streaming | Native SSE | Via Triton streaming protocol |
| OpenAI API | Yes | Via triton proxy |

---

## llama.cpp: The CPU/Edge Champion

### What llama.cpp Does

Georgi Gerganov's llama.cpp (2023) proved that a quantized transformer can run usefully fast on commodity hardware — a laptop CPU, an Apple M-series chip, or a consumer GPU. The project is a single C++ codebase with no external deep-learning dependencies that achieves high throughput through:

1. **GGUF quantization**: 2-bit through 8-bit per-channel integer quantization with mixed precision (Q4_K_M, Q5_K_M, etc.) that keeps key tensors at higher precision.
2. **Highly optimized BLAS routines**: AVX2/AVX-512 vectorized GEMM for x86, ARM NEON/SVE for ARM, Metal for Apple Silicon, and CUDA for NVIDIA GPUs — all in the same binary via compile-time backends.
3. **mmap model loading**: The model file is memory-mapped, so the OS controls paging. On a machine with enough RAM, the model loads in seconds; with limited RAM, the OS pages in only the layers currently needed.

### GGUF Format

GGUF (GPT-Generated Unified Format) stores model weights, tokenizer data, and metadata in a single binary file. Quantized weights use a block quantization scheme: weights are grouped into blocks of 32 values, each block stores a float32 scale factor and the quantized integers. For Q4_K_M, 4-bit integers are stored for most weights with 6-bit integers for sensitive layers (attention projections, etc.):

$$
\hat{w}_i = \text{scale} \times q_i, \quad q_i \in \{-8, -7, \ldots, 7\}
$$

where scale is a per-32-element float32. The storage cost for Q4_K_M is approximately $4.5$ bits per weight after accounting for the scale overhead.

```python
# Estimate GGUF model file size from parameter count
def gguf_size_gb(params_billions: float, bits_per_weight: float = 4.5) -> float:
    """
    Rough size estimate for a GGUF-quantized LLM.
    bits_per_weight: 4.5 for Q4_K_M, 5.5 for Q5_K_M, 8.5 for Q8_0
    """
    params = params_billions * 1e9
    bytes_per_weight = bits_per_weight / 8.0
    # Embeddings and norms are typically kept in FP16, ~5% of params
    emb_fraction = 0.05
    size_bytes = (
        params * (1 - emb_fraction) * bytes_per_weight
        + params * emb_fraction * 2.0  # FP16
    )
    return size_bytes / (1024**3)

# Llama-3-70B at Q4_K_M:
print(f"70B Q4_K_M: {gguf_size_gb(70, 4.5):.1f} GB")   # ~38 GB
print(f"70B Q5_K_M: {gguf_size_gb(70, 5.5):.1f} GB")   # ~46 GB
print(f"13B Q4_K_M: {gguf_size_gb(13, 4.5):.1f} GB")   # ~7 GB
```

### Ollama: A Developer-Friendly Frontend

Ollama wraps llama.cpp in a REST server with a model registry, automatic download, and a simple CLI. It is the fastest path from "zero" to running a local LLM:

```bash
# Install and run Llama-3-8B locally
ollama pull llama3.1:8b
ollama run llama3.1:8b "Explain transformer attention in two sentences."

# OpenAI-compatible API (on port 11434 by default)
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1:8b",
    "messages": [{"role": "user", "content": "What is speculative decoding?"}]
  }'
```

Ollama automatically detects available hardware (CUDA, Metal, CPU) and offloads as many layers as possible to GPU. The key flag is `--num-gpu` (or `OLLAMA_NUM_GPU` in the environment), which sets how many transformer layers run on GPU versus CPU.

### Performance Profile

On Apple M3 Max with 128 GB of unified memory, a Llama-3-70B Q4_K_M model (≈38 GB) fits entirely in unified memory and typically achieves 20–30 tokens/second — genuinely useful for local development. On a consumer NVIDIA 4090 (24 GB), the 13B Q4_K_M fits fully in VRAM and achieves 80–100 tokens/second. For anything requiring production throughput (hundreds of tokens/second, many concurrent users), llama.cpp is not the right tool.

---

## LMDeploy: Efficient Serving with TurboMind

LMDeploy, developed by Shanghai AI Laboratory (the InternLM team), provides a high-throughput serving stack that sits between TGI (easy to use) and TensorRT-LLM (maximum performance). Its core inference engine is TurboMind, a C++/CUDA implementation with:

- Custom blocked KV cache (similar in concept to PagedAttention)
- Continuous batching
- INT4 and INT8 quantization via AWQ
- A Python/gRPC serving layer

```bash
# Install and convert a model to TurboMind format
pip install lmdeploy

# Convert HF checkpoint to TurboMind internal format
lmdeploy convert llama3 /path/to/llama-3-8b-hf \
    --dst_path ./llama3-turbomind

# Launch the serving API
lmdeploy serve api_server ./llama3-turbomind \
    --server_port 23333 \
    --tp 1 \
    --cache_max_entry_count 0.8

# Or use the Python API directly
python -c "
from lmdeploy import pipeline, TurbomindEngineConfig

cfg = TurbomindEngineConfig(tp=1, cache_max_entry_count=0.8)
pipe = pipeline('./llama3-turbomind', backend_config=cfg)
response = pipe(['Hello! Explain TurboMind in one sentence.'])
print(response[0].text)
"
```

LMDeploy also supports the PyTorch backend (`--backend pytorch`) which is more portable but slower, mirroring TGI's design philosophy. The TurboMind backend is specifically optimized for NVIDIA GPUs and resembles TensorRT-LLM in its use of custom CUDA kernels — but without the full AOT compilation step, making model loading much faster.

---

## MLC-LLM: Universal Compilation via TVM

MLC-LLM (Machine Learning Compilation for LLMs), from the MLC team led by Tianqi Chen, applies Apache TVM's compilation machinery to LLM serving. The goal is one codebase that runs on essentially any hardware: NVIDIA CUDA, AMD ROCm, Apple Metal, Intel Vulkan, ARM CPU, and even WebGPU (in the browser).

### How MLC Differs

Where TensorRT-LLM compiles to a TensorRT engine (NVIDIA-specific), MLC compiles to TVM's intermediate representation (TIR), then lowers TIR to hardware-specific code via TVM's code generation backends. The compilation includes automatic schedule search (AutoTIR) that tunes GEMM tile sizes and memory layouts for the target device.

```python
# Compile a model for Metal (Apple Silicon) using MLC
import mlc_llm
from mlc_llm import MLCEngine

# mlc_llm.build() emits a compiled library (.dylib on macOS, .so on Linux)
# This step is slow once; subsequent loads are fast.
mlc_llm.build(
    model="HF://meta-llama/Llama-3.2-3B-Instruct",
    target="apple/m3-gpu",   # or "cuda", "rocm", "vulkan", "webgpu"
    quantization="q4f16_1",  # 4-bit weights, FP16 activations
)

# Run inference — identical Python API regardless of hardware backend
engine = MLCEngine("./dist/Llama-3.2-3B-Instruct-q4f16_1-MLC")
response = engine.chat.completions.create(
    messages=[{"role": "user", "content": "What is PagedAttention?"}],
    max_tokens=100,
)
print(response.choices[0].message.content)
```

MLC is the right choice when you need to deploy to heterogeneous hardware (a fleet with a mix of NVIDIA, AMD, and Apple GPUs), or when you need to run in a browser via WebLLM (which compiles to WebGPU/WASM). The trade-off is that compilation is slow and the tuned performance on any single GPU is typically below TensorRT-LLM.

---

## The Tradeoff Space: A Practitioner's Framework

!!! example "Worked example: memory budget for concurrent requests"
    Suppose you are serving Llama-3-70B on 4× A100-80 GB (320 GB total HBM).

    **Model weights (BF16):** $70 \times 10^9 \times 2\ \text{bytes} = 140\ \text{GB}$

    **Remaining for KV cache:** $320 - 140 = 180\ \text{GB}$

    **KV cache per token (per layer):** For Llama-3-70B, there are 80 layers, GQA with 8 KV heads, head dimension 128. Each KV entry in BF16:
    $$2 \times 8 \times 128 \times 2\ \text{bytes} = 4096\ \text{bytes} = 4\ \text{KB per token per layer}$$
    Total per token across all layers: $80 \times 4\ \text{KB} = 320\ \text{KB/token}$

    **Maximum concurrent tokens:** $180\ \text{GB} / 320\ \text{KB} \approx 562{,}500\ \text{tokens}$

    If your average context window is 2048 tokens, you can hold about **275 concurrent sequences** in memory before the system must evict KV blocks. TensorRT-LLM's `guaranteed_no_evict` scheduler will refuse new requests once this limit is hit; `max_utilization` will evict-and-recompute some requests to admit more.

    With INT8 KV cache quantization (cutting per-token cost in half), you'd reach ~550 concurrent sequences — a meaningful gain for high-fan-out workloads.

### Decision Matrix

| Use Case | Recommended Stack |
|---|---|
| Production NVIDIA fleet, SLA-critical | TensorRT-LLM + Triton |
| Cloud deployment, mixed GPU fleet | TGI or vLLM |
| On-premise single-node A100/H100 | vLLM or LMDeploy |
| Apple Silicon local / developer | Ollama (llama.cpp) |
| Consumer NVIDIA GPU, developer | Ollama or llama.cpp directly |
| Heterogeneous or browser deployment | MLC-LLM / WebLLM |
| RL rollout engine (speed is key) | vLLM or TensorRT-LLM |
| Low-latency streaming chat | TGI (good streaming UX) |

### Portability vs Peak Performance: The Core Tension

The fundamental constraint is this: the more a runtime commits to a specific hardware target, the more aggressive its optimizations can be. TensorRT-LLM achieves peak performance by:
1. Selecting the exact GEMM algorithm for each matrix shape at build time (profile-guided kernel selection)
2. Fusing operations that PyTorch would run as separate kernels (e.g., attention + residual + layer norm in one pass)
3. Emitting machine code rather than dispatching at runtime

Every step away from that specificity costs performance. TGI's PyTorch-based model runner dispatches CUDA kernels at runtime (some overhead) but works on any HuggingFace model on any CUDA GPU. llama.cpp's AVX2 kernels work on CPUs but obviously cannot exploit tensor cores.

The quantitative gap is real but often overstated in marketing: well-tuned vLLM or TGI on A100s typically reaches 85–95% of TensorRT-LLM's throughput at matched batch sizes, while being far simpler to deploy and maintain. The remaining 5–15% matters at scale (thousands of dollars per day) but is often dominated by other system costs (networking, load balancing, tokenization) at moderate scale.

---

## Kernel Highlights: What Makes These Engines Fast

### Fused Attention Kernels

All production serving stacks today incorporate FlashAttention or a variant. The key insight (Dao et al., 2022) is that the standard attention computation is memory-bandwidth bound, not compute bound, and by tiling the Q/K/V matrices to fit in on-chip SRAM, we avoid round-trips to HBM. See [FlashAttention I: IO-Awareness & The Online Softmax](../04-kernels-efficiency/02-flash-attention-1.html) for the full derivation.

For paged KV caches, standard FlashAttention (which assumes contiguous key/value tensors) must be modified. TensorRT-LLM, vLLM, and TGI all ship custom "paged FlashAttention" kernels that index into the block table at each attention step.

```cpp
// Simplified pseudocode for paged attention kernel (CUDA C++)
// Each thread block handles one query head for one sequence in the batch.

__global__ void paged_attention_kernel(
    float* output,           // [num_seqs, num_heads, head_dim]
    const float* queries,    // [num_seqs, num_heads, head_dim]
    const float** kv_blocks, // array of pointers to KV blocks
    const int* block_table,  // [num_seqs, max_blocks_per_seq]
    int head_dim,
    int block_size,          // tokens per KV block (e.g., 16)
    int seq_len
) {
    int seq_id = blockIdx.x;
    int head_id = blockIdx.y;

    // Accumulator for online softmax
    float acc[HEAD_DIM] = {0.0f};
    float max_score = -1e9f;
    float sum_exp = 0.0f;

    // Iterate over KV blocks for this sequence
    for (int block_idx = 0; block_idx * block_size < seq_len; ++block_idx) {
        int physical_block = block_table[seq_id * MAX_BLOCKS + block_idx];
        // Load K from this block, compute QK^T, update online softmax...
        // (full implementation follows standard online softmax pattern)
    }

    // Write final attended value to output
    // ...
}
```

For a full treatment of how PagedAttention fits into the memory manager, see [PagedAttention & KV-Cache Memory Management](../04-kernels-efficiency/06-paged-attention-kv.html).

### GEMM Kernel Tuning

Decode-phase matrix multiplications are "tall-and-skinny": the weight matrix is large (e.g., $[4096 \times 16384]$ for an FFN layer), but the activation tensor is only $[\text{batch\_size} \times 4096]$ — often with batch size below 32 during decode. This shape is very different from training GEMM, where batch × sequence length gives a "fat" activation matrix.

TensorRT-LLM uses a profiling pass during engine build to time dozens of cuBLAS/cuBLASLt configurations against your specific shapes and selects the best one. This profile-guided selection is a major source of its throughput advantage over frameworks that use PyTorch's general-purpose cuBLAS dispatch.

### KV Cache Quantization

Beyond paging, modern runtimes also quantize the KV cache itself. INT8 KV cache is now standard in TensorRT-LLM and supported in TGI and LMDeploy. The quantization is applied per-head-per-token with a float scale factor:

$$
\text{KV}_{\text{int8}} = \operatorname{clip}\!\left(\operatorname{round}\!\left(\frac{\text{KV}_{\text{fp16}}}{\text{scale}}\right), -128, 127\right)
$$

This halves KV cache memory at the cost of a small accuracy reduction (typically under 0.5 pp on standard benchmarks for INT8).

---

## Structured Output and Constrained Generation

All major serving stacks support constrained generation — forcing the model to produce valid JSON, follow a regex, or conform to a grammar. The mechanisms differ:

- **TGI** uses Outlines (a regex/grammar-to-finite-state-automaton library) to mask logits during sampling to only valid next tokens.
- **vLLM** integrates Outlines similarly.
- **TensorRT-LLM** requires a custom logit processor passed to the executor.
- **llama.cpp** has built-in GBNF (GGML BNF) grammar support.

For a full treatment of constrained generation mechanics, see [Structured & Constrained Generation](../07-inference-serving/10-structured-generation.html).

---

!!! interview "Interview Corner"
    **Q:** A hiring manager asks: "We're building a customer-support chatbot that needs to serve a fine-tuned Llama-3-70B model at under 200 ms time-to-first-token (TTFT) and under 20 ms per token (inter-token latency, ITL), with peak traffic of 500 concurrent users. We have a cluster of 8× A100-80 GB nodes. Which serving stack would you choose, and what are the key configuration decisions?"

    **A:** I would choose TensorRT-LLM backed by NVIDIA Triton, with the following reasoning:

    The TTFT and ITL requirements are tight — TTFT at 200 ms over a 70B model means the prefill must complete fast, which favors TensorRT-LLM's compiled prefill kernels. ITL under 20 ms constrains the decode throughput at batch size 1 to at least 50 tokens/sec; on A100 with TRT-LLM, a 70B BF16 model sharded over 8 GPUs comfortably exceeds this.

    Key configuration decisions:
    1. **Tensor parallelism = 8** across one node (all-reduce on NVLink, low latency). Evaluate whether a single 8-GPU node suffices before adding nodes.
    2. **FP8 quantization** (if H100 is available) or **INT8 SmoothQuant** (A100) to reduce model size and improve decode bandwidth.
    3. **INT8 KV cache** to roughly double the number of concurrent sequences that fit in KV memory.
    4. **`max_batch_size`** should be set based on the memory budget worked out above — roughly 250–500 concurrent tokens at the 70B scale.
    5. **Scheduler policy = `guaranteed_no_evict`** to avoid KV recomputation latency for low-TTFT guarantees; absorb load spikes via queuing in the Triton front-end.
    6. Monitor time-to-first-token and inter-token latency as SLO metrics via Triton's Prometheus metrics endpoint.

    If the team has limited NVIDIA expertise or needs to iterate on the model weekly, TGI or vLLM would be a reasonable second choice — both can reach the SLO with proper tuning and avoid the 30–60 minute build cycle.

---

## Stack Selection Cheatsheet

{{fig:trtllm-stack-selection-decision-tree}}

---

!!! key "Key Takeaways"
    - TensorRT-LLM achieves peak NVIDIA GPU throughput through AOT compilation: kernels are selected and fused at build time, eliminating runtime dispatch overhead. The cost is hardware-specific engines and 30–60 minute build times.
    - TGI provides a production-grade HTTP/gRPC server with continuous batching, custom FlashAttention kernels, and wide HuggingFace model support — no build step required, making it the default choice for teams that iterate quickly on models.
    - llama.cpp / Ollama democratize LLM inference on commodity hardware. GGUF's block quantization (Q4_K_M, Q5_K_M) trades a small quality loss for 4–8× memory reduction, enabling 70B models to run on a laptop or a single consumer GPU.
    - LMDeploy's TurboMind engine is a practical middle ground: custom CUDA kernels and blocked KV cache without the full AOT compilation step of TensorRT-LLM.
    - MLC-LLM uses TVM's compiler infrastructure to target heterogeneous hardware (CUDA, ROCm, Metal, Vulkan, WebGPU) from a single codebase — the right choice for browser deployment via WebLLM or mixed-hardware fleets.
    - The KV cache is the primary memory budget constraint at serving time. INT8 KV quantization roughly doubles the number of concurrent sequences you can hold, often improving throughput more than any single kernel optimization.
    - The portability vs peak performance trade-off is real but often modest: well-tuned TGI or vLLM reaches 85–95% of TensorRT-LLM's throughput in most workloads. The remaining gap matters most at very large scale (thousands of dollars/day in GPU cost) or tight latency SLOs.
    - For production deployments, the serving stack is just one component: request routing, prefix caching, speculative decoding, and disaggregated prefill/decode can each contribute as much to end-to-end efficiency as the choice of framework.

---

!!! sota "State of the Art & Resources (2026)"
    LLM serving has matured into a rich ecosystem of specialized runtimes ranging from NVIDIA's AOT-compiled TensorRT-LLM for peak GPU throughput to portable frameworks like llama.cpp and MLC-LLM for edge and heterogeneous hardware. The dominant trend through 2024–2026 is disaggregated prefill/decode serving, speculative decoding, and prefix-cache-aware scheduling — each delivering multiplicative throughput gains on top of the foundational PagedAttention memory model.

    **Foundational work**

    - [Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention* (2023)](https://arxiv.org/abs/2309.06180) — introduced block-paged KV cache management (vLLM); every major serving stack now implements a variant.

    **Recent advances (2023–2026)**

    - [Zhong et al., *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving* (2024)](https://arxiv.org/abs/2401.09670) — OSDI 2024 paper showing that separating prefill and decode onto dedicated GPU pools removes interference and sharply improves latency SLOs.
    - [Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs* (2024)](https://arxiv.org/abs/2312.07104) — NeurIPS 2024; introduces RadixAttention for KV-cache prefix reuse and compressed FSMs for constrained decoding; up to 6.4× throughput over prior baselines.
    - [Kolluru, *Comparative Analysis of LLM Inference Serving Systems: vLLM and HuggingFace TGI* (2025)](https://arxiv.org/abs/2511.17593) — empirical benchmark across throughput, latency, GPU memory, and scalability on LLaMA-2 models.

    **Open-source & tools**

    - [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) — NVIDIA's AOT-compilation library for LLMs; supports FP4/FP8, speculative decoding, multi-GPU tensor parallelism, and disaggregated serving.
    - [vllm-project/vllm](https://github.com/vllm-project/vllm) — the de-facto open-source serving engine (2 000+ contributors); PagedAttention, continuous batching, prefix caching, and broad model support.
    - [sgl-project/sglang](https://github.com/sgl-project/sglang) — high-performance framework with RadixAttention prefix caching and fast structured-output decoding; growing rapidly as a vLLM alternative.
    - [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — canonical C/C++ runtime for GGUF-quantized models; targets CPU, CUDA, Metal, Vulkan, and ROCm with no external ML dependencies.
    - [mlc-ai/mlc-llm](https://github.com/mlc-ai/mlc-llm) — TVM-based universal LLM compiler targeting NVIDIA, AMD, Apple Metal, Vulkan, WebGPU, and Android from a single codebase.
    - [InternLM/lmdeploy](https://github.com/InternLM/lmdeploy) — TurboMind C++/CUDA engine with blocked KV cache and AWQ quantization; strong mid-tier option between TGI and TensorRT-LLM.

    **Go deeper**

    - [TensorRT-LLM documentation — Overview](https://nvidia.github.io/TensorRT-LLM/overview.html) — official docs covering the executor API, quantization recipes, multi-GPU configs, and speculative decoding setup.
    - [NVIDIA Technical Blog: *TensorRT-LLM Speculative Decoding Boosts Inference Throughput by up to 3.6×* (2024)](https://developer.nvidia.com/blog/tensorrt-llm-speculative-decoding-boosts-inference-throughput-by-up-to-3-6x/) — step-by-step guide and benchmark results for draft-model speculative decoding on H200 GPUs.

## Further Reading

- **TensorRT-LLM repository** — NVIDIA, github.com/NVIDIA/TensorRT-LLM. The primary reference for engine build options, supported models, and the Executor API.
- **Text Generation Inference repository** — Hugging Face, github.com/huggingface/text-generation-inference. The Rust router source and Python model server are both readable and well-documented.
- **Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023** — foundational paper for block-based KV cache management; the ideas are implemented in vLLM, TGI, TRT-LLM, and LMDeploy.
- **Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," NeurIPS 2022** — the attention kernel that every production stack now ships.
- **llama.cpp repository** — Georgi Gerganov, github.com/ggerganov/llama.cpp. The GGUF specification and all backend implementations are in this repo.
- **MLC-LLM: Universal LLM Deployment** — Chen et al., MLC team, github.com/mlc-ai/mlc-llm. Describes the TVM-based compilation pipeline and cross-platform targets.
- **LMDeploy repository** — Shanghai AI Laboratory, github.com/InternLM/lmdeploy. TurboMind engine design documentation and AWQ quantization integration.
- **Aminabadi et al., "DeepSpeed-Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale," SC 2022** — covers multi-GPU inference kernels and the design philosophy behind high-throughput serving at scale.
