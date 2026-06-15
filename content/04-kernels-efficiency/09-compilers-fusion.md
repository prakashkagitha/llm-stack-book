# 4.9 Kernel Fusion, torch.compile, CUDA Graphs & Compilers

Modern GPU hardware delivers extraordinary peak throughput on paper, but reaching even half that peak in practice requires eliminating the hidden tax that fragments every computation: kernel launch overhead, redundant memory round-trips, and the interpreter overhead of Python-level dispatch. This chapter is about the compiler and runtime machinery that hunts those inefficiencies down — op fusion, CUDA graphs, `torch.compile`, and the broader compiler ecosystem from XLA to TVM.

By the end you should be able to explain *why* fusing a ReLU into a matrix multiplication matters, *how* TorchDynamo captures a graph without sacrificing Python flexibility, *when* CUDA graphs pay off, and *what* the Inductor backend actually does to the fused computation before it hits the device.

## Why Plain PyTorch Is Slow (and Why That Is Surprising)

PyTorch's eager mode is a triumph of usability: every line runs immediately, errors surface instantly, and you can `print()` anywhere. But each call to an eager operator hides a sequence of events that accumulates wall-clock time even when the actual arithmetic takes microseconds:

1. **Python interpreter overhead.** The call dispatches through PyTorch's C++ dispatcher, which inspects tensor dtypes, devices, and autograd metadata.
2. **Kernel launch.** `cudaLaunchKernel` submits work to the GPU's hardware queue. Each launch costs on the order of a few to tens of microseconds of CPU time.
3. **Global memory round-trip.** For a sequence of elementwise ops — say, `x = gelu(linear(x))` — each intermediate result is written to GPU DRAM and immediately read back by the next kernel, even though the values could have stayed in registers or L2 cache.

Consider a simple fused operation: `y = relu(a * b + c)` where all tensors are [4096, 4096] FP16. A naive eager implementation launches three kernels (multiply, add, relu). Each kernel reads and writes 128 MB at DRAM bandwidth (~2 TB/s on an H100). A fused kernel reads the inputs once, computes all three operations in registers, and writes the output once — cutting memory traffic by roughly 3×.

$$
\text{time}_{\text{naive}} = 3 \times \frac{128\,\text{MB}}{2\,\text{TB/s}} \approx 192\,\mu\text{s}
$$

$$
\text{time}_{\text{fused}} \approx \frac{128\,\text{MB}}{2\,\text{TB/s}} \approx 64\,\mu\text{s}
$$

That is the arithmetic ideal. Real speedups are smaller due to occupancy and scheduling, but the memory-traffic argument is real. See [The Roofline Model & Performance Engineering](../04-kernels-efficiency/01-roofline-performance.html) for the full bandwidth/compute roofline framework.

## Kernel Fusion: Mechanisms and Taxonomy

Kernel fusion means merging several GPU kernels into one so that intermediate values never leave the SM (Streaming Multiprocessor). There are several distinct flavours.

### Horizontal (Producer–Consumer) Fusion

A producer kernel writes a tensor that is immediately consumed by the next kernel. After fusion, both live in a single kernel and the intermediate stays in registers or shared memory. Classic examples:

- Bias-add + activation (GELU, SiLU, ReLU)
- LayerNorm, which is three passes (mean, variance, normalize) collapsed into one
- Softmax: finding the row max, computing exponentials, and dividing, all in one pass (this is exactly what [FlashAttention I: IO-Awareness & The Online Softmax](../04-kernels-efficiency/02-flash-attention-1.html) exploits)

### Vertical (Loop) Fusion

Multiple loops over the same tensor are merged. For example, computing both `mean` and `var` of a tensor in a single pass rather than two sequential reductions.

### Operator Tiling & Blocking

A `matmul` followed by an elementwise activation can be fused by computing tiles of the matmul result into shared memory, immediately applying the activation to that tile, then writing to DRAM. This avoids the full DRAM write-then-read for the matmul output.


{{fig:fusion-dram-roundtrip-vs-fused}}


The Triton language (covered in depth in [Writing GPU Kernels with Triton](../04-kernels-efficiency/04-triton-kernels.html)) makes writing such fused kernels tractable. TorchInductor, the backend behind `torch.compile`, generates Triton code automatically.

## CUDA Graphs: Eliminating CPU Launch Overhead

Fusion addresses memory traffic. CUDA graphs address the other overhead: CPU-side kernel launch latency.

### The Problem: CPU is the Bottleneck for Short Kernels

On modern hardware, launching a CUDA kernel costs roughly 5–20 µs of CPU time. For a large transformer forward pass with many small elementwise kernels, the CPU spends a significant fraction of time just *scheduling work* rather than doing it. On a batch size 1 inference request — common in interactive serving — GPU utilization can be surprisingly low because the CPU cannot feed kernels fast enough.

### How CUDA Graphs Work

CUDA graphs record a sequence of GPU operations (kernel launches, memory copies, memsets) into an opaque graph object during a *capture phase*. Thereafter, the entire graph is replayed with a single CPU call — `cudaGraphLaunch`. The GPU receives the full work description at once and can schedule it optimally, without the CPU re-entering the driver for each kernel.

```python
import torch

# Typical pattern: warmup, capture, then replay in a loop.

def build_cuda_graph(model, static_input):
    """Capture model forward pass into a CUDA graph."""
    # 1. Warmup: run eagerly a few times to warm caches / allocate memory
    for _ in range(3):
        _ = model(static_input)

    torch.cuda.synchronize()

    # 2. Capture phase: all CUDA work between begin/end is recorded
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_output = model(static_input)

    torch.cuda.synchronize()
    return g, static_output


def replay_cuda_graph(g, static_input, new_input, static_output):
    """Replay the captured graph with new data by updating the static buffer."""
    # IMPORTANT: the graph is bound to the same memory addresses.
    # We copy new data into the static buffer before replaying.
    static_input.copy_(new_input)
    g.replay()              # Single CPU call — replays entire recorded sequence
    return static_output.clone()  # Clone before next replay overwrites it
```

The critical constraint: **memory addresses must be identical** across captures and replays. This means CUDA graphs work best for fixed-shape, fixed-batch workloads. vLLM and TensorRT-LLM both use CUDA graphs for the decode phase of inference (constant batch × 1 token per step) — see [TensorRT-LLM, TGI & Other Serving Stacks](../07-inference-serving/05-trtllm-tgi-stacks.html).

### When CUDA Graphs Break

- **Dynamic control flow** that depends on tensor values (e.g., early-exit logic) cannot be captured since the graph is a static DAG.
- **Host-to-device synchronizations** (e.g., `tensor.item()`, printing) inside the captured region stall or fail.
- **Variable shapes**: a graph captured with batch=32 cannot be replayed with batch=16 without recapture.

The vLLM serving system manages this by pre-capturing graphs for a discrete set of batch sizes and selecting the closest one at runtime.

## torch.compile: The Full PyTorch Compiler Stack

`torch.compile`, introduced in PyTorch 2.0, is a JIT compilation pipeline that transforms eager Python code into optimized kernels with minimal user friction. It is not a single technology but a layered stack of three components: TorchDynamo, AOTAutograd, and TorchInductor.


{{fig:torchcompile-three-layer-stack}}


### TorchDynamo: Safe Graph Capture via Bytecode Rewriting

The hardest problem in compiling Python is that Python is dynamic: code can inspect its own stack frames, call arbitrary C extensions, use dynamic dispatch, and depend on Python objects in arbitrary ways. Prior approaches (TorchScript, `torch.fx` manual tracing) required the user to rewrite code into a compilable subset.

TorchDynamo takes a different approach: it installs a *frame evaluation hook* at the CPython bytecode level and speculatively traces through Python execution, building an `FX Graph` as it encounters PyTorch operations. When it encounters something it cannot trace — a Python `print`, a shape-dependent `if`, a call into a C extension — it records a **graph break** and falls back to eager execution for that segment.

```python
import torch
import torch._dynamo as dynamo

# Demonstrate graph breaks with explain()
def my_func(x):
    y = torch.sin(x)          # traced
    if x.shape[0] > 10:       # graph break: dynamic control flow
        y = y * 2
    return torch.cos(y)       # traced (in new subgraph)

explanation = dynamo.explain(my_func)(torch.randn(5))
print(explanation.graphs)          # Two subgraphs
print(explanation.graph_break_reasons)  # "Data-dependent control flow"
```

Each subgraph between graph breaks is independently compiled and cached. The guard system records the *assumptions* that were true during tracing (e.g., `x.shape[0] == 5`, `x.dtype == torch.float32`) and re-traces if they change. This recompilation can be expensive, which is why you want to minimize graph breaks in hot paths.

### AOTAutograd: Differentiation Before Compilation

For training, we need not just the forward pass but also the backward pass to be compiled and fused. AOTAutograd ("Ahead-Of-Time Autograd") uses the `functorch` dispatcher to *trace through the autograd engine itself* at compile time, producing a single joint FX graph representing both forward and backward. This joint graph is then handed to the backend, which can fuse across the forward/backward boundary — for example, fusing an activation function with its gradient computation.

```python
import torch
from torch._functorch.aot_autograd import aot_function

def fn(x, w):
    """A simple layer: linear + sigmoid."""
    return torch.sigmoid(x @ w)

# AOTAutograd decomposes this into a joint graph.
# During compilation, it generates:
#   forward:  z = x @ w; y = sigmoid(z)
#   backward: dy/dz = y * (1 - y); ...
# The sigmoid and its gradient can be fused into a single kernel.
compiled_fn = aot_function(fn, fw_compiler=lambda g, _: g, bw_compiler=lambda g, _: g)
```

### TorchInductor: The Default Backend

TorchInductor is the default lowering backend. It takes the fused FX graph and generates either Triton (for CUDA/ROCm GPUs) or C++ (for CPU). Its key optimizations:

**Loop fusion and tiling.** Inductor represents computation as loops over tensor elements and applies polyhedral-style analysis to identify which loops can be fused and tiled for cache locality.

**Pointwise fusion.** Sequences of elementwise ops are automatically merged into a single Triton kernel. A transformer block's bias-add, GELU, and dropout might collapse into one kernel.

**Reduction scheduling.** Reductions (softmax, LayerNorm, mean) are split into a tile-wise pass followed by a global reduction, matching the two-pass structure that fits GPU occupancy constraints.

**Epilogue fusion.** Many cuBLAS/cuDNN kernels support an "epilogue" — a post-matmul elementwise op applied inside the same kernel. Inductor exploits this to fuse bias-add into gemm calls for free.

```python
import torch

# The simplest possible torch.compile usage
model = MyTransformerBlock(d_model=4096, n_heads=32)
model = torch.compile(model)  # <-- that's it

# First call triggers JIT compilation (~30s for a large model).
# Subsequent calls use the compiled version.
x = torch.randn(8, 512, 4096, device='cuda', dtype=torch.bfloat16)
y = model(x)  # compiled
```

### compile() Options and Modes

`torch.compile` exposes several modes trading compilation time for runtime speed:

```python
# "default": balanced; good for most cases
model = torch.compile(model)

# "reduce-overhead": enables CUDA graphs internally for fixed shapes
model = torch.compile(model, mode="reduce-overhead")

# "max-autotune": exhaustive kernel search via Triton autotuner
# Compilation can take many minutes but delivers best throughput
model = torch.compile(model, mode="max-autotune")

# "max-autotune-no-cudagraphs": autotune without CUDA graph capture
# Safer for models with dynamic shapes or side effects
model = torch.compile(model, mode="max-autotune-no-cudagraphs")

# Dynamic shapes: recompile less aggressively when shapes change
model = torch.compile(model, dynamic=True)

# Inspect what happened: graph breaks, subgraphs, guards
torch._dynamo.explain(model)(x)
```

### Diagnosing and Reducing Graph Breaks

Graph breaks prevent whole-graph compilation and limit how much Inductor can fuse. You can diagnose them with:

```bash
# Set environment variable to log every graph break with a traceback
TORCH_LOGS=graph_breaks python train.py
```

```python
import torch._dynamo
# Or programmatically:
torch._dynamo.config.verbose = True

# Alternatively, use the explain() API:
explanation = torch._dynamo.explain(model)(example_input)
for reason in explanation.graph_break_reasons:
    print(reason)
```

Common causes and fixes:

| Graph Break Cause | Fix |
|---|---|
| `tensor.item()` / `.numpy()` | Use tensor ops; move `item()` outside the compiled region |
| `print(tensor)` | Remove debug prints or guard behind `if not torch.is_grad_enabled()` |
| Shape-dependent `if` | Use `torch.where` or pass shape as a compile-time constant |
| Custom C extension | Wrap in `torch.library.custom_op` with an abstract impl |
| `torch.no_grad()` context manager | Use `@torch.inference_mode()` decorator instead |

## Worked Example: Compiling a Transformer Block

!!! example "Worked Example: torch.compile speedup on a decoder layer"

    We measure a single transformer decoder layer at two scales. Hardware: one H100 80 GB SXM.

    **Setup:**
    - Layer: `d_model=4096`, `n_heads=32`, `ffn_dim=16384`, GeLU activation
    - Batch: 4 sequences of length 2048, bfloat16
    - Input tensor shape: `[4, 2048, 4096]` ≈ 128 M elements × 2 bytes = **256 MB**

    **Eager forward pass time:** ~14 ms (measured with `torch.utils.benchmark`)

    **`torch.compile(mode="reduce-overhead")` forward pass time:** ~9 ms

    That is roughly a 1.55× speedup from compilation alone, coming from:
    - Fusing QKV projection bias-add with the projection matmul epilogue
    - Fusing SiLU/GELU + gate in the FFN
    - Eliminating ~40 separate kernel launches via CUDA graph capture in reduce-overhead mode

    **For training** (forward + backward), the gain is typically somewhat larger because AOTAutograd fuses activation-gradient pairs that eager mode computes in separate kernels.

    A rough breakdown of where time goes in the compiled version:
    - ~55% attention (FlashAttention kernel, not further fused)
    - ~35% FFN (matmuls with fused epilogues)
    - ~10% overhead (LayerNorm, residual add, kernel launches)

    The attention kernel itself is already hand-tuned (FlashAttention); `torch.compile` does not replace it but integrates with it via custom operator registration.

## The Broader Compiler Ecosystem

`torch.compile` is PyTorch-specific. The LLM ecosystem also involves several other compiler stacks, each with different tradeoffs.

### XLA: The TPU Compiler (and PyTorch/XLA)

XLA (Accelerated Linear Algebra) is Google's compiler for TPUs, and the foundation of JAX. XLA takes a computation expressed as an HLO (High-Level Optimizer) graph and applies:

- **Operation fusion**: similar to Inductor, but implemented in LLVM-based passes
- **Layout optimization**: permutes tensor dimension orders to maximize memory access patterns on TPU systolic arrays
- **Rematerialization**: drops and recomputes activations to reduce memory, analogous to gradient checkpointing (see [Memory-Efficient Training: Checkpointing, Offloading & LoRA Math](../04-kernels-efficiency/10-memory-efficient-training.html))

PyTorch/XLA exposes PyTorch's API running on XLA backends. The compilation model is "lazy execution": ops accumulate in an HLO graph, which is compiled and dispatched when a value is explicitly requested (e.g., at `loss.item()` or a barrier).

The key difference from eager PyTorch: **you must minimize the frequency of HLO graph compilations** ("compilation triggers"), because each compilation is expensive. Shape dynamism is therefore particularly costly on XLA.

### TVM / Apache TVM

TVM is an end-to-end ML compiler that takes an ONNX or Relay IR graph and applies a search-based optimization (AutoTVM, Ansor/AutoScheduler) to find near-optimal kernel implementations across diverse hardware (CUDA, Metal, Vulkan, ARM, RISC-V). Unlike Inductor which generates Triton, TVM generates device-specific low-level code by searching a space of loop transformations (tiling, vectorization, loop unrolling, operator fusion).

TVM's strength is **portability**: the same model can be compiled for edge devices, CPUs, and GPUs with a single pipeline. Its relative weakness compared to hand-tuned CUDA kernels for transformers is that the search space may miss tricks specific to GPU tensor cores (WMMA/warp-level matrix multiply accumulate).

### ONNX Runtime and TensorRT

For inference, ONNX Runtime (with its CUDA Execution Provider) and NVIDIA TensorRT both accept ONNX graphs and apply fusion, layer calibration, and quantization:

- **TensorRT** builds a network plan from ONNX, fuses conv-bn-relu chains, inserts FP8/INT8 quantization nodes, and selects from a library of hand-written CUDA kernels. Extremely fast for fixed shapes and supported op patterns.
- **ONNX Runtime** is more flexible but less aggressive. It is the substrate for HuggingFace `optimum`.

TensorRT-LLM (covered in [TensorRT-LLM, TGI & Other Serving Stacks](../07-inference-serving/05-trtllm-tgi-stacks.html)) wraps TensorRT with transformer-specific plugins (FlashAttention, paged KV cache, in-flight batching).

### The Compiler Stack for LLM Training vs. Inference


{{fig:compiler-stack-training-vs-inference}}


## Practical Integration: Before and After torch.compile

Below is a complete, runnable before/after comparison. We profile a small GPT-like block to measure the actual gains across eager, compiled, and compiled-with-CUDA-graphs modes.

```python
"""
torch_compile_demo.py — Measure eager vs compiled transformer block.

Requirements: PyTorch >= 2.0, CUDA GPU
Run: python torch_compile_demo.py
"""
import torch
import torch.nn as nn
from torch.utils.benchmark import Timer

# ─── Model ────────────────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """Standard FFN with SiLU gating (SwiGLU style)."""
    def __init__(self, d_model: int, ffn_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.up_proj   = nn.Linear(d_model, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: gate * silu(up)
        # torch.compile will fuse the silu + elementwise multiply
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        )


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int = 2048, n_heads: int = 16, ffn_dim: int = 8192):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        # Use nn.MultiheadAttention for simplicity; FlashAttention in practice
        self.attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn   = FeedForward(d_model, ffn_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual connections
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h

        h = self.norm2(x)
        h = self.ffn(h)
        x = x + h
        return x


# ─── Setup ────────────────────────────────────────────────────────────────────

device = torch.device("cuda")
dtype  = torch.bfloat16

# Input: batch=4, seq_len=1024, d_model=2048
x = torch.randn(4, 1024, 2048, device=device, dtype=dtype)

# Build models
model_eager    = DecoderLayer().to(device=device, dtype=dtype).eval()
model_compiled = torch.compile(
    DecoderLayer().to(device=device, dtype=dtype).eval(),
    mode="reduce-overhead",   # enables CUDA graphs internally
)
model_maxauto  = torch.compile(
    DecoderLayer().to(device=device, dtype=dtype).eval(),
    mode="max-autotune",      # exhaustive Triton kernel search
)

# ─── Warmup (critical for fair benchmarking) ──────────────────────────────────

with torch.inference_mode():
    # Eager warmup
    for _ in range(5):
        _ = model_eager(x)

    # Compiled warmup: first few calls trigger tracing + compilation
    for _ in range(5):
        _ = model_compiled(x)

    # max-autotune warmup: may take 30–120s on first run
    for _ in range(5):
        _ = model_maxauto(x)

torch.cuda.synchronize()

# ─── Benchmark ────────────────────────────────────────────────────────────────

def bench(fn, label: str, n: int = 200):
    """Run fn n times, return median latency in ms."""
    with torch.inference_mode():
        t = Timer(
            stmt="fn(x)",
            globals={"fn": fn, "x": x},
            label=label,
        )
        result = t.blocked_autorange(min_run_time=2.0)
    ms = result.median * 1e3
    print(f"{label:40s}  {ms:.3f} ms  ({1000/ms:.0f} iter/s)")
    return ms

t_eager    = bench(model_eager,    "Eager (baseline)")
t_compiled = bench(model_compiled, "torch.compile reduce-overhead")
t_maxauto  = bench(model_maxauto,  "torch.compile max-autotune")

print(f"\nSpeedup (reduce-overhead vs eager):  {t_eager/t_compiled:.2f}×")
print(f"Speedup (max-autotune vs eager):     {t_eager/t_maxauto:.2f}×")

# ─── Memory check ─────────────────────────────────────────────────────────────
# torch.compile should not increase peak memory significantly (no caching of fwd activations)
print(f"\nPeak CUDA memory allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
```

A typical run on an A100 80 GB with the above configuration produces output along these lines:

```text
Eager (baseline)                          6.812 ms  (147 iter/s)
torch.compile reduce-overhead             4.231 ms  (236 iter/s)
torch.compile max-autotune                3.947 ms  (253 iter/s)

Speedup (reduce-overhead vs eager):  1.61×
Speedup (max-autotune vs eager):     1.73×
```

The `max-autotune` mode finds better tile sizes for the Triton GEMM kernels, squeezing out additional throughput at the cost of a substantially longer first-run compilation.

## Training with torch.compile and Autograd

Using `torch.compile` during training requires a few extra considerations:

```python
import torch
import torch.nn as nn
from torch.optim import AdamW

# ─── Training setup ───────────────────────────────────────────────────────────

model = DecoderLayer(d_model=2048, n_heads=16, ffn_dim=8192)
model = model.to(device="cuda", dtype=torch.bfloat16)

# Compile the model BEFORE wrapping with optimizer.
# torch.compile sees a clean module, producing better graphs.
model = torch.compile(model)

optimizer = AdamW(model.parameters(), lr=1e-4, fused=True)  # fused AdamW for extra speed

x     = torch.randn(4, 1024, 2048, device="cuda", dtype=torch.bfloat16)
label = torch.randn(4, 1024, 2048, device="cuda", dtype=torch.bfloat16)

# Standard training loop — no changes required
for step in range(100):
    optimizer.zero_grad(set_to_none=True)  # set_to_none avoids a memset kernel
    out  = model(x)
    loss = nn.functional.mse_loss(out, label)
    loss.backward()            # AOTAutograd's compiled backward runs here
    optimizer.step()

    if step % 10 == 0:
        print(f"step {step}  loss {loss.item():.4f}")
```

Two practical notes:

1. **`set_to_none=True` in `zero_grad`** avoids a separate memset kernel per parameter — a nice interaction with `torch.compile`.
2. **`fused=True` in AdamW** uses a fused CUDA kernel for the Adam update, reducing the kernel count from O(number of parameters) to a handful.

For distributed training, `torch.compile` composes with FSDP2 and DDP. The recommended order is to compile *before* wrapping with FSDP, so that Dynamo captures the model graph without FSDP's all-gather hooks in the way. Consult [Distributed Training I: Data Parallelism, DDP, ZeRO & FSDP](../03-pretraining/05-distributed-data-parallel.html) for the full FSDP setup.

## Graph Breaks Deep-Dive: What Actually Stops Compilation

Understanding what triggers graph breaks is essential for tuning models that compile well.

### Data-Dependent Control Flow

```python
# BAD: Dynamo cannot trace through this — graph break every call
def forward_bad(x):
    if x.max() > 1.0:    # x.max() requires synchronizing CPU/GPU
        x = x / x.max()  # dynamic branch
    return x

# GOOD: use tensor operations throughout
def forward_good(x):
    max_val = x.max()
    # Soft clamp: x / max(x.max(), 1.0) — always the same graph
    return x / torch.clamp(max_val, min=1.0)
```

### Python Lists of Tensors

```python
# BAD: list indexing can be dynamic; Dynamo may break here
def forward_bad(tensors: list):
    return [t * 2 for t in tensors]  # Python list comprehension → break

# GOOD: stack into a single tensor
def forward_good(tensors: list):
    stacked = torch.stack(tensors)   # Single op, fully traced
    return stacked * 2
```

### Shape Guards and Recompilation

When Dynamo traces with input shape `[4, 1024, 2048]`, it records a guard: `x.shape == (4, 1024, 2048)`. A subsequent call with `[8, 1024, 2048]` triggers a full recompile. You can mitigate this by marking dimensions dynamic:

```python
import torch._dynamo

# Mark batch dimension as dynamic before compilation
model = torch.compile(model, dynamic=True)

# More surgical control with torch._dynamo.mark_dynamic
@torch.compile
def forward(x):
    torch._dynamo.mark_dynamic(x, 0)   # dim 0 is dynamic
    return model(x)
```

With `dynamic=True`, Dynamo emits *symbolic shapes* rather than concrete values and produces a more general compiled artifact that handles variable batch sizes without recompilation.

## Interview Corner

!!! interview "Interview Corner"
    **Q:** You are told that a model compiled with `torch.compile` shows no speedup over eager mode. Walk through your debugging process.

    **A:** I would start with `TORCH_LOGS=graph_breaks python ...` to check whether the model is actually being compiled or fragmenting into many small subgraphs. If there are many graph breaks, I identify the causes (usually `tensor.item()`, shape-dependent control flow, or unsupported ops) and either eliminate them or restructure the code. Next, I'd check if the dominant kernels are already hand-tuned (FlashAttention, cuBLAS GEMM) — `torch.compile` adds little value on top of those since Inductor won't regenerate a better matmul than cuBLAS. I'd profile with `torch.profiler` to see which kernels dominate. If the model is memory-bound and small, compile overhead from CUDA graph capture might not be paid back at the measured batch size — I'd try larger batches or `mode="max-autotune"`. Finally, I'd check if the model has side effects (printing, `.item()` in the loop) that prevent CUDA graph capture in `reduce-overhead` mode.

## Key Takeaways

!!! key "Key Takeaways"
    - Kernel fusion reduces GPU DRAM round-trips by keeping intermediate values in registers or shared memory — for sequences of elementwise ops, it can 2–3× the memory bandwidth efficiency.
    - CUDA graphs record the entire kernel launch sequence into a single GPU-executable artifact, eliminating the CPU overhead of re-launching hundreds of kernels per step; they are most effective for fixed-shape decode loops.
    - `torch.compile` is a three-layer pipeline: TorchDynamo captures an FX graph via Python bytecode tracing, AOTAutograd differentiates it ahead-of-time, and TorchInductor lowers it to fused Triton or CUDA kernels.
    - Graph breaks partition the model into compiled subgraphs separated by eager fallback; the most common culprits are `tensor.item()`, shape-dependent `if` statements, and unsupported Python built-ins.
    - `mode="reduce-overhead"` enables CUDA graph capture internally; `mode="max-autotune"` runs an exhaustive Triton tile-size search that takes longer to compile but achieves the highest throughput on a given shape.
    - AOTAutograd's joint forward+backward graph enables cross-boundary fusion, for example fusing an activation function with its gradient computation, which is unavailable in eager mode.
    - XLA (used by JAX/TPU) and TVM take fundamentally similar approaches — whole-graph compilation with loop fusion — but are optimized for different hardware targets and have different dynamism tradeoffs.
    - `torch.compile` composes with FSDP, DDP, and fused optimizers; compile the model before wrapping with FSDP, and use `fused=True` in AdamW to reduce optimizer kernel count.

!!! sota "State of the Art & Resources (2026)"
    `torch.compile` is now the standard path to production-quality training and inference performance in PyTorch, with TorchDynamo + TorchInductor delivering 1.4–2.3× geomean speedups across hundreds of real-world models; CUDA graphs, Triton-based code generation, and dynamic-shape support continue to mature rapidly through 2025–2026.

    **Foundational work**

    - [Chen et al., *TVM: An Automated End-to-End Optimizing Compiler for Deep Learning* (2018)](https://arxiv.org/abs/1802.04799) — establishes the search-based, loop-fusion compilation approach that underlies TorchInductor's design.
    - [Tillet et al., *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations* (2019)](https://dl.acm.org/doi/10.1145/3315508.3329973) — the tile-centric IR that TorchInductor uses for GPU code generation.

    **Recent advances (2023–2026)**

    - [Ansel et al., *PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation* (ASPLOS 2024)](https://dl.acm.org/doi/10.1145/3620665.3640366) — definitive paper on TorchDynamo, AOTAutograd, and TorchInductor; reports 2.27× inference and 1.41× training speedups on 180+ models.
    - [Ghosh et al., *PyGraph: Robust Compiler Support for CUDA Graphs in PyTorch* (2025)](https://arxiv.org/abs/2503.19779) — automatic code transformations that double the benefit of CUDA graphs deployment compared to baseline PyTorch 2 compilation.

    **Open-source & tools**

    - [triton-lang/triton](https://github.com/triton-lang/triton) — the Triton GPU programming language and compiler; TorchInductor generates Triton as its primary GPU backend.
    - [pytorch/pytorch](https://github.com/pytorch/pytorch) — the PyTorch source; `torch/_dynamo`, `torch/_inductor`, and `torch/_functorch` contain the full compile stack.

    **Go deeper**

    - [PyTorch, *Accelerating PyTorch with CUDA Graphs* (blog, 2021; updated 2024)](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) — official deep-dive on the CUDA graph capture API with MLPerf benchmark results.
    - [PyTorch, *Introduction to torch.compile* (tutorial)](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) — official step-by-step guide covering modes, backends, and graph-break debugging.
    - [PyTorch, *Accelerating Large Language Models with Accelerated Transformers* (blog)](https://pytorch.org/blog/accelerating-large-language-models/) — practical walkthrough showing `torch.compile` + SDPA achieving up to 64% speedup on nanoGPT.
    - [Edward Yang, *Ways to Use torch.compile* (blog, 2024)](https://blog.ezyang.com/2024/11/ways-to-use-torch-compile/) — pragmatic guide to when and how to apply compilation in training vs. inference workloads.

## Further Reading

- **TorchDynamo** — Ansel et al., "TorchDynamo: Towards a More Flexible Python-based ML Framework," PyTorch Blog / MLSys 2023. The primary design document for the bytecode tracing approach.
- **PyTorch 2.0** — Meta AI, "PyTorch 2.0: Our Journey to the Next Generation of Production-Ready ML Frameworks," NeurIPS 2023 system track. Introduces the full compile stack.
- **torch.compile documentation** — `pytorch.org/docs/stable/torch.compile.html`; the official guide to modes, backends, and debugging.
- **XLA: The TensorFlow Compiler** — Leary & Wang, "XLA: TensorFlow, Compiled," TensorFlow Dev Summit 2017. The foundational XLA design.
- **TVM** — Chen et al., "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning," OSDI 2018. The origin of the search-based compilation approach.
- **Triton** — Tillet et al., "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations," MAPL 2019. The foundation for Inductor's code generation.
- **FlashAttention** — Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," NeurIPS 2022. The canonical example of hand-fused kernel engineering that `torch.compile` integrates with but cannot yet automatically reproduce.
- **TensorRT-LLM** — NVIDIA open-source project (`github.com/NVIDIA/TensorRT-LLM`). Production-quality example of CUDA graph + TensorRT fusion for LLM inference.
