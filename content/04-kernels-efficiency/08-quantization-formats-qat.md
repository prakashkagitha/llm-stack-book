# 4.8 Quantization II: INT4/INT8/FP8, GGUF, bitsandbytes & QAT

Post-training quantization (PTQ) compresses a trained model by changing the numeric format of its weights and, optionally, its activations. The previous chapter, [Quantization I: Post-Training Quantization (GPTQ, AWQ, SmoothQuant)](../04-kernels-efficiency/07-quantization-ptq.html), covered the calibration-based algorithms that decide *how* to pick the right scale factors. This chapter covers the *formats* those algorithms produce — INT4, NF4, INT8, FP8, and the GGUF k-quant family — and the inference runtimes and training techniques built around them. We also cover quantization-aware training (QAT) and QLoRA, which push accuracy further at the cost of more compute during training.

The core tension throughout is this: modern LLMs are memory-bandwidth-bound during autoregressive decoding (see [The Roofline Model & Performance Engineering](../04-kernels-efficiency/01-roofline-performance.html)), so using fewer bits per weight almost always helps throughput, but every bit removed increases the chance that the rounding error degrades generation quality beyond an acceptable threshold.

---

## The Quantization Landscape: A Taxonomy

Before diving into formats, it helps to fix terminology.

**Granularity** refers to how many weights share a single scale (and zero-point). Options range from per-tensor (one scalar), to per-row/per-column, to per-group (e.g., a block of 128 consecutive weights). Finer granularity costs more metadata but reduces quantization error dramatically — INT4 per-group-128 loses far less information than INT4 per-tensor.

**Scope** refers to what gets quantized:

- **Weight-only quantization (W-only):** Weights are stored in low precision; activations remain in BF16/FP16 at runtime. The kernel dequantizes weights on-the-fly and performs the GEMM in FP16. Memory footprint shrinks; arithmetic intensity of the GEMM itself is unchanged, but you win because you now move fewer bytes from HBM.
- **Weight + activation quantization (W+A):** Both weights and activations are quantized, usually to INT8 or INT8+INT4. The entire matrix multiply happens in low-precision integer arithmetic on hardware integer units, which can deliver higher TFLOP/s than FP16 on some GPU generations. The tradeoff is that activation distributions are much more dynamic and harder to quantize accurately.

**Symmetric vs. asymmetric:** Symmetric quantization maps $[-\alpha, +\alpha]$ linearly to $[-2^{b-1}, 2^{b-1}-1]$ — the zero-point is always 0, which simplifies dequantization math. Asymmetric allows a nonzero zero-point $z$ to shift the representable range, accommodating one-sided activation distributions (e.g., post-ReLU activations that are all positive).

The linear quantize-dequantize pair for a weight $w$ with scale $s$ and zero-point $z$ is:

$$
q = \operatorname{round}\!\left(\frac{w}{s}\right) + z, \quad \hat{w} = s \cdot (q - z)
$$

The quantization error per element is bounded by $\frac{s}{2}$, so the goal of calibration (GPTQ, AWQ, SmoothQuant) is to minimize $s$ for the most sensitive weight groups.

---

## INT8: The Safe Harbor

INT8 is the most widely deployed quantization format because the accuracy penalty is usually negligible and both Tensor Core (via `mma.sync`) and integer ALU paths are mature.

### INT8 Weight-Only (LLM.int8)

Tim Dettmers et al. introduced LLM.int8() as part of bitsandbytes. The key insight was that large language models have a small fraction of *outlier* activation channels — typically 0.1–1 % of channels depending on model size — that take values far outside the typical range. Quantizing these outliers with per-tensor INT8 causes catastrophic error.

The solution is **mixed-precision decomposition**: identify the handful of outlier columns at runtime, keep those multiplications in FP16, and quantize the rest as INT8 per-column. At 6.7 B parameters and above, this approach nearly eliminates the accuracy gap with FP16 while halving the memory footprint.

```python
# bitsandbytes INT8 weight-only quantization (load_in_8bit)
# Requires: pip install bitsandbytes transformers accelerate

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# `load_in_8bit=True` triggers LLM.int8() decomposition via bitsandbytes
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,       # weight-only INT8; activations remain FP16
    device_map="auto",       # spread layers across available GPUs
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# The model is now ~8 GB instead of ~16 GB for BF16
# Linear layers become bitsandbytes.nn.Linear8bitLt modules
for name, module in model.named_modules():
    if "Linear8bitLt" in type(module).__name__:
        print(f"{name}: weight dtype={module.weight.dtype}")
        break
```

{{fig:quant-llm-int8-outlier-decomposition}}

### INT8 Weight + Activation (SmoothQuant / TensorRT-LLM)

For weight+activation INT8, the serving stack typically uses per-token dynamic quantization of activations: at each layer, collect the max absolute value of the current token's activation vector, compute a scale, quantize to INT8, run the INT8 GEMM on Tensor Cores, and dequantize the output. This is done via CUTLASS or TensorRT-LLM's `int8_sq` plugin.

SmoothQuant (Xiao et al., 2022) migrates some of the per-channel outlier variance from activations into weights by applying a per-channel scaling factor $s_j$ before quantization:

$$
\mathbf{Y} = (\mathbf{X} \operatorname{diag}(\mathbf{s})^{-1}) \cdot (\operatorname{diag}(\mathbf{s}) \mathbf{W}^\top)
$$

After rescaling, both sides are smoother and quantize more cleanly. This is detailed further in [Quantization I: Post-Training Quantization (GPTQ, AWQ, SmoothQuant)](../04-kernels-efficiency/07-quantization-ptq.html).

---

## INT4 & NF4: Pushing to 4 Bits

4-bit quantization halves memory relative to INT8, enabling a 70 B model to fit on a single 80 GB A100 at around 35 GB. The challenge is that rounding errors are much larger with only 16 representable values.

### Per-Group INT4

The standard INT4 scheme uses **group quantization**: every $g$ consecutive weights (often $g = 128$) share one scale (and optionally one zero-point). The metadata overhead is $\frac{16}{g}$ bits per weight (one FP16 scale per group), negligible at $g = 128$.

For a weight matrix $\mathbf{W} \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$, each row is split into $\frac{d_\text{in}}{g}$ groups. Each group is quantized to INT4 values in $[-8, 7]$ (signed) or $[0, 15]$ (unsigned with zero-point). GPTQ and AWQ both output this format; the difference is only in *how* the scales are found.

Dequantization at inference time:

$$
\hat{w}_i = s_{\lfloor i/g \rfloor} \cdot (q_i - z_{\lfloor i/g \rfloor})
$$

where $s$ and $z$ are the group scale and zero-point. The kernel unpacks two 4-bit values from each byte, dequantizes to FP16, and then calls the FP16 GEMM.

### NF4: Normal Float 4

NF4 (Dettmers et al., QLoRA, 2023) is an *information-theoretically optimal* 4-bit data type for normally distributed weights. Instead of uniform spacing between the 16 quantization levels, NF4 spaces them at the quantiles of a standard normal distribution $\mathcal{N}(0,1)$. This minimizes expected squared quantization error for weights that are normally distributed (which pretrained LLM weights approximately are).

The 16 NF4 code points are the values $q_i$ such that:

$$
q_i = Q_\mathcal{N}\!\left(\frac{2i + 1}{2 \times 16}\right), \quad i = 0, 1, \ldots, 15
$$

where $Q_\mathcal{N}$ is the quantile function of the standard normal. At runtime, each weight group is rescaled by $s = \max(|\mathbf{w}|) / 0.9677$ (the scale that maps the group's maximum to the most extreme NF4 code point), then the nearest code point index is stored.

NF4 is the weight format used by QLoRA for the frozen base model. It achieves slightly lower perplexity than INT4 per-group on the same model at the same 4-bit budget, because its code points are better matched to the actual weight distribution.

{{fig:quant-nf4-vs-int4-codepoints}}

```python
# bitsandbytes NF4 quantization (load_in_4bit with bnb_4bit_quant_type="nf4")
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",       # vs "fp4" (less accurate but faster unpacking)
    bnb_4bit_compute_dtype=torch.bfloat16,  # dtype for the dequantized compute
    bnb_4bit_use_double_quant=True,  # double quantization (see below)
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
# Model is now ~4 GB instead of ~16 GB
```

### Double Quantization

Double quantization (also from QLoRA) applies a second quantization step to the scale factors themselves. Each group-128 scale is an FP32 value. With $g_2 = 256$ scales sharing a second-level scale (stored in FP8), the per-weight overhead of the primary scale is reduced from $32/128 = 0.25$ bits/weight to roughly $8/256 + 32/(256 \cdot 256) \approx 0.031 + 0.001 \approx 0.032$ bits/weight — an additional saving of about 0.22 bits/weight across the whole model.

!!! example "Worked example: memory budget for a 70 B model"
    A Llama-2-70B model has approximately 70 billion parameters. Let's compute memory under different quantization schemes.

    **BF16 baseline:** $70 \times 10^9 \times 2 \text{ bytes} = 140 \text{ GB}$. Requires two 80 GB A100s minimum.

    **INT8 weight-only (no scales overhead approx.):** $70 \times 10^9 \times 1 \text{ byte} \approx 70 \text{ GB}$. Fits on a single 80 GB A100 (with room for activations and KV cache).

    **INT4 per-group-128 (with FP16 scales):** Weights = $70 \times 10^9 \times 0.5 \text{ bytes} = 35 \text{ GB}$. Scales add $70 \times 10^9 / 128 \times 2 \text{ bytes} \approx 1.1 \text{ GB}$. Total: $\approx 36 \text{ GB}$.

    **NF4 + double quantization:** Weights $\approx 35$ GB. First-level FP8 scales: $70 \times 10^9 / 128 \times 1 \text{ byte} \approx 0.55$ GB. Second-level FP32 meta-scale: $70 \times 10^9 / (128 \times 256) \times 4 \text{ bytes} \approx 0.009$ GB. Total: $\approx 35.6$ GB — essentially the same as INT4, with a small extra saving from compressing the scales.

    In practice, a 70 B NF4+DQ model fits comfortably in 36–40 GB with overhead for the KV cache and activations, enabling single-GPU deployment on an A100-80GB or H100-80GB.

---

## FP8 Inference

FP8 introduces a floating-point format at 8 bits. Two variants exist:

- **E4M3:** 4 exponent bits, 3 mantissa bits. Dynamic range: roughly $[5.96 \times 10^{-8}, 448]$. Better for weights (need a wider range).
- **E5M2:** 5 exponent bits, 2 mantissa bits. Wider dynamic range, less precision per number. Better for gradients during training.

NVIDIA Hopper GPUs (H100, H200) introduced native FP8 Tensor Core support. The hardware performs FP8 × FP8 multiplications and accumulates in FP32, then scales and stores results in FP8 or FP16.

FP8 differs from INT8 in one crucial way: the scale granularity is coarser (per-tensor or per-row) and is baked into the hardware instruction as a scaling factor, rather than per-group. This makes FP8 better suited to weights and activations that are roughly uniform in magnitude, which is why FP8 shines for inference on GEMM-heavy forward passes (attention and FFN projections) but can degrade more than INT8 on outlier-heavy activations unless paired with SmoothQuant-style smoothing.

FP8 training is discussed in [Mixed Precision, bf16 & FP8 Training](../03-pretraining/08-mixed-precision-fp8.html) and [FlashAttention 2 & 3: Work Partitioning, Warp Specialization & FP8](../04-kernels-efficiency/03-flash-attention-2-3.html).

```python
# FP8 inference via TensorRT-LLM (conceptual API sketch)
# In practice, TRT-LLM's quantization workflow handles the conversion
# The snippet below shows the key concepts using transformer_engine directly.

import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

# Create FP8 recipe: E4M3 for forward, E5M2 for backward (if training)
fp8_recipe = DelayedScaling(
    fp8_format=Format.E4M3,
    amax_history_len=16,      # track amax over last 16 iters to set scale
    amax_compute_algo="max",
)

# A TransformerEngine linear layer that uses FP8 GEMM on Hopper
linear = te.Linear(4096, 4096, bias=False)

with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    x = torch.randn(1, 512, 4096, device="cuda", dtype=torch.bfloat16)
    y = linear(x)  # internally uses FP8 GEMM
    print(f"Output dtype: {y.dtype}")  # still BF16 after cast-back
```

For practical FP8 inference with vLLM:

```bash
# Enable FP8 quantization in vLLM (requires H100/H200 or Ada generation)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-70B-Instruct \
    --quantization fp8 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.92
```

---

## GGUF & llama.cpp K-Quants

GGUF (GPT-Generated Unified Format) is the binary container format used by llama.cpp, the most widely deployed CPU/edge inference engine. It replaces the older GGML format and stores model weights alongside all necessary metadata (tokenizer, architecture hyperparameters, rope parameters, etc.) in a single self-describing file.

### The K-Quant Family

llama.cpp's k-quants are a family of mixed-precision block quantization schemes. The name comes from the original author's handle and the "k" for the block size. The key innovation is **block-level mixed precision**: within each block of 256 weights, the scheme uses different bit widths for the "super-block scale" (stored at higher precision) versus the individual weight quantization steps.

The available k-quant types and their approximate sizes per weight:

| Format   | Bits/weight (approx) | Notes |
|----------|---------------------|-------|
| Q2_K     | 2.6 bits            | Very aggressive; noticeable quality loss |
| Q3_K_S/M/L | 3.0–3.5 bits     | S=small scales, M=medium, L=large scales |
| Q4_K_S/M | 4.1–4.6 bits        | Most popular tradeoff; M has larger super-block |
| Q5_K_S/M | 5.0–5.5 bits        | Near-BF16 quality on most models |
| Q6_K     | 6.6 bits            | Essentially lossless for 7–13 B models |
| Q8_0     | 8.0 bits            | INT8, simple scale-per-32-weights |

**Q4_K_M internal structure:** Each super-block covers 256 weights. It stores two fp16 super-block scales — `d` (applied to the sub-block scales) and `dmin` (applied to the sub-block mins) — plus 8 sub-block scales and 8 sub-block mins covering 8 sub-blocks of 32 weights each. Those 16 values (8 scales + 8 mins) are each quantized to 6 bits and packed into a 12-byte array. Individual weights are stored as 4-bit unsigned integers. This matches ggml's `block_q4_K` struct, which is the source of truth: `ggml_half d, dmin;` (the two fp16 super-block scales), `uint8_t scales[12];` (the sixteen 6-bit sub-block scales and mins), and `uint8_t qs[128];` (the 256 4-bit weights). The dequantization formula per weight is:

$$
\hat{w} = d \cdot s_j \cdot q \; - \; d_\text{min} \cdot m_j
$$

where $d$ and $d_\text{min}$ are the two fp16 super-block scales (for the sub-block scales and mins respectively), $s_j$ and $m_j$ are the 6-bit scale and min of the sub-block $j$ that weight $\hat{w}$ belongs to, and $q \in [0, 15]$ is the stored 4-bit weight.

{{fig:quant-q4km-superblock-anatomy}}

```python
# Convert a Hugging Face model to GGUF Q4_K_M using llama.cpp's converter
# First clone llama.cpp and install dependencies:
# git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
# pip install -r requirements.txt && make -j

# Step 1: Convert HF model to GGUF F16 (lossless intermediate)
# (Run from the llama.cpp directory)
```

```bash
# Step 1: Lossless F16 conversion
python convert_hf_to_gguf.py \
    /path/to/meta-llama/Meta-Llama-3-8B-Instruct \
    --outfile llama3-8b-f16.gguf \
    --outtype f16

# Step 2: Quantize to Q4_K_M (the recommended default for CPU/edge)
./llama-quantize llama3-8b-f16.gguf llama3-8b-Q4_K_M.gguf Q4_K_M

# Step 3: Run inference
./llama-cli \
    -m llama3-8b-Q4_K_M.gguf \
    -n 256 \
    -p "Explain quantization to a 5 year old:" \
    --n-gpu-layers 33   # offload 33 layers to GPU; rest runs on CPU RAM
```

The `--n-gpu-layers` flag enables **GPU+CPU split inference**: the first $n$ layers run on the GPU (fast), remaining layers on the CPU (using system RAM). This allows running a 70 B model with 24 GB VRAM + 32 GB system RAM — unthinkable with any other stack.

### Why GGUF for Edge Deployment?

GGUF's portability is unmatched: the same binary runs on macOS (Metal), Linux (CUDA or CPU), Windows (DirectML or CUDA), and even Android/iOS via llama.cpp bindings. For edge deployment, the k-quant Q4_K_M format on a 7 B model typically results in a ~4.1 GB file that runs at 20–40 tokens/second on a modern CPU — no GPU required.

---

## bitsandbytes: The PyTorch-Native Quantization Library

bitsandbytes (bnb) provides drop-in quantized linear layers for PyTorch. It is the primary quantization backend for Hugging Face Transformers and the PEFT library (used by QLoRA).

### Architecture of bnb Linear Layers

{{fig:quant-bnb-linear-dataflow}}

Both `Linear8bitLt` and `Linear4bit` are weight-only: the dequantized GEMM still runs in FP16 hardware. The bandwidth saving is in loading weights from HBM; once on-chip (in L2 or registers), the weights are converted to FP16 before multiply-accumulate.

### Implementing a Minimal NF4 Layer From Scratch

This reconstruction shows the exact mechanism — not production-ready, but pedagogically complete:

```python
import torch
import torch.nn as nn
import numpy as np

# The 16 NF4 code points (from QLoRA paper, normalized to [-1, 1])
NF4_CODES = torch.tensor([
    -1.0,       -0.6961928,  -0.5250730,  -0.3954816,
    -0.2849375, -0.1832600,  -0.0911578,  0.0,
     0.0795761,  0.1609030,   0.2461331,   0.3379990,
     0.4407979,  0.5626170,   0.7229568,   1.0,
], dtype=torch.float32)

def quantize_nf4(weight: torch.Tensor, group_size: int = 64):
    """
    Quantize a 1-D weight tensor to NF4 per-group.
    Returns: (packed_indices, scales) where packed_indices is uint8
    with two 4-bit indices per byte.
    """
    weight = weight.float()
    n = weight.numel()
    assert n % group_size == 0
    n_groups = n // group_size
    w_groups = weight.view(n_groups, group_size)

    # Scale each group so its max absolute value maps to 1.0
    scales = w_groups.abs().max(dim=1).values  # (n_groups,)
    scales = scales.clamp(min=1e-8)
    w_norm = w_groups / scales.unsqueeze(1)    # (n_groups, group_size) in [-1, 1]

    # Find nearest NF4 code point for each weight
    # Broadcast: (n_groups, group_size, 1) vs (16,)
    codes = NF4_CODES.to(weight.device)
    dists = (w_norm.unsqueeze(-1) - codes).abs()  # (n_groups, group_size, 16)
    indices = dists.argmin(dim=-1).byte()           # (n_groups, group_size), dtype=uint8

    # Pack two 4-bit indices into one byte
    indices_flat = indices.view(-1)  # (n,)
    packed = (indices_flat[0::2] << 4) | indices_flat[1::2]  # (n//2,) uint8

    return packed, scales

def dequantize_nf4(packed: torch.Tensor, scales: torch.Tensor, group_size: int = 64):
    """Unpack NF4 indices and reconstruct FP32 weights."""
    # Unpack nibbles
    hi = (packed >> 4).byte()
    lo = (packed & 0xF).byte()
    indices_flat = torch.stack([hi, lo], dim=1).view(-1)  # interleaved back

    n = indices_flat.numel()
    n_groups = n // group_size
    codes = NF4_CODES.to(packed.device)
    w_norm = codes[indices_flat.long()].view(n_groups, group_size)

    # Re-apply group scales
    w_reconstructed = w_norm * scales.unsqueeze(1)
    return w_reconstructed.view(-1)

# --- Demo ---
torch.manual_seed(42)
w = torch.randn(256)           # simulate a weight vector (one row of a linear layer)
packed, scales = quantize_nf4(w, group_size=64)

print(f"Original size:    {w.numel() * 4} bytes (FP32)")
print(f"Quantized size:   {packed.numel()} bytes (NF4 packed)")
print(f"Scales overhead:  {scales.numel() * 4} bytes (FP32 scales)")

w_hat = dequantize_nf4(packed, scales, group_size=64)
mse = ((w - w_hat) ** 2).mean().item()
snr = (w.var() / mse).item()
print(f"MSE:  {mse:.6f}")
print(f"SNR:  {snr:.1f}  (higher is better; >100 is practically lossless)")
```

---

## Quantization-Aware Training (QAT)

Post-training quantization (PTQ) is cheap — no training required — but QAT can close the accuracy gap for aggressive bit widths (INT4 and below) by teaching the model to be robust to quantization noise during training.

### The Straight-Through Estimator

The core challenge is that the rounding operation $\operatorname{round}(\cdot)$ has zero gradient almost everywhere. QAT works by using a **straight-through estimator (STE)** in the backward pass: the forward pass rounds normally, but the backward pass pretends the rounding did not happen and passes gradients through unchanged.

For a quantized weight $q = \operatorname{round}(w/s)$, the forward pass uses $q$, and the backward pass computes:

$$
\frac{\partial \mathcal{L}}{\partial w} \approx \frac{\partial \mathcal{L}}{\partial q}
$$

This is a biased estimator, but empirically it works well and allows the model to adjust its weights so that rounding hurts less.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class STEQuantize(torch.autograd.Function):
    """
    Quantize to b bits with straight-through estimator in backward.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float, bits: int):
        # Quantize: clamp to representable range, then round
        qmin = -(2 ** (bits - 1))
        qmax =  (2 ** (bits - 1)) - 1
        x_scaled = x / scale
        x_clipped = x_scaled.clamp(qmin, qmax)
        x_quant = x_clipped.round()
        # Store nothing for backward — STE passes gradient directly
        return x_quant * scale  # dequantized immediately (fake-quant)

    @staticmethod
    def backward(ctx, grad_output):
        # STE: pass gradient through unchanged
        return grad_output, None, None

class FakeQuantLinear(nn.Linear):
    """
    A drop-in replacement for nn.Linear that applies fake-quantization
    to weights during the forward pass (simulates INT4 weight quantization).
    """
    def __init__(self, *args, bits=4, group_size=128, **kwargs):
        super().__init__(*args, **kwargs)
        self.bits = bits
        self.group_size = group_size

    def get_scale(self, w: torch.Tensor) -> torch.Tensor:
        """Per-group symmetric scale: s = max(|w|) / (2^(b-1) - 1)"""
        w_groups = w.view(-1, self.group_size)
        s = w_groups.abs().max(dim=1).values / (2 ** (self.bits - 1) - 1)
        return s.unsqueeze(1).expand_as(w_groups).reshape_as(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply fake-quantization to weights
        scale = self.get_scale(self.weight)
        w_fq = STEQuantize.apply(self.weight, 1.0, self.bits)  # simplified
        return F.linear(x, w_fq, self.bias)

# Minimal QAT training loop sketch
def qat_finetune_step(model, batch, optimizer):
    """Replace all Linear layers with FakeQuantLinear, then fine-tune."""
    optimizer.zero_grad()
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()   # gradients flow through STE
    optimizer.step()
    return loss.item()
```

### QLoRA: Quantization + Low-Rank Adaptation

QLoRA (Dettmers et al., 2023) is arguably the most impactful combination of quantization and fine-tuning. The recipe:

1. **Freeze** the base model weights in NF4 (4-bit, per-group-64, double quantization).
2. **Add LoRA adapters** (small rank-$r$ matrices $A, B$ in BF16) alongside the frozen quantized layers.
3. **Fine-tune only the LoRA adapters.** Gradients flow through the NF4-dequantized base weights using STE, then into the BF16 LoRA params.
4. **4-bit NF4 paged optimizer states**: instead of keeping FP32 Adam states for the base model, only LoRA params have optimizer states — since they are tiny ($r \ll d$), this is cheap.

The key trick: **paged optimizers** (bnb's `PagedAdamW32bit`) keep optimizer states in CPU RAM and page them to GPU only when needed, preventing OOM on long sequences.

```python
# QLoRA fine-tuning with bitsandbytes + PEFT
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import bitsandbytes as bnb
import torch

# 1. Load base model in 4-bit NF4
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    quantization_config=bnb_config,
    device_map="auto",
)

# 2. Add LoRA adapters (only these parameters will be trained)
lora_config = LoraConfig(
    r=16,                        # rank — try 8–64 depending on task
    lora_alpha=32,               # scaling: effective_lr_scaling = alpha/r
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# Trainable params: ~20M / 8B total (0.25%) — massive memory saving

# 3. Use paged optimizer to handle memory spikes
optimizer = bnb.optim.PagedAdamW32bit(
    model.parameters(),
    lr=2e-4,
    weight_decay=0.01,
)
```

LoRA and the PEFT framework are covered in depth in [PEFT I: LoRA, QLoRA, DoRA & The Adapter Family](../05-posttraining-alignment/03-peft-lora-qlora.html). The memory-efficient training angle is covered in [Memory-Efficient Training: Checkpointing, Offloading & LoRA Math](../04-kernels-efficiency/10-memory-efficient-training.html).

---

## KV-Cache Quantization

In long-context inference, the KV cache can easily rival or exceed model weight memory. For a model with 32 layers, 32 heads, head dimension 128, generating a 32 K-token sequence, each of K and V is:

$$
32 \times 32 \times 128 \times 32000 \times 2 \text{ bytes (BF16)} \approx 8.4 \text{ GB}
$$

Quantizing the KV cache to INT8 halves this to 4.2 GB; INT4 reduces it to 2.1 GB. KV quantization is more delicate than weight quantization because keys and values are computed dynamically (they change every sequence) and have heavier-tailed distributions than model weights.

### Per-Token Dynamic Quantization of KV

The standard approach (used in TensorRT-LLM, vLLM, and FlexGen) is:

1. **Per-channel (head-level) symmetric INT8:** For each head at each layer, compute the scale from the running max absolute value of that head's K or V vector at the current position.
2. **Per-token INT8 with small group size (e.g., group=32 along the token dimension):** More accurate but slightly more complex indexing.

```python
# Simplified KV cache quantization kernel (conceptual)
import torch

def quantize_kv_int8(kv: torch.Tensor):
    """
    Quantize a KV tensor of shape (batch, heads, seq_len, head_dim) to INT8.
    Per-token (per-position) symmetric quantization.
    Returns int8 tensor + FP16 per-token scale tensor.
    """
    # kv: (B, H, T, D)
    # Compute per-position max-abs across head_dim
    scale = kv.abs().max(dim=-1, keepdim=True).values / 127.0  # (B, H, T, 1)
    scale = scale.clamp(min=1e-8)
    kv_int8 = (kv / scale).round().clamp(-128, 127).to(torch.int8)
    return kv_int8, scale.to(torch.float16)

def dequantize_kv(kv_int8: torch.Tensor, scale: torch.Tensor):
    """Recover approximate FP16 KV from INT8 + scale."""
    return kv_int8.to(torch.float16) * scale

# Memory comparison for a 32-layer, 32-head, 128-dim model at 8K context
B, H, T, D = 1, 32 * 32, 8192, 128  # flattened heads
kv = torch.randn(B, H, T, D)
kv_int8, scale = quantize_kv_int8(kv.view(B, 32, 32, T, D).view(B, H, T, D))

bf16_size = kv.numel() * 2  # bytes
int8_size  = kv_int8.numel() * 1 + scale.numel() * 2
print(f"BF16 KV size: {bf16_size / 1e9:.2f} GB")
print(f"INT8 KV size: {int8_size  / 1e9:.2f} GB  ({100*int8_size/bf16_size:.0f}% of BF16)")
```

**INT4 KV cache** (used in FlexGen for offloading) quantizes per-group-20 along the token dimension. Accuracy impact on generation quality is measurable on long-context tasks; for short contexts INT4 KV is essentially lossless. PagedAttention (discussed in [PagedAttention & KV-Cache Memory Management](../04-kernels-efficiency/06-paged-attention-kv.html)) already manages KV memory in blocks; each block can independently carry a scale, making per-block quantization natural.

---

## Accuracy–Performance Tradeoffs Across the Zoo

Choosing a quantization format involves balancing three axes: model quality (perplexity or downstream task score), inference memory, and inference throughput.

### The Perplexity Cost Hierarchy

Roughly, the accuracy ordering from best to worst for a well-implemented scheme on a 7 B model is:

{{fig:quant-perplexity-quality-spectrum}}

The gap between Q4_K_M and BF16 is typically 0.1–0.5 perplexity points on Wikitext-2 for a 7 B model — usually imperceptible in downstream quality. The gap grows for smaller models (3 B and below) and for tasks requiring precise factual recall.

### Throughput and Latency

For a single-user, memory-bandwidth-bound decode scenario on an A100:

| Format       | Weights memory | Approx decode throughput (relative) |
|--------------|---------------|--------------------------------------|
| BF16         | 1× (baseline) | 1× |
| INT8 W-only  | 0.5×          | ~1.5–1.8× |
| INT4 W-only  | 0.25×         | ~2.5–3.5× |
| FP8 (W+A)   | 0.5×          | ~2× (H100 only) |
| INT8 W+A     | 0.5×          | ~1.8–2.2× |

The large spread in INT4 throughput reflects kernel quality — `exllamav2`'s hand-tuned INT4 GEMM kernels significantly outperform naive dequantize-then-FP16-GEMM approaches.

### Practical Decision Tree

{{fig:quant-deployment-decision-tree}}

!!! interview "Interview Corner"
    **Q:** An interviewer asks: "QLoRA freezes the base model in NF4 and trains LoRA adapters in BF16. During backprop, how do gradients flow through the frozen NF4 weights, and why doesn't the quantization break gradient computation?"

    **A:** The frozen base model weights are stored in NF4, but during the forward pass they are *dequantized to BF16* before the matrix multiply. Backpropagation then uses the BF16 dequantized weights in the chain rule — specifically, the gradient with respect to the LoRA adapter parameters $A$ and $B$ involves multiplication by the dequantized weight matrix, which is in BF16 and fully differentiable. The NF4 quantization itself is treated as a fixed transformation with no gradient (the weights are frozen, so there is no $\partial \mathcal{L}/\partial W_\text{base}$ to compute). The rounding in NF4 is only a concern if we wanted to update the base weights, which QLoRA does not — it only updates $A$ and $B$. This is why QLoRA does not need a straight-through estimator: the frozen weights are simply a lookup table, and the LoRA paths are fully differentiable in BF16.

---

## Summary: Format Comparison Reference

| Format | Bits/w | Granularity | Runtime | Best use case |
|--------|--------|-------------|---------|---------------|
| BF16   | 16     | —           | Any GPU | Training, highest quality |
| FP8 E4M3 | 8   | Per-tensor  | H100+   | High-throughput inference |
| INT8 W-only | 8 | Per-col   | Any GPU | Drop-in quality-preserving compression |
| INT8 W+A | 8  | Per-token   | Ampere+ | Highest server throughput |
| NF4    | 4      | Per-group-64 | Any GPU | QLoRA base model |
| INT4 GPTQ | 4  | Per-group-128 | Any GPU | Server INT4 inference |
| Q4_K_M | ~4.5  | Block-256   | CPU/GPU | Edge / llama.cpp |
| Q8_0   | 8      | Per-32      | CPU     | Fast CPU inference |

!!! key "Key Takeaways"
    - Weight-only quantization (W-only) reduces memory bandwidth and footprint without changing arithmetic type; weight+activation quantization (W+A) additionally uses lower-precision integer arithmetic units for higher compute throughput.
    - NF4 is information-theoretically optimal for normally distributed weights: its 16 code points are placed at the quantiles of $\mathcal{N}(0,1)$, minimizing expected squared error at 4 bits.
    - Double quantization compresses the per-group scale factors themselves (from FP32 to FP8), saving an additional ~0.22 bits/weight across the whole model — meaningful at 70 B scale.
    - QLoRA combines NF4 base model storage with BF16 LoRA adapters and paged optimizers, enabling full fine-tuning of a 65 B model on a single 48 GB GPU; gradients never need to pass through the NF4 rounding because the base model weights are frozen.
    - llama.cpp's GGUF k-quants (Q4_K_M, Q5_K_M, Q6_K) use block-level mixed precision with super-block and sub-block scales, offering a smooth tradeoff between file size and quality for CPU/edge deployment.
    - FP8 (E4M3) inference on H100 GPUs achieves near-BF16 quality at roughly half the memory bandwidth, but requires per-tensor or per-row scaling and benefits from SmoothQuant-style activation smoothing.
    - KV-cache quantization (INT8 or INT4 per-token) can halve or quarter KV memory overhead at long contexts; per-token scales are required because KV distributions vary dramatically across positions.
    - Quantization-aware training with the straight-through estimator (STE) allows gradient flow through the rounding operation by passing upstream gradients unchanged in the backward pass, at the cost of a biased gradient estimate.
    - As a rule of thumb: Q4_K_M / NF4 is the recommended default for 7–70 B models when maximizing quality-per-GB; INT8 W+A (SmoothQuant) is the right choice when maximizing server throughput on Ampere/Hopper GPUs.

---

!!! sota "State of the Art & Resources (2026)"
    Quantization has become the default deployment strategy for LLMs: FP8 W8A8 is the production standard on H100/H200 datacenters, INT4 weight-only (AWQ/GPTQ) dominates single-GPU server use, and GGUF k-quants (Q4_K_M) remain the go-to for CPU and edge inference — with rotation-based methods like QuaRot now enabling full W4A4 including KV cache.

    **Foundational work**

    - [Dettmers et al., *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale* (2022)](https://arxiv.org/abs/2208.07339) — introduced mixed-precision decomposition to handle activation outliers, making INT8 practical for 6.7 B+ models.
    - [Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (2023)](https://arxiv.org/abs/2305.14314) — introduced NF4, double quantization, and paged optimizers, enabling 65 B fine-tuning on a single 48 GB GPU.
    - [Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* (2022)](https://arxiv.org/abs/2210.17323) — second-order OBC-based INT4 calibration; the algorithm behind most GGUF conversions and GPTQ server deployments.

    **Recent advances (2023–2026)**

    - [Lin et al., *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration* (2023)](https://arxiv.org/abs/2306.00978) — protects 1 % of salient weights by per-channel activation scaling; MLSys 2024 Best Paper; now a first-class backend in vLLM and TGI.
    - [Ashkboos et al., *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs* (2024)](https://arxiv.org/abs/2404.00456) — Hadamard rotation removes activation outliers end-to-end, enabling full W4A4 (weights, activations, and KV cache) with <0.5 PPL loss on Llama-2-70B.
    - [Kurtic et al., *"Give Me BF16 or Give Me Death"? Accuracy-Performance Trade-Offs in LLM Quantization* (2024)](https://arxiv.org/abs/2411.02355) — 500 K+ evaluations across the Llama-3.1 family; finds FP8 W8A8 lossless, INT8 W8A8 only 1–3 % degradation, and W4A16 the most cost-efficient for synchronous serving.

    **Open-source & tools**

    - [bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) — the canonical PyTorch INT8/NF4 quantization library; powers `load_in_8bit` and `load_in_4bit` in Hugging Face Transformers.
    - [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — reference C/C++ implementation of GGUF k-quants (Q2_K through Q8_0); runs on CPU, Metal, CUDA, and DirectML with no Python dependency.
    - [NVIDIA/TransformerEngine](https://github.com/NVIDIA/TransformerEngine) — NVIDIA's FP8 (and FP4) training and inference library for Hopper/Ada/Blackwell GPUs; includes delayed scaling, amax history, and PyTorch/JAX APIs.

    **Go deeper**

    - [Hugging Face blog: *Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA* (2023)](https://huggingface.co/blog/4bit-transformers-bitsandbytes) — step-by-step walkthrough of NF4, double quantization, and QLoRA in the Transformers ecosystem.
    - [vLLM Quantization docs](https://docs.vllm.ai/en/latest/features/quantization/) — production reference covering AWQ, GPTQ, FP8 W8A8, INT8 W8A8, INT4 W4A16, and quantized KV cache in the leading open-source serving engine.

## Further Reading

- **Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale"**, NeurIPS 2022 — the mixed-precision decomposition that made INT8 practical for very large models.
- **Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs"**, NeurIPS 2023 — introduces NF4, double quantization, paged optimizers, and the QLoRA recipe.
- **Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models"**, ICML 2023 — the per-channel migration trick that enables W8A8.
- **Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"**, ICLR 2023 — the second-order OBC-based algorithm that produces the INT4 weights used by many GGUF conversions.
- **bitsandbytes library** (Tim Dettmers / Hugging Face) — `github.com/TimDettmers/bitsandbytes` — the production Python/CUDA implementation of LLM.int8() and NF4.
- **llama.cpp** (Georgi Gerganov) — `github.com/ggerganov/llama.cpp` — the canonical k-quant and GGUF implementation; the source code in `ggml-quants.c` is the reference for Q4_K_M/Q5_K_M internals.
- **Sheng et al., "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU"**, ICML 2023 — demonstrates INT4 KV cache quantization and CPU offloading.
- **NVIDIA Transformer Engine** (`github.com/NVIDIA/TransformerEngine`) — reference implementation of FP8 training and inference on Hopper GPUs, including delayed scaling and amax history.
