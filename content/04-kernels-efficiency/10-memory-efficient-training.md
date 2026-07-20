# 4.10 Memory-Efficient Training: Checkpointing, Offloading & LoRA Math

Training large language models is fundamentally a memory management problem. A 7-billion-parameter model in full bf16 precision occupies about 14 GB just for its weights — but the actual GPU memory consumed during a training step can easily be 10× that figure once you account for gradients, optimizer states, and the intermediate tensors produced by the forward pass. Understanding *where* that memory goes, and which techniques claw it back, is essential knowledge for anyone who trains or fine-tunes LLMs.

This chapter builds a precise memory budget for a training step from first principles, then covers the three main families of solutions: activation/gradient checkpointing, CPU and NVMe offloading, and parameter-efficient fine-tuning (PEFT). We quantify the tradeoffs with real numbers, show working code, and answer the interview questions that consistently trip up strong candidates.

For background on GPU memory hierarchies and how data moves between HBM, L2, and SRAM, see [GPU Architecture & The Memory Hierarchy](../01-foundations/08-gpu-architecture.html). For the distributed-training memory picture (ZeRO, FSDP), see [Distributed Training I: Data Parallelism, DDP, ZeRO & FSDP](../03-pretraining/05-distributed-data-parallel.html). The mixed-precision regime in which all of this happens is covered in [Mixed Precision, bf16 & FP8 Training](../03-pretraining/08-mixed-precision-fp8.html).

## The Full Training Memory Budget

Before optimizing anything, we need a precise accounting. At any training step, GPU HBM holds six categories of tensors:

| Category | What it contains | Typical dtype |
|---|---|---|
| **Model parameters** $\Phi$ | All weights | bf16 or fp16 |
| **Master weights** (mixed precision) | fp32 copy kept by the optimizer | fp32 |
| **Gradients** $\nabla_\Phi \mathcal{L}$ | One gradient tensor per parameter | fp16 or fp32 |
| **Optimizer states** | e.g., Adam's $m_t$, $v_t$ per parameter | fp32 |
| **Activations** | Saved forward-pass tensors needed by backward | bf16/fp16 |
| **Temporary buffers** | Workspace for kernels, all-reduce buffers | mixed |

Let $P$ be the number of parameters. In **standard Adam mixed-precision** training:

$$
M_{\text{static}} = \underbrace{2P}_{\text{bf16 weights}} + \underbrace{4P}_{\text{fp32 master}} + \underbrace{2P}_{\text{bf16 grads}} + \underbrace{4P + 4P}_{\text{Adam }m_t, v_t}
$$

$$
M_{\text{static}} = 16P \text{ bytes}
$$

So for a 7B model: $16 \times 7 \times 10^9 = 112\,\text{GB}$ — already beyond a single 80 GB H100. And we haven't counted activations yet.

### The Activation Memory Equation

For a transformer trained with a batch of $B$ sequences of length $T$, with $L$ layers, hidden dimension $d$, and $h$ attention heads:

During the **forward pass**, each transformer block needs to store activations for the backward pass. The dominant contributors per layer, per token are:

- **Attention QKV projections**: $3 \times d$ (one fp16 tensor per projection)
- **Attention scores and softmax output**: $B \times h \times T \times T$ (the full $T \times T$ attention matrix per head)
- **Post-attention projections**: $B \times T \times d$
- **MLP intermediate**: $B \times T \times 4d$ (for a standard 4× MLP expansion)
- **Layer norms**: $B \times T \times d$

A common rule of thumb aggregates this to roughly:

$$
M_{\text{act}} \approx \underbrace{12 \times B \times T \times d \times L}_{\text{elements}} \times (\text{bytes/element})
$$

so in fp16/bf16 (2 bytes per element) this is $M_{\text{act}} \approx 24\,B\,T\,d\,L$ bytes. The coefficient 12 counts stored elements per token per layer; it is a rounded-down version of Megatron-LM's exact per-layer accounting $34\,s\,b\,h + 5\,a\,s^2\,b$ bytes (with $s=T$, $b=B$, $h=d$, $a$ = number of heads), whose first term is $\approx 17$ elements/token/layer. The second term $5\,a\,s^2\,b$ is the $T\times T$ attention-score matrix; it is absent when FlashAttention is used (it never materializes the scores), leaving only the term linear in $T$. The exact coefficient depends on architecture details (e.g., GQA reduces the attention term). The key observation is that activation memory scales as $B \times T \times L$ — it can dwarf the static weight cost for long sequences or large batches.

!!! example "Worked Example: LLaMA-7B Memory Budget"

    LLaMA-7B has: $L=32$ layers, $d=4096$, $h=32$ heads.

    **Static memory (standard Adam, bf16 weights):**
    $$16P = 16 \times 7 \times 10^9 \approx 112\,\text{GB}$$

    **Activation memory, batch 1, sequence length 2048:**
    $$M_{\text{act}} \approx \underbrace{12 \times 1 \times 2048 \times 4096 \times 32}_{\approx 3.2\text{B elements}} \times 2\,\text{bytes/element} \approx 6.4\,\text{GB}$$

    So with one GPU and standard training: $112 + 6.4 \approx 118\,\text{GB}$.
    An H100 SXM has 80 GB — this doesn't fit even with one sample per GPU.

    **With gradient checkpointing (no activations stored):**
    $$M_{\text{total}} \approx 112 + \sqrt{L} \times \text{(one layer's activations)} \approx 112 + \sqrt{32}\times 0.2 \approx 112 + 1.1 \approx 113\,\text{GB}$$
    (One layer's activations $\approx 6.4/32 \approx 0.2$ GB and $\sqrt{32}\approx 5.7$, so the $\sqrt{L}$ checkpoints cost $\approx 1.1$ GB. The simpler "store only each block's input" strategy costs $L\,B\,T\,d\times 2$ bytes $\approx 0.5$ GB.)

    Still too large — we need ZeRO or PEFT as well.

    **With QLoRA rank-16 (q, k, v, o projections; see §4.5 below):**

    Each adapted matrix adds $r(d_{\text{in}}+d_{\text{out}})$ parameters. With 4 attention
    matrices per layer (each $4096\times4096$) over $L=32$ layers:
    $$|\theta_{\text{LoRA}}| = 16 \times (4096 + 4096) \times 4 \times 32 \approx 16.8\text{M params}$$
    Adapter weights (bf16, 2 bytes/param): $2 \times 16.8\text{M} \approx 34\,\text{MB} \approx 0.03\,\text{GB}$.
    Frozen quantized base: $\approx 3.5\,\text{GB (4-bit)}$.
    Adam optimizer states on the adapters only (8 bytes/param):
    $$8 \times 16.8\text{M} \approx 0.13\,\text{GB}$$
    Total: $3.5 + 0.03 + 0.13 \approx$ **3.7 GB** — easily fits in a 6 GB consumer GPU.

## Activation Checkpointing: Recompute vs. Store

Activation checkpointing (also called **gradient checkpointing**) is the oldest and most universally applicable memory-reduction technique. The idea, introduced in the systems literature as "rematerialization" and popularized for deep learning by Chen et al. (2016) in *Training Deep Nets with Sublinear Memory Cost*, is simple:

> Do not store every intermediate tensor during the forward pass. Instead, recompute them on demand during the backward pass.

### The Recompute–Storage Tradeoff

Without checkpointing, storing all activations for an $L$-layer network costs $O(L)$ memory but zero extra compute. With full recomputation, you store only the input to each layer, paying one extra forward pass: memory drops to $O(1)$ (or $O(\sqrt{L})$ with optimal placement), compute goes up by roughly 33%.

The memory–compute tradeoff is:

$$
M_{\text{act}} = O\!\left(\frac{L}{k}\right), \quad \text{FLOPs}_{\text{extra}} = O(k)
$$

where $k$ is the number of "checkpoints" (layer boundaries where you store a tensor). Choosing $k = \sqrt{L}$ minimizes the product, giving $O(\sqrt{L})$ memory and $O(\sqrt{L})$ extra cost — the classic sublinear memory result.

In practice, modern frameworks let you checkpoint at the granularity of an entire transformer block, so the +33% compute overhead estimate is approximately correct for full-checkpointing of all blocks.

### PyTorch Activation Checkpointing in Practice

```python
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

# -----------------------------------------------------------------------
# Simple demonstration: a single transformer block with checkpointing.
# We wrap the forward in torch.utils.checkpoint.checkpoint so PyTorch
# will NOT save intermediate activations, and will recompute them during
# backward instead.
# -----------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """A minimal causal transformer block (MHA + FFN + layer norms)."""

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + attn_out
        # FFN with residual
        x = x + self.ffn(self.ln2(x))
        return x


class CheckpointedModel(nn.Module):
    """Wraps a stack of transformer blocks and applies gradient checkpointing."""

    def __init__(self, n_layers: int, d_model: int, n_heads: int,
                 use_checkpointing: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.use_checkpointing = use_checkpointing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            if self.use_checkpointing and x.requires_grad:
                # checkpoint() replaces the saved activations with a
                # recomputation graph.  use_reentrant=False is recommended
                # in PyTorch >= 2.0 for compatibility with compiled graphs.
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x


# -----------------------------------------------------------------------
# Memory comparison: illustrate the difference at inference scale.
# -----------------------------------------------------------------------
def measure_peak_memory(n_layers: int, use_checkpointing: bool,
                         batch: int = 2, seq_len: int = 512,
                         d_model: int = 1024, n_heads: int = 8) -> float:
    """Returns peak HBM usage in MB after one forward+backward pass."""
    torch.cuda.reset_peak_memory_stats()
    model = CheckpointedModel(n_layers, d_model, n_heads, use_checkpointing).cuda()
    model = model.to(torch.bfloat16)
    x = torch.randn(batch, seq_len, d_model, device="cuda",
                     dtype=torch.bfloat16, requires_grad=True)
    loss = model(x).mean()
    loss.backward()
    return torch.cuda.max_memory_allocated() / 1e6  # MB


if __name__ == "__main__":
    for ckpt in [False, True]:
        mb = measure_peak_memory(n_layers=24, use_checkpointing=ckpt)
        print(f"Checkpointing={ckpt}: peak memory = {mb:.1f} MB")
    # Typical output:
    # Checkpointing=False: peak memory = 3421.3 MB
    # Checkpointing=True:  peak memory = 1108.7 MB  (~3x reduction)
```

### Selective Recomputation

Full checkpointing of all blocks is conservative. Modern implementations (e.g., Megatron-LM's `--recompute-granularity selective`) let you choose *which* operations to recompute. The cost-benefit analysis:

- **Attention QK softmax** (the $T \times T$ matrix): large memory footprint, cheap to recompute since it is memory-bound not compute-bound.
- **MLP GELU activations**: moderate size, fast recompute.
- **Layer norm outputs**: small, fast recompute — often not worth the overhead.

The rule of thumb: recompute anything whose memory cost exceeds its FLOPs cost. FlashAttention (see [FlashAttention I: IO-Awareness & The Online Softmax](../04-kernels-efficiency/02-flash-attention-1.html)) is itself an extreme form of selective recomputation — it does not materialize the $N \times N$ attention matrix at all, recomputing softmax statistics on-the-fly, saving $O(N^2)$ memory per layer.

```python
# Selective checkpointing: only checkpoint the attention sub-block,
# not the cheaper FFN.  Saves ~60% of attention-related activation memory
# at a small recompute cost.

from torch.utils.checkpoint import checkpoint

class SelectiveCheckpointBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def _attn_only(self, x: torch.Tensor) -> torch.Tensor:
        """The sub-computation we want to recompute in backward."""
        normed = self.ln1(x)
        out, _ = self.attn(normed, normed, normed, need_weights=False)
        return x + out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Recompute attention activations; store FFN activations normally.
        if x.requires_grad:
            x = checkpoint(self._attn_only, x, use_reentrant=False)
        else:
            x = self._attn_only(x)
        return x + self.ffn(self.ln2(x))
```

## CPU and NVMe Offloading

When GPU memory is exhausted even after checkpointing, the next option is to spill tensors to cheaper, larger memory tiers.

### The Memory Hierarchy Revisited


{{fig:memeff-memory-hierarchy-tiers}}


PCIe bandwidth is ~50× slower than HBM bandwidth. This means CPU offloading is only viable if the tensor being offloaded is *not* needed every step, or the compute on GPU is long enough to hide the transfer latency.

### DeepSpeed ZeRO-Infinity and Offload

DeepSpeed's ZeRO-Offload and ZeRO-Infinity implement optimizer-state and parameter offloading. The key insight is that **Adam optimizer states** ($m_t$, $v_t$, and the fp32 master weights) are updated only once per step, after the gradient has been reduced. They are read once (to compute the weight update) and written once (with the new values). This access pattern tolerates the latency of a PCIe transfer because the Adam update itself is memory-bound and can be executed on CPU cheaply.

```python
# deepspeed_offload_config.json — enable ZeRO-3 with CPU offload
# Drop this into your DeepSpeed config to offload optimizer states + params.
```

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto"
  },
  "bf16": { "enabled": true },
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 16
}
```

With ZeRO-3 + CPU offload, the GPU holds only a working subset of parameters and optimizer states at any time, allowing effective training of 30B+ models on a single GPU — at the cost of significantly reduced throughput (roughly 5–10× slowdown versus in-memory training due to PCIe bandwidth saturation).

### Gradient Accumulation as an Offloading Strategy

Gradient accumulation is not traditionally called "offloading," but it achieves the same goal of separating the **memory cost** of a large effective batch from the **peak memory** of a single forward–backward pass. With accumulation steps $A$:

$$
M_{\text{peak}} = M(\text{micro-batch size} = B/A) + M_{\text{grad buffer}}
$$

The gradient buffer costs $2P$ bytes (fp16), held across all accumulation steps. But the activation peak at any step is determined by $B/A$, not $B$. This is cheap (no data movement), at the cost of $A$ forward–backward passes per parameter update.

```python
# Gradient accumulation — manual implementation in pure PyTorch.
# Using a micro-batch of 1 to simulate an effective batch of 8.

model.train()
optimizer.zero_grad()

ACCUMULATION_STEPS = 8
for step, (x, y) in enumerate(dataloader):
    x, y = x.cuda(), y.cuda()

    # Optionally use autocast for mixed precision.
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(x)
        # Scale loss by accumulation factor so gradients are averaged, not summed.
        loss = criterion(logits, y) / ACCUMULATION_STEPS

    loss.backward()  # Accumulates .grad on parameters — no optimizer step yet.

    if (step + 1) % ACCUMULATION_STEPS == 0:
        # Gradient clipping before the optimizer step.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
```

## The Math Behind LoRA and Why PEFT Slashes Memory

Parameter-efficient fine-tuning (PEFT) methods attack the memory problem from a different angle: instead of reducing the cost of training all parameters, they *freeze* most parameters and only train a small adapter. [PEFT I: LoRA, QLoRA, DoRA & The Adapter Family](../05-posttraining-alignment/03-peft-lora-qlora.html) covers the practical usage in depth; here we focus on the memory mathematics.

### The LoRA Factorization

LoRA (Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021) approximates a weight update $\Delta W \in \mathbb{R}^{d \times k}$ as a product of two low-rank matrices:

$$
\Delta W = B A, \quad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times k},\; r \ll \min(d, k)
$$

Only $A$ and $B$ are trained; the original $W$ is frozen. The number of trainable parameters per adapted weight matrix is:

$$
|\theta_{\text{LoRA}}| = r(d + k) \quad \text{vs.} \quad dk \text{ for full fine-tuning}
$$

The compression ratio is:

$$
\rho = \frac{r(d+k)}{dk} \approx \frac{r}{d} \quad \text{for } d \approx k
$$

For a 7B model with $d = k = 4096$ and rank $r = 16$: $\rho \approx 16/4096 = 0.4\%$. Only 0.4% of each weight matrix's parameters are trained.

### Memory Impact: Exact Accounting

With LoRA, the memory budget changes dramatically:

| Term | Full fine-tuning | LoRA ($r=16$) |
|---|---|---|
| **Frozen weights** (bf16) | $2P$ bytes (trainable) | $2P$ bytes (frozen, no grad) |
| **Adapter weights** (bf16) | — | $2 \cdot |\theta_{\text{LoRA}}|$ |
| **Gradients** | $2P$ bytes | $2 \cdot |\theta_{\text{LoRA}}|$ bytes |
| **Optimizer states** (Adam fp32) | $8P$ bytes | $8 \cdot |\theta_{\text{LoRA}}|$ bytes |

For a frozen weight tensor, PyTorch does not allocate a gradient buffer, so **frozen parameters contribute 0 bytes of gradient or optimizer state**. The savings are enormous: if LoRA covers all linear layers in a 7B model with rank 16, the optimizer state shrinks from $\sim$56 GB (fp32 Adam) to roughly $56 \times 0.004 = 0.22$ GB.

The frozen base model's weights still occupy $2P$ bytes, but they require no gradient storage. With 4-bit quantization of the base model (QLoRA), these compress further to $\frac{P}{2}$ bytes:

$$
M_{\text{QLoRA}} = \underbrace{\frac{P}{2}}_{\text{4-bit base}} + \underbrace{2 \cdot |\theta_{\text{LoRA}}|}_{\text{bf16 adapters}} + \underbrace{8 \cdot |\theta_{\text{LoRA}}|}_{\text{Adam states on adapters}}
$$

For LLaMA-7B with rank 16 covering all four attention projections (q, k, v, o), $|\theta_{\text{LoRA}}| \approx 16.8$M: approximately $3.5 + 0.03 + 0.13 \approx 3.7$ GB — fitting in a 6 GB GPU.

### LoRA From Scratch: A Full Implementation

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with a LoRA side path.

    During forward: output = x @ W^T + (x @ A^T) @ B^T * scale
    where W is frozen and only A, B are updated.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,   # LoRA scaling hyper-param; scale = alpha/rank
        dropout: float = 0.05,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.rank         = rank
        self.scale        = alpha / rank  # Hu et al. use this to keep LR independent of r

        # Frozen base weight (will be loaded from pretrained model)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        self.bias_param = nn.Parameter(
            torch.zeros(out_features), requires_grad=False
        ) if bias else None

        # Trainable LoRA matrices
        # A is initialized from N(0, 1/sqrt(r)) to give unit-variance init.
        # B is initialized to zero so ΔW = 0 at the start of training.
        self.lora_A = nn.Parameter(
            torch.empty(rank, in_features)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank)
        )
        self.lora_dropout = nn.Dropout(dropout)

        # Kaiming init for A (matches standard linear init scale)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int = 16,
                    alpha: float = 32.0) -> "LoRALinear":
        """Convert an existing nn.Linear to LoRALinear, preserving its weights."""
        bias = linear.bias is not None
        lora = cls(linear.in_features, linear.out_features,
                   rank=rank, alpha=alpha, bias=bias)
        with torch.no_grad():
            lora.weight.copy_(linear.weight)
            if bias:
                lora.bias_param.copy_(linear.bias)
        return lora

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard linear (no grad flows here since weight.requires_grad=False)
        base_out = F.linear(x, self.weight, self.bias_param)

        # LoRA side path: (x @ A^T) @ B^T, scaled
        # shapes: x[..., in_features] -> A[rank, in] -> B[out, rank]
        lora_out = F.linear(
            F.linear(self.lora_dropout(x), self.lora_A),  # [..., rank]
            self.lora_B                                     # [..., out]
        )
        return base_out + self.scale * lora_out

    def merge_weights(self) -> nn.Linear:
        """
        Merge the LoRA update into W for efficient inference.
        Returns a standard nn.Linear with merged weights.
        """
        merged_weight = self.weight + self.scale * (self.lora_B @ self.lora_A)
        linear = nn.Linear(self.in_features, self.out_features,
                           bias=self.bias_param is not None)
        with torch.no_grad():
            linear.weight.copy_(merged_weight)
            if self.bias_param is not None:
                linear.bias.copy_(self.bias_param)
        return linear

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"rank={self.rank}, scale={self.scale:.3f}")


# -----------------------------------------------------------------------
# Utility: inject LoRA into all attention Q, V projections of a GPT-style
# model.  This is the most common recipe (K and O are often frozen).
# -----------------------------------------------------------------------

def inject_lora(model: nn.Module, rank: int = 16, alpha: float = 32.0,
                target_modules: tuple = ("q_proj", "v_proj")) -> nn.Module:
    """
    Walk the module tree and replace named Linear sub-modules
    whose name ends with any string in target_modules with LoRALinear.
    Freezes all non-LoRA parameters.
    """
    # First, freeze everything
    for param in model.parameters():
        param.requires_grad_(False)

    # Replace target projections with LoRA versions
    for name, module in list(model.named_modules()):
        for target in target_modules:
            if name.endswith(target) and isinstance(module, nn.Linear):
                # Navigate to parent and set child
                parts = name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                lora_module = LoRALinear.from_linear(module, rank=rank, alpha=alpha)
                setattr(parent, parts[-1], lora_module)
                break  # Found target for this module; move on

    # Report trainable parameter count
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"LoRA injection complete: {trainable:,} / {total:,} params trainable "
          f"({100 * trainable / total:.3f}%)")
    return model
```

### QLoRA: Quantized Base + LoRA Adapters

QLoRA (Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, 2023) combines two ideas:

1. **NF4 (NormalFloat4)**: A 4-bit data type optimized for normally-distributed weights. Instead of linear quantization, NF4 assigns quantization levels at equal-probability points of a standard normal distribution, minimizing quantization error for the typical weight distribution.

2. **Double quantization**: The NF4 quantization constants themselves are quantized to 8 bits, saving roughly 0.37 bits per parameter on top of the base NF4 savings.

3. **Paged optimizer**: Uses NVIDIA's unified memory to page optimizer states to CPU DRAM on demand, preventing OOM crashes from memory spikes.

The base model is loaded in 4-bit and never updated; gradients flow only through the fp16/bf16 LoRA adapters using a straight-through-style mechanism.

```python
# QLoRA with bitsandbytes and HuggingFace Transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import torch

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",         # NormalFloat4 — best for LLM weights
    bnb_4bit_use_double_quant=True,    # Quantize the quantization constants too
    bnb_4bit_compute_dtype=torch.bfloat16,  # Activations/LoRA in bf16
)

# Load the model in 4-bit; the base weights are frozen automatically
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

# Prepare for k-bit training (adds gradient checkpointing + layer-norm fixes)
from peft import prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# Attach LoRA adapters to Q and V projections
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],   # expand to k_proj, o_proj for more capacity
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Typical output: trainable params: 4,194,304 || all params: 6,738,415,616
# || trainable%: 0.0623
```

## Practical Memory Accounting: A Step-by-Step Recipe

When you encounter an OOM error, the correct debugging strategy is systematic, not random. Here is the procedure we use in practice.

### Step 1: Measure the Static Footprint

```python
import torch

def memory_snapshot(label: str = "") -> None:
    """Print current and peak allocated GPU memory."""
    alloc = torch.cuda.memory_allocated() / 1e9
    peak  = torch.cuda.max_memory_allocated() / 1e9
    res   = torch.cuda.memory_reserved() / 1e9
    print(f"[{label}] allocated={alloc:.2f}GB  peak={peak:.2f}GB  reserved={res:.2f}GB")

# Profile a forward pass step-by-step:
torch.cuda.reset_peak_memory_stats()
memory_snapshot("start")

model = model.cuda()
memory_snapshot("after model load")  # Should be ~2P bytes for bf16 weights

x = x.cuda()
memory_snapshot("after input")

with torch.no_grad():
    out = model(x)
memory_snapshot("after forward (no_grad)")  # No activation retention

del out
torch.cuda.empty_cache()

out = model(x)                   # With grad enabled
memory_snapshot("after forward (with_grad)")  # Activations retained here!

out.sum().backward()
memory_snapshot("after backward")  # Grads allocated, activations freed
```

### Step 2: Use torch.cuda.memory\_stats for Detailed Breakdown

```python
# Dump the full memory stats dictionary to identify the largest allocation.
stats = torch.cuda.memory_stats()
for key, val in sorted(stats.items(), key=lambda kv: -kv[1]):
    if val > 0:
        print(f"  {key}: {val / 1e6:.1f} MB")
```

### Step 3: The Decision Tree


{{fig:memeff-oom-decision-tree}}


## Combining Techniques: The Memory Stack

These techniques are not mutually exclusive. In practice, large-scale fine-tuning stacks several simultaneously. The table below shows approximate peak memory for a 70B-parameter Llama-style model with a batch of 1 at sequence length 2048:

| Configuration | Approx. peak GPU memory |
|---|---|
| Vanilla full training (fp16 Adam) | $16 \times 70\,\text{B} \approx 1120\,\text{GB}$ |
| + ZeRO-3 (8 GPUs, no offload) | $1120 / 8 \approx 140\,\text{GB/GPU}$ (still needs 2×A100) |
| + ZeRO-3 + CPU offload | $\sim 40\,\text{GB/GPU}$ (1 H100, slow) |
| LoRA ($r=16$, all attn layers) | $\sim 30\,\text{GB/GPU}$ (2P frozen + small optimizer) |
| QLoRA ($r=16$, 4-bit base) | $\sim 15\,\text{GB/GPU}$ (4-bit base + bf16 adapters) |
| QLoRA + checkpointing | $\sim 12\,\text{GB/GPU}$ |

!!! warning "Common pitfall: forgetting to freeze properly"

    When using LoRA, many engineers rely on PEFT's `get_peft_model()` to freeze base weights automatically. If you manually set `param.requires_grad = False` on the base model *after* wrapping with PEFT, you may inadvertently freeze the adapter weights too. Always inspect `model.print_trainable_parameters()` and verify the count matches your expectation (approximately `2 * rank * (d_in + d_out) * num_adapted_layers`).

    A related pitfall: if gradient checkpointing is enabled and `use_reentrant=True` (the old default), operations that don't accept keyword arguments will error. Use `use_reentrant=False` in PyTorch 2.0+.

## Optimizer State Memory Reduction

Optimizer state is often the largest single memory consumer in full fine-tuning. Beyond LoRA, several optimizer designs intrinsically reduce state:

### Adafactor

Adafactor (Shazeer & Stern, 2018) replaces the full second-moment matrix with a factored rank-1 approximation:

$$
V_t \approx r_t \cdot c_t^\top, \quad r_t \in \mathbb{R}^{d},\; c_t \in \mathbb{R}^{k}
$$

This reduces optimizer state from $O(dk)$ (Adam's $v_t$ for a $d \times k$ weight) to $O(d + k)$. For a 4096×4096 linear layer, that is 16.7 million → 8,192 values: a 2,048× compression. Adafactor also omits the first moment $m_t$ (relying on relative step size), further halving the state. The tradeoff is that Adafactor can be less stable for fine-tuning on small datasets; many practitioners use it for pretraining but fall back to Adam for RLHF.

### 8-bit Adam

Dettmers et al. introduced 8-bit Adam in *8-bit Optimizers via Block-wise Quantization* (2022). The fp32 $m_t$ and $v_t$ are stored in 8-bit integers with per-block scaling factors. Memory for optimizer states drops from $8P$ to roughly $2P$ bytes — a 4× reduction — with negligible accuracy loss.

```python
# pip install bitsandbytes
import bitsandbytes as bnb

optimizer = bnb.optim.Adam8bit(
    [p for p in model.parameters() if p.requires_grad],
    lr=2e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
)
# Drop-in replacement for torch.optim.Adam; uses ~4x less optimizer state memory.
```

### Gradient Accumulation and FP16 Gradients

When gradient accumulation is used, gradients are held for multiple micro-batches. Using fp16 (rather than fp32) gradient buffers halves the $2P$ bytes to $P$ bytes. PyTorch's `autocast` + `GradScaler` handles this automatically in mixed-precision mode:

```python
scaler = torch.cuda.amp.GradScaler()  # Maintains a fp16 loss scale

for micro_batch_x, micro_batch_y in batches:
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = model(micro_batch_x, labels=micro_batch_y).loss / N_ACCUM
    # Backward in fp16; GradScaler prevents underflow
    scaler.scale(loss).backward()

scaler.step(optimizer)   # Unscales grads, then steps
scaler.update()
optimizer.zero_grad()
```

!!! interview "Interview Corner"

    **Q:** You have an 80 GB A100 and want to fine-tune a 13B-parameter model. Describe the memory budget for standard Adam training, explain which categories dominate, and list techniques in order of impact that would allow the training to fit.

    **A:** The static memory breakdown for a 13B model is:
    - bf16 weights: $2 \times 13 \times 10^9 \approx 26\,\text{GB}$
    - fp32 master copy: $4 \times 13 \times 10^9 \approx 52\,\text{GB}$
    - fp32 Adam $m_t + v_t$: another $\approx 104\,\text{GB}$

    Total static alone is $\approx 182\,\text{GB}$ — more than twice the 80 GB budget, without any activations. Optimizer states ($8P$ bytes) dominate for a well-optimized Adam.

    Techniques in decreasing memory impact:
    1. **LoRA (rank 16)**: reduces optimizer states from $8P$ to $8 \times 0.004P \approx 0.03P$. This alone saves ~99.6% of optimizer memory, bringing static to roughly $\approx 28\,\text{GB}$.
    2. **QLoRA (4-bit base)**: compresses frozen weights from 26 GB to $\approx 6.5\,\text{GB}$, total $\approx 8\,\text{GB}$.
    3. **Gradient checkpointing**: further reduces activation peak, at ~33% compute cost.
    4. **8-bit Adam** (without LoRA): halves optimizer state to $\approx 52\,\text{GB}$ total static — borderline without ZeRO.
    5. **ZeRO-3**: shards all state across GPUs; requires multiple GPUs.

    For a single 80 GB A100, QLoRA is the canonical answer.

## Key Implementation Patterns and Pitfalls

### Memory Pinning for CPU Offload

When offloading tensors to CPU DRAM, **pinned (page-locked) memory** dramatically accelerates PCIe transfers by allowing DMA without CPU involvement:

```python
# Allocate CPU tensor in pinned memory for fast GPU<->CPU transfer
cpu_tensor = torch.empty_like(gpu_tensor, device="cpu", pin_memory=True)

# Non-blocking copy from GPU to CPU
cpu_tensor.copy_(gpu_tensor, non_blocking=True)

# ... do GPU compute on other tensors while transfer completes ...

# When needed again, copy back
gpu_tensor.copy_(cpu_tensor, non_blocking=True)
torch.cuda.synchronize()  # Ensure transfer is complete before using gpu_tensor
```

DeepSpeed's ZeRO-Offload uses this pattern internally, pre-fetching the next layer's parameters while the current layer's backward pass is running.

### Gradient Checkpointing with Compiled Models

`torch.compile` and gradient checkpointing interact non-trivially. The recomputed sub-graph is traced separately from the outer graph. As of PyTorch 2.2+, the recommended pattern is:

```python
# Compile the model AFTER enabling checkpointing, not before.
model = CheckpointedModel(n_layers=32, d_model=4096, n_heads=32,
                          use_checkpointing=True)
# torch.compile will trace through checkpoint boundaries correctly
# with use_reentrant=False
model = torch.compile(model, mode="reduce-overhead")
```

If you compile first and then toggle checkpointing, the compiled graph may not include the recompute branches, and you will silently fall back to full activation storage.

### The `no_grad` vs. `detach` Distinction

A common confusion: `torch.no_grad()` prevents creation of the autograd graph but does not free existing activation tensors. `tensor.detach()` severs the graph at a specific point. When implementing custom checkpointing, always use:

```python
# Correct: save only the input, discard intermediate activations
saved_input = input.detach()  # Severs autograd graph; no grad fn stored
# ... run forward normally (intermediates are freed) ...
# In backward: rerun from saved_input (now re-attaches to the graph)
```

!!! sota "State of the Art & Resources (2026)"
    Memory-efficient training has matured into a layered stack: activation checkpointing, ZeRO-stage offloading, and LoRA/QLoRA compose cleanly and together enable fine-tuning of 70B+ models on consumer hardware. Active research frontiers in 2024–2026 include gradient-space low-rank projections (GaLore) for full-parameter pretraining on tight budgets and weight-decomposed adaptation (DoRA) for higher-fidelity LoRA updates.

    **Foundational work**

    - [Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021)](https://arxiv.org/abs/2106.09685) — introduces the rank-decomposed adapter that now underpins virtually all PEFT workflows.
    - [Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* (2020)](https://arxiv.org/abs/1910.02054) — defines ZeRO stages 1/2/3 and the theoretical memory analysis used throughout this chapter.

    **Recent advances (2023–2026)**

    - [Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (2023)](https://arxiv.org/abs/2305.14314) — NF4 4-bit base + bf16 LoRA adapters + paged optimizer; enables 65B fine-tuning on a single 48 GB GPU.
    - [Rajbhandari et al., *ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning* (2021)](https://arxiv.org/abs/2104.07857) — extends ZeRO to NVMe offloading, enabling models beyond GPU + CPU DRAM capacity.
    - [Liu et al., *DoRA: Weight-Decomposed Low-Rank Adaptation* (2024)](https://arxiv.org/abs/2402.09353) — decomposes weights into magnitude + direction; ICML 2024 oral, consistently outperforms LoRA on instruction-tuning benchmarks.
    - [Zhao et al., *GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection* (2024)](https://arxiv.org/abs/2403.03507) — projects gradients (not weights) to a low-rank subspace, enabling full-parameter pretraining of a 7B model on a 24 GB GPU; ICML 2024 oral.

    **Open-source & tools**

    - [huggingface/peft](https://github.com/huggingface/peft) — canonical Python library implementing LoRA, QLoRA, DoRA, IA³, and other PEFT methods with HuggingFace Transformers integration.
    - [bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) — provides 8-bit/4-bit quantization kernels (NF4, LLM.int8()) and 8-bit Adam/AdamW optimizers used by QLoRA.
    - [deepspeedai/DeepSpeed](https://github.com/deepspeedai/DeepSpeed) — production ZeRO-1/2/3, ZeRO-Offload, and ZeRO-Infinity implementations; drop-in JSON config as shown in this chapter.

    **Go deeper**

    - [HuggingFace blog: *Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA* (2023)](https://huggingface.co/blog/4bit-transformers-bitsandbytes) — practical walkthrough of loading and fine-tuning with 4-bit NF4, with Colab notebooks.
    - [Sebastian Raschka, *Practical Tips for Finetuning LLMs Using LoRA* (2023)](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) — eight evidence-based takeaways on rank selection, target modules, learning rate, and merging, drawn from systematic ablations.

## Further Reading

- Chen, T., Xu, B., Zhang, C., Guestrin, C. — *Training Deep Nets with Sublinear Memory Cost* (2016). The original activation checkpointing paper for deep networks.
- Hu, E., et al. — *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022. The foundational PEFT paper.
- Dettmers, T., et al. — *QLoRA: Efficient Finetuning of Quantized LLMs*, NeurIPS 2023. Combines NF4 quantization, double quantization, and paged optimizers.
- Dettmers, T., et al. — *8-bit Optimizers via Block-wise Quantization*, ICLR 2022.
- Rajbhandari, S., et al. — *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* (DeepSpeed), SC 2020. Covers ZeRO-1/2/3 and the theoretical memory analysis.
- Rajbhandari, S., et al. — *ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning*, SC 2021. Extends ZeRO to NVMe offloading.
- Shazeer, N., Stern, M. — *Adafactor: Adaptive Learning Rates with Sublinear Memory Cost*, ICML 2018.
- Dao, T., et al. — *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS 2022. Eliminates $O(T^2)$ attention materialization.
- HuggingFace PEFT library (github.com/huggingface/peft) — canonical implementation of LoRA, QLoRA, and adapter variants.

!!! key "Key Takeaways"

    - The full training memory budget is $16P$ bytes per parameter for standard Adam mixed-precision; optimizer states ($8P$) typically dominate — not weights.
    - Activation memory scales as $B \times T \times L$ and can exceed static costs for long sequences; the activation memory equation gives roughly $12 \cdot B \cdot T \cdot d \cdot L$ elements ($\approx 24\,B\,T\,d\,L$ bytes in fp16/bf16).
    - Gradient checkpointing trades ~33% extra compute for $O(\sqrt{L})$ activation memory; FlashAttention achieves a similar win for the $O(T^2)$ attention term.
    - CPU offloading (ZeRO-Offload, ZeRO-Infinity) works because optimizer states are accessed once per step, tolerating PCIe latency.
    - LoRA with rank $r$ reduces trainable parameters to $\rho \approx r/d$ of the full matrix count, eliminating nearly all optimizer state for frozen layers.
    - QLoRA = 4-bit base (NF4) + bf16 LoRA adapters + paged optimizer; enables 70B fine-tuning on a single consumer GPU.
    - 8-bit Adam provides a 4× optimizer state reduction with no change to architecture or training procedure.
    - Techniques compose: QLoRA + gradient checkpointing + gradient accumulation is the standard single-GPU fine-tuning stack.
    - When debugging OOM: identify whether the failure is in forward (activation issue) or during parameter update (optimizer state issue), then apply the appropriate remedy.
