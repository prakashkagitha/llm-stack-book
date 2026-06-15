# 5.3 PEFT I: LoRA, QLoRA, DoRA & The Adapter Family

In [Supervised Fine-Tuning & Instruction Tuning](../05-posttraining-alignment/01-sft-instruction-tuning.html) we fine-tuned a model the obvious way: load every weight, compute a loss, backpropagate, and update *all* of the parameters with an optimizer. For a 7-billion-parameter model in bf16 that is already a budget problem. The weights are 14 GB. Adam keeps two more states per parameter — first and second moment — and those, plus a master fp32 copy of the weights, push the *training* footprint to roughly 16 bytes per parameter: about 112 GB for a 7B model, before a single activation is stored. A 70B model is out of reach of anything but a multi-node cluster. And you pay this for *every* fine-tune: a separate 14 GB checkpoint per customer, per task, per experiment.

**Parameter-Efficient Fine-Tuning (PEFT)** is the family of methods that sidesteps this. The bet is simple and, empirically, correct: you do not need to move all the weights to adapt a pretrained model to a downstream task. You can freeze the giant pretrained backbone and train a *tiny* number of new parameters — often well under 1% of the total — and recover most, sometimes all, of the quality of full fine-tuning. The dominant member of this family, by a wide margin, is **LoRA** (Low-Rank Adaptation), and its quantization-aware cousin **QLoRA** is what made fine-tuning a 65B model on a single consumer GPU a reality.

This chapter is the deep dive on the *low-rank* branch of PEFT: the math of $W + BA$, where to apply it, how alpha scaling and initialization actually work, merging adapters back into the base weights, the four-bit machinery of QLoRA (NF4, double quantization, paged optimizers), and the 2023–2025 refinements — DoRA, rsLoRA, LoRA+, VeRA. We finish with a from-scratch LoRA implementation you can read end to end, and with how to *serve* hundreds of LoRAs on one GPU. The prompt/prefix-tuning and model-merging branches of PEFT live in the next chapter, [PEFT II: Prompt/Prefix Tuning, IA3, Model Merging & Soups](../05-posttraining-alignment/04-peft-prompt-merging.html).

## The intrinsic-dimension intuition: why low rank works at all

{{fig:lora}}

Before the mechanics, the *why*. The empirical premise behind LoRA, articulated by Hu et al. (*LoRA: Low-Rank Adaptation of Large Language Models*, 2021) and grounded in earlier work by Aghajanyan et al. on the *intrinsic dimension* of fine-tuning, is that **the change a fine-tune makes to a pretrained weight matrix has low effective rank.**

Think about what fine-tuning does. The pretrained model already encodes a vast amount of general knowledge. Adapting it to "answer politely in a customer-support voice" or "format outputs as JSON" is a small *delta* on top of that knowledge, not a wholesale relearning. Formally, if a layer's pretrained weight is $W_0 \in \mathbb{R}^{d \times k}$, full fine-tuning learns an update $\Delta W$ and uses $W_0 + \Delta W$. The hypothesis is that $\Delta W$, though it is a $d \times k$ matrix and could in principle have rank $\min(d,k)$, in practice lies very close to a matrix of rank $r \ll \min(d,k)$.

Any rank-$r$ matrix factors as a product of two thin matrices:

$$
\Delta W = B A, \qquad B \in \mathbb{R}^{d \times r}, \quad A \in \mathbb{R}^{r \times k}, \quad r \ll \min(d, k).
$$

Instead of learning the $d \times k$ entries of $\Delta W$ directly, we learn the $d r + r k$ entries of $B$ and $A$. For a typical attention projection with $d = k = 4096$ and $r = 16$, that is $4096^2 = 16{,}777{,}216$ parameters reduced to $2 \cdot 4096 \cdot 16 = 131{,}072$ — a **128×** reduction for that layer. The factorization is the entire trick. Everything else is engineering.

{{fig:lora-fullft-vs-lora-parallel-path}}

It is worth being precise about a subtlety: LoRA assumes the *update* is low rank, **not** that the pretrained weight $W_0$ is low rank. $W_0$ is full rank and we keep it exactly. We only constrain the *adaptation*. This is why LoRA can match full fine-tuning on many tasks while a naive "compress the model to low rank" approach would destroy it.

## LoRA mechanics: forward pass, alpha scaling, and initialization

### The forward pass

Take any linear layer in the network whose forward map is $h = W_0 x$ (we drop the bias for clarity; biases are tiny and usually left trainable or frozen as-is). LoRA replaces it with

$$
h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r}\, B A\, x.
$$

The frozen base path $W_0 x$ is untouched. In parallel we add a *bottleneck*: $x$ (dimension $k$) is projected **down** to rank $r$ by $A$, then **up** to dimension $d$ by $B$, and the result is scaled by $\alpha / r$. The two paths are summed. During training only $A$ and $B$ receive gradients; $W_0$ is frozen, so no optimizer state is allocated for it and — crucially — its gradient is never computed.

Two practical consequences fall out immediately. First, because $W_0$ is frozen, we can store it in a *lower precision* (this is exactly the door QLoRA walks through). Second, because the LoRA path is just an extra matrix multiply, at inference time we can *fold* it into $W_0$ and pay **zero** extra latency — more on merging below.

### The $\alpha/r$ scaling factor — what it is and why it exists

The scalar $\frac{\alpha}{r}$ in front of $BA$ is the single most misunderstood knob in LoRA. Here is the honest explanation.

$\alpha$ (alpha) is a constant you choose; $r$ is the rank. The factor decouples the *magnitude* of the adaptation from the *rank*. Suppose you tuned everything beautifully at $r = 8$ and now want to try $r = 16$ to give the adapter more capacity. If there were no scaling, doubling $r$ would roughly double the typical norm of $BA x$ (more rank-1 terms summing up), which effectively changes your learning rate and forces you to re-tune. By dividing by $r$, the *effective scale* of the update stays roughly constant as you sweep $r$, so the learning rate you found transfers. The authors recommend setting $\alpha$ once (commonly $\alpha = 2r$, i.e. an effective scale of 2, or $\alpha = r$ for a scale of 1) and then sweeping $r$ freely.

A common practitioner convention is "set $\alpha = 2r$." It is not magic — it just means the effective scale $\alpha/r = 2$. If you ever see a config with $r=16, \alpha=32$, that is exactly this. We will revisit this in the **rsLoRA** section, where Kalajdzievski (2023) shows the $1/r$ scaling is actually *suboptimal* at high rank and $1/\sqrt{r}$ is the principled choice.

!!! warning "Common pitfall"
    Changing `r` and `alpha` together in the same ratio is *not* a no-op even though the scale is constant, because rank changes capacity. And changing `alpha` alone with `r` fixed is *exactly* like changing the learning rate of the LoRA path. People burn days re-tuning learning rate when they have implicitly already changed it by editing alpha. Pin the effective scale `alpha/r` first, then treat rank as a capacity knob and learning rate as a learning-rate knob.

### Initialization — and why one matrix starts at zero

At the very start of training we need the adapted model to be *identical* to the pretrained model, so that $\Delta W = BA = 0$. If both $A$ and $B$ started random, the model would be perturbed before learning anything and the initial loss would spike. LoRA achieves $BA = 0$ at init by setting **one** of the two matrices to zero:

- $A$ is initialized with small random values (Kaiming/He uniform, as for any linear layer's weight).
- $B$ is initialized to **all zeros**.

So $BA = B \cdot A = 0 \cdot A = 0$ at step 0: the LoRA path contributes nothing and the forward pass equals the base model exactly. As training proceeds, gradients flow into $B$ (because $A$ is nonzero, $\partial(BAx)/\partial B \neq 0$) and into $A$ (once $B$ becomes nonzero). Why not the reverse — $B$ random, $A$ zero? Either choice gives $BA = 0$, but with $A=0$ the gradient to $B$ is initially zero ($\partial(BAx)/\partial B = (Ax)^\top = 0$), so $B$ would be stuck for the first step. Initializing $A$ random and $B$ zero gives an immediate gradient signal to $B$. Both conventions appear in the wild; the $A$-random, $B$-zero convention is the original and the most common.

!!! example "Worked example: parameter and memory savings on a 7B model"
    Consider a LLaMA-style 7B model: hidden size $d = 4096$, 32 layers. Suppose we apply LoRA with rank $r = 16$ to the four attention projections ($W_q, W_k, W_v, W_o$), each $4096 \times 4096$.

    **Trainable LoRA params per projection:** $B$ is $4096 \times 16$ and $A$ is $16 \times 4096$, so $2 \cdot 4096 \cdot 16 = 131{,}072$ params. With 4 projections × 32 layers: $4 \times 32 \times 131{,}072 \approx 16.8\text{M}$ trainable params.

    Against the full model of $\sim 6.7\text{B}$ params, that is **about 0.25%** trainable. If we also adapt the three MLP projections (gate, up, down — each roughly $4096 \times 11008$), the count rises but typically stays **under 1%**.

    **Optimizer-memory effect.** Full fine-tuning with Adam in mixed precision costs $\approx 16$ bytes/param (fp32 master weight 4, two Adam moments 8, plus the bf16 weight 2 and bf16 grad 2). For 6.7B that is $\approx 107$ GB just for weights+states. LoRA pays the 16 bytes/param *only on the 16.8M trainable params* — $\approx 270$ MB — plus the **frozen** base weights, which need only their bf16 copy ($\approx 13.4$ GB) and **no** gradient, **no** moments, **no** fp32 master copy. The optimizer-state line item drops from $\sim 80$ GB to well under 1 GB.

## Where to apply LoRA, and how merging works

### Which layers? The targeting decision

LoRA can wrap *any* linear layer. The original paper found that adapting only the attention projections — and within those, often just the query and value projections $W_q, W_v$ — was enough to match full fine-tuning on their benchmarks, at minimal cost. That made "attention-only LoRA" the early default.

The modern (2024–2025) consensus, driven by the QLoRA ablations and a great deal of community experience, is broader: **apply LoRA to *all* linear layers** — the four attention projections *and* the three MLP projections (gate/up/down in a SwiGLU block; see [The Transformer Block: Norms, Residuals, MLPs & Activations](../02-transformer/06-transformer-block.html)). This "LoRA everywhere" setting closes most of the remaining gap to full fine-tuning, because the MLP holds the majority of a transformer's parameters and a lot of its task-relevant capacity. The cost is still tiny in absolute terms.

What you do **not** wrap: LayerNorm/RMSNorm scales (you may train these directly — they are vectors, almost free), the token embedding and the LM head (sometimes trained directly if the task needs new tokens), and biases. A common recipe is "LoRA on all linears + train the norms."

{{fig:lora-transformer-block-attach-points}}

### Merging: zero-overhead inference

Because the LoRA contribution is linear and parallel to the base path, the trained adapter can be **merged** into the base weights once training is done:

$$
W_{\text{merged}} = W_0 + \frac{\alpha}{r} B A.
$$

After merging, the layer is again a single ordinary linear with weight $W_{\text{merged}}$, indistinguishable from a fully fine-tuned model at inference time. There is **no extra matmul, no extra memory, no added latency** — this is LoRA's killer property over earlier adapter methods (Houlsby et al., 2019), which inserted *serial* bottleneck modules that could not be folded away and thus added inference cost.

Merging is exact in full precision. The subtlety is **merging into a quantized base** (QLoRA): if $W_0$ was stored in 4-bit and you merge a bf16 adapter, you must either (a) dequantize $W_0$ to bf16, add $\frac{\alpha}{r}BA$, and keep the merged weight in bf16/fp16 (giving up the 4-bit memory win at inference), or (b) merge then *re-quantize*, which introduces a small additional quantization error. Most production paths that need 4-bit serving keep the adapter *unmerged* and serve it dynamically (see the multi-LoRA serving section).

!!! tip "Practitioner tip"
    Keep an *unmerged* copy of your adapter forever. The merged checkpoint is convenient for a single-task deployment, but the 30–100 MB adapter file is the thing you actually want to version, share, A/B test, and hot-swap. Adapters are composable artifacts; merged weights are not.

## A from-scratch LoRA implementation

Let us build LoRA with no library magic — just PyTorch — so the mechanics are unambiguous. This is a drop-in `nn.Linear` replacement plus the helper that walks a model and swaps layers. It is correct and runnable. (For the broader memory story this implements, see [Memory-Efficient Training: Checkpointing, Offloading & LoRA Math](../04-kernels-efficiency/10-memory-efficient-training.html).)

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """A frozen base Linear with a trainable low-rank adapter in parallel.

    Forward computes:  h = W0 @ x  +  (alpha / r) * B @ (A @ x)
    Only A and B are trainable; W0 (and bias) are frozen.
    """

    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32,
                 dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = r
        # The effective scale alpha/r decouples magnitude from rank.
        self.scaling = alpha / r

        # --- Frozen base weight (this is W0). Keep the original tensor. ---
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)        # freeze: no grad, no optimizer state

        # --- Trainable low-rank factors. ---
        # A: (r, in)  initialized random (Kaiming);  B: (out, r) initialized ZERO.
        # So B @ A = 0 at init  ->  the adapter starts as a no-op.
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False  # tracks whether the adapter is folded into base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)                      # frozen W0 @ x  (+ bias)
        if self.merged:
            return base_out                          # adapter already folded in
        # Low-rank path: down-project with A, up-project with B, scale.
        lora_out = F.linear(self.dropout(x), self.lora_A)   # (..., r)
        lora_out = F.linear(lora_out, self.lora_B)          # (..., out)
        return base_out + self.scaling * lora_out

    @torch.no_grad()
    def merge(self):
        """Fold (alpha/r) * B @ A into the base weight for zero-overhead inference."""
        if self.merged:
            return
        delta = self.scaling * (self.lora_B @ self.lora_A)  # (out, in)
        self.base.weight.add_(delta.to(self.base.weight.dtype))
        self.merged = True

    @torch.no_grad()
    def unmerge(self):
        """Reverse merge() — useful for hot-swapping adapters on a shared base."""
        if not self.merged:
            return
        delta = self.scaling * (self.lora_B @ self.lora_A)
        self.base.weight.sub_(delta.to(self.base.weight.dtype))
        self.merged = False


def inject_lora(model: nn.Module, target_names=("q_proj", "k_proj", "v_proj",
                                                 "o_proj", "gate_proj",
                                                 "up_proj", "down_proj"),
                r: int = 16, alpha: int = 32, dropout: float = 0.05):
    """Walk the module tree and replace matching nn.Linear layers with LoRALinear."""
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear) and child_name in target_names:
                setattr(module, child_name, LoRALinear(child, r, alpha, dropout))
    return model


def mark_only_lora_trainable(model: nn.Module):
    """Freeze everything, then unfreeze only the LoRA factors. Returns trainable count."""
    n_trainable = 0
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad_(True)
            n_trainable += p.numel()
        else:
            p.requires_grad_(False)
    return n_trainable
```

A tiny end-to-end sanity check — overfit a single batch and confirm only the adapter moves:

```python
torch.manual_seed(0)

# A toy "model": two linears we will adapt.
model = nn.Sequential(
    nn.Linear(64, 64, bias=False),
    nn.ReLU(),
    nn.Linear(64, 8, bias=False),
)
# Name the children so inject_lora can find them.
model[0].__class__.__name__  # still nn.Linear; we target by attribute name instead:

# Manually wrap (the helper targets by child attribute name; here we wrap directly):
model[0] = LoRALinear(model[0], r=4, alpha=8)
model[2] = LoRALinear(model[2], r=4, alpha=8)
n = mark_only_lora_trainable(model)
total = sum(p.numel() for p in model.parameters())
print(f"trainable {n} / {total}  ({100*n/total:.2f}%)")   # ~ a few % on this toy

# Snapshot a frozen base weight to prove it does NOT change.
W0_before = model[0].base.weight.detach().clone()

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
x = torch.randn(32, 64)
y = torch.randn(32, 8)
for step in range(200):
    opt.zero_grad()
    loss = F.mse_loss(model(x), y)
    loss.backward()
    opt.step()
print(f"final loss {loss.item():.4f}")
assert torch.equal(W0_before, model[0].base.weight), "base weight must stay frozen!"

# Merging must not change the output (within float tolerance).
model.eval()
with torch.no_grad():
    out_unmerged = model(x)
    model[0].merge(); model[2].merge()
    out_merged = model(x)
print("merge max-diff:", (out_unmerged - out_merged).abs().max().item())  # ~1e-6
```

The two assertions are the heart of LoRA's contract: **the base weight never moves during training**, and **merging is output-preserving**. If you ever fork LoRA code, these are the first two tests to write.

### The gradient flow, made explicit

It helps to see the backward pass. With $h = W_0 x + s\,BAx$ where $s = \alpha/r$, and an upstream gradient $g = \partial \mathcal{L}/\partial h$, the LoRA gradients are

$$
\frac{\partial \mathcal{L}}{\partial B} = s\, g\, (A x)^\top, \qquad
\frac{\partial \mathcal{L}}{\partial A} = s\, (B^\top g)\, x^\top, \qquad
\frac{\partial \mathcal{L}}{\partial W_0} = \text{(not computed — frozen)}.
$$

Two things to notice. First, computing these requires the *activation* $x$ and the cheap intermediate $Ax$, but **not** a gradient w.r.t. the giant $W_0$, which is why the backward pass is cheap. Second, the input $x$ to the layer must still be saved for the backward of the *frozen* path's contribution to upstream layers — LoRA reduces *optimizer* and *gradient* memory dramatically but does not by itself reduce *activation* memory. For that you combine LoRA with gradient checkpointing (see [Memory-Efficient Training](../04-kernels-efficiency/10-memory-efficient-training.html)).

## QLoRA: fine-tuning a 65B model on one GPU

LoRA shrinks the *optimizer and gradient* memory. But the frozen base weights still sit in memory — 13 GB for 7B in bf16, 130 GB for 65B. **QLoRA** (Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, 2023) attacks that last term: store the frozen base in **4-bit**, train the LoRA adapter in bf16 on top. Suddenly a 65B base fits in ~35 GB, and you can fine-tune it on a single 48 GB GPU. QLoRA is, more than any other single method, why fine-tuning frontier-scale models became democratized. It rests on three innovations.

### 1. NF4 — the 4-bit NormalFloat data type

Standard 4-bit integer quantization splits a weight's range into 16 evenly spaced bins. But neural-network weights are not uniformly distributed — they are approximately **zero-mean Gaussian**. Spending equal numbers of bins on the dense center and the sparse tails wastes precision where the data actually is.

**NF4 (4-bit NormalFloat)** is an *information-theoretically optimal* quantization for normally distributed data. The 16 quantization levels are placed at the *quantiles* of a standard normal distribution, so each bin holds roughly the same *probability mass* rather than the same *width*. Concretely: estimate the quantiles of $\mathcal{N}(0,1)$ that split it into 16 equal-mass regions, take the bin midpoints as the codebook, and make it symmetric with an exact zero. Because weights are first normalized to $[-1, 1]$ (by dividing by their block-wise absmax), they match the standard-normal assumption well, and NF4 reproduces them with markedly less error than INT4 at the same 4 bits. The dequantization is just a 16-entry lookup. See [Quantization II: INT4/INT8/FP8, GGUF, bitsandbytes & QAT](../04-kernels-efficiency/08-quantization-formats-qat.html) for the broader quantization toolbox.

```python
import torch

# The 16 NF4 codebook values (bin midpoints at the quantiles of N(0,1),
# normalized to [-1, 1] with an exact zero). These are constants in bitsandbytes.
NF4_CODEBOOK = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
     0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
     0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
     0.7229568362236023, 1.0,
])

def quantize_nf4_block(w_block: torch.Tensor):
    """Quantize one block of weights to NF4. Returns 4-bit codes + the fp scale."""
    absmax = w_block.abs().max()                  # block-wise scale (one fp number)
    w_norm = w_block / (absmax + 1e-8)            # normalize to ~[-1, 1]
    # Nearest codebook entry for each weight -> 4-bit index in [0, 15].
    dist = (w_norm.unsqueeze(-1) - NF4_CODEBOOK).abs()
    codes = dist.argmin(dim=-1).to(torch.uint8)   # store these 4-bit codes
    return codes, absmax

def dequantize_nf4_block(codes: torch.Tensor, absmax: torch.Tensor):
    """Reconstruct bf16 weights from 4-bit codes and the block scale (a lookup)."""
    return NF4_CODEBOOK[codes.long()] * absmax

# Demo: NF4 beats INT4 on Gaussian data.
torch.manual_seed(0)
w = torch.randn(4096)                              # one block of Gaussian weights
codes, scale = quantize_nf4_block(w)
w_hat = dequantize_nf4_block(codes, scale)
nf4_err = (w - w_hat).pow(2).mean().sqrt()

# Plain symmetric INT4 for comparison.
s = w.abs().max() / 7
w_int4 = (w / s).round().clamp(-7, 7) * s
int4_err = (w - w_int4).pow(2).mean().sqrt()
print(f"NF4 RMSE {nf4_err:.4f}   INT4 RMSE {int4_err:.4f}")  # NF4 noticeably lower
```

### 2. Double quantization — quantizing the quantization constants

NF4 uses a separate fp32 `absmax` scale **per block** (block size 64 in QLoRA). That is one 32-bit number for every 64 weights — an overhead of $32/64 = 0.5$ bits *per weight*, which is large when the weights themselves are 4 bits. **Double quantization (DQ)** quantizes those scales too: the per-block fp32 absmax values are themselves grouped (256 of them) and quantized to 8-bit, with one fp32 second-level scale per group. This drops the scale overhead from 0.5 bits/param to roughly $8/64 + 32/(64\cdot 256) \approx 0.127$ bits/param — saving on the order of 0.4 bits per parameter, which over a 65B model is several gigabytes. It is a small idea with a real payoff at scale.

### 3. Paged optimizers — surviving the memory spikes

Even with a 4-bit base and a tiny adapter, long sequences cause **activation-memory spikes** during the backward pass that can momentarily exceed GPU memory and OOM the job. QLoRA borrows the OS idea of **paging**: the optimizer states live in a *paged* allocation backed by CPU RAM, using NVIDIA's unified memory. When the GPU is about to run out, pages are automatically evicted to CPU and paged back in when needed — exactly like virtual memory swapping to disk. The job survives a transient spike instead of crashing, at the cost of some PCIe traffic during the spike. This is what makes single-GPU fine-tuning of very large models *robust*, not just nominally possible.

```python
# QLoRA in practice with HuggingFace + bitsandbytes + PEFT (the real path).
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                       # store the frozen base in 4-bit
    bnb_4bit_quant_type="nf4",               # NormalFloat-4 (not plain int4)
    bnb_4bit_use_double_quant=True,          # double quantization on the scales
    bnb_4bit_compute_dtype=torch.bfloat16,   # matmuls dequant to bf16 on the fly
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", quantization_config=bnb_config, device_map="auto",
)
model = prepare_model_for_kbit_training(model)   # enable grad ckpt, cast norms, etc.

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],   # LoRA everywhere
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# -> trainable params: ~40M || all params: ~6.7B || trainable%: ~0.6
# Train with paged optimizer:  optim="paged_adamw_8bit"  in your TrainingArguments.
```

The key mental model for QLoRA: **the frozen base is stored in NF4, but every matmul dequantizes the relevant weights to bf16 on the fly** (the `compute_dtype`). The 4-bit form is a *storage* format, not a *compute* format. The LoRA adapter is always full-precision bf16 and is where all the learning happens. Quantization error in the frozen base is "absorbed" by the adapter, which learns on top of the slightly-noisy base — this is a large part of why QLoRA matches 16-bit LoRA quality despite the aggressive compression.

!!! warning "Common pitfall"
    QLoRA saves *training* memory by quantizing the base, but a merged QLoRA adapter does **not** give you a 4-bit *inference* model for free. If you merge the bf16 adapter into a dequantized base, you get a bf16 model (full size). If you want 4-bit serving, keep the adapter unmerged and serve it on top of the quantized base, or quantize the *merged* model afresh with a PTQ method (GPTQ/AWQ — see [Quantization I](../04-kernels-efficiency/07-quantization-ptq.html)). Do not assume "I trained with QLoRA" means "I have a 4-bit deployable model."

## The LoRA family: DoRA, rsLoRA, LoRA+, VeRA

LoRA's simplicity invited a wave of refinements. The four below are the ones a 2025 practitioner should know by name and mechanism.

### rsLoRA — fixing the scaling factor at high rank

Recall the $\alpha/r$ scaling. Kalajdzievski (*A Rank-Stabilized Scaling Factor for LoRA*, 2023) showed analytically that dividing by $r$ over-shrinks the adapter as rank grows: the variance of $BAx$ scales such that the *gradient* through the adapter vanishes with large $r$, so increasing rank stops helping and can even hurt. The fix — **rank-stabilized LoRA (rsLoRA)** — is to scale by $1/\sqrt{r}$ instead:

$$
h = W_0 x + \frac{\alpha}{\sqrt{r}} B A x.
$$

This keeps the magnitude of the adapter's contribution and its gradient stable across ranks, so high-rank LoRA ($r = 64, 128, 256$) actually delivers the extra capacity you paid for. If you are using small ranks ($r \le 16$) the difference is minor; if you are pushing rank to chase quality, switch to rsLoRA. It is a one-line change (`use_rslora=True` in PEFT).

### LoRA+ — different learning rates for A and B

Hayou et al. (*LoRA+: Efficient Low Rank Adaptation*, 2024) observed that $A$ and $B$ play *asymmetric* roles — $B$ starts at zero and must grow, $A$ starts populated — and that using the *same* learning rate for both is suboptimal in a way that gets worse as the model width grows. The fix is to give $B$ a **larger** learning rate than $A$, by a fixed ratio $\lambda$ (often $\lambda = 16$):

$$
\eta_B = \lambda\, \eta_A, \qquad \lambda > 1.
$$

This typically improves both convergence speed and final quality at no extra parameter cost — it is purely an optimizer-grouping change. It composes with everything else (you can do LoRA+ on a QLoRA run).

### DoRA — decomposing weight magnitude and direction

**DoRA (Weight-Decomposed Low-Rank Adaptation;** Liu et al., 2024) is the most important post-LoRA refinement. The insight: a weight update changes both the *direction* of weight vectors and their *magnitude*, and full fine-tuning tends to make large changes to magnitude that plain LoRA struggles to express. DoRA decomposes each weight matrix into a magnitude vector $m$ and a direction matrix $V$:

$$
W = m \cdot \frac{V}{\lVert V \rVert_c},
$$

where $\lVert \cdot \rVert_c$ is the column-wise norm. DoRA then trains the magnitude $m$ (a small trainable vector, one scalar per column) **directly**, while adapting the *direction* with a LoRA-style low-rank update:

$$
W' = m \cdot \frac{W_0 + \frac{\alpha}{r} B A}{\lVert W_0 + \frac{\alpha}{r} B A \rVert_c}.
$$

By separating "how big" from "which way," DoRA's learning dynamics more closely resemble full fine-tuning, and it consistently beats LoRA at the *same* rank — often letting you reach LoRA-$r$ quality at $r/2$. The cost is a slightly more expensive forward (the column-norm and the magnitude rescale) and a few extra trainable parameters ($m$). DoRA can be merged just like LoRA. Here is the core forward in code:

```python
import torch, torch.nn as nn, torch.nn.functional as F

class DoRALinear(nn.Module):
    """DoRA: train direction via low-rank A,B and magnitude m directly."""
    def __init__(self, base: nn.Linear, r=16, alpha=32):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scaling = alpha / r
        out_f, in_f = base.weight.shape
        self.lora_A = nn.Parameter(torch.empty(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        # Magnitude m = the column-wise norm of the pretrained weight (init so W'=W0).
        with torch.no_grad():
            self.m = nn.Parameter(base.weight.norm(dim=0, keepdim=True))  # (1, in)

    def forward(self, x):
        # Effective directional weight = W0 + scaled low-rank update.
        delta = self.scaling * (self.lora_B @ self.lora_A)          # (out, in)
        V = self.base.weight + delta                                 # direction (unnormalized)
        V_norm = V.norm(dim=0, keepdim=True) + 1e-8                  # column norms (1, in)
        W_eff = self.m * (V / V_norm)                                # rescale to magnitude m
        return F.linear(x, W_eff)                                    # (no separate base path)
```

At init, `delta = 0` (because `lora_B = 0`), so `V = W0`, `V_norm = ||W0||_c`, `m = ||W0||_c`, and therefore `W_eff = ||W0||_c * (W0 / ||W0||_c) = W0` — the adapter is again a perfect no-op at step 0, as it must be.

### VeRA — sharing frozen random matrices across layers

**VeRA (Vector-based Random Matrix Adaptation;** Kopiczko et al., 2024) pushes parameter-efficiency to an extreme. Observe that in LoRA the matrices $A$ and $B$ are large; what if we *froze* a single pair of **random** $A$ and $B$ — shared across *all* layers — and only trained two tiny *scaling vectors* $d$ and $b$ that re-weight the rows/columns?

$$
h = W_0 x + \Lambda_b\, B\, \Lambda_d\, A\, x,
$$

where $B, A$ are random, frozen, and shared, and $\Lambda_d = \operatorname{diag}(d)$, $\Lambda_b = \operatorname{diag}(b)$ are the only trainable parameters (two vectors per layer). Because the big matrices are frozen random projections (and can even be regenerated from a seed rather than stored), VeRA's trainable footprint can be **10–100× smaller** than LoRA's at comparable quality on many tasks. It trades a little peak quality for an enormous reduction in adapter size — attractive when you must store thousands of per-user adapters. The lineage here connects to random-projection ideas (Johnson–Lindenstrauss) and to the intrinsic-dimension framing we opened with.

| Method | Trainable params | Key idea | When to reach for it |
| --- | --- | --- | --- |
| LoRA | $2 d r$ per layer | Low-rank $W+BA$ | Default; well-supported everywhere |
| QLoRA | LoRA + 4-bit base | NF4 + DQ + paging | Big model, small GPU |
| rsLoRA | same as LoRA | $1/\sqrt{r}$ scaling | High rank ($r \ge 64$) |
| LoRA+ | same as LoRA | $\eta_B > \eta_A$ | Free convergence boost |
| DoRA | LoRA + magnitude $m$ | Split magnitude/direction | Want full-FT quality at low rank |
| VeRA | two vectors/layer | Frozen shared random $A,B$ | Thousands of tiny adapters |

## Serving many LoRAs on one GPU

Here is the deployment scenario that makes LoRA a *systems* topic and not just a training trick. You run a platform with one base model and **hundreds or thousands** of fine-tuned variants — one per customer, per task, per language. Loading a full merged model per variant is impossible: 1,000 customers × 14 GB = 14 TB. But each *adapter* is only 30–100 MB. The opportunity: keep **one** copy of the base model on the GPU and *swap adapters per request*.

The challenge is that requests in a single batch want *different* adapters. Naively you would have to run one batch per adapter, destroying throughput (see [Continuous Batching & Request Scheduling](../07-inference-serving/02-continuous-batching.html)). The solution, introduced by **S-LoRA** (Sheng et al., 2023) and now standard in vLLM and friends, is a **batched, multi-adapter kernel**: compute the shared base matmul $W_0 X$ once for the whole batch, then apply *each request's own* $BA$ in a single grouped/segmented GEMM that gathers the right adapter per row.

{{fig:lora-multi-lora-batched-serving}}

Three systems ideas make this fast:

1. **Separate the base and adapter compute.** The base GEMM is large, dense, and shared — run it once at full efficiency. The adapter contribution is small and per-request — run it with a specialized grouped kernel (often called a **Batched/Segmented Gather-Matmul**, e.g. the `bgmv`/`sgmv` kernels in vLLM and Punica). Punica (Chen et al., 2024) contributed the SGMV kernel that makes the per-adapter step nearly free.
2. **A unified adapter memory pool.** Hold many adapters in a paged GPU buffer; the *active* ones stay resident, *cold* ones spill to CPU and page back on demand — the same paging idea as QLoRA's optimizer, applied to adapter weights at serve time.
3. **Rank-heterogeneous batching.** Different adapters can have different ranks. A good multi-LoRA kernel pads/segments by rank so a batch can mix an $r=8$ and an $r=64$ adapter without serializing.

In practice you enable this with a flag — e.g. vLLM's `enable_lora=True` plus a per-request `lora_request` — and the engine handles the gather/scatter. The result: thousands of customers, *near* base-model throughput, sub-second adapter hot-swap, and a memory bill of (one base) + (a small adapter cache). This is the economic backbone of "fine-tuned model" SaaS offerings.

!!! interview "Interview Corner"
    **Q:** A candidate says "LoRA reduces memory because we only store a small adapter." An interviewer pushes: *during training*, what specifically gets smaller, what does **not**, and why can QLoRA fine-tune a 65B model on one 48 GB GPU when plain LoRA cannot?

    **A:** LoRA's main *training* saving is on **optimizer state and gradients**, not on the base weights. Because the base is frozen, you compute no gradient for it and store no Adam moments or fp32 master copy for it — those 12–14 bytes/param vanish for the 99%+ of frozen parameters; you pay them only on the <1% trainable adapter. What does **not** shrink under plain LoRA: (1) the frozen base weights themselves still sit in memory (e.g. 13 GB in bf16 for 7B, ~130 GB for 65B), and (2) activation memory is essentially unchanged, since you still forward through the full network — that is why you combine LoRA with gradient checkpointing. QLoRA's extra move is to quantize the **frozen base to 4-bit (NF4)**, cutting the base term ~4× (130 GB → ~35 GB for 65B), with double-quantization shaving the scale overhead and a paged optimizer surviving transient activation spikes. So: plain LoRA can't fit 65B because the *base* alone overflows 48 GB; QLoRA fits because the base is now 4-bit while the adapter and its optimizer states stay tiny in bf16.

!!! note "Aside: LoRA's regularization side-effect"
    Constraining the update to rank $r$ is also a *regularizer*. On small fine-tuning datasets, full fine-tuning can over-fit and degrade the base model's general capabilities ("catastrophic forgetting"). LoRA's limited capacity often makes it *more* robust to forgetting and to small-data over-fitting — a quality benefit, not just an efficiency one. The flip side: on tasks that genuinely require large weight changes (e.g. learning a new domain from scratch, or extending to a new language), low rank can *underfit*, and full fine-tuning or a higher rank wins. Match the rank to how big the required update actually is.

## Choosing hyperparameters in practice

A compact decision guide, distilled from the QLoRA ablations and broad community experience (treat numbers as starting points, not laws):

- **Target modules:** start with *all* linear layers (attention + MLP). Attention-only is fine for cheap experiments but leaves quality on the table.
- **Rank $r$:** 8–16 for most instruction/style tasks; 32–64 when the task is harder or the dataset large; 64+ only with rsLoRA. More rank is not free quality — it is more capacity that can over-fit.
- **Alpha:** set the effective scale $\alpha/r \approx 2$ (so $\alpha = 2r$) and leave it; do not co-vary it casually with rank.
- **Dropout:** 0.05–0.1 on the LoRA path helps on smaller datasets; 0 is fine on large ones.
- **Learning rate:** LoRA tolerates **higher** LR than full fine-tuning — on the order of $1\text{e}{-4}$ to $3\text{e}{-4}$ is common, vs. $\sim 1\text{e}{-5}$ for full FT — because you are training a small, well-conditioned subspace. Use LoRA+ ($\eta_B = 16\,\eta_A$) for a free bump.
- **Quantize when memory-bound:** reach for QLoRA (NF4 + double-quant + paged AdamW-8bit) the moment the base doesn't fit; the quality cost vs. 16-bit LoRA is small in practice.
- **Upgrade path:** if LoRA underfits at your best rank, try **DoRA** (better quality per rank) before jumping to full fine-tuning.

This connects directly to the data-formatting and templating choices in [Chat Templates, Data Formatting & Sequence Packing](../05-posttraining-alignment/02-chat-templates-packing.html) — PEFT changes *which* parameters move, not *what* you train on, so your SFT data pipeline is identical.

!!! key "Key Takeaways"
    - **LoRA freezes the base and learns a low-rank update** $\Delta W = \frac{\alpha}{r} BA$ in parallel; only $A$ (random init) and $B$ (zero init) train, so the adapter is a no-op at step 0 and the base never moves.
    - The premise is **low intrinsic rank of the *update***, not of the weight: $W_0$ stays full-rank and exact; only the adaptation is constrained.
    - **$\alpha/r$ is an effective-scale knob**, not magic. Pin $\alpha/r$ (e.g. 2), sweep $r$ as a capacity knob; editing $\alpha$ alone is just changing the LoRA learning rate.
    - LoRA's main *training* saving is **optimizer state + gradients** on the frozen base (no Adam moments, no fp32 master, no base grad); activation memory is unchanged, so pair it with gradient checkpointing.
    - **Merging** folds $BA$ into $W_0$ for **zero inference overhead** — LoRA's edge over serial adapters; keep the unmerged adapter as the portable artifact.
    - **QLoRA** = 4-bit **NF4** frozen base + **double quantization** of scales + **paged optimizer**; it makes single-GPU fine-tuning of 65B-class models routine, with the bf16 adapter absorbing quantization error.
    - Know the family: **rsLoRA** ($1/\sqrt{r}$ scaling for high rank), **LoRA+** ($\eta_B > \eta_A$), **DoRA** (split magnitude/direction, best quality per rank), **VeRA** (shared frozen random matrices, tiny adapters).
    - **Multi-LoRA serving** (S-LoRA/Punica) runs the shared base GEMM once and applies per-request adapters with a segmented kernel — thousands of tuned variants on one base at near-base throughput.

!!! sota "State of the Art & Resources (2026)"
    LoRA and QLoRA remain the dominant parameter-efficient fine-tuning approaches in production as of 2026, with DoRA and rsLoRA the leading quality upgrades; multi-adapter serving (S-LoRA/vLLM) has made LoRA-based SaaS economically standard at scale.

    **Foundational work**

    - [Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021)](https://arxiv.org/abs/2106.09685) — the original method: freeze the base, train a rank-$r$ delta $BA$ in parallel; adapter merges for zero inference overhead.
    - [Aghajanyan et al., *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning* (2020)](https://arxiv.org/abs/2012.13255) — empirical grounding for why fine-tuning updates are low-rank: the optimization landscape has far lower intrinsic dimension than the parameter count.

    **Recent advances (2023–2026)**

    - [Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (2023)](https://arxiv.org/abs/2305.14314) — NF4 4-bit frozen base + double quantization + paged optimizer; made single-GPU fine-tuning of 65B models routine.
    - [Liu et al., *DoRA: Weight-Decomposed Low-Rank Adaptation* (2024)](https://arxiv.org/abs/2402.09353) — decomposes weights into magnitude and direction, trains magnitude directly; consistently beats LoRA at the same rank (ICML 2024 oral).
    - [Kalajdzievski, *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA* (rsLoRA, 2023)](https://arxiv.org/abs/2312.03732) — replaces $1/r$ scaling with $1/\sqrt{r}$ so high-rank adapters ($r \ge 64$) actually deliver the extra capacity.
    - [Hayou et al., *LoRA+: Efficient Low Rank Adaptation of Large Models* (2024)](https://arxiv.org/abs/2402.12354) — sets a higher learning rate for the $B$ matrix than $A$; free convergence speed-up (ICML 2024).
    - [Kopiczko et al., *VeRA: Vector-based Random Matrix Adaptation* (2024)](https://arxiv.org/abs/2310.11454) — shares frozen random $A, B$ across all layers; only tiny per-layer scaling vectors are trained, cutting adapter size 10–100× vs. LoRA (ICLR 2024).
    - [Sheng et al., *S-LoRA: Serving Thousands of Concurrent LoRA Adapters* (2023)](https://arxiv.org/abs/2311.03285) — unified paging + segmented GEMM kernel to serve thousands of adapters on one GPU at near-base throughput; design now integrated in vLLM.

    **Open-source & tools**

    - [huggingface/peft](https://github.com/huggingface/peft) — the reference implementation of LoRA, QLoRA, DoRA, rsLoRA, VeRA, and a dozen other PEFT methods; integrates directly with Transformers and TRL.
    - [bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) — NF4/INT8 quantization primitives and 8-bit paged optimizers that power QLoRA in practice.

    **Go deeper**

    - [HuggingFace Blog: *Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA* (2023)](https://huggingface.co/blog/4bit-transformers-bitsandbytes) — practical walkthrough of the full QLoRA stack with runnable code and memory benchmarks.

## Further reading

- Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Chen — *LoRA: Low-Rank Adaptation of Large Language Models* (2021). The original method and intrinsic-rank argument.
- Aghajanyan, Zettlemoyer, Gupta — *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning* (2020). The empirical groundwork for "fine-tuning is low-dimensional."
- Dettmers, Pagnoni, Holtzman, Zettlemoyer — *QLoRA: Efficient Finetuning of Quantized LLMs* (2023). NF4, double quantization, paged optimizers.
- Liu, Wang, Yin, Molchanov, Wang, Cheng, Chen — *DoRA: Weight-Decomposed Low-Rank Adaptation* (2024).
- Kalajdzievski — *A Rank-Stabilized Scaling Factor for Low-Rank Adaptation* (rsLoRA, 2023).
- Hayou, Ghosh, Yu — *LoRA+: Efficient Low Rank Adaptation of Large Models* (2024).
- Kopiczko, Blankevoort, Asano — *VeRA: Vector-based Random Matrix Adaptation* (2024).
- Houlsby et al. — *Parameter-Efficient Transfer Learning for NLP* (2019). The original (serial) adapter modules LoRA improved upon.
- Sheng et al. — *S-LoRA: Serving Thousands of Concurrent LoRA Adapters* (2023); Chen et al. — *Punica: Multi-Tenant LoRA Serving* (2024). The multi-adapter serving systems.
- The HuggingFace **PEFT** library and **bitsandbytes** repository — the reference implementations of everything in this chapter.
