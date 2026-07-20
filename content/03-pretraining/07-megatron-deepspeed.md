# 3.7 Megatron-LM, DeepSpeed & Parallelism in Practice

Knowing the theory of tensor, pipeline, and data parallelism is one thing. Knowing how to wire them together on a 512-GPU cluster, pick the right degrees, and then confirm that your hardware is actually doing useful work is another. This chapter bridges that gap. We take the parallelism primitives introduced in [Distributed Training I: Data Parallelism, DDP, ZeRO & FSDP](../03-pretraining/05-distributed-data-parallel.html) and [Distributed Training II: Tensor, Pipeline, Sequence & Expert Parallelism](../03-pretraining/06-distributed-model-parallel.html), and show how Megatron-LM and DeepSpeed compose them into production training runs.

By the end of this chapter you will understand the Megatron-Core abstraction layer, the full ZeRO hierarchy and its offload variants, the 4-D (DP × TP × PP × EP) parallelism space, how to reason about Model FLOP Utilization (MFU) and Hardware FLOP Utilization (HFU), and exactly which configuration levers to pull for a 70B-parameter run.

## Megatron-LM: A Framework Built Around 3-D Parallelism

Megatron-LM, developed at NVIDIA, was the first framework to train models beyond 100B parameters in a systematic way. The 2021 paper by Narayanan et al. ("Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM") introduced the idea of combining tensor parallelism (TP), pipeline parallelism (PP), and data parallelism (DP) in a principled way. The current codebase ships as two packages:

- **Megatron-LM** — the outer loop: launch scripts, training harness, logging, checkpointing.
- **Megatron-Core (megatron.core)** — a library of parallelism-aware transformer building blocks that other frameworks (NVIDIA NeMo, Databricks MosaicML, Aleph Alpha) can import.

### The 3-D Parallelism Layout

Given a cluster with $N$ GPUs, Megatron partitions them into a 3-D grid:

$$
N = \text{DP} \times \text{TP} \times \text{PP}
$$

Every GPU belongs to exactly one data-parallel replica, one tensor-parallel group, and one pipeline stage. The layout is typically described as:

{{fig:megatron-3d-grid-layout}}

The rule of thumb that emerges from practice: **place TP within a node** (so the all-reduce inside a TP group can use NVLink rather than crossing InfiniBand) and **use PP to span nodes** (pipeline sends are point-to-point and latency-tolerant). DP can be at any granularity but is often the outermost dimension.

### Megatron-Core Parallel State

Megatron-Core tracks the parallelism topology in a global `parallel_state` module. Understanding this is essential for debugging.

```python
# megatron/core/parallel_state.py  (simplified)
import torch
import torch.distributed as dist

_TP_GROUP = None  # tensor-parallel process group
_PP_GROUP = None  # pipeline-parallel process group
_DP_GROUP = None  # data-parallel process group

def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
):
    """
    Build the 3-D process group topology.
    Assumes dist.init_process_group() has already been called.
    world_size = TP * PP * DP is enforced implicitly.
    """
    global _TP_GROUP, _PP_GROUP, _DP_GROUP

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    dp_size = world_size // (tensor_model_parallel_size * pipeline_model_parallel_size)

    # --- Tensor-parallel groups ---
    # Each group of TP consecutive ranks forms one TP group.
    for i in range(pipeline_model_parallel_size * dp_size):
        ranks = list(range(i * tensor_model_parallel_size,
                           (i + 1) * tensor_model_parallel_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            _TP_GROUP = group

    # --- Pipeline-parallel groups ---
    # Stride by TP; each group of PP elements (spaced TP apart) is a PP group.
    for i in range(tensor_model_parallel_size * dp_size):
        ranks = list(range(i, world_size, tensor_model_parallel_size))[:pipeline_model_parallel_size]
        group = dist.new_group(ranks)
        if rank in ranks:
            _PP_GROUP = group

    # --- Data-parallel groups ---
    for i in range(tensor_model_parallel_size * pipeline_model_parallel_size):
        ranks = list(range(i, world_size, tensor_model_parallel_size * pipeline_model_parallel_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            _DP_GROUP = group

def get_tensor_model_parallel_group(): return _TP_GROUP
def get_pipeline_model_parallel_group(): return _PP_GROUP
def get_data_parallel_group(): return _DP_GROUP
```

Every Megatron-Core layer that performs a collective operation (e.g., the column/row linear layers in the transformer) calls `get_tensor_model_parallel_group()` to target the right process group. This design keeps the distributed logic out of user code.

### Sequence Parallelism in Megatron

The original Megatron TP implementation broadcasts the input activations to all TP ranks before computing. This means the activations (LayerNorm inputs, dropout outputs) are replicated across TP ranks and waste memory proportional to `TP`.

Megatron-LM's *sequence parallelism* (introduced in "Reducing Activation Recomputation in Large Transformer Models," Korthikanti et al. 2022) avoids this. Outside the TP-sharded GEMM blocks, the sequence dimension is sharded across TP ranks. The transition in/out of sequence-parallel regions uses `all-gather` before the column parallel GEMM and `reduce-scatter` after the row parallel GEMM — replacing an `all-reduce` with two smaller collectives.

{{fig:megatron-seqparallel-dataflow}}

Net effect: activation memory at steady state is `H * B * S / TP` rather than `H * B * S`. For TP=8 this is an 8× activation reduction — significant at long sequences.

## DeepSpeed: ZeRO and Beyond

DeepSpeed (Microsoft) complements Megatron by providing the optimizer-side solution. The core abstraction is **ZeRO** (Zero Redundancy Optimizer), which eliminates the redundant copies of optimizer states, gradients, and parameters that vanilla DDP maintains.

### ZeRO Stages Recap

The three ZeRO stages correspond to increasingly aggressive sharding across DP ranks. We cover the theory in [Distributed Training I](../03-pretraining/05-distributed-data-parallel.html); here we focus on the implementation details that matter for a real run.

{{fig:zero-stages}}

| Stage | What is sharded | Peak memory saving (DP=64) | Communication overhead |
|-------|----------------|---------------------------|------------------------|
| ZeRO-1 | Optimizer states (momentum, variance) | ~4× for Adam | Minimal: reduce-scatter on gradients |
| ZeRO-2 | + Gradients | ~8× | Same as ZeRO-1 |
| ZeRO-3 | + Parameters | ~64× | All-gather on forward, reduce-scatter on backward |

For a model with $P$ parameters stored in fp16 (2 bytes) and Adam optimizer states in fp32:

$$
\text{Memory per GPU (ZeRO-3)} = \frac{2P + 2P + 12P}{\text{DP}} = \frac{16P}{\text{DP}}
$$

The first $2P$ is the fp16 parameters, the second $2P$ the fp16 gradients, and the $12P$ comes from the fp32 master weights (4), Adam first moment (4), and second moment (4) — the same 16 bytes per parameter tallied in Step 1 below. ZeRO-3 shards all 16 bytes; each DP rank holds $16P / \text{DP}$ bytes of "owned" state plus the activations for its pipeline stage.

### ZeRO-Offload and ZeRO-Infinity

For clusters with more CPU memory or NVMe than GPU memory, DeepSpeed provides offload variants:

- **ZeRO-Offload**: moves optimizer states (and optionally gradients) to CPU RAM. The optimizer step runs on CPU, which is fine because optimizer steps are memory-bandwidth-bound (read param + states, write updated param) and do not require GPU arithmetic throughput.
- **ZeRO-Infinity**: extends offload to NVMe storage using heterogeneous memory management. A bandwidth-aware scheduler overlaps NVMe reads with GPU compute.

```python
# DeepSpeed config JSON for ZeRO-3 with CPU offload
import json

zero3_config = {
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",       # optimizer states live on CPU RAM
            "pin_memory": True     # page-locked for fast DMA transfer
        },
        "offload_param": {
            "device": "cpu",       # fp16 params also offloaded
            "pin_memory": True
        },
        "overlap_comm": True,      # overlap reduce-scatter with backward pass
        "contiguous_gradients": True,
        "sub_group_size": 1e9,     # process params in 1B-element chunks
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 0,           # dynamic loss scaling
        "loss_scale_window": 1000
    },
    "gradient_clipping": 1.0,
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 8,
}

with open("ds_config_zero3.json", "w") as f:
    json.dump(zero3_config, f, indent=2)
```

### Initializing a DeepSpeed Engine

```python
import deepspeed
import torch
import torch.nn as nn

class TinyTransformerBlock(nn.Module):
    """A minimal transformer block for demonstration."""
    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
        )

    def forward(self, x):
        # Pre-norm residual style (GPT-style)
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x

model = nn.Sequential(*[TinyTransformerBlock(1024, 16) for _ in range(24)])

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

# deepspeed.initialize wraps the model, optimizer, and dataloader
# into a DeepSpeedEngine that handles ZeRO sharding transparently.
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config="ds_config_zero3.json",
)

# Training step — identical to vanilla PyTorch
for batch in dataloader:
    inputs, labels = batch
    loss = model_engine(inputs)
    model_engine.backward(loss)    # handles gradient sharding internally
    model_engine.step()            # triggers all-gather, optimizer step, re-shard
```

## The 4-D Parallelism Space: Adding Expert Parallelism

Modern MoE (Mixture-of-Experts) models add a fourth dimension. See [Mixture-of-Experts (MoE) Architectures](../02-transformer/09-mixture-of-experts.html) for the architecture. The full parallelism space becomes:

$$
N = \text{DP} \times \text{TP} \times \text{PP} \times \text{EP}
$$

**Expert parallelism (EP)** shards the experts across EP ranks. Within a single MoE layer, tokens are dispatched to experts on different GPUs via all-to-all collectives. EP communicates *token activations* rather than parameters, so the all-to-all volume is proportional to sequence length and hidden size, not parameter count.

Megatron-Core's `MoELayer` handles the EP dimension natively. The key constraint: each EP group must see enough tokens to keep all experts loaded. An expert that processes very few tokens is wasted capacity — the *load imbalance* problem that auxiliary loss terms (introduced by Switch Transformer, Fedus et al.) address.

{{fig:megatron-4d-moe-collectives}}

## Choosing Parallelism Degrees: A Systematic Approach

The parallelism config is the single highest-leverage decision when launching a large training run. Getting it wrong can cost 30-50% of hardware efficiency. Here is a principled workflow.

### Step 1 — Fit the Model on One Node

Start with the model's parameter count $P$. For mixed-precision training with bf16 parameters and fp32 optimizer states:

$$
\text{Bytes per parameter} = 2 \text{ (bf16 param)} + 2 \text{ (bf16 grad)} + 4 \text{ (fp32 master)} + 8 \text{ (Adam)} = 16
$$

For a 70B model: $70 \times 10^9 \times 16 = 1{,}120 \text{ GB}$ of pure model state, before activations or intermediate buffers.

With 8× H100 80 GB per node (640 GB HBM), you need at least $\lceil 1120 / 80 \rceil = 14$ GPUs just for model state. In practice, activation memory can double this, so **TP=8 × PP=4 = 32 GPUs minimum** before any DP replication.

### Step 2 — Pick TP

TP is constrained by intra-node bandwidth (NVLink). The all-reduce inside a TP column-parallel GEMM must finish before the next GEMM begins; it sits on the critical path.

- TP=1: no communication, maximum arithmetic intensity.
- TP=2: doubles memory for attention and FFN weight distribution; all-reduce is 2× 25 GB/s NVLink streams.
- TP=4 or TP=8: recommended for nodes with 4 or 8 GPUs respectively and NVLink.
- TP > 8: crosses PCIe/InfiniBand; avoid unless forced.

Rule: **TP = number of GPUs per node** (or a divisor of it) and never span node boundaries.

### Step 3 — Pick PP

Pipeline parallelism introduces a bubble overhead. The bubble fraction for the 1F1B (one-forward-one-backward) schedule is approximately:

$$
\text{bubble fraction} \approx \frac{PP - 1}{PP - 1 + m}
$$

where $m$ is the number of micro-batches in flight. To keep bubble under 5%:

$$
m \geq 19(PP - 1)
$$

For PP=4 this means $m \geq 57$ micro-batches, which is achievable with large global batch sizes.

Interleaved pipeline schedules (Megatron's virtual pipeline parallelism) split each stage into $v$ chunks, reducing the bubble to:

$$
\text{bubble fraction (interleaved)} \approx \frac{1}{v} \cdot \frac{PP - 1}{PP - 1 + m} \approx \frac{PP - 1}{v \cdot m}
$$

at the cost of additional pipeline communication per micro-batch.

### Step 4 — Set DP from what remains

$$
\text{DP} = \frac{N_{\text{total GPUs}}}{\text{TP} \times \text{PP}}
$$

DP is "free" communication if you use ZeRO-1 or ZeRO-2 (the reduce-scatter / all-gather can be overlapped with the backward pass). ZeRO-3 adds synchronous all-gather per forward pass but eliminates parameter redundancy.

### Step 5 — Tune global batch size and gradient accumulation

Global batch size (GBS) drives convergence. There is a *critical batch size* $B_{\text{crit}}$ — well approximated by the *gradient noise scale* of McCandlish et al. ([*An Empirical Model of Large-Batch Training*, 2018](https://arxiv.org/abs/1812.06162)) — below which increasing the batch size buys a near-linear reduction in the number of optimizer steps, and above which the returns diminish sharply. $B_{\text{crit}}$ is *not* a function of vocabulary size and has no simple closed form; it grows over the course of training as the loss falls. Rather than a formula, practitioners target empirical global batch sizes on the order of 1M–4M tokens per step for runs above 10B parameters, then tune the learning rate against batch size as described in [Learning Rate Schedules, Warmup, Batch Size & Hyperparameters](../03-pretraining/10-lr-schedules-hparams.html).

Given a fixed GBS and a per-GPU micro-batch size (MBS), the number of gradient accumulation steps (GAS) is:

$$
\text{GAS} = \frac{\text{GBS}}{\text{MBS} \times \text{DP} \times S}
$$

## MFU and HFU: Measuring Real Hardware Utilization

You have launched the run. Now you want to know if you are getting good value from your hardware. Two metrics matter.

### Model FLOP Utilization (MFU)

MFU measures what fraction of the GPU's peak throughput is being used for *forward + backward* arithmetic of the model itself.

The number of FLOPs per token for a standard decoder-only Transformer is approximately:

$$
\text{FLOPs/token} \approx 6P + 12 \cdot n_\text{layers} \cdot d_\text{model} \cdot S
$$

where the $6P$ term comes from $\sim 2P$ for the forward pass (each parameter participates in roughly 2 multiply-adds) times 3 for the full backward pass, and the second term is the attention quadratic cost (often secondary for moderate sequence lengths).

For practical purposes, practitioners often use the simplified rule:

$$
\text{FLOPs/token} \approx 6P
$$

MFU is then:

$$
\text{MFU} = \frac{\text{FLOPs/token} \times \text{tokens/second}}{\text{peak FLOP/s of cluster}}
$$

**Target MFU for dense models on A100/H100 clusters is 35–55%.** Values below 25% suggest a misconfiguration (bubble too high, TP spans nodes, small micro-batch sizes hitting latency bottlenecks, checkpointing overhead).

### Hardware FLOP Utilization (HFU)

HFU counts *all* FLOPs actually issued to the GPU, including those in recomputed activations (gradient checkpointing). If you recompute one third of layers:

$$
\text{HFU} = \text{MFU} \times \frac{\text{total FLOPs issued}}{\text{model FLOPs}}
$$

With full activation recomputation, you pay for an extra forward pass just before each backward, giving forward $2P$ + recomputed forward $2P$ + backward $4P$ = $8P$ per token, so:

$$
\text{FLOPs/token (with full recompute)} \approx \frac{4}{3} \times 6P = 8P
$$

HFU ≥ MFU. A good cluster should see HFU of 55–70% on H100s with modern frameworks.

```python
def compute_mfu(
    model_params: int,        # number of parameters
    tokens_per_second: float, # observed training throughput
    peak_flops_per_sec: float, # e.g., 5.06e17 for dense bf16 on 512 H100 SXM5 GPUs
    n_layers: int = None,
    d_model: int = None,
    seq_len: int = None,
) -> float:
    """
    Compute Model FLOP Utilization.

    Uses the simplified 6P rule for forward+backward FLOPs per token.
    If n_layers, d_model, seq_len are provided, also adds attention cost.
    """
    flops_per_token = 6 * model_params

    # Attention cost (forward + backward): 12 * n_layers * d_model * seq_len per token.
    # Forward is 4 * d * S per layer (QK^T and AV, each 2 * d * S); x3 for the
    # backward pass, matching the fwd+bwd convention of the 6P base term.
    if all(v is not None for v in [n_layers, d_model, seq_len]):
        attn_flops = 12 * n_layers * d_model * seq_len
        flops_per_token += attn_flops

    achieved_flops = flops_per_token * tokens_per_second
    mfu = achieved_flops / peak_flops_per_sec
    return mfu


# Example: 70B model, 512 H100 SXM5 GPUs
# H100 SXM5 dense bf16 peak: ~989 TFLOP/s = 9.89e14 FLOP/s per GPU.
# (The 1979 TFLOP/s figure NVIDIA quotes is the 2:4-sparse rate; dense
#  training runs against half of it.)
H100_BF16_TFLOPS = 9.89e14
n_gpus = 512
peak_cluster_flops = H100_BF16_TFLOPS * n_gpus  # ~5.06e17 FLOP/s

# Observed: 1200 tokens/second per GPU = 614.4K tokens/second total
tokens_per_sec = 1200 * n_gpus

mfu = compute_mfu(
    model_params=70e9,
    tokens_per_second=tokens_per_sec,
    peak_flops_per_sec=peak_cluster_flops,
    n_layers=80,
    d_model=8192,
    seq_len=4096,
)
print(f"MFU: {mfu:.2%}")  # prints ~54.9% (~55%) for a well-configured run, incl. attention term
```

!!! example "Worked Example: Memory Budget for a 70B Run"

    **Setup**: 70B parameter model, bf16, 512 H100-80GB GPUs, TP=8, PP=8, DP=8, GBS=4M tokens, seq_len=4096.

    **Model state (per DP rank, ZeRO-1)**:
    - bf16 parameters: $70 \times 10^9 \times 2 = 140$ GB total, 140/1 per DP rank (ZeRO-1 does not shard params)
    - With TP=8, PP=8: each GPU holds $\frac{1}{8 \times 8} = \frac{1}{64}$ of the model = $140/64 \approx 2.2$ GB bf16 params
    - fp32 master weight copy: $140 \times 2 = 280$ GB total / 64 = 4.4 GB per GPU
    - Adam states: same as master copy = 4.4 GB per GPU
    - **Subtotal model state per GPU**: $2.2 + 4.4 + 4.4 = 11$ GB

    **Activation memory** (one pipeline stage, without recomputation):
    - Layers per pipeline stage: $80 / 8 = 10$ layers
    - Activations per layer ≈ $2 \times B \times S \times H$ bytes (input and output of attention block)
    - With MBS=2, $S=4096$, $H=8192$: $2 \times 2 \times 4096 \times 8192 \times 2 \approx 537$ MB per layer
    - 10 layers: $\approx 5.4$ GB activations per stage (before recompute)
    - With selective recompute (e.g., recompute attention blocks only): reduce by $\sim$40% → 3.2 GB

    **Total per GPU (approximate)**: $11 + 3.2 + 2$ (buffers/gradients) $= 16.2$ GB — well within 80 GB.

    **MFU check**:
    - FLOPs per token: $6 \times 70 \times 10^9 = 4.2 \times 10^{11}$
    - Peak cluster (dense bf16): $9.89 \times 10^{14} \times 512 \approx 5.06 \times 10^{17}$ FLOP/s
    - Need $\geq 6.0 \times 10^5$ tokens/s cluster-wide for 50% MFU: $6.0 \times 10^5 / 512 \approx 1{,}180$ tokens/s per GPU
    - A well-tuned 70B run on H100s achieves roughly 850–1,300 tokens/s per GPU, corresponding to MFU of 36–55% against the dense bf16 peak.

## A Complete Worked Configuration: 70B Pretraining Run

This section presents a concrete, production-style launch for a 70B dense model using Megatron-LM + DeepSpeed ZeRO-1.

### Cluster topology

```text
Cluster: 64 nodes × 8 H100-SXM5-80GB = 512 GPUs
Network: 400 Gb/s InfiniBand NDR (inter-node), 900 GB/s NVLink (intra-node)
Parallelism: TP=8, PP=8, DP=8  →  512 = 8 × 8 × 8
```

### Model config

```python
# model_config.py — Llama-style 70B architecture
MODEL_CONFIG = {
    "num_layers": 80,
    "hidden_size": 8192,
    "ffn_hidden_size": 28672,   # ~3.5x hidden_size for SwiGLU
    "num_attention_heads": 64,
    "num_key_value_heads": 8,   # GQA with 8 KV heads
    "max_position_embeddings": 8192,
    "vocab_size": 128256,       # Llama-3 vocabulary
    "activation_function": "swiglu",
    "normalization": "rmsnorm",
    "tie_embeddings": False,
}
```

### Megatron launch script

```bash
#!/bin/bash
# launch_70b.sh — SLURM-based Megatron-LM 70B launch

#SBATCH --nodes=64
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --mem=960G           # enough for ZeRO-Offload CPU tensors if needed

# ---- Parallelism degrees ----
TP=8
PP=8
DP=8  # implicit: 512 / (8*8)

# ---- Batch configuration ----
# Global batch size: ~4M tokens per step
# Seq len 4096, MBS=1 per GPU, GAS=128 → GBS = 1 * 512 * 128 * 4096 = 268M tokens... too large
# More typically: MBS=2, GAS=32 → GBS = 2 * 512 * 32 * 4096 = ~134M tokens/step — still large
# In practice: GBS set to 1M tokens = 244 sequences of 4096 tokens
# With DP=8, MBS=1, GAS=ceil(244/8/GAS_factor): tune per run
SEQ_LEN=4096
GLOBAL_BATCH_SIZE=2048      # sequences per step = 2048 × 4096 = 8.4M tokens
MICRO_BATCH_SIZE=2
GAS=$((GLOBAL_BATCH_SIZE / (DP * MICRO_BATCH_SIZE)))  # = 128

# ---- Training config ----
TRAIN_ITERS=500000
LR=3e-4
MIN_LR=3e-5
LR_WARMUP_ITERS=2000
LR_DECAY_STYLE=cosine
CLIP_GRAD=1.0
WEIGHT_DECAY=0.1

# ---- Paths ----
DATA_PATH=/mnt/storage/tokenized/llama3_merged
CHECKPOINT_PATH=/mnt/checkpoints/70b-run

torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  pretrain_gpt.py \
  --tensor-model-parallel-size $TP \
  --pipeline-model-parallel-size $PP \
  --num-layers 80 \
  --hidden-size 8192 \
  --ffn-hidden-size 28672 \
  --num-attention-heads 64 \
  --group-query-attention \
  --num-query-groups 8 \
  --seq-length $SEQ_LEN \
  --max-position-embeddings $SEQ_LEN \
  --micro-batch-size $MICRO_BATCH_SIZE \
  --global-batch-size $GLOBAL_BATCH_SIZE \
  --train-iters $TRAIN_ITERS \
  --lr $LR \
  --min-lr $MIN_LR \
  --lr-warmup-iters $LR_WARMUP_ITERS \
  --lr-decay-style $LR_DECAY_STYLE \
  --weight-decay $WEIGHT_DECAY \
  --clip-grad $CLIP_GRAD \
  --bf16 \
  --use-flash-attn \
  --recompute-activations \        # selective recompute: attention only
  --recompute-granularity selective \
  --use-distributed-optimizer \    # ZeRO-1 style optimizer sharding
  --overlap-grad-reduce \          # overlap DP gradient reduce with backward
  --overlap-param-gather \         # overlap ZeRO-3 param gather with forward
  --use-rope-scaling \
  --normalization RMSNorm \
  --swiglu \
  --tokenizer-type TikTokenizer \
  --data-path $DATA_PATH \
  --save $CHECKPOINT_PATH \
  --load $CHECKPOINT_PATH \
  --save-interval 1000 \
  --eval-interval 500 \
  --log-interval 10 \
  --tensorboard-dir $CHECKPOINT_PATH/tb \
  --wandb-project llm-stack-70b
```

### DeepSpeed config for this run

```json
{
  "zero_optimization": {
    "stage": 1,
    "overlap_comm": true,
    "allgather_partitions": true,
    "reduce_scatter": true,
    "allgather_bucket_size": 500000000,
    "reduce_bucket_size": 500000000
  },
  "bf16": {
    "enabled": true
  },
  "gradient_clipping": 1.0,
  "train_micro_batch_size_per_gpu": 2,
  "gradient_accumulation_steps": 128,
  "steps_per_print": 10,
  "wall_clock_breakdown": false
}
```

### Monitoring the run

```python
# parse_megatron_logs.py — extract MFU from Megatron-LM stdout
import re
import sys

LOG_LINE_RE = re.compile(
    r"iteration\s+(\d+)/\s*\d+.*?elapsed time per iteration \(ms\): ([\d.]+).*?"
    r"tokens-per-second-per-gpu: ([\d.]+)",
    re.DOTALL,
)

H100_BF16_PEAK_TFLOPS = 989.0  # dense bf16 Tensor Core, per GPU (1979 is the 2:4-sparse rate)
MODEL_PARAMS = 70e9

def toks_per_sec_to_mfu(tps_per_gpu: float) -> float:
    flops_per_tok = 6 * MODEL_PARAMS
    achieved_tflops = flops_per_tok * tps_per_gpu / 1e12
    return achieved_tflops / H100_BF16_PEAK_TFLOPS

for line in sys.stdin:
    m = LOG_LINE_RE.search(line)
    if m:
        iteration = int(m.group(1))
        ms_per_iter = float(m.group(2))
        tps = float(m.group(3))
        mfu = toks_per_sec_to_mfu(tps)
        print(f"iter {iteration:6d} | {ms_per_iter:6.0f} ms/it | {tps:5.0f} tok/s/gpu | MFU {mfu:.1%}")
```

## Practical Pitfalls and Tuning Knobs

### The TP Communication Bottleneck

Tensor parallelism sits on the critical path of the forward pass. If TP all-reduces are slow (e.g., because TP spans InfiniBand instead of NVLink), you can lose 20-40% of throughput. Always profile with `nsys profile` and check that `ncclAllReduce` calls within a TP group run at NVLink speed (≈ 600 GB/s aggregate bidirectional).

### Pipeline Bubble vs. Memory Tradeoff

Increasing PP reduces per-GPU memory but increases the bubble fraction. For PP=8 and $m=32$ micro-batches, the bubble is $(8-1)/(8-1+32) \approx 18\%$. Doubling micro-batches (increasing global batch or reducing MBS) drops this to 9%. Interleaved schedules (virtual PP) halve it again but increase inter-stage communication by a factor of $v$.

### Activation Recomputation Granularity

Megatron-LM offers three granularity levels:

| Mode | What is recomputed | Memory | Extra FLOPs |
|------|-------------------|--------|-------------|
| `full` | Entire layer | Minimum | +33% |
| `selective` | Attention softmax + dropout | Medium | +5-15% |
| `none` | Nothing | Maximum | +0% |

For large models, `selective` is the sweet spot — it eliminates the expensive-to-store softmax activations (which grow as $O(S^2)$ in sequence length) while retaining the cheaper MLP activations.

!!! warning "Gradient accumulation and ZeRO-3 interaction"

    When combining ZeRO-3 with gradient accumulation, each micro-batch forward pass triggers a parameter all-gather. With GAS=128, you do 128 all-gathers per optimizer step. Use `--overlap-param-gather` to pipeline these with compute, and set `stage3_max_live_parameters` large enough to buffer at least one full transformer block's parameters, otherwise you stall.

### Choosing Between Megatron and FSDP

PyTorch's Fully Sharded Data Parallel ([Distributed Training I](../03-pretraining/05-distributed-data-parallel.html)) covers a similar use case to ZeRO-3. Rule of thumb:

- **Megatron-Core + ZeRO-1** for runs with dedicated clusters and TP/PP requirements (>30B parameters).
- **FSDP2** for runs up to ~30B parameters where single-framework PyTorch is preferred and TP is unnecessary.
- **FSDP + TP (Tensor Parallelism via DTensor)** is the PyTorch-native path for very large models without Megatron dependency.

For mixed-precision choices and the role of bf16 vs fp8 in these runs, see [Mixed Precision, bf16 & FP8 Training](../03-pretraining/08-mixed-precision-fp8.html).

!!! tip "Profiling before tuning"

    Before changing any parallelism config, run NVIDIA Nsight Systems for one iteration:

    ```bash
    nsys profile \
        --trace cuda,nvtx \
        --output profile_iter \
        --capture-range cudaProfilerApi \
        python pretrain_gpt.py --profile-step 5 ...
    ```

    Look for: (1) long NCCL gaps (communication bottleneck), (2) back-to-back small kernels (micro-batch too small, memory-bound), (3) idle GPU time between pipeline stages (bubble). Each symptom has a distinct fix.

## Hyperparameter Sensitivity and Scaling the Config

Not all hyperparameters are scale-invariant. When you double the cluster and increase GBS, you typically need to:

1. **Scale the learning rate** with the square root of GBS (linear scaling works empirically up to a point, but for very large batches $\sqrt{\text{GBS}}$ scaling is safer). See [Learning Rate Schedules, Warmup, Batch Size & Hyperparameters](../03-pretraining/10-lr-schedules-hparams.html).
2. **Extend warmup** proportionally to GBS — a common heuristic is to warm up over 1–2B tokens regardless of batch size.
3. **Reduce gradient clipping threshold** as model depth grows to avoid spurious gradient explosions (see [Training Stability, Loss Spikes & Debugging Large Runs](../03-pretraining/11-training-stability.html)).

### Virtual Pipeline Parallelism

Megatron's interleaved pipeline (enabled with `--num-layers-per-virtual-pipeline-stage`) assigns each GPU multiple non-contiguous chunks of layers:

{{fig:megatron-virtual-pipeline-interleave}}

This halves the bubble at the cost of sending twice as many pipeline messages per micro-batch. In practice, for TP=8 where the pipeline messages are relatively small (one micro-batch of activations), interleaving is almost always worth it above PP=4.

!!! interview "Interview Corner"

    **Q:** You are given a 256-GPU cluster (8 GPUs/node, NVLink intra-node, InfiniBand inter-node) and asked to train a 70B dense model. Walk through how you would choose TP, PP, and DP, and justify each choice.

    **A:** Start with TP=8 — one full node — because all tensor-parallel all-reduces then stay on NVLink (fast) and never touch InfiniBand (slow). With TP=8 and 8 nodes remaining in the config, we choose PP=4 which gives 8/8=1 set of 4-stage pipelines per TP group and leaves DP=256/(8×4)=8. Verify memory: 70B params × 16 bytes / (8×4 TP×PP sharding) ≈ 35 GB model state per GPU, plus ~5-10 GB activations with selective recompute → comfortably fits 80 GB. For MFU, PP=4 with interleaved schedule (v=2) and 32+ micro-batches gives a bubble below 10%. If we needed more DP, we would scale the cluster rather than reducing TP/PP. If DP gradient communication becomes the bottleneck, we switch from ZeRO-1 to ZeRO-2 to reduce that traffic.

## Combining Megatron-Core with External Libraries

Megatron-Core is designed to be embedded. The typical NeMo or Databricks Mosaic setup looks like:

```python
# Pattern: Megatron-Core layer inside a custom training loop
from megatron.core import parallel_state
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel

# 1. Initialize distributed environment
import torch.distributed as dist
dist.init_process_group(backend="nccl")

# 2. Configure the model parallel topology
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=4,
)

# 3. Build the model using TransformerConfig
config = TransformerConfig(
    num_layers=80,
    hidden_size=8192,
    num_attention_heads=64,
    num_query_groups=8,         # GQA
    ffn_hidden_size=28672,
    use_cpu_initialization=False,
    bf16=True,
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=4,
)

model = GPTModel(
    config=config,
    transformer_layer_spec=get_gpt_layer_spec(),  # from megatron.core.models.gpt
    vocab_size=128256,
    max_sequence_length=8192,
)

# 4. Wrap with DeepSpeed for ZeRO-1
import deepspeed
model_engine, _, _, _ = deepspeed.initialize(model=model, config="ds_config_zero1.json")
```

This pattern — Megatron-Core for the TP/PP topology, DeepSpeed for the optimizer-side ZeRO sharding — is sometimes called **3D + ZeRO** and is the dominant approach for frontier model training runs as of 2025.

For inference serving after training, the parallelism story shifts toward pure TP (no PP, since autoregressive decode cannot pipeline) and often requires weight resharding from the training checkpoint format. See [Multi-GPU & Multi-Node Inference](../07-inference-serving/11-multi-gpu-inference.html) for the inference-side parallelism story.

!!! sota "State of the Art & Resources (2026)"
    As of 2026, the 3D + ZeRO pattern (Megatron-Core for tensor/pipeline parallelism, DeepSpeed for optimizer sharding) remains the dominant approach for frontier pretraining runs, with recent additions of context parallelism (CP) expanding the space to 4D or even 5D parallelism for long-sequence models. Production clusters regularly achieve 40–55% MFU on H100/H200 hardware using these frameworks.

    **Foundational work**

    - [Shoeybi et al., *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism* (2019)](https://arxiv.org/abs/1909.08053) — introduced tensor parallelism for transformers; the column/row parallel GEMM design still used today.
    - [Narayanan et al., *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM* (2021)](https://arxiv.org/abs/2104.04473) — established the 3D (TP × PP × DP) parallelism framework and the 1F1B pipeline schedule.
    - [Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* (2020)](https://arxiv.org/abs/1910.02054) — the three-stage ZeRO optimizer sharding scheme; foundational for all large-scale runs.
    - [Chowdhery et al., *PaLM: Scaling Language Modeling with Pathways* (2022)](https://arxiv.org/abs/2204.02311) — introduced the MFU metric and reported 57.8% MFU on 6144 TPUs; the benchmark for hardware efficiency analysis.

    **Recent advances (2023–2026)**

    - [Korthikanti et al., *Reducing Activation Recomputation in Large Transformer Models* (2023)](https://arxiv.org/abs/2205.05198) — sequence parallelism + selective activation recomputation; cuts activation memory by TP× with minimal overhead.
    - [Wang et al., *ZeRO++: Extremely Efficient Collective Communication for Giant Model Training* (2023)](https://arxiv.org/abs/2306.10209) — quantized weights and hierarchical partitioning cut ZeRO communication volume by 4×, up to 2.16× throughput gains.
    - [Rajbhandari et al., *ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning* (2021)](https://arxiv.org/abs/2104.07857) — extends ZeRO sharding to CPU and NVMe; enables trillion-parameter training on modest GPU counts.

    **Open-source & tools**

    - [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) — the reference implementation of Megatron-Core; includes TP, PP, CP, EP, and the distributed optimizer; actively maintained with H100/Blackwell support.
    - [deepspeedai/DeepSpeed](https://github.com/deepspeedai/DeepSpeed) — ZeRO stages 1–3, ZeRO-Infinity, ZeRO++, and DeepSpeed-MoE; integrates directly with Megatron-Core or HuggingFace Trainer.

    **Go deeper**

    - [Megatron-Core Parallelism Strategies Guide (NVIDIA docs)](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html) — official reference covering TP, PP, DP, CP, and EP configurations with recommended settings for common model scales.
    - [Zero Redundancy Optimizer Tutorial (DeepSpeed)](https://www.deepspeed.ai/tutorials/zero/) — hands-on walkthrough of ZeRO stages and configuration options, including offload variants.

## Further Reading

- Shoeybi, Patwary, et al. **"Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"** (2019). The original Megatron paper introducing TP for transformers.
- Narayanan, Shoeybi, et al. **"Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM"** (NeurIPS 2021). Introduces the 3-D parallelism framework and the 1F1B pipeline schedule.
- Korthikanti, Casper, et al. **"Reducing Activation Recomputation in Large Transformer Models"** (MLSys 2023). Introduces sequence parallelism and selective activation recomputation.
- Rajbhandari, Rasley, Ruwase, He. **"ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"** (SC 2020). The ZeRO paper.
- Rajbhandari, et al. **"ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning"** (SC 2021). ZeRO-Offload and NVMe offloading.
- Ren, et al. **"ZeRO-Offload: Democratizing Billion-Scale Model Training"** (USENIX ATC 2021).
- NVIDIA Megatron-Core GitHub: `NVIDIA/Megatron-LM` — the canonical reference implementation.
- Microsoft DeepSpeed GitHub: `microsoft/DeepSpeed` — ZeRO implementation and tutorials.
- Chowdhery, et al. **"PaLM: Scaling Language Modeling with Pathways"** (2022). Describes the 4D parallelism and MFU analysis methodology used at Google.

!!! key "Key Takeaways"

    - Megatron-LM organizes GPUs into a 3-D grid: TP (intra-node, NVLink), PP (inter-node, point-to-point), DP (outermost). The formula is $N = \text{DP} \times \text{TP} \times \text{PP}$.
    - Sequence parallelism in Megatron-Core shards activations along the sequence dimension outside TP-parallel regions, replacing one all-reduce with all-gather + reduce-scatter and cutting activation memory by a factor of TP.
    - DeepSpeed ZeRO has three stages: ZeRO-1 shards optimizer states, ZeRO-2 adds gradients, ZeRO-3 adds parameters. Combined memory reduction with DP=64 and ZeRO-3 can exceed 60×. ZeRO-Offload and ZeRO-Infinity extend sharding to CPU/NVMe.
    - For a 70B run, a representative production config is TP=8, PP=8, DP=8 on 512 H100-80GB GPUs with ZeRO-1 optimizer sharding and selective activation recomputation.
    - MFU measures what fraction of peak cluster FLOP/s is consumed by model arithmetic. Use $\text{FLOPs/token} \approx 6P$ for a quick estimate. Target 40–55% MFU for dense models; values below 25% indicate a misconfiguration.
    - Pipeline bubble fraction $\approx (PP-1)/(PP-1+m)$. Keep it below 5–10% by increasing micro-batch count $m$ or using interleaved (virtual PP) schedules.
    - Always profile before tuning. Nsight Systems traces quickly reveal whether the bottleneck is NCCL communication, pipeline bubbles, or kernel-launch overhead.
    - The 3D + ZeRO pattern (Megatron-Core for TP/PP + DeepSpeed for ZeRO optimizer) is the dominant approach for frontier pretraining runs; pure FSDP2 is the PyTorch-native alternative for runs that do not require TP.
    - Expert parallelism (EP) adds a fourth dimension for MoE models, communicating token activations (not parameters) via all-to-all across EP ranks.
