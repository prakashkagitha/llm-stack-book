#  Tooling & Environment Setup Cheatsheet

The practitioner's most expensive mistakes rarely happen in the model architecture. They happen in the environment: a CUDA version mismatch that silently falls back to CPU, a tokenizer pad-token bug that corrupts gradients for days, or a logging setup that only captures half the metrics before a run crashes. This appendix is a dense, opinionated reference — the "keep open in a tab" companion to the rest of the book. We cover every major library in the modern LLM stack, the gotchas that cost real engineers real time, and the one-liners you will reach for constantly.

The chapters on [Automatic Differentiation & PyTorch Internals](../01-foundations/07-autodiff-pytorch.html), [GPU Architecture & The Memory Hierarchy](../01-foundations/08-gpu-architecture.html), and [Distributed Training I: Data Parallelism, DDP, ZeRO & FSDP](../03-pretraining/05-distributed-data-parallel.html) provide the conceptual substrate. This appendix is the *executable* layer on top.

---

## The Library Landscape at a Glance

{{fig:tooling-library-stack}}

Understanding *which layer* owns a failure is half the debugging battle. A `CUDA error: device-side assert triggered` is almost always a PyTorch layer issue (wrong device, wrong dtype, invalid index); an `NCCL timeout` is a networking/driver issue; an `OutOfMemoryError` might be in the model layer, the optimizer state, or the KV cache.

---

## CUDA, Driver & Python Environment Setup

### Choosing a CUDA version

CUDA backward-compatibility is one-directional: a driver supports CUDA runtimes *up to* a certain version. Check compatibility with:

```bash
# Check installed driver and maximum CUDA version it supports
nvidia-smi

# Check the CUDA runtime version (must be <= driver's max)
nvcc --version

# Check that PyTorch sees the right CUDA
python -c "import torch; print(torch.version.cuda, torch.cuda.get_device_name(0))"
```

A common trap: your system `nvcc` points to CUDA 11.x, but you installed `torch` built against CUDA 12.x. PyTorch ships its own CUDA runtime and does *not* use the system one, but `flash-attn` and `bitsandbytes` compile against the system CUDA. The solution is to pin everything:

```bash
# Install the exact CUDA toolkit version your libraries need
# (example: CUDA 12.1 on Ubuntu 22.04)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-1
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
```

### Conda environment recipe

```bash
# Create a pinned environment — name encodes key versions
conda create -n llm-py311-cu121 python=3.11 -y
conda activate llm-py311-cu121

# Install PyTorch with the right CUDA build
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Verify immediately — never skip this
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available!"
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
print(f"Device: {torch.cuda.get_device_name(0)}")
t = torch.randn(4, 4, device='cuda')
print(f"Tensor device: {t.device}, dtype: {t.dtype}")
EOF
```

### Environment variables every LLM engineer should know

```bash
# Distributed training: tell each process which GPU to use
export CUDA_VISIBLE_DEVICES=0,1,2,3  # limit visibility to 4 GPUs

# NCCL tuning for multi-node runs
export NCCL_IB_DISABLE=0             # enable InfiniBand
export NCCL_DEBUG=INFO               # verbose NCCL logs for debugging
export NCCL_SOCKET_IFNAME=eth0       # bind to a specific NIC

# DeepSpeed / accelerate
export DS_BUILD_OPS=1                # pre-build DeepSpeed CUDA extensions
export TOKENIZERS_PARALLELISM=false  # avoid HuggingFace tokenizer fork warnings

# torch.compile / Triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache  # persist compiled kernels
export TRITON_CACHE_DIR=/tmp/triton_cache

# Hugging Face cache — avoid re-downloading 70 GB models every run
export HF_HOME=/data/hf_cache
export TRANSFORMERS_CACHE=/data/hf_cache/transformers
```

!!! warning "Common pitfall: CUDA_VISIBLE_DEVICES vs local_rank"
    When launching with `torchrun`, the framework sets `LOCAL_RANK` automatically. If you *also* set `CUDA_VISIBLE_DEVICES=0,1`, then local rank 0 maps to physical GPU 0 and local rank 1 maps to physical GPU 1. If you set `CUDA_VISIBLE_DEVICES=2,3`, then local rank 0 maps to physical GPU 2. Never set `CUDA_VISIBLE_DEVICES` inside `torchrun` scripts — let the launcher control it.

---

## Core Libraries: Installation & Version Pinning

### The dependency matrix

Certain libraries must be built against compatible versions of each other. This table reflects the state as of mid-2025:

| Component | Recommended version | Key constraint |
|---|---|---|
| Python | 3.10 or 3.11 | 3.12 has packaging incompatibilities |
| PyTorch | 2.3.x or 2.4.x | Pin CUDA build suffix |
| CUDA toolkit | 12.1 or 12.4 | Must match `flash-attn` build |
| `transformers` | ≥ 4.40 | Required for Llama-3, Gemma-2 |
| `datasets` | ≥ 2.18 | Arrow 15+ for large-scale streaming |
| `accelerate` | ≥ 0.30 | FSDP v2 support |
| `flash-attn` | ≥ 2.5 | FA3 kernels need Hopper GPU |
| `bitsandbytes` | ≥ 0.43 | Multi-backend (CUDA + ROCm) |
| `deepspeed` | ≥ 0.14 | ZeRO-3 + CPU offload fixes |
| `vllm` | ≥ 0.4 | PagedAttention v2 |
| `sglang` | ≥ 0.2 | RadixAttention, constraint generation |

```bash
# Install everything in one shot (adjust CUDA suffix as needed)
pip install \
    transformers>=4.40 \
    datasets>=2.18 \
    accelerate>=0.30 \
    trl>=0.8 \
    deepspeed>=0.14 \
    wandb \
    einops \
    sentencepiece \
    protobuf

# flash-attn must be installed AFTER torch — it compiles against installed torch
pip install flash-attn --no-build-isolation

# bitsandbytes: use the pre-built wheel for your CUDA version
pip install bitsandbytes>=0.43

# vLLM and SGLang are large and pin their own torch — install in separate envs
pip install vllm>=0.4
# OR
pip install sglang[all]
```

!!! warning "Common pitfall: flash-attn and --no-build-isolation"
    `flash-attn` needs to find your installed `torch` at compile time. Without `--no-build-isolation`, pip builds in an isolated environment where `torch` is not present and the build fails or produces a broken wheel. Always pass `--no-build-isolation` when installing `flash-attn`.

---

## PyTorch: Patterns & Power Features

### dtype cheatsheet for LLM work

```python
import torch

# Memory cost per parameter for common dtypes
dtype_bytes = {
    torch.float32: 4,   # full precision (baseline)
    torch.bfloat16: 2,  # preferred for LLM training on Ampere+
    torch.float16: 2,   # legacy; careful with gradient underflow
    torch.int8: 1,      # post-training quantization
    torch.float8_e4m3fn: 1,  # FP8, Hopper/Ada only (PyTorch >= 2.1)
}

# 7B model parameter memory at different dtypes
n_params = 7_000_000_000
for dtype, bytes_per_param in dtype_bytes.items():
    gb = n_params * bytes_per_param / 1e9
    print(f"{dtype}: {gb:.1f} GB")

# Output:
# torch.float32:   28.0 GB
# torch.bfloat16:  14.0 GB
# torch.float16:   14.0 GB
# torch.int8:       7.0 GB
# torch.float8_e4m3fn: 7.0 GB
```

See [Numerical Computing, Floating Point & Precision](../01-foundations/04-numerics-precision.html) for a detailed treatment of dtype trade-offs, and [Mixed Precision, bf16 & FP8 Training](../03-pretraining/08-mixed-precision-fp8.html) for training recipes.

### torch.compile: the one-liner speedup

```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B",
                                              torch_dtype=torch.bfloat16,
                                              device_map="cuda")

# Compile the forward pass — typically 20-40% speedup on Ampere+ for decode
# mode="reduce-overhead" is best for inference (fixed shapes)
# mode="max-autotune" is best for training (more upfront compile time)
model = torch.compile(model, mode="reduce-overhead", fullgraph=False)

# Warm up: the first few calls trigger JIT compilation
with torch.no_grad():
    dummy = torch.randint(0, 32000, (1, 128), device="cuda")
    for _ in range(3):   # warm-up passes
        _ = model(dummy)
```

See [Kernel Fusion, torch.compile, CUDA Graphs & Compilers](../04-kernels-efficiency/09-compilers-fusion.html) for the internals.

### Profiling with torch.profiler

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

# Profile a training step to find the bottleneck
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    with record_function("forward_pass"):
        output = model(input_ids)
    with record_function("loss_backward"):
        loss = output.loss
        loss.backward()

# Print top 20 CUDA operations sorted by total CUDA time
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# Export to Chrome trace for interactive visualization
prof.export_chrome_trace("/tmp/trace.json")
# Open chrome://tracing and load the file
```

---

## HuggingFace Ecosystem: transformers, datasets, accelerate

### Loading and inspecting models

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# --- Basic load ---
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)

# Ensure pad_token is set — required for batched training
if tok.pad_token is None:
    tok.pad_token = tok.eos_token        # most common fix
    tok.pad_token_id = tok.eos_token_id

# --- 4-bit quantized load via bitsandbytes (for fine-tuning on one GPU) ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",       # NormalFloat4, recommended for LLMs
    bnb_4bit_use_double_quant=True,  # nested quantization saves ~0.4 bit/param
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",  # distributes layers across available GPUs/CPU
)
print(f"Model footprint: {model.get_memory_footprint() / 1e9:.2f} GB")

# --- Inspect parameter counts ---
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total/1e9:.3f}B, Trainable: {trainable/1e9:.3f}B")
```

### datasets: streaming, shuffling, packing

```python
from datasets import load_dataset, DatasetDict
import datasets

# Stream a multi-hundred-GB dataset without downloading it fully
ds = load_dataset("allenai/c4", "en", split="train", streaming=True)

# Take a sample for iteration
sample_iter = ds.take(1000)
for example in sample_iter:
    text = example["text"]
    # ... tokenize, pack, etc.

# For training: load a local JSONL dataset
ds = load_dataset("json", data_files={"train": "/data/train.jsonl",
                                       "validation": "/data/val.jsonl"})

# Tokenize in parallel using map()
def tokenize(example):
    return tok(example["text"],
               truncation=True,
               max_length=2048,
               return_overflowing_tokens=True,   # yields multiple chunks if long
               return_length=True)

tokenized = ds.map(
    tokenize,
    batched=True,
    num_proc=16,                     # use 16 CPU cores
    remove_columns=ds["train"].column_names,
    desc="Tokenizing",
)
tokenized.save_to_disk("/data/tokenized_dataset")  # Arrow format, fast to reload
```

!!! tip "Practitioner tip: sequence packing for efficiency"
    Padding wastes compute. Use `DataCollatorForSeq2Seq` with `padding=True` for small datasets, but for large pretraining runs, pack multiple short examples into a single sequence (e.g., separated by `<|endoftext|>`) to reach close to 100% token utilization. See [Chat Templates, Data Formatting & Sequence Packing](../05-posttraining-alignment/02-chat-templates-packing.html) for the full treatment.

### accelerate: the multi-GPU abstraction layer

```python
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader

# Accelerator detects the environment automatically:
# - single GPU: wraps nothing
# - multi-GPU DDP: wraps in DistributedDataParallel
# - FSDP: shards the model
# - DeepSpeed: delegates to DS engine
accelerator = Accelerator(
    mixed_precision="bf16",       # automatic AMP
    gradient_accumulation_steps=4,
    log_with="wandb",
    project_dir="./runs",
)

model, optimizer, dataloader, scheduler = accelerator.prepare(
    model, optimizer, dataloader, scheduler
)

for batch in dataloader:
    with accelerator.accumulate(model):   # handles grad accum across devices
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)        # replaces loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    if accelerator.is_main_process:
        accelerator.log({"loss": loss.item()})
```

Launch commands:

```bash
# Single GPU
python train.py

# Multi-GPU on one node (4 GPUs)
accelerate launch --num_processes 4 train.py

# Multi-GPU with DeepSpeed ZeRO-3
accelerate launch --config_file ds_zero3.yaml train.py

# torchrun equivalent (lower level)
torchrun --nproc_per_node=4 --nnodes=1 train.py
```

Configure accelerate once and reuse:

```bash
accelerate config   # interactive wizard — writes ~/.cache/huggingface/accelerate/default_config.yaml
```

---

## DeepSpeed: ZeRO, Offloading & Config Files

DeepSpeed is the Swiss Army knife of large-model training. Its ZeRO optimizer stages progressively shard optimizer states (ZeRO-1), gradients (ZeRO-2), and model parameters (ZeRO-3) across GPUs, enabling models that would otherwise not fit.

For the full treatment see [Distributed Training I: Data Parallelism, DDP, ZeRO & FSDP](../03-pretraining/05-distributed-data-parallel.html) and [Megatron-LM, DeepSpeed & Parallelism in Practice](../03-pretraining/07-megatron-deepspeed.html).

### ZeRO-2 config (training fits in GPU memory, optimizer states are large)

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "fp16": { "enabled": false },
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "reduce_scatter": true,
    "allgather_bucket_size": 2e8,
    "reduce_bucket_size": 2e8,
    "overlap_comm": true
  },
  "gradient_clipping": 1.0,
  "steps_per_print": 50,
  "wall_clock_breakdown": false
}
```

### ZeRO-3 with CPU offload (fine-tune 70B on 4x80 GB GPUs)

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "bf16": { "enabled": true },
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
    "reduce_bucket_size": 1e9,
    "stage3_prefetch_bucket_size": 9e8,
    "stage3_param_persistence_threshold": 1e6,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "gradient_clipping": 1.0
}
```

!!! example "Worked example: ZeRO-3 memory savings for a 70B model"

    A 70B parameter model in bf16 requires $70 \times 10^9 \times 2 = 140 \text{ GB}$ just for model weights.

    With a standard Adam optimizer in fp32, optimizer states (momentum + variance + master weights) require $70 \times 10^9 \times (4 + 4 + 4) = 840 \text{ GB}$.

    With ZeRO-3 across $N = 8$ GPUs, each GPU holds $\frac{1}{8}$ of everything:

    $$
    \text{per-GPU weight memory} = \frac{140}{8} = 17.5 \text{ GB}
    $$
    $$
    \text{per-GPU optimizer memory} = \frac{840}{8} = 105 \text{ GB}
    $$

    Adding CPU offload moves the optimizer states entirely off-GPU. The 80 GB GPU now only needs to hold:
    - The active parameter shard: ~17.5 GB
    - Gradients (also sharded): ~17.5 GB
    - Activations for a micro-batch of 2 at sequence length 2048: ~4–8 GB (model-dependent)

    Total GPU requirement: ~40–43 GB per GPU, well within an 80 GB A100 or H100.

---

## flash-attn and bitsandbytes: Installation & Verification

### flash-attn

FlashAttention replaces the naive $O(N^2)$ memory attention with a tiled, IO-aware kernel that avoids materializing the full $N \times N$ attention matrix. See [FlashAttention I: IO-Awareness & The Online Softmax](../04-kernels-efficiency/02-flash-attention-1.html) and [FlashAttention 2 & 3: Work Partitioning, Warp Specialization & FP8](../04-kernels-efficiency/03-flash-attention-2-3.html) for the theory.

```bash
# Check your GPU compute capability first
python -c "import torch; print(torch.cuda.get_device_capability())"
# (8, 0) = A100, (8, 6) = A10G, (9, 0) = H100 — FA3 needs (9,0)

# Install (compiles in ~5-10 minutes)
pip install flash-attn --no-build-isolation

# Verify
python - <<'EOF'
import flash_attn
print(f"flash-attn version: {flash_attn.__version__}")

import torch
from flash_attn import flash_attn_qkvpacked_func

B, S, H, D = 2, 512, 32, 64  # batch, seq, heads, head_dim
qkv = torch.randn(B, S, 3, H, D, dtype=torch.bfloat16, device="cuda")
out = flash_attn_qkvpacked_func(qkv, dropout_p=0.0, causal=True)
print(f"Output shape: {out.shape}, dtype: {out.dtype}")
# Expected: torch.Size([2, 512, 32, 64])
EOF
```

Enable FlashAttention in transformers by setting `attn_implementation`:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # requires flash-attn installed
    device_map="auto",
)
```

### bitsandbytes

`bitsandbytes` enables 8-bit optimizers (reducing optimizer memory by 50%) and 4-bit weight quantization (NF4/INT4) for loading and fine-tuning large models on consumer hardware. See [Quantization II: INT4/INT8/FP8, GGUF, bitsandbytes & QAT](../04-kernels-efficiency/08-quantization-formats-qat.html) for the quantization theory.

```python
import bitsandbytes as bnb
import torch

# 8-bit Adam — same API as torch.optim.Adam, half the memory
optimizer = bnb.optim.Adam8bit(
    model.parameters(),
    lr=2e-4,
    betas=(0.9, 0.999),
)

# Check that quantized layers loaded correctly
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B",
                                              quantization_config=bnb_config,
                                              device_map="auto")

# Verify quantization
for name, module in model.named_modules():
    if hasattr(module, 'weight') and hasattr(module.weight, 'quant_type'):
        print(f"{name}: {module.weight.quant_type}, "
              f"shape={module.weight.shape}")
        break  # just show the first quantized layer
```

---

## vLLM and SGLang: Spinning Up Inference Servers

### vLLM

vLLM is the de facto standard for high-throughput LLM serving. Its key innovation — PagedAttention — is covered in [vLLM: Architecture, PagedAttention & Internals](../07-inference-serving/03-vllm-internals.html) and [PagedAttention & KV-Cache Memory Management](../04-kernels-efficiency/06-paged-attention-kv.html).

```bash
# Start an OpenAI-compatible server (drop-in replacement for OpenAI API)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \          # split model across 2 GPUs
    --max-model-len 8192 \              # cap context window
    --gpu-memory-utilization 0.90 \    # leave 10% headroom
    --port 8000

# Query the running server with curl
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "messages": [{"role": "user", "content": "Explain backpropagation."}],
    "max_tokens": 256
  }'
```

```python
# Use vLLM programmatically for offline batch inference
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    dtype="bfloat16",
    tensor_parallel_size=2,
    max_model_len=8192,
    gpu_memory_utilization=0.85,
    enable_prefix_caching=True,   # cache common prefixes (system prompts)
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=512,
    stop=["<|eot_id|>"],   # Llama-3 end-of-turn token
)

prompts = [
    "What is gradient descent?",
    "Explain the transformer architecture.",
]
outputs = llm.generate(prompts, sampling_params)
for out in outputs:
    print(out.outputs[0].text[:100])
```

### SGLang

SGLang adds a structured generation runtime and RadixAttention (prefix caching at the radix-tree level) on top of an efficient inference engine. See [SGLang: RadixAttention & Structured Programs](../07-inference-serving/04-sglang-radixattention.html).

```bash
# Launch SGLang server
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3-8B-Instruct \
    --tp 2 \
    --host 0.0.0.0 \
    --port 30000
```

```python
import sglang as sgl

# SGLang's structured generation: constrain output to a JSON schema
@sgl.function
def classify_sentiment(s, review):
    s += sgl.system("You are a sentiment classifier.")
    s += sgl.user(review)
    s += sgl.assistant(
        sgl.gen("sentiment",
                choices=["positive", "negative", "neutral"],  # constrained
                max_tokens=1)
    )

# Fork for parallel evaluation of multiple reviews
reviews = ["This movie was great!", "Terrible service.", "It was okay."]
states = classify_sentiment.run_batch(
    [{"review": r} for r in reviews],
    progress_bar=True,
)
for state, review in zip(states, reviews):
    print(f"{review!r} -> {state['sentiment']}")
```

---

## TRL and veRL: Post-Training Infrastructure

### TRL for SFT and RL

TRL (Transformer Reinforcement Learning) from HuggingFace is the most accessible entry point for post-training workflows. For the full conceptual treatment see [TRL: HuggingFace's RL Library](../06-rl-infra/03-trl.html).

```python
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch

model_id = "meta-llama/Meta-Llama-3-8B"
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)
tok = AutoTokenizer.from_pretrained(model_id)
tok.pad_token = tok.eos_token

# LoRA config: fine-tune only ~0.5% of parameters
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 41,943,040 || all params: 8,072,220,672 || trainable%: 0.5196

ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

training_args = SFTConfig(
    output_dir="./llama3-sft",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to="wandb",
    dataset_text_field="prompt",
    max_seq_length=2048,
    packing=True,                     # pack short examples together
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    tokenizer=tok,
)
trainer.train()
trainer.save_model("./llama3-sft-final")
```

```python
# GRPO (Group Relative Policy Optimization) with TRL
from trl import GRPOTrainer, GRPOConfig

def reward_fn(completions, prompts, **kwargs):
    """Custom reward: give +1 if response contains a number, else 0."""
    return [1.0 if any(c.isdigit() for c in comp) else 0.0
            for comp in completions]

grpo_config = GRPOConfig(
    output_dir="./llama3-grpo",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    learning_rate=1e-6,
    bf16=True,
    num_generations=8,    # G in GRPO: sample G completions per prompt
    max_new_tokens=256,
    report_to="wandb",
)
trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_fn,
    args=grpo_config,
    train_dataset=ds,
)
trainer.train()
```

For veRL (HybridFlow) and more complex multi-controller RL setups, see [veRL: HybridFlow & The Single-Controller Architecture](../06-rl-infra/04-verl.html).

---

## Weights & Biases (wandb): Experiment Tracking

W&B is the standard experiment tracker for LLM work. Getting logging right from the start saves enormous debugging time.

```python
import wandb
import os

# --- Initialize a run ---
wandb.init(
    project="llm-stack-experiments",
    name="llama3-sft-lora-r16",
    config={
        "model": "meta-llama/Meta-Llama-3-8B",
        "lora_r": 16,
        "lr": 2e-4,
        "batch_size": 16,
        "epochs": 3,
        "dataset": "ultrachat_200k",
    },
    tags=["sft", "lora", "llama3"],
)

# --- During training: log metrics ---
for step, batch in enumerate(dataloader):
    loss = train_step(batch)
    if step % 10 == 0:
        wandb.log({
            "train/loss": loss.item(),
            "train/lr": scheduler.get_last_lr()[0],
            "train/grad_norm": grad_norm,
            "train/tokens_per_sec": tokens_per_sec,
        }, step=step)

# --- Log model outputs as a Table for qualitative eval ---
columns = ["prompt", "generation", "reward"]
rows = []
for prompt, gen, reward in eval_samples:
    rows.append([prompt, gen, reward])

wandb.log({"eval/samples": wandb.Table(columns=columns, data=rows)})

# --- Finish the run ---
wandb.finish()
```

```bash
# Login once per machine
wandb login  # prompts for API key from wandb.ai

# Set offline mode for air-gapped clusters
export WANDB_MODE=offline   # logs to disk; sync later with `wandb sync`
export WANDB_DIR=/data/wandb_logs

# Resume a crashed run by its ID
wandb.init(project="...", id="abc123", resume="allow")
```

!!! tip "Practitioner tip: artifact versioning"
    Use `wandb.Artifact` to version your datasets and model checkpoints. This creates a provenance trail that links every model version back to the exact data and code that produced it — invaluable when you need to diagnose a regression three months later.

```python
# Log a model checkpoint as a W&B artifact
artifact = wandb.Artifact("llama3-sft-checkpoint", type="model")
artifact.add_dir("./llama3-sft-final")
wandb.log_artifact(artifact)

# Later: download the exact checkpoint
api = wandb.Api()
artifact = api.artifact("myteam/llm-experiments/llama3-sft-checkpoint:v3")
artifact.download(root="./restored_checkpoint")
```

---

## Debugging, Profiling & Common Gotchas

### The diagnostic toolkit

```bash
# Check GPU utilization and memory in real time
watch -n 1 nvidia-smi

# More detailed per-process GPU memory usage
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader

# Check GPU connectivity (NVLink, PCIe topology)
nvidia-smi topo -m

# Check NCCL version
python -c "import torch; print(torch.cuda.nccl.version())"

# Check if all-reduce works across GPUs (basic connectivity test)
torchrun --nproc_per_node=4 - <<'EOF'
import torch, torch.distributed as dist
dist.init_process_group("nccl")
rank = dist.get_rank()
t = torch.tensor([rank], dtype=torch.float32, device=f"cuda:{rank}")
dist.all_reduce(t, op=dist.ReduceOp.SUM)
expected = sum(range(dist.get_world_size()))
assert t.item() == expected, f"Expected {expected}, got {t.item()}"
if rank == 0:
    print(f"All-reduce test passed! Sum = {t.item()}")
dist.destroy_process_group()
EOF
```

### Memory profiling

```python
import torch

# Snapshot GPU memory before and after a suspicious operation
torch.cuda.reset_peak_memory_stats()
before = torch.cuda.memory_allocated() / 1e9

output = model(input_ids)   # the operation under scrutiny

after = torch.cuda.memory_allocated() / 1e9
peak = torch.cuda.max_memory_allocated() / 1e9
print(f"Before: {before:.2f} GB, After: {after:.2f} GB, Peak: {peak:.2f} GB")

# Full memory snapshot (PyTorch >= 2.1)
torch.cuda.memory._record_memory_history(max_entries=100_000)
output = model(input_ids)
snapshot = torch.cuda.memory._snapshot()
torch.cuda.memory._dump_snapshot("/tmp/mem_snapshot.pickle")
# Visualize at: https://pytorch.org/memory_viz
```

### Common errors and their fixes

```text
Error: CUDA error: device-side assert triggered
Fix:  Run with CUDA_LAUNCH_BLOCKING=1 to get a synchronous stack trace.
      Usually caused by out-of-range indices, wrong tensor shapes, or
      loss computed on -100 labels (ignore_index) without proper masking.

Error: RuntimeError: Expected all tensors to be on the same device
Fix:  Print t.device for each tensor before the failing op.
      Often caused by model on 'cuda:0' but input on 'cpu' or 'cuda:1'.

Error: torch.OutOfMemoryError: CUDA out of memory
Fix:  (1) Reduce batch size or sequence length.
      (2) Enable gradient checkpointing: model.gradient_checkpointing_enable()
      (3) Use bitsandbytes 4-bit loading.
      (4) Enable ZeRO-3 or FSDP.
      (5) Check for tensor leaks: are you accumulating outputs in a list?

Error: NCCL error: unhandled system error / Connection reset by peer
Fix:  Usually a node failure or network partition. Check:
      - All nodes reachable: ping / ssh
      - NCCL_SOCKET_IFNAME points to the right NIC
      - No firewall blocking NCCL ports (default: 29500)

Error: ValueError: Tokenizer does not have a padding token
Fix:  tokenizer.pad_token = tokenizer.eos_token
      Always do this before creating a DataCollator.
```

!!! interview "Interview Corner"
    **Q:** You launch a distributed training job with `torchrun --nproc_per_node=8` on a single 8xA100 node. After a few hundred steps, training hangs indefinitely with no error message. What are the most likely causes and how would you diagnose each?

    **A:** The most common causes of a silent hang in distributed training are:

    1. **NCCL deadlock from uneven collective calls.** If one rank enters an `all_reduce` or `barrier` that others do not (e.g., due to a conditional branch gated on `rank == 0`), the job hangs waiting for the missing rank. Diagnosis: `NCCL_DEBUG=INFO` will show which rank is stuck on which collective. Fix: ensure all collectives are executed by all ranks unconditionally.

    2. **CPU-side data loading bottleneck.** If `DataLoader` workers are slower than the GPU, the GPU stalls waiting for the next batch. Diagnosis: `nvidia-smi` shows near-zero GPU utilization. Fix: increase `num_workers`, pin memory with `pin_memory=True`, or use a faster data format (Arrow/HDF5 instead of JSONL).

    3. **Deadlock in custom forward/backward hooks.** Certain Python-level locks (e.g., a shared queue between processes) can deadlock if one process raises an exception while holding the lock. Diagnosis: `py-spy dump --pid <pid>` shows the Python call stack of the hung process. Fix: use `torch.distributed` primitives rather than Python multiprocessing primitives for cross-rank communication.

    4. **NVLink or InfiniBand failure.** A flaky interconnect causes one NCCL operation to time out. Diagnosis: `nvidia-smi nvlink -e` shows NVLink errors; `NCCL_DEBUG=WARN` shows IB errors. Fix: check hardware health with `ibdiagnet`; disable the faulty link.

---

## Useful One-Liners & Quick-Reference Commands

```bash
# ── Model inspection ──────────────────────────────────────────────
# Count parameters of any HF model without loading weights
python -c "
from transformers import AutoConfig
from transformers.utils import cached_file
import json, math
config = AutoConfig.from_pretrained('meta-llama/Meta-Llama-3-8B')
print(config)
"

# Download a model to local cache
huggingface-cli download meta-llama/Meta-Llama-3-8B --local-dir /data/llama3-8b

# Convert HF checkpoint to safetensors
python -c "
from transformers import AutoModelForCausalLM
import torch
model = AutoModelForCausalLM.from_pretrained('/data/llama3-8b', torch_dtype=torch.bfloat16)
model.save_pretrained('/data/llama3-8b-safetensors', safe_serialization=True)
"

# ── GPU monitoring ─────────────────────────────────────────────────
# Continuous GPU stats (refresh every 0.5s)
nvidia-smi dmon -s mu -d 0.5

# GPU power consumption (useful for cost estimation)
nvidia-smi --query-gpu=name,power.draw,power.limit --format=csv

# ── Cluster / distributed ──────────────────────────────────────────
# Launch 2-node job with torchrun (run on each node)
# Node 0 (master):
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
         --master_addr=10.0.0.1 --master_port=29500 train.py
# Node 1:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
         --master_addr=10.0.0.1 --master_port=29500 train.py

# ── Checkpoint utilities ───────────────────────────────────────────
# Inspect a safetensors checkpoint (keys, shapes, dtypes)
python -c "
from safetensors import safe_open
with safe_open('/data/model.safetensors', framework='pt', device='cpu') as f:
    for key in list(f.keys())[:10]:
        t = f.get_tensor(key)
        print(f'{key}: {t.shape}, {t.dtype}')
"

# ── Tokenizer debugging ────────────────────────────────────────────
# Inspect how a string tokenizes
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
text = 'Hello, world! The answer is 42.'
ids = tok.encode(text)
tokens = tok.convert_ids_to_tokens(ids)
print(list(zip(tokens, ids)))
"
```

---

!!! key "Key Takeaways"
    - Pin the full CUDA/PyTorch/flash-attn/bitsandbytes version matrix at project start; mismatches are the single most common source of mysterious errors.
    - `flash-attn` must be installed with `--no-build-isolation` after PyTorch; it compiles against your installed torch headers, not an isolated build environment.
    - Set `tokenizer.pad_token = tokenizer.eos_token` immediately after loading any decoder-only tokenizer; forgetting this corrupts batched training.
    - `accelerate launch` is the recommended single entry point for multi-GPU training — it handles DDP, FSDP, and DeepSpeed through the same training script with only a config change.
    - ZeRO-3 with CPU offload trades GPU memory for CPU memory and PCIe bandwidth; it enables fine-tuning 70B+ models on 4–8 consumer A100s but slows iteration per step.
    - `torch.compile(mode="reduce-overhead")` gives 20–40% speedup on inference with nearly zero code change; warm up with a few dummy passes before benchmarking.
    - W&B artifacts link every model version to its training data and code — use them from day one to enable reproducibility and regression debugging months later.
    - For silent hangs in distributed training, `NCCL_DEBUG=INFO` and `py-spy dump` are your first tools; the root cause is almost always an uneven collective or a CPU-side bottleneck.
    - vLLM and SGLang require separate environments from training because they pin specific torch versions; maintain `train.yml` and `serve.yml` conda environments.

## Further Reading

- **PyTorch Documentation: torch.compile** — official guide covering `mode=`, `fullgraph=`, and the TorchInductor backend. pytorch.org
- **DeepSpeed ZeRO: Memory Optimizations Toward Training Trillion Parameter Models** — Rajbhandari et al., 2020. The foundational ZeRO paper.
- **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** — Dao et al., 2022. Essential reading for understanding why flash-attn exists.
- **bitsandbytes: 8-bit Optimizers via Block-wise Quantization** — Dettmers et al., 2021. The paper behind the 8-bit Adam optimizer.
- **QLoRA: Efficient Finetuning of Quantized LLMs** — Dettmers et al., 2023. The canonical recipe combining NF4 quantization with LoRA.
- **TRL: Transformer Reinforcement Learning** — HuggingFace, github.com/huggingface/trl. The library reference for SFT, GRPO, PPO trainers.
- **vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention** — Kwon et al., 2023. The PagedAttention system paper.
- **Accelerate: Training and inference at scale made simple** — HuggingFace, github.com/huggingface/accelerate.
- **NVIDIA Deep Learning Performance Guide** — NVIDIA Developer documentation; covers `cudnn.benchmark`, TF32, and cuBLAS tuning flags.
