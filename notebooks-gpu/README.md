# GPU-tier notebooks

Real, runnable notebooks for the compute-heavy chapters, written for a **single NVIDIA H100 (80GB)** or **2x A100** — the tier where the ideas in these chapters actually show up (real FlashAttention/Triton kernels, `torch.compile`, quantization, fp8, a real training loop + MFU, and multi-GPU DDP/FSDP/tensor-parallel).

> These complement the CPU toy-scale notebooks under `notebooks/`. Each has a pip cell and a hardware banner; expected throughput/MFU/speedup numbers are given as rough orders of magnitude, not fabricated benchmarks — run them to get your own.


## Index

- [`03-pretraining/bf16-vs-fp8-throughput.ipynb`](03-pretraining/bf16-vs-fp8-throughput.ipynb) — bf16 vs FP8 Matmul Throughput on Hopper (8 code cells)
- [`03-pretraining/ddp-two-gpu-scaling.ipynb`](03-pretraining/ddp-two-gpu-scaling.ipynb) — Data-Parallel Training Across 2 GPUs (DDP) & Scaling Efficiency (7 code cells)
- [`03-pretraining/fsdp-two-gpu-sharding.ipynb`](03-pretraining/fsdp-two-gpu-sharding.ipynb) — Sharding a Model with FSDP (ZeRO) Across 2 GPUs (7 code cells)
- [`03-pretraining/optimizers-wallclock.ipynb`](03-pretraining/optimizers-wallclock.ipynb) — AdamW vs Lion vs Muon: Wall-Clock & Memory on GPU (8 code cells)
- [`03-pretraining/tensor-parallel-from-scratch.ipynb`](03-pretraining/tensor-parallel-from-scratch.ipynb) — Tensor Parallelism From Scratch: Splitting a Layer Across 2 GPUs (5 code cells)
- [`03-pretraining/train-small-gpt-mfu.ipynb`](03-pretraining/train-small-gpt-mfu.ipynb) — Train a Small GPT on One H100: Loss Curve, Throughput & MFU (7 code cells)
- [`04-kernels-efficiency/flash-attention-benchmark.ipynb`](04-kernels-efficiency/flash-attention-benchmark.ipynb) — FlashAttention vs Naive Attention: A GPU Benchmark (7 code cells)
- [`04-kernels-efficiency/int4-int8-quantization.ipynb`](04-kernels-efficiency/int4-int8-quantization.ipynb) — INT8 & NF4 Quantization: Memory and Latency on GPU (6 code cells)
- [`04-kernels-efficiency/memory-efficient-training.ipynb`](04-kernels-efficiency/memory-efficient-training.ipynb) — Fitting a Bigger Model: Activation Checkpointing + 8-bit Optimizer (8 code cells)
- [`04-kernels-efficiency/torch-compile-speedup.ipynb`](04-kernels-efficiency/torch-compile-speedup.ipynb) — torch.compile & CUDA Graphs: Measuring Real Speedups (9 code cells)
- [`04-kernels-efficiency/triton-fused-kernel.ipynb`](04-kernels-efficiency/triton-fused-kernel.ipynb) — Writing a Fused Triton Kernel (Fused Softmax) and Benchmarking It (9 code cells)
- [`07-inference-serving/multi-gpu-tp-inference.ipynb`](07-inference-serving/multi-gpu-tp-inference.ipynb) — Multi-GPU Inference: Tensor-Parallel Serving of a Model Too Big for One GPU (7 code cells)
- [`07-inference-serving/speculative-decoding-speedup.ipynb`](07-inference-serving/speculative-decoding-speedup.ipynb) — Speculative Decoding: Measuring the Real Wall-Clock Speedup (8 code cells)
- [`14-capstone/stack100m-single-gpu-h100.ipynb`](14-capstone/stack100m-single-gpu-h100.ipynb) — Capstone: Pretrain Stack-100M on a Single H100 (Real Run) (10 code cells)
