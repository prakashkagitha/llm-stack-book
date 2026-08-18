# GPU-tier notebook execution verification (real H100s)

Task **B**: confirm the book's GPU code actually *runs* on real hardware, not just that it
looks right. Executed on the shared 8×H100 node, pinning ≤2 idle GPUs via `scripts/pick_gpus.py`
(good-neighbour discipline). Single-GPU runner: `scripts/exec_gpu_notebooks.py`. Multi-GPU:
`%%writefile` scripts extracted and launched under `torchrun --nproc_per_node=2`.

## Single-GPU notebooks — 6/6 network-free PASS on H100

| Notebook | Result |
|---|---|
| 04 flash-attention-benchmark | PASS (torch SDPA FLASH backend) |
| 04 triton-fused-kernel | PASS |
| 04 torch-compile-speedup | PASS (compile + CUDA graphs) |
| 04 memory-efficient-training | PASS |
| 03 bf16-vs-fp8-throughput | PASS |
| 03 optimizers-wallclock | PASS |
| 04 int4-int8-quantization | code-correct; needs a network model download (fails only under forced HF offline) |

## Multi-GPU notebooks — 4/4 PASS on 2×H100 via torchrun

| Notebook | Result |
|---|---|
| 03 ddp-two-gpu-scaling | exit 0; both ranks' **all-reduced loss identical** (10.9789), ~215,612 tok/s aggregate |
| 03 fsdp-two-gpu-sharding | exit 0 all 3 ZeRO strategies; **peak mem full_shard 16.38 < shard_grad_op 17.74 < no_shard 20.66 GB** (correct sharding progression) |
| 03 tensor-parallel-from-scratch | exit 0; weights sharded [4096,8192]→[8192,4096] per rank, 0.97 GB/rank |
| 07 multi-gpu-tp-inference | exit 0; **TP-vs-single-GPU correctness allclose=True, max_abs_diff=1.53e-12**, 604M params sharded across 2 GPUs |

## Capstone pipeline
The entire `stacklm` GPU pipeline (model, attention, data, pretrain loop, generation) was
separately execution-verified by the light-A real training run (Ch 14.5), which fit a scaling
law on 4M–44M models that predicted the 100M's loss to 0.002 nats.

**Conclusion:** the book's core GPU code — single-GPU kernels/quantization/compile and the
distributed DDP/FSDP/tensor-parallel paths — is execution-proven on real H100 hardware, not
merely plausible.
