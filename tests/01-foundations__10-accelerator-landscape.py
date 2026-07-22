"""
Executable smoke test for content/01-foundations/10-accelerator-landscape.md

Tests the 2 heuristically CPU-runnable Python blocks, assembled in chapter order:
  - block #0 (line ~59, 31 lines): JAX on TPU -- data-parallel + tensor-parallel
    sharded matmul across a logical device mesh.
  - block #4 (line ~268, 43 lines): a toy hardware picker (dataclass-based
    capacity/bandwidth/FLOPS decision function).

SKIPPED (not assembled here -- see HARD RULES in the task, these are the
book's other blocks, none of which are safely CPU-runnable/standalone-Python):
  - block #1 (line ~111): AWS Trainium via `torch_xla` -- targets an XLA
    'xla:0' NeuronCore device via the AWS Neuron SDK. Neither `torch_xla`
    nor a NeuronCore is available in a plain-CPU CI box.
    SKIP(needs-gpu/needs-accelerator-runtime): torch_xla + NeuronCore device.
  - block #2 (line ~154): a HIP/C++ SAXPY kernel (`#include <hip/hip_runtime.h>`,
    `__global__ void saxpy(...)`) -- not Python at all.
    SKIP(non-python): C++ HIP kernel, needs `hipcc` and an AMD GPU.
  - block #3 (line ~201): Intel Gaudi via `habana_frameworks` -- targets the
    'hpu' device exposed by SynapseAI on real Gaudi hardware.
    SKIP(needs-gpu/needs-accelerator-runtime): habana_frameworks + HPU device.

Run:
    python3 tests/01-foundations__10-accelerator-landscape.py
"""
import os
import sys

# ---------------------------------------------------------------------------
# Block #0 (line ~59): "JAX on TPU: data-parallel + tensor-parallel matmul
# across a pod slice." `jax` is NOT in the CI-guaranteed import list
# (numpy/torch-cpu/einops/sklearn/stdlib), so it is imported defensively and
# the block is executed only if it is actually importable; otherwise it is
# honestly SKIPPED (never silently faked).
#
# CPU adaptation (device-count shim): the book's mesh is a *logical* TPU
# pod-slice mesh; on a single CPU host JAX normally reports exactly one
# device, so `mesh_utils.create_device_mesh((4, 2))` (which needs 8 devices)
# would fail. Forcing JAX to expose 8 CPU "devices" via XLA_FLAGS is the
# standard, well-documented way to exercise multi-device JAX sharding code
# on one machine -- it changes nothing about the sharding LOGIC being taught,
# only how many (fake) physical devices back it. Must be set before `import
# jax`.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

try:
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
    from jax.experimental import mesh_utils
except Exception:
    jax = None


def run_block_0():
    """
    Book code (line ~59), copied verbatim aside from one CPU-only glue
    change: the (8192, 8192) / (1024, 8192) illustrative shapes are shrunk to
    (512, 512) / (128, 512) to keep the CPU matmul small and fast. This is a
    "tiny shapes" substitution allowed by the harness rules (the numbers were
    already flagged in the book itself as "ILLUSTRATIVE order-of-magnitude" a
    few sections later) -- the divisibility-by-mesh-axis logic the block
    demonstrates is unchanged: model axis has 2 devices (512 % 2 == 0),
    data axis has 4 devices (128 % 4 == 0).
    """
    # 1) Discover the physical TPU chips and arrange them into a logical 2D mesh.
    #    Axis "data" = data parallelism; axis "model" = tensor parallelism.
    devices = mesh_utils.create_device_mesh((4, 2))   # e.g. an 8-chip slice
    mesh = Mesh(devices, axis_names=("data", "model"))

    def shard(x, spec):
        return jax.device_put(x, NamedSharding(mesh, spec))

    # 2) Shard a weight matrix's columns across the "model" axis (tensor parallel),
    #    and the activation batch across the "data" axis (data parallel).
    W = shard(jnp.ones((512, 512)),  P(None, "model"))   # [in, out/TP]
    x = shard(jnp.ones((128, 512)),  P("data", None))    # [batch/DP, in]

    @jax.jit                      # <-- this single decorator invokes XLA.
    def layer(x, W):
        # You write a plain matmul. XLA sees the shardings on x and W and
        # automatically emits the all-gather over "model" needed to form the
        # full output, fused with the matmul, tiled for the 128x128 MXU.
        return jnp.tanh(x @ W)

    y = layer(x, W)               # runs across all 8 (fake CPU) devices.
    print(y.shape, y.sharding)    # (128, 512), sharded as XLA decided.
    assert y.shape == (128, 512)
    assert bool(jnp.all(jnp.abs(y) <= 1.0))   # tanh output is bounded in [-1, 1]
    return y


if jax is not None:
    print("=== block #0: JAX/TPU sharded matmul (CPU, 8 simulated devices) ===")
    run_block_0()
    print("block #0 OK\n")
else:
    print("=== block #0 SKIPPED(missing-optional-dependency): `jax` is not "
          "installed. Block executes above if `jax` is importable; it is "
          "guarded per the harness rule for non-guaranteed third-party "
          "imports. ===\n")


# ---------------------------------------------------------------------------
# Block #4 (line ~268): "A toy hardware picker." Pure stdlib (dataclasses),
# no optional dependency -- runs unconditionally.
# ---------------------------------------------------------------------------
# A toy hardware picker. Numbers are ILLUSTRATIVE order-of-magnitude specs
# meant to teach the REASONING; always re-check the current datasheet.
from dataclasses import dataclass


@dataclass
class Chip:
    name: str
    hbm_gb: float          # capacity
    hbm_tbs: float         # bandwidth, TB/s
    fp8_tflops: float      # peak FP8 matmul throughput
    link_gbs: float        # per-chip interconnect bandwidth, GB/s


# Illustrative figures -- teach the method, not the exact values.
CHIPS = [
    Chip("H100-SXM",   80,  3.35,  1979,  450),   # NVLink
    Chip("MI300X",    192,  5.3,   2615,  448),   # Infinity Fabric
    Chip("TPU v5p",    95,  2.76,   918,  600),   # ICI (bf16-class peak)
    Chip("Gaudi3",    128,  3.7,   1835,  300),   # RoCE/Ethernet (aggregate)
]


def kv_bytes_per_token(L, H_kv, d_h, bytes_per_elt=2):
    return 2 * L * H_kv * d_h * bytes_per_elt          # K and V


def pick(workload, params_b, dtype_bytes, L, H_kv, d_h,
         ctx_len, batch, target="decode"):
    need_weights = params_b * 1e9 * dtype_bytes
    kv_per_tok   = kv_bytes_per_token(L, H_kv, d_h)
    need_kv      = kv_per_tok * ctx_len * batch
    need_total   = (need_weights + need_kv) / 1e9       # GB
    print(f"[{workload}] need ~{need_total:.0f} GB "
          f"(weights {need_weights/1e9:.0f} + KV {need_kv/1e9:.0f})")
    results = []
    for c in CHIPS:
        n_dev = -(-need_total // c.hbm_gb)              # ceil-divide to fit
        # decode is bandwidth-bound -> rank by aggregate HBM TB/s;
        # prefill/training is compute-bound -> rank by aggregate FP8 TFLOPS.
        score = (c.hbm_tbs if target == "decode" else c.fp8_tflops) * n_dev
        print(f"  {c.name:10s}: fits on {int(n_dev)} dev, "
              f"score={score:.0f} ({'TB/s' if target=='decode' else 'TFLOPS'} agg)")
        results.append((c.name, n_dev, score))
    return results


print("=== block #4: toy hardware picker ===")
# Llama-70B-ish: 70B params, fp8 weights, 80 layers, 8 KV heads (GQA),
# head_dim 128, 8k context, batch 32, decode-bound serving.
results = pick("serve-70B-decode", 70, 1, 80, 8, 128, ctx_len=8192, batch=32)

# Exercise the function's own logic, not just its printed output:
# check the worked-example numbers from the chapter's !!! example block.
# NOTE: the chapter originally mis-rounded 327,680 bytes/token as "~0.31 MB"
# (it is actually ~0.33 MB), which cascaded into "~82 GB" KV / "~152 GB"
# total. That was a real arithmetic bug in the book, fixed in the .md to the
# correct ~0.33 MB / ~86 GB / ~156 GB; the assertions below check the
# corrected (actual) numbers.
kv_per_tok = kv_bytes_per_token(80, 8, 128)
assert kv_per_tok == 327680, kv_per_tok  # matches the chapter's worked example
weights_gb = 70 * 1e9 * 1 / 1e9
kv_gb = kv_per_tok * 8192 * 32 / 1e9
total_gb = weights_gb + kv_gb
assert abs(total_gb - 156) < 1.0, total_gb  # chapter states "~156 GB" (corrected)

# The MI300X (192 GB) should fit the whole model on a single device;
# H100 (80 GB) should need at least 2.
by_name = {name: n_dev for name, n_dev, _ in results}
assert by_name["MI300X"] == 1
assert by_name["H100-SXM"] == 2
print("block #4 OK\n")

print("All runnable blocks completed.")
sys.exit(0)
