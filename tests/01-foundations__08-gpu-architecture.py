"""
Executable smoke test for content/01-foundations/08-gpu-architecture.md

Blocks tested (chapter's own numbering):
  - block #7 (~line 399): model/KV-cache footprint sanity check (capacity math)
  - block #8 (~line 435): classify_op() roofline-style compute/memory-bound classifier

Blocks explicitly SKIPPED (see reasons inline where relevant):
  - #0, #1, #4: non-python (ASCII diagrams / prose)
  - #2, #3: needs-gpu (Tensor Core / CUDA-only demonstrations)
  - #5, #6: fragment (not standalone; #6 is the arithmetic_intensity/roofline_perf
    block feeding into prose-only discussion elsewhere in the chapter and is not
    part of the two blocks assigned to this test)

Everything below is copied verbatim from the chapter's fenced code blocks #7 and #8,
concatenated in chapter order, with each top-level function actually called on a
tiny CPU-safe input so the demonstrated logic executes for real.
"""

import torch  # noqa: F401  (imported here to mirror block #8's own `import torch`)


# ---------------------------------------------------------------------------
# Block #7 (~line 399, 20 lines): model footprint / KV-cache capacity check
# ---------------------------------------------------------------------------

# Where does your 70B model physically fit? A capacity + bandwidth sanity check.
def model_footprint_gb(params_b, bytes_per_param):
    return params_b * 1e9 * bytes_per_param / 1e9   # -> GB

def kv_cache_gb(layers, kv_heads, head_dim, seq, batch, bytes_per_elem=2):
    # K and V, both, per layer, per (kv) head: factor of 2 for K&V.
    elems = 2 * layers * kv_heads * head_dim * seq * batch
    return elems * bytes_per_elem / 1e9

# Llama-70B-ish: 70B params, 80 layers, 8 KV heads (GQA), head_dim 128.
for dtype, bpp in [("bf16", 2), ("fp8", 1), ("int4", 0.5)]:
    w = model_footprint_gb(70, bpp)
    print(f"70B weights in {dtype:5s}: {w:6.1f} GB")

kv = kv_cache_gb(layers=80, kv_heads=8, head_dim=128,
                 seq=8192, batch=32, bytes_per_elem=2)
print(f"KV cache (8k ctx, batch 32, GQA-8, bf16): {kv:.1f} GB")
# 70B bf16 weights ~140 GB -> does NOT fit one 80GB H100; fits one 192GB B200,
# or shards across 2x H100 / 1x via int4. Bandwidth then sets decode speed.

# --- sanity checks on the chapter's own arithmetic (book's actual code, verbatim above) ---
w_bf16 = model_footprint_gb(70, 2)
assert abs(w_bf16 - 140.0) < 1e-6, "70B bf16 footprint should be ~140 GB as the prose states"
w_fp8 = model_footprint_gb(70, 1)
assert abs(w_fp8 - 70.0) < 1e-6
w_int4 = model_footprint_gb(70, 0.5)
assert abs(w_int4 - 35.0) < 1e-6
assert kv > 0


# ---------------------------------------------------------------------------
# Block #8 (~line 435, 23 lines): classify_op() roofline classifier
# ---------------------------------------------------------------------------

def classify_op(flops, hbm_bytes, peak_flops=990e12, peak_bw=3.35e12):
    """A 30-second back-of-envelope to decide where to spend optimization effort."""
    intensity = flops / hbm_bytes
    ridge = peak_flops / peak_bw
    attainable = min(peak_flops, intensity * peak_bw)
    bound = "compute" if intensity > ridge else "memory"
    return dict(intensity=round(intensity, 2),
                ridge=round(ridge, 1),
                bound=bound,
                attainable_TFLOPs=round(attainable / 1e12, 1),
                advice=("tile/fuse-for-TC; you can chase peak FLOPs"
                        if bound == "compute"
                        else "cut bytes: fuse, recompute, lower precision, reuse"))

# Example: a fused LayerNorm over a (batch*seq, hidden) tensor, bf16.
rows, hidden = 4096 * 2048, 8192       # tokens x hidden
elems = rows * hidden
# LayerNorm: ~a handful of FLOPs/elem; reads + writes the tensor once each.
ln = classify_op(flops=8 * elems, hbm_bytes=2 * elems * 2)   # read+write, 2B/elem
print("LayerNorm:", ln)   # -> memory-bound: the reason norms get fused into matmuls

# --- sanity checks on the chapter's own claim (book's actual code, verbatim above) ---
assert ln["bound"] == "memory", "chapter's prose claims LayerNorm is memory-bound"
assert ln["intensity"] == 2.0  # 8 elems FLOPs / 4 bytes/elem = 2.0 FLOP/byte

# A tiny second call on a deliberately compute-heavy op, to exercise the
# "compute" branch of classify_op too (still the book's own function, just a
# second, minimal invocation to demonstrate both branches actually run).
mm = classify_op(flops=2 * 8192**3, hbm_bytes=3 * 8192**2 * 2)
assert mm["bound"] == "compute"

print("All GPU-architecture chapter blocks (#7, #8) executed successfully.")
