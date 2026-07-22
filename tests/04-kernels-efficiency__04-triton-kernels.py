"""
Runnable-code test for content/04-kernels-efficiency/04-triton-kernels.md

Block inventory (from the chapter, heuristic classification):
  #0  fragment  - imports/prose fragment, not standalone           -> SKIP(fragment)
  #1  CPU-safe  - 2-D pointer-block arithmetic (line ~51, 5 lines)  -> TESTED
  #2  needs-gpu - Kernel 1: vector add (@triton.jit, torch cuda)    -> SKIP(needs-gpu)
  #3  needs-gpu - Kernel 1 driver (`if __name__ == "__main__"`)     -> SKIP(needs-gpu)
  #4  needs-gpu - Kernel 2: fused softmax kernel                    -> SKIP(needs-gpu)
  #5  needs-gpu - Kernel 2 driver                                   -> SKIP(needs-gpu)
  #6  needs-gpu - Kernel 3: matmul kernel (autotune, tl.dot)        -> SKIP(needs-gpu)
  #7  needs-gpu - Kernel 3 driver                                   -> SKIP(needs-gpu)
  #8  needs-gpu - Kernel 4: flash attention kernel                  -> SKIP(needs-gpu)
  #9  needs-gpu - Kernel 4 driver                                   -> SKIP(needs-gpu)
  #10 needs-gpu - Kernel 4 autotuned variant (skeleton, `pass` body) -> SKIP(needs-gpu)
  #11 needs-gpu - Kernel 5 preprocessing kernel (attn_bwd_preprocess) -> SKIP(needs-gpu)
  #12 needs-gpu - Kernel 5 dK/dV/dQ kernel (attn_bwd_dkdv_dq)       -> SKIP(needs-gpu)
  #13 needs-gpu - (further Kernel 5 material beyond the read window) -> SKIP(needs-gpu)

Only block #1 is CPU-runnable: it is pure pointer-arithmetic broadcasting
(`tl.arange` + `[:, None]` / `[None, :]` broadcasting), which the chapter text
explicitly says is "identical to NumPy". Triton itself requires a GPU runtime
(and typically isn't even importable without a CUDA-capable device), so the
book's own `tl.arange`/`tl.load` calls cannot execute in this CPU-only CI.

To faithfully test block #1's *logic* (the broadcasting arithmetic that turns
row/col indices into a block of memory offsets) without a GPU, we substitute
`tl.arange` with `numpy.arange` -- both produce a 1-D vector of indices, and
the `[:, None]` / `[None, :]` broadcasting rules are, per the book text,
identical between NumPy and Triton. The arithmetic expression itself
(`base_ptr + row * stride_m + col * stride_n`) is copied verbatim from the
book. We then verify the resulting offset grid against the known closed-form
answer for row-major pointer arithmetic.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Block #1 (content/04-kernels-efficiency/04-triton-kernels.md, line ~51):
#
#     # A BLOCK_M x BLOCK_N tile of pointers into a matrix with strides (stride_m, stride_n)
#     row = tl.arange(0, BLOCK_M)[:, None]   # shape (BLOCK_M, 1)
#     col = tl.arange(0, BLOCK_N)[None, :]   # shape (1, BLOCK_N)
#     ptrs = base_ptr + row * stride_m + col * stride_n   # shape (BLOCK_M, BLOCK_N)
#
# `tl.arange` is the Triton in-kernel analog of `np.arange`; the book itself
# states the `[:, None]` / `[None, :]` broadcasting is "identical to NumPy",
# so np.arange is a faithful, honest, CPU-only stand-in for tl.arange here.
# ---------------------------------------------------------------------------


def tl_arange(start, end):
    """CPU stand-in for triton.language.arange (int32 vector), used only so
    block #1's code can run verbatim below without a GPU/triton runtime."""
    return np.arange(start, end, dtype=np.int64)


def build_pointer_block(base_ptr, BLOCK_M, BLOCK_N, stride_m, stride_n):
    # --- verbatim block #1 body, with tl.arange -> tl_arange substitution ---
    row = tl_arange(0, BLOCK_M)[:, None]   # shape (BLOCK_M, 1)
    col = tl_arange(0, BLOCK_N)[None, :]   # shape (1, BLOCK_N)
    ptrs = base_ptr + row * stride_m + col * stride_n   # shape (BLOCK_M, BLOCK_N)
    return ptrs


def test_block1_pointer_arithmetic():
    BLOCK_M, BLOCK_N = 4, 6
    # Strides for a row-major matrix with, say, 10 columns (stride_m = n_cols,
    # stride_n = 1), plus a nonzero base_ptr to exercise the "+ base_ptr" term.
    base_ptr = 1000
    stride_m, stride_n = 10, 1

    ptrs = build_pointer_block(base_ptr, BLOCK_M, BLOCK_N, stride_m, stride_n)

    assert ptrs.shape == (BLOCK_M, BLOCK_N)

    # Closed-form check: ptrs[i, j] == base_ptr + i*stride_m + j*stride_n
    expected = base_ptr + np.arange(BLOCK_M)[:, None] * stride_m + np.arange(BLOCK_N)[None, :] * stride_n
    assert np.array_equal(ptrs, expected)

    # Spot-check a few concrete entries by hand.
    assert ptrs[0, 0] == base_ptr
    assert ptrs[0, 1] == base_ptr + 1
    assert ptrs[1, 0] == base_ptr + stride_m
    assert ptrs[3, 5] == base_ptr + 3 * stride_m + 5 * stride_n

    # Also exercise non-unit, non-contiguous strides (e.g. a column-major or
    # transposed view), which is exactly the trick Kernel-4's K-tile load
    # uses to get a transpose "for free" (discussed in the chapter text).
    base_ptr2 = 0
    stride_m2, stride_n2 = 1, 7   # rows step by 1, cols step by 7 (transposed-looking)
    ptrs2 = build_pointer_block(base_ptr2, 3, 2, stride_m2, stride_n2)
    expected2 = np.array([
        [0, 7],
        [1, 8],
        [2, 9],
    ])
    assert np.array_equal(ptrs2, expected2)

    print("block #1 (2-D pointer-block arithmetic): PASS")
    print(ptrs)


if __name__ == "__main__":
    test_block1_pointer_arithmetic()

    print()
    print("SKIP(fragment): block #0 is an import/prose fragment, not standalone.")
    print("SKIP(needs-gpu): blocks #2-#13 are @triton.jit kernels / torch(device='cuda')")
    print("  drivers -- Kernel 1 (vector add), Kernel 2 (fused softmax),")
    print("  Kernel 3 (matmul + autotune), Kernel 4 (flash attention forward,")
    print("  including its autotuned skeleton), and Kernel 5 (flash attention")
    print("  backward: preprocess + dK/dV/dQ kernels). All require an actual")
    print("  GPU with a CUDA-capable Triton runtime to compile and launch; they")
    print("  cannot be executed or meaningfully mocked on CPU-only CI without")
    print("  bypassing the very kernel logic being demonstrated.")
    print()
    print("All CPU-runnable blocks executed successfully.")
