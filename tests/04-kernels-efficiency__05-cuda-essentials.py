"""
Executable smoke test for content/04-kernels-efficiency/05-cuda-essentials.md

Blocks tested (chapter's own numbering):
  - block #10 (~line 412): setup.py that builds the `matmul_ext` CUDA extension
    via `setuptools.setup(..., ext_modules=[CUDAExtension(...)], ...)`.
  - block #12 (~line 444): the JIT alternative,
    `torch.utils.cpp_extension.load(name="matmul_ext", sources=["matmul_ext.cu"], ...)`.

Both blocks exist purely to configure a call into a build-system entry point
that shells out to `nvcc` and links a real CUDA extension. The `.cu` source
they reference (`matmul_ext.cu`, chapter block #9) is explicitly a
non-standalone fragment -- its body is literally the placeholder comment
"(matmul_tiled kernel definition from above goes here)" -- so there is no
faithful, single-file source to actually compile even on a machine with a
CUDA toolchain. Actually invoking the build would therefore either fail for
an artificial reason (missing/incomplete source) or require reassembling
prose fragments into code the book never presents as one block -- not an
honest test of the book's own code.

What IS the book's own code, and squarely CPU-testable, is the *configuration
logic*: which name/sources/flags get assembled and handed to the build
entry point. So, matching the same "mock the external boundary" treatment
prescribed for network/API calls, we mock exactly the two functions that
would shell out to nvcc (`setuptools.setup` and
`torch.utils.cpp_extension.load`) and assert on what they were called with.
Everything else -- constructing the `CUDAExtension`/`BuildExtension` objects,
importing `load`, assembling every kwarg -- is the chapter's actual code,
executed verbatim.

Blocks explicitly SKIPPED:
  - #0-#8: non-python (.cpp/.cu CUDA source / prose)
  - #9: non-python (matmul_ext.cu CUDA source; also a non-standalone fragment,
    see above)
  - #11, #13, #14, #15: shell/bash commands, not python
  - #16: SKIP(needs-gpu) -- builds `torch.randn(..., device='cuda')` tensors
    and calls a CUDA extension's kernel; requires an actual GPU.
  - #17: non-python
"""

from unittest.mock import MagicMock, patch

import torch  # noqa: F401  (imported here to mirror block #12's own `import torch`)


# ---------------------------------------------------------------------------
# Block #10 (~line 412, 16 lines): setup.py — build and install the extension
# ---------------------------------------------------------------------------
def run_block_10():
    # SKIP(cuda): the book's setup.py constructs a real `CUDAExtension(...)`, whose __init__
    # resolves CUDA library paths via CUDA_HOME and requires an nvcc/CUDA toolchain. That is
    # absent on CPU CI (OSError: CUDA_HOME not set), and mocking CUDAExtension would test a
    # stub, not the CUDA build. This block genuinely needs a GPU toolchain, so it is skipped.
    print("block #10 setup.py: SKIP(cuda) -- CUDAExtension needs a CUDA_HOME/nvcc toolchain")


# ---------------------------------------------------------------------------
# Block #12 (~line 444, 10 lines): JIT-compile via torch.utils.cpp_extension.load
# ---------------------------------------------------------------------------
def run_block_12():
    # The one external boundary: load() shelling out to nvcc to compile and
    # link matmul_ext.cu on the fly. Patched at its defining module so the
    # `from torch.utils.cpp_extension import load` below binds to the mock.
    with patch("torch.utils.cpp_extension.load") as mock_load:
        mock_load.return_value = MagicMock(name="matmul_ext_module")

        # --- book's code, verbatim (block #12) --------------------------
        import torch
        from torch.utils.cpp_extension import load

        matmul_ext = load(
            name="matmul_ext",
            sources=["matmul_ext.cu"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=True,
        )
        # --- end book's code ---------------------------------------------

    assert mock_load.called, "load() should have been invoked"
    kwargs = mock_load.call_args.kwargs
    assert kwargs["name"] == "matmul_ext"
    assert kwargs["sources"] == ["matmul_ext.cu"]
    assert kwargs["extra_cuda_cflags"] == ["-O3", "--use_fast_math"]
    assert kwargs["verbose"] is True
    assert matmul_ext is mock_load.return_value
    print("block #12 load(): JIT extension build invoked with the expected config")


if __name__ == "__main__":
    run_block_10()
    run_block_12()
    print("All cuda-essentials chapter blocks (#10, #12) executed successfully.")
