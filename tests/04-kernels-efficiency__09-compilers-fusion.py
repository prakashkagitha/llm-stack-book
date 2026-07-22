"""
Runs the CPU-runnable Python blocks from
content/04-kernels-efficiency/09-compilers-fusion.md, concatenated in order.
Each tested block's code is copied verbatim from the (now-fixed) chapter;
minimal glue needed to actually *execute* each block (small fixtures, tiny
shapes) is clearly marked "GLUE".

Blocks tested (all run on CPU; torch.compile falls back to its CPU/C++
Inductor backend, which is fine for correctness -- only benchmarking numbers
in the book are GPU-specific, not the compilation logic itself):

  #1  (line ~122) - TorchDynamo graph-break demo via dynamo.explain()
  #2  (line ~146) - AOTAutograd aot_function() joint fwd/bwd trace
  #6  (line ~222) - torch._dynamo.config.verbose + explain() on a real module
  #10 (line ~487) - forward_bad / forward_good: data-dependent control flow
  #11 (line ~503) - forward_bad / forward_good: Python list of tensors
  #12 (line ~518) - dynamic=True + torch._dynamo.mark_dynamic

SKIP (per task instructions, all need a real CUDA GPU or are non-Python /
shell fragments not in the tested set):
  #0  - build_cuda_graph/replay_cuda_graph -- SKIP(needs-gpu): torch.cuda.CUDAGraph
        requires an actual CUDA device; not runnable on CPU.
  #3  - DecoderLayer + torch.compile(..., mode="reduce-overhead"/"max-autotune")
        on device='cuda' -- SKIP(needs-gpu).
  #4  - torch_compile_demo.py full eager-vs-compiled benchmark script --
        SKIP(needs-gpu): hardcodes device="cuda", CUDA graphs, bf16 CUDA timing.
  #5  - `TORCH_LOGS=graph_breaks python train.py` -- SKIP(shell): not Python.
  #7  - training loop with AdamW(..., fused=True) on device="cuda" --
        SKIP(needs-gpu): fused AdamW requires CUDA; script also assumes a
        cuda-resident DecoderLayer.
  #8  - the fenced ```text sample benchmark output block -- SKIP(non-python).
  #9  - not separately assigned by the task; the DecoderLayer/FeedForward
        classes used in the GPU benchmark script are only defined there and
        not needed by any CPU-tested block.

Two real bugs were found in the book's code while getting these blocks to
actually run, and were fixed in the .md (and are mirrored below):

  1. Block #1 and block #6 both did
         for reason in explanation.graph_break_reasons:
     but `torch._dynamo.explain(...)`'s return object (ExplainOutput) has no
     `graph_break_reasons` attribute in current PyTorch (2.x) -- the real
     attribute is `break_reasons`. Using the book's original name raises
     `AttributeError: 'ExplainOutput' object has no attribute
     'graph_break_reasons'. Did you mean: 'graph_break_count'?`. Fixed to
     `explanation.break_reasons` in both places.

  2. Block #12 called `torch._dynamo.mark_dynamic(x, 0)` *inside* the body of
     a `@torch.compile`-decorated function. `mark_dynamic` is a tracing-time
     annotation that must be called on the tensor before it enters a
     compiled region; calling it from inside traced code raises
     `AssertionError: Attempt to trace forbidden callable
     <function mark_dynamic ...>`. Fixed by moving the `mark_dynamic` call
     above the `@torch.compile` function definition, so it runs eagerly on
     `x` before `forward(x)` is ever traced.
"""

import torch


# =====================================================================
# Block #1 (line ~122): TorchDynamo graph-break demo via dynamo.explain()
# =====================================================================

import torch._dynamo as dynamo

# Demonstrate graph breaks with explain()
def my_func(x):
    y = torch.sin(x)          # traced
    if x.shape[0] > 10:       # graph break: dynamic control flow
        y = y * 2
    return torch.cos(y)       # traced (in new subgraph)

explanation = dynamo.explain(my_func)(torch.randn(5))
print(explanation.graphs)          # Two subgraphs
print(explanation.break_reasons)   # bugfix: was graph_break_reasons (AttributeError)

assert len(explanation.graphs) >= 1
assert isinstance(explanation.break_reasons, list)
print("[block #1] dynamo.explain() ran OK:", len(explanation.graphs), "graph(s),",
      len(explanation.break_reasons), "break reason(s)")


# =====================================================================
# Block #2 (line ~146): AOTAutograd aot_function() joint fwd/bwd trace
# =====================================================================

from torch._functorch.aot_autograd import aot_function

def fn(x, w):
    """A simple layer: linear + sigmoid."""
    return torch.sigmoid(x @ w)

# AOTAutograd decomposes this into a joint graph.
# During compilation, it generates:
#   forward:  z = x @ w; y = sigmoid(z)
#   backward: dy/dz = y * (1 - y); ...
# The sigmoid and its gradient can be fused into a single kernel.
compiled_fn = aot_function(fn, fw_compiler=lambda g, _: g, bw_compiler=lambda g, _: g)

# GLUE: actually call the compiled function forward + backward on tiny tensors
# so the block's logic (joint fwd/bwd tracing) genuinely executes.
_x = torch.randn(4, 4, requires_grad=True)
_w = torch.randn(4, 4, requires_grad=True)
_out = compiled_fn(_x, _w)
_out.sum().backward()

assert _out.shape == (4, 4)
assert _x.grad is not None and _x.grad.shape == (4, 4)
assert _w.grad is not None and _w.grad.shape == (4, 4)
print("[block #2] aot_function joint fwd/bwd ran OK, out shape:", tuple(_out.shape))


# =====================================================================
# Block #6 (line ~222): torch._dynamo.config.verbose + explain() on a module
# =====================================================================

import torch._dynamo
# Or programmatically:
torch._dynamo.config.verbose = True

# Alternatively, use the explain() API:
# GLUE: the book leaves `model` / `example_input` implicit (established
# earlier in the chapter's narrative); supply a tiny standalone module here.
import torch.nn as nn
model = nn.Linear(8, 8)
example_input = torch.randn(2, 8)

explanation = torch._dynamo.explain(model)(example_input)
for reason in explanation.break_reasons:  # bugfix: was graph_break_reasons
    print(reason)

print("[block #6] torch._dynamo.explain(model) ran OK,",
      len(explanation.break_reasons), "break reason(s)")


# =====================================================================
# Block #10 (line ~487): data-dependent control flow, bad vs. good
# =====================================================================

# BAD: Dynamo cannot trace through this — graph break every call
def forward_bad(x):
    if x.max() > 1.0:    # x.max() requires synchronizing CPU/GPU
        x = x / x.max()  # dynamic branch
    return x

# GOOD: use tensor operations throughout
def forward_good(x):
    max_val = x.max()
    # Soft clamp: x / max(x.max(), 1.0) — always the same graph
    return x / torch.clamp(max_val, min=1.0)

# GLUE: call both variants on a tiny tensor and check they agree.
_probe = torch.tensor([0.5, 2.0, 1.5])
_bad_out = forward_bad(_probe.clone())
_good_out = forward_good(_probe.clone())
assert torch.allclose(_bad_out, _good_out)
print("[block #10] forward_bad/forward_good agree:", _good_out.tolist())


# =====================================================================
# Block #11 (line ~503): Python lists of tensors, bad vs. good
# =====================================================================

# BAD: list indexing can be dynamic; Dynamo may break here
def forward_bad(tensors: list):
    return [t * 2 for t in tensors]  # Python list comprehension → break

# GOOD: stack into a single tensor
def forward_good(tensors: list):
    stacked = torch.stack(tensors)   # Single op, fully traced
    return stacked * 2

# GLUE: call both variants on a tiny list of tensors and check they agree.
_tensors = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
_bad_list = forward_bad(_tensors)
_good_stacked = forward_good(_tensors)
assert torch.allclose(torch.stack(_bad_list), _good_stacked)
print("[block #11] forward_bad/forward_good agree:", _good_stacked.tolist())


# =====================================================================
# Block #12 (line ~518): dynamic=True + torch._dynamo.mark_dynamic
# =====================================================================

import torch._dynamo

# GLUE: the book's snippet references a preexisting `model`; reuse the tiny
# nn.Linear from block #6, freshly re-wrapped with dynamic=True.
model = torch.compile(nn.Linear(8, 8), dynamic=True)

# Mark batch dimension as dynamic before compilation
# (already imported torch._dynamo above)

# More surgical control with torch._dynamo.mark_dynamo:
# bugfix: mark_dynamic must be called BEFORE the tensor enters a compiled
# region (calling it from inside a @torch.compile'd function raises
# "Attempt to trace forbidden callable").
_x12 = torch.randn(4, 8)
torch._dynamo.mark_dynamic(_x12, 0)   # dim 0 is dynamic

@torch.compile
def forward(x):
    return model(x)

_y12 = forward(_x12)
assert _y12.shape == (4, 8)

# GLUE: call again with a different batch size to exercise the "dynamic
# dimension" behavior the block is demonstrating (no recompile should be
# required for the marked dimension).
_x12b = torch.randn(6, 8)
torch._dynamo.mark_dynamic(_x12b, 0)
_y12b = forward(_x12b)
assert _y12b.shape == (6, 8)

print("[block #12] dynamic=True + mark_dynamic ran OK for batch sizes 4 and 6")


print("\nAll CPU-runnable blocks in 04-kernels-efficiency/09-compilers-fusion.md executed successfully.")
