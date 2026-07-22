"""
Executable extraction of the runnable Python blocks in
content/01-foundations/07-autodiff-pytorch.md

Blocks are concatenated in document order (later blocks may reuse names
defined by earlier blocks, exactly as they would if a reader typed the
chapter into one REPL session). Each block is preceded by a comment noting
its block index and source line (from the original chapter file).

SKIPPED:
  - block #11 (line ~392, ```text schema string) -- non-python.

Everywhere a block references a name the chapter never actually defines
(e.g. a pseudocode `model`, `optimizer`, `input`) we add the smallest
possible fixture immediately before the block and mark it "# GLUE:".
The book's own logic/lines are otherwise copied verbatim.
"""

import torch
import torch.nn as nn

print("=== block #0 (line ~68): leaf tensors, requires_grad, grad_fn ===")

# Leaf tensor: created directly, not the result of an op
x = torch.tensor([2.0, 3.0], requires_grad=True)
w = torch.tensor([0.5, -1.0], requires_grad=True)

# Non-leaf (intermediate) tensor: result of an operation
y = (x * w).sum()   # y = 2*0.5 + 3*(-1.0) = 1.0 - 3.0 = -2.0

print(x.is_leaf)   # True
print(y.is_leaf)   # False
print(y.grad_fn)   # <SumBackward0 object at 0x...>

y.backward()

# Only leaf gradients are populated
print(x.grad)   # tensor([ 0.5, -1.0])  -- dy/dx_i = w_i
print(w.grad)   # tensor([ 2.0,  3.0])  -- dy/dw_i = x_i
# y.grad would be None (non-leaf, gradient not retained)

assert x.is_leaf and not y.is_leaf
assert torch.allclose(x.grad, torch.tensor([0.5, -1.0]))
assert torch.allclose(w.grad, torch.tensor([2.0, 3.0]))
assert y.grad is None

print("\n=== block #1 (line ~94): grad_fn graph, next_functions ===")

# Inspecting the graph manually
a = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(4.0, requires_grad=True)
c = a * b           # MulBackward0
d = c + a           # AddBackward0
e = d ** 2          # PowBackward0

print(e.grad_fn)                          # PowBackward0
print(e.grad_fn.next_functions)           # ((AddBackward0, 0),)
print(e.grad_fn.next_functions[0][0].next_functions)
# ((MulBackward0, 0), (AccumulateGrad, 0))  <- 'a' appears twice!

assert e.grad_fn.__class__.__name__ == "PowBackward0"
inner = e.grad_fn.next_functions[0][0].next_functions
names = sorted(fn.__class__.__name__ for fn, _ in inner)
assert names == ["AccumulateGrad", "MulBackward0"]

print("\n=== block #2 (line ~116): gradient accumulation pattern ===")

# GLUE: the book's snippet uses `model`, `optimizer`, and
# `accumulation_steps` as pseudocode (accumulation_steps is used both as
# an int divisor and as the thing iterated over). We supply a tiny linear
# model/optimizer and split "how many micro-batches" (an int) from the
# list of toy micro-batches actually iterated over, keeping the book's
# zero_grad / backward / step structure verbatim.
model = nn.Linear(3, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
accumulation_steps = 4
mini_batches = [torch.randn(2, 3) for _ in range(accumulation_steps)]

optimizer.zero_grad()          # clear accumulated grads
for mini_batch in mini_batches:
    loss = model(mini_batch).sum() / accumulation_steps
    loss.backward()            # accumulates into .grad
optimizer.step()               # update once with the full-batch gradient

assert model.weight.grad is not None
print("weight.grad after accumulation:", model.weight.grad)

print("\n=== block #3 (line ~134): torch.no_grad ===")

# GLUE: tiny stand-ins for `model` / `input`.
model = nn.Linear(3, 2)
input = torch.randn(4, 3)

with torch.no_grad():
    output = model(input)   # no graph built; saves memory and compute

assert output.grad_fn is None
assert not output.requires_grad

print("\n=== block #4 (line ~143): torch.inference_mode ===")

# GLUE: tiny stand-ins for `model` / `input_ids`.
model = nn.Linear(5, 3)
input_ids = torch.randn(2, 5)

with torch.inference_mode():
    logits = model(input_ids)
    probs = torch.softmax(logits, dim=-1)

assert probs.shape == (2, 3)
assert torch.allclose(probs.sum(dim=-1), torch.ones(2))

print("\n=== block #5 (line ~157): stop-gradient / detach ===")

# GLUE: tiny stand-ins for `target_encoder` / `online_encoder` / `x`.
target_encoder = nn.Linear(4, 4)
online_encoder = nn.Linear(4, 4)
x = torch.randn(2, 4)

# Stop-gradient for target network
with torch.no_grad():
    target_features = target_encoder(x)  # equivalent to detach here

# Or explicitly:
target_features = online_encoder(x).detach()

assert not target_features.requires_grad
assert target_features.grad_fn is None

print("\n=== block #6 (line ~174): Function interface (documentation skeleton) ===")

from torch.autograd import Function


class MyOp(Function):
    @staticmethod
    def forward(ctx, *inputs):
        # ctx is a context object for stashing tensors for backward
        # return output tensor(s)
        ...

    @staticmethod
    def backward(ctx, *grad_outputs):
        # grad_outputs: upstream gradients (one per forward output)
        # return gradient tensors (one per forward input, or None if not differentiable)
        ...


# This block is a pure interface skeleton in the book (bodies are `...`);
# there is nothing meaningful to call here -- the concrete, runnable
# implementation is StableSigmoid in block #7 below. Defining the class
# (which happens above, at module-exec time) is the block's execution.
assert issubclass(MyOp, Function)

print("\n=== block #7 (line ~198): StableSigmoid custom autograd.Function ===")

import torch
from torch.autograd import Function


class StableSigmoid(Function):
    """
    Sigmoid with a numerically stable forward and analytic backward.
    We store only the output (not input) for memory efficiency,
    since d_sigma/dx = sigma(x) * (1 - sigma(x)).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        # Use PyTorch's stable sigmoid internally
        y = torch.sigmoid(x)
        # Save output, not input -- saves memory for large activations
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (y,) = ctx.saved_tensors
        # Jacobian of sigmoid: dy/dx = y * (1 - y)
        # VJP: grad_input = grad_output * dy/dx  (element-wise for elt-wise ops)
        grad_input = grad_output * y * (1.0 - y)
        return grad_input


# Register as a callable
stable_sigmoid = StableSigmoid.apply


# --- Test correctness against autograd ---
torch.manual_seed(0)
x = torch.randn(4, requires_grad=True)
x_ref = x.detach().clone().requires_grad_(True)

y = stable_sigmoid(x)
y_ref = torch.sigmoid(x_ref)

# Forward agreement
assert torch.allclose(y, y_ref, atol=1e-6), "Forward mismatch"

# Backward agreement via gradient checking
y.sum().backward()
y_ref.sum().backward()
assert torch.allclose(x.grad, x_ref.grad, atol=1e-6), "Backward mismatch"

print("x      :", x.detach().numpy().round(4))
print("sigma  :", y.detach().numpy().round(4))
print("grad   :", x.grad.numpy().round(4))
# x      : [ 1.5410 -0.2934 -2.1788  0.5684]
# sigma  : [0.8238 0.4271 0.1017 0.6387]
# grad   : [0.1449 0.2446 0.0912 0.2307]  -- sigma*(1-sigma)

print("\n=== block #8 (line ~259): StraightThroughRound (STE) ===")


class StraightThroughRound(Function):
    """
    Forward:  y = round(x)   (non-differentiable: gradient is 0 a.e.)
    Backward: dy/dx = 1      (identity straight-through estimator)
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x.round()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # Pass gradient through unchanged -- the "straight-through" trick
        return grad_output


ste_round = StraightThroughRound.apply

x = torch.tensor([0.3, 1.7, -0.6], requires_grad=True)
y = ste_round(x)
print(y)           # tensor([ 0.,  2., -1.])
y.sum().backward()
print(x.grad)      # tensor([1., 1., 1.])  -- as if dy/dx = 1 everywhere

assert torch.allclose(y, torch.tensor([0.0, 2.0, -1.0]))
assert torch.allclose(x.grad, torch.ones(3))

print("\n=== block #9 (line ~289): gradcheck against stable_sigmoid ===")

from torch.autograd import gradcheck

# Use double precision for numerical stability of finite differences
x_check = torch.randn(3, dtype=torch.float64, requires_grad=True)
result = gradcheck(stable_sigmoid, (x_check,), eps=1e-6, atol=1e-4)
print(f"gradcheck passed: {result}")  # True
assert result is True

print("\n=== block #10 (line ~348): worked forward+backward numerical trace ===")

import torch

x = torch.tensor([[1.0], [0.0]])               # (2,1)
W1 = torch.tensor([[2., 1.], [-1., 3.]], requires_grad=True)
w2 = torch.tensor([[0.5], [0.5]], requires_grad=True)

z1 = W1 @ x
h = torch.relu(z1)
z2 = (w2.T @ h).squeeze()
L = 0.5 * z2 ** 2

L.backward()

print("W1.grad:\n", W1.grad)
# [[0.5, 0.],
#  [0. , 0.]]
print("w2.grad:\n", w2.grad)
# [[2.],
#  [0.]]

assert torch.allclose(W1.grad, torch.tensor([[0.5, 0.0], [0.0, 0.0]]))
assert torch.allclose(w2.grad, torch.tensor([[2.0], [0.0]]))

print("\n=== block #12 (line ~408): views vs copies ===")

import torch

x = torch.arange(12).reshape(3, 4)
y = x.T                  # Transpose: a view, not a copy
z = x[0:2, :]            # Slice: a view

print(x.data_ptr() == y.data_ptr())   # True -- same storage
print(x.data_ptr() == z.data_ptr())   # True -- same storage

# Modifying y modifies x
y[0, 0] = 999
print(x[0, 0])  # 999

assert x.data_ptr() == y.data_ptr()
assert x.data_ptr() == z.data_ptr()
assert x[0, 0].item() == 999

print("\n=== block #13 (line ~429): contiguity ===")

x = torch.arange(6).reshape(2, 3)
print(x.is_contiguous())   # True
print(x.stride())          # (3, 1)  -- step 3 elements between rows, 1 between cols

y = x.T                    # Transpose
print(y.is_contiguous())   # False
print(y.stride())          # (1, 3)  -- stride order reversed

# Force contiguous copy (creates new storage)
z = y.contiguous()
print(z.is_contiguous())   # True

assert x.is_contiguous() and x.stride() == (3, 1)
assert not y.is_contiguous() and y.stride() == (1, 3)
assert z.is_contiguous()

print("\n=== block #14 (line ~456): strides / zero-stride broadcasting ===")

# Broadcasting via zero stride
x = torch.tensor([1.0, 2.0, 3.0])
# Expand to (4, 3) without copying:
y = x.unsqueeze(0).expand(4, 3)
print(y.stride())        # (0, 1) -- stride-0 in the batch dimension
print(y.is_contiguous()) # False
print(y.storage().size() == 3)  # True -- only 3 elements in storage

assert y.stride() == (0, 1)
assert not y.is_contiguous()
assert y.storage().size() == 3

print("\n=== block #15 (line ~476): broadcasting semantics in backward ===")

a = torch.ones(3, 1, requires_grad=True)   # shape (3, 1)
b = torch.ones(3, 4)                       # shape (3, 4), no grad

c = a + b   # broadcasts a to (3, 4)
c.sum().backward()

# grad of 'a' sums over the broadcast dim:
print(a.grad)   # tensor([[4.], [4.], [4.]])  -- sum over 4 columns

assert torch.allclose(a.grad, torch.full((3, 1), 4.0))

print("\n=== block #16 (line ~503): gradient checkpointing ===")

from torch.utils.checkpoint import checkpoint


def block_forward(x, layer):
    return layer(x)


# GLUE: tiny stand-ins for `layer` / `x_in`, plus a backward call so the
# checkpoint's recompute-on-backward path actually executes.
layer = nn.Linear(4, 4)
x_in = torch.randn(2, 4, requires_grad=True)

# During backward, the forward of this segment will be re-run
x_out = checkpoint(block_forward, x_in, layer, use_reentrant=False)
x_out.sum().backward()

assert x_in.grad is not None
assert x_out.shape == (2, 4)

print("\n=== block #17 (line ~519): second-order gradients / create_graph ===")

x = torch.tensor(3.0, requires_grad=True)
y = x ** 3         # y = x^3, dy/dx = 3x^2

# First-order gradient
(grad_x,) = torch.autograd.grad(y, x, create_graph=True)
print(grad_x)   # tensor(27.)  -- 3 * 3^2

# Second-order gradient (differentiates grad_x w.r.t. x)
(grad2_x,) = torch.autograd.grad(grad_x, x)
print(grad2_x)  # tensor(18.)  -- 6x = 6*3

assert torch.allclose(grad_x, torch.tensor(27.0))
assert torch.allclose(grad2_x, torch.tensor(18.0))

print("\n=== block #18 (line ~538): torch.func (grad, vmap, jacrev) ===")

from torch.func import grad, vmap, jacrev

# Compute gradient of a scalar function
f = lambda x: (x ** 2).sum()
grad_f = grad(f)
print(grad_f(torch.tensor([1.0, 2.0, 3.0])))  # tensor([2., 4., 6.])

assert torch.allclose(grad_f(torch.tensor([1.0, 2.0, 3.0])), torch.tensor([2.0, 4.0, 6.0]))


# Batched Jacobian via vmap + jacrev
def model(params, x):
    return params @ x


# GLUE: tiny stand-ins for `params_batch` / `x_batch` (batch of 5,
# out_dim=3, in_dim=4) matching model(params, x) = params @ x.
params_batch = torch.randn(5, 3, 4)
x_batch = torch.randn(5, 4)

J = vmap(jacrev(model, argnums=1))(params_batch, x_batch)
print("Jacobian batch shape:", J.shape)  # (5, 3, 4)

assert J.shape == (5, 3, 4)
# For a linear map, the Jacobian w.r.t. x is just the weight matrix itself.
assert torch.allclose(J, params_batch, atol=1e-5)

print("\n=== block #19 (line ~594): set_detect_anomaly / NaN localization ===")

import torch

# A forward that is finite at x = 0 but whose *gradient* is not:
#   d/dx sqrt(x) = 0.5 / sqrt(x)  ->  inf at x = 0; the product rule below then
#   multiplies that inf by x = 0, so the backward pass hits 0 * inf = nan.
# (Plain x.sqrt() alone gives an inf grad, which anomaly mode does NOT flag --
#  it only checks for NaN -- so the product is what triggers the RuntimeError.)
x = torch.zeros(3, requires_grad=True)

# Without anomaly detection you only see a NaN grad, not WHERE it came from:
y = (x * x.sqrt()).sum()   # forward: tensor(0.) -- looks fine
y.backward()
print(x.grad)              # tensor([nan, nan, nan]) -- silent, no traceback

assert torch.isnan(x.grad).all()

# With anomaly detection, backward raises AT the offending forward op.
# This is a book-deliberate "this crashes" demo -- we assert-it-raises
# rather than "fixing" it.
x = torch.zeros(3, requires_grad=True)
raised = False
try:
    with torch.autograd.detect_anomaly():
        y = (x * x.sqrt()).sum()  # anomaly mode stores this line's stack trace
        y.backward()
except RuntimeError as err:
    raised = True
    print("Got expected RuntimeError under detect_anomaly():", str(err).splitlines()[0])
assert raised, "detect_anomaly() was expected to raise a RuntimeError on the NaN backward"

print("\nAll blocks executed successfully.")
