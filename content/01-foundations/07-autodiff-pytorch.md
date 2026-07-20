# 1.7 Automatic Differentiation & PyTorch Internals

Backpropagation is the algorithm that makes deep learning tractable, but writing it by hand for every new architecture is error-prone, slow, and frankly miserable. Automatic differentiation (autodiff) is the engineering discipline that solves this: given any composition of differentiable operations expressed as code, autodiff computes exact gradients — not finite differences, not symbolic algebra output — mechanically and efficiently. PyTorch's `autograd` engine is the most widely used reverse-mode autodiff system in research and production. Understanding it at the mechanism level pays dividends every time you write a custom loss, a non-standard layer, or debug a gradient that quietly went to zero.

This chapter covers the full stack: the mathematics of reverse-mode autodiff and the tape metaphor, the PyTorch computation graph and its memory model, leaf tensors and the `.grad` accumulation protocol, `no_grad` and inference mode, custom `autograd.Function`, and the lower layers of PyTorch (the dispatcher, ATen, views vs. copies, contiguity, and broadcasting semantics). We connect the theory to the practice with a fully-worked custom function example and numerical traces you can follow by hand.

We assume you have read [Neural Networks From Scratch: MLPs & Backprop](../01-foundations/06-neural-nets-from-scratch.html) and are comfortable with the chain rule. We also reference [Calculus, Optimization & Convexity](../01-foundations/03-calculus-optimization.html) for the Jacobian formalism and [GPU Architecture & The Memory Hierarchy](../01-foundations/08-gpu-architecture.html) for the hardware context that makes contiguity matter.

---

## Why Autodiff Exists: The Three Alternatives and Their Failures

Before diving into the mechanism, it is worth naming the alternatives that autodiff replaced.

**Manual backprop** requires the programmer to derive and implement the gradient of every operation. This was the norm in the symbolic-layer era (Theano, early Caffe). It is correct when done carefully, but it couples forward and backward code, makes architectural experiments tedious, and is a perennial source of subtle bugs.

**Numerical differentiation** (finite differences) approximates the derivative as:

$$
\frac{\partial f}{\partial x_i} \approx \frac{f(x + \epsilon e_i) - f(x - \epsilon e_i)}{2\epsilon}
$$

This requires $2N$ forward passes for $N$ parameters — completely impractical for networks with billions of parameters. It is still useful for **gradient checking** (verifying autodiff implementations), where you compare the autodiff gradient against a small finite-difference estimate for a handful of parameters.

**Symbolic differentiation** (as in computer algebra systems like Mathematica) manipulates expression trees algebraically. The output is a symbolic formula for the derivative. It produces exact answers but suffers from **expression swell** — derivatives of composite functions grow exponentially in size with depth, and the resulting code is typically far slower than an equivalent imperative implementation.

Reverse-mode autodiff threads the needle: it computes exact derivatives (to floating-point precision), scales in $O(1)$ forward passes regardless of $N$, and operates on ordinary imperative code. Its only real cost is memory for the intermediate activations stored during the forward pass — a cost we will see how to reduce with gradient checkpointing (covered in [Memory-Efficient Training: Checkpointing, Offloading & LoRA Math](../04-kernels-efficiency/10-memory-efficient-training.html)).

---

## Reverse-Mode Autodiff: The Tape

### The Jacobian-Vector Product View

Let $f : \mathbb{R}^n \to \mathbb{R}^m$ be a differentiable function. Its derivative at a point $x$ is a linear map $Df(x) : \mathbb{R}^n \to \mathbb{R}^m$, represented as the $m \times n$ Jacobian matrix $J$. For most neural network losses, $m = 1$, so the Jacobian is a $1 \times n$ row vector — the gradient.

Reverse-mode autodiff computes the **vector-Jacobian product (VJP)**:

$$
\bar{x} = \bar{y}^\top J
$$

where $\bar{y}$ is the upstream gradient (a $1 \times m$ row vector) and $\bar{x}$ is the resulting gradient with respect to $x$. When $m = 1$ and $\bar{y} = 1$, this recovers the ordinary gradient $\nabla_x f$.

The key insight is that we never materialize $J$ itself — we only ever compute VJPs. This makes reverse-mode autodiff efficient when the output dimension $m$ is small (e.g., $m = 1$ for scalar losses) regardless of how large $n$ is.

### The Tape Metaphor

During the **forward pass**, autograd records every operation applied to tensors that require gradients, building a directed acyclic graph (DAG). Edges point from outputs to inputs (in the *backward* direction). Nodes are `Function` objects that know how to compute the VJP for their operation. This data structure is called the **tape** or **Wengert list** (after Robert Wengert, who described it in 1964).

During the **backward pass**, we traverse the tape in reverse topological order. At each node, we:
1. Receive the upstream gradient $\bar{y}$ from the node above.
2. Call the node's `backward` function to compute the VJP: $\bar{x} = \bar{y}^\top J$.
3. Accumulate $\bar{x}$ into the gradient of the input tensor and pass it downstream.

After `loss.backward()` returns, every leaf tensor with `requires_grad=True` has its `.grad` field populated with the accumulated gradient.

{{fig:autodiff-tape-forward-backward}}

---

## PyTorch Autograd: The Computation Graph

### Tensors, `requires_grad`, and Leaf Nodes

Every PyTorch tensor has a `requires_grad` flag. When `True`, operations on it are tracked. A tensor is a **leaf** if it was created directly by user code (e.g., `nn.Parameter`, or a tensor created with `requires_grad=True`), rather than being the output of some tracked operation. After `backward()`, only leaf tensors' `.grad` fields are populated; intermediate (non-leaf) tensors' gradients are not retained by default (to save memory).

```python
import torch

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
```

The `grad_fn` attribute is a reference to the `Function` node that created this tensor. Following `.grad_fn.next_functions` traverses the graph toward the leaves.

### The `grad_fn` Graph

```python
# Inspecting the graph manually
a = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(4.0, requires_grad=True)
c = a * b           # MulBackward0
d = c + a           # AddBackward0
e = d ** 2          # PowBackward0

print(e.grad_fn)                          # PowBackward0
print(e.grad_fn.next_functions)           # ((AddBackward0, 0),)
print(e.grad_fn.next_functions[0][0].next_functions)
# ((MulBackward0, 0), (AccumulateGrad, 0))  ← 'a' appears twice!
```

Notice that `a` appears twice in the graph (once as an input to `c = a*b` and once as the second input to `d = c+a`). PyTorch correctly accumulates both gradient contributions into `a.grad`.

### Gradient Accumulation and `.grad_fn` Ownership

By default, `.grad` accumulates (adds) across multiple `.backward()` calls. This is intentional and exploited by gradient accumulation in training:

```python
optimizer.zero_grad()          # clear accumulated grads
for mini_batch in accumulation_steps:
    loss = model(mini_batch) / accumulation_steps
    loss.backward()            # accumulates into .grad
optimizer.step()               # update once with the full-batch gradient
```

If you forget `zero_grad()`, gradients from the previous step corrupt the current one — a classic bug.

---

## `torch.no_grad`, `inference_mode`, and Detach

### `torch.no_grad()`

Inside a `no_grad` context, autograd does not record operations even for tensors with `requires_grad=True`. No `grad_fn` is attached to outputs, and no tape is built. Use this for inference and for optimizer steps (parameter updates must not themselves be differentiated).

```python
with torch.no_grad():
    output = model(input)   # no graph built; saves memory and compute
```

### `torch.inference_mode()`

Introduced in PyTorch 1.9, `inference_mode` is a stronger version of `no_grad`: tensors created inside it are marked as "inference tensors" and cannot be used in a future `requires_grad` computation. This allows PyTorch to skip additional bookkeeping. Prefer it over `no_grad` for pure inference paths.

```python
with torch.inference_mode():
    logits = model(input_ids)
    probs = torch.softmax(logits, dim=-1)
```

### `.detach()`

`.detach()` returns a new tensor that shares the same storage but is detached from the computation graph — it has `requires_grad=False` and no `grad_fn`. Common uses:

- Logging or visualization of activations without building a graph.
- Stopping gradient flow in architectures like target networks in RL (where we want the target to be fixed).
- The "stop-gradient" trick in self-supervised learning (BYOL, SimSiam).

```python
# Stop-gradient for target network
with torch.no_grad():
    target_features = target_encoder(x)  # equivalent to detach here

# Or explicitly:
target_features = online_encoder(x).detach()
```

---

## Custom `autograd.Function`

PyTorch's built-in operations cover almost every need, but sometimes you need a custom forward/backward pair: a fused kernel, a numerically stable reformulation, or a straight-through estimator. `torch.autograd.Function` gives you a clean interface to plug into the autograd engine.

### The `Function` Interface

```python
import torch
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
```

The `ctx` object is the bridge: `ctx.save_for_backward(...)` stashes tensors (only tensors, not Python scalars), and `ctx.saved_tensors` retrieves them in `backward`. To stash non-tensor values, assign them as attributes (`ctx.alpha = alpha`).

### Example: Numerically Stable Sigmoid with Custom Backward

The naive sigmoid $\sigma(x) = 1/(1+e^{-x})$ can overflow for large negative $x$ (exp of large positive becomes inf) or lose precision for large positive $x$. The stable version clips large magnitudes and uses `torch.sigmoid` in practice, but here we implement it from scratch with a custom backward to illustrate the interface:

```python
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
```

### Example: Straight-Through Estimator (STE)

The STE is a classic trick used in quantization-aware training (QAT) and binary neural networks. The forward pass applies a non-differentiable rounding operation; the backward pass pretends the function was the identity, passing the upstream gradient through unchanged:

```python
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
```

### `gradcheck`: Numerically Verifying Custom Backwards

PyTorch provides `torch.autograd.gradcheck` to compare your custom `backward` against finite differences:

```python
from torch.autograd import gradcheck

# Use double precision for numerical stability of finite differences
x_check = torch.randn(3, dtype=torch.float64, requires_grad=True)
result = gradcheck(stable_sigmoid, (x_check,), eps=1e-6, atol=1e-4)
print(f"gradcheck passed: {result}")  # True
```

Always run `gradcheck` on new `Function` implementations — it catches sign errors, missing factors, and wrongly accumulated terms.

---

## Worked Numerical Example: Forward + Backward Trace

!!! example "Tracing autograd for a 2-layer network"
    Consider a tiny two-layer network with no bias, scalar output, and ReLU:

    $$
    z_1 = W_1 x, \quad h = \text{ReLU}(z_1), \quad z_2 = w_2^\top h, \quad L = \tfrac{1}{2}z_2^2
    $$

    Let $x = [1, 0]^\top$, $W_1 = \begin{bmatrix}2 & 1\\ -1 & 3\end{bmatrix}$, $w_2 = [0.5, 0.5]^\top$.

    **Forward pass:**

    $$
    z_1 = \begin{bmatrix}2\cdot1+1\cdot0\\ -1\cdot1+3\cdot0\end{bmatrix} = \begin{bmatrix}2\\ -1\end{bmatrix}
    $$

    $$
    h = \text{ReLU}(z_1) = \begin{bmatrix}2\\ 0\end{bmatrix}
    $$

    $$
    z_2 = 0.5 \cdot 2 + 0.5 \cdot 0 = 1.0
    $$

    $$
    L = \tfrac{1}{2}(1.0)^2 = 0.5
    $$

    **Backward pass (VJPs):**

    $\bar{z}_2 = dL/dz_2 = z_2 = 1.0$

    $\bar{w}_2 = \bar{z}_2 \cdot h = 1.0 \cdot [2, 0]^\top = [2, 0]^\top$

    $\bar{h} = \bar{z}_2 \cdot w_2 = 1.0 \cdot [0.5, 0.5]^\top = [0.5, 0.5]^\top$

    ReLU backward: mask where $z_1 > 0$ is $[1, 0]$, so:
    $\bar{z}_1 = \bar{h} \odot \mathbb{1}[z_1 > 0] = [0.5, 0.5]^\top \odot [1, 0]^\top = [0.5, 0]^\top$

    $\bar{W}_1 = \bar{z}_1 x^\top = [0.5, 0]^\top [1, 0] = \begin{bmatrix}0.5 & 0\\ 0 & 0\end{bmatrix}$

    $\bar{x} = W_1^\top \bar{z}_1 = \begin{bmatrix}2 & -1\\ 1 & 3\end{bmatrix} \begin{bmatrix}0.5\\ 0\end{bmatrix} = \begin{bmatrix}1\\ 0.5\end{bmatrix}$

    Let us verify with PyTorch:

    ```python
    import torch

    x  = torch.tensor([[1.0], [0.0]])               # (2,1)
    W1 = torch.tensor([[2., 1.], [-1., 3.]], requires_grad=True)
    w2 = torch.tensor([[0.5], [0.5]], requires_grad=True)

    z1 = W1 @ x
    h  = torch.relu(z1)
    z2 = (w2.T @ h).squeeze()
    L  = 0.5 * z2 ** 2

    L.backward()

    print("W1.grad:\n", W1.grad)
    # [[0.5, 0.],
    #  [0. , 0.]]
    print("w2.grad:\n", w2.grad)
    # [[2.],
    #  [0.]]
    ```

    The numbers match our hand trace exactly. The ReLU gate for the second neuron ($z_1[1] = -1 < 0$) is closed, so no gradient flows through it.

---

## The PyTorch Dispatcher and ATen

Understanding what happens *below* `autograd` helps when you need to write custom C++ extensions, debug shape mismatches, or understand `torch.compile`'s transformations.

### Layers of the PyTorch Stack

{{fig:autodiff-pytorch-stack-layers}}

The **dispatcher** is a routing table. Every operation is registered under one or more "dispatch keys" (tags attached to tensors based on device/dtype/layout). When you call `torch.mm(a, b)`, the dispatcher inspects the keys of `a` and `b` and calls the appropriate backend implementation. This architecture enables:

- **Backend extensibility**: XLA, MPS, custom hardware backends plug in without touching existing code.
- **Transforms**: `torch.compile`, `vmap`, `grad` (functorch) all work by inserting themselves at a dispatch key layer, intercepting operations.
- **Operator overriding**: You can register a custom kernel for a specific (op, backend) pair.

### ATen and the Operator Schema

ATen is PyTorch's C++ tensor library. Every operation in PyTorch ultimately maps to an ATen operator, defined with a schema string like:

```text
mm(Tensor self, Tensor mat2) -> Tensor
```

The schema specifies input/output types and is used by the dispatcher, `torch.fx` tracing, ONNX export, and the JIT compiler. You can browse all ~2000 ATen operators at `torch/_C/_VariableFunctions.pyi` or via `torch._C._VariableFunctions.__dir__()`.

---

## Views, Copies, Contiguity, and Memory Layout

### Views vs Copies

A **view** of a tensor shares the same underlying storage (memory buffer). Operations like `reshape`, `view`, `transpose`, `narrow`, `expand`, and indexing with slices typically return views:

```python
import torch

x = torch.arange(12).reshape(3, 4)
y = x.T                  # Transpose: a view, not a copy
z = x[0:2, :]            # Slice: a view

print(x.data_ptr() == y.data_ptr())   # True -- same storage
print(x.data_ptr() == z.data_ptr())   # True -- same storage

# Modifying y modifies x
y[0, 0] = 999
print(x[0, 0])  # 999
```

This is critical for autograd: gradients flow through views correctly because the autograd graph records view relationships. But it also means in-place operations on views can corrupt the computation graph — PyTorch will raise a `RuntimeError` if you do this during a backward pass.

### Memory Layout and Contiguity

A tensor is **contiguous** if its elements are laid out in row-major (C-style) order: the last dimension varies fastest. Formally, for a tensor with shape $(d_0, d_1, \ldots, d_{n-1})$ and strides $(s_0, s_1, \ldots, s_{n-1})$, contiguity means $s_k = \prod_{j=k+1}^{n-1} d_j$ for all $k$.

```python
x = torch.arange(6).reshape(2, 3)
print(x.is_contiguous())   # True
print(x.stride())          # (3, 1)  -- step 3 elements between rows, 1 between cols

y = x.T                    # Transpose
print(y.is_contiguous())   # False
print(y.stride())          # (1, 3)  -- stride order reversed

# Force contiguous copy (creates new storage)
z = y.contiguous()
print(z.is_contiguous())   # True
```

Why does contiguity matter? Most GPU kernels (and ATen CPU kernels) assume contiguous layout. Operating on non-contiguous tensors either triggers an implicit `.contiguous()` copy (hurting performance) or requires a strided kernel path. In tight training loops, a silent `.contiguous()` can add meaningful overhead.

!!! warning "Silent contiguous copies"
    `torch.nn.functional.layer_norm`, `nn.Conv2d`, and many other ops call `.contiguous()` internally when their input isn't already contiguous. A common source of unexpected memory traffic is `tensor.permute(...)` followed by an op that forces contiguity. Check with `tensor.is_contiguous()` and either permute earlier or fuse permutations.

### Strides and Non-standard Layouts

Strides generalize contiguity. A stride-$s$ tensor accesses element $(i, j)$ at memory offset $i \cdot s_0 + j \cdot s_1$. This enables:

- **Transposition**: swap strides without moving data.
- **Broadcasting**: set a stride to 0 to repeat data logically without copying.
- **Slicing**: adjust the base pointer and reduce the size along a dimension.

```python
# Broadcasting via zero stride
x = torch.tensor([1.0, 2.0, 3.0])
# Expand to (4, 3) without copying:
y = x.unsqueeze(0).expand(4, 3)
print(y.stride())        # (0, 1) -- stride-0 in the batch dimension
print(y.is_contiguous()) # False
print(y.storage().size() == 3)  # True -- only 3 elements in storage
```

### Broadcasting Semantics

NumPy-style broadcasting aligns shapes from the right and stretches size-1 dimensions:

$$
(B, 1, H, W) + (C, H, W) \to (B, C, H, W)
$$

PyTorch implements this via the stride-0 trick above: a size-1 dimension that gets broadcast is assigned stride 0. No data is copied. However, when autograd differentiates through a broadcast, the backward must **sum** the gradient over the broadcast dimensions to match the original tensor's shape. This is done automatically by `torch.Tensor.expand`'s backward and by `SumBackward`.

```python
a = torch.ones(3, 1, requires_grad=True)   # shape (3, 1)
b = torch.ones(3, 4)                       # shape (3, 4), no grad

c = a + b   # broadcasts a to (3, 4)
c.sum().backward()

# grad of 'a' sums over the broadcast dim:
print(a.grad)   # tensor([[4.], [4.], [4.]])  -- sum over 4 columns
```

---

## Interview Corner and Practical Patterns

!!! interview "Interview Corner"
    **Q:** Walk me through what happens when you call `loss.backward()` in PyTorch. What data structures are involved, and what does the engine actually execute?

    **A:** When you called forward operations on tensors with `requires_grad=True`, PyTorch built a DAG of `Function` nodes connected via `next_functions` pointers, with each node holding a `forward` closure and a `backward` implementation. `loss.backward()` seeds the process by setting the gradient of `loss` to 1.0, then it calls `torch.autograd.Engine`, which runs a topological sort of the DAG and processes nodes in reverse order using a thread pool. At each node it calls the node's `backward()` method, passing in the accumulated upstream gradient, and receives gradients for the node's inputs, which it accumulates into those tensors' `.grad` fields (for leaves) or pushes onto the work queue (for non-leaves). The key implementation detail is that gradients are *accumulated* (added), not assigned, which is what allows gradient accumulation across micro-batches. After the traversal completes, leaf tensors with `requires_grad=True` hold the full gradient in `.grad`. Non-leaf gradients are discarded unless you called `retain_grad()` on them. The graph itself is freed after `backward()` by default (`retain_graph=False`), releasing the stored intermediate activations.

!!! tip "Practitioner tip: retain_graph for multi-task losses"
    If you need to call `backward()` multiple times on the same graph (e.g., computing separate gradients for a shared encoder with two losses applied sequentially), use `loss1.backward(retain_graph=True)` for all but the last call. Without `retain_graph=True`, the graph is freed after the first `backward()` and subsequent calls raise `RuntimeError: Trying to backward through the graph a second time`.

### Gradient Checkpointing (Activation Recomputation)

For very deep networks or long transformer sequences, storing all activations for the backward pass dominates memory. `torch.utils.checkpoint.checkpoint` trades compute for memory by recomputing activations during the backward pass instead of storing them:

```python
from torch.utils.checkpoint import checkpoint

def block_forward(x, layer):
    return layer(x)

# During backward, the forward of this segment will be re-run
x_out = checkpoint(block_forward, x_in, layer, use_reentrant=False)
```

This roughly halves activation memory (at the cost of one extra forward pass worth of compute per checkpointed segment), and is nearly universally used in LLM pretraining. See [Memory-Efficient Training: Checkpointing, Offloading & LoRA Math](../04-kernels-efficiency/10-memory-efficient-training.html) for a full analysis.

### Second-Order Gradients and `create_graph`

PyTorch can differentiate through the backward pass itself by passing `create_graph=True` to `backward()` or using `torch.autograd.grad`. This is used for MAML (model-agnostic meta-learning), Hessian-vector products, and implicit differentiation:

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 3         # y = x^3, dy/dx = 3x^2

# First-order gradient
(grad_x,) = torch.autograd.grad(y, x, create_graph=True)
print(grad_x)   # tensor(27.)  -- 3 * 3^2

# Second-order gradient (differentiates grad_x w.r.t. x)
(grad2_x,) = torch.autograd.grad(grad_x, x)
print(grad2_x)  # tensor(18.)  -- 6x = 6*3
```

When `create_graph=True`, the autograd graph for the backward computation is itself tracked, enabling higher-order differentiation. This doubles (or more) the memory cost.

### `torch.func` (functorch): Functional Transforms

PyTorch's `torch.func` module (formerly `functorch`) exposes composable functional transforms over arbitrary PyTorch functions:

```python
from torch.func import grad, vmap, jacrev

# Compute gradient of a scalar function
f = lambda x: (x ** 2).sum()
grad_f = grad(f)
print(grad_f(torch.tensor([1.0, 2.0, 3.0])))  # tensor([2., 4., 6.])

# Batched Jacobian via vmap + jacrev
def model(params, x):
    return params @ x

J = vmap(jacrev(model, argnums=1))(params_batch, x_batch)
```

These transforms work at the dispatcher level, inserting themselves as dispatch keys. They compose: `vmap(grad(f))` gives a batched gradient function. This is the correct modern approach to Hessian-vector products and per-sample gradients (rather than looping or using `create_graph`).

---

## The Full Stack: From Python Op to GPU Kernel

Let us trace `torch.mm(a, b)` for two CUDA tensors end to end:

{{fig:autodiff-mm-fullstack-trace}}

The backward `MmBackward` is registered in ATen's derivative formulas. For $C = AB$:

$$
\bar{A} = \bar{C} B^\top, \qquad \bar{B} = A^\top \bar{C}
$$

These are themselves `mm` calls, so they go through the same dispatcher path and are executed as cuBLAS GEMMs.

### `torch.compile` and the Dispatcher

`torch.compile` (based on TorchDynamo + TorchInductor) traces the Python bytecode, extracts a subgraph as a `torch.fx.Graph`, applies fusion passes, and emits an optimized kernel. It interacts with the dispatcher by inserting a `CompiledFunctionBackend` dispatch key. Because the dispatcher is a clean abstraction boundary, `torch.compile` can replace and fuse sequences of ATen ops without touching user code. See [Kernel Fusion, torch.compile, CUDA Graphs & Compilers](../04-kernels-efficiency/09-compilers-fusion.html) for details.

---

## Numerical Precision Considerations in Autograd

Autograd inherits the numerical properties of the operations it differentiates. Two important failure modes:

**Vanishing/exploding gradients** occur when VJPs amplify or attenuate the gradient signal across many layers. ReLU gates reduce flow (any negative pre-activation kills the gradient path), while sigmoid and tanh saturate (their derivatives approach zero). Residual connections, layer norm, and careful initialization (Kaiming, Xavier) combat this. See [Neural Networks From Scratch: MLPs & Backprop](../01-foundations/06-neural-nets-from-scratch.html) and [Numerical Computing, Floating Point & Precision](../01-foundations/04-numerics-precision.html).

**Catastrophic cancellation in backward** can occur when forward activations are large and the gradient requires subtracting nearly equal numbers. Log-softmax with cross-entropy is a canonical example: naive implementation computes $\log(\sum \exp)$ and then subtracts, losing precision. PyTorch's `F.cross_entropy` uses the log-sum-exp trick to sidestep this, and the autograd graph for the fused version is more numerically stable than separately differentiating `log` and `softmax`.

!!! warning "In-place operations and autograd"
    In-place ops (those ending in `_`, like `relu_`, `add_`) can corrupt the computation graph because they overwrite a tensor that a backward function holds a reference to. PyTorch tracks version counters and raises a `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation` if you trigger this. The rule of thumb: avoid in-place ops on tensors that `requires_grad=True`. If you need them for memory reasons, ensure they happen outside the part of the graph that needs to be differentiated.

### Localizing NaNs in the Backward Pass: `set_detect_anomaly`

The hardest autograd bug to diagnose is a loss or gradient that becomes NaN or Inf during the *backward* pass while the *forward* pass looked perfectly finite — e.g. an operation whose value is well-defined at a point but whose derivative is not. The backward NaN surfaces far from its cause (in `optimizer.step()`, or as a NaN grad norm in a logging call), so the offending forward op is hidden several layers upstream.

`torch.autograd.set_detect_anomaly(True)` (or the scoped context manager `with torch.autograd.detect_anomaly():`) turns on anomaly detection, which does two things: it checks every op's output for **NaN** (note: it flags NaN, *not* Inf), and — crucially — it records the *forward*-pass Python stack trace for each `Function` node, so that when a backward computes a NaN it raises a `RuntimeError` whose traceback points at the exact forward line that created the offending op.

```python
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

# With anomaly detection, backward raises AT the offending forward op:
x = torch.zeros(3, requires_grad=True)
with torch.autograd.detect_anomaly():
    y = (x * x.sqrt()).sum()  # anomaly mode stores this line's stack trace
    y.backward()
# RuntimeError: Function 'SqrtBackward0' returned nan values in its 0th output.
# The traceback includes:  "y = (x * x.sqrt()).sum()"  <- the forward line to fix
```

The fix is typically an epsilon or a clamp on the input to the unstable op — e.g. `x.clamp_min(1e-12).sqrt()`, or adding `eps` inside the op — after which re-running under `detect_anomaly()` no longer raises. The same mechanism catches the more common real-world cause in LLM training: a `log`, `sqrt`, division, or `pow` fed a zero or negative value from an upstream overflow. `set_detect_anomaly(True)` is the global switch you flip once at the top of a debug run; `with torch.autograd.detect_anomaly():` scopes it to a single step.

!!! warning "Anomaly detection is debug-only"
    Enabling anomaly detection stores a Python stack trace for every op in the forward graph and NaN-checks every intermediate, which can slow training by 10x or more and greatly increases memory and host overhead. Never leave it on in a production or full training run — wrap only the single reproducing step, or gate it behind a `--debug-anomaly` flag, and turn it off (`torch.autograd.set_detect_anomaly(False)`) once the offending op is found.

In a large pretraining run, reach for this at STEP 3 ("isolate the step") of the loss-spike playbook in [Training Stability, Loss Spikes & Debugging Large Runs](../03-pretraining/11-training-stability.html): after rolling back to the pre-spike checkpoint and replaying the exact batch, wrapping that single forward+backward in `detect_anomaly()` pinpoints the op, complementing the forward-hook activation probe used there to find which layer's activations blew up first.

---

## Key Takeaways

!!! key "Key Takeaways"
    - Reverse-mode autodiff computes exact gradients in $O(1)$ forward passes by recording a tape of operations and traversing it in reverse, computing vector-Jacobian products (VJPs) at each node.
    - PyTorch builds the computation graph dynamically (define-by-run) during the forward pass; each output tensor has a `grad_fn` pointing to the `Function` node that created it, and `next_functions` edges point toward the leaves.
    - After `loss.backward()`, only **leaf** tensors (those created directly with `requires_grad=True`, typically `nn.Parameter`) accumulate gradients in `.grad`; intermediate tensors' gradients are discarded unless `retain_grad()` is called.
    - `torch.no_grad()` and `torch.inference_mode()` suppress graph construction for inference; `inference_mode` is stricter and slightly faster. Always use one or the other during eval.
    - Custom `autograd.Function` subclasses let you inject arbitrary forward/backward logic into the autograd graph; use `ctx.save_for_backward` for tensors, `gradcheck` to verify correctness.
    - PyTorch's **dispatcher** routes each operation to the correct backend (CPU, CUDA, XLA) based on dispatch keys; `torch.compile`, `vmap`, and `grad` (torch.func) are all dispatcher-level transforms that compose cleanly.
    - **Views** share storage with the original tensor (zero copy); **contiguity** determines whether kernels can operate without an implicit copy. Non-contiguous tensors frequently cause silent performance regressions.
    - Broadcasting is implemented via stride-0 dimensions; the backward pass of a broadcast automatically sums gradients over the expanded dimensions.
    - Gradient checkpointing (activation recomputation) trades one extra forward pass for roughly halved activation memory, and is standard practice in LLM pretraining.

---

!!! sota "State of the Art & Resources (2026)"
    Reverse-mode autodiff is a mature and stable discipline; the frontier today lies in composable functional transforms (vmap, jvp, vjp), ahead-of-time graph capture via `torch.compile`, and second-order methods for meta-learning and physics-based optimization. PyTorch's dispatcher abstraction continues to absorb new hardware backends and compiler passes without breaking user-facing APIs.

    **Foundational work**

    - [Baydin et al., *Automatic Differentiation in Machine Learning: a Survey* (2018)](https://arxiv.org/abs/1502.05767) — the definitive academic survey covering forward-mode, reverse-mode, and their relationships to symbolic and numerical differentiation.
    - [Frostig et al., *Decomposing Reverse-Mode Automatic Differentiation* (2021)](https://arxiv.org/abs/2105.09469) — the JAX/functorch perspective showing that reverse-mode AD decomposes into forward-mode linearization followed by transposition, simplifying composable implementations.
    - [Chen et al., *Training Deep Nets with Sublinear Memory Cost* (2016)](https://arxiv.org/abs/1604.06174) — the foundational gradient checkpointing paper; the O(√n) activation-memory algorithm standard in all LLM pretraining stacks.

    **Recent advances (2023–2026)**

    - [Ansel, Yang et al., *PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation* (ASPLOS 2024)](https://openreview.net/forum?id=jEWgxZ1XCe) — the paper behind `torch.compile` / TorchDynamo + TorchInductor; explains how autograd and compilation interact at the dispatcher level.

    **Open-source & tools**

    - [pytorch/functorch](https://github.com/pytorch/functorch) — the original JAX-like composable transform library (vmap, grad, jacrev) for PyTorch, now merged into `torch.func`.
    - [pytorch/pytorch `torch/csrc/autograd/`](https://github.com/pytorch/pytorch) — the canonical C++ autograd engine source; `engine.cpp` and `function.h` are the fastest path to understanding execution ordering and thread pools.

    **Go deeper**

    - [PyTorch official tutorial: *A Gentle Introduction to torch.autograd*](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) — the recommended starting point for understanding the autograd API and DAG mechanics.
    - [PyTorch docs: *Autograd mechanics*](https://docs.pytorch.org/docs/2.12/notes/autograd.html) — deep-dive reference covering saved tensors, in-place ops, multithreaded backward, and Wirtinger calculus for complex numbers.
    - [PyTorch docs: *Extending PyTorch* (custom autograd.Function)](https://docs.pytorch.org/docs/stable/notes/extending.html) — official reference for writing custom `Function` subclasses and registering new operators.
    - [E. Yang, *Let's Talk About the PyTorch Dispatcher* (2020)](https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/) — the canonical deep-dive into the dispatch key table, operator registration, and boxing/unboxing.
    - [PyTorch tutorial: *Jacobians, Hessians, hvp, vhp, and more* (torch.func)](https://docs.pytorch.org/tutorials/intermediate/jacobians_hessians.html) — practical guide to composing vmap, vjp, and jvp for per-sample gradients and higher-order derivatives.

## Further Reading

- Baydin et al., **"Automatic Differentiation in Machine Learning: a Survey"** (2018) — the definitive academic survey of all autodiff modes.
- Paszke et al., **"Automatic differentiation in PyTorch"** (NIPS 2017 Autodiff Workshop) — the original PyTorch autograd paper.
- Wengert, **"A simple automatic derivative evaluation program"** (1964) — the original Wengert list / tape paper.
- PyTorch documentation, **"Extending PyTorch"** — official reference for `torch.autograd.Function` and the dispatcher.
- Pytorch contributor docs, **"PyTorch Dispatcher internals"** (E. Yang, PyTorch blog, 2021) — deep dive into the dispatcher architecture.
- Frostig et al., **"Decomposing reverse-mode automatic differentiation"** (2021) — the JAX / functorch perspective on composable transforms.
- PyTorch GitHub, `torch/csrc/autograd/` — the C++ engine source; reading `engine.cpp` and `function.h` is the fastest way to understand execution ordering and thread pools.
