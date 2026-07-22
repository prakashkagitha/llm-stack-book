# 1.6 Neural Networks From Scratch: MLPs & Backprop

If you can build backpropagation from nothing — a few hundred lines of pure Python and NumPy, no PyTorch, no autograd library — then every large model in this book becomes legible. A 70-billion-parameter transformer is not a different *kind* of object than the two-layer multi-layer perceptron (MLP) we are about to construct; it is the same machine with more layers, fancier blocks, and a vastly larger matrix-multiply budget. The forward pass is still a chain of differentiable functions. The backward pass is still the chain rule applied mechanically, right to left. The optimizer is still "nudge each parameter against its gradient."

This chapter is the hinge of Part I. We take the linear algebra of [Linear Algebra for Deep Learning](../01-foundations/01-linear-algebra.html), the gradients and convexity of [Calculus, Optimization & Convexity](../01-foundations/03-calculus-optimization.html), and the learning-theory framing of [Machine Learning Fundamentals](../01-foundations/05-ml-fundamentals.html), and fuse them into a working neural network. We build twice: first a **scalar autograd engine** (micrograd-style) that makes the chain rule physically visible one operation at a time, then a **vectorized MLP** that is fast enough to actually train. Along the way we confront the things that make deep networks hard in practice — the choice of activation, the initialization scheme (Xavier and Kaiming), the twin demons of vanishing and exploding gradients, and the stabilizing trick of batch normalization. We close with the interview question every ML engineer is eventually asked: *derive and implement backprop on a whiteboard.*

When you finish, the autograd machinery of [Automatic Differentiation & PyTorch Internals](../01-foundations/07-autodiff-pytorch.html) will feel like an engineering refinement of something you already understand, not magic.

---

## The MLP: A Function You Can Differentiate

### What a neuron computes

The atom of a neural network is the **artificial neuron**: a weighted sum of inputs, a bias, and a nonlinearity. Given an input vector $\mathbf{x} \in \mathbb{R}^{d}$, weights $\mathbf{w} \in \mathbb{R}^{d}$, and bias $b \in \mathbb{R}$, the neuron outputs

$$
a = \phi\!\left(\mathbf{w}^\top \mathbf{x} + b\right) = \phi\!\left(\sum_{i=1}^{d} w_i x_i + b\right),
$$

where $\phi$ is an **activation function** (sigmoid, tanh, ReLU, …). The pre-activation $z = \mathbf{w}^\top \mathbf{x} + b$ is an affine function of the input; the activation $\phi$ bends it. Without $\phi$, stacking neurons would collapse into a single affine map — composition of linear maps is linear — and the whole network could represent nothing more than a single matrix. The nonlinearity is what buys *depth*.

A **layer** is a stack of neurons sharing the same input. Collect the per-neuron weight vectors as rows of a matrix $W \in \mathbb{R}^{h \times d}$ (for $h$ neurons) and the biases into $\mathbf{b} \in \mathbb{R}^{h}$. Then the layer is

$$
\mathbf{z} = W\mathbf{x} + \mathbf{b}, \qquad \mathbf{a} = \phi(\mathbf{z}),
$$

with $\phi$ applied elementwise. An **MLP** (also called a feed-forward network) is a chain of $L$ such layers:

$$
\mathbf{a}^{(0)} = \mathbf{x}, \qquad
\mathbf{z}^{(\ell)} = W^{(\ell)} \mathbf{a}^{(\ell-1)} + \mathbf{b}^{(\ell)}, \qquad
\mathbf{a}^{(\ell)} = \phi^{(\ell)}\!\left(\mathbf{z}^{(\ell)}\right),
$$

for $\ell = 1, \dots, L$. The final $\mathbf{a}^{(L)}$ is the network's prediction. Every modern architecture — including the feed-forward sublayer of a transformer block in [The Transformer Block: Norms, Residuals, MLPs & Activations](../02-transformer/06-transformer-block.html) — is built from exactly this primitive.

### Why depth and width matter

The **universal approximation theorem** (Cybenko, Hornik) says a single hidden layer with enough neurons can approximate any continuous function on a compact set to arbitrary accuracy. So why go deep? Because *width* needed to represent certain functions can be exponential, while *depth* lets you represent the same function with polynomially many parameters by composing features. Depth trades a hard combinatorial problem (find one enormous layer) for an easier compositional one (find many small layers, each building on the last). The catch is that depth makes the optimization landscape harder — which is the entire reason initialization, normalization, and residual connections exist.

### Batching: from vectors to matrices

We never train on one example at a time. Stack a mini-batch of $N$ inputs as rows of a matrix $X \in \mathbb{R}^{N \times d}$. The layer becomes a single matrix multiply:

$$
Z = X W^\top + \mathbf{1}\mathbf{b}^\top \in \mathbb{R}^{N \times h}, \qquad A = \phi(Z).
$$

Here $\mathbf{1} \in \mathbb{R}^{N}$ broadcasts the bias across rows. This **batched, row-major** convention (inputs as rows, $W$ stored so we right-multiply by $W^\top$, or equivalently store $W \in \mathbb{R}^{d \times h}$ and compute $XW$) is what every framework uses, because a big matmul is exactly what GPUs are built to do — see [GPU Architecture & The Memory Hierarchy](../01-foundations/08-gpu-architecture.html). For the rest of this chapter we store layer weights as $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ and compute $Z = XW + \mathbf{b}$.

```python
import numpy as np

def affine_forward(X, W, b):
    """One linear layer, batched.
    X: (N, d_in)  inputs as rows
    W: (d_in, d_out)
    b: (d_out,)
    returns Z: (N, d_out)
    """
    return X @ W + b                       # bias broadcasts over the N rows

# Sanity check shapes with a toy batch.
N, d_in, d_out = 4, 3, 5
X = np.random.randn(N, d_in)
W = np.random.randn(d_in, d_out)
b = np.random.randn(d_out)
Z = affine_forward(X, W, b)
print(Z.shape)                              # (4, 5)
```

---

## Backpropagation: The Chain Rule, Mechanized

### The training loop in one breath

We want to minimize a scalar **loss** $L(\theta)$ over parameters $\theta = \{W^{(\ell)}, \mathbf{b}^{(\ell)}\}$. Gradient descent does

$$
\theta \leftarrow \theta - \eta \, \nabla_\theta L,
$$

with **learning rate** $\eta$. The only hard part is computing $\nabla_\theta L$. For a network with millions of parameters, computing each partial derivative by a separate finite-difference probe would cost millions of forward passes. **Backpropagation** computes *all* partials in a single backward sweep — at roughly the same cost as one forward pass — by reusing intermediate results via the chain rule. This is the single most important algorithm in the book.

### The chain rule, stated precisely

Let the loss be a composition $L = f_k \circ f_{k-1} \circ \cdots \circ f_1$. The chain rule says the derivative of the composition is the product of the local derivatives. For scalars,

$$
\frac{dL}{dx} = \frac{dL}{du_k}\frac{du_k}{du_{k-1}} \cdots \frac{du_1}{dx}.
$$

For vector/matrix intermediates we replace each $\frac{du_i}{du_{i-1}}$ by a Jacobian and the products by matrix multiplies. The crucial insight that makes backprop efficient: we evaluate this product **right to left from the loss**, carrying a running quantity called the **upstream gradient** (or *adjoint*). For every intermediate tensor $u$ we define

$$
\bar{u} \;\equiv\; \frac{\partial L}{\partial u},
$$

a tensor the same shape as $u$. Backprop is the rule for turning the adjoint of an operation's *output* into the adjoint of its *inputs*. We call that rule a **vector-Jacobian product** (VJP): we never materialize the full Jacobian, only its product with the upstream gradient.

### Forward graph, backward graph

Think of the computation as a **directed acyclic graph** (DAG). Nodes are operations; edges carry tensors. The forward pass walks the graph from inputs to loss, caching whatever each node will need later. The backward pass walks the same graph *in reverse topological order*, and at each node converts output-adjoints into input-adjoints, accumulating into each tensor's $\bar{u}$.

```text
   FORWARD  (left -> right)                 BACKWARD (right -> left)
   x --[*W1+b1]--> z1 --[relu]--> a1            x̄  <--  z̄1  <--  ā1
                                  |                              |
        a1 --[*W2+b2]--> z2 --[softmax+CE]--> L      ā1 <- z̄2 <- L̄=1
   Each node caches its inputs.          Each node applies its VJP rule.
```

{{fig:backprop-two-sweep-dag}}

There are exactly two facts you must memorize, because *every* dense-layer gradient is one of them. Let $Z = XW + \mathbf{b}$ with upstream gradient $\bar{Z} = \partial L / \partial Z$. Then

$$
\boxed{\;\bar{W} = X^\top \bar{Z}, \qquad \bar{X} = \bar{Z}\, W^\top, \qquad \bar{\mathbf{b}} = \mathbf{1}^\top \bar{Z}.\;}
$$

The bias gradient sums the upstream gradient over the batch dimension. These three identities — derived in full in [Linear Algebra for Deep Learning](../01-foundations/01-linear-algebra.html) — are the workhorses of backprop. Notice the elegant symmetry: the input adjoint $\bar X$ reuses $W$, the weight adjoint $\bar W$ reuses $X$. Each operand of the matmul shows up transposed in the other's gradient.

{{fig:dense-layer-gradient-transpose-symmetry}}

### A scalar autograd engine (micrograd-style)

The cleanest way to *internalize* backprop is to build it for scalars, where there are no Jacobians to confuse you — every local derivative is a single number. We build a `Value` class that wraps a float, records how it was produced, and knows how to push gradients to its parents. This is the micrograd pattern popularized by Andrej Karpathy, reconstructed here.

```python
import math

class Value:
    """A scalar that remembers how it was computed, so it can backprop."""

    def __init__(self, data, _children=(), _op=""):
        self.data = data            # the forward value (a float)
        self.grad = 0.0             # dL/d(self), accumulated during backward
        self._backward = lambda: None   # closure: pushes grad to parents
        self._prev = set(_children)     # parent Values in the graph
        self._op = _op                  # label, for debugging/visualization

    # ---- arithmetic: each op builds a new node AND defines its local VJP ----
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1  -> just pass the upstream grad through
            self.grad  += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # d(a*b)/da = b, d(a*b)/db = a  -> multiply upstream by the OTHER operand
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad
        out._backward = _backward
        return out

    def __pow__(self, p):
        assert isinstance(p, (int, float))
        out = Value(self.data ** p, (self,), f"**{p}")

        def _backward():
            # d(a**p)/da = p * a**(p-1)
            self.grad += (p * self.data ** (p - 1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(0.0 if self.data < 0 else self.data, (self,), "relu")

        def _backward():
            # derivative is 0 where input<0, else 1 (the "gate" passes grad or blocks it)
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            # d(tanh)/dx = 1 - tanh(x)^2
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    # ---- the backward pass: reverse topological order over the DAG ----
    def backward(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)         # children appended before parents
        build(self)

        self.grad = 1.0                # seed: dL/dL = 1
        for v in reversed(topo):       # parents to children, i.e. loss -> inputs
            v._backward()              # each node pushes grad to its parents

    # ---- conveniences so we can write natural math ----
    def __neg__(self):        return self * -1
    def __sub__(self, o):     return self + (-o)
    def __radd__(self, o):    return self + o
    def __rmul__(self, o):    return self * o
    def __truediv__(self, o): return self * (o ** -1 if isinstance(o, Value) else o ** -1)
    def __repr__(self):       return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
```

Three things make this work. First, **each operation builds a new node and defines a closure** `_backward` that knows the local derivative — the VJP rule for that op. Second, **gradients accumulate** (`+=`, not `=`) because a value used in two places receives gradient along both paths; summing them is exactly the multivariate chain rule. Third, **the topological sort** guarantees we only call a node's `_backward` after its output's gradient is fully assembled.

Let us verify it against calculus by hand on a tiny expression. Build $L = (x \cdot w + b)$ then a ReLU then square the error against a target — a one-neuron regression.

```python
x, w, b = Value(2.0), Value(-3.0), Value(1.0)
z = x * w + b                 # z = 2*(-3) + 1 = -5
a = z.relu()                  # relu(-5) = 0
target = Value(4.0)
loss = (a - target) ** 2      # (0 - 4)^2 = 16
loss.backward()

print(f"loss = {loss.data}")  # 16.0
print(f"dL/dw = {w.grad}")    # 0.0  (ReLU gate is closed: z<0 blocks all gradient)
print(f"dL/dx = {x.grad}")    # 0.0
print(f"dL/db = {b.grad}")    # 0.0
```

Every gradient is zero because the ReLU's input was negative, so the gate is shut and no gradient flows back — a vivid demonstration of the **dead-ReLU** phenomenon we revisit below. Flip the sign of `w` to `+3.0` and you will see nonzero gradients flow, because the gate opens. This single closure-per-op design *is* reverse-mode automatic differentiation; [Automatic Differentiation & PyTorch Internals](../01-foundations/07-autodiff-pytorch.html) is the same idea scaled to tensors with a tape and C++ kernels.

!!! warning "Common pitfall: forgetting to zero gradients"

    Because gradients **accumulate** (`+=`), you must reset every parameter's `.grad` to zero before each backward pass, or you will sum gradients across iterations and your steps will be garbage. In PyTorch this is `optimizer.zero_grad()`; in our engine you would loop and set `p.grad = 0.0`. Accumulation is a feature — it is how you implement gradient accumulation over micro-batches — but forgetting to zero is one of the most common beginner bugs, and it fails silently (training just diverges or stalls).

---

## A Vectorized MLP in Pure NumPy

The scalar engine is for understanding; it would take hours to train MNIST one float at a time. Production backprop is **vectorized**: each layer is a matrix op and its VJP is a matrix op. We now write a complete, runnable MLP classifier in NumPy, with the layer gradients derived above. No autograd — we write the backward pass by hand, which is the best way to truly own it.

### Activations and their derivatives

We need each activation paired with its derivative (expressed in terms of cached forward quantities, which is what makes backprop cheap).

| Activation | $\phi(z)$ | $\phi'(z)$ | Notes |
|---|---|---|---|
| Sigmoid | $\sigma(z)=\frac{1}{1+e^{-z}}$ | $\sigma(z)\,(1-\sigma(z))$ | saturates; gradient $\le 0.25$ |
| Tanh | $\tanh(z)$ | $1-\tanh^2(z)$ | zero-centered; saturates |
| ReLU | $\max(0,z)$ | $\mathbb{1}[z>0]$ | cheap; can "die" |
| Leaky ReLU | $\max(\alpha z, z)$ | $\alpha$ if $z<0$ else $1$ | fixes dead units |
| GELU | $z\,\Phi(z)$ | $\Phi(z)+z\,\phi(z)$ | smooth; LLM default |

GELU (Gaussian Error Linear Unit; $\Phi$ is the standard normal CDF) is the activation inside most transformer feed-forward blocks; we cover it in [The Transformer Block](../02-transformer/06-transformer-block.html). For our from-scratch MLP we use ReLU, whose derivative is a simple gate.

```python
import numpy as np

def relu(z):        return np.maximum(0.0, z)
def relu_grad(z):   return (z > 0.0).astype(z.dtype)

def softmax(z):
    """Numerically stable softmax over the last axis. (See chapter 1.4 on numerics.)"""
    z = z - z.max(axis=-1, keepdims=True)        # subtract max -> no overflow in exp
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
```

The max-subtraction trick prevents $e^{z}$ from overflowing in float32 — the same numerical-stability concern detailed in [Numerical Computing, Floating Point & Precision](../01-foundations/04-numerics-precision.html).

### The softmax + cross-entropy gradient

For classification we end with a softmax over $C$ classes and the **cross-entropy** loss. Given logits $\mathbf{z} \in \mathbb{R}^{C}$, probabilities $\mathbf{p} = \operatorname{softmax}(\mathbf{z})$, and one-hot target $\mathbf{y}$, the per-example loss is $L = -\sum_c y_c \log p_c$. The gradient through the *combined* softmax-cross-entropy is famously clean:

$$
\frac{\partial L}{\partial \mathbf{z}} = \mathbf{p} - \mathbf{y}.
$$

This is one of the most satisfying identities in deep learning: the gradient of the logits is simply "predicted minus true." It is also why we fuse the two operations — computing them separately and chaining their Jacobians would be wasteful and numerically worse. The derivation lives in [Probability, Statistics & Information Theory](../01-foundations/02-probability-information.html); here we just use it.

### The full MLP class

```python
import numpy as np

class MLP:
    """A from-scratch multi-layer perceptron classifier with hand-written backprop.

    sizes: e.g. [784, 256, 128, 10] -> input 784, two hidden layers, 10 classes.
    Hidden layers use ReLU; the output layer feeds softmax + cross-entropy.
    """

    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.sizes = sizes
        self.W, self.b = [], []
        for d_in, d_out in zip(sizes[:-1], sizes[1:]):
            # Kaiming/He init for ReLU: variance 2/d_in keeps signal scale ~constant.
            std = np.sqrt(2.0 / d_in)
            self.W.append(rng.normal(0.0, std, size=(d_in, d_out)))
            self.b.append(np.zeros(d_out))

    def forward(self, X):
        """Return logits and a cache of intermediates needed for backward."""
        cache = {"A": [X]}                      # activations, A[0] = input
        cache["Z"] = []                          # pre-activations
        A = X
        L = len(self.W)
        for i in range(L):
            Z = A @ self.W[i] + self.b[i]        # affine
            cache["Z"].append(Z)
            if i < L - 1:                        # hidden layers: ReLU
                A = relu(Z)
            else:                                # output layer: leave as logits
                A = Z
            cache["A"].append(A)
        return A, cache                          # A is the logits (N, C)

    def loss_and_grads(self, X, y):
        """Forward, compute cross-entropy loss, then backprop all gradients."""
        N = X.shape[0]
        logits, cache = self.forward(X)
        P = softmax(logits)                      # (N, C) probabilities

        # cross-entropy loss, averaged over the batch
        # pick the prob of the true class for each row; -log of it
        correct_logp = -np.log(P[np.arange(N), y] + 1e-12)
        loss = correct_logp.mean()

        # ---- BACKWARD ----
        # Seed: dL/dlogits = (P - onehot(y)) / N  (softmax+CE fused gradient)
        dZ = P.copy()
        dZ[np.arange(N), y] -= 1.0
        dZ /= N                                  # because loss was a mean over N

        dW = [None] * len(self.W)
        db = [None] * len(self.b)
        L = len(self.W)
        for i in reversed(range(L)):
            A_prev = cache["A"][i]               # input to layer i, shape (N, d_in)
            # The two golden identities:
            dW[i] = A_prev.T @ dZ                # (d_in, d_out)  =  X^T @ dZ
            db[i] = dZ.sum(axis=0)               # sum upstream grad over the batch
            if i > 0:                            # propagate to the previous layer
                dA_prev = dZ @ self.W[i].T       # (N, d_in)  =  dZ @ W^T
                # gate the gradient through the ReLU of layer i-1
                dZ = dA_prev * relu_grad(cache["Z"][i - 1])
        return loss, dW, db

    def step(self, dW, db, lr):
        """Vanilla SGD parameter update."""
        for i in range(len(self.W)):
            self.W[i] -= lr * dW[i]
            self.b[i] -= lr * db[i]

    def predict(self, X):
        logits, _ = self.forward(X)
        return logits.argmax(axis=1)
```

Read the backward loop carefully — it is the entire chapter compressed into ten lines. We seed `dZ` with the fused softmax-cross-entropy gradient $(\mathbf{p}-\mathbf{y})/N$. Then for each layer, right to left, we apply $\bar W = X^\top \bar Z$ and $\bar{\mathbf b}=\mathbf 1^\top\bar Z$, propagate $\bar X = \bar Z W^\top$, and **gate** that gradient through the previous layer's ReLU by multiplying with `relu_grad`. That gate is the chain rule for the elementwise nonlinearity: $\bar Z^{(\ell-1)} = \bar A^{(\ell-1)} \odot \phi'(Z^{(\ell-1)})$.

### Training it, and checking gradients

Before trusting any hand-written backward pass, **gradient-check** it against finite differences. The numerical gradient of a scalar loss with respect to a parameter $\theta_i$ is

$$
\frac{\partial L}{\partial \theta_i} \approx \frac{L(\theta_i + \epsilon) - L(\theta_i - \epsilon)}{2\epsilon},
$$

the **central difference**, accurate to $O(\epsilon^2)$. If your analytic gradient and the numerical one agree to ~6 decimal places, your backprop is almost certainly correct.

```python
def gradient_check(net, X, y, eps=1e-5):
    """Compare analytic dW[0] against central-difference numerical gradient."""
    loss, dW, db = net.loss_and_grads(X, y)
    W = net.W[0]
    max_rel_err = 0.0
    rng = np.random.default_rng(1)
    for _ in range(20):                          # spot-check 20 random entries
        i, j = rng.integers(W.shape[0]), rng.integers(W.shape[1])
        orig = W[i, j]
        W[i, j] = orig + eps; Lp, *_ = net.loss_and_grads(X, y)
        W[i, j] = orig - eps; Lm, *_ = net.loss_and_grads(X, y)
        W[i, j] = orig                            # restore
        num = (Lp - Lm) / (2 * eps)
        ana = dW[0][i, j]
        rel = abs(num - ana) / (abs(num) + abs(ana) + 1e-12)
        max_rel_err = max(max_rel_err, rel)
    return max_rel_err

# ---- a tiny synthetic two-moons-ish dataset for a runnable demo ----
rng = np.random.default_rng(42)
N, D, C = 512, 2, 3
X = rng.normal(size=(N, D))
# label by angle to make 3 non-linearly-separable blobs
theta = np.arctan2(X[:, 1], X[:, 0])
y = ((theta + np.pi) / (2 * np.pi) * C).astype(int) % C

net = MLP([D, 64, 64, C], seed=0)
print("gradient check (relative error):", gradient_check(net, X, y))  # ~1e-7

for epoch in range(2000):
    loss, dW, db = net.loss_and_grads(X, y)
    net.step(dW, db, lr=0.2)
    if epoch % 400 == 0:
        acc = (net.predict(X) == y).mean()
        print(f"epoch {epoch:4d}  loss {loss:.4f}  acc {acc:.3f}")
```

Running this, the gradient check prints a relative error on the order of $10^{-7}$ — the gap between analytic and numerical gradients is pure floating-point noise, confirming the backward pass is exact. The loss falls and accuracy climbs toward 1.0 as the two hidden ReLU layers carve the angular decision boundary that no single linear layer could. You have now trained a neural network with code you could write on a whiteboard.

---

## Initialization and the Gradient Pathologies

A subtle truth: a correct backward pass is *necessary but not sufficient* to train a deep network. If you initialize the weights badly, the signal either explodes or vanishes as it propagates, and learning stalls. Understanding why leads directly to the Xavier and Kaiming initialization schemes.

### Variance propagation, forward and backward

Consider a linear layer $z_j = \sum_{i=1}^{d_{\text{in}}} W_{ij} x_i$ with weights drawn i.i.d. with mean $0$ and variance $\operatorname{Var}(W)$, and inputs $x_i$ i.i.d. with variance $\operatorname{Var}(x)$, all independent. The variance of each output is the sum of $d_{\text{in}}$ independent products:

$$
\operatorname{Var}(z) = d_{\text{in}} \cdot \operatorname{Var}(W) \cdot \operatorname{Var}(x).
$$

For the *forward* signal to neither grow nor shrink across a layer, we want $\operatorname{Var}(z) = \operatorname{Var}(x)$, which requires

$$
\operatorname{Var}(W) = \frac{1}{d_{\text{in}}}.
$$

Now run the same argument on the *backward* pass. The gradient flows back through $W^\top$, so by symmetry the gradient variance is preserved when $\operatorname{Var}(W) = 1/d_{\text{out}}$. You cannot satisfy both unless $d_{\text{in}}=d_{\text{out}}$, so **Xavier/Glorot initialization** compromises with the harmonic-mean-flavored average:

$$
\operatorname{Var}(W) = \frac{2}{d_{\text{in}} + d_{\text{out}}}.
$$

Xavier was derived assuming a symmetric activation with unit derivative near zero (like tanh). ReLU is different: it zeroes out half its inputs in expectation, halving the variance. **Kaiming/He initialization** corrects for this by doubling the variance:

$$
\operatorname{Var}(W) = \frac{2}{d_{\text{in}}} \quad \text{(for ReLU)}.
$$

That factor of 2 is exactly the `np.sqrt(2.0 / d_in)` in our `MLP.__init__`. It is not a hyperparameter to tune — it is a derived constant that keeps activation magnitudes roughly constant from the first layer to the last.

!!! example "Worked example: signal scale across 50 layers"

    Take a 50-layer deep linear-plus-ReLU stack with width $d=256$ and a unit-variance input. With **bad init** ($\operatorname{Var}(W)=1/d^2$, i.e. weights too small), each layer multiplies the activation standard deviation by roughly $\sqrt{d\cdot\operatorname{Var}(W)\cdot\tfrac12}=\sqrt{256\cdot(1/256^2)\cdot 0.5}\approx 0.044$. After 50 layers the signal scale is $0.044^{50}\approx 10^{-69}$ — total collapse; gradients vanish to zero and nothing learns.

    With **Kaiming init** ($\operatorname{Var}(W)=2/d$), each ReLU layer multiplies the std by $\sqrt{d\cdot(2/d)\cdot\tfrac12}=\sqrt{1}=1$. After 50 layers the signal scale is still $\approx 1.0$. The forward activations and backward gradients stay $O(1)$ all the way down. This is the entire point of principled initialization: it makes deep networks *trainable at all*.

{{fig:variance-propagation-three-regimes}}

```python
import numpy as np

def measure_signal(init_std, depth=50, width=256, seed=0):
    """Push a unit-variance batch through `depth` linear+ReLU layers, report std."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(1024, width))            # unit-variance input batch
    for _ in range(depth):
        W = rng.normal(0.0, init_std, size=(width, width))
        A = np.maximum(0.0, A @ W)                 # linear + ReLU
    return A.std()

d = 256
print("too small  :", measure_signal(1.0 / d))            # -> ~0 (vanishes)
print("kaiming    :", measure_signal(np.sqrt(2.0 / d)))   # -> ~O(1)
print("too large  :", measure_signal(np.sqrt(8.0 / d)))   # -> explodes
```

### Vanishing and exploding gradients

The variance argument has a dynamic consequence. In a deep chain, the gradient at layer 1 is a product of many Jacobian factors:

$$
\frac{\partial L}{\partial \mathbf{a}^{(1)}}
= \frac{\partial L}{\partial \mathbf{a}^{(L)}}
\prod_{\ell=2}^{L} \underbrace{\operatorname{diag}\!\big(\phi'(\mathbf{z}^{(\ell)})\big)\,W^{(\ell)}}_{\text{Jacobian of layer } \ell}.
$$

If the typical singular value of each Jacobian factor is below 1, the product shrinks geometrically — **vanishing gradients**: early layers receive almost no signal and barely update. If it is above 1, the product grows geometrically — **exploding gradients**: updates blow up, losses go to NaN. This is the deep-learning version of "raising a number to the 50th power": anything not exactly 1 runs away.

Saturating activations make vanishing worse. Sigmoid's derivative peaks at $0.25$ and is near zero in its tails; chain 20 sigmoids and the gradient is multiplied by at most $0.25^{20}\approx 10^{-12}$. This is precisely why sigmoid and tanh fell out of favor for deep hidden layers, replaced by ReLU (derivative exactly 1 in the active region). The three structural fixes the field converged on are:

1. **Good initialization** (Xavier/Kaiming) — start the product near 1.
2. **Non-saturating activations** (ReLU and friends) — keep per-layer derivatives near 1.
3. **Normalization layers** (BatchNorm, LayerNorm) and **residual connections** — actively re-center the signal and give gradients a multiplication-free highway. Residuals are the reason transformers can be hundreds of layers deep; see [The Transformer Block](../02-transformer/06-transformer-block.html).

For exploding gradients specifically, the standard runtime guard is **gradient clipping**: rescale the whole gradient if its global norm exceeds a threshold $\tau$.

```python
def clip_grads(grads, max_norm):
    """Global-norm gradient clipping, as used in essentially every LLM run."""
    total = np.sqrt(sum((g ** 2).sum() for g in grads))
    if total > max_norm:
        scale = max_norm / (total + 1e-6)
        grads = [g * scale for g in grads]
    return grads
```

This exact technique stabilizes large-model training runs; we return to it in [Training Stability, Loss Spikes & Debugging Large Runs](../03-pretraining/11-training-stability.html).

### The dead-ReLU trap

ReLU's blessing — a zero-or-one gate — is also its curse. If a neuron's pre-activation is negative for *every* training example (often because of a large negative bias or an unlucky update), its gradient is zero forever and it never recovers. This is a **dead ReLU**. We saw it concretely in the scalar engine, where a single negative pre-activation zeroed every upstream gradient. **Leaky ReLU** ($\phi(z)=\max(\alpha z, z)$ with small $\alpha\approx 0.01$) and its learnable cousin PReLU fix this by letting a trickle of gradient through the negative region, keeping the unit alive.

---

## Batch Normalization

Even with good initialization, the *distribution* of each layer's inputs shifts as training proceeds and earlier weights change — the original BatchNorm paper called this **internal covariate shift**. **Batch normalization** (Ioffe & Szegedy, 2015) attacks this directly: it normalizes each pre-activation to zero mean and unit variance *across the mini-batch*, then lets the network learn its own scale and shift.

### The BatchNorm transform

For a feature $z$ over a mini-batch of size $N$, BatchNorm computes the batch mean and variance, normalizes, and applies a learnable affine $(\gamma, \beta)$:

$$
\mu = \frac{1}{N}\sum_{n} z_n, \quad
\sigma^2 = \frac{1}{N}\sum_{n}(z_n-\mu)^2, \quad
\hat{z}_n = \frac{z_n - \mu}{\sqrt{\sigma^2 + \varepsilon}}, \quad
y_n = \gamma\,\hat{z}_n + \beta.
$$

The $\gamma$ and $\beta$ are trained like any other parameter; crucially, the network *can* undo the normalization (by learning $\gamma=\sqrt{\sigma^2+\varepsilon}$, $\beta=\mu$) if that is optimal, so BatchNorm never reduces representational power. At **inference time** there is no batch, so BatchNorm uses running averages of $\mu$ and $\sigma^2$ accumulated during training — a detail that bites people who forget to switch to eval mode.

The payoffs are large: BatchNorm permits much higher learning rates (the normalization caps how far activations can drift), reduces sensitivity to initialization, and has a mild regularizing effect because each example's normalization depends on the random composition of its batch.

### Backprop through BatchNorm

BatchNorm is a genuinely instructive backward pass because the normalization couples every example in the batch — $\mu$ and $\sigma^2$ depend on all of them — so the gradient is not elementwise. Differentiating the transform and collecting terms gives the canonical result:

$$
\frac{\partial L}{\partial z_n}
= \frac{\gamma}{\sqrt{\sigma^2+\varepsilon}}\left(
\bar{\hat z}_n
- \frac{1}{N}\sum_{m}\bar{\hat z}_m
- \frac{\hat z_n}{N}\sum_{m}\bar{\hat z}_m \hat z_m
\right),
$$

where $\bar{\hat z}_n = \partial L/\partial \hat z_n = \gamma \cdot \partial L/\partial y_n$. The two subtracted terms are corrections for how each input influences the shared mean and variance. Here is a complete, gradient-checkable implementation.

```python
import numpy as np

class BatchNorm1d:
    """From-scratch BatchNorm over the batch dimension, with hand-written backward."""

    def __init__(self, dim, momentum=0.1, eps=1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps = eps
        self.momentum = momentum
        self.running_mean = np.zeros(dim)
        self.running_var = np.ones(dim)
        self.cache = None

    def forward(self, Z, training=True):
        if training:
            mu = Z.mean(axis=0)                       # (dim,)
            var = Z.var(axis=0)                       # population variance
            std_inv = 1.0 / np.sqrt(var + self.eps)
            Zhat = (Z - mu) * std_inv
            out = self.gamma * Zhat + self.beta
            # update running stats for inference
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            self.cache = (Zhat, std_inv, self.gamma)
            return out
        else:
            Zhat = (Z - self.running_mean) / np.sqrt(self.running_var + self.eps)
            return self.gamma * Zhat + self.beta

    def backward(self, dOut):
        Zhat, std_inv, gamma = self.cache
        N = dOut.shape[0]
        # gradients of the affine params
        self.dgamma = (dOut * Zhat).sum(axis=0)
        self.dbeta = dOut.sum(axis=0)
        # gradient through the normalization (the coupled part)
        dZhat = dOut * gamma                          # (N, dim)
        dZ = (std_inv / N) * (
            N * dZhat
            - dZhat.sum(axis=0)                       # mean correction
            - Zhat * (dZhat * Zhat).sum(axis=0)       # variance correction
        )
        return dZ

# quick gradient check against central differences
rng = np.random.default_rng(0)
bn = BatchNorm1d(4)
Z = rng.normal(size=(8, 4))
G = rng.normal(size=(8, 4))                          # arbitrary upstream grad
out = bn.forward(Z); loss = (out * G).sum()
dZ = bn.backward(G)
eps = 1e-6; num = np.zeros_like(Z)
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        o = Z[i, j]
        Z[i, j] = o + eps; lp = (BatchNorm1d(4).forward(Z) * G).sum()
        Z[i, j] = o - eps; lm = (BatchNorm1d(4).forward(Z) * G).sum()
        Z[i, j] = o; num[i, j] = (lp - lm) / (2 * eps)
print("BN grad rel-error:", np.abs(num - dZ).max() / (np.abs(num).max() + 1e-12))
```

### BatchNorm vs LayerNorm, and why LLMs use the latter

BatchNorm normalizes *across the batch* for each feature; **LayerNorm** normalizes *across the features* for each example. The distinction matters enormously for transformers. BatchNorm's batch-coupling is a liability for language models: sequence lengths vary, batches are heterogeneous, and at inference (especially autoregressive decode, one token at a time) there is effectively no batch to normalize over. LayerNorm and its simpler cousin RMSNorm are per-token and batch-independent, so they behave identically in training and inference. That is why essentially every modern LLM uses LayerNorm/RMSNorm, not BatchNorm — a design choice we unpack in [The Transformer Block: Norms, Residuals, MLPs & Activations](../02-transformer/06-transformer-block.html).

{{fig:batchnorm-vs-layernorm-axes}}

!!! tip "Practitioner tip"

    The single most useful debugging instrument for a from-scratch network is a **gradient check**, and the second is a **histogram of per-layer activation and gradient statistics**. If early-layer gradients are orders of magnitude smaller than late-layer ones, you have a vanishing-gradient problem (fix the init or add normalization/residuals). If activation stds drift toward 0 or blow up as you go deeper, your initialization variance is wrong. Print `A.std()` per layer before you reach for anything fancier.

---

## Interview Corner: Backprop on a Whiteboard

The "implement backprop" question is a rite of passage. Interviewers want to see that you can derive the layer gradients from the chain rule, get the matrix shapes right, and reason about cost — not that you have memorized PyTorch.

!!! interview "Interview Corner"

    **Q:** On a whiteboard, derive the backward pass for a two-layer MLP with ReLU and a softmax-cross-entropy loss, and state the shapes and compute cost.

    **A:** Forward, with batch $X\in\mathbb{R}^{N\times d}$:
    $Z_1 = XW_1+\mathbf b_1$, $A_1=\mathrm{ReLU}(Z_1)$, $Z_2=A_1W_2+\mathbf b_2$, $P=\mathrm{softmax}(Z_2)$, $L=-\frac1N\sum\log P_{\text{true}}$.

    Backward, right to left. The fused softmax-CE gradient seeds everything: $\bar Z_2 = (P-Y)/N$ with shape $(N,C)$. Then the two golden identities give $\bar W_2 = A_1^\top \bar Z_2$ (shape $d_h\times C$) and $\bar{\mathbf b}_2 = \mathbf 1^\top \bar Z_2$. Propagate to the first layer: $\bar A_1 = \bar Z_2 W_2^\top$ (shape $N\times d_h$), gate through ReLU with $\bar Z_1 = \bar A_1 \odot \mathbb 1[Z_1>0]$, then $\bar W_1 = X^\top \bar Z_1$ and $\bar{\mathbf b}_1=\mathbf 1^\top\bar Z_1$. Cost: the backward pass is two matmuls per layer (one for the weight grad, one for the input grad), so it is about **2× the FLOPs of the forward pass**, and it requires caching the forward activations ($A_1$, and the sign of $Z_1$). Two things interviewers probe: *why is the softmax-CE gradient just $P-Y$?* (the Jacobians of softmax and the log telescope), and *why gate through ReLU with the input sign and not the output?* (because $\phi'(Z_1)=\mathbb 1[Z_1>0]$ depends on the pre-activation). Mention that gradient checking with central differences is how you would verify the implementation.

A second favorite: *why does backprop cost about the same as the forward pass rather than scaling with the number of parameters?* Because reverse-mode autodiff computes the gradient of one scalar output with respect to *all* inputs in a single sweep — each edge of the computation graph is traversed once forward and once backward. Contrast with forward-mode (and naive finite differences), which would need one pass *per parameter*. For a network with $P$ parameters and one scalar loss, reverse mode is the obvious choice; that asymmetry is the deepest reason every framework defaults to it, as we detail in [Automatic Differentiation & PyTorch Internals](../01-foundations/07-autodiff-pytorch.html).

---

## Key Takeaways

!!! key "Key Takeaways"

    - An MLP is a chain of affine maps and elementwise nonlinearities; the nonlinearity is non-negotiable, because composing linear maps just yields another linear map. Depth buys compositional efficiency at the price of a harder optimization landscape.
    - Backprop is the chain rule evaluated right-to-left, carrying an adjoint (upstream gradient) and applying a vector-Jacobian product at each node. Reverse-mode autodiff computes all parameter gradients in one backward sweep at ~2× forward cost.
    - The two identities you must own: for $Z=XW+\mathbf b$, $\bar W = X^\top\bar Z$, $\bar X = \bar Z W^\top$, $\bar{\mathbf b}=\mathbf 1^\top\bar Z$. Every dense-layer gradient is one of these.
    - The softmax + cross-entropy gradient collapses to $P-Y$ ("predicted minus true"); always fuse the two for correctness and numerical stability.
    - Gradients **accumulate**, so zero them before each backward pass — and always gradient-check a hand-written backward against central differences before trusting it.
    - Initialization is derived, not tuned: Xavier $\operatorname{Var}(W)=\frac{2}{d_{\text{in}}+d_{\text{out}}}$ for tanh-like activations, Kaiming $\operatorname{Var}(W)=\frac{2}{d_{\text{in}}}$ for ReLU. Get the variance wrong and signal vanishes or explodes geometrically with depth.
    - Vanishing/exploding gradients come from multiplying many Jacobian factors; the cures are good init, non-saturating activations (ReLU), normalization, residual connections, and gradient clipping.
    - BatchNorm normalizes across the batch and couples examples in its backward pass; LayerNorm/RMSNorm normalize per token and are batch-independent, which is why LLMs use them.

---

!!! sota "State of the Art & Resources (2026)"
    MLPs and backpropagation are a half-century-old foundation: the algorithms are settled, but the engineering art of initializing, normalizing, and stabilizing deep networks continues to evolve with every new architecture. Today's billion-parameter transformers are still built from the same forward-pass/backward-pass loop described in this chapter, extended by residual connections, layer normalization, and sophisticated optimizers.

    **Foundational work**

    - [Rumelhart, Hinton & Williams, *Learning Representations by Back-Propagating Errors* (1986)](https://www.nature.com/articles/323533a0) — The paper that put backpropagation on the map; derivation of the chain-rule training algorithm still used today.
    - [Glorot & Bengio, *Understanding the Difficulty of Training Deep Feedforward Neural Networks* (2010)](https://proceedings.mlr.press/v9/glorot10a.html) — Introduced Xavier initialization via forward/backward variance analysis; diagnosed why sigmoid saturates deep nets.
    - [He et al., *Delving Deep into Rectifiers* (2015)](https://arxiv.org/abs/1502.01852) — Derived the factor-of-2 Kaiming correction for ReLU networks and introduced PReLU.
    - [Ioffe & Szegedy, *Batch Normalization* (2015)](https://arxiv.org/abs/1502.03167) — Showed that normalizing pre-activations per mini-batch allows much higher learning rates and reduces sensitivity to initialization.

    **Open-source & tools**

    - [karpathy/micrograd](https://github.com/karpathy/micrograd) — ~150-line scalar autograd engine; the direct inspiration for the `Value` class in this chapter.
    - [karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) — Full lecture series building from micrograd up to GPT, with notebooks for every video.
    - [PyTorch — Build the Neural Network (official tutorial)](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html) — The canonical first step to moving from NumPy to production-grade autograd and `nn.Module`.

    **Go deeper**

    - [Karpathy, *The Spelled-Out Intro to Neural Networks and Backpropagation: Building Micrograd* (YouTube)](https://www.youtube.com/watch?v=VMj-3S1tku0) — 2.5-hour step-by-step walkthrough; best video resource for internalizing the chain rule.
    - [Goodfellow, Bengio & Courville, *Deep Learning* (2016) — Chapters 6 & 8](https://www.deeplearningbook.org/) — Canonical textbook treatment of MLPs, backprop, and optimization difficulties; free online.
    - [Stanford CS231n — Backpropagation Notes](https://cs231n.github.io/optimization-2/) — Excellent circuit-diagram intuition for how gradients flow through computational graphs.

## Further Reading

- **Rumelhart, Hinton & Williams, *Learning Representations by Back-Propagating Errors* (1986)** — The paper that put backpropagation on the map; short and worth reading in the original.
- **Andrej Karpathy, *micrograd* (GitHub) and *The spelled-out intro to neural networks and backpropagation*** — A ~150-line scalar autograd engine and an exceptional lecture; the direct inspiration for the `Value` class in this chapter.
- **Glorot & Bengio, *Understanding the Difficulty of Training Deep Feedforward Neural Networks* (2010)** — The Xavier initialization derivation via forward/backward variance preservation.
- **He, Zhang, Ren & Sun, *Delving Deep into Rectifiers* (2015)** — Kaiming initialization for ReLU networks, with the factor-of-2 correction.
- **Ioffe & Szegedy, *Batch Normalization* (2015)** — The original BatchNorm paper, including the forward transform and the backward derivation implemented above.
- **Goodfellow, Bengio & Courville, *Deep Learning* (2016), Chapters 6 and 8** — The canonical textbook treatment of feed-forward networks, backprop, and optimization difficulties.
- **Nielsen, *Neural Networks and Deep Learning* (online book)** — A gentle, code-first derivation of backprop that complements the matrix-form presentation here.

---

## Exercises

**1.** (Conceptual) The chapter insists the activation function $\phi$ is "non-negotiable." Suppose you build a 3-layer MLP but set every activation to the identity map, $\phi(z)=z$. Show algebraically that the whole network collapses to a single affine map, and explain in one sentence what this means for the function class the network can represent.

??? note "Solution"

    Write the three layers with identity activations:

    $$
    \mathbf{a}^{(1)} = W^{(1)}\mathbf{x} + \mathbf{b}^{(1)}, \quad
    \mathbf{a}^{(2)} = W^{(2)}\mathbf{a}^{(1)} + \mathbf{b}^{(2)}, \quad
    \mathbf{a}^{(3)} = W^{(3)}\mathbf{a}^{(2)} + \mathbf{b}^{(3)}.
    $$

    Substitute inward, from the last layer to the first:

    $$
    \mathbf{a}^{(3)} = W^{(3)}\!\left(W^{(2)}\!\left(W^{(1)}\mathbf{x}+\mathbf{b}^{(1)}\right)+\mathbf{b}^{(2)}\right)+\mathbf{b}^{(3)}.
    $$

    Distribute the matrix products:

    $$
    \mathbf{a}^{(3)} = \underbrace{W^{(3)}W^{(2)}W^{(1)}}_{\tilde W}\,\mathbf{x}
    \;+\; \underbrace{W^{(3)}W^{(2)}\mathbf{b}^{(1)} + W^{(3)}\mathbf{b}^{(2)} + \mathbf{b}^{(3)}}_{\tilde{\mathbf b}}.
    $$

    This is exactly one affine map $\mathbf{a}^{(3)} = \tilde W \mathbf{x} + \tilde{\mathbf b}$ with a single effective weight matrix $\tilde W$ and bias $\tilde{\mathbf b}$. Because a product of linear maps is linear, the three layers buy no extra expressive power: the network can represent nothing beyond a single linear (affine) function, no matter how many such layers you stack. The nonlinearity is what lets depth express functions a single matrix cannot.

**2.** (Quantitative) Trace the scalar autograd engine by hand. Let $x=2.0$, $w=3.0$, $b=1.0$, and $\text{target}=4.0$. Compute $z = x\cdot w + b$, $a = \mathrm{relu}(z)$, and $\text{loss} = (a-\text{target})^2$. Then work the backward pass by hand to find $\dfrac{\partial L}{\partial w}$, $\dfrac{\partial L}{\partial x}$, and $\dfrac{\partial L}{\partial b}$. (Note the sign of $w$ is flipped from the chapter's dead-ReLU example, so the gate is now open.)

??? note "Solution"

    Forward pass:

    $$
    z = x w + b = (2.0)(3.0) + 1.0 = 7.0, \qquad
    a = \mathrm{relu}(7.0) = 7.0, \qquad
    L = (a - 4.0)^2 = (3.0)^2 = 9.0.
    $$

    Backward pass, right to left, using the local VJP rules from the `Value` class. Seed $\bar L = \partial L/\partial L = 1$.

    Through the square, $L = (a-t)^2$ with $t=4.0$:

    $$
    \bar a = \frac{\partial L}{\partial a} = 2(a - t)\cdot \bar L = 2(7.0 - 4.0) = 6.0.
    $$

    Through the ReLU, whose gate is open because $z = 7.0 > 0$ (derivative $=1$):

    $$
    \bar z = \bar a \cdot \mathbb{1}[z>0] = 6.0 \cdot 1 = 6.0.
    $$

    Through $z = xw + b$. The bias adds, so it passes the gradient through; the multiply sends the *other* operand times the upstream:

    $$
    \frac{\partial L}{\partial b} = \bar z \cdot 1 = 6.0,
    $$
    $$
    \frac{\partial L}{\partial w} = \bar z \cdot x = 6.0 \cdot 2.0 = 12.0,
    $$
    $$
    \frac{\partial L}{\partial x} = \bar z \cdot w = 6.0 \cdot 3.0 = 18.0.
    $$

    So $\partial L/\partial w = 12.0$, $\partial L/\partial x = 18.0$, $\partial L/\partial b = 6.0$ — all nonzero, because the ReLU gate is open. (Contrast the chapter's example with $w=-3.0$, where $z<0$ shut the gate and every gradient was zero.)

**3.** (Quantitative) Use the variance-propagation rule to compare initializations for a ReLU layer. A hidden layer has $d_{\text{in}} = 512$ inputs and receives an input activation with per-component variance $\operatorname{Var}(x) = 1$. Recall that a ReLU layer preserves the forward standard deviation when weights are drawn with $\operatorname{Var}(W) = 2/d_{\text{in}}$. (a) What weight standard deviation does Kaiming init prescribe here? (b) If instead you (wrongly) used $\operatorname{Var}(W) = 1/d_{\text{in}}$ (Xavier-style, ignoring the ReLU factor of 2), by what factor does the activation *standard deviation* shrink per layer? (c) After 20 such layers, what is the activation std, starting from 1?

??? note "Solution"

    (a) Kaiming variance is $\operatorname{Var}(W) = 2/d_{\text{in}} = 2/512 = 1/256$. The standard deviation is

    $$
    \sqrt{2/512} = \sqrt{0.00390625} = 0.0625.
    $$

    (b) The pre-activation variance is $\operatorname{Var}(z) = d_{\text{in}}\cdot\operatorname{Var}(W)\cdot\operatorname{Var}(x)$, and the ReLU zeroes half its inputs in expectation, which halves the variance that survives to the next layer. So the per-layer variance multiplier is

    $$
    \tfrac{1}{2}\, d_{\text{in}}\,\operatorname{Var}(W)
    = \tfrac{1}{2}\cdot 512 \cdot \frac{1}{512} = \tfrac{1}{2}.
    $$

    The standard deviation multiplier is the square root of that:

    $$
    \sqrt{\tfrac{1}{2}} \approx 0.7071.
    $$

    So each layer shrinks the activation std by a factor of about $0.7071$ (a $\sim 29\%$ reduction). This is exactly why Xavier's $1/d_{\text{in}}$ under-scales ReLU nets and Kaiming restores the missing factor of 2: with $\operatorname{Var}(W)=2/d_{\text{in}}$ the multiplier becomes $\tfrac12\cdot 512\cdot\tfrac{2}{512}=1$, and std is preserved.

    (c) Starting from std $=1$, after 20 layers:

    $$
    (0.7071)^{20} = \left(\tfrac{1}{2}\right)^{10} = \frac{1}{1024} \approx 9.8\times 10^{-4}.
    $$

    The signal has collapsed by three orders of magnitude in only 20 layers — a concrete instance of the vanishing-signal (and, on the backward pass, vanishing-gradient) pathology the chapter warns about.

**4.** (Implementation) The chapter's `MLP` uses plain ReLU, which can suffer dead units. Implement Leaky ReLU (with slope $\alpha$ for negative inputs) and its derivative in the chapter's NumPy style, matching the signatures of `relu` / `relu_grad`. Then state the one-line change needed inside `MLP.forward` and `MLP.loss_and_grads` to use them, and explain in one sentence why this keeps dead units alive.

??? note "Solution"

    Leaky ReLU is $\phi(z)=\max(\alpha z, z)$, i.e. $z$ for $z>0$ and $\alpha z$ for $z\le 0$; its derivative is $1$ where $z>0$ and $\alpha$ where $z\le 0$. In the chapter's vectorized style:

    ```python
    import numpy as np

    def leaky_relu(z, alpha=0.01):
        # z where z > 0, else alpha * z  (vectorized, no Python loop)
        return np.where(z > 0.0, z, alpha * z)

    def leaky_relu_grad(z, alpha=0.01):
        # 1 where z > 0, else alpha
        return np.where(z > 0.0, 1.0, alpha).astype(z.dtype)
    ```

    Usage inside the `MLP`: in `forward`, the hidden-layer branch changes from

    ```python
        if i < L - 1:
            A = relu(Z)
    ```

    to `A = leaky_relu(Z)`; and in `loss_and_grads`, the gate line changes from

    ```python
        dZ = dA_prev * relu_grad(cache["Z"][i - 1])
    ```

    to `dZ = dA_prev * leaky_relu_grad(cache["Z"][i - 1])`. (The two must match: the forward activation and the derivative used to gate the backward gradient have to be the same function.)

    Why it keeps units alive: for $z\le 0$ the derivative is $\alpha \neq 0$ rather than $0$, so a trickle of gradient always flows back into a negative-pre-activation neuron, letting it update and potentially climb back into the active region — whereas plain ReLU's zero derivative there freezes a dead unit forever.

**5.** (Implementation + verification) A colleague hand-writes the backward pass for the output layer of the chapter's `MLP` but writes the fused softmax-cross-entropy seed as

    ```python
    dZ = P.copy()
    dZ[np.arange(N), y] -= 1.0
    # (forgot to divide by N)
    ```

    omitting the `dZ /= N`. Explain what this bug does to the gradients, and describe precisely what the `gradient_check` function from the chapter would report (a number, roughly) that reveals the bug. Then write the corrected two lines.

??? note "Solution"

    The chapter's loss is the *mean* cross-entropy over the batch, $L = \frac1N\sum_n L_n$. The correct fused seed is therefore $\bar Z = (P - Y)/N$; the $1/N$ comes from differentiating the mean. Omitting `dZ /= N` seeds $\bar Z = (P-Y)$ instead, so every downstream gradient — $\bar W = A^\top \bar Z$, $\bar{\mathbf b} = \mathbf 1^\top \bar Z$, and everything propagated to earlier layers — is too large by exactly a factor of $N$ (the batch size). Training would effectively use a learning rate $N$ times bigger than intended and likely diverge for any $N$ of realistic size.

    `gradient_check` compares the analytic gradient against the central-difference numerical gradient of the *same* loss (the mean CE), computing the relative error

    $$
    \text{rel} = \frac{|\text{num} - \text{ana}|}{|\text{num}| + |\text{ana}| + 10^{-12}}.
    $$

    Here $\text{ana} = N\cdot\text{num}$, so

    $$
    \text{rel} = \frac{|N\cdot\text{num} - \text{num}|}{|N\cdot\text{num}| + |\text{num}|}
    = \frac{N-1}{N+1}.
    $$

    For the chapter's demo batch $N = 512$ this is $511/513 \approx 0.996$ — a relative error near $1.0$, glaringly far from the $\sim 10^{-7}$ a correct backward pass produces. That huge relative error is exactly the signal that the analytic gradient is off by a constant factor. The fix restores the mean's $1/N$:

    ```python
    dZ = P.copy()
    dZ[np.arange(N), y] -= 1.0
    dZ /= N        # <- mean loss => gradients scaled by 1/N
    ```
