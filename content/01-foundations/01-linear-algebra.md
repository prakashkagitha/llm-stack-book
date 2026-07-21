# 1.1 Linear Algebra for Deep Learning

Linear algebra is the language deep learning is written in. Every forward pass through a neural network is a cascade of matrix multiplications. Every gradient update is a matrix-calculus identity. Every parameter-efficient fine-tuning technique — from LoRA to Adapters — exploits the geometric structure of high-dimensional weight matrices. If you skip this chapter, the rest of the book will feel like reading music without knowing what notes are.

This chapter is both a refresher and a precision tool. We assume you have met vectors and matrices before; our job is to make the machinery feel mechanical and concrete, show you exactly which operations dominate in practice, and build the intuition for *why* low-rank structure shows up everywhere in modern LLMs. We will derive, not just state. We will compute, not just describe.

Cross-references: [Calculus, Optimization & Convexity](../01-foundations/03-calculus-optimization.html) picks up matrix calculus in more detail; [Neural Networks From Scratch: MLPs & Backprop](../01-foundations/06-neural-nets-from-scratch.html) applies everything here to multi-layer networks; [Automatic Differentiation & PyTorch Internals](../01-foundations/07-autodiff-pytorch.html) shows how PyTorch tracks these gradients automatically; [PEFT I: LoRA, QLoRA, DoRA & The Adapter Family](../05-posttraining-alignment/03-peft-lora-qlora.html) builds directly on the SVD and low-rank ideas developed below.

---

## Vectors, Matrices, and Tensors

### Vectors

A **vector** $\mathbf{v} \in \mathbb{R}^n$ is an ordered list of $n$ real numbers. Geometrically it is an arrow from the origin; algebraically it is a column by convention:

$$
\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}
$$

The **dot product** (inner product) of two vectors is $\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^n u_i v_i = \mathbf{u}^\top \mathbf{v}$. Its geometric interpretation: $\mathbf{u}^\top \mathbf{v} = \|\mathbf{u}\| \|\mathbf{v}\| \cos\theta$, where $\theta$ is the angle between them. This identity underlies cosine similarity, which is the dominant similarity metric for embedding vectors in retrieval and attention.

### Matrices

A **matrix** $A \in \mathbb{R}^{m \times n}$ has $m$ rows and $n$ columns. It is simultaneously:

1. A rectangular array of numbers $A_{ij}$.
2. A linear map from $\mathbb{R}^n$ to $\mathbb{R}^m$: $\mathbf{y} = A\mathbf{x}$.
3. A collection of $n$ column vectors, each in $\mathbb{R}^m$.

The **transpose** $A^\top$ has rows and columns swapped: $(A^\top)_{ij} = A_{ji}$.

A **symmetric** matrix satisfies $A = A^\top$; positive semi-definite (PSD) means $\mathbf{x}^\top A \mathbf{x} \geq 0$ for all $\mathbf{x}$. Covariance matrices and Gram matrices are always PSD.

### Tensors

In deep learning, **tensor** typically means a multi-dimensional array with a fixed shape. A 3-D tensor $T \in \mathbb{R}^{B \times S \times D}$ might represent a batch of $B$ sequences of length $S$, each token encoded as a $D$-dimensional vector. PyTorch `torch.Tensor` is the workhorse; NumPy `ndarray` is the CPU equivalent.

```python
import numpy as np
import torch

# ----- Vectors -----
v = torch.tensor([1.0, 2.0, 3.0])          # shape (3,)
u = torch.tensor([4.0, 5.0, 6.0])

dot = torch.dot(v, u)                        # 1*4 + 2*5 + 3*6 = 32.0
cosine_sim = dot / (v.norm() * u.norm())     # ≈ 0.9746

print(f"dot={dot.item():.1f}, cos_sim={cosine_sim.item():.4f}")

# ----- Matrices -----
A = torch.randn(4, 3)   # 4×3 matrix
x = torch.randn(3)      # 3-vector
y = A @ x               # matrix-vector product, shape (4,)

# ----- 3-D Tensor (batch of sequences) -----
B, S, D = 2, 8, 512
hidden_states = torch.randn(B, S, D)        # typical transformer hidden states

# Batch matrix multiply across the batch dimension
Q = torch.randn(B, S, 64)   # queries
K = torch.randn(B, S, 64)   # keys
# Compute all pairwise dot products: (B, S, S)
scores = torch.bmm(Q, K.transpose(1, 2))    # batch matmul
print(f"Attention score tensor shape: {scores.shape}")  # (2, 8, 8)
```

---

## Matrix Multiplication: The Workhorse Operation

### Three equivalent views

Given $A \in \mathbb{R}^{m \times k}$ and $B \in \mathbb{R}^{k \times n}$, the product $C = AB \in \mathbb{R}^{m \times n}$ has:

$$
C_{ij} = \sum_{p=1}^{k} A_{ip} B_{pj}
$$

**View 1 — Dot products.** Entry $C_{ij}$ is the dot product of the $i$-th row of $A$ and $j$-th column of $B$.

**View 2 — Column combinations.** The $j$-th column of $C$ is $A$ times the $j$-th column of $B$: $C_{:,j} = A \cdot B_{:,j}$. So $C$ expresses each column of $B$ as a linear combination of $A$'s columns.

**View 3 — Outer products (rank-1 decomposition).** $C = \sum_{p=1}^k A_{:,p} \cdot B_{p,:}^\top$. Each term is a rank-1 matrix (column times row). This view is surprisingly important: low-rank approximations are truncated versions of this sum.

{{fig:matmul-three-views}}

### FLOP count and why it matters

Multiplying an $(m \times k)$ matrix by a $(k \times n)$ matrix costs $2mkn$ FLOPs (multiply-accumulate, factor of 2). For a transformer layer with hidden dim $d = 4096$:

- A single linear projection $W \in \mathbb{R}^{4096 \times 4096}$ applied to a batch $X \in \mathbb{R}^{B \times S \times 4096}$ costs $2 \cdot B \cdot S \cdot 4096^2 \approx 34 \times 10^9 \cdot B \cdot S$ FLOPs.
- At batch size 1 and sequence length 512, that is roughly 17 GFLOPs per linear layer.

Modern A100 GPUs deliver on the order of 312 TFLOPS in BF16. So in theory you can run hundreds of linear layers per second — but memory bandwidth is often the bottleneck, not raw FLOPs. See [The Roofline Model & Performance Engineering](../04-kernels-efficiency/01-roofline-performance.html) for the full story.

### Matrix multiplication in NumPy and PyTorch

```python
import torch
import time

# Benchmark matmul on CPU vs GPU
A = torch.randn(4096, 4096)
B = torch.randn(4096, 4096)

# CPU
t0 = time.perf_counter()
C_cpu = A @ B
t1 = time.perf_counter()
print(f"CPU matmul 4096×4096: {(t1-t0)*1000:.1f} ms")

# GPU (if available)
if torch.cuda.is_available():
    A_gpu = A.cuda().to(torch.bfloat16)
    B_gpu = B.cuda().to(torch.bfloat16)
    # Warm up
    _ = A_gpu @ B_gpu
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    C_gpu = A_gpu @ B_gpu
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"GPU matmul 4096×4096 (BF16): {(t1-t0)*1000:.2f} ms")

# Batched matmul — critical for transformer attention
# Simulate multi-head attention: 32 heads, seq_len=512, head_dim=128
H, S, d = 32, 512, 128
Q = torch.randn(H, S, d)
K = torch.randn(H, S, d)
# Q @ K^T -> (H, S, S) — scores for all heads at once
scores = Q @ K.transpose(-2, -1)  # uses broadcasting/batched matmul
print(f"Score shape: {scores.shape}")  # (32, 512, 512)
```

---

## Rank, Span, Basis, and Null Space

### The column space and span

The **column space** (image) of $A \in \mathbb{R}^{m \times n}$ is $\text{col}(A) = \{ A\mathbf{x} : \mathbf{x} \in \mathbb{R}^n \}$, the set of all linear combinations of $A$'s columns. This lives in $\mathbb{R}^m$.

The **rank** of $A$ is the dimension of its column space: $\text{rank}(A) = \dim(\text{col}(A))$. Key facts:
- $\text{rank}(A) \leq \min(m, n)$. When equality holds, $A$ is **full rank**.
- $\text{rank}(A) = \text{rank}(A^\top)$ — row rank equals column rank.
- $\text{rank}(AB) \leq \min(\text{rank}(A), \text{rank}(B))$.

The **null space** (kernel) of $A$ is $\ker(A) = \{\mathbf{x} : A\mathbf{x} = \mathbf{0}\}$. The **rank-nullity theorem** states:

$$
\text{rank}(A) + \text{nullity}(A) = n
$$

### Basis and linear independence

A set of vectors $\{\mathbf{b}_1, \ldots, \mathbf{b}_k\}$ is **linearly independent** if no vector is a linear combination of the others. A **basis** for a subspace is a linearly independent set that spans the subspace. The **standard basis** in $\mathbb{R}^n$ is $\{\mathbf{e}_1, \ldots, \mathbf{e}_n\}$ where $\mathbf{e}_i$ has a 1 in position $i$ and 0 elsewhere.

Why do ML engineers care? Because the rank of a weight matrix tells you about its **effective dimensionality** — how many truly independent directions the linear map uses. Empirically, the weight updates $\Delta W$ during fine-tuning tend to concentrate in a very low-dimensional subspace. This is the geometric intuition behind LoRA (see [PEFT I: LoRA, QLoRA, DoRA & The Adapter Family](../05-posttraining-alignment/03-peft-lora-qlora.html)).

```python
import numpy as np

# Check rank empirically
W = np.random.randn(768, 768)           # full-rank weight matrix
rank_full = np.linalg.matrix_rank(W)
print(f"Random 768×768 rank: {rank_full}")  # should be 768

# Low-rank matrix: W = A @ B where A is 768×4 and B is 4×768
r = 4
A_lr = np.random.randn(768, r)
B_lr = np.random.randn(r, 768)
W_lr = A_lr @ B_lr                      # rank at most 4
rank_lr = np.linalg.matrix_rank(W_lr)
print(f"Low-rank 768×768 (r=4) rank: {rank_lr}")  # 4

# Memory comparison
params_full = 768 * 768          # 589,824
params_lora = 768 * r + r * 768  # 6,144  (99% reduction!)
print(f"Full: {params_full:,} params, LoRA r=4: {params_lora:,} params")
```

---

## Eigendecomposition and the Singular Value Decomposition

### Eigenvalues and eigenvectors

For a **square** matrix $A \in \mathbb{R}^{n \times n}$, a vector $\mathbf{v} \neq \mathbf{0}$ is an **eigenvector** with **eigenvalue** $\lambda$ if:

$$
A\mathbf{v} = \lambda \mathbf{v}
$$

Geometrically, $A$ does not rotate $\mathbf{v}$ — it only stretches it by $\lambda$. If $A$ is symmetric, all eigenvalues are real and eigenvectors for distinct eigenvalues are orthogonal. The **eigendecomposition** is:

$$
A = Q \Lambda Q^\top
$$

where $Q$ is orthogonal (its columns are eigenvectors) and $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$.

Eigenvalues govern stability in optimization: the **condition number** $\kappa(A) = \lambda_{\max} / \lambda_{\min}$ measures how much the problem is ill-conditioned. Large condition numbers slow down gradient descent; this is part of why adaptive optimizers (Adam et al.) help — see [Optimizers: SGD, Adam, Adafactor, Lion, Muon & Shampoo](../03-pretraining/09-optimizers.html).

### The Singular Value Decomposition

The SVD works for **any** matrix $A \in \mathbb{R}^{m \times n}$, not just square symmetric ones:

$$
A = U \Sigma V^\top
$$

- $U \in \mathbb{R}^{m \times m}$: orthogonal, columns are **left singular vectors**.
- $\Sigma \in \mathbb{R}^{m \times n}$: diagonal with non-negative **singular values** $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$.
- $V \in \mathbb{R}^{n \times n}$: orthogonal, columns are **right singular vectors**.

The **economy (thin) SVD** keeps only the $r = \text{rank}(A)$ non-zero singular values, giving $U \in \mathbb{R}^{m \times r}$, $\Sigma \in \mathbb{R}^{r \times r}$, $V \in \mathbb{R}^{n \times r}$.

### Best low-rank approximation: the Eckart-Young theorem

{{fig:svd-geometry-and-lowrank}}

**Theorem (Eckart-Young, 1936).** Among all rank-$k$ matrices, the one closest to $A$ in Frobenius norm is:

$$
A_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^\top
$$

The approximation error is $\|A - A_k\|_F^2 = \sum_{i=k+1}^r \sigma_i^2$. This is the theoretical foundation for LoRA: if the weight update $\Delta W$ has a rapidly decaying singular value spectrum, keeping only the top-$r$ components captures most of the information with far fewer parameters.

```python
import torch
import matplotlib.pyplot as plt

# Demonstrate SVD and low-rank approximation
torch.manual_seed(42)
# Construct a matrix with known low-rank structure + noise
m, n, true_rank = 256, 256, 8
U_true = torch.randn(m, true_rank)
V_true = torch.randn(n, true_rank)
W = U_true @ V_true.T + 0.1 * torch.randn(m, n)  # rank-8 + noise

# Full SVD
U, S, Vh = torch.linalg.svd(W, full_matrices=False)
# S is shape (min(m,n),), Vh is shape (min(m,n), n)

print("Singular values (first 12):")
print(S[:12].numpy().round(2))
# Expected: 8 large values, then a cliff down to ~0.1 (noise floor)

# Low-rank approximation at rank r
def low_rank_approx(U, S, Vh, r):
    """Reconstruct W using only the top-r singular components."""
    return (U[:, :r] * S[:r]) @ Vh[:r, :]

# Compare reconstruction errors
for r in [1, 4, 8, 16, 32]:
    W_r = low_rank_approx(U, S, Vh, r)
    rel_err = (W - W_r).norm() / W.norm()
    n_params_full = m * n           # 65,536
    n_params_lr   = r * (m + n)     # e.g., 8*(256+256) = 4,096
    print(f"rank-{r:2d}: rel_err={rel_err:.4f}, "
          f"params={n_params_lr:,} vs {n_params_full:,}")
```

!!! example "Worked example: SVD on a weight matrix"
    Let $W \in \mathbb{R}^{256 \times 256}$ be a rank-8 matrix corrupted by Gaussian noise $\sigma = 0.1$. Running the code above yields singular values approximately:

    ```text
    [12.8, 11.4, 10.9, 10.3, 9.7, 9.1, 8.8, 8.3, 0.1, 0.1, ...]
    ```

    A rank-8 approximation achieves relative error of roughly 0.03 (3%), using only $8 \times (256 + 256) = 4{,}096$ parameters instead of $256^2 = 65{,}536$ — a 16× compression. A rank-16 approximation barely improves (the extra singular values are all noise), confirming that the information-carrying subspace truly has low dimension.

    In LoRA, we represent $\Delta W \approx AB$ where $A \in \mathbb{R}^{d \times r}$ and $B \in \mathbb{R}^{r \times d}$. If $\Delta W$ has a similar fast-decaying spectrum (which empirical studies confirm it does), rank $r \in \{4, 8, 16\}$ captures most fine-tuning signal with a fraction of the parameters.

---

## Norms: Measuring Size and Distance

### Vector norms

The **$\ell_p$ norm** of $\mathbf{v} \in \mathbb{R}^n$ is:

$$
\|\mathbf{v}\|_p = \left(\sum_{i=1}^n |v_i|^p\right)^{1/p}
$$

The three most important instances:

| Norm | Formula | Geometry | Use in ML |
|------|---------|----------|-----------|
| $\ell_1$ | $\sum_i |v_i|$ | Sum of absolute values | Sparsity-inducing regularization (Lasso) |
| $\ell_2$ | $\sqrt{\sum_i v_i^2}$ | Euclidean length | Weight decay, gradient clipping |
| $\ell_\infty$ | $\max_i |v_i|$ | Maximum absolute entry | Adversarial robustness |

**Gradient clipping** in transformer training (used universally) clips $\mathbf{g} \leftarrow \mathbf{g} \cdot \min(1, \theta / \|\mathbf{g}\|_2)$ where $\theta$ is typically $1.0$. See [Training Stability, Loss Spikes & Debugging Large Runs](../03-pretraining/11-training-stability.html).

### Matrix norms

Three matrix norms appear repeatedly:

**Frobenius norm**: $\|A\|_F = \sqrt{\sum_{i,j} A_{ij}^2} = \sqrt{\text{tr}(A^\top A)} = \sqrt{\sum_i \sigma_i^2}$. This is the natural $\ell_2$ norm on matrices viewed as flattened vectors.

**Spectral norm**: $\|A\|_2 = \sigma_{\max}(A)$, the largest singular value. This is the operator norm — it measures the maximum amount $A$ can stretch a unit vector. Spectral normalization in GANs and Lipschitz regularization in transformers use this.

**Nuclear norm**: $\|A\|_* = \sum_i \sigma_i$. The convex relaxation of rank; minimizing it promotes low-rank solutions.

```python
import torch

A = torch.randn(128, 256)

# Frobenius norm
frob = torch.linalg.norm(A, ord='fro')
# Equivalent: torch.sqrt((A**2).sum())

# Spectral norm (largest singular value)
S = torch.linalg.svdvals(A)          # sorted descending
spectral = S[0]

# Nuclear norm (sum of singular values)
nuclear = S.sum()

print(f"Frobenius: {frob:.2f}, Spectral: {spectral:.2f}, Nuclear: {nuclear:.2f}")

# Weight decay uses Frobenius; AdamW adds lambda * W to the gradient
# Spectral norm: PyTorch has torch.nn.utils.spectral_norm for Conv/Linear
```

---

## Matrix Calculus for Backpropagation

This is where linear algebra meets gradient descent. Understanding matrix calculus identities is essential for implementing or debugging backprop by hand, and it is a common interview topic.

### Jacobians and gradients

For a scalar function $f: \mathbb{R}^n \to \mathbb{R}$, the **gradient** $\nabla_{\mathbf{x}} f \in \mathbb{R}^n$ has entries $(\nabla_{\mathbf{x}} f)_i = \partial f / \partial x_i$. We adopt the **numerator layout** convention (a row Jacobian becomes the transpose of the gradient column vector).

For a vector function $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$, the **Jacobian** $J \in \mathbb{R}^{m \times n}$ has $J_{ij} = \partial f_i / \partial x_j$.

### Key identities

Let $A$ be a constant matrix and $\mathbf{x}$ a vector of variables.

$$
\frac{\partial}{\partial \mathbf{x}} (A\mathbf{x}) = A, \qquad \frac{\partial}{\partial \mathbf{x}} (\mathbf{x}^\top A \mathbf{x}) = (A + A^\top)\mathbf{x}
$$

For a scalar loss $L$, if $\mathbf{y} = A\mathbf{x}$, the chain rule gives:

$$
\frac{\partial L}{\partial \mathbf{x}} = A^\top \frac{\partial L}{\partial \mathbf{y}}
$$

This is the **transpose rule** for linear maps: the backward pass through a matrix multiply uses the transposed matrix. This is the single most important identity in deep learning backpropagation.

### Gradient of a linear layer

A linear layer computes $Y = XW$ where $X \in \mathbb{R}^{B \times D_{\text{in}}}$, $W \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$, $Y \in \mathbb{R}^{B \times D_{\text{out}}}$. Given $\partial L / \partial Y$, what are $\partial L / \partial W$ and $\partial L / \partial X$?

Think of each row of $X$ as an input vector and each row of $Y$ as the corresponding output. Differentiating entry-by-entry and then collecting:

$$
\frac{\partial L}{\partial W} = X^\top \frac{\partial L}{\partial Y}, \qquad \frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} W^\top
$$

{{fig:linear-layer-backprop-transpose}}

This is the most important formula to have on the tip of your tongue. It says: to compute how the loss changes with respect to the weights, multiply the transposed input by the upstream gradient; to backprop to the inputs, multiply the upstream gradient by the transposed weight.

!!! interview "Interview Corner"
    **Q:** Derive the gradients of $L = f(XW)$ with respect to $W$ and $X$, where $X \in \mathbb{R}^{B \times D_{\text{in}}}$, $W \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$, and $f$ is any differentiable scalar loss on the output.

    **A:** Let $G = \partial L / \partial Y \in \mathbb{R}^{B \times D_{\text{out}}}$ be the upstream gradient. By the chain rule on the bilinear map $Y = XW$:

    - For weight gradient: fixing $X$ constant and varying $W$ by $\delta W$, we get $\delta Y = X \delta W$, so $\delta L = \text{tr}(G^\top X \delta W) = \text{tr}((X^\top G)^\top \delta W)$, giving $\partial L / \partial W = X^\top G$.
    - For input gradient: fixing $W$ constant and varying $X$ by $\delta X$, we get $\delta Y = \delta X \cdot W$, so $\delta L = \text{tr}(G^\top \delta X W) = \text{tr}((G W^\top)^\top \delta X)$, giving $\partial L / \partial X = G W^\top$.

    The shapes confirm the formulas: $X^\top G$ is $(D_{\text{in}} \times B)(B \times D_{\text{out}}) = D_{\text{in}} \times D_{\text{out}}$ matching $W$; $G W^\top$ is $(B \times D_{\text{out}})(D_{\text{out}} \times D_{\text{in}}) = B \times D_{\text{in}}$ matching $X$. Verify in PyTorch by comparing `.grad` to manual computation.

### Manual backprop verification

```python
import torch

torch.manual_seed(0)
B, D_in, D_out = 4, 8, 6

# Create inputs and weights with requires_grad
X = torch.randn(B, D_in, requires_grad=True)
W = torch.randn(D_in, D_out, requires_grad=True)

# Forward
Y = X @ W                   # shape (B, D_out)
L = Y.sum()                 # simple scalar loss: sum of all outputs

# PyTorch autograd backward
L.backward()

# Manual computation of gradients
# dL/dY = ones(B, D_out) because L = sum(Y)
G = torch.ones(B, D_out)    # upstream gradient

# Our derived formulas:
dL_dW_manual = X.T @ G      # shape (D_in, D_out)  == X^T G
dL_dX_manual = G @ W.T      # shape (B, D_in)       == G W^T

# Compare with autograd
print("dL/dW matches:", torch.allclose(W.grad, dL_dW_manual))  # True
print("dL/dX matches:", torch.allclose(X.grad, dL_dX_manual))  # True

# ---- Linear layer with bias: Y = XW + b ----
# dL/db = G.sum(dim=0)   (sum over batch, same shape as b)
b = torch.zeros(D_out, requires_grad=True)
Y2 = X @ W.detach() + b    # use detached W to test only bias grad
L2 = Y2.sum()
L2.backward()
G2 = torch.ones(B, D_out)
dL_db_manual = G2.sum(dim=0)
print("dL/db matches:", torch.allclose(b.grad, dL_db_manual))  # True
```

### The chain rule for multi-layer networks

For a two-layer network $L(f_2(f_1(\mathbf{x})))$, the chain rule stacks: gradients flow backward through $f_2$, then through $f_1$. Each layer's backward pass is a matrix multiply using its transposed weight. See [Neural Networks From Scratch: MLPs & Backprop](../01-foundations/06-neural-nets-from-scratch.html) for the full derivation.

An important identity for softmax + cross-entropy (used everywhere in language models) is that the combined gradient is $\hat{p} - p_{\text{true}}$, where $\hat{p}$ is the predicted probability and $p_{\text{true}}$ is the one-hot target. This beautiful cancellation arises from the Jacobian of softmax combined with the log-derivative of cross-entropy.

---

## Orthogonality, Projections, and Change of Basis

### Orthogonal matrices

A matrix $Q \in \mathbb{R}^{n \times n}$ is **orthogonal** if $Q^\top Q = Q Q^\top = I$. Its columns form an orthonormal basis: they are unit vectors, pairwise perpendicular. Key property: $\|Q\mathbf{x}\|_2 = \|\mathbf{x}\|_2$ — orthogonal matrices preserve lengths and angles. This is why the $U$ and $V$ factors in SVD do not distort the geometry of the data; only $\Sigma$ stretches.

### Projections

The **orthogonal projection** of $\mathbf{b}$ onto the column space of $A$ is:

$$
\hat{\mathbf{b}} = A(A^\top A)^{-1} A^\top \mathbf{b} = P_A \mathbf{b}
$$

where $P_A = A(A^\top A)^{-1} A^\top$ is the **projection matrix**. Properties: $P_A^2 = P_A$ (idempotent), $P_A = P_A^\top$ (symmetric).

In neural networks, **residual connections** can be viewed geometrically as additive "correction" projections. The self-attention head projects queries and keys into a lower-dimensional subspace (head dimension $d_h = d_{\text{model}} / H$) before computing dot-product similarity; this is a learned projection. See [The Attention Mechanism From Scratch](../02-transformer/03-attention-from-scratch.html).

### Change of basis

If $Q$ is orthogonal, then $Q^\top A Q$ represents $A$ in the basis defined by $Q$'s columns. The eigendecomposition $A = Q\Lambda Q^\top$ is exactly this: in the eigenbasis, $A$ acts as a simple scaling by $\lambda_i$ along each axis. The PCA (Principal Components Analysis) transformation is a change of basis to the eigenbasis of the data covariance matrix.

```python
import torch

# Gram-Schmidt orthogonalization: produce an orthonormal basis
def gram_schmidt(V):
    """
    V: (n, k) matrix of k linearly-independent vectors in R^n.
    Returns Q: (n, k) matrix with orthonormal columns spanning col(V).
    """
    Q = []
    for v in V.T:               # iterate over columns
        v = v.clone().float()
        for q in Q:
            v = v - (v @ q) * q  # subtract projection onto previous basis vectors
        v = v / v.norm()         # normalize
        Q.append(v)
    return torch.stack(Q, dim=1)

V = torch.randn(8, 4)           # 4 random vectors in R^8
Q = gram_schmidt(V)

# Verify orthonormality: Q^T Q should be identity
print("Q^T Q =")
print((Q.T @ Q).round(decimals=5))  # should be I_4

# Projection matrix onto col(V)
P = Q @ Q.T                     # (8, 8) projection matrix
b = torch.randn(8)
b_proj = P @ b
# b_proj should lie in the same 4-D subspace, and (b - b_proj) perp to it
residual = b - b_proj
for q in Q.T:
    print(f"Residual · basis_vec = {(residual @ q).item():.6f}")  # ≈ 0
```

---

## Low-Rank Approximations and the LoRA Preview

### Why weights are (often) low-rank

Several lines of evidence suggest that pretrained language model weights and their fine-tuning updates concentrate in low-dimensional subspaces:

1. **Intrinsic dimensionality** (Aghajanyan et al., 2021): fine-tuning can be reformulated as optimization in a very low-dimensional space with minimal loss in performance.
2. **Spectral analysis of weight matrices**: plotting the singular value spectrum of pretrained transformer weight matrices reveals a rapid drop — a handful of large singular values capturing most of the "signal," followed by a long tail.
3. **Linear mode connectivity**: different fine-tuned models share much of their weight structure in the dominant singular directions.

### LoRA in matrix algebra terms

LoRA (Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021) freezes the pretrained weights $W_0 \in \mathbb{R}^{d \times d}$ and adds a trainable low-rank perturbation:

$$
W = W_0 + \Delta W = W_0 + BA
$$

where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times d}$ with $r \ll d$. During training, $W_0$ is frozen (no gradient computed); only $A$ and $B$ accumulate gradients.

The forward pass for input $\mathbf{x}$ becomes:

$$
\mathbf{y} = W_0 \mathbf{x} + B(A\mathbf{x}) = W_0 \mathbf{x} + \underbrace{B}_{\text{small}} \underbrace{(A\mathbf{x})}_{\text{rank-}r \text{ proj}}
$$

{{fig:lora-lowrank-bottleneck}}

Parameter count: full fine-tune needs $d^2$ parameters; LoRA needs $2dr$. For $d = 4096$, $r = 8$: full = 16.8M params; LoRA = 65.5K params — a 256× reduction.

At initialization, $A$ is sampled from a Gaussian and $B = 0$, so $\Delta W = 0$ and training begins from the pretrained solution.

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that adds a LoRA adapter.
    The pretrained weight W0 is frozen; only A and B are trained.
    """
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.rank  = rank
        self.alpha = alpha  # scaling factor; effective LR is alpha/rank
        self.scale = alpha / rank

        # Frozen pretrained weight (no gradient)
        self.W0 = nn.Parameter(
            torch.randn(out_features, in_features) * 0.01,
            requires_grad=False
        )

        # Low-rank trainable matrices
        # A initialized with kaiming_uniform (standard for Linear); B = 0
        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)

    def forward(self, x):
        # Standard path: y = x W0^T
        y0 = x @ self.W0.T

        # LoRA path: delta_y = x A^T B^T * scale
        # = (x @ A^T) @ B^T  -- rank-r bottleneck
        delta = (x @ self.A.T) @ self.B.T
        return y0 + self.scale * delta

    @property
    def params_trainable(self):
        return self.rank * (self.in_features + self.out_features)

    @property
    def params_total_full_finetuning(self):
        return self.in_features * self.out_features


# Quick sanity check
d = 512
layer = LoRALinear(d, d, rank=8, alpha=16)
x = torch.randn(4, 16, d)          # batch=4, seq=16, d=512
y = layer(x)
print(f"Output shape: {y.shape}")  # (4, 16, 512)

trainable   = sum(p.numel() for p in layer.parameters() if p.requires_grad)
total_equiv = layer.params_total_full_finetuning
print(f"LoRA trainable params: {trainable:,} vs full fine-tuning: {total_equiv:,}")
# e.g. 8,192 vs 262,144 — about 32× reduction
```

!!! note "Connection to SVD initialization"
    One can also initialize LoRA from the top-$r$ SVD of $W_0$: set $A = \Sigma_r^{1/2} V_r^\top$ and $B = U_r \Sigma_r^{1/2}$, so $BA = U_r \Sigma_r V_r^\top = A_r$ (the best rank-$r$ approximation). This starts LoRA from the most informative low-rank subspace and can improve convergence. Some variants (e.g., PiSSA) exploit exactly this idea.

---

## Practical Computing: Numerical Stability and Efficient Operations

### Avoiding explicit inverses

Never compute $A^{-1}$ and then multiply when you can instead solve $Ax = b$ directly. Matrix inversion is numerically unstable and costs $O(n^3)$ just like factorization, but without the numerical benefits of structured solvers. Use:

- `torch.linalg.solve(A, b)` for $A^{-1}b$
- `torch.linalg.lstsq(A, b)` for overdetermined systems
- Cholesky factorization when $A$ is PSD: `torch.linalg.cholesky(A)`

### Einsum notation

Einstein summation (`torch.einsum`) is the most general notation for tensor contractions. Every matrix multiplication, batch multiplication, outer product, and trace can be written as an einsum. It often compiles to efficient CUDA kernels.

```python
import torch

A = torch.randn(3, 4)
B = torch.randn(4, 5)

# Standard matmul: C_ij = sum_k A_ik B_kj
C1 = torch.einsum('ik,kj->ij', A, B)          # (3, 5)
C2 = A @ B
assert torch.allclose(C1, C2)

# Batch matmul: 32 heads, each (S, D) @ (D, S) -> (S, S)
Q = torch.randn(32, 64, 128)   # (heads, seq, dim)
K = torch.randn(32, 64, 128)

# "bhsd,bhtd->bhst" — but for this case, use:
scores = torch.einsum('hsd,htd->hst', Q, K)   # (32, 64, 64) attention scores

# Outer product: v_i w_j
v = torch.randn(5)
w = torch.randn(7)
outer = torch.einsum('i,j->ij', v, w)         # (5, 7)

# Trace: tr(A) = sum_i A_ii
sq = torch.randn(6, 6)
trace = torch.einsum('ii->', sq)              # scalar
print(f"trace: {trace.item():.4f}, check: {sq.diagonal().sum().item():.4f}")

# Frobenius inner product: <A, B>_F = tr(A^T B) = sum_{ij} A_{ij} B_{ij}
A2 = torch.randn(4, 4)
B2 = torch.randn(4, 4)
frob_inner = torch.einsum('ij,ij->', A2, B2)  # same as (A2 * B2).sum()
```

### einops: named-axis tensor ops

`einops` (a separate library, `pip install einops`; works transparently with PyTorch, NumPy, JAX, and TensorFlow tensors) replaces error-prone chains of `.view()`, `.reshape()`, `.permute()`, `.transpose()`, `.squeeze()`, and `.unsqueeze()` with a single readable pattern string. Where `einsum` expresses contractions (sums over shared indices), einops expresses pure re-layout and reductions: no axis is ever summed unless you explicitly reduce. The pattern is a mini-language: names left of `->` label the input axes, names on the right give the output order; parentheses group axes (merge, on the right) or split a composite axis (on the left, with the split size supplied as a keyword like `h=8`). Because every axis is named, the operation is self-documenting and shape bugs surface as readable errors rather than silently-wrong strides.

The three verbs: `rearrange` (reorder/split/merge axes, never changes the number of elements), `reduce` (rearrange + aggregate over axes that disappear, with reduction one of `'mean'`|`'sum'`|`'max'`|`'min'`|`'prod'`), and `repeat` (rearrange + tile/broadcast new axes into existence). The key invariant: any name appearing on exactly one side is created (`repeat`) or removed (`reduce`); `rearrange` requires the multiset of element-carrying names to match on both sides.

```python
import torch
from einops import rearrange, reduce, repeat

x = torch.randn(3, 4)
xt = rearrange(x, 'a b -> b a')               # (4, 3)  == x.T / x.transpose(0, 1)

# Merge axes (flatten)
B, S, D = 2, 4, 6
x2 = torch.randn(B, S, D)
flat = rearrange(x2, 'b s d -> b (s d)')      # (2, 24)  == x2.reshape(B, S*D)

# Split axes (inverse of the merge above)
back = rearrange(flat, 'b (s d) -> b s d', d=D)  # (2, 4, 6)  == flat.reshape(B, S, D)
assert torch.allclose(back, x2)

# Multi-head attention: split the model dim into (heads, head_dim)
B, S, D, H = 2, 4, 6, 2
d_head = D // H  # 3
x = torch.randn(B, S, D)

xh = rearrange(x, 'b s (h d) -> b h s d', h=H)   # (2, 2, 4, 3)
# The composite '(h d)' on the LEFT means D is read as h-major, d-minor --
# matching reshape(..., H, d_head), NOT reshape(..., d_head, H). Getting this
# ordering backwards is the classic multi-head bug.
x_manual = x.reshape(B, S, H, d_head).permute(0, 2, 1, 3)
assert torch.allclose(xh, x_manual)

# Merge heads back -- exact round trip
x_back = rearrange(xh, 'b h s d -> b s (h d)')
assert torch.allclose(x_back, x)              # exact inverse

# reduce: mean-pool over the sequence axis
pooled = reduce(x, 'b s d -> b d', 'mean')    # == x.mean(dim=1)
assert torch.allclose(pooled, x.mean(dim=1))

# repeat: broadcast a per-position mask across heads (creates a new axis)
mask = torch.zeros(B, S)
mask_h = repeat(mask, 'b s -> b h s', h=H)    # (2, 2, 4)  == mask[:, None, :].expand(B, H, S)

# all asserts above passing == correct
print('heads split:', xh.shape, '| round-trip OK')
# heads split: torch.Size([2, 2, 4, 3]) | round-trip OK
```

This book uses einops for Vision Transformer patchification (see [Vision Transformers](../10-multimodal-and-arch/01-vision-transformers.html)), where an image `(b c (h1 h2) (w1 w2))` is rearranged to a patch sequence `(b (h1 w1) (h2 w2 c))` in a single call; `einops.layers.torch.Rearrange` can likewise be dropped into an `nn.Sequential` as a shape-changing layer. The same named-axis idea underlies `torch.einsum` (above), and HF/timm model code uses einops widely, so the pattern language is worth internalizing.

### Memory layout: contiguous tensors

PyTorch stores tensors in row-major (C-contiguous) order by default. After a `.transpose()` or `.permute()`, the tensor may become non-contiguous, causing performance regressions in subsequent operations. Call `.contiguous()` before passing to `@` or `F.linear` when in doubt.

```python
import torch

# After transpose, tensor is non-contiguous
A = torch.randn(1024, 512)
B = A.T        # logical transpose, but storage is not re-arranged
print(B.is_contiguous())   # False

# Make contiguous (copies data, but subsequent ops are fast)
B_c = B.contiguous()
print(B_c.is_contiguous()) # True

# torch.linalg.svd and matmul accept non-contiguous but may be slower
# In practice, after a permute in multi-head attention, call .contiguous():
x = torch.randn(2, 8, 32, 64)          # (batch, heads, seq, dim)
x_perm = x.permute(0, 2, 1, 3)         # (batch, seq, heads, dim)
x_cont = x_perm.contiguous()           # ensures efficient downstream matmul
```

!!! warning "Common pitfall: implicit broadcasting with matmul"
    PyTorch matmul (`@`) broadcasts over leading batch dimensions: if $A$ is `(B, M, K)` and $B$ is `(K, N)`, the result is `(B, M, N)`. This is useful but can silently hide shape bugs. Always print shapes when debugging unexpected gradient values. Use `torch.Size` assertions in production code to fail fast on mismatches.

---

## Key Takeaways

!!! key "Key Takeaways"
    - Matrix multiplication is the fundamental operation of deep learning. Every linear layer, attention score, and projection is a matmul. Its FLOP cost is $O(mnk)$; memory bandwidth, not raw FLOPs, is often the bottleneck.
    - The SVD $A = U\Sigma V^\top$ decomposes any matrix into rotation, scaling, and rotation. The Eckart-Young theorem guarantees that truncating to the top-$r$ singular components gives the best rank-$r$ approximation in Frobenius norm.
    - Low-rank structure is empirically pervasive in fine-tuning updates, enabling LoRA: replacing $\Delta W$ with $BA$ (where $r \ll d$) compresses trainable parameters by orders of magnitude with minimal quality loss.
    - The two most important backprop identities are $\partial L / \partial W = X^\top G$ and $\partial L / \partial X = G W^\top$, where $G$ is the upstream gradient. Everything else in backprop follows from stacking these.
    - Vector and matrix norms ($\ell_2$, Frobenius, spectral) appear in regularization, gradient clipping, and Lipschitz analysis. Know which norm each technique uses.
    - Orthogonal matrices preserve lengths and angles; $U$ and $V$ in the SVD are orthogonal. Orthogonality is why attention projection heads do not distort the embedding geometry.
    - Einsum notation (`torch.einsum`) unifies all tensor contractions in a single API and often compiles to optimal CUDA kernels. Prefer it over manual reshapes when expressing complex multi-dimensional operations.
    - Never compute explicit matrix inverses in code; use `torch.linalg.solve` or factorization routines for numerical stability.

---

!!! sota "State of the Art & Resources (2026)"
    Linear algebra for deep learning is a mature, stable field — the core tools (SVD, matmul, backprop identities) are decades old — but active research continues around efficient low-rank methods, randomised numerical linear algebra, and spectral analysis of trained networks. The resources below span bedrock theory through cutting-edge practice.

    **Textbooks & courses**

    - [Gilbert Strang, MIT 18.06 Linear Algebra (OCW)](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) — Strang's legendary video lectures; chapters on eigenvalues and SVD are essential viewing for any ML practitioner.
    - [Goodfellow, Bengio & Courville, *Deep Learning* Ch. 2 — Linear Algebra](https://www.deeplearningbook.org/contents/linear_algebra.html) — The canonical ML-focused treatment, freely available online; covers exactly the notation and concepts used throughout this book.
    - [fast.ai: Computational Linear Algebra for Coders (fastai/numerical-linear-algebra)](https://github.com/fastai/numerical-linear-algebra) — Rachel Thomas's hands-on Jupyter notebook course; bridges abstract theory and NumPy/PyTorch implementation with applications like SVD-based background removal and NMF topic modelling.

    **Visual explainers**

    - [3Blue1Brown — Essence of Linear Algebra (YouTube playlist)](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) — 16-video series with landmark geometric animations; the best visual intuition for linear transformations, eigenvectors, and the SVD available anywhere.

    **Foundational papers**

    - [Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021)](https://arxiv.org/abs/2106.09685) — The paper that brought low-rank matrix algebra to the centre of LLM fine-tuning; directly motivates the LoRA preview section above.
    - [Aghajanyan et al., *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning* (2020)](https://arxiv.org/abs/2012.13255) — Shows empirically that fine-tuning updates live in a surprisingly low-dimensional subspace, providing the theoretical underpinning for rank-deficient weight updates.

    **Open-source & tools**

    - [PyTorch `torch.linalg` documentation](https://docs.pytorch.org/docs/2.12/linalg.html) — Official API reference for all linear algebra operations used in this chapter (SVD, norms, solvers, Cholesky); dispatches to cuBLAS/LAPACK under the hood.

    **Reference**

    - [The Matrix Cookbook — Petersen & Pedersen (2012)](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) — Dense desktop reference for matrix calculus identities; invaluable when deriving gradients of custom operations involving traces, inverses, or determinants.

---

## Further Reading

- **Gilbert Strang, *Introduction to Linear Algebra* (5th ed.)** — The clearest introductory treatment of matrix factorizations; Chapter 6 on eigenvalues and Chapter 7 on SVD are essential.
- **Goodfellow, Bengio & Courville, *Deep Learning* (2016), Chapter 2** — The canonical ML textbook treatment of linear algebra; available freely online.
- **Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021)** — Original LoRA paper demonstrating that fine-tuning updates have low intrinsic rank; directly motivates Section 6 above.
- **Aghajanyan, Zettlemoyer & Gupta, *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning* (2021)** — Shows that fine-tuning effective dimensionality is far smaller than parameter count.
- **NumPy `linalg` documentation** and **PyTorch `torch.linalg` documentation** — Comprehensive API references for the operations in this chapter; check for dispatch to LAPACK / cuBLAS under the hood.
- **Matrix Cookbook (Petersen & Pedersen)** — A dense reference sheet of matrix calculus identities; useful as a lookup table for deriving gradients of custom operations.
