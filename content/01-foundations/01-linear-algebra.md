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

- A single linear projection $W \in \mathbb{R}^{4096 \times 4096}$ applied to a batch $X \in \mathbb{R}^{B \times S \times 4096}$ costs $2 \cdot B \cdot S \cdot 4096^2 \approx 33.6 \times 10^6 \cdot B \cdot S$ FLOPs (about 33.6 MFLOPs *per token*).
- At batch size 1 and sequence length 512, that is $33.6 \times 10^6 \times 512 \approx 17$ GFLOPs per linear layer.

This per-token accounting is the seed of the famous $\approx 6N$ rule (forward + backward cost per token for a model of $N$ parameters): each weight participates in one multiply-accumulate in the forward pass ($2N$) and two more in the backward pass ($4N$, once for $\partial L/\partial X$ and once for $\partial L/\partial W$ — exactly the two identities derived later in this chapter). We use that rule to budget the Stack-100M run in [Mini Scaling Laws: Fit Your Own Law Before Spending the Budget](../14-capstone/05-mini-scaling-laws.html) and [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html).

By 2026, flagship datacenter GPUs have moved well past the once-"modern" A100 (312 TFLOPS in BF16, circa 2020) through the Hopper (H100/H200) generation to NVIDIA's Blackwell (B200/GB200) and Blackwell Ultra (GB300) chips, each delivering roughly an order of magnitude more raw BF16 throughput than the A100. So in theory you can run enormous numbers of linear layers per second — but memory bandwidth is still often the bottleneck, not raw FLOPs. See [The Roofline Model & Performance Engineering](../04-kernels-efficiency/01-roofline-performance.html) for the full story.

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

### What `@` actually calls: BLAS, tensor cores, and accumulation precision

`torch.matmul` is not a Python loop and not even a PyTorch kernel in the usual sense: it is a *dispatch*. On CPU it lands in a BLAS (Basic Linear Algebra Subprograms) library — oneAPI MKL, OpenBLAS, or oneDNN, depending on how your wheel was built. On NVIDIA GPUs it lands in **cuBLAS**/**cuBLASLt**, NVIDIA's closed-source BLAS, whose kernels are generated from the same tiling ideas as the open-source **CUTLASS** template library. Those kernels are what actually feed the *tensor cores*, the matmul-specific hardware units that give a modern GPU its headline TFLOPS number. Nothing you write in PyTorch beats them for plain dense matmul; the reason to learn Triton and CUDA later (see [Writing GPU Kernels with Triton](../04-kernels-efficiency/04-triton-kernels.html)) is to *fuse* the operations around the matmul, not to replace it.

Two precision knobs follow directly from this, and both matter the first time you train a model:

- **TF32 for fp32 matmuls.** Ampere and later tensor cores can execute an `fp32` matmul in TensorFloat-32 mode: inputs rounded to a 10-bit mantissa, accumulation still in fp32. It is typically several times faster and, in practice, harmless for training. PyTorch defaults to full fp32 (`"highest"`); you opt in with one line.
- **Accumulation is wider than storage.** A `bf16` matmul on tensor cores multiplies bf16 inputs but accumulates the length-$k$ sum in fp32, then writes bf16 out. This is why bf16 training works at all: bf16 has only ~8 bits of mantissa, and naively accumulating $k = 4096$ terms in bf16 would destroy the result. See [Mixed Precision, bf16 & FP8 Training](../03-pretraining/08-mixed-precision-fp8.html) and [Numerical Computing, Floating Point & Precision](../01-foundations/04-numerics-precision.html).

```python
import torch

# Opt in to TF32 tensor cores for fp32 matmuls (Ampere/Hopper/Blackwell).
# "highest" = true fp32 (default), "high" = TF32, "medium" = bf16-ish.
torch.set_float32_matmul_precision("high")

# Accumulation width: bf16 inputs, fp32 accumulate.
# Compare a bf16 matmul against an fp64 reference to see the error scale.
k = 4096
A = torch.randn(512, k)
B = torch.randn(k, 512)
ref = (A.double() @ B.double())                     # near-exact reference
err_bf16 = ((A.bfloat16() @ B.bfloat16()).double() - ref).abs().mean()
err_fp32 = ((A @ B).double() - ref).abs().mean()
print(f"mean |err|  bf16={err_bf16:.4f}   fp32={err_fp32:.6f}")
# bf16 error is ~1e-1 on entries of size ~sqrt(k)~64: a relative error
# ~1e-3, consistent with bf16's ~8-bit mantissa -- NOT with accumulating
# 4096 terms in bf16, which would be far worse.
```

!!! warning "Matmul is not bitwise reproducible"
    Floating-point addition is not associative, and cuBLAS picks a tiling (and sometimes a split-$k$ reduction order) based on shapes, GPU model, and library version. The same `A @ B` can therefore give bitwise-different results across machines or PyTorch versions. Never assert bitwise equality across hardware; use `torch.allclose` with a tolerance matched to the dtype (roughly `1e-6` for fp32, `1e-2` for bf16). For run-to-run determinism on one machine, `torch.use_deterministic_algorithms(True)` plus a fixed `CUBLAS_WORKSPACE_CONFIG` gets you most of the way, at a performance cost.

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

Eigenvalues govern stability in optimization. The **condition number** in the 2-norm is $\kappa_2(A) = \sigma_{\max}(A) / \sigma_{\min}(A)$ — a ratio of *singular* values in general, which reduces to $\lambda_{\max} / \lambda_{\min}$ exactly when $A$ is symmetric positive definite (the case that matters for a loss Hessian near a minimum). A large $\kappa$ means the loss surface is a long thin valley: plain gradient descent must use a step size set by the steepest direction $\lambda_{\max}$ while progress along the flattest direction $\lambda_{\min}$ crawls, so the number of steps to converge scales like $\kappa$. This is part of why adaptive optimizers (Adam et al.) help — per-coordinate step sizes are a cheap diagonal preconditioner that shrinks the *effective* condition number — see [Optimizers: SGD, Adam, Adafactor, Lion, Muon & Shampoo](../03-pretraining/09-optimizers.html) and [Calculus, Optimization & Convexity](../01-foundations/03-calculus-optimization.html).

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
    [306.8, 291.2, 268.7, 258.5, 246.6, 233.5, 209.2, 192.2, 3.1, 3.1, ...]
    ```

    The first eight singular values are large (the rank-8 signal), then the spectrum falls off a cliff to a noise floor near 3.1 — the singular values of the $0.1 \times \text{randn}(256,256)$ perturbation, whose scale is roughly $0.1 \cdot 2\sqrt{256} \approx 3.2$. A rank-8 approximation achieves relative error of roughly 0.03 (3%), using only $8 \times (256 + 256) = 4{,}096$ parameters instead of $256^2 = 65{,}536$ — a 16× compression. A rank-16 approximation barely improves (the extra singular values are all noise), confirming that the information-carrying subspace truly has low dimension.

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

For a scalar function $f: \mathbb{R}^n \to \mathbb{R}$, the **gradient** $\nabla_{\mathbf{x}} f \in \mathbb{R}^n$ has entries $(\nabla_{\mathbf{x}} f)_i = \partial f / \partial x_i$. For a vector function $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$, the **Jacobian** $J \in \mathbb{R}^{m \times n}$ has $J_{ij} = \partial f_i / \partial x_j$ (**numerator layout**: output index first).

Throughout this book we use the **ML convention**: Jacobians of vector functions are laid out numerator-first as above, but the gradient of a *scalar* loss with respect to *any* variable — vector, matrix, or 4-D tensor — is stored with **the same shape as that variable**. This is exactly what PyTorch does (`p.grad.shape == p.shape` always), and it is what lets us write $\partial L / \partial W$ as a matrix the same size as $W$ below. The two conventions only ever collide when you differentiate a vector by a vector, which in practice we avoid: autodiff never materializes a Jacobian, it only ever computes vector-Jacobian products (see [Automatic Differentiation & PyTorch Internals](../01-foundations/07-autodiff-pytorch.html)).

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

!!! note "Aside: orthogonalization is not just theory — it is the Muon optimizer"
    Given the SVD $G = U \Sigma V^\top$ of a gradient (or momentum) matrix, the matrix $UV^\top$ is the closest *semi-orthogonal* matrix to $G$: it keeps every singular *direction* but flattens every singular *value* to 1, so its spectral norm is exactly 1 and it stretches all directions equally. The **Muon** optimizer (Jordan et al., 2024) applies precisely this map to each 2-D parameter's momentum before stepping, so no single dominant singular direction can hog the update. Computing a full SVD every step would be far too slow, so Muon approximates $UV^\top$ with a handful of **Newton–Schulz** iterations — a fixed odd polynomial in $G$ applied repeatedly, which needs only matmuls and therefore runs at tensor-core speed. This is the clearest example in the book of a "pure linear algebra" fact becoming a production training technique; Stack-100M trains with it in [Optimizer & Schedule: Muon + MuonClip and Warmup-Stable-Decay](../14-capstone/06-optimizer-and-schedule.html), and the full derivation is in [Optimizers: SGD, Adam, Adafactor, Lion, Muon & Shampoo](../03-pretraining/09-optimizers.html).

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

### The library you would actually use: Hugging Face `peft`

You now understand LoRA well enough to have written it. In production nobody re-writes it: **`peft`** (Parameter-Efficient Fine-Tuning, `pip install peft`) does the module surgery for you — it walks the model, swaps every matching `nn.Linear` for a `lora.Linear` wrapper holding exactly the frozen $W_0$ and trainable $A$, $B$ above, and gives you save/load of adapters as ~megabyte files. It is the layer that TRL, Axolotl, and Unsloth all build on. The full treatment (QLoRA, DoRA, rank/target-module choice) is in [PEFT I: LoRA, QLoRA, DoRA & The Adapter Family](../05-posttraining-alignment/03-peft-lora-qlora.html); here is the whole API surface on a toy module so you can see it is the same math:

```python
# pip install peft
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

class TinyBlock(nn.Module):
    """Stand-in for a transformer block's projections."""
    def __init__(self, d=64):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)

    def forward(self, x):
        return self.v_proj(torch.relu(self.q_proj(x)))

base = TinyBlock()

cfg = LoraConfig(
    r=8,                                   # the rank r from the algebra above
    lora_alpha=16,                         # the alpha in scale = alpha / r
    lora_dropout=0.0,
    target_modules=["q_proj", "v_proj"],   # which Linear layers get adapters
    bias="none",
)
model = get_peft_model(base, cfg)
model.print_trainable_parameters()
# trainable params: 2,048 || all params: 10,368 || trainable%: 19.75
# (2 layers x r x (d_in + d_out) = 2 x 8 x 128 = 2,048 -- our formula exactly)

y = model(torch.randn(2, 64))              # trains like any nn.Module

# Fold the adapter into the base weights for zero-overhead inference:
# this is literally W0 + (alpha/r) * B @ A, the merge you derive in Exercise 5.
merged = model.merge_and_unload()
assert isinstance(merged, TinyBlock)
```

Note `LoraConfig(init_lora_weights="pissa")`, which implements the SVD-based initialization described next.

!!! note "Connection to SVD initialization"
    One can also initialize the adapter from the top-$r$ SVD of $W_0$: set $A = \Sigma_r^{1/2} V_r^\top$ and $B = U_r \Sigma_r^{1/2}$, so $BA = U_r \Sigma_r V_r^\top = A_r$, the Eckart-Young-optimal rank-$r$ approximation. This is the idea behind **PiSSA** (Principal Singular values and Singular vectors Adaptation). One detail is easy to get wrong: with $B \neq 0$ at initialization the layer no longer reproduces the pretrained function, so PiSSA also replaces the frozen base with the *residual* $W_0^{\text{res}} = W_0 - A_r$. Then $W_0^{\text{res}} + BA = W_0$ exactly at step 0 — the function is unchanged — but the trainable directions are now the *principal* ones rather than a random subspace, which empirically speeds early convergence. (The contrast is with LoRA's default $B = 0$, which achieves the same "no change at init" property the lazy way.) Exercise 5 implements the factorization; adding the residual subtraction is a two-line extension.

---

## Practical Computing: Numerical Stability and Efficient Operations

### Avoiding explicit inverses

Never compute $A^{-1}$ and then multiply when you can instead solve $Ax = b$ directly. Matrix inversion is numerically unstable and costs $O(n^3)$ just like factorization, but without the numerical benefits of structured solvers. Use:

- `torch.linalg.solve(A, b)` for $A^{-1}b$
- `torch.linalg.lstsq(A, b)` for overdetermined systems
- Cholesky factorization when $A$ is PSD: `torch.linalg.cholesky(A)`

### Randomized SVD: when the full factorization is too slow

A full `torch.linalg.svd` costs $O(mn\min(m,n))$ — for a $4096 \times 4096$ weight matrix that is seconds, and you may want it for every matrix in a 100M-parameter model. When you only need the top few singular components (which is the *only* case in this chapter: LoRA/PiSSA initialization, spectral diagnostics, weight compression), use a **randomized** algorithm instead: project $A$ onto a random $q$-dimensional sketch, orthonormalize, and factor the small matrix. PyTorch ships this as `torch.svd_lowrank` (and `torch.pca_lowrank`), which follows the Halko-Martinsson-Tropp randomized-range-finder scheme and costs a couple of matmuls instead of a dense factorization.

```python
import torch, time

torch.manual_seed(0)
W = torch.randn(2048, 8) @ torch.randn(8, 2048) + 0.01 * torch.randn(2048, 2048)

t0 = time.perf_counter(); U, S, Vh = torch.linalg.svd(W, full_matrices=False)
t_full = time.perf_counter() - t0

# q = target rank + a small oversampling margin; niter = power iterations
t0 = time.perf_counter(); Ur, Sr, Vr = torch.svd_lowrank(W, q=16, niter=4)
t_rand = time.perf_counter() - t0

print(f"full SVD {t_full*1e3:.0f} ms   randomized {t_rand*1e3:.0f} ms")
print("top-8 singular values agree:",
      torch.allclose(S[:8], Sr[:8], rtol=1e-3))   # True
# Note torch.svd_lowrank returns V (n x q), not Vh: reconstruct as U diag(S) V^T
approx = (Ur[:, :8] * Sr[:8]) @ Vr[:, :8].T
print("rel err of rank-8 randomized approx:",
      ((W - approx).norm() / W.norm()).item())
```

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
    - Orthogonal matrices preserve lengths and angles; $U$ and $V$ in the SVD are orthogonal. Replacing a gradient matrix $G = U\Sigma V^\top$ by $UV^\top$ — every direction kept, every scale equalized — is exactly what the Muon optimizer does via Newton-Schulz iterations.
    - `A @ B` dispatches to cuBLAS/oneDNN, not to PyTorch: your job is to feed those kernels well (contiguous, tensor-core-friendly shapes) and to set the precision knobs — `torch.set_float32_matmul_precision("high")` for TF32, and remember that bf16 matmuls accumulate in fp32. Matmul is therefore not bitwise reproducible across GPUs or library versions.
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
    - [einops (arogozhnikov/einops)](https://github.com/arogozhnikov/einops) — The named-axis `rearrange`/`reduce`/`repeat` library used throughout this book for shape manipulation; framework-agnostic and now standard in HF and timm model code.
    - [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) — Open-source CUDA C++ templates for tensor-core GEMM; the readable counterpart to closed-source cuBLAS, and the substrate many custom fused kernels are built on.
    - [Hugging Face `peft`](https://huggingface.co/docs/peft) — Production implementation of the LoRA algebra developed in this chapter (plus QLoRA, DoRA, PiSSA initialization, adapter merging); the library TRL, Axolotl, and Unsloth all build on.

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

---

## Exercises

**1.** (Conceptual) LoRA writes the weight update as $\Delta W = BA$ with $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times d}$. Explain why this construction forces $\text{rank}(\Delta W) \leq r$. Then answer: if you set $r = d$, does LoRA still save parameters compared to a full fine-tune? Use the parameter counts from the chapter to justify your answer.

??? note "Solution"
    From the rank rules in the chapter, $\text{rank}(BA) \leq \min(\text{rank}(B), \text{rank}(A))$. The matrix $B$ has only $r$ columns, so $\text{rank}(B) \leq r$; the matrix $A$ has only $r$ rows, so $\text{rank}(A) \leq r$. Therefore $\text{rank}(\Delta W) = \text{rank}(BA) \leq r$. The product $BA$ can never have more than $r$ independent directions no matter what values fill $B$ and $A$ -- the shared inner dimension $r$ is a hard bottleneck on the rank.

    Parameter counts: a full fine-tune of $W \in \mathbb{R}^{d \times d}$ trains $d^2$ parameters. LoRA trains $B$ and $A$, i.e. $dr + rd = 2dr$ parameters. Setting $r = d$ gives $2d \cdot d = 2d^2$ trainable parameters -- *twice* as many as the full fine-tune, and the rank bound $\text{rank}(\Delta W) \leq d$ imposes no constraint at all. So LoRA only helps when $r \ll d$: the savings ratio is $2dr / d^2 = 2r/d$, which is below 1 precisely when $r < d/2$. The method is designed for the regime $r \in \{4, 8, 16\}$ with $d$ in the thousands.

**2.** (Quantitative) A linear projection uses $W \in \mathbb{R}^{4096 \times 4096}$ applied to activations $X \in \mathbb{R}^{B \times S \times 4096}$ with $B = 2$, $S = 1024$. Using the chapter's rule that an $(m \times k)$ by $(k \times n)$ matmul costs $2mkn$ FLOPs:

  (a) Compute the FLOPs for the full projection $XW$.

  (b) Now add a LoRA adapter with rank $r = 8$, whose forward pass computes $B(A\mathbf{x})$ per token. Compute the *extra* FLOPs contributed by the adapter path across the whole batch, and express it as a fraction of the full projection cost.

??? note "Solution"
    **(a)** Treat the batch as $B \cdot S = 2 \cdot 1024 = 2048$ token vectors, each mapped by $W$ (a $4096 \times 4096$ operator). Per token that is a $(1 \times 4096)(4096 \times 4096)$ product costing $2 \cdot 4096 \cdot 4096 = 33{,}554{,}432$ FLOPs. Across all tokens:

    $$
    2 \cdot B \cdot S \cdot d^2 = 2 \cdot 2 \cdot 1024 \cdot 4096^2 = 68{,}719{,}476{,}736 \approx 68.7 \text{ GFLOPs}.
    $$

    **(b)** The adapter path per token is two small matmuls. First $A\mathbf{x}$: $A$ is $(r \times d)$, so $2 \cdot r \cdot d$ FLOPs. Then $B(\cdot)$: $B$ is $(d \times r)$ applied to an $r$-vector, another $2 \cdot d \cdot r$ FLOPs. Per token that is $4rd = 4 \cdot 8 \cdot 4096 = 131{,}072$ FLOPs. Across $B \cdot S = 2048$ tokens:

    $$
    4rd \cdot BS = 131{,}072 \cdot 2048 = 268{,}435{,}456 \approx 0.27 \text{ GFLOPs}.
    $$

    As a fraction of the full projection:

    $$
    \frac{4rd \cdot BS}{2d^2 \cdot BS} = \frac{2r}{d} = \frac{2 \cdot 8}{4096} = \frac{1}{256} \approx 0.39\%.
    $$

    The adapter adds under half a percent of compute on top of the frozen projection -- LoRA is cheap at inference in FLOPs, not just in trainable parameters.

**3.** (Quantitative) A matrix $A \in \mathbb{R}^{m \times n}$ has singular values $\sigma = (10, 8, 6, 2, 1)$ and rank 5. Using the Eckart-Young theorem:

  (a) Compute $\|A\|_F$.

  (b) Compute the Frobenius-norm error $\|A - A_2\|_F$ of the best rank-2 approximation.

  (c) Compute the *relative* error $\|A - A_2\|_F / \|A\|_F$, and compare it to the relative error of the best rank-3 approximation. What do the numbers say about the effective dimensionality of $A$?

??? note "Solution"
    Recall $\|A\|_F = \sqrt{\sum_i \sigma_i^2}$ and, by Eckart-Young, $\|A - A_k\|_F^2 = \sum_{i > k} \sigma_i^2$ (the dropped singular values).

    **(a)** $\sum_i \sigma_i^2 = 100 + 64 + 36 + 4 + 1 = 205$, so $\|A\|_F = \sqrt{205} \approx 14.32$.

    **(b)** Rank-2 keeps $\sigma_1, \sigma_2$ and drops $\sigma_3, \sigma_4, \sigma_5$:
    $$
    \|A - A_2\|_F^2 = 6^2 + 2^2 + 1^2 = 36 + 4 + 1 = 41, \qquad \|A - A_2\|_F = \sqrt{41} \approx 6.40.
    $$

    **(c)** Relative error at rank 2:
    $$
    \frac{\|A - A_2\|_F}{\|A\|_F} = \sqrt{\frac{41}{205}} = \sqrt{0.2} \approx 0.447 \; (44.7\%).
    $$
    Rank-3 drops only $\sigma_4, \sigma_5$: error$^2 = 4 + 1 = 5$, so relative error $= \sqrt{5/205} = \sqrt{0.0244} \approx 0.156 \; (15.6\%)$.

    Interpretation: adding the third component slashes the relative error from 45% to 16% because $\sigma_3 = 6$ still carries real signal, whereas $\sigma_4 = 2$ and $\sigma_5 = 1$ are small. The energy is concentrated in the top three directions -- this matrix behaves as if its effective dimensionality is about 3, which is exactly the low-rank structure that makes truncated-SVD compression (and LoRA) work.

**4.** (Quantitative / backprop) A linear layer computes $Y = XW$ with
$$
X = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \in \mathbb{R}^{2 \times 3}, \qquad W = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} \in \mathbb{R}^{3 \times 2},
$$
and the scalar loss is $L = \sum_{ij} Y_{ij}$ (the sum of all output entries). Using the chapter's identities $\partial L / \partial W = X^\top G$ and $\partial L / \partial X = G W^\top$, compute both gradient matrices by hand. First state what the upstream gradient $G = \partial L / \partial Y$ is.

??? note "Solution"
    Because $L = \sum_{ij} Y_{ij}$, each output entry contributes 1 to the loss, so
    $$
    G = \frac{\partial L}{\partial Y} = \mathbf{1}_{2 \times 2} = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}.
    $$

    **Weight gradient** $\partial L / \partial W = X^\top G$. Here $X^\top \in \mathbb{R}^{3 \times 2}$ and $G \in \mathbb{R}^{2 \times 2}$; since every column of $G$ is all ones, each output column equals the column sums of $X$, namely $(1{+}4,\, 2{+}5,\, 3{+}6) = (5, 7, 9)$:
    $$
    \frac{\partial L}{\partial W} = X^\top G = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}\begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} = \begin{bmatrix} 5 & 5 \\ 7 & 7 \\ 9 & 9 \end{bmatrix}.
    $$
    This has the shape of $W$ ($3 \times 2$), as required.

    **Input gradient** $\partial L / \partial X = G W^\top$. With $W^\top = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix}$ and $G$ all ones, each row of the result is the column sums of $W^\top$, namely $(1{+}0,\, 0{+}1,\, 1{+}1) = (1, 1, 2)$:
    $$
    \frac{\partial L}{\partial X} = G W^\top = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}\begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix} = \begin{bmatrix} 1 & 1 & 2 \\ 1 & 1 & 2 \end{bmatrix}.
    $$
    This has the shape of $X$ ($2 \times 3$). The shapes matching is the quick sanity check the chapter recommends: $X^\top G$ is $(3 \times 2)(2 \times 2) \to 3 \times 2$, and $G W^\top$ is $(2 \times 2)(2 \times 3) \to 2 \times 3$.

**5.** (Implementation) Extend the chapter's `LoRALinear` class with two features used in real LoRA deployments. (a) A `merged_weight()` method that folds the adapter back into a single dense weight $W_0 + \text{scale}\cdot BA$ so inference has zero adapter overhead; verify that a forward pass through the merged weight matches the original two-path forward. (b) An `init_from_svd(W)` method (PiSSA-style, from the chapter's SVD-initialization note) that sets $A$ and $B$ from the top-$r$ SVD of a target matrix $W$, so that $\text{scale}\cdot BA$ equals the best rank-$r$ approximation $A_r$. Verify this against a direct truncated SVD.

??? note "Solution"
    Recall the chapter's forward pass: $y = x W_0^\top + \text{scale}\cdot (x A^\top) B^\top = x\,(W_0 + \text{scale}\cdot BA)^\top$, so the folded weight is $W_0 + \text{scale}\cdot BA$. For the SVD init, the chapter's note gives $A = \Sigma_r^{1/2} V_r^\top$ and $B = U_r \Sigma_r^{1/2}$, so $BA = U_r \Sigma_r V_r^\top = A_r$. To make $\text{scale}\cdot BA = A_r$ exactly (the forward multiplies the adapter by `scale`), we divide $B$ by `scale`.

    ```python
    import torch
    import torch.nn as nn

    class LoRALinearSVD(LoRALinear):
        """LoRALinear plus weight-merging and SVD-based initialization."""

        def merged_weight(self):
            # forward:  y = x @ W0.T + scale * (x @ A.T) @ B.T
            #             = x @ (W0 + scale * B @ A).T
            return self.W0.data + self.scale * (self.B @ self.A)

        @torch.no_grad()
        def init_from_svd(self, W):
            """Init A, B from the top-r SVD of W so scale * B @ A = A_r."""
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            Ur  = U[:, :self.rank]          # (out, r)
            Sr  = S[:self.rank]             # (r,)
            Vhr = Vh[:self.rank, :]         # (r, in)
            sqrt_S = torch.sqrt(Sr)
            # B @ A = (Ur * sqrt_S) @ (sqrt_S[:,None] * Vhr) = Ur diag(Sr) Vhr = A_r
            # divide B by scale so that scale * B @ A == A_r exactly
            self.B.copy_((Ur * sqrt_S) / self.scale)   # (out, r)
            self.A.copy_(sqrt_S[:, None] * Vhr)        # (r, in)


    # ---- (a) merged weight matches the two-path forward ----
    torch.manual_seed(0)
    d, r = 64, 8
    layer = LoRALinearSVD(d, d, rank=r, alpha=16)
    # give B a nonzero value so the adapter path is actually exercised
    nn.init.normal_(layer.B, std=0.02)

    x = torch.randn(3, d)
    y_fwd    = layer(x)                        # original two-path forward
    y_merged = x @ layer.merged_weight().T     # single dense matmul
    print("merged matches forward:",
          torch.allclose(y_fwd, y_merged, atol=1e-5))   # True

    # ---- (b) SVD init reproduces the best rank-r approximation ----
    W_target = torch.randn(d, d)
    layer.init_from_svd(W_target)

    adapter = layer.scale * (layer.B @ layer.A)          # what the adapter encodes
    U, S, Vh = torch.linalg.svd(W_target, full_matrices=False)
    A_r = (U[:, :r] * S[:r]) @ Vh[:r, :]                 # direct rank-r truncation
    print("adapter == rank-r SVD:",
          torch.allclose(adapter, A_r, atol=1e-4))       # True
    ```

    Both checks print `True`. Part (a) confirms the algebraic identity $y = x(W_0 + \text{scale}\cdot BA)^\top$, meaning a trained adapter can be merged into the base weight for overhead-free deployment. Part (b) confirms that the factored initialization reproduces the Eckart-Young optimal rank-$r$ matrix, so training starts from the most informative low-rank subspace rather than from a random one.
