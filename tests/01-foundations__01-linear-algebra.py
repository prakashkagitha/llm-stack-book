"""CI-tested extracts of runnable code blocks from
content/01-foundations/01-linear-algebra.md

Each `block_*` function reproduces the book's ACTUAL code verbatim (modulo
wrapping in a function) and then asserts the claims the prose/comments make
about the results. Run directly: `python3 tests/01-foundations__01-linear-algebra.py`
"""
import numpy as np
import torch


def block_vectors_matrices_tensors():
    # content lines ~39-67
    import torch

    # ----- Vectors -----
    v = torch.tensor([1.0, 2.0, 3.0])          # shape (3,)
    u = torch.tensor([4.0, 5.0, 6.0])

    dot = torch.dot(v, u)                        # 1*4 + 2*5 + 3*6 = 32.0
    cosine_sim = dot / (v.norm() * u.norm())     # ~ 0.9746

    print(f"dot={dot.item():.1f}, cos_sim={cosine_sim.item():.4f}")

    # ----- Matrices -----
    A = torch.randn(4, 3)   # 4x3 matrix
    x = torch.randn(3)      # 3-vector
    y = A @ x               # matrix-vector product, shape (4,)

    # ----- 3-D Tensor (batch of sequences) -----
    B, S, D = 2, 8, 512
    hidden_states = torch.randn(B, S, D)

    # Batch matrix multiply across the batch dimension
    Q = torch.randn(B, S, 64)   # queries
    K = torch.randn(B, S, 64)   # keys
    scores = torch.bmm(Q, K.transpose(1, 2))    # batch matmul
    print(f"Attention score tensor shape: {scores.shape}")  # (2, 8, 8)

    # --- verify book's claims ---
    assert dot.item() == 32.0
    assert abs(cosine_sim.item() - 0.9746) < 1e-3
    assert y.shape == (4,)
    assert hidden_states.shape == (2, 8, 512)
    assert scores.shape == (2, 8, 8)


def block_matmul_benchmark():
    # content lines ~100-135
    import torch
    import time

    A = torch.randn(4096, 4096)
    B = torch.randn(4096, 4096)

    # CPU
    t0 = time.perf_counter()
    C_cpu = A @ B
    t1 = time.perf_counter()
    print(f"CPU matmul 4096x4096: {(t1-t0)*1000:.1f} ms")

    # GPU (if available)
    if torch.cuda.is_available():
        A_gpu = A.cuda().to(torch.bfloat16)
        B_gpu = B.cuda().to(torch.bfloat16)
        _ = A_gpu @ B_gpu
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        C_gpu = A_gpu @ B_gpu
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        print(f"GPU matmul 4096x4096 (BF16): {(t1-t0)*1000:.2f} ms")
    else:
        # SKIP(gpu): no CUDA device; GPU timing branch not exercised. CPU path
        # and the batched-matmul below (the load-bearing logic) still run.
        pass

    # Batched matmul -- critical for transformer attention
    H, S, d = 32, 512, 128
    Q = torch.randn(H, S, d)
    K = torch.randn(H, S, d)
    scores = Q @ K.transpose(-2, -1)  # uses broadcasting/batched matmul
    print(f"Score shape: {scores.shape}")  # (32, 512, 512)

    assert C_cpu.shape == (4096, 4096)
    assert scores.shape == (32, 512, 512)


def block_rank_empirical():
    # content lines ~162-182
    import numpy as np

    W = np.random.randn(768, 768)           # full-rank weight matrix
    rank_full = np.linalg.matrix_rank(W)
    print(f"Random 768x768 rank: {rank_full}")  # should be 768

    r = 4
    A_lr = np.random.randn(768, r)
    B_lr = np.random.randn(r, 768)
    W_lr = A_lr @ B_lr                      # rank at most 4
    rank_lr = np.linalg.matrix_rank(W_lr)
    print(f"Low-rank 768x768 (r=4) rank: {rank_lr}")  # 4

    params_full = 768 * 768          # 589,824
    params_lora = 768 * r + r * 768  # 6,144
    print(f"Full: {params_full:,} params, LoRA r=4: {params_lora:,} params")

    assert rank_full == 768
    assert rank_lr == 4
    assert params_full == 589_824
    assert params_lora == 6_144


def block_svd_lowrank():
    # content lines ~232-265
    import torch

    torch.manual_seed(42)
    m, n, true_rank = 256, 256, 8
    U_true = torch.randn(m, true_rank)
    V_true = torch.randn(n, true_rank)
    W = U_true @ V_true.T + 0.1 * torch.randn(m, n)  # rank-8 + noise

    U, S, Vh = torch.linalg.svd(W, full_matrices=False)

    print("Singular values (first 12):")
    print(S[:12].numpy().round(2))

    def low_rank_approx(U, S, Vh, r):
        """Reconstruct W using only the top-r singular components."""
        return (U[:, :r] * S[:r]) @ Vh[:r, :]

    errs = {}
    for r in [1, 4, 8, 16, 32]:
        W_r = low_rank_approx(U, S, Vh, r)
        rel_err = (W - W_r).norm() / W.norm()
        n_params_full = m * n
        n_params_lr = r * (m + n)
        print(f"rank-{r:2d}: rel_err={rel_err:.4f}, "
              f"params={n_params_lr:,} vs {n_params_full:,}")
        errs[r] = rel_err.item()

    # 8 large singular values, then a cliff (a rank-8 signal + noise floor)
    assert S[7] > 50.0, "top-8 singular values should be large"
    assert S[8] < 10.0, "9th singular value should drop off (noise floor)"
    # rank-8 captures the signal: relative error ~3%
    assert errs[8] < 0.10
    assert abs(errs[8] - 0.03) < 0.02
    # error decreases monotonically as we keep more components
    assert errs[1] > errs[4] > errs[8] > errs[16] > errs[32]


def block_norms():
    # content lines ~310-330
    import torch

    A = torch.randn(128, 256)

    frob = torch.linalg.norm(A, ord='fro')
    S = torch.linalg.svdvals(A)          # sorted descending
    spectral = S[0]
    nuclear = S.sum()

    print(f"Frobenius: {frob:.2f}, Spectral: {spectral:.2f}, Nuclear: {nuclear:.2f}")

    # identities stated in the prose:
    # ||A||_F = sqrt(sum sigma_i^2); spectral = max sigma; nuclear = sum sigma
    assert torch.allclose(frob, torch.sqrt((S**2).sum()), rtol=1e-4)
    assert torch.allclose(frob, torch.sqrt((A**2).sum()), rtol=1e-4)
    assert spectral == S.max()
    assert spectral <= nuclear


def block_manual_backprop():
    # content lines ~386-424
    import torch

    torch.manual_seed(0)
    B, D_in, D_out = 4, 8, 6

    X = torch.randn(B, D_in, requires_grad=True)
    W = torch.randn(D_in, D_out, requires_grad=True)

    Y = X @ W                   # shape (B, D_out)
    L = Y.sum()

    L.backward()

    G = torch.ones(B, D_out)    # upstream gradient

    dL_dW_manual = X.T @ G      # X^T G
    dL_dX_manual = G @ W.T      # G W^T

    matches_W = torch.allclose(W.grad, dL_dW_manual)
    matches_X = torch.allclose(X.grad, dL_dX_manual)
    print("dL/dW matches:", matches_W)
    print("dL/dX matches:", matches_X)

    # ---- Linear layer with bias: Y = XW + b ----
    b = torch.zeros(D_out, requires_grad=True)
    Y2 = X @ W.detach() + b
    L2 = Y2.sum()
    L2.backward()
    G2 = torch.ones(B, D_out)
    dL_db_manual = G2.sum(dim=0)
    matches_b = torch.allclose(b.grad, dL_db_manual)
    print("dL/db matches:", matches_b)

    assert matches_W and matches_X and matches_b


def block_gram_schmidt():
    # content lines ~456-489
    import torch

    def gram_schmidt(V):
        Q = []
        for v in V.T:               # iterate over columns
            v = v.clone().float()
            for q in Q:
                v = v - (v @ q) * q
            v = v / v.norm()
            Q.append(v)
        return torch.stack(Q, dim=1)

    torch.manual_seed(0)
    V = torch.randn(8, 4)
    Q = gram_schmidt(V)

    print("Q^T Q =")
    print((Q.T @ Q).round(decimals=5))

    P = Q @ Q.T
    b = torch.randn(8)
    b_proj = P @ b
    residual = b - b_proj
    for q in Q.T:
        print(f"Residual . basis_vec = {(residual @ q).item():.6f}")

    # orthonormal columns: Q^T Q == I_4
    assert torch.allclose(Q.T @ Q, torch.eye(4), atol=1e-5)
    # projection is idempotent and symmetric
    assert torch.allclose(P @ P, P, atol=1e-5)
    assert torch.allclose(P, P.T, atol=1e-5)
    # residual is orthogonal to the subspace
    assert torch.allclose(Q.T @ residual, torch.zeros(4), atol=1e-5)


def block_lora_linear():
    # content lines ~525-583
    import torch
    import torch.nn as nn

    class LoRALinear(nn.Module):
        def __init__(self, in_features, out_features, rank=8, alpha=16):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.rank = rank
            self.alpha = alpha
            self.scale = alpha / rank

            self.W0 = nn.Parameter(
                torch.randn(out_features, in_features) * 0.01,
                requires_grad=False
            )

            self.A = nn.Parameter(torch.empty(rank, in_features))
            self.B = nn.Parameter(torch.zeros(out_features, rank))
            nn.init.kaiming_uniform_(self.A, a=5**0.5)

        def forward(self, x):
            y0 = x @ self.W0.T
            delta = (x @ self.A.T) @ self.B.T
            return y0 + self.scale * delta

        @property
        def params_trainable(self):
            return self.rank * (self.in_features + self.out_features)

        @property
        def params_total_full_finetuning(self):
            return self.in_features * self.out_features

    torch.manual_seed(0)
    d = 512
    layer = LoRALinear(d, d, rank=8, alpha=16)
    x = torch.randn(4, 16, d)
    y = layer(x)
    print(f"Output shape: {y.shape}")  # (4, 16, 512)

    trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    total_equiv = layer.params_total_full_finetuning
    print(f"LoRA trainable params: {trainable:,} vs full fine-tuning: {total_equiv:,}")

    assert y.shape == (4, 16, 512)
    assert trainable == 8 * (d + d) == 8_192
    assert total_equiv == d * d == 262_144
    assert layer.params_trainable == trainable
    # B initialized to zero => delta W = 0 at init => output is pure W0 path
    y0_only = x @ layer.W0.T
    assert torch.allclose(y, y0_only), "at init B=0 so LoRA delta must be 0"
    # W0 is frozen
    assert not layer.W0.requires_grad


def block_einsum():
    # content lines ~604-636
    import torch

    A = torch.randn(3, 4)
    B = torch.randn(4, 5)

    C1 = torch.einsum('ik,kj->ij', A, B)          # (3, 5)
    C2 = A @ B
    assert torch.allclose(C1, C2)

    Q = torch.randn(32, 64, 128)
    K = torch.randn(32, 64, 128)
    scores = torch.einsum('hsd,htd->hst', Q, K)   # (32, 64, 64)

    v = torch.randn(5)
    w = torch.randn(7)
    outer = torch.einsum('i,j->ij', v, w)         # (5, 7)

    sq = torch.randn(6, 6)
    trace = torch.einsum('ii->', sq)
    print(f"trace: {trace.item():.4f}, check: {sq.diagonal().sum().item():.4f}")

    A2 = torch.randn(4, 4)
    B2 = torch.randn(4, 4)
    frob_inner = torch.einsum('ij,ij->', A2, B2)

    assert scores.shape == (32, 64, 64)
    assert outer.shape == (5, 7)
    assert torch.allclose(outer, torch.outer(v, w))
    assert torch.allclose(trace, sq.diagonal().sum())
    assert torch.allclose(frob_inner, (A2 * B2).sum())


def block_einops():
    # content lines ~644-687
    import torch
    from einops import rearrange, reduce, repeat

    torch.manual_seed(0)
    x = torch.randn(3, 4)
    xt = rearrange(x, 'a b -> b a')               # (4, 3)
    assert torch.allclose(xt, x.T)

    B, S, D = 2, 4, 6
    x2 = torch.randn(B, S, D)
    flat = rearrange(x2, 'b s d -> b (s d)')      # (2, 24)
    assert torch.allclose(flat, x2.reshape(B, S * D))

    back = rearrange(flat, 'b (s d) -> b s d', d=D)
    assert torch.allclose(back, x2)

    B, S, D, H = 2, 4, 6, 2
    d_head = D // H
    x = torch.randn(B, S, D)

    xh = rearrange(x, 'b s (h d) -> b h s d', h=H)   # (2, 2, 4, 3)
    x_manual = x.reshape(B, S, H, d_head).permute(0, 2, 1, 3)
    assert torch.allclose(xh, x_manual)

    x_back = rearrange(xh, 'b h s d -> b s (h d)')
    assert torch.allclose(x_back, x)

    pooled = reduce(x, 'b s d -> b d', 'mean')
    assert torch.allclose(pooled, x.mean(dim=1))

    mask = torch.zeros(B, S)
    mask_h = repeat(mask, 'b s -> b h s', h=H)    # (2, 2, 4)
    assert mask_h.shape == (2, 2, 4)
    assert torch.allclose(mask_h, mask[:, None, :].expand(B, H, S))

    print('heads split:', xh.shape, '| round-trip OK')


def block_contiguous():
    # content lines ~695-712
    import torch

    A = torch.randn(1024, 512)
    B = A.T
    is_c_before = B.is_contiguous()
    print(is_c_before)   # False

    B_c = B.contiguous()
    is_c_after = B_c.is_contiguous()
    print(is_c_after)    # True

    x = torch.randn(2, 8, 32, 64)
    x_perm = x.permute(0, 2, 1, 3)
    x_cont = x_perm.contiguous()

    assert is_c_before is False
    assert is_c_after is True
    assert x_perm.shape == (2, 32, 8, 64)
    assert x_cont.is_contiguous()
    assert torch.allclose(x_cont, x_perm)


BLOCKS = [
    block_vectors_matrices_tensors,
    block_matmul_benchmark,
    block_rank_empirical,
    block_svd_lowrank,
    block_norms,
    block_manual_backprop,
    block_gram_schmidt,
    block_lora_linear,
    block_einsum,
    block_einops,
    block_contiguous,
]


def main():
    for fn in BLOCKS:
        print(f"\n===== {fn.__name__} =====")
        fn()
    print(f"\nAll {len(BLOCKS)} code blocks executed and verified.")


if __name__ == "__main__":
    main()
