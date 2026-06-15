# 2.11 Beyond Attention: SSMs, Mamba, RWKV & Linear Attention

The transformer has dominated language modeling since Vaswani et al. introduced it in 2017. Yet it carries a fundamental cost: the standard attention mechanism scales quadratically in both compute and memory with sequence length. For a 128 K-token context, a naive attention over a single head requires computing an $N \times N$ score matrix with $N = 131072$ — about 128 billion floating-point multiplications before the value aggregation step. At that scale, alternatives to attention stop being academic curiosities and start being engineering necessities.

This chapter is about those alternatives. We will study what makes attention expensive, then tour the major architectural families that attempt to fix the scaling problem: linear attention, structured state space models (SSMs) like S4 and Mamba, recurrent language models like RWKV, and the Retention mechanism behind RetNet. We end by looking at hybrid architectures that mix attention and these alternatives — a pragmatic approach that is finding real adoption in production LLMs.

Before diving in, note that the upstream machinery these models plug into — tokenization, embeddings, residual-stream design, layer normalization — is shared with standard transformers, covered in [Tokenization: BPE, WordPiece, Unigram & Byte-Level](../02-transformer/01-tokenization.html), [Embeddings & The Input Pipeline](../02-transformer/02-embeddings-input.html), and [The Transformer Block: Norms, Residuals, MLPs & Activations](../02-transformer/06-transformer-block.html). The KV-cache implications of inference are discussed in depth in [The Anatomy of LLM Inference: Prefill, Decode & The KV Cache](../07-inference-serving/01-anatomy-inference.html).

---

## The Quadratic Bottleneck

Let us be precise about what "quadratic attention" actually means and where the cost comes from.

For a sequence of $N$ tokens with model dimension $d$ and head dimension $d_k$, the attention computation is:

$$
\text{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

The $QK^\top$ product produces an $N \times N$ matrix. This single step costs $O(N^2 d_k)$ FLOPs and requires $O(N^2)$ memory to store the attention weights. For $H$ heads:

- **FLOPs per layer**: $O(H \cdot N^2 d_k) = O(N^2 d)$
- **Memory**: $O(N^2)$ attention matrix, which dominates for large $N$

At inference, the autoregressive KV cache grows as $O(N \cdot d)$ per layer, which is linear — but the *prefill* step still reads $O(N^2)$ attention weights. FlashAttention (covered in [FlashAttention I: IO-Awareness & The Online Softmax](../04-kernels-efficiency/02-flash-attention-1.html)) tames the memory cost by not materializing the full matrix, but does not reduce the FLOP count.

!!! example "Worked example: attention FLOPs vs sequence length"

    Consider a 7B-parameter GPT-style model: $d = 4096$, $H = 32$ heads, $d_k = 128$, 32 layers.

    - At $N = 2048$ (typical pretraining): attention FLOPs per layer $\approx 2 \times 32 \times 2048^2 \times 128 \approx 34$ GFLOPs.
      Feed-forward FLOPs per layer $\approx 2 \times 2048 \times 4096 \times 16384 \approx 275$ GFLOPs.
      So attention is ~11% of compute at this length.

    - At $N = 131072$ (128 K context): attention FLOPs per layer scale to $\approx 34 \times (131072/2048)^2 \approx 139$ TFLOPs per layer — more than **500×** larger while the MLP cost only grows linearly.
      Now attention dominates by more than 500× over the MLP at 2048 length.

    This is why long-context models and streaming applications motivate $O(N)$ alternatives.

---

## Linear Attention: The Kernel Trick

The core insight of linear attention, introduced by Katharopoulos et al. (2020), is that the softmax in standard attention is not strictly necessary — it just ensures the output is a weighted sum of values with non-negative, normalized weights. If we can approximate the softmax with a kernel function, we can rewrite the computation to change the order of operations and get $O(N)$ cost.

### The Reformulation

Define a feature map $\phi: \mathbb{R}^{d_k} \rightarrow \mathbb{R}^r$ such that

$$
\exp(q^\top k / \sqrt{d_k}) \approx \phi(q)^\top \phi(k)
$$

Then:

$$
\text{Linear-Attention}(Q, K, V) = \frac{\phi(Q)\left(\phi(K)^\top V\right)}{\phi(Q)\left(\phi(K)^\top \mathbf{1}\right)}
$$

The key: compute $\phi(K)^\top V$ first. This is an $r \times d_v$ matrix that requires $O(N r d_v)$ operations and can be computed once. Then multiplying by $\phi(Q)$ (shape $N \times r$) costs another $O(N r d_v)$. Total is $O(N)$ rather than $O(N^2)$.

The price: vanilla linear attention drops the softmax normalization, losing the "focusing" ability that makes attention heads specialize. Empirically, models trained with simple linear attention tend to underperform transformers on tasks requiring selective copying of distant tokens.

### The Recurrent Form

Linear attention has an equivalent recurrent interpretation that is crucial for efficient inference. Define the outer-product state matrix:

$$
S_t = \sum_{i=1}^{t} \phi(k_i) \otimes v_i \in \mathbb{R}^{r \times d_v}
$$

Then the output at time step $t$ is:

$$
y_t = \frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t}
$$

where $z_t = \sum_{i=1}^t \phi(k_i)$ is a normalizer vector.

This recurrent form processes one token at a time in $O(r d_v)$ operations with $O(r d_v)$ state — analogous to an RNN. During training, the parallel form is faster; during autoregressive inference, the recurrent form is constant cost per step regardless of sequence length. This training-parallel / inference-recurrent duality is a recurring theme in this entire chapter.

```python
import torch
import torch.nn.functional as F

def elu_feature_map(x: torch.Tensor) -> torch.Tensor:
    """ELU+1 feature map: keeps positivity, cheap to compute."""
    return F.elu(x) + 1.0  # shape preserved; always > 0

def linear_attention_parallel(
    Q: torch.Tensor,  # (B, N, H, d_k)
    K: torch.Tensor,  # (B, N, H, d_k)
    V: torch.Tensor,  # (B, N, H, d_v)
    causal: bool = True,
) -> torch.Tensor:
    """
    Parallel (training) form of linear attention.
    For causal (autoregressive) models we cannot simply do K^T V first —
    we need the cumulative version. Here we show the non-causal form for clarity.
    The causal form requires a cumulative outer-product scan.
    """
    B, N, H, d_k = Q.shape
    d_v = V.shape[-1]

    phi_Q = elu_feature_map(Q)  # (B, N, H, d_k)
    phi_K = elu_feature_map(K)  # (B, N, H, d_k)

    # Compute K^T V: (B, H, d_k, d_v)
    # phi_K: (B, N, H, d_k) -> reshape to (B, H, N, d_k)
    phi_K_t = phi_K.permute(0, 2, 1, 3)  # (B, H, N, d_k)
    V_t     = V.permute(0, 2, 1, 3)       # (B, H, N, d_v)
    KV      = torch.einsum('bhnk,bhnv->bhkv', phi_K_t, V_t)  # (B, H, d_k, d_v)

    # Normalizer: sum of phi_K over N
    z = phi_K_t.sum(dim=2)  # (B, H, d_k)

    # Compute output
    phi_Q_t = phi_Q.permute(0, 2, 1, 3)  # (B, H, N, d_k)
    # Numerator: phi(Q) @ KV -> (B, H, N, d_v)
    num = torch.einsum('bhnk,bhkv->bhnv', phi_Q_t, KV)
    # Denominator: phi(Q) @ z -> (B, H, N)
    den = torch.einsum('bhnk,bhk->bhn', phi_Q_t, z).unsqueeze(-1)
    # Clamp for numerical stability
    den = den.clamp(min=1e-6)

    out = (num / den).permute(0, 2, 1, 3)  # back to (B, N, H, d_v)
    return out


def linear_attention_recurrent_step(
    q_t: torch.Tensor,  # (B, H, d_k) — current query
    k_t: torch.Tensor,  # (B, H, d_k) — current key
    v_t: torch.Tensor,  # (B, H, d_v) — current value
    S:   torch.Tensor,  # (B, H, d_k, d_v) — running state
    z:   torch.Tensor,  # (B, H, d_k)       — running normalizer
):
    """Single-step recurrent update — O(d_k * d_v) per step, O(d_k * d_v) memory."""
    phi_q = elu_feature_map(q_t)  # (B, H, d_k)
    phi_k = elu_feature_map(k_t)  # (B, H, d_k)

    # Update state: S += phi_k outer v
    S = S + torch.einsum('bhk,bhv->bhkv', phi_k, v_t)
    z = z + phi_k  # (B, H, d_k)

    # Output
    num = torch.einsum('bhk,bhkv->bhv', phi_q, S)   # (B, H, d_v)
    den = torch.einsum('bhk,bhk->bh', phi_q, z).unsqueeze(-1).clamp(min=1e-6)
    return num / den, S, z
```

---

## Structured State Space Models: S4

State space models (SSMs) originate in control theory. A continuous-time linear SSM maps an input signal $u(t)$ to an output signal $y(t)$ through a hidden state $x(t)$:

$$
\dot{x}(t) = A x(t) + B u(t)
$$

$$
y(t) = C x(t) + D u(t)
$$

where $A \in \mathbb{R}^{N \times N}$, $B \in \mathbb{R}^{N \times 1}$, $C \in \mathbb{R}^{1 \times N}$, and $D$ is a skip connection scalar. The Structured State Space Sequence model (S4), introduced by Gu et al. (2021), discretizes this system and makes $A$ efficiently structured (specifically, a Hippo matrix designed to preserve long-range history) so that the entire sequence can be computed as a convolution in $O(N \log N)$ time during training, or as a recurrence in $O(1)$ per step during inference.

### Discretization

The zero-order hold discretization with step size $\Delta$ gives:

$$
\bar{A} = e^{\Delta A}, \quad \bar{B} = (e^{\Delta A} - I) A^{-1} B, \quad \bar{C} = C
$$

The recurrence becomes:

$$
x_t = \bar{A} x_{t-1} + \bar{B} u_t, \quad y_t = \bar{C} x_t
$$

And the convolutional kernel is $\bar{K} = (\bar{C}\bar{B},\, \bar{C}\bar{A}\bar{B},\, \bar{C}\bar{A}^2\bar{B},\, \ldots)$, enabling $y = u * \bar{K}$ to be computed with an FFT in $O(N \log N)$.

The HiPPO (High-order Polynomial Projection Operator) initialization of $A$ — specifically the LegS HiPPO matrix — is what makes S4 effective at remembering distant tokens. The eigenvalue structure of the HiPPO-LegS matrix means the state $x_t$ projects the history $u_{\leq t}$ onto Legendre polynomials, giving near-optimal compression of the signal over time.

---

## Mamba: Selective State Space Models

S4 and its relatives (S5, DSS, GSS) proved that SSMs can model long sequences efficiently, but they struggled on tasks requiring selective recall — like looking up a specific value from a long context. The state matrix $A$, $B$, $C$ in S4 are fixed for all time steps (content-independent), so the model cannot decide *which* inputs to focus on.

Mamba (Gu & Dao, 2023) introduces **input-dependent (selective) parameters**: $B_t$, $C_t$, and $\Delta_t$ are now functions of the input $u_t$, computed through a linear projection. This selectivity mechanism is the central innovation.

{{fig:ssm-scan}}

### The Selective Scan

The Mamba recurrence for a single channel is:

$$
h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t
$$

$$
y_t = C_t h_t
$$

where $\bar{A}_t = \exp(\Delta_t A)$ and $\bar{B}_t = (\exp(\Delta_t A) - I) A^{-1} B_t$. Now $\Delta_t$, $B_t$, $C_t$ are all computed from the input, giving the model the ability to selectively gate what information flows into the state.

Critically, $A$ is kept diagonal (not the full HiPPO matrix) for efficiency, and initialized with a specific negative-real diagonal structure that encourages stable long-range memory.

```text
Mamba Block (per layer):

Input x: (B, L, D)
         |
    Linear proj → z: (B, L, D)   [skip branch, gated by SiLU]
         |
    Linear proj → u: (B, L, expand*D)   [SSM branch]
         |
    Depthwise conv (width 4)
         |
    SiLU activation
         |
    Selective SSM:
      Linear proj → delta: (B, L, expand*D)
      Linear proj → B:     (B, L, N)      [state dim N ~16]
      Linear proj → C:     (B, L, N)
      Discretize (A fixed diagonal, per-channel)
      Selective scan recurrence
         |
    Multiply by z (gating)
         |
    Linear proj back to D
         |
Output: (B, L, D)
```

The selective scan is the computationally hot path. Naively it is $O(L \cdot D \cdot N)$ with $O(D \cdot N)$ state, but running it sequentially on a GPU is extremely memory-bandwidth-bound. Mamba solves this with a custom parallel scan algorithm (prefix sum / work-efficient parallel scan) and a hardware-aware kernel that fuses the scan, keeps intermediate activations in SRAM, and avoids materializing the full state sequence in HBM.

```python
import torch
import torch.nn as nn
import math

class SelectiveSSM(nn.Module):
    """
    Simplified Mamba-style selective SSM for one feature dimension.
    Full Mamba also uses depthwise conv before this and expand*D channels;
    this illustrates the core selective scan logic.

    d_model: input/output dimension
    d_state: SSM state dimension N (Mamba uses 16)
    """
    def __init__(self, d_model: int, d_state: int = 16, dt_rank: int = None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        dt_rank = dt_rank or math.ceil(d_model / 16)

        # Log of A diagonal: initialized to evenly spaced negative reals
        # A = -exp(log_A), kept negative for stability
        A_log = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)
                          .unsqueeze(0).repeat(d_model, 1))
        self.A_log = nn.Parameter(A_log)  # (d_model, d_state)

        # D skip connection (one per channel)
        self.D = nn.Parameter(torch.ones(d_model))

        # Projections for input-dependent parameters
        self.x_proj = nn.Linear(d_model, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, d_model)
        returns: (B, L, d_model)
        """
        B, L, D = x.shape
        N = self.d_state

        # A: (D, N), kept negative for stability
        A = -torch.exp(self.A_log)  # (D, N)

        # Compute input-dependent B, C, delta
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*N)
        dt, B_ssm, C = x_dbl.split([self.dt_proj.in_features, N, N], dim=-1)
        dt = torch.nn.functional.softplus(self.dt_proj(dt))  # (B, L, D), positive

        # Discretize: zero-order hold
        # dA[t] = exp(dt[t] * A)  — shape (B, L, D, N)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B,L,D,N)
        # dB[t] = dt[t] * B[t] (simplified ZOH for diagonal A)
        dB = dt.unsqueeze(-1) * B_ssm.unsqueeze(2)  # (B, L, D, N)

        # Sequential scan (simple version — production uses parallel scan)
        h = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            # h: (B, D, N), dA[:,t]: (B, D, N), dB[:,t]: (B, D, N)
            # x[:,t]: (B, D)
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            # y: (B, D) via C projection
            y = (h * C[:, t].unsqueeze(2)).sum(-1)  # (B, D)
            ys.append(y)

        # Stack and add skip connection
        y_seq = torch.stack(ys, dim=1)  # (B, L, D)
        y_seq = y_seq + x * self.D.unsqueeze(0).unsqueeze(0)
        return y_seq


class MambaBlock(nn.Module):
    """
    Full Mamba block with expansion, depthwise conv, gating, and SSM.
    expand: expansion factor (Mamba uses 2)
    d_conv: depthwise conv width (Mamba uses 4)
    """
    def __init__(self, d_model: int, d_state: int = 16,
                 expand: int = 2, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand

        # Input projection: split into SSM branch (d_inner) and gate (d_inner)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Causal depthwise convolution over sequence dimension
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,          # left-pad for causality
            groups=self.d_inner,         # depthwise
            bias=True,
        )
        self.act = nn.SiLU()

        # SSM
        self.ssm = SelectiveSSM(self.d_inner, d_state=d_state)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) -> (B, L, d_model)"""
        B, L, _ = x.shape

        # Split into SSM branch u and gate z
        xz = self.in_proj(x)                   # (B, L, 2*d_inner)
        u, z = xz.chunk(2, dim=-1)             # each (B, L, d_inner)

        # Depthwise conv (operates on L dimension)
        u = u.transpose(1, 2)                  # (B, d_inner, L)
        u = self.conv1d(u)[..., :L]            # causal: trim right
        u = u.transpose(1, 2)                  # (B, L, d_inner)
        u = self.act(u)

        # SSM
        y = self.ssm(u)                        # (B, L, d_inner)

        # Gate: multiply by SiLU(z)
        y = y * self.act(z)

        # Output projection
        return self.out_proj(y)                # (B, L, d_model)
```

### Mamba-2 and the State Space Duality

Mamba-2 (Dao & Gu, 2024) reframes the selective SSM as a special case of **structured matrix multiplication**, revealing a duality between SSMs and linear attention. Specifically, with a scalar $A_t$ (instead of a diagonal matrix), the Mamba-2 selective scan is mathematically equivalent to linear attention with a specific data-dependent decay mask. This insight leads to a cleaner algorithm, better hardware utilization (tiled matrix multiplications instead of a sequential scan), and strong theoretical grounding.

The key operation in Mamba-2 is the **SSD (State Space Duality) layer**, which computes:

$$
Y = (L \circ M) V, \quad L_{ij} = C_i^\top h_j \text{ (if using state expansion)}
$$

where $M$ is a lower-triangular mask with elements $M_{ij} = \prod_{k=j+1}^{i} a_k$ (the cumulative product of scalar decays). This can be computed efficiently using block-diagonal structured matrix multiplications.

---

## RWKV: RNNs Strike Back

RWKV (Peng et al., 2023) takes a different philosophical approach: it revisits the recurrent neural network (RNN) architecture but designs it to be trainable in parallel (like a transformer) by expressing the recurrence as a time-weighted attention.

### The WKV Attention

The central operation in RWKV is WKV (time-mixing through weighted key-value aggregation):

$$
\text{wkv}_t = \frac{\sum_{i=1}^{t-1} e^{-(t-1-i)w + k_i} v_i + e^{u + k_t} v_t}{\sum_{i=1}^{t-1} e^{-(t-1-i)w + k_i} + e^{u + k_t}}
$$

Here $w \in \mathbb{R}^d$ is a *learned decay* vector (one per channel), $u \in \mathbb{R}^d$ is a *bonus* for the current token, and $k_t, v_t \in \mathbb{R}^d$ are per-token key and value vectors. This is essentially exponentially-decayed attention, where older tokens get exponentially down-weighted by a channel-wise learned rate.

The recurrent form has a scalar state per channel and can be written as:

$$
a_t = e^{w} a_{t-1} + e^{k_t} v_t, \quad b_t = e^{w} b_{t-1} + e^{k_t}
$$

$$
\text{wkv}_t = \frac{e^{u+k_t} v_t + a_{t-1}}{e^{u+k_t} + b_{t-1}}
$$

This recurrence runs in $O(1)$ per step with $O(d)$ state — vastly smaller state than a transformer's KV cache.

### RWKV Versions

| Version | Key change |
|---------|-----------|
| RWKV-4 | Original architecture; WKV + channel-mixing |
| RWKV-5 | Multi-head WKV; matrix-valued state per head |
| RWKV-6 | Input-dependent time decay (dynamic $w_t$); closer to Mamba |
| RWKV-7 | Revised gating; drops the explicit $u$ bonus; further SSM alignment |

RWKV-6 and RWKV-7 introduce *data-dependent* decays, similar to Mamba's selectivity, and represent a convergence of the two design philosophies.

```python
import torch
import torch.nn as nn

class RWKVTimeMixing(nn.Module):
    """
    RWKV-4 style time mixing block.
    n_embd: model dimension
    layer_id: used to initialize w differently per layer
    """
    def __init__(self, n_embd: int, layer_id: int = 0):
        super().__init__()
        self.n_embd = n_embd

        # Learned time decay (w): initialized to negative values (decay)
        # Stored as log(-w) for stability; w should be < 0
        self.w = nn.Parameter(torch.zeros(n_embd))
        nn.init.uniform_(self.w, -8.0, -0.5)

        # Time-first (u): bonus for current token
        self.u = nn.Parameter(torch.zeros(n_embd))
        nn.init.uniform_(self.u, -0.5, 0.5)

        # Token shift mixing ratios
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, n_embd))
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, n_embd))
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, n_embd))

        # Projections
        self.key    = nn.Linear(n_embd, n_embd, bias=False)
        self.value  = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.output = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, n_embd)
        Token shift: mix current and previous token for k, v, r inputs.
        Then run the WKV scan.
        """
        B, T, C = x.shape

        # Token shift: concat zero-padded version of x (shifted right by 1)
        x_prev = torch.cat([torch.zeros(B, 1, C, device=x.device), x[:, :-1]], dim=1)

        # Compute K, V, R with time-mixed inputs
        xk = x * self.time_mix_k + x_prev * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + x_prev * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + x_prev * (1 - self.time_mix_r)

        k = self.key(xk)        # (B, T, C)
        v = self.value(xv)      # (B, T, C)
        r = torch.sigmoid(self.receptance(xr))  # gating

        # WKV computation (simplified sequential loop over T)
        # In practice this is a custom CUDA kernel for speed
        w = -torch.exp(self.w)   # (C,), negative → exponential decay
        u = self.u               # (C,)

        # Running accumulators
        aa = torch.zeros(B, C, device=x.device)  # numerator state
        bb = torch.zeros(B, C, device=x.device)  # denominator state
        pp = torch.full((B, C), -1e38, device=x.device)  # log-sum for numerical stability

        wkv_outputs = []
        for t in range(T):
            kt = k[:, t]  # (B, C)
            vt = v[:, t]  # (B, C)

            # Numerically stable WKV update using log-space
            qq = torch.maximum(pp + w, kt + u)     # (B, C)
            a  = torch.exp(pp + w - qq) * aa + torch.exp(kt + u - qq) * vt
            b  = torch.exp(pp + w - qq) * bb + torch.exp(kt + u - qq)
            wkv_outputs.append(a / b.clamp(min=1e-8))

            # Update state
            qq2 = torch.maximum(pp + w, kt)
            aa  = torch.exp(pp + w - qq2) * aa + torch.exp(kt - qq2) * vt
            bb  = torch.exp(pp + w - qq2) * bb + torch.exp(kt - qq2)
            pp  = qq2

        wkv = torch.stack(wkv_outputs, dim=1)  # (B, T, C)
        out = r * wkv
        return self.output(out)
```

---

## RetNet and the Retention Mechanism

RetNet (Sun et al., 2023) proposes **Retention** — a recurrence-free, attention-free sequence model that explicitly targets the "training parallelism vs. inference efficiency" tradeoff with what the authors call the "impossible triangle" — the claim that retention achieves all three: training parallelism, $O(1)$ inference, and good performance.

The retention score between query $q_i$ and key $k_j$ is:

$$
\text{Retention}(Q, K, V) = \left(\sum_{m,n} (Q_m K_n^\top) \gamma^{m-n} \cdot \mathbf{1}_{m \geq n}\right) V
$$

where $\gamma \in (0, 1)$ is a per-head decay constant (not learned — fixed at training time based on head index). This can equivalently be written in a parallel (training) form as a lower-triangular masked matrix or in a recurrent (inference) form as an $O(d_k \times d_v)$ state matrix updated at each step.

Multi-scale retention uses different $\gamma$ values per head, allowing different heads to specialize in short-range versus long-range dependencies. Chunk-wise retention computes retention over fixed-size chunks, balancing the efficiency of both forms.

RetNet replaces the softmax normalization with a group normalization applied to the retention output — a practical fix for the scale instability of un-normalized attention.

---

## The FLOP / Memory Trade-off Table

Before we look at hybrids, it is worth summarizing the algorithmic tradeoffs:

| Model | Training FLOPs | Training Memory | Inference per step | State size |
|-------|---------------|-----------------|-------------------|------------|
| Transformer (full attn) | $O(N^2 d)$ | $O(N^2)$ | $O(N d)$ (grows) | $O(N d)$ KV cache |
| FlashAttention Transformer | $O(N^2 d)$ | $O(N d)$ (recomputed) | $O(N d)$ (grows) | $O(N d)$ KV cache |
| Linear Attention | $O(N d^2)$ (causal) | $O(N d)$ | $O(d^2)$ (constant) | $O(d_k \times d_v)$ |
| S4 / Mamba | $O(N d N_s)$ | $O(N d)$ | $O(d N_s)$ (constant) | $O(d N_s)$ |
| RWKV | $O(N d)$ | $O(N d)$ | $O(d)$ (constant) | $O(d)$ |
| RetNet | $O(N d^2)$ (chunk) | $O(N d)$ | $O(d^2)$ (constant) | $O(d_k \times d_v)$ |

$N_s$ denotes SSM state dimension (e.g., 16 in Mamba). Note that Mamba/SSM state size is far smaller than the KV cache of a transformer at long sequences.

!!! example "Worked example: inference memory comparison at 32K context"

    Consider a 7B model, 32 layers, $d = 4096$, 32 heads, $d_k = d_v = 128$, fp16.

    **Transformer KV cache at N=32768:**
    - Per layer: $2 \times N \times d \times 2$ bytes $= 2 \times 32768 \times 4096 \times 2 = 512$ MB
    - 32 layers: $512 \times 32 = 16384$ MB $\approx$ **16 GB** just for the KV cache!

    **Mamba state at any N:**
    - Per layer: $d \times N_s \times 2$ bytes $= 4096 \times 16 \times 2 = 128$ KB
    - 32 layers: $128 \times 32 = 4$ MB — **constant, regardless of N**

    **RWKV state at any N:**
    - Per layer: $d \times 2$ bytes $= 4096 \times 2 = 8$ KB (scalar state per channel)
    - 32 layers: $8 \times 32 = 256$ KB

    This is the fundamental inference memory advantage of SSM/recurrent models: they can handle arbitrarily long sequences at deployment time with a fixed memory footprint.

---

## Hybrid Architectures: Jamba, Zamba & Beyond

Pure SSM models have one well-documented weakness: they struggle at tasks requiring **precise in-context retrieval** — looking up an exact value from the context, copying specific tokens, or doing multi-hop reasoning across distant facts. This is because the fixed-size SSM state is a lossy compression of the history, while softmax attention can attend to exact past tokens.

The natural solution is to combine attention and SSM layers in the same model.

### Jamba (AI21 Labs, 2024)

Jamba interleaves Mamba SSM blocks and transformer (attention + MLP) blocks, with a ratio heavily weighted toward Mamba (roughly 1 attention layer per 8 Mamba layers). It also incorporates MoE (Mixture-of-Experts; see [Mixture-of-Experts (MoE) Architectures](../02-transformer/09-mixture-of-experts.html)) in the MLP blocks to increase model capacity without proportionally increasing compute.

The key design insight: a small number of attention layers provides the exact-retrieval capability that SSMs lack, while the majority of layers being Mamba keeps memory and compute efficient. Jamba demonstrated that this hybrid could match or exceed pure transformer models on standard benchmarks while using far less KV-cache memory at long contexts.

### Zamba and Other Hybrids

Zamba (Zyphra, 2024) uses a 1-attention-per-6-Mamba ratio with a shared global attention block that is applied at multiple depths in the network (parameter sharing). This reduces the number of attention-induced KV cache entries further.

The general design space for hybrids can be parameterized by:
1. **Attention-to-SSM ratio**: how many attention layers vs. SSM layers
2. **Placement**: interleaved evenly, attention at top/bottom, or clustered
3. **Attention type**: full attention, sliding window attention, or grouped-query attention (see [Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html))
4. **SSM variant**: Mamba-1, Mamba-2, GLA, or RWKV blocks

### A Minimal Hybrid Model

```python
import torch
import torch.nn as nn
from typing import Literal

# We assume MambaBlock and a standard TransformerBlock are already defined.
# Below shows how to compose them in a hybrid model.

class HybridBlock(nn.Module):
    """
    A single hybrid layer that is either an SSM block or an Attention block.
    block_type: 'mamba' | 'attention'
    """
    def __init__(
        self,
        d_model: int,
        block_type: Literal["mamba", "attention"],
        n_heads: int = 8,
        d_state: int = 16,
    ):
        super().__init__()
        self.block_type = block_type
        self.norm = nn.RMSNorm(d_model)

        if block_type == "mamba":
            self.block = MambaBlock(d_model, d_state=d_state)
        elif block_type == "attention":
            # Minimal multi-head self-attention
            self.block = nn.MultiheadAttention(
                d_model, n_heads, batch_first=True
            )

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        if self.block_type == "mamba":
            x = self.block(x)
        else:
            x, _ = self.block(x, x, x, attn_mask=attn_mask)
        return residual + x


class HybridLanguageModel(nn.Module):
    """
    A hybrid SSM-Attention language model.
    attn_every: place an attention layer every N layers (rest are Mamba).
    E.g. attn_every=8 → 1 attention layer per 7 Mamba layers.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        attn_every: int = 8,
        n_heads: int = 8,
        d_state: int = 16,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

        layers = []
        for i in range(n_layers):
            # Place attention layers at fixed intervals
            if (i + 1) % attn_every == 0:
                block_type = "attention"
            else:
                block_type = "mamba"
            layers.append(
                HybridBlock(d_model, block_type, n_heads=n_heads, d_state=d_state)
            )

        self.layers = nn.ModuleList(layers)
        self.norm_out = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, L) -> logits (B, L, vocab_size)"""
        x = self.embed(tokens)  # (B, L, d_model)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_out(x)
        return self.lm_head(x)


# Quick sanity check
if __name__ == "__main__":
    model = HybridLanguageModel(
        vocab_size=32000, d_model=512, n_layers=12, attn_every=4
    )
    # Count attention vs Mamba layers
    attn_count  = sum(1 for l in model.layers if l.block_type == "attention")
    mamba_count = sum(1 for l in model.layers if l.block_type == "mamba")
    print(f"Attention layers: {attn_count}, Mamba layers: {mamba_count}")
    # → Attention layers: 3, Mamba layers: 9

    tokens = torch.randint(0, 32000, (2, 128))
    logits = model(tokens)
    print(f"Output shape: {logits.shape}")  # (2, 128, 32000)
```

---

## Gated Linear Attention and the Convergence

The theoretical connections between all these architectures have become clearer over 2024-2025. The **Gated Linear Attention (GLA)** formulation (Yang et al., 2023) provides a unified view: all of these models can be seen as linear attention variants with different forms of *gating* and *decay*.

The general GLA recurrence is:

$$
S_t = G_t \odot S_{t-1} + k_t^\top v_t
$$

$$
o_t = q_t S_t
$$

where $G_t \in (0,1)^{d_k \times d_v}$ is a data-dependent gate (or decay) matrix. Different choices of $G_t$ recover different architectures:

- $G_t = \text{diag}(\gamma)$ (constant) → RetNet
- $G_t = \text{diag}(\exp(-\exp(w)))$ (input-dependent scalar per channel) → RWKV-style
- $G_t = \text{diag}(\exp(\Delta_t A))$ (Mamba-style discretization) → Mamba / selective SSM
- $G_t = I$ (no decay) → linear attention

This convergence has practical implications: hardware-efficient kernels developed for one variant (e.g., GLA's chunked parallel scan) can be adapted for others, and model designers can tune the gate expressiveness/cost tradeoff without changing the high-level architecture.

### HGRN and Hawk/Griffin

Other notable entries in this space include:

- **HGRN / HGRN-2** (Qin et al.): Hierarchical Gated Recurrent Network, using input-dependent forget gates similar to LSTM but designed for parallelism.
- **Hawk / Griffin** (De et al., Google DeepMind, 2024): A hybrid recurrent + local-attention model with a "Real Gated Linear Recurrence" (RGLR) layer that has been trained at the scale of multi-billion parameter models, demonstrating competitive performance with transformers.

---

## Where the Frontier Is Heading

As of mid-2026, the field has reached some pragmatic conclusions:

**Hybrids win in practice.** Pure SSM models have not displaced transformers in production LLMs. The combination of a small fraction of attention layers with SSM/linear-attention layers appears to offer the best tradeoff: the exact-retrieval capability of attention for a small fraction of total compute, with the memory efficiency of SSMs for the bulk of the sequence processing.

**The long-context use case is where alternatives shine.** For standard (up to 8K token) language modeling, modern transformers with FlashAttention are hard to beat. The advantage of SSM and linear-attention models grows dramatically as context exceeds 32K tokens.

**Training dynamics differ.** SSMs and RWKV models have different sensitivity to hyperparameters than transformers. The learning rate schedule, initialization, and gradient clipping values that work for transformers often need significant adjustment for recurrent models.

**Scaling laws are being remeasured.** Early (2022-2023) Chinchilla-style scaling laws were derived for transformers. Whether SSMs follow the same laws is an open research question. Preliminary results suggest competitive scaling, but the optimal model size / data tradeoff may differ.

**Hardware matters.** Modern GPUs are optimized for dense matrix multiplication — the operation that attention and dense linear layers perform. SSM-specific operations (sequential scans, depthwise convolutions) are less naturally mapped to tensor cores. This gives transformers an implementation advantage that partially offsets the algorithmic advantage of SSMs at long sequences. Future ASICs or NPUs designed for recurrent inference could shift this calculus.

!!! interview "Interview Corner"

    **Q:** Mamba uses a selective scan where $B_t$, $C_t$, $\Delta_t$ are input-dependent. Why is this selectivity important, and what would happen without it?

    **A:** Without selectivity (as in S4), the state transition matrices are fixed — every input updates the hidden state identically regardless of content. This means the model cannot selectively ignore irrelevant tokens or precisely copy specific values from context. In practice, content-independent SSMs underperform on tasks like selective copying, associative recall, or multi-hop lookup — tasks where transformers excel because softmax attention can precisely attend to specific past positions. Mamba's selectivity allows the model to set $\Delta_t \approx 0$ for irrelevant tokens (effectively not updating the state) and set $\Delta_t$ large for important tokens (strongly writing them into the state). This recovers much of the associative recall capability while maintaining the $O(1)$-per-step inference cost. Empirically, selective SSMs show much smaller gaps versus transformers on in-context learning benchmarks compared to non-selective SSMs.

!!! warning "Common pitfall: confusing training and inference complexity"

    A common mistake is to say "Mamba is $O(N)$" without qualification. During training with parallel scan, Mamba is $O(N \cdot D \cdot N_s)$ FLOPs — linear in sequence length, which is better than the $O(N^2)$ of attention for long $N$, but not negligible. During inference (autoregressive generation), Mamba is $O(D \cdot N_s)$ per step — *constant*, independent of how many tokens have been generated. The transformer at inference is $O(N \cdot D)$ per step because each new token must attend to all $N$ previous KV entries. So the inference advantage grows linearly with sequence length.

---

## Key Takeaways

!!! key "Key Takeaways"

    - Standard softmax attention is $O(N^2)$ in both FLOPs and memory, making it a bottleneck for sequences beyond tens of thousands of tokens.
    - Linear attention rewrites the attention computation using a kernel feature map, enabling $O(N)$ parallel training and $O(1)$ per-step inference, at the cost of reduced "focusing" ability.
    - S4 introduced the HiPPO matrix for stable long-range memory via an SSM; Mamba extended this with input-dependent (selective) parameters, allowing the model to filter irrelevant inputs.
    - RWKV achieves RNN-like inference cost with transformer-like training by expressing attention as an exponentially-decayed weighted sum, trainable in parallel via log-space prefix scans.
    - RetNet uses a fixed decay $\gamma$ per head, enabling three equivalent computation forms: parallel (training), recurrent (inference), and chunkwise (balanced).
    - Mamba-2/SSD and GLA reveal that SSMs, linear attention, and RWKV-style models are instances of the same gated linear recurrence framework, differing only in how the gate $G_t$ is parameterized.
    - Hybrid architectures (Jamba, Zamba, Griffin) that interleave a small number of attention layers with many SSM layers currently represent the practical state of the art: combining exact-retrieval capability with long-context memory efficiency.
    - The SSM inference advantage is most pronounced at context lengths exceeding 32K tokens, where the KV cache of a transformer can consume tens of gigabytes of memory while an SSM's state stays constant.

---

!!! sota "State of the Art & Resources (2026)"
    SSMs and linear-attention alternatives have rapidly matured from research curiosities into production-ready components. Hybrid architectures that mix a small fraction of full attention layers with Mamba or GLA blocks now represent the practical frontier, matching transformer quality at a fraction of the KV-cache cost for long contexts.

    **Foundational work**

    - [Katharopoulos et al., *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention* (2020)](https://arxiv.org/abs/2006.16236) — introduced the kernel-feature-map reformulation that enables O(N) training and O(1) inference.
    - [Gu et al., *Efficiently Modeling Long Sequences with Structured State Spaces (S4)* (2021)](https://arxiv.org/abs/2111.00396) — established the HiPPO-initialized SSM as a competitive sequence model with O(N log N) training via convolution.
    - [Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* (2023)](https://arxiv.org/abs/2312.00752) — added input-dependent selectivity to SSMs, closing the recall gap with transformers on associative tasks.

    **Recent advances (2023–2026)**

    - [Dao & Gu, *Transformers are SSMs: Structured State Space Duality (Mamba-2)* (2024)](https://arxiv.org/abs/2405.21060) — unified SSMs and linear attention under the SSD framework; enables 2–8× faster training via tiled matmuls.
    - [Yang et al., *Gated Linear Attention Transformers with Hardware-Efficient Training* (2023)](https://arxiv.org/abs/2312.06635) — general GLA framework that subsumes RetNet, RWKV, and Mamba as special cases; ships a CUDA/Triton implementation.
    - [De et al. (Google DeepMind), *Griffin: Mixing Gated Linear Recurrences with Local Attention* (2024)](https://arxiv.org/abs/2402.19427) — demonstrates that Real Gated Linear Recurrences with sparse local attention match or beat transformers at multi-billion parameter scale.
    - [Lieber et al. (AI21 Labs), *Jamba: A Hybrid Transformer-Mamba Language Model* (2024)](https://arxiv.org/abs/2403.19887) — first large-scale hybrid Mamba+MoE model; fits in 80 GB while handling 256 K context.
    - [Peng et al., *RWKV: Reinventing RNNs for the Transformer Era* (2023)](https://arxiv.org/abs/2305.13048) — WKV time-mixing enables transformer-quality LLMs with O(1) constant-size inference state.

    **Open-source & tools**

    - [state-spaces/mamba](https://github.com/state-spaces/mamba) — official Mamba/Mamba-2 implementation with hardware-aware selective-scan CUDA kernels and pretrained checkpoints.
    - [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM) — official RWKV training codebase (now at RWKV-7 "Goose"); Linux Foundation AI project.
    - [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) — Triton-based library with hardware-efficient kernels for GLA, Mamba, RetNet, and hybrid architectures across NVIDIA/AMD/Intel.

    **Go deeper**

    - [Tri Dao, *State Space Duality (Mamba-2) Part I — The Model* (2024)](https://tridao.me/blog/2024/mamba2-part1-model/) — the authors' own blog walkthrough of the SSD theory and how Mamba-2 achieves faster training through the SSM–linear-attention duality.
    - [Ayonrinde, *Mamba Explained* — The Gradient (2024)](https://thegradient.pub/mamba-explained/) — accessible conceptual overview of Mamba's selective scan and why selectivity is the key innovation over S4.

## Further Reading

- Gu, A., Goel, K., & Ré, C. (2021). **Efficiently Modeling Long Sequences with Structured State Spaces (S4).** ICLR 2022.
- Gu, A., & Dao, T. (2023). **Mamba: Linear-Time Sequence Modeling with Selective State Spaces.** arXiv:2312.00752.
- Dao, T., & Gu, A. (2024). **Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (Mamba-2).** ICML 2024.
- Peng, B., et al. (2023). **RWKV: Reinventing RNNs for the Transformer Era.** EMNLP 2023 Findings.
- Sun, Y., et al. (2023). **Retentive Network: A Successor to Transformer for Large Language Models (RetNet).** arXiv:2307.08621.
- Katharopoulos, A., et al. (2020). **Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.** ICML 2020.
- Yang, S., et al. (2023). **Gated Linear Attention Transformers with Hardware-Efficient Training.** arXiv:2312.06635.
- De, S., et al. (Google DeepMind, 2024). **Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models.** arXiv:2402.19427.
- AI21 Labs (2024). **Jamba: A Hybrid Transformer-Mamba Language Model.** arXiv:2403.19887.
- GitHub: **state-spaces/mamba** — official Mamba implementation and selective scan CUDA kernels.
- GitHub: **BlinkDL/RWKV-LM** — official RWKV training code.
