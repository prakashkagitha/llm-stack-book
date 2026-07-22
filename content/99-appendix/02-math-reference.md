#  The Math Reference Sheet

This appendix collects every core equation used across the LLM stack in one printable location. For each formula you get: the expression in display math, a one-line meaning reminder, and the key shape conventions. When you want a full derivation or motivation, the cross-links point you to the chapter where it lives. Think of this as the cheat sheet you tape to your monitor during a training run.

---

## 1. Core Neural Network Building Blocks

### Softmax

$$
\operatorname{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}
$$

**Meaning.** Converts a vector of raw logits $z \in \mathbb{R}^V$ into a probability distribution. Outputs are positive and sum to one.

**Numerically stable form.** In practice we subtract the maximum before exponentiating to avoid overflow:

$$
\operatorname{softmax}(z)_i = \frac{e^{z_i - \max_j z_j}}{\sum_{j} e^{z_j - \max_j z_j}}
$$

```python
import torch
import torch.nn.functional as F

def safe_softmax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Numerically stable softmax.
    Subtracting the max is equivalent — the shift cancels in numerator
    and denominator — but prevents exp() overflow when z is large.

    z: any shape, we reduce along `dim`.
    """
    z_shifted = z - z.max(dim=dim, keepdim=True).values  # broadcast-safe
    exp_z = torch.exp(z_shifted)
    return exp_z / exp_z.sum(dim=dim, keepdim=True)

# Quick sanity check
logits = torch.tensor([2.0, 1.0, 0.1])
probs  = safe_softmax(logits)
assert torch.allclose(probs.sum(), torch.tensor(1.0))
print(probs)  # tensor([0.6590, 0.2424, 0.0986])
```

See [Probability, Statistics & Information Theory](../01-foundations/02-probability-information.html) for the full information-theoretic derivation.

---

### Cross-Entropy Loss

$$
\mathcal{L}_{\text{CE}} = -\sum_{i=1}^{V} y_i \log \hat{p}_i
$$

For the **language modelling** setting where $y$ is a one-hot target token $t$:

$$
\mathcal{L}_{\text{CE}} = -\log \hat{p}_t = -\log \operatorname{softmax}(z)_t = \log\!\sum_j e^{z_j} - z_t
$$

The second form (log-sum-exp minus the correct logit) is the numerically preferred implementation and equals what `torch.nn.CrossEntropyLoss` computes directly from logits.

**Perplexity** is the exponential of the mean cross-entropy per token:

$$
\text{PPL} = \exp\!\left(\frac{1}{N}\sum_{n=1}^{N} \mathcal{L}_n\right)
$$

A model with PPL 20 needs on average 20 equally-likely guesses to land on the correct next token.

```python
import torch
import torch.nn as nn

# Internally: F.log_softmax + F.nll_loss, numerically fused
criterion = nn.CrossEntropyLoss(reduction='mean')

# logits: [batch, vocab];  targets: [batch] of integer token ids
logits  = torch.randn(4, 32_000)   # e.g. Llama-3 vocab
targets = torch.randint(0, 32_000, (4,))

loss = criterion(logits, targets)
perplexity = torch.exp(loss)
print(f"loss={loss:.4f}  PPL={perplexity:.1f}")
```

See [The Pretraining Objective & Loss](../03-pretraining/03-pretraining-objective.html) for the sequence-level formulation with causal masking.

---

### Layer Normalization

$$
\operatorname{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta
$$

where $\mu = \frac{1}{d}\sum_i x_i$, $\sigma^2 = \frac{1}{d}\sum_i (x_i - \mu)^2$, and $\gamma, \beta \in \mathbb{R}^d$ are learned scale and bias.

### RMSNorm

$$
\operatorname{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \varepsilon}}
$$

**Meaning.** RMSNorm drops the mean-centering step. It is used in Llama 2/3 and most modern decoders because it is faster (fewer ops, no mean subtraction) and empirically just as effective. There is no bias $\beta$.

```python
import torch, math

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    x     : (..., d_model)
    weight: (d_model,) — the learnable scale γ
    """
    # Mean of squares along the last dimension
    rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    return (x / rms) * weight

# Test against PyTorch's built-in
d = 512
x = torch.randn(2, 16, d)
w = torch.ones(d)

out_custom = rms_norm(x, w)
out_builtin = torch.nn.functional.rms_norm(x, (d,), w, eps=1e-6)
assert torch.allclose(out_custom, out_builtin, atol=1e-5)
```

See [The Transformer Block: Norms, Residuals, MLPs & Activations](../02-transformer/06-transformer-block.html).

---

## 2. Attention

### Scaled Dot-Product Attention

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

**Shapes.** $Q \in \mathbb{R}^{T_q \times d_k}$, $K \in \mathbb{R}^{T_k \times d_k}$, $V \in \mathbb{R}^{T_k \times d_v}$. Output has shape $T_q \times d_v$.

**Why $\sqrt{d_k}$?** The dot product $q^\top k$ has variance proportional to $d_k$ when entries are standard normal. Dividing by $\sqrt{d_k}$ restores unit variance and keeps softmax in a sensible gradient regime.

**Causal mask.** For autoregressive decoding, add $-\infty$ to positions where $j > i$ before the softmax so token $i$ cannot attend to future tokens.

### Multi-Head Attention (MHA)

$$
\text{MHA}(Q,K,V) = \operatorname{concat}(\text{head}_1,\ldots,\text{head}_h) W^O
$$

$$
\text{head}_i = \operatorname{Attention}(Q W_i^Q,\; K W_i^K,\; V W_i^V)
$$

where $W_i^Q, W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$, and $W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$.

**Grouped Query Attention (GQA).** Use $g$ KV heads shared across $h$ query heads where $g \ll h$. KV cache shrinks by factor $h/g$.

```python
import math
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(
    q: torch.Tensor,  # (B, H, T, d_k)
    k: torch.Tensor,  # (B, H, T, d_k)
    v: torch.Tensor,  # (B, H, T, d_v)
    causal: bool = True,
) -> torch.Tensor:
    """
    Pure-PyTorch reference implementation.
    In production use F.scaled_dot_product_attention (dispatches to FlashAttention).
    """
    d_k = q.size(-1)
    # (B, H, T_q, T_k)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if causal:
        T = scores.size(-1)
        mask = torch.triu(torch.ones(T, T, device=q.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

    attn = F.softmax(scores, dim=-1)      # (B, H, T_q, T_k)
    return torch.matmul(attn, v)          # (B, H, T_q, d_v)
```

See [The Attention Mechanism From Scratch](../02-transformer/03-attention-from-scratch.html) and [Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html).

---

### Rotary Position Embedding (RoPE)

RoPE rotates the query and key vectors before the dot product. For a 2-D pair $(x_{2i}, x_{2i+1})$ at position $m$:

$$
\begin{bmatrix} x_{2i}' \\ x_{2i+1}' \end{bmatrix}
=
\begin{bmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{bmatrix}
\begin{bmatrix} x_{2i} \\ x_{2i+1} \end{bmatrix}
$$

where $\theta_i = b^{-2i/d}$ with base $b = 10{,}000$ (original) or larger values (e.g. $b = 500{,}000$ for long-context Llama-3).

The inner product between rotated query at position $m$ and rotated key at position $n$ depends only on $m - n$, giving **relative** position encoding without any extra parameters.

```python
import torch

def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    x         : (B, T, H, d_head)
    freqs_cis : (T, d_head/2) complex tensor = cos + i*sin

    Applies RoPE by treating consecutive pairs of dimensions as complex numbers.
    """
    # Reshape to complex: (B, T, H, d_head/2) complex64
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # freqs_cis: (1, T, 1, d_head/2) after unsqueeze for broadcast
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    # Element-wise complex multiply = rotation
    x_rotated = x_complex * freqs_cis
    return torch.view_as_real(x_rotated).flatten(-2).type_as(x)

def build_rope_freqs(seq_len: int, d_head: int, base: float = 10_000.0) -> torch.Tensor:
    """Returns (seq_len, d_head/2) complex tensor of e^{i*m*theta_i}."""
    i  = torch.arange(0, d_head, 2).float()
    theta = 1.0 / (base ** (i / d_head))                # (d_head/2,)
    m  = torch.arange(seq_len).float()                  # (T,)
    freqs = torch.outer(m, theta)                        # (T, d_head/2)
    return torch.polar(torch.ones_like(freqs), freqs)    # complex
```

See [Positional Encodings: Sinusoidal, Learned, RoPE & ALiBi](../02-transformer/05-positional-encoding.html).

---

## 3. Optimizers

### AdamW

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

$$
\theta_t = \theta_{t-1} - \alpha \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon} + \lambda\, \theta_{t-1}\right)
$$

**Meaning.** Adam with **decoupled weight decay** $\lambda$. The weight-decay term $\lambda \theta$ is applied directly to the parameters *before* the gradient step, not folded into the gradient — this is the "W" in AdamW and it is the correct way to do $L_2$ regularization with adaptive optimizers.

**Typical hyperparameters for LLM pretraining:** $\beta_1 = 0.9$, $\beta_2 = 0.95$ (not 0.999), $\varepsilon = 10^{-8}$, $\lambda = 0.1$, cosine learning-rate schedule.

```python
import torch

def adamw_step(
    param:  torch.Tensor,   # parameter θ
    grad:   torch.Tensor,   # gradient g_t
    m:      torch.Tensor,   # first moment (in-place)
    v:      torch.Tensor,   # second moment (in-place)
    t:      int,            # step count (1-indexed)
    lr:     float = 3e-4,
    beta1:  float = 0.9,
    beta2:  float = 0.95,
    eps:    float = 1e-8,
    wd:     float = 0.1,
) -> None:
    """Single AdamW update step. Modifies param, m, v in-place."""
    # Bias-correct moments
    m.mul_(beta1).add_(grad, alpha=1 - beta1)
    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    # Weight decay applied directly to param, then Adam step
    param.mul_(1 - lr * wd)
    param.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)
```

See [Optimizers: SGD, Adam, Adafactor, Lion, Muon & Shampoo](../03-pretraining/09-optimizers.html).

---

## 4. Scaling Laws

### Kaplan et al. Scaling Law (2020)

$$
L(N, D) \approx \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D}
$$

where $N$ = parameters, $D$ = training tokens, $\alpha_N \approx 0.076$, $\alpha_D \approx 0.095$ (OpenAI estimates), and $N_c, D_c$ are dataset- and task-specific constants.

**Key finding:** scale $N$ faster than $D$; for a given compute budget $C$ the loss-optimal model is larger and undertrained.

### Hoffmann et al. (Chinchilla, 2022)

$$
L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}
$$

with $\alpha \approx 0.34$, $\beta \approx 0.28$ (fitted on language tasks). Minimising loss for fixed compute $C = 6ND$ gives:

$$
N_{\text{opt}} \propto C^{0.5}, \quad D_{\text{opt}} \propto C^{0.5}
$$

meaning **parameters and tokens should scale in a roughly 1:20 ratio** (20 tokens per parameter). A 70 B-parameter model is compute-optimal at about 1.4 T tokens.

### Compute Budget

$$
C \approx 6 N D \quad \text{(FLOPs)}
$$

The factor 6 accounts for 2 multiply-add ops per parameter per token in the forward pass, doubled for the backward pass.

!!! example "Worked example: Chinchilla-optimal 7 B model"

    Suppose you have a budget of $C = 7 \times 10^{22}$ FLOPs (roughly what it takes to train a 7 B parameter model with 2 T tokens on a cluster of A100s for a few weeks).

    Using $C = 6ND$:
    $$D_{\text{opt}} = \frac{C}{6 N} = \frac{7\times10^{22}}{6 \times 7\times10^9} \approx 1.67 \times 10^{12} \approx 1.67\text{ T tokens}$$

    Chinchilla says optimal $D/N \approx 20$, so for $N = 7\text{ B}$, optimal $D = 140\text{ B tokens}$ — but modern practice (Llama 2, Llama 3) deliberately over-trains far past the compute-optimal point because inference cost matters more at deployment.

See [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html).

---

## 5. LoRA and PEFT

### LoRA (Low-Rank Adaptation)

For a weight matrix $W_0 \in \mathbb{R}^{d \times k}$, freeze $W_0$ and inject a low-rank perturbation:

$$
W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B A
$$

where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, rank $r \ll \min(d,k)$, and $\alpha$ is a scaling hyperparameter.

**Initialisation.** $A$ is random Gaussian; $B$ is zero — so $\Delta W = 0$ at the start of fine-tuning, preserving the pretrained forward pass.

**Parameter savings.** Instead of training $dk$ parameters we train $r(d+k)$. For $d = k = 4096$ and $r = 16$: full fine-tuning costs $16.8\text{ M}$ params per matrix; LoRA costs $2 \times 4096 \times 16 = 131{,}072$ — a $128\times$ reduction.

**Merged form at inference.** Because the output is $(W_0 + BA)x$, we can merge $W \leftarrow W_0 + BA$ before deployment — zero extra latency.

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with LoRA.
    The base weight W0 is frozen; only A and B are trained.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        lora_alpha: float = 16.0,
        bias: bool = False,
    ):
        super().__init__()
        self.r = r
        self.scaling = lora_alpha / r

        # Frozen pretrained weight
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Trainable LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B stays zero → ΔW = 0 at init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base forward + LoRA delta, scaled
        base_out  = nn.functional.linear(x, self.weight, self.bias)
        lora_out  = nn.functional.linear(x, self.lora_A)        # (..., r)
        lora_out  = nn.functional.linear(lora_out, self.lora_B) # (..., out)
        return base_out + self.scaling * lora_out

    def merge(self) -> nn.Linear:
        """Collapse LoRA into base weight for zero-overhead inference."""
        merged_w = self.weight + self.scaling * (self.lora_B @ self.lora_A)
        lin = nn.Linear(
            self.weight.size(1), self.weight.size(0),
            bias=self.bias is not None
        )
        lin.weight = nn.Parameter(merged_w)
        if self.bias is not None:
            lin.bias = nn.Parameter(self.bias.clone())
        return lin
```

See [PEFT I: LoRA, QLoRA, DoRA & The Adapter Family](../05-posttraining-alignment/03-peft-lora-qlora.html) and [Memory-Efficient Training: Checkpointing, Offloading & LoRA Math](../04-kernels-efficiency/10-memory-efficient-training.html).

---

## 6. Reinforcement Learning Losses

### PPO Clipped Surrogate Objective

The Proximal Policy Optimization (PPO) objective for a language-model policy $\pi_\theta$:

$$
\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\, A_t,\; \operatorname{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\, A_t\right)\right]
$$

where:

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}
$$

is the **importance-sampling ratio** and $A_t$ is the advantage estimate. $\varepsilon = 0.2$ is standard.

**Meaning.** Maximise expected advantage but refuse to let the policy move too far from the reference. The clip prevents large destructive updates.

**KL Penalty variant** (used in early RLHF):

$$
\mathcal{L}^{\text{KL}}(\theta) = \mathbb{E}\!\left[r_t A_t\right] - \beta\, D_{\text{KL}}\!\left(\pi_\theta \;\|\; \pi_{\text{ref}}\right)
$$

```python
import torch

def ppo_loss(
    log_probs_new:  torch.Tensor,  # (B,) log π_θ(a|s)
    log_probs_old:  torch.Tensor,  # (B,) log π_old(a|s)
    advantages:     torch.Tensor,  # (B,) advantage estimates A_t
    clip_eps:       float = 0.2,
    reduce:         bool  = True,
) -> torch.Tensor:
    """
    PPO clipped surrogate loss.
    We *maximise* this, so negate before calling .backward().
    """
    ratio  = torch.exp(log_probs_new - log_probs_old)  # importance ratio
    # Clipped objective
    surr1  = ratio * advantages
    surr2  = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
    loss   = torch.min(surr1, surr2)  # take the pessimistic bound
    return loss.mean() if reduce else loss
```

See [Policy Gradients & PPO for Language Models](../05-posttraining-alignment/06-ppo-for-llms.html).

---

### GRPO (Group Relative Policy Optimisation)

GRPO replaces the value critic with a group-normalised reward baseline. For a group of $G$ responses $\{o_1, \ldots, o_G\}$ sampled from the old policy for the same prompt:

$$
A_i = \frac{r_i - \operatorname{mean}(\{r_j\}_{j=1}^G)}{\operatorname{std}(\{r_j\}_{j=1}^G)}
$$

The policy gradient objective over token positions $t$ in response $i$:

$$
\mathcal{L}^{\text{GRPO}}(\theta) = \frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \left[\min(r_{i,t} A_i,\; \text{clip}(r_{i,t}, 1-\varepsilon, 1+\varepsilon) A_i) - \beta\, D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})\right]
$$

**Meaning.** The reward within the group acts as its own baseline — no separate critic network needed. Used in DeepSeek-R1 and many reasoning-focused RLVR pipelines.

```python
import torch

def grpo_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """
    rewards : (G,) — one scalar reward per response in the group
    Returns : (G,) normalised advantages
    """
    mean = rewards.mean()
    std  = rewards.std(unbiased=False).clamp(min=1e-8)
    return (rewards - mean) / std

# Example: 8 responses with rewards from a verifier
rewards = torch.tensor([1.0, 0.0, 1.0, 0.5, 0.0, 1.0, 0.5, 0.0])
adv = grpo_advantages(rewards)
print(adv.round(decimals=2))
# tensor([ 1.15, -1.15,  1.15,  0.00, -1.15,  1.15,  0.00, -1.15])
```

See [GRPO, RLOO & Critic-Free RL](../05-posttraining-alignment/08-grpo-rloo.html) and [Advantage Estimation, KL Control & Stability Tricks](../06-rl-infra/09-advantage-kl-tricks.html).

---

### DPO (Direct Preference Optimisation)

DPO closes the loop between the reward model and policy update. Given a preference pair $(y_w, y_l)$ where $y_w$ is preferred over $y_l$:

$$
\mathcal{L}^{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)}\!\left[\log \sigma\!\left(\beta \log\frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log\frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]
$$

**Meaning.** The model is rewarded for increasing the likelihood of preferred completions *relative to the reference* while decreasing that of dispreferred completions. $\beta$ (typically 0.1–0.5) controls how tightly the policy stays near the reference.

**Log-ratio shorthand:**

$$
h_\theta(y, x) \triangleq \log\frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}
$$

$$
\mathcal{L}^{\text{DPO}} = -\mathbb{E}\!\left[\log \sigma\!\left(\beta \left(h_\theta(y_w, x) - h_\theta(y_l, x)\right)\right)\right]
$$

```python
import torch
import torch.nn.functional as F

def dpo_loss(
    log_probs_theta_w:  torch.Tensor,  # (B,) log π_θ(y_w | x)
    log_probs_theta_l:  torch.Tensor,  # (B,) log π_θ(y_l | x)
    log_probs_ref_w:    torch.Tensor,  # (B,) log π_ref(y_w | x)
    log_probs_ref_l:    torch.Tensor,  # (B,) log π_ref(y_l | x)
    beta:               float = 0.1,
) -> torch.Tensor:
    """
    Standard DPO loss (Rafailov et al. 2023).
    Each log_prob is the per-sequence sum of token log-probs.
    """
    log_ratio_w = log_probs_theta_w - log_probs_ref_w  # h_θ(y_w, x)
    log_ratio_l = log_probs_theta_l - log_probs_ref_l  # h_θ(y_l, x)
    # Positive when preferred response has higher relative log-prob
    gap  = beta * (log_ratio_w - log_ratio_l)
    loss = -F.logsigmoid(gap)  # = log(1 + exp(-gap)), numerically stable
    return loss.mean()
```

See [Direct Preference Optimization & Its Variants](../05-posttraining-alignment/07-dpo-and-variants.html).

---

## 7. KL Divergence Estimators

The full KL divergence between a learned policy $\pi$ and reference $\pi_{\text{ref}}$ is:

$$
D_{\text{KL}}(\pi \| \pi_{\text{ref}}) = \mathbb{E}_{a \sim \pi}\!\left[\log\frac{\pi(a)}{\pi_{\text{ref}}(a)}\right]
$$

In practice we estimate it from a single action sample. Three common Monte Carlo estimators, all unbiased or approximately unbiased:

| Estimator | Formula | Notes |
|-----------|---------|-------|
| **k1** (naive) | $\log r$ where $r = \pi / \pi_{\text{ref}}$ | Biased toward negative values; noisy |
| **k2** | $r - 1 - \log r$ | Always $\geq 0$; more stable |
| **k3** | $\frac{(r-1)^2}{2}$ | Second-order Taylor approx; cheapest |

**k2 is the standard choice** in RLHF systems (used in TRL, OpenRLHF). It lower-bounds the true KL by Jensen's inequality.

```python
import torch

def kl_estimators(
    log_pi:     torch.Tensor,  # (B, T) log-probs under current policy
    log_pi_ref: torch.Tensor,  # (B, T) log-probs under reference policy
) -> dict:
    """
    Returns three scalar KL estimates averaged over batch and token positions.
    """
    log_r = log_pi - log_pi_ref  # log importance ratio  (B, T)
    r     = log_r.exp()

    k1 = log_r                   # naive, can be negative
    k2 = r - 1.0 - log_r        # always >= 0 by log inequality
    k3 = (r - 1.0).pow(2) / 2   # Taylor approximation

    return {
        "k1": k1.mean().item(),
        "k2": k2.mean().item(),
        "k3": k3.mean().item(),
    }

# Sanity: for identical policies KL should be ~0
lp = torch.full((4, 512), -3.0)
print(kl_estimators(lp, lp))
# {'k1': 0.0, 'k2': 0.0, 'k3': 0.0}
```

See [Advantage Estimation, KL Control & Stability Tricks](../06-rl-infra/09-advantage-kl-tricks.html).

---

## 8. Quick-Reference Table

The table below lists every major equation in one place for rapid lookup.

| Name | Equation (compact) | Key shapes/params |
|------|--------------------|-------------------|
| Softmax | $p_i = e^{z_i}/\sum_j e^{z_j}$ | $z \in \mathbb{R}^V$ |
| Cross-Entropy | $\mathcal{L} = -\log p_t$ | scalar per token |
| Perplexity | $\exp(\bar{\mathcal{L}})$ | scalar per sequence |
| LayerNorm | $\gamma(x-\mu)/\sigma + \beta$ | learnable $\gamma, \beta \in \mathbb{R}^d$ |
| RMSNorm | $\gamma x / \text{RMS}(x)$ | no $\beta$, no mean |
| Attention | $\operatorname{softmax}(QK^\top/\sqrt{d_k})V$ | $Q,K: T\times d_k$; $V: T\times d_v$ |
| RoPE | complex rotation $x_m e^{im\theta}$ | base $b$, position $m$ |
| AdamW | $\theta \leftarrow \theta(1-\alpha\lambda) - \alpha \hat{m}/(\sqrt{\hat{v}}+\varepsilon)$ | $\beta_1{=}0.9, \beta_2{=}0.95$ |
| Scaling (Chinchilla) | $L = E + A/N^\alpha + B/D^\beta$ | $\alpha{\approx}0.34, \beta{\approx}0.28$ |
| Compute budget | $C \approx 6ND$ | FLOPs |
| LoRA | $W = W_0 + \frac{\alpha}{r}BA$ | $r \ll \min(d,k)$, $B=0$ at init |
| PPO | $\min(rA, \operatorname{clip}(r,1{\pm}\varepsilon)A)$ | $\varepsilon{=}0.2$ |
| GRPO advantage | $(r_i - \bar{r})/\sigma_r$ | group of $G$ responses |
| DPO | $-\log\sigma(\beta(h_w - h_l))$ | $\beta{\approx}0.1$ |
| KL (k2) | $r - 1 - \log r$ | $r = \pi/\pi_{\text{ref}}$ |

---

!!! interview "Interview Corner"

    **Q:** During RLHF with PPO, your KL penalty term keeps exploding after a few hundred steps even though you're clipping the PPO ratio. What are the likely causes and how would you fix them?

    **A:** Several things can cause runaway KL despite PPO clipping. First, check whether you are computing the KL **per token averaged over the sequence** or summing it — summing scales with sequence length and can dominate long completions. Second, verify the reference policy is truly frozen; if you accidentally pass gradients through it the reference drifts and the KL penalty is meaningless. Third, PPO clipping only constrains the probability ratio at *sampled* tokens; if the policy confidently shifts probability mass onto *unsampled* tokens the effective KL can still blow up — the k2 estimator $r - 1 - \log r$ catches this. Fourth, check for numerical instability in log-probs: with FP16, very low probability tokens can underflow to $-\infty$; use bfloat16 or clamp log-probs to a safe floor. Finally, the KL coefficient $\beta$ may simply be too small — an adaptive controller that scales $\beta$ to keep $D_{\text{KL}} \leq \delta_{\text{target}}$ (e.g. the "KL annealing" trick in TRL) is more robust than a fixed constant.

---

!!! key "Key Takeaways"

    - **Numerically stable forms matter.** Always use log-sum-exp softmax; use `F.cross_entropy` from logits (never `log(softmax(logits))`); use the k2 KL estimator.
    - **Softmax temperature** (dividing logits by $T$ before softmax) sharpens ($T < 1$) or flattens ($T > 1$) the distribution without changing argmax.
    - **Attention scaling by $\sqrt{d_k}$** prevents the softmax from saturating into near-one-hot distributions at large head dimensions.
    - **RMSNorm vs LayerNorm:** modern decoders prefer RMSNorm — same stabilization, fewer ops, no centering bias.
    - **AdamW decouples weight decay** from the gradient adaptive step; folding $L_2$ into the gradient (classic Adam) leads to incorrect regularization with per-parameter step sizes.
    - **Chinchilla says scale tokens as fast as parameters.** In practice over-training beyond compute-optimal is often better because inference is expensive.
    - **LoRA rank $r$** is a budget knob: higher $r$ gives more expressivity at the cost of more trainable parameters; $r=16$ or $r=64$ covers most SFT/RLHF use cases.
    - **PPO clips the probability ratio, not the KL** — so monitor the k2 KL estimator separately to catch distribution drift.
    - **DPO $\beta$** controls the strength of the KL constraint implicitly baked into the closed-form optimum; too small $\beta$ leads to over-fitting on preferences; too large underfits.

---

## Further Reading

- Vaswani et al., *Attention Is All You Need* (2017) — original scaled dot-product attention and multi-head formulation.
- Ba et al., *Layer Normalization* (2016) — the LayerNorm paper.
- Zhang and Sennrich, *Root Mean Square Layer Normalization* (2019) — RMSNorm derivation.
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021) — RoPE derivation.
- Loshchilov and Hutter, *Decoupled Weight Decay Regularization* (2019) — AdamW.
- Kaplan et al., *Scaling Laws for Neural Language Models* (2020) — OpenAI scaling law.
- Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla, 2022) — Deepmind scaling law.
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2022) — LoRA.
- Schulman et al., *Proximal Policy Optimization Algorithms* (2017) — PPO.
- Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* (2023) — DPO.
- Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (2024) — GRPO.
