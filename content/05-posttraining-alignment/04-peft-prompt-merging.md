# 5.4 PEFT II: Prompt/Prefix Tuning, IA3, Model Merging & Soups

The previous chapter ([PEFT I: LoRA, QLoRA, DoRA & The Adapter Family](../05-posttraining-alignment/03-peft-lora-qlora.html)) showed how low-rank weight updates let you adapt large models cheaply. This chapter covers the complementary half of the parameter-efficient fine-tuning (PEFT) landscape: methods that add learnable tokens to the input space rather than the weight space (prompt tuning, prefix tuning, P-tuning v2), a multiplication-based approach that touches only scale vectors (IA3), and a family of techniques that skip gradient-based tuning entirely by combining pre-trained or fine-tuned checkpoints after the fact (model merging, task arithmetic, model soups, Frankenmerges).

The through-line is the same practical constraint: a 70-billion-parameter base model is expensive to fully fine-tune, expensive to maintain as dozens of separate copies, and hard to ship. Each technique in this chapter is a different answer to the question "how do we get specialized behavior cheaply?"

---

## Soft Prompts: Motivation and Taxonomy

Before going into individual algorithms, it helps to see why the field explored the *input* rather than the *weights*.

Standard prompting (hard prompting) works entirely in the discrete token space: you write a text prefix and the model conditions on it. This is zero-cost but brittle — the gradient of the downstream loss never reaches the prompt text, so you cannot improve it by training.

The key insight of **soft prompting** is: why not make the prompt embeddings themselves trainable? Swap out the discrete token sequence for a set of continuous, differentiable embedding vectors that are learned end-to-end on a task. The frozen base model provides its full representational power; the learnable "soft tokens" provide the task steering.

{{fig:softprompt-frozen-backbone-gradient-flow}}

Three distinct families emerged:

| Method | What is learned | Where it attaches | Params per task |
|---|---|---|---|
| **Prompt tuning** (Lester et al., 2021) | $k$ embedding vectors, first-layer input | Input layer only | $k \times d$ |
| **Prefix tuning** (Li & Liang, 2021) | Key/value pairs injected at every layer | All attention layers | $2 \times L \times k \times d$ |
| **P-tuning v2** (Liu et al., 2022) | Deep prefix + per-layer MLP reparameterization | All layers | similar to prefix tuning |

---

{{fig:peft-intervention-sites}}

## Prompt Tuning

### The Core Algorithm

Lester et al. introduced prompt tuning as the minimalist end of the soft-token spectrum. The only change relative to standard inference is that a learned matrix

$$
P \in \mathbb{R}^{k \times d}
$$

is prepended to the sequence of ordinary token embeddings $X \in \mathbb{R}^{n \times d}$ before the first transformer layer. The concatenated input is

$$
\tilde{X} = \begin{bmatrix} P \\ X \end{bmatrix} \in \mathbb{R}^{(k+n) \times d}.
$$

During fine-tuning, the backbone weights are frozen; gradients flow only through $P$. During inference, $P$ is stored once and prepended to every example.

Parameter count: for $k = 100$ soft tokens and $d = 4096$ (LLaMA-7B width), we have $100 \times 4096 = 409{,}600$ parameters — roughly 0.006 % of the base model.

### Initialization Strategy

How you initialize $P$ matters. Lester et al. found that initializing each soft token from a real vocabulary embedding (rather than random noise) gives notably faster convergence and higher peak accuracy. The intuition: the model already knows how to "read" its own embedding space; starting from a real token gives the optimizer a warm start.

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

class SoftPromptWrapper(nn.Module):
    """
    Wraps a frozen causal LM with a learnable soft-prompt prefix.
    Only the `soft_prompt` embedding matrix is trainable.
    """
    def __init__(self, model, num_soft_tokens: int = 20, init_text: str = "Answer the following question:"):
        super().__init__()
        self.model = model
        # Freeze ALL base model parameters
        for p in model.parameters():
            p.requires_grad_(False)

        d_model = model.config.hidden_size

        # ---------- Initialization from real tokens ----------
        tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
        init_ids = tokenizer(init_text, return_tensors="pt").input_ids[0]

        # Pad or truncate to exactly num_soft_tokens
        embed_weight = model.get_input_embeddings().weight.data  # (vocab_size, d)
        if len(init_ids) >= num_soft_tokens:
            chosen_ids = init_ids[:num_soft_tokens]
        else:
            # Random fill for any extra slots
            random_ids = torch.randint(0, embed_weight.size(0), (num_soft_tokens - len(init_ids),))
            chosen_ids = torch.cat([init_ids, random_ids])

        init_embeds = embed_weight[chosen_ids].clone()  # (num_soft_tokens, d)

        # The one trainable parameter
        self.soft_prompt = nn.Parameter(init_embeds)

    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size = input_ids.size(0)

        # Embed the real input tokens
        input_embeds = self.model.get_input_embeddings()(input_ids)  # (B, T, d)

        # Expand soft prompt to batch: (B, k, d)
        prompt = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)

        # Concatenate along the token dimension
        combined = torch.cat([prompt, input_embeds], dim=1)  # (B, k+T, d)

        # Extend the attention mask to cover the soft tokens
        if attention_mask is not None:
            prompt_mask = torch.ones(batch_size, self.soft_prompt.size(0),
                                     device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        # Shift labels to account for the k prepended tokens
        # (for CLM loss, we don't want to predict from soft-token positions)
        if labels is not None:
            pad_labels = torch.full((batch_size, self.soft_prompt.size(0)), -100,
                                    device=labels.device, dtype=labels.dtype)
            labels = torch.cat([pad_labels, labels], dim=1)

        return self.model(inputs_embeds=combined, attention_mask=attention_mask, labels=labels)

# ---- Quick test ----
if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    wrapper = SoftPromptWrapper(model, num_soft_tokens=10)

    trainable = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in wrapper.parameters())
    print(f"Trainable: {trainable:,}  Total: {total:,}  Fraction: {trainable/total:.5%}")
    # Trainable: 7,680  Total: 124,447,232  Fraction: 0.00617%
```

### Scaling Behavior

A key empirical finding from the original paper: prompt tuning's gap relative to full fine-tuning shrinks as model size grows. For models with on the order of hundreds of millions of parameters, full fine-tuning wins by a meaningful margin. At very large scales — on the order of tens of billions of parameters — prompt tuning approaches the accuracy of full fine-tuning while using orders-of-magnitude fewer parameters. This scale-dependence is a recurring theme for input-space PEFT methods.

---

## Prefix Tuning

### Architecture

Li & Liang (2021) identified a limitation of prompt tuning: soft tokens only influence the model at the first layer. By the time information propagates through the second, third, and subsequent layers, the "prompt signal" has been blended into the residual stream in ways that may not remain task-specific. Their solution is to inject learned key–value pairs at **every** transformer layer's attention computation.

For layer $\ell$, the standard self-attention computes

$$
\text{Attn}^\ell(Q^\ell, K^\ell, V^\ell) = \text{softmax}\!\left(\frac{Q^\ell (K^\ell)^\top}{\sqrt{d_k}}\right) V^\ell.
$$

Prefix tuning prepends a learned prefix to both $K^\ell$ and $V^\ell$:

$$
\tilde{K}^\ell = \begin{bmatrix} P^K_\ell \\ K^\ell \end{bmatrix}, \quad
\tilde{V}^\ell = \begin{bmatrix} P^V_\ell \\ V^\ell \end{bmatrix},
$$

where $P^K_\ell, P^V_\ell \in \mathbb{R}^{k \times d_k}$. The attention query is unchanged and attends to both the learned prefix and the real tokens.

The effect: at every layer, the model has $k$ "virtual past tokens" whose key–value representations can be specialized to steer the layer's attention pattern for a particular task.

### Reparameterization Trick

Directly optimizing $P^K_\ell, P^V_\ell$ leads to training instability — the prefix tensors exist in a high-dimensional continuous space with no warm-start. Li & Liang found that routing the prefix through a small MLP helps:

$$
P^\ell = \text{MLP}_\theta(e^\ell),
$$

where $e^\ell$ is a row of a small trainable embedding matrix $E \in \mathbb{R}^{k \times d'}$ (with $d' \ll d$). At inference time, the MLP can be discarded; only the resulting $P^\ell$ tensors are stored.

```python
import torch
import torch.nn as nn

class PrefixEncoder(nn.Module):
    """
    Generates per-layer prefix key/value tensors from a compact embedding.
    At inference, call `.materialize()` to get the final prefix tensors
    (the MLP can then be discarded to save memory).
    """
    def __init__(self, num_layers: int, num_heads: int, d_head: int,
                 prefix_len: int = 10, bottleneck_dim: int = 512):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads  = num_heads
        self.d_head     = d_head
        self.prefix_len = prefix_len

        # Compact embedding: shape (prefix_len, bottleneck_dim)
        self.embedding = nn.Embedding(prefix_len, bottleneck_dim)

        # Two-layer MLP expands to (2 * num_layers * num_heads * d_head)
        # Factor of 2 = one for K, one for V
        out_dim = 2 * num_layers * num_heads * d_head
        self.mlp = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim * 2),
            nn.Tanh(),
            nn.Linear(bottleneck_dim * 2, out_dim),
        )

    def forward(self):
        # Token indices 0..prefix_len-1
        idx = torch.arange(self.prefix_len, device=self.embedding.weight.device)
        h = self.embedding(idx)                     # (prefix_len, bottleneck_dim)
        out = self.mlp(h)                           # (prefix_len, 2*L*H*d_head)

        # Reshape to (2, num_layers, prefix_len, num_heads, d_head)
        out = out.view(self.prefix_len, 2, self.num_layers, self.num_heads, self.d_head)
        out = out.permute(1, 2, 0, 3, 4)           # (2, L, prefix_len, H, d_head)
        # out[0] = K prefix across all layers; out[1] = V prefix
        return out[0], out[1]                       # each: (L, prefix_len, H, d_head)


# Sanity check parameter count
encoder = PrefixEncoder(num_layers=32, num_heads=32, d_head=128, prefix_len=10, bottleneck_dim=512)
trainable = sum(p.numel() for p in encoder.parameters())
print(f"Prefix encoder params: {trainable:,}")
# ~13 M params for a 70B-class attention shape — vs. 70B frozen
```

### P-tuning v2

P-tuning v2 (Liu et al., 2022) is essentially a cleaned-up, scaled version of prefix tuning applied to encoder-style and encoder-decoder models. Its main contributions are:

1. **Deep prefix across all layers** — confirmed the importance of per-layer injection (versus only the input layer) for complex NLU tasks.
2. **Removing the MLP reparameterization** — found that with careful initialization and learning-rate tuning, direct optimization of the prefix tensors is stable and slightly better.
3. **Verifiable results at different scales** — showed that deep prefix tuning can match full fine-tuning on hard sequence-labeling tasks (NER, SRL) even for smaller models (hundreds of millions of parameters), filling a gap where prompt tuning struggles.

---

## IA3: Infused Adapter by Inhibiting and Amplifying Inner Activations

### Motivation

Liu et al. (T-Few, 2022) asked: what is the minimal intervention that can still adapt behavior effectively? Instead of adding parameters (adapters) or input tokens (prefix tuning), IA3 **rescales** three specific activation vectors inside the transformer using learned scale vectors with as few as a few thousand parameters per task.

### Mechanism

For each transformer layer, IA3 introduces three learned vectors:

$$
l_k, l_v \in \mathbb{R}^{d_k}, \quad l_{ff} \in \mathbb{R}^{d_{ff}},
$$

and modifies the forward pass as:

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q (l_k \odot K)^\top}{\sqrt{d_k}}\right)(l_v \odot V),
$$

$$
\text{FFN}(x) = W_2 \cdot \bigl(l_{ff} \odot \sigma(W_1 x)\bigr),
$$

where $\odot$ denotes element-wise multiplication. The backbone $Q, K, V$ projections and the FFN weights are completely frozen; only $l_k, l_v, l_{ff}$ are trained.

The intuition is that element-wise rescaling can suppress or amplify the "channels" most relevant for a task without requiring any additive rank-1 update in weight space. Because the scale vectors multiply directly into the forward computation, they can be folded into the weight matrices at inference time with zero overhead:

$$
W'_K = \text{diag}(l_k) \cdot W_K, \quad W'_V = \text{diag}(l_v) \cdot W_V.
$$

After this fold, the model has the same parameter count as the base model and no extra matrix multiplication at runtime — a key advantage over adapters.

### Parameter Count

For a 7B model with $L = 32$ layers, $d_k = 128$, $d_{ff} = 14336$:

$$
\text{params} = L \times (d_k + d_k + d_{ff}) = 32 \times (128 + 128 + 14336) = 32 \times 14592 \approx 467{,}000.
$$

That is roughly 0.007 % of 7B — smaller than a LoRA rank-8 adapter.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class IA3Attention(nn.Module):
    """
    Single-head attention with IA3 scale vectors on K and V.
    In practice you would patch an existing multi-head module;
    here we show the mechanics clearly.
    """
    def __init__(self, d_model: int, d_k: int):
        super().__init__()
        self.d_k = d_k
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_k, bias=False)
        self.W_o = nn.Linear(d_k, d_model, bias=False)

        # Freeze backbone
        for p in [self.W_q, self.W_k, self.W_v, self.W_o]:
            for param in p.parameters():
                param.requires_grad_(False)

        # IA3 learnable scale vectors — initialized to 1 (identity)
        self.l_k = nn.Parameter(torch.ones(d_k))
        self.l_v = nn.Parameter(torch.ones(d_k))

    def forward(self, x):
        Q = self.W_q(x)                            # (B, T, d_k)
        K = self.W_k(x) * self.l_k                # element-wise scale on K
        V = self.W_v(x) * self.l_v                # element-wise scale on V

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn   = F.softmax(scores, dim=-1)
        out    = torch.matmul(attn, V)
        return self.W_o(out)

    def fold_weights(self):
        """
        Bake IA3 scales into W_k and W_v so inference has zero overhead.
        After calling this, l_k and l_v can be deleted.
        """
        with torch.no_grad():
            # W_k output dim is d_k; scale each row
            self.W_k.weight.mul_(self.l_k.unsqueeze(1))
            self.W_v.weight.mul_(self.l_v.unsqueeze(1))
        # Detach scale vectors (they're now baked in)
        del self.l_k, self.l_v
        print("IA3 weights folded — no runtime overhead.")


class IA3FFN(nn.Module):
    """FFN with IA3 scale on the intermediate activations."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff, bias=False)
        self.W2 = nn.Linear(d_ff, d_model, bias=False)
        for p in [self.W1, self.W2]:
            for param in p.parameters():
                param.requires_grad_(False)

        self.l_ff = nn.Parameter(torch.ones(d_ff))

    def forward(self, x):
        h = F.gelu(self.W1(x))    # (B, T, d_ff)
        h = h * self.l_ff         # IA3 scale on hidden activations
        return self.W2(h)
```

### T-Few: Few-Shot Fine-Tuning with IA3

The T-Few paper packaged IA3 into a practical recipe for few-shot learning on T0/T5 family models:

1. Pre-train a multi-task model (T0) across many tasks.
2. For a new task with only a handful of labeled examples, fine-tune only the IA3 vectors.
3. Add an "unlikelihood" regularization term that penalizes high probability on wrong answers (preventing memorization with tiny data).

The combination matched or beat much larger few-shot competitors while using a fraction of the compute.

---

## Model Merging: The Big Idea

All the methods above involve gradient-based training, however cheap. Model merging takes a different stance: **can we combine the knowledge in two or more independently trained checkpoints by arithmetic on their weight tensors?**

The answer is yes, surprisingly well, and the space of merging algorithms has exploded since 2022. The key enabling observation is that fine-tuned models that share the same pre-trained initialization live in a roughly convex basin of the loss landscape — their interpolation often stays in a low-loss region for multiple tasks simultaneously.

### Why Merging Beats Fine-Tuning in Some Scenarios

- **No access to training data.** Two proprietary fine-tunes can be merged without seeing each other's data.
- **No additional GPU hours.** Merging is a CPU-memory operation — no forward/backward passes.
- **Catastrophic forgetting avoidance.** Sequential fine-tuning on task B degrades task A; merging two task-specific models often retains both.
- **Ensemble-like generalization.** Merged models sometimes outperform any single constituent on held-out distributions.

See [PEFT I: LoRA, QLoRA, DoRA & The Adapter Family](../05-posttraining-alignment/03-peft-lora-qlora.html) for the gradient-based PEFT background, and [Supervised Fine-Tuning & Instruction Tuning](../05-posttraining-alignment/01-sft-instruction-tuning.html) for what fine-tuned checkpoints look like before merging.

---

## The Merging Algorithms

### Linear Interpolation (Weight Averaging)

The simplest merge: take a weighted average of two (or more) model parameter tensors.

$$
\theta_\text{merge} = (1 - \lambda)\,\theta_A + \lambda\,\theta_B.
$$

Model soups (Wortsman et al., 2022) used this idea to combine several fine-tuned checkpoints of the same base model, showing that the average generalized better than any individual checkpoint under distribution shift — a form of implicit ensembling in weight space.

**When it works:** $\theta_A$ and $\theta_B$ must share the same initialization (same base model). Models trained from completely different random seeds do not generally merge well; the parameters live in different basins.

### SLERP: Spherical Linear Interpolation

Linear interpolation can reduce the norm of the merged tensor when $\theta_A$ and $\theta_B$ point in different directions (just as the average of two unit vectors on a sphere has smaller magnitude). SLERP corrects this by interpolating along the great circle:

$$
\text{SLERP}(\theta_A, \theta_B, t) = \frac{\sin((1-t)\Omega)}{\sin\Omega}\,\hat{\theta}_A + \frac{\sin(t\Omega)}{\sin\Omega}\,\hat{\theta}_B,
$$

where $\Omega = \arccos(\hat{\theta}_A \cdot \hat{\theta}_B)$ is the angle between the unit-normalized versions of the two vectors and $t \in [0,1]$ is the interpolation parameter.

SLERP is applied independently to each weight tensor (or each row/column vector within a tensor), preserving magnitude throughout the interpolation.

```python
import torch

def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float, eps: float = 1e-8) -> torch.Tensor:
    """
    Spherical linear interpolation between two tensors.
    Works on flattened weight matrices.
    
    Args:
        v0, v1: weight tensors of the same shape (will be flattened, then reshaped)
        t: interpolation factor in [0, 1]
    Returns:
        Interpolated tensor, same shape as inputs
    """
    orig_shape = v0.shape
    v0_flat = v0.flatten().float()
    v1_flat = v1.flatten().float()

    # Normalise
    n0 = v0_flat / (v0_flat.norm() + eps)
    n1 = v1_flat / (v1_flat.norm() + eps)

    # Angle between the two vectors
    dot = torch.clamp((n0 * n1).sum(), -1.0, 1.0)
    omega = torch.acos(dot)

    if omega.abs() < 1e-6:
        # Nearly parallel — fall back to linear interpolation
        return ((1 - t) * v0 + t * v1).reshape(orig_shape)

    sin_omega = torch.sin(omega)
    out = (torch.sin((1 - t) * omega) / sin_omega) * v0_flat + \
          (torch.sin(t * omega) / sin_omega) * v1_flat

    return out.reshape(orig_shape)


def slerp_models(state_dict_a: dict, state_dict_b: dict, t: float = 0.5) -> dict:
    """Merge two model state dicts using per-tensor SLERP."""
    merged = {}
    for key in state_dict_a:
        if key not in state_dict_b:
            merged[key] = state_dict_a[key]
            continue
        wa = state_dict_a[key].float()
        wb = state_dict_b[key].float()
        if wa.shape != wb.shape:
            raise ValueError(f"Shape mismatch at {key}: {wa.shape} vs {wb.shape}")
        merged[key] = slerp(wa, wb, t)
    return merged
```

### Task Arithmetic

Ilharco et al. (2023) introduced a clean algebraic framing. Define the **task vector** for a fine-tune as:

$$
\tau_{\text{task}} = \theta_{\text{fine-tuned}} - \theta_{\text{pre-trained}}.
$$

Task vectors support a surprising range of operations:

- **Add** a skill: $\theta_\text{merge} = \theta_\text{pre-trained} + \lambda \tau_\text{task}$
- **Remove** a behavior: $\theta_\text{merge} = \theta_\text{pre-trained} - \lambda \tau_\text{task}$ (analogy negation)
- **Combine** multiple tasks: $\theta_\text{merge} = \theta_\text{pre-trained} + \lambda \sum_i \tau_i$

{{fig:task-vector-arithmetic-basin}}

The scalar $\lambda$ controls the "temperature" of the intervention — too large and the fine-tune dominates; too small and the effect disappears. Values around 0.3–0.8 work well in practice.

```python
def compute_task_vector(base_sd: dict, finetuned_sd: dict) -> dict:
    """Compute the task vector (delta weights) for one fine-tune."""
    return {k: finetuned_sd[k].float() - base_sd[k].float() for k in base_sd}


def task_arithmetic_merge(base_sd: dict, task_vectors: list[dict],
                          scale: float = 0.5) -> dict:
    """
    Merge multiple task vectors into the base model.
    
    Args:
        base_sd:      base model state dict
        task_vectors: list of per-task delta-weight dicts
        scale:        scalar lambda applied uniformly to all task vectors
    Returns:
        merged state dict
    """
    merged = {k: v.float().clone() for k, v in base_sd.items()}
    for tv in task_vectors:
        for k in tv:
            if k in merged:
                merged[k].add_(scale * tv[k])
    return merged
```

### TIES: Trim, Elect Sign, Disjoint Merge

A key failure mode of naive task arithmetic is **interference**: different tasks may push the same parameter in opposite directions. TIES-Merging (Yadav et al., 2023) addresses this in three steps:

**Step 1 — Trim.** For each task vector $\tau_i$, keep only the top-$p$% parameters by absolute magnitude; zero out the rest. This removes noise from small updates that are more likely to conflict than contribute.

$$
\hat{\tau}_i[j] = \begin{cases} \tau_i[j] & \text{if } |\tau_i[j]| \geq t_i^{(p)} \\ 0 & \text{otherwise} \end{cases}
$$

**Step 2 — Elect sign.** For each parameter position $j$, resolve the sign conflict by majority vote among all task vectors that have a nonzero entry at position $j$:

$$
\gamma_j = \operatorname{sign}\!\left(\sum_i \hat{\tau}_i[j]\right).
$$

**Step 3 — Disjoint merge.** Aggregate only the task vectors that agree with the elected sign:

$$
\tau_\text{merge}[j] = \frac{\sum_i \mathbb{1}[\hat{\tau}_i[j] \cdot \gamma_j > 0] \cdot \hat{\tau}_i[j]}{\sum_i \mathbb{1}[\hat{\tau}_i[j] \cdot \gamma_j > 0]}.
$$

The final model is:

$$
\theta_\text{merge} = \theta_\text{pre-trained} + \lambda \cdot \tau_\text{merge}.
$$

```python
import torch

def ties_merge(
    base_sd: dict,
    task_vectors: list[dict],
    scale: float = 0.5,
    trim_fraction: float = 0.8,          # keep top (1 - trim_fraction)
) -> dict:
    """
    TIES-Merging: Trim, Elect Sign, Disjoint Merge.
    
    Args:
        base_sd:          base model state dict
        task_vectors:     list of per-task delta dicts (same keys as base_sd)
        scale:            final scaling factor lambda
        trim_fraction:    fraction of params to zero out (e.g. 0.8 => keep top 20%)
    Returns:
        merged state dict
    """
    merged = {}

    for key in base_sd:
        base_val = base_sd[key].float()
        deltas = []

        # --- Step 1: Trim ---
        for tv in task_vectors:
            if key not in tv:
                continue
            delta = tv[key].float().clone()
            # Compute magnitude threshold at the (trim_fraction) quantile
            if delta.numel() > 1:
                threshold = torch.quantile(delta.abs().flatten(), trim_fraction)
                delta[delta.abs() < threshold] = 0.0
            deltas.append(delta)

        if not deltas:
            merged[key] = base_val
            continue

        stacked = torch.stack(deltas, dim=0)          # (num_tasks, *param_shape)

        # --- Step 2: Elect sign ---
        # Sum of all trimmed deltas to determine majority sign
        sign_sum = stacked.sum(dim=0)
        elected_sign = torch.sign(sign_sum)             # +1 or -1 per parameter
        # Handle exact zero: assign +1 arbitrarily
        elected_sign[elected_sign == 0] = 1.0

        # --- Step 3: Disjoint merge ---
        # Mask: keep delta only where it agrees with elected sign
        agree_mask = (stacked * elected_sign.unsqueeze(0)) > 0   # (num_tasks, *shape)

        # Numerator: sum of agreeing deltas
        numerator   = (stacked * agree_mask.float()).sum(dim=0)
        # Denominator: count of agreements per position
        denominator = agree_mask.float().sum(dim=0).clamp(min=1.0)

        task_vector_merged = numerator / denominator
        merged[key] = base_val + scale * task_vector_merged

    return merged


# ------------- Worked sketch: two tasks, small tensor ----------------
if __name__ == "__main__":
    torch.manual_seed(0)
    # Simulate a single weight tensor of size (4, 4)
    base  = {"W": torch.zeros(4, 4)}
    tv_a  = {"W": torch.randn(4, 4) * 0.3}   # task A gradient
    tv_b  = {"W": torch.randn(4, 4) * 0.3}   # task B gradient

    result = ties_merge(base, [tv_a, tv_b], scale=0.5, trim_fraction=0.6)
    print("Merged W:\n", result["W"].round(decimals=3))
    print("Nonzero fraction:", (result["W"] != 0).float().mean().item())
```

### DARE: Drop And REscale

DARE (Yu et al., 2023) takes an even simpler denoising approach: randomly zero out a fraction $p$ of the task vector entries and rescale the survivors by $1/(1-p)$ to preserve the expected magnitude. This is analogous to dropout applied to the delta weights.

$$
\hat{\tau}[j] = \begin{cases} \frac{\tau[j]}{1-p} & \text{with probability } 1-p \\ 0 & \text{with probability } p \end{cases}
$$

Despite its simplicity, DARE often reduces interference well because the interference signal tends to be distributed among many small parameters, while the task-specific signal is concentrated in fewer large ones. Random dropping disproportionately removes the former.

DARE is frequently combined with TIES (DARE-TIES): apply DARE's stochastic trimming first, then TIES's sign election and disjoint merge.

!!! example "Worked Example: Parameter Count and Memory for a TIES Merge"

    Suppose we want to merge three LLaMA-7B fine-tuned checkpoints, each stored in bfloat16.
    
    - Base model: ~7 billion parameters × 2 bytes/param = **14 GB**
    - Each fine-tune checkpoint: same 14 GB
    - Task vectors (delta weights): same size as the model = 14 GB each
    
    Total GPU/CPU memory needed to run TIES merge:
    
    - Load base: 14 GB
    - Load three task vectors: 3 × 14 GB = 42 GB
    - Working buffers (stacked deltas, masks): roughly one additional copy = ~14 GB
    - **Total: ~70 GB**
    
    This fits comfortably on a machine with 128 GB of CPU RAM and can be done entirely in float32 on CPU with no GPUs. Runtime on CPU with PyTorch is typically a few minutes for a 7B model — merging is genuinely cheap.
    
    For a 70B model at bfloat16, the same calculation gives ~140 GB per checkpoint × 5 tensors ≈ 700 GB — requires a well-equipped CPU server but remains feasible. GPU-resident merging at 70B would require a multi-GPU node.

---

## Model Soups and Frankenmerges

### Model Soups

Wortsman et al. (2022) coined "model soups" for the practice of averaging multiple fine-tuned checkpoints of the same pre-trained model. The recipe:

1. Fine-tune the same base model with several different hyperparameter configurations (learning rate, data augmentation, etc.).
2. Average all checkpoints that individually exceed some accuracy threshold.
3. The resulting "soup" typically generalizes better than any single ingredient.

The theoretical grounding connects to loss-landscape flatness and the work of Garipov et al. on loss surface geometry: fine-tunes from the same pre-trained model tend to lie in a connected low-loss region, so their average is also low-loss.

### Frankenmerges

The community (particularly on HuggingFace and the mergekit project) explored an extreme variant: merging **different models entirely**, combining, say, layers from a coding-specialized fine-tune with layers from a math-specialized fine-tune and layers from a general instruction-following model. This is sometimes called a Frankenmerge or model frankenstein.

The simplest Frankenmerge strategy selects layers by index:

```python
def frankenmerge(
    state_dicts: list[dict],
    layer_assignments: list[int],
    num_layers: int,
) -> dict:
    """
    Build a Frankenmerge by selecting each layer from a specific model.
    
    Args:
        state_dicts:      list of model state dicts (all same architecture)
        layer_assignments: list of length num_layers, value = index into state_dicts
        num_layers:       number of transformer layers
    Returns:
        merged state dict
    """
    merged = {}
    # Copy all non-layer parameters from model 0 (embed, lm_head, norms)
    for key, val in state_dicts[0].items():
        if not any(f".{i}." in key for i in range(num_layers)):
            merged[key] = val.clone()

    # Assign each layer from the specified model
    for layer_idx, model_idx in enumerate(layer_assignments):
        src = state_dicts[model_idx]
        for key in src:
            if f".{layer_idx}." in key:
                merged[key] = src[key].clone()

    return merged


# Example: 32-layer model — first 16 from model 0, last 16 from model 1
assignments = [0] * 16 + [1] * 16
# merged_sd = frankenmerge([sd_general, sd_coding], assignments, 32)
```

Frankenmerges can be surprisingly capable, but they are sensitive to layer ordering and often require empirical search over which layers to pull from which model. Tools like `mergekit` automate this exploration.

### Mergekit: The Practical Tool

`mergekit` (by Charles Goddard) is the de facto library for model merging in the open-source community. It supports TIES, DARE, SLERP, task arithmetic, and Frankenmerges via a YAML config:

```yaml
# mergekit config: TIES merge of two Mistral-7B fine-tunes
merge_method: ties
base_model: mistralai/Mistral-7B-v0.1
models:
  - model: my-org/mistral-7b-code-finetune
    parameters:
      weight: 0.5
      density: 0.2          # keep top 20% of delta by magnitude (trim_fraction=0.8)
  - model: my-org/mistral-7b-math-finetune
    parameters:
      weight: 0.5
      density: 0.2
parameters:
  normalize: true           # normalize task vectors before merging
  int8_mask: true           # use int8 masks to reduce RAM
dtype: bfloat16
```

```bash
# Install and run
pip install mergekit

mergekit-merge merge_config.yaml ./output-model \
    --cuda                   \   # use GPU if available
    --copy-tokenizer         \   # copy tokenizer from base model
    --lazy-unpickle              # stream large tensors to avoid OOM
```

---

## When Does Merging Beat Fine-Tuning?

This is the practical question you actually care about. Here is a decision framework:

| Scenario | Recommendation |
|---|---|
| You have training data and a GPU budget | Gradient-based fine-tuning (LoRA or full) — always the ceiling |
| You have two fine-tuned checkpoints, no data | TIES or SLERP merge — often within a few points of training |
| You want to combine skills without forgetting | Task arithmetic or TIES merge |
| You have many checkpoints of the same base | Model soup (average) — free accuracy boost |
| You want to negate a behavior | Task arithmetic subtraction |
| You want to test capability combinations cheaply | Frankenmerge + eval loop |
| Models come from different base checkpoints | Merging is unreliable; use fine-tuning or distillation |

The loss-landscape intuition is the key: merging works because fine-tunes from the same base model are geometrically close. If the models started from different initializations, the weight spaces are unrelated and merging is noise.

!!! interview "Interview Corner"

    **Q:** "You have two 7B LLaMA fine-tunes — one specialized for SQL generation, one for Python coding. You want a single model that does both well, but you have no training data and no GPU. What are your options and which would you choose?"

    **A:** The main options are (a) SLERP merge — interpolates along the unit sphere to preserve norms and avoids the magnitude collapse of linear averaging; (b) task arithmetic — subtract the base model from each fine-tune to get task vectors, then add both scaled task vectors back to the base; (c) TIES merge — same as task arithmetic but first trims small-magnitude parameters and resolves sign conflicts by majority vote, reducing interference between the two tasks.
    
    I would choose TIES with a moderate trim density (e.g., keep top 20-40% of each task vector), because SQL and Python code share many parameters (tokenization, syntax awareness) but diverge on dialect-specific idioms, and TIES's sign-election step explicitly handles the parameter-level conflicts that arise from that overlap. I would set the scale $\lambda$ to around 0.4–0.6 for each task vector and eval on a held-out set to tune it. If I had even a small validation set, I could do a grid search over $\lambda$ and density in CPU memory in minutes.

!!! warning "Common Pitfall: Merging Models with Different Tokenizers or Architectures"

    Merging only makes sense when both models have *exactly* the same architecture and tokenizer. If model A uses a 32k-token vocabulary and model B uses a 128k-token vocabulary, their embedding matrices have different shapes and cannot be averaged. Always verify `config.json` architecture fields and `tokenizer.json` vocabulary size match before attempting any merge.

!!! tip "Practitioner Tip: Use float32 for Merge Arithmetic"

    Even if your models are stored in bfloat16, always cast to float32 before computing task vectors and running merge arithmetic. The intermediate differences $\theta_\text{fine-tuned} - \theta_\text{base}$ can be very small, and bfloat16's limited mantissa precision (7 bits) causes significant rounding error when subtracting numbers of similar magnitude. Cast back to bfloat16 only at the end.

---

## Comparison and Selection Guide

```text
Method           Params trained   Modifies weights?  Inference overhead   Best for
───────────────  ───────────────  ─────────────────  ──────────────────   ────────────────────────
Prompt tuning    k × d            No                 +k tokens            Low-resource; huge models
Prefix tuning    2 × L × k × d_k No                 +k KV per layer      NLG/seq2seq; all layers
P-tuning v2      similar          No                 +k KV per layer      NLU (NER, SRL); robust
IA3              L × (2d_k+d_ff)  No (foldable)      Zero (after fold)    Few-shot; fast deploy
LoRA             2 × L × r × d   Yes (merge-able)    Zero (after merge)   General purpose
TIES merge       0 (no training)  Yes                Zero                 Multi-task combination
SLERP            0                Yes                Zero                 Two-model interpolation
Task arithmetic  0                Yes                Zero                 Adding/removing skills
Model soup       0                Yes                Zero                 Same-base checkpoint avg
```

For a complete treatment of LoRA and adapters, see [PEFT I: LoRA, QLoRA, DoRA & The Adapter Family](../05-posttraining-alignment/03-peft-lora-qlora.html). For the memory math behind training these methods, see [Memory-Efficient Training: Checkpointing, Offloading & LoRA Math](../04-kernels-efficiency/10-memory-efficient-training.html).

If your use case involves distribution shifts after merging, the evaluation framework in [The Evaluation Problem & Benchmark Landscape](../11-evaluation/01-eval-landscape.html) provides the right lens for measuring merged-model generalization.

---

!!! key "Key Takeaways"
    - **Prompt tuning** learns $k$ soft embedding vectors prepended to the input; it trains fewer than 0.01% of parameters and closes the gap with full fine-tuning only at very large model scales.
    - **Prefix tuning** injects learned key–value pairs at *every* attention layer, giving the model a per-layer task signal; it is more effective than prompt tuning on smaller models and harder tasks.
    - **IA3** multiplies learned scale vectors into key, value, and FFN activations; its ~0.007% parameter overhead can be *folded into weights* at inference for zero latency cost.
    - **Task arithmetic** defines a task vector as $\theta_\text{ft} - \theta_\text{base}$; tasks can be added, subtracted, and composed algebraically — no new training required.
    - **TIES-Merging** reduces inter-task interference by trimming small-magnitude parameters, electing a majority sign per position, and averaging only the agreeing values.
    - **DARE** provides a stochastic alternative to deterministic trimming: random dropout on task-vector entries with rescaling to preserve expectation.
    - **Model soups** average several fine-tunes of the same base model; the average generalizes better than any individual due to implicit ensembling in weight space.
    - **Merging only works reliably when models share the same pre-trained initialization.** Different base checkpoints live in geometrically unrelated parameter spaces.
    - **Cast to float32 before merge arithmetic** — bfloat16 rounding errors in small delta weights are a real and common bug.

---

!!! sota "State of the Art & Resources (2026)"
    Soft-prompt methods (prompt tuning, prefix tuning, IA3) are now mature and production-ready via the HuggingFace PEFT library, while model merging has evolved from a curiosity into a mainstream technique — methods like TIES, DARE, and task arithmetic are used routinely to combine open-weight fine-tunes without any retraining.

    **Foundational work**

    - [Lester et al., *The Power of Scale for Parameter-Efficient Prompt Tuning* (2021)](https://arxiv.org/abs/2104.08691) — establishes prompt tuning and its scale-dependent behavior.
    - [Li & Liang, *Prefix-Tuning: Optimizing Continuous Prompts for Generation* (2021)](https://arxiv.org/abs/2101.00190) — introduces per-layer key-value prefix injection and the MLP reparameterization trick.
    - [Liu et al., *P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks* (2022)](https://arxiv.org/abs/2110.07602) — validates deep prefix tuning on NLU tasks including NER and SRL.
    - [Liu et al. (T-Few), *Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning* (2022)](https://arxiv.org/abs/2205.05638) — introduces IA3 and the T-Few recipe for few-shot adaptation.
    - [Wortsman et al., *Model Soups: Averaging Weights of Multiple Fine-Tuned Models* (2022)](https://arxiv.org/abs/2203.05482) — coins model soups and demonstrates weight-space ensembling for free accuracy gains.

    **Recent advances (2023–2026)**

    - [Ilharco et al., *Editing Models with Task Arithmetic* (2023)](https://arxiv.org/abs/2212.04089) — formalizes task vectors as composable weight-space directions for adding, removing, and combining skills.
    - [Yadav et al., *TIES-Merging: Resolving Interference When Merging Models* (2023)](https://arxiv.org/abs/2306.01708) — trim-elect-sign pipeline that reduces parameter interference across merged task vectors.
    - [Yu et al., *Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch* (2023)](https://arxiv.org/abs/2311.03099) — introduces DARE, stochastic delta-weight dropout with rescaling to reduce merge interference.
    - [Yang et al., *Model Merging in LLMs, MLLMs, and Beyond* (2024)](https://arxiv.org/abs/2408.07666) — comprehensive survey of model merging methods, theory, and applications across ML subfields.

    **Open-source & tools**

    - [arcee-ai/mergekit](https://github.com/arcee-ai/mergekit) — the de facto model-merging toolkit; supports TIES, DARE, SLERP, task arithmetic, and Frankenmerges via YAML config.
    - [huggingface/peft](https://github.com/huggingface/peft) — HuggingFace PEFT library with production-ready implementations of prompt tuning, prefix tuning, P-tuning, and IA3.

    **Go deeper**

    - [HuggingFace PEFT: Soft Prompts Conceptual Guide](https://huggingface.co/docs/peft/conceptual_guides/prompting) — accessible walkthrough of prompt tuning, prefix tuning, and P-tuning with diagrams and code pointers.
    - [Goddard et al., *Arcee's MergeKit: A Toolkit for Merging Large Language Models* (2024)](https://arxiv.org/abs/2403.13257) — describes the engineering and algorithms behind mergekit, including out-of-core CPU-resident merging.

## Further Reading

- Lester, Brain et al. **"The Power of Scale for Parameter-Efficient Prompt Tuning"**, EMNLP 2021. The original prompt tuning paper; contains the key scaling analysis.
- Li & Liang. **"Prefix-Tuning: Optimizing Continuous Prompts for Generation"**, ACL 2021. Introduces per-layer prefix injection and the MLP reparameterization trick.
- Liu et al. **"P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks"**, ACL 2022. Deep prefix tuning for NLU tasks.
- Liu et al. (T-Few). **"Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning"**, NeurIPS 2022. Introduces IA3 and the T-Few training recipe.
- Wortsman et al. **"Model Soups: Averaging Weights of Multiple Fine-Tuned Models Improves Accuracy and Robustness"**, ICML 2022. The foundational model-soup paper.
- Ilharco et al. **"Editing Models with Task Arithmetic"**, ICLR 2023. Formalizes task vectors; shows arithmetic composition of skills.
- Yadav et al. **"TIES-Merging: Resolving Interference When Merging Models"**, NeurIPS 2023. TIES algorithm with comprehensive multi-task experiments.
- Yu et al. **"Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch"**, 2023. Introduces DARE (Drop And REscale).
- Goddard, Charles et al. **mergekit** (GitHub: arcee-ai/mergekit). The practical open-source library for all merging methods; supports YAML-based merge configs.
