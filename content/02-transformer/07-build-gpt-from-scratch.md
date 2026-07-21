# 2.7 Building a GPT From Scratch (nanoGPT-style)

This is the chapter where the parts become a machine. Over the last several chapters we forged components in isolation: a [tokenizer](../02-transformer/01-tokenization.html) that turns text into integers, an [embedding table](../02-transformer/02-embeddings-input.html) that turns those integers into vectors, [scaled dot-product attention](../02-transformer/03-attention-from-scratch.html) that mixes information across positions, [multi-head attention](../02-transformer/04-mha-gqa-mla.html) that lets it mix in several subspaces at once, [positional encodings](../02-transformer/05-positional-encoding.html) that break attention's permutation symmetry, and [the Transformer block](../02-transformer/06-transformer-block.html) that wraps attention and an MLP in norms and residuals. Each was a sharpened tool on the bench. Now we assemble them into a working, trainable, text-generating decoder-only GPT — the architecture behind GPT-2, GPT-3, Llama, and (with refinements) every frontier model.

We are deliberately following the spirit of Andrej Karpathy's **nanoGPT**: a single, readable file you can hold in your head, that trains on a laptop, and that scales — without structural changes — to a real run on a cluster. By the end you will have (1) a `GPTConfig` dataclass that names every architectural knob, (2) the full module hierarchy from token IDs to logits, (3) a principled weight-initialization scheme and an explanation of *why* it matters at depth, (4) a complete training loop on a tiny character-level dataset, and (5) an autoregressive `generate` method with temperature, top-k, and top-p sampling. Everything is runnable PyTorch, heavily commented, and small enough to actually understand. This is the capstone of Part II: the chapter that proves the preceding theory composes into something that *learns language*.

We assume you are fluent with [PyTorch and autograd](../01-foundations/07-autodiff-pytorch.html) and have read the block chapter; we will reference but not re-derive attention internals.

## The Anatomy of a GPT: From Token IDs to Next-Token Logits

Before any code, fix the shape of the whole computation in your mind. A decoder-only GPT is a function

$$
f_\theta : \{0, 1, \dots, V-1\}^{T} \longrightarrow \mathbb{R}^{T \times V},
$$

mapping a sequence of $T$ token IDs (drawn from a vocabulary of size $V$) to a $T \times V$ matrix of **logits**: for *every* position $t$, an unnormalized score for every possible next token. Position $t$'s logits are the model's prediction of token $t+1$. That single design decision — predict the next token at *every* position simultaneously — is what lets us extract $T$ training signals from one sequence in one forward pass, and it is enforced entirely by the [causal mask](../02-transformer/03-attention-from-scratch.html) inside attention.

The data flows through six stages. Read this diagram top to bottom; the rest of the chapter is just filling in each box with code.

{{fig:buildgpt-data-flow-stages}}

The **residual stream** is the central object. A vector of width `n_embd` enters at the embedding, and every block *reads from it and adds back to it* — attention adds a "what other tokens told me" term, the MLP adds a "what I computed from myself" term. Nothing overwrites; everything accumulates. By the final layer the residual stream at position $t$ is a rich summary of the prefix $1{:}t$, and a single linear map (`lm_head`) reads off the next-token distribution. Keep this picture — *width-`n_embd` highway with additive on-ramps* — and the code below will feel inevitable.

!!! note "Aside: why decoder-only, and why this is the dominant design"
    A GPT is the *decoder-only* member of the Transformer family: one stack, causal masking, trained on plain next-token prediction. It has no encoder and no cross-attention. The competing designs — encoder-only (BERT) and encoder–decoder (T5) — are covered in [Architecture Variants](../02-transformer/08-architecture-variants.html). Decoder-only won the scaling race because it is the simplest thing that does *everything*: the same next-token objective subsumes generation, in-context learning, and (after [post-training](../05-posttraining-alignment/01-sft-instruction-tuning.html)) instruction following, all without architectural specialization.

## The Config: Naming Every Knob

A model is a few hyperparameters plus a fixed wiring. Putting every architectural choice into one `dataclass` is not bureaucracy — it is the difference between a script and a *system*. The config is what you serialize into a checkpoint, what you sweep over in experiments, and what a teammate reads to understand your model in thirty seconds. We will mirror GPT-2's parameterization.

```python
from dataclasses import dataclass

@dataclass
class GPTConfig:
    # --- Sequence / vocabulary ---
    block_size: int = 256     # max context length T (positions the model can see)
    vocab_size: int = 65      # size of the token vocabulary V

    # --- Model shape ---
    n_layer: int = 6          # number of Transformer blocks stacked
    n_head:  int = 6          # attention heads per block (must divide n_embd)
    n_embd:  int = 384        # residual-stream width (a.k.a. d_model)

    # --- Regularization / numerics ---
    dropout: float = 0.0      # dropout prob (0.0 for small/clean data; 0.1+ to regularize)
    bias:    bool  = False    # use bias terms in Linear / LayerNorm? (False = modern default)
```

A few of these deserve a sentence of justification, because interviewers love to ask "what is `n_embd` and how does it relate to `n_head`?"

- **`n_embd` (the residual-stream width, often written $d_\text{model}$)** is the single most important capacity knob. Doubling it roughly quadruples the parameters in every linear layer and is the primary lever in [scaling laws](../03-pretraining/04-scaling-laws.html).
- **`n_head` must divide `n_embd`.** Each head operates in a subspace of dimension `head_dim = n_embd / n_head`. With `n_embd=384, n_head=6` we get `head_dim=64` — and 64 is a near-universal sweet spot (it is what the $\sqrt{d_k}$ analysis in [the attention chapter](../02-transformer/03-attention-from-scratch.html) was tuned around).
- **`block_size`** caps the context. It sets the size of the learned positional table and the $T \times T$ attention matrix. It is a hard limit at train time but can be extended afterward (see [Long-Context Pretraining](../03-pretraining/13-long-context-pretraining.html)).
- **`bias=False`** drops bias terms from linear layers and LayerNorms. Modern models (Llama, etc.) omit them: they barely help, and removing them slightly speeds training and simplifies the math. We follow that convention.

The defaults above describe a ~10M-parameter model — small enough to train on a single GPU (or patiently on CPU) on a toy corpus, large enough to learn real structure. We will compute the exact parameter count in a worked example below.

## Building the Modules, Bottom-Up

We now write the model as four nested modules: `LayerNorm` (so we can toggle bias), `CausalSelfAttention`, `MLP`, and `Block`. Each is a faithful, from-scratch implementation; together they are the body of the GPT. (For the deep *why* behind pre-norm placement, GELU, and the 4× MLP expansion, see [The Transformer Block](../02-transformer/06-transformer-block.html); here we focus on assembling correct, runnable code.)

### LayerNorm with an optional bias

PyTorch's `nn.LayerNorm` always has a bias. We want the option to drop it, so we write our own thin wrapper. The math is the standard per-token normalization:

$$
\operatorname{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta,
\qquad \mu = \tfrac{1}{d}\textstyle\sum_i x_i, \quad \sigma^2 = \tfrac{1}{d}\textstyle\sum_i (x_i - \mu)^2,
$$

where the mean and variance are computed over the **feature** dimension (the last axis, width `n_embd`), independently for every token.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    """LayerNorm with an optional bias. PyTorch's built-in does not allow bias=None."""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))               # gamma, scale
        self.bias   = nn.Parameter(torch.zeros(ndim)) if bias else None  # beta, shift

    def forward(self, x):
        # Normalize over the last dim only. eps keeps us safe when variance ~ 0.
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-5)
```

### Causal self-attention (all heads in one matmul)

This is the heart of the model. We compute Q, K, V for *all heads at once* via a single fused linear layer of width `3 * n_embd`, reshape into `(B, n_head, T, head_dim)`, run masked scaled-dot-product attention, then project back. Computing all heads in one matmul (rather than a Python loop over heads) is the standard performance trick — it keeps the GPU busy with one large GEMM instead of many small ones.

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # One projection produces Q, K, V together: output width = 3 * n_embd.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection W_O: mixes the per-head results back into the stream.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout  = nn.Dropout(config.dropout)   # on the attention weights
        self.resid_dropout = nn.Dropout(config.dropout)   # on the block output

        # Use PyTorch's fused (FlashAttention-style) kernel when available.
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            # Fallback: precompute a (1,1,T,T) causal mask buffer for manual attention.
            mask = torch.tril(torch.ones(config.block_size, config.block_size))
            self.register_buffer("bias", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()            # batch, sequence length, embedding dim (= n_embd)

        # Project once, then split into the three roles along the feature axis.
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)   # each (B, T, C)

        # Reshape (B, T, C) -> (B, n_head, T, head_dim) so heads run in parallel.
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)    # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)

        if self.flash:
            # Fused kernel: applies the causal mask, scaling, softmax, and dropout.
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Manual path — identical math, written out so you can read it.
            att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))   # (B,nh,T,T)
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v                                                     # (B,nh,T,hd)

        # Re-assemble heads: (B, nh, T, hd) -> (B, T, C). contiguous() before view.
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))            # output projection + dropout
```

Two things to internalize. First, `c_attn` having output width `3 * n_embd` and `c_proj` having width `n_embd` are *exactly* where the bulk of attention's parameters live — $4\,n_\text{embd}^2$ per block (three for QKV, one for the output projection). Second, the `is_causal=True` flag is doing the autoregressive masking; without it the model would see the future and the training loss would be a meaningless near-zero.

### The MLP (position-wise feedforward)

After mixing information *across* positions with attention, each token is processed *independently* by a two-layer MLP. The hidden layer is 4× wider than the residual stream — a ratio that has held remarkably constant across model generations. The nonlinearity is GELU.

```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)  # expand
        self.gelu   = nn.GELU()                                                       # nonlinearity
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)  # contract
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)      # (B, T, 4*n_embd)
        x = self.gelu(x)      # smooth, differentiable ReLU-like activation
        x = self.c_proj(x)    # back down to (B, T, n_embd)
        return self.dropout(x)
```

The two linear layers hold $2 \times (4\,n_\text{embd}^2) = 8\,n_\text{embd}^2$ parameters per block — **twice** what attention uses. In a standard GPT, the MLPs are where most of the parameters (and, the [interpretability](../05-posttraining-alignment/01-sft-instruction-tuning.html) literature argues, most of the stored "knowledge") live.

### The Block: pre-norm, residual, repeat

A block wires attention and MLP into the residual stream with **pre-normalization** — LayerNorm is applied *before* each sublayer, and the sublayer's output is *added* to the input:

$$
x \leftarrow x + \operatorname{Attn}(\operatorname{LN}_1(x)), \qquad
x \leftarrow x + \operatorname{MLP}(\operatorname{LN}_2(x)).
$$

```python
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # pre-norm attention, residual add
        x = x + self.mlp(self.ln_2(x))    # pre-norm MLP,       residual add
        return x
```

Pre-norm (normalize the *input* to each sublayer, leave the residual path a clean identity) is the single most important stability change between the original 2017 Transformer (which used post-norm) and every modern LLM. Because the residual path is never normalized, gradients flow straight from the loss to the embedding with no attenuation, letting you stack dozens of layers without the loss exploding. The block chapter derives this; here we just rely on it.

## The Full GPT Module

Now we assemble the body, the embeddings, the head, and — crucially — the weight initialization and parameter sharing. This single class *is* the model.

```python
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None and config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),   # token embeddings
            wpe  = nn.Embedding(config.block_size, config.n_embd),   # learned positions
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),       # final layernorm
        ))
        # Language-model head: residual stream -> vocabulary logits.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # WEIGHT TYING: share the token-embedding matrix with the output head.
        # wte maps id -> vector; lm_head maps vector -> id-logits. Same V×n_embd
        # matrix (transposed), so we tie them: saves V*n_embd params and helps.
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize all weights (see _init_weights), then apply a special
        # scaled init to the residual output projections (GPT-2 trick).
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                # Scale residual-path projections by 1/sqrt(2 * n_layer). Explained below.
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / (2 * config.n_layer) ** 0.5)

    def _init_weights(self, module):
        # Linear and Embedding weights ~ N(0, 0.02^2); biases zeroed. The 0.02 std
        # is the GPT-2 convention — small enough to keep early activations tame.
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, f"sequence length {T} > block_size"
        pos = torch.arange(0, T, dtype=torch.long, device=device)   # (T,) position ids

        # 1) Embed tokens and positions, then sum into the residual stream.
        tok_emb = self.transformer.wte(idx)    # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)    # (T, n_embd), broadcasts over batch
        x = self.transformer.drop(tok_emb + pos_emb)

        # 2) Run the stack of Transformer blocks.
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)           # final layernorm

        # 3) Project to logits and, if training, compute the loss.
        if targets is not None:
            logits = self.lm_head(x)           # (B, T, vocab_size) — all positions
            # Cross-entropy over the vocabulary, flattening batch & time together.
            # ignore_index=-1 lets us mask out padding positions in the targets.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            # Inference: we only need the LAST position's logits to predict next token.
            logits = self.lm_head(x[:, [-1], :])  # (B, 1, vocab_size) — saves compute
            loss = None
        return logits, loss
```

### Weight tying and the two-headed init, explained

Two non-obvious lines in that constructor carry real weight (pun intended).

**Weight tying** sets `self.transformer.wte.weight = self.lm_head.weight`. The input embedding maps a token ID to a vector; the output head maps a vector to a logit per token. These are inverse operations over the *same* vocabulary, and tying their matrices both **saves $V \times n_\text{embd}$ parameters** (for a 50k vocab and 768-dim model, that is ~38M parameters — a huge fraction of a small model) and acts as a regularizer that empirically improves perplexity. It traces to Press & Wolf, *Using the Output Embedding to Tie Word Vectors* (2017).

**The scaled residual init** (`std = 0.02 / sqrt(2 * n_layer)` on every `c_proj.weight`) addresses a subtle problem with deep residual networks. Each block adds two terms to the residual stream. If every addition has variance $\approx \sigma^2$, then after $n_\text{layer}$ blocks the stream's variance grows like $2\,n_\text{layer}\,\sigma^2$ — it accumulates with depth. To keep the residual stream's scale roughly constant, GPT-2 shrinks the *output* projection of each sublayer (the matrices feeding the residual add) by $1/\sqrt{2\,n_\text{layer}}$, so the $2\,n_\text{layer}$ contributions sum back to a sane variance. The factor of 2 is because each block has two residual adds (attention and MLP). This is initialization-time hygiene, but it markedly improves the stability of deep models.

!!! warning "Common pitfall: forgetting `model.eval()` flips dropout and changes your numbers"
    Dropout and any norm with running statistics behave differently in train vs. eval mode. Generating text or measuring validation loss without calling `model.eval()` leaves dropout *active*, injecting noise and corrupting your samples and metrics. Symmetrically, forgetting `model.train()` afterward disables dropout during training. Wrap evaluation in `model.eval()` + `torch.no_grad()` and restore `model.train()` when done — and remember our config defaults to `dropout=0.0`, which hides this bug until you turn dropout on.

!!! example "Worked example: counting the parameters of our default GPT"
    Take `n_layer=6, n_head=6, n_embd=384, block_size=256, vocab_size=65`. Let $d = n_\text{embd} = 384$.

    **Per block** (ignoring tiny LayerNorm params and biases, since `bias=False`):

    - Attention: `c_attn` is $d \times 3d$ and `c_proj` is $d \times d$, so $4d^2 = 4 \cdot 384^2 = 589{,}824$.
    - MLP: `c_fc` is $d \times 4d$ and `c_proj` is $4d \times d$, so $8d^2 = 8 \cdot 384^2 = 1{,}179{,}648$.
    - Block total: $12 d^2 = 1{,}769{,}472$.

    **All 6 blocks:** $6 \times 12 d^2 = 72 d^2 = 10{,}616{,}832 \approx 10.6\text{M}$.

    **Embeddings:** token table $V \times d = 65 \times 384 = 24{,}960$; position table $256 \times 384 = 98{,}304$. The `lm_head` is **tied** to `wte`, so it costs zero extra. Embedding total $\approx 123{,}264$.

    **Grand total:** $\approx 10.74\text{M}$ parameters. The blocks dominate ($\sim 99\%$); embeddings are a rounding error here *because the vocab is tiny*. Flip to a 50k BPE vocab and the (tied) embedding becomes $50{,}000 \times 384 \approx 19\text{M}$ — now larger than the entire transformer body. The general rule: $N \approx 12 \, n_\text{layer}\, n_\text{embd}^2$ for the body, a formula worth memorizing for [scaling-law](../03-pretraining/04-scaling-laws.html) and [FLOP](../04-kernels-efficiency/01-roofline-performance.html) estimates.

{{fig:gpt-parameter-budget}}

## Data, the Training Loop, and a Real Run on Tiny Shakespeare

A model is inert without a loss surface to descend. We now build the smallest honest training setup: a character-level dataset, a batching function, an optimizer with the standard weight-decay split, and a loop that actually drives the loss down. We use the classic **Tiny Shakespeare** corpus (a ~1 MB text file of Shakespeare's plays) at the character level, so the tokenizer is trivial — every distinct character is a token — and we can watch the model learn from gibberish to plausible-looking English in minutes.

### Tokenizing and batching

```python
import numpy as np

# --- Build a character-level vocabulary from the raw text ---
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()
chars = sorted(list(set(text)))            # e.g. 65 unique characters
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}   # string -> int
itos = {i: ch for i, ch in enumerate(chars)}   # int    -> string
encode = lambda s: [stoi[c] for c in s]        # text -> list[int]
decode = lambda l: "".join(itos[i] for i in l) # list[int] -> text

# Encode the entire corpus once, then split 90/10 into train/val.
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split, block_size, batch_size, device):
    """Sample `batch_size` random contiguous windows of length block_size.
    x is tokens [i : i+T]; y is the SAME window shifted by one: [i+1 : i+1+T].
    So y[t] is the next-token target for x[t] — the whole autoregressive labelling."""
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))   # random start offsets
    x = torch.stack([d[i     : i + block_size]     for i in ix])  # (B, T)
    y = torch.stack([d[i + 1 : i + 1 + block_size] for i in ix])  # (B, T), shifted
    return x.to(device), y.to(device)
```

The "shift by one" labelling is the entire supervised signal of language modeling: predicting `y` from `x` *is* next-token prediction, and because our `forward` computes a loss at every position, one `(B, T)` batch yields $B \times T$ next-token predictions. The loss is the mean cross-entropy across all of them, which equals the negative log-likelihood the [pretraining objective chapter](../03-pretraining/03-pretraining-objective.html) formalizes:

$$
\mathcal{L} = -\frac{1}{BT}\sum_{b=1}^{B}\sum_{t=1}^{T} \log p_\theta\big(y_{b,t} \mid x_{b, 1:t}\big).
$$

{{fig:next-token-training-signal}}

### The optimizer with a weight-decay split

A subtlety that separates toy code from real training: **not all parameters should be weight-decayed.** Matmul weights (2D tensors) benefit from L2 regularization; biases and LayerNorm gains (1D tensors) should *not* be decayed — shrinking them toward zero damages the model. We build two parameter groups accordingly. We use AdamW, the optimizer of choice for transformers (see [Optimizers](../03-pretraining/09-optimizers.html)).

```python
def configure_optimizer(model, weight_decay, lr, betas, device_type):
    # Collect trainable params, then split by tensor dimensionality.
    params = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params   = [p for p in params.values() if p.dim() >= 2]   # matmuls, embeddings
    nodecay_params = [p for p in params.values() if p.dim() <  2]   # biases, LN gains
    optim_groups = [
        {"params": decay_params,   "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    # 'fused' AdamW is a faster CUDA path when available.
    use_fused = (device_type == "cuda")
    extra = dict(fused=True) if use_fused else dict()
    return torch.optim.AdamW(optim_groups, lr=lr, betas=betas, **extra)
```

### The loop itself

Here is the complete, runnable training loop, with the gradient-clipping and periodic-evaluation machinery you would keep in a real run, but stripped of distributed-training plumbing (covered in [Distributed Training I](../03-pretraining/05-distributed-data-parallel.html)).

```python
import math

# --- Hyperparameters (a small but real configuration) ---
device = "cuda" if torch.cuda.is_available() else "cpu"
config = GPTConfig(block_size=256, vocab_size=vocab_size,
                   n_layer=6, n_head=6, n_embd=384, dropout=0.2, bias=False)
batch_size  = 64
max_iters   = 5000
eval_iter   = 250
grad_clip   = 1.0
learning_rate = 3e-4
warmup_iters  = 200       # linear LR warmup
lr_decay_iters = max_iters
min_lr        = 3e-5      # cosine floor (~lr/10)

model = GPT(config).to(device)
optimizer = configure_optimizer(model, weight_decay=0.1, lr=learning_rate,
                                betas=(0.9, 0.95), device_type=device)

def get_lr(it):
    """Linear warmup then cosine decay to min_lr — the standard LLM schedule."""
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)   # in [0, 1]
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))                 # cosine 1 -> 0
    return min_lr + coeff * (learning_rate - min_lr)

@torch.no_grad()
def estimate_loss(eval_batches=50):
    """Average loss over several batches of train and val — a stable metric."""
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_batches)
        for k in range(eval_batches):
            X, Y = get_batch(split, config.block_size, batch_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# --- Train ---
model.train()
for it in range(max_iters):
    # Set this step's learning rate on every param group.
    lr = get_lr(it)
    for g in optimizer.param_groups:
        g["lr"] = lr

    # Periodically report train/val loss (catches overfitting early).
    if it % eval_iter == 0 or it == max_iters - 1:
        losses = estimate_loss()
        print(f"step {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | lr {lr:.2e}")

    # One optimization step.
    X, Y = get_batch("train", config.block_size, batch_size, device)
    logits, loss = model(X, Y)            # forward: compute loss
    optimizer.zero_grad(set_to_none=True) # clear stale grads (None is faster than 0)
    loss.backward()                       # backward: autograd fills .grad
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # tame loss spikes
    optimizer.step()                      # AdamW update
```

Four lines in that inner loop are the *entire* mechanics of training a neural network, and they recur unchanged in every model in this book: `forward → zero_grad → backward → step`. Everything else — the schedule, clipping, evaluation, distributed wrappers, mixed precision — is engineering around those four lines.

### Checkpointing: save and resume

The A1 deliverable asks for "a training loop with checkpointing," and at single-GPU scale that is not a distributed-systems problem — it is about 15 lines of `torch.save`/`torch.load`. The sharded, topology-agnostic version for multi-GPU and multi-node jobs (FSDP + PyTorch Distributed Checkpoint / DCP) is covered in [Checkpointing, Fault Tolerance & Long-Running Jobs](../03-pretraining/12-checkpointing-fault-tolerance.html); everything below is the honest small-scale baseline that version generalizes.

```python
def save_checkpoint(path, model, optimizer, it, config):
    """Save everything needed to resume training bit-continuously."""
    ckpt = {
        "model":            model.state_dict(),
        "optimizer":        optimizer.state_dict(),
        "iter_num":         it,
        "config":           config,
        "torch_rng_state":  torch.get_rng_state(),
        "cuda_rng_state":   torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(ckpt, path)

def load_checkpoint(path, model, optimizer, device):
    """Restore model, optimizer, and RNG state; return the iter to resume AT."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    # RNG state must be a CPU ByteTensor regardless of map_location, hence .cpu().
    torch.set_rng_state(ckpt["torch_rng_state"].cpu())
    if ckpt["cuda_rng_state"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
    return ckpt["iter_num"] + 1   # resume at the NEXT step, not the saved one

# --- Wire it into the loop ---
resume   = False               # flip to True to resume from ckpt_path
ckpt_path = "ckpt.pt"

start_it = load_checkpoint(ckpt_path, model, optimizer, device) if resume else 0

model.train()
for it in range(start_it, max_iters):
    lr = get_lr(it)
    for g in optimizer.param_groups:
        g["lr"] = lr

    if it % eval_iter == 0 or it == max_iters - 1:
        losses = estimate_loss()
        print(f"step {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | lr {lr:.2e}")
        save_checkpoint(ckpt_path, model, optimizer, it, config)   # checkpoint on eval steps

    X, Y = get_batch("train", config.block_size, batch_size, device)
    logits, loss = model(X, Y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
```

Two things worth internalizing. First, **weight tying survives the round-trip automatically**: `wte.weight` and `lm_head.weight` are the *same* tensor object, so it appears once in `state_dict()` and reloads shared — there is nothing special to do. Second, **saving RNG state and the optimizer's moment buffers is what makes a kill/resume bit-continuous**: after resuming, the first logged loss should continue from roughly where it left off (modulo ordinary batch-sampling noise), *not* jump back up toward $\ln V$. If you see a jump on resume, you forgot to restore the optimizer state (AdamW's first/second moment estimates) or the RNG/data-iterator state — the model is stepping from a warm optimizer trajectory it thinks is cold.

For sizing intuition: our default ~10.7M-parameter model produces a checkpoint of roughly 130 MB in fp32 — the model weights plus AdamW's two moment buffers per parameter (~3× the raw parameter count) — trivial to write on a laptop or a single GPU. At multi-node scale, saving every rank's full state to one file stops being trivial; you either checkpoint only on rank 0 or use DCP's sharded format, both covered in [Checkpointing, Fault Tolerance & Long-Running Jobs](../03-pretraining/12-checkpointing-fault-tolerance.html).

!!! tip "Practitioner tip: what a healthy loss curve looks like here"
    For character-level Tiny Shakespeare with this config, the loss starts near $\ln(65) \approx 4.17$ — that is exactly the loss of a uniform random guesser over 65 characters, and seeing your *first* logged loss land near it is the cheapest sanity check that your data, masking, and loss are wired correctly. A working run drops below ~1.5 within a couple thousand steps; below ~1.0 it produces Shakespeare-flavored text with real words and pseudo-grammar. If your loss *starts* far from $\ln V$, suspect a labelling, masking, or initialization bug before you suspect the model.

!!! warning "Common pitfall: the off-by-one in targets, and a leaked causal mask"
    Two bugs produce a loss that looks suspiciously *too good* (near zero) early on. (1) If `y` is not shifted by exactly one position relative to `x`, you are training the model to copy its input — trivial and useless. (2) If the causal mask is missing or wrong, each position can read the token it is supposed to predict, so the model cheats. Both are invisible in the code's shapes and silent in autograd; the only tell is an implausibly low training loss that does *not* generalize. Always reason about whether position $t$ can see token $t+1$.

## Sampling: Turning Logits Into Text

A trained GPT is a next-token distribution. **Generation** is the loop that repeatedly samples a token, appends it, and feeds the extended sequence back in. This is the *autoregressive* loop, and it is where decoding strategy — temperature, top-k, top-p — lives. (The decoding chapter [Sampling Strategies & Decoding Algorithms](../07-inference-serving/09-sampling-decoding.html) goes deeper; here we implement the core ones from scratch.)

```python
@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
    """Autoregressively extend idx (B, T) by max_new_tokens.
       temperature: >1 flattens the distribution (more random), <1 sharpens it.
       top_k: keep only the k highest-prob tokens before sampling.
       top_p: nucleus sampling — keep the smallest set of tokens whose cumulative
              probability exceeds p."""
    model.eval()
    for _ in range(max_new_tokens):
        # 1) Crop context to block_size — the model cannot attend beyond it.
        idx_cond = idx if idx.size(1) <= model.config.block_size \
                   else idx[:, -model.config.block_size:]

        # 2) Forward pass; take logits at the LAST position only.
        logits, _ = model(idx_cond)            # (B, 1, vocab_size)
        logits = logits[:, -1, :] / temperature  # (B, vocab_size), temperature-scaled

        # 3) Optional top-k filtering: zero out everything below the k-th best logit.
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")

        # 4) Convert to probabilities.
        probs = F.softmax(logits, dim=-1)      # (B, vocab_size)

        # 5) Optional top-p (nucleus) filtering on the probabilities.
        if top_p is not None:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cumprobs = torch.cumsum(sorted_probs, dim=-1)
            # Mask tokens once cumulative prob has exceeded p (keep the boundary token).
            remove = cumprobs - sorted_probs > top_p
            sorted_probs[remove] = 0.0
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)   # renormalize
            # Scatter the filtered probs back to vocabulary order.
            probs = torch.zeros_like(probs).scatter_(1, sorted_idx, sorted_probs)

        # 6) Sample one token from the (filtered, renormalized) distribution.
        next_id = torch.multinomial(probs, num_samples=1)   # (B, 1)

        # 7) Append and repeat. The new token becomes part of next step's context.
        idx = torch.cat((idx, next_id), dim=1)
    return idx


# --- Generate from a single newline as the prompt ---
context = torch.tensor([[stoi["\n"]]], dtype=torch.long, device=device)
out_ids = generate(model, context, max_new_tokens=500, temperature=0.8, top_k=40)
print(decode(out_ids[0].tolist()))
```

### What the knobs actually do

**Temperature** $\tau$ divides the logits before softmax: $p_i \propto \exp(z_i / \tau)$. As $\tau \to 0$ the distribution collapses onto the single most likely token (greedy, deterministic); as $\tau \to \infty$ it approaches uniform (maximally random). $\tau = 1$ samples from the model's exact distribution. Practitioners use $\tau \approx 0.7$–$1.0$ for creative text and lower for factual or code generation.

**Top-k** keeps only the $k$ highest-probability tokens and renormalizes, hard-truncating the long tail of garbage tokens. **Top-p (nucleus)** sampling, from Holtzman et al., instead keeps the smallest set of tokens whose cumulative probability reaches $p$ (e.g. 0.9) — an *adaptive* cutoff that keeps many tokens when the model is uncertain and few when it is confident. Top-p generally produces more natural text than a fixed top-k because the truncation adapts to the local entropy of the distribution.

{{fig:sampling-temperature-topk-topp}}

!!! example "Worked example: temperature on a 4-token distribution"
    Suppose the model emits logits $z = [3.0,\ 1.0,\ 0.5,\ -1.0]$ for four candidate next characters.

    At $\tau = 1.0$: softmax gives $\approx [0.78,\ 0.105,\ 0.064,\ 0.014]$. The top token is likely but not certain.

    At $\tau = 0.5$ (sharper): logits become $[6,\ 2,\ 1,\ -2]$; softmax $\approx [0.97,\ 0.018,\ 0.0066,\ 0.00033]$. Now the top token is nearly certain — text becomes repetitive and "safe."

    At $\tau = 2.0$ (flatter): logits become $[1.5,\ 0.5,\ 0.25,\ -0.5]$; softmax $\approx [0.46,\ 0.17,\ 0.13,\ 0.062]$ (plus mass elsewhere) — the rare fourth token now has a real chance, so text gets more surprising and more error-prone. Same model, three very different writers, controlled by one scalar in the denominator.

### The KV-cache connection

Our `generate` recomputes attention over the *entire* growing prefix at every step — $\mathcal{O}(T^2)$ work to produce $T$ tokens. Real serving systems never do this: because the causal mask makes each token's keys and values independent of future tokens, you compute each position's K and V *once* and cache them, so each new token costs only $\mathcal{O}(T)$. That **KV cache** is the single most important inference optimization, and it is the subject of [The Anatomy of LLM Inference](../07-inference-serving/01-anatomy-inference.html). For learning and small-scale generation, the simple recompute-everything loop above is correct and clear; just know that production swaps it out.

!!! interview "Interview Corner"
    **Q:** You implemented a GPT and it trains, but generation produces repetitive, degenerate loops ("the the the the..."). The training loss is reasonable. What is going on, and how do you diagnose and fix it?

    **A:** Low training loss only means the model is good at *next-token prediction under teacher forcing* — it always conditions on the ground-truth prefix. At generation time it conditions on its *own* outputs, and any small bias toward high-probability tokens compounds, a phenomenon called exposure bias. Degenerate repetition is the classic symptom of decoding too greedily: with low or zero temperature (or pure argmax), the model locks onto its single most-likely continuation, which is often a self-reinforcing loop because repeating a phrase makes that phrase even more likely under the model. **Diagnosis:** first confirm the model itself is fine by checking validation loss and by sampling at $\tau = 1.0$ with no truncation — if that is coherent, the problem is the decoding configuration, not the weights. **Fixes, in order:** raise temperature toward ~0.8–1.0; switch from greedy/top-k to **top-p (nucleus) sampling** (~0.9), which Holtzman et al. showed specifically cures the "neural text degeneration" loops; add a repetition penalty that down-weights recently emitted tokens. If repetition persists even at $\tau = 1.0$ with nucleus sampling, *then* suspect undertraining or a bug (e.g., a too-small `block_size` so the model cannot see far enough back to avoid repeating). The key insight an interviewer wants: training loss and generation quality are decoupled, and the most common cause of bad samples from a correctly-trained model is the sampling strategy.

## From This File to a Frontier Model

What we just built is, structurally, GPT-2. The leap from this 10M-parameter character model to a frontier system is almost entirely *scale and refinement*, not new ideas — which is precisely why building this from scratch is so clarifying. The components that change are localized and we have chapters for each:

- **Tokenizer:** swap character-level for a [BPE tokenizer](../02-transformer/01-tokenization.html) with a 50k–200k vocabulary.
- **Positions:** replace the learned `wpe` table with [RoPE](../02-transformer/05-positional-encoding.html), which generalizes to longer contexts and is what every modern model uses.
- **Attention:** replace dense multi-head attention with [GQA or MLA](../02-transformer/04-mha-gqa-mla.html) to shrink the KV cache, and serve it with [FlashAttention](../04-kernels-efficiency/02-flash-attention-1.html).
- **Norm & activation:** swap LayerNorm for RMSNorm and GELU for SwiGLU (see [Modern Architecture Improvements](../02-transformer/10-modern-arch-improvements.html)).
- **MLP:** optionally replace the dense MLP with a [Mixture-of-Experts](../02-transformer/09-mixture-of-experts.html) layer to grow capacity without growing per-token FLOPs.
- **Scale:** more layers, more width, vastly more data — governed by [scaling laws](../03-pretraining/04-scaling-laws.html) and executed with [distributed training](../03-pretraining/05-distributed-data-parallel.html) and [mixed precision](../03-pretraining/08-mixed-precision-fp8.html).
- **Behavior:** the pretrained base model is then [supervised-fine-tuned](../05-posttraining-alignment/01-sft-instruction-tuning.html) and [RLHF-aligned](../05-posttraining-alignment/05-rlhf-reward-modeling.html) into an assistant.

Every one of those is a swap of a single module or a multiplication of a single number in our `GPTConfig`. The skeleton — embed, stack pre-norm blocks on a residual stream, project to logits, train with next-token cross-entropy, sample autoregressively — is the same skeleton at every scale. You have now built it end to end.

## A Modern GPT, Assembled

The previous section *listed* the module swaps that turn GPT-2 into a modern model. This section *does* them. We assemble RoPE + RMSNorm + SwiGLU + weight tying into a single trained model — exactly the Llama-family block — so you have a worked RoPE-into-attention integration instead of having to invent one. Rather than reinvent the helpers, we reuse the verified implementations from earlier chapters verbatim, and we train on the *same* Tiny Shakespeare setup as the baseline to confirm the modern stack reaches an equal-or-better loss.

```python
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════
# Reused verbatim from Chapter 2.5 (Positional Encoding)
# ═══════════════════════════════════════════════════════════════════════════

def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0,
                     device=None, dtype=torch.float32):
    """Precompute cos/sin tables for RoPE (Llama / rotate_half convention).
    Returns cos, sin of shape (seq_len, head_dim)."""
    assert head_dim % 2 == 0, "RoPE needs an even head dimension."
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)                 # (seq_len, d/2)
    emb = torch.cat([freqs, freqs], dim=-1)             # (seq_len, d)
    return emb.cos().to(dtype), emb.sin().to(dtype)

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Map (x1, x2) -> (-x2, x1) where x1,x2 are the two halves of the last dim."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x of shape (B, n_head, T, head_dim); cos, sin: (T, head_dim)."""
    cos = cos[None, None, :, :]   # (1, 1, T, head_dim) — broadcasts over batch & heads
    sin = sin[None, None, :, :]
    return x * cos + rotate_half(x) * sin

# ═══════════════════════════════════════════════════════════════════════════
# Reused verbatim from Chapter 2.6 (The Transformer Block)
# ═══════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm (Zhang & Sennrich, 2019). No bias, no mean subtraction."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight

class SwiGLUFFN(nn.Module):
    """FFN(x) = W_down( Swish(W_gate x) * W_up x ). hidden_dim defaults to
    8/3 * dim rounded up to a multiple of 64, matching Llama's convention."""
    def __init__(self, dim: int, hidden_dim: Optional[int] = None,
                 bias: bool = False, dropout: float = 0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(8 * dim / 3)
            hidden_dim = 64 * ((hidden_dim + 63) // 64)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_up   = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_down = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))
        up   = self.w_up(x)
        return self.dropout(self.w_down(gate * up))
```

For `n_embd=384`, `SwiGLUFFN`'s default hidden dim is `64 * ceil(8*384/3/64) = 1024`, giving `3 * 384 * 1024 = 1,179,648 = 8 * n_embd^2` parameters — **identical** to the GELU MLP's `2 * (4 * n_embd^2)`. The swap is capacity-neutral; every difference in the final loss comes from the architecture, not from extra parameters.

### The modern attention block: RoPE applied inside the heads

```python
class ModernCausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, cos, sin):
        # cos, sin: (T, head_dim), already sliced to this sequence length by the caller.
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)          # each (B, T, C)

        # Reshape into heads FIRST: (B, T, C) -> (B, n_head, T, head_dim).
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # THEN rotate — RoPE must be applied to q and k only, AFTER the head
        # split. Rotating the (B,T,C) tensor before splitting into heads would
        # pair dimensions across head boundaries instead of within a single
        # head's head_dim block — a silent correctness bug, not a crash.
        # v is never rotated: it carries content, not position.
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        # Under bf16/autocast, cos/sin must match q/k's dtype. Either build the
        # cache directly in the model's working dtype, or do the safer:
        #   q = apply_rope(q.float(), cos, sin).to(q.dtype)

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class ModernBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm_1 = RMSNorm(config.n_embd)
        self.attn   = ModernCausalSelfAttention(config)
        self.norm_2 = RMSNorm(config.n_embd)
        self.mlp    = SwiGLUFFN(config.n_embd, bias=config.bias, dropout=config.dropout)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm_1(x), cos, sin)
        x = x + self.mlp(self.norm_2(x))
        return x
```

### The modern GPT: no `wpe`, cache built once, weight-tied

```python
class ModernGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None and config.block_size is not None
        head_dim = config.n_embd // config.n_head
        assert head_dim % 2 == 0, "RoPE needs an even head_dim"   # 384/6=64: OK
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),   # NO wpe: RoPE replaces it
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([ModernBlock(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # WEIGHT TYING — unchanged from the baseline GPT.
        self.transformer.wte.weight = self.lm_head.weight

        # Build the RoPE cache ONCE, for the full block_size, and register it
        # as a non-persistent buffer: it is cheap to recompute, so we don't
        # want it bloating every checkpoint.
        cos, sin = build_rope_cache(config.block_size, head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            # Still exactly 2 residual adds per block (attn, mlp), so the
            # 1/sqrt(2*n_layer) factor from the baseline is unchanged.
            if pn.endswith("c_proj.weight") or pn.endswith("w_down.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / (2 * config.n_layer) ** 0.5)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"sequence length {T} > block_size"

        x = self.transformer.drop(self.transformer.wte(idx))   # (B, T, n_embd) — no pos_emb add
        cos = self.rope_cos[:T]     # slice the cache to the current sequence length
        sin = self.rope_sin[:T]

        for block in self.transformer.h:
            x = block(x, cos, sin)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss
```

Because our `generate` loop recomputes the entire prefix at every step, position ids are always `0..T-1` — so slicing the cache to `[:T]` is exactly correct, and the *same* `generate` function from the Sampling section works unchanged on a `ModernGPT` (it only ever calls `model(idx_cond)`). With a real KV cache, new tokens arrive one at a time at *absolute* positions beyond `T`, so you would instead index the RoPE cache at that absolute position rather than always starting from `0` — a detail deferred to [The Anatomy of LLM Inference](../07-inference-serving/01-anatomy-inference.html).

### Training and verifying the swap

Train it with the exact same hyperparameters and loop as the baseline — only the model class changes:

```python
model = ModernGPT(config).to(device)   # same GPTConfig: block_size=256, n_layer=6,
                                        # n_head=6, n_embd=384, dropout=0.2, bias=False
optimizer = configure_optimizer(model, weight_decay=0.1, lr=learning_rate,
                                betas=(0.9, 0.95), device_type=device)
# ... identical training loop as before, just with this model/optimizer ...

print(sum(p.numel() for p in model.parameters()))   # sanity-check the param count
```

Four checks confirm the integration is wired correctly, not just "not crashing":

1. **Param count.** `print(sum(p.numel() for p in model.parameters()))` should read **~10.6M** — slightly *smaller* than the baseline's 10.74M, because dropping the learned `wpe` table saves `256 * 384 = 98,304` parameters while RoPE adds none (it is a fixed, non-parametric cache).
2. **Init loss.** The first logged loss should still land near $\ln(65) \approx 4.174$ — the same uniform-guess sanity check as the baseline; RoPE and RMSNorm don't change what an untrained model's loss should look like.
3. **Final loss.** After the same 5000 iterations, validation loss should land essentially on top of the baseline's ~1.47 — commonly a hair lower, around 1.45–1.46. Reaching an equal-or-better loss than the GELU/LayerNorm/learned-position baseline is your confirmation that the RoPE-into-attention wiring, the RMSNorm placement, and the SwiGLU gating are all correct; a *much worse* loss here is the classic symptom of RoPE applied at the wrong tensor rank (see the load-bearing comment above).
4. For a numeric unit test of the rotation itself (not the whole model), reuse the RoPE relative-position property already verified in [Positional Encoding](../02-transformer/05-positional-encoding.html).

**Scaling this exercise to your hardware.** On a laptop or CPU, shrink to `n_layer=4, n_embd=128` and run a few hundred iterations — you should still watch the loss fall from ~4.17. On a single GPU, the defaults above take roughly 2–4 minutes for 5000 iterations on an A100 or 3090, matching the baseline's timing exactly, since param count and FLOPs are essentially unchanged. On an 8-GPU node or a multi-node cluster, the module itself is byte-for-byte the same — you only wrap `ModernGPT` in DDP or FSDP per [Distributed Data Parallelism](../03-pretraining/05-distributed-data-parallel.html); there is no architectural change at scale.

What you just built is, module for module, Llama's decoder block. The next section maps every piece onto the actual `transformers` source so you can read it directly.

## Reading the Real Thing: This Chapter in `transformers`

The `ModernGPT` above is structurally identical to Hugging Face's `LlamaForCausalLM`. That means learning to read one file — `transformers/models/llama/modeling_llama.py` — unlocks navigating the whole library, because every Llama-family model (and most decoder-only models released since) is a variation on the same skeleton.

**Module-for-module mapping:**

| Our name | `transformers` name | Note |
|---|---|---|
| `RMSNorm` | `LlamaRMSNorm` | Same math, same no-bias/no-mean-subtraction design. |
| `ModernCausalSelfAttention` | `LlamaAttention` | Uses *separate* `q_proj`/`k_proj`/`v_proj`/`o_proj` instead of our fused `c_attn`, and exposes `num_key_value_heads` for GQA — set it equal to `num_attention_heads` to recover our dense MHA. |
| `build_rope_cache` + `apply_rope` | `LlamaRotaryEmbedding` + `apply_rotary_pos_emb(q, k, cos, sin)` | Same `rotate_half` convention. Structurally, recent `transformers` builds `cos`/`sin` **once** at the `LlamaModel` level and threads `(cos, sin)` via `position_embeddings` into every `LlamaDecoderLayer` — exactly our design, where the cache is built once in `ModernGPT.__init__` and passed down to each `ModernBlock`. |
| `SwiGLUFFN` | `LlamaMLP` | `gate_proj`/`up_proj`/`down_proj` = our `w_gate`/`w_up`/`w_down`; `act_fn` = SiLU. |
| `ModernBlock` | `LlamaDecoderLayer` | `input_layernorm`, `self_attn`, `post_attention_layernorm`, `mlp` — same pre-norm wiring. |
| `ModernGPT` trunk | `LlamaModel` | `embed_tokens` = our `wte`, `layers` = our `h`, `norm` = our `ln_f`. |
| `ModernGPT` + `lm_head` + loss | `LlamaForCausalLM` | `transformers` shifts logits/labels **internally** inside `forward` (`shift_logits`/`shift_labels`), whereas we shift in `get_batch` — so you feed `transformers` *unshifted* labels. |
| weight tying | `config.tie_word_embeddings` | `True` for small models (e.g. Llama-3.2-1B, SmolLM), `False` for Llama-2/3 at 7B+, which untie the head. |

**Attention backends.** The `attn_implementation` argument to `from_pretrained` selects the attention kernel: `"eager"` is the readable, pure-PyTorch masked-softmax path (the manual `else` branch in our baseline `CausalSelfAttention`); `"sdpa"` routes through `F.scaled_dot_product_attention` (exactly our `is_causal=True` path); `"flash_attention_2"` calls the FlashAttention-2 kernels directly. You select it explicitly:

```python
model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="sdpa")
```

**Generation.** Our sample-append loop corresponds to `GenerationMixin.generate` in `transformers/generation/utils.py`. Temperature, top-k, and top-p become `LogitsProcessor`/`LogitsWarper` objects composed into a pipeline; the greedy-vs-sample choice is dispatched inside the internal `_sample` routine; and the KV cache we deferred throughout this chapter is `DynamicCache`. In short, `generate()` is our loop plus a KV cache, stopping criteria, and batching.

**A runnable orientation.** Load any small Llama-family checkpoint and print its module tree — it will read as an isomorphism to `ModernGPT`:

```python
from transformers import AutoModelForCausalLM

m = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
print(m)
# LlamaForCausalLM(
#   (model): LlamaModel(
#     (embed_tokens): Embedding(...)
#     (layers): ModuleList(
#       (0-N): LlamaDecoderLayer(
#         (self_attn): LlamaAttention(...)
#         (mlp): LlamaMLP(...)
#         (input_layernorm): LlamaRMSNorm(...)
#         (post_attention_layernorm): LlamaRMSNorm(...)
#       )
#     )
#     (norm): LlamaRMSNorm(...)
#   )
#   (lm_head): Linear(...)
# )
```

To read the source itself:

```python
python -c "import transformers, os; print(os.path.dirname(transformers.__file__))"
```

then open `models/llama/modeling_llama.py` for the model and `generation/utils.py` for `generate`. You now have the vocabulary to read both top to bottom.

!!! key "Key Takeaways"
    - A GPT is a function from token IDs to per-position next-token logits, built as: token + positional embeddings → a stack of pre-norm Transformer blocks on a shared **residual stream** → final LayerNorm → a linear `lm_head` to vocabulary logits.
    - The **residual stream** is the organizing principle: every block *reads from and adds to* a width-`n_embd` highway. Attention mixes information *across* tokens; the MLP processes *each* token independently.
    - The model body has $\approx 12\,n_\text{layer}\,n_\text{embd}^2$ parameters; MLPs hold $8 n_\text{embd}^2$ per block (twice attention's $4 n_\text{embd}^2$). The tied embedding/`lm_head` matrix dominates only when the vocabulary is large.
    - **Weight tying** (share `wte` with `lm_head`) saves $V \times n_\text{embd}$ params and regularizes; the **scaled residual init** ($\div\sqrt{2\,n_\text{layer}}$ on output projections) keeps the residual stream's variance bounded as depth grows.
    - Training is four lines — `forward → zero_grad → backward → step` — wrapped in a learning-rate schedule (warmup + cosine decay), gradient clipping, and a no-weight-decay group for 1-D parameters. The loss is mean next-token cross-entropy over all $B\times T$ positions.
    - The first logged loss should be $\approx \ln V$ (uniform-guess baseline); a loss far from it signals a labelling, masking, or init bug *before* you blame the model.
    - **Generation** is the autoregressive sample-append loop; **temperature**, **top-k**, and **top-p** shape the distribution. Repetitive degenerate text is almost always a decoding (too-greedy) problem, not a weights problem — reach for nucleus sampling.
    - Scaling this nanoGPT to a frontier model is mostly module swaps (BPE, RoPE, GQA, RMSNorm/SwiGLU, MoE) and more data/compute — the skeleton is invariant across scale.

!!! sota "State of the Art & Resources (2026)"
    The decoder-only GPT architecture introduced in GPT-2 (2019) remains the structural blueprint for every frontier LLM — Llama 3, GPT-4, Gemini, Claude — with targeted module swaps (RoPE, GQA, RMSNorm/SwiGLU) rather than structural overhauls. Andrej Karpathy's nanoGPT is the canonical minimal reference implementation that makes this lineage directly readable.

    **Foundational work**

    - [Vaswani et al., *Attention Is All You Need* (2017)](https://arxiv.org/abs/1706.03762) — the original Transformer; the block structure assembled in this chapter descends directly from it.
    - [Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2, 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — establishes the pre-norm decoder-only architecture, the 0.02 init, and the scaled-residual initialization used here.
    - [Press & Wolf, *Using the Output Embedding to Improve Language Models* (2017)](https://arxiv.org/abs/1608.05859) — the justification for weight tying between the input embedding and the output head.
    - [Holtzman et al., *The Curious Case of Neural Text Degeneration* (2020)](https://arxiv.org/abs/1904.09751) — diagnoses repetitive/degenerate sampling and introduces top-p (nucleus) sampling.
    - [Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (AdamW, 2019)](https://arxiv.org/abs/1711.05101) — the optimizer and the matmul-vs-bias weight-decay split used in the training loop.

    **Recent advances (2023–2026)**

    - [Grattafiori et al., *The Llama 3 Herd of Models* (Meta, 2024)](https://arxiv.org/abs/2407.21783) — shows how the nanoGPT skeleton scales to 405B parameters with RoPE, GQA, and SwiGLU, illustrating exactly the module-swap path described at the end of this chapter.
    - [Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (2022)](https://arxiv.org/abs/2205.14135) — the IO-aware kernel that powers `F.scaled_dot_product_attention`; understanding it explains why the `is_causal=True` path in this chapter's code is fast in practice.

    **Open-source & tools**

    - [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — the minimal, readable GPT-2 training repo this chapter follows; ~300 lines each for the model and the training loop.
    - [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt) — step-by-step git history accompanying the video lecture, letting you replay the build commit by commit.

    **Go deeper**

    - [Karpathy, *Let's build GPT: from scratch, in code, spelled out* (2023)](https://www.youtube.com/watch?v=kCc8FmEb1nY) — the ~2-hour video lecture that walks through every line of this chapter's code; the best single resource for internalizing how these pieces assemble.
    - [Karpathy, *Neural Networks: Zero to Hero* (course)](https://karpathy.ai/zero-to-hero.html) — the full course building from backprop fundamentals up through GPT; this chapter is the capstone of that curriculum.

## Further reading

- Karpathy — *nanoGPT* (code repository) and *Let's build GPT: from scratch, in code, spelled out* (video lecture). The minimal, readable GPT implementation this chapter follows; the canonical starting point.
- Radford, Wu, Child, Luan, Amodei, Sutskever — *Language Models are Unsupervised Multitask Learners* (GPT-2, 2019). Introduces the pre-norm decoder-only architecture, the 0.02 init, and the scaled-residual initialization we use.
- Vaswani et al. — *Attention Is All You Need* (2017). The original Transformer; the block structure assembled here descends directly from it.
- Press & Wolf — *Using the Output Embedding to Tie Word Vectors* (2017). The justification for weight tying between the input embedding and output head.
- Holtzman, Buys, Du, Forbes, Choi — *The Curious Case of Neural Text Degeneration* (2020). Diagnoses repetitive/degenerate sampling and introduces top-p (nucleus) sampling.
- Loshchilov & Hutter — *Decoupled Weight Decay Regularization* (AdamW, 2019). The optimizer and the matmul-vs-bias weight-decay split used in the training loop.
