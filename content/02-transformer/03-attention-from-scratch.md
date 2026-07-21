# 2.3 The Attention Mechanism From Scratch

If you remember one equation from this entire book, make it this one:

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

Every GPT, every Llama, every Claude, every Gemini — at its computational heart — is this line, stacked and repeated a few hundred times. The Multi-Layer Perceptrons (MLPs) store knowledge; the normalization layers keep gradients sane; the positional encodings tell the model *where* tokens are. But **attention is the only operation in the Transformer that lets one token's representation depend on another token's representation.** It is the mechanism for *mixing information across the sequence*. Remove it and you have an expensive per-token MLP that can never relate "it" to the noun three sentences ago.

This chapter builds that one line from the ground up. We will derive it as **differentiable soft retrieval** — a database lookup that returns a smooth blend of values rather than a single hard match — and we will see *why every piece is there*: why we project into queries, keys, and values; why we take a dot product; why we divide by $\sqrt{d_k}$ (a detail that looks cosmetic but actually controls whether your network trains at all); why softmax and not something else; and how masking turns the same machinery into either a bidirectional encoder or a causal language model.

We assume you are comfortable with [Linear Algebra for Deep Learning](../01-foundations/01-linear-algebra.html) (matrix multiplication, transposes, the dot product as similarity) and with the softmax and cross-entropy from [Probability, Statistics & Information Theory](../01-foundations/02-probability-information.html). We will write everything twice: once in pure NumPy so you can see the arithmetic with no framework magic, and once in PyTorch with autograd so it is actually usable. This is the crown-jewel chapter; we will take our time.

## From Hard Retrieval to Soft Retrieval

The cleanest way to *understand* attention is to start with a thing you already know: a dictionary lookup, or a key–value store like Redis or a Python `dict`.

A hard key–value store holds a set of `(key, value)` pairs. You issue a **query**, the store finds the key that exactly matches your query, and returns the associated **value**:

```python
store = {"paris": "france", "tokyo": "japan", "lima": "peru"}
store["tokyo"]   # -> "japan"   (exact match on the key "tokyo")
```

Three things are happening, and naming them precisely is the whole trick:

1. A **query** ($q$) — the thing you are looking up ("tokyo").
2. A set of **keys** ($k_1, \dots, k_n$) — the addressable handles ("paris", "tokyo", "lima").
3. A set of **values** ($v_1, \dots, v_n$) — the payloads you actually want back.

The lookup compares the query against every key, picks the matching one, and returns its value. In a Python `dict` the comparison is exact string equality and the selection is "take exactly one." That makes it **non-differentiable**: there is no smooth notion of "the query is 70% like this key." You cannot backpropagate through `store["tokyo"]`.

Attention is what you get when you make every step of that lookup *soft and differentiable*:

- **Keys and queries become vectors**, and "match" becomes a continuous **similarity score** — the dot product $q \cdot k$. A bigger dot product means a better match.
- **Selection becomes a weighted average** instead of a hard pick. We convert the scores into a probability distribution with softmax, then return the *blend* of all values weighted by how much each key matched. If the query is 70% like key 3 and 30% like key 7, we return $0.7\,v_3 + 0.3\,v_7$.

{{fig:attn-hard-vs-soft-retrieval}}

This "soft retrieval" framing is not a loose analogy bolted on after the fact — it is *literally* what the math computes, and it pays off immediately. It tells you what each tensor *means*: a query says "here is what I am looking for," a key says "here is what I contain, address me with this," and a value says "if you decide to read me, here is the information I hand back." It also tells you why attention can implement things like "copy the value from the most similar earlier token" (induction heads), "average over all tokens that mention the subject," or "ignore everything except the punctuation." Those are all just different shapes of the weight distribution.

!!! note "Aside: where Q, K, V come from"
    In self-attention, queries, keys, and values are all *learned linear projections of the same input*. If $x_i \in \mathbb{R}^{d_\text{model}}$ is the representation of token $i$, then $q_i = x_i W_Q$, $k_i = x_i W_K$, $v_i = x_i W_V$ with learned matrices $W_Q, W_K, W_V$. The token gets to ask a question ($q$), advertise itself ($k$), and offer content ($v$) — three different *roles* carved out of one representation. In cross-attention (encoder–decoder), the queries come from one sequence and the keys/values from another; see [Architecture Variants: Encoder-Decoder, Decoder-Only & Prefix-LM](../02-transformer/08-architecture-variants.html).

## Scaled Dot-Product Attention, One Query at a Time

{{fig:attention-flow}}

Let us derive the formula for a single query $q \in \mathbb{R}^{d_k}$ attending over $n$ key–value pairs. Stack the keys as rows of a matrix $K \in \mathbb{R}^{n \times d_k}$ and the values as rows of $V \in \mathbb{R}^{n \times d_v}$. (We allow $d_v \neq d_k$ in general, though in practice they are usually equal.)

**Step 1 — Score every key against the query.** The similarity between the query and key $j$ is the dot product $s_j = q \cdot k_j = \sum_{c=1}^{d_k} q_c\, k_{j,c}$. Computing all $n$ scores at once is a single matrix–vector product:

$$
s = K q \in \mathbb{R}^{n}, \qquad s_j = q \cdot k_j.
$$

The dot product is the natural similarity here for a reason: it is large and positive when $q$ and $k_j$ point in the same direction, near zero when they are orthogonal, and negative when they oppose. It rewards both *alignment of direction* and *magnitude*. (Compare to [Embeddings & The Input Pipeline](../02-transformer/02-embeddings-input.html), where the same dot-product-as-similarity intuition underlies nearest-neighbor search over embeddings.)

**Step 2 — Scale by $\sqrt{d_k}$.** We divide every score by $\sqrt{d_k}$:

$$
\tilde{s}_j = \frac{q \cdot k_j}{\sqrt{d_k}}.
$$

This is the "scaled" in scaled dot-product attention, and we devote the next section to *why* it matters. For now, take it as: it keeps the scores from blowing up as the head dimension $d_k$ grows.

**Step 3 — Softmax into attention weights.** Turn the scaled scores into a probability distribution over the $n$ positions:

$$
a_j = \operatorname{softmax}(\tilde{s})_j = \frac{e^{\tilde{s}_j}}{\sum_{\ell=1}^{n} e^{\tilde{s}_\ell}}, \qquad \sum_{j=1}^{n} a_j = 1, \quad a_j > 0.
$$

The vector $a \in \mathbb{R}^n$ is the **attention distribution** for this query: $a_j$ is the fraction of "attention" the query pays to position $j$. Softmax is the right choice because it is the smooth, differentiable relaxation of $\arg\max$: it concentrates mass on the largest scores (recovering near-hard selection when one score dominates) while staying differentiable everywhere and always producing a valid probability vector. We unpack softmax's numerical behavior — overflow, the log-sum-exp trick, the temperature induced by the scale — below.

**Step 4 — Blend the values.** The output is the attention-weighted sum of the value vectors:

$$
\operatorname{attn}(q, K, V) = \sum_{j=1}^{n} a_j\, v_j = a^\top V \in \mathbb{R}^{d_v}.
$$

That is the entire single-query mechanism. Notice the output lives in value-space ($\mathbb{R}^{d_v}$), is a *convex combination* of the value vectors (it lies inside their convex hull — it can never extrapolate beyond the values it is given), and is differentiable with respect to $q$, every $k_j$, and every $v_j$. Information has flowed from the positions the query cared about into a single output vector.

{{fig:attn-output-convex-hull}}

### Batching all queries at once

In practice we have not one query but one *per position*: $n$ tokens each emit a query, so $Q \in \mathbb{R}^{n \times d_k}$. We want every query to attend over every key independently. Stacking step 1 over all queries turns the matrix–vector product into a matrix–matrix product:

$$
S = \frac{QK^\top}{\sqrt{d_k}} \in \mathbb{R}^{n \times n}, \qquad S_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}}.
$$

This $n \times n$ object is **the attention matrix** (pre-softmax it is the "score matrix" or "logits"; post-softmax it is the "attention weights"). Row $i$ holds the scores of query $i$ against all keys. We softmax **along each row** (dimension `-1`, the key axis) so each query's weights sum to 1, then multiply by $V$:

$$
\operatorname{Attention}(Q, K, V) = \underbrace{\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)}_{A \,\in\, \mathbb{R}^{n \times n}} V \;\in\; \mathbb{R}^{n \times d_v}.
$$

Output row $i$ is query $i$'s personalized blend of the values. We have arrived at the headline equation, and now every symbol has a meaning rooted in soft retrieval.

{{fig:attn-sdpa-pipeline-masked}}

!!! warning "Common pitfall: softmax along the wrong axis"
    The single most common attention bug is applying softmax over the *query* axis instead of the *key* axis. In `S` of shape `(n_queries, n_keys)`, you must normalize over `dim=-1` (keys) so that **each query's** weights sum to 1. Normalizing over `dim=0` (queries) silently produces a model that trains — slowly, badly, and confusingly — to a worse loss. Always assert `A.sum(dim=-1)` is all-ones during development.

## Why Divide by $\sqrt{d_k}$? The Variance Argument

The scale factor $1/\sqrt{d_k}$ looks like a footnote. It is not. Get it wrong and softmax saturates, gradients vanish, and the network refuses to learn. Here is the argument from *Attention Is All You Need* (Vaswani et al., 2017), made fully explicit.

Treat the components of a query $q$ and a key $k$ as independent random variables, each with mean 0 and variance 1 — a reasonable approximation at initialization, where the projection weights are small-random and the inputs are normalized. The raw score is their dot product:

$$
s = q \cdot k = \sum_{c=1}^{d_k} q_c\, k_c.
$$

Each term $q_c k_c$ is a product of two independent zero-mean, unit-variance variables, so it has mean $\mathbb{E}[q_c k_c] = \mathbb{E}[q_c]\mathbb{E}[k_c] = 0$ and variance $\operatorname{Var}(q_c k_c) = \mathbb{E}[q_c^2]\mathbb{E}[k_c^2] = 1 \cdot 1 = 1$. Summing $d_k$ independent such terms, the variances add:

$$
\mathbb{E}[s] = 0, \qquad \operatorname{Var}(s) = \sum_{c=1}^{d_k} \operatorname{Var}(q_c k_c) = d_k.
$$

So the **standard deviation of the raw score grows like $\sqrt{d_k}$.** For a typical head dimension $d_k = 64$, scores have a standard deviation of $8$. For $d_k = 128$, about $11$. These are large numbers to feed into a softmax.

Why is that bad? Softmax is *extremely* sensitive to the spread of its inputs. If one logit is, say, 24 (three standard deviations out at $d_k=64$) and the rest sit near 0, then $e^{24} \approx 2.6\times10^{10}$ dwarfs every other term: the softmax output is essentially a one-hot vector. The distribution has **saturated** — it has effectively become a hard $\arg\max$. And the gradient of a saturated softmax is nearly zero: the Jacobian of softmax is $\operatorname{diag}(a) - a a^\top$, and when $a$ is one-hot, that matrix is all zeros. **No gradient flows back to the queries and keys.** Training stalls.

Dividing by $\sqrt{d_k}$ exactly cancels the growth:

$$
\operatorname{Var}\!\left(\frac{s}{\sqrt{d_k}}\right) = \frac{\operatorname{Var}(s)}{d_k} = \frac{d_k}{d_k} = 1.
$$

The scaled scores have unit variance regardless of head dimension. Softmax receives inputs of a sane, dimension-independent spread, starts life close to uniform (gradients flow to all positions), and *learns* to sharpen as training progresses. The scale is a **variance-preserving normalization** that decouples the temperature of the attention distribution from an architectural hyperparameter ($d_k$) it has no business being coupled to. This is the same family of reasoning that motivates careful initialization and normalization throughout deep nets; see [Neural Networks From Scratch: MLPs & Backprop](../01-foundations/06-neural-nets-from-scratch.html).

{{fig:scale-sqrt-dk-saturation}}

!!! example "Worked example: scores with and without the scale"
    Let $d_k = 64$ and draw $q, k \sim \mathcal{N}(0, I_{64})$. A representative raw dot product $q \cdot k$ lands around $\pm 8$ in magnitude (std $= \sqrt{64} = 8$); occasional scores reach $\pm 20$ or more.

    Suppose for one query the four raw scores against four keys are $[20, 4, -3, 1]$.

    **Without scaling**, softmax of $[20, 4, -3, 1]$:
    $e^{20} \approx 4.85\times10^8$ swamps $e^4 \approx 54.6$, $e^{-3}\approx 0.05$, $e^{1}\approx 2.72$. The weights are about $[0.99999988,\ 0.00000011,\ \approx 0,\ \approx 0]$ — a one-hot vector. Effectively a hard pick; gradient to the other keys is essentially zero.

    **With scaling** by $\sqrt{64} = 8$, the scores become $[2.5,\ 0.5,\ -0.375,\ 0.125]$. Now softmax gives roughly $[0.77,\ 0.10,\ 0.04,\ 0.07]$ — still peaked on the first key (good, it *was* the best match) but soft enough that gradient flows to all four. The model can still learn to adjust which key wins. Same data, wildly different trainability — entirely because of one $\div\sqrt{d_k}$.

## A From-Scratch Implementation in NumPy

Let us write the whole thing with nothing but NumPy so there is no framework to hide behind. We will build a numerically stable softmax first (this matters — naïve softmax overflows), then attention on top of it, then verify on a tiny hand-checkable example.

```python
import numpy as np

def softmax(x, axis=-1):
    """Numerically stable softmax along `axis`.

    The math identity softmax(x) = softmax(x - c) for any constant c lets us
    subtract the per-row max BEFORE exponentiating. This guarantees the largest
    exponent is exp(0) = 1, so we never overflow to +inf. Without this, a logit
    of, say, 1000 would make exp(1000) = inf and the whole row becomes NaN.
    """
    x_max = np.max(x, axis=axis, keepdims=True)      # largest logit per row
    e = np.exp(x - x_max)                             # in (0, 1], no overflow
    return e / np.sum(e, axis=axis, keepdims=True)    # normalize -> sums to 1


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Scaled dot-product attention, fully from scratch.

    Shapes (single, unbatched):
        Q : (n_q, d_k)   one query per output position
        K : (n_k, d_k)   one key per source position
        V : (n_k, d_v)   one value per source position
        mask : (n_q, n_k) or None. Entries that are True (or 1) are positions
               we are ALLOWED to attend to; False/0 positions are forbidden
               and get score -inf before softmax (so weight -> exactly 0).

    Returns:
        out     : (n_q, d_v)  the attended output, one vector per query
        weights : (n_q, n_k)  the attention distribution (rows sum to 1)
    """
    d_k = Q.shape[-1]

    # Step 1: raw similarity scores. scores[i, j] = q_i · k_j
    scores = Q @ K.T                                  # (n_q, n_k)

    # Step 2: scale so the variance of scores is ~1 regardless of d_k
    scores = scores / np.sqrt(d_k)

    # Step 3 (optional): masking. Set forbidden scores to -inf so that
    # exp(-inf) = 0 and those positions receive exactly zero attention weight.
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)

    # Step 4: softmax over the KEY axis (last axis) -> attention weights
    weights = softmax(scores, axis=-1)                # (n_q, n_k)

    # Step 5: weighted sum of values
    out = weights @ V                                 # (n_q, d_v)
    return out, weights
```

Now a tiny worked sanity check we can verify by hand. We make the keys *axis-aligned* (one-hot-ish) and the values easy-to-read tags, then issue a query that points almost exactly at the second key.

```python
np.random.seed(0)

# Three keys, deliberately pointing along distinct axes, in d_k = 4.
K = np.array([
    [1.0, 0.0, 0.0, 0.0],   # key 0
    [0.0, 1.0, 0.0, 0.0],   # key 1
    [0.0, 0.0, 1.0, 0.0],   # key 2
])
# Values are easy to recognize: value j is the number (j+1) repeated.
V = np.array([
    [10.0, 10.0],           # value 0
    [20.0, 20.0],           # value 1
    [30.0, 30.0],           # value 2
])
# One query that aligns strongly with key 1 (the second axis).
Q = np.array([[0.1, 3.0, 0.1, 0.0]])

out, w = scaled_dot_product_attention(Q, K, V)
print("attention weights:", np.round(w, 3))   # ~ [[0.13 0.74 0.13]]
print("output:", np.round(out, 2))            # ~ [[20.1 20.1]] -> mostly value 1
print("rows sum to 1:", np.allclose(w.sum(axis=-1), 1.0))  # True
```

The query points mostly along axis 1, so it scores highest against key 1, softmax concentrates ~0.74 of its mass there, and the output is pulled toward value 1 (`[20, 20]`) — a soft retrieval of "the value whose key best matches my query." Crank the query's second component up to, say, 30 and the weight on key 1 approaches 1.0 and the output approaches exactly `[20, 20]`: hard retrieval recovered in the limit. That single experiment *is* the intuition for everything attention does.

## Causal & Padding Masks: Same Machinery, Different Information Flow

The `mask` argument we slipped into the implementation is what makes attention flexible enough to serve as both a bidirectional encoder and an autoregressive decoder. A mask is just an additive bias on the score matrix — set forbidden entries to $-\infty$ before softmax so they receive exactly zero weight afterward. Two masks matter in practice.

### The causal (autoregressive) mask

A language model predicts token $t+1$ from tokens $1, \dots, t$. During training we feed the whole sequence at once for efficiency, but we must forbid each position from "peeking" at future tokens — otherwise the model could trivially copy the answer and learn nothing. We enforce this by allowing query $i$ to attend only to keys $j \le i$: a **lower-triangular** mask.

$$
\operatorname{mask}_{ij} = \begin{cases} 0 & j \le i \ \ (\text{allowed, keep score}) \\ -\infty & j > i \ \ (\text{forbidden, zero out}) \end{cases}
$$

{{fig:attn-causal-mask-grid}}

This single triangular mask is the entire reason a decoder-only Transformer can be trained with one forward pass yet still behave autoregressively at inference. It also makes the **KV cache** possible: because token $i$'s output never depends on future keys/values, those keys and values are fixed once computed and can be cached and reused at each decode step — the foundation of fast generation, covered in [The Anatomy of LLM Inference: Prefill, Decode & The KV Cache](../07-inference-serving/01-anatomy-inference.html). For the variants and trade-offs of where causal masking is or isn't applied (prefix-LM, encoder–decoder), see [Architecture Variants: Encoder-Decoder, Decoder-Only & Prefix-LM](../02-transformer/08-architecture-variants.html).

### The padding mask

To batch sequences of different lengths together, we pad short sequences with a special `<pad>` token up to a common length. Those pad positions are meaningless and must never contribute to or receive attention. A **padding mask** zeroes out every score whose *key* is a pad token, for every query. In a real model you combine the causal and padding masks by taking the logical AND (a position is attendable only if it is both non-future and non-pad).

!!! warning "Common pitfall: use a large negative number, not Python's −inf, in low precision"
    Setting masked scores to exactly $-\infty$ is mathematically clean, but in float16/bfloat16 a row that is *entirely* masked (which can happen with padding bugs) makes softmax compute `exp(-inf - (-inf)) = exp(nan) = nan`, poisoning the whole batch. Production kernels use a large finite negative number (e.g. the most negative representable value of the dtype, or `-1e9` in fp32) instead of true `-inf`, and they guard against fully-masked rows. We discuss the numerics further in [Numerical Computing, Floating Point & Precision](../01-foundations/04-numerics-precision.html).

### Causal attention in NumPy

```python
def causal_mask(n):
    """Lower-triangular boolean mask: entry (i, j) is True iff j <= i.
    True = allowed to attend. Shape (n, n)."""
    return np.tril(np.ones((n, n), dtype=bool))       # tril keeps j <= i

# Demonstrate that future tokens get exactly zero weight.
n, d_k, d_v = 4, 8, 8
np.random.seed(1)
Q = np.random.randn(n, d_k)
K = np.random.randn(n, d_k)
V = np.random.randn(n, d_v)

out, w = scaled_dot_product_attention(Q, K, V, mask=causal_mask(n))
# Upper triangle of the weight matrix (strictly future) must be all zeros:
print(np.round(w, 3))
print("no future leakage:", np.allclose(np.triu(w, k=1), 0.0))  # True
# Row 0 attends only to position 0, so its weight there must be exactly 1:
print("row 0 weight:", np.round(w[0], 3))   # [1. 0. 0. 0.]
```

## The Same Thing in PyTorch, With Autograd and Batching

The NumPy version teaches the mechanism; the PyTorch version is what you would actually train. The two differences that matter in real code are (1) a **batch and head dimension** so we process `(batch, heads, seq, d_k)` tensors at once, and (2) **autograd**, so gradients flow back through the whole computation automatically. The arithmetic is byte-for-byte the same.

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0,
                                 is_causal=False):
    """From-scratch SDPA matching torch.nn.functional.scaled_dot_product_attention.

    Tensor shapes (the canonical Transformer layout):
        q : (B, H, Lq, d_k)   B=batch, H=heads, Lq=query length
        k : (B, H, Lk, d_k)
        v : (B, H, Lk, d_v)
        attn_mask : broadcastable to (B, H, Lq, Lk). Either a boolean mask
                    (True = keep) or an additive float mask (-inf = forbid).
    Returns:
        out : (B, H, Lq, d_v)
    """
    d_k = q.size(-1)

    # (B,H,Lq,d_k) @ (B,H,d_k,Lk) -> (B,H,Lq,Lk). transpose(-2,-1) makes Kᵀ.
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)

    # Build / apply the causal mask if requested.
    if is_causal:
        Lq, Lk = q.size(-2), k.size(-2)
        # Lower-triangular: position i may see j <= i. Forbidden -> -inf.
        causal = torch.ones(Lq, Lk, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~causal, float("-inf"))

    # Apply an explicit mask (padding, or arbitrary attention bias).
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask          # additive bias (e.g. ALiBi)

    # Softmax over the KEY axis (last dim). Cast to fp32 for a stable softmax
    # even when q/k/v are bf16 — a standard trick in production kernels.
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)

    if dropout_p > 0.0:
        weights = F.dropout(weights, p=dropout_p)

    return torch.matmul(weights, v)              # (B,H,Lq,d_v)
```

### A complete, trainable self-attention module

Now we wrap the kernel in a module that *creates* the queries, keys, and values from a single input via learned projections — i.e. genuine self-attention. This is single-head; [Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html) generalizes it to many heads (and to the memory-saving GQA/MLA variants that modern models use).

```python
import torch.nn as nn

class SelfAttention(nn.Module):
    """Single-head self-attention: x -> (Q,K,V) -> attention -> output projection."""

    def __init__(self, d_model, d_k=None, causal=False, dropout=0.0):
        super().__init__()
        d_k = d_k or d_model
        self.causal = causal
        self.dropout_p = dropout
        # One learned linear map per role. No bias is the modern default.
        self.W_q = nn.Linear(d_model, d_k, bias=False)
        self.W_k = nn.Linear(d_model, d_k, bias=False)
        self.W_v = nn.Linear(d_model, d_k, bias=False)
        self.W_o = nn.Linear(d_k, d_model, bias=False)   # mix attended info back

    def forward(self, x, attn_mask=None):
        # x : (B, L, d_model). Project into the three roles.
        q = self.W_q(x)                          # (B, L, d_k): "what I seek"
        k = self.W_k(x)                          # (B, L, d_k): "how to address me"
        v = self.W_v(x)                          # (B, L, d_k): "what I return"

        # Add a singleton head dim so we can reuse the (B,H,L,d) kernel with H=1.
        q, k, v = (t.unsqueeze(1) for t in (q, k, v))    # (B, 1, L, d_k)

        attended = scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=self.causal,
        )                                         # (B, 1, L, d_k)

        attended = attended.squeeze(1)            # (B, L, d_k)
        return self.W_o(attended)                 # (B, L, d_model)


# Smoke test: shapes, gradient flow, and causality.
torch.manual_seed(0)
B, L, d_model = 2, 5, 16
x = torch.randn(B, L, d_model, requires_grad=True)
attn = SelfAttention(d_model, causal=True)

y = attn(x)
print("output shape:", y.shape)                  # (2, 5, 16)

y.sum().backward()                               # autograd through the whole op
print("grad flows to input:", x.grad is not None and x.grad.abs().sum() > 0)
```

Two things are worth emphasizing about this module. First, the **output projection $W_O$** is not decoration: after the attention step each position holds a blend of value vectors, and $W_O$ is what lets the model recombine and reshape that blend before it re-enters the residual stream. (With multiple heads, $W_O$ also mixes information *across* heads.) Second, notice that we never wrote a backward pass. Because every operation — `matmul`, `softmax`, `masked_fill`, the linear layers — is a differentiable primitive with a registered gradient, PyTorch's autograd assembles the full backward pass for us; see [Automatic Differentiation & PyTorch Internals](../01-foundations/07-autodiff-pytorch.html).

### Validating against PyTorch's built-in kernel

A good from-scratch implementation should agree with the library to floating-point tolerance. PyTorch ships a fused, memory-efficient `scaled_dot_product_attention` (it dispatches to a FlashAttention-style kernel under the hood). Let us check our hand-rolled version matches it.

```python
B, H, L, d = 2, 4, 7, 32
q = torch.randn(B, H, L, d)
k = torch.randn(B, H, L, d)
v = torch.randn(B, H, L, d)

ours = scaled_dot_product_attention(q, k, v, is_causal=True)
ref  = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # PyTorch builtin

print("max abs diff:", (ours - ref).abs().max().item())   # ~1e-6, fp32 noise
assert torch.allclose(ours, ref, atol=1e-5)
```

The built-in does the *exact same math* — it just never materializes the full $L \times L$ score matrix in slow high-bandwidth memory, computing softmax incrementally instead. That IO-aware reformulation is the heart of FlashAttention; we devote a whole chapter to it in [FlashAttention I: IO-Awareness & The Online Softmax](../04-kernels-efficiency/02-flash-attention-1.html). For everyday work, prefer the fused builtin; reach for the from-scratch version to understand, debug, or modify the mechanism.

## Cost, Complexity & What the Gradients Look Like

Two practical facts about attention dominate every systems decision downstream: its quadratic cost and the shape of its gradient.

### The quadratic bottleneck

For a sequence of length $n$, head dimension $d$, the score matrix $QK^\top$ is $n \times n$ and costs $\mathcal{O}(n^2 d)$ multiply-adds; the $AV$ product costs another $\mathcal{O}(n^2 d)$. Both **compute and the memory to store the attention matrix scale as $\mathcal{O}(n^2)$** in the sequence length. Double the context and you quadruple the attention cost.

!!! example "Worked example: the $n^2$ memory of the score matrix"
    Take a single attention head, sequence length $n = 8192$, processing one sequence (batch 1). The score matrix $A$ has $n^2 = 8192^2 \approx 6.7 \times 10^7$ entries. In fp32 (4 bytes) that is about **268 MB — for one head, one layer, one sequence.** A model with, say, 32 heads and 32 layers that naïvely materialized every score matrix at once would need on the order of $268\text{ MB} \times 32 \times 32 \approx 274\text{ GB}$ just for attention scratch — far more than any single GPU. This is *exactly* why FlashAttention (which never stores the full matrix) and the quadratic-cost framing of long-context work matter so much; see [The Roofline Model & Performance Engineering](../04-kernels-efficiency/01-roofline-performance.html) and [Long-Context Pretraining & Context Extension](../03-pretraining/13-long-context-pretraining.html). It also motivates the entirely different architectures in [Beyond Attention: SSMs, Mamba, RWKV & Linear Attention](../02-transformer/11-ssm-and-alternatives.html), which trade the $n^2$ for linear cost.

### The softmax Jacobian and why values get clean gradients

Where do gradients go when we backprop through attention? The output is $o_i = \sum_j a_{ij} v_j$. By the product rule, gradient flows along two routes: through the **values** $v_j$ (weighted by the attention $a_{ij}$) and through the **weights** $a_{ij}$ (and from there through softmax back into the scores, queries, and keys).

The value path is beautifully simple: $\partial o_i / \partial v_j = a_{ij}$. A value vector receives gradient *in proportion to how much it was attended to* — attended values get strong learning signal, ignored values get little. The query/key path runs through the softmax Jacobian. For a single row with weights $a$, the Jacobian of softmax is

$$
\frac{\partial a_m}{\partial s_\ell} = a_m(\delta_{m\ell} - a_\ell), \qquad J = \operatorname{diag}(a) - a a^\top.
$$

This is where the $\sqrt{d_k}$ scaling pays off a second time. When $a$ is near-uniform (early training, thanks to scaling) the Jacobian has healthy non-zero entries and gradients propagate to queries and keys. When $a$ is saturated (one-hot), $J \to 0$ and the query/key gradients vanish — the failure mode the scale factor exists to prevent. The scale is not just numerically tidy at the forward pass; it keeps the *backward* pass alive.

!!! tip "Practitioner tip: read the attention maps, but don't over-read them"
    Visualizing the rows of $A$ (which keys each query attends to) is a genuinely useful debugging and interpretability tool — you can literally watch induction heads form, see attention sinks land on the first token, or catch a padding-mask bug as stray weight on `<pad>`. But resist the temptation to treat attention weights as a faithful "explanation" of the model's decision: a value vector that is heavily attended to may carry little task-relevant information, and information also flows through the MLPs and residual stream. Attention weights show *where* information was mixed, not *why* the model produced a given output.

!!! interview "Interview Corner"
    **Q:** Walk me through scaled dot-product attention and justify the $\frac{1}{\sqrt{d_k}}$ factor specifically. What goes wrong if you drop it, and what goes wrong if you instead divide by $d_k$?

    **A:** Attention computes, for each query, a softmax-weighted average of value vectors, where the weights come from scaled dot-product similarities between the query and every key: $\operatorname{softmax}(QK^\top/\sqrt{d_k})\,V$. The $1/\sqrt{d_k}$ exists because, treating query and key components as independent zero-mean unit-variance variables, the raw dot product $q\cdot k$ has variance $d_k$ — so its standard deviation grows like $\sqrt{d_k}$. Feeding scores of that magnitude into softmax saturates it into a near one-hot distribution, and the softmax Jacobian $\operatorname{diag}(a)-aa^\top$ collapses toward zero, so queries and keys get essentially no gradient and the layer won't train. Dividing by $\sqrt{d_k}$ restores unit variance, keeping the distribution soft and gradients flowing, independent of head size. **If you drop the scale**, deeper/wider heads saturate at init and training stalls or is very brittle. **If you over-correct and divide by $d_k$**, you shrink the score variance to $1/d_k$ — the logits become tiny, softmax stays stuck near uniform, and the model struggles to ever sharpen its attention, learning slowly. $\sqrt{d_k}$ is the unique factor that makes the score variance dimension-independent (equal to 1). A strong follow-up: this is purely an initialization-time argument; once trained, the projections learn whatever effective temperature they need, but starting in the well-conditioned regime is what lets them get there.

## Putting It All Together

We started with a `dict` lookup and ended with a trainable, batched, masked, autograd-ready attention layer — and at no point did we wave our hands. Every component earned its place: queries/keys/values are the three roles a token plays in a soft retrieval; the dot product is the similarity; $\sqrt{d_k}$ keeps that similarity's variance dimension-independent so softmax stays soft and gradients stay alive; softmax is the differentiable relaxation of "pick the best match" into "blend by match quality"; the attention matrix is the explicit, inspectable record of who-attends-to-whom; and masking is a single additive bias that reshapes information flow into causal, padding-aware, or arbitrary patterns.

From here the path forks in three directions, all of which build directly on this chapter. **Up the stack**: replicate this kernel across many [heads](../02-transformer/04-mha-gqa-mla.html), inject [position information](../02-transformer/05-positional-encoding.html) (since bare attention is permutation-equivariant — it has no inherent notion of order), wrap it in [a Transformer block](../02-transformer/06-transformer-block.html) with norms and residuals and an MLP, and stack the blocks into a [full GPT](../02-transformer/07-build-gpt-from-scratch.html). **Down to the metal**: make this kernel fast and memory-frugal with [FlashAttention](../04-kernels-efficiency/02-flash-attention-1.html) and serve it with a [KV cache](../07-inference-serving/01-anatomy-inference.html). **Sideways**: question whether the $n^2$ cost is fundamental at all, in [SSMs, Mamba & linear attention](../02-transformer/11-ssm-and-alternatives.html). Whichever way you go, the equation at the top of this page is the thing you are scaling, accelerating, or replacing.

!!! key "Key Takeaways"
    - **Attention is differentiable soft retrieval.** A query scores against every key (dot-product similarity), softmax turns scores into a probability distribution, and the output is the weighted average of the values — a smooth, differentiable relaxation of a hard `dict` lookup.
    - The headline equation is $\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\big(\tfrac{QK^\top}{\sqrt{d_k}}\big)V$. $Q,K,V$ are learned linear projections of the input in self-attention; in cross-attention $Q$ comes from one sequence and $K,V$ from another.
    - **Scale by $\sqrt{d_k}$ because $\operatorname{Var}(q\cdot k)=d_k$.** Without it, softmax saturates as $d_k$ grows, its Jacobian collapses, and query/key gradients vanish. Dividing by $\sqrt{d_k}$ fixes the score variance at 1, independent of head size.
    - **The attention matrix $A$ is $n \times n$.** Row $i$ is query $i$'s distribution over keys; you must softmax along the *key* axis (`dim=-1`). It is inspectable but is *not* a faithful explanation of model behavior.
    - **Masking is a single additive bias.** A lower-triangular causal mask forbids attending to the future (enabling one-pass training and the KV cache); a padding mask ignores `<pad>` tokens. In low precision use a large finite negative number, not true $-\infty$.
    - **Cost is $\mathcal{O}(n^2 d)$ in compute and $\mathcal{O}(n^2)$ in memory** for the score matrix — the quadratic bottleneck that motivates FlashAttention, long-context tricks, and sub-quadratic alternatives.
    - The output is a convex combination of value vectors: it can interpolate among them but never extrapolate beyond their hull. Gradient to value $v_j$ is exactly its attention weight $a_{ij}$.
    - For real workloads use a fused kernel (`F.scaled_dot_product_attention`); use the from-scratch version to understand, debug, and modify the mechanism.

!!! sota "State of the Art & Resources (2026)"
    Scaled dot-product attention remains the dominant mechanism in frontier LLMs, but the field has moved aggressively on efficiency: IO-aware kernels (FlashAttention) have made the $n^2$ matrix tractable, GQA has cut KV-cache costs in nearly every production model, and differential attention variants are beginning to show quality gains at scale.

    **Foundational papers**

    - [Vaswani et al., *Attention Is All You Need* (2017)](https://arxiv.org/abs/1706.03762) — introduced scaled dot-product and multi-head attention; the $\sqrt{d_k}$ scaling argument originates here.
    - [Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (2022)](https://arxiv.org/abs/2205.14135) — showed that tiling attention across SRAM blocks avoids materializing the $n\times n$ matrix, enabling far longer contexts.

    **Pushing the frontier (2024–2026)**

    - [Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* (2023)](https://arxiv.org/abs/2307.08691) — restructured parallelism to reach ~70% of H100 theoretical FLOPs.
    - [Shah et al., *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision* (2024)](https://arxiv.org/abs/2407.08608) — exploits Hopper GPU warp-specialization and FP8 to reach ~75% H100 utilization (up to 740 TFLOPs/s FP16).
    - [Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (2023)](https://arxiv.org/abs/2305.13245) — grouped-query attention, now standard in Llama 2/3, Mistral, and most production decoders.
    - [Ye et al., *Differential Transformer* (2024)](https://arxiv.org/abs/2410.05258) — subtracts two softmax attention maps to cancel noise; ICLR 2025 oral, outperforms standard transformers on long-context and hallucination benchmarks.

    **Open-source & tools**

    - [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) — the reference implementation of FlashAttention (all versions); supports CUDA, ROCm, and FP8.

    **Go deeper**

    - [Jay Alammar, *The Illustrated Transformer* (2018)](https://jalammar.github.io/illustrated-transformer/) — the definitive visual walkthrough of Q/K/V and multi-head attention.
    - [Lilian Weng, *Attention? Attention!* (2018)](https://lilianweng.github.io/posts/2018-06-24-attention/) — broader survey of attention variants, from additive (Bahdanau) through self-attention.
    - [Elhage et al., *A Mathematical Framework for Transformer Circuits* (2021)](https://transformer-circuits.pub/2021/framework/index.html) — formalises attention as QK/OV circuits and introduces induction heads; foundational for mechanistic interpretability.

## Further reading

- Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin — *Attention Is All You Need* (2017). The original Transformer paper; introduces scaled dot-product and multi-head attention and the $\sqrt{d_k}$ scaling.
- Bahdanau, Cho, Bengio — *Neural Machine Translation by Jointly Learning to Align and Translate* (2014). The additive-attention precursor that first framed attention as differentiable soft alignment.
- Dao, Fu, Ermon, Rudra, Ré — *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (2022). How to compute exactly this operation without materializing the $n^2$ matrix.
- Alammar — *The Illustrated Transformer* (blog). A widely used visual walkthrough of Q/K/V and the attention computation.
- Karpathy — *nanoGPT* (code repository) and *Let's build GPT* (video). A minimal, readable from-scratch implementation of causal self-attention inside a full GPT.
- Elhage et al. (Anthropic) — *A Mathematical Framework for Transformer Circuits* (2021). Reads attention as soft retrieval/information-movement and introduces QK and OV circuits and induction heads.
