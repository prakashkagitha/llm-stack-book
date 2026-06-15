# 2.4 Multi-Head Attention, MQA, GQA & MLA

In [The Attention Mechanism From Scratch](../02-transformer/03-attention-from-scratch.html) we built a single attention head — one query/key/value projection feeding one scaled dot-product softmax. It works, but a single head is a surprisingly blunt instrument: every position is forced to mix information through *one* similarity function, *one* notion of "what is relevant." Real language needs many notions of relevance at once. To resolve "it" in a sentence, one part of the model wants to look at the nearest preceding noun; another wants to track the subject of the clause; another wants to watch the quotation marks; another wants positional, short-range smoothing. **Multi-head attention** (MHA) is how the Transformer runs many such retrievals in parallel and then fuses them.

This chapter does two things. First, it builds MHA properly — the reshape gymnastics, the parameter accounting, what "heads" buy you and what they cost. Second, and this is where most of the chapter's weight lies, it confronts the single biggest operational problem MHA creates at inference time: **the KV cache.** During autoregressive decoding, the keys and values of every past token must be kept in GPU memory, and for MHA that cache grows linearly with the number of heads, layers, and sequence length until it — not the model weights — becomes the thing that limits how many users you can serve and how long a context you can afford. The modern lineage of attention variants — **Multi-Query Attention (MQA)**, **Grouped-Query Attention (GQA)**, and DeepSeek's **Multi-head Latent Attention (MLA)** — are all, at their core, answers to one question: *how do we keep the expressive power of many heads while shrinking the KV cache?* By the end you will be able to derive the cache size of any of these schemes from first principles, implement the conversion from MHA to GQA in PyTorch, and argue crisply in an interview why GQA became the default.

We assume the single-head machinery and the $\sqrt{d_k}$ scaling from the previous chapter, and the GPU memory-hierarchy intuition from [GPU Architecture & The Memory Hierarchy](../01-foundations/08-gpu-architecture.html). The serving-side consequences are developed fully in [The Anatomy of LLM Inference: Prefill, Decode & The KV Cache](../07-inference-serving/01-anatomy-inference.html) and [PagedAttention & KV-Cache Memory Management](../04-kernels-efficiency/06-paged-attention-kv.html); here we develop the *architecture* that those systems serve.

## Why More Than One Head? The Motivation for MHA

Start from a limitation. A single attention head produces, for each query position, exactly one probability distribution over the keys and returns one blended value. That distribution is a *bottleneck*: whatever the head decides to attend to, it attends to with one shared softmax. If the model needs to simultaneously (a) copy the syntactic subject and (b) track a coreferent pronoun, a single head must somehow average those two retrieval patterns into one — and averaging two sharp distributions usually gives a blurry, less useful one. Worse, the *representation subspace* is shared: the keys and queries live in one $d_\text{model}$-dimensional space, so all "kinds" of similarity compete for the same dimensions.

Multi-head attention removes both bottlenecks with a simple idea: **split the model dimension into $h$ smaller subspaces and run an independent attention head in each, then concatenate.** Concretely, with model dimension $d_\text{model}$ and $h$ heads, each head operates on a head dimension $d_h = d_\text{model}/h$. Head $i$ gets its own learned projections $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)} \in \mathbb{R}^{d_\text{model}\times d_h}$, computes ordinary scaled dot-product attention in its little $d_h$-dimensional world, and the $h$ outputs are concatenated back to width $d_\text{model}$ and passed through a final output projection $W_O$.

$$
\operatorname{head}_i = \operatorname{Attention}(XW_Q^{(i)},\, XW_K^{(i)},\, XW_V^{(i)})
$$

$$
\operatorname{MHA}(X) = \operatorname{Concat}(\operatorname{head}_1, \dots, \operatorname{head}_h)\, W_O, \qquad W_O \in \mathbb{R}^{d_\text{model}\times d_\text{model}}.
$$

Three properties make this work and are worth internalizing:

- **Parallel, independent retrieval.** Each head has its own QKV subspace, so it can specialize in one *kind* of relationship — local syntax, long-range coreference, positional copying — without interference from the others. Interpretability work (Elhage et al., *A Mathematical Framework for Transformer Circuits*) repeatedly finds heads with crisp, human-legible jobs: "previous-token heads," "induction heads" that complete `[A][B] ... [A] -> [B]`, "duplicate-token heads," and so on.
- **Same total FLOPs as one big head, roughly.** Because $d_h = d_\text{model}/h$, the per-head cost shrinks exactly as the head count grows. Splitting $d_\text{model}=512$ into 8 heads of 64 costs essentially the same arithmetic as one 512-wide head — you get the expressivity of multiple distributions almost for free in compute.
- **The output projection $W_O$ mixes across heads.** Concatenation alone would keep heads in disjoint slots; $W_O$ is what lets the model recombine information *across* heads before it re-enters the residual stream. It is not optional decoration — it is the cross-head communication channel.

{{fig:mha-forward-dataflow}}

### A subtle but important point: heads are an *implementation reshape*

It is tempting to think of $h$ heads as $h$ literally separate `nn.Linear` layers. In practice we use **one** big projection of size $d_\text{model}\to d_\text{model}$ for $Q$ (and likewise for $K$, $V$), then *reshape* the output's last axis from $d_\text{model}$ into $(h, d_h)$. The two are mathematically identical — a block-structured big matrix is the same as $h$ small matrices stacked — but the single-matmul form is far friendlier to the GPU (one large GEMM instead of $h$ tiny ones). This reshape, `(B, L, d_model) -> (B, L, h, d_h) -> (B, h, L, d_h)`, is the source of nearly every shape bug in attention code, so we will be pedantic about it below.

!!! note "Aside: do heads really specialize, and how many do you need?"
    Empirically, many heads in a trained model can be pruned with little quality loss (Michel et al., *Are Sixteen Heads Really Better than One?*), which tells you the *number* of heads is over-provisioned for expressivity — most of the work is done by a minority of specialized heads. This observation is exactly what makes MQA and GQA plausible: if you don't need $h$ independent *value* lookups, maybe you don't need $h$ independent *key/value* projections either. Hold that thought; it is the seed of the whole chapter.

## Implementing Multi-Head Attention From Scratch

Here is a complete, batched, causal-capable MHA module. It uses one fused QKV projection and the standard reshape, and calls into PyTorch's fused `scaled_dot_product_attention` (the same math we hand-rolled in 2.3, just IO-aware under the hood). Read the comments on the reshapes carefully — they are the part that bites.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention (MHA).

    d_model : model/embedding width
    n_heads : number of attention heads; head_dim = d_model // n_heads
    All of Q, K, V have n_heads heads — this is the "vanilla" MHA that every
    KV-cache-reduction variant later in this chapter tries to slim down.
    """

    def __init__(self, d_model, n_heads, causal=True, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal
        self.dropout = dropout

        # One fused projection per role. Output width d_model = n_heads * head_dim.
        # Using three separate Linears (rather than one packed QKV) for clarity;
        # production code often fuses them into a single (d_model -> 3*d_model) GEMM.
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)  # cross-head mixing

    def _split_heads(self, x):
        # x: (B, L, d_model) -> (B, n_heads, L, head_dim)
        B, L, _ = x.shape
        # Split the last axis into (n_heads, head_dim), then move heads up front
        # so each head is a contiguous (L, head_dim) matrix the kernel can batch.
        x = x.view(B, L, self.n_heads, self.head_dim)   # (B, L, h, d_h)
        return x.transpose(1, 2)                          # (B, h, L, d_h)

    def _merge_heads(self, x):
        # x: (B, n_heads, L, head_dim) -> (B, L, d_model)
        B, H, L, Dh = x.shape
        x = x.transpose(1, 2).contiguous()                # (B, L, h, d_h)
        return x.view(B, L, H * Dh)                        # (B, L, d_model)

    def forward(self, x, attn_mask=None):
        # x: (B, L, d_model)
        q = self._split_heads(self.W_q(x))                # (B, h, L, d_h)
        k = self._split_heads(self.W_k(x))                # (B, h, L, d_h)
        v = self._split_heads(self.W_v(x))                # (B, h, L, d_h)

        # Fused SDPA: softmax(QKᵀ/√d_h + mask) V, computed without materializing
        # the L×L score matrix. is_causal builds the triangular mask internally.
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.causal and attn_mask is None,
        )                                                  # (B, h, L, d_h)

        out = self._merge_heads(out)                       # (B, L, d_model)
        return self.W_o(out)                               # (B, L, d_model)


# Smoke test: shapes, causality, gradient flow.
torch.manual_seed(0)
B, L, d_model, n_heads = 2, 6, 64, 8
x = torch.randn(B, L, d_model, requires_grad=True)
mha = MultiHeadAttention(d_model, n_heads, causal=True)
y = mha(x)
print("output:", y.shape)                                  # (2, 6, 64)
y.sum().backward()
print("grad flows:", x.grad.abs().sum().item() > 0)        # True
```

### Parameter count of MHA

Every linear is $d_\text{model}\times d_\text{model}$ (ignoring biases), and there are four of them — $W_Q, W_K, W_V, W_O$:

$$
P_\text{MHA} = 4\, d_\text{model}^2.
$$

For $d_\text{model}=4096$ that is $4\times 4096^2 \approx 67$ million parameters *per attention layer*. Note this is independent of $h$: rebalancing the head count does not change the parameter budget, only how the same dimensions are partitioned. This is the first hint that the heads themselves are nearly free; the expense — at inference — lives elsewhere, in the *cache*.

!!! warning "Common pitfall: forgetting `.contiguous()` after transpose"
    `transpose(1, 2)` returns a *view* with permuted strides, not a freshly laid-out tensor. Calling `.view(...)` on it to merge heads will raise a "view size is not compatible with input tensor's size and stride" error, or worse, silently work on some shapes and not others. Always `.contiguous()` before the merging `.view()`. The fused kernels (and `reshape`, which copies when needed) hide this, but hand-written attention code trips on it constantly.

## The KV Cache: Why Inference Memory, Not FLOPs, Becomes the Bottleneck

Autoregressive generation produces one token at a time. To generate token $t+1$, the model runs attention where the query is the single new token but the keys and values span *all* $t$ previous tokens. Recomputing $K$ and $V$ for the entire prefix at every step would be hopelessly wasteful — it would make decoding $\mathcal{O}(t^2)$ work per token. The fix is the **KV cache**: after we compute each token's key and value vectors, we *store* them, and at the next step we only compute $K$ and $V$ for the one new token and append it. The query is fresh each step; the keys and values accumulate.

{{fig:mha-kvcache-append}}

This makes per-step decode cost $\mathcal{O}(t)$ instead of $\mathcal{O}(t^2)$ — a huge win. But it moves the problem from compute to **memory**, and that memory is large. For standard MHA the cache must hold, for every **layer** $L_\text{layers}$, every **head** $h$, every **token** in the context $S$, a key vector and a value vector of dimension $d_h$, for every sequence in the **batch** $B$. Each number takes $P$ bytes (2 for fp16/bf16). The total:

$$
\text{KV bytes} = 2 \times B \times L_\text{layers} \times S \times h \times d_h \times P = 2 \, B \, L_\text{layers} \, S \, d_\text{model}\, P,
$$

using $h\, d_h = d_\text{model}$. The leading **2** is for *both* K and V. Notice the cache scales linearly in everything you care about at serving time — batch size (concurrency), context length, and depth — and crucially it is proportional to $d_\text{model}$, i.e. to the *full* width across all heads.

Why does this dominate? Because during decode, generating one token touches every weight once (compute-light, memory-bandwidth-bound) but must *read the entire KV cache* to do attention. The cache is read from high-bandwidth memory (HBM) on every single decode step. So the KV cache hurts you twice: it occupies precious HBM that could otherwise hold more concurrent requests, and it must be streamed through the memory system every step, making decode **memory-bandwidth bound** (see the roofline analysis in [The Roofline Model & Performance Engineering](../04-kernels-efficiency/01-roofline-performance.html)). Shrinking the cache simultaneously raises the batch size you can fit *and* speeds up each decode step. That is the prize the whole MQA/GQA/MLA family is chasing.

!!! example "Worked example: KV-cache size for a 70B-class model"
    Take a Llama-2-70B-style configuration: $L_\text{layers}=80$ layers, $d_\text{model}=8192$, $h=64$ heads of $d_h=128$, stored in bf16 ($P=2$ bytes). Standard MHA. For a **single** sequence ($B=1$) at context length $S=4096$:

    $$
    2 \times 1 \times 80 \times 4096 \times 8192 \times 2 \;\text{bytes} \approx 1.07 \times 10^{10}\ \text{bytes} \approx 10.7\ \text{GB}.
    $$

    Ten gigabytes of cache for *one* user at a modest 4K context. The model weights themselves are ~140 GB in bf16, so on an 8×80 GB node you have roughly 500 GB free after weights — meaning naïve MHA caps you at on the order of ~45 concurrent 4K sequences before the cache alone exhausts memory. Push the context to 32K and a *single* sequence wants ~86 GB of cache — more than an entire 80 GB GPU. This is the wall. Now suppose we replace MHA with GQA using 8 KV groups instead of 64 (the actual Llama-2-70B choice): the cache shrinks by exactly $64/8 = 8\times$, to about **1.34 GB** at 4K — and suddenly you can serve ~8× more users or 8× longer contexts from the same hardware. *That single architectural decision is worth more than most kernel optimizations.*

### Decomposing the cache: which factors can architecture actually change?

Look again at $2\, B\, L_\text{layers}\, S\, h\, d_h\, P$. As a serving operator you can pick $B$ and $S$ (workload), and you can quantize the cache to shrink $P$ (covered in [Quantization II: INT4/INT8/FP8, GGUF, bitsandbytes & QAT](../04-kernels-efficiency/08-quantization-formats-qat.html)). As an *architect* the only knobs are $L_\text{layers}$, $h$, and $d_h$ — and you cannot freely cut layers or head dimension without hurting quality. The clever insight of MQA/GQA is that the $h$ in the cache formula need not equal the $h$ used for *queries*. **You can have many query heads but few key/value heads.** That decoupling is the entire game.

## MQA and GQA: Sharing Keys and Values Across Heads

{{fig:gqa-kv-sharing}}

### Multi-Query Attention (MQA)

Multi-Query Attention (Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*, 2019) takes the decoupling to its extreme: keep all $h$ **query** heads, but use a **single** shared key head and a **single** shared value head. Every query head attends against the *same* $K$ and the *same* $V$.

$$
\operatorname{head}_i = \operatorname{Attention}\big(X W_Q^{(i)},\, X W_K,\, X W_V\big), \qquad i = 1,\dots,h,
$$

where now there is just one $W_K, W_V \in \mathbb{R}^{d_\text{model}\times d_h}$ shared across all $i$. The KV cache shrinks by a factor of $h$ — instead of caching $h$ key vectors and $h$ value vectors per token, you cache exactly one of each:

$$
\text{KV bytes (MQA)} = 2\, B\, L_\text{layers}\, S\, d_h\, P \quad\Longrightarrow\quad \frac{1}{h}\ \text{the size of MHA}.
$$

For the 70B example, MQA would cut the 10.7 GB cache to roughly $10.7/64 \approx 0.17$ GB. The decode step also reads $h\times$ less KV from HBM, so it is much faster. The cost: quality. Forcing all query heads to share one key/value subspace removes most of the representational diversity on the K/V side. In practice MQA can cause a measurable quality regression and, notably, **training instability** for large models — the shared KV head becomes a fragile bottleneck. MQA was used in some production models (e.g. PaLM, Falcon) but the regression motivated a middle ground.

### Grouped-Query Attention (GQA)

Grouped-Query Attention (Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, 2023) interpolates between MHA and MQA with a single integer knob $g$ = the number of **KV groups** (also written $n_\text{kv}$, the number of key/value heads). The $h$ query heads are partitioned into $g$ groups; all query heads in a group share one key head and one value head.

- $g = h$  ⟶ every query head has its own KV head ⟶ **this is exactly MHA**.
- $g = 1$  ⟶ all query heads share one KV head ⟶ **this is exactly MQA**.
- $1 < g < h$ ⟶ **GQA**, the spectrum in between.

The KV cache scales with $g$, not $h$:

$$
\boxed{\ \text{KV bytes (GQA)} = 2\, B\, L_\text{layers}\, S\, g\, d_h\, P \ }
$$

so choosing $g = h/8$ gives an $8\times$ cache reduction. The beauty of GQA is empirical: a small number of groups (commonly $g = 8$) recovers nearly all of MHA's quality while keeping most of MQA's memory savings. Llama-2-70B and Llama-3 use $g=8$; Mistral-7B uses $g=8$ with 32 query heads. The pattern — 8 KV heads regardless of how many query heads — is now the de facto standard, partly because $g=8$ aligns naturally with 8-way tensor parallelism (each GPU owns one KV head; see [Distributed Training II: Tensor, Pipeline, Sequence & Expert Parallelism](../03-pretraining/06-distributed-model-parallel.html)).

{{fig:mhagqamla-headsharing-spectrum}}

### Implementing the MHA → GQA conversion

The mechanical heart of GQA is *repeating* each KV head to match its group of query heads before the dot product, so the kernel still sees aligned $(h, L, d_h)$ tensors. Here is GQA from scratch, written so that `n_kv_heads == n_heads` recovers MHA and `n_kv_heads == 1` recovers MQA — one module covering the whole spectrum.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def repeat_kv(x, n_rep):
    """Expand KV heads to match query heads (the GQA broadcast).

    x: (B, n_kv, L, d_h). Repeat each KV head n_rep times along the head axis,
    producing (B, n_kv * n_rep, L, d_h) = (B, n_heads, L, d_h).

    This is a memory-cheap expand (no data copy until the kernel reads it):
    head order becomes [kv0, kv0, ..., kv1, kv1, ...], so query head j uses
    KV group j // n_rep. Mirrors Llama's reference implementation.
    """
    B, n_kv, L, d_h = x.shape
    if n_rep == 1:
        return x
    # Insert a length-1 axis after the head axis, expand it, then flatten.
    x = x[:, :, None, :, :]                       # (B, n_kv, 1,     L, d_h)
    x = x.expand(B, n_kv, n_rep, L, d_h)          # (B, n_kv, n_rep, L, d_h)
    return x.reshape(B, n_kv * n_rep, L, d_h)     # (B, n_heads, L, d_h)


class GroupedQueryAttention(nn.Module):
    """GQA that unifies MHA / GQA / MQA via a single n_kv_heads knob.

        n_kv_heads == n_heads  ->  MHA  (one KV head per query head)
        1 < n_kv_heads < n_heads -> GQA (KV heads shared within groups)
        n_kv_heads == 1        ->  MQA  (one KV head for all query heads)
    """

    def __init__(self, d_model, n_heads, n_kv_heads=None, causal=True, dropout=0.0):
        super().__init__()
        n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        assert n_heads % n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"
        assert d_model % n_heads == 0

        self.n_heads = n_heads            # number of QUERY heads
        self.n_kv_heads = n_kv_heads      # number of KEY/VALUE heads (= groups g)
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads  # query heads per KV head
        self.causal = causal
        self.dropout = dropout

        # KEY INSIGHT: Q projects to the FULL d_model (n_heads * head_dim),
        # but K and V project only to n_kv_heads * head_dim — a SMALLER matrix.
        # That smaller K/V projection is exactly what shrinks the cache.
        self.W_q = nn.Linear(d_model, n_heads    * self.head_dim, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.W_o = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, x, attn_mask=None):
        B, L, _ = x.shape

        # Project. Q has n_heads heads; K, V have only n_kv_heads heads.
        q = self.W_q(x).view(B, L, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # q: (B, n_heads, L, d_h)   k,v: (B, n_kv_heads, L, d_h)

        # Broadcast each KV head across its group so shapes align with q.
        # (At inference, you cache the UNREPEATED k, v — size ∝ n_kv_heads —
        #  and repeat on the fly; that is where the memory saving lives.)
        k = repeat_kv(k, self.n_rep)             # (B, n_heads, L, d_h)
        v = repeat_kv(v, self.n_rep)             # (B, n_heads, L, d_h)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.causal and attn_mask is None,
        )                                         # (B, n_heads, L, d_h)

        out = out.transpose(1, 2).contiguous().view(B, L, self.n_heads * self.head_dim)
        return self.W_o(out)


# Verify the spectrum: MHA, GQA, and MQA all run and produce correct shapes.
torch.manual_seed(0)
B, L, d_model, n_heads = 2, 7, 64, 8
x = torch.randn(B, L, d_model)

for n_kv in (8, 2, 1):                            # MHA, GQA(g=2), MQA
    attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads=n_kv)
    y = attn(x)
    # Cache width per token (one layer, one sequence), bf16 = 2 bytes, ×2 for K&V:
    kv_bytes = 2 * n_kv * (d_model // n_heads) * 2
    print(f"n_kv={n_kv}: out {tuple(y.shape)}, KV bytes/token/layer = {kv_bytes}")
# n_kv=8: 256 B   n_kv=2: 64 B   n_kv=1: 32 B  -> 8× and 16× smaller than MHA
```

The two load-bearing facts in that code: (1) `W_k` and `W_v` output `n_kv_heads * head_dim`, which is *smaller* than `d_model` — that is where parameters and, more importantly, cache are saved; (2) `repeat_kv` is a cheap `expand` that does not copy the cached data, so at inference you store only the small KV tensors and broadcast them into the kernel. The query side is untouched, so the model keeps all $h$ query heads' worth of expressivity in *how it asks questions*, sacrificing only the diversity of *what it can address*.

!!! note "Aside: uptraining — converting an existing MHA checkpoint to GQA"
    A delightful result from the GQA paper is that you don't have to train from scratch. You can take a pretrained MHA model and *construct* a GQA model by **mean-pooling** the key/value projection weights within each group — average the $W_K$ of the heads in a group to get the group's shared $W_K$ — then "uptrain" with a small fraction (e.g. ~5%) of the original pretraining compute to recover quality. This made GQA a cheap retrofit, which is a big reason it spread so fast: model providers could ship GQA variants of existing models without a full pretrain.

## Multi-head Latent Attention (MLA): Compress the Cache Itself

GQA shrinks the cache by reducing the *number* of KV heads. DeepSeek's **Multi-head Latent Attention** (MLA), introduced with DeepSeek-V2, attacks the same target from a completely different angle: keep many KV heads' worth of expressivity, but **cache a low-rank compressed latent** of the keys and values instead of the keys and values themselves. It is, in spirit, a learned low-rank factorization of the KV cache.

### The core idea: low-rank joint compression

In MHA, for each token the keys and values across all heads together form a $2 d_\text{model}$-dimensional object that we cache. MLA introduces a small **latent dimension** $d_c \ll d_\text{model}$ and a *down-projection* $W^{DKV}$ that maps each token's hidden state into a single compressed latent vector $c^{KV}_t \in \mathbb{R}^{d_c}$. **Only this latent is cached.** At attention time, two *up-projection* matrices $W^{UK}$ and $W^{UV}$ reconstruct the per-head keys and values from the latent:

$$
c^{KV}_t = W^{DKV} h_t \in \mathbb{R}^{d_c}, \qquad k_t = W^{UK} c^{KV}_t, \qquad v_t = W^{UV} c^{KV}_t .
$$

Because $d_c$ is small (DeepSeek-V2 uses $d_c$ on the order of a few hundred, far below $d_\text{model}$), the cached object per token is $d_c$ numbers instead of $2\, h\, d_h$ numbers. The keys and values for *all* heads are regenerated from that one shared latent, so MLA retains far more head diversity than MQA/GQA at a comparable cache budget. The queries are similarly given a low-rank treatment (a query down/up pair) to save *training* activation memory, though queries are not cached.

{{fig:mhagqamla-cache-layout-compare}}

### The absorption trick and why RoPE forces a split

Two subtleties make MLA more than a textbook low-rank trick.

**(1) Weight absorption.** A naive reading says you must up-project the latent to full keys/values at every step, which would cost compute. But the up-projections are *linear and fixed*, so they can be **absorbed** into neighboring matrices: $W^{UK}$ folds into the query projection $W^{Q}$ (since attention scores are $q^\top k = q^\top W^{UK} c^{KV} = (W^{UK\top} q)^\top c^{KV}$), and $W^{UV}$ folds into the output projection $W^{O}$. After absorption you can compute attention *directly against the cached latent* without ever materializing the full per-head K/V — so MLA keeps the tiny cache *and* avoids an extra projection in the decode loop. This is the engineering reason MLA is fast, not just small.

**(2) RoPE incompatibility, and the decoupled fix.** Rotary position embeddings (RoPE; see [Positional Encodings: Sinusoidal, Learned, RoPE & ALiBi](../02-transformer/05-positional-encoding.html)) apply a *position-dependent rotation* to keys. That rotation does not commute with the absorption trick — a position-dependent matrix cannot be folded into a position-independent up-projection. DeepSeek's solution is a **decoupled RoPE**: each head's key is split into two parts, a larger *content* part carried by the compressed latent (no RoPE, absorbable) and a smaller *positional* part that does carry RoPE and is computed/cached separately (a small shared per-token RoPE key). The query is split analogously. So MLA caches *two* things per token — the big content latent $c^{KV}$ and a small decoupled RoPE key — but the sum is still dramatically smaller than GQA's cache while preserving full multi-head expressivity. The net effect reported by DeepSeek is a KV cache comparable to (or smaller than) GQA with 2.25 KV groups while matching or exceeding MHA quality.

### A minimal MLA forward pass

The following strips MLA to its essentials (content path only, RoPE omitted for clarity) so you can see the down/up structure. Production MLA adds the decoupled RoPE keys and the absorption optimization above.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadLatentAttentionCore(nn.Module):
    """Simplified MLA: cache a low-rank latent c^{KV}, reconstruct per-head K,V.

    Omits decoupled-RoPE and weight-absorption for clarity. The point to see:
    the CACHED object is c_kv of width d_c (small), not the full per-head K,V.
    """

    def __init__(self, d_model, n_heads, d_c, causal=True):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_c = d_c                                   # latent dim, d_c << d_model
        self.causal = causal

        # Down-projection: hidden -> compressed latent (this is what gets cached).
        self.W_dkv = nn.Linear(d_model, d_c, bias=False)
        # Up-projections: latent -> per-head keys / values.
        self.W_uk = nn.Linear(d_c, n_heads * self.head_dim, bias=False)
        self.W_uv = nn.Linear(d_c, n_heads * self.head_dim, bias=False)
        # Queries: a normal (optionally low-rank) projection; not cached.
        self.W_q = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.W_o = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, x):
        B, L, _ = x.shape
        H, Dh = self.n_heads, self.head_dim

        c_kv = self.W_dkv(x)                              # (B, L, d_c)  <-- CACHED
        k = self.W_uk(c_kv).view(B, L, H, Dh).transpose(1, 2)   # (B, H, L, Dh)
        v = self.W_uv(c_kv).view(B, L, H, Dh).transpose(1, 2)   # (B, H, L, Dh)
        q = self.W_q(x).view(B, L, H, Dh).transpose(1, 2)       # (B, H, L, Dh)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        out = out.transpose(1, 2).contiguous().view(B, L, H * Dh)
        return self.W_o(out)


torch.manual_seed(0)
m = MultiHeadLatentAttentionCore(d_model=512, n_heads=8, d_c=128)
y = m(torch.randn(2, 16, 512))
print("MLA out:", y.shape)                               # (2, 16, 512)
# Cache per token/layer = d_c numbers (+ a small decoupled-RoPE key in full MLA),
# vs MHA's 2 * d_model. Here 128 vs 1024 -> ~8× smaller, full head diversity kept.
```

MLA is more complex to implement and to serve (the kernel and the cache layout differ from the GQA path that vLLM/SGLang were originally built around), which is why GQA remains the broad default and MLA is found mainly in models specifically designed for it (the DeepSeek-V2/V3 line). But MLA is the clearest demonstration that the cache, not the head count, is the real object of optimization — and that you can compress it directly.

## Memory–Quality Tradeoffs: Choosing a Scheme

Put the four schemes side by side. Let $h$ be the number of query heads, $g$ the KV groups, $d_h$ the head dim, $d_c$ the MLA latent. All cache figures are *per token, per layer, per sequence*, in elements (multiply by bytes-per-element and by $B\, L_\text{layers}\, S$ for totals).

| Scheme | KV heads | Cache (elements/token/layer) | Relative cache | Quality | Used by (examples) |
|---|---|---|---|---|---|
| MHA   | $h$ | $2\,h\,d_h = 2\,d_\text{model}$ | $1\times$ (baseline) | best | original Transformer, GPT-2/3, Llama-1 |
| GQA   | $g$ | $2\,g\,d_h$ | $g/h$ | ≈ MHA | Llama-2-70B/3, Mistral, Gemma |
| MQA   | $1$ | $2\,d_h$ | $1/h$ | noticeable drop, can be unstable | PaLM, Falcon |
| MLA   | (latent) | $\approx d_c\,(+\text{small RoPE key})$ | small, tunable | ≈ MHA or better | DeepSeek-V2/V3 |

How to reason about the choice:

- **GQA is the safe default.** It gives a large, tunable cache reduction (pick $g$) with negligible quality loss at $g \approx 8$, it is trivially supported by every serving stack, and it can be retrofitted onto an MHA checkpoint by uptraining. If you are building a conventional dense or MoE model in 2024–2026, GQA is the path of least resistance and the strong baseline. ([Mixture-of-Experts (MoE) Architectures](../02-transformer/09-mixture-of-experts.html) addresses the *FFN* cost; GQA addresses the *attention cache* cost — they compose.)
- **MQA only when memory is desperate and you accept the risk.** The full $1/h$ reduction is tempting, but the quality regression and training instability mean most teams stop at GQA. MQA makes sense in tightly constrained settings (small on-device models, extreme batch) where every byte counts.
- **MLA when you control the whole stack and want the Pareto frontier.** MLA pushes the memory–quality frontier beyond GQA, but it demands custom kernels, a different cache layout, and the decoupled-RoPE machinery. It pays off most for very long contexts and at large scale, where the absolute cache savings are enormous — but it is an architectural commitment, not a drop-in.
- **Quantizing the cache is orthogonal and stacks.** All four schemes can additionally store the cache in fp8 or int8/int4, multiplying the savings (see [PagedAttention & KV-Cache Memory Management](../04-kernels-efficiency/06-paged-attention-kv.html)). GQA + fp8 cache is a very common, very effective combination.

!!! tip "Practitioner tip: pick $g$ to match your tensor-parallel degree"
    A neat systems consideration: when you shard attention across $T$ GPUs with tensor parallelism, you want each GPU to own a whole number of KV heads so it can compute its share of attention without cross-GPU KV gather. Choosing $g$ equal to (or a multiple of) your TP degree — e.g. $g=8$ for 8-way TP — makes the sharding clean and avoids replicating KV heads across GPUs. This is part of why $g=8$ is so common: it matches the 8-GPU node. If you instead picked $g=6$ on an 8-GPU node, two GPUs would sit idle on the KV side or you'd replicate, wasting the very memory you were trying to save.

!!! warning "Common pitfall: comparing cache savings without holding context length fixed"
    A frequent analysis error is to celebrate an $8\times$ cache reduction and then immediately spend it all on a longer context, concluding "GQA didn't help." The cache scales linearly in $S$, so an $8\times$ reduction lets you go to $8\times$ the context *or* $8\times$ the batch *or* some product of the two — it is a budget, not a free lunch. Always state what you are holding fixed (concurrency, context, or hardware) when quoting a savings factor, or the comparison is meaningless.

!!! interview "Interview Corner"
    **Q:** Why did Grouped-Query Attention become the default for modern LLMs over both standard Multi-Head Attention and Multi-Query Attention? Walk me through the tradeoff and give me the cache formula.

    **A:** The bottleneck GQA targets is the **KV cache at inference**, not FLOPs or parameters. During autoregressive decode you must keep every past token's keys and values in HBM; for standard MHA the cache is $2\, B\, L_\text{layers}\, S\, h\, d_h\, P$ bytes — it scales with the number of heads $h$, and it both consumes memory that limits concurrency and must be streamed every decode step, making decode memory-bandwidth bound. **MQA** cuts this by a full factor of $h$ by sharing a single key/value head across all query heads, but collapsing all KV diversity into one head causes a measurable quality drop and training instability at scale. **GQA** keeps all $h$ query heads but uses $g$ KV groups (each shared by $h/g$ query heads), so the cache becomes $2\, B\, L_\text{layers}\, S\, g\, d_h\, P$ — a tunable $g/h$ reduction. The empirical finding is that a small $g$ (typically 8) recovers essentially all of MHA's quality while giving most of MQA's memory savings, so it sits at the sweet spot of the memory–quality curve. Three more reasons it won: $g=8$ aligns with 8-way tensor parallelism (clean sharding, one KV head per GPU); you can *uptrain* an existing MHA checkpoint into GQA by mean-pooling KV weights with ~5% extra compute rather than pretraining from scratch; and it requires no special kernels. A strong closing point: GQA's savings are orthogonal to KV quantization and to MoE, so they stack — and the frontier beyond GQA is MLA, which compresses the cache into a low-rank latent rather than just reducing head count.

## Putting It All Together

The arc of this chapter is a single tension and four answers to it. Multi-head attention is the right idea: run many specialized retrievals in parallel and fuse them with $W_O$, getting the expressivity of multiple attention distributions for roughly the FLOPs of one wide head. But MHA's gift — independent keys and values per head — is exactly what makes its **KV cache** balloon at inference, and that cache, not the weights, is what caps concurrency and context length on real hardware. MQA, GQA, and MLA are three points on the resulting memory–quality frontier: MQA shares one KV head (maximal savings, real quality cost), GQA shares KV heads within groups (the pragmatic default, nearly free quality-wise at $g=8$), and MLA caches a low-rank latent that reconstructs all heads (the frontier, at the cost of implementation complexity and a decoupled-RoPE wrinkle).

The single most useful skill to walk away with is the ability to *derive a cache size in your head*: $2\, B\, L_\text{layers}\, S \times (\text{KV-head count}) \times d_h \times P$. Plug in a model and a workload and you immediately know whether you are weight-bound or cache-bound, how much context you can afford, and how much a given attention variant buys you. That number drives more serving decisions than almost anything else in the stack.

From here, the cache reappears everywhere downstream: the serving systems that *manage* it page-by-page in [PagedAttention & KV-Cache Memory Management](../04-kernels-efficiency/06-paged-attention-kv.html) and [vLLM: Architecture, PagedAttention & Internals](../07-inference-serving/03-vllm-internals.html); the kernels that *read* it efficiently in [FlashAttention I: IO-Awareness & The Online Softmax](../04-kernels-efficiency/02-flash-attention-1.html); the prefix-sharing tricks that *reuse* it in [Prefix Caching & KV-Cache Reuse](../07-inference-serving/07-prefix-caching.html); and the quantization that *shrinks* it further in [Quantization II](../04-kernels-efficiency/08-quantization-formats-qat.html). Next we give attention its missing sense of position in [Positional Encodings: Sinusoidal, Learned, RoPE & ALiBi](../02-transformer/05-positional-encoding.html) — including the RoPE that complicated MLA — before assembling the full [Transformer block](../02-transformer/06-transformer-block.html).

!!! key "Key Takeaways"
    - **Multi-head attention splits $d_\text{model}$ into $h$ subspaces** and runs independent scaled-dot-product heads in parallel, then concatenates and mixes them with $W_O$. It costs roughly the same FLOPs as one wide head but lets the model attend to many *kinds* of relationships at once; parameters are $4\,d_\text{model}^2$ per layer, independent of $h$.
    - **The KV cache is the real inference bottleneck.** Its size is $2\, B\, L_\text{layers}\, S \times (\text{KV-head count}) \times d_h \times P$ bytes — for a 70B model at 4K context, standard MHA wants ~10 GB *per sequence*, and it must be streamed from HBM every decode step (memory-bandwidth bound).
    - **The query-head count and the KV-head count can be decoupled.** Many query heads can share few key/value heads — this single observation generates MQA, GQA, and MLA.
    - **MQA** uses one shared KV head ($1/h$ cache) but loses quality and can destabilize training. **GQA** uses $g$ KV groups ($g/h$ cache); at $g\approx 8$ it recovers ≈ MHA quality and is the modern default.
    - **GQA implementation** = smaller $W_K, W_V$ projections (output $g\,d_h$, not $d_\text{model}$) plus a cheap `repeat_kv` broadcast to align KV heads with query heads; setting $n_\text{kv}=h$ recovers MHA and $n_\text{kv}=1$ recovers MQA.
    - **MLA (DeepSeek)** caches a low-rank latent $c^{KV}$ and up-projects per-head K/V from it, with weight absorption to keep decode cheap and a decoupled RoPE key to remain position-aware — pushing the memory–quality frontier beyond GQA at the cost of complexity.
    - **An MHA checkpoint can be uptrained into GQA** by mean-pooling KV weights within groups and fine-tuning with a small fraction of pretraining compute — a cheap retrofit that accelerated GQA's adoption.
    - **Cache savings are a budget, not a free lunch**, and they stack with KV quantization and TP-aligned head counts ($g=8$ for 8-way tensor parallelism). Always state what you hold fixed when quoting a reduction factor.

!!! sota "State of the Art & Resources (2026)"
    Multi-head attention variants are now a mature design space: GQA (typically g=8) is the universal default across open-weight models, MLA has pushed the Pareto frontier further for long-context serving, and FlashAttention-3/4 kernels make reading the KV cache fast enough that the cache *size* remains the binding constraint.

    **Foundational work**

    - [Vaswani et al., *Attention Is All You Need* (2017)](https://arxiv.org/abs/1706.03762) — introduces multi-head attention with the concatenate-then-project structure that all variants build on.
    - [Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need* (2019)](https://arxiv.org/abs/1911.02150) — defines multi-query attention (MQA) and frames the KV cache as the primary decoding bottleneck.
    - [Michel, Levy & Neubig, *Are Sixteen Heads Really Better than One?* (2019)](https://arxiv.org/abs/1905.10650) — shows most heads are prunable, providing the empirical motivation for head-sharing schemes.

    **Recent advances (2023–2026)**

    - [Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (2023)](https://arxiv.org/abs/2305.13245) — introduces grouped-query attention and the mean-pool uptraining recipe that made GQA the modern default.
    - [DeepSeek-AI, *DeepSeek-V2* (2024)](https://arxiv.org/abs/2405.04434) — introduces Multi-head Latent Attention (MLA) with low-rank KV compression, weight absorption, and decoupled RoPE, cutting KV cache by 93% vs MHA.
    - [Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* (2023)](https://arxiv.org/abs/2307.08691) — 2× speedup over FlashAttention-1 via improved thread-block parallelism; the kernel underlying PyTorch's `scaled_dot_product_attention`.

    **Open-source & tools**

    - [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) — reference CUDA/ROCm implementation of FlashAttention 1–4; now supports MQA/GQA attention patterns natively.
    - [vllm-project/vllm](https://github.com/vllm-project/vllm) — production LLM serving engine with PagedAttention, GQA support, and KV-cache quantization baked in.
    - [sgl-project/sglang](https://github.com/sgl-project/sglang) — high-performance serving framework with first-class MLA support (7× faster DeepSeek MLA vs earlier baselines) and RadixAttention prefix caching.

    **Go deeper**

    - [Sebastian Raschka, *Multi-Head Latent Attention (MLA)*](https://sebastianraschka.com/llm-architecture-gallery/mla/) — clear visual walkthrough of MLA's low-rank compression, weight absorption, and real-world adoption across DeepSeek-V3, Kimi K2, and GLM-5.

## Further reading

- Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin — *Attention Is All You Need* (2017). Introduces multi-head attention and the concatenate-then-project structure.
- Shazeer — *Fast Transformer Decoding: One Write-Head is All You Need* (2019). The Multi-Query Attention paper; frames the KV cache as the decoding bottleneck.
- Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai — *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (2023). Introduces Grouped-Query Attention and the mean-pool uptraining recipe.
- DeepSeek-AI — *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model* (2024) and *DeepSeek-V3 Technical Report* (2024). Introduce and refine Multi-head Latent Attention, weight absorption, and decoupled RoPE.
- Michel, Levy, Neubig — *Are Sixteen Heads Really Better than One?* (2019). Evidence that many heads are prunable, motivating the head-sharing intuition behind MQA/GQA.
- Elhage, Nanda, Olsson, et al. (Anthropic) — *A Mathematical Framework for Transformer Circuits* (2021). Reads individual heads as interpretable information-movement operations (induction heads, previous-token heads).
- Dao, Fu, Ermon, Rudra, Ré — *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (2022). The IO-aware kernel that makes reading the KV cache efficient and underlies the fused `scaled_dot_product_attention` used throughout this chapter.
