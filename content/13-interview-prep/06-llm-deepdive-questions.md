# 13.6 LLM-Specific Deep-Dive Questions

The breadth round ([ML Breadth: Rapid-Fire Concepts & Model Answers](../13-interview-prep/02-ml-breadth-rapidfire.html)) tests whether you know machine learning. The LLM deep-dive tests whether you actually understand the *modern* stack — the part that didn't exist when most ML textbooks were written. At Google, this often arrives mid-conversation: the interviewer name-drops "KV cache" or "GQA" or "DPO" and watches you either light up or stall. A strong candidate doesn't recite a definition; they explain the *mechanism*, the *failure mode it fixes*, the *cost*, and the *tradeoff*.

This chapter is a question bank with model answers, organized by topic. Each answer is written the way you should *say* it out loud: lead with the one-sentence essence, then the mechanism, then a number or a tradeoff. We cross-link to the full chapters so you can go deeper, but everything you need to *survive the question* is here. Treat the code blocks as the reconstructions you should be able to whiteboard from memory under pressure — if you can write the eight-line attention and the online-softmax recurrence, you have already beaten most candidates.

A meta-note on style: when you don't know something, say what you *do* know and reason toward the rest. "I haven't read the EAGLE paper closely, but speculative decoding generally works by..." is a strong answer. Bluffing a fabricated number is the single fastest way to lose an LLM interviewer's trust.

## Attention, RoPE, KV-Cache & FlashAttention

These four are the load-bearing wall. If you can't explain them crisply, the interviewer learns everything they need in five minutes.

### Explain self-attention from scratch

**Essence.** Attention lets every token mix in information from every other token, weighted by learned relevance. Each token emits a *query*, every token exposes a *key* and a *value*; the query dots against all keys to produce weights, and the output is the weighted sum of values.

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

The $\sqrt{d_k}$ divisor is not cosmetic: if $q,k$ have unit-variance components, $q\cdot k$ has variance $d_k$, so without scaling the logits grow with head dimension, the softmax saturates, and gradients vanish. You should be able to reconstruct this in a few lines:

```python
import torch, torch.nn.functional as F

def attention(q, k, v, causal=True):
    # q,k,v: (batch, heads, seq, d_head)
    d_head = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / d_head ** 0.5   # (b,h,T,T)
    if causal:
        T = q.shape[-2]
        mask = torch.triu(torch.ones(T, T, device=q.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))  # no peeking ahead
    weights = F.softmax(scores, dim=-1)                   # rows sum to 1
    return weights @ v                                    # (b,h,T,d_head)
```

The causal mask is what makes a decoder-only LM autoregressive: position $t$ may attend to $\le t$ only. The full development — why dot-product, why multi-head, the geometry — is in [The Attention Mechanism From Scratch](../02-transformer/03-attention-from-scratch.html).

**Cost.** The $QK^\top$ matrix is $T\times T$, so attention is $O(T^2 d)$ compute and, naively, $O(T^2)$ memory. That quadratic is the root cause of half the engineering in this book.

### Explain RoPE and why it beats learned positional embeddings

**Essence.** Rotary Position Embedding encodes position by *rotating* the query and key vectors by an angle proportional to their absolute position, in 2D subspaces. Because a dot product of two rotated vectors depends only on the *difference* of their rotation angles, the attention score between positions $m$ and $n$ becomes a function of $m-n$ — RoPE injects relative position while applying an absolute rotation.

For a 2D pair $(x_1,x_2)$ at position $m$ with frequency $\theta$:

$$
R_{m,\theta}\begin{bmatrix}x_1\\x_2\end{bmatrix}=\begin{bmatrix}\cos m\theta & -\sin m\theta\\ \sin m\theta & \cos m\theta\end{bmatrix}\begin{bmatrix}x_1\\x_2\end{bmatrix}
$$

The head dimension is split into $d/2$ such pairs, each with its own frequency $\theta_i=\text{base}^{-2i/d}$ (base typically 10000). Low-index pairs rotate fast (capture local order); high-index pairs rotate slowly (capture long-range position).

```python
def build_rope_cache(seq_len, d_head, base=10000.0, device="cpu"):
    # inverse frequencies: one per dimension-pair
    inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2, device=device).float() / d_head))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)            # (seq, d_head/2)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rope(x, cos, sin):
    # x: (b, h, seq, d_head); rotate consecutive pairs (x_even, x_odd)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, None], sin[None, None]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2], out[..., 1::2] = rx1, rx2
    return out
```

**Why it wins.** (1) It's *relative*, which generalizes better than absolute position. (2) It's applied to Q and K only — no extra parameters, no added embedding to the residual stream. (3) It extends to longer contexts via frequency tricks (NTK scaling, YaRN) without retraining from scratch — you just stretch or interpolate the angles. This is the foundation of context extension; see [Positional Encodings: Sinusoidal, Learned, RoPE & ALiBi](../02-transformer/05-positional-encoding.html) and [Long-Context Pretraining & Context Extension](../03-pretraining/13-long-context-pretraining.html).

### What is the KV cache and why does it dominate inference memory?

**Essence.** During autoregressive decoding, generating token $t+1$ requires attending over all previous keys and values. Those K and V tensors depend only on past tokens, not the current one, so we *cache* them and reuse them — turning per-step attention from $O(t^2)$ recomputation into $O(t)$.

The cache size is the number every LLM systems interviewer wants you to derive on the spot:

$$
\text{KV bytes}=2\times L\times n_{\text{kv}}\times d_{\text{head}}\times T\times B\times \text{bytes}
$$

where 2 is for K and V, $L$ is layers, $n_{\text{kv}}$ is KV heads, $T$ is sequence length, $B$ is batch. The KV cache, not the weights, is what limits how many concurrent requests you can serve — see the worked example below and [The Anatomy of LLM Inference: Prefill, Decode & The KV Cache](../07-inference-serving/01-anatomy-inference.html). Its fragmentation problem is solved by PagedAttention ([PagedAttention & KV-Cache Memory Management](../04-kernels-efficiency/06-paged-attention-kv.html)).

### Explain FlashAttention without hand-waving

**Essence.** FlashAttention computes exact attention without ever materializing the $T\times T$ score matrix in slow GPU memory (HBM). It tiles Q, K, V into blocks, keeps a running softmax in fast on-chip SRAM, and is therefore *IO-bound-aware*: it trades a little extra compute for a massive reduction in HBM traffic, which is the actual bottleneck.

The trick is the **online softmax**: you can compute a numerically-stable softmax-weighted sum in one pass if you carry a running max $m$ and running denominator $\ell$, rescaling the accumulator when a new block reveals a larger max.

```python
def online_softmax_attention(q, K_blocks, V_blocks):
    # q: (d,)  one query; K_blocks/V_blocks: list of (block, d) tiles
    import math
    m = float("-inf")      # running max of logits
    l = 0.0                # running sum of exp(logit - m)
    acc = torch.zeros_like(q)   # running output (un-normalized)
    for Kb, Vb in zip(K_blocks, V_blocks):
        s = (Kb @ q) / q.shape[-1] ** 0.5      # logits for this tile
        m_new = max(m, s.max().item())
        # rescale old accumulator + denom to the new max
        scale = math.exp(m - m_new)
        p = torch.exp(s - m_new)               # tile probabilities
        l = l * scale + p.sum().item()
        acc = acc * scale + (p[:, None] * Vb).sum(0)
        m = m_new
    return acc / l
```

That recurrence is the whole idea — FlashAttention 1 (Dao et al.) is this, tiled and fused into a CUDA kernel; FA2 improves work partitioning across warps; FA3 adds FP8 and Hopper async copies. The reason it's faster despite recomputation in the backward pass: attention is *memory-bandwidth bound*, not compute bound, so cutting HBM reads/writes from $O(T^2)$ to $O(T)$ wins. Full treatment: [FlashAttention I: IO-Awareness & The Online Softmax](../04-kernels-efficiency/02-flash-attention-1.html) and [FlashAttention 2 & 3](../04-kernels-efficiency/03-flash-attention-2-3.html).

!!! example "Worked example: KV-cache size for a 70B-class model"

    Take a Llama-70B-style config: $L=80$ layers, hidden $d_{\text{model}}=8192$, $n_{\text{head}}=64$ query heads, $d_{\text{head}}=128$, but **GQA with $n_{\text{kv}}=8$** KV heads. Serve at sequence length $T=8192$, batch $B=1$, weights in bf16 (2 bytes), so the KV cache is also bf16.

    Per token, per layer the cache stores K and V: $2\times n_{\text{kv}}\times d_{\text{head}}=2\times 8\times 128=2048$ values $=4096$ bytes.

    Across all layers and the full sequence:
    $$
    4096\ \text{B} \times L \times T = 4096 \times 80 \times 8192 \approx 2.68\times10^{9}\ \text{B} \approx 2.5\ \text{GiB per request.}
    $$

    Now compare to **multi-head attention with $n_{\text{kv}}=64$**: the cache would be $8\times$ larger, about **20 GiB per request**. On an 80 GB H100 holding ~140 GB of... wait — the 70B weights alone are ~140 GB in bf16 and won't fit; assume tensor parallelism across 2 GPUs so weights are ~70 GB/GPU, leaving ~10 GB for KV. With MHA you could barely fit *one* request; with GQA you fit *four*. That 8× is exactly why every modern model uses GQA. (FP8 KV cache would halve these again.)

## Why GQA, MQA & MLA

### Walk through MHA → MQA → GQA

**Essence.** These are points on a spectrum trading attention quality for KV-cache size. Multi-Head Attention (MHA) gives every query head its own K and V. Multi-Query Attention (MQA) shares *one* K/V across all query heads — minimal cache, but a quality hit and training instability. Grouped-Query Attention (GQA) is the compromise: split query heads into $g$ groups, each group shares one K/V head.

```text
MHA:  Q heads: [q0 q1 q2 q3 q4 q5 q6 q7]
      K/V:     [k0 k1 k2 k3 k4 k5 k6 k7]    (8 KV heads — big cache)

GQA:  Q heads: [q0 q1 q2 q3 | q4 q5 q6 q7]
      K/V:     [   k0       |    k1     ]    (2 KV heads — 4x smaller)

MQA:  Q heads: [q0 q1 q2 q3   q4 q5 q6 q7]
      K/V:     [          k0            ]    (1 KV head — smallest)
```

The KV cache shrinks by the factor $n_{\text{head}}/n_{\text{kv}}$. Since the cache, not FLOPs, bounds concurrency during decode, GQA directly multiplies your serving throughput — that's the answer to "why GQA": **it cuts KV-cache memory by the group factor with negligible quality loss, raising the batch size you can serve and thus tokens/sec/GPU.**

### What is MLA and how does it differ?

**Essence.** Mult-head Latent Attention (DeepSeek) goes further: instead of caching K and V directly, it caches a *low-rank latent* that is up-projected to K and V on the fly. The cache stores a small compressed vector per token, decoupling cache size from head count almost entirely, while preserving more expressive per-head attention than GQA. It also folds RoPE in via a separate decoupled key path. Tradeoff: more compute and more architectural complexity at train time. Full comparison: [Multi-Head Attention, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html).

!!! interview "Interview Corner"

    **Q:** A teammate proposes switching a trained MHA model to MQA to save serving memory, by averaging the K/V projection weights across heads. Will it work?

    **A:** Not out of the box. You can mean-pool the K and V projections to *initialize* a GQA/MQA variant — that's exactly the "uptraining" recipe from the GQA paper — but you must then fine-tune for a small fraction of the original compute to recover quality; a zero-shot swap degrades sharply because the query heads were trained against head-specific keys. I'd convert to GQA (say 8 KV heads, not 1) for a better quality/memory point, mean-pool to initialize, and uptrain on a few billion tokens. I'd verify with perplexity and a downstream eval that the recovered model matches baseline before shipping.

## Preference Optimization: DPO vs PPO vs GRPO

This cluster separates people who've read the RLHF papers from people who've trained the models. The full pipeline is in [The RLHF Pipeline & Reward Modeling](../05-posttraining-alignment/05-rlhf-reward-modeling.html).

### Explain the classic RLHF/PPO pipeline

**Essence.** Three stages: (1) **SFT** — supervised fine-tune on demonstrations. (2) **Reward model** — collect pairwise human preferences and train a model $r_\phi(x,y)$ to score responses, using the Bradley–Terry loss. (3) **PPO** — optimize the policy $\pi_\theta$ to maximize reward while a KL penalty keeps it near the SFT reference $\pi_{\text{ref}}$:

$$
\max_\theta\ \mathbb{E}_{x\sim D,\,y\sim\pi_\theta}\big[r_\phi(x,y)\big]-\beta\,\mathbb{D}_{\text{KL}}\big(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big)
$$

PPO needs four models in memory simultaneously: policy, reference, reward, and a **value/critic** network for advantage estimation. It's powerful but heavy, sensitive to hyperparameters, and notoriously fiddly. See [Policy Gradients & PPO for Language Models](../05-posttraining-alignment/06-ppo-for-llms.html).

### Why DPO, and what is it really doing?

**Essence.** Direct Preference Optimization removes the reward model and the RL loop entirely. The key insight: the RLHF objective has a *closed-form optimal policy*, and you can algebraically invert it so the reward is expressed in terms of the policy itself. Substituting into the Bradley–Terry preference likelihood yields a simple supervised classification loss on preference pairs $(y_w \succ y_l)$:

$$
\mathcal{L}_{\text{DPO}}=-\mathbb{E}_{(x,y_w,y_l)}\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right]
$$

The model implicitly *becomes* its own reward model. Here is the loss in full — note it's just two forward passes and a log-sigmoid:

```python
import torch.nn.functional as F

def dpo_loss(policy_logp_w, policy_logp_l,    # sum log-prob of chosen/rejected under policy
             ref_logp_w, ref_logp_l,          # ...under frozen reference
             beta=0.1):
    # "implicit reward" = beta * log-ratio of policy to reference
    pi_logratio  = policy_logp_w - policy_logp_l     # policy prefers chosen by this much
    ref_logratio = ref_logp_w   - ref_logp_l         # reference's baseline preference
    logits = beta * (pi_logratio - ref_logratio)     # how much we *increase* the margin
    loss = -F.logsigmoid(logits).mean()              # push chosen above rejected
    # implicit rewards (for logging margins / accuracy):
    chosen_reward   = beta * (policy_logp_w - ref_logp_w).detach()
    rejected_reward = beta * (policy_logp_l - ref_logp_l).detach()
    return loss, chosen_reward, rejected_reward
```

**Tradeoff.** DPO is far simpler, more stable, and cheaper (two models, no rollouts). Its weaknesses: it learns only from the *offline* preference pairs you collected — no exploration — so it can over-fit to the dataset's quirks and is prone to *length bias* and pushing down the probability of *both* responses if not regularized. On-policy methods (PPO, GRPO) can keep improving by generating fresh samples. See [Direct Preference Optimization & Its Variants](../05-posttraining-alignment/07-dpo-and-variants.html).

### What is GRPO and why did it take over for reasoning?

**Essence.** Group Relative Policy Optimization (DeepSeek) is PPO *without the critic*. Instead of a learned value network to estimate the baseline, it samples a *group* of $G$ responses per prompt, scores each, and uses the group's mean reward as the baseline. The advantage of a response is just its reward standardized within the group:

$$
\hat{A}_i=\frac{r_i-\operatorname{mean}(r_1,\dots,r_G)}{\operatorname{std}(r_1,\dots,r_G)}
$$

```python
def grpo_advantages(rewards):
    # rewards: (group_size,) scalar reward per sampled completion for ONE prompt
    mean = rewards.mean()
    std  = rewards.std() + 1e-6
    return (rewards - mean) / std        # within-group standardized advantage
```

**Why it took over.** (1) It drops the value network — halving memory and removing the hardest-to-tune component. (2) The group baseline is well-suited to **verifiable rewards** (math/code where reward is 0/1 correctness), the engine behind reasoning models. A correct answer in a group of mostly-wrong answers gets a big positive advantage; that's a clean learning signal. See [GRPO, RLOO & Critic-Free RL](../05-posttraining-alignment/08-grpo-rloo.html) and [RL with Verifiable Rewards](../05-posttraining-alignment/09-rlvr-reasoning.html). The KL term and advantage normalization details matter a lot in practice — [Advantage Estimation, KL Control & Stability Tricks](../06-rl-infra/09-advantage-kl-tricks.html).

| Method | Reward model? | Critic? | On-policy rollouts? | Main weakness |
|---|---|---|---|---|
| PPO | Yes | Yes (value net) | Yes | 4 models, fiddly tuning |
| DPO | No (implicit) | No | No (offline) | No exploration, length bias |
| GRPO | Often verifier | No (group baseline) | Yes | Needs many samples/prompt |

!!! interview "Interview Corner"

    **Q:** Your DPO-trained model gets higher reward-model scores but humans say its answers got *worse and longer*. What happened and what do you do?

    **A:** Two classic DPO pathologies. First, **length bias**: if chosen responses in your pairs were on average longer, DPO learns "longer = better" as a spurious feature. Second, **reward over-optimization / off-distribution drift**: DPO can push up the implicit margin by mostly *lowering* the rejected logprob, drifting from the reference into regions the preference data never covered. Fixes: add a length penalty or use length-normalized log-probs; switch to a length-debiased variant; increase $\beta$ to stay closer to the reference; add an SFT term on the chosen responses to anchor likelihoods. I'd validate with human eval or a held-out judge that *controls for length*, not just the reward model the policy is gaming — this is reward hacking, covered in [Reward Hacking, Over-Optimization & Alignment Failures](../05-posttraining-alignment/13-reward-hacking-failures.html).

## Quantization

The interviewer wants to know you understand *what* you're quantizing (weights? activations? KV cache?) and *why* it works without destroying the model.

### How does quantization work and why doesn't it break the model?

**Essence.** Quantization maps high-precision weights (fp16/bf16) to low-bit integers (int8, int4) using a scale (and optional zero-point), reconstructing approximate values at compute time. It works because (1) neural nets are over-parameterized and robust to weight noise, and (2) most of the cost of LLM inference is *memory bandwidth* — moving weights from HBM — so storing weights in 4 bits means 4× fewer bytes to move, a near-4× decode speedup, even if you up-cast to do the matmul.

Symmetric per-group quantization in a few lines:

```python
def quantize_int4_symmetric(w, group_size=128):
    # w: (out, in) weight matrix; quantize along input dim in groups
    out, inn = w.shape
    w = w.reshape(out, inn // group_size, group_size)
    absmax = w.abs().amax(dim=-1, keepdim=True)          # per-group scale
    scale = absmax / 7.0                                  # int4 range [-8,7] -> use 7
    q = torch.clamp(torch.round(w / scale), -8, 7)        # quantized integers
    return q.to(torch.int8), scale                        # store 4-bit packed + scale

def dequantize(q, scale, group_size=128):
    out = q.shape[0]
    return (q.float() * scale).reshape(out, -1)           # approximate original
```

### PTQ vs QAT, and what's the hard part?

**Essence.** **Post-Training Quantization (PTQ)** quantizes a finished model with a small calibration set — fast, no retraining (GPTQ, AWQ, SmoothQuant). **Quantization-Aware Training (QAT)** simulates quantization during training so the model learns to be robust — more accurate at very low bits, but expensive. The hard part is **outlier activations**: a few channels have huge magnitudes that blow up the quantization range and destroy precision for everyone else. SmoothQuant migrates the difficulty from activations into weights via a per-channel scale; AWQ protects the *salient* weight channels identified by activation magnitude; GPTQ uses second-order (Hessian) information to quantize weights while compensating for the error. KV-cache quantization is a separate axis that directly extends context length. Full treatment: [Quantization I: PTQ](../04-kernels-efficiency/07-quantization-ptq.html) and [Quantization II: Formats & QAT](../04-kernels-efficiency/08-quantization-formats-qat.html); the float side connects to [Mixed Precision, bf16 & FP8 Training](../03-pretraining/08-mixed-precision-fp8.html).

!!! warning "Common pitfall"

    "Int4 makes everything 4× faster" is only true in the *memory-bound* decode regime. During *prefill* (and any compute-bound large-batch matmul) you may be limited by GPU FLOPs, not bandwidth, and the dequantize overhead can make int4 *slower* than a well-tuned fp16/fp8 kernel. Always ask: am I bandwidth-bound or compute-bound? That's the roofline question — see [The Roofline Model & Performance Engineering](../04-kernels-efficiency/01-roofline-performance.html).

## Scaling Laws, MoE & Architecture

### State the Chinchilla result and its practical consequence

**Essence.** Kaplan et al. (2020) found loss falls as a power law in parameters $N$, data $D$, and compute $C$. Chinchilla (Hoffmann et al., 2022) refined the *compute-optimal* allocation: for a fixed compute budget, you should scale parameters and tokens **roughly equally** — about **20 tokens per parameter** — whereas earlier models like GPT-3 were badly *under-trained* (too big, too little data). The loss model:

$$
L(N,D)=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}
$$

with $\alpha,\beta\approx 0.34$. Given compute $C\approx 6ND$ FLOPs, you minimize $L$ subject to that constraint.

**The practical twist interviewers love.** Chinchilla optimizes *training* compute. But in production, **inference compute dominates** over a model's lifetime, so it's rational to "over-train" a *smaller* model on far more than 20 tokens/param (Llama-3 used trillions of tokens for 8B). You trade extra training cost for a permanently cheaper, faster model to serve. Knowing this distinction — *compute-optimal training* vs *inference-optimal deployment* — is the mark of someone who's actually shipped. See [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html).

!!! example "Worked example: sizing a run with a compute budget"

    Suppose you have a budget of $C=10^{22}$ FLOPs. Using $C\approx 6ND$ and the Chinchilla rule $D\approx 20N$:

    $$
    C = 6N(20N)=120N^2 \;\Rightarrow\; N=\sqrt{C/120}=\sqrt{10^{22}/120}\approx 9.1\times10^{9}.
    $$

    So a compute-optimal model is roughly **9B parameters** trained on $D\approx 20N\approx 1.8\times10^{11}=180$B tokens. If instead inference cost dominates and you want a 3B model, you'd push $D$ to ~600B+ tokens (well past compute-optimal) to recover quality — spending more training FLOPs to win on every future inference call.

### Explain Mixture-of-Experts and the routing problem

**Essence.** An MoE replaces the dense feed-forward layer with many expert FFNs and a *router* that sends each token to a small subset (top-$k$, often $k=2$). The result: you scale *total* parameters (knowledge capacity) while keeping *active* parameters per token — and thus FLOPs — roughly constant. A model can have 600B total but activate only ~37B per token.

```python
def moe_layer(x, experts, router, k=2):
    # x: (tokens, d_model); experts: list of FFN modules; router: Linear(d_model -> n_experts)
    logits = router(x)                          # (tokens, n_experts)
    weights, idx = torch.topk(logits.softmax(-1), k, dim=-1)   # pick top-k experts
    out = torch.zeros_like(x)
    for slot in range(k):
        for e in range(len(experts)):
            mask = idx[:, slot] == e            # tokens routed to expert e in this slot
            if mask.any():
                out[mask] += weights[mask, slot:slot+1] * experts[e](x[mask])
    return out
```

**The hard parts.** (1) **Load balancing** — without an auxiliary loss, the router collapses to a few favorite experts; you add a balancing loss (or use loss-free bias tricks) to spread tokens. (2) **All-to-all communication** — experts are sharded across GPUs (expert parallelism), so routing means shuffling tokens across the network, which can bottleneck. (3) **Training instability** and the discrete, non-differentiable top-$k$. The tradeoff: huge memory footprint (all experts must be resident) for lower FLOPs/token. Deep dive: [Mixture-of-Experts (MoE) Architectures](../02-transformer/09-mixture-of-experts.html); the parallelism in [Distributed Training II](../03-pretraining/06-distributed-model-parallel.html).

### Name three modern architecture choices and why

Quick-fire, the way you'd rattle them off: **Pre-norm** (LayerNorm/RMSNorm *before* the sublayer) for stable gradients in deep stacks; **RMSNorm** over LayerNorm (drops the mean-centering, cheaper, works as well); **SwiGLU** activation (gated GLU variant, better than ReLU/GELU at equal params); **RoPE** for relative position; **GQA** for cheap KV cache; **no bias terms** in linears (free stability). See [Modern Architecture Improvements & Design Choices](../02-transformer/10-modern-arch-improvements.html).

## Inference Acceleration & Hallucination

### Explain speculative decoding

**Essence.** Decode is memory-bandwidth bound: each step loads the entire model to produce *one* token, wasting compute. Speculative decoding uses a small, fast **draft** model to propose $k$ tokens cheaply, then the big **target** model verifies all $k$ in a *single parallel forward pass*. A rejection-sampling acceptance rule guarantees the output distribution is **identical** to the target model's — it's exact, not approximate. You get a speedup equal to the average number of tokens accepted per target pass.

```python
def speculative_step(target, draft, prefix, k=4):
    # 1) draft proposes k tokens autoregressively (cheap)
    draft_tokens, draft_probs = draft.sample(prefix, n=k)
    # 2) target scores all k+1 positions in ONE forward pass (parallel)
    target_probs = target.forward(prefix + draft_tokens)   # (k+1, vocab)
    # 3) accept/reject each drafted token to preserve target's distribution
    accepted = []
    for i, tok in enumerate(draft_tokens):
        p, q = target_probs[i][tok], draft_probs[i][tok]
        if torch.rand(1) < min(1.0, p / q):     # accept with prob min(1, p/q)
            accepted.append(tok)
        else:
            # reject: resample from the adjusted residual distribution (p - q)_+
            accepted.append(sample_residual(target_probs[i], draft_probs[i]))
            return accepted                      # stop at first rejection
    accepted.append(sample(target_probs[k]))     # bonus token: all k accepted
    return accepted
```

**Why it's free quality-wise:** the math guarantees the marginal output distribution equals the target's. Variants: **Medusa** adds extra decoding heads to the target itself (no separate draft); **EAGLE** drafts in feature space; **lookahead** uses n-gram jacobi iteration. The speedup depends entirely on draft acceptance rate — a well-aligned draft on predictable text can accept 3–4 tokens/pass. Full coverage: [Speculative Decoding: Draft Models, Medusa, EAGLE & Lookahead](../07-inference-serving/06-speculative-decoding.html). Why decode is memory-bound in the first place: [The Anatomy of LLM Inference](../07-inference-serving/01-anatomy-inference.html).

### What causes hallucination, mechanistically?

**Essence.** Hallucination — confident, fluent, *false* output — isn't a bug in one place; it's the product of how LLMs are trained and decoded. Give a layered answer:

1. **The objective rewards plausibility, not truth.** Next-token prediction (and the MLE/cross-entropy loss) optimizes for *likely* continuations given the training distribution. The model has no built-in notion of factual grounding; it interpolates over patterns. See [The Pretraining Objective & Loss](../03-pretraining/03-pretraining-objective.html).
2. **No epistemic uncertainty signal.** The model outputs a probability distribution over tokens, but a high-probability token is not a calibrated "I'm sure this is true." After RLHF, models often become *worse*-calibrated — confidently wrong — because preference data rewards confident, helpful-sounding answers.
3. **Training-time incentives to guess.** If the SFT/RL data punishes "I don't know" and rewards attempts, the model learns that guessing beats abstaining — a direct cause of fabrication on the long tail.
4. **Knowledge gaps & staleness.** Facts absent from (or rare in) pretraining, or events after the cutoff, force the model to extrapolate. Closed-book recall of rare entities is where hallucination spikes.
5. **Decoding pressure.** Sampling (temperature, top-p) injects randomness that can tip a borderline-correct answer into a wrong one; the model must emit *some* token even when no continuation is well-supported.

**Mitigations** map onto each cause: **RAG** grounds answers in retrieved evidence ([Retrieval-Augmented Generation Architectures](../09-rag-retrieval/03-rag-architectures.html)); **calibration / abstention training** rewards "I don't know"; **verifiers and tool use** check claims ([Tool Use & Function Calling](../08-agents-harness/01-tool-use-function-calling.html)); **LLM-as-judge / self-consistency** catches inconsistency ([LLM-as-a-Judge](../11-evaluation/02-llm-as-judge.html)); lower temperature reduces stochastic errors. The honest framing for an interview: hallucination is *intrinsic* to a likelihood model with no grounding; we *manage* it with retrieval, verification, and calibration rather than "fixing" it.

!!! interview "Interview Corner"

    **Q:** Is hallucination something we can fully eliminate by scaling up the model?

    **A:** No — scaling reduces it on the head of the distribution (common facts) but can't eliminate it. The model is fundamentally a probabilistic next-token predictor with no access to ground truth at inference time; for any query whose answer isn't recoverable from its weights — post-cutoff events, private data, rare entities — it must either abstain or extrapolate, and extrapolation is hallucination. Worse, there's a known tension: aggressive RLHF for helpfulness can degrade calibration, making a bigger model *more* confidently wrong. The durable fixes are architectural and systemic — grounding via retrieval, external verifiers/tools for checkable claims, and explicitly training the model to express uncertainty and abstain — not raw scale. I'd frame it to a PM as *risk management*, with measurable abstention and grounding-rate metrics, not a solved problem.

## Putting It Together: How to Answer in the Room

A meta-pattern for *every* question above:

1. **One-sentence essence first.** "GQA shares K/V across groups of query heads to shrink the KV cache." Don't make them wait.
2. **Mechanism.** The formula or the data flow — show you understand *how*, not just *that*.
3. **A number or a tradeoff.** "8× smaller cache → 8× more concurrent requests." Magnitudes prove you've touched the system.
4. **The failure mode it addresses.** Every technique exists because something broke. Naming the break shows depth.
5. **Honest edges.** "DPO is simpler but can't explore; that's why reasoning models use GRPO." Knowing the *limits* is more impressive than evangelizing.

Practice saying these out loud. The difference between a hire and a no-hire on the LLM deep-dive is rarely knowledge — it's *fluency*: the ability to go from "explain flash attention" to a clear, structured, three-layer answer in twenty seconds without a script. The chapters cross-linked throughout give you the depth; this chapter gives you the compression.

!!! key "Key Takeaways"

    - Attention is $\operatorname{softmax}(QK^\top/\sqrt{d_k})V$; the $\sqrt{d_k}$ prevents softmax saturation, and the causal mask makes it autoregressive — be able to write both from memory.
    - RoPE encodes *relative* position by rotating Q and K, costs zero parameters, and enables context extension; the KV cache (size $\propto 2\,L\,n_{\text{kv}}\,d_{\text{head}}\,T\,B$) — not the weights — bounds serving concurrency.
    - GQA cuts the KV cache by the query/KV-head ratio with negligible quality loss; MLA compresses further via a cached low-rank latent. Always derive the cache size in GiB when asked.
    - PPO needs four models including a critic; DPO drops the reward model via a closed-form inversion (offline, simple, but no exploration and length-biased); GRPO drops the critic via a group-relative baseline and shines with verifiable rewards.
    - Quantization wins mainly because decode is *memory-bandwidth bound*; the hard problem is outlier activations (SmoothQuant/AWQ/GPTQ). Int4 is not automatically faster in compute-bound regimes.
    - Chinchilla says ~20 tokens/param is *training*-compute-optimal; inference-cost reality justifies "over-training" smaller models. MoE scales total params while holding FLOPs/token via top-$k$ routing, paying in memory and all-to-all comms.
    - Speculative decoding is an *exact* speedup (draft proposes, target verifies in parallel, rejection sampling preserves the distribution). Hallucination is intrinsic to ungrounded likelihood models — manage it with retrieval, verifiers, and calibration, not scale alone.
    - In the room: essence → mechanism → number/tradeoff → failure mode → honest limits. Fluency beats recall.

## Further reading

- Vaswani et al., *Attention Is All You Need* (2017) — the transformer and scaled dot-product attention.
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* — RoPE.
- Dao et al., *FlashAttention* and *FlashAttention-2* — IO-aware exact attention.
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models* — grouped-query attention and uptraining.
- Ouyang et al., *Training Language Models to Follow Instructions with Human Feedback (InstructGPT)* — the RLHF/PPO pipeline.
- Rafailov et al., *Direct Preference Optimization* — DPO and the implicit reward derivation.
- Shao et al. / DeepSeek, *DeepSeekMath* (GRPO) and *DeepSeek-V3 / R1* — GRPO, MLA, and the reasoning recipe.
- Hoffmann et al., *Training Compute-Optimal Large Language Models (Chinchilla)*; Kaplan et al., *Scaling Laws for Neural Language Models*.
- Shazeer et al., *Outrageously Large Neural Networks (Sparsely-Gated MoE)*; Fedus et al., *Switch Transformers*.
- Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*; Chen et al., *Accelerating LLM Decoding with Speculative Sampling*.
- Frantar et al., *GPTQ*; Lin et al., *AWQ*; Xiao et al., *SmoothQuant* — post-training quantization.
