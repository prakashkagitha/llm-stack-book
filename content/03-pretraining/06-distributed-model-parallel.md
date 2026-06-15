# 3.6 Distributed Training II: Tensor, Pipeline, Sequence & Expert Parallelism

In [Distributed Training I](../03-pretraining/05-distributed-data-parallel.html) we learned to *replicate* a model across many GPUs and split the **data**. That strategy — data parallelism (DP), and its memory-sharded cousins ZeRO and FSDP — has one non-negotiable requirement: a single replica's *worth of state* (parameters, gradients, optimizer states, and at least one microbatch of activations) must fit on the device, or be shardable into something that fits. When that breaks, you must split the **model itself**.

A 70B-parameter model in bf16 is 140 GB of weights alone. Add fp32 optimizer states (Adam's two moments plus an fp32 master copy) and you are well over half a terabyte before a single token of activation memory. No single GPU — not an 80 GB H100, not a 192 GB MI300X — holds that. You have no choice but to *carve the model up* and spread the pieces across devices. This chapter is about the three orthogonal axes for doing so:

- **Tensor parallelism (TP)** — split *within* a layer (each matmul is shared across GPUs).
- **Pipeline parallelism (PP)** — split *across* layers (each GPU owns a contiguous block of layers).
- **Sequence / context parallelism (SP/CP)** — split *along the sequence dimension* (each GPU owns part of the token sequence).
- **Expert parallelism (EP)** — split *across the experts* of a Mixture-of-Experts layer.

These compose with data parallelism into what practitioners call **3D, 4D, or 5D parallelism**. Getting the composition right — and understanding exactly *where the communication lives* — is the single highest-leverage systems skill in large-scale pretraining. It is also a favorite interview topic precisely because it forces you to reason about the [memory hierarchy](../01-foundations/08-gpu-architecture.html), [collective communication](../01-foundations/09-parallel-collectives.html), and the [transformer block](../02-transformer/06-transformer-block.html) all at once.

## The Memory Wall and the Three Axes of Splitting

Let us first quantify *why* we split. Consider a dense decoder-only transformer with $L$ layers, hidden size $h$, and FFN expansion factor 4. The parameter count is dominated by

$$
N \approx L \cdot \left( \underbrace{4 h^2}_{\text{attention } QKVO} + \underbrace{8 h^2}_{\text{FFN up+down}} \right) = 12 L h^2 .
$$

For training, every parameter carries a **memory multiplier**. With mixed-precision Adam (see [Mixed Precision, bf16 & FP8 Training](../03-pretraining/08-mixed-precision-fp8.html)) and the optimizer details from [Optimizers](../03-pretraining/09-optimizers.html), the per-parameter footprint is roughly:

| State | Precision | Bytes / param |
|---|---|---|
| Weights | bf16 | 2 |
| Gradients | bf16 (or fp32) | 2–4 |
| Adam momentum $m$ | fp32 | 4 |
| Adam variance $v$ | fp32 | 4 |
| fp32 master weights | fp32 | 4 |
| **Total** | | **~16** |

So a model needs on the order of $16N$ bytes of *static* state, plus activations. For a 70B model that is $\approx 1.1$ TB — which **ZeRO/FSDP can sharded across the DP group**. But ZeRO does *not* reduce the activation memory of a single microbatch, and it does *not* reduce the size of the largest single tensor you must materialize. When a single layer's activations, or a single forward pass, are too big for one device, sharding optimizer state is not enough. That is where the model-parallel axes come in.

The mental model: **DP/ZeRO splits state across replicas of the whole model; model parallelism splits the computation graph itself.** They are orthogonal and combine multiplicatively.

{{fig:dmp-orthogonal-axes}}

## Tensor Parallelism: Splitting the Matmul

Tensor parallelism — introduced at scale by **Megatron-LM (Shoeybi et al., 2019)** — observes that the heavy lifting of a transformer is a sequence of large matrix multiplications, and a matrix multiplication can be partitioned across devices with a *single* collective per region. The art is choosing partitions so that consecutive matmuls compose without a collective *between* them.

### Column and Row Parallel Linear Layers

Take a linear layer $Y = XA$ where $X \in \mathbb{R}^{s \times h}$ (sequence $\times$ hidden) and $A \in \mathbb{R}^{h \times h'}$. There are two ways to split $A$ across $t$ GPUs.

**Column-parallel.** Split $A$ along its *output* columns: $A = [A_1, A_2, \dots, A_t]$ where each $A_i \in \mathbb{R}^{h \times h'/t}$. Each GPU holds the full input $X$ (replicated) and computes a *slice* of the output:

$$
Y_i = X A_i \in \mathbb{R}^{s \times h'/t}, \qquad Y = [Y_1, \dots, Y_t].
$$

No communication is needed to *produce* $Y$ — each GPU just holds a different chunk of the output columns. The output is **sharded** along the feature dimension.

**Row-parallel.** Split $A$ along its *input* rows: $A = [A_1; A_2; \dots; A_t]$, $A_i \in \mathbb{R}^{h/t \times h'}$, and correspondingly split the input $X = [X_1, \dots, X_t]$ along its columns. Each GPU computes a *partial sum*:

$$
Y = \sum_{i=1}^{t} X_i A_i .
$$

Each GPU produces a full-shaped $Y$ but containing only its partial contribution; an **all-reduce** sums them to the correct result.

The magic trick: **chain a column-parallel layer into a row-parallel layer and the intermediate never needs to be gathered.** A column-parallel layer leaves its output *sharded along features*; a row-parallel layer *wants* its input sharded along features. They fit like puzzle pieces, and you pay exactly **one all-reduce** at the very end (plus one in the backward pass).

{{fig:dmp-tp-column-row-flow}}

### Applying It to the Transformer Block

Megatron maps this pattern onto both sublayers of a transformer block.

**MLP block** $Y = \text{GeLU}(XA)B$:

- $A$ (the up-projection, $h \to 4h$) is **column-parallel** → the $4h$ intermediate is sharded across GPUs.
- GeLU is elementwise, so it acts independently on each shard — *no communication*, and crucially we did not have to gather before the nonlinearity (which would have been wrong to split naively).
- $B$ (the down-projection, $4h \to h$) is **row-parallel** → one all-reduce produces the final output.

**Attention block.** This is even more natural because attention is *already* partitioned by heads. With $a$ attention heads and $t$ TP ranks, give each GPU $a/t$ heads:

- The $Q, K, V$ projections are **column-parallel** — each GPU produces the Q/K/V for its own heads only.
- Each GPU runs full self-attention (softmax, the $QK^\top$, the $\times V$) *for its heads* with no cross-GPU communication. (This is why TP and [Multi-Head Attention](../02-transformer/04-mha-gqa-mla.html) are such a good fit, and why GQA changes the K/V sharding story — see the warning below.)
- The output projection $O$ is **row-parallel** → one all-reduce.

So **each transformer block needs exactly two all-reduces in the forward pass** (one after attention's output proj, one after the MLP's down proj) and two in the backward pass. In Megatron's notation these are the operators $f$ and $g$: $f$ is identity in forward / all-reduce in backward; $g$ is all-reduce in forward / identity in backward.

```python
import torch
import torch.distributed as dist
import torch.nn as nn

# Assume a TP process group `tp_group` of size `tp` already initialized.
# These two autograd functions place the all-reduces in exactly the
# right spots: g = forward-allreduce, f = backward-allreduce.

class _CopyToTPRegion(torch.autograd.Function):
    """f operator: identity forward, all-reduce backward."""
    @staticmethod
    def forward(ctx, x): return x
    @staticmethod
    def backward(ctx, grad):
        dist.all_reduce(grad, group=tp_group)   # sum grads from all TP ranks
        return grad

class _ReduceFromTPRegion(torch.autograd.Function):
    """g operator: all-reduce forward, identity backward."""
    @staticmethod
    def forward(ctx, x):
        dist.all_reduce(x, group=tp_group)      # sum partial outputs
        return x
    @staticmethod
    def backward(ctx, grad): return grad

copy_to_region   = _CopyToTPRegion.apply
reduce_from_region = _ReduceFromTPRegion.apply

class ColumnParallelLinear(nn.Module):
    """Y = X A, with A split along output columns across `tp` ranks.
    Output is sharded along the feature dim (gather_output=False)."""
    def __init__(self, in_f, out_f, tp, rank, bias=True):
        super().__init__()
        assert out_f % tp == 0
        self.out_local = out_f // tp
        # Each rank only allocates its slice of the weight.
        self.weight = nn.Parameter(torch.empty(self.out_local, in_f))
        self.bias   = nn.Parameter(torch.zeros(self.out_local)) if bias else None
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        x = copy_to_region(x)                    # f: ensures correct bwd all-reduce
        y = torch.nn.functional.linear(x, self.weight, self.bias)
        return y                                  # shape [*, out_f/tp], sharded

class RowParallelLinear(nn.Module):
    """Y = X A, with A split along input rows; input already sharded.
    Produces the full output via an all-reduce (g operator)."""
    def __init__(self, in_f, out_f, tp, rank, bias=True):
        super().__init__()
        assert in_f % tp == 0
        self.in_local = in_f // tp
        self.weight = nn.Parameter(torch.empty(out_f, self.in_local))
        # bias is added ONCE, after the reduce, so only rank 0 should hold it
        self.bias = nn.Parameter(torch.zeros(out_f)) if (bias and rank == 0) else None
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):                         # x sharded along features
        y = torch.nn.functional.linear(x, self.weight)   # partial sum
        y = reduce_from_region(y)                 # g: all-reduce -> full output
        if self.bias is not None:
            y = y + self.bias
        return y

# A Megatron-style MLP: column then row, nonlinearity in between, zero gathers.
class ParallelMLP(nn.Module):
    def __init__(self, h, tp, rank):
        super().__init__()
        self.fc1 = ColumnParallelLinear(h, 4 * h, tp, rank)   # h -> 4h, sharded
        self.fc2 = RowParallelLinear(4 * h, h, tp, rank)      # 4h -> h, all-reduce
    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))
```

### The Communication Cost of TP

The all-reduce in each block moves a tensor of shape $s \times h$. With ring all-reduce over $t$ devices, each device sends and receives $\approx 2 \cdot \frac{t-1}{t} \cdot (s \cdot h \cdot 2\text{ bytes})$ per all-reduce. Two all-reduces forward + two backward = **four all-reduces per layer per step**. This is *a lot* of traffic, and it sits squarely on the critical path: the GPUs cannot proceed past the all-reduce until it completes.

This is the defining constraint of tensor parallelism: **it must run over the fastest interconnect you have.** On a DGX/HGX node that is NVLink/NVSwitch (hundreds of GB/s, sometimes ~900 GB/s aggregate). Cross *more than* the NVLink domain — e.g. over InfiniBand between nodes — and TP collapses your throughput because the per-layer all-reduces serialize behind a 10–25× slower link. **Rule of thumb: keep the TP group inside one node, $t \le 8$ (or whatever your NVLink domain is).**

!!! warning "GQA/MQA changes the K/V sharding"
    With Grouped-Query Attention or Multi-Query Attention (see [MHA, MQA, GQA & MLA](../02-transformer/04-mha-gqa-mla.html)) there are fewer K/V heads than Q heads. If the number of KV heads is smaller than the TP degree $t$, you cannot give each rank a distinct KV head. Megatron handles this by *replicating* KV heads across the ranks that share them, or by requiring $t \le$ (number of KV groups). Forgetting this produces silent wrong results or shape errors — check your KV-head-to-TP divisibility before launching.

### Sequence Parallelism as TP's Free Companion

Look again at the Megatron block: between the two TP regions sit the **LayerNorm/RMSNorm and the dropout/residual-add**, which Megatron *replicates* (every TP rank does the same redundant work on the full $s \times h$ tensor). That replication wastes both compute and, more importantly, **activation memory**: each rank stores the full LayerNorm activations.

**Sequence parallelism (SP)** — in this Megatron sense (Korthikanti et al., 2022) — splits those replicated regions along the *sequence* dimension instead. The norm and residual are now done on $s/t \times h$ shards. The catch: the boundaries between an SP region (sharded on sequence) and a TP region (sharded on hidden/features) require a conversion. Megatron shows that the *same* all-reduce of the TP region can be **decomposed into a reduce-scatter + all-gather** that achieves the layout conversion *for the same total communication volume*. So Megatron-style SP is essentially free communication-wise and meaningfully cuts activation memory — it is now standard and always-on in Megatron. (Note: this "sequence parallelism" is a memory optimization *within* a TP group, and is distinct from **context parallelism / Ring Attention**, covered later, which is a true sequence split for long context.)

## Pipeline Parallelism: Splitting Across Layers

Tensor parallelism is bounded by the NVLink domain. To go bigger we split the model *depth-wise*: GPU 0 holds layers $0..k$, GPU 1 holds layers $k+1..2k$, and so on. A microbatch flows GPU 0 → GPU 1 → ... → GPU $p-1$ in the forward pass, and the gradients flow back. The only communication is **point-to-point** (send/recv of the activation tensor at the stage boundary) — cheap, and tolerant of slower inter-node links. This is **pipeline parallelism (PP)**.

The problem is the **bubble**. If you naively run one whole batch through the pipeline, while stage 0 computes, stages $1..p-1$ sit idle; while stage $p-1$ computes, the rest sit idle. Utilization is $1/p$ — catastrophic.

{{fig:pipeline-bubble}}

### GPipe: Microbatching to Fill the Pipe

**GPipe (Huang et al., 2019)** fixes this by chopping the minibatch into $m$ **microbatches** and streaming them. Once the pipe is full, multiple stages work in parallel on different microbatches. The schedule is "all-forward, then all-backward":


{{fig:dmp-gpipe-schedule}}


The bubble is the fill + drain time. With $p$ stages and $m$ microbatches, the **bubble fraction** is

$$
\text{bubble fraction} = \frac{p - 1}{m + p - 1}.
$$

Increase $m$ and the bubble shrinks. The rule: **$m \gg p$**. With $p=8$ and $m=8$, the bubble is $7/15 \approx 47\%$ — terrible. With $m = 64$, it is $7/71 \approx 10\%$. But GPipe has a memory problem: to do all forwards before any backward, it must **stash the activations of all $m$ in-flight microbatches** on each stage. Larger $m$ means lower bubble but higher peak activation memory — a direct tension. (Activation recomputation, from [Memory-Efficient Training](../04-kernels-efficiency/10-memory-efficient-training.html), is the usual escape valve.)

### 1F1B: The Steady-State Schedule

**PipeDream's 1F1B** ("one-forward-one-backward") schedule, adopted by Megatron, fixes the memory blowup. Once the pipeline is full, each stage *alternates*: do one forward, then one backward, then one forward, and so on. The key consequence: a stage only needs to keep activations for the microbatches *currently in flight through it*, which is at most $p - s$ for stage $s$ — bounded by $p$, **independent of $m$**.


{{fig:dmp-1f1b-schedule}}


The bubble fraction is the *same* $\frac{p-1}{m+p-1}$ as GPipe — 1F1B's win is **memory**, not bubble. It lets you run a large $m$ (small bubble) without storing $m$ microbatches' activations. This is why 1F1B is the default in essentially every production framework.

```python
# Minimal 1F1B driver (single-stage view). In reality each rank runs this with
# send/recv to its neighbors. `num_micro` = m, `stage` in [0, p-1], `p` = #stages.
def run_1f1b(stage, p, num_micro, fwd_step, bwd_step, recv_act, send_act,
             recv_grad, send_grad):
    warmup = p - stage - 1                 # how many forwards before first backward
    warmup = min(warmup, num_micro)
    steady = num_micro - warmup
    act_queue = []                         # activations awaiting their backward

    # ---- warmup: only forwards, prime the pipe ----
    for _ in range(warmup):
        x = recv_act() if stage > 0 else next_input()
        y, act = fwd_step(x)               # act = saved tensors for backward
        send_act(y) if stage < p - 1 else None
        act_queue.append(act)

    # ---- steady state: 1 forward then 1 backward, bounded memory ----
    for i in range(steady):
        x = recv_act() if stage > 0 else next_input()
        y, act = fwd_step(x)
        send_act(y) if stage < p - 1 else None
        act_queue.append(act)
        # immediately do a backward for the OLDEST in-flight microbatch
        g = recv_grad() if stage < p - 1 else loss_grad()
        gx = bwd_step(act_queue.pop(0), g)
        send_grad(gx) if stage > 0 else None

    # ---- cooldown: drain remaining backwards ----
    for _ in range(warmup):
        g = recv_grad() if stage < p - 1 else loss_grad()
        gx = bwd_step(act_queue.pop(0), g)
        send_grad(gx) if stage > 0 else None
```

### Interleaved 1F1B (Virtual Pipeline Stages)

We can shrink the bubble *without* more microbatches. **Interleaved 1F1B** (Megatron-LM, Narayanan et al., 2021) gives each physical GPU *several non-contiguous* chunks of layers — "virtual stages." With $v$ virtual stages per device, the pipeline has $p \cdot v$ logical stages, and the bubble shrinks to

$$
\text{bubble fraction} = \frac{1}{v} \cdot \frac{p - 1}{m + p - 1}.
$$

A factor $v$ improvement in the bubble. The cost is $v\times$ more point-to-point communication (more, smaller sends) and a more intricate schedule. With fast intra-cluster links this is usually a great trade, and interleaving is standard in large Megatron runs.

```text
Interleaved (v=2): each GPU owns TWO chunks. GPU0 = {layers 0-1, 8-9}, etc.
The pipe is "deeper" (more stages) so fill/drain is proportionally smaller,
at the price of more boundary send/recv ops.
```

### Zero-Bubble and the Frontier

The bubble is fundamentally about the *forward → backward dependency*. **Zero-Bubble Pipeline (Qi et al., 2023)** observes that the backward pass actually splits into two pieces: the gradient w.r.t. the *input* (needed to keep the pipeline flowing upstream) and the gradient w.r.t. the *weights* (needed only before the optimizer step, and *not* on the critical path). By scheduling the weight-gradient computation into the bubbles, the bubble can be driven to near zero. DeepSeek-V3's "DualPipe" pushes this further by overlapping forward and backward across a *bidirectional* pipeline and hiding communication. These are the current frontier — but plain interleaved 1F1B remains the workhorse you should reach for first.

!!! warning "Pipeline parallelism needs load-balanced stages"
    The pipeline runs at the speed of its *slowest* stage. The embedding layer (huge vocab matmul) and the final LM head + loss are unusually heavy. If you naively put $L/p$ layers per stage, stage 0 (with embeddings) and stage $p-1$ (with the LM head and cross-entropy over the full vocab) become stragglers and stall everyone. Megatron rebalances by giving the first/last stages *fewer* transformer layers, and sometimes splits the loss computation. Always profile per-stage time, not just per-stage layer count.

## Sequence & Context Parallelism: Splitting the Sequence

For very long context (32k, 128k, 1M tokens — see [Long-Context Pretraining](../03-pretraining/13-long-context-pretraining.html)), the bottleneck is no longer parameters but **activation memory and the $O(s^2)$ attention computation**, both scaling with sequence length $s$. Neither TP nor PP addresses this directly — TP shards heads, PP shards layers, but every device still processes all $s$ tokens of its slice. **Context parallelism (CP)** shards the *token sequence itself* across devices: GPU 0 owns tokens $0..s/c - 1$, GPU 1 owns the next chunk, and so on, for a CP group of size $c$.

The MLP and the projections are trivially sequence-parallel (they act per-token). The hard part is **attention**, where every query must attend to *all* keys/values — including those living on other devices.

### Ring Attention

**Ring Attention (Liu et al., 2023)** solves this by combining the [FlashAttention](../04-kernels-efficiency/02-flash-attention-1.html) online-softmax trick with a communication ring. Each device starts with its own block of $Q$, $K$, $V$. It computes the local attention contribution, then the $K$/$V$ blocks are passed around a ring (device $i \to i+1$) so that, over $c$ steps, every device sees every other device's $K$/$V$ — and crucially, **the $K$/$V$ for the next step is being sent while the current step is being computed**, so communication hides under computation.

The online-softmax running statistics (the running max $\ell$ and running sum $m$, from FlashAttention) let each device *incrementally* fold in each incoming K/V block without ever materializing the full $s \times s$ attention matrix:

```python
# Ring Attention: each of c devices owns a contiguous query/key/value block.
# We rotate K,V around the ring; online softmax accumulates the result.
# `send_recv_ring(t)` sends t to rank+1 and returns the tensor from rank-1.
import torch, math

def ring_attention(q, k, v, cp_group_size, head_dim):
    # q,k,v: local blocks, shape [b, heads, s_local, d]
    scale = 1.0 / math.sqrt(head_dim)
    # running online-softmax state, FlashAttention-style
    out   = torch.zeros_like(q)                      # accumulated output
    l_run = torch.zeros(*q.shape[:-1], 1, device=q.device)   # running sum of exp
    m_run = torch.full((*q.shape[:-1], 1), -1e30, device=q.device)  # running max
    k_cur, v_cur = k, v
    for step in range(cp_group_size):
        # Begin sending current K,V to the next rank; overlaps with the matmul.
        k_next, v_next = send_recv_ring(k_cur), send_recv_ring(v_cur)
        s = torch.matmul(q, k_cur.transpose(-1, -2)) * scale   # [b,h,s_loc,s_loc]
        # (apply causal mask here if needed, accounting for block offsets)
        m_new = torch.maximum(m_run, s.max(dim=-1, keepdim=True).values)
        p = torch.exp(s - m_new)                     # rescaled exp
        corr = torch.exp(m_run - m_new)              # correction for old terms
        l_run = corr * l_run + p.sum(dim=-1, keepdim=True)
        out   = corr * out + torch.matmul(p, v_cur)  # fold this block's V
        m_run = m_new
        k_cur, v_cur = k_next, v_next                # rotate to next block
    return out / l_run                               # normalize at the very end
```

The communication per step is one $K$ and one $V$ block ($\approx 2 \cdot \frac{s}{c} \cdot h \cdot 2$ bytes), and there are $c$ steps — so total volume per device is $O(s \cdot h)$, independent of $c$, and it overlaps with the $O(s^2/c)$ local compute. As $s$ grows, compute dominates communication and Ring Attention scales to arbitrarily long sequences as long as you add devices.

!!! warning "Causal masking unbalances the ring"
    With a causal mask, early query blocks attend to fewer keys than late ones, so a naive Ring Attention has devices doing wildly different amounts of work (some K/V blocks are entirely masked out for a given Q block). Production implementations (e.g. **Striped/Zig-Zag Ring Attention**, and Megatron-CP) renumber or interleave the token assignment so each device gets a balanced mix of early and late positions. Ignore this and your "8-way CP" runs at the speed of the busiest rank.

## Expert Parallelism: Scaling Mixture-of-Experts

A **Mixture-of-Experts (MoE)** layer (see [Mixture-of-Experts Architectures](../02-transformer/09-mixture-of-experts.html)) replaces the single FFN with $E$ expert FFNs and a router that sends each token to its top-$k$ experts. The whole point is that the *active* compute per token stays constant (only $k$ of $E$ experts fire) while the *parameter count* grows with $E$. But that means the parameters are enormous — far too big to replicate. **Expert parallelism (EP)** places different experts on different devices: with an EP group of size $e$, each device holds $E/e$ experts.

Because the router sends tokens to experts that live on *other* devices, EP's signature communication is a pair of **all-to-all** collectives:

1. **Dispatch all-to-all:** after routing, each device sends each token to the device holding its chosen expert(s).
2. (Each device runs its local experts on the tokens it received.)
3. **Combine all-to-all:** send the expert outputs back to the device that owns each token, where they are weighted by the router scores and summed.


{{fig:dmp-ep-all-to-all}}


```python
# Sketch of one expert-parallel MoE layer. `ep_group` spans `e` ranks;
# this rank owns experts [rank*E_local : (rank+1)*E_local].
import torch, torch.distributed as dist, torch.nn.functional as F

def moe_forward(x, gate, experts_local, E, e, k=1):
    # x: [tokens, h];  gate: Linear(h, E);  experts_local: list of FFNs on this rank
    logits = gate(x)                                   # [tokens, E]
    topk = logits.topk(k, dim=-1)                      # choose k experts/token
    probs = F.softmax(topk.values, dim=-1)             # routing weights
    expert_ids = topk.indices                          # [tokens, k]

    # Build send buffers: bucket tokens by the DEVICE that owns their expert.
    E_local = E // e
    dest_rank = expert_ids // E_local                  # which device per assignment
    # (real code: sort tokens by dest_rank, compute per-rank counts, pad)
    send_buf, counts = bucket_tokens_by_rank(x, dest_rank, e)

    # 1) DISPATCH: all-to-all sends each token to its expert's owner.
    recv_buf = torch.empty_like(send_buf)
    dist.all_to_all_single(recv_buf, send_buf, group=ep_group)

    # 2) Run the local experts on received tokens.
    local_out = run_local_experts(recv_buf, experts_local, E_local)

    # 3) COMBINE: all-to-all back to each token's original owner, then weight+sum.
    out = torch.empty_like(local_out)
    dist.all_to_all_single(out, local_out, group=ep_group)
    return weighted_scatter_add(out, probs, counts)    # combine top-k, scale by probs
```

The deciding factor for EP performance is **load balance**. If the router sends most tokens to a few popular experts, those devices become stragglers while others idle, and the all-to-all is dominated by the heaviest bucket. This is why MoE training relies on an **auxiliary load-balancing loss** (or DeepSeek-style auxiliary-loss-free bias correction) to spread tokens evenly, and on **expert capacity** limits that drop or reroute overflow tokens. EP is almost always combined with TP and DP, and the all-to-all collectives are extremely bandwidth-sensitive — keep the EP group on fast links, and overlap dispatch/combine with the attention compute of the next layer where possible.

## Combining Everything: 3D, 4D & 5D Parallelism

No single axis suffices at frontier scale. Real systems compose them. The total number of GPUs is the product:

$$
G = \underbrace{d}_{\text{DP}} \times \underbrace{t}_{\text{TP}} \times \underbrace{p}_{\text{PP}} \times \underbrace{c}_{\text{CP}} \times \underbrace{e}_{\text{EP}} .
$$

Each GPU belongs to one *group* per axis. The orchestration trick is **mapping these groups onto the physical network topology** so that the chattiest collectives ride the fastest links. The canonical ordering, fastest-comm axis innermost:


{{fig:dmp-axis-placement-priority}}


This is why you see configs like "TP=8 (within a node), PP=12 (across nodes), DP=16 (across racks)." TP is locked inside the NVLink domain; PP and DP span the InfiniBand/Ethernet fabric where their lighter, less frequent communication is tolerable.

A useful way to think about it: **TP and PP both reduce per-device memory and let you fit a bigger model; DP buys throughput; CP buys context length; EP buys parameter count.** You first pick TP/PP/CP/EP large enough that one model replica fits and trains efficiently, then set DP to consume the remaining GPUs for throughput.

!!! example "Worked example: sharding a 70B model on a 512-GPU cluster"
    Take a 70B dense model: $L = 80$ layers, $h = 8192$, $a = 64$ heads, sequence $s = 8192$, on **512 H100s (80 GB)** arranged as 64 nodes × 8 GPUs (NVLink within a node, InfiniBand across).

    **Static memory.** Full training state is $\approx 16 \times 70\text{B} = 1120$ GB — about $14\times$ an 80 GB GPU. We must shard the model at least 14-way *before* DP.

    **Step 1 — Tensor parallelism.** Set $t = 8$ to fill the NVLink domain. Weights+optimizer per GPU drop to $\approx 1120/8 = 140$ GB. Still too big for one GPU — TP alone is not enough.

    **Step 2 — Pipeline parallelism.** Add $p = 8$ stages across 8 nodes. Now each GPU holds $80/8 = 10$ layers, and static state per GPU is $\approx 1120/(8 \cdot 8) = 17.5$ GB. Comfortably fits, leaving room for activations and the KV/communication buffers.

    **Step 3 — Data parallelism.** We have used $t \times p = 64$ GPUs for one replica. The cluster has 512, so $d = 512/64 = 8$ data-parallel replicas. Layer in ZeRO-1 (shard optimizer state across the 8 DP ranks) to shave static memory further.

    **Step 4 — Bubble check.** With $p = 8$ and a global batch of, say, $m = 64$ microbatches per replica, the 1F1B bubble is $\frac{p-1}{m+p-1} = \frac{7}{71} \approx 9.9\%$. Add interleaving with $v = 2$ and it halves to $\approx 5\%$.

    **Result:** a **3D config TP=8 × PP=8 × DP=8 = 512 GPUs**, model fits with headroom, pipeline bubble ~5–10%, and every heavy collective (TP all-reduce) stays on NVLink. This is a realistic, near-optimal layout — and exactly the kind of back-of-envelope an interviewer wants to see.

!!! interview "Interview Corner"
    **Q:** You're training a model where one layer's activations fit on a GPU but the *full model's* parameters do not. You're told inter-node bandwidth is 10× slower than intra-node NVLink. How do you choose between tensor and pipeline parallelism, and how do you place them?

    **A:** Tensor parallelism does a synchronous all-reduce *twice per layer per direction* — that traffic is on the critical path, so TP must live entirely inside the fast NVLink domain (typically $t \le 8$). Pipeline parallelism only does point-to-point sends at stage boundaries (a handful per step), so it tolerates the 10×-slower inter-node link well. So: use TP *within* a node to get the model small enough to fit the NVLink domain, then use PP *across* nodes to add more capacity. Concretely, set TP = node size (e.g. 8), then PP across nodes to fit the parameter budget, then DP across the remaining GPUs for throughput. Two cautions: keep the pipeline bubble small with many microbatches ($m \gg p$) and/or interleaving, and load-balance the pipeline stages (the embedding and LM-head stages are heavy — give them fewer transformer layers).

!!! tip "Practitioner tip"
    Don't reach for model parallelism prematurely. The decision ladder is: (1) plain DDP if it fits; (2) ZeRO/FSDP to shard optimizer/grad/param state — this alone handles surprisingly large models with *no* model-parallel complexity (see [Distributed Training I](../03-pretraining/05-distributed-data-parallel.html)); (3) add TP once a single replica won't fit even sharded, keeping it intra-node; (4) add PP to cross node boundaries; (5) add CP only when context length (not parameters) is the binding constraint; (6) EP only for MoE. Every axis you add multiplies the debugging surface — add the *fewest* that make the run fit and run efficiently.

## Where the Communication Lives: A Summary Table

The single most important thing to internalize is *what collective each axis pays, how often, and over which tensor*. This is what lets you reason about whether a config will be fast on your hardware.

| Axis | Splits | Collective | Frequency | Tensor moved | Link needed |
|---|---|---|---|---|---|
| **DP / ZeRO** | data (replicas) | all-reduce (or RS+AG) | once per **step** | full gradients | slowest OK |
| **TP** | within a layer | all-reduce ($\times 4$/layer) | every **layer** | $s \times h$ activations | NVLink only |
| **PP** | across layers | point-to-point send/recv | stage boundaries | $s \times h$ activations | inter-node OK |
| **CP (Ring)** | sequence | ring send/recv (overlapped) | per attention | $K,V$ blocks | fast, overlaps |
| **EP** | experts | all-to-all ($\times 2$) | every **MoE layer** | routed tokens | fast (bandwidth) |

Notice the frequency column. TP and EP pay *per layer*, so they demand the fastest links and the tightest placement. DP pays *per step* (after gradient accumulation over all microbatches), so it tolerates the slow outer fabric. PP's communication is cheap point-to-point but introduces the *bubble*, a compute-utilization tax rather than a bandwidth one. CP's communication overlaps with compute and so is nearly hidden when sequences are long. This table — internalized — is most of what you need to design or debug a large training run, and it connects directly to the [collective communication](../01-foundations/09-parallel-collectives.html) primitives and the practical framework details in [Megatron-LM, DeepSpeed & Parallelism in Practice](../03-pretraining/07-megatron-deepspeed.html).

!!! note "Activation memory is the quiet killer"
    Engineers obsess over parameter memory because it's easy to compute, but at long sequence lengths and large microbatch counts, **activation memory often dominates and is what actually OOMs your run.** TP (with Megatron sequence parallelism) and CP both attack activation memory directly; PP's 1F1B bounds in-flight activations; and gradient checkpointing/recomputation trades compute to slash it further. When a large run OOMs, your first hypothesis should usually be activations, not weights.

## Key Takeaways

!!! key "Key Takeaways"
    - **Three orthogonal model-parallel axes** plus DP: TP splits *within* a layer (matmuls), PP splits *across* layers, CP/SP splits *along the sequence*, EP splits *across MoE experts*. They compose multiplicatively into 3D/4D/5D parallelism, and total GPUs $= d \cdot t \cdot p \cdot c \cdot e$.
    - **Tensor parallelism = column-then-row matmul partitioning.** A column-parallel layer feeding a row-parallel layer needs exactly one all-reduce per region (two per transformer block forward), with the nonlinearity acting on sharded features in between. It is bandwidth-heavy and *must stay inside the NVLink domain* ($t \le 8$).
    - **Pipeline parallelism trades the bubble for cheap point-to-point comms.** GPipe streams $m$ microbatches; 1F1B keeps the same $\frac{p-1}{m+p-1}$ bubble but bounds activation memory; interleaving divides the bubble by $v$; zero-bubble schedules push it toward zero. Keep $m \gg p$ and load-balance the stages.
    - **Context parallelism (Ring Attention)** shards the token sequence and rotates K/V around a ring, using the FlashAttention online softmax to fold in remote blocks while overlapping communication with compute — the key to million-token context. Mind causal-mask load imbalance.
    - **Expert parallelism** places experts on different devices and pays two all-to-all collectives (dispatch + combine) per MoE layer; its performance is gated by **router load balance** and expert capacity.
    - **Place axes by communication intensity:** TP and EP (per-layer collectives) on the fastest links; PP and DP (boundary/per-step) on slower fabric. This single placement principle drives most real configs.
    - **Climb the ladder, don't leap:** DDP → ZeRO/FSDP → +TP (intra-node) → +PP (inter-node) → +CP (for context) → +EP (for MoE). Add the fewest axes that make the run fit and run efficiently — each one multiplies the debugging surface.
    - **Activation memory, not parameters, is often what OOMs you** at scale; TP+sequence-parallel, CP, 1F1B, and recomputation are your levers against it.

!!! sota "State of the Art & Resources (2026)"
    Tensor, pipeline, sequence, and expert parallelism are now mature, production-proven techniques: every frontier training run (GPT-4, Llama, DeepSeek-V3) uses some combination of all four axes. Active research is pushing toward zero-bubble schedules, compute-communication overlap (DualPipe), and smarter MoE load balancing — squeezing the last few percent of MFU out of clusters of thousands of GPUs.

    **Foundational work**

    - [Shoeybi et al., *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism* (2019)](https://arxiv.org/abs/1909.08053) — introduced column/row parallel linear layers and the two-all-reduce-per-block TP formulation that every framework still uses.
    - [Huang et al., *GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism* (2019)](https://arxiv.org/abs/1811.06965) — established the microbatch pipeline schedule and the bubble-fraction formula.
    - [Narayanan et al., *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM* (2021)](https://arxiv.org/abs/2104.04473) — interleaved 1F1B virtual stages, 3D-parallelism analysis, and the first trillion-parameter training result.
    - [Korthikanti et al., *Reducing Activation Recomputation in Large Transformer Models* (2022)](https://arxiv.org/abs/2205.05198) — Megatron sequence parallelism (SP) and selective activation recomputation; SP is now always-on in Megatron.

    **Recent advances (2023–2026)**

    - [Liu et al., *Ring Attention with Blockwise Transformers for Near-Infinite Context* (2023)](https://arxiv.org/abs/2310.01889) — overlaps K/V ring-rotation with FlashAttention blockwise compute to scale context parallelism to arbitrarily long sequences.
    - [Qi et al., *Zero Bubble Pipeline Parallelism* (2024)](https://arxiv.org/abs/2401.10241) — splits the backward pass to schedule weight gradients into bubble slots, achieving near-zero bubble fraction under synchronous semantics; up to 31 % throughput gain.
    - [DeepSeek-AI, *DeepSeek-V3 Technical Report* (2024)](https://arxiv.org/abs/2412.19437) — details DualPipe (bidirectional pipeline overlapping forward/backward with all-to-all MoE communication) and 64-way expert parallelism across 8 nodes; the clearest public description of 5D parallelism at scale.

    **Open-source & tools**

    - [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) — the reference implementation of TP, PP, SP, CP, and EP; most large public training runs are based on or validated against it.
    - [deepseek-ai/DualPipe](https://github.com/deepseek-ai/DualPipe) — standalone PyTorch implementation of the DualPipe bidirectional pipeline algorithm from DeepSeek-V3/R1 training.

    **Go deeper**

    - [NVIDIA Technical Blog: *Scaling Language Model Training to a Trillion Parameters Using Megatron*](https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/) — accessible walkthrough of 3D parallelism placement and the trillion-parameter milestone.
    - [Megatron-Core Parallelism Strategies Guide](https://docs.nvidia.com/megatron-core/developer-guide/0.16.0/user-guide/parallelism-guide.html) — official reference covering TP, PP, CP, EP, and recommended configs for LLaMA, GPT, Mixtral, and DeepSeek models.

## Further reading

- Shoeybi, Patwary, Puri, et al., *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism* (2019) — the column/row tensor-parallel formulation.
- Narayanan, Shoeybi, Casper, et al., *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM* (2021) — interleaved 1F1B and the 3D-parallelism analysis.
- Korthikanti, Casper, Lym, et al., *Reducing Activation Recomputation in Large Transformer Models* (2022) — Megatron sequence parallelism and selective recomputation.
- Huang, Cheng, Bapna, et al., *GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism* (2019).
- Narayanan, Harlap, Phanishayee, et al., *PipeDream: Generalized Pipeline Parallelism for DNN Training* (2019) — the 1F1B schedule.
- Qi, Wan, Huang, et al., *Zero Bubble Pipeline Parallelism* (2023).
- Liu, Zaharia, Abbeel, *Ring Attention with Blockwise Transformers for Near-Infinite Context* (2023).
- Lepikhin, Lee, Xu, et al., *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding* (2020) — expert parallelism and all-to-all dispatch/combine.
- Rajbhandari, Rasley, Ruwase, He, *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* (2020) — the DP-side counterpart to this chapter.
- The **Megatron-LM** and **DeepSpeed** open-source repositories — the reference implementations of every schedule discussed here.
