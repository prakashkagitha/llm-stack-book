# 7.2 Continuous Batching & Request Scheduling

In [The Anatomy of LLM Inference](../07-inference-serving/01-anatomy-inference.html) we saw that a single decode step is wildly memory-bound: you stream gigabytes of weights through the GPU's compute units just to produce **one** token for **one** request. The arithmetic intensity of a decode step is on the order of a handful of FLOPs per byte loaded, while a modern GPU wants hundreds. The fix is batching — run many requests through the same weight load so the cost is amortized. But naive batching, the kind you'd write in a tutorial, leaves most of that throughput on the floor.

This chapter is about the scheduling discipline that real serving systems (vLLM, TGI, TensorRT-LLM, SGLang) use to keep the GPU saturated: **continuous batching**, also called **iteration-level scheduling** or **in-flight batching**. We will build the idea up from static batching, see exactly where it bleeds, introduce Orca's key insight, and then write a toy scheduler from scratch that you could extend into something real. By the end you should be able to draw the scheduler loop on a whiteboard, reason about head-of-line blocking and preemption, and estimate the throughput multiplier continuous batching buys you.

## Why naive batching wastes the GPU

Let us be precise about what we are batching. An autoregressive decode generates tokens one step at a time. For a batch of $B$ requests, step $t$ of generation does a forward pass that, per layer, multiplies a $B \times d$ activation matrix by the weight matrices, attends over each request's KV cache, and emits one new token per request. The weights are loaded from high-bandwidth memory (HBM) **once per step regardless of $B$**. So the marginal cost of adding a request to a decode step is nearly free in memory traffic — it adds a row to a GEMM that was bandwidth-bound anyway. That is the whole economic case for batching: throughput scales close to linearly with $B$ until you saturate compute or run out of KV-cache memory.

### Static batching and its three leaks

The simplest scheme is **static batching** (sometimes "request-level batching"): collect $B$ requests, run prefill on all of them, then loop decode steps until **every** request in the batch has finished, then return all results and start the next batch. It is easy to implement and it is what a plain `model.generate()` call over a padded batch gives you. It also leaks throughput in three distinct ways.

**Leak 1 — Variable sequence lengths cause tail idling.** Requests in a batch finish at different times. One request emits an end-of-sequence (EOS) token after 12 tokens; another generates 800. In static batching the short request's *slot* sits in the batch, occupying memory and compute, doing nothing useful, until the longest request finishes. If output lengths are heavy-tailed (they usually are — chat, code, and reasoning traces vary enormously), the effective utilization of the batch is the *mean* length divided by the *max* length, which can be well under 50%.


{{fig:contbatch-static-tail-idling}}


**Leak 2 — Head-of-line (HOL) blocking at admission.** A request that arrives at step 1, just after a batch was formed, must wait for the *entire* batch to drain before it is even admitted. Under static batching, time-to-first-token (TTFT) for a freshly arrived request is bounded below by the remaining decode length of whatever batch is currently running. With long generations that can be seconds. This is a latency disaster for interactive serving.

**Leak 3 — Padding waste in prefill.** To run prefill as one tensor, static batching pads all prompts to the longest prompt in the batch. If prompts are 50, 60, and 2000 tokens, the two short ones are padded to 2000 and you compute (and often attend over) a pile of pad tokens. (Real systems avoid this with ragged/packed prefill — see [Chat Templates, Data Formatting & Sequence Packing](../05-posttraining-alignment/02-chat-templates-packing.html) — but the naive version pays it.)

### Dynamic batching: better, still request-level

A first improvement, common in pre-LLM serving (Triton's dynamic batcher, TF-Serving), is **dynamic batching**: wait up to a small window (say 5–50 ms) to accumulate arriving requests, then dispatch them as one batch. This raises $B$ and amortizes weights better, and it bounds the queueing delay by the window. But it is still **request-level**: once a batch is dispatched, its membership is frozen until completion. Leaks 1 and 2 remain — you have only changed *when* batches form, not the fact that a slot is locked for the whole generation. Dynamic batching is the right tool for fixed-shape models (a ResNet classifier, an embedding model with one forward pass). It is the wrong tool for autoregressive generation, where each request runs a *different number of forward passes*.

The fundamental mismatch is this: generation is **iterative and variable-length**, but request-level batching makes a scheduling decision **once per request**. We want to make a scheduling decision **once per iteration**.

## The key insight: iteration-level scheduling

The pivotal idea comes from **Orca** (Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models*, OSDI 2022). Orca's contribution is **iteration-level scheduling** (also called continuous batching or in-flight batching): instead of scheduling a batch and running it to completion, the scheduler runs **one decode iteration at a time**, and *after every iteration* it is free to remove finished requests and admit new ones.

Reframe the server's main loop. Instead of:

```text
WHILE requests remain:
    batch = pick B requests
    prefill(batch)
    WHILE any request in batch unfinished:
        decode_step(batch)          # membership frozen
    emit results for batch
```

we run:

```text
WHILE server is up:
    batch = scheduler.pick_running_set()    # chosen FRESH each iteration
    outputs = model.step(batch)             # exactly ONE forward pass
    for req in batch:
        req.append(outputs[req])
        if req.is_finished():
            scheduler.retire(req)            # frees its slot + KV pages NOW
    scheduler.admit_new_if_room()            # backfill the freed slots
```

The batch is re-derived every single forward pass. The moment request A emits EOS, its slot and its KV-cache pages are released and a waiting request can take its place **on the very next iteration** — no waiting for B, C, and D to finish. This directly plugs Leak 1 (no tail idling: a finished slot is immediately reused) and Leak 2 (no HOL blocking at admission: a new request is admitted as soon as there is room, typically within one iteration ≈ tens of milliseconds).

### Selective batching: the subtlety Orca had to solve

There is a wrinkle that makes iteration-level batching non-trivial. If you want a single batched forward pass to contain requests at *different points* in their generation — some doing their first forward pass (prefill, with many input tokens), others doing a decode step (one token) — the tensor shapes do not line up. A decode request contributes one query position; a prefill request contributes hundreds. You cannot simply stack them into one rectangular `[B, seq, d]` tensor without padding back to the very waste we are trying to avoid.

Orca's answer is **selective batching**: batch the operations that *can* be batched across heterogeneous requests (the big token-wise GEMMs — QKV projection, the MLP, the output projection — which act independently per token and just need all tokens flattened into one long `[total_tokens, d]` matrix), and run **attention** per-request (or per-group), because attention needs each token to attend only over its own request's KV cache. Concretely:

- **Linear/MLP layers** are *position-independent*: token $i$'s output depends only on token $i$'s input. So flatten every token from every request — prefill tokens and decode tokens alike — into one tall matrix and do one big GEMM. Maximum batching, no padding.
- **Attention** is *position-dependent and per-sequence*: token $i$ of request A must attend over A's keys/values, not B's. So split back out by request, run attention for each (modern kernels like FlashAttention with a `varlen`/`cu_seqlens` interface do exactly this in one launch using cumulative sequence-length offsets — see [FlashAttention I: IO-Awareness & The Online Softmax](../04-kernels-efficiency/02-flash-attention-1.html)).

This "flatten for GEMMs, split for attention" pattern is what makes it possible to mix prefill and decode in one iteration at all, and it is the conceptual ancestor of **chunked prefill** (covered in [Disaggregated Prefill/Decode & Chunked Prefill](../07-inference-serving/08-disaggregated-chunked-prefill.html)). Ragged attention via cumulative sequence lengths is the standard way every serving stack implements it today.


{{fig:contbatch-selective-batching-flatten}}


## The scheduler: state, decisions, and constraints

Now we get concrete about the component that decides the running set each iteration. A continuous-batching scheduler maintains three logical pools and makes one decision per loop.


{{fig:contbatch-scheduler-state-machine}}


- **WAITING** — requests that have arrived but hold no KV-cache pages yet. Ordered by the scheduling policy (FCFS, priority, deadline, …).
- **RUNNING** — requests in the batch for the upcoming forward pass; each owns KV-cache pages for its tokens so far.
- **FINISHED** — emitted EOS or hit `max_tokens`; results streamed back, resources freed.

Each iteration, the scheduler answers: *which requests run this step, and do I need to evict anyone to make room?* The two binding constraints are:

1. **KV-cache memory.** Every running request's keys and values occupy memory that grows with its sequence length. The KV cache, not raw FLOPs, is almost always the limiting resource. With **PagedAttention** ([PagedAttention & KV-Cache Memory Management](../04-kernels-efficiency/06-paged-attention-kv.html)) the cache is allocated in fixed-size *blocks* (e.g. 16 tokens each), so admitting a request means reserving enough free blocks for its current length, and a request growing by one token may need to grab one more block. The scheduler must track the free-block count and refuse to admit (or must preempt) when blocks run out.

2. **Per-iteration token budget.** To bound iteration latency you cap the number of tokens processed per step, `max_num_batched_tokens` (and often a separate `max_num_seqs`, a cap on the number of concurrent requests). Prefill tokens are the expensive ones; a 4000-token prompt arriving in one shot would blow a latency target, which is exactly why chunked prefill exists — it splits that prefill across several iterations so each step stays under budget. The scheduler decides how much prefill work to admit per iteration against this budget.

### A throughput model for the scheduler

It helps to have a back-of-envelope model. Let the GPU sustain a decode step of $B$ requests in time $\tau_{\text{dec}}(B)$. Because decode is memory-bound, $\tau_{\text{dec}}(B) \approx \tau_0 + \beta B$ where $\tau_0$ (the fixed cost of streaming weights once) dominates and $\beta$ (the small marginal per-request cost) is tiny until you approach compute saturation. Token throughput is

$$
\text{tokens/sec} = \frac{B}{\tau_{\text{dec}}(B)} = \frac{B}{\tau_0 + \beta B} \xrightarrow[B \to \infty]{} \frac{1}{\beta}.
$$

The curve rises steeply with $B$ at first (you are amortizing the fixed $\tau_0$ over more requests) and then flattens toward the compute roofline $1/\beta$ (see [The Roofline Model & Performance Engineering](../04-kernels-efficiency/01-roofline-performance.html)). The scheduler's job is to keep $B$ as large as the KV-cache budget allows, so you operate on the flat, high-throughput part of the curve rather than the starved low-$B$ part. Static batching keeps $B$ *high on average but low at the tail* (the batch drains down to one straggler running at $B=1$); continuous batching keeps $B$ pinned near the maximum continuously by backfilling. That gap is where the throughput multiplier comes from.

## A toy continuous-batching scheduler, from scratch

{{fig:continuous-batching}}

Let us build a runnable simulator that captures the real mechanics: a token budget, a KV-block budget, iteration-level admission and retirement, prefill/decode interleaving, FCFS scheduling, and preemption. We mock the model with a deterministic per-request output length so the scheduling logic is the star. The structure mirrors vLLM's scheduler closely enough to be a useful mental model.

```python
"""toy_scheduler.py — a from-scratch continuous-batching (iteration-level) scheduler.

We simulate the *control flow* of an LLM server. The "model" is mocked: each request
has a known prompt length and a known number of output tokens, and one decode iteration
produces exactly one token per running request. The interesting part is the SCHEDULER.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from enum import Enum, auto
import itertools

BLOCK_SIZE = 16          # KV-cache tokens per page/block (PagedAttention-style)
TOTAL_BLOCKS = 400       # total KV blocks on the device -> the hard memory budget
MAX_BATCHED_TOKENS = 512 # per-iteration token budget (caps prefill cost / latency)
MAX_NUM_SEQS = 64        # cap on concurrent running requests


class Status(Enum):
    WAITING = auto()     # arrived, no KV allocated yet
    RUNNING = auto()     # in the current running set, owns KV blocks
    FINISHED = auto()


@dataclass
class Request:
    rid: int
    arrival: int                 # iteration index at which it arrived
    prompt_len: int              # number of prompt tokens (prefill work)
    output_len: int              # how many tokens it will generate (mock)
    priority: int = 0            # lower = more important (for the priority policy)
    status: Status = Status.WAITING
    prefilled: bool = False      # has its prompt been processed yet?
    generated: int = 0           # output tokens produced so far
    blocks: int = 0              # KV blocks currently held
    # bookkeeping for metrics
    first_token_iter: int | None = None
    finish_iter: int | None = None

    @property
    def cur_len(self) -> int:
        # tokens currently materialized in the KV cache
        return self.prompt_len + self.generated

    def blocks_needed(self, extra_tokens: int = 0) -> int:
        # ceil division: how many blocks to hold cur_len + extra_tokens
        total = self.cur_len + extra_tokens
        return (total + BLOCK_SIZE - 1) // BLOCK_SIZE

    @property
    def done(self) -> bool:
        return self.prefilled and self.generated >= self.output_len


class Scheduler:
    def __init__(self, policy: str = "fcfs"):
        self.policy = policy
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.finished: list[Request] = []
        self.free_blocks = TOTAL_BLOCKS
        self.iter = 0
        # metrics
        self.iter_log: list[dict] = []

    # ---- block accounting -------------------------------------------------
    def _alloc(self, req: Request, want: int) -> bool:
        """Try to give `req` enough blocks to hold `want` total. Returns success."""
        need = max(0, want - req.blocks)
        if need > self.free_blocks:
            return False
        self.free_blocks -= need
        req.blocks += need
        return True

    def _free(self, req: Request):
        self.free_blocks += req.blocks
        req.blocks = 0

    # ---- policy ordering of the waiting queue -----------------------------
    def _waiting_order(self) -> list[Request]:
        reqs = list(self.waiting)
        if self.policy == "fcfs":
            return reqs                                  # already arrival-ordered
        if self.policy == "priority":
            # priority first, ties broken by arrival (no starvation within a prio band
            # because arrival is the tiebreaker)
            return sorted(reqs, key=lambda r: (r.priority, r.arrival))
        if self.policy == "shortest":
            # shortest-remaining-output first: great for mean latency, risks starvation
            return sorted(reqs, key=lambda r: r.output_len)
        raise ValueError(self.policy)

    # ---- preemption -------------------------------------------------------
    def _preempt_one(self) -> bool:
        """Evict the lowest-priority / newest running request back to WAITING.
        Returns True if something was preempted (and its blocks freed)."""
        if not self.running:
            return False
        # victim selection: in FCFS we recompute (LIFO — preempt the most-recently
        # admitted, i.e. largest arrival). In priority, preempt the worst priority.
        if self.policy == "priority":
            victim = max(self.running, key=lambda r: (r.priority, r.arrival))
        else:
            victim = max(self.running, key=lambda r: r.arrival)
        self.running.remove(victim)
        self._free(victim)                  # recompute KV on resume (recomputation)
        victim.status = Status.WAITING
        victim.prefilled = False            # we dropped its KV -> must re-prefill
        victim.generated = victim.generated # keep count of progress for the mock
        self.waiting.appendleft(victim)     # resume soon
        return True

    # ---- the heart: one scheduling decision -------------------------------
    def schedule(self) -> list[Request]:
        """Decide the running set for this iteration and how many tokens each runs."""
        token_budget = MAX_BATCHED_TOKENS

        # (1) Every already-RUNNING request needs to grow by one decode token.
        #     Ensure each can hold one more token; if a request needs a NEW block
        #     and we are out of memory, preempt to make room (or it stalls).
        survivors = []
        for req in self.running:
            want = req.blocks_needed(extra_tokens=1)
            while not self._alloc(req, want):
                if not self._preempt_one():
                    break                    # nothing left to evict; this req stalls
                # after a preemption, free_blocks grew — retry the alloc
            else:
                survivors.append(req)
                token_budget -= 1            # one decode token
                continue
            # alloc still failed even after preemptions: drop req back to waiting
            self._free(req)
            req.status = Status.WAITING
            req.prefilled = False
            self.waiting.appendleft(req)
        self.running = survivors

        # (2) Admit WAITING requests (prefill) while budget + memory + slot allow.
        for req in self._waiting_order():
            if len(self.running) >= MAX_NUM_SEQS:
                break
            # Chunked prefill: run up to `chunk` prompt tokens this iteration.
            remaining_prompt = req.prompt_len  # (mock: prefill prompt in one shot if it fits)
            chunk = min(remaining_prompt, token_budget)
            if chunk <= 0:
                break
            want = req.blocks_needed(extra_tokens=0)   # blocks for the prompt
            if not self._alloc(req, want):
                # not enough KV memory to even start this request; stop admitting
                # (could also try preemption, but FCFS usually just waits)
                continue
            req.status = Status.RUNNING
            req.prefilled = True
            self.waiting.remove(req)
            self.running.append(req)
            token_budget -= chunk

        return self.running

    # ---- mock model step + retirement -------------------------------------
    def step(self):
        batch = self.schedule()
        # MOCK MODEL: every running request emits exactly one token this iteration.
        for req in batch:
            if req.first_token_iter is None:
                req.first_token_iter = self.iter
            req.generated += 1
        # Retire finished requests immediately (frees their slot + KV NOW).
        still_running = []
        for req in batch:
            if req.done:
                req.status = Status.FINISHED
                req.finish_iter = self.iter
                self._free(req)
                self.finished.append(req)
            else:
                still_running.append(req)
        self.running = still_running
        # log batch size + memory occupancy for throughput analysis
        self.iter_log.append({
            "iter": self.iter,
            "batch": len(batch),
            "free_blocks": self.free_blocks,
            "waiting": len(self.waiting),
        })
        self.iter += 1

    def add(self, req: Request):
        self.waiting.append(req)
```

A driver that feeds requests in over time and reports utilization:

```python
def run_sim(requests: list[Request], policy: str = "fcfs", max_iters: int = 10_000):
    sched = Scheduler(policy=policy)
    pending = sorted(requests, key=lambda r: r.arrival)
    i = 0
    while sched.iter < max_iters:
        # release all requests whose arrival time has come
        while pending and pending[0].arrival <= sched.iter:
            sched.add(pending.pop(0))
        if not sched.waiting and not sched.running and not pending:
            break
        sched.step()
    return sched


if __name__ == "__main__":
    import random
    random.seed(0)
    # 200 requests, Poisson-ish arrivals, heavy-tailed output lengths.
    reqs = []
    t = 0
    for k in range(200):
        t += random.randint(0, 2)                     # interarrival 0..2 iters
        prompt = random.choice([20, 40, 80, 600])     # mix of short & long prompts
        out = random.choice([8, 16, 32, 256, 256])    # heavy tail on outputs
        reqs.append(Request(rid=k, arrival=t, prompt_len=prompt, output_len=out))

    s = run_sim([Request(**vars(r)) for r in reqs], policy="fcfs")
    total_tokens = sum(r.output_len for r in s.finished)
    iters = s.iter
    mean_batch = sum(e["batch"] for e in s.iter_log) / len(s.iter_log)
    print(f"finished={len(s.finished)} iters={iters}")
    print(f"total output tokens={total_tokens}  tokens/iter={total_tokens/iters:.1f}")
    print(f"mean running batch size={mean_batch:.1f}")
    ttfts = [r.first_token_iter - r.arrival for r in s.finished]
    print(f"mean TTFT (iters)={sum(ttfts)/len(ttfts):.2f}  max TTFT={max(ttfts)}")
```

The two things to study in this code are (a) `schedule()` re-deriving the running set *every call* — that is iteration-level scheduling in one method — and (b) the preempt-then-retry loop in step (1), which is how a server survives running out of KV memory without crashing. Swap `policy="fcfs"` for `"priority"` or `"shortest"` and watch mean TTFT and tail TTFT trade off.

!!! note "Aside: recomputation vs. swapping on preemption"
    Our toy preempts by *dropping* the victim's KV cache and re-prefilling it on resume (recomputation). Real systems offer a choice: **recompute** (cheap memory, costs a redundant prefill) or **swap** the KV blocks out to CPU/host memory and copy them back on resume (no recompute, costs PCIe bandwidth and host RAM). vLLM supports both. Recomputation usually wins when prompts are short or prefill is cheap relative to the swap copy; swapping wins for very long contexts where re-prefilling is expensive.

## Prefill/decode interleaving and its scheduling tension

Prefill and decode are different beasts. **Prefill** processes the whole prompt in parallel — it is *compute-bound* and short-lived (one big iteration). **Decode** processes one token per request — it is *memory-bound* and long-lived (many small iterations). Mixing them in one continuous-batching loop creates a real tension.

Consider a steady state of many requests happily decoding at batch size 48, throughput humming along the flat part of the curve. Now a request with a 4000-token prompt arrives. If the scheduler admits the whole prefill in the next iteration, that iteration must process 4000 prefill tokens — a heavy, compute-bound step that takes many times longer than a normal decode step. Every decoding request is *stalled* behind it; their inter-token latency (ITL) spikes. Users perceive this as the stream "stuttering." This is **prefill-induced latency interference**.

Two scheduling philosophies address it:

- **Prefill-prioritizing (decode-blocking), Orca-style / classic vLLM default.** When a prompt is ready, run its prefill as a dedicated iteration (or admit it ahead of decodes). Maximizes prefill throughput and minimizes TTFT, but causes the decode stutter above.

- **Chunked prefill / "piggybacking."** Split the long prefill into chunks of, say, 512 tokens and spread them across several iterations, *co-scheduling* each chunk alongside the ongoing decodes (the selective-batching flatten trick makes this one fused forward pass). Each iteration stays under the token budget, so decode ITL stays smooth and the prefill still completes in a few steps. This is the modern default in vLLM and SGLang and is covered in depth in [Disaggregated Prefill/Decode & Chunked Prefill](../07-inference-serving/08-disaggregated-chunked-prefill.html).

In our toy, `MAX_BATCHED_TOKENS` is exactly the knob that would enforce chunking — extend the admission loop so a request whose `remaining_prompt > token_budget` runs a *chunk* this iteration and keeps the rest for next time (track `prefilled_tokens` per request instead of a boolean). The decode tokens of running requests are admitted first (step 1), so they always get their slice of the budget; prefill chunks backfill the remainder. That ordering — decodes first, then prefill chunks — is what keeps ITL smooth.

The deepest version of this idea is **disaggregation**: run prefill on one pool of GPUs and decode on another, connected by a KV-cache transfer, so the two workloads never interfere at all. That, too, is in the disaggregation chapter; here the point is just that the *scheduler* is where prefill and decode meet, and how you interleave them is the single biggest lever on the latency/throughput trade-off.

## Priority, fairness, and preemption

FCFS (first-come-first-served) is the default and it is fine when all requests are equal. Production servers rarely have that luxury. Three concerns recur.

**Priority / multi-tenancy.** You may serve a paying "interactive" tier that needs low latency and a "batch" tier that only needs throughput. A priority scheduler orders the WAITING queue by tier and, crucially, can **preempt** a low-priority running request to admit a high-priority arrival when KV memory is full. Preemption is the teeth behind priority — without it, a queue full of low-priority long-runners can hold all the KV blocks and a high-priority request waits indefinitely. The cost is the recompute-or-swap on the victim (the aside above).

**Fairness and starvation.** Pure shortest-job-first minimizes mean latency but can starve a long request forever if short ones keep arriving. Pure priority can starve a low-priority tenant. The standard mitigations are (1) **aging** — bump a waiting request's effective priority the longer it waits, so it eventually wins; (2) **per-tenant quotas / weighted fair queueing** — guarantee each tenant a share of the running batch slots; and (3) bounding preemption — never preempt a request that has already been preempted $N$ times. Our `priority` policy avoids *intra-band* starvation by breaking ties on arrival, but does nothing for *inter-band* starvation; aging is the fix.

**Continuous-batching preemption mechanics.** Because the running set is re-derived every iteration, preemption is naturally cheap to *decide* (just don't include the victim next iteration) — the cost is purely in what you do with its KV cache. The scheduler must:

1. choose a victim (lowest priority, or most-recently-admitted for LIFO recompute-friendliness);
2. free or swap-out its KV blocks;
3. return it to WAITING (ideally at the front, so it resumes soon);
4. on resume, either recompute its prefill or swap its KV back in, then continue decoding.

A subtle but important rule: prefer to preempt the request whose progress you'll waste *least* and whose blocks you'll recover *most*. LIFO (preempt newest) tends to satisfy both for recomputation, which is why vLLM's default recompute preemption is LIFO-ish.

!!! warning "Common pitfall: thrashing under memory pressure"
    If the scheduler admits aggressively right up to the last KV block, the very next decode step (every running request wants one more token) can immediately force a preemption — which frees blocks, which tempts the scheduler to re-admit, which fills memory again, which forces another preemption. The system *thrashes*, spending its time swapping/recomputing instead of generating. Defenses: keep a small **headroom** of free blocks (admit only up to, say, 90% occupancy), use a **watermark** so you stop admitting before memory is full, and add **hysteresis** so you don't re-admit a just-preempted request until enough blocks are free to make real progress. This is the inference analogue of OS page-fault thrashing.

!!! example "Worked example: the continuous-batching throughput multiplier"
    Take a decode-bound workload on one GPU. Suppose the device can hold a maximum running batch of $B_{\max} = 64$ requests within its KV-cache budget, a decode step takes $\tau_0 = 20$ ms of fixed weight-streaming cost plus a marginal $\beta = 0.2$ ms per request, and output lengths are heavy-tailed with mean 100 and max 1000 tokens.

    **Static batching.** Form a batch of 64, run until the last finishes. The batch occupies its slots for $\max = 1000$ steps, but the *useful* work is $64 \times \overline{\text{len}} = 64 \times 100 = 6400$ tokens. Average batch size over the run is roughly $\overline{\text{len}}/\max \times 64 \approx 0.1 \times 64 \approx 6.4$ (slots drain as short requests finish). Effective throughput sits at the *low* end of the curve much of the time, and a new request can wait up to 1000 decode steps to be admitted — TTFT on the order of $1000 \times \tau_{\text{dec}} \approx 1000 \times 21\,\text{ms} \approx 21$ s in the worst case.

    **Continuous batching.** Finished slots are backfilled every iteration, so the running batch stays pinned near $B_{\max} = 64$. Step time $\tau_{\text{dec}}(64) = 20 + 0.2 \times 64 = 32.8$ ms, giving

    $$
    \text{throughput} = \frac{64}{32.8\ \text{ms}} \approx 1951\ \text{tokens/sec},
    $$

    versus static batching effectively operating near $B \approx 6.4$:

    $$
    \frac{6.4}{20 + 0.2 \times 6.4\ \text{ms}} = \frac{6.4}{21.3\ \text{ms}} \approx 300\ \text{tokens/sec}.
    $$

    That is a **~6–7× throughput improvement** purely from scheduling — no kernel changes, no quantization. And a newly arrived request is admitted within ~1 iteration once a slot frees, so TTFT drops from seconds to tens of milliseconds. The exact multiplier depends on how heavy-tailed your output lengths are: the longer the tail (the bigger $\max/\text{mean}$), the worse static batching's tail idling, and the larger the win. Published reports of "up to 23×" come from workloads with extreme length variance; treat the *mechanism* (pinning $B$ near $B_{\max}$ continuously) as the durable truth and the exact number as workload-dependent.

## How the real systems do it

Every production engine is a continuous-batching scheduler with different emphases:

- **Orca** introduced iteration-level scheduling and selective batching — the blueprint.
- **vLLM** pairs continuous batching with **PagedAttention** so the KV cache is non-contiguous blocks, making admission/eviction a block-allocation problem and enabling near-zero memory fragmentation; its scheduler does FCFS with preemption (recompute or swap) and now defaults to chunked prefill. See [vLLM: Architecture, PagedAttention & Internals](../07-inference-serving/03-vllm-internals.html).
- **TGI** (HuggingFace Text Generation Inference) calls it *continuous batching* and exposes a `waiting_served_ratio` and token-budget knobs.
- **TensorRT-LLM** calls it **in-flight batching** and fuses it with NVIDIA's optimized kernels.
- **SGLang** adds **RadixAttention** for prefix-cache reuse on top of continuous batching, so shared prompt prefixes don't re-prefill — see [SGLang: RadixAttention & Structured Programs](../07-inference-serving/04-sglang-radixattention.html).

The common skeleton is identical to our toy: a per-iteration `schedule()` that (1) reserves a decode token + block for each running request (preempting under pressure), (2) admits/chunks prefill into the remaining token budget, (3) runs one fused forward pass with ragged attention, and (4) retires finished requests immediately. Master the toy and you understand all of them. The end-to-end serving design that wraps this scheduler — admission control, autoscaling, routing — is the subject of [Designing an LLM Serving System](../12-production-mlops/01-serving-system-design.html), and the latency/throughput/cost trade-offs it implies are quantified in [Inference Economics: Latency, Throughput & Cost](../07-inference-serving/12-inference-economics.html).

!!! interview "Interview Corner"
    **Q:** A teammate says "we already use dynamic batching, so we get continuous batching's benefits." Are they right? Explain the difference and where dynamic batching still leaves throughput on the table for an autoregressive LLM.

    **A:** No — they're conflating two different things. *Dynamic batching* waits a short window to gather arriving requests, then dispatches them as one batch whose membership is **frozen for the whole generation**. That helps amortize weight loads at *admission* time, and it's the right tool for fixed-shape models (one forward pass each). But LLM generation is iterative and variable-length: each request runs a *different number of forward passes*. So dynamic batching still suffers two leaks. First, **tail idling**: short requests finish early but their slots stay locked until the longest request in the batch completes, dragging the effective batch size — and thus throughput — way down at the tail. Second, **head-of-line blocking at admission**: a request arriving just after a batch forms must wait for the entire batch to drain before it's even admitted, spiking TTFT to seconds under long generations. *Continuous (iteration-level) batching*, from Orca, re-derives the running set **every forward pass**: it retires finished requests and backfills new ones each iteration, pinning the batch near its memory-limited maximum continuously and admitting newcomers within ~one iteration. The mechanism that makes mixing prefill and decode in one step possible is *selective batching* — flatten all tokens for the position-independent GEMMs (QKV/MLP) and run attention per-request via ragged/`cu_seqlens` kernels. Net effect: typically several-fold higher throughput and an order-of-magnitude lower TTFT versus dynamic batching, with the multiplier growing as output-length variance grows.

!!! key "Key Takeaways"
    - **Decode is memory-bound; batching amortizes the weight load.** Throughput rises with batch size $B$ as $B/(\tau_0 + \beta B)$ and flattens toward the compute roofline $1/\beta$. The scheduler's job is to keep $B$ pinned near its memory-limited maximum.
    - **Static and dynamic (request-level) batching freeze batch membership for the whole generation**, leaking throughput to tail idling (short requests' slots locked until the longest finishes) and leaking latency to head-of-line blocking at admission.
    - **Continuous batching = iteration-level scheduling (Orca).** Re-derive the running set every forward pass: retire finished requests and backfill waiting ones immediately. This removes both leaks and is the basis of vLLM, TGI, TensorRT-LLM, and SGLang.
    - **Selective batching makes heterogeneous batches possible:** flatten all tokens for position-independent GEMMs (QKV, MLP), and run attention per-request via ragged/`cu_seqlens` kernels. This is also what enables chunked prefill.
    - **The two binding constraints are KV-cache memory (track free blocks; PagedAttention makes it block allocation) and the per-iteration token budget** (`max_num_batched_tokens`) that bounds latency.
    - **Prefill and decode interfere at the scheduler.** A big prefill stalls ongoing decodes; chunked prefill (and ultimately disaggregation) smooths inter-token latency by spreading prefill across iterations under the token budget — schedule decodes first, then backfill prefill chunks.
    - **Preemption is the teeth behind priority and the safety valve under memory pressure** — evict a victim's KV (recompute or swap-out), return it to WAITING, resume later. Guard against thrashing with watermarks, headroom, and hysteresis; guard against starvation with aging or per-tenant quotas.
    - **The throughput multiplier is workload-dependent:** the heavier the output-length tail, the worse static batching's idling and the larger continuous batching's win (commonly several-fold, more under extreme variance).

!!! sota "State of the Art & Resources (2026)"
    Continuous batching (iteration-level scheduling) is now standard in every major LLM serving engine. Research has moved on to chunked prefill, disaggregated prefill/decode, and KV-cache-aware scheduling as the frontiers for squeezing the last latency and throughput from inference infrastructure.

    **Foundational work**

    - [Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models* (OSDI 2022)](https://www.usenix.org/conference/osdi22/presentation/yu) — the paper that named iteration-level scheduling and selective batching; everything since builds on this blueprint.
    - [Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (SOSP 2023)](https://arxiv.org/abs/2309.06180) — vLLM; pairs continuous batching with OS-paging-inspired KV-cache management for 2–4× further throughput gains.

    **Recent advances (2023–2026)**

    - [Agrawal et al., *SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills* (2023)](https://arxiv.org/abs/2308.16369) — original chunked-prefill proposal; shows that splitting prefill across iterations eliminates decode stalls.
    - [Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve* (OSDI 2024)](https://arxiv.org/abs/2403.02310) — full system integrating stall-free scheduling + chunked prefill; up to 6.9× throughput improvement over vLLM on large models.
    - [Zhong et al., *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving* (2024)](https://arxiv.org/abs/2401.09670) — assigns prefill and decode to separate GPU pools, eliminating interference entirely; the production direction adopted by Meta, NVIDIA Dynamo, and others.
    - [Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs* (NeurIPS 2024)](https://arxiv.org/abs/2312.07104) — adds RadixAttention (prefix-cache reuse via radix tree) on top of continuous batching, cutting redundant prefill for shared-prefix workloads.

    **Open-source & tools**

    - [vllm-project/vllm](https://github.com/vllm-project/vllm) — the reference continuous-batching serving engine; PagedAttention, chunked prefill, and preemption all in one codebase; read the `Scheduler` class directly.
    - [sgl-project/sglang](https://github.com/sgl-project/sglang) — SGLang's high-performance runtime with RadixAttention prefix caching; strong performance on multi-turn and structured-output workloads.

    **Go deeper**

    - [Anyscale, *How continuous batching enables 23× throughput in LLM inference* (2023)](https://www.anyscale.com/blog/continuous-batching-llm-inference) — concise engineering explainer with benchmarks; the widely-cited piece that brought the idea to a broad ML audience.
    - [LMSYS, *Fast and Expressive LLM Inference with RadixAttention and SGLang* (2024)](https://www.lmsys.org/blog/2024-01-17-sglang/) — visual walkthrough of how a radix tree accelerates KV-cache reuse in a continuous-batching server.

## Further reading

- Yu, Jeong, Kim, Kim, Chun, *Orca: A Distributed Serving System for Transformer-Based Generative Models* (OSDI 2022) — the paper that introduced iteration-level scheduling and selective batching.
- Kwon, Li, Zhuang, et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (SOSP 2023) — vLLM; continuous batching plus paged KV-cache.
- Agrawal et al., *SARATHI* / *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve* — chunked prefill and stall-free batching.
- Anyscale, *How Continuous Batching Enables 23× Throughput in LLM Inference* — a widely cited engineering write-up of the mechanism and its measured impact.
- The vLLM and Hugging Face Text Generation Inference (TGI) repositories — read the `Scheduler` / batching code directly; it mirrors the toy in this chapter.
