# %% [markdown]
# # torch.compile & CUDA Graphs: Measuring Real Speedups
#
# > **Hardware:** 1x H100 80GB. Runtime: several minutes (the `max-autotune`
# > compile/recompile passes dominate). Not executed in the book — run it to get
# > your own numbers.
#
# You will measure eager vs. `torch.compile` forward+backward latency on a small
# transformer block, inspect the graph-break/fusion story with `torch._dynamo.explain`,
# compare static vs. dynamic shapes, and capture a CUDA graph for a fixed-shape decode step.
#
# See [the chapter](https://prakashkagitha.github.io/llm-stack-book/04-kernels-efficiency/09-compilers-fusion.html) for the full explanation.

# %%
# The H100 image ships a recent CUDA-enabled PyTorch (>=2.3), which is all this
# notebook needs: torch.compile / TorchInductor and CUDA graphs are core torch, so
# there are no extra dependencies. The line below is a near no-op when a satisfying
# torch is already installed. We deliberately do NOT force an --upgrade: swapping the
# torch build at runtime can pull a wheel mismatched to the CUDA driver and would only
# take effect after a kernel restart.
%pip install -q "torch>=2.3"

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

assert torch.cuda.is_available(), "This notebook requires a CUDA GPU (targets 1x H100 80GB)."
device = torch.device("cuda")
dtype = torch.bfloat16  # H100 has fast native bf16 tensor cores; use it everywhere below

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# Modest but realistic compile-relevant knobs. TF32 matmuls are fine for the parts of
# this notebook that stay in eager fp32 (none here, but harmless to set).
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

props = torch.cuda.get_device_properties(0)
print(f"GPU: {props.name}, {props.total_memory / 1e9:.1f} GB, sm_{props.major}{props.minor}")
print(f"torch version: {torch.__version__}")

# %% [markdown]
# ## 1. A small GPT-style block
#
# We use a minimal pre-norm transformer block (LayerNorm -> causal self-attention via
# `F.scaled_dot_product_attention` -> LayerNorm -> MLP with GELU) — the same shape of
# computation as one decoder layer in a GPT model. It is intentionally *small* (many
# cheap ops relative to the two big matmuls) because that is exactly the regime where
# `torch.compile`'s kernel fusion has the most to fuse away: LayerNorm, bias-adds,
# residual adds, and GELU are all memory-bound elementwise/reduction ops that eager
# mode launches as separate kernels.

# %%
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)  # [B, T, 3C]
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T, Hd]
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # SDPA dispatches to a fused (flash-style) attention kernel on H100 in bf16.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class GPTBlock(nn.Module):
    """One pre-norm transformer decoder block: attn + MLP, each with a residual add."""

    def __init__(self, d_model=1024, n_heads=16, mlp_ratio=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model, bias=True),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model, bias=True),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


D_MODEL, N_HEADS = 1024, 16
BATCH, SEQ = 8, 1024

model_eager = GPTBlock(D_MODEL, N_HEADS).to(device=device, dtype=dtype)
print(model_eager)
n_params = sum(p.numel() for p in model_eager.parameters())
print(f"params: {n_params / 1e6:.2f}M")

# %% [markdown]
# ## 2. Latency helper: `torch.cuda.Event`, not `time.time()`
#
# GPU kernels launch asynchronously, so wall-clock timers on the CPU either measure
# launch overhead (too optimistic) or force a sync that hides pipelining (too
# pessimistic and non-representative). `torch.cuda.Event(enable_timing=True)` records
# timestamps on the GPU's own stream, and `torch.cuda.synchronize()` before/after
# ensures we are not timing a half-finished queue.

# %%
def timed_forward_backward(model, x, n_iters=30, n_warmup=10):
    """Return (mean_ms, peak_mem_bytes) for forward+backward over n_iters, after warmup.

    Warmup matters doubly here: (a) cuDNN/cuBLAS algorithm selection and CUDA context
    setup have one-time costs, and (b) torch.compile is *lazy* — the first call(s)
    trigger tracing + Triton kernel codegen (and, in max-autotune, a search over
    candidate kernels), which can take seconds to minutes and must never be counted
    as steady-state latency.
    """
    target = torch.zeros_like(x)

    for _ in range(n_warmup):
        model.zero_grad(set_to_none=True)
        out = model(x)
        loss = F.mse_loss(out, target)
        loss.backward()
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    # Reset the peak-memory counter AFTER warmup so we report steady-state working set.
    torch.cuda.reset_peak_memory_stats()
    start_evt.record()
    for _ in range(n_iters):
        model.zero_grad(set_to_none=True)
        out = model(x)
        loss = F.mse_loss(out, target)
        loss.backward()
    end_evt.record()
    torch.cuda.synchronize()

    mean_ms = start_evt.elapsed_time(end_evt) / n_iters
    peak_mem = torch.cuda.max_memory_allocated()
    return mean_ms, peak_mem


x = torch.randn(BATCH, SEQ, D_MODEL, device=device, dtype=dtype, requires_grad=True)

eager_ms, eager_mem = timed_forward_backward(model_eager, x)
print(f"eager:            {eager_ms:.3f} ms/iter, peak mem {eager_mem / 1e9:.2f} GB")

# %% [markdown]
# ## 3. Compile with `mode='max-autotune'` and re-measure
#
# `torch.compile` wraps the module: TorchDynamo traces the forward (and, via
# AOTAutograd, the backward) into an FX graph, and TorchInductor lowers it to fused
# Triton kernels. `mode='max-autotune'` searches over multiple kernel configurations
# (tile sizes, fusion choices) per shape — that search happens once, lazily, on the
# first call with a given input shape, and is exactly what the warmup above is for.

# %%
model_compiled = GPTBlock(D_MODEL, N_HEADS).to(device=device, dtype=dtype)
model_compiled.load_state_dict(model_eager.state_dict())  # identical weights, fair comparison
model_compiled = torch.compile(model_compiled, mode="max-autotune")

compiled_ms, compiled_mem = timed_forward_backward(model_compiled, x)
print(f"torch.compile:    {compiled_ms:.3f} ms/iter, peak mem {compiled_mem / 1e9:.2f} GB")
print(f"speedup:          {eager_ms / compiled_ms:.2f}x")

# %% [markdown]
# **What to expect:** on an H100 in bf16, this kind of block (small-ish d_model, many
# elementwise/reduction ops between two large matmuls) typically compiles to roughly
# **1.2-2x** faster forward+backward than eager, mostly from fusing LayerNorm +
# bias-add + residual-add + GELU into fewer kernel launches and fewer DRAM round
# trips — the big `nn.Linear` matmuls and the SDPA attention kernel were already
# near-optimal in eager mode, so compile helps *around* them, not *to* them. Peak
# memory is usually similar or slightly lower under compile (fused kernels need fewer
# scratch buffers); it should not be dramatically higher. Treat the exact multiplier
# as something you measure — it depends on shape, torch version, and autotune cache.
#
# ## 4. Where did the time go? Graph breaks and fusion
#
# `torch._dynamo.explain` re-traces the model and reports how many separate FX graphs
# TorchDynamo produced and why it broke between them. Zero graph breaks means the
# *entire* forward became one graph Inductor could optimize as a whole; each break is
# a fallback to eager for that segment, which caps how much can be fused across it.
# `TORCH_LOGS=graph_breaks` (set as an env var before starting Python) gives the same
# information as a running log instead of a one-shot report — useful when the break
# happens deep inside a training loop you can't easily wrap in `explain`.

# %%
# torch._dynamo.explain(fn) returns a callable; call it with the inputs to get an
# ExplainOutput dataclass (fields: graph_count, graph_break_count, op_count,
# break_reasons, ...). This is the modern (torch >= 2.1) API form.
explanation = torch._dynamo.explain(model_eager)(x)
print(f"graph breaks:        {explanation.graph_break_count}")
print(f"graph count:         {explanation.graph_count}")
print(f"ops captured:        {explanation.op_count}")
# Uncomment to see the full per-break reasons (module names, source lines, and cause):
# print(explanation)

# %% [markdown]
# For this block you should see **0 (or very few) graph breaks** — it is pure tensor
# ops with no `.item()` calls, no shape-dependent Python `if`s, and no `print`s in the
# hot path, so Dynamo can trace it as a single FX graph. Real GPT models often show a
# handful of breaks around things like KV-cache index bookkeeping, sampling logic
# (`.item()`/`.tolist()` to pull a token id back to Python), or logging — each one
# is worth finding and removing if the loop is latency-critical, since it partitions
# fusion opportunities on either side of it.
#
# ## 5. Static vs. dynamic shapes: recompilation cost
#
# By default `torch.compile` specializes on the exact input shape it first sees, and
# **recompiles** if a new shape shows up (e.g. a shorter final batch, or growing
# sequence length during generation). `dynamic=True` tells Dynamo to trace with
# symbolic shapes up front, trading a bit of per-step overhead for far fewer
# recompilations when shapes genuinely vary. We demonstrate the recompile cost
# directly: call the *static*-mode compiled model with a new sequence length and time
# just that first call vs. a repeat call at the same new shape.

# %%
model_dynamic = GPTBlock(D_MODEL, N_HEADS).to(device=device, dtype=dtype)
model_dynamic.load_state_dict(model_eager.state_dict())
model_dynamic = torch.compile(model_dynamic, dynamic=True)

x_512 = torch.randn(BATCH, 512, D_MODEL, device=device, dtype=dtype, requires_grad=True)


def one_shot_fwd_bwd_ms(model, inp):
    """Time a single forward+backward, including any first-call (re)compilation.

    CUDA events measure the GPU timeline: the host-side tracing/autotune stall shows up
    as GPU idle time between the two recorded events, so the first-call number here
    genuinely includes compilation wall time.
    """
    torch.cuda.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    model.zero_grad(set_to_none=True)
    out = model(inp)
    F.mse_loss(out, torch.zeros_like(out)).backward()
    end_evt.record()
    torch.cuda.synchronize()
    return start_evt.elapsed_time(end_evt)

# Static-mode model (model_compiled) was warmed up at SEQ=1024 in section 3.
# Feeding it SEQ=512 now forces a fresh specialization/recompile at the new shape.
first_call_ms = one_shot_fwd_bwd_ms(model_compiled, x_512)
second_call_ms = one_shot_fwd_bwd_ms(model_compiled, x_512)
print(f"static-shape model, new seq_len first call:  {first_call_ms:.1f} ms (includes recompile)")
print(f"static-shape model, same seq_len 2nd call:    {second_call_ms:.1f} ms (steady state)")

# The dynamic=True model traces once with symbolic shapes and should not pay a large
# recompile penalty when seq_len changes again after its own first (warmup) call.
_ = one_shot_fwd_bwd_ms(model_dynamic, x)       # warmup / initial symbolic trace at SEQ
dyn_new_shape_ms = one_shot_fwd_bwd_ms(model_dynamic, x_512)  # new seq_len, no full recompile expected
print(f"dynamic=True model, new seq_len call:          {dyn_new_shape_ms:.1f} ms")

# %% [markdown]
# **What to expect:** the static-shape model's first call at the new sequence length
# should take much longer than its own steady-state number — recompilation can add
# anywhere from tens of milliseconds to many seconds, and a full `max-autotune` search
# at a fresh shape can take longer still — while its second call at the *same* new
# shape drops back to a fast, compiled steady state (a fresh specialization was
# cached). The `dynamic=True` model should handle the shape change with little or no
# extra latency, at the cost of usually being a bit slower than a fully
# shape-specialized compile at any single fixed shape. Use static shapes (padding to a
# fixed bucket, if needed) whenever the shape truly is fixed; use `dynamic=True` when
# shapes genuinely vary, e.g. prompt lengths.
#
# ## 6. CUDA graph capture for a fixed-shape decode step
#
# Autoregressive decoding runs the *same* module on the *same* shape (batch, 1 new
# token) thousands of times, and at batch size 1 the actual kernels are tiny — most
# of the wall-clock time is CPU-side kernel-launch overhead (very roughly a handful to
# a few tens of microseconds per launch), not GPU compute. A CUDA graph records the
# whole sequence of kernel launches for one step once, then replays it with a single
# `cudaGraphLaunch`, eliminating that per-kernel CPU overhead. The hard constraint:
# every replay must reuse the *exact same* input/output memory addresses, so we
# allocate static buffers, copy new data into them each step, and treat the graph as
# consuming whatever is in those buffers at replay time.

# %%
decode_model = GPTBlock(D_MODEL, N_HEADS).to(device=device, dtype=dtype).eval()
BATCH_DECODE = 1
static_x = torch.randn(BATCH_DECODE, 1, D_MODEL, device=device, dtype=dtype)

# Warmup on a side stream before capture: CUDA graph capture requires the workload to
# have already gone through cuDNN/cuBLAS algorithm selection etc. on a non-default
# stream, per the documented torch.cuda.graph capture recipe.
warmup_stream = torch.cuda.Stream()
warmup_stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(warmup_stream), torch.no_grad():
    for _ in range(5):
        _ = decode_model(static_x)
torch.cuda.current_stream().wait_stream(warmup_stream)
torch.cuda.synchronize()

graph = torch.cuda.CUDAGraph()
static_out = None
with torch.no_grad(), torch.cuda.graph(graph):
    static_out = decode_model(static_x)  # captured once; not actually run yet

def replay_step(new_token_embed):
    """Copy new data into the static input buffer, replay the captured graph,
    and read the result out of the static output buffer (in-place semantics)."""
    static_x.copy_(new_token_embed)
    graph.replay()
    return static_out

# %% [markdown]
# Now compare eager decode-step latency against the CUDA-graph replay latency, both
# at the same fixed (batch=1, seq=1) shape — this is the single-token decode step you
# would call once per generated token in an inference server.

# %%
def timed_decode(step_fn, n_iters=200, n_warmup=20):
    dummy = torch.randn(BATCH_DECODE, 1, D_MODEL, device=device, dtype=dtype)
    for _ in range(n_warmup):
        step_fn(dummy)
    torch.cuda.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(n_iters):
        step_fn(dummy)
    end_evt.record()
    torch.cuda.synchronize()
    return start_evt.elapsed_time(end_evt) / n_iters


@torch.no_grad()
def eager_decode_step(tok):
    return decode_model(tok)

eager_decode_ms = timed_decode(eager_decode_step)
graph_decode_ms = timed_decode(replay_step)
print(f"eager decode step:       {eager_decode_ms * 1000:.1f} us")
print(f"CUDA-graph decode step:  {graph_decode_ms * 1000:.1f} us")
print(f"speedup:                 {eager_decode_ms / graph_decode_ms:.2f}x")

# %% [markdown]
# **What to expect:** at batch=1 with this small block, the CUDA-graph replay should
# be noticeably faster in wall-clock terms than the eager step — often on the order
# of a similar **1.2-2x** (sometimes more at even smaller batch sizes or shorter
# blocks, since launch overhead is a larger fraction of a smaller total). The gain
# comes entirely from removing CPU dispatch overhead, so it is largest exactly when
# GPU compute per step is small — i.e. small batch, short sequence, few layers per
# kernel-launch-bound micro-op. This is also precisely what `mode='reduce-overhead'`
# in `torch.compile` automates for you (compile + auto CUDA-graph capture together)
# when your shapes are fixed.

# %% [markdown]
# ## What you should see
#
# - **Eager vs. `torch.compile(mode='max-autotune')` forward+backward:** roughly a
#   **1.2-2x** speedup for a block like this — most of the gain is fusing the many
#   small LayerNorm/bias-add/residual/GELU ops around the two big matmuls, not
#   speeding up the matmuls or attention themselves (those were already near-optimal
#   via cuBLAS/SDPA in eager mode). The exact multiplier is measure-it territory.
# - **Graph breaks:** this clean block should trace as ~0 graph breaks; real
#   generation loops often pick up a few around `.item()`/sampling/logging, and each
#   one caps fusion on either side of it — `torch._dynamo.explain` and
#   `TORCH_LOGS=graph_breaks` are how you find them.
# - **Static vs. dynamic shapes:** a shape-specialized (static) compile pays a real
#   recompilation cost (tens of ms to several seconds, more under max-autotune) the
#   first time a new shape appears, then is fast at that shape; `dynamic=True` avoids
#   most recompilation across shape changes at some cost to peak per-shape performance
#   — pick based on whether your shapes are truly fixed (padding/bucketing) or
#   genuinely variable.
# - **CUDA graphs for decode:** at small batch sizes, kernel-launch overhead (a handful
#   to a few tens of microseconds per launch on the CPU side) dominates wall-clock
#   time; capturing and replaying a CUDA graph for the fixed-shape decode step removes
#   that overhead and typically yields another **~1.2-2x**, which is why serving
#   frameworks capture one graph per fixed (batch, kv-length-bucket) shape.
#
# **Next step:** see the FlashAttention/Triton-kernels notebook for what happens
# *inside* the fused attention kernel these speedups are riding on top of, and the
# chapter's coverage of `mode='reduce-overhead'`, TensorRT-LLM, and TVM for how other
# compiler stacks approach the same fusion/graph-capture tradeoffs.
