"""CI-tested extracts of runnable code blocks from
content/06-rl-infra/04-verl.md

Each `block_*` function reproduces the book's ACTUAL code verbatim (modulo
wrapping in a function and small, honestly-labeled glue/fixtures) and then
asserts the claims the prose/comments make about the results. Run directly:
`python3 tests/06-rl-infra__04-verl.py`

Blocks tested:
  - block #0 (line ~17): the single-controller pseudo-code driver loop.
    Executed against tiny mock workers + a real GAE implementation as glue,
    since the book leaves `rollout_worker`, `compute_gae`, etc. undefined
    (it is illustrative pseudo-code by design).
  - block #2 (line ~114): the miniature reconstruction of veRL's dispatch
    mechanism (`register`, `DP_COMPUTE_PROTO` dispatch/collect,
    `WorkerGroupProxy`). Executed against a fake in-process Ray shim (no
    real `ray`, no network/cluster) and a toy `WorkerClass` with one
    `@register`-decorated method.

Blocks skipped (see task instructions for this chapter):
  - block #1 (`fit()` driver method, line ~64): non-standalone fragment —
    a method body referencing `self.actor_rollout_wg`, `self.ref_policy_wg`,
    etc. that are never defined in the snippet (they'd be real veRL
    WorkerGroup objects backed by Ray/FSDP/vLLM).
  - block #3 (`reshard_column_parallel`, line ~204): needs a real
    `torch.distributed` process group (and normally GPUs) to exercise
    `dist.all_gather`.
  - block #4 (YAML resource/placement config, line ~278): not Python.
  - block #5 (`math_verifiable_reward` / `grpo_group_advantage`, line
    ~340): calls into an undefined `is_math_equiv` and expects a real
    HuggingFace tokenizer; not meaningfully CPU-runnable as a standalone
    unit without inventing substantial logic the chapter doesn't supply.

Real bug found & fixed in the book's block #2 (`WorkerGroupProxy.call`):
the list comprehension destructured each per-rank args-tuple with the
pattern `for w, (a,) in zip(...)` and then called `w...remote(*a)`. Since
`dispatch_dp_compute_proto` appends `(chunks[dp_rank],)` (a 1-tuple whose
sole element IS the args-tuple for that call), destructuring with `(a,)`
unwraps ONE level too many, leaving `a` bound to the raw `DataProto` chunk
itself rather than to the args-tuple `(chunks[dp_rank],)`. `.remote(*a)`
then tries to unpack a bare `DataProto` as if it were an iterable of
positional arguments, raising `TypeError: argument after * must be an
iterable, not DataProto`. The fix (mirrored in content/06-rl-infra/04-verl.md
and below) is `for w, a in zip(self.workers, per_rank_args)` — treat each
`per_rank_args` entry as the whole args-tuple and let `.remote(*a)` do the
one, correct level of unpacking.
"""

import functools

import numpy as np


# ============================================================================
# block #0 (line ~17): single-controller pseudo-code driver loop.
# ============================================================================

def block_single_controller_loop():
    # content lines ~17-26

    # --- glue: the book leaves these undefined since the snippet is
    # illustrative pseudo-code ("Single-controller pseudo-code: the WHOLE
    # algorithm in one readable loop."). We supply a real (if toy) GAE
    # implementation and tiny mock "remote" workers so the loop actually
    # executes end-to-end on CPU. ---
    def compute_gae(rewards, values, gamma=0.99, lam=0.95):
        rewards = np.asarray(rewards, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        T = len(rewards)
        advantages = np.zeros(T)
        last_gae = 0.0
        next_value = 0.0
        for t in reversed(range(T)):
            delta = rewards[t] + gamma * next_value - values[t]
            last_gae = delta + gamma * lam * last_gae
            advantages[t] = last_gae
            next_value = values[t]
        returns = advantages + values
        return advantages, returns

    class MockRolloutWorker:
        def generate(self, batch):
            return [f"response-to-{p}" for p in batch]

    class MockRewardWorker:
        def score(self, batch, responses):
            return np.array([float(len(r)) for r in responses])

    class MockCriticWorker:
        def value(self, batch, responses):
            return np.array([0.5 * len(r) for r in responses])

        def update(self, batch, responses, returns):
            self.last_returns = returns

    class MockActorWorker:
        def update(self, batch, responses, advantages):
            self.last_advantages = advantages

    prompts = [["prompt-a", "prompt-bb"], ["prompt-ccc"]]
    rollout_worker = MockRolloutWorker()
    reward_worker = MockRewardWorker()
    critic_worker = MockCriticWorker()
    actor_worker = MockActorWorker()

    # --- verbatim book code (content lines ~17-26) ---
    # Single-controller pseudo-code: the WHOLE algorithm in one readable loop.
    for batch in prompts:
        responses = rollout_worker.generate(batch)          # remote call
        rewards   = reward_worker.score(batch, responses)   # remote call
        values    = critic_worker.value(batch, responses)   # remote call
        advantages, returns = compute_gae(rewards, values)  # LOCAL on the driver
        actor_worker.update(batch, responses, advantages)   # remote call
        critic_worker.update(batch, responses, returns)     # remote call
    # --- end verbatim book code ---

    # --- verify the loop actually ran, end to end, over every batch ---
    assert actor_worker.last_advantages.shape == (len(prompts[-1]),)
    assert critic_worker.last_returns.shape == (len(prompts[-1]),)
    assert np.all(np.isfinite(actor_worker.last_advantages))
    print(f"[block #0] driver loop ran over {len(prompts)} batches; "
          f"final advantages={actor_worker.last_advantages}")


# ============================================================================
# block #2 (line ~114): miniature reconstruction of veRL's dispatch mechanism.
# ============================================================================

def block_dispatch_mechanism():
    # content lines ~114-171

    # --- glue: minimal stand-in for veRL's DataProto (only `chunk` for
    # dispatch and `concat` for collect are needed, exactly what the block
    # uses) ---
    class DataProto:
        def __init__(self, values):
            self.values = list(values)

        def chunk(self, n):
            size = len(self.values) // n
            assert size * n == len(self.values), "toy batch must divide evenly"
            return [DataProto(self.values[i * size:(i + 1) * size]) for i in range(n)]

        @staticmethod
        def concat(parts):
            merged = []
            for p in parts:
                merged.extend(p.values)
            return DataProto(merged)

        def __repr__(self):
            return f"DataProto({self.values!r})"

    # --- verbatim book code (content lines ~114-171, with the one-line fix
    # to `WorkerGroupProxy.call` described in the module docstring / mirrored
    # in content/06-rl-infra/04-verl.md) ---
    # A miniature reconstruction of veRL's dispatch mechanism. The real one is more
    # careful about padding, async futures, and TP/PP replication, but this captures
    # the single-controller -> multi-controller fan-out/fan-in exactly.

    # Dispatch functions: given a WorkerGroup and the call's args, return a LIST of
    # (args, kwargs) — one entry per rank.
    def dispatch_dp_compute_proto(worker_group, batch):
        dp_size = worker_group.dp_size            # number of data-parallel groups
        tp_size = worker_group.tp_size            # ranks per DP group (TP * PP)
        chunks = batch.chunk(dp_size)             # split the BATCH across DP groups only
        per_rank = []
        for dp_rank in range(dp_size):
            for _ in range(tp_size):              # replicate the chunk to every TP rank
                per_rank.append((chunks[dp_rank],))   # in this DP group
        return per_rank                            # length == world_size

    # Collect functions: given the list of per-rank outputs, fold them into one result.
    def collect_dp_compute_proto(worker_group, outputs):
        dp_size = worker_group.dp_size
        tp_size = worker_group.tp_size
        # Keep only ONE representative per DP group (TP ranks computed identical batch
        # outputs); concatenate across DP groups to reconstruct the full batch order.
        reps = [outputs[dp_rank * tp_size] for dp_rank in range(dp_size)]
        return DataProto.concat(reps)

    DISPATCH = {
        "DP_COMPUTE_PROTO": (dispatch_dp_compute_proto, collect_dp_compute_proto),
    }

    def register(dispatch):
        """Decorator placed on Worker methods. Records the dispatch mode so the
        WorkerGroup proxy knows how to fan out / fan in when the DRIVER calls it."""
        def decorator(fn):
            fn._dispatch_mode = dispatch
            @functools.wraps(fn)
            def inner(self, *args, **kwargs):     # runs ON the worker (one rank)
                return fn(self, *args, **kwargs)
            return inner
        return decorator

    class WorkerGroupProxy:
        """Lives on the DRIVER. `self.workers` are Ray actor handles (one per rank)."""
        def __init__(self, workers, dp_size, tp_size):
            self.workers, self.dp_size, self.tp_size = workers, dp_size, tp_size

        def call(self, method_name, batch):
            dispatch_fn, collect_fn = DISPATCH[
                getattr(WorkerClass, method_name)._dispatch_mode]
            per_rank_args = dispatch_fn(self, batch)            # split + replicate
            # Launch the SAME method on every rank in parallel (Ray remote calls).
            # FIX (see module docstring): this was `for w, (a,) in zip(...)`, which
            # double-unwrapped the per-rank args-tuple and crashed `.remote(*a)`.
            futures = [w.__getattr__(method_name).remote(*a)
                       for w, a in zip(self.workers, per_rank_args)]
            outputs = ray.get(futures)                          # gather all ranks
            return collect_fn(self, outputs)                    # fold into one DataProto
    # --- end verbatim book code ---

    # --- glue: a toy WorkerClass with one @register-decorated method (the
    # book's WorkerGroupProxy.call() looks this class up by name to find
    # the dispatch mode — `getattr(WorkerClass, method_name)._dispatch_mode`
    # — so it must exist), plus a fake in-process "Ray" so no real Ray
    # cluster / network is involved. Ray actor handles define a custom
    # `__getattr__` that returns a `.remote(...)`-callable proxy; we mimic
    # that minimally and execute the "remote" call synchronously in-process. ---
    class WorkerClass:
        def __init__(self, rank):
            self.rank = rank

        @register(dispatch="DP_COMPUTE_PROTO")
        def generate_sequences(self, batch):
            return DataProto([f"rank{self.rank}:{v}" for v in batch.values])

    class _FakeRemoteMethod:
        def __init__(self, bound_method):
            self._bound_method = bound_method

        def remote(self, *args, **kwargs):
            # Executes synchronously in-process; stands in for a Ray future.
            return self._bound_method(*args, **kwargs)

    class _FakeActorHandle:
        def __init__(self, worker_instance):
            self._worker = worker_instance

        def __getattr__(self, name):
            bound = getattr(self._worker, name)
            return _FakeRemoteMethod(bound)

    class _FakeRay:
        @staticmethod
        def get(futures):
            return list(futures)

    ray = _FakeRay()

    dp_size, tp_size = 2, 2
    workers = [_FakeActorHandle(WorkerClass(rank=i)) for i in range(dp_size * tp_size)]
    proxy = WorkerGroupProxy(workers, dp_size=dp_size, tp_size=tp_size)

    batch = DataProto(["p0", "p1", "p2", "p3"])
    result = proxy.call("generate_sequences", batch)

    # --- verify the fan-out/fan-in did what the prose claims: the driver
    # wrote one line, `world_size` (4) ranks ran, and the collector folded
    # them back into ONE DataProto with one representative per DP group. ---
    assert isinstance(result, DataProto)
    assert result.values == ["rank0:p0", "rank0:p1", "rank2:p2", "rank2:p3"], result.values
    print(f"[block #2] dispatched to {len(workers)} workers "
          f"(dp_size={dp_size}, tp_size={tp_size}); collected: {result.values}")


BLOCKS = [
    block_single_controller_loop,
    block_dispatch_mechanism,
]


def main():
    for fn in BLOCKS:
        print(f"\n===== {fn.__name__} =====")
        fn()
    print(f"\nAll {len(BLOCKS)} code blocks executed and verified.")


if __name__ == "__main__":
    main()
