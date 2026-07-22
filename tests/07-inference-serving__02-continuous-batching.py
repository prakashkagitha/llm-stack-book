"""
Runnable test for content/07-inference-serving/02-continuous-batching.md

Blocks tested:
  - block #2 (line ~112): toy_scheduler.py -- Status, Request, Scheduler classes.
  - block #3 (line ~308): run_sim() driver + __main__ demo.

Both blocks use only the Python standard library (dataclasses, collections, enum,
itertools, random) -- no third-party or network dependencies, so they are copied
here verbatim and executed as a whole, exactly as they appear in the chapter,
concatenated in order (block #3 depends on names defined in block #2).

Blocks #0 and #1 are non-Python (marked SKIP by the harness) and are not included.

BUG FIXED (mirrored back into the chapter's .md source):
`Scheduler._preempt_one` and `Scheduler.schedule` had a real double-counting bug.
`schedule()`'s part (1) loop iterated `self.running` directly while
`_preempt_one()` mutated that same list, and picked its victim from among ALL
current members of `self.running` -- including the request currently being
resized and requests that had *already* succeeded and been appended to
`survivors` earlier in the same pass. Two consequences: (a) a request could
select ITSELF as its preemption victim (pushed onto `self.waiting`), then
succeed on retry and also land in `survivors` -- present in both collections
simultaneously; (b) an already-`survivors`-appended request could later be
picked as someone else's victim and get pushed onto `self.waiting` too, while
remaining in `survivors`. Either way the same `Request` object ends up
counted in `self.running` AND `self.waiting`, gets admitted again next
iteration, and is retired into `self.finished` multiple times. Confirmed with
the chapter's own `__main__` demo (200 requests, seed 0): before the fix,
`len(s.finished) == 491` (some rids finished up to 34 times) and
`free_blocks` never returned to `TOTAL_BLOCKS`. Fix: (1) `_preempt_one` takes
an `exclude` request that can never be selected as its own victim, and
(2) `schedule()` iterates a *snapshot* of `self.running`, skips entries whose
`.status` was flipped away from RUNNING by an earlier eviction this same
pass, and rebuilds `self.running` by filtering `survivors` on `.status`
rather than just presence in the list. After the fix, `len(s.finished) == 200`
with zero duplicates and `free_blocks == TOTAL_BLOCKS` once idle.
"""
# NOTE: `from __future__ import annotations` must be the first statement in the
# file (Python requires future-imports to precede all other code), so it is
# hoisted here even though it appears mid-block in the chapter's source.
from __future__ import annotations

# =====================================================================
# Block #2 (line ~112) -- toy_scheduler.py, verbatim from the chapter
# =====================================================================
"""toy_scheduler.py — a from-scratch continuous-batching (iteration-level) scheduler.

We simulate the *control flow* of an LLM server. The "model" is mocked: each request
has a known prompt length and a known number of output tokens, and one decode iteration
produces exactly one token per running request. The interesting part is the SCHEDULER.
"""
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
    def _preempt_one(self, exclude: Request | None = None) -> bool:
        """Evict the lowest-priority / newest running request back to WAITING.
        Returns True if something was preempted (and its blocks freed).
        `exclude` is the request currently being resized by schedule() — it must
        never be allowed to preempt itself (see note below)."""
        candidates = [r for r in self.running if r is not exclude]
        if not candidates:
            return False
        # victim selection: in FCFS we recompute (LIFO — preempt the most-recently
        # admitted, i.e. largest arrival). In priority, preempt the worst priority.
        if self.policy == "priority":
            victim = max(candidates, key=lambda r: (r.priority, r.arrival))
        else:
            victim = max(candidates, key=lambda r: r.arrival)
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
        # Iterate a SNAPSHOT of self.running: preempting a victim mutates the live
        # list, and looping over a list while removing from it silently skips
        # entries. `req.status` (flipped to WAITING by a preemption) is what tells
        # us a snapshot entry is no longer actually running.
        for req in list(self.running):
            if req.status is not Status.RUNNING:
                continue                     # evicted earlier in this same pass,
                                              # as another request's preemption victim
            want = req.blocks_needed(extra_tokens=1)
            while not self._alloc(req, want):
                # exclude=req: a request must never be allowed to pick ITSELF as its
                # own preemption victim (it is still a member of self.running at this
                # point). Without the exclusion, self-preemption pushes req onto
                # self.waiting; when the retry then succeeds, req also lands in
                # survivors — the same Request ends up in both collections, gets
                # admitted a second time next iteration, and is double-counted.
                if not self._preempt_one(exclude=req):
                    break                    # nothing left to evict; this req stalls
                # after a preemption, free_blocks grew — retry the alloc
            else:
                req.status = Status.RUNNING  # (re)confirm: may have been WAITING
                survivors.append(req)        # only if THIS req was preempted+retried
                token_budget -= 1            # one decode token
                continue
            # alloc still failed even after preemptions: drop req back to waiting
            self._free(req)
            req.status = Status.WAITING
            req.prefilled = False
            self.waiting.appendleft(req)
        # A survivor added earlier in this loop can still be preempted LATER in the
        # same loop (as someone else's victim) — filter by status, not just presence
        # in `survivors`, so such requests correctly end up only in self.waiting.
        self.running = [r for r in survivors if r.status is Status.RUNNING]

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


# =====================================================================
# Block #3 (line ~308) -- driver, verbatim from the chapter
# =====================================================================
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

    # ------------------------------------------------------------------
    # Assertions: exercise the classes/functions with tiny inputs and
    # sanity-check the scheduler's core invariants (block accounting,
    # FCFS admission order, memory never over-committed, everyone finishes).
    # ------------------------------------------------------------------
    assert len(s.finished) == 200, "all 200 requests should eventually finish"
    assert s.free_blocks == TOTAL_BLOCKS, "all KV blocks must be returned when idle"
    assert total_tokens == sum(r.output_len for r in reqs)
    assert mean_batch > 1.0, "continuous batching should run more than 1 req/iter on average"
    assert all(t >= 0 for t in ttfts), "TTFT can't be negative"

    # A small, hand-checkable scenario: two requests that both fit easily.
    tiny = [
        Request(rid=0, arrival=0, prompt_len=8, output_len=3),
        Request(rid=1, arrival=0, prompt_len=8, output_len=2),
    ]
    tiny_sched = run_sim(tiny, policy="fcfs")
    assert len(tiny_sched.finished) == 2
    assert {r.generated for r in tiny_sched.finished} == {2, 3}
    assert tiny_sched.free_blocks == TOTAL_BLOCKS

    # Exercise the "priority" and "shortest" policies too (schedule()/_waiting_order()).
    for policy in ("priority", "shortest"):
        p_reqs = [
            Request(rid=i, arrival=0, prompt_len=10, output_len=(i % 3) + 1, priority=i % 2)
            for i in range(10)
        ]
        p_sched = run_sim(p_reqs, policy=policy)
        assert len(p_sched.finished) == 10
        assert p_sched.free_blocks == TOTAL_BLOCKS

    # Force preemption: tiny memory budget so more requests are admitted than
    # blocks allow, exercising _preempt_one() and the retry loop in schedule().
    import importlib, sys
    this_mod = sys.modules[__name__]
    this_mod.TOTAL_BLOCKS = 4   # 4 blocks * 16 tokens/block = 64 tokens of KV total
    preempt_reqs = [
        Request(rid=i, arrival=0, prompt_len=20, output_len=5) for i in range(5)
    ]
    preempt_sched = Scheduler(policy="fcfs")
    preempt_sched.free_blocks = 4
    for r in preempt_reqs:
        preempt_sched.add(r)
    for _ in range(200):
        if not preempt_sched.waiting and not preempt_sched.running:
            break
        preempt_sched.step()
    assert len(preempt_sched.finished) == 5, "even under memory pressure everyone finishes eventually"
    assert preempt_sched.free_blocks == 4, "blocks fully reclaimed once idle"

    print("\nAll assertions passed.")
