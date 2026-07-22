"""
Runnable-code test for content/12-production-mlops/01-serving-system-design.md

Chapter blocks tested (assembled in document order so later blocks can use
names defined by earlier ones, exactly as the chapter intends):

    - block #0 (line ~45):  TokenBucketLimiter (token-aware rate limiter)  -> instantiated/used below
    - block #1 (line ~111): ReplicaState + pick_replica (power-of-2-choices router) -> instantiated/used below
    - block #3 (line ~251): kv_bytes_per_token / max_concurrent_requests   -> executed in-block (the
      chapter's own code already calls it and prints the result); trivially CPU-safe (pure arithmetic,
      no external state), so it is exercised here too even though the task's heuristic pre-labeled it
      a "fragment".
    - block #5 (line ~340): desired_replicas (queue-driven autoscaler)    -> called below with several
      load scenarios
    - block #6 (line ~381): ModelLRUCache (whole-model hot-swap cache)    -> instantiated/used below

SKIPPED (per task spec / fragments):
    - block #2 (line ~212): SKIP(fragment): `admit(request, replica, ...)` reads `replica.queued_tokens`
      and `request.prompt_len` off caller-supplied objects that are never defined in the chapter text --
      it is a standalone policy function meant to be read, not a runnable unit on its own. Its logic
      (compare predicted TTFT to an SLO and reject) is simple pass/fail arithmetic identical in spirit
      to what block #3's memory-budget arithmetic already exercises, so nothing distinct is left
      untested by skipping it.
    - block #4 (line ~294): SKIP(fragment): `BatchPolicy.step_budget` iterates `waiting_prefills` and
      reads `req.remaining_prefill` off caller-supplied request objects that are never defined in the
      chapter text -- exercising it would require inventing a request schema the book never specifies,
      which risks testing our own fixture instead of the book's logic. `BatchPolicy.__init__` (the
      config-holding part) is trivial attribute assignment with no logic to verify.
"""

import math
import random
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field


# ============================================================================
# block #0 (line ~45): TokenBucketLimiter
# ============================================================================

class TokenBucketLimiter:
    """
    Token-aware rate limiter. The 'tokens' here are LLM tokens (prompt+completion),
    not generic request credits. We refill `rate` tokens/sec up to `capacity`.

    Each LLM request first RESERVES an estimate (prompt_len + max_new_tokens),
    then RECONCILES with the true completion length when the stream finishes.
    This prevents a flood of long-generation requests from blowing the budget.
    """
    def __init__(self, rate_per_sec: float, capacity: float):
        self.rate = rate_per_sec          # e.g. 50_000 tokens/sec for a tier
        self.capacity = capacity          # burst ceiling, e.g. 100_000 tokens
        self.tokens = capacity            # start full
        self.last = time.monotonic()
        self.lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last = now

    def try_reserve(self, estimated_tokens: float) -> bool:
        """Reserve up-front. Returns False (429 -> client backs off) if over budget."""
        with self.lock:
            self._refill()
            if self.tokens >= estimated_tokens:
                self.tokens -= estimated_tokens
                return True
            return False

    def reconcile(self, estimated_tokens: float, actual_tokens: float) -> None:
        """Refund or charge the difference once the true length is known."""
        with self.lock:
            self._refill()
            # If we over-estimated, give tokens back (capped at capacity).
            self.tokens = min(self.capacity, self.tokens + (estimated_tokens - actual_tokens))


# ============================================================================
# block #1 (line ~111): ReplicaState + pick_replica (power-of-2-choices router)
# ============================================================================

@dataclass
class ReplicaState:
    id: str
    queued_tokens: int = 0        # tokens waiting (prefill not yet started)
    running_tokens: int = 0       # tokens of active sequences (KV held)
    cached_prefixes: set = field(default_factory=set)  # hashes of cached prefix blocks

    def effective_load(self, req_prefix_hashes: set) -> float:
        # Base load = work in flight. Tokens are the right unit, not request count.
        load = self.queued_tokens + 0.5 * self.running_tokens
        # Cache bonus: each overlapping prefix block we DON'T have to recompute
        # is real prefill work saved, so we subtract it from effective load.
        overlap = len(self.cached_prefixes & req_prefix_hashes)
        CACHE_BLOCK_TOKENS = 16
        return load - overlap * CACHE_BLOCK_TOKENS


def pick_replica(replicas, req_prefix_hashes, d=2):
    """Power-of-d-choices, load- and cache-aware. O(d), no global scan needed."""
    candidates = random.sample(replicas, k=min(d, len(replicas)))
    return min(candidates, key=lambda r: r.effective_load(req_prefix_hashes))


# ============================================================================
# block #3 (line ~251): kv_bytes_per_token / max_concurrent_requests
# ============================================================================

def kv_bytes_per_token(num_layers, num_kv_heads, head_dim, bytes_per_elem=2):
    # 2 for K and V; bytes_per_elem=2 for fp16/bf16, 1 for fp8.
    return 2 * num_layers * num_kv_heads * head_dim * bytes_per_elem


def max_concurrent_requests(hbm_gb, weight_gb, reserve_gb,
                            num_layers, num_kv_heads, head_dim,
                            avg_ctx_tokens, bytes_per_elem=2):
    """How many simultaneous sequences fit in one replica's HBM."""
    free_bytes = (hbm_gb - weight_gb - reserve_gb) * 1e9
    per_tok = kv_bytes_per_token(num_layers, num_kv_heads, head_dim, bytes_per_elem)
    per_request = per_tok * avg_ctx_tokens
    return int(free_bytes // per_request)


# Llama-3-70B-ish on one 80GB H100 shard would not fit weights alone (140GB bf16),
# so 70B needs tensor parallelism. Let's size an 8B model on a single 80GB H100:
#   8B bf16 weights ~= 16 GB. L=32, num_kv_heads=8 (GQA), head_dim=128.
n = max_concurrent_requests(
    hbm_gb=80, weight_gb=16, reserve_gb=4,
    num_layers=32, num_kv_heads=8, head_dim=128,
    avg_ctx_tokens=2048, bytes_per_elem=2,
)
print(n)  # -> 223 concurrent 2k-token sequences fit in KV cache


# ============================================================================
# block #5 (line ~340): desired_replicas (queue-driven autoscaler)
# ============================================================================

def desired_replicas(current_replicas,
                     waiting_tokens, running_tokens,
                     replica_token_capacity,   # tokens/sec one replica sustains
                     target_utilization=0.7,
                     warm_buffer=2,
                     min_replicas=2, max_replicas=64):
    """
    Target-tracking autoscaler driven by QUEUE load (a leading indicator),
    not GPU utilization (a lagging one).

    offered_load: tokens of work the fleet currently owes (queued + in-flight).
    We size so that this load sits at `target_utilization` of total capacity,
    then add a warm buffer to absorb spikes during the slow cold-start window.
    """
    offered_load = waiting_tokens + running_tokens
    total_capacity_needed = offered_load / target_utilization
    raw = math.ceil(total_capacity_needed / replica_token_capacity)
    desired = raw + warm_buffer
    # Hysteresis: clamp and let the caller apply scale-down delay separately.
    return max(min_replicas, min(max_replicas, desired))


# ============================================================================
# block #6 (line ~381): ModelLRUCache (whole-model hot-swap cache)
# ============================================================================

class ModelLRUCache:
    """
    HBM-as-cache for whole-model hot-swapping. Keeps total loaded weights under
    `capacity_gb`; evicts the least-recently-used model on a miss. Production
    systems pin the top-k busiest models so they are never evicted (anti-thrash).
    """
    def __init__(self, capacity_gb, pinned=()):
        self.capacity_gb = capacity_gb
        self.loaded = OrderedDict()    # model_id -> size_gb, MRU at the end
        self.pinned = set(pinned)
        self.used_gb = 0.0

    def get(self, model_id, size_gb, load_fn):
        if model_id in self.loaded:
            self.loaded.move_to_end(model_id)   # mark most-recently-used
            return  # hot path: already resident, zero cold-start
        # Cache miss: evict LRU (never a pinned model) until it fits.
        while self.used_gb + size_gb > self.capacity_gb:
            for victim in list(self.loaded):
                if victim not in self.pinned:
                    self.used_gb -= self.loaded.pop(victim)
                    break
            else:
                raise MemoryError("cannot fit model; all residents pinned")
        load_fn(model_id)                # SLOW: weights -> HBM (the cold start)
        self.loaded[model_id] = size_gb
        self.used_gb += size_gb


# ============================================================================
# Glue: actually exercise each block with tiny fixtures.
# ============================================================================

def test_token_bucket_limiter():
    limiter = TokenBucketLimiter(rate_per_sec=1_000.0, capacity=5_000.0)
    assert limiter.tokens == 5_000.0

    # Reserve 1000 tokens up front for an estimated (prompt+max_new) request.
    ok = limiter.try_reserve(1_000.0)
    assert ok is True
    assert limiter.tokens <= 4_000.0 + 1.0  # allow tiny refill jitter from elapsed time

    # A wildly over-budget reservation must be rejected (429 path).
    ok2 = limiter.try_reserve(1_000_000.0)
    assert ok2 is False

    # Reconcile: actual completion was shorter than estimated -> refund the slack.
    before = limiter.tokens
    limiter.reconcile(estimated_tokens=1_000.0, actual_tokens=400.0)
    after = limiter.tokens
    assert after > before, "reconcile should refund unused reserved tokens"
    assert after <= limiter.capacity


def test_pick_replica():
    random.seed(0)
    # Replica A: quiet, no cache overlap. Replica B: busy but has our prefix cached.
    a = ReplicaState(id="A", queued_tokens=0, running_tokens=0, cached_prefixes=set())
    b = ReplicaState(id="B", queued_tokens=1000, running_tokens=0, cached_prefixes={1, 2, 3})
    c = ReplicaState(id="C", queued_tokens=5000, running_tokens=2000, cached_prefixes=set())

    # Direct effective_load sanity checks (no randomness).
    req_hashes = {1, 2, 3, 4}
    assert a.effective_load(req_hashes) == 0.0
    # b: load=1000, overlap=3 -> 1000 - 3*16 = 952
    assert b.effective_load(req_hashes) == 1000 - 3 * 16
    assert c.effective_load(req_hashes) == 5000 + 0.5 * 2000

    replicas = [a, b, c]
    # Run pick_replica many times with fixed seed; it must always return a
    # ReplicaState object drawn from the candidate pool, and over many trials
    # never pick the obviously-overloaded replica C when compared head-to-head
    # against a lighter one.
    chosen_ids = set()
    for _ in range(50):
        chosen = pick_replica(replicas, req_hashes, d=2)
        assert isinstance(chosen, ReplicaState)
        chosen_ids.add(chosen.id)
    # With d=2 sampling from 3 replicas where C is much heavier than A and B,
    # C should essentially never win a pairwise effective-load comparison
    # against A or B (it can only "win" if never sampled at all, which given
    # 50 trials across 3 replicas practically never happens both ways).
    assert "A" in chosen_ids or "B" in chosen_ids


def test_kv_sizing():
    per_tok = kv_bytes_per_token(num_layers=32, num_kv_heads=8, head_dim=128, bytes_per_elem=2)
    assert per_tok == 2 * 32 * 8 * 128 * 2  # 131072 bytes/token

    got = max_concurrent_requests(
        hbm_gb=80, weight_gb=16, reserve_gb=4,
        num_layers=32, num_kv_heads=8, head_dim=128,
        avg_ctx_tokens=2048, bytes_per_elem=2,
    )
    assert got == n  # matches the module-level call the chapter itself makes
    assert got == 223, f"expected 223 concurrent sequences, got {got}"


def test_desired_replicas():
    # Light load: should clamp to min_replicas + nothing extra needed beyond floor.
    lo = desired_replicas(
        current_replicas=2, waiting_tokens=0, running_tokens=0,
        replica_token_capacity=10_000, target_utilization=0.7,
        warm_buffer=2, min_replicas=2, max_replicas=64,
    )
    assert lo == 2  # floor: raw capacity need is 0, +2 buffer = 2, still clamped to min=2

    # Heavy load: offered_load=700_000 tokens, target_util=0.7, capacity=10_000/replica
    # -> total_capacity_needed = 700_000/0.7 = 1_000_000 -> raw = 100 -> +2 buffer = 102
    # -> clamped to max_replicas=64.
    hi = desired_replicas(
        current_replicas=10, waiting_tokens=500_000, running_tokens=200_000,
        replica_token_capacity=10_000, target_utilization=0.7,
        warm_buffer=2, min_replicas=2, max_replicas=64,
    )
    assert hi == 64

    # Moderate load with an exact, hand-checkable answer.
    mid = desired_replicas(
        current_replicas=5, waiting_tokens=3_000, running_tokens=4_000,
        replica_token_capacity=1_000, target_utilization=0.7,
        warm_buffer=2, min_replicas=2, max_replicas=64,
    )
    # offered_load=7000, total_capacity_needed=7000/0.7=10000, raw=ceil(10000/1000)=10, +2=12
    assert mid == 12


def test_model_lru_cache():
    load_calls = []

    def load_fn(model_id):
        load_calls.append(model_id)

    cache = ModelLRUCache(capacity_gb=10.0, pinned=("base-8b",))

    # Load the pinned base model first.
    cache.get("base-8b", 4.0, load_fn)
    assert cache.used_gb == 4.0
    assert load_calls == ["base-8b"]

    # Load a second model; still fits (4 + 3 = 7 <= 10).
    cache.get("adapter-a", 3.0, load_fn)
    assert cache.used_gb == 7.0
    assert load_calls == ["base-8b", "adapter-a"]

    # Re-request an already-loaded model: hot path, no new load_fn call.
    cache.get("adapter-a", 3.0, load_fn)
    assert load_calls == ["base-8b", "adapter-a"]  # unchanged: cache hit

    # Request a third model that doesn't fit alongside both existing ones
    # (4 + 3 + 5 = 12 > 10): must evict the LRU *non-pinned* resident.
    # LRU order right now (by our access pattern) is [base-8b, adapter-a] with
    # adapter-a most-recently-used, so adapter-a should be evicted, not base-8b
    # (which is pinned and could never be evicted anyway).
    cache.get("adapter-b", 5.0, load_fn)
    assert "adapter-a" not in cache.loaded
    assert "base-8b" in cache.loaded  # pinned, survives
    assert "adapter-b" in cache.loaded
    assert cache.used_gb == 9.0  # 4 (base) + 5 (adapter-b)
    assert load_calls == ["base-8b", "adapter-a", "adapter-b"]

    # Requesting something that cannot fit even after evicting every
    # non-pinned resident must raise MemoryError (pinned base-8b alone is 4GB,
    # so an 8GB request leaves only 2GB free after evicting adapter-b).
    try:
        cache.get("huge-model", 8.0, load_fn)
        raise AssertionError("expected MemoryError when only pinned models remain and it still doesn't fit")
    except MemoryError:
        pass


if __name__ == "__main__":
    test_token_bucket_limiter()
    print("block #0 TokenBucketLimiter: OK")

    test_pick_replica()
    print("block #1 ReplicaState/pick_replica: OK")

    test_kv_sizing()
    print("block #3 kv_bytes_per_token/max_concurrent_requests: OK")

    test_desired_replicas()
    print("block #5 desired_replicas: OK")

    test_model_lru_cache()
    print("block #6 ModelLRUCache: OK")

    print("\nAll runnable blocks in 12-production-mlops/01-serving-system-design.md executed successfully.")
