"""CI-tested extracts of runnable code blocks from
content/07-inference-serving/07-prefix-caching.md

Each `block_*` function reproduces the book's ACTUAL code verbatim (modulo
wrapping in a function) and then exercises it with tiny fixtures, asserting
the claims the prose/comments make about the results. Run directly:
`python3 tests/07-inference-serving__07-prefix-caching.py`

Blocks in this chapter:
  - block #0 (line ~89, ~140 lines): RadixNode / RadixTree (SGLang RadixAttention)
  - block #1 (line ~245, ~90 lines): APCBlockAllocator (vLLM Automatic Prefix Caching)
  - block #2 (bash)   -> SKIP(shell): `python -m vllm...` launch command, not Python.
  - block #3 (python) -> SKIP(network): `requests.get("http://localhost:8000/metrics")`
                          against a live vLLM server — real network call, forbidden in CI.
  - block #4 (bash)   -> SKIP(shell): `python -m sglang.launch_server ...`, not Python.
  - block #5 (python) -> SKIP(network/weights): HF `transformers` prefix-reuse demo — needs
                          to download SmolLM2-135M and run a forward pass, forbidden in CI.
  - block #6 (python) -> SKIP(out-of-scope): `build_prompt`/`build_prompt_bad` string-
                          formatting example. It is actually CPU-safe (no network/GPU), but
                          is not one of the 2 blocks assigned for this chapter's test.
  - block #7 (python) -> SKIP(out-of-scope): `pad_to_block_boundary` helper. Also CPU-safe,
                          also not one of the 2 assigned blocks for this chapter's test.
"""
# The book's block #0 opens with `from __future__ import annotations`, which
# Python requires to be the first statement of a *module* (not inside a
# function). We hoist it to the top of this file — semantics are unchanged,
# since it only affects annotation evaluation, and this file has no
# conflicting annotations.
from __future__ import annotations


def block_radix_tree():
    # content block #0 (verbatim, minus the future-import hoisted above)
    from dataclasses import dataclass, field

    @dataclass
    class RadixNode:
        """
        A node in the RadixAttention trie.

        `token_ids` is the edge label leading into this node, and `kv` holds the
        KV tensors for *exactly those tokens*, one entry per token. Here each
        entry is a placeholder bytes object; in a real implementation it is a
        physical GPU block reference. Storing KV per edge is what makes an edge
        split cheap — cutting the label at index `lcp` cuts the KV list at the
        same index — and it is what lets a lookup return a *partial* edge match
        without claiming more cached tokens than it actually has.
        """
        token_ids: list[int]              # edge label (tokens along this edge)
        kv: list[bytes]                   # per-token KV for this edge
        children: dict[int, "RadixNode"] = field(default_factory=dict)
        ref_count: int = 0                # how many active requests use this node
        last_access_time: float = 0.0

        def is_leaf(self) -> bool:
            return len(self.children) == 0

    class RadixTree:
        """
        Minimal RadixAttention trie for prefix caching.
        Supports insert, longest-prefix lookup, and LRU eviction.
        """

        def __init__(self):
            self.root = RadixNode(token_ids=[], kv=[])
            self._clock = 0.0

        def _tick(self) -> float:
            self._clock += 1.0
            return self._clock

        def insert(self, token_ids: list[int], kv: list[bytes]) -> None:
            """
            Insert a fully prefilled sequence (and its per-token KV) into the trie.
            Splits existing edges as needed (standard radix tree insertion).
            """
            assert len(kv) == len(token_ids), "one KV entry per token"
            node = self.root
            idx = 0  # position in token_ids

            while idx < len(token_ids):
                first_token = token_ids[idx]

                if first_token not in node.children:
                    # No matching child — create a new leaf holding the suffix.
                    node.children[first_token] = RadixNode(
                        token_ids=token_ids[idx:],
                        kv=kv[idx:],
                        last_access_time=self._tick(),
                    )
                    return

                child = node.children[first_token]
                # Find longest common prefix between token_ids[idx:] and child.token_ids.
                lcp = 0
                while (lcp < len(child.token_ids) and
                       idx + lcp < len(token_ids) and
                       child.token_ids[lcp] == token_ids[idx + lcp]):
                    lcp += 1

                if lcp == len(child.token_ids):
                    # Fully matched the existing edge — descend.
                    idx += lcp
                    node = child
                    continue

                # Partial match — split the edge at `lcp` (here 1 <= lcp < len(edge)).
                split_node = RadixNode(
                    token_ids=child.token_ids[:lcp],
                    kv=child.kv[:lcp],
                    last_access_time=self._tick(),
                )
                # Old child keeps the suffix of the label and the matching KV slice.
                child.token_ids = child.token_ids[lcp:]
                child.kv = child.kv[lcp:]
                split_node.children[child.token_ids[0]] = child
                # Attach split_node to parent.
                node.children[first_token] = split_node
                idx += lcp
                # If the inserted sequence ends exactly at the split point there is
                # nothing left to attach — split_node already owns its KV. (Missing
                # this case is the classic radix-insert bug: it fires the first time
                # a request is a strict prefix of one already in the tree.)
                if idx < len(token_ids):
                    split_node.children[token_ids[idx]] = RadixNode(
                        token_ids=token_ids[idx:],
                        kv=kv[idx:],
                        last_access_time=self._tick(),
                    )
                return
            # Falling out of the loop means the sequence terminated on an existing
            # node boundary: its KV is already stored along the path we walked.

        def match_prefix(self, token_ids: list[int]) -> tuple[int, list[bytes]]:
            """
            Find the longest cached prefix of token_ids.
            Returns (length_matched, per-token KV for exactly those tokens) —
            the two are always consistent, including on a partial edge match.
            """
            node = self.root
            idx = 0
            matched_kv: list[bytes] = []

            while idx < len(token_ids):
                first_token = token_ids[idx]
                if first_token not in node.children:
                    break

                child = node.children[first_token]
                child.last_access_time = self._tick()  # update LRU stamp

                lcp = 0
                while (lcp < len(child.token_ids) and
                       idx + lcp < len(token_ids) and
                       child.token_ids[lcp] == token_ids[idx + lcp]):
                    lcp += 1

                matched_kv.extend(child.kv[:lcp])   # only the tokens that matched
                idx += lcp

                if lcp < len(child.token_ids):
                    break  # partial match — descent stops here

                node = child

            return idx, matched_kv

    # --- exercise the trie with tiny fixtures ---
    def kvs(seq):
        """KV is a pure function of the token path, so a per-token placeholder
        keyed on the token id is a faithful stand-in."""
        return [f"kv{t}".encode() for t in seq]

    tree = RadixTree()

    # Session 1: message A followed by B (A = [1,2,3], A+B = [1,2,3,4,5]).
    tree.insert([1, 2, 3], kvs([1, 2, 3]))
    tree.insert([1, 2, 3, 4, 5], kvs([1, 2, 3, 4, 5]))

    # A brand-new request sharing only message A should match the first 3 tokens.
    matched_len, kv = tree.match_prefix([1, 2, 3, 9, 9])
    print(f"match_prefix([1,2,3,9,9]) -> matched_len={matched_len}, kv={kv}")
    assert matched_len == 3
    assert kv == kvs([1, 2, 3])

    # A request replaying the full A+B session should match all 5 tokens.
    matched_len2, kv2 = tree.match_prefix([1, 2, 3, 4, 5])
    print(f"match_prefix([1,2,3,4,5]) -> matched_len={matched_len2}, kv={kv2}")
    assert matched_len2 == 5
    assert kv2 == kvs([1, 2, 3, 4, 5])

    # A completely disjoint prompt should have zero cached prefix.
    matched_len3, kv3 = tree.match_prefix([7, 8, 9])
    assert matched_len3 == 0
    assert kv3 == []

    # Inserting a diverging continuation ([1,2,3,4,9]) should split the
    # existing "4,5" edge at the common prefix "4".
    tree.insert([1, 2, 3, 4, 9], kvs([1, 2, 3, 4, 9]))
    matched_len4, kv4 = tree.match_prefix([1, 2, 3, 4, 9])
    assert matched_len4 == 5
    assert kv4 == kvs([1, 2, 3, 4, 9])
    # The original A+B path must still resolve correctly after the split.
    matched_len5, kv5 = tree.match_prefix([1, 2, 3, 4, 5])
    assert matched_len5 == 5
    assert kv5 == kvs([1, 2, 3, 4, 5])

    # The split node itself must serve its own segment: [1,2,3,4] is cached
    # even though no inserted sequence *ends* there.
    matched_len6, kv6 = tree.match_prefix([1, 2, 3, 4])
    assert matched_len6 == 4
    assert kv6 == kvs([1, 2, 3, 4])

    # (length, kv) must stay consistent on a PARTIAL edge match: matching
    # 3 tokens of a 5-token edge must return exactly 3 KV entries, never the
    # whole edge's blob.
    partial = RadixTree()
    partial.insert([1, 2, 3, 4, 5], kvs([1, 2, 3, 4, 5]))
    n, kvp = partial.match_prefix([1, 2, 3, 77])
    assert (n, kvp) == (3, kvs([1, 2, 3])), (n, kvp)
    assert len(kvp) == n

    # A later request that is a STRICT PREFIX of one already in the tree must
    # insert cleanly (this used to raise IndexError).
    truncation = RadixTree()
    truncation.insert([1, 2, 3], kvs([1, 2, 3]))
    truncation.insert([1, 2], kvs([1, 2]))
    assert truncation.match_prefix([1, 2]) == (2, kvs([1, 2]))
    assert truncation.match_prefix([1, 2, 3]) == (3, kvs([1, 2, 3]))

    # Multi-turn chain: root -> A -> A+B -> A+B+C all resolve.
    chain = RadixTree()
    for seq in ([1, 2], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]):
        chain.insert(seq, kvs(seq))
    assert chain.match_prefix([1, 2, 3, 4, 5, 6, 7]) == (6, kvs([1, 2, 3, 4, 5, 6]))


def block_apc_allocator():
    # content block #1 (verbatim)
    from collections import OrderedDict

    class APCBlockAllocator:
        def __init__(self, num_blocks: int, block_size: int):
            self.block_size = block_size
            self.free_blocks: list[int] = list(range(num_blocks))
            # Map from content hash -> physical block id
            self.prefix_cache: dict[int, int] = {}
            # LRU ordering: block_id -> None, oldest first. An OrderedDict gives
            # O(1) "promote to most-recent" and O(1) "pop the oldest"; a plain
            # append-only timestamp list would NOT be an LRU, because a hot block's
            # stale early entry would still sort to the front and be evicted first.
            self.lru_order: "OrderedDict[int, None]" = OrderedDict()
            # Blocks allocated but not yet filled by prefill: their hashes may not
            # be published until the KV tensors actually exist.
            self.pending_publish: list[tuple[int, int]] = []

        def _compute_block_hash(self, prev_hash: int, token_ids: list[int]) -> int:
            """Chain the previous block hash with current token IDs."""
            import hashlib
            # Encode each token id in 4 bytes. (`bytes(token_ids)` would raise for
            # any id > 255, and real vocabularies run to 32k-256k entries.)
            packed = b"".join(t.to_bytes(4, 'little') for t in token_ids)
            data = prev_hash.to_bytes(8, 'little') + packed
            return int.from_bytes(hashlib.sha256(data).digest()[:8], 'little')

        def allocate_or_reuse(
            self,
            token_ids_blocks: list[list[int]]
        ) -> tuple[list[int], int]:
            """
            Given a list of blocks (each a list of token IDs), return:
              - block_ids: physical block IDs for the full context
              - num_cached: how many leading blocks came from cache
            """
            block_ids = []
            prev_hash = 0
            num_cached = 0
            still_contiguous = True  # only a run of hits from position 0 counts

            for i, block_tokens in enumerate(token_ids_blocks):
                h = self._compute_block_hash(prev_hash, block_tokens)

                if h in self.prefix_cache:
                    # Cache hit: reuse this physical block.
                    phys_id = self.prefix_cache[h]
                    block_ids.append(phys_id)
                    if still_contiguous:
                        num_cached += 1
                    self._touch(phys_id)
                else:
                    # Cache miss: allocate a fresh physical block.
                    # Evict if necessary. A later block may still hash-hit (its
                    # chained hash is unaffected by *this* block being evicted),
                    # but the caller may only skip prefill for a contiguous run
                    # from position 0, so stop counting here.
                    still_contiguous = False
                    if not self.free_blocks:
                        self._evict_lru()
                    phys_id = self.free_blocks.pop()
                    self._touch(phys_id)
                    block_ids.append(phys_id)
                    # Do NOT publish the hash yet: the block's KV tensors do not
                    # exist until prefill writes them, and a concurrent request
                    # that hit this hash would read uninitialised GPU memory.
                    self.pending_publish.append((h, phys_id))

                prev_hash = h

            return block_ids, num_cached

        def publish_pending(self):
            """
            Called by the scheduler AFTER prefill has written the KV tensors for
            the newly allocated blocks. This is the invariant vLLM enforces: a
            block enters the prefix cache only once it is complete and written.
            """
            for h, phys_id in self.pending_publish:
                self.prefix_cache[h] = phys_id
                self._touch(phys_id)
            self.pending_publish.clear()

        def _touch(self, block_id: int):
            """Mark a block as most-recently-used."""
            self.lru_order[block_id] = None
            self.lru_order.move_to_end(block_id)

        def _evict_lru(self):
            """Remove the least-recently-used cached block."""
            while self.lru_order:
                block_id, _ = self.lru_order.popitem(last=False)  # oldest first
                # Remove from prefix cache if it's still there and not pinned.
                for k, v in list(self.prefix_cache.items()):
                    if v == block_id:
                        del self.prefix_cache[k]
                        self.free_blocks.append(block_id)
                        return

    # --- exercise the allocator with tiny fixtures ---
    # block_size is nominal here (blocks are pre-segmented lists of token ids).
    allocator = APCBlockAllocator(num_blocks=4, block_size=4)

    # First request: two fresh blocks -> both cache misses.
    blocks_req1 = [[1, 2, 3, 4], [5, 6, 7, 8]]
    ids1, cached1 = allocator.allocate_or_reuse(blocks_req1)
    print(f"req1 block_ids={ids1}, num_cached={cached1}")
    assert cached1 == 0
    assert len(ids1) == 2
    # Nothing is published until prefill has written the KV tensors.
    assert len(allocator.prefix_cache) == 0
    allocator.publish_pending()
    assert len(allocator.prefix_cache) == 2

    # Second request: shares the first block (same prefix), diverges on the second.
    blocks_req2 = [[1, 2, 3, 4], [9, 9, 9, 9]]
    ids2, cached2 = allocator.allocate_or_reuse(blocks_req2)
    allocator.publish_pending()
    print(f"req2 block_ids={ids2}, num_cached={cached2}")
    assert cached2 == 1                    # only the first block was a hit
    assert ids2[0] == ids1[0]              # reused the same physical block
    assert ids2[1] != ids1[1]              # second block is a fresh allocation

    # Fill remaining capacity and force eviction: num_blocks=4, 3 already used
    # (blocks from req1 + the one new block from req2), 1 free slot left.
    blocks_req3 = [[100, 101, 102, 103]]   # brand-new content -> cache miss
    ids3, cached3 = allocator.allocate_or_reuse(blocks_req3)
    allocator.publish_pending()
    assert cached3 == 0
    assert allocator.free_blocks == []     # pool now fully allocated (4 blocks)

    # A fourth request with entirely new content must trigger LRU eviction
    # since the free list is empty.
    blocks_req4 = [[200, 201, 202, 203]]
    cache_size_before = len(allocator.prefix_cache)
    hot_hash = allocator._compute_block_hash(0, [1, 2, 3, 4])   # hit twice: hottest
    ids4, cached4 = allocator.allocate_or_reuse(blocks_req4)
    allocator.publish_pending()
    print(f"req4 block_ids={ids4}, num_cached={cached4}, "
          f"cache_size before={cache_size_before} after={len(allocator.prefix_cache)}")
    assert cached4 == 0
    # Eviction removed exactly one LRU entry before inserting the new one,
    # so the cache size is unchanged (one out, one in).
    assert len(allocator.prefix_cache) == cache_size_before
    assert allocator.free_blocks == []     # the freed slot was immediately consumed
    # The genuinely LEAST-recently-used entry went, not the hottest one.
    assert hot_hash in allocator.prefix_cache, "LRU evicted the most-recently-used block!"

    # num_cached counts only a CONTIGUOUS run of hits from position 0: a block
    # that hits after an earlier miss must not be counted, or the caller would
    # skip prefill for KV that was never computed.
    gap = APCBlockAllocator(num_blocks=8, block_size=4)
    seq = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]
    gap.allocate_or_reuse(seq)
    gap.publish_pending()
    h0 = gap._compute_block_hash(0, seq[0])
    del gap.prefix_cache[h0]               # simulate eviction of block 0 only
    _, ncached = gap.allocate_or_reuse(seq)
    assert ncached == 0, ncached           # blocks 1 and 2 still hit, but not leading


BLOCKS = [
    block_radix_tree,
    block_apc_allocator,
]


def main():
    for fn in BLOCKS:
        print(f"\n===== {fn.__name__} =====")
        fn()
    print(f"\nAll {len(BLOCKS)} code blocks executed and verified.")
    print(
        "\nSKIPPED (see module docstring for details):\n"
        "  #2 (bash)   - SKIP(shell): vLLM server launch command, not Python\n"
        "  #3 (python) - SKIP(network): requests.get() against a live vLLM metrics endpoint\n"
        "  #4 (bash)   - SKIP(shell): SGLang server launch command, not Python\n"
        "  #5 (python) - SKIP(network/weights): HF transformers demo downloads model weights\n"
        "  #6 (python) - SKIP(out-of-scope): CPU-safe but not one of the 2 assigned blocks\n"
        "  #7 (python) - SKIP(out-of-scope): CPU-safe but not one of the 2 assigned blocks"
    )


if __name__ == "__main__":
    main()
