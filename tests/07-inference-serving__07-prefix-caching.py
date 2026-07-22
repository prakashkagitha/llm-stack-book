"""CI-tested extracts of runnable code blocks from
content/07-inference-serving/07-prefix-caching.md

Each `block_*` function reproduces the book's ACTUAL code verbatim (modulo
wrapping in a function) and then exercises it with tiny fixtures, asserting
the claims the prose/comments make about the results. Run directly:
`python3 tests/07-inference-serving__07-prefix-caching.py`

Blocks in this chapter:
  - block #0 (line ~87, 130 lines): RadixNode / RadixTree (SGLang RadixAttention)
  - block #1 (line ~235, 73 lines): APCBlockAllocator (vLLM Automatic Prefix Caching)
  - block #2 (line ~451, bash)   -> SKIP(shell): `python -m vllm...` launch command, not Python.
  - block #3 (line ~463, python) -> SKIP(network): `requests.get("http://localhost:8000/metrics")`
                                     against a live vLLM server — real network call, forbidden in CI.
  - block #4 (line ~478, bash)   -> SKIP(shell): `python -m sglang.launch_server ...`, not Python.
  - block #5 (line ~490, python) -> SKIP(out-of-scope): `build_prompt`/`build_prompt_bad` string-
                                     formatting example. It is actually CPU-safe (no network/GPU), but
                                     is not one of the 2 blocks assigned for this chapter's test.
  - block #6 (line ~525, python) -> SKIP(out-of-scope): `pad_to_block_boundary` helper. Also CPU-safe,
                                     also not one of the 2 assigned blocks for this chapter's test.
"""
# The book's block #0 opens with `from __future__ import annotations`, which
# Python requires to be the first statement of a *module* (not inside a
# function). We hoist it to the top of this file — semantics are unchanged,
# since it only affects annotation evaluation, and this file has no
# conflicting annotations.
from __future__ import annotations


def block_radix_tree():
    # content lines ~88-216 (verbatim, minus the future-import hoisted above)
    import hashlib
    from dataclasses import dataclass, field
    from typing import Optional

    @dataclass
    class RadixNode:
        """
        A node in the RadixAttention trie.

        In a real implementation, kv_ptr would be a reference to a GPU
        memory block. Here we use a placeholder bytes object to represent
        the serialised KV tensors.
        """
        token_ids: list[int]      # edge label (tokens along this edge)
        kv_data: Optional[bytes]  # cached KV tensors (None if not yet materialized)
        children: dict[int, "RadixNode"] = field(default_factory=dict)
        ref_count: int = 0        # how many active requests are using this node
        last_access_time: float = 0.0

        def is_leaf(self) -> bool:
            return len(self.children) == 0

    class RadixTree:
        """
        Minimal RadixAttention trie for prefix caching.
        Supports insert, longest-prefix lookup, and LRU eviction.
        """

        def __init__(self):
            self.root = RadixNode(token_ids=[], kv_data=None)
            self._clock = 0.0

        def _tick(self) -> float:
            self._clock += 1.0
            return self._clock

        def insert(self, token_ids: list[int], kv_data: bytes) -> None:
            """
            Insert a fully prefilled sequence into the trie.
            Splits existing edges as needed (standard radix tree insertion).
            """
            node = self.root
            idx = 0  # position in token_ids

            while idx < len(token_ids):
                first_token = token_ids[idx]

                if first_token not in node.children:
                    # No matching child — create a new leaf.
                    new_node = RadixNode(
                        token_ids=token_ids[idx:],
                        kv_data=kv_data,  # store full KV data at leaf
                        last_access_time=self._tick(),
                    )
                    node.children[first_token] = new_node
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
                else:
                    # Partial match — split the edge.
                    # Create an intermediate node with the common prefix.
                    split_node = RadixNode(
                        token_ids=child.token_ids[:lcp],
                        kv_data=None,  # split node has no KV data of its own
                        last_access_time=self._tick(),
                    )
                    # Old child gets the suffix as its new edge label.
                    child.token_ids = child.token_ids[lcp:]
                    split_node.children[child.token_ids[0]] = child
                    # New leaf gets the remaining tokens.
                    remainder_node = RadixNode(
                        token_ids=token_ids[idx + lcp:],
                        kv_data=kv_data,
                        last_access_time=self._tick(),
                    )
                    split_node.children[token_ids[idx + lcp]] = remainder_node
                    # Attach split_node to parent.
                    node.children[first_token] = split_node
                    return

        def match_prefix(self, token_ids: list[int]) -> tuple[int, Optional[bytes]]:
            """
            Find the longest cached prefix of token_ids.
            Returns (length_matched, kv_data_of_deepest_matched_node).
            """
            node = self.root
            idx = 0
            best_len = 0
            best_kv = None

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

                idx += lcp

                if child.kv_data is not None:
                    best_len = idx
                    best_kv = child.kv_data

                if lcp < len(child.token_ids):
                    break  # partial match — descent stops here

                node = child

            return best_len, best_kv

    # --- exercise the trie with tiny fixtures ---
    tree = RadixTree()

    # Session 1: message A followed by B (A = [1,2,3], A+B = [1,2,3,4,5]).
    tree.insert([1, 2, 3], kv_data=b"kv_A")
    tree.insert([1, 2, 3, 4, 5], kv_data=b"kv_A_B")

    # A brand-new request sharing only message A should match the first 3 tokens.
    matched_len, kv = tree.match_prefix([1, 2, 3, 9, 9])
    print(f"match_prefix([1,2,3,9,9]) -> matched_len={matched_len}, kv={kv}")
    assert matched_len == 3
    assert kv == b"kv_A"

    # A request replaying the full A+B session should match all 5 tokens.
    matched_len2, kv2 = tree.match_prefix([1, 2, 3, 4, 5])
    print(f"match_prefix([1,2,3,4,5]) -> matched_len={matched_len2}, kv={kv2}")
    assert matched_len2 == 5
    assert kv2 == b"kv_A_B"

    # A completely disjoint prompt should have zero cached prefix.
    matched_len3, kv3 = tree.match_prefix([7, 8, 9])
    assert matched_len3 == 0
    assert kv3 is None

    # Inserting a diverging continuation ([1,2,3,4,9]) should split the
    # existing "4,5" edge at the common prefix "4".
    tree.insert([1, 2, 3, 4, 9], kv_data=b"kv_A_C")
    matched_len4, kv4 = tree.match_prefix([1, 2, 3, 4, 9])
    assert matched_len4 == 5
    assert kv4 == b"kv_A_C"
    # The original A+B path must still resolve correctly after the split.
    matched_len5, kv5 = tree.match_prefix([1, 2, 3, 4, 5])
    assert matched_len5 == 5
    assert kv5 == b"kv_A_B"


def block_apc_allocator():
    # content lines ~236-307 (verbatim)
    class APCBlockAllocator:
        def __init__(self, num_blocks: int, block_size: int):
            self.block_size = block_size
            self.free_blocks: list[int] = list(range(num_blocks))
            # Map from content hash -> physical block id
            self.prefix_cache: dict[int, int] = {}
            # LRU ordering: (timestamp, block_id) — use a sortedcontainer in practice
            self.lru_order: list[tuple[float, int]] = []
            self._time = 0

        def _compute_block_hash(self, prev_hash: int, token_ids: list[int]) -> int:
            """Chain the previous block hash with current token IDs."""
            import hashlib
            data = prev_hash.to_bytes(8, 'little') + bytes(token_ids)
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

            for i, block_tokens in enumerate(token_ids_blocks):
                h = self._compute_block_hash(prev_hash, block_tokens)

                if h in self.prefix_cache:
                    # Cache hit: reuse this physical block.
                    phys_id = self.prefix_cache[h]
                    block_ids.append(phys_id)
                    num_cached += 1
                    self._touch(phys_id)
                else:
                    # Cache miss: allocate a fresh physical block.
                    # Evict if necessary.
                    if not self.free_blocks:
                        self._evict_lru()
                    phys_id = self.free_blocks.pop()
                    # We will compute KV for this block during prefill.
                    # After prefill completes, insert into cache.
                    self.prefix_cache[h] = phys_id
                    self._touch(phys_id)
                    block_ids.append(phys_id)

                prev_hash = h

            return block_ids, num_cached

        def _touch(self, block_id: int):
            """Update LRU timestamp for a block."""
            self._time += 1
            self.lru_order.append((self._time, block_id))

        def _evict_lru(self):
            """Remove the least-recently-used cached block."""
            self.lru_order.sort()
            while self.lru_order:
                _, block_id = self.lru_order.pop(0)
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
    assert len(allocator.prefix_cache) == 2

    # Second request: shares the first block (same prefix), diverges on the second.
    blocks_req2 = [[1, 2, 3, 4], [9, 9, 9, 9]]
    ids2, cached2 = allocator.allocate_or_reuse(blocks_req2)
    print(f"req2 block_ids={ids2}, num_cached={cached2}")
    assert cached2 == 1                    # only the first block was a hit
    assert ids2[0] == ids1[0]              # reused the same physical block
    assert ids2[1] != ids1[1]              # second block is a fresh allocation

    # Fill remaining capacity and force eviction: num_blocks=4, 3 already used
    # (blocks from req1 + the one new block from req2), 1 free slot left.
    blocks_req3 = [[100, 101, 102, 103]]   # brand-new content -> cache miss
    ids3, cached3 = allocator.allocate_or_reuse(blocks_req3)
    assert cached3 == 0
    assert allocator.free_blocks == []     # pool now fully allocated (4 blocks)

    # A fourth request with entirely new content must trigger LRU eviction
    # since the free list is empty.
    blocks_req4 = [[200, 201, 202, 203]]
    cache_size_before = len(allocator.prefix_cache)
    ids4, cached4 = allocator.allocate_or_reuse(blocks_req4)
    print(f"req4 block_ids={ids4}, num_cached={cached4}, "
          f"cache_size before={cache_size_before} after={len(allocator.prefix_cache)}")
    assert cached4 == 0
    # Eviction removed exactly one LRU entry before inserting the new one,
    # so the cache size is unchanged (one out, one in).
    assert len(allocator.prefix_cache) == cache_size_before
    assert allocator.free_blocks == []     # the freed slot was immediately consumed


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
        "  #5 (python) - SKIP(out-of-scope): CPU-safe but not one of the 2 assigned blocks\n"
        "  #6 (python) - SKIP(out-of-scope): CPU-safe but not one of the 2 assigned blocks"
    )


if __name__ == "__main__":
    main()
