"""
Runs the CPU-runnable Python blocks from:
    content/07-inference-serving/01-anatomy-inference.md

Blocks tested (verbatim from the chapter, with tiny fixtures/glue added):
    - block #0 (line ~29): generate_naive()          -- naive O(T^2) decode loop
    - block #1 (line ~78): CachedSelfAttention        -- from-scratch KV-cached attention
    - block #2 (line ~153): kv_cache_bytes()          -- trivially CPU-safe pure arithmetic,
                                                          included as a bonus even though the
                                                          task listed it as a default-skip fragment
    - block #4 (line ~303): measure_streaming_latency() -- TTFT/TPOT timing wrapper

Blocks explicitly SKIPPED (left un-executed, per the task's CPU-runnability heuristics):
    - block #3 (line ~220): bench_decode()  -- SKIP(needs-gpu): hardcodes device="cuda",
      calls torch.cuda.synchronize(); no CPU-meaningful equivalent without rewriting the
      book's logic, which would defeat the point of the demo (it's specifically measuring
      GPU memory-bandwidth-bound behavior).
    - block #5 (line ~382): generate()      -- SKIP(needs-gpu + needs-hf-model): hardcodes
      device="cuda" and expects a HF-style model/tokenizer with past_key_values / use_cache
      semantics that we cannot fabricate faithfully offline without rewriting the very
      KV-cache mechanics being demonstrated.
"""

import time

import torch
import torch.nn.functional as F


# =====================================================================
# Block #0 (line ~29 in the chapter) -- verbatim
# =====================================================================

@torch.no_grad()
def generate_naive(model, input_ids, max_new_tokens):
    """
    The simplest correct autoregressive decoder: re-run the FULL forward
    pass on the entire growing sequence at every step. Correct, but O(T^2)
    in attention work because we recompute K and V for all old tokens
    every single step. This is what the KV cache fixes.
    """
    for _ in range(max_new_tokens):
        # logits over the WHOLE sequence; we only need the last position
        logits = model(input_ids).logits        # [B, seq_len, vocab]
        next_token = logits[:, -1, :].argmax(-1, keepdim=True)  # greedy
        input_ids = torch.cat([input_ids, next_token], dim=1)   # grow seq
    return input_ids


# =====================================================================
# Block #1 (line ~78 in the chapter) -- verbatim
# =====================================================================

class CachedSelfAttention(torch.nn.Module):
    """
    Single-head causal self-attention with an explicit KV cache.
    Strips away batching/multi-head bookkeeping to expose the mechanism.
    """
    def __init__(self, d_model, d_head):
        super().__init__()
        self.Wq = torch.nn.Linear(d_model, d_head, bias=False)
        self.Wk = torch.nn.Linear(d_model, d_head, bias=False)
        self.Wv = torch.nn.Linear(d_model, d_head, bias=False)
        self.scale = d_head ** -0.5
        self.k_cache = None  # [seq_so_far, d_head]
        self.v_cache = None

    def reset(self):
        self.k_cache = None
        self.v_cache = None

    def forward(self, x):
        # x: [n_new_tokens, d_model].
        #   - prefill: n_new_tokens == S (the whole prompt)
        #   - decode:  n_new_tokens == 1 (just the last token)
        q = self.Wq(x)                      # [n_new, d_head]
        k = self.Wk(x)                      # [n_new, d_head]
        v = self.Wv(x)                      # [n_new, d_head]

        # Append the new keys/values to the cache (the whole trick).
        if self.k_cache is None:
            self.k_cache, self.v_cache = k, v
        else:
            self.k_cache = torch.cat([self.k_cache, k], dim=0)
            self.v_cache = torch.cat([self.v_cache, v], dim=0)

        # Attend the new query(ies) against ALL cached keys/values.
        scores = (q @ self.k_cache.T) * self.scale  # [n_new, seq_so_far]

        # Causal mask: needed during prefill (n_new = S) so that token i
        # cannot see j > i. During decode (n_new = 1) the single new query
        # is allowed to see everything in the cache, so no mask is required.
        if q.shape[0] > 1:
            seq = self.k_cache.shape[0]
            past = seq - q.shape[0]
            i = torch.arange(q.shape[0]).unsqueeze(1) + past
            j = torch.arange(seq).unsqueeze(0)
            scores = scores.masked_fill(j > i, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        return attn @ self.v_cache          # [n_new, d_head]


# =====================================================================
# Block #2 (line ~153 in the chapter) -- verbatim
# Task labeled this a default-skip "fragment", but it is pure arithmetic
# with no GPU/network dependency, so it is trivially CPU-safe -- run it.
# =====================================================================

def kv_cache_bytes(batch, seq_len, n_layers, n_kv_heads, head_dim, dtype_bytes=2):
    """KV-cache size in bytes. Factor 2 = keys AND values."""
    return 2 * batch * seq_len * n_layers * n_kv_heads * head_dim * dtype_bytes


# =====================================================================
# Block #4 (line ~303 in the chapter) -- verbatim
# =====================================================================

def measure_streaming_latency(stream_fn, prompt):
    """
    Wrap a streaming generation call and report TTFT, mean TPOT, and the
    per-token ITL series. `stream_fn(prompt)` must yield one token at a time.
    """
    import time
    t_start = time.perf_counter()
    t_prev = None
    ttft = None
    itls = []                      # inter-token latencies (seconds)
    n_tokens = 0
    for _token in stream_fn(prompt):
        now = time.perf_counter()
        if ttft is None:
            ttft = now - t_start   # arrival -> first token
        else:
            itls.append(now - t_prev)
        t_prev = now
        n_tokens += 1
    tpot = sum(itls) / len(itls) if itls else float("nan")
    return {
        "ttft_ms": ttft * 1e3,
        "tpot_ms": tpot * 1e3,                # mean inter-token latency
        "p99_itl_ms": sorted(itls)[int(0.99 * len(itls)) - 1] * 1e3 if itls else None,
        "output_tokens": n_tokens,
        "decode_tps": (n_tokens - 1) / sum(itls) if itls else float("nan"),
    }


# =====================================================================
# Glue / fixtures + actual execution of each block
# =====================================================================

class TinyLMOutput:
    """Minimal stand-in for a HF-style CausalLM output (only `.logits`)."""
    def __init__(self, logits):
        self.logits = logits


class TinyLM(torch.nn.Module):
    """
    Tiny toy LM: embed -> linear -> vocab logits. Exists purely so that
    `generate_naive` (block #0) has something with a `.logits`-returning
    `__call__` to drive, on CPU, in a few milliseconds.
    """
    def __init__(self, vocab_size=32, d_model=16):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, d_model)
        self.proj = torch.nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        return TinyLMOutput(self.proj(x))


def test_generate_naive():
    torch.manual_seed(0)
    model = TinyLM(vocab_size=32, d_model=16).eval()
    prompt_len = 6
    input_ids = torch.randint(0, 32, (1, prompt_len))
    max_new_tokens = 5

    out = generate_naive(model, input_ids, max_new_tokens)

    assert out.shape == (1, prompt_len + max_new_tokens), out.shape
    # first prompt_len tokens must be unchanged (naive loop only appends)
    assert torch.equal(out[:, :prompt_len], input_ids)
    print(f"[block #0] generate_naive: grew {prompt_len} -> {out.shape[1]} tokens OK")


def test_cached_self_attention():
    torch.manual_seed(0)
    d_model, d_head = 16, 8
    attn = CachedSelfAttention(d_model, d_head).eval()

    # ---- Prefill: whole prompt at once (S rows) ----
    S = 5
    prompt = torch.randn(S, d_model)
    with torch.no_grad():
        out_prefill = attn(prompt)
    assert out_prefill.shape == (S, d_head), out_prefill.shape
    assert attn.k_cache.shape == (S, d_head)
    assert attn.v_cache.shape == (S, d_head)

    # ---- Decode: one new token at a time, cache should grow by 1 each step ----
    for step in range(3):
        new_tok = torch.randn(1, d_model)
        with torch.no_grad():
            out_decode = attn(new_tok)
        assert out_decode.shape == (1, d_head), out_decode.shape
        assert attn.k_cache.shape == (S + step + 1, d_head)
        assert attn.v_cache.shape == (S + step + 1, d_head)

    # Sanity check: causal masking during prefill means position 0's output
    # depends only on token 0. Rerun a fresh instance with a modified
    # LAST prompt token and confirm position-0 output is unaffected.
    torch.manual_seed(0)
    attn2 = CachedSelfAttention(d_model, d_head).eval()
    prompt2 = prompt.clone()
    prompt2[-1] = torch.randn(d_model)  # perturb only the last token
    with torch.no_grad():
        out2 = attn2(prompt2)
    assert torch.allclose(out2[0], out_prefill[0], atol=1e-6), (
        "causal mask violated: position 0 output changed when a LATER "
        "token was perturbed"
    )
    print("[block #1] CachedSelfAttention: prefill + 3 decode steps OK, "
          "causal masking verified")


def test_kv_cache_bytes():
    # Llama-3-8B-style config from the chapter's worked example.
    total_bytes = kv_cache_bytes(batch=1, seq_len=8192, n_layers=32,
                                  n_kv_heads=8, head_dim=128, dtype_bytes=2)
    gib = total_bytes / 2**30
    # Chapter claims "~1.0 GiB per 8k-token sequence" -- verify.
    assert abs(gib - 1.0) < 1e-9, f"expected ~1.0 GiB, got {gib}"
    print(f"[block #2] kv_cache_bytes: {gib:.2f} GiB per 8k-token sequence OK "
          f"(matches chapter's worked example)")


def test_measure_streaming_latency():
    def fake_stream_fn(prompt):
        # Yields a handful of tokens with tiny, deterministic-ish delays so
        # the timing wrapper's own logic (TTFT/TPOT/ITL bookkeeping) runs
        # for real, without any network or GPU dependency.
        for _ in range(6):
            time.sleep(0.001)
            yield "tok"

    result = measure_streaming_latency(fake_stream_fn, prompt="hello")

    assert result["output_tokens"] == 6
    assert result["ttft_ms"] > 0
    assert result["tpot_ms"] > 0
    assert result["p99_itl_ms"] is not None and result["p99_itl_ms"] > 0
    assert result["decode_tps"] > 0
    print(f"[block #4] measure_streaming_latency: {result}")


if __name__ == "__main__":
    test_generate_naive()
    test_cached_self_attention()
    test_kv_cache_bytes()
    test_measure_streaming_latency()

    print("\nSKIP(needs-gpu): block #3 bench_decode() -- hardcodes device='cuda' "
          "and torch.cuda.synchronize(); demonstrates GPU memory-bandwidth-bound "
          "behavior that has no faithful CPU equivalent.")
    print("SKIP(needs-gpu + needs-hf-model): block #5 generate() -- hardcodes "
          "device='cuda' and requires a real HF-style model/tokenizer with "
          "past_key_values/use_cache semantics.")

    print("\nAll CPU-runnable blocks executed successfully.")
