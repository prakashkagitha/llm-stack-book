"""
Runs the CPU-runnable Python code blocks from:
content/13-interview-prep/06-llm-deepdive-questions.md

Tested blocks:
  - block #0 (line ~23):  attention()                    -- causal scaled-dot-product attention
  - block #1 (line ~54):  build_rope_cache()/apply_rope() -- RoPE cache + rotation
  - block #2 (line ~93):  online_softmax_attention()      -- FlashAttention's online-softmax recurrence
  - block #4 (line ~180): dpo_loss()                      -- Direct Preference Optimization loss
  - block #5 (line ~207): grpo_advantages()               -- within-group standardized advantage
  - block #6 (line ~239): quantize_int4_symmetric()/dequantize() -- symmetric per-group int4
  - block #7 (line ~290): moe_layer()                     -- top-k Mixture-of-Experts routing

Skipped blocks (non-python / genuine fragments):
  - block #3 (line ~133): MHA/GQA/MQA ascii diagram        -- SKIP(non-python text/diagram block)
  - block #8 (line ~316): speculative_step()               -- SKIP(fragment: calls undefined
        sample_residual()/sample() helpers and requires draft/target model objects with
        .sample()/.forward() interfaces that the chapter never defines)
"""

import math
import torch


# ---------------------------------------------------------------------------
# Block #0 (line ~23) -- verbatim from the chapter
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Block #1 (line ~54) -- verbatim from the chapter
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Block #2 (line ~93) -- verbatim from the chapter
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Block #4 (line ~180) -- verbatim from the chapter
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Block #5 (line ~207) -- verbatim from the chapter
# ---------------------------------------------------------------------------
def grpo_advantages(rewards):
    # rewards: (group_size,) scalar reward per sampled completion for ONE prompt
    mean = rewards.mean()
    std  = rewards.std() + 1e-6
    return (rewards - mean) / std        # within-group standardized advantage


# ---------------------------------------------------------------------------
# Block #6 (line ~239) -- verbatim from the chapter
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Block #7 (line ~290) -- verbatim from the chapter
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Exercise the blocks with tiny CPU fixtures
# ---------------------------------------------------------------------------

def test_attention():
    torch.manual_seed(0)
    b, h, T, d_head = 1, 2, 4, 8
    q = torch.randn(b, h, T, d_head)
    k = torch.randn(b, h, T, d_head)
    v = torch.randn(b, h, T, d_head)

    out = attention(q, k, v, causal=True)
    assert out.shape == (b, h, T, d_head)
    assert torch.isfinite(out).all()

    # Causal mask sanity check: token 0's output must depend only on v[...,0,:]
    # (all attention weight on itself since nothing else is visible).
    out0 = attention(q[:, :, :1], k[:, :, :1], v[:, :, :1], causal=True)
    assert torch.allclose(out0, v[:, :, :1], atol=1e-5)

    # Non-causal vs causal should differ once T > 1 (unless by fluke).
    out_noncausal = attention(q, k, v, causal=False)
    assert not torch.allclose(out, out_noncausal)

    print("block #0 attention(): OK", tuple(out.shape))


def test_rope():
    torch.manual_seed(3)
    b, h, T, d_head = 1, 2, 5, 8
    cos, sin = build_rope_cache(T, d_head)
    assert cos.shape == (T, d_head // 2)
    assert sin.shape == (T, d_head // 2)

    x = torch.randn(b, h, T, d_head)
    y = apply_rope(x, cos, sin)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()

    # A rotation is norm-preserving within each (even, odd) pair, hence over the
    # whole vector; RoPE must not change the length of q/k.
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)

    # Position 0 has rotation angle 0, so RoPE is the identity there.
    assert torch.allclose(x[:, :, 0], y[:, :, 0], atol=1e-6)

    # Position > 0 must actually rotate (differ from the input).
    assert not torch.allclose(x[:, :, 1], y[:, :, 1])
    print("block #1 build_rope_cache()/apply_rope(): OK", tuple(y.shape))


def test_online_softmax_attention():
    torch.manual_seed(1)
    d = 8
    T = 6
    block_size = 2

    q = torch.randn(d)
    K = torch.randn(T, d)
    V = torch.randn(T, d)

    K_blocks = [K[i:i + block_size] for i in range(0, T, block_size)]
    V_blocks = [V[i:i + block_size] for i in range(0, T, block_size)]

    out = online_softmax_attention(q, K_blocks, V_blocks)
    assert out.shape == (d,)
    assert torch.isfinite(out).all()

    # Correctness check: the online-softmax recurrence must reproduce plain
    # (non-tiled) softmax attention over the same full K, V exactly.
    scores = (K @ q) / d ** 0.5
    weights = F.softmax(scores, dim=-1)
    expected = weights @ V

    assert torch.allclose(out, expected, atol=1e-5), (out, expected)
    print("block #2 online_softmax_attention(): OK, matches plain softmax attention")


def test_dpo_loss():
    torch.manual_seed(2)
    batch = 5
    policy_logp_w = torch.randn(batch)
    policy_logp_l = torch.randn(batch)
    ref_logp_w = torch.randn(batch)
    ref_logp_l = torch.randn(batch)

    loss, chosen_reward, rejected_reward = dpo_loss(
        policy_logp_w, policy_logp_l, ref_logp_w, ref_logp_l, beta=0.1
    )
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert chosen_reward.shape == (batch,)
    assert rejected_reward.shape == (batch,)

    # Sanity: if the policy perfectly separates chosen/rejected relative to the
    # reference (large positive logit), the loss should be near zero.
    big = torch.full((batch,), 10.0)
    zero = torch.zeros(batch)
    easy_loss, _, _ = dpo_loss(big, zero, zero, zero, beta=1.0)
    assert easy_loss.item() < 1e-3

    print("block #4 dpo_loss(): OK, loss =", loss.item())


def test_grpo_advantages():
    torch.manual_seed(4)
    rewards = torch.randn(8)
    adv = grpo_advantages(rewards)
    assert adv.shape == rewards.shape
    assert torch.isfinite(adv).all()
    # Standardized advantages are (approximately) zero-mean.
    assert abs(adv.mean().item()) < 1e-5

    # The single correct answer in a mostly-wrong group gets a large positive
    # advantage; the wrong ones get equal negative advantage.
    verifiable = torch.tensor([1.0, 0.0, 0.0, 0.0])
    va = grpo_advantages(verifiable)
    assert va[0] > 0
    assert (va[1:] < 0).all()
    assert torch.allclose(va[1], va[2]) and torch.allclose(va[2], va[3])
    print("block #5 grpo_advantages(): OK", va.tolist())


def test_quantize_int4():
    torch.manual_seed(5)
    w = torch.randn(4, 256)
    group_size = 128
    q, scale = quantize_int4_symmetric(w, group_size=group_size)

    # Quantized values stay in the signed int4 range [-8, 7].
    assert q.min().item() >= -8 and q.max().item() <= 7
    assert q.dtype == torch.int8

    dq = dequantize(q, scale, group_size=group_size)
    assert dq.shape == w.shape
    assert torch.isfinite(dq).all()

    # Round-trip error is bounded by half a quantization step (scale/2) per group.
    per_group_scale = scale.reshape(-1)
    assert (w - dq).abs().max().item() <= per_group_scale.max().item() * 0.5 + 1e-5
    # Dequant must be a genuine approximation, not a no-op that returns w.
    assert not torch.allclose(w, dq)
    print("block #6 quantize_int4_symmetric()/dequantize(): OK, max err =",
          (w - dq).abs().max().item())


def test_moe_layer():
    torch.manual_seed(6)
    d_model = 8
    n_experts = 4
    n_tokens = 6
    experts = [torch.nn.Linear(d_model, d_model) for _ in range(n_experts)]
    router = torch.nn.Linear(d_model, n_experts)

    x = torch.randn(n_tokens, d_model)
    out = moe_layer(x, experts, router, k=2)
    assert out.shape == (n_tokens, d_model)
    assert torch.isfinite(out).all()

    # Cross-check the routing/combine against an explicit reference computation.
    logits = router(x)
    weights, idx = torch.topk(logits.softmax(-1), 2, dim=-1)
    ref = torch.zeros_like(x)
    for t in range(n_tokens):
        for slot in range(2):
            e = idx[t, slot].item()
            ref[t] += weights[t, slot] * experts[e](x[t:t + 1]).squeeze(0)
    assert torch.allclose(out, ref, atol=1e-5)
    print("block #7 moe_layer(): OK", tuple(out.shape))


if __name__ == "__main__":
    test_attention()
    test_rope()
    test_online_softmax_attention()
    test_dpo_loss()
    test_grpo_advantages()
    test_quantize_int4()
    test_moe_layer()
    print("All tested blocks executed successfully.")
