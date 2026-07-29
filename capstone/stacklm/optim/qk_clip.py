"""MuonClip / QK-clip (Kimi K2, 2025), adapted to Stack-100M's QK-norm (Ch. 14.6).

The clip must act on whatever parameter actually controls the attention-logit
scale, and that depends on the architecture:

* **QK-norm ON (Stack-100M's default).** `q = gamma_q * RMSNorm(W_Q x)` is
  invariant to any rescaling of `W_Q`, so clipping the projections is a
  mathematical no-op. The learned RMSNorm gains are the only free scale, so we
  clip `q_norm.weight` / `k_norm.weight`. Because Stack-100M shares ONE gain
  vector of length `head_dim` across all heads of a layer, the clip's
  granularity is the layer (we use the layer's worst head).
* **QK-norm OFF (Kimi K2's configuration).** Then `W_Q`/`W_K` *are* the scale,
  and we use Kimi's per-head clip -- GQA-aware, so each shared `W_K` slice is
  scaled once by its group's worst logit rather than once per query head.

Attribute names (`n_heads`, `n_kv_heads`, `head_dim`, `groups`) match
`stacklm.model.attention.Attention`.
"""
import torch


@torch.no_grad()
def qk_clip_(model, max_logits, tau: float = 100.0) -> int:
    """Rescale attention parameters so recorded logits fall back to <= tau.

    `max_logits[layer_idx]` is a (n_heads,) tensor recorded by the attention
    module (see `Attention.forward(..., record=...)`). Returns the number of
    layers that actually fired, which is worth logging: a rising trigger rate
    late in training means the learning rate is too high.

    A dense (n_layers, n_heads) tensor is also accepted -- Ch. 14.7's `train.py`
    harvests into one, since a preallocated running-max buffer is cheaper than
    rebuilding a dict every step. It is converted here rather than indexed
    directly, because `layer_idx in tensor` is elementwise VALUE membership in
    PyTorch, not key lookup, and would silently do the wrong thing.
    """
    if torch.is_tensor(max_logits):
        max_logits = {i: max_logits[i] for i in range(max_logits.size(0))}
    fired = 0
    for layer_idx, block in enumerate(model.blocks):
        if layer_idx not in max_logits:
            continue
        s_max = max_logits[layer_idx].float()
        attn = block.attn
        if isinstance(attn.q_norm, torch.nn.Identity):
            fired += _clip_projections_(attn, s_max, tau)    # Kimi K2 config
        else:
            fired += _clip_qk_norm_gains_(attn, s_max, tau)  # Stack-100M
    return fired


@torch.no_grad()
def _clip_qk_norm_gains_(attn, s_max, tau: float) -> int:
    """QK-norm path: the gains are the temperature, so clip them.

    Scaling gamma_q and gamma_k each by sqrt(tau/S) multiplies every logit in
    the layer by exactly tau/S (RoPE is a rotation, so it commutes with the
    scale). Stack-100M's gain is shared across heads, so this is a per-LAYER
    clip driven by the worst head; widen the gains to (n_heads, 1, head_dim) if you
    want Kimi's per-head granularity.
    """
    s = float(s_max.max())
    if s <= tau:
        return 0
    eta = (tau / s) ** 0.5
    attn.q_norm.weight.mul_(eta)
    attn.k_norm.weight.mul_(eta)
    return 1


@torch.no_grad()
def _clip_projections_(attn, s_max, tau: float) -> int:
    """No-QK-norm path (Kimi K2): per-query-head W_Q, per-KV-head shared W_K."""
    hd = attn.head_dim
    group = attn.groups                                  # q-heads per kv-head
    fired = 0

    # (1) Per-query-head scale on W_Q.
    for h in range(attn.n_heads):
        if float(s_max[h]) <= tau:
            continue
        eta = (tau / float(s_max[h])) ** 0.5
        attn.wq.weight[h * hd:(h + 1) * hd].mul_(eta)
        fired += 1

    # (2) Per-KV-head scale on the SHARED W_K, using the group's worst logit.
    for kv in range(attn.n_kv_heads):
        s_grp = float(s_max[kv * group:(kv + 1) * group].max())
        if s_grp <= tau:
            continue
        eta = (tau / s_grp) ** 0.5
        attn.wk.weight[kv * hd:(kv + 1) * hd].mul_(eta)
    return fired
