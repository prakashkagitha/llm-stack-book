"""MuonClip / QK-clip (Kimi K2, 2025): after each optimizer step, rescale W_Q and
the shared W_K so per-head attention logits stay under `tau`. This is the
stability fix that made Muon work at scale. GQA-aware: W_K is shared across a
group of query heads, so we scale it by the group's worst logit.
"""
import torch


@torch.no_grad()
def qk_clip_(model, max_logits_per_head, tau: float = 100.0):
    for layer_idx, block in enumerate(model.blocks):
        if layer_idx not in max_logits_per_head:
            continue
        s_max = max_logits_per_head[layer_idx]           # (n_heads,)
        attn = block.attn
        hd = attn.head_dim
        group = attn.n_heads // attn.n_kv_heads          # q-heads per kv-head (=4)

        # (1) Per-query-head scale on W_Q.
        for h in range(attn.n_heads):
            if float(s_max[h]) <= tau:
                continue
            eta = (tau / float(s_max[h])) ** 0.5
            attn.wq.weight[h * hd:(h + 1) * hd].mul_(eta)

        # (2) Per-KV-head scale on the SHARED W_K, using the group's worst logit.
        for kv in range(attn.n_kv_heads):
            s_grp = float(s_max[kv * group:(kv + 1) * group].max())
            if s_grp <= tau:
                continue
            eta = (tau / s_grp) ** 0.5
            attn.wk.weight[kv * hd:(kv + 1) * hd].mul_(eta)
