"""
Executable test for content/10-multimodal-and-arch/05-unified-any-to-any.md

Runs the chapter's CPU-runnable Python blocks, in order, concatenated into one
module, with minimal glue/fixtures so each block's own logic actually executes.

Blocks tested:
  - block #0 (~line 108): qk_norm, modality_z_loss, ChameleonBlock
  - block #1 (~line 248): transfusion_attention_mask (+ the chapter's own inline
    asserts / self-test, run verbatim)
  - block #2 (~line 333): data_mix schedule dict
  - block #4 (~line 400): ModalityAwareMoE
  - block #5 (~line 537): Batch / UnifiedModel minimal any-to-any training step

Blocks skipped:
  - block #3 (~line 373): non-Python (AnyGPT prompt-format text listing), no code
    to execute.

Bug found & fixed in the book (mirrored here): in the UnifiedModel.forward
diffusion-loss computation (block #5), `target_velocity` is built from
`batch.noise - batch.image_patches`, both of which are already image-only
tensors of shape (B, T_img, D_patch). The book then re-filtered this
already-image-only tensor with `target_velocity[batch.is_image.reshape(-1)]`,
a full-sequence-length (B*T_total) boolean mask — a shape mismatch that raises
at runtime. The extra filter line is wrong/superfluous and has been removed
both in the chapter and in this test.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

torch.manual_seed(0)


# =====================================================================
# Block #0 (~line 108): QK-Norm, modality z-loss, ChameleonBlock
# =====================================================================

def qk_norm(q: torch.Tensor, k: torch.Tensor, eps: float = 1e-6):
    """
    Query-key normalisation as used in Chameleon.
    Normalises each query and key vector independently before dot-product attention.

    Args:
        q: (batch, heads, seq_len, head_dim)
        k: (batch, heads, seq_len, head_dim)
    Returns:
        q_norm, k_norm: same shape as inputs, unit-norm along head_dim
    """
    q_norm = q / (q.norm(dim=-1, keepdim=True) + eps)
    k_norm = k / (k.norm(dim=-1, keepdim=True) + eps)
    return q_norm, k_norm


def modality_z_loss(logits: torch.Tensor,
                    text_mask: torch.Tensor,
                    image_mask: torch.Tensor,
                    alpha: float = 1e-4) -> torch.Tensor:
    """
    Auxiliary z-loss that penalises large logit magnitudes, computed
    separately for text-position and image-position outputs.

    This stabilises training by preventing the model from using very
    large logits for either modality, which can cause softmax saturation.

    Args:
        logits:     (batch, seq_len, vocab_size)  — raw pre-softmax logits
        text_mask:  (batch, seq_len)               — True for text positions
        image_mask: (batch, seq_len)               — True for image positions
        alpha:      weight of the auxiliary loss
    """
    def z_loss_for_mask(mask):
        # log-sum-exp of logits; penalise if large
        lse = torch.logsumexp(logits, dim=-1)          # (batch, seq_len)
        masked_lse = lse[mask]
        return (masked_lse ** 2).mean()

    z_text  = z_loss_for_mask(text_mask)
    z_image = z_loss_for_mask(image_mask)
    return alpha * (z_text + z_image)


# ---- Minimal unified forward pass skeleton ----

class ChameleonBlock(torch.nn.Module):
    """Simplified Chameleon transformer block with QK-Norm."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj   = torch.nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = torch.nn.Linear(d_model, d_model, bias=False)
        self.v_proj   = torch.nn.Linear(d_model, d_model, bias=False)
        self.o_proj   = torch.nn.Linear(d_model, d_model, bias=False)
        self.norm1    = torch.nn.RMSNorm(d_model)
        self.norm2    = torch.nn.RMSNorm(d_model)
        # Per-head learnable scale factors for QK-Norm
        self.q_scale  = torch.nn.Parameter(torch.ones(n_heads, self.head_dim))
        self.k_scale  = torch.nn.Parameter(torch.ones(n_heads, self.head_dim))
        ffn_dim = 4 * d_model
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, ffn_dim, bias=False),
            torch.nn.SiLU(),
            torch.nn.Linear(ffn_dim, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor = None):
        B, T, D = x.shape
        H, Dh   = self.n_heads, self.head_dim

        # Pre-norm attention
        h = self.norm1(x)
        q = self.q_proj(h).view(B, T, H, Dh).transpose(1, 2)  # (B, H, T, Dh)
        k = self.k_proj(h).view(B, T, H, Dh).transpose(1, 2)
        v = self.v_proj(h).view(B, T, H, Dh).transpose(1, 2)

        # QK-Norm: normalise then rescale with learned per-head scale
        q = F.normalize(q, dim=-1) * self.q_scale.unsqueeze(0).unsqueeze(2)
        k = F.normalize(k, dim=-1) * self.k_scale.unsqueeze(0).unsqueeze(2)

        # Standard scaled dot-product attention
        attn_out = F.scaled_dot_product_attention(q, k, v,
                                                   attn_mask=causal_mask,
                                                   is_causal=(causal_mask is None))
        attn_out = attn_out.transpose(1, 2).reshape(B, T, D)
        x = x + self.o_proj(attn_out)

        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))
        return x


def test_block0():
    # --- exercise qk_norm ---
    B, H, T, Dh = 2, 2, 5, 4
    q = torch.randn(B, H, T, Dh)
    k = torch.randn(B, H, T, Dh)
    q_n, k_n = qk_norm(q, k)
    assert q_n.shape == q.shape and k_n.shape == k.shape
    norms = q_n.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    # --- exercise modality_z_loss ---
    vocab = 16
    logits = torch.randn(B, T, vocab)
    text_mask  = torch.tensor([[True, True, True, False, False]] * B)
    image_mask = ~text_mask
    zloss = modality_z_loss(logits, text_mask, image_mask, alpha=1e-4)
    assert zloss.dim() == 0 and zloss.item() >= 0.0

    # --- exercise ChameleonBlock ---
    d_model, n_heads = 8, 2
    block = ChameleonBlock(d_model, n_heads)
    x = torch.randn(B, T, d_model)
    out = block(x)
    assert out.shape == (B, T, d_model)
    assert torch.isfinite(out).all()
    print("[block0] qk_norm / modality_z_loss / ChameleonBlock OK")


# =====================================================================
# Block #1 (~line 248): Transfusion block-causal attention mask
# Run verbatim, including the chapter's own inline self-test asserts.
# =====================================================================

def transfusion_attention_mask(token_types: list, seq_len: int) -> torch.Tensor:
    """
    Build the Transfusion attention mask.

    token_types: list of 'text' or 'image' for each position.
    Returns a boolean mask (True = attend) of shape (seq_len, seq_len).

    Rules:
      - Text position i attends to all positions j <= i (causal).
      - Image position i attends to: all text positions before the image block,
        all image positions in the same block (bidirectional within-image),
        nothing after position i (causal across blocks).
    """
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)

    # Assign each image position to its block index
    image_block = {}
    current_block = 0
    in_image = False
    for idx, t in enumerate(token_types):
        if t == 'image' and not in_image:
            current_block += 1
            in_image = True
        elif t == 'text':
            in_image = False
        if t == 'image':
            image_block[idx] = current_block

    for i, ti in enumerate(token_types):
        for j, tj in enumerate(token_types):
            if ti == 'text':
                # Text attends causally to every position at or before it.
                if j <= i:
                    mask[i, j] = True
            elif ti == 'image':
                same_block = (tj == 'image'
                              and image_block[j] == image_block[i])
                if same_block:
                    # Bidirectional WITHIN the image block: attend to every
                    # position in the same block, INCLUDING j > i. This is the
                    # whole point of Transfusion's block-causal mask.
                    mask[i, j] = True
                elif j <= i:
                    # Causal to everything before the block: earlier text
                    # tokens and earlier image blocks.
                    mask[i, j] = True
    return mask


def test_block1():
    # Quick test: text-image-text sequence. Image block occupies positions 3,4,5,6.
    types = ['text'] * 3 + ['image'] * 4 + ['text'] * 2
    mask  = transfusion_attention_mask(types, len(types))

    # Attention INSIDE the image block is bidirectional (attends both ways):
    assert mask[3, 6] and mask[6, 3], "image block must be bidirectional"
    assert mask[4, 6] and mask[6, 4], "image block must be bidirectional"
    # Image tokens still attend causally to earlier text (positions 0,1,2) ...
    assert mask[3, 0] and mask[6, 2], "image attends earlier text"
    # ... but NOT to text that comes AFTER the block (positions 7,8):
    assert not mask[3, 7] and not mask[6, 8], "image must not see future text"
    # Text stays purely causal: no attending to future image or future text.
    assert not mask[2, 3], "text must not attend to future image tokens"
    assert not mask[0, 1], "text is causal"
    assert mask[7, 3] and mask[8, 6], "later text attends earlier image (causal)"
    print("Transfusion mask OK: bidirectional image block, causal elsewhere")
    print(mask.int())
    return types, mask


# =====================================================================
# Block #2 (~line 333): Data-mixing schedule
# =====================================================================

def test_block2():
    # Example data-mixing schedule (token counts, not document counts)
    data_mix = {
        "text_only":            0.50,  # 50% of token budget
        "image_only":           0.10,  # 10% — unconditional image generation
        "text_then_image":      0.20,  # 20% — T→I generation
        "image_then_text":      0.15,  # 15% — image understanding (captioning, VQA)
        "interleaved_doc":      0.05,  # 5%  — full web-page style documents
    }
    # These proportions are illustrative; actual models tune them empirically.
    assert math.isclose(sum(data_mix.values()), 1.0, rel_tol=1e-9)
    print("[block2] data_mix sums to 1.0:", data_mix)
    return data_mix


# =====================================================================
# Block #4 (~line 400): Modality-aware MoE
# =====================================================================

class ModalityAwareMoE(nn.Module):
    """
    Simplified MoE FFN with a soft prior that encourages image tokens
    to use image-specialist experts and text tokens to use text-specialist experts.

    This is NOT a hard separation — the router can override the prior
    when beneficial, allowing cross-modal expert sharing.
    """

    def __init__(self, d_model: int, d_ff: int, n_experts: int,
                 top_k: int = 2, n_image_experts: int = None):
        super().__init__()
        self.n_experts = n_experts
        self.top_k     = top_k
        # Split experts conceptually: first half text-specialist, second half image-specialist
        self.n_image_experts = n_image_experts or (n_experts // 2)

        # Router: maps each token to a distribution over experts
        self.router = nn.Linear(d_model, n_experts, bias=False)

        # Expert FFNs
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff, bias=False),
                nn.SiLU(),
                nn.Linear(d_ff, d_model, bias=False),
            )
            for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor,
                is_image_token: torch.Tensor,   # bool (batch*seq,)
                prior_strength: float = 0.1) -> torch.Tensor:
        """
        x: (B*T, d_model) — flattened token sequence
        is_image_token: (B*T,) boolean mask
        prior_strength: how strongly to nudge routing toward modality-specialist experts
        """
        B = x.shape[0]
        router_logits = self.router(x)  # (B*T, n_experts)

        # Build a soft prior: image tokens get a bonus to image-specialist experts
        prior = torch.zeros_like(router_logits)
        text_expert_range  = slice(0, self.n_experts - self.n_image_experts)
        image_expert_range = slice(self.n_experts - self.n_image_experts, self.n_experts)
        prior[is_image_token,  image_expert_range] =  prior_strength
        prior[~is_image_token, text_expert_range]  =  prior_strength
        router_logits = router_logits + prior

        # Top-k routing
        scores, indices = router_logits.topk(self.top_k, dim=-1)  # (B*T, top_k)
        weights = F.softmax(scores, dim=-1)                        # normalise selected experts

        # Dispatch: accumulate expert outputs
        out = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = indices[:, k]          # (B*T,) — which expert for this slot
            w          = weights[:, k:k+1]      # (B*T, 1)
            for e in range(self.n_experts):
                mask = (expert_idx == e)
                if mask.any():
                    out[mask] = out[mask] + w[mask] * self.experts[e](x[mask])
        return out


def test_block4():
    d_model, d_ff, n_experts, top_k = 8, 16, 4, 2
    moe = ModalityAwareMoE(d_model, d_ff, n_experts, top_k=top_k, n_image_experts=2)
    N = 6
    x = torch.randn(N, d_model)
    is_image_token = torch.tensor([True, True, False, False, False, True])
    out = moe(x, is_image_token, prior_strength=0.1)
    assert out.shape == (N, d_model)
    assert torch.isfinite(out).all()
    print("[block4] ModalityAwareMoE forward OK, out shape:", tuple(out.shape))
    return moe


# =====================================================================
# Block #5 (~line 537): Minimal any-to-any (Transfusion-style) training step
#
# BUG FIX mirrored from the chapter: the original book code re-filtered
# `target_velocity` (already image-only, shape (B*T_img, D_patch)) with a
# full-sequence-length boolean mask `batch.is_image.reshape(-1)`
# (length B*T_total), which is a shape mismatch and crashes at runtime.
# That extra filter line has been removed (see fix in the .md).
# =====================================================================

@dataclass
class Batch:
    """A mixed-modal training batch."""
    input_ids:        torch.Tensor   # (B, T_total) — text token ids at text positions (dummy elsewhere)
    text_labels:      torch.Tensor   # (B, T_total) — text ids at text positions, -100 at image positions
    image_patches:    torch.Tensor   # (B, T_img, D_patch)  — continuous patch embeddings
    image_labels:     torch.Tensor   # (B, T_img, D_patch)  — clean patch targets
    noise:            torch.Tensor   # (B, T_img, D_patch)  — sampled ε
    t:                torch.Tensor   # (B,)          — diffusion timestep in [0, 1]
    attention_mask:   torch.Tensor   # (T_total, T_total)   — block-causal mask
    is_image:         torch.Tensor   # (B, T_total) bool — marks image positions


class UnifiedModel(nn.Module):
    def __init__(self, transformer, text_head, diffusion_head,
                 d_model: int, d_patch: int, vocab_size: int):
        super().__init__()
        self.transformer    = transformer    # shared trunk
        self.text_embed     = nn.Embedding(vocab_size, d_model)
        self.patch_proj     = nn.Linear(d_patch, d_model)   # image patch → d_model
        self.text_head      = text_head      # (d_model → vocab_size)
        self.diffusion_head = diffusion_head # (d_model + 1 → d_patch)  (+1 for timestep)

    def embed_sequence(self, text_ids: torch.Tensor,
                       noisy_patches: torch.Tensor,
                       is_image: torch.Tensor) -> torch.Tensor:
        """
        Build a single (B, T_total, d_model) embedding tensor by interleaving
        text embeddings and projected patch embeddings.
        """
        B, T = is_image.shape
        out  = torch.zeros(B, T, self.text_embed.embedding_dim,
                           device=text_ids.device)

        # Text positions
        out[~is_image] = self.text_embed(text_ids[~is_image.squeeze()])

        # Image positions: project noisy patch to d_model
        out[is_image]  = self.patch_proj(noisy_patches.reshape(-1, noisy_patches.shape[-1]))
        return out

    def forward(self, batch: Batch):
        # 1. Construct noisy image patches  x_t = (1 - t) * clean + t * noise
        t      = batch.t.unsqueeze(-1).unsqueeze(-1)        # (B, 1, 1)
        noisy  = (1 - t) * batch.image_patches + t * batch.noise   # (B, T_img, D_patch)

        # 2. Embed full mixed-modal sequence
        x = self.embed_sequence(batch.input_ids, noisy, batch.is_image)

        # 3. Transformer forward pass with block-causal mask
        h = self.transformer(x, attention_mask=batch.attention_mask)  # (B, T_total, d_model)

        # 4. Text loss — cross-entropy on text positions
        text_logits  = self.text_head(h[~batch.is_image])              # (N_text, vocab)
        text_labels  = batch.text_labels[batch.text_labels != -100]
        lm_loss      = F.cross_entropy(text_logits, text_labels)

        # 5. Diffusion loss — flow-matching velocity on image positions
        h_img        = h[batch.is_image]                               # (N_img, d_model)
        # Concatenate timestep as an extra feature
        t_expanded   = batch.t.repeat_interleave(batch.is_image.sum(-1))  # (N_img,)
        h_with_t     = torch.cat([h_img, t_expanded.unsqueeze(-1)], dim=-1)
        pred_velocity = self.diffusion_head(h_with_t)                  # (N_img, D_patch)

        # Target velocity: (ε - x_0). noise/image_patches are already image-only
        # tensors of shape (B, T_img, D_patch), so this reshape is already aligned
        # with h_img/pred_velocity — no further masking is needed.
        # [BUG FIX] book originally had:
        #   target_velocity = target_velocity[batch.is_image.reshape(-1)]
        # which re-filters an already image-only tensor with a full-sequence
        # mask (length B*T_total vs B*T_img rows) — shape mismatch, crashes.
        target_velocity = (batch.noise - batch.image_patches).reshape(-1, batch.noise.shape[-1])
        flow_loss        = F.mse_loss(pred_velocity, target_velocity)

        # 6. Combine losses
        lam   = 1.0   # balance factor; tune on a small sweep
        total = lm_loss + lam * flow_loss

        return {
            "loss":       total,
            "lm_loss":    lm_loss.detach(),
            "flow_loss":  flow_loss.detach(),
        }


class _DummyTransformer(nn.Module):
    """Minimal stand-in for the (deliberately-omitted, per the chapter's own
    note) transformer trunk — a shared trunk that maps (B,T,d_model) ->
    (B,T,d_model). The chapter explicitly states the real transformer
    implementation is out of scope for this sketch."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        return self.proj(x)


def test_block5(mask_types, block_mask):
    d_model, d_patch, vocab_size = 8, 5, 16

    # Reuse the block-causal mask + token-type layout from block #1's own
    # test sequence: 3 text, 4 image, 2 text -> T_total=9, T_img=4.
    T_total = len(mask_types)
    T_img = sum(1 for t in mask_types if t == 'image')
    is_image_row = torch.tensor([t == 'image' for t in mask_types])

    B = 2
    is_image = is_image_row.unsqueeze(0).repeat(B, 1)  # (B, T_total)

    # input_ids: real token ids at text positions, dummy 0 at image positions.
    input_ids = torch.zeros(B, T_total, dtype=torch.long)
    input_ids[~is_image] = torch.randint(0, vocab_size, ((~is_image).sum().item(),))

    # text_labels: real ids at text positions, -100 (ignore) at image positions,
    # aligned exactly with is_image so the forward pass's boolean filters agree.
    text_labels = torch.full((B, T_total), -100, dtype=torch.long)
    text_labels[~is_image] = torch.randint(0, vocab_size, ((~is_image).sum().item(),))

    image_patches = torch.randn(B, T_img, d_patch)
    image_labels  = image_patches.clone()
    noise         = torch.randn(B, T_img, d_patch)
    t             = torch.rand(B)

    batch = Batch(
        input_ids=input_ids,
        text_labels=text_labels,
        image_patches=image_patches,
        image_labels=image_labels,
        noise=noise,
        t=t,
        attention_mask=block_mask,   # reused verbatim from block #1
        is_image=is_image,
    )

    transformer    = _DummyTransformer(d_model)
    text_head      = nn.Linear(d_model, vocab_size)
    diffusion_head = nn.Linear(d_model + 1, d_patch)

    model = UnifiedModel(transformer, text_head, diffusion_head,
                          d_model=d_model, d_patch=d_patch, vocab_size=vocab_size)

    out = model(batch)
    assert set(out.keys()) == {"loss", "lm_loss", "flow_loss"}
    for k, v in out.items():
        assert torch.isfinite(v).all(), f"{k} is not finite"
    assert out["loss"].dim() == 0
    print("[block5] UnifiedModel forward OK:",
          {k: round(v.item(), 4) for k, v in out.items()})


if __name__ == "__main__":
    test_block0()
    mask_types, block_mask = test_block1()
    test_block2()
    test_block4()
    test_block5(mask_types, block_mask)
    print("\nAll unified any-to-any chapter blocks executed successfully.")
