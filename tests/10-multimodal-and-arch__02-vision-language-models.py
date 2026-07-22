"""
Executable test for content/10-multimodal-and-arch/02-vision-language-models.md

Assembles the chapter's CPU-runnable Python blocks, in the order they appear
in the chapter, into one runnable module.

Blocks tested (chapter order):
    #0 (line ~53)  - TwoLayerMLP / MiniLLaVA / splice_visual_tokens
                     (LLaVA-style projector skeleton) + the chapter's own
                     "Quick shape check" __main__ demo.
    #1 (line ~208) - GatedCrossAttention (Flamingo-style gated cross-attn)
                     + the chapter's own __main__ demo.
    #3 (line ~435) - build_optimizer_stage2 (per-component learning rates)

Blocks skipped:
    #2 (line ~343) - tile_image / encode_tiles_to_visual_tokens: SKIP(fragment
                     + needs-gpu). `encode_tiles_to_visual_tokens` defaults to
                     `device="cuda"` and calls an undefined `preprocess(...)`
                     helper that is never defined anywhere in the chapter, so
                     the block cannot execute standalone even on CPU.
    #4 (line ~497) - encode_document_pages: SKIP(fragment). Depends on block
                     #2's undefined `preprocess` (via `tile_image` /
                     `encode_tiles_to_visual_tokens`) and on an external
                     `tokenizer` object that is never constructed in the
                     chapter; it is presented purely as illustrative glue
                     code, not a standalone runnable example.

Block #0 defines MiniLLaVA and splice_visual_tokens but the chapter's own
__main__ demo only instantiates TwoLayerMLP directly (it deliberately omits
weights/a real CLIP/LLaMA model, per the block's own comment "Quick shape
check (no weights needed)"). To honestly exercise MiniLLaVA and
splice_visual_tokens too (per the "every tested block must actually execute"
rule), this test adds tiny CPU fixtures standing in for the vision encoder
and the LLM (a fake CLIP-shaped vision tower and a fake HF-CausalLM-shaped
decoder) and drives a full MiniLLaVA.forward() call end to end, then also
exercises splice_visual_tokens on a tiny single-example input. transformers
is imported guarded (never called) purely because the book's code imports
CLIPVisionModel / LlamaForCausalLM etc. as type-hint context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import CLIPVisionModel, CLIPImageProcessor
    from transformers import LlamaForCausalLM, LlamaTokenizer
except Exception:
    CLIPVisionModel = CLIPImageProcessor = None
    LlamaForCausalLM = LlamaTokenizer = None

torch.manual_seed(0)


# =============================================================================
# Block #0 (line ~53): Minimal LLaVA-style projector and forward pass
# =============================================================================

class TwoLayerMLP(nn.Module):
    """LLaVA-1.5's MLP connector: Linear -> GELU -> Linear."""
    def __init__(self, vision_dim: int, llm_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, num_patches, vision_dim]
        return self.proj(x)  # -> [batch, num_patches, llm_dim]


class MiniLLaVA(nn.Module):
    """
    Minimal LLaVA-style VLM skeleton.

    Args:
        vision_model: a CLIP ViT that returns last_hidden_state
        projector:    TwoLayerMLP mapping vision_dim -> llm_dim
        llm:          causal decoder-only LLM
        llm_dim:      hidden size of the LLM (e.g. 4096 for LLaMA-7B)
    """
    def __init__(self, vision_model, projector, llm, llm_dim: int):
        super().__init__()
        self.vision_model = vision_model
        self.projector = projector
        self.llm = llm
        self.llm_dim = llm_dim

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        pixel_values: [B, 3, H, W] in CLIP's normalized space
        Returns projected visual embeddings: [B, N_patches, llm_dim]
        """
        with torch.no_grad():  # Stage 1: vision encoder frozen
            vision_out = self.vision_model(pixel_values=pixel_values)
        # vision_out.last_hidden_state: [B, 1 + N_patches, vision_dim]
        # Drop CLS token (index 0), keep only patch tokens
        patch_embeds = vision_out.last_hidden_state[:, 1:, :]
        # Project into LLM space
        visual_tokens = self.projector(patch_embeds)  # [B, N_patches, llm_dim]
        return visual_tokens

    def forward(
        self,
        pixel_values: torch.Tensor,       # [B, 3, H, W]
        input_ids: torch.Tensor,          # [B, T]   (text tokens)
        attention_mask: torch.Tensor,     # [B, N_patches + T]
        labels: torch.Tensor | None = None,  # [B, T]
    ):
        # 1. Encode visual tokens
        visual_tokens = self.encode_image(pixel_values)  # [B, N, D_llm]
        B, N, D = visual_tokens.shape

        # 2. Get text embeddings from LLM's word embedding table
        text_embeds = self.llm.model.embed_tokens(input_ids)  # [B, T, D_llm]

        # 3. Concatenate: [visual | text] along the sequence dimension
        inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)
        # inputs_embeds: [B, N + T, D_llm]

        # 4. If labels are provided, mask loss over visual positions
        if labels is not None:
            # Pad labels for the visual token positions with -100 (ignore index)
            visual_labels = torch.full(
                (B, N), fill_value=-100, dtype=labels.dtype, device=labels.device
            )
            labels = torch.cat([visual_labels, labels], dim=1)

        # 5. Forward pass through LLM using embeddings (not input_ids)
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs


# NOTE: MiniLLaVA.forward above always prepends all visual tokens before the text,
# which is a functional simplification. Real LLaVA splices projected image tokens at
# the position of an <image> placeholder inside the chat template (so, e.g., a system
# prompt can precede the image). The helper below shows that real placeholder-splicing.
def splice_visual_tokens(text_embeds, input_ids, visual_tokens, labels, image_token_id):
    # single example (no batch dim): text_embeds [T,D], input_ids [T],
    # visual_tokens [N,D], labels [T] (-100 on prompt/system positions), scalar image_token_id
    pos = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
    assert pos.numel() == 1, "expected exactly one <image> placeholder"
    p = pos.item()
    N = visual_tokens.shape[0]
    new_embeds = torch.cat([text_embeds[:p], visual_tokens, text_embeds[p+1:]], dim=0)  # [T-1+N, D]
    vis_labels = torch.full((N,), -100, dtype=labels.dtype, device=labels.device)
    new_labels = torch.cat([labels[:p], vis_labels, labels[p+1:]], dim=0)               # [T-1+N]
    new_mask   = torch.ones(new_embeds.shape[0], dtype=torch.long, device=new_embeds.device)
    # check: new_embeds.shape[0] == input_ids.shape[0] - 1 + N
    return new_embeds, new_labels, new_mask


# --- Quick shape check (no weights needed) --- (verbatim `if __name__ == "__main__"` body)
B, H, W = 2, 336, 336  # LLaVA uses 336x336
CLIP_PATCH_SIZE = 14
N_PATCHES = (H // CLIP_PATCH_SIZE) ** 2  # 576 patches
VISION_DIM = 1024   # CLIP ViT-L/14
LLM_DIM   = 4096   # LLaMA-7B hidden size

proj = TwoLayerMLP(VISION_DIM, LLM_DIM)
fake_patches = torch.randn(B, N_PATCHES, VISION_DIM)
out = proj(fake_patches)
print(f"Patches in:  {fake_patches.shape}")  # [2, 576, 1024]
print(f"Tokens out:  {out.shape}")            # [2, 576, 4096]
print(f"Projector params: {sum(p.numel() for p in proj.parameters()):,}")
# 1024*4096 + 4096*4096 (+ biases) ≈ 21M params — tiny vs 7B LLM
# (prints 20,979,712 = 4,198,400 + 16,781,312)
assert out.shape == (B, N_PATCHES, LLM_DIM)
assert sum(p.numel() for p in proj.parameters()) == 20_979_712


# --- Test glue: drive MiniLLaVA and splice_visual_tokens end-to-end on CPU ---
# Tiny fake CLIP-shaped vision tower and a tiny fake HF-CausalLM-shaped decoder,
# scaled way down so the *book's own* MiniLLaVA/TwoLayerMLP code runs unmodified.
class _FakeVisionOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class _FakeVisionModel(nn.Module):
    """Stands in for a frozen CLIPVisionModel: returns [B, 1+N, vision_dim]."""
    def __init__(self, vision_dim: int, n_patches: int):
        super().__init__()
        self.n_patches = n_patches
        self.vision_dim = vision_dim
        # a tiny linear "encoder" so the output actually depends on pixel_values
        self.patchify = nn.Conv2d(3, vision_dim, kernel_size=8, stride=8)
        self.cls = nn.Parameter(torch.zeros(1, 1, vision_dim))

    def forward(self, pixel_values: torch.Tensor):
        feat = self.patchify(pixel_values)              # [B, vision_dim, h, w]
        Bp, C, h, w = feat.shape
        feat = feat.flatten(2).transpose(1, 2)           # [B, h*w, vision_dim]
        cls = self.cls.expand(Bp, -1, -1)
        hidden = torch.cat([cls, feat], dim=1)            # [B, 1+h*w, vision_dim]
        return _FakeVisionOutput(hidden)


class _FakeLLMOutput:
    def __init__(self, loss, logits):
        self.loss = loss
        self.logits = logits


class _FakeLLMInner(nn.Module):
    def __init__(self, vocab_size: int, llm_dim: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, llm_dim)


class _FakeLLM(nn.Module):
    """Stands in for a causal LLM with a HF-CausalLM-shaped call signature:
    llm.model.embed_tokens(...) for embeddings, and
    llm(inputs_embeds=..., attention_mask=..., labels=...) -> loss/logits.
    """
    def __init__(self, vocab_size: int, llm_dim: int):
        super().__init__()
        self.model = _FakeLLMInner(vocab_size, llm_dim)
        self.lm_head = nn.Linear(llm_dim, vocab_size)

    def forward(self, inputs_embeds, attention_mask, labels=None):
        logits = self.lm_head(inputs_embeds)  # [B, S, vocab]
        loss = None
        if labels is not None:
            # standard next-token shift, exactly like a real HF CausalLM
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return _FakeLLMOutput(loss, logits)


# tiny sizes so this stays fast: 32x32 "image" -> 4x4=16 patches with an 8x8 patchifier
tiny_vision_dim = 32
tiny_llm_dim = 48
tiny_vocab = 100
tiny_H = tiny_W = 32
tiny_n_patches = (tiny_H // 8) * (tiny_W // 8)  # 16

fake_vision_model = _FakeVisionModel(tiny_vision_dim, tiny_n_patches)
fake_projector = TwoLayerMLP(tiny_vision_dim, tiny_llm_dim)
fake_llm = _FakeLLM(tiny_vocab, tiny_llm_dim)
mini_llava = MiniLLaVA(fake_vision_model, fake_projector, fake_llm, llm_dim=tiny_llm_dim)

B2, T2 = 2, 5
pixel_values = torch.randn(B2, 3, tiny_H, tiny_W)
input_ids = torch.randint(0, tiny_vocab, (B2, T2))
labels = input_ids.clone()
attention_mask = torch.ones(B2, tiny_n_patches + T2, dtype=torch.long)

mini_llava.eval()
mini_out = mini_llava(pixel_values, input_ids, attention_mask, labels=labels)
assert mini_out.logits.shape == (B2, tiny_n_patches + T2, tiny_vocab)
assert mini_out.loss is not None and torch.isfinite(mini_out.loss)
print(f"MiniLLaVA logits shape: {mini_out.logits.shape}, loss: {mini_out.loss.item():.4f}")

# exercise splice_visual_tokens on a single (unbatched) example
T3, D3, N3 = 6, tiny_llm_dim, 4
image_token_id = 999
text_ids = torch.randint(0, tiny_vocab, (T3,))
text_ids[2] = image_token_id  # single <image> placeholder at position 2
text_embeds = torch.randn(T3, D3)
visual_tokens = torch.randn(N3, D3)
text_labels = torch.randint(0, tiny_vocab, (T3,))
new_embeds, new_labels, new_mask = splice_visual_tokens(
    text_embeds, text_ids, visual_tokens, text_labels, image_token_id
)
assert new_embeds.shape == (T3 - 1 + N3, D3)
assert new_labels.shape == (T3 - 1 + N3,)
assert new_mask.shape == (T3 - 1 + N3,)
assert torch.all(new_labels[2:2 + N3] == -100)
print(f"splice_visual_tokens: {T3} text + {N3} visual -> {new_embeds.shape[0]} tokens")


# =============================================================================
# Block #1 (line ~208): Flamingo-style gated cross-attention
# =============================================================================

class GatedCrossAttention(nn.Module):
    """
    Flamingo-style gated cross-attention block.

    The hidden states h query into visual features z_v.
    The gate alpha (scalar) starts at 0 so the skip connection
    initially passes h unchanged.
    """
    def __init__(self, d_model: int, d_vision: int, n_heads: int = 8):
        super().__init__()
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        # Query comes from LLM hidden states
        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        # Key/Value come from vision encoder output
        self.k_proj   = nn.Linear(d_vision, d_model, bias=False)
        self.v_proj   = nn.Linear(d_vision, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        # Gating scalar, init to 0 => tanh(0) = 0 => no perturbation
        self.alpha    = nn.Parameter(torch.zeros(1))
        self.norm_q   = nn.LayerNorm(d_model)
        self.norm_kv  = nn.LayerNorm(d_vision)

    def forward(
        self,
        h: torch.Tensor,    # [B, T_text, d_model]  LLM hidden states
        z: torch.Tensor,    # [B, N_vis,  d_vision] vision encoder output
    ) -> torch.Tensor:
        B, T, D = h.shape
        _, N, _ = z.shape

        # Normalize inputs before attention (Flamingo uses a norm here)
        q = self.q_proj(self.norm_q(h))    # [B, T, D]
        k = self.k_proj(self.norm_kv(z))   # [B, N, D]
        v = self.v_proj(self.norm_kv(z))   # [B, N, D]

        # Reshape for multi-head attention
        def split_heads(x):
            B, L, D = x.shape
            return x.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
            # -> [B, n_heads, L, d_head]

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # Scaled dot-product attention (cross-attention: q from h, kv from z)
        scale = self.d_head ** -0.5
        attn  = (q @ k.transpose(-2, -1)) * scale  # [B, n_heads, T, N]
        attn  = F.softmax(attn, dim=-1)
        out   = attn @ v                            # [B, n_heads, T, d_head]

        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)                    # [B, T, D]

        # Gated residual: starts at 0, grows during training
        return h + torch.tanh(self.alpha) * out


# Demo (verbatim `if __name__ == "__main__"` body)
B, T_text = 2, 64          # batch, text tokens
N_vis     = 64             # 8x8 visual tokens (pooled)
d_model   = 2048           # LLM width (e.g. Chinchilla-style 7B)
d_vision  = 1024           # CLIP ViT-L/14

gca = GatedCrossAttention(d_model, d_vision, n_heads=16)
h   = torch.randn(B, T_text, d_model)
z   = torch.randn(B, N_vis,  d_vision)
out = gca(h, z)
print(f"Input shape:  {h.shape}")      # [2, 64, 2048]
print(f"Output shape: {out.shape}")    # [2, 64, 2048]
print(f"Alpha (gate): {gca.alpha.item():.4f}")  # 0.0000 at init
assert out.shape == h.shape
assert gca.alpha.item() == 0.0
# gate starts closed: tanh(0) = 0, so output must equal the input h exactly at init
assert torch.allclose(out, h)


# =============================================================================
# Block #3 (line ~435): InternVL 2 stage-2 optimizer with per-component LRs
# =============================================================================

def build_optimizer_stage2(model, lr_llm=2e-5, lr_vision=2e-6, lr_proj=1e-4):
    """
    Stage 2 optimizer with different learning rates for each component.
    Vision encoder gets 10x lower LR than LLM to prevent forgetting.
    Projector gets highest LR since it is the newest and most undertrained part.
    """
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters()
                       if "vision_model" in n and p.requires_grad],
            "lr": lr_vision,
            "name": "vision_encoder",
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if "projector" in n and p.requires_grad],
            "lr": lr_proj,
            "name": "projector",
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if "llm" in n and p.requires_grad],
            "lr": lr_llm,
            "name": "llm",
        },
    ]
    # Filter out empty groups (e.g. if vision encoder is frozen)
    param_groups = [g for g in param_groups if len(g["params"]) > 0]
    from torch.optim import AdamW
    optimizer = AdamW(param_groups, weight_decay=0.0)
    return optimizer


# Reuse mini_llava (has named submodules "vision_model", "projector", "llm",
# exactly matching the substring filters build_optimizer_stage2 relies on).
optimizer = build_optimizer_stage2(mini_llava)
group_names = {g["name"]: g["lr"] for g in optimizer.param_groups}
assert group_names == {"vision_encoder": 2e-6, "projector": 1e-4, "llm": 2e-5}
print(f"Optimizer param groups: {group_names}")

# Sanity-check the optimizer actually works: one backward + step on the
# MiniLLaVA loss computed above should not raise and should change weights.
before = fake_projector.proj[0].weight.clone()
mini_llava.train()
out2 = mini_llava(pixel_values, input_ids, attention_mask, labels=labels)
optimizer.zero_grad()
out2.loss.backward()
optimizer.step()
after = fake_projector.proj[0].weight
assert not torch.allclose(before, after), "optimizer step should update projector weights"
print("optimizer step updated projector weights: OK")

print("\nAll tested blocks (#0, #1, #3) executed successfully.")
