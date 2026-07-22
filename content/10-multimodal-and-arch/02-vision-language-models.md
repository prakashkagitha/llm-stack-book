# 10.2 Vision-Language Models

Vision-Language Models (VLMs) solve a deceptively simple problem: given a pixel grid, produce tokens that an LLM can reason about. That bridge — turning an image into something a language model understands — is where almost all the architectural creativity lives. Get it wrong and you get a model that can describe cats but cannot read a receipt. Get it right and you get a model that can answer questions about medical scans, extract tables from PDFs, navigate UIs by screenshot, or write code from a whiteboard photo.

This chapter covers the full arc: the two dominant connection strategies (projectors and cross-attention), the token explosion problem, native-resolution processing, OCR/document understanding, training recipes, and the modern unified recipe that powers models like LLaVA-1.6, Qwen-VL, and InternVL. We assume you are comfortable with the Transformer architecture (see [The Attention Mechanism From Scratch](../02-transformer/03-attention-from-scratch.html)) and with Vision Transformers (covered in [Vision Transformers & Image Encoders](../10-multimodal-and-arch/01-vision-transformers.html)).

## The Connection Problem

A language model speaks in tokens drawn from a vocabulary of 32,000–128,000 entries, each an integer index. A vision encoder speaks in patch embeddings: dense floating-point vectors, one per $16 \times 16$ pixel grid cell. These two representations live in completely different spaces. The "connection problem" is: how do you translate patch embeddings into something the LLM's residual stream can accept?

There are three broad answers:

1. **Projector (LLaVA-style):** Run the vision encoder, flatten its output sequence of visual tokens, pass them through a learned linear projection (or a small MLP), then prepend/interleave those projected vectors into the LLM's token embedding sequence as if they were text token embeddings.

2. **Cross-attention (Flamingo-style):** Freeze or lightly fine-tune the LLM, then insert new cross-attention layers that attend from LLM hidden states to the vision encoder's output. The LLM's causal self-attention layers are unchanged; the cross-attention layers are interleaved and learn to "look at" the image on demand.

3. **Unified tokenizer (native multimodal):** Quantize patches into discrete visual tokens from a codebook (VQ-VAE style), then feed image and text tokens as a single flat sequence to one model. GPT-4o and Chameleon use variants of this. We cover it more in [Unified & Any-to-Any Models](../10-multimodal-and-arch/05-unified-any-to-any.html).

The projector approach (strategy 1) has become the dominant recipe because it is simple, cheap to train, and easy to compose with any pretrained LLM.

Strategies 1 and 2 are both forms of *late fusion*: a separately pretrained vision encoder produces features that a lightweight connector (projector or cross-attention) injects into a largely-frozen LLM, so the two modalities are computed independently and only merged partway through. Strategy 3 is *early fusion*: image and text become a single joint token stream that one model attends over from its very first layer, with no separate vision tower at inference (Chameleon quantizes patches into codebook tokens; Fuyu skips the encoder entirely and feeds linearly-projected raw patches straight into the decoder). Late fusion is cheaper to train and composes with any pretrained LLM, which explains its dominance; early fusion is architecturally simpler and unifies understanding with generation, but it must be trained from scratch on mixed data.

{{fig:vlm-projector-dataflow}}

## LLaVA: The Projector Paradigm

LLaVA (Large Language and Vision Assistant, Liu et al., 2023) popularized the minimal projector design. The architecture has three components:

1. A frozen CLIP ViT (typically ViT-L/14 @ 336px) that produces $N$ patch embeddings of dimension $D_v = 1024$.
2. A projector $W$ (originally a single linear layer, later a two-layer MLP with GELU) that maps $D_v \to D_\text{llm}$.
3. A causal LLM (Vicuna or LLaMA) that receives the projected visual tokens followed by the tokenized text prompt.

The training recipe is staged:

**Stage 1 — Feature Alignment.** Only the projector is trained. The LLM and vision encoder are frozen. The model learns to map visual features into the LLM's embedding space using a large corpus of image–caption pairs (CC3M-scale). Loss: language modeling on the caption tokens only. This stage takes on the order of 1 GPU-day.

**Stage 2 — Instruction Tuning.** The projector and the LLM are both unfrozen (or the LLM is LoRA-tuned). The model is trained on multimodal instruction-following data: (image, question, answer) triples. Visual encoder weights may remain frozen to preserve CLIP's representation quality. This stage is where the model learns to follow instructions about images.

The projector equation is clean:

$$
\mathbf{H}_v = \text{Projector}(\mathbf{Z}_v) \in \mathbb{R}^{N \times D_\text{llm}}
$$

where $\mathbf{Z}_v \in \mathbb{R}^{N \times D_v}$ is the ViT's output (CLS token optionally removed, leaving only patch tokens). The LLM then processes the concatenated sequence:

$$
[\mathbf{H}_v \| \mathbf{E}_\text{text}] \in \mathbb{R}^{(N + T) \times D_\text{llm}}
$$

where $\mathbf{E}_\text{text}$ are the standard token embeddings for the text prompt and $T$ is the number of text tokens.

```python
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor
from transformers import LlamaForCausalLM, LlamaTokenizer

# Minimal LLaVA-style projector and forward pass
# (Simplified for exposition; omit generation logic for brevity)

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


# --- Quick shape check (no weights needed) ---
if __name__ == "__main__":
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
```

LLaVA-1.5 replaced the linear projector with this MLP connector, achieving significant gains on benchmarks like VQAv2 and MMBench. The lesson: the projector is a hyperparameter worth tuning.

## Flamingo: Cross-Attention for In-Context Multimodal Learning

Flamingo (Alayrac et al., DeepMind 2022) took a different path. Rather than prepending visual tokens, it freezes a large pretrained LLM and inserts new *gated cross-attention* layers between every transformer block. Each cross-attention layer attends from LLM hidden states to vision encoder outputs.

The gated cross-attention update for layer $\ell$ is:

$$
\mathbf{h}^{(\ell)} \leftarrow \mathbf{h}^{(\ell)} + \tanh(\alpha) \cdot \text{CrossAttn}^{(\ell)}(\mathbf{h}^{(\ell)}, \mathbf{Z}_v)
$$

where $\alpha$ is a learnable scalar initialized to 0, so the gate starts closed and the LLM is barely perturbed at the beginning of training. This initialization lets Flamingo inherit all of the pretrained LLM's text capabilities while gradually integrating visual information.

The key architectural difference from the projector approach:

| Dimension | LLaVA-style projector | Flamingo cross-attn |
|---|---|---|
| Visual tokens in LLM stream | Yes — they occupy sequence positions | No — LLM residual stream unchanged |
| New parameters | Projector only (~21M) | Cross-attn KV projections in every layer |
| LLM context consumed by image | Proportional to N_patches (e.g. 576) | Zero — image stored externally |
| Few-shot image interleaving | Awkward — prepend all images | Natural — interleaved in context |
| Fine-tuning complexity | Straightforward | More complex; two parameter groups |

{{fig:vlm-projector-vs-crossattn}}

Flamingo's cross-attention design enables *interleaved* image–text few-shot inference natively: you can pass "image1, caption1, image2, caption2, image3, ?" as a single context. The cross-attention layers route each text token to the most recently preceding image (via a masking strategy in the attention). This is harder to achieve with projectors because all image tokens compete for context space.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

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


# Demo
if __name__ == "__main__":
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
```

## The Visual Token Explosion

The most important practical concern in VLM engineering is the **visual token explosion**: the number of tokens consumed by one image scales as $(H/p)^2$ where $p$ is the patch size. A $336 \times 336$ image with $14 \times 14$ patches gives $576$ tokens. Scale to $672 \times 672$ and you get $2304$ tokens. These tokens sit in the LLM's context and cost $O(N^2)$ attention in the prefill phase.

{{fig:vlm-token-explosion}}

!!! example "Worked Example: Token Count and Memory Cost"

    Suppose we use LLaVA-1.6 with a 4K resolution tile strategy on a single $1344 \times 336$ image (panoramic scan).
    LLaVA-1.6 divides the image into tiles:
    - A $1344 \times 336$ image is split into $4 \times 1 = 4$ tiles of $336 \times 336$ each plus a low-resolution "thumbnail" tile.
    - Each $336 \times 336$ tile produces $576$ patch tokens through CLIP ViT-L/14.
    - 4 tiles + 1 thumbnail $\times 576 = 2880$ visual tokens.

    At LLaMA-2-7B with $D_\text{llm} = 4096$, each token costs $2 \text{ (K and V)} \times 4096 \text{ dims} \times 2 \text{ bytes} = 16{,}384$ bytes (bf16) in the KV cache per layer, across 32 layers:

    $$
    \text{KV memory} = 2880 \times 32 \times 2 \times 4096 \times 2 \,\text{bytes} \approx 1.5\,\text{GB}
    $$

    just for the visual prefix of a single image in a batch of 1. For a batch of 8 with long text answers (say 256 text tokens), total KV cache grows to roughly 12 GB, saturating an A100-40GB at modest batch sizes. This is the core tension: more visual tokens = higher resolution fidelity, more context window consumed, more memory.

    For comparison, a pure text conversation with 1024 tokens at the same model uses $1024 \times 32 \times 2 \times 4096 \times 2 \approx 536\,\text{MB}$ — the image costs $3\times$ more KV memory than a long text prompt.

Several techniques reduce this cost:

- **Visual token compression (Q-Former):** BLIP-2 (Li et al., 2023) introduces a Querying Transformer (Q-Former) that extracts a fixed $K=32$ query vectors from the full $N$ visual tokens. The Q-Former's learned queries cross-attend to the vision encoder and output $K$ compressed representations regardless of image resolution.

- **Average pooling:** Some models simply average-pool adjacent patches (e.g. $2 \times 2$ average, reducing by $4\times$) before projecting.

- **Dynamic resolution with token merging:** Retain high-res patches near salient regions and merge low-information patches using token merging algorithms like ToMe (Bolya et al., 2023).

See [PagedAttention & KV-Cache Memory Management](../04-kernels-efficiency/06-paged-attention-kv.html) for infrastructure techniques that partially mitigate visual prefix costs at serving time.

## Native Resolution and Any-Resolution

Early VLMs like LLaVA-1.0 resize all images to a fixed $336 \times 336$ before encoding. This is fine for simple visual QA but destroys the detail needed for:
- OCR on dense documents (small text becomes illegible)
- Chart and table reading
- Counting objects in crowded scenes
- Medical imaging

The solution is **any-resolution (AnyRes)** processing, introduced in LLaVA-1.6 and independently in InternVL and Qwen-VL.

The recipe:

1. **Tile the image.** Given an image of arbitrary resolution, divide it into overlapping or non-overlapping tiles that each fit the fixed encoder resolution ($336 \times 336$). Also encode a down-sampled global "thumbnail" to give the model spatial context.

2. **Encode each tile independently** through the same ViT. Each tile produces the same number of patch tokens.

3. **Concatenate and project.** All tile embeddings are concatenated (along the sequence dimension) and passed through the projector. A separator token is optionally inserted between tiles.

4. **Positional cues.** The model needs to know which tile came from which spatial position. Strategies include: (a) adding tile-position embeddings, (b) inserting special `<row_start>` / `<col_end>` tokens, (c) encoding row/column as text like `"row 0, col 2:"`.

{{fig:vlm-anyres-tiling}}

```python
from PIL import Image
import torch

def tile_image(
    image: Image.Image,
    tile_size: int = 336,
    max_tiles: int = 6,
) -> list[Image.Image]:
    """
    Divide an image into tiles of `tile_size x tile_size`.
    Always returns a thumbnail as the last tile.
    Limits total tiles to max_tiles (excluding thumbnail).
    """
    W, H = image.size
    # Compute number of tiles in each dimension
    n_cols = max(1, round(W / tile_size))
    n_rows = max(1, round(H / tile_size))
    # Clip to max_tiles (e.g. 6 for LLaVA-1.6)
    if n_rows * n_cols > max_tiles:
        # Reduce proportionally — simplified version
        scale = (max_tiles / (n_rows * n_cols)) ** 0.5
        n_cols = max(1, int(n_cols * scale))
        n_rows = max(1, int(n_rows * scale))

    # Resize image to exact grid dimensions
    resized = image.resize((n_cols * tile_size, n_rows * tile_size), Image.BICUBIC)

    tiles = []
    for r in range(n_rows):
        for c in range(n_cols):
            left   = c * tile_size
            upper  = r * tile_size
            right  = left + tile_size
            lower  = upper + tile_size
            tiles.append(resized.crop((left, upper, right, lower)))

    # Thumbnail: whole image squeezed into one tile
    thumbnail = image.resize((tile_size, tile_size), Image.BICUBIC)
    tiles.append(thumbnail)
    return tiles  # length = n_rows * n_cols + 1


def encode_tiles_to_visual_tokens(
    tiles: list,
    vision_encoder,       # returns [1, 1+N_patches, D_v]
    projector,            # maps [1, N_patches, D_v] -> [1, N_patches, D_llm]
    device: str = "cuda",
) -> torch.Tensor:
    """
    Encode each tile through the vision encoder + projector.
    Returns concatenated visual tokens: [1, K * N_patches, D_llm]
    where K = len(tiles).
    """
    all_tokens = []
    for tile_img in tiles:
        # Preprocess (CLIP normalization, to_tensor, add batch dim)
        pixel_values = preprocess(tile_img).unsqueeze(0).to(device)  # [1, 3, 336, 336]
        with torch.no_grad():
            enc = vision_encoder(pixel_values=pixel_values)
        # Drop CLS token, project
        patch_embeds = enc.last_hidden_state[:, 1:, :]  # [1, 576, D_v]
        tokens = projector(patch_embeds)                # [1, 576, D_llm]
        all_tokens.append(tokens)
    # Concatenate along sequence dimension
    visual_tokens = torch.cat(all_tokens, dim=1)        # [1, K*576, D_llm]
    return visual_tokens


# Example: 672x336 image -> 2 tiles + 1 thumbnail = 3 * 576 = 1728 visual tokens
```

InternVL 1.5 and 2.x push this further with **dynamic high resolution**: tiles are selected based on the image's aspect ratio and content type, and the model is trained with a curriculum that starts at low resolution and increases to $4 \times 4$ tiles (up to 2304 tokens excluding thumbnail). Qwen-VL uses a similar approach but introduces a **position-aware visual attention** mechanism so that patches from different tiles maintain correct 2-D relative positions when processed in the LLM.

## Qwen-VL and InternVL: The Modern Recipe

Qwen-VL (Bai et al., Alibaba, 2023) and InternVL (Chen et al., Shanghai AI Lab, 2023–2024) represent the current state-of-practice. Both move beyond the minimal LLaVA recipe in two important ways:

### Larger, Better Vision Encoders

LLaVA-1.5 uses CLIP ViT-L/14 (~307M params). InternVL uses InternViT-6B — a 6 billion parameter vision encoder trained with CLIP contrastive learning at high resolution. The larger encoder dramatically improves dense captioning, OCR, and chart understanding, at the cost of higher inference compute (see [Vision Transformers & Image Encoders](../10-multimodal-and-arch/01-vision-transformers.html) for details on ViT scaling).

### Interleaved Visual–Text Pretraining

Rather than two isolated stages, modern models train on large corpora of documents where images and text are naturally interleaved — web pages, research papers, textbooks, e-commerce listings. This teaches the model to integrate images and text in context, not just answer single-image questions.

### Training Stages (InternVL 2 Recipe)

{{fig:vlm-internvl2-training-stages}}

The crucial engineering decision at stage 2 is whether to unfreeze the vision encoder. Freezing it is safer (less catastrophic forgetting of CLIP-style alignment) but limits the model's ability to adapt to domain-specific visual features. InternVL 2 unfreezes the encoder at stage 2 with a lower learning rate, which empirically helps on OCR and dense chart tasks.

```python
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

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
    optimizer = AdamW(param_groups, weight_decay=0.0)
    return optimizer
```

## OCR and Document Understanding

One of the most commercially important capabilities of modern VLMs is reading text in images: receipts, invoices, PDFs, whiteboards, code screenshots. This task goes by many names — OCR-VQA, document understanding, visual text recognition — and it stresses VLMs in a specific way: the model must both *see* fine-grained character shapes and *understand* the semantic meaning of the text in context.

### Why Standard VLMs Struggle with OCR

A $336 \times 336$ image contains ~100,000 pixels. A character at 12pt font in a standard document is roughly $10 \times 10$ pixels. At $14 \times 14$ patch size, a single patch covers 196 pixels — the model sees at most a few characters per patch, smeared together. Fine text recognition requires either:

1. **Higher resolution:** More pixels per patch, more patches per image.
2. **Specialized pretraining data:** The model needs to have "read" thousands of document images with ground-truth OCR labels during training.
3. **Large visual encoder:** Bigger encoders capture finer-grained spatial detail.

InternVL 2 and Qwen-VL address all three. Both are trained on datasets like DocVQA, TextVQA, ChartQA, and DVQA. Qwen-VL additionally introduced a specialized visual grounding pretraining task: given an image and a text string, predict the bounding box `[x1, y1, x2, y2]` of the text region. This forces the model to localize text before reading it.

### The DocOwl / mPLUG-Owl Pattern for Documents

For multi-page documents, the standard recipe is:

1. Convert each page to an image (e.g. with `pdf2image`).
2. Apply any-resolution tiling per page.
3. Concatenate all page tokens with separator tokens: `<page_1_tokens> <page_sep> <page_2_tokens> ...`
4. Truncate to the LLM's max context if the document is very long.

This runs into the context limit quickly. A 10-page document with 4 tiles/page = 40 tiles = $40 \times 576 = 23{,}040$ visual tokens. At LLaMA-3-8B's 8K context window, that fills the context entirely. Qwen-VL2 extends context to 32K tokens partly to handle longer documents.

```python
def encode_document_pages(
    pages: list,              # list of PIL Images, one per page
    vision_encoder,
    projector,
    tokenizer,
    max_tiles_per_page: int = 4,
    page_sep_token: str = "<page_sep>",
) -> dict:
    """
    Encode a multi-page document into a flat visual token sequence
    with page separator tokens interspersed.

    Returns a dict with:
      - 'inputs_embeds': [1, total_tokens, D_llm]
      - 'attention_mask': [1, total_tokens]
    """
    page_sep_id = tokenizer.convert_tokens_to_ids(page_sep_token)
    sep_embed   = projector.proj[0].weight.new_zeros(1, 1, projector.proj[-1].out_features)
    # In practice, <page_sep> is a learned embedding from the LLM vocab

    all_embeds = []
    for page_img in pages:
        tiles  = tile_image(page_img, max_tiles=max_tiles_per_page)
        tokens = encode_tiles_to_visual_tokens(tiles, vision_encoder, projector)
        # tokens: [1, K*576, D_llm]
        all_embeds.append(tokens)
        all_embeds.append(sep_embed)  # separator between pages

    # Concatenate all pages
    doc_embeds = torch.cat(all_embeds, dim=1)  # [1, total_visual_tokens, D_llm]
    total_len  = doc_embeds.shape[1]
    attn_mask  = torch.ones(1, total_len, dtype=torch.long)
    return {"inputs_embeds": doc_embeds, "attention_mask": attn_mask}
```

## Training Data and the Multimodal Data Mix

The capability of a VLM is governed at least as much by its training data mix as by its architecture. The data pipeline typically combines:

| Dataset type | Examples | Purpose |
|---|---|---|
| Image–caption pairs | LAION-400M, CC12M, COYO-700M | Stage 1 alignment |
| VQA datasets | VQAv2, GQA, OK-VQA | Basic visual question answering |
| OCR/text reading | TextVQA, OCR-VQA, DocVQA | Text recognition capability |
| Chart/plot QA | ChartQA, DVQA, FigureQA | Structured visual data reading |
| Visual instruction tuning | LLaVA-Instruct-150K, ShareGPT4V | Instruction following |
| Grounding | RefCOCO, Flickr30K Entities | Spatial localization |
| Science/Math | AI2D, MMMU, ScienceQA | Domain knowledge with visuals |
| Document understanding | DocVQA, InfoVQA | Multi-element document pages |
| Interleaved web data | MMC4, OBELISC | Multi-image context |

Data quality matters more than quantity. LLaVA-1.5 uses only 665K instruction-tuning examples but outperforms models trained on millions of lower-quality samples. Techniques like ShareGPT4V (Zhang et al., 2023) use GPT-4V to generate higher-quality captions for existing images, bootstrapping quality at scale.

The loss function is standard cross-entropy over text tokens only. Visual tokens are in the input but produce no loss signal:

$$
\mathcal{L} = -\sum_{t \in \text{text positions}} \log p_\theta(x_t \mid \mathbf{H}_v, x_{<t})
$$

This is the same next-token prediction loss used in [The Pretraining Objective & Loss](../03-pretraining/03-pretraining-objective.html), but with the visual tokens as context.

!!! interview "Interview Corner"

    **Q:** A candidate is designing a VLM that must read receipts, answer questions about charts, and describe natural scene images — all in one model. Walk through the key architectural choices and training decisions.

    **A:** Start with the connection strategy. A projector (MLP connector) is simpler to train and works well; cross-attention (Flamingo-style) adds overhead but handles multi-image interleaving more naturally. For this use-case, a projector suffices.

    For the vision encoder, use a large ViT (ViT-L or larger) pretrained with CLIP — CLIP encoders generalize broadly across tasks. For OCR on receipts, enable any-resolution (AnyRes) tiling: split high-res images into 336×336 tiles, encode each independently, concatenate. This is essential for fine text at low font sizes.

    Training stages: (1) align the projector with frozen encoder + LLM on image-caption pairs, (2) fine-tune the projector and LLM jointly on diverse instruction-following data that includes OCR, VQA, chart, and captioning tasks. Use task-type-weighted sampling to balance the three capability areas.

    The key pitfall to watch: the visual token budget. A 4-tile receipt image produces 4×576 = 2304 visual tokens. In a chat loop with conversation history, context fills quickly. Consider token pooling (Q-Former or average pooling) to reduce visual tokens by 4–8× at the cost of some fine-grained detail.

    Also tune learning rates per component: lower LR for the vision encoder (2e-6), higher LR for the projector (1e-4), mid LR for the LLM (2e-5). This prevents catastrophic forgetting of the encoder's pretrained CLIP alignment.

## The Dominant Multimodal Recipe (2024–2025)

Across LLaVA-1.6, InternVL 2, Qwen-VL 2, and similar models, a clear consensus recipe has emerged. If you are building a new VLM today, this is the default starting point:

{{fig:vlm-dominant-recipe}}

The remaining open questions the field is actively working on:

- **Video:** Sampling frames from a video and treating them as a bag of tiles works but loses temporal structure. Models like Video-LLaMA and LLaVA-Video encode temporal position but still consume enormous context windows.
- **3D/depth:** Encoding depth maps or point clouds alongside RGB images is non-trivial; see [Unified & Any-to-Any Models](../10-multimodal-and-arch/05-unified-any-to-any.html).
- **Token compression at scale:** 2304 visual tokens per image is manageable for single-image tasks but kills throughput in high-QPS serving environments. Token merging, Q-Former, and selective token dropping are active research areas. See [Inference Economics: Latency, Throughput & Cost](../07-inference-serving/12-inference-economics.html).
- **Alignment with longer reasoning:** Combining visual grounding with chain-of-thought (see [Reasoning, Chain-of-Thought & Test-Time Compute](../05-posttraining-alignment/10-reasoning-test-time-compute.html)) remains challenging; visual hallucinations increase with longer generation.

!!! warning "Common pitfall: Forgetting to mask visual token losses"

    When computing the language modeling loss, you must set the label for every visual token position to -100 (the standard ignore index in PyTorch CrossEntropyLoss). If you forget this, the model will try to predict "the next visual token" from a text-only loss signal, which is nonsensical, leads to spurious gradient updates through the projector, and corrupts training. Always verify your loss computation by checking that the loss is computed only over text positions.

!!! tip "Practitioner tip: ViT-L vs bigger encoders for production"

    In production, ViT-L/14 (CLIP, 307M params) is often the right call even when bigger encoders are available. InternViT-6B improves benchmark scores but adds ~6B params to your serving cost — roughly doubling the model size for marginal gains on most tasks outside dense OCR. Unless your application specifically requires fine-grained text reading in 4K images, profile with ViT-L first.

!!! key "Key Takeaways"

    - VLMs bridge a vision encoder and an LLM via two main strategies: **projector (LLaVA-style)** prepends projected visual tokens into the LLM's sequence; **cross-attention (Flamingo-style)** inserts new cross-attention layers that let LLM hidden states query visual features without consuming context positions.
    - The **projector approach** dominates in 2024–2025 due to its simplicity: a two-layer MLP maps CLIP ViT patch embeddings into the LLM embedding space. Only ~21M new parameters are needed.
    - The **visual token explosion** is the central engineering constraint: a 336px image generates 576 tokens, and any-resolution tiling multiplies this by the number of tiles. KV-cache memory and prefill FLOPS scale accordingly.
    - **Any-resolution (AnyRes) tiling** — dividing an image into multiple 336×336 tiles and encoding each independently — is the standard solution for high-resolution and OCR tasks. LLaVA-1.6, InternVL 2, and Qwen-VL all use this approach.
    - **Training is staged:** first align the projector with frozen encoder + LLM; then co-train the projector and LLM (and optionally the encoder at a lower LR) on diverse instruction-following data.
    - **OCR and document understanding** require high resolution (tiling), large encoders, and specialized training data (DocVQA, TextVQA, ChartQA). The model must localize then read text.
    - **Data quality beats data quantity:** 665K high-quality instruction examples (LLaVA-1.5) outperforms millions of noisy samples. GPT-4V-recaptioned data improves downstream quality significantly.
    - Differential learning rates across components (low for vision encoder, mid for LLM, high for projector) prevent catastrophic forgetting and accelerate projector convergence.
    - **Token compression** (Q-Former, average pooling, token merging) trades resolution fidelity for context efficiency — essential in high-throughput serving scenarios.

!!! sota "State of the Art & Resources (2026)"
    Vision-language models have converged on a projector-based recipe (ViT encoder + MLP connector + LLM) as the dominant open-source paradigm, with InternVL 2.5, Qwen2-VL, and LLaVA-NeXT variants matching or exceeding GPT-4V on standard benchmarks. Active frontiers include native-resolution dynamic tokenization, video VLMs, and efficient token compression for high-throughput serving.

    **Foundational work**

    - [Liu et al., *Visual Instruction Tuning* (2023)](https://arxiv.org/abs/2304.08485) — The original LLaVA paper establishing the projector paradigm for VLMs (NeurIPS 2023 Oral).
    - [Alayrac et al., *Flamingo: a Visual Language Model for Few-Shot Learning* (2022)](https://arxiv.org/abs/2204.14198) — Gated cross-attention design enabling interleaved image–text few-shot inference (NeurIPS 2022).
    - [Li et al., *BLIP-2: Bootstrapping Language-Image Pre-Training with Frozen Image Encoders and Large Language Models* (2023)](https://arxiv.org/abs/2301.12597) — Q-Former token compression extracting fixed-size visual representations from any-resolution encoders.

    **Recent advances (2023–2026)**

    - [Liu et al., *Improved Baselines with Visual Instruction Tuning* (2023)](https://arxiv.org/abs/2310.03744) — LLaVA-1.5: MLP connector replacing the linear projector, achieving state-of-the-art on 11 benchmarks with 1.2M training samples.
    - [Bai et al., *Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond* (2023)](https://arxiv.org/abs/2308.12966) — Adds visual grounding and OCR-specific pretraining tasks to the standard VLM recipe.
    - [Chen et al., *InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks* (2023)](https://arxiv.org/abs/2312.14238) — Scales the vision encoder to 6B parameters (InternViT-6B), dramatically improving OCR and chart understanding (CVPR 2024 Oral).
    - [Wang et al., *Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution* (2024)](https://arxiv.org/abs/2409.12191) — Native dynamic resolution via M-RoPE positional encoding; 72B model matches GPT-4o on multimodal benchmarks.

    **Open-source & tools**

    - [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA) — Reference implementation of LLaVA through 1.6, including training scripts, LoRA fine-tuning, and SGLang serving integration.
    - [OpenGVLab/InternVL](https://github.com/OpenGVLab/InternVL) — Full InternVL family (1B–240B), training code, and evaluation scripts; leading open-source alternative to GPT-4o on MMMU.
    - [open-compass/VLMEvalKit](https://github.com/open-compass/VLMEvalKit) — One-command evaluation toolkit supporting 220+ VLMs across 80+ benchmarks (DocVQA, ChartQA, MMBench, etc.).

    **Go deeper**

    - [LLaVA-NeXT: Improved reasoning, OCR, and world knowledge](https://llava-vl.github.io/blog/2024-01-30-llava-next/) — Official blog post detailing the any-resolution tiling strategy and benchmark results for LLaVA-1.6.

## Further Reading

- **LLaVA** — Liu et al., "Visual Instruction Tuning," NeurIPS 2023. The minimal projector recipe.
- **LLaVA-1.5** — Liu et al., "Improved Baselines with Visual Instruction Tuning," CVPR 2024. MLP connector and better data.
- **LLaVA-1.6 (LLaVA-NeXT)** — Liu et al., 2024. Any-resolution tiling, higher-res tiles.
- **Flamingo** — Alayrac et al., "Flamingo: a Visual Language Model for Few-Shot Learning," NeurIPS 2022. Gated cross-attention design.
- **BLIP-2** — Li et al., "BLIP-2: Bootstrapping Language-Image Pre-Training with Frozen Image Encoders and Large Language Models," ICML 2023. Q-Former token compression.
- **InternVL** — Chen et al., "InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks," CVPR 2024.
- **Qwen-VL** — Bai et al., "Qwen-VL: A Versatile Vision-Language Model's Large Language Model," arXiv 2023.
- **ShareGPT4V** — Zhang et al., "ShareGPT4V: Improving Large Multi-Modal Models with Better Captions," arXiv 2023.
- **DocOwl** — Ye et al., "mPLUG-DocOwl: Modularized Multimodal Large Language Model for Document Understanding," arXiv 2023.

## Exercises

**1.** (Conceptual) In `MiniLLaVA.forward`, the visual token positions are given the label `-100` before the loss is computed. Explain in one or two sentences *why* this is necessary, and describe concretely what would go wrong during training if you forgot it and instead passed real vocabulary ids (or zeros) as labels for those positions.

??? note "Solution"
    The language-modeling loss is cross-entropy over the LLM vocabulary at each position. Visual tokens are continuous projected vectors that do **not** correspond to any vocabulary entry, so there is no meaningful "correct next token" to predict at a visual position. Setting the label to `-100` (PyTorch's `ignore_index`) means the loss is computed only over text positions, matching the chapter's objective

    $$
    \mathcal{L} = -\sum_{t \in \text{text positions}} \log p_\theta(x_t \mid \mathbf{H}_v, x_{<t}).
    $$

    If you instead supplied real ids (or zeros) as labels for the $N$ visual positions, the model would be trained to "predict the next visual token" from a text-classification head. That target is nonsensical: it injects spurious gradients that flow back through the projector (and the LLM), pulling the connector away from the alignment it is supposed to learn and corrupting training. Note the causal shift also means a bad label at the last visual position directly supervises the prediction of the first text token, so the damage is not confined to the visual block.

**2.** (Quantitative) You feed a $1008 \times 672$ document image to a LLaVA-1.6-style AnyRes pipeline built on CLIP ViT-L/14 at $336 \times 336$, using the chapter's `tile_image` logic (non-overlapping $336$-px tiles plus one global thumbnail tile). (a) How many $336 \times 336$ tiles are produced, and how many total tiles including the thumbnail? (b) How many patch tokens does each tile yield? (c) What is the total visual-token count for the image?

??? note "Solution"
    (a) `tile_image` sets `n_cols = round(1008/336) = 3` and `n_rows = round(672/336) = 2`, giving $3 \times 2 = 6$ full-resolution tiles. Since $6 \le$ `max_tiles = 6`, no clipping happens. Adding the one thumbnail tile gives $6 + 1 = 7$ tiles total.

    (b) Each $336 \times 336$ tile is split into $14 \times 14$ patches, so it has $(336/14)^2 = 24^2 = 576$ patch tokens.

    (c) Total visual tokens:

    $$
    7 \text{ tiles} \times 576 \text{ tokens/tile} = 4032 \text{ visual tokens.}
    $$

**3.** (Quantitative) Take the $4032$ visual tokens from Exercise 2 and place them in the KV cache of LLaMA-2-7B ($32$ layers, $D_\text{llm} = 4096$) in bf16 ($2$ bytes/element), using the chapter's costing (store both K and V). (a) What is the KV-cache size, in bytes and in GB, for just this visual prefix at batch size 1? (b) If a $2 \times 2$ average-pool were applied to each tile's patches before projection, cutting tokens by $4\times$, how much KV memory would the prefix use instead?

??? note "Solution"
    Per token, per layer, we store K and V: $2 \times 4096 \times 2\,\text{bytes} = 16{,}384$ bytes. Across $32$ layers that is $16{,}384 \times 32 = 524{,}288$ bytes/token ($=0.5$ MB/token).

    (a) For $4032$ tokens:

    $$
    4032 \times 524{,}288 = 2{,}113{,}929{,}216 \text{ bytes} = \frac{2{,}113{,}929{,}216}{1024^3} \approx 1.97 \text{ GB.}
    $$

    (b) A $2\times2$ average-pool reduces $4032$ tokens to $4032/4 = 1008$ tokens, so the KV cost falls by exactly $4\times$:

    $$
    1008 \times 524{,}288 = 528{,}482{,}304 \text{ bytes} \approx 0.49 \text{ GB.}
    $$

    Pooling turns a ~2 GB visual prefix into ~0.5 GB, at the cost of coarser spatial detail (a real concern for the OCR use-case this document image implies).

**4.** (Conceptual) You must serve a 5-shot, image-interleaved prompt of the form `image1, caption1, ..., image5, caption5, image6, ?` where the model should describe `image6`. Compare the projector (LLaVA-style) and cross-attention (Flamingo-style) connectors for this task along two axes: (a) how each handles the *interleaving* of 6 images with text, and (b) roughly how much LLM context each image consumes. Using the chapter's numbers ($576$ tokens/image), quantify the projector's context cost for all 6 images.

??? note "Solution"
    (a) **Interleaving.** The Flamingo cross-attention design handles this natively: images are stored externally as vision features, and interleaved gated cross-attention layers let each text token attend to the most recently preceding image via an attention mask. So `image1, caption1, ..., image6, ?` is a single natural context. The projector instead splices each image's tokens into the sequence at its `<image>` placeholder; interleaving is possible but the images compete directly for sequence positions, and every image permanently occupies context.

    (b) **Context cost.** With a projector, each image costs its full visual-token budget in the LLM stream; with cross-attention the LLM residual stream is unchanged and the image consumes *zero* context positions (it lives in the external KV of the cross-attn layers). For the projector at $576$ tokens/image:

    $$
    6 \text{ images} \times 576 = 3456 \text{ visual tokens},
    $$

    consumed *before* any of the caption text is counted. For dense many-image few-shot prompts, this is exactly the regime where Flamingo-style cross-attention was designed to win; the projector is favored when there are few images and training simplicity matters.

**5.** (Implementation) Implement a `pool_2x2` connector step that average-pools the $2 \times 2$ neighborhoods of a tile's CLIP patch tokens *before* the projector, reducing visual tokens by $4\times$ (the "average pooling" technique from the token-compression section). It should take patch embeddings of shape `[B, 576, D_v]` (CLS already dropped, from a $24 \times 24$ patch grid) and return `[B, 144, D_v]`, correctly respecting the 2-D spatial layout of the patches. Then show where it slots into `MiniLLaVA.encode_image`.

??? note "Solution"
    The patch tokens are laid out row-major over a $24 \times 24$ grid, so token index $n = r \cdot 24 + c$. To pool $2\times2$ spatial neighborhoods we must first restore that 2-D grid, pool, then flatten back to a sequence.

    ```python
    import torch
    import torch.nn.functional as F

    def pool_2x2(patch_embeds: torch.Tensor, grid: int = 24) -> torch.Tensor:
        """
        Average-pool 2x2 neighborhoods of CLIP patch tokens to cut the
        visual-token count by 4x before projection.

        patch_embeds: [B, grid*grid, D_v]   (CLS already dropped)
        returns:      [B, (grid//2)**2, D_v]
        """
        B, N, D = patch_embeds.shape
        assert N == grid * grid, "patch count must equal grid*grid"
        # [B, N, D] -> [B, D, N] -> [B, D, grid, grid]  (row-major grid)
        x = patch_embeds.transpose(1, 2).reshape(B, D, grid, grid)
        # Pool 2x2 spatial neighborhoods -> [B, D, grid/2, grid/2]
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        # Flatten back to a token sequence: [B, (grid/2)**2, D]
        x = x.flatten(2).transpose(1, 2)
        return x
    ```

    It slots into `encode_image` between dropping the CLS token and projecting:

    ```python
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():                    # vision encoder frozen
            vision_out = self.vision_model(pixel_values=pixel_values)
        patch_embeds = vision_out.last_hidden_state[:, 1:, :]  # [B, 576, D_v]
        patch_embeds = pool_2x2(patch_embeds, grid=24)         # [B, 144, D_v]
        visual_tokens = self.projector(patch_embeds)           # [B, 144, D_llm]
        return visual_tokens
    ```

    Quick check:

    ```python
    x = torch.randn(2, 576, 1024)   # [B, N_patches, D_v]
    print(pool_2x2(x).shape)        # torch.Size([2, 144, 1024])
    ```

    Pooling *before* the projector (rather than after) keeps the projector's input dimension at $D_v$ and cuts both the projector's per-image FLOPS and the downstream visual-token count by $4\times$, at the cost of halving spatial resolution to a $12 \times 12$ grid.
