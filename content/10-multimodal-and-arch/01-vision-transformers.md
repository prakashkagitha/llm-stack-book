# 10.1 Vision Transformers & Image Encoders

Images are not tokens — and yet modern vision models treat them exactly like tokens. The insight that a 224×224 pixel image can be split into a sequence of 196 small patches, each embedded into a vector exactly like a word, and then fed into an unmodified Transformer, turned out to be one of the most generative ideas in the last decade of deep learning. This chapter builds that idea from the ground up: how you patchify an image, why it works, what goes wrong if you do it naively, and how CLIP, SigLIP, DINOv2, and modern multimodal pipelines extend the basic ViT into the powerful image encoders that power GPT-4o, Gemini, and Claude.

Before reading this chapter, make sure you are comfortable with [The Attention Mechanism From Scratch](../02-transformer/03-attention-from-scratch.html) and [The Transformer Block: Norms, Residuals, MLPs & Activations](../02-transformer/06-transformer-block.html). This chapter feeds directly into [Vision-Language Models](../10-multimodal-and-arch/02-vision-language-models.html), where we cover how a trained image encoder is wired into a language model.

---

## Why Convolutions Are Not Enough

{{fig:vit-patchify}}

Convolutional Neural Networks (CNNs) dominated image recognition from AlexNet (2012) through EfficientNet (2019). They work by sliding local filters across the image in a hierarchy, gradually building up larger receptive fields. This inductive bias — locality and translation equivariance — is well-matched to natural images, and it made CNNs sample-efficient and fast.

So why replace them? Three reasons.

**Limited global context.** A pixel in the top-left of the image cannot directly attend to a pixel in the bottom-right until very deep layers, where the effective receptive field has grown large. Modeling long-range spatial dependencies (a person's face relative to their hands) requires many layers or dilated convolutions.

**Awkward text-image unification.** When building a model that jointly processes text and images, two separate architecture families (Transformer for text, CNN for images) create a fundamental impedance mismatch. Every cross-modal attention mechanism becomes a bespoke engineering project.

**Scaling behavior.** Transformers scale more predictably with data and compute than CNNs. When pre-trained on very large datasets, a Transformer-based image encoder tends to outperform CNN baselines, even though CNNs are often better with limited data.

The ViT (Vision Transformer), introduced by Dosovitskiy et al. in "An Image Is Worth 16x16 Words" (2020), solves all three problems with an elegant single insight: **treat image patches as tokens**.

---

## The Patchify Operation

### From Pixels to Tokens

An input image $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$ is split into $N$ non-overlapping patches of size $P \times P$:

$$
N = \frac{H \times W}{P^2}
$$

For the canonical ViT-B/16 configuration: $H=W=224$, $P=16$, $C=3$, giving $N = \frac{224 \times 224}{16^2} = 196$ patches. Each patch is a $16 \times 16 \times 3 = 768$-dimensional flattened vector. A linear projection maps each flattened patch to the model's hidden dimension $D$:

$$
\mathbf{z}_0 = [\mathbf{x}_\text{cls};\; \mathbf{E}\mathbf{p}_1;\; \mathbf{E}\mathbf{p}_2;\; \ldots;\; \mathbf{E}\mathbf{p}_N] + \mathbf{E}_\text{pos}
$$

where $\mathbf{E} \in \mathbb{R}^{D \times (P^2 C)}$ is the patch embedding matrix, $\mathbf{p}_i$ is the $i$-th flattened patch, and $\mathbf{E}_\text{pos} \in \mathbb{R}^{(N+1) \times D}$ is a learned positional embedding table. The prepended $\mathbf{x}_\text{cls}$ is a learnable classification token, analogous to BERT's `[CLS]`.

### Image Preprocessing

Before patchification, images must be normalized into a consistent distribution. The standard pipeline is:

1. **Resize and crop** — resize the shorter edge to 256, center-crop to 224×224 (or random-crop during training).
2. **Convert to float** — divide pixel values by 255 to get $[0, 1]$.
3. **Normalize channels** — subtract per-channel mean and divide by per-channel standard deviation, typically using ImageNet statistics: mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225).

This maps pixel intensities to a zero-mean, unit-variance distribution, making the linear patch projection well-conditioned.

### Code: Patchify + Patch Embedding

```python
import torch
import torch.nn as nn
import einops  # pip install einops

# ---------------------------------------------------------------------------
# patchify: split an image tensor into a sequence of patch vectors
# ---------------------------------------------------------------------------
def patchify(images: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    """
    Args:
        images: (B, C, H, W) float tensor, pre-normalized
        patch_size: side length of each square patch (P)
    Returns:
        patches: (B, N, P*P*C) where N = (H/P) * (W/P)
    """
    B, C, H, W = images.shape
    assert H % patch_size == 0 and W % patch_size == 0, \
        f"Image size ({H},{W}) must be divisible by patch_size={patch_size}"
    
    # Rearrange from (B, C, H, W) → (B, N, P*P*C)
    # 'h1 h2 w1 w2' means height is split into h1 groups of h2=P rows, same for width
    patches = einops.rearrange(
        images,
        'b c (h1 h2) (w1 w2) -> b (h1 w1) (h2 w2 c)',
        h2=patch_size, w2=patch_size
    )
    return patches  # (B, N, P*P*C)


# ---------------------------------------------------------------------------
# PatchEmbed: learnable linear projection of patches into D-dim vectors
# ---------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    """
    Equivalent to a Conv2d with kernel_size=stride=patch_size, but implemented
    as a matrix multiply for clarity. In practice, nn.Conv2d is often used
    because it can be more cache-efficient on hardware.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2   # 196 for 224/16
        self.patch_dim = patch_size * patch_size * in_chans  # 768 = 16*16*3

        # A single weight matrix E ∈ R^{D × (P²C)}
        self.proj = nn.Linear(self.patch_dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) pre-normalized image
        returns: (B, N, D) patch embeddings
        """
        patches = patchify(x, self.patch_size)  # (B, N, P²C)
        return self.proj(patches)               # (B, N, D)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B, C, H, W = 4, 3, 224, 224
    images = torch.randn(B, C, H, W)

    embed = PatchEmbed(img_size=224, patch_size=16, in_chans=3, embed_dim=768)
    out = embed(images)
    print(f"Input shape:  {images.shape}")   # (4, 3, 224, 224)
    print(f"Output shape: {out.shape}")       # (4, 196, 768)
    print(f"Params in PatchEmbed: {sum(p.numel() for p in embed.parameters()):,}")
    # Expected: 768 * 768 + 768 = 590,592  (weight + bias)
```

!!! example "Worked example: parameter count and memory for ViT-B/16"
    ViT-B/16 has $D=768$, 12 Transformer layers, 12 heads.

    **Patch embedding matrix**: $D \times (P^2 C) = 768 \times 768 = 589{,}824$ parameters.

    **Positional embedding table**: $(N+1) \times D = 197 \times 768 = 151{,}296$ parameters.

    **CLS token**: $D = 768$ parameters.

    **Each Transformer block** (QKV projections + output proj + two LayerNorms + MLP):
    $$4 \times D^2 + 2 \times 4D^2 + 4D = 4 \times 768^2 + 8 \times 768^2 = 12 \times 768^2 \approx 7.08\text{M}$$

    12 layers × 7.08M = 84.9M, plus ~0.85M for embeddings and the final norm/head.
    Total: roughly **86M parameters**, or about **344 MB in fp32**, **172 MB in bf16**.

    **Sequence length**: 196 image patches + 1 CLS = **197 tokens**.
    Attention is $O(N^2 D)$ per layer, not unbearably long for this patch size — but dynamic high-resolution inputs (e.g., 1024×1024 with 16-pixel patches) give $N=4096$, which is where [FlashAttention 2 & 3](../04-kernels-efficiency/03-flash-attention-2-3.html) becomes critical.

---

## The Full ViT Architecture

With patch embeddings in hand, the rest of ViT is a vanilla Transformer encoder. The architecture is deliberately unchanged from BERT:


{{fig:vit-arch-full-pipeline}}


### Code: Minimal ViT Block

```python
import torch
import torch.nn as nn
import math


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention — see chapter 2.4 for the full treatment."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)  # 1/√d_k scaling factor

        # Fused QKV projection: one matrix, split afterward
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D)  where N = num_patches+1 (includes CLS)
        returns: (B, N, D)
        """
        B, N, D = x.shape
        H = self.num_heads

        # Project and split into queries, keys, values
        qkv = self.qkv(x)                         # (B, N, 3D)
        q, k, v = qkv.chunk(3, dim=-1)            # each (B, N, D)

        # Reshape to (B, H, N, head_dim) for multi-head computation
        def reshape(t):
            return t.view(B, N, H, self.head_dim).transpose(1, 2)
        q, k, v = reshape(q), reshape(k), reshape(v)  # each (B, H, N, head_dim)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / self.scale   # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Aggregate values and reshape back
        out = (attn @ v)                          # (B, H, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)                 # (B, N, D)


class ViTBlock(nn.Module):
    """
    One Transformer block in ViT, using Pre-LayerNorm (LN before attention).
    Pre-LN improves training stability vs the original Post-LN used by BERT.
    See: chapter 2.6 for a detailed discussion of norm placement.
    """
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)  # typically 4× hidden dim = 3072

        # Two-layer MLP with GELU activation — standard ViT uses GELU
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN residual: x = x + Attn(LN(x))
        x = x + self.attn(self.norm1(x))
        # Pre-LN residual: x = x + MLP(LN(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Minimal ViT implementation matching the ViT-B/16 configuration.
    For the full model (ViT-L, ViT-H, ViT-G), only embed_dim, num_heads,
    and depth change — everything else stays the same.
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches  # 196

        # Learnable CLS token (prepended to the patch sequence)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Learnable positional embeddings for N+1 positions (patches + CLS)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.pos_drop = nn.Dropout(dropout)

        # Stack of Transformer blocks
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head: linear layer on top of the CLS token
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights following the ViT paper
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W) pre-normalized images
        returns: (B, num_classes) logits
        """
        B = x.shape[0]
        x = self.patch_embed(x)   # (B, N, D)

        # Expand and prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls, x], dim=1)           # (B, N+1, D)

        # Add positional embeddings and apply dropout
        x = x + self.pos_embed                   # (B, N+1, D)
        x = self.pos_drop(x)

        # Apply Transformer blocks
        for block in self.blocks:
            x = block(x)                         # (B, N+1, D)

        x = self.norm(x)                         # (B, N+1, D)

        # Extract CLS token output and classify
        cls_out = x[:, 0]                        # (B, D)
        return self.head(cls_out)                # (B, num_classes)


# Verify shapes
if __name__ == "__main__":
    model = VisionTransformer()  # ViT-B/16 defaults
    imgs = torch.randn(2, 3, 224, 224)
    logits = model(imgs)
    print(f"Logits shape: {logits.shape}")   # (2, 1000)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")    # ~86M
```

---

## Positional Embeddings in ViT

ViT uses **learned** 1D positional embeddings by default — one embedding vector per patch position (0 to $N$, including CLS). This contrasts with the sinusoidal scheme in the original Transformer (see [Positional Encodings: Sinusoidal, Learned, RoPE & ALiBi](../02-transformer/05-positional-encoding.html)).

**Why not 2D?** In practice, 1D learned embeddings work nearly as well as 2D variants, partly because the model learns 2D structure implicitly. The patches have a natural 2D grid structure, and the positional embedding that gets added to patch $(i, j)$ effectively encodes the spatial position.

**Resolution generalization** is a real problem: if you train ViT-B/16 on 224×224 and want to fine-tune on 384×384 (giving $N = 576$ patches), the learned position embeddings are the wrong size. The standard fix is **bicubic interpolation** of the position embedding grid — treating the 14×14 embedding table as a 2D feature map and resizing it to 24×24.

{{fig:pos-embed-interpolation}}

```python
import torch
import torch.nn.functional as F

def interpolate_pos_embed(pos_embed: torch.Tensor,
                          new_h: int, new_w: int,
                          patch_size: int = 16) -> torch.Tensor:
    """
    Interpolate positional embeddings from training resolution to a new resolution.

    Args:
        pos_embed: (1, N+1, D) learned position embeddings (includes CLS at index 0)
        new_h, new_w: new image height and width in pixels
        patch_size: patch size in pixels
    Returns:
        (1, new_N+1, D) interpolated positional embeddings
    """
    # Separate CLS token from patch position embeddings
    cls_pos = pos_embed[:, :1, :]           # (1, 1, D)
    patch_pos = pos_embed[:, 1:, :]         # (1, N, D)

    # Infer original grid size from N = h_orig * w_orig
    N_orig = patch_pos.shape[1]
    h_orig = w_orig = int(N_orig ** 0.5)    # e.g., 14 for 196 patches
    D = patch_pos.shape[-1]

    # Reshape to 2D grid, interpolate
    patch_pos = patch_pos.reshape(1, h_orig, w_orig, D)
    patch_pos = patch_pos.permute(0, 3, 1, 2)  # (1, D, h_orig, w_orig)

    new_h_patches = new_h // patch_size
    new_w_patches = new_w // patch_size

    patch_pos = F.interpolate(
        patch_pos,
        size=(new_h_patches, new_w_patches),
        mode='bicubic',
        align_corners=False
    )  # (1, D, new_h_patches, new_w_patches)

    patch_pos = patch_pos.permute(0, 2, 3, 1)  # (1, new_h_patches, new_w_patches, D)
    patch_pos = patch_pos.reshape(1, -1, D)     # (1, new_N, D)

    return torch.cat([cls_pos, patch_pos], dim=1)  # (1, new_N+1, D)
```

---

## CLIP: Contrastive Language-Image Pre-training

ViT trained on ImageNet learns to classify 1000 categories. CLIP (Radford et al., OpenAI, 2021) generalized this dramatically: train an image encoder and a text encoder jointly on 400 million (image, alt-text) pairs from the internet, using a **contrastive objective** that pulls matching pairs together and pushes mismatched pairs apart.

### The CLIP Loss

Given a batch of $N$ (image, text) pairs, CLIP computes embeddings:

$$
\mathbf{i}_k = f_\text{image}(\mathbf{x}_k) / \|f_\text{image}(\mathbf{x}_k)\|_2, \quad
\mathbf{t}_k = f_\text{text}(\mathbf{w}_k) / \|f_\text{text}(\mathbf{w}_k)\|_2
$$

and then computes a similarity matrix $\mathbf{S} \in \mathbb{R}^{N \times N}$:

$$
S_{ij} = \tau \cdot \mathbf{i}_i \cdot \mathbf{t}_j^\top
$$

where $\tau$ is a learnable temperature parameter (initialized around $1/0.07 \approx 14.3$). The loss is symmetric cross-entropy applied both row-wise (each image's text should rank first) and column-wise (each text's image should rank first):

$$
\mathcal{L}_\text{CLIP} = -\frac{1}{2N} \left( \sum_{k=1}^N \log \frac{e^{S_{kk}}}{\sum_j e^{S_{kj}}} + \sum_{k=1}^N \log \frac{e^{S_{kk}}}{\sum_i e^{S_{ik}}} \right)
$$

This is InfoNCE loss applied in two directions. The loss essentially treats every off-diagonal pair in the batch as a negative example, which is why CLIP needs very large batch sizes (on the order of 32,768) to have enough negatives.

### Architectural Choices

CLIP uses a **ViT** or a **ResNet** as the image encoder, and a Transformer (similar to GPT-2) as the text encoder. For the image encoder, ViT-L/14 (patch size 14, large model) is the standard CLIP backbone. Both encoders project their final representation into a shared 512- or 768-dimensional space via a linear projection head.

At inference, CLIP enables **zero-shot classification**: compute the cosine similarity between the image embedding and the embeddings of text prompts like "a photo of a cat", "a photo of a dog", etc. The predicted class is the one with the highest similarity.

```python
import torch
import torch.nn.functional as F


def clip_contrastive_loss(image_embeds: torch.Tensor,
                          text_embeds: torch.Tensor,
                          logit_scale: torch.Tensor) -> torch.Tensor:
    """
    Compute the symmetric CLIP contrastive loss (InfoNCE in both directions).

    Args:
        image_embeds: (N, D) L2-normalized image embeddings
        text_embeds:  (N, D) L2-normalized text embeddings
        logit_scale:  scalar parameter, stored as log(τ) and exp()'d for stability
    Returns:
        scalar loss
    """
    # Exponentiate the log-temperature; clamp to prevent collapse or explosion
    scale = logit_scale.exp().clamp(max=100.0)

    # Compute (N, N) similarity matrix: S[i,j] = τ * <image_i, text_j>
    logits_per_image = scale * image_embeds @ text_embeds.t()  # (N, N)
    logits_per_text  = logits_per_image.t()                   # (N, N)

    # Ground-truth: diagonal is the positive pair
    N = image_embeds.shape[0]
    labels = torch.arange(N, device=image_embeds.device)

    # Cross-entropy in both directions, then average
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text,  labels)
    return (loss_i + loss_t) / 2.0


def zero_shot_classify(image_embed: torch.Tensor,
                       class_text_embeds: torch.Tensor) -> torch.Tensor:
    """
    Args:
        image_embed:       (D,)   L2-normalized embedding of a single image
        class_text_embeds: (C, D) L2-normalized embeddings for C class prompts
    Returns:
        (C,) softmax probabilities
    """
    # Cosine similarities (temperature=1 here; CLIP uses learned τ)
    sims = image_embed @ class_text_embeds.t()   # (C,)
    return sims.softmax(dim=-1)
```

---

## SigLIP: Replacing InfoNCE with Sigmoid Loss

CLIP's softmax-based InfoNCE loss has a subtle problem: computing $\log \sum_j e^{S_{kj}}$ requires gathering the full batch's logits to one device, which creates a communication bottleneck when training on thousands of GPUs. Zhai et al. (Google, "Sigmoid Loss for Language Image Pre-Training", 2023) proposed **SigLIP** to address this.

{{fig:clip-siglip-contrastive-matrix}}

Instead of a softmax over $N$ negatives, SigLIP applies a **sigmoid** binary cross-entropy to each pair independently:

$$
\mathcal{L}_\text{SigLIP} = -\frac{1}{N^2} \sum_{i=1}^N \sum_{j=1}^N
\left[ y_{ij} \log \sigma(\tau \cdot \mathbf{i}_i \cdot \mathbf{t}_j + b)
+ (1 - y_{ij}) \log \sigma(-\tau \cdot \mathbf{i}_i \cdot \mathbf{t}_j - b) \right]
$$

where $y_{ij} = 1$ if $i = j$ (positive pair) and $0$ otherwise, and $b$ is a learnable bias initialized to a negative value (around $-10$) to counteract the class-imbalance of having many more negatives than positives in the full $N^2$ grid.

**Key advantages of SigLIP over CLIP:**
- No global normalization (softmax denominator), so loss computation shards trivially across devices.
- Better accuracy with smaller batch sizes, because sigmoid loss does not need large $N$ to have a meaningful denominator.
- A learnable bias $b$ lets the model calibrate the raw similarity threshold.

SigLIP forms the image encoder backbone in several recent vision-language models (for example, the Gemini/PaliGemma family uses SigLIP-So400M-14).

---

## DINOv2: Self-Supervised Vision Features

While CLIP and SigLIP rely on paired image-text data, DINOv2 (Oquab et al., Meta AI, 2023) learns powerful image representations using **only images**, via self-supervised distillation. It builds on DINO (self-DIstillation with NO labels) and DINO v2 introduces:

1. **A curated dataset** (LVD-142M) filtered for high-quality, diverse images using embedding-space deduplication.
2. **A student-teacher distillation setup**: a student ViT processes augmented crops; a teacher ViT (EMA of the student weights) processes a larger crop. The student is trained to match the teacher's patch-level features.
3. **DINO + iBOT objectives**: DINO aligns the CLS tokens (global features); iBOT (image BERT) masks random patches and predicts the teacher's patch representations, learning local spatial features.
4. **Register tokens**: Newly introduced learnable tokens appended to the sequence (discussed below).

DINOv2 models (ViT-S, ViT-B, ViT-L, ViT-G/14) produce exceptionally clean spatial features: patch attention maps reveal semantic regions without any dense annotation.

### Register Tokens

Darcet et al. (Meta AI, "Vision Transformers Need Registers", 2023) discovered that ViT attention maps often contain **artifact tokens** — patches with anomalously high attention scores that don't correspond to semantic content. These arise when the model has too few positions to store global information, forcing it to "park" redundant global context in arbitrary high-norm patch tokens.

{{fig:vit-register-tokens}}

The fix is **register tokens**: $R$ extra learnable tokens (typically $R = 4$ or $8$) appended to the sequence before the Transformer blocks. They serve as scratch space for global reasoning:

$$
\mathbf{z}_0 = [\mathbf{x}_\text{cls};\; \mathbf{E}\mathbf{p}_1;\; \ldots;\; \mathbf{E}\mathbf{p}_N;\; \mathbf{r}_1;\; \ldots;\; \mathbf{r}_R]
$$

The register tokens are discarded at the output; only the patch tokens and CLS token are used downstream. With registers, DINOv2's spatial attention maps become significantly cleaner and spatial tasks (depth estimation, segmentation) improve.

```python
class VisionTransformerWithRegisters(nn.Module):
    """
    ViT with register tokens, following Darcet et al. (2023).
    Register tokens are appended after patch tokens and discarded at output.
    """
    def __init__(
        self,
        img_size=224, patch_size=14, in_chans=3, embed_dim=1024,
        depth=24, num_heads=16, mlp_ratio=4.0,
        num_registers=4,    # key new parameter
        num_classes=0,      # 0 = return features, not logits
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Register tokens — no positional embedding added to them
        self.num_registers = num_registers
        self.register_tokens = nn.Parameter(
            torch.zeros(1, num_registers, embed_dim)
        )

        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.register_tokens, std=0.02)

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        patches = self.patch_embed(x)                     # (B, N, D)

        # Build sequence: [CLS | patches] with positional embeddings
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, patches], dim=1)              # (B, N+1, D)
        x = x + self.pos_embed                            # add pos embeddings

        # Append register tokens (no positional embedding for registers)
        regs = self.register_tokens.expand(B, -1, -1)
        x = torch.cat([x, regs], dim=1)                   # (B, N+1+R, D)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Discard register tokens; return CLS and patch tokens
        # Register tokens occupy the last R positions
        x = x[:, :-self.num_registers]                    # (B, N+1, D)
        cls_out = x[:, 0]                                 # (B, D)
        patch_out = x[:, 1:]                              # (B, N, D)

        return cls_out, patch_out   # global feature, spatial features
```

---

## From Raw Pixels to Transformer Input: The Full Pipeline

Let's put the preprocessing and patchification pipeline together as it appears in production code:

```python
import torchvision.transforms.v2 as T
from PIL import Image
import torch

# Standard ViT/CLIP preprocessing (ImageNet statistics)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# CLIP uses slightly different mean/std
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


def build_vit_transform(img_size: int = 224,
                        mean=IMAGENET_MEAN,
                        std=IMAGENET_STD,
                        is_train: bool = False) -> T.Compose:
    """
    Build the standard preprocessing pipeline for a ViT model.

    During training, we add random resized crop and horizontal flip for
    data augmentation. During inference, we use deterministic center crop.
    """
    if is_train:
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.2, 1.0)),  # random crop
            T.RandomHorizontalFlip(),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),   # → [0, 1]
            T.Normalize(mean=mean, std=std),         # → ~N(0,1)
        ])
    else:
        return T.Compose([
            T.Resize(int(img_size * 256 / 224)),    # resize short edge
            T.CenterCrop(img_size),                 # deterministic center crop
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=mean, std=std),
        ])


# Example usage
if __name__ == "__main__":
    transform = build_vit_transform(224, is_train=False)

    # Load a PIL image
    img = Image.new("RGB", (640, 480), color=(128, 64, 32))
    tensor = transform(img)               # (3, 224, 224)
    print(f"Preprocessed tensor shape: {tensor.shape}")    # torch.Size([3, 224, 224])
    print(f"Value range: [{tensor.min():.2f}, {tensor.max():.2f}]")  # ~[-2, 2]

    # Batch and feed to model
    batch = tensor.unsqueeze(0)           # (1, 3, 224, 224)
    model = VisionTransformer()
    model.eval()
    with torch.no_grad():
        logits = model(batch)
    print(f"Logits shape: {logits.shape}")  # (1, 1000)
```

!!! warning "Common pitfall: wrong normalization statistics"
    Using ImageNet statistics for a CLIP model (or vice versa) is a frequent source of bugs that can silently degrade downstream performance by several percent. Always check which normalization constants a checkpoint was trained with. Some models (e.g., SigLIP) use image-specific mean/std; others use a simple $[-1, 1]$ rescaling with mean=0.5, std=0.5. Store preprocessing alongside the model checkpoint.

---

## Comparing the Major Vision Encoder Families

| Model | Training Signal | Architecture | Key Innovation | Typical Use Case |
|-------|-----------------|--------------|----------------|-----------------|
| ViT-B/16 | Supervised (ImageNet) | ViT | Patches as tokens | Classification baseline |
| ViT-L/14 | Supervised (JFT-300M) | ViT | Scale | High-accuracy backbone |
| CLIP ViT-L/14 | Image-text contrastive | ViT + text Transformer | Zero-shot, joint embedding | VLMs, zero-shot retrieval |
| SigLIP-So400M/14 | Image-text sigmoid loss | ViT-So400M | No global softmax, small batches | Gemini/PaliGemma backbone |
| DINOv2 ViT-G/14 | Self-supervised distillation | ViT + registers | Dense spatial features, no labels | Segmentation, depth, VLMs |
| EVA-CLIP | Image-text + masked prediction | ViT-E | Scalable vision encoder | Open-source VLMs |

!!! note "ViT-G vs ViT-H vs ViT-E: a quick size guide"
    ViT-B (86M) < ViT-L (307M) < ViT-H (632M) < ViT-G (1.1B) < ViT-22B (22B).
    The naming is not perfectly consistent across papers; always check `embed_dim` and `depth` rather than the letter designation alone.

---

## Interview Corner & Key Takeaways

!!! interview "Interview Corner"
    **Q:** You want to adapt a CLIP ViT-L/14 (trained on 224×224 images) to process 448×448 images. What breaks and how do you fix it?

    **A:** The patch embedding weights are fine (they embed $14 \times 14$ patches regardless of image size). What breaks is the **positional embedding table**: trained on $16 \times 16 = 256$ patch positions, but a 448×448 image with 14-pixel patches has $32 \times 32 = 1024$ positions. You fix this with **bicubic interpolation** of the 2D position embedding grid: reshape the 256 embeddings into a 16×16 grid, resize to 32×32 via bicubic interpolation, and flatten. This is the standard `interpolate_pos_embed` trick used during fine-tuning. After interpolation, a few gradient steps to fine-tune the position embeddings at the new resolution generally restores most of the accuracy. Alternatively, RoPE (Rotary Position Embeddings) can be used instead of learned embeddings precisely to avoid this resolution-lock problem — several recent ViT variants adopt 2D RoPE for this reason.

!!! interview "Interview Corner"
    **Q:** What is the difference between CLIP and SigLIP, and when would you prefer one over the other?

    **A:** Both are contrastive image-text models. CLIP uses a softmax (InfoNCE) loss that requires computing a normalization term over all $N$ items in the batch — demanding very large batches (tens of thousands) and global gather operations across GPUs. SigLIP replaces this with a per-pair sigmoid binary cross-entropy that avoids global normalization. SigLIP scales more gracefully to large fleets of GPUs and works better at smaller batch sizes. If you have abundant GPU memory and want the absolute best zero-shot accuracy at a given model size, CLIP with large batches is competitive; SigLIP is generally the better engineering choice for modern multimodal training pipelines. SigLIP is the backbone used in PaliGemma and the Gemini visual encoder family.

!!! key "Key Takeaways"
    - ViT divides an image into non-overlapping $P \times P$ patches, flattens each to a vector, and linearly projects them — treating patches exactly like word tokens. For ViT-B/16 on 224×224 images this gives 196 tokens of dimension 768.
    - A learned CLS token is prepended; its final output serves as the global image representation for classification. Alternatively, the patch tokens can be averaged (patch-avg pooling) or used as spatial features for dense tasks.
    - Positional embeddings in ViT are learned, not sinusoidal. When changing resolution, the 2D grid must be bicubically interpolated to generate embeddings for the new number of patches.
    - CLIP trains an image encoder and a text encoder jointly with symmetric InfoNCE contrastive loss on 400M+ image-text pairs, enabling zero-shot classification by comparing image embeddings to text prompt embeddings.
    - SigLIP replaces CLIP's softmax loss with sigmoid binary cross-entropy applied per pair, eliminating the need for large batch sizes and global gather operations — preferred in modern large-scale training pipelines.
    - DINOv2 learns rich spatial features with no text supervision by distilling a teacher ViT into a student, jointly optimizing CLS-level (DINO) and patch-level (iBOT) objectives.
    - Register tokens (Darcet et al., 2023) add $R$ extra learnable positions to absorb global "scratch" information, eliminating the artifact tokens that appear in plain ViT attention maps. Registers are discarded at the output.
    - Image preprocessing (resize, crop, normalize) must match the training pipeline exactly. CLIP and ImageNet models use different normalization constants; mixing them silently degrades performance.
    - These image encoders are the building block for the [Vision-Language Models](../10-multimodal-and-arch/02-vision-language-models.html) discussed in the next chapter, where patch token sequences are fused into a language model's context.

---

!!! sota "State of the Art & Resources (2026)"
    Vision Transformers are the dominant image encoder architecture across classification, dense prediction, and multimodal systems; the field has shifted from supervised ImageNet training toward large-scale contrastive (CLIP/SigLIP) and self-supervised (DINOv2) pre-training, with encoder scale now reaching 22B parameters.

    **Foundational work**

    - [Dosovitskiy et al., *An Image Is Worth 16×16 Words* (2020)](https://arxiv.org/abs/2010.11929) — the paper that established patch-based ViT and proved a pure Transformer matches CNNs at scale.
    - [Radford et al. (OpenAI), *Learning Transferable Visual Models From Natural Language Supervision* (2021)](https://arxiv.org/abs/2103.00020) — CLIP: contrastive image-text pre-training enabling zero-shot classification.

    **Recent advances (2023–2026)**

    - [Zhai et al. (Google), *Sigmoid Loss for Language Image Pre-Training* (2023)](https://arxiv.org/abs/2303.15343) — SigLIP replaces CLIP's softmax with per-pair sigmoid loss, removing the global-gather bottleneck; backbone of PaliGemma and Gemini.
    - [Oquab et al. (Meta AI), *DINOv2: Learning Robust Visual Features without Supervision* (2023)](https://arxiv.org/abs/2304.07193) — self-supervised distillation on 142M curated images yields all-purpose spatial features that outperform weakly-supervised encoders.
    - [Darcet et al. (Meta AI), *Vision Transformers Need Registers* (2023)](https://arxiv.org/abs/2309.16588) — identifies high-norm artifact tokens in ViT attention maps and fixes them with learnable register tokens, improving dense-prediction quality.
    - [Dehghani et al. (Google), *Scaling Vision Transformers to 22 Billion Parameters* (2023)](https://arxiv.org/abs/2302.05442) — ViT-22B shows LLM-like scaling laws in vision with parallel layers and QK-norm for training stability.
    - [Fang et al. (BAAI), *EVA: Exploring the Limits of Masked Visual Representation Learning at Scale* (2022)](https://arxiv.org/abs/2211.07636) — masked image reconstruction of CLIP features scales ViT to 1B parameters, forming the EVA-CLIP open-source encoder family.

    **Open-source & tools**

    - [huggingface/pytorch-image-models (timm)](https://github.com/huggingface/pytorch-image-models) — the largest collection of PyTorch vision backbones (ViT, DeiT, SigLIP, EVA-CLIP, DINOv2) with pretrained weights and training scripts.
    - [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) — official Meta AI code and pretrained ViT-S/B/L/G DINOv2 models, including register-token variants.
    - [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) — open-source CLIP training codebase supporting LAION-2B, DataComp, and custom datasets; includes SigLIP variants.

    **Go deeper**

    - [Meta AI Blog: *DINOv2: State-of-the-art computer vision models with self-supervised learning* (2023)](https://ai.meta.com/blog/dino-v2-computer-vision-self-supervised-learning/) — accessible overview of DINOv2's design, capabilities, and real-world applications.

## Further Reading

- Dosovitskiy et al., "An Image Is Worth 16×16 Words: Transformers for Image Recognition at Scale" (2020) — the original ViT paper.
- Radford et al. (OpenAI), "Learning Transferable Visual Models From Natural Language Supervision" (2021) — CLIP.
- Zhai et al. (Google), "Sigmoid Loss for Language Image Pre-Training" (SigLIP, 2023).
- Oquab et al. (Meta AI), "DINOv2: Learning Robust Visual Features without Supervision" (2023).
- Darcet et al. (Meta AI), "Vision Transformers Need Registers" (2023) — register tokens.
- Fang et al., "EVA: Exploring the Limits of Masked Visual Representation Learning at Scale" (2022) — EVA-CLIP.
- Touvron et al., "Training data-efficient image transformers & distillation through attention" (DeiT, 2021) — training ViT with limited data via distillation.
- timm library (Ross Wightman) — the de facto PyTorch hub for pretrained vision models; includes ViT, DeiT, DINOv2, EVA-CLIP, and many variants: `github.com/huggingface/pytorch-image-models`.

---

## Exercises

**1.** (Conceptual) The chapter states that ViT "treats image patches as tokens" and feeds them into an *unmodified* Transformer encoder. Yet a language model's self-attention over word tokens is usually **causal** (each token attends only to earlier tokens), while ViT's is not. Why does ViT use *bidirectional* (full, non-causal) attention over its patch tokens rather than a causal mask? What would break if you imposed a causal mask on the 196 patch tokens?

??? note "Solution"
    Causal masking exists in autoregressive language models because they *generate* text left-to-right: at training time, token $t$ must not see tokens $> t$, or the model could cheat by copying the answer it is being asked to predict. ViT is not a generative model — it is an **encoder** that produces a representation of a *complete, already-observed* image for a downstream task (classification, retrieval, dense features). There is no notion of "future" patches to hide: all 196 patches are available simultaneously, and each patch should be free to attend to every other patch.

    Recall from "Why Convolutions Are Not Enough" that a core motivation for ViT was **global context** — letting the top-left of the image directly attend to the bottom-right in a single layer. A causal mask, imposed over the arbitrary raster (row-major) ordering used to flatten patches into a sequence, would forbid exactly that: a patch could only attend to patches earlier in the raster order. This would (a) destroy the permutation-agnostic global mixing that makes ViT work, and (b) inject a meaningless directional bias — the raster ordering is an implementation artifact, not a semantic sequence, so "patch 5 may not see patch 100" corresponds to no real spatial constraint. The CLS token, prepended at position 0, would under a causal mask be able to attend to *nothing* except itself, making it useless as a global summary. Bidirectional attention is therefore not just allowed but required.

**2.** (Quantitative) You want to run the ViT from the chapter on high-resolution $512 \times 512$ images using patch size $P = 16$ (keeping everything else at the ViT-B/16 defaults: $D = 768$).

- (a) How many patch tokens $N$ result? What is the full sequence length including the CLS token?
- (b) Attention cost per layer scales as $O(N^2 D)$ (using the sequence length including CLS). By what factor does the per-layer attention compute grow relative to the standard $224 \times 224$ input?
- (c) The learned `pos_embed` table was trained for the $224 \times 224$ model. What is its shape, what shape do you need at $512 \times 512$, and what does the chapter say you must do to bridge the gap?

??? note "Solution"
    **(a)** With $P = 16$: along each side there are $512 / 16 = 32$ patches, so
    $$N = 32 \times 32 = 1024 \text{ patch tokens.}$$
    Including the prepended CLS token, the sequence length is $N + 1 = \mathbf{1025}$ tokens.

    **(b)** At $224 \times 224$ the sequence length is $196 + 1 = 197$. Attention compute $\propto (\text{seq len})^2$ (the $D$ factor is unchanged), so the growth factor is
    $$\frac{1025^2}{197^2} = \frac{1{,}050{,}625}{38{,}809} \approx 27.1\times.$$
    So the per-layer attention cost grows roughly **27-fold**. (If you approximate using patch counts only, $1024^2 / 196^2 = 1{,}048{,}576 / 38{,}416 \approx 27.3\times$ — essentially the same.) This quadratic blow-up is exactly why the chapter notes that FlashAttention becomes critical at high resolution.

    **(c)** The `pos_embed` table has shape $(1, N+1, D) = (1, 197, 768)$ — one learned vector per position for 196 patches plus CLS. At $512 \times 512$ you need $(1, 1025, 768)$. The chapter's fix is **bicubic interpolation** of the positional-embedding grid: separate the CLS position, reshape the 196 patch embeddings into a $14 \times 14 \times D$ grid, bicubically resize it to $32 \times 32 \times D$, flatten back to 1024 vectors, and re-attach the CLS embedding — exactly the `interpolate_pos_embed` function in the chapter.

**3.** (Quantitative) Verify the CLS-plus-embeddings parameter budget for a **ViT-L/14** image encoder, the standard CLIP backbone. Use $D = 1024$, patch size $P = 14$, $C = 3$, image size $224 \times 224$. Count only the *input-side* parameters: the patch-embedding matrix (with bias), the CLS token, and the positional-embedding table. Then state how many Transformer layers $L$ you would need for the blocks alone to reach roughly 300M parameters, using the chapter's per-block estimate of $12 D^2$.

??? note "Solution"
    First, the grid: $224 / 14 = 16$ patches per side, so $N = 16 \times 16 = 256$ patch tokens.

    **Patch-embedding matrix** $\mathbf{E} \in \mathbb{R}^{D \times (P^2 C)}$ plus bias:
    $$P^2 C = 14 \times 14 \times 3 = 588, \qquad 1024 \times 588 + 1024 = 602{,}112 + 1{,}024 = 603{,}136.$$

    **CLS token**: $D = 1024$ parameters.

    **Positional-embedding table** $(N+1) \times D = 257 \times 1024 = 263{,}168$ parameters.

    Input-side total:
    $$603{,}136 + 1{,}024 + 263{,}168 = \mathbf{867{,}328} \approx 0.87\text{M parameters.}$$

    **Blocks needed.** Per the chapter, one Transformer block costs about $12 D^2$:
    $$12 \times 1024^2 = 12 \times 1{,}048{,}576 = 12{,}582{,}912 \approx 12.58\text{M per block.}$$
    To reach ~300M from the blocks alone:
    $$L \approx \frac{300\text{M}}{12.58\text{M}} \approx 23.8 \Rightarrow L = 24 \text{ layers.}$$
    This matches the real ViT-L configuration ($D = 1024$, depth $= 24$, ~307M parameters), and shows the input-side embeddings (~0.87M) are a tiny fraction (<0.3%) of the total — the parameters live overwhelmingly in the 24 Transformer blocks.

**4.** (Quantitative) Consider a CLIP training batch of $N = 4$ image-text pairs. The image and text embeddings are already L2-normalized, and it happens that the similarity matrix of raw cosine similarities $\mathbf{i}_i \cdot \mathbf{t}_j$ is

$$
\begin{bmatrix}
0.9 & 0.1 & 0.2 & 0.0 \\
0.0 & 0.8 & 0.1 & 0.1 \\
0.2 & 0.2 & 0.7 & 0.3 \\
0.1 & 0.0 & 0.2 & 0.6
\end{bmatrix}
$$

with the temperature $\tau = 10$ (so logits are $10\times$ the entries above). Compute the **image-to-text** InfoNCE loss term $-\frac{1}{N}\sum_k \log \frac{e^{S_{kk}}}{\sum_j e^{S_{kj}}}$ (the first of the two symmetric terms). Work row 1 in full and give the final averaged value.

??? note "Solution"
    After scaling by $\tau = 10$, row $k$ of the logit matrix is $10 \times$ the given entries; we need the softmax probability of the *diagonal* entry in each row, then $-\log$ of it, averaged.

    **Row 1** logits: $[9, 1, 2, 0]$. Exponentials:
    $$e^{9} = 8103.08,\quad e^{1} = 2.718,\quad e^{2} = 7.389,\quad e^{0} = 1.$$
    Denominator $= 8103.08 + 2.718 + 7.389 + 1 = 8114.19$. Diagonal probability $= 8103.08 / 8114.19 = 0.99863$. Loss$_1 = -\log(0.99863) = 0.00137$.

    **Row 2** logits: $[0, 8, 1, 1]$. $e^{0}=1,\ e^{8}=2980.96,\ e^{1}=2.718,\ e^{1}=2.718$. Denominator $= 2987.40$. Diagonal prob $= 2980.96/2987.40 = 0.99785$. Loss$_2 = -\log(0.99785) = 0.00216$.

    **Row 3** logits: $[2, 2, 7, 3]$. $e^{2}=7.389,\ e^{2}=7.389,\ e^{7}=1096.63,\ e^{3}=20.086$. Denominator $= 1131.49$. Diagonal prob $= 1096.63/1131.49 = 0.96919$. Loss$_3 = -\log(0.96919) = 0.03129$.

    **Row 4** logits: $[1, 0, 2, 6]$. $e^{1}=2.718,\ e^{0}=1,\ e^{2}=7.389,\ e^{6}=403.43$. Denominator $= 414.54$. Diagonal prob $= 403.43/414.54 = 0.97319$. Loss$_4 = -\log(0.97319) = 0.02717$.

    Average:
    $$\mathcal{L}_{\text{img}\to\text{txt}} = \tfrac{1}{4}(0.00137 + 0.00216 + 0.03129 + 0.02717) = \tfrac{1}{4}(0.06199) \approx \mathbf{0.0155}.$$
    The loss is small because the diagonal (correct) pair dominates each row after the $\tau = 10$ scaling sharpens the softmax — this is what a well-trained CLIP model looks like on an easy batch. The full symmetric CLIP loss would average this with the analogous text-to-image term computed down the columns.

**5.** (Implementation) The chapter's `VisionTransformer` classifies using only the CLS token (`cls_out = x[:, 0]`). The Key Takeaways note an alternative: **patch-average (mean) pooling** over the patch tokens, which several encoders (e.g., SigLIP) prefer. Write a subclass `MeanPoolViT` that reuses the parent's `patch_embed`, `pos_embed`, `blocks`, and `norm`, but (a) does **not** prepend a CLS token, and (b) produces the classification logits by mean-pooling the final patch tokens. Keep the chapter's Pre-LN style and shapes. Note one consequence for the `pos_embed` shape.

??? note "Solution"
    Without a CLS token the sequence length is exactly $N$ (not $N+1$), so `pos_embed` must have shape $(1, N, D)$ instead of $(1, N+1, D)$. We rebuild it in the subclass rather than reuse the parent's, and we skip the `cls_token` entirely. Everything else — patch embedding, blocks, final norm — is reused unchanged.

    ```python
    import torch
    import torch.nn as nn

    class MeanPoolViT(VisionTransformer):
        """
        ViT variant with NO CLS token: classification uses mean-pooled patch tokens.
        Reuses PatchEmbed, the Transformer blocks, and the final LayerNorm from the
        parent VisionTransformer.
        """
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # The parent built pos_embed of shape (1, N+1, D) for [CLS | patches].
            # We have no CLS token, so we need exactly N positions.
            num_patches = self.patch_embed.num_patches
            embed_dim = self.pos_embed.shape[-1]
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            # cls_token inherited from the parent is simply left unused.

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, 3, H, W) pre-normalized images -> (B, num_classes) logits
            x = self.patch_embed(x)          # (B, N, D)  -- no CLS prepended
            x = x + self.pos_embed           # (B, N, D)  add positional embeddings
            x = self.pos_drop(x)

            for block in self.blocks:        # Pre-LN Transformer blocks (reused)
                x = block(x)                 # (B, N, D)

            x = self.norm(x)                 # (B, N, D)

            # Mean-pool over the patch (sequence) dimension for the global feature
            pooled = x.mean(dim=1)           # (B, D)
            return self.head(pooled)         # (B, num_classes)


    if __name__ == "__main__":
        model = MeanPoolViT()  # ViT-B/16 defaults, but CLS-free
        imgs = torch.randn(2, 3, 224, 224)
        logits = model(imgs)
        print(logits.shape)              # torch.Size([2, 1000])
        print(model.pos_embed.shape)     # torch.Size([1, 196, 768])  -- N, not N+1
    ```

    The output logits still have shape `(B, 1000)`, but the global representation now comes from averaging all 196 patch tokens rather than reading a single dedicated CLS position. A practical benefit: mean pooling is permutation-symmetric and forces every patch to contribute to the global feature, which the chapter's register-token discussion links to cleaner spatial behavior; the tradeoff is losing the dedicated learnable summary slot that CLS provides.
