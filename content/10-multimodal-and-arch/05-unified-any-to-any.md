# 10.5 Unified & Any-to-Any Models

The transformer was born as a text-only machine. Within a few years it absorbed images, audio, video, and code — not by bolting on task-specific heads, but by learning to treat every modality as a sequence of tokens and predicting them all under a single autoregressive objective. This chapter traces how that transformation happened, dissects the architectural choices that make it work, and looks honestly at where the frontier still lives.

The canonical multi-modal stack of 2022–2023 was a connector story: a frozen vision encoder feeds a bridge module that projects visual features into the language model's token space, and the language model generates text. That approach, covered in [Vision-Language Models](../10-multimodal-and-arch/02-vision-language-models.html), is powerful but fundamentally asymmetric — the model understands images but cannot generate them. Unified models collapse the asymmetry. One model, one loss, any modality in or out.

## Why Unification Matters

Three forces push toward full unification.

**Representational leverage.** Language pretraining gives the model commonsense and reasoning. Vision pretraining gives fine-grained perceptual features. When the two share weights and are jointly trained, each modality can borrow representations from the other. A model that generates captions is learning a visual-to-language grounding that benefits visual question answering, and vice versa.

**Inference simplicity.** Deploying ten specialized models — an image classifier, a caption generator, an image generator, a speech recognizer, a TTS system — means ten inference graphs, ten serving pods, ten latency SLAs. A single unified model with multiplexed inputs is operationally simpler and can serve more tasks per GPU.

**Emergent cross-modal reasoning.** Models trained jointly on interleaved image-text data develop capabilities neither modality alone would yield: counting objects in a scene and reasoning about their spatial relationships; generating an image that matches a textual description while simultaneously writing alt-text for it; editing an image in response to a spoken instruction. These capabilities seem to emerge from the shared representational substrate and are hard to engineer explicitly.

The cost is real: unified models are harder to train, suffer modality-specific collapse risks, and need careful data mixing. We will address all of these.

## The Tokenize-Everything Paradigm

The key insight is that autoregressive language modeling already knows how to predict from a discrete vocabulary. If every modality can be expressed as a sequence of integers from some codebook, the LM objective generalises immediately.

### Discrete Image Tokens

**VQ-VAE and VQ-GAN.** A vector-quantised variational autoencoder (VQ-VAE, van den Oord et al., 2017) learns an encoder $E$, a codebook $\mathbf{C} \in \mathbb{R}^{K \times d}$, and a decoder $D$. Given an image $x \in \mathbb{R}^{H \times W \times 3}$, the encoder outputs a spatial feature map $z_e \in \mathbb{R}^{h \times w \times d}$ (with $h = H/f$, $w = W/f$ for downsampling factor $f$). Each spatial location is replaced by the nearest codebook vector:

$$
z_q[i,j] = \arg\min_{k} \| z_e[i,j] - \mathbf{C}_k \|_2
$$

The decoder reconstructs $\hat{x} = D(z_q)$. Training minimises reconstruction loss plus a commitment loss:

$$
\mathcal{L}_\text{VQVAE} = \| x - \hat{x} \|^2 + \| \text{sg}(z_e) - z_q \|^2 + \beta \| z_e - \text{sg}(z_q) \|^2
$$

where $\text{sg}(\cdot)$ is the stop-gradient operator. After training, each image becomes a 1-D token sequence of length $h \times w$, with each token in $\{0, \ldots, K-1\}$. A 256×256 image with $f=8$ yields a $32 \times 32 = 1024$-token sequence — the same length as a medium-length paragraph of text.

{{fig:vqvae-image-to-tokens-pipeline}}

VQ-GAN (Esser et al., 2021) improves reconstruction fidelity by adding an adversarial loss, producing crisper tokens better suited to autoregressive generation. It became the tokenizer of choice for early image-generation transformers.

**FSQ and lookup-free quantisation.** More recent work like Finite Scalar Quantisation (FSQ, Mentzer et al., 2023) replaces the learned codebook with a simple per-channel rounding scheme, removing the training instability of codebook collapse and dead codes. Each dimension of the latent is independently rounded to a small set of values; the codebook is implicit. This matters for unified training because VQ-VAE codebooks can degrade when exposed to multi-domain data; FSQ is more robust.

### Discrete Audio Tokens

Audio is tokenised in two stages. **Acoustic tokens** capture low-level waveform detail (think EnCodec, Défossez et al., 2022, or SoundStream): a 1-D convolutional encoder outputs a compressed representation which is quantised with residual vector quantisation (RVQ) — a cascade of $n_q$ VQ stages each coding the residual of the previous. A second type, **semantic tokens** (HuBERT, Hsu et al., 2021), capture higher-level phoneme-like features extracted from a self-supervised speech model. Unified models that handle speech typically use semantic tokens for language alignment and acoustic tokens for high-fidelity synthesis.

### Byte-Level and Patch-Level Strategies

Not all modalities suit discrete tokenisation. An alternative is to work at the **patch level** — split the input into fixed-size blocks, project each to a $d$-dimensional vector, and treat the result as a sequence of continuous embeddings. This is exactly what [Vision Transformers & Image Encoders](../10-multimodal-and-arch/01-vision-transformers.html) do for understanding tasks. For generation, continuous embeddings require a diffusion head or flow-matching decoder rather than a softmax, which we cover in §10.5.5.

## Autoregressive Image Generation

Given discrete image tokens, image generation becomes next-token prediction — the same objective that trains GPT. This is the idea behind DALL-E (Ramesh et al., 2021) and ImageGPT (Chen et al., 2020).

{{fig:unified-ar-image-seq-layout}}

The model is trained with cross-entropy loss on all positions, but at inference time only the image portion is generated autoregressively. The text prefix provides conditioning through cross-attention in the causally masked attention — no architectural change is needed beyond the standard transformer.

### Scaling Challenges for Autoregressive Image Generation

A $256 \times 256$ image at $f=8$ produces 1024 tokens. A $512 \times 512$ image at the same downsampling yields 4096 tokens. Since attention is quadratic in sequence length, naive autoregressive generation at high resolution is expensive. Several mitigations exist:

1. **Hierarchical generation.** Generate low-resolution tokens first, then condition a second pass on those to fill in high-resolution details.
2. **Masked autoregressive (MAR) generation.** Generate in parallel over multiple masked positions per step (Chang et al., 2022, MaskGIT), trading sequential ordering for speed.
3. **More aggressive tokenisers.** Increase the downsampling ratio $f$ to reduce token count at the cost of reconstruction fidelity.

!!! example "Worked example: image token budget"

    Consider a 512×512 RGB image. With a VQ-GAN at $f=16$, the spatial grid is
    $\frac{512}{16} \times \frac{512}{16} = 32 \times 32 = 1024$ tokens.
    At $f=8$ that becomes $64 \times 64 = 4096$ tokens.

    With a batch size of 32 and mixed text+image sequences of 4096 image tokens + 256 text
    tokens = 4352 total tokens, one forward pass processes
    $32 \times 4352 = 139{,}264$ tokens. At BF16 with a 7B model (roughly
    $14 \times 10^{12}$ multiply-adds per token), that's about
    $139{,}264 \times 14 \times 10^{12} \approx 1.95 \times 10^{18}$ FLOPs per step.
    A single H100 at 989 TFLOPS of BF16 throughput handles this in about 2 seconds —
    meaning training at high image resolution demands serious compute even for one update.

## Chameleon: A Native Multi-Modal Transformer

Meta's Chameleon (2024) is a landmark unified model that processes and generates both text and images within a single transformer, with no separate vision encoder and no modality-specific heads beyond the token embedding and unembedding layers.

### Architecture

Chameleon's architecture is deliberately minimal:

- A single vocabulary of $V = 65{,}536$ tokens: standard BPE text tokens plus $8{,}192$ image codebook tokens from a custom VQ-VAE.
- A standard decoder-only transformer (similar to LLaMA, see [Building a GPT From Scratch (nanoGPT-style)](../02-transformer/07-build-gpt-from-scratch.html)) with no modifications to handle different modalities.
- Images are encoded by the VQ-VAE into sequences of 1024 tokens, then concatenated with text tokens in the natural document order.
- The causal mask is uniform — image tokens attend to all preceding tokens (text or image) exactly as text tokens do.

{{fig:unified-chameleon-interleaved-seq}}

### Training Stability Challenges

Chameleon's paper is remarkably candid about training instability. The joint vocabulary creates a softmax over 65K entries; image and text tokens have very different frequency distributions, making gradients noisy. Several techniques help:

**Query-key normalisation (QK-Norm).** Apply RMS normalisation to queries and keys before computing attention logits. This prevents attention logit explosion when the model encounters unusual token combinations at modality boundaries. Without QK-Norm, training diverges within the first few thousand steps on interleaved data.

**Dropout at modality boundaries.** A small amount of dropout on image token embeddings acts as regularisation, preventing the model from overrelying on memorised image codebook assignments.

**Modality-aware z-loss.** An auxiliary loss that penalises the logit magnitudes per modality separately, ensuring neither text nor image vocabulary dominates the softmax.

```python
import torch
import torch.nn.functional as F

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
```

### Data Mixture and Modality Balance

Chameleon is trained on text, image-only, and interleaved text-image documents. The paper reports that maintaining roughly equal token counts across modalities (with a modest over-representation of text) is critical; if image tokens are under-represented, the model forgets how to generate them even though it can still understand them. This is analogous to the catastrophic forgetting discussed in [Supervised Fine-Tuning & Instruction Tuning](../05-posttraining-alignment/01-sft-instruction-tuning.html).

## Transfusion: Mixing Discrete and Diffusion Objectives

Discrete tokenisation loses information. A VQ-GAN codebook of size 8192 can represent roughly $\log_2 8192 = 13$ bits per spatial location; a raw 8-bit pixel RGB patch carries $3 \times 8 = 24$ bits. The quantisation gap degrades image generation quality at small scales.

**Transfusion** (Zhou et al., 2024, Meta) sidesteps this by using a hybrid objective: autoregressive next-token prediction for text, and diffusion (specifically flow matching) for image patches — all within the same transformer.

### Architecture

{{fig:unified-transfusion-arch}}

**Text positions** use standard AR cross-entropy loss. **Image positions** receive continuous patch embeddings (linear projection of raw pixels, no quantisation) and are trained with a denoising diffusion / flow-matching objective. Specifically, Transfusion uses **flow matching** (Lipman et al., 2022): given a clean image patch $x_0$ and a noise sample $\epsilon \sim \mathcal{N}(0, I)$, define

$$
x_t = (1 - t) x_0 + t \epsilon, \quad t \in [0, 1]
$$

The model predicts the velocity field $v_\theta(x_t, t)$ and is trained to minimise

$$
\mathcal{L}_\text{flow} = \mathbb{E}_{t, x_0, \epsilon} \| v_\theta(x_t, t) - (\epsilon - x_0) \|^2
$$

The total training loss combines both objectives:

$$
\mathcal{L}_\text{Transfusion} = \mathcal{L}_\text{LM} + \lambda \mathcal{L}_\text{flow}
$$

where $\lambda$ is a scalar balancing the two (often around 1 after normalising by modality).

### Attention Masking in Transfusion

Text tokens are causal — each attends only to previous tokens. Image patch tokens within one image attend to each other with **bidirectional** attention (the diffusion objective does not require causal structure within an image), but each image block still causally follows all preceding tokens. This creates a block-causal mask:

{{fig:transfusion-block-causal-mask}}

```python
import torch

def transfusion_attention_mask(token_types: list[str], seq_len: int) -> torch.Tensor:
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
```

### Why Transfusion Outperforms Pure Discrete Tokenisation

For image generation benchmarks (FID scores, recall), Transfusion trades slightly lower text quality for significantly better image quality compared to Chameleon-style discrete image tokens, at the same model and compute scale. The intuition is that continuous representations preserve the full information content of the image; the diffusion head is trained to reconstruct it directly rather than through a bottleneck codebook.

## Mixed-Modal Pretraining: Data and Training Recipes

Getting unified training to converge requires careful attention to data mixing, curriculum design, and loss weighting. The challenges compound those already present in standard pretraining (see [Pretraining Data: Sources, Crawling & The Data Pipeline](../03-pretraining/01-pretraining-data.html)).

### Data Format: Interleaved Documents

Early multi-modal datasets (LAION-5B) were image-caption pairs. Interleaved models like Flamingo (Alayrac et al., 2022) and OpenFlamingo require richer document-level interleaving: web pages, Wikipedia articles, and scientific papers where images and text co-occur naturally. MMC4 (Zhu et al., 2023) and OBELICS (Laurençon et al., 2023) are large-scale interleaved corpora aligned with images and their surrounding text.

For unified generation models, the data must also include image-only documents (to teach the model to generate images unconditionally), text-conditioned image data (to learn the T→I task), and image-to-text data (to learn captions, OCR, VQA). The mix ratio matters:

```python
# Example data-mixing schedule (token counts, not document counts)
data_mix = {
    "text_only":            0.50,  # 50% of token budget
    "image_only":           0.10,  # 10% — unconditional image generation
    "text_then_image":      0.20,  # 20% — T→I generation
    "image_then_text":      0.15,  # 15% — image understanding (captioning, VQA)
    "interleaved_doc":      0.05,  # 5%  — full web-page style documents
}
# These proportions are illustrative; actual models tune them empirically.
```

### Curriculum: Staged Training

Starting joint training from scratch on all modalities simultaneously can cause early instability, because the image tokeniser is frozen but the transformer needs to discover the image-token distribution while simultaneously learning language. A staged curriculum works better:

1. **Stage 1: Text-only pretraining.** Initialise the language backbone on a large text corpus. This follows standard scaling-law-optimal data and compute (see [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html)).
2. **Stage 2: Multi-modal warmup.** Introduce image tokens at a low mixing ratio (5–10%), train with a lower learning rate, and freeze the image tokeniser.
3. **Stage 3: Full joint training.** Scale to the full multi-modal data mixture; unfreeze all components.

### Loss Weighting Across Modalities

Naively, each token contributes equally to the loss. But image tokens and text tokens have very different per-token entropies — text cross-entropy loss is typically 2–4 nats/token at a well-trained state, while an image token prediction problem over a codebook of 8192 has maximum entropy $\ln 8192 \approx 9$ nats. Raw token-level averaging over-weights image generation relative to text comprehension.

A simple fix: compute modality-specific loss normalisation,

$$
\mathcal{L}_\text{total} = \frac{\mathcal{L}_\text{text}}{N_\text{text}} + w_\text{img} \cdot \frac{\mathcal{L}_\text{image}}{N_\text{image}}
$$

where $w_\text{img}$ is tuned on a small validation grid to balance gradient magnitudes.

## Any-to-Any Generation: Extending to Audio and Video

The "any-to-any" aspiration means the same model handles text, images, audio, and video — any combination of inputs and outputs. This is architecturally straightforward if every modality is tokenised; the challenge is combinatorial data and training cost.

### AnyGPT

AnyGPT (Zhan et al., 2024) is an example of a fully discrete any-to-any model. It unifies text, image (VQ-GAN tokens), audio (SoundStorm-style semantic tokens), and music under a single autoregressive transformer. The vocabulary is the union of all per-modality codebooks. Special delimiter tokens mark modality boundaries:

```text
[TEXT_START] "Describe this sound:" [TEXT_END]
[AUDIO_START] 512 847 231 ... (semantic audio tokens) [AUDIO_END]
[TEXT_START] "A dog barking in a park." [TEXT_END]
```

The model is trained on all four modalities jointly. At inference it can be prompted with any combination: text → image, image + text → audio, audio → text, and so on.

### Video Tokenisation

Video is the most expensive modality. A 10-second clip at 30 FPS with resolution 256×256 contains 300 frames. At $f=8$ per frame, that is $300 \times 32 \times 32 = 307{,}200$ spatial tokens before any temporal compression. Temporal tokenisers (3D VQ-VAE, causal video codecs) add a temporal downsampling factor $f_t$, reducing the token budget by $f_t$ — at $f_t = 4$, we get 76,800 tokens, still very long.

Efficient video models (like those in the Emu family from Meta, or Sora-style architectures) address this with:
- **Spatial-temporal factored tokenisation**: separate spatial and temporal downsampling.
- **Hierarchical models**: a fast draft pass at low resolution, then a slow refinement pass.
- **Chunk-causal attention**: process video in short overlapping windows.

### Speech In and Out

For a model that both understands and synthesises speech, two tokeniser levels are combined: semantic tokens capture content (used for ASR-like conditioning) and acoustic tokens carry prosody and speaker identity (used for TTS-like synthesis). The model generates semantic tokens first (short sequence), then conditions an acoustic decoder on them to produce high-fidelity audio. See [Audio, Speech & Multimodal Fusion](../10-multimodal-and-arch/03-audio-speech-multimodal.html) for the speech modelling detail.

## The Role of Mixture-of-Experts in Unified Models

Unified models face a modality-interference challenge: image tokens and text tokens require very different computation patterns, yet they share all transformer weights. A natural solution is [Mixture-of-Experts (MoE) Architectures](../02-transformer/09-mixture-of-experts.html) with modality-aware routing.

In a MoE unified model, each token is routed to $k$ experts out of $E$ total in the FFN sub-layer. If the router learns that image tokens consistently go to a subset of experts and text tokens to a different subset, the model effectively allocates capacity separately to each modality while still allowing cross-modal interactions through the shared attention layer.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

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
```

!!! warning "Modality collapse in MoE unified models"

    A common failure mode: the router discovers early in training that routing all image
    tokens to a handful of experts minimises loss. Those experts become image-only
    specialists, while the rest handle text. As training continues, the image experts
    never see text and the text experts never see images, so cross-modal reasoning
    fails to develop. Mitigation: an entropy regularisation loss on the routing
    distribution, or a hard constraint that at least one of the top-$k$ selected
    experts must be a "general" expert shared across modalities.

## Open Problems and the Road Ahead

Unified any-to-any models are exciting precisely because so much remains open.

### Evaluation

There is no single benchmark that holistically evaluates a unified model. Text generation is measured by perplexity and instruction-following benchmarks; image generation by FID, IS, and CLIP-score; image understanding by VQA benchmarks; audio by WER and MOS. A model can excel on text and fail on images. Building an integrated eval harness (see [Building Eval Harnesses](../11-evaluation/03-eval-harnesses.html)) for cross-modal coherence remains an open research challenge.

### Context Length and Modality-Mixing Ratio

As models incorporate more modalities, the effective context length needed to represent a rich multi-modal interaction grows dramatically. A short conversation with three images and audio clips might require tens of thousands of tokens. Techniques from [Long-Context Pretraining & Context Extension](../03-pretraining/13-long-context-pretraining.html) — RoPE with extended base frequencies, YaRN, and sequence parallelism — are being adapted for multi-modal contexts.

### Autoregressive vs Diffusion Tradeoffs

Autoregressive generation is sequentially slow for images — generating 1024 tokens at decode latency of 20 ms/token costs about 20 seconds for one image. Diffusion and flow-matching run $N$ denoising steps but each step processes the entire image in parallel, typically completing in under 1 second on a single GPU. Transfusion's hybrid is a bet that the quality advantage of diffusion justifies the architectural complexity. The community has not converged on a winner; masked autoregressive models (MaskGIT, MAR) occupy an interesting middle ground.

### Tokeniser Quality Bottleneck

The quality ceiling for discrete-token models is set by the tokeniser's reconstruction fidelity. A VQ-GAN that cannot perfectly reconstruct fine texture will produce blurry images no matter how good the transformer is. Improving tokenisers (higher codebook utilisation, residual quantisation layers, adversarial fine-tuning) is a first-order lever. The community's shift toward continuous representations (as in Transfusion) is partly motivated by this ceiling.

### Cross-Modal Alignment and Grounding

"Any-to-any" generation requires more than modality conversion — it requires understanding the semantic correspondence between modalities. A model that generates an image of "a red ball on a blue table" must know what red, ball, blue, and table look like. Getting this right requires large-scale image-text data with tight semantic alignment, not just document-level co-occurrence. Contrastively trained encoders (CLIP, SigLIP) provide a useful auxiliary training signal even inside unified architectures.

!!! interview "Interview Corner"

    **Q:** Chameleon and Transfusion are both unified text-image models, but they tokenise
    images very differently. What are the tradeoffs, and when would you choose one over the other?

    **A:** Chameleon uses a discrete VQ-VAE codebook: each image patch is mapped to one of
    $K$ integers, turning image generation into next-token prediction with a single cross-entropy
    loss over a joint vocabulary. This is architecturally simple — no changes to the
    transformer or loss function — and allows arbitrary interleaving of text and image
    tokens. The downside is information loss from quantisation: fine textures and subtle
    colour gradients are discarded, capping image fidelity at the codebook's resolution.

    Transfusion keeps image patches as continuous vectors and trains a flow-matching
    (diffusion) head on them while using the standard LM loss for text. The transformer
    processes continuous patch embeddings alongside discrete text tokens, using a
    bidirectional attention mask within image blocks. This preserves full image information
    and yields much better generation quality at equal compute. The cost is architectural
    complexity: you need a diffusion head, a flow-matching training loop, and a
    mixed-objective loss with a tuned $\lambda$.

    Choose Chameleon-style tokenisation when: you want maximum training simplicity, you
    are primarily an understanding-first model, or your downstream application tolerates
    moderate image quality (web thumbnails, UI mockups). Choose Transfusion-style when:
    image generation quality is a first-order product requirement, you are willing to
    manage the diffusion inference loop, and you want the model to eventually serve as
    a foundation for high-resolution generation.

## A Minimal Any-to-Any Training Loop

The following code sketch shows how to assemble a training step for a Transfusion-style model with text and image modalities, using PyTorch Lightning conventions.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

@dataclass
class Batch:
    """A mixed-modal training batch."""
    input_ids:        torch.Tensor   # (B, T_text)  — text token ids
    text_labels:      torch.Tensor   # (B, T_text)  — same, with -100 for masked positions
    image_patches:    torch.Tensor   # (B, T_img, D_patch)  — continuous patch embeddings
    image_labels:     torch.Tensor   # (B, T_img, D_patch)  — clean patch targets
    noise:            torch.Tensor   # (B, T_img, D_patch)  — sampled ε
    t:                torch.Tensor   # (B,)          — diffusion timestep in [0, 1]
    attention_mask:   torch.Tensor   # (B, T_total, T_total) — block-causal mask
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

        # Target velocity: (ε - x_0)
        target_velocity = (batch.noise - batch.image_patches).reshape(-1, batch.noise.shape[-1])
        target_velocity = target_velocity[batch.is_image.reshape(-1)]
        flow_loss        = F.mse_loss(pred_velocity, target_velocity)

        # 6. Combine losses
        lam   = 1.0   # balance factor; tune on a small sweep
        total = lm_loss + lam * flow_loss

        return {
            "loss":       total,
            "lm_loss":    lm_loss.detach(),
            "flow_loss":  flow_loss.detach(),
        }
```

Note: this sketch omits the attention mask construction, the transformer implementation, and the data loader. A production implementation would use FlashAttention (see [FlashAttention I: IO-Awareness & The Online Softmax](../04-kernels-efficiency/02-flash-attention-1.html)) and handle variable-length sequences with sequence packing (see [Chat Templates, Data Formatting & Sequence Packing](../05-posttraining-alignment/02-chat-templates-packing.html)).

!!! key "Key Takeaways"

    - Unified models collapse understanding and generation across modalities into a single
      autoregressive (or hybrid) transformer, eliminating modality-specific architectures.
    - Discrete tokenisation (VQ-VAE/VQ-GAN) is architecturally the simplest path: images
      become integer sequences and the standard LM cross-entropy loss applies unchanged.
      The cost is a quantisation ceiling on image quality.
    - Transfusion's hybrid approach — AR cross-entropy for text, flow-matching diffusion for
      continuous image patches — achieves better generation fidelity at the expense of
      training and inference complexity.
    - Chameleon-style training requires QK-Norm and modality-aware z-loss to stabilise the
      joint vocabulary softmax over 65K tokens from two very different distributions.
    - Data mixing ratios and training curriculum (text-first warmup → multi-modal ramp-up)
      are critical hyperparameters; poor mixing causes modality-specific forgetting.
    - MoE architectures naturally extend to unified models by letting the router learn
      modality-specialist experts, but require entropy regularisation to avoid full
      modality collapse in routing.
    - Any-to-any generation (text, image, audio, video) is architecturally straightforward
      given per-modality tokenisers but faces combinatorial data, evaluation, and context-
      length challenges that remain active research problems.
    - Evaluation for unified models is inherently multi-dimensional; no single benchmark
      captures cross-modal coherence and modality-specific quality simultaneously.

!!! sota "State of the Art & Resources (2026)"
    Unified any-to-any models have moved from proof-of-concept to production-scale systems: models like Chameleon (discrete token fusion), Transfusion (hybrid AR + diffusion), Show-o, and Janus-Pro all emerged between 2024–2025, demonstrating that a single transformer can match or exceed specialist models on both understanding and generation benchmarks across text, images, and video.

    **Foundational work**

    - [Esser et al., *Taming Transformers for High-Resolution Image Synthesis* (VQ-GAN, 2021)](https://arxiv.org/abs/2012.09841) — introduced the VQ-GAN discrete image tokenizer that underlies most autoregressive image-generation models.
    - [Lipman et al., *Flow Matching for Generative Modeling* (ICLR 2023)](https://arxiv.org/abs/2210.02747) — the flow-matching objective used by Transfusion and many subsequent continuous-modality generation heads.

    **Recent advances (2023–2026)**

    - [Chameleon Team, *Chameleon: Mixed-Modal Early-Fusion Foundation Models* (2024)](https://arxiv.org/abs/2405.09818) — landmark fully-discrete unified model; reveals QK-Norm and z-loss as critical stabilisation tricks for joint text+image vocabularies.
    - [Zhou et al., *Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model* (2024)](https://arxiv.org/abs/2408.11039) — hybrid AR (text) + flow-matching (image) training in one transformer; scales to 7B with quality rivalling specialist diffusion models.
    - [Zhan et al., *AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling* (ACL 2024)](https://arxiv.org/abs/2402.12226) — extends the discrete-token paradigm to text, images, audio, and music under one autoregressive model.
    - [Xie et al., *Show-o: One Single Transformer to Unify Multimodal Understanding and Generation* (ICLR 2025)](https://arxiv.org/abs/2408.12528) — mixes causal AR for text with discrete diffusion for images; Show-o2 extends support to video (2025).
    - [Chen et al., *Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling* (2025)](https://arxiv.org/abs/2501.17811) — decouples visual encoding pathways for understanding vs. generation, closing the quality gap with specialist models at 7B scale.
    - [Zhang et al., *Unified Multimodal Understanding and Generation Models: Advances, Challenges, and Opportunities* (2025)](https://arxiv.org/abs/2505.02567) — comprehensive survey categorising the field into diffusion-based, autoregressive-based, and hybrid paradigms.

    **Open-source & tools**

    - [facebookresearch/chameleon](https://github.com/facebookresearch/chameleon) — official inference code and evaluation prompts for Meta's Chameleon model.
    - [showlab/show-o](https://github.com/showlab/show-o) — training and inference code for Show-o and Show-o2, with pretrained checkpoints on Hugging Face.
    - [deepseek-ai/Janus](https://github.com/deepseek-ai/Janus) — Janus, JanusFlow, and Janus-Pro implementations with MIT-licensed code and model weights.

## Further Reading

- Ramesh et al., "Zero-Shot Text-to-Image Generation" (DALL-E), ICML 2021.
- Esser et al., "Taming Transformers for High-Resolution Image Synthesis" (VQ-GAN), CVPR 2021.
- van den Oord et al., "Neural Discrete Representation Learning" (VQ-VAE), NeurIPS 2017.
- Alayrac et al., "Flamingo: a Visual Language Model for Few-Shot Learning", NeurIPS 2022.
- Chameleon Team, "Chameleon: Mixed-Modal Early-Fusion Foundation Models", Meta, 2024.
- Zhou et al., "Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model", Meta, 2024.
- Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023.
- Chang et al., "MaskGIT: Masked Generative Image Transformer", CVPR 2022.
- Zhan et al., "AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling", 2024.
- Mentzer et al., "Finite Scalar Quantization: VQ-VAE Made Simple", ICLR 2024.
- Défossez et al., "High Fidelity Neural Audio Compression" (EnCodec), 2022.
