"""
Runnable-code test for content/02-transformer/08-architecture-variants.md

Blocks tested (assembled in chapter order, later blocks may reuse names from
earlier ones exactly as in the chapter):
  - block #0 (line ~59):  BERT-style MLM masking + loss (apply_mlm_mask, mlm_loss)
  - block #2 (line ~159): Encoder-decoder cross-attention module (CrossAttention)
  - block #3 (line ~255): From-scratch decoder-only transformer
                           (CausalSelfAttention, TransformerBlock, DecoderOnlyTransformer)
  - block #6 (line ~554): Mask-construction reference functions (causal_mask,
                           bidirectional_mask, prefix_lm_mask, encoder_decoder_mask,
                           apply_causal_mask_sdpa)

Skipped:
  - block #1 (line ~152): non-python (text diagram of T5 span-corruption I/O)
  - block #4 (line ~424): fragment — make_prefix_lm_mask() is a standalone helper
                           whose only use is the worked numerical example right
                           after it; not part of the assembled module's flow of
                           tested blocks per the task spec (default SKIP)
  - block #5 (line ~454): non-python (plain-text mask visualization output)

No network / external-API calls are used anywhere in this chapter's code, so no
mocking is required.

BUG FIXED (mirrored in content/02-transformer/08-architecture-variants.md):
  In block #0's apply_mlm_mask(), the original book code called
  `torch.randint(0, vocab_size, replace_with_random.sum().item())` — passing
  a bare int as the `size` argument. torch.randint requires `size` to be a
  tuple/list of ints, so this raised TypeError whenever any positions were
  selected for random-token replacement. Fixed to
  `torch.randint(0, vocab_size, (replace_with_random.sum().item(),))`.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# =====================================================================
# Block #0 (line ~59) — BERT-style MLM masking objective
# =====================================================================

def apply_mlm_mask(input_ids: torch.Tensor,
                   vocab_size: int,
                   mask_token_id: int,
                   mask_prob: float = 0.15) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply BERT-style MLM masking to a batch of token ids.

    Args:
        input_ids: shape (B, T)
        vocab_size: size of vocabulary
        mask_token_id: id of the [MASK] token
        mask_prob: fraction of tokens to select for masking

    Returns:
        masked_input: (B, T) — input with some tokens replaced
        labels:       (B, T) — original ids at masked positions, -100 elsewhere
                               (-100 is ignored by F.cross_entropy)
    """
    B, T = input_ids.shape
    # Draw a Bernoulli mask: which positions are selected (15%)
    selected = torch.rand(B, T) < mask_prob          # (B, T) bool

    # Of the selected positions:
    #   80% → [MASK]
    #   10% → random token
    #   10% → unchanged (but still included in loss)
    rand_roll = torch.rand(B, T)
    replace_with_mask   = selected & (rand_roll < 0.80)
    replace_with_random = selected & (rand_roll >= 0.80) & (rand_roll < 0.90)
    # The rest (0.90–1.0) remain as original — no action needed

    masked_input = input_ids.clone()
    masked_input[replace_with_mask]   = mask_token_id
    masked_input[replace_with_random] = torch.randint(
        0, vocab_size, (replace_with_random.sum().item(),)
    )

    # Labels: original token at selected positions, -100 elsewhere
    labels = torch.full_like(input_ids, fill_value=-100)
    labels[selected] = input_ids[selected]

    return masked_input, labels


def mlm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    logits: (B, T, V) — raw logits over vocab
    labels: (B, T)    — original token ids at masked positions, -100 elsewhere
    """
    B, T, V = logits.shape
    # F.cross_entropy ignores positions where label == -100
    return F.cross_entropy(logits.view(B * T, V), labels.view(B * T))


# =====================================================================
# Block #2 (line ~159) — Encoder-decoder cross-attention
# =====================================================================

class CrossAttention(nn.Module):
    """
    Encoder-decoder cross-attention: queries come from decoder,
    keys/values come from encoder hidden states.
    Fully standard attention — no causal mask on the encoder side.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)  # from decoder
        self.W_k = nn.Linear(d_model, d_model, bias=False)  # from encoder
        self.W_v = nn.Linear(d_model, d_model, bias=False)  # from encoder
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self,
                decoder_hidden: torch.Tensor,   # (B, T_dec, D)
                encoder_hidden: torch.Tensor,   # (B, T_enc, D)
                encoder_mask: torch.Tensor | None = None  # (B, T_enc) bool
                ) -> torch.Tensor:
        B, T_dec, D = decoder_hidden.shape
        T_enc = encoder_hidden.shape[1]
        H = self.n_heads

        # Project, then reshape to (B, H, T, d_k)
        def split_heads(x: torch.Tensor, T: int) -> torch.Tensor:
            return x.view(B, T, H, self.d_k).transpose(1, 2)

        Q = split_heads(self.W_q(decoder_hidden), T_dec)   # (B, H, T_dec, d_k)
        K = split_heads(self.W_k(encoder_hidden), T_enc)   # (B, H, T_enc, d_k)
        V = split_heads(self.W_v(encoder_hidden), T_enc)   # (B, H, T_enc, d_k)

        # Scaled dot-product attention
        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)  # (B, H, T_dec, T_enc)

        if encoder_mask is not None:
            # encoder_mask: (B, T_enc) — True where token is PAD
            pad_mask = encoder_mask[:, None, None, :]    # broadcast over H, T_dec
            scores = scores.masked_fill(pad_mask, float('-inf'))

        attn = torch.softmax(scores, dim=-1)             # (B, H, T_dec, T_enc)
        out  = attn @ V                                   # (B, H, T_dec, d_k)

        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, T_dec, D)
        return self.W_o(out)


# =====================================================================
# Block #3 (line ~255) — From-scratch decoder-only transformer
# =====================================================================

class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention."""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 2048):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k    = d_model // n_heads
        self.n_heads = n_heads
        self.d_model = d_model

        # Fused QKV projection for efficiency
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        # Register causal mask as a buffer (not a parameter)
        # Lower-triangular: M[i,j]=1 iff j <= i
        causal_mask = torch.ones(max_seq_len, max_seq_len, dtype=torch.bool).tril()
        self.register_buffer("causal_mask", causal_mask)  # (T_max, T_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        returns: (B, T, D)
        """
        B, T, D = x.shape
        H, d_k = self.n_heads, self.d_k

        # Compute Q, K, V via fused projection, then split
        qkv = self.qkv(x)                       # (B, T, 3D)
        Q, K, V = qkv.split(D, dim=-1)          # each (B, T, D)

        # Reshape to (B, H, T, d_k) for multi-head attention
        def reshape(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, H, d_k).transpose(1, 2)

        Q, K, V = map(reshape, (Q, K, V))

        # Scaled dot-product attention with causal mask
        scores = Q @ K.transpose(-2, -1) * (d_k ** -0.5)   # (B, H, T, T)

        # Apply causal mask: positions where mask==False get -inf
        mask = self.causal_mask[:T, :T]          # (T, T)
        scores = scores.masked_fill(~mask, float('-inf'))

        attn  = F.softmax(scores, dim=-1)        # (B, H, T, T)
        out   = attn @ V                          # (B, H, T, d_k)

        # Merge heads: (B, H, T, d_k) → (B, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block (decoder-only, no cross-attention)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int = 2048):
        super().__init__()
        self.norm1  = nn.LayerNorm(d_model)
        self.attn   = CausalSelfAttention(d_model, n_heads, max_seq_len)
        self.norm2  = nn.LayerNorm(d_model)
        # Feed-forward: expand to 4x, then contract
        self.ff     = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm: normalize before the sub-layer, add residual after
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class DecoderOnlyTransformer(nn.Module):
    """
    Minimal GPT-style decoder-only language model.
    Uses learned absolute positional embeddings (GPT-2 style).
    """

    def __init__(self,
                 vocab_size: int,
                 d_model: int   = 256,
                 n_heads: int   = 4,
                 n_layers: int  = 6,
                 max_seq_len: int = 512):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)  # learned pos emb

        d_ff = 4 * d_model
        self.blocks  = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, max_seq_len)
            for _ in range(n_layers)
        ])
        self.norm_f  = nn.LayerNorm(d_model)             # final norm
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: embedding and LM head share weights (saves params + improves quality)
        self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        """GPT-2 style initialization."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx: (B, T) — token ids
        returns logits: (B, T, vocab_size)
        """
        B, T = idx.shape
        positions = torch.arange(T, device=idx.device).unsqueeze(0)  # (1, T)

        x = self.tok_emb(idx) + self.pos_emb(positions)    # (B, T, D)

        for block in self.blocks:
            x = block(x)

        x = self.norm_f(x)
        return self.lm_head(x)                              # (B, T, vocab_size)

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new_tokens: int = 64,
                 temperature: float = 1.0) -> torch.Tensor:
        """Greedy/temperature sampling. Prompt: (1, T_prompt)."""
        for _ in range(max_new_tokens):
            logits = self.forward(prompt)[:, -1, :]         # (1, vocab_size)
            logits = logits / temperature
            next_tok = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            prompt = torch.cat([prompt, next_tok], dim=1)
        return prompt


# --- Quick sanity check ---
if __name__ == "__main__":
    model = DecoderOnlyTransformer(vocab_size=1000, d_model=128, n_heads=4, n_layers=4)
    x = torch.randint(0, 1000, (2, 32))   # batch=2, seq_len=32
    logits = model(x)
    print(f"Output shape: {logits.shape}")   # should be (2, 32, 1000)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")


# =====================================================================
# Block #6 (line ~554) — Masking-pattern reference functions
# =====================================================================

def causal_mask(T: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Standard lower-triangular causal mask. Returns (T, T) bool mask."""
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))


def bidirectional_mask(T: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """All-ones mask — every position attends to every position."""
    return torch.ones(T, T, dtype=torch.bool, device=device)


def prefix_lm_mask(prefix_len: int, total_len: int,
                   device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Prefix-LM mask: bidirectional within prefix, causal for generation."""
    mask = causal_mask(total_len, device)
    mask[:prefix_len, :prefix_len] = True   # full bidirectional for prefix block
    return mask


def encoder_decoder_mask(T_dec: int, T_enc: int,
                          pad_mask: torch.Tensor | None = None,
                          device: torch.device = torch.device("cpu")) -> dict:
    """
    Returns the two masks needed for an encoder-decoder model:
      - 'self':  causal mask for decoder self-attention  (T_dec, T_dec)
      - 'cross': encoder padding mask for cross-attention (T_enc,) bool: True=PAD
    """
    self_mask  = causal_mask(T_dec, device)
    cross_mask = pad_mask if pad_mask is not None else torch.zeros(T_enc, dtype=torch.bool, device=device)
    return {"self": self_mask, "cross": cross_mask}


# --- Common gotcha: using the wrong dtype ---
# torch.where and masked_fill expect a *bool* mask, not float.
# torch.nn.functional.scaled_dot_product_attention (PyTorch 2.0+) expects
# an *additive* attn_mask (float, 0 or -inf), NOT a bool mask.
# Always double-check which convention your attention function uses.

def apply_causal_mask_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
                           ) -> torch.Tensor:
    """
    PyTorch 2.0+ scaled_dot_product_attention with causal mask.
    The is_causal=True flag efficiently generates the causal mask internally,
    avoiding the O(T^2) mask tensor allocation.
    """
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,
        is_causal=True,    # ← most efficient way to do causal masking in PyTorch 2+
        dropout_p=0.0,
    )


# =====================================================================
# Glue: actually execute every block above with tiny CPU fixtures
# =====================================================================

def test_block0_mlm_mask_and_loss():
    B, T, V = 4, 16, 50
    mask_token_id = V  # reserve one extra id for [MASK]
    vocab_size = V + 1
    input_ids = torch.randint(0, V, (B, T))

    masked_input, labels = apply_mlm_mask(input_ids, vocab_size, mask_token_id, mask_prob=0.3)
    assert masked_input.shape == (B, T)
    assert labels.shape == (B, T)
    # At masked-out positions, mask token or random token or unchanged is fine;
    # at unselected positions, label must be -100.
    unselected = labels == -100
    assert torch.equal(masked_input[unselected], input_ids[unselected])

    # Run the loss on random logits — checks the cross_entropy plumbing works
    # even when a batch happens to have zero selected positions.
    logits = torch.randn(B, T, vocab_size, requires_grad=True)
    loss = mlm_loss(logits, labels)
    assert loss.dim() == 0
    loss.backward()
    print(f"[block0] mlm_loss = {loss.item():.4f}")


def test_block2_cross_attention():
    B, T_dec, T_enc, D, H = 2, 5, 7, 32, 4
    cross_attn = CrossAttention(d_model=D, n_heads=H)
    decoder_hidden = torch.randn(B, T_dec, D)
    encoder_hidden = torch.randn(B, T_enc, D)

    out = cross_attn(decoder_hidden, encoder_hidden)
    assert out.shape == (B, T_dec, D)

    # Also exercise the padding-mask branch.
    pad_mask = torch.zeros(B, T_enc, dtype=torch.bool)
    pad_mask[:, -2:] = True  # last two encoder positions are padding
    out_masked = cross_attn(decoder_hidden, encoder_hidden, encoder_mask=pad_mask)
    assert out_masked.shape == (B, T_dec, D)
    assert not torch.allclose(out, out_masked)
    print(f"[block2] CrossAttention output shape = {tuple(out.shape)}")


def test_block3_decoder_only_transformer():
    # The chapter's own __main__ sanity check already ran at import time
    # (model, x, logits, n_params are module-level names defined above).
    assert logits.shape == (2, 32, 1000)
    assert n_params > 0
    print(f"[block3] sanity-check output shape = {tuple(logits.shape)}, params = {n_params:,}")

    # Additionally exercise autoregressive generate() with a tiny prompt.
    small_model = DecoderOnlyTransformer(vocab_size=50, d_model=16, n_heads=2, n_layers=2, max_seq_len=32)
    prompt = torch.randint(0, 50, (1, 4))
    generated = small_model.generate(prompt, max_new_tokens=6, temperature=1.0)
    assert generated.shape == (1, 4 + 6)
    print(f"[block3] generate() output shape = {tuple(generated.shape)}")


def test_block6_mask_functions():
    T = 6
    cmask = causal_mask(T)
    assert cmask.shape == (T, T)
    assert cmask.dtype == torch.bool
    assert cmask[0, 1].item() is False and cmask[1, 0].item() is True

    bmask = bidirectional_mask(T)
    assert bool(bmask.all())

    pmask = prefix_lm_mask(prefix_len=3, total_len=T)
    # prefix block is fully bidirectional
    assert bool(pmask[:3, :3].all())
    # generation rows remain causal (no peeking at the future)
    assert pmask[3, 4].item() is False

    ed = encoder_decoder_mask(T_dec=4, T_enc=5)
    assert ed["self"].shape == (4, 4)
    assert ed["cross"].shape == (5,)
    assert not bool(ed["cross"].any())  # default: no padding

    # Exercise the SDPA convenience wrapper end-to-end.
    B, H, T_, d_k = 2, 4, 6, 8
    q = torch.randn(B, H, T_, d_k)
    k = torch.randn(B, H, T_, d_k)
    v = torch.randn(B, H, T_, d_k)
    out = apply_causal_mask_sdpa(q, k, v)
    assert out.shape == (B, H, T_, d_k)
    print(f"[block6] SDPA causal output shape = {tuple(out.shape)}")


if __name__ == "__main__":
    test_block0_mlm_mask_and_loss()
    test_block2_cross_attention()
    test_block3_decoder_only_transformer()
    test_block6_mask_functions()
    print("\nAll architecture-variants blocks executed successfully.")
