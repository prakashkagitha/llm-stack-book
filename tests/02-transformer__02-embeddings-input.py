"""
Runs the CPU-runnable Python blocks from
content/02-transformer/02-embeddings-input.md, concatenated in order so that
later blocks can rely on names defined by earlier ones (as they do in the
chapter itself). Each block is copied verbatim from the chapter; only the
minimal glue needed to make blocks that only *define* something (a class)
actually execute has been added, and is clearly marked "GLUE".

Blocks covered (all 8 heuristically CPU-runnable blocks in the chapter):
  #0 (line ~41)  - nn.Embedding shape / memory accounting
  #1 (line ~65)  - recommended weight initialization
  #2 (line ~80)  - lookup == index_select equivalence
  #3 (line ~125) - TiedTransformerHead (weight tying)
  #4 (line ~217) - InputPipeline (token + positional embedding)
  #5 (line ~322) - sparse gradient demonstration
  #6 (line ~374) - padding_idx behavior
  #7 (line ~409) - LlamaInputPipeline (token-only pipeline)

No blocks were skipped.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 70)
print("Block #0 (line ~41): nn.Embedding shape / memory accounting")
print("=" * 70)

# --- verbatim from the chapter ---
import torch
import torch.nn as nn

# Typical GPT-2 small vocabulary and model dimension
V = 50_257   # GPT-2 vocab size (BPE)
d_model = 768  # embedding dimension

embedding = nn.Embedding(V, d_model)

# The raw weight tensor: shape [V, d_model]
print(embedding.weight.shape)   # torch.Size([50257, 768])

# Memory: float32
bytes_f32 = V * d_model * 4
print(f"Embedding table (fp32): {bytes_f32 / 1e6:.1f} MB")  # ~154 MB

# Memory: bfloat16 (typical training)
bytes_bf16 = V * d_model * 2
print(f"Embedding table (bf16): {bytes_bf16 / 1e6:.1f} MB")  # ~77 MB
# --- end verbatim ---

assert embedding.weight.shape == (V, d_model)
assert abs(bytes_f32 / 1e6 - 154.4) < 1.0
assert abs(bytes_bf16 / 1e6 - 77.2) < 1.0


print()
print("=" * 70)
print("Block #1 (line ~65): recommended weight initialization")
print("=" * 70)

# --- verbatim from the chapter ---
# Recommended initialization used by GPT-2 / nanoGPT
nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
# --- end verbatim ---

# minimal honest check that init actually ran (std is small, not the default N(0,1))
assert embedding.weight.std().item() < 0.05


print()
print("=" * 70)
print("Block #2 (line ~80): lookup == index_select equivalence")
print("=" * 70)

# --- verbatim from the chapter ---
# Directly equivalent to nn.Embedding forward pass
ids = torch.randint(0, V, (4, 16))   # batch=4, seq_len=16

# Method 1: nn.Embedding (recommended; handles padding_idx, sparse gradients)
out1 = embedding(ids)                 # shape: [4, 16, 768]

# Method 2: Direct indexing (identical for inference, slightly faster in some cases)
out2 = embedding.weight[ids]          # shape: [4, 16, 768]

assert torch.allclose(out1, out2)
# --- end verbatim ---

print(f"out1.shape={tuple(out1.shape)}  out2.shape={tuple(out2.shape)}  allclose OK")


print()
print("=" * 70)
print("Block #3 (line ~125): TiedTransformerHead (weight tying)")
print("=" * 70)

# --- verbatim from the chapter ---
import torch
import torch.nn as nn
import torch.nn.functional as F

class TiedTransformerHead(nn.Module):
    """
    Minimal example of weight tying between token embedding and unembedding.
    In a full model, the Transformer layers would sit between embed() and unembed().
    """
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        # The single shared weight matrix: shape [vocab_size, d_model]
        self.embed = nn.Embedding(vocab_size, d_model)
        # No separate nn.Linear for the output: we share weights explicitly.

    def token_embed(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: [B, T] LongTensor  ->  [B, T, d_model] float"""
        return self.embed(ids)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        hidden: [B, T, d_model]
        returns logits: [B, T, vocab_size]

        F.linear computes  hidden @ weight.T + bias.
        We pass the embedding weight directly, with no bias.
        """
        return F.linear(hidden, self.embed.weight)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """End-to-end: token IDs -> logits (no Transformer layers for brevity)."""
        x = self.token_embed(ids)          # [B, T, d_model]
        # ... Transformer layers would go here ...
        return self.logits(x)              # [B, T, vocab_size]

# Verify parameter count
model = TiedTransformerHead(vocab_size=32_000, d_model=4096)
total = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total:,}")   # 32000 * 4096 = 131,072,000

# Verify the weights are truly shared (same object, not a copy)
assert model.embed.weight.data_ptr() == model.embed.weight.data_ptr()

# Gradient flows through both paths during training:
# dL/d(W_E) accumulates gradients from both token_embed() and logits().
# --- end verbatim ---

assert total == 32_000 * 4096

# GLUE: the chapter's snippet defines and instantiates the module but never
# actually runs a forward pass through it. Exercise forward() end-to-end with
# a tiny batch so the tying (embed + tied-unembed) is genuinely executed.
tied_ids = torch.randint(0, 32_000, (2, 5))
tied_logits = model(tied_ids)
assert tied_logits.shape == (2, 5, 32_000)
print(f"TiedTransformerHead forward OK, logits.shape={tuple(tied_logits.shape)}")


print()
print("=" * 70)
print("Block #4 (line ~217): InputPipeline (token + positional embedding)")
print("=" * 70)

# --- verbatim from the chapter ---
import torch
import torch.nn as nn

# ------------------------------------------------------------------
# Minimal end-to-end input pipeline for a decoder-only LM
# ------------------------------------------------------------------

class InputPipeline(nn.Module):
    """
    Converts a batch of token ID sequences into the first hidden state
    that will be fed to a Transformer block.

    Args:
        vocab_size:  Size of the token vocabulary (V).
        d_model:     Embedding dimension (d).
        max_seq_len: Maximum sequence length supported.
        dropout:     Dropout applied to the summed embedding (regularization).
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Token embedding table: shape [V, d_model]
        self.token_embed = nn.Embedding(vocab_size, d_model)

        # Learned positional embedding: shape [max_seq_len, d_model]
        # (Many modern models use RoPE instead; see Chapter 2.5)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.drop = nn.Dropout(dropout)

        # Initialize weights following GPT-2 conventions
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

    def forward(self, ids: torch.LongTensor) -> torch.Tensor:
        """
        ids: [B, T] integer tensor of token IDs (0 <= ids[i,j] < vocab_size)
        returns: [B, T, d_model] float tensor — the first hidden state
        """
        B, T = ids.shape
        device = ids.device

        # --- Token embeddings ---
        # Each integer is replaced by its row in the embedding table.
        tok_emb = self.token_embed(ids)           # [B, T, d_model]

        # --- Positional embeddings ---
        # Build position indices [0, 1, ..., T-1] and look them up.
        pos = torch.arange(T, device=device)      # [T]
        pos_emb = self.pos_embed(pos)             # [T, d_model]
        # Broadcast: [B, T, d_model] + [T, d_model] -> [B, T, d_model]

        # --- Sum and apply dropout ---
        x = self.drop(tok_emb + pos_emb)          # [B, T, d_model]
        return x


# Quick sanity check
pipeline = InputPipeline(vocab_size=50_257, d_model=768, max_seq_len=1024)
ids = torch.randint(0, 50_257, (4, 512))          # batch=4, seq_len=512
x = pipeline(ids)
print(x.shape)   # torch.Size([4, 512, 768])
print(f"Output norm (mean token): {x.norm(dim=-1).mean().item():.3f}")
# --- end verbatim ---

assert x.shape == (4, 512, 768)


print()
print("=" * 70)
print("Block #5 (line ~322): sparse gradient demonstration")
print("=" * 70)

# --- verbatim from the chapter ---
import torch
import torch.nn as nn

embed = nn.Embedding(50_257, 768)
ids = torch.tensor([[464, 3797, 3332, 319, 262, 2603, 13]])  # one sentence

x = embed(ids)             # [1, 7, 768]
loss = x.sum()             # toy loss
loss.backward()

grad = embed.weight.grad   # shape: [50257, 768]
# Most rows are exactly zero — only rows [464, 3797, 3332, 319, 262, 2603, 13]
# have nonzero gradients.
nonzero_rows = (grad.abs().sum(dim=1) > 0).sum().item()
print(f"Rows with nonzero gradient: {nonzero_rows} / 50257")
# Output: 7 / 50257   (one per unique token in the batch)
# --- end verbatim ---

assert nonzero_rows == 7


print()
print("=" * 70)
print("Block #6 (line ~374): padding_idx behavior")
print("=" * 70)

# --- verbatim from the chapter ---
# padding_idx ensures the pad token's embedding stays zero and receives
# no gradient — crucial for variable-length batched sequences.
embed = nn.Embedding(
    num_embeddings=128_256,
    embedding_dim=4096,
    padding_idx=128004,  # <|pad|> token ID
)

# The pad embedding is initialized to zero and frozen there.
print(embed.weight[128004].norm())   # tensor(0.)

# During backward, gradient for padding_idx is zeroed out automatically.
# --- end verbatim ---

assert embed.weight[128004].norm().item() == 0.0

# GLUE: confirm the "no gradient" claim by actually running a backward pass
# that includes the pad token, and checking its gradient row stays zero.
pad_ids = torch.tensor([[128004, 5, 6, 128004]])
pad_out = embed(pad_ids)
pad_out.sum().backward()
assert embed.weight.grad[128004].abs().sum().item() == 0.0
print("padding_idx row stays zero after forward+backward: OK")


print()
print("=" * 70)
print("Block #7 (line ~409): LlamaInputPipeline (token-only pipeline)")
print("=" * 70)

# --- verbatim from the chapter ---
import torch
import torch.nn as nn

class LlamaInputPipeline(nn.Module):
    """
    Llama-3 style input pipeline:
      - token_embed only (no positional embedding added here)
      - RoPE is applied inside attention layers (not shown here)
      - RMSNorm is the first op inside each Transformer block (not here)
    """
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        # Simple lookup; no padding_idx by default in Llama
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        # Initialization: std scales with d_model
        nn.init.normal_(self.embed_tokens.weight, std=d_model ** -0.5)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """input_ids: [B, T]  ->  hidden_states: [B, T, d_model]"""
        return self.embed_tokens(input_ids)
# --- end verbatim ---

# GLUE: the chapter defines the class but never instantiates/calls it.
# Instantiate with a small vocab/d_model and run a forward pass so the
# block is genuinely executed.
llama_pipeline = LlamaInputPipeline(vocab_size=32_000, d_model=256)
input_ids = torch.randint(0, 32_000, (2, 10))
hidden_states = llama_pipeline(input_ids)
assert hidden_states.shape == (2, 10, 256)
print(f"LlamaInputPipeline forward OK, hidden_states.shape={tuple(hidden_states.shape)}")


print()
print("=" * 70)
print("ALL 8 BLOCKS EXECUTED SUCCESSFULLY")
print("=" * 70)
