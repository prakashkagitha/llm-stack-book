"""
Runnability test for content/03-pretraining/11-training-stability.md

Tests the 4 CPU-runnable Python blocks from the chapter, concatenated in
order (later blocks reuse names from earlier blocks, mirroring the chapter):

  - block #1  (line ~105): batch_anomaly_score / should_skip_batch
  - block #2  (line ~199): QKNormAttention (uses nn.RMSNorm)
  - block #4  (line ~293): init_transformer_weights
  - block #8  (line ~463): TrainingMonitor

Skipped blocks (not tested here):
  - #0  non-python (loss-curve ASCII diagram)
  - #3  fragment (worked-example math, not standalone code)
  - #5  fragment (z_loss function alone is trivial/tiny and self-contained;
        not one of the 4 required blocks, so not exercised here)
  - #6  needs-gpu (training_step uses torch.autocast(device_type='cuda') and
        a GradScaler workflow meant for CUDA mixed precision)
  - #7  non-python (spike decision tree ASCII diagram)
  - #9  needs-gpu (check_for_nan_distributed needs torch.distributed process
        group init; not standalone on CPU without a distributed launcher)
  - #10 non-python (debugging checklist text)
  - #11 needs-gpu (stability_probe.py explicitly raises NotImplementedError
        for load_model_and_batch, and probe_activation_norms uses
        torch.autocast(device_type='cuda'))
  - #12 non-python (pre-run hardening checklist text)
"""

import math
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

# wandb is a third-party, network-capable logging framework. The chapter's
# TrainingMonitor block does `import wandb` and calls `wandb.log(...)`.
# We mock it at the boundary with an in-memory stand-in so the block's OWN
# logic (metric assembly, spike-window math, activation-stat sampling)
# still executes offline, with zero network calls.
try:
    import wandb as _real_wandb  # noqa: F401
except Exception:
    _real_wandb = None


class _DummyWandb:
    """Offline stand-in for the wandb module: records logged dicts locally."""

    def __init__(self):
        self.logged = []

    def log(self, metrics):
        self.logged.append(metrics)


wandb = _DummyWandb()


# =====================================================================
# Block #1 (line ~105): batch anomaly detection
# =====================================================================

def batch_anomaly_score(
    input_ids: torch.Tensor,  # (B, T)
    ngram_n: int = 4,
    repetition_threshold: float = 0.4,
) -> torch.Tensor:
    """
    Returns a per-example anomaly score in [0, 1].
    High score = potentially bad batch element.

    Two signals:
      1. Repetition: fraction of n-grams that are duplicates.
      2. Entropy: low token-level entropy suggests degenerate text.
    """
    B, T = input_ids.shape
    scores = torch.zeros(B)

    for b in range(B):
        tokens = input_ids[b].tolist()

        # Signal 1: n-gram repetition fraction
        ngrams = [tuple(tokens[i:i+ngram_n]) for i in range(T - ngram_n)]
        if ngrams:
            counts = Counter(ngrams)
            # fraction of positions that are a repeated n-gram
            repeated = sum(v - 1 for v in counts.values() if v > 1)
            rep_frac = repeated / len(ngrams)
        else:
            rep_frac = 0.0

        # Signal 2: unigram entropy (in bits)
        tok_counts = Counter(tokens)
        total = len(tokens)
        entropy = -sum(
            (c / total) * (torch.log2(torch.tensor(c / total)).item())
            for c in tok_counts.values()
        )
        # For a typical English document, entropy > 8 bits; < 3 is suspicious.
        entropy_score = max(0.0, 1.0 - entropy / 8.0)

        scores[b] = 0.5 * rep_frac + 0.5 * entropy_score

    return scores


def should_skip_batch(input_ids: torch.Tensor, threshold: float = 0.35) -> bool:
    """Return True if the batch contains too many anomalous examples."""
    scores = batch_anomaly_score(input_ids)
    # Skip if average score is high OR if any single example is very bad
    return bool(scores.mean() > threshold or scores.max() > 0.75)


# =====================================================================
# Block #2 (line ~199): QK-Norm attention
# =====================================================================

class QKNormAttention(nn.Module):
    """
    Multi-head attention with per-head QK normalization.
    Prevents attention logit overflow, a common source of training spikes.
    """
    def __init__(self, d_model: int, n_heads: int, eps: float = 1e-6):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Learnable scale parameters, one per head dimension.
        # RMSNorm without learned scale would fix norm to 1.0;
        # a learnable scale restores representational flexibility.
        self.q_norm = nn.RMSNorm(self.d_k, eps=eps)
        self.k_norm = nn.RMSNorm(self.d_k, eps=eps)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        B, T, D = x.shape

        # Project to Q, K, V and reshape to (B, n_heads, T, d_k)
        def split_heads(t):
            return t.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        Q = split_heads(self.W_q(x))  # (B, H, T, d_k)
        K = split_heads(self.W_k(x))
        V = split_heads(self.W_v(x))

        # --- QK Norm: the key stability ingredient ---
        # Normalize along the head dimension; norms are now bounded.
        Q = self.q_norm(Q)
        K = self.k_norm(K)

        # Scaled dot-product attention (safe now that Q, K are normalized)
        scale = self.d_k ** -0.5
        attn = (Q @ K.transpose(-2, -1)) * scale  # (B, H, T, T)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


# =====================================================================
# Block #4 (line ~293): stability-oriented weight init
# =====================================================================

def init_transformer_weights(model: nn.Module, n_layers: int, d_model: int):
    """
    Stability-oriented weight initialization for a GPT-style transformer.
    Based on the GPT-2/NanoGPT pattern with depth scaling.
    """
    for name, param in model.named_parameters():
        if param.dim() < 2:
            # Biases, norms — leave as default (zeros / ones)
            continue

        if 'embedding' in name:
            # Embedding table: small init to keep logits bounded at step 0
            nn.init.normal_(param, mean=0.0, std=d_model ** -0.5)

        elif 'c_proj' in name or 'out_proj' in name:
            # Residual-path output projections (attn output + MLP output).
            # Scaled down by 1/sqrt(2 * n_layers) so that the residual stream
            # norm grows as O(1) rather than O(sqrt(L)) at initialization.
            std = (2 * n_layers) ** -0.5
            nn.init.normal_(param, mean=0.0, std=std)

        elif 'q_proj' in name or 'k_proj' in name:
            # Query/Key projections: initialize so logit std ≈ 1
            d_k = param.shape[0]  # assuming (d_k, d_model) layout
            nn.init.normal_(param, mean=0.0, std=d_k ** -0.25)

        else:
            # Default: Kaiming/He for everything else
            nn.init.normal_(param, mean=0.0, std=0.02)


# =====================================================================
# Block #8 (line ~463): training monitor
# =====================================================================

class TrainingMonitor:
    """
    Collects and logs training health signals.
    Designed to add minimal overhead: most metrics are computed from
    tensors already in memory.
    """
    def __init__(self, log_every: int = 10, spike_window: int = 50):
        self.log_every = log_every
        self.spike_window = spike_window
        self.loss_history = []
        self.step = 0

    def log_step(
        self,
        loss: float,
        grad_norm: float,
        model: torch.nn.Module,
        lr: float,
    ):
        self.step += 1
        self.loss_history.append(loss)

        if self.step % self.log_every != 0:
            return

        metrics = {
            'train/loss': loss,
            'train/perplexity': math.exp(min(loss, 20)),  # clamp to avoid overflow
            'train/grad_norm': grad_norm,
            'train/lr': lr,
            'train/step': self.step,
        }

        # --- Activation statistics (sampled from a few layers) ---
        # Hook-based; only active when we log.
        act_stats = self._sample_activation_stats(model)
        metrics.update(act_stats)

        # --- Spike detector ---
        if len(self.loss_history) >= self.spike_window:
            window = self.loss_history[-self.spike_window:]
            baseline = sum(window[:self.spike_window // 2]) / (self.spike_window // 2)
            recent = sum(window[self.spike_window // 2:]) / (self.spike_window // 2)
            metrics['train/spike_delta'] = recent - baseline

        wandb.log(metrics)

    @torch.no_grad()
    def _sample_activation_stats(self, model: torch.nn.Module) -> dict:
        """
        Compute per-layer activation norms for the most recent forward pass.
        In practice, wire this to forward hooks registered on a few layers.
        Here we illustrate with a direct parameter-norm proxy.
        """
        stats = {}
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                # Track weight matrix spectral norm proxy (Frobenius / sqrt(numel))
                rms = param.norm() / (param.numel() ** 0.5)
                short = name.replace('.weight', '').replace('model.', '')
                stats[f'weights/{short}_rms'] = rms.item()
        return stats


# =====================================================================
# Test harness: actually execute each block's code with tiny CPU inputs
# =====================================================================

def main():
    torch.manual_seed(0)

    # --- Block #1: batch_anomaly_score / should_skip_batch ---
    vocab_size = 100
    B, T = 4, 20

    # A "clean" batch: random tokens, high entropy, low repetition.
    clean_batch = torch.randint(0, vocab_size, (B, T))

    # A degenerate batch: mostly repeated token id 7 (low entropy, high
    # n-gram repetition) — exactly the "aaaa...aaaa" pattern the chapter
    # describes.
    degenerate_batch = torch.full((B, T), 7, dtype=torch.long)
    degenerate_batch[:, 0] = 3  # keep it a valid LongTensor with one outlier

    clean_scores = batch_anomaly_score(clean_batch)
    degenerate_scores = batch_anomaly_score(degenerate_batch)
    assert clean_scores.shape == (B,)
    assert degenerate_scores.shape == (B,)
    # The degenerate (repeated-token) batch must score higher than the
    # random one — this is the core claim the block makes.
    assert degenerate_scores.mean().item() > clean_scores.mean().item(), (
        "degenerate batch should score higher than a clean random batch"
    )

    assert should_skip_batch(degenerate_batch) is True, "degenerate batch should be flagged for skipping"
    assert should_skip_batch(clean_batch) is False, "clean random batch should not be flagged"
    print(f"[block #1] clean mean score={clean_scores.mean().item():.3f}, "
          f"degenerate mean score={degenerate_scores.mean().item():.3f} -- OK")

    # --- Block #2: QKNormAttention ---
    d_model, n_heads = 32, 4
    attn = QKNormAttention(d_model=d_model, n_heads=n_heads)
    x = torch.randn(2, 6, d_model)
    out = attn(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all(), "QKNormAttention output should be finite"

    # Also exercise the masked branch.
    mask = torch.ones(2, 1, 6, 6)
    mask[:, :, :, 3:] = 0  # mask out the last few positions
    out_masked = attn(x, mask=mask)
    assert out_masked.shape == x.shape
    assert torch.isfinite(out_masked).all(), "masked attention output should be finite"
    print(f"[block #2] QKNormAttention forward output shape={tuple(out.shape)} -- OK")

    # --- Block #4: init_transformer_weights ---
    n_layers = 3

    class ToyBlock(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.c_proj = nn.Linear(d_model, d_model, bias=False)

    class ToyTransformer(nn.Module):
        def __init__(self, d_model, n_layers, vocab_size):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.blocks = nn.ModuleList(ToyBlock(d_model) for _ in range(n_layers))
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

        def named_parameters(self, *a, **kw):
            # Prefix names so the substring checks in init_transformer_weights
            # ('embedding', 'c_proj', 'q_proj'/'k_proj', 'out_proj') match,
            # exactly as they would in a real named-module hierarchy.
            return super().named_parameters(*a, **kw)

    toy_model = ToyTransformer(d_model=d_model, n_layers=n_layers, vocab_size=vocab_size)
    init_transformer_weights(toy_model, n_layers=n_layers, d_model=d_model)

    emb_std = toy_model.embedding.weight.std().item()
    expected_emb_std = d_model ** -0.5
    assert abs(emb_std - expected_emb_std) / expected_emb_std < 0.5, (
        f"embedding std {emb_std:.4f} far from expected {expected_emb_std:.4f}"
    )

    out_proj_std = toy_model.out_proj.weight.std().item()
    expected_out_std = (2 * n_layers) ** -0.5
    assert abs(out_proj_std - expected_out_std) / expected_out_std < 0.5, (
        f"out_proj std {out_proj_std:.4f} far from expected {expected_out_std:.4f}"
    )
    print(f"[block #4] embedding std={emb_std:.4f} (expected ~{expected_emb_std:.4f}), "
          f"out_proj std={out_proj_std:.4f} (expected ~{expected_out_std:.4f}) -- OK")

    # --- Block #8: TrainingMonitor ---
    monitor = TrainingMonitor(log_every=5, spike_window=10)
    tiny_model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))

    # Simulate 20 steps: a smooth decreasing loss, then a spike near the end
    # so the spike-delta branch of log_step actually executes.
    losses = [3.0 - 0.05 * i for i in range(15)] + [3.5, 3.6, 3.4, 3.2, 3.0]
    for i, loss_val in enumerate(losses):
        monitor.log_step(loss=loss_val, grad_norm=1.0 + 0.1 * i, model=tiny_model, lr=3e-4)

    assert monitor.step == len(losses)
    assert len(wandb.logged) > 0, "TrainingMonitor should have logged at least once"
    last_logged = wandb.logged[-1]
    assert 'train/loss' in last_logged
    assert 'train/perplexity' in last_logged
    assert any(k.startswith('weights/') for k in last_logged), "activation/weight stats should be present"
    assert 'train/spike_delta' in last_logged, "spike_delta should appear once loss_history >= spike_window"
    print(f"[block #8] logged {len(wandb.logged)} metric dicts, last spike_delta="
          f"{last_logged['train/spike_delta']:.4f} -- OK")

    print("\nAll tested blocks executed successfully (blocks #1, #2, #4, #8).")


if __name__ == "__main__":
    main()
