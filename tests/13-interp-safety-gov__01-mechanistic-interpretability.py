"""
Runnable-code test for content/13-interp-safety-gov/01-mechanistic-interpretability.md

Chapter blocks tested (assembled in document order):

    - block #0 (line ~73): linear probe + Hewitt & Liang control-task selectivity check
      -- executed against tiny synthetic activations/labels (glue fixtures below).
    - block #4 (line ~357): TopKSAE (minimal Top-K sparse autoencoder) class
      -- instantiated and run forward on a tiny toy batch of "residual-stream" activations.

SKIPPED (per task spec):
    - block #1 (line ~114): SKIP(needs-gpu/network): from-scratch logit lens on HuggingFace
      GPT-2 -- downloads model weights from the HF hub (network) and the harness explicitly
      forbids network calls in this test.
    - block #2 (line ~161): SKIP(non-python): fenced ```text``` block of illustrative logit-lens
      output, not code.
    - block #3 (line ~216): SKIP(needs-gpu/network): activation-patching experiment on
      HuggingFace GPT-2 -- again requires downloading model weights over the network.
    - block #5 (line ~416): SKIP(fragment): `steering_hook` is defined and shown with commented
      -out call-site pseudocode (`# h = model.transformer.h[LAYER]...`) operating on a live
      HookedTransformer/GPT-2 forward-hook `out` tuple that doesn't exist without a real model
      loaded (network-gated, see block #1/#3). We still define+call it below against a tiny
      hand-rolled stand-in "hook output" tuple so its actual arithmetic executes, honestly
      noting it is not the full chapter scenario.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression


# ============================================================================
# Fixtures for block #0: tiny synthetic "cached activations" + labels.
# The chapter's prose says:
#   h_train: (N, d_model) activations cached at a chosen layer/position; y_train: (N,) labels
# We build a toy dataset where the label is linearly decodable from h (so the real
# probe should clearly outperform the random-label control probe), matching the
# chapter's own framing of "a trustworthy linear probe has HIGH real accuracy and
# LOW control accuracy".
# ============================================================================

rng_fixture = np.random.default_rng(42)
N_train, N_val, d_model_probe = 200, 60, 16

# A random "feature direction" the label is defined along, plus noise -- mimics a
# linearly-represented feature per the linear representation hypothesis (§1.2).
true_dir = rng_fixture.normal(size=d_model_probe)
true_dir /= np.linalg.norm(true_dir)


def make_split(n, seed):
    r = np.random.default_rng(seed)
    h = r.normal(size=(n, d_model_probe)).astype(np.float32)
    score = h @ true_dir
    y = (score > 0).astype(np.int64)
    return torch.from_numpy(h), torch.from_numpy(y)


h_train, y_train = make_split(N_train, 1)
h_val, y_val = make_split(N_val, 2)


# ============================================================================
# block #0 (line ~73): linear probe + Hewitt & Liang control task
# ============================================================================

# h_train: (N, d_model) activations cached at a chosen layer/position; y_train: (N,) labels
# We deliberately use a LINEAR probe with strong L2 regularization -- low capacity by design.
probe = LogisticRegression(C=0.5, max_iter=2000)
probe.fit(h_train.numpy(), y_train.numpy())
real_acc = probe.score(h_val.numpy(), y_val.numpy())

# Control task (Hewitt & Liang): assign each *input type* a fixed random label, refit, re-score.
import numpy as np
rng = np.random.default_rng(0)
ctrl_labels = rng.integers(0, 2, size=y_train.shape)          # random but held fixed per example
ctrl = LogisticRegression(C=0.5, max_iter=2000).fit(h_train.numpy(), ctrl_labels)
ctrl_acc = ctrl.score(h_val.numpy(), rng.integers(0, 2, size=y_val.shape))

print(f"probe acc={real_acc:.3f}  control acc={ctrl_acc:.3f}  selectivity={real_acc-ctrl_acc:.3f}")
# A trustworthy linear probe has HIGH real accuracy and LOW control accuracy → high selectivity.

# Honest check on the book's own claim, using our constructed (linearly-separable-ish) data:
# the real probe should clearly beat chance, and the control (random labels) should be near chance.
assert real_acc > 0.8, f"expected the real linear probe to decode the planted direction well, got {real_acc}"
assert 0.2 < ctrl_acc < 0.8, f"control-task probe should hover near chance, got {ctrl_acc}"


# ============================================================================
# block #4 (line ~357): TopKSAE -- minimal Top-K sparse autoencoder
# ============================================================================

class TopKSAE(nn.Module):
    """A minimal Top-K sparse autoencoder for residual-stream activations."""
    def __init__(self, d_model, d_sae, k):
        super().__init__()
        self.k = k
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.W_enc = nn.Parameter(torch.randn(d_model, d_sae) / d_model**0.5)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        # Tie decoder to a *unit-norm* dictionary so feature directions are comparable.
        self.W_dec = nn.Parameter(F.normalize(torch.randn(d_sae, d_model), dim=1))

    def encode(self, x):
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc          # (B, d_sae)
        # Top-K: keep only the k largest activations per token; zero the rest.
        topv, topi = pre.topk(self.k, dim=-1)
        f = torch.zeros_like(pre).scatter_(-1, topi, F.relu(topv))
        return f

    def forward(self, x):
        f = self.encode(x)
        x_hat = f @ self.W_dec + self.b_dec
        # No L1 term needed: Top-K enforces sparsity structurally.
        recon = F.mse_loss(x_hat, x)
        return x_hat, f, recon

# Training loop sketch (activations streamed from a frozen LM at one hook point):
# sae = TopKSAE(d_model=768, d_sae=768*32, k=32)
# for x in activation_loader:               # x: (B, 768), L2-normalized is common
#     x_hat, f, loss = sae(x)
#     loss.backward(); opt.step(); opt.zero_grad()
#     sae.W_dec.data = F.normalize(sae.W_dec.data, dim=1)   # re-project to unit norm

# --- Minimal honest glue: instantiate on a tiny toy shape and actually run a few
# training steps so the encode/forward/backward path (the point of the block) executes.
torch.manual_seed(0)
d_model_sae, d_sae, k = 8, 64, 4
sae = TopKSAE(d_model=d_model_sae, d_sae=d_sae, k=k)
opt = torch.optim.Adam(sae.parameters(), lr=1e-2)

x_toy = torch.randn(16, d_model_sae)

losses = []
for step in range(20):
    x_hat, f, loss = sae(x_toy)
    opt.zero_grad()
    loss.backward()
    opt.step()
    with torch.no_grad():
        sae.W_dec.data = F.normalize(sae.W_dec.data, dim=1)   # re-project to unit norm
    losses.append(loss.item())

print(f"SAE reconstruction MSE: start={losses[0]:.4f} end={losses[-1]:.4f}")

# Sanity: Top-K sparsity is structural -- exactly k of d_sae features fire per row.
with torch.no_grad():
    f_final = sae.encode(x_toy)
    nonzero_per_row = (f_final != 0).sum(dim=-1)
assert (nonzero_per_row <= k).all(), "Top-K should keep at most k active features per token"
assert x_hat.shape == x_toy.shape
# Training should reduce reconstruction error on this toy batch.
assert losses[-1] < losses[0], f"expected reconstruction loss to decrease, got {losses[0]} -> {losses[-1]}"


# ============================================================================
# block #5 (line ~416): SKIP(fragment) for the full chapter scenario -- steering_hook
# is written against a live model's forward-hook signature (`out` = HookedTransformer
# tuple) which requires the network-gated GPT-2 load from block #1/#3. We still define
# and exercise the function's actual arithmetic against a tiny hand-built stand-in
# hook payload so the block's own logic (broadcast-add of alpha*v into a residual
# slice) runs for real, honestly noting this is glue, not the chapter's own call site.
# ============================================================================

# Steering by adding a precomputed direction `v` (unit norm) to the residual stream.
def steering_hook(v, alpha, positions=slice(None)):
    v = v.to(dtype=torch.float32)
    def hook(_m, _i, out):
        h = out[0]
        h[:, positions, :] = h[:, positions, :] + alpha * v   # broadcast add along d_model
        return (h,) + tuple(out[1:])
    return hook

# h = model.transformer.h[LAYER].register_forward_hook(steering_hook(v, alpha=+8.0))
# ... generate ...  then h.remove()
# alpha is in units of residual-stream norm; sweep it — too small = no effect,
# too large = incoherent text. The usable window is model/layer specific.

# --- Minimal honest glue: fake a (batch, seq, d_model) residual tensor and the
# (out,) tuple shape a real forward hook would receive, then call the built hook fn.
v_dir = torch.randn(d_model_sae)
v_dir = v_dir / v_dir.norm()
resid_before = torch.zeros(2, 5, d_model_sae)
fake_out = (resid_before.clone(),)

hook_fn = steering_hook(v_dir, alpha=2.0)
result = hook_fn(None, None, fake_out)
resid_after = result[0]

expected = resid_before + 2.0 * v_dir
assert torch.allclose(resid_after, expected), "steering_hook should add alpha*v to every position"
print("steering_hook glue check passed: residual shifted by alpha * v as expected")


print("\nAll mechanistic-interpretability blocks under test executed successfully.")
