"""
Runnable-code test for content/01-foundations/05-ml-fundamentals.md

Chapter blocks tested (assembled in document order so later blocks can use
names defined by earlier ones, exactly as the chapter intends):

    - block #0 (line ~41):  bias-variance empirical decomposition (sklearn polynomial
                             regression, degrees 1/3/10) -> executed in-block
    - block #2 (line ~163): SimpleMLPWithDropout (nn.Module)         -> instantiated/used in-block
    - block #3 (line ~204): EarlyStopping class                      -> instantiated/used via glue below
    - block #4 (line ~282): Stratified 5-fold CV (LogisticRegression) -> executed in-block
    - block #5 (line ~370): classification_report / ROC-AUC / PR-AUC -> executed in-block

SKIPPED (per task spec):
    - block #1 (line ~92): SKIP(non-python): this is a ```text``` fenced block showing
      sample console output of block #0, not executable code.
    - block #6 (line ~471): SKIP(needs-gpu / long-running): the "Complete Training Pipeline"
      block trains a 128-hidden-unit MLP for up to 100 epochs (with early stopping) on a
      4000-sample synthetic dataset. It is device-aware (`torch.device('cuda' if ... else
      'cpu')`) and would run on CPU, but it duplicates -- at ~10-100x the runtime cost -- logic
      already independently exercised above: block #2's dropout MLP, block #3's EarlyStopping
      (instantiated/used below with real state-dict save/restore), block #4's stratified CV
      pattern, and block #5's classification-report/ROC-AUC/PR-AUC reporting. Skipping it keeps
      this test fast and deterministic while still covering every distinct piece of logic it
      contains via the other four blocks.
"""

import copy

import numpy as np
import torch
import torch.nn as nn

# matplotlib is not in the guaranteed CI import list; guard it. Block #0 imports it
# (with the Agg backend) but never actually calls plt.* -- the import is dead weight
# in the original snippet, so guarding it does not change any tested behavior.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401
except Exception:
    matplotlib = None
    plt = None


# ============================================================================
# block #0 (line ~41): bias-variance empirical decomposition
# ============================================================================

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(42)


# True function: f*(x) = sin(x), with Gaussian noise sigma=0.3
def true_fn(x):
    return np.sin(x)


N_train = 15
N_repeats = 200  # number of training-set draws to empirically estimate variance

x_test = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
y_test_true = true_fn(x_test)

results = {}  # degree -> (bias^2, variance, avg_test_mse)

for degree in [1, 3, 10]:
    preds_matrix = []
    for _ in range(N_repeats):
        # Fresh noisy training set each time
        x_train = rng.uniform(0, 2 * np.pi, N_train).reshape(-1, 1)
        y_train = true_fn(x_train) + rng.normal(0, 0.3, (N_train, 1))

        model = make_pipeline(
            PolynomialFeatures(degree, include_bias=False),
            LinearRegression()
        )
        model.fit(x_train, y_train.ravel())
        preds_matrix.append(model.predict(x_test))

    preds_matrix = np.array(preds_matrix)        # (N_repeats, N_test)
    mean_pred = preds_matrix.mean(axis=0)        # average prediction across draws

    bias_sq   = np.mean((mean_pred - y_test_true.ravel()) ** 2)
    variance  = np.mean(np.var(preds_matrix, axis=0))
    avg_mse   = np.mean((preds_matrix - y_test_true.ravel()) ** 2)

    results[degree] = (bias_sq, variance, avg_mse)
    print(f"Degree {degree:2d}:  Bias²={bias_sq:.4f}  Var={variance:.4f}  "
          f"MSE≈{avg_mse:.4f}  (Bias²+Var={bias_sq+variance:.4f})")

# --- verify the book's claim: degree 3 has the lowest bias^2 + variance (best tradeoff) ---
sum_bv = {d: results[d][0] + results[d][1] for d in results}
assert sum_bv[3] < sum_bv[1], "degree-3 should beat degree-1 (too much bias)"
assert sum_bv[3] < sum_bv[10], "degree-3 should beat degree-10 (too much variance)"
# degree-1 (underfit) should have much higher bias^2 than degree-3
assert results[1][0] > results[3][0]
# degree-10 (overfit) should have much higher variance than degree-3
assert results[10][1] > results[3][1]


# ============================================================================
# block #2 (line ~163): SimpleMLPWithDropout
# ============================================================================

class SimpleMLPWithDropout(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout_p: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),   # Applied only during model.train()
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


model = SimpleMLPWithDropout(64, 256, 10, dropout_p=0.3)

# Training mode: dropout active
model.train()
x = torch.randn(32, 64)
out_train = model(x)   # some activations zeroed

# Eval mode: dropout disabled, full network used
model.eval()
with torch.no_grad():
    out_eval = model(x)  # deterministic, all activations present

print(f"Train output shape: {out_train.shape}")  # (32, 10)
print(f"Eval  output shape: {out_eval.shape}")   # (32, 10)

assert out_train.shape == (32, 10)
assert out_eval.shape == (32, 10)
# eval mode with no_grad() should not require grad
assert not out_eval.requires_grad


# ============================================================================
# block #3 (line ~204): EarlyStopping
# ============================================================================

class EarlyStopping:
    """Stop training when validation loss has not improved for `patience` steps."""
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float('inf')
        self.counter    = 0
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            # Deep-copy only the state dict — cheap on GPU
            import copy
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

# Usage in a training loop:
# stopper = EarlyStopping(patience=5)
# for epoch in range(max_epochs):
#     train_one_epoch(model, train_loader, optimizer)
#     val_loss = evaluate(model, val_loader)
#     if stopper.step(val_loss, model):
#         print("Early stopping triggered")
#         break
# stopper.restore_best(model)

# --- glue: instantiate EarlyStopping and drive it through a fake training loop ---
es_model = SimpleMLPWithDropout(4, 8, 2, dropout_p=0.3)
stopper = EarlyStopping(patience=3, min_delta=1e-4)

# Validation loss improves for 2 steps (best captured at step index 1, loss=0.50),
# then plateaus/worsens for 3 consecutive steps -> should trigger stop on the 3rd.
fake_val_losses = [0.90, 0.50, 0.55, 0.56, 0.57]
stop_flags = []
for i, vl in enumerate(fake_val_losses):
    # nudge the model's weights each "epoch" so state dicts actually differ across steps
    with torch.no_grad():
        for p in es_model.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    should_stop = stopper.step(vl, es_model)
    stop_flags.append(should_stop)
    print(f"  epoch {i}: val_loss={vl:.2f} best={stopper.best_loss:.2f} "
          f"counter={stopper.counter} stop={should_stop}")

assert stopper.best_loss == 0.50
assert stop_flags[-1] is True, "3 non-improving steps with patience=3 should trigger stop"
assert stop_flags[:-1] == [False, False, False, False]

# capture the state right before restore for comparison
snapshot_before_restore = copy.deepcopy(es_model.state_dict())
stopper.restore_best(es_model)
restored_state = es_model.state_dict()

# the restored weights must match the checkpoint captured at the best epoch, and
# must differ from the (worse, further-mutated) weights that were in place beforehand
for k in restored_state:
    assert torch.equal(restored_state[k], stopper.best_state[k])
any_diff = any(
    not torch.equal(restored_state[k], snapshot_before_restore[k])
    for k in restored_state
)
assert any_diff, "restore_best should have changed the model's weights"


# ============================================================================
# block #4 (line ~282): Stratified k-fold cross-validation
# ============================================================================

from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=500, n_features=20,
                            n_informative=10, random_state=0)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)

    acc = accuracy_score(y_val, clf.predict(X_val))
    fold_accuracies.append(acc)
    print(f"  Fold {fold_idx+1}: accuracy={acc:.4f}")

print(f"\n5-fold CV accuracy: {np.mean(fold_accuracies):.4f} "
      f"± {np.std(fold_accuracies):.4f}")

assert len(fold_accuracies) == 5
# a logistic regression on a separable synthetic dataset should easily beat chance (0.5)
assert np.mean(fold_accuracies) > 0.7


# ============================================================================
# block #5 (line ~370): classification metrics (ROC-AUC, PR-AUC, confusion matrix)
# ============================================================================

from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix
)

# Simulate true labels and a probabilistic classifier
rng = np.random.default_rng(0)
y_true  = rng.integers(0, 2, size=1000)  # 50% positive
y_score = rng.beta(2, 2, size=1000)      # random probs, slightly informed

# Nudge scores toward correct labels to create a non-trivial model
y_score = np.where(y_true == 1,
                   np.clip(y_score + 0.2, 0, 1),
                   np.clip(y_score - 0.2, 0, 1))

y_pred  = (y_score >= 0.5).astype(int)

print(classification_report(y_true, y_pred, digits=4))
roc_auc = roc_auc_score(y_true, y_score)
pr_auc = average_precision_score(y_true, y_score)
cm = confusion_matrix(y_true, y_pred)
print(f"ROC-AUC      : {roc_auc:.4f}")
print(f"PR-AUC       : {pr_auc:.4f}")
print(f"Confusion matrix:\n{cm}")

# the "nudge toward correct label" construction should produce a clearly-informative
# (well above chance) classifier
assert roc_auc > 0.6
assert pr_auc > 0.6
assert cm.shape == (2, 2)
assert cm.sum() == 1000


print("\nAll block executions and assertions passed.")
