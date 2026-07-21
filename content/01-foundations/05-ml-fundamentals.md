# 1.5 Machine Learning Fundamentals

Machine learning is the art of extracting structure from data automatically, rather than writing explicit rules. Before we can understand how large language models are trained, fine-tuned, and evaluated, we need a rock-solid foundation in the concepts that govern *all* supervised learning systems: the bias-variance tradeoff, regularization, evaluation protocols, and the metrics that tell us whether a model is actually working. This chapter is deliberately broad — it is the "interview breadth core" that interviewers at Google and elsewhere probe before diving into LLM-specific depth.

We will build every idea from first principles and anchor each concept in code you can run. The chapter connects forward to [Neural Networks From Scratch: MLPs & Backprop](../01-foundations/06-neural-nets-from-scratch.html), [Calculus, Optimization & Convexity](../01-foundations/03-calculus-optimization.html), and eventually to how these ideas manifest in LLM training (see [The Pretraining Objective & Loss](../03-pretraining/03-pretraining-objective.html)).

---

## Learning Paradigms: Supervised, Unsupervised, and Self-Supervised

Every machine learning algorithm lives in one of three paradigms, distinguished by the type of feedback signal available during training.

**Supervised learning** is the most classical setting. We have a dataset $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$ of input-output pairs and we want to learn a function $f_\theta: \mathcal{X} \to \mathcal{Y}$ that minimizes some loss $\mathcal{L}(f_\theta(\mathbf{x}), y)$ over unseen inputs. Classification (discrete $y$) and regression (continuous $y$) are the canonical sub-cases. A neural network trained with cross-entropy on ImageNet labels is supervised learning.

**Unsupervised learning** receives only inputs $\{x_i\}$ with no labels. The goal is to discover *structure* — clusters (k-means, DBSCAN), latent representations (PCA, autoencoders, VAEs), or density models (GMMs, normalizing flows). The signal comes from the data geometry itself.

**Self-supervised learning (SSL)** sits at the intersection. Labels are automatically derived from the data itself, without human annotation. A model predicts a masked or shifted portion of the input. This is the paradigm that makes LLMs possible: the next-token prediction objective creates billions of (context, next-token) pairs from raw text for free. BERT's masked-language-modeling and contrastive objectives like SimCLR are also SSL. See [The Pretraining Objective & Loss](../03-pretraining/03-pretraining-objective.html) for the language-model variant in depth.

{{fig:mlfund-paradigm-spectrum}}

---

## The Bias-Variance Tradeoff

Why does a model that fits training data perfectly often perform worse on new data? The answer lies in the bias-variance decomposition of expected prediction error.

For a regression problem with true function $f^*(x)$ and additive noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$, the expected mean squared error of an estimator $\hat{f}$ at a point $x$ decomposes as:

$$
\mathbb{E}\left[(y - \hat{f}(x))^2\right] = \underbrace{\left(\mathbb{E}[\hat{f}(x)] - f^*(x)\right)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}\left[\left(\hat{f}(x) - \mathbb{E}[\hat{f}(x)]\right)^2\right]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Irreducible noise}}
$$

- **Bias** measures systematic error — how far off is the average prediction from the truth? A linear model fit to non-linear data has high bias.
- **Variance** measures sensitivity to the specific training set — does the model change dramatically when trained on a different draw from the same distribution? A degree-20 polynomial has high variance.
- **Irreducible noise** $\sigma^2$ is the floor: even the perfect model cannot escape it.

The classic tradeoff: as model complexity increases, bias falls but variance rises. The optimal model complexity minimizes their sum.

{{fig:mlfund-bias-variance-decomposition}}

```python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
```

Running this typically produces something like (magnitudes depend on the RNG seed):

```text
Degree  1:  Bias²=0.4810  Var=0.0062  MSE≈0.5783  (Bias²+Var=0.4872)
Degree  3:  Bias²=0.0041  Var=0.0101  MSE≈0.1041  (Bias²+Var=0.0142)
Degree 10:  Bias²=0.0015  Var=0.2341  MSE≈0.3256  (Bias²+Var=0.2356)
```

Degree-3 wins: low enough bias to capture the sinusoid, low enough variance because 15 training points constrain the fit well.

!!! note "The modern twist: double descent"
    In deep learning, the bias-variance tradeoff's U-shaped test-error curve turns into a *double-descent* curve. After the classical interpolation threshold (where the model perfectly memorizes training data), test error *decreases again* as model size grows further. This is an active research area — see Belkin et al., "Reconciling modern machine-learning practice and the classical bias–variance trade-off," PNAS 2019.

---

## Overfitting, Underfitting, and the Generalization Puzzle

**Underfitting** (high bias): the model is too simple to capture the signal. Training loss and test loss are both high. Fix: more capacity, more features, less regularization, longer training.

**Overfitting** (high variance): the model memorizes training data idiosyncrasies. Training loss is low but test loss is high. The gap is called the *generalization gap*. Fix: more data, regularization, or a simpler model.

The fundamental measure of generalization is the *population risk*:

$$
R(f) = \mathbb{E}_{(\mathbf{x}, y) \sim p_{\text{data}}}\left[\mathcal{L}(f(\mathbf{x}), y)\right]
$$

We cannot compute this directly; we estimate it with the *empirical risk* on held-out data. Classical VC theory bounds the generalization gap as roughly $O(\sqrt{d_{\text{VC}} / N})$ where $d_{\text{VC}}$ is the Vapnik–Chervonenkis dimension of the model class. For neural networks, VC theory is loose — modern understanding relies on PAC-Bayes, neural tangent kernel theory, and empirical scaling laws (see [Scaling Laws: Kaplan, Chinchilla & Beyond](../03-pretraining/04-scaling-laws.html)).

**Why LLMs seem to avoid classical overfitting.** Self-supervised objectives on internet-scale data ($N \sim 10^{12}$ tokens) push the training-distribution gap close to zero. The model generalizes not because it memorizes less but because the training distribution is so broad that it effectively approximates $p_{\text{data}}$.

---

## Regularization

Regularization refers to any technique that reduces generalization gap, usually by adding a penalty to the training objective or by constraining the model during training.

### L2 Regularization (Weight Decay)

Add a penalty proportional to the squared norm of the parameters:

$$
\mathcal{L}_{\text{reg}} = \mathcal{L}_{\text{data}} + \frac{\lambda}{2} \|\theta\|_2^2
$$

The gradient update becomes:

$$
\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}_{\text{data}} - \eta \lambda \theta = (1 - \eta\lambda)\theta - \eta \nabla_\theta \mathcal{L}_{\text{data}}
$$

The $(1 - \eta\lambda)$ factor is why this is also called *weight decay*: parameters are shrunk toward zero at every step. L2 has a Bayesian interpretation as a Gaussian prior $\theta \sim \mathcal{N}(0, \lambda^{-1}I)$ — MAP estimation with this prior yields L2 regularization.

### L1 Regularization (Lasso)

$$
\mathcal{L}_{\text{reg}} = \mathcal{L}_{\text{data}} + \lambda \|\theta\|_1
$$

L1 promotes *sparsity*: many parameters are driven to exactly zero, performing automatic feature selection. Bayesian interpretation: Laplace prior on weights. In practice, L1 is rarely used for neural networks (subgradients complicate optimization) but is standard for linear models and feature selection.

{{fig:mlfund-l1-l2-geometry}}

### Dropout

Dropout (Srivastava et al., 2014) randomly zeroes each activation with probability $p$ during training:

$$
\tilde{h}_i = \frac{1}{1-p} \cdot \text{Bernoulli}(1-p) \cdot h_i
$$

The $1/(1-p)$ scale factor keeps the expected activation unchanged (inverted dropout). At inference, no masking is applied. Intuition: each forward pass uses a different *subnetwork*, forcing the model to learn redundant representations that cannot co-adapt. An equivalent view is that dropout approximates an ensemble of $2^d$ subnetworks (where $d$ is the number of units).

```python
import torch
import torch.nn as nn

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
```

### Early Stopping

Rather than adding a penalty to the loss, *early stopping* monitors validation loss and halts training when it begins to increase. In practice, keep a running best checkpoint and restore it at the end.

```python
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
```

Early stopping is, in a sense, equivalent to L2 regularization for gradient descent (see Goodfellow et al., 2016, Chapter 7 for the proof sketch). It has zero computational overhead and is almost always applied when training LLMs via the validation perplexity curve.

### Other Regularization Techniques

| Technique | Mechanism | Typical use |
|---|---|---|
| Batch Normalization | Normalizes activations, adds noise that acts like regularization | CNNs, Transformers |
| Layer Normalization | Same but over feature dim; stable for sequence models | Transformers |
| Data augmentation | Expands effective dataset size | Vision, NLP paraphrasing |
| Label smoothing | Softens targets from 0/1 to $\epsilon/(K-1)$, $1-\epsilon$ | Classification heads |
| Gradient clipping | Clips gradient norm; prevents exploding gradients | RNNs, Transformers |

---

## Train / Validation / Test Splits and Cross-Validation

### The Three-Way Split

The gold standard protocol separates data into three non-overlapping subsets:

- **Training set** — used to compute gradients and update parameters.
- **Validation set** — used to tune hyperparameters, select checkpoints, and estimate generalization during development. No gradient flows here.
- **Test set** — used *once* at the end to report final performance. Treating test data as if it were validation data leads to *test set contamination*, a subtle form of data leakage.

A common split for medium-sized datasets is 70/15/15 or 80/10/10. For very large datasets (millions of examples) even 1% is often enough for validation and test.

### Cross-Validation

When data is scarce, $k$-fold cross-validation makes better use of every example:

1. Partition data into $k$ equal folds.
2. Train $k$ times, each time holding out one fold as validation.
3. Average the $k$ validation scores as the generalization estimate.

$$
\text{CV}_k = \frac{1}{k} \sum_{i=1}^{k} \text{Score}_i
$$

For $k = N$ (leave-one-out CV, LOOCV) the estimate is nearly unbiased but expensive. Stratified $k$-fold preserves class ratios in each fold.

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

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
```

### Data Leakage — The Silent Killer

**Data leakage** occurs when information from the validation or test set contaminates the model during training or preprocessing. It causes the model to appear better than it truly is on unseen data.

Common leakage sources:

1. **Preprocessing with full-dataset statistics.** Fitting a StandardScaler on all data before splitting; the test set's statistics influence the training normalization.
2. **Feature engineering that looks ahead.** A time-series feature like "average sales next week" leaks future labels into training features.
3. **Duplicates across splits.** Near-duplicate samples in both train and test; especially common with web-scraped LLM pretraining data (see [Data Cleaning, Deduplication & Quality Filtering](../03-pretraining/02-data-cleaning-dedup.html)).
4. **Label leakage.** A feature that is a direct proxy for the label (e.g., including the diagnosis code when predicting disease).

!!! warning "Always fit preprocessing on training data only"
    Any scaler, tokenizer vocabulary, or normalization constant must be computed on the *training* split and then *applied* (not refit) to validation and test. In `sklearn`, use `Pipeline` objects so that `.fit_transform(X_train)` and `.transform(X_val)` are clearly separated.

---

## Evaluation Metrics

Choosing the right metric is at least as important as choosing the right model. The loss function guides training; evaluation metrics guide *decisions*.

### Classification Metrics

For binary classification with a threshold at 0.5:

$$
\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

The $F_\beta$ score generalizes this:

$$
F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}
$$

- $\beta > 1$: weight recall higher (e.g., medical screening — false negatives are costly).
- $\beta < 1$: weight precision higher (e.g., spam filter — false positives annoy users).

**The confusion matrix** for binary classification:

{{fig:mlfund-confusion-matrix}}

For multi-class problems, we report per-class precision/recall and average them:

- **Macro average**: unweighted mean across classes — treats all classes equally regardless of size.
- **Weighted average**: weighted by class support — appropriate for imbalanced datasets.

### ROC and AUC

The **Receiver Operating Characteristic (ROC)** curve plots True Positive Rate (TPR = recall) vs. False Positive Rate (FPR = $FP / (FP + TN)$) as the classification threshold varies from 1 to 0. A perfect classifier passes through $(0, 1)$; a random classifier lies on the diagonal.

**AUC (Area Under the ROC Curve)** summarizes the entire curve in one number. AUC = 0.5 is random; AUC = 1.0 is perfect. AUC has a useful probabilistic interpretation: it equals the probability that the model ranks a randomly chosen positive example higher than a randomly chosen negative.

$$
\text{AUC} = P(\hat{p}(x^+) > \hat{p}(x^-))
$$

**PR-AUC** (Precision-Recall AUC) is preferred for highly imbalanced datasets, where even a near-trivial model can achieve high ROC-AUC by virtue of the many true negatives.

```python
import numpy as np
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
print(f"ROC-AUC      : {roc_auc_score(y_true, y_score):.4f}")
print(f"PR-AUC       : {average_precision_score(y_true, y_score):.4f}")
print(f"Confusion matrix:\n{confusion_matrix(y_true, y_pred)}")
```

### Calibration

A model is **calibrated** if its predicted probability $\hat{p}$ equals the empirical frequency of the positive class among examples assigned that score. Formally, for all $p \in [0,1]$:

$$
P(y = 1 \mid \hat{p}(x) = p) = p
$$

A model that outputs 0.9 for 1000 examples should have approximately 900 of them be positive. Miscalibration matters enormously in production: a classifier guarding a medical device with $\hat{p} = 0.9$ should not be trusted if the true frequency is 0.5.

The **Expected Calibration Error (ECE)** estimates calibration by binning predictions:

$$
\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|
$$

where $B_m$ is the $m$-th bin of predictions, $\text{acc}(B_m)$ is the fraction of positives in that bin, and $\text{conf}(B_m)$ is the mean predicted probability.

Neural networks are often *overconfident* — their raw softmax scores are not well-calibrated. **Temperature scaling** is a simple post-hoc fix: divide logits by a learned scalar $T > 1$ before softmax, which smooths the output distribution.

!!! example "Worked example: Precision, Recall, F1"
    Suppose we are evaluating a spam filter on 1000 emails: 100 true spam, 900 ham. Our model predicts:

    - TP = 80  (spam correctly flagged)
    - FP = 20  (ham incorrectly flagged as spam)
    - FN = 20  (spam that slipped through)
    - TN = 880 (ham correctly passed)

    $$\text{Precision} = \frac{80}{80+20} = 0.80$$

    $$\text{Recall} = \frac{80}{80+20} = 0.80$$

    $$F_1 = \frac{2 \times 0.80 \times 0.80}{0.80 + 0.80} = 0.80$$

    $$\text{Accuracy} = \frac{80+880}{1000} = 0.96$$

    Accuracy looks impressive at 96%, but this is inflated by the easy ham class. If spam were only 10 of 1000 emails (1% prevalence) and we flagged nothing, accuracy would be 99% — completely useless. This is why F1 and AUC are the right metrics for imbalanced classification.

### Regression Metrics

| Metric | Formula | When to use |
|---|---|---|
| MAE | $\frac{1}{N}\sum|y_i - \hat{y}_i|$ | Robust to outliers |
| MSE | $\frac{1}{N}\sum(y_i - \hat{y}_i)^2$ | Differentiable; penalizes large errors |
| RMSE | $\sqrt{\text{MSE}}$ | Same units as target |
| $R^2$ | $1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$ | Fraction of variance explained |
| MAPE | $\frac{1}{N}\sum|\frac{y_i - \hat{y}_i}{y_i}|$ | Relative error; undefined if $y_i = 0$ |

For language models, the primary metric is **perplexity**, which is the exponentiated average cross-entropy per token:

$$
\text{PPL} = \exp\!\left(-\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(w_t \mid w_{<t})\right)
$$

A perplexity of 10 means the model is, on average, as uncertain as choosing uniformly among 10 equally likely next tokens.

---

## The Generalization Puzzle

How does any learning algorithm generalize at all? The deep answer lies in *inductive biases* — assumptions baked into the model architecture and training procedure that help it prefer simpler, more structured solutions.

**The No Free Lunch theorem** (Wolpert, 1997) states that no algorithm outperforms all others averaged over all possible problem distributions. Generalization only makes sense relative to a prior on the problem class. Neural networks are not magic; they generalize because the inductive biases of weight sharing (CNNs), sequential attention (Transformers), and gradient descent with small learning rate happen to align well with the structure of natural data (images, language, code).

**Occam's Razor in practice.** Regularization implements a soft preference for *simpler* models. The minimum description length (MDL) principle formalizes this: the best model is the one that minimizes total description length of both the model and the data given the model. L2 regularization corresponds to preferring models near the origin (short code) under a Gaussian prior.

**Implicit regularization by SGD.** Stochastic gradient descent, even without explicit regularization, biases toward flat minima (Keskar et al., 2017) and low-rank solutions (Li et al., 2018). The noise injected by mini-batches prevents convergence to sharp, narrow minima that generalize poorly. This is directly relevant to LLM training — the batch size, learning rate schedule, and optimizer choice all implicitly regularize the model. See [Optimizers: SGD, Adam, Adafactor, Lion, Muon & Shampoo](../03-pretraining/09-optimizers.html) for the optimizer perspective.

**Grokking.** A recently observed phenomenon (Power et al., 2022) where a model first memorizes training data (overfitting), then — after many more training steps — suddenly generalizes. This suggests that generalization can emerge from extended optimization, not just early stopping, and challenges the simple narrative that training longer always hurts.

---

## A Complete Training Pipeline in Code

The following implements a clean train/val/test workflow with L2 regularization, dropout, early stopping, and evaluation reporting. It is intentionally simple enough to read in one sitting.

```python
"""
A self-contained binary classification training pipeline demonstrating:
  - Proper train/val/test split
  - L2 weight decay + dropout regularization
  - Early stopping with best-checkpoint restoration
  - Precision, Recall, F1, ROC-AUC reporting
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import copy

# ─── 1. Dataset ───────────────────────────────────────────────────────────────
X, y = make_classification(
    n_samples=4000, n_features=30, n_informative=15,
    n_redundant=5, random_state=0
)

# Split: 70% train, 15% val, 15% test
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42)
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_train_raw, y_train, test_size=0.15/(1-0.15), stratify=y_train, random_state=42)

# ─── 2. Preprocessing — fit ONLY on training data ─────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)   # fit+transform
X_val   = scaler.transform(X_val_raw)          # transform only
X_test  = scaler.transform(X_test_raw)         # transform only

# Convert to torch tensors
def to_tensor(X, y):
    return TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )

train_loader = DataLoader(to_tensor(X_train, y_train), batch_size=64, shuffle=True)
val_loader   = DataLoader(to_tensor(X_val,   y_val),   batch_size=256)
test_loader  = DataLoader(to_tensor(X_test,  y_test),  batch_size=256)

# ─── 3. Model ──────────────────────────────────────────────────────────────────
class Classifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout_p: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout_p),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout_p),
            nn.Linear(hidden, 1)         # raw logit; sigmoid applied in loss
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)   # shape: (batch,)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = Classifier(X_train.shape[1]).to(device)

optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4   # L2 regularization via AdamW's decoupled weight decay
)
criterion = nn.BCEWithLogitsLoss()   # numerically stable sigmoid + BCE

# ─── 4. Early Stopping ─────────────────────────────────────────────────────────
patience, best_val_loss, wait = 10, float('inf'), 0
best_state = None

# ─── 5. Training loop ──────────────────────────────────────────────────────────
for epoch in range(100):
    # --- Train ---
    model.train()
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        optimizer.step()

    # --- Validate ---
    model.eval()
    val_losses = []
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            val_losses.append(criterion(model(X_b), y_b).item())
    val_loss = np.mean(val_losses)

    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        wait          = 0
        best_state    = copy.deepcopy(model.state_dict())
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stop at epoch {epoch+1}  (best val_loss={best_val_loss:.4f})")
            break

# ─── 6. Restore best and evaluate on test set ──────────────────────────────────
model.load_state_dict(best_state)
model.eval()

all_logits, all_labels = [], []
with torch.no_grad():
    for X_b, y_b in test_loader:
        logits = model(X_b.to(device)).cpu().numpy()
        all_logits.extend(logits)
        all_labels.extend(y_b.numpy())

y_score = torch.sigmoid(torch.tensor(all_logits)).numpy()
y_pred  = (y_score >= 0.5).astype(int)
y_true  = np.array(all_labels).astype(int)

print(classification_report(y_true, y_pred, digits=4))
print(f"ROC-AUC: {roc_auc_score(y_true, y_score):.4f}")
```

---

## Interview Corner

!!! interview "Interview Corner"
    **Q:** You train a model that achieves 99% training accuracy but only 70% validation accuracy on a balanced binary classification task. What is likely wrong, and what would you try?

    **A:** This is a textbook overfitting signature: the model has memorized training labels rather than learning generalizable patterns. Diagnostic steps:

    1. **Confirm there is no data leakage.** Did preprocessing (scaling, tokenization, feature engineering) use the full dataset instead of only training data? Are there duplicates between splits?
    2. **Examine the learning curves.** If train loss is near zero while val loss is increasing, the model is in the overfitting regime. If val loss is near train loss (both poor), it is underfitting.
    3. **Apply regularization.** Add/increase dropout, L2 weight decay, or label smoothing. For neural networks, `AdamW` with weight_decay between 1e-4 and 1e-2 is a good default.
    4. **Reduce model capacity.** Try fewer layers or narrower hidden dimensions.
    5. **Increase data.** More training data is the most reliable fix. Consider data augmentation if applicable.
    6. **Early stopping.** If you trained to convergence on the training loss, re-run and stop at the validation loss minimum.

    **Follow-up:** "What if validation accuracy is also 99%?" — Then check whether the test set is similarly easy or whether the task has been contaminated. Also sanity-check the metric: with a 99:1 class imbalance, 99% accuracy is trivially achieved by predicting all negatives.

---

## Key Takeaways

!!! key "Key Takeaways"
    - **Three paradigms:** Supervised (labeled targets), unsupervised (structure from $p(x)$), self-supervised (labels derived from input — the foundation of LLM pretraining).
    - **Bias-variance tradeoff:** Bias is systematic error from model mis-specification; variance is sensitivity to training noise. Both contribute to test error. Overly simple models underfit (high bias); overly complex ones overfit (high variance).
    - **Regularization toolkit:** L2 (weight decay) encourages small weights; L1 encourages sparsity; dropout trains implicit ensembles; early stopping uses validation loss as a stopping criterion. AdamW implements decoupled weight decay that is better than adding L2 to Adam naively.
    - **Data protocol hygiene:** Always fit preprocessing statistics on training data only. Never touch the test set until final evaluation. Use stratified splits for imbalanced classes.
    - **Data leakage** is the most common cause of over-optimistic reported results in production ML: duplicates across splits, future-leaking features, and test-set-contaminated preprocessing are the three main culprits.
    - **Metric choice matters:** Accuracy is misleading under class imbalance. Prefer F1 or PR-AUC for imbalanced classification. Use ROC-AUC for threshold-free ranking evaluation. For language models, use perplexity as the primary metric, but always validate on downstream task metrics too.
    - **Calibration is a separate property from discrimination.** A model can have high AUC but terrible calibration. ECE measures calibration; temperature scaling is the simplest fix.
    - **Generalization is not magic:** it comes from inductive biases that align with data structure. Implicit regularization from SGD noise, flat minima preference, and architectural choices (attention, convolutions) all contribute. There is no free lunch — knowing your problem distribution is the starting point.

---

!!! sota "State of the Art & Resources (2026)"
    Classical ML fundamentals — bias-variance tradeoff, regularization, evaluation metrics, and generalization theory — remain the bedrock of every modern system, including LLMs. The field has refined and extended these ideas through double-descent phenomena, improved calibration methods, and large-scale empirical studies of grokking, but the core principles are stable and well-covered by several excellent free resources.

    **Textbooks & courses**

    - [Goodfellow, Bengio & Courville, *Deep Learning* (2016)](https://www.deeplearningbook.org/) — Chapters 5–7 give the definitive treatment of ML basics, capacity, regularization, and optimization.
    - [Murphy, *Probabilistic Machine Learning: An Introduction* (2022)](https://probml.github.io/pml-book/book1.html) — Free MIT Press draft; covers bias-variance, regularization, and Bayesian perspectives with modern notation.
    - [Zhang et al., *Dive into Deep Learning* (d2l.ai)](https://d2l.ai/) — Runnable, multi-framework textbook with hands-on code for every core concept; adopted at 500+ universities.
    - [Stanford CS229 Machine Learning — course page](https://cs229.stanford.edu/) — Authoritative lecture notes on supervised learning, regularization, and learning theory.

    **Foundational papers**

    - [Belkin et al., *Reconciling modern machine learning practice and the classical bias–variance trade-off* (2019)](https://arxiv.org/abs/1812.11118) — Introduces the double-descent curve; shows why overparameterized models can still generalize.
    - [Guo et al., *On Calibration of Modern Neural Networks* (2017)](https://arxiv.org/abs/1706.04599) — Establishes that modern nets are overconfident; introduces temperature scaling and ECE.
    - [Keskar et al., *On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima* (2017)](https://arxiv.org/abs/1609.04836) — Explains why small-batch SGD generalizes better via flat minima.

    **Recent advances (2022–2026)**

    - [Power et al., *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets* (2022)](https://arxiv.org/abs/2201.02177) — Demonstrates late generalization; challenges the simple "training longer hurts" narrative.
    - [Nakkiran et al., *Deep Double Descent: Where Bigger Models and More Data Hurt* (2020)](https://arxiv.org/abs/1912.02292) — Shows double descent as a function of model size, dataset size, and training epochs; extends Belkin et al. to practice.

    **Visual explainers & tools**

    - [MLU-Explain: Bias-Variance Tradeoff](https://mlu-explain.github.io/bias-variance/) — Interactive visual essay from Amazon ML University; great for building intuition before the math.
    - [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html) — Canonical reference for cross-validation, metrics (ROC, PR-AUC, ECE), and regularized linear models with working Python code.

---

## Further Reading

- Goodfellow, Bengio & Courville, *Deep Learning* (2016) — Chapters 5–7 cover the statistical foundations, ML basics, and regularization with full mathematical derivations.
- Vapnik, *The Nature of Statistical Learning Theory* (1995) — the original framework for VC dimension, structural risk minimization, and support vector machines.
- Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks from Overfitting," *JMLR* 2014.
- Belkin et al., "Reconciling modern machine-learning practice and the classical bias–variance trade-off," *PNAS* 2019 — double-descent phenomenon.
- Power et al., "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets," *ICLR 2022 workshop* — late generalization in neural networks.
- Guo et al., "On Calibration of Modern Neural Networks," *ICML* 2017 — temperature scaling and ECE.
- Keskar et al., "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima," *ICLR* 2017 — why batch size affects generalization.
- Wolpert, "The Lack of A Priori Distinctions Between Learning Algorithms," *Neural Computation* 1996 — No Free Lunch theorems.
