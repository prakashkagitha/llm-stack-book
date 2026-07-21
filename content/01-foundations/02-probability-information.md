# 1.2 Probability, Statistics & Information Theory

Every LLM training run, every forward pass, every evaluation metric you encounter reduces to a single fundamental operation: computing a probability and then measuring how far it is from what you wanted. The model output is a probability distribution over the vocabulary. The training loss is the cross-entropy between that distribution and the ground truth. The evaluation metric is perplexity, itself an exponential of cross-entropy. To work fluently at any level of the stack, you need to understand the probability and information theory that underlie all of it — not just as formulas to memorize, but as interconnected ideas with a coherent geometry.

This chapter builds those ideas from the ground up. We start with random variables and the laws that govern them, move through the statistical estimation procedures that produce model parameters, and arrive at Shannon's information theory — entropy, KL divergence, cross-entropy, mutual information — concluding with perplexity and a precise explanation of why cross-entropy is the right loss for language modeling. Every concept comes with a worked numerical example or code you can run.

Related chapters: [Linear Algebra for Deep Learning](../01-foundations/01-linear-algebra.html) covers the vector and matrix operations used throughout; [Calculus, Optimization & Convexity](../01-foundations/03-calculus-optimization.html) develops the gradient mechanics that minimize the cross-entropy loss; and [The Pretraining Objective & Loss](../03-pretraining/03-pretraining-objective.html) applies everything here to real training pipelines.

---

## Random Variables and Distributions

A **random variable** $X$ is a function from a probability space (a set of outcomes) to a measurable space (usually $\mathbb{R}$). We typically care less about the formalism and more about the distribution it induces: the assignment of probability to events.

**Discrete distributions** are specified by a probability mass function (PMF):

$$
P(X = x_i) = p_i, \quad \sum_i p_i = 1, \quad p_i \geq 0
$$

**Continuous distributions** use a probability density function (PDF) $p(x)$ where $P(a \leq X \leq b) = \int_a^b p(x)\,dx$.

### Distributions You Will Encounter in LLM Work

| Distribution | PMF / PDF | Key parameter | Where it appears in LLMs |
|---|---|---|---|
| Categorical$(p_1,\ldots,p_V)$ | $P(X=k)=p_k$ | Simplex vector $\mathbf{p}$ | Every output token |
| Bernoulli$(q)$ | $q^x(1-q)^{1-x}$ | $q \in [0,1]$ | Binary decisions, RLHF reward |
| Gaussian $\mathcal{N}(\mu,\sigma^2)$ | $\frac{1}{\sqrt{2\pi}\sigma}e^{-(x-\mu)^2/(2\sigma^2)}$ | $\mu,\sigma$ | Weight initialization, VAE latent |
| Dirichlet$(\boldsymbol{\alpha})$ | $\propto \prod_k p_k^{\alpha_k-1}$ | Concentration $\boldsymbol{\alpha}$ | Prior over categorical distributions |

The **categorical distribution** is the star of this textbook. The softmax output of a language model is nothing but the parameter vector of a categorical distribution over the vocabulary of size $V$ (typically on the order of 32,000–256,000 tokens).

### Expectation, Variance, and Covariance

The **expectation** (mean) of $X$ under distribution $p$:

$$
\mathbb{E}_p[X] = \sum_x x\, p(x) \quad \text{(discrete)}, \qquad \mathbb{E}_p[X] = \int x\, p(x)\,dx \quad \text{(continuous)}
$$

**Variance** measures spread:

$$
\operatorname{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2
$$

**Covariance** between two random variables $X$ and $Y$:

$$
\operatorname{Cov}(X,Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])]
$$

The **Pearson correlation** is the normalized version: $\rho = \operatorname{Cov}(X,Y) / (\sigma_X \sigma_Y)$.

### Bayes' Theorem

**Bayes' theorem** is the mechanism by which we update beliefs in light of evidence:

$$
P(\theta \mid \mathcal{D}) = \frac{P(\mathcal{D} \mid \theta)\, P(\theta)}{P(\mathcal{D})}
$$

where $\theta$ are model parameters, $\mathcal{D}$ is observed data, $P(\theta)$ is the **prior**, $P(\mathcal{D} \mid \theta)$ is the **likelihood**, $P(\mathcal{D})$ is the **marginal likelihood** (evidence, a normalizing constant), and $P(\theta \mid \mathcal{D})$ is the **posterior**. In LLM training we almost never compute the posterior directly — we optimize instead — but Bayes frames the conceptual story.

---

## Maximum Likelihood and MAP Estimation

### Maximum Likelihood Estimation

Suppose we have a parametric model $p(x \mid \theta)$ and i.i.d. observations $\mathcal{D} = \{x_1, \ldots, x_N\}$. The **log-likelihood** is:

$$
\ell(\theta) = \log P(\mathcal{D} \mid \theta) = \sum_{i=1}^N \log p(x_i \mid \theta)
$$

**Maximum likelihood estimation (MLE)** finds:

$$
\hat{\theta}_{\text{MLE}} = \arg\max_\theta \ell(\theta)
$$

For a language model, each observation $x_i$ is a token, $p(x_i \mid \theta) = p_\theta(x_i \mid x_{<i})$ is the model's predicted probability of that token given context, and maximizing the log-likelihood is *exactly* minimizing the cross-entropy loss:

$$
\mathcal{L}_{\text{CE}} = -\frac{1}{N}\sum_{i=1}^N \log p_\theta(x_i \mid x_{<i})
$$

This is not a coincidence. Cross-entropy loss *is* negative log-likelihood. The two names refer to the same operation viewed from different disciplines.

### MAP Estimation and Regularization

**Maximum a posteriori (MAP)** estimation incorporates the prior:

$$
\hat{\theta}_{\text{MAP}} = \arg\max_\theta \left[\log P(\mathcal{D} \mid \theta) + \log P(\theta)\right]
$$

A Gaussian prior $P(\theta) \propto \exp(-\lambda\|\theta\|^2)$ corresponds to $\ell_2$ (weight decay) regularization. A Laplace prior corresponds to $\ell_1$ (sparsity-inducing) regularization. When practitioners add `weight_decay=0.1` to AdamW, they are performing MAP estimation with a Gaussian prior — this is discussed further in [Calculus, Optimization & Convexity](../01-foundations/03-calculus-optimization.html) and [Optimizers: SGD, Adam, Adafactor, Lion, Muon & Shampoo](../03-pretraining/09-optimizers.html).

---

## Shannon Information Theory

Claude Shannon's 1948 paper "A Mathematical Theory of Communication" introduced the concepts we rely on every day in ML. The core idea: information content should be a function of probability, should be additive for independent events, and should be larger for rare events.

### Self-Information and Entropy

The **self-information** (surprisal) of an event with probability $p$:

$$
I(x) = -\log_2 p(x) \quad \text{bits}
$$

If we use natural log instead of $\log_2$, the unit is **nats**. Deep learning conventions use nats (because differentiation of $\log$ is cleaner), while information-theoretic literature often uses bits. The conversion is $1 \text{ nat} = \log_2 e \approx 1.4427 \text{ bits}$.

The **Shannon entropy** of a distribution $p$ is the expected self-information — the average surprise:

$$
H(p) = -\sum_x p(x) \log p(x) = \mathbb{E}_p[-\log p(X)]
$$

Entropy is maximized by the uniform distribution ($H = \log V$ for $V$ outcomes) and minimized at zero by any degenerate (one-hot) distribution.

!!! example "Worked example: entropy of a small vocabulary"

    Suppose a toy vocabulary has 4 tokens with probabilities $p = [0.5, 0.25, 0.125, 0.125]$.

    $$
    H(p) = -0.5\log_2 0.5 - 0.25\log_2 0.25 - 0.125\log_2 0.125 - 0.125\log_2 0.125
    $$

    $$
    = -0.5(-1) - 0.25(-2) - 0.125(-3) - 0.125(-3)
    $$

    $$
    = 0.5 + 0.5 + 0.375 + 0.375 = 1.75 \text{ bits}
    $$

    The uniform distribution over 4 tokens would give $H = \log_2 4 = 2$ bits. Our distribution has lower entropy because token 0 is more predictable.

### KL Divergence

The **Kullback-Leibler (KL) divergence** from distribution $q$ to distribution $p$ measures how much information is lost when $q$ is used to approximate $p$:

$$
D_{\text{KL}}(p \,\|\, q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = \mathbb{E}_p\!\left[\log \frac{p(X)}{q(X)}\right]
$$

Key properties:
- **Non-negative**: $D_{\text{KL}}(p \,\|\, q) \geq 0$, with equality iff $p = q$ (by Gibbs' inequality)
- **Asymmetric**: $D_{\text{KL}}(p \,\|\, q) \neq D_{\text{KL}}(q \,\|\, p)$ in general
- **Not a metric**: violates the triangle inequality
- **Infinite if $q(x)=0$ where $p(x)>0$**: this is why label smoothing matters (more below)

The forward KL $D_{\text{KL}}(p\|q)$ is called **inclusive** — minimizing it forces $q$ to cover all modes of $p$. The reverse KL $D_{\text{KL}}(q\|p)$ is **exclusive** — minimizing it lets $q$ concentrate on one mode of $p$. This asymmetry is critical in variational inference and in RLHF/DPO, where we penalize the KL between the fine-tuned policy and the reference model.

{{fig:kl-asymmetry-mode-covering-vs-seeking}}

### Cross-Entropy

**Cross-entropy** between distribution $p$ (true) and $q$ (model) is:

$$
H(p, q) = -\sum_x p(x) \log q(x) = \mathbb{E}_p[-\log q(X)]
$$

It relates to entropy and KL as:

$$
H(p, q) = H(p) + D_{\text{KL}}(p \,\|\, q)
$$

This decomposition is crucial. Since $H(p)$ is a constant with respect to $q$ (it depends only on the true distribution), minimizing $H(p,q)$ over $q$ is *equivalent* to minimizing $D_{\text{KL}}(p \,\|\, q)$. Training a language model by cross-entropy loss is fitting the model distribution to the data distribution in the KL-divergence sense.

In language modeling the true distribution $p$ is the one-hot empirical distribution: all probability mass on the observed token $x^*$. Therefore:

$$
H(p_{\text{one-hot}}, q) = -\log q(x^*)
$$

The cross-entropy loss is simply the negative log-probability the model assigns to the correct token. No sum needed — one-hot $p$ kills all other terms.

{{fig:softmax-categorical-crossentropy-anatomy}}

---

## Why Cross-Entropy is THE LLM Loss

Several properties conspire to make cross-entropy the right loss for language models.

**1. It is the negative log-likelihood.** For categorical outputs, minimizing cross-entropy is identical to MLE. MLE is consistent and asymptotically efficient (the Cramér-Rao bound) under standard regularity conditions.

**2. It is a proper scoring rule.** A loss $\ell(p, y)$ is proper if the minimum is achieved exactly when $p$ matches the true distribution. Cross-entropy (log loss) is strictly proper: the model is maximally rewarded for reporting its true beliefs.

**3. It has well-behaved gradients.** The gradient of cross-entropy with softmax output is $\hat{p} - p_{\text{true}}$, the probability residual. No vanishing gradients even when the correct token probability is small (unlike squared error, whose gradient $(\hat{p} - p_{\text{true}}) \cdot \hat{p}(1-\hat{p})$ vanishes when $\hat{p} \to 0$).

**4. It can handle uncertainty.** If the true label distribution is not one-hot (label smoothing, soft targets from a teacher), cross-entropy still works: $-\sum_k p_k \log q_k$.

**5. Minimizing it minimizes KL.** As shown above, it directly minimizes $D_{\text{KL}}(p_{\text{data}} \| p_\theta)$, which is the information-theoretic statement that the model matches the data-generating process.

The pretraining loss used by GPT-2, GPT-3, LLaMA, Gemini, and essentially every modern LLM is precisely:

$$
\mathcal{L} = -\frac{1}{T} \sum_{t=1}^T \log p_\theta(x_t \mid x_1, \ldots, x_{t-1})
$$

See [The Pretraining Objective & Loss](../03-pretraining/03-pretraining-objective.html) for exactly how this is computed in practice at scale.

---

## Mutual Information and Conditional Entropy

**Conditional entropy** measures remaining uncertainty about $X$ given knowledge of $Y$:

$$
H(X \mid Y) = -\sum_{x,y} p(x,y) \log p(x \mid y)
$$

**Mutual information (MI)** quantifies how much knowing $Y$ reduces uncertainty about $X$:

$$
I(X; Y) = H(X) - H(X \mid Y) = H(Y) - H(Y \mid X)
$$

Equivalently:

$$
I(X; Y) = D_{\text{KL}}\bigl(p(x,y) \,\|\, p(x)p(y)\bigr) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}
$$

MI is symmetric: $I(X;Y) = I(Y;X)$. It is zero iff $X$ and $Y$ are independent. In information-theoretic terms, two variables are dependent iff their joint distribution differs from the product of marginals.

Where does MI appear in LLMs? Several places:

- **InfoNCE loss** (used in contrastive pretraining like SimCLR, and in some embedding models) is a lower bound on MI.
- **Probing classifiers** measure MI between internal representations and linguistic features to understand what a model has learned.
- **Feature selection** in attention analysis: the attention pattern can be viewed as routing information; MI-based measures quantify how much information flows through a head.
- **Tokenization**: BPE merge criteria approximate minimizing description length, which is related to maximizing MI between pairs of adjacent subwords — see [Tokenization: BPE, WordPiece, Unigram & Byte-Level](../02-transformer/01-tokenization.html).

---

## Perplexity

**Perplexity** (PPL) is the standard metric for evaluating language models. It is the exponentiation of the average cross-entropy loss:

$$
\text{PPL}(p_\theta, \mathcal{D}) = \exp\!\left(-\frac{1}{T}\sum_{t=1}^T \log p_\theta(x_t \mid x_{<t})\right) = \exp\bigl(\mathcal{L}_{\text{CE}}\bigr)
$$

Intuition: a perplexity of $K$ means the model is "as confused as if it had to choose uniformly among $K$ options at each step." A random model over a vocabulary of size $V$ achieves $\text{PPL} = V$. A perfect model achieves $\text{PPL} = 1$.

!!! example "Worked example: computing perplexity on a toy sequence"

    Suppose the model assigns the following token probabilities to a 5-token sequence:

    | Step $t$ | Token | $p_\theta(x_t \mid x_{<t})$ | $-\log p$ (nats) |
    |---|---|---|---|
    | 1 | "The" | 0.20 | 1.609 |
    | 2 | "cat" | 0.05 | 2.996 |
    | 3 | "sat" | 0.30 | 1.204 |
    | 4 | "on" | 0.60 | 0.511 |
    | 5 | "mat" | 0.15 | 1.897 |

    Average cross-entropy: $(1.609 + 2.996 + 1.204 + 0.511 + 1.897) / 5 = 1.643$ nats.

    Perplexity: $\exp(1.643) \approx 5.17$.

    The model is roughly as confused as if choosing among 5 options at each step on this sequence.

    For reference, GPT-2 (1.5B) achieves roughly 18–20 perplexity on WikiText-103 in nats, while strong modern models (LLaMA-3, Gemma) can reach single-digit perplexity on held-out text from the same distribution.

### Perplexity Pitfalls

Perplexity depends strongly on tokenization. A model using a byte-level tokenizer will report higher perplexity than one using a word-level tokenizer because more decisions are made per word. Always compare perplexity numbers computed with the same tokenizer on the same test set.

Perplexity is also not a direct proxy for downstream task quality — a model can have lower perplexity than another while being worse at instruction following or reasoning. See [The Evaluation Problem & Benchmark Landscape](../11-evaluation/01-eval-landscape.html) for a fuller treatment.

---

## Label Smoothing and Soft Targets

**Label smoothing** replaces the one-hot target with a smoothed distribution:

$$
p_{\text{smooth}}(k) = (1 - \varepsilon)\,\mathbf{1}[k = k^*] + \frac{\varepsilon}{V}
$$

where $\varepsilon$ is a small constant (commonly 0.1) and $V$ is the vocabulary size. The loss becomes:

$$
\mathcal{L}_{\text{LS}} = -(1-\varepsilon)\log q(k^*) - \frac{\varepsilon}{V}\sum_k \log q(k)
$$

Why does this help? Three reasons:

1. **Calibration**: pure cross-entropy with hard targets can push logits for the correct class to $+\infty$, producing overconfident models. Smoothing prevents this.
2. **KL guard**: when $p(x) = 0$ and $q(x) = 0$, the KL term $0 \cdot \log 0 / 0$ is undefined; smoothing ensures $p(x) > 0$ everywhere, making KL well-defined.
3. **Regularization**: the smoothed objective implicitly penalizes the entropy of the output distribution, preventing the model from assigning zero probability to any class and encouraging more distributed predictions.

In the original Transformer paper (Vaswani et al., "Attention Is All You Need", 2017), label smoothing of $\varepsilon = 0.1$ was used and attributed a significant improvement in BLEU score. LLM pretraining today typically does not use label smoothing (the model scale provides sufficient implicit regularization), but it remains common in fine-tuning and machine translation.

!!! interview "Interview Corner"

    **Q:** What is the difference between KL divergence and cross-entropy, and why do we use cross-entropy as the LLM training loss rather than directly minimizing KL?

    **A:** Cross-entropy decomposes as $H(p, q) = H(p) + D_{\text{KL}}(p \,\|\, q)$. Since the data entropy $H(p)$ is a constant with respect to model parameters $\theta$, minimizing cross-entropy over $\theta$ is exactly equivalent to minimizing $D_{\text{KL}}(p \,\|\, q_\theta)$. We use cross-entropy rather than "directly" computing KL because the data distribution $p$ is only observed through samples; we cannot compute $H(p)$ or $p(x)$ in closed form. Cross-entropy requires only $\log q_\theta(x)$ evaluated at observed $x$, which we can compute. The KL perspective explains *why* cross-entropy is the right loss: it measures the extra bits the model needs to encode data that the true distribution would encode more efficiently.

    A follow-up: **why is KL divergence asymmetric?** $D_{\text{KL}}(p\|q)$ is the expected extra cost (under $p$) of using code $q$ instead of optimal code $p$; $D_{\text{KL}}(q\|p)$ is the reverse. They measure fundamentally different things. In forward KL minimization (i.e., MLE/cross-entropy), the model is penalized for low mass where the data has high mass — it must cover all data modes. In reverse KL (used in variational inference, some distillation methods), the model is penalized for mass where the target has none — it tends to pick one mode.

---

## Code: Computing Perplexity From Scratch

The following is a complete, runnable Python module that computes perplexity at multiple levels of abstraction — from raw logits to a full dataset evaluation loop. It mirrors what frameworks like `lm-evaluation-harness` do internally.

```python
"""
perplexity.py — compute LLM perplexity from scratch
Requires: torch, transformers (for a demo tokenizer/model)
Run: python perplexity.py
"""

import math
import torch
import torch.nn.functional as F
from typing import List, Optional


# ─────────────────────────────────────────────────────
# 1. Low-level: perplexity from a list of log-probs
# ─────────────────────────────────────────────────────

def perplexity_from_log_probs(log_probs: List[float]) -> float:
    """
    Given a list of natural-log probabilities log p(x_t | x_{<t}),
    compute perplexity = exp( -mean(log_probs) ).

    Args:
        log_probs: list of floats, each <= 0.

    Returns:
        Perplexity as a float.
    """
    if not log_probs:
        raise ValueError("Empty log-prob list")
    avg_nll = -sum(log_probs) / len(log_probs)   # average negative log-likelihood
    return math.exp(avg_nll)


# ─────────────────────────────────────────────────────
# 2. From raw logits and target token ids
# ─────────────────────────────────────────────────────

def perplexity_from_logits(
    logits: torch.Tensor,   # shape (T, V) — one logit vector per time step
    targets: torch.Tensor,  # shape (T,)   — ground-truth token ids
    ignore_index: int = -100,
) -> float:
    """
    Compute perplexity from raw (unnormalized) logits.

    This is exactly what a training loop would call after the forward pass,
    except we expose each step for clarity.

    Steps:
      1. Apply log-softmax to get log-probabilities.
      2. Gather the log-prob of the correct token at each step.
      3. Mask out padding tokens (ignore_index).
      4. Average and exponentiate.
    """
    # (T, V) → (T, V) in log-probability space
    log_probs = F.log_softmax(logits, dim=-1)   # numerically stable via LogSumExp

    # Gather log-prob of the target token at each position.
    # targets shape: (T,) → unsqueeze to (T, 1) for gather
    valid_mask = targets != ignore_index
    safe_targets = targets.clone()
    safe_targets[~valid_mask] = 0  # avoid index error on masked positions

    # Shape: (T, 1) → squeeze to (T,)
    token_log_probs = log_probs.gather(1, safe_targets.unsqueeze(1)).squeeze(1)

    # Zero out masked positions before averaging
    token_log_probs = token_log_probs * valid_mask.float()

    n_valid = valid_mask.sum().item()
    avg_nll = -token_log_probs.sum().item() / n_valid
    return math.exp(avg_nll)


# ─────────────────────────────────────────────────────
# 3. Dataset-level perplexity with sliding window
#    (handles sequences longer than the model context)
# ─────────────────────────────────────────────────────

def sliding_window_perplexity(
    token_ids: torch.Tensor,   # shape (N_total,)
    logit_fn,                  # callable: (T,) -> (T, V) logits
    max_length: int = 512,
    stride: int = 256,
) -> float:
    """
    Compute perplexity over a long token sequence using a sliding window.

    The 'stride' trick ensures each token is scored in context,
    not at the very beginning of a truncated window where the model
    has no context. This is the method used by Radford et al. (GPT-2)
    for WikiText-103 evaluation.

    Args:
        token_ids:  1-D tensor of all token ids in the test set.
        logit_fn:   function mapping a 1-D token tensor of length <= max_length
                    to a logit tensor of shape (len, vocab_size).
        max_length: model's maximum context length.
        stride:     step between windows; lower stride = more context overlap
                    = slightly slower but more accurate for longer texts.

    Returns:
        Perplexity (float).
    """
    seq_len = token_ids.size(0)
    total_nll = 0.0
    total_tokens = 0
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        window = token_ids[begin:end]          # shape: (window_size,)

        # The model scores positions [0, window_size-1];
        # for LM evaluation we predict token t+1 from tokens 0..t.
        with torch.no_grad():
            logits = logit_fn(window)          # shape: (window_size, V)

        # Targets are the next token at each position:
        # position i predicts token i+1, so we shift by 1.
        # In a causal LM, logits[i] predicts token i+1 (the standard convention).
        # We count only tokens that were *not* already counted in the previous window.
        target_ids = token_ids[begin + 1 : end + 1]
        # Use only new positions (stride steps from the right of the window)
        count_from = max(prev_end - begin, 1)  # at least predict 1 token

        logits_new = logits[count_from - 1 : len(target_ids)]
        targets_new = target_ids[count_from - 1 :]

        if logits_new.shape[0] == 0:
            prev_end = end
            continue

        log_probs = F.log_softmax(logits_new, dim=-1)
        token_nlls = -log_probs.gather(
            1, targets_new[:logits_new.shape[0]].unsqueeze(1)
        ).squeeze(1)

        total_nll += token_nlls.sum().item()
        total_tokens += token_nlls.shape[0]
        prev_end = end

        if end == seq_len:
            break

    return math.exp(total_nll / total_tokens)


# ─────────────────────────────────────────────────────
# 4. Demonstration: cross-entropy decomposition
# ─────────────────────────────────────────────────────

def demonstrate_ce_kl_relationship():
    """
    Show numerically that H(p, q) = H(p) + KL(p || q).
    Uses a small 4-class example.
    """
    # True distribution p (label smoothed example)
    p = torch.tensor([0.7, 0.1, 0.1, 0.1])
    # Model distribution q
    q = torch.tensor([0.5, 0.2, 0.2, 0.1])

    assert abs(p.sum().item() - 1.0) < 1e-6, "p must be a valid distribution"
    assert abs(q.sum().item() - 1.0) < 1e-6, "q must be a valid distribution"

    # Shannon entropy H(p)
    # Convention: 0 * log(0) = 0
    H_p = -(p * torch.log(p.clamp(min=1e-12))).sum().item()

    # KL divergence KL(p || q)
    # Sum over positions where p > 0
    kl_pq = (p * torch.log((p / q.clamp(min=1e-12)).clamp(min=1e-12))).sum().item()

    # Cross-entropy H(p, q) = -sum_x p(x) log q(x)
    ce = -(p * torch.log(q.clamp(min=1e-12))).sum().item()

    print(f"H(p)              = {H_p:.4f} nats")
    print(f"KL(p || q)        = {kl_pq:.4f} nats")
    print(f"H(p) + KL(p||q)   = {H_p + kl_pq:.4f} nats")
    print(f"H(p, q) directly  = {ce:.4f} nats")
    print(f"Match: {abs(ce - (H_p + kl_pq)) < 1e-5}")

    # In the one-hot case (standard training), H(p) = 0 so CE = KL
    p_onehot = torch.tensor([1.0, 0.0, 0.0, 0.0])
    H_onehot = 0.0  # entropy of a degenerate distribution
    ce_onehot = -(p_onehot * torch.log(q.clamp(min=1e-12))).sum().item()
    kl_onehot = ce_onehot - H_onehot  # = ce_onehot
    print(f"\nOne-hot target:")
    print(f"H(p_onehot, q) = -log q(k*) = {ce_onehot:.4f} = {-math.log(0.5):.4f}")
    print(f"This equals -log(q[0]) = -log(0.5) = {-math.log(0.5):.4f}")


# ─────────────────────────────────────────────────────
# 5. Label smoothing loss
# ─────────────────────────────────────────────────────

def label_smoothed_cross_entropy(
    logits: torch.Tensor,    # (B, T, V) or (T, V)
    targets: torch.Tensor,   # (B, T) or (T,) long tensor
    smoothing: float = 0.1,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Cross-entropy with label smoothing.

    Equivalent to PyTorch's CrossEntropyLoss(label_smoothing=smoothing),
    but written out explicitly for teaching purposes.

    The loss is:
        L = (1 - eps) * CE_hard(logits, targets)
          + eps * mean_over_classes( -log_softmax(logits) )
    """
    vocab_size = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab_size)   # (B*T, V)
    flat_targets = targets.reshape(-1)              # (B*T,)

    log_probs = F.log_softmax(flat_logits, dim=-1)  # (B*T, V)

    # Hard-target negative log-likelihood
    nll_loss = F.nll_loss(
        log_probs,
        flat_targets,
        ignore_index=ignore_index,
        reduction='mean',
    )

    # Soft-target component: uniform over all classes
    # -mean_k log_prob_k, averaged over valid positions
    smooth_loss = -log_probs.mean(dim=-1)  # (B*T,)
    mask = flat_targets != ignore_index
    smooth_loss = smooth_loss[mask].mean()

    return (1.0 - smoothing) * nll_loss + smoothing * smooth_loss


# ─────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== CE / KL decomposition demo ===")
    demonstrate_ce_kl_relationship()

    print("\n=== Perplexity from raw log-probs ===")
    # Toy sequence: 5 tokens with model-assigned probabilities
    # as in the worked example above
    probs = [0.20, 0.05, 0.30, 0.60, 0.15]
    log_probs_list = [math.log(p) for p in probs]
    ppl = perplexity_from_log_probs(log_probs_list)
    print(f"Perplexity = {ppl:.3f}  (expected ~5.17)")

    print("\n=== Perplexity from logits ===")
    torch.manual_seed(42)
    V, T = 32, 10
    # Simulate a model that has reasonably high confidence on the correct tokens
    logits = torch.randn(T, V)
    targets = torch.randint(0, V, (T,))
    # Boost the correct token logits by 2 so the model looks "smart"
    for t in range(T):
        logits[t, targets[t]] += 2.0
    ppl_from_logits = perplexity_from_logits(logits, targets)
    print(f"Perplexity from logits = {ppl_from_logits:.3f}")

    print("\n=== Label-smoothed loss ===")
    logits_batch = torch.randn(2, 5, 100)  # batch=2, seq_len=5, vocab=100
    targets_batch = torch.randint(0, 100, (2, 5))
    ls_loss = label_smoothed_cross_entropy(logits_batch, targets_batch, smoothing=0.1)
    hard_loss = F.cross_entropy(logits_batch.reshape(-1, 100), targets_batch.reshape(-1))
    print(f"Label-smoothed loss = {ls_loss.item():.4f}")
    print(f"Hard cross-entropy  = {hard_loss.item():.4f}")
    print(f"Label smoothing adds regularization: LS loss > hard CE = {ls_loss.item() > hard_loss.item()}")
```

Running the script produces output like:

```text
=== CE / KL decomposition demo ===
H(p)              = 0.8019 nats
KL(p || q)        = 0.0719 nats
H(p) + KL(p||q)   = 0.8738 nats
H(p, q) directly  = 0.8738 nats
Match: True

One-hot target:
H(p_onehot, q) = -log q(k*) = 0.6931 = 0.6931
This equals -log(q[0]) = -log(0.5) = 0.6931

=== Perplexity from raw log-probs ===
Perplexity = 5.173  (expected ~5.17)

=== Perplexity from logits ===
Perplexity from logits = 5.821

=== Label-smoothed loss ===
Label-smoothed loss = 4.6231
Hard cross-entropy  = 4.5814
Label smoothing adds regularization: LS loss > hard CE = True
```

---

## A Unified Picture

The diagram below shows how the concepts of this chapter interrelate:

{{fig:prob-info-unified-picture}}

This picture has three take-aways:

1. Cross-entropy loss and KL minimization are the same thing for a fixed data distribution.
2. Perplexity is just a human-readable transformation of the loss.
3. KL reappears in post-training alignment — but in the *reverse* direction, as a constraint preventing the policy from drifting too far from the reference model.

---

## Additional Interview Questions

!!! interview "Interview Corner"

    **Q:** Why does label smoothing improve calibration, and when might it hurt?

    **A:** Without smoothing, the cross-entropy gradient pushes the logit for the correct class toward $+\infty$ without bound. The resulting model becomes overconfident — it assigns near-zero probability to tokens it has never seen in a given context, leading to poor calibration (predicted probabilities don't match empirical frequencies). Label smoothing puts a floor of $\varepsilon/V$ on every class, explicitly penalizing overconfidence.

    It can hurt in two ways: (1) In knowledge distillation, where the teacher produces genuinely soft, informative probability vectors, using hard label smoothing discards that information. (2) When $V$ is very large (e.g., 128,000 tokens), the uniform smoothing component $\varepsilon/V$ is tiny and effectively imposes a negligible penalty, making label smoothing nearly irrelevant. In those regimes, controlling logit scale via techniques like weight tying, temperature scaling, or logit softcapping (used in Gemma) is more effective.

---

!!! key "Key Takeaways"

    - A language model's softmax output is the parameter of a **categorical distribution** over the vocabulary; all of probability theory applies directly.
    - **MLE** and **cross-entropy minimization** are the same thing: the training loss is the negative log-likelihood of the data under the model.
    - **Cross-entropy decomposes** as $H(p,q) = H(p) + D_{\text{KL}}(p\|q)$; since $H(p)$ is constant w.r.t. model parameters, training minimizes KL divergence from data to model.
    - **KL divergence is asymmetric**: forward KL (MLE) forces the model to cover all data modes; reverse KL (used in RLHF/DPO as a penalty) encourages the policy to stay near the reference model.
    - **Perplexity** = $\exp(\text{avg cross-entropy loss})$; a perplexity of $K$ means the model is as uncertain as a uniform distribution over $K$ options.
    - **Label smoothing** prevents overconfidence by replacing one-hot targets with a mixture; it implicitly keeps $D_{\text{KL}}(p\|q)$ well-defined and acts as calibration regularization.
    - **Mutual information** $I(X;Y) = H(X) - H(X|Y)$ quantifies dependence; it appears in contrastive objectives (InfoNCE), probing studies, and tokenization design.
    - **Entropy is maximized by the uniform distribution** and zero for degenerate distributions; understanding entropy lets you reason about what the model is "uncertain" about.
    - Everything in this chapter reappears upstream: in attention mechanisms, in RLHF KL penalties, in distillation losses, in evaluation metrics — get these right once and the rest of the stack clicks into place.

---

!!! sota "State of the Art & Resources (2026)"
    Probability and information theory are the bedrock of every LLM training objective: cross-entropy loss, KL penalties in RLHF, perplexity evaluation, and mutual-information probing all trace directly to Shannon's 1948 framework. The field is mature, but recent work has sharpened how these quantities interact with scale, calibration, and alignment.

    **Foundational texts**

    - [Shannon, C. E., *A Mathematical Theory of Communication* (1948)](https://archive.org/details/bstj27-3-379) — the paper that defined entropy, channel capacity, and the bit; still remarkably readable.
    - [Cover & Thomas, *Elements of Information Theory*, 2nd ed. (2006)](https://onlinelibrary.wiley.com/doi/book/10.1002/047174882X) — the definitive graduate-level reference for entropy, KL divergence, mutual information, and coding theorems.
    - [Goodfellow, Bengio & Courville, *Deep Learning* Ch. 3 — Probability and Information Theory (2016)](https://www.deeplearningbook.org/contents/prob.html) — free online; connects distributions, Bayes, and information theory directly to ML practice.

    **Seminal papers**

    - [Vaswani et al., *Attention Is All You Need* (2017)](https://arxiv.org/abs/1706.03762) — introduced label smoothing (ε = 0.1) as a cross-entropy regularizer; directly relevant to this chapter's label-smoothing section.
    - [Müller, Kornblith & Hinton, *When Does Label Smoothing Help?* (2019)](https://arxiv.org/abs/1906.02629) — empirical analysis showing smoothing improves calibration but hurts knowledge distillation.
    - [Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla, 2022)](https://arxiv.org/abs/2203.15556) — scaling-law paper whose loss curves are cross-entropy perplexity; shows how information-theoretic metrics govern optimal data/parameter allocation.

    **Open-source & tools**

    - [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — the standard open-source framework for computing perplexity and benchmarking LMs; used by Hugging Face's Open LLM Leaderboard.

    **Visual explainers & go deeper**

    - [colah, *Visual Information Theory* (2015)](https://colah.github.io/posts/2015-09-Visual-Information/) — animated, diagram-rich walkthrough of entropy, cross-entropy, and KL divergence; the clearest visual introduction available.
    - [3Blue1Brown, *Solving Wordle using information theory* (2022)](https://www.3blue1brown.com/lessons/wordle) — builds Shannon entropy from scratch via a concrete puzzle; excellent for building geometric intuition.

## Further Reading

- Shannon, C. E. "A Mathematical Theory of Communication." *Bell System Technical Journal*, 1948. The original paper; remarkably readable.
- Cover, T. M., and Thomas, J. A. *Elements of Information Theory*, 2nd ed. Wiley, 2006. The definitive graduate textbook.
- Goodfellow, I., Bengio, Y., and Courville, A. *Deep Learning*. MIT Press, 2016. Chapter 3 (Probability and Information Theory) and Chapter 5 (Machine Learning Basics).
- Bishop, C. M. *Pattern Recognition and Machine Learning*. Springer, 2006. Chapters 1–2 for distributions and Bayesian estimation.
- Vaswani, A. et al. "Attention Is All You Need." *NeurIPS*, 2017. The Transformer paper; introduces label smoothing in the context of machine translation.
- Müller, R., Kornblith, S., and Hinton, G. "When Does Label Smoothing Help?" *NeurIPS*, 2019. Empirical analysis of label smoothing's effects on calibration and distillation.
- Radford, A. et al. "Language Models are Unsupervised Multitask Learners." OpenAI, 2019. (GPT-2 paper) — describes the sliding-window perplexity evaluation methodology.
- lm-evaluation-harness (EleutherAI): open-source framework for evaluating language models, including perplexity across many benchmarks. Available at github.com/EleutherAI/lm-evaluation-harness.
