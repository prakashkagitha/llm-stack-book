# 10.4 Diffusion Models & Generative Modeling (Breadth)

Generative modeling is the art of learning a distribution $p_\theta(\mathbf{x})$ from data and then sampling from it. For a decade, the dominant paradigms were variational autoencoders (VAEs), generative adversarial networks (GANs), and autoregressive models. Diffusion models — arriving in earnest around 2020 with Ho et al.'s DDPM — quietly displaced GANs as the state of the art for image synthesis, and today they underpin nearly every major text-to-image and text-to-video system (Stable Diffusion, Flux, Imagen, Sora). As an LLM engineer you will encounter diffusion-based image encoders as vision backbone components (see [Vision Transformers & Image Encoders](../10-multimodal-and-arch/01-vision-transformers.html)), diffusion-based image/video generation coupled to language models (see [Vision-Language Models](../10-multimodal-and-arch/02-vision-language-models.html)), and even diffusion applied to *text tokens* as a competitor to autoregressive LLMs. This chapter gives you the conceptual and mathematical toolkit to reason about all of these.

## Why Diffusion? Motivation and Context

Before diffusion, GANs produced sharp images but were notoriously difficult to train (mode collapse, training instability) and did not yield a tractable likelihood. Normalizing flows gave exact likelihoods but required architectures with constrained Jacobians. VAEs optimized a lower bound on likelihood and tended to produce blurry samples. Autoregressive models like PixelCNN and later GPT-style models over image tokens gave tractable likelihoods and stable training, but sequential generation over millions of pixels was slow.

Diffusion models offered a different trade-off: training is stable (just regression), likelihoods are tractable (or at least well-bounded), samples are high quality, and the generation process is *iterative* — you can trade compute for quality at inference time. The cost is that sampling requires many forward passes (typically 20–1000 steps) rather than one. Subsequent work on fast samplers (DDIM, DPM-Solver, flow matching) has compressed this to a handful of steps, eroding the main disadvantage.

For an LLM engineer the key insight is that the same attention-heavy U-Net or Transformer architecture used in diffusion is related to the architectures you already understand, and the conditioning mechanisms used to inject text prompts into image generation are direct siblings of cross-attention in encoder-decoder LLMs (see [Architecture Variants: Encoder-Decoder, Decoder-Only & Prefix-LM](../02-transformer/08-architecture-variants.html)).

## The Forward Process: Gradually Destroying Data

{{fig:diffusion}}

The central idea of a diffusion model is to define a *forward* Markov chain that gradually adds Gaussian noise to a data sample until it becomes indistinguishable from pure noise, then train a neural network to *reverse* that process.

### Formal Definition

Let $\mathbf{x}_0 \sim q(\mathbf{x}_0)$ be a sample from the data distribution. The forward process defines a chain of latent variables $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_T$ via:

$$
q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t;\; \sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\; \beta_t \mathbf{I})
$$

where $\{\beta_t\}_{t=1}^T$ is a *noise schedule* — a sequence of small positive scalars, typically increasing from something like $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$ (linear schedule) or following a cosine schedule (Nichol & Dhariwal).

The magic is that there is a *closed-form* marginal for any timestep $t$ directly from $\mathbf{x}_0$. Define $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$. Then:

$$
q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t;\; \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0,\; (1 - \bar{\alpha}_t)\,\mathbf{I})
$$

Equivalently, we can sample $\mathbf{x}_t$ in one shot:

$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0},\mathbf{I})
$$

This is the *reparameterization* that makes training efficient: we can jump to any noise level without stepping through all intermediate $t$.

As $t \to T$, $\bar{\alpha}_T \approx 0$, so $\mathbf{x}_T \approx \mathcal{N}(\mathbf{0}, \mathbf{I})$ regardless of the original data. The schedule is designed to guarantee this.

### Noise Schedules

Three common schedules:

| Schedule | $\bar{\alpha}_t$ | Notes |
|---|---|---|
| Linear | Linear from 1 to $\approx 0$ | DDPM default; can be too aggressive for high-res |
| Cosine (Nichol & Dhariwal) | $\cos^2\!\left(\frac{t/T + s}{1+s}\cdot\frac{\pi}{2}\right)$ | Smoother; stays noisy less at end |
| Flow matching (linear interpolant) | $1 - t/T$ | Used in rectified flow; simpler |

!!! example "Worked Example: Noise Levels"
    Suppose $T = 1000$ and we use a linear schedule with $\beta_1 = 10^{-4}$ and $\beta_{1000} = 0.02$. At step $t=500$:

    $$\bar{\alpha}_{500} = \prod_{s=1}^{500} (1 - \beta_s) \approx \exp\!\left(-\sum_{s=1}^{500}\beta_s\right)$$

    With a linear schedule, $\beta_s \approx 10^{-4} + (s-1)\cdot\frac{0.02-10^{-4}}{999}$. The sum $\sum_{s=1}^{500}\beta_s \approx \frac{500}{2}(10^{-4} + 0.01) \approx 2.525$, giving $\bar{\alpha}_{500} \approx e^{-2.525} \approx 0.08$.

    So at $t=500$ we have: $\mathbf{x}_{500} = \sqrt{0.08}\,\mathbf{x}_0 + \sqrt{0.92}\,\boldsymbol{\epsilon} \approx 0.283\,\mathbf{x}_0 + 0.959\,\boldsymbol{\epsilon}$.

    The signal-to-noise ratio (SNR) is $\text{SNR}(500) = \bar{\alpha}_t/(1-\bar{\alpha}_t) \approx 0.08/0.92 \approx 0.087$ — the sample is dominated by noise. At $t=100$, $\bar{\alpha}_{100} \approx e^{-0.505} \approx 0.60$, SNR $\approx 1.5$ — still mostly signal. This asymmetry is why the cosine schedule was proposed: the linear schedule destroys structure too quickly at early steps.

## The Reverse Process and Training Objective (DDPM)

The generative model tries to learn the reverse of the forward process:

$$
p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1};\; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t),\; \sigma_t^2 \mathbf{I})
$$

We want to maximize the likelihood $p_\theta(\mathbf{x}_0)$ of the data. The variational lower bound (ELBO) gives a tractable training objective. After several algebraic manipulations, Ho et al. showed that the dominant term is:

$$
\mathcal{L}_\text{simple} = \mathbb{E}_{t,\mathbf{x}_0,\boldsymbol{\epsilon}}\!\left[\left\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\!\left(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon},\; t\right)\right\|^2\right]
$$

This is just mean-squared-error between the true noise $\boldsymbol{\epsilon}$ and the neural network's *prediction of that noise* $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$. The simplicity is striking: we sample a random timestep, corrupt a training example, and ask the network to guess the noise. Training is stable because it is regression throughout.

??? note "Optional: the full derivation from the ELBO to $\mathcal{L}_\text{simple}$"
    Start from the standard variational bound on the negative log-likelihood, obtained by importance-weighting with the forward chain $q$:

    $$
    -\log p_\theta(\mathbf{x}_0) \le \mathbb{E}_q\!\left[-\log\frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)}\right] = \underbrace{D_\text{KL}\!\left(q(\mathbf{x}_T\mid\mathbf{x}_0)\,\|\,p(\mathbf{x}_T)\right)}_{\mathcal{L}_T} + \sum_{t>1}\underbrace{D_\text{KL}\!\left(q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)\,\|\,p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)\right)}_{\mathcal{L}_{t-1}} \underbrace{-\log p_\theta(\mathbf{x}_0\mid\mathbf{x}_1)}_{\mathcal{L}_0}
    $$

    $\mathcal{L}_T$ has no parameters (it is a constant that the schedule design drives to $\approx 0$). The middle terms are KLs between *Gaussians*, because the forward posterior conditioned on $\mathbf{x}_0$ is available in closed form by Bayes' rule on the two Gaussians $q(\mathbf{x}_t\mid\mathbf{x}_{t-1})$ and $q(\mathbf{x}_{t-1}\mid\mathbf{x}_0)$:

    $$
    q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0) = \mathcal{N}\!\left(\tilde{\boldsymbol\mu}_t(\mathbf{x}_t,\mathbf{x}_0),\ \tilde\beta_t\mathbf{I}\right),\quad
    \tilde{\boldsymbol\mu}_t = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\mathbf{x}_0 + \frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\mathbf{x}_t,\quad
    \tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t
    $$

    For two Gaussians with the *same* fixed covariance $\sigma_t^2\mathbf{I}$, the KL collapses to a scaled squared distance between means, $\mathcal{L}_{t-1}=\frac{1}{2\sigma_t^2}\|\tilde{\boldsymbol\mu}_t-\boldsymbol\mu_\theta\|^2$. Substituting $\mathbf{x}_0=(\mathbf{x}_t-\sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon)/\sqrt{\bar\alpha_t}$ into $\tilde{\boldsymbol\mu}_t$ and matching it with the $\boldsymbol\mu_\theta$ parameterization stated just below, the $\mathbf{x}_t$ terms cancel and only the noise residual survives:

    $$
    \mathcal{L}_{t-1} = \mathbb{E}_{\mathbf{x}_0,\boldsymbol\epsilon}\!\left[\frac{\beta_t^2}{2\sigma_t^2\,\alpha_t\,(1-\bar\alpha_t)}\left\|\boldsymbol\epsilon-\boldsymbol\epsilon_\theta(\mathbf{x}_t,t)\right\|^2\right]
    $$

    So the true ELBO is a *weighted* noise-prediction regression. Ho et al.'s $\mathcal{L}_\text{simple}$ simply sets every weight to 1. That is not the ELBO — it up-weights large-$t$ (high-noise) terms relative to the bound — but it empirically improves sample quality, because the ELBO over-invests capacity in imperceptible high-frequency detail at small $t$. Kingma et al.'s *Variational Diffusion Models* made this precise: for any variance-preserving schedule the diffusion loss depends on the schedule only through the SNR endpoints, and different loss weightings are exactly different reweightings of $\mathrm{SNR}(t)$ — which is why "choose a weighting" (min-SNR-$\gamma$, $P_2$, EDM's $\lambda(\sigma)$, SD3's logit-normal timestep sampling) is one of the few knobs that reliably moves quality.

Given the predicted noise, the mean of the reverse step is:

$$
\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}}\!\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right)
$$

Alternatively, the network can be parameterized to predict $\mathbf{x}_0$ directly ($\mathbf{x}$-prediction), the score function $\nabla_{\mathbf{x}_t}\log q(\mathbf{x}_t)$, or the *velocity* $\mathbf{v} = \sqrt{\bar\alpha_t}\,\boldsymbol\epsilon - \sqrt{1-\bar\alpha_t}\,\mathbf{x}_0$ (Salimans & Ho, *Progressive Distillation*). All four are related by invertible linear maps at fixed $t$, so they parameterize the *same* model — but they are *not* the same objective, because a plain MSE in each output space implies a different implicit per-timestep weighting (Exercise 5 works this out for $\mathbf{x}_0$-prediction). $\mathbf{v}$-prediction is the practical default for high-resolution and distilled models: it stays well-conditioned at both $t\to0$ (where $\boldsymbol\epsilon$-prediction is nearly information-free) and $t\to T$ (where $\mathbf{x}_0$-prediction is), which is why SD 2.x, progressive distillation, and most video models use it.

!!! warning "Common pitfall: non-zero terminal SNR"
    The standard linear/cosine schedules do not actually reach $\bar\alpha_T = 0$ — DDPM's linear schedule leaves $\bar\alpha_T \approx 4\times10^{-5}$, i.e. a small but non-zero terminal SNR. Training therefore never shows the network a *truly* pure-noise input, yet sampling always starts from one. Under $\boldsymbol\epsilon$-prediction the leak is mild; it becomes visible as a *mean-brightness bias* — such models struggle to generate very dark or very bright images and drift toward medium grey, regardless of the prompt. Lin et al. (*Common Diffusion Noise Schedules and Sample Steps Are Flawed*, WACV 2024) diagnosed this and prescribe the fix: rescale the schedule to enforce $\bar\alpha_T = 0$ exactly, switch to $\mathbf{v}$-prediction (which stays defined at zero SNR), and start sampling at the true last timestep. In `diffusers` this is the `rescale_betas_zero_snr=True` scheduler flag paired with `prediction_type="v_prediction"`. Flow-matching models get this for free: the path *starts* at pure noise by construction.

```python
import torch
import torch.nn as nn
import math

# ------------------------------------------------------------------ #
#  Minimal DDPM: forward process + training loss                      #
# ------------------------------------------------------------------ #

def make_linear_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    """Return betas, alphas_bar for a linear noise schedule."""
    betas = torch.linspace(beta_start, beta_end, T)   # (T,)
    alphas = 1.0 - betas                               # (T,)
    alphas_bar = torch.cumprod(alphas, dim=0)          # (T,)  ā_t
    return betas, alphas, alphas_bar


def q_sample(x0: torch.Tensor, t: torch.Tensor, alphas_bar: torch.Tensor):
    """
    Sample x_t from q(x_t | x_0) in one shot using the reparameterization:
        x_t = sqrt(ā_t) * x0 + sqrt(1 - ā_t) * eps
    Args:
        x0:        (B, C, H, W) clean images in [-1, 1]
        t:         (B,)         integer timesteps
        alphas_bar:(T,)         precomputed ā_t values
    Returns:
        x_t, eps   both (B, C, H, W)
    """
    eps = torch.randn_like(x0)
    # gather ā_t for each sample in the batch (.to() keeps the CPU-built
    # schedule usable when x0/t live on GPU — a classic silent crash)
    ab = alphas_bar.to(x0.device)[t].view(-1, 1, 1, 1)   # broadcast over C,H,W
    x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
    return x_t, eps


def ddpm_loss(model: nn.Module, x0: torch.Tensor, alphas_bar: torch.Tensor):
    """
    Simple DDPM training loss: L_simple = E[||eps - eps_theta(x_t, t)||^2].
    model: a U-Net that takes (x_t, t) and predicts noise eps_hat.
    """
    B = x0.shape[0]
    T = alphas_bar.shape[0]
    # sample a random timestep for each item in the batch
    t = torch.randint(0, T, (B,), device=x0.device)
    x_t, eps = q_sample(x0, t, alphas_bar)
    # predict the noise; model typically also takes a time embedding
    eps_hat = model(x_t, t)
    return (eps - eps_hat).pow(2).mean()


# ------------------------------------------------------------------ #
#  DDPM Reverse Sampler                                               #
# ------------------------------------------------------------------ #

@torch.no_grad()
def ddpm_sample(model: nn.Module, shape: tuple,
                betas: torch.Tensor, alphas: torch.Tensor,
                alphas_bar: torch.Tensor, device: str = "cpu"):
    """
    Ancestral sampling: start from Gaussian noise, iteratively denoise.
    Returns x_0 after T reverse steps.
    """
    T = betas.shape[0]
    x = torch.randn(shape, device=device)   # x_T ~ N(0, I)

    for t in reversed(range(T)):
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
        eps_hat = model(x, t_batch)

        # Compute mean of reverse step
        alpha_t  = alphas[t]
        ab_t     = alphas_bar[t]
        beta_t   = betas[t]

        # mu_theta = (x_t - beta_t / sqrt(1 - ab_t) * eps_hat) / sqrt(alpha_t)
        coeff = beta_t / (1 - ab_t).sqrt()
        mean  = (x - coeff * eps_hat) / alpha_t.sqrt()

        if t > 0:
            # Variance: the simple choice is just beta_t I
            noise = torch.randn_like(x)
            x = mean + beta_t.sqrt() * noise
        else:
            x = mean   # no noise at final step

    return x
```

## Score Matching and the Connection to SDEs

A complementary view, developed by Song et al. and earlier Hyvärinen, frames diffusion as *score matching*. The *score* of a distribution is:

$$
\mathbf{s}(\mathbf{x}) = \nabla_\mathbf{x} \log p(\mathbf{x})
$$

It is a vector field pointing toward higher-density regions. If we know the score, we can use *Langevin dynamics* to draw samples:

$$
\mathbf{x}_{k+1} = \mathbf{x}_k + \eta\, \nabla_\mathbf{x} \log p(\mathbf{x}_k) + \sqrt{2\eta}\,\boldsymbol{\epsilon}_k
$$

Training a network $\mathbf{s}_\theta \approx \nabla_\mathbf{x} \log p(\mathbf{x})$ is called *score matching*. Ho et al.'s noise prediction is equivalent: the score of the noisy distribution relates to the noise predictor as:

$$
\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t \mid \mathbf{x}_0) = -\frac{\boldsymbol{\epsilon}}{\sqrt{1 - \bar{\alpha}_t}}
$$

Song et al. (Score SDE) generalized the whole framework to *stochastic differential equations* (SDEs). The forward process is an SDE:

$$
d\mathbf{x} = \mathbf{f}(\mathbf{x}, t)\,dt + g(t)\,d\mathbf{w}
$$

and the reverse is also an SDE whose drift term depends on the score:

$$
d\mathbf{x} = \left[\mathbf{f}(\mathbf{x}, t) - g(t)^2 \nabla_\mathbf{x} \log p_t(\mathbf{x})\right]dt + g(t)\,d\bar{\mathbf{w}}
$$

This unifies DDPM, SMLD (denoising score matching with multiple noise levels), and continuous-time variants under a single formalism and enables the ODE sampler that DDIM exploits.

### DDIM: Deterministic Sampling

Denoising Diffusion Implicit Models (Song et al., DDIM) derived a *non-Markovian* inference process that shares the same marginals as DDPM but allows a deterministic (ODE-based) trajectory. The update becomes:

$$
\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\underbrace{\frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}_\theta}{\sqrt{\bar{\alpha}_t}}}_{\text{predicted }\mathbf{x}_0} + \sqrt{1-\bar{\alpha}_{t-1} - \sigma_t^2}\,\boldsymbol{\epsilon}_\theta + \sigma_t\,\boldsymbol{\epsilon}
$$

Setting $\sigma_t = 0$ makes the process fully deterministic: you can think of it as Euler integration of the *probability flow ODE*. DDIM allows *subsampling* — you can skip timesteps and use, say, 50 steps instead of 1000 with minimal quality loss, giving a 20× speedup.

Once you accept the ODE view, DDIM is just the crudest solver, and the whole numerical-integration literature becomes available. **DPM-Solver / DPM-Solver++** (Lu et al.) exploit the *semi-linear* structure of the probability-flow ODE — the linear drift term is integrated analytically and only the neural part is approximated — giving second- and third-order multistep schemes that reach DDIM's 50-step quality in roughly 10–20 steps. **UniPC** adds a predictor–corrector wrapper on top. **EDM** (Karras et al., *Elucidating the Design Space of Diffusion-Based Generative Models*) rewrites the whole framework in terms of a single noise level $\sigma$ and uses Heun's second-order method with a tuned $\sigma$ spacing. These are drop-in replacements for `ddim_sample`: same trained network, different integrator. Below ~4 steps, no solver saves you — the ODE trajectory is genuinely curved — and you must instead *distill* (progressive distillation, consistency models / latent consistency models, adversarial distillation as in SDXL-Turbo and FLUX.1 [schnell]) or train with straight paths from the start, which is what flow matching does.

```python
@torch.no_grad()
def ddim_sample(model: nn.Module, shape: tuple, alphas_bar: torch.Tensor,
                num_steps: int = 50, eta: float = 0.0, device: str = "cpu"):
    """
    DDIM sampler.  eta=0 => deterministic; eta=1 => DDPM-like stochastic.
    Uses a uniform sub-sequence of timesteps.
    """
    T = alphas_bar.shape[0]
    # Select a uniform subsequence: e.g., [980, 960, ..., 20, 0]
    step_size = T // num_steps
    timesteps = list(reversed(range(0, T, step_size)))  # [T-1, ..., 0]

    x = torch.randn(shape, device=device)

    for i, t in enumerate(timesteps):
        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1

        t_batch = torch.full((shape[0],), t, dtype=torch.long, device=device)
        eps_hat = model(x, t_batch)

        ab_t    = alphas_bar[t]
        ab_prev = alphas_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0)

        # Predict x0 from current x_t and predicted noise
        x0_pred = (x - (1 - ab_t).sqrt() * eps_hat) / ab_t.sqrt()
        x0_pred = x0_pred.clamp(-1, 1)  # optional clamping

        # Direction pointing to x_t
        sigma_t = eta * ((1 - ab_prev) / (1 - ab_t)).sqrt() * (1 - ab_t / ab_prev).sqrt()
        dir_xt  = (1 - ab_prev - sigma_t**2).sqrt() * eps_hat

        noise   = torch.randn_like(x) if eta > 0 and t_prev >= 0 else 0.0
        x       = ab_prev.sqrt() * x0_pred + dir_xt + sigma_t * noise

    return x
```

## Classifier Guidance and Classifier-Free Guidance

Unconditional diffusion models generate beautiful images from pure noise. To generate *specific* content — "a photo of a cat wearing a hat" — we need to condition on a signal. Two approaches dominate.

### Classifier Guidance

Dhariwal & Nichol showed that if you have a classifier $p_\phi(y \mid \mathbf{x}_t)$ trained on noisy images, you can steer the reverse diffusion using:

$$
\tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, y) = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) - \sqrt{1-\bar{\alpha}_t}\,\gamma\,\nabla_{\mathbf{x}_t}\log p_\phi(y \mid \mathbf{x}_t)
$$

The guidance scale $\gamma > 0$ amplifies the classifier signal. Larger $\gamma$ yields more class-faithful but less diverse samples.

### Classifier-Free Guidance (CFG)

CFG (Ho & Salimans) eliminated the need for a separate classifier. During training, the conditioning signal $\mathbf{c}$ (e.g., a text embedding) is randomly dropped with some probability $p_\text{drop}$ (typically 10–20%), replacing it with a null token $\emptyset$. At inference, predictions from the *conditional* and *unconditional* models are interpolated:

$$
\tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, \mathbf{c}) = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \emptyset) + w\,\left[\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \emptyset)\right]
$$

where $w$ is the guidance scale (common values: 3–15). Larger $w$ increases adherence to the prompt at the cost of sample diversity and can cause over-saturation. CFG doubles the number of forward passes per step but requires no separate classifier.

Intuitively, $w=0$ is unconditioned, $w=1$ is the pure conditional model, and $w>1$ *extrapolates* beyond the conditional distribution, sharpening the conditioning signal. This extrapolation is why CFG boosts both quality and prompt adherence simultaneously, and also why it can produce slightly unrealistic outputs at large guidance scales.

{{fig:cfg-extrapolation-vectors}}

!!! interview "Interview Corner"
    **Q:** A colleague proposes increasing the classifier-free guidance scale from 7 to 20 to get better text alignment in a text-to-image model. What are the trade-offs, and what would you recommend instead?

    **A:** Increasing CFG scale $w$ amplifies the gradient of $\log p(\mathbf{c}|\mathbf{x})$ beyond the data manifold, so the model extrapolates aggressively. The result is sharper conditioning but lower sample diversity, mode collapse onto over-saturated outputs (vivid colors, hard edges), and artifacts (over-sharpened edges, unnatural saturation). At $w=20$ the generated images often look "AI-ish" with blown-out highlights. Better options: (1) use *dynamic thresholding* (Saharia et al., Imagen), which clips the predicted $\mathbf{x}_0$ percentile-wise instead of naively clamping, preserving saturation; (2) use *AutoGuidance* or *Perturbed-Attention Guidance* which degrade the negative sample more gracefully; (3) fine-tune the text encoder or add an adapter rather than cranking CFG; (4) try a lower $w$ with more denoising steps. I'd stay at $w \leq 12$ and apply dynamic thresholding.

## Latent Diffusion and Stable Diffusion

Running diffusion in pixel space for high-resolution images is computationally prohibitive: 1000 steps × a 512×512×3 U-Net forward pass is expensive. Rombach et al. proposed *Latent Diffusion Models* (LDMs), the architecture underlying Stable Diffusion.

### Architecture Overview

{{fig:diffgen-latent-diffusion-pipeline}}

The key insight is that the *perceptual* information in an image lives in a much lower-dimensional space. The VAE compresses 512×512×3 = 786,432 values to 64×64×4 = 16,384 values — a 48× reduction. Diffusion then runs on this compact latent space, reducing compute by roughly $8^2 = 64\times$ per forward pass (since attention scales quadratically with spatial resolution).

### U-Net Denoiser Architecture

The denoiser in most LDMs is a U-Net with:
- Convolutional downsampling/upsampling backbone
- Residual blocks (ResNet-style) at each resolution
- Self-attention layers (at lower resolutions, where sequences are short enough)
- Cross-attention layers for text conditioning: queries from spatial features, keys/values from text embeddings

The DiT (Diffusion Transformer, Peebles & Xie) replaced the U-Net entirely with a Vision Transformer (ViT) operating on latent patches, conditioning on timestep and text through adaptive layer norm. DiT scales predictably with model size and is used in Stable Diffusion 3, Flux, and Sora-like systems.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalTimeEmbedding(nn.Module):
    """Encode scalar timestep t into a D-dimensional embedding."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,)  integer timesteps
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )  # (half,)
        args = t[:, None].float() * freqs[None, :]  # (B, half)
        emb  = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)
        return emb


class ResBlock(nn.Module):
    """ResNet block with time-step conditioning via AdaGN or addition."""
    def __init__(self, channels: int, time_emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        # project time embedding to channel-wise scale + shift
        self.time_proj = nn.Linear(time_emb_dim, channels * 2)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # t_emb: (B, time_emb_dim)
        h = self.conv1(F.silu(self.norm1(x)))
        # inject time via scale + shift (AdaGN-style)
        scale, shift = self.time_proj(F.silu(t_emb)).chunk(2, dim=-1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h   # residual connection


class TinyUNet(nn.Module):
    """
    Minimal 2-level U-Net denoiser for illustration.
    Accepts (x_t, t) and returns predicted noise eps_hat.
    In production: multiple levels, attention, cross-attention for conditioning.
    """
    def __init__(self, in_channels: int = 4, base_ch: int = 64, time_dim: int = 256):
        super().__init__()
        self.time_emb  = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp  = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4), nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        self.enc1 = ResBlock(base_ch, time_dim)
        self.enc2 = ResBlock(base_ch * 2, time_dim)
        self.mid  = ResBlock(base_ch * 2, time_dim)
        self.dec1 = ResBlock(base_ch * 2, time_dim)
        self.dec2 = ResBlock(base_ch, time_dim)

        self.in_proj  = nn.Conv2d(in_channels, base_ch, 1)
        self.down     = nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1)
        self.up       = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.out_proj = nn.Conv2d(base_ch, in_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(self.time_emb(t))   # (B, time_dim)
        h1 = self.enc1(self.in_proj(x), t_emb)    # (B, C, H, W)
        h2 = self.enc2(self.down(h1), t_emb)      # (B, 2C, H/2, W/2)
        h2 = self.mid(h2, t_emb)
        h  = self.dec1(h2, t_emb)
        h  = self.dec2(self.up(h) + h1, t_emb)    # skip connection
        return self.out_proj(h)
```

### Doing It For Real: Hugging Face `diffusers`

Nobody re-derives $\bar\alpha_t$ in production. `diffusers` is the PyTorch library that owns this layer of the stack: it factors every model into a *scheduler* (the noise schedule + the sampler update, i.e. everything in our `make_linear_schedule`/`ddim_sample`), a *denoiser* (`UNet2DConditionModel` or `SD3Transformer2DModel`/`FluxTransformer2DModel`), a *VAE* (`AutoencoderKL`), and *text encoders* pulled from `transformers`. A `Pipeline` is just the glue. Crucially the scheduler API mirrors the two functions we wrote by hand — `add_noise` is `q_sample`, `step` is one reverse update — so you can swap DDIM for DPM-Solver++ or a flow-matching Euler solver without retraining.

```python
# pip install diffusers transformers accelerate peft safetensors
import torch
from diffusers import (AutoencoderKL, UNet2DConditionModel, DDPMScheduler,
                       DPMSolverMultistepScheduler, StableDiffusionXLPipeline)

# ---- (1) Training: one latent-diffusion step, library primitives ------- #
sched = DDPMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear",
                      prediction_type="epsilon")   # or "v_prediction"

def training_step(vae: AutoencoderKL, unet: UNet2DConditionModel,
                  pixels: torch.Tensor, text_emb: torch.Tensor):
    """pixels: (B,3,H,W) in [-1,1]; text_emb: (B, L, D) from a CLIP/T5 encoder."""
    with torch.no_grad():                                   # VAE is frozen
        latents = vae.encode(pixels).latent_dist.sample()
        latents = latents * vae.config.scaling_factor       # ~0.18215 for SD1.x
    noise = torch.randn_like(latents)
    t = torch.randint(0, sched.config.num_train_timesteps,
                      (latents.shape[0],), device=latents.device)
    noisy = sched.add_noise(latents, noise, t)              # == our q_sample
    pred  = unet(noisy, t, encoder_hidden_states=text_emb).sample
    # target depends on the parameterization the scheduler was configured with
    target = noise if sched.config.prediction_type == "epsilon" \
             else sched.get_velocity(latents, noise, t)
    return torch.nn.functional.mse_loss(pred.float(), target.float())

# ---- (2) Inference: swap the sampler, no retraining ------------------- #
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()          # fits a 24 GB card comfortably
img = pipe("a photo of a cat wearing a hat", num_inference_steps=25,
           guidance_scale=7.0).images[0]
```

Three practical notes. **(a)** Conditioning drop-out for CFG is your job at training time — replace `text_emb` with the encoder's embedding of `""` for ~10% of examples, exactly the $\emptyset$ token of the CFG section. **(b)** Fine-tuning is almost always LoRA, not full fine-tuning: `diffusers` integrates `peft`, and its `examples/text_to_image/train_text_to_image_lora_sdxl.py` script is the canonical starting point — the same low-rank adapters covered in [PEFT I: LoRA, QLoRA, DoRA & The Adapter Family](../05-posttraining-alignment/03-peft-lora-qlora.html), applied to the U-Net's attention projections. **(c)** Rectified-flow models use `FlowMatchEulerDiscreteScheduler` and pipelines `StableDiffusion3Pipeline` / `FluxPipeline`; FLUX.1 [schnell] is distilled to run at `num_inference_steps=4` with `guidance_scale=0.0` (guidance is *baked into* the weights via guidance distillation, so a second forward pass would be wasted). Beyond `diffusers`, the practitioner ecosystem is **ComfyUI** (node-graph inference), **k-diffusion** (Karras/EDM samplers), and **kohya-ss/sd-scripts** (LoRA/DreamBooth training).

## Flow Matching: A Cleaner Generalization

Flow matching (Lipman et al.; Liu et al., *Flow Straight and Fast*; Albergo & Vanden-Eijnden) is a simpler and more general framework that subsumes diffusion but removes the requirement for a Markov chain.

### The Idea

Define a probability path $p_t$ interpolating between noise $p_0 = \mathcal{N}(\mathbf{0},\mathbf{I})$ at $t=0$ and data $p_1$ at $t=1$. A *velocity field* $\mathbf{v}_t(\mathbf{x})$ generates this path via the continuity equation (a.k.a. transport equation):

$$
\frac{\partial p_t}{\partial t} + \nabla \cdot (p_t \mathbf{v}_t) = 0
$$

We train a neural network $\mathbf{v}_\theta(\mathbf{x}, t)$ to match this field. The training objective for *conditional flow matching* (CFM) is:

$$
\mathcal{L}_\text{CFM} = \mathbb{E}_{t,\mathbf{x}_0,\mathbf{x}_1}\!\left[\left\|\mathbf{v}_\theta(\mathbf{x}_t, t) - (\mathbf{x}_1 - \mathbf{x}_0)\right\|^2\right]
$$

where $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\,\mathbf{x}_1$ is the straight-line interpolant between a noise sample $\mathbf{x}_0$ and data sample $\mathbf{x}_1$. The target velocity is simply $\mathbf{x}_1 - \mathbf{x}_0$ — constant along each path.

This is *Rectified Flow* (Liu et al.) in discrete language: training pairs are $(x_0^\text{noise}, x_1^\text{data})$, the path is a straight line, and the network learns to predict the direction. At inference, solving the ODE from $t=0$ to $t=1$ recovers a generated sample.

### Why Flow Matching Wins in Practice

1. **Fewer steps**: Straight-line paths need fewer integration steps. Diffusion paths are curved (the noise-added marginals trace complex curves), requiring many small steps. Flow matching paths are linear, so a 4-step Euler solver often suffices.
2. **Simpler math**: No SDE formalism, no noise schedule, no $\bar{\alpha}_t$ product.
3. **Exact likelihood**: Since the velocity field defines an ODE, the change-of-variables formula gives an exact log-likelihood (though it requires ODE integration).
4. **Flexible couplings**: You are not restricted to $\mathcal{N}(\mathbf{0},\mathbf{I})$ as $p_0$; you can use a learned prior or a distribution related to the task.

Flow matching is now the backbone of Stable Diffusion 3, Flux.1, and many audio generation models.

{{fig:flow-vs-diffusion-paths}}

```python
def flow_matching_loss(model: nn.Module, x1: torch.Tensor):
    """
    Conditional flow matching training step (Rectified Flow variant).
    x1: (B, C, H, W)  clean data samples (e.g., images)
    model: predicts velocity field v(x_t, t)
    """
    B = x1.shape[0]
    # Sample noise from N(0, I)
    x0 = torch.randn_like(x1)
    # Sample a uniform time t in [0, 1]
    t  = torch.rand(B, device=x1.device)
    # Interpolate: x_t = (1 - t) * x0 + t * x1
    t_broadcast = t.view(B, 1, 1, 1)
    x_t = (1 - t_broadcast) * x0 + t_broadcast * x1
    # Target velocity: dx_t/dt = x1 - x0  (constant along path)
    target_v = x1 - x0
    # Predict velocity
    v_hat = model(x_t, t)
    return (v_hat - target_v).pow(2).mean()


@torch.no_grad()
def flow_matching_sample(model: nn.Module, shape: tuple,
                         num_steps: int = 8, device: str = "cpu"):
    """
    Euler ODE integration from t=0 (noise) to t=1 (data).
    With straight-line paths, even 4-8 steps is often enough.
    """
    x = torch.randn(shape, device=device)
    dt = 1.0 / num_steps
    ts = torch.linspace(0.0, 1.0 - dt, num_steps, device=device)

    for t_val in ts:
        t_batch = torch.full((shape[0],), t_val, device=device)
        v = model(x, t_batch)
        x = x + v * dt   # Euler step

    return x
```

## Diffusion Language Models

The success of diffusion in continuous domains (images, audio) raises a natural question: can it work for *discrete* text? After all, text lives on a finite vocabulary, not a Gaussian manifold.

### Challenges

1. **Discrete space**: Adding Gaussian noise to a token index produces a float, not a token. Standard Gaussian diffusion does not apply directly.
2. **Non-continuous interpolation**: There is no obvious straight-line path between the token "cat" and "dog."
3. **Mask-based diffusion**: One approach (Austin et al., D3PM; Chang et al., MaskGIT; Sahoo et al., MDLM) replaces noise corruption with *masking*: the forward process randomly masks tokens, and the reverse process predicts the masked tokens (much like masked language modeling). This is called *absorbing diffusion*, because `[MASK]` is an absorbing state of the corruption chain — once a token is masked it stays masked.
4. **Embedding-space diffusion**: Another approach (Li et al., Diffusion-LM; Dieleman et al., *Continuous Diffusion for Categorical Data*; Lovelace et al., *Latent Diffusion for Language Generation*) runs diffusion in a continuous embedding or latent space, then decodes back to tokens. The challenge is round-trip fidelity: denoised embeddings may not correspond to any real token, so these methods need a rounding/clamping step or a learned decoder. In practice absorbing diffusion has won — every scaled diffusion LM (LLaDA, Mercury, Gemini Diffusion) is mask-based.

### Masked Diffusion Language Models

The most practical approach uses masking as the noise process:

$$
q(\mathbf{x}_t \mid \mathbf{x}_0) : \text{independently mask each token with probability } \gamma(t)
$$

The neural network (typically a Transformer) predicts all unmasked tokens simultaneously given context, similar to BERT's masked language modeling but repeated iteratively. During generation:

1. Start with all tokens masked.
2. Predict token probabilities for all positions.
3. Unmask the most confident subset.
4. Repeat until all tokens are unmasked.

This gives parallel generation, a key advantage over autoregressive models which generate strictly left-to-right. Models like MDLM and Plaid operate this way and can generate text of length $L$ in $O(K)$ passes for a fixed $K$ (e.g., 10) rather than $O(L)$ autoregressive steps.

Structurally the recipe is tiny: take the Transformer of [The Transformer Block: Norms, Residuals, MLPs & Activations](../02-transformer/06-transformer-block.html), drop the causal mask so attention is bidirectional, add one vocabulary row for a reserved `[MASK]` id, and train with masked cross-entropy reweighted by $1/t$ where $t$ is the mask rate — that weight is exactly what turns "BERT with a random mask rate" into a genuine likelihood bound (the NELBO of absorbing-state diffusion), which is why MDLM perplexities are comparable to autoregressive ones rather than incomparable pseudo-likelihoods. Two caveats the throughput headline hides: bidirectional attention means there is **no KV cache**, so $K$ diffusion passes cost $K$ full prefills rather than $K$ cheap decode steps (semi-autoregressive *block* diffusion, which diffuses one block at a time and caches the settled prefix, recovers much of this); and the generation *length* must be fixed up front, so EOS and padding need explicit conventions.

Because this is the one part of the diffusion story that operates directly on the book's core object — text — it gets a full treatment of its own, with from-scratch training loss, a confidence-based parallel sampler, block-causal masking, and the inference economics: see [Diffusion & Non-Autoregressive Language Models](../02-transformer/12-diffusion-nonAR-lms.html). If you want to run the ablation on the capstone corpus, the ingredients are already assembled — the tokenizer of [A Byte-Level BPE Tokenizer From Scratch](../14-capstone/03-tokenizer.html) plus one `[MASK]` id, the model of [The Stack-100M Architecture](../14-capstone/04-architecture.html) with the causal mask disabled, and the loop of [The Pretraining Run: A Complete Single-GPU Training Loop](../14-capstone/07-pretraining-run.html) with the masked-diffusion loss swapped in for next-token prediction. Stack-100M itself stays autoregressive: at that budget AR is the better-understood and far better-tooled choice.

### Comparison to Autoregressive Generation

| Dimension | Autoregressive (AR) | Diffusion / Masked |
|---|---|---|
| Generation order | Strict left-to-right | Any order / parallel |
| Steps required | $L$ (one per token) | $K \ll L$ passes |
| Conditional probability | Exact $p(x_i \mid x_{<i})$ | Approximate |
| Revision / editing | Hard (requires resampling) | Natural: re-mask and regenerate |
| Long-range coherence | Strong (attention over all previous) | Weaker (depends on architecture) |
| Current quality | Best for NLU/NLG tasks | Competitive at ~8B scale (LLaDA); commercial dLLMs shipping |
| Controllability | Via prompting | Easier arbitrary-region infilling |

As of 2026, diffusion LMs have scaled up and reached production. LLaDA-8B (Nie et al., 2025) matches LLaMA3-8B on in-context-learning benchmarks, while commercial diffusion LLMs (Inception Labs' Mercury) and Google DeepMind's experimental Gemini Diffusion exploit parallel token decoding for large throughput gains — Gemini Diffusion reports on the order of 1,400 tokens/sec. Autoregressive models still tend to hold an edge on the very longest-range reasoning, but the gap has narrowed sharply, and diffusion's native support for infilling makes it attractive for code editing, style transfer, and constrained generation scenarios (see [Structured & Constrained Generation](../07-inference-serving/10-structured-generation.html)).

{{fig:masked-diffusion-vs-ar-decoding}}

## Relevance for LLM Engineers

Why should an engineer focused on language models care about diffusion?

**1. Vision-language architectures.** Multimodal LLMs increasingly pair a language model with a diffusion decoder for image generation (e.g., Gemini's image output, or LLaVA-style models with SDXL as a decoder). Understanding the conditioning interface — how CLIP or T5 text embeddings are fed as cross-attention keys/values into the denoiser — is necessary to build and debug these systems (see [Vision-Language Models](../10-multimodal-and-arch/02-vision-language-models.html)).

**2. Reward models and RLHF for diffusion.** Just as LLMs are fine-tuned with RLHF (see [The RLHF Pipeline & Reward Modeling](../05-posttraining-alignment/05-rlhf-reward-modeling.html)), diffusion models are fine-tuned with reinforcement learning from human feedback using reward gradients backpropagated through the sampling chain (DDPO, ReFL). The policy gradient machinery is the same; the action space is the denoised image.

**3. Diffusion as a generative backend.** Systems like Stable Diffusion serve as compute-heavy generation backends; LLM engineers writing serving stacks need to reason about the inference throughput of iterative samplers, batch sizing across steps, and caching of the text encoder (which runs once per prompt rather than once per step). Note that none of the LLM serving engines apply here — vLLM/SGLang exist to manage a KV cache and variable-length autoregressive decoding, neither of which a denoiser has. Image serving instead means `diffusers` plus `torch.compile`, fused attention kernels, batching the CFG pair into one forward pass, and optionally TensorRT/ONNX export; the workload is compute-bound and fixed-shape, which is the *opposite* of the memory-bound, ragged-shape LLM decode regime analysed in [The Anatomy of LLM Inference: Prefill, Decode & The KV Cache](../07-inference-serving/01-anatomy-inference.html).

**4. Score functions and energy-based intuitions.** The score function $\nabla_\mathbf{x} \log p(\mathbf{x})$ and the energy-based model perspective appear in contrastive learning losses, noise-contrastive estimation (the NCE loss used in word2vec), and the connection between reward models and energy-based models in RLHF. Diffusion gives you a concrete, visual intuition for these abstract objects.

**5. Interview breadth.** Google ML interviews frequently test "ML breadth" questions that span generative models. You need to be able to describe the DDPM training objective, CFG, and how latent diffusion differs from pixel diffusion in roughly two minutes each.

!!! note "Diffusion vs. AR: A Unifying Lens"
    Both diffusion and autoregressive models define a factored generative process: AR factorizes as a chain of conditionals over tokens, while diffusion factorizes as a chain of conditionals over noise levels. The network in both cases is a Transformer whose job is to approximate a conditional distribution. The difference lies in *what* is being conditioned on: in AR, conditioning is on previous tokens; in diffusion, conditioning is on a noisy version of the output and a timestep. Flow matching makes this even cleaner: the network learns a vector field interpolating between two distributions.

## Key Mathematical Connections (Summary Table)

| Concept | Formula / object | Intuition |
|---|---|---|
| Forward marginal | $q(\mathbf{x}_t\mid\mathbf{x}_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\mathbf{x}_0, (1-\bar\alpha_t)\mathbf{I})$ | Jump to any noise level in one shot |
| Training loss | $\mathbb{E}\|\boldsymbol\epsilon - \boldsymbol\epsilon_\theta(\mathbf{x}_t,t)\|^2$ | Just regression on the added noise |
| Score | $\nabla_{\mathbf{x}_t}\log q(\mathbf{x}_t) = -\boldsymbol\epsilon/\sqrt{1-\bar\alpha_t}$ | Score and noise predictor are the same thing |
| CFG formula | $(1-w)\boldsymbol\epsilon_\emptyset + w\,\boldsymbol\epsilon_\mathbf{c}$ (with $w>1$ extrapolating) | Steering toward condition |
| DDIM ODE step | Euler step on probability flow ODE | Makes sampling deterministic and subsampable |
| Flow matching loss | $\mathbb{E}\|\mathbf{v}_\theta(\mathbf{x}_t,t) - (\mathbf{x}_1-\mathbf{x}_0)\|^2$ | Predict straight-line velocity |

!!! example "End-to-End Magnitude Check: Stable Diffusion Inference"
    Consider Stable Diffusion XL (SDXL) generating a 1024×1024 image:

    - VAE compresses to latent $128\times128\times4$ (8× downsampling). Latent has $128\times128\times4=65{,}536$ values.
    - The U-Net denoiser has roughly 2.6B parameters (SDXL). Each step requires two forward passes with CFG.
    - At 20 DDIM steps: $20 \times 2 = 40$ U-Net forward passes total.
    - At float16, a single U-Net pass on a 1024×1024 latent requires on the order of 1.5–2 TFLOPs. Total for 40 passes: $\sim60$–80 TFLOPs.
    - On an A100 (312 TFLOPS fp16): $\approx 0.2$–$0.25$ seconds compute; actual wall time with memory I/O is roughly 1–3 seconds on a single A100.
    - Compare to pixel-space diffusion at the same resolution: the U-Net would operate on $1024\times1024\times3$ activations, roughly 64× larger spatial volume, making each step $\sim$64× more expensive — latent diffusion's entire raison d'être.

!!! sota "State of the Art & Resources (2026)"
    Diffusion and flow-matching models are the dominant paradigm for image and video generation: the field has moved from DDPM's 1000-step pixel-space sampling to rectified-flow transformers (SD3, Flux.1) that produce state-of-the-art images in fewer than 10 steps, while diffusion language models have scaled to the 8B range (LLaDA) and reached commercial deployment (Inception Labs' Mercury, Google's experimental Gemini Diffusion), offering a genuinely parallel alternative to autoregressive text generation.

    **Foundational work**

    - [Ho et al., *Denoising Diffusion Probabilistic Models* (2020)](https://arxiv.org/abs/2006.11239) — the DDPM paper that launched the modern diffusion era; introduces the simplified noise-prediction loss.
    - [Song et al., *Score-Based Generative Modeling through SDEs* (2021)](https://arxiv.org/abs/2011.13456) — unifies DDPM and score matching under a continuous SDE/ODE framework, enabling the probability-flow ODE used by DDIM and later samplers.
    - [Song et al., *Denoising Diffusion Implicit Models* (2021)](https://arxiv.org/abs/2010.02502) — reformulates sampling as ODE integration; enables deterministic generation and 10–50× step reduction.

    **Recent advances (2023–2026)**

    - [Dhariwal & Nichol, *Diffusion Models Beat GANs on Image Synthesis* (2021)](https://arxiv.org/abs/2105.05233) — introduces classifier guidance and the improved U-Net architecture that made diffusion models the quality leader.
    - [Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models* (2022)](https://arxiv.org/abs/2112.10752) — latent diffusion / Stable Diffusion; shows 64× compute savings by running diffusion in VAE latent space.
    - [Peebles & Xie, *Scalable Diffusion Models with Transformers* (DiT, 2023)](https://arxiv.org/abs/2212.09748) — replaces the U-Net with a ViT operating on latent patches; the architecture behind Sora and SD3.
    - [Lipman et al., *Flow Matching for Generative Modeling* (2023)](https://arxiv.org/abs/2210.02747) — simulation-free training of continuous normalizing flows; the theoretical basis for modern straight-path samplers.
    - [Liu et al., *Flow Straight and Fast: Rectified Flow* (2023)](https://arxiv.org/abs/2209.03003) — straight-line interpolant between noise and data; backbone of SD3 and Flux.1.
    - [Esser et al., *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis* (2024)](https://arxiv.org/abs/2403.03206) — the Stable Diffusion 3 paper; combines DiT with rectified flow and multimodal attention for state-of-the-art text-to-image quality.
    - [Sahoo et al., *Simple and Effective Masked Diffusion Language Models* (NeurIPS 2024)](https://arxiv.org/abs/2406.07524) — MDLM: shows masked-discrete diffusion matches autoregressive perplexity on standard benchmarks, powering parallel text generation.
    - [Nie et al., *Large Language Diffusion Models* (LLaDA, 2025)](https://arxiv.org/abs/2502.09992) — scales masked diffusion to 8B parameters, matching LLaMA3-8B on in-context-learning benchmarks and showing that strong LLM capabilities are not unique to autoregressive training.

    **Open-source & tools**

    - [black-forest-labs/flux](https://github.com/black-forest-labs/flux) — official inference code for FLUX.1 [dev/schnell], a 12B-parameter rectified-flow transformer; the current open-weight state of the art for text-to-image.
    - [huggingface/diffusers](https://github.com/huggingface/diffusers) — the standard PyTorch library for diffusion pipelines (SDXL, SD3, Flux, video models); includes schedulers, LoRA, and quantization support.

    **Go deeper**

    - [Lilian Weng, *What are Diffusion Models?* (updated 2024)](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) — the canonical long-form tutorial covering DDPM through latent diffusion, flow matching, and consistency models with clean notation.

## Further Reading

- Ho, Jain & Abbeel, *Denoising Diffusion Probabilistic Models* (NeurIPS 2020) — the foundational DDPM paper.
- Song, Meng & Ermon, *Denoising Diffusion Implicit Models* (ICLR 2021) — DDIM, deterministic sampling.
- Song, Sohl-Dickstein, Kingma, Kumar, Ermon & Poole, *Score-Based Generative Modeling through Stochastic Differential Equations* (ICLR 2021) — the SDE unification.
- Dhariwal & Nichol, *Diffusion Models Beat GANs on Image Synthesis* (NeurIPS 2021) — classifier guidance, improved U-Net.
- Ho & Salimans, *Classifier-Free Diffusion Guidance* (NeurIPS Workshop 2021) — CFG.
- Rombach, Blattmann, Lorenz, Esser & Ommer, *High-Resolution Image Synthesis with Latent Diffusion Models* (CVPR 2022) — latent diffusion / Stable Diffusion.
- Lipman, Chen, Ben-Hamu, Nickel & Le, *Flow Matching for Generative Modeling* (ICLR 2023) — conditional flow matching.
- Liu, Gong & Liu, *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow* (ICLR 2023) — rectified flow.
- Peebles & Xie, *Scalable Diffusion Models with Transformers* (DiT, ICCV 2023) — Transformer denoiser, used in Sora-like systems.
- Austin, Johnson, Ho, Tarlow & van den Berg, *Structured Denoising Diffusion Models in Discrete State-Spaces* (D3PM, NeurIPS 2021) — diffusion over discrete tokens.
- Kingma, Salimans, Poole & Ho, *Variational Diffusion Models* (NeurIPS 2021) — the ELBO in SNR form; shows loss weighting and noise schedule are the same knob.
- Salimans & Ho, *Progressive Distillation for Fast Sampling of Diffusion Models* (ICLR 2022) — introduces $\mathbf{v}$-prediction and halving-the-steps distillation.
- Karras, Aittala, Aila & Laine, *Elucidating the Design Space of Diffusion-Based Generative Models* (EDM, NeurIPS 2022) — disentangles schedule, preconditioning, weighting and solver; the reference for practical sampler design.
- Lu et al., *DPM-Solver++* (2022) — high-order exponential-integrator solvers that cut step counts to ~10–20.
- Lin, Liu, Li & Yang, *Common Diffusion Noise Schedules and Sample Steps Are Flawed* (WACV 2024) — the zero-terminal-SNR fix.
- Song, Dhariwal, Chen & Sutskever, *Consistency Models* (ICML 2023) — one- and few-step generation by distilling the probability-flow ODE.
- Saharia et al., *Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding* (Imagen, NeurIPS 2022) — dynamic thresholding and cascaded diffusion.

!!! key "Key Takeaways"
    - Diffusion models define a forward noising process with a closed-form marginal $q(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\mathbf{x}_0,(1-\bar\alpha_t)\mathbf{I})$, enabling any timestep to be sampled in one shot.
    - The DDPM training objective reduces to simple noise-prediction regression: $\mathbb{E}\|\boldsymbol\epsilon - \boldsymbol\epsilon_\theta(\mathbf{x}_t,t)\|^2$. Stability comes from regression; no adversarial training required.
    - Score matching and diffusion are equivalent — the noise predictor $\boldsymbol\epsilon_\theta$ is a re-scaled score of the noisy distribution — and $\boldsymbol\epsilon$-, $\mathbf{x}_0$-, score- and $\mathbf{v}$-prediction are all one model under linear reparameterization, yet different *objectives*, since each implies a different per-timestep weighting; $\mathbf{v}$-prediction plus a zero-terminal-SNR schedule is the fix for the medium-grey brightness bias of the classic schedules.
    - DDIM reformulates sampling as ODE integration, enabling deterministic, subsampable trajectories (20–50 steps instead of 1000); higher-order solvers (DPM-Solver++, UniPC, EDM/Heun) reach the same quality in 10–20, and going below ~4 steps requires distillation rather than a better integrator.
    - Classifier-free guidance (CFG) steers generation by extrapolating beyond the conditional model: $\tilde{\boldsymbol\epsilon} = \boldsymbol\epsilon_\emptyset + w(\boldsymbol\epsilon_\mathbf{c} - \boldsymbol\epsilon_\emptyset)$ with $w>1$. Higher $w$ improves prompt adherence but risks over-saturation.
    - Latent diffusion (Stable Diffusion) runs the diffusion process in the compressed latent space of a VAE, reducing spatial resolution 8× and compute $\sim$64× versus pixel space.
    - Flow matching (Rectified Flow, CFM) is a cleaner framework: train the network to predict the straight-line velocity $\mathbf{x}_1 - \mathbf{x}_0$; requires fewer ODE steps and is now the backbone of SD3 and Flux.
    - Diffusion LMs apply masking as the noise process and generate text in parallel (all positions at once), offering an alternative to strict left-to-right autoregressive generation with natural infilling support.
    - As an LLM engineer you need diffusion literacy for multimodal architectures, RLHF-for-diffusion, serving system design, and breadth interview questions.

## Exercises

**1.** (Conceptual) The forward marginal is $q(\mathbf{x}_t\mid\mathbf{x}_0)=\mathcal{N}(\sqrt{\bar\alpha_t}\,\mathbf{x}_0,\,(1-\bar\alpha_t)\mathbf{I})$. Explain (a) why the *signal* coefficient is $\sqrt{\bar\alpha_t}$ and the *noise* coefficient is $\sqrt{1-\bar\alpha_t}$ rather than, say, $\bar\alpha_t$ and $1-\bar\alpha_t$; and (b) why the schedule is engineered so that $\bar\alpha_T\approx 0$. What would go wrong at sampling time if $\bar\alpha_T$ were, say, $0.5$ instead?

??? note "Solution"
    (a) Under the one-shot reparameterization $\mathbf{x}_t=\sqrt{\bar\alpha_t}\,\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon$ with $\boldsymbol\epsilon\sim\mathcal{N}(\mathbf0,\mathbf I)$, the *variance* of a scaled Gaussian scales with the *square* of the coefficient. So the coefficient $\sqrt{1-\bar\alpha_t}$ contributes variance $(1-\bar\alpha_t)$, matching the covariance $(1-\bar\alpha_t)\mathbf I$ in the marginal. The choice $\sqrt{\bar\alpha_t}$ for the signal and $\sqrt{1-\bar\alpha_t}$ for the noise makes the two variance contributions sum to $\bar\alpha_t\cdot\mathrm{Var}(\mathbf{x}_0)+(1-\bar\alpha_t)$; when the data is standardized to unit variance this keeps $\mathrm{Var}(\mathbf{x}_t)\approx 1$ at every $t$ (a *variance-preserving* process). Using $\bar\alpha_t$ and $1-\bar\alpha_t$ directly as coefficients would make the total variance shrink toward $\bar\alpha_t^2+(1-\bar\alpha_t)^2<1$ and break that invariant.

    (b) Sampling starts from $\mathbf{x}_T\sim\mathcal{N}(\mathbf0,\mathbf I)$ — pure noise, containing no information about any particular $\mathbf{x}_0$. The marginal $q(\mathbf{x}_T\mid\mathbf{x}_0)=\mathcal{N}(\sqrt{\bar\alpha_T}\,\mathbf{x}_0,(1-\bar\alpha_T)\mathbf I)$ matches that prior only when $\bar\alpha_T\approx0$, so that $\sqrt{\bar\alpha_T}\,\mathbf{x}_0\approx\mathbf0$ and the covariance $\approx\mathbf I$ regardless of $\mathbf{x}_0$. If instead $\bar\alpha_T=0.5$, the true endpoint of the forward chain would still retain $\sqrt{0.5}\approx0.71$ of the original signal, but the sampler *ignores* this and initializes from $\mathcal{N}(\mathbf0,\mathbf I)$. This train/test mismatch means the network is fed a $\mathbf{x}_T$ from a distribution it never saw during training, producing biased or degraded samples — the reverse process would have to "hallucinate" the missing $0.71\,\mathbf{x}_0$ of structure it was never trained to reconstruct from noise alone.

**2.** (Quantitative) Consider a *constant* noise schedule $\beta_t=0.01$ for all $t$, so $\alpha_t=0.99$ and $\bar\alpha_t=0.99^{\,t}$. (a) At what timestep $t$ does the signal energy fall to half, i.e. $\bar\alpha_t=0.5$? (b) Write out the numeric coefficients of $\mathbf{x}_0$ and $\boldsymbol\epsilon$ at that step. (c) What is the signal-to-noise ratio $\mathrm{SNR}(t)=\bar\alpha_t/(1-\bar\alpha_t)$ there, and how does it compare to the chapter's linear-schedule value at $t=500$?

??? note "Solution"
    (a) Solve $0.99^{\,t}=0.5$. Taking logs, $t=\dfrac{\ln 0.5}{\ln 0.99}=\dfrac{-0.6931}{-0.010050}\approx 68.97$, so $t\approx 69$. (Cross-check with the chapter's exponential approximation $\bar\alpha_t\approx\exp(-\sum\beta_s)=e^{-0.01t}$: setting $e^{-0.01t}=0.5$ gives $t=\ln 2/0.01=69.3$ — consistent.)

    (b) At $t=69$, $\bar\alpha_t\approx0.5$, so
    $$\mathbf{x}_{69}=\sqrt{0.5}\,\mathbf{x}_0+\sqrt{0.5}\,\boldsymbol\epsilon\approx 0.707\,\mathbf{x}_0+0.707\,\boldsymbol\epsilon.$$
    Signal and noise contribute equally.

    (c) $\mathrm{SNR}(69)=\dfrac{0.5}{1-0.5}=1.0$ (equivalently $0\,$dB). The chapter's linear schedule gives $\bar\alpha_{500}\approx0.08$ and $\mathrm{SNR}(500)\approx0.087$ — more than $10\times$ noisier. So the halfway point of *this* gentle constant schedule ($t\approx69$) is far less corrupted than the linear DDPM schedule is at its own midpoint $t=500$, illustrating how aggressively the standard linear schedule destroys structure early on.

**3.** (Quantitative) At one denoising step a text-to-image model produces an unconditional noise prediction $\boldsymbol\epsilon_\emptyset=(0.20,\,-0.10)$ and a conditional one $\boldsymbol\epsilon_\mathbf{c}=(0.50,\,0.30)$ for a two-pixel toy latent. Using classifier-free guidance $\tilde{\boldsymbol\epsilon}=\boldsymbol\epsilon_\emptyset+w(\boldsymbol\epsilon_\mathbf{c}-\boldsymbol\epsilon_\emptyset)$: (a) compute $\tilde{\boldsymbol\epsilon}$ for $w=7.5$; (b) give $\tilde{\boldsymbol\epsilon}$ for $w=0$ and $w=1$ and say what each corresponds to; (c) comparing the norm of $\tilde{\boldsymbol\epsilon}$ at $w=7.5$ to that of $\boldsymbol\epsilon_\mathbf{c}$, explain in one sentence why large $w$ is described as "extrapolation beyond the conditional distribution."

??? note "Solution"
    Let $\mathbf{d}=\boldsymbol\epsilon_\mathbf{c}-\boldsymbol\epsilon_\emptyset=(0.50-0.20,\;0.30-(-0.10))=(0.30,\,0.40)$, with $\|\mathbf{d}\|=\sqrt{0.30^2+0.40^2}=0.5$.

    (a) $\tilde{\boldsymbol\epsilon}=\boldsymbol\epsilon_\emptyset+7.5\,\mathbf{d}=(0.20+7.5\cdot0.30,\;-0.10+7.5\cdot0.40)=(0.20+2.25,\;-0.10+3.00)=(2.45,\,2.90)$.

    (b) $w=0$: $\tilde{\boldsymbol\epsilon}=\boldsymbol\epsilon_\emptyset=(0.20,-0.10)$ — the purely *unconditional* prediction. $w=1$: $\tilde{\boldsymbol\epsilon}=\boldsymbol\epsilon_\mathbf{c}=(0.50,0.30)$ — the pure *conditional* model, no guidance.

    (c) $\|\tilde{\boldsymbol\epsilon}\|_{w=7.5}=\sqrt{2.45^2+2.90^2}=\sqrt{6.00+8.41}=\sqrt{14.41}\approx3.80$, versus $\|\boldsymbol\epsilon_\mathbf{c}\|=\sqrt{0.25+0.09}=\sqrt{0.34}\approx0.58$. The guided vector is $\sim6.6\times$ longer and points well past the conditional point along the direction $\mathbf{d}$ — the model is pushed *further* than the conditional prediction itself, i.e. outside the region the conditional model would ever output, which is exactly the extrapolation that sharpens prompt adherence but risks over-saturation at large $w$.

**4.** (Implementation) The chapter's `ddim_sample` is unconditional. Modify it into a `ddim_sample_cfg` that accepts a conditioning tensor `c`, a null embedding `null_c`, and a guidance scale `w`, applying classifier-free guidance at every step. Assume the model signature is now `model(x_t, t, c)`. Keep everything else (the $\mathbf{x}_0$ prediction, the deterministic $\eta=0$ path) intact.

??? note "Solution"
    We run the denoiser twice per step — once with the real conditioning `c`, once with `null_c` — and combine the two noise predictions with the CFG formula $\tilde{\boldsymbol\epsilon}=\boldsymbol\epsilon_\emptyset+w(\boldsymbol\epsilon_\mathbf{c}-\boldsymbol\epsilon_\emptyset)$ before using $\tilde{\boldsymbol\epsilon}$ everywhere the original sampler used `eps_hat`. Everything downstream is unchanged.

    ```python
    @torch.no_grad()
    def ddim_sample_cfg(model, shape, alphas_bar, c, null_c,
                        w: float = 7.5, num_steps: int = 50,
                        eta: float = 0.0, device: str = "cpu"):
        """DDIM sampler with classifier-free guidance. model(x_t, t, c)."""
        T = alphas_bar.shape[0]
        step_size = T // num_steps
        timesteps = list(reversed(range(0, T, step_size)))

        x = torch.randn(shape, device=device)

        for i, t in enumerate(timesteps):
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            t_batch = torch.full((shape[0],), t, dtype=torch.long, device=device)

            # --- classifier-free guidance: two forward passes ---
            eps_c    = model(x, t_batch, c)
            eps_null = model(x, t_batch, null_c)
            eps_hat  = eps_null + w * (eps_c - eps_null)

            ab_t    = alphas_bar[t]
            ab_prev = alphas_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0)

            # predict x0 from the *guided* noise
            x0_pred = (x - (1 - ab_t).sqrt() * eps_hat) / ab_t.sqrt()
            x0_pred = x0_pred.clamp(-1, 1)

            sigma_t = eta * ((1 - ab_prev) / (1 - ab_t)).sqrt() \
                          * (1 - ab_t / ab_prev).sqrt()
            dir_xt  = (1 - ab_prev - sigma_t**2).sqrt() * eps_hat

            noise = torch.randn_like(x) if eta > 0 and t_prev >= 0 else 0.0
            x = ab_prev.sqrt() * x0_pred + dir_xt + sigma_t * noise

        return x
    ```

    Note the cost: CFG doubles the forward passes per step (matching the chapter's "CFG doubles the number of forward passes per step"). A batched implementation would instead concatenate `c` and `null_c` along the batch dimension and split the output, doing both predictions in one call.

**5.** (Hard: derivation + implementation) The chapter's `ddpm_loss` uses the *noise-prediction* parameterization. An equivalent choice is *$\mathbf{x}_0$-prediction*, where the network outputs $\hat{\mathbf{x}}_0$ directly. (a) Starting from $\mathbf{x}_t=\sqrt{\bar\alpha_t}\,\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon$, derive the exact linear map between a noise prediction $\hat{\boldsymbol\epsilon}$ and the corresponding $\hat{\mathbf{x}}_0$ (both directions). (b) Explain why the naive MSE loss $\|\mathbf{x}_0-\hat{\mathbf{x}}_0\|^2$ is *not* identical to $\mathcal{L}_\text{simple}$, and give the per-timestep weight that makes them equal. (c) Implement an `x0_pred_loss` in the chapter's style that reproduces $\mathcal{L}_\text{simple}$ exactly.

??? note "Solution"
    (a) Rearranging the forward reparameterization for $\mathbf{x}_0$:
    $$\hat{\mathbf{x}}_0=\frac{\mathbf{x}_t-\sqrt{1-\bar\alpha_t}\,\hat{\boldsymbol\epsilon}}{\sqrt{\bar\alpha_t}},\qquad\text{and inversely}\qquad \hat{\boldsymbol\epsilon}=\frac{\mathbf{x}_t-\sqrt{\bar\alpha_t}\,\hat{\mathbf{x}}_0}{\sqrt{1-\bar\alpha_t}}.$$
    These are the two forms already used in the chapter (the $\mathbf{x}_0$ expression is exactly the "predicted $\mathbf{x}_0$" term inside `ddim_sample` and the DDIM update).

    (b) Substitute the errors. Since $\mathbf{x}_t$ is shared by both the true and predicted quantities, the noise error and the $\mathbf{x}_0$ error differ only by a scalar:
    $$\boldsymbol\epsilon-\hat{\boldsymbol\epsilon}=\frac{-\sqrt{\bar\alpha_t}\,(\mathbf{x}_0-\hat{\mathbf{x}}_0)}{\sqrt{1-\bar\alpha_t}}\quad\Longrightarrow\quad\|\boldsymbol\epsilon-\hat{\boldsymbol\epsilon}\|^2=\frac{\bar\alpha_t}{1-\bar\alpha_t}\,\|\mathbf{x}_0-\hat{\mathbf{x}}_0\|^2=\mathrm{SNR}(t)\,\|\mathbf{x}_0-\hat{\mathbf{x}}_0\|^2.$$
    So a plain $\mathbf{x}_0$-MSE is *not* equal to $\mathcal{L}_\text{simple}$: it is missing the per-timestep factor $\mathrm{SNR}(t)=\bar\alpha_t/(1-\bar\alpha_t)$. Weighting the $\mathbf{x}_0$ loss by $\mathrm{SNR}(t)$ recovers $\mathcal{L}_\text{simple}$ exactly. (Conceptually: at large $t$, $\mathrm{SNR}\to0$ down-weights $\mathbf{x}_0$ errors, because when the input is nearly pure noise, reconstructing $\mathbf{x}_0$ precisely is neither possible nor important; noise-prediction bakes this weighting in automatically.)

    (c) A drop-in variant matching `ddpm_loss`'s interface (`model` now predicts $\hat{\mathbf{x}}_0$):

    ```python
    def x0_pred_loss(model, x0, alphas_bar):
        """x0-prediction loss weighted by SNR(t) == L_simple exactly."""
        B = x0.shape[0]
        T = alphas_bar.shape[0]
        t = torch.randint(0, T, (B,), device=x0.device)
        x_t, eps = q_sample(x0, t, alphas_bar)      # reuse chapter helper

        x0_hat = model(x_t, t)                        # network predicts x0

        ab  = alphas_bar[t].view(-1, 1, 1, 1)
        snr = ab / (1.0 - ab)                         # SNR(t), broadcast
        return (snr * (x0 - x0_hat).pow(2)).mean()
    ```

    With the $\mathrm{SNR}(t)$ weight, `x0_pred_loss` and `ddpm_loss` optimize the same objective (they differ only by which quantity the network's head emits); dropping the `snr` factor gives the common *unweighted* $\mathbf{x}_0$-prediction loss, which is a legitimate but different objective that emphasizes high-noise timesteps less.
