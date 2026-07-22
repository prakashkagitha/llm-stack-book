"""
Runs the CPU-runnable Python blocks from
content/10-multimodal-and-arch/04-diffusion-generative.md, concatenated in
order so that later blocks can rely on names defined by earlier ones (as
they do in the chapter itself). Each block is copied verbatim from the
chapter; minimal glue needed to actually EXECUTE the defined
functions/classes (tiny fixture tensors, small T, small spatial size) is
added below and clearly marked "GLUE".

Blocks covered (2 CPU-runnable blocks per task spec):
  #0 (line ~90)  - Minimal DDPM: make_linear_schedule, q_sample, ddpm_loss,
                    ddpm_sample (forward process + training loss + ancestral
                    reverse sampler).
  #2 (line ~313) - SinusoidalTimeEmbedding, ResBlock, TinyUNet: a minimal
                    2-level U-Net noise predictor, used here as the `model`
                    argument to the block #0 functions.

Blocks explicitly SKIPPED (per task spec, default-skip heuristic):
  #1 - SKIP(fragment): ddim_sample (line ~223). Flagged as a non-standalone
                        fragment by the chapter's block heuristic. It reuses
                        the same alphas_bar/model machinery already exercised
                        via block #0's ddpm_sample, so skipping it does not
                        reduce coverage of anything new; left untested per
                        the default-skip rule for fragments.
  #3 - SKIP(fragment): flow_matching_loss / flow_matching_sample (line ~423).
                        A separate generative paradigm (flow matching) that
                        is self-contained but flagged as a fragment (it does
                        not import torch/nn itself and stands alone from the
                        DDPM/U-Net code exercised above). Left untested per
                        the default-skip rule.

No network calls, no GPU required. Runtime is a few seconds on CPU (T=50
diffusion steps through a tiny 2-level U-Net on 8x8 "latents").

No bugs were found in the tested blocks' code -- both ran correctly as
written once given a real nn.Module (TinyUNet) as the `model` argument.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ======================================================================
# Block #0 (line ~90): Minimal DDPM -- forward process + training loss
# ======================================================================

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
    # gather ā_t for each sample in the batch
    ab = alphas_bar[t].view(-1, 1, 1, 1)   # broadcast over C,H,W
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


# ======================================================================
# Block #2 (line ~313): SinusoidalTimeEmbedding, ResBlock, TinyUNet
# ======================================================================

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


# ======================================================================
# GLUE: exercise the blocks above end-to-end on CPU with tiny fixtures.
# ======================================================================

def main():
    # ---- Block #0: forward process (make_linear_schedule + q_sample) ----
    # GLUE: T=50 (tiny schedule) instead of the book's illustrative T=1000,
    # purely to keep the 1000-step ancestral sampler fast on CPU; the
    # schedule/formulas themselves are untouched.
    T = 50
    betas, alphas, alphas_bar = make_linear_schedule(T)
    assert betas.shape == (T,) and alphas.shape == (T,) and alphas_bar.shape == (T,)
    assert torch.all(alphas_bar[1:] <= alphas_bar[:-1])  # monotonically decreasing

    # GLUE: tiny "latent" batch standing in for (B, C, H, W) images in [-1, 1]
    B, C, H, W = 2, 4, 8, 8
    x0 = torch.rand(B, C, H, W) * 2 - 1  # in [-1, 1]
    t = torch.randint(0, T, (B,))
    x_t, eps = q_sample(x0, t, alphas_bar)
    assert x_t.shape == x0.shape
    assert eps.shape == x0.shape

    # Sanity-check the closed-form marginal directly against the formula in
    # the chapter: x_t = sqrt(abar_t) * x0 + sqrt(1 - abar_t) * eps
    ab = alphas_bar[t].view(-1, 1, 1, 1)
    expected_x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
    assert torch.allclose(x_t, expected_x_t)

    # ---- Block #2: TinyUNet as the eps-predicting model ----
    # GLUE: small base_ch/time_dim to keep the model tiny and fast on CPU.
    model = TinyUNet(in_channels=C, base_ch=8, time_dim=16)
    model.eval()

    # exercise SinusoidalTimeEmbedding + ResBlock directly, not just via TinyUNet
    temb_module = SinusoidalTimeEmbedding(16)
    temb = temb_module(t)
    assert temb.shape == (B, 16)

    res = ResBlock(channels=8, time_emb_dim=16)
    dummy_feat = torch.randn(B, 8, H, W)
    res_out = res(dummy_feat, temb)
    assert res_out.shape == dummy_feat.shape

    eps_hat = model(x_t, t)
    assert eps_hat.shape == x0.shape
    assert torch.isfinite(eps_hat).all()

    # ---- Block #0: training loss, using TinyUNet as `model` ----
    loss = ddpm_loss(model, x0, alphas_bar)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    print(f"ddpm_loss = {loss.item():.4f}")

    # ---- Block #0: reverse ancestral sampler, using TinyUNet as `model` ----
    samples = ddpm_sample(model, (B, C, H, W), betas, alphas, alphas_bar, device="cpu")
    assert samples.shape == (B, C, H, W)
    assert torch.isfinite(samples).all()
    print(f"ddpm_sample output stats: mean={samples.mean().item():.4f} "
          f"std={samples.std().item():.4f}")

    print("ALL OK")


if __name__ == "__main__":
    main()
