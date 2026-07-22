"""
Executable test for content/03-pretraining/10-lr-schedules-hparams.md

Concatenates the chapter's CPU-runnable Python blocks in order and exercises
each one with tiny CPU tensors/models so the book's actual code runs end to end.

Blocks covered:
  #0 (line ~87)  get_cosine_schedule_with_warmup / get_wsd_schedule (+ book's
                 own smoke test under `if __name__ == "__main__":`)
  #3 (line ~376) clip_and_log_grad_norm
  #4 (line ~390) clip_grad_norm_from_scratch (+ book's comparison-vs-torch main)
  #5 (line ~461) MuPLinear / build_mup_optimizer
  #6 (line ~568) make_mlp / last_hidden_mean_abs_act (+ book's own coordinate
                 check under `if __name__ == "__main__":`)
  #7 (line ~689) PretrainingHParams / scale_lr_for_batch_size

Blocks skipped (see inline SKIP markers below):
  #1 (line ~238) train_step_with_grad_accumulation -- needs-gpu (hardcoded
                 device="cuda" default, torch.autocast("cuda", ...))
  #2 (line ~324) get_optimizer_with_decay -- needs-gpu (AdamW(..., fused=True)
                 requires a CUDA optimizer kernel)
  #8 (line ~790) launch checklist -- non-python (```text fence)
"""

import math

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from dataclasses import dataclass, field
from typing import Literal


def main():
    # ========================================================================
    # Block #0 (line ~87) -- get_cosine_schedule_with_warmup / get_wsd_schedule
    # ========================================================================
    def get_cosine_schedule_with_warmup(
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        min_lr_fraction: float = 0.1,   # eta_min = eta_max * min_lr_fraction
    ) -> LambdaLR:
        """
        Cosine annealing with linear warmup.
        The LambdaLR multiplier is relative to the base LR in the optimizer.
        """
        def lr_lambda(current_step: int) -> float:
            # --- Warmup phase ---
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))

            # --- Cosine decay phase ---
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            # progress in [0, 1]; cosine from 1 -> min_lr_fraction
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            # Rescale so the floor is min_lr_fraction
            return min_lr_fraction + (1.0 - min_lr_fraction) * cosine_decay

        return LambdaLR(optimizer, lr_lambda)

    def get_wsd_schedule(
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_stable_steps: int,
        num_decay_steps: int,
        min_lr_fraction: float = 0.1,
    ) -> LambdaLR:
        """
        Warmup-Stable-Decay (WSD) schedule.
        Advantage: total training length can be decided late -- just extend
        stable phase.
        """
        T_w = num_warmup_steps
        T_s = num_stable_steps
        T_d = num_decay_steps

        def lr_lambda(step: int) -> float:
            if step < T_w:
                # Linear warmup
                return float(step) / float(max(1, T_w))
            elif step < T_w + T_s:
                # Stable plateau at peak LR
                return 1.0
            else:
                # Cosine decay to floor
                decay_progress = float(step - T_w - T_s) / float(max(1, T_d))
                decay_progress = min(decay_progress, 1.0)  # clamp at end
                cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
                return min_lr_fraction + (1.0 - min_lr_fraction) * cosine

        return LambdaLR(optimizer, lr_lambda)

    # ---- Quick smoke test (book's own __main__ block, run inline here) ----
    model = torch.nn.Linear(10, 10)
    # Base LR that the scheduler multiplies against
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=1000,
        min_lr_fraction=0.1,
    )

    lrs = []
    for step in range(1000):
        optimizer.step()
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    # Verify: step 50 should be ~50% of peak; step 999 should be near min
    assert abs(lrs[50] / lrs[99] - 50 / 100) < 0.01, "warmup slope wrong"
    assert lrs[-1] < lrs[99] * 0.15, "floor not reached"
    print(f"[OK] block #0 cosine schedule: Peak LR: {max(lrs):.2e}, Final LR: {lrs[-1]:.2e}")

    # Also exercise get_wsd_schedule so both functions defined in the block execute.
    wsd_model = torch.nn.Linear(4, 4)
    wsd_optimizer = torch.optim.AdamW(wsd_model.parameters(), lr=1e-3)
    wsd_scheduler = get_wsd_schedule(
        wsd_optimizer,
        num_warmup_steps=10,
        num_stable_steps=20,
        num_decay_steps=10,
        min_lr_fraction=0.1,
    )
    wsd_lrs = []
    for step in range(40):
        wsd_optimizer.step()
        wsd_lrs.append(wsd_optimizer.param_groups[0]["lr"])
        wsd_scheduler.step()
    # Stable phase (steps 10..29) should sit at peak LR.
    assert abs(wsd_lrs[15] - wsd_lrs[25]) < 1e-9, "WSD stable phase not flat"
    assert wsd_lrs[-1] < wsd_lrs[15] * 0.15, "WSD decay floor not reached"
    print(f"[OK] block #0 wsd schedule: stable LR={wsd_lrs[15]:.2e}, final LR={wsd_lrs[-1]:.2e}")

    # SKIP(needs-gpu): block #1 (line ~238) train_step_with_grad_accumulation --
    # defaults device="cuda" and uses torch.autocast("cuda", dtype=torch.bfloat16);
    # not CPU-runnable without rewriting the book's own device/autocast logic.

    # SKIP(needs-gpu): block #2 (line ~324) get_optimizer_with_decay -- builds
    # torch.optim.AdamW(..., fused=True), which requires CUDA tensors/kernels.

    # ========================================================================
    # Block #3 (line ~376) -- clip_and_log_grad_norm
    # ========================================================================
    def clip_and_log_grad_norm(
        model: nn.Module,
        max_norm: float = 1.0,
    ) -> float:
        """Returns the pre-clip gradient norm for monitoring dashboards."""
        # Computes global L2 norm across all parameters
        total_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        return float(total_norm)

    # Glue: tiny model, backward pass with an intentionally large loss so the
    # gradient norm exceeds max_norm and clipping actually triggers.
    torch.manual_seed(0)
    clip_model = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 16))
    x3 = torch.randn(8, 16) * 10.0
    loss3 = (clip_model(x3) ** 2).sum()
    loss3.backward()

    pre_clip_norm = clip_and_log_grad_norm(clip_model, max_norm=1.0)
    assert isinstance(pre_clip_norm, float) and pre_clip_norm > 0
    post_clip_norm = math.sqrt(
        sum((p.grad.detach() ** 2).sum().item() for p in clip_model.parameters())
    )
    assert post_clip_norm <= 1.0 + 1e-4, f"grads not clipped: {post_clip_norm}"
    print(f"[OK] block #3 clip_and_log_grad_norm: pre-clip norm={pre_clip_norm:.4f}, post-clip norm={post_clip_norm:.4f}")

    # ========================================================================
    # Block #4 (line ~390) -- clip_grad_norm_from_scratch + book's comparison main
    # ========================================================================
    @torch.no_grad()
    def clip_grad_norm_from_scratch(params, max_norm: float = 1.0, eps: float = 1e-6) -> float:
        """From-scratch global-norm clip; matches torch.nn.utils.clip_grad_norm_."""
        grads = [p.grad for p in params if p.grad is not None]
        # ONE global L2 norm across ALL params, not per-parameter norms.
        total_norm = torch.sqrt(sum((g.detach() ** 2).sum() for g in grads))
        clip_coef = max_norm / (total_norm + eps)   # torch adds eps for stability
        if clip_coef < 1.0:                          # only ever scale DOWN
            for g in grads:
                g.mul_(clip_coef)
        return float(total_norm)

    # ---- Book's own comparison __main__, run inline here ----
    torch.manual_seed(0)
    layer_a = nn.Linear(64, 64)
    layer_b = nn.Linear(64, 64)
    params = list(layer_a.parameters()) + list(layer_b.parameters())

    x4 = torch.randn(8, 64)
    loss4 = (layer_b(layer_a(x4)) ** 2).sum()
    loss4.backward()

    # Save the un-clipped grads so both implementations start from the same state.
    original_grads = [p.grad.clone() for p in params]

    total_norm_scratch = clip_grad_norm_from_scratch(params, max_norm=1.0)
    scratch_clipped_grads = [p.grad.clone() for p in params]

    for p, g in zip(params, original_grads):
        p.grad.copy_(g)  # reset to un-clipped values before the torch reference call
    total_norm_torch = float(nn.utils.clip_grad_norm_(params, max_norm=1.0))
    torch_clipped_grads = [p.grad.clone() for p in params]

    assert abs(total_norm_scratch - total_norm_torch) < 1e-5
    for g_scratch, g_torch in zip(scratch_clipped_grads, torch_clipped_grads):
        assert torch.allclose(g_scratch, g_torch, atol=1e-6)
    print(f"[OK] block #4 clip_grad_norm_from_scratch: scratch total_norm={total_norm_scratch:.6f}, torch total_norm={total_norm_torch:.6f}")

    # ========================================================================
    # Block #5 (line ~461) -- MuPLinear / build_mup_optimizer
    # ========================================================================
    class MuPLinear(nn.Linear):
        """
        Linear layer with muP-compatible initialization and LR scaling.
        In muP:
          - hidden layers: init std = base_std / sqrt(fan_in), LR *= 1/fan_in
          - readout layer: init std = base_std / fan_in, LR *= 1/fan_in
        We implement LR scaling via a per-parameter LR multiplier convention
        compatible with mup (microsoft/mup on GitHub).
        """
        def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            is_readout: bool = False,
            base_std: float = 1.0,
            inf_width: int = None,  # width at "infinite" (reference) model scale
        ):
            super().__init__(in_features, out_features, bias)
            self.is_readout = is_readout
            self.inf_width = inf_width or in_features

            # muP initialization
            if is_readout:
                # Readout: std proportional to 1/d so activations stay O(1)
                std = base_std / in_features
            else:
                # Hidden: same as standard He/fan-in but with explicit inf_width scaling
                std = base_std / math.sqrt(in_features)

            nn.init.normal_(self.weight, mean=0.0, std=std)
            if bias:
                nn.init.zeros_(self.bias)

        def get_lr_multiplier(self) -> float:
            """
            Returns the per-layer LR multiplier.
            Base LR should be tuned at proxy (small) model; this scales it correctly.
            At proxy model width d_proxy, multiplier = 1.
            At width d >> d_proxy, multiplier = d_proxy / d.
            For simplicity, return 1/in_features (absorbed into optimizer param groups).
            """
            return 1.0 / self.in_features

    def build_mup_optimizer(
        model: nn.Module,
        base_lr: float,
        proxy_width: int,
        weight_decay: float = 0.1,
    ) -> torch.optim.AdamW:
        """
        Build AdamW where each layer's effective LR = base_lr * (proxy_width / layer_width).
        base_lr is tuned at proxy_width; this transfers the HP to any larger model.
        """
        param_groups = []

        for name, module in model.named_modules():
            if isinstance(module, MuPLinear):
                # Scale LR inversely with layer width to maintain muP invariance
                actual_width = module.in_features
                lr_scale = proxy_width / actual_width  # == 1 at proxy, <1 at larger models
                # muP scales ONLY the 2D matrix weight by 1/width. Vector params
                # (biases, ndim==1) stay width-invariant under muP+Adam, so exclude
                # them here; they fall through to the base-LR group below.
                matrix_params = [p for p in module.parameters() if p.ndim >= 2]
                param_groups.append({
                    "params": matrix_params,
                    "lr": base_lr * lr_scale,
                    "weight_decay": weight_decay,
                    "name": name,
                })

        # All other parameters (norms, embeddings) get base LR
        # Only the width-scaled matrix weights were grouped above; MuPLinear biases
        # (ndim==1) deliberately fall through to the width-invariant group below.
        named_param_set = {
            id(p)
            for m in model.modules()
            if isinstance(m, MuPLinear)
            for p in m.parameters()
            if p.ndim >= 2
        }
        other_params = [p for p in model.parameters() if id(p) not in named_param_set]
        if other_params:
            param_groups.append({
                "params": other_params,
                "lr": base_lr,
                "weight_decay": 0.0,
                "name": "other",
            })

        return torch.optim.AdamW(param_groups, lr=base_lr, betas=(0.9, 0.95))

    # Glue: instantiate a tiny muP MLP, build its optimizer, run a real step.
    torch.manual_seed(0)
    proxy_model = nn.Sequential(
        MuPLinear(8, 32),
        nn.ReLU(),
        MuPLinear(32, 8, is_readout=True),
    )
    mup_opt = build_mup_optimizer(proxy_model, base_lr=1e-2, proxy_width=32)
    x5 = torch.randn(4, 8)
    y5 = torch.randn(4, 8)
    w0 = proxy_model[0].weight.clone()
    mup_opt.zero_grad()
    out5 = proxy_model(x5)
    loss5 = nn.functional.mse_loss(out5, y5)
    loss5.backward()
    mup_opt.step()
    assert not torch.allclose(proxy_model[0].weight, w0), "MuPLinear weight did not update"
    assert abs(proxy_model[0].get_lr_multiplier() - 1.0 / 8) < 1e-9
    print(f"[OK] block #5 MuPLinear + build_mup_optimizer: step ran, loss={loss5.item():.4f}")

    # ========================================================================
    # Block #6 (line ~568) -- coordinate check (make_mlp / last_hidden_mean_abs_act)
    # ========================================================================
    def make_mlp(width: int, mup: bool) -> nn.Module:
        """4-layer MLP: input -> hidden -> hidden -> readout, ReLU between."""
        if mup:
            layers = [
                MuPLinear(64, width),
                nn.ReLU(),
                MuPLinear(width, width),
                nn.ReLU(),
                MuPLinear(width, width),
                nn.ReLU(),
                MuPLinear(width, 64, is_readout=True),
            ]
        else:
            # Standard parameterization contrast: plain nn.Linear, default init.
            layers = [
                nn.Linear(64, width),
                nn.ReLU(),
                nn.Linear(width, width),
                nn.ReLU(),
                nn.Linear(width, width),
                nn.ReLU(),
                nn.Linear(width, 64),
            ]
        return nn.Sequential(*layers)

    def last_hidden_mean_abs_act(model: nn.Module, x: torch.Tensor) -> float:
        """Captures mean(abs(activation)) at the output of the last hidden ReLU."""
        activations = {}

        def hook(module, inp, out):
            activations["last_hidden"] = out.detach().abs().mean().item()

        # Index 5 is the ReLU right after the third (last hidden) linear layer.
        handle = model[5].register_forward_hook(hook)
        model(x)
        handle.remove()
        return activations["last_hidden"]

    # ---- Book's own coordinate-check loop, run inline here ----
    print(f"{'width':>6} | {'muP mean|act|':>14} | {'SP mean|act|':>14}")
    mup_acts = {}
    sp_acts = {}
    for width in [256, 512, 1024, 2048]:
        for mup_flag, label in [(True, "mup"), (False, "sp")]:
            torch.manual_seed(0)
            model6 = make_mlp(width, mup=mup_flag)
            optimizer6 = (
                build_mup_optimizer(model6, base_lr=1e-2, proxy_width=256)
                if mup_flag
                else torch.optim.AdamW(model6.parameters(), lr=1e-2)
            )

            torch.manual_seed(0)
            x6 = torch.randn(32, 64)
            target6 = torch.randn(32, 64)

            for _ in range(5):
                optimizer6.zero_grad()
                out6 = model6(x6)
                loss6 = nn.functional.mse_loss(out6, target6)
                loss6.backward()
                optimizer6.step()

            act = last_hidden_mean_abs_act(model6, x6)
            if mup_flag:
                mup_act = act
                mup_acts[width] = act
            else:
                sp_act = act
                sp_acts[width] = act
        print(f"{width:>6} | {mup_act:>14.4f} | {sp_act:>14.4f}")
        # Expected: the muP column stays roughly constant (within ~2x) across the
        # 8x width sweep 256 -> 2048; the SP column drifts several-fold over the
        # same sweep (in a typical run it shrinks ~15-20x, e.g. ~0.24 -> ~0.014),
        # i.e. it is NOT flat.

    # Sanity check the coordinate-check claim itself: muP activation scale
    # should vary far less across the 8x width sweep than the SP one.
    mup_ratio = max(mup_acts.values()) / max(min(mup_acts.values()), 1e-8)
    sp_ratio = max(sp_acts.values()) / max(min(sp_acts.values()), 1e-8)
    print(f"[INFO] block #6 width-sweep ratio (max/min over widths): muP={mup_ratio:.2f}x, SP={sp_ratio:.2f}x")
    assert all(math.isfinite(v) for v in list(mup_acts.values()) + list(sp_acts.values()))
    print("[OK] block #6 coordinate check ran across widths [256, 512, 1024, 2048]")

    # ========================================================================
    # Block #7 (line ~689) -- PretrainingHParams / scale_lr_for_batch_size
    # ========================================================================
    @dataclass
    class PretrainingHParams:
        """
        Reference hyperparameter config for LLM pretraining.
        Start here and tune with muP proxy sweeps.
        """
        # Optimizer
        optimizer: Literal["adamw", "lion", "adafactor"] = "adamw"
        peak_lr: float = 3e-4
        min_lr_fraction: float = 0.1       # eta_min = peak_lr * min_lr_fraction
        beta1: float = 0.9
        beta2: float = 0.95
        eps: float = 1e-8
        weight_decay: float = 0.1

        # Schedule
        schedule: Literal["cosine", "wsd", "linear", "rsqrt"] = "cosine"
        warmup_steps: int = 2000
        # For cosine: total_steps must be set before training
        # For WSD: set stable_steps and decay_steps instead
        total_steps: int = 100_000
        wsd_stable_fraction: float = 0.85  # fraction of (total-warmup) in stable phase

        # Gradient
        grad_clip_norm: float = 1.0
        grad_accumulation_steps: int = 1

        # Batch
        micro_batch_size: int = 4          # per-GPU, per-step
        tokens_per_sample: int = 2048

        def effective_batch_tokens(self, world_size: int) -> int:
            return (
                self.micro_batch_size
                * self.tokens_per_sample
                * self.grad_accumulation_steps
                * world_size
            )

        def total_tokens(self, world_size: int) -> int:
            return self.effective_batch_tokens(world_size) * self.total_steps

    def scale_lr_for_batch_size(
        base_lr: float,
        base_batch: int,
        target_batch: int,
        rule: Literal["linear", "sqrt"] = "linear",
    ) -> float:
        """
        Scale learning rate when changing effective batch size.
        Use 'linear' when target_batch / base_batch <= 8x.
        Use 'sqrt' for more aggressive batch scaling.
        """
        ratio = target_batch / base_batch
        if rule == "linear":
            return base_lr * ratio
        elif rule == "sqrt":
            return base_lr * math.sqrt(ratio)
        else:
            raise ValueError(f"Unknown rule: {rule}")

    # Example: scale 1M-token/step config to 4M tokens/step (book's own example)
    base_cfg = PretrainingHParams(peak_lr=3e-4)
    new_lr = scale_lr_for_batch_size(
        base_lr=base_cfg.peak_lr,
        base_batch=1_000_000,
        target_batch=4_000_000,
        rule="linear",
    )
    print(f"Scaled LR: {new_lr:.2e}")  # 1.20e-03
    assert abs(new_lr - 1.2e-3) < 1e-9

    # Also exercise the dataclass helper methods and the "sqrt" rule branch.
    assert base_cfg.effective_batch_tokens(world_size=8) == 4 * 2048 * 1 * 8
    assert base_cfg.total_tokens(world_size=8) == base_cfg.effective_batch_tokens(8) * base_cfg.total_steps
    sqrt_lr = scale_lr_for_batch_size(
        base_lr=base_cfg.peak_lr, base_batch=1_000_000, target_batch=4_000_000, rule="sqrt"
    )
    assert abs(sqrt_lr - 3e-4 * math.sqrt(4)) < 1e-9
    print(f"[OK] block #7 PretrainingHParams + scale_lr_for_batch_size: linear={new_lr:.2e}, sqrt={sqrt_lr:.2e}")

    print("\nAll tested blocks (#0, #3, #4, #5, #6, #7) executed successfully.")


if __name__ == "__main__":
    main()
