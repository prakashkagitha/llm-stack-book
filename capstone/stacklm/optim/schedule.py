"""Learning-rate schedules (Ch. 14.6). WSD (Warmup-Stable-Decay, MiniCPM/Hu et
al. 2024) is the default; the decay phase is where mid-training anneals on premium
data. A cosine schedule is included for contrast.
"""
import math


def wsd_lr(step: int, *, peak_lr: float, warmup_steps: int, total_steps: int,
           decay_frac: float = 0.2, final_frac: float = 0.0) -> float:
    """Warmup -> constant "stable" -> sqrt decay to `final_frac * peak_lr`."""
    decay_steps = int(decay_frac * total_steps)
    stable_end = total_steps - decay_steps
    if step < warmup_steps:                              # warmup
        return peak_lr * (step + 1) / warmup_steps
    if step < stable_end:                                # stable
        return peak_lr
    progress = (step - stable_end) / max(1, decay_steps)  # decay
    decay_mult = 1.0 - math.sqrt(min(progress, 1.0))      # 1 -> 0
    floor = final_frac * peak_lr
    return floor + (peak_lr - floor) * decay_mult


def wsd_decay_multiplier(mid_step: int, num_decay_steps: int,
                         final_frac: float = 0.0, shape: str = "1-sqrt") -> float:
    """The decay-phase multiplier used by mid-training (Ch. 14.8)."""
    t = min(mid_step, num_decay_steps) / max(1, num_decay_steps)   # progress in [0, 1]
    if shape == "1-sqrt":
        decayed = 1.0 - math.sqrt(t)
    elif shape == "linear":
        decayed = 1.0 - t
    elif shape == "cosine":
        decayed = 0.5 * (1.0 + math.cos(math.pi * t))
    else:
        raise ValueError(shape)
    return final_frac + (1.0 - final_frac) * decayed


def cosine_lr(step: int, *, peak_lr: float, warmup_steps: int, total_steps: int,
              final_frac: float = 0.1) -> float:
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(progress, 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    floor = final_frac * peak_lr
    return floor + (peak_lr - floor) * cos
