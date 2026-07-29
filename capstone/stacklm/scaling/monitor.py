"""Use the ladder as a LIVE monitor for the flagship run (Ch. 14.5).

The fitted law predicts the final, LR-decayed loss; two hours into the run you
are mid-stable-phase at full LR and sitting well above it. Two corrections make
the comparison legitimate, and both are free if you saved the ladder's curves.
"""
import numpy as np


def measure_decay_drop(curves, decay_frac: float = 0.20) -> float:
    """Median (loss at start of decay) - (final loss) across the ladder rungs.
    `curves` is a list of (tokens, val_loss) sequences from ladder_results.jsonl.
    Expect a tenth of a nat at this scale -- but YOUR ladder tells you."""
    drops = []
    for tok, loss in curves:
        tok, loss = np.asarray(tok, float), np.asarray(loss, float)
        i = int(np.searchsorted(tok, tok[-1] * (1.0 - decay_frac)))
        i = min(i, len(loss) - 1)
        drops.append(loss[i] - loss[-1])
    return float(np.median(drops))


def project_final_loss(tokens, losses, total_tokens, beta, decay_drop,
                       warmup_skip: float = 0.25) -> float:
    """Project a live STABLE-PHASE curve to the run's final, decayed loss.

    Within the stable phase the loss falls like the law's data term, so it is
    linear in D^-beta with the beta YOUR ladder measured. Fit a line in that
    coordinate, extrapolate to the full budget, subtract the decay drop.

    warmup_skip drops the first fraction of points: the warmup transient is not
    on the power law and will drag the line.
    """
    t, l = np.asarray(tokens, float), np.asarray(losses, float)
    keep = t > t[-1] * warmup_skip
    x = t[keep] ** (-beta)
    slope, intercept = np.polyfit(x, l[keep], 1)
    return float(intercept + slope * total_tokens ** (-beta) - decay_drop)
