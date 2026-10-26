"""Small statistics helpers: bootstrap confidence intervals and paired run comparison."""
import numpy as np


def bootstrap_ci(values, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Mean and (1-alpha) percentile bootstrap CI, ignoring NaNs. Returns (mean, lo, hi)."""
    x = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def paired_diff_ci(a, b, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Mean per-question difference (b - a) with bootstrap CI, on questions where both runs have a score.

    More sensitive than comparing two independent means, because question difficulty cancels out.
    An interval that excludes 0 is evidence of a real change.
    """
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None and not np.isnan(x) and not np.isnan(y)]
    if not pairs:
        return float("nan"), float("nan"), float("nan")
    return bootstrap_ci([y - x for x, y in pairs], n_boot=n_boot, alpha=alpha, seed=seed)
