"""Point-forecast metrics and the multivariate Diebold-Mariano test used in the EPF literature."""
from __future__ import annotations

import numpy as np
from scipy import stats


def mae(y, f):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(f))))


def rmse(y, f):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(f)) ** 2)))


def smape(y, f):
    y, f = np.asarray(y, float), np.asarray(f, float)
    den = (np.abs(y) + np.abs(f)) / 2
    m = den > 1e-9
    return float(100 * np.mean(np.abs(y - f)[m] / den[m]))


def rmae(y, f, f_naive):
    """MAE relative to the naive benchmark (Lago et al. 2021): < 1 means better than naive."""
    return mae(y, f) / mae(y, f_naive)


def dm_test(y, f_a, f_b, norm: int = 1) -> tuple[float, float]:
    """Multivariate DM test on daily loss differentials (Ziel & Weron 2018; Lago et al. 2021).

    y, f_a, f_b: arrays shaped (days, 24). Loss of a day = ||error||_norm over its 24 hours.
    H0: E[L_a - L_b] <= 0;  H1: forecast b is MORE accurate than a (one-sided).
    Returns (DM statistic, p-value). A small p-value means b significantly beats a."""
    y, f_a, f_b = (np.asarray(v, float) for v in (y, f_a, f_b))
    la = np.linalg.norm(y - f_a, ord=norm, axis=1)
    lb = np.linalg.norm(y - f_b, ord=norm, axis=1)
    d = la - lb
    n = len(d)
    stat = d.mean() / (d.std(ddof=0) / np.sqrt(n))
    return float(stat), float(1 - stats.norm.cdf(stat))
