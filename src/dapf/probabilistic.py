"""Probabilistic day-ahead price forecasts built on top of the stored point forecasts.

All methods are ex ante: the quantiles for delivery day D use only point forecasts and
actual prices of days before D (the D-1 prices are known at D-1 gate closure because they
were set in the D-2 auction). Arrays are shaped (days, 24) for point forecasts and
(days, 24, n_quantiles) for quantile forecasts.

Methods
* ``historical_simulation``: point forecast + empirical quantiles of its own past errors at
  the same hour (rolling window).
* ``conformal_symmetric``: split-conformal style central intervals, point forecast +/- the
  finite-sample-corrected quantile of past absolute errors at the same hour.
* ``qra``: Quantile Regression Averaging (Nowotarski & Weron 2015): per hour and quantile a
  linear quantile regression of the actual price on the LEAR and GBM point forecasts over a
  rolling window, solved exactly as an LP (HiGHS).
* ``conformalize``: CQR step that recalibrates any quantile forecast on its own recent
  interval misses (Romano, Patterson & Candes 2019).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, identity

TAUS = np.round(np.arange(0.05, 0.951, 0.05), 2)  # 19 levels


def _check_window(t: int, window: int) -> None:
    if t < window:
        raise ValueError(f"day index {t} has fewer than {window} days of history")


def historical_simulation(y: np.ndarray, f: np.ndarray, t: int, window: int,
                          taus: np.ndarray = TAUS) -> np.ndarray:
    """Quantiles (24, n_tau) for day t from the errors of days t-window .. t-1."""
    _check_window(t, window)
    err = y[t - window:t] - f[t - window:t]                    # (window, 24)
    q = np.quantile(err, taus, axis=0, method="linear").T      # (24, n_tau)
    return f[t][:, None] + q


def conformal_symmetric(y: np.ndarray, f: np.ndarray, t: int, window: int,
                        taus: np.ndarray = TAUS) -> np.ndarray:
    """Symmetric conformal quantiles: level tau maps to the central (|2 tau - 1|) interval.

    The radius for coverage 1-a is the ceil((n+1)(1-a))-th smallest absolute error of the
    window (standard split-conformal finite-sample correction), capped at the maximum.
    """
    _check_window(t, window)
    a = np.sort(np.abs(y[t - window:t] - f[t - window:t]), axis=0)  # (window, 24)
    n = window
    out = np.empty((24, len(taus)))
    for j, tau in enumerate(taus):
        cov = abs(2 * tau - 1)
        if cov == 0:
            out[:, j] = f[t]
            continue
        k = min(int(np.ceil((n + 1) * cov)), n) - 1
        sign = 1.0 if tau > 0.5 else -1.0
        out[:, j] = f[t] + sign * a[k]
    return out


def quantile_regression(X: np.ndarray, y: np.ndarray, tau: float) -> np.ndarray:
    """Exact linear quantile regression (pinball loss) as an LP. X must include a constant."""
    n, p = X.shape
    # variables: beta+ (p), beta- (p), u+ (n), u- (n)  -> X(b+ - b-) + u+ - u- = y
    c = np.concatenate([np.zeros(2 * p), np.full(n, tau), np.full(n, 1 - tau)])
    Xs = csr_matrix(X)
    A = hstack([Xs, -Xs, identity(n), -identity(n)], format="csr")
    res = linprog(c, A_eq=A, b_eq=y, bounds=(0, None), method="highs")
    if not res.success:
        raise RuntimeError(f"quantile regression failed: {res.message}")
    return res.x[:p] - res.x[p:2 * p]


def qra(y: np.ndarray, inputs: list[np.ndarray], t: int, window: int,
        taus: np.ndarray = TAUS) -> np.ndarray:
    """QRA quantiles (24, n_tau) for day t, per hour, sorted to remove quantile crossing."""
    _check_window(t, window)
    out = np.empty((24, len(taus)))
    for h in range(24):
        Xw = np.column_stack([np.ones(window)] + [f[t - window:t, h] for f in inputs])
        x_t = np.concatenate([[1.0], [f[t, h] for f in inputs]])
        yw = y[t - window:t, h]
        for j, tau in enumerate(taus):
            out[h, j] = x_t @ quantile_regression(Xw, yw, float(tau))
    return np.sort(out, axis=1)


# ---------------------------------------------------------------- scores

def pinball(y: np.ndarray, q: np.ndarray, taus: np.ndarray = TAUS) -> np.ndarray:
    """Pinball loss per observation and quantile; y (...), q (..., n_tau)."""
    d = y[..., None] - q
    return np.maximum(taus * d, (taus - 1) * d)


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))


def winkler(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float) -> float:
    """Interval (Winkler) score of a central (1-alpha) interval; lower is better."""
    w = hi - lo
    w = w + (2 / alpha) * np.maximum(lo - y, 0) + (2 / alpha) * np.maximum(y - hi, 0)
    return float(np.mean(w))


def prob_below(q: np.ndarray, threshold: float = 0.0, taus: np.ndarray = TAUS) -> np.ndarray:
    """P(price < threshold) by linear interpolation of the quantile function.

    Below the lowest / above the highest stored quantile the tail mass is split in half
    (0.025 / 0.975 for the 5-95 % grid): an approximation, stated in the README.
    """
    lo_p, hi_p = taus[0] / 2, 1 - (1 - taus[-1]) / 2
    flat = q.reshape(-1, q.shape[-1])
    p = np.array([np.interp(threshold, row, taus, left=lo_p, right=hi_p) for row in flat])
    p[flat[:, 0] > threshold] = lo_p
    p[flat[:, -1] < threshold] = hi_p
    return p.reshape(q.shape[:-1])


def dm_pinball(loss_a: np.ndarray, loss_b: np.ndarray) -> tuple[float, float]:
    """One-sided DM test on daily summed pinball losses (days, ...); small p: b beats a."""
    from scipy import stats
    la = loss_a.reshape(loss_a.shape[0], -1).sum(axis=1)
    lb = loss_b.reshape(loss_b.shape[0], -1).sum(axis=1)
    d = la - lb
    stat = d.mean() / (d.std(ddof=0) / np.sqrt(len(d)))
    return float(stat), float(1 - stats.norm.cdf(stat))


def conformalize(q_hist: np.ndarray, y_hist: np.ndarray, q_t: np.ndarray,
                 taus: np.ndarray = TAUS) -> np.ndarray:
    """Conformalized quantile regression (CQR, Romano et al. 2019) on a rolling history.

    q_hist (k, 24, n_tau) past quantile forecasts, y_hist (k, 24) the matching actuals,
    q_t (24, n_tau) today's raw quantiles. For each symmetric pair (tau, 1 - tau) the
    conformity score is max(lo - y, y - hi); the interval is
    widened (or narrowed, if the score quantile is negative) by the finite-sample-corrected
    quantile of the pooled scores (all hours of the history). The median is unchanged.
    """
    out = q_t.copy()
    n_tau = len(taus)
    yh = y_hist.reshape(-1)
    n = yh.size
    for j in range(n_tau // 2):
        lo_h = q_hist[..., j].reshape(-1)
        hi_h = q_hist[..., n_tau - 1 - j].reshape(-1)
        s = np.sort(np.maximum(lo_h - yh, yh - hi_h))
        cov = 1 - 2 * taus[j]
        k = min(int(np.ceil((n + 1) * cov)), n) - 1
        out[:, j] -= s[k]
        out[:, n_tau - 1 - j] += s[k]
    return np.sort(out, axis=1)
