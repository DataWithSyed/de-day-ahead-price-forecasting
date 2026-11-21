"""Risk-aware (CVaR) day-ahead battery schedules from probabilistic price fans.

The quantile fan of a delivery day only gives marginal distributions per hour. To value a
24-hour schedule we need joint price paths, so scenarios are built with an **empirical copula**
(Schaake-shuffle style, Clark et al. 2004): the rank pattern across hours of each of the
previous ``window`` days' point-forecast errors is mapped through today's per-hour quantile
functions. Everything is ex ante (only errors of days before D are used).

The schedule is a fixed day-ahead position (charge c_t, discharge d_t), so the profit of
scenario s is linear in the schedule:

    pi_s = dt * [ p_s . (d - c) - wear * sum(d) ]

and the risk-aware LP (Rockafellar & Uryasev 2000) is

    max  E_s[pi_s] + lam * CVaR_alpha(pi)
    CVaR_alpha(pi) = max_eta  eta - 1/(alpha S) sum_s max(eta - pi_s, 0)   (lower tail)

with the same battery constraints as ``dapf.bess.optimise_day``. lam = 0 is the risk-neutral
stochastic schedule, which equals the deterministic LP on the scenario-mean price path.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, lil_matrix, vstack, identity

from dapf.bess import Battery


def quantile_function(q: np.ndarray, taus: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Evaluate per-hour quantile functions at levels u.

    q: (T, n_tau) sorted quantiles; u: (S, T) levels in (0, 1). Linear interpolation inside
    [taus[0], taus[-1]] and linear extrapolation of the outermost segment beyond it
    (ASSUMED tail shape; u is clipped to [0.005, 0.995]).
    """
    q = np.sort(np.asarray(q, float), axis=1)
    taus = np.asarray(taus, float); u = np.clip(np.asarray(u, float), 0.005, 0.995)
    out = np.empty(u.shape)
    lo_slope = (q[:, 1] - q[:, 0]) / (taus[1] - taus[0])
    hi_slope = (q[:, -1] - q[:, -2]) / (taus[-1] - taus[-2])
    for t in range(q.shape[0]):
        ut = u[:, t]
        v = np.interp(ut, taus, q[t])
        lo, hi = ut < taus[0], ut > taus[-1]
        v[lo] = q[t, 0] - lo_slope[t] * (taus[0] - ut[lo])
        v[hi] = q[t, -1] + hi_slope[t] * (ut[hi] - taus[-1])
        out[:, t] = v
    return out


def copula_levels(past_errors: np.ndarray) -> np.ndarray:
    """Empirical copula of past errors (W, T) -> uniform levels (W, T) = rank / (W + 1) per hour."""
    e = np.asarray(past_errors, float)
    ranks = e.argsort(axis=0).argsort(axis=0) + 1
    return ranks / (e.shape[0] + 1.0)


def scenarios(q: np.ndarray, taus: np.ndarray, past_errors: np.ndarray) -> np.ndarray:
    """Joint price scenarios (W, T): today's fan q (T, n_tau) with the rank dependence of past_errors."""
    return quantile_function(q, taus, copula_levels(past_errors))


def optimise_day_cvar(scen: np.ndarray, b: Battery, lam: float = 0.0, alpha: float = 0.1,
                      dt_h: float = 1.0, soc_init: float = 0.0) -> tuple[np.ndarray, np.ndarray, dict]:
    """Schedule maximising E[pi] + lam * CVaR_alpha(pi) over equally weighted scenarios (S, T).

    lam = np.inf maximises CVaR alone. Returns (charge, discharge, info) with the ex-ante
    expected profit, VaR (eta) and CVaR of the chosen schedule under the scenarios.
    """
    P = np.asarray(scen, float); S, T = P.shape; eta_b = b.eta
    wE, wC = (0.0, 1.0) if np.isinf(lam) else (1.0, float(lam))
    nx = 3 * T                                         # c, d, soc
    n = nx + 1 + S                                     # + eta (VaR) + z_s
    pbar = P.mean(axis=0)
    cost = np.zeros(n)                                 # linprog minimises -> negate the objective
    cost[:T] = wE * pbar * dt_h
    cost[T:2 * T] = wE * (-pbar + b.wear_eur_mwh) * dt_h
    cost[nx] = -wC
    cost[nx + 1:] = wC / (alpha * S)
    # SoC dynamics
    A_eq = lil_matrix((T, n)); b_eq = np.zeros(T)
    for t in range(T):
        A_eq[t, 2 * T + t] = 1.0; A_eq[t, t] = -eta_b * dt_h; A_eq[t, T + t] = dt_h / eta_b
        if t > 0:
            A_eq[t, 2 * T + t - 1] = -1.0
        else:
            b_eq[t] = soc_init
    # throughput cap
    cap = lil_matrix((1, n)); cap[0, T:2 * T] = dt_h / eta_b
    # z_s >= eta - pi_s  <=>  eta - pi_s - z_s <= 0,  pi_s = dt*(p_s.(d-c) - wear*sum d)
    pi_c = csr_matrix(P * dt_h)                         # coefficient of c in -pi_s
    pi_d = csr_matrix((-P + b.wear_eur_mwh) * dt_h)     # coefficient of d in -pi_s
    A_z = hstack([pi_c, pi_d, csr_matrix((S, T)), csr_matrix(np.ones((S, 1))), -identity(S, format="csr")])
    A_ub = vstack([csr_matrix(cap), A_z]).tocsr()
    b_ub = np.concatenate([[b.cycles_per_day * b.energy_mwh], np.zeros(S)])
    bounds = [(0, b.power_mw)] * (2 * T) + [(0, b.energy_mwh)] * T + [(None, None)] + [(0, None)] * S
    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=csr_matrix(A_eq), b_eq=b_eq, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    c, d = res.x[:T], res.x[T:2 * T]
    net = d - c
    c, d = np.clip(-net, 0, None), np.clip(net, 0, None)
    pi = dt_h * (P @ (d - c) - b.wear_eur_mwh * d.sum())
    info = dict(exp_profit=float(pi.mean()), var=float(var_cvar(pi, alpha)[0]), cvar=float(var_cvar(pi, alpha)[1]))
    return c, d, info


def var_cvar(x: np.ndarray, alpha: float = 0.1) -> tuple[float, float]:
    """Lower-tail VaR and CVaR of equally weighted outcomes x (Rockafellar-Uryasev definition)."""
    x = np.sort(np.asarray(x, float)); n = len(x)
    k = alpha * n; m = int(np.floor(k))
    var = x[min(m, n - 1)] if k > m else x[max(m - 1, 0)]
    tail = x[:m].sum() + (k - m) * (x[m] if m < n else 0.0)
    return float(var), float(tail / k)


def block_bootstrap_ci(diff: np.ndarray, block: int = 7, n_boot: int = 4000, seed: int = 0,
                       level: float = 0.95) -> tuple[float, float]:
    """Moving-block bootstrap CI of the mean of a daily series."""
    diff = np.asarray(diff, float); n = len(diff); rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block)); starts = rng.integers(0, n - block + 1, (n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    means = diff[idx].mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))
