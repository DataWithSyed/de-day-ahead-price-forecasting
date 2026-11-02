"""Daily BESS arbitrage on day-ahead prices: optimise on a forecast, settle on actual prices.

Linear programme per delivery day (solved with HiGHS via scipy.optimize.linprog):
    max  dt * [ sum_t price_t * (d_t - c_t) - wear * sum_t d_t ]
    s.t. soc_t = soc_{t-1} + dt * (eta * c_t - d_t / eta),   soc_0 = soc_init,  0 <= soc_t <= E
         0 <= c_t, d_t <= P,   dt * sum_t d_t / eta <= cycles * E (throughput cap)
dt is the market time unit in hours (1.0 for hourly, 0.25 for the 15-minute MTU); c_t, d_t are MW.
The schedule is FIXED once optimised (a day-ahead position) and its cash flow is computed with actual prices.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, lil_matrix


@dataclass(frozen=True)
class Battery:
    power_mw: float = 1.0
    energy_mwh: float = 2.0
    rte: float = 0.88              # AC round-trip efficiency (assumption)
    cycles_per_day: float = 1.0    # throughput cap, equivalent full cycles per day
    wear_eur_mwh: float = 5.0      # wear-cost proxy per MWh discharged (assumption)

    @property
    def eta(self) -> float:
        return float(np.sqrt(self.rte))


def optimise_day(price: np.ndarray, b: Battery, soc_init: float = 0.0, dt_h: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (charge, discharge) MW schedules (steps of dt_h hours) that maximise forecast-price profit."""
    price = np.asarray(price, float); T = len(price); eta = b.eta
    n = 3 * T                                      # c, d, soc
    cost = np.concatenate([price * dt_h, (-price + b.wear_eur_mwh) * dt_h, np.zeros(T)])
    A_eq = lil_matrix((T, n)); b_eq = np.zeros(T)
    for t in range(T):
        A_eq[t, 2 * T + t] = 1.0; A_eq[t, t] = -eta * dt_h; A_eq[t, T + t] = dt_h / eta
        if t > 0:
            A_eq[t, 2 * T + t - 1] = -1.0
        else:
            b_eq[t] = soc_init
    A_ub = lil_matrix((1, n)); A_ub[0, T:2 * T] = dt_h / eta
    b_ub = np.array([b.cycles_per_day * b.energy_mwh])
    bounds = [(0, b.power_mw)] * (2 * T) + [(0, b.energy_mwh)] * T
    res = linprog(cost, A_ub=csr_matrix(A_ub), b_ub=b_ub, A_eq=csr_matrix(A_eq), b_eq=b_eq, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    c, d = res.x[:T], res.x[T:2 * T]
    # remove simultaneous charge/discharge artefacts (never optimal with eta<1, but keep schedules clean)
    net = d - c
    return np.clip(-net, 0, None), np.clip(net, 0, None)


def backtest(actual: np.ndarray, forecast: np.ndarray, b: Battery, dt_h: float = 1.0) -> dict:
    """actual, forecast: (days, steps per day), e.g. 24 hourly or 96 quarter-hourly prices (dt_h = 0.25).
    Each day starts and ends empty (soc_init=0; ending empty is optimal)."""
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    profit = np.zeros(len(actual)); thr = np.zeros(len(actual))
    for i in range(len(actual)):
        c, d = optimise_day(forecast[i], b, dt_h=dt_h)
        profit[i] = float(dt_h * (actual[i] @ (d - c) - b.wear_eur_mwh * d.sum()))
        thr[i] = dt_h * d.sum()
    return dict(profit_eur=profit.sum(), daily_profit=profit, discharged_mwh=thr.sum(),
                cycles=thr.sum() / (b.eta * b.energy_mwh), loss_days=int((profit < 0).sum()))
