from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dapf.bess import Battery, optimise_day
from dapf.probabilistic import TAUS
from dapf.risk import (block_bootstrap_ci, copula_levels, optimise_day_cvar, quantile_function,
                       scenarios, var_cvar)

RES = Path(__file__).resolve().parents[1] / "results"


def test_var_cvar_lower_tail():
    x = np.arange(1, 11.0)
    assert var_cvar(x, 0.1) == (1.0, 1.0)
    v, c = var_cvar(x, 0.25)                      # tail = 1 + 2 + 0.5*3 over 2.5
    assert v == 3.0 and np.isclose(c, 1.8)


def test_quantile_function_interpolates_and_extrapolates():
    q = np.tile(np.linspace(-90, 90, len(TAUS)), (2, 1))   # 10 EUR per 0.05 level -> 200 EUR per unit
    u = np.array([[0.5, 0.05], [0.025, 0.975]])
    v = quantile_function(q, TAUS, u)
    assert np.allclose(v, [[0.0, -90.0], [-95.0, 95.0]])


def test_copula_preserves_rank_pattern_and_marginals():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(56, 1))
    err = base + 0.1 * rng.normal(size=(56, 24))              # strongly co-moving hours
    u = copula_levels(err)
    assert np.allclose(np.sort(u[:, 0]), np.arange(1, 57) / 57)
    q = np.sort(rng.normal(80, 30, (24, len(TAUS))), axis=1)
    s = scenarios(q, TAUS, err)
    assert s.shape == (56, 24)
    assert np.corrcoef(s[:, 0], s[:, 23])[0, 1] > 0.9           # dependence carried into prices
    assert (np.abs(np.median(s, axis=0) - q[:, 9]) <= (q[:, 10] - q[:, 8]) + 1e-9).all()


def test_lambda_zero_equals_deterministic_on_scenario_mean():
    rng = np.random.default_rng(2)
    scen = rng.normal(80, 40, (40, 24)) + 30 * np.sin(np.arange(24) / 24 * 2 * np.pi)
    b = Battery()
    c0, d0, info = optimise_day_cvar(scen, b, lam=0.0)
    c1, d1 = optimise_day(scen.mean(axis=0), b)
    obj = lambda c, d: scen.mean(axis=0) @ (d - c) - b.wear_eur_mwh * d.sum()
    assert np.isclose(obj(c0, d0), obj(c1, d1), atol=1e-6)
    assert np.isclose(info["exp_profit"], obj(c0, d0), atol=1e-6)


def test_identical_scenarios_reduce_to_deterministic():
    price = np.array([10.0] * 12 + [100.0] * 12)
    b = Battery(rte=0.81, wear_eur_mwh=0.0)
    c, d, info = optimise_day_cvar(np.tile(price, (20, 1)), b, lam=5.0)
    cd, dd = optimise_day(price, b)
    assert np.isclose(price @ (d - c), price @ (dd - cd), atol=1e-6)   # same optimal value (schedule may tie)
    assert np.isclose(info["cvar"], info["exp_profit"])


def test_higher_lambda_raises_exante_cvar_and_lowers_mean():
    rng = np.random.default_rng(4)
    shape = 40 * np.sin(np.arange(24) / 24 * 2 * np.pi)
    scen = 80 + shape + rng.normal(0, 1, (60, 1)) * rng.normal(0, 25, 24) + rng.normal(0, 30, (60, 24))
    b = Battery()
    res = [optimise_day_cvar(scen, b, lam=l)[2] for l in (0.0, 1.0, np.inf)]
    assert res[0]["exp_profit"] >= res[1]["exp_profit"] - 1e-6 >= res[2]["exp_profit"] - 2e-6
    assert res[2]["cvar"] >= res[1]["cvar"] - 1e-6 >= res[0]["cvar"] - 2e-6
    assert res[2]["cvar"] >= -1e-6                                # doing nothing has CVaR 0


def test_block_bootstrap_ci_contains_mean():
    x = np.random.default_rng(5).normal(2.0, 1.0, 300)
    lo, hi = block_bootstrap_ci(x)
    assert lo < x.mean() < hi and hi - lo < 1.0


@pytest.mark.skipif(not (RES / "risk_schedule.csv").exists(), reason="run scripts/run_risk_schedule.py first")
def test_published_risk_results_consistent():
    r = pd.read_csv(RES / "risk_schedule.csv")
    pm = pd.read_csv(RES / "prob_bess_median.csv")
    for bn, g in r.groupby("battery"):
        pf = g.loc[g.schedule == "Perfect foresight", "profit"].iloc[0]
        assert (g.profit <= pf + 1e-6).all()
        # same 280 days, same LP: point schedules reproduce run_probabilistic.py
        ens = pm[(pm.battery == bn) & (pm.schedule_from == "Ensemble")].profit_eur_per_mw.iloc[0]
        assert np.isclose(g.loc[g.schedule == "Ensemble point", "profit"].iloc[0], ens, rtol=1e-6)
        assert np.isclose(pf, pm[pm.battery == bn].perfect_foresight_eur_per_mw.iloc[0], rtol=1e-6)
