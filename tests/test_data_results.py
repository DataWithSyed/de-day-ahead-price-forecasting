from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dapf.bess import Battery, optimise_day
from dapf.data import _to_hourly
from dapf.evaluate import mae
from dapf.features import calendar

RES = Path(__file__).resolve().parents[1] / "results"


def test_quarter_hours_averaged_to_hourly():
    t0 = 1759269600                                       # 2025-09-30 22:00 UTC
    secs = [t0 + 900 * i for i in range(8)]
    s = _to_hourly(secs, [10, 20, 30, 40, 0, 0, 0, 100], "price")
    assert list(s.values) == [25.0, 25.0]
    s2 = _to_hourly([t0, t0 + 3600], [5.0, 7.0], "price")   # already hourly: unchanged
    assert list(s2.values) == [5.0, 7.0]


def test_calendar_flags():
    c = calendar(pd.DatetimeIndex(["2026-12-24", "2026-12-25", "2026-10-03"]))
    assert list(c["holiday"]) == [False, True, True]
    assert list(c["dow"]) == [3, 4, 5]


def test_cycle_cap_respected():
    b = Battery(power_mw=1.0, energy_mwh=1.0, rte=0.9, cycles_per_day=1.0, wear_eur_mwh=0.0)
    price = np.tile([0.0, 0.0, 200.0, 200.0], 6)          # six profitable spreads in a day
    c, d = optimise_day(price, b)
    assert d.sum() / b.eta <= 1.0 + 1e-6


@pytest.mark.skipif(not (RES / "metrics.csv").exists(), reason="run scripts/run_backtest.py first")
def test_published_metrics_match_forecast_files():
    A = pd.read_csv(RES / "actual_test.csv", index_col=0)
    met = pd.read_csv(RES / "metrics.csv").set_index("model")
    F = {m: pd.read_csv(RES / f"forecast_{m.lower()}.csv", index_col=0) for m in met.index}
    for m, f in F.items():
        assert f.index.equals(A.index) and not f.isna().any().any()
        assert abs(mae(A.to_numpy(), f.to_numpy()) - met.loc[m, "MAE"]) < 1e-2
    assert np.allclose(F["Ensemble"].to_numpy(), (F["LEAR"].to_numpy() + F["GBM"].to_numpy()) / 2, atol=2e-3)
    assert len(A) == 364


@pytest.mark.skipif(not (RES / "bess_value.csv").exists(), reason="run scripts/run_backtest.py first")
def test_perfect_foresight_dominates_every_forecast():
    b = pd.read_csv(RES / "bess_value.csv")
    for _, g in b.groupby("battery"):
        pf = g.loc[g.model == "Perfect foresight", "profit_eur_per_mw"].iloc[0]
        assert (g.profit_eur_per_mw <= pf + 1e-6).all()
