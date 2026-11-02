import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd

from dapf.bess import Battery, backtest, optimise_day
from dapf.quarterhour import ex_ante_shape, hourly_mean, hourly_to_qh, intrahour_deviation, to_qh_panel


def _qh_series(start, end, seed=0):
    idx = pd.date_range(start, end, freq="15min", tz="UTC", inclusive="left")
    rng = np.random.default_rng(seed)
    return pd.Series(80 + 30 * np.sin(np.arange(len(idx)) / 10) + rng.normal(0, 5, len(idx)), index=idx)


def test_qh_panel_dst_days_have_96_columns_and_no_gaps():
    s = _qh_series("2026-03-27", "2026-03-31")             # spring DST on 29 Mar 2026 (23 h)
    s2 = _qh_series("2025-10-24", "2025-10-28", seed=1)    # autumn DST on 26 Oct 2025 (25 h)
    for ser, day in ((s, "2026-03-29"), (s2, "2025-10-26")):
        Q = to_qh_panel(ser)
        assert Q.shape[1] == 96
        assert Q.loc[day].notna().all()


def test_dt_quarter_hour_lp_matches_hourly_lp_on_flat_prices():
    """Hourly prices repeated 4x at dt=0.25 must give exactly the hourly optimum (same energy, same limits)."""
    rng = np.random.default_rng(5)
    b = Battery(1.0, 2.0)
    P = rng.normal(90, 45, (20, 24))
    h = backtest(P, P, b)
    q = backtest(np.repeat(P, 4, axis=1), np.repeat(P, 4, axis=1), b, dt_h=0.25)
    assert np.isclose(h["profit_eur"], q["profit_eur"], rtol=1e-7)
    assert np.isclose(h["cycles"], q["cycles"], rtol=1e-7)


def test_quarter_hour_lp_respects_energy_and_cycle_limits():
    rng = np.random.default_rng(2)
    b = Battery(1.0, 1.0)
    p = rng.normal(80, 60, 96)
    c, d = optimise_day(p, b, dt_h=0.25)
    soc = np.cumsum(0.25 * (b.eta * c - d / b.eta))
    assert soc.min() > -1e-6 and soc.max() < b.energy_mwh + 1e-6
    assert 0.25 * d.sum() / b.eta <= b.cycles_per_day * b.energy_mwh + 1e-6


def test_shape_is_ex_ante_and_zero_mean_within_each_hour():
    Q = to_qh_panel(_qh_series("2026-01-01", "2026-03-01")).dropna()
    day = pd.Timestamp("2026-02-10")
    sh = ex_ante_shape(Q, day)
    assert np.allclose(sh.reshape(24, 4).sum(axis=1), 0, atol=1e-9)
    Q2 = Q.copy(); Q2.loc[Q2.index >= day] = Q2.loc[Q2.index >= day] * 10 - 500   # change D and later
    assert np.allclose(ex_ante_shape(Q2, day), sh)
    assert np.allclose(intrahour_deviation(Q).to_numpy().reshape(len(Q), 24, 4).sum(axis=2), 0, atol=1e-9)
    F = hourly_to_qh(hourly_mean(Q.iloc[:3]), np.tile(sh, (3, 1)))
    assert np.allclose(hourly_mean(F), hourly_mean(Q.iloc[:3]))


def test_cached_qh_prices_average_to_the_hourly_extract():
    """The native 15-min extract averages to the hourly extract the forecasts were trained on."""
    from dapf.data import load_dataset, load_qh_prices
    from dapf.features import to_daily_panel
    qh = load_qh_prices(ROOT / "data/de_lu_qh_price.csv.gz")
    P = to_daily_panel(load_dataset(ROOT / "data/de_lu_hourly.csv.gz"))["price"]
    Q = to_qh_panel(qh)
    days = Q.index.intersection(P.index)[1:-1]
    assert len(days) > 360
    diff = np.abs(hourly_mean(Q.loc[days]) - P.loc[days].to_numpy(float))
    # the non-existent 02:00 hour of the spring DST day is interpolated differently (hourly vs 15-min grid)
    diff[days.get_loc(pd.Timestamp("2026-03-29")), 2] = 0.0
    assert np.nanmax(diff) < 0.02 + 1e-9          # otherwise only rounding to 2 decimals
