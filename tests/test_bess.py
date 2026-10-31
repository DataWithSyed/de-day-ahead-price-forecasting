import numpy as np

from dapf.bess import Battery, backtest, optimise_day


def test_two_price_day_full_cycle():
    b = Battery(power_mw=1.0, energy_mwh=2.0, rte=0.81, cycles_per_day=1.0, wear_eur_mwh=0.0)
    price = np.array([10.0] * 12 + [100.0] * 12)
    c, d = optimise_day(price, b)
    assert np.isclose(c.sum() * b.eta, 2.0, atol=1e-6)          # fills the battery
    assert np.isclose(d.sum(), 2.0 * b.eta, atol=1e-6)           # empties it once (cycle cap)
    assert c[12:].sum() < 1e-9 and d[:12].sum() < 1e-9
    assert (c <= 1 + 1e-9).all() and (d <= 1 + 1e-9).all()


def test_no_trade_when_spread_below_losses():
    b = Battery(rte=0.8, wear_eur_mwh=0.0)
    price = np.array([100.0] * 12 + [110.0] * 12)               # 110*0.8 < 100: arbitrage loses money
    c, d = optimise_day(price, b)
    assert c.sum() < 1e-9 and d.sum() < 1e-9


def test_soc_stays_within_bounds():
    rng = np.random.default_rng(0)
    b = Battery()
    price = rng.normal(80, 40, 24)
    c, d = optimise_day(price, b)
    soc = np.cumsum(b.eta * c - d / b.eta)
    assert soc.min() > -1e-6 and soc.max() < b.energy_mwh + 1e-6


def test_perfect_forecast_is_upper_bound():
    rng = np.random.default_rng(3)
    actual = rng.normal(80, 40, (30, 24))
    noisy = actual + rng.normal(0, 30, actual.shape)
    b = Battery()
    pf, fc = backtest(actual, actual, b), backtest(actual, noisy, b)
    assert pf["profit_eur"] >= fc["profit_eur"] - 1e-6
    assert pf["loss_days"] == 0
