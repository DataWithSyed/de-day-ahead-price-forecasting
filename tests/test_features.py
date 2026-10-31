import datetime as dt

import numpy as np
import pandas as pd

from dapf.features import easter_sunday, gbm_long, german_national_holidays, lear_design, to_daily_panel


def synthetic_hourly(start="2025-03-20", end="2025-11-05"):
    idx = pd.date_range(start, end, freq="1h", tz="UTC", inclusive="left")
    n = len(idx); rng = np.random.default_rng(0)
    return pd.DataFrame({"price": np.arange(n, dtype=float), "load_fc": 50000 + rng.normal(0, 100, n),
                         "solar_fc": np.clip(rng.normal(5000, 3000, n), 0, None), "wind_onshore_fc": 10000.0,
                         "wind_offshore_fc": 2000.0}, index=idx)


def test_easter_and_holidays():
    assert easter_sunday(2024) == dt.date(2024, 3, 31)
    assert easter_sunday(2025) == dt.date(2025, 4, 20)
    assert easter_sunday(2026) == dt.date(2026, 4, 5)
    h = german_national_holidays(2026)
    assert dt.date(2026, 4, 3) in h and dt.date(2026, 4, 6) in h      # Good Friday, Easter Monday
    assert dt.date(2026, 5, 14) in h and dt.date(2026, 5, 25) in h     # Ascension, Whit Monday
    assert dt.date(2026, 10, 3) in h and len(h) == 9


def test_daily_panel_handles_dst_days():
    p = to_daily_panel(synthetic_hourly())["price"]
    full_days = p.loc["2025-03-21":"2025-11-03"]                 # first/last local days are only partly covered
    assert p.shape[1] == 24 and not full_days.isna().any().any()
    # spring-forward day 2025-03-30 (23 h): the missing 02:00 hour is interpolated between 01:00 and 03:00
    spring = p.loc["2025-03-30"]
    assert np.isclose(spring[2], (spring[1] + spring[3]) / 2)
    # fall-back day 2025-10-26 (25 h): the two 02:00 hours are averaged
    fall = p.loc["2025-10-26"]
    assert np.isclose(fall[2] - fall[1], 1.5) and np.isclose(fall[3] - fall[2], 1.5)


def test_lear_design_is_ex_ante():
    panel = to_daily_panel(synthetic_hourly())
    X, P = lear_design(panel)
    day = pd.Timestamp("2025-06-10")
    # every lagged price column for day D equals the price of D-L, never of D itself
    for L in (1, 2, 3, 7):
        assert np.allclose(X.loc[day, [f"p_d{L}_h{h}" for h in range(24)]].to_numpy(float),
                           P.loc[day - pd.Timedelta(days=L)].to_numpy(float))
    price_cols = [c for c in X.columns if c.startswith("p_")]
    assert X.loc[day, price_cols].max() < P.loc[day].min()       # monotone synthetic prices: all lags are strictly older
    assert X.loc[day, "dow_1"] == 1.0                            # Tuesday


def test_gbm_long_is_ex_ante_and_aligned():
    L = gbm_long(to_daily_panel(synthetic_hourly()))
    row = L[(L.date == pd.Timestamp("2025-06-10")) & (L.hour == 15)].iloc[0]
    assert row["p_d1"] < row["y"] and row["p_d1_max"] < row["y"] and row["p_7d_mean"] < row["y"]
    assert row["p_d7"] < row["p_d2"] < row["p_d1"]
    assert L.groupby("date").size().eq(24).all()


def test_missing_forecast_day_filled_with_previous_day_but_price_not():
    df = synthetic_hourly()
    gap = (df.index >= "2025-06-09 22:00") & (df.index < "2025-06-10 22:00")   # local day 2025-06-10
    df.loc[gap, "solar_fc"] = np.nan
    df.loc[gap, "price"] = np.nan
    p = to_daily_panel(df)
    assert pd.Timestamp("2025-06-10") in p["solar_fc"].index
    assert np.allclose(p["solar_fc"].loc["2025-06-10"], p["solar_fc"].loc["2025-06-09"])
    assert p["price"].loc["2025-06-10"].isna().all()
    assert all(v.index.equals(p["price"].index) for v in p.values())
