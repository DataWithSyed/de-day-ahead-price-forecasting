"""Turn the hourly UTC dataset into a day x hour (local time) panel and build ex-ante feature sets.

Ex-ante rule (SDAC day-ahead auction, gate closure 12:00 CET on D-1): when forecasting delivery day D we may use
* all day-ahead prices up to and including day D-1 (published on D-2 ~12:45 CET), and
* the TSO day-ahead forecasts of load / solar / wind for day D (published on D-1 before gate closure).
Nothing from day D's prices is ever used.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

TZ = "Europe/Berlin"
SERIES = ("price", "load_fc", "solar_fc", "wind_onshore_fc", "wind_offshore_fc")


def easter_sunday(year: int) -> dt.date:
    """Anonymous Gregorian algorithm (Meeus/Jones/Butcher)."""
    a = year % 19; b, c = divmod(year, 100); d, e = divmod(b, 4); f = (b + 8) // 25; g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30; i, k = divmod(c, 4); l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451; month, day = divmod(h + l - 7 * m + 114, 31)
    return dt.date(year, month, day + 1)


def german_national_holidays(year: int) -> set[dt.date]:
    """Nationwide German public holidays (regional ones are ignored)."""
    e = easter_sunday(year)
    fixed = [(1, 1), (5, 1), (10, 3), (12, 25), (12, 26)]
    out = {dt.date(year, m, d) for m, d in fixed}
    out |= {e - dt.timedelta(days=2), e + dt.timedelta(days=1), e + dt.timedelta(days=39), e + dt.timedelta(days=50)}
    return out


def to_daily_panel(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return {series: DataFrame(index=local date, columns=0..23)}.

    DST handling: the 25-hour autumn day averages the two 02:00 hours; the 23-hour spring day gets its missing
    02:00 hour by linear interpolation along the day. Missing hours are interpolated (max 3 h); a forecast day that
    is entirely missing is filled with the previous day's forecast. Prices are never filled across days."""
    loc = df.tz_convert(TZ) if df.index.tz is not None else df.tz_localize("UTC").tz_convert(TZ)
    key = pd.DataFrame({"date": loc.index.date, "hour": loc.index.hour})
    dates = pd.to_datetime(sorted(set(key["date"])))
    out = {}
    for s in SERIES:
        tab = pd.DataFrame({"date": key["date"].values, "hour": key["hour"].values, "v": loc[s].values})
        p = tab.pivot_table(index="date", columns="hour", values="v", aggfunc="mean").reindex(columns=range(24))
        p.index = pd.to_datetime(p.index)
        p = p.reindex(dates).interpolate(axis=1, limit=3, limit_direction="both")
        if s != "price":
            # A forecast day missing in the source (e.g. solar/onshore wind on 2026-10-24) is filled with the
            # previous day's forecast profile - information a trader would also have had (ex-ante safe).
            p = p.ffill(limit=1)
        out[s] = p
    return out


def residual_load(panel: dict) -> pd.DataFrame:
    return panel["load_fc"] - panel["solar_fc"] - panel["wind_onshore_fc"] - panel["wind_offshore_fc"]


def calendar(dates: pd.DatetimeIndex) -> pd.DataFrame:
    hol = set().union(*[german_national_holidays(y) for y in range(dates.min().year, dates.max().year + 1)])
    c = pd.DataFrame(index=dates)
    c["dow"] = dates.day_of_week
    c["month"] = dates.month
    c["holiday"] = [d.date() in hol for d in dates]
    return c


def lear_design(panel: dict, price_lags=(1, 2, 3, 7)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide design matrix for the LEAR-style model (one row per day, 24 targets)."""
    P = panel["price"]; X = []
    for L in price_lags:
        X.append(P.shift(L, freq="D").reindex(P.index).add_prefix(f"p_d{L}_h"))
    wind = panel["wind_onshore_fc"] + panel["wind_offshore_fc"]
    for name, S in (("load", panel["load_fc"]), ("solar", panel["solar_fc"]), ("wind", wind)):
        X.append(S.add_prefix(f"{name}_d0_h"))
        X.append(S.shift(1, freq="D").reindex(P.index).add_prefix(f"{name}_d1_h"))
    cal = calendar(P.index)
    dow = pd.get_dummies(cal["dow"], prefix="dow").astype(float)
    X.append(dow); X.append(cal[["holiday"]].astype(float))
    X = pd.concat(X, axis=1)
    X.columns = [str(c) for c in X.columns]
    return X, P


def gbm_long(panel: dict) -> pd.DataFrame:
    """Long (day, hour) table for the gradient-boosting model. Column 'y' is the target price."""
    P = panel["price"]; idx = P.index
    rl = residual_load(panel); wind = panel["wind_onshore_fc"] + panel["wind_offshore_fc"]
    lag = lambda S, L: S.shift(L, freq="D").reindex(idx)
    p1, p2, p7 = lag(P, 1), lag(P, 2), lag(P, 7)
    cal = calendar(idx)
    frames = []
    for h in range(24):
        f = pd.DataFrame(index=idx)
        f["hour"] = h; f["dow"] = cal["dow"]; f["month"] = cal["month"]; f["holiday"] = cal["holiday"].astype(int)
        f["load"] = panel["load_fc"][h]; f["solar"] = panel["solar_fc"][h]; f["wind"] = wind[h]; f["resload"] = rl[h]
        f["res_share"] = (panel["solar_fc"][h] + wind[h]) / panel["load_fc"][h]
        f["resload_mean"] = rl.mean(axis=1); f["resload_max"] = rl.max(axis=1); f["resload_min"] = rl.min(axis=1)
        f["resload_d1"] = lag(rl, 1)[h]; f["resload_delta"] = f["resload"] - f["resload_d1"]
        f["solar_sum"] = panel["solar_fc"].sum(axis=1); f["wind_mean"] = wind.mean(axis=1)
        f["p_d1"] = p1[h]; f["p_d2"] = p2[h]; f["p_d7"] = p7[h]
        f["p_d1_mean"] = p1.mean(axis=1); f["p_d1_max"] = p1.max(axis=1); f["p_d1_min"] = p1.min(axis=1)
        f["p_d1_h19"] = p1[19]; f["p_d1_h12"] = p1[12]
        f["p_7d_mean"] = sum(lag(P, L).mean(axis=1) for L in range(1, 8)) / 7
        f["y"] = P[h]
        frames.append(f.reset_index(names="date"))
    return pd.concat(frames, ignore_index=True).sort_values(["date", "hour"]).reset_index(drop=True)
