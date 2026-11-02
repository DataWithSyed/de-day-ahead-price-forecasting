"""Quarter-hour (15-min MTU) helpers: local-time day x 96 panel, ex-ante intra-hour shape, hourly -> 15-min forecasts.

Since delivery day 1 Oct 2025 the SDAC day-ahead auction clears 96 quarter-hour prices per day. The forecasting models
in this repo work on hourly averages; these helpers test how much of the extra 15-minute arbitrage a battery can still
capture when its hourly forecast is given a 15-minute shape learned only from past auction results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TZ = "Europe/Berlin"


def to_qh_panel(s: pd.Series) -> pd.DataFrame:
    """15-min UTC series -> DataFrame(index=local date, columns=0..95), column = 4 * local hour + quarter.

    Same DST rule as the hourly panel: on the 25-hour autumn day the two 02:xx quarters are averaged; on the 23-hour
    spring day the four missing 02:xx quarters are interpolated linearly along the day."""
    loc = s.tz_convert(TZ)
    tab = pd.DataFrame({"date": loc.index.date, "col": loc.index.hour * 4 + loc.index.minute // 15, "v": loc.values})
    p = tab.pivot_table(index="date", columns="col", values="v", aggfunc="mean").reindex(columns=range(96))
    p.index = pd.to_datetime(p.index)
    return p.interpolate(axis=1, limit=4, limit_direction="both")


def hourly_mean(Q: pd.DataFrame | np.ndarray) -> np.ndarray:
    """(days, 96) -> (days, 24) mean of the four quarters of each hour."""
    q = np.asarray(Q, float)
    return q.reshape(len(q), 24, 4).mean(axis=2)


def intrahour_deviation(Q: pd.DataFrame) -> pd.DataFrame:
    """Each quarter's price minus the mean of its hour (sums to zero within every hour)."""
    h = np.repeat(hourly_mean(Q), 4, axis=1)
    return pd.DataFrame(Q.to_numpy(float) - h, index=Q.index, columns=Q.columns)


def ex_ante_shape(Q: pd.DataFrame, day: pd.Timestamp, window_days: int = 28, stat: str = "median") -> np.ndarray:
    """96-vector intra-hour shape for delivery day ``day`` from the ``window_days`` delivery days before it.

    Only days strictly before D are used: their auction results were published by D-1, before D's gate closure.
    The shape is re-centred so that it sums to zero within each hour (it moves energy within an hour, never the
    hourly level, which stays with the hourly forecast)."""
    past = Q[(Q.index < day) & (Q.index >= day - pd.Timedelta(days=window_days))]
    if past.empty:
        return np.zeros(96)
    dev = intrahour_deviation(past.dropna())
    v = (dev.median() if stat == "median" else dev.mean()).to_numpy(float)
    return v - np.repeat(v.reshape(24, 4).mean(axis=1), 4)


def hourly_to_qh(F_hourly: np.ndarray, shape: np.ndarray | None = None) -> np.ndarray:
    """(days, 24) hourly forecast -> (days, 96): each hour repeated four times, plus an optional (days, 96) shape."""
    f = np.repeat(np.asarray(F_hourly, float), 4, axis=1)
    return f if shape is None else f + np.asarray(shape, float)
