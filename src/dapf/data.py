"""Download and cache Energy-Charts (Fraunhofer ISE) data for the DE-LU bidding zone.

Data licence: CC BY 4.0, source Energy-Charts.info (prices: Bundesnetzagentur | SMARD.de).
The public API is rate-limited (~2 requests/min per endpoint), so requests are
done one calendar year at a time and HTTP 429 ``Retry-After`` is honoured.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

API = "https://api.energy-charts.info"
TZ = "Europe/Berlin"
FORECAST_TYPES = ("load", "solar", "wind_onshore", "wind_offshore")


def _get(endpoint: str, params: dict, max_tries: int = 12) -> dict:
    for _ in range(max_tries):
        r = requests.get(f"{API}/{endpoint}", params=params, timeout=120)
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", "40")) + 2
            print(f"  429 on {endpoint}, waiting {wait}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"gave up on {endpoint} {params}")


def _to_hourly(unix_seconds, values, name: str) -> pd.Series:
    idx = pd.to_datetime(unix_seconds, unit="s", utc=True)
    s = pd.Series(values, index=idx, name=name, dtype="float64")
    # Since 1 Oct 2025 the SDAC uses a 15-minute MTU; we average to hourly.
    return s.resample("1h").mean()


def fetch_year(year: int, end: str | None = None) -> pd.DataFrame:
    start = f"{year}-01-01"
    end = end or f"{year}-12-31"
    cols = []
    p = _get("price", {"bzn": "DE-LU", "start": start, "end": end})
    cols.append(_to_hourly(p["unix_seconds"], p["price"], "price"))
    for pt in FORECAST_TYPES:
        f = _get("public_power_forecast", {"country": "de", "production_type": pt,
                                           "forecast_type": "day-ahead", "start": start, "end": end})
        cols.append(_to_hourly(f["unix_seconds"], f["forecast_values"], f"{pt}_fc"))
    return pd.concat(cols, axis=1)


def build_dataset(years=(2023, 2024, 2025, 2026), last_day: str = "2026-10-30",
                  out: Path | str = "data/de_lu_hourly.csv.gz") -> pd.DataFrame:
    parts = []
    for y in years:
        print(f"fetching {y}")
        parts.append(fetch_year(y, end=last_day if str(y) == last_day[:4] else None))
    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated()]
    df.index.name = "time_utc"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.round(2).to_csv(out, compression="gzip")
    return df


def load_dataset(path: Path | str = "data/de_lu_hourly.csv.gz") -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df
