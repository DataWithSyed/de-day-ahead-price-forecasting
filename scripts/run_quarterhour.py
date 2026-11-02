"""Quarter-hour (15-min MTU) extension: how much extra arbitrage do 96 prices per day offer a battery, and how much of
it survives when the schedule is built from this repo's *hourly* forecasts plus an ex-ante intra-hour shape?

Inputs: results/forecast_*.csv (out-of-sample hourly forecasts from run_backtest.py) and the native 15-min DE-LU
day-ahead prices (data/de_lu_qh_price.csv.gz, Energy-Charts / Bundesnetzagentur | SMARD.de, CC BY 4.0).
Every schedule is optimised on information available before D's gate closure and settled on actual 15-min prices.

Strategies per forecast model (Naive, LEAR, GBM, Ensemble):
  hourly        : hourly LP on the hourly forecast, held flat over each hour (= the main backtest)
  qh_shaped     : 15-min LP on the hourly forecast + median intra-hour shape of the previous 28 delivery days
Plus references:
  PF hourly     : perfect foresight on hourly averages (hourly schedule)
  PF hourly+shape : perfect hourly levels + the same ex-ante shape (isolates the shape forecast)
  PF 15-min     : perfect foresight on the actual 15-min prices (upper bound)
  Naive 15-min  : the naive rule applied directly to 15-min prices (D-7 on Mon/Sat/Sun, else D-1)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dapf.bess import Battery, backtest  # noqa: E402
from dapf.data import load_qh_prices  # noqa: E402
from dapf.quarterhour import ex_ante_shape, hourly_mean, hourly_to_qh, to_qh_panel  # noqa: E402

RES = ROOT / "results"
MODELS = ("Naive", "LEAR", "GBM", "Ensemble")
WINDOW, WINDOWS_SENS = 28, (7, 14, 28, 56)
BATT = {"1h": Battery(1.0, 1.0), "2h": Battery(1.0, 2.0), "4h": Battery(1.0, 4.0)}

A = pd.read_csv(RES / "actual_test.csv", index_col=0, parse_dates=True)
days = A.index
F = {m: pd.read_csv(RES / f"forecast_{m.lower()}.csv", index_col=0, parse_dates=True).loc[days].to_numpy(float) for m in MODELS}
Q = to_qh_panel(load_qh_prices(ROOT / "data/de_lu_qh_price.csv.gz"))
Qt = Q.loc[days].to_numpy(float)
assert not np.isnan(Qt).any(), "missing 15-min prices in the test window"
Ah = A.to_numpy(float)
print(f"test days {len(days)}  max |hourly mean of 15-min - hourly extract| = {np.nanmax(np.abs(hourly_mean(Qt) - Ah)):.3f} EUR/MWh")

shapes = {w: np.vstack([ex_ante_shape(Q, d, window_days=w) for d in days]) for w in WINDOWS_SENS}
S = shapes[WINDOW]

# naive applied natively to 15-min prices
dow = days.day_of_week
lag = np.where(np.isin(dow, (0, 5, 6)), 7, 1)
naive_qh = np.vstack([Q.loc[d - pd.Timedelta(days=int(L))].to_numpy(float) for d, L in zip(days, lag)])

# 1. accuracy of the 15-min price path
acc = []
for m in MODELS:
    for kind, X in (("flat", hourly_to_qh(F[m])), ("shaped", hourly_to_qh(F[m], S))):
        acc.append(dict(model=m, qh_forecast=kind, MAE_15min=np.abs(Qt - X).mean(), RMSE_15min=np.sqrt(((Qt - X) ** 2).mean())))
acc.append(dict(model="Naive 15-min", qh_forecast="native", MAE_15min=np.abs(Qt - naive_qh).mean(), RMSE_15min=np.sqrt(((Qt - naive_qh) ** 2).mean())))
acc.append(dict(model="Perfect hourly", qh_forecast="flat", MAE_15min=np.abs(Qt - hourly_to_qh(Ah)).mean(), RMSE_15min=np.sqrt(((Qt - hourly_to_qh(Ah)) ** 2).mean())))
acc.append(dict(model="Perfect hourly", qh_forecast="shaped", MAE_15min=np.abs(Qt - hourly_to_qh(Ah, S)).mean(), RMSE_15min=np.sqrt(((Qt - hourly_to_qh(Ah, S)) ** 2).mean())))
acc = pd.DataFrame(acc)
# share of intra-hour variance explained by the ex-ante shape
dev = Qt - hourly_to_qh(hourly_mean(Qt))
shape_r2 = 1 - ((dev - S) ** 2).sum() / (dev ** 2).sum()

# 2. battery value, settled on actual 15-min prices
rows = []
for bn, b in BATT.items():
    pf_q = backtest(Qt, Qt, b, dt_h=0.25)
    def add(name, strategy, r):
        rows.append(dict(battery=bn, model=name, strategy=strategy, profit_eur_per_mw=r["profit_eur"],
                         pct_of_pf_15min=100 * r["profit_eur"] / pf_q["profit_eur"], cycles=r["cycles"], loss_days=r["loss_days"]))
    add("Perfect foresight", "15-min", pf_q)
    # an hourly schedule held flat over 4 quarters earns exactly the hourly-mean price -> settle on the 15-min path
    add("Perfect foresight", "hourly", backtest(Qt, hourly_to_qh(Ah), b, dt_h=0.25))
    add("Perfect foresight", "hourly + ex-ante shape", backtest(Qt, hourly_to_qh(Ah, S), b, dt_h=0.25))
    add("Naive 15-min", "15-min", backtest(Qt, naive_qh, b, dt_h=0.25))
    for m in MODELS:
        add(m, "hourly", backtest(Qt, hourly_to_qh(F[m]), b, dt_h=0.25))
        add(m, "hourly + ex-ante shape", backtest(Qt, hourly_to_qh(F[m], S), b, dt_h=0.25))
    print(bn, "done", flush=True)
val = pd.DataFrame(rows)

# 3. shape-window sensitivity (Ensemble, 2 h battery)
b = BATT["2h"]; sens = []
for w, Sw in shapes.items():
    r = backtest(Qt, hourly_to_qh(F["Ensemble"], Sw), b, dt_h=0.25)
    sens.append(dict(window_days=w, profit_eur_per_mw=r["profit_eur"], MAE_15min=np.abs(Qt - hourly_to_qh(F["Ensemble"], Sw)).mean(),
                     shape_r2=1 - ((dev - Sw) ** 2).sum() / (dev ** 2).sum()))
sens = pd.DataFrame(sens)

# mean intra-hour deviation profile in the test window (descriptive, for the figure)
prof = pd.DataFrame(dev.reshape(len(days), 24, 4).mean(axis=0), columns=["q1", "q2", "q3", "q4"])
prof.index.name = "local_hour"

acc.to_csv(RES / "qh_forecast_accuracy.csv", index=False)
val.to_csv(RES / "qh_bess_value.csv", index=False)
sens.to_csv(RES / "qh_shape_window_sensitivity.csv", index=False)
prof.to_csv(RES / "qh_mean_intrahour_deviation.csv")
summary = dict(test_days=len(days), shape_window_days=WINDOW, shape_r2_intrahour=float(shape_r2),
               intrahour_dev_std=float(dev.std()), intrahour_dev_mean_abs=float(np.abs(dev).mean()),
               accuracy=acc.to_dict("records"), bess=val.to_dict("records"), window_sensitivity=sens.to_dict("records"))
json.dump(summary, open(RES / "qh_summary.json", "w"), indent=2)
pd.set_option("display.width", 200)
print(acc.round(2)); print(val.round(1)); print(sens.round(3)); print(f"shape R2 {shape_r2:.3f}, |dev| {np.abs(dev).mean():.2f}")
