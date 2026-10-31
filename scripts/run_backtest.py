"""Rolling out-of-sample backtest of DE-LU day-ahead price forecasts + BESS value of each forecast.

Test window: 2025-11-01 .. 2026-10-30 (364 delivery days, all after the 15-min MTU go-live, averaged to hourly).
* Naive     : Lago et al. (2021) benchmark (D-7 on Mon/Sat/Sun, else D-1)
* LEAR      : 24 hourly Lasso models on asinh-scaled data, re-fitted every day on a rolling 728-day window
* GBM       : HistGradientBoosting (absolute-error loss) on 25 engineered features, expanding window, re-fitted every 7 days
* Ensemble  : simple average of LEAR and GBM
Every forecast for day D uses only prices up to D-1 and TSO day-ahead load/solar/wind forecasts for D.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # LEAR fits are parallelised over hours instead
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dapf.bess import Battery, backtest  # noqa: E402
from dapf.data import load_dataset  # noqa: E402
from dapf.evaluate import dm_test, mae, rmae, rmse, smape  # noqa: E402
from dapf.features import gbm_long, lear_design, to_daily_panel  # noqa: E402
from dapf.models import GBM_FEATURES, LEAR, make_gbm, naive_forecast  # noqa: E402

TEST_START, TEST_END = pd.Timestamp("2025-11-01"), pd.Timestamp("2026-10-30")
LEAR_WINDOW_DAYS, GBM_REFIT_DAYS = 728, 7
RES = ROOT / "results"; RES.mkdir(exist_ok=True)

df = load_dataset(ROOT / "data/de_lu_hourly.csv.gz")
panel = to_daily_panel(df)
X, P = lear_design(panel)
L = gbm_long(panel)
days = P.loc[TEST_START:TEST_END].index
print(f"panel {P.index.min().date()}..{P.index.max().date()}  test days {len(days)}", flush=True)

fc = {m: pd.DataFrame(index=days, columns=range(24), dtype=float) for m in ("Naive", "LEAR", "GBM")}
n_sel = []
t0 = time.time(); gbm = None
for k, day in enumerate(days):
    fc["Naive"].loc[day] = naive_forecast(P, day)
    tr = (X.index < day) & (X.index >= day - pd.Timedelta(days=LEAR_WINDOW_DAYS))
    lear = LEAR(n_jobs=8).fit(X[tr], P[tr])
    fc["LEAR"].loc[day] = lear.predict(X.loc[[day]])[0]
    n_sel.append(lear.n_selected().mean())
    if k % GBM_REFIT_DAYS == 0:
        trl = L[(L.date < day)].dropna(subset=GBM_FEATURES + ["y"])
        gbm = make_gbm().fit(trl[GBM_FEATURES], trl["y"])
    te = L[L.date == day].sort_values("hour")
    fc["GBM"].loc[day] = gbm.predict(te[GBM_FEATURES])
    if k % 30 == 0:
        print(f"  {day.date()}  {time.time() - t0:.0f}s", flush=True)
fc["Ensemble"] = (fc["LEAR"] + fc["GBM"]) / 2
Y = P.loc[days].to_numpy(float)

rows = []
for m, F in fc.items():
    f = F.to_numpy(float)
    rows.append(dict(model=m, MAE=mae(Y, f), RMSE=rmse(Y, f), sMAPE_pct=smape(Y, f), rMAE=rmae(Y, f, fc["Naive"].to_numpy(float))))
metrics = pd.DataFrame(rows)
models = list(fc)
dm = pd.DataFrame(index=models, columns=models, dtype=float)
for a in models:
    for b_ in models:
        dm.loc[a, b_] = np.nan if a == b_ else dm_test(Y, fc[a].to_numpy(float), fc[b_].to_numpy(float))[1]

batt = {"1h": Battery(1.0, 1.0), "2h": Battery(1.0, 2.0), "4h": Battery(1.0, 4.0)}
brow = []
for bn, b in batt.items():
    pf = backtest(Y, Y, b)
    brow.append(dict(battery=bn, model="Perfect foresight", profit_eur_per_mw=pf["profit_eur"], pct_of_perfect=100.0, cycles=pf["cycles"], loss_days=pf["loss_days"]))
    for m, F in fc.items():
        r = backtest(Y, F.to_numpy(float), b)
        brow.append(dict(battery=bn, model=m, profit_eur_per_mw=r["profit_eur"], pct_of_perfect=100 * r["profit_eur"] / pf["profit_eur"], cycles=r["cycles"], loss_days=r["loss_days"]))
bess = pd.DataFrame(brow)

# error by hour and by month (for figures / discussion)
err_hour = pd.DataFrame({m: np.abs(Y - F.to_numpy(float)).mean(axis=0) for m, F in fc.items()})
err_month = pd.DataFrame({m: pd.Series(np.abs(Y - F.to_numpy(float)).mean(axis=1), index=days).groupby(days.to_period("M")).mean() for m, F in fc.items()})
err_month.index = err_month.index.astype(str)
gbm_imp = None
try:
    from sklearn.inspection import permutation_importance
    hold = L[(L.date >= TEST_START) & (L.date <= TEST_END)].dropna(subset=GBM_FEATURES + ["y"])
    pi = permutation_importance(gbm, hold[GBM_FEATURES], hold["y"], scoring="neg_mean_absolute_error", n_repeats=5, random_state=0)
    gbm_imp = pd.Series(pi.importances_mean, index=GBM_FEATURES).sort_values(ascending=False)
    gbm_imp.to_csv(RES / "gbm_permutation_importance.csv", header=["mae_increase_eur_mwh"])
except Exception as exc:  # pragma: no cover
    print("importance skipped:", exc)

for m, F in fc.items():
    F.round(3).to_csv(RES / f"forecast_{m.lower()}.csv")
P.loc[days].round(3).to_csv(RES / "actual_test.csv")
metrics.to_csv(RES / "metrics.csv", index=False)
dm.to_csv(RES / "dm_pvalues.csv")
bess.to_csv(RES / "bess_value.csv", index=False)
err_hour.to_csv(RES / "mae_by_hour.csv", index_label="hour")
err_month.to_csv(RES / "mae_by_month.csv", index_label="month")
summary = dict(test_start=str(TEST_START.date()), test_end=str(TEST_END.date()), test_days=len(days),
               lear_mean_selected_features=float(np.mean(n_sel)), lear_features_available=int(X.shape[1]),
               test_mean_price=float(Y.mean()), test_price_std=float(Y.std()), test_negative_hours=int((Y < 0).sum()),
               metrics=metrics.to_dict("records"), bess=bess.to_dict("records"), runtime_s=time.time() - t0)
json.dump(summary, open(RES / "summary.json", "w"), indent=2)
pd.set_option("display.width", 200)
print(metrics.round(3)); print(dm.round(4)); print(bess.round(1))
print(json.dumps({k: v for k, v in summary.items() if k not in ("metrics", "bess")}, indent=1))
