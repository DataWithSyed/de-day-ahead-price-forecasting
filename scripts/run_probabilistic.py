"""Probabilistic extension: prediction intervals and quantile forecasts for the DE-LU day-ahead price.

Post-processes the stored out-of-sample point forecasts of run_backtest.py (results/forecast_*.csv,
results/actual_test.csv). For every delivery day D the quantiles use only forecasts and prices of
earlier days, so the result is ex ante. The first WARMUP days of the test year are calibration only;
all methods are scored on the same remaining days.

Methods (19 quantiles, 5 % .. 95 %):
* HS-Naive  : naive point forecast + empirical quantiles of its past errors (benchmark)
* HS-Ens    : ensemble point forecast + empirical quantiles of its past errors
* CP-Ens    : ensemble +/- split-conformal radius of past absolute errors (symmetric)
* QRA       : Quantile Regression Averaging of the LEAR and GBM forecasts (Nowotarski & Weron 2015)
* QRA+CQR   : QRA recalibrated on its own interval misses of the previous CQR_DAYS days (pooled over
              hours), conformalized quantile regression (Romano et al. 2019)
Rolling calibration windows of 56 days (main) and 28 days (sensitivity), per hour of the day.
Scoring starts after WARMUP + CQR_DAYS days so that every method is scored on the same days.

    python scripts/run_probabilistic.py        # about 15-20 minutes, single core (QRA LPs)
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dapf.bess import Battery, backtest  # noqa: E402
from dapf.evaluate import mae  # noqa: E402
from dapf.probabilistic import (TAUS, conformal_symmetric, conformalize, coverage,  # noqa: E402
                                dm_pinball, historical_simulation, pinball, prob_below, qra, winkler)

RES = ROOT / "results"
WARMUP = 56
CQR_DAYS = 28
WINDOWS = (56, 28)
MAIN_W = 56


def load(name):
    return pd.read_csv(RES / f"{name}.csv", index_col=0, parse_dates=True)


Ydf = load("actual_test")
days = Ydf.index
Y = Ydf.to_numpy(float)
F = {m: load(f"forecast_{m}").to_numpy(float) for m in ("naive", "lear", "gbm", "ensemble")}
ev = np.arange(WARMUP, len(days))          # days with quantile forecasts
print(f"test days {len(days)}, forecast days {len(ev)}: {days[ev[0]].date()} .. {days[ev[-1]].date()}", flush=True)

t0 = time.time()
Q = {}  # (method, window) -> (len(ev), 24, n_tau)
for W in WINDOWS:
    Q[("HS-Naive", W)] = np.stack([historical_simulation(Y, F["naive"], t, W) for t in ev])
    Q[("HS-Ens", W)] = np.stack([historical_simulation(Y, F["ensemble"], t, W) for t in ev])
    Q[("CP-Ens", W)] = np.stack([conformal_symmetric(Y, F["ensemble"], t, W) for t in ev])
    q = []
    for i, t in enumerate(ev):
        q.append(qra(Y, [F["lear"], F["gbm"]], t, W))
        if i % 50 == 0:
            print(f"  QRA W={W}: day {i}/{len(ev)}  {time.time() - t0:.0f} s", flush=True)
    Q[("QRA", W)] = np.stack(q)
    qr = Q[("QRA", W)]
    cq = np.full_like(qr, np.nan)
    for i in range(CQR_DAYS, len(ev)):
        cq[i] = conformalize(qr[i - CQR_DAYS:i], Y[ev[i - CQR_DAYS:i]], qr[i])
    Q[("QRA+CQR", W)] = cq

# score every method on the same days (CQR needs CQR_DAYS of QRA history)
Q = {k: v[CQR_DAYS:] for k, v in Q.items()}
ev = ev[CQR_DAYS:]
print(f"scored days {len(ev)}: {days[ev[0]].date()} .. {days[ev[-1]].date()}", flush=True)
y = Y[ev]
i05, i10, i25, i50, i75, i90, i95 = (int(np.argmin(np.abs(TAUS - v))) for v in (0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95))

# ---- negative-price probability: climatology benchmark = share of negative hours at the same hour, past W days
neg = (y < 0).astype(float)
clim = np.stack([(Y[t - MAIN_W:t] < 0).mean(axis=0) for t in ev])
brier_clim = float(np.mean((clim - neg) ** 2))

rows = []
for (m, W), q in Q.items():
    pb = pinball(y, q)
    p_neg = prob_below(q, 0.0)
    brier = float(np.mean((p_neg - neg) ** 2))
    rows.append(dict(method=m, window_days=W,
                     pinball=float(pb.mean()),
                     median_MAE=mae(y, q[..., i50]),
                     cov50=coverage(y, q[..., i25], q[..., i75]),
                     cov80=coverage(y, q[..., i10], q[..., i90]),
                     cov90=coverage(y, q[..., i05], q[..., i95]),
                     width90=float(np.mean(q[..., i95] - q[..., i05])),
                     winkler90=winkler(y, q[..., i05], q[..., i95], 0.10),
                     brier_neg=brier, brier_skill_vs_clim=1 - brier / brier_clim))
metrics = pd.DataFrame(rows)
print(metrics.round(3).to_string(index=False), flush=True)

# ---- DM tests on daily pinball loss (main window)
names = ["HS-Naive", "HS-Ens", "CP-Ens", "QRA", "QRA+CQR"]
L = {m: pinball(y, Q[(m, MAIN_W)]) for m in names}
dm = pd.DataFrame(index=names, columns=names, dtype=float)
for a in names:
    for b in names:
        dm.loc[a, b] = np.nan if a == b else dm_pinball(L[a], L[b])[1]

# ---- reliability (share of actuals below each quantile), main window
rel = pd.DataFrame({m: [(y < Q[(m, MAIN_W)][..., j]).mean() for j in range(len(TAUS))] for m in names},
                   index=pd.Index(TAUS, name="tau"))

# ---- coverage of the 80 % interval by hour and by month (main window)
cov_hour = pd.DataFrame({m: ((y >= Q[(m, MAIN_W)][..., i10]) & (y <= Q[(m, MAIN_W)][..., i90])).mean(axis=0) for m in names})
cov_hour.index.name = "hour"
evd = days[ev]
cov_month = pd.DataFrame({m: pd.Series(((y >= Q[(m, MAIN_W)][..., i10]) & (y <= Q[(m, MAIN_W)][..., i90])).mean(axis=1),
                                       index=evd).groupby(evd.to_period("M")).mean() for m in names})
cov_month.index = cov_month.index.astype(str)
width_month = pd.DataFrame({m: pd.Series((Q[(m, MAIN_W)][..., i95] - Q[(m, MAIN_W)][..., i05]).mean(axis=1),
                                         index=evd).groupby(evd.to_period("M")).mean() for m in names})
width_month.index = width_month.index.astype(str)

# ---- battery: does the QRA median make a better schedule than the ensemble mean? same scored days
brow = []
for bn, bt in {"1h": Battery(1.0, 1.0), "2h": Battery(1.0, 2.0), "4h": Battery(1.0, 4.0)}.items():
    pf = backtest(y, y, bt)["profit_eur"]
    for m, f in {"Ensemble": F["ensemble"][ev], "QRA median": Q[("QRA", MAIN_W)][..., i50], "QRA+CQR median": Q[("QRA+CQR", MAIN_W)][..., i50], "Naive": F["naive"][ev]}.items():
        r = backtest(y, f, bt)
        brow.append(dict(battery=bn, schedule_from=m, profit_eur_per_mw=r["profit_eur"], pct_of_perfect=100 * r["profit_eur"] / pf,
                         perfect_foresight_eur_per_mw=pf, loss_days=r["loss_days"]))
bess = pd.DataFrame(brow)
print(bess.round(1).to_string(index=False), flush=True)

# ---- save
metrics.to_csv(RES / "prob_metrics.csv", index=False)
dm.to_csv(RES / "prob_dm_pvalues.csv")
rel.to_csv(RES / "prob_reliability.csv")
cov_hour.to_csv(RES / "prob_coverage80_by_hour.csv")
cov_month.to_csv(RES / "prob_coverage80_by_month.csv")
width_month.to_csv(RES / "prob_width90_by_month.csv")
bess.to_csv(RES / "prob_bess_median.csv", index=False)
qmain = Q[("QRA+CQR", MAIN_W)]
long = pd.DataFrame(qmain.reshape(-1, len(TAUS)).round(3), columns=[f"q{int(round(100 * t)):02d}" for t in TAUS])
long.insert(0, "hour", np.tile(np.arange(24), len(ev)))
long.insert(0, "date", np.repeat(evd.strftime("%Y-%m-%d"), 24))
long.to_csv(RES / "prob_quantiles_qra_cqr.csv.gz", index=False)
summary = dict(scored_start=str(evd[0].date()), scored_end=str(evd[-1].date()), scored_days=int(len(ev)),
               warmup_days=WARMUP, cqr_days=CQR_DAYS, windows=list(WINDOWS), main_window=MAIN_W, taus=[float(t) for t in TAUS],
               negative_hours_scored=int(neg.sum()), brier_climatology=brier_clim,
               metrics=metrics.to_dict("records"), runtime_s=time.time() - t0)
(RES / "prob_summary.json").write_text(json.dumps(summary, indent=1))
print(dm.round(4)); print(f"done in {time.time() - t0:.0f} s")
