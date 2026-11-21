"""Risk-aware battery schedules from the probabilistic fans: does CVaR or a scenario-mean schedule beat the point forecast?

Uses the stored QRA+CQR fan (results/prob_quantiles_qra_cqr.csv.gz, 280 scored days) and the stored
ensemble forecasts. For each delivery day D, joint price scenarios are built from the fan with the
empirical copula of the ensemble's errors over the previous COPULA_W days (ex ante). Two fans:

* QRA+CQR fan     : the published fan as is
* Ens-centred fan : the same fan shifted so its median equals the ensemble point forecast
                    (the ensemble median was the better schedule driver in run_probabilistic.py)

Each scenario set drives the CVaR LP of dapf.risk for several risk weights lam; the schedule is fixed
and settled on actual prices, exactly like the point-forecast backtests.

    python scripts/run_risk_schedule.py      # about 2-4 minutes, single core
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dapf.bess import Battery, optimise_day  # noqa: E402
from dapf.probabilistic import TAUS  # noqa: E402
from dapf.risk import block_bootstrap_ci, optimise_day_cvar, scenarios, var_cvar  # noqa: E402

RES = ROOT / "results"
COPULA_W = 56
ALPHA = 0.10
LAMS = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, np.inf]
BATTERIES = {"1h": Battery(1.0, 1.0), "2h": Battery(1.0, 2.0), "4h": Battery(1.0, 4.0)}

A = pd.read_csv(RES / "actual_test.csv", index_col=0, parse_dates=True)
E = pd.read_csv(RES / "forecast_ensemble.csv", index_col=0, parse_dates=True)
qlong = pd.read_csv(RES / "prob_quantiles_qra_cqr.csv.gz", parse_dates=["date"])
qcols = [c for c in qlong.columns if c.startswith("q")]
qdays = pd.DatetimeIndex(qlong.date.unique())
Qf = qlong[qcols].to_numpy(float).reshape(len(qdays), 24, len(qcols))
pos = A.index.get_indexer(qdays)
assert (pos >= COPULA_W).all() and (pos >= 0).all()
Y, F = A.to_numpy(float), E.to_numpy(float)
y, f = Y[pos], F[pos]
i50 = int(np.argmin(np.abs(TAUS - 0.5)))
fans = {"QRA+CQR fan": Qf, "Ens-centred fan": Qf - Qf[..., i50:i50 + 1] + f[..., None]}
print(f"{len(qdays)} days {qdays[0].date()} .. {qdays[-1].date()}", flush=True)

t0 = time.time()
# scenarios per day (ex ante: errors of the COPULA_W days before D)
SC = {k: [scenarios(q[i], TAUS, Y[p - COPULA_W:p] - F[p - COPULA_W:p]) for i, p in enumerate(pos)] for k, q in fans.items()}

# scenario sanity: share of actual daily-mean prices inside the scenario 10-90 % range of daily means
scen_check = {}
for k in fans:
    dm = np.array([s.mean(axis=1) for s in SC[k]])           # (days, S)
    lo, hi = np.quantile(dm, 0.1, axis=1), np.quantile(dm, 0.9, axis=1)
    scen_check[k] = float(((y.mean(axis=1) >= lo) & (y.mean(axis=1) <= hi)).mean())

def settle(c, d, p, b):
    return float(p @ (d - c) - b.wear_eur_mwh * d.sum()), float(d.sum())

rows, daily = [], {}
for bn, b in BATTERIES.items():
    det = {"Perfect foresight": y, "Ensemble point": f, "QRA+CQR median": Qf[..., i50]}
    for name, fc in det.items():
        pr = np.empty(len(y)); thr = np.empty(len(y))
        for i in range(len(y)):
            c, d = optimise_day(fc[i], b); pr[i], thr[i] = settle(c, d, y[i], b)
        daily[(bn, name, "")] = pr
        rows.append(dict(battery=bn, schedule=name, fan="", lam=np.nan, profit=pr.sum(), thr=thr.sum(),
                         exante_var_breach=np.nan, exante_mean_bias=np.nan))
    for k in fans:
        for lam in LAMS:
            pr = np.empty(len(y)); thr = np.empty(len(y)); ev = np.empty(len(y)); vr = np.empty(len(y))
            for i in range(len(y)):
                c, d, info = optimise_day_cvar(SC[k][i], b, lam=lam, alpha=ALPHA)
                pr[i], thr[i] = settle(c, d, y[i], b); ev[i], vr[i] = info["exp_profit"], info["var"]
            nm = "Scenario mean (lam=0)" if lam == 0 else ("Pure CVaR" if np.isinf(lam) else f"CVaR lam={lam:g}")
            daily[(bn, nm, k)] = pr
            rows.append(dict(battery=bn, schedule=nm, fan=k, lam=lam, profit=pr.sum(), thr=thr.sum(),
                             exante_var_breach=float((pr < vr - 1e-9).mean()), exante_mean_bias=float((ev - pr).mean())))
        print(f"  {bn} {k} done {time.time() - t0:.0f} s", flush=True)

out = pd.DataFrame(rows)
recs = []
for _, r in out.iterrows():
    pr = daily[(r.battery, r.schedule, r.fan)]
    ref = daily[(r.battery, "Ensemble point", "")]
    pf = daily[(r.battery, "Perfect foresight", "")].sum()
    v, cv = var_cvar(pr, ALPHA)
    lo, hi = block_bootstrap_ci(pr - ref)
    recs.append(dict(pct_of_perfect=100 * r.profit / pf, eur_per_day=pr.mean(), daily_std=pr.std(ddof=1),
                     realised_cvar10_eur_day=cv, worst_day=pr.min(), loss_days=int((pr < 0).sum()),
                     cycles=r.thr / (BATTERIES[r.battery].eta * BATTERIES[r.battery].energy_mwh),
                     diff_vs_ens_eur_day=(pr - ref).mean(), ci_lo=lo, ci_hi=hi))
out = pd.concat([out, pd.DataFrame(recs)], axis=1)
out.to_csv(RES / "risk_schedule.csv", index=False)
pd.DataFrame({f"{b}|{s}|{k}": v for (b, s, k), v in daily.items()}, index=qdays.strftime("%Y-%m-%d")).round(2).to_csv(RES / "risk_schedule_daily.csv.gz")
summary = dict(days=int(len(qdays)), start=str(qdays[0].date()), end=str(qdays[-1].date()), copula_window=COPULA_W, alpha=ALPHA,
               lams=[None if np.isinf(l) else l for l in LAMS], scenarios_per_day=COPULA_W,
               daily_mean_in_scenario_10_90=scen_check, runtime_s=time.time() - t0)
(RES / "risk_summary.json").write_text(json.dumps(summary, indent=1))
pd.set_option("display.width", 250)
print(out.drop(columns=["lam", "thr"]).round(2).to_string(index=False))
print(json.dumps(summary, indent=1))
