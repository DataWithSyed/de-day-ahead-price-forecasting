"""Figures from results/ (run scripts/run_backtest.py first)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]; RES, FIG = ROOT / "results", ROOT / "figures"; FIG.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .3, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
COL = {"Naive": "#9a9a9a", "LEAR": "#1f5f8b", "GBM": "#e07b39", "Ensemble": "#4c9a6a", "Perfect foresight": "#222222"}
load = lambda m: pd.read_csv(RES / f"forecast_{m.lower()}.csv", index_col=0, parse_dates=True)
A = pd.read_csv(RES / "actual_test.csv", index_col=0, parse_dates=True)
F = {m: load(m) for m in ("Naive", "LEAR", "GBM", "Ensemble")}
met = pd.read_csv(RES / "metrics.csv"); bess = pd.read_csv(RES / "bess_value.csv")

# 1. headline: a test week + MAE bars + BESS value
fig = plt.figure(figsize=(14, 7.5)); gs = fig.add_gridspec(2, 3)
ax = fig.add_subplot(gs[0, :])
wk = A.index[(A.index >= "2026-04-13") & (A.index < "2026-04-20")]
t = pd.date_range(wk[0], periods=24 * len(wk), freq="1h")
ax.plot(t, A.loc[wk].to_numpy().ravel(), color="k", lw=1.8, label="actual DE-LU day-ahead")
for m in ("Naive", "LEAR", "GBM", "Ensemble"):
    ax.plot(t, F[m].loc[wk].to_numpy().ravel(), color=COL[m], lw=1, label=m, ls="--" if m == "Naive" else "-")
ax.set_ylabel("EUR/MWh"); ax.set_title("Out-of-sample day-ahead forecasts, week of 13 Apr 2026 (hourly averages)"); ax.legend(ncol=5, fontsize=8)
ax = fig.add_subplot(gs[1, 0])
ax.bar(met.model, met.MAE, color=[COL[m] for m in met.model])
for i, (v, r) in enumerate(zip(met.MAE, met.rMAE)):
    ax.text(i, v + 0.3, f"{v:.1f}\nrMAE {r:.2f}", ha="center", fontsize=7)
ax.set_ylabel("MAE (EUR/MWh)"); ax.set_title("Accuracy, 364 test days (Nov 2025 - Oct 2026)", fontsize=9); ax.set_ylim(0, met.MAE.max() * 1.25)
ax = fig.add_subplot(gs[1, 1])
eh = pd.read_csv(RES / "mae_by_hour.csv", index_col=0)
for m in eh:
    ax.plot(eh.index, eh[m], color=COL[m], label=m, ls="--" if m == "Naive" else "-")
ax.set_xlabel("hour of day (local)"); ax.set_ylabel("MAE (EUR/MWh)"); ax.set_title("Error by delivery hour", fontsize=9); ax.legend(fontsize=7)
ax = fig.add_subplot(gs[1, 2])
b2 = bess[bess.battery == "2h"]
ax.bar(b2.model, b2.profit_eur_per_mw / 1e3, color=[COL[m] for m in b2.model])
for i, (v, pc) in enumerate(zip(b2.profit_eur_per_mw / 1e3, b2.pct_of_perfect)):
    ax.text(i, v + 0.5, f"{pc:.0f} %", ha="center", fontsize=7)
ax.set_ylabel("k EUR per MW over the test year"); ax.set_title("1 MW / 2 MWh BESS arbitrage value of each forecast", fontsize=9)
ax.tick_params(axis="x", labelsize=7, rotation=15)
fig.tight_layout(); fig.savefig(FIG / "01_headline.png"); plt.close()

# 2. monthly MAE + price level
em = pd.read_csv(RES / "mae_by_month.csv", index_col=0)
fig, ax = plt.subplots(figsize=(11, 3.8))
x = np.arange(len(em)); w = 0.2
for i, m in enumerate(em):
    ax.bar(x + (i - 1.5) * w, em[m], w, color=COL[m], label=m)
ax2 = ax.twinx(); mp = A.mean(axis=1).groupby(A.index.to_period("M")).mean()
ax2.plot(x, mp.values, "k.-", label="mean price"); ax2.set_ylabel("mean price (EUR/MWh)"); ax2.grid(False)
ax.set_xticks(x); ax.set_xticklabels(em.index, rotation=45, fontsize=7); ax.set_ylabel("MAE (EUR/MWh)")
ax.set_title("Monthly MAE by model"); ax.legend(fontsize=7, ncol=4, loc="upper left"); ax2.legend(fontsize=7, loc="upper right")
fig.tight_layout(); fig.savefig(FIG / "02_monthly_mae.png"); plt.close()

# 3. BESS value by duration + GBM importance
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
piv = bess.pivot(index="battery", columns="model", values="pct_of_perfect")[["Naive", "LEAR", "GBM", "Ensemble"]]
piv.plot.bar(ax=ax[0], color=[COL[m] for m in piv.columns], rot=0)
ax[0].set_ylim(80, 100); ax[0].set_ylabel("% of perfect-foresight profit"); ax[0].set_xlabel("battery duration (1 MW)"); ax[0].set_title("Value captured vs forecast and duration", fontsize=9); ax[0].legend(fontsize=7, ncol=4, loc="upper left")
imp_p = RES / "gbm_permutation_importance.csv"
if imp_p.exists():
    imp = pd.read_csv(imp_p, index_col=0).iloc[:12, 0][::-1]
    ax[1].barh(imp.index, imp.values, color=COL["GBM"]); ax[1].set_xlabel("MAE increase when permuted (EUR/MWh)")
    ax[1].set_title("GBM permutation importance (last fitted model on the test period; partly in-sample)", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "03_bess_duration_importance.png"); plt.close()
print("figures written")
