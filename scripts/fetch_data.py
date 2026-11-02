"""Fetch DE-LU day-ahead prices and TSO day-ahead load/solar/wind forecasts (Energy-Charts, CC BY 4.0),
plus the native 15-minute day-ahead prices since the 15-min MTU go-live (used by run_quarterhour.py)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dapf.data import build_dataset, fetch_qh_prices  # noqa: E402

if __name__ == "__main__":
    df = build_dataset(out=ROOT / "data/de_lu_hourly.csv.gz")
    print(df.describe().T)
    print(df.isna().sum())
    qh = fetch_qh_prices(out=ROOT / "data/de_lu_qh_price.csv.gz")
    print(f"15-min prices: {len(qh)} values {qh.index.min()} .. {qh.index.max()}, missing {qh.isna().sum()}")
