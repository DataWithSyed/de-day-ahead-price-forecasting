#!/usr/bin/env bash
# Reproduce everything from the cached data extract (add --download to refresh it from the Energy-Charts API).
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" == "--download" ]]; then python scripts/fetch_data.py; fi
python scripts/run_backtest.py
python scripts/make_figures.py
python -m pytest -q
