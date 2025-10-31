#!/usr/bin/env python
"""Compute tracking error and monthly hit-rate versus benchmark for OOS returns.

The script expects `results/oos_returns_all_strategies_*.csv` produced by the
baseline comparison harness. By default it uses the latest file, treats
`Risk Parity` as the portfolio and `60/40` as benchmark, and saves a CSV +
JSON summary under `results/tracking_metrics/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PORTFOLIO_COLUMN = "Risk Parity"
BENCHMARK_COLUMN = "60/40"
OUTPUT_DIR = Path("results") / "tracking_metrics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _latest_oos_file() -> Path:
    candidates = sorted(Path("results").glob("oos_returns_all_strategies_*.csv"))
    if not candidates:
        raise FileNotFoundError("No OOS returns file found in results/ directory.")
    return candidates[-1]


def compute_metrics(df: pd.DataFrame) -> dict[str, float]:
    if PORTFOLIO_COLUMN not in df or BENCHMARK_COLUMN not in df:
        raise KeyError(
            f"Columns {PORTFOLIO_COLUMN!r} and/or {BENCHMARK_COLUMN!r} not found in dataset."
        )

    portfolio = df[PORTFOLIO_COLUMN].astype(float)
    benchmark = df[BENCHMARK_COLUMN].astype(float)

    diff = portfolio - benchmark
    daily_te = diff.std(ddof=1)
    tracking_error_annual = float(daily_te * np.sqrt(252.0))

    monthly = df.resample("M").sum()
    monthly_rel = monthly[PORTFOLIO_COLUMN] - monthly[BENCHMARK_COLUMN]
    hit_rate = float((monthly_rel > 0).mean())

    active_return_annual = float((portfolio - benchmark).mean() * 252.0)

    return {
        "tracking_error_annual": tracking_error_annual,
        "tracking_error_daily": float(daily_te),
        "hit_rate_monthly": hit_rate,
        "active_return_annual": active_return_annual,
    }


def main() -> None:
    latest = _latest_oos_file()
    print(f"Using OOS returns file: {latest}")
    returns = pd.read_csv(latest, index_col=0, parse_dates=True)

    metrics = compute_metrics(returns)
    print("📊 Tracking Metrics:")
    for key, value in metrics.items():
        print(f"   • {key}: {value:.4f}")

    timestamp = latest.stem.split("_")[-1]
    summary_path = OUTPUT_DIR / f"tracking_summary_{timestamp}.json"
    csv_path = OUTPUT_DIR / f"tracking_monthly_{timestamp}.csv"

    returns.resample("M").sum()[[PORTFOLIO_COLUMN, BENCHMARK_COLUMN]].to_csv(csv_path)
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "source": latest.name,
                "portfolio": PORTFOLIO_COLUMN,
                "benchmark": BENCHMARK_COLUMN,
                "metrics": metrics,
            },
            fh,
            indent=2,
        )

    print()
    print("Saved:")
    print(f"   • Monthly aggregates: {csv_path}")
    print(f"   • Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
