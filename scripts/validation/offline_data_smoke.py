#!/usr/bin/env python3
"""Offline data validation smoke test.

This script materialises a tiny deterministic dataset, runs the preprocessing
step to generate returns, and executes a short walk-forward backtest. It is
designed to run without network access so CI can flag regressions in the
data/portfolio stack without depending on Yahoo Finance or external APIs.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from arara_quant.backtesting import run_backtest
from arara_quant.config import Settings
from arara_quant.data.loader import preprocess_data


def _generate_prices(n_days: int = 90, seed: int = 42) -> pd.DataFrame:
    """Create a small deterministic price panel."""

    dates = pd.bdate_range("2023-01-02", periods=n_days)
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0004, 0.01, size=(len(dates), 3))
    cumulative = np.exp(np.cumsum(log_returns, axis=0))
    prices = 100.0 * cumulative
    return pd.DataFrame(prices, index=dates, columns=["AAA", "BBB", "CCC"])


def main() -> None:
    settings = Settings.from_env()
    raw_dir = Path(settings.raw_data_dir)
    processed_dir = Path(settings.processed_data_dir)

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    prices = _generate_prices()
    raw_file = raw_dir / "mini_prices.csv"
    prices.to_csv(raw_file, index=True)

    preprocess_data(raw_file.name, "mini_returns.parquet")
    returns_path = processed_dir / "mini_returns.parquet"

    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = Path(tmp_dir) / "mini_backtest.yaml"
        config_text = f"""base_currency: USD
data:
  returns: {returns_path}
walkforward:
  train_days: 42
  test_days: 10
  purge_days: 2
  embargo_days: 2
  evaluation_horizons: [10]
portfolio:
  capital: 1_000_000
optimizer:
  lambda: 5.0
  eta: 0.0
  tau: 0.25
  min_weight: 0.0
  max_weight: 0.8
  solver: OSQP
estimators:
  mu: {{ method: shrunk_50, window_days: 42, strength: 0.5 }}
  sigma: {{ method: ledoit_wolf, window_days: 42 }}
"""
        config_path.write_text(config_text, encoding="utf-8")

        result = run_backtest(config_path, dry_run=False, settings=settings)
        if result.metrics is None:
            raise SystemExit("Backtest did not produce metrics")

        windows = 0 if result.split_metrics is None else len(result.split_metrics)
        print("✓ Offline data validation smoke completed")
        print(f"  NAV: {result.metrics.cumulative_nav:.4f}")
        print(f"  Windows: {windows}")


if __name__ == "__main__":
    main()
