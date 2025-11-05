from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from arara_quant.backtesting.engine import run_backtest
from arara_quant.config import Settings, reset_settings_cache


def _make_settings(tmp_path: Path) -> Settings:
    data_dir = tmp_path / "data"
    configs_dir = tmp_path / "configs"
    logs_dir = tmp_path / "logs"
    for directory in (data_dir / "processed", configs_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return Settings.from_env(
        overrides={
            "project_root": tmp_path,
            "DATA_DIR": data_dir,
            "CONFIGS_DIR": configs_dir,
            "LOGS_DIR": logs_dir,
        }
    )


def teardown_function() -> None:
    reset_settings_cache()


def test_run_backtest_produces_walkforward_metrics(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)

    dates = pd.bdate_range("2022-01-03", periods=320)
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(
        rng.normal(0.0006, 0.01, size=(len(dates), 3)),
        index=dates,
        columns=["AAA", "BBB", "CCC"],
    )
    returns_path = settings.processed_data_dir / "synthetic_returns.parquet"
    returns.to_parquet(returns_path)

    config_text = f"""
base_currency: USD
data:
  returns: {returns_path}
walkforward:
  train_days: 126
  test_days: 21
  purge_days: 5
  embargo_days: 5
  evaluation_horizons: [21, 63]
portfolio:
  capital: 1_000_000
optimizer:
  lambda: 5.0
  eta: 0.1
  tau: 0.25
  min_weight: 0.0
  max_weight: 0.5
  solver: OSQP
estimators:
  mu: {{ method: shrunk_50, window_days: 126, strength: 0.5 }}
  sigma: {{ method: ledoit_wolf, window_days: 126 }}
""".strip()

    config_path = settings.configs_dir / "wf_config.yaml"
    config_path.write_text(config_text, encoding="utf-8")

    result = run_backtest(config_path, dry_run=False, settings=settings)

    assert result.split_metrics is not None and not result.split_metrics.empty
    assert set(result.split_metrics.columns) >= {
        "total_return",
        "sharpe_ratio",
        "turnover",
    }

    assert result.horizon_metrics is not None
    horizons = result.horizon_metrics.set_index("horizon_days")
    assert {21, 63}.intersection(horizons.index) == {21, 63}
