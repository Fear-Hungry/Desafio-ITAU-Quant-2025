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


def _run_synthetic_backtest(
    tmp_path: Path,
) -> tuple["BacktestResult", pd.DataFrame, Settings]:
    """Execute a small walk-forward backtest used across tests."""

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
    return result, returns, settings


def teardown_function() -> None:
    reset_settings_cache()


def test_run_backtest_produces_walkforward_metrics(tmp_path: Path) -> None:
    result, _, _ = _run_synthetic_backtest(tmp_path)

    assert result.split_metrics is not None and not result.split_metrics.empty
    assert set(result.split_metrics.columns) >= {
        "total_return",
        "sharpe_ratio",
        "turnover",
    }

    assert result.horizon_metrics is not None
    horizons = result.horizon_metrics.set_index("horizon_days")
    assert {21, 63}.intersection(horizons.index) == {21, 63}


def test_turnover_matches_half_l1_pretrade(tmp_path: Path) -> None:
    result, returns, _ = _run_synthetic_backtest(tmp_path)

    assert result.split_metrics is not None and not result.split_metrics.empty
    assert result.weights is not None and not result.weights.empty

    split_metrics = result.split_metrics.copy()
    split_metrics["test_start"] = pd.to_datetime(split_metrics["test_start"])
    split_metrics = split_metrics.set_index("test_start").sort_index()

    weights = result.weights.sort_index()
    returns = returns.sort_index()

    prev_post_trade = pd.Series(0.0, index=weights.columns, dtype=float)
    prev_date: pd.Timestamp | None = None

    for date, post_trade in weights.iterrows():
        post_trade = post_trade.astype(float)
        if prev_date is None:
            pre_trade = prev_post_trade.reindex(post_trade.index).fillna(0.0)
        else:
            seg = returns.loc[(returns.index > prev_date) & (returns.index <= date)]
            drifted = prev_post_trade.copy()
            if not seg.empty:
                growth = (1.0 + seg).prod(axis=0)
                growth = growth.reindex(drifted.index).fillna(1.0)
                drifted = drifted * growth
                total = float(drifted.sum())
                if total > 0 and np.isfinite(total):
                    drifted = drifted / total
            pre_trade = drifted.reindex(post_trade.index).fillna(0.0)

        turnover_half = 0.5 * float(np.abs(post_trade - pre_trade).sum())
        reported = float(split_metrics.loc[date, "turnover"])
        assert np.isclose(
            turnover_half,
            reported,
            atol=1e-8,
        ), f"Turnover mismatch on {date.date()}: computed={turnover_half}, reported={reported}"

        prev_post_trade = post_trade
        prev_date = date
