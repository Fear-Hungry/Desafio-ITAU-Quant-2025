from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from arara_quant.backtesting import run_backtest
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


def _write_returns(tmp_path: Path) -> Path:
    dates = pd.bdate_range("2020-01-01", periods=180)
    data = {
        "AAA": np.full(len(dates), 0.0005),
        "BBB": np.linspace(0.0002, 0.001, len(dates)),
        "CCC": np.full(len(dates), 0.0001),
    }
    returns = pd.DataFrame(data, index=dates)
    returns_path = tmp_path / "data" / "processed" / "returns_fixture.csv"
    returns.to_csv(returns_path)
    return returns_path


def _write_config(tmp_path: Path, returns_path: Path) -> Path:
    template = """
base_currency: USD
data:
  returns: {returns_path}
walkforward:
  train_days: 60
  test_days: 20
  purge_days: 5
  embargo_days: 0
optimizer:
  tau: 0.20
estimators:
  costs:
    linear_bps: 10
"""
    config_path = tmp_path / "configs" / "backtest.yaml"
    config_content = template.format(returns_path=returns_path).strip()
    config_path.write_text(config_content, encoding="utf-8")
    return config_path


def _write_config_with_risk(tmp_path: Path, returns_path: Path) -> Path:
    template = """
base_currency: USD
data:
  returns: {returns_path}
walkforward:
  train_days: 60
  test_days: 20
  purge_days: 5
  embargo_days: 0
optimizer:
  lambda: 3.0
  eta: 0.0
  tau: 0.20
  min_weight: 0.0
  max_weight: 0.9
estimators:
  mu:
    method: simple
  sigma:
    method: ledoit_wolf
  costs:
    linear_bps: 0.0
portfolio:
  capital: 1_000_000
  risk:
    budgets:
      - {{"name": "Growth", "tickers": ["AAA", "BBB"], "max_weight": 0.35}}
      - {{"name": "Defensive", "tickers": ["CCC"], "min_weight": 0.65}}
    max_leverage: 1.0
  rounding:
    lot_sizes: 0.0001
""".strip()
    config_content = template.format(returns_path=returns_path).strip()
    config_path = tmp_path / "configs" / "backtest_with_risk.yaml"
    config_path.write_text(config_content, encoding="utf-8")
    return config_path


def teardown_function() -> None:  # pragma: no cover - helper cleanup
    reset_settings_cache()


def test_run_backtest_dry_run(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    returns_path = _write_returns(tmp_path)
    config_path = _write_config(tmp_path, returns_path)

    result = run_backtest(config_path, settings=settings, dry_run=True)
    assert result.dry_run is True
    assert result.metrics is None
    assert result.ledger is None
    summary = result.to_dict()
    assert summary["status"] == "preview"


def test_run_backtest_produces_metrics(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    returns_path = _write_returns(tmp_path)
    config_path = _write_config(tmp_path, returns_path)

    result = run_backtest(config_path, settings=settings, dry_run=False)

    assert result.dry_run is False
    assert result.metrics is not None
    assert result.metrics.total_return > 0
    assert result.ledger is not None
    nav_end = result.ledger.frame["nav"].iloc[-1]
    assert nav_end > 1.0

    assert result.trades is not None
    turnovers = result.trades["turnover"].astype(float)
    if not turnovers.empty:
        assert turnovers.iloc[0] <= 1.05
        if len(turnovers) > 1:
            assert (turnovers.iloc[1:] <= 0.20 + 1e-4).all()

    weights = result.weights
    assert weights is not None
    assert not weights.empty
    pd.testing.assert_index_equal(weights.columns, pd.Index(["AAA", "BBB", "CCC"]))

    payload = result.to_dict(include_timeseries=True)
    assert "metrics" in payload
    assert "ledger" in payload
    assert payload["status"] == "completed"


def test_run_backtest_applies_risk_budgets(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    returns_path = _write_returns(tmp_path)
    config_path = _write_config_with_risk(tmp_path, returns_path)

    result = run_backtest(config_path, settings=settings, dry_run=False)

    weights = result.weights
    assert weights is not None
    assert not weights.empty
    weights = weights.reindex(columns=["AAA", "BBB", "CCC"]).astype(float)
    growth_band = weights[["AAA", "BBB"]].sum(axis=1)
    defensive_weight = weights["CCC"]
    assert (growth_band <= 0.35 + 1e-3).all()
    assert (defensive_weight >= 0.65 - 1e-3).all()
