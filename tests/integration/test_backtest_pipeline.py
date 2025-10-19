from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from itau_quant.backtesting import run_backtest
from itau_quant.config import Settings, reset_settings_cache


def _make_settings(tmp_path: Path) -> Settings:
    data_dir = tmp_path / "data"
    configs_dir = tmp_path / "configs"
    logs_dir = tmp_path / "logs"
    for directory in (data_dir / "processed", configs_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    reset_settings_cache()
    return Settings.from_env(
        overrides={
            "project_root": tmp_path,
            "DATA_DIR": data_dir,
            "CONFIGS_DIR": configs_dir,
            "LOGS_DIR": logs_dir,
        }
    )


def _write_returns(tmp_path: Path, file_name: str = "returns.csv") -> Path:
    dates = pd.bdate_range("2020-01-01", periods=120)
    data = {
        "AAA": np.linspace(0.0005, 0.001, len(dates)),
        "BBB": np.linspace(-0.0002, 0.0008, len(dates)),
        "CCC": np.full(len(dates), 0.0003),
    }
    frame = pd.DataFrame(data, index=dates)
    path = tmp_path / "data" / "processed" / file_name
    frame.to_csv(path)
    return path


def _write_config(tmp_path: Path, returns_path: Path, file_name: str = "backtest.yaml") -> Path:
    template = f"""
base_currency: USD
data:
  returns: {returns_path.as_posix()}
walkforward:
  train_days: 60
  test_days: 20
  purge_days: 2
  embargo_days: 2
optimizer:
  lambda: 4.0
  eta: 0.1
  tau: 0.25
estimators:
  mu:
    method: simple
  sigma:
    method: ledoit_wolf
portfolio:
  capital: 1_000_000
  rounding:
    lot_sizes: 0.0001
"""
    config_path = tmp_path / "configs" / file_name
    config_path.write_text(template.strip(), encoding="utf-8")
    return config_path


def test_backtest_pipeline_end_to_end(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    returns_path = _write_returns(tmp_path)
    config_path = _write_config(tmp_path, returns_path)

    result = run_backtest(config_path, settings=settings, dry_run=False)

    assert result.metrics is not None
    assert result.metrics.total_return > -1.0
    assert result.ledger is not None
    assert not result.ledger.frame.empty
    assert result.notes == []

    payload = result.to_dict(include_timeseries=True)
    assert payload["status"] == "completed"
    assert "metrics" in payload and "ledger" in payload
    assert len(payload.get("weights", [])) > 0


def test_backtest_pipeline_fails_with_insufficient_data(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    returns_path = _write_returns(tmp_path, file_name="short_returns.csv")
    config_path = _write_config(tmp_path, returns_path)

    # overwrite returns with short series
    short_dates = pd.bdate_range("2020-01-01", periods=30)
    pd.DataFrame(
        {"AAA": 0.001, "BBB": 0.0005, "CCC": 0.0002},
        index=short_dates,
    ).to_csv(returns_path)

    with pytest.raises(ValueError):
        run_backtest(config_path, settings=settings, dry_run=False)
