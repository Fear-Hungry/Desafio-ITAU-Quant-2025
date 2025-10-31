from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from itau_quant.config import Settings, reset_settings_cache
from itau_quant.optimization.solvers import run_optimizer


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
    dates = pd.bdate_range("2022-01-03", periods=120)
    returns = pd.DataFrame(
        {
            "AAA": np.linspace(0.0012, 0.0022, len(dates)),
            "BBB": np.linspace(0.0009, 0.0016, len(dates)),
            "CCC": np.linspace(-0.0002, 0.0003, len(dates)),
        },
        index=dates,
    )
    path = tmp_path / "data" / "processed" / "returns_optimizer.csv"
    returns.to_csv(path)
    return path


def _write_config(tmp_path: Path, returns_path: Path) -> Path:
    config = tmp_path / "configs" / "optimizer.yaml"
    config.write_text(
        f"""
base_currency: USD
data:
  returns: {returns_path}
optimizer:
  lambda: 4.0
  eta: 0.1
  tau: 0.25
  min_weight: 0.0
  max_weight: 0.6
estimators:
  mu:
    method: simple
    window_days: 60
  sigma:
    method: ledoit_wolf
    window_days: 60
  costs:
    linear_bps: 12
state:
  previous_weights:
    AAA: 0.3
    BBB: 0.4
    CCC: 0.3
""".strip(),
        encoding="utf-8",
    )
    return config


def _write_config_with_portfolio_budget(tmp_path: Path, returns_path: Path) -> Path:
    config = tmp_path / "configs" / "optimizer_with_budget.yaml"
    config.write_text(
        f"""
base_currency: USD
data:
  returns: {returns_path}
optimizer:
  lambda: 4.0
  eta: 0.1
  tau: 1.0
  min_weight: 0.0
  max_weight: 0.9
estimators:
  mu:
    method: simple
    window_days: 60
  sigma:
    method: ledoit_wolf
    window_days: 60
  costs:
    linear_bps: 8
state:
  previous_weights:
    AAA: 0.2
    BBB: 0.2
    CCC: 0.6
portfolio:
  risk:
    budgets:
      - name: cap_high_return_assets
        max_weight: 0.3
        tickers:
          - AAA
          - BBB
""".strip(),
        encoding="utf-8",
    )
    return config


def _write_cvar_config(tmp_path: Path, returns_path: Path) -> Path:
    config = tmp_path / "configs" / "optimizer_cvar.yaml"
    config.write_text(
        f"""
base_currency: USD
data:
  returns: {returns_path}
risk_limits:
  cvar_alpha: 0.9
  cvar_max: 0.06
optimizer:
  objective: mean_cvar
  lambda: 2.5
  min_weight: 0.0
  max_weight: 0.7
  target_return: 0.0005
estimators:
  mu:
    method: simple
    window_days: 60
  sigma:
    method: ledoit_wolf
    window_days: 60
state:
  previous_weights:
    AAA: 0.34
    BBB: 0.33
    CCC: 0.33
""".strip(),
        encoding="utf-8",
    )
    return config


def teardown_function() -> None:
    reset_settings_cache()


def test_run_optimizer_dry_run(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    returns_path = _write_returns(tmp_path)
    config_path = _write_config(tmp_path, returns_path)

    result = run_optimizer(config_path, dry_run=True, settings=settings)
    assert result.dry_run is True
    assert result.weights is None
    payload = result.to_dict()
    assert payload["status"] == "preview"


def test_run_optimizer_executes_and_returns_weights(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    returns_path = _write_returns(tmp_path)
    config_path = _write_config(tmp_path, returns_path)

    result = run_optimizer(config_path, dry_run=False, settings=settings)

    assert result.dry_run is False
    assert result.weights is not None
    assert abs(result.weights.sum() - 1.0) < 1e-6
    assert (result.weights >= -1e-8).all()
    assert result.turnover is not None
    assert result.turnover < 0.60
    assert result.summary is not None and result.summary.is_optimal()

    assert any("Turnover" in note for note in result.notes)

    payload = result.to_dict(include_weights=True)
    assert payload["status"] in {"optimal", "OPTIMAL"}
    assert "weights" in payload


def test_run_optimizer_respects_portfolio_budgets(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    returns_path = _write_returns(tmp_path)
    config_path = _write_config_with_portfolio_budget(tmp_path, returns_path)

    result = run_optimizer(config_path, dry_run=False, settings=settings)

    assert result.summary is not None and result.summary.is_optimal()
    assert result.weights is not None
    high_return_group = result.weights.reindex(["AAA", "BBB"]).fillna(0.0).sum()
    assert high_return_group <= 0.3 + 1e-6


def test_run_optimizer_executes_mean_cvar(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    returns_path = _write_returns(tmp_path)
    config_path = _write_cvar_config(tmp_path, returns_path)

    result = run_optimizer(config_path, dry_run=False, settings=settings)

    assert result.summary is not None and result.summary.is_optimal()
    assert result.weights is not None
    assert abs(result.weights.sum() - 1.0) < 1e-6
    assert result.metrics is not None
    assert result.metrics.get("cvar") is not None
    assert result.metrics["cvar"] <= 0.06 + 1e-6
    assert result.metrics["expected_return"] >= 0.0005 - 1e-8
