from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from arara_quant.backtesting import run_backtest
from arara_quant.config import Settings, reset_settings_cache
from arara_quant.optimization.core.mv_qp import MeanVarianceConfig, solve_mean_variance


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


@pytest.fixture(scope="module")
def synthetic_backtest_config(tmp_path_factory: pytest.TempPathFactory):
    """Provision a tiny deterministic backtest configuration for benchmarks."""

    tmp_path = tmp_path_factory.mktemp("benchmark")
    settings = _make_settings(tmp_path)

    dates = pd.bdate_range("2022-01-03", periods=180)
    rng = np.random.default_rng(123)
    returns = pd.DataFrame(
        rng.normal(0.0005, 0.01, size=(len(dates), 3)),
        index=dates,
        columns=["AAA", "BBB", "CCC"],
    )
    returns_path = settings.processed_data_dir / "tiny_returns.parquet"
    returns.to_parquet(returns_path)

    config_text = f"""
base_currency: USD
data:
  returns: {returns_path}
walkforward:
  train_days: 84
  test_days: 21
  purge_days: 2
  embargo_days: 2
  evaluation_horizons: [21]
portfolio:
  capital: 500_000
optimizer:
  lambda: 5.0
  eta: 0.0
  tau: 0.25
  min_weight: 0.0
  max_weight: 0.7
  solver: OSQP
estimators:
  mu: {{ method: shrunk_50, window_days: 84, strength: 0.5 }}
  sigma: {{ method: ledoit_wolf, window_days: 84 }}
""".strip()

    config_path = settings.configs_dir / "benchmark_backtest.yaml"
    config_path.write_text(config_text, encoding="utf-8")

    try:
        yield config_path, settings
    finally:
        reset_settings_cache()


def test_mean_variance_solver_microbenchmark() -> None:
    """Mean-variance solver should remain fast on a small synthetic problem."""

    rng = np.random.default_rng(0)
    assets = [f"A{i}" for i in range(20)]
    mu = pd.Series(rng.normal(0.0004, 0.01, size=len(assets)), index=assets)
    random_matrix = rng.normal(size=(len(assets), len(assets)))
    cov = pd.DataFrame(
        random_matrix @ random_matrix.T + np.eye(len(assets)) * 1e-4,
        index=assets,
        columns=assets,
    )

    lower = pd.Series(0.0, index=assets)
    upper = pd.Series(0.15, index=assets)
    previous = pd.Series(1.0 / len(assets), index=assets)

    config = MeanVarianceConfig(
        risk_aversion=5.0,
        turnover_penalty=0.0,
        turnover_cap=None,
        lower_bounds=lower,
        upper_bounds=upper,
        previous_weights=previous,
        cost_vector=None,
        budgets=None,
        solver="OSQP",
        solver_kwargs={"eps_abs": 1e-6, "eps_rel": 1e-6},
    )

    start = time.perf_counter()
    result = solve_mean_variance(mu, cov, config)
    elapsed = time.perf_counter() - start

    assert result.summary.is_optimal(), f"Solver status: {result.summary.status}"
    assert result.weights.sum() == pytest.approx(1.0)
    # Generous upper bound to catch regressions without being flaky.
    assert elapsed < 1.5, f"Mean-variance solver took {elapsed:.3f}s"


def test_backtest_pipeline_microbenchmark(synthetic_backtest_config) -> None:
    """Tiny walk-forward backtest should complete within a reasonable bound."""

    config_path, settings = synthetic_backtest_config

    start = time.perf_counter()
    result = run_backtest(config_path, dry_run=False, settings=settings)
    elapsed = time.perf_counter() - start

    assert result.metrics is not None
    assert result.split_metrics is not None and not result.split_metrics.empty
    assert elapsed < 3.0, f"Backtest pipeline took {elapsed:.3f}s"
