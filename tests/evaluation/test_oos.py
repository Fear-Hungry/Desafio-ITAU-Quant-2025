from __future__ import annotations

import numpy as np
import pandas as pd
from arara_quant.evaluation.oos import (
    OOSResult,
    StrategySpec,
    compare_baselines,
    default_strategies,
    stress_test,
)


def _synthetic_returns(rows: int = 160, cols: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=rows, freq="B")
    data = rng.normal(0.0005, 0.01, size=(rows, cols))
    columns = [f"A{i}" for i in range(cols)]
    return pd.DataFrame(data, index=dates, columns=columns)


def test_compare_baselines_with_custom_strategies() -> None:
    returns = _synthetic_returns(rows=180, cols=3)

    def equal_weight(train: pd.DataFrame, _: pd.Series | None) -> pd.Series:
        return pd.Series(1.0 / train.shape[1], index=train.columns)

    def momentum_tilt(train: pd.DataFrame, _: pd.Series | None) -> pd.Series:
        mu = train.mean().clip(lower=0.0)
        if mu.sum() == 0:
            return equal_weight(train, None)
        weights = mu / mu.sum()
        return weights

    strategies = [
        StrategySpec("equal", equal_weight),
        StrategySpec("momentum", momentum_tilt),
    ]

    result: OOSResult = compare_baselines(
        returns,
        strategies=strategies,
        train_window=60,
        test_window=15,
        purge_window=0,
        embargo_window=0,
        costs_bps=0.0,
        max_position=1.0,
    )

    assert set(result.metrics.index) == {"equal", "momentum"}
    assert not result.returns.empty
    assert all(col in result.returns.columns for col in ["equal", "momentum"])
    expected_cols = {
        "total_return",
        "annualized_return",
        "volatility",
        "sharpe",
        "cvar_95",
        "max_drawdown",
        "avg_turnover",
        "total_cost",
        "sharpe_ci_low",
        "sharpe_ci_high",
    }
    assert set(result.metrics.columns) == expected_cols


def test_default_strategies_contains_shrunk_mv() -> None:
    names = {spec.name for spec in default_strategies()}
    assert "shrunk_mv" in names


def test_stress_test_returns_period_metrics() -> None:
    dates = pd.date_range("2020-01-01", periods=6, freq="B")
    data = pd.DataFrame(
        {
            "robust": [0.01, -0.02, 0.015, 0.0, 0.01, -0.005],
            "baseline": [0.012, -0.03, 0.02, -0.01, 0.005, -0.01],
        },
        index=dates,
    )
    periods = {"covid": (dates[0], dates[-1])}
    stress_df = stress_test(data, periods)

    assert not stress_df.empty
    assert set(stress_df["strategy"]) == {"robust", "baseline"}
    assert set(stress_df["period"]) == {"covid"}
    assert {"total_return", "max_drawdown"}.issubset(stress_df.columns)


def test_compare_baselines_bootstrap_ci() -> None:
    returns = _synthetic_returns(rows=220, cols=3)
    result = compare_baselines(
        returns,
        strategies=default_strategies(max_position=1.0)[:1],
        train_window=100,
        test_window=20,
        purge_window=0,
        embargo_window=0,
        bootstrap_iterations=200,
        confidence=0.90,
        random_state=123,
    )
    sharpe_ci_low = result.metrics.iloc[0]["sharpe_ci_low"]
    sharpe_ci_high = result.metrics.iloc[0]["sharpe_ci_high"]
    assert sharpe_ci_low <= sharpe_ci_high
    assert not np.isnan(sharpe_ci_low)
