import numpy as np
import pandas as pd
import pytest
from itau_quant.evaluation.stats import (
    aggregate_performance,
    annualized_return,
    excess_vs_benchmark,
    sharpe_ratio,
)


def _toy_returns() -> pd.Series:
    data = pd.Series([0.01, -0.005, 0.015, 0.0], name="portfolio")
    data.index = pd.date_range("2024-01-01", periods=len(data), freq="B")
    return data


def test_annualized_return_matches_manual():
    returns = _toy_returns()
    periods = 12
    expected = np.prod(1.0 + returns) ** (periods / len(returns)) - 1.0
    result = annualized_return(returns, periods_per_year=periods)
    assert result.loc["portfolio"] == pytest.approx(expected)


def test_sharpe_ratio_matches_manual_standardisation():
    returns = _toy_returns()
    periods = 252
    mean = returns.mean()
    vol = returns.std(ddof=1)
    expected = mean / vol * np.sqrt(periods)
    computed = sharpe_ratio(returns, periods_per_year=periods)
    assert computed.loc["portfolio"] == pytest.approx(expected)


def test_excess_vs_benchmark_active_return_and_tracking_error():
    strategy = pd.DataFrame(
        {"portfolio": [0.02, 0.01, 0.0, 0.03]},
        index=pd.date_range("2024-01-01", periods=4, freq="B"),
    )
    benchmark = pd.DataFrame(
        {"portfolio": [0.01, 0.0, -0.01, 0.02]},
        index=strategy.index,
    )
    diff = strategy["portfolio"] - benchmark["portfolio"]
    periods = 12
    expected_active = diff.mean() * periods
    expected_te = diff.std(ddof=1) * np.sqrt(periods)

    metrics = excess_vs_benchmark(strategy, benchmark, periods_per_year=periods)

    assert metrics.active_return.loc["portfolio"] == pytest.approx(expected_active)
    assert metrics.tracking_error.loc["portfolio"] == pytest.approx(expected_te)


def test_aggregate_performance_contains_expected_metrics():
    returns = _toy_returns().to_frame()
    table = aggregate_performance(returns, periods_per_year=252)

    assert ("performance", "annualized_return") in table.index
    assert ("risk", "max_drawdown") in table.index

    hit = (returns["portfolio"] > 0).mean()
    assert table.loc[("performance", "hit_rate"), "portfolio"] == pytest.approx(hit)
