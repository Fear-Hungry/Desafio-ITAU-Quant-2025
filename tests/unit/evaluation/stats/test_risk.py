import numpy as np
import pandas as pd
import pytest
from itau_quant.evaluation.stats import (
    RiskSummary,
    aggregate_risk_metrics,
    conditional_value_at_risk,
    max_drawdown,
    risk_contribution,
)


def test_max_drawdown_zero_for_monotonic_growth():
    returns = pd.DataFrame(
        {"portfolio": [0.01, 0.015, 0.02]},
        index=pd.date_range("2023-01-02", periods=3, freq="B"),
    )
    dd = max_drawdown(returns)
    assert dd.loc["portfolio"] == pytest.approx(0.0)


def test_conditional_value_at_risk_matches_manual_tail_mean():
    returns = pd.Series([-0.04, -0.03, 0.01, 0.02], name="portfolio")
    alpha = 0.75
    tail_prob = 1.0 - alpha
    quantile = returns.quantile(tail_prob, interpolation="lower")
    tail = returns[returns <= quantile]
    expected = tail.mean()

    result = conditional_value_at_risk(returns, alpha=alpha)
    assert result.loc["portfolio"] == pytest.approx(expected)


def test_risk_contribution_percentages_sum_to_one():
    weights = pd.Series([0.6, 0.4], index=["A", "B"], name="latest")
    cov = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    result = risk_contribution(weights, cov)
    percentages = result.percentage.loc[result.percentage.index[0]]
    assert percentages.sum() == pytest.approx(1.0)

    portfolio_variance = float(result.component.sum(axis=1).iloc[0])
    direct_var = float(weights @ cov @ weights)
    assert portfolio_variance == pytest.approx(direct_var)


def test_aggregate_risk_metrics_produces_summary_with_contributions():
    idx = pd.date_range("2024-01-01", periods=6, freq="B")
    returns = pd.DataFrame({"portfolio": np.linspace(-0.01, 0.02, 6)}, index=idx)
    benchmark = pd.DataFrame({"portfolio": np.linspace(-0.005, 0.015, 6)}, index=idx)
    weights = pd.Series([0.55, 0.45], index=["A", "B"], name=idx[-1])
    cov = pd.DataFrame(
        [[0.05, 0.015], [0.015, 0.04]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    summary = aggregate_risk_metrics(
        returns,
        benchmark=benchmark,
        weights=weights,
        covariance=cov,
        periods_per_year=252,
    )

    assert isinstance(summary, RiskSummary)
    assert ("relative", "tracking_error") in summary.metrics.index
    assert summary.metrics.loc[("positioning", "realized_leverage"), "portfolio"] > 0
    assert summary.risk_contribution is not None
    assert not summary.drawdowns.empty
