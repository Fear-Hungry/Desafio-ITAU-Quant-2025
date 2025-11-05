from __future__ import annotations

import math

import pandas as pd
from arara_quant.backtesting import metrics


def test_cumulative_nav_handles_empty_series() -> None:
    result = metrics.cumulative_nav(pd.Series(dtype=float))
    assert result.empty


def test_compute_performance_metrics_with_risk_free_float() -> None:
    returns = pd.Series(
        [0.01, -0.005, 0.02], index=pd.date_range("2020-01-01", periods=3, freq="D")
    )
    rf = 0.02

    result = metrics.compute_performance_metrics(
        returns, risk_free=rf, periods_in_year=252
    )

    nav = metrics.cumulative_nav(returns)
    expected_total_return = float(nav.iloc[-1] - 1.0)
    assert math.isclose(result.total_return, expected_total_return, rel_tol=1e-6)

    rf_daily = (1.0 + rf) ** (1 / 252) - 1
    excess = returns - rf_daily
    sharpe_expected = excess.mean() / returns.std(ddof=0) * math.sqrt(252)
    assert math.isclose(result.sharpe_ratio, sharpe_expected, rel_tol=1e-6)

    expected_drawdown = metrics.max_drawdown(nav)
    assert math.isclose(result.max_drawdown, expected_drawdown, rel_tol=1e-6)
