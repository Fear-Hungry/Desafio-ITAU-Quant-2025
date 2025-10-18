import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from itau_quant.evaluation.plots import (
    TearsheetFigure,
    generate_tearsheet,
    plot_cumulative_returns,
    plot_drawdown,
    plot_risk_contribution,
    plot_rolling_sharpe,
    plot_rolling_volatility,
    plot_turnover,
)


def _returns() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    data = {
        "strategy": np.linspace(-0.01, 0.02, len(idx)),
        "alt": np.linspace(0.005, 0.015, len(idx)),
    }
    return pd.DataFrame(data, index=idx)


def test_plot_cumulative_returns_accepts_benchmark():
    returns = _returns()
    benchmark = returns[["strategy"]].rename(columns={"strategy": "benchmark"}) * 0.8
    ax = plot_cumulative_returns(returns, benchmark=benchmark)
    assert len(ax.lines) == returns.shape[1] + benchmark.shape[1]


def test_plot_drawdown_creates_collections():
    ax = plot_drawdown(_returns())
    assert len(ax.collections) >= 1


def test_plot_rolling_metrics_create_lines():
    returns = _returns()
    ax_sharpe = plot_rolling_sharpe(returns, window=3, periods_per_year=12)
    ax_vol = plot_rolling_volatility(returns, window=3)
    assert len(ax_sharpe.lines) == returns.shape[1] + 1
    assert len(ax_vol.lines) == returns.shape[1]


def test_plot_turnover_with_band():
    turnover = _returns()["strategy"].abs()
    ax = plot_turnover(turnover, target_band=(0.01, 0.03))
    assert len(ax.lines) == 1
    assert len(ax.patches) >= 1  # band


def test_plot_risk_contribution_bars():
    weights = pd.Series([0.6, 0.4], index=["A", "B"], name="latest")
    cov = pd.DataFrame([[0.05, 0.01], [0.01, 0.04]], index=["A", "B"], columns=["A", "B"])
    ax = plot_risk_contribution(weights, cov)
    assert len(ax.patches) == len(weights)


def test_generate_tearsheet_wraps_axes():
    returns = _returns()
    ax = plot_cumulative_returns(returns)
    figures = generate_tearsheet([("CumReturn", ax)])
    assert len(figures) == 1
    assert isinstance(figures[0], TearsheetFigure)
