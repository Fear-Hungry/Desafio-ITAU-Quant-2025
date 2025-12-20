import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from arara_quant.evaluation.plots import (
    plot_daily_returns_dashboard,
    plot_nav_cumulative,
    plot_risk_return_scatter,
    plot_underwater_drawdown,
)


def _nav_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    nav = 1.0 + np.linspace(0.0, 0.05, len(dates))
    nav[10:15] -= 0.03  # inject a drawdown
    return pd.DataFrame({"date": dates, "nav": nav})


def test_plot_nav_cumulative_returns_axis():
    ax = plot_nav_cumulative(_nav_frame())
    assert len(ax.lines) >= 1


def test_plot_underwater_drawdown_returns_axis_with_bars():
    ax = plot_underwater_drawdown(_nav_frame())
    assert len(ax.patches) > 0  # bars


def test_plot_daily_returns_dashboard_has_4_subplots():
    fig = plot_daily_returns_dashboard(_nav_frame())
    assert len(fig.axes) == 4


def test_plot_risk_return_scatter_returns_axis_with_points():
    baselines = pd.DataFrame(
        {
            "strategy": ["equal_weight", "risk_parity"],
            "annualized_return": [0.04, 0.03],
            "sharpe": [0.2, 0.15],
        }
    )
    ax = plot_risk_return_scatter(
        prism_return_pct=5.0, prism_sharpe=0.1, baselines=baselines
    )
    assert len(ax.collections) >= 1

