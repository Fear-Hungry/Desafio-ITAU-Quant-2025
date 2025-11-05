import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from arara_quant.evaluation.plots import (
    plot_drawdown_contributors,
    plot_parameter_sensitivity,
    plot_signal_distribution,
    plot_turnover_vs_cost,
    plot_weight_stability,
)


def _weight_history() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=15, freq="B")
    data = {
        "A": np.linspace(0.4, 0.3, len(idx)),
        "B": np.linspace(0.6, 0.7, len(idx)),
    }
    return pd.DataFrame(data, index=idx)


def test_plot_weight_stability_returns_axis():
    ax = plot_weight_stability(_weight_history(), rolling_window=3)
    assert len(ax.lines) == 2


def test_plot_signal_distribution_hist():
    signals = pd.Series(np.random.normal(size=50), name="signal")
    ax = plot_signal_distribution(signals, bins=10)
    assert len(ax.patches) > 0


def test_plot_parameter_sensitivity_two_params():
    df = pd.DataFrame(
        {
            "lambda": [1, 2, 3, 4],
            "eta": [0.1, 0.2, 0.3, 0.4],
            "sharpe": [0.5, 0.6, 0.55, 0.7],
        }
    )
    ax = plot_parameter_sensitivity(
        df, param_columns=["lambda", "eta"], metric_column="sharpe"
    )
    assert len(ax.collections) == 1


def test_plot_turnover_vs_cost_scatter():
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    turnover = pd.Series(np.linspace(0.01, 0.05, len(idx)), index=idx, name="turnover")
    costs = pd.Series(np.linspace(0.001, 0.003, len(idx)), index=idx, name="turnover")
    ax = plot_turnover_vs_cost(turnover, costs)
    assert len(ax.collections) == 1


def test_plot_drawdown_contributors_stackplot():
    weights = _weight_history()
    drawdowns = pd.Series(
        np.linspace(0, -0.1, len(weights.index)), index=weights.index, name="dd"
    )
    ax = plot_drawdown_contributors(drawdowns, weights)
    assert len(ax.collections) >= 1  # stackplot returns PolyCollection(s)
