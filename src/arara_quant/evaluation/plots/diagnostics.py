"""Diagnostic visualisations used to complement the tearsheet."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd


def _prepare_axis(ax: plt.Axes | None = None, *, title: str | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    if title:
        ax.set_title(title)
    return ax


def _to_frame(data: pd.DataFrame | pd.Series, *, name: str) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, pd.Series):
        df = data.to_frame(name=data.name or name)
    else:
        raise TypeError(f"{name} must be a pandas Series or DataFrame")
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if df.empty:
        raise ValueError(f"{name} is empty after dropping NaNs")
    return df.astype(float)


def plot_weight_stability(
    weights: pd.DataFrame | pd.Series,
    *,
    rolling_window: int = 20,
    ax: plt.Axes | None = None,
    title: str = "Weight Stability",
) -> plt.Axes:
    """Plot rolling standard deviation of weights to highlight stability."""

    if rolling_window <= 0:
        raise ValueError("rolling_window must be positive")

    frame = _to_frame(weights, name="weights")
    ax = _prepare_axis(ax, title=title)

    for column in frame.columns:
        rolling_std = frame[column].rolling(window=rolling_window).std(ddof=0)
        ax.plot(frame.index, rolling_std, label=column)

    ax.set_ylabel("Rolling std")
    ax.set_xlabel("Date")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return ax


def plot_signal_distribution(
    signals: pd.DataFrame | pd.Series,
    *,
    bins: int = 30,
    ax: plt.Axes | None = None,
    title: str = "Signal Distribution",
) -> plt.Axes:
    """Plot histogram(s) of signal values."""

    if bins <= 0:
        raise ValueError("bins must be positive")

    frame = _to_frame(signals, name="signals")
    ax = _prepare_axis(ax, title=title)

    for column in frame.columns:
        ax.hist(frame[column].dropna(), bins=bins, alpha=0.5, label=column)

    ax.set_xlabel("Signal value")
    ax.set_ylabel("Frequency")
    ax.legend(loc="best")
    return ax


def plot_parameter_sensitivity(
    results: pd.DataFrame,
    *,
    param_columns: Sequence[str],
    metric_column: str,
    ax: plt.Axes | None = None,
    title: str = "Parameter Sensitivity",
) -> plt.Axes:
    """Scatter plot showing how a metric changes with one or two parameters."""

    if metric_column not in results.columns:
        raise ValueError(f"metric_column '{metric_column}' not present in results")
    if not param_columns:
        raise ValueError("param_columns cannot be empty")

    df = results.dropna(subset=[metric_column, *param_columns])
    if df.empty:
        raise ValueError("results contain no rows after dropping NaNs")

    ax = _prepare_axis(ax, title=title)
    metric = df[metric_column]

    if len(param_columns) == 1:
        param = df[param_columns[0]]
        scatter = ax.scatter(
            param, metric, c=metric, cmap="viridis", edgecolor="k", alpha=0.8
        )
        ax.set_xlabel(param_columns[0])
        ax.set_ylabel(metric_column)
        plt.colorbar(scatter, ax=ax, label=metric_column)
    else:
        x = df[param_columns[0]]
        y = df[param_columns[1]]
        scatter = ax.scatter(x, y, c=metric, cmap="viridis", edgecolor="k", alpha=0.8)
        ax.set_xlabel(param_columns[0])
        ax.set_ylabel(param_columns[1])
        plt.colorbar(scatter, ax=ax, label=metric_column)

    ax.grid(True, alpha=0.3)
    return ax


def plot_turnover_vs_cost(
    turnover: pd.Series | pd.DataFrame,
    costs: pd.Series | pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
    title: str = "Turnover vs Costs",
) -> plt.Axes:
    """Scatter plot comparing realised turnover and costs."""

    turnover_frame = _to_frame(turnover, name="turnover")
    costs_frame = _to_frame(costs, name="costs")

    combined = turnover_frame.join(
        costs_frame, how="inner", lsuffix="_turnover", rsuffix="_cost"
    )
    if combined.empty:
        raise ValueError("turnover and costs do not share common index")

    ax = _prepare_axis(ax, title=title)

    for column in turnover_frame.columns:
        cost_col = column if column in costs_frame.columns else costs_frame.columns[0]
        ax.scatter(
            combined[f"{column}_turnover"], combined[f"{cost_col}_cost"], label=column
        )

    ax.set_xlabel("Turnover")
    ax.set_ylabel("Cost")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return ax


def plot_drawdown_contributors(
    drawdowns: pd.Series | pd.DataFrame,
    weight_history: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
    title: str = "Drawdown Contributors",
) -> plt.Axes:
    """Plot approximate contribution of each asset during drawdowns."""

    dd_frame = _to_frame(drawdowns, name="drawdowns")
    weights_frame = _to_frame(weight_history, name="weights")

    aligned_dd = dd_frame.reindex(weights_frame.index).ffill()
    contributions = weights_frame.mul(aligned_dd.iloc[:, 0], axis=0)

    ax = _prepare_axis(ax, title=title)
    ax.stackplot(
        contributions.index, contributions.T, labels=contributions.columns, alpha=0.7
    )

    ax.set_ylabel("Contribution")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    return ax


__all__ = [
    "plot_weight_stability",
    "plot_signal_distribution",
    "plot_parameter_sensitivity",
    "plot_turnover_vs_cost",
    "plot_drawdown_contributors",
]
