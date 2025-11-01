"""Walk-Forward Analysis Visualizations.

Functions to create specialized charts for walk-forward validation results,
including parameter evolution, per-window performance, and consistency analysis.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _prepare_axis(ax: Optional[plt.Axes] = None, *, title: Optional[str] = None) -> plt.Axes:
    """Prepare matplotlib axis for plotting."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    return ax


def plot_parameter_evolution(
    split_metrics: pd.DataFrame,
    *,
    parameters: Optional[list[str]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Parameter Evolution Across Windows",
) -> plt.Axes:
    """Plot evolution of key parameters (Sharpe, turnover, etc.) across walk-forward windows.

    Parameters
    ----------
    split_metrics : pd.DataFrame
        DataFrame with per-window results
    parameters : list of str, optional
        List of column names to plot. If None, defaults to
        ["sharpe_ratio", "turnover", "max_drawdown"]
    ax : plt.Axes, optional
        Matplotlib axes to plot on. If None, creates new figure.
    title : str
        Plot title

    Returns
    -------
    plt.Axes
        The axes used for plotting

    Examples
    --------
    >>> split_df = pd.DataFrame({
    ...     "sharpe_ratio": [1.2, 0.8, 1.5],
    ...     "turnover": [0.15, 0.20, 0.12],
    ...     "max_drawdown": [-0.03, -0.08, -0.05],
    ... })
    >>> ax = plot_parameter_evolution(split_df)
    >>> ax.get_title()
    'Parameter Evolution Across Windows'
    """
    if split_metrics.empty:
        raise ValueError("split_metrics is empty")

    if parameters is None:
        parameters = ["sharpe_ratio", "turnover", "max_drawdown"]

    # Filter to available columns
    parameters = [p for p in parameters if p in split_metrics.columns]
    if not parameters:
        raise ValueError("No valid parameters found in split_metrics")

    ax = _prepare_axis(ax, title=title)

    # Normalize each parameter to [0, 1] for comparable plotting
    n_windows = len(split_metrics)
    x = np.arange(n_windows)

    for param in parameters:
        values = split_metrics[param].values
        if np.all(np.isnan(values)):
            continue

        # Plot with offset for readability if multiple params
        ax.plot(x, values, marker="o", label=param.replace("_", " ").title(), linewidth=2)

    ax.set_xlabel("Window Index", fontsize=10)
    ax.set_ylabel("Parameter Value", fontsize=10)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add zero line if any parameter can be negative
    if any("drawdown" in p.lower() or "return" in p.lower() for p in parameters):
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    return ax


def plot_per_window_sharpe(
    split_metrics: pd.DataFrame,
    *,
    ax: Optional[plt.Axes] = None,
    title: str = "Sharpe Ratio by Window",
    highlight_negative: bool = True,
) -> plt.Axes:
    """Bar chart of Sharpe ratio for each walk-forward window.

    Parameters
    ----------
    split_metrics : pd.DataFrame
        DataFrame with per-window results, must contain "sharpe_ratio" column
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    title : str
        Plot title
    highlight_negative : bool, default=True
        Color negative Sharpe bars differently

    Returns
    -------
    plt.Axes
        The axes used for plotting
    """
    if split_metrics.empty or "sharpe_ratio" not in split_metrics.columns:
        raise ValueError("split_metrics must contain 'sharpe_ratio' column")

    ax = _prepare_axis(ax, title=title)

    sharpe_values = split_metrics["sharpe_ratio"].values
    n_windows = len(sharpe_values)
    x = np.arange(n_windows)

    if highlight_negative:
        colors = ["#d32f2f" if s < 0 else "#388e3c" for s in sharpe_values]
    else:
        colors = "#1976d2"

    ax.bar(x, sharpe_values, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linestyle="-", linewidth=1)

    ax.set_xlabel("Window Index", fontsize=10)
    ax.set_ylabel("Sharpe Ratio (OOS)", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Add mean line
    mean_sharpe = float(np.mean(sharpe_values))
    ax.axhline(mean_sharpe, color="blue", linestyle="--", linewidth=1.5, label=f"Mean: {mean_sharpe:.2f}")
    ax.legend(loc="best", fontsize=9)

    return ax


def plot_consistency_scatter(
    split_metrics: pd.DataFrame,
    *,
    ax: Optional[plt.Axes] = None,
    title: str = "Consistency: Period N vs N+1 Returns",
) -> plt.Axes:
    """Scatter plot of consecutive window returns to assess consistency.

    Parameters
    ----------
    split_metrics : pd.DataFrame
        DataFrame with per-window results, must contain "total_return" column
    ax : plt.Axes, optional
        Matplotlib axes to plot on
    title : str
        Plot title

    Returns
    -------
    plt.Axes
        The axes used for plotting

    Notes
    -----
    High R² indicates consistent performance across windows.
    Random scatter suggests unstable strategy.
    """
    if split_metrics.empty or "total_return" not in split_metrics.columns:
        raise ValueError("split_metrics must contain 'total_return' column")

    if len(split_metrics) < 2:
        raise ValueError("Need at least 2 windows for consistency analysis")

    ax = _prepare_axis(ax, title=title)

    returns = split_metrics["total_return"].values
    current = returns[1:]
    previous = returns[:-1]

    ax.scatter(previous, current, alpha=0.7, s=80, edgecolor="black", linewidth=0.5)

    # Compute and display R²
    if len(current) > 1 and np.std(current) > 0 and np.std(previous) > 0:
        correlation = np.corrcoef(current, previous)[0, 1]
        r_squared = correlation**2

        # Add regression line
        slope, intercept = np.polyfit(previous, current, 1)
        x_line = np.array([previous.min(), previous.max()])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color="red", linestyle="--", linewidth=2, label=f"R² = {r_squared:.3f}")
    else:
        r_squared = 0.0

    # Add diagonal (perfect consistency)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, "k-", alpha=0.3, linewidth=1, label="Perfect Consistency")

    ax.set_xlabel("Period N Return", fontsize=10)
    ax.set_ylabel("Period N+1 Return", fontsize=10)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)
    ax.axvline(0, color="gray", linestyle="-", linewidth=0.5)

    return ax


def plot_walkforward_summary(
    split_metrics: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (14, 8),
) -> plt.Figure:
    """Create comprehensive walk-forward summary with multiple subplots.

    Parameters
    ----------
    split_metrics : pd.DataFrame
        DataFrame with per-window results
    figsize : tuple of float
        Figure size (width, height)

    Returns
    -------
    plt.Figure
        Figure containing all walk-forward analysis charts
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Walk-Forward Performance Analysis", fontsize=14, fontweight="bold")

    # Top-left: Parameter evolution
    plot_parameter_evolution(split_metrics, ax=axes[0, 0])

    # Top-right: Per-window Sharpe
    plot_per_window_sharpe(split_metrics, ax=axes[0, 1])

    # Bottom-left: Consistency scatter
    if len(split_metrics) >= 2:
        plot_consistency_scatter(split_metrics, ax=axes[1, 0])
    else:
        axes[1, 0].text(
            0.5, 0.5, "Need ≥2 windows for consistency", ha="center", va="center", fontsize=10
        )
        axes[1, 0].axis("off")

    # Bottom-right: Turnover + Cost evolution
    if "turnover" in split_metrics.columns and "cost_fraction" in split_metrics.columns:
        ax_bottom_right = axes[1, 1]
        x = np.arange(len(split_metrics))
        ax_bottom_right.plot(x, split_metrics["turnover"], marker="o", label="Turnover", linewidth=2)
        ax_secondary = ax_bottom_right.twinx()
        ax_secondary.plot(
            x, split_metrics["cost_fraction"] * 10000, marker="s", color="orange", label="Cost (bps)", linewidth=2
        )
        ax_bottom_right.set_xlabel("Window Index", fontsize=10)
        ax_bottom_right.set_ylabel("Turnover", fontsize=10, color="blue")
        ax_secondary.set_ylabel("Cost (bps)", fontsize=10, color="orange")
        ax_bottom_right.set_title("Turnover & Cost Evolution", fontsize=10, fontweight="bold")
        ax_bottom_right.legend(loc="upper left", fontsize=9)
        ax_secondary.legend(loc="upper right", fontsize=9)
        ax_bottom_right.grid(True, alpha=0.3)
    else:
        axes[1, 1].text(
            0.5, 0.5, "Turnover/Cost data unavailable", ha="center", va="center", fontsize=10
        )
        axes[1, 1].axis("off")

    plt.tight_layout()
    return fig
