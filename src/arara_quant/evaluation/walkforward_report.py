"""Walk-Forward Performance Analysis and Reporting.

This module provides functions to compute comprehensive walk-forward statistics,
including summary metrics, per-window analysis, parameter stability, and stress period detection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardSummary:
    """Aggregated statistics from walk-forward validation."""

    n_windows: int
    success_rate: float  # Fraction of windows with positive returns
    avg_sharpe: float
    avg_return: float
    avg_volatility: float
    avg_drawdown: float
    avg_turnover: float
    avg_cost: float
    consistency_r2: float  # R² between consecutive window returns
    best_window_nav: float
    worst_window_nav: float
    range_ratio: float  # best_nav / worst_nav


@dataclass(frozen=True)
class StressPeriod:
    """Identified stress period in walk-forward results."""

    window_index: int
    test_start: str
    test_end: str
    sharpe: float
    max_drawdown: float
    return_: float
    label: str  # e.g., "Pandemic 2020", "Inflation 2022"


def compute_wf_summary_stats(
    split_metrics: pd.DataFrame,
    *,
    drawdown_threshold: float = -0.10,
) -> WalkForwardSummary:
    """Compute aggregated walk-forward summary statistics.

    Parameters
    ----------
    split_metrics : pd.DataFrame
        DataFrame with per-window results, must contain columns:
        - total_return, annualized_return, annualized_volatility
        - sharpe_ratio, max_drawdown, cumulative_nav
        - turnover, cost_fraction
    drawdown_threshold : float, default=-0.10
        Threshold for defining stress periods (e.g., -10%)

    Returns
    -------
    WalkForwardSummary
        Aggregated statistics across all windows

    Examples
    --------
    >>> split_df = pd.DataFrame({
    ...     "total_return": [0.05, -0.02, 0.08],
    ...     "sharpe_ratio": [1.2, -0.5, 1.8],
    ...     "max_drawdown": [-0.03, -0.12, -0.05],
    ...     "cumulative_nav": [1.05, 1.03, 1.11],
    ...     "annualized_return": [0.25, -0.10, 0.40],
    ...     "annualized_volatility": [0.12, 0.10, 0.15],
    ...     "turnover": [0.15, 0.20, 0.12],
    ...     "cost_fraction": [0.0015, 0.0020, 0.0012],
    ... })
    >>> summary = compute_wf_summary_stats(split_df)
    >>> summary.n_windows
    3
    >>> summary.success_rate
    0.666...
    """
    if split_metrics.empty:
        raise ValueError("split_metrics is empty")

    required_cols = {
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "cumulative_nav",
        "turnover",
        "cost_fraction",
    }
    missing = required_cols - set(split_metrics.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    n_windows = len(split_metrics)
    positive_windows = (split_metrics["total_return"] > 0).sum()
    success_rate = float(positive_windows) / n_windows if n_windows > 0 else 0.0

    avg_sharpe = float(split_metrics["sharpe_ratio"].mean())
    avg_return = float(split_metrics["annualized_return"].mean())
    avg_volatility = float(split_metrics["annualized_volatility"].mean())
    avg_drawdown = float(split_metrics["max_drawdown"].mean())
    avg_turnover = float(split_metrics["turnover"].mean())
    avg_cost = float(split_metrics["cost_fraction"].mean())

    # Compute consistency (R² between consecutive window returns)
    if n_windows > 1:
        returns = split_metrics["total_return"].values
        current = returns[1:]
        previous = returns[:-1]
        if len(current) > 1 and np.std(current) > 0 and np.std(previous) > 0:
            correlation = np.corrcoef(current, previous)[0, 1]
            consistency_r2 = float(correlation**2)
        else:
            consistency_r2 = 0.0
    else:
        consistency_r2 = 0.0

    best_nav = float(split_metrics["cumulative_nav"].max())
    worst_nav = float(split_metrics["cumulative_nav"].min())
    range_ratio = best_nav / worst_nav if worst_nav > 0 else np.inf

    return WalkForwardSummary(
        n_windows=n_windows,
        success_rate=success_rate,
        avg_sharpe=avg_sharpe,
        avg_return=avg_return,
        avg_volatility=avg_volatility,
        avg_drawdown=avg_drawdown,
        avg_turnover=avg_turnover,
        avg_cost=avg_cost,
        consistency_r2=consistency_r2,
        best_window_nav=best_nav,
        worst_window_nav=worst_nav,
        range_ratio=range_ratio,
    )


def _generate_simple_latex(df: pd.DataFrame) -> str:
    """Fallback LaTeX output that avoids optional pandas dependencies."""

    if df.empty:
        return ""

    headers = df.columns.tolist()
    column_spec = "l" * len(headers) or "l"
    lines = [f"\\begin{{tabular}}{{{column_spec}}}", "\\hline"]
    lines.append(" & ".join(headers) + r" \\")
    lines.append("\\hline")

    for _, row in df.iterrows():
        formatted = []
        for value in row:
            if isinstance(value, (int, float, np.number)):
                formatted.append(f"{float(value):.4f}")
            else:
                formatted.append(str(value))
        lines.append(" & ".join(formatted) + r" \\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def build_per_window_table(
    split_metrics: pd.DataFrame,
    *,
    format: str = "markdown",
) -> str:
    """Build formatted table of per-window results.

    Parameters
    ----------
    split_metrics : pd.DataFrame
        DataFrame with per-window results
    format : str, default="markdown"
        Output format: "markdown", "latex", or "csv"

    Returns
    -------
    str
        Formatted table string
    """
    if split_metrics.empty:
        return ""

    # Select key columns for display
    display_cols = [
        "test_end",
        "sharpe_ratio",
        "annualized_return",
        "max_drawdown",
        "turnover",
        "cost_fraction",
    ]
    existing_cols = [c for c in display_cols if c in split_metrics.columns]
    table_df = split_metrics[existing_cols].copy()

    # Rename for readability
    rename_map = {
        "test_end": "Window End",
        "sharpe_ratio": "Sharpe (OOS)",
        "annualized_return": "Return (OOS)",
        "max_drawdown": "Drawdown (OOS)",
        "turnover": "Turnover",
        "cost_fraction": "Cost",
    }
    table_df = table_df.rename(columns=rename_map)

    if format == "markdown":
        # Simple markdown table generator (no tabulate dependency needed)
        lines = []
        headers = table_df.columns.tolist()
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for _, row in table_df.iterrows():
            values = [
                f"{v:.4f}" if isinstance(v, (int, float)) else str(v) for v in row
            ]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)
    elif format == "latex":
        try:
            return table_df.to_latex(index=False, float_format="%.4f")
        except Exception:
            return _generate_simple_latex(table_df)
    elif format == "csv":
        return table_df.to_csv(index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")


def identify_stress_periods(
    split_metrics: pd.DataFrame,
    *,
    drawdown_threshold: float = -0.15,
    sharpe_threshold: float = -0.5,
) -> list[StressPeriod]:
    """Identify windows with significant stress (high drawdown or negative Sharpe).

    Parameters
    ----------
    split_metrics : pd.DataFrame
        DataFrame with per-window results
    drawdown_threshold : float, default=-0.15
        Windows with drawdown worse than this are flagged
    sharpe_threshold : float, default=-0.5
        Windows with Sharpe below this are flagged

    Returns
    -------
    List[StressPeriod]
        List of identified stress periods

    Examples
    --------
    >>> split_df = pd.DataFrame({
    ...     "test_start": ["2020-02-01", "2022-01-01"],
    ...     "test_end": ["2020-03-01", "2022-02-01"],
    ...     "sharpe_ratio": [-2.5, -0.8],
    ...     "max_drawdown": [-0.25, -0.18],
    ...     "annualized_return": [-0.50, -0.15],
    ... })
    >>> periods = identify_stress_periods(split_df, drawdown_threshold=-0.15)
    >>> len(periods)
    2
    """
    if split_metrics.empty:
        return []

    stress_windows = split_metrics[
        (split_metrics["max_drawdown"] < drawdown_threshold)
        | (split_metrics["sharpe_ratio"] < sharpe_threshold)
    ]

    stress_periods = []
    for idx, row in stress_windows.iterrows():
        # Auto-label based on date
        test_start = row.get("test_start", "")
        label = _auto_label_period(test_start)

        period = StressPeriod(
            window_index=int(idx) if isinstance(idx, (int, np.integer)) else 0,
            test_start=str(row.get("test_start", "")),
            test_end=str(row.get("test_end", "")),
            sharpe=float(row["sharpe_ratio"]),
            max_drawdown=float(row["max_drawdown"]),
            return_=float(row.get("annualized_return", 0.0)),
            label=label,
        )
        stress_periods.append(period)

    return stress_periods


def _auto_label_period(date_str: str) -> str:
    """Auto-label stress period based on date."""
    if not date_str:
        return "Unknown"

    year = date_str[:4]
    if "2020" in year:
        return "Pandemic 2020"
    elif "2022" in year:
        return "Inflation 2022"
    elif "2023" in year and "03" in date_str:
        return "Banking Crisis 2023"
    elif "2008" in year or "2009" in year:
        return "Financial Crisis 2008-09"
    else:
        return f"Stress {year}"


def compute_range_ratio(split_metrics: pd.DataFrame) -> dict[str, float]:
    """Compute performance range statistics.

    Parameters
    ----------
    split_metrics : pd.DataFrame
        DataFrame with per-window results containing 'cumulative_nav'

    Returns
    -------
    Dict[str, float]
        Dictionary with keys: best_nav, worst_nav, range_ratio, nav_std
    """
    if split_metrics.empty or "cumulative_nav" not in split_metrics.columns:
        return {
            "best_nav": np.nan,
            "worst_nav": np.nan,
            "range_ratio": np.nan,
            "nav_std": np.nan,
        }

    navs = split_metrics["cumulative_nav"].values
    best = float(np.max(navs))
    worst = float(np.min(navs))
    ratio = best / worst if worst > 0 else np.inf
    std = float(np.std(navs, ddof=1))

    return {
        "best_nav": best,
        "worst_nav": worst,
        "range_ratio": ratio,
        "nav_std": std,
    }


def format_wf_summary_markdown(summary: WalkForwardSummary) -> str:
    """Format WalkForwardSummary as Markdown table.

    Parameters
    ----------
    summary : WalkForwardSummary
        Summary statistics to format

    Returns
    -------
    str
        Markdown-formatted summary table
    """
    lines = [
        "## Walk-Forward Summary Statistics",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| **Number of OOS Windows** | {summary.n_windows} |",
        f"| **Success Rate** | {summary.success_rate:.1%} |",
        f"| **Average Sharpe (OOS)** | {summary.avg_sharpe:.2f} |",
        f"| **Average Return (OOS)** | {summary.avg_return:.2%} |",
        f"| **Average Volatility (OOS)** | {summary.avg_volatility:.2%} |",
        f"| **Average Max Drawdown** | {summary.avg_drawdown:.2%} |",
        f"| **Average Turnover** | {summary.avg_turnover:.2%} |",
        f"| **Average Cost** | {summary.avg_cost * 10000:.1f} bps |",
        f"| **Consistency (R²)** | {summary.consistency_r2:.3f} |",
        f"| **Best Window NAV** | {summary.best_window_nav:.4f} |",
        f"| **Worst Window NAV** | {summary.worst_window_nav:.4f} |",
        f"| **Range Ratio** | {summary.range_ratio:.2f} |",
        "",
    ]
    return "\n".join(lines)
