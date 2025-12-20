"""Out-of-sample (OOS) figures used in reports.

The helpers in this module are intentionally data-driven: they accept pre-loaded
DataFrames (typically from ``arara_quant.reports.canonical``) and return the
Matplotlib ``Axes``/``Figure`` objects so downstream code can compose and save
figures consistently.
"""

from __future__ import annotations

from collections.abc import Iterable

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = [
    "BaselinePoint",
    "build_default_baseline_points",
    "plot_daily_returns_dashboard",
    "plot_nav_cumulative",
    "plot_risk_return_scatter",
    "plot_underwater_drawdown",
]


@dataclass(frozen=True, slots=True)
class BaselinePoint:
    """Single baseline point in risk/return space."""

    name: str
    annual_return_pct: float
    sharpe_ratio: float


def _palette() -> Iterable[str | None]:
    cycle = plt.rcParams.get("axes.prop_cycle")
    if cycle is None:
        return iter([None])
    colors = cycle.by_key().get("color", [None])
    return iter(colors or [None])


def _to_nav_frame(
    nav: pd.DataFrame,
    *,
    date_column: str = "date",
    nav_column: str = "nav",
) -> pd.DataFrame:
    if not isinstance(nav, pd.DataFrame):
        raise TypeError("nav must be a pandas DataFrame")
    missing = {date_column, nav_column}.difference(nav.columns)
    if missing:
        raise ValueError(f"nav missing required columns: {sorted(missing)}")

    frame = nav.copy()
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    frame[nav_column] = pd.to_numeric(frame[nav_column], errors="coerce")
    frame = frame.dropna(subset=[date_column, nav_column])
    frame = frame.sort_values(date_column).reset_index(drop=True)
    if frame.empty:
        raise ValueError("nav is empty after cleaning")
    return frame


def _daily_returns_from_nav(
    nav: pd.DataFrame,
    *,
    nav_column: str = "nav",
    daily_return_column: str = "daily_return",
) -> pd.Series:
    if daily_return_column in nav.columns:
        s = pd.to_numeric(nav[daily_return_column], errors="coerce")
        if s.notna().any():
            return s.fillna(0.0).astype(float)

    nav_values = pd.to_numeric(nav[nav_column], errors="coerce").astype(float)
    returns = nav_values.pct_change().fillna(0.0)
    return returns


def _drawdown_from_nav(nav_values: np.ndarray) -> np.ndarray:
    running_max = np.maximum.accumulate(nav_values)
    drawdown = nav_values / running_max - 1.0
    return drawdown


def plot_nav_cumulative(
    nav: pd.DataFrame,
    *,
    date_column: str = "date",
    nav_column: str = "nav",
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Plot daily NAV over time with max-drawdown annotation."""

    frame = _to_nav_frame(nav, date_column=date_column, nav_column=nav_column)

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 7))

    dates = frame[date_column].to_numpy()
    nav_values = frame[nav_column].to_numpy(dtype=float)

    palette = _palette()
    nav_color = next(palette)
    accent_color = next(palette)
    dd_color = next(palette)

    ax.plot(dates, nav_values, linewidth=2.5, label="NAV (OOS Daily)", color=nav_color)
    ax.fill_between(dates, float(nav_values[0]), nav_values, alpha=0.25, color=nav_color)

    nav_final = float(nav_values[-1])
    ax.axhline(
        y=nav_final,
        color=accent_color,
        linestyle="--",
        linewidth=1.5,
        alpha=0.6,
        label=f"Final NAV: {nav_final:.4f}",
    )

    drawdown = _drawdown_from_nav(nav_values)
    min_dd_idx = int(np.argmin(drawdown))
    max_dd = float(drawdown[min_dd_idx])
    ax.axvline(x=dates[min_dd_idx], color=dd_color, linestyle=":", alpha=0.6, linewidth=2)
    ax.scatter(
        [dates[min_dd_idx]],
        [nav_values[min_dd_idx]],
        color=dd_color,
        s=90,
        zorder=5,
        marker="o",
        edgecolors=accent_color,
        linewidth=1.5,
    )
    ax.annotate(
        f"Max DD\n{max_dd:.2%}",
        xy=(dates[min_dd_idx], nav_values[min_dd_idx]),
        xytext=(0, -30),
        textcoords="offset points",
        ha="center",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.6},
        arrowprops={"arrowstyle": "->", "color": dd_color, "alpha": 0.8},
    )

    ax.set_xlabel("Date", fontsize=12, fontweight="bold")
    ax.set_ylabel("NAV", fontsize=12, fontweight="bold")
    if title is None:
        start = pd.to_datetime(dates[0]).date()
        end = pd.to_datetime(dates[-1]).date()
        title = f"Out-of-Sample Daily NAV ({start} to {end})"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc="upper left")
    ax.set_ylim([min(0.7, float(nav_values.min()) * 0.98), max(1.1, nav_final * 1.05)])

    return ax


def plot_underwater_drawdown(
    nav: pd.DataFrame,
    *,
    date_column: str = "date",
    nav_column: str = "nav",
    ax: plt.Axes | None = None,
    title: str | None = None,
    limit_levels_pct: tuple[float, float] = (-15.0, -20.0),
) -> plt.Axes:
    """Plot the underwater (drawdown) series as a bar chart."""

    frame = _to_nav_frame(nav, date_column=date_column, nav_column=nav_column)

    if ax is None:
        _, ax = plt.subplots(figsize=(14, 7))

    dates = frame[date_column].to_numpy()
    nav_values = frame[nav_column].to_numpy(dtype=float)

    drawdowns_pct = _drawdown_from_nav(nav_values) * 100.0
    palette = _palette()
    bar_color = next(palette)
    threshold_color = next(palette)
    worst_color = next(palette)

    ax.bar(dates, drawdowns_pct, color=bar_color, alpha=0.6, width=1.0, label="Drawdown (%)")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.3)

    for level in limit_levels_pct:
        ax.axhline(
            y=level,
            color=threshold_color if level == limit_levels_pct[0] else worst_color,
            linestyle="--",
            linewidth=2,
            alpha=0.6,
            label=f"Threshold: {level:.0f}%",
        )

    min_dd_idx = int(np.argmin(drawdowns_pct))
    worst_dd = float(drawdowns_pct[min_dd_idx])
    ax.scatter(dates[min_dd_idx], worst_dd, color=worst_color, s=90, zorder=5)
    ax.annotate(
        f"Worst: {worst_dd:.1f}%\n({pd.to_datetime(dates[min_dd_idx]).date()})",
        xy=(dates[min_dd_idx], worst_dd),
        xytext=(20, -20),
        textcoords="offset points",
        fontsize=10,
        ha="left",
        bbox={"boxstyle": "round,pad=0.5", "fc": "white", "alpha": 0.6},
        arrowprops={
            "arrowstyle": "->",
            "connectionstyle": "arc3,rad=0",
            "color": worst_color,
        },
    )

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Drawdown (%)", fontsize=11)
    if title is None:
        start = pd.to_datetime(dates[0]).date()
        end = pd.to_datetime(dates[-1]).date()
        title = f"Underwater Drawdown ({start} to {end})"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=11, loc="lower left")
    ax.set_ylim([float(drawdowns_pct.min()) * 1.1, 2.0])

    return ax


def plot_daily_returns_dashboard(
    nav: pd.DataFrame,
    *,
    date_column: str = "date",
    nav_column: str = "nav",
    daily_return_column: str = "daily_return",
    figsize: tuple[float, float] = (14, 10),
) -> plt.Figure:
    """Create a 2x2 dashboard with daily return diagnostics."""

    frame = _to_nav_frame(nav, date_column=date_column, nav_column=nav_column)
    daily_returns = _daily_returns_from_nav(
        frame, nav_column=nav_column, daily_return_column=daily_return_column
    ).to_numpy(dtype=float)
    daily_returns_pct = daily_returns * 100.0

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    palette = _palette()
    hist_color = next(palette)
    cumulative_color = next(palette)
    vol_color = next(palette)
    pie_colors = [next(palette) for _ in range(3)]

    # 1) Daily returns histogram
    axes[0, 0].hist(
        daily_returns_pct, bins=50, color=hist_color, alpha=0.6, edgecolor="black"
    )
    mean = float(np.mean(daily_returns_pct))
    median = float(np.median(daily_returns_pct))
    axes[0, 0].axvline(
        mean,
        color=next(palette),
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean:.3f}%",
    )
    axes[0, 0].axvline(
        median,
        color=next(palette),
        linestyle="--",
        linewidth=2,
        label=f"Median: {median:.3f}%",
    )
    axes[0, 0].set_xlabel("Daily Return (%)", fontsize=10)
    axes[0, 0].set_ylabel("Frequency", fontsize=10)
    axes[0, 0].set_title("Distribution of Daily Returns", fontsize=11, fontweight="bold")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    # 2) Cumulative return over time (derived from NAV for consistency)
    nav_values = frame[nav_column].to_numpy(dtype=float)
    cumulative_pct = (nav_values / nav_values[0] - 1.0) * 100.0
    axes[0, 1].plot(frame[date_column].to_numpy(), cumulative_pct, linewidth=2, color=cumulative_color)
    axes[0, 1].fill_between(
        frame[date_column].to_numpy(), 0.0, cumulative_pct, alpha=0.25, color=cumulative_color
    )
    axes[0, 1].set_xlabel("Date", fontsize=10)
    axes[0, 1].set_ylabel("Cumulative Return (%)", fontsize=10)
    axes[0, 1].set_title("Cumulative Return (from NAV)", fontsize=11, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3)

    # 3) Rolling volatility (21D)
    rolling_vol = pd.Series(daily_returns_pct).rolling(21).std()
    axes[1, 0].plot(frame[date_column].to_numpy(), rolling_vol, linewidth=1.5, color=vol_color, alpha=0.8)
    axes[1, 0].fill_between(frame[date_column].to_numpy(), rolling_vol, alpha=0.2, color=vol_color)
    vol_mean = float(np.nanmean(rolling_vol.to_numpy(dtype=float)))
    axes[1, 0].axhline(
        vol_mean,
        color=next(palette),
        linestyle="--",
        linewidth=1.5,
        label=f"Mean Vol: {vol_mean:.2f}%",
    )
    axes[1, 0].set_xlabel("Date", fontsize=10)
    axes[1, 0].set_ylabel("21-Day Rolling Volatility (%)", fontsize=10)
    axes[1, 0].set_title("Rolling Volatility (21D)", fontsize=11, fontweight="bold")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    # 4) Positive vs negative days
    positive_days = int(np.sum(daily_returns_pct > 0))
    negative_days = int(np.sum(daily_returns_pct < 0))
    zero_days = int(np.sum(daily_returns_pct == 0))

    sizes = [positive_days, negative_days, zero_days]
    labels = [
        f"Positive\n({positive_days} days)",
        f"Negative\n({negative_days} days)",
        f"Zero\n({zero_days} days)",
    ]
    axes[1, 1].pie(
        sizes,
        labels=labels,
        colors=pie_colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 10},
    )
    axes[1, 1].set_title("Distribution of Daily Outcomes", fontsize=11, fontweight="bold")

    fig.tight_layout()
    return fig


def build_default_baseline_points() -> list[BaselinePoint]:
    """Return the baseline points embedded in the OOS figures script."""

    return [
        BaselinePoint("Equal-Weight", 4.32, 0.2618),
        BaselinePoint("Risk Parity", 3.99, 0.2304),
        BaselinePoint("Min-Var (LW)", 1.30, -0.5476),
        BaselinePoint("Shrunk MV", 3.63, 0.1770),
        BaselinePoint("60/40", 3.86, 0.2268),
        BaselinePoint("HRP", 0.87, -0.3049),
    ]


def plot_risk_return_scatter(
    *,
    prism_return_pct: float,
    prism_sharpe: float,
    baselines: pd.DataFrame | None = None,
    baseline_points: list[BaselinePoint] | None = None,
    ax: plt.Axes | None = None,
    title: str = "Risk/Return: Strategy vs Baselines",
) -> plt.Axes:
    """Scatter plot comparing PRISM-R against baselines in risk/return space."""

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8))

    palette = _palette()
    baseline_color = next(palette)
    prism_color = next(palette)

    points: list[BaselinePoint] = []
    if baselines is not None and not baselines.empty:
        cols = {c.lower(): c for c in baselines.columns}
        name_col = cols.get("strategy") or cols.get("name") or cols.get("strategy_name")
        return_col = cols.get("annualized_return") or cols.get("annual_return") or cols.get("return")
        sharpe_col = cols.get("sharpe") or cols.get("sharpe_ratio")
        if name_col and return_col and sharpe_col:
            for _, row in baselines.iterrows():
                try:
                    points.append(
                        BaselinePoint(
                            str(row[name_col]),
                            float(row[return_col]) * 100.0,
                            float(row[sharpe_col]),
                        )
                    )
                except Exception:
                    continue

    if not points:
        points = list(baseline_points or build_default_baseline_points())

    if points:
        returns = np.array([p.annual_return_pct for p in points], dtype=float)
        sharpes = np.array([p.sharpe_ratio for p in points], dtype=float)

        ax.scatter(
            returns,
            sharpes,
            s=200,
            alpha=0.6,
            color=baseline_color,
            label="Baseline strategies",
            edgecolors=baseline_color,
            linewidth=1.5,
        )
        for p in points:
            ax.annotate(
                p.name,
                (p.annual_return_pct, p.sharpe_ratio),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                alpha=0.85,
            )

    ax.scatter(
        [float(prism_return_pct)],
        [float(prism_sharpe)],
        s=500,
        alpha=0.85,
        color=prism_color,
        marker="D",
        label="PRISM-R",
        edgecolors=prism_color,
        linewidth=2.5,
        zorder=5,
    )
    ax.annotate(
        "PRISM-R",
        (float(prism_return_pct), float(prism_sharpe)),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=11,
        fontweight="bold",
        color=prism_color,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.6},
    )

    ax.set_xlabel("Annualized Return (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Sharpe Ratio", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc="upper left")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)

    return ax
