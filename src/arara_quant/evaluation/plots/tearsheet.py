"""Standard figures used in performance tearsheets.

All functions accept a pandas ``Series``/``DataFrame`` and return the Matplotlib
``Axes`` used (the ``Figure`` can be retrieved via ``ax.figure``).  This keeps the
API flexible for notebooks while allowing downstream code to stitch multiple
plots into reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..stats.risk import drawdown_series, risk_contribution

ReturnsLike = Union[pd.Series, pd.DataFrame]
WeightsLike = Union[pd.Series, pd.DataFrame]
MatrixLike = Union[pd.DataFrame, np.ndarray]


def _to_frame(data: ReturnsLike) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, pd.Series):
        df = data.to_frame()
    else:
        raise TypeError("returns must be a pandas Series or DataFrame")
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if df.empty:
        raise ValueError("returns are empty after dropping NaNs")
    return df.astype(float)


def _prepare_axis(ax: plt.Axes | None = None, *, title: str | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    if title:
        ax.set_title(title)
    return ax


def plot_cumulative_returns(
    returns: ReturnsLike,
    *,
    benchmark: ReturnsLike | None = None,
    ax: plt.Axes | None = None,
    title: str = "Cumulative Returns",
) -> plt.Axes:
    """Plot cumulative growth of each strategy column versus an optional benchmark."""

    frame = _to_frame(returns)
    ax = _prepare_axis(ax, title=title)

    cumulative = (1.0 + frame).cumprod()
    for column in cumulative.columns:
        ax.plot(cumulative.index, cumulative[column], label=column)

    if benchmark is not None:
        bench = _to_frame(benchmark).reindex(frame.index)
        bench_cum = (1.0 + bench).cumprod()
        for column in bench_cum.columns:
            label = (
                f"Benchmark - {column}"
                if column not in cumulative.columns
                else "Benchmark"
            )
            ax.plot(bench_cum.index, bench_cum[column], linestyle="--", label=label)

    ax.set_ylabel("Growth (1 = start)")
    ax.set_xlabel("Date")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return ax


def plot_drawdown(
    returns: ReturnsLike,
    *,
    ax: plt.Axes | None = None,
    title: str = "Drawdown",
) -> plt.Axes:
    """Plot the drawdown series for each column."""

    frame = _to_frame(returns)
    dd = drawdown_series(frame)
    ax = _prepare_axis(ax, title=title)

    for column in dd.columns:
        ax.fill_between(dd.index, dd[column], 0, alpha=0.3, label=column)

    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return ax


def plot_rolling_sharpe(
    returns: ReturnsLike,
    *,
    window: int = 126,
    periods_per_year: float = 252.0,
    ax: plt.Axes | None = None,
    title: str = "Rolling Sharpe",
) -> plt.Axes:
    """Plot the rolling Sharpe ratio with a simple estimator (no HAC)."""

    if window <= 1:
        raise ValueError("window must be greater than 1")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    frame = _to_frame(returns)
    ax = _prepare_axis(ax, title=title)

    for column in frame.columns:
        rolling_mean = frame[column].rolling(window=window).mean()
        rolling_std = frame[column].rolling(window=window).std(ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            roll_sharpe = rolling_mean / rolling_std * np.sqrt(periods_per_year)
        ax.plot(frame.index, roll_sharpe, label=column)

    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Sharpe")
    ax.set_xlabel("Date")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return ax


def plot_rolling_volatility(
    returns: ReturnsLike,
    *,
    window: int = 126,
    ax: plt.Axes | None = None,
    title: str = "Rolling Volatility",
) -> plt.Axes:
    """Plot rolling realisations of standard deviation."""

    if window <= 1:
        raise ValueError("window must be greater than 1")

    frame = _to_frame(returns)
    ax = _prepare_axis(ax, title=title)

    for column in frame.columns:
        rolling_vol = frame[column].rolling(window=window).std(ddof=1)
        ax.plot(frame.index, rolling_vol, label=column)

    ax.set_ylabel("Volatility")
    ax.set_xlabel("Date")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return ax


def plot_turnover(
    turnover: pd.Series | pd.DataFrame,
    *,
    target_band: tuple[float, float] | None = None,
    ax: plt.Axes | None = None,
    title: str = "Turnover",
) -> plt.Axes:
    """Plot turnover history with optional target band."""

    if isinstance(turnover, pd.DataFrame):
        frame = turnover.copy()
    elif isinstance(turnover, pd.Series):
        frame = turnover.to_frame(name=turnover.name or "turnover")
    else:
        raise TypeError("turnover must be Series or DataFrame")

    frame = frame.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if frame.empty:
        raise ValueError("turnover is empty")

    ax = _prepare_axis(ax, title=title)

    for column in frame.columns:
        ax.plot(frame.index, frame[column], label=column)

    if target_band is not None:
        low, high = target_band
        ax.axhspan(low, high, color="grey", alpha=0.1, label="Target band")

    ax.set_ylabel("Turnover")
    ax.set_xlabel("Date")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return ax


def plot_risk_contribution(
    weights: WeightsLike,
    cov: MatrixLike,
    *,
    ax: plt.Axes | None = None,
    title: str = "Risk Contribution",
) -> plt.Axes:
    """Plot stacked percentage risk contributions of the latest weights."""

    result = risk_contribution(weights, cov)
    latest = result.percentage.iloc[-1]
    ax = _prepare_axis(ax, title=title)

    ax.bar(latest.index, latest.values, color=plt.cm.tab20.colors[: len(latest)])
    ax.set_ylabel("Risk share")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    return ax


@dataclass
class TearsheetFigure:
    title: str
    figure: plt.Figure


def generate_tearsheet(
    figures: Sequence[tuple[str, plt.Axes]]
) -> list[TearsheetFigure]:
    """Wrap axes in a serialisable structure for downstream reporting."""

    wrapped: list[TearsheetFigure] = []
    for title, ax in figures:
        if not isinstance(ax, plt.Axes):
            raise TypeError("generate_tearsheet expects (title, Axes) tuples")
        wrapped.append(TearsheetFigure(title=title, figure=ax.figure))
    return wrapped


__all__ = [
    "plot_cumulative_returns",
    "plot_drawdown",
    "plot_rolling_sharpe",
    "plot_rolling_volatility",
    "plot_turnover",
    "plot_risk_contribution",
    "generate_tearsheet",
    "TearsheetFigure",
]
