"""Deterministic risk/return measures."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "historical_cvar",
    "tracking_error",
    "information_ratio",
    "rolling_metric",
]


def volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return float("nan")
    return float(returns.std(ddof=0) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    rf: float | pd.Series = 0.0,
    periods_per_year: int = 252,
) -> float:
    returns = pd.Series(returns).dropna()
    if isinstance(rf, pd.Series):
        rf_series = rf.reindex(returns.index).fillna(0.0)
    else:
        rf_series = pd.Series(float(rf), index=returns.index)
    excess = returns - rf_series
    vol = excess.std(ddof=0)
    if vol == 0:
        return float("nan")
    return float(excess.mean() / vol * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    rf: float = 0.0,
    target: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    returns = pd.Series(returns).dropna()
    downside = returns[returns < target] - target
    if downside.empty:
        return float("nan")
    downside_vol = np.sqrt((downside**2).mean())
    if downside_vol == 0:
        return float("nan")
    excess = returns.mean() - rf
    return float(excess / downside_vol * np.sqrt(periods_per_year))


def max_drawdown(nav: pd.Series) -> tuple[float, pd.Series]:
    nav = pd.Series(nav).dropna()
    if nav.empty:
        return float("nan"), nav
    rolling_max = nav.cummax()
    drawdown = nav / rolling_max - 1.0
    return float(drawdown.min()), drawdown


def historical_cvar(returns: pd.Series, alpha: float = 0.95) -> float:
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return float("nan")
    losses = -returns.sort_values()
    cutoff = int(np.ceil((1 - alpha) * len(losses)))
    cutoff = max(cutoff, 1)
    return float(losses.head(cutoff).mean())


def tracking_error(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    diff = (
        pd.Series(strategy_returns).align(pd.Series(benchmark_returns), join="inner")[0]
        - pd.Series(benchmark_returns).align(pd.Series(strategy_returns), join="inner")[
            0
        ]
    )
    diff = diff.dropna()
    if diff.empty:
        return float("nan")
    return float(diff.std(ddof=0) * np.sqrt(periods_per_year))


def information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    diff = (
        pd.Series(strategy_returns).align(pd.Series(benchmark_returns), join="inner")[0]
        - pd.Series(benchmark_returns).align(pd.Series(strategy_returns), join="inner")[
            0
        ]
    )
    diff = diff.dropna()
    if diff.empty:
        return float("nan")
    te = diff.std(ddof=0)
    if te == 0:
        return float("nan")
    return float(diff.mean() / te * np.sqrt(periods_per_year))


def rolling_metric(series: pd.Series, window: int, func) -> pd.Series:
    series = pd.Series(series)
    return series.rolling(window).apply(lambda x: func(pd.Series(x)), raw=False)
