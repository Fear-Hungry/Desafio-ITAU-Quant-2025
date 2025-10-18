"""Helper functions to compute portfolio metrics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
import pandas as pd

__all__ = [
    "PortfolioMetrics",
    "compute_performance_metrics",
    "cumulative_nav",
    "max_drawdown",
]


@dataclass(frozen=True)
class PortfolioMetrics:
    """Summary statistics for a portfolio timeseries."""

    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    cumulative_nav: float

    def as_dict(self) -> Mapping[str, float]:
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "annualized_volatility": self.annualized_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "cumulative_nav": self.cumulative_nav,
        }


def cumulative_nav(returns: pd.Series) -> pd.Series:
    """Compute cumulative NAV given a series of returns."""

    if returns.empty:
        return pd.Series(dtype=float)
    nav = (1.0 + returns).cumprod()
    nav.name = "nav"
    return nav


def max_drawdown(nav: pd.Series) -> float:
    """Return the maximum drawdown from a NAV series."""

    if nav.empty:
        return 0.0
    cumulative_max = nav.cummax()
    drawdown = nav / cumulative_max - 1.0
    return float(drawdown.min())


def _prepare_rf_series(rf: float | pd.Series | None, index: pd.Index) -> pd.Series:
    if rf is None:
        return pd.Series(0.0, index=index)
    if isinstance(rf, (int, float)):
        # Assume rf is annualised; convert to daily equivalent.
        daily = (1.0 + float(rf)) ** (1.0 / 252.0) - 1.0
        return pd.Series(daily, index=index)
    return rf.reindex(index).fillna(0.0)


def compute_performance_metrics(
    returns: pd.Series,
    *,
    risk_free: float | pd.Series | None = None,
    periods_in_year: int = 252,
) -> PortfolioMetrics:
    """Compute standard performance statistics for daily returns."""

    if returns.empty:
        return PortfolioMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    nav = cumulative_nav(returns)
    total_return = float(nav.iloc[-1] - 1.0)

    n_periods = len(returns)
    annualised_return = float((1.0 + total_return) ** (periods_in_year / n_periods) - 1.0)

    volatility = float(returns.std(ddof=0) * math.sqrt(periods_in_year))

    rf_series = _prepare_rf_series(risk_free, returns.index)
    excess_returns = returns - rf_series
    sharpe = 0.0
    denom = returns.std(ddof=0)
    if denom > 0:
        sharpe = float(excess_returns.mean() / denom * math.sqrt(periods_in_year))

    drawdown = max_drawdown(nav)

    return PortfolioMetrics(
        total_return=total_return,
        annualized_return=annualised_return,
        annualized_volatility=volatility,
        sharpe_ratio=sharpe,
        max_drawdown=drawdown,
        cumulative_nav=float(nav.iloc[-1]),
    )
