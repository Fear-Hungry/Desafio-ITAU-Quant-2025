"""Frequency conversion utilities for returns."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "returns_to_monthly",
    "returns_to_weekly",
    "annualize_return",
    "annualize_volatility",
    "monthly_to_annual",
]


def returns_to_monthly(
    daily_returns: pd.DataFrame | pd.Series,
) -> pd.DataFrame | pd.Series:
    """
    Convert daily returns to monthly compounded returns.

    Uses month-end frequency and compounds returns: (1+r1)*(1+r2)*... - 1

    Parameters
    ----------
    daily_returns : DataFrame or Series
        Daily returns (simple, not log)

    Returns
    -------
    monthly_returns : DataFrame or Series (same type as input)
        Monthly compounded returns

    Examples
    --------
    >>> daily_rets = pd.Series([0.01, 0.02, -0.01],
    ...                         index=pd.date_range('2020-01-01', periods=3))
    >>> monthly_rets = returns_to_monthly(daily_rets)
    """
    return daily_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)


def returns_to_weekly(
    daily_returns: pd.DataFrame | pd.Series,
) -> pd.DataFrame | pd.Series:
    """
    Convert daily returns to weekly compounded returns.

    Uses week-end frequency (Sunday) and compounds returns.

    Parameters
    ----------
    daily_returns : DataFrame or Series
        Daily returns

    Returns
    -------
    weekly_returns : DataFrame or Series
        Weekly compounded returns
    """
    return daily_returns.resample("W").apply(lambda x: (1 + x).prod() - 1)


def annualize_return(periodic_return: float, periods_per_year: int) -> float:
    """
    Annualize a periodic return using compounding.

    Parameters
    ----------
    periodic_return : float
        Return over one period (e.g., monthly return)
    periods_per_year : int
        Number of periods in one year (12 for monthly, 252 for daily)

    Returns
    -------
    annual_return : float
        Annualized return

    Examples
    --------
    >>> annualize_return(0.01, 12)  # 1% monthly
    0.12682503013196977  # ~12.68% annual
    """
    return (1 + periodic_return) ** periods_per_year - 1


def annualize_volatility(periodic_vol: float, periods_per_year: int) -> float:
    """
    Annualize volatility using square-root-of-time scaling.

    Assumes i.i.d. returns (no autocorrelation).

    Parameters
    ----------
    periodic_vol : float
        Volatility over one period
    periods_per_year : int
        Number of periods in one year

    Returns
    -------
    annual_vol : float
        Annualized volatility

    Examples
    --------
    >>> annualize_volatility(0.03, 12)  # 3% monthly vol
    0.10392304845413264  # ~10.4% annual
    """
    return periodic_vol * np.sqrt(periods_per_year)


def monthly_to_annual(
    monthly_stat: float, is_return: bool = True, periods_per_year: int = 12
) -> float:
    """
    Convert monthly statistic to annual.

    Parameters
    ----------
    monthly_stat : float
        Monthly return or volatility
    is_return : bool
        True for returns (use compounding), False for vol (use sqrt scaling)
    periods_per_year : int
        Periods per year (default 12 for monthly)

    Returns
    -------
    annual_stat : float
        Annualized statistic
    """
    if is_return:
        return annualize_return(monthly_stat, periods_per_year)
    else:
        return annualize_volatility(monthly_stat, periods_per_year)
