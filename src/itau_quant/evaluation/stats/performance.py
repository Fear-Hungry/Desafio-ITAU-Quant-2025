"""Performance metrics used throughout evaluation workflows.

The functions below are deliberately pandas-friendly: they accept ``Series`` or
``DataFrame`` inputs, preserve column labels, tolerate missing observations, and
return results in structures that can be fed directly into reports/plots.  All
returns are assumed to be expressed in decimal form (``0.01 == 1%``).

The module is intentionally self-contained so the same primitives can be
reused by the backtesting and optimisation layers without bringing in external
dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .risk import max_drawdown

ReturnsLike = Union[pd.Series, pd.DataFrame]
RiskFreeLike = Union[float, pd.Series, pd.DataFrame]

DEFAULT_PERIODS_PER_YEAR = 252


def _to_frame(data: ReturnsLike) -> pd.DataFrame:
    """Coerce supported inputs into a float ``DataFrame``."""

    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, pd.Series):
        df = data.to_frame()
    else:
        raise TypeError("returns must be a pandas Series or DataFrame")

    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if df.empty:
        raise ValueError("returns series is empty after dropping NaNs")
    return df.astype(float)


def _prepare_risk_free(rf: RiskFreeLike, index: pd.Index) -> pd.DataFrame:
    """Align a risk-free input with the provided index."""

    if isinstance(rf, (int, float)):
        rf_series = pd.Series(float(rf), index=index)
    elif isinstance(rf, pd.Series):
        rf_series = rf.reindex(index).ffill()
    elif isinstance(rf, pd.DataFrame):
        if rf.shape[1] != 1:
            raise ValueError("risk-free DataFrame must have a single column")
        rf_series = rf.iloc[:, 0].reindex(index).ffill()
    else:
        raise TypeError("rf must be float, Series or single-column DataFrame")

    return rf_series.to_frame(name="rf")


def _column_apply(
    df: pd.DataFrame,
    func: Callable[[pd.Series], float],
) -> pd.Series:
    """Apply ``func`` to each column individually preserving labels."""

    results = {}
    for column in df.columns:
        series = df[column].dropna()
        if series.empty:
            results[column] = np.nan
            continue
        results[column] = func(series)
    return pd.Series(results, dtype=float, index=df.columns)


def cumulative_return(returns: ReturnsLike) -> pd.Series:
    """Return the total compounded return for each column."""

    df = _to_frame(returns)

    def _cum(series: pd.Series) -> float:
        if (series <= -1.0).any():
            return np.nan
        growth = float(np.prod(1.0 + series))
        return growth - 1.0

    return _column_apply(df, _cum)


def annualized_return(
    returns: ReturnsLike,
    *,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
) -> pd.Series:
    """Compute the annualised geometric return for each column."""

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    df = _to_frame(returns)

    def _annualise(series: pd.Series) -> float:
        if (series <= -1.0).any():
            return np.nan
        n = series.count()
        if n == 0:
            return np.nan
        total_growth = float(np.prod(1.0 + series))
        if total_growth <= 0:
            return np.nan
        return total_growth ** (periods_per_year / n) - 1.0

    return _column_apply(df, _annualise)


def annualized_volatility(
    returns: ReturnsLike,
    *,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
    ddof: int = 1,
) -> pd.Series:
    """Annualised standard deviation of returns."""

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    if ddof < 0:
        raise ValueError("ddof must be non-negative")

    df = _to_frame(returns)

    def _vol(series: pd.Series) -> float:
        if series.count() <= ddof:
            return np.nan
        std = float(series.std(ddof=ddof))
        return std * np.sqrt(periods_per_year)

    return _column_apply(df, _vol)


def _newey_west_variance(series: pd.Series, lags: Optional[int]) -> float:
    """Heteroskedasticity and autocorrelation consistent variance estimate."""

    x = series.to_numpy(dtype=float)
    n = len(x)
    if n == 0:
        return np.nan
    demeaned = x - np.nanmean(x)
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
        lags = max(lags, 1)
    lags = min(lags, n - 1)
    gamma0 = float(np.nansum(demeaned * demeaned) / n)
    var = gamma0
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1)
        cov = float(np.nansum(demeaned[lag:] * demeaned[:-lag]) / n)
        var += 2.0 * weight * cov
    return var


def sharpe_ratio(
    returns: ReturnsLike,
    *,
    rf: RiskFreeLike = 0.0,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
    method: str = "simple",
    lags: Optional[int] = None,
) -> pd.Series:
    """Compute the annualised Sharpe ratio."""

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    df = _to_frame(returns)
    rf_df = _prepare_risk_free(rf, df.index)
    rf_aligned = rf_df.iloc[:, 0]
    method = method.lower()
    if method not in {"simple", "hac"}:
        raise ValueError("method must be either 'simple' or 'hac'")

    def _sharpe(series: pd.Series) -> float:
        clean = series.dropna()
        if clean.empty:
            return np.nan
        rf_series = rf_aligned.loc[clean.index]
        excess = clean - rf_series
        mean = float(excess.mean())
        if method == "simple":
            std = float(excess.std(ddof=1))
        else:
            var = _newey_west_variance(excess, lags)
            std = float(np.sqrt(max(var, 0.0)))
        if not np.isfinite(std) or std == 0.0:
            return np.nan
        return mean / std * np.sqrt(periods_per_year)

    return _column_apply(df, _sharpe)


def sortino_ratio(
    returns: ReturnsLike,
    *,
    rf: RiskFreeLike = 0.0,
    target: float = 0.0,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
) -> pd.Series:
    """Compute the Sortino ratio using a downside semideviation."""

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    df = _to_frame(returns)
    rf_df = _prepare_risk_free(rf, df.index)
    rf_aligned = rf_df.iloc[:, 0]

    def _sortino(series: pd.Series) -> float:
        clean = series.dropna()
        if clean.empty:
            return np.nan
        excess = clean - rf_aligned.loc[clean.index]
        downside = np.minimum(0.0, excess - target)
        denom = float(np.sqrt(np.mean(np.square(downside))))
        if denom == 0.0:
            return np.nan
        mean = float(np.mean(excess - target))
        return mean / denom * np.sqrt(periods_per_year)

    return _column_apply(df, _sortino)


def calmar_ratio(
    returns: ReturnsLike,
    *,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
) -> pd.Series:
    """Return the Calmar ratio (CAGR divided by max drawdown)."""

    ann_ret = annualized_return(returns, periods_per_year=periods_per_year)
    drawdown = max_drawdown(returns)
    result = ann_ret.copy()
    for column in result.index:
        dd = drawdown.get(column, np.nan)
        if dd == 0 or not np.isfinite(dd):
            result.loc[column] = np.nan
        else:
            result.loc[column] = ann_ret.loc[column] / abs(dd)
    return result


def hit_rate(returns: ReturnsLike, *, threshold: float = 0.0) -> pd.Series:
    """Share of observations exceeding ``threshold`` per column."""

    df = _to_frame(returns)

    def _hit(series: pd.Series) -> float:
        clean = series.dropna()
        if clean.empty:
            return np.nan
        return float((clean > threshold).mean())

    return _column_apply(df, _hit)


def _align_benchmark(
    strategy: pd.DataFrame,
    benchmark: ReturnsLike,
) -> pd.DataFrame:
    if benchmark is None:
        raise ValueError("benchmark cannot be None")
    bench = _to_frame(benchmark)
    bench = bench.reindex(strategy.index)
    if bench.shape[1] == 1 and bench.columns[0] not in strategy.columns:
        broadened = pd.concat([bench.iloc[:, 0]] * len(strategy.columns), axis=1)
        broadened.columns = strategy.columns
        return broadened
    missing = [col for col in strategy.columns if col not in bench.columns]
    if missing:
        raise ValueError(
            "Benchmark is missing columns: %s" % ", ".join(sorted(missing))
        )
    return bench[strategy.columns]


@dataclass(frozen=True)
class ExcessMetrics:
    """Container with active-return statistics vs. a benchmark."""

    active_return: pd.Series
    tracking_error: pd.Series
    information_ratio: pd.Series
    beta: pd.Series


def excess_vs_benchmark(
    returns: ReturnsLike,
    benchmark: ReturnsLike,
    *,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
) -> ExcessMetrics:
    """Compute active statistics relative to a benchmark."""

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    strategy = _to_frame(returns)
    bench = _align_benchmark(strategy, benchmark)

    diff = strategy.subtract(bench)

    def _active_return(series: pd.Series) -> float:
        if series.empty:
            return np.nan
        mean = float(series.mean())
        return mean * periods_per_year

    active_return = _column_apply(diff, _active_return)
    tracking = annualized_volatility(diff, periods_per_year=periods_per_year)

    info_ratio = active_return / tracking

    def _beta(col: str) -> float:
        s = strategy[col].dropna()
        if s.empty:
            return np.nan
        b = bench[col].reindex(s.index).dropna()
        aligned = s.reindex(b.index).dropna()
        if aligned.empty:
            return np.nan
        bench_aligned = b.loc[aligned.index]
        cov = float(np.cov(aligned, bench_aligned, ddof=1)[0, 1])
        var = float(np.var(bench_aligned, ddof=1))
        if var == 0:
            return np.nan
        return cov / var

    beta_values = pd.Series({col: _beta(col) for col in strategy.columns}, dtype=float)

    return ExcessMetrics(
        active_return=active_return,
        tracking_error=tracking,
        information_ratio=info_ratio,
        beta=beta_values,
    )


def aggregate_performance(
    returns: ReturnsLike,
    *,
    benchmark: Optional[ReturnsLike] = None,
    rf: RiskFreeLike = 0.0,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
) -> pd.DataFrame:
    """Compute a panel of performance metrics."""

    strategy = _to_frame(returns)

    metrics: list[Tuple[Tuple[str, str], pd.Series]] = []

    metrics.append((("performance", "cumulative_return"), cumulative_return(strategy)))
    metrics.append(
        (("performance", "annualized_return"),
         annualized_return(strategy, periods_per_year=periods_per_year)),
    )
    metrics.append(
        (("performance", "annualized_volatility"),
         annualized_volatility(strategy, periods_per_year=periods_per_year)),
    )
    metrics.append(
        (("performance", "sharpe_ratio"),
         sharpe_ratio(strategy, rf=rf, periods_per_year=periods_per_year)),
    )
    metrics.append(
        (("performance", "sortino_ratio"),
         sortino_ratio(strategy, rf=rf, periods_per_year=periods_per_year)),
    )
    metrics.append((("performance", "hit_rate"), hit_rate(strategy)))
    metrics.append((("risk", "max_drawdown"), max_drawdown(strategy)))
    metrics.append(
        (("risk", "calmar_ratio"), calmar_ratio(strategy, periods_per_year=periods_per_year)),
    )

    if benchmark is not None:
        excess = excess_vs_benchmark(strategy, benchmark, periods_per_year=periods_per_year)
        metrics.append((("relative", "active_return"), excess.active_return))
        metrics.append((("relative", "tracking_error"), excess.tracking_error))
        metrics.append((("relative", "information_ratio"), excess.information_ratio))
        metrics.append((("relative", "beta"), excess.beta))

    idx = pd.MultiIndex.from_tuples([key for key, _ in metrics], names=["category", "metric"])
    data = pd.DataFrame([series.reindex(strategy.columns) for _, series in metrics], index=idx)
    return data


__all__ = [
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "hit_rate",
    "excess_vs_benchmark",
    "aggregate_performance",
    "cumulative_return",
    "ExcessMetrics",
]
