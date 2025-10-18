"""Risk metrics and decompositions used by reporting pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

ReturnsLike = Union[pd.Series, pd.DataFrame]
WeightsLike = Union[pd.Series, pd.DataFrame]
MatrixLike = Union[pd.DataFrame, np.ndarray]

DEFAULT_PERIODS_PER_YEAR = 252


def _to_frame(data: ReturnsLike, name: str = "returns") -> pd.DataFrame:
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


def drawdown_series(returns: ReturnsLike) -> pd.DataFrame:
    """Compute path-wise drawdowns from periodic returns."""

    df = _to_frame(returns)
    if (df <= -1.0).any().any():
        raise ValueError("returns contain losses <= -100%, cannot compute drawdown")
    nav = (1.0 + df).cumprod()
    peaks = nav.cummax()
    drawdown = nav / peaks - 1.0
    return drawdown


def max_drawdown(returns: ReturnsLike, *, return_details: bool = False):
    """Maximum drawdown for each column.

    When ``return_details`` is ``True`` a tuple ``(series, details)`` is
    returned, where ``details`` maps the column name to ``(depth, start, end)``.
    """

    drawdown = drawdown_series(returns)
    min_drawdown = drawdown.min()
    if not return_details:
        return min_drawdown

    details = {}
    for column in drawdown.columns:
        series = drawdown[column]
        depth = float(series.min())
        if np.isnan(depth):
            details[column] = (np.nan, None, None)
            continue
        trough_idx = series.idxmin()
        peak_mask = series.loc[:trough_idx] == 0
        start = peak_mask.loc[:trough_idx].index[-1] if peak_mask.any() else series.index[0]
        details[column] = (depth, start, trough_idx)
    return min_drawdown, details


def conditional_value_at_risk(
    returns: ReturnsLike,
    *,
    alpha: float = 0.95,
    method: str = "historical",
) -> pd.Series:
    """Historical expected shortfall for each column."""

    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    method = method.lower()
    if method != "historical":
        raise ValueError("Only the 'historical' method is currently supported")

    df = _to_frame(returns)
    tail_prob = 1.0 - alpha

    def _cvar(series: pd.Series) -> float:
        if series.empty:
            return np.nan
        quantile = float(series.quantile(tail_prob, interpolation="lower"))
        tail = series[series <= quantile]
        if tail.empty:
            return quantile
        return float(tail.mean())

    return pd.Series({col: _cvar(df[col].dropna()) for col in df.columns}, dtype=float)


def tracking_error(
    returns: ReturnsLike,
    benchmark: ReturnsLike,
    *,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
    ddof: int = 1,
) -> pd.Series:
    """Annualised tracking error (standard deviation of active returns)."""

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    if ddof < 0:
        raise ValueError("ddof must be non-negative")

    strategy = _to_frame(returns)
    bench = _align_benchmark(strategy, benchmark)
    active = strategy.subtract(bench)

    def _te(series: pd.Series) -> float:
        if series.count() <= ddof:
            return np.nan
        return float(series.std(ddof=ddof) * np.sqrt(periods_per_year))

    return pd.Series({col: _te(active[col].dropna()) for col in active.columns}, dtype=float)


def beta_to_benchmark(returns: ReturnsLike, benchmark: ReturnsLike) -> pd.Series:
    """OLS beta of strategy returns against the benchmark."""

    strategy = _to_frame(returns)
    bench = _align_benchmark(strategy, benchmark)

    betas = {}
    for column in strategy.columns:
        s = strategy[column].dropna()
        b = bench[column].reindex(s.index).dropna()
        aligned = s.reindex(b.index).dropna()
        if aligned.empty:
            betas[column] = np.nan
            continue
        bench_aligned = b.loc[aligned.index]
        cov = float(np.cov(aligned, bench_aligned, ddof=1)[0, 1])
        var = float(np.var(bench_aligned, ddof=1))
        betas[column] = np.nan if var == 0 else cov / var
    return pd.Series(betas, dtype=float)


def realized_leverage(weights: WeightsLike) -> pd.Series:
    """Sum of absolute weights over time."""

    if isinstance(weights, pd.Series):
        value = float(weights.abs().sum())
        name = weights.name if weights.name is not None else 0
        return pd.Series([value], index=[name], name="leverage")
    if isinstance(weights, pd.DataFrame):
        clean = weights.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return clean.abs().sum(axis=1).rename("leverage")
    raise TypeError("weights must be Series or DataFrame")


@dataclass(frozen=True)
class RiskContributionResult:
    component: pd.DataFrame
    marginal: pd.DataFrame
    percentage: pd.DataFrame
    portfolio_volatility: pd.Series


def risk_contribution(weights: WeightsLike, cov: MatrixLike) -> RiskContributionResult:
    """Compute marginal and component risk contributions."""

    if isinstance(weights, pd.Series):
        weights_df = weights.to_frame().T
    elif isinstance(weights, pd.DataFrame):
        weights_df = weights.copy()
    else:
        raise TypeError("weights must be Series or DataFrame")

    weights_df = weights_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if isinstance(cov, pd.DataFrame):
        cov_df = cov.copy().apply(pd.to_numeric, errors="coerce")
    else:
        cov_arr = np.asarray(cov, dtype=float)
        if cov_arr.ndim != 2 or cov_arr.shape[0] != cov_arr.shape[1]:
            raise ValueError("cov must be a square matrix")
        cov_df = pd.DataFrame(cov_arr, index=weights_df.columns, columns=weights_df.columns)

    missing = [col for col in weights_df.columns if col not in cov_df.columns]
    if missing:
        raise ValueError(f"cov matrix missing columns: {', '.join(missing)}")
    cov_df = cov_df.loc[weights_df.columns, weights_df.columns]

    cov_matrix = cov_df.to_numpy(dtype=float)
    weights_arr = weights_df.to_numpy(dtype=float)

    sigma_w = (cov_matrix @ weights_arr.T).T  # Σ w for each row
    component = weights_arr * sigma_w
    portfolio_var = component.sum(axis=1)
    portfolio_vol = np.sqrt(np.clip(portfolio_var, 0.0, None))

    with np.errstate(divide="ignore", invalid="ignore"):
        percentage = np.where(portfolio_var[:, None] != 0, component / portfolio_var[:, None], np.nan)

    component_df = pd.DataFrame(component, index=weights_df.index, columns=weights_df.columns)
    marginal_df = pd.DataFrame(sigma_w, index=weights_df.index, columns=weights_df.columns)
    percentage_df = pd.DataFrame(percentage, index=weights_df.index, columns=weights_df.columns)
    portfolio_vol_series = pd.Series(portfolio_vol, index=weights_df.index, name="portfolio_volatility")

    return RiskContributionResult(
        component=component_df,
        marginal=marginal_df,
        percentage=percentage_df,
        portfolio_volatility=portfolio_vol_series,
    )


@dataclass(frozen=True)
class RiskSummary:
    metrics: pd.DataFrame
    drawdowns: pd.DataFrame
    risk_contribution: Optional[RiskContributionResult]


def _align_benchmark(strategy: pd.DataFrame, benchmark: ReturnsLike) -> pd.DataFrame:
    if benchmark is None:
        raise ValueError("benchmark cannot be None")
    bench = _to_frame(benchmark, name="benchmark").reindex(strategy.index)
    if bench.shape[1] == 1 and bench.columns[0] not in strategy.columns:
        expanded = pd.concat([bench.iloc[:, 0]] * len(strategy.columns), axis=1)
        expanded.columns = strategy.columns
        return expanded
    missing = [col for col in strategy.columns if col not in bench.columns]
    if missing:
        raise ValueError(f"benchmark missing columns: {', '.join(sorted(missing))}")
    return bench[strategy.columns]


def _weights_to_frame(weights: WeightsLike) -> pd.DataFrame:
    if isinstance(weights, pd.Series):
        df = weights.to_frame().T
        df.index = [weights.name] if weights.name is not None else [0]
        return df
    if isinstance(weights, pd.DataFrame):
        return weights.copy()
    raise TypeError("weights must be Series or DataFrame")


def aggregate_risk_metrics(
    returns: ReturnsLike,
    *,
    benchmark: Optional[ReturnsLike] = None,
    weights: Optional[WeightsLike] = None,
    covariance: Optional[MatrixLike] = None,
    alpha: float = 0.95,
    periods_per_year: float = DEFAULT_PERIODS_PER_YEAR,
) -> RiskSummary:
    """Bundle key realised-risk metrics for reporting."""

    strategy = _to_frame(returns)
    drawdowns = drawdown_series(strategy)
    max_dd = drawdowns.min()
    cvar = conditional_value_at_risk(strategy, alpha=alpha)

    def _annualised_vol(s: pd.Series) -> float:
        if s.count() <= 1:
            return np.nan
        return float(s.std(ddof=1) * np.sqrt(periods_per_year))

    volatility = pd.Series({col: _annualised_vol(strategy[col].dropna()) for col in strategy.columns}, dtype=float)

    metrics = [
        (("risk", "max_drawdown"), max_dd),
        (("risk", f"cvar_{int((1 - alpha) * 100)}pct"), cvar),
        (("risk", "volatility"), volatility),
    ]

    contribution_result: Optional[RiskContributionResult] = None

    if benchmark is not None:
        bench_aligned = _align_benchmark(strategy, benchmark)
        te = tracking_error(strategy, bench_aligned, periods_per_year=periods_per_year)
        beta = beta_to_benchmark(strategy, bench_aligned)
        metrics.append((("relative", "tracking_error"), te))
        metrics.append((("relative", "beta"), beta))

    weights_frame: Optional[pd.DataFrame] = None
    if weights is not None:
        weights_frame = _weights_to_frame(weights).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        lev_series = realized_leverage(weights_frame)
        leverage_value = float(lev_series.iloc[-1])
        leverage_row = pd.Series(leverage_value, index=strategy.columns, dtype=float)
        metrics.append((("positioning", "realized_leverage"), leverage_row))

    if covariance is not None and weights_frame is not None:
        latest_weights = weights_frame.iloc[[-1]]
        contribution_result = risk_contribution(latest_weights, covariance)

    index = pd.MultiIndex.from_tuples([key for key, _ in metrics], names=["category", "metric"])
    metrics_df = pd.DataFrame([series.reindex(strategy.columns) for _, series in metrics], index=index)

    return RiskSummary(metrics=metrics_df, drawdowns=drawdowns, risk_contribution=contribution_result)


__all__ = [
    "drawdown_series",
    "max_drawdown",
    "conditional_value_at_risk",
    "tracking_error",
    "beta_to_benchmark",
    "realized_leverage",
    "risk_contribution",
    "RiskContributionResult",
    "RiskSummary",
    "aggregate_risk_metrics",
]
