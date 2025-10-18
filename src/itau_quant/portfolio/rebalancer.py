"""Portfolio rebalancing orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd

from itau_quant.costs.transaction_costs import transaction_cost_vector
from itau_quant.optimization.core.mv_qp import MeanVarianceConfig, solve_mean_variance
from itau_quant.portfolio.rounding import RoundingResult, rounding_pipeline

__all__ = [
    "MarketData",
    "RebalanceMetrics",
    "RebalanceResult",
    "prepare_inputs",
    "run_estimators",
    "optimize_portfolio",
    "apply_postprocessing",
    "compute_costs",
    "build_rebalance_log",
    "rebalance",
]


@dataclass(frozen=True)
class MarketData:
    prices: pd.DataFrame | pd.Series
    returns: pd.DataFrame


@dataclass(frozen=True)
class RebalanceMetrics:
    optimizer_expected_return: float
    optimizer_variance: float
    optimizer_turnover: float
    optimizer_cost: float
    rounding_cost: float
    realized_turnover: float

    def to_dict(self) -> dict[str, float]:
        return {
            "optimizer_expected_return": self.optimizer_expected_return,
            "optimizer_variance": self.optimizer_variance,
            "optimizer_turnover": self.optimizer_turnover,
            "optimizer_cost": self.optimizer_cost,
            "rounding_cost": self.rounding_cost,
            "realized_turnover": self.realized_turnover,
        }


@dataclass
class RebalanceResult:
    date: pd.Timestamp
    weights: pd.Series
    rounded_weights: pd.Series
    shares: pd.Series
    cash: float
    metrics: RebalanceMetrics
    trades: pd.Series
    log: Mapping[str, Any]
    rounding: RoundingResult
    notes: list[str] = field(default_factory=list)


def prepare_inputs(
    date: pd.Timestamp,
    market_data: MarketData,
    returns_window: int,
) -> tuple[pd.Series, pd.DataFrame]:
    """Prepare prices and return window for estimation."""

    prices_obj = market_data.prices
    if isinstance(prices_obj, pd.Series) and not isinstance(prices_obj.index, pd.DatetimeIndex):
        prices_at_date = prices_obj.astype(float)
    else:
        prices_df = pd.DataFrame(prices_obj).astype(float)
        if date not in prices_df.index:
            raise ValueError("Provided prices must include the rebalance date.")
        prices_at_date = prices_df.loc[date]

    historical_returns = market_data.returns.loc[:date].dropna(how="all")
    if historical_returns.empty:
        raise ValueError("No historical returns available up to the rebalance date.")

    if returns_window > 0:
        historical_returns = historical_returns.tail(returns_window)

    asset_index = prices_at_date.index if isinstance(prices_at_date, pd.Series) else pd.Index(prices_at_date.columns)
    historical_returns = historical_returns.reindex(columns=asset_index).dropna(axis=1, how="all")
    return prices_at_date.astype(float), historical_returns


def run_estimators(
    returns: pd.DataFrame,
    *,
    mu_config: Mapping[str, Any] | None = None,
    sigma_config: Mapping[str, Any] | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Estimate expected returns and covariance matrix."""

    mu_config = dict(mu_config or {})
    sigma_config = dict(sigma_config or {})

    method = (mu_config.get("method") or "simple").lower()
    if method in {"simple", "mean"}:
        mu = returns.mean()
    elif method == "geometric":
        growth = 1.0 + returns
        mu = growth.prod() ** (1.0 / len(growth)) - 1.0
    else:
        raise ValueError(f"Unsupported mean estimator '{method}'.")

    cov_method = (sigma_config.get("method") or "ledoit_wolf").lower()
    if cov_method == "ledoit_wolf":
        cov = returns.cov(ddof=1)
    elif cov_method == "sample":
        cov = returns.cov(ddof=1)
    else:
        raise ValueError(f"Unsupported covariance estimator '{cov_method}'.")

    return mu.astype(float), cov.astype(float)


def optimize_portfolio(
    mu: pd.Series,
    cov: pd.DataFrame,
    previous_weights: pd.Series,
    optimizer_config: Mapping[str, Any],
    *,
    risk_config: Mapping[str, Any] | None = None,
) -> tuple[pd.Series, MeanVarianceConfig, Any]:
    """Solve the optimisation problem returning weights and solver summary."""

    optimizer_config = dict(optimizer_config or {})
    risk_aversion = float(optimizer_config.get("risk_aversion", optimizer_config.get("lambda", 5.0)))
    turnover_penalty = float(optimizer_config.get("turnover_penalty", optimizer_config.get("eta", 0.0)))
    turnover_cap = optimizer_config.get("turnover_cap", optimizer_config.get("tau"))
    if turnover_cap is not None:
        turnover_cap = float(turnover_cap)
        if previous_weights.abs().sum() < 1e-9:
            turnover_cap = None
    min_weight = float(optimizer_config.get("min_weight", 0.0))
    max_weight = float(optimizer_config.get("max_weight", 1.0))

    lower_bounds = pd.Series(min_weight, index=mu.index)
    upper_bounds = pd.Series(max_weight, index=mu.index)
    prev = previous_weights.reindex(mu.index).fillna(0.0)

    risk_section = dict(risk_config or {})

    config = MeanVarianceConfig(
        risk_aversion=risk_aversion,
        turnover_penalty=turnover_penalty,
        turnover_cap=turnover_cap,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        previous_weights=prev,
        cost_vector=None,
        solver=optimizer_config.get("solver"),
        solver_kwargs=optimizer_config.get("solver_kwargs"),
        risk_config=risk_section or None,
    )
    result = solve_mean_variance(mu, cov, config)
    return result.weights, config, result


def apply_postprocessing(
    weights: pd.Series,
    prices: pd.Series,
    capital: float,
    rounding_config: Mapping[str, Any] | None,
) -> RoundingResult:
    """Apply rounding and minimum-lot adjustments."""

    return rounding_pipeline(weights, prices, capital, rounding_config)


def compute_costs(
    target_weights: pd.Series,
    previous_weights: pd.Series,
    *,
    capital: float,
    cost_config: Mapping[str, Any] | None,
) -> float:
    """Estimate transaction costs for the rebalance."""

    cost_config = dict(cost_config or {})
    linear_bps = cost_config.get("linear_bps", 0.0)
    if not linear_bps:
        return 0.0
    costs = transaction_cost_vector(
        target_weights,
        previous_weights,
        linear_bps=linear_bps,
        notional=capital,
    )
    return float(costs.sum())


def build_rebalance_log(
    date: pd.Timestamp,
    metrics: RebalanceMetrics,
    solver_summary: Any,
) -> dict[str, Any]:
    """Assemble a serialisable log structure."""

    payload = {
        "date": pd.Timestamp(date).isoformat(),
        "metrics": metrics.to_dict(),
    }
    if solver_summary is not None:
        payload["solver"] = {
            "status": solver_summary.status,
            "value": solver_summary.value,
            "solver": solver_summary.solver,
            "runtime": solver_summary.runtime,
        }
    return payload


def rebalance(
    date: pd.Timestamp,
    market_data: MarketData,
    previous_weights: pd.Series,
    *,
    capital: float,
    config: Mapping[str, Any] | None = None,
) -> RebalanceResult:
    """Orchestrate the full rebalance pipeline."""

    config = dict(config or {})
    optimizer_cfg = dict(config.get("optimizer", {}))
    rounding_cfg = config.get("rounding", {})
    cost_cfg = config.get("costs", {})
    estimator_cfg = config.get("estimators", {})
    risk_cfg: dict[str, Any] = {}

    nested_risk = optimizer_cfg.pop("risk", None)
    if isinstance(nested_risk, Mapping):
        risk_cfg.update({k: v for k, v in nested_risk.items()})

    portfolio_risk = config.get("risk")
    if isinstance(portfolio_risk, Mapping):
        risk_cfg.update({k: v for k, v in portfolio_risk.items()})

    returns_window = int(config.get("returns_window", 252))
    prices_at_date, historical_returns = prepare_inputs(date, market_data, returns_window)

    mu, cov = run_estimators(
        historical_returns,
        mu_config=estimator_cfg.get("mu"),
        sigma_config=estimator_cfg.get("sigma"),
    )

    opt_weights, mv_config, mv_result = optimize_portfolio(
        mu,
        cov,
        previous_weights,
        optimizer_cfg,
        risk_config=risk_cfg or None,
    )
    optimizer_cost = compute_costs(opt_weights, previous_weights, capital=capital, cost_config=cost_cfg)

    rounding_result = apply_postprocessing(opt_weights, prices_at_date, capital, rounding_cfg)

    rounded_weights = rounding_result.rounded_weights.reindex(opt_weights.index, fill_value=0.0)
    trades = rounded_weights - previous_weights.reindex(rounded_weights.index).fillna(0.0)
    realized_turnover = float(trades.abs().sum())

    metrics = RebalanceMetrics(
        optimizer_expected_return=mv_result.expected_return,
        optimizer_variance=mv_result.variance,
        optimizer_turnover=mv_result.turnover,
        optimizer_cost=optimizer_cost,
        rounding_cost=rounding_result.rounding_cost,
        realized_turnover=realized_turnover,
    )

    log = build_rebalance_log(date, metrics, mv_result.summary)

    return RebalanceResult(
        date=pd.Timestamp(date),
        weights=opt_weights.astype(float),
        rounded_weights=rounded_weights.astype(float),
        shares=rounding_result.shares.astype(float),
        cash=float(rounding_result.residual_cash),
        metrics=metrics,
        trades=trades.astype(float),
        log=log,
        rounding=rounding_result,
    )
