"""Portfolio rebalancing orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from itau_quant.costs.transaction_costs import transaction_cost_vector
from itau_quant.risk.regime import detect_regime, regime_multiplier
from itau_quant.estimators import cov as covariance_estimators
from itau_quant.optimization.core.mv_qp import (
    MeanVarianceConfig,
    MeanVarianceResult,
    solve_mean_variance,
)
from itau_quant.optimization.heuristics.hrp import heuristic_allocation
from itau_quant.portfolio.cardinality_pipeline import apply_cardinality_constraint
from itau_quant.portfolio.rounding import RoundingResult, rounding_pipeline
from itau_quant.risk.budgets import RiskBudget, load_budgets, validate_budgets

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
    allocator: str
    notes: list[str] = field(default_factory=list)


def prepare_inputs(
    date: pd.Timestamp,
    market_data: MarketData,
    returns_window: int,
) -> tuple[pd.Series, pd.DataFrame]:
    """Prepare prices and return window for estimation."""

    prices_obj = market_data.prices
    if isinstance(prices_obj, pd.Series) and not isinstance(
        prices_obj.index, pd.DatetimeIndex
    ):
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

    asset_index = (
        prices_at_date.index
        if isinstance(prices_at_date, pd.Series)
        else pd.Index(prices_at_date.columns)
    )
    historical_returns = historical_returns.reindex(columns=asset_index).dropna(
        axis=1, how="all"
    )
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

    min_history = int(sigma_config.get("min_history", 30))
    min_history = max(min_history, 2)
    valid_mask = returns.count() >= min_history
    returns_filtered = returns.loc[:, valid_mask]
    if returns_filtered.empty:
        raise ValueError("No assets meet the minimum history requirement for estimation.")

    clean_returns = returns_filtered.dropna(axis=0, how="any")
    if clean_returns.shape[0] < 2:
        raise ValueError(
            "Not enough observations after removing missing data for estimation."
        )

    method = (mu_config.get("method") or "simple").lower()
    if method in {"simple", "mean"}:
        mu = clean_returns.mean()
    elif method == "geometric":
        growth = 1.0 + clean_returns
        mu = growth.prod() ** (1.0 / len(growth)) - 1.0
    else:
        raise ValueError(f"Unsupported mean estimator '{method}'.")

    cov_method = (sigma_config.get("method") or "ledoit_wolf").lower()
    if cov_method == "ledoit_wolf":
        cov = clean_returns.cov(ddof=1)
    elif cov_method == "sample":
        cov = clean_returns.cov(ddof=1)
    else:
        raise ValueError(f"Unsupported covariance estimator '{cov_method}'.")
    cov = cov.astype(float)
    assets = clean_returns.columns
    mu = mu.reindex(assets).astype(float)
    cov = cov.reindex(index=assets, columns=assets)
    cov = covariance_estimators.project_to_psd(cov, epsilon=1e-9)

    return mu.astype(float), cov.astype(float)


def optimize_portfolio(
    mu: pd.Series,
    cov: pd.DataFrame,
    previous_weights: pd.Series,
    optimizer_config: Mapping[str, Any],
    *,
    risk_config: Mapping[str, Any] | None = None,
) -> tuple[
    pd.Series,
    MeanVarianceConfig,
    MeanVarianceResult,
    dict[str, Any] | None,
    dict[str, Any],
]:
    """Solve the optimisation problem returning weights and solver summary."""

    optimizer_config = dict(optimizer_config or {})
    risk_aversion = float(
        optimizer_config.get("risk_aversion", optimizer_config.get("lambda", 5.0))
    )
    turnover_penalty = float(
        optimizer_config.get("turnover_penalty", optimizer_config.get("eta", 0.0))
    )
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

    budgets: Sequence[RiskBudget] | None = None
    raw_budgets = risk_section.pop("budgets", None)
    if raw_budgets:
        if isinstance(raw_budgets, RiskBudget):
            budgets = [raw_budgets]
        else:
            if not isinstance(raw_budgets, (list, tuple, set)):
                raw_iter = [raw_budgets]
            else:
                raw_iter = list(raw_budgets)
            direct: list[RiskBudget] = []
            loadable: list[Mapping[str, Any]] = []
            for item in raw_iter:
                if isinstance(item, RiskBudget):
                    direct.append(item)
                elif isinstance(item, Mapping):
                    loadable.append(item)  # type: ignore[arg-type]
                else:
                    raise TypeError("Budgets must be mappings or RiskBudget objects.")
            budgets = list(direct)
            if loadable:
                budgets.extend(load_budgets(loadable))
        validate_budgets(budgets, mu.index)

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
        budgets=budgets,
    )
    result = solve_mean_variance(mu, cov, config)
    fallback_relaxed_turnover = False
    if (
        config.turnover_cap is not None
        and hasattr(result, "summary")
        and not result.summary.is_optimal()
    ):
        config_relaxed = replace(config, turnover_cap=None)
        try:
            alt_result = solve_mean_variance(mu, cov, config_relaxed)
        except Exception:  # prism_w: keep original failure details
            alt_result = None
        else:
            if hasattr(alt_result, "summary") and alt_result.summary.is_optimal():
                result = alt_result
                config = config_relaxed
                fallback_relaxed_turnover = True

    weights_out = result.weights
    cardinality_info: dict[str, Any] | None = None

    def _resolve_cardinality(cfg: Mapping[str, Any]) -> dict[str, Any] | None:
        section = cfg.get("cardinality")
        if isinstance(section, Mapping):
            resolved = dict(section)
        else:
            resolved = None
        if resolved is None:
            legacy_k = cfg.get("cardinality_k")
            legacy_min = cfg.get("cardinality_kmin")
            legacy_max = cfg.get("cardinality_kmax")
            if legacy_k is not None:
                resolved = {
                    "enable": True,
                    "mode": "fixed_k",
                    "k_fixed": int(legacy_k),
                }
            elif legacy_min is not None or legacy_max is not None:
                k_min = int(legacy_min) if legacy_min is not None else int(legacy_max)
                k_max = int(legacy_max) if legacy_max is not None else int(legacy_min)
                if k_min > k_max:
                    k_min, k_max = k_max, k_min
                resolved = {
                    "enable": True,
                    "mode": "dynamic_neff",
                    "k_min": k_min,
                    "k_max": k_max,
                }
        if resolved and "enable" not in resolved:
            resolved["enable"] = True
        return resolved

    # Apply cardinality constraint if enabled
    cardinality_config = _resolve_cardinality(optimizer_config) or {}
    if cardinality_config.get("enable", False):
        cost_config = optimizer_config.get("costs")
        weights_card, card_info = apply_cardinality_constraint(
            weights=result.weights,
            mu=mu,
            cov=cov,
            mv_config=config,
            cardinality_config=cardinality_config,
            cost_config=cost_config,
        )
        weights_out = weights_card
        cardinality_info = card_info
        turnover_card = card_info.get("turnover_after_cardinality")
        if turnover_card is None:
            turnover_card = float((weights_card - prev).abs().sum())
        cost_card = card_info.get("cost_after_cardinality", result.cost)
        expected_return_card = card_info.get(
            "reopt_expected_return", result.expected_return
        )
        variance_card = card_info.get("reopt_variance", result.variance)
        result = replace(
            result,
            weights=weights_card,
            turnover=turnover_card,
            cost=cost_card,
            expected_return=expected_return_card,
            variance=variance_card,
        )

    solver_flags = {
        "turnover_cap_relaxed": True if fallback_relaxed_turnover else None,
    }

    return weights_out, config, result, cardinality_info, solver_flags


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
    aligned_prev = previous_weights.reindex(target_weights.index).fillna(0.0)
    costs = transaction_cost_vector(
        target_weights,
        aligned_prev,
        linear_bps=linear_bps,
        notional=capital,
    )
    return float(costs.sum())


def build_rebalance_log(
    date: pd.Timestamp,
    metrics: RebalanceMetrics,
    solver_summary: Any,
    *,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a serialisable log structure."""

    payload: dict[str, Any] = {
        "date": pd.Timestamp(date).isoformat(),
        "metrics": metrics.to_dict(),
    }
    if solver_summary is not None:
        if isinstance(solver_summary, Mapping):
            payload["solver"] = dict(solver_summary)
        else:
            payload["solver"] = {
                "status": solver_summary.status,
                "value": solver_summary.value,
                "solver": solver_summary.solver,
                "runtime": solver_summary.runtime,
            }
    if extra:
        payload.update(
            {key: value for key, value in extra.items() if value is not None}
        )
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
    cost_cfg = dict(config.get("costs", {}) or {})
    estimator_cfg = config.get("estimators", {})
    risk_cfg: dict[str, Any] = {}

    if cost_cfg and "costs" not in optimizer_cfg:
        optimizer_cfg["costs"] = cost_cfg

    top_level_cardinality = config.get("cardinality")
    if (
        isinstance(top_level_cardinality, Mapping)
        and "cardinality" not in optimizer_cfg
    ):
        optimizer_cfg["cardinality"] = dict(top_level_cardinality)

    nested_risk = optimizer_cfg.pop("risk", None)
    if isinstance(nested_risk, Mapping):
        risk_cfg.update({k: v for k, v in nested_risk.items()})

    portfolio_risk = config.get("risk")
    if isinstance(portfolio_risk, Mapping):
        risk_cfg.update({k: v for k, v in portfolio_risk.items()})

    returns_window = int(config.get("returns_window", 252))
    prices_at_date, historical_returns = prepare_inputs(
        date, market_data, returns_window
    )

    mu, cov = run_estimators(
        historical_returns,
        mu_config=estimator_cfg.get("mu"),
        sigma_config=estimator_cfg.get("sigma"),
    )

    baseline_cfg_raw = config.get("baseline")
    allocation_method = "optimizer"
    solver_summary: Any | None = None
    solver_extra: dict[str, Any] = {}

    if baseline_cfg_raw:
        if isinstance(baseline_cfg_raw, Mapping):
            baseline_cfg = dict(baseline_cfg_raw)
        elif isinstance(baseline_cfg_raw, str):
            baseline_cfg = {"method": baseline_cfg_raw}
        else:
            raise TypeError("baseline configuration must be a mapping or string")
        method = baseline_cfg.pop("method", baseline_cfg.pop("name", None))
        if method is None:
            raise ValueError("baseline configuration must define a method")
        heuristic_data = {
            "assets": list(mu.index),
            "asset_index": list(mu.index),
            "covariance": cov,
            "returns": historical_returns,
        }
        heuristic_result = heuristic_allocation(
            heuristic_data,
            method=method,
            config=baseline_cfg,
        )
        target_weights = (
            heuristic_result.weights.reindex(mu.index).fillna(0.0).astype(float)
        )
        cov_aligned = cov.reindex(
            index=target_weights.index, columns=target_weights.index
        ).astype(float)
        mu_vector = mu.reindex(target_weights.index).to_numpy(dtype=float)
        weights_array = target_weights.to_numpy(dtype=float)
        optimizer_expected_return = float(mu_vector @ weights_array)
        optimizer_variance = float(
            weights_array @ cov_aligned.to_numpy(dtype=float) @ weights_array
        )
        optimizer_turnover = float(
            np.abs(
                target_weights
                - previous_weights.reindex(target_weights.index).fillna(0.0)
            ).sum()
        )
        optimizer_cost = compute_costs(
            target_weights, previous_weights, capital=capital, cost_config=cost_cfg
        )
        allocation_method = f"heuristic:{heuristic_result.method}"
        heuristic_payload = {"method": heuristic_result.method}
        if heuristic_result.diagnostics:
            heuristic_payload["diagnostics"] = heuristic_result.diagnostics
        solver_extra = {
            "allocation": allocation_method,
            "weights_pre_rounding": target_weights.to_dict(),
            "heuristic": heuristic_payload,
        }
    else:
        base_lambda = float(
            optimizer_cfg.get("risk_aversion", optimizer_cfg.get("lambda", 5.0))
        )
        adjusted_lambda = base_lambda
        regime_snapshot = None
        regime_multiplier_value: float | None = None

        regime_cfg = optimizer_cfg.get("regime_detection")
        if regime_cfg:
            regime_snapshot = detect_regime(historical_returns, config=regime_cfg)
            regime_multiplier_value = regime_multiplier(regime_snapshot, regime_cfg)
            adjusted_lambda = base_lambda * regime_multiplier_value
            optimizer_cfg = dict(optimizer_cfg)  # avoid mutating caller's config
            optimizer_cfg["risk_aversion"] = adjusted_lambda
            optimizer_cfg["lambda"] = adjusted_lambda
            optimizer_cfg.pop("regime_detection", None)

        (
            opt_weights,
            mv_config,
            mv_result,
            card_info,
            solver_flags,
        ) = optimize_portfolio(
            mu,
            cov,
            previous_weights,
            optimizer_cfg,
            risk_config=risk_cfg or None,
        )
        target_weights = opt_weights.reindex(mu.index, fill_value=0.0).astype(float)
        if mv_config.turnover_cap is not None and mv_config.turnover_cap > 0:
            from itau_quant.backtesting.risk_monitor import apply_turnover_cap
            adjusted, adjusted_turnover = apply_turnover_cap(
                previous_weights,
                target_weights,
                max_turnover=mv_config.turnover_cap,
            )
            if not adjusted.equals(target_weights):
                solver_extra["turnover_cap_scaled"] = True
                solver_extra["turnover_adjusted"] = adjusted_turnover
            target_weights = adjusted
        optimizer_cost = compute_costs(
            target_weights, previous_weights, capital=capital, cost_config=cost_cfg
        )
        optimizer_expected_return = mv_result.expected_return
        optimizer_variance = mv_result.variance
        optimizer_turnover = mv_result.turnover
        solver_summary = mv_result.summary
        solver_extra = {
            "allocation": allocation_method,
            "weights_pre_rounding": target_weights.to_dict(),
            "risk_aversion": mv_config.risk_aversion,
            "turnover_penalty": mv_config.turnover_penalty,
        }
        if regime_snapshot is not None:
            solver_extra["regime_state"] = regime_snapshot.to_dict()
            solver_extra["regime_multiplier"] = regime_multiplier_value
            solver_extra["lambda_base"] = base_lambda
            solver_extra["lambda_adjusted"] = adjusted_lambda
        if mv_config.budgets:
            solver_extra["budgets"] = [
                {
                    "name": budget.name,
                    "min_weight": budget.min_weight,
                    "max_weight": budget.max_weight,
                    "n_assets": len(budget.tickers),
                }
                for budget in mv_config.budgets
            ]
        if card_info:
            solver_extra["cardinality"] = card_info
        if solver_flags.get("turnover_cap_relaxed"):
            solver_extra["turnover_cap_relaxed"] = True

    rounding_result = apply_postprocessing(
        target_weights, prices_at_date, capital, rounding_cfg
    )

    rounded_weights = rounding_result.rounded_weights.reindex(
        target_weights.index, fill_value=0.0
    )
    trades = rounded_weights - previous_weights.reindex(rounded_weights.index).fillna(
        0.0
    )
    realized_turnover = float(trades.abs().sum())

    metrics = RebalanceMetrics(
        optimizer_expected_return=optimizer_expected_return,
        optimizer_variance=optimizer_variance,
        optimizer_turnover=optimizer_turnover,
        optimizer_cost=optimizer_cost,
        rounding_cost=rounding_result.rounding_cost,
        realized_turnover=realized_turnover,
    )

    log = build_rebalance_log(
        date,
        metrics,
        solver_summary,
        extra=solver_extra,
    )

    return RebalanceResult(
        date=pd.Timestamp(date),
        weights=target_weights.astype(float),
        rounded_weights=rounded_weights.astype(float),
        shares=rounding_result.shares.astype(float),
        cash=float(rounding_result.residual_cash),
        metrics=metrics,
        trades=trades.astype(float),
        log=log,
        rounding=rounding_result,
        allocator=allocation_method,
    )
