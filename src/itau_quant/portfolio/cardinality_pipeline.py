"""Cardinality pipeline for portfolio rebalancing.

This module orchestrates the cardinality constraint logic:
1. Compute N_eff from unconstrained solution
2. Calibrate K dynamically from N_eff + costs
3. Score and select top-K assets
4. Reoptimize on reduced support
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from itau_quant.estimators import cov as covariance_estimators
from itau_quant.costs.cost_model import CostModel, estimate_costs_by_class
from itau_quant.optimization.core.mv_qp import MeanVarianceConfig, solve_mean_variance
from itau_quant.optimization.heuristics.cardinality import (
    compute_effective_number,
    select_support_topk,
    suggest_k_dynamic,
)

__all__ = ["apply_cardinality_constraint"]


def apply_cardinality_constraint(
    weights: pd.Series,
    mu: pd.Series,
    cov: pd.DataFrame,
    mv_config: MeanVarianceConfig,
    cardinality_config: dict[str, Any],
    cost_config: dict[str, Any] | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    """Apply cardinality constraint via post-processing + reoptimization.

    Args:
        weights: Unconstrained QP solution
        mu: Expected returns
        cov: Covariance matrix
        mv_config: Mean-variance config (used for reoptimization)
        cardinality_config: Cardinality configuration
        cost_config: Cost model configuration (optional)

    Returns:
        (weights_cardinality, info_dict) tuple

    The info dict contains:
        - neff: Effective number of unconstrained solution
        - k_suggested: Calibrated K
        - k_from_neff: K based purely on N_eff
        - k_range_from_cost: (min, max) from cost analysis
        - selected_assets: List of selected tickers
        - reopt_status: Status of reoptimization

    Examples:
        >>> weights = pd.Series({'A': 0.4, 'B': 0.3, 'C': 0.2, 'D': 0.1})
        >>> mu = pd.Series({'A': 0.10, 'B': 0.08, 'C': 0.12, 'D': 0.06})
        >>> cov = pd.DataFrame([[0.04, 0.01, 0.01, 0.01],
        ...                     [0.01, 0.04, 0.01, 0.01],
        ...                     [0.01, 0.01, 0.04, 0.01],
        ...                     [0.01, 0.01, 0.01, 0.04]], index=mu.index, columns=mu.index)
        >>> config = MeanVarianceConfig(risk_aversion=4.0)
        >>> card_config = {'enable': True, 'mode': 'fixed_k', 'k_fixed': 2}
        >>> w_card, info = apply_cardinality_constraint(weights, mu, cov, config, card_config)
        >>> len(w_card[w_card > 0])
        2
    """
    if not cardinality_config or not bool(cardinality_config.get("enable", True)):
        info = {
            "reopt_status": "disabled",
            "selected_assets": [],
            "note": "Cardinality constraint disabled via configuration",
        }
        return weights.copy(), info

    # Extract config
    mode = cardinality_config.get("mode", "dynamic_neff_cost")
    k_fixed = cardinality_config.get("k_fixed", 22)
    k_min = cardinality_config.get("k_min", 12)
    k_max = cardinality_config.get("k_max", 32)
    neff_multiplier = cardinality_config.get("neff_multiplier", 0.8)

    # Scoring parameters
    alpha_weight = cardinality_config.get("score_weight", 1.0)
    alpha_turnover = cardinality_config.get("score_turnover", -0.2)
    alpha_return = cardinality_config.get("score_return", 0.1)
    alpha_cost = cardinality_config.get("score_cost", -0.15)
    tie_breaker = cardinality_config.get("tie_breaker", "low_turnover")
    epsilon = float(cardinality_config.get("epsilon", 1e-4))
    min_active_weight = float(cardinality_config.get("min_active_weight", epsilon))

    universe = weights.index

    # Step 1: Compute N_eff
    neff = compute_effective_number(weights)

    # Step 2: Estimate costs
    costs_bps = None
    if cost_config:
        cost_model = CostModel.from_config(cost_config)
        costs_bps = estimate_costs_by_class(weights.index, cost_model)

    # Step 3: Determine K
    if mode == "fixed_k":
        k = k_fixed
        k_info = {"k_suggested": k, "neff": neff}
    elif mode == "dynamic_neff":
        from itau_quant.optimization.heuristics.cardinality import suggest_k_from_neff

        k = suggest_k_from_neff(neff, k_min, k_max, neff_multiplier)
        k_info = {"k_suggested": k, "k_from_neff": k, "neff": neff}
    else:  # dynamic_neff_cost
        if costs_bps is None:
            # Fallback to N_eff only
            from itau_quant.optimization.heuristics.cardinality import (
                suggest_k_from_neff,
            )

            k = suggest_k_from_neff(neff, k_min, k_max, neff_multiplier)
            k_info = {"k_suggested": k, "k_from_neff": k, "neff": neff}
        else:
            k_info = suggest_k_dynamic(neff, costs_bps, k_min, k_max, neff_multiplier)
            k = k_info["k_suggested"]

    # Step 4: Select support
    significance_threshold = min_active_weight
    significant_count = (weights > significance_threshold).sum()
    if significant_count < k:
        significance_threshold = epsilon
        significant_count = (weights > significance_threshold).sum()

    selected = select_support_topk(
        weights=weights,
        k=k,
        weights_prev=mv_config.previous_weights
        if hasattr(mv_config, "previous_weights")
        else None,
        mu=mu,
        costs_bps=costs_bps,
        alpha_weight=alpha_weight,
        alpha_turnover=alpha_turnover,
        alpha_return=alpha_return,
        alpha_cost=alpha_cost,
        tie_breaker=tie_breaker,
        epsilon=significance_threshold,
    )

    selected = pd.Index(selected)
    if selected.empty and k > 0:
        fallback_rank = weights.abs().sort_values(ascending=False)
        selected = pd.Index(fallback_rank.index[:k])

    # If no reduction (selected all significant assets), return original
    if len(selected) >= significant_count:
        k_info["selected_assets"] = selected.tolist()
        k_info["reopt_status"] = "not_needed"
        k_info["note"] = (
            f"No reduction needed: K={k} >= significant_count={significant_count}"
        )
        return weights, k_info

    # Step 5: Reoptimize on reduced support
    mu_k = mu.loc[selected]
    cov_k = cov.loc[selected, selected]
    cov_k = covariance_estimators.project_to_psd(cov_k, epsilon=1e-9)

    # Build new config for reoptimization
    prev_full = getattr(mv_config, "previous_weights", None)
    if prev_full is None:
        prev_full = pd.Series(0.0, index=universe, dtype=float)
    else:
        prev_full = prev_full.reindex(universe, fill_value=0.0).astype(float)
    prev_k = prev_full.reindex(selected, fill_value=0.0)

    lower_full = getattr(mv_config, "lower_bounds", None)
    if lower_full is None:
        lower_full = pd.Series(0.0, index=universe, dtype=float)
    else:
        lower_full = lower_full.reindex(universe, fill_value=0.0).astype(float)
    lower_k = lower_full.reindex(selected, fill_value=0.0)

    upper_full = getattr(mv_config, "upper_bounds", None)
    if upper_full is None:
        upper_full = pd.Series(1.0, index=universe, dtype=float)
    else:
        upper_full = upper_full.reindex(universe, fill_value=1.0).astype(float)
    upper_k = upper_full.reindex(selected, fill_value=1.0)

    config_k = MeanVarianceConfig(
        risk_aversion=mv_config.risk_aversion,
        turnover_penalty=mv_config.turnover_penalty
        if hasattr(mv_config, "turnover_penalty")
        else 0.0,
        turnover_cap=None,
        lower_bounds=lower_k,
        upper_bounds=upper_k,
        previous_weights=prev_k,
        cost_vector=mv_config.cost_vector,
        solver=mv_config.solver,
        solver_kwargs=mv_config.solver_kwargs,
        risk_config=mv_config.risk_config
        if hasattr(mv_config, "risk_config")
        else None,
        budgets=mv_config.budgets if hasattr(mv_config, "budgets") else None,
    )

    try:
        result = solve_mean_variance(mu_k, cov_k, config_k)
        k_info["selected_assets"] = selected.tolist()
        k_info["reopt_status"] = (
            result.summary.status if hasattr(result.summary, "status") else "completed"
        )
        if getattr(mv_config, "turnover_cap", None) is not None:
            k_info["turnover_cap_relaxed"] = True
        k_info["reopt_expected_return"] = float(result.expected_return)
        k_info["reopt_variance"] = float(result.variance)
        reopt_weights = result.weights.reindex(selected).fillna(0.0).astype(float)
        full_weights = pd.Series(0.0, index=universe, dtype=float)
        full_weights.loc[selected] = reopt_weights
        total = float(full_weights.sum())
        if total > 0:
            full_weights /= total
        k_info["final_support"] = full_weights[
            full_weights > significance_threshold
        ].index.tolist()
        trades_full = full_weights - prev_full
        k_info["turnover_after_cardinality"] = float(trades_full.abs().sum())
        if mv_config.cost_vector is not None:
            costs_vector = (
                mv_config.cost_vector.reindex(universe).fillna(0.0).astype(float)
            )
            k_info["cost_after_cardinality"] = float(
                (costs_vector.abs() * trades_full.abs()).sum()
            )
        return full_weights, k_info
    except Exception as e:
        # Fallback: return unconstrained solution
        k_info["selected_assets"] = selected.tolist()
        k_info["reopt_status"] = "failed"
        k_info["reopt_error"] = str(e)
        return weights, k_info
