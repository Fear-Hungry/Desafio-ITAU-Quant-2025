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
    epsilon = cardinality_config.get("epsilon", 1e-4)

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
            from itau_quant.optimization.heuristics.cardinality import suggest_k_from_neff

            k = suggest_k_from_neff(neff, k_min, k_max, neff_multiplier)
            k_info = {"k_suggested": k, "k_from_neff": k, "neff": neff}
        else:
            k_info = suggest_k_dynamic(neff, costs_bps, k_min, k_max, neff_multiplier)
            k = k_info["k_suggested"]

    # Step 4: Select support
    selected = select_support_topk(
        weights=weights,
        k=k,
        weights_prev=mv_config.previous_weights if hasattr(mv_config, "previous_weights") else None,
        mu=mu,
        costs_bps=costs_bps,
        alpha_weight=alpha_weight,
        alpha_turnover=alpha_turnover,
        alpha_return=alpha_return,
        alpha_cost=alpha_cost,
        tie_breaker=tie_breaker,
        epsilon=epsilon,
    )

    # If no reduction (selected all significant assets), return original
    significant_count = (weights > epsilon).sum()
    if len(selected) >= significant_count:
        k_info["selected_assets"] = selected.tolist()
        k_info["reopt_status"] = "not_needed"
        k_info["note"] = f"No reduction needed: K={k} >= significant_count={significant_count}"
        return weights, k_info

    # Step 5: Reoptimize on reduced support
    mu_k = mu.loc[selected]
    cov_k = cov.loc[selected, selected]

    # Build new config for reoptimization
    prev_k = mv_config.previous_weights.reindex(selected, fill_value=0.0) if mv_config.previous_weights is not None else pd.Series(0.0, index=selected)

    lower_k = mv_config.lower_bounds.reindex(selected, fill_value=0.0) if mv_config.lower_bounds is not None else pd.Series(0.0, index=selected)
    upper_k = mv_config.upper_bounds.reindex(selected, fill_value=1.0) if mv_config.upper_bounds is not None else pd.Series(1.0, index=selected)

    config_k = MeanVarianceConfig(
        risk_aversion=mv_config.risk_aversion,
        turnover_penalty=mv_config.turnover_penalty if hasattr(mv_config, "turnover_penalty") else 0.0,
        turnover_cap=mv_config.turnover_cap if hasattr(mv_config, "turnover_cap") else None,
        lower_bounds=lower_k,
        upper_bounds=upper_k,
        previous_weights=prev_k,
        cost_vector=mv_config.cost_vector,
        solver=mv_config.solver,
        solver_kwargs=mv_config.solver_kwargs,
        risk_config=mv_config.risk_config if hasattr(mv_config, "risk_config") else None,
    )

    try:
        result = solve_mean_variance(mu_k, cov_k, config_k)
        k_info["selected_assets"] = selected.tolist()
        k_info["reopt_status"] = result.summary.get("status", "completed") if hasattr(result, "summary") else "completed"
        k_info["reopt_expected_return"] = result.expected_return
        k_info["reopt_variance"] = result.variance
        return result.weights, k_info
    except Exception as e:
        # Fallback: return unconstrained solution
        k_info["selected_assets"] = selected.tolist()
        k_info["reopt_status"] = "failed"
        k_info["reopt_error"] = str(e)
        return weights, k_info
