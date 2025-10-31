"""Programa quadrático média-variância com turnover e custos opcionais."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import cvxpy as cp
import numpy as np
import pandas as pd

from .solver_utils import SolverSummary, solve_problem
from itau_quant.risk.constraints import build_constraints
from itau_quant.risk.budgets import (
    RiskBudget,
    budgets_to_constraints,
    load_budgets,
    validate_budgets,
)

__all__ = [
    "MeanVarianceConfig",
    "MeanVarianceResult",
    "solve_mean_variance",
    "calibrate_lambda_for_target_vol",
]


@dataclass(frozen=True)
class MeanVarianceConfig:
    risk_aversion: float
    turnover_penalty: float
    turnover_cap: float | None
    lower_bounds: pd.Series
    upper_bounds: pd.Series
    previous_weights: pd.Series
    cost_vector: pd.Series | None
    budgets: Sequence[RiskBudget] | None = None
    solver: str | None = None
    solver_kwargs: Mapping[str, Any] | None = None
    risk_config: Mapping[str, Any] | None = None
    factor_loadings: pd.DataFrame | None = None
    ridge_penalty: float = 0.0  # γ for L2 weight regularization
    target_vol: float | None = None  # Target volatility (auto-calibrate λ)


@dataclass(frozen=True)
class MeanVarianceResult:
    weights: pd.Series
    objective_value: float
    expected_return: float
    variance: float
    turnover: float
    cost: float
    summary: SolverSummary


def calibrate_lambda_for_target_vol(
    mu: pd.Series,
    cov: pd.DataFrame,
    config: MeanVarianceConfig,
    target_vol: float,
    *,
    lam_lo: float = 1e-4,
    lam_hi: float = 1e3,
    tol: float = 1e-4,
    max_iter: int = 40,
) -> tuple[MeanVarianceResult, float]:
    """Calibrate risk_aversion λ to hit target volatility via bisection.

    Parameters
    ----------
    mu, cov, config : as in solve_mean_variance
    target_vol : float
        Target annualized volatility
    lam_lo, lam_hi : float
        Search bounds for λ
    tol : float
        Tolerance for vol matching
    max_iter : int
        Maximum iterations

    Returns
    -------
    result : MeanVarianceResult
        Optimized portfolio at calibrated λ
    lambda_calibrated : float
        The λ that achieves target_vol
    """
    for _ in range(max_iter):
        lam_mid = np.sqrt(lam_lo * lam_hi)

        # Create temporary config with this λ
        config_temp = MeanVarianceConfig(
            risk_aversion=lam_mid,
            turnover_penalty=config.turnover_penalty,
            turnover_cap=config.turnover_cap,
            lower_bounds=config.lower_bounds,
            upper_bounds=config.upper_bounds,
            previous_weights=config.previous_weights,
            cost_vector=config.cost_vector,
            budgets=config.budgets,
            solver=config.solver,
            solver_kwargs=config.solver_kwargs,
            risk_config=config.risk_config,
            factor_loadings=config.factor_loadings,
            ridge_penalty=config.ridge_penalty,
            target_vol=None,  # Disable recursion
        )

        result = solve_mean_variance(mu, cov, config_temp)

        if not result.summary.is_optimal():
            raise RuntimeError(
                f"Calibration failed at λ={lam_mid:.4f}, status={result.summary.status}"
            )

        vol = np.sqrt(result.variance)

        if abs(vol - target_vol) < tol:
            return result, lam_mid

        # Bisection logic
        if vol > target_vol:
            lam_lo = lam_mid  # Increase risk aversion
        else:
            lam_hi = lam_mid  # Decrease risk aversion

    # Return best attempt after max_iter
    return result, lam_mid


def solve_mean_variance(
    mu: pd.Series,
    cov: pd.DataFrame,
    config: MeanVarianceConfig,
) -> MeanVarianceResult:
    """Solve a convex mean-variance allocation problem.

    If config.target_vol is specified, automatically calibrates risk_aversion
    to achieve the target volatility via bisection search.
    """

    # Auto-calibrate λ if target_vol specified
    if config.target_vol is not None:
        result, lambda_cal = calibrate_lambda_for_target_vol(
            mu, cov, config, config.target_vol
        )
        return result

    if mu.empty:
        raise ValueError("mu must not be empty")
    if cov.shape[0] != cov.shape[1]:
        raise ValueError("cov must be square")
    if not (mu.index == cov.index).all():
        cov = cov.reindex(index=mu.index, columns=mu.index)

    assets = list(mu.index)
    n_assets = len(assets)

    lower = config.lower_bounds.reindex(assets).astype(float)
    upper = config.upper_bounds.reindex(assets).astype(float)
    prev = config.previous_weights.reindex(assets).astype(float)

    mu_vec = mu.reindex(assets).to_numpy(dtype=float)
    cov_matrix = cov.reindex(index=assets, columns=assets).to_numpy(dtype=float)

    w = cp.Variable(n_assets)
    cov_psd = cp.psd_wrap(cov_matrix)
    objective_terms = [mu_vec @ w - config.risk_aversion * cp.quad_form(w, cov_psd)]

    trades = w - prev.to_numpy(dtype=float)
    total_turnover = cp.norm1(trades)

    if config.turnover_penalty > 0:
        objective_terms.append(-config.turnover_penalty * total_turnover)

    if config.cost_vector is not None:
        cost_vec = config.cost_vector.reindex(assets).fillna(0.0).to_numpy(dtype=float)
        objective_terms.append(-cp.sum(cp.multiply(cost_vec, cp.abs(trades))))

    # Ridge penalty (L2 regularization on weights)
    if config.ridge_penalty > 0:
        objective_terms.append(-config.ridge_penalty * cp.sum_squares(w))

    objective = cp.Maximize(cp.sum(objective_terms))

    constraints = [cp.sum(w) == 1.0]
    constraints.append(w >= lower.to_numpy(dtype=float))
    constraints.append(w <= upper.to_numpy(dtype=float))

    # Budget constraints from MeanVarianceConfig and/or risk_config
    budgets: list[RiskBudget] = []
    if config.budgets:
        budgets.extend(config.budgets)

    risk_config_for_builder: dict[str, Any] | None = None
    if config.risk_config:
        risk_config_for_builder = dict(config.risk_config)
        raw_budgets = risk_config_for_builder.pop("budgets", None)
        if raw_budgets:
            if isinstance(raw_budgets, Mapping):
                raw_items = [raw_budgets]
            else:
                raw_items = list(raw_budgets)
            direct: list[RiskBudget] = []
            loadable: list[Mapping[str, object]] = []
            for item in raw_items:
                if isinstance(item, RiskBudget):
                    direct.append(item)
                elif isinstance(item, Mapping):
                    loadable.append(item)
                else:
                    raise TypeError(
                        "Budget entries must be RiskBudget instances or mappings."
                    )
            budgets.extend(direct)
            if loadable:
                budgets.extend(load_budgets(loadable))

    if budgets:
        unique: dict[tuple[Any, ...], RiskBudget] = {}
        for budget in budgets:
            key = (
                budget.name,
                tuple(sorted(budget.tickers)),
                budget.min_weight,
                budget.max_weight,
                budget.target,
                budget.tolerance,
            )
            unique[key] = budget
        budgets = list(unique.values())
        validate_budgets(budgets, assets)
        budget_cons = budgets_to_constraints(w, budgets, assets)
        constraints.extend(budget_cons)

    slack_var: cp.Variable | None = None
    if config.turnover_cap is not None and config.turnover_cap > 0:
        slack_var = cp.Variable(nonneg=True, name="turnover_slack")
        constraints.append(
            total_turnover <= float(config.turnover_cap) + slack_var
        )
        base_weight = max(config.risk_aversion, 1.0) * 100.0
        penalty_weight = max(config.turnover_penalty, base_weight)
        objective_terms.append(-penalty_weight * slack_var)

    if risk_config_for_builder:
        context: dict[str, Any] = {
            "weights_var": w,
            "asset_index": assets,
            "covariance": cov_matrix,
            "previous_weights": prev.to_numpy(dtype=float),
        }
        if config.factor_loadings is not None:
            factor_df = config.factor_loadings.reindex(index=assets).astype(float)
            context["factor_loadings"] = factor_df
        constraints.extend(build_constraints(risk_config_for_builder, context))

    problem = cp.Problem(objective, constraints)

    # Use CLARABEL by default (ECOS not installed) with strict tolerances
    solver_to_use = config.solver or "CLARABEL"

    # Solver-specific parameters
    if solver_to_use.upper() == "CLARABEL":
        solver_kwargs_final = {
            "tol_gap_abs": 1e-8,
            "tol_gap_rel": 1e-8,
            "tol_feas": 1e-8,
            "max_iter": 20000,
        }
    elif solver_to_use.upper() == "OSQP":
        solver_kwargs_final = {
            "eps_abs": 1e-8,
            "eps_rel": 1e-8,
            "max_iter": 20000,
        }
    else:  # ECOS or others
        solver_kwargs_final = {
            "abstol": 1e-8,
            "reltol": 1e-8,
            "feastol": 1e-8,
            "max_iters": 20000,
        }

    if config.solver_kwargs:
        solver_kwargs_final.update(config.solver_kwargs)

    summary = solve_problem(
        problem, solver=solver_to_use, solver_kwargs=solver_kwargs_final
    )

    raw_weights = w.value
    if raw_weights is None:
        raw_weights = prev.to_numpy(dtype=float)
    else:
        raw_weights = np.asarray(raw_weights).ravel()
    weights = pd.Series(raw_weights, index=assets, dtype=float)
    weights = weights.fillna(0.0)
    weights /= weights.sum() if weights.sum() != 0 else 1.0

    # Validate budget constraints if provided
    if budgets and summary.is_optimal():
        import warnings

        tol = 1e-4

        for budget in budgets:
            actual = sum(
                weights.get(t, 0.0) for t in budget.tickers if t in weights.index
            )

            if budget.max_weight is not None:
                if actual > budget.max_weight + tol:
                    warnings.warn(
                        f"Budget '{budget.name}' violates max: "
                        f"{actual:.4f} > {budget.max_weight:.4f}",
                        UserWarning,
                    )

            if budget.min_weight is not None:
                if actual < budget.min_weight - tol:
                    warnings.warn(
                        f"Budget '{budget.name}' violates min: "
                        f"{actual:.4f} < {budget.min_weight:.4f}",
                        UserWarning,
                    )

    expected_return = float(
        mu.reindex(assets).to_numpy(dtype=float) @ weights.to_numpy()
    )
    variance = float(weights.to_numpy() @ cov_matrix @ weights.to_numpy())
    turnover = float(np.abs(weights - prev).sum())
    cost = 0.0
    if config.cost_vector is not None:
        cost = float(
            (weights - prev).abs().reindex(assets).fillna(0.0).to_numpy()
            @ config.cost_vector.reindex(assets).fillna(0.0).to_numpy()
        )

    return MeanVarianceResult(
        weights=weights,
        objective_value=float(problem.value)
        if problem.value is not None
        else float("nan"),
        expected_return=expected_return,
        variance=variance,
        turnover=turnover,
        cost=cost,
        summary=summary,
    )
