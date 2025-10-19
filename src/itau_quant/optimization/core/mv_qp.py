"""Programa quadrático média-variância com turnover e custos opcionais."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import cvxpy as cp
import numpy as np
import pandas as pd

from .solver_utils import SolverSummary, solve_problem
from itau_quant.risk.constraints import build_constraints

__all__ = [
    "MeanVarianceConfig",
    "MeanVarianceResult",
    "solve_mean_variance",
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
    solver: str | None = None
    solver_kwargs: Mapping[str, Any] | None = None
    risk_config: Mapping[str, Any] | None = None
    factor_loadings: pd.DataFrame | None = None


@dataclass(frozen=True)
class MeanVarianceResult:
    weights: pd.Series
    objective_value: float
    expected_return: float
    variance: float
    turnover: float
    cost: float
    summary: SolverSummary


def solve_mean_variance(
    mu: pd.Series,
    cov: pd.DataFrame,
    config: MeanVarianceConfig,
) -> MeanVarianceResult:
    """Solve a convex mean-variance allocation problem."""

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
    objective_terms = [mu_vec @ w - config.risk_aversion * cp.quad_form(w, cov_matrix)]

    trades = w - prev.to_numpy(dtype=float)

    if config.turnover_penalty > 0:
        objective_terms.append(-config.turnover_penalty * cp.norm1(trades))

    if config.cost_vector is not None:
        cost_vec = config.cost_vector.reindex(assets).fillna(0.0).to_numpy(dtype=float)
        objective_terms.append(-cp.sum(cp.multiply(cost_vec, cp.abs(trades))))

    objective = cp.Maximize(cp.sum(objective_terms))

    constraints = [cp.sum(w) == 1.0]
    constraints.append(w >= lower.to_numpy(dtype=float))
    constraints.append(w <= upper.to_numpy(dtype=float))

    if config.turnover_cap is not None and config.turnover_cap > 0:
        constraints.append(cp.norm1(trades) <= float(config.turnover_cap))

    if config.risk_config:
        context: dict[str, Any] = {
            "weights_var": w,
            "asset_index": assets,
            "covariance": cov_matrix,
            "previous_weights": prev.to_numpy(dtype=float),
        }
        if config.factor_loadings is not None:
            factor_df = config.factor_loadings.reindex(index=assets).astype(float)
            context["factor_loadings"] = factor_df
        constraints.extend(build_constraints(config.risk_config, context))

    problem = cp.Problem(objective, constraints)
    summary = solve_problem(problem, solver=config.solver, solver_kwargs=config.solver_kwargs)

    weights = pd.Series(np.asarray(w.value).ravel(), index=assets, dtype=float)
    weights = weights.fillna(0.0)
    weights /= weights.sum() if weights.sum() != 0 else 1.0

    expected_return = float(mu.reindex(assets).to_numpy(dtype=float) @ weights.to_numpy())
    variance = float(weights.to_numpy() @ cov_matrix @ weights.to_numpy())
    turnover = float(np.abs(weights - prev).sum())
    cost = 0.0
    if config.cost_vector is not None:
        cost = float(
            np.abs(weights - prev)
            .reindex(assets)
            .fillna(0.0)
            .to_numpy()
            @ config.cost_vector.reindex(assets).fillna(0.0).to_numpy()
        )

    return MeanVarianceResult(
        weights=weights,
        objective_value=float(problem.value) if problem.value is not None else float("nan"),
        expected_return=expected_return,
        variance=variance,
        turnover=turnover,
        cost=cost,
        summary=summary,
    )
