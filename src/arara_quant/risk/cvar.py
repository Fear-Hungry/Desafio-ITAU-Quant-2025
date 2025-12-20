"""Helpers for CVaR formulations (Rockafellar-Uryasev).

This module is the single source of truth for the auxiliary-variable CVaR
construction reused by:
- ``arara_quant.optimization.constraints.cvar`` (constraint builder)
- ``arara_quant.optimization.core.cvar_lp`` (mean-CVaR solver)
- ``arara_quant.optimization.core.constraints_builder`` (composable constraints)
"""

from __future__ import annotations

from collections.abc import Mapping

import cvxpy as cp
import numpy as np
import pandas as pd

__all__ = [
    "build_cvar_lp",
    "cvar_objective",
    "add_cvar_constraint",
    "solve_cvar_portfolio",
    "historical_scenarios",
    "validate_cvar_inputs",
]


def validate_cvar_inputs(returns_matrix: np.ndarray, alpha: float) -> None:
    if returns_matrix.ndim != 2:
        raise ValueError("returns_matrix must be 2-dimensional")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1)")


def build_cvar_lp(
    returns_matrix: np.ndarray,
    weights: cp.Expression,
    alpha: float,
) -> tuple[cp.Variable, cp.Variable, list[cp.Constraint]]:
    """Build auxiliary variables (VaR, u) and constraints for CVaR."""

    validate_cvar_inputs(returns_matrix, alpha)
    n_scenarios = returns_matrix.shape[0]

    var = cp.Variable()
    u = cp.Variable(n_scenarios)
    scenario_returns = returns_matrix @ weights

    constraints = [u >= 0, u >= -scenario_returns - var]
    return var, u, constraints


def cvar_objective(var: cp.Variable, u: cp.Variable, alpha: float) -> cp.Expression:
    n = u.shape[0]
    return var + (1 / (1 - alpha)) * cp.sum(u) / n


def add_cvar_constraint(
    returns_matrix: np.ndarray,
    weights: cp.Expression,
    alpha: float,
    max_cvar: float,
) -> tuple[cp.Variable, cp.Variable, list[cp.Constraint]]:
    var, u, constraints = build_cvar_lp(returns_matrix, weights, alpha)
    cvar_expr = cvar_objective(var, u, alpha)
    constraints.append(cvar_expr <= float(max_cvar))
    return var, u, constraints


def solve_cvar_portfolio(
    returns_matrix: np.ndarray,
    expected_returns: np.ndarray,
    alpha: float,
    *,
    risk_aversion: float,
    solver: str | None = None,
    solver_kwargs: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Solve a mean-CVaR portfolio problem."""

    n_assets = returns_matrix.shape[1]
    weights = cp.Variable(n_assets)
    var, u, constraints = build_cvar_lp(returns_matrix, weights, alpha)
    cvar = cvar_objective(var, u, alpha)
    expected = expected_returns @ weights
    objective = cp.Maximize(expected - risk_aversion * cvar)

    constraints.extend([cp.sum(weights) == 1, weights >= 0])

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, **(solver_kwargs or {}))

    return {
        "weights": np.asarray(weights.value).ravel(),
        "cvar": float(cvar.value) if cvar.value is not None else np.nan,
        "expected_return": (
            float(expected.value) if expected.value is not None else np.nan
        ),
        "status": problem.status,
    }


def historical_scenarios(
    returns: pd.DataFrame, window: int | None = None
) -> np.ndarray:
    """Return matrix of historical scenarios (each row a scenario)."""

    data = returns.tail(window) if window else returns
    return data.to_numpy(dtype=float)
