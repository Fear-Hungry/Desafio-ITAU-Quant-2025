"""Mean-CVaR optimisation (Rockafellar-Uryasev LP) with portfolio constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import cvxpy as cp
import numpy as np
import pandas as pd

from itau_quant.optimization.core.solver_utils import SolverSummary, solve_problem
from itau_quant.risk.cvar import build_cvar_lp, cvar_objective, historical_scenarios

__all__ = ["CvarConfig", "CvarResult", "solve_cvar_lp"]


@dataclass(frozen=True)
class CvarConfig:
    alpha: float
    risk_aversion: float
    long_only: bool = True
    lower_bounds: pd.Series | None = None
    upper_bounds: pd.Series | None = None
    turnover_penalty: float = 0.0
    turnover_cap: float | None = None
    previous_weights: pd.Series | None = None
    target_return: float | None = None
    max_cvar: float | None = None
    solver: str | None = None
    solver_kwargs: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class CvarResult:
    weights: pd.Series
    expected_return: float
    cvar: float
    var: float
    turnover: float
    summary: SolverSummary


def _align_series(
    series: pd.Series | None, index: pd.Index, fill_value: float
) -> pd.Series:
    if series is None:
        return pd.Series(fill_value, index=index, dtype=float)
    return series.reindex(index).fillna(fill_value).astype(float)


def solve_cvar_lp(
    returns: pd.DataFrame,
    expected_returns: pd.Series,
    config: CvarConfig,
) -> CvarResult:
    if returns.empty:
        raise ValueError("returns must not be empty")
    if expected_returns.empty:
        raise ValueError("expected_returns must not be empty")

    assets = expected_returns.index
    returns = returns.reindex(columns=assets).dropna(how="all")
    if returns.empty:
        raise ValueError("returns contain only NaNs after alignment")

    scenarios = historical_scenarios(returns)
    weights_var = cp.Variable(len(assets))

    var, aux, constraints = build_cvar_lp(scenarios, weights_var, config.alpha)
    cvar_expr = cvar_objective(var, aux, config.alpha)
    expected_expr = expected_returns.to_numpy(dtype=float) @ weights_var

    if config.previous_weights is not None:
        prev_series = config.previous_weights.reindex(assets).fillna(0.0).astype(float)
    else:
        prev_series = pd.Series(0.0, index=assets, dtype=float)
    prev_vector = prev_series.to_numpy(dtype=float)

    objective_terms: list[cp.Expression] = [
        expected_expr - config.risk_aversion * cvar_expr
    ]
    if config.turnover_penalty > 0:
        objective_terms.append(
            -config.turnover_penalty * cp.norm1(weights_var - prev_vector)
        )

    constraints.append(cp.sum(weights_var) == 1.0)
    if config.long_only:
        constraints.append(weights_var >= 0)

    lower = _align_series(config.lower_bounds, assets, 0.0)
    upper = _align_series(config.upper_bounds, assets, 1.0)
    constraints.extend(
        [weights_var >= lower.to_numpy(), weights_var <= upper.to_numpy()]
    )

    if config.target_return is not None:
        constraints.append(expected_expr >= float(config.target_return))
    if config.max_cvar is not None:
        constraints.append(cvar_expr <= float(config.max_cvar))
    if config.turnover_cap is not None:
        constraints.append(
            cp.norm1(weights_var - prev_vector) <= float(config.turnover_cap)
        )

    problem = cp.Problem(cp.Maximize(cp.sum(objective_terms)), constraints)
    summary = solve_problem(
        problem, solver=config.solver, solver_kwargs=config.solver_kwargs
    )

    solution = pd.Series(
        np.asarray(weights_var.value).ravel(), index=assets, dtype=float
    ).fillna(0.0)
    turnover = 0.5 * float(np.abs(solution - prev_series).sum())  # one-way turnover

    return CvarResult(
        weights=solution,
        expected_return=(
            float(expected_expr.value)
            if expected_expr.value is not None
            else float("nan")
        ),
        cvar=float(cvar_expr.value) if cvar_expr.value is not None else float("nan"),
        var=float(var.value) if var.value is not None else float("nan"),
        turnover=turnover,
        summary=summary,
    )
