"""Maximizar Sharpe via SOCP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import cvxpy as cp
import numpy as np
import pandas as pd

from .solver_utils import SolverSummary, solve_problem

__all__ = [
    "SharpeSocpResult",
    "build_sharpe_socp",
    "add_cost_terms",
    "solve_sharpe_socp",
    "normalize_weights",
    "sharpe_socp",
]


@dataclass(frozen=True)
class SharpeSocpResult:
    weights: pd.Series
    sharpe: float
    expected_return: float
    volatility: float
    summary: SolverSummary


def _as_series(values: Sequence[float] | pd.Series, index: Sequence[str]) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.reindex(index).astype(float)
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size != len(index):
        raise ValueError("values must be 1-dimensional and match asset index length")
    return pd.Series(array, index=index, dtype=float)


def _as_dataframe(
    matrix: pd.DataFrame | np.ndarray, index: Sequence[str]
) -> pd.DataFrame:
    if isinstance(matrix, pd.DataFrame):
        return matrix.reindex(index=index, columns=index).astype(float)
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("covariance matrix must be square")
    if array.shape[0] != len(index):
        raise ValueError("covariance must match asset index length")
    return pd.DataFrame(array, index=index, columns=index, dtype=float)


def _matrix_sqrt(matrix: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, epsilon)
    root = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    return root


def build_sharpe_socp(
    mu: Sequence[float] | pd.Series,
    cov: pd.DataFrame | np.ndarray,
) -> tuple[cp.Expression, list[cp.Constraint], cp.Variable, Sequence[str]]:
    """Return objective expression, base constraints and weight variable."""

    if isinstance(mu, pd.Series):
        assets = list(mu.index)
    elif isinstance(cov, pd.DataFrame):
        assets = list(cov.index)
    else:
        mu_array = np.asarray(mu, dtype=float)
        assets = list(range(mu_array.size))

    mu_series = _as_series(mu, assets)
    cov_df = _as_dataframe(cov, assets)
    cov_matrix = cov_df.to_numpy(dtype=float)
    cov_sqrt = _matrix_sqrt(cov_matrix)

    weights = cp.Variable(len(assets))
    t = cp.Variable()

    constraints = [
        mu_series.to_numpy(dtype=float) @ weights >= t,
        cp.norm(cov_sqrt @ weights, 2) <= 1.0,
    ]
    objective_expr = t
    return objective_expr, constraints, weights, assets


def add_cost_terms(
    objective_expr: cp.Expression,
    weights: cp.Variable,
    previous_weights: Sequence[float] | pd.Series | None,
    costs: Mapping[str, float] | None,
    *,
    asset_index: Sequence[str] | None = None,
) -> cp.Expression:
    """Subtract cost penalties from the objective expression."""

    if not costs:
        return objective_expr

    if previous_weights is None:
        prev = np.zeros(weights.shape[0], dtype=float)
    else:
        if asset_index is None:
            index = list(range(weights.shape[0]))
        else:
            index = asset_index
        prev = _as_series(previous_weights, index).to_numpy(dtype=float)

    delta = weights - prev
    penalty = 0.0

    linear = (
        costs.get("linear") or costs.get("linear_penalty") or costs.get("linear_bps")
    )
    if linear:
        penalty += float(linear) * cp.norm1(delta)

    quadratic = costs.get("quadratic")
    if quadratic:
        penalty += float(quadratic) * cp.sum_squares(delta)

    return objective_expr - penalty


def normalize_weights(
    weights: Sequence[float] | pd.Series, *, index: Sequence[str] | None = None
) -> pd.Series:
    """Ensure weights sum to unity and replace NaNs with zero."""

    if isinstance(weights, pd.Series):
        series = weights.astype(float).fillna(0.0)
    else:
        array = np.asarray(weights, dtype=float)
        if index is None:
            index = list(range(array.size))
        series = pd.Series(array, index=index, dtype=float).fillna(0.0)
    series[np.abs(series) < 1e-12] = 0.0
    total = float(series.sum())
    if abs(total) > 1e-12:
        series = series / total
    return series.astype(float)


def solve_sharpe_socp(
    mu: Sequence[float] | pd.Series,
    cov: pd.DataFrame | np.ndarray,
    *,
    config: Mapping[str, object] | None = None,
) -> SharpeSocpResult:
    """Solve the Sharpe maximisation problem using SOCP."""

    config = dict(config or {})
    rf = float(config.get("risk_free", 0.0))
    ridge = float(config.get("ridge", 1e-8))

    if isinstance(mu, pd.Series):
        assets = list(mu.index)
    elif isinstance(cov, pd.DataFrame):
        assets = list(cov.index)
    else:
        assets = list(range(np.asarray(mu).size))

    mu_series = _as_series(mu, assets) - rf
    cov_df = _as_dataframe(cov, assets)
    cov_df = cov_df + np.eye(len(assets)) * ridge

    objective_expr, constraints, weights, _ = build_sharpe_socp(mu_series, cov_df)

    budget = config.get("budget")
    if budget is not None:
        constraints.append(cp.sum(weights) == float(budget))

    bounds = config.get("bounds")
    if bounds is not None:
        lower, upper = bounds
        constraints.append(weights >= float(lower))
        constraints.append(weights <= float(upper))

    leverage = config.get("leverage")
    if leverage is not None:
        constraints.append(cp.norm1(weights) <= float(leverage))

    previous_weights = config.get("previous_weights")
    turnover_limit = config.get("turnover_limit")
    if turnover_limit is not None and previous_weights is not None:
        prev_array = _as_series(previous_weights, assets).to_numpy(dtype=float)
        constraints.append(cp.norm1(weights - prev_array) <= float(turnover_limit))

    costs = config.get("costs")
    objective_expr = add_cost_terms(
        objective_expr, weights, previous_weights, costs, asset_index=assets
    )

    problem = cp.Problem(cp.Maximize(objective_expr), constraints)
    summary = solve_problem(
        problem,
        solver=config.get("solver"),
        solver_kwargs=config.get("solver_kwargs"),
    )

    if weights.value is None:
        raw_weights = pd.Series(np.zeros(len(assets)), index=assets, dtype=float)
    else:
        raw_weights = pd.Series(
            np.asarray(weights.value).ravel(), index=assets, dtype=float
        )

    weights_series = normalize_weights(raw_weights)
    mu_vec = mu_series.to_numpy(dtype=float)
    cov_matrix = cov_df.to_numpy(dtype=float)
    expected_return = float(mu_vec @ weights_series.to_numpy(dtype=float))
    variance = float(
        weights_series.to_numpy(dtype=float)
        @ cov_matrix
        @ weights_series.to_numpy(dtype=float)
    )
    volatility = float(np.sqrt(max(variance, 0.0)))
    sharpe = expected_return / volatility if volatility > 0 else float("nan")

    return SharpeSocpResult(
        weights=weights_series,
        sharpe=sharpe,
        expected_return=expected_return,
        volatility=volatility,
        summary=summary,
    )


def sharpe_socp(
    mu: Sequence[float] | pd.Series,
    cov: pd.DataFrame | np.ndarray,
    *,
    config: Mapping[str, object] | None = None,
) -> SharpeSocpResult:
    """Convenience wrapper around :func:`solve_sharpe_socp`."""

    return solve_sharpe_socp(mu, cov, config=config)
