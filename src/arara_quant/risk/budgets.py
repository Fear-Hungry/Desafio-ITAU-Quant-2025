"""Risk budget helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Sequence

import cvxpy as cp
import numpy as np
import pandas as pd

__all__ = [
    "RiskBudget",
    "load_budgets",
    "validate_budgets",
    "budgets_to_constraints",
    "budget_slack",
    "aggregate_by_budget",
    "BudgetViolation",
    "BudgetViolationError",
    "find_budget_violations",
    "project_weights_to_budget_feasible",
]


@dataclass(frozen=True)
class RiskBudget:
    """Container describing min/max allocation for a group of assets."""

    name: str
    tickers: Sequence[str]
    min_weight: float | None = None
    max_weight: float | None = None
    target: float | None = None
    tolerance: float | None = None

    def __post_init__(self) -> None:
        if not self.tickers:
            raise ValueError("RiskBudget must define at least one ticker.")
        if self.min_weight is not None and self.max_weight is not None:
            if float(self.max_weight) < float(self.min_weight):
                raise ValueError("max_weight cannot be smaller than min_weight.")


def load_budgets(config: Iterable[Mapping[str, object]]) -> list[RiskBudget]:
    """Instantiate budgets from iterable of dictionaries."""

    budgets: list[RiskBudget] = []
    for entry in config:
        budgets.append(
            RiskBudget(
                name=str(entry["name"]),
                tickers=list(entry.get("tickers", [])),
                min_weight=entry.get("min_weight"),
                max_weight=entry.get("max_weight"),
                target=entry.get("target"),
                tolerance=entry.get("tolerance"),
            )
        )
    return budgets


def validate_budgets(budgets: Iterable[RiskBudget], universe: Sequence[str]) -> None:
    """Ensure budgets reference valid tickers and have sensible limits."""

    universe_set = set(universe)
    for budget in budgets:
        missing = [ticker for ticker in budget.tickers if ticker not in universe_set]
        if missing:
            raise ValueError(
                f"Budget '{budget.name}' references unknown tickers: {missing}"
            )
        if budget.max_weight is not None and budget.max_weight > 1.0 + 1e-6:
            raise ValueError(f"Budget '{budget.name}' has max_weight greater than 1.")


def budgets_to_constraints(
    weights_var: cp.Variable,
    budgets: Iterable[RiskBudget],
    asset_index: Sequence[str],
) -> list[cp.Constraint]:
    """Convert budgets into CVXPy constraints for optimisation problems."""

    index_map = {asset: i for i, asset in enumerate(asset_index)}
    constraints: list[cp.Constraint] = []

    for budget in budgets:
        indices = [
            index_map[ticker] for ticker in budget.tickers if ticker in index_map
        ]
        if not indices:
            continue
        subset_sum = cp.sum(weights_var[indices])
        if budget.min_weight is not None:
            constraints.append(subset_sum >= float(budget.min_weight))
        if budget.max_weight is not None:
            constraints.append(subset_sum <= float(budget.max_weight))

    return constraints


def budget_slack(weights: pd.Series, budgets: Iterable[RiskBudget]) -> pd.Series:
    """Compute slack to the max-weight limit for each budget."""

    slack: dict[str, float] = {}
    for budget in budgets:
        group_weight = weights.reindex(budget.tickers).fillna(0.0).sum()
        if budget.max_weight is None:
            slack_val = np.inf
        else:
            slack_val = float(budget.max_weight) - group_weight
        slack[budget.name] = slack_val
    return pd.Series(slack)


def aggregate_by_budget(
    weights: pd.Series,
    returns: pd.Series | pd.DataFrame,
    budgets: Iterable[RiskBudget],
) -> pd.DataFrame:
    """Aggregate portfolio statistics per budget."""

    if isinstance(returns, pd.DataFrame):
        mean_returns = returns.mean()
    else:
        mean_returns = returns

    records: list[dict[str, float]] = []
    for budget in budgets:
        tickers = list(budget.tickers)
        w = weights.reindex(tickers).fillna(0.0)
        agg_weight = w.sum()
        expected_return = (w * mean_returns.reindex(tickers).fillna(0.0)).sum()
        records.append(
            {
                "budget": budget.name,
                "weight": float(agg_weight),
                "expected_return": float(expected_return),
            }
        )
    return pd.DataFrame(records)


@dataclass(frozen=True)
class BudgetViolation:
    """Simple container describing a budget violation."""

    name: str
    bound: Literal["min", "max"]
    limit: float
    exposure: float
    tolerance: float

    def as_dict(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "bound": self.bound,
            "limit": float(self.limit),
            "exposure": float(self.exposure),
            "tolerance": float(self.tolerance),
        }


class BudgetViolationError(RuntimeError):
    """Raised when portfolio weights cannot satisfy the configured budgets."""

    def __init__(
        self,
        message: str,
        violations: Sequence[BudgetViolation] | None = None,
    ) -> None:
        super().__init__(message)
        self.violations = list(violations or [])


def _budget_exposure(weights: pd.Series, budget: RiskBudget) -> float:
    return float(weights.reindex(budget.tickers).fillna(0.0).sum())


def find_budget_violations(
    weights: pd.Series,
    budgets: Iterable[RiskBudget],
    *,
    tol: float = 1e-6,
) -> list[BudgetViolation]:
    """Return all budget violations for the provided weight vector."""

    violations: list[BudgetViolation] = []
    for budget in budgets:
        exposure = _budget_exposure(weights, budget)
        if budget.max_weight is not None and exposure > float(budget.max_weight) + tol:
            violations.append(
                BudgetViolation(
                    name=budget.name,
                    bound="max",
                    limit=float(budget.max_weight),
                    exposure=exposure,
                    tolerance=tol,
                )
            )
        if budget.min_weight is not None and exposure < float(budget.min_weight) - tol:
            violations.append(
                BudgetViolation(
                    name=budget.name,
                    bound="min",
                    limit=float(budget.min_weight),
                    exposure=exposure,
                    tolerance=tol,
                )
            )
    return violations


def _resolve_bound_array(
    bounds: pd.Series | Sequence[float] | float | None,
    asset_index: Sequence[str],
    *,
    default: float | None,
) -> np.ndarray | None:
    if bounds is None:
        if default is None:
            return None
        return np.full(len(asset_index), float(default), dtype=float)
    if isinstance(bounds, pd.Series):
        return bounds.reindex(asset_index).astype(float).to_numpy()
    if isinstance(bounds, (list, tuple)):
        return np.asarray(bounds, dtype=float)
    return np.full(len(asset_index), float(bounds), dtype=float)


def project_weights_to_budget_feasible(
    weights: pd.Series,
    budgets: Sequence[RiskBudget],
    *,
    lower_bounds: pd.Series | Sequence[float] | float | None = None,
    upper_bounds: pd.Series | Sequence[float] | float | None = None,
    tolerance: float = 1e-6,
) -> pd.Series:
    """Project ``weights`` onto the feasible region defined by ``budgets``."""

    if not budgets:
        return weights.copy().astype(float)

    asset_index = list(weights.index)
    target = weights.reindex(asset_index).fillna(0.0).astype(float)
    target_total = float(target.sum())
    if not np.isfinite(target_total) or abs(target_total) < tolerance:
        target_total = 1.0

    w_var = cp.Variable(len(asset_index))
    objective = cp.Minimize(cp.sum_squares(w_var - target.to_numpy(dtype=float)))
    constraints: list[cp.Constraint] = [cp.sum(w_var) == target_total]

    lower_array = _resolve_bound_array(lower_bounds, asset_index, default=None)
    if lower_array is not None:
        constraints.append(w_var >= lower_array)

    upper_array = _resolve_bound_array(upper_bounds, asset_index, default=None)
    if upper_array is not None:
        constraints.append(w_var <= upper_array)

    constraints.extend(budgets_to_constraints(w_var, budgets, asset_index))

    problem = cp.Problem(objective, constraints)

    last_error: Exception | None = None
    for solver in ("CLARABEL", "OSQP", "ECOS"):
        try:
            problem.solve(solver=solver, verbose=False)
        except Exception as exc:  # pragma: no cover - solver specific
            last_error = exc
            continue
        if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            break
    else:
        raise BudgetViolationError(
            f"Failed to enforce budgets; solver status={problem.status}",
            find_budget_violations(target, budgets, tol=tolerance),
        ) from last_error

    solution = pd.Series(w_var.value, index=asset_index, dtype=float).fillna(0.0)
    total = float(solution.sum())
    if not np.isclose(total, 1.0, atol=tolerance):
        if total == 0:
            raise BudgetViolationError(
                "Projection produced zero total weight.",
                find_budget_violations(solution, budgets, tol=tolerance),
            )
        solution /= total

    violations = find_budget_violations(solution, budgets, tol=tolerance)
    if violations:
        raise BudgetViolationError(
            "Unable to satisfy budgets after projection.", violations
        )

    return solution
