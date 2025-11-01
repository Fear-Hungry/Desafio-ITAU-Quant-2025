"""Risk budget helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

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
