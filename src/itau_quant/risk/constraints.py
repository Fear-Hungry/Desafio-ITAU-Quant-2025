"""Reusable portfolio risk constraints."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import cvxpy as cp
import numpy as np
import pandas as pd

from .budgets import RiskBudget, budgets_to_constraints

__all__ = [
    "weight_sum_constraint",
    "box_constraints",
    "group_constraints",
    "factor_exposure_constraints",
    "leverage_constraint",
    "tracking_error_constraint",
    "turnover_constraint",
    "build_constraints",
]


def weight_sum_constraint(weights: cp.Variable, target: float = 1.0) -> cp.Constraint:
    return cp.sum(weights) == float(target)


def box_constraints(
    weights: cp.Variable,
    lower: Sequence[float] | float,
    upper: Sequence[float] | float,
) -> list[cp.Constraint]:
    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    return [weights >= lower_array, weights <= upper_array]


def group_constraints(
    weights: cp.Variable,
    budgets: Iterable[RiskBudget],
    asset_index: Sequence[str],
) -> list[cp.Constraint]:
    return budgets_to_constraints(weights, budgets, asset_index)


def factor_exposure_constraints(
    weights: cp.Variable,
    factor_loadings: pd.DataFrame,
    exposure_limits: Mapping[str, tuple[float | None, float | None]],
) -> list[cp.Constraint]:
    constraints: list[cp.Constraint] = []
    for factor, (lower, upper) in exposure_limits.items():
        loading = factor_loadings.loc[:, factor].to_numpy(dtype=float)
        exposure = loading @ weights
        if lower is not None:
            constraints.append(exposure >= float(lower))
        if upper is not None:
            constraints.append(exposure <= float(upper))
    return constraints


def leverage_constraint(weights: cp.Variable, max_leverage: float) -> cp.Constraint:
    return cp.norm1(weights) <= float(max_leverage)


def tracking_error_constraint(
    weights: cp.Variable,
    benchmark_weights: Sequence[float],
    cov: np.ndarray,
    max_te: float,
) -> cp.Constraint:
    diff = weights - np.asarray(benchmark_weights, dtype=float)
    return cp.quad_form(diff, cov) <= float(max_te) ** 2


def turnover_constraint(
    weights: cp.Variable,
    previous_weights: Sequence[float],
    max_turnover: float,
) -> cp.Constraint:
    return cp.norm1(weights - np.asarray(previous_weights, dtype=float)) <= float(max_turnover)


def build_constraints(
    config: Mapping[str, object],
    context: Mapping[str, object],
) -> list[cp.Constraint]:
    """Build constraints from declarative configuration."""

    constraints: list[cp.Constraint] = []
    weights_var: cp.Variable = context["weights_var"]

    if config.get("weight_sum"):
        constraints.append(weight_sum_constraint(weights_var, target=float(config["weight_sum"])))

    if "box" in config:
        box_cfg = config["box"]
        constraints.extend(box_constraints(weights_var, box_cfg["lower"], box_cfg["upper"]))

    if "budgets" in config:
        budgets_cfg = load_budget_objects(config["budgets"])
        constraints.extend(group_constraints(weights_var, budgets_cfg, context["asset_index"]))

    if "factor_exposure" in config:
        fe_cfg = config["factor_exposure"]
        constraints.extend(
            factor_exposure_constraints(
                weights_var,
                context["factor_loadings"],
                fe_cfg,
            )
        )

    if "max_leverage" in config:
        constraints.append(leverage_constraint(weights_var, float(config["max_leverage"])))

    if "tracking_error" in config:
        te_cfg = config["tracking_error"]
        constraints.append(
            tracking_error_constraint(
                weights_var,
                te_cfg["benchmark"],
                context["covariance"],
                te_cfg["max_te"],
            )
        )

    if "turnover" in config:
        turnover_cfg = config["turnover"]
        constraints.append(
            turnover_constraint(weights_var, turnover_cfg["previous"], turnover_cfg["max_turnover"])
        )

    return constraints


def load_budget_objects(raw_budgets: Iterable[Mapping[str, object]]) -> list[RiskBudget]:
    budgets: list[RiskBudget] = []
    for entry in raw_budgets:
        budgets.append(
            RiskBudget(
                name=str(entry["name"]),
                tickers=list(entry.get("tickers", [])),
                min_weight=entry.get("min_weight"),
                max_weight=entry.get("max_weight"),
            )
        )
    return budgets
