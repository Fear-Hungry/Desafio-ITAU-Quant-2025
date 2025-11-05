from __future__ import annotations

import cvxpy as cp
import pandas as pd
import pytest
from itau_quant.risk.budgets import (
    BudgetViolationError,
    RiskBudget,
    aggregate_by_budget,
    budget_slack,
    budgets_to_constraints,
    find_budget_violations,
    load_budgets,
    project_weights_to_budget_feasible,
    validate_budgets,
)


def test_load_and_validate_budgets():
    config = [
        {"name": "Equities", "tickers": ["AAA", "BBB"], "max_weight": 0.7},
        {"name": "Bonds", "tickers": ["CCC"], "min_weight": 0.2},
    ]
    budgets = load_budgets(config)
    validate_budgets(budgets, ["AAA", "BBB", "CCC"])


def test_budgets_to_constraints_and_slack():
    budgets = [
        RiskBudget(name="Equities", tickers=["AAA", "BBB"], max_weight=0.6),
        RiskBudget(name="Bonds", tickers=["CCC"], min_weight=0.2),
    ]
    weights_var = cp.Variable(3)
    constraints = budgets_to_constraints(weights_var, budgets, ["AAA", "BBB", "CCC"])
    assert len(constraints) == 2

    weights = pd.Series({"AAA": 0.3, "BBB": 0.2, "CCC": 0.5})
    slack = budget_slack(weights, budgets)
    assert slack["Equities"] == pytest.approx(0.1)


def test_aggregate_by_budget_returns_dataframe():
    budgets = [
        RiskBudget(name="Equities", tickers=["AAA", "BBB"], max_weight=0.6),
    ]
    weights = pd.Series({"AAA": 0.4, "BBB": 0.2})
    returns = pd.DataFrame({"AAA": [0.01, -0.02], "BBB": [0.0, 0.03]})
    agg = aggregate_by_budget(weights, returns, budgets)
    assert agg.loc[0, "budget"] == "Equities"


def test_find_budget_violations_flags_infeasible_weights():
    budgets = [
        RiskBudget(name="Growth", tickers=["AAA", "BBB"], max_weight=0.35),
        RiskBudget(name="Defensive", tickers=["CCC"], min_weight=0.65),
    ]
    weights = pd.Series({"AAA": 0.2, "BBB": 0.2, "CCC": 0.6})
    violations = find_budget_violations(weights, budgets, tol=1e-4)
    assert len(violations) == 2
    assert {v.bound for v in violations} == {"max", "min"}


def test_project_weights_to_budget_feasible_respects_limits():
    budgets = [
        RiskBudget(name="Growth", tickers=["AAA", "BBB"], max_weight=0.35),
        RiskBudget(name="Defensive", tickers=["CCC"], min_weight=0.65),
    ]
    weights = pd.Series({"AAA": 0.3, "BBB": 0.25, "CCC": 0.45})
    projected = project_weights_to_budget_feasible(weights, budgets, tolerance=1e-6)
    assert projected.sum() == pytest.approx(1.0)
    growth = projected[["AAA", "BBB"]].sum()
    defensive = projected["CCC"]
    assert growth <= 0.35 + 1e-6
    assert defensive >= 0.65 - 1e-6


def test_project_weights_to_budget_feasible_raises_when_impossible():
    budgets = [
        RiskBudget(name="AllAssets", tickers=["AAA", "BBB"], min_weight=1.1),
    ]
    weights = pd.Series({"AAA": 0.5, "BBB": 0.5})
    with pytest.raises(BudgetViolationError):
        project_weights_to_budget_feasible(weights, budgets)
