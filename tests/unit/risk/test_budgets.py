from __future__ import annotations

import cvxpy as cp
import pandas as pd
import pytest
from arara_quant.risk.budgets import (
    RiskBudget,
    aggregate_by_budget,
    budget_slack,
    budgets_to_constraints,
    load_budgets,
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
