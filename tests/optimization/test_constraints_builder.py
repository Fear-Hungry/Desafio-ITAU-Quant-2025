from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
from arara_quant.optimization.core import constraints_builder as cb


def test_budget_constraints_handle_leverage():
    weights = cp.Variable(3)
    constraints = cb.build_budget_constraints(
        weights,
        {"target": 1.0, "max_leverage": 1.5},
    )
    assert len(constraints) == 2


def test_bound_constraints_with_series():
    weights = cp.Variable(3)
    lower = pd.Series([0.0, 0.1, 0.2])
    upper = [0.6, 0.7, 0.8]
    constraints = cb.build_bound_constraints(weights, lower, upper)
    assert len(constraints) == 2


def test_turnover_constraint_forces_previous_when_zero_cap():
    weights = cp.Variable(2)
    constraints = cb.build_turnover_constraints(weights, [0.3, 0.7], max_turnover=0.0)
    assert len(constraints) == 1


def test_sector_constraints_with_mapping():
    weights = cp.Variable(3)
    sector_map = {"AAA": "Equity", "BBB": "Equity", "CCC": "Fixed"}
    limits = {"Equity": (None, 0.7), "Fixed": (0.3, None)}
    constraints = cb.build_sector_exposure_constraints(
        weights,
        sector_map,
        limits,
        asset_index=["AAA", "BBB", "CCC"],
    )
    assert len(constraints) == 2


def test_risk_constraints_include_volatility_and_cvar():
    weights = cp.Variable(2)
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    scenarios = np.array([[0.01, 0.02], [-0.03, 0.01], [0.02, -0.02]])
    constraints = cb.build_risk_constraints(
        weights,
        cov,
        {
            "volatility": {"max": 0.2},
            "cvar": {"max": 0.05, "alpha": 0.95, "scenario_returns": scenarios},
        },
    )
    assert len(constraints) >= 4


def test_compose_constraints_end_to_end():
    weights = cp.Variable(3)
    config = {
        "budget": {"target": 1.0, "max_leverage": 1.4},
        "bounds": {"lower": 0.0, "upper": 0.6},
        "turnover": {"max": 0.5, "previous": [0.3, 0.4, 0.3]},
        "sector": {
            "map": ["Equity", "Equity", "Fixed"],
            "limits": {"Equity": {"max": 0.7}, "Fixed": {"min": 0.3}},
        },
        "risk": {
            "volatility": {"max": 0.25},
            "covariance": np.eye(3) * 0.05,
        },
    }
    constraints = cb.compose_constraints(
        weights,
        config,
        asset_index=["AAA", "BBB", "CCC"],
        previous_weights=[0.3, 0.4, 0.3],
        covariance=np.eye(3) * 0.05,
    )
    assert len(constraints) >= 6
