from __future__ import annotations

import cvxpy as cp
import numpy as np
from itau_quant.risk.constraints import (
    build_constraints,
    leverage_constraint,
    tracking_error_constraint,
)


def test_leverage_constraint_builds_norm1():
    weights = cp.Variable(3)
    constraint = leverage_constraint(weights, 1.2)
    assert isinstance(constraint, cp.constraints.constraint.Constraint)


def test_tracking_error_constraint_shape():
    weights = cp.Variable(2)
    cov = np.eye(2)
    constraint = tracking_error_constraint(weights, [0.5, 0.5], cov, 0.1)
    assert isinstance(constraint, cp.constraints.constraint.Constraint)


def test_build_constraints_end_to_end():
    weights = cp.Variable(3)
    config = {
        "weight_sum": 1.0,
        "max_leverage": 1.2,
        "budgets": [{"name": "Group", "tickers": ["A", "B"], "max_weight": 0.7}],
    }
    context = {"weights_var": weights, "asset_index": ["A", "B", "C"]}
    constraints = build_constraints(config, context)
    assert len(constraints) >= 2
