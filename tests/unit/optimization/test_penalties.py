from __future__ import annotations

import cvxpy as cp
import pandas as pd
import pytest
from itau_quant.optimization.core import penalties as penalties_mod


def test_l1_and_l2_penalties_are_convex():
    weights = cp.Variable(3)
    expr_l1 = penalties_mod.l1_penalty(weights, gamma=0.5)
    expr_l2 = penalties_mod.l2_penalty(weights, gamma=1.2)
    assert expr_l1.is_convex()
    assert expr_l2.is_convex()


def test_group_lasso_penalty_handles_named_groups():
    weights = cp.Variable(4)
    groups = {"Tech": ["AAPL", "MSFT"], "Bonds": ["IEF", "TLT"]}
    penalty = penalties_mod.group_lasso_penalty(
        weights, groups, gamma=0.3, asset_index=["AAPL", "MSFT", "IEF", "TLT"]
    )
    assert penalty.is_convex()


def test_turnover_penalty_uses_previous_weights():
    weights = cp.Variable(2)
    prev = pd.Series([0.6, 0.4], index=["AAA", "BBB"])
    penalty = penalties_mod.turnover_penalty(
        weights, prev, gamma=0.7, asset_index=["AAA", "BBB"], normalised=True
    )
    assert penalty.is_convex()


def test_penalty_factory_combines_terms():
    weights = cp.Variable(3)
    config = {
        "l1": {"gamma": 0.2},
        "turnover": {"gamma": 0.5, "previous": [0.3, 0.4, 0.3], "normalised": False},
        "cardinality": {"gamma": 0.1, "k_target": 2},
    }
    penalties = penalties_mod.penalty_factory(
        weights, config, asset_index=["A", "B", "C"]
    )
    assert len(penalties) == 3
    for expr in penalties:
        assert expr.is_convex()


def test_cardinality_penalty_rejects_invalid_method():
    weights = cp.Variable(2)
    with pytest.raises(ValueError):
        penalties_mod.cardinality_soft_penalty(
            weights, k_target=1, gamma=0.5, method="invalid"
        )
