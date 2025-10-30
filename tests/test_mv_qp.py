from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from itau_quant.optimization.core.mv_qp import (
    MeanVarianceConfig,
    solve_mean_variance,
)
from itau_quant.risk.budgets import RiskBudget


def _default_config(**overrides) -> MeanVarianceConfig:
    assets = overrides.pop("assets", ["A", "B", "C"])
    lower = pd.Series(0.0, index=assets, dtype=float)
    upper = pd.Series(1.0, index=assets, dtype=float)
    prev = pd.Series(0.0, index=assets, dtype=float)
    return MeanVarianceConfig(
        risk_aversion=overrides.pop("risk_aversion", 5.0),
        turnover_penalty=overrides.pop("turnover_penalty", 0.0),
        turnover_cap=overrides.pop("turnover_cap", None),
        lower_bounds=overrides.pop("lower_bounds", lower),
        upper_bounds=overrides.pop("upper_bounds", upper),
        previous_weights=overrides.pop("previous_weights", prev),
        cost_vector=overrides.pop("cost_vector", None),
        solver=overrides.pop("solver", None),
        solver_kwargs=overrides.pop("solver_kwargs", {}),
        risk_config=overrides.pop("risk_config", None),
        budgets=overrides.pop("budgets", None),
    )


def test_mv_qp_prefers_highest_return() -> None:
    mu = pd.Series([0.08, 0.12, 0.06], index=["A", "B", "C"], dtype=float)
    cov = pd.DataFrame(np.eye(3) * 0.1, index=mu.index, columns=mu.index)
    config = _default_config(risk_aversion=3.0)

    result = solve_mean_variance(mu, cov, config)

    assert pytest.approx(result.weights.sum(), rel=1e-6) == 1.0
    assert (result.weights >= -1e-8).all()
    assert (result.weights <= 1.0 + 1e-8).all()
    assert result.weights["B"] > result.weights["A"]
    assert result.summary.is_optimal()


def test_mv_qp_turnover_cap_respected() -> None:
    mu = pd.Series([0.05, 0.09, 0.04], index=["A", "B", "C"], dtype=float)
    cov = pd.DataFrame(
        [[0.08, 0.02, 0.01], [0.02, 0.07, 0.015], [0.01, 0.015, 0.05]],
        index=mu.index,
        columns=mu.index,
        dtype=float,
    )
    prev = pd.Series([1 / 3, 1 / 3, 1 / 3], index=mu.index, dtype=float)
    config = _default_config(previous_weights=prev, turnover_cap=0.10, risk_aversion=4.0)

    result = solve_mean_variance(mu, cov, config)

    turnover = np.abs(result.weights - prev).sum()
    assert turnover <= 0.10 + 1e-4


def test_mv_qp_cost_penalty_discourages_large_trade() -> None:
    mu = pd.Series([0.09, 0.11, 0.07], index=["A", "B", "C"], dtype=float)
    cov = pd.DataFrame(np.eye(3) * 0.06, index=mu.index, columns=mu.index)
    prev = pd.Series([0.4, 0.3, 0.3], index=mu.index, dtype=float)
    cost_vector = pd.Series([0.0, 5e-4, 0.0], index=mu.index, dtype=float)
    base_config = _default_config(previous_weights=prev, cost_vector=None, turnover_penalty=0.0, risk_aversion=2.0)
    base_result = solve_mean_variance(mu, cov, base_config)

    penalised_config = _default_config(
        previous_weights=prev,
        cost_vector=cost_vector,
        turnover_penalty=0.0,
        risk_aversion=2.0,
    )
    penalised_result = solve_mean_variance(mu, cov, penalised_config)

    base_move = abs(base_result.weights["B"] - prev["B"])
    penalised_move = abs(penalised_result.weights["B"] - prev["B"])
    assert penalised_move <= base_move + 1e-6
    assert penalised_result.cost > 0


def test_mv_qp_respects_budget_constraints() -> None:
    mu = pd.Series([0.12, 0.11, 0.02], index=["A", "B", "C"], dtype=float)
    cov = pd.DataFrame(np.eye(3) * 0.05, index=mu.index, columns=mu.index)
    budget = RiskBudget(name="growth_pair", tickers=["A", "B"], max_weight=0.40)
    config = _default_config(risk_aversion=1.0, budgets=[budget])

    result = solve_mean_variance(mu, cov, config)

    group_weight = float(result.weights.loc[["A", "B"]].sum())
    assert group_weight <= 0.40 + 1e-4


def test_mv_qp_parses_budget_from_risk_config() -> None:
    mu = pd.Series([0.10, 0.09, 0.03], index=["A", "B", "C"], dtype=float)
    cov = pd.DataFrame(np.eye(3) * 0.04, index=mu.index, columns=mu.index)
    risk_cfg = {
        "budgets": [
            {"name": "cap", "tickers": ["A", "B"], "max_weight": 0.35},
        ]
    }
    config = _default_config(risk_aversion=1.0, risk_config=risk_cfg)

    result = solve_mean_variance(mu, cov, config)

    assert float(result.weights.loc[["A", "B"]].sum()) <= 0.35 + 1e-4
