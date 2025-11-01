from __future__ import annotations

import numpy as np
import pandas as pd
from itau_quant.optimization.core import cvar_lp


def test_solve_cvar_lp_returns_feasible_solution() -> None:
    returns = pd.DataFrame(
        [[0.01, 0.02], [-0.02, 0.03], [0.015, -0.01]],
        columns=["AAA", "BBB"],
    )
    expected = returns.mean()
    config = cvar_lp.CvarConfig(alpha=0.95, risk_aversion=5.0)

    result = cvar_lp.solve_cvar_lp(returns, expected, config)

    assert np.isclose(result.weights.sum(), 1.0)
    assert result.summary.is_optimal()


def test_cvar_with_target_return_constraint() -> None:
    returns = pd.DataFrame(
        [[0.02, -0.01], [0.03, 0.01], [-0.02, 0.04]],
        columns=["AAA", "BBB"],
    )
    expected = returns.mean()
    config = cvar_lp.CvarConfig(alpha=0.9, risk_aversion=2.0, target_return=0.01)

    result = cvar_lp.solve_cvar_lp(returns, expected, config)

    assert result.expected_return >= 0.01 - 1e-9
