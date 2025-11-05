from __future__ import annotations

import numpy as np
from arara_quant.risk.cvar import historical_scenarios, solve_cvar_portfolio


def test_solve_cvar_portfolio_returns_weights():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.001, 0.01, size=(100, 3))
    mu = returns.mean(axis=0)
    result = solve_cvar_portfolio(returns, mu, alpha=0.95, risk_aversion=5.0)
    assert result["weights"].shape == (3,)
    assert abs(result["weights"].sum() - 1.0) < 1e-6


def test_historical_scenarios_tail():
    import pandas as pd

    returns = pd.DataFrame({"A": [0.01, -0.02, 0.03], "B": [0.0, 0.01, -0.01]})
    scenarios = historical_scenarios(returns, window=2)
    assert scenarios.shape == (2, 2)
