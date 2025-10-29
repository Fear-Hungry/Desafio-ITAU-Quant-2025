from __future__ import annotations

import numpy as np
import pandas as pd

from itau_quant.optimization.core import sharpe_socp as sharpe_mod


def _analytic_sharpe_solution(mu: pd.Series, cov: pd.DataFrame) -> pd.Series:
    inv = np.linalg.inv(cov.to_numpy(dtype=float))
    raw = inv @ mu.to_numpy(dtype=float)
    weights = pd.Series(raw, index=mu.index, dtype=float)
    weights /= weights.sum()
    return weights


def test_sharpe_socp_matches_analytic_solution_diagonal_cov():
    mu = pd.Series([0.10, 0.18, 0.12], index=["AAA", "BBB", "CCC"])
    cov = pd.DataFrame(np.diag([0.04, 0.09, 0.05]), index=mu.index, columns=mu.index)

    result = sharpe_mod.sharpe_socp(mu, cov)

    expected = _analytic_sharpe_solution(mu, cov)
    np.testing.assert_allclose(result.weights.loc[mu.index], expected, atol=1e-2)
    assert result.summary.is_optimal()


def test_sharpe_socp_penalises_turnover_with_linear_cost():
    mu = pd.Series([0.05, 0.15], index=["AAA", "BBB"])
    cov = pd.DataFrame([[0.03, 0.01], [0.01, 0.04]], index=mu.index, columns=mu.index)
    previous = pd.Series([0.8, 0.2], index=mu.index)

    aggressive = sharpe_mod.sharpe_socp(mu, cov, config={"bounds": (0.0, 1.0)})
    penalised = sharpe_mod.sharpe_socp(
        mu,
        cov,
        config={
            "bounds": (0.0, 1.0),
            "previous_weights": previous,
            "costs": {"linear": 10.0},
        },
    )

    aggressive_turnover = np.abs(aggressive.weights - previous).sum()
    penalised_turnover = np.abs(penalised.weights - previous).sum()
    assert penalised_turnover <= aggressive_turnover + 1e-6


def test_normalize_weights_handles_nan_and_renormalises():
    series = pd.Series([0.6, np.nan, 0.3], index=["A", "B", "C"])
    normalised = sharpe_mod.normalize_weights(series)
    assert np.isclose(normalised.sum(), 1.0)
    assert normalised["B"] == 0.0
