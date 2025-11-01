from __future__ import annotations

import numpy as np
import pandas as pd
from itau_quant.optimization.core import risk_parity as rp


def test_risk_contribution_sums_to_one():
    cov = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]], index=["A", "B"], columns=["A", "B"]
    )
    weights = pd.Series([0.6, 0.4], index=cov.index)
    contributions = rp.risk_contribution(weights, cov)
    assert np.isclose(contributions.sum(), 1.0)
    assert (contributions > 0).all()


def test_iterative_risk_parity_nearly_equalises_contributions():
    cov = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]], index=["A", "B"], columns=["A", "B"]
    )
    result = rp.iterative_risk_parity(cov)
    contributions = rp.risk_contribution(result, cov)
    assert np.allclose(contributions, contributions.mean(), atol=1e-3)


def test_solve_log_barrier_produces_valid_weights():
    cov = pd.DataFrame(
        [[0.025, 0.006], [0.006, 0.04]], index=["X", "Y"], columns=["X", "Y"]
    )
    weights = rp.solve_log_barrier(cov, bounds=(0.0, 0.8))
    assert np.isclose(weights.sum(), 1.0)
    assert (weights >= 0).all()


def test_cluster_risk_parity_respects_clusters():
    cov = pd.DataFrame(
        [
            [0.04, 0.01, 0.02, 0.015],
            [0.01, 0.03, 0.018, 0.012],
            [0.02, 0.018, 0.05, 0.02],
            [0.015, 0.012, 0.02, 0.06],
        ],
        index=["A", "B", "C", "D"],
        columns=["A", "B", "C", "D"],
    )
    clusters = {"Cluster1": ["A", "B"], "Cluster2": ["C", "D"]}
    weights = rp.cluster_risk_parity(cov, clusters)
    assert np.isclose(weights.sum(), 1.0)
    assert (weights >= 0).all()
    assert {asset for asset in weights.index if weights.loc[asset] > 0}.issubset(
        set(cov.index)
    )


def test_risk_parity_wrapper_returns_result():
    cov = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]], index=["A", "B"], columns=["A", "B"]
    )
    result = rp.risk_parity(cov, config={"method": "iterative"})
    assert np.isclose(result.weights.sum(), 1.0)
    assert result.converged
