from __future__ import annotations

import numpy as np
import pandas as pd
from itau_quant.optimization.heuristics import hrp as hrp_mod


def test_equal_weight_allocation():
    assets = ["AAA", "BBB", "CCC"]
    weights = hrp_mod.equal_weight(assets)
    assert np.isclose(weights.sum(), 1.0)
    np.testing.assert_allclose(weights.values, np.full(3, 1 / 3), atol=1e-8)


def test_inverse_variance_matches_expected_on_diagonal():
    cov = pd.DataFrame(
        np.diag([0.04, 0.09, 0.16]), index=["A", "B", "C"], columns=["A", "B", "C"]
    )
    weights = hrp_mod.inverse_variance_portfolio(cov)
    expected = pd.Series([0.5901639, 0.2622951, 0.1475410], index=cov.index)
    np.testing.assert_allclose(weights.loc[cov.index], expected, atol=1e-6)


def test_hierarchical_risk_parity_returns_unit_sum():
    cov = pd.DataFrame(
        [
            [0.04, 0.01, 0.02],
            [0.01, 0.03, 0.015],
            [0.02, 0.015, 0.05],
        ],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    weights = hrp_mod.hierarchical_risk_parity(cov)
    assert np.isclose(weights.sum(), 1.0)
    assert (weights >= 0).all()


def test_cluster_then_allocate_returns_normalised_weights():
    rng = np.random.default_rng(42)
    returns = pd.DataFrame(
        rng.normal(size=(120, 4)),
        columns=["AAA", "BBB", "CCC", "DDD"],
    )
    weights = hrp_mod.cluster_then_allocate(returns, n_clusters=2)
    assert np.isclose(weights.sum(), 1.0)
    assert (weights >= 0).all()


def test_heuristic_allocation_dispatches_methods():
    cov = pd.DataFrame(
        np.diag([0.04, 0.09]), index=["AAA", "BBB"], columns=["AAA", "BBB"]
    )
    returns = pd.DataFrame(
        np.random.default_rng(1).normal(size=(60, 3)),
        columns=["X", "Y", "Z"],
    )

    result_eq = hrp_mod.heuristic_allocation(
        {"assets": ["AAA", "BBB"]}, method="equal_weight"
    )
    assert result_eq.weights.index.tolist() == ["AAA", "BBB"]
    assert result_eq.diagnostics == {"count": 2}

    result_iv = hrp_mod.heuristic_allocation(
        {"covariance": cov}, method="inverse_variance"
    )
    assert np.isclose(result_iv.weights.sum(), 1.0)
    assert "variances" in result_iv.diagnostics

    result_hrp = hrp_mod.heuristic_allocation({"covariance": cov}, method="hrp")
    assert np.isclose(result_hrp.weights.sum(), 1.0)
    assert "ordered_assets" in result_hrp.diagnostics

    result_cluster = hrp_mod.heuristic_allocation(
        {"returns": returns}, method="cluster", config={"n_clusters": 2}
    )
    assert np.isclose(result_cluster.weights.sum(), 1.0)
    assert "labels" in result_cluster.diagnostics
