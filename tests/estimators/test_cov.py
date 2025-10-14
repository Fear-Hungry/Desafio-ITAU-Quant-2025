"""Tests for covariance estimation utilities."""

import numpy as np
import pandas as pd
import pytest

from itau_quant.estimators import cov


def _toy_returns() -> pd.DataFrame:
    data = np.array(
        [
            [0.01, 0.02, 0.015],
            [0.011, 0.018, 0.016],
            [0.013, 0.017, 0.014],
            [0.012, 0.021, 0.015],
        ]
    )
    return pd.DataFrame(data, columns=["A", "B", "C"])


def test_sample_cov_matches_numpy():
    returns = _toy_returns()
    cov_df = cov.sample_cov(returns)
    expected = np.cov(returns.to_numpy().T, ddof=1)
    np.testing.assert_allclose(cov_df.to_numpy(), expected, rtol=1e-12, atol=1e-12)
    assert list(cov_df.columns) == ["A", "B", "C"]


def test_project_to_psd_clips_negative_eigenvalues():
    matrix = np.array([[2.0, 3.0], [3.0, 2.0]])
    matrix[1, 1] = -1.0  # induce negative eigenvalue
    projected = cov.project_to_psd(matrix, epsilon=1e-9)
    eigvals = np.linalg.eigvalsh(projected if isinstance(projected, np.ndarray) else projected.to_numpy())
    assert np.all(eigvals >= 0)


def test_ledoit_wolf_shrinkage_output():
    returns = _toy_returns()
    cov_df, shrinkage = cov.ledoit_wolf_shrinkage(returns)
    assert 0.0 <= shrinkage <= 1.0
    eigvals = np.linalg.eigvalsh(cov_df.to_numpy())
    assert np.all(eigvals >= -1e-9)


def test_nonlinear_shrinkage_psd_and_labels():
    returns = _toy_returns()
    nonlinear = cov.nonlinear_shrinkage(returns)
    np.testing.assert_array_equal(nonlinear.index, returns.columns)
    eigvals = np.linalg.eigvalsh(nonlinear.to_numpy())
    assert np.all(eigvals >= -1e-9)


def test_tyler_m_estimator_trace_normalisation():
    returns = _toy_returns()
    tyler = cov.tyler_m_estimator(returns)
    assert np.isclose(np.trace(tyler.to_numpy()), tyler.shape[0], atol=1e-6)
    eigvals = np.linalg.eigvalsh(tyler.to_numpy())
    assert np.all(eigvals >= -1e-9)


@pytest.mark.parametrize("nu", [4.0, 10.0])
def test_student_t_cov_scales_sample(nu):
    returns = _toy_returns()
    sample_cov = cov.sample_cov(returns)
    student_cov = cov.student_t_cov(returns, nu=nu)
    factor = nu / (nu - 2.0)
    np.testing.assert_allclose(student_cov.to_numpy(), sample_cov.to_numpy() * factor)


@pytest.mark.parametrize("method", ["diag", "ridge", "shrink_to_identity"])
def test_regularize_cov_methods(method):
    matrix = cov.sample_cov(_toy_returns())
    regularized = cov.regularize_cov(matrix, method=method, floor=0.05)
    eigvals = np.linalg.eigvalsh(regularized.to_numpy() if isinstance(regularized, pd.DataFrame) else regularized)
    assert np.all(eigvals >= -1e-9)
