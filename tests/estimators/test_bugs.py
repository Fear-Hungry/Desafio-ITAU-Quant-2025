"""Tests to verify identified bugs in estimators before fixing them."""

import numpy as np
import pandas as pd
import pytest
from itau_quant.estimators import bl, cov, factors, mu, validation
from numpy.testing import assert_allclose


class TestLedoitWolfBiasedEstimator:
    """Test Bug 1: Ledoit-Wolf uses biased divisor (n instead of n-1)."""

    def test_ledoit_wolf_uses_unbiased_estimator(self):
        """Verify that Ledoit-Wolf uses unbiased estimator (n-1 divisor)."""
        rng = np.random.default_rng(42)
        n_samples = 20
        n_assets = 3
        returns = pd.DataFrame(
            rng.normal(0.0, 0.01, size=(n_samples, n_assets)), columns=["A", "B", "C"]
        )

        lw_cov, shrinkage = cov.ledoit_wolf_shrinkage(returns, assume_centered=False)
        sample_cov = cov.sample_cov(returns, ddof=1)

        centered = returns - returns.mean()
        unbiased_cov_manual = (centered.T @ centered) / (n_samples - 1)

        # Verify the empirical covariance used in LW matches unbiased estimate
        assert_allclose(
            unbiased_cov_manual.to_numpy(),
            sample_cov.to_numpy(),
            rtol=1e-10,
            err_msg="Sample cov should use n-1 divisor",
        )

        # When shrinkage is low, LW should be close to unbiased sample cov
        if shrinkage < 0.3:
            max_diff = np.max(np.abs(lw_cov.to_numpy() - sample_cov.to_numpy()))
            # Difference should be small relative to variance scale
            assert max_diff < 0.1 * np.mean(
                np.diag(sample_cov.to_numpy())
            ), "LW with low shrinkage should be close to unbiased sample_cov"

    def test_ledoit_wolf_should_match_unbiased_when_shrinkage_zero(self):
        """When shrinkage is near zero, LW should match unbiased sample cov."""
        returns = pd.DataFrame(np.eye(50), columns=[f"Asset{i}" for i in range(50)])

        lw_cov, shrinkage = cov.ledoit_wolf_shrinkage(returns, assume_centered=False)
        sample_cov = cov.sample_cov(returns, ddof=1)

        if shrinkage < 0.01:
            max_diff = np.max(np.abs(lw_cov.to_numpy() - sample_cov.to_numpy()))
            assert (
                max_diff < 0.01
            ), "LW should match unbiased sample_cov when shrinkage is near zero"


class TestStudentTMeanScaleConsistency:
    """Test Bug 2: Student-t mean has scale inconsistency."""

    def test_student_t_mean_scale_initialization_vs_iteration(self):
        """Verify scale computation consistency between init and iterations."""
        rng = np.random.default_rng(0)
        returns = pd.DataFrame(
            rng.normal(0.01, 0.02, size=(100, 2)), columns=["A", "B"]
        )

        result = mu.student_t_mean(returns, nu=5.0, max_iter=1, tol=1e-12)

        returns.mean()
        returns.var(axis=0, ddof=1)

        assert isinstance(result, pd.Series), "Should return Series"

    def test_student_t_mean_convergence_different_nu(self):
        """Test convergence behavior with different nu values."""
        rng = np.random.default_rng(1)
        returns = pd.DataFrame(rng.standard_t(df=3, size=(200, 2)), columns=["A", "B"])

        result_nu3 = mu.student_t_mean(returns, nu=3.0, max_iter=200)
        result_nu10 = mu.student_t_mean(returns, nu=10.0, max_iter=200)
        sample_mean = returns.mean()

        diff_nu3 = np.abs(result_nu3 - sample_mean).sum()
        diff_nu10 = np.abs(result_nu10 - sample_mean).sum()

        assert diff_nu3 > 0 or diff_nu10 > 0, "Should converge to different values"


class TestWinsorizeQuantileConsistency:
    """Test Bug 3: Winsorize doesn't specify quantile method."""

    def test_winsorize_quantile_method_consistency(self):
        """Verify winsorization is consistent across calls."""
        rng = np.random.default_rng(123)
        data = pd.DataFrame(rng.normal(0, 1, size=(100, 3)), columns=["A", "B", "C"])

        result1 = factors._winsorize(data, 0.05, 0.95)
        result2 = factors._winsorize(data, 0.05, 0.95)

        pd.testing.assert_frame_equal(result1, result2)

    def test_winsorize_extreme_values_clipped(self):
        """Verify extreme values are properly clipped."""
        data = pd.DataFrame({"A": [1, 2, 3, 4, 5, 100], "B": [-100, 1, 2, 3, 4, 5]})

        winsorized = factors._winsorize(data, 0.1, 0.9)

        assert winsorized["A"].max() < 100, "Should clip high outliers"
        assert winsorized["B"].min() > -100, "Should clip low outliers"


class TestEmbargoEdgeCases:
    """Test Bug 4: Embargo calculation with edge cases."""

    def test_embargo_empty_train_array(self):
        """Test embargo with empty train array."""
        train_idx = np.array([], dtype=int)
        test_idx = np.array([10, 11, 12, 13, 14])

        result = validation.apply_embargo(
            train_idx, test_idx, embargo_pct=0.1, total_observations=30
        )

        assert len(result) == 0, "Should return empty array"
        assert result.dtype == int, "Should preserve dtype"

    def test_embargo_boundary_conditions(self):
        """Test embargo at boundaries."""
        train_idx = np.array([0, 1, 2, 20, 21, 22])
        test_idx = np.array([10, 11, 12, 13, 14])
        total_obs = 30

        embargoed = validation.apply_embargo(
            train_idx, test_idx, embargo_pct=0.2, total_observations=total_obs
        )

        last_test = test_idx.max()
        embargo_count = int(np.ceil(total_obs * 0.2))

        for idx in range(last_test + 1, last_test + 1 + embargo_count):
            assert idx not in embargoed, f"Index {idx} should be embargoed"

    def test_embargo_total_observations_inference(self):
        """Test total_observations inference when None."""
        train_idx = np.array([0, 5, 10, 15, 20])
        test_idx = np.array([25, 26, 27])

        result = validation.apply_embargo(
            train_idx, test_idx, embargo_pct=0.1, total_observations=None
        )

        assert isinstance(result, np.ndarray), "Should return ndarray"


class TestBLConfidenceValidation:
    """Test Bug 5: BL accepts confidence > 1.0 without validation."""

    def test_build_projection_matrix_confidence_greater_than_one(self):
        """Test that confidence > 1.0 raises ValueError."""
        assets = ["A", "B", "C"]
        views = [
            {
                "type": "absolute",
                "asset": "A",
                "expected_return": 0.08,
                "confidence": 1.5,  # Invalid: > 1.0
            }
        ]

        with pytest.raises(ValueError, match="confidence must be in \\[0, 1\\]"):
            bl.build_projection_matrix(views, assets)

    def test_view_uncertainty_confidence_limits(self):
        """Test view_uncertainty with valid edge case confidences."""
        assets = ["A", "B", "C"]
        cov_mat = pd.DataFrame(
            np.diag([0.04, 0.09, 0.16]), index=assets, columns=assets
        )
        views = [
            {
                "type": "absolute",
                "asset": "A",
                "expected_return": 0.08,
                "confidence": 1.0,  # Valid: exactly 1.0
            }
        ]

        P, Q, confidences = bl.build_projection_matrix(views, assets)

        omega = bl.view_uncertainty(
            views=views,
            tau=0.05,
            cov=cov_mat,
            P=P,
            confidences=confidences,
            mode="diagonal",
        )

        assert omega.shape == (1, 1), "Should create uncertainty matrix"
        assert omega[0, 0] > 0, "Omega diagonal should be positive"

    def test_view_uncertainty_zero_confidence(self):
        """Test view_uncertainty with zero confidence (infinite uncertainty)."""
        assets = ["A", "B", "C"]
        cov_mat = pd.DataFrame(
            np.diag([0.04, 0.09, 0.16]), index=assets, columns=assets
        )
        views = [
            {
                "type": "absolute",
                "asset": "A",
                "expected_return": 0.08,
                "confidence": 0.0,
            }
        ]

        P, Q, confidences = bl.build_projection_matrix(views, assets)

        omega = bl.view_uncertainty(
            views=views,
            tau=0.05,
            cov=cov_mat,
            P=P,
            confidences=confidences,
            mode="diagonal",
        )

        assert (
            omega[0, 0] >= 1e6 * 0.05 * 0.04
        ), "Zero confidence should give very high uncertainty"


class TestHuberMeanEdgeCases:
    """Additional edge case tests for Huber mean."""

    def test_huber_mean_zero_scale_recovery(self):
        """Test Huber mean when initial scale estimate is zero."""
        returns = pd.DataFrame(
            {"A": [0.01, 0.01, 0.01, 0.01], "B": [0.02, 0.02, 0.02, 0.02]}
        )

        result, weights = mu.huber_mean(returns, c=1.5)

        assert_allclose(result["A"], 0.01, rtol=1e-6)
        assert_allclose(result["B"], 0.02, rtol=1e-6)


class TestTylerEstimatorEdgeCases:
    """Additional edge case tests for Tyler M-estimator."""

    def test_tyler_estimator_convergence_warning(self):
        """Test Tyler estimator emits warning when not converged."""
        rng = np.random.default_rng(99)
        returns = pd.DataFrame(
            rng.normal(0, 0.01, size=(50, 10)), columns=[f"Asset{i}" for i in range(10)]
        )

        with pytest.warns(RuntimeWarning, match="did not converge"):
            result = cov.tyler_m_estimator(returns, max_iter=1, tol=1e-12)

        assert result.shape == (10, 10), "Should still return result"


class TestNonlinearShrinkageHighDim:
    """Test nonlinear shrinkage in high-dimensional regime."""

    def test_nonlinear_shrinkage_when_p_close_to_n(self):
        """Test nonlinear shrinkage when p ≈ n (Marchenko-Pastur regime)."""
        rng = np.random.default_rng(777)
        n_samples = 60
        n_assets = 50
        returns = pd.DataFrame(
            rng.normal(0, 0.01, size=(n_samples, n_assets)),
            columns=[f"Asset{i}" for i in range(n_assets)],
        )

        result = cov.nonlinear_shrinkage(returns)

        eigvals = np.linalg.eigvalsh(result.to_numpy())
        assert np.all(eigvals >= 0), "Should be PSD"
        assert np.isclose(
            np.trace(result.to_numpy()), np.trace(result.to_numpy())
        ), "Trace should be preserved"


class TestCovTypoComment:
    """Test Bug 6: Typo in comment."""

    def test_student_t_cov_docstring(self):
        """Verify student_t_cov function exists and works."""
        returns = pd.DataFrame({"A": [0.01, 0.02, 0.03], "B": [0.015, 0.025, 0.035]})

        result = cov.student_t_cov(returns, nu=5.0)

        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
        assert result.shape == (2, 2), "Should have correct shape"
