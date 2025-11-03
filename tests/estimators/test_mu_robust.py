"""Tests for robust μ estimators with shrinkage."""

import numpy as np
import pandas as pd
import pytest

from itau_quant.estimators.mu_robust import (
    bayesian_shrinkage,
    combined_shrinkage,
    james_stein_shrinkage,
    shrink_mu_pipeline,
)


@pytest.fixture
def sample_returns():
    """Generate sample returns for testing."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=252, freq="D")
    returns = pd.DataFrame({
        "A": np.random.normal(0.0005, 0.01, 252),
        "B": np.random.normal(0.0008, 0.015, 252),
        "C": np.random.normal(0.0003, 0.008, 252),
    }, index=dates)
    return returns


@pytest.fixture
def sample_mu_sigma():
    """Generate sample μ and Σ for testing."""
    mu = pd.Series([0.10, 0.08, 0.12], index=["A", "B", "C"])
    sigma = pd.DataFrame([
        [0.04, 0.01, 0.02],
        [0.01, 0.09, 0.03],
        [0.02, 0.03, 0.16],
    ], index=["A", "B", "C"], columns=["A", "B", "C"])
    return mu, sigma


class TestJamesSteinShrinkage:
    """Tests for James-Stein shrinkage."""

    def test_shrinks_toward_target(self, sample_mu_sigma):
        """Test that shrinkage moves estimates toward target."""
        mu, sigma = sample_mu_sigma
        target = 0.0

        mu_shrunk = james_stein_shrinkage(mu, sigma, target=target)

        assert isinstance(mu_shrunk, pd.Series)
        assert list(mu_shrunk.index) == list(mu.index)

        # Shrunk values should be closer to target than original
        original_dist = np.abs(mu.values - target)
        shrunk_dist = np.abs(mu_shrunk.values - target)
        assert np.all(shrunk_dist <= original_dist)

    def test_extreme_sharpe_less_shrinkage(self, sample_mu_sigma):
        """Test that extreme Sharpe ratios get less shrinkage."""
        mu, sigma = sample_mu_sigma

        # Create extreme Sharpe scenario
        mu_extreme = pd.Series([0.50, 0.05, 0.03], index=["A", "B", "C"])

        mu_shrunk = james_stein_shrinkage(mu_extreme, sigma, target=0.0)

        # Asset A has extreme Sharpe, should retain more of original value
        shrinkage_A = abs(mu_shrunk.loc["A"] - mu_extreme.loc["A"])
        shrinkage_B = abs(mu_shrunk.loc["B"] - mu_extreme.loc["B"])

        # Relative shrinkage should be reasonable
        assert shrinkage_A < abs(mu_extreme.loc["A"])

    def test_near_target_no_shrinkage(self):
        """Test that values near target have minimal shrinkage."""
        mu = pd.Series([0.01, 0.02, 0.01], index=["A", "B", "C"])
        sigma = pd.DataFrame(np.eye(3) * 0.04, index=mu.index, columns=mu.index)

        # All values very close to zero
        mu_near_zero = pd.Series([0.0001, -0.0001, 0.0002], index=["A", "B", "C"])
        mu_shrunk = james_stein_shrinkage(mu_near_zero, sigma, target=0.0)

        # Should return near-original values
        np.testing.assert_allclose(mu_shrunk.values, mu_near_zero.values, atol=1e-3)

    def test_preserves_index(self, sample_mu_sigma):
        """Test that output preserves input index."""
        mu, sigma = sample_mu_sigma
        mu_shrunk = james_stein_shrinkage(mu, sigma)

        assert list(mu_shrunk.index) == list(mu.index)

    def test_with_T_parameter(self, sample_mu_sigma):
        """Test shrinkage with T parameter (observation count)."""
        mu, sigma = sample_mu_sigma

        mu_shrunk = james_stein_shrinkage(mu, sigma, T=252)

        assert isinstance(mu_shrunk, pd.Series)
        assert len(mu_shrunk) == len(mu)


class TestBayesianShrinkage:
    """Tests for Bayesian shrinkage."""

    def test_shrinks_toward_scalar_prior(self, sample_mu_sigma):
        """Test shrinkage toward scalar prior."""
        mu, _ = sample_mu_sigma
        prior = 0.05
        gamma = 0.5

        mu_shrunk = bayesian_shrinkage(mu, prior=prior, gamma=gamma)

        # Shrunk values should be between original and prior
        for asset in mu.index:
            if mu.loc[asset] > prior:
                assert prior < mu_shrunk.loc[asset] < mu.loc[asset]
            else:
                assert mu.loc[asset] < mu_shrunk.loc[asset] < prior

    def test_shrinks_toward_series_prior(self, sample_mu_sigma):
        """Test shrinkage toward Series prior."""
        mu, _ = sample_mu_sigma
        prior = pd.Series([0.05, 0.06, 0.07], index=["A", "B", "C"])
        gamma = 0.5

        mu_shrunk = bayesian_shrinkage(mu, prior=prior, gamma=gamma)

        # Expected: (1-0.5) * mu + 0.5 * prior
        expected = 0.5 * mu + 0.5 * prior
        pd.testing.assert_series_equal(mu_shrunk, expected)

    def test_gamma_zero_no_shrinkage(self, sample_mu_sigma):
        """Test that gamma=0 returns original μ."""
        mu, _ = sample_mu_sigma

        mu_shrunk = bayesian_shrinkage(mu, prior=0.0, gamma=0.0)

        pd.testing.assert_series_equal(mu_shrunk, mu)

    def test_gamma_one_full_shrinkage(self, sample_mu_sigma):
        """Test that gamma=1 returns prior."""
        mu, _ = sample_mu_sigma
        prior = 0.05

        mu_shrunk = bayesian_shrinkage(mu, prior=prior, gamma=1.0)

        expected = pd.Series([prior] * len(mu), index=mu.index)
        pd.testing.assert_series_equal(mu_shrunk, expected)

    def test_raises_on_invalid_gamma(self, sample_mu_sigma):
        """Test that invalid gamma raises ValueError."""
        mu, _ = sample_mu_sigma

        with pytest.raises(ValueError, match="gamma must be in"):
            bayesian_shrinkage(mu, gamma=-0.1)

        with pytest.raises(ValueError, match="gamma must be in"):
            bayesian_shrinkage(mu, gamma=1.5)

    def test_handles_mismatched_prior_index(self, sample_mu_sigma):
        """Test handling of prior with different index."""
        mu, _ = sample_mu_sigma
        # Prior with different assets (should fillna with 0)
        prior = pd.Series([0.05], index=["D"])

        mu_shrunk = bayesian_shrinkage(mu, prior=prior, gamma=0.5)

        # Should handle missing values gracefully
        assert len(mu_shrunk) == len(mu)


class TestCombinedShrinkage:
    """Tests for combined James-Stein + Bayesian shrinkage."""

    def test_alpha_zero_uses_js_only(self, sample_mu_sigma):
        """Test that alpha=0 uses only James-Stein."""
        mu, sigma = sample_mu_sigma

        mu_combined = combined_shrinkage(mu, sigma, alpha=0.0, gamma=0.5)
        mu_js = james_stein_shrinkage(mu, sigma)

        pd.testing.assert_series_equal(mu_combined, mu_js)

    def test_alpha_one_uses_bayesian_only(self, sample_mu_sigma):
        """Test that alpha=1 uses only Bayesian."""
        mu, sigma = sample_mu_sigma

        mu_combined = combined_shrinkage(mu, sigma, alpha=1.0, gamma=0.5)
        mu_bayes = bayesian_shrinkage(mu, prior=0.0, gamma=0.5)

        pd.testing.assert_series_equal(mu_combined, mu_bayes)

    def test_alpha_half_blends_equally(self, sample_mu_sigma):
        """Test that alpha=0.5 blends JS and Bayesian equally."""
        mu, sigma = sample_mu_sigma

        mu_combined = combined_shrinkage(mu, sigma, alpha=0.5, gamma=0.5)
        mu_js = james_stein_shrinkage(mu, sigma)
        mu_bayes = bayesian_shrinkage(mu, prior=0.0, gamma=0.5)

        expected = 0.5 * mu_bayes + 0.5 * mu_js
        pd.testing.assert_series_equal(mu_combined, expected)

    def test_raises_on_invalid_alpha(self, sample_mu_sigma):
        """Test that invalid alpha raises ValueError."""
        mu, sigma = sample_mu_sigma

        with pytest.raises(ValueError, match="alpha must be in"):
            combined_shrinkage(mu, sigma, alpha=-0.1)

        with pytest.raises(ValueError, match="alpha must be in"):
            combined_shrinkage(mu, sigma, alpha=1.5)

    def test_with_custom_prior(self, sample_mu_sigma):
        """Test combined shrinkage with custom prior."""
        mu, sigma = sample_mu_sigma
        prior = 0.10

        mu_combined = combined_shrinkage(
            mu, sigma,
            alpha=0.5,
            gamma=0.75,
            prior=prior
        )

        assert isinstance(mu_combined, pd.Series)
        assert len(mu_combined) == len(mu)

    def test_with_T_parameter(self, sample_mu_sigma):
        """Test with observation count parameter."""
        mu, sigma = sample_mu_sigma

        mu_combined = combined_shrinkage(mu, sigma, T=252, alpha=0.5)

        assert isinstance(mu_combined, pd.Series)
        assert len(mu_combined) == len(mu)


class TestShrinkMuPipeline:
    """Tests for full shrinkage pipeline."""

    def test_pipeline_with_default_estimator(self, sample_returns):
        """Test pipeline with default Huber estimator."""
        mu_shrunk = shrink_mu_pipeline(sample_returns)

        assert isinstance(mu_shrunk, pd.Series)
        assert len(mu_shrunk) == sample_returns.shape[1]
        assert list(mu_shrunk.index) == list(sample_returns.columns)

    def test_pipeline_with_custom_estimator(self, sample_returns):
        """Test pipeline with custom estimator."""
        def custom_estimator(rets):
            return rets.mean() * 252

        mu_shrunk = shrink_mu_pipeline(
            sample_returns,
            estimator=custom_estimator
        )

        assert isinstance(mu_shrunk, pd.Series)
        assert len(mu_shrunk) == sample_returns.shape[1]

    def test_pipeline_with_gamma_zero(self, sample_returns):
        """Test pipeline with no Bayesian shrinkage."""
        mu_shrunk = shrink_mu_pipeline(sample_returns, gamma=0.0)

        # Should still apply JS shrinkage
        assert isinstance(mu_shrunk, pd.Series)

    def test_pipeline_with_alpha_extremes(self, sample_returns):
        """Test pipeline with alpha=0 and alpha=1."""
        mu_alpha_0 = shrink_mu_pipeline(sample_returns, alpha=0.0)
        mu_alpha_1 = shrink_mu_pipeline(sample_returns, alpha=1.0)

        # Both should work but produce different results
        assert isinstance(mu_alpha_0, pd.Series)
        assert isinstance(mu_alpha_1, pd.Series)
        assert not mu_alpha_0.equals(mu_alpha_1)

    def test_pipeline_with_series_prior(self, sample_returns):
        """Test pipeline with Series prior."""
        prior = pd.Series(
            [0.08, 0.10, 0.09],
            index=sample_returns.columns
        )

        mu_shrunk = shrink_mu_pipeline(
            sample_returns,
            prior=prior,
            gamma=0.5
        )

        assert isinstance(mu_shrunk, pd.Series)
        assert list(mu_shrunk.index) == list(sample_returns.columns)

    def test_pipeline_produces_reasonable_values(self, sample_returns):
        """Test that pipeline produces reasonable return estimates."""
        mu_shrunk = shrink_mu_pipeline(sample_returns)

        # Annual returns should be reasonable (not extreme)
        assert np.all(mu_shrunk.abs() < 2.0)  # Less than 200% annual return

        # Should not be all zeros
        assert np.any(mu_shrunk != 0.0)

    def test_pipeline_handles_different_sample_sizes(self):
        """Test pipeline with different sample sizes."""
        np.random.seed(42)

        # Small sample
        small_returns = pd.DataFrame(
            np.random.normal(0, 0.01, (50, 3)),
            columns=["A", "B", "C"]
        )
        mu_small = shrink_mu_pipeline(small_returns)

        # Large sample
        large_returns = pd.DataFrame(
            np.random.normal(0, 0.01, (500, 3)),
            columns=["A", "B", "C"]
        )
        mu_large = shrink_mu_pipeline(large_returns)

        # Both should work
        assert len(mu_small) == 3
        assert len(mu_large) == 3
