"""Integration tests for cardinality in portfolio rebalancing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from itau_quant.optimization.core.mv_qp import MeanVarianceConfig
from itau_quant.portfolio.cardinality_pipeline import apply_cardinality_constraint


class TestCardinalityPipeline:
    """Tests for cardinality pipeline integration."""

    @pytest.fixture
    def sample_portfolio(self):
        """Create sample portfolio data."""
        np.random.seed(42)
        n = 20
        tickers = [f"ASSET{i}" for i in range(n)]

        # Unconstrained weights (10 significant)
        weights = pd.Series(np.random.dirichlet([1] * 10).tolist() + [0.0] * 10, index=tickers)

        # Expected returns
        mu = pd.Series(np.random.uniform(0.05, 0.15, n), index=tickers)

        # Covariance (random positive definite)
        A = np.random.randn(n, n)
        cov = pd.DataFrame((A @ A.T + np.eye(n) * 0.1) / 100, index=tickers, columns=tickers)

        return weights, mu, cov

    def test_fixed_k_reduces_to_k_assets(self, sample_portfolio):
        """Fixed K mode selects exactly K assets."""
        weights, mu, cov = sample_portfolio
        config = MeanVarianceConfig(risk_aversion=4.0)
        card_config = {"enable": True, "mode": "fixed_k", "k_fixed": 5}

        w_card, info = apply_cardinality_constraint(weights, mu, cov, config, card_config)

        # Should have exactly 5 non-zero weights
        assert (w_card > 1e-6).sum() == 5
        assert info["k_suggested"] == 5
        assert "selected_assets" in info

    def test_dynamic_neff_mode_calibrates_k(self, sample_portfolio):
        """Dynamic N_eff mode calibrates K based on effective number."""
        weights, mu, cov = sample_portfolio
        config = MeanVarianceConfig(risk_aversion=4.0)
        card_config = {"enable": True, "mode": "dynamic_neff", "k_min": 3, "k_max": 15}

        w_card, info = apply_cardinality_constraint(weights, mu, cov, config, card_config)

        # K should be between min and max
        assert 3 <= info["k_suggested"] <= 15
        assert "neff" in info
        assert (w_card > 1e-6).sum() <= info["k_suggested"]

    def test_dynamic_neff_cost_uses_cost_model(self, sample_portfolio):
        """Dynamic N_eff + cost mode uses cost information."""
        weights, mu, cov = sample_portfolio
        config = MeanVarianceConfig(risk_aversion=4.0)
        card_config = {
            "enable": True,
            "mode": "dynamic_neff_cost",
            "k_min": 3,
            "k_max": 15,
        }
        cost_config = {
            "round_trip_bps": {"us_equities_core": 8, "crypto_btc": 45},
            "default_cost_bps": 15.0,
        }

        w_card, info = apply_cardinality_constraint(
            weights, mu, cov, config, card_config, cost_config
        )

        # Should have cost-based K range
        assert "k_range_from_cost" in info
        assert isinstance(info["k_range_from_cost"], tuple)

    def test_reoptimization_status_reported(self, sample_portfolio):
        """Reoptimization status is reported in info."""
        weights, mu, cov = sample_portfolio
        config = MeanVarianceConfig(risk_aversion=4.0)
        card_config = {"enable": True, "mode": "fixed_k", "k_fixed": 5}

        w_card, info = apply_cardinality_constraint(weights, mu, cov, config, card_config)

        assert "reopt_status" in info
        # Should either succeed or report failure
        assert info["reopt_status"] in ["optimal", "optimal_inaccurate", "failed", "not_needed"]

    def test_weights_sum_to_one(self, sample_portfolio):
        """Final weights sum to 1."""
        weights, mu, cov = sample_portfolio
        config = MeanVarianceConfig(risk_aversion=4.0)
        card_config = {"enable": True, "mode": "fixed_k", "k_fixed": 8}

        w_card, info = apply_cardinality_constraint(weights, mu, cov, config, card_config)

        assert w_card.sum() == pytest.approx(1.0, abs=1e-4)

    def test_respects_min_weight(self, sample_portfolio):
        """Respects minimum weight floor."""
        weights, mu, cov = sample_portfolio
        min_weight = 0.05
        lower_bounds = pd.Series(min_weight, index=weights.index)
        config = MeanVarianceConfig(risk_aversion=4.0, lower_bounds=lower_bounds)
        card_config = {"enable": True, "mode": "fixed_k", "k_fixed": 5}

        w_card, info = apply_cardinality_constraint(weights, mu, cov, config, card_config)

        # Non-zero weights should be >= min_weight
        active = w_card[w_card > 1e-6]
        assert (active >= min_weight - 1e-4).all()

    def test_fallback_on_reopt_failure(self, sample_portfolio):
        """Falls back to unconstrained if reopt fails."""
        weights, mu, cov = sample_portfolio

        # Create infeasible config (min_weight too high for K)
        min_weight = 0.5  # 5 assets × 0.5 = 2.5 > 1 (infeasible)
        lower_bounds = pd.Series(min_weight, index=weights.index)
        config = MeanVarianceConfig(risk_aversion=4.0, lower_bounds=lower_bounds)
        card_config = {"enable": True, "mode": "fixed_k", "k_fixed": 5}

        w_card, info = apply_cardinality_constraint(weights, mu, cov, config, card_config)

        # Should report failure but return something
        assert info["reopt_status"] in ["failed", "infeasible", "not_needed"]
        assert w_card.sum() > 0  # Should return fallback

    def test_previous_weights_considered_in_scoring(self, sample_portfolio):
        """Previous weights affect scoring via turnover penalty."""
        weights, mu, cov = sample_portfolio

        # Set previous weights
        prev_weights = pd.Series(0.0, index=weights.index)
        prev_weights.iloc[:5] = 0.2  # Previous portfolio had first 5 assets

        config = MeanVarianceConfig(risk_aversion=4.0, previous_weights=prev_weights)
        card_config = {
            "enable": True,
            "mode": "fixed_k",
            "k_fixed": 5,
            "score_turnover": -0.5,  # Strong turnover penalty
        }

        w_card, info = apply_cardinality_constraint(weights, mu, cov, config, card_config)

        # With turnover penalty, should prefer previous assets
        selected = info["selected_assets"]
        # At least some overlap expected (not deterministic due to weights)
        assert len(set(selected) & set(prev_weights[prev_weights > 0].index)) >= 2

    def test_neff_reported_correctly(self, sample_portfolio):
        """N_eff calculation is reported."""
        weights, mu, cov = sample_portfolio
        config = MeanVarianceConfig(risk_aversion=4.0)
        card_config = {"enable": True, "mode": "fixed_k", "k_fixed": 5}

        w_card, info = apply_cardinality_constraint(weights, mu, cov, config, card_config)

        # N_eff should be positive and less than total assets
        assert 0 < info["neff"] <= len(weights)


class TestCardinalityDisabled:
    """Tests when cardinality is disabled."""

    def test_disabled_returns_original_weights(self):
        """When disabled, returns original weights unchanged."""
        np.random.seed(42)
        n = 10
        tickers = [f"ASSET{i}" for i in range(n)]
        weights = pd.Series(np.random.dirichlet([1] * n), index=tickers)
        mu = pd.Series(np.random.uniform(0.05, 0.15, n), index=tickers)
        A = np.random.randn(n, n)
        cov = pd.DataFrame((A @ A.T + np.eye(n) * 0.1) / 100, index=tickers, columns=tickers)

        config = MeanVarianceConfig(risk_aversion=4.0)
        card_config = {"enable": False}  # Disabled

        w_card, info = apply_cardinality_constraint(weights, mu, cov, config, card_config)

        # Should return original weights
        pd.testing.assert_series_equal(w_card, weights)
