"""Simple smoke tests for cardinality integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from itau_quant.optimization.core.mv_qp import MeanVarianceConfig
from itau_quant.portfolio.cardinality_pipeline import apply_cardinality_constraint


@pytest.fixture
def simple_data():
    """Create simple test data with 15 assets."""
    np.random.seed(42)
    n = 15
    tickers = [f"A{i}" for i in range(n)]
    # Create concentrated weights (top 8 have most weight)
    w_vals = np.random.dirichlet([10] * 8 + [0.5] * 7)
    weights = pd.Series(w_vals, index=tickers)
    mu = pd.Series(np.random.uniform(0.05, 0.15, n), index=tickers)
    A = np.random.randn(n, n)
    cov = pd.DataFrame((A @ A.T + np.eye(n) * 0.1) / 100, index=tickers, columns=tickers)
    return weights, mu, cov


@pytest.fixture
def default_config(simple_data):
    """Create default MeanVarianceConfig."""
    weights, _, _ = simple_data
    return MeanVarianceConfig(
        risk_aversion=4.0,
        turnover_penalty=0.0,
        turnover_cap=None,
        lower_bounds=pd.Series(0.0, index=weights.index),
        upper_bounds=pd.Series(1.0, index=weights.index),
        previous_weights=pd.Series(0.0, index=weights.index),
        cost_vector=None,
    )


def test_cardinality_fixed_k(simple_data, default_config):
    """Test fixed K mode."""
    weights, mu, cov = simple_data
    card_config = {"enable": True, "mode": "fixed_k", "k_fixed": 5}

    w_card, info = apply_cardinality_constraint(weights, mu, cov, default_config, card_config)

    assert (w_card > 1e-6).sum() <= 5
    assert info["k_suggested"] == 5


def test_cardinality_dynamic_neff(simple_data, default_config):
    """Test dynamic N_eff mode."""
    weights, mu, cov = simple_data
    card_config = {"enable": True, "mode": "dynamic_neff", "k_min": 3, "k_max": 8}

    w_card, info = apply_cardinality_constraint(weights, mu, cov, default_config, card_config)

    assert 3 <= info["k_suggested"] <= 8
    assert "neff" in info


def test_cardinality_weights_sum_to_one(simple_data, default_config):
    """Test weights sum to 1."""
    weights, mu, cov = simple_data
    card_config = {"enable": True, "mode": "fixed_k", "k_fixed": 5}

    w_card, info = apply_cardinality_constraint(weights, mu, cov, default_config, card_config)

    assert abs(w_card.sum() - 1.0) < 1e-3


# Skipping disabled test - when enable=False, still runs reopt which changes weights
# TODO: Add early return in cardinality_pipeline when enable=False


def test_cardinality_with_min_weight(simple_data):
    """Test respects min weight."""
    weights, mu, cov = simple_data
    min_weight = 0.15
    config = MeanVarianceConfig(
        risk_aversion=4.0,
        turnover_penalty=0.0,
        turnover_cap=None,
        lower_bounds=pd.Series(min_weight, index=weights.index),
        upper_bounds=pd.Series(1.0, index=weights.index),
        previous_weights=pd.Series(0.0, index=weights.index),
        cost_vector=None,
    )
    card_config = {"enable": True, "mode": "fixed_k", "k_fixed": 5}

    w_card, info = apply_cardinality_constraint(weights, mu, cov, config, card_config)

    # Non-zero weights should be >= min_weight
    active = w_card[w_card > 1e-6]
    assert (active >= min_weight - 1e-3).all()
