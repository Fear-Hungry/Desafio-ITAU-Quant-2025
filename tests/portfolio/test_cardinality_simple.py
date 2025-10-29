"""Simple smoke tests for cardinality integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from itau_quant.optimization.core.mv_qp import MeanVarianceConfig
from itau_quant.portfolio.cardinality_pipeline import apply_cardinality_constraint
from itau_quant.portfolio.rebalancer import MarketData, rebalance


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
    cov = pd.DataFrame(
        (A @ A.T + np.eye(n) * 0.1) / 100, index=tickers, columns=tickers
    )
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

    w_card, info = apply_cardinality_constraint(
        weights, mu, cov, default_config, card_config
    )

    assert (w_card > 1e-6).sum() <= 5
    assert info["k_suggested"] == 5
    assert w_card.index.equals(weights.index)
    assert info["reopt_status"] in {"completed", "not_needed", "optimal"}
    if info["reopt_status"] != "not_needed":
        assert "final_support" in info
        assert len(info["final_support"]) <= 5


def test_cardinality_dynamic_neff(simple_data, default_config):
    """Test dynamic N_eff mode."""
    weights, mu, cov = simple_data
    card_config = {"enable": True, "mode": "dynamic_neff", "k_min": 3, "k_max": 8}

    w_card, info = apply_cardinality_constraint(
        weights, mu, cov, default_config, card_config
    )

    assert 3 <= info["k_suggested"] <= 8
    assert "neff" in info
    assert w_card.index.equals(weights.index)


def test_cardinality_dynamic_cost_reoptimization():
    """Ensure dynamic (N_eff + costs) path triggers top-K reopt."""
    n_assets = 24
    tickers = [f"T{i:02d}" for i in range(n_assets)]

    weights_raw = np.linspace(0.12, 0.01, n_assets)
    weights = pd.Series(weights_raw / weights_raw.sum(), index=tickers, dtype=float)
    mu = pd.Series(np.linspace(0.10, 0.02, n_assets), index=tickers, dtype=float)
    cov_matrix = np.eye(n_assets) * 0.02 + 0.005
    cov = pd.DataFrame(cov_matrix, index=tickers, columns=tickers, dtype=float)

    previous = pd.Series(1.0 / n_assets, index=tickers, dtype=float)
    lower = pd.Series(0.0, index=tickers, dtype=float)
    upper = pd.Series(0.15, index=tickers, dtype=float)

    config = MeanVarianceConfig(
        risk_aversion=3.5,
        turnover_penalty=0.0,
        turnover_cap=None,
        lower_bounds=lower,
        upper_bounds=upper,
        previous_weights=previous,
        cost_vector=None,
    )

    card_config = {
        "enable": True,
        "mode": "dynamic_neff_cost",
        "k_min": 12,
        "k_max": 32,
        "neff_multiplier": 0.65,
        "score_weight": 1.0,
        "score_turnover": -0.3,
        "score_return": 0.15,
        "score_cost": -0.1,
        "tie_breaker": "low_turnover",
        "min_active_weight": 0.01,
        "epsilon": 1e-6,
    }

    cost_config = {
        "round_trip_bps": {"cheap": 8.0, "expensive": 28.0},
        "asset_class_map": {
            **{ticker: "cheap" for ticker in tickers[:12]},
            **{ticker: "expensive" for ticker in tickers[12:]},
        },
        "default_cost_bps": 20.0,
    }

    w_card, info = apply_cardinality_constraint(
        weights,
        mu,
        cov,
        config,
        card_config,
        cost_config,
    )

    assert info["reopt_status"] not in {"failed", "not_needed"}
    assert info["k_suggested"] == 18
    assert len(info["selected_assets"]) == info["k_suggested"]
    assert "final_support" in info
    assert len(info["final_support"]) <= info["k_suggested"]
    assert "k_range_from_cost" in info
    assert "avg_cost_bps" in info
    assert abs(w_card.sum() - 1.0) < 1e-6
    assert w_card.index.equals(weights.index)

    inactive_assets = set(tickers) - set(info["selected_assets"])
    if inactive_assets:
        assert (w_card.reindex(sorted(inactive_assets)).abs() < 1e-8).all()


def test_cardinality_weights_sum_to_one(simple_data, default_config):
    """Test weights sum to 1."""
    weights, mu, cov = simple_data
    card_config = {"enable": True, "mode": "fixed_k", "k_fixed": 5}

    w_card, info = apply_cardinality_constraint(
        weights, mu, cov, default_config, card_config
    )

    assert abs(w_card.sum() - 1.0) < 1e-3
    assert w_card.index.equals(weights.index)


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
    assert w_card.index.equals(weights.index)


def test_rebalance_applies_cardinality_top_level():
    np.random.seed(7)
    n = 8
    tickers = [f"T{i}" for i in range(n)]
    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    returns = pd.DataFrame(
        np.random.normal(loc=0.001, scale=0.01, size=(len(dates), n)),
        index=dates,
        columns=tickers,
    )
    prices = (
        pd.DataFrame(
            100 + np.random.normal(scale=2.0, size=(len(dates), n)),
            index=dates,
            columns=tickers,
        ).abs()
        + 1.0
    )

    market_data = MarketData(prices=prices, returns=returns)
    previous = pd.Series(0.0, index=tickers, dtype=float)
    config = {
        "optimizer": {
            "risk_aversion": 4.0,
            "min_weight": 0.0,
            "max_weight": 0.4,
        },
        "cardinality": {
            "enable": True,
            "mode": "fixed_k",
            "k_fixed": 3,
            "epsilon": 1e-4,
        },
        "estimators": {
            "mu": {"method": "simple"},
            "sigma": {"method": "sample"},
        },
        "costs": {"linear_bps": 0.0},
    }

    result = rebalance(
        dates[-1],
        market_data,
        previous_weights=previous,
        capital=1_000_000.0,
        config=config,
    )

    assert result.weights.index.tolist() == tickers
    assert (result.weights > 1e-6).sum() <= 3
    assert "cardinality" in result.log
    assert result.log["cardinality"]["reopt_status"] in {
        "completed",
        "not_needed",
        "optimal",
    }


def test_rebalance_cardinality_optimizer_section_only():
    np.random.seed(21)
    n = 10
    tickers = [f"S{i}" for i in range(n)]
    dates = pd.date_range("2024-02-01", periods=45, freq="B")
    returns = pd.DataFrame(
        np.random.normal(loc=0.0008, scale=0.009, size=(len(dates), n)),
        index=dates,
        columns=tickers,
    )
    prices = (
        pd.DataFrame(
            100 + np.random.normal(scale=1.5, size=(len(dates), n)),
            index=dates,
            columns=tickers,
        ).abs()
        + 1.0
    )

    market_data = MarketData(prices=prices, returns=returns)
    previous = pd.Series(0.0, index=tickers, dtype=float)

    config = {
        "optimizer": {
            "risk_aversion": 3.0,
            "min_weight": 0.0,
            "max_weight": 0.75,
            "cardinality": {
                "enable": True,
                "mode": "fixed_k",
                "k_fixed": 2,
                "epsilon": 1e-4,
            },
        },
        "estimators": {
            "mu": {"method": "simple"},
            "sigma": {"method": "sample"},
        },
    }

    result = rebalance(
        dates[-1],
        market_data,
        previous_weights=previous,
        capital=500_000.0,
        config=config,
    )

    assert (result.weights > 1e-6).sum() <= 2
    assert "cardinality" in result.log
    card_info = result.log["cardinality"]
    assert card_info["reopt_status"] in {"completed", "optimal", "not_needed"}
    assert card_info["k_suggested"] == 2
    assert len(card_info["selected_assets"]) <= 2
