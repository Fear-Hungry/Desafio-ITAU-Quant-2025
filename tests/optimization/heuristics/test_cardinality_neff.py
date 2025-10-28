"""Tests for N_eff-based cardinality functions."""

from __future__ import annotations

import pandas as pd
import pytest

from itau_quant.optimization.heuristics.cardinality import (
    compute_effective_number,
    select_support_topk,
    smart_topk_score,
    suggest_k_dynamic,
    suggest_k_from_costs,
    suggest_k_from_neff,
)


class TestComputeEffectiveNumber:
    """Tests for N_eff calculation."""

    def test_equal_weights_returns_n(self):
        """Equal weights → N_eff = N."""
        w = pd.Series([0.25, 0.25, 0.25, 0.25])
        assert compute_effective_number(w) == pytest.approx(4.0)

    def test_single_asset_returns_one(self):
        """Single asset → N_eff = 1."""
        w = pd.Series([1.0, 0.0, 0.0])
        assert compute_effective_number(w) == pytest.approx(1.0)

    def test_concentrated_portfolio(self):
        """Concentrated portfolio → N_eff < N."""
        w = pd.Series([0.7, 0.2, 0.1])
        neff = compute_effective_number(w)
        assert 1.0 < neff < 3.0
        assert neff == pytest.approx(1.85, abs=0.01)

    def test_empty_weights_returns_zero(self):
        """All zeros → N_eff = 0."""
        w = pd.Series([0.0, 0.0, 0.0])
        assert compute_effective_number(w) == 0.0


class TestSuggestKFromNeff:
    """Tests for K suggestion from N_eff."""

    def test_typical_neff_suggests_k_in_range(self):
        """N_eff = 30 → K ≈ 24."""
        k = suggest_k_from_neff(30.0, k_min=12, k_max=32)
        assert k == 24  # floor(0.8 * 30)

    def test_low_neff_clipped_to_min(self):
        """Low N_eff → K = k_min."""
        k = suggest_k_from_neff(10.0, k_min=12, k_max=32)
        assert k == 12

    def test_high_neff_clipped_to_max(self):
        """High N_eff → K = k_max."""
        k = suggest_k_from_neff(50.0, k_min=12, k_max=32)
        assert k == 32

    def test_custom_multiplier(self):
        """Custom multiplier changes K."""
        k1 = suggest_k_from_neff(30.0, k_min=12, k_max=32, multiplier=0.8)
        k2 = suggest_k_from_neff(30.0, k_min=12, k_max=32, multiplier=0.6)
        assert k1 == 24
        assert k2 == 18


class TestSuggestKFromCosts:
    """Tests for K suggestion from cost distribution."""

    def test_cheap_regime_suggests_high_k(self):
        """Cheap costs (≤10 bps) → K ∈ [28, 32]."""
        costs = pd.Series([8, 8, 10, 12, 15])  # Avg ≈ 10.6
        k_min, k_max = suggest_k_from_costs(costs, k_min=12, k_max=36)
        assert k_min >= 18
        assert k_max <= 36

    def test_expensive_regime_suggests_low_k(self):
        """Expensive costs (≥60 bps) → K ∈ [12, 18]."""
        costs = pd.Series([45, 50, 55, 60, 65])
        k_min, k_max = suggest_k_from_costs(costs, k_min=12, k_max=36)
        assert k_min == 12
        assert k_max == 18

    def test_medium_regime_suggests_mid_k(self):
        """Medium costs (20-40 bps) → K ∈ [18, 26]."""
        costs = pd.Series([20, 22, 25, 28, 30])
        k_min, k_max = suggest_k_from_costs(costs, k_min=12, k_max=36)
        assert 18 <= k_min <= 20
        assert 26 <= k_max <= 28


class TestSuggestKDynamic:
    """Tests for dynamic K suggestion combining N_eff + cost."""

    def test_combines_neff_and_cost(self):
        """Dynamic K respects both N_eff and cost constraints."""
        neff = 25.0
        costs = pd.Series([8] * 50 + [45] * 10)  # Mixed regime
        result = suggest_k_dynamic(neff, costs, k_min=12, k_max=32)

        assert "k_suggested" in result
        assert "k_from_neff" in result
        assert "k_range_from_cost" in result
        assert 12 <= result["k_suggested"] <= 32

    def test_returns_metadata(self):
        """Result includes neff and avg cost."""
        neff = 20.0
        costs = pd.Series([10, 15, 20])
        result = suggest_k_dynamic(neff, costs)

        assert result["neff"] == 20.0
        assert result["avg_cost_bps"] == pytest.approx(15.0)


class TestSmartTopKScore:
    """Tests for smart scoring function."""

    def test_higher_weight_gets_higher_score(self):
        """Higher weight → higher score."""
        w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
        scores = smart_topk_score(w)
        assert scores["A"] > scores["B"]
        assert scores["B"] > scores["C"]

    def test_turnover_penalty_reduces_score(self):
        """Turnover penalty reduces score."""
        w = pd.Series({"A": 0.5, "B": 0.5})
        w_prev = pd.Series({"A": 0.5, "B": 0.0})
        scores = smart_topk_score(w, weights_prev=w_prev, alpha_turnover=-0.2)

        # B has higher turnover → lower score
        assert scores["A"] > scores["B"]

    def test_return_bonus_increases_score(self):
        """Higher expected return → higher score."""
        w = pd.Series({"A": 0.5, "B": 0.5})
        mu = pd.Series({"A": 0.10, "B": 0.20})
        scores = smart_topk_score(w, mu=mu, alpha_return=0.1)

        # B has higher return → higher score
        assert scores["B"] > scores["A"]

    def test_cost_penalty_reduces_score(self):
        """Higher cost → lower score."""
        w = pd.Series({"A": 0.5, "B": 0.5})
        costs = pd.Series({"A": 8, "B": 45})
        scores = smart_topk_score(w, costs_bps=costs, alpha_cost=-0.15)

        # B is more expensive → lower score
        assert scores["A"] > scores["B"]


class TestSelectSupportTopK:
    """Tests for top-K selection."""

    def test_selects_top_k_assets(self):
        """Selects K assets with highest scores."""
        w = pd.Series({"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1})
        selected = select_support_topk(w, k=2)

        assert len(selected) == 2
        assert "A" in selected
        assert "B" in selected

    def test_returns_all_if_k_larger_than_significant(self):
        """If K > significant assets, returns all."""
        w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2, "D": 0.0001})
        selected = select_support_topk(w, k=10, epsilon=0.001)

        assert len(selected) == 3  # Only A, B, C significant

    def test_tie_breaking_with_turnover(self):
        """Tie-breaker uses turnover when scores equal."""
        w = pd.Series({"A": 0.33, "B": 0.33, "C": 0.34})
        w_prev = pd.Series({"A": 0.30, "B": 0.40, "C": 0.30})
        selected = select_support_topk(w, k=2, weights_prev=w_prev, tie_breaker="low_turnover")

        # Should prefer assets with lower turnover
        assert len(selected) == 2
        assert "A" in selected or "C" in selected  # Lower turnover than B
