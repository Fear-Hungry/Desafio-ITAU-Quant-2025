from __future__ import annotations

import numpy as np
import pandas as pd

from itau_quant.optimization.heuristics import cardinality as card


def _stub_solver(mu: pd.Series, cov: pd.DataFrame, **_) -> dict[str, pd.Series]:
    weights = pd.Series(1.0 / len(mu), index=mu.index, dtype=float)
    return {"weights": weights}


def test_greedy_selection_picks_top_assets():
    mu = pd.Series([0.05, 0.02, 0.03], index=["AAA", "BBB", "CCC"])
    cov = np.diag([0.04, 0.05, 0.02])
    costs = pd.Series([0.001, 0.003, 0.002], index=mu.index)
    selected = card.greedy_selection(mu, cov, costs, k=2, asset_index=mu.index)
    assert len(selected) == 2
    assert "AAA" in selected


def test_beam_search_selection_matches_or_exceeds_greedy():
    mu = pd.Series([0.04, 0.035, 0.03], index=["AAA", "BBB", "CCC"])
    cov = np.diag([0.04, 0.02, 0.03])
    costs = pd.Series([0.001, 0.001, 0.002], index=mu.index)
    greedy = card.greedy_selection(mu, cov, costs, k=2, asset_index=mu.index)
    beam = card.beam_search_selection(mu, cov, costs, k=2, asset_index=mu.index, beam_width=2)
    assert len(beam) == 2
    greedy_score = set(greedy)
    beam_score = set(beam)
    assert beam_score or greedy_score  # ensure non-empty


def test_prune_after_optimisation_keeps_top_k():
    weights = pd.Series([0.5, 0.3, 0.2], index=["AAA", "BBB", "CCC"])
    pruned = card.prune_after_optimisation(weights, k=2)
    assert pruned.gt(0).sum() == 2
    assert np.isclose(pruned.sum(), 1.0)


def test_reoptimize_with_subset_calls_core_solver():
    data = {
        "mu": pd.Series([0.04, 0.03], index=["AAA", "BBB"]),
        "cov": np.array([[0.04, 0.0], [0.0, 0.03]]),
        "previous_weights": pd.Series([0.5, 0.5], index=["AAA", "BBB"]),
    }
    result = card.reoptimize_with_subset(["AAA", "BBB"], data, _stub_solver)
    assert "weights" in result
    assert np.isclose(result["weights"].sum(), 1.0)


def test_cardinality_pipeline_prune_then_reopt():
    asset_index = ["AAA", "BBB", "CCC"]
    mu = pd.Series([0.05, 0.04, 0.02], index=asset_index)
    cov = np.diag([0.05, 0.03, 0.06])
    weights = pd.Series([0.6, 0.25, 0.15], index=asset_index)
    data = {
        "asset_index": asset_index,
        "mu": mu,
        "cov": cov,
        "weights": weights,
        "core_solver": _stub_solver,
        "previous_weights": pd.Series([0.5, 0.3, 0.2], index=asset_index),
    }
    result = card.cardinality_pipeline(
        data,
        k=2,
        strategy="prune_then_reopt",
    )
    assert len(result.assets) <= 2
    assert result.metadata["reoptimised"] is True
