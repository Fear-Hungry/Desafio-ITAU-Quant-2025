from __future__ import annotations

import numpy as np
import pandas as pd

from itau_quant.optimization.ga import population as pop


def test_encode_decode_roundtrip() -> None:
    universe = ["AAA", "BBB", "CCC", "DDD"]
    original_assets = ["AAA", "DDD"]
    params = {"lambda_risk": 3.0}

    individual = pop.encode_individual(original_assets, params, universe)
    decoded_assets, decoded_params = pop.decode_individual(individual, universe)

    assert set(decoded_assets) == set(original_assets)
    assert decoded_params == params


def test_random_individual_respects_cardinality() -> None:
    universe = ["A", "B", "C", "D", "E"]
    config = {"cardinality": {"min": 2, "max": 3}, "hyperparams": {"eta": {"min": 0.1, "max": 0.5}}}
    rng = np.random.default_rng(42)

    individual = pop.random_individual(universe, config, rng)

    assert 2 <= individual.cardinality() <= 3
    assert "eta" in individual.params


def test_diversified_population_ensures_variety() -> None:
    universe = ["A", "B", "C", "D", "E", "F"]
    config = {"cardinality": {"min": 2, "max": 4}, "diversity": {"min_jaccard": 0.3}}
    rng = np.random.default_rng(123)

    population = pop.diversified_population(universe, config, size=6, rng=rng)

    masks = [tuple(ind.assets_mask) for ind in population]
    assert len(set(masks)) >= 3


def test_warm_start_population_creates_expected_masks() -> None:
    universe = ["AAA", "BBB", "CCC"]
    weights = pd.DataFrame(
        [[0.6, 0.3, 0.1], [0.0, 0.5, 0.5]],
        columns=universe,
    )
    individuals = pop.warm_start_population(universe, weights, {"warm_threshold": 0.05})

    assert len(individuals) == 2
    assert individuals[0].cardinality() == 3
    assert individuals[1].cardinality() == 2


def test_ensure_feasible_enforces_bounds() -> None:
    universe = ["A", "B", "C", "D"]
    mask = np.array([True, True, True, True])
    individual = pop.Individual(mask, {"lambda": 5.0})
    rng = np.random.default_rng(7)

    feasible = pop.ensure_feasible(
        individual,
        {"cardinality": {"min": 1, "max": 2}, "bounds": {"lambda": (1.0, 3.0)}},
        rng=rng,
    )

    assert feasible.cardinality() == 2
    assert 1.0 <= feasible.params["lambda"] <= 3.0
