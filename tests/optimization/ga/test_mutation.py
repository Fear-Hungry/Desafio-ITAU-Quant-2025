from __future__ import annotations

import numpy as np
from itau_quant.optimization.ga import mutation
from itau_quant.optimization.ga.population import Individual


def test_flip_asset_selection_changes_mask() -> None:
    individual = Individual(np.array([1, 0, 0], dtype=bool), {})
    rng = np.random.default_rng(1)
    mutated = mutation.flip_asset_selection(individual, prob=1.0, rng=rng, min_assets=1)
    assert mutated.assets_mask.any()
    assert mutated.metadata.get("last_mutation") == "flip"


def test_gaussian_jitter_respects_bounds() -> None:
    individual = Individual(np.array([1, 1, 0], dtype=bool), {"lambda": 1.0})
    rng = np.random.default_rng(3)
    mutated = mutation.gaussian_jitter_params(
        individual, sigma=0.5, bounds={"lambda": (0.5, 2.0)}, rng=rng
    )
    assert 0.5 <= mutated.params["lambda"] <= 2.0


def test_mutation_pipeline_combines_operations() -> None:
    individual = Individual(np.array([1, 1, 0, 0], dtype=bool), {"lambda": 1.5, "k": 3})
    config = {
        "flip_prob": 0.5,
        "gaussian": {"sigma": 0.1, "bounds": {"lambda": (1.0, 2.0)}},
        "discrete": [{"param": "k", "values": [3, 4, 5], "prob": 1.0}],
        "swap_prob": 0.5,
        "swap_count": 1,
        "universe": ["A", "B", "C", "D"],
        "cardinality": {"min": 1, "max": 3},
    }
    rng = np.random.default_rng(5)
    mutated = mutation.mutation_pipeline(individual, config, rng)
    assert 1 <= mutated.cardinality() <= 3
