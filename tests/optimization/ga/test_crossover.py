from __future__ import annotations

import numpy as np
from arara_quant.optimization.ga import crossover
from arara_quant.optimization.ga.population import Individual


def _parents() -> tuple[Individual, Individual]:
    parent_a = Individual(np.array([1, 1, 0, 0], dtype=bool), {"lambda": 1.0})
    parent_b = Individual(np.array([0, 1, 1, 0], dtype=bool), {"lambda": 2.0})
    return parent_a, parent_b


def test_single_point_crossover_generates_children() -> None:
    parent_a, parent_b = _parents()
    rng = np.random.default_rng(7)
    child_a, child_b = crossover.single_point_crossover(parent_a, parent_b, rng)
    assert child_a.assets_mask.size == parent_a.assets_mask.size
    assert child_b.assets_mask.size == parent_b.assets_mask.size


def test_crossover_factory_with_uniform_method() -> None:
    parent_a, parent_b = _parents()
    rng = np.random.default_rng(11)
    operator = crossover.crossover_factory({"method": "uniform", "prob": 0.6})
    child_a, child_b = operator(parent_a, parent_b, rng)
    assert child_a.assets_mask.shape == parent_a.assets_mask.shape
    assert child_b.assets_mask.shape == parent_b.assets_mask.shape
