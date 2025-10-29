from __future__ import annotations

import numpy as np

from itau_quant.optimization.ga.population import Individual
from itau_quant.optimization.ga import selection


def _population() -> list[Individual]:
    masks = [
        np.array([1, 0, 0, 0], dtype=bool),
        np.array([1, 1, 0, 0], dtype=bool),
        np.array([1, 1, 1, 0], dtype=bool),
    ]
    return [Individual(mask, {"score": idx}) for idx, mask in enumerate(masks)]


def test_tournament_selection_prefers_high_fitness() -> None:
    pop = _population()
    fitness = [0.1, 0.5, 1.0]
    rng = np.random.default_rng(123)
    selected = selection.tournament_selection(pop, fitness, tournament_size=3, rng=rng)
    assert selected is pop[-1]


def test_roulette_wheel_selection_bias() -> None:
    pop = _population()
    fitness = [0.1, 0.2, 1.0]
    rng = np.random.default_rng(4)
    id_map = {id(ind): idx for idx, ind in enumerate(pop)}
    counts = np.zeros(len(pop), dtype=int)
    for _ in range(200):
        chosen = selection.roulette_wheel_selection(pop, fitness, rng)
        counts[id_map[id(chosen)]] += 1
    assert counts[-1] > counts[0]


def test_selection_pipeline_multiple_methods() -> None:
    pop = _population()
    fitness = [0.1, 0.5, 1.0]
    rng = np.random.default_rng(9)
    parents = selection.selection_pipeline(pop, fitness, {"method": ["tournament", "roulette"], "tournament_size": 2}, rng, num_parents=2)
    assert len(parents) == 2
