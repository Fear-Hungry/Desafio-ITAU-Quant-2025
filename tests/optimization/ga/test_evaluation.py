from __future__ import annotations

import numpy as np
import pandas as pd

from itau_quant.optimization.ga.population import Individual
from itau_quant.optimization.ga import evaluation


def _core_solver(candidate: dict[str, object]) -> dict[str, object]:
    assets = candidate["assets"]
    scores = {"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4}
    value = float(np.mean([scores[a] for a in assets]))
    weights = pd.Series(1.0 / len(assets), index=assets)
    return {
        "weights": weights,
        "metrics": {"objective": value, "turnover": 0.05},
        "status": "ok",
    }


def test_evaluate_population_returns_fitness() -> None:
    individuals = [
        Individual(np.array([1, 0, 0, 0], dtype=bool), {}),
        Individual(np.array([0, 1, 1, 0], dtype=bool), {}),
    ]
    data = {"universe": ["A", "B", "C", "D"]}
    config = {"metric": "objective"}

    results = evaluation.evaluate_population(individuals, data, _core_solver, config)

    assert len(results) == 2
    assert results[0].fitness < results[1].fitness


def test_handle_failures_assigns_low_fitness() -> None:
    individual = Individual(np.array([1, 0], dtype=bool), {})
    result = evaluation.handle_failures(individual, RuntimeError("fail"), {"failure_fitness": -99})
    assert result.fitness == -99
