from __future__ import annotations

import numpy as np
import pandas as pd

from itau_quant.optimization.ga import genetic
from itau_quant.optimization.ga.population import Individual


SCORES = {"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.5}


def _core_solver(candidate: dict[str, object]) -> dict[str, object]:
    assets = candidate["assets"]
    if not assets:
        raise ValueError("empty asset set")
    score = float(np.mean([SCORES[a] for a in assets]))
    weights = pd.Series(1.0 / len(assets), index=assets)
    return {
        "weights": weights,
        "metrics": {"objective": score},
        "status": "ok",
    }


def test_run_genetic_algorithm_improves_fitness() -> None:
    universe = list(SCORES.keys())
    data = {"universe": universe}
    config = {
        "seed": 1,
        "population": {"size": 6, "cardinality": {"min": 1, "max": 3}},
        "generations": 5,
        "selection": {"method": "tournament", "tournament_size": 3},
        "mutation": {
            "flip_prob": 0.2,
            "cardinality": {"min": 1, "max": 3},
            "universe": universe,
        },
        "evaluation": {"metric": "objective"},
        "elitism": 0.2,
    }

    result = genetic.run_genetic_algorithm(data, config, _core_solver)

    assert result.best_result.fitness >= 0.3
    assert result.history
