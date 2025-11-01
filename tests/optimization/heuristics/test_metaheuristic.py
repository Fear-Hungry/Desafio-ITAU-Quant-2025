from __future__ import annotations

import numpy as np
import pandas as pd
from itau_quant.optimization.core.mv_qp import MeanVarianceConfig
from itau_quant.optimization.heuristics.metaheuristic import metaheuristic_outer


def test_metaheuristic_outer_tunes_cardinality_and_params() -> None:
    assets = [f"A{i}" for i in range(5)]
    mu = pd.Series([0.12, 0.10, 0.08, 0.07, 0.05], index=assets, dtype=float)
    base_cov = 0.015 * np.ones((5, 5))
    np.fill_diagonal(base_cov, 0.03)
    cov = pd.DataFrame(base_cov, index=assets, columns=assets, dtype=float)

    previous = pd.Series(0.0, index=assets, dtype=float)
    lower = pd.Series(0.0, index=assets, dtype=float)
    upper = pd.Series(0.6, index=assets, dtype=float)

    base_config = MeanVarianceConfig(
        risk_aversion=4.0,
        turnover_penalty=0.0,
        turnover_cap=None,
        lower_bounds=lower,
        upper_bounds=upper,
        previous_weights=previous,
        cost_vector=None,
    )

    ga_config = {
        "seed": 123,
        "generations": 4,
        "population": {
            "size": 12,
            "cardinality": {"min": 2, "max": 4},
            "hyperparams": {
                "lambda": {"choices": [1.0, 6.0]},
                "eta": {"choices": [0.0, 0.05]},
                "tau": {"choices": [None, 0.4]},
            },
        },
        "mutation": {
            "flip_prob": 0.25,
            "constraints": {"cardinality": {"min": 2, "max": 4}},
            "universe": assets,
            "discrete": [
                {"param": "lambda", "values": [1.0, 6.0], "prob": 0.4},
                {"param": "eta", "values": [0.0, 0.05], "prob": 0.4},
            ],
        },
        "crossover": {
            "method": "uniform",
            "constraints": {"cardinality": {"min": 2, "max": 4}},
        },
        "selection": {"method": "tournament", "tournament_size": 3},
        "evaluation": {"metric": "expected_return"},
        "elitism": 0.2,
        "constraints": {"cardinality": {"min": 2, "max": 4}},
    }

    result = metaheuristic_outer(
        mu,
        cov,
        base_config,
        ga_config=ga_config,
        cardinality_target=(3, 3),
        penalty_weights={"cardinality": 5.0},
    )

    assert abs(result.weights.sum() - 1.0) < 1e-8
    active = result.weights[result.weights > 1e-6]
    assert len(active) == 3
    assert result.metrics["cardinality"] == 3
    assert result.metrics["fitness"] == result.fitness
    assert result.status in {"optimal", "OPTIMAL"}
    assert result.history, "GA should produce generation summaries"
    assert set(result.selected_assets) == set(active.index)
    assert "lambda" in result.params
