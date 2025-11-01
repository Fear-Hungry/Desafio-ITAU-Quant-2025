"""Fitness evaluation helpers for the GA pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .population import Individual

__all__ = [
    "EvaluationResult",
    "build_candidate_solution",
    "run_core_optimizer",
    "compute_fitness",
    "handle_failures",
    "evaluate_population",
]


@dataclass(frozen=True)
class EvaluationResult:
    individual: Individual
    fitness: float
    weights: pd.Series | None
    metrics: Mapping[str, Any]
    status: str
    diagnostics: Mapping[str, Any]


def build_candidate_solution(
    individual: Individual, data: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    universe = data.get("universe")
    if universe is None:
        raise ValueError("evaluation data must contain 'universe'")
    assets = individual.active_assets(universe)
    candidate = {
        "assets": assets,
        "params": dict(individual.params),
        "metadata": dict(individual.metadata),
        "data": data,
        "config": config,
    }
    return candidate


def run_core_optimizer(
    candidate: Mapping[str, Any],
    core_solver: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> Mapping[str, Any]:
    result = core_solver(candidate)
    if not isinstance(result, Mapping):
        raise TypeError("core_solver must return a mapping")
    return result


def compute_fitness(
    weights: pd.Series | Sequence[float] | None,
    metrics: Mapping[str, Any],
    penalties: Mapping[str, float] | None,
    config: Mapping[str, Any],
) -> float:
    if penalties is None:
        penalties = {}
    metric_key = config.get("metric", "objective")
    base_metric = metrics.get(metric_key)
    if base_metric is None:
        fallback = (
            metrics.get("sharpe")
            or metrics.get("objective_value")
            or metrics.get("return")
        )
        if fallback is None:
            raise ValueError(
                f"metric '{metric_key}' not available in metrics {list(metrics)}"
            )
        base_metric = fallback
    fitness = float(base_metric)

    turnover_penalty = penalties.get("turnover")
    if turnover_penalty is not None:
        fitness -= float(turnover_penalty)

    concentration_penalty = penalties.get("concentration")
    if concentration_penalty is not None:
        fitness -= float(concentration_penalty)

    if weights is not None and config.get("l2_penalty"):
        arr = np.asarray(weights, dtype=float)
        fitness -= float(config["l2_penalty"]) * float(np.linalg.norm(arr))
    return fitness


def handle_failures(
    individual: Individual, error: Exception, config: Mapping[str, Any]
) -> EvaluationResult:
    fitness = float(config.get("failure_fitness", -1e9))
    diagnostics = {"error": str(error)}
    return EvaluationResult(individual, fitness, None, {}, "failed", diagnostics)


def _evaluate_single(
    individual: Individual,
    data: Mapping[str, Any],
    core_solver: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    config: Mapping[str, Any],
) -> EvaluationResult:
    candidate = build_candidate_solution(individual, data, config)
    result = run_core_optimizer(candidate, core_solver)
    weights = result.get("weights")
    if weights is not None and not isinstance(weights, pd.Series):
        weights = pd.Series(weights, index=candidate.get("assets"))
    metrics = result.get("metrics", {})
    status = str(result.get("status", "ok"))
    penalties = result.get("penalties", {})
    fitness = compute_fitness(weights, metrics, penalties, config)
    diagnostics = result.get("diagnostics", {})
    return EvaluationResult(individual, fitness, weights, metrics, status, diagnostics)


def evaluate_population(
    population: Sequence[Individual],
    data: Mapping[str, Any],
    core_solver: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[EvaluationResult]:
    results: list[EvaluationResult] = []
    for individual in population:
        try:
            result = _evaluate_single(individual, data, core_solver, config)
        except Exception as exc:  # pragma: no cover - safety net
            result = handle_failures(individual, exc, config)
        results.append(result)
    return results
