"""Genetic algorithm driver that orchestrates population evolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .crossover import crossover_factory
from .evaluation import EvaluationResult, evaluate_population
from .mutation import mutation_pipeline
from .population import (
    Individual,
    diversified_population,
    ensure_feasible,
    warm_start_population,
)
from .selection import selection_pipeline

__all__ = [
    "GenerationSummary",
    "GeneticRun",
    "run_genetic_algorithm",
]


@dataclass(frozen=True)
class GenerationSummary:
    generation: int
    best_fitness: float
    average_fitness: float
    diversity: float
    best_individual: Individual
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeneticRun:
    best_result: EvaluationResult
    history: list[GenerationSummary]
    population: list[Individual]


def _calculate_diversity(population: Sequence[Individual]) -> float:
    if len(population) < 2:
        return 0.0
    masks = np.stack([ind.assets_mask for ind in population])
    similarities = masks @ masks.T
    cardinalities = masks.sum(axis=1)
    union = cardinalities[:, None] + cardinalities[None, :] - similarities
    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard = 1.0 - np.where(union > 0, similarities / union, 0.0)
    upper = jaccard[np.triu_indices(len(population), k=1)]
    if upper.size == 0:
        return 0.0
    return float(np.nanmean(upper))


def _initial_population(
    universe: Sequence[str], config: Mapping[str, Any], rng: np.random.Generator
) -> list[Individual]:
    pop_cfg = config.get("population", {})
    size = int(pop_cfg.get("size", 20))
    base_population = diversified_population(universe, pop_cfg, size, rng)
    warm_data = pop_cfg.get("warm_start")
    if warm_data is not None:
        base_population[:0] = warm_start_population(universe, warm_data, pop_cfg)
    return base_population[:size]


def run_genetic_algorithm(
    data: Mapping[str, Any],
    config: Mapping[str, Any],
    core_solver: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    rng: np.random.Generator | None = None,
) -> GeneticRun:
    if "universe" not in data:
        raise ValueError("data must include a 'universe' sequence")
    universe = list(data["universe"])
    rng = rng or np.random.default_rng(config.get("seed"))

    population = _initial_population(universe, config, rng)
    mutation_cfg = dict(config.get("mutation", {}))
    mutation_cfg.setdefault("constraints", config.get("constraints"))
    mutation_cfg.setdefault("cardinality", config.get("cardinality", {}))
    mutation_cfg.setdefault("universe", universe)

    crossover_cfg = config.get("crossover", {})
    crossover_cfg = dict(crossover_cfg)
    crossover_cfg.setdefault("constraints", config.get("constraints"))
    crossover = crossover_factory(crossover_cfg)

    selection_cfg = config.get(
        "selection", {"method": "tournament", "tournament_size": 3}
    )
    evaluation_cfg = config.get("evaluation", {"metric": "objective"})

    generations = int(config.get("generations", 10))
    elitism_ratio = float(config.get("elitism", 0.1))
    stagnation_limit = int(config.get("stagnation", max(3, generations)))

    history: list[GenerationSummary] = []
    best_result: EvaluationResult | None = None
    stagnation_counter = 0

    for generation in range(generations):
        evaluations = evaluate_population(population, data, core_solver, evaluation_cfg)
        fitness_scores = [result.fitness for result in evaluations]
        best_idx = int(np.argmax(fitness_scores))
        generation_best = evaluations[best_idx]
        if best_result is None or generation_best.fitness > best_result.fitness:
            best_result = generation_best
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        summary = GenerationSummary(
            generation=generation,
            best_fitness=float(generation_best.fitness),
            average_fitness=float(np.mean(fitness_scores)),
            diversity=_calculate_diversity(population),
            best_individual=generation_best.individual,
        )
        history.append(summary)

        if stagnation_counter >= stagnation_limit:
            break

        elite_count = max(1, int(round(elitism_ratio * len(population))))
        elite_indices = np.argsort(fitness_scores)[::-1][:elite_count]
        elites = [population[idx] for idx in elite_indices]

        next_population: list[Individual] = elites.copy()
        while len(next_population) < len(population):
            parents = selection_pipeline(
                population, fitness_scores, selection_cfg, rng, num_parents=2
            )
            child_a, child_b = crossover(parents[0], parents[1], rng)
            child_a = mutation_pipeline(child_a, mutation_cfg, rng)
            child_b = mutation_pipeline(child_b, mutation_cfg, rng)
            child_a = ensure_feasible(child_a, config.get("constraints"), rng)
            child_b = ensure_feasible(child_b, config.get("constraints"), rng)
            next_population.append(child_a)
            if len(next_population) < len(population):
                next_population.append(child_b)

        population = next_population[: len(population)]

    if best_result is None:
        raise RuntimeError("genetic algorithm failed to evaluate any individual")
    return GeneticRun(best_result=best_result, history=history, population=population)
