"""Parent selection helpers for the genetic algorithm."""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np

from .population import Individual, jaccard_distance

__all__ = [
    "tournament_selection",
    "roulette_wheel_selection",
    "stochastic_universal_sampling",
    "diversity_preserving_selection",
    "selection_pipeline",
]


def _validate_inputs(
    population: Sequence[Individual], fitness: Sequence[float]
) -> None:
    if len(population) != len(fitness):
        raise ValueError("population and fitness must have the same length")
    if not population:
        raise ValueError("population must not be empty")


def tournament_selection(
    population: Sequence[Individual],
    fitness: Sequence[float],
    tournament_size: int,
    rng: np.random.Generator,
) -> Individual:
    _validate_inputs(population, fitness)
    tournament_size = max(1, min(tournament_size, len(population)))
    indices = rng.choice(len(population), size=tournament_size, replace=False)
    best_idx = max(indices, key=lambda idx: fitness[idx])
    return population[best_idx]


def _normalise_fitness(fitness: Sequence[float]) -> np.ndarray:
    values = np.asarray(fitness, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("fitness scores contain no finite values")
    min_value = values[finite].min()
    if min_value < 0:
        values = values - min_value + 1e-9
    total = values.sum()
    if total == 0:
        values = np.full_like(values, 1.0 / len(values))
    else:
        values = values / total
    return values


def roulette_wheel_selection(
    population: Sequence[Individual], fitness: Sequence[float], rng: np.random.Generator
) -> Individual:
    _validate_inputs(population, fitness)
    probabilities = _normalise_fitness(fitness)
    idx = rng.choice(len(population), p=probabilities)
    return population[idx]


def stochastic_universal_sampling(
    population: Sequence[Individual],
    fitness: Sequence[float],
    num_samples: int,
    rng: np.random.Generator,
) -> list[Individual]:
    _validate_inputs(population, fitness)
    num_samples = max(1, min(num_samples, len(population)))
    probabilities = _normalise_fitness(fitness)
    cumulative = np.cumsum(probabilities)
    step = 1.0 / num_samples
    start = rng.uniform(0, step)
    pointers = start + step * np.arange(num_samples)

    selected: list[Individual] = []
    idx = 0
    for pointer in pointers:
        while pointer > cumulative[idx]:
            idx += 1
        selected.append(population[idx])
    return selected


def diversity_preserving_selection(
    population: Sequence[Individual],
    fitness: Sequence[float],
    rng: np.random.Generator,
    *,
    diversity_metric: Callable[[Individual, Individual], float] | None = None,
    num_parents: int = 2,
) -> list[Individual]:
    _validate_inputs(population, fitness)
    diversity_metric = diversity_metric or (
        lambda a, b: jaccard_distance(a.assets_mask, b.assets_mask)
    )
    order = np.argsort(fitness)[::-1]
    selected: list[Individual] = []
    for idx in order:
        candidate = population[idx]
        if not selected:
            selected.append(candidate)
            if len(selected) == num_parents:
                break
            continue
        distance = np.mean([diversity_metric(candidate, other) for other in selected])
        if distance >= 0.1 or len(selected) == 0:
            selected.append(candidate)
        if len(selected) == num_parents:
            break
    while len(selected) < num_parents:
        selected.append(population[order[len(selected)]])
    return selected


def selection_pipeline(
    population: Sequence[Individual],
    fitness: Sequence[float],
    config: Mapping[str, object],
    rng: np.random.Generator,
    *,
    num_parents: int | None = None,
) -> list[Individual]:
    _validate_inputs(population, fitness)
    cfg = dict(config or {})
    method = cfg.pop("method", "tournament")
    num_parents = num_parents or int(cfg.pop("num_parents", 2))

    if isinstance(method, Sequence) and not isinstance(method, (str, bytes)):
        parents: list[Individual] = []
        for sub_method in method:
            if len(parents) >= num_parents:
                break
            parents.extend(
                selection_pipeline(
                    population,
                    fitness,
                    {"method": sub_method, **cfg},
                    rng,
                    num_parents=1,
                )
            )
        return parents[:num_parents]

    method = str(method).lower()
    parents: list[Individual] = []
    while len(parents) < num_parents:
        if method == "tournament":
            size = int(cfg.get("tournament_size", 3))
            parents.append(tournament_selection(population, fitness, size, rng))
        elif method == "roulette":
            parents.append(roulette_wheel_selection(population, fitness, rng))
        elif method in {"sus", "stochastic_universal"}:
            needed = num_parents - len(parents)
            parents.extend(
                stochastic_universal_sampling(population, fitness, needed, rng)
            )
        elif method == "diversity":
            parents.extend(
                diversity_preserving_selection(
                    population, fitness, rng, num_parents=num_parents
                )
            )
        else:
            raise ValueError(f"unknown selection method '{method}'")
    return parents[:num_parents]
