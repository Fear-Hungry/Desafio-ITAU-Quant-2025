"""Utilities for genetic-algorithm individuals and populations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "Individual",
    "encode_individual",
    "decode_individual",
    "random_individual",
    "diversified_population",
    "warm_start_population",
    "ensure_feasible",
    "jaccard_distance",
]


def _to_bool_mask(mask: Sequence[bool] | np.ndarray, size: int | None = None) -> np.ndarray:
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 1:
        raise ValueError("assets_mask must be 1-dimensional")
    if size is not None and array.size != size:
        raise ValueError("assets_mask size mismatch")
    return array


@dataclass(frozen=True)
class Individual:
    """Immutable container describing a GA individual."""

    assets_mask: np.ndarray
    params: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mask = _to_bool_mask(self.assets_mask)
        object.__setattr__(self, "assets_mask", mask)
        object.__setattr__(self, "params", dict(self.params))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def cardinality(self) -> int:
        return int(self.assets_mask.sum())

    def active_assets(self, universe: Sequence[str]) -> list[str]:
        if len(universe) != self.assets_mask.size:
            raise ValueError("universe length must match mask length")
        return [asset for asset, active in zip(universe, self.assets_mask) if active]

    def copy(self, *, assets_mask: Sequence[bool] | np.ndarray | None = None, params: Mapping[str, Any] | None = None, metadata: Mapping[str, Any] | None = None) -> "Individual":
        new_mask = _to_bool_mask(assets_mask, self.assets_mask.size) if assets_mask is not None else self.assets_mask.copy()
        new_params = dict(self.params if params is None else params)
        new_metadata = dict(self.metadata if metadata is None else metadata)
        return Individual(new_mask, new_params, new_metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "assets_mask": self.assets_mask.astype(int).tolist(),
            "params": dict(self.params),
            "metadata": dict(self.metadata),
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "Individual":
        return Individual(
            np.asarray(payload["assets_mask"], dtype=bool),
            payload.get("params", {}),
            payload.get("metadata", {}),
        )


def encode_individual(assets: Iterable[str], params: Mapping[str, Any], universe: Sequence[str]) -> Individual:
    universe_index = {asset: idx for idx, asset in enumerate(universe)}
    mask = np.zeros(len(universe), dtype=bool)
    for asset in assets:
        if asset not in universe_index:
            raise ValueError(f"asset '{asset}' not present in universe")
        mask[universe_index[asset]] = True
    return Individual(mask, params, {"origin": "encoded"})


def decode_individual(individual: Individual, universe: Sequence[str]) -> tuple[list[str], dict[str, Any]]:
    assets = individual.active_assets(universe)
    params = dict(individual.params)
    return assets, params


def _sample_hyperparams(hyper_cfg: Mapping[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, spec in hyper_cfg.items():
        if isinstance(spec, Mapping):
            if "choices" in spec:
                params[name] = rng.choice(list(spec["choices"]))
            else:
                low = float(spec.get("min", spec.get("low", 0.0)))
                high = float(spec.get("max", spec.get("high", 1.0)))
                if low > high:
                    raise ValueError(f"invalid range for {name}: [{low}, {high}]")
                value = rng.uniform(low, high)
                if spec.get("dtype") == "int" or spec.get("type") == "int":
                    params[name] = int(round(value))
                else:
                    params[name] = float(value)
        elif isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
            if not spec:
                raise ValueError(f"empty choices for hyperparameter '{name}'")
            params[name] = rng.choice(list(spec))
        else:
            params[name] = spec
    return params


def _cardinality_bounds(universe_size: int, config: Mapping[str, Any]) -> tuple[int, int]:
    card_cfg = config.get("cardinality", {})
    min_assets = int(card_cfg.get("min", min(5, universe_size)))
    max_assets = int(card_cfg.get("max", universe_size))
    min_assets = max(1, min(min_assets, universe_size))
    max_assets = max(min_assets, min(max_assets, universe_size))
    return min_assets, max_assets


def random_individual(universe: Sequence[str], config: Mapping[str, Any], rng: np.random.Generator) -> Individual:
    if not universe:
        raise ValueError("universe must not be empty")
    min_assets, max_assets = _cardinality_bounds(len(universe), config)
    k = int(rng.integers(min_assets, max_assets + 1))
    indices = np.zeros(len(universe), dtype=bool)
    indices[rng.choice(len(universe), size=k, replace=False)] = True
    params = _sample_hyperparams(config.get("hyperparams", {}), rng)
    metadata = {"origin": "random"}
    return Individual(indices, params, metadata)


def jaccard_distance(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersect = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return 1.0 - (intersect / union)


def diversified_population(universe: Sequence[str], config: Mapping[str, Any], size: int, rng: np.random.Generator) -> list[Individual]:
    if size <= 0:
        raise ValueError("population size must be positive")
    diversity_cfg = config.get("diversity", {})
    min_jaccard = float(diversity_cfg.get("min_jaccard", 0.2))
    max_attempts = max(size * 10, 50)

    population: list[Individual] = []
    attempts = 0
    while len(population) < size and attempts < max_attempts:
        candidate = random_individual(universe, config, rng)
        if all(jaccard_distance(candidate.assets_mask, indiv.assets_mask) >= min_jaccard for indiv in population):
            population.append(candidate)
        attempts += 1

    while len(population) < size:
        population.append(random_individual(universe, config, rng))
    return population


def warm_start_population(universe: Sequence[str], historical_weights: pd.DataFrame | Mapping[str, Any] | Sequence[pd.Series], config: Mapping[str, Any]) -> list[Individual]:
    if isinstance(historical_weights, pd.DataFrame):
        records = [historical_weights.iloc[idx] for idx in range(len(historical_weights))]
    elif isinstance(historical_weights, Mapping):
        records = [pd.Series(historical_weights)]
    else:
        records = list(historical_weights)

    population: list[Individual] = []
    for weights in records:
        if not isinstance(weights, pd.Series):
            weights = pd.Series(weights)
        weights = weights.reindex(index=universe).fillna(0.0)
        mask = weights.to_numpy(dtype=float) > config.get("warm_threshold", 1e-6)
        params = dict(config.get("default_params", {}))
        metadata = {"origin": "warm_start"}
        population.append(Individual(mask, params, metadata))
    return population


def ensure_feasible(individual: Individual, constraints: Mapping[str, Any] | None, rng: np.random.Generator | None = None) -> Individual:
    if not constraints:
        return individual

    mask = individual.assets_mask.copy()
    params = dict(individual.params)
    metadata = dict(individual.metadata)
    rng = rng or np.random.default_rng()

    card_cfg = constraints.get("cardinality", {})
    min_assets = int(card_cfg.get("min", 1))
    max_assets = int(card_cfg.get("max", mask.size))

    active_indices = np.flatnonzero(mask)
    if active_indices.size < min_assets:
        available = np.flatnonzero(~mask)
        if available.size == 0:
            raise ValueError("cannot satisfy minimum cardinality constraint")
        rng.shuffle(available)
        mask[available[: min_assets - active_indices.size]] = True
    elif active_indices.size > max_assets:
        rng.shuffle(active_indices)
        mask[active_indices[: active_indices.size - max_assets]] = False

    bounds = constraints.get("bounds", {})
    for key, bound in bounds.items():
        if key not in params:
            continue
        low, high = bound
        value = params[key]
        if isinstance(value, (int, np.integer)):
            params[key] = int(np.clip(value, low, high))
        else:
            params[key] = float(np.clip(value, low, high))

    metadata.setdefault("constraints", {})
    metadata["constraints"]["enforced"] = True
    return Individual(mask, params, metadata)
