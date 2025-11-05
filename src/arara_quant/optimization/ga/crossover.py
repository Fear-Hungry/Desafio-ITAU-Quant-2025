"""Crossover operators for GA individuals."""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

from .population import Individual, ensure_feasible

__all__ = [
    "single_point_crossover",
    "uniform_crossover",
    "blend_crossover",
    "subset_exchange",
    "crossover_factory",
]


def _merge_metadata(
    parent_a: Individual, parent_b: Individual, method: str
) -> dict[str, object]:
    metadata = {
        "parents": [dict(parent_a.metadata), dict(parent_b.metadata)],
        "crossover": method,
    }
    return metadata


def _combine_params(
    params_a: Mapping[str, float],
    params_b: Mapping[str, float],
    chooser: Callable[[str, float, float], float],
) -> dict[str, float]:
    combined: dict[str, float] = {}
    keys = set(params_a) | set(params_b)
    for key in keys:
        combined[key] = chooser(key, params_a.get(key), params_b.get(key))
    return combined


def single_point_crossover(
    parent_a: Individual, parent_b: Individual, rng: np.random.Generator
) -> tuple[Individual, Individual]:
    if parent_a.assets_mask.size != parent_b.assets_mask.size:
        raise ValueError("parent masks must have equal length")
    length = parent_a.assets_mask.size
    if length == 1:
        return parent_a, parent_b
    point = int(rng.integers(1, length))
    child1_mask = np.concatenate(
        [parent_a.assets_mask[:point], parent_b.assets_mask[point:]]
    )
    child2_mask = np.concatenate(
        [parent_b.assets_mask[:point], parent_a.assets_mask[point:]]
    )

    def chooser(key, a, b):
        return (
            parent_a.params.get(key, b)
            if rng.random() < 0.5
            else parent_b.params.get(key, a)
        )

    child1_params = _combine_params(parent_a.params, parent_b.params, chooser)
    child2_params = _combine_params(parent_a.params, parent_b.params, chooser)
    meta1 = _merge_metadata(parent_a, parent_b, "single_point")
    meta2 = _merge_metadata(parent_b, parent_a, "single_point")
    return Individual(child1_mask, child1_params, meta1), Individual(
        child2_mask, child2_params, meta2
    )


def uniform_crossover(
    parent_a: Individual,
    parent_b: Individual,
    rng: np.random.Generator,
    *,
    prob: float = 0.5,
) -> tuple[Individual, Individual]:
    if parent_a.assets_mask.size != parent_b.assets_mask.size:
        raise ValueError("parent masks must have equal length")
    mask_choice = rng.random(parent_a.assets_mask.size) < prob
    child1_mask = np.where(mask_choice, parent_a.assets_mask, parent_b.assets_mask)
    child2_mask = np.where(mask_choice, parent_b.assets_mask, parent_a.assets_mask)

    def chooser(key: str, value_a: float | None, value_b: float | None) -> float:
        if value_a is None:
            return value_b
        if value_b is None:
            return value_a
        return value_a if rng.random() < prob else value_b

    child1_params = _combine_params(parent_a.params, parent_b.params, chooser)
    child2_params = _combine_params(parent_a.params, parent_b.params, chooser)
    meta1 = _merge_metadata(parent_a, parent_b, "uniform")
    meta2 = _merge_metadata(parent_b, parent_a, "uniform")
    return Individual(child1_mask, child1_params, meta1), Individual(
        child2_mask, child2_params, meta2
    )


def blend_crossover(
    params_a: Mapping[str, float],
    params_b: Mapping[str, float],
    rng: np.random.Generator,
    *,
    alpha: float = 0.5,
) -> dict[str, float]:
    blended: dict[str, float] = {}
    keys = set(params_a) | set(params_b)
    for key in keys:
        a_val = params_a.get(key)
        b_val = params_b.get(key)
        if isinstance(a_val, (int, float)) and isinstance(b_val, (int, float)):
            low = min(a_val, b_val) - alpha * abs(a_val - b_val)
            high = max(a_val, b_val) + alpha * abs(a_val - b_val)
            blended[key] = float(rng.uniform(low, high))
        else:
            blended[key] = a_val if rng.random() < 0.5 else b_val
    return blended


def subset_exchange(
    parent_a: Individual, parent_b: Individual, rng: np.random.Generator, *, k: int
) -> tuple[Individual, Individual]:
    mask_a = parent_a.assets_mask.copy()
    mask_b = parent_b.assets_mask.copy()
    active_a = np.flatnonzero(mask_a)
    active_b = np.flatnonzero(mask_b)
    if active_a.size == 0 or active_b.size == 0:
        return parent_a, parent_b
    swaps = min(k, active_a.size, active_b.size)
    rng.shuffle(active_a)
    rng.shuffle(active_b)
    mask_a[active_a[:swaps]] = False
    mask_b[active_b[:swaps]] = False
    mask_a[active_b[:swaps]] = True
    mask_b[active_a[:swaps]] = True
    meta1 = _merge_metadata(parent_a, parent_b, "subset_exchange")
    meta2 = _merge_metadata(parent_b, parent_a, "subset_exchange")
    child1 = parent_a.copy(assets_mask=mask_a, metadata=meta1)
    child2 = parent_b.copy(assets_mask=mask_b, metadata=meta2)
    return child1, child2


def crossover_factory(
    config: Mapping[str, object]
) -> Callable[
    [Individual, Individual, np.random.Generator], tuple[Individual, Individual]
]:
    cfg = dict(config or {})
    method = cfg.get("method", "uniform")
    method = str(method).lower()
    alpha = float(cfg.get("alpha", 0.5))
    subset_k = int(cfg.get("subset_k", 1))

    def crossover(
        parent_a: Individual, parent_b: Individual, rng: np.random.Generator
    ) -> tuple[Individual, Individual]:
        if method == "single_point":
            return single_point_crossover(parent_a, parent_b, rng)
        if method == "uniform":
            return uniform_crossover(
                parent_a, parent_b, rng, prob=float(cfg.get("prob", 0.5))
            )
        if method == "blend":
            params = blend_crossover(parent_a.params, parent_b.params, rng, alpha=alpha)
            mask_choice = rng.random(parent_a.assets_mask.size) < 0.5
            child_mask = np.where(
                mask_choice, parent_a.assets_mask, parent_b.assets_mask
            )
            metadata = _merge_metadata(parent_a, parent_b, "blend")
            child = parent_a.copy(
                assets_mask=child_mask, params=params, metadata=metadata
            )
            return child, parent_b
        if method == "subset_exchange":
            return subset_exchange(parent_a, parent_b, rng, k=subset_k)
        raise ValueError(f"unsupported crossover method '{method}'")

    def wrapped(
        parent_a: Individual, parent_b: Individual, rng: np.random.Generator
    ) -> tuple[Individual, Individual]:
        child_a, child_b = crossover(parent_a, parent_b, rng)
        constraints = cfg.get("constraints")
        if constraints:
            child_a = ensure_feasible(child_a, constraints, rng)
            child_b = ensure_feasible(child_b, constraints, rng)
        return child_a, child_b

    return wrapped
