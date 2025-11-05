"""Mutation operators used by the genetic algorithm suite."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .population import Individual, ensure_feasible

__all__ = [
    "flip_asset_selection",
    "gaussian_jitter_params",
    "discrete_adjustment",
    "swap_assets",
    "mutation_pipeline",
]


def flip_asset_selection(
    individual: Individual,
    prob: float,
    rng: np.random.Generator,
    *,
    min_assets: int = 1,
    max_assets: int | None = None,
) -> Individual:
    if not 0.0 <= prob <= 1.0:
        raise ValueError("prob must be in [0, 1]")
    mask = individual.assets_mask.copy()
    flips = rng.random(mask.size) < prob
    if not flips.any():
        return individual
    mask = np.logical_xor(mask, flips)
    if not mask.any():
        idx = rng.integers(0, mask.size)
        mask[idx] = True
    max_assets = max_assets or mask.size
    meta = dict(individual.metadata)
    meta["last_mutation"] = "flip"
    mutated = individual.copy(assets_mask=mask, metadata=meta)
    return ensure_feasible(
        mutated, {"cardinality": {"min": min_assets, "max": max_assets}}, rng
    )


def gaussian_jitter_params(
    individual: Individual,
    sigma: float | Mapping[str, float],
    bounds: Mapping[str, tuple[float, float]],
    rng: np.random.Generator,
) -> Individual:
    params = dict(individual.params)
    if not params:
        return individual
    meta = dict(individual.metadata)
    meta["last_mutation"] = "gaussian_jitter"
    for name, (low, high) in bounds.items():
        if name not in params:
            continue
        sigma_value = sigma[name] if isinstance(sigma, Mapping) else sigma
        if sigma_value <= 0:
            continue
        jitter = float(rng.normal(0.0, sigma_value))
        new_value = params[name] + jitter
        new_value = float(np.clip(new_value, low, high))
        if isinstance(params[name], (int, np.integer)):
            params[name] = int(round(new_value))
        else:
            params[name] = new_value
    return individual.copy(params=params, metadata=meta)


def discrete_adjustment(
    individual: Individual,
    param_name: str,
    values: Sequence[Any],
    prob: float,
    rng: np.random.Generator,
) -> Individual:
    if not values:
        raise ValueError("values must not be empty")
    if rng.random() >= prob:
        return individual
    params = dict(individual.params)
    current = params.get(param_name)
    choices = [value for value in values if value != current]
    if not choices:
        return individual
    params[param_name] = rng.choice(choices)
    meta = dict(individual.metadata)
    meta["last_mutation"] = f"discrete:{param_name}"
    return individual.copy(params=params, metadata=meta)


def swap_assets(
    individual: Individual,
    universe: Sequence[str],
    rng: np.random.Generator,
    *,
    num_swaps: int = 1,
) -> Individual:
    mask = individual.assets_mask.copy()
    active = np.flatnonzero(mask)
    inactive = np.flatnonzero(~mask)
    if active.size == 0 or inactive.size == 0:
        return individual
    swaps = min(num_swaps, active.size, inactive.size)
    rng.shuffle(active)
    rng.shuffle(inactive)
    mask[active[:swaps]] = False
    mask[inactive[:swaps]] = True
    meta = dict(individual.metadata)
    meta.setdefault("swap_history", []).append(swaps)
    meta["last_mutation"] = "swap"
    return individual.copy(assets_mask=mask, metadata=meta)


def mutation_pipeline(
    individual: Individual,
    config: Mapping[str, Any],
    rng: np.random.Generator,
) -> Individual:
    mutated = individual
    cfg = dict(config or {})
    card_cfg = cfg.get("cardinality", {})
    min_assets = int(card_cfg.get("min", 1))
    max_assets = card_cfg.get("max")

    flip_prob = float(cfg.get("flip_prob", 0.0))
    if flip_prob > 0:
        mutated = flip_asset_selection(
            mutated, flip_prob, rng, min_assets=min_assets, max_assets=max_assets
        )

    gaussian_cfg = cfg.get("gaussian")
    if gaussian_cfg:
        sigma = gaussian_cfg.get("sigma", 0.0)
        bounds = gaussian_cfg.get("bounds", {})
        mutated = gaussian_jitter_params(mutated, sigma, bounds, rng)

    for adjustment in cfg.get("discrete", []):
        mutated = discrete_adjustment(
            mutated,
            adjustment["param"],
            adjustment.get("values", []),
            adjustment.get("prob", cfg.get("discrete_prob", 0.2)),
            rng,
        )

    if cfg.get("swap_prob") and rng.random() < cfg.get("swap_prob"):
        universe = cfg.get("universe")
        if universe is None:
            raise ValueError("swap mutation requires 'universe' in config")
        mutated = swap_assets(
            mutated, universe, rng, num_swaps=int(cfg.get("swap_count", 1))
        )

    constraints = cfg.get("constraints")
    if constraints:
        mutated = ensure_feasible(mutated, constraints, rng)
    return mutated
