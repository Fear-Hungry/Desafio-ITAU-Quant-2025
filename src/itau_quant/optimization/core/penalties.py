"""Funções de penalização para CVXPy."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import cvxpy as cp
import numpy as np
import pandas as pd

__all__ = [
    "l1_penalty",
    "l2_penalty",
    "group_lasso_penalty",
    "cardinality_soft_penalty",
    "turnover_penalty",
    "penalty_factory",
]


def _ensure_gamma(gamma: float, name: str) -> float:
    value = float(gamma)
    if value < 0:
        raise ValueError(f"{name} must have non-negative gamma")
    return value


def _ensure_series(candidate: Sequence[float] | Mapping[str, float] | pd.Series, index: Sequence[str]) -> pd.Series:
    if isinstance(candidate, pd.Series):
        return candidate.reindex(index).astype(float).fillna(0.0)
    if isinstance(candidate, Mapping):
        return pd.Series(candidate, dtype=float).reindex(index).fillna(0.0)
    array = np.asarray(candidate, dtype=float)
    if array.ndim != 1 or array.size != len(index):
        raise ValueError("sequence must be 1-dimensional and match asset index size")
    return pd.Series(array, index=index, dtype=float)


def l1_penalty(weights: cp.Expression, gamma: float) -> cp.Expression:
    """Return L1 regularisation term ``gamma * ||w||_1``."""

    gamma_value = _ensure_gamma(gamma, "l1 penalty")
    if gamma_value == 0:
        return cp.Constant(0.0)
    return gamma_value * cp.norm1(weights)


def l2_penalty(weights: cp.Expression, gamma: float) -> cp.Expression:
    """Return L2 (ridge) penalty ``gamma * ||w||_2^2``."""

    gamma_value = _ensure_gamma(gamma, "l2 penalty")
    if gamma_value == 0:
        return cp.Constant(0.0)
    return gamma_value * cp.sum_squares(weights)


def group_lasso_penalty(
    weights: cp.Expression,
    groups: Mapping[str, Iterable[int | str]],
    gamma: float,
    *,
    asset_index: Sequence[str] | None = None,
) -> cp.Expression:
    """Return sum of group L2 norms promoting structured sparsity."""

    gamma_value = _ensure_gamma(gamma, "group lasso penalty")
    if gamma_value == 0 or not groups:
        return cp.Constant(0.0)

    if asset_index is None:
        raise ValueError("asset_index is required for group_lasso_penalty")
    asset_index = list(asset_index)

    norms: list[cp.Expression] = []
    for _, members in groups.items():
        member_list = list(members)
        if not member_list:
            continue
        indices: list[int] = []
        for member in member_list:
            if isinstance(member, int):
                indices.append(member)
            else:
                try:
                    indices.append(asset_index.index(str(member)))
                except ValueError as exc:  # pragma: no cover - defensive guard
                    raise ValueError(f"unknown asset '{member}' in group_lasso_penalty") from exc
        if not indices:
            continue
        group_weights = cp.vstack([weights[i] for i in indices])
        norms.append(cp.norm(group_weights, 2))

    if not norms:
        return cp.Constant(0.0)
    return gamma_value * cp.sum(norms)


def cardinality_soft_penalty(
    weights: cp.Expression,
    k_target: int,
    gamma: float,
    *,
    method: str = "huber",
    epsilon: float = 1e-3,
) -> cp.Expression:
    """Return convex surrogate encouraging at most ``k_target`` non-zero weights."""

    gamma_value = _ensure_gamma(gamma, "cardinality penalty")
    if gamma_value == 0:
        return cp.Constant(0.0)

    if k_target <= 0:
        raise ValueError("k_target must be positive")

    abs_weights = cp.abs(weights)
    if method.lower() == "huber":
        threshold = 1.0 / max(k_target, 1)
        penalty = cp.sum(cp.huber(abs_weights, threshold))
    elif method.lower() == "linear":
        threshold = 1.0 / max(k_target, 1)
        penalty = cp.sum(cp.pos(abs_weights - threshold))
    else:
        raise ValueError("unsupported method for cardinality_soft_penalty")

    if epsilon > 0:
        penalty += epsilon * cp.sum(abs_weights)
    return gamma_value * penalty


def turnover_penalty(
    weights: cp.Expression,
    previous_weights: Sequence[float] | Mapping[str, float] | pd.Series,
    gamma: float,
    *,
    asset_index: Sequence[str] | None = None,
    normalised: bool = False,
) -> cp.Expression:
    """Return turnover penalty ``gamma * ||w - w_prev||_1``."""

    gamma_value = _ensure_gamma(gamma, "turnover penalty")
    if gamma_value == 0:
        return cp.Constant(0.0)

    if asset_index is None:
        raise ValueError("asset_index is required for turnover_penalty")

    prev_series = _ensure_series(previous_weights, asset_index)
    diff = weights - prev_series.to_numpy()
    scale = 0.5 if normalised else 1.0
    return gamma_value * scale * cp.norm1(diff)


def penalty_factory(
    weights: cp.Expression,
    config: Mapping[str, object] | None,
    *,
    asset_index: Sequence[str],
    context: Mapping[str, object] | None = None,
) -> list[cp.Expression]:
    """Create penalty expressions from a declarative configuration."""

    if not config:
        return []

    context = dict(context or {})
    penalties: list[cp.Expression] = []

    if "l1" in config:
        entry = config["l1"]
        gamma = float(entry["gamma"]) if isinstance(entry, Mapping) else float(entry)
        penalties.append(l1_penalty(weights, gamma))

    if "l2" in config:
        entry = config["l2"]
        gamma = float(entry["gamma"]) if isinstance(entry, Mapping) else float(entry)
        penalties.append(l2_penalty(weights, gamma))

    if "group_lasso" in config:
        entry = config["group_lasso"]
        if not isinstance(entry, Mapping):
            raise TypeError("group_lasso configuration must be a mapping")
        gamma = float(entry.get("gamma", 0.0))
        groups = entry.get("groups", {})
        group_index = entry.get("asset_index", asset_index)
        penalties.append(group_lasso_penalty(weights, groups, gamma, asset_index=group_index))

    if "cardinality" in config:
        entry = config["cardinality"]
        if not isinstance(entry, Mapping):
            raise TypeError("cardinality configuration must be a mapping")
        gamma = float(entry.get("gamma", 0.0))
        k_target = int(entry.get("k_target", 0))
        method = str(entry.get("method", "huber"))
        epsilon = float(entry.get("epsilon", 1e-3))
        penalties.append(cardinality_soft_penalty(weights, k_target, gamma, method=method, epsilon=epsilon))

    if "turnover" in config:
        entry = config["turnover"]
        if not isinstance(entry, Mapping):
            raise TypeError("turnover configuration must be a mapping")
        gamma = float(entry.get("gamma", 0.0))
        prev = entry.get("previous") or context.get("previous_weights")
        if prev is None:
            raise ValueError("turnover penalty requires previous weights")
        normalised = bool(entry.get("normalised", False))
        penalties.append(
            turnover_penalty(
                weights,
                prev,
                gamma,
                asset_index=entry.get("asset_index", asset_index),
                normalised=normalised,
            )
        )

    return [pen for pen in penalties if not isinstance(pen, cp.Constant) or pen.value != 0.0]
