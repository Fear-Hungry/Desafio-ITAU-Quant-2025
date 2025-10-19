"""Pós-processamento de pesos otimizados."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "project_to_simplex",
    "clip_to_bounds",
    "rebalance_equal_weight",
    "round_to_lots",
    "enforce_cardinality",
    "postprocess_pipeline",
]


def _ensure_series(weights: Sequence[float] | pd.Series, index: Sequence[str] | None = None) -> pd.Series:
    if isinstance(weights, pd.Series):
        series = weights.astype(float).fillna(0.0)
        if index is not None:
            series = series.reindex(index, fill_value=0.0)
        return series
    array = np.asarray(weights, dtype=float)
    if array.ndim != 1:
        raise ValueError("weights must be 1-dimensional")
    if index is None:
        index = range(len(array))
    if len(index) != array.size:
        raise ValueError("index length mismatch with weights")
    return pd.Series(array, index=index, dtype=float).fillna(0.0)


def _renormalise(series: pd.Series) -> pd.Series:
    total = float(series.sum())
    if not np.isfinite(total) or abs(total) < 1e-14:
        return series.copy()
    return series / total


def project_to_simplex(weights: Sequence[float] | pd.Series) -> pd.Series:
    """Project weights onto the probability simplex (non-negative, sum=1)."""

    series = _ensure_series(weights)
    values = series.to_numpy(dtype=float)
    values[~np.isfinite(values)] = 0.0

    n = values.size
    if n == 0:
        return series

    sorted_vals = np.sort(values)[::-1]
    cumulative = np.cumsum(sorted_vals)
    rho_candidates = sorted_vals * np.arange(1, n + 1) > (cumulative - 1)
    if not np.any(rho_candidates):
        theta = cumulative[-1] / n - 1.0 / n
    else:
        rho = np.nonzero(rho_candidates)[0][-1]
        theta = (cumulative[rho] - 1.0) / (rho + 1)
    projected = np.maximum(values - theta, 0.0)
    projected /= projected.sum() if projected.sum() > 0 else 1.0
    return pd.Series(projected, index=series.index, name=series.name)


def clip_to_bounds(
    weights: Sequence[float] | pd.Series,
    lower: float | Sequence[float] | pd.Series,
    upper: float | Sequence[float] | pd.Series,
    *,
    renormalise: bool = True,
) -> pd.Series:
    """Clip weights between lower/upper bounds and optionally renormalise."""

    series = _ensure_series(weights)

    if isinstance(lower, pd.Series):
        lower_series = lower.reindex(series.index).astype(float).fillna(-np.inf)
    elif isinstance(lower, Sequence) and not isinstance(lower, (str, bytes)):
        lower_series = pd.Series(lower, index=series.index, dtype=float)
    else:
        lower_series = pd.Series(float(lower), index=series.index, dtype=float)

    if isinstance(upper, pd.Series):
        upper_series = upper.reindex(series.index).astype(float).fillna(np.inf)
    elif isinstance(upper, Sequence) and not isinstance(upper, (str, bytes)):
        upper_series = pd.Series(upper, index=series.index, dtype=float)
    else:
        upper_series = pd.Series(float(upper), index=series.index, dtype=float)

    clipped = series.clip(lower=lower_series, upper=upper_series)
    if renormalise:
        positive_sum = clipped.sum()
        if positive_sum > 0:
            clipped = clipped / positive_sum
    return clipped


def rebalance_equal_weight(selected_assets: Sequence[str]) -> pd.Series:
    """Return equal weight allocation for the provided assets."""

    assets = list(dict.fromkeys(selected_assets))
    if not assets:
        raise ValueError("selected_assets must not be empty")
    weight = 1.0 / len(assets)
    return pd.Series(weight, index=assets, dtype=float)


def round_to_lots(
    weights: Sequence[float] | pd.Series,
    lot_size: float | Mapping[str, float],
    *,
    renormalise: bool = True,
) -> pd.Series:
    """Round weights to lot multiples (per-asset or scalar)."""

    series = _ensure_series(weights)

    if isinstance(lot_size, Mapping):
        lots = pd.Series(lot_size, dtype=float).reindex(series.index).fillna(0.0)
    else:
        lots = pd.Series(float(lot_size), index=series.index, dtype=float)

    lots = lots.clip(lower=0.0)

    def _round(base: pd.Series) -> pd.Series:
        rounded_local = base.astype(float).copy()
        for asset, lot in lots.items():
            if lot <= 0:
                continue
            rounded_local.loc[asset] = np.round(rounded_local.loc[asset] / lot) * lot
        return rounded_local

    rounded = _round(series)
    if not renormalise:
        return rounded

    positive_lots = lots[lots > 0]
    if positive_lots.empty:
        total = rounded.sum()
        return rounded / total if total > 0 else rounded

    diff = 1.0 - rounded.sum()
    positive_assets = [asset for asset in series.index if series.loc[asset] > 1e-12]
    order = series.loc[positive_assets].sort_values(ascending=False).index.tolist()
    if not order:
        order = series.sort_values(ascending=False).index.tolist()
    if not order:
        return rounded

    iteration = 0
    tolerance = positive_lots.min()
    while abs(diff) >= max(tolerance, 1e-9) and iteration < 1000:
        asset = order[iteration % len(order)]
        lot = positive_lots.get(asset, tolerance)
        if lot <= 0:
            iteration += 1
            continue
        if diff > 0:
            rounded.loc[asset] += lot
            diff -= lot
        else:
            candidate = rounded.loc[asset] - lot
            if candidate >= -1e-12:
                rounded.loc[asset] = max(candidate, 0.0)
                diff += lot
        iteration += 1

    return rounded


def enforce_cardinality(weights: Sequence[float] | pd.Series, k: int) -> pd.Series:
    """Keep the largest ``k`` absolute weights and renormalise the remainder."""

    if k <= 0:
        raise ValueError("k must be positive")

    series = _ensure_series(weights)
    if k >= series.size:
        series = series.clip(lower=0.0)
        return _renormalise(series)

    abs_order = series.abs().sort_values(ascending=False)
    keep_assets = abs_order.iloc[:k].index
    trimmed = series.where(series.index.isin(keep_assets), 0.0)
    trimmed = trimmed.clip(lower=0.0)
    total = trimmed.sum()
    if total <= 0:
        equal_weight = 1.0 / k
        trimmed.loc[keep_assets] = equal_weight
        return trimmed
    return trimmed / total


def postprocess_pipeline(
    weights: Sequence[float] | pd.Series,
    config: Mapping[str, object] | None = None,
) -> pd.Series:
    """Apply a configurable sequence of post-processing transformations."""

    config = dict(config or {})
    result = _ensure_series(weights)
    support_locked = False

    if config.get("fillna", True):
        result = result.fillna(0.0)

    if "enforce_cardinality" in config:
        card_cfg = config["enforce_cardinality"]
        if not isinstance(card_cfg, Mapping):
            raise TypeError("enforce_cardinality configuration must be a mapping")
        result = enforce_cardinality(result, int(card_cfg.get("k", 0)))
        support_locked = True

    if "clip" in config:
        clip_cfg = config["clip"]
        if not isinstance(clip_cfg, Mapping):
            raise TypeError("clip configuration must be a mapping")
        result = clip_to_bounds(
            result,
            clip_cfg.get("lower", 0.0),
            clip_cfg.get("upper", 1.0),
            renormalise=bool(clip_cfg.get("renormalise", True)),
        )

    if "round_lots" in config:
        round_cfg = config["round_lots"]
        if not isinstance(round_cfg, Mapping):
            raise TypeError("round_lots configuration must be a mapping")
        result = round_to_lots(
            result,
            round_cfg.get("lot_size", 0.0),
            renormalise=bool(round_cfg.get("renormalise", True)),
        )

    project_flag = config.get("project_simplex", True)
    if project_flag:
        if support_locked:
            result = result.clip(lower=0.0)
            total = result.sum()
            if total > 0:
                result = result / total
        else:
            result = project_to_simplex(result)
    elif config.get("renormalise", False):
        result = _renormalise(result)

    return result.astype(float)
