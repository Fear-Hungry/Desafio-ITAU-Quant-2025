"""Heuristic strategies to enforce approximate cardinality constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "SelectionResult",
    "greedy_selection",
    "beam_search_selection",
    "prune_after_optimisation",
    "reoptimize_with_subset",
    "cardinality_pipeline",
]


def _ensure_series(values: Sequence[float] | pd.Series | None, index: Sequence[str], *, name: str) -> pd.Series:
    if values is None:
        raise ValueError(f"{name} is required")
    if isinstance(values, pd.Series):
        return values.reindex(index).astype(float)
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size != len(index):
        raise ValueError(f"{name} must be 1-dimensional and match asset index size")
    return pd.Series(array, index=index, dtype=float)


def _ensure_dataframe(matrix: pd.DataFrame | np.ndarray, index: Sequence[str]) -> pd.DataFrame:
    if isinstance(matrix, pd.DataFrame):
        return matrix.reindex(index=index, columns=index).astype(float)
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("covariance must be square")
    if array.shape[0] != len(index):
        raise ValueError("covariance dimension mismatch with asset index")
    return pd.DataFrame(array, index=index, columns=index, dtype=float)


def _portfolio_sharpe(mu: pd.Series, cov: pd.DataFrame, assets: Sequence[str]) -> float:
    if not assets:
        return -np.inf
    subset_mu = mu.loc[assets]
    subset_cov = cov.loc[assets, assets]
    weights = np.full(len(assets), 1.0 / len(assets), dtype=float)
    variance = float(weights @ subset_cov.to_numpy() @ weights)
    if variance <= 0:
        return float(subset_mu.mean())
    return float(subset_mu.mean() / np.sqrt(variance))


def _score_candidate(
    mu: pd.Series,
    cov: pd.DataFrame,
    costs: pd.Series,
    candidate: Sequence[str],
) -> float:
    sharpe = _portfolio_sharpe(mu, cov, candidate)
    cost_penalty = float(costs.loc[list(candidate)].mean()) if candidate else 0.0
    return sharpe - cost_penalty


@dataclass(frozen=True)
class SelectionResult:
    assets: list[str]
    weights: pd.Series | None = None
    metadata: Mapping[str, object] | None = None


def greedy_selection(
    mu: Sequence[float] | pd.Series,
    cov: pd.DataFrame | np.ndarray,
    costs: Sequence[float] | pd.Series | None,
    k: int,
    *,
    asset_index: Sequence[str],
) -> list[str]:
    """Select ``k`` assets using a simple forward greedy heuristic."""

    if k <= 0:
        raise ValueError("k must be positive")

    mu_series = _ensure_series(mu, asset_index, name="mu")
    cov_df = _ensure_dataframe(cov, asset_index)
    default_costs = np.zeros(len(asset_index), dtype=float) if costs is None else costs
    cost_series = _ensure_series(default_costs, asset_index, name="costs")

    available = list(asset_index)
    selected: list[str] = []

    while available and len(selected) < k:
        best_asset = None
        best_score = -np.inf
        for asset in available:
            candidate = selected + [asset]
            score = _score_candidate(mu_series, cov_df, cost_series, candidate)
            if score > best_score:
                best_score = score
                best_asset = asset
        if best_asset is None:
            break
        selected.append(best_asset)
        available.remove(best_asset)

    return selected


def beam_search_selection(
    mu: Sequence[float] | pd.Series,
    cov: pd.DataFrame | np.ndarray,
    costs: Sequence[float] | pd.Series | None,
    k: int,
    *,
    asset_index: Sequence[str],
    beam_width: int = 3,
) -> list[str]:
    """Beam-search exploration retaining top ``beam_width`` candidate baskets."""

    if beam_width <= 0:
        raise ValueError("beam_width must be positive")

    mu_series = _ensure_series(mu, asset_index, name="mu")
    cov_df = _ensure_dataframe(cov, asset_index)
    default_costs = np.zeros(len(asset_index), dtype=float) if costs is None else costs
    cost_series = _ensure_series(default_costs, asset_index, name="costs")

    beams: list[list[str]] = [[]]
    for _ in range(min(k, len(asset_index))):
        candidates: list[list[str]] = []
        for basket in beams:
            remaining = [asset for asset in asset_index if asset not in basket]
            for asset in remaining:
                candidate = basket + [asset]
                candidates.append(candidate)
        if not candidates:
            break
        scored = sorted(
            candidates,
            key=lambda basket: _score_candidate(mu_series, cov_df, cost_series, basket),
            reverse=True,
        )
        beams = scored[:beam_width]

    return beams[0] if beams else []


def prune_after_optimisation(
    weights: Sequence[float] | pd.Series,
    k: int,
    *,
    method: str = "magnitude",
) -> pd.Series:
    """Keep top-``k`` holdings according to the specified pruning method."""

    if k <= 0:
        raise ValueError("k must be positive")
    if method not in {"magnitude", "long_only"}:
        raise ValueError("unsupported pruning method")

    series = weights.copy() if isinstance(weights, pd.Series) else pd.Series(weights, dtype=float)
    series = series.astype(float).fillna(0.0)
    if series.size <= k:
        return series / series.sum() if series.sum() != 0 else series

    if method == "long_only":
        series = series.clip(lower=0.0)
    ranking = series.abs().sort_values(ascending=False)
    keep = ranking.index[:k]
    pruned = series.where(series.index.isin(keep), 0.0)
    total = pruned.sum()
    if total == 0:
        return pruned
    return pruned / total


def reoptimize_with_subset(
    subset: Iterable[str],
    data: Mapping[str, object],
    core_solver: Callable[..., Mapping[str, object]],
    *,
    solver_kwargs: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Run the core solver restricted to ``subset`` and return its output."""

    subset_list = list(dict.fromkeys(subset))
    if not subset_list:
        raise ValueError("subset must not be empty")

    full_index = data.get("asset_index", subset_list)
    mu = _ensure_series(data["mu"], full_index, name="mu")
    mu = mu.reindex(subset_list)
    cov_raw = data["cov"]
    if isinstance(cov_raw, pd.DataFrame):
        cov = cov_raw.reindex(index=subset_list, columns=subset_list).astype(float)
    else:
        full_matrix = np.asarray(cov_raw, dtype=float)
        if full_matrix.shape[0] != full_matrix.shape[1]:
            raise ValueError("covariance must be square")
        if full_matrix.shape[0] == len(full_index):
            cov_df = pd.DataFrame(full_matrix, index=full_index, columns=full_index, dtype=float)
            cov = cov_df.reindex(index=subset_list, columns=subset_list)
        elif full_matrix.shape[0] == len(subset_list):
            cov = pd.DataFrame(full_matrix, index=subset_list, columns=subset_list, dtype=float)
        else:
            raise ValueError("covariance dimension mismatch with subset")
    previous_default = data.get("previous_weights")
    if previous_default is None:
        previous_default = pd.Series(np.zeros(len(full_index)), index=full_index, dtype=float)
    previous = _ensure_series(previous_default, full_index, name="previous").reindex(subset_list)

    kwargs = dict(solver_kwargs or {})
    kwargs.setdefault("previous_weights", previous)
    kwargs.setdefault("asset_index", subset_list)
    result = core_solver(mu=mu, cov=cov, **kwargs)
    return result


def cardinality_pipeline(
    data: Mapping[str, object],
    k: int,
    strategy: str,
    *,
    config: Mapping[str, object] | None = None,
) -> SelectionResult:
    """Master orchestrator selecting assets and optionally re-optimising."""

    config = dict(config or {})
    asset_index = list(data["asset_index"])
    mu = _ensure_series(data["mu"], asset_index, name="mu")
    cov = _ensure_dataframe(data["cov"], asset_index)
    default_costs = data.get("costs")
    if default_costs is None:
        default_costs = np.zeros(len(asset_index), dtype=float)
    costs = _ensure_series(default_costs, asset_index, name="costs")

    strategy = strategy.lower()
    if strategy == "greedy":
        selected = greedy_selection(mu, cov, costs, k, asset_index=asset_index)
        return SelectionResult(assets=selected)
    if strategy == "beam":
        beam_width = int(config.get("beam_width", 3))
        selected = beam_search_selection(mu, cov, costs, k, asset_index=asset_index, beam_width=beam_width)
        return SelectionResult(assets=selected, metadata={"beam_width": beam_width})
    if strategy == "prune_then_reopt":
        base_weights = _ensure_series(data.get("weights"), asset_index, name="weights")
        pruned = prune_after_optimisation(base_weights, k, method=str(config.get("method", "magnitude")))
        selected_assets = pruned.loc[pruned.ne(0.0)].index.tolist()
        solver = data.get("core_solver")
        if solver is None:
            raise ValueError("core_solver callable required for prune_then_reopt strategy")
        solver_kwargs = config.get("solver_kwargs")
        result = reoptimize_with_subset(selected_assets, data, solver, solver_kwargs=solver_kwargs)
        weights = result.get("weights")
        weights_series = (
            weights if isinstance(weights, pd.Series) else pd.Series(weights, index=selected_assets, dtype=float)
        )
        return SelectionResult(assets=selected_assets, weights=weights_series, metadata={"reoptimised": True})

    raise ValueError("unsupported cardinality strategy")
