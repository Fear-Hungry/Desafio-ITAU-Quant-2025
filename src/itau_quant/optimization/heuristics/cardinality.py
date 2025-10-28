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
    # New: N_eff-based cardinality
    "compute_effective_number",
    "suggest_k_from_neff",
    "suggest_k_from_costs",
    "suggest_k_dynamic",
    "smart_topk_score",
    "select_support_topk",
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


# ============================================================================
# N_eff-based Cardinality (Dynamic K Calibration)
# ============================================================================


def compute_effective_number(weights: pd.Series) -> float:
    """Compute effective number of assets N_eff = 1 / Σw².

    Args:
        weights: Portfolio weights

    Returns:
        Effective number (between 1 and len(weights))

    Examples:
        >>> w = pd.Series([0.25, 0.25, 0.25, 0.25])
        >>> compute_effective_number(w)
        4.0
        >>> w = pd.Series([1.0, 0.0, 0.0])
        >>> compute_effective_number(w)
        1.0
    """
    w = weights[weights > 0]
    if len(w) == 0:
        return 0.0
    return float(1.0 / (w**2).sum())


def suggest_k_from_neff(
    neff: float,
    k_min: int,
    k_max: int,
    multiplier: float = 0.8,
) -> int:
    """Suggest K from N_eff using K ≈ multiplier · N_eff.

    Args:
        neff: Effective number from unconstrained solution
        k_min: Minimum K
        k_max: Maximum K
        multiplier: Scaling factor (default 0.8)

    Returns:
        Suggested K

    Examples:
        >>> suggest_k_from_neff(30.0, k_min=12, k_max=32)
        24
        >>> suggest_k_from_neff(10.0, k_min=12, k_max=32)
        12
    """
    k_target = int(np.floor(multiplier * neff))
    return max(k_min, min(k_max, k_target))


def suggest_k_from_costs(
    costs_bps: pd.Series,
    k_min: int = 12,
    k_max: int = 36,
) -> tuple[int, int]:
    """Suggest K range based on cost distribution.

    Args:
        costs_bps: Cost in bps for each asset
        k_min: Absolute minimum K
        k_max: Absolute maximum K

    Returns:
        (k_suggested_min, k_suggested_max) tuple

    Examples:
        >>> costs = pd.Series([8, 8, 10, 12, 15])
        >>> suggest_k_from_costs(costs)  # Mostly cheap
        (28, 32)
    """
    median_cost = costs_bps.median()
    mean_cost = costs_bps.mean()
    avg_cost = (median_cost + mean_cost) / 2

    if avg_cost <= 10:
        return (max(k_min, 28), min(k_max, 36))
    elif avg_cost <= 25:
        return (max(k_min, 18), min(k_max, 26))
    else:
        return (max(k_min, 12), min(k_max, 18))


def suggest_k_dynamic(
    neff: float,
    costs_bps: pd.Series,
    k_min: int = 12,
    k_max: int = 32,
    neff_multiplier: float = 0.8,
) -> dict[str, float | int | tuple[int, int]]:
    """Combine N_eff and cost analysis for K suggestion.

    Args:
        neff: Effective number
        costs_bps: Cost in bps per asset
        k_min: Minimum K
        k_max: Maximum K
        neff_multiplier: Multiplier for N_eff

    Returns:
        Dict with k_suggested, k_from_neff, k_range_from_cost, neff, avg_cost_bps
    """
    k_from_neff = suggest_k_from_neff(neff, k_min, k_max, neff_multiplier)
    k_cost_min, k_cost_max = suggest_k_from_costs(costs_bps, k_min, k_max)

    k_final = max(k_cost_min, min(k_cost_max, k_from_neff))

    return {
        "k_suggested": k_final,
        "k_from_neff": k_from_neff,
        "k_range_from_cost": (k_cost_min, k_cost_max),
        "neff": neff,
        "avg_cost_bps": float(costs_bps.mean()),
    }


def smart_topk_score(
    weights: pd.Series,
    weights_prev: pd.Series | None = None,
    mu: pd.Series | None = None,
    costs_bps: pd.Series | None = None,
    alpha_weight: float = 1.0,
    alpha_turnover: float = -0.2,
    alpha_return: float = 0.1,
    alpha_cost: float = -0.15,
) -> pd.Series:
    """Compute smart score for asset ranking.

    score = α_w·w + α_to·|w-w_prev| + α_μ·μ + α_c·cost

    Args:
        weights: Current QP weights
        weights_prev: Previous weights (for turnover penalty)
        mu: Expected returns (for return bonus)
        costs_bps: Transaction costs (for cost penalty)
        alpha_weight: Weight coefficient
        alpha_turnover: Turnover penalty (should be negative)
        alpha_return: Return bonus
        alpha_cost: Cost penalty (should be negative)

    Returns:
        Score series (higher = better)
    """
    score = alpha_weight * weights

    if weights_prev is not None:
        w_prev_aligned = weights_prev.reindex(weights.index, fill_value=0.0)
        turnover = (weights - w_prev_aligned).abs()
        score += alpha_turnover * turnover

    if mu is not None:
        mu_aligned = mu.reindex(weights.index, fill_value=0.0)
        score += alpha_return * mu_aligned

    if costs_bps is not None:
        costs_aligned = costs_bps.reindex(weights.index, fill_value=15.0)
        costs_norm = (costs_aligned - costs_aligned.min()) / (costs_aligned.max() - costs_aligned.min() + 1e-8)
        score += alpha_cost * costs_norm

    return score


def select_support_topk(
    weights: pd.Series,
    k: int,
    weights_prev: pd.Series | None = None,
    mu: pd.Series | None = None,
    costs_bps: pd.Series | None = None,
    alpha_weight: float = 1.0,
    alpha_turnover: float = -0.2,
    alpha_return: float = 0.1,
    alpha_cost: float = -0.15,
    tie_breaker: str = "low_turnover",
    epsilon: float = 1e-4,
) -> pd.Index:
    """Select K assets using smart scoring.

    Args:
        weights: Current weights
        k: Number of assets to select
        weights_prev: Previous weights
        mu: Expected returns
        costs_bps: Transaction costs
        alpha_weight: Weight coefficient
        alpha_turnover: Turnover penalty
        alpha_return: Return bonus
        alpha_cost: Cost penalty
        tie_breaker: "low_turnover", "high_return", or "high_weight"
        epsilon: Threshold for zero weights

    Returns:
        Index of selected K assets
    """
    significant = weights[weights > epsilon]
    if len(significant) <= k:
        return significant.index

    scores = smart_topk_score(
        significant,
        weights_prev=weights_prev,
        mu=mu,
        costs_bps=costs_bps,
        alpha_weight=alpha_weight,
        alpha_turnover=alpha_turnover,
        alpha_return=alpha_return,
        alpha_cost=alpha_cost,
    )

    ranked = scores.sort_values(ascending=False)
    kth_score = ranked.iloc[k - 1]
    ties = ranked[ranked == kth_score]

    if len(ties) > 1:
        if tie_breaker == "low_turnover" and weights_prev is not None:
            w_prev_aligned = weights_prev.reindex(ties.index, fill_value=0.0)
            turnover = (weights.reindex(ties.index) - w_prev_aligned).abs()
            tie_winner = turnover.idxmin()
        elif tie_breaker == "high_return" and mu is not None:
            mu_aligned = mu.reindex(ties.index, fill_value=0.0)
            tie_winner = mu_aligned.idxmax()
        else:
            tie_winner = weights.reindex(ties.index).idxmax()

        selected = ranked.iloc[: k - 1].index.tolist()
        selected.append(tie_winner)
        return pd.Index(selected)

    return ranked.iloc[:k].index
