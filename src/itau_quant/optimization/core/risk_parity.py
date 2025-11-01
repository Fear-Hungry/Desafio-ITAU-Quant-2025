"""Risk Parity - igualar contribuições de risco."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "risk_contribution",
    "solve_log_barrier",
    "iterative_risk_parity",
    "cluster_risk_parity",
    "risk_parity",
]


def _ensure_series(
    values: Sequence[float] | pd.Series, index: Sequence[str]
) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.reindex(index).astype(float)
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size != len(index):
        raise ValueError("values must be 1-dimensional and match covariance index")
    return pd.Series(array, index=index, dtype=float)


def _ensure_dataframe(
    matrix: pd.DataFrame | np.ndarray, index: Sequence[str]
) -> pd.DataFrame:
    if isinstance(matrix, pd.DataFrame):
        return matrix.reindex(index=index, columns=index).astype(float)
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("covariance must be square")
    if array.shape[0] != len(index):
        raise ValueError("covariance must match asset index length")
    return pd.DataFrame(array, index=index, columns=index, dtype=float)


def _normalise(weights: pd.Series) -> pd.Series:
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return weights / total


def risk_contribution(
    weights: Sequence[float] | pd.Series, cov: pd.DataFrame | np.ndarray
) -> pd.Series:
    """Return percentage risk contributions of each asset."""

    if isinstance(cov, pd.DataFrame):
        index = list(cov.index)
    else:
        index = list(range(np.asarray(cov).shape[0]))

    weights_series = _ensure_series(weights, index)
    weights_series = _normalise(weights_series.clip(lower=0.0))
    cov_df = _ensure_dataframe(cov, index)

    portfolio_var = float(weights_series.values @ cov_df.values @ weights_series.values)
    if portfolio_var <= 0:
        raise ValueError("portfolio variance must be positive")

    marginal = cov_df.values @ weights_series.values
    contributions = weights_series.values * marginal
    return pd.Series(contributions / portfolio_var, index=index, dtype=float)


def solve_log_barrier(
    cov: pd.DataFrame | np.ndarray,
    target_risk: Sequence[float] | pd.Series | None = None,
    *,
    bounds: tuple[float, float] | None = None,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> pd.Series:
    """Solve risk parity using a log-barrier formulation."""

    cov_df = _ensure_dataframe(
        cov,
        cov.index if isinstance(cov, pd.DataFrame) else range(np.asarray(cov).shape[0]),
    )
    assets = list(cov_df.index)

    if target_risk is None:
        target = np.full(len(assets), 1.0 / len(assets), dtype=float)
    else:
        target_series = _ensure_series(target_risk, assets)
        target = _normalise(target_series).to_numpy(dtype=float)

    lower, upper = bounds or (0.0, 1.0)
    lower = float(lower)
    upper = float(upper)
    if lower < 0 or upper <= 0 or lower >= upper:
        raise ValueError("bounds must satisfy 0 <= lower < upper")

    weights = np.full(len(assets), 1.0 / len(assets), dtype=float)

    for _ in range(max_iter):
        inv_weights = 1.0 / np.clip(weights, lower + 1e-12, None)
        gradient = cov_df.values @ weights - target * np.sum(
            weights * (cov_df.values @ weights)
        )
        gradient -= inv_weights
        hessian = cov_df.values + np.diag(inv_weights**2)

        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break

        step_size = 1.0
        while step_size > 1e-6:
            candidate = weights - step_size * step
            candidate = np.clip(candidate, lower, upper)
            candidate /= candidate.sum()
            if np.all(candidate > lower) and np.linalg.norm(candidate - weights) > 0:
                weights = candidate
                break
            step_size *= 0.5

        if np.linalg.norm(gradient) < tol:
            break

    return pd.Series(weights / weights.sum(), index=assets, dtype=float)


def iterative_risk_parity(
    cov: pd.DataFrame | np.ndarray,
    init_weights: Sequence[float] | pd.Series | None = None,
    *,
    tol: float = 1e-7,
    max_iter: int = 5_000,
) -> pd.Series:
    """Multiplicative iteration to equalise risk contributions."""

    cov_df = _ensure_dataframe(
        cov,
        cov.index if isinstance(cov, pd.DataFrame) else range(np.asarray(cov).shape[0]),
    )
    assets = list(cov_df.index)

    if init_weights is None:
        weights = np.full(len(assets), 1.0 / len(assets), dtype=float)
    else:
        weights = _normalise(_ensure_series(init_weights, assets)).to_numpy(dtype=float)

    for _ in range(max_iter):
        rc = risk_contribution(weights, cov_df)
        if np.allclose(rc, rc.mean(), atol=tol):
            break
        adjustment = rc.mean() / np.clip(rc.to_numpy(dtype=float), 1e-12, None)
        weights *= np.clip(adjustment, 1e-3, 1e3)
        weights = np.clip(weights, 0.0, None)
        weights /= weights.sum()

    return pd.Series(weights, index=assets, dtype=float)


def cluster_risk_parity(
    cov: pd.DataFrame | np.ndarray,
    clusters: Mapping[str, Iterable[str]] | Iterable[Iterable[str]],
    *,
    method: str = "iterative",
    **kwargs: object,
) -> pd.Series:
    """Risk parity applied hierarchically across provided clusters."""

    cov_df = _ensure_dataframe(
        cov,
        cov.index if isinstance(cov, pd.DataFrame) else range(np.asarray(cov).shape[0]),
    )
    assets = list(cov_df.index)

    if isinstance(clusters, Mapping):
        cluster_map = {str(key): list(values) for key, values in clusters.items()}
    else:
        cluster_map = {
            f"cluster_{idx}": list(group) for idx, group in enumerate(clusters)
        }

    filtered_clusters: list[str] = []
    cluster_weights: list[pd.Series] = []
    cluster_assignments: dict[str, str] = {}
    for name, members in cluster_map.items():
        members = [asset for asset in members if asset in assets]
        if not members:
            continue
        sub_cov = cov_df.loc[members, members]
        if method == "log_barrier":
            weights = solve_log_barrier(sub_cov, **kwargs)
        else:
            weights = iterative_risk_parity(sub_cov, **kwargs)
        weights = _normalise(weights)
        cluster_weights.append(weights)
        filtered_clusters.append(name)
        for asset in members:
            cluster_assignments[asset] = name

    if not filtered_clusters:
        raise ValueError("no valid clusters found for risk parity")

    group_cov = pd.DataFrame(
        0.0, index=filtered_clusters, columns=filtered_clusters, dtype=float
    )
    for i, cluster_i in enumerate(filtered_clusters):
        for j, cluster_j in enumerate(filtered_clusters):
            assets_i = [
                asset
                for asset, cluster in cluster_assignments.items()
                if cluster == cluster_i
            ]
            assets_j = [
                asset
                for asset, cluster in cluster_assignments.items()
                if cluster == cluster_j
            ]
            if not assets_i or not assets_j:
                continue
            sub_cov = cov_df.loc[assets_i, assets_j]
            weight_i = (
                cluster_weights[i].reindex(assets_i).fillna(0.0).to_numpy(dtype=float)
            )
            weight_j = (
                cluster_weights[j].reindex(assets_j).fillna(0.0).to_numpy(dtype=float)
            )
            group_cov.loc[cluster_i, cluster_j] = float(
                weight_i @ sub_cov.to_numpy(dtype=float) @ weight_j
            )

    if method == "log_barrier":
        cluster_alloc = solve_log_barrier(group_cov, **kwargs)
    else:
        cluster_alloc = iterative_risk_parity(group_cov, **kwargs)

    final_weights = pd.Series(0.0, index=assets, dtype=float)
    for cluster_name, weights in zip(filtered_clusters, cluster_weights):
        alloc = float(cluster_alloc.get(cluster_name, 0.0))
        final_weights.loc[weights.index] = alloc * weights
    return _normalise(final_weights)


@dataclass(frozen=True)
class RiskParityResult:
    weights: pd.Series
    contributions: pd.Series
    method: str
    converged: bool
    notes: list[str]


def risk_parity(
    cov: pd.DataFrame | np.ndarray,
    *,
    init_weights: Sequence[float] | pd.Series | None = None,
    config: Mapping[str, object] | None = None,
) -> RiskParityResult:
    """High-level wrapper computing a risk parity allocation."""

    config = dict(config or {})
    method = str(config.get("method", "iterative")).lower()

    cov_df = _ensure_dataframe(
        cov,
        cov.index if isinstance(cov, pd.DataFrame) else range(np.asarray(cov).shape[0]),
    )
    assets = list(cov_df.index)

    weights: pd.Series
    converged = True
    notes: list[str] = []

    try:
        if method == "log_barrier":
            weights = solve_log_barrier(
                cov_df,
                target_risk=config.get("target_risk"),
                bounds=config.get("bounds"),
                max_iter=int(config.get("max_iter", 500)),
                tol=float(config.get("tol", 1e-8)),
            )
        elif method == "cluster":
            cluster_cfg = config.get("clusters")
            if not cluster_cfg:
                raise ValueError(
                    "cluster risk parity requires 'clusters' configuration"
                )
            weights = cluster_risk_parity(
                cov_df,
                cluster_cfg,
                method=str(config.get("cluster_method", "iterative")),
            )
        else:
            weights = iterative_risk_parity(
                cov_df,
                init_weights=init_weights,
                tol=float(config.get("tol", 1e-7)),
                max_iter=int(config.get("max_iter", 5_000)),
            )
    except Exception as exc:  # pragma: no cover - defensive guard
        converged = False
        notes.append(f"risk parity failed with {exc}; falling back to equal weight")
        weights = pd.Series(1.0 / len(assets), index=assets, dtype=float)

    contributions = risk_contribution(weights, cov_df)
    return RiskParityResult(
        weights=_normalise(weights),
        contributions=contributions,
        method=method,
        converged=converged,
        notes=notes,
    )
