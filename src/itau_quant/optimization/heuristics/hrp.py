"""Hierarchical Risk Parity e baselines heurísticos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans

__all__ = [
    "equal_weight",
    "inverse_variance_portfolio",
    "hierarchical_risk_parity",
    "cluster_then_allocate",
    "heuristic_allocation",
]


def _ensure_series(values: Sequence[float] | pd.Series, index: Sequence[str]) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.reindex(index).astype(float)
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size != len(index):
        raise ValueError("values must be 1-dimensional and match asset index")
    return pd.Series(array, index=index, dtype=float)


def _ensure_dataframe(matrix: pd.DataFrame | np.ndarray, index: Sequence[str]) -> pd.DataFrame:
    if isinstance(matrix, pd.DataFrame):
        return matrix.reindex(index=index, columns=index).astype(float)
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("matrix must be square")
    if array.shape[0] != len(index):
        raise ValueError("matrix dimension mismatch with asset index")
    return pd.DataFrame(array, index=index, columns=index, dtype=float)


def _normalise(weights: pd.Series) -> pd.Series:
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("weights must sum to positive value")
    return weights / total


def _distance_matrix(cov: pd.DataFrame) -> np.ndarray:
    corr = cov.corr()
    corr = corr.fillna(1.0)
    dist = np.sqrt(0.5 * (1.0 - corr.to_numpy(dtype=float)))
    dist[np.isnan(dist)] = 0.0
    return dist


def equal_weight(assets: Sequence[str]) -> pd.Series:
    """Return 1/N allocation over ``assets``."""

    assets = list(dict.fromkeys(assets))
    if not assets:
        raise ValueError("assets must not be empty")
    weight = 1.0 / len(assets)
    return pd.Series(weight, index=assets, dtype=float)


def inverse_variance_portfolio(cov: pd.DataFrame | np.ndarray) -> pd.Series:
    """Allocate weights proportional to inverse variances."""

    if isinstance(cov, pd.DataFrame):
        assets = list(cov.index)
    else:
        assets = list(range(np.asarray(cov).shape[0]))
    cov_df = _ensure_dataframe(cov, assets)
    variances = np.diag(cov_df.to_numpy(dtype=float))
    if np.any(variances <= 0):
        raise ValueError("variances must be positive")
    inv_var = 1.0 / variances
    weights = pd.Series(inv_var, index=assets, dtype=float)
    return _normalise(weights)


def _quasi_diagonalise(linkage_matrix: np.ndarray) -> list[int]:
    sort_ix = dendrogram(linkage_matrix, no_plot=True)["leaves"]
    return list(sort_ix)


def _recursive_bisection(cov: pd.DataFrame, ordered_assets: Sequence[int]) -> pd.Series:
    weights = pd.Series(1.0, index=ordered_assets, dtype=float)

    def split(cluster: Sequence[int]) -> None:
        if len(cluster) <= 1:
            return
        mid = len(cluster) // 2
        left = cluster[:mid]
        right = cluster[mid:]
        cov_left = cov.loc[left, left]
        cov_right = cov.loc[right, right]
        risk_left = np.sum(cov_left.values)
        risk_right = np.sum(cov_right.values)
        allocation_left = risk_right / (risk_left + risk_right)
        allocation_right = 1.0 - allocation_left
        weights.loc[left] *= allocation_left
        weights.loc[right] *= allocation_right
        split(left)
        split(right)

    split(list(ordered_assets))
    return _normalise(weights)


def hierarchical_risk_parity(
    cov: pd.DataFrame | np.ndarray,
    method: str = "single",
    *,
    return_details: bool = False,
) -> pd.Series | tuple[pd.Series, dict[str, object]]:
    """Hierarchical Risk Parity following López de Prado."""

    if isinstance(cov, pd.DataFrame):
        assets = list(cov.index)
    else:
        assets = list(range(np.asarray(cov).shape[0]))

    cov_df = _ensure_dataframe(cov, assets)
    dist = _distance_matrix(cov_df)
    linkage_matrix = linkage(squareform(dist, checks=False), method=method)
    order = _quasi_diagonalise(linkage_matrix)
    ordered_assets = [assets[i] for i in order]
    ordered_cov = cov_df.loc[ordered_assets, ordered_assets]
    weights = _recursive_bisection(ordered_cov, ordered_assets)
    weights = weights.reindex(assets).fillna(0.0)
    diagnostics = {
        "ordered_assets": ordered_assets,
        "linkage_method": method,
    }
    return (weights, diagnostics) if return_details else weights


def cluster_then_allocate(
    returns: pd.DataFrame,
    n_clusters: int = 3,
    *,
    method: str = "kmeans",
    return_details: bool = False,
) -> pd.Series | tuple[pd.Series, dict[str, object]]:
    """Cluster assets then apply equal-weight within clusters."""

    if returns.empty:
        raise ValueError("returns must not be empty")
    assets = list(returns.columns)
    cov = returns.cov()
    if method.lower() != "kmeans":
        raise ValueError("only kmeans clustering is currently supported")
    model = KMeans(n_clusters=min(n_clusters, len(assets)), n_init=10, random_state=42)
    labels = model.fit_predict(cov.to_numpy(dtype=float))
    weights = pd.Series(0.0, index=assets, dtype=float)
    for label in np.unique(labels):
        members = [asset for asset, cluster in zip(assets, labels) if cluster == label]
        if members:
            weights.loc[members] = 1.0 / len(members)
    weights = _normalise(weights)
    diagnostics = {
        "labels": {asset: int(label) for asset, label in zip(assets, labels)},
        "cluster_method": method,
        "n_clusters": int(len(np.unique(labels))),
    }
    return (weights, diagnostics) if return_details else weights


@dataclass(frozen=True)
class HeuristicResult:
    weights: pd.Series
    method: str
    diagnostics: Mapping[str, object] | None = None


def heuristic_allocation(
    data: Mapping[str, object],
    *,
    method: str,
    config: Mapping[str, object] | None = None,
) -> HeuristicResult:
    """Dispatch to heuristic allocator according to ``method``."""

    config = dict(config or {})
    method = method.lower()

    if method == "equal_weight":
        assets = data.get("assets")
        if not assets:
            assets = data.get("asset_index")
        if not assets:
            raise ValueError("equal_weight requires 'assets' list")
        weights = equal_weight(assets)
        diagnostics = {"count": len(weights)}
        return HeuristicResult(weights=weights, method=method, diagnostics=diagnostics)

    if method == "inverse_variance":
        cov = data.get("covariance")
        if cov is None:
            cov = data.get("cov")
        if cov is None:
            raise ValueError("inverse_variance requires 'covariance'")
        cov_df = _ensure_dataframe(cov, cov.index if isinstance(cov, pd.DataFrame) else range(np.asarray(cov).shape[0]))
        weights = inverse_variance_portfolio(cov_df)
        variances = np.diag(cov_df.to_numpy(dtype=float))
        diagnostics = {
            "variances": {asset: float(var) for asset, var in zip(cov_df.index, variances)},
        }
        return HeuristicResult(weights=weights, method=method, diagnostics=diagnostics)

    if method == "hrp":
        cov = data.get("covariance")
        if cov is None:
            cov = data.get("cov")
        if cov is None:
            raise ValueError("HRP requires 'covariance'")
        weights, diagnostics = hierarchical_risk_parity(
            cov,
            method=config.get("linkage", "single"),
            return_details=True,
        )
        return HeuristicResult(weights=weights, method=method, diagnostics=diagnostics)

    if method == "cluster":
        returns = data.get("returns")
        if returns is None:
            raise ValueError("cluster allocation requires 'returns'")
        weights, diagnostics = cluster_then_allocate(
            returns,
            n_clusters=int(config.get("n_clusters", 3)),
            method=str(config.get("cluster_method", "kmeans")),
            return_details=True,
        )
        return HeuristicResult(weights=weights, method=method, diagnostics=diagnostics)

    raise ValueError(f"Unsupported heuristic method '{method}'")
