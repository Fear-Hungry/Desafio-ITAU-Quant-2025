"""Composable CVXPy constraint builders shared across optimisation problems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import cvxpy as cp
import numpy as np
import pandas as pd

__all__ = [
    "BudgetConfig",
    "build_budget_constraints",
    "build_bound_constraints",
    "build_turnover_constraints",
    "build_sector_exposure_constraints",
    "build_risk_constraints",
    "compose_constraints",
]


@dataclass(frozen=True)
class BudgetConfig:
    """Configuration for budget-style constraints."""

    target: float = 1.0
    lower: float | None = None
    upper: float | None = None
    max_leverage: float | None = None
    min_leverage: float | None = None


def _ensure_numpy(value: float | Sequence[float] | pd.Series, size: int, *, name: str) -> np.ndarray:
    array: np.ndarray
    if isinstance(value, pd.Series):
        array = value.to_numpy(dtype=float)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        array = np.asarray(value, dtype=float)
    else:
        array = np.full(size, float(value), dtype=float)
    if array.ndim != 1 or array.size != size:
        raise ValueError(f"{name} must be 1-dimensional with length {size}")
    return array


def build_budget_constraints(weights: cp.Expression, config: BudgetConfig | Mapping[str, float] | None) -> list[cp.Constraint]:
    """Build budget and leverage constraints."""

    if config is None:
        return []
    if isinstance(config, Mapping):
        resolved = BudgetConfig(
            target=float(config.get("target", 1.0)),
            lower=config.get("lower"),
            upper=config.get("upper"),
            max_leverage=config.get("max_leverage"),
            min_leverage=config.get("min_leverage"),
        )
    else:
        resolved = config

    constraints: list[cp.Constraint] = []

    target = float(resolved.target)
    if resolved.lower is not None or resolved.upper is not None:
        if resolved.lower is not None:
            constraints.append(cp.sum(weights) >= float(resolved.lower))
        if resolved.upper is not None:
            constraints.append(cp.sum(weights) <= float(resolved.upper))
    else:
        constraints.append(cp.sum(weights) == target)

    if resolved.max_leverage is not None:
        constraints.append(cp.norm1(weights) <= float(resolved.max_leverage))

    if resolved.min_leverage is not None:
        constraints.append(cp.norm1(weights) >= float(resolved.min_leverage))

    return constraints


def build_bound_constraints(
    weights: cp.Expression,
    lower: float | Sequence[float] | pd.Series,
    upper: float | Sequence[float] | pd.Series,
) -> list[cp.Constraint]:
    """Build per-asset bound constraints."""

    size = weights.shape[0]
    if size is None:
        raise ValueError("weights variable must be 1-dimensional with known length")
    lower_array = _ensure_numpy(lower, size, name="lower bounds")
    upper_array = _ensure_numpy(upper, size, name="upper bounds")
    if np.any(lower_array > upper_array):
        raise ValueError("lower bounds exceed upper bounds for at least one asset")
    return [weights >= lower_array, weights <= upper_array]


def build_turnover_constraints(
    weights: cp.Expression,
    previous_weights: Sequence[float] | pd.Series,
    max_turnover: float,
) -> list[cp.Constraint]:
    """Limit L1 turnover relative to previous holdings."""

    size = weights.shape[0]
    if size is None:
        raise ValueError("weights variable must be 1-dimensional with known length")
    previous_array = _ensure_numpy(previous_weights, size, name="previous weights")
    if max_turnover < 0:
        raise ValueError("max_turnover must be non-negative")
    if max_turnover == 0:
        return [weights == previous_array]
    return [cp.norm1(weights - previous_array) <= float(max_turnover)]


def build_sector_exposure_constraints(
    weights: cp.Expression,
    sector_map: Mapping[str, str] | pd.Series | Sequence[str],
    limits: Mapping[str, tuple[float | None, float | None] | Mapping[str, float]],
    *,
    asset_index: Sequence[str] | None = None,
) -> list[cp.Constraint]:
    """Restrict aggregate exposure by sector or cluster."""

    size = weights.shape[0]
    if size is None:
        raise ValueError("weights variable must be 1-dimensional with known length")

    if isinstance(sector_map, pd.Series):
        if asset_index is None:
            asset_index = list(sector_map.index)
        sector_series = sector_map.reindex(asset_index).astype(str)
    elif isinstance(sector_map, Mapping):
        if asset_index is None:
            raise ValueError("asset_index is required when sector_map is a mapping")
        sector_series = pd.Series(sector_map, dtype=str).reindex(asset_index)
    else:
        labels = list(sector_map)
        if asset_index is None:
            asset_index = list(range(len(labels)))
        if len(labels) != len(asset_index):
            raise ValueError("sector_map length must match asset index length")
        sector_series = pd.Series(labels, index=asset_index, dtype=str)

    if len(asset_index) != size:
        raise ValueError("asset_index length mismatch with weights dimension")

    constraints: list[cp.Constraint] = []
    for sector, bound in limits.items():
        if isinstance(bound, Mapping):
            lower = bound.get("min")
            upper = bound.get("max")
        else:
            lower, upper = bound
        mask = (sector_series == sector).astype(float).to_numpy()
        if mask.sum() == 0:
            continue
        exposure = mask @ weights
        if lower is not None:
            constraints.append(exposure >= float(lower))
        if upper is not None:
            constraints.append(exposure <= float(upper))
    return constraints


def _as_matrix(value: pd.DataFrame | np.ndarray, *, index: Sequence[str] | None = None) -> tuple[np.ndarray, list[str]]:
    if isinstance(value, pd.DataFrame):
        if index is not None:
            matrix = value.reindex(index=index, columns=index).to_numpy(dtype=float)
            return matrix, list(index)
        return value.to_numpy(dtype=float), list(value.index)
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("covariance matrix must be square")
    if index is None:
        index = list(range(array.shape[0]))
    return array, list(index)


def build_risk_constraints(
    weights: cp.Expression,
    covariance: pd.DataFrame | np.ndarray | None,
    config: Mapping[str, object] | None,
) -> list[cp.Constraint]:
    """Build risk-oriented constraints (volatility, variance, CVaR, tracking-error)."""

    if not config:
        return []

    constraints: list[cp.Constraint] = []
    cov_matrix: np.ndarray | None = None

    if any(key in config for key in ("volatility", "variance", "tracking_error")):
        if covariance is None:
            raise ValueError("covariance is required for volatility or tracking-error constraints")
        cov_matrix, _ = _as_matrix(covariance)

    if "volatility" in config:
        vol_cfg = config["volatility"]
        if not isinstance(vol_cfg, Mapping) or "max" not in vol_cfg:
            raise ValueError("volatility constraint requires a 'max' value")
        sigma_max = float(vol_cfg["max"])
        constraints.append(cp.quad_form(weights, cov_matrix) <= sigma_max**2)

    if "variance" in config:
        var_cfg = config["variance"]
        if not isinstance(var_cfg, Mapping) or "max" not in var_cfg:
            raise ValueError("variance constraint requires a 'max' value")
        constraints.append(cp.quad_form(weights, cov_matrix) <= float(var_cfg["max"]))

    if "tracking_error" in config:
        te_cfg = config["tracking_error"]
        if not isinstance(te_cfg, Mapping):
            raise ValueError("tracking_error constraint must be provided as mapping")
        benchmark = np.asarray(te_cfg.get("benchmark"), dtype=float)
        if benchmark.ndim != 1 or benchmark.size != cov_matrix.shape[0]:
            raise ValueError("benchmark weights must match covariance dimension")
        diff = weights - benchmark
        constraints.append(cp.quad_form(diff, cov_matrix) <= float(te_cfg["max"]) ** 2)

    if "cvar" in config:
        cvar_cfg = config["cvar"]
        if not isinstance(cvar_cfg, Mapping):
            raise ValueError("cvar constraint must be provided as mapping")
        scenario_returns = np.asarray(cvar_cfg.get("scenario_returns"), dtype=float)
        if scenario_returns.ndim != 2 or scenario_returns.shape[1] != weights.shape[0]:
            raise ValueError("scenario_returns must have shape (n_scenarios, n_assets)")
        alpha = float(cvar_cfg.get("alpha", 0.95))
        max_cvar = float(cvar_cfg["max"])
        num_scenarios = scenario_returns.shape[0]
        losses = -scenario_returns @ weights
        aux = cp.Variable(num_scenarios)
        t_var = cp.Variable()
        constraints.extend(
            [
                aux >= 0,
                aux >= losses - t_var,
                t_var + (1.0 / ((1.0 - alpha) * num_scenarios)) * cp.sum(aux) <= max_cvar,
            ]
        )

    return constraints


def compose_constraints(
    weights: cp.Expression,
    config: Mapping[str, object] | None,
    *,
    asset_index: Sequence[str],
    previous_weights: Sequence[float] | pd.Series | None = None,
    covariance: pd.DataFrame | np.ndarray | None = None,
) -> list[cp.Constraint]:
    """Compose constraints according to a declarative configuration."""

    if not config:
        return []

    constraints: list[cp.Constraint] = []

    if "budget" in config:
        constraints.extend(build_budget_constraints(weights, config["budget"]))

    if "bounds" in config:
        bounds_cfg = config["bounds"]
        if not isinstance(bounds_cfg, Mapping):
            raise TypeError("bounds configuration must be a mapping")
        lower = bounds_cfg.get("lower", 0.0)
        upper = bounds_cfg.get("upper", 1.0)
        constraints.extend(build_bound_constraints(weights, lower, upper))

    if "turnover" in config:
        turnover_cfg = config["turnover"]
        if not isinstance(turnover_cfg, Mapping):
            raise TypeError("turnover configuration must be a mapping")
        prev = turnover_cfg.get("previous") or previous_weights
        if prev is None:
            raise ValueError("turnover constraint requires previous weights")
        max_turnover = float(turnover_cfg.get("max_turnover", turnover_cfg.get("max")))
        constraints.extend(build_turnover_constraints(weights, prev, max_turnover))

    if "sector" in config:
        sector_cfg = config["sector"]
        if not isinstance(sector_cfg, Mapping):
            raise TypeError("sector configuration must be a mapping")
        sector_map = sector_cfg.get("map")
        limits = sector_cfg.get("limits")
        if sector_map is None or limits is None:
            raise ValueError("sector constraint requires 'map' and 'limits'")
        sector_index = sector_cfg.get("asset_index", asset_index)
        constraints.extend(
            build_sector_exposure_constraints(
                weights,
                sector_map,
                limits,
                asset_index=sector_index,
            )
        )

    if "risk" in config:
        risk_cfg = config["risk"]
        if not isinstance(risk_cfg, Mapping):
            raise TypeError("risk configuration must be a mapping")
        risk_cov = risk_cfg.get("covariance", covariance)
        constraints.extend(build_risk_constraints(weights, risk_cov, risk_cfg))

    return constraints
