"""Utilities to convert continuous weights into executable orders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

__all__ = [
    "RoundingResult",
    "weights_to_shares",
    "round_to_lots",
    "shares_to_weights",
    "allocate_residual_cash",
    "estimate_rounding_costs",
    "rounding_pipeline",
]


@dataclass(frozen=True)
class RoundingResult:
    """Outcome of the rounding pipeline."""

    original_weights: pd.Series
    weights: pd.Series
    shares: pd.Series
    rounded_weights: pd.Series
    lot_sizes: pd.Series
    residual_cash: float
    rounding_cost: float

    def to_dict(self) -> dict[str, float]:
        return {
            "residual_cash": float(self.residual_cash),
            "rounding_cost": float(self.rounding_cost),
        }


def weights_to_shares(
    weights: pd.Series,
    capital: float,
    prices: pd.Series,
) -> pd.Series:
    """Convert weights to share quantities before rounding."""

    aligned_prices = prices.reindex(weights.index).astype(float)
    if aligned_prices.isnull().any():
        missing = aligned_prices[aligned_prices.isnull()].index.tolist()
        raise ValueError(f"Missing prices for assets: {missing}")
    notional = weights.astype(float) * float(capital)
    shares = notional / aligned_prices
    return shares.fillna(0.0)


def round_to_lots(
    shares: pd.Series,
    lot_size_map: Mapping[str, float] | pd.Series | float,
    *,
    method: str = "nearest",
) -> pd.Series:
    """Round share quantities to valid lot sizes."""

    if isinstance(lot_size_map, (int, float)):
        lot_sizes = pd.Series(float(lot_size_map), index=shares.index)
    else:
        lot_sizes = pd.Series(lot_size_map, dtype=float).reindex(shares.index).fillna(1.0)
    lot_sizes = lot_sizes.clip(lower=1e-9)

    scaled = shares / lot_sizes
    if method == "nearest":
        rounded_units = np.round(scaled)
    elif method in {"floor", "down"}:
        rounded_units = np.floor(scaled)
    elif method in {"ceil", "up"}:
        rounded_units = np.ceil(scaled)
    else:
        raise ValueError(f"Unsupported rounding method '{method}'.")
    rounded_shares = pd.Series(rounded_units, index=shares.index, dtype=float) * lot_sizes
    return rounded_shares


def shares_to_weights(shares: pd.Series, capital: float, prices: pd.Series) -> pd.Series:
    """Convert share quantities back to weights."""

    aligned_prices = prices.reindex(shares.index).astype(float)
    capital_used = (shares.astype(float) * aligned_prices).sum()
    if capital <= 0:
        raise ValueError("capital must be positive.")
    weights = (shares * aligned_prices) / float(capital)
    return weights.fillna(0.0), float(capital_used)


def allocate_residual_cash(
    weights: pd.Series,
    residual_cash: float,
    capital: float,
    *,
    priority: str = "largest_weight",
) -> pd.Series:
    """Distribute residual cash weight over assets according to priority."""

    if abs(residual_cash) < 1e-12:
        return weights

    residual_weight = residual_cash / float(capital)
    adjusted = weights.copy()
    if priority == "largest_weight":
        target_asset = adjusted.abs().idxmax()
        adjusted[target_asset] += residual_weight
    elif priority == "spread":
        increment = residual_weight / len(adjusted)
        adjusted = adjusted + increment
    else:
        raise ValueError(f"Unsupported priority '{priority}'.")
    return adjusted


def _linear_cost_from_config(
    diff_weights: pd.Series,
    capital: float,
    cost_model: Mapping[str, float] | None,
) -> float:
    if not cost_model:
        return 0.0
    linear = cost_model.get("linear_bps", 0.0)
    if isinstance(linear, Mapping):
        series = pd.Series({k: float(v) for k, v in linear.items()}, dtype=float)
        linear_series = series.reindex(diff_weights.index).fillna(series.mean())
    else:
        linear_series = pd.Series(float(linear), index=diff_weights.index)
    trade_notional = diff_weights.abs() * float(capital)
    cost = (trade_notional * (linear_series / 10_000.0)).sum()
    return float(cost)


def estimate_rounding_costs(
    original_weights: pd.Series,
    rounded_weights: pd.Series,
    *,
    capital: float,
    cost_model: Mapping[str, float] | None = None,
) -> float:
    """Estimate the monetary cost introduced by rounding."""

    diff = (rounded_weights.reindex(original_weights.index, fill_value=0.0) - original_weights).fillna(0.0)
    return _linear_cost_from_config(diff, capital, cost_model)


def rounding_pipeline(
    weights: pd.Series,
    prices: pd.Series,
    capital: float,
    config: Mapping[str, object] | None = None,
) -> RoundingResult:
    """Run the full rounding pipeline returning shares/weights/costs."""

    config = dict(config or {})
    lot_sizes = config.get("lot_sizes", 1.0)
    method = str(config.get("method", "nearest")).lower()
    cost_model = config.get("cost_model")
    priority = config.get("priority", "largest_weight")
    allocate = bool(config.get("allocate_residual", False))

    base_weights = weights.astype(float)
    shares = weights_to_shares(base_weights, capital, prices)
    rounded_shares = round_to_lots(shares, lot_sizes, method=method)
    rounded_weights, capital_used = shares_to_weights(rounded_shares, capital, prices)

    residual_cash = float(capital - capital_used)
    if allocate and abs(residual_cash) > 1e-9:
        rounded_weights = allocate_residual_cash(
            rounded_weights,
            residual_cash,
            capital,
            priority=str(priority),
        )

    rounding_cost = estimate_rounding_costs(
        base_weights,
        rounded_weights,
        capital=capital,
        cost_model=cost_model if isinstance(cost_model, Mapping) else None,
    )

    lot_series = (
        pd.Series(lot_sizes, dtype=float).reindex(base_weights.index).fillna(1.0)
        if not isinstance(lot_sizes, (int, float))
        else pd.Series(float(lot_sizes), index=base_weights.index)
    )

    return RoundingResult(
        original_weights=base_weights,
        weights=base_weights,
        shares=rounded_shares.astype(float),
        rounded_weights=rounded_weights.astype(float),
        lot_sizes=lot_series.astype(float),
        residual_cash=residual_cash,
        rounding_cost=rounding_cost,
    )
