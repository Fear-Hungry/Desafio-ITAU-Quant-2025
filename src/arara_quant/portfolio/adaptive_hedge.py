"""Adaptive tail hedge allocation based on market regime.

This module implements dynamic tail hedge sizing that scales with market conditions.
In calm markets, hedge allocation is minimal to reduce drag. In stress/crash regimes,
hedge allocation increases to protect against tail risk.

Regime Mapping
--------------
- **calm**: Low volatility (≤12%), no drawdown → Minimal hedge (2-3%)
- **neutral**: Normal volatility (12-25%) → Base hedge (5%)
- **stressed**: High volatility (≥25%) → Elevated hedge (10-12%)
- **crash**: Severe drawdown (≤-15%) → Maximum hedge (15%)

Key Functions
-------------
- `compute_hedge_allocation`: Compute target hedge% based on regime
- `apply_hedge_rebalance`: Adjust portfolio weights to meet hedge target
- `evaluate_hedge_performance`: Measure hedge effectiveness in OOS periods
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from arara_quant.risk.regime import RegimeSnapshot

__all__ = [
    "compute_hedge_allocation",
    "apply_hedge_rebalance",
    "evaluate_hedge_performance",
]


def compute_hedge_allocation(
    regime: str | RegimeSnapshot,
    *,
    config: Mapping[str, Any] | None = None,
) -> float:
    """Compute target tail hedge allocation based on market regime.

    Parameters
    ----------
    regime : str or RegimeSnapshot
        Current market regime (calm, neutral, stressed, crash)
    config : Mapping[str, Any], optional
        Adaptive hedge configuration with keys:
        - base_allocation: float (default: 0.05)
        - regime_multipliers: dict[str, float]
        - max_allocation: dict[str, float]
        - min_allocation: float (default: 0.02)

    Returns
    -------
    float
        Target allocation to tail hedge assets (0.0 to 1.0)

    Examples
    --------
    >>> config = {
    ...     "base_allocation": 0.05,
    ...     "regime_multipliers": {"calm": 0.5, "stressed": 2.0, "crash": 3.0},
    ...     "max_allocation": {"calm": 0.03, "stressed": 0.12, "crash": 0.15},
    ... }
    >>> compute_hedge_allocation("calm", config=config)
    0.025  # base * 0.5 = 2.5%

    >>> compute_hedge_allocation("crash", config=config)
    0.15  # base * 3.0 = 15%, capped at max
    """

    config = dict(config or {})

    # Extract regime label
    if isinstance(regime, RegimeSnapshot):
        regime_label = regime.label
    else:
        regime_label = str(regime).lower()

    # Default configuration
    base_alloc = float(config.get("base_allocation", 0.05))
    min_alloc = float(config.get("min_allocation", 0.02))

    multipliers = config.get("regime_multipliers", {})
    default_multipliers = {
        "calm": 0.5,
        "neutral": 1.0,
        "stressed": 2.0,
        "crash": 3.0,
    }
    multipliers = {**default_multipliers, **multipliers}

    max_allocations = config.get("max_allocation", {})
    default_max = {
        "calm": 0.03,
        "neutral": 0.05,
        "stressed": 0.12,
        "crash": 0.15,
    }
    max_allocations = {**default_max, **max_allocations}

    # Compute target allocation
    multiplier = multipliers.get(regime_label, 1.0)
    target_alloc = base_alloc * multiplier

    # Apply caps
    cap = max_allocations.get(regime_label, 0.15)
    target_alloc = min(target_alloc, cap)

    # Apply floor
    target_alloc = max(target_alloc, min_alloc)

    return float(target_alloc)


def apply_hedge_rebalance(
    weights: pd.Series,
    *,
    hedge_assets: list[str],
    target_hedge_allocation: float,
) -> pd.Series:
    """Adjust portfolio weights to meet target hedge allocation.

    This function scales non-hedge assets to free up capital for the hedge bucket,
    then distributes the hedge allocation equally among hedge assets.

    Parameters
    ----------
    weights : pd.Series
        Current portfolio weights (must sum to 1.0)
    hedge_assets : list[str]
        List of tickers considered tail hedge assets
    target_hedge_allocation : float
        Target total weight for hedge bucket (0.0 to 1.0)

    Returns
    -------
    pd.Series
        Adjusted weights with hedge allocation enforced

    Examples
    --------
    >>> weights = pd.Series({"SPY": 0.60, "AGG": 0.30, "GLD": 0.10})
    >>> hedge_assets = ["GLD"]
    >>> adjusted = apply_hedge_rebalance(weights, hedge_assets=hedge_assets,
    ...                                   target_hedge_allocation=0.15)
    >>> adjusted["GLD"]
    0.15  # Increased from 0.10 to 0.15

    >>> (adjusted["SPY"] + adjusted["AGG"] + adjusted["GLD"])
    1.0  # Still sums to 1.0
    """

    weights = weights.copy()

    # Separate hedge and non-hedge assets
    hedge_mask = weights.index.isin(hedge_assets)
    current_hedge_weight = weights[hedge_mask].sum()

    # If target is already met (within tolerance), return as-is
    if abs(current_hedge_weight - target_hedge_allocation) < 1e-6:
        return weights

    # Scale non-hedge assets
    non_hedge_weights = weights[~hedge_mask]
    non_hedge_total = non_hedge_weights.sum()

    if non_hedge_total > 0:
        # Target for non-hedge bucket
        non_hedge_target = 1.0 - target_hedge_allocation
        scaling_factor = non_hedge_target / non_hedge_total
        weights[~hedge_mask] = non_hedge_weights * scaling_factor

    # Distribute hedge allocation equally among hedge assets
    n_hedge = hedge_mask.sum()
    if n_hedge > 0:
        hedge_per_asset = target_hedge_allocation / n_hedge
        weights[hedge_mask] = hedge_per_asset
    else:
        # No hedge assets in portfolio; cannot apply hedge
        # Return weights unchanged and warn
        import warnings

        warnings.warn(
            "No hedge assets found in portfolio. Cannot apply hedge allocation.",
            stacklevel=2,
        )

    # Normalize to ensure sum = 1.0
    total = weights.sum()
    if abs(total - 1.0) > 1e-6:
        weights = weights / total

    return weights


def evaluate_hedge_performance(
    returns: pd.DataFrame,
    *,
    hedge_assets: list[str],
    regime_labels: pd.Series,
    stress_regimes: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate tail hedge effectiveness during stress periods.

    Computes correlation, drawdown mitigation, and cost drag of hedge assets
    in different regimes.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns (rows=dates, columns=assets)
    hedge_assets : list[str]
        List of hedge asset tickers
    regime_labels : pd.Series
        Regime classification for each date
    stress_regimes : list[str], optional
        Regimes considered "stress" (default: ["stressed", "crash"])

    Returns
    -------
    dict[str, Any]
        Diagnostic metrics:
        - correlation_stress: avg correlation between hedge and risky assets in stress
        - correlation_calm: avg correlation in calm periods
        - hedge_return_stress: avg hedge return in stress
        - hedge_return_calm: avg hedge return in calm
        - cost_drag_annual: annual cost drag in calm periods

    Examples
    --------
    >>> returns = pd.DataFrame({...})  # Asset returns
    >>> regimes = pd.Series(["calm", "calm", "crash", ...], index=returns.index)
    >>> metrics = evaluate_hedge_performance(
    ...     returns, hedge_assets=["TLT", "GLD"], regime_labels=regimes
    ... )
    >>> metrics["correlation_stress"]
    -0.42  # Negative correlation in stress = good hedge
    """

    stress_regimes = stress_regimes or ["stressed", "crash"]

    # Align returns with regime labels
    aligned_returns = returns.loc[regime_labels.index]
    hedge_cols = [col for col in hedge_assets if col in aligned_returns.columns]
    risky_cols = [col for col in aligned_returns.columns if col not in hedge_assets]

    if not hedge_cols or not risky_cols:
        return {
            "correlation_stress": None,
            "correlation_calm": None,
            "hedge_return_stress": None,
            "hedge_return_calm": None,
            "cost_drag_annual": None,
        }

    # Split by regime
    stress_mask = regime_labels.isin(stress_regimes)
    calm_mask = ~stress_mask

    # Hedge returns in stress vs calm
    hedge_returns = aligned_returns[hedge_cols].mean(axis=1)
    risky_returns = aligned_returns[risky_cols].mean(axis=1)

    hedge_return_stress = hedge_returns[stress_mask].mean() if stress_mask.any() else 0.0
    hedge_return_calm = hedge_returns[calm_mask].mean() if calm_mask.any() else 0.0

    # Correlation in stress vs calm
    if stress_mask.any():
        corr_stress = hedge_returns[stress_mask].corr(risky_returns[stress_mask])
    else:
        corr_stress = None

    if calm_mask.any():
        corr_calm = hedge_returns[calm_mask].corr(risky_returns[calm_mask])
    else:
        corr_calm = None

    # Annual cost drag (assume 5% hedge in calm, compute annual return drag)
    hedge_allocation_calm = 0.05
    if hedge_return_calm < 0:
        cost_drag = hedge_allocation_calm * abs(hedge_return_calm) * 252
    else:
        cost_drag = 0.0

    return {
        "correlation_stress": float(corr_stress) if corr_stress is not None else None,
        "correlation_calm": float(corr_calm) if corr_calm is not None else None,
        "hedge_return_stress": float(hedge_return_stress),
        "hedge_return_calm": float(hedge_return_calm),
        "cost_drag_annual": float(cost_drag),
        "n_hedge_assets": len(hedge_cols),
        "n_stress_days": int(stress_mask.sum()),
        "n_calm_days": int(calm_mask.sum()),
    }
