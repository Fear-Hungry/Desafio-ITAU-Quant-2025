"""Asset class-based cost model for cardinality optimization.

This module provides a simplified cost model that maps assets to predefined
cost tiers based on asset class characteristics. Used for K calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

__all__ = ["CostModel", "estimate_costs_by_class", "classify_asset"]


# Default cost mapping (round-trip in bps)
DEFAULT_COSTS_BPS = {
    "us_equities_core": 8,
    "us_sectors_factors": 10,
    "dev_ex_us": 12,
    "em_equities": 15,
    "reits_us": 12,
    "reits_ex_us": 28,
    "treasuries": 6,
    "tips": 8,
    "ig_credit": 12,
    "high_yield": 18,
    "em_bonds_usd": 20,
    "em_bonds_local": 32,
    "commodities_gold": 8,
    "commodities_basket": 28,
    "usd_fx": 12,
    "crypto_btc": 45,
    "crypto_eth": 55,
}


@dataclass
class CostModel:
    """Cost model for asset classes.

    Attributes:
        costs_bps: Mapping asset class → round-trip cost in bps
        asset_map: Mapping ticker → asset class
        default_bps: Fallback cost for unknown assets
    """

    costs_bps: dict[str, float] = field(
        default_factory=lambda: DEFAULT_COSTS_BPS.copy()
    )
    asset_map: dict[str, str] = field(default_factory=dict)
    default_bps: float = 15.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> CostModel:
        """Create from config dict.

        Args:
            config: Config with keys:
                - costs_bps or round_trip_bps
                - asset_class_map (optional)
                - default_cost_bps (optional)

        Returns:
            CostModel instance
        """
        costs_key = "costs_bps" if "costs_bps" in config else "round_trip_bps"
        return cls(
            costs_bps=config.get(costs_key, DEFAULT_COSTS_BPS.copy()),
            asset_map=config.get("asset_class_map", {}),
            default_bps=config.get("default_cost_bps", 15.0),
        )

    def get_cost(self, asset_class: str) -> float:
        """Get cost for asset class.

        Args:
            asset_class: Class identifier

        Returns:
            Cost in bps
        """
        return self.costs_bps.get(asset_class, self.default_bps)


def classify_asset(ticker: str) -> str:
    """Classify asset by ticker pattern (heuristic).

    Args:
        ticker: Asset ticker

    Returns:
        Asset class string

    Examples:
        >>> classify_asset("SPY")
        'us_equities_core'
        >>> classify_asset("FBTC")
        'crypto_btc'
        >>> classify_asset("GLD")
        'commodities_gold'
    """
    t = ticker.upper()

    # Crypto
    if any(x in t for x in ["BTC", "IBIT", "FBTC", "GBTC"]):
        return "crypto_btc"
    if any(x in t for x in ["ETH", "ETHA", "ETHE"]):
        return "crypto_eth"

    # Core US equities
    if t in ["SPY", "QQQ", "IWM", "VTI", "VOO"]:
        return "us_equities_core"

    # Sectors/factors
    if t.startswith("XL") or t in [
        "USMV",
        "MTUM",
        "QUAL",
        "VLUE",
        "SIZE",
        "VUG",
        "VTV",
    ]:
        return "us_sectors_factors"

    # Developed ex-US
    if t in ["EFA", "VGK", "VPL", "EWJ", "EWG", "EWU"]:
        return "dev_ex_us"

    # Emerging markets
    if t in ["EEM", "EWZ", "MCHI", "INDA", "EZA"]:
        return "em_equities"

    # REITs
    if t in ["VNQ", "O", "PSA"]:
        return "reits_us"
    if t in ["VNQI"]:
        return "reits_ex_us"

    # Treasuries
    if t in ["SHY", "IEI", "IEF", "TLT", "VGSH", "VGIT"]:
        return "treasuries"

    # TIPS
    if t in ["TIP"]:
        return "tips"

    # Credit
    if t in ["AGG", "LQD", "VCIT", "VCSH", "MUB", "BNDX"]:
        return "ig_credit"
    if t in ["HYG"]:
        return "high_yield"

    # EM bonds
    if t in ["EMB"]:
        return "em_bonds_usd"
    if t in ["EMLC"]:
        return "em_bonds_local"

    # Commodities
    if t in ["GLD", "SLV", "PPLT"]:
        return "commodities_gold"
    if t in ["DBC", "DBA", "CORN", "USO", "UNG"]:
        return "commodities_basket"

    # FX
    if t in ["UUP"]:
        return "usd_fx"

    # Default
    return "unknown"


def estimate_costs_by_class(
    assets: pd.Index,
    model: CostModel,
) -> pd.Series:
    """Estimate cost for each asset.

    Args:
        assets: Asset tickers
        model: Cost model

    Returns:
        Series with cost in bps

    Examples:
        >>> assets = pd.Index(['SPY', 'FBTC', 'GLD'])
        >>> model = CostModel()
        >>> costs = estimate_costs_by_class(assets, model)
        >>> costs['SPY']
        8.0
        >>> costs['FBTC']
        45.0
    """
    costs = pd.Series(index=assets, dtype=float)

    for asset in assets:
        # Try explicit mapping first
        if asset in model.asset_map:
            asset_class = model.asset_map[asset]
            costs[asset] = model.get_cost(asset_class)
        else:
            # Fallback to heuristic
            asset_class = classify_asset(asset)
            if asset_class == "unknown":
                costs[asset] = model.default_bps
            else:
                costs[asset] = model.get_cost(asset_class)

    return costs
