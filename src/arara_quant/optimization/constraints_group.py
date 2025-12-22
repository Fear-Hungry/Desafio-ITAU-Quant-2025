"""Group-level constraints for portfolio optimization.

This module provides utilities to enforce constraints at the asset class
or sector level (e.g., equity ≤ 40%, crypto ≤ 10%).
"""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    import cvxpy as cp

__all__ = ["GroupConstraint", "build_group_constraints", "validate_group_caps"]


@dataclass
class GroupConstraint:
    """Constraint on a group of assets.

    Attributes:
        name: Group name (e.g., "equity", "crypto")
        max_weight: Maximum total weight for the group
        min_weight: Minimum total weight (optional)
        per_asset_max: Optional max weight per asset in the group
        assets: List of asset tickers in the group
        assets_regex: Regex pattern to match assets (alternative to explicit list)
    """

    name: str
    max_weight: float
    min_weight: float = 0.0
    assets: list[str] | None = None
    assets_regex: str | None = None
    per_asset_max: float | None = None

    def matches(self, asset: str) -> bool:
        """Check if asset belongs to this group.

        Args:
            asset: Asset ticker

        Returns:
            True if asset is in the group
        """
        if self.assets and asset in self.assets:
            return True
        if self.assets_regex and re.match(self.assets_regex, asset):
            return True
        return False

    def get_assets(self, universe: pd.Index) -> list[str]:
        """Get all assets from universe that belong to this group.

        Args:
            universe: Full universe of asset tickers

        Returns:
            List of matching assets
        """
        if self.assets:
            return [a for a in self.assets if a in universe]
        if self.assets_regex:
            pattern = re.compile(self.assets_regex)
            return [a for a in universe if pattern.match(a)]
        return []


def build_group_constraints(
    w: cp.Variable,
    groups: list[GroupConstraint],
    asset_index: pd.Index,
) -> list[cp.Constraint]:
    """Build CVXPY constraints for asset groups.

    Args:
        w: CVXPY weight variable
        groups: List of group constraints
        asset_index: Asset universe

    Returns:
        List of CVXPY constraints

    Examples:
        >>> import cvxpy as cp
        >>> w = cp.Variable(3)
        >>> asset_index = pd.Index(['SPY', 'QQQ', 'GLD'])
        >>> group = GroupConstraint(
        ...     name="equity",
        ...     max_weight=0.8,
        ...     assets=['SPY', 'QQQ']
        ... )
        >>> constraints = build_group_constraints(w, [group], asset_index)
        >>> len(constraints)
        1
    """
    cp = importlib.import_module("cvxpy")

    constraints = []

    for group in groups:
        group_assets = group.get_assets(asset_index)
        if not group_assets:
            continue

        # Get indices
        indices = [asset_index.get_loc(a) for a in group_assets]

        # Sum of group weights
        group_sum = cp.sum(w[indices])

        # Max constraint
        if group.max_weight < 1.0:
            constraints.append(group_sum <= group.max_weight)

        # Min constraint
        if group.min_weight > 0.0:
            constraints.append(group_sum >= group.min_weight)

        if group.per_asset_max is not None:
            cap = float(group.per_asset_max)
            for idx in indices:
                constraints.append(w[idx] <= cap)

    return constraints


def validate_group_caps(
    weights: pd.Series,
    groups: list[GroupConstraint],
    tolerance: float = 1e-4,
) -> tuple[bool, dict[str, Any]]:
    """Validate that weights respect group constraints.

    Args:
        weights: Portfolio weights
        groups: List of group constraints
        tolerance: Numerical tolerance

    Returns:
        (is_valid, violations_dict) tuple

    Examples:
        >>> weights = pd.Series({'SPY': 0.5, 'QQQ': 0.3, 'GLD': 0.2})
        >>> group = GroupConstraint(name="equity", max_weight=0.7, assets=['SPY', 'QQQ'])
        >>> is_valid, info = validate_group_caps(weights, [group])
        >>> is_valid
        False
        >>> 'equity' in info['violations']
        True
    """
    violations = {}
    is_valid = True

    for group in groups:
        group_assets = group.get_assets(weights.index)
        if not group_assets:
            continue

        group_weight = weights[group_assets].sum()

        # Check max
        if group_weight > group.max_weight + tolerance:
            is_valid = False
            violations[group.name] = {
                "type": "max_exceeded",
                "limit": group.max_weight,
                "actual": float(group_weight),
                "violation": float(group_weight - group.max_weight),
            }

        # Check min
        if group_weight < group.min_weight - tolerance:
            is_valid = False
            violations[group.name] = {
                "type": "min_violated",
                "limit": group.min_weight,
                "actual": float(group_weight),
                "violation": float(group.min_weight - group_weight),
            }

    info = {
        "is_valid": is_valid,
        "violations": violations,
        "group_weights": {
            group.name: float(weights[group.get_assets(weights.index)].sum())
            for group in groups
        },
    }

    return is_valid, info


def parse_group_config(config: dict[str, Any]) -> list[GroupConstraint]:
    """Parse group constraints from config dict.

    Args:
        config: Config with group definitions

    Returns:
        List of GroupConstraint objects

    Examples:
        >>> config = {
        ...     "equity": {
        ...         "max_weight": 0.4,
        ...         "assets_regex": "(SPY|QQQ|IWM)"
        ...     },
        ...     "crypto": {
        ...         "max_weight": 0.1,
        ...         "assets": ["FBTC", "IBIT"]
        ...     }
        ... }
        >>> groups = parse_group_config(config)
        >>> len(groups)
        2
        >>> groups[0].name
        'equity'
    """
    groups = []

    for name, spec in config.items():
        group = GroupConstraint(
            name=name,
            max_weight=spec.get("max_weight", 1.0),
            min_weight=spec.get("min_weight", 0.0),
            assets=spec.get("assets"),
            assets_regex=spec.get("assets_regex"),
            per_asset_max=spec.get("per_asset_max"),
        )
        groups.append(group)

    return groups
