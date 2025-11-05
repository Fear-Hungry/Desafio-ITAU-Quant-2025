"""Tests for group constraints."""

from __future__ import annotations

import pandas as pd
import pytest
from arara_quant.optimization.constraints_group import (
    GroupConstraint,
    parse_group_config,
    validate_group_caps,
)


class TestGroupConstraint:
    """Tests for GroupConstraint class."""

    def test_matches_explicit_assets(self):
        """Explicit asset list matching."""
        group = GroupConstraint(name="equity", max_weight=0.4, assets=["SPY", "QQQ"])
        assert group.matches("SPY") is True
        assert group.matches("GLD") is False

    def test_matches_regex(self):
        """Regex pattern matching."""
        group = GroupConstraint(
            name="sectors", max_weight=0.5, assets_regex=r"XL[A-Z]+"
        )
        assert group.matches("XLK") is True
        assert group.matches("XLF") is True
        assert group.matches("SPY") is False

    def test_get_assets_filters_universe(self):
        """get_assets returns only assets in universe."""
        universe = pd.Index(["SPY", "QQQ", "GLD", "TLT"])
        group = GroupConstraint(
            name="equity", max_weight=0.4, assets=["SPY", "QQQ", "IWM"]
        )
        assets = group.get_assets(universe)

        assert set(assets) == {"SPY", "QQQ"}  # IWM not in universe

    def test_get_assets_with_regex(self):
        """get_assets with regex matches correctly."""
        universe = pd.Index(["XLK", "XLF", "SPY", "QQQ"])
        group = GroupConstraint(
            name="sectors", max_weight=0.5, assets_regex=r"XL[A-Z]+"
        )
        assets = group.get_assets(universe)

        assert set(assets) == {"XLK", "XLF"}


class TestValidateGroupCaps:
    """Tests for group cap validation."""

    def test_accepts_valid_weights(self):
        """Valid weights pass validation."""
        weights = pd.Series({"SPY": 0.3, "QQQ": 0.2, "GLD": 0.5})
        group = GroupConstraint(name="equity", max_weight=0.6, assets=["SPY", "QQQ"])
        is_valid, info = validate_group_caps(weights, [group])

        assert is_valid is True
        assert len(info["violations"]) == 0

    def test_detects_max_violation(self):
        """Detects when group exceeds max."""
        weights = pd.Series({"SPY": 0.5, "QQQ": 0.3, "GLD": 0.2})
        group = GroupConstraint(name="equity", max_weight=0.7, assets=["SPY", "QQQ"])
        is_valid, info = validate_group_caps(weights, [group])

        assert is_valid is False
        assert "equity" in info["violations"]
        assert info["violations"]["equity"]["type"] == "max_exceeded"
        assert info["violations"]["equity"]["actual"] == pytest.approx(0.8)

    def test_detects_min_violation(self):
        """Detects when group below min."""
        weights = pd.Series({"SPY": 0.1, "QQQ": 0.1, "GLD": 0.8})
        group = GroupConstraint(
            name="equity", min_weight=0.3, max_weight=1.0, assets=["SPY", "QQQ"]
        )
        is_valid, info = validate_group_caps(weights, [group])

        assert is_valid is False
        assert "equity" in info["violations"]
        assert info["violations"]["equity"]["type"] == "min_violated"

    def test_reports_group_weights(self):
        """Info dict includes all group weights."""
        weights = pd.Series({"SPY": 0.4, "QQQ": 0.2, "GLD": 0.4})
        groups = [
            GroupConstraint(name="equity", max_weight=0.8, assets=["SPY", "QQQ"]),
            GroupConstraint(name="commodities", max_weight=0.5, assets=["GLD"]),
        ]
        is_valid, info = validate_group_caps(weights, groups)

        assert "group_weights" in info
        assert info["group_weights"]["equity"] == pytest.approx(0.6)
        assert info["group_weights"]["commodities"] == pytest.approx(0.4)


class TestParseGroupConfig:
    """Tests for config parsing."""

    def test_parses_group_definitions(self):
        """Parses group config correctly."""
        config = {
            "equity": {"max_weight": 0.4, "assets_regex": r"(SPY|QQQ|IWM)"},
            "crypto": {"max_weight": 0.1, "assets": ["FBTC", "IBIT"]},
        }
        groups = parse_group_config(config)

        assert len(groups) == 2
        assert groups[0].name == "equity"
        assert groups[0].max_weight == 0.4
        assert groups[1].name == "crypto"
        assert groups[1].assets == ["FBTC", "IBIT"]

    def test_defaults_min_weight_to_zero(self):
        """min_weight defaults to 0."""
        config = {"equity": {"max_weight": 0.4}}
        groups = parse_group_config(config)

        assert groups[0].min_weight == 0.0
