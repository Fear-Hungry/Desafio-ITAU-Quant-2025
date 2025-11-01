"""Tests for asset class-based cost model."""

from __future__ import annotations

import pandas as pd
from itau_quant.costs.cost_model import (
    CostModel,
    classify_asset,
    estimate_costs_by_class,
)


class TestClassifyAsset:
    """Tests for heuristic asset classification."""

    def test_classifies_us_core_equities(self):
        """SPY, QQQ, IWM → us_equities_core."""
        assert classify_asset("SPY") == "us_equities_core"
        assert classify_asset("QQQ") == "us_equities_core"
        assert classify_asset("IWM") == "us_equities_core"

    def test_classifies_crypto_btc(self):
        """Bitcoin ETFs → crypto_btc."""
        assert classify_asset("FBTC") == "crypto_btc"
        assert classify_asset("IBIT") == "crypto_btc"
        assert classify_asset("GBTC") == "crypto_btc"

    def test_classifies_crypto_eth(self):
        """Ethereum ETFs → crypto_eth."""
        assert classify_asset("ETHA") == "crypto_eth"
        assert classify_asset("ETHE") == "crypto_eth"

    def test_classifies_sectors(self):
        """XL* → us_sectors_factors."""
        assert classify_asset("XLK") == "us_sectors_factors"
        assert classify_asset("XLF") == "us_sectors_factors"
        assert classify_asset("USMV") == "us_sectors_factors"

    def test_classifies_commodities_gold(self):
        """GLD, SLV, PPLT → commodities_gold."""
        assert classify_asset("GLD") == "commodities_gold"
        assert classify_asset("SLV") == "commodities_gold"
        assert classify_asset("PPLT") == "commodities_gold"

    def test_classifies_em_bonds(self):
        """EMB → em_bonds_usd, EMLC → em_bonds_local."""
        assert classify_asset("EMB") == "em_bonds_usd"
        assert classify_asset("EMLC") == "em_bonds_local"

    def test_unknown_returns_unknown(self):
        """Unknown ticker → 'unknown'."""
        assert classify_asset("FOOBAR") == "unknown"


class TestCostModel:
    """Tests for CostModel class."""

    def test_get_cost_returns_configured_value(self):
        """get_cost returns value from mapping."""
        model = CostModel()
        assert model.get_cost("us_equities_core") == 8.0
        assert model.get_cost("crypto_btc") == 45.0

    def test_get_cost_returns_default_for_unknown(self):
        """Unknown class → default cost."""
        model = CostModel(default_bps=20.0)
        assert model.get_cost("unknown_class") == 20.0

    def test_from_config_creates_model(self):
        """from_config parses dict correctly."""
        config = {
            "round_trip_bps": {
                "us_equities_core": 10,
                "crypto_btc": 50,
            },
            "default_cost_bps": 25.0,
        }
        model = CostModel.from_config(config)
        assert model.get_cost("us_equities_core") == 10.0
        assert model.default_bps == 25.0


class TestEstimateCostsByClass:
    """Tests for estimate_costs_by_class."""

    def test_estimates_costs_for_known_assets(self):
        """Known assets get correct costs."""
        assets = pd.Index(["SPY", "FBTC", "GLD"])
        model = CostModel()
        costs = estimate_costs_by_class(assets, model)

        assert costs["SPY"] == 8.0
        assert costs["FBTC"] == 45.0
        assert costs["GLD"] == 8.0

    def test_uses_explicit_mapping_when_provided(self):
        """Explicit asset_map overrides heuristic."""
        assets = pd.Index(["SPY", "CUSTOM"])
        model = CostModel(asset_map={"CUSTOM": "crypto_btc"})
        costs = estimate_costs_by_class(assets, model)

        assert costs["CUSTOM"] == 45.0  # Uses crypto_btc cost

    def test_fallback_to_default_for_unknown(self):
        """Unknown assets → default cost."""
        assets = pd.Index(["FOOBAR"])
        model = CostModel(default_bps=20.0)
        costs = estimate_costs_by_class(assets, model)

        assert costs["FOOBAR"] == 20.0
