"""Tests for Pydantic configuration schemas."""

from __future__ import annotations

import pytest
from arara_quant.config.schemas import (
    AssetGroupConstraints,
    DataConfig,
    EstimatorConfig,
    PortfolioConfig,
    ProductionConfig,
    UniverseConfig,
)
from pydantic import ValidationError

# Tests for AssetGroupConstraints


def test_asset_group_constraints_valid():
    """Test creating valid AssetGroupConstraints."""
    group = AssetGroupConstraints(
        assets=["SPY", "QQQ", "IWM"],
        max=0.50,
        per_asset_max=0.20,
    )

    assert len(group.assets) == 3
    assert group.max == 0.50
    assert group.per_asset_max == 0.20


def test_asset_group_constraints_uppercase_tickers():
    """Test that tickers are automatically uppercased."""
    group = AssetGroupConstraints(
        assets=["spy", "qqq", "iwm"],
        max=0.30,
    )

    assert group.assets == ["SPY", "QQQ", "IWM"]


def test_asset_group_constraints_invalid_max():
    """Test that max > 1 raises ValidationError."""
    with pytest.raises(ValidationError):
        AssetGroupConstraints(
            assets=["SPY"],
            max=1.5,  # Invalid: > 1
        )


def test_asset_group_constraints_empty_assets():
    """Test that empty assets list raises ValidationError."""
    with pytest.raises(ValidationError):
        AssetGroupConstraints(
            assets=[],  # Invalid: min_length=1
            max=0.50,
        )


def test_asset_group_constraints_optional_per_asset_max():
    """Test that per_asset_max is optional."""
    group = AssetGroupConstraints(
        assets=["SPY"],
        max=0.30,
    )

    assert group.per_asset_max is None


# Tests for UniverseConfig


def test_universe_config_valid():
    """Test creating valid UniverseConfig."""
    config = UniverseConfig(
        name="TEST_UNIVERSE",
        tickers=["SPY", "QQQ", "AGG"],
        description="Test universe",
    )

    assert config.name == "TEST_UNIVERSE"
    assert len(config.tickers) == 3
    assert config.description == "Test universe"


def test_universe_config_uppercase_tickers():
    """Test that tickers are automatically uppercased."""
    config = UniverseConfig(
        name="test",
        tickers=["spy", "qqq"],
    )

    assert config.tickers == ["SPY", "QQQ"]


def test_universe_config_duplicate_tickers_raises():
    """Test that duplicate tickers raise ValidationError."""
    with pytest.raises(ValidationError, match="Duplicate tickers"):
        UniverseConfig(
            name="test",
            tickers=["SPY", "spy", "QQQ"],  # spy appears twice
        )


def test_universe_config_empty_tickers_raises():
    """Test that empty tickers list raises ValidationError."""
    with pytest.raises(ValidationError):
        UniverseConfig(
            name="test",
            tickers=[],  # Invalid: min_length=1
        )


def test_universe_config_optional_description():
    """Test that description is optional."""
    config = UniverseConfig(
        name="minimal",
        tickers=["SPY"],
    )

    assert config.description is None


# Tests for DataConfig


def test_data_config_defaults():
    """Test DataConfig with default values."""
    config = DataConfig()

    assert config.start_date is None
    assert config.end_date is None
    assert config.lookback_years == 3
    assert config.min_history_days == 252


def test_data_config_valid_dates():
    """Test DataConfig with valid date strings."""
    config = DataConfig(
        start_date="2020-01-01",
        end_date="2023-12-31",
        lookback_years=5,
        min_history_days=500,
    )

    assert config.start_date == "2020-01-01"
    assert config.end_date == "2023-12-31"
    assert config.lookback_years == 5
    assert config.min_history_days == 500


def test_data_config_invalid_date_format():
    """Test that invalid date format raises ValidationError."""
    with pytest.raises(ValidationError, match="Date must be in YYYY-MM-DD format"):
        DataConfig(start_date="01/01/2020")  # Wrong format


def test_data_config_invalid_lookback_years():
    """Test that lookback_years <= 0 raises ValidationError."""
    with pytest.raises(ValidationError):
        DataConfig(lookback_years=0)  # Must be > 0


def test_data_config_invalid_min_history_days():
    """Test that min_history_days <= 0 raises ValidationError."""
    with pytest.raises(ValidationError):
        DataConfig(min_history_days=-100)  # Must be > 0


# Tests for EstimatorConfig


def test_estimator_config_defaults():
    """Test EstimatorConfig with default values."""
    config = EstimatorConfig()

    assert config.window_days == 252
    assert config.mu_method == "simple"
    assert config.sigma_method == "ledoit_wolf"
    assert config.huber_delta == 1.5
    assert config.shrink_strength == 0.5


def test_estimator_config_custom_values():
    """Test EstimatorConfig with custom values."""
    config = EstimatorConfig(
        window_days=126,
        mu_method="huber",
        sigma_method="tyler",
        huber_delta=2.0,
        shrink_strength=0.8,
    )

    assert config.window_days == 126
    assert config.mu_method == "huber"
    assert config.sigma_method == "tyler"
    assert config.huber_delta == 2.0
    assert config.shrink_strength == 0.8


def test_estimator_config_allows_graphical_lasso():
    config = EstimatorConfig(sigma_method="graphical_lasso")
    assert config.sigma_method == "graphical_lasso"


@pytest.mark.parametrize("method", ["oas", "mincovdet"])
def test_estimator_config_allows_new_cov_methods(method):
    config = EstimatorConfig(sigma_method=method)
    assert config.sigma_method == method


def test_estimator_config_invalid_mu_method():
    """Test that invalid mu_method raises ValidationError."""
    with pytest.raises(ValidationError):
        EstimatorConfig(mu_method="invalid_method")


def test_estimator_config_invalid_sigma_method():
    """Test that invalid sigma_method raises ValidationError."""
    with pytest.raises(ValidationError):
        EstimatorConfig(sigma_method="nonexistent")


def test_estimator_config_invalid_shrink_strength():
    """Test that shrink_strength outside [0, 1] raises ValidationError."""
    with pytest.raises(ValidationError):
        EstimatorConfig(shrink_strength=1.5)  # > 1


# Tests for PortfolioConfig


def test_portfolio_config_defaults():
    """Test PortfolioConfig with default values."""
    config = PortfolioConfig()

    assert config.risk_aversion == 3.0
    assert config.max_position == 0.15
    assert config.min_position == 0.0
    assert config.turnover_penalty == 0.10
    assert config.estimation_window == 252
    assert config.shrinkage_method == "ledoit_wolf"
    assert config.estimators is not None
    assert config.data is not None


def test_portfolio_config_custom_values():
    """Test PortfolioConfig with custom values."""
    config = PortfolioConfig(
        risk_aversion=5.0,
        max_position=0.20,
        min_position=0.01,
        turnover_penalty=0.25,
        estimation_window=126,
        shrinkage_method="nonlinear",
    )

    assert config.risk_aversion == 5.0
    assert config.max_position == 0.20
    assert config.min_position == 0.01
    assert config.turnover_penalty == 0.25
    assert config.estimation_window == 126
    assert config.shrinkage_method == "nonlinear"


def test_portfolio_config_accepts_oas_shrinkage():
    config = PortfolioConfig(shrinkage_method="oas")
    assert config.shrinkage_method == "oas"


def test_portfolio_config_min_greater_than_max_raises():
    """Test that min_position > max_position raises ValidationError."""
    with pytest.raises(ValidationError, match="min_position must be <= max_position"):
        PortfolioConfig(
            min_position=0.20,
            max_position=0.10,  # Invalid: min > max
        )


def test_portfolio_config_invalid_risk_aversion():
    """Test that negative risk_aversion raises ValidationError."""
    with pytest.raises(ValidationError):
        PortfolioConfig(risk_aversion=-1.0)  # Must be >= 0


def test_portfolio_config_invalid_max_position():
    """Test that max_position > 1 raises ValidationError."""
    with pytest.raises(ValidationError):
        PortfolioConfig(max_position=1.5)  # Must be <= 1


def test_portfolio_config_nested_estimators():
    """Test PortfolioConfig with nested EstimatorConfig."""
    estimators = EstimatorConfig(
        window_days=180,
        mu_method="huber",
    )
    config = PortfolioConfig(estimators=estimators)

    assert config.estimators.window_days == 180
    assert config.estimators.mu_method == "huber"


# Tests for ProductionConfig


def test_production_config_valid():
    """Test creating valid ProductionConfig."""
    config = ProductionConfig(
        vol_target=0.11,
        vol_tolerance=0.01,
        turnover_target=0.12,
        turnover_tolerance=0.02,
        max_position=0.15,
        cardinality_k=30,
        transaction_cost_bps=10.0,
    )

    assert config.vol_target == 0.11
    assert config.vol_tolerance == 0.01
    assert config.turnover_target == 0.12
    assert config.turnover_tolerance == 0.02
    assert config.max_position == 0.15
    assert config.cardinality_k == 30
    assert config.transaction_cost_bps == 10.0
    assert config.estimation_window == 252  # default


def test_production_config_invalid_vol_target():
    """Test that vol_target > 1 raises ValidationError."""
    with pytest.raises(ValidationError):
        ProductionConfig(
            vol_target=1.5,  # > 1
            vol_tolerance=0.01,
            turnover_target=0.10,
            turnover_tolerance=0.02,
            max_position=0.15,
            cardinality_k=30,
            transaction_cost_bps=10.0,
        )


def test_production_config_invalid_cardinality():
    """Test that cardinality_k <= 0 raises ValidationError."""
    with pytest.raises(ValidationError):
        ProductionConfig(
            vol_target=0.10,
            vol_tolerance=0.01,
            turnover_target=0.10,
            turnover_tolerance=0.02,
            max_position=0.15,
            cardinality_k=0,  # Must be > 0
            transaction_cost_bps=10.0,
        )


def test_production_config_with_groups():
    """Test ProductionConfig with asset group constraints."""
    groups = {
        "crypto": AssetGroupConstraints(
            assets=["BTC", "ETH"],
            max=0.05,
        ),
        "commodities": AssetGroupConstraints(
            assets=["GLD", "SLV"],
            max=0.10,
        ),
    }

    config = ProductionConfig(
        vol_target=0.10,
        vol_tolerance=0.01,
        turnover_target=0.10,
        turnover_tolerance=0.02,
        max_position=0.15,
        cardinality_k=30,
        transaction_cost_bps=10.0,
        groups=groups,
    )

    assert len(config.groups) == 2
    assert "crypto" in config.groups
    assert config.groups["crypto"].max == 0.05


def test_production_config_group_names_lowercase():
    """Test that group names are automatically lowercased."""
    groups = {
        "CRYPTO": AssetGroupConstraints(
            assets=["BTC"],
            max=0.05,
        ),
    }

    config = ProductionConfig(
        vol_target=0.10,
        vol_tolerance=0.01,
        turnover_target=0.10,
        turnover_tolerance=0.02,
        max_position=0.15,
        cardinality_k=30,
        transaction_cost_bps=10.0,
        groups=groups,
    )

    assert "crypto" in config.groups
    assert "CRYPTO" not in config.groups


def test_production_config_optional_groups():
    """Test that groups is optional (empty dict by default)."""
    config = ProductionConfig(
        vol_target=0.10,
        vol_tolerance=0.01,
        turnover_target=0.10,
        turnover_tolerance=0.02,
        max_position=0.15,
        cardinality_k=30,
        transaction_cost_bps=10.0,
    )

    assert config.groups == {}
