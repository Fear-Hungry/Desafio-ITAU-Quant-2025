"""Pydantic schemas for configuration validation.

This module defines typed configuration schemas using Pydantic v2 for:
- Asset universe (tickers and metadata)
- Portfolio optimization parameters
- Production system settings
- Asset group constraints
- Data loading configuration

All YAML configuration files in configs/ should validate against these schemas.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

__all__ = [
    "AssetGroupConstraints",
    "UniverseConfig",
    "DataConfig",
    "PortfolioConfig",
    "ProductionConfig",
    "EstimatorConfig",
]


class AssetGroupConstraints(BaseModel):
    """Constraints for a group of assets (e.g., crypto, commodities).

    Attributes
    ----------
    assets : List[str]
        Tickers in this group
    max : float
        Maximum total weight for the group (0-1)
    per_asset_max : Optional[float]
        Maximum weight per individual asset in group (0-1)
    """

    assets: list[str] = Field(min_length=1, description="List of tickers in this group")
    max: float = Field(ge=0, le=1, description="Maximum total group weight")
    per_asset_max: float | None = Field(
        default=None, ge=0, le=1, description="Maximum weight per asset in group"
    )

    @field_validator("assets")
    @classmethod
    def validate_tickers_uppercase(cls, v: list[str]) -> list[str]:
        """Ensure all tickers are uppercase."""
        return [ticker.upper() for ticker in v]


class UniverseConfig(BaseModel):
    """Asset universe configuration.

    Defines the set of assets to be used in portfolio optimization.

    Attributes
    ----------
    name : str
        Universe name (e.g., "ARARA", "CONSERVATIVE")
    tickers : List[str]
        List of ticker symbols
    description : Optional[str]
        Human-readable description
    """

    name: str = Field(description="Universe name identifier")
    tickers: list[str] = Field(
        min_length=1, description="List of ticker symbols to include"
    )
    description: str | None = Field(default=None, description="Universe description")

    @field_validator("tickers")
    @classmethod
    def validate_tickers_unique_uppercase(cls, v: list[str]) -> list[str]:
        """Ensure tickers are unique and uppercase."""
        tickers_upper = [ticker.upper() for ticker in v]
        if len(tickers_upper) != len(set(tickers_upper)):
            raise ValueError("Duplicate tickers found")
        return tickers_upper


class DataConfig(BaseModel):
    """Data loading and preprocessing configuration.

    Attributes
    ----------
    start_date : Optional[str]
        Start date (YYYY-MM-DD) or None for auto-calculation
    end_date : Optional[str]
        End date (YYYY-MM-DD) or None for today
    lookback_years : int
        Number of years to look back if start_date not specified
    min_history_days : int
        Minimum required observations per asset
    """

    start_date: str | None = Field(
        default=None, description="Start date (YYYY-MM-DD) or None"
    )
    end_date: str | None = Field(
        default=None, description="End date (YYYY-MM-DD) or None for today"
    )
    lookback_years: int = Field(
        default=3, gt=0, description="Years to look back if start_date not provided"
    )
    min_history_days: int = Field(
        default=252, gt=0, description="Minimum required observations per asset"
    )

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date_format(cls, v: str | None) -> str | None:
        """Validate date format if provided."""
        if v is None:
            return v
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError(f"Date must be in YYYY-MM-DD format, got: {v}")


class EstimatorConfig(BaseModel):
    """Configuration for μ and Σ estimators.

    Attributes
    ----------
    window_days : int
        Estimation window in trading days
    mu_method : Literal
        Mean return estimator (simple, huber, trimmed, shrunk_50)
    sigma_method : Literal
        Covariance estimator (sample, ledoit_wolf, nonlinear, tyler)
    huber_delta : float
        Delta parameter for Huber mean (only used if mu_method='huber')
    """

    window_days: int = Field(default=252, gt=0, description="Estimation window (days)")
    mu_method: Literal["simple", "huber", "trimmed", "shrunk_50"] = Field(
        default="simple", description="Mean return estimator"
    )
    sigma_method: Literal["sample", "ledoit_wolf", "nonlinear", "tyler"] = Field(
        default="ledoit_wolf", description="Covariance estimator"
    )
    huber_delta: float = Field(default=1.5, gt=0, description="Huber delta parameter")
    shrink_strength: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Shrinkage intensity towards prior when using shrunk_50",
    )


class PortfolioConfig(BaseModel):
    """Portfolio optimization configuration.

    Standard mean-variance optimization with transaction costs and turnover control.

    Attributes
    ----------
    risk_aversion : float
        Risk aversion parameter (λ), typically 2-6
    max_position : float
        Maximum weight per asset (0-1)
    min_position : float
        Minimum weight per asset (0-1), 0 for long-only
    turnover_penalty : float
        L1 turnover penalty (η), typically 0.01-0.5
    estimation_window : int
        Rolling window for estimation (trading days)
    shrinkage_method : Literal
        Covariance shrinkage method
    """

    risk_aversion: float = Field(
        default=3.0, ge=0, description="Risk aversion (lambda)"
    )
    max_position: float = Field(
        default=0.15, ge=0, le=1, description="Max weight per asset"
    )
    min_position: float = Field(
        default=0.0, ge=0, le=1, description="Min weight per asset (0 for long-only)"
    )
    turnover_penalty: float = Field(
        default=0.10, ge=0, description="L1 turnover penalty (eta)"
    )
    estimation_window: int = Field(
        default=252, gt=0, description="Estimation window (trading days)"
    )
    shrinkage_method: Literal["ledoit_wolf", "nonlinear", "tyler"] = Field(
        default="ledoit_wolf", description="Covariance shrinkage method"
    )
    estimators: EstimatorConfig | None = Field(
        default_factory=EstimatorConfig, description="Estimator configuration"
    )
    data: DataConfig | None = Field(
        default_factory=DataConfig, description="Data loading configuration"
    )

    @field_validator("min_position")
    @classmethod
    def validate_min_less_than_max(cls, v: float, info) -> float:
        """Ensure min_position <= max_position."""
        if "max_position" in info.data and v > info.data["max_position"]:
            raise ValueError("min_position must be <= max_position")
        return v


class ProductionConfig(BaseModel):
    """Production system configuration (ERC/Risk Parity).

    Configuration for production-ready Equal Risk Contribution portfolio with
    calibrated risk and turnover targets.

    Attributes
    ----------
    vol_target : float
        Target annualized volatility (e.g., 0.11 for 11%)
    vol_tolerance : float
        Volatility tolerance band (e.g., 0.01 for ±1%)
    turnover_target : float
        Target monthly turnover (e.g., 0.12 for 12%)
    turnover_tolerance : float
        Turnover tolerance band
    max_position : float
        Maximum weight per asset
    cardinality_k : int
        Target number of active positions
    transaction_cost_bps : float
        One-way transaction cost in basis points
    estimation_window : int
        Estimation window in trading days
    groups : Dict[str, AssetGroupConstraints]
        Asset group constraints (optional)
    """

    vol_target: float = Field(ge=0, le=1, description="Target annualized volatility")
    vol_tolerance: float = Field(ge=0, le=0.1, description="Volatility tolerance band")
    turnover_target: float = Field(ge=0, le=1, description="Target monthly turnover")
    turnover_tolerance: float = Field(
        ge=0, le=0.1, description="Turnover tolerance band"
    )
    max_position: float = Field(ge=0, le=1, description="Maximum weight per asset")
    cardinality_k: int = Field(gt=0, description="Target number of active positions")
    transaction_cost_bps: float = Field(
        ge=0, description="One-way transaction cost (bps)"
    )
    estimation_window: int = Field(
        default=252, gt=0, description="Estimation window (days)"
    )
    groups: dict[str, AssetGroupConstraints] = Field(
        default_factory=dict, description="Asset group constraints"
    )
    estimators: EstimatorConfig | None = Field(
        default_factory=EstimatorConfig, description="Estimator configuration"
    )

    @field_validator("groups")
    @classmethod
    def validate_group_names(
        cls, v: dict[str, AssetGroupConstraints]
    ) -> dict[str, AssetGroupConstraints]:
        """Ensure group names are lowercase."""
        return {k.lower(): v for k, v in v.items()}
