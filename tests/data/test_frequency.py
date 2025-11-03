"""Tests for frequency conversion utilities."""

import numpy as np
import pandas as pd
import pytest

from itau_quant.data.processing.frequency import (
    annualize_return,
    annualize_volatility,
    monthly_to_annual,
    returns_to_monthly,
    returns_to_weekly,
)


class TestReturnsToMonthly:
    """Tests for returns_to_monthly function."""

    def test_converts_daily_to_monthly_series(self):
        """Test converting daily returns to monthly for Series."""
        dates = pd.date_range("2024-01-01", "2024-01-31", freq="D")
        daily_returns = pd.Series(0.01, index=dates)  # 1% daily

        monthly = returns_to_monthly(daily_returns)

        assert isinstance(monthly, pd.Series)
        assert len(monthly) == 1
        # Compound: (1.01)^31 - 1 ≈ 0.3613
        assert monthly.iloc[0] == pytest.approx(1.01**31 - 1, rel=1e-6)

    def test_converts_daily_to_monthly_dataframe(self):
        """Test converting daily returns to monthly for DataFrame."""
        dates = pd.date_range("2024-01-01", "2024-02-29", freq="D")
        daily_returns = pd.DataFrame({
            "A": 0.01,
            "B": -0.005,
        }, index=dates)

        monthly = returns_to_monthly(daily_returns)

        assert isinstance(monthly, pd.DataFrame)
        assert len(monthly) == 2  # Jan and Feb
        assert list(monthly.columns) == ["A", "B"]

    def test_handles_negative_returns(self):
        """Test that negative returns are handled correctly."""
        dates = pd.date_range("2024-01-01", "2024-01-10", freq="D")
        daily_returns = pd.Series([-0.01] * 10, index=dates)

        monthly = returns_to_monthly(daily_returns)

        expected = (1 - 0.01)**10 - 1
        assert monthly.iloc[0] == pytest.approx(expected, rel=1e-6)

    def test_empty_series_returns_empty(self):
        """Test empty input returns empty output."""
        empty = pd.Series([], dtype=float)
        empty.index = pd.DatetimeIndex([])

        monthly = returns_to_monthly(empty)

        assert len(monthly) == 0


class TestReturnsToWeekly:
    """Tests for returns_to_weekly function."""

    def test_converts_daily_to_weekly_series(self):
        """Test converting daily returns to weekly for Series."""
        dates = pd.date_range("2024-01-01", "2024-01-14", freq="D")
        daily_returns = pd.Series(0.01, index=dates)  # 1% daily

        weekly = returns_to_weekly(daily_returns)

        assert isinstance(weekly, pd.Series)
        assert len(weekly) >= 2  # At least 2 weeks

    def test_converts_daily_to_weekly_dataframe(self):
        """Test converting daily returns to weekly for DataFrame."""
        dates = pd.date_range("2024-01-01", "2024-01-21", freq="D")
        daily_returns = pd.DataFrame({
            "A": 0.01,
            "B": 0.02,
        }, index=dates)

        weekly = returns_to_weekly(daily_returns)

        assert isinstance(weekly, pd.DataFrame)
        assert list(weekly.columns) == ["A", "B"]

    def test_compounds_returns_correctly(self):
        """Test that weekly returns are compounded correctly."""
        dates = pd.date_range("2024-01-01", "2024-01-07", freq="D")
        daily_returns = pd.Series([0.01] * 7, index=dates)

        weekly = returns_to_weekly(daily_returns)

        expected = (1.01)**7 - 1
        assert weekly.iloc[0] == pytest.approx(expected, rel=1e-6)


class TestAnnualizeReturn:
    """Tests for annualize_return function."""

    def test_annualizes_monthly_return(self):
        """Test annualizing monthly returns."""
        monthly_return = 0.01  # 1% per month
        annual = annualize_return(monthly_return, 12)

        expected = (1.01)**12 - 1
        assert annual == pytest.approx(expected, rel=1e-6)

    def test_annualizes_daily_return(self):
        """Test annualizing daily returns."""
        daily_return = 0.001  # 0.1% per day
        annual = annualize_return(daily_return, 252)

        expected = (1.001)**252 - 1
        assert annual == pytest.approx(expected, rel=1e-6)

    def test_zero_return(self):
        """Test that zero return stays zero when annualized."""
        annual = annualize_return(0.0, 12)
        assert annual == 0.0

    def test_negative_return(self):
        """Test annualizing negative returns."""
        monthly_return = -0.05  # -5% per month
        annual = annualize_return(monthly_return, 12)

        expected = (1 - 0.05)**12 - 1
        assert annual == pytest.approx(expected, rel=1e-6)
        assert annual < 0  # Should remain negative


class TestAnnualizeVolatility:
    """Tests for annualize_volatility function."""

    def test_annualizes_monthly_volatility(self):
        """Test annualizing monthly volatility."""
        monthly_vol = 0.03  # 3% monthly
        annual = annualize_volatility(monthly_vol, 12)

        expected = 0.03 * np.sqrt(12)
        assert annual == pytest.approx(expected, rel=1e-6)

    def test_annualizes_daily_volatility(self):
        """Test annualizing daily volatility."""
        daily_vol = 0.01  # 1% daily
        annual = annualize_volatility(daily_vol, 252)

        expected = 0.01 * np.sqrt(252)
        assert annual == pytest.approx(expected, rel=1e-6)

    def test_zero_volatility(self):
        """Test that zero volatility stays zero."""
        annual = annualize_volatility(0.0, 12)
        assert annual == 0.0

    def test_square_root_scaling(self):
        """Test that volatility scales by square root of time."""
        vol = 0.02
        annual_12 = annualize_volatility(vol, 12)
        annual_252 = annualize_volatility(vol, 252)

        # Ratio should be sqrt(252/12)
        expected_ratio = np.sqrt(252 / 12)
        actual_ratio = annual_252 / annual_12
        assert actual_ratio == pytest.approx(expected_ratio, rel=1e-6)


class TestMonthlyToAnnual:
    """Tests for monthly_to_annual function."""

    def test_converts_return_with_compounding(self):
        """Test converting monthly return to annual using compounding."""
        monthly_return = 0.01
        annual = monthly_to_annual(monthly_return, is_return=True)

        expected = (1.01)**12 - 1
        assert annual == pytest.approx(expected, rel=1e-6)

    def test_converts_volatility_with_sqrt_scaling(self):
        """Test converting monthly volatility to annual using sqrt scaling."""
        monthly_vol = 0.03
        annual = monthly_to_annual(monthly_vol, is_return=False)

        expected = 0.03 * np.sqrt(12)
        assert annual == pytest.approx(expected, rel=1e-6)

    def test_custom_periods_per_year(self):
        """Test using custom periods per year."""
        stat = 0.01
        annual_return = monthly_to_annual(stat, is_return=True, periods_per_year=252)

        expected = (1.01)**252 - 1
        assert annual_return == pytest.approx(expected, rel=1e-6)

    def test_delegates_correctly(self):
        """Test that function delegates to correct annualization function."""
        stat = 0.02

        # Should use annualize_return
        annual_ret = monthly_to_annual(stat, is_return=True, periods_per_year=12)
        expected_ret = annualize_return(stat, 12)
        assert annual_ret == pytest.approx(expected_ret)

        # Should use annualize_volatility
        annual_vol = monthly_to_annual(stat, is_return=False, periods_per_year=12)
        expected_vol = annualize_volatility(stat, 12)
        assert annual_vol == pytest.approx(expected_vol)
