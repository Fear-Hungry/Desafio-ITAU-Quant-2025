from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from itau_quant.data.processing import corporate_actions as ca


def _sample_actions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "event_type": "split",
                "ex_date": "2020-01-02",
                "effective_date": "2020-01-03",
                "ratio": 2.0,
            },
            {
                "ticker": "AAA",
                "event_type": "cash_dividend",
                "ex_date": "2020-01-03",
                "cash_amount": 0.5,
            },
            {
                "ticker": "AAA",
                "event_type": "spinoff",
                "ex_date": "2020-01-04",
                "ratio": 0.2,
            },
        ]
    )


# =============================================================================
# CorporateAction dataclass tests
# =============================================================================


def test_corporate_action_from_mapping_full_data() -> None:
    payload = {
        "ticker": "AAPL",
        "event_type": "split",
        "ex_date": "2020-08-31",
        "effective_date": "2020-08-31",
        "ratio": 4.0,
        "cash_amount": None,
    }
    action = ca.CorporateAction.from_mapping(payload)
    assert action.ticker == "AAPL"
    assert action.event_type == "split"
    assert action.ex_date == pd.Timestamp("2020-08-31")
    assert action.effective_date == pd.Timestamp("2020-08-31")
    assert action.ratio == 4.0
    assert action.cash_amount is None


def test_corporate_action_from_mapping_minimal_data() -> None:
    payload = {
        "ticker": "MSFT",
        "event_type": "dividend",
        "ex_date": "2020-11-18",
    }
    action = ca.CorporateAction.from_mapping(payload)
    assert action.ticker == "MSFT"
    assert action.event_type == "dividend"
    assert action.ex_date == pd.Timestamp("2020-11-18")
    assert action.effective_date is None
    assert action.ratio is None
    assert action.cash_amount is None


def test_corporate_action_normalizes_event_type() -> None:
    payload = {
        "ticker": "GOOG",
        "event_type": "SPLIT",
        "ex_date": "2022-07-15",
        "ratio": 20.0,
    }
    action = ca.CorporateAction.from_mapping(payload)
    assert action.event_type == "split"


# =============================================================================
# load_corporate_actions tests
# =============================================================================


def test_load_corporate_actions_empty_list() -> None:
    result = ca.load_corporate_actions(["AAPL", "MSFT"], actions=[])
    assert result.empty


def test_load_corporate_actions_none_actions() -> None:
    result = ca.load_corporate_actions(["AAPL"], actions=None)
    assert result.empty


def test_load_corporate_actions_filters_tickers() -> None:
    actions = [
        {"ticker": "AAPL", "event_type": "split", "ex_date": "2020-08-31", "ratio": 4.0},
        {"ticker": "MSFT", "event_type": "dividend", "ex_date": "2020-11-18", "cash_amount": 0.56},
        {"ticker": "GOOG", "event_type": "split", "ex_date": "2022-07-15", "ratio": 20.0},
    ]
    result = ca.load_corporate_actions(["AAPL", "MSFT"], actions=actions)
    assert len(result) == 2
    assert "GOOG" not in result["ticker"].values
    assert "AAPL" in result["ticker"].values
    assert "MSFT" in result["ticker"].values


def test_load_corporate_actions_sorts_by_ticker_and_date() -> None:
    actions = [
        {"ticker": "MSFT", "event_type": "dividend", "ex_date": "2020-11-18", "cash_amount": 0.56},
        {"ticker": "AAPL", "event_type": "split", "ex_date": "2020-08-31", "ratio": 4.0},
        {"ticker": "AAPL", "event_type": "dividend", "ex_date": "2020-05-08", "cash_amount": 0.82},
    ]
    result = ca.load_corporate_actions(["AAPL", "MSFT"], actions=actions)
    assert result.iloc[0]["ticker"] == "AAPL"
    assert result.iloc[0]["ex_date"] == pd.Timestamp("2020-05-08").normalize()
    assert result.iloc[1]["ticker"] == "AAPL"
    assert result.iloc[1]["ex_date"] == pd.Timestamp("2020-08-31").normalize()
    assert result.iloc[2]["ticker"] == "MSFT"


def test_load_corporate_actions_normalizes_dates() -> None:
    actions = [
        {"ticker": "AAPL", "event_type": "split", "ex_date": "2020-08-31 14:30:00", "effective_date": "2020-08-31 09:00:00", "ratio": 4.0},
    ]
    result = ca.load_corporate_actions(["AAPL"], actions=actions)
    assert result.iloc[0]["ex_date"] == pd.Timestamp("2020-08-31").normalize()
    assert result.iloc[0]["effective_date"] == pd.Timestamp("2020-08-31").normalize()


# =============================================================================
# calculate_adjustment_factors tests
# =============================================================================


def test_calculate_adjustment_factors_empty_actions() -> None:
    index = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    empty_actions = pd.DataFrame(columns=["ticker", "event_type", "ex_date", "ratio", "cash_amount"])
    factors = ca.calculate_adjustment_factors(empty_actions, index)
    assert factors.shape == (3, 2)
    assert (factors["price"] == 1.0).all()
    assert (factors["cash_dividend"] == 1.0).all()


def test_calculate_adjustment_factors_split_only() -> None:
    index = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    actions = pd.DataFrame([
        {
            "ticker": "AAPL",
            "event_type": "split",
            "ex_date": pd.Timestamp("2020-01-02"),
            "ratio": 2.0,
            "cash_amount": None,
        }
    ])
    factors = ca.calculate_adjustment_factors(actions, index)
    assert factors.loc[pd.Timestamp("2020-01-01"), "price_AAPL"] == 1.0
    assert factors.loc[pd.Timestamp("2020-01-02"), "price_AAPL"] == 0.5
    assert factors.loc[pd.Timestamp("2020-01-03"), "price_AAPL"] == 0.5


def test_calculate_adjustment_factors_dividend_only() -> None:
    index = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    actions = pd.DataFrame([
        {
            "ticker": "MSFT",
            "event_type": "cash_dividend",
            "ex_date": pd.Timestamp("2020-01-02"),
            "ratio": None,
            "cash_amount": 0.56,
        }
    ])
    factors = ca.calculate_adjustment_factors(actions, index)
    assert factors.loc[pd.Timestamp("2020-01-01"), "cash_MSFT"] == 1.0
    assert factors.loc[pd.Timestamp("2020-01-02"), "cash_MSFT"] == 1.56
    assert factors.loc[pd.Timestamp("2020-01-03"), "cash_MSFT"] == 1.56


def test_calculate_adjustment_factors_spinoff_only() -> None:
    index = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    actions = pd.DataFrame([
        {
            "ticker": "XYZ",
            "event_type": "spinoff",
            "ex_date": pd.Timestamp("2020-01-02"),
            "ratio": 0.15,
            "cash_amount": None,
        }
    ])
    factors = ca.calculate_adjustment_factors(actions, index)
    assert factors.loc[pd.Timestamp("2020-01-01"), "price_XYZ"] == 1.0
    assert factors.loc[pd.Timestamp("2020-01-02"), "price_XYZ"] == pytest.approx(0.85)
    assert factors.loc[pd.Timestamp("2020-01-03"), "price_XYZ"] == pytest.approx(0.85)


def test_calculate_adjustment_factors_combined_events() -> None:
    index = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"])
    actions = _sample_actions()
    actions["ex_date"] = pd.to_datetime(actions["ex_date"])

    factors = ca.calculate_adjustment_factors(actions, index)

    price_factors = factors["price_AAA"].reindex(index).fillna(1.0)
    assert price_factors.loc[pd.Timestamp("2020-01-03")] == 0.5
    assert price_factors.loc[pd.Timestamp("2020-01-05")] == 0.4


def test_calculate_adjustment_factors_multiple_tickers() -> None:
    index = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    actions = pd.DataFrame([
        {"ticker": "AAPL", "event_type": "split", "ex_date": pd.Timestamp("2020-01-02"), "ratio": 2.0, "cash_amount": None},
        {"ticker": "MSFT", "event_type": "cash_dividend", "ex_date": pd.Timestamp("2020-01-02"), "ratio": None, "cash_amount": 0.5},
    ])
    factors = ca.calculate_adjustment_factors(actions, index)
    assert "price_AAPL" in factors.columns
    assert "cash_AAPL" in factors.columns
    assert "price_MSFT" in factors.columns
    assert "cash_MSFT" in factors.columns
    assert factors.loc[pd.Timestamp("2020-01-02"), "price_AAPL"] == 0.5
    assert factors.loc[pd.Timestamp("2020-01-02"), "cash_MSFT"] == 1.5


def test_calculate_adjustment_factors_event_not_in_index() -> None:
    index = pd.to_datetime(["2020-01-01", "2020-01-03"])
    actions = pd.DataFrame([
        {"ticker": "AAPL", "event_type": "split", "ex_date": pd.Timestamp("2020-01-02"), "ratio": 2.0, "cash_amount": None},
    ])
    factors = ca.calculate_adjustment_factors(actions, index)
    assert factors.loc[pd.Timestamp("2020-01-01"), "price_AAPL"] == 1.0
    assert factors.loc[pd.Timestamp("2020-01-03"), "price_AAPL"] == 1.0


def test_calculate_adjustment_factors_multiple_events_same_date() -> None:
    index = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    actions = pd.DataFrame([
        {"ticker": "AAPL", "event_type": "split", "ex_date": pd.Timestamp("2020-01-02"), "ratio": 2.0, "cash_amount": None},
        {"ticker": "AAPL", "event_type": "cash_dividend", "ex_date": pd.Timestamp("2020-01-02"), "ratio": None, "cash_amount": 0.82},
    ])
    factors = ca.calculate_adjustment_factors(actions, index)
    assert factors.loc[pd.Timestamp("2020-01-02"), "price_AAPL"] == 0.5
    assert factors.loc[pd.Timestamp("2020-01-02"), "cash_AAPL"] == pytest.approx(1.82)


# =============================================================================
# apply_price_adjustments tests
# =============================================================================


def test_apply_price_adjustments_empty_prices() -> None:
    empty_prices = pd.DataFrame()
    factors = pd.DataFrame({"price": [1.0, 1.0]}, index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
    result = ca.apply_price_adjustments(empty_prices, factors)
    assert result.empty


def test_apply_price_adjustments_and_returns() -> None:
    index = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"])
    prices = pd.DataFrame({"AAA": [100, 50, 50, 40]}, index=index)
    actions = _sample_actions()
    actions["ex_date"] = pd.to_datetime(actions["ex_date"])

    factors = ca.calculate_adjustment_factors(actions, index)
    adjusted_prices = ca.apply_price_adjustments(prices, factors)

    assert adjusted_prices.loc["2020-01-02", "AAA"] == 25
    assert adjusted_prices.loc["2020-01-04", "AAA"] == 16


def test_apply_price_adjustments_no_ticker_specific_factors() -> None:
    index = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    prices = pd.DataFrame({"AAPL": [100.0, 110.0, 120.0]}, index=index)
    factors = pd.DataFrame({"price": [1.0, 1.0, 1.0], "cash_dividend": [1.0, 1.0, 1.0]}, index=index)
    adjusted = ca.apply_price_adjustments(prices, factors)
    pd.testing.assert_frame_equal(adjusted, prices)


def test_apply_price_adjustments_multiple_tickers() -> None:
    index = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    prices = pd.DataFrame({"AAPL": [100, 50, 55], "MSFT": [200, 210, 220]}, index=index)
    actions = pd.DataFrame([
        {"ticker": "AAPL", "event_type": "split", "ex_date": pd.Timestamp("2020-01-02"), "ratio": 2.0, "cash_amount": None},
    ])
    factors = ca.calculate_adjustment_factors(actions, index)
    adjusted = ca.apply_price_adjustments(prices, factors)
    assert adjusted.loc["2020-01-02", "AAPL"] == 25
    assert adjusted.loc["2020-01-03", "AAPL"] == 27.5
    assert adjusted.loc["2020-01-02", "MSFT"] == 210
    assert adjusted.loc["2020-01-03", "MSFT"] == 220


# =============================================================================
# apply_return_adjustments tests
# =============================================================================


def test_apply_return_adjustments_empty_returns() -> None:
    empty_returns = pd.DataFrame()
    actions = pd.DataFrame([
        {"ticker": "AAPL", "event_type": "cash_dividend", "ex_date": pd.Timestamp("2020-01-02"), "cash_amount": 0.82}
    ])
    result = ca.apply_return_adjustments(empty_returns, actions)
    assert result.empty


def test_apply_return_adjustments_empty_actions() -> None:
    index = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    returns = pd.DataFrame({"AAPL": [0.0, 0.05, -0.02]}, index=index)
    empty_actions = pd.DataFrame(columns=["ticker", "event_type", "ex_date", "cash_amount"])
    result = ca.apply_return_adjustments(returns, empty_actions)
    pd.testing.assert_frame_equal(result, returns)


def test_apply_return_adjustments_no_dividends() -> None:
    index = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    returns = pd.DataFrame({"AAPL": [0.0, 0.05, -0.02]}, index=index)
    actions = pd.DataFrame([
        {"ticker": "AAPL", "event_type": "split", "ex_date": pd.Timestamp("2020-01-02"), "ratio": 2.0, "cash_amount": None}
    ])
    result = ca.apply_return_adjustments(returns, actions)
    pd.testing.assert_frame_equal(result, returns)


def test_apply_return_adjustments_cash_dividend() -> None:
    index = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    returns = pd.DataFrame({"AAPL": [0.0, 0.05, -0.02]}, index=index)
    actions = pd.DataFrame([
        {"ticker": "AAPL", "event_type": "cash_dividend", "ex_date": pd.Timestamp("2020-01-02"), "cash_amount": 0.82, "ratio": None}
    ])
    result = ca.apply_return_adjustments(returns, actions)
    assert result.loc[pd.Timestamp("2020-01-02"), "AAPL"] == pytest.approx(0.05 + 0.82)
    assert result.loc[pd.Timestamp("2020-01-01"), "AAPL"] == 0.0
    assert result.loc[pd.Timestamp("2020-01-03"), "AAPL"] == -0.02


def test_apply_return_adjustments_dividend_not_in_index() -> None:
    index = pd.to_datetime(["2020-01-01", "2020-01-03"])
    returns = pd.DataFrame({"AAPL": [0.0, -0.02]}, index=index)
    actions = pd.DataFrame([
        {"ticker": "AAPL", "event_type": "dividend", "ex_date": pd.Timestamp("2020-01-02"), "cash_amount": 0.82, "ratio": None}
    ])
    result = ca.apply_return_adjustments(returns, actions)
    pd.testing.assert_frame_equal(result, returns)


def test_apply_return_adjustments_ticker_not_in_returns() -> None:
    index = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    returns = pd.DataFrame({"AAPL": [0.0, 0.05, -0.02]}, index=index)
    actions = pd.DataFrame([
        {"ticker": "MSFT", "event_type": "cash_dividend", "ex_date": pd.Timestamp("2020-01-02"), "cash_amount": 0.56, "ratio": None}
    ])
    result = ca.apply_return_adjustments(returns, actions)
    pd.testing.assert_frame_equal(result, returns)


def test_apply_return_adjustments_zero_cash_amount() -> None:
    index = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    returns = pd.DataFrame({"AAPL": [0.0, 0.05, -0.02]}, index=index)
    actions = pd.DataFrame([
        {"ticker": "AAPL", "event_type": "cash_dividend", "ex_date": pd.Timestamp("2020-01-02"), "cash_amount": 0.0, "ratio": None}
    ])
    result = ca.apply_return_adjustments(returns, actions)
    pd.testing.assert_frame_equal(result, returns)
