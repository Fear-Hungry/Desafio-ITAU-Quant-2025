from __future__ import annotations

import pandas as pd

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


def test_calculate_adjustment_factors_combined_events() -> None:
    index = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"])
    actions = _sample_actions()
    actions["ex_date"] = pd.to_datetime(actions["ex_date"])

    factors = ca.calculate_adjustment_factors(actions, index)

    price_factors = factors["price_AAA"].reindex(index).fillna(1.0)
    assert price_factors.loc[pd.Timestamp("2020-01-03")] == 0.5
    assert price_factors.loc[pd.Timestamp("2020-01-05")] == 0.4


def test_apply_price_adjustments_and_returns() -> None:
    index = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"])
    prices = pd.DataFrame({"AAA": [100, 50, 50, 40]}, index=index)
    actions = _sample_actions()
    actions["ex_date"] = pd.to_datetime(actions["ex_date"])

    factors = ca.calculate_adjustment_factors(actions, index)
    adjusted_prices = ca.apply_price_adjustments(prices, factors)

    assert adjusted_prices.loc["2020-01-02", "AAA"] == 25
    assert adjusted_prices.loc["2020-01-04", "AAA"] == 16
