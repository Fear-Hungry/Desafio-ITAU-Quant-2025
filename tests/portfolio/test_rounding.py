from __future__ import annotations

import pandas as pd

from itau_quant.portfolio.rounding import rounding_pipeline


def test_rounding_pipeline_nearest_method():
    weights = pd.Series({"AAA": 0.4, "BBB": 0.35, "CCC": 0.25})
    prices = pd.Series({"AAA": 100.0, "BBB": 50.0, "CCC": 25.0})
    capital = 1_000_000.0
    config = {
        "lot_sizes": {"AAA": 10, "BBB": 5, "CCC": 1},
        "method": "nearest",
        "cost_model": {"linear_bps": 12},
    }

    result = rounding_pipeline(weights, prices, capital, config)

    shares = result.shares
    assert (shares["AAA"] % 10 == 0) and (shares["BBB"] % 5 == 0)

    capital_used = float((result.shares * prices).sum())
    assert abs(capital_used + result.residual_cash - capital) < 1.0
    assert result.rounded_weights.ge(0).all()
    assert result.rounding_cost >= 0.0
