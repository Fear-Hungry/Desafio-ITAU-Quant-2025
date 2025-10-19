from __future__ import annotations

import pandas as pd

from itau_quant.backtesting import bookkeeping


def test_build_ledger_aligns_inputs_and_computes_nav() -> None:
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    net = pd.Series([0.01, -0.02], index=dates[:2])
    gross = pd.Series([0.015, -0.01, 0.02], index=dates)
    costs = pd.Series([0.0, 0.001], index=dates[:2])
    turnover = pd.Series([0.05], index=[dates[0]])

    ledger = bookkeeping.build_ledger(
        dates=dates,
        gross_returns=gross,
        net_returns=net,
        costs=costs,
        turnover=turnover,
    )

    frame = ledger.frame
    expected_nav = pd.Series([1.01, 0.9898, 0.9898], index=dates, name="nav")
    pd.testing.assert_series_equal(frame["nav"], expected_nav, check_exact=False, atol=1e-6)
    assert frame.loc[dates[2], "net_return"] == 0.0
    assert frame.loc[dates[1], "costs"] == 0.001
    assert frame.loc[dates[2], "turnover"] == 0.0

    as_dict = ledger.as_dict()
    assert "net_return" in as_dict
    assert len(as_dict["net_return"]) == len(dates)
