from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from itau_quant.data.loader import DataLoader, DataBundle


def _mock_prices() -> pd.DataFrame:
    idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    return pd.DataFrame({"AAA": [100.0, 50.0, 40.0]}, index=idx)


def _mock_multi_ticker_prices() -> pd.DataFrame:
    idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"])
    return pd.DataFrame({
        "AAA": [100.0, 50.0, 50.0, 55.0],
        "BBB": [200.0, 210.0, 220.0, 230.0]
    }, index=idx)


def _liquidity_stats(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    return pd.DataFrame(
        {
            "is_liquid": [True] * len(df.columns),
            "coverage": [1.0] * len(df.columns),
            "non_na": [len(idx)] * len(df.columns),
            "max_gap": [0] * len(df.columns),
            "first_valid": [idx[0]] * len(df.columns),
            "last_valid": [idx[-1]] * len(df.columns),
        },
        index=df.columns,
    )


def test_dataloader_applies_corporate_actions(monkeypatch):
    prices = _mock_prices()
    returns = prices.pct_change().fillna(0.0)

    from itau_quant.data import loader as dl

    monkeypatch.setattr(dl, "yf_download", lambda tickers, start, end: prices.copy())
    monkeypatch.setattr(dl, "filter_liquid_assets", lambda df, **_: (df, _liquidity_stats(df)))
    monkeypatch.setattr(dl, "validate_panel", lambda df: None)
    monkeypatch.setattr(dl, "fred_download_dtb3", lambda start, end: pd.Series(0.0, index=prices.index))
    monkeypatch.setattr(dl, "compute_excess_returns", lambda ret, rf: ret)
    monkeypatch.setattr(dl, "rebalance_schedule", lambda index, mode: index)
    monkeypatch.setattr(dl, "request_hash", lambda *args, **kwargs: "hash")
    monkeypatch.setattr(dl, "save_parquet", lambda *args, **kwargs: None)
    monkeypatch.setattr(dl, "_calculate_returns", lambda df, method="log": returns.copy())

    actions = [
        {
            "ticker": "AAA",
            "event_type": "split",
            "ex_date": "2020-01-02",
            "ratio": 2.0,
        }
    ]

    loader = DataLoader(tickers=["AAA"], actions=actions)
    bundle = loader.load()
    assert isinstance(bundle, DataBundle)
    adjusted_prices = bundle.prices["AAA"]
    assert adjusted_prices.loc[pd.Timestamp("2020-01-02")] == 25.0


def test_dataloader_applies_dividend_event(monkeypatch):
    idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    prices = pd.DataFrame({"AAA": [100.0, 100.0, 100.0]}, index=idx)
    returns = prices.pct_change().fillna(0.0)

    from itau_quant.data import loader as dl

    monkeypatch.setattr(dl, "yf_download", lambda tickers, start, end: prices.copy())
    monkeypatch.setattr(dl, "filter_liquid_assets", lambda df, **_: (df, _liquidity_stats(df)))
    monkeypatch.setattr(dl, "validate_panel", lambda df: None)
    monkeypatch.setattr(dl, "fred_download_dtb3", lambda start, end: pd.Series(0.0, index=prices.index))
    monkeypatch.setattr(dl, "compute_excess_returns", lambda ret, rf: ret)
    monkeypatch.setattr(dl, "rebalance_schedule", lambda index, mode: index)
    monkeypatch.setattr(dl, "request_hash", lambda *args, **kwargs: "hash")
    monkeypatch.setattr(dl, "save_parquet", lambda *args, **kwargs: None)
    monkeypatch.setattr(dl, "_calculate_returns", lambda df, method="log": returns.copy())

    actions = [
        {
            "ticker": "AAA",
            "event_type": "cash_dividend",
            "ex_date": "2020-01-02",
            "cash_amount": 5.0,
        }
    ]

    loader = DataLoader(tickers=["AAA"], actions=actions)
    bundle = loader.load()
    assert isinstance(bundle, DataBundle)
    adjusted_prices = bundle.prices["AAA"]
    assert adjusted_prices.loc[pd.Timestamp("2020-01-01")] == 100.0
    assert adjusted_prices.loc[pd.Timestamp("2020-01-02")] == pytest.approx(100.0 * 1.0)
    assert adjusted_prices.loc[pd.Timestamp("2020-01-03")] == pytest.approx(100.0 * 1.0)


def test_dataloader_applies_spinoff_event(monkeypatch):
    idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    prices = pd.DataFrame({"AAA": [100.0, 80.0, 85.0]}, index=idx)
    returns = prices.pct_change().fillna(0.0)

    from itau_quant.data import loader as dl

    monkeypatch.setattr(dl, "yf_download", lambda tickers, start, end: prices.copy())
    monkeypatch.setattr(dl, "filter_liquid_assets", lambda df, **_: (df, _liquidity_stats(df)))
    monkeypatch.setattr(dl, "validate_panel", lambda df: None)
    monkeypatch.setattr(dl, "fred_download_dtb3", lambda start, end: pd.Series(0.0, index=prices.index))
    monkeypatch.setattr(dl, "compute_excess_returns", lambda ret, rf: ret)
    monkeypatch.setattr(dl, "rebalance_schedule", lambda index, mode: index)
    monkeypatch.setattr(dl, "request_hash", lambda *args, **kwargs: "hash")
    monkeypatch.setattr(dl, "save_parquet", lambda *args, **kwargs: None)
    monkeypatch.setattr(dl, "_calculate_returns", lambda df, method="log": returns.copy())

    actions = [
        {
            "ticker": "AAA",
            "event_type": "spinoff",
            "ex_date": "2020-01-02",
            "ratio": 0.2,
        }
    ]

    loader = DataLoader(tickers=["AAA"], actions=actions)
    bundle = loader.load()
    assert isinstance(bundle, DataBundle)
    adjusted_prices = bundle.prices["AAA"]
    assert adjusted_prices.loc[pd.Timestamp("2020-01-01")] == 100.0
    assert adjusted_prices.loc[pd.Timestamp("2020-01-02")] == pytest.approx(80.0 * 0.8)
    assert adjusted_prices.loc[pd.Timestamp("2020-01-03")] == pytest.approx(85.0 * 0.8)


def test_dataloader_applies_multiple_sequential_events(monkeypatch):
    idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"])
    prices = pd.DataFrame({"AAA": [100.0, 50.0, 50.0, 40.0]}, index=idx)
    returns = prices.pct_change().fillna(0.0)

    from itau_quant.data import loader as dl

    monkeypatch.setattr(dl, "yf_download", lambda tickers, start, end: prices.copy())
    monkeypatch.setattr(dl, "filter_liquid_assets", lambda df, **_: (df, _liquidity_stats(df)))
    monkeypatch.setattr(dl, "validate_panel", lambda df: None)
    monkeypatch.setattr(dl, "fred_download_dtb3", lambda start, end: pd.Series(0.0, index=prices.index))
    monkeypatch.setattr(dl, "compute_excess_returns", lambda ret, rf: ret)
    monkeypatch.setattr(dl, "rebalance_schedule", lambda index, mode: index)
    monkeypatch.setattr(dl, "request_hash", lambda *args, **kwargs: "hash")
    monkeypatch.setattr(dl, "save_parquet", lambda *args, **kwargs: None)
    monkeypatch.setattr(dl, "_calculate_returns", lambda df, method="log": returns.copy())

    actions = [
        {
            "ticker": "AAA",
            "event_type": "split",
            "ex_date": "2020-01-02",
            "ratio": 2.0,
        },
        {
            "ticker": "AAA",
            "event_type": "spinoff",
            "ex_date": "2020-01-04",
            "ratio": 0.2,
        }
    ]

    loader = DataLoader(tickers=["AAA"], actions=actions)
    bundle = loader.load()
    assert isinstance(bundle, DataBundle)
    adjusted_prices = bundle.prices["AAA"]
    assert adjusted_prices.loc[pd.Timestamp("2020-01-01")] == 100.0
    assert adjusted_prices.loc[pd.Timestamp("2020-01-02")] == 25.0
    assert adjusted_prices.loc[pd.Timestamp("2020-01-03")] == 25.0
    assert adjusted_prices.loc[pd.Timestamp("2020-01-04")] == pytest.approx(40.0 * 0.5 * 0.8)


def test_dataloader_applies_events_on_same_date(monkeypatch):
    idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    prices = pd.DataFrame({"AAA": [100.0, 50.0, 55.0]}, index=idx)
    returns = prices.pct_change().fillna(0.0)

    from itau_quant.data import loader as dl

    monkeypatch.setattr(dl, "yf_download", lambda tickers, start, end: prices.copy())
    monkeypatch.setattr(dl, "filter_liquid_assets", lambda df, **_: (df, _liquidity_stats(df)))
    monkeypatch.setattr(dl, "validate_panel", lambda df: None)
    monkeypatch.setattr(dl, "fred_download_dtb3", lambda start, end: pd.Series(0.0, index=prices.index))
    monkeypatch.setattr(dl, "compute_excess_returns", lambda ret, rf: ret)
    monkeypatch.setattr(dl, "rebalance_schedule", lambda index, mode: index)
    monkeypatch.setattr(dl, "request_hash", lambda *args, **kwargs: "hash")
    monkeypatch.setattr(dl, "save_parquet", lambda *args, **kwargs: None)
    monkeypatch.setattr(dl, "_calculate_returns", lambda df, method="log": returns.copy())

    actions = [
        {
            "ticker": "AAA",
            "event_type": "split",
            "ex_date": "2020-01-02",
            "ratio": 2.0,
        },
        {
            "ticker": "AAA",
            "event_type": "cash_dividend",
            "ex_date": "2020-01-02",
            "cash_amount": 1.0,
        }
    ]

    loader = DataLoader(tickers=["AAA"], actions=actions)
    bundle = loader.load()
    assert isinstance(bundle, DataBundle)
    adjusted_prices = bundle.prices["AAA"]
    assert adjusted_prices.loc[pd.Timestamp("2020-01-01")] == 100.0
    assert adjusted_prices.loc[pd.Timestamp("2020-01-02")] == 25.0
    assert adjusted_prices.loc[pd.Timestamp("2020-01-03")] == 27.5


def test_dataloader_applies_events_to_different_tickers(monkeypatch):
    prices = _mock_multi_ticker_prices()
    returns = prices.pct_change().fillna(0.0)

    from itau_quant.data import loader as dl

    monkeypatch.setattr(dl, "yf_download", lambda tickers, start, end: prices.copy())
    monkeypatch_setattr = lambda df, **_: (df, _liquidity_stats(df))
    monkeypatch.setattr(dl, "filter_liquid_assets", monkeypatch_setattr)
    monkeypatch.setattr(dl, "validate_panel", lambda df: None)
    monkeypatch.setattr(dl, "fred_download_dtb3", lambda start, end: pd.Series(0.0, index=prices.index))
    monkeypatch.setattr(dl, "compute_excess_returns", lambda ret, rf: ret)
    monkeypatch.setattr(dl, "rebalance_schedule", lambda index, mode: index)
    monkeypatch.setattr(dl, "request_hash", lambda *args, **kwargs: "hash")
    monkeypatch.setattr(dl, "save_parquet", lambda *args, **kwargs: None)
    monkeypatch.setattr(dl, "_calculate_returns", lambda df, method="log": returns.copy())

    actions = [
        {
            "ticker": "AAA",
            "event_type": "split",
            "ex_date": "2020-01-02",
            "ratio": 2.0,
        },
        {
            "ticker": "BBB",
            "event_type": "cash_dividend",
            "ex_date": "2020-01-03",
            "cash_amount": 5.0,
        }
    ]

    loader = DataLoader(tickers=["AAA", "BBB"], actions=actions)
    bundle = loader.load()
    assert isinstance(bundle, DataBundle)

    adjusted_aaa = bundle.prices["AAA"]
    assert adjusted_aaa.loc[pd.Timestamp("2020-01-01")] == 100.0
    assert adjusted_aaa.loc[pd.Timestamp("2020-01-02")] == 25.0

    adjusted_bbb = bundle.prices["BBB"]
    assert adjusted_bbb.loc[pd.Timestamp("2020-01-02")] == 210.0
    assert adjusted_bbb.loc[pd.Timestamp("2020-01-03")] == pytest.approx(220.0 * 1.0)


def test_dataloader_with_no_actions_provided(monkeypatch):
    idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    prices = pd.DataFrame({"AAA": [100.0, 110.0, 120.0]}, index=idx)
    returns = prices.pct_change().fillna(0.0)

    from itau_quant.data import loader as dl

    monkeypatch.setattr(dl, "yf_download", lambda tickers, start, end: prices.copy())
    monkeypatch.setattr(dl, "filter_liquid_assets", lambda df, **_: (df, _liquidity_stats(df)))
    monkeypatch.setattr(dl, "validate_panel", lambda df: None)
    monkeypatch.setattr(dl, "fred_download_dtb3", lambda start, end: pd.Series(0.0, index=prices.index))
    monkeypatch.setattr(dl, "compute_excess_returns", lambda ret, rf: ret)
    monkeypatch.setattr(dl, "rebalance_schedule", lambda index, mode: index)
    monkeypatch.setattr(dl, "request_hash", lambda *args, **kwargs: "hash")
    monkeypatch.setattr(dl, "save_parquet", lambda *args, **kwargs: None)
    monkeypatch.setattr(dl, "_calculate_returns", lambda df, method="log": returns.copy())

    loader = DataLoader(tickers=["AAA"], actions=None)
    bundle = loader.load()
    assert isinstance(bundle, DataBundle)
    pd.testing.assert_frame_equal(bundle.prices, prices)


def test_dataloader_with_empty_actions_list(monkeypatch):
    idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    prices = pd.DataFrame({"AAA": [100.0, 110.0, 120.0]}, index=idx)
    returns = prices.pct_change().fillna(0.0)

    from itau_quant.data import loader as dl

    monkeypatch.setattr(dl, "yf_download", lambda tickers, start, end: prices.copy())
    monkeypatch.setattr(dl, "filter_liquid_assets", lambda df, **_: (df, _liquidity_stats(df)))
    monkeypatch.setattr(dl, "validate_panel", lambda df: None)
    monkeypatch.setattr(dl, "fred_download_dtb3", lambda start, end: pd.Series(0.0, index=prices.index))
    monkeypatch.setattr(dl, "compute_excess_returns", lambda ret, rf: ret)
    monkeypatch.setattr(dl, "rebalance_schedule", lambda index, mode: index)
    monkeypatch.setattr(dl, "request_hash", lambda *args, **kwargs: "hash")
    monkeypatch.setattr(dl, "save_parquet", lambda *args, **kwargs: None)
    monkeypatch.setattr(dl, "_calculate_returns", lambda df, method="log": returns.copy())

    loader = DataLoader(tickers=["AAA"], actions=[])
    bundle = loader.load()
    assert isinstance(bundle, DataBundle)
    pd.testing.assert_frame_equal(bundle.prices, prices)
