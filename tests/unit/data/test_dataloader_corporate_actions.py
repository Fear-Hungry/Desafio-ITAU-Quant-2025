from __future__ import annotations

import numpy as np
import pandas as pd

from itau_quant.data.loader import DataLoader, DataBundle


def _mock_prices() -> pd.DataFrame:
    idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    return pd.DataFrame({"AAA": [100.0, 50.0, 40.0]}, index=idx)


def test_dataloader_applies_corporate_actions(monkeypatch):
    prices = _mock_prices()
    returns = prices.pct_change().fillna(0.0)

    from itau_quant.data import loader as dl

    monkeypatch.setattr(dl, "yf_download", lambda tickers, start, end: prices.copy())
    monkeypatch.setattr(dl, "filter_liquid_assets", lambda df: (df, pd.DataFrame({"is_liquid": True}, index=df.columns)))
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
