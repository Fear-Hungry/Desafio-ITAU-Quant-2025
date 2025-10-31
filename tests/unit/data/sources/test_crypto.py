from __future__ import annotations

import json

import pandas as pd
import pytest

from itau_quant.data.sources import crypto


def test_sanitize_symbols_normalizes_input():
    assert crypto._sanitize_symbols([" btc-usd ", "ETH spot"]) == ["BTC-USD", "ETHSPOT"]


def test_download_crypto_prices_with_mocked_provider(monkeypatch):
    payload = [
        {
            "priceData": [
                {
                    "date": "2020-01-01T00:00:00Z",
                    "open": 10000,
                    "high": 10500,
                    "low": 9800,
                    "close": 10200,
                    "volume": 123,
                },
                {
                    "date": "2020-01-02T00:00:00Z",
                    "open": 10200,
                    "high": 10300,
                    "low": 9900,
                    "close": 10050,
                    "volume": 456,
                },
            ]
        }
    ]

    def fake_request(url, *, params, headers, config):
        return payload

    monkeypatch.setattr(crypto, "_request_json", fake_request)

    frame = crypto.download_crypto_prices(["BTCUSD"], start="2020-01-01", end="2020-01-02", fields=("Close",), provider="tiingo")
    close = frame["close"]
    assert list(close.columns) == ["BTCUSD"]
    assert close.iloc[0, 0] == 10200
