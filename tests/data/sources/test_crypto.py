from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
import pytest
from arara_quant.data.sources import crypto


def _fake_frame(symbols: Sequence[str], fields: Sequence[str]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    cols = pd.MultiIndex.from_product([symbols, [field.title() for field in fields]])
    data = pd.DataFrame(1.0, index=idx, columns=cols)
    for col in data.columns:
        data[col] = range(1, len(idx) + 1)
    return data


def test_download_crypto_prices_unknown_provider() -> None:
    with pytest.raises(ValueError, match="unsupported crypto provider"):
        crypto.download_crypto_prices(["BTCUSD"], provider="unknown")


def test_download_crypto_prices_caching(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    call_counter = {"count": 0}

    def fake_prices(self, symbols, *, start, end, fields):  # type: ignore[override]
        call_counter["count"] += 1
        return _fake_frame(symbols, fields)

    monkeypatch.setattr(crypto.CryptoDownloader, "prices", fake_prices)

    result = crypto.download_crypto_prices(
        ["BTCUSD"],
        provider="tiingo",
        cache=True,
        cache_dir=tmp_path,
    )
    assert call_counter["count"] == 1
    assert not result.empty

    # Second call should hit cache and avoid invoking loader again
    result_cached = crypto.download_crypto_prices(
        ["BTCUSD"],
        provider="tiingo",
        cache=True,
        cache_dir=tmp_path,
    )
    assert call_counter["count"] == 1
    pd.testing.assert_frame_equal(result, result_cached, check_freq=False)


def test_download_crypto_prices_normalizes_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_prices(self, symbols, *, start, end, fields):  # type: ignore[override]
        return _fake_frame(symbols, fields)

    monkeypatch.setattr(crypto.CryptoDownloader, "prices", fake_prices)

    frame = crypto.download_crypto_prices(
        ["BTC-USD"], provider="tiingo", fields=("Close", "Volume")
    )
    assert frame.columns.nlevels == 2
    level_zero = list(frame.columns.get_level_values(0).unique())
    assert level_zero == ["close", "volume"]
    level_one = list(frame.columns.get_level_values(1).unique())
    assert level_one == ["BTC-USD"]
