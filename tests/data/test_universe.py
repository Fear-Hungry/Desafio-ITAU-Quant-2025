from __future__ import annotations

from itau_quant.data.universe import get_arara_metadata, get_arara_universe


def test_arara_universe_has_expected_count_and_members():
    tickers = get_arara_universe()
    assert isinstance(tickers, list)
    assert len(tickers) == 37
    expected = {
        "SPY", "QQQ", "IWM", "EFA", "EEM",
        "XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLK", "XLI", "XLB", "XLRE", "XLU",
        "USMV", "MTUM", "QUAL", "VLUE", "SIZE",
        "VNQ", "VNQI",
        "SHY", "IEI", "IEF", "TLT",
        "TIP",
        "LQD", "HYG", "EMB", "EMLC",
        "GLD", "DBC",
        "UUP",
        "IBIT", "ETHA",
    }
    assert set(tickers) == expected


def test_arara_metadata_contains_expected_fields():
    metadata = get_arara_metadata()
    spy = metadata["SPY"]
    assert spy["asset_class"] == "Equity"
    assert spy["currency"] == "USD"
    assert spy["max_weight"] == 0.20
    for ticker, info in metadata.items():
        assert "asset_class" in info
        assert "max_weight" in info
