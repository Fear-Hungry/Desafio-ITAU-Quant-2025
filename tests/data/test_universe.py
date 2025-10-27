from __future__ import annotations

from itau_quant.data.universe import get_arara_metadata, get_arara_universe


def test_arara_universe_has_expected_count_and_members():
    tickers = get_arara_universe()
    assert isinstance(tickers, list)
    assert len(tickers) == 69
    expected = {
        "SPY",
        "QQQ",
        "IWM",
        "EFA",
        "VGK",
        "VPL",
        "EWJ",
        "EWG",
        "EEM",
        "EWZ",
        "INDA",
        "MCHI",
        "EWU",
        "EZA",
        "XLC",
        "XLY",
        "XLP",
        "XLE",
        "XLF",
        "XLV",
        "XLK",
        "XLI",
        "XLB",
        "XLRE",
        "XLU",
        "USMV",
        "MTUM",
        "QUAL",
        "VLUE",
        "SIZE",
        "VYM",
        "SCHD",
        "SPLV",
        "VUG",
        "VTV",
        "VNQ",
        "VNQI",
        "O",
        "PSA",
        "SHY",
        "IEI",
        "IEF",
        "TLT",
        "TIP",
        "VGSH",
        "VGIT",
        "AGG",
        "MUB",
        "LQD",
        "HYG",
        "VCIT",
        "VCSH",
        "EMB",
        "EMLC",
        "BNDX",
        "GLD",
        "SLV",
        "PPLT",
        "DBC",
        "USO",
        "UNG",
        "DBA",
        "CORN",
        "UUP",
        "IBIT",
        "ETHA",
        "FBTC",
        "GBTC",
        "ETHE",
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
