"""Universo ARARA e utilitários relacionados.

Define a lista de tickers ARARA e helpers para expor o universo.
"""

from __future__ import annotations

from typing import List

ARARA_TICKERS: List[str] = [
    "SPY", "QQQ", "IWM",
    "EFA",
    "EEM",
    "XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLK", "XLI", "XLB", "XLRE", "XLU",
    "USMV", "MTUM", "QUAL", "VLUE", "SIZE",
    "VNQ", "VNQI",
    "SHY", "IEI", "IEF", "TLT",
    "TIP",
    "LQD", "HYG", "EMB", "EMLC",
    "GLD", "DBC",
    "UUP",
    "IBIT", "ETHA",
]


def get_arara_universe() -> List[str]:
    return list(ARARA_TICKERS)
