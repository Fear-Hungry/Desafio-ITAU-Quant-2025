from __future__ import annotations

from datetime import datetime
from typing import Iterable, List, Optional

import logging
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr

logger = logging.getLogger(__name__)


# Universo ARARA (37 ETFs)
ARARA_TICKERS: List[str] = [
    "SPY", "QQQ", "IWM",  # EUA amplo
    "EFA",                  # Desenvolvidos ex-US
    "EEM",                  # Emergentes
    # Setores EUA
    "XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLK", "XLI", "XLB", "XLRE", "XLU",
    # Fatores EUA
    "USMV", "MTUM", "QUAL", "VLUE", "SIZE",
    # Imobiliário
    "VNQ", "VNQI",
    # Treasuries
    "SHY", "IEI", "IEF", "TLT",
    # TIPS
    "TIP",
    # Crédito
    "LQD", "HYG", "EMB", "EMLC",
    # Commodities
    "GLD", "DBC",
    # Câmbio USD
    "UUP",
    # Cripto (ETFs spot)
    "IBIT", "ETHA",
]


def get_arara_universe() -> List[str]:
    return list(ARARA_TICKERS)


def download_prices_yf(
    tickers: Iterable[str],
    start: Optional[str | datetime] = None,
    end: Optional[str | datetime] = None,
    progress: bool = False,
) -> pd.DataFrame:
    """Baixa Adj Close via yfinance e retorna DataFrame wide (datas x tickers)."""
    tickers_list = list(dict.fromkeys([t.strip().upper() for t in tickers]))
    if not tickers_list:
        raise ValueError("Lista de tickers vazia.")

    logger.info(
        "Baixando preços (yfinance): %s (start=%s, end=%s)",
        ",".join(tickers_list), start, end,
    )
    data = yf.download(
        tickers=tickers_list,
        start=start,
        end=end,
        auto_adjust=False,
        progress=progress,
        group_by="ticker",
        threads=True,
    )
    if isinstance(data.columns, pd.MultiIndex):
        try:
            adj = data.xs("Adj Close", axis=1, level=1)
        except Exception:
            logger.warning("Adj Close ausente; usando Close.")
            adj = data.xs("Close", axis=1, level=1)
    else:
        col = "Adj Close" if "Adj Close" in data.columns else "Close"
        adj = data[[col]].copy()
        adj.columns = tickers_list[:1]
    adj = adj.reindex(columns=[c for c in tickers_list if c in adj.columns])
    adj = adj.sort_index().ffill().dropna(how="all")
    return adj


def download_fred_dtb3(
    start: Optional[str | datetime] = None, end: Optional[str | datetime] = None
) -> pd.Series:
    """Baixa DTB3 da FRED (% a.a.) e converte para r_f diário aproximado."""
    logger.info("Baixando FRED DTB3 (start=%s, end=%s)", start, end)
    s = pdr.DataReader("DTB3", "fred", start, end)["DTB3"].astype(float)
    rf_daily = (s / 100.0) / 360.0
    rf_daily.name = "rf_daily"
    return rf_daily.asfreq("B").ffill()


