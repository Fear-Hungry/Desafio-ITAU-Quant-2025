"""Integração com Yahoo Finance (yfinance).

`download_prices(tickers, start=None, end=None, progress=False)`
    - Limpa a lista de tickers (stripping/case/ordem).
    - Chamadas a ``yf.download`` usam `auto_adjust=False` para permitir cálculo
      manual de ajustes.
    - Extrai ``Adj Close`` quando disponível (fallback para ``Close``).
    - Reordena colunas conforme solicitação original, aplica forward-fill e
      remove colunas totalmente vazias.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Iterable

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def download_prices(
    tickers: Iterable[str],
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    tickers_list = list(dict.fromkeys([t.strip().upper() for t in tickers]))
    if not tickers_list:
        raise ValueError("Lista de tickers vazia.")

    logger.info(
        "Baixando preços (yfinance): %s (start=%s, end=%s)",
        ",".join(tickers_list),
        start,
        end,
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
