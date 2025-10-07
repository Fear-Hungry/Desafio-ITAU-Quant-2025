"""Fonte de dados: FRED.

Utilitários para baixar séries macroeconômicas (ex.: DTB3) e derivar r_f diário.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import logging
from pandas_datareader import data as pdr
import pandas as pd

logger = logging.getLogger(__name__)


def download_dtb3(
    start: Optional[str | datetime] = None,
    end: Optional[str | datetime] = None,
) -> pd.Series:
    """Baixa DTB3 (% a.a.) e retorna r_f diário aproximado (BR business days)."""
    logger.info("Baixando FRED DTB3 (start=%s, end=%s)", start, end)
    s = pdr.DataReader("DTB3", "fred", start, end)["DTB3"].astype(float)
    rf_daily = (s / 100.0) / 360.0
    rf_daily.name = "rf_daily"
    return rf_daily.asfreq("B").ffill()
