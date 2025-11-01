"""Integração com FRED (Federal Reserve Economic Data).

`download_dtb3(start=None, end=None)`
    - Usa ``pandas_datareader`` para puxar a série DTB3 (% a.a.).
    - Converte para taxa diária (divide por 360 e normaliza para decimal).
    - Reamostra para frequência de negócios (``B``) com forward-fill, retornando
      ``pd.Series`` nomeada ``rf_daily``.
    - Lança ``ImportError`` amigável quando a dependência opcional não está
      instalada, sugerindo como habilitar.
"""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from pandas_datareader import data as pdr
except ImportError:  # pragma: no cover - depende de extra opcional
    pdr = None


def download_dtb3(
    start: str | datetime | None = None,
    end: str | datetime | None = None,
) -> pd.Series:
    """Baixa DTB3 (% a.a.) e retorna r_f diário aproximado (BR business days)."""
    if pdr is None:
        raise ImportError(
            "pandas_datareader não está instalado. "
            "Instale com 'poetry add pandas-datareader' para usar download_dtb3."
        )
    logger.info("Baixando FRED DTB3 (start=%s, end=%s)", start, end)
    s = pdr.DataReader("DTB3", "fred", start, end)["DTB3"].astype(float)
    rf_daily = (s / 100.0) / 360.0
    rf_daily.name = "rf_daily"
    return rf_daily.asfreq("B").ffill()
