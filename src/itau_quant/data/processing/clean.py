"""Limpeza e validações.

Normalização de timezones, sanity checks e validações de integridade de dados.
"""

from __future__ import annotations

from typing import Iterable
import pandas as pd
from pandas import DatetimeIndex


def ensure_dtindex(idx: Iterable) -> DatetimeIndex:
    """Converte para DatetimeIndex ordenado, sem timezone.

    - Aceita iteráveis de datas e strings.
    - Ordena e remove duplicatas.
    - Remove timezone (tz-naive) para comparações e agrupamentos consistentes.
    """
    if not isinstance(idx, DatetimeIndex):
        out = DatetimeIndex(pd.to_datetime(list(idx)))
    else:
        out = DatetimeIndex(idx)
    # ordenar + únicos
    out = DatetimeIndex(sorted(out.unique()))
    # normalizar timezone
    if getattr(out, "tz", None) is not None:
        out = out.tz_localize(None)
    return out


def normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Garante DatetimeIndex ordenado e sem timezone para DataFrames."""
    out = df.copy()
    out.index = ensure_dtindex(out.index)
    return out


def validate_panel(prices: pd.DataFrame) -> None:
    """Checks básicos de sanidade do painel de preços."""
    assert prices.index.is_monotonic_increasing, "Index fora de ordem"
    assert prices.index.is_unique, "Index com duplicatas"
    assert prices.notna().any().any(), "Painel vazio ou todo NaN"
