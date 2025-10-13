"""
Calendário e regrinhas BMS.

Helpers para alinhamento de datas de negócios usando o PRÓPRIO índice de preços,
incluindo Business Month Start (BMS) e variações de agenda de rebalance.

Regra de ouro: derive tudo do índice de negociação (prices.index). Isso já embute
feriados e pregões perdidos do provedor e evita inconsistências com 'B' puro.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Union, Iterable

import numpy as np
import pandas as pd
from .clean import ensure_dtindex

Datelike = Union[str, pd.Timestamp, datetime]


def business_month_starts(idx: Iterable) -> pd.DatetimeIndex:
    """Retorna o 1º dia DE NEGOCIAÇÃO observado de cada mês (BMS)."""
    idx = ensure_dtindex(idx)
    if len(idx) == 0:
        return idx
    s = pd.Series(idx, index=idx)
    firsts = s.groupby(s.index.to_period("M")).min().values
    return pd.DatetimeIndex(firsts)


def business_month_ends(idx: Iterable) -> pd.DatetimeIndex:
    """Retorna o último dia de negociação observado de cada mês (BME)."""
    idx = ensure_dtindex(idx)
    if len(idx) == 0:
        return idx
    s = pd.Series(idx, index=idx)
    lasts = s.groupby(s.index.to_period("M")).max().values
    return pd.DatetimeIndex(lasts)


def weekly_last_trading_day(idx: Iterable, anchor: str = "W-FRI") -> pd.DatetimeIndex:
    """Retorna o último dia de negociação de cada semana ancorada (ex.: 'W-FRI').

    Usa o próprio índice para pegar o 'último disponível' da semana, não a sexta genérica.
    """
    idx = ensure_dtindex(idx)
    if len(idx) == 0:
        return idx
    s = pd.Series(idx, index=idx)
    lasts = s.groupby(s.index.to_period(anchor)).max().values
    return pd.DatetimeIndex(lasts)


def next_trading_day(index: Iterable, when: Datelike) -> pd.Timestamp:
    """Próximo dia de negociação >= when, segundo o índice fornecido."""
    idx = ensure_dtindex(index)
    if len(idx) == 0:
        raise ValueError("Índice de negociação vazio")

    ts = pd.Timestamp(when).tz_localize(
        None) if isinstance(when, (str, datetime)) else when
    pos = idx.searchsorted(ts, side="left")

    if pos >= len(idx):
        raise IndexError(
            "Não há próximo pregão além da última data informada.")

    return idx[pos]


def prev_trading_day(index: Iterable, when: Datelike) -> pd.Timestamp:
    """Dia de negociação imediatamente anterior <= when, segundo o índice fornecido."""

    idx = ensure_dtindex(index)
    if len(idx) == 0:
        raise ValueError("Índice de negociação vazio")

    ts = pd.Timestamp(when).tz_localize(
        None) if isinstance(when, (str, datetime)) else when
    pos = idx.searchsorted(ts, side="right") - 1
    if pos < 0:
        raise IndexError("Não há dia de negociação anterior à data informada.")
    return idx[pos]


def clamp_to_index(index: Iterable, dates: Iterable[Datelike]) -> pd.DatetimeIndex:
    """Interseção: mantém apenas datas que existem no índice de negociação."""
    idx = ensure_dtindex(index)
    dts = ensure_dtindex(dates)
    return idx.intersection(dts)


def rebalance_schedule(
    index: Iterable,
    mode: str = "BMS",
    start: Optional[Datelike] = None,
    end: Optional[Datelike] = None,
    days: Optional[int] = None,
    weekly_anchor: str = "W-FRI",
) -> pd.DatetimeIndex:
    """Gera as datas de rebalance conforme o índice de negociação.

    Parâmetros
    ----------
    index : Iterable
        Índice de negociação (ex.: prices.index).
    mode : {'BMS','BME','WEEKLY'}
        BMS = primeiro pregão do mês; BME = último pregão do mês;
        WEEKLY = último pregão da semana (âncora).
    start, end : datas opcionais para recorte.
    weekly_anchor : str
        Âncora de semana do pandas (ex.: 'W-FRI', 'W-THU') quando mode='WEEKLY'.

    Retorna
    -------
    DatetimeIndex com as datas de rebalance.
    """
    idx = ensure_dtindex(index)
    if start is not None:
        idx = idx[idx >= pd.Timestamp(start)]
    if end is not None:
        idx = idx[idx <= pd.Timestamp(end)]
    if len(idx) == 0:
        return idx

    if mode.upper() == "BMS":
        sched = business_month_starts(idx)
    elif mode.upper() == "BME":
        sched = business_month_ends(idx)
    elif mode.upper() == "WEEKLY":
        sched = weekly_last_trading_day(idx, anchor=weekly_anchor)
    else:
        raise ValueError(
            f"Modo invalido: {mode}. Use 'BMS', 'BME' ou 'WEEKLY'.")

    return clamp_to_index(idx, sched)
