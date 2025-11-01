from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from itau_quant.data.processing.calendar import (
    business_month_ends,
    business_month_starts,
    clamp_to_index,
    next_trading_day,
    prev_trading_day,
    weekly_last_trading_day,
)


def _mk_index():
    # índice com dias úteis; simulamos pregões removendo alguns dias
    idx = pd.bdate_range("2024-01-01", periods=40)
    # remove alguns dias para simular feriados
    idx = idx.delete([3, 10, 11, 20])
    return idx


def test_business_month_starts_and_ends():
    idx = _mk_index()
    bms = business_month_starts(idx)
    bme = business_month_ends(idx)

    assert len(bms) >= 2 and len(bme) >= 2
    # BMS deve ser o primeiro dia presente daquele mês no índice
    assert bms[0] == idx[idx.to_period("M") == idx[0].to_period("M")][0]
    # BME deve ser o último dia presente daquele mês no índice
    assert bme[0] == idx[idx.to_period("M") == idx[0].to_period("M")][-1]


def test_weekly_last_trading_day_and_neighbors():
    idx = _mk_index()
    wlast = weekly_last_trading_day(idx, anchor="W-FRI")
    assert len(wlast) >= 3

    # Próximo e anterior pregões
    some_day = idx[5]  # dia existente
    assert next_trading_day(idx, some_day) == some_day
    assert prev_trading_day(idx, some_day) <= some_day


def test_clamp_to_index():
    idx = _mk_index()
    dates = [idx[0], idx[5], idx[-1], pd.Timestamp("1900-01-01")]  # inclui um fora
    clamped = clamp_to_index(idx, dates)
    assert clamped.min() >= idx.min()
    assert clamped.max() <= idx.max()
    # A data fora não deve estar presente
    assert pd.Timestamp("1900-01-01") not in clamped
