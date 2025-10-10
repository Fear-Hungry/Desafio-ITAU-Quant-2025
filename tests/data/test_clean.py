from __future__ import annotations

import pandas as pd

from itau_quant.data.processing.clean import ensure_dtindex, normalize_index, validate_panel


def test_ensure_dtindex_normalizes_and_sorts():
    idx = ["2024-01-02", "2024-01-01", "2024-01-01"]
    out = ensure_dtindex(idx)
    assert isinstance(out, pd.DatetimeIndex)
    assert out.is_monotonic_increasing
    assert out.is_unique


def test_normalize_index_dataframe():
    df = pd.DataFrame({"A": [1, 2]}, index=["2024-01-02", "2024-01-01"])
    out = normalize_index(df)
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.is_monotonic_increasing


def test_validate_panel_basic_checks():
    df = pd.DataFrame({"A": [1.0, 2.0]},
                      index=pd.bdate_range("2024-01-01", periods=2))
    validate_panel(df)  # não deve lançar
