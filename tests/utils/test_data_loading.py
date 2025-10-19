from __future__ import annotations

import pandas as pd
import pytest

from itau_quant.utils import data_loading


def test_to_datetime_index_converts_series_index() -> None:
    index = pd.Index(["2020-01-01", "2020-01-02"])
    converted = data_loading.to_datetime_index(index)
    assert isinstance(converted, pd.DatetimeIndex)
    assert converted.tz is None


def test_read_vector_single_row(tmp_path) -> None:
    path = tmp_path / "vector.csv"
    frame = pd.DataFrame([[0.1, 0.2]], columns=["A", "B"])
    frame.to_csv(path)
    vector = data_loading.read_vector(path)
    assert isinstance(vector, pd.Series)
    assert pytest.approx(vector["A"]) == 0.1
