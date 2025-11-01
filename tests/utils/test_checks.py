from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from itau_quant.utils import checks


def test_assert_no_nans_raises_on_nan_dataframe() -> None:
    frame = pd.DataFrame({"A": [1.0, np.nan]})
    with pytest.raises(ValueError):
        checks.assert_no_nans(frame, context="test")


def test_assert_shape_accepts_valid_shape() -> None:
    frame = pd.DataFrame(np.ones((3, 2)))
    checks.assert_shape(frame, expected_shape=(3, 2))
    with pytest.raises(ValueError):
        checks.assert_shape(frame, expected_shape=(2, 2))


def test_assert_psd_detects_non_psd_matrix() -> None:
    matrix = np.array([[1, 2], [2, 1]])
    with pytest.raises(ValueError):
        checks.assert_psd(matrix)


def test_validate_returns_frame_requires_datetime_index() -> None:
    frame = pd.DataFrame(
        {"A": [0.01, 0.02]}, index=pd.date_range("2020-01-01", periods=2)
    )
    checks.validate_returns_frame(frame)
    frame_bad = frame.copy()
    frame_bad.index = [0, 1]
    with pytest.raises(TypeError):
        checks.validate_returns_frame(frame_bad)
