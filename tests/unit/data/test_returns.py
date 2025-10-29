from __future__ import annotations

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from itau_quant.data.processing.returns import calculate_returns


def test_calculate_returns_log_and_simple():
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    prices = pd.DataFrame({"A": [100, 101, 99, 99, 100], "B": [
                          50, 50, 50.5, 51, 51]}, index=idx)

    r_log = calculate_returns(prices, method="log")
    r_simple = calculate_returns(prices, method="simple")

    assert r_log.shape == r_simple.shape
    # Para variações pequenas, log ~ simple
    diff = (np.exp(r_log) - 1) - r_simple
    assert (diff.abs().max().max() <
            1e-6) or (diff.dropna().abs().max().max() < 1e-5)
