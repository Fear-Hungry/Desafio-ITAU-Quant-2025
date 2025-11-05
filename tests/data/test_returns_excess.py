from __future__ import annotations

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from arara_quant.data.processing.returns import compute_excess_returns


def test_compute_excess_returns_alignment_and_broadcast():
    idx = pd.bdate_range("2024-01-01", periods=5)
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.0, -0.02, 0.01, 0.0],
            "B": [0.02, 0.01, 0.0, -0.01, 0.005],
        },
        index=idx,
    )
    # menor e sem B para todo período
    rf = pd.Series([0.0001, 0.0002, 0.0001], index=idx[:3])

    xr = compute_excess_returns(returns, rf)
    assert xr.index.equals(returns.index)
    # Checa broadcast por linha: primeira linha subtrai 0.0001 em ambas as colunas
    np.testing.assert_allclose(
        xr.loc[idx[0]].values, returns.loc[idx[0]].values - 0.0001
    )
    # Última linha usa ffill do rf (0.0001)
    np.testing.assert_allclose(
        xr.loc[idx[-1]].values, returns.loc[idx[-1]].values - 0.0001
    )
