from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from itau_quant.data.sources.yf import download_prices


@pytest.mark.skip(reason="evita chamadas de rede em CI; rodar manualmente")
def test_download_prices_smoke():
    df = download_prices(["SPY", "EFA"], start="2024-01-01", end="2024-02-01")
    assert isinstance(df, pd.DataFrame)
    assert set(["SPY", "EFA"]).issuperset(df.columns)
    assert df.notna().any().any()
