from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from itau_quant.data.sources.fred import download_dtb3


@pytest.mark.skip(reason="evita chamadas de rede em CI; rodar manualmente")
def test_download_dtb3_smoke():
    rf = download_dtb3(start="2024-01-01", end="2024-02-01")
    assert isinstance(rf, pd.Series)
    assert rf.index.freqstr == "B" or rf.index.inferred_freq in ("B", None)
    assert rf.notna().any()
