from __future__ import annotations

import pytest
from itau_quant.data.loader import DataLoader


@pytest.mark.skip(reason="evita chamadas de rede em CI; rodar manualmente")
def test_dataloader_load_smoke():
    dl = DataLoader(start="2024-01-01", end="2024-02-01", mode="BMS")
    bundle = dl.load()
    assert bundle.prices.index.equals(bundle.returns.index)
    assert bundle.excess_returns.index.equals(bundle.returns.index)
    assert len(bundle.bms) > 0
