from __future__ import annotations

import pytest

from itau_quant.data.loader import DataLoader, DataBundle


@pytest.mark.skip(reason="evita chamadas de rede em CI; rodar manualmente")
def test_dataloader_load_bundle_smoke():
    dl = DataLoader(start="2024-01-01", end="2024-02-01", mode="BMS")
    bundle = dl.load()
    assert isinstance(bundle, DataBundle)
    assert bundle.prices.index.equals(bundle.returns.index)
    assert bundle.excess_returns.index.equals(bundle.returns.index)

