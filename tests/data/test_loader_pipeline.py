from __future__ import annotations

import pandas as pd
import pytest

from itau_quant.data.loader import download_and_cache_arara_prices, preprocess_data
from itau_quant.data.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR


@pytest.mark.skip(reason="evita chamadas de rede em CI; rodar manualmente")
def test_download_and_preprocess_arara_smoke(tmp_path):
    # Força diretórios a existirem
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Baixa um intervalo curto para smoke (pode ser alterado p/ local mocks)
    csv_path = download_and_cache_arara_prices(
        start="2024-01-01", end="2024-02-01")
    assert csv_path.exists()

    df = preprocess_data(csv_path.name, "returns_arara_test.parquet")
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0 and df.shape[1] > 0
