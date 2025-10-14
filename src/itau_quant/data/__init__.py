"""API pública consolidada do subpacote ``itau_quant.data``.

Organização
-----------
- Universo/Metadados → ``get_arara_universe``, ``get_arara_metadata``.
- Facade do pipeline → ``DataLoader``, ``DataBundle`` e helpers de download.
- Processamento → retornos, calendário, winsorização e afins.
- Utilidades → cache, paths padrão, persistência.

Recomendação: consumidores externos importam diretamente daqui para preservar
encapsulamento das camadas internas.
"""

from .universe import get_arara_metadata, get_arara_universe
from .loader import (
    load_asset_prices,
    calculate_returns,
    download_and_cache_arara_prices,
    preprocess_data,
    download_and_preprocess_arara,
    download_fred_dtb3,
    DataLoader,
    DataBundle,
)
from .processing.clean import winsorize_outliers
from .processing.returns import compute_excess_returns
from .processing.calendar import (
    business_month_starts,
    business_month_ends,
    rebalance_schedule,
)

__all__ = [
    "get_arara_universe",
    "get_arara_metadata",
    "load_asset_prices",
    "calculate_returns",
    "download_and_cache_arara_prices",
    "preprocess_data",
    "download_and_preprocess_arara",
    "download_fred_dtb3",
    "DataLoader",
    "DataBundle",
    "winsorize_outliers",
    "compute_excess_returns",
    "business_month_starts",
    "business_month_ends",
    "rebalance_schedule",
]
