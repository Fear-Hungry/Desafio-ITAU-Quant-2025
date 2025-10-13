"""Public API for the data subpackage."""

from .universe import get_arara_universe
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
from .processing.returns import compute_excess_returns
from .processing.calendar import (
    business_month_starts,
    business_month_ends,
    rebalance_schedule,
)

__all__ = [
    "get_arara_universe",
    "load_asset_prices",
    "calculate_returns",
    "download_and_cache_arara_prices",
    "preprocess_data",
    "download_and_preprocess_arara",
    "download_fred_dtb3",
    "DataLoader",
    "DataBundle",
    "compute_excess_returns",
    "business_month_starts",
    "business_month_ends",
    "rebalance_schedule",
]
