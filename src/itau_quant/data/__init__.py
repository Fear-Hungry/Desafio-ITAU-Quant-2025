from __future__ import annotations

from .loader import DataLoader, DataBundle, download_and_preprocess_arara, download_fred_dtb3
from .universe import get_arara_universe
from .processing.calendar import business_month_starts, business_month_ends, rebalance_schedule
from .processing.returns import calculate_returns, compute_excess_returns

__all__ = [
    "DataLoader",
    "DataBundle",
    "download_and_preprocess_arara",
    "download_fred_dtb3",
    "get_arara_universe",
    "business_month_starts",
    "business_month_ends",
    "rebalance_schedule",
    "calculate_returns",
    "compute_excess_returns",
]
