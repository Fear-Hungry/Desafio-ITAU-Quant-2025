"""Pipeline step 1: Data acquisition and preprocessing.

This module extracts the data download and preprocessing logic from
run_01_data_pipeline.py into a reusable function that can be called
programmatically or from the CLI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from itau_quant.config import Settings
from itau_quant.data import download_and_cache_arara_prices, preprocess_data
from itau_quant.utils.logging_config import get_logger

__all__ = ["download_and_prepare_data"]

logger = get_logger(__name__)


def download_and_prepare_data(
    *,
    start: str | None = None,
    end: str | None = None,
    raw_file_name: str = "prices_arara.csv",
    processed_file_name: str = "returns_arara.parquet",
    force_download: bool = False,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Execute data pipeline: download prices and generate returns.

    This function orchestrates the data acquisition and preprocessing stages:
    1. Downloads (or reuses cached) price data from Yahoo Finance
    2. Converts prices to log returns
    3. Persists both raw and processed data

    Args:
        start: Optional start date in ISO format (YYYY-MM-DD)
        end: Optional end date in ISO format (YYYY-MM-DD)
        raw_file_name: Filename for cached raw prices (CSV)
        processed_file_name: Filename for processed returns (Parquet)
        force_download: If True, ignore cache and download fresh data
        settings: Settings object (uses default if None)

    Returns:
        dict containing:
            - status: "completed"
            - raw_path: Absolute path to raw CSV file
            - processed_path: Absolute path to processed Parquet file
            - n_days: Number of trading days in returns
            - n_assets: Number of assets/tickers

    Raises:
        Exception: If download or preprocessing fails

    Examples:
        >>> result = download_and_prepare_data(start="2020-01-01")
        >>> print(f"Downloaded {result['n_assets']} assets")
    """
    settings = settings or Settings.from_env()

    raw_path = Path(settings.raw_data_dir) / raw_file_name

    # Check if we can reuse cached data
    if raw_path.exists() and not force_download:
        logger.info("Reusing cached raw data at %s", raw_path)
    else:
        logger.info("Downloading ARARA prices (force=%s)", force_download)
        try:
            raw_path = download_and_cache_arara_prices(
                start=start,
                end=end,
                raw_file_name=raw_file_name,
            )
        except Exception as exc:
            logger.error("Failed to download prices: %s", exc)
            raise

    # Preprocess: convert prices to returns
    logger.info("Preprocessing cached data into log returns")
    returns = preprocess_data(raw_file_name, processed_file_name)
    processed_path = Path(settings.processed_data_dir) / processed_file_name

    return {
        "status": "completed",
        "raw_path": str(raw_path),
        "processed_path": str(processed_path),
        "n_days": int(returns.shape[0]),
        "n_assets": int(returns.shape[1]),
    }
