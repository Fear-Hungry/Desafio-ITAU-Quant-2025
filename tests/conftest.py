from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def pytest_sessionstart(session) -> None:
    """
    Materialise the production returns snapshot from the raw ARARA price history.

    The integration smoke test loads ``data/processed/returns_full.parquet``, which
    is not versioned. When the file is absent we rebuild it deterministically from
    ``data/raw/prices_arara.csv`` so that tests run against the real dataset used in
    the project.
    """

    data_path = Path("data/processed/returns_full.parquet")
    if data_path.exists():
        return

    raw_path = Path("data/raw/prices_arara.csv")
    if not raw_path.exists():
        # In CI environments, raw data may not be available. Skip data preparation.
        # Integration tests that require real data should check for data availability.
        return

    prices = pd.read_csv(raw_path, parse_dates=["Date"], index_col="Date").sort_index()
    prices = prices.ffill().dropna(axis=1, how="all")

    returns = prices.pct_change().dropna(how="all")
    returns = returns.replace([np.inf, -np.inf], np.nan)

    # Retain assets with at least one year of data to mimic the production bundle.
    min_obs = returns.notna().sum()
    valid_assets = min_obs[min_obs >= 252].index
    filtered = returns.loc[:, valid_assets].dropna()

    data_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(data_path)
