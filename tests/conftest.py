from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def pytest_sessionstart(session) -> None:
    """
    Ensure integration tests have access to a precomputed returns dataset.

    The production smoke test loads ``data/processed/returns_full.parquet`` at import
    time. When developers clone the project the file is not versioned, so we lazily
    materialise a lightweight synthetic sample to keep the test runnable while
    preserving the original workflow.
    """

    data_path = Path("data/processed/returns_full.parquet")
    if data_path.exists():
        return

    data_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate two years of business-day returns for a dozen assets.
    index = pd.bdate_range(end=pd.Timestamp.today(), periods=756, name="date")
    asset_names = [f"ASSET_{i:02d}" for i in range(12)]
    rng = np.random.default_rng(seed=42)
    values = rng.normal(loc=0.0004, scale=0.01, size=(len(index), len(asset_names)))
    synthetic_returns = pd.DataFrame(values, index=index, columns=asset_names)

    synthetic_returns.to_parquet(data_path)
