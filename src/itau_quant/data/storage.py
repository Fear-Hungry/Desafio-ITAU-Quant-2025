"""Parquet IO utilities with simple conventions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def save_parquet(path: Path, obj: Any) -> None:
    """Persist DataFrame/Series using zstd compression."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, pd.Series):
        obj.to_frame().to_parquet(path, compression="zstd", index=True)
    else:
        obj.to_parquet(path, compression="zstd", index=True)


def load_parquet(path: Path) -> pd.DataFrame:
    """Load Parquet artefact into a pandas object."""
    return pd.read_parquet(path)
