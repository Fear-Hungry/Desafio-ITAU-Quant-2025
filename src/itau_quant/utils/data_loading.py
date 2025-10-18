"""Lightweight helpers to load tabular data for optimisation/backtests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = ["read_dataframe", "read_vector"]


def read_dataframe(path: Path) -> pd.DataFrame | pd.Series:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv"}:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if suffix in {".feather"}:
        frame = pd.read_feather(path)
        frame = frame.set_index(frame.columns[0]) if not frame.columns[0] == "index" else frame
        return frame
    raise ValueError(f"Unsupported data format for {path}")


def read_vector(path: Path) -> pd.Series:
    obj = read_dataframe(path)
    if isinstance(obj, pd.Series):
        return obj.astype(float)
    if obj.empty:
        raise ValueError(f"Vector file at {path} is empty")
    if obj.shape[0] == 1:
        return obj.iloc[0].astype(float)
    if obj.shape[1] == 1:
        return obj.iloc[:, 0].astype(float)
    raise ValueError("Cannot infer vector from data frame; please provide series or single row.")
