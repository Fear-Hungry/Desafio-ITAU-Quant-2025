"""Pipeline step 1: Data acquisition and preprocessing.

This module orchestrates the ingestion of market data using the shared
``DataLoader`` facade. It handles caching, persistence of artefacts under
deterministic hash-based filenames and backwards-compatible legacy outputs
used by scripts and notebooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from arara_quant.config import Settings
from arara_quant.data import DataLoader, get_arara_universe
from arara_quant.data.storage import save_parquet
from arara_quant.utils.logging_config import get_logger

__all__ = ["download_and_prepare_data"]

logger = get_logger(__name__)


def _to_str_path(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def _baseline_tickers(tickers: Iterable[str] | None) -> list[str]:
    if tickers is None:
        return get_arara_universe()
    return list(tickers)


def download_and_prepare_data(
    *,
    start: str | None = None,
    end: str | None = None,
    tickers: Iterable[str] | None = None,
    mode: str = "BMS",
    actions: Sequence[Mapping[str, object]] | None = None,
    raw_file_name: str | None = "prices_arara.csv",
    processed_file_name: str | None = "returns_arara.parquet",
    force_download: bool = False,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Execute data pipeline: download prices, compute returns and persist artefacts.

    Args:
        start: Optional start date in ISO format (YYYY-MM-DD) for downloads.
        end: Optional end date in ISO format (YYYY-MM-DD) for downloads.
        tickers: Iterable of tickers to override the default ARARA universe.
        mode: Calendar mode passed to ``rebalance_schedule`` (e.g., ``"BMS"``).
        actions: Optional corporate action records to adjust the price history.
        raw_file_name: When provided, legacy CSV written under ``data/raw`` for compatibility.
        processed_file_name: When provided, legacy returns Parquet under ``data/processed``.
        force_download: If True, bypass cached artefacts and fetch fresh data.
        settings: Optional :class:`Settings` instance (auto-resolved when ``None``).

    Returns:
        dict with execution metadata and artefact locations. Keys of interest:
            - ``status``: textual stage outcome (``"completed"`` on success).
            - ``returns_file``: hashed Parquet filename under processed data dir.
            - ``returns_path``: absolute path to the hashed returns Parquet.
            - ``raw_path`` / ``processed_path``: legacy outputs (may be ``None``).
            - ``request_id``: deterministic hash of tickers and date range.
            - ``from_cache``: whether artefacts were reused instead of downloaded.
            - ``n_days`` / ``n_assets``: shape of the returns panel.
    """

    settings = settings or Settings.from_env()
    tickers_list = _baseline_tickers(tickers)

    loader = DataLoader(
        tickers=tickers_list,
        start=start,
        end=end,
        mode=mode,
        actions=list(actions) if actions is not None else None,
    )
    bundle = loader.load(force_download=force_download, cache=True)
    artefacts = loader.artifacts

    raw_path: Path | None = None
    if raw_file_name:
        raw_path = Path(settings.raw_data_dir) / raw_file_name
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        bundle.prices.to_csv(raw_path, index=True)

    processed_path: Path | None = None
    if processed_file_name:
        processed_path = Path(settings.processed_data_dir) / processed_file_name
        save_parquet(processed_path, bundle.returns)

    artefact_paths: dict[str, str | None] = {}
    for key in (
        "prices_path",
        "returns_path",
        "excess_path",
        "rf_path",
        "metadata_path",
    ):
        value = artefacts.get(key)
        artefact_paths[key] = str(value) if isinstance(value, Path) else None

    returns_path = artefacts.get("returns_path")
    returns_file = (
        returns_path.name
        if isinstance(returns_path, Path)
        else (processed_path.name if processed_path is not None else None)
    )
    prices_path = artefacts.get("prices_path")
    prices_file = prices_path.name if isinstance(prices_path, Path) else None
    excess_path = artefacts.get("excess_path")
    excess_file = excess_path.name if isinstance(excess_path, Path) else None
    rf_path = artefacts.get("rf_path")
    rf_file = rf_path.name if isinstance(rf_path, Path) else None
    metadata_path = artefacts.get("metadata_path")
    metadata_file = metadata_path.name if isinstance(metadata_path, Path) else None

    start_ts = bundle.prices.index.min() if not bundle.prices.empty else pd.NaT
    end_ts = bundle.prices.index.max() if not bundle.prices.empty else pd.NaT
    start_iso = start_ts.strftime("%Y-%m-%d") if pd.notna(start_ts) else None
    end_iso = end_ts.strftime("%Y-%m-%d") if pd.notna(end_ts) else None

    if artefacts.get("from_cache"):
        logger.info(
            "DataLoader reutilizou cache (request_id=%s)", artefacts["request_id"]
        )
    else:
        logger.info(
            "DataLoader persistiu novos artefatos (request_id=%s)",
            artefacts["request_id"],
        )

    metadata = artefacts.get("metadata") or {}
    trimmed_metadata = {k: v for k, v in metadata.items() if k != "tickers"}

    result: dict[str, Any] = {
        "status": "completed",
        "raw_path": _to_str_path(raw_path),
        "processed_path": _to_str_path(processed_path),
        "n_days": int(bundle.returns.shape[0]),
        "n_assets": int(bundle.returns.shape[1]),
        "request_id": artefacts.get("request_id"),
        "from_cache": bool(artefacts.get("from_cache", False)),
        "returns_file": returns_file,
        "prices_file": prices_file,
        "excess_returns_file": excess_file,
        "rf_file": rf_file,
        "metadata_file": metadata_file,
        "returns_path": artefact_paths["returns_path"],
        "prices_path": artefact_paths["prices_path"],
        "excess_returns_path": artefact_paths["excess_path"],
        "rf_path": artefact_paths["rf_path"],
        "metadata_path": artefact_paths["metadata_path"],
        "start_date": start_iso,
        "end_date": end_iso,
        "n_requested_tickers": len(tickers_list),
    }
    if trimmed_metadata:
        result["artefact_metadata"] = trimmed_metadata

    return result
