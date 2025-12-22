#!/usr/bin/env python
"""Execute the standard data ingestion and preprocessing pipeline.

Steps
-----
1. Ensure ARARA universe prices are available under ``data/raw``.
2. Convert prices to log returns and persist them in ``data/processed``.

The script reuses existing cached data by default to avoid unnecessary downloads.
Pass ``--force-download`` to fetch fresh prices via Yahoo Finance.

This script is now a thin wrapper around the reusable pipeline.data module.
"""

from __future__ import annotations

import argparse
from datetime import datetime

from arara_quant.config import Settings
from arara_quant.pipeline.data import download_and_prepare_data
from arara_quant.utils.logging_config import get_logger, log_dict

logger = get_logger("runners.run_01_data_pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the data ingestion pipeline.")
    parser.add_argument(
        "--raw-file",
        default="prices_arara.csv",
        help="File name stored under data/raw/ with cached prices.",
    )
    parser.add_argument(
        "--processed-file",
        default="returns_arara.parquet",
        help="Output file name stored under data/processed/ with log returns.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Optional start date (YYYY-MM-DD) when downloading fresh prices.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Optional end date (YYYY-MM-DD) when downloading fresh prices.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Ignore cached raw data and trigger a fresh download.",
    )
    return parser.parse_args()


def _coerce_date(raw: str | None) -> str | None:
    if raw is None:
        return None
    try:
        return datetime.fromisoformat(raw).date().isoformat()
    except ValueError as exc:
        raise SystemExit(f"Invalid date '{raw}': {exc}") from exc


def main() -> None:
    args = parse_args()
    settings = Settings.from_env()

    start = _coerce_date(args.start)
    end = _coerce_date(args.end)

    # Call the reusable module function
    try:
        result = download_and_prepare_data(
            start=start,
            end=end,
            raw_file_name=args.raw_file,
            processed_file_name=args.processed_file,
            force_download=args.force_download,
            settings=settings,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Failed to execute data pipeline: %s", exc)
        raise SystemExit(1) from exc

    log_dict(logger, "Data pipeline completed", result)


if __name__ == "__main__":
    main()
