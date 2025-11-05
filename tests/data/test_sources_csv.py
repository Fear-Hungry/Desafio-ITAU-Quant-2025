from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from arara_quant.data.sources.csv import CSVSchemaError, load_price_panel


def _write_csv(tmp_path: Path, filename: str, rows: list[dict[str, object]]) -> Path:
    df = pd.DataFrame(rows)
    path = tmp_path / filename
    df.to_csv(path, index=False)
    return path


def test_load_price_panel_success(tmp_path: Path):
    csv_path = _write_csv(
        tmp_path,
        "prices.csv",
        [
            {"date": "2024-01-01", "SPY": 470.0, "QQQ": 400.0},
            {"date": "2024-01-02", "SPY": 471.5, "QQQ": 402.2},
        ],
    )

    df = load_price_panel(csv_path, expected_columns=["SPY", "QQQ"])

    assert list(df.columns) == ["SPY", "QQQ"]
    expected_index = pd.DatetimeIndex(["2024-01-01", "2024-01-02"])
    assert df.index.equals(expected_index)


def test_load_price_panel_missing_expected_column(tmp_path: Path):
    csv_path = _write_csv(
        tmp_path,
        "prices.csv",
        [
            {"date": "2024-01-01", "SPY": 470.0},
            {"date": "2024-01-02", "SPY": 471.5},
        ],
    )

    with pytest.raises(CSVSchemaError):
        load_price_panel(csv_path, expected_columns=["SPY", "QQQ"])


def test_load_price_panel_rejects_duplicate_dates(tmp_path: Path):
    csv_path = _write_csv(
        tmp_path,
        "prices.csv",
        [
            {"date": "2024-01-01", "SPY": 470.0},
            {"date": "2024-01-01", "SPY": 471.5},
        ],
    )

    with pytest.raises(CSVSchemaError):
        load_price_panel(csv_path, expected_columns=["SPY"])
