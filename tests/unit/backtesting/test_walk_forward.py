from __future__ import annotations

import pandas as pd
from arara_quant.backtesting.walk_forward import generate_walk_forward_splits


def test_generate_walk_forward_basic_split() -> None:
    index = pd.date_range("2020-01-01", periods=10, freq="D")
    splits = list(
        generate_walk_forward_splits(
            index,
            train_window=4,
            test_window=2,
        )
    )
    assert len(splits) == 3
    first = splits[0]
    pd.testing.assert_index_equal(first.train_index, index[:4])
    pd.testing.assert_index_equal(first.test_index, index[4:6])


def test_generate_walk_forward_with_purge_and_embargo() -> None:
    index = pd.date_range("2021-01-01", periods=12, freq="D")
    splits = list(
        generate_walk_forward_splits(
            index,
            train_window=5,
            test_window=2,
            purge_window=1,
            embargo_window=1,
        )
    )
    # Ensure the embargo skips one observation between test windows
    assert splits[0].test_index.max() < splits[1].test_index.min()


def test_generate_walk_forward_raises_on_invalid_window() -> None:
    index = pd.date_range("2020-01-01", periods=5, freq="D")
    try:
        list(generate_walk_forward_splits(index, train_window=0, test_window=2))
    except ValueError as exc:
        assert "must be positive" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-positive windows")
