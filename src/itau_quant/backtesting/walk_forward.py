"""Walk-forward utilities with optional purging and embargo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd

__all__ = ["WalkForwardSplit", "generate_walk_forward_splits"]


@dataclass(frozen=True)
class WalkForwardSplit:
    train_index: pd.DatetimeIndex
    test_index: pd.DatetimeIndex

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "train": [ts.isoformat() for ts in self.train_index],
            "test": [ts.isoformat() for ts in self.test_index],
        }


def generate_walk_forward_splits(
    index: pd.DatetimeIndex,
    *,
    train_window: int,
    test_window: int,
    purge_window: int = 0,
    embargo_window: int = 0,
) -> Iterator[WalkForwardSplit]:
    """Yield ``WalkForwardSplit`` objects for the supplied index."""

    if train_window <= 0 or test_window <= 0:
        raise ValueError("train_window and test_window must be positive integers")

    dates = pd.DatetimeIndex(index)
    n = len(dates)
    cursor = train_window + purge_window

    while cursor + test_window <= n:
        train_end_pos = cursor - purge_window - 1
        train_start_pos = train_end_pos - train_window + 1
        if train_start_pos < 0:
            cursor += test_window
            continue

        test_start_pos = cursor
        test_end_pos = test_start_pos + test_window

        train_slice = dates[train_start_pos : train_end_pos + 1]
        test_slice = dates[test_start_pos:test_end_pos]

        if train_slice.empty or test_slice.empty:
            break

        yield WalkForwardSplit(train_index=train_slice, test_index=test_slice)

        cursor = test_end_pos + embargo_window
