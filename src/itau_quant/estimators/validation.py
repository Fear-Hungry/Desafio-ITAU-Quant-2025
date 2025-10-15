"""Temporal validation helpers enforcing purging and embargo rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

__all__ = [
    "temporal_split",
    "purge_train_indices",
    "apply_embargo",
    "PurgedKFold",
    "evaluate_estimator",
]


def _ensure_index(index: Union[pd.Index, Sequence]) -> pd.Index:
    """Return an increasing pandas index from supported inputs."""

    if isinstance(index, pd.Index):
        idx = index
    else:
        idx = pd.Index(index)
    if idx.empty:
        raise ValueError("index cannot be empty.")
    if not idx.is_monotonic_increasing:
        raise ValueError("index must be sorted in ascending order.")
    return idx


def temporal_split(
    index: Union[pd.Index, Sequence],
    n_splits: int,
    min_train: int,
    min_test: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate ordered train/test splits honouring minimum window sizes.

    Parameters
    ----------
    index
        Sorted index describing the chronological order of observations.
    n_splits
        Number of folds to produce.
    min_train
        Minimum number of observations that must precede a test window.
    min_test
        Size (in observations) of each test window.
    """

    if n_splits <= 0:
        raise ValueError("n_splits must be positive.")
    if min_train <= 0 or min_test <= 0:
        raise ValueError("min_train and min_test must be positive.")

    idx = _ensure_index(index)
    n_samples = len(idx)
    min_required = min_train + min_test
    if n_samples < min_required:
        raise ValueError("Not enough observations for the requested windows.")

    available_starts = np.arange(min_train, n_samples - min_test + 1)
    if len(available_starts) < n_splits:
        raise ValueError("Not enough room to generate the requested splits.")

    start_chunks = np.array_split(available_starts, n_splits)
    test_starts = [chunk[0] for chunk in start_chunks]

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for start in test_starts:
        stop = start + min_test
        train_pre = np.arange(0, start)
        train_post = np.arange(stop, n_samples)
        train_idx = np.concatenate([train_pre, train_post])
        if train_idx.size < min_train:
            raise ValueError("Training window shorter than min_train.")
        test_idx = np.arange(start, stop)
        splits.append((train_idx, test_idx))

    return splits


def purge_train_indices(
    train_idx: Iterable[int],
    test_idx: Iterable[int],
    purge_window: Union[int, float] = 0,
) -> np.ndarray:
    """Remove training indices that fall immediately before the test window."""

    train_arr = np.asarray(train_idx, dtype=int)
    test_arr = np.asarray(test_idx, dtype=int)

    if purge_window is None:
        purge_window = 0

    purge_window = int(np.ceil(float(purge_window)))
    if purge_window <= 0:
        return np.sort(train_arr)

    if test_arr.size == 0:
        return np.sort(train_arr)

    first_test = int(test_arr.min())
    cutoff = first_test - purge_window
    mask = (train_arr < cutoff) | (train_arr >= first_test)
    purged = np.sort(train_arr[mask])
    return purged


def apply_embargo(
    train_idx: Iterable[int],
    test_idx: Iterable[int],
    embargo_pct: float = 0.0,
    total_observations: Optional[int] = None,
) -> np.ndarray:
    """Apply an embargo after the test window to avoid look-ahead bias."""

    train_arr = np.asarray(train_idx, dtype=int)
    test_arr = np.asarray(test_idx, dtype=int)

    if embargo_pct is None:
        embargo_pct = 0.0
    if embargo_pct < 0 or embargo_pct >= 1:
        raise ValueError("embargo_pct must lie in [0, 1).")
    if embargo_pct == 0 or test_arr.size == 0:
        return np.sort(train_arr)

    if train_arr.size == 0:
        return np.array([], dtype=int)

    if total_observations is None:
        max_train = int(train_arr.max()) if train_arr.size > 0 else 0
        max_test = int(test_arr.max()) if test_arr.size > 0 else 0
        total_observations = max(max_train, max_test) + 1

    embargo_count = int(np.ceil(total_observations * embargo_pct))
    if embargo_count <= 0:
        return np.sort(train_arr)

    last_test = int(test_arr.max())
    embargo_start = last_test + 1
    embargo_end = min(total_observations, embargo_start + embargo_count)
    embargo_range = np.arange(embargo_start, embargo_end)
    mask = ~np.isin(train_arr, embargo_range)
    embargoed = np.sort(train_arr[mask])
    return embargoed


@dataclass
class PurgedKFold:
    """K-fold splitter with purging and embargo for temporal datasets."""

    n_splits: int = 5
    min_train: int = 60
    min_test: int = 10
    purge_window: Union[int, float] = 0
    embargo_pct: float = 0.0

    def get_n_splits(self, X=None, y=None, groups=None) -> int:  # noqa: D401
        """Return the number of configured splits."""

        return self.n_splits

    def split(
        self,
        X: Union[pd.DataFrame, pd.Series, np.ndarray, Sequence],
        y: Optional[Sequence] = None,
        groups: Optional[Sequence] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield purged train/test indices according to the configuration."""

        if isinstance(X, (pd.Series, pd.DataFrame)):
            index = _ensure_index(X.index)
        else:
            X = np.asarray(X)
            index = pd.RangeIndex(len(X))

        splits = temporal_split(index, self.n_splits, self.min_train, self.min_test)
        total_obs = len(index)

        for train_idx, test_idx in splits:
            purged_train = purge_train_indices(train_idx, test_idx, self.purge_window)
            embargoed_train = apply_embargo(
                purged_train, test_idx, self.embargo_pct, total_observations=total_obs
            )
            yield embargoed_train, test_idx


def evaluate_estimator(
    estimator_fn: Callable[
        [pd.DataFrame, pd.DataFrame], Union[pd.Series, pd.DataFrame]
    ],
    data: pd.DataFrame,
    scoring: Callable[[pd.DataFrame, Union[pd.Series, pd.DataFrame]], float],
    splitter: PurgedKFold,
) -> pd.Series:
    """Apply cross-validation and return the per-fold scores."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    scores: List[float] = []

    for train_idx, test_idx in splitter.split(data):
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]
        predictions = estimator_fn(train_data, test_data)
        score = scoring(test_data, predictions)
        scores.append(float(score))

    return pd.Series(scores, name="score")
