"""Tests for temporal validation utilities."""

import numpy as np
import pandas as pd
import pytest
from arara_quant.estimators import validation


def _build_index(n: int = 30) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="D")


def test_temporal_split_generates_expected_windows():
    idx = _build_index(30)
    splits = validation.temporal_split(idx, n_splits=3, min_train=10, min_test=5)

    assert len(splits) == 3
    for train_idx, test_idx in splits:
        assert len(test_idx) == 5
        assert len(train_idx) == len(idx) - 5
        assert np.intersect1d(train_idx, test_idx).size == 0
        # Ensure both sides of the test window contribute to the training set.
        assert np.any(train_idx > test_idx.max())


def test_temporal_split_counts_full_training_length():
    idx = _build_index(18)
    splits = validation.temporal_split(idx, n_splits=2, min_train=8, min_test=4)

    for train_idx, test_idx in splits:
        assert train_idx.size >= 8
        assert train_idx.size == len(idx) - test_idx.size


def test_purge_train_indices_removes_recent_observations():
    train_idx = np.concatenate([np.arange(0, 10), np.arange(15, 30)])
    test_idx = np.arange(10, 15)
    purged = validation.purge_train_indices(train_idx, test_idx, purge_window=3)

    assert not np.isin([7, 8, 9], purged).any()
    assert np.isin([0, 6, 15, 20], purged).all()


def test_apply_embargo_removes_future_samples():
    train_idx = np.concatenate([np.arange(0, 10), np.arange(15, 30)])
    test_idx = np.arange(10, 15)
    embargoed = validation.apply_embargo(
        train_idx, test_idx, embargo_pct=0.1, total_observations=30
    )

    assert not np.isin([15, 16, 17], embargoed).any()
    assert np.isin([18, 19, 20], embargoed).all()


def test_purged_kfold_produces_non_overlapping_folds():
    data = pd.DataFrame({"x": np.arange(40)}, index=_build_index(40))
    splitter = validation.PurgedKFold(
        n_splits=3,
        min_train=10,
        min_test=5,
        purge_window=2,
        embargo_pct=0.1,
    )

    for train_idx, test_idx in splitter.split(data):
        assert np.intersect1d(train_idx, test_idx).size == 0
        if test_idx.size > 0:
            first_test = test_idx.min()
            purge = int(np.ceil(splitter.purge_window))
            forbidden_pre = np.arange(first_test - purge, first_test)
            assert not np.isin(forbidden_pre, train_idx).any()
            last_test = test_idx.max()
            embargo_len = int(np.ceil(len(data) * splitter.embargo_pct))
            embargo_range = np.arange(last_test + 1, last_test + 1 + embargo_len)
            assert not np.isin(embargo_range, train_idx).any()


def test_evaluate_estimator_pipeline_returns_scores():
    rng = np.random.default_rng(0)
    data = pd.DataFrame({"ret": rng.normal(0.0, 0.01, size=50)}, index=_build_index(50))
    splitter = validation.PurgedKFold(
        n_splits=4, min_train=15, min_test=5, purge_window=2
    )

    def estimator(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
        mean_ret = train["ret"].mean()
        return pd.Series(mean_ret, index=test.index)

    def mse(test: pd.DataFrame, preds: pd.Series) -> float:
        return float(np.mean((test["ret"] - preds) ** 2))

    scores = validation.evaluate_estimator(estimator, data, mse, splitter)
    assert isinstance(scores, pd.Series)
    assert len(scores) == splitter.n_splits
    assert np.all(np.isfinite(scores))


def test_temporal_split_raises_on_small_sample():
    idx = _build_index(5)
    with pytest.raises(ValueError):
        validation.temporal_split(idx, n_splits=2, min_train=4, min_test=2)
