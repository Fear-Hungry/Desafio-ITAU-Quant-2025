import numpy as np
import pandas as pd
import pytest
from arara_quant.evaluation.stats import (
    block_bootstrap,
    bootstrap_metric,
    compare_vs_benchmark,
    confidence_interval,
)


def _toy_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=20, freq="B")
    values = np.linspace(-0.02, 0.03, len(idx))
    return pd.DataFrame({"portfolio": values}, index=idx)


def test_block_bootstrap_preserves_shape():
    samples = block_bootstrap(_toy_df(), block_size=5, n_samples=3, random_state=42)
    assert len(samples) == 3
    for sample in samples:
        assert isinstance(sample, pd.DataFrame)
        assert sample.shape == (20, 1)


def test_bootstrap_metric_mean_estimator_closely_matches_sample_mean():
    data = _toy_df()
    observed_mean = data.mean()
    bootstrap_results = bootstrap_metric(
        lambda df: df.mean(), data, n_samples=200, block_size=4, random_state=7
    )
    assert bootstrap_results.shape[0] == 200
    assert bootstrap_results.mean().iloc[0] == pytest.approx(
        observed_mean.iloc[0], rel=0.2
    )


def test_confidence_interval_percentile():
    samples = np.array([1, 2, 3, 4, 5], dtype=float)
    lower, upper = confidence_interval(samples, alpha=0.2, method="percentile")
    assert lower <= upper
    assert lower == pytest.approx(1.4)
    assert upper == pytest.approx(4.6)


def test_compare_vs_benchmark_scalar_and_dataframe():
    samples = np.array([0.5, 0.6, 0.4, 0.7])
    prob = compare_vs_benchmark(samples, 0.5)
    assert prob == pytest.approx(0.5)

    df_samples = pd.DataFrame({"portfolio": samples, "alt": [0.1, 0.2, 0.3, 0.4]})
    probs = compare_vs_benchmark(df_samples, pd.Series({"portfolio": 0.5, "alt": 0.2}))
    assert set(probs.index) == {"portfolio", "alt"}
    assert probs["portfolio"] == pytest.approx(0.5)
    assert probs["alt"] == pytest.approx(0.5)
