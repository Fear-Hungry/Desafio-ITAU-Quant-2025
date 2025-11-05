"""Bootstrap utilities tailored for time-series metrics."""

from __future__ import annotations

from typing import Callable, Sequence, Union

import numpy as np
import pandas as pd

ReturnsLike = Union[pd.Series, pd.DataFrame]
MetricOutput = Union[float, pd.Series, pd.DataFrame]


def _quantile(array: np.ndarray, q: float) -> float:
    try:
        return float(np.quantile(array, q, method="linear"))
    except TypeError:
        # numpy < 1.22 uses ``interpolation`` instead of ``method``
        return float(np.quantile(array, q, interpolation="linear"))


def _to_frame(data: ReturnsLike) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, pd.Series):
        df = data.to_frame()
    else:
        raise TypeError("returns must be a pandas Series or DataFrame")
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if df.empty:
        raise ValueError("returns series is empty after dropping NaNs")
    return df.astype(float)


def _check_block_size(block_size: int, n_obs: int) -> None:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if block_size > n_obs:
        raise ValueError("block_size cannot exceed the sample length")


def block_bootstrap(
    returns: ReturnsLike,
    *,
    block_size: int,
    n_samples: int,
    random_state: int | None = None,
) -> list[pd.DataFrame]:
    """Classic moving-block bootstrap preserving short-term dependence."""

    df = _to_frame(returns)
    n_obs = len(df)
    _check_block_size(block_size, n_obs)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    rng = np.random.default_rng(random_state)
    samples: list[pd.DataFrame] = []

    for _ in range(n_samples):
        indices: list[int] = []
        while len(indices) < n_obs:
            start = int(rng.integers(0, n_obs - block_size + 1))
            block = list(range(start, start + block_size))
            indices.extend(block)
        indices = indices[:n_obs]
        sample = df.iloc[indices].reset_index(drop=True)
        samples.append(sample)

    return samples


def stationary_bootstrap(
    returns: ReturnsLike,
    *,
    p: float,
    n_samples: int,
    random_state: int | None = None,
) -> list[pd.DataFrame]:
    """Stationary bootstrap with geometrically distributed block lengths."""

    if not 0 < p <= 1:
        raise ValueError("p must lie in (0, 1]")
    df = _to_frame(returns)
    n_obs = len(df)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    rng = np.random.default_rng(random_state)
    samples: list[pd.DataFrame] = []

    for _ in range(n_samples):
        indices: list[int] = []
        while len(indices) < n_obs:
            if not indices or rng.random() < p:
                start = int(rng.integers(0, n_obs))
            else:
                start = (indices[-1] + 1) % n_obs
            indices.append(start)
        indices = indices[:n_obs]
        sample = df.iloc[indices].reset_index(drop=True)
        samples.append(sample)

    return samples


def _iid_bootstrap(
    returns: pd.DataFrame,
    n_samples: int,
    rng: np.random.Generator,
) -> list[pd.DataFrame]:
    samples: list[pd.DataFrame] = []
    n_obs = len(returns)
    for _ in range(n_samples):
        indices = rng.integers(0, n_obs, size=n_obs)
        samples.append(returns.iloc[indices].reset_index(drop=True))
    return samples


def bootstrap_metric(
    metric_fn: Callable[[pd.DataFrame], MetricOutput],
    returns: ReturnsLike,
    *,
    n_samples: int,
    block_size: int | None = None,
    stationary_p: float | None = None,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Evaluate ``metric_fn`` across bootstrap samples.

    Returns a ``DataFrame`` where each row corresponds to one bootstrap draw and
    columns mirror the metric output (scalar → single column named ``value``).
    """

    df = _to_frame(returns)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    rng = np.random.default_rng(random_state)

    if stationary_p is not None:
        samples = stationary_bootstrap(
            df, p=stationary_p, n_samples=n_samples, random_state=random_state
        )
    elif block_size is not None:
        samples = block_bootstrap(
            df, block_size=block_size, n_samples=n_samples, random_state=random_state
        )
    else:
        samples = _iid_bootstrap(df, n_samples=n_samples, rng=rng)

    metric_rows = []
    columns: Sequence[str] | None = None

    for sample in samples:
        evaluated = metric_fn(sample)
        if isinstance(evaluated, pd.DataFrame):
            flattened = evaluated.stack(dropna=False)
            metric_rows.append(flattened)
            columns = flattened.index if columns is None else columns
        elif isinstance(evaluated, pd.Series):
            metric_rows.append(evaluated)
            columns = evaluated.index if columns is None else columns
        else:
            metric_rows.append(pd.Series(float(evaluated), index=["value"]))
            columns = ["value"]

    result = pd.DataFrame(metric_rows)
    if columns is not None:
        result = result.reindex(columns=columns)
    result.index.name = "bootstrap_id"
    return result


def confidence_interval(
    samples: Sequence[float] | pd.Series | np.ndarray,
    *,
    alpha: float = 0.05,
    method: str = "percentile",
) -> tuple[float, float]:
    """Compute bootstrap confidence intervals."""

    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    array = np.asarray(samples, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("No finite bootstrap samples available")

    lower_prob = alpha / 2.0
    upper_prob = 1.0 - lower_prob
    method = method.lower()

    if method == "percentile":
        lower = _quantile(array, lower_prob)
        upper = _quantile(array, upper_prob)
        return lower, upper
    if method == "basic":
        theta_hat = float(np.mean(array))
        lower = 2 * theta_hat - _quantile(array, upper_prob)
        upper = 2 * theta_hat - _quantile(array, lower_prob)
        return lower, upper
    raise ValueError("Unsupported confidence interval method")


def compare_vs_benchmark(
    metric_samples: Sequence[float] | pd.Series | np.ndarray | pd.DataFrame,
    benchmark_metric: float | pd.Series,
) -> float | pd.Series:
    """Probability that the sampled metric exceeds the benchmark."""

    if isinstance(metric_samples, pd.DataFrame):
        if isinstance(benchmark_metric, pd.Series):
            bench = benchmark_metric.reindex(metric_samples.columns).astype(float)
        else:
            bench = pd.Series(float(benchmark_metric), index=metric_samples.columns)
        probs = {
            column: float(compare_vs_benchmark(metric_samples[column], bench[column]))
            for column in metric_samples.columns
        }
        return pd.Series(probs, dtype=float)

    samples = np.asarray(metric_samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        raise ValueError("metric_samples contains no finite values")

    if isinstance(benchmark_metric, pd.Series):
        if benchmark_metric.size != 1:
            raise ValueError(
                "benchmark_metric series must contain a single value when comparing against 1D samples"
            )
        benchmark_value = float(benchmark_metric.iloc[0])
    else:
        benchmark_value = float(benchmark_metric)

    probability = float(np.mean(samples > benchmark_value))
    return probability


__all__ = [
    "block_bootstrap",
    "stationary_bootstrap",
    "bootstrap_metric",
    "confidence_interval",
    "compare_vs_benchmark",
]
