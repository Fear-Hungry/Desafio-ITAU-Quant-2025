#!/usr/bin/env python
"""Compute block-bootstrap confidence intervals for Sharpe ratios.

Uses the OOS returns produced by `run_baselines_comparison.py` to estimate
confidence intervals for selected strategies. By default, exponential moving
block bootstrap with block size 21 (one trading month) and 2,000 resamples.

Outputs:
    results/bootstrap_ci/bootstrap_sharpe_{timestamp}.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

STRATEGIES = ["1/N", "Risk Parity", "MV Robust (Shrunk20)", "Min-Var (LW)"]
BOOTSTRAP_ITERATIONS = 2000
BLOCK_SIZE = 21
CONFIDENCE = 0.95
OUTPUT_DIR = Path("results") / "bootstrap_ci"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def latest_oos_returns() -> Path:
    files = sorted(Path("results").glob("oos_returns_all_strategies_*.csv"))
    if not files:
        raise FileNotFoundError("No OOS return files found in results/")
    return files[-1]


def block_bootstrap_sharpe(series: pd.Series) -> tuple[float, float, float]:
    data = series.dropna().to_numpy(dtype=float)
    n = len(data)
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    mean = data.mean()
    std = data.std(ddof=1)
    sharpe = mean / std * np.sqrt(252.0) if std > 0 else 0.0

    rng = np.random.default_rng(1234)
    samples = np.empty(BOOTSTRAP_ITERATIONS, dtype=float)
    for i in range(BOOTSTRAP_ITERATIONS):
        idx = []
        while len(idx) < n:
            start = rng.integers(0, n)
            block = data[start : start + BLOCK_SIZE]
            idx.extend(block.tolist())
        resample = np.array(idx[:n], dtype=float)
        mu = resample.mean()
        sd = resample.std(ddof=1)
        samples[i] = mu / sd * np.sqrt(252.0) if sd > 0 else 0.0

    alpha = (1.0 - CONFIDENCE) / 2.0
    ci_low = float(np.quantile(samples, alpha))
    ci_high = float(np.quantile(samples, 1.0 - alpha))
    return float(sharpe), ci_low, ci_high


def main() -> None:
    oos_file = latest_oos_returns()
    returns = pd.read_csv(oos_file, index_col=0, parse_dates=True)

    results: dict[str, dict[str, float]] = {}
    for name in STRATEGIES:
        if name not in returns:
            continue
        sharpe, low, high = block_bootstrap_sharpe(returns[name])
        results[name] = {
            "sharpe_point": sharpe,
            "sharpe_ci_low": low,
            "sharpe_ci_high": high,
            "confidence": CONFIDENCE,
            "iterations": BOOTSTRAP_ITERATIONS,
            "block_size": BLOCK_SIZE,
        }

    timestamp = oos_file.stem.split("_")[-1]
    output_path = OUTPUT_DIR / f"bootstrap_sharpe_{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump({"source": oos_file.name, "results": results}, fh, indent=2)

    print(f"Bootstrap results saved to {output_path}")


if __name__ == "__main__":
    main()
