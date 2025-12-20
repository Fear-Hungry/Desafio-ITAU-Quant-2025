#!/usr/bin/env python
"""Compute block-bootstrap confidence intervals for Sharpe ratios.

Uses the OOS returns produced by `scripts/research/run_baselines_comparison.py` to estimate
confidence intervals for selected strategies. By default, exponential moving
block bootstrap with block size 21 (one trading month) and 2,000 resamples.

Outputs:
    outputs/results/bootstrap_ci/bootstrap_sharpe_{timestamp}.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from arara_quant.config import get_settings

SETTINGS = get_settings()

# Mapping from friendly names (used in reports) to columns in the OOS
# returns dataframe.
STRATEGIES = {
    "1/N": "equal_weight",
    "Risk Parity": "risk_parity",
    "MV Robust (shrunk)": "shrunk_mv",
    "Min-Var (LW)": "min_variance_lw",
    "60/40": "sixty_forty",
}
BOOTSTRAP_ITERATIONS = 2000
BLOCK_SIZE = 21
CONFIDENCE = 0.95
OUTPUT_DIR = SETTINGS.results_dir / "bootstrap_ci"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def latest_oos_returns() -> Path:
    baseline_path = SETTINGS.results_dir / "baselines" / "baseline_returns_oos.parquet"
    if baseline_path.exists():
        return baseline_path

    legacy = sorted(SETTINGS.results_dir.glob("oos_returns_all_strategies_*.csv"))
    if legacy:
        return legacy[-1]
    raise FileNotFoundError(
        "Nenhum arquivo de retornos OOS encontrado. Rode "
        "`scripts/research/run_baselines_comparison.py` primeiro."
    )


def block_bootstrap_sharpe(series: pd.Series) -> tuple[float, float, float]:
    data = series.dropna().to_numpy(dtype=float)
    n = len(data)
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    mean = data.mean()
    std = data.std(ddof=1)
    sharpe = mean / std * np.sqrt(252.0) if std > 0 else 0.0

    extended = np.concatenate([data, data[: BLOCK_SIZE - 1]]) if n > 0 else data
    rng = np.random.default_rng(1234)
    samples = np.empty(BOOTSTRAP_ITERATIONS, dtype=float)
    for i in range(BOOTSTRAP_ITERATIONS):
        drawn: list[float] = []
        while len(drawn) < n:
            start = rng.integers(0, n)
            block = extended[start : start + BLOCK_SIZE]
            drawn.extend(block.tolist())
        resample = np.array(drawn[:n], dtype=float)
        mu = resample.mean()
        sd = resample.std(ddof=1)
        samples[i] = mu / sd * np.sqrt(252.0) if sd > 0 else 0.0

    alpha = (1.0 - CONFIDENCE) / 2.0
    ci_low = float(np.quantile(samples, alpha))
    ci_high = float(np.quantile(samples, 1.0 - alpha))
    return float(sharpe), ci_low, ci_high


def main() -> None:
    oos_file = latest_oos_returns()
    if oos_file.suffix == ".parquet":
        returns = pd.read_parquet(oos_file)
    else:
        returns = pd.read_csv(oos_file, index_col=0, parse_dates=True)

    results: dict[str, dict[str, float]] = {}
    for label, column in STRATEGIES.items():
        if column not in returns:
            continue
        sharpe, low, high = block_bootstrap_sharpe(returns[column])
        results[label] = {
            "sharpe_point": sharpe,
            "sharpe_ci_low": low,
            "sharpe_ci_high": high,
            "confidence": CONFIDENCE,
            "iterations": BOOTSTRAP_ITERATIONS,
            "block_size": BLOCK_SIZE,
        }

    timestamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"bootstrap_sharpe_{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump({"source": oos_file.name, "results": results}, fh, indent=2)

    print(f"Bootstrap results saved to {output_path}")


if __name__ == "__main__":
    main()
