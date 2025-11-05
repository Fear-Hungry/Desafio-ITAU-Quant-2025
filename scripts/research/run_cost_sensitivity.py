#!/usr/bin/env python
"""Sensitivity of baseline strategies to transaction costs and turnover caps."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from arara_quant.evaluation.oos import (
    StrategySpec,
    compare_baselines,
    default_strategies,
)

UNIVERSE_RETURNS = Path("results/baselines/baseline_returns_oos.parquet")
COST_GRID = [30.0, 50.0, 75.0]
TURNOVER_CAP_GRID = [None, 0.25, 0.20]
OUTPUT_DIR = Path("results") / "cost_sensitivity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_returns() -> pd.DataFrame:
    if UNIVERSE_RETURNS.exists():
        return pd.read_parquet(UNIVERSE_RETURNS)
    raise FileNotFoundError(
        "Baseline returns parquet not found. Run `run_baselines_comparison.py` first."
    )


def adjust_strategies(turnover_cap: float | None) -> Iterable[StrategySpec]:
    strategies = default_strategies(max_position=0.10)
    tuned = []
    for spec in strategies:
        if spec.name != "shrunk_mv" or turnover_cap is None:
            tuned.append(spec)
            continue

        def builder(train_returns: pd.DataFrame, prev: pd.Series | None, cap=turnover_cap):
            strat = default_strategies(max_position=0.10)[4]  # shrunk_mv index
            weights = strat.builder(train_returns, prev)
            if prev is None:
                prev = pd.Series(0.0, index=weights.index)
            if cap is not None:
                delta = (weights - prev).abs()
                if delta.sum() > cap:
                    weights = prev + (weights - prev) * (cap / delta.sum())
            return weights

        tuned.append(StrategySpec(name="shrunk_mv_cap", builder=builder))
    return tuned


def main() -> None:
    returns = load_returns()
    all_metrics = []

    for cost in COST_GRID:
        for turnover_cap in TURNOVER_CAP_GRID:
            strategies = adjust_strategies(turnover_cap)
            oos = compare_baselines(
                returns,
                strategies=strategies,
                train_window=252,
                test_window=21,
                purge_window=5,
                embargo_window=5,
                costs_bps=cost,
                max_position=0.10,
                bootstrap_iterations=1000,
                confidence=0.90,
                block_size=21,
            )
            metrics = oos.metrics.copy()
            metrics["costs_bps"] = cost
            metrics["turnover_cap"] = turnover_cap if turnover_cap is not None else "none"
            all_metrics.append(metrics)

    combined = pd.concat(all_metrics, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "cost_turnover_sensitivity.csv", index=False)
    print("Saved metrics to:", OUTPUT_DIR / "cost_turnover_sensitivity.csv")


if __name__ == "__main__":
    main()
