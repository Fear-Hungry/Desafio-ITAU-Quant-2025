"""Execution utilities for the backtesting loop."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["ExecutionResult", "simulate_execution"]


@dataclass(frozen=True)
class ExecutionResult:
    weights: pd.Series
    turnover: float
    cost: float


def simulate_execution(
    previous_weights: pd.Series,
    target_weights: pd.Series,
    *,
    linear_cost_bps: float = 0.0,
) -> ExecutionResult:
    """Return realised turnover and linear costs between two weight vectors."""

    prev = previous_weights.reindex(target_weights.index, fill_value=0.0).astype(float)
    target = target_weights.reindex(prev.index, fill_value=0.0).astype(float)

    turnover = float(np.abs(target - prev).sum())
    cost = turnover * float(linear_cost_bps) / 10_000.0
    return ExecutionResult(weights=target, turnover=turnover, cost=cost)
