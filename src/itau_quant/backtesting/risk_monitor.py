"""Risk utilities used during the backtest."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd

TurnoverBandStatus = Literal["below", "within", "above"]

__all__ = [
    "TurnoverSnapshot",
    "apply_turnover_cap",
    "evaluate_turnover_band",
]


@dataclass(frozen=True)
class TurnoverSnapshot:
    date: pd.Timestamp
    turnover: float
    status: TurnoverBandStatus


def apply_turnover_cap(
    previous_weights: pd.Series,
    target_weights: pd.Series,
    *,
    max_turnover: float | None,
    tol: float = 1e-9,
) -> tuple[pd.Series, float]:
    """Scale trades so that ``L1`` turnover does not exceed ``max_turnover``."""

    prev = previous_weights.reindex(target_weights.index, fill_value=0.0).astype(float)
    target = target_weights.reindex(prev.index, fill_value=0.0).astype(float)

    diff = target - prev
    turnover = float(np.abs(diff).sum())

    if max_turnover is None or max_turnover <= 0 or turnover <= max_turnover + tol:
        return target, turnover

    if turnover == 0:
        return target, 0.0

    scale = float(max_turnover / turnover)
    adjusted = prev + diff * scale
    turnover_adjusted = float(np.abs(adjusted - prev).sum())
    return adjusted, turnover_adjusted


def evaluate_turnover_band(turnover: float, band: Iterable[float] | None) -> TurnoverBandStatus:
    """Classify turnover relative to a target band (low/within/high)."""

    if band is None:
        return "within"
    values = list(band)
    if len(values) != 2:
        raise ValueError("turnover band must have exactly two values")
    lower, upper = sorted(float(x) for x in values)
    if turnover < lower:
        return "below"
    if turnover > upper:
        return "above"
    return "within"
