"""Utilities to generate and maintain rebalance schedules."""

from __future__ import annotations

from typing import Iterable, Mapping

import pandas as pd

__all__ = [
    "generate_schedule",
    "apply_overrides",
    "respect_trading_halts",
    "next_rebalance_date",
    "scheduler",
]


def _align_to_trading_days(
    tentative: Iterable[pd.Timestamp],
    trading_days: pd.DatetimeIndex,
) -> list[pd.Timestamp]:
    aligned: list[pd.Timestamp] = []
    trading_days = pd.DatetimeIndex(trading_days).sort_values().unique()
    for date in tentative:
        candidates = trading_days[trading_days >= date]
        if not candidates.empty:
            aligned.append(candidates[0])
    return aligned


def generate_schedule(
    trading_days: pd.DatetimeIndex,
    *,
    frequency: str = "monthly",
    anchor: str = "BMS",
) -> pd.DatetimeIndex:
    """Generate rebalance schedule anchored to available trading days."""

    if trading_days.empty:
        raise ValueError("trading_days must not be empty")
    trading_days = pd.DatetimeIndex(trading_days).sort_values().unique()
    start = trading_days[0]
    end = trading_days[-1]

    freq = frequency.lower()
    anchor = anchor.upper()

    if freq == "monthly":
        pandas_freq = "BMS" if anchor == "BMS" else "BM"
    elif freq == "weekly":
        pandas_freq = "W-FRI"
    elif freq == "quarterly":
        pandas_freq = "BQS" if anchor == "BMS" else "BQ"
    elif freq == "daily":
        return trading_days
    else:
        raise ValueError(f"Unsupported frequency '{frequency}'")

    tentative = pd.date_range(start=start, end=end, freq=pandas_freq)
    aligned = _align_to_trading_days(tentative, trading_days)
    return pd.DatetimeIndex(aligned).unique().sort_values()


def apply_overrides(
    schedule: pd.DatetimeIndex,
    overrides: Mapping[str, Iterable[pd.Timestamp]] | None,
) -> pd.DatetimeIndex:
    """Insert or remove specific dates from the schedule."""

    schedule = pd.DatetimeIndex(schedule).sort_values().unique()
    if not overrides:
        return schedule

    additions = [pd.Timestamp(d).normalize() for d in overrides.get("add", [])]
    removals = {pd.Timestamp(d).normalize() for d in overrides.get("remove", [])}

    combined = schedule.union(pd.DatetimeIndex(additions)).sort_values()
    filtered = [date for date in combined if date.normalize() not in removals]
    return pd.DatetimeIndex(filtered).unique()


def respect_trading_halts(
    schedule: pd.DatetimeIndex,
    trading_days: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    """Filter out dates that are not trading days."""

    trading_days = pd.DatetimeIndex(trading_days).normalize()
    schedule = pd.DatetimeIndex(schedule).normalize()
    return schedule.intersection(trading_days).sort_values()


def next_rebalance_date(
    current_date: pd.Timestamp,
    schedule: pd.DatetimeIndex,
) -> pd.Timestamp | None:
    """Return the next rebalance date on or after ``current_date``."""

    current_date = pd.Timestamp(current_date).normalize()
    future = schedule[schedule >= current_date]
    if future.empty:
        return None
    return future[0]


def scheduler(
    trading_days: pd.DatetimeIndex,
    config: Mapping[str, object] | None = None,
) -> pd.DatetimeIndex:
    """High-level helper that composes scheduling utilities."""

    config = dict(config or {})
    frequency = str(config.get("frequency", "monthly"))
    anchor = str(config.get("anchor", "BMS"))

    schedule = generate_schedule(trading_days, frequency=frequency, anchor=anchor)
    schedule = apply_overrides(schedule, config.get("overrides"))

    calendar = config.get("trading_calendar")
    if calendar is not None:
        schedule = respect_trading_halts(schedule, pd.DatetimeIndex(calendar))

    return schedule
