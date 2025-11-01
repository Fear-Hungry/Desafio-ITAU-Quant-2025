"""Rebalance trigger helpers for extraordinary events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

__all__ = [
    "TriggerEvent",
    "drawdown_trigger",
    "cvar_trigger",
    "volatility_trigger",
    "signal_change_trigger",
    "trigger_engine",
    "cooldown_manager",
]


@dataclass(frozen=True)
class TriggerEvent:
    trigger: str
    value: float
    threshold: float
    details: Mapping[str, Any]


def drawdown_trigger(nav: pd.Series, *, threshold: float) -> TriggerEvent | None:
    """Trigger when the running drawdown breaches the threshold."""

    nav = nav.dropna()
    if nav.empty:
        return None
    rolling_max = nav.cummax()
    drawdown = nav / rolling_max - 1.0
    current = float(drawdown.iloc[-1])
    if current <= float(threshold):
        return TriggerEvent(
            trigger="drawdown",
            value=current,
            threshold=float(threshold),
            details={"drawdown_series": drawdown},
        )
    return None


def cvar_trigger(
    returns: pd.Series,
    *,
    window: int,
    alpha: float,
    limit: float,
) -> TriggerEvent | None:
    """Trigger when rolling CVaR exceeds the configured limit."""

    if window <= 0 or not (0 < alpha < 1):
        raise ValueError("window must be >0 and alpha in (0,1)")
    tail = returns.dropna().tail(window)
    if tail.empty:
        return None
    losses = -tail.sort_values()
    cutoff = int(np.ceil((1 - alpha) * len(losses)))
    cutoff = max(cutoff, 1)
    cvar = float(losses.head(cutoff).mean())
    if cvar >= float(limit):
        return TriggerEvent(
            trigger="cvar",
            value=cvar,
            threshold=float(limit),
            details={"window": window, "alpha": alpha},
        )
    return None


def volatility_trigger(
    returns: pd.Series,
    *,
    window: int,
    multiplier: float,
) -> TriggerEvent | None:
    """Trigger when rolling volatility spikes relative to history."""

    if window <= 1:
        raise ValueError("window must be greater than 1")
    history = returns.dropna()
    if len(history) < window:
        return None
    recent = history.tail(window)
    base_vol = history.std(ddof=0)
    recent_vol = recent.std(ddof=0)
    if base_vol == 0:
        return None
    ratio = float(recent_vol / base_vol)
    if ratio >= float(multiplier):
        return TriggerEvent(
            trigger="volatility",
            value=ratio,
            threshold=float(multiplier),
            details={"recent_vol": recent_vol, "base_vol": base_vol},
        )
    return None


def signal_change_trigger(
    signals: pd.Series,
    *,
    threshold: float,
) -> TriggerEvent | None:
    """Trigger when the latest signal changes abruptly."""

    signals = signals.dropna()
    if len(signals) < 2:
        return None
    delta = float(signals.iloc[-1] - signals.iloc[-2])
    if abs(delta) >= abs(threshold):
        return TriggerEvent(
            trigger="signal_change",
            value=delta,
            threshold=float(threshold),
            details={
                "previous": float(signals.iloc[-2]),
                "current": float(signals.iloc[-1]),
            },
        )
    return None


def cooldown_manager(
    last_trigger_date: pd.Timestamp | None,
    cooldown_period: int | pd.Timedelta,
    current_date: pd.Timestamp,
) -> bool:
    """Return ``True`` when new triggers are allowed."""

    if last_trigger_date is None:
        return True
    current_date = pd.Timestamp(current_date)
    last_trigger_date = pd.Timestamp(last_trigger_date)

    if isinstance(cooldown_period, pd.Timedelta):
        return current_date >= last_trigger_date + cooldown_period
    days = int(cooldown_period)
    return current_date >= last_trigger_date + pd.Timedelta(days=days)


def trigger_engine(
    state: Mapping[str, Any],
    data: Mapping[str, Any],
    config: Mapping[str, Mapping[str, Any]],
) -> list[TriggerEvent]:
    """Evaluate configured triggers and return the list of events."""

    events: list[TriggerEvent] = []
    last_date = state.get("last_trigger_date")
    cooldown = config.get("cooldown", {})
    cooldown_period = cooldown.get("days", 0)
    current_date = pd.Timestamp(data.get("date", pd.Timestamp.today()))

    if cooldown_period and not cooldown_manager(
        last_date, cooldown_period, current_date
    ):
        return events

    if "drawdown" in config:
        nav = pd.Series(data.get("nav", []))
        event = drawdown_trigger(
            nav, threshold=float(config["drawdown"].get("threshold", -0.05))
        )
        if event:
            events.append(event)

    if "cvar" in config:
        returns = pd.Series(data.get("returns", []))
        cfg = config["cvar"]
        event = cvar_trigger(
            returns,
            window=int(cfg.get("window", 20)),
            alpha=float(cfg.get("alpha", 0.95)),
            limit=float(cfg.get("limit", 0.05)),
        )
        if event:
            events.append(event)

    if "volatility" in config:
        returns = pd.Series(data.get("returns", []))
        cfg = config["volatility"]
        event = volatility_trigger(
            returns,
            window=int(cfg.get("window", 20)),
            multiplier=float(cfg.get("multiplier", 2.0)),
        )
        if event:
            events.append(event)

    if "signal_change" in config:
        signals = pd.Series(data.get("signals", []))
        event = signal_change_trigger(
            signals,
            threshold=float(config["signal_change"].get("threshold", 0.1)),
        )
        if event:
            events.append(event)

    return events
