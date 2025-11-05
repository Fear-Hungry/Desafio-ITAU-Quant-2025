"""
Utilities for classifying market regimes and adjusting portfolio parameters.

The goal is to provide a lightweight helper that looks at recent portfolio
returns, estimates volatility and drawdown, and maps the current environment to
labels such as ``calm``, ``stressed`` or ``crash``. The caller can then use the
label to scale risk aversion or apply other guardrails.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

__all__ = [
    "RegimeSnapshot",
    "detect_regime",
    "regime_multiplier",
]


@dataclass(frozen=True)
class RegimeSnapshot:
    """Summary of the identified market regime."""

    label: str
    volatility: float
    drawdown: float
    window: int

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "label": self.label,
            "volatility": float(self.volatility),
            "drawdown": float(self.drawdown),
            "window": int(self.window),
        }


def _equal_weight_returns(returns: pd.DataFrame) -> pd.Series:
    """Compute equal-weight portfolio returns for the given cross-section."""

    valid = returns.dropna(how="all")
    if valid.empty:
        return pd.Series(dtype=float)
    weights = pd.Series(1.0 / valid.shape[1], index=valid.columns, dtype=float)
    return valid.mul(weights, axis=1).sum(axis=1)


def detect_regime(
    returns: pd.DataFrame,
    *,
    config: Mapping[str, object] | None = None,
) -> RegimeSnapshot:
    """Detect the prevailing market regime based on recent returns.

    Parameters
    ----------
    returns:
        Historical asset returns (daily). The function uses an equal-weight
        aggregate as a simple proxy for the portfolio.
    config:
        Optional mapping with the following keys:
        - ``window_days`` (int, default 63): rolling window used for statistics.
        - ``vol_thresholds`` (mapping): expected annualised volatility for
          ``calm`` and ``stressed`` regimes (defaults: calm=0.12, stressed=0.25).
        - ``drawdown_crash`` (float, default -0.15): drawdown threshold to flag
          a crash regime.

    Returns
    -------
    RegimeSnapshot
        The detected regime label alongside realised volatility and drawdown.
    """

    settings = dict(config or {})
    window = int(settings.get("window_days", 63))
    eq_returns = _equal_weight_returns(returns)
    if eq_returns.empty:
        return RegimeSnapshot("neutral", 0.0, 0.0, window)

    window_returns = eq_returns.tail(window if window > 0 else len(eq_returns))
    if window_returns.empty:
        window_returns = eq_returns

    if len(window_returns) < 2:
        volatility = float(window_returns.std(ddof=0) * np.sqrt(252.0))
    else:
        volatility = float(window_returns.std(ddof=1) * np.sqrt(252.0))

    cumulative = (1.0 + window_returns).cumprod()
    if cumulative.empty:
        drawdown = 0.0
    else:
        peak = cumulative.cummax()
        drawdown = float((cumulative / peak - 1.0).iloc[-1])

    thresholds = settings.get("vol_thresholds", {}) or {}
    calm_threshold = float(thresholds.get("calm", 0.12))
    stressed_threshold = float(thresholds.get("stressed", 0.25))
    crash_threshold = float(settings.get("drawdown_crash", -0.15))

    if drawdown <= crash_threshold:
        label = "crash"
    elif volatility >= stressed_threshold:
        label = "stressed"
    elif volatility <= calm_threshold:
        label = "calm"
    else:
        label = "neutral"

    return RegimeSnapshot(
        label=label, volatility=volatility, drawdown=drawdown, window=window
    )


def regime_multiplier(
    snapshot: RegimeSnapshot,
    config: Mapping[str, object] | None = None,
) -> float:
    """Return the risk-aversion multiplier associated with ``snapshot``.

    If ``config`` contains a ``multipliers`` mapping the function will look up
    the entry for ``snapshot.label``. The optional ``default`` key acts as
    fallback. When no mapping is supplied the function returns ``1.0``.
    """

    settings = dict(config or {})
    mapping = settings.get("multipliers", {}) or {}
    if isinstance(mapping, Mapping):
        if snapshot.label in mapping:
            return float(mapping[snapshot.label])
        if "default" in mapping:
            return float(mapping["default"])
    return 1.0
