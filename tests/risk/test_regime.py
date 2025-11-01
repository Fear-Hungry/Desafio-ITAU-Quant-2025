from __future__ import annotations

import numpy as np
import pandas as pd
from itau_quant.risk.regime import RegimeSnapshot, detect_regime, regime_multiplier


def _make_returns(level: float, noise: float, periods: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    data = rng.normal(loc=level, scale=noise, size=(periods, 3))
    dates = pd.bdate_range("2022-01-03", periods=periods)
    return pd.DataFrame(data, index=dates, columns=["AAA", "BBB", "CCC"])


def test_detect_regime_identifies_calm_environment() -> None:
    returns = _make_returns(0.0003, 0.002, periods=80)
    snapshot = detect_regime(returns, config={"window_days": 63})
    assert isinstance(snapshot, RegimeSnapshot)
    assert snapshot.label in {"calm", "neutral"}
    assert snapshot.window == 63
    assert snapshot.volatility >= 0.0


def test_detect_regime_flags_stressed_and_crash() -> None:
    stressed = _make_returns(0.0, 0.02, periods=80)
    snap_stressed = detect_regime(
        stressed,
        config={"window_days": 40, "vol_thresholds": {"calm": 0.10, "stressed": 0.20}},
    )
    assert snap_stressed.label == "stressed"

    crash_returns = pd.DataFrame(
        {
            "AAA": np.linspace(0.0, -0.05, 50),
            "BBB": np.linspace(0.0, -0.04, 50),
        },
        index=pd.bdate_range("2023-01-02", periods=50),
    )
    snap_crash = detect_regime(crash_returns, config={"drawdown_crash": -0.05})
    assert snap_crash.label == "crash"
    assert snap_crash.drawdown <= -0.05


def test_regime_multiplier_uses_mapping() -> None:
    snapshot = RegimeSnapshot(
        label="stressed", volatility=0.3, drawdown=-0.1, window=63
    )
    multiplier = regime_multiplier(
        snapshot,
        config={"multipliers": {"stressed": 1.5, "default": 1.0}},
    )
    assert multiplier == 1.5

    fallback = regime_multiplier(snapshot, config=None)
    assert fallback == 1.0
