from __future__ import annotations

import pandas as pd
from itau_quant.portfolio.triggers import (
    TriggerEvent,
    cooldown_manager,
    drawdown_trigger,
    trigger_engine,
    volatility_trigger,
)


def test_drawdown_trigger_detects_drop():
    nav = pd.Series([100, 105, 95, 90], index=pd.date_range("2024-01-01", periods=4))
    event = drawdown_trigger(nav, threshold=-0.1)
    assert isinstance(event, TriggerEvent)
    assert event.trigger == "drawdown"


def test_volatility_trigger_comparison():
    returns = pd.Series(
        [0.001, 0.002, -0.001, 0.003, 0.004, -0.005, 0.006],
        index=pd.date_range("2024-01-01", periods=7),
    )
    event = volatility_trigger(returns, window=3, multiplier=1.2)
    assert isinstance(event, TriggerEvent)
    assert event.trigger == "volatility"


def test_trigger_engine_with_cooldown():
    nav = pd.Series([100, 110, 90], index=pd.date_range("2024-01-01", periods=3))
    returns = nav.pct_change().dropna()
    config = {
        "drawdown": {"threshold": -0.05},
        "cooldown": {"days": 5},
    }
    state = {"last_trigger_date": None}
    data = {"nav": nav, "returns": returns, "date": nav.index[-1]}

    events = trigger_engine(state, data, config)
    assert events and events[0].trigger == "drawdown"

    state = {"last_trigger_date": nav.index[-1]}
    events = trigger_engine(state, data, config)
    assert events == []


def test_cooldown_manager_with_timedelta():
    last_date = pd.Timestamp("2024-01-01")
    current = pd.Timestamp("2024-01-03")
    assert cooldown_manager(last_date, pd.Timedelta(days=1), current) is True
    assert cooldown_manager(last_date, pd.Timedelta(days=5), current) is False
