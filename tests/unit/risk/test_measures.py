from __future__ import annotations

import pandas as pd

from itau_quant.risk.measures import (
    historical_cvar,
    information_ratio,
    max_drawdown,
    sharpe_ratio,
    volatility,
)


def test_volatility_and_sharpe_ratio():
    returns = pd.Series([0.01, -0.005, 0.002])
    vol = volatility(returns, periods_per_year=1)
    sr = sharpe_ratio(returns, rf=0.0, periods_per_year=1)
    assert vol > 0
    assert isinstance(sr, float)


def test_max_drawdown_and_cvar():
    nav = pd.Series([100, 105, 90, 95])
    drawdown, series = max_drawdown(nav)
    assert drawdown <= 0
    cvar = historical_cvar(pd.Series([0.01, -0.02, 0.03]), alpha=0.9)
    assert cvar > 0


def test_information_ratio_zero_when_identical():
    strat = pd.Series([0.01, 0.02, -0.01])
    bench = strat.copy()
    ir = information_ratio(strat, bench)
    assert pd.isna(ir)
