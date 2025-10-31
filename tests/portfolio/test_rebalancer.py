from __future__ import annotations

import numpy as np
import pandas as pd

from itau_quant.portfolio import MarketData, rebalance


def test_rebalance_pipeline_returns_weights():
    dates = pd.bdate_range("2023-01-01", periods=80)
    returns = pd.DataFrame(
        {
            "AAA": np.linspace(0.0002, 0.0008, len(dates)),
            "BBB": np.full(len(dates), 0.0005),
            "CCC": np.linspace(-0.0001, 0.0004, len(dates)),
        },
        index=dates,
    )
    prices = pd.DataFrame(
        {
            "AAA": np.full(len(dates), 100.0),
            "BBB": np.full(len(dates), 50.0),
            "CCC": np.full(len(dates), 25.0),
        },
        index=dates,
    )
    market = MarketData(prices=prices, returns=returns)

    previous = pd.Series({"AAA": 0.3, "BBB": 0.4, "CCC": 0.3})
    config = {
        "optimizer": {
            "risk_aversion": 3.0,
            "turnover_penalty": 0.05,
            "turnover_cap": 0.3,
            "min_weight": 0.0,
            "max_weight": 0.6,
        },
        "rounding": {"lot_sizes": 1, "method": "nearest", "cost_model": {"linear_bps": 5}},
        "costs": {"linear_bps": 8},
        "returns_window": 60,
    }

    result = rebalance(
        date=dates[-1],
        market_data=market,
        previous_weights=previous,
        capital=1_000_000.0,
        config=config,
    )

    assert isinstance(result.weights, pd.Series)
    assert abs(result.rounded_weights.sum() + result.cash / 1_000_000.0 - 1.0) < 0.02
    assert result.metrics.optimizer_turnover >= 0.0
    assert result.trades.abs().sum() >= 0.0
    assert "solver" in result.log


def test_rebalance_applies_regime_detection():
    dates = pd.bdate_range("2022-01-03", periods=90)
    rng = np.random.default_rng(42)
    returns = pd.DataFrame(
        rng.normal(loc=0.0002, scale=0.02, size=(len(dates), 3)),
        index=dates,
        columns=["AAA", "BBB", "CCC"],
    )
    prices = pd.DataFrame(
        {
            "AAA": np.full(len(dates), 100.0),
            "BBB": np.full(len(dates), 55.0),
            "CCC": np.full(len(dates), 45.0),
        },
        index=dates,
    )
    market = MarketData(prices=prices, returns=returns)
    previous = pd.Series({"AAA": 0.0, "BBB": 0.0, "CCC": 0.0})

    config = {
        "optimizer": {
            "risk_aversion": 3.0,
            "turnover_penalty": 0.0,
            "turnover_cap": 0.20,
            "min_weight": 0.0,
            "max_weight": 0.7,
            "regime_detection": {
                "window_days": 45,
                "vol_thresholds": {"calm": 0.10, "stressed": 0.18},
                "drawdown_crash": -0.20,
                "multipliers": {"stressed": 1.6, "default": 1.0},
            },
        },
        "rounding": {"lot_sizes": 1},
        "costs": {"linear_bps": 0.0},
        "returns_window": 60,
    }

    result = rebalance(
        date=dates[-1],
        market_data=market,
        previous_weights=previous,
        capital=1_000_000.0,
        config=config,
    )

    assert "regime_state" in result.log
    assert result.log["regime_state"]["label"] in {"stressed", "crash"}
    assert result.log.get("lambda_adjusted", 0.0) >= config["optimizer"]["risk_aversion"]
