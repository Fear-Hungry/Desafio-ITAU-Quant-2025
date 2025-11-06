from __future__ import annotations

import pandas as pd
from arara_quant.backtesting.execution import simulate_execution


def test_simulate_execution_computes_turnover_and_cost() -> None:
    prev = pd.Series({"BBB": 0.4, "AAA": 0.6})
    target = pd.Series({"AAA": 0.2, "BBB": 0.5})

    result = simulate_execution(prev, target, linear_cost_bps=25)

    assert result.weights.loc["AAA"] == 0.2
    expected_turnover_two_way = abs(0.2 - 0.6) + abs(0.5 - 0.4)
    expected_turnover = 0.5 * expected_turnover_two_way
    assert result.turnover == expected_turnover
    assert result.cost == expected_turnover * 25 / 10_000
