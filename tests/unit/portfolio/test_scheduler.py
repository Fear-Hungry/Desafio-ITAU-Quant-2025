from __future__ import annotations

import pandas as pd
from itau_quant.portfolio.scheduler import (
    generate_schedule,
    next_rebalance_date,
    scheduler,
)


def test_generate_schedule_monthly_bms():
    dates = pd.bdate_range("2024-01-01", "2024-03-31")
    schedule = generate_schedule(dates, frequency="monthly", anchor="BMS")
    assert list(schedule) == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-02-01"),
        pd.Timestamp("2024-03-01"),
    ]


def test_scheduler_with_overrides_and_halts():
    dates = pd.bdate_range("2024-01-01", "2024-02-28")
    overrides = {
        "add": [pd.Timestamp("2024-01-15"), pd.Timestamp("2024-02-02")],
        "remove": [pd.Timestamp("2024-02-01")],
    }
    calendar = dates.drop(pd.Timestamp("2024-01-15"))
    schedule = scheduler(
        dates,
        {
            "frequency": "monthly",
            "anchor": "BMS",
            "overrides": overrides,
            "trading_calendar": calendar,
        },
    )
    assert pd.Timestamp("2024-01-15") not in schedule  # removed by trading halt
    assert pd.Timestamp("2024-02-02") in schedule


def test_next_rebalance_date():
    schedule = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-31")]
    )
    assert next_rebalance_date(pd.Timestamp("2024-01-15"), schedule) == pd.Timestamp(
        "2024-01-31"
    )
    assert next_rebalance_date(pd.Timestamp("2024-02-01"), schedule) is None
