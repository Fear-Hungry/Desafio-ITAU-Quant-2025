from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_oos_figures import (
    filter_to_oos_period,
    load_nav_daily,
    load_oos_config,
)


def test_oos_figure_filters_align_with_config_and_metrics() -> None:
    """Ensure figure helpers honor the OOS config and match consolidated metrics."""

    oos_config = load_oos_config()
    df_nav = load_nav_daily()
    df_oos = filter_to_oos_period(df_nav, oos_config)

    assert not df_oos.empty, "Filtered OOS dataframe should not be empty"

    start_date = pd.to_datetime(oos_config["start_date"])
    end_date = pd.to_datetime(oos_config["end_date"])

    assert df_oos["date"].iloc[0] == start_date
    assert df_oos["date"].iloc[-1] == end_date

    metrics_path = Path("reports/oos_consolidated_metrics.json")
    with metrics_path.open(encoding="utf-8") as handle:
        metrics = json.load(handle)

    nav_final = float(df_oos["nav"].iloc[-1])
    total_return = nav_final - 1.0
    n_days = len(df_oos)
    daily_returns = df_oos["daily_return"].to_numpy(dtype=float)

    annualized_return = (nav_final ** (252 / n_days)) - 1.0
    annualized_volatility = float(np.std(daily_returns, ddof=1) * np.sqrt(252))

    assert abs(nav_final - metrics["nav_final"]) < 1e-12
    assert abs(total_return - metrics["total_return"]) < 1e-12
    assert abs(annualized_return - metrics["annualized_return"]) < 1e-9
    assert abs(annualized_volatility - metrics["annualized_volatility"]) < 1e-9
    assert n_days == int(metrics["n_days"])
