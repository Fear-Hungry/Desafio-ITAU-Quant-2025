"""Lightweight bookkeeping helpers used by the backtest engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

__all__ = ["PortfolioLedger", "build_ledger"]


@dataclass(frozen=True)
class PortfolioLedger:
    """Container for the main timeseries generated during a simulation."""

    frame: pd.DataFrame

    def as_dict(self) -> Mapping[str, list[float]]:
        return {column: self.frame[column].tolist() for column in self.frame.columns}


def build_ledger(
    *,
    dates: pd.Index,
    gross_returns: pd.Series,
    net_returns: pd.Series,
    costs: pd.Series,
    turnover: pd.Series,
) -> PortfolioLedger:
    """Aggregate core timeseries for reporting."""

    net_returns = net_returns.reindex(dates, fill_value=0.0)
    gross_returns = gross_returns.reindex(dates, fill_value=0.0)
    costs = costs.reindex(dates, fill_value=0.0)
    turnover = turnover.reindex(dates, fill_value=0.0)

    nav = (1.0 + net_returns.fillna(0.0)).cumprod()
    ledger = pd.DataFrame(
        {
            "nav": nav,
            "gross_return": gross_returns.fillna(0.0),
            "net_return": net_returns.fillna(0.0),
            "costs": costs.fillna(0.0),
            "turnover": turnover.fillna(0.0),
        },
        index=dates,
    )
    ledger.index.name = "date"
    return PortfolioLedger(frame=ledger)
