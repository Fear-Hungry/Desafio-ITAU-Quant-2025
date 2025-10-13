"""Funções de processamento de retornos.

Cálculo de retornos log/percentual e helpers relacionados.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_returns(prices_df: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    prices = prices_df.sort_index()
    if method == "log":
        ratio = prices.divide(prices.shift(1))
        ratio = ratio.where(ratio > 0)
        rets = np.log(ratio)
    else:
        rets = prices.pct_change()
    return rets.dropna(how="all")


def compute_excess_returns(returns: pd.DataFrame, rf_daily: pd.Series) -> pd.DataFrame:
    """Alinha e subtrai r_f diário dos retornos.

    r_f é broadcast por índice de datas.
    """
    rf = rf_daily.reindex(returns.index).fillna(method="ffill")
    return returns.sub(rf, axis=0)
