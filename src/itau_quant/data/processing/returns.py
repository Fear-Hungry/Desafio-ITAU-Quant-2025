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
    """Calcula excess returns alinhando r_f a dias úteis e index de returns.

    - Converte rf para business-days (B) e faz forward-fill
    - Reindexa para o índice de returns e subtrai por broadcast
    """
    if rf_daily is None or len(rf_daily) == 0:
        return returns.copy()
    rf_b = rf_daily.asfreq("B").ffill()
    rf_aligned = rf_b.reindex(returns.index).ffill()
    # broadcast por linha
    xr = returns.sub(rf_aligned, axis=0)
    return xr
