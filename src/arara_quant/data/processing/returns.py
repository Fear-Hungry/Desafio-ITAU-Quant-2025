"""Transformações relacionadas a retornos.

`calculate_returns(prices_df, method="log")`
    - Ordena o índice temporal.
    - Calcula retornos log (default) via `np.log(price_t / price_{t-1})` ou
      percentuais (`pct_change`).
    - Descarta linhas iniciais totalmente nulas após o shift.

`compute_excess_returns(returns, rf_daily)`
    - Reindexa o risk-free diário para as mesmas datas do painel.
    - Aplica forward-fill para lacunas e subtrai linha a linha.
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
    rf = rf_daily.reindex(returns.index).ffill()
    return returns.sub(rf, axis=0)
