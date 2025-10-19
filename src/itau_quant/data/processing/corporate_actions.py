# Placeholder implementation notice for future contributors:
# - Escopo: encapsular ajustes retroativos (splits, dividendos em dinheiro,
#   spin-offs, fusões) garantindo continuidade das séries.
# - API mínima proposta:
#     * `load_corporate_actions(tickers, start=None, end=None, source="tiingo")`
#           - Retorna DataFrame estruturado: columns `event_type`, `ex_date`,
#             `effective_date`, `ratio`, `cash_amount`, `ticker`.
#     * `calculate_adjustment_factors(actions_df, index)`
#           - Constrói fatores cumulativos para preço e volume alinhados ao índice
#             do painel principal (usa `ensure_dtindex`).
#     * `apply_price_adjustments(prices, factors)`
#           - Aplica fatores multiplicativos nos preços históricos; idem para
#             volumes caso disponíveis.
#     * `apply_return_adjustments(returns, dividends)`
#           - Ajusta retornos simples quando somente ``Close`` está disponível.
# - Integração com `DataLoader`:
#     * Rodar logo após `normalize_index` e antes de `filter_liquid_assets` para
#       não impactar métricas de liquidez.
#     * Persistir fatores em `data/processed/corporate_actions/` usando
#       `storage.save_parquet` e hash de requisição para auditoria.
# - Considerações técnicas:
#     * Suportar merges de múltiplos eventos no mesmo dia (ex.: dois splits).
#     * Dividends podem ser em cash ou percentuais; alinhar com timezone tz-naive.
#     * Permitir fallback para arquivos CSV locais quando APIs não disponíveis.
# - Testes recomendados (`tests/data/processing/test_corporate_actions.py` futuro):
#     * Split 2:1 seguido de dividendo extraordinário.
#     * Spin-off que requer fator proporcional (ex.: ticker original perde 20%).
#     * Eventos inexistentes para parte dos tickers (fatores = 1).

"""Corporate action processing utilities (splits, dividends, spin-offs)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from itau_quant.utils.data_loading import to_datetime_index
from itau_quant.utils.logging_config import get_logger

logger = get_logger(__name__)

__all__ = [
    "CorporateAction",
    "load_corporate_actions",
    "calculate_adjustment_factors",
    "apply_price_adjustments",
    "apply_return_adjustments",
]


@dataclass(frozen=True)
class CorporateAction:
    ticker: str
    event_type: str
    ex_date: pd.Timestamp
    effective_date: pd.Timestamp | None
    ratio: float | None = None
    cash_amount: float | None = None

    @staticmethod
    def from_mapping(payload: Mapping[str, object]) -> "CorporateAction":
        return CorporateAction(
            ticker=str(payload["ticker"]),
            event_type=str(payload["event_type"]).lower(),
            ex_date=pd.Timestamp(payload["ex_date"]),
            effective_date=pd.Timestamp(payload["effective_date"]) if payload.get("effective_date") else None,
            ratio=float(payload["ratio"]) if payload.get("ratio") is not None else None,
            cash_amount=float(payload["cash_amount"]) if payload.get("cash_amount") is not None else None,
        )


def load_corporate_actions(
    tickers: Sequence[str],
    *,
    actions: Iterable[Mapping[str, object]] | None = None,
) -> pd.DataFrame:
    """Load corporate action data into a normalized DataFrame."""

    if actions is None:
        logger.info("No corporate actions provided; returning empty frame.")
        return pd.DataFrame(columns=["ticker", "event_type", "ex_date", "effective_date", "ratio", "cash_amount"])

    records = []
    for entry in actions:
        action = CorporateAction.from_mapping(entry)
        if tickers and action.ticker not in tickers:
            continue
        records.append(
            {
                "ticker": action.ticker,
                "event_type": action.event_type,
                "ex_date": action.ex_date.normalize(),
                "effective_date": action.effective_date.normalize() if action.effective_date is not None else None,
                "ratio": action.ratio,
                "cash_amount": action.cash_amount,
            }
        )

    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    frame.sort_values(["ticker", "ex_date"], inplace=True)
    return frame.reset_index(drop=True)


def calculate_adjustment_factors(actions: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Compute cumulative adjustment factors aligned with the provided index."""

    if actions.empty:
        return pd.DataFrame(1.0, index=index, columns=["price", "cash_dividend"])

    index = to_datetime_index(pd.Index(index)).sort_values()
    factors = pd.DataFrame(1.0, index=index, columns=["price", "cash_dividend"])

    grouped = actions.groupby("ticker")
    for ticker, group in grouped:
        ticker_factors = pd.DataFrame(1.0, index=index, columns=["price", "cash_dividend"])
        for _, action in group.iterrows():
            ex_date = pd.Timestamp(action["ex_date"])
            if ex_date not in ticker_factors.index:
                continue
            if action["event_type"] == "split" and action["ratio"] not in (None, 0):
                ratio = float(action["ratio"])
                ticker_factors.loc[ex_date:, "price"] /= ratio
            elif action["event_type"] in {"cash_dividend", "dividend"} and action["cash_amount"] not in (None, 0):
                cash = float(action["cash_amount"])
                ticker_factors.loc[ex_date:, "cash_dividend"] += cash
            elif action["event_type"] == "spinoff" and action["ratio"] not in (None, 0):
                ratio = float(action["ratio"])
                ticker_factors.loc[ex_date:, "price"] *= (1.0 - ratio)
        factors[f"price_{ticker}"] = ticker_factors["price"]
        factors[f"cash_{ticker}"] = ticker_factors["cash_dividend"]

    return factors


def apply_price_adjustments(prices: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Apply multiplicative price factors to historical prices."""

    if prices.empty:
        return prices
    adjusted = prices.copy()
    for column in prices.columns:
        factor_col = f"price_{column}"
        base_factor = factors["price"] if factor_col not in factors.columns else factors[factor_col]
        aligned = base_factor.reindex(adjusted.index, method="ffill").fillna(1.0)
        adjusted[column] = adjusted[column] * aligned
    return adjusted


def apply_return_adjustments(returns: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Adjust simple returns to include cash dividends when prices are unadjusted."""

    if returns.empty or actions.empty:
        return returns
    adjusted = returns.copy()
    dividends = actions[actions["event_type"].isin({"cash_dividend", "dividend"})]
    if dividends.empty:
        return returns
    grouped = dividends.groupby("ticker")
    for ticker, group in grouped:
        if ticker not in adjusted.columns:
            continue
        series = adjusted[ticker].copy()
        for _, action in group.iterrows():
            ex_date = pd.Timestamp(action["ex_date"])
            cash = float(action.get("cash_amount") or 0.0)
            if cash == 0:
                continue
            if ex_date not in series.index:
                continue
            price_value = series.name  # placeholder for compatibility
            idx = series.index.get_loc(ex_date)
            base_return = series.iloc[idx]
            series.iloc[idx] = base_return + cash
        adjusted[ticker] = series
    return adjusted
