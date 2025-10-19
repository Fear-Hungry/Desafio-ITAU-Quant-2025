"""Orquestra o pipeline de dados da carteira ARARA.

Componentes principais:
- `load_asset_prices(file_name)`: lê CSVs brutos em `data/raw/`.
- `calculate_returns(prices_df, method)`: thin wrapper para `processing.returns`.
- `download_and_cache_arara_prices(...)`: baixa preços via Yahoo Finance,
  salvando snapshot raw.
- `preprocess_data(raw_file_name, processed_file_name)`: converte preços em
  retornos e persiste em `data/processed/`.
- `download_fred_dtb3(...)`: proxy público para `sources.fred`.
- `DataLoader`: fachada de alto nível a ser usada por backtests. Fluxo:
      1. Baixa preços (`sources.yf.download_prices`).
      2. Normaliza índice, aplica filtros de liquidez e valida painel.
      3. Calcula retornos log, risk-free diário, excess returns.
      4. Gera agenda de rebalance (`processing.calendar`).
      5. Salva Parquets com hash determinístico (`cache.request_hash` + `storage`).
      6. Retorna `DataBundle` contendo prices/returns/rf/excess/bms/inception_mask.
- `DataBundle`: dataclass que encapsula os artefatos prontos para consumo pelas
  camadas de otimização/backtesting.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
from typing import Iterable, Mapping, Optional

import pandas as pd

from .cache import request_hash
from .paths import PROCESSED_DATA_DIR, RAW_DATA_DIR
from .processing.calendar import rebalance_schedule
from .processing.clean import filter_liquid_assets, normalize_index, validate_panel
from .processing.returns import calculate_returns as _calculate_returns
from .processing.returns import compute_excess_returns
from .sources.fred import download_dtb3 as fred_download_dtb3
from .sources.yf import download_prices as yf_download
from .storage import save_parquet
from .universe import get_arara_universe

logger = logging.getLogger(__name__)

__all__ = [
    "get_arara_universe",
    "load_asset_prices",
    "calculate_returns",
    "download_and_cache_arara_prices",
    "preprocess_data",
    "download_and_preprocess_arara",
    "download_fred_dtb3",
    "DataLoader",
    "DataBundle",
]


def load_asset_prices(file_name: str) -> pd.DataFrame:
    """Load raw price data from ``data/raw`` by file name."""
    raw_file_path = RAW_DATA_DIR / file_name
    if not raw_file_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado em: {raw_file_path}")

    return pd.read_csv(raw_file_path, index_col=0, parse_dates=True)


def calculate_returns(prices_df: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """Simple wrapper that delegates to processing.returns.calculate_returns."""
    return _calculate_returns(prices_df, method=method)


def download_and_cache_arara_prices(
    start: Optional[str | datetime] = None,
    end: Optional[str | datetime] = None,
    raw_file_name: str = "prices_arara.csv",
) -> Path:
    """Download ARARA universe prices and persist a CSV under ``data/raw``."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    prices = yf_download(get_arara_universe(), start=start, end=end)
    out_path = RAW_DATA_DIR / raw_file_name
    prices.to_csv(out_path, index=True)
    logger.info("Preços ARARA salvos em %s", out_path)
    return out_path


def preprocess_data(raw_file_name: str, processed_file_name: str) -> pd.DataFrame:
    """Load cached prices, compute returns and persist them under ``data/processed``."""
    logger.info("Iniciando pré-processamento de dados…")
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    prices = load_asset_prices(raw_file_name)
    returns = calculate_returns(prices)

    processed_file_path = PROCESSED_DATA_DIR / processed_file_name
    returns.to_parquet(processed_file_path)
    logger.info("Dados processados e salvos em: %s", processed_file_path)
    return returns


def download_and_preprocess_arara(
    start: Optional[str | datetime] = None,
    end: Optional[str | datetime] = None,
    processed_file_name: str = "returns_arara.parquet",
) -> pd.DataFrame:
    """Convenience function that downloads prices and returns processed returns."""
    raw_path = download_and_cache_arara_prices(start=start, end=end)
    return preprocess_data(raw_path.name, processed_file_name)


def download_fred_dtb3(
    start: Optional[str | datetime] = None,
    end: Optional[str | datetime] = None,
) -> pd.Series:
    """Public wrapper around the FRED DTB3 downloader."""
    return fred_download_dtb3(start=start, end=end)


@dataclass(frozen=True)
class DataBundle:
    prices: pd.DataFrame
    returns: pd.DataFrame
    rf_daily: pd.Series
    excess_returns: pd.DataFrame
    bms: pd.DatetimeIndex
    inception_mask: pd.Series


class DataLoader:
    """High-level orchestrator that wires raw sources, processing and storage."""

    def __init__(
        self,
        tickers: Optional[Iterable[str]] = None,
        start: Optional[str | datetime] = None,
        end: Optional[str | datetime] = None,
        mode: str = "BMS",
        actions: Optional[list[Mapping[str, object]]] = None,
    ) -> None:
        self.tickers = list(tickers) if tickers is not None else get_arara_universe()
        self.start = start
        self.end = end
        self.mode = mode
        self.actions = actions

    def load(self) -> DataBundle:
        """Download prices/r_f, compute returns and persist Parquet snapshots."""
        logger.info(
            "DataLoader: iniciando carga (tickers=%d, start=%s, end=%s, mode=%s)",
            len(self.tickers),
            self.start,
            self.end,
            self.mode,
        )
        prices = yf_download(self.tickers, start=self.start, end=self.end)
        prices = normalize_index(prices)

        actions_cfg = {
            "records": None,
        }
        if self.actions:
            actions = load_corporate_actions(self.tickers, actions=self.actions)
            factors = calculate_adjustment_factors(actions, prices.index)
            prices = apply_price_adjustments(prices, factors)

        prices, liquidity_stats = filter_liquid_assets(prices)
        illiquid: list[str] = []
        if not liquidity_stats.empty:
            liquidity_flags = liquidity_stats["is_liquid"]
            illiquid = liquidity_flags.index[~liquidity_flags].tolist()
        if illiquid:
            logger.warning(
                "Removendo %d ativos com baixa liquidez: %s",
                len(illiquid),
                ", ".join(illiquid),
            )
        if prices.shape[1] == 0:
            raise ValueError("Nenhum ativo restante após filtros de liquidez.")
        validate_panel(prices)

        returns = _calculate_returns(prices, method="log")
        rf = fred_download_dtb3(self.start, self.end)
        excess = compute_excess_returns(returns, rf)
        bms = rebalance_schedule(prices.index, mode=self.mode)
        inception_mask = prices.apply(lambda series: series.first_valid_index())

        hash_id = request_hash(self.tickers, self.start, self.end)
        save_parquet(PROCESSED_DATA_DIR / f"returns_{hash_id}.parquet", returns)
        save_parquet(PROCESSED_DATA_DIR / f"excess_returns_{hash_id}.parquet", excess)

        logger.info(
            "DataLoader: janela efetiva [%s → %s], BMS=%d",
            prices.index.min(),
            prices.index.max(),
            len(bms),
        )
        return DataBundle(
            prices=prices,
            returns=returns,
            rf_daily=rf,
            excess_returns=excess,
            bms=bms,
            inception_mask=inception_mask,
        )
from .processing.corporate_actions import (
    apply_price_adjustments,
    calculate_adjustment_factors,
    load_corporate_actions,
)
