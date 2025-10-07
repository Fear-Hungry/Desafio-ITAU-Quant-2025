"""Fachada de dados ARARA.

Responsável por orquestrar operações de alto nível: baixar preços/r_f,
calcular retornos e persistir artefatos. Delega a submódulos em
`data.paths`, `data.universe`, `data.sources.*` e `data.processing.*`.
"""

from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable

import pandas as pd
import logging
from .paths import RAW_DATA_DIR, PROCESSED_DATA_DIR
from .universe import get_arara_universe
from .sources.yf import download_prices as yf_download
from .sources.fred import download_dtb3 as fred_download_dtb3
from .processing.returns import (
    calculate_returns as _calculate_returns,
    compute_excess_returns,
)
from .processing.clean import normalize_index, validate_panel
from .processing.calendar import rebalance_schedule
from .storage import save_parquet
from .cache import request_hash

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
    """
    Carrega os preços dos ativos de um arquivo CSV no diretório de dados brutos.

    Args:
        file_name: O nome do arquivo a ser carregado (ex: 'asset_prices.csv').

    Returns:
        Um DataFrame do pandas com os preços dos ativos.
    """
    raw_file_path = RAW_DATA_DIR / file_name
    if not raw_file_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado em: {raw_file_path}")

    # Supõe que a primeira coluna é a data e a define como índice
    df = pd.read_csv(raw_file_path, index_col=0, parse_dates=True)
    return df


def calculate_returns(prices_df: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    return _calculate_returns(prices_df, method=method)


def download_and_cache_arara_prices(
    start: Optional[str | datetime] = None,
    end: Optional[str | datetime] = None,
    raw_file_name: str = "prices_arara.csv",
) -> Path:
    """Baixa preços do universo ARARA e salva CSV em data/raw/.

    Retorna o caminho do arquivo salvo.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    prices, _ = yf_download(get_arara_universe(),
                            start=start, end=end, with_volume=False)
    out_path = RAW_DATA_DIR / raw_file_name
    prices.to_csv(out_path, index=True)
    logger.info("Preços ARARA salvos em %s", out_path)
    return out_path


def preprocess_data(raw_file_name: str, processed_file_name: str) -> pd.DataFrame:
    """
    Orquestra o pipeline completo de pré-processamento de dados:
    1. Carrega os preços brutos.
    2. Calcula os retornos.
    3. Salva os retornos processados em um novo arquivo.
    4. Retorna o DataFrame de retornos.

    Args:
        raw_file_name: Nome do arquivo de dados brutos.
        processed_file_name: Nome do arquivo para salvar os dados processados.

    Returns:
        DataFrame com retornos limpos e prontos para análise.
    """
    logger.info("Iniciando pré-processamento de dados...")

    # Garante que o diretório de destino exista
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
    """Baixa preços do universo ARARA, calcula retornos e salva em data/processed/.

    Mantém compatibilidade com o pipeline baseado em arquivos, mas sem exigir
    um CSV pré-existente.
    """
    raw_path = download_and_cache_arara_prices(start=start, end=end)
    return preprocess_data(raw_path.name, processed_file_name)


def download_fred_dtb3(start: Optional[str | datetime] = None, end: Optional[str | datetime] = None) -> pd.Series:
    """Wrapper público para baixar r_f diário aproximado (DTB3)."""
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
    def __init__(
        self,
        tickers: Optional[Iterable[str]] = None,
        start: Optional[str | datetime] = None,
        end: Optional[str | datetime] = None,
        mode: str = "BMS",
    ) -> None:
        self.tickers = list(
            tickers) if tickers is not None else get_arara_universe()
        self.start = start
        self.end = end
        self.mode = mode

    def load(self) -> DataBundle:
        logger.info(
            "DataLoader: iniciando carga (tickers=%d, start=%s, end=%s, mode=%s)",
            len(self.tickers),
            self.start,
            self.end,
            self.mode,
        )

        prices, _vol = yf_download(
            self.tickers, start=self.start, end=self.end, with_volume=True
        )
        prices = normalize_index(prices)
        validate_panel(prices)

        returns = _calculate_returns(prices, method="log")
        rf = fred_download_dtb3(self.start, self.end)
        excess = compute_excess_returns(returns, rf)
        bms = rebalance_schedule(prices.index, mode=self.mode)
        inception_mask = prices.apply(lambda s: s.first_valid_index())

        # salvar com hash do pedido
        hash_id = request_hash(self.tickers, self.start, self.end)
        save_parquet(PROCESSED_DATA_DIR /
                     f"returns_{hash_id}.parquet", returns)
        save_parquet(PROCESSED_DATA_DIR /
                     f"excess_returns_{hash_id}.parquet", excess)

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
