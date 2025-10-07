from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
import logging

logger = logging.getLogger(__name__)


def _find_project_root() -> Path:
    """Resolve a raiz do projeto procurando por pyproject.toml para ser robusto
    a mudanças de localização do arquivo dentro de src/.
    """
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    # fallback: três níveis acima mantém compatibilidade com estrutura atual
    return Path(__file__).resolve().parents[3]


PROJECT_ROOT = _find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


# Universo ARARA (37 ETFs)
ARARA_TICKERS: List[str] = [
    # Ações EUA (amplo)
    "SPY",
    "QQQ",
    "IWM",
    # Desenvolvidos ex-US
    "EFA",
    # Emergentes
    "EEM",
    # Setores EUA
    "XLC",
    "XLY",
    "XLP",
    "XLE",
    "XLF",
    "XLV",
    "XLK",
    "XLI",
    "XLB",
    "XLRE",
    "XLU",
    # Fatores (EUA)
    "USMV",
    "MTUM",
    "QUAL",
    "VLUE",
    "SIZE",
    # Imobiliário
    "VNQ",
    "VNQI",
    # Treasuries (curva)
    "SHY",
    "IEI",
    "IEF",
    "TLT",
    # TIPS
    "TIP",
    # Crédito
    "LQD",
    "HYG",
    "EMB",
    "EMLC",
    # Commodities
    "GLD",
    "DBC",
    # Câmbio USD
    "UUP",
    # Cripto (ETFs spot)
    "IBIT",
    "ETHA",
]


def get_arara_universe() -> List[str]:
    """Retorna a lista de tickers do universo ARARA."""
    return list(ARARA_TICKERS)


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


def calculate_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula os retornos percentuais (decimais) a partir de um DataFrame de preços.

    Args:
        prices_df: DataFrame com preços dos ativos.

    Returns:
        DataFrame com os retornos dos ativos.
    """
    return prices_df.pct_change().dropna()


def download_prices_yf(
    tickers: Iterable[str],
    start: Optional[str | datetime] = None,
    end: Optional[str | datetime] = None,
    progress: bool = False,
) -> pd.DataFrame:
    """Baixa preços ajustados (Adj Close) via yfinance para os tickers informados.

    Retorna um DataFrame wide (índice datas, colunas tickers) em USD.
    Falhas por ticker são logadas e ignoradas.
    """
    tickers_list = list(dict.fromkeys([t.strip().upper() for t in tickers]))
    if not tickers_list:
        raise ValueError("Lista de tickers vazia.")

    logger.info(
        "Baixando preços com yfinance: %s (start=%s, end=%s)",
        ",".join(tickers_list),
        start,
        end,
    )

    data = yf.download(
        tickers=tickers_list,
        start=start,
        end=end,
        auto_adjust=False,
        progress=progress,
        group_by="ticker",
        threads=True,
    )

    # Extrair Adj Close em formato wide
    if isinstance(data.columns, pd.MultiIndex):
        # MultiIndex: nível 0 ticker, nível 1 campo
        try:
            adj = data.xs("Adj Close", axis=1, level=1)
        except Exception:
            # fallback para 'Close' quando Adj Close ausente
            logger.warning("Adj Close ausente; usando Close como fallback.")
            adj = data.xs("Close", axis=1, level=1)
    else:
        # SingleIndex: quando apenas 1 ticker
        col = "Adj Close" if "Adj Close" in data.columns else "Close"
        adj = data[[col]].copy()
        adj.columns = tickers_list[:1]

    # Ordenar colunas conforme tickers_list e dropar colunas totalmente vazias
    adj = adj.reindex(columns=[c for c in tickers_list if c in adj.columns])
    adj = adj.dropna(how="all")

    # Forward-fill em feriados não coincidentes; remove linhas completamente vazias
    adj = adj.sort_index().ffill().dropna(how="all")
    return adj


def download_fred_dtb3(
    start: Optional[str | datetime] = None, end: Optional[str | datetime] = None
) -> pd.Series:
    """Baixa a série DTB3 (Secondary Market 3-Month T-Bill, % a.a.) da FRED.

    Converte para taxa diária aproximada: r_f_daily ≈ (DTB3/100)/360.
    """
    logger.info("Baixando FRED DTB3 (start=%s, end=%s)", start, end)
    s = pdr.DataReader("DTB3", "fred", start, end)["DTB3"]
    s = s.astype(float)
    rf_daily = s.div(100.0).div(360.0)
    rf_daily.name = "rf_daily"
    return rf_daily


def download_and_cache_arara_prices(
    start: Optional[str | datetime] = None,
    end: Optional[str | datetime] = None,
    raw_file_name: str = "prices_arara.csv",
) -> Path:
    """Baixa preços do universo ARARA e salva CSV em data/raw/.

    Retorna o caminho do arquivo salvo.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    prices = download_prices_yf(get_arara_universe(), start=start, end=end)
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
