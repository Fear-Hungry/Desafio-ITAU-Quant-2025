# Placeholder implementation notice for future contributors:
# - Objetivo: replicar o padrão de `sources.yf`/`sources.csv` para ETFs/ETNs
#   cripto, provendo dados prontos para `processing.clean`.
# - API sugerida:
#     * `download_crypto_prices(tickers, start=None, end=None, provider="tiingo", fields=("Adj Close",))`
#          - Retornar DataFrame wide com índice diário.
#          - Permitir seleção de campos extras (volume, NAV, premium).
#     * `download_crypto_ohlcv(...)`
#          - Para casos onde OHLC completo é requerido em backtests intradiários.
# - Providers esperados: Coinbase Advanced, Kaiko, Tiingo, Binance, AlphaVantage.
#     * Cada provider deve ter helper: `_load_from_tiingo`, `_load_from_kaiko`, etc.
#     * Implementar retries exponenciais, tratamento de rate-limit (`429`) e
#       logs amigáveis (`logger.warning`).
# - Normalização/cross-cutting concerns:
#     * `_sanitize_symbols(tickers)` — padroniza separadores, sufixos regionais.
#     * `_convert_currency(df, from_currency, to_currency, fx_source)` — caso o
#       ativo seja cotado em outra moeda (ex.: BTC em USD vs BRL).
#     * `_ensure_session_calendar(df, calendar)` — fecha gaps forçando forward-fill
#       ou reamostragem conforme calendário alvo (CME/NYSE/NasdaqCrypto).
# - Metadados:
#     * Retornar dicionário adicional ou armazenar em `df.attrs` com informações:
#       exchange, base_currency, quote_currency, timezone e provider.
# - Persistência opcional:
#     * Reutilizar `cache.request_hash` + `storage.save_parquet` criando diretório
#       `data/raw/crypto` para facilitar auditoria.
# - Testes sugeridos (`tests/data/sources/test_crypto.py` futuro):
#     * Mock de provider retornando OHLCV e validação de normalização de colunas.
#     * Erro amigável quando provider desconhecido é solicitado.
#     * Conversão de timezone/moeda preservando ordem cronológica.

"""Crypto data sources (ETFs, ETNs and spot exchanges)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Iterable, Mapping, Sequence

import pandas as pd
import requests

from itau_quant.utils.logging_config import get_logger

logger = get_logger(__name__)

__all__ = [
    "CryptoProviderConfig",
    "download_crypto_prices",
    "download_crypto_ohlcv",
]


@dataclass(frozen=True)
class CryptoProviderConfig:
    name: str
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 10.0
    retries: int = 3


def _sanitize_symbols(symbols: Sequence[str]) -> list[str]:
    cleaned = []
    for symbol in symbols:
        symbol = symbol.strip().upper()
        symbol = symbol.replace("-SPOT", "").replace(" ", "")
        symbol = symbol.replace("-", "")
        cleaned.append(symbol)
    return cleaned


def _request_json(url: str, *, params: Mapping[str, object] | None, headers: Mapping[str, str] | None, config: CryptoProviderConfig) -> dict[str, object]:
    for attempt in range(config.retries):
        response = requests.get(url, params=params, headers=headers, timeout=config.timeout)
        if response.status_code == 429:
            logger.warning("Rate limited by %s (attempt %s/%s)", config.name, attempt + 1, config.retries)
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError(f"provider {config.name} failed after {config.retries} retries")


def _load_from_tiingo(symbols: Sequence[str], start: str | None, end: str | None, config: CryptoProviderConfig, fields: Sequence[str]) -> pd.DataFrame:
    base_url = config.base_url or "https://api.tiingo.com/tiingo/crypto/prices"
    headers = {"Content-Type": "application/json", "Authorization": f"Token {config.api_key}"} if config.api_key else {}
    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        params = {"tickers": symbol, "startDate": start, "endDate": end}
        payload = _request_json(base_url, params=params, headers=headers, config=config)
        if not payload:
            continue
        data = payload[0]["priceData"]
        frame = pd.DataFrame(data)
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.set_index("date").sort_index()
        selected = frame.rename(columns=str.title)[list(fields)]
        selected.columns = pd.MultiIndex.from_product([[symbol], selected.columns])
        frames.append(selected)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)


def _pivot_close(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        return frame.xs(field, level=1, axis=1)
    return frame[[field]].copy()


class CryptoDownloader:
    def __init__(self, provider: str, api_key: str | None = None, **kwargs: object) -> None:
        self.config = CryptoProviderConfig(name=provider.lower(), api_key=api_key, **kwargs)

    @cached_property
    def loader(self) -> callable:
        if self.config.name == "tiingo":
            return _load_from_tiingo
        raise ValueError(f"unsupported crypto provider '{self.config.name}'")

    def prices(self, symbols: Sequence[str], *, start: str | None, end: str | None, fields: Sequence[str]) -> pd.DataFrame:
        symbols = _sanitize_symbols(symbols)
        frame = self.loader(symbols, start, end, self.config, fields)
        if frame.empty:
            logger.warning("No crypto data returned for %s", symbols)
        return frame


def download_crypto_prices(
    tickers: Sequence[str],
    *,
    start: str | None = None,
    end: str | None = None,
    provider: str = "tiingo",
    fields: Sequence[str] = ("Close",),
    api_key: str | None = None,
) -> pd.DataFrame:
    downloader = CryptoDownloader(provider, api_key=api_key)
    frame = downloader.prices(tickers, start=start, end=end, fields=fields)
    if frame.empty:
        return frame
    result = {}
    for field in fields:
        result[field.lower()] = _pivot_close(frame, field)
    return pd.concat(result, axis=1)


def download_crypto_ohlcv(
    tickers: Sequence[str],
    *,
    start: str | None = None,
    end: str | None = None,
    provider: str = "tiingo",
    api_key: str | None = None,
) -> pd.DataFrame:
    fields = ("Open", "High", "Low", "Close", "Volume")
    frame = download_crypto_prices(tickers, start=start, end=end, provider=provider, fields=fields, api_key=api_key)
    return frame
