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

import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd
import requests

from arara_quant.config import Settings, get_settings
from arara_quant.data.cache import request_hash
from arara_quant.data.storage import load_parquet, save_parquet
from arara_quant.utils.logging_config import get_logger

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
    quote_currency: str = "USD"
    timezone: str = "UTC"
    session: str = "D"


def _sanitize_symbols(symbols: Sequence[str]) -> list[str]:
    cleaned = []
    for symbol in symbols:
        symbol = symbol.strip().upper()
        symbol = symbol.replace("-SPOT", "")
        # Preserve standard separators like "-" so downstream reporting keeps
        # canonical tickers (e.g., BTC-USD) while still stripping noise.
        symbol = re.sub(r"[^A-Z0-9\-]", "", symbol)
        # Collapse repeated hyphens that may appear after cleanup.
        symbol = re.sub(r"-{2,}", "-", symbol)
        cleaned.append(symbol)
    return cleaned


def _request_json(
    url: str,
    *,
    params: Mapping[str, object] | None,
    headers: Mapping[str, str] | None,
    config: CryptoProviderConfig,
) -> Any:
    for attempt in range(config.retries):
        response = requests.get(
            url, params=params, headers=headers, timeout=config.timeout
        )
        if response.status_code == 429:
            logger.warning(
                "Rate limited by %s (attempt %s/%s)",
                config.name,
                attempt + 1,
                config.retries,
            )
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError(f"provider {config.name} failed after {config.retries} retries")


def _load_from_tiingo(
    symbols: Sequence[str],
    start: str | None,
    end: str | None,
    config: CryptoProviderConfig,
    fields: Sequence[str],
) -> pd.DataFrame:
    base_url = config.base_url or "https://api.tiingo.com/tiingo/crypto/prices"
    headers = (
        {"Content-Type": "application/json", "Authorization": f"Token {config.api_key}"}
        if config.api_key
        else {}
    )
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
        selected = frame.rename(columns=str.title)
        available = [
            col
            for col in selected.columns
            if col
            in {
                "Open",
                "High",
                "Low",
                "Close",
                "AdjClose",
                "AdjCloseBid",
                "AdjCloseAsk",
                "Volume",
            }
        ]
        selected = selected[available]
        selected.columns = pd.MultiIndex.from_product([[symbol], selected.columns])
        frames.append(selected)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)


def _load_from_coinbase(
    symbols: Sequence[str],
    start: str | None,
    end: str | None,
    config: CryptoProviderConfig,
    fields: Sequence[str],
) -> pd.DataFrame:
    base_url = config.base_url or "https://api.exchange.coinbase.com/products"
    headers = {"Content-Type": "application/json"}
    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        product = symbol if "-" in symbol else f"{symbol[:-3]}-{symbol[-3:]}"
        url = f"{base_url}/{product}/candles"
        params: dict[str, object] = {"granularity": 86400}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        payload = _request_json(url, params=params, headers=headers, config=config)
        if not payload:
            continue
        # API returns [time, low, high, open, close, volume]
        data = pd.DataFrame(
            payload, columns=["timestamp", "low", "high", "open", "close", "volume"]
        )
        data["timestamp"] = pd.to_datetime(data["timestamp"], unit="s", utc=True)
        data = data.set_index("timestamp").sort_index()
        data = data.rename(columns=str.title)
        frames.append(pd.concat({symbol: data}, axis=1))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)


def _load_from_binance(
    symbols: Sequence[str],
    start: str | None,
    end: str | None,
    config: CryptoProviderConfig,
    fields: Sequence[str],
) -> pd.DataFrame:
    base_url = config.base_url or "https://api.binance.com/api/v3/klines"
    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        params: dict[str, object] = {"symbol": symbol, "interval": "1d", "limit": 1000}
        if start:
            params["startTime"] = int(pd.Timestamp(start).timestamp() * 1000)
        if end:
            params["endTime"] = int(pd.Timestamp(end).timestamp() * 1000)
        payload = _request_json(base_url, params=params, headers=None, config=config)
        if not payload:
            continue
        data = pd.DataFrame(
            payload,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trades",
                "taker_base",
                "taker_quote",
                "ignore",
            ],
        )
        data["open_time"] = pd.to_datetime(data["open_time"], unit="ms", utc=True)
        data = data.set_index("open_time").sort_index()
        selected = data[["open", "high", "low", "close", "volume"]].astype(float)
        selected = selected.rename(columns=str.title)
        frames.append(pd.concat({symbol: selected}, axis=1))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)


def _pivot_close(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        return frame.xs(field, level=1, axis=1)
    return frame[[field]].copy()


class CryptoDownloader:
    def __init__(
        self, provider: str, api_key: str | None = None, **kwargs: object
    ) -> None:
        name = provider.lower()
        defaults = _PROVIDERS.get(name)
        if defaults is None:
            raise ValueError(f"unsupported crypto provider '{provider}'")
        merged = {**defaults.get("config", {}), **kwargs}
        self.config = CryptoProviderConfig(name=name, api_key=api_key, **merged)

    @cached_property
    def loader(self) -> Callable[..., pd.DataFrame]:
        return _PROVIDERS[self.config.name]["loader"]

    @cached_property
    def symbol_formatter(self) -> Callable[[str], str]:
        return _PROVIDERS[self.config.name].get("format_symbol", lambda s: s)

    def prices(
        self,
        symbols: Sequence[str],
        *,
        start: str | None,
        end: str | None,
        fields: Sequence[str],
    ) -> pd.DataFrame:
        cleaned = _sanitize_symbols(symbols)
        provider_symbols = [self.symbol_formatter(symbol) for symbol in cleaned]
        frame = self.loader(provider_symbols, start, end, self.config, fields)
        if getattr(frame, "empty", True):
            return frame
        mapping = dict(zip(provider_symbols, cleaned))
        if isinstance(frame.columns, pd.MultiIndex):
            new_cols = [(mapping.get(sym, sym), field) for sym, field in frame.columns]
            frame.columns = pd.MultiIndex.from_tuples(new_cols)
        else:
            frame.columns = [mapping.get(col, col) for col in frame.columns]
        if frame.empty:
            logger.warning("No crypto data returned for %s", symbols)
        return frame


def _normalize_field_name(name: str) -> str:
    mapping = {
        "adjclose": "adj_close",
        "adjclosebid": "adj_close_bid",
        "adjcloseask": "adj_close_ask",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "quote_volume": "quote_volume",
    }
    return mapping.get(str(name).replace(" ", "").lower(), str(name).lower())


def _normalize_frame(
    frame: pd.DataFrame,
    *,
    fields: Sequence[str],
    provider: str,
    config: CryptoProviderConfig,
) -> pd.DataFrame:
    if frame.empty:
        return frame

    data = frame.copy()
    data.index = pd.to_datetime(data.index).tz_localize(None)
    data = data.sort_index()

    normalized_columns: list[tuple[str, str]] = []
    if isinstance(data.columns, pd.MultiIndex):
        for sym, field in data.columns:
            normalized_columns.append((sym, _normalize_field_name(field)))
        data.columns = pd.MultiIndex.from_tuples(
            normalized_columns, names=["symbol", "field"]
        )
    else:
        data.columns = pd.MultiIndex.from_tuples(
            [(col, "close") for col in data.columns], names=["symbol", "field"]
        )

    data = data.swaplevel(0, 1, axis=1)
    canonical_fields = [_normalize_field_name(field) for field in fields]
    available = [
        field for field in canonical_fields if field in data.columns.get_level_values(0)
    ]
    if not available:
        raise ValueError(
            f"requested fields {fields} not available for provider '{provider}'"
        )
    data = data.loc[:, available]
    data = data.sort_index(axis=1)
    data.attrs.update(
        {
            "provider": provider,
            "quote_currency": config.quote_currency,
            "timezone": config.timezone,
        }
    )
    return data


def _apply_fx_conversion(
    frame: pd.DataFrame,
    *,
    quote_currency: str,
    target_currency: str | None,
    fx_series: pd.Series | None,
) -> pd.DataFrame:
    if (
        target_currency is None
        or target_currency == quote_currency
        or fx_series is None
    ):
        return frame
    aligned_fx = fx_series.reindex(frame.index).ffill()
    price_fields = {
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "adj_close_bid",
        "adj_close_ask",
    }
    data = frame.copy()
    for field in data.columns.get_level_values(0).unique():
        if field not in price_fields:
            continue
        data.loc[:, (field, slice(None))] = data.loc[:, (field, slice(None))].mul(
            aligned_fx, axis=0
        )
    data.attrs["converted_to"] = target_currency
    return data


_PROVIDERS: dict[str, dict[str, Any]] = {
    "tiingo": {
        "loader": _load_from_tiingo,
        "config": {
            "base_url": "https://api.tiingo.com/tiingo/crypto/prices",
            "quote_currency": "USD",
        },
    },
    "coinbase": {
        "loader": _load_from_coinbase,
        "config": {
            "base_url": "https://api.exchange.coinbase.com/products",
            "quote_currency": "USD",
        },
    },
    "binance": {
        "loader": _load_from_binance,
        "config": {
            "base_url": "https://api.binance.com/api/v3/klines",
            "quote_currency": "USD",
        },
    },
}


def download_crypto_prices(
    tickers: Sequence[str],
    *,
    start: str | None = None,
    end: str | None = None,
    provider: str = "tiingo",
    fields: Sequence[str] = ("Close",),
    api_key: str | None = None,
    cache: bool = False,
    force_refresh: bool = False,
    cache_dir: str | Path | None = None,
    settings: Settings | None = None,
    target_currency: str | None = None,
    fx_series: pd.Series | None = None,
    provider_kwargs: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    provider_kwargs = dict(provider_kwargs or {})
    cleaned = _sanitize_symbols(tickers)
    canonical_fields = [_normalize_field_name(field) for field in fields]

    cache_path: Path | None = None
    if cache:
        settings = settings or get_settings()
        base_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else settings.raw_data_dir / "crypto"
        )
        identifier = request_hash(cleaned + [provider], start, end)
        field_tag = "_".join(canonical_fields)
        target_tag = target_currency or "DEFAULT"
        cache_path = (
            base_dir / provider / f"{field_tag}_{target_tag}_{identifier}.parquet"
        )
        if not force_refresh and cache_path.exists():
            logger.info("Loading crypto data from cache: %s", cache_path)
            return load_parquet(cache_path)

    downloader = CryptoDownloader(provider, api_key=api_key, **provider_kwargs)
    frame = downloader.prices(cleaned, start=start, end=end, fields=fields)
    if frame.empty:
        return frame

    normalized = _normalize_frame(
        frame, fields=fields, provider=provider, config=downloader.config
    )
    converted = _apply_fx_conversion(
        normalized,
        quote_currency=downloader.config.quote_currency,
        target_currency=target_currency,
        fx_series=fx_series,
    )

    if cache and cache_path is not None:
        save_parquet(cache_path, converted)
    return converted


def download_crypto_ohlcv(
    tickers: Sequence[str],
    *,
    start: str | None = None,
    end: str | None = None,
    provider: str = "tiingo",
    api_key: str | None = None,
    cache: bool = False,
    force_refresh: bool = False,
    cache_dir: str | Path | None = None,
    settings: Settings | None = None,
    target_currency: str | None = None,
    fx_series: pd.Series | None = None,
    provider_kwargs: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    fields = ("Open", "High", "Low", "Close", "Volume")
    return download_crypto_prices(
        tickers,
        start=start,
        end=end,
        provider=provider,
        fields=fields,
        api_key=api_key,
        cache=cache,
        force_refresh=force_refresh,
        cache_dir=cache_dir,
        settings=settings,
        target_currency=target_currency,
        fx_series=fx_series,
        provider_kwargs=provider_kwargs,
    )
