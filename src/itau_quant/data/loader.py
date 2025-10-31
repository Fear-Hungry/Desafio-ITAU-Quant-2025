"""Orquestra o pipeline de dados da carteira ARARA.

Componentes principais:
- ``load_asset_prices(file_name)``: lê CSVs brutos em ``data/raw/``.
- ``calculate_returns(prices_df, method)``: thin wrapper para ``processing.returns``.
- ``download_and_cache_arara_prices(...)``: baixa preços via Yahoo Finance,
  salvando snapshot raw.
- ``preprocess_data(raw_file_name, processed_file_name)``: converte preços em
  retornos e persiste em ``data/processed/``.
- ``download_fred_dtb3(...)``: proxy público para ``sources.fred``.
- ``DataLoader``: fachada de alto nível a ser usada por backtests. Fluxo:
      1. Baixa preços (``sources.yf.download_prices``).
      2. Normaliza índice, aplica filtros de liquidez e valida painel.
      3. Calcula retornos log, risk-free diário, excess returns.
      4. Gera agenda de rebalance (``processing.calendar``).
      5. Salva artefatos Parquet com hash determinístico (``cache.request_hash`` + ``storage``).
      6. Retorna ``DataBundle`` contendo prices/returns/rf/excess/bms/inception_mask.
- ``DataBundle``: dataclass que encapsula os artefatos prontos para consumo pelas
  camadas de otimização/backtesting.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import pandas as pd

from .cache import request_hash
from .paths import PROCESSED_DATA_DIR, RAW_DATA_DIR
from .processing.calendar import rebalance_schedule
from .processing.clean import filter_liquid_assets, normalize_index, validate_panel
from .processing.corporate_actions import (
    apply_price_adjustments,
    calculate_adjustment_factors,
    load_corporate_actions,
)
from .processing.returns import calculate_returns as _calculate_returns
from .processing.returns import compute_excess_returns
from .sources.fred import download_dtb3 as fred_download_dtb3
from .sources.yf import download_prices as yf_download
from .storage import load_parquet, save_parquet
from .sources.crypto import download_crypto_prices as crypto_download
from .universe import get_arara_metadata, get_arara_universe

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
        self._artifacts: dict[str, Any] | None = None

    @property
    def artifacts(self) -> dict[str, Any]:
        """Return artefact metadata for the last ``load`` execution."""
        if self._artifacts is None:
            raise RuntimeError("DataLoader.load must be called before accessing artefacts.")
        return dict(self._artifacts)

    def _build_artifact_paths(self) -> dict[str, Any]:
        request_id = request_hash(self.tickers, self.start, self.end)
        return {
            "request_id": request_id,
            "prices_path": RAW_DATA_DIR / f"prices_{request_id}.parquet",
            "returns_path": PROCESSED_DATA_DIR / f"returns_{request_id}.parquet",
            "excess_path": PROCESSED_DATA_DIR / f"excess_returns_{request_id}.parquet",
            "rf_path": PROCESSED_DATA_DIR / f"rf_daily_{request_id}.parquet",
            "metadata_path": PROCESSED_DATA_DIR / f"bundle_{request_id}.json",
        }

    @staticmethod
    def _cache_available(paths: Mapping[str, Path]) -> bool:
        required = ("prices_path", "returns_path", "excess_path", "rf_path")
        return all(Path(paths[name]).exists() for name in required)

    @staticmethod
    def _load_rf_series(obj: pd.DataFrame | pd.Series) -> pd.Series:
        if isinstance(obj, pd.Series):
            series = obj.copy()
        else:
            series = obj.squeeze("columns")
        series.name = series.name or "rf_daily"
        return series

    def _load_from_cache(self, paths: Mapping[str, Path]) -> DataBundle:
        logger.info("DataLoader: reutilizando artefatos em cache (id=%s)", paths["request_id"])
        prices = load_parquet(Path(paths["prices_path"]))
        returns = load_parquet(Path(paths["returns_path"]))
        excess = load_parquet(Path(paths["excess_path"]))
        rf_obj = load_parquet(Path(paths["rf_path"]))
        rf = self._load_rf_series(rf_obj)

        prices = normalize_index(prices)
        returns = normalize_index(returns)
        excess = normalize_index(excess)
        rf = rf.reindex(returns.index).ffill()

        bms = rebalance_schedule(prices.index, mode=self.mode)
        inception_mask = prices.apply(lambda series: series.first_valid_index())

        metadata_path = Path(paths["metadata_path"])
        metadata: dict[str, Any] | None = None
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metadata = None

        self._artifacts = {
            "request_id": paths["request_id"],
            "prices_path": Path(paths["prices_path"]),
            "returns_path": Path(paths["returns_path"]),
            "excess_path": Path(paths["excess_path"]),
            "rf_path": Path(paths["rf_path"]),
            "metadata_path": Path(paths["metadata_path"]),
            "from_cache": True,
            "metadata": metadata,
        }

        return DataBundle(
            prices=prices,
            returns=returns,
            rf_daily=rf,
            excess_returns=excess,
            bms=bms,
            inception_mask=inception_mask,
        )

    def _persist_metadata(
        self,
        paths: Mapping[str, Path],
        *,
        bundle: DataBundle,
        liquidity_stats: pd.DataFrame,
    ) -> dict[str, Any]:
        generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        metadata = {
            "request_id": paths["request_id"],
            "tickers": self.tickers,
            "start": str(self.start),
            "end": str(self.end),
            "mode": self.mode,
            "generated_at": generated_at,
            "n_assets": int(bundle.returns.shape[1]),
            "n_days": int(bundle.returns.shape[0]),
        }
        if not liquidity_stats.empty and "is_liquid" in liquidity_stats:
            liquid = int(liquidity_stats["is_liquid"].sum())
            illiquid_assets = liquidity_stats.index[~liquidity_stats["is_liquid"]].tolist()
            metadata["liquidity"] = {
                "total": int(len(liquidity_stats)),
                "liquid": liquid,
                "illiquid": int(len(liquidity_stats) - liquid),
                "min_coverage": float(liquidity_stats["coverage"].min()),
                "min_history": int(liquidity_stats["non_na"].min()),
            }
            if illiquid_assets:
                metadata["illiquid_assets"] = illiquid_assets
        metadata_path = Path(paths["metadata_path"])
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return metadata

    def _persist_artifacts(
        self,
        paths: Mapping[str, Path],
        *,
        bundle: DataBundle,
        liquidity_stats: pd.DataFrame,
    ) -> dict[str, Any]:
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        save_parquet(Path(paths["prices_path"]), bundle.prices)
        save_parquet(Path(paths["returns_path"]), bundle.returns)
        save_parquet(Path(paths["excess_path"]), bundle.excess_returns)
        save_parquet(Path(paths["rf_path"]), bundle.rf_daily)

        metadata = self._persist_metadata(paths, bundle=bundle, liquidity_stats=liquidity_stats)

        artefacts = {
            "request_id": paths["request_id"],
            "prices_path": Path(paths["prices_path"]),
            "returns_path": Path(paths["returns_path"]),
            "excess_path": Path(paths["excess_path"]),
            "rf_path": Path(paths["rf_path"]),
            "metadata_path": Path(paths["metadata_path"]),
            "from_cache": False,
            "metadata": metadata,
        }
        return artefacts

    def _split_tickers_by_source(self) -> tuple[list[str], list[str]]:
        metadata = get_arara_metadata()
        crypto_tickers: list[str] = []
        yf_tickers: list[str] = []

        for ticker in self.tickers:
            entry = metadata.get(ticker)
            if entry and str(entry.get("asset_class", "")).lower() == "crypto":
                crypto_tickers.append(ticker)
            else:
                yf_tickers.append(ticker)
        return yf_tickers, crypto_tickers

    @staticmethod
    def _normalize_crypto_frame(frame: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
        if frame.empty:
            return frame

        columns = frame.columns.get_level_values(0)
        for candidate in ("adj_close", "close"):
            if candidate in columns:
                data = frame.loc[:, (candidate, slice(None))].copy()
                data.columns = [symbol.upper() for symbol in data.columns.get_level_values(1)]
                ordered = [ticker for ticker in tickers if ticker in data.columns]
                return data.reindex(columns=ordered).sort_index()
        raise ValueError("Crypto frame missing expected close/adj_close fields.")

    def _download_crypto_assets(self, tickers: Sequence[str], *, force_download: bool) -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame()
        try:
            frame = crypto_download(
                tickers,
                start=self.start,
                end=self.end,
                cache=True,
                force_refresh=force_download,
            )
        except Exception as exc:  # pragma: no cover - network/provider failures
            logger.warning(
                "Falha ao baixar cripto via provider (%s); tentando Yahoo Finance.", exc
            )
            try:
                fallback = yf_download(tickers, start=self.start, end=self.end)
                return fallback
            except Exception as fallback_exc:  # pragma: no cover - defensive
                raise RuntimeError(
                    f"Não foi possível obter dados de cripto para {tickers}: {fallback_exc}"
                ) from fallback_exc

        if frame.empty:
            logger.warning(
                "Provider de cripto retornou painel vazio para %s; tentando fallback Yahoo Finance.",
                ", ".join(tickers),
            )
            return yf_download(tickers, start=self.start, end=self.end)

        return self._normalize_crypto_frame(frame, tickers)

    def load(self, *, force_download: bool = False, cache: bool = True) -> DataBundle:
        """Download prices/r_f, compute returns and persist Parquet snapshots."""
        logger.info(
            "DataLoader: iniciando carga (tickers=%d, start=%s, end=%s, mode=%s)",
            len(self.tickers),
            self.start,
            self.end,
            self.mode,
        )

        paths = self._build_artifact_paths()
        if cache and not force_download and self._cache_available(paths):
            return self._load_from_cache(paths)

        yf_tickers, crypto_tickers = self._split_tickers_by_source()

        price_frames: list[pd.DataFrame] = []
        if yf_tickers:
            yf_prices = yf_download(yf_tickers, start=self.start, end=self.end)
            price_frames.append(yf_prices)

        if crypto_tickers:
            crypto_prices = self._download_crypto_assets(crypto_tickers, force_download=force_download)
            if crypto_prices.empty:
                logger.warning("Painel cripto vazio; removendo tickers: %s", ", ".join(crypto_tickers))
            else:
                price_frames.append(crypto_prices)

        if not price_frames:
            raise ValueError("Nenhum dado de preços foi obtido para os tickers solicitados.")

        prices = pd.concat(price_frames, axis=1, join="outer")
        prices = normalize_index(prices)

        if self.actions:
            actions = load_corporate_actions(self.tickers, actions=self.actions)
            factors = calculate_adjustment_factors(actions, prices.index)
            prices = apply_price_adjustments(prices, factors)

        prices, liquidity_stats = filter_liquid_assets(prices)
        illiquid: list[str] = []
        if not liquidity_stats.empty and "is_liquid" in liquidity_stats:
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

        try:
            rf = fred_download_dtb3(self.start, self.end)
        except ImportError as exc:  # pragma: no cover - depende de extra opcional
            logger.warning("FRED downloader indisponível (%s); assumindo rf=0.", exc)
            rf = pd.Series(0.0, index=returns.index, name="rf_daily")

        excess = compute_excess_returns(returns, rf)
        bms = rebalance_schedule(prices.index, mode=self.mode)
        inception_mask = prices.apply(lambda series: series.first_valid_index())

        bundle = DataBundle(
            prices=prices,
            returns=returns,
            rf_daily=rf.reindex(returns.index).ffill(),
            excess_returns=excess,
            bms=bms,
            inception_mask=inception_mask,
        )

        logger.info(
            "DataLoader: janela efetiva [%s → %s], BMS=%d",
            prices.index.min(),
            prices.index.max(),
            len(bms),
        )

        if cache:
            self._artifacts = self._persist_artifacts(
                paths,
                bundle=bundle,
                liquidity_stats=liquidity_stats,
            )
        else:
            self._artifacts = {
                "request_id": paths["request_id"],
                "prices_path": None,
                "returns_path": None,
                "excess_path": None,
                "rf_path": None,
                "metadata_path": None,
                "from_cache": False,
                "metadata": None,
            }

        return bundle
