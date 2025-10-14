"""Limpeza e validações de painéis de preços.

Guia das funções
----------------
`ensure_dtindex(idx)`
    Normaliza coleções de datas (strings, datetime, índices) para `DatetimeIndex`
    ordenado e tz-naive.

`normalize_index(df)`
    Reordena o DataFrame seguindo o índice temporal, validando duplicatas e
    removendo timezone.

`validate_panel(prices)`
    Sanity checks rápidos aplicados antes de seguir no pipeline (ordem/uniqueness).

`compute_liquidity_stats(prices)`
    Calcula cobertura (% de dados disponíveis), maior gap de NaNs e datas de
    início/fim válidas por ticker.

`filter_liquid_assets(prices, min_history, min_coverage, max_gap)`
    Usa as estatísticas acima para remover ativos ilíquidos, retornando tanto o
    painel filtrado quanto os diagnósticos.

`winsorize_outliers(data, lower, upper, per_column)`
    Aplica winsorização baseada em quantis para atenuar outliers; suporta Series
    (tratamento individual) e DataFrames (colunas ou painel global).
"""

from __future__ import annotations

from typing import Iterable, Tuple, Union
import pandas as pd
from pandas import DatetimeIndex


def _longest_nan_streak(series: pd.Series) -> int:
    """Return the maximum number of consecutive NaN values in the series."""
    max_gap = 0
    current_gap = 0
    for is_nan in series.isna():
        if is_nan:
            current_gap += 1
            if current_gap > max_gap:
                max_gap = current_gap
        else:
            current_gap = 0
    return max_gap


def ensure_dtindex(idx: Iterable) -> DatetimeIndex:
    """Converte para DatetimeIndex ordenado, sem timezone.

    - Aceita iteráveis de datas e strings.
    - Ordena e remove duplicatas.
    - Remove timezone (tz-naive) para comparações e agrupamentos consistentes.
    """
    if not isinstance(idx, DatetimeIndex):
        out = DatetimeIndex(pd.to_datetime(list(idx)))
    else:
        out = DatetimeIndex(idx)
    # ordenar + únicos
    out = DatetimeIndex(sorted(out.unique()))
    # normalizar timezone
    if getattr(out, "tz", None) is not None:
        out = out.tz_localize(None)
    return out


def normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna cópia com índice DatetimeIndex ordenado, tz-naive e alinhado aos dados.

    Mantém os valores corretos associados às datas ao reordenar as linhas e
    normalizar o índice. Lança ValueError se houver duplicatas após a conversão.
    """
    if df.empty:
        return df.copy()

    normalized = df.copy()
    idx = pd.DatetimeIndex(pd.to_datetime(normalized.index))

    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)

    sorted_idx, order = idx.sort_values(return_indexer=True)
    if sorted_idx.has_duplicates:
        raise ValueError("Index contém datas duplicatas após normalização")

    normalized = normalized.iloc[order]
    normalized.index = sorted_idx
    return normalized


def validate_panel(prices: pd.DataFrame) -> None:
    """Checks básicos de sanidade do painel de preços."""
    assert prices.index.is_monotonic_increasing, "Index fora de ordem"
    assert prices.index.is_unique, "Index com duplicatas"
    assert prices.notna().any().any(), "Painel vazio ou todo NaN"


def compute_liquidity_stats(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute coverage stats that support liquidity filters."""
    if prices.empty:
        return pd.DataFrame()

    total_obs = len(prices.index)
    stats = []
    for ticker in prices.columns:
        series = prices[ticker]
        non_na = int(series.notna().sum())
        coverage = float(non_na / total_obs) if total_obs else 0.0
        stats.append(
            {
                "ticker": ticker,
                "non_na": non_na,
                "coverage": coverage,
                "max_gap": _longest_nan_streak(series),
                "first_valid": series.first_valid_index(),
                "last_valid": series.last_valid_index(),
            }
        )
    return pd.DataFrame(stats).set_index("ticker")


def filter_liquid_assets(
    prices: pd.DataFrame,
    *,
    min_history: int = 252,
    min_coverage: float = 0.85,
    max_gap: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Drop assets that fail basic liquidity screens.

    Parameters
    ----------
    prices
        Wide dataframe (index=date, columns=tickers) containing price levels.
    min_history
        Minimum number of available observations required. Values larger than the
        sample length are clipped automatically.
    min_coverage
        Minimum share of non-missing observations (0-1).
    max_gap
        Maximum tolerated streak of consecutive missing values.

    Returns
    -------
    filtered_prices, stats
        ``filtered_prices`` keeps only tickers that satisfy all thresholds. ``stats``
        carries diagnostic columns plus the boolean ``is_liquid``.
    """
    if prices.empty:
        return prices.copy(), pd.DataFrame()

    stats = compute_liquidity_stats(prices)
    effective_min_history = min(len(prices), max(min_history, 0))
    stats["is_liquid"] = (
        (stats["non_na"] >= effective_min_history)
        & (stats["coverage"] >= min_coverage)
        & (stats["max_gap"] <= max_gap)
    )
    liquid_tickers = stats.index[stats["is_liquid"]].tolist()
    filtered = prices.loc[:, liquid_tickers]
    return filtered, stats


DataLike = Union[pd.Series, pd.DataFrame]


def _winsorize_series(
    series: pd.Series, lower: float, upper: float
) -> pd.Series:
    if series.empty:
        return series.copy()
    quantiles = series.quantile([lower, upper])
    lower_bound = quantiles.iloc[0]
    upper_bound = quantiles.iloc[1]
    if pd.isna(lower_bound) and pd.isna(upper_bound):
        return series.copy()
    return series.clip(lower_bound, upper_bound)


def winsorize_outliers(
    data: DataLike,
    *,
    lower: float = 0.01,
    upper: float = 0.99,
    per_column: bool = True,
) -> DataLike:
    """Clip observations outside quantile thresholds (Winsorization).

    Parameters
    ----------
    data
        Series/DataFrame containing numeric observations to be winsorized.
    lower, upper
        Quantile cut-offs in the [0, 1] interval. Requires ``lower < upper``.
    per_column
        When ``True`` (default) compute quantiles separately for each column.
        When ``False`` compute over the flattened panel and clip using global
        bounds.
    """
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("Parâmetros 'lower' e 'upper' devem obedecer 0 <= lower < upper <= 1.")

    if isinstance(data, pd.Series):
        return _winsorize_series(data, lower, upper)
    if not isinstance(data, pd.DataFrame):
        raise TypeError("winsorize_outliers aceita apenas pandas Series ou DataFrame.")
    if data.empty:
        return data.copy()

    if per_column:
        lower_bounds = data.quantile(lower, axis=0)
        upper_bounds = data.quantile(upper, axis=0)
        return data.clip(lower=lower_bounds, upper=upper_bounds, axis=1)

    flattened = data.stack(future_stack=True).dropna()
    if flattened.empty:
        return data.copy()
    q_low = flattened.quantile(lower)
    q_high = flattened.quantile(upper)
    return data.clip(lower=q_low, upper=q_high)
