"""Ingestão de dados locais (CSV/Excel) com schema padronizado.

`CSVSchemaError`
    Exceção específica para sinalizar ausência de colunas, erros de parsing de
    datas ou falta de observações.

`load_price_panel(path, index_col="date", expected_columns=None, ...)`
    - Lê o arquivo com pandas, garantindo presença da coluna de índice.
    - Restringe colunas opcionais (`expected_columns`) e converte-as para número.
    - Controla exclusão de linhas/colunas vazias (``drop_empty_*`` flags).
    - Converte o índice para ``DatetimeIndex`` e aplica ``normalize_index``.
    - Erros de schema resultam em ``CSVSchemaError`` com mensagem descritiva.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

from ..processing.clean import normalize_index

__all__ = ["CSVSchemaError", "load_price_panel"]


class CSVSchemaError(ValueError):
    """Raise when a CSV file does not match the expected schema."""


def _ensure_numeric(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    converted = df.copy()
    for column in columns:
        converted[column] = pd.to_numeric(converted[column], errors="coerce")
    return converted


def load_price_panel(
    path: str | Path,
    *,
    index_col: str = "date",
    expected_columns: Optional[Iterable[str]] = None,
    parse_dates: bool = True,
    drop_empty_rows: bool = True,
    drop_empty_columns: bool = True,
) -> pd.DataFrame:
    """Load a wide price panel from a CSV file.

    Parameters
    ----------
    path
        CSV file containing a ``date`` column plus one column per ticker.
    index_col
        Name of the column that should become the DatetimeIndex.
    expected_columns
        Optional iterable restricting which ticker columns are retained. Raises
        ``CSVSchemaError`` if any requested column is missing.
    parse_dates
        When ``True``, attempts to convert the index column to timestamps.
    drop_empty_rows
        Whether rows with all missing values should be discarded.
    drop_empty_columns
        Whether columns empty after conversion should be discarded.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {csv_path}")

    df = pd.read_csv(csv_path)
    if index_col not in df.columns:
        raise CSVSchemaError(f"Coluna de índice '{index_col}' ausente em {csv_path}.")

    columns = [col for col in df.columns if col != index_col]
    if expected_columns is not None:
        expected = [col.strip() for col in expected_columns]
        missing = [col for col in expected if col not in columns]
        if missing:
            raise CSVSchemaError(
                f"Colunas esperadas ausentes: {', '.join(sorted(missing))}"
            )
        columns = expected

    if parse_dates:
        try:
            df[index_col] = pd.to_datetime(df[index_col], utc=False)
        except (pd.errors.OutOfBoundsDatetime, ValueError) as exc:
            raise CSVSchemaError(f"Falha ao converter datas: {exc}") from exc

    df = df.set_index(index_col)
    df = df.sort_index()
    df = df.loc[:, columns]
    df = _ensure_numeric(df, columns)

    if drop_empty_rows:
        df = df.dropna(how="all")
    if drop_empty_columns:
        df = df.dropna(axis=1, how="all")

    if df.empty:
        raise CSVSchemaError("CSV sem observações válidas após limpeza.")

    try:
        df = normalize_index(df)
    except ValueError as exc:
        raise CSVSchemaError(str(exc)) from exc

    return df
