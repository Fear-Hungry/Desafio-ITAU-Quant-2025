"""Persistência em Parquet seguindo convenções do projeto.

Rotinas expostas
----------------
`save_parquet(path, obj)`
    - Cria hierarquia de diretórios automaticamente.
    - Serializa DataFrame/Series com compressão ``zstd`` e índice preservado.
    - Utilizada por ``DataLoader`` e stages de pré-processamento.

`load_parquet(path)`
    - Retorna pandas DataFrame/Series com o schema original.
    - Serve tanto para reprocessamento quanto para consumo em notebooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def save_parquet(path: Path, obj: Any) -> None:
    """Persist DataFrame/Series using zstd compression."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, pd.Series):
        obj.to_frame().to_parquet(path, compression="zstd", index=True)
    else:
        obj.to_parquet(path, compression="zstd", index=True)


def load_parquet(path: Path) -> pd.DataFrame:
    """Load Parquet artefact into a pandas object."""
    return pd.read_parquet(path)
