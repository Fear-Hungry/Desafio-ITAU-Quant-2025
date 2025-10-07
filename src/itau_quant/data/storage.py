"""IO utilitários.

Salvar/carregar CSV/Parquet com compressão e versionamento simples.
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path


def save_parquet(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, pd.Series):
        obj.to_frame().to_parquet(path, compression="zstd", index=True)
    else:
        obj.to_parquet(path, compression="zstd", index=True)


def load_parquet(path: Path):
    return pd.read_parquet(path)
