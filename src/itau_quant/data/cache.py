"""Cache helpers para nomear artefatos de forma determinística.

Motivação
---------
Backtests e pipelines de ETL geram múltiplos arquivos intermediários. Para
evitar reprocessamento redundante reutilizamos snapshots seguindo a convenção:

``request_hash(tickers, start, end)``
    - Normaliza a carga (ordena/remover duplicatas dos tickers).
    - Serializa as datas como strings e gera hash SHA-256 truncado (12 chars).
    - Pode ser usado em caminhos, e.g. ``returns_<hash>.parquet``.

Exemplo
-------
``hash_id = request_hash(["SPY", "EFA"], "2020-01-01", "2024-01-01")``
``save_parquet(PROCESSED_DATA_DIR / f"returns_{hash_id}.parquet", df)``
"""

from __future__ import annotations

import hashlib
import json
from typing import Iterable, Optional


def request_hash(
    tickers: Iterable[str],
    start: Optional[str],
    end: Optional[str],
) -> str:
    """Build a deterministic hash for caching artefacts on disk."""
    payload = {
        "tickers": sorted(set(tickers)),
        "start": str(start),
        "end": str(end),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode())
    return digest.hexdigest()[:12]
