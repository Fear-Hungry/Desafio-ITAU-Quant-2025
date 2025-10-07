"""Cache simples para dados.

Hash de requests (tickers+janela) → nome de arquivo, utilitário para reuso.
"""

from __future__ import annotations
import hashlib
import json


def request_hash(tickers, start, end) -> str:
    payload = {"tickers": sorted(set(tickers)),
               "start": str(start), "end": str(end)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
