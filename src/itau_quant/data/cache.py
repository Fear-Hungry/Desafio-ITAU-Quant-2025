"""Lightweight helpers for request hashing."""

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
