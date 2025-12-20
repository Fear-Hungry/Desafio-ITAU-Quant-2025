#!/usr/bin/env python3
"""Legacy entry-point for README turnover stats updates.

This script was moved to ``scripts/reporting/update_readme_turnover_stats.py``.
It remains here as a thin wrapper to preserve CI and documentation references.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_import_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"

    for candidate in (repo_root, src_root):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


_bootstrap_import_path()

from scripts.reporting.update_readme_turnover_stats import main  # noqa: E402


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
