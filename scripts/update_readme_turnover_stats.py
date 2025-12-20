#!/usr/bin/env python3
"""Legacy entry-point for README turnover stats updates.

This script was moved to ``scripts/reporting/update_readme_turnover_stats.py``.
It remains here as a thin wrapper to preserve CI and documentation references.
"""

from __future__ import annotations

from scripts.reporting.update_readme_turnover_stats import main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

