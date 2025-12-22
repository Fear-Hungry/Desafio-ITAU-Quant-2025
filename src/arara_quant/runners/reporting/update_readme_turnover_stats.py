#!/usr/bin/env python3
"""Update README Table 5.1 with turnover distribution stats.

This script is a thin wrapper around :func:`arara_quant.reports.update_readme_turnover_stats`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from arara_quant.config import get_settings
from arara_quant.reports.generators import update_readme_turnover_stats
from arara_quant.reports.canonical import resolve_oos_config_path


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()

    ap = argparse.ArgumentParser(
        description="Update README table turnover columns from canonical artefacts."
    )
    ap.add_argument(
        "--readme",
        type=Path,
        default=settings.project_root / "README.md",
        help="Path to README.md to update in-place.",
    )
    ap.add_argument(
        "--summary",
        type=Path,
        default=settings.results_dir / "oos_canonical" / "turnover_dist_stats.csv",
        help="CSV with baseline turnover distribution stats.",
    )
    ap.add_argument(
        "--per-window-prism",
        type=Path,
        default=settings.walkforward_dir / "per_window_results.csv",
        help="CSV with PRISM-R per-window results (fallback when trades.csv missing).",
    )
    ap.add_argument(
        "--prism-trades",
        type=Path,
        default=settings.walkforward_dir / "trades.csv",
        help="CSV with PRISM-R trade-level turnover (preferred source when present).",
    )
    ap.add_argument(
        "--oos-config",
        type=Path,
        default=resolve_oos_config_path(settings),
        help="YAML with OOS start/end dates (defaults to configs/oos_period.yaml).",
    )
    ap.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Overwrite existing values (not only placeholders).",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    settings = get_settings()
    updated = update_readme_turnover_stats(
        settings=settings,
        readme_path=args.readme,
        baseline_summary_csv=args.summary,
        prism_per_window_csv=args.per_window_prism,
        prism_trades_csv=args.prism_trades,
        oos_config_path=args.oos_config,
        force_overwrite=args.force_overwrite,
    )
    print(f"Updated rows: {updated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

