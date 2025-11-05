#!/usr/bin/env python3
"""
Run the canonical walk-forward backtest using the official CLI pipeline.

This wrapper delegates all heavy lifting to ``arara_quant.backtesting.run_backtest``
and reuses the same reporting helpers leveraged by ``poetry run arara-quant``.
The goal is to avoid divergence between research scripts and production code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from arara_quant.backtesting import run_backtest
from arara_quant.cli import _generate_wf_report
from arara_quant.config import get_settings

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "optimizer_example.yaml"
DEFAULT_OOS_CONFIG = REPO_ROOT / "configs" / "oos_period.yaml"
DEFAULT_WF_DIR = REPO_ROOT / "reports" / "walkforward"


def _read_oos_period(path: Path) -> tuple[str | None, str | None]:
    if not path.exists():
        return None, None
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    block = data.get("oos_evaluation", {}) if isinstance(data, dict) else {}
    return block.get("start_date"), block.get("end_date")


def _print_header(config_path: Path, start: str | None, end: str | None) -> None:
    print("=" * 80)
    print("  PRISM-R — Canonical Walk-Forward Backtest")
    print("=" * 80)
    print(f"Config file : {config_path}")
    if start and end:
        print(f"OOS period  : {start} → {end}")
    print()


def _print_metrics(result) -> None:
    metrics = result.metrics
    if metrics is None:
        print("No metrics available (dry-run?).")
        return
    print("=== Performance Summary ===")
    print(f"Total Return        : {metrics.total_return:+.2%}")
    print(f"Annualized Return   : {metrics.annualized_return:+.2%}")
    print(f"Annualized Vol      : {metrics.annualized_volatility:.2%}")
    print(f"Sharpe Ratio        : {metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown        : {metrics.max_drawdown:.2%}")
    avg_turnover = None
    if result.walkforward_summary is not None:
        avg_turnover = result.walkforward_summary.avg_turnover
    elif result.trades is not None and not result.trades.empty:
        avg_turnover = float(result.trades["turnover"].astype(float).mean())
    if avg_turnover is not None:
        print(f"Turnover (avg)      : {avg_turnover:.2%}")
    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the canonical walk-forward backtest using official pipeline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to backtest configuration YAML (default: configs/optimizer_example.yaml)",
    )
    parser.add_argument(
        "--no-wf-report",
        dest="wf_report",
        action="store_false",
        help="Skip generating the walk-forward report artefacts.",
    )
    parser.set_defaults(wf_report=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_WF_DIR,
        help="Directory to store walk-forward artefacts (default: reports/walkforward).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the configuration without executing the simulation.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    if not config_path.exists():
        parser.error(f"Config file not found: {config_path}")

    start, end = _read_oos_period(DEFAULT_OOS_CONFIG)
    _print_header(config_path, start, end)

    settings = get_settings()
    result = run_backtest(config_path, settings=settings, dry_run=args.dry_run)

    if args.dry_run:
        print("Dry-run completed. Use --dry-run/--no-dry-run to switch modes.")
        return 0

    _print_metrics(result)

    if args.wf_report:
        _generate_wf_report(result, output_dir=str(args.output_dir))

    print("Backtest finished ✔")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
