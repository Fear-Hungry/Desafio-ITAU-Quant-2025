#!/usr/bin/env python3
"""
Generate comprehensive walk-forward analysis figure with 4 subplots:
1. Parameter Evolution (lambda, positions)
2. Sharpe Ratio per window
3. Consistency metrics (hit rate, vol of returns)
4. Turnover and Costs

Output: outputs/reports/figures/walkforward_analysis_YYYYMMDD.png
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from arara_quant.config import get_settings
from arara_quant.evaluation.plots.walkforward import plot_walkforward_summary
from arara_quant.evaluation.walkforward_report import compute_wf_summary_stats
from arara_quant.reports.canonical import load_walkforward_windows_raw


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()

    ap = argparse.ArgumentParser(
        description="Generate the canonical walk-forward summary figure."
    )
    ap.add_argument(
        "--raw-csv",
        type=Path,
        default=settings.walkforward_dir / "per_window_results_raw.csv",
        help="Path to per_window_results_raw.csv (default: outputs/reports/walkforward/per_window_results_raw.csv).",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=settings.figures_dir
        / f"walkforward_analysis_{datetime.now().strftime('%Y%m%d')}.png",
        help="Output PNG path (default: outputs/reports/figures/walkforward_analysis_YYYYMMDD.png).",
    )
    ap.add_argument(
        "--style",
        default="seaborn-v0_8-darkgrid",
        help="Matplotlib style to apply (default: seaborn-v0_8-darkgrid).",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    settings.figures_dir.mkdir(parents=True, exist_ok=True)

    if args.style:
        try:
            plt.style.use(args.style)
        except Exception:
            plt.style.use("default")

    if args.raw_csv != settings.walkforward_dir / "per_window_results_raw.csv":
        df = pd.read_csv(args.raw_csv)
    else:
        df = load_walkforward_windows_raw(settings)

    if df.empty:
        raise SystemExit("per_window_results_raw.csv is empty")

    try:
        summary = compute_wf_summary_stats(df)
        print("=== Walk-Forward Summary (raw split_metrics) ===")
        print(f"Windows: {summary.n_windows}")
        print(f"Avg Sharpe: {summary.avg_sharpe:.3f}")
        print(f"Success Rate: {summary.success_rate:.1%}")
        print(f"Avg Turnover: {summary.avg_turnover:.2%}")
        print(f"Avg Cost: {summary.avg_cost:.4f}")
        print()
    except Exception:
        pass

    fig = plot_walkforward_summary(df, figsize=(16, 10))
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
