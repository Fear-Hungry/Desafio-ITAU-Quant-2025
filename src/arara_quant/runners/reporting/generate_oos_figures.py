#!/usr/bin/env python3
"""
Generate OOS validation figures from canonical nav_daily.csv (single source of truth).

Figures generated:
1. NAV cumulative (2020-01-02 to 2025-10-09 OOS period)
2. Drawdown underwater plot
3. Sharpe vs Return scatter (PRISM-R vs baselines)
4. Window-level metrics distribution
"""

from pathlib import Path

import matplotlib.pyplot as plt
from arara_quant.config import get_settings
from arara_quant.evaluation.plots.oos import (
    plot_daily_returns_dashboard,
    plot_nav_cumulative,
    plot_risk_return_scatter,
    plot_underwater_drawdown,
)
from arara_quant.reports.canonical import (
    ensure_output_dirs,
    load_baseline_metrics_oos,
    load_nav_daily,
    load_oos_consolidated_metrics,
    load_oos_period,
    subset_to_oos_period,
)

SETTINGS = get_settings()
ensure_output_dirs(SETTINGS)

REPO_ROOT = SETTINGS.project_root
CONFIG_DIR = SETTINGS.configs_dir
REPORTS_DIR = SETTINGS.reports_dir
WALKFORWARD_DIR = SETTINGS.walkforward_dir
FIGURES_DIR = SETTINGS.figures_dir

SAVED_PREFIX = "✓ Saved: "

def setup_style():
    """Setup matplotlib style."""
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (14, 7)
    plt.rcParams['font.size'] = 10

def _build_output_path(stem: str, suffix: str) -> Path:
    return FIGURES_DIR / f"{stem}_{suffix}.png"


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{SAVED_PREFIX}{path}")

def main():
    """Generate all OOS figures from canonical nav_daily.csv."""
    print("=" * 70)
    print("GENERATING OOS FIGURES (FROM nav_daily.csv - SINGLE SOURCE OF TRUTH)")
    print("=" * 70)

    setup_style()

    # Load configuration and data
    print("\nLoading configuration and data...")
    oos_period = load_oos_period(SETTINGS)
    df_nav = load_nav_daily(SETTINGS)
    df_oos = subset_to_oos_period(df_nav, oos_period, date_column="date")
    figure_suffix = oos_period.end.strftime("%Y%m%d")

    print(f"✓ OOS Period: {oos_period.start.date()} to {oos_period.end.date()}")
    print(f"✓ Data loaded: {len(df_oos)} trading days")

    # Load PRISM-R metrics for the baseline scatter plot
    metrics = load_oos_consolidated_metrics(SETTINGS)
    prism_sharpe = float(metrics.get("sharpe_ratio", 0.0))
    prism_return = float(metrics.get("annualized_return", 0.0)) * 100.0

    baselines_df = None
    try:
        baselines_df = load_baseline_metrics_oos(SETTINGS)
    except Exception:
        baselines_df = None

    # Generate figures via shared library helpers
    ax_nav = plot_nav_cumulative(df_oos, title="PRISM-R Out-of-Sample Daily NAV")
    _save_figure(ax_nav.figure, _build_output_path("oos_nav_cumulative", figure_suffix))

    ax_dd = plot_underwater_drawdown(df_oos, title="PRISM-R Underwater Drawdown")
    _save_figure(ax_dd.figure, _build_output_path("oos_drawdown_underwater", figure_suffix))

    ax_scatter = plot_risk_return_scatter(
        prism_return_pct=prism_return,
        prism_sharpe=prism_sharpe,
        baselines=baselines_df,
        title="Risk/Return: PRISM-R vs Baselines",
    )
    _save_figure(ax_scatter.figure, _build_output_path("oos_baseline_comparison", figure_suffix))

    fig_dist = plot_daily_returns_dashboard(df_oos)
    _save_figure(fig_dist, _build_output_path("oos_daily_distribution", figure_suffix))

    print("\n" + "=" * 70)
    print("✅ ALL FIGURES GENERATED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nFigures saved to: {FIGURES_DIR}/")

if __name__ == "__main__":
    main()
