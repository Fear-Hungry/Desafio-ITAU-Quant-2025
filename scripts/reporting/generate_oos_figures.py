#!/usr/bin/env python3
"""
Generate OOS validation figures from canonical nav_daily.csv (single source of truth).

Figures generated:
1. NAV cumulative (2020-01-02 to 2025-10-09 OOS period)
2. Drawdown underwater plot
3. Sharpe vs Return scatter (PRISM-R vs baselines)
4. Window-level metrics distribution
"""

from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arara_quant.config import get_settings
from arara_quant.reports.canonical import (
    ensure_output_dirs,
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
TEXTCOORDS_OFFSET_POINTS = "offset points"
COL_SHARPE_EXCESS_TBILL = "Sharpe (excesso T-Bill)"
COL_RETURN_PCT = "Return (%)"
DAYS_LABEL_TEMPLATE = "{kind}\n({count} days)"


def _palette():
    colors = plt.rcParams.get("axes.prop_cycle", None)
    if colors is None:
        return cycle([None])
    return cycle(colors.by_key().get("color", [None]))

def setup_style():
    """Setup matplotlib style."""
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (14, 7)
    plt.rcParams['font.size'] = 10

def _build_output_path(stem: str, suffix: str) -> Path:
    """Return standardized figure path for a given suffix."""

    return FIGURES_DIR / f"{stem}_{suffix}.png"


def generate_nav_figure(df_oos: pd.DataFrame, suffix: str):
    """Generate cumulative NAV figure from nav_daily.csv (OOS period)."""
    print("\n=== Generating NAV Cumulative Figure ===")

    dates = df_oos['date'].values
    nav_curve = df_oos['nav'].values

    _, ax = plt.subplots(figsize=(14, 7))
    palette = _palette()
    nav_color = next(palette)
    accent_color = next(palette)

    # Plot cumulative NAV from canonical nav_daily.csv
    ax.plot(dates, nav_curve, linewidth=2.5, label='NAV (OOS Daily)', color=nav_color)
    ax.fill_between(dates, 1.0, nav_curve, alpha=0.3, color=nav_color)

    # Mark final NAV
    nav_final = nav_curve[-1]
    ax.axhline(y=nav_final, color=accent_color, linestyle='--', linewidth=1.5, alpha=0.6, label=f'Final NAV: {nav_final:.4f}')
    ax.text(
        dates[-1],
        nav_final + 0.015,
        f'{nav_final:.4f}',
        fontsize=11,
        color=accent_color,
        ha='right',
        fontweight='bold',
    )

    # Mark max drawdown point
    running_max = np.maximum.accumulate(nav_curve)
    drawdown = nav_curve / running_max - 1.0
    min_dd_idx = int(np.argmin(drawdown))
    max_dd = drawdown[min_dd_idx]

    dd_color = next(palette)
    ax.axvline(x=dates[min_dd_idx], color=dd_color, linestyle=':', alpha=0.6, linewidth=2)
    ax.scatter(
        [dates[min_dd_idx]],
        [nav_curve[min_dd_idx]],
        color=dd_color,
        s=100,
        zorder=5,
        marker='o',
        edgecolors=accent_color,
        linewidth=2,
    )
    ax.text(
        dates[min_dd_idx],
        nav_curve[min_dd_idx] - 0.08,
        f'Max DD\n{max_dd:.2%}',
        fontsize=10,
        ha='center',
        color=accent_color,
        fontweight='bold',
        bbox={'boxstyle': 'round,pad=0.3', 'facecolor': 'white', 'alpha': 0.3},
    )

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('NAV', fontsize=12, fontweight='bold')
    ax.set_title(f"PRISM-R Out-of-Sample Daily NAV ({pd.to_datetime(dates[0]).date()} to {pd.to_datetime(dates[-1]).date()})", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper left')
    ax.set_ylim([0.7, max(1.1, nav_final * 1.05)])

    plt.tight_layout()
    output = _build_output_path("oos_nav_cumulative", suffix)
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"{SAVED_PREFIX}{output}")
    print(f"  NAV Final: {nav_final:.4f} | Max DD: {max_dd:.2%}")
    plt.close()


def generate_drawdown_figure(df_oos: pd.DataFrame, suffix: str):
    """Generate underwater drawdown plot from nav_daily.csv."""
    print("\n=== Generating Drawdown Underwater Figure ===")

    dates = df_oos['date'].values
    nav_curve = df_oos['nav'].values

    # Compute drawdowns
    running_max = np.maximum.accumulate(nav_curve)
    drawdowns = (nav_curve / running_max - 1.0) * 100

    palette = _palette()
    bar_color = next(palette)
    threshold_color = next(palette)
    worst_color = next(palette)

    _, ax = plt.subplots(figsize=(14, 7))

    # Underwater plot (negative drawdown)
    ax.bar(dates, drawdowns, color=bar_color, alpha=0.6, width=1.0, label='Drawdown %')

    # Mark key thresholds
    ax.axhline(y=-15, color=threshold_color, linestyle='--', linewidth=2, alpha=0.6, label='Limit: -15%')
    ax.axhline(y=-20, color=worst_color, linestyle='--', linewidth=2, alpha=0.6, label='Severe: -20%')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)

    # Mark worst drawdown
    min_dd_idx = int(np.argmin(drawdowns))
    worst_dd = float(drawdowns[min_dd_idx])
    ax.scatter(dates[min_dd_idx], worst_dd, color=worst_color, s=100, zorder=5)
    ax.annotate(f"Worst: {worst_dd:.1f}%\n({pd.to_datetime(dates[min_dd_idx]).strftime('%Y-%m-%d')})",
                xy=(dates[min_dd_idx], worst_dd),
                xytext=(20, -20), textcoords=TEXTCOORDS_OFFSET_POINTS,
                fontsize=10, ha='left',
                bbox={'boxstyle': 'round,pad=0.5', 'fc': 'white', 'alpha': 0.5},
                arrowprops={'arrowstyle': '->', 'connectionstyle': 'arc3,rad=0', 'color': worst_color})

    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Drawdown (%)', fontsize=11)
    ax.set_title(f"PRISM-R Underwater Drawdown ({pd.to_datetime(dates[0]).date()} to {pd.to_datetime(dates[-1]).date()})", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=11, loc='lower left')
    ax.set_ylim([min(drawdowns) * 1.1, 2])

    plt.tight_layout()
    output = _build_output_path("oos_drawdown_underwater", suffix)
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"{SAVED_PREFIX}{output}")
    print(f"  Max Drawdown: {worst_dd:.1f}%")
    plt.close()


def generate_baseline_comparison_figure(suffix: str):
    """Generate Sharpe (excesso T‑Bill) vs Return scatter (PRISM-R vs baselines)."""
    print("\n=== Generating Baseline Comparison Figure ===")

    # Load consolidated metrics for PRISM-R
    metrics = load_oos_consolidated_metrics(SETTINGS)

    # Use Sharpe computed on daily excess vs T‑Bill from consolidated metrics
    prism_sharpe = float(metrics.get('sharpe_ratio', 0))
    prism_return = float(metrics.get('annualized_return', 0)) * 100  # Convert to %

    # Baseline data (from historical validation)
    # These are standard baselines for comparison
    # Baselines (Sharpe em excesso ao T‑Bill) e retornos anualizados estimados
    # Valores de Sharpe calculados em excesso ao T‑Bill diário no período OOS canônico
    df_baselines = pd.DataFrame({
        'Strategy': ['Equal-Weight', 'Risk Parity', 'Min-Var (LW)', 'Shrunk MV', '60/40', 'HRP'],
        COL_SHARPE_EXCESS_TBILL: [0.2618, 0.2304, -0.5476, 0.1770, 0.2268, -0.3049],
        COL_RETURN_PCT: [4.32, 3.99, 1.30, 3.63, 3.86, 0.87]
    })

    _, ax = plt.subplots(figsize=(12, 8))
    palette = _palette()
    baseline_color = next(palette)
    prism_color = next(palette)

    # Plot baselines as blue circles
    if not df_baselines.empty:
        ax.scatter(
            df_baselines[COL_RETURN_PCT].values,
            df_baselines[COL_SHARPE_EXCESS_TBILL].values,
            s=200,
            alpha=0.6,
            color=baseline_color,
            label='Baseline Strategies',
            edgecolors=baseline_color,
            linewidth=1.5,
        )

        # Label each baseline
        for _, row in df_baselines.iterrows():
            ax.annotate(row['Strategy'],
                       (row[COL_RETURN_PCT], row[COL_SHARPE_EXCESS_TBILL]),
                       xytext=(5, 5), textcoords=TEXTCOORDS_OFFSET_POINTS, fontsize=9, alpha=0.8)

    # Plot PRISM-R as red diamond (emphasize)
    ax.scatter(
        [prism_return],
        [prism_sharpe],
        s=500,
        alpha=0.8,
        color=prism_color,
        marker='D',
        label='PRISM-R (Our Strategy)',
        edgecolors=prism_color,
        linewidth=2.5,
        zorder=5,
    )
    ax.annotate('PRISM-R', (prism_return, prism_sharpe), xytext=(10, 10), textcoords=TEXTCOORDS_OFFSET_POINTS,
               fontsize=11, fontweight='bold', color=prism_color,
               bbox={'boxstyle': 'round,pad=0.3', 'facecolor': 'white', 'alpha': 0.6})

    ax.set_xlabel('Annualized Return (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel(COL_SHARPE_EXCESS_TBILL, fontsize=12, fontweight='bold')
    ax.set_title('Risk-Return: PRISM-R vs Baselines (Sharpe em excesso ao T-Bill)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper left')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

    # Source info
    ax.text(0.99, 0.02, f'Source: oos_consolidated_metrics.json (excesso T-Bill) | {len(df_baselines)} baselines',
            transform=ax.transAxes, fontsize=9, alpha=0.7, va='bottom', ha='right',
            bbox={'boxstyle': 'round,pad=0.4', 'facecolor': 'lightgray', 'alpha': 0.3})

    plt.tight_layout()
    output = _build_output_path("oos_baseline_comparison", suffix)
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"{SAVED_PREFIX}{output}")
    print(f"  PRISM-R: Sharpe={prism_sharpe:.4f}, Return={prism_return:.2f}%")
    plt.close()


def generate_daily_distribution_figure(df_oos: pd.DataFrame, suffix: str):
    """Generate daily returns distribution figure."""
    print("\n=== Generating Daily Returns Distribution Figure ===")

    daily_returns = df_oos['daily_return'].values * 100  # Convert to %
    palette = _palette()
    hist_color = next(palette)
    cumulative_color = next(palette)
    vol_color = next(palette)
    pie_colors = [next(palette) for _ in range(3)]

    _, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Daily Returns Histogram
    axes[0, 0].hist(daily_returns, bins=50, color=hist_color, alpha=0.6, edgecolor='black')
    mean_color = next(palette)
    median_color = next(palette)
    axes[0, 0].axvline(np.mean(daily_returns), color=mean_color, linestyle='--', linewidth=2, label=f'Mean: {np.mean(daily_returns):.3f}%')
    axes[0, 0].axvline(np.median(daily_returns), color=median_color, linestyle='--', linewidth=2, label=f'Median: {np.median(daily_returns):.3f}%')
    axes[0, 0].set_xlabel('Daily Return (%)', fontsize=10)
    axes[0, 0].set_ylabel('Frequency', fontsize=10)
    axes[0, 0].set_title('Distribution of Daily Returns', fontsize=11, fontweight='bold')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # 2. Cumulative Returns
    cumulative = np.cumsum(daily_returns)
    axes[0, 1].plot(df_oos['date'].values, cumulative, linewidth=2, color=cumulative_color)
    axes[0, 1].fill_between(df_oos['date'].values, 0, cumulative, alpha=0.3, color=cumulative_color)
    axes[0, 1].set_xlabel('Date', fontsize=10)
    axes[0, 1].set_ylabel('Cumulative Return (%)', fontsize=10)
    axes[0, 1].set_title('Cumulative Daily Returns Over Time', fontsize=11, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Rolling Volatility (21-day)
    rolling_vol = pd.Series(daily_returns).rolling(21).std()
    axes[1, 0].plot(df_oos['date'].values, rolling_vol, linewidth=1.5, color=vol_color, alpha=0.7)
    axes[1, 0].fill_between(df_oos['date'].values, rolling_vol, alpha=0.2, color=vol_color)
    axes[1, 0].axhline(np.mean(rolling_vol), color=next(palette), linestyle='--', linewidth=1.5, label=f'Mean Vol: {np.mean(rolling_vol):.2f}%')
    axes[1, 0].set_xlabel('Date', fontsize=10)
    axes[1, 0].set_ylabel('21-Day Rolling Volatility (%)', fontsize=10)
    axes[1, 0].set_title('Rolling Volatility (21-day window)', fontsize=11, fontweight='bold')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Positive vs Negative Days
    positive_days = np.sum(daily_returns > 0)
    negative_days = np.sum(daily_returns < 0)
    zero_days = np.sum(daily_returns == 0)

    sizes = [positive_days, negative_days, zero_days]
    labels = [
        DAYS_LABEL_TEMPLATE.format(kind='Positive', count=positive_days),
        DAYS_LABEL_TEMPLATE.format(kind='Negative', count=negative_days),
        DAYS_LABEL_TEMPLATE.format(kind='Zero', count=zero_days),
    ]

    axes[1, 1].pie(sizes, labels=labels, colors=pie_colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
    axes[1, 1].set_title('Distribution of Positive/Negative Days', fontsize=11, fontweight='bold')

    plt.tight_layout()
    output = _build_output_path("oos_daily_distribution", suffix)
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"{SAVED_PREFIX}{output}")
    print(f"  Daily stats: Mean={np.mean(daily_returns):.3f}%, Std={np.std(daily_returns):.3f}%, Win rate={(positive_days/len(daily_returns)*100):.1f}%")
    plt.close()

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

    # Generate all figures
    generate_nav_figure(df_oos, figure_suffix)
    generate_drawdown_figure(df_oos, figure_suffix)
    generate_baseline_comparison_figure(figure_suffix)
    generate_daily_distribution_figure(df_oos, figure_suffix)

    print("\n" + "=" * 70)
    print("✅ ALL FIGURES GENERATED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nFigures saved to: {FIGURES_DIR}/")

if __name__ == "__main__":
    main()
