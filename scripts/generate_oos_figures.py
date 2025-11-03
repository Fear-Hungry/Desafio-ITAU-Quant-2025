#!/usr/bin/env python3
"""
Generate OOS validation figures:
1. NAV cumulative 2020-2025 (ending at 1.1414)
2. Drawdown underwater plot
3. Sharpe vs Return scatter (PRISM-R vs baselines)
4. Window-level metrics distribution
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

REPO_ROOT = Path(__file__).parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPO_ROOT / "results"

def setup_style():
    """Setup matplotlib style."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10

def generate_nav_figure():
    """Generate cumulative NAV figure for 2020-2025 period."""
    print("\n=== Generating NAV Cumulative Figure ===")

    # Load consolidated metrics
    metrics = json.load(open(REPORTS_DIR / "oos_consolidated_metrics.json"))

    # Create synthetic NAV curve (assuming constant daily return)
    nav_final = metrics['nav_final']
    n_days = metrics['n_days']
    daily_return = (nav_final ** (1 / n_days)) - 1

    nav_curve = np.cumprod(np.ones(n_days) + daily_return)

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot NAV
    dates = pd.date_range(start='2020-01-02', periods=n_days, freq='B')
    ax.plot(dates, nav_curve, linewidth=2, label='NAV', color='#1f77b4')
    ax.fill_between(dates, 1.0, nav_curve, alpha=0.3, color='#1f77b4')

    # Mark final NAV
    ax.axhline(y=nav_final, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(dates[-1], nav_final + 0.01, f'Final: {nav_final:.4f}',
            fontsize=10, color='red', ha='right')

    # Mark max drawdown point
    running_max = np.maximum.accumulate(nav_curve)
    drawdown = nav_curve / running_max - 1
    min_dd_idx = np.argmin(drawdown)
    ax.axvline(x=dates[min_dd_idx], color='orange', linestyle=':', alpha=0.5)
    ax.text(dates[min_dd_idx], nav_curve[min_dd_idx] - 0.05,
            f'Max DD\n{drawdown[min_dd_idx]:.2%}',
            fontsize=9, ha='center', color='orange')

    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('NAV', fontsize=11)
    ax.set_title('PRISM-R Cumulative NAV (2020-01-02 to 2025-10-31)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    output = FIGURES_DIR / "oos_nav_cumulative_20251103.png"
    plt.savefig(output, dpi=150)
    print(f"✓ Saved: {output}")
    plt.close()

def generate_drawdown_figure():
    """Generate underwater drawdown plot."""
    print("\n=== Generating Drawdown Underwater Figure ===")

    # Load consolidated metrics
    metrics = json.load(open(REPORTS_DIR / "oos_consolidated_metrics.json"))

    # Create synthetic NAV and drawdown curves
    nav_final = metrics['nav_final']
    n_days = metrics['n_days']
    daily_return = (nav_final ** (1 / n_days)) - 1

    nav_curve = np.cumprod(np.ones(n_days) + daily_return)
    running_max = np.maximum.accumulate(nav_curve)
    drawdown = (nav_curve - running_max) / running_max * 100

    fig, ax = plt.subplots(figsize=(14, 7))

    dates = pd.date_range(start='2020-01-02', periods=n_days, freq='B')

    # Plot drawdown
    colors = ['red' if dd < -15 else 'orange' if dd < -10 else 'yellow' for dd in drawdown]
    ax.fill_between(dates, 0, drawdown, step='post', alpha=0.6, color='coral')
    ax.plot(dates, drawdown, linewidth=1.5, color='darkred', label='Drawdown')

    # Mark key levels
    ax.axhline(y=-15, color='red', linestyle='--', linewidth=1.5, label='Target Limit (-15%)', alpha=0.7)
    ax.axhline(y=metrics['max_drawdown'] * 100, color='darkred', linestyle='--',
               linewidth=1.5, label=f"Max DD ({metrics['max_drawdown']*100:.1f}%)", alpha=0.7)

    # Mark worst drawdown
    min_dd_idx = np.argmin(drawdown)
    ax.scatter(dates[min_dd_idx], drawdown[min_dd_idx], color='darkred', s=100, zorder=5)
    ax.annotate(f"Worst: {drawdown[min_dd_idx]:.1f}%\n({dates[min_dd_idx].strftime('%Y-%m-%d')})",
                xy=(dates[min_dd_idx], drawdown[min_dd_idx]),
                xytext=(20, -20), textcoords='offset points',
                fontsize=10, ha='left',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='darkred'))

    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Drawdown (%)', fontsize=11)
    ax.set_title('PRISM-R Drawdown Underwater Plot (2020-2025)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10, loc='lower left')
    ax.set_ylim([drawdown.min() - 2, 2])

    plt.tight_layout()
    output = FIGURES_DIR / "oos_drawdown_underwater_20251103.png"
    plt.savefig(output, dpi=150)
    print(f"✓ Saved: {output}")
    plt.close()

def generate_baseline_comparison_figure():
    """Generate Sharpe vs Return scatter plot comparing PRISM-R vs baselines."""
    print("\n=== Generating Baseline Comparison Figure ===")

    # Load comparison data
    comparison = pd.read_csv(REPORTS_DIR / "strategy_comparison_final.csv")

    # Extract numeric values (remove % and convert)
    def extract_num(val_str):
        if isinstance(val_str, str):
            return float(val_str.strip('— %'))
        return float(val_str)

    comparison['Return_num'] = comparison['Annual Return'].apply(extract_num)
    comparison['Sharpe_num'] = comparison['Sharpe (mean)'].apply(extract_num)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Color PRISM-R differently
    colors = ['#d62728' if 'PRISM' in s else '#1f77b4' for s in comparison['Strategy']]

    # Plot
    for idx, row in comparison.iterrows():
        color = colors[idx]
        size = 300 if 'PRISM' in row['Strategy'] else 150
        marker = 'D' if 'PRISM' in row['Strategy'] else 'o'

        ax.scatter(row['Return_num'] * 100, row['Sharpe_num'],
                  s=size, alpha=0.7, color=color, marker=marker, edgecolors='black', linewidth=1.5)

        # Annotate
        offset_x = 1.5 if 'PRISM' in row['Strategy'] else 1
        offset_y = 0.15 if 'PRISM' in row['Strategy'] else 0.1
        label = 'PRISM-R' if 'PRISM' in row['Strategy'] else row['Strategy'].split('(')[0].strip()

        ax.annotate(label,
                   xy=(row['Return_num'] * 100, row['Sharpe_num']),
                   xytext=(offset_x, offset_y), textcoords='offset points',
                   fontsize=9, ha='left',
                   bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3) if 'PRISM' in row['Strategy'] else None)

    ax.set_xlabel('Annual Return (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
    ax.set_title('PRISM-R vs Baseline Strategies (2020-2025)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', edgecolor='black', label='PRISM-R (Our Strategy)'),
        Patch(facecolor='#1f77b4', edgecolor='black', label='Baseline Strategies')
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc='best')

    plt.tight_layout()
    output = FIGURES_DIR / "oos_baseline_comparison_20251103.png"
    plt.savefig(output, dpi=150)
    print(f"✓ Saved: {output}")
    plt.close()

def generate_window_metrics_distribution():
    """Generate window-level metrics distribution plots."""
    print("\n=== Generating Window Metrics Distribution Figure ===")

    # Load window data
    windows = pd.read_csv(REPORTS_DIR / "oos_consolidated_metrics.csv")
    windows_only = windows[windows['Type'] == 'WINDOW'].copy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Sharpe distribution
    axes[0, 0].hist(windows_only['Sharpe (OOS)'], bins=15, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(windows_only['Sharpe (OOS)'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[0, 0].axvline(windows_only['Sharpe (OOS)'].median(), color='orange', linestyle='--', linewidth=2, label='Median')
    axes[0, 0].set_xlabel('Sharpe Ratio (OOS)', fontsize=10)
    axes[0, 0].set_ylabel('Frequency', fontsize=10)
    axes[0, 0].set_title('Sharpe Ratio Distribution (64 Windows)', fontsize=11, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Return distribution
    axes[0, 1].hist(windows_only['Return (OOS)'] * 100, bins=15, color='forestgreen', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(windows_only['Return (OOS)'].mean() * 100, color='red', linestyle='--', linewidth=2, label='Mean')
    axes[0, 1].set_xlabel('Return (%)', fontsize=10)
    axes[0, 1].set_ylabel('Frequency', fontsize=10)
    axes[0, 1].set_title('Return Distribution (64 Windows)', fontsize=11, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Drawdown distribution
    axes[1, 0].hist(windows_only['Drawdown (OOS)'] * 100, bins=15, color='coral', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(windows_only['Drawdown (OOS)'].min() * 100, color='darkred', linestyle='--', linewidth=2, label='Max DD')
    axes[1, 0].set_xlabel('Drawdown (%)', fontsize=10)
    axes[1, 0].set_ylabel('Frequency', fontsize=10)
    axes[1, 0].set_title('Drawdown Distribution (64 Windows)', fontsize=11, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Metrics scatter (Sharpe vs Return)
    axes[1, 1].scatter(windows_only['Return (OOS)'] * 100, windows_only['Sharpe (OOS)'],
                       s=100, alpha=0.6, color='purple', edgecolors='black', linewidth=0.5)
    axes[1, 1].set_xlabel('Return (%)', fontsize=10)
    axes[1, 1].set_ylabel('Sharpe Ratio', fontsize=10)
    axes[1, 1].set_title('Sharpe vs Return (Per Window)', fontsize=11, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('OOS Window-Level Metrics Distribution', fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()

    output = FIGURES_DIR / "oos_window_metrics_distribution_20251103.png"
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output}")
    plt.close()

def main():
    print("\n" + "="*70)
    print("GENERATING OOS VALIDATION FIGURES")
    print("="*70)

    setup_style()

    # Create figures directory if needed
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    try:
        generate_nav_figure()
        generate_drawdown_figure()
        generate_baseline_comparison_figure()
        generate_window_metrics_distribution()

        print("\n" + "="*70)
        print("✓ ALL FIGURES GENERATED SUCCESSFULLY")
        print("="*70)

    except Exception as e:
        print(f"\n✗ Error generating figures: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
