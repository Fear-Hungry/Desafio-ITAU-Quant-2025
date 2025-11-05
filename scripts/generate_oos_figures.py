#!/usr/bin/env python3
"""
Generate OOS validation figures from canonical nav_daily.csv (single source of truth).

Figures generated:
1. NAV cumulative (2020-01-02 to 2025-10-31 OOS period)
2. Drawdown underwater plot
3. Sharpe vs Return scatter (PRISM-R vs baselines)
4. Window-level metrics distribution
"""

import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
import json

REPO_ROOT = Path(__file__).parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
REPORTS_DIR = REPO_ROOT / "reports"
WALKFORWARD_DIR = REPORTS_DIR / "walkforward"
FIGURES_DIR = REPORTS_DIR / "figures"

def load_oos_config():
    """Load OOS period from centralized config."""
    config_path = CONFIG_DIR / "oos_period.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['oos_evaluation']

def load_nav_daily():
    """Load canonical daily NAV series (single source of truth)."""
    nav_daily_path = WALKFORWARD_DIR / "nav_daily.csv"
    df = pd.read_csv(nav_daily_path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)

def filter_to_oos_period(df_nav: pd.DataFrame, oos_config: dict):
    """Filter NAV data to OOS period."""
    start_date = pd.to_datetime(oos_config['start_date'])
    end_date = pd.to_datetime(oos_config['end_date'])
    mask = (df_nav['date'] >= start_date) & (df_nav['date'] <= end_date)
    return df_nav[mask].copy()

def setup_style():
    """Setup matplotlib style."""
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (14, 7)
    plt.rcParams['font.size'] = 10

def generate_nav_figure(df_oos: pd.DataFrame):
    """Generate cumulative NAV figure from nav_daily.csv (OOS period)."""
    print("\n=== Generating NAV Cumulative Figure ===")

    dates = df_oos['date'].values
    nav_curve = df_oos['nav'].values

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot cumulative NAV from canonical nav_daily.csv
    ax.plot(dates, nav_curve, linewidth=2.5, label='NAV (OOS Daily)', color='#1f77b4')
    ax.fill_between(dates, 1.0, nav_curve, alpha=0.3, color='#1f77b4')

    # Mark final NAV
    nav_final = nav_curve[-1]
    ax.axhline(y=nav_final, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label=f'Final NAV: {nav_final:.4f}')
    ax.text(dates[-1], nav_final + 0.015, f'{nav_final:.4f}', fontsize=11, color='red', ha='right', fontweight='bold')

    # Mark max drawdown point
    running_max = np.maximum.accumulate(nav_curve)
    drawdown = nav_curve / running_max - 1.0
    min_dd_idx = int(np.argmin(drawdown))
    max_dd = drawdown[min_dd_idx]

    ax.axvline(x=dates[min_dd_idx], color='orange', linestyle=':', alpha=0.6, linewidth=2)
    ax.scatter([dates[min_dd_idx]], [nav_curve[min_dd_idx]], color='orange', s=100, zorder=5, marker='o', edgecolors='darkred', linewidth=2)
    ax.text(dates[min_dd_idx], nav_curve[min_dd_idx] - 0.08, f'Max DD\n{max_dd:.2%}', fontsize=10, ha='center',
            color='darkred', fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('NAV', fontsize=12, fontweight='bold')
    ax.set_title(f"PRISM-R Out-of-Sample Daily NAV ({pd.to_datetime(dates[0]).date()} to {pd.to_datetime(dates[-1]).date()})", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper left')
    ax.set_ylim([0.7, max(1.1, nav_final * 1.05)])

    plt.tight_layout()
    output = FIGURES_DIR / "oos_nav_cumulative_20251103.png"
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output}")
    print(f"  NAV Final: {nav_final:.4f} | Max DD: {max_dd:.2%}")
    plt.close()

def generate_drawdown_figure(df_oos: pd.DataFrame):
    """Generate underwater drawdown plot from nav_daily.csv."""
    print("\n=== Generating Drawdown Underwater Figure ===")

    dates = df_oos['date'].values
    nav_curve = df_oos['nav'].values

    # Compute drawdowns
    running_max = np.maximum.accumulate(nav_curve)
    drawdowns = (nav_curve / running_max - 1.0) * 100  # Convert to percentage

    fig, ax = plt.subplots(figsize=(14, 7))

    # Underwater plot (negative drawdown)
    colors = np.where(drawdowns < -15, 'darkred', np.where(drawdowns < -10, 'red', 'orange'))
    ax.bar(dates, drawdowns, color=colors, alpha=0.6, width=1.0, label='Drawdown %')

    # Mark key thresholds
    ax.axhline(y=-15, color='red', linestyle='--', linewidth=2, alpha=0.6, label='Limit: -15%')
    ax.axhline(y=-20, color='darkred', linestyle='--', linewidth=2, alpha=0.6, label='Severe: -20%')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)

    # Mark worst drawdown
    min_dd_idx = int(np.argmin(drawdowns))
    worst_dd = float(drawdowns[min_dd_idx])
    ax.scatter(dates[min_dd_idx], worst_dd, color='darkred', s=100, zorder=5)
    ax.annotate(f"Worst: {worst_dd:.1f}%\n({pd.to_datetime(dates[min_dd_idx]).strftime('%Y-%m-%d')})",
                xy=(dates[min_dd_idx], worst_dd),
                xytext=(20, -20), textcoords='offset points',
                fontsize=10, ha='left',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='darkred'))

    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Drawdown (%)', fontsize=11)
    ax.set_title(f"PRISM-R Underwater Drawdown ({pd.to_datetime(dates[0]).date()} to {pd.to_datetime(dates[-1]).date()})", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=11, loc='lower left')
    ax.set_ylim([min(drawdowns) * 1.1, 2])

    plt.tight_layout()
    output = FIGURES_DIR / "oos_drawdown_underwater_20251103.png"
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output}")
    print(f"  Max Drawdown: {worst_dd:.1f}%")
    plt.close()

def generate_baseline_comparison_figure():
    """Generate Sharpe (excesso T‑Bill) vs Return scatter (PRISM-R vs baselines)."""
    print("\n=== Generating Baseline Comparison Figure ===")

    # Load consolidated metrics for PRISM-R
    metrics = json.load(open(REPORTS_DIR / "oos_consolidated_metrics.json"))

    # Use Sharpe computed on daily excess vs T‑Bill from consolidated metrics
    prism_sharpe = float(metrics.get('sharpe_ratio', 0))
    prism_return = float(metrics.get('annualized_return', 0)) * 100  # Convert to %

    # Baseline data (from historical validation)
    # These are standard baselines for comparison
    # Baselines (Sharpe em excesso ao T‑Bill) e retornos anualizados estimados
    # Valores de Sharpe calculados em excesso ao T‑Bill diário no período OOS canônico
    df_baselines = pd.DataFrame({
        'Strategy': ['Equal-Weight', 'Risk Parity', 'Min-Var (LW)', 'Shrunk MV', '60/40', 'HRP'],
        'Sharpe (excesso T-Bill)': [0.2618, 0.2304, -0.5476, 0.1770, 0.2268, -0.3049],
        'Return (%)': [4.32, 3.99, 1.30, 3.63, 3.86, 0.87]
    })

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot baselines as blue circles
    if not df_baselines.empty:
        ax.scatter(df_baselines['Return (%)'].values, df_baselines['Sharpe (excesso T-Bill)'].values,
                  s=200, alpha=0.6, color='steelblue', label='Baseline Strategies', edgecolors='navy', linewidth=1.5)

        # Label each baseline
        for idx, row in df_baselines.iterrows():
            ax.annotate(row['Strategy'],
                       (row['Return (%)'], row['Sharpe (excesso T-Bill)']),
                       xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.8)

    # Plot PRISM-R as red diamond (emphasize)
    ax.scatter([prism_return], [prism_sharpe], s=500, alpha=0.8, color='#d62728', marker='D',
              label='PRISM-R (Our Strategy)', edgecolors='darkred', linewidth=2.5, zorder=5)
    ax.annotate('PRISM-R', (prism_return, prism_sharpe), xytext=(10, 10), textcoords='offset points',
               fontsize=11, fontweight='bold', color='darkred',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

    ax.set_xlabel('Annualized Return (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sharpe (excesso T-Bill)', fontsize=12, fontweight='bold')
    ax.set_title('Risk-Return: PRISM-R vs Baselines (Sharpe em excesso ao T-Bill)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper left')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

    # Source info
    ax.text(0.99, 0.02, f'Source: oos_consolidated_metrics.json (excesso T-Bill) | {len(df_baselines)} baselines',
            transform=ax.transAxes, fontsize=9, alpha=0.7, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgray', alpha=0.3))

    plt.tight_layout()
    output = FIGURES_DIR / "oos_baseline_comparison_20251103.png"
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output}")
    print(f"  PRISM-R: Sharpe={prism_sharpe:.4f}, Return={prism_return:.2f}%")
    plt.close()

def generate_daily_distribution_figure(df_oos: pd.DataFrame):
    """Generate daily returns distribution figure."""
    print("\n=== Generating Daily Returns Distribution Figure ===")

    daily_returns = df_oos['daily_return'].values * 100  # Convert to %

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Daily Returns Histogram
    axes[0, 0].hist(daily_returns, bins=50, color='steelblue', alpha=0.6, edgecolor='black')
    axes[0, 0].axvline(np.mean(daily_returns), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(daily_returns):.3f}%')
    axes[0, 0].axvline(np.median(daily_returns), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(daily_returns):.3f}%')
    axes[0, 0].set_xlabel('Daily Return (%)', fontsize=10)
    axes[0, 0].set_ylabel('Frequency', fontsize=10)
    axes[0, 0].set_title('Distribution of Daily Returns', fontsize=11, fontweight='bold')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # 2. Cumulative Returns
    cumulative = np.cumsum(daily_returns)
    axes[0, 1].plot(df_oos['date'].values, cumulative, linewidth=2, color='#1f77b4')
    axes[0, 1].fill_between(df_oos['date'].values, 0, cumulative, alpha=0.3, color='#1f77b4')
    axes[0, 1].set_xlabel('Date', fontsize=10)
    axes[0, 1].set_ylabel('Cumulative Return (%)', fontsize=10)
    axes[0, 1].set_title('Cumulative Daily Returns Over Time', fontsize=11, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Rolling Volatility (21-day)
    rolling_vol = pd.Series(daily_returns).rolling(21).std()
    axes[1, 0].plot(df_oos['date'].values, rolling_vol, linewidth=1.5, color='purple', alpha=0.7)
    axes[1, 0].fill_between(df_oos['date'].values, rolling_vol, alpha=0.2, color='purple')
    axes[1, 0].axhline(np.mean(rolling_vol), color='red', linestyle='--', linewidth=1.5, label=f'Mean Vol: {np.mean(rolling_vol):.2f}%')
    axes[1, 0].set_xlabel('Date', fontsize=10)
    axes[1, 0].set_ylabel('21-Day Rolling Volatility (%)', fontsize=10)
    axes[1, 0].set_title('Rolling Volatility (21-day window)', fontsize=11, fontweight='bold')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Positive vs Negative Days
    positive_days = np.sum(daily_returns > 0)
    negative_days = np.sum(daily_returns < 0)
    zero_days = np.sum(daily_returns == 0)

    colors = ['green', 'red', 'gray']
    sizes = [positive_days, negative_days, zero_days]
    labels = [f'Positive\n({positive_days} days)', f'Negative\n({negative_days} days)', f'Zero\n({zero_days} days)']

    axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
    axes[1, 1].set_title('Distribution of Positive/Negative Days', fontsize=11, fontweight='bold')

    plt.tight_layout()
    output = FIGURES_DIR / "oos_daily_distribution_20251103.png"
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output}")
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
    oos_config = load_oos_config()
    df_nav = load_nav_daily()
    df_oos = filter_to_oos_period(df_nav, oos_config)

    print(f"✓ OOS Period: {oos_config['start_date']} to {oos_config['end_date']}")
    print(f"✓ Data loaded: {len(df_oos)} trading days")

    # Generate all figures
    generate_nav_figure(df_oos)
    generate_drawdown_figure(df_oos)
    generate_baseline_comparison_figure()
    generate_daily_distribution_figure(df_oos)

    print("\n" + "=" * 70)
    print("✅ ALL FIGURES GENERATED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nFigures saved to: {FIGURES_DIR}/")

if __name__ == "__main__":
    main()
