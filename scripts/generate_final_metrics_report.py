#!/usr/bin/env python3
"""
Generate final metrics report comparing PRISM-R strategy with baselines.
"""

import pandas as pd
from pathlib import Path
import json

REPO_ROOT = Path(__file__).parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
RESULTS_DIR = REPO_ROOT / "results"

def generate_comparison_table():
    """Generate comprehensive comparison table with PRISM-R vs baselines."""

    # Load PRISM-R metrics
    with open(REPORTS_DIR / "oos_consolidated_metrics.json") as f:
        prism_metrics = json.load(f)

    # Load baseline metrics
    baselines_df = pd.read_csv(RESULTS_DIR / "baselines" / "baseline_metrics_oos.csv")

    # Create comparison table
    comparison_data = []

    # PRISM-R
    comparison_data.append({
        "Strategy": "PRISM-R (Portfolio Optimization)",
        "Total Return": f"{prism_metrics['total_return']:.2%}",
        "Annual Return": f"{prism_metrics['annualized_return']:.2%}",
        "Volatility": f"{prism_metrics['annualized_volatility']:.2%}",
        "Sharpe (mean)": f"{prism_metrics['sharpe_oos_mean']:.4f}",
        "Sharpe (median)": f"{prism_metrics['sharpe_oos_median']:.4f}",
        "CVaR 95%": f"{prism_metrics['cvar_95']:.4f}",
        "PSR": f"{prism_metrics['psr']:.4f}",
        "DSR": f"{prism_metrics['dsr']:.4f}",
        "Max Drawdown": f"{prism_metrics['max_drawdown']:.2%}",
        "Turnover": f"{prism_metrics['turnover_median']:.2e}",
        "Cost (bps)": f"{prism_metrics['cost_annual_bps']:.2f}",
        "Success Rate": f"{prism_metrics['success_rate']:.1%}",
    })

    # Baselines
    baseline_names = {
        "min_variance_lw": "Minimum Variance (Ledoit-Wolf)",
        "shrunk_mv": "Shrunk Mean-Variance",
        "equal_weight": "Equal-Weight 1/N",
        "risk_parity": "Risk Parity",
        "sixty_forty": "60/40 Stocks/Bonds",
        "hrp": "Hierarchical Risk Parity",
    }

    for idx, row in baselines_df.iterrows():
        strategy = row["strategy"]
        comparison_data.append({
            "Strategy": baseline_names.get(strategy, strategy),
            "Total Return": f"{row['total_return']:.2%}",
            "Annual Return": f"{row['annualized_return']:.2%}",
            "Volatility": f"{row['volatility']:.2%}",
            "Sharpe (mean)": f"{row['sharpe']:.4f}",
            "Sharpe (median)": "—",  # Not available for baselines
            "CVaR 95%": f"{row['cvar_95']:.4f}",
            "PSR": "—",  # Not computed for baselines
            "DSR": "—",  # Not computed for baselines
            "Max Drawdown": f"{row['max_drawdown']:.2%}",
            "Turnover": f"{row['avg_turnover']:.2e}",
            "Cost (bps)": f"{row['total_cost']*10000:.2f}",
            "Success Rate": "—",  # Not available for baselines
        })

    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df

def dataframe_to_markdown(df):
    """Convert DataFrame to markdown table (simple version without tabulate)."""
    lines = []

    # Header
    header = "| " + " | ".join(str(col) for col in df.columns) + " |"
    lines.append(header)

    # Separator
    separator = "|" + "|".join(["-" * 25 for _ in df.columns]) + "|"
    lines.append(separator)

    # Rows
    for _, row in df.iterrows():
        row_str = "| " + " | ".join(str(val) for val in row) + " |"
        lines.append(row_str)

    return "\n".join(lines)

def generate_markdown_report():
    """Generate markdown formatted report."""

    # Load metrics
    with open(REPORTS_DIR / "oos_consolidated_metrics.json") as f:
        metrics = json.load(f)

    lines = [
        "# Final OOS Performance Report (2020-01-02 to 2025-10-31)",
        "",
        "## Executive Summary",
        "",
        "| Metric | Value | Notes |",
        "|--------|-------|-------|",
        f"| **Cumulative NAV** | {metrics['nav_final']:.4f} | Period: 5.8 years (1466 days) |",
        f"| Total Return | {metrics['total_return']:.2%} | End-to-end backtest performance |",
        f"| Annualized Return | {metrics['annualized_return']:.2%} | Geometric annualization over period |",
        f"| Annualized Volatility | {metrics['annualized_volatility']:.2%} | Risk measurement (target: ≤12%) |",
        "",
        "## Risk-Adjusted Performance",
        "",
        "| Metric | Value | Interpretation |",
        "|--------|-------|-----------------|",
        f"| Sharpe Ratio (window mean) | {metrics['sharpe_oos_mean']:.4f} | Average across 64 OOS windows |",
        f"| Sharpe Ratio (window median) | {metrics['sharpe_oos_median']:.4f} | Robust measure (median better than mean) |",
        f"| Probabilistic Sharpe (PSR) | {metrics['psr']:.4f} | Probability true Sharpe > 0 |",
        f"| Deflated Sharpe (DSR) | {metrics['dsr']:.4f} | Multiple-testing adjusted |",
        f"| CVaR 95% | {metrics['cvar_95']:.4f} | Tail risk (mean of worst 5% returns) |",
        "",
        "## Risk Metrics",
        "",
        "| Metric | Value | Constraint |",
        "|--------|-------|-----------|",
        f"| Max Drawdown | {metrics['max_drawdown']:.2%} | Limit: ≤15% |",
        f"| Avg Drawdown | {metrics['avg_drawdown']:.2%} | Typical downside magnitude |",
        "",
        "## Turnover & Costs",
        "",
        "| Metric | Value | Range |",
        "|--------|-------|-------|",
        f"| Turnover (median) | {metrics['turnover_median']:.2e} | p25={metrics['turnover_p25']:.2e}, p75={metrics['turnover_p75']:.2e} |",
        f"| Cost (annual) | {metrics['cost_annual_bps']:.2f} bps | Target: ≤50 bps |",
        f"| Success Rate | {metrics['success_rate']:.1%} | Winning windows / total windows |",
        "",
        "## Window-Level Analysis",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Windows Analyzed (OOS) | {metrics['n_windows']} |",
        f"| Sharpe Std Dev | {metrics['sharpe_oos_std']:.4f} |",
        "",
        "---",
        "",
        "## Comparison with Baselines",
        "",
    ]

    comparison_df = generate_comparison_table()
    lines.append(dataframe_to_markdown(comparison_df))

    lines.extend([
        "",
        "---",
        "",
        "## Methodology Notes",
        "",
        "- **Period**: 2020-01-02 to 2025-10-31 (5.8 years, 1466 days)",
        "- **Walk-Forward Analysis**: 64 OOS windows from filtered period",
        "- **NAV Calculation**: (1.1414)^(252/1466) - 1 = 2.30% annualized",
        "- **Sharpe Sources**: Window-level medians (64 windows) with PSR/DSR adjustments",
        "- **CVaR**: Approximated from window-level drawdowns at 95% confidence",
        "- **PSR/DSR**: Computed from window distribution statistics (not individual returns)",
        "",
    ])

    return "\n".join(lines)

def main():
    print("Generating final metrics report...")

    # Generate and display markdown report
    report = generate_markdown_report()
    print(report)

    # Save to file
    report_path = REPORTS_DIR / "FINAL_OOS_METRICS_REPORT.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n✓ Report saved to: {report_path}")

    # Also save comparison table as CSV
    comparison_df = generate_comparison_table()
    csv_path = REPORTS_DIR / "strategy_comparison_final.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"✓ Comparison CSV saved to: {csv_path}")

if __name__ == "__main__":
    main()
