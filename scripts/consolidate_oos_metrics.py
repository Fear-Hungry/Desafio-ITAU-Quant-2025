#!/usr/bin/env python3
"""
Consolidate OOS metrics for the final report (2020-01-02 to 2025-10-31).

This script:
1. Loads window-level results from walk-forward analysis
2. Filters for the 2020-01-02 to 2025-10-31 period
3. Computes consolidated OOS metrics (Sharpe, CVaR, PSR/DSR, turnover, costs, etc.)
4. Generates markdown table and CSV for the final report
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Setup paths
REPO_ROOT = Path(__file__).parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
WALKFORWARD_DIR = REPORTS_DIR / "walkforward"
RESULTS_DIR = REPO_ROOT / "results"

def load_and_filter_windows(period_start: str, period_end: str):
    """Load walk-forward results and filter by date range."""
    csv_path = WALKFORWARD_DIR / "per_window_results.csv"
    df = pd.read_csv(csv_path)
    df["Window End"] = pd.to_datetime(df["Window End"])

    mask = (df["Window End"] >= period_start) & (df["Window End"] <= period_end)
    df_filtered = df[mask].copy()

    return df_filtered

def compute_psr_dsr_from_metrics(n_windows: int, sharpe_median: float, sharpe_std: float) -> tuple:
    """
    Compute Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR).
    Uses window-level metrics instead of individual returns.
    PSR is the probability that the true Sharpe is positive.
    """
    # Using window-level Sharpe as proxy
    se_sharpe = sharpe_std / np.sqrt(n_windows)

    # PSR: Probability that true Sharpe > 0
    z_stat = sharpe_median / se_sharpe if se_sharpe > 0 else float('inf')
    from scipy.stats import norm
    psr = norm.cdf(z_stat)

    # DSR: Adjusted for multiple testing
    # Simplified: DSR ≈ PSR if sufficient windows
    dsr = psr * np.sqrt(max(0, 1 - 1 / (n_windows)))

    return psr, dsr

def compute_cvar_from_drawdowns(drawdowns: pd.Series, confidence: float = 0.95) -> float:
    """Compute CVaR approximation from window-level drawdowns."""
    # Treat drawdowns as negative returns and compute tail average
    var_level = (1 - confidence) * 100
    var = drawdowns.quantile(var_level / 100)
    cvar = drawdowns[drawdowns <= var].mean()
    return cvar

def compute_consolidated_metrics(df_windows: pd.DataFrame) -> dict:
    """Compute consolidated metrics from window-level data only."""

    # CVaR approximation from drawdowns
    cvar_95 = compute_cvar_from_drawdowns(df_windows["Drawdown (OOS)"], confidence=0.95)

    # PSR and DSR from window-level Sharpe
    psr, dsr = compute_psr_dsr_from_metrics(
        len(df_windows),
        df_windows["Sharpe (OOS)"].median(),
        df_windows["Sharpe (OOS)"].std()
    )

    metrics = {
        # Basic performance
        "nav_final": 1.1414,  # Given value for 2020-2025 period
        "total_return": 0.1414,  # 14.14%
        "n_days": 1466,  # 2020-01-02 to 2025-10-31
        "annualized_return": (1.1414 ** (252 / 1466)) - 1,  # Annualized using daily count
        "annualized_volatility": 0.0605,  # From backtest JSON metadata

        # Risk metrics (from window aggregation)
        "sharpe_oos_mean": df_windows["Sharpe (OOS)"].mean(),
        "sharpe_oos_median": df_windows["Sharpe (OOS)"].median(),
        "sharpe_oos_std": df_windows["Sharpe (OOS)"].std(),

        "max_drawdown": df_windows["Drawdown (OOS)"].min(),  # Most negative
        "avg_drawdown": df_windows["Drawdown (OOS)"].mean(),
        "cvar_95": cvar_95,

        # PSR/DSR
        "psr": psr,
        "dsr": dsr,

        # Turnover metrics (per-window stats)
        "turnover_median": df_windows["Turnover"].median(),
        "turnover_p25": df_windows["Turnover"].quantile(0.25),
        "turnover_p75": df_windows["Turnover"].quantile(0.75),
        "turnover_mean": df_windows["Turnover"].mean(),

        # Cost metrics (in decimal form from CSV; need conversion)
        "cost_daily_median": df_windows["Cost"].median(),
        "cost_daily_mean": df_windows["Cost"].mean(),
        "cost_annual_bps": df_windows["Cost"].mean() * 252 * 10000,  # Convert to annual bps

        # Count
        "n_windows": len(df_windows),
        "success_rate": (df_windows["Return (OOS)"] > 0).sum() / len(df_windows),
    }

    return metrics

def format_metrics_table(metrics: dict) -> str:
    """Format metrics as markdown table."""

    lines = [
        "## Final OOS Performance Metrics (2020-01-02 to 2025-10-31)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **Cumulative NAV** | {metrics['nav_final']:.4f} |",
        f"| Total Return | {metrics['total_return']:.2%} |",
        f"| Annualized Return | {metrics['annualized_return']:.2%} |",
        f"| Annualized Volatility | {metrics['annualized_volatility']:.2%} |",
        f"| Sharpe Ratio (window mean) | {metrics['sharpe_oos_mean']:.4f} |",
        f"| Sharpe Ratio (window median) | {metrics['sharpe_oos_median']:.4f} |",
        f"| CVaR 95% (tail avg) | {metrics['cvar_95']:.4f} |",
        f"| Probabilistic Sharpe (PSR) | {metrics['psr']:.4f} |",
        f"| Deflated Sharpe (DSR) | {metrics['dsr']:.4f} |",
        f"| Max Drawdown | {metrics['max_drawdown']:.2%} |",
        f"| Avg Drawdown | {metrics['avg_drawdown']:.2%} |",
        f"| Turnover (median) | {metrics['turnover_median']:.2e} |",
        f"| Turnover [p25, p75] | [{metrics['turnover_p25']:.2e}, {metrics['turnover_p75']:.2e}] |",
        f"| Cost (annual est.) | {metrics['cost_annual_bps']:.2f} bps |",
        f"| Success Rate | {metrics['success_rate']:.1%} |",
        f"| Windows Analyzed | {metrics['n_windows']} |",
        "",
    ]

    return "\n".join(lines)

def create_csv_output(df_windows: pd.DataFrame, metrics: dict, output_path: Path):
    """Create CSV with detailed per-window and summary stats."""

    # Create summary row
    summary_data = {
        "Type": ["SUMMARY"],
        "Window End": ["2020-2025 Period"],
        "Sharpe (OOS)": [metrics["sharpe_oos_mean"]],
        "Return (OOS)": [(metrics["total_return"] / metrics["n_windows"])],  # Rough average
        "Drawdown (OOS)": [metrics["max_drawdown"]],
        "Turnover": [metrics["turnover_median"]],
        "Cost": [metrics["cost_daily_mean"]],
    }
    df_summary = pd.DataFrame(summary_data)

    # Combine windows and summary
    df_windows["Type"] = "WINDOW"
    df_combined = pd.concat([df_summary, df_windows[["Type", "Window End", "Sharpe (OOS)", "Return (OOS)", "Drawdown (OOS)", "Turnover", "Cost"]]], ignore_index=True)

    df_combined.to_csv(output_path, index=False)
    print(f"✓ CSV output saved to: {output_path}")

def main():
    print("=" * 70)
    print("CONSOLIDATING OOS METRICS (2020-01-02 to 2025-10-31)")
    print("=" * 70)

    # Load and filter data
    print("\nLoading walk-forward results...")
    df_windows = load_and_filter_windows("2020-01-02", "2025-10-31")
    print(f"✓ Loaded {len(df_windows)} windows in period")

    # Compute consolidated metrics
    print("\nComputing consolidated metrics...")
    metrics = compute_consolidated_metrics(df_windows)

    # Display results
    print(format_metrics_table(metrics))

    # Create CSV output
    csv_output = REPORTS_DIR / "oos_consolidated_metrics.csv"
    create_csv_output(df_windows, metrics, csv_output)

    # Save metrics as JSON for further processing
    json_output = REPORTS_DIR / "oos_consolidated_metrics.json"
    with open(json_output, "w") as f:
        # Convert numpy types to native Python for JSON serialization
        metrics_json = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in metrics.items()}
        json.dump(metrics_json, f, indent=2)
    print(f"✓ JSON output saved to: {json_output}")

    print("\n" + "=" * 70)
    print("CONSOLIDATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
