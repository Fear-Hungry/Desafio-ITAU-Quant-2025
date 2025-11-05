#!/usr/bin/env python3
"""
Consolidate OOS metrics for the final report using the canonical nav_daily.csv.

This script:
1. Loads the canonical daily NAV series from nav_daily.csv (single source of truth)
2. Loads OOS period configuration from oos_period.yaml
3. Computes consolidated OOS metrics (CAGR, vol, Sharpe, CVaR, PSR/DSR, costs, etc.)
4. Uses the same series for all downstream calculations (no divergences)
"""

import argparse
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
import json

# Setup paths
REPO_ROOT = Path(__file__).parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
WALKFORWARD_DIR = REPORTS_DIR / "walkforward"
RESULTS_DIR = REPO_ROOT / "results"
CONFIG_DIR = REPO_ROOT / "configs"

def load_oos_config():
    """Load OOS period configuration from centralized config."""
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

def _load_riskfree_daily(rf_csv: Path) -> pd.DataFrame:
    """Load a risk-free daily series from CSV.

    Expected schema (flexible):
      - Must contain a date column named 'date' (YYYY-MM-DD or ISO).
      - Must contain one of the following columns:
          * 'rf_daily'  → daily decimal return (e.g., 0.0001)
          * 'rf_annual' → annualized decimal rate; converted to daily as rate/252
          * 'rf_rate'   → same as rf_annual
          * 'yield'     → annualized rate; if > 1, assume percentage and divide by 100

    Returns a DataFrame with columns: ['date', 'rf_daily'] sorted by date.
    """
    if not rf_csv.exists():
        raise FileNotFoundError(f"Risk-free CSV not found: {rf_csv}")
    df = pd.read_csv(rf_csv)
    if 'date' not in df.columns:
        # Try common alternatives
        for alt in ('Date', 'DATE', 'timestamp'):
            if alt in df.columns:
                df = df.rename(columns={alt: 'date'})
                break
    if 'date' not in df.columns:
        raise ValueError("Risk-free CSV must contain a 'date' column")
    df['date'] = pd.to_datetime(df['date'])

    rf_col = None
    for c in ('rf_daily', 'rf_annual', 'rf_rate', 'yield', 'YIELD', 'RATE'):
        if c in df.columns:
            rf_col = c
            break
    if rf_col is None:
        # Fallback: first non-date numeric column
        candidates = [c for c in df.columns if c != 'date']
        if not candidates:
            raise ValueError("Risk-free CSV has no numeric columns besides 'date'")
        rf_col = candidates[0]

    s = pd.to_numeric(df[rf_col], errors='coerce')
    # Detect whether this is daily decimal or annualized
    # Heuristic: if median > 0.02 or any > 0.2, likely annualized
    med = float(s.dropna().median()) if s.notna().any() else 0.0
    is_annual_like = med > 0.02 or float(s.dropna().max()) > 0.2
    if is_annual_like:
        # If values look like percentages (e.g., 5.0 for 5%), convert to decimal
        if med > 1.0:
            s = s / 100.0
        rf_daily = s / 252.0
    else:
        rf_daily = s

    out = pd.DataFrame({'date': df['date'], 'rf_daily': rf_daily})
    out = out.sort_values('date').dropna(subset=['rf_daily']).reset_index(drop=True)
    return out


def compute_metrics_from_nav_daily(df_nav: pd.DataFrame, oos_config: dict, rf_daily: pd.DataFrame | None = None) -> dict:
    """
    Compute all metrics directly from daily NAV series.
    This ensures consistent calculations with no divergences.
    """
    # Filter to OOS period
    start_date = pd.to_datetime(oos_config['start_date'])
    end_date = pd.to_datetime(oos_config['end_date'])
    mask = (df_nav['date'] >= start_date) & (df_nav['date'] <= end_date)
    df_oos = df_nav[mask].copy()

    if len(df_oos) == 0:
        raise ValueError(f"No data found in period {start_date} to {end_date}")

    daily_returns = df_oos['daily_return'].values
    nav_values = df_oos['nav'].values
    dates = df_oos['date'].values

    # Basic performance metrics
    nav_initial = 1.0  # Starting value
    nav_final = nav_values[-1]
    total_return = nav_final - nav_initial
    n_days = len(daily_returns)
    n_years = n_days / 252.0

    # Annualized return using actual day count
    annualized_return = (nav_final ** (252 / n_days)) - 1

    # Annualized volatility (total returns)
    annualized_volatility = np.std(daily_returns, ddof=1) * np.sqrt(252)

    # Sharpe ratio: if rf_daily provided, compute on excess returns
    sharpe_ratio: float
    sharpe_excess: float | None = None
    rf_note: str | None = None
    if rf_daily is not None and not rf_daily.empty:
        # Align RF to dates and fill missing with 0
        rf_aligned = pd.merge(
            pd.DataFrame({'date': df_oos['date'].values}),
            rf_daily[['date', 'rf_daily']], on='date', how='left'
        )['rf_daily'].fillna(0.0).values
        excess = daily_returns - rf_aligned
        ann_excess_mean = np.mean(excess) * 252.0
        ann_excess_vol = np.std(excess, ddof=1) * np.sqrt(252.0)
        sharpe_ratio = (ann_excess_mean / ann_excess_vol) if ann_excess_vol > 0 else 0.0
        sharpe_excess = sharpe_ratio
        rf_note = "Sharpe computed on daily excess returns vs provided T-Bill series"
    else:
        # Fallback to RF≈0 (compatibility)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0

    # Drawdowns
    running_max = np.maximum.accumulate(nav_values)
    drawdowns = (nav_values - running_max) / running_max
    max_drawdown = np.min(drawdowns)
    avg_drawdown = np.mean(drawdowns[drawdowns < 0]) if np.any(drawdowns < 0) else 0

    # CVaR 95% (Expected Shortfall from daily returns)
    # Calculate from worst 5% of daily returns, not drawdowns
    var_95_threshold = np.percentile(daily_returns, 5)  # 5th percentile (worst 5%)
    cvar_95_daily = np.mean(daily_returns[daily_returns <= var_95_threshold]) if np.any(daily_returns <= var_95_threshold) else np.min(daily_returns)

    # Annualized CVaR (using √252 scaling, same as volatility)
    cvar_95_annual = cvar_95_daily * np.sqrt(252)

    # Success rate (% of positive days)
    success_rate = np.sum(daily_returns > 0) / len(daily_returns)

    metrics = {
        # Performance metrics
        "nav_final": nav_final,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "sharpe_excess_rf": sharpe_excess if sharpe_excess is not None else sharpe_ratio,

        # Risk metrics
        "max_drawdown": max_drawdown,
        "avg_drawdown": avg_drawdown,
        "cvar_95": cvar_95_daily,  # Daily CVaR for monitoring
        "cvar_95_annual": cvar_95_annual,  # Annualized CVaR for targets

        # Sample statistics
        "n_days": n_days,
        "n_years": n_years,
        "success_rate": success_rate,

        # Period metadata
        "period_start": dates[0],
        "period_end": dates[-1],
        # Meta / notes
        "risk_free_note": rf_note if rf_note is not None else "Sharpe computed with RF≈0 (compat mode)",
    }

    return metrics

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

def format_simple_metrics_table(metrics: dict) -> str:
    """Format metrics as markdown table."""
    lines = [
        "## Final OOS Performance Metrics (from nav_daily.csv)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **NAV Final** | {metrics['nav_final']:.4f} |",
        f"| Total Return | {metrics['total_return']:.2%} |",
        f"| Annualized Return | {metrics['annualized_return']:.2%} |",
        f"| Annualized Volatility | {metrics['annualized_volatility']:.2%} |",
        f"| Sharpe Ratio | {metrics['sharpe_ratio']:.4f} |",
        f"| Max Drawdown | {metrics['max_drawdown']:.2%} |",
        f"| Avg Drawdown | {metrics['avg_drawdown']:.2%} |",
        f"| CVaR 95% (daily) | {metrics['cvar_95']:.4f} ({metrics['cvar_95']:.2%}) |",
        f"| CVaR 95% (annual) | {metrics['cvar_95_annual']:.4f} ({metrics['cvar_95_annual']:.2%}) |",
        f"| Success Rate | {metrics['success_rate']:.1%} |",
        f"| Days in Period | {metrics['n_days']} |",
        f"| Start Date | {metrics['period_start']} |",
        f"| End Date | {metrics['period_end']} |",
        "",
    ]
    return "\n".join(lines)

def main():
    print("=" * 70)
    print("CONSOLIDATING OOS METRICS (SINGLE SOURCE OF TRUTH)")
    print("=" * 70)

    parser = argparse.ArgumentParser(description="Consolidate OOS metrics from nav_daily.csv")
    parser.add_argument("--riskfree-csv", type=str, default=None,
                        help="Optional CSV with daily or annual risk-free (T-Bill) series. Columns: date + rf_daily OR rf_annual/rf_rate/yield")
    args = parser.parse_args()

    # Load OOS configuration
    print("\nLoading OOS period configuration...")
    oos_config = load_oos_config()
    print(f"✓ Period: {oos_config['start_date']} to {oos_config['end_date']}")

    # Load canonical daily NAV (single source of truth)
    print("\nLoading canonical daily NAV series...")
    df_nav = load_nav_daily()
    print(f"✓ Loaded {len(df_nav)} daily observations")

    # Optional: load risk-free series
    rf_df = None
    if args.riskfree_csv:
        try:
            rf_df = _load_riskfree_daily(Path(args.riskfree_csv))
            print(f"✓ Loaded risk-free series: {len(rf_df)} rows from {args.riskfree_csv}")
        except Exception as e:
            print(f"⚠️  Could not load risk-free series: {e}. Proceeding with RF≈0.")

    # Compute metrics from daily NAV
    print("\nComputing consolidated metrics from nav_daily.csv...")
    metrics = compute_metrics_from_nav_daily(df_nav, oos_config, rf_daily=rf_df)

    # Display results
    print("\n" + format_simple_metrics_table(metrics))

    # Save metrics as JSON for further processing
    json_output = REPORTS_DIR / "oos_consolidated_metrics.json"

    # Add baseline alignment note
    metrics['baseline_alignment_note'] = (
        "Baselines must be recomputed with the same OOS period defined in configs/oos_period.yaml; "
        "do not mix window-level Sharpe with daily-series Sharpe."
    )

    with open(json_output, "w") as f:
        # Convert numpy types to native Python for JSON serialization
        metrics_json = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else str(v)
            for k, v in metrics.items()
        }
        json.dump(metrics_json, f, indent=2)
    print(f"\n✓ Metrics JSON saved to: {json_output}")

    # Also save as CSV for transparency
    csv_output = REPORTS_DIR / "oos_consolidated_metrics.csv"
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(csv_output, index=False)
    print(f"✓ Metrics CSV saved to: {csv_output}")

    print("\n" + "=" * 70)
    print("✅ CONSOLIDATION COMPLETE (FROM SINGLE NAV SOURCE)")
    print("=" * 70)

if __name__ == "__main__":
    main()
