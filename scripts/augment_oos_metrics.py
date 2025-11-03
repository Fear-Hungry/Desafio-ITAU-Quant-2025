#!/usr/bin/env python3
"""
Augment consolidated OOS metrics with window-level statistics for PRISM-R.

This script reads:
- reports/oos_consolidated_metrics.json (single source consolidated metrics)
- reports/walkforward/per_window_results.csv (walk-forward window metrics)

Then it augments the consolidated metrics JSON with the following fields
computed from per-window results:
- sharpe_oos_mean, sharpe_oos_median, sharpe_oos_std
- psr, dsr  (computed from window-level Sharpe distribution)
- max_drawdown, avg_drawdown (from window-level drawdowns)
- cvar_95 (tail mean of worst 5% window drawdowns)
- turnover_median, turnover_p25, turnover_p75, turnover_mean (if present)
- cost_daily_median, cost_daily_mean, cost_annual_bps (if present)
- n_windows (count)
- success_rate (fraction of windows with positive OOS return, if present)

Usage:
    poetry run python scripts/augment_oos_metrics.py
    # or with custom paths:
    poetry run python scripts/augment_oos_metrics.py \
        --metrics-json reports/oos_consolidated_metrics.json \
        --windows-csv reports/walkforward/per_window_results.csv \
        --save-csv
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METRICS_JSON = REPO_ROOT / "reports" / "oos_consolidated_metrics.json"
DEFAULT_WINDOWS_CSV = REPO_ROOT / "reports" / "walkforward" / "per_window_results.csv"
DEFAULT_METRICS_CSV = REPO_ROOT / "reports" / "oos_consolidated_metrics.csv"


def _phi(z: float) -> float:
    """Standard normal CDF without external deps."""
    # Guard infinities
    if math.isinf(z):
        return 1.0 if z > 0 else 0.0
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def compute_psr_dsr(n_windows: int, sharpe_median: float, sharpe_std: float) -> tuple[float, float]:
    """
    Compute Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR)
    from window-level Sharpe statistics.

    This follows the simplified approach used in project scripts:
    - se = std / sqrt(n)
    - z = median / se
    - PSR = Phi(z)
    - DSR ≈ PSR * sqrt(1 - 1/n)  (simple deflation for multiple testing)
    """
    if n_windows <= 1 or sharpe_std <= 0:
        # Degenerate case: no dispersion or single window
        return (1.0 if sharpe_median > 0 else 0.0, 1.0 if sharpe_median > 0 else 0.0)

    se = sharpe_std / math.sqrt(n_windows)
    z = sharpe_median / se if se > 0 else float("inf")
    psr = _phi(z)
    dsr = psr * math.sqrt(max(0.0, 1.0 - 1.0 / n_windows))
    return psr, dsr


def compute_cvar_from_drawdowns(drawdowns: pd.Series, confidence: float = 0.95) -> float:
    """Compute CVaR of window-level drawdowns (tail mean of worst 5%)."""
    if drawdowns.empty:
        return float("nan")
    # Note: drawdowns should be negative (e.g., -0.15 = -15%)
    alpha = 1.0 - confidence
    var = drawdowns.quantile(alpha, interpolation="linear")
    tail = drawdowns[drawdowns <= var]
    return float(tail.mean()) if not tail.empty else float(var)


def load_metrics_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Metrics JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_metrics_json(path: Path, data: Dict[str, Any]) -> None:
    # Ensure JSON-serializable (convert numpy types)
    def _to_native(x: Any) -> Any:
        if isinstance(x, (np.floating, np.integer)):
            return float(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
        return x

    sanitized = {k: _to_native(v) for k, v in data.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sanitized, f, indent=2)
    print(f"✓ Updated metrics JSON: {path}")


def maybe_update_metrics_csv(path: Path, metrics: Dict[str, Any]) -> None:
    """Optional: write a single-row CSV with the updated metrics."""
    df = pd.DataFrame([metrics])
    df.to_csv(path, index=False)
    print(f"✓ Updated metrics CSV: {path}")


def augment_with_window_stats(
    metrics: Dict[str, Any], df_windows: pd.DataFrame
) -> Dict[str, Any]:
    """Compute and inject window-level statistics into consolidated metrics."""
    augmented = dict(metrics)  # shallow copy

    # Sanity checks for expected columns
    required_cols = ["Sharpe (OOS)", "Drawdown (OOS)"]
    for col in required_cols:
        if col not in df_windows.columns:
            raise ValueError(f"Missing required column in per-window CSV: '{col}'")

    n_windows = int(len(df_windows))
    sharpe = df_windows["Sharpe (OOS)"].astype(float)
    drawdown = df_windows["Drawdown (OOS)"].astype(float)

    augmented["n_windows"] = n_windows
    augmented["sharpe_oos_mean"] = float(sharpe.mean())
    augmented["sharpe_oos_median"] = float(sharpe.median())
    augmented["sharpe_oos_std"] = float(sharpe.std(ddof=1) if n_windows > 1 else 0.0)

    # PSR/DSR
    psr, dsr = compute_psr_dsr(
        n_windows=n_windows,
        sharpe_median=augmented["sharpe_oos_median"],
        sharpe_std=augmented["sharpe_oos_std"],
    )
    augmented["psr"] = float(psr)
    augmented["dsr"] = float(dsr)

    # Drawdown stats (window-level)
    augmented["max_drawdown"] = float(drawdown.min()) if not drawdown.empty else float("nan")
    augmented["avg_drawdown"] = float(drawdown.mean()) if not drawdown.empty else float("nan")
    augmented["cvar_95"] = compute_cvar_from_drawdowns(drawdown, confidence=0.95)

    # Turnover (if present)
    if "Turnover" in df_windows.columns:
        turnover = df_windows["Turnover"].astype(float)
        augmented["turnover_median"] = float(turnover.median())
        augmented["turnover_p25"] = float(turnover.quantile(0.25))
        augmented["turnover_p75"] = float(turnover.quantile(0.75))
        augmented["turnover_mean"] = float(turnover.mean())

    # Cost (if present)
    if "Cost" in df_windows.columns:
        cost = df_windows["Cost"].astype(float)
        augmented["cost_daily_median"] = float(cost.median())
        augmented["cost_daily_mean"] = float(cost.mean())
        augmented["cost_annual_bps"] = float(cost.mean() * 252.0 * 10000.0)

    # Success rate (windows with positive OOS return)
    if "Return (OOS)" in df_windows.columns:
        ret_win = df_windows["Return (OOS)"].astype(float)
        augmented["success_rate"] = float((ret_win > 0).mean())

    return augmented


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment consolidated OOS metrics with window-level stats for PRISM-R"
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=DEFAULT_METRICS_JSON,
        help="Path to oos_consolidated_metrics.json",
    )
    parser.add_argument(
        "--windows-csv",
        type=Path,
        default=DEFAULT_WINDOWS_CSV,
        help="Path to walk-forward per_window_results.csv",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Also write updated metrics CSV (oos_consolidated_metrics.csv)",
    )
    args = parser.parse_args()

    print("=" * 78)
    print("AUGMENTING CONSOLIDATED METRICS WITH WINDOW-LEVEL STATISTICS")
    print("=" * 78)

    # Load existing metrics and per-window results
    print(f"\nLoading consolidated metrics JSON:\n  {args.metrics_json}")
    metrics = load_metrics_json(args.metrics_json)
    print("✓ Loaded consolidated metrics")

    print(f"\nLoading per-window results CSV:\n  {args.windows_csv}")
    if not args.windows_csv.exists():
        raise FileNotFoundError(f"Per-window CSV not found: {args.windows_csv}")
    df_windows = pd.read_csv(args.windows_csv)
    print(f"✓ Loaded {len(df_windows)} window rows")

    # Augment and save
    print("\nComputing window-level statistics and augmenting JSON...")
    augmented = augment_with_window_stats(metrics, df_windows)

    # Display short summary
    print("\nSummary (key fields):")
    keys = [
        "n_windows",
        "sharpe_oos_mean",
        "sharpe_oos_median",
        "sharpe_oos_std",
        "psr",
        "dsr",
        "max_drawdown",
        "avg_drawdown",
        "cvar_95",
        "turnover_median",
        "turnover_p25",
        "turnover_p75",
        "turnover_mean",
        "cost_daily_mean",
        "cost_annual_bps",
        "success_rate",
    ]
    for k in keys:
        if k in augmented:
            print(f"  - {k}: {augmented[k]}")

    # Persist
    save_metrics_json(args.metrics_json, augmented)

    # Optional CSV update (single-row table)
    if args.save_csv:
        maybe_update_metrics_csv(DEFAULT_METRICS_CSV, augmented)

    print("\n✅ Augmentation complete.")


if __name__ == "__main__":
    main()
