#!/usr/bin/env python3
"""Generate tearsheet plots from backtest JSON results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn-v0_8-darkgrid")

REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_backtest_json(path: Path) -> dict:
    """Load backtest JSON, handling console output prefix."""
    with open(path, "r") as f:
        content = f.read()

    # Find first '{' to skip console output
    json_start = content.find("{")
    if json_start > 0:
        content = content[json_start:]

    return json.loads(content)


def plot_cumulative_nav(data: dict, output_name: str = "tearsheet_cumulative_nav.png"):
    """Plot cumulative NAV from ledger."""
    ledger = data.get("ledger", {})
    if not ledger or "nav" not in ledger:
        print("No ledger NAV data found")
        return

    navs = ledger["nav"]
    # Use indices for x-axis since we have daily NAV but only monthly trade dates
    x_axis = list(range(len(navs)))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_axis, navs, linewidth=2, color="#2E86AB", markersize=0)
    ax.set_title("Cumulative Portfolio NAV", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV (Net Asset Value)")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Initial Capital")

    # Add final NAV annotation
    final_nav = navs[-1]
    ax.annotate(
        f"Final NAV: {final_nav:.4f}",
        xy=(len(navs) - 1, final_nav),
        xytext=(-100, 20),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )

    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / output_name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {output_name}")


def plot_drawdown(data: dict, output_name: str = "tearsheet_drawdown.png"):
    """Plot drawdown series from ledger."""
    ledger = data.get("ledger", {})
    if not ledger or "nav" not in ledger:
        print("No ledger NAV data found")
        return

    navs = np.array(ledger["nav"])
    # Use indices for x-axis
    x_axis = list(range(len(navs)))

    # Calculate drawdown
    running_max = np.maximum.accumulate(navs)
    drawdown = (navs - running_max) / running_max

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(
        range(len(drawdown)), drawdown * 100, 0, color="#A23B72", alpha=0.6
    )
    ax.plot(drawdown * 100, color="#A23B72", linewidth=1.5)
    ax.set_title("Portfolio Drawdown", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)

    # Annotate max drawdown
    max_dd_idx = np.argmin(drawdown)
    max_dd = drawdown[max_dd_idx] * 100
    ax.annotate(
        f"Max DD: {max_dd:.2f}%",
        xy=(max_dd_idx, max_dd),
        xytext=(50, -30),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="red", alpha=0.7),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        color="white",
    )

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / output_name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {output_name}")


def plot_cost_decomposition(data: dict, output_name: str = "tearsheet_cost_decomposition.png"):
    """Plot cost breakdown over time."""
    ledger = data.get("ledger", [])
    if not ledger:
        print("No ledger data found")
        return

    dates = [entry["date"] for entry in ledger]
    costs = [entry.get("cost_fraction", 0) * 10000 for entry in ledger]  # Convert to bps
    turnovers = [entry.get("turnover", 0) * 100 for entry in ledger]  # Convert to %

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Cost plot
    ax1.bar(range(len(costs)), costs, color="#F18F01", alpha=0.7)
    ax1.set_title("Transaction Costs Over Time", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Cost (bps)")
    ax1.grid(True, alpha=0.3, axis="y")
    avg_cost = np.mean([c for c in costs if c > 0]) if any(c > 0 for c in costs) else 0
    ax1.axhline(y=avg_cost, color="red", linestyle="--", label=f"Avg: {avg_cost:.2f} bps")
    ax1.legend()

    # Turnover plot
    ax2.bar(range(len(turnovers)), turnovers, color="#6A994E", alpha=0.7)
    ax2.set_title("Portfolio Turnover", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Rebalance Event")
    ax2.set_ylabel("Turnover (%)")
    ax2.grid(True, alpha=0.3, axis="y")
    avg_turnover = np.mean([t for t in turnovers if t > 0]) if any(t > 0 for t in turnovers) else 0
    ax2.axhline(y=avg_turnover, color="blue", linestyle="--", label=f"Avg: {avg_turnover:.2f}%")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / output_name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {output_name}")


def plot_risk_contribution(data: dict, output_name: str = "tearsheet_risk_contribution_by_budget.png"):
    """Plot risk contribution by asset class (budget groups)."""
    # Get final weights
    ledger = data.get("ledger", [])
    if not ledger:
        print("No ledger data found")
        return

    final_weights = ledger[-1].get("weights", {})

    # Group by asset class (simplified - would need config for proper grouping)
    # For now, just show top 10 assets by weight
    weights_series = pd.Series(final_weights)
    weights_series = weights_series[weights_series > 0.001]  # Filter tiny positions
    weights_series = weights_series.sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, len(weights_series)))
    weights_series.plot(kind="barh", ax=ax, color=colors)
    ax.set_title("Final Portfolio Allocation (Top 15 Assets)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Weight (%)")
    ax.set_ylabel("Asset")

    # Convert x-axis to percentage
    ax.set_xticklabels([f"{x*100:.1f}%" for x in ax.get_xticks()])
    ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / output_name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {output_name}")


def plot_walkforward_nav(output_name: str = "walkforward_nav_20251101.png"):
    """Create simplified walk-forward NAV plot from per-window results."""
    wf_file = Path("reports/walkforward/per_window_results.csv")
    if not wf_file.exists():
        print("Walk-forward results not found")
        return

    df = pd.read_csv(wf_file)

    # Calculate cumulative returns
    df["Cumulative Return"] = (1 + df["Return (OOS)"]).cumprod()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # NAV evolution
    ax1.plot(df.index, df["Cumulative Return"], linewidth=2, color="#2E86AB", label="Cumulative NAV")
    ax1.fill_between(df.index, 1, df["Cumulative Return"], alpha=0.3, color="#2E86AB")
    ax1.set_title("Walk-Forward Cumulative NAV", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Cumulative Return")
    ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Sharpe per window
    ax2.bar(df.index, df["Sharpe (OOS)"], color="#6A994E", alpha=0.7)
    ax2.set_title("Sharpe Ratio per OOS Window", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Window Index")
    ax2.set_ylabel("Sharpe Ratio")
    ax2.axhline(y=0, color="black", linewidth=1)
    ax2.grid(True, alpha=0.3, axis="y")

    # Add average line
    avg_sharpe = df["Sharpe (OOS)"].mean()
    ax2.axhline(y=avg_sharpe, color="red", linestyle="--", label=f"Avg: {avg_sharpe:.2f}")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / output_name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {output_name}")


def main():
    """Generate all tearsheet plots."""
    print("=== Generating Tearsheet Plots ===\n")

    # Load backtest data
    backtest_file = REPORTS_DIR / "backtest_optimizer_example.json"
    if not backtest_file.exists():
        print(f"Error: {backtest_file} not found")
        return

    data = load_backtest_json(backtest_file)

    # Generate plots
    plot_cumulative_nav(data)
    plot_drawdown(data)
    plot_cost_decomposition(data)
    plot_risk_contribution(data)
    plot_walkforward_nav()

    print(f"\n✓ All plots saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
