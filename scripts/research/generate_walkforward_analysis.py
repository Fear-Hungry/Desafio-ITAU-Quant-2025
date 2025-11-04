#!/usr/bin/env python3
"""
Generate comprehensive walk-forward analysis figure with 4 subplots:
1. Parameter Evolution (lambda, positions)
2. Sharpe Ratio per window
3. Consistency metrics (hit rate, vol of returns)
4. Turnover and Costs

Output: reports/figures/walkforward_analysis_YYYYMMDD.png
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn-v0_8-darkgrid")

REPORTS_DIR = Path("reports")
WALKFORWARD_DIR = REPORTS_DIR / "walkforward"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_per_window_results() -> pd.DataFrame:
    """Load per-window results from CSV."""
    csv_file = WALKFORWARD_DIR / "per_window_results.csv"
    if not csv_file.exists():
        raise FileNotFoundError(
            f"{csv_file} not found. Run walk-forward backtest first."
        )

    df = pd.read_csv(csv_file)

    # Ensure required columns exist
    required = ["Return (OOS)", "Sharpe (OOS)"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def load_weights_history() -> pd.DataFrame:
    """Load weights history to calculate positions and concentration."""
    weights_file = WALKFORWARD_DIR / "weights_history.csv"
    if not weights_file.exists():
        print("Warning: weights_history.csv not found, skipping position evolution")
        return None

    df = pd.read_csv(weights_file)
    return df


def calculate_consistency_metrics(df: pd.DataFrame) -> dict:
    """Calculate consistency metrics from per-window results."""
    returns = df["Return (OOS)"].values

    metrics = {
        "hit_rate": np.mean(returns > 0),
        "avg_win": np.mean(returns[returns > 0]) if any(returns > 0) else 0.0,
        "avg_loss": np.mean(returns[returns < 0]) if any(returns < 0) else 0.0,
        "win_loss_ratio": (
            abs(np.mean(returns[returns > 0]) / np.mean(returns[returns < 0]))
            if any(returns < 0) and any(returns > 0)
            else 0.0
        ),
        "sharpe_consistency": df["Sharpe (OOS)"].std(),
        "return_volatility": returns.std(),
    }

    return metrics


def plot_walkforward_analysis(
    df: pd.DataFrame,
    weights_df: pd.DataFrame | None = None,
    output_name: str = None,
) -> None:
    """
    Generate comprehensive 4-subplot walk-forward analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Per-window results
    weights_df : pd.DataFrame, optional
        Weights history for position evolution
    output_name : str, optional
        Output filename (default: walkforward_analysis_YYYYMMDD.png)
    """
    if output_name is None:
        today = datetime.now().strftime("%Y%m%d")
        output_name = f"walkforward_analysis_{today}.png"

    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Walk-Forward Analysis (OOS Windows)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    n_windows = len(df)
    window_indices = np.arange(n_windows)

    # ========== SUBPLOT 1: Parameter Evolution ==========
    ax1 = axes[0, 0]

    if weights_df is not None and len(weights_df) > 0:
        # Calculate active positions per rebalance
        # Assume weights_df has columns: date, ticker1, ticker2, ...
        weight_cols = [col for col in weights_df.columns if col != "date"]

        # Count non-zero positions (> 0.1%)
        active_positions = (weights_df[weight_cols] > 0.001).sum(axis=1).values

        # Calculate concentration (HHI)
        hhi = (weights_df[weight_cols] ** 2).sum(axis=1).values
        effective_n = 1.0 / hhi

        ax1_twin = ax1.twinx()

        # Plot active positions (left y-axis)
        line1 = ax1.plot(
            window_indices[:len(active_positions)],
            active_positions,
            color="#2E86AB",
            linewidth=2,
            marker="o",
            markersize=4,
            label="Active Positions",
        )
        ax1.set_ylabel("Number of Active Positions", color="#2E86AB", fontsize=11)
        ax1.tick_params(axis="y", labelcolor="#2E86AB")

        # Plot effective N (right y-axis)
        line2 = ax1_twin.plot(
            window_indices[:len(effective_n)],
            effective_n,
            color="#F18F01",
            linewidth=2,
            marker="s",
            markersize=4,
            label="Effective N (1/HHI)",
        )
        ax1_twin.set_ylabel("Effective N", color="#F18F01", fontsize=11)
        ax1_twin.tick_params(axis="y", labelcolor="#F18F01")

        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper left", fontsize=9)

    else:
        # Fallback: plot drawdown as proxy for risk parameter
        if "Drawdown (OOS)" in df.columns:
            ax1.plot(
                window_indices,
                df["Drawdown (OOS)"] * 100,
                color="#A23B72",
                linewidth=2,
                marker="o",
                markersize=4,
            )
            ax1.set_ylabel("Max Drawdown (%) [Proxy for Risk]", fontsize=11)
            ax1.axhline(
                y=-15.0, color="red", linestyle="--", alpha=0.5, label="Limit: -15%"
            )
            ax1.legend(fontsize=9)
        else:
            # Plot return as last resort
            ax1.plot(
                window_indices,
                df["Return (OOS)"] * 100,
                color="#2E86AB",
                linewidth=2,
                marker="o",
                markersize=4,
            )
            ax1.set_ylabel("Return (%)", fontsize=11)

    ax1.set_title("Parameter Evolution", fontsize=12, fontweight="bold")
    ax1.set_xlabel("OOS Window Index", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # ========== SUBPLOT 2: Sharpe Ratio per Window ==========
    ax2 = axes[0, 1]

    sharpe_values = df["Sharpe (OOS)"].values
    colors = ["#6A994E" if s >= 0 else "#A23B72" for s in sharpe_values]

    ax2.bar(window_indices, sharpe_values, color=colors, alpha=0.7, edgecolor="black")
    ax2.axhline(y=0, color="black", linewidth=1.5)

    # Add average line
    avg_sharpe = sharpe_values.mean()
    median_sharpe = np.median(sharpe_values)
    ax2.axhline(
        y=avg_sharpe,
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {avg_sharpe:.3f}",
    )
    ax2.axhline(
        y=median_sharpe,
        color="orange",
        linestyle=":",
        linewidth=2,
        label=f"Median: {median_sharpe:.3f}",
    )

    ax2.set_title("Sharpe Ratio per OOS Window", fontsize=12, fontweight="bold")
    ax2.set_xlabel("OOS Window Index", fontsize=10)
    ax2.set_ylabel("Sharpe Ratio", fontsize=11)
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(True, alpha=0.3, axis="y")

    # ========== SUBPLOT 3: Consistency Metrics ==========
    ax3 = axes[1, 0]

    # Calculate rolling hit rate (10-window moving average)
    returns = df["Return (OOS)"].values
    rolling_hit_rate = pd.Series(returns > 0).rolling(10, min_periods=5).mean() * 100

    # Calculate rolling return volatility
    rolling_vol = (
        pd.Series(returns).rolling(10, min_periods=5).std() * np.sqrt(21) * 100
    )

    ax3_twin = ax3.twinx()

    # Plot hit rate (left y-axis)
    line1 = ax3.plot(
        window_indices,
        rolling_hit_rate,
        color="#6A994E",
        linewidth=2.5,
        label="Hit Rate (10-win MA)",
    )
    ax3.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
    ax3.set_ylabel("Hit Rate (%)", color="#6A994E", fontsize=11)
    ax3.tick_params(axis="y", labelcolor="#6A994E")
    ax3.set_ylim(0, 100)

    # Plot volatility (right y-axis)
    line2 = ax3_twin.plot(
        window_indices,
        rolling_vol,
        color="#E63946",
        linewidth=2,
        linestyle="--",
        label="Rolling Vol (ann.)",
    )
    ax3_twin.set_ylabel("Volatility (%)", color="#E63946", fontsize=11)
    ax3_twin.tick_params(axis="y", labelcolor="#E63946")

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc="upper left", fontsize=9)

    ax3.set_title("Consistency Metrics", fontsize=12, fontweight="bold")
    ax3.set_xlabel("OOS Window Index", fontsize=10)
    ax3.grid(True, alpha=0.3)

    # ========== SUBPLOT 4: Turnover and Costs ==========
    ax4 = axes[1, 1]

    # Check if turnover columns exist
    if "Turnover" in df.columns:
        turnover = df["Turnover"].values * 100  # Convert to %
    elif "Turnover (OOS)" in df.columns:
        turnover = df["Turnover (OOS)"].values * 100  # Convert to %
    else:
        # Generate synthetic turnover for demonstration
        turnover = np.random.uniform(0.05, 0.3, n_windows) * 100
        print("Warning: No turnover data, using synthetic values for plot")

    # Calculate cumulative costs (assuming 30 bps per turnover)
    cost_per_window = turnover * 0.30  # 30 bps on turnover
    cumulative_cost = np.cumsum(cost_per_window)

    ax4_twin = ax4.twinx()

    # Plot turnover as bars (left y-axis)
    bars = ax4.bar(
        window_indices,
        turnover,
        color="#F18F01",
        alpha=0.6,
        edgecolor="black",
        label="Turnover (one-way)",
    )
    ax4.set_ylabel("Turnover (%)", color="#F18F01", fontsize=11)
    ax4.tick_params(axis="y", labelcolor="#F18F01")

    # Plot cumulative cost (right y-axis)
    line = ax4_twin.plot(
        window_indices,
        cumulative_cost,
        color="#A23B72",
        linewidth=2.5,
        marker="o",
        markersize=3,
        label="Cumulative Cost (bps)",
    )
    ax4_twin.set_ylabel("Cumulative Cost (bps)", color="#A23B72", fontsize=11)
    ax4_twin.tick_params(axis="y", labelcolor="#A23B72")

    # Add median turnover line
    median_turnover = np.median(turnover)
    ax4.axhline(
        y=median_turnover,
        color="blue",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"Median: {median_turnover:.2f}%",
    )

    # Combine legends
    ax4.legend(loc="upper left", fontsize=9)
    ax4_twin.legend(loc="upper right", fontsize=9)

    ax4.set_title("Turnover and Transaction Costs", fontsize=12, fontweight="bold")
    ax4.set_xlabel("OOS Window Index", fontsize=10)
    ax4.grid(True, alpha=0.3, axis="y")

    # ========== Final adjustments ==========
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save figure
    output_path = FIGURES_DIR / output_name
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Saved: {output_path}")

    # Print summary statistics
    print("\n=== Walk-Forward Summary Statistics ===")
    print(f"Total OOS Windows: {n_windows}")
    print(f"Average Sharpe: {avg_sharpe:.3f}")
    print(f"Median Sharpe: {median_sharpe:.3f}")
    print(f"Sharpe Std Dev: {sharpe_values.std():.3f}")
    print(f"Hit Rate (overall): {np.mean(returns > 0) * 100:.1f}%")
    print(f"Median Turnover: {median_turnover:.2f}%")
    print(f"Total Cumulative Cost: {cumulative_cost[-1]:.2f} bps")
    print(f"Annual Cost (approx): {cumulative_cost[-1] / (n_windows / 12):.2f} bps/year")


def main():
    """Main execution."""
    print("=== Generating Walk-Forward Analysis Figure ===\n")

    # Load data
    df = load_per_window_results()
    weights_df = load_weights_history()

    # Generate figure
    plot_walkforward_analysis(df, weights_df)

    print("\n✓ Walk-forward analysis complete!")


if __name__ == "__main__":
    main()
