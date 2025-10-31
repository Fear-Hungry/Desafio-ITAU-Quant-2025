#!/usr/bin/env python
"""Generate summary charts for OOS performance and sensitivities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")

RESULTS_DIR = Path("results")
FIG_DIR = Path("reports") / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def latest_oos_file() -> Path:
    files = sorted(RESULTS_DIR.glob("oos_returns_all_strategies_*.csv"))
    if not files:
        raise FileNotFoundError("No OOS returns file found in results/.")
    return files[-1]


def load_returns() -> tuple[pd.DataFrame, str]:
    path = latest_oos_file()
    df = pd.read_csv(path, index_col=0, parse_dates=True).astype(float)
    return df, path.stem.split("_")[-1]


def plot_nav(df: pd.DataFrame, suffix: str) -> None:
    columns = []
    for candidate in ["Risk Parity", "1/N", "60/40", "ACWI"]:
        if candidate in df.columns:
            columns.append(candidate)
    if len(columns) < 2:
        raise ValueError("Not enough series available to plot NAV comparison.")

    nav = (1.0 + df[columns]).cumprod()
    fig, ax = plt.subplots(figsize=(10, 6))
    nav.plot(ax=ax)
    ax.set_title("Cumulative NAV Comparison")
    ax.set_ylabel("NAV (rebased to 1.0)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"nav_comparison_{suffix}.png", dpi=200)
    plt.close(fig)


def plot_monthly_distribution(df: pd.DataFrame, suffix: str) -> None:
    required = [col for col in ("Risk Parity", "60/40") if col in df.columns]
    if len(required) < 2:
        raise ValueError("Risk Parity and 60/40 series are required for monthly distribution plot.")

    monthly = df[required].resample("ME").sum()
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(monthly.min().min(), monthly.max().max(), 25)
    for name in required:
        ax.hist(monthly[name], bins=bins, alpha=0.6, label=name, density=True)
    ax.set_title("Distribution of Monthly Returns")
    ax.set_xlabel("Monthly Return")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"monthly_distribution_{suffix}.png", dpi=200)
    plt.close(fig)


def plot_sharpe_with_ci(bootstrap_path: Path, suffix: str) -> None:
    data = json.loads(bootstrap_path.read_text())
    results = data.get("results", {})
    if not results:
        raise ValueError("Bootstrap file does not contain results.")

    strategies = []
    sharpe = []
    err_low = []
    err_high = []
    for name, stats in results.items():
        strategies.append(name)
        sharpe.append(stats["sharpe_point"])
        err_low.append(stats["sharpe_point"] - stats["sharpe_ci_low"])
        err_high.append(stats["sharpe_ci_high"] - stats["sharpe_point"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(strategies, sharpe, yerr=[err_low, err_high], capsize=6, color="#4C72B0")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title(f"Sharpe with {int(data['meta']['confidence'] * 100)}% CI (Block size {data['meta']['block_size']})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"sharpe_confidence_{suffix}.png", dpi=200)
    plt.close(fig)


def cost_sensitivity_plot(df: pd.DataFrame, suffix: str) -> None:
    if "Risk Parity" not in df.columns:
        raise ValueError("Risk Parity series not available for cost sensitivity.")

    base_cost = 30.0
    costs = [30.0, 50.0, 75.0]
    sharpe_values = []
    ann_returns = []

    base_series = df["Risk Parity"].copy()

    for cost in costs:
        series = base_series.copy()
        extra = (cost - base_cost) / 10_000.0
        if extra > 0:
            series.iloc[::21] -= extra
        mean = series.mean()
        std = series.std(ddof=1)
        sharpe = mean / std * np.sqrt(252.0) if std > 0 else 0.0
        ann_return = (1.0 + series).prod() ** (252.0 / len(series)) - 1.0
        sharpe_values.append(sharpe)
        ann_returns.append(ann_return)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(costs, sharpe_values, marker="o", label="Sharpe ratio")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Round-trip transaction cost (bps)")
    ax.set_ylabel("Sharpe ratio (approx.)")
    ax.set_title("ERC Cost Sensitivity (approximate)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"cost_sensitivity_{suffix}.png", dpi=200)
    plt.close(fig)


def main() -> None:
    returns, suffix = load_returns()
    plot_nav(returns, suffix)
    plot_monthly_distribution(returns, suffix)

    bootstrap_files = sorted((RESULTS_DIR / "bootstrap_ci").glob("bootstrap_sharpe_*.json"))
    if not bootstrap_files:
        raise FileNotFoundError("Bootstrap CI file not found. Run run_bootstrap_ci.py first.")
    plot_sharpe_with_ci(bootstrap_files[-1], suffix)

    cost_sensitivity_plot(returns, suffix)
    print(f"Charts written to {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()
