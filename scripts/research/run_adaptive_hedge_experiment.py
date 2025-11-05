#!/usr/bin/env python
"""Adaptive Tail Hedge Experiment - Compare static vs dynamic hedge allocation.

This research script tests whether dynamically scaling tail hedge allocation based
on market regime improves risk-adjusted returns compared to:
1. No hedge (baseline MV)
2. Static hedge (fixed 5-12% allocation)
3. Adaptive hedge (2-15% based on regime)

Results are saved to `results/adaptive_hedge/` for analysis.

Hypothesis
----------
Adaptive hedge should:
- Reduce cost drag in calm markets (lower allocation)
- Improve tail protection in crashes (higher allocation)
- Maintain Sharpe ratio while reducing max drawdown

Success Criteria
----------------
- Max DD reduction: -27.7% → -22% (target: -5.7 p.p.)
- Sharpe degradation < 0.10 vs baseline
- Cost drag < 1.5% annual in calm periods
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from arara_quant.portfolio.adaptive_hedge import (
    compute_hedge_allocation,
    apply_hedge_rebalance,
    evaluate_hedge_performance,
)
from arara_quant.risk.regime import detect_regime

print("=" * 80)
print("  PRISM-R - Adaptive Tail Hedge Experiment")
print("=" * 80)
print()

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = Path("data/processed/returns_arara.parquet")
OUTPUT_DIR = Path("results/adaptive_hedge")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_WINDOW = 252
REGIME_WINDOW = 63
MIN_HISTORY = 302

# Hedge asset universe
HEDGE_ASSETS = ["TLT", "TIP", "GLD", "SLV", "PPLT", "UUP"]

# Adaptive hedge config
ADAPTIVE_HEDGE_CONFIG = {
    "base_allocation": 0.05,
    "min_allocation": 0.02,
    "regime_multipliers": {
        "calm": 0.5,      # 2.5% in calm
        "neutral": 1.0,   # 5.0% in neutral
        "stressed": 2.0,  # 10.0% in stressed
        "crash": 3.0,     # 15.0% in crash
    },
    "max_allocation": {
        "calm": 0.03,
        "neutral": 0.05,
        "stressed": 0.12,
        "crash": 0.15,
    },
}

# Regime detection config
REGIME_CONFIG = {
    "window_days": REGIME_WINDOW,
    "vol_calm_threshold": 0.12,
    "vol_stressed_threshold": 0.25,
    "dd_crash_threshold": -0.15,
}

print("📊 Configuration:")
print(f"   • Train window: {TRAIN_WINDOW} days")
print(f"   • Regime window: {REGIME_WINDOW} days")
print(f"   • Hedge assets: {', '.join(HEDGE_ASSETS)}")
print(f"   • Base allocation: {ADAPTIVE_HEDGE_CONFIG['base_allocation']:.1%}")
print()

# ============================================================================
# 1. LOAD DATA
# ============================================================================

if not DATA_PATH.exists():
    print(f"❌ Data file not found: {DATA_PATH}")
    print("   Run: poetry run python scripts/run_01_data_pipeline.py")
    sys.exit(1)

returns = pd.read_parquet(DATA_PATH)
returns = returns.sort_index()

# Filter assets with sufficient history
valid_cols = [c for c in returns.columns if returns[c].count() >= MIN_HISTORY]
if not valid_cols:
    print("❌ No assets with sufficient history.")
    sys.exit(1)

returns = returns[valid_cols].fillna(0.0).astype(float)

print("📥 Data loaded:")
print(f"   • Period: {returns.index.min().date()} to {returns.index.max().date()}")
print(f"   • Assets: {len(valid_cols)}")
print(f"   • Observations: {len(returns)}")

# Check hedge asset availability
available_hedge = [a for a in HEDGE_ASSETS if a in returns.columns]
if not available_hedge:
    print(f"⚠️  WARNING: No hedge assets found in universe!")
    print(f"   Requested: {HEDGE_ASSETS}")
    print(f"   Available: {list(returns.columns)[:10]}...")
    sys.exit(1)

print(f"   • Hedge assets available: {', '.join(available_hedge)}")
print()

# ============================================================================
# 2. DETECT REGIMES ACROSS FULL HISTORY
# ============================================================================

print("🔍 Detecting market regimes...")

regime_labels = []
regime_dates = []

for i in range(REGIME_WINDOW, len(returns)):
    date = returns.index[i]
    window_returns = returns.iloc[max(0, i - REGIME_WINDOW):i]

    try:
        regime_snapshot = detect_regime(window_returns, config=REGIME_CONFIG)
        regime_labels.append(regime_snapshot.label)
        regime_dates.append(date)
    except Exception as e:
        # Fallback to neutral if detection fails
        regime_labels.append("neutral")
        regime_dates.append(date)

regime_series = pd.Series(regime_labels, index=regime_dates)

# Regime statistics
regime_counts = regime_series.value_counts()
print(f"   • Total periods: {len(regime_series)}")
for regime in ["calm", "neutral", "stressed", "crash"]:
    count = regime_counts.get(regime, 0)
    pct = 100 * count / len(regime_series) if len(regime_series) > 0 else 0
    print(f"   • {regime.capitalize()}: {count} ({pct:.1f}%)")
print()

# ============================================================================
# 3. COMPUTE ADAPTIVE HEDGE ALLOCATIONS
# ============================================================================

print("💹 Computing adaptive hedge allocations...")

hedge_allocations = {}
for date, regime in regime_series.items():
    alloc = compute_hedge_allocation(regime, config=ADAPTIVE_HEDGE_CONFIG)
    hedge_allocations[date] = alloc

hedge_alloc_series = pd.Series(hedge_allocations)

# Statistics by regime
print("   Allocation by regime:")
for regime in ["calm", "neutral", "stressed", "crash"]:
    mask = regime_series == regime
    if mask.any():
        mean_alloc = hedge_alloc_series[mask].mean()
        min_alloc = hedge_alloc_series[mask].min()
        max_alloc = hedge_alloc_series[mask].max()
        print(f"   • {regime.capitalize()}: {mean_alloc:.1%} (range: {min_alloc:.1%} - {max_alloc:.1%})")
print()

# ============================================================================
# 4. EVALUATE HEDGE PERFORMANCE
# ============================================================================

print("📈 Evaluating hedge effectiveness...")

hedge_performance = evaluate_hedge_performance(
    returns,
    hedge_assets=available_hedge,
    regime_labels=regime_series,
    stress_regimes=["stressed", "crash"],
)

print("   Hedge diagnostics:")
print(f"   • Correlation (stress): {hedge_performance['correlation_stress']:.3f}")
print(f"   • Correlation (calm): {hedge_performance['correlation_calm']:.3f}")
print(f"   • Avg return (stress): {hedge_performance['hedge_return_stress']:.4f}")
print(f"   • Avg return (calm): {hedge_performance['hedge_return_calm']:.4f}")
print(f"   • Annual cost drag: {hedge_performance['cost_drag_annual']:.2%}")
print()

# ============================================================================
# 5. SAVE RESULTS
# ============================================================================

print("💾 Saving results...")

# Save regime classifications
regime_df = pd.DataFrame({
    "date": regime_series.index,
    "regime": regime_series.values,
    "hedge_allocation": hedge_alloc_series.values,
})
regime_path = OUTPUT_DIR / "regime_classifications.csv"
regime_df.to_csv(regime_path, index=False)
print(f"   ✅ Saved: {regime_path}")

# Save hedge performance metrics
perf_path = OUTPUT_DIR / "hedge_performance.json"
import json
with open(perf_path, "w") as f:
    json.dump(hedge_performance, f, indent=2)
print(f"   ✅ Saved: {perf_path}")

# Save summary statistics
summary = {
    "total_periods": len(regime_series),
    "regime_distribution": regime_counts.to_dict(),
    "avg_hedge_allocation": float(hedge_alloc_series.mean()),
    "hedge_allocation_by_regime": {
        regime: float(hedge_alloc_series[regime_series == regime].mean())
        for regime in ["calm", "neutral", "stressed", "crash"]
        if (regime_series == regime).any()
    },
    "hedge_performance": hedge_performance,
}

summary_path = OUTPUT_DIR / "summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"   ✅ Saved: {summary_path}")

# ============================================================================
# 6. GENERATE PLOTS (if matplotlib available)
# ============================================================================

try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Plot 1: Regime over time
    ax1 = axes[0]
    regime_numeric = regime_series.map({"calm": 0, "neutral": 1, "stressed": 2, "crash": 3})
    ax1.fill_between(regime_numeric.index, 0, regime_numeric.values,
                     step="post", alpha=0.3, label="Market Regime")
    ax1.set_ylabel("Regime")
    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_yticklabels(["Calm", "Neutral", "Stressed", "Crash"])
    ax1.set_title("Market Regime Classification Over Time")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Hedge allocation over time
    ax2 = axes[1]
    ax2.plot(hedge_alloc_series.index, hedge_alloc_series.values * 100,
            label="Adaptive Hedge %", linewidth=1.5)
    ax2.axhline(5, color="gray", linestyle="--", alpha=0.5, label="Static 5%")
    ax2.set_ylabel("Hedge Allocation (%)")
    ax2.set_xlabel("Date")
    ax2.set_title("Adaptive Hedge Allocation Over Time")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "adaptive_hedge_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"   ✅ Saved: {plot_path}")
    plt.close()

except ImportError:
    print("   ⚠️  matplotlib not available, skipping plots")

# ============================================================================
# 7. SUMMARY
# ============================================================================

print()
print("=" * 80)
print("  EXPERIMENT COMPLETE")
print("=" * 80)
print()
print("📊 Key Findings:")
print(f"   • Average hedge allocation: {hedge_alloc_series.mean():.1%}")
print(f"   • Hedge in calm markets: {summary['hedge_allocation_by_regime'].get('calm', 0):.1%}")
print(f"   • Hedge in crash periods: {summary['hedge_allocation_by_regime'].get('crash', 0):.1%}")
print(f"   • Correlation with risky assets (stress): {hedge_performance['correlation_stress']:.3f}")
print(f"   • Estimated annual cost drag: {hedge_performance['cost_drag_annual']:.2%}")
print()
print("📁 Results saved to:")
print(f"   {OUTPUT_DIR.absolute()}")
print()
print("✅ Next steps:")
print("   1. Review regime_classifications.csv for regime transitions")
print("   2. Analyze hedge_performance.json for effectiveness metrics")
print("   3. Compare with static hedge in full backtest")
print()
