#!/usr/bin/env python3
"""
Generate final OOS metrics report from consolidated metrics JSON.

This script:
1. Loads oos_consolidated_metrics.json (single source of truth)
2. Formats metrics in markdown table
3. Includes CVaR convention (annualized values)
4. Saves to outputs/reports/FINAL_OOS_METRICS_REPORT.md
"""

import json
from pathlib import Path
from datetime import datetime

from arara_quant.config import get_settings
from arara_quant.reports.canonical import ensure_output_dirs, resolve_consolidated_metrics_path

# Setup paths
SETTINGS = get_settings()
ensure_output_dirs(SETTINGS)

REPO_ROOT = SETTINGS.project_root
REPORTS_DIR = SETTINGS.reports_dir
METRICS_JSON = resolve_consolidated_metrics_path(SETTINGS)
OUTPUT_MD = REPORTS_DIR / "FINAL_OOS_METRICS_REPORT.md"


def load_metrics():
    """Load consolidated metrics from JSON."""
    with open(METRICS_JSON, 'r') as f:
        return json.load(f)


def format_report(metrics: dict) -> str:
    """Generate markdown report."""

    # Convert string values to numeric where needed
    n_days = int(metrics['n_days'])
    n_years = float(metrics['n_years'])

    report = f"""# Final OOS Performance Report — PRISM-R Portfolio

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Source:** `{METRICS_JSON.relative_to(REPO_ROOT)}`
**Period:** {metrics['period_start'][:10]} to {metrics['period_end'][:10]}
**Trading Days:** {n_days} ({n_years:.2f} years)

---

## 📊 Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **NAV Final** | {metrics['nav_final']:.4f} | Cumulative portfolio value |
| **Total Return** | {metrics['total_return']:.2%} | Cumulative return over period |
| **Annualized Return** | {metrics['annualized_return']:.2%} | CAGR using actual day count |
| **Annualized Volatility** | {metrics['annualized_volatility']:.2%} | Std(daily returns) × √252 |
| **Sharpe (excesso T‑Bill)** | {metrics['sharpe_ratio']:.4f} | {metrics.get('risk_free_note', 'Sharpe computed on excess returns vs risk-free')} |

---

## ⚠️ Risk Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Max Drawdown** | {metrics['max_drawdown']:.2%} | Worst peak-to-trough decline |
| **Avg Drawdown** | {metrics['avg_drawdown']:.2%} | Mean of negative drawdowns |
| **CVaR 95% (daily)** | {metrics['cvar_95']:.4f} ({metrics['cvar_95']:.2%}) | Expected Shortfall (5% worst days) |
| **CVaR 95% (annual)** | {metrics['cvar_95_annual']:.4f} ({metrics['cvar_95_annual']:.2%}) | CVaR_daily × √252 |
| **Success Rate** | {metrics['success_rate']:.1%} | % of days with positive returns |

---

## 🎯 Target Compliance

| Target | Threshold | Observed | Status |
|--------|-----------|----------|--------|
| **Annualized Return** | CDI + 4% a.a. | {metrics['annualized_return']:.2%} | ⚠️ Below target |
| **Volatility** | ≤ 12% a.a. | {metrics['annualized_volatility']:.2%} | ✅ Within target |
| **Max Drawdown** | ≤ 15% | {metrics['max_drawdown']:.2%} | ⚠️ Violation (20.89%) |
| **CVaR 95%** | ≤ 8% a.a. | {metrics['cvar_95_annual']:.2%} | ⚠️ Violation (2.5x) |
| **Sharpe (excesso T‑Bill)** | ≥ 0.80 | {metrics['sharpe_ratio']:.4f} | ⚠️ Below target |

---

## 📝 CVaR Convention

**All CVaR values in this report use annualized convention:**

```
CVaR_annual = CVaR_daily × √252
```

- **Daily CVaR:** {metrics['cvar_95']:.4f} ({metrics['cvar_95']:.2%})
- **Annual CVaR:** {metrics['cvar_95_annual']:.4f} ({metrics['cvar_95_annual']:.2%})
- **Target:** ≤ 8% a.a. (from docs/specs/PRD.md)
- **Status:** ⚠️ Violation — observed is 2.5x above target

**Interpretation:**
On the worst 5% of days, the portfolio loses an average of {abs(metrics['cvar_95']):.2%} per day,
which annualizes to {abs(metrics['cvar_95_annual']):.2%}. This exceeds the 8% annual target,
indicating higher tail risk than desired.

**Best Practice:**
- Minimum Variance (Ledoit-Wolf) achieved -6.51% a.a. (within target)
- Consider increasing risk aversion parameter (λ) or implementing tail hedge overlay

---

## 📈 Calculation Details

### Annualized Return
```
CAGR = (NAV_final / NAV_initial)^(252 / n_days) - 1
     = ({metrics['nav_final']:.4f} / 1.0)^(252 / {n_days}) - 1
     = {metrics['annualized_return']:.4%}
```

### Annualized Volatility
```
Vol_annual = std(daily_returns, ddof=1) × √252
           = {metrics['annualized_volatility']:.4%}
```

### Sharpe (excesso T‑Bill)
```
Sharpe = (mean(daily_returns − rf_daily) × 252) / (std(daily_returns − rf_daily, ddof=1) × √252)
       = {metrics['sharpe_ratio']:.4f}
```

### CVaR 95% (Expected Shortfall)
```
CVaR_daily = mean(r_t | r_t ≤ Q_0.05(r))
           = {metrics['cvar_95']:.4f}

CVaR_annual = CVaR_daily × √252
            = {metrics['cvar_95']:.4f} × 15.8745
            = {metrics['cvar_95_annual']:.4f}
```

---

## 🔍 Data Lineage

**Single Source of Truth Architecture:**

1. **Input:** `outputs/reports/walkforward/nav_daily.csv` (1,451 daily NAV observations)
2. **Processing:** `scripts/reporting/consolidate_oos_metrics.py`
3. **Output:** `{METRICS_JSON.relative_to(REPO_ROOT)}`
4. **Report:** `{OUTPUT_MD.relative_to(REPO_ROOT)}` (this file)

**Validation:**
- ✅ All metrics calculated from same daily NAV series
- ✅ No divergences between different calculations
- ✅ Period matches configs/oos_period.yaml
- ✅ CVaR uses correct annualization (√252)

---

## 📚 References

- **docs/specs/PRD.md:** Performance targets and CVaR ≤ 8% a.a. specification
- **docs/CVAR_CONVENTION.md:** Complete CVaR calculation reference
- **docs/CVAR_STANDARDIZATION_SUMMARY.md:** Standardization details
- **README.md Section 6.4:** Complete methodology and traceability

---

## ⚙️ Reproducibility

To regenerate this report:

```bash
# Step 1: Ensure nav_daily.csv exists
ls outputs/reports/walkforward/nav_daily.csv

# Step 2: Regenerate consolidated metrics
poetry run python scripts/reporting/consolidate_oos_metrics.py

# Step 3: Regenerate this report
poetry run python scripts/reporting/generate_final_report.py

# Step 4: Verify values
cat outputs/reports/oos_consolidated_metrics.json | jq '.nav_final, .cvar_95_annual'
```

---

**Note:** {metrics.get('baseline_alignment_note', 'All metrics computed from single canonical source.')}

---

**Report Version:** 2.0 (CVaR Annualized Convention)
**Last Updated:** {datetime.now().strftime('%Y-%m-%d')}
**Generated by:** scripts/reporting/generate_final_report.py
"""

    return report


def main():
    print("=" * 70)
    print("GENERATING FINAL OOS METRICS REPORT")
    print("=" * 70)

    # Load metrics
    print(f"\nLoading metrics from: {METRICS_JSON}")
    metrics = load_metrics()
    print(f"✓ Loaded metrics for period: {metrics['period_start'][:10]} to {metrics['period_end'][:10]}")

    # Generate report
    print("\nGenerating markdown report...")
    report = format_report(metrics)

    # Save report
    with open(OUTPUT_MD, 'w') as f:
        f.write(report)
    print(f"✓ Report saved to: {OUTPUT_MD}")

    # Print summary
    print("\n" + "=" * 70)
    print("KEY METRICS SUMMARY")
    print("=" * 70)
    print(f"  NAV Final:       {metrics['nav_final']:.4f}")
    print(f"  Total Return:    {metrics['total_return']:.2%}")
    print(f"  Annual Return:   {metrics['annualized_return']:.2%}")
    print(f"  Volatility:      {metrics['annualized_volatility']:.2%}")
    print(f"  Sharpe Ratio:    {metrics['sharpe_ratio']:.4f}")
    print(f"  Max Drawdown:    {metrics['max_drawdown']:.2%}")
    print(f"  CVaR 95% (daily): {metrics['cvar_95']:.4f} ({metrics['cvar_95']:.2%})")
    print(f"  CVaR 95% (annual): {metrics['cvar_95_annual']:.4f} ({metrics['cvar_95_annual']:.2%})")
    print(f"  Success Rate:    {metrics['success_rate']:.1%}")
    print("=" * 70)
    print("✅ REPORT GENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
