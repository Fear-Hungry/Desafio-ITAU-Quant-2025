# PRISM-R Orchestration Guide

**Complete guide to running the PRISM-R validation pipeline**

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Execution Modes](#execution-modes)
4. [Pipeline Stages](#pipeline-stages)
5. [Output Structure](#output-structure)
6. [CLI Commands Reference](#cli-commands-reference)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Usage](#advanced-usage)

---

## Overview

The PRISM-R project provides a comprehensive orchestration system for portfolio validation:

- **Master Orchestrator** (`scripts/run_master_validation.py`) - Coordinates 7-stage validation pipeline
- **Unified CLI** (`arara-quant`) - 11 commands for backtesting, comparison, optimization
- **Research Scripts** (15 scripts) - Sensitivity analyses and experimentation
- **Validation Suite** (4 scripts) - Constraint and estimator validation

**Key Benefits:**
- ✅ Reproducible validation workflow
- ✅ Automated result aggregation
- ✅ Executive summary generation
- ✅ Parallel execution support (experimental)
- ✅ Resume capability from any stage

---

## Quick Start

### Using Makefile (Recommended)

```bash
# Full validation (20-30 minutes)
make validate-all

# Quick smoke test (5 minutes)
make validate-quick

# Production pre-deploy validation (10 minutes)
make validate-production
```

### Using Python Directly

```bash
# Full validation with all stages
poetry run python scripts/run_master_validation.py --mode full

# Quick test (skips sensitivity and validation suite)
poetry run python scripts/run_master_validation.py --mode quick --skip-download

# Production validation (tests production config only)
poetry run python scripts/run_master_validation.py --mode production
```

### Using Individual CLI Commands

```bash
# Run single backtest
poetry run arara-quant backtest --config configs/optimizer_example.yaml --no-dry-run

# Compare baseline strategies
poetry run arara-quant compare-baselines

# Run full pipeline (data + backtest)
poetry run arara-quant run-full-pipeline --config configs/optimizer_example.yaml
```

---

## Execution Modes

### Mode: `quick`

**Purpose:** Fast smoke test for development/debugging
**Duration:** ~5 minutes
**Stages Executed:**
- Stage 1: Data pipeline (skipped if `--skip-download`)
- Stage 2: Primary backtest (1 config only)
- Stage 3: Baseline comparisons
- Stage 6: Results aggregation
- Stage 7: Report generation

**Skipped:**
- Sensitivity analyses (Stage 4)
- Validation suite (Stage 5)

**Use Case:** Quick sanity check before committing code

```bash
poetry run python scripts/run_master_validation.py \
    --mode quick \
    --skip-download
```

---

### Mode: `full`

**Purpose:** Comprehensive validation for research/final submission
**Duration:** ~20-30 minutes
**Stages Executed:** All 7 stages
- Stage 1: Data pipeline
- Stage 2: Primary backtests (3 configs)
- Stage 3: Baseline comparisons
- Stage 4: Sensitivity analyses (cost, window, covariance)
- Stage 5: Validation suite (4 test scripts)
- Stage 6: Results aggregation
- Stage 7: Report generation

**Use Case:** Before final submission or major changes

```bash
poetry run python scripts/run_master_validation.py \
    --mode full
```

---

### Mode: `production`

**Purpose:** Pre-deployment validation for production config
**Duration:** ~10 minutes
**Stages Executed:**
- Stage 1: Data pipeline (skipped if `--skip-download`)
- Stage 2: Production backtest (`configs/production_erc_v2.yaml`)
- Stage 3: Baseline comparisons
- Stage 5: Validation suite (constraint tests)
- Stage 6: Results aggregation
- Stage 7: Report generation

**Skipped:**
- Sensitivity analyses (Stage 4) - already calibrated

**Use Case:** Pre-deployment checks for production system

```bash
poetry run python scripts/run_master_validation.py \
    --mode production \
    --skip-download
```

---

## Pipeline Stages

### Stage 1: Data Pipeline

**Purpose:** Download and process market data
**Duration:** ~2-5 minutes (first run), ~10 seconds (cached)

**Actions:**
- Downloads ETF prices from yfinance (2010-01-01 to today)
- Applies corporate action adjustments
- Computes log returns
- Winsorizes outliers
- Saves to Parquet format (default: `data/processed/returns_arara.parquet`)

**Command:**
```bash
poetry run python scripts/run_01_data_pipeline.py \
    --force-download \
    --start 2010-01-01
```

**Skip Condition:** Use `--skip-download` to use cached data

---

### Stage 2: Primary Backtests

**Purpose:** Run walk-forward backtest on key configurations
**Duration:** ~5-10 minutes per config

**Configurations Tested:**

| Mode | Configs |
|------|---------|
| `quick` | `configs/optimizer_example.yaml` |
| `full` | `configs/optimizer_example.yaml`<br>`configs/optimizer_regime_aware.yaml`<br>`configs/optimizer_adaptive_hedge.yaml` |
| `production` | `configs/production_erc_v2.yaml` |

**Command (single config):**
```bash
poetry run arara-quant backtest \
    --config configs/optimizer_example.yaml \
    --no-dry-run \
    --wf-report \
    --json > outputs/reports/backtest_example.json
```

**Outputs:**
- `backtest_<config_name>.json` - Serialized backtest results
- Walk-forward report in `outputs/reports/walkforward/`
- Tearsheet visualizations in `outputs/reports/figures/`

**Key Metrics:**
- Sharpe Ratio (HAC-adjusted)
- Max Drawdown
- CVaR (5%)
- Annual Turnover
- Transaction Costs

---

### Stage 3: Baseline Comparisons

**Purpose:** Compare proposed strategy vs simple baselines
**Duration:** ~3-5 minutes

**Baselines:**
1. **Equal-Weight (1/N)** - Naive diversification
2. **Minimum Variance (MV)** - Ledoit-Wolf shrinkage
3. **Risk Parity (RP)** - Equal risk contribution

**Command:**
```bash
poetry run arara-quant compare-baselines
```

**Outputs:**
- `outputs/results/baselines/comparison_metrics.csv`
- `outputs/results/baselines/sharpe_comparison.csv`

**Success Criteria:**
- Proposed Sharpe > Best Baseline Sharpe + 0.10
- Max Drawdown < Best Baseline Drawdown

---

### Stage 4: Sensitivity Analyses

**Purpose:** Test robustness to parameter changes
**Duration:** ~5-10 minutes per analysis

**Analyses:**

1. **Cost Sensitivity** (`run_cost_sensitivity.py`)
   - Tests linear transaction costs: 5, 10, 15, 20 bps
   - Validates Sharpe remains > 0.60 even at 20 bps

2. **Window Sensitivity** (`run_window_sensitivity.py`)
   - Tests estimation windows: 126, 189, 252, 315, 378 days
   - Validates performance across market cycles

3. **Covariance Sensitivity** (`run_covariance_sensitivity.py`)
   - Tests estimators: sample, Ledoit-Wolf, nonlinear shrinkage, Tyler M
   - Validates robustness to covariance estimation

**Command (example):**
```bash
poetry run python scripts/research/run_cost_sensitivity.py
```

**Outputs:**
- `outputs/results/cost_sensitivity/metrics_by_cost.csv`
- `outputs/results/window_sensitivity/metrics_by_window.csv`
- `outputs/results/cov_sensitivity/metrics_by_estimator.csv`

---

### Stage 5: Validation Suite

**Purpose:** Verify constraint satisfaction and estimator correctness
**Duration:** ~5 minutes

**Tests:**

1. **Comprehensive Tests** (`run_comprehensive_tests.py`)
   - Budget constraint (Σw = 1)
   - Box constraints (0 ≤ w_i ≤ u_i)
   - Group constraints (crypto ≤ 5%, US equity ≤ 35%)

2. **Constraint Tests** (`run_constraint_tests.py`)
   - Turnover cap (‖w - w_prev‖₁ ≤ 0.20)
   - Cardinality (20 ≤ K ≤ 35)

3. **Estimator Tests** (`run_estimator_tests.py`)
   - PSD projection correctness
   - Black-Litterman posterior equations
   - Huber M-estimator convergence

4. **Sensitivity Tests** (`run_sensitivity_tests.py`)
   - Stress test regime detection
   - Defensive mode triggers

**Command (example):**
```bash
poetry run python scripts/validation/run_comprehensive_tests.py
```

**Outputs:**
- `outputs/results/validation/constraint_violations.csv`
- `outputs/results/validation/estimator_errors.csv`

---

### Stage 6: Results Aggregation

**Purpose:** Collect all results into master tables
**Duration:** ~30 seconds

**Actions:**
- Collects backtest JSONs from Stage 2
- Aggregates baseline comparisons from Stage 3
- Merges sensitivity analyses from Stage 4
- Compiles validation results from Stage 5

**Outputs:**
- `master_backtest_results.csv` - All backtest metrics
- `master_baseline_results.csv` - Baseline comparisons
- `master_cost_sensitivity.csv` - Cost analysis
- `master_window_sensitivity.csv` - Window tuning
- `master_cov_sensitivity.csv` - Covariance estimator comparison

**Example Aggregation Code:**
```python
import pandas as pd
from pathlib import Path

# Collect all backtest JSONs
backtest_files = Path("outputs/reports/validation_XXX").glob("backtest_*.json")
results = []
for f in backtest_files:
    data = json.load(f.open())
    results.append(data)

# Aggregate into DataFrame
df = pd.DataFrame(results)
df.to_csv("master_backtest_results.csv", index=False)
```

---

### Stage 7: Report Generation

**Purpose:** Generate executive summary with key findings
**Duration:** ~10 seconds

**Report Sections:**
1. Execution summary (runtime, errors)
2. Backtest performance metrics
3. Baseline comparison table
4. Sensitivity analysis highlights
5. Validation test results
6. Next steps

**Outputs:**
- `VALIDATION_SUMMARY.md` - Executive summary
- `execution_log.json` - Detailed command log

**Example Report:**
```markdown
# PRISM-R Validation Summary

**Generated:** 2025-11-01T10:30:00

**Mode:** full

**Completed Stages:** ['stage1', 'stage2', 'stage3', 'stage4', 'stage5', 'stage6', 'stage7']

## Execution Summary

- Total commands: 23
- Successful: 23
- Failed: 0
- Total runtime: 28.5 minutes

## Key Findings

### Backtest Performance
- Sharpe Ratio: 0.482 (regime-aware)
- Max Drawdown: -12.84%
- CVaR (5%): 7.8%

### Baseline Comparison
- 1/N Sharpe: 0.29
- MV Sharpe: 0.44
- RP Sharpe: 0.38
- **Proposed beats best baseline by +0.042**

### Robustness
- Performance stable across cost scenarios (5-20 bps)
- Window sensitivity: 252-day optimal
- All constraints satisfied
```

---

## Output Structure

```
outputs/reports/validation_YYYYMMDD_HHMMSS/
├── VALIDATION_SUMMARY.md          # Executive summary
├── execution_log.json             # Detailed command log
├── backtest_optimizer_example.json
├── backtest_optimizer_regime_aware.json
├── backtest_optimizer_adaptive_hedge.json
├── master_backtest_results.csv    # Aggregated backtest metrics
├── master_baseline_results.csv    # Baseline comparisons
├── master_cost_sensitivity.csv    # Cost analysis
├── master_window_sensitivity.csv  # Window tuning
└── master_cov_sensitivity.csv     # Covariance sensitivity

outputs/reports/walkforward/
├── summary_stats.md               # Walk-forward summary
├── per_window_results.csv         # Results per window
├── stress_periods.md              # Stress period analysis
└── walkforward_analysis.png       # Visualization

outputs/reports/figures/
├── tearsheet_portfolio.png
├── drawdown_evolution.png
├── turnover_tracking.png
├── regime_classification.png
└── cumulative_returns.png

outputs/results/
├── baselines/
│   ├── comparison_metrics.csv
│   └── sharpe_comparison.csv
├── cost_sensitivity/
│   └── metrics_by_cost.csv
├── window_sensitivity/
│   └── metrics_by_window.csv
├── cov_sensitivity/
│   └── metrics_by_estimator.csv
└── validation/
    ├── constraint_violations.csv
    └── estimator_errors.csv
```

---

## CLI Commands Reference

### Core Commands

```bash
# Show current configuration
poetry run arara-quant show-settings [--json]

# Optimize portfolio (dry-run)
poetry run arara-quant optimize --config <yaml>

# Run backtest
poetry run arara-quant backtest --config <yaml> --no-dry-run [--wf-report] [--json]

# Full pipeline (data + backtest)
poetry run arara-quant run-full-pipeline \
    --config <yaml> \
    --start YYYY-MM-DD \
    --end YYYY-MM-DD \
    [--skip-download] \
    [--skip-backtest] \
    [--output-dir DIR] \
    [--json]
```

### Research Commands

```bash
# Compare baseline strategies
poetry run arara-quant compare-baselines

# Compare estimators
poetry run arara-quant compare-estimators

# Grid search hyperparameters
poetry run arara-quant grid-search

# Test forecast skill
poetry run arara-quant test-skill

# Walk-forward validation
poetry run arara-quant walkforward
```

### Production Commands

```bash
# Deploy production portfolio
poetry run arara-quant production-deploy [--version {v1|v2}]
```

### Example Commands

```bash
# Run example portfolios
poetry run arara-quant run-example {arara|robust}
```

---

## Troubleshooting

### Error: "Data file not found"

**Cause:** Data pipeline hasn't been run or cache is stale

**Solution:**
```bash
# Force fresh download
poetry run python scripts/run_01_data_pipeline.py --force-download
```

---

### Error: "CVXPY solver failed"

**Cause:** Infeasible constraints or numerical instability

**Solutions:**

1. **Relax turnover cap:**
   ```yaml
   # In config YAML
   optimizer:
     turnover_penalty: 0.05  # Reduce from 0.10
   ```

2. **Use more robust covariance estimator:**
   ```yaml
   estimators:
     sigma_method: ledoit_wolf  # Instead of 'sample'
   ```

3. **Increase epsilon for PSD projection:**
   ```yaml
   optimizer:
     epsilon: 1e-8  # Increase from 1e-10
   ```

---

### Error: "Stage timeout (>3600s)"

**Cause:** Script running longer than 1 hour

**Solution:** Use `--resume-from` to continue from failed stage

```bash
# Resume from Stage 4
poetry run python scripts/run_master_validation.py \
    --mode full \
    --resume-from stage4
```

---

### Error: "Sharpe below baseline"

**Cause:** Strategy underperforming simple baselines

**Diagnostic Steps:**

1. **Check estimation window:**
   ```bash
   poetry run python scripts/research/run_window_sensitivity.py
   ```

2. **Test different estimators:**
   ```bash
   poetry run arara-quant compare-estimators
   ```

3. **Review constraint tightness:**
   - Are turnover caps too strict?
   - Are cardinality constraints forcing poor selections?

4. **Verify regime detection:**
   ```bash
   poetry run python scripts/research/run_regime_stress.py
   ```

---

### Error: "Constraint violation detected"

**Cause:** Optimizer returned infeasible weights

**Diagnostic:**
```bash
poetry run python scripts/validation/run_constraint_tests.py
```

**Common Violations:**

1. **Budget constraint:** Σw ≠ 1 (tolerance 1e-6)
   - Check for numerical precision issues
   - Increase `epsilon` in config

2. **Turnover cap:** ‖w - w_prev‖₁ > cap
   - Increase turnover cap in config
   - Reduce turnover penalty

3. **Group constraints:** Group allocation > limit
   - Review `configs/asset_groups.yaml`
   - Adjust per-asset limits

---

### Slow Execution

**Optimization:**

1. **Use cached data:**
   ```bash
   poetry run python scripts/run_master_validation.py \
       --mode full \
       --skip-download
   ```

2. **Skip sensitivity analyses:**
   ```bash
   poetry run python scripts/run_master_validation.py \
       --mode full \
       --skip-sensitivity
   ```

3. **Use quick mode for development:**
   ```bash
   make validate-quick
   ```

4. **Enable parallel execution (experimental):**
   ```bash
   poetry run python scripts/run_master_validation.py \
       --mode full \
       --parallel
   ```

---

## Advanced Usage

### Resume from Specific Stage

If pipeline fails at Stage 4, resume without re-running earlier stages:

```bash
poetry run python scripts/run_master_validation.py \
    --mode full \
    --resume-from stage4
```

---

### Custom Output Directory

Specify custom directory for results:

```bash
poetry run python scripts/run_master_validation.py \
    --mode full \
    --output-dir outputs/reports/custom_validation
```

---

### Parallel Execution (Experimental)

Run independent analyses concurrently:

```bash
poetry run python scripts/run_master_validation.py \
    --mode full \
    --parallel
```

**Warning:** Experimental feature. May cause race conditions in file I/O.

---

### Running Individual Stages

For debugging, run stages individually:

```bash
# Stage 1: Data
poetry run python scripts/run_01_data_pipeline.py --force-download

# Stage 2: Backtest
poetry run arara-quant backtest --config configs/optimizer_example.yaml --no-dry-run

# Stage 3: Baselines
poetry run arara-quant compare-baselines

# Stage 4: Sensitivity (example)
poetry run python scripts/research/run_cost_sensitivity.py

# Stage 5: Validation (example)
poetry run python scripts/validation/run_comprehensive_tests.py
```

---

### Integrating New Research Scripts

To add a new research script to the pipeline:

1. **Create script in `scripts/research/`:**
   ```python
   # scripts/research/run_my_experiment.py

   def main():
       # Your analysis
       results.to_csv("outputs/results/my_experiment/metrics.csv")

   if __name__ == "__main__":
       main()
   ```

2. **Add to orchestrator:**
   ```python
   # In scripts/run_master_validation.py

   SENSITIVITY_SCRIPTS = [
       # ... existing scripts
       "scripts/research/run_my_experiment.py",
   ]
   ```

3. **Update aggregation logic:**
   ```python
   # In stage6_aggregate_results()

   if (RESULTS_DIR / "my_experiment").exists():
       df = pd.read_csv("outputs/results/my_experiment/metrics.csv")
       df.to_csv(self.output_dir / "master_my_experiment.csv")
   ```

---

### CI/CD Integration

**GitHub Actions Workflow Example:**

```yaml
# .github/workflows/validate.yml

name: Portfolio Validation

on:
  push:
    branches: [main]
  pull_request:

jobs:
  validate:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Install Poetry
        run: pipx install poetry

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          cache: 'poetry'

      - name: Install dependencies
        run: poetry install

      - name: Run quick validation
        run: make validate-quick

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: validation-results
          path: outputs/reports/validation_*/
```

---

## Best Practices

1. **Always use `--skip-download` after first run** to save time
2. **Run `validate-quick` before committing** changes
3. **Run `validate-full` before final submission**
4. **Review `VALIDATION_SUMMARY.md`** for errors and warnings
5. **Check `execution_log.json`** for detailed diagnostics
6. **Use `--resume-from` if pipeline fails** partway through
7. **Keep configs in version control** for reproducibility
8. **Document parameter choices** in config YAML comments

---

## Next Steps

After running validation:

1. **Review Results:**
   - Check `master_backtest_results.csv` for Sharpe > 0.80
   - Verify baseline beat by ≥ 0.10
   - Confirm constraints satisfied

2. **Update Documentation:**
   - Add metrics to `README.md` Section 5
   - Document any configuration changes

3. **Iterate if Needed:**
   - If Sharpe < target, run grid search
   - If constraints violated, adjust config
   - If unstable, test different estimators

4. **Deploy to Production:**
   ```bash
   make validate-production
   poetry run arara-quant production-deploy --version v2
   ```

---

## Support

For issues or questions:

- Review `VALIDATION_SUMMARY.md` in output directory
- Check `execution_log.json` for detailed errors
- Consult `docs/QUICKSTART.md` for basic usage
- See `docs/specs/PRD.md` for mathematical formulation
- Review `IMPLEMENTACAO_ROBUSTA.md` for estimator details

---

**Last Updated:** 2025-11-01
**Version:** 1.0.0
