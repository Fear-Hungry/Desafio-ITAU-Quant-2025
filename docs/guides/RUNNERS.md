# Runners Directory

This directory contains executable Python runner modules organized by purpose. All runners can be executed via `python -m` or through the unified `arara-quant` CLI.

## Directory Structure

```
src/arara_quant/runners/
├── core/         # Pipeline wrappers (data → estimates → optimize)
├── reporting/    # OOS metrics, figures, and markdown reports
├── validation/   # Validation runners + master orchestrator
├── baselines/    # Baseline utilities/exports
├── data/         # Data utilities (e.g., risk-free series fetch)
├── examples/     # Demonstration and tutorial runners
├── research/     # Research and analysis runners
└── production/   # Production deployment runners
```

## Usage

### Direct Execution
```bash
# Run a runner directly
python -m arara_quant.runners.examples.run_portfolio_arara

# Canonical OOS reporting pipeline (from nav_daily.csv)
python -m arara_quant.runners.reporting.consolidate_oos_metrics --psr-n-trials 1
python -m arara_quant.runners.reporting.generate_oos_figures
```

### Via CLI (Recommended)
```bash
# Install CLI entry point
poetry install

# Run through unified interface
poetry run arara-quant data --force-download
poetry run arara-quant estimate --config configs/optimization/optimizer_example.yaml
poetry run arara-quant run-full-pipeline --config configs/optimization/optimizer_example.yaml --skip-backtest
poetry run arara-quant run-example arara
poetry run arara-quant compare-baselines
poetry run arara-quant production-deploy --version v2
```

## Runner Categories

### Examples (`examples/`)
Demonstration runners for learning and testing:
- `run_portfolio_arara.py` - Basic ARARA portfolio example
- `run_portfolio_arara_robust.py` - Robust optimization with advanced features

**CLI:** `poetry run arara-quant run-example [arara|robust]`

### Research (`research/`)
Analysis and experimentation runners:
- `run_baselines_comparison.py` - Compare baseline strategies (1/N, MV, RP)
- `run_estimator_comparison.py` - Compare μ and Σ estimators
- `run_grid_search_shrinkage.py` - Grid search for shrinkage parameters
- `run_mu_skill_test.py` - Test μ forecasting skill
- `run_backtest_walkforward.py` - Walk-forward backtest validation

**CLI:**
```bash
poetry run arara-quant compare-baselines
poetry run arara-quant compare-estimators
poetry run arara-quant grid-search
poetry run arara-quant test-skill
poetry run arara-quant walkforward
```

### Production (`production/`)
Production-ready deployment runners:
- `run_portfolio_production_erc.py` - Risk Parity (ERC) production system
- `run_portfolio_production_erc_v2.py` - Calibrated ERC system (recommended)

**CLI:** `poetry run arara-quant production-deploy [--version v1|v2]`

## Development

All runner modules in this directory:
1. Import from `src/arara_quant` package (not from root)
2. Can be run via `python -m` or via CLI
3. Follow the project's code style (ruff, black)
4. Are version controlled with git

## Migration Notes

These runner modules were previously located in the legacy command-line folder. They have been reorganized into `src/arara_quant/runners/` for better maintainability and clarity.
