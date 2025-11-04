# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-11

### Added
- **Walk-Forward Performance Reporting** with comprehensive analysis
  - 162 OOS windows validation (2010-2025)
  - Sharpe OOS: 1.30, Annual Return: 13.60%
  - Success Rate: 59.9% (97/162 profitable windows)
  - Automated stress period detection (56 periods identified)
  - Visual dashboard with parameter evolution, Sharpe bars, consistency scatter
  - CLI flag `--wf-report` for report generation

- **Backtesting Engine** with purge/embargo validation
  - Walk-forward splits with temporal ordering
  - Transaction cost modeling (linear + slippage)
  - Portfolio rebalancing with turnover constraints
  - Comprehensive metrics (Sharpe HAC, Sortino, CVaR, Calmar)

- **Robust Estimators**
  - Black-Litterman framework (reverse optimization, view projection, posterior)
  - Covariance: Ledoit-Wolf, nonlinear shrinkage, Tyler M-estimator
  - Mean returns: Huber, Student-t, Bayesian shrinkage
  - Factor models (time-series/cross-sectional regression, PCA)

- **Optimization Core**
  - Mean-variance QP with costs and turnover penalties
  - CVaR-based optimization (LP/SOCP)
  - Risk parity allocation
  - Cardinality constraints (heuristics + GA)

- **Configuration System**
  - Pydantic-based YAML validation
  - Multiple universe configurations (ARARA basic/robust)
  - Portfolio and production configs
  - Asset group constraints

- **CLI Interface** with 12 commands
  - Core: `optimize`, `backtest`, `show-settings`
  - Pipeline: `run-full-pipeline`
  - Research: `compare-baselines`, `compare-estimators`, `walkforward`
  - Production: `production-deploy` (ERC v1/v2)

- **Testing Infrastructure**
  - 685+ test functions across 187 test files
  - Test-to-code ratio: 0.62
  - Comprehensive coverage for estimators, optimization, backtesting
  - Config loader and schemas tests (NEW)

- **CI/CD Pipeline** (NEW)
  - GitHub Actions workflow for test/lint/type-check
  - Pre-commit hooks (ruff, black, mypy)
  - Python 3.11 and 3.12 matrix testing
  - Coverage reporting (pytest-cov)

- **Documentation**
  - Comprehensive README with results and methodology
  - Walk-forward analysis section with 162 windows stats
  - CLAUDE.md for AI assistant guidance
  - Reproducibility instructions

### Changed
- Updated pyproject.toml with proper authors and license field
- Corrected .gitignore to exclude artifacts (reports/*.json, *.png)
- Removed poetry.lock from .gitignore (proper for applications)

### Fixed
- Turnover slack variable in MV QP solver for numerical stability
- Markdown table generation without tabulate dependency

### Performance
- **Out-of-Sample (162 windows):**
  - Sharpe: 1.30 (HAC-adjusted)
  - Annual Return: 13.60%
  - Volatility: 10.05%
  - Max Drawdown (avg per window): -2.70%
  - Turnover: 0.62% (ultra-low)
  - Cost: 0.1 bps
  - Consistency R²: 0.051

- **Stress Test Results:**
  - 56 stress periods identified (34.6% of windows)
  - Pandemic 2020: 4 critical windows (worst: -25.3% DD)
  - Inflation 2022: 5 severe windows (worst: -7.11% DD)
  - Banking Crisis 2023: 1 window (-4.67% DD)

### Security
- Added MIT LICENSE
- No credentials or secrets in repository

[0.1.0]: https://github.com/Fear-Hungry/Desafio-ITAU-Quant/releases/tag/v0.1.0
