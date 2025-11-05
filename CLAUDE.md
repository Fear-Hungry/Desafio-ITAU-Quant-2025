# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**DON'T MAKE IT**
- DON'T CREATE '.md' FILES
- I KNOW I'M RIGHT, JUST MAKE IT WITH NOT MISTAKES

## Project Overview

**PRISM-R (Portfolio Risk Intelligence System)** - Carteira ARARA is a quantitative multi-asset portfolio optimization platform designed for the ITAU Quant Challenge. The system implements robust portfolio optimization with real transaction costs, turnover constraints, and walk-forward validation.

**Core Mission:** Deliver risk-adjusted returns (target CDI + 4% annually) with strict risk control (volatility ≤ 12%, max drawdown ≤ 15%) across a global ETF universe of 40+ assets.

## Essential Commands

### Environment Setup
```bash
# Install dependencies
poetry install

# Activate virtual environment (if needed)
poetry shell
```

### Testing
```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tests/estimators/test_bl.py

# Run single test function
poetry run pytest tests/estimators/test_bl.py::test_reverse_optimization_infers_risk_aversion

# Run with verbose output
poetry run pytest -v

# Run with coverage
poetry run pytest --cov=src/itau_quant
```

### Code Quality
```bash
# Lint code with ruff
poetry run ruff check src tests

# Format code with black
poetry run black src tests

# Type checking (when implemented)
poetry run mypy src
```

### Data Pipeline
```bash
# Download and process data (when implemented)
poetry run python -m itau_quant.data.loader --universe configs/universe_arara.yaml
```

## Architecture Overview

The codebase follows a layered architecture designed for modularity and testability:

```
┌─────────────────────────────┐
│  Data Layer                 │ ← Data ingestion, cleaning, storage (Parquet)
├─────────────────────────────┤
│  Estimators                 │ ← Robust μ/Σ estimation, Black-Litterman, factors
├─────────────────────────────┤
│  Optimization Core          │ ← QP/SOCP solvers with costs, turnover, cardinality
├─────────────────────────────┤
│  Portfolio Management       │ ← Rebalancing, rounding, scheduling
├─────────────────────────────┤
│  Backtesting Engine         │ ← Walk-forward validation with purging/embargo
├─────────────────────────────┤
│  Evaluation & Reporting     │ ← OOS metrics, tearsheets, 10-page report
└─────────────────────────────┘
```

### Key Module Responsibilities

**`src/itau_quant/estimators/`** - Statistical estimation layer
- **`bl.py`**: Complete Black-Litterman framework (reverse optimization, view projection, posterior returns)
- **`cov.py`**: Covariance estimators (sample, Ledoit-Wolf, nonlinear shrinkage, Tyler M-estimator, Student-t)
- **`mu.py`**: Expected return estimators (Huber mean, Student-t, Bayesian shrinkage, confidence intervals)
- **`factors.py`**: Factor model utilities (time-series/cross-sectional regression, PCA, beta shrinkage)
- **`validation.py`**: Temporal validation (PurgedKFold, purging/embargo logic)

**Critical Design Pattern:** All estimators accept pandas DataFrames/Series and return the same, preserving asset labels throughout the pipeline. This ensures end-to-end traceability.

**`src/itau_quant/optimization/`** - Portfolio optimization
- **`core/mv_qp.py`**: Mean-variance QP with turnover penalties and transaction costs
- **`core/cvar_lp.py`**: CVaR-based optimization (LP/SOCP formulations)
- **`core/risk_parity.py`**: Risk parity allocation
- **`heuristics/cardinality.py`**: Heuristics for cardinality constraints (K_min ≤ |w|_0 ≤ K_max)
- **`ga/`**: Genetic algorithm for non-convex extensions

**Design Constraint:** The optimizer incorporates costs and turnover **inside** the objective function, not as post-processing. This is critical for realistic performance.

**`src/itau_quant/backtesting/`** - Walk-forward validation
- **`engine.py`**: Main backtest orchestrator
- **`walk_forward.py`**: Temporal splitting with purging/embargo
- **`execution.py`**: Order execution simulation with slippage
- **`metrics.py`**: Performance metrics (Sharpe HAC, Sortino, CVaR, drawdown, turnover)

**Anti-Pattern to Avoid:** Never use future data in training (look-ahead bias). The `validation.py` module enforces temporal ordering with purging (remove training samples too close to test start) and embargo (remove training samples after test end).

**`src/itau_quant/data/`** - Data management
- **`sources/`**: Connectors for yfinance, FRED, CSV, crypto APIs
- **`processing/`**: Return calculation, corporate actions, calendar handling
- **`storage.py`**: Parquet-based persistence with versioning
- **`universe.py`**: Asset universe definitions and filtering

**Data Flow:** Raw prices (CSV/API) → Adjusted close → Log returns → Winsorized/cleaned → Parquet storage → Estimators

## Critical Implementation Details

### Black-Litterman Implementation (estimators/bl.py)

The BL framework chains these functions:
1. **`reverse_optimization(weights, cov, risk_aversion)`** → Derives implied equilibrium returns π from market weights
2. **`build_projection_matrix(views, assets)`** → Constructs P (view matrix) and Q (view returns)
3. **`view_uncertainty(P, confidences, tau, cov)`** → Builds Ω (view uncertainty matrix) with diagonal/scalar/custom modes
4. **`posterior_returns(pi, cov, P, Q, Omega, tau)`** → Combines prior and views into posterior μ_BL, Σ_BL
5. **`black_litterman(...)`** → Orchestrator that chains above steps with PSD projection

**Numerical Stability:** All linear solves use `_solve_psd()` helper with Cholesky decomposition + jitter (default 1e-10) to handle near-singular matrices.

**tau Parameter:** Scales prior uncertainty. When tau → 0, prior dominates. When tau → ∞, views dominate. Default: 0.025.

### Optimization Objective Function

Mean-variance with explicit cost modeling:
```
max_w  μᵀw - λ wᵀΣw - η ‖w - w_{t-1}‖₁ - cᵀ|w - w_{t-1}|
```

Where:
- `μᵀw`: Expected return
- `λ wᵀΣw`: Risk penalty (λ calibrated to target volatility 10-12%)
- `η ‖w - w_{t-1}‖₁`: Turnover penalty (keeps monthly turnover 5-20%)
- `cᵀ|w - w_{t-1}|`: Linear transaction costs (30 bps per round-trip)

**Constraint Blocks:**
1. Budget: `1ᵀw = 1`, box: `0 ≤ w_i ≤ u_i`
2. Group limits (e.g., crypto ≤ 5%, US equity ≤ 35%)
3. Turnover cap: `‖w - w_{t-1}‖₁ ≤ τ` (default τ = 0.20)
4. Cardinality: `K_min ≤ Σ_i z_i ≤ K_max` with `w_i ≤ U_i z_i`, `z_i ∈ {0,1}` (20 ≤ K ≤ 35)
5. FX exposure: `|Σ_i FX_i · w_i| ≤ 0.30` (net USD exposure vs BRL)

### Validation Framework (estimators/validation.py)

**PurgedKFold Splitter:**
- Generates temporal train/test splits
- **Purging:** Removes training samples within `purge_window` observations before test start (avoids label leakage from overlapping returns)
- **Embargo:** Removes training samples within `embargo_pct` observations after test end (accounts for serial correlation)

**Usage Pattern:**
```python
from itau_quant.estimators.validation import PurgedKFold

splitter = PurgedKFold(
    n_splits=5,
    min_train=252,  # 1 year
    min_test=21,    # 1 month
    purge_window=2, # Remove 2 days before test
    embargo_pct=0.05 # Remove 5% of total obs after test
)

for train_idx, test_idx in splitter.split(returns_df):
    # train_idx and test_idx are always non-overlapping
    # train never contains data from [test_start - purge : test_end + embargo]
```

### Estimator Conventions

**All estimators follow this contract:**
- **Input:** pandas DataFrame (rows = observations, columns = assets) or Series
- **Output:** pandas DataFrame/Series with preserved asset labels
- **NaN handling:** Dropped with warnings, never silently filled
- **Numerical stability:** PSD projection with configurable epsilon (default 1e-6 to 1e-9)

**Parameter Naming:**
- `ddof`: Degrees of freedom for variance (0 = population, 1 = sample). **Use ddof=1 for unbiased estimates.**
- `epsilon`: Floor for eigenvalues in PSD projection
- `window`: Rolling window size (in observations, not calendar days)
- `tau`: Scaling factor for prior uncertainty (Black-Litterman, shrinkage)

## Configuration System

The project uses a **Pydantic + YAML** configuration system for type-safe, validated configurations. All configuration parameters previously hardcoded in scripts have been extracted to YAML files in `configs/`.

### Configuration Files

**Universe Configurations:**
- `configs/universe_arara.yaml` - Basic ARARA universe (27 tickers)
- `configs/universe_arara_robust.yaml` - Robust universe with spot crypto ETFs (30 tickers)

**Portfolio Configurations:**
- `configs/portfolio_arara_basic.yaml` - Standard mean-variance settings
- `configs/portfolio_arara_robust.yaml` - Conservative settings with Huber estimator

**Production Configurations:**
- `configs/production_erc_v2.yaml` - ERC v2 with calibrated volatility/turnover targets
- `configs/asset_groups.yaml` - Group constraints across asset classes

### Configuration Schemas

All configs are validated using Pydantic models in `src/itau_quant/config/schemas.py`:

- **`UniverseConfig`**: Asset universe (name, tickers, description)
- **`PortfolioConfig`**: Portfolio optimization parameters (risk_aversion, position limits, turnover)
- **`ProductionConfig`**: Production system settings (vol_target, cardinality, group constraints)
- **`EstimatorConfig`**: Estimator parameters (window, mu_method, sigma_method, huber_delta)
- **`DataConfig`**: Data loading parameters (start_date, lookback_years, min_history_days)
- **`AssetGroupConstraints`**: Group-level constraints (assets list, max, per_asset_max)

### Loading Configurations

```python
from itau_quant.config import load_config, UniverseConfig, PortfolioConfig

# Load and validate configs
universe = load_config("configs/universe_arara.yaml", UniverseConfig)
portfolio = load_config("configs/portfolio_arara_basic.yaml", PortfolioConfig)

# Access validated fields
tickers = universe.tickers  # List[str], uppercase, validated
risk_aversion = portfolio.risk_aversion  # float, ge=0
max_position = portfolio.max_position  # float, 0-1 range validated
```

### Example YAML Config

**`configs/portfolio_arara_basic.yaml`:**
```yaml
risk_aversion: 3.0
max_position: 0.15  # 15% per asset
min_position: 0.0   # long-only
turnover_penalty: 0.10
estimation_window: 252  # 1 year
shrinkage_method: ledoit_wolf

estimators:
  window_days: 252
  mu_method: simple
  sigma_method: ledoit_wolf

data:
  lookback_years: 3
  min_history_days: 302
```

### Using Configs in Scripts

All example and production scripts now accept `--universe` and `--portfolio` (or `--config`) arguments:

```bash
# Use default configs
poetry run python scripts/examples/run_portfolio_arara.py

# Use custom configs
poetry run python scripts/examples/run_portfolio_arara.py \
    --universe configs/universe_arara_robust.yaml \
    --portfolio configs/portfolio_arara_robust.yaml
```

### Configuration Best Practices

1. **Never hardcode parameters** - Extract to YAML configs
2. **Use Pydantic validation** - Leverage field validators for constraints
3. **Provide sensible defaults** - Use `Field(default=...)` for optional parameters
4. **Document in YAML comments** - Add inline comments explaining parameter choices
5. **Version configs** - Commit config files to git for reproducibility

## Testing Philosophy

**Test Structure:** Mirrors `src/` hierarchy (e.g., `tests/estimators/test_bl.py` tests `src/itau_quant/estimators/bl.py`)

**Test Categories:**
1. **Unit tests:** Pure function logic with synthetic data
2. **Integration tests:** Multi-module interactions
3. **Numerical tests:** Verify mathematical identities (e.g., BL posterior equations)
4. **Edge case tests:** Empty inputs, singular matrices, convergence failures

**Fixtures Pattern:**
```python
@pytest.fixture
def market_data() -> dict:
    """Standard 3-asset test scenario."""
    assets = ["A", "B", "C"]
    cov = pd.DataFrame(np.diag([0.04, 0.09, 0.16]), index=assets, columns=assets)
    weights = pd.Series([1/3, 1/3, 1/3], index=assets)
    pi, delta = reverse_optimization(weights, cov, risk_aversion=3.0)
    return {"assets": assets, "cov": cov, "weights": weights, "pi": pi, "delta": delta}
```

**Assertion Helpers:**
- `assert_series_equal()` / `assert_frame_equal()` from pandas.testing
- `assert_allclose()` from numpy.testing for numerical comparisons (use `atol` and `rtol`)

## Common Development Patterns

### Adding a New Estimator

1. Place in appropriate `estimators/` file (mu.py, cov.py, factors.py, etc.)
2. Follow signature pattern:
   ```python
   def my_estimator(
       data: Union[pd.DataFrame, pd.Series, np.ndarray],
       *,
       param1: float = default,
       **kwargs
   ) -> pd.DataFrame:
       """Docstring with parameters, returns, and examples."""
       clean = _ensure_dataframe(data, min_obs=2)
       # ... implementation ...
       return result_df
   ```
3. Add to `__all__` export list
4. Create tests in `tests/estimators/test_*.py` with at least:
   - Synthetic data test (known ground truth)
   - Edge case test (empty, singular, NaN)
   - Numerical stability test

### Adding a New Optimizer

1. Create solver in `optimization/core/`
2. Implement interface:
   ```python
   def solve_portfolio(
       mu: pd.Series,
       cov: pd.DataFrame,
       constraints: dict,
       costs: Optional[dict] = None,
       w_prev: Optional[pd.Series] = None,
       **solver_opts
   ) -> dict:
       """Returns dict with keys: weights, obj_value, status, solver_time"""
   ```
3. Register in `optimization/solvers.py` dispatcher
4. Add constraint builders in `optimization/core/constraints_builder.py`

### Extending Backtesting

1. New metrics go in `backtesting/metrics.py` with signature:
   ```python
   def my_metric(returns: pd.Series, **kwargs) -> float:
       """Docstring."""
       # Return single scalar
   ```
2. Risk monitors go in `backtesting/risk_monitor.py`:
   ```python
   def check_trigger(portfolio_state: dict, limits: dict) -> bool:
       """Returns True if rebalance triggered."""
   ```
3. Update `backtesting/engine.py` to call new components

## Project Conventions

### Code Style
- **Line length:** 88 characters (Black default)
- **Imports:** Sorted with ruff (stdlib, third-party, local)
- **Type hints:** Encouraged but not enforced everywhere (legacy code exists)
- **Docstrings:** NumPy style for public functions

### Variable Naming
- `mu`: Expected returns (vector or Series)
- `cov` / `Sigma`: Covariance matrix (DataFrame or ndarray)
- `w` / `weights`: Portfolio weights (Series)
- `pi`: Equilibrium returns (Black-Litterman prior)
- `P`, `Q`: View matrix and view returns (Black-Litterman)
- `Omega`: View uncertainty matrix (Black-Litterman)
- `tau`: Prior scaling factor
- `lambda_`: Risk aversion (underscore to avoid Python keyword)

### File Organization
- `__init__.py` files should import public API symbols
- Private helpers start with `_` (e.g., `_ensure_dataframe`, `_solve_psd`)
- Constants go in `config/constants.py` (e.g., `TRADING_DAYS_PER_YEAR = 252`)

## Performance Targets (Out-of-Sample)

The following targets drive design decisions:

| Metric              | Target       | Constraint                               |
|---------------------|--------------|------------------------------------------|
| Sharpe Ratio (HAC)  | ≥ 0.80       | Must beat best baseline - 0.10           |
| Max Drawdown        | ≤ 15%        | Triggers defensive mode at 15%           |
| CVaR (5%)           | ≤ 8% annual  | Tail risk control (√252 × CVaR daily)    |
| Volatility          | ≤ 12% annual | Annualized rolling std                   |
| Monthly Turnover    | 5% - 20%     | Controlled via transaction costs (c = 30 bps) in objective |
| Annual Costs        | ≤ 50 bps     | Linear + slippage model                  |
| Tracking Error      | Monitor      | vs MSCI ACWI (60%) + AGG (40%), unhedged |

**Baseline Strategies (Must Outperform):**
- Equal-weight (1/N)
- Minimum variance (with Ledoit-Wolf shrinkage)
- Risk parity

## Known Issues and Workarounds

### Issue: Singular Covariance Matrices
**When:** High-dimensional regimes (N assets close to T observations) or highly correlated assets
**Fix:** Use `ledoit_wolf_shrinkage()` or `nonlinear_shrinkage()` instead of `sample_cov()`

### Issue: CVXPY Solver Failures
**When:** Tight cardinality constraints + small turnover caps
**Workaround:** Relax turnover cap first, then tighten. Use ECOS for QP, Clarabel for SOCP.

### Issue: Look-Ahead Bias in Backtests
**Prevention:** Always use `PurgedKFold` with `purge_window ≥ 2` and `embargo_pct ≥ 0.05`

## Roadmap and Current Status

**Completed:**
- ✅ Data loader with Parquet persistence
- ✅ Robust estimators (μ, Σ) with extensive tests
- ✅ Black-Litterman complete implementation
- ✅ Factor model utilities (time-series/cross-sectional regression, PCA)
- ✅ Temporal validation (PurgedKFold)

**In Progress:**
- 🚧 Optimization core (MV-QP with costs/turnover)
- 🚧 Cardinality heuristics
- 🚧 Backtesting engine

**Planned:**
- 📋 CVaR-based optimization (mean-CVaR LP/SOCP)
- 📋 Walk-forward validation pipeline
- 📋 Performance reporting (10-page PDF)
- 📋 GenAI section for final report (15% of grade)

## Important Files to Review

**For understanding the system:**
1. `README.md` - Project overview and universe description
2. `PRD.md` - Detailed product requirements and formulation
3. `src/itau_quant/estimators/bl.py` - Black-Litterman reference implementation
4. `configs/optimizer_example.yaml` - Complete config example (when created)

**For contributing:**
1. `tests/estimators/` - Test patterns and fixtures
2. `src/itau_quant/estimators/README.md` - Estimator module conventions
3. `pyproject.toml` - Dependencies and tool configs

## Debugging Tips

**Enable debug logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Inspect optimizer internals:**
```python
result = optimize_portfolio(..., solver_opts={"verbose": True})
print(f"Status: {result['status']}")
print(f"Objective: {result['obj_value']}")
```

**Check matrix conditioning:**
```python
import numpy as np
cond_number = np.linalg.cond(cov.to_numpy())
print(f"Condition number: {cond_number:.2e}")  # Should be < 1e12
```

**Validate temporal splits:**
```python
for i, (train_idx, test_idx) in enumerate(splitter.split(data)):
    assert train_idx.max() < test_idx.min(), f"Fold {i}: Future leakage!"
```
