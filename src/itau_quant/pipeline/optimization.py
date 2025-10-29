"""Pipeline step 3: Portfolio optimization.

This module extracts the portfolio optimization logic from run_03_optimize.py
into a reusable function. It solves the mean-variance optimization problem
to determine optimal portfolio weights.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from itau_quant.config import Settings
from itau_quant.optimization.core.mv_qp import MeanVarianceConfig, solve_mean_variance
from itau_quant.utils.logging_config import get_logger
from itau_quant.utils.yaml_loader import load_yaml_text

__all__ = ["optimize_portfolio"]

logger = get_logger(__name__)


def _load_mu(path: Path) -> pd.Series:
    """Load expected returns from Parquet file.

    Handles various formats:
    - Series directly
    - DataFrame with single column
    - DataFrame with 'mu' column
    """
    frame = pd.read_parquet(path)

    if isinstance(frame, pd.Series):
        return frame.astype(float)

    if frame.shape[1] == 1:
        return frame.iloc[:, 0].astype(float)

    if "mu" in frame.columns:
        return frame["mu"].astype(float)

    raise ValueError(f"μ file {path} must contain a single column or 'mu' column")


def optimize_portfolio(
    *,
    mu_file: str = "mu_estimate.parquet",
    cov_file: str = "cov_estimate.parquet",
    risk_aversion: float = 4.0,
    max_weight: float = 0.15,
    min_weight: float = 0.0,
    turnover_cap: float | None = None,
    turnover_penalty: float = 0.0,
    ridge_penalty: float = 0.0,
    output_file: str = "optimized_weights.parquet",
    solver: str | None = None,
    config_path: str | Path | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Optimize portfolio weights using mean-variance framework.

    Solves the quadratic program:
        maximize: μᵀw - λ·wᵀΣw - η·‖w - w_prev‖₁ - ridge·‖w‖₂²
        subject to:
            - ∑w = 1 (fully invested)
            - w_min ≤ w_i ≤ w_max (bounds)
            - ‖w - w_prev‖₁ ≤ τ (optional turnover cap)

    Args:
        mu_file: Input file with expected returns (μ)
        cov_file: Input file with covariance matrix (Σ)
        risk_aversion: Risk aversion coefficient (λ), higher = more conservative
        max_weight: Maximum weight per asset (upper bound)
        min_weight: Minimum weight per asset (lower bound, typically 0)
        turnover_cap: Optional L1 turnover constraint (e.g., 0.2 = 20% max turnover)
        turnover_penalty: L1 penalty coefficient for turnover (soft constraint)
        ridge_penalty: L2 regularization on weights (prevents concentration)
        output_file: Output filename for optimized weights
        solver: CVXPy solver to use (e.g., "CLARABEL", "ECOS", "OSQP")
        settings: Settings object (uses default if None)

    Returns:
        dict containing:
            - status: "completed"
            - output: Absolute path to weights Parquet file
            - expected_return: Annualized expected return
            - volatility: Annualized volatility (std dev)
            - sharpe: Sharpe ratio (return/volatility)
            - n_assets: Number of assets with non-zero weight
            - turnover: Total turnover (L1 norm of trades)

    Raises:
        FileNotFoundError: If μ or Σ files don't exist
        ValueError: If optimization fails

    Examples:
        >>> result = optimize_portfolio(risk_aversion=5.0, max_weight=0.10)
        >>> print(f"Selected {result['n_assets']} assets")
        >>> print(f"Sharpe ratio: {result['sharpe']:.2f}")
    """
    settings = settings or Settings.from_env()

    # Load config from YAML if provided to override defaults
    if config_path is not None:
        config_file = Path(config_path)
        if config_file.exists():
            logger.info("Loading optimizer config from %s", config_file)
            raw = load_yaml_text(config_file.read_text(encoding="utf-8"))
            optimizer_section = raw.get("optimizer", {})

            # Override parameters with values from YAML
            risk_aversion = float(optimizer_section.get("lambda", risk_aversion))
            max_weight = float(optimizer_section.get("max_weight", max_weight))
            min_weight = float(optimizer_section.get("min_weight", min_weight))
            turnover_penalty = float(optimizer_section.get("eta", turnover_penalty))
            tau_from_yaml = optimizer_section.get("tau")
            if tau_from_yaml is not None:
                turnover_cap = float(tau_from_yaml)
            solver_from_yaml = optimizer_section.get("solver")
            if solver_from_yaml is not None:
                solver = str(solver_from_yaml)

            logger.info(
                "Loaded config: λ=%.1f, max_weight=%.2f, eta=%.2f, tau=%s, solver=%s",
                risk_aversion,
                max_weight,
                turnover_penalty,
                turnover_cap,
                solver,
            )

    mu_path = Path(settings.processed_data_dir) / mu_file
    cov_path = Path(settings.processed_data_dir) / cov_file

    if not mu_path.exists():
        raise FileNotFoundError(f"μ file not found: {mu_path}")
    if not cov_path.exists():
        raise FileNotFoundError(f"Σ file not found: {cov_path}")

    logger.info("Loading μ from %s", mu_path)
    logger.info("Loading Σ from %s", cov_path)

    # Load parameters
    mu = _load_mu(mu_path)
    cov = pd.read_parquet(cov_path)

    # Align covariance with mu
    cov = cov.reindex(index=mu.index, columns=mu.index)

    if cov.isnull().any().any():
        raise ValueError("Σ contains NaNs after aligning with μ")

    # Build configuration
    assets = mu.index
    lower = pd.Series(min_weight, index=assets, dtype=float)
    upper = pd.Series(max_weight, index=assets, dtype=float)
    previous = pd.Series(0.0, index=assets, dtype=float)  # No previous position

    # CRITICAL FIX: Skip turnover_cap on first allocation (when prev=0)
    # Otherwise problem becomes infeasible: ||w-0||_1 ≤ tau < 1.0 conflicts with sum(w)=1.0
    effective_turnover_cap = None if (previous == 0).all() else turnover_cap
    if effective_turnover_cap is None and turnover_cap is not None:
        logger.info("Skipping turnover_cap on initial allocation (previous_weights=0)")

    config = MeanVarianceConfig(
        risk_aversion=float(risk_aversion),
        turnover_penalty=float(turnover_penalty),
        turnover_cap=effective_turnover_cap,
        ridge_penalty=float(ridge_penalty),
        lower_bounds=lower,
        upper_bounds=upper,
        previous_weights=previous,
        cost_vector=None,
        solver=solver,
    )

    logger.info(
        "Running mean-variance optimization (λ=%.2f, max_weight=%.0f%%)",
        config.risk_aversion,
        max_weight * 100,
    )

    # Solve optimization
    result = solve_mean_variance(mu, cov, config)

    # Save weights
    output_path = Path(settings.project_root) / "results" / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    weights_sorted = result.weights.sort_values(ascending=False)
    weights_sorted.to_frame("weight").to_parquet(output_path)

    logger.info("Saved optimized weights to %s", output_path)

    # Compute metrics
    annual_return = float(result.expected_return)
    annual_vol = float(math.sqrt(result.variance))
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0
    n_assets = int((result.weights > 0.001).sum())

    return {
        "status": "completed",
        "output": str(output_path),
        "expected_return": annual_return,
        "volatility": annual_vol,
        "sharpe": sharpe,
        "n_assets": n_assets,
        "turnover": float(result.turnover),
        "risk_aversion": float(risk_aversion),
    }
