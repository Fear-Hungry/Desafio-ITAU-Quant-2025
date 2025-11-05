"""Pipeline step 2: Parameter estimation (μ, Σ).

This module extracts the parameter estimation logic from run_02_estimate_params.py
into a reusable function. It computes expected returns (μ) and covariance matrix (Σ)
using robust estimators.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from arara_quant.config import Settings
from arara_quant.estimators.cov import ledoit_wolf_shrinkage, sample_cov
from arara_quant.estimators.mu import huber_mean, mean_return, shrunk_mean
from arara_quant.utils.logging_config import get_logger

__all__ = ["estimate_parameters"]

logger = get_logger(__name__)


def estimate_parameters(
    *,
    returns_file: str = "returns_arara.parquet",
    window: int = 252,
    mu_method: str = "shrunk_50",
    cov_method: str = "ledoit_wolf",
    huber_delta: float = 1.5,
    shrink_strength: float = 0.5,
    annualize: bool = True,
    mu_output: str = "mu_estimate.parquet",
    cov_output: str = "cov_estimate.parquet",
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Estimate expected returns (μ) and covariance (Σ) from historical data.

    This function applies robust estimators to historical returns:
    - μ (expected return): Huber mean (robust to outliers) or simple mean
    - Σ (covariance): Ledoit-Wolf shrinkage or sample covariance

    Args:
        returns_file: Input parquet file with historical returns
        window: Number of most recent observations to use (rolling window)
        mu_method: Method for expected return ("shrunk_50", "huber", or "simple")
        cov_method: Method for covariance ("ledoit_wolf" or "sample")
        huber_delta: Robustness parameter for Huber estimator (typically 1.5)
        shrink_strength: Shrinkage intensity towards the prior (default 0.5)
        annualize: If True, annualize estimates (252 trading days)
        mu_output: Output filename for expected returns
        cov_output: Output filename for covariance matrix
        settings: Settings object (uses default if None)

    Returns:
        dict containing:
            - status: "completed"
            - mu_output: Absolute path to μ Parquet file
            - cov_output: Absolute path to Σ Parquet file
            - shrinkage: Shrinkage intensity (if Ledoit-Wolf, else None)
            - window_used: Actual window size used (min of window and data length)
            - annualized: Whether estimates were annualized

    Raises:
        FileNotFoundError: If returns file doesn't exist
        ValueError: If data is empty or invalid

    Examples:
        >>> result = estimate_parameters(window=252, annualize=True)
        >>> print(f"Shrinkage: {result['shrinkage']:.3f}")
    """
    settings = settings or Settings.from_env()

    returns_path = Path(settings.processed_data_dir) / returns_file
    if not returns_path.exists():
        raise FileNotFoundError(f"Returns file not found: {returns_path}")

    logger.info("Loading returns from %s", returns_path)
    returns = pd.read_parquet(returns_path)

    if returns.empty:
        raise ValueError(f"Returns file {returns_path} is empty.")

    # Use the most recent window observations
    window_used = min(window, len(returns))
    sample = returns.tail(window_used)

    # Estimate expected returns (μ)
    mu_key = (mu_method or "").lower()
    if not 0.0 <= shrink_strength <= 1.0:
        raise ValueError("shrink_strength must lie in [0, 1].")

    logger.info("Estimating μ via %s (window=%d)", mu_key, window_used)

    if mu_key == "huber":
        mu_daily, _ = huber_mean(sample, c=huber_delta)
    elif mu_key in {"simple", "mean"}:
        mu_daily = mean_return(sample, method="simple")
    elif mu_key in {"shrunk", "shrunk_50", "shrinkage"}:
        mu_daily = shrunk_mean(sample, strength=shrink_strength, prior=0.0)
    else:
        raise ValueError(f"Unsupported mu_method: {mu_method}")

    logger.info("Estimating Σ via %s", cov_method)

    # Estimate covariance (Σ)
    shrinkage = None
    if cov_method == "ledoit_wolf":
        cov_daily, shrinkage = ledoit_wolf_shrinkage(sample)
        shrinkage = float(shrinkage)
    elif cov_method == "sample":
        cov_daily = sample_cov(sample)
    else:
        raise ValueError(f"Unsupported cov_method: {cov_method}")

    # Annualize if requested
    if annualize:
        mu = mu_daily * 252.0
        cov = cov_daily * 252.0
    else:
        mu = mu_daily
        cov = cov_daily

    # Save outputs
    mu_path = Path(settings.processed_data_dir) / mu_output
    cov_path = Path(settings.processed_data_dir) / cov_output

    mu_path.parent.mkdir(parents=True, exist_ok=True)
    cov_path.parent.mkdir(parents=True, exist_ok=True)

    mu.to_frame("mu").to_parquet(mu_path)
    cov.to_parquet(cov_path)

    logger.info("Saved μ to %s", mu_path)
    logger.info("Saved Σ to %s", cov_path)

    return {
        "status": "completed",
        "mu_output": str(mu_path),
        "cov_output": str(cov_path),
        "shrinkage": shrinkage,
        "window_used": int(window_used),
        "annualized": bool(annualize),
        "mu_method": mu_method,
        "cov_method": cov_method,
    }
