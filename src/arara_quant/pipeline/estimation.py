"""Pipeline step 2: Parameter estimation (μ, Σ).

This module extracts the parameter estimation logic from run_02_estimate_params.py
into a reusable function. It computes expected returns (μ) and covariance matrix (Σ)
using robust estimators.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from arara_quant.config import Settings
from arara_quant.config.constants import TRADING_DAYS_IN_YEAR
from arara_quant.config.params_default import DEFAULT_OPTIMIZER_YAML
from arara_quant.estimators.cov import (
    ledoit_wolf_shrinkage,
    min_cov_det,
    nonlinear_shrinkage,
    oas_shrinkage,
    sample_cov,
)
from arara_quant.utils.data_loading import read_dataframe
from arara_quant.estimators.mu import (
    huber_mean,
    mean_return,
    shrunk_mean,
    student_t_mean,
)
from arara_quant.utils.logging_config import get_logger
from arara_quant.utils.yaml_loader import load_yaml_text

__all__ = ["estimate_parameters"]

logger = get_logger(__name__)


def _save_series(series: pd.Series, path: Path, *, column: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    frame = series.to_frame(column)
    if suffix == ".csv":
        frame.to_csv(path)
        return
    frame.to_parquet(path)


def _save_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(path)
        return
    frame.to_parquet(path)


def estimate_parameters(
    *,
    returns_file: str = "returns_arara.parquet",
    window: int | None = None,
    mu_method: str | None = None,
    cov_method: str | None = None,
    huber_delta: float | None = None,
    shrink_strength: float | None = None,
    student_t_nu: float | None = None,
    annualize: bool | None = None,
    mu_output: str = "mu_estimate.parquet",
    cov_output: str = "cov_estimate.parquet",
    config_path: str | Path | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Estimate expected returns (μ) and covariance (Σ) from historical data.

    This function applies robust estimators to historical returns:
    - μ (expected return): Huber mean (robust to outliers) or simple mean
    - Σ (covariance): Ledoit-Wolf/OAS shrinkage, MinCovDet, or sample covariance

    Args:
        returns_file: Input file with historical returns (csv/parquet/pickle/feather)
        window: Number of most recent observations to use (rolling window)
        mu_method: Method for expected return ("simple", "huber", "shrunk_50", "student_t")
        cov_method: Method for covariance ("ledoit_wolf", "nonlinear", "oas", "mincovdet", "sample")
        huber_delta: Robustness parameter for Huber estimator (typically 1.5)
        shrink_strength: Shrinkage intensity towards the prior (default 0.5)
        annualize: If True, annualize estimates (252 trading days)
        mu_output: Output filename for expected returns (csv/parquet)
        cov_output: Output filename for covariance matrix (csv/parquet)
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

    if config_path is not None:
        candidate = Path(config_path)
        if not candidate.is_absolute():
            in_configs = settings.configs_dir / candidate
            candidate = (
                in_configs if in_configs.exists() else settings.project_root / candidate
            )
        if candidate.exists():
            raw = load_yaml_text(candidate.read_text(encoding="utf-8"))
            estimators_section = raw.get("estimators", {})
            if isinstance(estimators_section, Mapping):
                mu_section = (
                    estimators_section.get("mu", {})
                    if isinstance(estimators_section.get("mu", {}), Mapping)
                    else {}
                )
                sigma_section = (
                    estimators_section.get("sigma", {})
                    if isinstance(estimators_section.get("sigma", {}), Mapping)
                    else {}
                )

                if mu_method is None:
                    mu_method = str(
                        mu_section.get(
                            "method", DEFAULT_OPTIMIZER_YAML.estimators_mu.method
                        )
                    )
                if cov_method is None:
                    cov_method = str(
                        sigma_section.get(
                            "method", DEFAULT_OPTIMIZER_YAML.estimators_sigma.method
                        )
                    )

                if huber_delta is None and "delta" in mu_section:
                    huber_delta = float(mu_section["delta"])
                if shrink_strength is None and "strength" in mu_section:
                    shrink_strength = float(mu_section["strength"])
                if student_t_nu is None and "nu" in mu_section:
                    student_t_nu = float(mu_section["nu"])

                sigma_method = str(
                    sigma_section.get(
                        "method", DEFAULT_OPTIMIZER_YAML.estimators_sigma.method
                    )
                )
                sigma_key = sigma_method.strip().lower().replace("-", "_")
                if sigma_key == "ledoit_wolf" and bool(sigma_section.get("nonlinear", False)):
                    if cov_method is None:
                        cov_method = "nonlinear"
                    else:
                        cov_key = cov_method.strip().lower().replace("-", "_")
                        if cov_key == "ledoit_wolf":
                            cov_method = "nonlinear"

                mu_window = int(mu_section.get("window_days", 0) or 0)
                sigma_window = int(sigma_section.get("window_days", 0) or 0)
                if window is None and (mu_window or sigma_window):
                    window = max(mu_window, sigma_window, TRADING_DAYS_IN_YEAR)
        else:
            logger.warning("Estimator config not found at %s (skipping)", candidate)

    if mu_method is None:
        mu_method = DEFAULT_OPTIMIZER_YAML.estimators_mu.method
    if cov_method is None:
        cov_method = DEFAULT_OPTIMIZER_YAML.estimators_sigma.method
    if huber_delta is None:
        huber_delta = DEFAULT_OPTIMIZER_YAML.estimators_mu.huber_delta
    if shrink_strength is None:
        shrink_strength = DEFAULT_OPTIMIZER_YAML.estimators_mu.shrink_strength
    if student_t_nu is None:
        student_t_nu = DEFAULT_OPTIMIZER_YAML.estimators_mu.student_t_nu
    if annualize is None:
        annualize = True
    if window is None:
        window = TRADING_DAYS_IN_YEAR

    returns_path = Path(settings.processed_data_dir) / returns_file
    if not returns_path.exists():
        raise FileNotFoundError(f"Returns file not found: {returns_path}")

    logger.info("Loading returns from %s", returns_path)
    loaded = read_dataframe(returns_path)
    returns = loaded.to_frame() if isinstance(loaded, pd.Series) else loaded
    if not isinstance(returns.index, pd.DatetimeIndex):
        returns.index = pd.to_datetime(returns.index)
    returns = returns.sort_index()

    if returns.empty:
        raise ValueError(f"Returns file {returns_path} is empty.")

    # Use the most recent window observations
    window_used = min(int(window), len(returns))
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
    elif mu_key in {"student_t", "student-t"}:
        mu_daily = student_t_mean(sample, nu=float(student_t_nu))
    else:
        raise ValueError(f"Unsupported mu_method: {mu_method}")

    cov_key = (cov_method or "").strip().lower().replace("-", "_")
    logger.info("Estimating Σ via %s", cov_key or cov_method)

    # Estimate covariance (Σ)
    shrinkage: float | None = None
    if cov_key == "ledoit_wolf":
        cov_daily, shrinkage = ledoit_wolf_shrinkage(sample)
        shrinkage = float(shrinkage)
    elif cov_key == "nonlinear":
        cov_daily = nonlinear_shrinkage(sample)
    elif cov_key == "oas":
        cov_daily, shrinkage = oas_shrinkage(sample)
        shrinkage = float(shrinkage)
    elif cov_key in {"sample", "empirical"}:
        cov_daily = sample_cov(sample)
    elif cov_key in {"mincovdet", "min_cov_det", "mcd"}:
        cov_daily = min_cov_det(sample)
    else:
        raise ValueError(f"Unsupported cov_method: {cov_method}")

    # Annualize if requested
    if annualize:
        mu = mu_daily * float(TRADING_DAYS_IN_YEAR)
        cov = cov_daily * float(TRADING_DAYS_IN_YEAR)
    else:
        mu = mu_daily
        cov = cov_daily

    # Save outputs
    mu_path = Path(settings.processed_data_dir) / mu_output
    cov_path = Path(settings.processed_data_dir) / cov_output

    _save_series(mu, mu_path, column="mu")
    _save_frame(cov, cov_path)

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
