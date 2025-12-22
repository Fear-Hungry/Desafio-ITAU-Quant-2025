"""Pipeline step 2: Parameter estimation (μ, Σ).

This module extracts the parameter estimation logic from run_02_estimate_params.py
into a reusable function. It computes expected returns (μ) and covariance matrix (Σ)
using robust estimators.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Callable, Mapping

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

__all__ = ["EstimationRequest", "estimate_parameters"]

logger = get_logger(__name__)


@dataclass(frozen=True)
class _EstimationOptions:
    mu_method: str | None = None
    cov_method: str | None = None
    huber_delta: float | None = None
    shrink_strength: float | None = None
    student_t_nu: float | None = None
    annualize: bool | None = None
    window: int | None = None


@dataclass(frozen=True)
class _ResolvedEstimationOptions:
    mu_method: str
    cov_method: str
    huber_delta: float
    shrink_strength: float
    student_t_nu: float
    annualize: bool
    window: int


@dataclass(frozen=True)
class EstimationRequest:
    """Input/output configuration for estimate_parameters."""

    returns_file: str = "returns_arara.parquet"
    window: int | None = None
    mu_method: str | None = None
    cov_method: str | None = None
    huber_delta: float | None = None
    shrink_strength: float | None = None
    student_t_nu: float | None = None
    annualize: bool | None = None
    mu_output: str = "mu_estimate.parquet"
    cov_output: str = "cov_estimate.parquet"
    config_path: str | Path | None = None


_ESTIMATION_REQUEST_FIELDS = frozenset(
    field.name for field in fields(EstimationRequest)
)


@dataclass(frozen=True)
class _EstimationSavePayload:
    mu: pd.Series
    cov: pd.DataFrame
    mu_output: str
    cov_output: str


@dataclass(frozen=True)
class _EstimationResultPayload:
    mu_path: Path
    cov_path: Path
    shrinkage: float | None
    window_used: int


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


def _coerce_estimation_request(
    request: EstimationRequest | None,
    overrides: Mapping[str, Any],
) -> EstimationRequest:
    if request is None:
        request = EstimationRequest()
    if not overrides:
        return request

    unknown = set(overrides) - _ESTIMATION_REQUEST_FIELDS
    if unknown:
        unknown_list = ", ".join(sorted(unknown))
        raise TypeError(f"Unknown estimate_parameters arguments: {unknown_list}")
    return replace(request, **overrides)


def _save_estimates(
    settings: Settings,
    payload: _EstimationSavePayload,
) -> tuple[Path, Path]:
    mu_path = Path(settings.processed_data_dir) / payload.mu_output
    cov_path = Path(settings.processed_data_dir) / payload.cov_output
    _save_series(payload.mu, mu_path, column="mu")
    _save_frame(payload.cov, cov_path)
    logger.info("Saved μ to %s", mu_path)
    logger.info("Saved Σ to %s", cov_path)
    return mu_path, cov_path


def _resolve_config_candidate(config_path: str | Path, settings: Settings) -> Path:
    candidate = Path(config_path)
    if candidate.is_absolute():
        return candidate

    in_configs = settings.configs_dir / candidate
    return in_configs if in_configs.exists() else settings.project_root / candidate


def _load_estimators_sections(config_file: Path) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    raw = load_yaml_text(config_file.read_text(encoding="utf-8"))
    estimators_section = raw.get("estimators", {})
    if not isinstance(estimators_section, Mapping):
        return {}, {}

    mu_section = estimators_section.get("mu", {})
    sigma_section = estimators_section.get("sigma", {})
    if not isinstance(mu_section, Mapping):
        mu_section = {}
    if not isinstance(sigma_section, Mapping):
        sigma_section = {}
    return mu_section, sigma_section


def _normalize_cov_method(method: str | None) -> str:
    return (method or "").strip().lower().replace("-", "_")


def _maybe_override_ledoit_wolf_with_nonlinear(cov_method: str | None) -> str | None:
    if cov_method is None:
        return "nonlinear"
    return "nonlinear" if _normalize_cov_method(cov_method) == "ledoit_wolf" else cov_method


def _override_str_if_none(
    value: str | None,
    section: Mapping[str, Any],
    *,
    key: str,
    default: str,
) -> str | None:
    if value is not None:
        return value
    return str(section.get(key, default))


def _override_float_if_none(
    value: float | None,
    section: Mapping[str, Any],
    *,
    key: str,
) -> float | None:
    if value is not None:
        return value
    if key not in section:
        return value
    return float(section[key])


def _apply_sigma_nonlinear_override(
    cov_method: str | None,
    sigma_section: Mapping[str, Any],
) -> str | None:
    sigma_method = str(
        sigma_section.get("method", DEFAULT_OPTIMIZER_YAML.estimators_sigma.method)
    )
    if _normalize_cov_method(sigma_method) != "ledoit_wolf":
        return cov_method
    if not bool(sigma_section.get("nonlinear", False)):
        return cov_method
    return _maybe_override_ledoit_wolf_with_nonlinear(cov_method)


def _apply_window_override(
    window: int | None,
    mu_section: Mapping[str, Any],
    sigma_section: Mapping[str, Any],
) -> int | None:
    if window is not None:
        return window

    mu_window = int(mu_section.get("window_days", 0) or 0)
    sigma_window = int(sigma_section.get("window_days", 0) or 0)
    if (mu_window, sigma_window) == (0, 0):
        return window
    return max(mu_window, sigma_window, TRADING_DAYS_IN_YEAR)


def _apply_estimator_config_overrides(
    *,
    config_path: str | Path | None,
    settings: Settings,
    options: _EstimationOptions,
) -> _EstimationOptions:
    if config_path is None:
        return options

    candidate = _resolve_config_candidate(config_path, settings)
    if not candidate.exists():
        logger.warning("Estimator config not found at %s (skipping)", candidate)
        return options

    mu_section, sigma_section = _load_estimators_sections(candidate)

    mu_method = _override_str_if_none(
        options.mu_method,
        mu_section,
        key="method",
        default=DEFAULT_OPTIMIZER_YAML.estimators_mu.method,
    )
    cov_method = _override_str_if_none(
        options.cov_method,
        sigma_section,
        key="method",
        default=DEFAULT_OPTIMIZER_YAML.estimators_sigma.method,
    )
    huber_delta = _override_float_if_none(options.huber_delta, mu_section, key="delta")
    shrink_strength = _override_float_if_none(
        options.shrink_strength, mu_section, key="strength"
    )
    student_t_nu = _override_float_if_none(options.student_t_nu, mu_section, key="nu")
    cov_method = _apply_sigma_nonlinear_override(cov_method, sigma_section)
    window = _apply_window_override(options.window, mu_section, sigma_section)

    return _EstimationOptions(
        mu_method=mu_method,
        cov_method=cov_method,
        huber_delta=huber_delta,
        shrink_strength=shrink_strength,
        student_t_nu=student_t_nu,
        annualize=options.annualize,
        window=window,
    )


def _apply_estimator_defaults(options: _EstimationOptions) -> _ResolvedEstimationOptions:
    mu_method = (
        options.mu_method
        if options.mu_method is not None
        else DEFAULT_OPTIMIZER_YAML.estimators_mu.method
    )
    cov_method = (
        options.cov_method
        if options.cov_method is not None
        else DEFAULT_OPTIMIZER_YAML.estimators_sigma.method
    )
    huber_delta = (
        options.huber_delta
        if options.huber_delta is not None
        else DEFAULT_OPTIMIZER_YAML.estimators_mu.huber_delta
    )
    shrink_strength = (
        options.shrink_strength
        if options.shrink_strength is not None
        else DEFAULT_OPTIMIZER_YAML.estimators_mu.shrink_strength
    )
    student_t_nu = (
        options.student_t_nu
        if options.student_t_nu is not None
        else DEFAULT_OPTIMIZER_YAML.estimators_mu.student_t_nu
    )
    annualize = True if options.annualize is None else options.annualize
    window = TRADING_DAYS_IN_YEAR if options.window is None else options.window

    return _ResolvedEstimationOptions(
        mu_method=str(mu_method),
        cov_method=str(cov_method),
        huber_delta=float(huber_delta),
        shrink_strength=float(shrink_strength),
        student_t_nu=float(student_t_nu),
        annualize=bool(annualize),
        window=int(window),
    )


def _resolve_estimation_options(
    raw_options: _EstimationOptions,
    config_path: str | Path | None,
    settings: Settings,
) -> _ResolvedEstimationOptions:
    options = _apply_estimator_config_overrides(
        config_path=config_path,
        settings=settings,
        options=raw_options,
    )
    return _apply_estimator_defaults(options)


def _load_returns_frame(returns_path: Path) -> pd.DataFrame:
    loaded = read_dataframe(returns_path)
    frame = loaded.to_frame() if isinstance(loaded, pd.Series) else loaded
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index)
    return frame.sort_index()


def _load_returns_sample(
    settings: Settings,
    returns_file: str,
    window: int,
) -> tuple[pd.DataFrame, int]:
    returns_path = Path(settings.processed_data_dir) / returns_file
    if not returns_path.exists():
        raise FileNotFoundError(f"Returns file not found: {returns_path}")

    logger.info("Loading returns from %s", returns_path)
    returns = _load_returns_frame(returns_path)
    if returns.empty:
        raise ValueError(f"Returns file {returns_path} is empty.")

    window_used = min(int(window), len(returns))
    sample = returns.tail(window_used)
    return sample, int(window_used)


def _mu_estimator_huber(
    sample: pd.DataFrame,
    options: _ResolvedEstimationOptions,
) -> pd.Series:
    mu_daily, _ = huber_mean(sample, c=options.huber_delta)
    return mu_daily


def _mu_estimator_simple(
    sample: pd.DataFrame,
    _options: _ResolvedEstimationOptions,
) -> pd.Series:
    return mean_return(sample, method="simple")


def _mu_estimator_shrunk(
    sample: pd.DataFrame,
    options: _ResolvedEstimationOptions,
) -> pd.Series:
    return shrunk_mean(sample, strength=options.shrink_strength, prior=0.0)


def _mu_estimator_student_t(
    sample: pd.DataFrame,
    options: _ResolvedEstimationOptions,
) -> pd.Series:
    return student_t_mean(sample, nu=float(options.student_t_nu))


_MU_METHOD_ALIASES: dict[str, str] = {
    "huber": "huber",
    "simple": "simple",
    "mean": "simple",
    "shrunk": "shrunk",
    "shrunk_50": "shrunk",
    "shrinkage": "shrunk",
    "student_t": "student_t",
    "student-t": "student_t",
}

_MuEstimator = Callable[[pd.DataFrame, _ResolvedEstimationOptions], pd.Series]
_MU_METHOD_DISPATCH: dict[str, _MuEstimator] = {
    "huber": _mu_estimator_huber,
    "simple": _mu_estimator_simple,
    "shrunk": _mu_estimator_shrunk,
    "student_t": _mu_estimator_student_t,
}


def _estimate_mu_daily(
    sample: pd.DataFrame,
    *,
    options: _ResolvedEstimationOptions,
    window_used: int,
) -> pd.Series:
    mu_key = (options.mu_method or "").lower()
    if not 0.0 <= options.shrink_strength <= 1.0:
        raise ValueError("shrink_strength must lie in [0, 1].")

    logger.info("Estimating μ via %s (window=%d)", mu_key, window_used)

    canonical = _MU_METHOD_ALIASES.get(mu_key)
    if canonical is None:
        raise ValueError(f"Unsupported mu_method: {options.mu_method}")

    estimator = _MU_METHOD_DISPATCH[canonical]
    return estimator(sample, options)


def _cov_estimator_ledoit_wolf(sample: pd.DataFrame) -> tuple[pd.DataFrame, float | None]:
    cov_daily, shrinkage = ledoit_wolf_shrinkage(sample)
    return cov_daily, float(shrinkage)


def _cov_estimator_nonlinear(sample: pd.DataFrame) -> tuple[pd.DataFrame, float | None]:
    return nonlinear_shrinkage(sample), None


def _cov_estimator_oas(sample: pd.DataFrame) -> tuple[pd.DataFrame, float | None]:
    cov_daily, shrinkage = oas_shrinkage(sample)
    return cov_daily, float(shrinkage)


def _cov_estimator_sample(sample: pd.DataFrame) -> tuple[pd.DataFrame, float | None]:
    return sample_cov(sample), None


def _cov_estimator_mincovdet(sample: pd.DataFrame) -> tuple[pd.DataFrame, float | None]:
    return min_cov_det(sample), None


_COV_METHOD_ALIASES: dict[str, str] = {
    "ledoit_wolf": "ledoit_wolf",
    "nonlinear": "nonlinear",
    "oas": "oas",
    "sample": "sample",
    "empirical": "sample",
    "mincovdet": "mincovdet",
    "min_cov_det": "mincovdet",
    "mcd": "mincovdet",
}

_CovEstimator = Callable[[pd.DataFrame], tuple[pd.DataFrame, float | None]]
_COV_METHOD_DISPATCH: dict[str, _CovEstimator] = {
    "ledoit_wolf": _cov_estimator_ledoit_wolf,
    "nonlinear": _cov_estimator_nonlinear,
    "oas": _cov_estimator_oas,
    "sample": _cov_estimator_sample,
    "mincovdet": _cov_estimator_mincovdet,
}


def _estimate_cov_daily(
    sample: pd.DataFrame,
    *,
    cov_method: str,
) -> tuple[pd.DataFrame, float | None]:
    cov_key = _normalize_cov_method(cov_method)
    logger.info("Estimating Σ via %s", cov_key or cov_method)

    canonical = _COV_METHOD_ALIASES.get(cov_key)
    if canonical is None:
        raise ValueError(f"Unsupported cov_method: {cov_method}")

    estimator = _COV_METHOD_DISPATCH[canonical]
    return estimator(sample)


def _annualize_estimates(
    mu_daily: pd.Series,
    cov_daily: pd.DataFrame,
    *,
    annualize: bool,
) -> tuple[pd.Series, pd.DataFrame]:
    if not annualize:
        return mu_daily, cov_daily

    factor = float(TRADING_DAYS_IN_YEAR)
    return mu_daily * factor, cov_daily * factor


def _estimate_mu_cov(
    sample: pd.DataFrame,
    options: _ResolvedEstimationOptions,
    window_used: int,
) -> tuple[pd.Series, pd.DataFrame, float | None]:
    mu_daily = _estimate_mu_daily(sample, options=options, window_used=window_used)
    cov_daily, shrinkage = _estimate_cov_daily(sample, cov_method=options.cov_method)
    mu, cov = _annualize_estimates(mu_daily, cov_daily, annualize=options.annualize)
    return mu, cov, shrinkage


def _build_estimation_result(
    payload: _EstimationResultPayload,
    options: _ResolvedEstimationOptions,
) -> dict[str, Any]:
    return {
        "status": "completed",
        "mu_output": str(payload.mu_path),
        "cov_output": str(payload.cov_path),
        "shrinkage": payload.shrinkage,
        "window_used": int(payload.window_used),
        "annualized": options.annualize,
        "mu_method": options.mu_method,
        "cov_method": options.cov_method,
    }


def estimate_parameters(
    request: EstimationRequest | None = None,
    *,
    settings: Settings | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Estimate expected returns (μ) and covariance (Σ) from historical data.

    This function applies robust estimators to historical returns:
    - μ (expected return): Huber mean (robust to outliers) or simple mean
    - Σ (covariance): Ledoit-Wolf/OAS shrinkage, MinCovDet, or sample covariance

    Args:
        request: Optional EstimationRequest containing input/output configuration.
            If omitted, any EstimationRequest field can be passed as a keyword argument.
            Keyword overrides take precedence when both are provided.
        settings: Settings object (uses default if None).
        **overrides: Backwards-compatible keyword overrides for request fields.

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
    request = _coerce_estimation_request(request, overrides)

    raw_options = _EstimationOptions(
        mu_method=request.mu_method,
        cov_method=request.cov_method,
        huber_delta=request.huber_delta,
        shrink_strength=request.shrink_strength,
        student_t_nu=request.student_t_nu,
        annualize=request.annualize,
        window=request.window,
    )
    resolved = _resolve_estimation_options(raw_options, request.config_path, settings)

    sample, window_used = _load_returns_sample(
        settings, request.returns_file, resolved.window
    )
    mu, cov, shrinkage = _estimate_mu_cov(sample, resolved, window_used)
    save_payload = _EstimationSavePayload(
        mu=mu,
        cov=cov,
        mu_output=request.mu_output,
        cov_output=request.cov_output,
    )
    mu_path, cov_path = _save_estimates(settings, save_payload)
    result_payload = _EstimationResultPayload(
        mu_path=mu_path,
        cov_path=cov_path,
        shrinkage=shrinkage,
        window_used=window_used,
    )
    return _build_estimation_result(result_payload, resolved)
