"""Expected-return estimators with robustness and shrinkage flavours."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from itau_quant.estimators import bl

ReturnsLike = Union[pd.DataFrame, pd.Series, np.ndarray, Sequence[Sequence[float]]]

__all__ = [
    "mean_return",
    "huber_mean",
    "student_t_mean",
    "bayesian_shrinkage_mean",
    "confidence_intervals",
    "blend_with_black_litterman",
    "annualize",
]


def _ensure_returns_frame(returns: ReturnsLike) -> pd.DataFrame:
    """Convert supported inputs into a float DataFrame without NaNs."""

    if isinstance(returns, pd.DataFrame):
        df = returns.copy()
    elif isinstance(returns, pd.Series):
        df = returns.to_frame()
    else:
        array = np.asarray(returns, dtype=float)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2:
            raise ValueError("returns must be 1D/2D array-like.")
        df = pd.DataFrame(array)

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=0, how="any")
    if df.empty:
        raise ValueError("No valid observations remaining after cleaning.")
    return df.astype(float)


def mean_return(
    returns: ReturnsLike,
    *,
    method: str = "simple",
) -> pd.Series:
    """Baseline mean estimator supporting simple and geometric averages."""

    clean = _ensure_returns_frame(returns)
    method = method.lower()

    if method == "simple":
        means = clean.mean(axis=0)
    elif method == "geometric":
        growth = 1.0 + clean
        if (growth <= 0).any().any():
            raise ValueError("Geometric mean requires all returns > -100%.")
        means = growth.prod(axis=0) ** (1.0 / len(clean)) - 1.0
    else:
        raise ValueError("Unsupported mean method '%s'." % method)

    return means.astype(float)


def huber_mean(
    returns: ReturnsLike,
    *,
    c: float = 1.5,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Huber-robust mean estimate and effective weights per observation."""

    if c <= 0:
        raise ValueError("c must be positive.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if tol <= 0:
        raise ValueError("tol must be positive.")

    clean = _ensure_returns_frame(returns)
    values = clean.to_numpy(dtype=float)
    n_obs, n_assets = values.shape

    mu = np.median(values, axis=0)
    scale = np.median(np.abs(values - mu), axis=0) / 0.6745
    scale = np.where(scale <= 1e-8, values.std(axis=0, ddof=1), scale)
    scale = np.where(scale <= 1e-8, 1e-4, scale)

    weights = np.ones_like(values)

    for _ in range(max_iter):
        residuals = values - mu
        threshold = c * scale
        abs_res = np.abs(residuals)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = threshold / np.maximum(abs_res, 1e-12)
        new_weights = np.minimum(1.0, ratio)
        new_weights = np.where(abs_res == 0, 1.0, new_weights)

        weighted = new_weights * values
        denom = np.maximum(new_weights.sum(axis=0), 1e-12)
        mu_new = weighted.sum(axis=0) / denom

        if np.linalg.norm(mu_new - mu) <= tol * max(np.linalg.norm(mu), 1.0):
            weights = new_weights
            mu = mu_new
            break

        mu = mu_new
        weights = new_weights

        residuals_new = values - mu
        scale = np.sqrt(
            (weights * residuals_new**2).sum(axis=0)
            / np.maximum(weights.sum(axis=0), 1e-12)
        )
        robust_scale = np.median(np.abs(residuals_new), axis=0) / 0.6745
        scale = np.where(scale <= 1e-8, robust_scale, scale)
        scale = np.where(scale <= 1e-8, 1e-4, scale)

    mu_series = pd.Series(mu, index=clean.columns, dtype=float)
    weights_df = pd.DataFrame(weights, index=clean.index, columns=clean.columns)

    return mu_series, weights_df


def student_t_mean(
    returns: ReturnsLike,
    *,
    nu: float = 5.0,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> pd.Series:
    """Maximum-likelihood location estimator for Student-t returns."""

    if nu <= 1:
        raise ValueError("nu must exceed 1 for a finite mean.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if tol <= 0:
        raise ValueError("tol must be positive.")

    clean = _ensure_returns_frame(returns)
    values = clean.to_numpy(dtype=float)
    mu = values.mean(axis=0)
    scale = values.var(axis=0, ddof=1)
    scale = np.where(scale <= 1e-8, 1e-4, scale)

    for _ in range(max_iter):
        residuals = values - mu
        scaled_res = residuals**2 / scale
        weights = (nu + 1.0) / (nu + scaled_res)
        mu_new = (weights * values).sum(axis=0) / weights.sum(axis=0)
        if np.linalg.norm(mu_new - mu) <= tol * max(np.linalg.norm(mu), 1.0):
            mu = mu_new
            break
        mu = mu_new
        weighted_var = (weights * (values - mu) ** 2).sum(axis=0) / weights.sum(axis=0)
        scale = np.where(weighted_var <= 1e-8, 1e-4, weighted_var)

    return pd.Series(mu, index=clean.columns, dtype=float)


def bayesian_shrinkage_mean(
    returns: ReturnsLike,
    *,
    prior: Optional[Union[pd.Series, Sequence[float], float]] = None,
    strength: float = 0.2,
) -> pd.Series:
    """Linear shrinkage of the sample mean towards a prior benchmark."""

    if strength < 0 or strength > 1:
        raise ValueError("strength must lie in [0, 1].")

    clean = _ensure_returns_frame(returns)
    sample = clean.mean(axis=0)

    if prior is None:
        prior_series = pd.Series(0.0, index=sample.index)
    elif isinstance(prior, pd.Series):
        prior_series = prior.astype(float).reindex(sample.index)
    else:
        prior_array = np.asarray(prior, dtype=float).flatten()
        if prior_array.size == 1:
            prior_series = pd.Series(prior_array[0], index=sample.index)
        elif prior_array.size == len(sample.index):
            prior_series = pd.Series(prior_array, index=sample.index)
        else:
            raise ValueError("prior length must be 1 or match number of assets.")

    if prior_series.isnull().any():
        raise ValueError("prior must provide values for all assets.")

    shrunk = (1.0 - strength) * sample + strength * prior_series
    return shrunk.astype(float)


def confidence_intervals(
    returns: ReturnsLike,
    *,
    method: str = "bootstrap",
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Empirical confidence intervals for mean returns."""

    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive.")

    clean = _ensure_returns_frame(returns)
    method = method.lower()

    if method != "bootstrap":
        raise ValueError("Only 'bootstrap' method is currently supported.")

    rng = np.random.default_rng(random_state)
    n_obs, n_assets = clean.shape
    samples = np.empty((n_bootstrap, n_assets), dtype=float)
    values = clean.to_numpy(dtype=float)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n_obs, size=n_obs)
        samples[i] = values[idx].mean(axis=0)

    lower = np.quantile(samples, alpha / 2.0, axis=0)
    upper = np.quantile(samples, 1.0 - alpha / 2.0, axis=0)

    return pd.DataFrame(
        {"lower": lower, "upper": upper},
        index=clean.columns,
        dtype=float,
    )


def blend_with_black_litterman(
    mu_prior: Union[pd.Series, Sequence[float], np.ndarray],
    cov: Union[pd.DataFrame, np.ndarray],
    *,
    views: Optional[Sequence[dict]] = None,
    **kwargs,
) -> pd.Series:
    """Delegate to Black-Litterman when views are present, otherwise passthrough."""

    if isinstance(cov, pd.DataFrame):
        assets = list(cov.columns)
        cov_array = cov.to_numpy(dtype=float)
    else:
        cov_array = np.asarray(cov, dtype=float)
        if cov_array.ndim != 2 or cov_array.shape[0] != cov_array.shape[1]:
            raise ValueError("cov must be a square matrix.")
        assets = [f"a{i}" for i in range(cov_array.shape[0])]

    if views is None or len(views) == 0:
        if isinstance(mu_prior, pd.Series):
            mu_series = mu_prior.astype(float).reindex(assets)
            if mu_series.isnull().any():
                raise ValueError("mu_prior must provide values for all assets.")
            mu_series.index = assets
            return mu_series
        mu_arr = np.asarray(mu_prior, dtype=float).flatten()
        if mu_arr.ndim != 1 or len(mu_arr) != len(assets):
            raise ValueError("mu_prior must align with covariance dimension.")
        return pd.Series(mu_arr, index=assets, dtype=float)

    bl_result = bl.black_litterman(
        cov=pd.DataFrame(cov_array, index=assets, columns=assets),
        pi=mu_prior,
        views=list(views),
        **kwargs,
    )
    return bl_result["mu_bl"].astype(float)


def annualize(
    mu: Union[pd.Series, pd.DataFrame, Sequence[float], np.ndarray],
    *,
    periods_per_year: float,
    compound: bool = False,
) -> Union[pd.Series, pd.DataFrame]:
    """Convert periodic mean returns to annualised figures."""

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")

    if isinstance(mu, pd.DataFrame):
        data = mu.astype(float)
        if compound:
            return (1.0 + data) ** periods_per_year - 1.0
        return data * periods_per_year
    if isinstance(mu, pd.Series):
        data = mu.astype(float)
        if compound:
            return (1.0 + data) ** periods_per_year - 1.0
        return data * periods_per_year

    array = np.asarray(mu, dtype=float)
    if array.ndim == 1:
        if compound:
            annual = (1.0 + array) ** periods_per_year - 1.0
        else:
            annual = array * periods_per_year
        return pd.Series(annual)

    if array.ndim == 2:
        if compound:
            annual = (1.0 + array) ** periods_per_year - 1.0
        else:
            annual = array * periods_per_year
        return pd.DataFrame(annual)

    raise ValueError("mu must be 1D or 2D structure.")
