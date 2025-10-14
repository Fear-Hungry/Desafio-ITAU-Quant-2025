"""Factor model utilities used across expected-return estimators.

The helpers in this module provide a light-weight factor modelling toolkit that
favours Pandas-friendly inputs/outputs so the downstream orchestration layer can
chain operations without losing labels.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from itau_quant.estimators import cov as cov_estimators

DataLike = Union[pd.DataFrame, pd.Series, np.ndarray, Sequence[Sequence[float]]]

__all__ = [
    "prepare_factor_data",
    "time_series_regression",
    "cross_sectional_regression",
    "shrink_betas",
    "factor_covariance",
    "implied_asset_returns",
    "principal_component_factors",
]


def _to_dataframe(data: DataLike, *, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Convert supported inputs into a float DataFrame with unique index."""

    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, pd.Series):
        df = data.to_frame()
    else:
        array = np.asarray(data, dtype=float)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2:
            raise ValueError("Input must be convertible to a 2D array.")
        df = pd.DataFrame(array, columns=columns)

    df = df.sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.astype(float)


def _winsorize(df: pd.DataFrame, lower: float, upper: float) -> pd.DataFrame:
    """Clip extreme observations column-wise according to quantile cut-offs."""

    lower_bounds = df.quantile(lower)
    upper_bounds = df.quantile(upper)
    return df.clip(lower=lower_bounds, upper=upper_bounds, axis=1)


def _zscore(df: pd.DataFrame, window: Optional[int]) -> pd.DataFrame:
    """Return rolling (or global) z-scores with sensible safeguards."""

    if window is None or window <= 1:
        mean = df.mean(axis=0)
        std = df.std(axis=0, ddof=0)
        std = std.replace(0, np.nan)
        z = (df - mean) / std
    else:
        rolling_mean = df.rolling(window=window, min_periods=2).mean()
        rolling_std = df.rolling(window=window, min_periods=2).std(ddof=0)
        rolling_std = rolling_std.replace(0, np.nan)
        z = (df - rolling_mean) / rolling_std

    z = z.replace([np.inf, -np.inf], np.nan)
    z = z.dropna(how="any")
    return z


def prepare_factor_data(
    prices: DataLike,
    factor_returns: DataLike,
    window: Optional[int] = 60,
    *,
    winsor_limits: Tuple[float, float] = (0.01, 0.99),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Align and standardise asset and factor returns.

    Parameters
    ----------
    prices
        Price history for the assets (index: datetime, columns: tickers).
    factor_returns
        Returns of systematic factors already expressed in excess of risk-free.
    window
        Window length for rolling z-score normalisation.  When ``None`` the
        entire sample mean/std are used.
    winsor_limits
        Tuple with lower/upper quantiles used to clip outliers.

    Returns
    -------
    (asset_returns, factor_returns)
        Both DataFrames z-scored and aligned on the intersection of timestamps.
    """

    if not 0 <= winsor_limits[0] < winsor_limits[1] <= 1:
        raise ValueError("winsor_limits must satisfy 0 <= lower < upper <= 1.")

    price_df = _to_dataframe(prices)
    asset_returns = price_df.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    factor_df = _to_dataframe(factor_returns)
    factor_df = factor_df.dropna(how="all")

    common_index = asset_returns.index.intersection(factor_df.index)
    if common_index.empty:
        raise ValueError("Assets and factors share no overlapping dates.")

    asset_returns = asset_returns.loc[common_index]
    factor_df = factor_df.loc[common_index]

    mask = (
        asset_returns.notna().all(axis=1) & factor_df.notna().all(axis=1)
    )
    asset_returns = asset_returns.loc[mask]
    factor_df = factor_df.loc[mask]

    asset_returns = _winsorize(asset_returns, *winsor_limits)
    factor_df = _winsorize(factor_df, *winsor_limits)

    asset_z = _zscore(asset_returns, window)
    factor_z = _zscore(factor_df, window)

    common_index = asset_z.index.intersection(factor_z.index)
    if common_index.empty:
        raise ValueError("Insufficient data after normalisation; adjust window.")

    asset_z = asset_z.loc[common_index]
    factor_z = factor_z.loc[common_index]

    return asset_z, factor_z


def time_series_regression(
    returns: DataLike,
    factors: DataLike,
    *,
    add_constant: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.Series], pd.DataFrame]:
    """Estimate betas via time-series regression for each asset.

    Returns
    -------
    betas : DataFrame (n_factors, n_assets)
        Exposure of every asset to each factor.
    alphas : Series (n_assets) or ``None``
        Intercepts when ``add_constant`` is ``True``.
    residuals : DataFrame (n_observations, n_assets)
        Regression residuals preserving the original index.
    """

    returns_df = _to_dataframe(returns)
    factors_df = _to_dataframe(factors)

    common_index = returns_df.index.intersection(factors_df.index)
    if common_index.empty:
        raise ValueError("Returns and factors have no overlapping observations.")

    returns_df = returns_df.loc[common_index]
    factors_df = factors_df.loc[common_index]

    mask = returns_df.notna().all(axis=1) & factors_df.notna().all(axis=1)
    returns_df = returns_df.loc[mask]
    factors_df = factors_df.loc[mask]

    if returns_df.empty or factors_df.empty:
        raise ValueError("No data left after cleaning inputs.")

    X = factors_df.to_numpy(dtype=float)
    if add_constant:
        X = np.column_stack([np.ones(len(X)), X])
        factor_names = ["const", *factors_df.columns.tolist()]
    else:
        factor_names = factors_df.columns.tolist()

    Y = returns_df.to_numpy(dtype=float)
    coef, residuals, _, _ = np.linalg.lstsq(X, Y, rcond=None)

    if add_constant:
        alphas = pd.Series(coef[0], index=returns_df.columns, dtype=float)
        beta_values = coef[1:]
    else:
        alphas = None
        beta_values = coef

    betas = pd.DataFrame(beta_values, index=factor_names[1:] if add_constant else factor_names, columns=returns_df.columns)

    fitted = X @ coef
    resid = returns_df.to_numpy() - fitted
    residuals_df = pd.DataFrame(resid, index=returns_df.index, columns=returns_df.columns)

    return betas, alphas, residuals_df


def cross_sectional_regression(
    betas: pd.DataFrame,
    future_returns: Union[pd.Series, pd.DataFrame, Sequence[float]],
    *,
    add_constant: bool = True,
) -> Tuple[pd.Series, Optional[float]]:
    """Estimate factor premia from cross-sectional returns.

    Parameters
    ----------
    betas
        DataFrame with index as factor names and columns as assets.
    future_returns
        Realised return vector aligned with ``betas`` columns.
    add_constant
        If ``True`` includes an intercept in the regression.
    """

    if isinstance(future_returns, pd.DataFrame):
        if future_returns.shape[1] != 1:
            raise ValueError("future_returns DataFrame must have a single column.")
        future_series = future_returns.iloc[:, 0]
    elif isinstance(future_returns, pd.Series):
        future_series = future_returns
    else:
        future_series = pd.Series(future_returns, index=betas.columns, dtype=float)

    future_series = future_series.astype(float)

    betas_df = betas.copy()

    common_assets = betas_df.columns.intersection(future_series.dropna().index)
    if common_assets.empty:
        raise ValueError("No overlapping assets between betas and future_returns.")

    X = betas_df[common_assets].T.to_numpy(dtype=float)
    y = future_series.loc[common_assets].to_numpy(dtype=float)

    if add_constant:
        X = np.column_stack([np.ones(len(X)), X])

    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    if add_constant:
        alpha = float(coef[0])
        premia_vals = coef[1:]
    else:
        alpha = None
        premia_vals = coef

    premia = pd.Series(premia_vals, index=betas_df.index, dtype=float)

    return premia, alpha


def shrink_betas(
    betas: pd.DataFrame,
    *,
    method: str = "ridge",
    alpha: float = 0.1,
) -> pd.DataFrame:
    """Apply simple shrinkage schemes to reduce noise in betas."""

    if alpha < 0:
        raise ValueError("alpha must be non-negative.")

    betas_df = betas.astype(float).copy()
    method = method.lower()

    if method == "ridge":
        shrunk = betas_df / (1.0 + alpha)
    elif method == "lasso":
        values = betas_df.to_numpy()
        shrunk_values = np.sign(values) * np.maximum(np.abs(values) - alpha, 0.0)
        shrunk = pd.DataFrame(shrunk_values, index=betas_df.index, columns=betas_df.columns)
    elif method == "grand_mean":
        mean_beta = betas_df.mean(axis=1)
        target = pd.DataFrame(
            np.repeat(mean_beta.values[:, None], betas_df.shape[1], axis=1),
            index=betas_df.index,
            columns=betas_df.columns,
        )
        shrunk = (1.0 - alpha) * betas_df + alpha * target
    else:
        raise ValueError("Unsupported shrinkage method '%s'." % method)

    return shrunk


def factor_covariance(
    factors: DataLike,
    *,
    method: str = "sample",
    **kwargs,
) -> pd.DataFrame:
    """Estimate the factor covariance matrix using selected methodology."""

    factors_df = _to_dataframe(factors).dropna(how="any")

    method = method.lower()
    if method == "sample":
        return cov_estimators.sample_cov(factors_df, **kwargs)
    if method == "ledoit_wolf":
        cov_df, _ = cov_estimators.ledoit_wolf_shrinkage(factors_df, **kwargs)
        return cov_df
    if method == "tyler":
        return cov_estimators.tyler_m_estimator(factors_df, **kwargs)
    if method == "nonlinear":
        return cov_estimators.nonlinear_shrinkage(factors_df, **kwargs)

    raise ValueError("Unsupported covariance estimation method '%s'." % method)


def implied_asset_returns(
    betas: pd.DataFrame,
    factor_premia: Union[pd.Series, Sequence[float]],
    residual_alpha: Optional[Union[pd.Series, Sequence[float]]] = None,
) -> pd.Series:
    """Reconstruct expected asset returns from factor premia."""

    beta_df = betas.astype(float)

    if isinstance(factor_premia, pd.Series):
        premia = factor_premia.astype(float)
    else:
        premia = pd.Series(factor_premia, index=beta_df.index, dtype=float)

    common_factors = beta_df.index.intersection(premia.index)
    if len(common_factors) != len(beta_df.index):
        missing = set(beta_df.index) - set(common_factors)
        raise ValueError(f"Missing premia for factors: {sorted(missing)}")

    expected = beta_df.loc[common_factors].T @ premia.loc[common_factors]

    if residual_alpha is not None:
        if isinstance(residual_alpha, pd.Series):
            alpha_series = residual_alpha.astype(float)
        else:
            alpha_series = pd.Series(residual_alpha, index=beta_df.columns, dtype=float)
        expected = expected.add(alpha_series, fill_value=0.0)

    return expected.sort_index()


def principal_component_factors(
    returns: DataLike,
    n_components: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Extract statistical factors via PCA.

    Returns
    -------
    factor_returns : DataFrame (n_obs, n_components)
        Time series of the principal components scaled to match the data.
    loadings : DataFrame (n_assets, n_components)
        Asset exposures to each component.
    explained_variance_ratio : ndarray (n_components,)
        Share of total variance explained by every component.
    """

    returns_df = _to_dataframe(returns)
    returns_df = returns_df.dropna(how="any")

    if n_components <= 0:
        raise ValueError("n_components must be positive.")

    n_samples, n_assets = returns_df.shape
    if n_components > min(n_samples, n_assets):
        raise ValueError("n_components exceeds the allowable rank.")

    demeaned = returns_df - returns_df.mean(axis=0)
    matrix = demeaned.to_numpy(dtype=float)

    U, singular_values, Vt = np.linalg.svd(matrix, full_matrices=False)

    comps = min(n_components, len(singular_values))
    singular_values = singular_values[:comps]
    U = U[:, :comps]
    Vt = Vt[:comps, :]

    factor_ts = U * singular_values
    loadings = Vt.T

    component_names = [f"PC{i+1}" for i in range(comps)]

    factor_returns = pd.DataFrame(factor_ts, index=returns_df.index, columns=component_names)
    loadings_df = pd.DataFrame(loadings, index=returns_df.columns, columns=component_names)

    variance = (singular_values**2) / (n_samples - 1)
    total_variance = (np.linalg.norm(matrix, ord="fro") ** 2) / (n_samples - 1)
    if np.isclose(total_variance, 0.0):
        explained_ratio = np.zeros_like(variance)
    else:
        explained_ratio = variance / total_variance

    return factor_returns, loadings_df, explained_ratio
