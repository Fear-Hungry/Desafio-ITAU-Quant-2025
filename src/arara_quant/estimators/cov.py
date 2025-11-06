"""Covariance estimators used across the allocation stack.

The implementations below provide a mix of classical (sample, Ledoit-Wolf),
robust (Tyler M-estimator), and heavy-tail (Student-\t) estimators together
with a few utility helpers.  Each routine accepts a ``pandas.DataFrame``
containing aligned asset returns so that labels are preserved end-to-end.

References
----------
Ledoit, O. and Wolf, M. (2004), *A Well-Conditioned Estimator for Large-Dimensional
    Covariance Matrices*. Journal of Multivariate Analysis 88.
Ledoit, O. and Wolf, M. (2018), *Nonlinear Shrinkage of the Covariance Matrix Estimator
    via Random Matrix Theory*. The Annals of Statistics 46.
Tyler, D. (1987), *A Distribution-Free M-Estimator of Multivariate Scatter*. The
    Annals of Statistics 15.
"""

from __future__ import annotations

import warnings
from typing import Sequence, Union

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError

ArrayOrFrame = Union[pd.DataFrame, pd.Series, np.ndarray, Sequence[Sequence[float]]]

__all__ = [
    "sample_cov",
    "ledoit_wolf_shrinkage",
    "nonlinear_shrinkage",
    "tyler_m_estimator",
    "student_t_cov",
    "project_to_psd",
    "regularize_cov",
    "graphical_lasso_cov",
]


def _ensure_dataframe(returns: ArrayOrFrame, min_obs: int = 2) -> pd.DataFrame:
    """Uniformly convert inputs to a float DataFrame and drop NaNs.

    Parameters
    ----------
    returns
        Historical returns arranged as observations (rows) by assets (columns).
    min_obs
        Minimum number of non-null observations required.

    Returns
    -------
    pandas.DataFrame
        Cleaned view of the input ready for estimation.
    """

    if isinstance(returns, pd.DataFrame):
        df = returns.copy()
    elif isinstance(returns, pd.Series):
        df = returns.to_frame()
    else:
        array = np.asarray(returns, dtype=float)
        if array.ndim != 2:
            raise ValueError("returns must be a 2D array-like structure.")
        df = pd.DataFrame(array)

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=0, how="any")

    if df.empty or df.shape[0] < min_obs:
        raise ValueError("Not enough observations after removing NaNs.")
    if df.shape[1] < 1:
        raise ValueError("returns must contain at least one column.")

    return df.astype(float)


def _format_matrix(matrix: np.ndarray, labels: Sequence[str]) -> pd.DataFrame:
    """Return a symmetric DataFrame preserving asset labels."""

    sym = 0.5 * (matrix + matrix.T)
    return pd.DataFrame(sym, index=labels, columns=labels)


def _warn_if_ill_conditioned(matrix: np.ndarray, threshold: float = 1e12) -> None:
    """Emit warnings for poorly conditioned matrices."""

    try:
        cond_number = np.linalg.cond(matrix)
    except LinAlgError:
        warnings.warn("Covariance matrix appears singular.", RuntimeWarning)
        return

    if not np.isfinite(cond_number):
        warnings.warn("Covariance matrix conditioning is not finite.", RuntimeWarning)
        return

    if cond_number > threshold:
        warnings.warn(
            f"Covariance matrix is poorly conditioned (cond > {threshold:.1e}).",
            RuntimeWarning,
        )


def project_to_psd(
    matrix: pd.DataFrame | np.ndarray, epsilon: float = 1e-6
) -> pd.DataFrame | np.ndarray:
    """Project a symmetric matrix onto the positive semi-definite cone.

    The routine clips negative eigenvalues to ``epsilon`` and reconstructs the
    matrix.  Small ``epsilon`` values preserve the original structure while
    avoiding numerical violations of PSD constraints.
    """

    if isinstance(matrix, pd.DataFrame):
        labels = matrix.index
        array = matrix.to_numpy(dtype=float)
    else:
        labels = None
        array = np.asarray(matrix, dtype=float)

    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("matrix must be square.")

    array = 0.5 * (array + array.T)
    eigvals, eigvecs = np.linalg.eigh(array)
    eigvals_clipped = np.clip(eigvals, epsilon, None)
    projected = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    projected = 0.5 * (projected + projected.T)

    if labels is not None:
        return pd.DataFrame(projected, index=labels, columns=labels)
    return projected


def sample_cov(
    returns: ArrayOrFrame,
    *,
    ddof: int = 1,
    min_obs: int = 2,
) -> pd.DataFrame:
    """Classical sample covariance with shape validation and conditioning checks."""

    if ddof < 0:
        raise ValueError("ddof must be non-negative.")

    clean = _ensure_dataframe(returns, min_obs=min_obs)
    cov = clean.cov(ddof=ddof)
    cov = cov.astype(float)
    _warn_if_ill_conditioned(cov.to_numpy())

    return cov


def ledoit_wolf_shrinkage(
    returns: ArrayOrFrame,
    *,
    assume_centered: bool = False,
) -> tuple[pd.DataFrame, float]:
    """Ledoit-Wolf linear shrinkage estimator.

    Parameters
    ----------
    returns
        Observações de retornos (n_observations x n_assets).
    assume_centered
        Se ``True`` assume que ``returns`` já está centrado em zero.

    Returns
    -------
    (covariance, shrinkage)
        DataFrame com a matriz shrinkada e o parâmetro ótimo calculado.
    """

    clean = _ensure_dataframe(returns, min_obs=2)
    values = clean.to_numpy(dtype=float)
    if not assume_centered:
        values = values - values.mean(axis=0, keepdims=True)

    n_samples, n_assets = values.shape
    if n_samples <= 1:
        raise ValueError("At least two observations required for shrinkage.")

    emp_cov = (values.T @ values) / float(n_samples)
    mu = np.trace(emp_cov) / n_assets
    target = np.eye(n_assets) * mu

    diff = emp_cov - target
    alpha = np.linalg.norm(diff, ord="fro") ** 2
    if alpha <= 0:
        shrinkage = 0.0
    else:
        beta = 0.0
        for row in values:
            outer = np.outer(row, row)
            beta += np.linalg.norm(outer - emp_cov, ord="fro") ** 2
        beta /= n_samples**2
        shrinkage = float(np.clip(beta / alpha, 0.0, 1.0))

    shrunk = shrinkage * target + (1.0 - shrinkage) * emp_cov
    cov = _format_matrix(shrunk, clean.columns)
    cov = project_to_psd(cov, epsilon=1e-9)
    _warn_if_ill_conditioned(cov.to_numpy())

    return cov, float(shrinkage)


def nonlinear_shrinkage(
    returns: ArrayOrFrame,
    *,
    epsilon: float = 1e-6,
) -> pd.DataFrame:
    """Non-linear shrinkage based on random matrix theory heuristics.

    The implementation follows the spirit of Ledoit & Wolf (2018) by shrinking
    eigenvalues lying in the Marchenko-Pastur bulk towards their average.  While
    not a full QuEST solution, it provides a stable PSD covariance suited for
    portfolio optimisation in high-dimensional regimes (``p`` close to ``n``).
    """

    clean = _ensure_dataframe(returns, min_obs=2)
    values = clean.to_numpy()
    values = values - values.mean(axis=0, keepdims=True)
    n_samples, n_assets = values.shape

    if n_samples <= 1:
        raise ValueError("At least two observations required for shrinkage.")

    sample_covariance = np.cov(values, rowvar=False, ddof=1)
    std = np.sqrt(np.diag(sample_covariance))
    std = np.where(std > 0, std, epsilon)
    corr = sample_covariance / np.outer(std, std)
    corr = 0.5 * (corr + corr.T)

    eigvals, eigvecs = np.linalg.eigh(corr)
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    ratio = n_assets / float(n_samples)
    sqrt_ratio = np.sqrt(ratio)
    lambda_plus = (1.0 + sqrt_ratio) ** 2
    lambda_minus = (1.0 - sqrt_ratio) ** 2 if ratio < 1 else 0.0

    bulk_mask = (eigvals >= lambda_minus) & (eigvals <= lambda_plus)
    if bulk_mask.any():
        bulk_mean = float(np.mean(eigvals[bulk_mask]))
        eigvals[bulk_mask] = bulk_mean

    eigvals = np.clip(eigvals, epsilon, None)
    cleaned_corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    cleaned_corr = 0.5 * (cleaned_corr + cleaned_corr.T)
    cleaned_cov = cleaned_corr * np.outer(std, std)

    cov = _format_matrix(cleaned_cov, clean.columns)
    cov = project_to_psd(cov, epsilon=epsilon)
    _warn_if_ill_conditioned(cov.to_numpy())

    return cov


def graphical_lasso_cov(
    returns: ArrayOrFrame,
    *,
    alpha: float = 0.01,
    max_iter: int = 100,
    tol: float = 1e-4,
    assume_centered: bool = False,
    enet_tol: float = 1e-4,
    mode: str = "cd",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Graphical Lasso covariance/precision estimator.

    Parameters
    ----------
    returns
        Historical returns arranged as observations by assets.
    alpha
        Regularisation strength (L1 penalty). Must be positive.
    max_iter
        Maximum number of solver iterations.
    tol
        Convergence tolerance for the coordinate descent loop.
    assume_centered
        Whether to skip explicit mean subtraction.
    enet_tol
        Duality gap threshold for the internal lasso solver.
    mode
        One of {'cd', 'lars'} as supported by ``sklearn``.

    Returns
    -------
    (covariance, precision)
        DataFrames containing the estimated covariance and precision matrices.
    """

    if alpha <= 0:
        raise ValueError("alpha must be strictly positive for Graphical Lasso.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if tol <= 0 or enet_tol <= 0:
        raise ValueError("tol and enet_tol must be positive.")

    clean = _ensure_dataframe(returns, min_obs=2)
    values = clean.to_numpy(dtype=float)

    try:
        from sklearn.covariance import GraphicalLasso
    except ImportError as exc:  # pragma: no cover - dependency guaranteed in prod
        raise ImportError(
            "GraphicalLasso requires scikit-learn to be installed."
        ) from exc

    model = GraphicalLasso(
        alpha=float(alpha),
        max_iter=int(max_iter),
        tol=float(tol),
        assume_centered=assume_centered,
        enet_tol=float(enet_tol),
        mode=mode,
    )
    model.fit(values)

    cov_df = _format_matrix(model.covariance_, clean.columns)
    precision_df = _format_matrix(model.precision_, clean.columns)

    _warn_if_ill_conditioned(cov_df.to_numpy())

    return cov_df, precision_df


def tyler_m_estimator(
    returns: ArrayOrFrame,
    *,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """Tyler's M-estimator for elliptical distributions.

    The scatter matrix is normalised to have trace equal to the number of
    assets, which ensures scale invariance.
    """

    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if tol <= 0:
        raise ValueError("tol must be positive.")

    clean = _ensure_dataframe(returns, min_obs=2)
    values = clean.to_numpy(dtype=float)
    values = values - values.mean(axis=0, keepdims=True)
    n_samples, n_assets = values.shape

    scatter = np.cov(values, rowvar=False, ddof=1)
    scatter = project_to_psd(scatter, epsilon=1e-12)

    for _ in range(max_iter):
        scatter_inv = np.linalg.pinv(scatter)
        mahal = np.sum((values @ scatter_inv) * values, axis=1)
        if np.any(mahal <= 0):
            raise ValueError("Encountered non-positive Mahalanobis distance.")
        weights = n_assets / mahal
        weighted = (values.T * weights) @ values / n_samples
        weighted = project_to_psd(weighted, epsilon=1e-12)
        trace = np.trace(weighted)
        if trace <= 0:
            raise ValueError("Tyler estimator produced invalid scatter.")
        updated = weighted * (n_assets / trace)
        diff = np.linalg.norm(scatter - updated, ord="fro")
        norm = np.linalg.norm(scatter, ord="fro")
        scatter = updated
        if diff <= tol * max(norm, 1.0):
            break
    else:
        warnings.warn(
            "Tyler estimator did not converge within max_iter.", RuntimeWarning
        )

    cov = _format_matrix(scatter, clean.columns)
    cov = project_to_psd(cov, epsilon=1e-9)
    _warn_if_ill_conditioned(cov.to_numpy())

    return cov


def student_t_cov(
    returns: ArrayOrFrame,
    *,
    nu: float,
) -> pd.DataFrame:
    """Covariance under a multivariate Student-t model.

    For ``nu`` degrees of freedom greater than two, the theoretical covariance is
    :math:`\frac{\nu}{\nu - 2}` times the scatter matrix.  We therefore scale the
    sample covariance accordingly to recover the population covariance.
    """

    if nu <= 2:
        raise ValueError("nu must be greater than 2 for a finite covariance.")

    cov = sample_cov(returns, ddof=1)
    scaled = cov * (nu / (nu - 2.0))
    scaled = scaled.astype(float)
    _warn_if_ill_conditioned(scaled.to_numpy())

    return scaled


def regularize_cov(
    matrix: pd.DataFrame | np.ndarray,
    *,
    method: str = "diag",
    floor: float | None = None,
) -> pd.DataFrame | np.ndarray:
    """Apply simple conditioning fixes on a covariance matrix.

    Parameters
    ----------
    matrix
        Covariance matrix to regularise.
    method
        ``"diag"`` floors the diagonal; ``"ridge"`` adds ``floor`` times the
        identity; ``"shrink_to_identity"`` mixes the matrix with the identity.
    floor
        Scalar controlling the strength of the regularisation.  When omitted,
        sensible defaults are chosen per method.
    """

    if isinstance(matrix, pd.DataFrame):
        labels = matrix.index
        array = matrix.to_numpy(dtype=float)
    else:
        labels = None
        array = np.asarray(matrix, dtype=float)

    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("matrix must be square.")

    method = method.lower()

    if method == "diag":
        diag = np.diag(array).copy()
        if floor is None:
            floor = float(np.median(diag)) * 1e-3 if np.all(diag > 0) else 1e-6
        diag = np.maximum(diag, floor)
        array = array.copy()
        np.fill_diagonal(array, diag)
    elif method == "ridge":
        if floor is None:
            floor = 1e-4
        array = array + float(floor) * np.eye(array.shape[0])
    elif method == "shrink_to_identity":
        if floor is None:
            floor = 0.1
        floor = float(np.clip(floor, 0.0, 1.0))
        trace = np.trace(array) / array.shape[0]
        identity = np.eye(array.shape[0]) * trace
        array = (1.0 - floor) * array + floor * identity
    else:
        raise ValueError(f"Unsupported regularisation method '{method}'.")

    array = project_to_psd(array, epsilon=1e-9)

    if labels is not None:
        return pd.DataFrame(array, index=labels, columns=labels)
    return array
