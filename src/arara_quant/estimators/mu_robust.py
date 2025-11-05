"""Robust μ estimators with shrinkage to combat overfit.

Implements James-Stein shrinkage and Bayesian shrinkage to reduce
estimation error and close the ex-ante / OOS performance gap.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

__all__ = [
    "james_stein_shrinkage",
    "bayesian_shrinkage",
    "combined_shrinkage",
    "shrink_mu_pipeline",
]


def james_stein_shrinkage(
    mu: pd.Series,
    sigma: pd.DataFrame,
    *,
    T: int | None = None,
    target: float = 0.0,
) -> pd.Series:
    """Apply James-Stein shrinkage to Sharpe ratios.

    Shrinks individual Sharpe ratios toward grand mean to reduce estimation error.
    Based on Stein (1956) and modern portfolio applications.

    Parameters
    ----------
    mu : pd.Series
        Expected returns (annualized)
    sigma : pd.DataFrame
        Covariance matrix (annualized)
    T : int, optional
        Number of observations used to estimate μ (for scaling)
        If None, assumes shrinkage based on magnitude only
    target : float
        Target Sharpe ratio to shrink toward (default 0)

    Returns
    -------
    pd.Series
        Shrunk expected returns
    """
    assets = list(mu.index)
    mu_vec = mu.reindex(assets).to_numpy(dtype=float)
    sigma_mat = sigma.reindex(index=assets, columns=assets).to_numpy(dtype=float)

    # Compute individual Sharpe ratios
    vol = np.sqrt(np.diag(sigma_mat))
    sharpe = mu_vec / (vol + 1e-12)

    # James-Stein shrinkage factor
    # ϕ = max(0, 1 - (N-2) / ||z||²)
    N = len(assets)
    z_squared_sum = np.sum((sharpe - target) ** 2)

    if z_squared_sum < 1e-12:
        # All Sharpes near target, no shrinkage needed
        return mu

    shrinkage_factor = max(0.0, 1.0 - (N - 2) / z_squared_sum)

    # Shrink Sharpes
    sharpe_shrunk = target + shrinkage_factor * (sharpe - target)

    # Convert back to μ
    mu_shrunk = sharpe_shrunk * vol

    return pd.Series(mu_shrunk, index=assets, dtype=float)


def bayesian_shrinkage(
    mu: pd.Series,
    *,
    prior: pd.Series | float = 0.0,
    gamma: float = 0.75,
) -> pd.Series:
    """Bayesian shrinkage toward a prior.

    μ_shrunk = (1-γ) μ + γ μ_prior

    Parameters
    ----------
    mu : pd.Series
        Expected returns (annualized)
    prior : pd.Series or float
        Prior expected returns (default 0)
    gamma : float
        Shrinkage intensity ∈ [0,1]
        0 = no shrinkage, 1 = full shrinkage to prior

    Returns
    -------
    pd.Series
        Shrunk expected returns
    """
    if not 0 <= gamma <= 1:
        raise ValueError(f"gamma must be in [0,1], got {gamma}")

    assets = list(mu.index)

    if isinstance(prior, (int, float)):
        prior_vec = np.full(len(assets), float(prior))
    else:
        prior_vec = prior.reindex(assets).fillna(0.0).to_numpy(dtype=float)

    mu_vec = mu.reindex(assets).to_numpy(dtype=float)

    mu_shrunk = (1 - gamma) * mu_vec + gamma * prior_vec

    return pd.Series(mu_shrunk, index=assets, dtype=float)


def combined_shrinkage(
    mu: pd.Series,
    sigma: pd.DataFrame,
    *,
    T: int | None = None,
    prior: pd.Series | float = 0.0,
    gamma: float = 0.75,
    alpha: float = 0.5,
) -> pd.Series:
    """Combined James-Stein + Bayesian shrinkage.

    μ_final = α × μ_bayesian + (1-α) × μ_JS

    Parameters
    ----------
    mu : pd.Series
        Expected returns (annualized)
    sigma : pd.DataFrame
        Covariance matrix (annualized)
    T : int, optional
        Number of observations for JS scaling
    prior : pd.Series or float
        Prior for Bayesian shrinkage (default 0)
    gamma : float
        Bayesian shrinkage intensity (default 0.75)
    alpha : float
        Blend weight: α=1 uses only Bayesian, α=0 uses only JS

    Returns
    -------
    pd.Series
        Combined shrunk expected returns
    """
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0,1], got {alpha}")

    mu_js = james_stein_shrinkage(mu, sigma, T=T, target=0.0)
    mu_bayes = bayesian_shrinkage(mu, prior=prior, gamma=gamma)

    mu_combined = alpha * mu_bayes + (1 - alpha) * mu_js

    return mu_combined


def shrink_mu_pipeline(
    returns: pd.DataFrame,
    *,
    estimator: Callable[[pd.DataFrame], pd.Series] | None = None,
    gamma: float = 0.75,
    alpha: float = 0.5,
    prior: pd.Series | float = 0.0,
) -> pd.Series:
    """Full pipeline: estimate μ robustly, then shrink.

    Convenient wrapper for common workflow.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical returns (rows = time, cols = assets)
    estimator : callable, optional
        Function to estimate μ (default: Huber mean)
        Should accept DataFrame and return Series
    gamma : float
        Bayesian shrinkage intensity
    alpha : float
        Blend weight (Bayesian vs JS)
    prior : pd.Series or float
        Prior for Bayesian shrinkage

    Returns
    -------
    pd.Series
        Shrunk expected returns (annualized)
    """
    from .cov import ledoit_wolf_shrinkage
    from .mu import huber_mean

    # Estimate μ robustly
    if estimator is None:
        mu_raw = huber_mean(returns, c=1.5)[0] * 252
    else:
        mu_raw = estimator(returns)

    # Estimate Σ
    sigma, _ = ledoit_wolf_shrinkage(returns)
    sigma_annual = sigma * 252

    # Apply combined shrinkage
    mu_shrunk = combined_shrinkage(
        mu_raw,
        sigma_annual,
        T=len(returns),
        prior=prior,
        gamma=gamma,
        alpha=alpha,
    )

    return mu_shrunk
