"""CVaR (Conditional Value-at-Risk) constraint for portfolio optimization.

Implements CVaR as a convex SOCP constraint using auxiliary variables.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd

__all__ = ["build_cvar_constraint"]


def build_cvar_constraint(
    weights_var: cp.Variable,
    returns: pd.DataFrame | np.ndarray,
    *,
    alpha: float = 0.05,
    cvar_limit: float = 0.10,
) -> list:
    """Build CVaR constraint for portfolio optimization.

    CVaR_α(w) ≤ cvar_limit

    Uses the dual representation with auxiliary variables:
    - ν = VaR (Value-at-Risk at level α)
    - ξ_t = excess losses beyond VaR

    CVaR_α = ν + (1/(1-α)) * E[ξ]

    Parameters
    ----------
    weights_var : cp.Variable
        Portfolio weights variable (from CVXPY problem)
    returns : pd.DataFrame or np.ndarray
        Historical returns (rows = time, cols = assets)
        Used to compute scenario-based CVaR
    alpha : float
        Confidence level (default 0.05 for 95% CVaR)
    cvar_limit : float
        Maximum allowed CVaR (as positive number, e.g., 0.10 for -10% tail loss)

    Returns
    -------
    list of cp.Constraint
        CVXPY constraints to add to the problem
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0,1), got {alpha}")

    if isinstance(returns, pd.DataFrame):
        returns_matrix = returns.to_numpy(dtype=float)
    else:
        returns_matrix = np.asarray(returns, dtype=float)

    T, N = returns_matrix.shape

    if len(weights_var) != N:
        raise ValueError(
            f"weights_var length ({len(weights_var)}) must match returns columns ({N})"
        )

    # Auxiliary variables
    nu = cp.Variable()  # VaR
    xi = cp.Variable(T, nonneg=True)  # Excess losses

    constraints = []

    # CVaR formula: ν + (1/(1-α)) * (1/T) * Σ_t ξ_t ≤ cvar_limit
    # Note: CVaR is usually negative (loss), so we compare with positive limit
    cvar_expr = nu + (1.0 / ((1 - alpha) * T)) * cp.sum(xi)
    constraints.append(cvar_expr <= cvar_limit)

    # Link ξ_t to portfolio losses
    # ξ_t ≥ -r_t' w - ν  (loss at time t beyond VaR)
    portfolio_returns = returns_matrix @ weights_var  # Shape (T,)

    for t in range(T):
        constraints.append(xi[t] >= -portfolio_returns[t] - nu)

    return constraints
