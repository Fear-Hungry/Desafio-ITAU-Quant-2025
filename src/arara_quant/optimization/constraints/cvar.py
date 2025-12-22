"""CVaR (Conditional Value-at-Risk) constraint for portfolio optimization.

Implements CVaR as a convex linear constraint using auxiliary variables.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd

from arara_quant.risk.cvar import add_cvar_constraint, validate_cvar_inputs

__all__ = ["build_cvar_constraint"]


def build_cvar_constraint(
    weights_var: cp.Variable,
    returns: pd.DataFrame | np.ndarray,
    *,
    alpha: float = 0.95,
    cvar_limit: float = 0.10,
) -> list[cp.Constraint]:
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
        Confidence level (default 0.95 for 5% tail CVaR)
    cvar_limit : float
        Maximum allowed CVaR (as positive number, e.g., 0.10 for -10% tail loss)

    Returns
    -------
    list of cp.Constraint
        CVXPY constraints to add to the problem
    """
    if isinstance(returns, pd.DataFrame):
        returns_matrix = returns.to_numpy(dtype=float)
    else:
        returns_matrix = np.asarray(returns, dtype=float)

    validate_cvar_inputs(returns_matrix, alpha)
    _, n_assets = returns_matrix.shape

    if len(weights_var) != n_assets:
        raise ValueError(
            "weights_var length "
            f"({len(weights_var)}) must match returns columns ({n_assets})"
        )

    _, _, constraints = add_cvar_constraint(
        returns_matrix,
        weights_var,
        alpha,
        max_cvar=float(cvar_limit),
    )
    return constraints
