"""Facilita import de estimadores de expectativa e risco."""

from .mu_robust import (
    bayesian_shrinkage,
    combined_shrinkage,
    james_stein_shrinkage,
    shrink_mu_pipeline,
)

__all__ = [
    "bayesian_shrinkage",
    "combined_shrinkage",
    "james_stein_shrinkage",
    "shrink_mu_pipeline",
]
