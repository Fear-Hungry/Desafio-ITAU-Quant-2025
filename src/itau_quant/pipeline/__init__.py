"""Pipeline orchestration for data, estimation, and optimization stages.

This module provides reusable functions for each stage of the portfolio
construction pipeline, allowing both scripted execution and programmatic use.
"""

from __future__ import annotations

__all__ = [
    "download_and_prepare_data",
    "estimate_parameters",
    "optimize_portfolio",
]

from .data import download_and_prepare_data
from .estimation import estimate_parameters
from .optimization import optimize_portfolio
