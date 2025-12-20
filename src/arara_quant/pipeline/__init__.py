"""Pipeline orchestration for data, estimation, and optimization stages.

This package is intentionally *lazy* to keep import-time dependencies minimal.
The data pipeline depends on optional runtime packages (e.g. ``yfinance``),
so top-level imports are resolved on-demand via :func:`__getattr__`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "download_and_prepare_data",
    "estimate_parameters",
    "optimize_portfolio",
    "run_full_pipeline",
    "PipelineError",
    "validate_write_permissions",
]

if TYPE_CHECKING:  # pragma: no cover
    from .data import download_and_prepare_data as download_and_prepare_data
    from .estimation import estimate_parameters as estimate_parameters
    from .optimization import optimize_portfolio as optimize_portfolio
    from .orchestrator import PipelineError as PipelineError
    from .orchestrator import run_full_pipeline as run_full_pipeline
    from .orchestrator import validate_write_permissions as validate_write_permissions


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name == "download_and_prepare_data":
        from .data import download_and_prepare_data

        return download_and_prepare_data
    if name == "estimate_parameters":
        from .estimation import estimate_parameters

        return estimate_parameters
    if name == "optimize_portfolio":
        from .optimization import optimize_portfolio

        return optimize_portfolio
    if name in {"PipelineError", "run_full_pipeline", "validate_write_permissions"}:
        from .orchestrator import (
            PipelineError,
            run_full_pipeline,
            validate_write_permissions,
        )

        return {
            "PipelineError": PipelineError,
            "run_full_pipeline": run_full_pipeline,
            "validate_write_permissions": validate_write_permissions,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover
    return sorted({*globals(), *__all__})
