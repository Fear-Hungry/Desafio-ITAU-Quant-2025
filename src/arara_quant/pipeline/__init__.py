"""Pipeline orchestration for data, estimation, and optimization stages.

This module provides reusable functions for each stage of the portfolio
construction pipeline, allowing both scripted execution and programmatic use.
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

if TYPE_CHECKING:
    from .data import download_and_prepare_data
    from .estimation import estimate_parameters
    from .optimization import optimize_portfolio
    from .orchestrator import PipelineError, run_full_pipeline, validate_write_permissions


def __getattr__(name: str) -> Any:  # pragma: no cover - import-time dispatch
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
        from .orchestrator import PipelineError, run_full_pipeline, validate_write_permissions

        return {
            "PipelineError": PipelineError,
            "run_full_pipeline": run_full_pipeline,
            "validate_write_permissions": validate_write_permissions,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - import-time dispatch
    return sorted([*globals().keys(), *__all__])
