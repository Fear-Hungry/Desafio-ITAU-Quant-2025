"""Full pipeline orchestrator.

This module coordinates the execution of all pipeline stages:
1. Data acquisition and preprocessing
2. Parameter estimation (μ, Σ)
3. Portfolio optimization
4. Backtesting (optional)

The orchestrator validates write permissions before starting, handles errors
gracefully, and returns a structured result dictionary suitable for logging
and persistence.
"""

from __future__ import annotations

import os
import stat
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from arara_quant.backtesting import run_backtest
from arara_quant.config import Settings
from arara_quant.pipeline.data import download_and_prepare_data
from arara_quant.pipeline.estimation import estimate_parameters
from arara_quant.pipeline.optimization import optimize_portfolio
from arara_quant.reports.metadata import get_git_commit, hash_file
from arara_quant.reports.serializer import save_results
from arara_quant.utils.logging_config import get_logger

__all__ = ["PipelineError", "validate_write_permissions", "run_full_pipeline"]

logger = get_logger(__name__)


class PipelineError(Exception):
    """Raised when pipeline execution fails."""

    pass


def validate_write_permissions(output_dir: Path) -> None:
    """Validate that we can write to the output directory.

    This function attempts to create the directory and write a test file
    to ensure the pipeline has necessary permissions before starting
    potentially long-running operations.

    Args:
        output_dir: Directory to validate write access

    Raises:
        PipelineError: If directory cannot be created or written to

    Examples:
        >>> validate_write_permissions(Path("outputs/reports"))
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        raise PipelineError(f"Cannot create directory {output_dir}: {e}") from e

    if not _can_write_directory(output_dir):
        raise PipelineError(
            f"No write permission to {output_dir}: insufficient privileges"
        )

    test_file = output_dir / ".write_test"
    try:
        test_file.touch()
        test_file.unlink()
    except (OSError, PermissionError) as e:
        raise PipelineError(f"No write permission to {output_dir}: {e}") from e


def _can_write_directory(path: Path) -> bool:
    """Return True if the current process should be able to create files in path."""
    try:
        st = path.stat()
    except FileNotFoundError:
        return True

    mode = st.st_mode
    uid = os.geteuid()
    gids = {os.getegid(), *os.getgroups()}

    if uid == 0:
        return bool(mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))

    if uid == st.st_uid:
        return bool(mode & stat.S_IWUSR) and bool(mode & stat.S_IXUSR)

    if st.st_gid in gids:
        return bool(mode & stat.S_IWGRP) and bool(mode & stat.S_IXGRP)

    return bool(mode & stat.S_IWOTH) and bool(mode & stat.S_IXOTH)


def run_full_pipeline(
    config_path: str | Path,
    *,
    start: str | None = None,
    end: str | None = None,
    skip_download: bool = False,
    skip_backtest: bool = False,
    output_dir: str | Path = "outputs/reports",
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Execute the complete portfolio construction pipeline.

    This function orchestrates all stages of the pipeline in sequence:
    1. Data: Download prices and compute returns
    2. Estimation: Compute expected returns (μ) and covariance (Σ)
    3. Optimization: Solve for optimal portfolio weights
    4. Backtesting: Run walk-forward validation (optional)

    The function validates write permissions before starting and handles
    errors gracefully, returning a structured result dictionary.

    Args:
        config_path: Path to YAML configuration file
        start: Optional start date in ISO format (YYYY-MM-DD)
        end: Optional end date in ISO format (YYYY-MM-DD)
        skip_download: If True, reuse cached data (faster for testing)
        skip_backtest: If True, skip the backtesting stage
        output_dir: Directory for saving reports and results
        settings: Settings object (uses default if None)

    Returns:
        dict containing:
            - status: "completed" or "failed"
            - metadata: Execution metadata (timestamp, config, etc)
            - stages: Results from each pipeline stage
            - duration_seconds: Total execution time
            - error: Error message (if failed)

    Raises:
        PipelineError: If pipeline execution fails

    Examples:
        >>> result = run_full_pipeline(
        ...     config_path="configs/optimization/optimizer_example.yaml",
        ...     skip_download=True
        ... )
        >>> print(result['status'])
        completed
    """
    settings = settings or Settings.from_env()
    output_path = Path(output_dir)

    # Validate write permissions early to fail fast
    logger.info("Validating write permissions to %s", output_path)
    validate_write_permissions(output_path)

    start_time = time.perf_counter()
    logger.info("Starting full pipeline execution")

    # Collect metadata including git commit and config hash
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "config_path": str(config_path),
        "start_date": start,
        "end_date": end,
        "git_commit": get_git_commit(),
    }

    # Add config hash if file exists
    config_file = Path(config_path)
    if config_file.exists():
        try:
            metadata["config_hash"] = hash_file(config_file)
        except (OSError, FileNotFoundError):
            logger.warning("Could not hash config file: %s", config_path)

    results: dict[str, Any] = {
        "metadata": metadata,
        "stages": {},
    }

    try:
        # Stage 1: Data acquisition and preprocessing
        logger.info("Stage 1/4: Data acquisition and preprocessing")
        stage_start = time.perf_counter()

        data_result = download_and_prepare_data(
            start=start,
            end=end,
            force_download=not skip_download,
            settings=settings,
        )
        data_result["duration_seconds"] = time.perf_counter() - stage_start
        results["stages"]["data"] = data_result

        logger.info(
            "Stage 1 completed: %d assets, %d days",
            data_result["n_assets"],
            data_result["n_days"],
        )

        # Stage 2: Parameter estimation
        logger.info("Stage 2/4: Parameter estimation (μ, Σ)")
        stage_start = time.perf_counter()

        estimation_kwargs: dict[str, Any] = {"settings": settings}
        estimation_kwargs["config_path"] = config_path
        returns_file = data_result.get("returns_file")
        if returns_file:
            logger.info("Stage 2 will use returns artefact %s", returns_file)
            estimation_kwargs["returns_file"] = returns_file

        estimation_result = estimate_parameters(**estimation_kwargs)
        estimation_result["duration_seconds"] = time.perf_counter() - stage_start
        results["stages"]["estimation"] = estimation_result

        logger.info(
            "Stage 2 completed: shrinkage=%.3f, window=%d",
            estimation_result.get("shrinkage") or 0.0,
            estimation_result["window_used"],
        )

        # Stage 3: Portfolio optimization
        logger.info("Stage 3/4: Portfolio optimization")
        stage_start = time.perf_counter()

        optimization_result = optimize_portfolio(
            config_path=config_path, settings=settings
        )
        optimization_result["duration_seconds"] = time.perf_counter() - stage_start
        results["stages"]["optimization"] = optimization_result

        logger.info(
            "Stage 3 completed: %d assets selected, Sharpe=%.2f",
            optimization_result["n_assets"],
            optimization_result["sharpe"],
        )

        # Stage 4: Backtesting (optional)
        if not skip_backtest:
            logger.info("Stage 4/4: Walk-forward backtesting")
            stage_start = time.perf_counter()

            backtest_result = run_backtest(
                config_path=str(config_path),
                dry_run=False,
            )

            # Convert BacktestResult object to dict
            if hasattr(backtest_result, "to_dict"):
                backtest_dict = backtest_result.to_dict(include_timeseries=False)  # type: ignore[call-arg]
            elif isinstance(backtest_result, dict):
                backtest_dict = dict(backtest_result)
            else:
                raise PipelineError(
                    f"Unexpected backtest result type: {type(backtest_result)!r}"
                )

            # Extract relevant metrics from backtest
            backtest_summary = {
                "status": backtest_dict.get("status", "completed"),
                "duration_seconds": time.perf_counter() - stage_start,
            }

            if "metrics" in backtest_dict:
                backtest_summary["metrics"] = backtest_dict["metrics"]

            if "notes" in backtest_dict:
                backtest_summary["notes"] = backtest_dict["notes"]
                backtest_summary["n_notes"] = len(backtest_dict["notes"])

            results["stages"]["backtest"] = backtest_summary

            logger.info(
                "Stage 4 completed: status=%s, %d notes",
                backtest_summary["status"],
                backtest_summary.get("n_notes", 0),
            )
        else:
            logger.info("Stage 4/4: Backtesting (skipped)")
            results["stages"]["backtest"] = {
                "status": "skipped",
                "duration_seconds": 0.0,
            }

        # Success
        results["status"] = "completed"
        results["duration_seconds"] = time.perf_counter() - start_time

        logger.info(
            "Pipeline completed successfully in %.1f seconds",
            results["duration_seconds"],
        )

        # Save results to disk (JSON + Markdown)
        try:
            json_path, md_path = save_results(results, output_path)
            logger.info("Results saved to %s", json_path)
            logger.info("Markdown report saved to %s", md_path)
            logger.info("Latest run symlinks updated")
        except Exception as e:
            logger.warning("Failed to save results: %s", e)

    except Exception as e:
        # Failure - capture error details
        results["status"] = "failed"
        results["error"] = str(e)
        results["duration_seconds"] = time.perf_counter() - start_time

        logger.error(
            "Pipeline failed after %.1f seconds: %s",
            results["duration_seconds"],
            e,
            exc_info=True,
        )

        raise PipelineError(f"Pipeline execution failed: {e}") from e

    return results
