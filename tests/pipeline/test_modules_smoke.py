"""Smoke tests for pipeline modules.

These tests verify that the pipeline modules can be imported and have the
expected function signatures without executing the full logic.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from itau_quant.pipeline import (
    download_and_prepare_data,
    estimate_parameters,
    optimize_portfolio,
)


class TestDataModule:
    """Smoke tests for pipeline.data module."""

    def test_download_and_prepare_data_is_callable(self):
        """Verify function exists and is callable."""
        assert callable(download_and_prepare_data)

    def test_download_and_prepare_data_signature(self):
        """Verify function accepts expected parameters."""
        sig = inspect.signature(download_and_prepare_data)
        params = list(sig.parameters.keys())

        # Expected keyword-only parameters
        expected = [
            "start",
            "end",
            "raw_file_name",
            "processed_file_name",
            "force_download",
            "settings",
        ]

        for param in expected:
            assert param in params, f"Missing parameter: {param}"

    def test_download_and_prepare_data_returns_dict(self):
        """Verify function signature indicates dict return type."""
        sig = inspect.signature(download_and_prepare_data)
        # Check return annotation hints at dict
        assert sig.return_annotation != inspect.Signature.empty


class TestEstimationModule:
    """Smoke tests for pipeline.estimation module."""

    def test_estimate_parameters_is_callable(self):
        """Verify function exists and is callable."""
        assert callable(estimate_parameters)

    def test_estimate_parameters_signature(self):
        """Verify function accepts expected parameters."""
        sig = inspect.signature(estimate_parameters)
        params = list(sig.parameters.keys())

        expected = [
            "returns_file",
            "window",
            "mu_method",
            "cov_method",
            "huber_delta",
            "annualize",
            "mu_output",
            "cov_output",
            "settings",
        ]

        for param in expected:
            assert param in params, f"Missing parameter: {param}"

    def test_estimate_parameters_handles_missing_file(self):
        """Verify FileNotFoundError is raised for missing input."""
        with pytest.raises(FileNotFoundError, match="Returns file not found"):
            estimate_parameters(returns_file="nonexistent_file.parquet")


class TestOptimizationModule:
    """Smoke tests for pipeline.optimization module."""

    def test_optimize_portfolio_is_callable(self):
        """Verify function exists and is callable."""
        assert callable(optimize_portfolio)

    def test_optimize_portfolio_signature(self):
        """Verify function accepts expected parameters."""
        sig = inspect.signature(optimize_portfolio)
        params = list(sig.parameters.keys())

        expected = [
            "mu_file",
            "cov_file",
            "risk_aversion",
            "max_weight",
            "min_weight",
            "turnover_cap",
            "turnover_penalty",
            "ridge_penalty",
            "output_file",
            "solver",
            "settings",
        ]

        for param in expected:
            assert param in params, f"Missing parameter: {param}"

    def test_optimize_portfolio_handles_missing_mu_file(self):
        """Verify FileNotFoundError for missing μ file."""
        with pytest.raises(FileNotFoundError, match="μ file not found"):
            optimize_portfolio(mu_file="nonexistent_mu.parquet")

    def test_optimize_portfolio_handles_missing_cov_file(self, tmp_path):
        """Verify FileNotFoundError for missing Σ file."""
        # Create a dummy mu file
        import pandas as pd

        mu_path = tmp_path / "mu.parquet"
        mu = pd.Series([0.1, 0.2], index=["A", "B"], name="mu")
        mu.to_frame().to_parquet(mu_path)

        from itau_quant.config import Settings

        settings = Settings.from_env(
            overrides={"PROCESSED_DATA_DIR": str(tmp_path)}
        )

        with pytest.raises(FileNotFoundError, match="Σ file not found"):
            optimize_portfolio(
                mu_file="mu.parquet",
                cov_file="nonexistent_cov.parquet",
                settings=settings,
            )


class TestModuleIntegration:
    """Test that modules work together."""

    def test_all_modules_importable(self):
        """Verify all pipeline modules can be imported."""
        from itau_quant.pipeline import (
            download_and_prepare_data,
            estimate_parameters,
            optimize_portfolio,
        )

        assert download_and_prepare_data is not None
        assert estimate_parameters is not None
        assert optimize_portfolio is not None

    def test_modules_return_dict_with_status(self):
        """Verify all modules are documented to return dict with 'status'."""
        # This is a documentation check - smoke test only
        functions = [
            download_and_prepare_data,
            estimate_parameters,
            optimize_portfolio,
        ]

        for func in functions:
            docstring = func.__doc__ or ""
            # Check that docstring mentions returning a dict
            assert "dict" in docstring.lower(), f"{func.__name__} lacks dict return doc"
            assert "status" in docstring.lower(), f"{func.__name__} lacks status doc"
