"""Tests for pipeline orchestrator."""

from __future__ import annotations

import stat
from unittest.mock import patch

import pytest
from arara_quant.pipeline.orchestrator import (
    PipelineError,
    run_full_pipeline,
    validate_write_permissions,
)


class TestValidateWritePermissions:
    """Tests for write permission validation."""

    def test_validate_write_permissions_creates_directory(self, tmp_path):
        """Verify that missing directory is created."""
        output_dir = tmp_path / "new_reports"
        assert not output_dir.exists()

        validate_write_permissions(output_dir)

        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_validate_write_permissions_succeeds_on_existing_dir(self, tmp_path):
        """Verify no error on existing writable directory."""
        # Should not raise
        validate_write_permissions(tmp_path)

    def test_validate_write_permissions_fails_on_readonly_dir(self, tmp_path):
        """Verify PipelineError on read-only directory."""
        # Make directory read-only
        tmp_path.chmod(stat.S_IRUSR | stat.S_IXUSR)

        try:
            with pytest.raises(PipelineError, match="No write permission"):
                validate_write_permissions(tmp_path)
        finally:
            # Restore permissions for cleanup
            tmp_path.chmod(stat.S_IRWXU)


class TestRunFullPipeline:
    """Tests for full pipeline orchestration."""

    @patch("arara_quant.pipeline.orchestrator.download_and_prepare_data")
    @patch("arara_quant.pipeline.orchestrator.estimate_parameters")
    @patch("arara_quant.pipeline.orchestrator.optimize_portfolio")
    @patch("arara_quant.pipeline.orchestrator.run_backtest")
    def test_run_full_pipeline_executes_all_stages(
        self,
        mock_backtest,
        mock_optimize,
        mock_estimate,
        mock_data,
        tmp_path,
    ):
        """Verify all 4 stages are executed in order."""
        # Setup mocks
        mock_data.return_value = {
            "status": "completed",
            "n_assets": 10,
            "n_days": 100,
            "returns_file": "returns_cache.parquet",
        }
        mock_estimate.return_value = {
            "status": "completed",
            "shrinkage": 0.2,
            "window_used": 100,
        }
        mock_optimize.return_value = {
            "status": "completed",
            "n_assets": 5,
            "sharpe": 1.5,
        }
        mock_backtest.return_value = {
            "status": "completed",
            "metrics": {"sharpe_ratio": 1.2},
        }

        # Execute
        result = run_full_pipeline(
            config_path="test_config.yaml",
            output_dir=tmp_path,
        )

        # Verify
        assert result["status"] == "completed"
        assert "metadata" in result
        assert "stages" in result
        assert "data" in result["stages"]
        assert "estimation" in result["stages"]
        assert "optimization" in result["stages"]
        assert "backtest" in result["stages"]

        # Check call order
        assert mock_data.called
        mock_estimate.assert_called_once()
        assert (
            mock_estimate.call_args.kwargs.get("returns_file")
            == "returns_cache.parquet"
        )
        assert mock_optimize.called
        assert mock_backtest.called

    @patch("arara_quant.pipeline.orchestrator.download_and_prepare_data")
    @patch("arara_quant.pipeline.orchestrator.estimate_parameters")
    @patch("arara_quant.pipeline.orchestrator.optimize_portfolio")
    def test_run_full_pipeline_skips_backtest_when_requested(
        self,
        mock_optimize,
        mock_estimate,
        mock_data,
        tmp_path,
    ):
        """Verify backtest can be skipped."""
        mock_data.return_value = {
            "status": "completed",
            "n_assets": 10,
            "n_days": 100,
            "returns_file": "returns_cache.parquet",
        }
        mock_estimate.return_value = {
            "status": "completed",
            "shrinkage": 0.2,
            "window_used": 100,
        }
        mock_optimize.return_value = {
            "status": "completed",
            "n_assets": 5,
            "sharpe": 1.5,
        }

        result = run_full_pipeline(
            config_path="test_config.yaml",
            skip_backtest=True,
            output_dir=tmp_path,
        )

        assert result["status"] == "completed"
        assert result["stages"]["backtest"]["status"] == "skipped"

    @patch("arara_quant.pipeline.orchestrator.download_and_prepare_data")
    @patch("arara_quant.pipeline.orchestrator.estimate_parameters")
    def test_run_full_pipeline_handles_optimization_failure(
        self,
        mock_estimate,
        mock_data,
        tmp_path,
    ):
        """Verify graceful handling of optimization failure."""
        mock_data.return_value = {
            "status": "completed",
            "n_assets": 10,
            "n_days": 100,
            "returns_file": "returns_cache.parquet",
        }
        mock_estimate.return_value = {
            "status": "completed",
            "shrinkage": 0.2,
            "window_used": 100,
        }

        # Make optimization fail
        with patch(
            "arara_quant.pipeline.orchestrator.optimize_portfolio",
            side_effect=ValueError("Optimization failed"),
        ):
            with pytest.raises(PipelineError, match="Pipeline execution failed"):
                run_full_pipeline(
                    config_path="test_config.yaml",
                    output_dir=tmp_path,
                )

    @patch("arara_quant.pipeline.orchestrator.download_and_prepare_data")
    @patch("arara_quant.pipeline.orchestrator.estimate_parameters")
    @patch("arara_quant.pipeline.orchestrator.optimize_portfolio")
    def test_run_full_pipeline_returns_expected_structure(
        self,
        mock_optimize,
        mock_estimate,
        mock_data,
        tmp_path,
    ):
        """Verify result dictionary has expected structure."""
        mock_data.return_value = {
            "status": "completed",
            "n_assets": 10,
            "n_days": 100,
            "returns_file": "returns_cache.parquet",
        }
        mock_estimate.return_value = {
            "status": "completed",
            "shrinkage": 0.2,
            "window_used": 100,
        }
        mock_optimize.return_value = {
            "status": "completed",
            "n_assets": 5,
            "sharpe": 1.5,
        }

        result = run_full_pipeline(
            config_path="test_config.yaml",
            skip_backtest=True,
            output_dir=tmp_path,
        )

        # Verify top-level structure
        assert "status" in result
        assert "metadata" in result
        assert "stages" in result
        assert "duration_seconds" in result

        # Verify metadata structure
        metadata = result["metadata"]
        assert "timestamp" in metadata
        assert "config_path" in metadata
        assert "start_date" in metadata
        assert "end_date" in metadata

        # Verify each stage has duration
        for stage_name in ["data", "estimation", "optimization"]:
            assert "duration_seconds" in result["stages"][stage_name]

    @patch("arara_quant.pipeline.orchestrator.download_and_prepare_data")
    def test_run_full_pipeline_passes_parameters_correctly(
        self,
        mock_data,
        tmp_path,
    ):
        """Verify parameters are passed to data stage correctly."""
        mock_data.return_value = {
            "status": "completed",
            "n_assets": 10,
            "n_days": 100,
            "returns_file": "returns_cache.parquet",
        }

        with patch("arara_quant.pipeline.orchestrator.estimate_parameters"):
            with patch("arara_quant.pipeline.orchestrator.optimize_portfolio"):
                run_full_pipeline(
                    config_path="test_config.yaml",
                    start="2020-01-01",
                    end="2023-12-31",
                    skip_download=True,
                    skip_backtest=True,
                    output_dir=tmp_path,
                )

        # Verify data stage was called with correct params
        call_kwargs = mock_data.call_args.kwargs
        assert call_kwargs["start"] == "2020-01-01"
        assert call_kwargs["end"] == "2023-12-31"
        assert call_kwargs["force_download"] is False  # skip_download=True

    def test_run_full_pipeline_validates_permissions_before_starting(self, tmp_path):
        """Verify write permissions are checked before any stage."""
        # Make directory read-only
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)

        try:
            # Should fail before calling any pipeline stages
            with pytest.raises(PipelineError, match="No write permission"):
                run_full_pipeline(
                    config_path="test_config.yaml",
                    output_dir=readonly_dir,
                )
        finally:
            # Restore permissions for cleanup
            readonly_dir.chmod(stat.S_IRWXU)
