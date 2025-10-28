"""Tests for result serialization (JSON + Markdown)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from itau_quant.reports.serializer import generate_markdown, save_results


class TestGenerateMarkdown:
    """Tests for Markdown generation."""

    def test_generate_markdown_includes_header(self):
        """Verify Markdown includes expected header."""
        results = {
            "status": "completed",
            "metadata": {
                "timestamp": "2025-10-28T12:00:00",
                "config_path": "test.yaml",
            },
            "stages": {},
        }

        md = generate_markdown(results)

        assert "# Execução Pipeline ARARA" in md
        assert "2025-10-28T12:00:00" in md
        assert "test.yaml" in md

    def test_generate_markdown_includes_optimization_metrics(self):
        """Verify optimization metrics are included."""
        results = {
            "status": "completed",
            "metadata": {"timestamp": "2025-10-28T12:00:00"},
            "stages": {
                "optimization": {
                    "status": "completed",
                    "n_assets": 10,
                    "sharpe": 1.5,
                    "expected_return": 0.15,
                    "volatility": 0.10,
                    "risk_aversion": 4.0,
                }
            },
        }

        md = generate_markdown(results)

        assert "Otimização" in md
        assert "10" in md  # n_assets
        assert "1.50" in md  # sharpe

    def test_generate_markdown_includes_stage_status(self):
        """Verify stage status indicators are present."""
        results = {
            "status": "completed",
            "metadata": {"timestamp": "2025-10-28T12:00:00"},
            "stages": {
                "data": {"status": "completed", "duration_seconds": 1.0},
                "estimation": {"status": "completed", "duration_seconds": 0.5},
                "backtest": {"status": "skipped", "duration_seconds": 0.0},
            },
        }

        md = generate_markdown(results)

        assert "✅" in md  # completed emoji
        assert "⏭️" in md  # skipped emoji
        assert "Data: completed" in md

    def test_generate_markdown_includes_backtest_metrics(self):
        """Verify backtest metrics in executive summary."""
        results = {
            "status": "completed",
            "metadata": {"timestamp": "2025-10-28T12:00:00"},
            "stages": {
                "backtest": {
                    "status": "completed",
                    "metrics": {
                        "total_return": 0.25,
                        "sharpe_ratio": 1.8,
                        "max_drawdown": -0.12,
                    },
                }
            },
        }

        md = generate_markdown(results)

        assert "Resumo Executivo" in md
        assert "25.00%" in md  # total return
        assert "1.800" in md  # sharpe


class TestSaveResults:
    """Tests for save_results function."""

    def test_save_results_creates_json_and_markdown(self, tmp_path):
        """Verify both JSON and Markdown files are created."""
        results = {
            "status": "completed",
            "metadata": {"timestamp": "2025-10-28T12:00:00"},
            "stages": {},
        }

        json_path, md_path = save_results(results, tmp_path)

        assert json_path.exists()
        assert md_path.exists()
        assert json_path.suffix == ".json"
        assert md_path.suffix == ".md"

    def test_save_results_json_is_valid(self, tmp_path):
        """Verify saved JSON can be parsed."""
        results = {
            "status": "completed",
            "metadata": {"timestamp": "2025-10-28T12:00:00"},
            "stages": {"data": {"status": "completed"}},
        }

        json_path, _ = save_results(results, tmp_path)

        # Verify JSON is valid
        loaded = json.loads(json_path.read_text())
        assert loaded["status"] == "completed"
        assert loaded["stages"]["data"]["status"] == "completed"

    def test_save_results_creates_symlinks(self, tmp_path):
        """Verify latest_run symlinks are created."""
        results = {
            "status": "completed",
            "metadata": {"timestamp": "2025-10-28T12:00:00"},
            "stages": {},
        }

        save_results(results, tmp_path, create_symlink=True)

        latest_json = tmp_path / "latest_run.json"
        latest_md = tmp_path / "latest_run.md"

        assert latest_json.exists()
        assert latest_md.exists()
        assert latest_json.is_symlink()
        assert latest_md.is_symlink()

    def test_save_results_skips_symlinks_when_requested(self, tmp_path):
        """Verify symlinks are not created when create_symlink=False."""
        results = {
            "status": "completed",
            "metadata": {"timestamp": "2025-10-28T12:00:00"},
            "stages": {},
        }

        save_results(results, tmp_path, create_symlink=False)

        latest_json = tmp_path / "latest_run.json"
        latest_md = tmp_path / "latest_run.md"

        assert not latest_json.exists()
        assert not latest_md.exists()

    def test_save_results_updates_symlinks_on_multiple_runs(self, tmp_path):
        """Verify symlinks point to latest run after multiple saves."""
        results1 = {
            "status": "completed",
            "metadata": {"timestamp": "2025-10-28T12:00:00"},
            "stages": {},
        }
        results2 = {
            "status": "completed",
            "metadata": {"timestamp": "2025-10-28T13:00:00"},
            "stages": {},
        }

        save_results(results1, tmp_path)
        json_path2, _ = save_results(results2, tmp_path)

        latest_json = tmp_path / "latest_run.json"

        # Symlink should point to second run
        assert latest_json.resolve() == json_path2

    def test_save_results_sanitizes_timestamp_in_filename(self, tmp_path):
        """Verify colons are replaced in filename."""
        results = {
            "status": "completed",
            "metadata": {"timestamp": "2025-10-28T12:30:45.123456"},
            "stages": {},
        }

        json_path, _ = save_results(results, tmp_path)

        # Filename should not contain colons
        assert ":" not in json_path.name
        assert "run_2025-10-28T12-30-45" in json_path.name

    def test_save_results_creates_output_dir_if_missing(self, tmp_path):
        """Verify output directory is created if it doesn't exist."""
        output_dir = tmp_path / "nested" / "reports"
        assert not output_dir.exists()

        results = {
            "status": "completed",
            "metadata": {"timestamp": "2025-10-28T12:00:00"},
            "stages": {},
        }

        save_results(results, output_dir)

        assert output_dir.exists()
