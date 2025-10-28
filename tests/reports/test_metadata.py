"""Tests for metadata collection utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from itau_quant.reports.metadata import get_git_commit, hash_file


class TestGetGitCommit:
    """Tests for git commit extraction."""

    def test_get_git_commit_returns_string(self):
        """Verify function returns a string."""
        commit = get_git_commit()
        assert isinstance(commit, str)

    def test_get_git_commit_returns_valid_hash_or_unknown(self):
        """Verify return value is either a valid hash or 'unknown'."""
        commit = get_git_commit()
        # Either a 40-char hex string (full hash) or "unknown"
        if commit != "unknown":
            assert len(commit) == 40
            assert all(c in "0123456789abcdef" for c in commit.lower())


class TestHashFile:
    """Tests for file hashing."""

    def test_hash_file_returns_consistent_hash(self, tmp_path):
        """Verify same content produces same hash."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("hello world")

        hash1 = hash_file(file_path)
        hash2 = hash_file(file_path)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex digest

    def test_hash_file_different_content_different_hash(self, tmp_path):
        """Verify different content produces different hash."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"

        file1.write_text("hello")
        file2.write_text("world")

        hash1 = hash_file(file1)
        hash2 = hash_file(file2)

        assert hash1 != hash2

    def test_hash_file_raises_on_missing_file(self, tmp_path):
        """Verify FileNotFoundError on missing file."""
        missing = tmp_path / "nonexistent.txt"

        with pytest.raises(FileNotFoundError, match="File not found"):
            hash_file(missing)

    def test_hash_file_known_hash(self, tmp_path):
        """Verify hash matches known SHA256."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("test")

        # Known SHA256 of "test"
        expected = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
        assert hash_file(file_path) == expected
