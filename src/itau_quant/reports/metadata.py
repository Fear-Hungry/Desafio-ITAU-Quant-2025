"""Collect run metadata (git commit, config hash, etc).

This module provides utilities to capture execution context metadata
such as git commit hash and configuration file checksums for reproducibility.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

__all__ = ["get_git_commit", "hash_file"]


def get_git_commit() -> str:
    """Get current git commit hash.

    Returns the full SHA-1 hash of the current HEAD commit. Returns "unknown"
    if git is not available or the command fails (e.g., not in a git repo).

    Returns:
        Git commit hash (40 chars) or "unknown"

    Examples:
        >>> commit = get_git_commit()
        >>> len(commit) in {7, 40}  # Short or full hash
        True
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


def hash_file(path: Path) -> str:
    """Compute SHA256 hash of a file.

    Args:
        path: Path to file to hash

    Returns:
        Hex digest of SHA256 hash (64 chars)

    Raises:
        FileNotFoundError: If file doesn't exist
        OSError: If file cannot be read

    Examples:
        >>> from pathlib import Path
        >>> path = Path("test.txt")
        >>> path.write_text("hello")
        5
        >>> hash_file(path)
        '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824'
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return hashlib.sha256(path.read_bytes()).hexdigest()
