"""Reports and result serialization module.

This module provides utilities for persisting pipeline results in structured
formats (JSON, Markdown) and collecting execution metadata (git commit, config hash).
"""

from __future__ import annotations

__all__ = [
    "get_git_commit",
    "hash_file",
    "save_results",
    "generate_markdown",
    "ConfigValidationResult",
    "TurnoverStats",
    "update_readme_turnover_stats",
    "validate_configs",
]

from .metadata import get_git_commit, hash_file
from .serializer import generate_markdown, save_results
from .validators import ConfigValidationResult, validate_configs
from .generators import TurnoverStats, update_readme_turnover_stats
