#!/usr/bin/env python3
"""Validate all YAML configuration files.

This script is a thin wrapper around ``arara_quant.reports.validators``.
"""

from __future__ import annotations

import sys

from arara_quant.config import get_settings
from arara_quant.reports.validators import validate_configs


def main() -> int:
    settings = get_settings()
    result = validate_configs(settings)

    errors_by_path = {path: message for path, message in result.errors}
    for config_file in result.validated:
        if config_file in errors_by_path:
            print(f"✗ {config_file.name}: {errors_by_path[config_file]}", file=sys.stderr)
        else:
            print(f"✓ {config_file.name}")

    return 1 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
