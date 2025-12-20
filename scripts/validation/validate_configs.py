#!/usr/bin/env python3
"""Validate all YAML configuration files."""

import sys

from arara_quant.config import (
    PortfolioConfig,
    UniverseConfig,
    get_settings,
    load_config,
)


def main():
    """Validate all configuration files."""
    configs_dir = get_settings().configs_dir
    errors = []

    # Validate universe configs
    for config_file in configs_dir.glob("universe_*.yaml"):
        try:
            load_config(str(config_file), UniverseConfig)
            print(f"✓ {config_file.name}")
        except Exception as e:
            errors.append(f"✗ {config_file.name}: {e}")

    # Validate portfolio configs
    for config_file in configs_dir.glob("portfolio_*.yaml"):
        try:
            load_config(str(config_file), PortfolioConfig)
            print(f"✓ {config_file.name}")
        except Exception as e:
            errors.append(f"✗ {config_file.name}: {e}")

    # Print errors and exit
    for error in errors:
        print(error, file=sys.stderr)

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
