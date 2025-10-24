"""Configuration loading and validation utilities.

This module provides functions to load YAML configuration files and validate them
against Pydantic schemas. It handles file resolution, parsing, and error reporting.

Example
-------
>>> from itau_quant.config.loader import load_config
>>> from itau_quant.config.schemas import PortfolioConfig
>>>
>>> config = load_config("configs/portfolio_arara_basic.yaml", PortfolioConfig)
>>> print(config.risk_aversion)
3.0
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypeVar, Type, Union, Optional

import yaml
from pydantic import BaseModel, ValidationError

__all__ = ["load_config", "save_config", "ConfigError"]

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ConfigError(Exception):
    """Raised when configuration loading or validation fails."""

    pass


def _resolve_config_path(file_path: Union[str, Path], project_root: Optional[Path] = None) -> Path:
    """Resolve configuration file path.

    Parameters
    ----------
    file_path : str or Path
        Path to configuration file (absolute or relative)
    project_root : Path, optional
        Project root directory. If None, auto-detect from this file's location.

    Returns
    -------
    Path
        Resolved absolute path

    Raises
    ------
    FileNotFoundError
        If file doesn't exist
    """
    path = Path(file_path)

    # If absolute and exists, return it
    if path.is_absolute() and path.exists():
        return path

    # If relative, try relative to project root
    if project_root is None:
        # Auto-detect project root (3 levels up from this file)
        project_root = Path(__file__).resolve().parents[3]

    resolved = project_root / path
    if resolved.exists():
        return resolved

    # Try relative to current working directory
    if path.exists():
        return path.resolve()

    raise FileNotFoundError(f"Config file not found: {file_path}")


def load_config(
    file_path: Union[str, Path],
    schema: Type[T],
    *,
    project_root: Optional[Path] = None,
    strict: bool = True,
) -> T:
    """Load and validate YAML configuration file.

    Parameters
    ----------
    file_path : str or Path
        Path to YAML configuration file
    schema : Type[BaseModel]
        Pydantic model class to validate against
    project_root : Path, optional
        Project root directory for path resolution
    strict : bool, default=True
        If True, raise exception on validation errors.
        If False, log warnings and return default instance.

    Returns
    -------
    BaseModel
        Validated configuration object

    Raises
    ------
    ConfigError
        If file loading or validation fails (only when strict=True)

    Examples
    --------
    >>> from itau_quant.config.schemas import UniverseConfig
    >>> config = load_config("configs/universe_arara.yaml", UniverseConfig)
    >>> print(config.name)
    'ARARA'
    """
    try:
        # Resolve file path
        resolved_path = _resolve_config_path(file_path, project_root)

        # Load YAML
        logger.debug(f"Loading config from: {resolved_path}")
        with open(resolved_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            raise ConfigError(f"Empty configuration file: {file_path}")

        # Validate against schema
        try:
            config = schema.model_validate(data)
            logger.info(f"Successfully loaded config: {resolved_path.name}")
            return config
        except ValidationError as e:
            error_msg = f"Configuration validation failed for {file_path}:\n{e}"
            if strict:
                raise ConfigError(error_msg) from e
            else:
                logger.warning(error_msg)
                logger.warning("Returning default configuration")
                return schema()

    except FileNotFoundError as e:
        if strict:
            raise ConfigError(f"Configuration file not found: {file_path}") from e
        else:
            logger.warning(f"Config file not found: {file_path}, using defaults")
            return schema()
    except yaml.YAMLError as e:
        error_msg = f"Invalid YAML syntax in {file_path}: {e}"
        if strict:
            raise ConfigError(error_msg) from e
        else:
            logger.warning(error_msg)
            return schema()


def save_config(
    config: BaseModel, file_path: Union[str, Path], *, project_root: Optional[Path] = None
) -> Path:
    """Save Pydantic configuration model to YAML file.

    Parameters
    ----------
    config : BaseModel
        Configuration object to save
    file_path : str or Path
        Destination file path
    project_root : Path, optional
        Project root directory for path resolution

    Returns
    -------
    Path
        Absolute path to saved file

    Examples
    --------
    >>> from itau_quant.config.schemas import PortfolioConfig
    >>> config = PortfolioConfig(risk_aversion=5.0)
    >>> save_config(config, "configs/my_portfolio.yaml")
    """
    path = Path(file_path)

    # If relative, make it relative to project root
    if not path.is_absolute():
        if project_root is None:
            project_root = Path(__file__).resolve().parents[3]
        path = project_root / path

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Export model to dict and save as YAML
    data = config.model_dump(mode="python", exclude_none=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            indent=2,
        )

    logger.info(f"Saved configuration to: {path}")
    return path


def load_configs_batch(
    configs: dict[str, tuple[Union[str, Path], Type[BaseModel]]], *, project_root: Optional[Path] = None
) -> dict[str, BaseModel]:
    """Load multiple configuration files at once.

    Parameters
    ----------
    configs : dict
        Mapping of config names to (file_path, schema) tuples
    project_root : Path, optional
        Project root directory

    Returns
    -------
    dict
        Mapping of config names to loaded configuration objects

    Examples
    --------
    >>> from itau_quant.config.schemas import UniverseConfig, PortfolioConfig
    >>> configs = load_configs_batch({
    ...     "universe": ("configs/universe_arara.yaml", UniverseConfig),
    ...     "portfolio": ("configs/portfolio_arara_basic.yaml", PortfolioConfig),
    ... })
    >>> print(configs["universe"].name)
    'ARARA'
    """
    loaded = {}
    for name, (file_path, schema) in configs.items():
        try:
            loaded[name] = load_config(file_path, schema, project_root=project_root)
        except ConfigError as e:
            logger.error(f"Failed to load config '{name}': {e}")
            raise

    return loaded
