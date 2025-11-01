"""Tests for configuration loading and validation module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, Field

from itau_quant.config.loader import (
    ConfigError,
    _resolve_config_path,
    load_config,
    save_config,
)


# Test schemas
class SimpleConfig(BaseModel):
    """Simple test configuration."""

    name: str
    value: int = 10
    enabled: bool = True


class StrictConfig(BaseModel):
    """Strict test configuration with validation."""

    risk_aversion: float = Field(ge=0, le=100)
    max_position: float = Field(ge=0, le=1)


@pytest.fixture
def temp_config_file(tmp_path: Path) -> Path:
    """Create a temporary YAML config file."""
    config_file = tmp_path / "test_config.yaml"
    data = {"name": "test", "value": 42, "enabled": False}
    with open(config_file, "w") as f:
        yaml.dump(data, f)
    return config_file


@pytest.fixture
def invalid_yaml_file(tmp_path: Path) -> Path:
    """Create a file with invalid YAML syntax."""
    invalid_file = tmp_path / "invalid.yaml"
    with open(invalid_file, "w") as f:
        f.write("name: test\nvalue: [unclosed list\n")
    return invalid_file


@pytest.fixture
def empty_config_file(tmp_path: Path) -> Path:
    """Create an empty YAML file."""
    empty_file = tmp_path / "empty.yaml"
    empty_file.touch()
    return empty_file


# Tests for _resolve_config_path


def test_resolve_absolute_existing_path(temp_config_file: Path):
    """Test resolving absolute path that exists."""
    resolved = _resolve_config_path(temp_config_file)
    assert resolved == temp_config_file
    assert resolved.is_absolute()


def test_resolve_relative_to_project_root(tmp_path: Path):
    """Test resolving relative path from project root."""
    # Create nested structure
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_file = config_dir / "test.yaml"
    config_file.write_text("name: test")

    # Resolve relative to tmp_path
    resolved = _resolve_config_path("configs/test.yaml", project_root=tmp_path)
    assert resolved == config_file
    assert resolved.exists()


def test_resolve_relative_to_cwd(tmp_path: Path, monkeypatch):
    """Test resolving relative path from current working directory."""
    config_file = tmp_path / "local.yaml"
    config_file.write_text("name: test")

    # Change to tmp_path and resolve
    monkeypatch.chdir(tmp_path)
    resolved = _resolve_config_path("local.yaml")
    assert resolved.name == "local.yaml"
    assert resolved.exists()


def test_resolve_nonexistent_file_raises():
    """Test that resolving non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        _resolve_config_path("nonexistent.yaml")


# Tests for load_config


def test_load_config_valid_yaml(temp_config_file: Path):
    """Test loading valid YAML configuration."""
    config = load_config(temp_config_file, SimpleConfig)

    assert isinstance(config, SimpleConfig)
    assert config.name == "test"
    assert config.value == 42
    assert config.enabled is False


def test_load_config_with_defaults(tmp_path: Path):
    """Test loading config with missing optional fields uses defaults."""
    config_file = tmp_path / "minimal.yaml"
    config_file.write_text("name: minimal_test")

    config = load_config(config_file, SimpleConfig)
    assert config.name == "minimal_test"
    assert config.value == 10  # default
    assert config.enabled is True  # default


def test_load_config_invalid_yaml_strict(invalid_yaml_file: Path):
    """Test that invalid YAML raises ConfigError in strict mode."""
    with pytest.raises(ConfigError, match="Invalid YAML syntax"):
        load_config(invalid_yaml_file, SimpleConfig, strict=True)


def test_load_config_invalid_yaml_non_strict(invalid_yaml_file: Path):
    """Test that invalid YAML returns defaults in non-strict mode."""
    config = load_config(invalid_yaml_file, SimpleConfig, strict=False)
    assert isinstance(config, SimpleConfig)
    # Returns default instance
    assert config.value == 10


def test_load_config_empty_file_raises(empty_config_file: Path):
    """Test that empty config file raises ConfigError."""
    with pytest.raises(ConfigError, match="Empty configuration file"):
        load_config(empty_config_file, SimpleConfig)


def test_load_config_missing_file_strict():
    """Test that missing file raises ConfigError in strict mode."""
    with pytest.raises(ConfigError, match="Configuration file not found"):
        load_config("nonexistent.yaml", SimpleConfig, strict=True)


def test_load_config_missing_file_non_strict():
    """Test that missing file returns defaults in non-strict mode."""
    config = load_config("nonexistent.yaml", SimpleConfig, strict=False)
    assert isinstance(config, SimpleConfig)
    assert config.value == 10


def test_load_config_validation_error_strict(tmp_path: Path):
    """Test that validation error raises ConfigError in strict mode."""
    config_file = tmp_path / "invalid_values.yaml"
    config_file.write_text("risk_aversion: -5\nmax_position: 1.5")

    with pytest.raises(ConfigError, match="Configuration validation failed"):
        load_config(config_file, StrictConfig, strict=True)


def test_load_config_validation_error_non_strict(tmp_path: Path):
    """Test that validation error returns defaults in non-strict mode."""
    config_file = tmp_path / "invalid_values.yaml"
    config_file.write_text("risk_aversion: -5\nmax_position: 1.5")

    config = load_config(config_file, StrictConfig, strict=False)
    assert isinstance(config, StrictConfig)
    # Returns default instance (if model has defaults)


# Tests for save_config


def test_save_config_creates_file(tmp_path: Path):
    """Test that save_config creates a valid YAML file."""
    config = SimpleConfig(name="saved_config", value=99, enabled=True)
    output_file = tmp_path / "output.yaml"

    saved_path = save_config(config, output_file)

    assert saved_path.exists()
    assert saved_path == output_file

    # Verify content
    with open(saved_path, "r") as f:
        data = yaml.safe_load(f)
    assert data["name"] == "saved_config"
    assert data["value"] == 99
    assert data["enabled"] is True


def test_save_config_overwrites_existing(tmp_path: Path):
    """Test that save_config overwrites existing files."""
    output_file = tmp_path / "overwrite.yaml"
    output_file.write_text("old: data")

    config = SimpleConfig(name="new_config", value=123)
    save_config(config, output_file)

    with open(output_file, "r") as f:
        data = yaml.safe_load(f)
    assert "old" not in data
    assert data["name"] == "new_config"


def test_save_config_creates_parent_dirs(tmp_path: Path):
    """Test that save_config creates parent directories."""
    nested_file = tmp_path / "configs" / "nested" / "config.yaml"
    config = SimpleConfig(name="nested")

    saved_path = save_config(config, nested_file)

    assert saved_path.exists()
    assert saved_path.parent.exists()


def test_save_and_load_roundtrip(tmp_path: Path):
    """Test that saving and loading config preserves data."""
    original = SimpleConfig(name="roundtrip", value=777, enabled=False)
    config_file = tmp_path / "roundtrip.yaml"

    # Save
    save_config(original, config_file)

    # Load
    loaded = load_config(config_file, SimpleConfig)

    assert loaded.name == original.name
    assert loaded.value == original.value
    assert loaded.enabled == original.enabled


def test_load_config_with_project_root(tmp_path: Path):
    """Test load_config with explicit project_root parameter."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_file = config_dir / "project.yaml"
    config_file.write_text("name: project_test\nvalue: 555")

    config = load_config("configs/project.yaml", SimpleConfig, project_root=tmp_path)
    assert config.name == "project_test"
    assert config.value == 555
