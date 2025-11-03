"""Tests for YAML loader utility."""

import textwrap
import pytest
from pathlib import Path
from itau_quant.utils import yaml_loader
from itau_quant.utils.yaml_loader import load_yaml_text, read_yaml


def test_load_yaml_parses_mapping(tmp_path):
    """Original test - parses simple mapping."""
    EXPECTED_NUMBER = 10
    content = textwrap.dedent(
        f"""
        key: value
        nested:
          number: {EXPECTED_NUMBER}
        """
    )
    path = tmp_path / "config.yaml"
    path.write_text(content)

    loaded = yaml_loader.read_yaml(path)
    assert loaded["key"] == "value"
    assert loaded["nested"]["number"] == EXPECTED_NUMBER


class TestLoadYamlText:
    """Tests for load_yaml_text function."""

    def test_loads_simple_mapping(self):
        """Test loading simple key-value pairs."""
        text = """
name: test
value: 123
enabled: true
"""
        result = load_yaml_text(text)

        assert result["name"] == "test"
        assert result["value"] == 123
        assert result["enabled"] is True

    def test_loads_nested_mapping(self):
        """Test loading nested mappings."""
        text = """
database:
  host: localhost
  port: 5432
  credentials:
    user: admin
"""
        result = load_yaml_text(text)

        assert result["database"]["host"] == "localhost"
        assert result["database"]["port"] == 5432
        assert result["database"]["credentials"]["user"] == "admin"

    def test_loads_lists(self):
        """Test loading lists."""
        text = """
tickers:
  - SPY
  - QQQ
  - IWM
"""
        result = load_yaml_text(text)

        assert result["tickers"] == ["SPY", "QQQ", "IWM"]

    def test_loads_mixed_types(self):
        """Test loading mixed data types."""
        text = """
string: hello
integer: 42
float: 3.14
boolean_true: true
boolean_false: false
"""
        result = load_yaml_text(text)

        assert result["string"] == "hello"
        assert result["integer"] == 42
        assert result["float"] == 3.14
        assert result["boolean_true"] is True
        assert result["boolean_false"] is False

    def test_ignores_comments(self):
        """Test that comments are ignored."""
        text = """
# This is a comment
name: test
value: 123
"""
        result = load_yaml_text(text)

        assert result["name"] == "test"
        assert result["value"] == 123

    def test_handles_empty_lines(self):
        """Test that empty lines are handled."""
        text = """
name: test

value: 123
"""
        result = load_yaml_text(text)

        assert result["name"] == "test"
        assert result["value"] == 123

    def test_loads_empty_document(self):
        """Test loading empty document."""
        result = load_yaml_text("")
        assert result == {}

    def test_raises_on_non_mapping_root(self):
        """Test that non-mapping root raises error."""
        text = "- item1\n- item2"

        with pytest.raises(ValueError, match="root must be a mapping"):
            load_yaml_text(text)


class TestReadYaml:
    """Tests for read_yaml function."""

    def test_reads_yaml_file(self, tmp_path):
        """Test reading YAML from file."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("name: test\nvalue: 123\n", encoding="utf-8")

        result = read_yaml(yaml_file)

        assert result["name"] == "test"
        assert result["value"] == 123

    def test_reads_complex_yaml_file(self, tmp_path):
        """Test reading complex YAML structure from file."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("""
database:
  host: localhost
  port: 5432

servers:
  - name: server1
  - name: server2

settings:
  debug: true
  timeout: 30
""", encoding="utf-8")

        result = read_yaml(yaml_file)

        assert result["database"]["host"] == "localhost"
        assert len(result["servers"]) == 2
        assert result["settings"]["debug"] is True

    def test_raises_on_nonexistent_file(self, tmp_path):
        """Test that reading nonexistent file raises error."""
        nonexistent = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError):
            read_yaml(nonexistent)

    def test_handles_utf8_content(self, tmp_path):
        """Test handling of UTF-8 encoded content."""
        yaml_file = tmp_path / "utf8.yaml"
        yaml_file.write_text("name: Test\ndescription: Description\n", encoding="utf-8")

        result = read_yaml(yaml_file)

        assert "name" in result
        assert "description" in result

    def test_preserves_numeric_types(self, tmp_path):
        """Test that numeric types are preserved."""
        yaml_file = tmp_path / "numbers.yaml"
        yaml_file.write_text("""
integer: 42
float: 3.14159
negative: -10
""", encoding="utf-8")

        result = read_yaml(yaml_file)

        assert isinstance(result["integer"], int)
        assert result["integer"] == 42
        assert isinstance(result["float"], float)
        assert result["float"] == pytest.approx(3.14159)
        assert result["negative"] == -10

    def test_handles_empty_file(self, tmp_path):
        """Test handling of empty YAML file."""
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("", encoding="utf-8")

        result = read_yaml(yaml_file)

        assert result == {}

    def test_loads_real_config_structure(self, tmp_path):
        """Test loading realistic configuration structure."""
        yaml_file = tmp_path / "portfolio_config.yaml"
        yaml_file.write_text("""
portfolio:
  name: ARARA

estimators:
  window_days: 252

optimizer:
  risk_aversion: 3.0
  max_position: 0.15

tickers:
  - SPY
  - QQQ
""", encoding="utf-8")

        result = read_yaml(yaml_file)

        assert result["portfolio"]["name"] == "ARARA"
        assert result["estimators"]["window_days"] == 252
        assert result["optimizer"]["risk_aversion"] == 3.0
        assert len(result["tickers"]) == 2
        assert "SPY" in result["tickers"]
