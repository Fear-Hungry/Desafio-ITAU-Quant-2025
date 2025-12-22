# Test Coverage Guide

This document explains how to work with test coverage in the PRISM-R project.

## Current Coverage Status

**Total Coverage: 71.02%** (as of 2025-11-02)

- **Target:** ≥ 70% (enforced by CI)
- **Total Lines:** 9,884 statements
- **Covered:** 7,020 statements
- **Missing:** 2,864 statements

## Coverage Reports

### Generating Coverage Reports

```bash
# Generate HTML coverage report
poetry run pytest --cov=src/arara_quant --cov-report=html

# Open report in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux

# Generate terminal report with missing lines
poetry run pytest --cov=src/arara_quant --cov-report=term-missing

# Generate XML report (for CI/Codecov)
poetry run pytest --cov=src/arara_quant --cov-report=xml
```

### Coverage Configuration

Coverage settings are configured in `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["src/arara_quant"]
omit = ["*/tests/*", "*/__pycache__/*"]

[tool.coverage.report]
precision = 2
show_missing = true
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
]
```

## Coverage by Module

### Well-Covered Modules (≥90%)

These modules have excellent test coverage:

- `data/processing/corporate_actions.py` - 100%
- `data/processing/frequency.py` - 100%
- `data/processing/returns.py` - 100%
- `utils/math_ops.py` - 98.63%
- `backtesting/risk_monitor.py` - 97.62%
- `config/schemas.py` - 99%
- `optimization/ga/genetic.py` - 93.81%

### Modules Needing Attention (<50%)

These modules need additional test coverage:

- `cli.py` - 46.39% (command-line interface)
- `data/sources/yf.py` - 32.00% (Yahoo Finance connector)
- `estimators/mu_robust.py` - 19.57% (robust estimators)
- `pipeline/data.py` - 22.41% (data pipeline)
- `pipeline/estimation.py` - 27.78% (estimation pipeline)
- `pipeline/optimization.py` - 29.17% (optimization pipeline)
- `utils/yaml_loader.py` - 19.75% (YAML utilities)

### Modules Excluded from Coverage

The following modules are intentionally excluded or have 0% coverage:

- `diagnostics/mu_skill.py` - Research/diagnostic tool
- `optimization/erc_calibrated.py` - Experimental feature
- `portfolio/adaptive_hedge.py` - Experimental feature
- `utils/production_logger.py` - Production monitoring
- `utils/production_monitor.py` - Production monitoring

## Coverage Enforcement

### CI/CD Pipeline

The CI pipeline enforces a minimum coverage of 70%:

```yaml
- name: Run tests with coverage
  run: |
    poetry run pytest --cov=src/arara_quant \
      --cov-report=xml \
      --cov-report=term-missing \
      --cov-fail-under=70

- name: Check coverage threshold
  run: |
    poetry run coverage report --fail-under=70
```

### Codecov Integration

Coverage reports are automatically uploaded to Codecov on every PR:

- **Project Target:** 70% minimum
- **Patch Target:** 75% for new code
- **Threshold:** ±1% allowed variance

Configuration in `.codecov.yml`:

```yaml
coverage:
  status:
    project:
      target: 70%
      threshold: 1%
    patch:
      target: 75%
      threshold: 5%
```

## Improving Coverage

### 1. Identify Uncovered Lines

Use the HTML report to find specific uncovered lines:

```bash
poetry run pytest --cov=src/arara_quant --cov-report=html
open htmlcov/index.html
```

Click on any module to see which lines are not covered (highlighted in red).

### 2. Write Tests for Critical Paths

Focus on:

1. **Happy path:** Normal execution flow
2. **Edge cases:** Boundary conditions, empty inputs
3. **Error handling:** Exception paths
4. **Integration:** Module interactions

### 3. Example: Adding Tests for `cli.py`

Current coverage: 46.39%

**Uncovered areas:**
- Command parsing
- Error handling
- Configuration loading
- Output formatting

**Test strategy:**
```python
# tests/cli/test_cli_commands.py
def test_backtest_command_with_valid_config():
    """Test backtest command with valid configuration."""
    result = runner.invoke(cli, ['backtest', '--config', 'config.yaml'])
    assert result.exit_code == 0

def test_backtest_command_with_missing_config():
    """Test backtest command with missing config file."""
    result = runner.invoke(cli, ['backtest', '--config', 'missing.yaml'])
    assert result.exit_code != 0
    assert 'not found' in result.output.lower()
```

### 4. Testing Data Sources

For modules like `data/sources/yf.py` (32% coverage):

**Strategy:**
- Mock external API calls
- Test data transformation logic
- Verify error handling

```python
@patch('yfinance.download')
def test_download_prices_handles_api_failure(mock_download):
    """Test handling of API failures."""
    mock_download.side_effect = ConnectionError("API unavailable")

    with pytest.raises(DataSourceError):
        download_prices(['SPY'], start='2020-01-01')
```

### 5. Pipeline Testing

For `pipeline/` modules (20-30% coverage):

**Strategy:**
- Test each pipeline stage independently
- Test end-to-end pipeline flow
- Mock I/O operations

```python
def test_data_pipeline_end_to_end(tmp_path):
    """Test complete data pipeline."""
    # Setup
    config = DataConfig(...)
    pipeline = DataPipeline(config, output_dir=tmp_path)

    # Execute
    result = pipeline.run()

    # Verify
    assert result.success
    assert (tmp_path / "returns.parquet").exists()
```

## Best Practices

### DO

✅ Test public APIs and interfaces
✅ Test edge cases and error conditions
✅ Mock external dependencies (APIs, file I/O)
✅ Use fixtures for common test data
✅ Add tests when fixing bugs
✅ Aim for 80%+ coverage on business logic

### DON'T

❌ Chase 100% coverage blindly
❌ Test trivial getters/setters
❌ Test third-party library code
❌ Write tests just to increase percentage
❌ Skip testing complex logic

## Coverage Goals by Module Type

| Module Type | Target Coverage | Priority |
|-------------|----------------|----------|
| Core business logic | 85-95% | High |
| Data processing | 80-90% | High |
| Optimization | 75-85% | High |
| Utilities | 70-80% | Medium |
| CLI/Interface | 60-70% | Medium |
| Experimental | 40-60% | Low |

## Continuous Improvement

### Monthly Coverage Review

1. Generate coverage report
2. Identify modules below target
3. Prioritize based on criticality
4. Add tests incrementally
5. Update documentation

### Coverage Trends

Track coverage over time:

```bash
# Generate coverage badge
poetry run coverage-badge -o coverage.svg -f

# View historical trends on Codecov
# Visit: https://codecov.io/gh/YOUR_USERNAME/arara-quant-lab
```

## Troubleshooting

### "No data was collected"

If pytest-cov reports no data:

```bash
# Ensure module is imported
poetry run python -c "import arara_quant; print('OK')"

# Run with verbose output
poetry run pytest --cov=src/arara_quant --cov-report=term -v

# Check coverage configuration
cat pyproject.toml | grep -A 10 "\[tool.coverage"
```

### Coverage Differs Between Local and CI

Common causes:

1. **Different Python versions:** Run same version locally
2. **Missing dependencies:** Ensure all test deps installed
3. **Platform differences:** Some modules may be platform-specific

```bash
# Match CI environment
poetry env use 3.11
poetry install
poetry run pytest --cov=src/arara_quant
```

### Coverage Badge Not Updating

```bash
# Manually upload to Codecov
poetry run pytest --cov=src/arara_quant --cov-report=xml
curl -s https://codecov.io/bash | bash
```

## Resources

- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [Codecov Documentation](https://docs.codecov.io/)
- [Coverage.py Guide](https://coverage.readthedocs.io/)
- [Testing Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)

## Questions?

For questions about coverage:

1. Check this guide first
2. Review existing tests in `tests/`
3. Check CI logs for failures
4. Open an issue with coverage details
