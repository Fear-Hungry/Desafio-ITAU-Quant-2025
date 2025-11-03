.PHONY: help install install-dev clean clean-all test test-fast test-cov test-watch lint format type-check validate validate-configs data download preprocess backtest optimize report coverage docs serve-docs ci

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

##@ General

help: ## Display this help message
	@echo "$(BLUE)PRISM-R (Portfolio Risk Intelligence System)$(NC)"
	@echo "$(BLUE)============================================$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make $(GREEN)<target>$(NC)\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(YELLOW)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Installation

install: ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	poetry install --no-dev --no-interaction

install-dev: ## Install all dependencies including dev tools
	@echo "$(BLUE)Installing all dependencies...$(NC)"
	poetry install --no-interaction

install-pre-commit: install-dev ## Install pre-commit hooks
	@echo "$(BLUE)Installing pre-commit hooks...$(NC)"
	poetry run pre-commit install

update: ## Update dependencies
	@echo "$(BLUE)Updating dependencies...$(NC)"
	poetry update

##@ Testing

test: ## Run all tests
	@echo "$(BLUE)Running full test suite...$(NC)"
	poetry run pytest -v

test-fast: ## Run tests without slow markers
	@echo "$(BLUE)Running fast tests only...$(NC)"
	poetry run pytest -v -k "not slow" -x

test-cov: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	poetry run pytest --cov=src/itau_quant --cov-report=html --cov-report=term-missing -v

test-cov-xml: ## Run tests with coverage XML (for CI)
	@echo "$(BLUE)Running tests with XML coverage...$(NC)"
	poetry run pytest --cov=src/itau_quant --cov-report=xml --cov-report=term

test-watch: ## Run tests in watch mode
	@echo "$(BLUE)Running tests in watch mode...$(NC)"
	poetry run pytest-watch

test-estimators: ## Run estimator tests only
	@echo "$(BLUE)Running estimator tests...$(NC)"
	poetry run pytest tests/estimators/ -v

test-optimization: ## Run optimization tests only
	@echo "$(BLUE)Running optimization tests...$(NC)"
	poetry run pytest tests/optimization/ -v

test-backtesting: ## Run backtesting tests only
	@echo "$(BLUE)Running backtesting tests...$(NC)"
	poetry run pytest tests/backtesting/ -v

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	poetry run pytest tests/integration/ -v

test-bl: ## Run Black-Litterman tests only
	@echo "$(BLUE)Running Black-Litterman tests...$(NC)"
	poetry run pytest tests/estimators/test_bl.py -v

##@ Code Quality

lint: ## Run linter (ruff)
	@echo "$(BLUE)Running linter...$(NC)"
	poetry run ruff check src tests

lint-fix: ## Run linter with auto-fix
	@echo "$(BLUE)Running linter with auto-fix...$(NC)"
	poetry run ruff check --fix src tests

format: ## Format code with black
	@echo "$(BLUE)Formatting code...$(NC)"
	poetry run black src tests

format-check: ## Check code formatting without changes
	@echo "$(BLUE)Checking code formatting...$(NC)"
	poetry run black --check src tests

type-check: ## Run type checker (mypy)
	@echo "$(BLUE)Running type checker...$(NC)"
	poetry run mypy src --ignore-missing-imports --no-strict-optional || true

check-all: lint format-check type-check ## Run all code quality checks
	@echo "$(GREEN)All code quality checks completed!$(NC)"

##@ Validation

validate: validate-configs test-fast lint ## Run all validation checks (fast)
	@echo "$(GREEN)All validation checks passed!$(NC)"

validate-full: validate-configs test lint type-check ## Run all validation checks (full)
	@echo "$(GREEN)All validation checks passed!$(NC)"

validate-all: ## Run master validation (full)
	@echo "$(BLUE)Running master validation (full mode)...$(NC)"
	poetry run python scripts/run_master_validation.py --mode full

validate-quick: ## Run master validation (quick)
	@echo "$(BLUE)Running master validation (quick mode)...$(NC)"
	poetry run python scripts/run_master_validation.py --mode quick --skip-download

validate-production: ## Run master validation (production)
	@echo "$(BLUE)Running master validation (production mode)...$(NC)"
	poetry run python scripts/run_master_validation.py --mode production --skip-download

validate-configs: ## Validate all YAML configuration files
	@echo "$(BLUE)Validating configuration files...$(NC)"
	@poetry run python scripts/validate_configs.py

##@ Data Pipeline

data: download preprocess ## Run full data pipeline (download + preprocess)
	@echo "$(GREEN)Data pipeline completed!$(NC)"

download: ## Download raw data
	@echo "$(BLUE)Downloading data...$(NC)"
	poetry run python scripts/run_01_data_pipeline.py --force-download

preprocess: ## Preprocess data
	@echo "$(BLUE)Preprocessing data...$(NC)"
	poetry run python scripts/run_01_data_pipeline.py --skip-download

data-clean: ## Download and clean data from 2010
	@echo "$(BLUE)Downloading and cleaning data...$(NC)"
	poetry run python scripts/run_01_data_pipeline.py --force-download --start 2010-01-01

##@ Portfolio Operations

optimize: ## Run portfolio optimization
	@echo "$(BLUE)Running portfolio optimization...$(NC)"
	poetry run itau-quant optimize --config configs/portfolio_arara_basic.yaml

backtest: ## Run backtest with default config
	@echo "$(BLUE)Running backtest...$(NC)"
	poetry run itau-quant backtest --config configs/optimizer_example.yaml --no-dry-run

backtest-dry: ## Run backtest in dry-run mode
	@echo "$(BLUE)Running backtest (dry-run)...$(NC)"
	poetry run itau-quant backtest --config configs/optimizer_example.yaml

walkforward: ## Run walk-forward validation
	@echo "$(BLUE)Running walk-forward validation...$(NC)"
	poetry run python scripts/examples/run_walkforward_arara.py

##@ Reporting

report: ## Generate performance report
	@echo "$(BLUE)Generating performance report...$(NC)"
	poetry run python scripts/analysis/generate_report.py

tearsheet: ## Generate tearsheet plots
	@echo "$(BLUE)Generating tearsheet...$(NC)"
	poetry run python scripts/analysis/plot_tearsheet.py

##@ Coverage

coverage: test-cov ## Generate coverage report (HTML)
	@echo "$(GREEN)Coverage report generated in htmlcov/$(NC)"
	@echo "$(BLUE)Open htmlcov/index.html to view$(NC)"

coverage-report: ## Display coverage report in terminal
	@echo "$(BLUE)Generating coverage report...$(NC)"
	poetry run coverage report

coverage-badge: ## Generate coverage badge
	@echo "$(BLUE)Generating coverage badge...$(NC)"
	poetry run coverage-badge -o coverage.svg -f

##@ Documentation

docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(NC)"
	@echo "$(YELLOW)Documentation builder not yet configured$(NC)"

serve-docs: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation...$(NC)"
	@echo "$(YELLOW)Documentation server not yet configured$(NC)"

##@ Cleanup

clean: ## Clean temporary files and caches
	@echo "$(BLUE)Cleaning temporary files...$(NC)"
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "$(GREEN)Cleanup completed!$(NC)"

clean-data: ## Clean processed data files
	@echo "$(YELLOW)Warning: This will delete processed data files!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf data/processed/*.parquet; \
		echo "$(GREEN)Data files cleaned!$(NC)"; \
	else \
		echo "$(BLUE)Cancelled.$(NC)"; \
	fi

clean-all: clean clean-data ## Clean everything including data
	@echo "$(GREEN)Complete cleanup finished!$(NC)"

##@ CI/CD

ci: check-all test-cov-xml validate-configs ## Simulate CI pipeline locally
	@echo "$(GREEN)CI simulation completed!$(NC)"

ci-quick: lint format-check test-fast ## Quick CI checks
	@echo "$(GREEN)Quick CI checks completed!$(NC)"

##@ Docker (Future)

docker-build: ## Build Docker image
	@echo "$(YELLOW)Docker support not yet implemented$(NC)"

docker-run: ## Run Docker container
	@echo "$(YELLOW)Docker support not yet implemented$(NC)"

##@ Utilities

show-config: ## Display current configuration
	@echo "$(BLUE)Project Configuration:$(NC)"
	@poetry run itau-quant show --format json

tree: ## Show project directory tree
	@echo "$(BLUE)Project Structure:$(NC)"
	@command -v tree >/dev/null 2>&1 && tree -I '__pycache__|*.pyc|*.egg-info|.git|.venv|htmlcov|.pytest_cache|.mypy_cache|.ruff_cache' -L 3 || echo "$(YELLOW)tree command not installed$(NC)"

env-info: ## Show environment information
	@echo "$(BLUE)Environment Information:$(NC)"
	@echo "Python version: $$(poetry run python --version)"
	@echo "Poetry version: $$(poetry --version)"
	@echo "Virtual env: $$(poetry env info -p)"

list-deps: ## List installed dependencies
	@echo "$(BLUE)Installed Dependencies:$(NC)"
	@poetry show --tree

outdated: ## Show outdated dependencies
	@echo "$(BLUE)Outdated Dependencies:$(NC)"
	@poetry show --outdated

##@ Quick Starts

quickstart: install-dev data-clean validate ## Full quickstart (install + data + validate)
	@echo "$(GREEN)Quickstart completed! Ready to run backtests.$(NC)"

quickstart-min: install validate-configs ## Minimal quickstart (install + validate)
	@echo "$(GREEN)Minimal setup completed!$(NC)"

dev-setup: install-dev install-pre-commit ## Setup development environment
	@echo "$(GREEN)Development environment ready!$(NC)"
