PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: install test test-bl validate-all validate-quick validate-production

install:
	$(PYTHON) -m pip install -r requirements-dev.txt

test:
	PYTHONPATH=src $(PYTHON) -m pytest

test-bl:
	PYTHONPATH=src $(PYTHON) -m pytest tests/estimators/test_bl.py

# Validation targets
validate-all:
	poetry run python scripts/run_master_validation.py --mode full

validate-quick:
	poetry run python scripts/run_master_validation.py --mode quick --skip-download

validate-production:
	poetry run python scripts/run_master_validation.py --mode production --skip-download
