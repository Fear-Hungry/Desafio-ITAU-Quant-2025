PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: install test test-bl

install:
	$(PYTHON) -m pip install -r requirements-dev.txt

test:
	PYTHONPATH=src $(PYTHON) -m pytest

test-bl:
	PYTHONPATH=src $(PYTHON) -m pytest tests/estimators/test_bl.py
