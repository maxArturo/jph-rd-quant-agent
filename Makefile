VENV := .venv
BIN := $(VENV)/bin

.PHONY: check lint typecheck test venv

check: lint typecheck test

lint:
	$(BIN)/ruff check .

typecheck:
	$(BIN)/pyright

test:
	$(BIN)/pytest

venv:
	python3 -m venv $(VENV)
	$(BIN)/pip install -e '.[dev]'
