# =====================================================
# SkyGuard Makefile
# Common developer entry points.
# =====================================================

PY      := python3
PIP     := $(PY) -m pip
VENV    := .venv
ACT     := source $(VENV)/bin/activate

.DEFAULT_GOAL := help

.PHONY: help venv install install-dev verify test fmt lint clean run docker-build docker-up docker-down security-scan integration-test release

help:
	@echo "SkyGuard developer targets:"
	@echo "  make venv            Create virtualenv at $(VENV)"
	@echo "  make install         Install runtime deps"
	@echo "  make install-dev     Install runtime + dev deps"
	@echo "  make verify          Run environment verification"
	@echo "  make test            Run unit tests"
	@echo "  make integration-test Run integration tests"
	@echo "  make fmt             Auto-format with black + isort"
	@echo "  make lint            Lint with flake8 + mypy"
	@echo "  make security-scan   Run bandit + pip-audit"
	@echo "  make run             Run skyguard CLI"
	@echo "  make docker-build    Build Docker image"
	@echo "  make docker-up       Start docker-compose stack"
	@echo "  make docker-down     Stop docker-compose stack"
	@echo "  make release         Build dist packages"

venv:
	@if [ ! -d "$(VENV)" ]; then $(PY) -m venv $(VENV); fi
	@echo "Virtualenv ready: $(VENV)"

install: venv
	$(ACT) && $(PIP) install --upgrade pip wheel setuptools
	$(ACT) && $(PIP) install -r requirements.txt

install-dev: venv
	$(ACT) && $(PIP) install --upgrade pip wheel setuptools
	$(ACT) && $(PIP) install -r requirements-dev.txt
	$(ACT) && pre-commit install

verify:
	$(ACT) && $(PY) scripts/verify_install.py

bench:
	$(ACT) && $(PY) scripts/benchmark_device.py

test:
	$(ACT) && $(PY) -m pytest -q

integration-test:
	$(ACT) && $(PY) -m pytest -m integration -v

fmt:
	$(ACT) && black src tests scripts
	$(ACT) && isort src tests scripts

lint:
	$(ACT) && flake8 src tests
	$(ACT) && mypy src

security-scan:
	$(ACT) && bandit -r src/ -ll
	$(ACT) && pip-audit --strict --desc

release: clean
	$(ACT) && $(PY) -m build

run:
	$(ACT) && $(PY) -m skyguard

docker-build:
	docker build -t skyguard:dev .

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
