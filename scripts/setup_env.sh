#!/usr/bin/env bash
# =====================================================
# SkyGuard environment bootstrap (macOS / Linux)
# Idempotent. Safe to re-run.
# =====================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

echo "==> Project root : $PROJECT_ROOT"
echo "==> Python       : $($PYTHON_BIN --version)"
echo "==> Virtualenv   : $VENV_DIR"

# 1. venv
if [ ! -d "$VENV_DIR" ]; then
  echo "==> Creating virtualenv..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  echo "==> Virtualenv already exists, reusing"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# 2. pip bootstrap
echo "==> Upgrading pip/wheel/setuptools..."
python -m pip install --upgrade pip wheel setuptools

# 3. Choose deps
if [ "${1:-dev}" = "runtime" ]; then
  echo "==> Installing runtime dependencies..."
  pip install -r requirements.txt
else
  echo "==> Installing dev dependencies (runtime + lint + test)..."
  pip install -r requirements-dev.txt
fi

# 4. pre-commit hooks
if [ -f .pre-commit-config.yaml ] && [ "${1:-dev}" = "dev" ]; then
  echo "==> Installing pre-commit hooks..."
  pre-commit install || true
fi

# 5. env file
if [ ! -f .env ] && [ -f .env.example ]; then
  cp .env.example .env
  echo "==> Created .env from .env.example (please review and edit)"
fi

# 6. git init (optional, only if not a repo yet)
if [ ! -d .git ]; then
  if [ "${SKYGUARD_INIT_GIT:-0}" = "1" ]; then
    echo "==> Initializing git repository..."
    git init -q -b main
    git add .
    git -c user.email="dev@skyguard.local" -c user.name="SkyGuard Dev" \
        commit -q -m "chore: initial scaffold (Sprint 1)"
  else
    echo "==> Skipping git init (set SKYGUARD_INIT_GIT=1 to enable)"
  fi
fi

# 7. verify
echo
echo "==> Verifying installation..."
python scripts/verify_install.py || {
  echo "!! Verification failed. Please review the output above."
  exit 1
}

echo
echo "All set. Activate with:  source $VENV_DIR/bin/activate"
echo "Then run:               skyguard info"
