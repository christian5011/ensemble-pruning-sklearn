#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${1:-$ROOT_DIR/.venv}"

if command -v python3 >/dev/null 2>&1; then
	PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
	PYTHON_BIN="python"
else
	echo "Python was not found. Install Python 3 and try again." >&2
	exit 1
fi

echo "Using Python interpreter: $PYTHON_BIN"
echo "Creating virtual environment at: $VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"

VENV_PIP="$VENV_DIR/bin/pip"

echo "Upgrading pip inside the virtual environment"
"$VENV_PIP" install --upgrade pip

echo "Installing package in editable mode"
"$VENV_PIP" install --no-cache-dir -e "$ROOT_DIR"

echo "Installing local development and demo extras"
"$VENV_PIP" install --no-cache-dir -r "$ROOT_DIR/requirements-dev.txt"

cat <<EOF

Local environment setup complete.

Activate it with:
  source "$VENV_DIR/bin/activate"

Then you can run:
  python examples/ensemble_pruning_demo.py
EOF
