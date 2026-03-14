#!/usr/bin/env bash
set -euo pipefail

# Creates/refreshes a local venv using Python 3.13.
# This repo uses modern typing syntax (PEP 604 unions, etc.) that requires Python >=3.10.
# Business objective: standardize on Python 3.13.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! command -v python3.13 >/dev/null 2>&1; then
  echo "ERROR: python3.13 not found on PATH." >&2
  echo "Install Python 3.13 (e.g. via pyenv/asdf/homebrew/python.org), then re-run:" >&2
  echo "  scripts/bootstrap_venv.sh" >&2
  exit 1
fi

# Remove old venv if present (it may be built with a different interpreter).
rm -rf .venv

python3.13 -m venv .venv

# shellcheck disable=SC1091
source .venv/bin/activate

python -VV
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "OK: .venv created with $(python -V)"
