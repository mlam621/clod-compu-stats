#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found. Run ./setup.sh first."
    exit 1
fi

exec "$VENV_DIR/bin/python" -m compu_clod_stats "$@"
