#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "Creating virtual environment at $VENV_DIR ..."
python3 -m venv "$VENV_DIR"

echo "Installing compu-clod-stats in development mode ..."
"$VENV_DIR/bin/pip" install --upgrade pip -q
"$VENV_DIR/bin/pip" install -e "$SCRIPT_DIR" -q

if command -v pipx &>/dev/null; then
    echo "Installing global 'ccs' command via pipx ..."
    pipx install --force "$SCRIPT_DIR" -q
else
    echo "Note: pipx not found — skipping global 'ccs' command install."
    echo "  Install pipx and re-run, or use: pipx install $SCRIPT_DIR"
fi

echo "Setup complete. Run ./run.sh or 'ccs' to start the dashboard."
