#!/usr/bin/env bash
# Works when sourced from zsh or bash.
# Does setup in a subshell (so it can't mess with your prompt options),
# then activates .venv in the current shell if sourced.

REQ_FILE="requirements.txt"
VENV_DIR=".venv"
PYVER_FILE=".python-version"

# Detect "sourced" (works in both bash and zsh):
_is_sourced() { return 0 2>/dev/null; }
if _is_sourced; then SOURCED=1; else SOURCED=0; fi

# Resolve the directory this script lives in (bash + zsh):
if [ -n "${BASH_VERSION-}" ]; then
  SCRIPT_PATH="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION-}" ]; then
  SCRIPT_PATH="${(%):-%x}"
else
  SCRIPT_PATH="$0"
fi

PROJECT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
PROJECT_NAME="$(basename "$PROJECT_DIR")"

echo "[INFO] Project: $PROJECT_NAME"
echo "[INFO] Project dir: $PROJECT_DIR"

# Run setup in a subshell so we don't change the caller's shell options/prompt
(
  set -e

  cd "$PROJECT_DIR"

  if ! command -v uv >/dev/null 2>&1; then
    echo "[ERROR] uv not found on PATH on the Mac mini."
    exit 1
  fi

  # Read optional python version
  PYVER=""
  if [ -f "$PYVER_FILE" ]; then
    PYVER="$(grep -E '^\s*[^#\s]+' "$PYVER_FILE" | head -n 1 | tr -d '[:space:]' || true)"
  fi
  echo "[INFO] Python requested: ${PYVER:-<default>}"

  # Ensure venv exists
  if [ ! -x "$VENV_DIR/bin/python" ]; then
    if [ -n "$PYVER" ]; then
      echo "[STEP] Creating venv with Python $PYVER"
      uv venv --python "$PYVER" "$VENV_DIR"
    else
      echo "[STEP] Creating venv with default Python"
      uv venv "$VENV_DIR"
    fi
  else
    echo "[STEP] Venv exists: $VENV_DIR"
  fi

  PY="$VENV_DIR/bin/python"

  # Install requirements (idempotent)
  if [ -f "$REQ_FILE" ]; then
    echo "[STEP] Installing requirements from $REQ_FILE"
    uv pip install --python "$PY" -r "$REQ_FILE"
  else
    echo "[WARN] No $REQ_FILE found; skipping dependency install."
  fi
)

# Activate only if sourced
if [ "$SOURCED" -eq 1 ]; then
  # shellcheck disable=SC1091
  source "$PROJECT_DIR/$VENV_DIR/bin/activate"
  echo "[DONE] Activated: $VENV_DIR"
else
  echo "[INFO] To activate in this terminal run:"
  echo "       source ./setup.sh"
fi
