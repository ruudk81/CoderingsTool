PYTHON PROJECT SETUP (Mac mini + VS Code)

This project uses a per-project virtual environment (.venv), managed with uv, and runs on a central Mac mini.
VS Code (via Remote-SSH or Remote Tunnels) is used purely as an editor and terminal.

Use this guide when opening the project for the first time or after cloning it.


--------------------------------------------------
1. REQUIRED FILES IN THE PROJECT FOLDER
--------------------------------------------------

The project root should contain:

REQUIRED:
- setup.sh
  Shell script that creates/updates the virtual environment and installs dependencies.

STRONGLY RECOMMENDED:
- requirements.txt
  Lists all Python packages required for this project.

OPTIONAL (Python version control):
- .python-version
  Contains ONE line with the Python version to use, for example:
    3.9
  or:
    3.11

If .python-version is not present, the default/latest Python available on the Mac mini will be used.


--------------------------------------------------
2. OPEN A TERMINAL IN VS CODE
--------------------------------------------------

1. Open the project folder in VS Code
2. Open a new terminal:
   - Menu: Terminal -> New Terminal
   - The terminal should be a zsh shell running on the Mac mini


--------------------------------------------------
3. PREPARE AND ACTIVATE THE ENVIRONMENT
--------------------------------------------------

Run the following commands IN ORDER:

  chmod +x setup.sh
  source ./setup.sh

What this does:
- Creates .venv if it does not exist
- Uses the Python version from .python-version (if present)
- Installs or updates packages from requirements.txt
- Activates the virtual environment in the current terminal

If successful, your prompt will look like:

  (.venv) user@mac-mini project %


--------------------------------------------------
4. INITIALIZE CLAUDE CODE (ONE-TIME)
--------------------------------------------------

If using Claude Code, run the slash commeand once per project:

  /init

This creates a CLAUDE.md file with project context.
If you don't use Claude Code, skip this step.


--------------------------------------------------
5. SELECT THE PYTHON INTERPRETER IN VS CODE (ONE-TIME)
--------------------------------------------------

This step is required only ONCE per project.

1. Press Cmd + Shift + P (macOS) or Ctrl + Shift + P (Windows)
2. Type: Python: Select Interpreter
3. Choose:
   ./ .venv / bin / python

VS Code remembers this choice for the project.


--------------------------------------------------
6. VERIFY (OPTIONAL BUT RECOMMENDED)
--------------------------------------------------

In the terminal:

  which python
  python -V

Both should point to .venv.

In Python:

  import sys
  print(sys.executable)

Expected output:
  .../project/.venv/bin/python


--------------------------------------------------
DAILY WORKFLOW (TL;DR)
--------------------------------------------------

1. Open project in VS Code
2. Open a terminal
3. Run:
     source ./setup.sh
4. Start coding


--------------------------------------------------
NOTES
--------------------------------------------------

- Always run code AFTER activating .venv
- Do not copy virtual environments between machines
- All execution happens on the Mac mini
- VS Code is only an interface (editor + terminal)


--------------------------------------------------
APPENDIX: setup.sh
--------------------------------------------------

If setup.sh is missing, copy it from another project or use the standard version agreed for this setup.

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
