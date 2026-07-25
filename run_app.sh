#!/usr/bin/env bash
#
# Launch the CoderingsTool Streamlit app.
#
# Usage:  ./run_app.sh            (from anywhere)
#         ./run_app.sh --server.port 8502   (pass extra streamlit flags)
#
# Uses the project's .venv streamlit directly, so it does NOT depend on the
# venv being "activated" (sidesteps the cd-hook PATH issue).

set -euo pipefail

# Always run from the project root (this script lives there), regardless of CWD.
cd "$(dirname "$0")"

STREAMLIT=".venv/bin/streamlit"
if [[ ! -x "$STREAMLIT" ]]; then
  echo "❌ $STREAMLIT not found. Create/populate the venv first, e.g.:" >&2
  echo "   python -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

# View directly over LAN — the VS Code Remote-SSH port-forward is slow
# (multi-MB Streamlit frontend + websocket through the tunnel).
HOST="$(scutil --get LocalHostName 2>/dev/null || hostname -s)"
echo "🚀 Starting CoderingsTool — open http://${HOST}.local:8501 (direct over LAN)"
exec "$STREAMLIT" run src/app/app.py --server.headless true "$@"
