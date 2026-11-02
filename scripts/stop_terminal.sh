#!/bin/bash
# Stops the dashboard server started by start_terminal.sh / the Dock app.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${SWING_PORT:-8000}"
stopped=0
if [ -f "$REPO/data/server.pid" ]; then
  pid="$(cat "$REPO/data/server.pid")"
  if kill "$pid" 2>/dev/null; then stopped=1; fi
  rm -f "$REPO/data/server.pid"
fi
if [ "$stopped" -eq 0 ] && command -v lsof >/dev/null 2>&1; then
  pids="$(lsof -ti "tcp:$PORT" -sTCP:LISTEN 2>/dev/null)"
  if [ -n "$pids" ]; then kill $pids && stopped=1; fi
fi
[ "$stopped" -eq 1 ] && echo "Dashboard stopped." || echo "Dashboard was not running."
