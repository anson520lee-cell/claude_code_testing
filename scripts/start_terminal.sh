#!/bin/bash
# One-click launcher (this is what the "Swing Terminal" Dock app runs):
#   1. starts the dashboard server in the background if it is not running yet,
#   2. starts a data update in the background when a newer US session has closed (no-op otherwise),
#   3. opens the dashboard in the default browser.
# Research only - nothing here places orders.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${SWING_PORT:-8000}"
URL="http://127.0.0.1:${PORT}"
PY="$REPO/.venv/bin/python"
mkdir -p "$REPO/data"
cd "$REPO" || exit 1
if [ -f "$REPO/.env" ]; then set -a; . "$REPO/.env"; set +a; fi   # e.g. SEC_USER_AGENT

notify() {
  echo "$1"
  if command -v osascript >/dev/null 2>&1; then
    osascript -e "display alert \"Swing Terminal\" message \"$1\"" >/dev/null 2>&1
  fi
}

if [ ! -x "$PY" ]; then
  notify "Not installed yet. Open Terminal and run: bash \"$REPO/scripts/install_mac.sh\""
  exit 1
fi

up() { curl -s -o /dev/null --max-time 2 "$URL/api/sessions"; }
STARTED="$REPO/data/server.started"

# A server started before the code or config changed (git pull + install, monthly universe rebuild) is
# restarted, unless it is in the middle of a run.
stale() {
  [ ! -f "$STARTED" ] && return 0
  [ "$REPO/data/.deps_updated" -nt "$STARTED" ] && return 0     # yfinance was upgraded automatically
  [ -n "$(find "$REPO/src" "$REPO/serve.py" "$REPO/config" -newer "$STARTED" \( -name '*.py' -o -name '*.yaml' \) -print -quit 2>/dev/null)" ]
}
busy() { curl -s --max-time 2 "$URL/api/run/status" | grep -q '"running": *true'; }
if up && stale && ! busy; then
  bash "$REPO/scripts/stop_terminal.sh" >/dev/null 2>&1
  for _ in $(seq 1 20); do up || break; sleep 0.5; done
fi

if ! up; then
  # keep the server log small over months of use
  if [ -f "$REPO/data/server.log" ] && [ "$(wc -c < "$REPO/data/server.log")" -gt 5000000 ]; then
    tail -c 1000000 "$REPO/data/server.log" > "$REPO/data/server.log.tmp" && mv "$REPO/data/server.log.tmp" "$REPO/data/server.log"
  fi
  touch "$STARTED"
  nohup "$PY" serve.py --port "$PORT" >> "$REPO/data/server.log" 2>&1 &
  echo $! > "$REPO/data/server.pid"
  for _ in $(seq 1 60); do up && break; sleep 0.5; done
  if ! up; then
    notify "The dashboard did not start. See $REPO/data/server.log"
    exit 1
  fi
fi

# Refresh data in the background if a newer session has closed (returns immediately).
curl -s -X POST --max-time 10 "$URL/api/run?if_new=true" >/dev/null 2>&1 || true

if [ -z "${SWING_NO_BROWSER:-}" ]; then
  if command -v open >/dev/null 2>&1; then open "$URL"
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL" >/dev/null 2>&1
  else echo "Open $URL in your browser."; fi
fi
