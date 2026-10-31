#!/bin/bash
# One-time setup on macOS:
#   - Python virtual environment + dependencies (.venv)
#   - optional SEC contact (stored in .env, never committed)
#   - "Swing Terminal.app" in ~/Applications (one click: start dashboard, update data, open browser)
#   - optional: add the app to the Dock
#   - optional: hourly background job that runs the daily analysis once per new US session
#   - optional: first data download now
# Usage:  bash scripts/install_mac.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="$HOME/Applications/Swing Terminal.app"
AGENT="$HOME/Library/LaunchAgents/com.swingterminal.daily.plist"
cd "$REPO"

ask() {  # ask "question" default(y/n)
  local answer
  read -r -p "$1 [$( [ "$2" = y ] && echo Y/n || echo y/N )] " answer || true
  answer="${answer:-$2}"
  [[ "$answer" =~ ^[Yy] ]]
}

echo "== Swing Research Terminal - macOS setup =="
echo "Folder: $REPO"

# 1. Python >= 3.10
PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" >/dev/null 2>&1 && \
     "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    PYTHON="$(command -v "$candidate")"; break
  fi
done
if [ -z "$PYTHON" ]; then
  echo "Python 3.10 or newer is required (macOS's built-in python3 may be older)."
  echo "Install it from https://www.python.org/downloads/macos/  or with Homebrew:  brew install python@3.12"
  exit 1
fi
echo "Using $PYTHON ($("$PYTHON" --version))"

# 2. virtual environment + dependencies
if [ ! -x .venv/bin/python ]; then "$PYTHON" -m venv .venv; fi
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -r requirements.txt
echo "Dependencies installed."

# 3. SEC contact (SEC EDGAR refuses requests without a name + email in the User-Agent)
if [ ! -f .env ] || ! grep -q SEC_USER_AGENT .env; then
  echo
  echo "SEC EDGAR asks every client to identify itself (name + email). It is only sent to sec.gov."
  read -r -p "Your name and email, e.g. 'Jane Doe jane@example.com' (Enter to skip): " ua || true
  if [ -n "${ua:-}" ]; then
    printf 'SEC_USER_AGENT="%s"\n' "${ua//\"/}" >> .env
    echo "Saved to .env (not committed to git)."
  else
    echo "Skipped - fundamentals will fall back to Yahoo; add SEC_USER_AGENT to .env later."
  fi
fi

chmod +x scripts/start_terminal.sh scripts/stop_terminal.sh

# 4. the app
mkdir -p "$HOME/Applications"
rm -rf "$APP"
osacompile -o "$APP" -e "do shell script \"/bin/bash \" & quoted form of \"$REPO/scripts/start_terminal.sh\" & \" > /dev/null 2>&1 &\""
echo "Created: $APP"

# 5. Dock
if ask "Add 'Swing Terminal' to the Dock?" y; then
  defaults write com.apple.dock persistent-apps -array-add \
    "<dict><key>tile-data</key><dict><key>file-data</key><dict><key>_CFURLString</key><string>$APP</string><key>_CFURLStringType</key><integer>0</integer></dict></dict></dict>"
  killall Dock || true
  echo "Added to the Dock."
else
  echo "You can drag '$APP' to the Dock yourself."
fi

# 6. hourly background job (runs the analysis only when a new session has closed)
if ask "Update data automatically in the background (hourly check, runs once per new US session)?" y; then
  mkdir -p "$HOME/Library/LaunchAgents"
  cat > "$AGENT" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.swingterminal.daily</string>
  <key>ProgramArguments</key><array>
    <string>/bin/bash</string><string>-c</string>
    <string>cd '$REPO' &amp;&amp; { [ -f .env ] &amp;&amp; set -a &amp;&amp; . ./.env &amp;&amp; set +a; true; } &amp;&amp; .venv/bin/python daily_run.py --if-new >> data/daily_run.log 2>&amp;1</string>
  </array>
  <key>StartInterval</key><integer>3600</integer>
  <key>RunAtLoad</key><false/>
</dict></plist>
PLIST
  launchctl unload "$AGENT" >/dev/null 2>&1 || true
  launchctl load "$AGENT"
  echo "Background job installed (log: data/daily_run.log). Remove with: launchctl unload \"$AGENT\" && rm \"$AGENT\""
fi

# 7. first data download
if ask "Download data and run the first analysis now (2-4 minutes)?" y; then
  if [ -f .env ]; then set -a; . ./.env; set +a; fi
  .venv/bin/python daily_run.py
fi

echo
echo "Done. Click 'Swing Terminal' in the Dock (or run: bash scripts/start_terminal.sh)."
echo "Stop the dashboard server with: bash scripts/stop_terminal.sh"
