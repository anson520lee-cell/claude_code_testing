#!/bin/bash
# One-time setup on macOS:
#   - Python virtual environment + dependencies (.venv)
#   - optional SEC contact (stored in .env, never committed)
#   - "Swing Terminal.app" in ~/Applications (one click: start dashboard, update data, open browser)
#   - optional: add the app to the Dock
#   - optional: hourly background job that runs the daily analysis once per new US session
#   - optional: intraday price refresh every 30 minutes in US market hours (display only, no re-scoring)
#   - optional: first data download now
# Usage:  bash scripts/install_mac.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="$HOME/Applications/Swing Terminal.app"
AGENT="$HOME/Library/LaunchAgents/com.swingterminal.daily.plist"
INTRADAY_AGENT="$HOME/Library/LaunchAgents/com.swingterminal.intraday.plist"
cd "$REPO"

ask() {  # ask "question" default(y/n)
  local answer
  read -r -p "$1 [$( [ "$2" = y ] && echo Y/n || echo y/N )] " answer || true
  answer="${answer:-$2}"
  [[ "$answer" =~ ^[Yy] ]]
}

echo "== Swing Research Terminal - macOS setup =="
echo "Folder: $REPO"
# stop a dashboard left running by an older version so the next click starts the new code
bash "$REPO/scripts/stop_terminal.sh" >/dev/null 2>&1 || true

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
echo
echo "Installing dependencies (pandas, numpy, yfinance, FastAPI ... about 150 MB on the first run)."
echo "On a slow connection this can take 10 minutes - the progress bars below keep moving while it downloads."
.venv/bin/python -m pip install --upgrade pip
# --upgrade: every update also brings yfinance to its latest release (Yahoo changes break old versions)
.venv/bin/python -m pip install --upgrade -r requirements.txt
.venv/bin/python -c "import pandas, numpy, yfinance, fastapi, uvicorn, matplotlib, yaml" \
  || { echo "Dependency installation failed - scroll up for the ERROR lines and re-run this script."; exit 1; }
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
# custom icon (assets/icon_1024.png -> .icns)
ICONSET="$(mktemp -d)/SwingTerminal.iconset"
mkdir -p "$ICONSET"
for size in 16 32 128 256 512; do
  sips -z "$size" "$size" assets/icon_1024.png --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
  double=$((size * 2))
  sips -z "$double" "$double" assets/icon_1024.png --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
done
if iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/applet.icns"; then
  touch "$APP"
else
  echo "(Could not set the custom icon; the app still works.)"
fi
echo "Created: $APP"

# 5. Dock
if defaults read com.apple.dock persistent-apps 2>/dev/null | grep -q "Swing Terminal.app"; then
  killall Dock || true   # already in the Dock: refresh it so the new icon shows
  echo "Already in the Dock (icon refreshed)."
elif ask "Add 'Swing Terminal' to the Dock?" y; then
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

# 6b. intraday prices every 30 minutes in market hours (9:30, 10:00, ... 16:00 New York; the script checks
#     New York time itself, so daylight saving and the Mac's own time zone do not matter)
if ask "Refresh prices every 30 minutes while the US market is open (display only; signals stay after-close)?" y; then
  mkdir -p "$HOME/Library/LaunchAgents"
  cat > "$INTRADAY_AGENT" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.swingterminal.intraday</string>
  <key>ProgramArguments</key><array>
    <string>/bin/bash</string><string>-c</string>
    <string>cd '$REPO' &amp;&amp; mkdir -p data &amp;&amp; .venv/bin/python intraday_run.py >> data/intraday.log 2>&amp;1</string>
  </array>
  <key>StartInterval</key><integer>300</integer>
  <key>RunAtLoad</key><false/>
</dict></plist>
PLIST
  launchctl unload "$INTRADAY_AGENT" >/dev/null 2>&1 || true
  launchctl load "$INTRADAY_AGENT"
  echo "Intraday job installed (log: data/intraday.log). Remove with: launchctl unload \"$INTRADAY_AGENT\" && rm \"$INTRADAY_AGENT\""
fi

# 7. first data download
if ask "Download data and run the first analysis now (2-4 minutes)?" y; then
  if [ -f .env ]; then set -a; . ./.env; set +a; fi
  .venv/bin/python daily_run.py
fi

echo
echo "Done. Click 'Swing Terminal' in the Dock (or run: bash scripts/start_terminal.sh)."
echo "Stop the dashboard server with: bash scripts/stop_terminal.sh"
