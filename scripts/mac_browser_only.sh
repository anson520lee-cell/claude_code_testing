#!/bin/bash
# Turn this Mac into a browser-only client once the terminal runs on another computer (e.g. the Windows PC,
# see docs/REMOTE.md):
#   - stops the local dashboard server,
#   - removes the hourly analysis job and the intraday price job (the other computer does that work now),
#   - rebuilds "Swing Terminal.app" (Dock / ~/Applications) so a click opens the remote address instead.
# Nothing is deleted: the local data/ folder stays. To make the Mac run everything again:
#   bash scripts/install_mac.sh
# Usage:  bash scripts/mac_browser_only.sh https://swing.your-tailnet.ts.net
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
URL="${1:-}"
if [[ ! "$URL" =~ ^https?://[A-Za-z0-9.-]+(:[0-9]+)?(/.*)?$ ]]; then
  echo "Usage: bash scripts/mac_browser_only.sh https://swing.your-tailnet.ts.net"
  echo "(the address the Windows setup printed at the end)"
  exit 1
fi
APP="$HOME/Applications/Swing Terminal.app"
bash "$REPO/scripts/stop_terminal.sh" || true
for label in com.swingterminal.daily com.swingterminal.intraday; do
  plist="$HOME/Library/LaunchAgents/$label.plist"
  if [ -f "$plist" ]; then
    launchctl unload "$plist" >/dev/null 2>&1 || true
    rm -f "$plist"
    echo "Removed background job $label"
  fi
done
mkdir -p "$HOME/Applications" "$REPO/data"
icon=""
[ -f "$APP/Contents/Resources/applet.icns" ] && icon="$(mktemp -d)/applet.icns" && cp "$APP/Contents/Resources/applet.icns" "$icon"
rm -rf "$APP"
osacompile -o "$APP" -e "open location \"$URL\""
if [ -n "$icon" ]; then cp "$icon" "$APP/Contents/Resources/applet.icns" && touch "$APP"; fi
printf '%s\n' "$URL" > "$REPO/data/remote_url"
echo
echo "Done. 'Swing Terminal' now opens $URL"
echo "This Mac no longer runs the analysis. Undo with:  bash scripts/install_mac.sh"
