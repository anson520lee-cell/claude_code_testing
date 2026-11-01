# Windows 10 / 11

This branch (`claude/inspiring-einstein-uqxuuo-windows`) is the Windows version of the terminal. It is the same
program as the macOS branch plus the changes Windows needs; the macOS branch is not affected.

## Install (once)

1. Install **Python 3.12** from https://www.python.org/downloads/windows/ - tick **"Add python.exe to PATH"** -
   and **Git** from https://git-scm.com/download/win (defaults are fine). Or in PowerShell:
   `winget install Python.Python.3.12` and `winget install Git.Git`. Close and reopen PowerShell afterwards.
2. Open PowerShell and download the Windows branch into your user folder:

       cd $HOME
       git clone -b claude/inspiring-einstein-uqxuuo-windows https://github.com/anson520lee-cell/claude_code_testing.git swing-terminal
       cd swing-terminal

3. Run the installer (answers: Enter = yes):

       powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1

   It creates `.venv`, installs the dependencies, asks for the SEC contact (name + email, stored in `.env`),
   creates a **Swing Terminal** shortcut on the Desktop and in the Start menu, registers an hourly background
   task ("Swing Terminal hourly update" in Task Scheduler), optionally an intraday price task ("Swing Terminal
   intraday prices": every 30 minutes in US market hours, display only - see docs/TERMINAL.md) and runs the
   first analysis.

Double-click **Swing Terminal** to open the dashboard (http://127.0.0.1:8000). To pin it to the taskbar:
Start menu -> right-click Swing Terminal -> Pin to taskbar.

## Always-on host (open it from your phone / Mac)

Make this PC run the terminal around the clock and give it a web address (details and alternatives:
`docs/REMOTE.md`):

1. Install Tailscale: `winget install Tailscale.Tailscale` (or https://tailscale.com/download), click the
   Tailscale icon -> Log in (a free account; Google / Microsoft / GitHub sign-in works).
2. In the project folder run:

       powershell -ExecutionPolicy Bypass -File scripts\setup_windows_host.ps1

   (add `-Private` to keep the address for your own Tailscale devices only). Windows asks for administrator
   rights. The script asks for the dashboard password (hidden typing), turns off sleep while plugged in,
   starts the server at every PC start-up even before anyone signs in, lets the hourly and intraday tasks
   run while nobody is signed in, and publishes the dashboard with Tailscale Funnel. The first time,
   Tailscale may print a link to enable HTTPS / Funnel for your account - open it and click Enable.
3. It prints the address, e.g. `https://swing.tail1a2b3c.ts.net`. Open it on any device and sign in with
   the password. In the Tailscale admin console (https://login.tailscale.com/admin/machines) choose this PC ->
   Disable key expiry, so it does not drop off after 180 days.

The server still listens only on `127.0.0.1:8000`; Tailscale relays to it. Change the password:
`.venv\Scripts\python.exe serve.py --set-password`. Undo: `scripts\setup_windows_host.ps1 -Remove`.
On the Mac, `bash scripts/mac_browser_only.sh <address>` stops its own jobs and points the Dock app at the PC.

## Update

    cd $HOME\swing-terminal
    git pull
    powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1

## What differs from macOS

| | macOS | Windows |
|---|---|---|
| Launcher | Dock app (`scripts/start_terminal.sh`) | Desktop / Start-menu shortcut (`scripts/start_terminal.ps1`) |
| Background job | launchd, hourly | Task Scheduler, hourly (`pythonw`, no window) |
| Intraday prices | launchd every 5 min → `intraday_run.py` | Task Scheduler every 5 min → `intraday_run.py --log data\intraday.log` |
| Stop the server | `bash scripts/stop_terminal.sh` | `powershell -ExecutionPolicy Bypass -File scripts\stop_terminal.ps1` |
| Price-alert pop-up | macOS notification | none - triggered alerts show on the dashboard |
| Logs | `data/daily_run.log`, `data/server.log` | same |

Code changes that make it work on Windows: file locks via `msvcrt` (`src/daily/locks.py`), `.env` read by
Python itself, UTF-8 mode for every script (Windows otherwise uses the local code page, e.g. cp950), the
`tzdata` time-zone package, retries when a file being replaced is open in the dashboard, and `--log FILE`
for the window-less background task and server.

## Remove

    powershell -ExecutionPolicy Bypass -File scripts\setup_windows_host.ps1 -Remove   # if you set up the host
    Unregister-ScheduledTask -TaskName "Swing Terminal hourly update" -Confirm:$false
    Unregister-ScheduledTask -TaskName "Swing Terminal intraday prices" -Confirm:$false
    powershell -ExecutionPolicy Bypass -File scripts\stop_terminal.ps1

then delete the shortcuts and the `swing-terminal` folder.
