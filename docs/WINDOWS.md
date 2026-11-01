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
   task ("Swing Terminal hourly update" in Task Scheduler) and runs the first analysis.

Double-click **Swing Terminal** to open the dashboard (http://127.0.0.1:8000). To pin it to the taskbar:
Start menu -> right-click Swing Terminal -> Pin to taskbar.

## Update

    cd $HOME\swing-terminal
    git pull
    powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1

## What differs from macOS

| | macOS | Windows |
|---|---|---|
| Launcher | Dock app (`scripts/start_terminal.sh`) | Desktop / Start-menu shortcut (`scripts/start_terminal.ps1`) |
| Background job | launchd, hourly | Task Scheduler, hourly (`pythonw`, no window) |
| Stop the server | `bash scripts/stop_terminal.sh` | `powershell -ExecutionPolicy Bypass -File scripts\stop_terminal.ps1` |
| Price-alert pop-up | macOS notification | none - triggered alerts show on the dashboard |
| Logs | `data/daily_run.log`, `data/server.log` | same |

Code changes that make it work on Windows: file locks via `msvcrt` (`src/daily/locks.py`), `.env` read by
Python itself, UTF-8 mode for every script (Windows otherwise uses the local code page, e.g. cp950), the
`tzdata` time-zone package, retries when a file being replaced is open in the dashboard, and `--log FILE`
for the window-less background task and server.

## Remove

    Unregister-ScheduledTask -TaskName "Swing Terminal hourly update" -Confirm:$false
    powershell -ExecutionPolicy Bypass -File scripts\stop_terminal.ps1

then delete the shortcuts and the `swing-terminal` folder.
