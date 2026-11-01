# One-time setup on Windows 10 / 11. Run it from the project folder in PowerShell:
#     powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1
# It sets up:
#   - Python virtual environment + dependencies (.venv)
#   - optional SEC contact (stored in .env, never committed)
#   - "Swing Terminal" shortcuts on the Desktop and in the Start menu (one click: start the dashboard,
#     update data, open the browser) - pin it to the taskbar from the Start menu if you like
#   - optional: hourly background task that runs the daily analysis once per new US session
#   - optional: first data download now
# Research only - nothing here places orders. Re-run it after every update (git pull).
$ErrorActionPreference = "Continue"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$TaskName = "Swing Terminal hourly update"

function Ask([string]$Question, [string]$Default) {
    $hint = if ($Default -eq "y") { "Y/n" } else { "y/N" }
    $answer = Read-Host "$Question [$hint]"
    if (-not $answer) { $answer = $Default }
    return ($answer -match "^[Yy]")
}
function Fail([string]$Message) {
    Write-Host ""
    Write-Host $Message -ForegroundColor Red
    Read-Host "Press Enter to close"
    exit 1
}

Write-Host "== Swing Research Terminal - Windows setup =="
Write-Host "Folder: $Repo"
# stop a dashboard left running by an older version so the next click starts the new code
& "$Repo\scripts\stop_terminal.ps1" | Out-Null

# 1. Python >= 3.10 (the "py" launcher comes with the python.org installer)
function Find-Python {
    foreach ($version in @("3.12", "3.13", "3.11", "3.10")) {
        $exe = & py "-$version" -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $exe) { return "$exe".Trim() }
    }
    $exe = & python -c "import sys; print(sys.executable if sys.version_info >= (3, 10) else '')" 2>$null
    if ($LASTEXITCODE -eq 0 -and "$exe".Trim()) { return "$exe".Trim() }
    return $null
}
$Python = Find-Python
if (-not $Python) {
    Fail ("Python 3.10 or newer is required. Install Python 3.12 from https://www.python.org/downloads/windows/ " +
          "(tick 'Add python.exe to PATH'), or run:  winget install Python.Python.3.12  - then run this script again.")
}
Write-Host "Using $Python"

# 2. virtual environment + dependencies
$VenvPy = "$Repo\.venv\Scripts\python.exe"
$VenvPyw = "$Repo\.venv\Scripts\pythonw.exe"
if (-not (Test-Path $VenvPy)) {
    & $Python -m venv .venv
    if ($LASTEXITCODE -ne 0) { Fail "Could not create the virtual environment (.venv)." }
}
Write-Host ""
Write-Host "Installing dependencies (pandas, numpy, yfinance, FastAPI ... about 150 MB on the first run)."
Write-Host "On a slow connection this can take 10 minutes."
& $VenvPy -m pip install --upgrade pip
# --upgrade: every update also brings yfinance to its latest release (Yahoo changes break old versions)
& $VenvPy -m pip install --upgrade -r requirements.txt
& $VenvPy -c "import pandas, numpy, yfinance, fastapi, uvicorn, matplotlib, yaml, tzdata"
if ($LASTEXITCODE -ne 0) { Fail "Dependency installation failed - scroll up for the ERROR lines and run this script again." }
Write-Host "Dependencies installed."

# 3. SEC contact (SEC EDGAR refuses requests without a name + email in the User-Agent)
$EnvFile = "$Repo\.env"
if (-not ((Test-Path $EnvFile) -and (Select-String -Path $EnvFile -Pattern "SEC_USER_AGENT" -Quiet))) {
    Write-Host ""
    Write-Host "SEC EDGAR asks every client to identify itself (name + email). It is only sent to sec.gov."
    $ua = Read-Host "Your name and email, e.g. 'Jane Doe jane@example.com' (Enter to skip)"
    if ($ua) {
        $ua = $ua.Replace('"', '')
        [System.IO.File]::AppendAllText($EnvFile, "SEC_USER_AGENT=`"$ua`"`r`n", (New-Object System.Text.UTF8Encoding($false)))
        Write-Host "Saved to .env (not committed to git)."
    } else {
        Write-Host "Skipped - fundamentals will fall back to Yahoo; add SEC_USER_AGENT to .env later."
    }
}

# 4. shortcuts (Desktop + Start menu)
$PowerShellExe = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
$Arguments = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$Repo\scripts\start_terminal.ps1`""
$Shell = New-Object -ComObject WScript.Shell
$StartMenu = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
foreach ($folder in @([Environment]::GetFolderPath("Desktop"), $StartMenu)) {
    $link = $Shell.CreateShortcut((Join-Path $folder "Swing Terminal.lnk"))
    $link.TargetPath = $PowerShellExe
    $link.Arguments = $Arguments
    $link.WorkingDirectory = $Repo
    $link.IconLocation = "$Repo\assets\icon.ico"
    $link.WindowStyle = 7
    $link.Description = "Swing research terminal (research only, no trading)"
    $link.Save()
}
Write-Host "Created 'Swing Terminal' on the Desktop and in the Start menu."
Write-Host "(To pin it to the taskbar: Start menu -> right-click 'Swing Terminal' -> Pin to taskbar.)"

# 5. hourly background task (runs the analysis only when a new session has closed)
if (Ask "Update data automatically in the background (hourly check, runs once per new US session)?" "y") {
    try {
        $action = New-ScheduledTaskAction -Execute $VenvPyw `
            -Argument "-X utf8 daily_run.py --if-new --log data\daily_run.log" -WorkingDirectory $Repo
        $trigger = New-ScheduledTaskTrigger -Daily -At "00:07"
        $trigger.Repetition = (New-ScheduledTaskTrigger -Once -At "00:07" -RepetitionInterval (New-TimeSpan -Hours 1) `
            -RepetitionDuration (New-TimeSpan -Hours 23 -Minutes 55)).Repetition
        $options = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
            -ExecutionTimeLimit (New-TimeSpan -Hours 2) -MultipleInstances IgnoreNew
        Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $options `
            -Description "Swing research terminal: daily analysis once per new US session (research only)" -Force `
            -ErrorAction Stop | Out-Null
        Write-Host "Background task installed (log: data\daily_run.log)."
        Write-Host "Remove it with:  Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:`$false"
    } catch {
        Write-Host "Could not create the background task: $($_.Exception.Message)" -ForegroundColor Yellow
        Write-Host "The terminal still works; it updates whenever you open it."
    }
}

# 6. first data download
if (Ask "Download data and run the first analysis now (2-4 minutes)?" "y") {
    & $VenvPy -X utf8 daily_run.py
}

Write-Host ""
Write-Host "Done. Double-click 'Swing Terminal' on the Desktop (or in the Start menu)."
Write-Host "Stop the dashboard server with:  powershell -ExecutionPolicy Bypass -File scripts\stop_terminal.ps1"
Read-Host "Press Enter to close"
