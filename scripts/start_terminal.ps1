# One-click launcher on Windows (what the "Swing Terminal" shortcut runs):
#   1. starts the dashboard server in the background if it is not running yet (restarts it after an update;
#      on an always-on host it starts the "Swing Terminal server" task instead),
#   2. starts a data update in the background when a newer US session has closed (no-op otherwise),
#   3. opens the dashboard in the default browser.
# Research only - nothing here places orders.
$ErrorActionPreference = "Continue"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$Port = if ($env:SWING_PORT) { $env:SWING_PORT } else { "8000" }
$Url = "http://127.0.0.1:$Port"
$Pyw = "$Repo\.venv\Scripts\pythonw.exe"
$Started = "$Repo\data\server.started"
$env:PYTHONUTF8 = "1"
New-Item -ItemType Directory -Force "$Repo\data" | Out-Null

function Show-Message([string]$Text) {
    Add-Type -AssemblyName PresentationFramework
    [System.Windows.MessageBox]::Show($Text, "Swing Terminal") | Out-Null
}
if (-not (Test-Path $Pyw)) {
    Show-Message "Not installed yet. Open PowerShell in $Repo and run:  powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1"
    exit 1
}
function Test-Up {
    try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 "$Url/api/sessions" | Out-Null; return $true } catch { return $false }
}
# a server started before the code, config or yfinance changed (git pull + install, monthly universe rebuild,
# automatic yfinance upgrade) is restarted, unless it is in the middle of a run
function Test-Stale {
    if (-not (Test-Path $Started)) { return $true }
    $since = (Get-Item $Started).LastWriteTime
    $deps = "$Repo\data\.deps_updated"
    if ((Test-Path $deps) -and ((Get-Item $deps).LastWriteTime -gt $since)) { return $true }
    if ((Get-Item "$Repo\serve.py").LastWriteTime -gt $since) { return $true }
    $newer = Get-ChildItem -Path "$Repo\src", "$Repo\config" -Recurse -File -Include *.py, *.yaml -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTime -gt $since } | Select-Object -First 1
    return [bool]$newer
}
function Test-Busy {
    try { return [bool](Invoke-RestMethod -TimeoutSec 2 "$Url/api/run/status").running } catch { return $false }
}

if ((Test-Up) -and (Test-Stale) -and -not (Test-Busy)) {
    & "$Repo\scripts\stop_terminal.ps1" | Out-Null
    for ($i = 0; $i -lt 20 -and (Test-Up); $i++) { Start-Sleep -Milliseconds 500 }
}

# always-on host (scripts\setup_windows_host.ps1): the start-up task owns the server
$HostTask = Get-ScheduledTask -TaskName "Swing Terminal server" -ErrorAction SilentlyContinue
if (-not (Test-Up) -and $HostTask) {
    New-Item -ItemType File -Force $Started | Out-Null
    try { Start-ScheduledTask -TaskName "Swing Terminal server" -ErrorAction Stop } catch { }
    for ($i = 0; $i -lt 60 -and -not (Test-Up); $i++) { Start-Sleep -Milliseconds 500 }
}
if (-not (Test-Up)) {
    New-Item -ItemType File -Force $Started | Out-Null
    $server = Start-Process -FilePath $Pyw -ArgumentList @("-X", "utf8", "serve.py", "--port", $Port, "--log", "data\server.log") `
        -WorkingDirectory $Repo -WindowStyle Hidden -PassThru
    Set-Content -Path "$Repo\data\server.pid" -Value $server.Id
    for ($i = 0; $i -lt 60 -and -not (Test-Up); $i++) { Start-Sleep -Milliseconds 500 }
    if (-not (Test-Up)) {
        Show-Message "The dashboard did not start. See $Repo\data\server.log"
        exit 1
    }
}

# refresh data in the background if a newer session has closed (returns immediately)
try { Invoke-RestMethod -Method Post -TimeoutSec 10 "$Url/api/run?if_new=true" | Out-Null } catch { }

if (-not $env:SWING_NO_BROWSER) { Start-Process $Url }
