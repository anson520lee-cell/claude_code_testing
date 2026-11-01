# Makes this Windows PC the always-on host of the terminal, reachable from your phone / Mac by a web address.
# Run it once after install_windows.ps1, from the project folder in PowerShell:
#     powershell -ExecutionPolicy Bypass -File scripts\setup_windows_host.ps1            (public address)
#     powershell -ExecutionPolicy Bypass -File scripts\setup_windows_host.ps1 -Private   (only your Tailscale devices)
# It asks for administrator rights (Windows shows a prompt) and then:
#   1. sets the dashboard login password if none is set (asked here, stored only as a hash in data\auth.json)
#   2. keeps the PC awake while plugged in (no sleep / hibernate; the screen may still turn off)
#   3. runs the dashboard server at every start-up, even before anyone signs in, and restarts it if it stops
#   4. lets the hourly analysis and intraday price tasks run while nobody is signed in
#   5. publishes the dashboard with Tailscale: https://<name>.<tailnet>.ts.net
#        default  = Tailscale Funnel: any device on the internet reaches the login page
#        -Private = Tailscale Serve: only devices signed in to your Tailscale account
# The server itself only listens on 127.0.0.1:8000; Tailscale relays to it. Research only - no orders.
# Undo: powershell -ExecutionPolicy Bypass -File scripts\setup_windows_host.ps1 -Remove
param(
    [switch]$Private,
    [switch]$Remove,
    [string]$MachineName = "swing",
    [string]$RunAs = ""
)
$ErrorActionPreference = "Continue"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$ServerTask = "Swing Terminal server"
$OtherTasks = @("Swing Terminal hourly update", "Swing Terminal intraday prices")
$Port = if ($env:SWING_PORT) { $env:SWING_PORT } else { "8000" }
$VenvPy = "$Repo\.venv\Scripts\python.exe"
$VenvPyw = "$Repo\.venv\Scripts\pythonw.exe"

function Done([int]$Code) { Write-Host ""; Read-Host "Press Enter to close"; exit $Code }

# --- administrator rights (the tasks run as YOUR account; remember who that is before elevating)
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $RunAs) { $RunAs = "$env:USERDOMAIN\$env:USERNAME" }
if (-not $isAdmin) {
    $argList = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$PSCommandPath`"", "-RunAs", "`"$RunAs`"",
                 "-MachineName", $MachineName)
    if ($Private) { $argList += "-Private" }
    if ($Remove) { $argList += "-Remove" }
    Write-Host "Asking for administrator rights (needed for start-up tasks and power settings) ..."
    try { Start-Process powershell.exe -Verb RunAs -ArgumentList $argList -ErrorAction Stop } catch {
        Write-Host "Administrator rights were refused - nothing changed." -ForegroundColor Yellow
    }
    exit 0
}

function Find-Tailscale {
    $cmd = Get-Command tailscale.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($p in @("$env:ProgramFiles\Tailscale\tailscale.exe", "${env:ProgramFiles(x86)}\Tailscale\tailscale.exe")) {
        if (Test-Path $p) { return $p }
    }
    return $null
}
$Tailscale = Find-Tailscale

if ($Remove) {
    Unregister-ScheduledTask -TaskName $ServerTask -Confirm:$false -ErrorAction SilentlyContinue
    if ($Tailscale) { & $Tailscale funnel reset 2>$null | Out-Null; & $Tailscale serve reset 2>$null | Out-Null }
    Write-Host "Removed the start-up server task and the Tailscale address. The power settings and the hourly /"
    Write-Host "intraday tasks stay as they are (re-run install_windows.ps1 to register those for sign-in only)."
    Done 0
}

Write-Host "== Swing Terminal - always-on host setup =="
Write-Host "Folder: $Repo    Tasks run as: $RunAs"
if (-not (Test-Path $VenvPyw)) {
    Write-Host "Not installed yet - run  powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1  first." -ForegroundColor Red
    Done 1
}

# 1. password (the address is reachable from other devices, so the login must be on)
$status = (& $VenvPy -X utf8 serve.py --password-status 2>$null | Select-Object -Last 1)
if ("$status".Trim() -ne "set") {
    Write-Host ""
    Write-Host "Choose the dashboard password (typing is hidden; at least 6 characters)."
    do { & $VenvPy -X utf8 serve.py --set-password } while ($LASTEXITCODE -ne 0)
} else {
    Write-Host "Login password: already set (change it with  .venv\Scripts\python.exe serve.py --set-password)."
}

# 2. stay awake on mains power (screen-off is fine; sleep / hibernate would take the dashboard offline)
powercfg /change standby-timeout-ac 0 | Out-Null
powercfg /change hibernate-timeout-ac 0 | Out-Null
Write-Host "Power: sleep and hibernate turned off while plugged in."

# 3. + 4. tasks under your account, "run whether signed in or not" (S4U: no stored password)
$principal = New-ScheduledTaskPrincipal -UserId $RunAs -LogonType S4U -RunLevel Limited
try {
    $action = New-ScheduledTaskAction -Execute $VenvPyw `
        -Argument "-X utf8 serve.py --port $Port --log data\server.log" -WorkingDirectory $Repo
    $trigger = New-ScheduledTaskTrigger -AtStartup
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
        -MultipleInstances IgnoreNew
    Register-ScheduledTask -TaskName $ServerTask -Action $action -Trigger $trigger -Settings $settings `
        -Principal $principal -Description "Swing research terminal: dashboard server on 127.0.0.1:$Port (research only)" `
        -Force -ErrorAction Stop | Out-Null
    Write-Host "Start-up task '$ServerTask' installed (log: data\server.log)."
} catch {
    Write-Host "Could not create the start-up task: $($_.Exception.Message)" -ForegroundColor Red
    Done 1
}
foreach ($name in $OtherTasks) {
    $task = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
    if ($task) {
        try {
            Set-ScheduledTask -TaskName $name -Principal $principal -ErrorAction Stop | Out-Null
            Write-Host "'$name' now also runs while nobody is signed in."
        } catch { Write-Host "Could not update '$name': $($_.Exception.Message)" -ForegroundColor Yellow }
    }
}
# restart the server under the task (stops one started by the Desktop shortcut)
& "$Repo\scripts\stop_terminal.ps1" | Out-Null
Start-Sleep -Seconds 2
Start-ScheduledTask -TaskName $ServerTask
$up = $false
for ($i = 0; $i -lt 60 -and -not $up; $i++) {
    Start-Sleep -Milliseconds 500
    try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 "http://127.0.0.1:$Port/api/sessions" | Out-Null; $up = $true } catch { }
}
if ($up) { Write-Host "Dashboard server running on http://127.0.0.1:$Port" } else {
    Write-Host "The server did not answer yet - see data\server.log" -ForegroundColor Yellow
}

# 5. Tailscale address
Write-Host ""
if (-not $Tailscale) {
    Write-Host "Tailscale is not installed. Install it (winget install Tailscale.Tailscale, or https://tailscale.com/download),"
    Write-Host "sign in, then run this script again to get the web address." -ForegroundColor Yellow
    Done 0
}
$state = $null
try { $state = (& $Tailscale status --json | ConvertFrom-Json) } catch { }
if (-not $state -or $state.BackendState -ne "Running") {
    Write-Host "Tailscale is installed but not signed in. Click the Tailscale icon -> Log in, then run this script again." -ForegroundColor Yellow
    Done 0
}
& $Tailscale set --unattended=true 2>$null | Out-Null       # stay connected when nobody is signed in
if ($MachineName) { & $Tailscale set --hostname=$MachineName 2>$null | Out-Null }
& $Tailscale funnel reset 2>$null | Out-Null
& $Tailscale serve reset 2>$null | Out-Null
$target = "http://127.0.0.1:$Port"
Write-Host "Publishing the dashboard with Tailscale (the first time it may print a link to enable HTTPS /"
Write-Host "Funnel for your account - open it, click Enable, and the command continues) ..."
if ($Private) {
    & $Tailscale serve --bg --https=443 $target
    & $Tailscale serve --bg --http=80 $target
} else {
    & $Tailscale funnel --bg --https=443 $target
}
if ($LASTEXITCODE -ne 0) {
    Write-Host "Tailscale did not accept the setting (see the message above). Fix it and run this script again." -ForegroundColor Yellow
    Done 1
}
$state = (& $Tailscale status --json | ConvertFrom-Json)
$dns = "$($state.Self.DNSName)".TrimEnd(".")
$url = "https://$dns"
Set-Content -Path "$Repo\data\remote_url.txt" -Value $url -Encoding ascii
Write-Host ""
Write-Host "Done. Your terminal's address:" -ForegroundColor Green
Write-Host "    $url" -ForegroundColor Green
if ($Private) {
    Write-Host "Reachable from devices signed in to your Tailscale account (short form: http://$($dns.Split('.')[0]))."
} else {
    Write-Host "Reachable from any device on the internet; everyone gets the login page first."
    Write-Host "(The public name can take up to 10 minutes to start working the first time.)"
}
Write-Host ""
Write-Host "Two one-time clicks in the Tailscale admin console (https://login.tailscale.com/admin/machines):"
Write-Host "  - this PC -> ... -> Disable key expiry   (otherwise it drops off after 180 days)"
Write-Host "  - optional: this PC -> ... -> Edit machine name, if the address above is not '$MachineName'"
Done 0
