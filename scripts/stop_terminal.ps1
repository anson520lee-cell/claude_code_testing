# Stops the dashboard server started by start_terminal.ps1 / the "Swing Terminal" shortcut.
$Repo = Split-Path -Parent $PSScriptRoot
$Port = if ($env:SWING_PORT) { [int]$env:SWING_PORT } else { 8000 }
$PidFile = "$Repo\data\server.pid"
$stopped = $false
function Stop-IfPython([int]$Id) {
    $proc = Get-Process -Id $Id -ErrorAction SilentlyContinue
    if ($proc -and $proc.ProcessName -like "python*") {   # never kill an unrelated process that reused the id
        Stop-Process -Id $Id -Force -ErrorAction SilentlyContinue
        return $true
    }
    return $false
}
if (Test-Path $PidFile) {
    $id = Get-Content $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($id -match "^\d+$") { $stopped = Stop-IfPython ([int]$id) }
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
}
if (-not $stopped) {
    $listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    foreach ($c in $listeners) { if (Stop-IfPython ([int]$c.OwningProcess)) { $stopped = $true } }
}
if ($stopped) { "Dashboard stopped." } else { "Dashboard was not running." }
