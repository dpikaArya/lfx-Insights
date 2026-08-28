<#
.SYNOPSIS
    Start the LFX Insights local Web UI (single service: the API server).

.DESCRIPTION
    Launches the LFX Insights FastAPI server, which serves both the API
    (POST /api/insights) and the browser UI (GET /) from http://127.0.0.1:8000/.

    - Reuses an already-running, healthy API on port 8000 if present.
    - Reuses an already-running local Ollama; it NEVER starts another Ollama
      service (start Ollama separately if it is not running).
    - Does NOT start the Office Add-in HTTPS server.

    Open http://127.0.0.1:8000/ in a browser after this script reports ready.

.EXAMPLE
    .\start_web_ui.ps1
#>

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$Root = $ScriptDir

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  LFX Insights Web UI Launcher" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# Virtualenv Python if present, else PATH python.
if (Test-Path (Join-Path $Root ".venv312\Scripts\python.exe")) {
    $py = Resolve-Path (Join-Path $Root ".venv312\Scripts\python.exe")
} elseif (Test-Path (Join-Path $Root "venv\Scripts\python.exe")) {
    $py = Resolve-Path (Join-Path $Root "venv\Scripts\python.exe")
} else {
    $py = "python"
}

$env:PYTHONPATH = Join-Path $Root "src"

$PORT = 8000
$apiPid = $null
$startedHere = $false

function Test-ApiHealthy {
    $tmp = Join-Path $env:TEMP "lfx_webui_health.txt"
    try { & curl.exe -4 -s -m 2 --noproxy "*" -o $tmp "http://127.0.0.1:$PORT/health" 2>$null | Out-Null } catch {}
    if (-not (Test-Path $tmp)) { return $false }
    $body = Get-Content $tmp -Raw -ErrorAction SilentlyContinue
    return ($null -ne $body -and $body.Contains('"status"'))
}

# ---- 1. Ollama (reuse only) ------------------------------------------------
Write-Host "[1/2] Ollama" -ForegroundColor Cyan
$ollamaUp = $false
try {
    $t = Test-NetConnection -ComputerName 127.0.0.1 -Port 11434 -WarningAction SilentlyContinue
    $ollamaUp = $t.TcpTestSucceeded
} catch {}
if ($ollamaUp) {
    Write-Host "  [OK] Ollama already running on 127.0.0.1:11434 (reused)." -ForegroundColor Green
} else {
    Write-Host "  [SKIP] Ollama not running. This launcher will NOT start it." -ForegroundColor Yellow
    Write-Host "         Start it separately if needed:  ollama serve" -ForegroundColor Gray
}

# ---- 2. LFX Insights API (serves the Web UI too) --------------------------
Write-Host ""
Write-Host "[2/2] LFX Insights API" -ForegroundColor Cyan

if (Test-ApiHealthy) {
    Write-Host "  [OK] API already running and healthy at http://127.0.0.1:$PORT" -ForegroundColor Green
} else {
    Write-Host "  [..] Starting API on http://127.0.0.1:$PORT ..." -ForegroundColor Yellow
    $proc = Start-Process -FilePath $py `
        -ArgumentList "-m", "lfx_insights.api" `
        -WorkingDirectory $Root `
        -PassThru `
        -WindowStyle Minimized
    $apiPid = $proc.Id
    $startedHere = $true

    if ($proc.HasExited) {
        Write-Host "  [FAIL] API exited immediately (code $($proc.ExitCode))." -ForegroundColor Red
        exit 1
    }

    $ok = $false
    for ($i = 0; $i -lt 15; $i++) {
        Start-Sleep -Seconds 1
        if (Test-ApiHealthy) { $ok = $true; break }
    }
    if (-not $ok) {
        Write-Host "  [FAIL] API did not become healthy within 15s." -ForegroundColor Red
        if (-not $proc.HasExited) { Stop-Process -Id $apiPid -Force -ErrorAction SilentlyContinue }
        exit 1
    }
    Write-Host "  [OK] API ready at http://127.0.0.1:$PORT (PID $apiPid)" -ForegroundColor Green
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Open the LFX Insights Web UI at:" -ForegroundColor Green
Write-Host "    http://127.0.0.1:$PORT/" -ForegroundColor White
Write-Host "============================================" -ForegroundColor Cyan

if ($startedHere) {
    Write-Host ""
    Write-Host "Press Ctrl+C to stop the API started by this launcher." -ForegroundColor Gray
    try {
        while ($true) { Start-Sleep -Seconds 2 }
    } finally {
        if ($apiPid -and (Get-Process -Id $apiPid -ErrorAction SilentlyContinue)) {
            Write-Host "`nStopping API (PID $apiPid)..." -ForegroundColor Yellow
            Stop-Process -Id $apiPid -Force -ErrorAction SilentlyContinue
        }
    }
} else {
    Write-Host ""
    Write-Host "API was already running; this launcher will not stop it. Press Ctrl+C to exit." -ForegroundColor Gray
    try { while ($true) { Start-Sleep -Seconds 2 } } finally {}
}
