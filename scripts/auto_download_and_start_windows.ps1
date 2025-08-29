<#
Auto-download a ggml model, install text-generation-webui on Windows, start the web UI, and optionally start the backend.

Usage:
  # Interactive mode (it will prompt for a model URL)
  .\scripts\auto_download_and_start_windows.ps1

  # Non-interactive (provide model URL)
  .\scripts\auto_download_and_start_windows.ps1 -ModelUrl 'https://example.com/models/ggml-small.bin' -StartBackend

Parameters:
- ModelUrl: direct HTTPS URL for a ggml model file. If omitted, the script will prompt for it.
- ModelFileName: optional filename to save model as (defaults to filename derived from URL).
- WebUiPort: port for the webui API (default 5000).
- StartBackend: switch, if present the script will set LOCAL_LLM_API for the user and attempt to start the backend (requires Python on PATH or will use system 'py').

Notes & safety:
- Model files are large (often GB). Only use URLs you trust and have rights to download.
- Run this script locally; I cannot run it for you.
- Run PowerShell as Administrator only if you need to write to C:\tools.
#>
param(
    [string]$ModelUrl = "",
    [string]$ModelFileName = "",
    [int]$WebUiPort = 5000,
    [switch]$StartBackend
)

function ErrExit($m) { Write-Host $m -ForegroundColor Red; exit 1 }
function Info($m) { Write-Host $m -ForegroundColor Cyan }

# Prompt for ModelUrl if missing
if (-not $ModelUrl -or $ModelUrl.Trim() -eq '') {
    $ModelUrl = Read-Host "Enter direct HTTPS URL to ggml model file (or press Enter to cancel)"
    if (-not $ModelUrl -or $ModelUrl.Trim() -eq '') { ErrExit 'No model URL provided. Aborting.' }
}

if (-not $ModelFileName -or $ModelFileName.Trim() -eq '') {
    $ModelFileName = [System.IO.Path]::GetFileName([Uri]$ModelUrl).Trim()
    if (-not $ModelFileName) { ErrExit 'Could not derive a filename from the URL; provide -ModelFileName explicitly.' }
}

$toolsDir = 'C:\tools\text-generation-webui'
$repoRoot = (Get-Location).ProviderPath

# Ensure git
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { ErrExit 'git is not available on PATH. Install Git for Windows and retry.' }

# Clone if missing
if (-not (Test-Path $toolsDir)) {
    Info "Cloning text-generation-webui into $toolsDir"
    try {
        $p = Start-Process -FilePath git -ArgumentList 'clone','https://github.com/oobabooga/text-generation-webui.git',$toolsDir -NoNewWindow -Wait -PassThru -ErrorAction Stop
        if ($p.ExitCode -ne 0) { ErrExit 'git clone failed.' }
    } catch {
        ErrExit "git clone failed: $($_.Exception.Message)"
    }
} else { Info "text-generation-webui already present at $toolsDir" }

Push-Location -Path $toolsDir
try {
    # venv
    if (-not (Test-Path '.venv')) {
        Info 'Creating Python virtualenv (.venv)'
        $venvOk = $false
        try {
            & python -m venv .venv
            $venvOk = $true
        } catch {}
        if (-not $venvOk) {
            try {
                & python3 -m venv .venv
                $venvOk = $true
            } catch {}
        }
        if (-not $venvOk) { ErrExit 'Failed to create virtualenv. Ensure Python is installed.' }
    }
    $pythonExe = Join-Path $toolsDir '.venv\Scripts\python.exe'
    if (-not (Test-Path $pythonExe)) { ErrExit "Virtualenv python not found at $pythonExe" }

    Info 'Upgrading pip and installing requirements... (this may take a while)'
    & $pythonExe -m pip install -U pip
    if (Test-Path 'requirements.txt') { & $pythonExe -m pip install -r requirements.txt }

    # models dir
    $modelsDir = Join-Path $toolsDir 'models'
    if (-not (Test-Path $modelsDir)) { New-Item -ItemType Directory -Path $modelsDir | Out-Null }

    $dest = Join-Path $modelsDir $ModelFileName
    if (Test-Path $dest) { Info "Model already exists at $dest. Skipping download." }
    else {
        Info "Downloading model from $ModelUrl to $dest (this may be many GB)..."
        try { Invoke-WebRequest -Uri $ModelUrl -OutFile $dest -UseBasicParsing -Verbose } catch { ErrExit "Download failed: $($_.Exception.Message)" }
        Info "Downloaded model to $dest"
    }

    # determine server script
    $serverScript = if (Test-Path (Join-Path $toolsDir 'server.py')) { 'server.py' } elseif (Test-Path (Join-Path $toolsDir 'launch.py')) { 'launch.py' } else { '' }
    if (-not $serverScript) { ErrExit 'No server entrypoint (server.py or launch.py) found in webui root.' }

    # start webui server
    $startArgs = @($serverScript, '--api', '--listen', '--port', $WebUiPort.ToString(), '--model', (Join-Path 'models' $ModelFileName))
    Info "Starting web UI with: $($pythonExe) $($startArgs -join ' ')"
    $proc = Start-Process -FilePath $pythonExe -ArgumentList $startArgs -WindowStyle Hidden -PassThru
    Start-Sleep -Seconds 3
    if ($proc.HasExited) { ErrExit "Web UI process exited unexpectedly (code $($proc.ExitCode)). Check logs in $toolsDir" }
    Info "Web UI started (PID $($proc.Id)). API should be at http://127.0.0.1:$WebUiPort"

    # set user env var for backend to call model
    $apiUrl = "http://127.0.0.1:$WebUiPort"
    Info "Setting LOCAL_LLM_API environment variable for current user to $apiUrl"
    setx LOCAL_LLM_API $apiUrl | Out-Null
    Info "Note: setx will take effect in newly opened shells. If you start backend from this shell, export the variable instead:"
    Write-Host "  `$
$env:LOCAL_LLM_API = '$apiUrl'" -ForegroundColor Yellow

    if ($StartBackend) {
        # try to start backend in repo root
        Info 'Attempting to start backend (uvicorn) in repository root using system python/py'
        Pop-Location
        Push-Location -Path $repoRoot
        try {
            $bproc = Start-Process -FilePath 'py' -ArgumentList '-m','uvicorn','backend.main:app','--reload','--port','8000' -WindowStyle Hidden -PassThru
            Start-Sleep -Seconds 2
            if ($bproc.HasExited) { ErrExit "Backend process exited (code $($bproc.ExitCode)). Check that Python and dependencies exist." }
            Info "Backend started (PID $($bproc.Id))."
        } catch {
            ErrExit "Failed to start backend: $($_.Exception.Message)"
        }
    }
} finally { Pop-Location }

Info 'Done. Test generation with:'
Write-Host "  Invoke-RestMethod -Method Post -Uri '$apiUrl/api/v1/generate' -ContentType 'application/json' -Body '{\"prompt\":\"hello\",\"max_new_tokens\":8}'"
Write-Host "Then check backend ping: Invoke-RestMethod -Method Get -Uri 'http://127.0.0.1:8000/api/llm/ping'"
