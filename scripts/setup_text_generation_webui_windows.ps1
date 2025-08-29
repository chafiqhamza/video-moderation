<#
PowerShell helper: Install and optionally start text-generation-webui natively on Windows with a local ggml model.
Run this in an elevated PowerShell if you need permission to create C:\tools.

Usage examples:
# Prepare environment and do NOT start the server
.
# .\scripts\setup_text_generation_webui_windows.ps1 -ModelUrl '' -ModelFileName '' -Port 5000

# Prepare, download model, and start server
.
# .\scripts\setup_text_generation_webui_windows.ps1 -ModelUrl 'https://example.com/models/ggml-small.bin' -ModelFileName 'ggml-small.bin' -StartServer

Parameters:
- ModelUrl (optional): direct HTTPS URL to a ggml model file to download into the webui models directory.
- ModelFileName (optional): filename to save the downloaded model as (defaults to filename part of ModelUrl).
- Port (optional): port to start the API server on (default: 5000).
- StartServer (switch): if present, will try to start the server after setup.

Notes:
- Model files are large. Ensure you have disk space and a fast/paid connection if required.
- Only download models you are authorized to use.
- If download fails or you don't provide a model URL, you can manually place a model in C:\tools\text-generation-webui\models\
#>
param(
    [string]$ModelUrl = "",
    [string]$ModelFileName = "",
    [int]$Port = 5000,
    [switch]$StartServer
)

function ErrExit($msg) {
    Write-Host $msg -ForegroundColor Red
    exit 1
}

$toolsDir = 'C:\tools\text-generation-webui'

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    ErrExit 'git is not available on PATH. Install Git for Windows and retry.'
}

if (-not (Test-Path $toolsDir)) {
    Write-Host "Cloning text-generation-webui into $toolsDir..."
    git clone https://github.com/oobabooga/text-generation-webui.git $toolsDir
} else {
    Write-Host "text-generation-webui already exists at $toolsDir"
}

Push-Location -Path $toolsDir
try {
    if (-not (Test-Path '.venv')) {
        Write-Host 'Creating Python virtual environment (.venv)...'
        python -m venv .venv
    }

    $pythonExe = Join-Path $toolsDir '.venv\Scripts\python.exe'
    if (-not (Test-Path $pythonExe)) {
        ErrExit "Python executable not found in virtualenv at $pythonExe. Ensure Python is installed and available as 'python'."
    }

    Write-Host 'Upgrading pip and installing requirements...'
    & $pythonExe -m pip install -U pip
    if (Test-Path 'requirements.txt') {
        & $pythonExe -m pip install -r requirements.txt
    }

    # Ensure models dir
    $modelsDir = Join-Path $toolsDir 'models'
    if (-not (Test-Path $modelsDir)) { New-Item -ItemType Directory -Path $modelsDir | Out-Null }

    # Download model if URL provided
    if ($ModelUrl -and $ModelUrl.Trim() -ne '') {
        if (-not $ModelFileName -or $ModelFileName.Trim() -eq '') {
            $ModelFileName = Split-Path -Leaf $ModelUrl
            if (-not $ModelFileName) { ErrExit 'Could not determine a filename from ModelUrl; provide -ModelFileName explicitly.' }
        }
        $dest = Join-Path $modelsDir $ModelFileName
        if (Test-Path $dest) {
            Write-Host "Model already exists at $dest. Skipping download."
        } else {
            Write-Host "Downloading model to $dest (this may take a while)..."
            try {
                Invoke-WebRequest -Uri $ModelUrl -OutFile $dest -UseBasicParsing -Verbose
                Write-Host "Downloaded model to $dest"
            } catch {
                ErrExit "Model download failed: $($_.Exception.Message)"
            }
        }
    } else {
        Write-Host 'No ModelUrl provided. Place a ggml model file into the models directory manually when ready.'
    }

    # Decide which server script exists
    $serverScript = ''
    if (Test-Path (Join-Path $toolsDir 'server.py')) { $serverScript = 'server.py' }
    elseif (Test-Path (Join-Path $toolsDir 'launch.py')) { $serverScript = 'launch.py' }

    if ($StartServer) {
        if (-not $serverScript) {
            ErrExit 'No server script (server.py or launch.py) found in webui root; cannot start server.'
        }
        if (-not $ModelFileName -or -not (Test-Path (Join-Path $modelsDir $ModelFileName))) {
            Write-Host 'Warning: model file not present in models/. Server may fail to start. Proceeding anyway.' -ForegroundColor Yellow
        }

        $serverArgs = @()
        if ($serverScript -eq 'server.py') {
            $serverArgs += 'server.py'
        } else {
            $serverArgs += 'launch.py'
        }
    # Build args as separate elements (avoid stray semicolons or stray quoted lines)
    $serverArgs += '--api'
        $serverArgs += '--listen'
        $serverArgs += '--port'
        $serverArgs += [string]$Port
        if ($ModelFileName) {
            $serverArgs += '--model'
            $serverArgs += (Join-Path 'models' $ModelFileName)
        }

        Write-Host "Starting web UI server using $pythonExe with args: $($serverArgs -join ' ')"
        try {
            $proc = Start-Process -FilePath $pythonExe -ArgumentList $serverArgs -WindowStyle Hidden -PassThru
            Write-Host "Started server (PID: $($proc.Id)). Check logs or the process list to confirm."
            Write-Host "API should be reachable at http://127.0.0.1:$Port from this machine."
            # Persist LOCAL_LLM_API so the backend can pick it up in new shells
            try {
                setx LOCAL_LLM_API "http://127.0.0.1:$Port" | Out-Null
                Write-Host "Set LOCAL_LLM_API to http://127.0.0.1:$Port (applies to new shells)."
            } catch {
                Write-Host "Could not set persistent env var LOCAL_LLM_API: $($_.Exception.Message)" -ForegroundColor Yellow
            }
        } catch {
            Write-Host "Failed to start web UI server: $($_.Exception.Message)" -ForegroundColor Red
            Write-Host "You can start the server manually by activating the venv and running the server script:" 
            Write-Host "  .\.venv\Scripts\Activate.ps1"
            if ($serverScript) { Write-Host "  python $serverScript --model models/<model-file> --api --listen --port $Port" } 
            else { Write-Host "  python server.py --model models/<model-file> --api --listen --port $Port" }
        }
    } else {
        Write-Host "Setup complete. To start the server, activate the venv and run (from $toolsDir):"
        Write-Host "  .\.venv\Scripts\Activate.ps1"
        if ($serverScript) {
            Write-Host "  python $serverScript --model models/<model-file> --api --listen --port $Port"
        } else {
            Write-Host "  python server.py --model models/<model-file> --api --listen --port $Port"
        }
    }
} finally {
    Pop-Location
}

Write-Host 'Finished.'
