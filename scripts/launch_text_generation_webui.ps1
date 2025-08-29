<#
PowerShell helper: launch the text-generation-webui web UI with API enabled on port 5000.
#>
$toolsDir = "C:\tools\text-generation-webui"
if (-Not (Test-Path $toolsDir)) {
    Write-Host "Directory $toolsDir not found. Run setup_text_generation_webui.ps1 first."
    exit 1
}
Push-Location -Path $toolsDir
if (-Not (Test-Path ".\venv\Scripts\Activate.ps1")) {
    Write-Host "Virtual environment activate script not found. Did setup complete successfully?"
    Pop-Location
    exit 1
}

Write-Host "Activating virtual environment and launching web UI..."
.\venv\Scripts\Activate.ps1
python launch.py --listen --api --port 5000
Pop-Location
