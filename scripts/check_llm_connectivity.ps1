<#
Lightweight connectivity checker for local LLM and backend.
Usage:
  # Basic test with defaults (LLM at 127.0.0.1:5000, backend at 127.0.0.1:8000)
  .\scripts\check_llm_connectivity.ps1

  # Custom endpoints / timeout
  .\scripts\check_llm_connectivity.ps1 -LLMApiUrl 'http://127.0.0.1:5000' -BackendUrl 'http://127.0.0.1:8000' -TimeoutSec 5

This script:
- POSTs a tiny prompt to the LLM API endpoint (/api/v1/generate) to check reachability.
- GETs backend /api/llm/ping to confirm backend can reach the LLM.
- Prints clear troubleshooting hints and exits with non-zero status on failure.

Notes:
- Uses .NET HttpClient to set a short timeout that works on Windows PowerShell 5.1 and PowerShell 7.
- Adjust URLs to match your web UI version if endpoints differ.
#>
param(
    [string]$LLMApiUrl = "http://127.0.0.1:5000",
    [string]$BackendUrl = "http://127.0.0.1:8000",
    [int]$TimeoutSec = 5
)

function Write-Info($m) { Write-Host $m -ForegroundColor Cyan }
function Write-Success($m) { Write-Host $m -ForegroundColor Green }
function Write-Warn($m) { Write-Host $m -ForegroundColor Yellow }
function Write-Err($m) { Write-Host $m -ForegroundColor Red }

# Helper: POST small JSON to LLM generate endpoint
function Test-LLM {
    param($url, $timeout)
    $uri = "$url/api/v1/generate"
    Write-Info "Testing LLM API at: $uri (timeout=${timeout}s)"
    # Try Invoke-RestMethod first (works on PS5.1 and PS7)
    try {
        $bodyObj = @{ prompt = 'ping'; max_new_tokens = 8 }
        $json = ConvertTo-Json $bodyObj -Depth 5
        $resp = Invoke-RestMethod -Method Post -Uri $uri -ContentType 'application/json' -Body $json -ErrorAction Stop
        $preview = $resp | Out-String
        if ($preview.Length -gt 400) { $preview = $preview.Substring(0,400) + '...' }
        Write-Success "LLM API reachable. Sample response (truncated):"
        Write-Host $preview
        return $true
    } catch {
        # Fallback: try HttpClient if available
        try {
            $client = New-Object System.Net.Http.HttpClient
            $client.Timeout = [System.TimeSpan]::FromSeconds($timeout)
            $content = New-Object System.Net.Http.StringContent($json, [System.Text.Encoding]::UTF8, 'application/json')
            $task = $client.PostAsync($uri, $content)
            $resp = $task.GetAwaiter().GetResult()
            $respBody = $resp.Content.ReadAsStringAsync().GetAwaiter().GetResult()
            if ($resp.IsSuccessStatusCode) {
                Write-Success "LLM API reachable (status: $($resp.StatusCode)). Sample response (truncated):"
                $preview = $respBody
                if ($preview.Length -gt 400) { $preview = $preview.Substring(0,400) + '...' }
                Write-Host $preview
                return $true
            } else {
                Write-Err "LLM API returned status $($resp.StatusCode). Response: $respBody"
                return $false
            }
        } catch {
            Write-Err "LLM API check failed: $($_.Exception.Message)"
            return $false
        }
    }
}

# Helper: GET backend ping endpoint
function Test-BackendPing {
    param($url, $timeout)
    $uri = "$url/api/llm/ping"
    Write-Info "Testing backend ping at: $uri (timeout=${timeout}s)"
    try {
        $resp = Invoke-RestMethod -Method Get -Uri $uri -ErrorAction Stop
        Write-Success "Backend ping OK. Response:"
        Write-Host ($resp | Out-String)
        return $true
    } catch {
        # Fallback to HttpClient if needed
        try {
            $client = New-Object System.Net.Http.HttpClient
            $client.Timeout = [System.TimeSpan]::FromSeconds($timeout)
            $task = $client.GetAsync($uri)
            $resp = $task.GetAwaiter().GetResult()
            $respBody = $resp.Content.ReadAsStringAsync().GetAwaiter().GetResult()
            if ($resp.IsSuccessStatusCode) {
                Write-Success "Backend ping OK (status: $($resp.StatusCode)). Response:"
                Write-Host $respBody
                return $true
            } else {
                Write-Err "Backend ping returned status $($resp.StatusCode). Response: $respBody"
                return $false
            }
        } catch {
            Write-Err "Backend ping failed: $($_.Exception.Message)"
            return $false
        }
    }
}

Write-Info "Checking local LLM and backend connectivity..."
$okLLM = Test-LLM -url $LLMApiUrl -timeout $TimeoutSec
if (-not $okLLM) {
    Write-Warn "LLM seems unreachable. Common fixes:"
    Write-Warn " - Ensure web UI or model server is running (text-generation-webui or GPT4All server)."
    Write-Warn " - If running in WSL, ensure server started with --listen and port forwarded (127.0.0.1:$($LLMApiUrl.Split(':')[-1]))."
    Write-Warn " - Check firewall or that the server actually listens on the configured port."
}

$okBackend = Test-BackendPing -url $BackendUrl -timeout $TimeoutSec
if (-not $okBackend) {
    Write-Warn "Backend ping failed. Common fixes:"
    Write-Warn " - Ensure backend is running on port 8000 and the app's /api/llm/ping endpoint is available."
    Write-Warn " - If you changed LOCAL_LLM_API, make sure backend restarted with the correct env var."
}

if ($okLLM -and $okBackend) {
    Write-Success "All checks passed. Your frontend should detect the LLM as available now."
    exit 0
} else {
    Write-Err "One or more checks failed. Fix the issues above and retry."
    exit 2
}
