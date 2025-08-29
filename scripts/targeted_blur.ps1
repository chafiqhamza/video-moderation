<#
Targeted blur helper
- Backs up the original file before any edits
- Exports a short review clip covering 14.5–17.0s ±0.5s (from 14.0 to 17.5s)
- Produces a version with a box blur enabled only between 14.5 and 17.0 seconds
- Keeps audio untouched (stream-copied)
- Outputs timestamped files so you can compare and revert easily

Usage:
    powershell -ExecutionPolicy Bypass -File .\scripts\targeted_blur.ps1 -InputFile "C:\path\to\your\video.mp4"

If you prefer a different start/end, pass -StartSeconds and -EndSeconds.
#>
param(
    [Parameter(Mandatory=$true)][string]$InputFile,
    [double]$StartSeconds = 14.5,
    [double]$EndSeconds = 17.0,
    [double]$PadSeconds = 0.5
)

function Fail([string]$msg){ Write-Error $msg; exit 1 }

if (-not (Test-Path $InputFile)) { Fail "Input file not found: $InputFile" }

# Check ffmpeg availability
try {
    $ff = (& ffmpeg -version) 2>$null
} catch {
    Fail "ffmpeg not found in PATH. Please install ffmpeg and ensure it's on PATH."
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$full = Get-Item -LiteralPath $InputFile
$dir = $full.DirectoryName
$base = [System.IO.Path]::GetFileNameWithoutExtension($full.Name)
$ext = $full.Extension

# Compute review clip window with padding
 $reviewStart = [math]::Max(0, ($StartSeconds - $PadSeconds))
 $reviewDuration = ($EndSeconds - $StartSeconds) + (2 * $PadSeconds)

# Format numbers using invariant culture to ensure decimal point is '.' not ','
$culture = [System.Globalization.CultureInfo]::InvariantCulture
$reviewStartStr = [string]::Format($culture, '{0:F3}', $reviewStart)
$reviewDurationStr = [string]::Format($culture, '{0:F3}', $reviewDuration)
$startStr = [string]::Format($culture, '{0:F3}', $StartSeconds)
$endStr = [string]::Format($culture, '{0:F3}', $EndSeconds)

# Filenames (timestamped)
$backupPath = Join-Path $dir ("{0}.orig_{1}{2}" -f $base, $ts, $ext)
 # Use formatted strings (invariant culture) for filenames so decimals use '.'
 $reviewShortStart = [string]::Format($culture, '{0:F1}', $reviewStart)
 $reviewShortDur = [string]::Format($culture, '{0:F1}', $reviewDuration)
 $startShort = [string]::Format($culture, '{0:F2}', $StartSeconds)
 $endShort = [string]::Format($culture, '{0:F2}', $EndSeconds)
 $reviewPath = Join-Path $dir ("{0}_review_{1}s_{2}s_{3}" -f $base, $reviewShortStart, $reviewShortDur, "_$ts$ext")
 $blurPath = Join-Path $dir ("{0}_blur_{1}-{2}_{3}{4}" -f $base, $startShort, $endShort, $ts, $ext)

Write-Host "Input:    $InputFile"
Write-Host "Backup:   $backupPath"
Write-Host "Review:   $reviewPath (from $reviewStart for $reviewDuration seconds)"
Write-Host "Blurred:  $blurPath (box blur enabled between $StartSeconds and $EndSeconds seconds)"

# 1) Back up original file
Write-Host "Backing up original file..."
Copy-Item -LiteralPath $InputFile -Destination $backupPath -Force
if (-not (Test-Path $backupPath)) { Fail "Failed to create backup at $backupPath" }
Write-Host "Backup created."

# 2) Create review clip (fast seek using -ss before -i)
Write-Host "Creating review clip..."
 # Build argument array for ffmpeg to avoid PowerShell passing a single string
 $reviewArgs = @(
     '-y',
     '-ss', $reviewStartStr,
     '-i', $InputFile,
     '-t', $reviewDurationStr,
     '-c:v', 'libx264',
     '-preset', 'veryfast',
     '-crf', '18',
     '-c:a', 'copy',
     $reviewPath
 )
 $rc = & ffmpeg @reviewArgs
if ($LASTEXITCODE -ne 0) { Write-Warning "ffmpeg returned non-zero exit code for review clip ($LASTEXITCODE); check output above." } else { Write-Host "Review clip created: $reviewPath" }

# 3) Apply box blur only for the target interval, keep audio untouched
# Using the enable expression so the filter only runs between the times specified.
# boxblur parameters are tuned conservatively; you can increase the first number for heavier blur.
Write-Host "Applying targeted box blur..."
 # Build filter expression and argument array for blur operation
 $filterExpr = "boxblur=10:enable='between(t,$startStr,$endStr)'"
 $blurArgs = @(
     '-y',
     '-i', $InputFile,
     '-vf', $filterExpr,
     '-c:v', 'libx264',
     '-preset', 'veryfast',
     '-crf', '18',
     '-c:a', 'copy',
     $blurPath
 )
 $bc = & ffmpeg @blurArgs
if ($LASTEXITCODE -ne 0) { Write-Warning "ffmpeg returned non-zero exit code for blur operation ($LASTEXITCODE); check output above." } else { Write-Host "Blurred output created: $blurPath" }

# 4) Produce a small metadata log with the created paths
# PowerShell 5.1 doesn't support C-style ternary expressions; compute values explicitly.
$reviewFull = $null
if (Test-Path $reviewPath) { $reviewFull = (Get-Item -LiteralPath $reviewPath).FullName }
$blurFull = $null
if (Test-Path $blurPath) { $blurFull = (Get-Item -LiteralPath $blurPath).FullName }

$log = @{
    created = Get-Date -Format o
    input = (Get-Item -LiteralPath $InputFile).FullName
    backup = (Get-Item -LiteralPath $backupPath).FullName
    review = $reviewFull
    blurred = $blurFull
    start = $StartSeconds
    end = $EndSeconds
    pad = $PadSeconds
}
$logPath = Join-Path $dir ("{0}_edit_log_{1}.json" -f $base, $ts)
$log | ConvertTo-Json -Depth 3 | Out-File -FilePath $logPath -Encoding utf8
Write-Host "Log written to: $logPath"

Write-Host "Done. Inspect the review clip first. If it targets the right moment, inspect the blurred file next. If you prefer a cut instead of blur, re-run with the 'cut' variant as requested."
