# PowerShell script to start the TTS server with caching
param(
    [string]$Device = "cuda",
    [string]$Text = $null
)

# Create cache directory for future use
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$CacheDir = Join-Path $ScriptDir "model_cache"

if (-not (Test-Path $CacheDir)) {
    New-Item -ItemType Directory -Path $CacheDir -Force | Out-Null
}

# Activate virtual environment
$VenvActivate = Join-Path (Split-Path -Parent $ScriptDir) "venv\Scripts\Activate.ps1"

if (Test-Path $VenvActivate) {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & $VenvActivate
} else {
    Write-Host "Virtual environment not found at $VenvActivate" -ForegroundColor Red
    Write-Host "Make sure you have a 'venv' folder in the parent directory." -ForegroundColor Red
    exit 1
}

# Run the server
$ServerScript = Join-Path $ScriptDir "server.py"

if ($Text) {
    Write-Host "Running in single-generation mode..." -ForegroundColor Green
    python $ServerScript --device $Device --text $Text
} else {
    Write-Host "Starting interactive TTS server..." -ForegroundColor Green
    python $ServerScript --device $Device
}
