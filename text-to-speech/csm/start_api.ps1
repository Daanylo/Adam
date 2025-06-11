# PowerShell script to start the TTS API server
param(
    [string]$Device = "cuda",
    [string]$ApiHost = "127.0.0.1",
    [int]$Port = 5000,
    [string]$OutputDir = "./audio_output",
    [switch]$Debug
)

# Create output directory for audio files
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$OutputPath = Join-Path $ScriptDir $OutputDir

if (-not (Test-Path $OutputPath)) {
    New-Item -ItemType Directory -Path $OutputPath -Force | Out-Null
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

# Install Flask and Flask-CORS if not already installed
Write-Host "Checking dependencies..." -ForegroundColor Cyan
pip install flask flask-cors --quiet

# Run the API server
$ApiScript = Join-Path $ScriptDir "api_server.py"

Write-Host "Starting TTS API server..." -ForegroundColor Green
Write-Host "API will be available at: http://$ApiHost`:$Port" -ForegroundColor Cyan
Write-Host "Audio files will be saved to: $OutputPath" -ForegroundColor Cyan
Write-Host ""
Write-Host "API Endpoints:" -ForegroundColor Yellow
Write-Host "  GET  /health        - Health check" -ForegroundColor Gray
Write-Host "  GET  /ready         - Check if model is ready" -ForegroundColor Gray
Write-Host "  POST /generate      - Generate speech from text" -ForegroundColor Gray
Write-Host "  GET  /audio/<file>  - Download audio file" -ForegroundColor Gray
Write-Host "  GET  /files         - List all audio files" -ForegroundColor Gray
Write-Host "  GET  /info          - Server information" -ForegroundColor Gray
Write-Host ""

if ($Debug) {
    python $ApiScript --device $Device --host $ApiHost --port $Port --output-dir $OutputDir --debug
} else {
    python $ApiScript --device $Device --host $ApiHost --port $Port --output-dir $OutputDir
}
