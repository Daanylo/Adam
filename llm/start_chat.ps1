# PowerShell script to run the AI Chat Tool
param(
    [string]$Model = "llama3.2:1b",
    [string]$OllamaUrl = "http://localhost:11434",
    [string]$TtsUrl = "http://localhost:5000",
    [switch]$NoAudio,
    [switch]$ListModels,
    [switch]$Help
)

if ($Help) {
    Write-Host "AI Chat Tool - PowerShell Launcher" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor Yellow
    Write-Host "  .\start_chat.ps1 [options]" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  -Model <name>        Ollama model to use (default: llama3.2)" -ForegroundColor Gray
    Write-Host "  -OllamaUrl <url>     Ollama API URL (default: http://localhost:11434)" -ForegroundColor Gray
    Write-Host "  -TtsUrl <url>        TTS API URL (default: http://localhost:5000)" -ForegroundColor Gray
    Write-Host "  -NoAudio             Disable audio playback" -ForegroundColor Gray
    Write-Host "  -ListModels          List available Ollama models" -ForegroundColor Gray
    Write-Host "  -Help                Show this help message" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\start_chat.ps1                           # Use default settings" -ForegroundColor Gray
    Write-Host "  .\start_chat.ps1 -Model 'llama3.1'        # Use specific model" -ForegroundColor Gray
    Write-Host "  .\start_chat.ps1 -NoAudio                 # Disable audio" -ForegroundColor Gray
    Write-Host "  .\start_chat.ps1 -ListModels              # List models" -ForegroundColor Gray
    exit 0
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ChatScript = Join-Path $ScriptDir "ai_chat_tool.py"
$RequirementsFile = Join-Path $ScriptDir "ai_chat_requirements.txt"

# Check if chat script exists
if (-not (Test-Path $ChatScript)) {
    Write-Host " Chat script not found: $ChatScript" -ForegroundColor Red
    exit 1
}

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host " Found Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host " Python not found. Please install Python and add it to PATH." -ForegroundColor Red
    exit 1
}

# Install requirements if file exists
if (Test-Path $RequirementsFile) {
    Write-Host " Installing/checking requirements..." -ForegroundColor Cyan
    python -m pip install -r $RequirementsFile --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Host " Warning: Some packages may not have installed correctly" -ForegroundColor Yellow
    }
}

# Build command arguments
$commandArgs = @()
$commandArgs += "--model", $Model
$commandArgs += "--ollama-url", $OllamaUrl
$commandArgs += "--tts-url", $TtsUrl

if ($NoAudio) {
    $commandArgs += "--no-audio"
}

if ($ListModels) {
    $commandArgs += "--list-models"
}

# Display startup info
if (-not $ListModels) {
    Write-Host ""
    Write-Host " Starting AI Chat Tool..." -ForegroundColor Green
    Write-Host "   Model: $Model" -ForegroundColor Cyan
    Write-Host "   Ollama: $OllamaUrl" -ForegroundColor Cyan
    Write-Host "   TTS: $TtsUrl" -ForegroundColor Cyan
    $audioStatus = if ($NoAudio) { 'Disabled' } else { 'Enabled' }
    Write-Host "   Audio: $audioStatus" -ForegroundColor Cyan
    Write-Host ""
}

# Run the chat tool
python $ChatScript @commandArgs
