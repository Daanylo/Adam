# Setup script for AI Chat Tool
param(
    [switch]$InstallPython,
    [switch]$InstallOllama,
    [switch]$Help
)

if ($Help) {
    Write-Host "AI Chat Tool Setup" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "This script helps set up the AI Chat Tool environment." -ForegroundColor Gray
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor Yellow
    Write-Host "  .\setup.ps1 [options]" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "  -InstallPython       Show Python installation instructions" -ForegroundColor Gray
    Write-Host "  -InstallOllama       Show Ollama installation instructions" -ForegroundColor Gray
    Write-Host "  -Help                Show this help message" -ForegroundColor Gray
    exit 0
}

Write-Host "🔧 AI Chat Tool Setup" -ForegroundColor Cyan
Write-Host ""

# Check Python
Write-Host "1. Checking Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "   ✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "   ❌ Python not found" -ForegroundColor Red
    if ($InstallPython) {
        Write-Host ""
        Write-Host "   Python Installation:" -ForegroundColor Cyan
        Write-Host "   1. Go to https://www.python.org/downloads/" -ForegroundColor Gray
        Write-Host "   2. Download and install Python 3.8 or later" -ForegroundColor Gray
        Write-Host "   3. Make sure to check 'Add Python to PATH' during installation" -ForegroundColor Gray
        Write-Host "   4. Restart PowerShell after installation" -ForegroundColor Gray
    }
}

# Check Ollama
Write-Host ""
Write-Host "2. Checking Ollama..." -ForegroundColor Yellow
try {
    $ollamaCheck = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
    Write-Host "   ✅ Ollama is running" -ForegroundColor Green
    
    # Parse and display models
    $response = $ollamaCheck.Content | ConvertFrom-Json
    if ($response.models -and $response.models.Count -gt 0) {
        Write-Host "   📦 Available models:" -ForegroundColor Cyan
        foreach ($model in $response.models) {
            Write-Host "      - $($model.name)" -ForegroundColor Gray
        }
    } else {
        Write-Host "   ⚠️  No models found. You may need to pull a model first." -ForegroundColor Yellow
        Write-Host "      Example: ollama pull llama3.2" -ForegroundColor Gray
    }
} catch {
    Write-Host "   ❌ Ollama not running or not installed" -ForegroundColor Red
    if ($InstallOllama) {
        Write-Host ""
        Write-Host "   Ollama Installation:" -ForegroundColor Cyan
        Write-Host "   1. Go to https://ollama.com/download" -ForegroundColor Gray
        Write-Host "   2. Download and install Ollama for Windows" -ForegroundColor Gray
        Write-Host "   3. After installation, open a new terminal and run:" -ForegroundColor Gray
        Write-Host "      ollama pull llama3.2" -ForegroundColor Gray
        Write-Host "   4. Wait for the model to download" -ForegroundColor Gray
    }
}

# Install Python requirements
Write-Host ""
Write-Host "3. Installing Python requirements..." -ForegroundColor Yellow
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RequirementsFile = Join-Path $ScriptDir "ai_chat_requirements.txt"

if (Test-Path $RequirementsFile) {
    try {
        python -m pip install -r $RequirementsFile
        Write-Host "   ✅ Python packages installed" -ForegroundColor Green
    } catch {
        Write-Host "   ❌ Failed to install Python packages" -ForegroundColor Red
        Write-Host "   Try running: python -m pip install requests pygame" -ForegroundColor Gray
    }
} else {
    Write-Host "   ⚠️  Requirements file not found" -ForegroundColor Yellow
}

# Check TTS API
Write-Host ""
Write-Host "4. Checking TTS API..." -ForegroundColor Yellow
try {
    $ttsCheck = Invoke-WebRequest -Uri "http://localhost:5000/health" -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
    Write-Host "   ✅ TTS API is running" -ForegroundColor Green
} catch {
    Write-Host "   ❌ TTS API not running" -ForegroundColor Red
    Write-Host "   Make sure to start your TTS server first:" -ForegroundColor Gray
    Write-Host "   .\text-to-speech\csm\start_api.ps1" -ForegroundColor Gray
}

Write-Host ""
Write-Host "🏁 Setup Summary:" -ForegroundColor Cyan
Write-Host "   Once all components are running, you can start the chat tool with:" -ForegroundColor Gray
Write-Host "   .\start_chat.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "   For help with the chat tool:" -ForegroundColor Gray
Write-Host "   .\start_chat.ps1 -Help" -ForegroundColor Yellow
