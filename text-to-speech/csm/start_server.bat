@echo off
REM Set environment variables for model caching
set HF_HOME=%~dp0model_cache
set TRANSFORMERS_CACHE=%~dp0model_cache
set TORCH_HOME=%~dp0model_cache

REM Create cache directory if it doesn't exist
if not exist "%~dp0model_cache" mkdir "%~dp0model_cache"

REM Activate virtual environment and run server
echo Activating virtual environment and starting server...
call "%~dp0..\venv\Scripts\activate.bat" && python "%~dp0server.py" %*
