@echo off
echo ==============================================
echo   🎭 MAESTRO MCP SERVER - STARTING...
echo ==============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ ERROR: Python is not installed or not in PATH
    echo Please run install.bat first
    pause
    exit /b 1
)

echo ✅ Python found!
echo.
echo 🚀 Starting MAESTRO MCP Server...
echo.
echo Server will be available at: http://localhost:8000
echo Press Ctrl+C to stop the server
echo.

python run.py 