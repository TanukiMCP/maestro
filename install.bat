@echo off
echo ==============================================
echo   🎭 MAESTRO MCP SERVER - EASY INSTALLER
echo ==============================================
echo.
echo This will install all required dependencies...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo ✅ Python found!
python --version

REM Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ ERROR: pip is not available
    echo Please reinstall Python with pip included
    pause
    exit /b 1
)

echo ✅ pip found!

REM Install dependencies
echo.
echo 📦 Installing dependencies...
echo This may take a few minutes...
echo.

pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ❌ ERROR: Failed to install dependencies
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

echo.
echo ✅ Installation completed successfully!
echo.
echo 🚀 You can now run the server with:
echo    python run.py
echo.
echo Or double-click the "start-server.bat" file
echo.
pause 