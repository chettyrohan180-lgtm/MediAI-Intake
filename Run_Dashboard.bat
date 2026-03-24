@echo off
color 0b
title MediAI Hospital System
echo ==============================================
echo    MediAI Hospital System - Web Dashboard
echo ==============================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in your system PATH.
    echo Please install Python 3.9 or higher to run this project.
    pause
    exit /b
)

:: Create Virtual Environment if it doesn't exist
if not exist .\.venv (
    echo [1/3] Creating a private Python virtual environment...
    python -m venv .venv
)

:: Activate the environment
echo [2/3] Activating virtual environment...
call .\.venv\Scripts\activate.bat

:: Install Requirements
echo [3/3] Installing dependencies (this may take a moment on first run)...
python -m pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt >nul 2>&1

echo.
echo ==============================================
echo           Starting Web Application
echo ==============================================
echo.
echo [INFO] The AI server is launching. 
echo [INFO] A new browser window should pop up automatically!
echo [INFO] If it doesn't open, manually go to: http://localhost:8000
echo.
echo Press CTRL+C in this window at any time to turn off the server.

uvicorn api:app

pause
