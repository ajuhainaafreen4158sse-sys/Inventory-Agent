@echo off
REM Switch to the directory where this script is located
cd /d "%~dp0"

echo [1/2] Checking dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install dependencies.
    pause
    exit /b %errorlevel%
)

echo [2/2] Starting Dental Agents Worker...
REM Run the worker module. We are in the parent directory of 'dental_agents' package.
python -m dental_agents.worker

pause
