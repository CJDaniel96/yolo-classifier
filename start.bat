@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Environment missing. Run setup.bat first.
    pause
    exit /b 2
)
".venv\Scripts\python.exe" -m aoi_service --config "%~dp0config.json" %*
set "SERVICE_EXIT=%ERRORLEVEL%"
if not "%SERVICE_EXIT%"=="0" (
    echo.
    echo Service stopped with exit code %SERVICE_EXIT%. See logs\service.log.
    pause
)
exit /b %SERVICE_EXIT%
