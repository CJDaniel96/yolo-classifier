@echo off
setlocal
cd /d "%~dp0"
set "TORCH_CHANNEL=cpu"
if not "%~1"=="" set "TORCH_CHANNEL=%~1"
if /I "%TORCH_CHANNEL%"=="cpu" goto channel_ok
if /I "%TORCH_CHANNEL%"=="cu130" goto channel_ok
echo Usage: setup.bat [cpu^|cu130]
exit /b 2

:channel_ok
if exist ".venv\Scripts\python.exe" goto install
py -3.12 -m venv .venv
if errorlevel 1 goto failed

:install
".venv\Scripts\python.exe" -c "import sys; assert sys.version_info[:2] == (3, 12), 'Python 3.12 is required'"
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install --upgrade --force-reinstall torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/%TORCH_CHANNEL%
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements-windows.lock
if errorlevel 1 goto failed
echo.
echo Setup complete. Place your OBB model in models\best.pt and edit config.json.
echo Set inference.device to cpu or 0 to match your selected runtime.
pause
exit /b 0

:failed
echo.
echo Setup failed. Review the error above. Install Python 3.12 with the py launcher first.
pause
exit /b 1
