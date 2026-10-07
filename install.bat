@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if not errorlevel 1 (
    py -3.12 -m venv .venv
    if errorlevel 1 py -3.11 -m venv .venv
) else (
    where python >nul 2>nul
    if errorlevel 1 goto missing_python
    python -m venv .venv
)
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
rem The official Windows PyTorch wheel supports CPU and NVIDIA CUDA.
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" setup_models.py
if errorlevel 1 goto failed
echo.
echo Installation complete. Run run.bat to start the application.
pause
exit /b 0
:missing_python
echo Install 64-bit Python 3.11 or 3.12 from python.org and enable Add Python to PATH.
pause
exit /b 1
:failed
echo Installation failed. Check the message above and retry install.bat.
pause
exit /b 1
