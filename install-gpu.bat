@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run install.bat first.
  pause
  exit /b 1
)
echo Installing CuPy CUDA backend...
.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
if errorlevel 1 (
  echo GPU backend installation failed.
  pause
  exit /b 1
)
echo.
echo GPU backend installed. Checking device...
.venv\Scripts\python.exe -m pyllama device
pause
