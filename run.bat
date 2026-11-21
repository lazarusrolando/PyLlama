\
@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title PyLlama Local Runtime

if not exist ".venv\Scripts\python.exe" (
  echo.
  echo [PyLlama] Creating virtual environment...
  py -3 -m venv .venv
  if errorlevel 1 (
    echo Failed to create virtual environment. Install Python 3.10+ first.
    pause
    exit /b 1
  )
  echo [PyLlama] Installing CPU/API dependencies...
  .venv\Scripts\python.exe -m pip install --upgrade pip
  .venv\Scripts\python.exe -m pip install -r requirements.txt
  if errorlevel 1 (
    echo Dependency installation failed.
    pause
    exit /b 1
  )
)

:MENU
cls
echo ============================================================
echo                         PyLlama
echo              Local GGUF Runtime - CPU + GPU
echo ============================================================
echo.
.venv\Scripts\python.exe -m pyllama device
echo.
echo Models are NEVER downloaded by PyLlama.
echo Manually place your .gguf files inside the models\ folder.
echo.
echo [1] Open models folder
echo [2] List installed models
echo [3] Inspect GGUF model
echo [4] Run CLI chat (AUTO)
echo [5] Run CLI chat (CPU)
echo [6] Run CLI chat (GPU/CUDA)
echo [7] Start API server (AUTO)
echo [8] Start API server (GPU/CUDA)
echo [9] Generate from prompt
echo [0] Exit
echo.
set /p CHOICE=Select: 
if "%CHOICE%"=="1" goto OPEN_MODELS
if "%CHOICE%"=="2" goto LIST
if "%CHOICE%"=="3" goto INSPECT
if "%CHOICE%"=="4" (
  set DEVICE=auto
  goto CHAT
)
if "%CHOICE%"=="5" (
  set DEVICE=cpu
  goto CHAT
)
if "%CHOICE%"=="6" (
  set DEVICE=cuda
  goto CHAT
)
if "%CHOICE%"=="7" (
  set DEVICE=auto
  goto SERVER
)
if "%CHOICE%"=="8" (
  set DEVICE=cuda
  goto SERVER
)
if "%CHOICE%"=="9" goto GENERATE
if "%CHOICE%"=="0" exit /b 0
goto MENU

:OPEN_MODELS
if not exist "models" mkdir "models"
start "" explorer "%CD%\models"
pause
goto MENU

:LIST
.venv\Scripts\python.exe -m pyllama list
pause
goto MENU

:INSPECT
.venv\Scripts\python.exe -m pyllama list
echo.
set /p MODEL=Installed model name/path: 
.venv\Scripts\python.exe -m pyllama inspect --model "%MODEL%"
pause
goto MENU

:CHAT
.venv\Scripts\python.exe -m pyllama list
echo.
set /p MODEL=Installed model name/path: 
.venv\Scripts\python.exe -m pyllama chat --model "%MODEL%" --dtype float16 --device %DEVICE%
pause
goto MENU

:SERVER
.venv\Scripts\python.exe -m pyllama list
echo.
set /p MODEL=Installed model name/path: 
.venv\Scripts\python.exe -m pyllama serve --model "%MODEL%" --dtype float16 --device %DEVICE% --port 8080
pause
goto MENU

:GENERATE
.venv\Scripts\python.exe -m pyllama list
echo.
set /p MODEL=Installed model name/path: 
set /p PROMPT=Prompt: 
set /p DEVICE=Device [auto/cpu/cuda]: 
if "%DEVICE%"=="" set DEVICE=auto
.venv\Scripts\python.exe -m pyllama generate --model "%MODEL%" --dtype float16 --device %DEVICE% --prompt "%PROMPT%" --max-tokens 128
pause
goto MENU
