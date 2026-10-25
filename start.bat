@echo off
rem Start VanishCam on Windows (macOS/Linux: start.sh).
rem Usage: start.bat [--mode normal^|goku] [--blur-background] [--no-virtual-camera] [...]
setlocal
cd /d "%~dp0"

set "VENV_PY=.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
  echo [VanishCam] Creating the local Python environment...
  rem MediaPipe has no wheels for very new Pythons, so prefer 3.12.
  where py >nul 2>nul
  if not errorlevel 1 (
    py -3.12 -m venv .venv 2>nul || py -3 -m venv .venv
  ) else (
    python -m venv .venv
  )
  if errorlevel 1 goto :error
)

"%VENV_PY%" -c "import cv2, mediapipe, numpy, pyvirtualcam" >nul 2>nul
if errorlevel 1 (
  echo [VanishCam] Installing dependencies ^(internet needed only this time^)...
  "%VENV_PY%" -m pip install --upgrade pip
  if errorlevel 1 goto :error
  "%VENV_PY%" -m pip install -r requirements.txt
  if errorlevel 1 goto :error
)

"%VENV_PY%" -c "from vanish_cam import download_models; download_models()"
if errorlevel 1 goto :error

echo [VanishCam] Starting the webcam...
"%VENV_PY%" vanish_cam.py %*
if errorlevel 1 goto :error
exit /b 0

:error
echo.
echo [VanishCam] Could not start. See README.md for common fixes.
pause
exit /b 1
