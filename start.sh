#!/usr/bin/env bash
# macOS/Linux equivalent of AVVIA_LOCALE.bat.
# Usage: ./start.sh [--mode normal|naruto] [--no-virtual-camera] [...]
set -euo pipefail
cd "$(dirname "$0")"

VENV_DIR=".venv"
VENV_PY="$VENV_DIR/bin/python"

if [ ! -x "$VENV_PY" ]; then
  # MediaPipe has no wheels for very new Pythons, so prefer 3.12/3.13.
  PYTHON=""
  for candidate in python3.12 python3.13 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON="$candidate"
      break
    fi
  done
  if [ -z "$PYTHON" ]; then
    echo "[Vanish Lab] Python 3.12 not found. Install it with: brew install python@3.12" >&2
    exit 1
  fi
  echo "[Vanish Lab] Creating the local Python environment with $PYTHON..."
  "$PYTHON" -m venv "$VENV_DIR"
fi

if ! "$VENV_PY" -c "import cv2, mediapipe, numpy, pyvirtualcam, pygame" >/dev/null 2>&1; then
  echo "[Vanish Lab] Installing dependencies (internet needed only this time)..."
  "$VENV_PY" -m pip install --upgrade pip
  "$VENV_PY" -m pip install -r requirements.txt
fi

"$VENV_PY" -c "from vanish_cam import download_models; download_models()"

echo "[Vanish Lab] Starting the webcam..."
exec "$VENV_PY" vanish_cam.py "$@"
