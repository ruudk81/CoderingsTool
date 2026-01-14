@echo off
REM Windows setup script using uv (equivalent to setup.sh)
REM Usage: setup_windows.bat

set REQ_FILE=requirements.txt
set VENV_DIR=.venv
set PYVER_FILE=.python-version

echo [INFO] Project dir: %~dp0
cd /d %~dp0

REM Check if uv is available
where uv >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] uv not found on PATH
    exit /b 1
)

REM Read python version if file exists
set PYVER=
if exist %PYVER_FILE% (
    for /f "tokens=*" %%a in (%PYVER_FILE%) do set PYVER=%%a
)
echo [INFO] Python requested: %PYVER%

REM Create venv if it doesn't exist
if not exist %VENV_DIR%\Scripts\python.exe (
    if defined PYVER (
        echo [STEP] Creating venv with Python %PYVER%
        uv venv --python %PYVER% %VENV_DIR%
    ) else (
        echo [STEP] Creating venv with default Python
        uv venv %VENV_DIR%
    )
) else (
    echo [STEP] Venv exists: %VENV_DIR%
)

REM Install requirements
if exist %REQ_FILE% (
    echo [STEP] Installing requirements from %REQ_FILE%
    uv pip install --python %VENV_DIR%\Scripts\python.exe -r %REQ_FILE%
) else (
    echo [WARN] No %REQ_FILE% found; skipping dependency install.
)

REM Activate venv
echo [STEP] Activating virtual environment...
call %VENV_DIR%\Scripts\activate.bat
echo [DONE] Activated: %VENV_DIR%
echo.
echo To run the app: streamlit run src\app.py
