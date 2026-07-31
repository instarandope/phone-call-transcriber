@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo  ============================================================
echo   Call Transcriber - setup
echo  ============================================================
echo.

REM ---------------------------------------------------------------- Python --
set PY=
where py >nul 2>&1 && set PY=py -3
if not defined PY (
    where python >nul 2>&1 && set PY=python
)
if not defined PY (
    echo  [XX] Python was not found.
    echo.
    echo       Install Python 3.11 or newer from https://www.python.org/downloads/
    echo       On the first screen of the installer, tick "Add python.exe to PATH".
    echo.
    pause
    exit /b 1
)

%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if errorlevel 1 (
    echo  [XX] Python 3.11 or newer is required.
    %PY% --version
    echo.
    pause
    exit /b 1
)
for /f "delims=" %%v in ('%PY% --version 2^>^&1') do echo  [ok] %%v

REM ------------------------------------------------------------ virtualenv --
if not exist ".venv\Scripts\python.exe" (
    echo  [..] Creating virtual environment
    %PY% -m venv .venv
    if errorlevel 1 (
        echo  [XX] Could not create the virtual environment.
        pause
        exit /b 1
    )
)
set VENV_PY=.venv\Scripts\python.exe
echo  [ok] Virtual environment ready

REM ------------------------------------------------------------- packages ---
echo  [..] Installing packages ^(a few minutes the first time^)
"%VENV_PY%" -m pip install --upgrade pip --quiet
"%VENV_PY%" -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo  [XX] Package installation failed. Scroll up for the reason.
    pause
    exit /b 1
)
echo  [ok] Packages installed

REM --------------------------------------------------------------- config ---
if not exist "config.toml" (
    copy /y "config.example.toml" "config.toml" >nul
    echo  [ok] Created config.toml
) else (
    echo  [ok] config.toml already exists ^(left alone^)
)

REM --------------------------------------------------------------- Ollama ---
where ollama >nul 2>&1
if errorlevel 1 (
    echo.
    echo  [XX] Ollama is not installed. It runs the local model that turns
    echo       transcripts into work orders.
    echo.
    echo       Download it from https://ollama.com/download , then run this
    echo       script again. Everything else above is already done.
    echo.
    pause
    exit /b 1
)
echo  [ok] Ollama is installed

for /f "tokens=2 delims== " %%m in ('findstr /b "model" config.toml') do set OLLAMA_MODEL=%%~m
if not defined OLLAMA_MODEL set OLLAMA_MODEL=gemma3:4b

echo  [..] Downloading the language model ^(%OLLAMA_MODEL%, a few GB^)
ollama pull %OLLAMA_MODEL%
if errorlevel 1 (
    echo  [XX] Could not download %OLLAMA_MODEL%.
    echo       Check that Ollama is running, then run:  ollama pull %OLLAMA_MODEL%
    pause
    exit /b 1
)
echo  [ok] Language model ready

REM -------------------------------------------------------- speech model ----
echo  [..] Downloading the speech model
set PYTHONPATH=%~dp0src
"%VENV_PY%" -c "from call_transcriber import config, transcribe; transcribe.load_model(config.load().transcribe)"
if errorlevel 1 (
    echo  [!!] The speech model did not download. It will retry on first run.
) else (
    echo  [ok] Speech model ready
)

REM ------------------------------------------------------------- autostart --
echo.
set /p AUTOSTART=  Start automatically when Windows starts? [Y/n]:
if /i "!AUTOSTART!"=="n" goto :skip_autostart

powershell -NoProfile -Command ^
  "$s=(New-Object -ComObject WScript.Shell).CreateShortcut(\"$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup\Call Transcriber.lnk\"); $s.TargetPath='%~dp0start-hidden.vbs'; $s.WorkingDirectory='%~dp0'; $s.Description='Local phone call transcriber'; $s.Save()"
if errorlevel 1 (
    echo  [!!] Could not create the startup shortcut. Start it manually with run.bat.
) else (
    echo  [ok] Will start with Windows
)
:skip_autostart

REM ----------------------------------------------------------------- done ---
echo.
echo  ============================================================
echo   Setup finished. Checking everything...
echo  ============================================================
echo.
"%VENV_PY%" -m call_transcriber doctor

echo.
echo   Next:
echo     run.bat devices   - find your adapter's name for config.toml
echo     run.bat           - start listening
echo.
pause
