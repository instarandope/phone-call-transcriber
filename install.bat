@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo  ============================================================
echo   Call Transcriber - setup
echo  ============================================================
echo.

REM ---------------------------------------------------------------- Python --
REM Two different failures used to produce the same message here: "no Python
REM at all" and "Python, but too old". They need different fixes, so they are
REM detected and reported separately.
REM
REM `where python` is not a usable test. Windows ships a stub python.exe that
REM only opens the Microsoft Store -- it answers `where` and then fails when
REM run. So each candidate is tested by actually executing it.
set PY=
set PYVER=

py -3 -c "import sys" >nul 2>&1 && set PY=py -3
if not defined PY (
    python -c "import sys" >nul 2>&1 && set PY=python
)

if not defined PY (
    echo  [XX] Python is not installed on this PC.
    echo.
    echo       1. Go to https://www.python.org/downloads/
    echo       2. Download the latest Windows installer
    echo       3. On the FIRST screen, tick "Add python.exe to PATH" ^(easy to miss^)
    echo       4. Install, then run install.bat again
    echo.
    echo       If you think it IS installed: close this window and open a new
    echo       one first. A window opened before the install still has the old
    echo       PATH and cannot see it.
    echo.
    pause
    exit /b 1
)

for /f "delims=" %%v in ('%PY% --version 2^>^&1') do set PYVER=%%v

REM Written with max() rather than a comparison so no angle bracket ever
REM reaches cmd, which would treat it as a redirect.
%PY% -c "import sys; sys.exit(0 if max((3,11), sys.version_info[:2]) == sys.version_info[:2] else 1)" >nul 2>&1
if errorlevel 1 (
    echo  [XX] Found %PYVER%, but 3.11 or newer is needed.
    echo.
    echo       Install a current version from https://www.python.org/downloads/
    echo       and tick "Add python.exe to PATH" on the first screen. The new
    echo       version installs alongside the old one; nothing is removed.
    echo.
    pause
    exit /b 1
)
echo  [ok] %PYVER%

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
    echo.
    echo       If you see any of these:
    echo         - "Microsoft Visual C++ 14.0 or greater is required"
    echo         - "Building wheel for ... error"
    echo         - "no matching distribution found"
    echo.
    echo       ...then your Python is newer than some of these packages support
    echo       yet, and pip is trying to compile them from scratch. Several of
    echo       them contain C++ code and only ship ready-built for versions
    echo       they have caught up with.
    echo.
    echo       Fix: install Python 3.12 from https://www.python.org/downloads/
    echo       - scroll past the yellow button to the version list. Then delete
    echo       the .venv folder next to this script and run install.bat again.
    echo       If you have a newer Python installed too, uninstall it first so
    echo       the launcher picks 3.12.
    echo.
    pause
    exit /b 1
)
echo  [ok] Packages installed

REM Everything installed, so the wheels exist -- but say so if this is a
REM Python newer than the project has actually been exercised on.
"%VENV_PY%" -c "import sys; sys.exit(0 if max(sys.version_info[:2], (3,13)) == (3,13) else 1)" >nul 2>&1
if errorlevel 1 (
    echo  [!!] %PYVER% is newer than this project has been tried against.
    echo       It installed cleanly, so it should be fine - but if anything
    echo       behaves oddly later, Python 3.12 is the version to fall back to.
)

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
echo   Next - all of these are double-clickable, no typing needed:
echo     devices.bat   - find your adapter's name for config.toml
echo     doctor.bat    - check everything is ready
echo     levels.bat    - meter your phone line, get the threshold values
echo     run.bat       - start listening for calls
echo.
pause
