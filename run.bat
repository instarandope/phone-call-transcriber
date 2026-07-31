@echo off
REM Start the transcriber. Any arguments are passed through, so:
REM   run.bat            listen for calls
REM   run.bat devices    list audio inputs
REM   run.bat doctor     check the setup
REM   run.bat test x.wav process an existing recording

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Not installed yet -- run install.bat first.
    pause
    exit /b 1
)

set PYTHONPATH=%~dp0src
".venv\Scripts\python.exe" -m call_transcriber %*

REM Only hold the window open when something went wrong and there is an error
REM to read; a clean Ctrl-C exit shouldn't need a keypress.
if errorlevel 1 pause
