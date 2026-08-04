@echo off
REM Start the transcriber. Any arguments are passed through, so:
REM   run.bat            listen for calls
REM   run.bat devices    list audio inputs
REM   run.bat doctor     check the setup
REM   run.bat levels     measure your line, get threshold values
REM   run.bat test x.wav process an existing recording
REM   run.bat purge      securely delete any kept recordings

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Not installed yet -- run install.bat first.
    pause
    exit /b 1
)

REM Present is not the same as runnable. Antivirus quarantines Python
REM interpreters fairly often -- a copied .exe that captures audio,
REM downloads models and loads native libraries reads like malware to a
REM heuristic scanner -- and the symptom is a bare "Access is denied", or
REM commands that simply do nothing. Neither says what to do about it.
".venv\Scripts\python.exe" -c "pass" >nul 2>&1
if errorlevel 1 (
    echo.
    echo   Windows will not run .venv\Scripts\python.exe
    echo.
    echo   The file is there, but it cannot be launched. Almost always this
    echo   is antivirus having quarantined it, or a policy that blocks
    echo   programs from running on this drive.
    echo.
    echo     1. Check your antivirus protection history for python.exe
    echo     2. Restore it and add this folder as an exclusion,
    echo        or delete the .venv folder and run install.bat again
    echo     3. If this folder is on a USB or external drive, try it
    echo        from C:\ instead
    echo.
    pause
    exit /b 1
)

set PYTHONPATH=%~dp0src
REM -u keeps stdout unbuffered. A C library that crashes takes the
REM interpreter with it, and anything still sitting in the buffer dies
REM with it -- which turns "it failed at step three" into total silence.
".venv\Scripts\python.exe" -u -m call_transcriber %*
set EXITCODE=%ERRORLEVEL%

REM A negative or huge code is a native crash rather than a Python error,
REM so say so instead of leaving a blank console to interpret.
if %EXITCODE% GEQ 2 (
    echo.
    echo   The program stopped unexpectedly ^(exit code %EXITCODE%^).
    echo   See call-transcriber-crash.log next to this script.
)

REM Only hold the window open when something went wrong and there is an error
REM to read; a clean Ctrl-C exit shouldn't need a keypress. The helper scripts
REM set CT_NO_PAUSE because they pause themselves afterwards.
if errorlevel 1 if not defined CT_NO_PAUSE pause
