@echo off
REM Double-click this to bring the program up to date. Replaces the five-step
REM dance of browser, download, extract, "replace files in the destination",
REM install.bat -- with one click that does the same thing and reports what
REM actually changed.
REM
REM Your config.toml, your recordings and your downloaded models are not in
REM the published archive at all, so an update cannot touch them.

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Not installed yet -- run install.bat first.
    pause
    exit /b 1
)

set PYTHONPATH=%~dp0src
".venv\Scripts\python.exe" -u -m call_transcriber update
if errorlevel 1 (
    echo.
    echo   The update did not complete. Nothing was half-applied: files are
    echo   replaced one at a time only after the whole archive has downloaded.
    echo.
    pause
    exit /b 1
)

REM Cheap when nothing changed, and the one thing people forget when they
REM update by hand. A new dependency otherwise shows up much later as a
REM feature that refuses to start.
echo.
echo  [..] Checking packages
".venv\Scripts\python.exe" -m pip install -r requirements.txt --quiet --disable-pip-version-check
if errorlevel 1 (
    echo  [!!] Package install failed. Run install.bat to see why.
) else (
    echo  [ok] Packages up to date
)

echo.
echo  ============================================================
echo   Checking everything still works
echo  ============================================================
echo.
".venv\Scripts\python.exe" -u -m call_transcriber doctor

echo.
pause
