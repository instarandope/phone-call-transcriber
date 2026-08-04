@echo off
REM Double-click this to check the whole setup: packages, adapter, models,
REM output folder. Same as typing `run.bat doctor` in a terminal.
set CT_NO_PAUSE=1
call "%~dp0run.bat" doctor
echo.
pause
