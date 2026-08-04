@echo off
REM Double-click this to list audio inputs, so you can find the adapter's
REM name for config.toml. Same as typing `run.bat devices` in a terminal.
set CT_NO_PAUSE=1
call "%~dp0run.bat" devices
echo.
pause
