@echo off
REM Double-click this to measure your phone line and get the two threshold
REM values for config.toml. Same as typing `run.bat levels` in a terminal.
set CT_NO_PAUSE=1
call "%~dp0run.bat" levels
echo.
pause
