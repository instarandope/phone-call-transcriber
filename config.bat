@echo off
REM Double-click this to see which config.toml is being used and which of your
REM settings are actually in effect. The answer to "I changed something and
REM nothing happened". Same as typing `run.bat config` in a terminal.
set CT_NO_PAUSE=1
call "%~dp0run.bat" config
echo.
pause
