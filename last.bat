@echo off
REM Double-click this to bring back the most recent work order and put it on
REM the clipboard again -- for when the window got closed by accident.
set CT_NO_PAUSE=1
call "%~dp0run.bat" last
echo.
pause
