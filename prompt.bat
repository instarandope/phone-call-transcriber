@echo off
REM Double-click this to see exactly what the model is told to look for in a
REM call. Edit fields.py to change it, then run this again to check it reads
REM the way you meant.
set CT_NO_PAUSE=1
call "%~dp0run.bat" prompt
echo.
pause
