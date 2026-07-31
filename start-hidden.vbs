' Launches the transcriber with no console window.
' Used by the Startup shortcut so the app is invisible until a call finishes
' and the work order pops up.
Dim shell, fso, here
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
here = fso.GetParentFolderName(WScript.ScriptFullName)

shell.CurrentDirectory = here
' 0 = hidden window, False = don't wait for it to exit
shell.Run """" & here & "\run.bat"" --tray", 0, False
