' Starts the transcriber with no console window and no taskbar button -- just
' the tray icon. This is what the Startup shortcut runs, and you can
' double-click it yourself any time.
'
' It launches pythonw.exe rather than python.exe. pythonw has no console at
' all, so there is no window to hide and nothing for the taskbar to show. A
' hidden console would also deadlock if anything ever paused for a keypress
' that could not arrive.

Dim shell, fso, here, pythonw, env

Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
here = fso.GetParentFolderName(WScript.ScriptFullName)

pythonw = here & "\.venv\Scripts\pythonw.exe"

If Not fso.FileExists(pythonw) Then
    MsgBox "Call Transcriber is not installed yet." & vbCrLf & vbCrLf & _
           "Run install.bat first, then try this again.", _
           vbExclamation, "Call Transcriber"
    WScript.Quit 1
End If

Set env = shell.Environment("PROCESS")
env("PYTHONPATH") = here & "\src"

shell.CurrentDirectory = here
' 0 = no window, False = don't wait for it to finish
shell.Run """" & pythonw & """ -m call_transcriber run --tray", 0, False
