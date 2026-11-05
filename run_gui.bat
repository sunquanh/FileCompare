@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

where pythonw >nul 2>&1
if errorlevel 1 goto fallback

start "" pythonw "%~dp0dedup_gui.py" %*
exit /b 0

:fallback
python "%~dp0dedup_gui.py" %*
if errorlevel 1 pause
