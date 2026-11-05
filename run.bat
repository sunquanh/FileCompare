@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
python "%~dp0dedup_finder.py" %*
if errorlevel 1 pause
