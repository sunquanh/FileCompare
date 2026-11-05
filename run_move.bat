@echo off
cd /d "%~dp0"
echo This will MOVE originals. Press Ctrl+C to abort.
pause
set PYTHONIOENCODING=utf-8
python "%~dp0dedup_finder.py" --mode move %*
if errorlevel 1 pause
