@echo off
cd /d "%~dp0"
echo ============================================================
echo  Dedup + organize by file type (files will be MOVED)
echo ============================================================
echo.
set PYTHONIOENCODING=utf-8
python "%~dp0dedup_finder.py" --organize --mode move %*
if errorlevel 1 pause
