@echo off
cd /d "%~dp0"
if not exist venv\Scripts\activate.bat (python -m venv venv)
call venv\Scripts\activate.bat
set PYTHONUTF8=1
python main.py %*