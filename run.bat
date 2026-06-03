@echo off
cd /d "%~dp0"
if not exist venv\Scripts\activate.bat (python -m venv venv)
call venv\Scripts\activate.bat
rem Clear PYTHONUTF8 first to prevent issues with empty or invalid inherited values,
rem then set it cleanly to the expected value of "1".
set PYTHONUTF8=
set PYTHONUTF8=1
python main.py %*
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] OpenAssist exited with code %ERRORLEVEL%
    pause
)
