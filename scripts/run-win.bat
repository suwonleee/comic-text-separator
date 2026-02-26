@echo off
chcp 65001 >nul
cd /d "%~dp0\.."

if not exist ".venv" (
    echo 가상 환경이 없습니다. 먼저 install-win.bat를 실행하세요.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

python -c "from gui.app import gui_main; gui_main()"

if errorlevel 1 (
    pause
)
