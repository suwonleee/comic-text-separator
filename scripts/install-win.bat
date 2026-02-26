@echo off
chcp 65001 >nul
cd /d "%~dp0\.."

echo Python 설치 여부를 확인합니다...

where python >nul 2>&1
if errorlevel 1 (
    echo.
    echo Python을 찾을 수 없습니다.
    echo 아래 주소에서 Python 3.10 이상을 설치해 주세요:
    echo   https://www.python.org/downloads/windows/
    echo 설치 시 "Add Python to PATH" 체크박스를 반드시 선택하세요.
    pause
    exit /b 1
)

python -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)"
if errorlevel 1 (
    echo.
    echo Python 버전이 너무 낮습니다. Python 3.10 이상이 필요합니다.
    echo 아래 주소에서 최신 Python을 설치해 주세요:
    echo   https://www.python.org/downloads/windows/
    echo 설치 시 "Add Python to PATH" 체크박스를 반드시 선택하세요.
    pause
    exit /b 1
)

echo Python 버전 확인 완료.

if not exist ".venv" (
    echo 가상 환경을 생성합니다...
    python -m venv .venv
)

echo 가상 환경을 활성화합니다...
call .venv\Scripts\activate.bat

nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo NVIDIA GPU 감지 — CUDA 지원으로 설치합니다...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --quiet
)

echo pip를 업그레이드합니다...
pip install --upgrade pip --quiet

echo 프로젝트를 설치합니다...
pip install -e . --quiet

echo.
echo 설치 완료!
echo scripts\run-win.bat 파일을 더블클릭하여 실행하세요.
pause
