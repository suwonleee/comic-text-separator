#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"/..

echo "Python 3 설치 여부를 확인합니다..."

if ! command -v python3 &>/dev/null; then
    echo ""
    echo "❌ Python 3를 찾을 수 없습니다."
    echo "아래 주소에서 Python 3.10 이상을 설치해 주세요:"
    echo "  https://www.python.org/downloads/macos/"
    read -p "Press Enter to close..."
    exit 1
fi

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)"; then
    echo ""
    echo "❌ Python 버전이 너무 낮습니다. Python 3.10 이상이 필요합니다."
    echo "아래 주소에서 최신 Python을 설치해 주세요:"
    echo "  https://www.python.org/downloads/macos/"
    read -p "Press Enter to close..."
    exit 1
fi

echo "Python 버전 확인 완료."

if [ ! -d ".venv" ]; then
    echo "가상 환경을 생성합니다..."
    python3 -m venv .venv
fi

echo "가상 환경을 활성화합니다..."
source .venv/bin/activate

echo "pip를 업그레이드합니다..."
pip install --upgrade pip --quiet

echo "프로젝트를 설치합니다..."
pip install -e . --quiet

echo ""
echo "✅ 설치 완료!"
echo "scripts/run-mac.command 파일을 더블클릭하여 실행하세요."
read -p "Press Enter to close..."
