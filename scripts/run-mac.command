#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"/..

if [ ! -d ".venv" ]; then
    echo "❌ 가상 환경이 없습니다. 먼저 install-mac.command를 실행하세요."
    read -p "Press Enter to close..."
    exit 1
fi

source .venv/bin/activate

python -c "from gui.app import gui_main; gui_main()"

if [ $? -ne 0 ]; then
    read -p "Press Enter to close..."
fi
