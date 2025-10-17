#!/bin/bash
# AI Lens Helper GUI 실행 스크립트

echo "🚀 AI Lens Helper GUI 시작..."
echo ""
echo "YOLO+CLIP 기반 전시품 인식 시스템"
echo "======================================"
echo ""
echo "GUI 창이 열리면:"
echo "  Tab 1: 인덱스 빌드 (학습)"
echo "  Tab 2: Inference (테스트)"
echo ""
echo "자세한 사용법: GUI_USAGE.md 참고"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3를 찾을 수 없습니다."
    echo "Python 3.8 이상을 설치해주세요."
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python 버전: $python_version"

# Fix OpenMP library conflict (macOS specific)
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

# Run GUI
python3 gui_app.py

echo ""
echo "GUI가 종료되었습니다."
