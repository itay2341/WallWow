#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== 1. System dependencies (headless OpenGL) ==="
if ! dpkg -s libosmesa6-dev &>/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq libosmesa6-dev freeglut3-dev
fi

echo "=== 2. Clone TripoSR (if missing) ==="
if [ ! -d "TripoSR" ]; then
    git clone https://github.com/VAST-AI-Research/TripoSR.git
fi

echo "=== 3. Create virtual environment (if missing) ==="
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

echo "=== 4. Install PyTorch (CUDA 12.1) ==="
pip install -q --upgrade pip wheel setuptools
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "=== 5. Install Python dependencies ==="
pip install -q -r requirements.txt

echo "=== 6. Install TripoSR extra dependencies ==="
pip install -q xatlas moderngl "imageio[ffmpeg]"

echo "=== 7. Create output directory ==="
mkdir -p output

echo "=== Setup complete ==="
echo "Activate the venv with: source .venv/bin/activate"
echo "Run with: python run.py --prompt 'a geometric cube'"
