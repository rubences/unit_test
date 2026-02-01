#!/bin/bash
# Installation guide for Moto-Edge-RL Pipeline

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   Moto-Edge-RL Training Pipeline - Installation Guide      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check Python version
echo "[1/6] Checking Python installation..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "  ✓ Python ${PYTHON_VERSION} installed"

if ! command -v python3 &> /dev/null; then
    echo "  ✗ Python3 not found. Install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment
echo ""
echo "[2/6] Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  ✓ Virtual environment created"
else
    echo "  ✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "[3/6] Activating virtual environment..."
source venv/bin/activate
echo "  ✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "[4/6] Upgrading pip..."
pip install --quiet --upgrade pip setuptools wheel
echo "  ✓ pip upgraded"

# Install dependencies
echo ""
echo "[5/6] Installing dependencies (this may take 5-10 minutes)..."
pip install --quiet -r requirements.txt
echo "  ✓ Dependencies installed"

# Verify installation
echo ""
echo "[6/6] Verifying installation..."
python3 << 'EOF'
import sys

packages = {
    'numpy': 'NumPy',
    'gymnasium': 'Gymnasium',
    'stable_baselines3': 'Stable-Baselines3',
    'torch': 'PyTorch',
    'h5py': 'HDF5',
    'tensorflow': 'TensorFlow',
    'onnx': 'ONNX',
}

all_ok = True
for package, name in packages.items():
    try:
        __import__(package)
        print(f"  ✓ {name}")
    except ImportError:
        print(f"  ✗ {name} (not installed)")
        all_ok = False

if all_ok:
    print("\n✅ All dependencies installed successfully!")
else:
    print("\n⚠️  Some dependencies are missing.")
    sys.exit(1)
EOF

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║            Installation Complete!                          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Validate setup: python3 validate_pipeline.py"
echo "  3. Run pipeline: ./run_pipeline.sh"
echo ""
