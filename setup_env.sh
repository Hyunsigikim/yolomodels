#!/bin/bash

echo "Creating virtual environment..."
python3 -m venv .yolov8
if [ $? -ne 0 ]; then
    echo "Failed to create virtual environment."
    exit 1
fi

echo "Activating virtual environment..."
source .yolov8/bin/activate
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment."
    exit 1
fi

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing PyTorch (CUDA 11.8)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if [ $? -ne 0 ]; then
    echo "Warning: PyTorch installation failed. You may need to install manually for your CUDA version."
fi

echo "Installing requirements..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Warning: Some requirements failed to install."
fi

echo ""
echo "Verifying installation..."
python -c "import numpy; import cv2; print('numpy', numpy.__version__); print('cv2', cv2.__version__)"
python -c "from ultralytics import YOLO; print('ultralytics import OK')"

echo ""
echo "Setup complete. To activate the environment later, run:"
echo "source .yolov8/bin/activate"
