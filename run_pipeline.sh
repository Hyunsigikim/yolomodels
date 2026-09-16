#!/bin/bash

echo "=================================================="
echo "[YOLO Pipeline] Checking virtual environment..."
echo "=================================================="

if [ ! -f ".yolov8/bin/activate" ]; then
    echo "[Info] Virtual environment '.yolov8' not found. Creating and setting up..."
    
    python3 -m venv .yolov8
    if [ $? -ne 0 ]; then
        echo "[Error] Failed to create virtual environment."
        exit 1
    fi
    
    source .yolov8/bin/activate
    if [ $? -ne 0 ]; then
        echo "[Error] Failed to activate virtual environment."
        exit 1
    fi
    
    echo "[Info] Upgrading pip..."
    python -m pip install --upgrade pip
    
    echo "[Info] Installing PyTorch (CUDA 11.8)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    if [ $? -ne 0 ]; then
        echo "[Warning] PyTorch installation failed."
    fi
    
    echo "[Info] Installing requirements..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "[Warning] Requirements installation failed."
    fi
else
    echo "[Info] Virtual environment '.yolov8' found. Activating..."
    source .yolov8/bin/activate
    if [ $? -ne 0 ]; then
        echo "[Error] Failed to activate virtual environment."
        exit 1
    fi
fi

echo ""
echo "=================================================="
echo "[YOLO Pipeline] Step 1: Splitting dataset..."
echo "=================================================="
python split_dataset.py "$@"
if [ $? -ne 0 ]; then
    echo "[Error] Dataset split failed."
    exit 1
fi

echo ""
echo "=================================================="
echo "[YOLO Pipeline] Step 2: Training YOLO model..."
echo "=================================================="
python train.py --epochs 50 --model-size yolov8s.pt
if [ $? -ne 0 ]; then
    echo "[Error] YOLO model training failed."
    exit 1
fi

echo ""
echo "=================================================="
echo "[YOLO Pipeline] Pipeline completed successfully!"
echo "=================================================="
