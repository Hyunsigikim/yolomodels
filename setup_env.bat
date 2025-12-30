@echo off
echo Creating virtual environment...
python -m venv .yolov8
if errorlevel 1 (
    echo Failed to create virtual environment.
    pause
    exit /b 1
)

echo Activating virtual environment...
call .\.yolov8\Scripts\activate
if errorlevel 1 (
    echo Failed to activate virtual environment.
    pause
    exit /b 1
)

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing PyTorch (CUDA 11.8)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
    echo Warning: PyTorch installation failed. You may need to install manually for your CUDA version.
)

echo Installing requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo Warning: Some requirements failed to install.
)

echo.
echo Verifying installation...
python -c "import numpy; import cv2; print('numpy', numpy.__version__); print('cv2', cv2.__version__)"
python -c "from ultralytics import YOLO; print('ultralytics import OK')"

echo.
echo Setup complete. To activate the environment later, run:
echo .\.yolov8\Scripts\activate
pause
