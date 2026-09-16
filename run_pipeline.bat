@echo off
chcp 65001 >nul
echo ==================================================
echo [YOLO Pipeline] Checking virtual environment...
echo ==================================================

if not exist ".\.yolov8\Scripts\activate.bat" (
    echo [Info] Virtual environment '.yolov8' not found. Creating and setting up...
    
    python -m venv .yolov8
    if errorlevel 1 (
        echo [Error] Failed to create virtual environment.
        pause
        exit /b 1
    )
    
    call .\.yolov8\Scripts\activate
    if errorlevel 1 (
        echo [Error] Failed to activate virtual environment.
        pause
        exit /b 1
    )
    
    echo [Info] Upgrading pip...
    python -m pip install --upgrade pip
    
    echo [Info] Installing PyTorch (CUDA 11.8)...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    if errorlevel 1 (
        echo [Warning] PyTorch installation failed.
    )
    
    echo [Info] Installing requirements...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [Warning] Requirements installation failed.
    )
) else (
    echo [Info] Virtual environment '.yolov8' found. Activating...
    call .\.yolov8\Scripts\activate
    if errorlevel 1 (
        echo [Error] Failed to activate virtual environment.
        pause
        exit /b 1
    )
)

echo.
echo ==================================================
echo [YOLO Pipeline] Running All-in-One Pipeline...
echo ==================================================
python main.py %*
if errorlevel 1 (
    echo.
    echo [Error] Pipeline execution encountered an error.
    pause
    exit /b 1
)

pause
