@echo off
chcp 949 >nul
echo Deep Learning Course Docker Environment Starting...

REM Check if Docker Desktop is running
docker info >nul 2>&1
if errorlevel 1 (
    echo Docker Desktop is not running.
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)

REM Check GPU availability
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo NVIDIA GPU not detected. Running without GPU support...
) else (
    echo NVIDIA GPU detected. Running with GPU support...
)

REM Start Docker container
echo Starting Docker container...
cd examples
docker-compose up --build   

echo.
echo Jupyter Notebook has started.
echo Please access http://localhost:8888 in your browser.
echo.
echo Press any key to exit...
pause >nul

REM Stop container
docker-compose down
cd .. 